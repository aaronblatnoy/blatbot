"""Inkbox <-> OpenAI GPT-Live voice bridge for live phone calls.

GPT-Live is a full-duplex voice model: it listens while it speaks, runs the
spoken conversation by itself, and *delegates* anything that needs real work
to a backend. We use client delegation, so the backend is this gateway: when
the model delegates, the bridge builds the request from the call transcript,
hands it to the same consult callback the Realtime bridge uses (which in gate
mode is the router -> gate -> executor path), and returns the outcome with
``session.commentary.append`` for the model to say in its own words.

Differences from ``realtime.py`` that matter here:

* One WebSocket at ``/v1/live/sessions``; first message is ``session.start``.
* The model, instructions, audio format, voice and delegation mode are fixed
  at startup.
* There are no tools, no ``response.create`` and no turn detection to tune.
  The model decides when to talk, when to stop, and when to delegate.
* ``session.delegation.created`` carries an id and no task text. The request
  is reconstructed from ``session.input_transcript.delta`` /
  ``session.output_transcript.delta``.
* Output audio arrives as ``session.output_audio.delta`` with no "done" event.

The bridge exposes the same ``run()`` / ``close()`` surface as
``OpenedRealtimeBridge`` so the gateway can use either.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:  # pragma: no cover - import guard mirrors realtime.py
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

from .realtime import (
    AgentConsultCallback,
    CallEndedCallback,
    PostCallActionsCallback,
    RealtimeBridgeConnectError,
    RealtimeCallMeta,
    RealtimeConfig,
    _escape_contact_memory_tags,
    _maybe_close_ws,
)

logger = logging.getLogger("inkbox_claude.live")

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
DEFAULT_LIVE_MODEL = "gpt-live-1"
# G.711 mu-law at 8 kHz: what the phone leg carries, passed through unconverted.
AUDIO_FORMAT_TELEPHONY = {"type": "audio/pcmu", "rate": 8000}
# Append events are capped at 500 tokens; stay well inside that.
MAX_APPEND_CHARS = 1600
STARTED_TIMEOUT_S = 10.0
CLOSE_TIMEOUT_S = 5.0
# Input transcript fragments trail the audio slightly. Give the last words of
# the request a moment to arrive before building the backend request.
TRANSCRIPT_SETTLE_S = 0.9
CONTEXT_TURNS = 8


LIVE_PERSONA = (
    "You are Blatbot, Aaron Blatnoy's assistant, answering his assistant line. You're warm and easy to talk "
    "to, like someone who's glad the person called. Match the caller: mirror their energy, pace and mood. If "
    "they're upbeat and joking, loosen up and joke back. If they're brisk and busy, be quick and to the point. "
    "If they're tired, stressed or annoyed, be calm and on their side. Friendly by default, never flat, stiff "
    "or robotic, and never more hyped than they are. Be genuine about it: talk like a real person who likes "
    "them, not like someone playing a part. "
    "One firm rule on delivery: no uptalk. Aaron cannot stand it. Statements end with your pitch dropping, "
    "like a full stop; never make a statement sound like a question, and don't tack on check-ins like "
    "'okay?', 'right?' or 'sound good?'. Warmth comes from your tone and your words, not from rising pitch."
)


def live_model() -> str:
    return str(os.getenv("INKBOX_LIVE_MODEL") or DEFAULT_LIVE_MODEL).strip()


def live_voice(config: RealtimeConfig) -> str:
    """INKBOX_LIVE_VOICE if set (Live has voices Realtime lacks), else the shared voice."""
    return str(os.getenv("INKBOX_LIVE_VOICE") or config.voice).strip()


@dataclass
class _LiveState:
    # Merged turns: [speaker, text], speaker is "caller" or "blatbot".
    turns: List[List[str]] = field(default_factory=list)
    # Index into ``turns`` up to which the backend has already been shown.
    delegated_upto: int = 0
    closed: bool = False
    started: bool = False
    greeted: bool = False
    # Input is muted while the introduction plays so a "hello?" or pickup noise can't cut it off.
    intro_muted: bool = False
    # GPT-Live streams output continuously, silence included, so "audio stopped arriving" never
    # happens. The intro is over when speech has been heard and then gone quiet.
    intro_speech_seen: bool = False
    intro_last_speech_at: float = 0.0
    stream_id: Optional[str] = None
    last_transcript_at: float = field(default_factory=time.monotonic)
    tasks: Set["asyncio.Task[None]"] = field(default_factory=set)
    seen_event_types: Set[str] = field(default_factory=set)
    event_seq: int = 0

    def next_event_id(self, prefix: str) -> str:
        self.event_seq += 1
        return f"{prefix}_{self.event_seq}"

    def add_fragment(self, speaker: str, text: str) -> None:
        if not text:
            return
        self.last_transcript_at = time.monotonic()
        if self.turns and self.turns[-1][0] == speaker:
            self.turns[-1][1] += text
        else:
            self.turns.append([speaker, text])

    @property
    def transcript(self) -> List[Tuple[str, str]]:
        return [("user" if s == "caller" else "agent", t.strip()) for s, t in self.turns if t.strip()]


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------


def build_live_instructions(meta: RealtimeCallMeta, additional: str = "") -> str:
    """Conversation prompt for GPT-Live, in the structure OpenAI recommends.

    The model is a surface: it is told who it is, who it is talking to, and that
    anything beyond talk gets handed off. It is told nothing about what sits
    behind the hand-off.
    """
    lines: List[str] = []
    # Let the chosen voice sound like itself: no pitch/energy/intonation steering on Live.
    try:
        from .config import _BLATBOT_DELIVERY
        additional = additional.replace(_BLATBOT_DELIVERY.strip(), "").replace("  ", " ")
        from .config import _BLATBOT_VOICE
        # OpenAI's own template wording. A characterful persona ("quick, dry humor, a guy in his
        # twenties") makes the model act a part, and that acting is what reads as affected.
        additional = additional.replace(_BLATBOT_VOICE.strip(), LIVE_PERSONA)
    except Exception:  # pragma: no cover
        pass
    if additional.strip():
        lines += [additional.strip(), ""]
    if is_owner_call(meta):
        lines.append("You are talking to Aaron himself. Call him Aaron.")
    elif meta.contact_known and meta.contact_name:
        lines.append(f"You are talking to {_escape_contact_memory_tags(meta.contact_name)}.")
    lines += [
        "This is a live phone call and it is yours to run: chat, react to what they say, ask what you need "
        "to ask, and follow the caller when they change subject. Keep replies short.",
        "You are Blatbot, Aaron's assistant, and that is all there is to it. You don't know or talk about how "
        "you work. If someone asks whether you're an AI, say yes, you're Aaron's AI assistant, and move on.",
        "",
        "Backchannel policy: Use light backchannels, a brief 'mm' or 'right' while the caller is mid-thought. "
        "Never talk over them to do it.",
        "",
        "Interruption policy: Stop speaking when the caller interrupts, and listen. The caller is often "
        "walking outside in New York City, so expect traffic, sirens, wind and other people's voices. Do not "
        "treat a cough, a horn, music or a nearby conversation as the caller speaking or as a new request. "
        "Keep listening while the caller pauses to think. If you only caught part of what they said, ask them "
        "to repeat that part rather than guess. Keep answers short and put the important part first.",
        "",
        "Delegation policy:",
        "Handing something off is how you get anything done or found out. When you hand off, it is simply "
        "you checking or doing something: say a short natural line like 'let me look' or 'one sec', keep "
        "talking normally while it runs, and bring the outcome in when it arrives.",
        "Backend tools:",
        "- Anything beyond conversation: doing things, looking things up, checking, changing, sending, "
        "scheduling, following up, and anything about Aaron or the people he deals with.",
        "",
        "Delegate to the backend when:",
        "- The caller wants something done, found out, checked, changed or followed up, whatever it is. Don't "
        "judge whether it is possible; hand it off and you will be told.",
        "- You need a fact you were not given in this call.",
        "- A correction changes work already requested.",
        "",
        "Do not delegate to the backend when:",
        "- It is just conversation: greetings, small talk, thanks, banter.",
        "- You can answer from this conversation or a result you already got in it.",
        "- You need a brief clarification first. Ask it, then hand off once with everything.",
        "",
        "Delegate before giving an answer that depends on backend work.",
        "Do not guess the result while waiting.",
        "Never say something was booked, sent, changed or confirmed unless a result told you so. If a result "
        "says Aaron has to approve something first, tell the caller that.",
    ]
    return "\n".join(lines)


def _digits(value: Any) -> str:
    import re
    return re.sub(r"\D", "", str(value or ""))[-10:]


def is_owner_call(meta: RealtimeCallMeta) -> bool:
    """True when the call is from Aaron's own number (INKBOX_APPROVER_PHONE).

    Decided from the caller's phone number, not the contact name: his contact
    record may have no name, and a name is not an identity check anyway.
    """
    owner = _digits(os.getenv("INKBOX_APPROVER_PHONE"))
    return bool(owner) and _digits(meta.remote_phone_number) == owner


def build_live_greeting(meta: RealtimeCallMeta) -> str:
    first = ""
    if meta.contact_known and meta.contact_name:
        first = _escape_contact_memory_tags(meta.contact_name.split()[0])
    if meta.direction == "outbound" and meta.outbound_opening:
        return (
            "Speak first, in English. Open the call by saying this naturally, with no greeting before it, "
            f"then pause and listen: {_escape_contact_memory_tags(meta.outbound_opening)}"
        )
    if meta.direction == "outbound" and meta.outbound_purpose:
        return (
            f"Speak first, in English. Greet {first or 'them'} briefly, say you're Blatbot, Aaron Blatnoy's "
            f"assistant, and that you're calling because: {_escape_contact_memory_tags(meta.outbound_purpose)}. "
            "Then pause and listen."
        )
    if is_owner_call(meta):
        line = "Hey Aaron, it's Blatbot."
    else:
        line = "Hello, this is Blatbot, Aaron Blatnoy's assistant."
    # Wording follows OpenAI's greeting example ("Greet the caller now in English. ... Then pause
    # and listen."); a differently phrased version left the model silent on an empty line.
    return (
        f"Greet the caller now in English. Immediately say: \"{line}\" Then pause and listen."
    )

def build_delegation_query(state: _LiveState, meta: RealtimeCallMeta) -> str:
    """Reconstruct what the caller wants from the transcript.

    GPT-Live's delegation event has no task text, so the backend gets the part
    of the call it has not seen yet plus a little earlier context.
    """
    turns = [(s, t.strip()) for s, t in state.turns if t.strip()]
    new_from = min(state.delegated_upto, len(turns))
    earlier = turns[max(0, new_from - CONTEXT_TURNS):new_from]
    fresh = turns[new_from:] or turns[-2:]
    state.delegated_upto = len(turns)

    def fmt(rows: List[Tuple[str, str]]) -> str:
        return "\n".join(f"{s}: {t}" for s, t in rows) or "(nothing)"

    who = meta.contact_name or meta.remote_phone_number or "the caller"
    parts = [
        f"Live phone call with {who}. Blatbot's voice side has handed this off because the caller needs "
        "something done or looked up. The lines below are speech-recognition transcript, so expect misheard "
        "words, unfinished phrases and later corrections; trust the latest version of what they said.",
    ]
    if earlier:
        parts += ["", "Earlier in the call (already handled, for context only):", fmt(earlier)]
    parts += ["", "What was just said (act on this):", fmt(fresh), "",
              "Work out what the caller is asking for right now and handle it."]
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Bridge
# ----------------------------------------------------------------------


@dataclass
class OpenedLiveBridge:
    """A started GPT-Live session, ready to bridge to an Inkbox call."""

    session: Any
    openai_ws: Any
    state: _LiveState
    config: RealtimeConfig
    meta: RealtimeCallMeta
    _closed: bool = False

    async def run(
        self,
        *,
        inkbox_ws: Any,
        on_agent_consult: AgentConsultCallback,
        on_post_call_actions: PostCallActionsCallback,
        on_call_ended: CallEndedCallback,
    ) -> None:
        state, ows = self.state, self.openai_ws
        tasks: List["asyncio.Task[None]"] = []
        try:
            tasks = [
                asyncio.create_task(_inkbox_to_live_pump(inkbox_ws, ows, state, self.meta),
                                    name=f"live-inkbox-pump-{self.meta.call_id}"),
                asyncio.create_task(_live_to_inkbox_pump(ows, inkbox_ws, state, self.meta, on_agent_consult),
                                    name=f"live-openai-pump-{self.meta.call_id}"),
            ]
            done, _ = await asyncio.wait(set(tasks), return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if not t.cancelled() and t.exception():
                    logger.warning("[live] pump %s raised: %s", t.get_name(), t.exception())
        finally:
            state.closed = True
            for t in tasks:
                if not t.done():
                    t.cancel()
            await _maybe_close_ws(inkbox_ws)
            await self.close()
            with suppress(Exception):
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
            # Delegated work keeps running after hangup; the consult callback
            # delivers its result by text when the call is no longer active.
        with suppress(Exception):
            await on_call_ended(state.transcript)

    async def identify(self, remote_phone_number: str, contact_name: str = "", notes: str = "") -> None:
        """The caller was identified after pickup. Tell the model who it is talking to."""
        self.meta.remote_phone_number = remote_phone_number or self.meta.remote_phone_number
        if contact_name:
            self.meta.contact_name, self.meta.contact_known = contact_name, True
        if is_owner_call(self.meta):
            who = ("You now know who is on the line: it is Aaron himself, your boss. Call him Aaron and treat him "
                   "as the person you work for. Do not announce that you just worked this out and do not "
                   "re-introduce yourself; just carry on naturally.")
        elif contact_name:
            who = f"You now know who is on the line: {_escape_contact_memory_tags(contact_name)}. Carry on naturally."
        else:
            return
        with suppress(Exception):
            await self.openai_ws.send_str(json.dumps({
                "type": "session.instructions.append", "event_id": self.state.next_event_id("identity"),
                "delegation_id": None, "content": who,
            }))
            if notes.strip():
                await self.openai_ws.send_str(json.dumps({
                    "type": "session.thinking.append", "event_id": self.state.next_event_id("notes"),
                    "delegation_id": None, "content": notes.strip()[:MAX_APPEND_CHARS],
                }))
        logger.info("[live] caller identified during the call (owner=%s)", is_owner_call(self.meta))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Graceful close so the session finalizes and usage is recorded.
        with suppress(Exception):
            await self.openai_ws.send_str(json.dumps({"type": "session.close"}))
            deadline = time.monotonic() + CLOSE_TIMEOUT_S
            while time.monotonic() < deadline:
                msg = await asyncio.wait_for(self.openai_ws.receive(), timeout=max(0.1, deadline - time.monotonic()))
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                if json.loads(msg.data).get("type") == "session.closed":
                    break
        with suppress(Exception):
            await self.openai_ws.close()
        with suppress(Exception):
            await self.session.close()


async def open_inkbox_live_bridge(*, config: RealtimeConfig, meta: RealtimeCallMeta) -> OpenedLiveBridge:
    """Connect to GPT-Live and start the session before the call commits to raw media."""
    if aiohttp is None:
        raise RealtimeBridgeConnectError("aiohttp not available")
    if not config.has_credential:
        raise RealtimeBridgeConnectError("no OpenAI API key configured")
    session = aiohttp.ClientSession()
    ows = None
    try:
        ows = await asyncio.wait_for(
            session.ws_connect(os.getenv("INKBOX_LIVE_URL") or LIVE_URL,
                               headers={"Authorization": f"Bearer {config.api_key}"}, heartbeat=30),
            timeout=config.connect_timeout_s,
        )
        await ows.send_str(json.dumps({
            "type": "session.start",
            "event_id": "start_1",
            "session": {
                "model": live_model(),
                "instructions": build_live_instructions(meta, config.additional_instructions),
                "audio": {"format": AUDIO_FORMAT_TELEPHONY, "output": {"voice": live_voice(config)}},
                "delegation": {"type": "client"},
            },
        }))
        deadline = time.monotonic() + STARTED_TIMEOUT_S
        while True:
            msg = await asyncio.wait_for(ows.receive(), timeout=max(0.1, deadline - time.monotonic()))
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise RealtimeBridgeConnectError(f"Live socket closed before start ({msg.type})")
            frame = json.loads(msg.data)
            if frame.get("type") == "session.started":
                sess = frame.get("session") or {}
                sid = sess.get("id")
                applied = ((sess.get("audio") or {}).get("output") or {}).get("voice")
                logger.info("[live] session started %s voice requested=%s applied_by_openai=%s",
                            sid, live_voice(config), applied)
                break
            if frame.get("type") == "error":
                raise RealtimeBridgeConnectError(f"Live rejected session.start: {frame.get('error')}")
        state = _LiveState(started=True)
        return OpenedLiveBridge(session=session, openai_ws=ows, state=state, config=config, meta=meta)
    except Exception as exc:
        if ows is not None:
            with suppress(Exception):
                await ows.close()
        with suppress(Exception):
            await session.close()
        if isinstance(exc, RealtimeBridgeConnectError):
            raise
        raise RealtimeBridgeConnectError(exc) from exc


async def _send_greeting(ows: Any, state: _LiveState, meta: RealtimeCallMeta) -> None:
    if state.greeted:
        return
    state.greeted = True
    with suppress(Exception):
        # Not session.input_audio.mute: the session timeline is driven by input audio, and muting
        # stalled it so the introduction sometimes never came. The pump sends silence in place of
        # the caller's audio instead, which keeps the timeline moving.
        state.intro_muted = True
        task = asyncio.create_task(_unmute_after_intro(ows, state), name="live-intro-unmute")
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)
    with suppress(Exception):
        await ows.send_str(json.dumps({
            "type": "session.instructions.append",
            "event_id": state.next_event_id("greeting"),
            "delegation_id": None,
            "content": build_live_greeting(meta),
        }))


_MULAW_SILENCE = frozenset((0xFF, 0x7F, 0xFE, 0x7E))


def _silence_like(payload_b64: str) -> str:
    """Mu-law silence of the same length as the given chunk."""
    import base64
    try:
        n = len(base64.b64decode(payload_b64))
    except Exception:
        n = 160
    return base64.b64encode(b"\xff" * n).decode("ascii")


def _has_speech(delta_b64: str) -> bool:
    """True if a G.711 mu-law chunk carries more than line silence."""
    import base64
    try:
        raw = base64.b64decode(delta_b64)
    except Exception:
        return False
    if not raw:
        return False
    loud = sum(1 for b in raw if b not in _MULAW_SILENCE)
    return loud > len(raw) * 0.2


async def _unmute_after_intro(ows: Any, state: _LiveState) -> None:
    """Unmute once the introduction has been spoken and the line has gone quiet (hard cap 6 s)."""
    started = time.monotonic()
    while not state.closed and time.monotonic() - started < 6.0:
        await asyncio.sleep(0.05)
        if state.intro_speech_seen and time.monotonic() - state.intro_last_speech_at >= 0.45:
            break
    state.intro_muted = False
    logger.info("[live] intro done after %.1fs (spoke=%s); caller unmuted",
                time.monotonic() - started, state.intro_speech_seen)


async def _inkbox_to_live_pump(inkbox_ws: Any, ows: Any, state: _LiveState, meta: RealtimeCallMeta) -> None:
    """Caller audio -> GPT-Live, unconverted. Input must keep flowing, silence included."""
    async for msg in inkbox_ws:
        if state.closed:
            return
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                frame = json.loads(msg.data)
            except (TypeError, ValueError):
                continue
            event = (frame.get("event") or "").lower()
            if event == "start":
                state.stream_id = frame.get("stream_id") or state.stream_id
                await _send_greeting(ows, state, meta)
            elif event == "media":
                if not state.greeted:
                    await _send_greeting(ows, state, meta)
                payload = (frame.get("media") or {}).get("payload")
                if payload:
                    if state.intro_muted:
                        payload = _silence_like(payload)
                    await ows.send_str(json.dumps({"type": "session.input_audio.append", "audio": payload}))
            elif event in {"stop", "closed", "hangup"}:
                logger.info("[live] Inkbox WS signaled %s", event)
                return
        elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
            return


async def _live_to_inkbox_pump(ows: Any, inkbox_ws: Any, state: _LiveState, meta: RealtimeCallMeta,
                               on_agent_consult: AgentConsultCallback) -> None:
    """GPT-Live audio -> caller; transcripts collected; delegations dispatched."""
    async for msg in ows:
        if state.closed:
            return
        if msg.type != aiohttp.WSMsgType.TEXT:
            if msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                return
            continue
        try:
            frame = json.loads(msg.data)
        except (TypeError, ValueError):
            continue
        if not isinstance(frame, dict):
            continue
        ftype = frame.get("type", "")

        if ftype == "session.output_audio.delta":
            delta = frame.get("delta") or ""
            if delta and state.intro_muted and _has_speech(delta):
                state.intro_speech_seen = True
                state.intro_last_speech_at = time.monotonic()
            if delta:
                out: Dict[str, Any] = {"event": "media", "media": {"payload": delta, "track": "outbound"}}
                if state.stream_id:
                    out["stream_id"] = state.stream_id
                try:
                    await inkbox_ws.send_str(json.dumps(out))
                except Exception as exc:
                    logger.debug("[live] Inkbox WS send failed: %s", exc)
                    return
        elif ftype == "session.input_transcript.delta":
            state.add_fragment("caller", str(frame.get("delta") or ""))
        elif ftype == "session.output_transcript.delta":
            state.add_fragment("blatbot", str(frame.get("delta") or ""))
        elif ftype == "session.delegation.created":
            deleg = frame.get("delegation") or {}
            did = str(deleg.get("id") or "")
            if did and (deleg.get("target") or "client") == "client":
                logger.info("[live] delegation %s", did)
                task = asyncio.create_task(_run_delegation(ows, state, meta, did, on_agent_consult),
                                           name=f"live-delegation-{did}")
                state.tasks.add(task)
                task.add_done_callback(state.tasks.discard)
        elif ftype == "session.closed":
            logger.info("[live] session closed: %s usage=%s", frame.get("reason"), frame.get("usage"))
            return
        elif ftype == "error":
            logger.warning("[live] error frame: %s", frame.get("error"))
        elif ftype not in state.seen_event_types:
            # New API: note each unfamiliar event type once so gaps are visible in the log.
            state.seen_event_types.add(ftype)
            logger.info("[live] event type seen: %s", ftype)


async def _run_delegation(ows: Any, state: _LiveState, meta: RealtimeCallMeta, delegation_id: str,
                          on_agent_consult: AgentConsultCallback) -> None:
    """Build the request from the transcript, run it through the gateway, return the outcome."""
    # Let the tail of the caller's sentence finish transcribing.
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline and time.monotonic() - state.last_transcript_at < TRANSCRIPT_SETTLE_S:
        await asyncio.sleep(0.1)
    query = build_delegation_query(state, meta)
    try:
        answer = await on_agent_consult(query, state.transcript)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("[live] delegation %s failed: %s", delegation_id, exc)
        answer = "That didn't go through because of a problem on my end. It can be tried again."
    answer = " ".join(str(answer or "").split()) or "Nothing came back on that."
    if len(answer) > MAX_APPEND_CHARS:
        answer = answer[:MAX_APPEND_CHARS].rsplit(" ", 1)[0] + " ..."
    if state.closed:
        return  # they hung up; the consult callback already sent the result by text
    with suppress(Exception):
        await ows.send_str(json.dumps({
            "type": "session.commentary.append",
            "event_id": state.next_event_id("result"),
            "delegation_id": delegation_id,
            "content": answer,
        }))
