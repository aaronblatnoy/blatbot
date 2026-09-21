"""Inkbox ↔ OpenAI Realtime API voice bridge for live phone calls.

When Realtime is configured, the gateway pre-opens an OpenAI Realtime
WebSocket *before* accepting the Inkbox call in raw-media mode, then runs
two pumps for the call's duration:

* caller audio (Inkbox ``media`` frames, base64 μ-law) → OpenAI
  ``input_audio_buffer.append``; server-side VAD handles turn-taking.
* OpenAI ``response.output_audio.delta`` → Inkbox ``media`` frames, so the
  model's own voice is what the caller hears.

The Realtime model runs the spoken conversation itself. It only reaches
back to Claude Code through the ``consult_agent`` tool — and only when the
caller asks for real work or account/contact context. The consult runs in the caller's
shared :class:`~inkbox_claude.sessions.ContactSession` and its text answer
is handed back to the model, which speaks it. If OpenAI can't be reached
the gateway falls back to Inkbox STT/TTS (see ``_handle_call_ws``).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

try:
    from .prompts import _escape_contact_memory_tags, contact_memories_block
except ImportError:  # pragma: no cover - direct local import/test fallback
    from prompts import _escape_contact_memory_tags, contact_memories_block

try:
    import aiohttp
except ImportError:  # pragma: no cover - aiohttp is a runtime dep
    aiohttp = None  # type: ignore

logger = logging.getLogger("inkbox_claude.realtime")


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2"
DEFAULT_VOICE = "cedar"
# μ-law telephony audio, matching the codec Inkbox bridges from the carrier.
AUDIO_FORMAT_TELEPHONY = {"type": "audio/pcmu"}
INPUT_TRANSCRIPTION_MODEL = "whisper-1"

CONSULT_TOOL_NAME = "consult_agent"
POST_CALL_ACTION_TOOL_NAME = "register_post_call_action"
EDIT_POST_CALL_ACTION_TOOL_NAME = "edit_post_call_action"
DELETE_POST_CALL_ACTION_TOOL_NAME = "delete_post_call_action"
HANG_UP_CALL_TOOL_NAME = "hang_up_call"

DEFAULT_CONSULT_TIMEOUT_S = 300.0
DEFAULT_CONNECT_TIMEOUT_S = 8.0
# hang_up_call is two-step: a second call within this window actually hangs up.
HANGUP_CONFIRM_WINDOW_S = 60.0
# Brief grace so the model's spoken goodbye reaches the caller before we drop.
HANGUP_CLOSE_DELAY_S = 2.0
# Never let a cancelled consult/task hold the call WebSocket cleanup forever.
TASK_CANCEL_TIMEOUT_S = 2.0


# A consult takes (query, recent_transcript) and returns Claude's spoken-
# friendly answer. The gateway wires this to the caller's ContactSession.
AgentConsultCallback = Callable[[str, List[Tuple[str, str]]], Awaitable[str]]
# After the call ends with queued actions: (actions, transcript) → run them.
PostCallActionsCallback = Callable[[List[Dict[str, str]], List[Tuple[str, str]]], Awaitable[None]]
# After a call with no queued actions: (transcript) → reflect / follow up.
CallEndedCallback = Callable[[List[Tuple[str, str]]], Awaitable[None]]


# ----------------------------------------------------------------------
# Config / per-call types
# ----------------------------------------------------------------------


@dataclass
class RealtimeConfig:
    """Realtime voice configuration, populated from the env in config.py."""

    enabled: bool = False
    api_key: str = ""
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    additional_instructions: str = ""
    consult_timeout_s: float = DEFAULT_CONSULT_TIMEOUT_S
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S
    fallback_to_inkbox_stt_tts: bool = True
    base_url: str = REALTIME_URL
    # Gate mode: the voice model is only a mouth and ears. Every fact and every
    # action goes through consult, and after-call queues are not offered.
    gate_mode: bool = False
    vocabulary: str = ""

    @property
    def has_credential(self) -> bool:
        return bool(self.api_key)


@dataclass
class RealtimeCallMeta:
    """Per-call metadata threaded to the greeting and instructions."""

    call_id: str
    remote_phone_number: Optional[str]
    direction: str = "inbound"
    agent_identity_handle: Optional[str] = None
    agent_identity_email: Optional[str] = None
    agent_identity_phone: Optional[str] = None
    # Whether the identity also has the shared Inkbox iMessage line enabled —
    # lets the spoken prompt draw the dedicated-vs-shared-line distinction.
    agent_imessage_enabled: bool = False
    project_dir: Optional[str] = None
    contact_known: bool = False
    contact_id: Optional[str] = None
    contact_name: Optional[str] = None
    contact_emails: List[str] = field(default_factory=list)
    contact_phones: List[str] = field(default_factory=list)
    contact_company: Optional[str] = None
    contact_job_title: Optional[str] = None
    contact_notes: Optional[str] = None
    contact_memories: List[str] = field(default_factory=list)
    # Outbound calls only: why this agent placed the call, threaded from
    # ``inkbox_place_call`` so the live session opens with context, not cold.
    outbound_purpose: Optional[str] = None
    outbound_opening: Optional[str] = None
    outbound_context: Optional[str] = None
    outbound_reason: Optional[str] = None
    outbound_scheduled_by: Optional[str] = None
    outbound_conversation_summary: Optional[str] = None


@dataclass
class _BridgeState:
    transcript: List[Tuple[str, str]] = field(default_factory=list)
    # Work the model asked to run after the call: [{"action", "details"}].
    post_call_actions: List[Dict[str, str]] = field(default_factory=list)
    closed: bool = False
    greeting_triggered: bool = False
    # Inkbox-assigned stream id from the `start` event; echoed on outbound
    # media / audio_done frames.
    stream_id: Optional[str] = None
    # Monotonic time the model first armed hang_up_call. A second call within
    # HANGUP_CONFIRM_WINDOW_S performs the real hangup. None = not armed.
    hangup_armed_at: Optional[float] = None
    # In-flight consult dispatches. The consult runs a full Claude Code turn
    # (seconds), so it is dispatched as a background task to keep the
    # OpenAI→Inkbox audio pump flowing; tracked here so call teardown can
    # cancel them.
    consult_tasks: Set["asyncio.Task[None]"] = field(default_factory=set)
    # Monotonic time anyone last made sound (model audio out, or caller speech
    # start). Used to fill dead air during a consult without talking over people.
    last_sound_at: float = field(default_factory=time.monotonic)
    # True between OpenAI's response.created and response.done. A tool result
    # that lands mid-response must wait its turn instead of colliding with it.
    response_active: bool = False
    gate_mode: bool = False
    barge_in_enabled: bool = False
    # Street-noise tolerance: the bridge, not OpenAI, decides what counts as an
    # interruption. Sound has to keep going for BARGE_IN_MS before the agent is cut off.
    speech_started_at: Optional[float] = None
    barge_task: Optional["asyncio.Task[None]"] = None
    # Estimated monotonic time the phone finishes playing audio already sent.
    # OpenAI generates speech faster than it plays, so "response done" is not
    # "the agent has stopped talking".
    playback_until: float = 0.0


# ----------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------


def build_realtime_instructions(meta: RealtimeCallMeta, additional: str = "", gate_mode: bool = False) -> str:
    """Compose the system prompt sent to the Realtime model.

    Args:
        meta (RealtimeCallMeta): Per-call context (caller, project).
        additional (str): Operator-supplied extra instructions.

    Returns:
        str: The instruction string for the ``session.update``.
    """
    lines = [
        "You are on a live phone call. Speak the way people talk on the phone.",
    ] if gate_mode else [
        "You are the configured Claude Code Inkbox agent speaking on a live Inkbox phone call.",
        "Use natural, concise spoken replies. Keep most answers to one or two short sentences.",
        "You are a voice; do not read out code, file paths, diffs, or logs verbatim.",
        "Do not mention implementation details unless the caller asks.",
    ]
    if meta.agent_identity_handle:
        lines.append(f"Your Inkbox identity handle: {meta.agent_identity_handle}.")
    if meta.agent_identity_email:
        lines.append(f"Your Inkbox agent email address: {meta.agent_identity_email}.")
    if meta.agent_identity_phone:
        lines.append(
            f"Your dedicated phone line (your own number, for SMS and voice calls): "
            f"{meta.agent_identity_phone}.",
        )
    if meta.agent_imessage_enabled:
        lines.append(
            "You also have a shared Inkbox iMessage line — voice calls and iMessage "
            "with people connected to you over iMessage. Its number is managed by "
            "Inkbox: never state or promise a number for it. The current call may be "
            "running over either line; calls follow the conversation's channel "
            "(iMessage contacts are called over the shared line, SMS/phone contacts "
            "over your dedicated number).",
        )
    if meta.remote_phone_number:
        lines.append(f"Remote phone number: {meta.remote_phone_number}.")
    if meta.contact_known:
        lines.append(
            "Known Inkbox contact info is already loaded; do not look them up or ask for details you already have.",
        )
        if meta.contact_name:
            lines.append(f"Contact name: {_escape_contact_memory_tags(meta.contact_name)}.")
        if meta.contact_id:
            lines.append(f"Inkbox contact id: {meta.contact_id}.")
        if meta.contact_company:
            lines.append(f"Contact company: {_escape_contact_memory_tags(meta.contact_company)}.")
        if meta.contact_job_title:
            lines.append(f"Contact title: {_escape_contact_memory_tags(meta.contact_job_title)}.")
        if meta.contact_emails:
            lines.append(f"Contact email(s): {', '.join(meta.contact_emails)}.")
        if meta.contact_phones:
            lines.append(f"Contact phone(s): {', '.join(meta.contact_phones)}.")
        if meta.contact_notes:
            lines.append(f"Contact notes: {_escape_contact_memory_tags(meta.contact_notes)}")
        memories = contact_memories_block(meta.contact_memories)
        if memories:
            lines.append(memories)
    else:
        lines.append(
            "No matching Inkbox contact record is loaded; use the phone number or a neutral greeting.",
        )
    if meta.direction == "outbound":
        if meta.outbound_purpose:
            lines.append(
                "This is an outbound call you placed. Purpose: "
                + _escape_contact_memory_tags(meta.outbound_purpose)
            )
        if meta.outbound_reason:
            lines.append(f"Reason for the call: {_escape_contact_memory_tags(meta.outbound_reason)}")
        if meta.outbound_scheduled_by:
            lines.append(
                "This call was scheduled by: "
                + _escape_contact_memory_tags(meta.outbound_scheduled_by)
            )
        if meta.outbound_conversation_summary:
            lines.append(
                "Summary of the prior conversation that led to this call:\n"
                + _escape_contact_memory_tags(meta.outbound_conversation_summary),
            )
        if meta.outbound_context:
            lines.append(
                f"Relevant outbound-call context:\n{_escape_contact_memory_tags(meta.outbound_context)}"
            )
        if meta.outbound_opening:
            lines.append(
                "Preferred opening message (say this naturally as your first turn): "
                + _escape_contact_memory_tags(meta.outbound_opening),
            )
        lines.append(
            "For outbound calls, do not open with a generic offer to help. Start by explaining why you are calling, then ask the next specific question or give the requested update.",
        )
    if gate_mode:
        lines.extend([
            "This is your call. Run it like a sharp, easygoing human assistant would: talk freely, react, joke "
            "if it fits, ask what you need to ask, think out loud, change subject with the caller. Use what "
            "you know from the notes below and from earlier in the call. There is no script.",
            f"{CONSULT_TOOL_NAME} is how you get anything done or found out. Whenever something is more than "
            "talk, whatever it is, use it: say what's needed in plain English with the details you were given. "
            "Don't decide in advance whether it's possible; you'll be told. It takes a few seconds, like a "
            "person typing, so keep chatting and bring the outcome in when it lands.",
            "Everything else is just conversation, and that is yours. Don't use the tool for it.",
            "You are Blatbot, Aaron's assistant, and that is all there is to it. You don't know or talk about "
            "how you work. When you check or do something, it's simply you doing it: 'let me look', 'give me "
            "a sec', 'done'. If someone asks whether you're an AI, say yes, you're Aaron's AI assistant, and "
            "move on.",
            "One honest-assistant rule: don't say something was booked, sent, changed or confirmed unless "
            "you were told so, and don't make up facts about Aaron's schedule or messages. If the tool "
            "says Aaron has to approve something first, tell the caller that.",
            f"When the caller is wrapping up, say goodbye and call {HANG_UP_CALL_TOOL_NAME}.",
        ])
    else:
      lines.extend([
          "Do not perform a context lookup before greeting the caller. Do not say you are waiting on a lookup or checking context.",
          f"To do real work NOW in the project ({meta.project_dir or 'the working directory'}) "
          f"or Inkbox account - look up contacts, inspect texts/calls, use Inkbox tools, "
          f"read or edit files, run commands or tests, check git, or search the codebase - "
          f"call {CONSULT_TOOL_NAME} with a plain-English request. It runs the Claude Code "
          "agent in the caller's ongoing conversation and returns a spoken-friendly answer; read that answer back in your own voice.",
          f"If the caller wants work done AFTER the call (or accepts a deferral), call "
          f"{POST_CALL_ACTION_TOOL_NAME} to queue it. Tell them it's queued for after the "
          "call; do not claim it is already done.",
          f"If the caller changes or cancels queued after-call work, call "
          f"{EDIT_POST_CALL_ACTION_TOOL_NAME} or {DELETE_POST_CALL_ACTION_TOOL_NAME} with "
          f"the action_index returned when it was queued. If {CONSULT_TOOL_NAME} already "
          f"did the work a queued action describes, delete that action so it isn't repeated.",
          f"When the caller says goodbye or the conversation is clearly done, call "
          f"{HANG_UP_CALL_TOOL_NAME}: the first call arms hangup and asks you to say a short "
          "goodbye; after the goodbye, call it once more to actually end the call.",
          f"Do NOT call {CONSULT_TOOL_NAME} for greetings, small talk, or questions you "
          "can answer directly from the loaded call context. Use it whenever the caller wants "
          "something done in code, asks for contact/account context you do not already have, "
          "or needs an Inkbox tool lookup.",
          "While a tool runs you may say a brief 'one moment' so the caller isn't left in silence.",
      ])
    if gate_mode:
        # The voice agent is a surface: it gets who it is and who it's talking to,
        # nothing about the platform underneath.
        drop = ("identity handle", "contact id", "shared Inkbox iMessage", "never state or promise a number")
        kept = []
        for ln in lines:
            if any(d in ln for d in drop):
                continue
            ln = (ln.replace("Your Inkbox agent email address", "Your email address")
                    .replace("Known Inkbox contact info is already loaded", "You already know this caller")
                    .replace("No matching Inkbox contact record is loaded", "You don't know who this caller is yet")
                    .replace("Inkbox ", ""))
            kept.append(ln)
        lines = kept
    if additional.strip():
        lines += ["", additional.strip()]
    return "\n".join(lines)


def build_realtime_greeting(meta: RealtimeCallMeta, gate_mode: bool = False) -> str:
    """Instructions for the proactive opening line spoken at pickup."""
    first_name = (
        _escape_contact_memory_tags(meta.contact_name.split()[0])
        if meta.contact_known and meta.contact_name
        else "there"
    )
    if meta.direction == "outbound" and meta.outbound_opening:
        return (
            "Open the call by saying this naturally as the very first thing, with no greeting before it:\n"
            f"{_escape_contact_memory_tags(meta.outbound_opening)}"
        )
    if meta.direction == "outbound" and meta.outbound_purpose:
        return (
            f"Greet {first_name} briefly, then immediately explain that you are calling because: "
            f"{_escape_contact_memory_tags(meta.outbound_purpose)}. "
            "Do not ask a generic how-can-I-help question."
        )
    if gate_mode:
        who = f"The person calling is {first_name}. " if first_name != "there" else ""
        return (
            f"{who}You speak first. The moment the call connects, introduce yourself: say hello, that this is "
            "Blatbot, and that you're Aaron Blatnoy's assistant (if it's Aaron calling, just that it's "
            "Blatbot), then ask what they need. Your own words, relaxed, one or two short sentences, like a "
            "person answering a phone. Then stop and let them talk."
        )
    return (
        f"Greet the caller now as the very first thing you say. Say something like "
        f"'Hi {first_name}, this is your Claude Code Inkbox agent - how can I help?' "
        f"Keep it to one short sentence and then wait for them to respond."
    )


# ----------------------------------------------------------------------
# Tool schema
# ----------------------------------------------------------------------


def _consult_tool_schema(gate_mode: bool = False) -> Dict[str, Any]:
    if gate_mode:
        return {
            "type": "function",
            "name": CONSULT_TOOL_NAME,
            "description": (
                "How you get anything done or found out. Anything beyond conversation goes here: whenever "
                "the caller wants something done, looked up, checked, changed or followed up, or you need "
                "something you don't already know. Don't judge whether it's possible; ask, and you'll get back "
                "the outcome, including when it can't be done or needs Aaron's approval first. Takes a few "
                "seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What's needed, in plain English, with who is asking and whatever details they gave."
                        ),
                    },
                },
                "required": ["query"],
            },
        }
    return {
        "type": "function",
        "name": CONSULT_TOOL_NAME,
        "description": (
            "Hand a request to the Claude Code agent working in the project, when "
            "the caller wants real work done - look up contacts, inspect Inkbox "
            "texts/calls/email, read/edit files, run commands or tests, check git "
            "status, search the codebase, etc. The request runs in the caller's "
            "ongoing conversation and you get back a spoken-friendly "
            "answer to read aloud. Do NOT use this for greetings or small talk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to ask Claude Code, in plain English. Include enough "
                        "context that it can act standalone."
                    ),
                },
            },
            "required": ["query"],
        },
    }


def _post_call_action_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": POST_CALL_ACTION_TOOL_NAME,
        "description": (
            "Queue work for Claude Code to do AFTER this call ends — e.g. open a "
            "PR, run a long task, email/text the caller a summary. Tell the caller "
            "it's queued; do NOT claim it is already done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Plain-English task for Claude Code. Include the outcome wanted.",
                },
                "details": {
                    "type": "string",
                    "description": "Optional extra context, constraints, or draft text.",
                },
            },
            "required": ["action"],
        },
    }


def _edit_post_call_action_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": EDIT_POST_CALL_ACTION_TOOL_NAME,
        "description": (
            "Edit a queued after-call action by its one-based action_index "
            "(returned by register_post_call_action) when the caller changes it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_index": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "One-based index of the queued action to edit.",
                },
                "action": {
                    "type": "string",
                    "description": "Replacement task. Omit to keep the current task.",
                },
                "details": {
                    "type": "string",
                    "description": "Replacement details. Empty string clears details.",
                },
            },
            "required": ["action_index"],
        },
    }


def _delete_post_call_action_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": DELETE_POST_CALL_ACTION_TOOL_NAME,
        "description": (
            "Delete a queued after-call action by its one-based action_index "
            "when the caller cancels it or it's already been handled."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_index": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "One-based index of the queued action to delete.",
                },
            },
            "required": ["action_index"],
        },
    }


def _hang_up_call_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": HANG_UP_CALL_TOOL_NAME,
        "description": (
            "End the live phone call. TWO-STEP: the first call does NOT hang up — "
            "it prompts you to say a short goodbye. After the goodbye, call "
            "hang_up_call again to actually end the call. Use only when the caller "
            "asks to hang up, says goodbye, or the conversation is clearly complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional short reason for ending the call.",
                },
            },
            "required": [],
        },
    }


# ----------------------------------------------------------------------
# Bridge lifecycle
# ----------------------------------------------------------------------


class RealtimeBridgeConnectError(Exception):
    """Raised when OpenAI Realtime cannot be opened before Inkbox accept."""

    def __init__(self, cause: Any):
        self.cause = cause
        super().__init__(f"OpenAI Realtime connect failed: {cause}")


@dataclass
class OpenedRealtimeBridge:
    """A connected OpenAI Realtime session, ready to bridge to Inkbox."""

    session: Any
    openai_ws: Any
    state: _BridgeState
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
        """Bridge the open OpenAI session to ``inkbox_ws`` for the whole call.

        Args:
            inkbox_ws (Any): The accepted Inkbox call WebSocket (raw media).
            on_agent_consult (AgentConsultCallback): Runs a consult and returns text.
            on_post_call_actions (PostCallActionsCallback): Runs queued actions after hangup.
            on_call_ended (CallEndedCallback): Runs a follow-up reflection when no actions queued.

        Returns:
            None: Returns when either side closes the socket.
        """
        state = self.state
        openai_ws = self.openai_ws
        try:
            inkbox_task = asyncio.create_task(
                _inkbox_to_openai_pump(inkbox_ws, openai_ws, state, self.meta),
                name=f"realtime-inkbox-pump-{self.meta.call_id}",
            )
            openai_task = asyncio.create_task(
                _openai_to_inkbox_pump(
                    openai_ws=openai_ws,
                    inkbox_ws=inkbox_ws,
                    state=state,
                    config=self.config,
                    meta=self.meta,
                    on_agent_consult=on_agent_consult,
                ),
                name=f"realtime-openai-pump-{self.meta.call_id}",
            )
            done, _pending = await asyncio.wait(
                {inkbox_task, openai_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc:
                    logger.warning("[realtime] pump %s raised: %s", task.get_name(), exc)
        finally:
            state.closed = True
            tasks = [
                task for task in (
                    locals().get("inkbox_task"),
                    locals().get("openai_task"),
                )
                if task is not None
            ]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await _maybe_close_ws(inkbox_ws)
            await self.close()
            await _settle_tasks(tasks, label="pump")
            await _cancel_consult_tasks(state)

        # After teardown: run queued after-call work, or a follow-up reflection.
        await _dispatch_post_call(state, on_post_call_actions, on_call_ended)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            await self.openai_ws.close()
        with suppress(Exception):
            await self.session.close()


async def open_inkbox_realtime_bridge(
    *, config: RealtimeConfig, meta: RealtimeCallMeta
) -> OpenedRealtimeBridge:
    """Open OpenAI Realtime before the Inkbox WebSocket commits to media mode.

    Args:
        config (RealtimeConfig): Resolved realtime settings (key, model, voice).
        meta (RealtimeCallMeta): Per-call context for the session prompt.

    Returns:
        OpenedRealtimeBridge: A connected bridge ready to ``run()``.

    Raises:
        RealtimeBridgeConnectError: If aiohttp is missing, no key is set, or
            the WebSocket handshake fails / times out.
    """
    if aiohttp is None:
        raise RealtimeBridgeConnectError("aiohttp not available")
    if not config.has_credential:
        raise RealtimeBridgeConnectError("no OpenAI API key configured")

    session = aiohttp.ClientSession()
    openai_ws = None
    try:
        separator = "&" if "?" in config.base_url else "?"
        url = f"{config.base_url}{separator}{urlencode({'model': config.model})}"
        openai_ws = await asyncio.wait_for(
            session.ws_connect(
                url, headers={"Authorization": f"Bearer {config.api_key}"}, heartbeat=30
            ),
            timeout=config.connect_timeout_s,
        )
        await _send_session_update(openai_ws, config, meta)
        return OpenedRealtimeBridge(
            session=session,
            openai_ws=openai_ws,
            state=_BridgeState(gate_mode=config.gate_mode),
            config=config,
            meta=meta,
        )
    except Exception as exc:
        if openai_ws is not None:
            with suppress(Exception):
                await openai_ws.close()
        with suppress(Exception):
            await session.close()
        if isinstance(exc, RealtimeBridgeConnectError):
            raise
        raise RealtimeBridgeConnectError(exc) from exc


async def _cancel_consult_tasks(state: _BridgeState) -> None:
    """Cancel in-flight consult tasks and let them settle."""
    tasks = list(state.consult_tasks)
    state.consult_tasks.clear()
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    await _settle_tasks(tasks, label="consult")


async def _settle_tasks(tasks: List["asyncio.Task[Any]"], *, label: str) -> None:
    """Let cancelled background tasks drain, but never block call teardown."""
    if not tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=TASK_CANCEL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        names = ", ".join(task.get_name() for task in tasks)
        logger.warning("[realtime] timed out waiting for %s task cancellation: %s", label, names)


# ----------------------------------------------------------------------
# Session config + pumps
# ----------------------------------------------------------------------


async def _send_session_update(
    openai_ws: Any, config: RealtimeConfig, meta: RealtimeCallMeta
) -> None:
    """Send the initial ``session.update`` configuring audio, VAD, and tools."""
    payload = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": config.model,
            "instructions": build_realtime_instructions(meta, config.additional_instructions, config.gate_mode),
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": AUDIO_FORMAT_TELEPHONY,
                    # Phone handset audio: filter line noise and the agent's own echo
                    # before VAD sees it, so they don't register as the caller speaking.
                    "noise_reduction": {"type": os.getenv("INKBOX_REALTIME_NOISE_REDUCTION") or "near_field"},
                    "transcription": ({"model": INPUT_TRANSCRIPTION_MODEL, "prompt": config.vocabulary}
                                      if config.vocabulary else {"model": INPUT_TRANSCRIPTION_MODEL}),
                    # Server-side VAD: the model auto-detects speech start/stop,
                    # auto-responds, and supports barge-in. The bridge never
                    # triggers response.create per turn itself.
                    "turn_detection": {
                        "type": "server_vad",
                        # Higher threshold: breaths, clicks and echo no longer count as
                        # speech, so they stop cutting the agent off mid-sentence. A real
                        # interruption (someone actually talking) still gets through.
                        "threshold": float(os.getenv("INKBOX_REALTIME_VAD_THRESHOLD") or 0.85),
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": int(os.getenv("INKBOX_REALTIME_VAD_SILENCE_MS") or 700),
                        "create_response": True,
                        # Gate mode: OpenAI never cuts the agent off by itself. A siren, a
                        # passer-by or a cough would do it constantly on a city street. The
                        # bridge interrupts only when sound is sustained (see BARGE_IN_MS).
                        "interrupt_response": not config.gate_mode,
                    },
                },
                "output": {
                    "format": AUDIO_FORMAT_TELEPHONY,
                    "voice": config.voice,
                },
            },
            "tools": ([_consult_tool_schema(gate_mode=True), _hang_up_call_tool_schema()] if config.gate_mode else [
                _consult_tool_schema(),
                _post_call_action_tool_schema(),
                _edit_post_call_action_tool_schema(),
                _delete_post_call_action_tool_schema(),
                _hang_up_call_tool_schema(),
            ]),
            # Gate mode: the model handles pure conversation itself and must send
            # anything factual or actionable through consult (enforced by prompt;
            # the relay of a consult answer is a speech-only response).
            "tool_choice": "auto",
        },
    }
    await openai_ws.send_str(json.dumps(payload))


async def _maybe_send_greeting(
    openai_ws: Any, state: _BridgeState, meta: RealtimeCallMeta
) -> None:
    """Fire the proactive opening line once, so calls don't open with silence."""
    if state.greeting_triggered:
        return
    state.greeting_triggered = True
    try:
        await openai_ws.send_str(json.dumps({
            "type": "response.create",
            "response": {"instructions": build_realtime_greeting(meta, state.gate_mode), "tool_choice": "none"},
        }))
    except Exception as exc:
        logger.debug("[realtime] greeting send failed: %s", exc)


async def _inkbox_to_openai_pump(
    inkbox_ws: Any, openai_ws: Any, state: _BridgeState, meta: RealtimeCallMeta
) -> None:
    """Forward caller audio from Inkbox to OpenAI; fire the opening greeting.

    Inkbox sends ``{"event": "media", "media": {"payload": "<b64>"}}``; we
    re-emit as ``input_audio_buffer.append`` and let server-side VAD drive
    turns. The greeting fires on ``start`` (or first media if no start).
    """
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
                await _maybe_send_greeting(openai_ws, state, meta)
            elif event == "media":
                if not state.greeting_triggered:
                    await _maybe_send_greeting(openai_ws, state, meta)
                payload_b64 = (frame.get("media") or {}).get("payload")
                if payload_b64:
                    await openai_ws.send_str(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": payload_b64,
                    }))
            elif event in {"stop", "closed", "hangup"}:
                logger.info("[realtime] Inkbox WS signaled %s", event)
                return
        elif msg.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        }:
            return


async def _openai_to_inkbox_pump(
    *,
    openai_ws: Any,
    inkbox_ws: Any,
    state: _BridgeState,
    config: RealtimeConfig,
    meta: RealtimeCallMeta,
    on_agent_consult: AgentConsultCallback,
) -> None:
    """Forward model audio to Inkbox and handle ``consult_agent`` calls."""
    # Function-call accumulation keyed by item_id. The name arrives on
    # output_item.added; args stream via ...arguments.delta and finalize on
    # ...arguments.done. Dedupe by call_id so a call dispatches at most once.
    fn_calls: Dict[str, Dict[str, str]] = {}
    dispatched: set = set()

    async def _finalize_fn_call(entry: Dict[str, str]) -> None:
        cid = (entry or {}).get("call_id") or ""
        if not cid or cid in dispatched:
            return
        dispatched.add(cid)
        logger.info(
            "[realtime] dispatching tool call name=%s call_id=%s",
            entry.get("name") or "",
            cid,
        )
        coro = _dispatch_tool_call(
            openai_ws=openai_ws,
            inkbox_ws=inkbox_ws,
            call_id=cid,
            name=entry.get("name") or "",
            arguments_json=entry.get("args") or "{}",
            state=state,
            config=config,
            on_agent_consult=on_agent_consult,
        )
        # The consult runs a full Claude Code turn (seconds). Awaiting it here
        # would freeze this read loop — no audio, no barge-in — so dispatch it
        # as a background task; it submits the tool result when it finishes,
        # which is exactly the async-tool flow gpt-realtime expects.
        task = asyncio.create_task(coro, name=f"realtime-consult-{cid}")
        state.consult_tasks.add(task)
        def _done(done_task: "asyncio.Task[None]") -> None:
            state.consult_tasks.discard(done_task)
            if done_task.cancelled():
                logger.info("[realtime] tool call cancelled call_id=%s", cid)
                return
            exc = done_task.exception()
            if exc:
                logger.warning("[realtime] tool call task failed call_id=%s: %s", cid, exc)

        task.add_done_callback(_done)

    async def _relay_transcript(party: str, text: str) -> None:
        # Realtime runs the WS in raw-media mode, so Inkbox does not create its
        # own STT transcript. Mirror finalized turns back into the call record.
        with suppress(Exception):
            await inkbox_ws.send_str(json.dumps({
                "event": "transcript",
                "party": party,
                "text": text,
                "is_final": True,
            }))

    async for msg in openai_ws:
        if state.closed:
            return
        if msg.type != aiohttp.WSMsgType.TEXT:
            if msg.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            }:
                return
            continue
        try:
            frame = json.loads(msg.data)
        except (TypeError, ValueError):
            continue
        if not isinstance(frame, dict):
            continue
        ftype = frame.get("type", "")

        # GA: response.output_audio.delta; beta: response.audio.delta.
        if ftype in ("response.output_audio.delta", "response.audio.delta"):
            delta_b64 = frame.get("delta") or ""
            state.last_sound_at = time.monotonic()
            if delta_b64:
                # 8 kHz, 1 byte per sample (G.711): bytes / 8000 = seconds of speech.
                secs = (len(delta_b64) * 3 / 4) / 8000.0
                state.playback_until = max(state.playback_until, time.monotonic()) + secs
            if delta_b64:
                out: Dict[str, Any] = {
                    "event": "media",
                    "media": {"payload": delta_b64, "track": "outbound"},
                }
                if state.stream_id:
                    out["stream_id"] = state.stream_id
                try:
                    await inkbox_ws.send_str(json.dumps(out))
                except Exception as exc:
                    logger.debug("[realtime] Inkbox WS send failed: %s", exc)
                    return

        # A response's audio finished — tell Inkbox to flush/play.
        elif ftype in ("response.output_audio.done", "response.audio.done"):
            done: Dict[str, Any] = {"event": "audio_done"}
            if state.stream_id:
                done["stream_id"] = state.stream_id
            with suppress(Exception):
                await inkbox_ws.send_str(json.dumps(done))

        # Caller started speaking (barge-in) — drop queued outbound audio.
        elif ftype == "response.created":
            state.response_active = True
        elif ftype in ("response.done", "response.cancelled"):
            state.response_active = False
            if config.gate_mode and not state.barge_in_enabled:
                state.barge_in_enabled = True  # the introduction is out; interruptions allowed from here
        elif ftype == "input_audio_buffer.speech_started":
            state.last_sound_at = time.monotonic()
            if not config.gate_mode:
                state.playback_until = 0.0
                with suppress(Exception):
                    await inkbox_ws.send_str(json.dumps({"event": "clear"}))
                continue
            state.speech_started_at = time.monotonic()
            if not state.barge_in_enabled:
                continue  # still introducing itself; let it finish
            agent_talking = state.response_active or time.monotonic() < state.playback_until
            if not agent_talking:
                continue  # nothing to interrupt
            hold_ms = int(os.getenv("INKBOX_REALTIME_BARGE_IN_MS") or 550)

            async def _barge(started: float = state.speech_started_at, hold: float = hold_ms / 1000.0) -> None:
                await asyncio.sleep(hold)
                if state.speech_started_at != started:
                    return  # the sound stopped (a blip): the agent keeps talking
                state.playback_until = 0.0
                with suppress(Exception):
                    await openai_ws.send_str(json.dumps({"type": "response.cancel"}))
                with suppress(Exception):
                    await inkbox_ws.send_str(json.dumps({"event": "clear"}))

            if state.barge_task is not None and not state.barge_task.done():
                state.barge_task.cancel()
            state.barge_task = asyncio.create_task(_barge())
        elif ftype == "input_audio_buffer.speech_stopped":
            state.last_sound_at = time.monotonic()
            state.speech_started_at = None

        # Transcripts (for logging / consult context).
        elif ftype in (
            "response.output_audio_transcript.done",
            "response.audio_transcript.done",
        ):
            text = (frame.get("transcript") or "").strip()
            if text:
                state.transcript.append(("agent", text))
                await _relay_transcript("local", text)
        elif ftype == "conversation.item.input_audio_transcription.completed":
            text = (frame.get("transcript") or "").strip()
            if text:
                state.transcript.append(("caller", text))
                await _relay_transcript("remote", text)

        # Function-call lifecycle.
        elif ftype == "response.output_item.added":
            item = frame.get("item") or {}
            if item.get("type") == "function_call":
                item_id = item.get("id") or frame.get("item_id") or ""
                if item_id:
                    fn_calls[item_id] = {
                        "call_id": item.get("call_id") or "",
                        "name": item.get("name") or "",
                        "args": item.get("arguments") or "",
                    }
        elif ftype == "response.function_call_arguments.delta":
            key = frame.get("item_id") or frame.get("call_id") or ""
            if not key:
                continue
            entry = fn_calls.setdefault(key, {"call_id": "", "name": "", "args": ""})
            if not entry.get("call_id") and frame.get("call_id"):
                entry["call_id"] = frame["call_id"]
            if not entry.get("name") and frame.get("name"):
                entry["name"] = frame["name"]
            entry["args"] = (entry.get("args") or "") + (frame.get("delta") or "")
        elif ftype == "response.function_call_arguments.done":
            key = frame.get("item_id") or frame.get("call_id") or ""
            entry = fn_calls.get(key) or fn_calls.get(frame.get("call_id") or "") or {}
            if frame.get("call_id"):
                entry["call_id"] = frame["call_id"]
            if frame.get("name"):
                entry["name"] = frame["name"]
            if frame.get("arguments"):
                entry["args"] = frame["arguments"]
            await _finalize_fn_call(entry)
        # Fallback: a completed function_call item.
        elif ftype in ("response.output_item.done", "conversation.item.done"):
            item = frame.get("item") or {}
            if item.get("type") == "function_call":
                await _finalize_fn_call({
                    "call_id": item.get("call_id") or "",
                    "name": item.get("name") or "",
                    "args": item.get("arguments") or "{}",
                })
        elif ftype == "error":
            err = frame.get("error") or {}
            if err.get("code") in ("conversation_already_has_active_response", "response_cancel_not_active"):
                logger.debug("[realtime] benign turn-taking frame: %s", err.get("code"))
            else:
                logger.warning("[realtime] OpenAI error frame: %s", err)


# ----------------------------------------------------------------------
# Tool dispatch
# ----------------------------------------------------------------------


async def _dispatch_tool_call(
    *,
    openai_ws: Any,
    inkbox_ws: Any,
    call_id: str,
    name: str,
    arguments_json: str,
    state: _BridgeState,
    config: RealtimeConfig,
    on_agent_consult: AgentConsultCallback,
) -> None:
    """Handle a function call from the Realtime model.

    Dispatches the five call tools: consult, register/edit/delete post-call
    action, and the two-step hang_up_call.
    """
    try:
        args = json.loads(arguments_json or "{}")
    except (TypeError, ValueError):
        args = {}

    if name == POST_CALL_ACTION_TOOL_NAME:
        await _handle_register_action(openai_ws, call_id, args, state)
        return
    if name == EDIT_POST_CALL_ACTION_TOOL_NAME:
        await _handle_edit_action(openai_ws, call_id, args, state)
        return
    if name == DELETE_POST_CALL_ACTION_TOOL_NAME:
        await _handle_delete_action(openai_ws, call_id, args, state)
        return
    if name == HANG_UP_CALL_TOOL_NAME:
        await _handle_hang_up(openai_ws, inkbox_ws, call_id, args, state)
        return
    if name != CONSULT_TOOL_NAME:
        await _submit_tool_result(
            openai_ws, call_id, {"error": f"Tool '{name}' is not available on calls."}
        )
        return

    query = (args.get("query") or "").strip()
    if not query:
        await _submit_tool_result(openai_ws, call_id, {"error": "missing query argument"})
        return

    # Best-effort interim cue so the caller hears something while Claude works.
    # Gate mode: the model leads and says its own lead-in, so nothing is injected.
    with suppress(Exception):
        if not config.gate_mode:
          await openai_ws.send_str(json.dumps({
            "type": "response.create",
            "response": {
                "tool_choice": "none",
                "instructions": (
                    "Say only a very short, natural acknowledgement of two to four words, the way a person "
                    "does while they look something up. Vary it each time, for example 'Sure, one sec.', "
                    "'Mm, let me check.', 'Okay, give me a moment.' Say nothing else and give no information."
                ) if config.gate_mode else "Say only 'One moment.'",
            },
        }))

    async def _hold_lines() -> None:
        # A person doing something on the phone doesn't go silent for ten seconds.
        # Every so often, while the consult is still running, let the model say one
        # brief in-character hold line. Speech only; errors (e.g. the caller or the
        # model is already talking) are ignored.
        # Fill dead air, not time: speak only once the line has been quiet for a
        # couple of seconds, so a lead-in or the caller talking pushes it back.
        quiet_needed = 2.5
        spoken = 0
        while spoken < 8:
            await asyncio.sleep(0.5)
            if time.monotonic() - state.last_sound_at < quiet_needed:
                continue
            spoken += 1
            quiet_needed = 5.0  # after the first fill, let silences breathe a bit more
            state.last_sound_at = time.monotonic()
            with suppress(Exception):
                await openai_ws.send_str(json.dumps({
                    "type": "response.create",
                    "response": {
                        "tool_choice": "none",
                        "instructions": (
                            "You are still waiting on the lookup you started and the line has gone quiet. Fill the gap "
                            "the way a person does while working: one short, natural line, different from anything "
                            "you said before on this call. If you have not said anything since starting the lookup, "
                            "say what you are doing ('okay, pulling up his calendar now'). Otherwise a hold line "
                            "('still loading', 'bear with me', 'almost there') or a light, relevant remark. "
                            "Give no information from the lookup and make no promises."
                        ),
                    },
                }))

    hold_task = None  # the model leads; the bridge injects nothing into the conversation
    try:
        answer = await asyncio.wait_for(
            on_agent_consult(query, list(state.transcript)),
            timeout=config.consult_timeout_s,
        )
    except asyncio.TimeoutError:
        if hold_task is not None:
            hold_task.cancel()
        await _submit_tool_result(openai_ws, call_id, {
            "error": "consult timed out",
            "message": "Tell the caller you couldn't finish that right now; offer to follow up.",
        })
        return
    except Exception as exc:
        if hold_task is not None:
            hold_task.cancel()
        logger.warning("[realtime] consult failed: %s", exc)
        await _submit_tool_result(openai_ws, call_id, {
            "error": f"consult error: {exc}",
            "message": "Apologize briefly and ask if you can help another way.",
        })
        return

    if hold_task is not None:
        hold_task.cancel()
    if config.gate_mode:
        await _submit_tool_result(openai_ws, call_id, {
            "status": "ok",
            "result": answer,
            "note": "Tell the caller in your own words; keep the facts as given.",
        }, state=state)
        return
    await _submit_tool_result(openai_ws, call_id, {
        "status": "ok",
        "answer": answer,
        "instructions": "Read the answer back to the caller in your own voice. Keep it natural and concise.",
    })


async def _handle_register_action(
    openai_ws: Any, call_id: str, args: Dict[str, Any], state: _BridgeState
) -> None:
    """Queue an after-call action; the model is told it's queued, not done."""
    action = (args.get("action") or "").strip()
    if not action:
        await _submit_tool_result(openai_ws, call_id, {"error": "missing action argument"})
        return
    state.post_call_actions.append({"action": action, "details": (args.get("details") or "").strip()})
    await _submit_tool_result(openai_ws, call_id, {
        "status": "queued",
        "action_index": len(state.post_call_actions),
        "action_count": len(state.post_call_actions),
        "message": "Tell the caller the action is queued for after the call; do not claim it is done.",
    })


async def _handle_edit_action(
    openai_ws: Any, call_id: str, args: Dict[str, Any], state: _BridgeState
) -> None:
    """Edit a queued action in place by its one-based index."""
    index = _action_index(args)
    if index < 1 or index > len(state.post_call_actions):
        await _submit_tool_result(openai_ws, call_id, {
            "error": "invalid action_index", "action_count": len(state.post_call_actions),
        })
        return
    if "action" not in args and "details" not in args:
        await _submit_tool_result(openai_ws, call_id, {"error": "missing action or details argument"})
        return
    queued = state.post_call_actions[index - 1]
    if "action" in args:
        new_action = (args.get("action") or "").strip()
        if not new_action:
            await _submit_tool_result(openai_ws, call_id, {"error": "action cannot be empty"})
            return
        queued["action"] = new_action
    if "details" in args:
        queued["details"] = (args.get("details") or "").strip()
    await _submit_tool_result(openai_ws, call_id, {
        "status": "updated", "action_index": index, "action": queued,
        "message": "If the caller needs to know, confirm briefly the queued work was changed.",
    })


async def _handle_delete_action(
    openai_ws: Any, call_id: str, args: Dict[str, Any], state: _BridgeState
) -> None:
    """Remove a queued action by its one-based index."""
    index = _action_index(args)
    if index < 1 or index > len(state.post_call_actions):
        await _submit_tool_result(openai_ws, call_id, {
            "error": "invalid action_index", "action_count": len(state.post_call_actions),
        })
        return
    deleted = state.post_call_actions.pop(index - 1)
    await _submit_tool_result(openai_ws, call_id, {
        "status": "deleted", "deleted_action": deleted,
        "action_count": len(state.post_call_actions),
        "message": "If the caller needs to know, confirm briefly it was canceled.",
    })


async def _handle_hang_up(
    openai_ws: Any, inkbox_ws: Any, call_id: str, args: Dict[str, Any], state: _BridgeState,
    *, gate_mode: bool = False,
) -> None:
    """Two-step hangup: arm + goodbye, then drop the line on the second call."""
    if inkbox_ws is None:
        await _submit_tool_result(openai_ws, call_id, {"error": "hangup unavailable without Inkbox websocket"})
        return

    now = time.monotonic()
    armed = state.hangup_armed_at
    # First attempt (or a stale arm past the window) → arm and say goodbye
    # rather than dropping the caller mid-farewell.
    if gate_mode and armed is None:
        # Forced-tool sessions cannot speak and then call the tool again, so the
        # bridge says the goodbye as a speech-only response and ends the call itself.
        state.hangup_armed_at = now
        await _submit_tool_result(openai_ws, call_id, {"status": "ending"}, response={
            "tool_choice": "none",
            "instructions": "Say one brief, natural goodbye and nothing else.",
        })
        await asyncio.sleep(4.0)
        armed = state.hangup_armed_at
        now = time.monotonic()
        call_id = ""
    if not gate_mode and (armed is None or (now - armed) > HANGUP_CONFIRM_WINDOW_S):
        state.hangup_armed_at = now
        await _submit_tool_result(openai_ws, call_id, {
            "status": "confirm_goodbye",
            "message": (
                "Don't hang up yet. Say a brief, natural goodbye now, then call "
                "hang_up_call once more to actually end the call."
            ),
        })
        return

    # Second attempt within the window → perform the real hangup.
    reason = (args.get("reason") or "").strip()
    # Inkbox ends the call on a `stop` event; `hangup` is ignored server-side.
    stop_frame: Dict[str, Any] = {"event": "stop"}
    if reason:
        stop_frame["reason"] = reason
    if state.stream_id:
        stop_frame["stream_id"] = state.stream_id
    # Don't ask the model to speak again — we're ending the call.
    if call_id:
        await _submit_tool_result(
            openai_ws, call_id,
            {"status": "hangup_requested", "reason": reason, "message": "The call is ending now."},
            create_response=False,
        )
    try:
        # Let the spoken goodbye land before we drop the carrier leg: wait for the
        # agent's current response to finish and for its audio to finish playing
        # on the phone (speech is generated faster than it plays), then a beat.
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if not state.response_active and time.monotonic() >= state.playback_until:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.8)
        await inkbox_ws.send_str(json.dumps(stop_frame))
    except Exception as exc:
        logger.debug("[realtime] hangup frame send failed: %s", exc)
    state.closed = True
    await _maybe_close_ws(inkbox_ws)
    await _maybe_close_ws(openai_ws)


def _action_index(args: Dict[str, Any]) -> int:
    try:
        return int(args.get("action_index"))
    except (TypeError, ValueError):
        return 0


async def _dispatch_post_call(
    state: _BridgeState,
    on_post_call_actions: PostCallActionsCallback,
    on_call_ended: CallEndedCallback,
) -> None:
    """Run exactly one follow-up after the call: queued actions, else a reflection."""
    if state.post_call_actions:
        try:
            await on_post_call_actions(list(state.post_call_actions), list(state.transcript))
        except Exception as exc:
            logger.warning("[realtime] post-call action dispatch failed: %s", exc)
    else:
        try:
            await on_call_ended(list(state.transcript))
        except Exception as exc:
            logger.warning("[realtime] call-ended dispatch failed: %s", exc)


async def _maybe_close_ws(ws: Any) -> None:
    """Close a WS whether its close() is sync or a coroutine."""
    close = getattr(ws, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        pass


async def _submit_tool_result(
    openai_ws: Any, call_id: str, output: Dict[str, Any], *, create_response: bool = True,
    response: Optional[Dict[str, Any]] = None,
    state: Optional["_BridgeState"] = None,
) -> None:
    """Submit a function_call_output and (optionally) prompt the model to speak.

    Args:
        openai_ws (Any): The OpenAI Realtime WebSocket.
        call_id (str): The function call id being answered.
        output (dict): The tool result payload.
        create_response (bool): Whether to ask the model to respond afterward.
            False on hangup, where we don't want another spoken turn.
    """
    try:
        await openai_ws.send_str(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output),
            },
        }))
        if not create_response:
            return
        # Bare response.create — let the session's audio settings apply (GA
        # rejects a modalities field here).
        if state is not None:
            # Wait for a natural opening: the agent has finished its current
            # response, its audio has finished playing on the phone, and the
            # caller isn't mid-sentence. Then speak the result as a new turn.
            deadline = time.monotonic() + 25.0
            while time.monotonic() < deadline and not state.closed:
                now = time.monotonic()
                if (not state.response_active and now >= state.playback_until + 0.3
                        and now - state.last_sound_at >= 0.6):
                    break
                await asyncio.sleep(0.1)
        if response:
            await openai_ws.send_str(json.dumps({"type": "response.create", "response": response}))
            return
        await openai_ws.send_str(json.dumps({"type": "response.create"}))
    except Exception as exc:
        logger.debug("[realtime] submit_tool_result failed: %s", exc)
