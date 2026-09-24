"""GPT-Live bridge: drive both pumps with fake sockets. No network."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

import aiohttp

from inkbox_claude import live
from inkbox_claude import realtime as rt


def make_meta(**kw):
    fields = {f: None for f in rt.RealtimeCallMeta.__dataclass_fields__}
    fields.update(call_id="call-1", direction="inbound", contact_known=True, contact_name="Aaron Blatnoy",
                  remote_phone_number="+10000000000", contact_memories=[])
    fields.update(kw)
    return rt.RealtimeCallMeta(**fields)


class FakeWS:
    def __init__(self, script=()):
        self.script, self.sent = list(script), []

    async def send_str(self, x):
        self.sent.append(json.loads(x))

    async def close(self):
        pass

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for delay, frame in self.script:
            await asyncio.sleep(delay)
            yield NS(type=aiohttp.WSMsgType.TEXT, data=json.dumps(frame))


def test_instructions_are_a_surface_prompt_with_openai_policy_headings():
    text = live.build_live_instructions(make_meta(), "You are Blatbot, a relaxed guy.")
    for heading in ("Backchannel policy:", "Interruption policy:", "Delegation policy:", "Delegate to the backend when:"):
        assert heading in text
    low = text.lower()
    assert not any(w in low for w in ("claude", "openai", "inkbox", "gateway", "router", "ledger", "tool call"))
    assert "Blatbot" in live.build_live_greeting(make_meta())


def test_caller_audio_is_forwarded_and_greeting_is_sent_once():
    ink = FakeWS([(0, {"event": "start", "stream_id": "s1"}),
                  (0, {"event": "media", "media": {"payload": "QUJD"}}),
                  (0, {"event": "media", "media": {"payload": "REVG"}}),
                  (0, {"event": "stop"})])
    ows = FakeWS()
    state = live._LiveState(started=True)
    asyncio.run(live._inkbox_to_live_pump(ink, ows, state, make_meta()))
    types = [x["type"] for x in ows.sent]
    assert types[0] == "session.instructions.append" and "session.input_audio.mute" not in types
    assert types.count("session.instructions.append") == 1 and types.count("session.input_audio.append") == 2
    import base64
    # while the introduction plays, the caller's audio is replaced by silence of the same length
    assert base64.b64decode(ows.sent[1]["audio"]) == b"\xff" * 3
    intro = ows.sent[0]
    assert intro["delegation_id"] is None and "Blatbot" in intro["content"] and intro["content"].startswith("Greet the caller now")
    assert state.stream_id == "s1"


def test_input_is_unmuted_after_the_introduction_plays(monkeypatch):
    ows = FakeWS()
    state = live._LiveState(started=True, intro_muted=True)
    import time as _t
    state.intro_speech_seen, state.intro_last_speech_at = True, _t.monotonic()
    asyncio.run(live._unmute_after_intro(ows, state))
    assert ows.sent == [] and state.intro_muted is False


def test_delegation_builds_request_from_transcript_and_returns_commentary(monkeypatch):
    monkeypatch.setattr(live, "TRANSCRIPT_SETTLE_S", 0.05)
    script = [
        (0, {"type": "session.output_transcript.delta", "delta": "Hey, it's Blatbot."}),
        (0, {"type": "session.input_transcript.delta", "delta": "What's on my "}),
        (0, {"type": "session.input_transcript.delta", "delta": "calendar tomorrow?"}),
        (0, {"type": "session.output_audio.delta", "delta": "QUJD"}),
        (0, {"type": "session.delegation.created", "delegation": {"id": "item_1", "type": "delegation", "target": "client"}}),
        (0.4, {"type": "session.usage.updated"}),
    ]
    ows, ink = FakeWS(script), FakeWS()
    state = live._LiveState(started=True, stream_id="s1")
    seen = {}

    async def consult(query, transcript):
        seen["query"], seen["transcript"] = query, transcript
        return "Two   events tomorrow:\nSam at 5 PM, Alex at 6 PM."

    async def go():
        await live._live_to_inkbox_pump(ows, ink, state, make_meta(), consult)
        await asyncio.gather(*state.tasks, return_exceptions=True)
    asyncio.run(go())

    assert ink.sent[0] == {"event": "media", "media": {"payload": "QUJD", "track": "outbound"}, "stream_id": "s1"}
    assert "What's on my calendar tomorrow?" in seen["query"] and "act on this" in seen["query"]
    assert seen["transcript"][-1] == ("user", "What's on my calendar tomorrow?")
    out = [x for x in ows.sent if x["type"] == "session.commentary.append"]
    assert len(out) == 1 and out[0]["delegation_id"] == "item_1"
    assert out[0]["content"] == "Two events tomorrow: Sam at 5 PM, Alex at 6 PM."


def test_second_delegation_only_sends_what_is_new(monkeypatch):
    state = live._LiveState()
    state.add_fragment("caller", "Check my calendar tomorrow.")
    state.add_fragment("blatbot", "One sec.")
    q1 = live.build_delegation_query(state, make_meta())
    state.add_fragment("blatbot", "You have Sam at five.")
    state.add_fragment("caller", "Move Sam to six.")
    q2 = live.build_delegation_query(state, make_meta())
    fresh = q2.split("What was just said (act on this):")[1]
    assert "Move Sam to six." in fresh and "Check my calendar tomorrow." not in fresh
    assert "Check my calendar tomorrow." in q2.split("What was just said")[0]
    assert "Check my calendar tomorrow." in q1


def test_long_results_are_cut_to_fit_the_append_limit(monkeypatch):
    monkeypatch.setattr(live, "TRANSCRIPT_SETTLE_S", 0.01)
    ows = FakeWS()
    state = live._LiveState()
    state.add_fragment("caller", "Read me everything.")

    async def consult(q, t):
        return "word " * 2000
    asyncio.run(live._run_delegation(ows, state, make_meta(), "item_9", consult))
    assert len(ows.sent[0]["content"]) <= live.MAX_APPEND_CHARS + 4


def test_no_commentary_after_hangup(monkeypatch):
    monkeypatch.setattr(live, "TRANSCRIPT_SETTLE_S", 0.01)
    ows = FakeWS()
    state = live._LiveState()
    state.add_fragment("caller", "Email Bob.")

    async def consult(q, t):
        state.closed = True  # caller hung up while the work ran
        return "Sent."
    asyncio.run(live._run_delegation(ows, state, make_meta(), "item_2", consult))
    assert ows.sent == []


def test_live_drops_delivery_steering_but_keeps_persona():
    from inkbox_claude.config import _BLATBOT_VOICE, _BLATBOT_DELIVERY
    text = live.build_live_instructions(make_meta(), _BLATBOT_VOICE + _BLATBOT_DELIVERY + "\n\nNOTES FOR THIS CALL")
    assert "Match the caller" in text and "no uptalk" in text and "NOTES FOR THIS CALL" in text
    assert "chest voice" not in text and "melodic swoops" not in text


def test_silence_frames_are_not_mistaken_for_speech():
    import base64
    assert live._has_speech(base64.b64encode(bytes([0xFF, 0x7F] * 80)).decode()) is False
    assert live._has_speech(base64.b64encode(bytes(range(40, 200))).decode()) is True


def test_owner_is_recognised_by_phone_number_not_contact_name(monkeypatch):
    monkeypatch.setenv("INKBOX_APPROVER_PHONE", "+1 (555) 010-7788")
    nameless_owner = make_meta(contact_known=False, contact_name=None, remote_phone_number="+15550107788")
    assert "Hey Aaron, it's Blatbot." in live.build_live_greeting(nameless_owner)
    assert "Aaron himself" in live.build_live_instructions(nameless_owner)
    impostor = make_meta(contact_name="Aaron Blatnoy", remote_phone_number="+12125550100")
    assert "Aaron Blatnoy's assistant" in live.build_live_greeting(impostor)


def test_late_identity_tells_the_model_it_is_the_owner(monkeypatch):
    monkeypatch.setenv("INKBOX_APPROVER_PHONE", "+15550107788")
    ows = FakeWS()
    bridge = live.OpenedLiveBridge(session=None, openai_ws=ows, state=live._LiveState(), config=rt.RealtimeConfig(),
                                   meta=make_meta(contact_known=False, contact_name=None, remote_phone_number=None))
    asyncio.run(bridge.identify("+15550107788", "", "NOTES FOR THIS CALL ..."))
    kinds = [x["type"] for x in ows.sent]
    assert kinds == ["session.instructions.append", "session.thinking.append"]
    assert "Aaron himself" in ows.sent[0]["content"] and live.is_owner_call(bridge.meta)
    ows2 = FakeWS()
    b2 = live.OpenedLiveBridge(session=None, openai_ws=ows2, state=live._LiveState(), config=rt.RealtimeConfig(),
                               meta=make_meta(contact_known=False, contact_name=None, remote_phone_number=None))
    asyncio.run(b2.identify("+12125550100", "", ""))
    assert ows2.sent == []  # an unknown stranger: nothing to tell the model


def test_live_is_enforced_unless_fallback_is_explicit(monkeypatch):
    from inkbox_claude import gateway as gw
    monkeypatch.delenv("INKBOX_VOICE_FALLBACK", raising=False)
    monkeypatch.delenv("INKBOX_VOICE_API", raising=False)
    assert gw._voice_api() == "live" and gw._voice_fallback_allowed() is False
    monkeypatch.setenv("INKBOX_VOICE_FALLBACK", "realtime")
    assert gw._voice_fallback_allowed() is True
