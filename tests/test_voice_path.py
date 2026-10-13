"""Voice-call path tests: the 2026-10-13 incident where the owner called, Claude
Code's session limit made every hand-off fail in seconds, and he heard a generic
stranger-fallback line seven times with nothing texted afterwards.

Drives the real GateSession.voice_consult, _run_for_call and phrase_result_for_call
(through manager.GateSessionManager, with the same fakes tests/test_gate.py uses) --
no network, no live executor."""

from __future__ import annotations

import asyncio

import pytest

from inkbox_claude.gate.manager import OWNER_THREAD
from inkbox_claude.gate.router import RouterOutput, RouterRequest
from tests.test_gate import make_manager, phone_manager, voice_meta, AARON_PHONE

STRANGER_PHONE = "+15559990000"

# The exact phrase from the incident (the reset time and wording are made up but the
# shape -- "session limit" plus a "resets <time>" clause -- is the real one).
SESSION_LIMIT_ERROR = (
    "Claude Code returned an error result: You've hit your session limit "
    "· resets 3:10pm (America/New_York) (exit code: 1)"
)

FORBIDDEN_SUBSTRINGS = (
    "Claude Code", "exit code", "session limit ·", "JSON",
    "I have let Aaron know",
)


def _voice_meta(sender=AARON_PHONE):
    return {"call_id": "call-1", "sender": sender, "contact": {"id": "p1", "name": "Caller"}}


def test_owner_on_a_call_hears_the_usage_limit_sentence_not_the_stranger_line(tmp_path):
    """Aaron, trusted on the phone, asks for something; the run fails exactly the way
    it did in the incident. He hears wording.py's own usage-limit sentence -- never
    the generic grounding fallback ("I do not have an answer ready..."), never the raw
    Claude Code error, never a model's rephrasing of either."""
    m, sent = phone_manager(tmp_path, trust=True, active=True)
    m.executor.result = {"ok": False, "error": SESSION_LIMIT_ERROR, "tool_calls": [],
                         "raw": SESSION_LIMIT_ERROR}
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Check the schedule", scopes=["web"], summary="check the schedule"))

    async def go():
        return await m.get("p1").voice_consult("what's on my schedule", _voice_meta())
    out = asyncio.run(go())

    assert "usage limit" in out.lower()
    assert "3:10pm" in out
    assert "try again after that" in out.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in out, f"leaked {bad!r} into the spoken result: {out!r}"
    assert "i do not have an answer ready" not in out.lower()


def test_unverified_caller_gets_no_owner_power_even_while_a_result_is_phrased(tmp_path):
    """A caller who is not Aaron's trusted number never becomes is_approver() True,
    including in the window where a call result is being phrased/spoken -- item 1's
    identity fix must not accidentally widen who counts as the owner."""
    m, sent = phone_manager(tmp_path, trust=True, active=True)
    m.executor.result = {"ok": True, "summary": "Found nothing scheduled.",
                         "tool_calls": ["WebSearch"], "raw": "x"}
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Check the schedule", scopes=["web"], summary="check the schedule"))
    m.router.next_note = RouterOutput(reply="Nothing is on your schedule.", request=None)

    seen_approver = []
    real_route = m.router.route
    async def route(**kw):
        seen_approver.append(kw.get("is_approver"))
        return await real_route(**kw)
    m.router.route = route

    session = m.get("p1")

    async def go():
        return await session.voice_consult("what's on my schedule", _voice_meta(STRANGER_PHONE))
    asyncio.run(go())

    assert seen_approver and all(a is False for a in seen_approver)
    assert session.is_approver() is False


def test_failed_call_after_hangup_is_delivered_once_on_one_channel(tmp_path):
    """After the caller hangs up, a failed request's result reaches exactly one
    channel, exactly once, and never carries the raw error."""
    m, sent = phone_manager(tmp_path, trust=True, active=False)
    m.executor.result = {"ok": False, "error": "connection refused",
                         "tool_calls": ["mcp__coolify__deploy"], "raw": "connection refused"}
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Deploy the app", scopes=["web"], summary="deploy the app"))

    async def go():
        await m.get("p1").voice_consult("deploy the app for me", _voice_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())

    delivered = [row for row in sent if "it could not reach Coolify" in row[1]]
    assert len(delivered) == 1, f"expected exactly one delivery, got {delivered}"
    assert "connection refused" not in delivered[0][1]


def test_all_rows_of_one_call_land_on_one_thread_record(tmp_path):
    """Every row this call writes -- the inbound during voice_consult, and the
    outbound/system rows _run_for_call and phrase_result_for_call add afterwards --
    belong to the same thread record: the owner's (OWNER_THREAD) when the trusted
    caller is Aaron. Before item 1's fix, the inbound landed on OWNER_THREAD while
    the closing outbound row landed on the call's own chat_id ("p1") instead."""
    m, sent = phone_manager(tmp_path, trust=True, active=True)
    m.executor.result = {"ok": True, "summary": "Two events tomorrow.",
                         "tool_calls": ["mcp__google-workspace__get_events"], "raw": "x"}
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="List events", scopes=["stern_calendar"], summary="list tomorrow's events"))
    m.router.next_note = RouterOutput(reply="Two events tomorrow.", request=None)

    async def go():
        return await m.get("p1").voice_consult("what's tomorrow", _voice_meta())
    out = asyncio.run(go())

    owner_rows = m.store.history(OWNER_THREAD, limit=50)
    p1_rows = m.store.history("p1", limit=50)
    assert any(r["kind"] == "inbound" for r in owner_rows)
    assert any(r["kind"] == "outbound" and r["text"] == out for r in owner_rows)
    # Nothing about this call split onto the chat_id's own (separate) record.
    assert not any(r["kind"] == "outbound" and r["text"] == out for r in p1_rows)


def test_plain_title_used_for_a_call_hand_off_transcript_not_the_raw_transcript(tmp_path):
    """A request born from a voice hand-off is summarized by its raw transcript
    (wording._is_handoff_shaped); the failure message names it by plain_title, never
    by dumping that transcript back at the caller."""
    from inkbox_claude.gate import wording
    m, sent = phone_manager(tmp_path, trust=True, active=True)
    handoff_summary = "live phone call with an unidentified caller: deploy the thing, context: none"
    m.executor.result = {"ok": False, "error": "connection refused", "tool_calls": ["mcp__coolify__deploy"],
                         "raw": "connection refused"}
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Deploy the thing", scopes=["web"], summary=handoff_summary))

    async def go():
        return await m.get("p1").voice_consult("deploy the thing", _voice_meta())
    out = asyncio.run(go())
    assert handoff_summary not in out
    assert "blatbot:" not in out.lower()


def test_a_second_usage_limit_failure_within_ten_minutes_is_told_once(tmp_path):
    """Item 7's remainder: two different asks on the same thread both hit the
    session limit within the 10-minute window; the second one is not re-announced
    in full (he was already told), but it still shows up on the ledger."""
    m, sent = phone_manager(tmp_path, trust=True, active=False)
    m.executor.result = {"ok": False, "error": SESSION_LIMIT_ERROR, "tool_calls": [],
                         "raw": SESSION_LIMIT_ERROR}
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Check the schedule", scopes=["web"], summary="check the schedule"))

    async def go():
        await m.get("p1").voice_consult("what's on my schedule", _voice_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    first_told = [row for row in sent if "usage limit" in row[1].lower()]
    assert len(first_told) == 1

    # A second, different ask on the same thread fails the same way shortly after.
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Email the report", scopes=["web"], summary="email the report"))

    async def go2():
        await m.get("p1").voice_consult("send the report to Bob", _voice_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go2())
    told_again = [row for row in sent if "usage limit" in row[1].lower()]
    assert len(told_again) == 1, f"should not be told a second time, got {told_again}"


def test_fragments_of_the_same_failed_ask_do_not_start_a_second_run(tmp_path):
    """Item 4, the incident's actual shape: the realtime voice model hands off
    several fragments of one spoken request a few seconds apart. The first
    fragment's request fails in a few seconds -- too fast for the existing
    running_for_thread fold to catch a later fragment. A further fragment of
    the SAME ask (the router names the same task) never starts a second run:
    the failure is spoken once, then \"Still stuck on that\" at most once more,
    then nothing further."""
    m, sent = phone_manager(tmp_path, trust=True, active=True)
    m.executor.result = {"ok": False, "error": "connection refused",
                         "tool_calls": ["mcp__coolify__deploy"], "raw": "connection refused"}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Deploy the app",
                                 request=RouterRequest(prompt="Deploy the app", scopes=["web"],
                                                       summary="deploy the app"))

    async def first():
        return await m.get("p1").voice_consult("deploy the app for me", _voice_meta())
    out1 = asyncio.run(first())
    assert "it could not reach Coolify" in out1
    task_id = m.store.task_id_for_request(1)
    assert task_id is not None

    # A second fragment of the same ask: the router names the same task.
    m.router.next = RouterOutput(reply="On it.", task=f"T{task_id}",
                                 request=RouterRequest(prompt="deploy it", scopes=["web"], summary="deploy it"))

    async def second():
        return await m.get("p1").voice_consult("go ahead and deploy it", _voice_meta())
    out2 = asyncio.run(second())
    assert "it could not reach Coolify" in out2, "the failure is spoken once per ask"

    async def third():
        return await m.get("p1").voice_consult("did that work", _voice_meta())
    out3 = asyncio.run(third())
    assert out3 == "Still stuck on that. I'll text you."

    async def fourth():
        return await m.get("p1").voice_consult("please try", _voice_meta())
    out4 = asyncio.run(fourth())
    assert out4 == "Got it."

    # Never a second run: only the first fragment's request was ever executed.
    assert m.executor.ran == [1]
    rows = m.store._db.execute("SELECT id FROM requests WHERE chat_id=?", ("p1",)).fetchall()
    assert len(rows) == 1


def test_a_genuinely_new_ask_on_the_same_call_still_creates_a_request(tmp_path):
    """A different ask (a different task) on the same call, right after a failure,
    is not folded -- it creates and runs its own request."""
    m, sent = phone_manager(tmp_path, trust=True, active=True)
    m.executor.result = {"ok": False, "error": "connection refused",
                         "tool_calls": ["mcp__coolify__deploy"], "raw": "connection refused"}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Deploy the app",
                                 request=RouterRequest(prompt="Deploy the app", scopes=["web"],
                                                       summary="deploy the app"))
    asyncio.run(m.get("p1").voice_consult("deploy the app for me", _voice_meta()))

    m.executor.result = {"ok": True, "summary": "Two events tomorrow.",
                         "tool_calls": ["mcp__google-workspace__get_events"], "raw": "x"}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Check tomorrow's calendar",
                                 request=RouterRequest(prompt="list events", scopes=["stern_calendar"],
                                                       summary="list tomorrow's events"))
    m.router.next_note = RouterOutput(reply="Two events tomorrow.", request=None)

    async def go():
        return await m.get("p1").voice_consult("actually, what's on my calendar tomorrow", _voice_meta())
    out = asyncio.run(go())
    assert out == "Two events tomorrow."
    assert m.executor.ran == [1, 2]


def test_still_stuck_line_also_texts_the_failure_exactly_once(tmp_path):
    """Item C9: when the "Still stuck on that" line is spoken (the second fragment
    of a repeated failed ask), the failure message is also delivered once to a text
    channel -- not just spoken -- and it is never sent again, including if the fold
    fires a third time or the request is later handled at hang-up."""
    m, sent = phone_manager(tmp_path, trust=True, active=True)
    m.executor.result = {"ok": False, "error": "connection refused",
                         "tool_calls": ["mcp__coolify__deploy"], "raw": "connection refused"}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Deploy the app",
                                 request=RouterRequest(prompt="Deploy the app", scopes=["web"],
                                                       summary="deploy the app"))

    async def first():
        return await m.get("p1").voice_consult("deploy the app for me", _voice_meta())
    out1 = asyncio.run(first())
    assert "it could not reach Coolify" in out1
    task_id = m.store.task_id_for_request(1)

    m.router.next = RouterOutput(reply="On it.", task=f"T{task_id}",
                                 request=RouterRequest(prompt="deploy it", scopes=["web"], summary="deploy it"))

    async def second():
        return await m.get("p1").voice_consult("go ahead and deploy it", _voice_meta())
    asyncio.run(second())

    async def third():
        return await m.get("p1").voice_consult("did that work", _voice_meta())
    out3 = asyncio.run(third())
    assert out3 == "Still stuck on that. I'll text you."

    texted = [row for row in sent if row[2] in ("sms", "imessage") and "it could not reach Coolify" in row[1]]
    assert len(texted) == 1, f"expected exactly one text, got {texted}"

    # A fourth fragment folds again -- the text is not repeated.
    async def fourth():
        return await m.get("p1").voice_consult("please try again", _voice_meta())
    asyncio.run(fourth())
    texted_again = [row for row in sent if row[2] in ("sms", "imessage") and "it could not reach Coolify" in row[1]]
    assert len(texted_again) == 1, "the failure must never be texted a second time"
