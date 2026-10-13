"""Tests for inkbox_claude.gate.wording: turning a failed run's status into
plain words, and assembling the whole result message a person receives."""

from __future__ import annotations

from inkbox_claude.gate import wording


BUFFER_ERROR = (
    "Failed to decode JSON: JSON message exceeded maximum buffer size of "
    "1048576 bytes while reading from the executor's stdout pipe."
)


def test_buffer_size_error_maps_to_plain_clause():
    status = {"ok": False, "error": BUFFER_ERROR, "tool_calls": []}
    assert wording.failure_in_words(status) == "it ran out of room handling something too large"
    assert wording.next_step(status) == 'Say "try again" and I will.'


def test_timeout_maps_to_plain_clause():
    status = {"ok": False, "error": "timed out after 300s", "tool_calls": []}
    assert wording.failure_in_words(status) == "it took too long"


def test_timeout_wins_over_denied_tools_present():
    # A timeout can carry a denied list from steps taken before it ran out of
    # time; the timeout signature still wins.
    status = {"ok": False, "error": "timed out after 300s",
              "denied": [{"tool": "mcp__tamid-admin__tamid_update_member", "reason": "scope"}]}
    assert wording.failure_in_words(status) == "it took too long"


def test_permission_refusal_names_the_scope_label(monkeypatch):
    fake_scopes = {"tamid_write": {"tools": ["mcp__tamid-drive__modify_sheet_values"]}}
    monkeypatch.setattr(wording._scopes, "SCOPES", fake_scopes)
    monkeypatch.setattr(wording.schedprop, "LABELS", {"tamid_write": "change TAMID's sheets"})
    status = {"ok": False, "error": "", "denied": [{"tool": "mcp__tamid-drive__modify_sheet_values", "reason": "scope"}]}
    assert wording.failure_in_words(status) == "it is not allowed to change TAMID's sheets"
    assert wording.next_step(status) == 'Say "let it change TAMID\'s sheets" and I will finish.'


def test_permission_refusal_with_unknown_tool_falls_back():
    status = {"ok": False, "error": "", "denied": [{"tool": "mcp__made-up__frobnicate", "reason": "scope"}]}
    assert wording.failure_in_words(status) == "it is not allowed to do part of this"
    assert wording.next_step(status) == 'Say "let it do part of this" and I will finish.'


def test_held_without_denied_tools_is_also_a_refusal():
    status = {"ok": False, "error": "", "held": True, "denied": []}
    assert wording.failure_in_words(status) == "it is not allowed to do part of this"


def test_network_failure_names_the_system_from_the_last_tool_call():
    status = {"ok": False, "error": "connection refused", "tool_calls": ["mcp__coolify__deploy"]}
    assert wording.failure_in_words(status) == "it could not reach Coolify"


def test_login_failure_names_the_system():
    status = {"ok": False, "error": "401 unauthorized, session expired",
              "tool_calls": ["mcp__quickbooks__qb_run_report"]}
    assert wording.failure_in_words(status) == "it could not reach QuickBooks"


def test_network_failure_with_unidentifiable_tool_is_generic():
    status = {"ok": False, "error": "connection refused", "tool_calls": ["mcp__some-unlisted-server__thing"]}
    assert wording.failure_in_words(status) == "a service it needs did not respond"


def test_network_failure_with_no_tool_calls_is_generic():
    status = {"ok": False, "error": "DNS resolution failed", "tool_calls": []}
    assert wording.failure_in_words(status) == "a service it needs did not respond"


def test_unrecognized_error_is_generic():
    status = {"ok": False, "error": "KeyError: 'foo'", "tool_calls": []}
    assert wording.failure_in_words(status) == "something broke on my side"
    assert wording.next_step(status) == 'Say "try again" and I will.'


def test_failure_message_shape_has_no_raw_text():
    status = {"ok": False, "error": BUFFER_ERROR, "final_status": "", "tool_calls": []}
    msg = wording.failure_message("pull the Q3 numbers", status)
    assert msg == ('I couldn\'t finish pull the Q3 numbers: it ran out of room handling '
                   'something too large. Say "try again" and I will.')
    assert "Traceback" not in msg and "JSON" not in msg and "Exception" not in msg
    assert "bytes" not in msg and "scope" not in msg
    assert not any(c in msg for c in "*_`#")


def test_failure_message_offers_partial_findings_when_present():
    status = {"ok": False, "error": "timed out after 300s",
              "final_status": "Checked 4 of 10 vendors before it ran out of time."}
    msg = wording.failure_message("audit the vendor list", status)
    assert msg.endswith(' I have part of it; say "send what you have".')
    assert "Checked 4 of 10 vendors" not in msg  # the raw narration itself is never echoed


def test_failure_message_has_no_partial_offer_when_nothing_was_found():
    status = {"ok": False, "error": "timed out after 300s", "final_status": ""}
    msg = wording.failure_message("audit the vendor list", status)
    assert "send what you have" not in msg


def test_failure_message_never_says_that_failed():
    for status in (
        {"ok": False, "error": BUFFER_ERROR},
        {"ok": False, "error": "timed out after 10s"},
        {"ok": False, "error": "", "denied": [{"tool": "x", "reason": "scope"}]},
        {"ok": False, "error": "weird traceback garbage"},
    ):
        msg = wording.failure_message("do a thing", status)
        assert "That failed" not in msg


def test_success_message_keeps_a_sentence_as_is():
    assert wording.success_message("I checked the calendar: three events on Tuesday.") == \
        "I checked the calendar: three events on Tuesday."


def test_success_message_adds_done_prefix_to_a_fragment():
    assert wording.success_message("booked") == "Done. booked"


def test_success_message_handles_empty_result():
    assert wording.success_message("") == "Done."
    assert wording.success_message("   ") == "Done."


# -- one channel after a call ends -------------------------------------------
# These reuse the gate test harness (tests/test_gate.py) rather than duplicate it.

import asyncio

from inkbox_claude.gate.manager import OWNER_THREAD
from inkbox_claude.gate.router import RouterOutput, RouterRequest
from tests.test_gate import make_manager, voice_meta, approver_meta, AARON_PHONE


def _phone_manager(tmp_path, *, trust, active):
    m, sent = make_manager(tmp_path)
    m.approver_phone = AARON_PHONE
    m.voice_trust_approver = trust
    m.call_active = lambda chat_id: active
    return m, sent


def test_failed_voice_request_after_hangup_sends_once_and_stores_once(tmp_path):
    m, sent = _phone_manager(tmp_path, trust=True, active=False)
    m.executor.result = {"ok": False, "error": "connection refused", "tool_calls": ["mcp__coolify__deploy"], "raw": "connection refused"}
    # The initial ack ("On it.") and the eventual spoken-result phrasing are two
    # different router.route calls (action=True to decide, action=False to phrase a
    # result); giving them different canned text keeps this test from confusing one
    # for the other.
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Deploy the app", scopes=["web"], summary="deploy the app"))
    m.router.next_note = RouterOutput(reply="Could not reach Coolify.", request=None)

    async def go():
        await m.get("p1").handle_inbound("deploy the app for me", "voice", voice_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())

    result_sends = [row for row in sent if "Could not reach Coolify" in row[1]]
    assert len(result_sends) == 1, f"expected exactly one send of the result, got {result_sends}"
    assert result_sends[0][2] in ("imessage", "sms")
    rows = m.store.history("p1", limit=50) + m.store.history(OWNER_THREAD, limit=50)
    stored = [row for row in rows if row["kind"] == "outbound" and "Could not reach Coolify" in row["text"]]
    assert len(stored) == 1, f"expected exactly one stored outbound row for the result, got {stored}"
    assert "connection refused" not in result_sends[0][1]


# -- messages while a request is still running --------------------------------

def test_fixed_progress_phrases_are_recognized():
    for text in ("are you working on it?", "Are you working on this", "status",
                 "how's it going?", "did you start?", "any update?"):
        assert wording.is_progress_question(text)


def test_question_mark_with_no_new_task_verb_is_a_progress_question():
    assert wording.is_progress_question("is that done yet?")
    assert wording.is_progress_question("almost there?")


def test_question_mark_naming_a_new_task_is_not_a_progress_question():
    assert not wording.is_progress_question("can you also email Bob?")
    assert not wording.is_progress_question("could you book Tuesday at 6 instead?")


def test_plain_statement_is_not_a_progress_question():
    assert not wording.is_progress_question("it's in the Fall 2026 responses sheet")
    assert not wording.is_progress_question("")


def test_progress_status_line_uses_latest_step_when_available():
    started = 1_700_000_000.0
    line = wording.progress_status_line(started, "now working on get events", "pull my schedule")
    assert line.startswith("Yes. Started ")
    assert line.endswith("now working on get events.")


def test_progress_status_line_falls_back_to_summary():
    started = 1_700_000_000.0
    line = wording.progress_status_line(started, None, "pull my schedule")
    assert line.endswith("still working on pull my schedule.")


def test_latest_step_from_tool_calls():
    assert wording.latest_step_from_tool_calls(["mcp__tamid-drive__read_sheet_values"]) == \
        "now working on read sheet values"
    assert wording.latest_step_from_tool_calls([]) is None
    assert wording.latest_step_from_tool_calls(None) is None


def test_progress_reply_during_a_run(tmp_path):
    m, sent = make_manager(tmp_path)
    m.executor = _HoldingExecutor()
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="x",
                                 request=RouterRequest(prompt="pull it", scopes=["web"], summary="pull the report"))

    async def go():
        s = m.get("aaron")
        await s.handle_inbound("pull the report", "imessage", approver_meta())
        await m.executor.started.wait()
        await s.handle_inbound("are you working on it?", "imessage", approver_meta())
        m.executor.release.set()
        await asyncio.sleep(0.05)
    asyncio.run(go())
    outbound = [t for _, t, *_ in sent]
    assert outbound[0] == "On it."
    assert outbound[1].startswith("Yes. Started ") and "pull the report" in outbound[1]


class _HoldingExecutor:
    """A minimal stand-in: blocks on run() until released, like test_gate's
    HoldingExecutor, without importing test_gate's FakeExecutor base (keeps this
    file's harness fixtures self-contained aside from make_manager/voice_meta)."""

    def __init__(self):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.ran = []
        self.result = {"ok": True, "summary": "Found it.", "tool_calls": ["mcp__host__rows_where"], "raw": "x"}

    async def run(self, req, context="", prior_work="", model=None, **kw):
        self.started.set()
        self.ran.append(req.id)
        await self.release.wait()
        return self.result
