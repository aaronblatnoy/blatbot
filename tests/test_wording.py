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
