"""Executor behaviour owned by the schedules-v2 run status contract: session
handling for scheduled vs. ordinary requests, the scheduled-run prompt's
start-time line, and the RESULT:/NOTE:/STATUS: closing-line contract (parsed
by schedrun.parse_closing, consumed by executor.run's status dict)."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage

from inkbox_claude.gate import executor as gate_executor
from inkbox_claude.gate import schedrun
from inkbox_claude.gate.executor import Executor
from inkbox_claude.gate.store import Request, sha256


def make_request(**over) -> Request:
    base = dict(id=1, chat_id="c1", sender="s", sender_name="", mode="imessage", subject="",
                original_message="m", summary="m", scopes=["web"], prompt="m", state="approved",
                revision=0, status=None, raw_output=None, created_at=0, updated_at=0)
    base.update(over)
    base["prompt_sha256"] = sha256(base["prompt"])
    return Request(**base)


class ScriptedQuery:
    """Stands in for claude_agent_sdk.query: plays back a scripted sequence of
    assistant turns, running each turn's tool calls through the executor's own
    can_use_tool (as the real SDK would) so permission denials behave exactly
    as they would live -- a denied call never gets a ToolResultBlock."""

    def __init__(self, turns: List[Dict[str, Any]], session_id: str = "sess-1", cost_usd: Optional[float] = None):
        self.turns = turns
        self.session_id = session_id
        self.cost_usd = cost_usd
        self.captured_prompt: Optional[str] = None
        self.captured_options: Any = None

    async def __call__(self, prompt, options):
        self.captured_prompt = prompt
        self.captured_options = options
        for turn in self.turns:
            blocks = []
            for text in turn.get("texts_before", []):
                blocks.append(TextBlock(text=text))
            tool_use_blocks = []
            for i, call in enumerate(turn.get("calls", [])):
                tub = ToolUseBlock(id=f"{id(turn)}-{i}", name=call["tool"], input=call.get("args", {}))
                blocks.append(tub)
                tool_use_blocks.append((tub, call))
            if blocks:
                yield AssistantMessage(content=blocks, model="sonnet")
            for tub, call in tool_use_blocks:
                result = await options.can_use_tool(call["tool"], call.get("args", {}), None)
                if result.__class__.__name__ == "PermissionResultAllow":
                    yield UserMessage(content=[ToolResultBlock(tool_use_id=tub.id, content="done",
                                                               is_error=not call.get("ok", True))])
            for text in turn.get("closing_texts", []):
                yield AssistantMessage(content=[TextBlock(text=text)], model="sonnet")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                            num_turns=len(self.turns), session_id=self.session_id,
                            total_cost_usd=self.cost_usd)


def run_with_script(monkeypatch, req, scripted, **kwargs):
    monkeypatch.setattr(gate_executor, "query", scripted)
    ex = Executor(mcp_server=None, cwd="/tmp")
    return ex, asyncio.run(ex.run(req, **kwargs))


def test_scheduled_request_never_resumes_and_uses_a_different_session_each_run(tmp_path, monkeypatch):
    # Executor._sessions is shared process-wide (a pre-existing design choice, not
    # something this package touches), so this checks the one key a scheduled run
    # could possibly use ("sess-a"'s session id) is never stored under it, rather
    # than asserting the whole dict is empty.
    ex = Executor(mcp_server=None, cwd=str(tmp_path))
    before = dict(ex._sessions)
    req = make_request(schedule_id=5, schedule_kind="every")
    first = ScriptedQuery([{"closing_texts": ["RESULT: done.\nSTATUS: OK"]}], session_id="sess-a")
    monkeypatch.setattr(gate_executor, "query", first)
    asyncio.run(ex.run(req))
    assert first.captured_options.resume is None
    assert "sess-a" not in ex._sessions.values()      # never stored for reuse
    assert dict(ex._sessions) == before

    second = ScriptedQuery([{"closing_texts": ["RESULT: done again.\nSTATUS: OK"]}], session_id="sess-b")
    monkeypatch.setattr(gate_executor, "query", second)
    asyncio.run(ex.run(req))
    assert second.captured_options.resume is None
    assert "sess-b" not in ex._sessions.values()


def test_ordinary_request_still_resumes_its_thread_session(tmp_path, monkeypatch):
    ex = Executor(mcp_server=None, cwd=str(tmp_path))
    req = make_request(chat_id="owner-thread")
    first = ScriptedQuery([{"closing_texts": ["All set.\nSTATUS: OK"]}], session_id="sess-x")
    monkeypatch.setattr(gate_executor, "query", first)
    asyncio.run(ex.run(req))
    assert first.captured_options.resume is None
    assert ex._sessions["owner-thread"] == "sess-x"

    second = ScriptedQuery([{"closing_texts": ["Still here.\nSTATUS: OK"]}], session_id="sess-y")
    monkeypatch.setattr(gate_executor, "query", second)
    asyncio.run(ex.run(req))
    assert second.captured_options.resume == "sess-x"
    assert ex._sessions["owner-thread"] == "sess-y"


def test_start_time_line_uses_the_schedules_own_time_zone_for_a_different_utc_day():
    # 11:30 PM Eastern on Oct 12 2026 is 03:30 UTC on Oct 13: a case where the UTC
    # calendar day and the schedule's own local day disagree.
    tz = ZoneInfo("America/New_York")
    local = datetime(2026, 10, 12, 23, 30, tzinfo=tz)
    started_at = local.timestamp()
    line = schedrun.start_time_line(started_at, "America/New_York")
    assert line.startswith("This run started Monday, October 12, 2026, 11:30 PM")
    assert "(America/New_York)" in line
    prompt = schedrun.build_prompt("Check the page.", started_at, "America/New_York")
    assert prompt.startswith(line)
    assert "Check the page." in prompt


def test_build_prompt_includes_previous_note_only_when_supplied():
    started_at = time.time()
    no_note = schedrun.build_prompt("Check it.", started_at, "America/New_York")
    assert "NOTE FROM THE PREVIOUS RUN" not in no_note
    with_note = schedrun.build_prompt("Check it.", started_at, "America/New_York", note="Total was $200.")
    assert "NOTE FROM THE PREVIOUS RUN:\nTotal was $200." in with_note
    assert with_note.index("Check it.") < with_note.index("NOTE FROM THE PREVIOUS RUN")


def test_denied_call_is_reported_in_denied_and_excluded_from_tool_results(tmp_path, monkeypatch):
    req = make_request(scopes=["web"])  # web does not grant the contacts tool below
    scripted = ScriptedQuery([
        {"calls": [{"tool": "mcp__inkbox__inkbox_create_contact", "args": {"name": "x"}}]},
        {"closing_texts": ["RESULT: could not finish.\nSTATUS: FAILED"]},
    ])
    ex, out = run_with_script(monkeypatch, req, scripted)
    assert out["denied"] == [{"tool": "mcp__inkbox__inkbox_create_contact", "reason": "scope"}]
    assert out["tool_results"] == []


def test_successful_write_appears_in_tool_results_with_ok_true(tmp_path, monkeypatch):
    req = make_request(scopes=["contacts"])
    scripted = ScriptedQuery([
        {"calls": [{"tool": "mcp__inkbox__inkbox_create_contact", "args": {"name": "x"}, "ok": True}]},
        {"closing_texts": ["RESULT: added the contact.\nSTATUS: OK"]},
    ])
    ex, out = run_with_script(monkeypatch, req, scripted)
    assert out["tool_results"] == [{"tool": "mcp__inkbox__inkbox_create_contact", "ok": True}]
    assert out["denied"] == []


def test_schedule_run_status_fields_from_closing_lines(tmp_path, monkeypatch):
    req = make_request(schedule_id=9, schedule_kind="every", scopes=["web"])
    closing = "RESULT: Total went from $4,027.39 to $4,922.45.\nNOTE: Total was $4,922.45 as of this run.\nSTATUS: OK"
    scripted = ScriptedQuery([{"closing_texts": [closing]}], cost_usd=0.0123)
    ex, out = run_with_script(monkeypatch, req, scripted, timezone="America/New_York", note="prior note")
    assert out["ok"] is True and out["stop"] is False
    assert out["result_line"] == "Total went from $4,027.39 to $4,922.45."
    assert out["note"] == "Total was $4,922.45 as of this run."
    assert "RESULT:" not in out["final_status"] and "NOTE:" not in out["final_status"] and "STATUS:" not in out["final_status"]
    assert out["cost_usd"] == 0.0123
    assert isinstance(out["started_at"], float) and isinstance(out["finished_at"], float)
    assert out["finished_at"] >= out["started_at"]
    assert "This run started" in scripted.captured_prompt
    assert "NOTE FROM THE PREVIOUS RUN:\nprior note" in scripted.captured_prompt


def test_schedule_run_status_stop(tmp_path, monkeypatch):
    req = make_request(schedule_id=9, schedule_kind="every")
    scripted = ScriptedQuery([{"closing_texts": ["RESULT: Nothing left to watch; ending this schedule.\nSTATUS: STOP"]}])
    ex, out = run_with_script(monkeypatch, req, scripted)
    assert out["stop"] is True and out["ok"] is True


def test_narration_before_last_tool_call_excluded_from_final_status(tmp_path, monkeypatch):
    req = make_request(schedule_id=9, schedule_kind="every", scopes=["web"])
    scripted = ScriptedQuery([
        {"texts_before": ["Let me check the page first."],
         "calls": [{"tool": "WebFetch", "args": {"url": "https://example.test"}, "ok": True}]},
        {"closing_texts": ["Checked the team page.\nRESULT: checked, no change.\nSTATUS: OK"]},
    ])
    ex, out = run_with_script(monkeypatch, req, scripted)
    assert "Let me check the page first." not in out["final_status"]
    assert out["final_status"] == "Checked the team page."


# -- scheduled-run sends: owner-only goes through, anyone else is held ------

def test_scheduled_send_to_owner_own_verified_phone_is_allowed(tmp_path, monkeypatch):
    req = make_request(schedule_id=9, schedule_kind="every", scopes=["sms_send", "email_send"])
    scripted = ScriptedQuery([
        {"calls": [{"tool": "mcp__inkbox__inkbox_send_sms", "args": {"to": "5125550100", "text": "hi"}, "ok": True}]},
        {"closing_texts": ["RESULT: texted Aaron the total.\nSTATUS: OK"]},
    ])
    ex = Executor(mcp_server=None, cwd=str(tmp_path), protected=["5125550100"])
    monkeypatch.setattr(gate_executor, "query", scripted)
    out = asyncio.run(ex.run(req))
    assert out["tool_results"] == [{"tool": "mcp__inkbox__inkbox_send_sms", "ok": True}]
    assert ex.held_send is None
    assert out.get("confirm") is None


def test_scheduled_send_to_third_party_is_held(tmp_path, monkeypatch):
    req = make_request(schedule_id=9, schedule_kind="every", scopes=["sms_send", "email_send"])
    scripted = ScriptedQuery([
        {"calls": [{"tool": "mcp__inkbox__inkbox_send_sms", "args": {"to": "9995551234", "text": "hi"}, "ok": True}]},
        {"closing_texts": ["RESULT: tried to text someone else.\nSTATUS: OK"]},
    ])
    ex = Executor(mcp_server=None, cwd=str(tmp_path), protected=["5125550100"])
    monkeypatch.setattr(gate_executor, "query", scripted)
    out = asyncio.run(ex.run(req))
    assert out["tool_results"] == []
    assert ex.held_send is not None and ex.held_send["to"] == "9995551234"
    assert out["confirm"]["kind"] == "send"


def test_scheduled_send_mixed_owner_and_third_party_is_held(tmp_path, monkeypatch):
    req = make_request(schedule_id=9, schedule_kind="every", scopes=["sms_send", "email_send"])
    scripted = ScriptedQuery([
        {"calls": [{"tool": "mcp__inkbox__inkbox_send_email", "args": {"to": ["5125550100", "9995551234"], "text": "hi"}, "ok": True}]},
        {"closing_texts": ["RESULT: tried to email two people.\nSTATUS: OK"]},
    ])
    ex = Executor(mcp_server=None, cwd=str(tmp_path), protected=["5125550100"])
    monkeypatch.setattr(gate_executor, "query", scripted)
    out = asyncio.run(ex.run(req))
    assert out["tool_results"] == []
    assert ex.held_send is not None


def test_scheduled_send_with_no_recipient_determinable_is_held(tmp_path, monkeypatch):
    req = make_request(schedule_id=9, schedule_kind="every", scopes=["sms_send", "email_send"])
    scripted = ScriptedQuery([
        {"calls": [{"tool": "mcp__inkbox__inkbox_send_sms", "args": {"text": "hi"}, "ok": True}]},
        {"closing_texts": ["RESULT: could not tell who this reaches.\nSTATUS: OK"]},
    ])
    ex = Executor(mcp_server=None, cwd=str(tmp_path), protected=["5125550100"])
    monkeypatch.setattr(gate_executor, "query", scripted)
    out = asyncio.run(ex.run(req))
    assert out["tool_results"] == []
    assert ex.held_send is not None and ex.held_send["to"] == "(no recipient in the call)"


# -- schedrun.parse_closing pure-function coverage -------------------------

def test_parse_closing_basic_result_note_status():
    parsed = schedrun.parse_closing("Did the thing.\nRESULT: Total changed by $5.\nNOTE: watch the next cycle.\nSTATUS: OK")
    assert parsed["result_line"] == "Total changed by $5."
    assert parsed["note"] == "watch the next cycle."
    assert parsed["status"] == "OK"
    assert parsed["stop"] is False
    assert parsed["final_status"] == "Did the thing."


def test_parse_closing_stop_status():
    parsed = schedrun.parse_closing("RESULT: that is everything.\nSTATUS: STOP")
    assert parsed["status"] == "STOP" and parsed["stop"] is True


def test_parse_closing_missing_result_line():
    parsed = schedrun.parse_closing("Ran fine.\nSTATUS: OK")
    assert parsed["result_line"] == ""
    assert parsed["final_status"] == "Ran fine."
    assert parsed["status"] == "OK"


def test_parse_closing_multiline_note():
    text = "RESULT: Done.\nNOTE: Remember:\nthe total was 42\nand donors were 5.\nSTATUS: OK"
    parsed = schedrun.parse_closing(text)
    assert parsed["note"] == "Remember:\nthe total was 42\nand donors were 5."
    assert parsed["result_line"] == "Done."


def test_parse_closing_keeps_a_very_long_closing_block_whole():
    huge = "x" * 5000
    text = f"RESULT: {huge}\nSTATUS: OK"
    parsed = schedrun.parse_closing(text)
    assert parsed["result_line"] == huge
    assert len(parsed["result_line"]) == 5000


def test_parse_closing_changed_yes():
    parsed = schedrun.parse_closing("RESULT: total moved.\nSTATUS: OK\nCHANGED: yes")
    assert parsed["changed"] is True
    assert "CHANGED" not in parsed["final_status"]


def test_parse_closing_changed_no():
    parsed = schedrun.parse_closing("RESULT: nothing new.\nSTATUS: OK\nCHANGED: no")
    assert parsed["changed"] is False


def test_parse_closing_changed_absent_is_none():
    parsed = schedrun.parse_closing("RESULT: fine.\nSTATUS: OK")
    assert parsed["changed"] is None


# ---------------------------------------------------------------------------
# _norm_handle / _is_owner_handle (item 8): only a phone-shaped value is
# reduced to its last 10 digits; anything else is compared whole.
# ---------------------------------------------------------------------------

def test_norm_handle_conversation_id_matches_only_itself():
    protected = ["conv-aaron"]
    assert gate_executor._is_owner_handle("conv-aaron", protected)
    assert not gate_executor._is_owner_handle("conv-other", protected)


def test_norm_handle_phone_with_country_code_matches():
    protected = ["+1 (555) 000-1234"]
    assert gate_executor._is_owner_handle("5550001234", protected)
    assert gate_executor._is_owner_handle("+15550001234", protected)


def test_norm_handle_third_party_sharing_digits_does_not_match():
    # A conversation id that happens to contain the same digits as a real
    # phone number must not collide with it: it is not phone-shaped, so it
    # is compared whole, never reduced to its last 10 digits.
    protected = ["5550001234"]
    assert not gate_executor._is_owner_handle("conv-5550001234", protected)


def test_norm_handle_telegram_prefix_stripped():
    protected = ["telegram:998877"]
    assert gate_executor._is_owner_handle("998877", protected)
    assert gate_executor._is_owner_handle("Telegram:998877", protected)

def test_parse_closing_blocked_and_needs_reach_the_status_fields():
    from inkbox_claude.gate import schedrun
    parsed = schedrun.parse_closing(
        "Looked everywhere.\nRESULT: The team chat is not in any thread I can read.\n"
        "NEEDS: a way to read Telegram groups\nSTATUS: BLOCKED")
    assert parsed["blocked"] is True
    assert parsed["needs"] == "a way to read Telegram groups"
    assert parsed["status"] == "BLOCKED"
