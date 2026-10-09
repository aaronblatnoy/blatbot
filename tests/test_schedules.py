"""Schedules use the normal request path and keep their approved boundaries."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from inkbox_claude.gate import cron
from inkbox_claude.gate import executor as gate_executor
from inkbox_claude.gate import manager as gm
from inkbox_claude.gate import hosttools
from inkbox_claude.gate.router import RouterOutput, RouterRequest, ScheduleProposal
from inkbox_claude.gate.store import Store


class FakeRouter:
    def __init__(self):
        self.next = RouterOutput(reply=None, request=None)

    async def plan_query(self, **kwargs):
        return None

    async def route(self, **kwargs):
        return self.next.model_copy(deep=True)

    async def phrase(self, instruction, facts):
        return "Before I do this, I need your yes: remove the named item."


class FakeExecutor:
    def __init__(self, result=None):
        self.result = result or {"ok": True, "summary": "looked it up", "raw": "looked it up",
                                 "tool_calls": ["mcp__x__read"], "wrote": False}
        self.ran = []

    async def run(self, req, **kwargs):
        self.ran.append((req, kwargs))
        return dict(self.result)


def manager(tmp_path, result=None):
    sent = []

    async def send_fn(target, text, mode, meta):
        sent.append((target, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id="conv-aaron", approver_phone="")
    out = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                                store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    out.router = FakeRouter()
    out.executor = FakeExecutor(result)
    out.task_picker = None
    return out, sent


def proposal(kind="once", now=None, **changes):
    now = time.time() if now is None else now
    spec = {"title": "Scheduled check", "prompt": "Check the roster", "kind": kind,
            "timezone": "America/New_York", "report_mode": "always"}
    if kind == "once":
        spec["run_at"] = now + 3600
    elif kind == "recurring":
        spec["cron"] = "0 9 * * 1-5"
    else:
        spec.update(run_at=now, max_runs=20, deadline=now + 7 * 86400)
    spec.update(changes)
    return spec


def test_cron_parser_and_standard_day_rule():
    spec = cron.parse("*/15 9-17 * * 1,3,5")
    assert spec.minute == {0, 15, 30, 45}
    with pytest.raises(cron.CronError):
        cron.parse("0 9 * *")
    tz = ZoneInfo("America/New_York")
    after = datetime(2026, 10, 5, 8, 59, tzinfo=tz).timestamp()
    assert datetime.fromtimestamp(cron.next_fire("0 9 31 * 1", after), tz).day == 5


def test_cron_skips_spring_gap_and_fires_once_in_fall_fold():
    tz = ZoneInfo("America/New_York")
    spring = datetime(2026, 3, 7, 3, 0, tzinfo=tz).timestamp()
    got = datetime.fromtimestamp(cron.next_fire("30 2 * * *", spring), tz)
    assert (got.month, got.day, got.hour, got.minute) == (3, 9, 2, 30)
    fall = datetime(2026, 10, 31, 2, 0, tzinfo=tz).timestamp()
    first = cron.next_fire("30 1 * * *", fall)
    local = datetime.fromtimestamp(first, tz)
    assert (local.month, local.day, local.hour, local.fold) == (11, 1, 1, 0)
    second = datetime.fromtimestamp(cron.next_fire("30 1 * * *", first), tz)
    assert (second.month, second.day, second.fold) == (11, 2, 0)


def test_cron_rejects_impossible_dates_and_skips_nonmatching_days(monkeypatch):
    with pytest.raises(cron.CronError, match="never occurs"):
        cron.parse("0 9 30 2 *")
    calls = 0
    original = cron._matches

    def counted(spec, local):
        nonlocal calls
        calls += 1
        return original(spec, local)

    monkeypatch.setattr(cron, "_matches", counted)
    tz = ZoneInfo("America/New_York")
    after = datetime(2026, 3, 1, 0, 0, tzinfo=tz).timestamp()
    got = datetime.fromtimestamp(cron.next_fire("0 9 29 2 *", after), tz)
    assert (got.year, got.month, got.day, got.hour) == (2028, 2, 29, 9)
    assert calls < 1500


def test_once_and_recurring_mint_normal_requests_and_collapse_missed_runs(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("scheduled")

    async def go():
        once = await m.schedules.propose(owner=True, chat_id="owner", task_id=task["id"],
                                         spec=proposal("once"), scopes=["web"])
        req = m.store.get_request(m.store.get_schedule(once.id).last_request_id)
        await m.schedules.decide_proposal(req, "yes")
        m.store.update_schedule(once.id, next_run=time.time() - 10)
        await m.schedules.tick()
        await asyncio.sleep(0.05)
        run = m.store.get_request(m.store.get_schedule(once.id).last_request_id)
        assert run.schedule_id == once.id and run.mode == "schedule" and run.state == "done"
        assert m.store.get_schedule(once.id).state == "done"

        rec = await m.schedules.propose(owner=True, chat_id="owner", task_id=task["id"],
                                        spec=proposal("recurring", cron="* * * * *"), scopes=["web"])
        await m.schedules.decide_proposal(m.store.get_request(m.store.get_schedule(rec.id).last_request_id), "yes")
        m.store.update_schedule(rec.id, next_run=time.time() - 86400)
        await m.schedules.tick()
        await asyncio.sleep(0.05)
        current = m.store.get_schedule(rec.id)
        assert current.run_count == 1 and current.next_run > time.time()

    asyncio.run(go())


def test_continue_notes_next_wakeup_and_missing_tool_finishes(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("long work")

    async def go():
        schedule = await m.schedules.propose(owner=True, chat_id="owner", task_id=task["id"],
                                             spec=proposal("continue"), scopes=["web"])
        await m.schedules.decide_proposal(m.store.get_request(m.store.get_schedule(schedule.id).last_request_id), "yes")
        m.store.update_schedule(schedule.id, run_count=1)
        req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                     subject="", original_message="work", summary="work", scopes=["web"], prompt="work",
                                     state="running", task_id=task["id"], schedule_id=schedule.id,
                                     schedule_kind="continue")
        await m.schedules.finished(req, {"ok": True, "summary": "progress", "tool_calls": ["x"],
                                         "continue": {"delay_minutes": 5, "note": "Finished the first batch."}})
        current = m.store.get_schedule(schedule.id)
        assert current.state == "active" and current.notes == ["Finished the first batch."]
        assert current.next_run >= time.time() + 299
        await m.schedules.finished(req, {"ok": True, "summary": "all done", "tool_calls": ["x"]})
        assert m.store.get_schedule(schedule.id).state == "done"

    asyncio.run(go())


def test_overlap_is_skipped_and_noted(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("repeat")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="repeat", prompt="check",
                                       kind="recurring", cron="* * * * *", scopes=["web"],
                                       timezone="America/New_York", report_mode="always", next_run=time.time() - 1)
    m.store.update_schedule(schedule.id, state="active")
    running = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                     subject="", original_message="check", summary="repeat", scopes=["web"],
                                     prompt="check", state="running", task_id=task["id"], schedule_id=schedule.id,
                                     schedule_kind="recurring")
    asyncio.run(m.schedules.tick())
    current = m.store.get_schedule(schedule.id)
    assert current.run_count == 0 and f"request #{running.id} was still running" in current.notes[-1]


def test_non_owner_refused_and_edit_returns_to_proposed(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("scheduled")
    with pytest.raises(PermissionError):
        asyncio.run(m.schedules.propose(owner=False, chat_id="other", task_id=task["id"],
                                        spec=proposal(), scopes=["web"]))

    async def go():
        schedule = await m.schedules.propose(owner=True, chat_id="owner", task_id=task["id"],
                                             spec=proposal(), scopes=["web"])
        edited = await m.schedules.edit(schedule.id, owner=True, changes={"prompt": "Check the calendar",
                                                                          "scopes": ["calendar"]})
        assert edited.state == "proposed" and edited.revision == 1 and edited.scopes == ["calendar"]
        assert m.store.get_request(edited.last_request_id).state == "pending"

    asyncio.run(go())


def test_proposal_edit_keeps_prompt_and_replaces_prior_owner_instructions(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("edit")

    async def go():
        schedule = await m.schedules.propose(owner=True, chat_id="owner", task_id=task["id"],
                                             spec=proposal(prompt="Check the calendar"), scopes=["calendar"])
        first = m.store.get_request(m.store.get_schedule(schedule.id).last_request_id)
        await m.schedules.decide_proposal(first, "edit", "Only include work events.")
        current = m.store.get_schedule(schedule.id)
        assert current.prompt == ("Check the calendar\n\n--- AARON'S INSTRUCTIONS ---\n"
                                  "Only include work events.")
        second = m.store.get_request(current.last_request_id)
        await m.schedules.decide_proposal(second, "edit", "Include event locations.")
        assert m.store.get_schedule(schedule.id).prompt == (
            "Check the calendar\n\n--- AARON'S INSTRUCTIONS ---\nInclude event locations.")

    asyncio.run(go())


def test_schedule_actions_enforce_state_transitions(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("states")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="states", prompt="check",
                                       kind="recurring", cron="* * * * *", scopes=["web"],
                                       timezone="America/New_York", report_mode="always", next_run=time.time())

    async def go():
        for action in ("pause", "resume", "run_now"):
            with pytest.raises(ValueError, match="is proposed"):
                await m.schedules.action(schedule.id, action, owner=True)
        m.store.update_schedule(schedule.id, state="active")
        paused = await m.schedules.action(schedule.id, "pause", owner=True)
        assert paused.state == "paused"
        resumed = await m.schedules.action(schedule.id, "resume", owner=True)
        assert resumed.state == "active"
        await m.schedules.action(schedule.id, "run_now", owner=True)
        m.store.update_schedule(schedule.id, state="done")
        with pytest.raises(ValueError, match="is done"):
            await m.schedules.action(schedule.id, "run_now", owner=True)
        deleted = await m.schedules.action(schedule.id, "delete", owner=True)
        assert deleted.state == "done" and m.store.get_schedule(schedule.id) is None

    asyncio.run(go())


def test_expired_schedule_proposal_is_rejected(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("proposal")

    async def go():
        schedule = await m.schedules.propose(owner=True, chat_id="owner", task_id=task["id"],
                                             spec=proposal(), scopes=["web"])
        req = m.store.get_request(m.store.get_schedule(schedule.id).last_request_id)
        m.store._db.execute("UPDATE requests SET created_at=0 WHERE id=?", (req.id,))
        m.store._db.commit()
        await m._expire_pending()
        current = m.store.get_schedule(schedule.id)
        assert current.state == "rejected"
        assert current.last_outcome == "The proposal expired unanswered."

    asyncio.run(go())


def test_plain_language_owner_proposes_but_non_owner_is_refused_in_code(tmp_path):
    m, sent = manager(tmp_path)
    schedule = ScheduleProposal(**proposal("recurring"))
    m.router.next = RouterOutput(
        reply=None, task="new", task_title="Scheduled check", schedule=schedule,
        request=RouterRequest(prompt="ignored", scopes=["web"], summary="Scheduled check"),
    )
    owner_meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(m.get("aaron-route").handle_inbound("check every weekday", "imessage", owner_meta))
    assert len(m.store.schedules()) == 1 and m.store.schedules()[0].state == "proposed"

    other_meta = {"sender": "other@example.com", "to": "other@example.com"}
    asyncio.run(m.get("other-route").handle_inbound("check every weekday", "email", other_meta))
    assert len(m.store.schedules()) == 1
    assert any("Only Aaron can create or change a schedule" in row[1] for row in sent)


def test_plain_language_schedule_uses_strict_minimal_scope_judgment(tmp_path):
    m, _ = manager(tmp_path)
    schedule = ScheduleProposal(**proposal("recurring", prompt="Read my calendar", title="Calendar check"))
    session = m.get("aaron-scope-route")

    async def decide(**kwargs):
        return RouterOutput(reply=None, task="new", task_title="Calendar check", schedule=schedule), None

    class ScopePicker:
        async def judge_scopes_tree(self, **kwargs):
            assert kwargs == {"prompt": "Read my calendar", "summary": "Calendar check", "generous": False}
            return {"scopes": ["calendar"], "probabilities": {"calendar": 0.9, "web": 0.8}}

    session.decide = decide
    m.task_picker = ScopePicker()
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(session.handle_inbound("check my calendar every weekday", "imessage", meta))
    assert m.store.schedules()[0].scopes == ["calendar"]


def test_scheduled_scopes_never_widen_for_owner(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("frozen")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="frozen", prompt="send it",
                                       kind="once", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=time.time(), next_run=time.time())
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule", subject="",
                                 original_message="send it", summary="frozen", scopes=["web"], prompt="send it",
                                 state="running", task_id=task["id"], schedule_id=schedule.id, schedule_kind="once")
    status = {"ok": False, "summary": "I need email_send", "error": "email_send is missing", "tool_calls": []}
    out = asyncio.run(m._grant_what_it_asked_for(req, status, "", None))
    assert out is status and m.store.get_request(req.id).scopes == ["web"]


def test_destructive_scheduled_run_still_parks_for_confirmation(tmp_path):
    confirm = {"ok": False, "summary": "confirmation required", "raw": "", "tool_calls": [],
               "confirm": {"tool": "mcp__x__delete_item", "args": {"id": "item-123"},
                           "about": ["Item item-123"], "kind": "destructive"}}
    m, _ = manager(tmp_path, confirm)
    task = m.store.create_task("remove")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="remove", prompt="remove it",
                                       kind="once", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=time.time(), next_run=time.time())
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule", subject="",
                                 original_message="remove it", summary="remove", scopes=["web"], prompt="remove it",
                                 state="approved", task_id=task["id"], schedule_id=schedule.id, schedule_kind="once")
    asyncio.run(m.execute(req))
    assert m.store.get_request(req.id).state == "pending"


def test_ordinary_destructive_call_is_denied_without_being_parked(tmp_path, monkeypatch):
    m, _ = manager(tmp_path)
    task = m.store.create_task("ordinary")
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="imessage",
                                 subject="", original_message="remove it", summary="remove", scopes=["sjba_site_write"],
                                 prompt="remove it", state="running", task_id=task["id"])
    captured = {}

    class Options:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    async def fake_query(prompt, options):
        denial = await options.can_use_tool(
            "mcp__sjba-admin__sjba_delete_event", {"id": "item-123"}, None)
        captured["message"] = denial.message
        return
        yield

    monkeypatch.setattr(gate_executor, "ClaudeAgentOptions", Options)
    monkeypatch.setattr(gate_executor, "query", fake_query)
    result = asyncio.run(gate_executor.Executor(mcp_server=None, cwd=str(tmp_path)).run(req))
    assert captured["message"] == gate_executor.ORDINARY_DESTRUCTIVE_DENIAL
    assert "Report exactly what you would change (name, time, id) in your status and stop" in captured["message"]
    assert "confirm" not in result


def test_declined_and_expired_held_calls_finish_scheduled_runs(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("held")

    async def go():
        once = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="once", prompt="remove",
                                       kind="once", scopes=["web"], timezone="America/New_York",
                                       report_mode="changed", run_at=time.time(), next_run=None)
        m.store.update_schedule(once.id, state="active", run_count=1)
        held = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                      subject="", original_message="remove", summary="once", scopes=["web"],
                                      prompt="remove", state="pending", task_id=task["id"], schedule_id=once.id,
                                      schedule_kind="once")
        m.store.set_state(held.id, "pending", status={"confirm": {"kind": "destructive"}})
        sent.clear()
        await m.decide_request(held.id, "no")
        current = m.store.get_schedule(once.id)
        assert current.state == "done" and current.next_run is None
        assert current.consecutive_failures == 0 and "declined" in current.last_outcome
        assert len(sent) == 1

        continuing = m.store.create_schedule(
            chat_id="owner", task_id=task["id"], title="continue", prompt="remove", kind="continue",
            scopes=["web"], timezone="America/New_York", report_mode="changed", run_at=time.time(),
            next_run=None, max_runs=20, deadline=time.time() + 86400)
        m.store.update_schedule(continuing.id, state="active", run_count=1)
        expired = m.store.create_request(
            chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule", subject="",
            original_message="remove", summary="continue", scopes=["web"], prompt="remove", state="pending",
            task_id=task["id"], schedule_id=continuing.id, schedule_kind="continue")
        m.store.set_state(expired.id, "pending", status={"confirm": {"kind": "destructive"}})
        m.store._db.execute("UPDATE requests SET created_at=0 WHERE id=?", (expired.id,))
        m.store._db.commit()
        sent.clear()
        await m._expire_pending()
        current = m.store.get_schedule(continuing.id)
        assert current.state == "done" and current.next_run is None
        assert current.consecutive_failures == 0 and "expired" in current.last_outcome
        assert len(sent) == 1

    asyncio.run(go())


def test_recurring_held_call_keeps_its_next_run(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("repeat")
    next_run = time.time() + 600
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="repeat", prompt="remove",
                                       kind="recurring", cron="* * * * *", scopes=["web"],
                                       timezone="America/New_York", report_mode="changed", next_run=next_run)
    m.store.update_schedule(schedule.id, state="active", run_count=1)
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="remove", summary="repeat", scopes=["web"],
                                 prompt="remove", state="pending", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="recurring")
    m.store.set_state(req.id, "pending", status={"confirm": {"kind": "destructive"}})
    asyncio.run(m.decide_request(req.id, "no"))
    current = m.store.get_schedule(schedule.id)
    assert current.state == "active" and current.next_run == next_run


def test_interrupted_scheduled_run_finishes_through_schedule_service(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("restart")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="restart", prompt="work",
                                       kind="once", scopes=["web"], timezone="America/New_York",
                                       report_mode="changed", run_at=time.time(), next_run=None)
    m.store.update_schedule(schedule.id, state="active", run_count=1)
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="work", summary="restart", scopes=["web"],
                                 prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="once")
    asyncio.run(m.recover_interrupted())
    assert m.store.get_request(req.id).state == "failed"
    assert m.store.get_schedule(schedule.id).state == "failed"
    assert any("interrupted by a restart" in row[1] for row in sent)
    assert all("Say the word" not in row[1] for row in sent)


def test_continue_limits_minimum_delay_and_tool_is_not_ordinary(tmp_path):
    with pytest.raises(ValueError):
        hosttools.schedule_continue({"delay_minutes": 4, "note": "too soon"})
    from inkbox_claude.gate.scopes import tools_for
    assert "mcp__host__schedule_continue" not in tools_for(["web"])
    m, _ = manager(tmp_path)
    task = m.store.create_task("bounded")
    now = time.time()
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="bounded", prompt="work",
                                       kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=now, next_run=now, max_runs=1,
                                       deadline=now + 86400)
    m.store.update_schedule(schedule.id, state="active", run_count=1)
    asyncio.run(m.schedules.tick(now + 1))
    assert m.store.get_schedule(schedule.id).state == "done"
    deadline = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="deadline", prompt="work",
                                       kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=now, next_run=now, max_runs=20,
                                       deadline=now - 1)
    m.store.update_schedule(deadline.id, state="active")
    asyncio.run(m.schedules.tick(now))
    assert m.store.get_schedule(deadline.id).state == "done"


def test_changed_reports_only_successful_writes_failures_and_final_continue(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("reports")
    recurring = m.store.create_schedule(
        chat_id="owner", task_id=task["id"], title="reports", prompt="check", kind="recurring",
        cron="* * * * *", scopes=["web"], timezone="America/New_York", report_mode="changed",
        next_run=time.time() + 60)
    m.store.update_schedule(recurring.id, state="active", run_count=1)

    def request(schedule, kind):
        return m.store.create_request(
            chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule", subject="",
            original_message="check", summary=schedule.title, scopes=["web"], prompt="check", state="done",
            task_id=task["id"], schedule_id=schedule.id, schedule_kind=kind)

    async def go():
        read = request(recurring, "recurring")
        await m.schedules.finished(read, {"ok": True, "summary": "read", "tool_calls": ["mcp__x__read"],
                                            "steps": [{"tool": "mcp__x__read", "ok": True}]})
        assert sent == []
        write = request(recurring, "recurring")
        await m.schedules.finished(write, {"ok": True, "summary": "sent", "tool_calls": ["mcp__x__send"],
                                             "steps": [{"tool": "mcp__x__send", "ok": True}]})
        assert len(sent) == 1
        sent.clear()
        failed_write = request(recurring, "recurring")
        await m.schedules.finished(
            failed_write, {"ok": True, "summary": "not sent", "tool_calls": ["mcp__x__send"],
                           "steps": [{"tool": "mcp__x__send", "ok": False}]})
        assert sent == []
        failed = request(recurring, "recurring")
        await m.schedules.finished(failed, {"ok": False, "summary": "failed", "tool_calls": []})
        assert len(sent) == 1

        sent.clear()
        continuing = m.store.create_schedule(
            chat_id="owner", task_id=task["id"], title="finish", prompt="check", kind="continue",
            scopes=["web"], timezone="America/New_York", report_mode="changed", run_at=time.time(),
            next_run=None, max_runs=20, deadline=time.time() + 86400)
        m.store.update_schedule(continuing.id, state="active", run_count=1)
        final = request(continuing, "continue")
        await m.schedules.finished(final, {"ok": True, "summary": "finished", "tool_calls": ["mcp__x__read"]})
        assert m.store.get_schedule(continuing.id).state == "done"
        assert len(sent) == 1

    asyncio.run(go())


def test_three_failures_pause_and_global_pause_stops_firing(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("repeat")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="repeat", prompt="check",
                                       kind="recurring", cron="* * * * *", scopes=["web"],
                                       timezone="America/New_York", report_mode="changed", next_run=time.time())
    m.store.update_schedule(schedule.id, state="active")

    async def fail_three():
        for n in range(3):
            m.store.update_schedule(schedule.id, run_count=n + 1)
            req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                         subject="", original_message="check", summary="repeat", scopes=["web"],
                                         prompt="check", state="failed", task_id=task["id"], schedule_id=schedule.id,
                                         schedule_kind="recurring")
            await m.schedules.finished(req, {"ok": False, "error": f"failure {n + 1}", "tool_calls": []})
    asyncio.run(fail_three())
    assert m.store.get_schedule(schedule.id).state == "paused"
    assert any("three consecutive failed runs" in row[1] for row in sent)

    m.store.update_schedule(schedule.id, state="active", next_run=time.time() - 1, run_count=0)
    m.store.set_setting("GATE_SCHEDULES_PAUSED", "1")
    asyncio.run(m.schedules.tick())
    assert m.store.get_schedule(schedule.id).run_count == 0


def test_existing_database_migrates_in_place(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE requests (
          id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL, sender TEXT, sender_name TEXT,
          mode TEXT, subject TEXT, original_message TEXT NOT NULL, summary TEXT NOT NULL,
          scopes_json TEXT NOT NULL, prompt TEXT NOT NULL, prompt_sha256 TEXT NOT NULL, state TEXT NOT NULL,
          revision INTEGER NOT NULL DEFAULT 0, status_json TEXT, raw_output TEXT,
          created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
    """)
    db.commit()
    db.close()
    store = Store(str(path))
    request_columns = {row[1] for row in store._db.execute("PRAGMA table_info(requests)")}
    tables = {row[0] for row in store._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"schedule_id", "schedule_kind"} <= request_columns
    assert "schedules" in tables


def test_owner_command_needs_the_schedule_word_and_reports_a_refused_action(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("weekly")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="weekly", prompt="check",
                                       kind="recurring", cron="0 9 * * 5", scopes=["web"],
                                       timezone="America/New_York", report_mode="always", next_run=time.time() + 60)
    assert asyncio.run(m.schedules.command(f"delete {schedule.id}", owner=True)) is False
    assert m.store.get_schedule(schedule.id) is not None
    # Still proposed, so resume is refused and said so rather than raised.
    assert asyncio.run(m.schedules.command(f"resume S{schedule.id}", owner=True)) is True
    assert m.store.get_schedule(schedule.id).state == "proposed"
    assert any("only paused or failed schedules can resume" in row[1] for row in sent)
    assert asyncio.run(m.schedules.command(f"delete schedule {schedule.id}", owner=False)) is False
    assert asyncio.run(m.schedules.command(f"delete schedule {schedule.id}", owner=True)) is True
    assert m.store.get_schedule(schedule.id) is None
