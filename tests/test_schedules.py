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
    assert current.run_count == 0 and current.notes == []
    assert f"request #{running.id} was still running" in current.last_outcome


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


def test_schedule_prompt_naming_a_tool_grants_its_scope_without_asking_jev(tmp_path):
    # Reproduces a real failure: a schedule's prompt spelled out inkbox_lookup_contact
    # and inkbox_create_contact by name, but the probability judgment run against the
    # schedule's bare prompt text scored "contacts" too low to pass the write-leaf bar,
    # so the schedule was approved and persisted without it. A prompt that names its own
    # tools is unambiguous: honor that literally and never ask Jev for those scopes.
    m, _ = manager(tmp_path)
    prompt = ("Check the page. If it changed, look it up with inkbox_lookup_contact; if "
              "missing, create it with inkbox_create_contact. Then call schedule_continue.")
    schedule = ScheduleProposal(title="Fundraising monitor", prompt=prompt, kind="continue")
    session = m.get("aaron-contacts-route")

    async def decide(**kwargs):
        return RouterOutput(reply=None, task="new", task_title="Fundraising monitor", schedule=schedule), None

    class ScopePicker:
        async def judge_scopes_tree(self, **kwargs):
            raise AssertionError("Jev must not be consulted when the prompt names its own tools")

    session.decide = decide
    m.task_picker = ScopePicker()
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(session.handle_inbound("set up a fundraising monitor that alerts me", "imessage", meta))
    scopes = m.store.schedules()[0].scopes
    assert "contacts" in scopes
    allowed = gm.SCOPES["contacts"]["tools"]
    assert "mcp__inkbox__inkbox_lookup_contact" in allowed
    assert "mcp__inkbox__inkbox_create_contact" in allowed


def test_schedule_naming_an_unknown_scope_is_refused(tmp_path):
    m, sent = manager(tmp_path)
    schedule = ScheduleProposal(title="Bad scope", prompt="Read my calendar", kind="once",
                                run_at="2026-11-01T09:00:00", scopes=["not_a_real_scope"])
    session = m.get("aaron-bad-scope-route")

    async def decide(**kwargs):
        return RouterOutput(reply=None, task="new", task_title="Bad scope", schedule=schedule), None

    session.decide = decide
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(session.handle_inbound("set this up", "imessage", meta))
    assert m.store.schedules() == []
    assert any("not_a_real_scope" in row[1] for row in sent)


def test_schedule_scopes_named_by_the_owner_are_honored_without_asking_jev(tmp_path):
    m, _ = manager(tmp_path)
    schedule = ScheduleProposal(title="Named scopes", prompt="Watch the page", kind="once",
                                run_at="2026-11-01T09:00:00", scopes=["contacts", "web"])
    session = m.get("aaron-named-scope-route")

    async def decide(**kwargs):
        return RouterOutput(reply=None, task="new", task_title="Named scopes", schedule=schedule), None

    class ScopePicker:
        async def judge_scopes_tree(self, **kwargs):
            raise AssertionError("Jev must not be consulted when the owner named scopes")

    session.decide = decide
    m.task_picker = ScopePicker()
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(session.handle_inbound("watch the page, it can touch contacts", "imessage", meta))
    assert set(m.store.schedules()[0].scopes) == {"contacts", "web"}



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


def test_an_ordinary_word_in_a_schedule_prompt_does_not_name_a_tool():
    from inkbox_claude.gate.manager import _schedule_prompt_tool_scopes
    from inkbox_claude.gate.schedules import _prompt_tool_mismatch
    prompt = "Check the service logs and the database, then deploy nothing and report on the system."
    assert _schedule_prompt_tool_scopes(prompt) == []
    assert _prompt_tool_mismatch(prompt, ["web"]) == []
    assert "contacts" in _schedule_prompt_tool_scopes("Then call inkbox_create_contact for the tracker.")


# -- schedule-fix regression tests -----------------------------------------


def test_pause_during_inflight_continue_run_keeps_note_and_outcome_no_new_next_run(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("pause mid continue")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="pause mid continue",
                                       prompt="work", kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=time.time(), next_run=time.time(),
                                       max_runs=20, deadline=time.time() + 86400)
    m.store.update_schedule(schedule.id, state="active", run_count=1, next_run=None)
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="work", summary="pause mid continue", scopes=["web"],
                                 prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="continue")
    m.store.set_state(req.id, "running", status={"schedule_revision": 0})

    async def go():
        paused = await m.schedules.action(schedule.id, "pause", owner=True)
        assert paused.state == "paused"
        sent.clear()
        handled = await m.schedules.finished(
            req, {"ok": True, "summary": "progress note", "tool_calls": ["x"],
                  "continue": {"delay_minutes": 5, "note": "should not be added"}})
        assert handled is True
        current = m.store.get_schedule(schedule.id)
        assert current.state == "paused" and current.next_run is None
        assert current.notes == []
        assert current.last_outcome == "progress note"
        assert sent == []  # a run landing on a schedule the owner already moved on gets no owner notice
        memory = m.store.task_memory_for_task(schedule.task_id)
        assert "had moved on" in memory

    asyncio.run(go())


def test_edit_during_inflight_continue_run_requires_reapproval_and_ignores_stale_revision_signal(tmp_path):
    m, sent = manager(tmp_path, result={
        "ok": True, "summary": "progress", "raw": "progress", "tool_calls": ["x"],
        "continue": {"delay_minutes": 5, "note": "should not land"},
    })
    task = m.store.create_task("continue edit")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="continue edit", prompt="work",
                                       kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=time.time(), next_run=time.time(),
                                       max_runs=20, deadline=time.time() + 86400)
    m.store.update_schedule(schedule.id, state="active", run_count=1)
    started_revision = schedule.revision
    # Mirrors _fire exactly: create_request() returns an object without the status
    # set_state() is about to write, and that same stale object is what gets executed.
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="work", summary="continue edit", scopes=["web"],
                                 prompt="work", state="approved", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="continue")
    m.store.set_state(req.id, "approved", status={"schedule_revision": started_revision})

    async def go():
        edited = await m.schedules.edit(schedule.id, owner=True, changes={"prompt": "work harder"})
        assert edited.state == "proposed" and edited.revision == started_revision + 1
        pending = m.store.get_request(edited.last_request_id)
        await m.schedules.decide_proposal(pending, "yes")
        active_again = m.store.get_schedule(schedule.id)
        assert active_again.state == "active" and active_again.revision == started_revision + 1
        next_run_before = active_again.next_run

        sent.clear()
        await m.execute(req)  # the stale (revision 0) run finally reports in
        final = m.store.get_schedule(schedule.id)
        assert final.state == "active" and final.revision == started_revision + 1
        assert final.notes == []  # the continue note from the stale run never landed
        assert final.next_run == next_run_before  # not rescheduled by the stale signal
        assert m.store.get_request(req.id).state == "done"

    asyncio.run(go())


def test_delete_during_inflight_run_sends_no_notice_and_nothing_fires_again(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("delete mid run")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="delete mid run", prompt="work",
                                       kind="recurring", cron="* * * * *", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", next_run=time.time())
    m.store.update_schedule(schedule.id, state="active")
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="work", summary="delete mid run", scopes=["web"],
                                 prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="recurring")

    async def go():
        await m.schedules.action(schedule.id, "delete", owner=True)
        assert m.store.get_schedule(schedule.id) is None
        sent.clear()
        handled = await m.schedules.finished(req, {"ok": False, "error": "boom", "tool_calls": []})
        assert handled is True
        assert sent == []
        await m.schedules.tick()
        assert sent == []  # nothing left to fire

    asyncio.run(go())


def test_owner_message_during_schedule_run_is_not_held_for_it(tmp_path):
    # store.running_for_thread must ignore a schedule's own running request on the
    # owner's thread, so an owner message arriving meanwhile is decided as its own
    # turn rather than silently absorbed into the schedule's run.
    m, sent = manager(tmp_path)
    task = m.store.create_task("mid-run owner message")
    schedule = m.store.create_schedule(chat_id="aaron-mid-run", task_id=task["id"], title="mid-run", prompt="work",
                                       kind="recurring", cron="* * * * *", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", next_run=time.time())
    m.store.update_schedule(schedule.id, state="active")
    running = m.store.create_request(chat_id="aaron-mid-run", sender="Aaron", sender_name="Aaron", mode="schedule",
                                     subject="", original_message="work", summary="mid-run", scopes=["web"],
                                     prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                     schedule_kind="recurring")
    assert m.store.running_for_thread("aaron-mid-run") is None

    session = m.get("aaron-mid-run")

    async def decide(**kwargs):
        out = RouterOutput(reply=None, request=RouterRequest(
            prompt="look something up", scopes=["web"], summary="owner side request"))
        return out, m.store.create_task("owner side request")

    session.decide = decide
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(session.handle_inbound("while that's running, also check this", "imessage", meta))
    created = [r for r in m.store.requests_in_state("approved") + m.store.requests_in_state("done")
              if r.chat_id == "aaron-mid-run" and r.id != running.id]
    assert len(created) == 1
    assert created[0].schedule_id is None


def test_owner_message_during_failed_scheduled_run_is_handled_exactly_once(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("fails with meanwhile message")
    schedule = m.store.create_schedule(chat_id="aaron-fail-mid-run", task_id=task["id"], title="fails mid run",
                                       prompt="work", kind="recurring", cron="* * * * *", scopes=["web"],
                                       timezone="America/New_York", report_mode="always", next_run=time.time())
    m.store.update_schedule(schedule.id, state="active")
    req = m.store.create_request(chat_id="aaron-fail-mid-run", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="work", summary="fails mid run", scopes=["web"],
                                 prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="recurring")

    session = m.get("aaron-fail-mid-run")
    calls = {"n": 0}

    async def decide(**kwargs):
        calls["n"] += 1
        out = RouterOutput(reply=None, request=RouterRequest(
            prompt="look something up", scopes=["web"], summary="owner side request"))
        return out, m.store.create_task("owner side request")

    session.decide = decide
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    # The message arrives while the schedule is "running" -- it must get its own
    # turn right away, since running_for_thread no longer holds it for the schedule.
    asyncio.run(session.handle_inbound("while that's running, also check this", "imessage", meta))
    assert calls["n"] == 1
    created_after_arrival = [r for r in m.store.requests_in_state("approved") + m.store.requests_in_state("done")
                            if r.chat_id == "aaron-fail-mid-run" and r.id != req.id]
    assert len(created_after_arrival) == 1

    # The scheduled run now ends FAILED. It must not re-decide the same message again.
    m.executor = FakeExecutor({"ok": False, "summary": "it failed", "error": "boom", "raw": "", "tool_calls": []})
    asyncio.run(m.execute(req))
    assert calls["n"] == 1  # still exactly one decision for that inbound message
    still_only_one = [r for r in m.store.requests_in_state("approved") + m.store.requests_in_state("done")
                      if r.chat_id == "aaron-fail-mid-run" and r.id != req.id]
    assert len(still_only_one) == 1


def test_executor_keys_schedule_session_by_schedule_not_chat_id(tmp_path, monkeypatch):
    from claude_agent_sdk import ResultMessage

    class Options:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    async def fake_query(prompt, options):
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                            num_turns=1, session_id="sess-abc")

    monkeypatch.setattr(gate_executor, "ClaudeAgentOptions", Options)
    monkeypatch.setattr(gate_executor, "query", fake_query)

    store = Store(str(tmp_path / "exec.db"))
    task = store.create_task("session keys")
    sched_req = store.create_request(chat_id="owner-chat", sender="Aaron", sender_name="Aaron", mode="schedule",
                                     subject="", original_message="go", summary="go", scopes=["web"], prompt="go",
                                     state="running", task_id=task["id"], schedule_id=9, schedule_kind="once")
    ordinary_req = store.create_request(chat_id="owner-chat", sender="Aaron", sender_name="Aaron", mode="imessage",
                                        subject="", original_message="go", summary="go", scopes=["web"], prompt="go",
                                        state="running", task_id=task["id"])
    ex = gate_executor.Executor(mcp_server=None, cwd=str(tmp_path))
    asyncio.run(ex.run(sched_req))
    asyncio.run(ex.run(ordinary_req))
    assert ex._sessions.get("schedule:9") == "sess-abc"
    assert ex._sessions.get("owner-chat") == "sess-abc"
    assert "schedule:9" not in ("owner-chat",)


def test_next_run_prompt_carries_notes_without_operational_lines_and_run_now_blocked(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("notes in prompt")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="notes in prompt", prompt="work",
                                       kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=time.time(), next_run=time.time(),
                                       max_runs=20, deadline=time.time() + 86400)
    m.store.update_schedule(schedule.id, state="active", run_count=1, notes=["Finished batch one."])

    async def go():
        req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                     subject="", original_message="work", summary="notes in prompt", scopes=["web"],
                                     prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                     schedule_kind="continue")
        # As in the real flow, execute() marks the request done/failed before handing
        # off to schedules.finished() -- a request still "running" would otherwise look
        # like an overlapping run and block the next firing.
        m.store.set_state(req.id, "done")
        await m.schedules.finished(req, {"ok": True, "summary": "progress", "tool_calls": ["x"],
                                         "continue": {"delay_minutes": 5, "note": "Finished batch two."}})
        await m.schedules.tick(time.time() + 301)
        latest = m.store.get_request(m.store.get_schedule(schedule.id).last_request_id)
        assert "Finished batch one." in latest.prompt and "Finished batch two." in latest.prompt
        assert "Skipped a firing" not in latest.prompt

        # run_now while a run is in flight does not change next_run and says so clearly.
        before = m.store.get_schedule(schedule.id).next_run
        m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                               subject="", original_message="work", summary="notes in prompt",
                               scopes=["web"], prompt="work", state="running", task_id=task["id"],
                               schedule_id=schedule.id, schedule_kind="continue")
        with pytest.raises(ValueError, match="already has a run in flight"):
            await m.schedules.action(schedule.id, "run_now", owner=True)
        assert m.store.get_schedule(schedule.id).next_run == before

    asyncio.run(go())


def test_restart_recovers_approved_and_running_schedule_requests_then_fires_once(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("double restart")
    approved_sched = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="approved-crash",
                                             prompt="work", kind="once", scopes=["web"],
                                             timezone="America/New_York", report_mode="changed",
                                             run_at=time.time(), next_run=None)
    m.store.update_schedule(approved_sched.id, state="active", run_count=1)
    approved_req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                          subject="", original_message="work", summary="approved-crash",
                                          scopes=["web"], prompt="work", state="approved", task_id=task["id"],
                                          schedule_id=approved_sched.id, schedule_kind="once")

    running_sched = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="running-crash",
                                            prompt="work", kind="recurring", cron="* * * * *", scopes=["web"],
                                            timezone="America/New_York", report_mode="changed",
                                            next_run=time.time() - 120)
    m.store.update_schedule(running_sched.id, state="active", run_count=1)
    running_req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                         subject="", original_message="work", summary="running-crash",
                                         scopes=["web"], prompt="work", state="running", task_id=task["id"],
                                         schedule_id=running_sched.id, schedule_kind="recurring")

    sent.clear()
    asyncio.run(m.recover_interrupted())
    assert m.store.get_request(approved_req.id).state == "failed"
    assert m.store.get_request(running_req.id).state == "failed"
    # "once" ends failed outright; "recurring" just records the failed run and stays
    # active -- it was already rescheduled by _fire before the crash.
    assert m.store.get_schedule(approved_sched.id).state == "failed"
    recovered_recurring = m.store.get_schedule(running_sched.id)
    assert recovered_recurring.state == "active" and recovered_recurring.consecutive_failures == 1
    notices = [row[1] for row in sent if "interrupted by a restart" in row[1]]
    assert len(notices) == 2  # exactly one owner notice per recovered request

    m.store.update_schedule(running_sched.id, next_run=time.time() - 60)
    asyncio.run(m.schedules.tick())
    asyncio.run(asyncio.sleep(0.05))
    first = m.store.get_schedule(running_sched.id)
    assert first.run_count == 2 and first.next_run > time.time()

    # Second tick, same moment: must not fire again.
    before_next_run = first.next_run
    asyncio.run(m.schedules.tick(time.time()))
    asyncio.run(asyncio.sleep(0.05))
    second = m.store.get_schedule(running_sched.id)
    assert second.run_count == 2 and second.next_run == before_next_run


def test_recover_interrupted_restricts_approved_recovery_to_schedule_requests(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("ordinary approved")
    ordinary_req = m.store.create_request(chat_id="other", sender="x", sender_name="x", mode="imessage",
                                          subject="", original_message="do it", summary="ordinary", scopes=["web"],
                                          prompt="do it", state="approved", task_id=task["id"])
    sched_task = m.store.create_task("schedule approved")
    schedule = m.store.create_schedule(chat_id="owner", task_id=sched_task["id"], title="sched", prompt="work",
                                       kind="once", scopes=["web"], timezone="America/New_York",
                                       report_mode="changed", run_at=time.time(), next_run=None)
    m.store.update_schedule(schedule.id, state="active", run_count=1)
    sched_req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                       subject="", original_message="work", summary="sched", scopes=["web"],
                                       prompt="work", state="approved", task_id=sched_task["id"],
                                       schedule_id=schedule.id, schedule_kind="once")
    sent.clear()
    asyncio.run(m.recover_interrupted())
    assert m.store.get_request(ordinary_req.id).state == "approved"
    assert m.store.get_request(sched_req.id).state == "failed"
    assert m.store.get_schedule(schedule.id).state == "failed"


def test_request_running_for_schedule_ignores_stale_run_and_does_not_block_firing(tmp_path):
    m, _ = manager(tmp_path)
    task = m.store.create_task("stale blocker")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="stale blocker", prompt="work",
                                       kind="recurring", cron="* * * * *", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", next_run=time.time() - 1)
    m.store.update_schedule(schedule.id, state="active")
    stuck = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                   subject="", original_message="work", summary="stale blocker", scopes=["web"],
                                   prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                   schedule_kind="recurring")
    old = time.time() - (m.store.RUN_STALE_S + 60)
    m.store._db.execute("UPDATE requests SET updated_at=? WHERE id=?", (old, stuck.id))
    m.store._db.commit()
    assert m.store.request_running_for_schedule(schedule.id) is None
    assert m.store.get_request(stuck.id).state == "failed"

    asyncio.run(m.schedules.tick())
    asyncio.run(asyncio.sleep(0.05))
    assert m.store.get_schedule(schedule.id).run_count == 1


def test_continue_delay_past_deadline_ends_with_deadline_outcome(tmp_path):
    from inkbox_claude.gate.schedules import _shown
    m, sent = manager(tmp_path)
    task = m.store.create_task("deadline")
    now = time.time()
    deadline = now + 600
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="deadline", prompt="work",
                                       kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=now, next_run=now, max_runs=20,
                                       deadline=deadline)
    m.store.update_schedule(schedule.id, state="active", run_count=1)
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="work", summary="deadline", scopes=["web"],
                                 prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="continue")
    sent.clear()
    asyncio.run(m.schedules.finished(req, {"ok": True, "summary": "progress", "tool_calls": ["x"],
                                           "continue": {"delay_minutes": 60, "note": "would be past deadline"}}))
    current = m.store.get_schedule(schedule.id)
    assert current.state == "done" and current.next_run is None
    assert _shown(deadline, schedule.timezone) in current.last_outcome


def test_no_continue_call_ends_with_final_notice(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("no continue")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="no continue", prompt="work",
                                       kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=time.time(), next_run=time.time(),
                                       max_runs=20, deadline=time.time() + 86400)
    m.store.update_schedule(schedule.id, state="active", run_count=1)
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="work", summary="no continue", scopes=["web"],
                                 prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="continue")
    sent.clear()
    asyncio.run(m.schedules.finished(req, {"ok": True, "summary": "all done", "tool_calls": ["x"]}))
    assert m.store.get_schedule(schedule.id).state == "done"
    assert len(sent) == 1
    assert "No further runs" in sent[0][1]


def test_two_continue_calls_in_one_run_last_wins(tmp_path, monkeypatch):
    # Claude Code executor path: the in-process host tool clears+overwrites continue_state
    # on every call, so only the LAST call's delay/note survive into the run's status.
    import claude_agent_sdk
    from inkbox_claude.gate import hosttools as hosttools_mod
    original = claude_agent_sdk.create_sdk_mcp_server
    captured = {}

    def fake_create(*a, **kw):
        captured["tools"] = kw.get("tools")
        return original(*a, **kw)

    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", fake_create)
    state = {}
    hosttools_mod.sdk_server(state)
    handler = next(t.handler for t in captured["tools"] if t.name == "schedule_continue")
    asyncio.run(handler({"delay_minutes": 5, "note": "first"}))
    asyncio.run(handler({"delay_minutes": 10, "note": "second and final"}))
    assert state == {"delay_minutes": 10, "note": "second and final"}

    # Jev path: reversed(steps) picks the last matching schedule_continue call.
    from inkbox_claude.gate.jevagent import JevAgent

    class FakeJudge:
        calls = 0

    class FakeProse:
        calls = 0

    steps = [
        {"tool": "mcp__host__schedule_continue", "args": {"delay_minutes": 5, "note": "first"},
         "ok": True, "result": "ok"},
        {"tool": "mcp__x__read", "args": {}, "ok": True, "result": "ok"},
        {"tool": "mcp__host__schedule_continue", "args": {"delay_minutes": 10, "note": "second and final"},
         "ok": True, "result": "ok"},
    ]
    status = JevAgent._status(True, "", steps, FakeJudge(), FakeProse(), time.time())
    assert status["continue"] == {"delay_minutes": 10, "note": "second and final"}


def test_failed_run_with_continue_keeps_going_with_countdown_then_third_failure_pauses_once(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("failing continue")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="failing continue", prompt="work",
                                       kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=time.time(), next_run=time.time(),
                                       max_runs=20, deadline=time.time() + 86400)
    m.store.update_schedule(schedule.id, state="active", run_count=1)

    async def go():
        for n in range(1, 4):
            req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                         subject="", original_message="work", summary="failing continue",
                                         scopes=["web"], prompt="work", state="running", task_id=task["id"],
                                         schedule_id=schedule.id, schedule_kind="continue")
            sent.clear()
            await m.schedules.finished(req, {
                "ok": False, "error": f"failure {n}", "tool_calls": ["x"],
                "continue": {"delay_minutes": 10, "note": f"note {n}"},
            })
            current = m.store.get_schedule(schedule.id)
            if n < 3:
                assert current.state == "active" and current.consecutive_failures == n
                assert current.notes[-1] == f"note {n}"
                assert len(sent) == 1
                assert f"{3 - n} more failure(s) pause this schedule" in sent[0][1]
            else:
                assert current.state == "paused" and current.next_run is None
                assert any("three consecutive failed runs" in row[1] for row in sent)

        # No further notice sent while paused.
        sent.clear()
        stale_req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                           subject="", original_message="work", summary="failing continue",
                                           scopes=["web"], prompt="work", state="running", task_id=task["id"],
                                           schedule_id=schedule.id, schedule_kind="continue")
        await m.schedules.finished(stale_req, {"ok": False, "error": "late", "tool_calls": [],
                                                "continue": {"delay_minutes": 10, "note": "too late"}})
        assert sent == []
        assert m.store.get_schedule(schedule.id).state == "paused"

    asyncio.run(go())


def test_run_notice_uses_final_status_lists_denied_tools_and_next_run_countdown(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("final status notice")
    schedule = m.store.create_schedule(chat_id="owner", task_id=task["id"], title="final status notice",
                                       prompt="work", kind="continue", scopes=["web"], timezone="America/New_York",
                                       report_mode="always", run_at=time.time(), next_run=time.time(),
                                       max_runs=20, deadline=time.time() + 86400)
    m.store.update_schedule(schedule.id, state="active", run_count=1, consecutive_failures=1)
    req = m.store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="schedule",
                                 subject="", original_message="work", summary="final status notice", scopes=["web"],
                                 prompt="work", state="running", task_id=task["id"], schedule_id=schedule.id,
                                 schedule_kind="continue")
    closing = ("Looked at the roster, cross-checked two tables, and tried to email the weekly summary to the "
              "list, but the send tool was refused by scope. Will retry the send next run after Aaron widens "
              "the schedule's scopes, or will give up after the run limit. " * 3).strip()
    assert len(closing) > 400
    narration_1 = "First I searched the roster for pending signups and found eleven."
    narration_2 = "Then I attempted the summary email, which the scope refused outright."

    sent.clear()
    handled = asyncio.run(m.schedules.finished(req, {
        "ok": False, "final_status": closing, "summary": f"{narration_1}\n{narration_2}\n{closing}",
        "tool_calls": ["mcp__inkbox__inkbox_send_email"],
        "steps": [{"tool": "mcp__inkbox__inkbox_send_email", "args": {}, "ok": False, "denied": "scope"}],
        "continue": {"delay_minutes": 10, "note": "will retry the email next run"},
    }))
    assert handled is True
    assert len(sent) == 1
    notice = sent[0][1]
    assert closing in notice
    assert narration_1 not in notice and narration_2 not in notice
    assert "mcp__inkbox__inkbox_send_email refused: outside scope (granted: web)" in notice
    assert "Next run:" in notice
    assert "1 more failure(s) pause this schedule" in notice
    current = m.store.get_schedule(schedule.id)
    assert current.state == "active" and current.consecutive_failures == 2


def test_activation_notice_has_title_first_run_time_and_scopes(tmp_path):
    m, sent = manager(tmp_path)
    task = m.store.create_task("activate")

    async def go():
        schedule = await m.schedules.propose(owner=True, chat_id="owner", task_id=task["id"],
                                             spec=proposal("recurring", cron="0 9 * * 1-5"),
                                             scopes=["calendar", "web"])
        req = m.store.get_request(m.store.get_schedule(schedule.id).last_request_id)
        sent.clear()
        await m.schedules.decide_proposal(req, "yes")
        assert len(sent) == 1
        notice = sent[0][1]
        activated = m.store.get_schedule(schedule.id)
        assert activated.title in notice
        assert "calendar" in notice and "web" in notice
        assert "First run" in notice

    asyncio.run(go())


def test_executor_final_status_strips_only_the_closing_status_line(tmp_path, monkeypatch):
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage

    class Options:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    narration_1 = TextBlock(text="First I will look up the roster.")
    tool_1 = ToolUseBlock(id="t1", name="mcp__x__read", input={})
    narration_2 = TextBlock(text="Now I will check the second table.")
    tool_2 = ToolUseBlock(id="t2", name="mcp__x__read2", input={})
    closing = TextBlock(text="All eleven signups were found and matched.\nSTATUS: OK")

    async def fake_query(prompt, options):
        yield AssistantMessage(content=[narration_1], model="m")
        yield AssistantMessage(content=[tool_1], model="m")
        yield AssistantMessage(content=[narration_2], model="m")
        yield AssistantMessage(content=[tool_2], model="m")
        yield AssistantMessage(content=[closing], model="m")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                            num_turns=1, session_id="s1")

    monkeypatch.setattr(gate_executor, "ClaudeAgentOptions", Options)
    monkeypatch.setattr(gate_executor, "query", fake_query)
    store = Store(str(tmp_path / "exec_gate.db"))
    task = store.create_task("executor final status")
    req = store.create_request(chat_id="owner", sender="Aaron", sender_name="Aaron", mode="imessage", subject="",
                               original_message="go", summary="go", scopes=["web"], prompt="go",
                               state="running", task_id=task["id"])
    result = asyncio.run(gate_executor.Executor(mcp_server=None, cwd=str(tmp_path)).run(req))
    assert result["final_status"] == "All eleven signups were found and matched."
    assert "STATUS: OK" not in result["final_status"]
    assert narration_1.text not in result["final_status"]
    assert narration_2.text not in result["final_status"]
    assert narration_1.text in result["raw"]
    assert "STATUS: OK" in result["raw"]


def test_jev_final_status_is_header_and_tail_only(tmp_path):
    from inkbox_claude.gate.jevagent import JevAgent

    class FakeJudge:
        calls = 0

    class FakeProse:
        calls = 0

    steps = [
        {"tool": "mcp__x__read", "args": {}, "ok": True, "result": "eleven rows found"},
    ]
    status = JevAgent._status(True, "", steps, FakeJudge(), FakeProse(), time.time())
    assert "eleven rows found" not in status["final_status"]
    assert "STATUS: OK" in status["final_status"]
    assert "eleven rows found" in status["raw"]
