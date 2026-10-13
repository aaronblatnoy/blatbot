"""status/stuck/today: plain, code-built answers to "what are you doing"
questions, both through schedcmd.handle() (the router's schedule_command path,
live graph or legacy -- both reach the identical call) and through the fixed
phrase list that matches before the router ever runs."""

from __future__ import annotations

import time

import pytest

from inkbox_claude.gate import schedcmd
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.router import ScheduleCommand
from inkbox_claude.gate.store import Store


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.notices = []
        self.schedules = sched_mod.ScheduleService(self)

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule, text))


class FakeSession:
    def __init__(self, chat_id="owner", approver=True):
        self.chat_id = chat_id
        self.record_id = chat_id
        self.inbound_id = None
        self._approver = approver

    def is_approver(self):
        return self._approver


@pytest.fixture
def env(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    task = store.create_task("status test")
    return store, manager, task["id"]


def _command(action):
    return ScheduleCommand(action=action)


# ---------------------------------------------------------------------------
# fixed phrase matching -- never calls the router
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,action", [
    ("what are you working on", "status"),
    ("What are you doing?", "status"),
    ("anything running", "status"),
    ("what's stuck", "stuck"),
    ("What's stuck?", "stuck"),
    ("what needs me", "stuck"),
    ("what's waiting on me", "stuck"),
    ("what did you do today", "today"),
    ("What did you do today?", "today"),
    ("what happened today", "today"),
])
def test_fixed_phrases_match_whole_message_case_insensitive_question_mark_optional(text, action):
    assert schedcmd.match_fixed_phrase(text) == action


def test_fixed_phrase_does_not_match_inside_a_longer_sentence():
    assert schedcmd.match_fixed_phrase("hey, what are you doing right now") is None


def test_unrelated_text_does_not_match():
    assert schedcmd.match_fixed_phrase("set up a new schedule for me") is None


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_nothing_running_names_the_next_scheduled_run(env):
    store, manager, task_id = env
    schedule = await manager.schedules.propose(
        owner=True, chat_id="owner", task_id=task_id, title="Team total watch",
        prompt="check the team page", cadence_kind="every", cadence={"minutes": 240},
        scopes=["web"], origin_mode="imessage", origin_chat_id="owner")
    await manager.schedules.decide(schedule.id, True)
    reply = await schedcmd.handle(manager, FakeSession(), _command("status"))
    assert reply.startswith("Nothing is running. Next: Team total watch at ")
    assert "Traceback" not in reply


@pytest.mark.asyncio
async def test_status_reports_a_running_request(env):
    store, manager, task_id = env
    schedule = await manager.schedules.propose(
        owner=True, chat_id="owner", task_id=task_id, title="Team total watch",
        prompt="check the team page", cadence_kind="every", cadence={"minutes": 240},
        scopes=["web"], origin_mode="imessage", origin_chat_id="owner")
    activated = await manager.schedules.decide(schedule.id, True)
    req = store.create_request(chat_id="owner", sender="owner", sender_name="owner", mode="imessage",
                               subject="", original_message="", summary="", scopes=["web"],
                               prompt="check the team page", state="pending", task_id=task_id,
                               schedule_id=activated.id)
    store.set_state(req.id, "running", status={"tool_calls": [{"tool": "mcp__web__web_search"}]})
    reply = await schedcmd.handle(manager, FakeSession(), _command("status"))
    assert "Team total watch" in reply
    assert "web search" in reply
    assert "scope" not in reply.lower()
    assert "#" not in reply


# ---------------------------------------------------------------------------
# stuck
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stuck_names_a_blocked_schedule_with_a_let_it_fix(env):
    store, manager, task_id = env
    task_id2 = store.create_task("blocked")["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id2, title="Donor tracker sync",
        prompt="Check the donor page and call inkbox_update_contact to update the tracker note.",
        kind="every", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="every", cadence={"minutes": 60}, origin_mode="imessage", origin_chat_id="owner")
    schedule = store.update_schedule(schedule.id, state="active", next_run=time.time() + 60)
    run_id = store.create_schedule_run(schedule.id, None, time.time() - 60)
    store.finish_schedule_run(run_id, finished_at=time.time() - 30, complete=False, outcome="incomplete",
                              result_line="could not update the tracker note", final_status="blocked",
                              denied=[{"tool": "mcp__inkbox__inkbox_update_contact", "reason": "scope"}],
                              note="", changed=True, cost_usd=None)
    reply = await schedcmd.handle(manager, FakeSession(), _command("stuck"))
    assert reply.startswith("One thing is stuck.")
    assert "Donor tracker sync" in reply
    assert '"let it' in reply
    assert "scope" not in reply.lower()
    assert "#" not in reply


@pytest.mark.asyncio
async def test_stuck_names_a_paused_schedule_with_a_stop_it_fix(env):
    store, manager, task_id = env
    task_id2 = store.create_task("paused")["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id2, title="Weekly backup check",
        prompt="check the backup log", kind="weekly", scopes=["web"], timezone="America/New_York",
        report_mode="changed", cadence_kind="weekly", cadence={"day": "mon", "time": "06:00"},
        origin_mode="imessage", origin_chat_id="owner")
    store.update_schedule(schedule.id, state="paused", last_outcome="three runs in a row could not finish")
    reply = await schedcmd.handle(manager, FakeSession(), _command("stuck"))
    assert "Weekly backup check" in reply
    assert '"stop it"' in reply


@pytest.mark.asyncio
async def test_nothing_stuck(env):
    store, manager, task_id = env
    reply = await schedcmd.handle(manager, FakeSession(), _command("stuck"))
    assert reply == "Nothing is stuck."


# ---------------------------------------------------------------------------
# today
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_today_lists_finished_things_newest_first(env):
    store, manager, task_id = env
    s1 = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Morning brief", prompt="p", kind="daily",
        scopes=["web"], timezone="America/New_York", report_mode="changed", cadence_kind="daily",
        cadence={"time": "07:00"}, origin_mode="imessage", origin_chat_id="owner")
    s2 = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Evening wrap", prompt="p", kind="daily",
        scopes=["web"], timezone="America/New_York", report_mode="changed", cadence_kind="daily",
        cadence={"time": "18:00"}, origin_mode="imessage", origin_chat_id="owner")
    now = time.time()
    r1 = store.create_schedule_run(s1.id, None, now - 3600)
    store.finish_schedule_run(r1, finished_at=now - 3500, complete=True, outcome="complete",
                              result_line="all quiet overnight", final_status="all quiet overnight",
                              denied=[], note="", changed=True, cost_usd=None)
    r2 = store.create_schedule_run(s2.id, None, now - 60)
    store.finish_schedule_run(r2, finished_at=now - 30, complete=True, outcome="complete",
                              result_line="wrapped up the day", final_status="wrapped up the day",
                              denied=[], note="", changed=True, cost_usd=None)
    reply = await schedcmd.handle(manager, FakeSession(), _command("today"))
    assert reply.index("Evening wrap") < reply.index("Morning brief")
    assert "Traceback" not in reply
    assert "markdown" not in reply


@pytest.mark.asyncio
async def test_nothing_done_today(env):
    store, manager, task_id = env
    reply = await schedcmd.handle(manager, FakeSession(), _command("today"))
    assert reply == "Nothing finished today yet."


# ---------------------------------------------------------------------------
# owner-only, no request created
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_owner_gets_todays_behavior_not_an_answer(env):
    store, manager, task_id = env
    reply = await schedcmd.handle(manager, FakeSession(approver=False), _command("status"))
    assert reply == "Only Aaron can manage schedules."


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["status", "stuck", "today"])
async def test_no_request_is_created(env, action):
    store, manager, task_id = env
    before = len(store.recent_requests(limit=1000))
    await schedcmd.handle(manager, FakeSession(), _command(action))
    after = len(store.recent_requests(limit=1000))
    assert after == before


# ---------------------------------------------------------------------------
# a reply never contains a scope name, an id marker, a traceback, or markdown
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["status", "stuck", "today"])
async def test_reply_shape_is_clean(env, action):
    store, manager, task_id = env
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Something", prompt="call inkbox_update_contact",
        kind="every", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="every", cadence={"minutes": 30}, origin_mode="imessage", origin_chat_id="owner")
    store.update_schedule(schedule.id, state="active", next_run=time.time() + 60)
    reply = await schedcmd.handle(manager, FakeSession(), _command(action))
    assert "Traceback" not in reply
    for marker in ("**", "##", "```", "* "):
        assert marker not in reply
    assert "#" not in reply
