"""status/stuck/today: plain, code-built answers to "what are you doing"
questions, assembled from the SAME row builders console/api.py's
/console/api/overview uses (gate/overview.py) -- never a second,
independently-guessed diagnosis of the same schedule. Exercised through
schedcmd.handle() (the router's schedule_command path, live graph or legacy
-- both reach the identical call), through the fixed phrase list that
matches before the router ever runs, and (for one phrasing) through
GateSession.decide() itself on both the graph and legacy paths."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate import schedcmd
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.router import RouterOutput, ScheduleCommand
from inkbox_claude.gate.store import Store

APPROVER_CONV = "conv-aaron"


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.notices = []
        self.schedules = sched_mod.ScheduleService(self)

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule, text))
        return self.store.add_message(schedule.origin_chat_id or "owner", "outbound", text,
                                      schedule.origin_mode or "imessage", role="notice",
                                      schedule_id=schedule.id)


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


def _command(action, target=None, **changes):
    return ScheduleCommand(action=action, target=target, changes=changes)


def _paused_no_cadence(store, task_id, *, title="Pledge drive watch"):
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title=title,
        prompt=("Check the pledge page every 4 hours and tell me the total.\n\n"
                "When done, call schedule_continue with delay_minutes 240 to run again.\n\n"
                "If the total changed, call inkbox_update_contact with the new total."),
        kind="", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="", cadence={}, origin_mode="imessage", origin_chat_id="owner")
    return store.update_schedule(schedule.id, state="paused", last_outcome="needs a cadence")


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
    ("repair it", "repair"),
    ("Repair it?", "repair"),
])
def test_fixed_phrases_match_whole_message_case_insensitive_question_mark_optional(text, action):
    assert schedcmd.match_fixed_phrase(text) == action


def test_fixed_phrase_does_not_match_inside_a_longer_sentence():
    assert schedcmd.match_fixed_phrase("hey, what are you doing right now") is None


def test_named_repair_phrase_is_not_in_the_fixed_list_it_needs_the_router():
    assert schedcmd.match_fixed_phrase("repair the pledge one") is None


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
    assert reply.startswith("Nothing is running. Next: Team total watch, ")
    assert "Traceback" not in reply


@pytest.mark.asyncio
async def test_status_reports_a_running_schedule(env):
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
    store.set_state(req.id, "running")
    reply = await schedcmd.handle(manager, FakeSession(), _command("status"))
    assert "Team total watch" in reply
    assert "running now" in reply
    assert "scope" not in reply.lower()
    assert "#" not in reply


# ---------------------------------------------------------------------------
# stuck
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stuck_names_a_schedule_needing_repair_with_repair_it(env):
    store, manager, task_id = env
    _paused_no_cadence(store, task_id)
    reply = await schedcmd.handle(manager, FakeSession(), _command("stuck"))
    assert reply.startswith("One thing is stuck.")
    assert "Pledge drive watch" in reply
    assert "every 4 hours" in reply
    assert '"repair it"' in reply
    assert "scope" not in reply.lower()
    assert "#" not in reply


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


@pytest.mark.asyncio
async def test_a_plain_paused_schedule_with_no_chat_fix_is_not_listed(env):
    """A schedule paused for a reason neither repair nor a missing capability
    explains has no chat fix (say is None in the shared builder) -- only the
    console can act on it (Resume/Stop buttons), so the chat reply leaves it
    out rather than offering "stop it" as if that were a diagnosis."""
    store, manager, task_id = env
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Weekly backup check",
        prompt="check the backup log", kind="weekly", scopes=["web"], timezone="America/New_York",
        report_mode="changed", cadence_kind="weekly", cadence={"day": "mon", "time": "06:00"},
        origin_mode="imessage", origin_chat_id="owner")
    store.update_schedule(schedule.id, state="paused", last_outcome="three runs in a row could not finish")
    reply = await schedcmd.handle(manager, FakeSession(), _command("stuck"))
    assert reply == "Nothing is stuck."


@pytest.mark.asyncio
async def test_stuck_names_each_of_several_things_with_its_own_word(env):
    store, manager, task_id = env
    _paused_no_cadence(store, task_id, title="Pledge drive watch")
    task_id2 = store.create_task("failed ask")["id"]
    req = store.create_request(chat_id="owner", sender="owner", sender_name="A Donor", mode="imessage",
                               subject="", original_message="find the sponsorship deck",
                               summary="find the sponsorship deck", scopes=["web"],
                               prompt="find the sponsorship deck", state="pending", task_id=task_id2)
    store.set_state(req.id, "failed", status={"error": "could not reach the page"})
    reply = await schedcmd.handle(manager, FakeSession(), _command("stuck"))
    assert reply.startswith("Two things are stuck.")
    assert '"repair the pledge one"' in reply
    assert '"try the' in reply and "again" in reply


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
async def test_today_normalizes_a_first_person_result_and_excludes_a_failed_run(env):
    """A failed run belongs under Stuck, never also under Done today, and a
    complete run's own first-person text is never shown as the scheduler's
    own words."""
    store, manager, task_id = env
    s1 = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Morning brief", prompt="p", kind="daily",
        scopes=["web"], timezone="America/New_York", report_mode="changed", cadence_kind="daily",
        cadence={"time": "07:00"}, origin_mode="imessage", origin_chat_id="owner")
    s1 = store.update_schedule(s1.id, state="active", next_run=time.time() + 3600)
    now = time.time()
    r1 = store.create_schedule_run(s1.id, None, now - 3600)
    store.finish_schedule_run(r1, finished_at=now - 3500, complete=True, outcome="complete",
                              result_line="I could not find anything new, so nothing changed.",
                              final_status="checked and found nothing new", denied=[], note="",
                              changed=True, cost_usd=None)
    s2 = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Evening wrap", prompt="p", kind="daily",
        scopes=["web"], timezone="America/New_York", report_mode="changed", cadence_kind="daily",
        cadence={"time": "18:00"}, origin_mode="imessage", origin_chat_id="owner")
    s2 = store.update_schedule(s2.id, state="active", next_run=time.time() + 3600)
    r2 = store.create_schedule_run(s2.id, None, now - 60)
    store.finish_schedule_run(r2, finished_at=now - 30, complete=False, outcome="failed",
                              result_line="could not reach the page", final_status="gave up after 3 tries",
                              denied=[], note="", changed=True, cost_usd=None)
    reply = await schedcmd.handle(manager, FakeSession(), _command("today"))
    assert "Evening wrap" not in reply  # the failed run belongs under Stuck, not here
    assert not reply.lower().startswith("i could not")
    assert "Morning brief" in reply


@pytest.mark.asyncio
async def test_nothing_done_today(env):
    store, manager, task_id = env
    reply = await schedcmd.handle(manager, FakeSession(), _command("today"))
    assert reply == "Nothing finished today."


# ---------------------------------------------------------------------------
# repair, via schedule_command
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repair_bare_asks_how_often_then_applies_the_timing_reply(env):
    store, manager, task_id = env
    schedule = _paused_no_cadence(store, task_id, title="Morning signal check")
    # This one's prompt states no old timing at all, so the plan has none to recover.
    schedule = store.update_schedule(schedule.id, prompt="Check the signal page and tell me the total.")
    reply = await schedcmd.handle(manager, FakeSession(), _command("repair", target="Morning signal check"))
    assert reply == ""
    assert manager.notices
    question = manager.notices[-1][1]
    assert "how often" in question.lower()
    # Answering with a timing on this thread resolves back to the same schedule via
    # the notice tag (the same rule that lets a reply to any other schedule notice
    # answer that schedule), the same way resolve() already matches a bare "yes".
    reply2 = await schedcmd.handle(
        manager, FakeSession(), _command("repair", cadence_kind="every", cadence={"minutes": 240}))
    updated = store.get_schedule(schedule.id)
    assert updated.cadence_kind == "every"
    assert updated.cadence["minutes"] == 240
    assert reply2 != ""


@pytest.mark.asyncio
async def test_repair_named_resolves_to_the_right_schedule_among_several(env):
    store, manager, task_id = env
    _paused_no_cadence(store, task_id, title="Pledge drive watch")
    task_id2 = store.create_task("other")["id"]
    other = store.create_schedule(
        chat_id="owner", task_id=task_id2, title="Donor tracker sync", prompt="check the donor page",
        kind="every", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="every", cadence={"minutes": 60}, origin_mode="imessage", origin_chat_id="owner")
    store.update_schedule(other.id, state="active", next_run=time.time() + 60)
    reply = await schedcmd.handle(manager, FakeSession(), _command("repair", target="the pledge one"))
    assert "Pledge drive watch" in reply or reply == ""
    pledge = next(s for s in store.schedules() if s.title == "Pledge drive watch")
    assert pledge.state == "proposed"  # repair_apply sent one proposal (cap+prompt also changed)


@pytest.mark.asyncio
async def test_repair_on_a_healthy_schedule_says_so(env):
    store, manager, task_id = env
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Fine already", prompt="check the page",
        kind="every", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="every", cadence={"minutes": 60}, origin_mode="imessage", origin_chat_id="owner")
    store.update_schedule(schedule.id, state="active", next_run=time.time() + 60)
    reply = await schedcmd.handle(manager, FakeSession(), _command("repair", target="Fine already"))
    assert reply == "Fine already does not need repair."


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


# ---------------------------------------------------------------------------
# requirement 5: GateSession.decide() itself never calls the router for a
# fixed phrasing, on either the live graph path or the legacy path.
# ---------------------------------------------------------------------------

class _RouterNeverCalled:
    async def route(self, **kw):
        raise AssertionError("the router must not be called for a fixed phrase")

    async def plan_query(self, **kw):
        return None

    async def phrase(self, *a, **kw):
        return ""


class _FakePicker:
    async def pick(self, **kw):
        return {"choice": "none"}

    async def judge_action(self, **kw):
        return {"needs_action": False, "p": 0.0}


def _make_manager(tmp_path, *, legacy: bool):
    sent = []

    async def send_fn(chat_id, text, mode, meta):
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV)
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    m.jev_agent = None
    m.caller = None
    m.router = _RouterNeverCalled()
    if legacy:
        m.router_defines_requests = True
    else:
        m.task_picker = _FakePicker()
        m.router_defines_requests = False
    return m, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True], ids=["graph_path", "legacy_path"])
async def test_fixed_phrase_never_reaches_the_router_on_either_path(tmp_path, legacy):
    m, sent = _make_manager(tmp_path, legacy=legacy)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("what are you doing", mode="imessage",
                                 meta={"sender": "+15550100001", "conversation_id": APPROVER_CONV})
    assert m.store.requests_in_state("pending") == []
    assert m.store.requests_in_state("approved") == []
    assert sent, "a reply should have gone out"
    assert len(sent) == 1
    reply_text = sent[-1][1]
    assert "running" in reply_text.lower()
