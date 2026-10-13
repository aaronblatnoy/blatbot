"""The scheduler core: propose/decide/edit/action, firing, finishing a run, and
notice text -- against the real SQLite store, with the schedprop seam faked
(schedprop.py is another package's file and does not exist in this tree)."""

from __future__ import annotations

import asyncio
import time
import types

import pytest

import inkbox_claude.gate as gate_pkg
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.store import Store


class FakeSchedprop:
    """A stand-in for schedprop.py. Labels are deterministic words so tests can
    assert on exact notice text without caring about scopes.yaml content."""

    LABELS = {"web": "read public web pages", "contacts": "look up and edit contacts",
             "calendar": "look at the calendar"}

    def format_proposal(self, schedule):
        return f"[proposal] {schedule.title}"

    def labels(self, scopes):
        return " and ".join(self.LABELS.get(s, s) for s in scopes)

    def missing_capabilities(self, prompt, scopes):
        # "contacts" is treated as required whenever the prompt says "contact"
        # and the schedule was not granted it -- enough to drive the refusal tests.
        if "contact" in prompt.lower() and "contacts" not in scopes:
            return ["contacts"]
        return []


@pytest.fixture(autouse=True)
def fake_schedprop(monkeypatch):
    fake = FakeSchedprop()
    monkeypatch.setattr(gate_pkg, "schedprop", fake, raising=False)
    return fake


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.notices = []     # (schedule, text)
        self.approver_texts = []
        self.executed = []

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule, text))
        return self.store.add_message(schedule.origin_chat_id or "owner", "outbound", text,
                                      schedule.origin_mode or "imessage", role="notice",
                                      schedule_id=schedule.id)

    async def send_to_approver(self, text):
        self.approver_texts.append(text)

    async def execute(self, req):
        # The real manager marks a request 'running' before the executor runs and
        # 'done'/'failed' once it returns; a fake that never does leaves the row
        # 'approved' forever, which looks like an overlapping run to the next tick.
        self.store.set_state(req.id, "running")
        self.executed.append(req)


@pytest.fixture
def env(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    service = sched_mod.ScheduleService(manager)
    task = store.create_task("schedule test")
    return store, manager, service, task["id"]


async def _propose(service, task_id, **over):
    kw = dict(owner=True, chat_id="owner", task_id=task_id, title="Team total watch",
             prompt="check the team page", cadence_kind="every", cadence={"minutes": 240},
             scopes=["web"], origin_mode="imessage", origin_chat_id="owner")
    kw.update(over)
    return await service.propose(**kw)


@pytest.mark.asyncio
async def test_propose_refuses_prompt_naming_tool_outside_scopes(env):
    store, manager, service, task_id = env
    with pytest.raises(ValueError, match="contacts"):
        await _propose(service, task_id, prompt="check the team page, update the contact note",
                       scopes=["web"])


@pytest.mark.asyncio
async def test_propose_sends_proposal_through_notice_seam(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    assert schedule.state == "proposed"
    assert manager.notices[-1] == (schedule, f"[proposal] {schedule.title}")


@pytest.mark.asyncio
async def test_decide_approve_starts_at_cadence_time_not_now(env):
    store, manager, service, task_id = env
    now = time.time()
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    assert activated.state == "active"
    assert activated.next_run is not None
    assert activated.next_run > now  # the cadence's next time, not immediate


@pytest.mark.asyncio
async def test_decide_approve_start_now_runs_immediately(env):
    store, manager, service, task_id = env
    before = time.time()
    schedule = await _propose(service, task_id, start_now=True)
    activated = await service.decide(schedule.id, True)
    assert activated.next_run is not None
    assert activated.next_run <= time.time()
    assert activated.next_run >= before


@pytest.mark.asyncio
async def test_decide_reject_ends_schedule(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    ended = await service.decide(schedule.id, False)
    assert ended.state == "ended"


@pytest.mark.asyncio
async def test_first_run_notifies(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req = manager.executed[-1]
    status = {"ok": True, "result_line": "all clear", "final_status": "all clear",
             "started_at": time.time(), "finished_at": time.time(), "denied": [], "held": False}
    before = len(manager.notices)
    await service.finished(req, status)
    store.set_state(req.id, "done")
    assert len(manager.notices) == before + 1
    assert "all clear" in manager.notices[-1][1]


@pytest.mark.asyncio
async def test_complete_run_same_result_sends_nothing_but_still_logs_a_row(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req1 = manager.executed[-1]
    status = {"ok": True, "result_line": "no change", "final_status": "no change",
             "finished_at": time.time(), "denied": [], "held": False}
    await service.finished(req1, status)
    store.set_state(req1.id, "done")
    runs_before = len(store.schedule_runs(schedule.id))
    notices_before = len(manager.notices)

    updated = store.get_schedule(schedule.id)
    await service._fire(updated, updated.next_run)
    await asyncio.sleep(0)
    req2 = manager.executed[-1]
    await service.finished(req2, dict(status))
    store.set_state(req2.id, "done")

    assert len(store.schedule_runs(schedule.id)) == runs_before + 1
    assert len(manager.notices) == notices_before  # nothing sent: unchanged and not report_every_time


@pytest.mark.asyncio
async def test_complete_run_different_result_sends_short_notice_with_next_time(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req1 = manager.executed[-1]
    await service.finished(req1, {"ok": True, "result_line": "no change", "finished_at": time.time(),
                                  "denied": [], "held": False})
    store.set_state(req1.id, "done")

    updated = store.get_schedule(schedule.id)
    await service._fire(updated, updated.next_run)
    await asyncio.sleep(0)
    req2 = manager.executed[-1]
    await service.finished(req2, {"ok": True, "result_line": "total went up", "finished_at": time.time(),
                                  "denied": [], "held": False})
    store.set_state(req2.id, "done")
    text = manager.notices[-1][1]
    assert "total went up" in text
    assert "Next check" in text


@pytest.mark.asyncio
async def test_finished_still_notices_a_paused_mid_flight_run_with_no_next_run_line(env):
    # The owner paused the schedule while a run was in flight. The run's result is still
    # told to him (no silent drop), but with no "Next check" line and the schedule stays
    # paused -- the run finishing does not reactivate anything.
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req = manager.executed[-1]

    await service.action(schedule.id, "pause", owner=True)

    before = len(manager.notices)
    await service.finished(req, {"ok": True, "result_line": "found something before the pause landed",
                                 "finished_at": time.time(), "denied": [], "held": False, "changed": True})
    assert len(manager.notices) == before + 1
    text = manager.notices[-1][1]
    assert "found something before the pause landed" in text
    assert "Next check" not in text
    assert store.get_schedule(schedule.id).state == "paused"


@pytest.mark.asyncio
async def test_finished_uses_reported_changed_true_even_if_text_is_unchanged(env):
    # The model said CHANGED: yes even though result_line/note text is identical to
    # the previous run's -- its judgment is the source of truth, not the text diff.
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req1 = manager.executed[-1]
    await service.finished(req1, {"ok": True, "result_line": "same text", "note": "same note",
                                  "finished_at": time.time(), "denied": [], "held": False, "changed": True})
    store.set_state(req1.id, "done")
    before = len(manager.notices)
    updated = store.get_schedule(schedule.id)
    await service._fire(updated, updated.next_run)
    await asyncio.sleep(0)
    req2 = manager.executed[-1]
    await service.finished(req2, {"ok": True, "result_line": "same text", "note": "same note",
                                  "finished_at": time.time(), "denied": [], "held": False, "changed": True})
    store.set_state(req2.id, "done")
    assert len(manager.notices) == before + 1


@pytest.mark.asyncio
async def test_finished_uses_reported_changed_false_even_if_text_differs(env):
    # The model said CHANGED: no even though the text differs from the previous run's --
    # again, its own judgment wins over the text-diff heuristic.
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req1 = manager.executed[-1]
    await service.finished(req1, {"ok": True, "result_line": "first text", "finished_at": time.time(),
                                  "denied": [], "held": False, "changed": True})
    store.set_state(req1.id, "done")
    before = len(manager.notices)
    updated = store.get_schedule(schedule.id)
    await service._fire(updated, updated.next_run)
    await asyncio.sleep(0)
    req2 = manager.executed[-1]
    await service.finished(req2, {"ok": True, "result_line": "totally different text",
                                  "finished_at": time.time(), "denied": [], "held": False, "changed": False})
    store.set_state(req2.id, "done")
    assert len(manager.notices) == before   # no notice sent despite differing text


@pytest.mark.asyncio
async def test_finished_falls_back_to_text_heuristic_when_changed_absent(env):
    # No "changed" key at all in status (the old contract, or a run whose closing text
    # never matched CHANGED:): falls back to comparing result_line/note text.
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req1 = manager.executed[-1]
    await service.finished(req1, {"ok": True, "result_line": "same text", "finished_at": time.time(),
                                  "denied": [], "held": False})
    store.set_state(req1.id, "done")
    before = len(manager.notices)
    updated = store.get_schedule(schedule.id)
    await service._fire(updated, updated.next_run)
    await asyncio.sleep(0)
    req2 = manager.executed[-1]
    await service.finished(req2, {"ok": True, "result_line": "same text", "finished_at": time.time(),
                                  "denied": [], "held": False})
    store.set_state(req2.id, "done")
    assert len(manager.notices) == before   # text identical, no "changed" key -> no notice


@pytest.mark.asyncio
async def test_report_every_time_always_notifies(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id, report_every_time=True)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req = manager.executed[-1]
    status = {"ok": True, "result_line": "same as before", "finished_at": time.time(),
             "denied": [], "held": False}
    await service.finished(req, status)
    store.set_state(req.id, "done")
    before = len(manager.notices)
    updated = store.get_schedule(schedule.id)
    await service._fire(updated, updated.next_run)
    await asyncio.sleep(0)
    req2 = manager.executed[-1]
    await service.finished(req2, dict(status))
    store.set_state(req2.id, "done")
    assert len(manager.notices) == before + 1


@pytest.mark.asyncio
async def test_denied_tool_records_incomplete_and_names_capability_even_when_ok(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req = manager.executed[-1]
    status = {"ok": True, "result_line": "total changed", "finished_at": time.time(),
             "denied": [{"tool": "mcp__tamid-drive__list_calendars", "reason": "scope"}], "held": False}
    await service.finished(req, status)
    store.set_state(req.id, "done")
    run = store.last_schedule_run(schedule.id)
    assert run["outcome"] == "incomplete"
    assert not run["complete"]
    text = manager.notices[-1][1]
    assert "could not finish" in text
    assert "look at the calendar" in text
    assert 'let it' in text
    assert '"stop it"' in text


@pytest.mark.asyncio
async def test_failed_run_wording_has_countdown(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req = manager.executed[-1]
    status = {"ok": False, "error": "timed out", "result_line": "", "final_status": "timed out",
             "finished_at": time.time(), "denied": [], "held": False}
    await service.finished(req, status)
    store.set_state(req.id, "done")
    text = manager.notices[-1][1]
    assert "I will try again at" in text
    assert "After 2 more misses" in text


@pytest.mark.asyncio
async def test_three_consecutive_incomplete_runs_pause_and_then_nothing_fires(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    status = {"ok": False, "error": "boom", "finished_at": time.time(), "denied": [], "held": False}
    current = activated
    for _ in range(3):
        await service._fire(current, current.next_run)
        await asyncio.sleep(0)
        req = manager.executed[-1]
        await service.finished(req, dict(status))
        store.set_state(req.id, "done")
        current = store.get_schedule(schedule.id)
    assert current.state == "paused"
    assert "paused after three" in manager.notices[-1][1]
    runs_before = len(manager.executed)
    await service.tick(now=current.next_run or time.time() + 10)
    assert len(manager.executed) == runs_before  # paused: nothing fires


@pytest.mark.asyncio
async def test_status_stop_ends_schedule(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req = manager.executed[-1]
    status = {"ok": True, "stop": True, "result_line": "all done", "finished_at": time.time(),
             "denied": [], "held": False}
    await service.finished(req, status)
    store.set_state(req.id, "done")
    updated = store.get_schedule(schedule.id)
    assert updated.state == "ended"


@pytest.mark.asyncio
async def test_until_ends_schedule_without_running(env):
    store, manager, service, task_id = env
    now = time.time()
    schedule = await _propose(service, task_id, until=now - 1)
    activated = await service.decide(schedule.id, True)
    before = len(manager.executed)
    await service.tick(now=activated.next_run + 5)
    after = store.get_schedule(schedule.id)
    assert after.state == "ended"
    assert len(manager.executed) == before


@pytest.mark.asyncio
async def test_overlap_skip_writes_skipped_row_and_no_notice(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)  # leaves a request in 'approved'
    await asyncio.sleep(0)
    before_notices = len(manager.notices)
    refetched = store.get_schedule(schedule.id)
    await service._fire(refetched, refetched.next_run or time.time())
    await asyncio.sleep(0)
    runs = store.schedule_runs(schedule.id)
    assert runs[0]["outcome"] == "skipped"
    assert len(manager.notices) == before_notices


@pytest.mark.asyncio
async def test_notices_tag_the_message_row_with_schedule_id(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    message_id = await manager.send_schedule_notice(schedule, "a notice")
    tagged = store.schedule_for_message(message_id)
    assert tagged is not None and tagged.id == schedule.id


@pytest.mark.asyncio
async def test_valid_actions_per_state(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    assert set(service.valid_actions(schedule)) == {"approve", "reject", "edit"}
    activated = await service.decide(schedule.id, True)
    assert set(service.valid_actions(activated)) == {"pause", "stop", "run_now", "edit"}
    paused = await service.action(schedule.id, "pause", owner=True)
    assert set(service.valid_actions(paused)) == {"resume", "stop", "edit"}
    ended = await service.action(schedule.id, "stop", owner=True)
    assert set(service.valid_actions(ended)) == {"start_again", "edit"}


@pytest.mark.asyncio
async def test_action_refuses_invalid_for_state(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    with pytest.raises(ValueError):
        await service.action(schedule.id, "pause", owner=True)  # still proposed, not active


@pytest.mark.asyncio
async def test_edit_cadence_applies_immediately(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    result = await service.edit(schedule.id, owner=True,
                                changes={"cadence_kind": "daily", "cadence": {"time": "08:00"}})
    assert result["needs_approval"] is False
    assert result["schedule"].cadence_kind == "daily"
    assert result["schedule"].state == "active"


@pytest.mark.asyncio
async def test_edit_prompt_needs_approval(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    before_revision = activated.revision
    result = await service.edit(schedule.id, owner=True, changes={"prompt": "check the other page"})
    assert result["needs_approval"] is True
    assert result["schedule"].state == "proposed"
    assert result["schedule"].revision == before_revision + 1


@pytest.mark.asyncio
async def test_next_run_executor_call_gets_only_last_note_and_timezone(env):
    """schedules.py itself never builds the executor prompt; it just carries
    last_note forward on the schedule row for manager._run_executor to read."""
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req = manager.executed[-1]
    await service.finished(req, {"ok": True, "result_line": "ok", "note": "watch the second column",
                                 "finished_at": time.time(), "denied": [], "held": False})
    store.set_state(req.id, "done")
    updated = store.get_schedule(schedule.id)
    assert updated.last_note == "watch the second column"
    assert updated.timezone == schedule.timezone


@pytest.mark.asyncio
async def test_long_final_status_stored_whole_never_in_notice(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    await service._fire(activated, activated.next_run)
    await asyncio.sleep(0)
    req = manager.executed[-1]
    huge = "x" * 9000
    await service.finished(req, {"ok": True, "result_line": "short result", "final_status": huge,
                                 "finished_at": time.time(), "denied": [], "held": False})
    store.set_state(req.id, "done")
    run = store.last_schedule_run(schedule.id)
    assert len(run["final_status"]) == 9000
    for schedule_obj, text in manager.notices:
        assert huge not in text


def test_no_jev_import_anywhere_in_schedule_files():
    import inspect
    from inkbox_claude.gate import schedules as s, cadence as c
    for mod in (s, c):
        source = inspect.getsource(mod)
        assert "jev" not in source.lower()


@pytest.mark.asyncio
async def test_rejecting_an_edit_of_a_never_approved_proposal_ends_it(env):
    """edit() stashes a revert snapshot even when the schedule being edited was
    itself still 'proposed' (he never said yes to it even once). Rejecting that
    edit must not leave a 'paused' schedule resume could restart unapproved."""
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    assert schedule.state == "proposed"
    await service.edit(schedule.id, owner=True, changes={"prompt": "check the other page"})
    edited = store.get_schedule(schedule.id)
    assert edited.state == "proposed"

    rejected = await service.decide(schedule.id, False)
    assert rejected.state == "ended"
    assert "resume" not in service.valid_actions(rejected)


@pytest.mark.asyncio
async def test_expiring_an_edit_of_a_never_approved_proposal_ends_it(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    await service.edit(schedule.id, owner=True, changes={"prompt": "check the other page"})

    await service.expire_stale_proposals(max_age=0, now=time.time() + 100000)
    ended = store.get_schedule(schedule.id)
    assert ended.state == "ended"
    assert "resume" not in service.valid_actions(ended)


@pytest.mark.asyncio
async def test_a_once_cadence_already_in_the_past_is_refused_on_propose(env):
    store, manager, service, task_id = env
    past = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 3600))
    with pytest.raises(ValueError, match="already passed"):
        await _propose(service, task_id, cadence_kind="once", cadence={"at": past},
                      timezone="America/New_York")


@pytest.mark.asyncio
async def test_a_once_cadence_edited_into_the_past_is_refused(env):
    store, manager, service, task_id = env
    schedule = await _propose(service, task_id)
    activated = await service.decide(schedule.id, True)
    assert activated.state == "active"
    past = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 3600))
    with pytest.raises(ValueError, match="already passed"):
        await service.edit(schedule.id, owner=True, changes={"cadence_kind": "once", "cadence": {"at": past}})
