"""Plain-language schedule management in chat: answering an open proposal with
a bare yes/no, resolving which schedule Aaron means, and the router's
schedule_command turned into a plain reply -- against the real SQLite store
and the real ScheduleService, with a fake manager/session standing in for
sends and approval."""

from __future__ import annotations

import time

import pytest

from inkbox_claude.gate import schedcmd
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.router import RouterOutput, ScheduleCommand, ScheduleProposal
from inkbox_claude.gate.store import Store


class FakeManager:
    """Stands in for GateSessionManager: a real store, a real ScheduleService,
    and sends recorded instead of delivered."""

    def __init__(self, store):
        self.store = store
        self.notices = []     # (schedule, text)
        self.executed = []
        self.schedules = sched_mod.ScheduleService(self)

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule, text))
        return self.store.add_message(schedule.origin_chat_id or "owner", "outbound", text,
                                      schedule.origin_mode or "imessage", role="notice",
                                      schedule_id=schedule.id)

    async def send_to_approver(self, text):
        pass

    async def execute(self, req):
        self.store.set_state(req.id, "running")
        self.executed.append(req)


class FakeSession:
    """Stands in for GateSession: just enough surface for schedcmd to use."""

    def __init__(self, chat_id="owner", approver=True, inbound_id=None):
        self.chat_id = chat_id
        self.record_id = chat_id
        self.inbound_id = inbound_id
        self._approver = approver
        self.sent = []   # (text, role)

    def is_approver(self):
        return self._approver

    async def send_to_sender(self, text, role="answer"):
        self.sent.append((text, role))
        return True


@pytest.fixture
def env(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    task = store.create_task("schedule command test")
    return store, manager, task["id"]


async def _propose(manager, task_id, **over):
    kw = dict(owner=True, chat_id="owner", task_id=task_id, title="Team total watch",
             prompt="check the team page", cadence_kind="every", cadence={"minutes": 240},
             scopes=["web"], origin_mode="imessage", origin_chat_id="owner")
    kw.update(over)
    return await manager.schedules.propose(**kw)


async def _active(manager, task_id, **over):
    schedule = await _propose(manager, task_id, **over)
    return await manager.schedules.decide(schedule.id, True)


def _command(action, target=None, **changes):
    return ScheduleCommand(action=action, target=target, changes=changes)


# ---------------------------------------------------------------------------
# answer_proposal(): bare yes/no before the router
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bare_yes_answers_the_single_open_proposal(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.answer_proposal(manager, session, "yes")
    assert reply == ""
    assert store.get_schedule(schedule.id).state == "active"


@pytest.mark.asyncio
async def test_bare_no_rejects_the_single_open_proposal(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.answer_proposal(manager, session, "cancel that")
    assert reply == ""
    assert store.get_schedule(schedule.id).state == "ended"


@pytest.mark.asyncio
async def test_bare_yes_not_captured_when_no_open_proposal(env):
    store, manager, task_id = env
    session = FakeSession()
    assert await schedcmd.answer_proposal(manager, session, "yes") is None


@pytest.mark.asyncio
async def test_non_yes_no_text_not_captured_even_with_open_proposal(env):
    store, manager, task_id = env
    await _propose(manager, task_id)
    session = FakeSession()
    assert await schedcmd.answer_proposal(manager, session, "make it every 2 hours instead") is None


@pytest.mark.asyncio
async def test_two_open_proposals_ask_which(env):
    store, manager, task_id = env
    a = await _propose(manager, task_id, title="Team total watch")
    b = await _propose(manager, task_id, title="Monday brief")
    session = FakeSession()
    reply = await schedcmd.answer_proposal(manager, session, "yes")
    assert reply == 'Which one: "Monday brief" or "Team total watch"?'
    # Neither was decided: the ambiguous reply does not pick one for him.
    assert store.get_schedule(a.id).state == "proposed"
    assert store.get_schedule(b.id).state == "proposed"


# ---------------------------------------------------------------------------
# resolve(): which schedule he means
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_by_exact_title(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    found = schedcmd.resolve(store, "Team total watch", "owner", None)
    assert found.id == schedule.id


@pytest.mark.asyncio
async def test_resolve_by_partial_title(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    found = schedcmd.resolve(store, "team watch", "owner", None)
    assert found.id == schedule.id


@pytest.mark.asyncio
async def test_resolve_by_reply_to_tagged_notice(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    msg_id = store.add_message("owner", "outbound", "notice text", "imessage",
                               role="notice", schedule_id=schedule.id)
    found = schedcmd.resolve(store, None, "owner", msg_id)
    assert found.id == schedule.id


@pytest.mark.asyncio
async def test_resolve_by_it_after_a_recent_notice(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    store.add_message("owner", "outbound", "notice text", "imessage",
                      role="notice", schedule_id=schedule.id)
    found = schedcmd.resolve(store, "", "owner", None)
    assert found.id == schedule.id


@pytest.mark.asyncio
async def test_resolve_by_handle(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    found = schedcmd.resolve(store, f"S{schedule.id}", "owner", None)
    assert found.id == schedule.id


@pytest.mark.asyncio
async def test_resolve_empty_target_falls_back_to_only_active_schedule(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    found = schedcmd.resolve(store, "", "owner", None)
    assert found.id == schedule.id


@pytest.mark.asyncio
async def test_resolve_ambiguous_partial_match_returns_a_list(env):
    store, manager, task_id = env
    a = await _active(manager, task_id, title="Morning brief")
    b = await _active(manager, task_id, title="Morning totals")
    found = schedcmd.resolve(store, "morning", "owner", None)
    assert isinstance(found, list)
    assert {s.id for s in found} == {a.id, b.id}


@pytest.mark.asyncio
async def test_resolve_no_match_returns_none(env):
    store, manager, task_id = env
    await _active(manager, task_id)
    assert schedcmd.resolve(store, "something else entirely", "owner", None) is None


# ---------------------------------------------------------------------------
# handle(): list / last
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_list_empty(env):
    store, manager, task_id = env
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("list"))
    assert reply == "Nothing is scheduled."


@pytest.mark.asyncio
async def test_handle_list_shows_cadence_last_result_and_next(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    store.create_schedule_run(schedule.id, None, time.time() - 60)
    store.finish_schedule_run(
        store.schedule_runs(schedule.id)[0]["id"], finished_at=time.time() - 30, complete=True,
        outcome="complete", result_line="all clear", final_status="all clear", denied=[],
        note="", changed=False, cost_usd=None,
    )
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("list"))
    assert "Team total watch" in reply
    assert "every 4 hours" in reply
    assert "found no change" in reply
    assert "next" in reply
    assert "#" not in reply and "S0" not in reply


@pytest.mark.asyncio
async def test_handle_last_reports_result_and_the_run_before_when_it_differed(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    run1 = store.create_schedule_run(schedule.id, None, time.time() - 120)
    store.finish_schedule_run(run1, finished_at=time.time() - 110, complete=True, outcome="complete",
                              result_line="total $4,027.39", final_status="", denied=[], note="",
                              changed=True, cost_usd=None)
    run2 = store.create_schedule_run(schedule.id, None, time.time() - 10)
    store.finish_schedule_run(run2, finished_at=time.time() - 5, complete=True, outcome="complete",
                              result_line="total $4,922.45", final_status="", denied=[], note="",
                              changed=True, cost_usd=None)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("last", target="Team total watch"))
    assert "total $4,922.45" in reply
    assert "Before that" in reply
    assert "total $4,027.39" in reply


@pytest.mark.asyncio
async def test_handle_last_says_why_when_the_run_could_not_finish(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    run = store.create_schedule_run(schedule.id, None, time.time() - 10)
    store.finish_schedule_run(run, finished_at=time.time() - 5, complete=False, outcome="incomplete",
                              result_line="it is not allowed to look up and edit contacts",
                              final_status="", denied=[{"tool": "x__y", "reason": "scope"}], note="",
                              changed=True, cost_usd=None)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("last", target="Team total watch"))
    assert "could not finish" in reply
    assert "look up and edit contacts" in reply


# ---------------------------------------------------------------------------
# handle(): simple actions, by each resolution method
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pause_by_full_title(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target="Team total watch"))
    assert reply == "Paused Team total watch."
    assert store.get_schedule(schedule.id).state == "paused"


@pytest.mark.asyncio
async def test_pause_by_partial_title(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target="total watch"))
    assert reply == "Paused Team total watch."
    assert store.get_schedule(schedule.id).state == "paused"


@pytest.mark.asyncio
async def test_pause_by_reply_to_tagged_notice(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    msg_id = store.add_message("owner", "outbound", "notice", "imessage", role="notice",
                               schedule_id=schedule.id)
    session = FakeSession(inbound_id=msg_id)
    reply = await schedcmd.handle(manager, session, _command("pause", target=None))
    assert reply == "Paused Team total watch."
    assert store.get_schedule(schedule.id).state == "paused"


@pytest.mark.asyncio
async def test_pause_by_it_after_a_recent_notice(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    store.add_message("owner", "outbound", "notice", "imessage", role="notice", schedule_id=schedule.id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target=None))
    assert reply == "Paused Team total watch."
    assert store.get_schedule(schedule.id).state == "paused"


@pytest.mark.asyncio
async def test_pause_by_handle(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target=f"S{schedule.id}"))
    assert reply == "Paused Team total watch."


@pytest.mark.asyncio
async def test_resume(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    await manager.schedules.action(schedule.id, "pause", owner=True)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("resume", target="Team total watch"))
    assert reply.startswith("Team total watch is running again, next check ")
    assert store.get_schedule(schedule.id).state == "active"


@pytest.mark.asyncio
async def test_resume_on_an_ended_schedule_starts_it_again(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    await manager.schedules.action(schedule.id, "stop", owner=True)
    assert store.get_schedule(schedule.id).state == "ended"
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("resume", target="Team total watch"))
    assert reply.startswith("Team total watch is running again, next check ")
    assert store.get_schedule(schedule.id).state == "active"


@pytest.mark.asyncio
async def test_run_now(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("run_now", target="Team total watch"))
    assert reply == "Running Team total watch now."


@pytest.mark.asyncio
async def test_stop(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("stop", target="Team total watch"))
    assert reply == 'Stopped Team total watch. Say "start it again" if you want it back.'
    assert store.get_schedule(schedule.id).state == "ended"


@pytest.mark.asyncio
async def test_invalid_action_for_the_current_state_returns_a_plain_sentence(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    await manager.schedules.action(schedule.id, "pause", owner=True)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target="Team total watch"))
    assert "not valid" in reply
    assert "#" not in reply
    assert "S0" not in reply and "S1" not in reply


# ---------------------------------------------------------------------------
# handle(): ambiguous / no match
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ambiguous_target_asks_which(env):
    store, manager, task_id = env
    await _active(manager, task_id, title="Morning brief")
    await _active(manager, task_id, title="Morning totals")
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target="morning"))
    assert reply.startswith("Which one: ")
    assert "Morning brief" in reply and "Morning totals" in reply


@pytest.mark.asyncio
async def test_no_match_names_it_plainly_and_lists(env):
    store, manager, task_id = env
    await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target="nonexistent thing"))
    assert reply.startswith("I don't have a schedule like that.")
    assert "Team total watch" in reply


# ---------------------------------------------------------------------------
# handle(): change
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_change_cadence_applies_and_next_run_is_right(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command(
        "change", target="Team total watch", cadence_kind="daily", cadence={"time": "08:00"}))
    assert reply.startswith("Changed. Team total watch now runs daily at 8:00 AM, next ")
    updated = store.get_schedule(schedule.id)
    assert updated.cadence_kind == "daily"
    assert updated.next_run is not None


@pytest.mark.asyncio
async def test_change_capabilities_needs_his_yes(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command(
        "change", target="Team total watch", scopes=["contacts"]))
    assert "Reply yes to confirm" in reply
    # The change is pending his yes, not live yet: it needs a fresh decide()
    # to go back to active (same as any other edited-while-active proposal).
    updated = store.get_schedule(schedule.id)
    assert updated.state == "proposed"
    assert "contacts" in updated.scopes


@pytest.mark.asyncio
async def test_change_while_proposal_open_revises_and_resends_it(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command(
        "change", target="Team total watch", cadence_kind="every", cadence={"minutes": 120}))
    # The proposal itself is resent (title + "starting"/"Reply yes"), not an
    # "applied" announcement, since it never started running.
    assert "Reply yes to start" in reply
    updated = store.get_schedule(schedule.id)
    assert updated.state == "proposed"
    assert updated.cadence_kind == "every"
    assert updated.cadence["minutes"] == 120


# ---------------------------------------------------------------------------
# handle(): add_capability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_capability_applies_and_replies_with_labels(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command(
        "add_capability", target="Team total watch", scopes=["contacts"]))
    assert reply.startswith("Done. It can now ")
    updated = store.get_schedule(schedule.id)
    assert "contacts" in updated.scopes


@pytest.mark.asyncio
async def test_add_capability_reruns_after_an_incomplete_run(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    run = store.create_schedule_run(schedule.id, None, time.time() - 10)
    store.finish_schedule_run(run, finished_at=time.time() - 5, complete=False, outcome="incomplete",
                              result_line="could not update the tracker note", final_status="",
                              denied=[{"tool": "x__y", "reason": "scope"}], note="", changed=True,
                              cost_usd=None)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command(
        "add_capability", target="Team total watch", scopes=["contacts"]))
    assert "Running the check again now" in reply
    # run_now sets next_run so the tick loop picks it up right away; it does
    # not execute synchronously inside the chat reply.
    updated = store.get_schedule(schedule.id)
    assert updated.next_run is not None and updated.next_run <= time.time() + 1


@pytest.mark.asyncio
async def test_add_capability_with_nothing_mapped_asks_what_to_allow(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("add_capability", target="Team total watch"))
    assert reply.startswith("What should it be allowed to do")
    assert store.get_schedule(schedule.id).scopes == ["web"]


# ---------------------------------------------------------------------------
# Owner-only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_owner_cannot_manage_schedules(env):
    store, manager, task_id = env
    await _active(manager, task_id)
    session = FakeSession(approver=False)
    reply = await schedcmd.handle(manager, session, _command("list"))
    assert reply == "Only Aaron can manage schedules."


# ---------------------------------------------------------------------------
# Router schema
# ---------------------------------------------------------------------------

def test_router_output_accepts_schedule_command():
    out = RouterOutput.model_validate({
        "reply": None, "request": None,
        "schedule_command": {"action": "pause", "target": "the team one", "changes": {}},
    })
    assert out.schedule_command.action == "pause"
    assert out.schedule is None


def test_router_output_still_accepts_schedule_proposal():
    out = RouterOutput.model_validate({
        "reply": None, "request": None,
        "schedule": {"title": "Team total watch", "prompt": "check the page",
                    "cadence_kind": "every", "cadence": {"minutes": 240}, "scopes": []},
    })
    assert isinstance(out.schedule, ScheduleProposal)
    assert out.schedule_command is None


# ---------------------------------------------------------------------------
# No scope names, "#", "S<digit>" or markdown in anything handle() returns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_reply_leaks_scope_names_ids_or_markdown(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    run = store.create_schedule_run(schedule.id, None, time.time() - 10)
    store.finish_schedule_run(run, finished_at=time.time() - 5, complete=False, outcome="incomplete",
                              result_line="could not update the tracker note", final_status="",
                              denied=[{"tool": "x__y", "reason": "scope"}], note="", changed=True,
                              cost_usd=None)
    session = FakeSession()
    replies = [
        await schedcmd.handle(manager, session, _command("list")),
        await schedcmd.handle(manager, session, _command("last", target="Team total watch")),
        await schedcmd.handle(manager, session, _command("add_capability", target="Team total watch")),
        await schedcmd.handle(manager, session, _command("pause", target="nonexistent")),
    ]
    for reply in replies:
        assert "#" not in reply
        for ch in "*_`[]":
            assert ch not in reply
        assert not __import__("re").search(r"\bS\d+\b", reply)


@pytest.mark.asyncio
async def test_change_until_parses_bare_date_to_end_of_day_epoch(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command(
        "change", target="Team total watch", until="2026-12-31"))
    assert "Changed." in reply
    updated = store.get_schedule(schedule.id)
    assert isinstance(updated.until, float)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dt = datetime.fromtimestamp(updated.until, ZoneInfo(updated.timezone))
    assert (dt.month, dt.day, dt.hour, dt.minute) == (12, 31, 23, 59)


@pytest.mark.asyncio
async def test_change_until_unparseable_gets_plain_reply_and_no_store_change(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command(
        "change", target="Team total watch", until="not a date"))
    assert "didn't understand" in reply
    updated = store.get_schedule(schedule.id)
    assert updated.until == schedule.until


# ---------------------------------------------------------------------------
# resolve(): filler words dropped, prefix match on title words (item C)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_drops_filler_words_and_prefix_matches(env):
    store, manager, task_id = env
    await _active(manager, task_id, title="Team total watch")
    found = schedcmd.resolve(store, "the team one", "owner", None)
    assert found is not None and found.title == "Team total watch"


@pytest.mark.asyncio
async def test_resolve_filler_only_target_treated_as_empty(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id, title="Team total watch")
    # "that one" is all filler words -- with exactly one non-ended schedule on
    # the thread, the empty-target path should resolve it.
    found = schedcmd.resolve(store, "that one", "owner", None)
    assert found is not None and found.id == schedule.id


@pytest.mark.asyncio
async def test_resolve_prefix_match_requires_four_letter_title_word(env):
    store, manager, task_id = env
    await _active(manager, task_id, title="Fix bug")
    # "b" is a prefix of "bug" (3 letters) -- too short to count as a match.
    found = schedcmd.resolve(store, "the b", "owner", None)
    assert found is None


@pytest.mark.asyncio
async def test_resolve_two_prefix_matches_still_ask_which(env):
    store, manager, task_id = env
    await _active(manager, task_id, title="Monday brief")
    await _active(manager, task_id, title="Monthly report")
    found = schedcmd.resolve(store, "the mon one", "owner", None)
    assert isinstance(found, list) and len(found) == 2
