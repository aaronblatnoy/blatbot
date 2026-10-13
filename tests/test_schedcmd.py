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


def _make_stale(store, chat_id):
    """Backdate every outbound message on this thread so the window rule
    (latest-or-15-minutes) fails for all of them, then add a fresh unrelated
    outbound message after -- the shape item 5's bug needed: a proposal is
    open, but it is no longer the kind of 'bare yes answers it' situation."""
    with store._lock:
        store._db.execute("UPDATE messages SET created_at=created_at-3600 WHERE chat_id=?", (chat_id,))
        store._db.commit()
    store.add_message(chat_id, "outbound", "by the way, here's something else", "imessage", role="ack")


@pytest.mark.asyncio
async def test_bare_answer_outside_the_window_asks_instead_of_dropping_silently(env):
    # A proposal is open, but its own notice is neither the latest outbound
    # message nor within 15 minutes: the owner must never be left with no
    # reply at all (item 5) -- he is asked to confirm explicitly instead.
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    _make_stale(store, "owner")
    session = FakeSession()
    reply = await schedcmd.answer_proposal(manager, session, "yes")
    assert reply == f'Do you mean start "{schedule.title}"? Say "yes, start it".'
    assert store.get_schedule(schedule.id).state == "proposed"  # not decided either way


@pytest.mark.asyncio
async def test_bare_no_outside_the_window_asks_to_drop(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    _make_stale(store, "owner")
    session = FakeSession()
    reply = await schedcmd.answer_proposal(manager, session, "no")
    assert reply == f'Do you mean drop "{schedule.title}"? Say "no, drop it".'
    assert store.get_schedule(schedule.id).state == "proposed"


@pytest.mark.asyncio
async def test_yes_start_it_resolves_to_the_open_proposal_with_no_target(env):
    # The recovery phrase answer_proposal offers back must actually work: a
    # schedule_command(approve) with no target resolves to the open proposal
    # on this thread before the 12-hour recent-notice rule.
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    _make_stale(store, "owner")
    session = FakeSession()
    await schedcmd.handle(manager, session, _command("approve", target=None))
    assert store.get_schedule(schedule.id).state == "active"


@pytest.mark.asyncio
async def test_no_drop_it_resolves_to_the_open_proposal_with_no_target(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    _make_stale(store, "owner")
    session = FakeSession()
    await schedcmd.handle(manager, session, _command("reject", target=None))
    assert store.get_schedule(schedule.id).state == "ended"


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
    assert reply == "Team total watch is already paused."
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
    said_id = store.add_message("owner", "inbound", "let it edit contacts", "imessage")
    session = FakeSession(inbound_id=said_id)
    reply = await schedcmd.handle(manager, session, _command(
        "add_capability", target="Team total watch", scopes=["contacts"]))
    assert reply.startswith("Done. It can now ")
    updated = store.get_schedule(schedule.id)
    assert "contacts" in updated.scopes


@pytest.mark.asyncio
async def test_add_capability_ignores_the_routers_suggested_scope(env):
    # Item D: changes["scopes"] is the router's own guess at what the prompt
    # needs -- it must never become permission on its own, only his words do.
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    said_id = store.add_message("owner", "inbound", "sounds fine, go ahead", "imessage")
    session = FakeSession(inbound_id=said_id)
    reply = await schedcmd.handle(manager, session, _command(
        "add_capability", target="Team total watch", scopes=["contacts"]))
    assert reply == "What should it be allowed to do?"
    assert "contacts" not in store.get_schedule(schedule.id).scopes


@pytest.mark.asyncio
async def test_add_capability_but_not_clause_grants_only_the_permitted_scope(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    said_id = store.add_message(
        "owner", "inbound", "let it edit contacts but don't send email", "imessage")
    session = FakeSession(inbound_id=said_id)
    reply = await schedcmd.handle(manager, session, _command(
        "add_capability", target="Team total watch", scopes=["contacts", "email_send"]))
    assert "contacts" in store.get_schedule(schedule.id).scopes
    assert "email_send" not in store.get_schedule(schedule.id).scopes
    assert reply.startswith("Done. It can now ")


@pytest.mark.asyncio
async def test_add_capability_reruns_after_an_incomplete_run(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    run = store.create_schedule_run(schedule.id, None, time.time() - 10)
    store.finish_schedule_run(run, finished_at=time.time() - 5, complete=False, outcome="incomplete",
                              result_line="could not update the tracker note", final_status="",
                              denied=[{"tool": "mcp__inkbox__inkbox_update_contact", "reason": "scope"}], note="",
                              changed=True, cost_usd=None)
    said_id = store.add_message("owner", "inbound", "let it edit contacts", "imessage")
    session = FakeSession(inbound_id=said_id)
    reply = await schedcmd.handle(manager, session, _command(
        "add_capability", target="Team total watch", scopes=["contacts"]))
    assert "Running the check again now" in reply
    # run_now sets next_run so the tick loop picks it up right away; it does
    # not execute synchronously inside the chat reply.
    updated = store.get_schedule(schedule.id)
    assert updated.next_run is not None and updated.next_run <= time.time() + 1


@pytest.mark.asyncio
async def test_add_capability_vague_yes_after_denial_grants_exactly_what_was_denied(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    run = store.create_schedule_run(schedule.id, None, time.time() - 10)
    store.finish_schedule_run(run, finished_at=time.time() - 5, complete=False, outcome="incomplete",
                              result_line="could not update the tracker note", final_status="",
                              denied=[{"tool": "mcp__inkbox__inkbox_update_contact", "reason": "scope"}], note="",
                              changed=True, cost_usd=None)
    said_id = store.add_message("owner", "inbound", "let it do that", "imessage")
    session = FakeSession(inbound_id=said_id)
    reply = await schedcmd.handle(manager, session, _command("add_capability", target="Team total watch"))
    assert "contacts" in store.get_schedule(schedule.id).scopes
    assert reply.startswith("Done. It can now ")


@pytest.mark.asyncio
async def test_add_capability_bare_label_reply_grants_the_refused_scope(env):
    # The bot's own refusal asks "What should it be allowed to do: <label>?";
    # replying with exactly that label (no "let it") grants it, because the
    # schedule's last run was refused that scope -- the simplest sound way to
    # know the question was about it.
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    run = store.create_schedule_run(schedule.id, None, time.time() - 10)
    store.finish_schedule_run(run, finished_at=time.time() - 5, complete=False, outcome="incomplete",
                              result_line="could not update the tracker note", final_status="",
                              denied=[{"tool": "mcp__inkbox__inkbox_update_contact", "reason": "scope"}], note="",
                              changed=True, cost_usd=None)
    said_id = store.add_message("owner", "inbound", "look up and edit contacts", "imessage")
    session = FakeSession(inbound_id=said_id)
    reply = await schedcmd.handle(manager, session, _command("add_capability", target="Team total watch"))
    assert "contacts" in store.get_schedule(schedule.id).scopes
    assert reply.startswith("Done. It can now ")


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


# ---------------------------------------------------------------------------
# answer_proposal(): narrowed vocabulary and the recency/window rule (item G)
# ---------------------------------------------------------------------------

def test_yes_vocabulary_is_narrow():
    for phrase in ("yes", "y", "yep", "yeah", "yup", "yes please", "yes do it"):
        assert schedcmd.is_yes(phrase), phrase
    for phrase in ("ok", "okay", "sure", "go ahead", "sounds good", "please do"):
        assert not schedcmd.is_yes(phrase), phrase


def test_no_vocabulary_is_narrow():
    for phrase in ("no", "n", "nope", "nah", "no thanks", "cancel that"):
        assert schedcmd.is_no(phrase), phrase
    for phrase in ("cancel", "don't", "dont", "never mind", "nevermind", "skip it", "not now"):
        assert not schedcmd.is_no(phrase), phrase


@pytest.mark.asyncio
async def test_bare_yes_not_captured_once_window_passes_and_other_message_followed(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    # Back-date the proposal's own notice past the 15 minute window...
    with store._lock:
        store._db.execute("UPDATE messages SET created_at=? WHERE schedule_id=?",
                          (time.time() - 3600, schedule.id))
    # ...and let something else go out on the same thread after it.
    store.add_message("owner", "outbound", "unrelated reply", "imessage", role="answer")
    session = FakeSession()
    # The bare "yes" is not captured as answering this proposal directly --
    # but a proposal IS open on this thread, so he is never left with no
    # reply at all (item 5): he is asked to confirm explicitly instead.
    reply = await schedcmd.answer_proposal(manager, session, "yes")
    assert reply == f'Do you mean start "{schedule.title}"? Say "yes, start it".'
    assert store.get_schedule(schedule.id).state == "proposed"


@pytest.mark.asyncio
async def test_bare_yes_captured_when_proposal_is_still_the_latest_outbound(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    # Old, but nothing has gone out on the thread since -- still captures.
    with store._lock:
        store._db.execute("UPDATE messages SET created_at=? WHERE schedule_id=?",
                          (time.time() - 3600, schedule.id))
    session = FakeSession()
    reply = await schedcmd.answer_proposal(manager, session, "yes")
    assert reply == ""
    assert store.get_schedule(schedule.id).state == "active"


@pytest.mark.asyncio
async def test_bare_yes_captured_within_fifteen_minutes_even_if_not_latest(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    store.add_message("owner", "outbound", "unrelated reply", "imessage", role="answer")
    session = FakeSession()
    reply = await schedcmd.answer_proposal(manager, session, "yes")
    assert reply == ""
    assert store.get_schedule(schedule.id).state == "active"


# ---------------------------------------------------------------------------
# handle(): approve / reject actions, and stop on a proposed schedule (item G)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_action_starts_the_proposal(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("approve", target="Team total watch"))
    assert reply == ""
    assert store.get_schedule(schedule.id).state == "active"


@pytest.mark.asyncio
async def test_reject_action_ends_the_proposal(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("reject", target="Team total watch"))
    assert reply == ""
    assert store.get_schedule(schedule.id).state == "ended"


@pytest.mark.asyncio
async def test_stop_on_a_proposed_schedule_rejects_it(env):
    store, manager, task_id = env
    schedule = await _propose(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("stop", target="Team total watch"))
    assert reply == ""
    assert store.get_schedule(schedule.id).state == "ended"


# ---------------------------------------------------------------------------
# No raw error text anywhere a schedcmd path replies (item L)
# ---------------------------------------------------------------------------

_FORBIDDEN_SUBSTRINGS = ["cadence", "scope", "valid actions", "proposed", "#", "*", "_", "`", "["]


def _assert_clean(reply: str) -> None:
    low = reply.lower()
    for bad in _FORBIDDEN_SUBSTRINGS:
        assert bad not in low, f"{bad!r} leaked in: {reply!r}"
    import re as _re
    assert not _re.search(r"\bs\d", low), f"an S<digit> handle leaked in: {reply!r}"


@pytest.mark.asyncio
async def test_pause_on_an_already_paused_schedule_is_clean(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    await manager.schedules.action(schedule.id, "pause", owner=True)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target="Team total watch"))
    assert reply == "Team total watch is already paused."
    _assert_clean(reply)


@pytest.mark.asyncio
async def test_resume_on_an_already_active_schedule_is_clean(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("resume", target="Team total watch"))
    assert reply == "Team total watch is already running."
    _assert_clean(reply)


@pytest.mark.asyncio
async def test_pause_on_a_proposed_schedule_is_clean(env):
    store, manager, task_id = env
    await _propose(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target="Team total watch"))
    assert reply == "Team total watch has not started yet; say yes or no first."
    _assert_clean(reply)


@pytest.mark.asyncio
async def test_action_on_an_ended_schedule_is_clean(env):
    store, manager, task_id = env
    schedule = await _active(manager, task_id)
    await manager.schedules.action(schedule.id, "stop", owner=True)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("pause", target="Team total watch"))
    assert reply == 'Team total watch has ended. Say "start it again" to bring it back.'
    _assert_clean(reply)


@pytest.mark.asyncio
async def test_cadence_too_short_on_change_is_clean(env):
    store, manager, task_id = env
    await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command(
        "change", target="Team total watch", cadence_kind="every", cadence={"minutes": 1}))
    assert reply == "It has to run at least every 5 minutes."
    _assert_clean(reply)


def test_friendly_action_error_maps_missing_capability():
    reply = schedcmd._friendly_action_error(
        None, ValueError("missing capability: look up and edit contacts"))
    assert reply == "It isn't allowed to look up and edit contacts yet."
    _assert_clean(reply)


def test_friendly_action_error_maps_cadence_too_short():
    reply = schedcmd._friendly_action_error(
        None, schedcmd.cadence_module.CadenceError("an every cadence must be at least 5 minutes"))
    assert reply == "It has to run at least every 5 minutes."
    _assert_clean(reply)


@pytest.mark.asyncio
async def test_list_and_last_sentences_are_clean(env):
    store, manager, task_id = env
    await _active(manager, task_id)
    session = FakeSession()
    reply = await schedcmd.handle(manager, session, _command("list"))
    _assert_clean(reply)
    reply = await schedcmd.handle(manager, session, _command("last", target="Team total watch"))
    _assert_clean(reply)
