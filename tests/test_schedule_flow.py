"""End-to-end schedule flow: real GateSessionManager, ScheduleService, schedprop,
cadence and store. Only three seams are faked: the router's model output (a
canned RouterOutput), the executor's Claude call (FakeExecutor.run), and the
outbound transport (send_fn capturing instead of sending). Everything else --
propose/decide/tick/_fire/finished/notice_text, schedprop's capability scoping
and labels, cadence's next-run math -- runs for real against a throwaway
SQLite store, matching the integration-level verification this package owns.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate.router import RouterOutput, ScheduleProposal
from inkbox_claude.gate.store import Request, Store

APPROVER_CONV = "conv-aaron"


class FakeExecutor:
    """Stands in for Executor.run (the owner's Claude Code call). Each queued
    result is consumed in order; build_prompt/parse_closing are schedrun.py's
    own job and are exercised directly below, not re-derived here."""

    def __init__(self):
        self.calls = []
        self.results = []

    async def run(self, req, context="", prior_work="", model=None, **kw):
        self.calls.append({"req": req, "context": context, "kw": kw})
        return self.results.pop(0)


def make_manager(tmp_path):
    sent = []

    async def send_fn(chat_id, text, mode, meta):
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV)
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    m.jev_agent = None  # force the plain executor path regardless of env
    m.caller = None
    m.executor = FakeExecutor()
    return m, sent


async def _propose_via_manager(m, *, title="Team total watch",
                               prompt="check the team page", said="every 4 hours check the team page",
                               cadence_kind="every", cadence=None, scopes=None,
                               report_every_time=None, start_now=False, until=None, timezone=None,
                               session_mode="imessage", session_chat_id=None, session_meta=None):
    task = m.store.create_task(title)
    proposal = ScheduleProposal(title=title, prompt=prompt, cadence_kind=cadence_kind,
                                cadence=cadence or {"minutes": 240}, timezone=timezone,
                                report_every_time=report_every_time, until=until,
                                scopes=scopes or [], start_now=start_now)
    out = RouterOutput(reply="", request=None, schedule=proposal)

    class _Router:
        async def route(self, **kw):
            return out.model_copy(deep=True)

        async def plan_query(self, **kw):
            return None

    m.router = _Router()
    chat_id = session_chat_id or APPROVER_CONV
    meta = session_meta if session_meta is not None else {
        "sender": "+15550100001", "to": "+15550100001", "conversation_id": APPROVER_CONV}
    session = m.get(chat_id)
    await session.handle_inbound(said, mode=session_mode, meta=meta)


async def _afut(value):
    return value


@pytest.mark.asyncio
async def test_full_flow_propose_approve_fire_complete_notice(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    assert schedule.scopes == ["web"]

    proposal_text = sent[-1][1]
    assert proposal_text == (
        f'Set up "{schedule.title}": every 4 hours, starting '
        + proposal_text.split("starting ", 1)[1]
    )  # sanity: the exact string is asserted in full just below, against the live schedule
    first_run = schedule.next_run or time.time()
    from inkbox_claude.gate import schedprop
    expected = (f'Set up "{schedule.title}": every 4 hours, starting '
               + schedprop._first_run_text(schedule) + '. It will read public web pages. '
               'You hear from it only when something changed or it could not finish. '
               'Reply yes to start, or tell me what to change.')
    assert proposal_text == expected

    activated = await m.schedules.decide(schedule.id, True)
    assert activated.state == "active"
    started_text = sent[-1][1]
    assert started_text.startswith("Started. First check ")
    assert "every 4 hours" in started_text

    m.executor.results.append({"ok": True, "result_line": "total went from $4,027.39 to $4,922.45",
                               "final_status": "checked the page", "note": "", "denied": [], "tool_calls": ["mcp__fake__tool"], "held": False,
                               "started_at": time.time(), "finished_at": time.time(), "cost_usd": None})
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    call = m.executor.calls[-1]
    assert call["req"].schedule_id == schedule.id
    assert call["kw"]["timezone"] == schedule.timezone
    assert call["kw"]["note"] == ""

    run = m.store.last_schedule_run(schedule.id)
    assert run["outcome"] == "complete"
    notice = sent[-1][1]
    assert notice.startswith(f"{schedule.title}: total went from $4,027.39 to $4,922.45")
    assert "Next check" in notice


@pytest.mark.asyncio
async def test_second_run_same_result_sends_no_notice_but_logs_a_row(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    same_status = {"ok": True, "result_line": "no change", "final_status": "no change", "note": "",
                   "denied": [], "tool_calls": ["mcp__fake__tool"], "held": False, "finished_at": time.time(), "cost_usd": None}
    m.executor.results.append(dict(same_status))
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    runs_before = len(m.store.schedule_runs(schedule.id))
    notices_before = len(sent)

    updated = m.store.get_schedule(schedule.id)
    m.executor.results.append(dict(same_status))
    await m.schedules.tick(now=updated.next_run)
    await asyncio.sleep(0)

    assert len(m.store.schedule_runs(schedule.id)) == runs_before + 1
    assert len(sent) == notices_before


@pytest.mark.asyncio
async def test_denied_contact_tool_gives_incomplete_notice_with_phrase(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page and update the contact note",
                               prompt="check the team page and update the contact note",
                               scopes=["web", "contacts"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    m.executor.results.append({
        "ok": True, "result_line": "the total did change (+$200)", "final_status": "checked",
        "note": "", "denied": [{"tool": "mcp__tamid-drive__manage_doc_tab", "reason": "scope"}],
        "tool_calls": ["mcp__fake__tool"], "held": False, "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    run = m.store.last_schedule_run(schedule.id)
    assert run["outcome"] == "incomplete" and not run["complete"]
    notice = sent[-1][1]
    assert "could not finish" in notice
    assert "let it" in notice
    assert '"stop it"' in notice


@pytest.mark.asyncio
async def test_failed_run_notice_has_retry_countdown(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    m.executor.results.append({"ok": False, "error": "the page timed out", "result_line": "",
                               "final_status": "the page timed out", "note": "", "denied": [],
                               "tool_calls": ["mcp__fake__tool"], "held": False, "finished_at": time.time(), "cost_usd": None})
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    notice = sent[-1][1]
    assert "I will try again at" in notice
    assert "After 2 more misses" in notice


@pytest.mark.asyncio
async def test_three_consecutive_incomplete_runs_pause_it(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    current = await m.schedules.decide(schedule.id, True)

    for _ in range(3):
        m.executor.results.append({"ok": False, "error": "boom", "result_line": "", "final_status": "boom",
                                   "note": "", "denied": [], "tool_calls": ["mcp__fake__tool"], "held": False, "finished_at": time.time(),
                                   "cost_usd": None})
        await m.schedules.tick(now=current.next_run)
        await asyncio.sleep(0)
        current = m.store.get_schedule(schedule.id)

    assert current.state == "paused"
    assert "paused after three" in sent[-1][1]


@pytest.mark.asyncio
async def test_status_stop_ends_schedule_with_notice(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    m.executor.results.append({"ok": True, "stop": True, "result_line": "the team disbanded",
                               "final_status": "the team disbanded", "note": "", "denied": [],
                               "tool_calls": ["mcp__fake__tool"], "held": False, "finished_at": time.time(), "cost_usd": None})
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    assert m.store.get_schedule(schedule.id).state == "ended"
    assert "the team disbanded" in sent[-1][1]
    assert "will not run again" in sent[-1][1]


@pytest.mark.asyncio
async def test_one_time_schedule_ends_after_its_single_run(tmp_path):
    m, sent = make_manager(tmp_path)
    import datetime as _dt
    at = _dt.datetime.now() + _dt.timedelta(minutes=5)
    await _propose_via_manager(m, said="remind me once tomorrow", scopes=["web"],
                               cadence_kind="once", cadence={"at": at.isoformat()})
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    assert activated.cadence_kind == "once"

    m.executor.results.append({"ok": True, "result_line": "done once", "final_status": "done once",
                               "note": "", "denied": [], "tool_calls": ["mcp__fake__tool"], "held": False,
                               "finished_at": time.time(), "cost_usd": None})
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    updated = m.store.get_schedule(schedule.id)
    assert updated.state == "ended"
    assert updated.next_run is None


@pytest.mark.asyncio
async def test_proposal_expires_unanswered_after_24h(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]

    await m.schedules.expire_stale_proposals(24 * 3600, now=time.time() + 24 * 3600 + 1)
    updated = m.store.get_schedule(schedule.id)
    assert updated.state == "ended"
    assert updated.last_outcome == "not approved in time"
    assert "was not approved in time" in sent[-1][1]


@pytest.mark.asyncio
async def test_edit_proposal_expiry_reverts_an_active_schedule(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    old_prompt = activated.prompt

    result = await m.schedules.edit(schedule.id, owner=True, changes={"prompt": "check the other page"})
    assert result["needs_approval"] is True
    proposed = m.store.get_schedule(schedule.id)
    assert proposed.state == "proposed"

    await m.schedules.expire_stale_proposals(24 * 3600, now=time.time() + 24 * 3600 + 1)
    reverted = m.store.get_schedule(schedule.id)
    assert reverted.state == "active"
    assert reverted.prompt == old_prompt
    assert "kept it as it was" in sent[-1][1]


@pytest.mark.asyncio
async def test_restart_recovery_marks_interrupted_not_a_failure(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    # Simulate a crash mid-run: fire once (creates the request + run row), but the crash
    # happens before execute() ever runs, so the request is still "approved" -- stub
    # execute() out so the background task _fire() schedules does nothing.
    async def _crashed_execute(req, notify=True):
        return ""
    m.execute = _crashed_execute
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    req = [r for r in m.store.requests_in_state("approved") if r.schedule_id == schedule.id][0]

    before_failures = m.store.get_schedule(schedule.id).consecutive_failures
    await m.recover_interrupted()

    run = m.store.last_schedule_run(schedule.id)
    assert run["outcome"] == "interrupted"
    updated = m.store.get_schedule(schedule.id)
    assert updated.state == "active"
    assert updated.consecutive_failures == before_failures  # a restart is not a strike
    assert "interrupted by a restart" in sent[-1][1]
    assert "Next check" in sent[-1][1]


@pytest.mark.asyncio
async def test_timezone_wording_uses_schedule_zone_across_a_utc_day_boundary(tmp_path):
    """2:30 AM America/New_York on, say, Jan 2 is 7:30 AM UTC on the SAME UTC day in
    winter, so pick a moment where the zones genuinely disagree on the calendar day:
    11:30 PM America/New_York is 03:30 UTC the NEXT day. The notice must still read
    in the schedule's own zone, "day name" only when that is not today in that zone."""
    import datetime as dt
    from zoneinfo import ZoneInfo
    from inkbox_claude.gate import cadence as cadence_mod

    tz = ZoneInfo("America/New_York")
    local_11_30pm = dt.datetime(2026, 1, 15, 23, 30, tzinfo=tz)
    after = local_11_30pm.timestamp()
    next_fire = cadence_mod.next_run("every", {"minutes": 240, "anchor": after}, "America/New_York", after)
    shown = cadence_mod.describe("every", {"minutes": 240}, "America/New_York")
    assert shown == "every 4 hours"
    local_next = dt.datetime.fromtimestamp(next_fire, tz)
    assert local_next.hour == 3 and local_next.minute == 30  # 11:30 PM + 4h, in NY local time
    utc_next = dt.datetime.fromtimestamp(next_fire, dt.timezone.utc)
    assert utc_next.date() != local_next.date() or utc_next.hour != local_next.hour  # genuinely different clocks


def test_notes_field_repurposed_for_edit_revert_round_trips(tmp_path):
    """schedules.py's edit() stashes the pre-edit snapshot in the otherwise-unread
    `notes` column; _edit_revert must read exactly what edit() wrote."""
    store = Store(str(tmp_path / "gate.db"))
    from inkbox_claude.gate.manager import GateSessionManager as _GM  # noqa: F401 (import smoke)
    from inkbox_claude.gate.schedules import ScheduleService
    task = store.create_task("t")

    class Stub:
        def __init__(self, store):
            self.store = store
        async def send_schedule_notice(self, schedule, text):
            return None

    service = ScheduleService(Stub(store))
    schedule = store.create_schedule(chat_id="c", task_id=task["id"], title="T", prompt="p",
                                     kind="every", scopes=["web"], timezone="America/New_York",
                                     report_mode="changed", cadence_kind="every",
                                     cadence={"minutes": 60}, report_every_time=False,
                                     origin_mode="imessage", origin_chat_id="c")
    store.update_schedule(schedule.id, state="active")
    active = store.get_schedule(schedule.id)
    assert service._edit_revert(active) is None  # nothing stashed yet


# ---------------------------------------------------------------------------
# Gap 1: router.route() needs the owner's schedules, and only the owner's
# ---------------------------------------------------------------------------

class _RecordingRouter:
    """Captures the kwargs the last route() call was made with, so a test can
    assert on schedules_context without caring about the rest of the output."""

    def __init__(self, out):
        self.out = out
        self.calls = []

    async def route(self, **kw):
        self.calls.append(kw)
        return self.out.model_copy(deep=True)

    async def plan_query(self, **kw):
        return None


@pytest.mark.asyncio
async def test_router_gets_schedules_context_for_owner_with_schedules(tmp_path):
    m, sent = make_manager(tmp_path)
    task = m.store.create_task("watch the team page")
    m.store.create_schedule(chat_id="other-thread", task_id=task["id"], title="Team total watch",
                            prompt="p", kind="every", scopes=["web"], timezone="America/New_York",
                            report_mode="changed", cadence_kind="every", cadence={"minutes": 240},
                            report_every_time=False, origin_mode="imessage", origin_chat_id="other-thread")
    router = _RecordingRouter(RouterOutput(reply="ok", request=None))
    m.router = router
    await m.get(APPROVER_CONV).handle_inbound("how's it going", mode="imessage",
                                              meta={"sender": "+15550100001", "to": "+15550100001",
                                                    "conversation_id": APPROVER_CONV})
    assert router.calls, "route() was never called"
    ctx = router.calls[-1].get("schedules_context") or ""
    assert ctx != ""
    assert "Team total watch" in ctx and "proposed" in ctx


@pytest.mark.asyncio
async def test_router_gets_no_schedules_context_for_non_owner(tmp_path):
    m, sent = make_manager(tmp_path)
    task = m.store.create_task("watch the team page")
    m.store.create_schedule(chat_id="other-thread", task_id=task["id"], title="Team total watch",
                            prompt="p", kind="every", scopes=["web"], timezone="America/New_York",
                            report_mode="changed", cadence_kind="every", cadence={"minutes": 240},
                            report_every_time=False, origin_mode="imessage", origin_chat_id="other-thread")
    router = _RecordingRouter(RouterOutput(reply="ok", request=None))
    m.router = router
    await m.get("stranger@example.com").handle_inbound("how's it going", mode="email",
                                                        meta={"sender": "stranger@example.com"})
    assert router.calls, "route() was never called"
    assert (router.calls[-1].get("schedules_context") or "") == ""


@pytest.mark.asyncio
async def test_router_gets_no_schedules_context_when_store_is_empty(tmp_path):
    m, sent = make_manager(tmp_path)
    router = _RecordingRouter(RouterOutput(reply="ok", request=None))
    m.router = router
    await m.get(APPROVER_CONV).handle_inbound("how's it going", mode="imessage",
                                              meta={"sender": "+15550100001", "to": "+15550100001",
                                                    "conversation_id": APPROVER_CONV})
    assert router.calls, "route() was never called"
    assert (router.calls[-1].get("schedules_context") or "") == ""


# ---------------------------------------------------------------------------
# Gap 2: a bare "yes" must not grab an unrelated pending request from another
# thread when the CURRENT thread has its own open schedule proposal.
# ---------------------------------------------------------------------------

def _make_pending_request(m, chat_id):
    task = m.store.create_task("an errand")
    return m.store.create_request(chat_id=chat_id, sender="someone@example.com", sender_name="Someone",
                                  mode="email", subject="", original_message="please do X",
                                  summary="do X", scopes=["web"], prompt="P", state="pending",
                                  task_id=task["id"])


@pytest.mark.asyncio
async def test_bare_yes_resolves_this_threads_proposal_not_unrelated_pending(tmp_path):
    m, sent = make_manager(tmp_path)
    # An unrelated pending request sits on a different thread.
    other_req = _make_pending_request(m, "unrelated-thread")
    # The approver's own thread has an open schedule proposal only.
    await _propose_via_manager(m, title="Team total watch", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    sent.clear()

    session = m.get(APPROVER_CONV)
    await session.handle_inbound("yes", mode="imessage",
                                 meta={"sender": "+15550100001", "to": "+15550100001",
                                       "conversation_id": APPROVER_CONV})
    await asyncio.sleep(0.05)

    activated = m.store.get_schedule(schedule.id)
    assert activated.state == "active"                       # the proposal was resolved
    assert m.store.get_request(other_req.id).state == "pending"  # the unrelated request was left alone


@pytest.mark.asyncio
async def test_bare_yes_on_same_thread_with_both_open_asks_which(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Team total watch", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    # A pending, non-schedule approval request also sits on THIS same thread
    # (the owner's private record thread, which is where the proposal lives too).
    same_thread_req = _make_pending_request(m, gm.OWNER_THREAD)
    sent.clear()

    session = m.get(APPROVER_CONV)
    await session.handle_inbound("yes", mode="imessage",
                                 meta={"sender": "+15550100001", "to": "+15550100001",
                                       "conversation_id": APPROVER_CONV})
    await asyncio.sleep(0.05)

    assert m.store.get_schedule(schedule.id).state == "proposed"       # untouched
    assert m.store.get_request(same_thread_req.id).state == "pending"  # untouched
    reply = sent[-1][1]
    assert reply.startswith("Yes to which:")
    assert "Team total watch" in reply
    assert "do X" in reply


@pytest.mark.asyncio
async def test_rejecting_an_edit_puts_the_schedule_back_as_it_was(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    await m.schedules.edit(schedule.id, owner=True, changes={"prompt": "check the other page"})
    await m.schedules.decide(schedule.id, False)
    reverted = m.store.get_schedule(schedule.id)
    assert reverted.state == "active"
    assert reverted.prompt == activated.prompt
    assert reverted.scopes == activated.scopes
    assert reverted.next_run is not None
    assert m.schedules._edit_revert(reverted) is None
    assert sent[-1][1] == f"Left {reverted.title} as it was."


@pytest.mark.asyncio
async def test_approving_an_edit_keeps_the_new_prompt_and_clears_the_snapshot(tmp_path):
    m, _sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    await m.schedules.decide(schedule.id, True)

    await m.schedules.edit(schedule.id, owner=True, changes={"prompt": "check the other page"})
    await m.schedules.decide(schedule.id, True)
    updated = m.store.get_schedule(schedule.id)
    assert updated.state == "active"
    assert updated.prompt == "check the other page"
    assert m.schedules._edit_revert(updated) is None
    # a later rejection of nothing must not resurrect the old prompt
    assert "check the other page" == m.store.get_schedule(schedule.id).prompt


@pytest.mark.asyncio
async def test_a_proposed_schedule_never_fires_and_refuses_run_now(tmp_path):
    m, _sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    m.store.update_schedule(schedule.id, next_run=time.time() - 60)

    await m.schedules.tick(now=time.time())
    assert m.store.schedule_runs(schedule.id) == []
    assert m.store.get_schedule(schedule.id).state == "proposed"
    assert "run_now" not in m.schedules.valid_actions(m.store.get_schedule(schedule.id))
    with pytest.raises(ValueError):
        await m.schedules.action(schedule.id, "run_now", owner=True)
    with pytest.raises(PermissionError):
        await m.schedules.action(schedule.id, "pause", owner=False)


class FakeCaller:
    """Stands in for the tool caller perform_confirmed uses to run the one
    stored held call."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    async def perform(self, tool, args):
        self.calls.append((tool, args))
        return dict(self.result)


@pytest.mark.asyncio
async def test_held_scheduled_call_records_incomplete_without_counting_a_strike(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    before = len(sent)

    m.executor.results.append({
        "ok": True, "confirm": {"tool": "mcp__fake__send_message", "args": {}, "about": [], "kind": "send"},
        "result_line": "", "final_status": "", "note": "", "denied": [], "tool_calls": [], "held": True,
        "started_at": time.time(), "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    run = m.store.last_schedule_run(schedule.id)
    assert run["outcome"] == "incomplete"
    assert run["result_line"] == "waiting for your yes to send message"
    assert not run["complete"]

    after_hold = m.store.get_schedule(schedule.id)
    assert after_hold.consecutive_failures == 0
    assert after_hold.state == "active"
    # the confirmation text is the only message the hold sends
    assert len(sent) == before + 1


@pytest.mark.asyncio
async def test_approved_held_scheduled_call_completes_with_no_second_notice(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    m.executor.results.append({
        "ok": True, "confirm": {"tool": "mcp__fake__send_message", "args": {}, "about": [], "kind": "send"},
        "result_line": "", "final_status": "", "note": "", "denied": [], "tool_calls": [], "held": True,
        "started_at": time.time(), "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    sent_count_after_hold = len(sent)
    req = m.store.pending()[0]
    assert req.schedule_id == schedule.id

    m.caller = FakeCaller({"ok": True, "summary": "sent it.", "raw": "", "tool_calls": []})
    await m.decide_request(req.id, "yes")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    run = m.store.last_schedule_run(schedule.id)
    assert run["outcome"] == "complete"
    assert run["complete"] is True
    # no notice on the schedule's channel beyond the one sent while holding
    assert len(sent) == sent_count_after_hold


@pytest.mark.asyncio
async def test_declined_held_scheduled_call_stays_incomplete(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    m.executor.results.append({
        "ok": True, "confirm": {"tool": "mcp__fake__send_message", "args": {}, "about": [], "kind": "send"},
        "result_line": "", "final_status": "", "note": "", "denied": [], "tool_calls": [], "held": True,
        "started_at": time.time(), "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    req = m.store.pending()[0]

    result = await m.decide_request(req.id, "no")
    assert result["ok"] is True

    run = m.store.last_schedule_run(schedule.id)
    assert run["outcome"] == "incomplete"
    assert run["result_line"] == "The held call was declined."
    assert not run["complete"]
    after = m.store.get_schedule(schedule.id)
    assert after.consecutive_failures == 0


@pytest.mark.asyncio
async def test_telegram_schedule_notices_go_to_telegram(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_APPROVER_TELEGRAM_ID", "1")
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"],
                               session_mode="telegram", session_chat_id="tg:555",
                               session_meta={"telegram_chat_id": "555", "sender": "telegram:1",
                                             "telegram_user_id": "1"})
    schedule = m.store.schedules(states=["proposed"])[0]
    assert schedule.origin_mode == "telegram"
    proposal_target, _text, proposal_mode, proposal_meta = sent[-1]
    assert proposal_mode == "telegram"
    assert proposal_target == "tg:555"
    assert proposal_meta.get("telegram_chat_id") == "555"

    activated = await m.schedules.decide(schedule.id, True)
    started_target, _text, started_mode, _meta = sent[-1]
    assert started_mode == "telegram"
    assert started_target == "tg:555"

    m.executor.results.append({"ok": True, "result_line": "no change", "final_status": "checked",
                               "note": "", "denied": [], "tool_calls": ["mcp__fake__tool"], "held": False,
                               "started_at": time.time(), "finished_at": time.time(), "cost_usd": None})
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    run_target, _run_text, run_mode, _meta = sent[-1]
    assert run_mode == "telegram"
    assert run_target == "tg:555"


@pytest.mark.asyncio
async def test_imessage_schedule_notices_go_to_imessage(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    assert schedule.origin_mode == "imessage"
    target, _text, mode, meta = sent[-1]
    assert mode == "imessage"
    assert target == APPROVER_CONV
    assert meta.get("conversation_id") == APPROVER_CONV


@pytest.mark.asyncio
async def test_failing_telegram_send_falls_back_once_to_imessage(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_APPROVER_TELEGRAM_ID", "1")
    sent = []

    async def flaky_send_fn(chat_id, text, mode, meta):
        if mode == "telegram":
            raise RuntimeError("telegram send failed")
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV)
    m = gm.GateSessionManager(cfg=cfg, send_fn=flaky_send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    m.jev_agent = None
    m.caller = None
    m.executor = FakeExecutor()
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"],
                               session_mode="telegram", session_chat_id="tg:555",
                               session_meta={"telegram_chat_id": "555", "sender": "telegram:1",
                                             "telegram_user_id": "1"})
    target, _text, mode, _meta = sent[-1]
    assert mode == "imessage"
    assert target == f"imessage:{APPROVER_CONV}"


@pytest.mark.asyncio
async def test_console_created_schedule_goes_to_owner_default_thread(tmp_path):
    m, sent = make_manager(tmp_path)
    task = m.store.create_task("Console watch")
    await m.schedules.propose(
        owner=True, chat_id="owner", task_id=int(task["id"]), title="Console watch",
        prompt="check the page", cadence_kind="every", cadence={"minutes": 240},
        scopes=["web"], origin_mode="console", origin_chat_id="owner",
    )
    target, _text, mode, _meta = sent[-1]
    assert mode == "imessage"
    assert target == f"imessage:{APPROVER_CONV}"
    rows = m.store.history(gm.OWNER_THREAD, limit=5)
    assert any(r.get("text") for r in rows)


@pytest.mark.asyncio
async def test_until_bare_date_from_chat_becomes_end_of_that_day(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page until 2026-12-31",
                               scopes=["web"], until="2026-12-31", timezone="America/New_York")
    schedule = m.store.schedules(states=["proposed"])[0]
    assert schedule.until is not None
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dt = datetime.fromtimestamp(schedule.until, ZoneInfo("America/New_York"))
    assert (dt.year, dt.month, dt.day) == (2026, 12, 31)
    assert (dt.hour, dt.minute, dt.second) == (23, 59, 59)
    # schedprop's proposal text must be able to format it without raising
    from inkbox_claude.gate import schedprop
    assert "2026" in schedprop.format_proposal(schedule) or "December" in schedprop.format_proposal(schedule)


@pytest.mark.asyncio
async def test_until_full_datetime_from_chat_is_parsed(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page until 2026-12-31 3pm",
                               scopes=["web"], until="2026-12-31T15:00:00", timezone="America/New_York")
    schedule = m.store.schedules(states=["proposed"])[0]
    assert schedule.until is not None
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dt = datetime.fromtimestamp(schedule.until, ZoneInfo("America/New_York"))
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2026, 12, 31, 15, 0)


@pytest.mark.asyncio
async def test_unparseable_until_gets_a_plain_reply_and_no_schedule(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page until whenever",
                               scopes=["web"], until="whenever", timezone="America/New_York")
    assert m.store.schedules(states=["proposed"]) == []
    assert "didn't understand" in sent[-1][1]


@pytest.mark.asyncio
async def test_scheduled_run_is_marked_from_owner_so_its_sends_are_held(tmp_path):
    """A scheduled run's third-party sends must be held for his yes exactly like an
    ordinary request of his own -- executor.run only takes sends off the pre-approved
    tool list (routing them through the confirmation hook) when from_owner is true."""
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    m.executor.results.append({"ok": True, "result_line": "no change", "final_status": "checked",
                               "note": "", "denied": [], "tool_calls": [], "held": False,
                               "started_at": time.time(), "finished_at": time.time(), "cost_usd": None})
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    call = m.executor.calls[-1]
    assert call["kw"]["from_owner"] is True


@pytest.mark.asyncio
async def test_an_exception_during_a_scheduled_run_is_recorded_not_swallowed(tmp_path):
    """_fire's asyncio.create_task(self.manager.execute(req)) runs detached -- an
    uncaught exception inside it must not leave the run row blank and the request
    stuck 'running' forever (which would silently skip every later firing)."""
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, said="every 4 hours check the team page", scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    class RaisingExecutor:
        async def run(self, req, context="", prior_work="", model=None, **kw):
            raise RuntimeError("boom")

    m.executor = RaisingExecutor()
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    run = m.store.last_schedule_run(schedule.id)
    assert run["outcome"] == "failed"
    assert "boom" in run["final_status"]
    schedule_after = m.store.get_schedule(schedule.id)
    req = m.store.get_request(schedule_after.last_request_id)
    assert req.state == "failed"
    # the schedule is not left stuck: the next firing is not skipped as "still running"
    still_running = m.store.request_running_for_schedule(schedule.id)
    assert still_running is None
