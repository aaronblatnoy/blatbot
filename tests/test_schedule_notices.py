"""The notices spec (second version), end to end: the blocked outcome, the
exact shape of each notice, the fix-phrase round trip through the real
command handler, and the digest/watch report-every-time default. Builds on
the same real-stack pattern as test_schedule_flow.py (real GateSessionManager,
ScheduleService, schedprop, cadence, store; only the router's model output,
the executor's Claude call, and outbound transport are faked), so every
wording assertion here is against the real seams, not a stand-in."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate import schedcmd
from inkbox_claude.gate import schedrun
from inkbox_claude.gate.router import RouterOutput, ScheduleCommand, ScheduleProposal
from inkbox_claude.gate.store import Store

APPROVER_CONV = "conv-aaron"


class FakeExecutor:
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
    m.jev_agent = None
    m.caller = None
    m.executor = FakeExecutor()
    return m, sent


async def _propose_via_manager(m, *, title="Team chat digest",
                               prompt="every morning send me a summary of the team chat",
                               said="every morning send me a summary of the team chat",
                               cadence_kind="daily", cadence=None, scopes=None,
                               report_every_time=None, start_now=False, until=None, timezone=None):
    task = m.store.create_task(title)
    proposal = ScheduleProposal(title=title, prompt=prompt, cadence_kind=cadence_kind,
                                cadence=cadence or {"time": "08:00"}, timezone=timezone,
                                report_every_time=report_every_time, until=until,
                                scopes=scopes or [], start_now=start_now)
    out = RouterOutput(reply="", request=None, schedule=proposal)

    class _Router:
        async def route(self, **kw):
            return out.model_copy(deep=True)

        async def plan_query(self, **kw):
            return None

    m.router = _Router()
    meta = {"sender": "+15550100001", "to": "+15550100001", "conversation_id": APPROVER_CONV}
    session = m.get(APPROVER_CONV)
    await session.handle_inbound(said, mode="imessage", meta=meta)


def _command(action, target=None, **changes):
    return ScheduleCommand(action=action, target=target, changes=changes)


class _Session:
    def __init__(self, chat_id=APPROVER_CONV, inbound_id=None):
        self.chat_id = chat_id
        self.record_id = chat_id
        self.inbound_id = inbound_id

    def is_approver(self):
        return True

    async def send_to_sender(self, text, role="answer"):
        return True


# ---------------------------------------------------------------------------
# schedrun.parse_closing: BLOCKED and NEEDS
# ---------------------------------------------------------------------------

def test_parse_closing_blocked_and_needs():
    text = ("RESULT: I could not find a team chat in any reachable iMessage, SMS, or Telegram source, "
            "so no summary could be produced today.\n"
            "NEEDS: a way to read Telegram groups\n"
            "STATUS: BLOCKED")
    parsed = schedrun.parse_closing(text)
    assert parsed["status"] == "BLOCKED"
    assert parsed["blocked"] is True
    assert parsed["stop"] is False
    assert parsed["needs"] == "a way to read Telegram groups"
    assert parsed["result_line"].startswith("I could not find a team chat")
    assert "NEEDS" not in parsed["final_status"] and "STATUS" not in parsed["final_status"]


def test_parse_closing_needs_absent_when_not_blocked():
    parsed = schedrun.parse_closing("RESULT: all clear.\nSTATUS: OK")
    assert parsed["needs"] == ""
    assert parsed["blocked"] is False


# ---------------------------------------------------------------------------
# End to end: blocked pauses, clears next_run, never strikes, never retries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocked_run_pauses_clears_next_run_and_gives_the_exact_notice(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, scopes=["inbox_read"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    m.executor.results.append({
        "ok": False, "result_line": "I could not find a team chat in any reachable iMessage, SMS, or "
                                    "Telegram source, so no summary could be produced today.",
        "final_status": "", "note": "", "needs": "a way to read Telegram groups",
        "denied": [], "tool_calls": ["mcp__fake__tool"], "tool_results": [], "held": False,
        "blocked": True, "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    updated = m.store.get_schedule(schedule.id)
    assert updated.state == "paused"
    assert updated.next_run is None
    assert updated.consecutive_failures == 0  # blocked never counts toward three strikes
    run = m.store.last_schedule_run(schedule.id)
    assert run["outcome"] == "blocked"

    notice = sent[-1][1]
    assert notice == (
        'Team chat digest could not run: it could not find a team chat in any reachable iMessage, SMS, or '
        'Telegram source, so no summary could be produced today. Say "let it read your Telegram chats" and '
        'I will run it now, or "stop it". I won\'t retry on my own.'
    )

    # A later tick must not refire it on its own -- it stays paused with no
    # next_run, so due_schedules() never selects it again.
    before = len(m.executor.calls)
    await m.schedules.tick(now=time.time() + 24 * 3600)
    await asyncio.sleep(0)
    assert len(m.executor.calls) == before
    assert m.store.get_schedule(schedule.id).state == "paused"


@pytest.mark.asyncio
async def test_blocked_run_offers_a_real_capability_when_its_needs_names_one(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Price watch", prompt="watch the price and flag it to the team",
                               said="watch the price and flag it to the team",
                               cadence_kind="every", cadence={"minutes": 240}, scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)

    m.executor.results.append({
        "ok": False, "result_line": "I was unable to update the contact note.",
        "final_status": "", "note": "", "needs": "a way to look up and edit contacts",
        "denied": [], "tool_calls": ["mcp__fake__tool"], "tool_results": [], "held": False,
        "blocked": True, "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    notice = sent[-1][1]
    assert 'Say "let it look up and edit contacts" and I will run it now, or "stop it".' in notice
    assert "it was unable to update the contact note" in notice

    # The exact phrase offered must be accepted by the real command handler,
    # and must run exactly once more (no second "Done." message).
    said_id = m.store.add_message("owner", "inbound", "let it look up and edit contacts", "imessage")
    session = _Session(inbound_id=said_id)
    reply = await schedcmd.handle(m, session, _command(
        "add_capability", target="Price watch", scopes=["contacts"]))
    assert reply == ""
    updated = m.store.get_schedule(schedule.id)
    assert "contacts" in updated.scopes
    assert updated.state == "active"
    assert updated.next_run is not None and updated.next_run <= time.time() + 1


# ---------------------------------------------------------------------------
# Notice shapes, asserted whole-string, one case each
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_notice_whole_string(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Price watch", prompt="watch the price and alert me if it changes",
                               said="watch the price and alert me if it changes",
                               cadence_kind="every", cadence={"minutes": 240}, scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    m.executor.results.append({"ok": True, "result_line": "the price dropped to $9.", "final_status": "checked.",
                               "note": "", "denied": [], "tool_calls": ["mcp__fake__tool"], "tool_results": [
                                   {"tool": "mcp__fake__tool", "ok": True}],
                               "held": False, "changed": True, "finished_at": time.time(), "cost_usd": None})
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    updated = m.store.get_schedule(schedule.id)
    next_text = schedcmd._clock_text(updated.next_run, updated.timezone)
    assert sent[-1][1] == f"Price watch: the price dropped to $9. Next check {next_text}."


@pytest.mark.asyncio
async def test_incomplete_notice_whole_string(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Contact note watch",
                               prompt="watch the page and update the contact note if it changes",
                               said="watch the page and update the contact note if it changes",
                               cadence_kind="every", cadence={"minutes": 240}, scopes=["web", "contacts"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    m.executor.results.append({
        "ok": True, "result_line": "it is not allowed to look up and edit contacts, so the note was not written.",
        "final_status": "", "note": "", "denied": [{"tool": "mcp__inkbox__inkbox_update_contact", "reason": "scope"}],
        "tool_calls": ["mcp__fake__tool"], "tool_results": [{"tool": "mcp__fake__tool", "ok": True}],
        "held": False, "changed": True, "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    assert sent[-1][1] == (
        "Contact note watch could not finish: it is not allowed to look up and edit contacts. "
        "it is not allowed to look up and edit contacts, so the note was not written. "
        'Say "let it look up and edit contacts" and I will run it again, or "stop it".'
    )


@pytest.mark.asyncio
async def test_failed_notice_whole_string_first_miss(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Price watch", prompt="watch the price and alert me if it changes",
                               said="watch the price and alert me if it changes",
                               cadence_kind="every", cadence={"minutes": 240}, scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    m.executor.results.append({"ok": False, "error": "the page timed out", "result_line": "",
                               "final_status": "the page timed out", "note": "", "denied": [],
                               "tool_calls": ["mcp__fake__tool"], "held": False, "finished_at": time.time(),
                               "cost_usd": None})
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    updated = m.store.get_schedule(schedule.id)
    when = schedcmd._clock_text(updated.next_run, updated.timezone)
    assert sent[-1][1] == f"Price watch could not finish: it took too long. I will try again at {when}."


@pytest.mark.asyncio
async def test_paused_after_three_notice_whole_string(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Price watch", prompt="watch the price and alert me if it changes",
                               said="watch the price and alert me if it changes",
                               cadence_kind="every", cadence={"minutes": 240}, scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    for _ in range(3):
        m.executor.results.append({"ok": False, "error": "the page timed out", "result_line": "",
                                   "final_status": "the page timed out", "note": "", "denied": [],
                                   "tool_calls": ["mcp__fake__tool"], "held": False, "finished_at": time.time(),
                                   "cost_usd": None})
        await m.schedules.tick(now=activated.next_run)
        await asyncio.sleep(0)
        activated = m.store.get_schedule(schedule.id)
    assert sent[-1][1] == ('Price watch is paused after three runs that could not finish. '
                           'Say "start it again" when it is fixed.')


# ---------------------------------------------------------------------------
# Every quoted phrase offered in a notice is accepted by the real handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_it_phrase_is_accepted_after_a_blocked_notice(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, scopes=["inbox_read"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    m.executor.results.append({
        "ok": False, "result_line": "I could not find a team chat.", "final_status": "",
        "note": "", "needs": "a way to read Telegram groups", "denied": [], "tool_calls": ["mcp__fake__tool"],
        "tool_results": [], "held": False, "blocked": True, "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)

    session = _Session()
    reply = await schedcmd.handle(m, session, _command("stop", target="Team chat digest"))
    assert m.store.get_schedule(schedule.id).state == "ended"
    assert 'Stopped Team chat digest.' in reply


@pytest.mark.asyncio
async def test_let_it_do_that_phrase_is_accepted_after_an_unlabeled_incomplete_notice(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Contact note watch",
                               prompt="watch the page and update the contact note if it changes",
                               said="watch the page and update the contact note if it changes",
                               cadence_kind="every", cadence={"minutes": 240}, scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    activated = await m.schedules.decide(schedule.id, True)
    # mcp__tamid-drive__manage_doc_tab belongs to no known scope in the
    # registry, so the notice offers the vague "let it do that" phrase.
    m.executor.results.append({
        "ok": True, "result_line": "the total did change.", "final_status": "", "note": "",
        "denied": [{"tool": "mcp__tamid-drive__manage_doc_tab", "reason": "scope"}],
        "tool_calls": ["mcp__fake__tool"], "tool_results": [{"tool": "mcp__fake__tool", "ok": True}],
        "held": False, "changed": True, "finished_at": time.time(), "cost_usd": None,
    })
    await m.schedules.tick(now=activated.next_run)
    await asyncio.sleep(0)
    assert 'Say "let it do that" and I will run it again, or "stop it".' in sent[-1][1]

    said_id = m.store.add_message("owner", "inbound", "let it do that", "imessage")
    session = _Session(inbound_id=said_id)
    reply = await schedcmd.handle(m, session, _command("add_capability", target="Contact note watch"))
    # Nothing in the registry maps to the denied tool, so there is nothing
    # concrete to grant -- the real handler still has to give a coherent
    # answer rather than raising or silently doing nothing.
    assert reply == "What should it be allowed to do?"


# ---------------------------------------------------------------------------
# Reason normalization: the incident's exact RESULT sentence
# ---------------------------------------------------------------------------

def test_incident_reason_normalizes_without_i_could_not_or_double_dots():
    from inkbox_claude.gate.schedules import _normalize_reason
    reason = _normalize_reason(
        "I could not find a team chat in any reachable iMessage, SMS, or Telegram source, "
        "so no summary could be produced today.")
    assert reason == ("it could not find a team chat in any reachable iMessage, SMS, or Telegram source, "
                      "so no summary could be produced today")
    assert ".." not in reason
    assert not reason.startswith("I ")


@pytest.mark.parametrize("line,expected_start", [
    ("I couldn't reach the page.", "it couldn't reach the page"),
    ("I can't send that.", "it can't send that"),
    ("I cannot verify the total.", "it cannot verify the total"),
    ("I was unable to open the sheet.", "it was unable to open the sheet"),
    ("Telegram access failed.", "Telegram access failed"),  # a name stays capitalized
])
def test_normalize_reason_rewrites_first_person_and_keeps_names(line, expected_start):
    from inkbox_claude.gate.schedules import _normalize_reason
    assert _normalize_reason(line) == expected_start


@pytest.mark.parametrize("line", ["", ".", "   ", "..."])
def test_normalize_reason_never_blank_or_punctuation_only(line):
    from inkbox_claude.gate.schedules import _normalize_reason
    reason = _normalize_reason(line)
    assert reason == "it did not say why"
    assert ".." not in reason
    assert reason.strip() == reason


# ---------------------------------------------------------------------------
# Digest vs watch report-every-time default, through the real proposal path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_digest_wording_proposal_hears_every_time(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Team chat digest",
                               prompt="every morning send me a summary of the team chat",
                               said="every morning send me a summary of the team chat",
                               cadence_kind="daily", cadence={"time": "08:00"}, scopes=["inbox_read"])
    schedule = m.store.schedules(states=["proposed"])[0]
    assert schedule.report_every_time is True
    assert "every time it runs" in sent[-1][1]


@pytest.mark.asyncio
async def test_watch_wording_proposal_hears_only_on_change(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Price watch", prompt="watch the price and alert me if it changes",
                               said="watch the price and alert me if it changes",
                               cadence_kind="every", cadence={"minutes": 240}, scopes=["web"])
    schedule = m.store.schedules(states=["proposed"])[0]
    assert schedule.report_every_time is False
    assert "only when something changed or it could not finish" in sent[-1][1]


# ---------------------------------------------------------------------------
# No chat sentence ever says "Waiting"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_and_last_never_say_waiting_across_every_state(tmp_path):
    m, sent = make_manager(tmp_path)
    await _propose_via_manager(m, title="Proposed one", scopes=["inbox_read"])
    proposed = m.store.schedules(states=["proposed"])[0]

    active = await m.schedules.decide(proposed.id, True)
    m.store.update_schedule(active.id, title="Active one")
    paused = m.store.create_schedule(chat_id="owner", task_id=m.store.create_task("x")["id"],
                                     title="Paused one", prompt="p", kind="every", scopes=["web"],
                                     timezone="America/New_York", report_mode="changed", cadence_kind="every",
                                     cadence={"minutes": 240}, report_every_time=False, until=None,
                                     origin_mode="imessage", origin_chat_id="owner")
    m.store.update_schedule(paused.id, state="paused", next_run=None,
                            last_outcome="it could not read that Telegram group")

    session = _Session()
    reply = await schedcmd.handle(m, session, _command("list"))
    assert "waiting" not in reply.lower()
    assert "paused: it could not read that Telegram group" in reply
