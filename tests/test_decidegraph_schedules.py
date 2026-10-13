"""Schedule commands and proposals through the real decision graph
(inkbox_claude.gate.decidegraph), the live path when GATE_DECIDE_GRAPH is on
(the default) and a task picker is configured. The legacy sequence in
GateSession.decide already threads schedules_context and the router's whole
RouterOutput through untouched; this file guards the graph path specifically,
since it used to drop schedule_command and the schedules_context entirely.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate.router import RouterOutput, ScheduleCommand, ScheduleProposal

APPROVER_CONV = "conv-aaron"


class FakePicker:
    """Stands in for TypeSafe's Jev task picker: always "none" (new task would
    also do), and no tool action is needed -- this file is about the
    schedule_command/schedule path, not the ordinary-request path."""

    async def pick(self, **kw):
        return {"choice": "none"}

    async def judge_action(self, **kw):
        return {"needs_action": False, "p": 0.0}


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
    # Force the graph's Jev-first path on, the way a live TYPESAFE_API_KEY would.
    m.task_picker = FakePicker()
    m.router_defines_requests = False
    return m, sent


def _route_with(out: RouterOutput):
    class _Router:
        async def route(self, **kw):
            assert "schedules_context" in kw, "decidegraph must pass schedules_context to every route() call"
            return out.model_copy(deep=True)

        async def plan_query(self, **kw):
            return None

        async def phrase(self, *a, **kw):
            return ""

    return _Router()


@pytest.mark.asyncio
async def test_list_command_reaches_schedcmd_and_creates_no_request(tmp_path):
    m, sent = make_manager(tmp_path)
    out = RouterOutput(reply=None, request=None,
                       schedule_command=ScheduleCommand(action="list", target=None, changes={}))
    m.router = _route_with(out)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("what do you have scheduled", mode="imessage",
                                 meta={"sender": "+15550100001", "conversation_id": APPROVER_CONV})
    assert m.store.requests_in_state("pending") == []
    assert m.store.requests_in_state("approved") == []
    assert sent, "schedcmd's reply should have gone out"
    reply_text = sent[-1][1]
    assert "schedule" in reply_text.lower() or "nothing" in reply_text.lower() or "have" in reply_text.lower()


@pytest.mark.asyncio
async def test_schedule_proposal_through_the_graph_creates_no_ordinary_request(tmp_path):
    m, sent = make_manager(tmp_path)
    proposal = ScheduleProposal(title="Team total watch", prompt="check the team page", cadence_kind="every",
                               cadence={"minutes": 240}, timezone=None, report_every_time=None, until=None,
                               scopes=["web"], start_now=False)
    out = RouterOutput(reply=None, request=None, schedule=proposal)
    m.router = _route_with(out)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("every 4 hours check the team page", mode="imessage",
                                 meta={"sender": "+15550100001", "conversation_id": APPROVER_CONV})
    assert m.store.requests_in_state("pending") == []
    assert m.store.requests_in_state("approved") == []
    schedules = m.store.schedules(states=["proposed"])
    assert len(schedules) == 1
    assert schedules[0].title == "Team total watch"
