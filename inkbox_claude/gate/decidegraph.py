"""The gateway's decision for one inbound message, as a LangGraph state graph.

Every step the gateway takes between "a message arrived" and "a request exists
(or only a reply goes out)" is a node, so the flow can be read, traced and
re-wired edge by edge:

    START -> pick_task -> judge_action -> write_reply -> check_reply
          -> (needs action) attach_task -> build_request -> judge_scopes -> inherit_scopes -> record_event -> END
          -> (task named)   attach_task -> record_event -> END
          -> (neither)      END

The nodes call the same typed judgments (Jev) and the same reply writer
(DeepSeek) the session used before; the graph only owns the order and the
state. GATE_DECIDE_GRAPH=0 runs the session's plain sequence instead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


class DecideState(TypedDict, total=False):
    # inputs
    body: str                       # the message as stored
    message: str                    # what the models are shown (voice: the transcript excerpt)
    prior: List[Dict[str, Any]]     # conversation before this message
    mode: str
    memory: str                     # the ledger in view
    found: str                      # tasks surfaced by the lookup
    # judgments
    task_choice: Optional[str]      # "T12" | None (Jev picked new or abstained)
    pick: str                       # raw pick: "T12" | "new" | "none" | ""
    needs_action: bool
    p_action: float
    # outputs
    out: Any                        # RouterOutput
    task: Optional[Dict[str, Any]]


def _s(config: RunnableConfig):
    return config["configurable"]["session"]


async def pick_task(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    label = "Aaron, the owner" if s.is_approver() else (s._sender_name() or s._sender())
    pick = await s.m.task_picker.pick(message=state["message"], history=state["prior"],
                                      candidates=s.candidate_tasks(), sender_label=label,
                                      router_hint=None, proposed_title="")
    choice = pick.get("choice")
    return {"pick": choice or "", "task_choice": None if choice in (None, "none", "new") else choice}


async def judge_action(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    from .router import RouterOutput
    label = "Aaron, the owner" if s.is_approver() else (s._sender_name() or s._sender())
    peek = s._peek_task(RouterOutput(task=state["task_choice"]))
    act = await s.m.task_picker.judge_action(message=state["message"], history=state["prior"], task=peek,
                                             sender_label=label, router_said_action=None)
    needs = act.get("needs_action")
    p = float(act.get("p") or 0.0)
    if needs is None:
        # Grey band. For the owner, acting costs seconds and not acting costs a re-ask,
        # so lean to action; for anyone else a request means an approval text to Aaron.
        needs = p >= (0.4 if s.is_approver() else 0.5)
    logger.info("[gate %s] jev-first: task=%s action=%s (p=%.2f)", s.chat_id,
                state["task_choice"] or ("new" if state["pick"] == "new" else None), needs, p)
    return {"needs_action": bool(needs), "p_action": p}


async def write_reply(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    from .router import RouterOutput
    peek = s._peek_task(RouterOutput(task=state["task_choice"]))
    out = await s.m.router.route(history=state["prior"], message=state["message"], mode=state["mode"],
                                 sender=s._sender(), contact_notes=s._contact_notes(), is_approver=s.is_approver(),
                                 task_memory=state["memory"], found_tasks=state["found"],
                                 action=state["needs_action"], action_task=(peek or {}).get("title") or "")
    out.request = None                    # the router never defines the request on this path
    return {"out": out}


async def check_reply(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    """A reply may not promise work that will not run, and may not answer a
    question whose answer a request will deliver."""
    s = _s(config)
    from . import manager as gm
    out = state["out"]
    if not state["needs_action"] and out.reply and gm._promises_action(out.reply):
        logger.info("[gate %s] reply promised an action but none was decided; rewriting", s.chat_id)
        out2 = await s.m.router.route(history=state["prior"], message=state["message"], mode=state["mode"],
                                      sender=s._sender(), contact_notes=s._contact_notes(), is_approver=s.is_approver(),
                                      task_memory=state["memory"], found_tasks=state["found"], action=False,
                                      action_task="(NOTE: no request has been created and none will run for this "
                                                  "message; do not say you are doing, submitting or checking anything. "
                                                  "Answer from what is known, or ask what is needed.)")
        out.reply = out2.reply if out2.reply and not gm._promises_action(out2.reply) else None
    if state["needs_action"] and out.reply and not gm._is_acknowledgement(out.reply):
        logger.info("[gate %s] dropped a %d-word reply written alongside a request; result will follow",
                    s.chat_id, len(out.reply.split()))
        out.reply = None
    if state["task_choice"] is not None:
        out.task = state["task_choice"]         # Jev's pick wins; router's stands only when Jev abstained
    elif state["pick"] == "none" and not state["needs_action"]:
        out.task = None
    return {"out": out}


def after_check(state: DecideState, config: RunnableConfig) -> str:
    from . import manager as gm
    if state["needs_action"]:
        return "attach_task"
    if gm._names_a_task(state["out"].task):
        return "attach_task"
    return END


async def attach_task(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    task = s.m.resolve_task(s, state["out"])
    body = state["body"] if state["mode"] != "voice" else f"(phone) {state['body']}"
    s._ensure_inbound_on_task(task, body)
    return {"task": task}


def after_attach(state: DecideState, config: RunnableConfig) -> str:
    return "build_request" if state["needs_action"] else "record_event"


async def build_request(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    """Code builds the request from the record: prompt from the source message and
    ledger, summary from the task title. No model paraphrases it."""
    s = _s(config)
    from . import manager as gm
    from .router import RouterRequest
    out, task = state["out"], state["task"]
    emails = [e for e in gm._EMAIL.findall(state["message"]) if e.lower() != s._sender().lower()]
    out.request = RouterRequest(prompt=s.build_task_prompt(state["body"], state["prior"], task),
                                scopes=["web"], summary=str(task.get("title") or state["message"][:100])[:160],
                                counterpart=emails[0] if (s.is_approver() and emails) else None)
    return {"out": out}


async def judge_scopes(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    await s.jev_scopes(state["out"], strict=True)
    return {"out": state["out"]}


async def inherit_scopes(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    """A follow-up on a task keeps the tools its earlier requests had."""
    s = _s(config)
    from .scopes import SCOPES
    out, task = state["out"], state["task"]
    inherited = [sc for sc in s.m.store.scopes_for_task(task["id"]) if sc in SCOPES]
    if inherited:
        merged = list(out.request.scopes) + [sc for sc in inherited if sc not in out.request.scopes]
        if merged != out.request.scopes:
            logger.info("[gate %s] scopes: +%s inherited from T%s", s.chat_id,
                        [sc for sc in inherited if sc not in out.request.scopes], task["id"])
        out.request.scopes = merged
    return {"out": out}


async def record_event(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    await s.jev_event(state["task"], state["body"], state["prior"], state["out"])
    return {}


def build_graph():
    g = StateGraph(DecideState)
    for name, fn in (("pick_task", pick_task), ("judge_action", judge_action), ("write_reply", write_reply),
                     ("check_reply", check_reply), ("attach_task", attach_task), ("build_request", build_request),
                     ("judge_scopes", judge_scopes), ("inherit_scopes", inherit_scopes), ("record_event", record_event)):
        g.add_node(name, fn)
    g.add_edge(START, "pick_task")
    g.add_edge("pick_task", "judge_action")
    g.add_edge("judge_action", "write_reply")
    g.add_edge("write_reply", "check_reply")
    g.add_conditional_edges("check_reply", after_check, {"attach_task": "attach_task", END: END})
    g.add_conditional_edges("attach_task", after_attach, {"build_request": "build_request", "record_event": "record_event"})
    g.add_edge("build_request", "judge_scopes")
    g.add_edge("judge_scopes", "inherit_scopes")
    g.add_edge("inherit_scopes", "record_event")
    g.add_edge("record_event", END)
    return g.compile()


_GRAPH = None


def graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def decide(session: Any, *, body: str, message: str, prior: List[Dict[str, Any]], mode: str,
                 memory: str, found: str):
    """Run one message through the graph. Returns (RouterOutput, task or None)."""
    initial: DecideState = {"body": body, "message": message, "prior": prior, "mode": mode, "memory": memory,
                            "found": found, "task_choice": None, "pick": "", "needs_action": False, "p_action": 0.0,
                            "out": None, "task": None}
    final = await graph().ainvoke(initial, config={"configurable": {"session": session}})
    return final["out"], final.get("task")
