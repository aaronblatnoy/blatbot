"""The gateway's decision for one inbound message, as a LangGraph state graph.

Every step between "a message arrived" and "a request exists (or only a reply
goes out)" is a node. Independent judgments run in parallel:

    START -> [pick_task || judge_action] -> join
          -> [write_reply (DeepSeek)  ||  attach_task -> build_request -> [judge_scopes -> inherit_scopes || record_event]]
          -> finalize -> END

The reply writer never gates the request: the task is attached and the request
built (with the message as the task's title if the writer has not named it yet)
while DeepSeek writes the words. finalize merges: filters the reply, applies the
writer's title and summary, and assembles the RouterOutput the session acts on.
GATE_DECIDE_GRAPH=0 runs the session's plain sequence instead.
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
    # the reply writer's outputs (DeepSeek)
    reply: Optional[str]
    task_title: Optional[str]
    task_summary: Optional[str]
    # the request path's outputs (code + Jev)
    task: Optional[Dict[str, Any]]
    request: Any                    # RouterRequest or None
    # result
    out: Any                        # RouterOutput, assembled by finalize


def _s(config: RunnableConfig):
    return config["configurable"]["session"]


def _label(s: Any) -> str:
    return "Aaron, the owner" if s.is_approver() else (s._sender_name() or s._sender())


# ----------------------------------------------------------------------------- judgments (parallel)


async def pick_task(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    pick = await s.m.task_picker.pick(message=state["message"], history=state["prior"],
                                      candidates=s.candidate_tasks(state["message"]), sender_label=_label(s),
                                      router_hint=None, proposed_title="")
    choice = pick.get("choice")
    return {"pick": choice or "", "task_choice": None if choice in (None, "none", "new") else choice}


async def judge_action(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    """Runs alongside pick_task, so it is shown this thread's most recent task as the
    likely one rather than the pick, which is not known yet."""
    s = _s(config)
    recent = s.candidate_tasks(state["message"])[:1]
    act = await s.m.task_picker.judge_action(message=state["message"], history=state["prior"],
                                             task=recent[0] if recent else None, sender_label=_label(s),
                                             router_said_action=None)
    needs = act.get("needs_action")
    p = float(act.get("p") or 0.0)
    if needs is None:
        # Grey band. For the owner, acting costs seconds and not acting costs a re-ask,
        # so lean to action; for anyone else a request means an approval text to Aaron.
        needs = p >= (0.4 if s.is_approver() else 0.5)
    return {"needs_action": bool(needs), "p_action": p}


async def join_judgments(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    logger.info("[gate %s] jev-first: task=%s action=%s (p=%.2f)", s.chat_id,
                state["task_choice"] or ("new" if state["pick"] == "new" else None), state["needs_action"], state["p_action"])
    return {}


def after_join(state: DecideState, config: RunnableConfig) -> List[str]:
    from . import manager as gm
    s = _s(config)
    branches: List[str] = []
    # With a request on the way, the owner gets no written acknowledgement: on a call the
    # voice model holds the line, on iMessage the typing indicator shows the work, and a
    # short line goes out only if the run takes long (see GateSessionManager.execute).
    # Anyone else still hears that Aaron will be asked.
    # The writer still runs (it names the task and writes where it stands); only its
    # acknowledgement is dropped in finalize for the owner's call or iMessage requests.
    if not (state["needs_action"] and state["mode"] == "voice"):
        branches.append("write_reply")
    if state["needs_action"] or gm._names_a_task(state["task_choice"]):
        branches.append("attach_task")
    return branches or ["finalize"]


# ----------------------------------------------------------------------------- the words (DeepSeek)


async def write_reply(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    from .router import RouterOutput
    peek = s._peek_task(RouterOutput(task=state["task_choice"]))
    out = await s.m.router.route(history=state["prior"], message=state["message"], mode=state["mode"],
                                 sender=s._sender(), contact_notes=s._contact_notes(), is_approver=s.is_approver(),
                                 task_memory=state["memory"], found_tasks=state["found"],
                                 action=state["needs_action"], action_task=(peek or {}).get("title") or "")
    return {"reply": out.reply, "task_title": out.task_title, "task_summary": out.task_summary}


# ----------------------------------------------------------------------------- the request (code + Jev)


async def attach_task(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    """Load the picked task or create one. The title is the message for now; the
    reply writer's title, if any, is applied in finalize."""
    s = _s(config)
    from .router import RouterOutput
    out = RouterOutput(task=state["task_choice"] or ("new" if state["pick"] == "new" else None), request=None)
    task = s.m.resolve_task(s, out, message=state["body"])
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
    task = state["task"]
    emails = [e for e in gm._EMAIL.findall(state["message"]) if e.lower() != s._sender().lower()]
    req = RouterRequest(prompt=s.build_task_prompt(state["body"], state["prior"], task),
                        scopes=["web"], summary=str(task.get("title") or state["message"][:100])[:160],
                        counterpart=emails[0] if (s.is_approver() and emails) else None)
    return {"request": req}


async def judge_scopes(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    from .router import RouterOutput
    out = RouterOutput(request=state["request"])
    await s.jev_scopes(out, strict=True)
    return {"request": out.request}


async def inherit_scopes(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    """A follow-up on a task keeps the tools its earlier requests had."""
    s = _s(config)
    from .scopes import SCOPES
    req, task = state["request"], state["task"]
    inherited = [sc for sc in s.m.store.scopes_for_task(task["id"]) if sc in SCOPES]
    if inherited:
        merged = list(req.scopes) + [sc for sc in inherited if sc not in req.scopes]
        if merged != req.scopes:
            logger.info("[gate %s] scopes: +%s inherited from T%s", s.chat_id,
                        [sc for sc in inherited if sc not in req.scopes], task["id"])
        req.scopes = merged
    return {"request": req}


async def record_event(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    s = _s(config)
    from .router import RouterOutput
    # jev_event only reads whether a request exists and the writer's summary (applied later).
    out = RouterOutput(request=state.get("request"), task_summary=None)
    await s.jev_event(state["task"], state["body"], state["prior"], out)
    return {}


# ----------------------------------------------------------------------------- merge


async def finalize(state: DecideState, config: RunnableConfig) -> Dict[str, Any]:
    """Assemble the RouterOutput: filtered reply, request, the writer's title and summary."""
    s = _s(config)
    from . import manager as gm
    from .router import RouterOutput
    out = RouterOutput(reply=state.get("reply"), task=state["task_choice"], task_title=state.get("task_title"),
                       task_summary=state.get("task_summary"), request=state.get("request") if state["needs_action"] else None)
    picker = s.m.task_picker
    verdict = (await picker.judge_reply(reply=out.reply, message=state["message"])
               if out.reply and picker is not None and hasattr(picker, "judge_reply") else {})
    promises = verdict.get("promises_action")
    if promises is None:
        promises = gm._promises_action(out.reply or "")
    is_ack = verdict.get("is_acknowledgement")
    if is_ack is None:
        is_ack = gm._is_acknowledgement(out.reply or "")
    if not state["needs_action"] and out.reply and promises:
        logger.info("[gate %s] reply promised an action but none was decided; rewriting", s.chat_id)
        out2 = await s.m.router.route(history=state["prior"], message=state["message"], mode=state["mode"],
                                      sender=s._sender(), contact_notes=s._contact_notes(), is_approver=s.is_approver(),
                                      task_memory=state["memory"], found_tasks=state["found"], action=False,
                                      action_task="(NOTE: no request has been created and none will run for this "
                                                  "message; do not say you are doing, submitting or checking anything. "
                                                  "Answer from what is known, or ask what is needed.)")
        if out2.reply and not gm._promises_action(out2.reply):
            out.reply = out2.reply
        else:
            # The writer insists the sender wants something done while the judgment said no.
            # Silence reads as being ignored; ask the one question that resolves it.
            logger.info("[gate %s] writer still promised an action; asking instead of going silent", s.chat_id)
            out.reply = "Do you want me to go ahead with that? Say yes and I will."
    if state["needs_action"] and out.reply and s.is_approver() and state["mode"] == "imessage":
        # The typing indicator shows the work. The writer's acknowledgement is kept aside and
        # sent only if the run turns out to be slow (GateSessionManager._ack_if_slow).
        s.deferred_ack = out.reply if is_ack else None
        out.reply = None
    if state["needs_action"] and out.reply and not is_ack:
        logger.info("[gate %s] dropped a %d-word reply written alongside a request; result will follow",
                    s.chat_id, len(out.reply.split()))
        out.reply = None
    task = state.get("task")
    if task is not None:
        out.task = f"T{task['id']}"
        title = " ".join((out.task_title or "").split())
        if title and title != task.get("title") and task.get("title") in (None, "", "Untitled task", " ".join(state["body"].split())[:90]):
            s.m.store.set_task_title(task["id"], title)
            if out.request is not None and out.request.summary == task.get("title"):
                out.request.summary = title[:160]
        if out.task_summary:
            s.m.store.set_task_summary(task["id"], out.task_summary)
    elif state["pick"] == "none" and not state["needs_action"]:
        out.task = None
    return {"out": out}


# ----------------------------------------------------------------------------- graph


def build_graph():
    g = StateGraph(DecideState)
    for name, fn in (("pick_task", pick_task), ("judge_action", judge_action), ("join_judgments", join_judgments),
                     ("write_reply", write_reply), ("attach_task", attach_task), ("build_request", build_request),
                     ("judge_scopes", judge_scopes), ("inherit_scopes", inherit_scopes), ("record_event", record_event),
                     ("finalize", finalize)):
        g.add_node(name, fn)
    g.add_edge(START, "pick_task")
    g.add_edge(START, "judge_action")
    g.add_edge(["pick_task", "judge_action"], "join_judgments")
    g.add_conditional_edges("join_judgments", after_join, ["write_reply", "attach_task", "finalize"])
    g.add_conditional_edges("attach_task", after_attach, {"build_request": "build_request", "record_event": "record_event"})
    g.add_edge("build_request", "judge_scopes")
    g.add_edge("build_request", "record_event")
    g.add_edge("judge_scopes", "inherit_scopes")
    # finalize waits for every branch that ran (LangGraph defers a node until all its
    # active predecessors in the superstep have finished).
    g.add_edge("write_reply", "finalize")
    g.add_edge("inherit_scopes", "finalize")
    g.add_edge("record_event", "finalize")
    g.add_edge("finalize", END)
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
                            "reply": None, "task_title": None, "task_summary": None, "task": None, "request": None,
                            "out": None}
    final = await graph().ainvoke(initial, config={"configurable": {"session": session}})
    return final["out"], final.get("task")
