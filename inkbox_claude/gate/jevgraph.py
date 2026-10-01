"""The Jev agent as a LangGraph state graph.

The loop in jevagent.py is expressed as nodes over one typed state, with the two
fan-outs (filling arguments for several tools, calling several tools) as parallel
branches whose results merge back through reducers. Judgments, argument filling,
reading of large results, guards and the status report are the same functions as
before; only the control flow and the state live here.

    judge_done ──done──▶ finish
        │ not done
        ▼
      plan ──unsure/give_up──▶ finish
        │ chosen + extras
        ▼
   fill_one ×N (parallel) ──▶ guard ──confirm──▶ finish
        │ batch                  │ refused/identical ──▶ judge_done
        ▼                        ▼
   call_one ×N (parallel) ──▶ record ──▶ judge_done

Runtime objects (ToolBox, Judge, Prose, the request, the agent) are passed in the
run config, never in the state, so the state stays plain data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import time
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

import os

from . import jevagent as ja
from .scopes import SCOPE_TREE_SYSTEMS, TOOL_DESCRIPTIONS, TOOL_SYSTEM, tools_by_system

logger = logging.getLogger(__name__)


def _merge_round(old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filled arguments accumulate within a round; a new round replaces them."""
    if new and new[0].get("__round__") is not None and old and old[0].get("__round__") != new[0].get("__round__"):
        return list(new)
    return list(old) + list(new)


class AgentState(TypedDict, total=False):
    goal: str
    background: str
    task_context: str
    facts: Dict[str, Any]
    steps: Annotated[List[Dict[str, Any]], operator.add]      # every tool call made, in arrival order
    filled: Annotated[List[Dict[str, Any]], _merge_round]     # this round's (tool, args) after filling
    round: int
    step: int
    p_done: float
    loop_state: Dict[str, Any]
    chosen: Optional[str]
    extras: List[str]
    probs: Dict[str, float]
    excluded: List[str]
    batch: List[Tuple[str, Dict[str, Any]]]
    outcome: Optional[str]          # "ok" | "fail" | "confirm"
    error: str
    confirm: Optional[Dict[str, Any]]
    partial: Optional[float]


def _rt(config: RunnableConfig) -> Dict[str, Any]:
    return config["configurable"]["rt"]


# ----------------------------------------------------------------------------- nodes


async def judge_done(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    judge, req, context = rt["judge"], rt["req"], rt["context"]
    steps = state.get("steps", [])
    loop_state = {"goal": req.original_message, "task_context": context, "background": state.get("background") or "",
                  **ja._now_facts(),
                  "steps_so_far": [{"tool": s["tool"], "args": s["args"], "ok": s["ok"], "result": ja._seen(s)} for s in steps]}
    loop_state = ja._fit(loop_state)
    p_done = 0.0
    if steps:
        p_done = await judge.yes(loop_state, {
            "question": "Can `goal` now be answered, or is the action it asks for complete, WITH CONFIDENCE "
                        "from `steps_so_far` alone: is every fact the answer needs already in the evidence?",
            "criteria": {"true": "Everything the goal asked for has been done or, for a question, the retrieved "
                                 "results contain what is needed to answer it. Counting, filtering or comparing rows "
                                 "that are already retrieved is NOT a further tool call; the answer is written from "
                                 "the results. An empty result from the right place also answers the question.",
                         "false": "Something the goal asked for has not happened yet, or the results retrieved so far "
                                  "do not contain the needed information and a different call is needed."}})
        logger.info("jev graph step %d: p(done)=%.2f", state.get("step", 0) + 1, p_done)
    return {"p_done": p_done, "loop_state": loop_state}


def after_judge(state: AgentState, config: RunnableConfig) -> str:
    rt = _rt(config)
    if state.get("steps") and state["p_done"] >= ja.DONE_MIN:
        return "finish_ok"
    if state.get("step", 0) >= rt["agent"].max_steps:
        return "finish_limit"
    return "plan"


NONE_HERE = "none_of_these"
SYSTEM_MIN = float(os.getenv("JEV_AGENT_SYSTEM_MIN") or 0.25)


def _narrow_reads(options: Dict[str, str], choice: Optional[str], sys_p: Dict[str, float]) -> Dict[str, str]:
    """Read tools worth asking about alongside the pick. Every granted read tool is asked:
    a source can be worth collecting from in parallel even when it is not the single best
    next step, and that is exactly what the per-tool usefulness noul decides."""
    if not choice or choice == ja.GIVE_UP:
        return {}
    return {t: d for t, d in options.items() if t != ja.GIVE_UP}


async def choose_tool(judge: "ja.Judge", loop_state: Any, options: Dict[str, str], step: int
                      ) -> Tuple[Optional[str], float, Dict[str, float], Dict[str, float]]:
    """The next tool, chosen as a tree: first which SYSTEM the next step uses (a yes/no per
    system the task was granted), then which tool within that system (a choice over a
    handful plus none_of_these). One system granted: a flat choice, as before."""
    tools = {t: d for t, d in options.items() if t != ja.GIVE_UP}
    groups = tools_by_system(list(tools))
    rules = ["The gateway delivers the findings to the person who asked; never send them the answer.",
             "Read before you write. A wrong first pick or an empty lookup means try the next likely one.",
             f"Choose {NONE_HERE} only when no tool here is the right next step."]
    if len(groups) <= 1:
        opts = dict(tools)
        opts[ja.GIVE_UP] = options[ja.GIVE_UP]
        rules[-1] = f"Choose {ja.GIVE_UP} only when no tool here can still help."
        ch, cf, pr = await judge.choose(loop_state, {"question": "What is the single best next step toward `goal`?", "rules": rules}, opts)
        return ch, cf, pr, {}
    # One request: a noul per system AND a choice within every system. Code then takes
    # the choice from the likeliest system that did not answer none_of_these.
    questions: Dict[str, Any] = {f"sys::{s}": {"type": "noul", "instructions": {
        "system": SCOPE_TREE_SYSTEMS.get(s, s),
        "tools_here": [t.split("__")[-1] for t in ts][:12],
        "question": "Does the single best NEXT step toward `goal`, given `steps_so_far`, use this system?"},
        "criteria": {"true": "The next call belongs here.", "false": "The next call is elsewhere, or nothing here helps now."}}
        for s, ts in groups.items()}
    for s, ts in groups.items():
        opts = {t: tools[t] for t in ts}
        opts[NONE_HERE] = "None of these is the right next step."
        questions[f"tool::{s}"] = {"type": "choice",
                                   "instructions": {"question": "If the next step uses this system, which tool is it?",
                                                    "rules": rules},
                                   "criteria": opts}
    answers = await judge.ask(loop_state, questions)
    sys_p = {s: float((answers.get(f"sys::{s}") or {}).get("noul") or 0.0) for s in groups}
    ordered = [s for s, p in sorted(sys_p.items(), key=lambda kv: -kv[1]) if p >= SYSTEM_MIN]
    logger.info("jev graph step %d: systems %s", step + 1, [(s, round(sys_p[s], 2)) for s in sorted(sys_p, key=lambda k: -sys_p[k])][:4])
    all_probs: Dict[str, float] = {}
    for s in ordered[:2]:
        a = answers.get(f"tool::{s}") or {}
        probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
        choice = a.get("choice") or (max(probs, key=probs.get) if probs else None)
        conf = probs.get(choice, float(a.get("confidence") or 0.0)) if choice else 0.0
        for t, p in probs.items():
            if t != NONE_HERE:
                all_probs[t] = max(all_probs.get(t, 0.0), p * sys_p[s])
        if choice and choice != NONE_HERE and conf >= getattr(judge, "min_conf", 0.45):
            return choice, conf, all_probs, sys_p
    if not ordered:
        return ja.GIVE_UP, 1.0 - max(sys_p.values(), default=0.0), all_probs, sys_p
    return None, 0.0, all_probs, sys_p


async def plan(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    judge, agent = rt["judge"], rt["agent"]
    steps, excluded = state.get("steps", []), set(state.get("excluded", []))
    loop_state, p_done = state["loop_state"], state.get("p_done", 0.0)
    step = state.get("step", 0)
    options = {k: v for k, v in rt["tool_options"].items() if k not in excluded} or dict(rt["tool_options"])
    if len(steps) >= 2 and steps[-1]["tool"] == steps[-2]["tool"] and len(options) > 1:
        options.pop(steps[-1]["tool"], None)          # two in a row: the next move must differ
    options[ja.GIVE_UP] = "The goal cannot be achieved with these tools or the information available."
    choice, conf, probs, sys_p = await choose_tool(judge, loop_state, options, step)
    useful = await agent._useful_reads(judge, loop_state, _narrow_reads(options, choice, sys_p))
    reads_only = bool(steps) and any(s["ok"] for s in steps) and all(not ja.is_write_tool(s["tool"]) for s in steps if s["ok"])
    if choice is None:
        logger.info("jev graph: unsure; top options %s", sorted(probs.items(), key=lambda kv: -kv[1])[:5])
        if reads_only:
            # Reads exhausted: deliver what was gathered with the confidence stated.
            logger.info("jev graph: reads exhausted at confidence %.2f; delivering findings as partial", p_done)
            return {"outcome": "ok", "chosen": None, "extras": [], "probs": probs, "partial": p_done if p_done < ja.DONE_MIN else None}
        return {"outcome": "fail", "chosen": None, "extras": [], "probs": probs,
                "error": f"insufficient evidence to answer with confidence (done {p_done:.2f}); no further useful step found"}
    if choice == ja.GIVE_UP:
        if reads_only:
            if p_done < ja.DONE_MIN:
                logger.info("jev graph: give_up after reads at confidence %.2f; delivering findings as partial", p_done)
            return {"outcome": "ok", "chosen": None, "extras": [], "probs": probs, "partial": p_done if p_done < ja.DONE_MIN else None}
        return {"outcome": "fail", "chosen": None, "extras": [], "probs": probs,
                "error": "gave up: goal not achievable with the granted tools"}
    extras = []
    if not ja.is_write_tool(choice):
        extras = [t for t, pu in sorted(useful.items(), key=lambda kv: -kv[1])
                  if pu >= ja.PARALLEL_MIN and t != choice and not ja.is_write_tool(t) and t not in excluded][:ja.PARALLEL_MAX]
    return {"outcome": None, "chosen": choice, "extras": extras, "probs": probs, "round": state.get("round", 0) + 1,
            "filled": [{"__round__": state.get("round", 0) + 1, "tool": "__reset__"}]}


def after_plan(state: AgentState, config: RunnableConfig):
    if state.get("outcome"):
        return "finish"
    rnd = state["round"]
    return [Send("fill_one", {"tool": t, "round": rnd, "primary": t == state["chosen"], **_carry(state)})
            for t in [state["chosen"]] + list(state.get("extras", []))]


def _carry(state: AgentState) -> Dict[str, Any]:
    return {"facts": state.get("facts", {}), "steps_view": state.get("steps", [])}


async def fill_one(item: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    args = await rt["agent"]._fill_args(rt["box"], rt["judge"], rt["prose"], item["tool"], item["facts"],
                                        item["steps_view"], rt["req"])
    return {"filled": [{"__round__": item["round"], "tool": item["tool"], "args": args, "primary": item["primary"]}]}


async def guard(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    agent, judge, prose, req, box = rt["agent"], rt["judge"], rt["prose"], rt["req"], rt["box"]
    steps, p_done = state.get("steps", []), state.get("p_done", 0.0)
    filled = [f for f in state.get("filled", []) if f.get("tool") != "__reset__" and f.get("__round__") == state["round"]]
    primary = next((f for f in filled if f["primary"]), None)
    if primary is None or primary["args"] is None:
        # The call cannot be made yet (an id not in evidence): a failed attempt; go collect.
        chosen = state["chosen"]
        if sum(1 for s in steps if s["tool"] == chosen and not s["ok"]) >= 2:
            return {"outcome": "fail", "batch": [], "error": f"could not determine arguments for {chosen}"}
        return {"outcome": None, "batch": [], "step": state.get("step", 0) + 1,
                "steps": [{"tool": chosen, "args": {}, "ok": False, "primary": True,
                           "result": "ERROR: this call needs a value (an id or item) that no gathered result contains yet; "
                                     "look the item up first"}]}
    choice, args = primary["tool"], primary["args"]
    if ja.sends_to_requester(choice, args, agent.protected + [req.sender, req.chat_id]):
        logger.info("jev graph: refusing %s to the requester; finishing with findings", choice)
        return {"outcome": "ok", "batch": []}
    if ja.is_destructive(choice, args):
        about = ja._mentions(steps, args)
        if not about:
            # Never ask the owner to confirm a blind delete: the target must be described by
            # gathered evidence. Record a failed attempt and go collect.
            logger.info("jev graph: %s target not in evidence; looking things up first", choice.split("__")[-1])
            return {"outcome": None, "batch": [], "step": state.get("step", 0) + 1,
                    "steps": [{"tool": choice, "args": args, "ok": False, "primary": True,
                               "result": "ERROR: the target of this change is not identified in any gathered result; "
                                         "look the item up (list or search it) before changing it"}]}
        logger.info("jev graph: %s needs the owner's confirmation; pausing", choice.split("__")[-1])
        return {"outcome": "confirm", "batch": [], "error": "confirmation required",
                "confirm": {"tool": choice, "args": args, "about": about}}
    identical = lambda a: any(s["ok"] and s["tool"] == choice and s["args"] == a for s in steps)
    if identical(args) and p_done < ja.DONE_MIN:
        again = await agent._fill_args(box, judge, prose, choice, state.get("facts", {}), steps, req, avoid=args)
        if again is not None and again != args:
            logger.info("jev graph: re-filled %s to avoid a repeat", choice.split("__")[-1])
            args = again
    if identical(args):
        if p_done >= ja.DONE_MIN:
            logger.info("jev graph: would repeat %s identically; treating as done", choice)
            return {"outcome": "ok", "batch": []}
        logger.info("jev graph: would repeat %s identically at p_done %.2f; excluding it and re-picking", choice, p_done)
        return {"outcome": None, "batch": [], "excluded": list(set(state.get("excluded", [])) | {choice}), "step": state.get("step", 0) + 1}
    batch: List[Tuple[str, Dict[str, Any]]] = [(choice, args)]
    for f in filled:
        if f["primary"] or f["args"] is None:
            continue
        if not any(s["ok"] and s["tool"] == f["tool"] and s["args"] == f["args"] for s in steps):
            batch.append((f["tool"], f["args"]))
    if len(batch) > 1:
        logger.info("jev graph step %d: also collecting %s", state.get("step", 0) + 1, [t.split("__")[-1] for t, _ in batch[1:]])
    return {"outcome": None, "batch": batch}


def after_guard(state: AgentState, config: RunnableConfig):
    if state.get("outcome"):
        return "finish"
    if not state.get("batch"):
        return "judge_done"
    return [Send("call_one", {"tool": t, "args": a, "primary": i == 0}) for i, (t, a) in enumerate(state["batch"])]


async def call_one(item: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    ok, result, evidence, err = await rt["agent"]._call(rt["box"], rt["judge"], item["tool"], item["args"], rt["req"])
    if ok:
        return {"steps": [{"tool": item["tool"], "args": item["args"], "result": result, "ok": True, "evidence": evidence,
                           "primary": item["primary"]}]}
    return {"steps": [{"tool": item["tool"], "args": item["args"], "result": f"ERROR: {err}", "ok": False,
                       "primary": item["primary"], "error": err}]}


async def record(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """After a round of calls: log, place results in the facts, apply the failure rule."""
    steps = state.get("steps", [])
    n = len(state.get("batch", []))
    new = steps[-n:] if n else []
    facts = dict(state.get("facts", {}))
    step_no = state.get("step", 0) + 1
    for s in new:
        short = s["tool"].split("__")[-1]
        shown = json.dumps({k: v for k, v in s["args"].items() if k != "user_google_email"}, ensure_ascii=False)[:300]
        if s["ok"]:
            logger.info("jev graph step %d: %s(%s) -> %s", step_no, short, shown, " ".join(ja._text(s["result"]).split())[:300])
            facts[f"result_of_{short}_{len(steps)}"] = s.get("evidence") or ja._text(s["result"])
        else:
            logger.info("jev graph step %d: %s(%s) -> ERROR %s", step_no, short, shown, " ".join(str(s.get("error")).split())[:300])
    for s in new:
        if s["ok"] or not s.get("primary"):
            continue
        same = sum(1 for x in steps if x["tool"] == s["tool"] and x["args"] == s["args"] and not x["ok"])
        total = sum(1 for x in steps if x["tool"] == s["tool"] and not x["ok"])
        if same >= 2 or total >= 3:
            return {"facts": facts, "step": step_no, "outcome": "fail", "error": f"{s['tool']} keeps failing: {s.get('error')}"}
    return {"facts": facts, "step": step_no, "outcome": None}


def after_record(state: AgentState, config: RunnableConfig) -> str:
    return "finish" if state.get("outcome") else "judge_done"


async def finish(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    return {}


# ----------------------------------------------------------------------------- graph


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("judge_done", judge_done)
    g.add_node("plan", plan)
    g.add_node("fill_one", fill_one)
    g.add_node("guard", guard)
    g.add_node("call_one", call_one)
    g.add_node("record", record)
    g.add_node("finish", finish)
    g.add_edge(START, "judge_done")
    g.add_conditional_edges("judge_done", after_judge, {"finish_ok": "finish", "finish_limit": "finish", "plan": "plan"})
    g.add_conditional_edges("plan", after_plan, ["fill_one", "finish"])
    g.add_edge("fill_one", "guard")
    g.add_conditional_edges("guard", after_guard, ["call_one", "finish", "judge_done"])
    g.add_edge("call_one", "record")
    g.add_conditional_edges("record", after_record, {"finish": "finish", "judge_done": "judge_done"})
    g.add_edge("finish", END)
    return g.compile()


_GRAPH = None


def graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def run(agent: "ja.JevAgent", req: Any, context: str = "") -> Dict[str, Any]:
    """Run one request through the graph. Same status dict as JevAgent.run."""
    if ja.sha256(req.prompt) != req.prompt_sha256:
        return {"ok": False, "error": "prompt hash mismatch; refused to run", "tool_calls": []}
    judge, prose = ja.Judge(), ja.Prose(agent.router)
    started = time.time()
    allowed = ja.tools_for(req.scopes)
    background = req.prompt if req.prompt != req.original_message else ""
    facts: Dict[str, Any] = {"request": req.original_message, "task_context": context, "background": background,
                             **ja._now_facts(), "accounts": {"org_google": ja.ORG_ACCOUNT, "owner_google": ja.OWNER_ACCOUNT}}
    steps: List[Dict[str, Any]] = []
    try:
        async with ja.ToolBox(agent.inkbox_server, agent.mcp_config) as box:
            tool_options: Dict[str, str] = {}
            for name in allowed:
                try:
                    tool_options[name] = ja.TOOL_PURPOSE.get(name) or TOOL_DESCRIPTIONS.get(name) \
                        or (await box.schema(name))["description"] or name
                except ValueError:
                    continue
                except Exception as exc:
                    logger.warning("jev graph: cannot describe %s: %s", name, exc)
            rt = {"agent": agent, "judge": judge, "prose": prose, "req": req, "context": context, "box": box,
                  "tool_options": tool_options}
            initial: AgentState = {"goal": req.original_message, "background": background, "task_context": context,
                                   "facts": facts, "steps": [], "filled": [], "round": 0, "step": 0, "p_done": 0.0,
                                   "excluded": [], "outcome": None, "error": "", "confirm": None}
            final = await graph().ainvoke(initial, config={"configurable": {"rt": rt}, "recursion_limit": 400})
            steps = final.get("steps", [])
            outcome = final.get("outcome")
            if outcome == "confirm":
                st = agent._status(False, "confirmation required", steps, judge, prose, started)
                st["confirm"] = final["confirm"]
                return st
            if outcome == "fail":
                return agent._status(False, final.get("error") or "failed", steps, judge, prose, started, probs=final.get("probs"))
            if outcome is None and final.get("step", 0) >= agent.max_steps and not (steps and final.get("p_done", 0) >= ja.DONE_MIN):
                return agent._status(False, "step limit reached", steps, judge, prose, started)
    except Exception as exc:
        logger.exception("jev graph failed for request %s", req.id)
        return agent._status(False, str(exc), steps, judge, prose, started)
    return agent._status(True, "", steps, judge, prose, started, partial=final.get("partial"))
