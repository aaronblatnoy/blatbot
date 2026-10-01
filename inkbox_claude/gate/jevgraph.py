"""The Jev agent as a LangGraph state graph: collect until nothing more is needed,
then either answer or act, and act in rounds until the action is complete.

    START -> need_more ──yes──> collect -> fill_one (xN) -> guard -> call_one (xN) -> record ─┐
               │  ^                                                                           │
               │  └───────────────────────────────────────────────────────────────────────────┘
               no
               v
             kind ──answer──> finish            (the state goes to the reply writer)
               │
              act
               v
             plan_write -> fill_one -> guard -> call_one -> record -> action_done ──yes──> finish
                                                                          │ no
                                                                          └──────> need_more

need_more     one Jev noul: is more information needed before the goal can be answered
              or acted on? (nothing gathered yet: collect without asking)
collect       one Jev call: a usefulness noul per granted read tool; every tool over the
              bar is called in the same round, arguments filled in parallel. None over the
              bar: a choice among the reads (or give up).
kind          one Jev choice: answer from what is gathered, or act with a write tool.
plan_write    one Jev call: the write tool (by system, then tool) or done.
guard         code: no send to the requester; a destructive call needs evidence of its
              target and then the owner's yes; no identical repeats.
action_done   one Jev noul: is the action the goal asks for complete? No: back to need_more.

Every Jev question in a node is one request. Rounds are bounded by max_steps; a tool
failing twice on the same call, or three times in all, ends the run.
"""

from __future__ import annotations

import json
import logging
import operator
import os
import time
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import jevagent as ja
from .scopes import SCOPE_TREE_SYSTEMS, TOOL_DESCRIPTIONS, TOOL_SYSTEM, _SERVER_HOME as _KNOWN_HOMES, purpose_for, tools_by_system

logger = logging.getLogger(__name__)

NEED_MORE_MIN = float(os.getenv("JEV_AGENT_NEED_MORE_MIN") or 0.5)   # p(more information needed) at or above: collect
COLLECT_MIN = float(os.getenv("JEV_AGENT_COLLECT_MIN") or 0.65)       # a read tool this useful is called this round
COLLECT_MAX = int(os.getenv("JEV_AGENT_COLLECT_MAX") or 4)
ACTION_DONE_MIN = float(os.getenv("JEV_AGENT_ACTION_DONE_MIN") or 0.7)
SYSTEM_MIN = float(os.getenv("JEV_AGENT_SYSTEM_MIN") or 0.25)
NONE_HERE = "none_of_these"
DONE_WRITE = "nothing_to_change"
ANSWER, ACT = "answer", "act"


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
    phase: str                      # "collect" | "write"
    p_more: float                   # last need_more probability
    loop_state: Dict[str, Any]
    chosen: Optional[str]           # the primary tool this round
    extras: List[str]               # other reads called alongside it
    probs: Dict[str, float]
    excluded: List[str]
    batch: List[Tuple[str, Dict[str, Any]]]
    items: List[Dict[str, Any]]     # the concrete things to write, one per change (multi-item goals)
    item_index: int
    outcome: Optional[str]          # "ok" | "fail" | "confirm"
    error: str
    confirm: Optional[Dict[str, Any]]
    partial: Optional[float]


def _rt(config: RunnableConfig) -> Dict[str, Any]:
    return config["configurable"]["rt"]


def _loop_state(state: AgentState, rt: Dict[str, Any]) -> Dict[str, Any]:
    steps = state.get("steps", [])
    ls = {"goal": rt["req"].original_message, "task_context": rt["context"], "background": state.get("background") or "",
          **ja._now_facts(),
          "steps_so_far": [{"tool": s["tool"], "args": s["args"], "ok": s["ok"], "result": ja._seen(s)} for s in steps]}
    return ja._fit(ls)


def _reads(rt: Dict[str, Any], state: AgentState) -> Dict[str, str]:
    excluded = set(state.get("excluded", []))
    return {t: d for t, d in rt["tool_options"].items() if not ja.is_write_tool(t) and t not in excluded}


def _writes(rt: Dict[str, Any], state: AgentState) -> Dict[str, str]:
    excluded = set(state.get("excluded", []))
    return {t: d for t, d in rt["tool_options"].items() if ja.is_write_tool(t) and t not in excluded}


# ----------------------------------------------------------------------------- need_more


async def need_more(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    judge = rt["judge"]
    steps = state.get("steps", [])
    loop_state = _loop_state(state, rt)
    if not steps:
        # Nothing gathered yet: collect without asking (there are reads), else go straight to kind.
        p = 1.0 if _reads(rt, state) else 0.0
        return {"p_more": p, "loop_state": loop_state}
    p_done = await judge.yes(loop_state, {
        "question": "Can `goal` now be answered, or is the action it asks for complete, WITH CONFIDENCE "
                    "from `steps_so_far` alone: is every fact the answer needs already in the evidence?",
        "criteria": {"true": "Everything the goal asked for has been done or, for a question, the retrieved "
                             "results contain what is needed to answer it, including an exact count from rows_where "
                             "when the goal asks how many. An empty result from a BROAD query (a name or address "
                             "alone, no date window, no operators) in the right place answers the question. A failed "
                             "or unauthorized call to a source the goal did not ask about does not make the answer "
                             "incomplete.",
                     "false": "Something the goal asked for has not happened yet, or the results do not contain the "
                              "needed information. A search that returned NOTHING with a narrow or filtered query "
                              "(a date window, several terms, an operator) is not proof of absence: a broader query "
                              "is still needed. A count or filter over many retrieved rows still needs rows_where "
                              "unless it already ran."}})
    p = 1.0 - p_done
    logger.info("jev graph round %d: p(more information needed)=%.2f", state.get("step", 0) + 1, p)
    return {"p_more": p, "loop_state": loop_state}


def after_need_more(state: AgentState, config: RunnableConfig) -> str:
    rt = _rt(config)
    if state.get("step", 0) >= rt["agent"].max_steps:
        return "finish_limit"
    if state["p_more"] >= NEED_MORE_MIN and _reads(rt, state):
        return "collect"
    return "kind"


# ----------------------------------------------------------------------------- collect


async def collect(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Which read tools to call this round: every one judged useful, together."""
    rt = _rt(config)
    judge, agent = rt["judge"], rt["agent"]
    steps = state.get("steps", [])
    loop_state = state["loop_state"]
    reads = _reads(rt, state)
    rnd = state.get("round", 0) + 1
    if len(reads) >= 2:
        useful = await agent._useful_reads(judge, loop_state, reads)
    else:
        # One read left: call it if it has not been called yet; otherwise choose (it may be a repeat).
        useful = {t: 1.0 for t in reads if not any(s["ok"] and s["tool"] == t for s in steps)}
    ranked = sorted(useful.items(), key=lambda kv: -kv[1])
    chosen = [t for t, p in ranked if p >= COLLECT_MIN][:COLLECT_MAX]
    probs = dict(useful)
    if not chosen:
        # No read stands out: one choice over every tool, with giving up as an option.
        # Writes are offered here only once reads are spent: acting while the facts are
        # still missing is how a wrong event gets created.
        opts = {**reads, **(_writes(rt, state) if (not reads or state["p_more"] < NEED_MORE_MIN) else {})}
        if len(steps) >= 2 and steps[-1]["tool"] == steps[-2]["tool"] and len(opts) > 1:
            opts.pop(steps[-1]["tool"], None)          # two in a row: the next move must differ
        opts[ja.GIVE_UP] = "The goal cannot be achieved with these tools or the information available."
        choice, conf, probs = await judge.choose(
            loop_state, {"question": "What is the single best next step toward `goal`?",
                         "rules": ["The gateway delivers the findings to the person who asked; never send them the answer.",
                                   "Read before you write. A wrong first pick or an empty lookup means try the next likely one.",
                                   f"Choose {ja.GIVE_UP} only when no tool here can still help."]}, opts)
        logger.info("jev graph round %d: next step %s (conf %.2f)", rnd, choice, conf)
        if choice is not None and choice != ja.GIVE_UP and ja.is_write_tool(choice):
            return {"outcome": None, "chosen": choice, "extras": [], "probs": probs, "round": rnd, "phase": "write",
                    "filled": [{"__round__": rnd, "tool": "__reset__"}]}
        if choice is None or choice == ja.GIVE_UP:
            gathered = any(s["ok"] for s in steps)
            if gathered:
                logger.info("jev graph: lookups exhausted; deciding from what was gathered")
                return {"outcome": None, "chosen": None, "extras": [], "probs": probs, "partial": 1.0 - state["p_more"],
                        "excluded": list(set(state.get("excluded", [])) | set(reads))}
            if choice == ja.GIVE_UP:
                return {"outcome": "fail", "chosen": None, "extras": [], "probs": probs,
                        "error": "gave up: goal not achievable with the granted tools"}
            return {"outcome": "fail", "chosen": None, "extras": [], "probs": probs,
                    "error": "insufficient evidence to answer with confidence; no further useful step found"}
        chosen = [choice]
    logger.info("jev graph round %d: collecting %s", rnd, [t.split("__")[-1] for t in chosen])
    return {"outcome": None, "chosen": chosen[0], "extras": chosen[1:], "probs": probs, "round": rnd, "phase": "collect",
            "filled": [{"__round__": rnd, "tool": "__reset__"}]}


def after_collect(state: AgentState, config: RunnableConfig):
    if state.get("outcome"):
        return "finish"
    if state.get("chosen") is None:
        return "kind"                     # lookups exhausted: decide from what was gathered
    rnd = state["round"]
    return [Send("fill_one", {"tool": t, "round": rnd, "primary": t == state["chosen"], **_carry(state)})
            for t in [state["chosen"]] + list(state.get("extras", []))]


def _carry(state: AgentState) -> Dict[str, Any]:
    return {"facts": state.get("facts", {}), "steps_view": state.get("steps", [])}


# ----------------------------------------------------------------------------- kind


async def kind(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    judge = rt["judge"]
    writes = _writes(rt, state)
    if not writes and not state.get("steps"):
        pass
    # Arriving here with more information still wanted means the lookups ran out:
    # what is delivered is partial, with the confidence stated for the reply writer.
    partial = (1.0 - state["p_more"]) if state.get("steps") and state["p_more"] >= NEED_MORE_MIN else state.get("partial")
    if not writes:
        # Could the goal still be an action? Then say plainly that no change capability was granted.
        choice, conf, _ = await judge.choose(state["loop_state"], {
            "question": "With the evidence in `steps_so_far`, what does `goal` call for now?"},
            {ANSWER: "A question: it is answered in words from the evidence. Nothing in the world needs changing.",
             ACT: "Something must be done: created, changed, moved, cancelled, sent, written to a sheet or site."}, min_p=0.34)
        if choice == ACT:
            return {"outcome": "fail", "error": "this request was granted no capability to make that change; it can "
                                                "only read. Ask again naming what to change and it will be granted.",
                    "partial": partial}
        return {"phase": "answer", "partial": partial}
    if not state.get("steps"):
        return {"phase": "write", "partial": partial}         # nothing to answer from: the goal is an action
    choice, conf, _ = await judge.choose(state["loop_state"], {
        "question": "With the evidence in `steps_so_far`, what does `goal` call for now?",
        "rules": ["The gateway delivers the findings to the person who asked; never send them the answer."]},
        {ANSWER: "A question: it is answered in words from the evidence. Nothing in the world needs changing.",
         ACT: "Something must be done with a tool: created, changed, moved, cancelled, sent to someone else, "
              "written to a sheet or site."}, min_p=0.34)
    logger.info("jev graph: kind=%s (conf %.2f)", choice, conf)
    return {"phase": "write" if choice == ACT else "answer", "partial": partial}


def after_kind(state: AgentState, config: RunnableConfig) -> str:
    if state.get("phase") != "write":
        return "finish"
    return "plan_write" if state.get("items") is not None else "plan_items"


async def plan_items(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Before writing: the concrete things the goal asks to create, change or send, one
    per separate change, extracted by the prose model from the goal and the evidence as a
    JSON list. One item for a single change. Each item then drives one write round, its
    fields offered as argument candidates."""
    rt = _rt(config)
    prose = rt["prose"]
    facts = state.get("facts", {})
    evidence = {k: (v if len(str(v)) <= 6000 else str(v)[:6000] + " ...") for k, v in facts.items() if k.startswith("result_of_")}
    try:
        text = await prose.write(
            "List the concrete items the goal asks to CREATE, CHANGE, MOVE, CANCEL or SEND, one object per separate "
            "change, as a JSON array and nothing else. Fields, in plain words, only those that apply: what (a short "
            "title), who (name and email if known), when_start (date and time in words, e.g. 'Thu 10/1 6:30 PM'), "
            "when_end (or the duration, e.g. '30 min'), where, text (an exact message body if one is to be sent), "
            "target (what existing thing is being changed). Take dates, times and names from the evidence and the "
            "conversation; do not invent any. A single change is a one-element array. If nothing is to be written, [].",
            {"goal": rt["req"].original_message, "background": state.get("background") or "", "evidence": evidence,
             **ja._now_facts()})
        start, end = text.find("["), text.rfind("]")
        items = json.loads(text[start:end + 1]) if start >= 0 and end > start else []
        items = [it for it in items if isinstance(it, dict)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("jev graph: item extraction failed (%s); writing from the goal alone", exc)
        items = []
    logger.info("jev graph: %d item(s) to write: %s", len(items), [str(it.get("what") or it)[:60] for it in items][:8])
    return {"items": items, "item_index": 0}


def _with_item(state: AgentState) -> Dict[str, Any]:
    """Facts plus the item this write round is about."""
    facts = dict(state.get("facts", {}))
    items, idx = state.get("items") or [], state.get("item_index", 0)
    if items and idx < len(items):
        facts["current_item"] = items[idx]
        facts["items_done"] = idx
        facts["items_total"] = len(items)
    return facts


# ----------------------------------------------------------------------------- plan_write


async def plan_write(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    judge = rt["judge"]
    writes = _writes(rt, state)
    rnd = state.get("round", 0) + 1
    options = dict(writes)
    options[DONE_WRITE] = "Nothing needs changing: the action is already done or the goal needs no write."
    options[ja.GIVE_UP] = "The goal cannot be achieved with these tools or the information available."
    loop_state = dict(state["loop_state"])
    items, idx = state.get("items") or [], state.get("item_index", 0)
    if items and idx < len(items):
        loop_state["this_change"] = {"item": items[idx], "number": idx + 1, "of": len(items)}
    choice, conf, probs = await choose_tool(judge, loop_state, options, rnd, done_key=DONE_WRITE)
    logger.info("jev graph round %d: write %s (conf %.2f)", rnd, choice, conf)
    if choice == DONE_WRITE:
        return {"outcome": "ok", "chosen": None, "extras": [], "probs": probs}
    if choice == ja.GIVE_UP:
        return {"outcome": "fail", "chosen": None, "extras": [], "probs": probs,
                "error": "gave up: goal not achievable with the granted tools"}
    if choice is None:
        return {"outcome": "fail", "chosen": None, "extras": [], "probs": probs,
                "error": "insufficient evidence to act with confidence; could not determine which action the goal calls for"}
    return {"outcome": None, "chosen": choice, "extras": [], "probs": probs, "round": rnd, "phase": "write",
            "filled": [{"__round__": rnd, "tool": "__reset__"}]}


def after_plan_write(state: AgentState, config: RunnableConfig):
    if state.get("outcome"):
        return "finish"
    carry = dict(_carry(state))
    carry["facts"] = _with_item(state)
    return [Send("fill_one", {"tool": state["chosen"], "round": state["round"], "primary": True, **carry})]


async def choose_tool(judge: "ja.Judge", loop_state: Any, options: Dict[str, str], step: int, done_key: str
                      ) -> Tuple[Optional[str], float, Dict[str, float]]:
    """A tool chosen as a tree, in ONE request: a noul per system and a choice within every
    system (plus the done option). Code takes the choice from the likeliest system that did
    not answer none_of_these. One system: a flat choice."""
    tools = {t: d for t, d in options.items() if t not in (done_key, ja.GIVE_UP)}
    groups = tools_by_system(list(tools))
    rules = ["The gateway delivers the findings to the person who asked; never send them the answer.",
             "Change only what the goal asks for; read results already gathered say what exists."]
    if len(groups) <= 1:
        opts = dict(tools)
        opts[done_key] = options[done_key]
        if ja.GIVE_UP in options:
            opts[ja.GIVE_UP] = options[ja.GIVE_UP]
        return await judge.choose(loop_state, {"question": "What is the single best next action toward `goal`?",
                                               "rules": rules}, opts)
    questions: Dict[str, Any] = {f"sys::{s}": {"type": "noul", "instructions": {
        "system": SCOPE_TREE_SYSTEMS.get(s, s),
        "tools_here": [t.split("__")[-1] for t in ts][:12],
        "question": "Does the single best NEXT action toward `goal`, given `steps_so_far`, use this system?"},
        "criteria": {"true": "The next call belongs here.", "false": "The next call is elsewhere, or nothing here is needed."}}
        for s, ts in groups.items()}
    for s, ts in groups.items():
        opts = {t: tools[t] for t in ts}
        opts[NONE_HERE] = "None of these is the right next action."
        questions[f"tool::{s}"] = {"type": "choice",
                                   "instructions": {"question": "If the next action uses this system, which tool is it?",
                                                    "rules": rules},
                                   "criteria": opts}
    questions["done"] = {"type": "noul", "instructions": {"question": options[done_key]},
                         "criteria": {"true": "No write is needed now.", "false": "A write is still needed."}}
    answers = await judge.ask(loop_state, questions)
    p_done = float((answers.get("done") or {}).get("noul") or 0.0)
    sys_p = {s: float((answers.get(f"sys::{s}") or {}).get("noul") or 0.0) for s in groups}
    ordered = [s for s, p in sorted(sys_p.items(), key=lambda kv: -kv[1]) if p >= SYSTEM_MIN]
    logger.info("jev graph round %d: write systems %s done=%.2f", step,
                [(s, round(sys_p[s], 2)) for s in sorted(sys_p, key=lambda k: -sys_p[k])][:4], p_done)
    all_probs: Dict[str, float] = {}
    if p_done >= 0.7 and not ordered:
        return done_key, p_done, all_probs
    for s in ordered[:2]:
        a = answers.get(f"tool::{s}") or {}
        probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
        choice = a.get("choice") or (max(probs, key=probs.get) if probs else None)
        conf = probs.get(choice, float(a.get("confidence") or 0.0)) if choice else 0.0
        for t, p in probs.items():
            if t != NONE_HERE:
                all_probs[t] = max(all_probs.get(t, 0.0), p * sys_p[s])
        if choice and choice != NONE_HERE and conf >= getattr(judge, "min_conf", 0.45):
            return choice, conf, all_probs
    if p_done >= 0.5:
        return done_key, p_done, all_probs
    return None, 0.0, all_probs


# ----------------------------------------------------------------------------- fill, guard, call, record


async def fill_one(item: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    args = await rt["agent"]._fill_args(rt["box"], rt["judge"], rt["prose"], item["tool"], item["facts"],
                                        item["steps_view"], rt["req"])
    return {"filled": [{"__round__": item["round"], "tool": item["tool"], "args": args, "primary": item["primary"]}]}


async def guard(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    agent, judge, prose, req, box = rt["agent"], rt["judge"], rt["prose"], rt["req"], rt["box"]
    steps = state.get("steps", [])
    filled = [f for f in state.get("filled", []) if f.get("tool") != "__reset__" and f.get("__round__") == state["round"]]
    primary = next((f for f in filled if f["primary"]), None)
    if primary is None or primary["args"] is None:
        # The call cannot be made yet (an id not in evidence): a failed attempt; collect more.
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
            # gathered evidence. Record a failed attempt and collect.
            logger.info("jev graph: %s target not in evidence; looking things up first", choice.split("__")[-1])
            return {"outcome": None, "batch": [], "step": state.get("step", 0) + 1,
                    "steps": [{"tool": choice, "args": args, "ok": False, "primary": True,
                               "result": "ERROR: the target of this change is not identified in any gathered result; "
                                         "look the item up (list or search it) before changing it"}]}
        logger.info("jev graph: %s needs the owner's confirmation; pausing", choice.split("__")[-1])
        return {"outcome": "confirm", "batch": [], "error": "confirmation required",
                "confirm": {"tool": choice, "args": args, "about": about}}
    identical = lambda a: any(s["ok"] and s["tool"] == choice and s["args"] == a for s in steps)
    if identical(args):
        again = await agent._fill_args(box, judge, prose, choice, state.get("facts", {}), steps, req, avoid=args)
        if again is not None and again != args:
            logger.info("jev graph: re-filled %s to avoid a repeat", choice.split("__")[-1])
            args = again
    extras: List[Tuple[str, Dict[str, Any]]] = []
    for f in filled:
        if f["primary"] or f["args"] is None:
            continue
        if not any(s["ok"] and s["tool"] == f["tool"] and s["args"] == f["args"] for s in steps):
            extras.append((f["tool"], f["args"]))
    if identical(args):
        if state.get("phase") == "write":
            logger.info("jev graph: would repeat the write %s identically; treating the action as done", choice)
            return {"outcome": "ok", "batch": []}
        logger.info("jev graph: would repeat %s identically; excluding it", choice)
        excluded = list(set(state.get("excluded", [])) | {choice})
        if extras:
            return {"outcome": None, "batch": extras, "excluded": excluded}     # the new reads still run
        return {"outcome": None, "batch": [], "excluded": excluded, "step": state.get("step", 0) + 1}
    return {"outcome": None, "batch": [(choice, args)] + extras}


def after_guard(state: AgentState, config: RunnableConfig):
    if state.get("outcome"):
        return "finish"
    if not state.get("batch"):
        return "need_more"
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
            logger.info("jev graph round %d: %s(%s) -> %s", step_no, short, shown, " ".join(ja._text(s["result"]).split())[:300])
            # The full text, never the digest: rows_where counts over it exactly.
            facts[f"result_of_{short}_{len(steps) - len(new) + new.index(s)}"] = ja._text(s["result"])
        else:
            logger.info("jev graph round %d: %s(%s) -> ERROR %s", step_no, short, shown, " ".join(str(s.get("error")).split())[:300])
    for s in new:
        if s["ok"] or not s.get("primary"):
            continue
        same = sum(1 for x in steps if x["tool"] == s["tool"] and x["args"] == s["args"] and not x["ok"])
        total = sum(1 for x in steps if x["tool"] == s["tool"] and not x["ok"])
        if same >= 2 or total >= 3:
            return {"facts": facts, "step": step_no, "outcome": "fail", "error": f"{s['tool']} keeps failing: {s.get('error')}"}
    _rt(config)["box"].facts = facts
    return {"facts": facts, "step": step_no, "outcome": None}


def after_record(state: AgentState, config: RunnableConfig) -> str:
    if state.get("outcome"):
        return "finish"
    if state.get("phase") != "write":
        return "need_more"
    return "next_item"


async def next_item(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """After a write round: the next item if the write succeeded and items remain."""
    steps = state.get("steps", [])
    last = next((s for s in reversed(steps) if s.get("primary")), None)
    items, idx = state.get("items") or [], state.get("item_index", 0)
    if last is not None and last["ok"] and items and idx + 1 < len(items):
        logger.info("jev graph: item %d of %d done; on to the next", idx + 1, len(items))
        return {"item_index": idx + 1}
    return {}


def after_next_item(state: AgentState, config: RunnableConfig) -> str:
    items, idx = state.get("items") or [], state.get("item_index", 0)
    steps = state.get("steps", [])
    last = next((s for s in reversed(steps) if s.get("primary")), None)
    if last is not None and last["ok"] and items and idx < len(items) and idx > 0 and state.get("round", 0) > 0:
        # item_index was just advanced: write the next one
        return "plan_write"
    return "action_done"


# ----------------------------------------------------------------------------- action_done


async def action_done(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    rt = _rt(config)
    loop_state = _loop_state(state, rt)
    p = await rt["judge"].yes(loop_state, {
        "question": "Is the action `goal` asks for now complete WITH CONFIDENCE, as `steps_so_far` shows?",
        "criteria": {"true": "Every change or send the goal asked for has succeeded; nothing is left to do.",
                     "false": "A part of the action has not happened yet, failed, or needs another call."}})
    logger.info("jev graph round %d: p(action complete)=%.2f", state.get("step", 0), p)
    return {"p_more": 1.0 - p, "loop_state": loop_state, "partial": None if p >= ACTION_DONE_MIN else state.get("partial")}


def after_action_done(state: AgentState, config: RunnableConfig) -> str:
    rt = _rt(config)
    if 1.0 - state["p_more"] >= ACTION_DONE_MIN:
        return "finish"
    if state.get("step", 0) >= rt["agent"].max_steps:
        return "finish_limit"
    return "need_more"


async def finish(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    return {}


# ----------------------------------------------------------------------------- graph


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in (("need_more", need_more), ("collect", collect), ("kind", kind), ("plan_items", plan_items),
                     ("plan_write", plan_write), ("fill_one", fill_one), ("guard", guard), ("call_one", call_one),
                     ("record", record), ("next_item", next_item), ("action_done", action_done), ("finish", finish)):
        g.add_node(name, fn)
    g.add_edge(START, "need_more")
    g.add_conditional_edges("need_more", after_need_more, {"collect": "collect", "kind": "kind", "finish_limit": "finish"})
    g.add_conditional_edges("collect", after_collect, ["fill_one", "kind", "finish"])
    g.add_conditional_edges("kind", after_kind, {"plan_items": "plan_items", "plan_write": "plan_write", "finish": "finish"})
    g.add_edge("plan_items", "plan_write")
    g.add_conditional_edges("plan_write", after_plan_write, ["fill_one", "finish"])
    g.add_edge("fill_one", "guard")
    g.add_conditional_edges("guard", after_guard, ["call_one", "finish", "need_more"])
    g.add_edge("call_one", "record")
    g.add_conditional_edges("record", after_record, {"finish": "finish", "need_more": "need_more", "next_item": "next_item"})
    g.add_conditional_edges("next_item", after_next_item, {"plan_write": "plan_write", "action_done": "action_done"})
    g.add_conditional_edges("action_done", after_action_done, {"finish": "finish", "finish_limit": "finish", "need_more": "need_more"})
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
                    tool_options[name] = purpose_for(name) if (name in ja.TOOL_PURPOSE or TOOL_DESCRIPTIONS.get(name)
                                                              or name.split("__")[1] in _KNOWN_HOMES) \
                        else ((await box.schema(name))["description"] or name)
                except ValueError:
                    continue
                except Exception as exc:
                    logger.warning("jev graph: cannot describe %s: %s", name, exc)
            box.facts = facts                      # rows_where reads gathered results from here
            rt = {"agent": agent, "judge": judge, "prose": prose, "req": req, "context": context, "box": box,
                  "tool_options": tool_options}
            initial: AgentState = {"goal": req.original_message, "background": background, "task_context": context,
                                   "facts": facts, "steps": [], "filled": [], "round": 0, "step": 0, "phase": "collect",
                                   "p_more": 1.0, "excluded": [], "outcome": None, "error": "", "confirm": None, "partial": None,
                                   "items": None, "item_index": 0}
            final = await graph().ainvoke(initial, config={"configurable": {"rt": rt}, "recursion_limit": 400})
            steps = final.get("steps", [])
            outcome = final.get("outcome")
            if outcome == "confirm":
                st = agent._status(False, "confirmation required", steps, judge, prose, started)
                st["confirm"] = final["confirm"]
                return st
            if outcome == "fail":
                return agent._status(False, final.get("error") or "failed", steps, judge, prose, started, probs=final.get("probs"))
            if outcome is None and final.get("step", 0) >= agent.max_steps and final.get("p_more", 0) >= NEED_MORE_MIN:
                return agent._status(False, "step limit reached", steps, judge, prose, started)
            partial = final.get("partial")
            if partial is not None and partial >= ja.DONE_MIN:
                partial = None
    except Exception as exc:
        logger.exception("jev graph failed for request %s", req.id)
        return agent._status(False, str(exc), steps, judge, prose, started)
    return agent._status(True, "", steps, judge, prose, started, partial=partial)
