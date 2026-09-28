"""A tool-calling agent driven by typed judgments (TypeSafe Jev) instead of a chat model.

Replaces the Claude Code executor for a request when GATE_EXECUTOR=jev. The loop:

  1. Jev picks the next tool from the scope's list (a Choice), or "done" / "give_up".
  2. Code fills that tool's arguments. Values that can be selected from what is
     already known (a calendar event id, a row, a contact, a date range) are picked
     by Jev from a candidate list. Values that must be written (an email body, a
     bio, a note) are composed by the prose model (DeepSeek), and only when Jev has
     judged that prose is needed.
  3. Code calls the tool, records the result, and loops.

Jev never sees a tool and never writes text. DeepSeek never chooses a tool. Code
owns the loop, the argument schemas, and every side effect. When the agent cannot
make progress it stops with STATUS: FAILED and the gate falls back to Claude Code
if GATE_EXECUTOR_FALLBACK=claude.

Configuration:
  GATE_EXECUTOR             claude (default) | jev
  GATE_EXECUTOR_FALLBACK    claude (default) | none   what to do when the jev agent gives up
  JEV_AGENT_MAX_STEPS       default 12
  JEV_AGENT_MIN_CONF        default 0.55   below this a pick is treated as "unsure"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import AsyncExitStack
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

from .scopes import ORG_ACCOUNT, OWNER_ACCOUNT, sends_to_requester, tools_for
from .store import Request, sha256
from .taskpick import TYPESAFE_URL

logger = logging.getLogger(__name__)

TZ = ZoneInfo("America/New_York")
DONE, GIVE_UP = "done", "give_up"


def enabled() -> bool:
    return (os.getenv("GATE_EXECUTOR") or "claude").strip().lower() == "jev" and bool(os.getenv("TYPESAFE_API_KEY"))


# ----------------------------------------------------------------------------
# Tool access: in-process Inkbox tools + the external MCP servers, one client each
# ----------------------------------------------------------------------------


async def _call_server(server: Any, method: str, request_type: Any, params: Any) -> Any:
    """Invoke one of the in-process mcp Server's handlers directly, across library
    versions: newer servers keep (ctx, params) entries keyed by method string;
    older ones keep request-typed callables keyed by request class."""
    get = getattr(server, "get_request_handler", None)
    entry = get(method) if callable(get) else None
    if entry is not None and hasattr(entry, "handler"):
        res = await entry.handler(None, params)
        return res
    for attr in ("request_handlers", "_request_handlers"):
        table = getattr(server, attr, None) or {}
        h = table.get(request_type) or table.get(method)
        if h is None:
            continue
        if hasattr(h, "handler"):
            return await h.handler(None, params)
        res = await h(request_type(method=method, params=params) if params is not None else request_type(method=method))
        return getattr(res, "root", res)
    raise RuntimeError(f"no handler for {method}")


def _input_schema(t: Any) -> Dict[str, Any]:
    return getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}


def split_tool(name: str) -> Tuple[str, str]:
    """mcp__<server>__<tool> -> (server, tool)."""
    m = re.match(r"^mcp__(.+?)__(.+)$", name)
    if not m:
        raise ValueError(f"not an mcp tool name: {name}")
    return m.group(1), m.group(2)


class ToolBox:
    """Uniform access to every tool the gate can grant, by full name
    (mcp__<server>__<tool>). External servers are started lazily and kept for
    the life of one request."""

    def __init__(self, inkbox_server: Any, mcp_config: Dict[str, Dict[str, Any]]):
        self._inkbox = inkbox_server
        self._cfg = mcp_config
        self._stack = AsyncExitStack()
        self._sessions: Dict[str, Any] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    async def __aenter__(self) -> "ToolBox":
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.__aexit__(*exc)

    async def _session(self, server: str) -> Any:
        if server in self._sessions:
            return self._sessions[server]
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        cfg = self._cfg.get(server)
        if not cfg:
            raise RuntimeError(f"no launch config for MCP server {server}")
        params = StdioServerParameters(command=cfg["command"], args=list(cfg.get("args") or []),
                                       env={**os.environ, **(cfg.get("env") or {})})
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        for t in listed.tools:
            self._schemas[f"mcp__{server}__{t.name}"] = {"description": t.description or "", "schema": _input_schema(t)}
        self._sessions[server] = session
        return session

    async def _inkbox_tools(self) -> Dict[str, Dict[str, Any]]:
        """Tools of the in-process Inkbox server (an mcp.server.Server inside the
        SDK config dict), read through its own list_tools handler."""
        if "inkbox" in self._sessions:
            return self._sessions["inkbox"]
        from mcp import types as mt
        inst = self._inkbox["instance"] if isinstance(self._inkbox, dict) else self._inkbox
        res = await _call_server(inst, "tools/list", mt.ListToolsRequest, None)
        out = {}
        for t in res.tools:
            out[t.name] = {"description": t.description or "", "schema": _input_schema(t)}
            self._schemas[f"mcp__inkbox__{t.name}"] = out[t.name]
        self._sessions["inkbox"] = out
        return out

    async def schema(self, name: str) -> Dict[str, Any]:
        server, short = split_tool(name)
        if server == "inkbox":
            return (await self._inkbox_tools()).get(short, {"description": "", "schema": {}})
        await self._session(server)
        return self._schemas.get(name, {"description": "", "schema": {}})

    async def call(self, name: str, args: Dict[str, Any]) -> Any:
        server, short = split_tool(name)
        if server == "inkbox":
            from mcp import types as mt
            await self._inkbox_tools()
            inst = self._inkbox["instance"] if isinstance(self._inkbox, dict) else self._inkbox
            res = await _call_server(inst, "tools/call", mt.CallToolRequest,
                                     mt.CallToolRequestParams(name=short, arguments=args))
        else:
            session = await self._session(server)
            res = await session.call_tool(short, arguments=args)
        parts = []
        for c in getattr(res, "content", []) or []:
            txt = getattr(c, "text", None)
            parts.append(txt if txt is not None else str(c))
        text = "\n".join(parts)
        if getattr(res, "isError", False):
            raise RuntimeError(text)
        return text


def mcp_config_from_claude_json() -> Dict[str, Dict[str, Any]]:
    """The same launch config Claude Code uses on this machine."""
    try:
        d = json.load(open(os.path.expanduser("~/.claude.json")))
        return dict(d.get("mcpServers") or {})
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Jev calls
# ----------------------------------------------------------------------------


class JudgeTooLarge(Exception):
    """TypeSafe refused the state as over its token limit."""


class Judge:
    def __init__(self) -> None:
        self.api_key = (os.getenv("TYPESAFE_API_KEY") or "").strip()
        self.model = (os.getenv("TYPESAFE_MODEL") or "jev-latest").strip()
        self.min_conf = float(os.getenv("JEV_AGENT_MIN_CONF") or 0.45)
        self.calls = 0

    async def ask(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """One TypeSafe request. A state over the model's token limit is retried once
        with the large raw tool results replaced by size markers (candidate values
        were already extracted from the full text, and the full text still reaches
        the reply model and the owner untouched). Any other failure returns no
        answers, which every caller treats as "unsure"."""
        self.calls += 1
        try:
            return await self._post(state, questions)
        except JudgeTooLarge:
            slim = _slim_state(state)
            logger.warning("jev agent: state over TypeSafe's limit; retrying with %d large result(s) replaced by markers",
                           slim[1])
            try:
                return await self._post(slim[0], questions)
            except Exception as exc:
                logger.warning("jev agent: judgment failed after slimming: %s", exc)
                return {}
        except Exception as exc:
            logger.warning("jev agent: judgment failed: %s", exc)
            return {}

    async def _post(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                                  json={"state": state, "model": self.model, "questions": questions})
            if r.status_code == 400 and "max_tokens_exceeded" in r.text:
                raise JudgeTooLarge(r.text)
            r.raise_for_status()
            return r.json()["answers"]

    async def choose(self, state: Any, question: Any, options: Dict[str, Any],
                     min_p: Optional[float] = None) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Best option by probability. Returns None as the pick when the top
        option's probability is under `min_p` (default JEV_AGENT_MIN_CONF)."""
        a = (await self.ask(state, {"q": {"type": "choice", "instructions": question, "criteria": options}}))["q"]
        probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
        top = a.get("choice") or (max(probs, key=probs.get) if probs else None)
        p = probs.get(top, float(a.get("confidence") or 0.0)) if top else 0.0
        floor = self.min_conf if min_p is None else min_p
        return (top if top and p >= floor else None), p, probs

    async def yes(self, state: Any, question: Any) -> float:
        a = (await self.ask(state, {"q": {"type": "noul", "instructions": question}}))["q"]
        return float(a.get("noul") or 0.0)


# ----------------------------------------------------------------------------
# Prose (DeepSeek), only when Jev says prose is needed
# ----------------------------------------------------------------------------


class Prose:
    def __init__(self, router: Any):
        self.router = router  # the gate's Router: reuse its key/model/_chat
        self.calls = 0

    async def write(self, instruction: str, facts: Dict[str, Any]) -> str:
        self.calls += 1
        sys_p = ("You write short pieces of text for Blatbot, Executive Assistant to Aaron Blatnoy. "
                 "Use only the facts given. No emojis. No em dashes. Sign emails exactly:\n"
                 "Best,\nBlatbot\nExecutive Assistant to Aaron Blatnoy\n"
                 "Never mention how you work or any tooling. Output only the requested text.")
        user = f"FACTS:\n{json.dumps(facts, ensure_ascii=False, default=str, indent=1)}\n\nWRITE:\n{instruction}"
        raw = await self.router._chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                      json_mode=False)
        return raw.strip()


# ----------------------------------------------------------------------------
# The agent
# ----------------------------------------------------------------------------


_ERROR_RE = re.compile(r"^\s*(#+\s*)?(error|api error|httperror|traceback|exception|invalid|failed|unauthori[sz]ed|forbidden|not found)\b"
                       r"|\berror calling tool\b|\bHttpError \d{3}\b|\breturned \"[^\"]*(error|invalid|denied|not found)", re.I)


def _looks_like_error(result: Any) -> bool:
    """Whether a tool result that was not flagged isError still reads as a failure."""
    t = result if isinstance(result, str) else _text(result)
    head = t.lstrip()[:400]
    return bool(_ERROR_RE.search(head))


def _tried_values(steps: List[Dict[str, Any]], tool: str, arg: str) -> List[Any]:
    """Values this tool has already been called with for this argument, in order."""
    out: List[Any] = []
    for st in steps:
        if st["tool"] == tool and arg in (st.get("args") or {}):
            v = st["args"][arg]
            if v not in out:
                out.append(v)
    return out


_LARGE = 20000


def _slim_state(state: Any) -> Tuple[Any, int]:
    """Copy of `state` with every string over _LARGE characters replaced by a marker.
    Used only when TypeSafe rejects the state as too large. Returns (state, count)."""
    n = 0

    def walk(v: Any) -> Any:
        nonlocal n
        if isinstance(v, str) and len(v) > _LARGE:
            n += 1
            return f"(a {len(v):,}-character tool result; its ids, addresses and phones are offered as candidates)"
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v
    return walk(state), n


def _text(v: Any) -> str:
    """Whole value as text. Nothing is ever truncated on its way to a model."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)


def _now_facts() -> Dict[str, str]:
    now = datetime.now(TZ)
    return {
        "now": now.strftime("%A, %B %d, %Y, %-I:%M %p %Z"),
        "today_iso": now.date().isoformat(),
        "tomorrow_iso": (now.date() + timedelta(days=1)).isoformat(),
        "timezone": "America/New_York",
    }


def _date_ranges(now: Optional[datetime] = None) -> Dict[str, Dict[str, str]]:
    """Canonical ranges as ISO datetimes in New York, keyed by name. Jev picks the
    name; code supplies the start or end depending on the argument."""
    now = now or datetime.now(TZ)
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = day - timedelta(days=day.weekday())
    month_start = day.replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    def rng(a: datetime, b: datetime, label: str) -> Dict[str, str]:
        return {"start": a.isoformat(), "end": b.isoformat(), "label": label}
    return {
        "today": rng(day, day + timedelta(days=1), "today"),
        "tomorrow": rng(day + timedelta(days=1), day + timedelta(days=2), "tomorrow"),
        "yesterday": rng(day - timedelta(days=1), day, "yesterday"),
        "this_week": rng(monday, monday + timedelta(days=7), "this week, Monday through Sunday"),
        "last_week": rng(monday - timedelta(days=7), monday, "last week, Monday through Sunday"),
        "next_week": rng(monday + timedelta(days=7), monday + timedelta(days=14), "next week, Monday through Sunday"),
        "next_7_days": rng(now, now + timedelta(days=7), "from now through the next seven days"),
        "past_7_days": rng(now - timedelta(days=7), now, "the past seven days up to now"),
        "this_month": rng(month_start, next_month, "this calendar month"),
        "next_30_days": rng(now, now + timedelta(days=30), "from now through the next thirty days"),
    }


def _date_role(name: str, desc: str) -> Optional[str]:
    """'start' or 'end' for a datetime argument, None when it is not a range bound."""
    n = name.lower()
    if not any(k in n or k in desc.lower() for k in ("time", "date")):
        return None
    if any(k in n for k in ("min", "start", "from", "after", "begin")):
        return "start"
    if any(k in n for k in ("max", "end", "until", "before", "to_")) or n in ("to",):
        return "end"
    return None


class JevAgent:
    def __init__(self, *, inkbox_server: Any, router: Any, mcp_config: Optional[Dict[str, Any]] = None,
                 max_steps: Optional[int] = None, protected: Optional[List[str]] = None):
        self.protected = list(protected or [])
        self.inkbox_server = inkbox_server
        self.router = router
        self.mcp_config = mcp_config if mcp_config is not None else mcp_config_from_claude_json()
        self.max_steps = int(max_steps or os.getenv("JEV_AGENT_MAX_STEPS") or 12)

    async def run(self, req: Request, context: str = "") -> Dict[str, Any]:
        if sha256(req.prompt) != req.prompt_sha256:
            return {"ok": False, "error": "prompt hash mismatch; refused to run", "tool_calls": []}
        judge, prose = Judge(), Prose(self.router)
        allowed = tools_for(req.scopes)
        started = time.time()
        steps: List[Dict[str, Any]] = []
        facts: Dict[str, Any] = {"request": req.original_message, "task_context": context, **_now_facts(),
                                 "accounts": {"org_google": ORG_ACCOUNT, "owner_google": OWNER_ACCOUNT}}
        try:
            async with ToolBox(self.inkbox_server, self.mcp_config) as box:
                tool_options: Dict[str, str] = {}
                for name in allowed:
                    try:
                        tool_options[name] = (await box.schema(name))["description"] or name
                    except ValueError:
                        continue  # a Claude built-in (WebSearch, WebFetch): not callable here
                    except Exception as exc:
                        logger.warning("jev agent: cannot describe %s: %s", name, exc)
                p_done = 0.0
                for step in range(self.max_steps):
                    state = {"goal": req.original_message, "task_context": context, **_now_facts(),
                             "steps_so_far": [f"{s['tool']}({_text(s['args'])}) -> {_text(s['result'])}" for s in steps]}
                    if steps:
                        p_done = await judge.yes(state, {
                            "question": "Is `goal` now fully achieved by `steps_so_far`, so no further tool call is needed?",
                            "criteria": {"true": "Everything the goal asked for has been done or, for a question, "
                                                 "the right lookup has been made (an empty result from the right "
                                                 "place still answers the question).",
                                         "false": "Something the goal asked for has not happened yet, or the lookup "
                                                  "was aimed at the wrong target and a different call is needed."}})
                        logger.info("jev agent step %d: p(done)=%.2f", step + 1, p_done)
                        if p_done >= 0.6:
                            break
                    options = dict(tool_options)
                    if len(steps) >= 2 and steps[-1]["tool"] == steps[-2]["tool"] and len(options) > 1:
                        # Two calls of the same tool in a row: the next move must be a different
                        # one (open a result, read a sheet), not a third search.
                        options.pop(steps[-1]["tool"], None)
                    options[GIVE_UP] = "The goal cannot be achieved with these tools or the information available."
                    choice, conf, probs = await judge.choose(
                        state,
                        {"question": "What is the single best next step toward `goal`?",
                         "rules": ["The gateway delivers whatever is found to the person who asked; never send or text the "
                                   "answer to the requester. Sending tools are only for messaging OTHER people the goal names.",
                                   "Read before you write: look things up before booking, sending, or changing.",
                                   "When something must be found in a document, sheet, calendar or inbox: search, open the "
                                   "most likely result, and if it is not the right one open the next likely one, or search "
                                   "again with different words. A wrong first pick or an empty search is not a reason to give up.",
                                   "Do not repeat a step that already succeeded with the same arguments.",
                                   f"Choose {GIVE_UP} if a needed tool is missing or a step failed twice."]},
                        options)
                    logger.info("jev agent step %d: %s (conf %.2f)", step + 1, choice, conf)
                    if choice is None:
                        if steps and steps[-1]["ok"] and not is_write_tool(steps[-1]["tool"]) and p_done >= 0.3:
                            # Nothing stands out after a read and the goal is plausibly met:
                            # what was gathered is the answer (the gate's reply model phrases it).
                            logger.info("jev agent: no clear next step after reads (p_done %.2f); finishing", p_done)
                            break
                        return self._status(False, "unsure which step to take next", steps, judge, prose, started,
                                            probs=probs)
                    if choice == GIVE_UP:
                        if steps and p_done >= 0.3 and all(s["ok"] and not is_write_tool(s["tool"]) for s in steps):
                            # Everything so far was a successful read and the goal is plausibly met:
                            # what was found IS the answer.
                            # The gateway delivers it to the requester; nothing needs sending.
                            logger.info("jev agent: give_up after reads only; finishing with what was found")
                            break
                        return self._status(False, "gave up: goal not achievable with the granted tools", steps, judge, prose, started)
                    args = await self._fill_args(box, judge, prose, choice, facts, steps, req)
                    if args is None:
                        return self._status(False, f"could not determine arguments for {choice}", steps, judge, prose, started)
                    if sends_to_requester(choice, args, self.protected + [req.sender, req.chat_id]):
                        # The gateway delivers the answer; a send to the requester would duplicate it.
                        logger.info("jev agent: refusing %s to the requester; finishing with findings", choice)
                        break
                    if any(s["ok"] and s["tool"] == choice and s["args"] == args for s in steps):
                        # The best next move is one already made: nothing better exists, so the work is done.
                        logger.info("jev agent: would repeat %s with identical arguments; treating as done", choice)
                        break
                    try:
                        result = await box.call(choice, args)
                        if _looks_like_error(result):
                            # Many MCP tools report failures as ordinary text. Count it as a
                            # failed step so the loop tries different arguments, and never
                            # "finish with findings" on it.
                            raise RuntimeError(result if isinstance(result, str) else _text(result))
                        steps.append({"tool": choice, "args": args, "result": result, "ok": True})
                        facts[f"result_of_{choice.split('__')[-1]}_{len(steps)}"] = _text(result)
                    except Exception as exc:
                        steps.append({"tool": choice, "args": args, "result": f"ERROR: {exc}", "ok": False})
                        if sum(1 for s in steps if s["tool"] == choice and not s["ok"]) >= 2:
                            return self._status(False, f"{choice} failed twice: {exc}", steps, judge, prose, started)
                else:
                    return self._status(False, "step limit reached", steps, judge, prose, started)
        except Exception as exc:
            logger.exception("jev agent failed for request %s", req.id)
            return self._status(False, str(exc), steps, judge, prose, started)
        return self._status(True, "", steps, judge, prose, started)

    async def _fill_args(self, box: ToolBox, judge: Judge, prose: Prose, tool: str, facts: Dict[str, Any],
                         steps: List[Dict[str, Any]], req: Request) -> Optional[Dict[str, Any]]:
        """Fill one tool's arguments in ONE TypeSafe request: for each optional
        argument a Noul (supply it?), for each argument with values already in
        play a Choice (which one?), for enums/booleans a Choice. Only arguments
        that must be written go to the prose model afterwards."""
        meta = await box.schema(tool)
        schema = meta.get("schema") or {}
        props: Dict[str, Any] = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        server, short = split_tool(tool)
        args: Dict[str, Any] = {}
        state = {"goal": req.original_message, "tool": short, "tool_description": meta.get("description") or "",
                 "arguments": {n: {"description": str(sp.get("description") or ""), "type": sp.get("type") or "string",
                                   "required": n in required} for n, sp in props.items()},
                 "known_facts": {k: _text(v) for k, v in facts.items()},
                 "steps_so_far": [f"{s['tool']} -> {_text(s['result'])}" for s in steps]}
        questions: Dict[str, Any] = {}
        cands: Dict[str, List[Any]] = {}
        for name, spec in props.items():
            if name == "user_google_email":
                args[name] = ORG_ACCOUNT if server == "tamid-drive" else OWNER_ACCOUNT
                continue
            typ = spec.get("type") or "string"
            if name not in required:
                questions[f"supply::{name}"] = {"type": "noul", "instructions": {
                    "question": f"Should the argument `arguments.{name}` be supplied for this `tool` call, given `goal`?",
                    "criteria": {"true": "A value is needed for the call to do what the goal requires.",
                                 "false": "Leave it out; the default is right or it does not apply."}}}
            if typ == "boolean" or spec.get("enum"):
                opts = {str(o): f"{name} = {o}" for o in (spec.get("enum") or [True, False])}
                questions[f"pick::{name}"] = {"type": "choice", "instructions": f"What value should `arguments.{name}` take for this call?",
                                              "criteria": opts}
                continue
            role = _date_role(name, str(spec.get("description") or ""))
            if role and typ == "string":
                ranges = _date_ranges()
                opts = {k: f"{v['label']} ({v['start'][:10]} to {v['end'][:10]})" for k, v in ranges.items()}
                opts["write_new"] = "None of these ranges; a specific date or time must be composed from the goal."
                questions[f"range::{name}"] = {"type": "choice", "instructions":
                                               f"`arguments.{name}` is the {role} of a time range. Which named range does `goal` mean?",
                                               "criteria": opts}
                continue
            cs = _candidate_values(name, str(spec.get("description") or ""), typ, facts, steps)
            tried = _tried_values(steps, tool, name)
            if cs and tried and len(cs) > len(tried):
                # Trial and error: a value this tool already got for this argument is
                # not offered again, so the next likely document/event/row gets tried.
                cs = [c for c in cs if c not in tried]
            if cs:
                cands[name] = cs
                opts = {f"c{i}": {"value": _text(v)} for i, v in enumerate(cands[name])}
                opts["write_new"] = "None of these; the value must be composed from the goal (a date/time, a title, a body, a query)."
                questions[f"pick::{name}"] = {"type": "choice", "instructions": f"Which of these is the right value for `arguments.{name}`?",
                                              "criteria": opts}
        answers = await judge.ask(state, questions) if questions else {}

        def _top(qid: str, floor: float) -> Optional[str]:
            a = answers.get(qid) or {}
            probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
            top = a.get("choice") or (max(probs, key=probs.get) if probs else None)
            return top if top and probs.get(top, 0.0) >= floor else None

        to_write: List[Tuple[str, str, str]] = []
        for name, spec in props.items():
            if name in args:
                continue
            typ = spec.get("type") or "string"
            desc = str(spec.get("description") or "")
            if name not in required and float((answers.get(f"supply::{name}") or {}).get("noul") or 0.0) < 0.6:
                continue
            if f"range::{name}" in questions:
                pick = _top(f"range::{name}", 0.3)
                if pick and pick != "write_new":
                    args[name] = _date_ranges()[pick][_date_role(name, desc) or "start"]
                    continue
                if pick is None and name not in required:
                    continue
            if typ == "boolean" or spec.get("enum"):
                pick = _top(f"pick::{name}", 0.3)
                if pick is None:
                    if name in required:
                        return None
                    continue
                args[name] = (pick == "True") if typ == "boolean" else _coerce(pick, typ)
                continue
            if name in cands:
                pick = _top(f"pick::{name}", 0.3)
                if pick and pick != "write_new":
                    args[name] = _coerce(cands[name][int(pick[1:])], typ)
                    continue
                if pick is None and name not in required:
                    continue
            to_write.append((name, typ, desc))
        # Must be written: independent values, so all prose calls run at once.
        if to_write:
            facts_text = {k: _text(v) for k, v in facts.items()}
            def _instr(name: str, typ: str, desc: str) -> str:
                tried = _tried_values(steps, tool, name)
                again = (f" This tool was already called with `{name}` = {', '.join(repr(t) for t in tried)} and that did "
                         f"not find what the goal needs; produce a DIFFERENT value (other words, a broader or narrower "
                         f"search, another likely name)." if tried else "")
                return (f"Produce ONLY the value for the tool argument `{name}` ({typ}). {desc} "
                        f"Use ISO 8601 with the -04:00 / -05:00 New York offset for any datetime. "
                        f"Output the bare value: no quotes, no label, no explanation. If the facts do not contain enough "
                        f"to determine it, output exactly UNKNOWN.{again}")
            texts = await asyncio.gather(*[prose.write(_instr(name, typ, desc),
                {"goal": req.original_message, "tool": short, "facts": facts_text}) for name, typ, desc in to_write])
            for (name, typ, desc), text in zip(to_write, texts):
                if not _plausible_value(name, text, typ):
                    logger.info("jev agent: no usable value for %s.%s (%r)", short, name, text)
                    if name in required:
                        return None
                    continue
                args[name] = _coerce(text, typ)
        missing = [r for r in required if r not in args]
        if missing:
            logger.warning("jev agent: missing required args %s for %s", missing, tool)
            return None
        return args

    @staticmethod
    def _status(ok: bool, error: str, steps: List[Dict[str, Any]], judge: Judge, prose: Prose,
                started: float, probs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        lines = [f"{'Done' if ok else 'Failed'} via Jev agent in {time.time() - started:.1f}s "
                 f"({len(steps)} tool call(s), {judge.calls} Jev judgment(s), {prose.calls} prose call(s))."]
        # Full results, never truncated: the reply model answers from this text.
        for s in steps:
            r = s["result"] if isinstance(s["result"], str) else json.dumps(s["result"], ensure_ascii=False, default=str)
            lines.append(f"- {s['tool'].split('__')[-1]}:\n{r}")
        if error:
            lines.append(f"Reason: {error}")
        lines.append("STATUS: OK" if ok else "STATUS: FAILED")
        raw = "\n".join(lines)
        wrote = any(s["ok"] and is_write_tool(s["tool"]) for s in steps)
        return {"ok": ok, "summary": raw, "raw": raw, "tool_calls": [s["tool"] for s in steps], "wrote": wrote,
                "steps": [{"tool": s["tool"], "args": s["args"], "ok": s["ok"]} for s in steps],
                "error": error or None, "engine": "jev", "jev_calls": judge.calls, "prose_calls": prose.calls,
                "seconds": round(time.time() - started, 1), "probs": probs}


_WRITE_RE = re.compile(r"(send|create|update|modify|delete|manage|append|publish|place_call|move|set_|replace|import|insert|format|resize|run_script|reply|complete|fail"
                       r"|browser_click|browser_type|browser_fill_form|browser_select_option|browser_press_key|browser_drag|browser_drop|browser_file_upload|browser_handle_dialog)", re.I)


def is_write_tool(name: str) -> bool:
    """Whether a tool changes the world. A successful write must never be
    retried by the Claude fallback (double email, duplicate event)."""
    return bool(_WRITE_RE.search(name.split("__")[-1]))


_LABELED_ID_RE = re.compile(r"\b(?:ID|id|Id)\s*[:=]\s*\"?([^\s\"'|,)>]+)")
_ID_RE = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{16,}\b")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?1?[ (.-]*\d{3}[ ).-]*\d{3}[ .-]*\d{4}")


def _candidate_values(name: str, desc: str, typ: str, facts: Dict[str, Any], steps: List[Dict[str, Any]]) -> List[Any]:
    """Values already in play that could fill this argument: ids, addresses, phones,
    and short strings from tool results. Jev selects among them."""
    blob = "\n".join(_text(v) for v in facts.values()) + "\n" + "\n".join(_text(s["result"]) for s in steps)
    lname = (name + " " + desc).lower()
    out: List[Any] = []
    if "email" in lname or name in ("to", "cc", "attendee", "attendees"):
        out += _EMAIL_RE.findall(blob)
    if "phone" in lname or "sms" in lname or "number" in lname or name in ("to", "to_number"):
        out += [m.strip() for m in _PHONE_RE.findall(blob)]
    if name.endswith("_id") or name in ("id", "spreadsheet_id", "form_id", "event_id", "message_id", "thread_id", "file_id", "calendar_id"):
        out += _LABELED_ID_RE.findall(blob) + _ID_RE.findall(blob)
    seen: List[Any] = []
    for v in out:
        v = v.strip()
        if v and v not in seen:
            seen.append(v)
    return seen


def _plausible_value(name: str, text: str, typ: str) -> bool:
    """Reject prose where a value belongs: UNKNOWN, sentences in an id slot, empties."""
    t = (text or "").strip()
    if not t or t.upper().startswith("UNKNOWN"):
        return False
    lname = name.lower()
    if lname.endswith("_id") or lname in ("id", "spreadsheet_id", "form_id", "event_id", "file_id", "calendar_id", "message_id", "thread_id"):
        return " " not in t
    if lname in ("email", "user_google_email") or lname.endswith("_email"):
        return "@" in t and " " not in t
    if lname in ("to", "to_number") or "phone" in lname:
        return ("@" in t and " " not in t) or sum(ch.isdigit() for ch in t) >= 7
    if typ in ("integer", "number"):
        return re.fullmatch(r"-?\d+(\.\d+)?", t) is not None
    if "time" in lname or "date" in lname:
        return re.match(r"\d{4}-\d{2}-\d{2}", t) is not None
    return True


def _coerce(v: Any, typ: str) -> Any:
    if typ == "integer":
        try:
            return int(str(v).strip())
        except Exception:
            return v
    if typ == "number":
        try:
            return float(str(v).strip())
        except Exception:
            return v
    if typ == "array":
        if isinstance(v, list):
            return v
        s = str(v).strip()
        try:
            j = json.loads(s)
            return j if isinstance(j, list) else [j]
        except Exception:
            return [x.strip() for x in s.split(",") if x.strip()]
    if typ == "object":
        if isinstance(v, dict):
            return v
        try:
            return json.loads(str(v))
        except Exception:
            return {}
    return str(v).strip().strip('"')
