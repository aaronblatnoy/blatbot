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
import contextlib
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

from .scopes import ORG_ACCOUNT, OWNER_ACCOUNT, TOOL_DESCRIPTIONS, TOOL_PURPOSE, sends_to_requester, tools_for
from . import hosttools
from .store import Request, sha256
from .taskpick import TYPESAFE_URL

logger = logging.getLogger(__name__)

TZ = ZoneInfo("America/New_York")
DONE, GIVE_UP = "done", "give_up"
DONE_MIN = float(os.getenv("JEV_AGENT_DONE_MIN") or 0.7)
PARALLEL_MIN = float(os.getenv("JEV_AGENT_PARALLEL_MIN") or 0.7)   # extra read tools worth calling alongside the pick
PARALLEL_MAX = int(os.getenv("JEV_AGENT_PARALLEL_MAX") or 3)


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


SEARCH_URL = os.getenv("GATE_SEARCH_URL") or "http://127.0.0.1:8888/search"

VIRTUAL_TOOLS: Dict[str, Dict[str, Any]] = {
    "mcp__host__host_status": {"description": hosttools.DESCRIPTION, "schema": hosttools.SCHEMA},
    # A web search done THROUGH the browser: the Playwright server opens the local
    # SearXNG results page (which queries Google/Bing server-side, so no bot walls)
    # and reads it. Two browser calls, one tool from the agent's point of view.
    "mcp__playwright__browser_search": {
        "description": TOOL_PURPOSE["mcp__playwright__browser_search"],
        "schema": {"type": "object", "required": ["query"],
                   "properties": {"query": {"type": "string", "description": "what to search the web for: a name plus a distinguishing word (school, club, company), or a topic"}}},
    },
}


class _SessionWorker:
    """One MCP server, owned by one asyncio task for its whole life.

    anyio cancel scopes (inside stdio_client and ClientSession) must be entered and
    exited by the same task. The agent graph calls tools from many parallel tasks,
    so every call is handed to this worker over a queue and answered on a future;
    the worker alone touches the session. Workers outlive a request: a server
    started once stays warm until the gateway exits or the worker fails."""

    def __init__(self, server: str, cfg: Dict[str, Any]):
        self.server, self.cfg = server, cfg
        self.queue: "asyncio.Queue[Optional[Tuple[Any, asyncio.Future]]]" = asyncio.Queue()
        self.ready: "asyncio.Future[None]" = asyncio.get_event_loop().create_future()
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.task = asyncio.create_task(self._run(), name=f"mcp-{server}")
        self.failed: Optional[BaseException] = None

    async def _run(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        params = StdioServerParameters(command=self.cfg["command"], args=list(self.cfg.get("args") or []),
                                       env={**os.environ, **(self.cfg.get("env") or {})})
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    for t in listed.tools:
                        self.schemas[f"mcp__{self.server}__{t.name}"] = {"description": t.description or "",
                                                                         "schema": _input_schema(t)}
                    if not self.ready.done():
                        self.ready.set_result(None)
                    while True:
                        item = await self.queue.get()
                        if item is None:
                            return
                        fn, fut = item
                        try:
                            res = await fn(session)
                            if not fut.done():
                                fut.set_result(res)
                        except BaseException as exc:  # noqa: BLE001
                            if not fut.done():
                                fut.set_exception(exc)
        except BaseException as exc:  # noqa: BLE001
            self.failed = exc
            if not self.ready.done():
                self.ready.set_exception(exc)
            # Anyone still queued gets the failure instead of hanging.
            while not self.queue.empty():
                item = self.queue.get_nowait()
                if item is not None and not item[1].done():
                    item[1].set_exception(exc)
            logger.warning("mcp worker %s ended: %s", self.server, exc)

    async def request(self, fn: Any) -> Any:
        if self.failed is not None or self.task.done():
            raise RuntimeError(f"MCP server {self.server} is not running: {self.failed}")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self.queue.put((fn, fut))
        return await fut

    async def close(self) -> None:
        await self.queue.put(None)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(self.task, 10)


class _SessionProxy:
    """What ToolBox hands out in place of a ClientSession: the same two methods,
    executed on the worker's task."""

    def __init__(self, worker: _SessionWorker):
        self._w = worker

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        return await self._w.request(lambda s: s.call_tool(name, arguments=arguments or {}))

    async def list_tools(self) -> Any:
        return await self._w.request(lambda s: s.list_tools())


class McpPool:
    """Process-wide warm pool of MCP server workers, keyed by server name."""

    def __init__(self) -> None:
        self._workers: Dict[str, _SessionWorker] = {}
        self._lock = asyncio.Lock()

    async def get(self, server: str, cfg: Dict[str, Any]) -> _SessionWorker:
        async with self._lock:
            w = self._workers.get(server)
            if w is None or w.failed is not None or w.task.done():
                if w is not None:
                    logger.info("mcp pool: relaunching %s", server)
                w = _SessionWorker(server, cfg)
                self._workers[server] = w
        await w.ready
        return w

    async def close_all(self) -> None:
        for w in list(self._workers.values()):
            await w.close()
        self._workers.clear()


_POOL: Optional[McpPool] = None


def pool() -> McpPool:
    global _POOL
    if _POOL is None:
        _POOL = McpPool()
    return _POOL


class ToolBox:
    """Uniform access to every tool the gate can grant, by full name
    (mcp__<server>__<tool>). External servers come from the process-wide warm
    pool: started on first use, then kept across requests."""

    def __init__(self, inkbox_server: Any, mcp_config: Dict[str, Dict[str, Any]]):
        self._inkbox = inkbox_server
        self._cfg = mcp_config
        self._sessions: Dict[str, Any] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    async def __aenter__(self) -> "ToolBox":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None  # sessions belong to the pool, not to one request

    async def _session(self, server: str) -> Any:
        if server in self._sessions:
            return self._sessions[server]
        cfg = self._cfg.get(server)
        if not cfg:
            raise RuntimeError(f"no launch config for MCP server {server}")
        if cfg.get("type") not in (None, "stdio") or not cfg.get("command"):
            raise RuntimeError(f"MCP server {server} is not a stdio server (type={cfg.get('type')})")
        worker = await pool().get(server, cfg)
        self._schemas.update(worker.schemas)
        proxy = _SessionProxy(worker)
        self._sessions[server] = proxy
        return proxy

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
        if name in VIRTUAL_TOOLS:
            return VIRTUAL_TOOLS[name]
        server, short = split_tool(name)
        if server == "inkbox":
            return (await self._inkbox_tools()).get(short, {"description": "", "schema": {}})
        await self._session(server)
        return self._schemas.get(name, {"description": "", "schema": {}})

    async def call(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "mcp__playwright__browser_search":
            return await self._browser_search(str(args.get("query") or ""))
        if name == "mcp__host__host_status":
            return await hosttools.host_status(args)
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


def _json_results_from_snapshot(snap: str) -> List[Dict[str, Any]]:
    """The browser renders a JSON response as one text node; the snapshot shows it as
    a YAML double-quoted scalar ("{\\"query\\": ...}"). Unquote once, then parse."""
    for line in snap.splitlines():
        m = re.search(r'\]: ("\{.*\}")\s*$', line)
        if not m:
            continue
        try:
            inner = json.loads(m.group(1))          # the YAML/JSON-quoted scalar -> raw JSON text
            data = json.loads(inner)
            return list(data.get("results") or [])
        except Exception:
            continue
    body = snap[snap.find("{"):snap.rfind("}") + 1] if "{" in snap else ""
    try:
        return list(json.loads(body).get("results") or [])
    except Exception:
        return []


async def _toolbox_browser_search(box: "ToolBox", query: str) -> str:
    """Search the web through the browser: open the local SearXNG results (JSON
    view) in Playwright and read them off the page. Titles, links, snippets."""
    import urllib.parse
    url = f"{SEARCH_URL}?q={urllib.parse.quote(query)}&format=json&language=en"
    await box.call("mcp__playwright__browser_navigate", {"url": url})
    snap = await box.call("mcp__playwright__browser_snapshot", {})
    results: List[Dict[str, Any]] = _json_results_from_snapshot(snap)
    if not results:
        urls = [u for u in re.findall(r"https?://[^\s\"'<>]+", snap) if "127.0.0.1" not in u and "localhost" not in u]
        if not urls:
            return f"No results for {query!r}. Try different words."
        return f"Web search results for {query!r} (links only):\n" + "\n".join(f"{i+1}. {u}" for i, u in enumerate(urls[:12]))
    out = []
    for i, r in enumerate(results[:12]):
        line = f"{i+1}. {r.get('title', '').strip()}\n   url: {r.get('url', '').strip()}"
        if r.get("content"):
            line += f"\n   {' '.join(str(r['content']).split())}"
        out.append(line)
    return f"Web search results for {query!r}:\n" + "\n".join(out)


ToolBox._browser_search = _toolbox_browser_search  # type: ignore[attr-defined]


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
        state = _fit(state, reserve=len(json.dumps(questions, ensure_ascii=False, default=str)))
        try:
            return await self._post(state, questions)
        except JudgeTooLarge:
            slim = _slim_state(state)
            logger.warning("jev agent: state still over TypeSafe's limit; retrying with %d large value(s) replaced by markers",
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
        if top and p < floor and min_p is None and len(probs) >= 6:
            # Many options spread the mass: a leader with twice the runner-up's
            # probability is a clear pick even when its absolute probability is low.
            second = max((v for k, v in probs.items() if k != top), default=0.0)
            if p >= 0.2 and p >= 2 * second:
                return top, p, probs
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
                       r"|\berror calling tool\b|\bHttpError \d{3}\b|\breturned \"[^\"]*(error|invalid|denied|not found)"
                       r"|linkedin\.com/authwall|Page Title: Sign (In|Up)|Page Title: Join LinkedIn|Sign in to (view|continue|see)|Verifying you.?re not a bot", re.I)


def _looks_like_error(result: Any) -> bool:
    """Whether a tool result that was not flagged isError still reads as a failure."""
    t = result if isinstance(result, str) else _text(result)
    head = t.lstrip()[:400]
    return bool(_ERROR_RE.search(head))


_FIXED_ARGS = {"user_google_email"}


def _tried_values(steps: List[Dict[str, Any]], tool: str, arg: str, sole: bool = False) -> List[Any]:
    """Values this tool has already been called with for this argument, in order.
    With sole=True only calls where this argument was the ONLY variable one count:
    re-reading the same spreadsheet with a different tab is a new attempt, not a
    repeat, so the spreadsheet id must stay offerable."""
    out: List[Any] = []
    for st in steps:
        a = st.get("args") or {}
        if st["tool"] == tool and arg in a:
            if sole and len([k for k in a if k not in _FIXED_ARGS]) != 1:
                continue
            v = a[arg]
            if v not in out:
                out.append(v)
    return out


_BUDGET = 70000         # characters per TypeSafe request (state + questions) that stay under its ~32k-token ceiling
_REPORT_WHOLE = 60000    # a result up to this size goes to the reply writer whole; larger ones as goal-relevant evidence
_LARGE = 60000          # a single result is read into evidence only when it nears the request budget
_PART = 48000           # largest text one narrowing judgment sees
_LEAF = 6000            # stop narrowing here: small enough to extract candidates from precisely
_MAX_CANDIDATES = 250   # a Choice takes at most 255 options


def _fit(state: Any, reserve: int = 0) -> Any:
    """Keep a judgment request under the API budget. Raw results stay raw; only when
    state + questions (`reserve` chars) would overload Jev is the largest string
    replaced by its digest, largest first, until it fits. Below the budget nothing
    is touched."""
    budget = max(_BUDGET - reserve, 20000)
    if len(json.dumps(state, ensure_ascii=False, default=str)) <= budget:
        return state
    state = json.loads(json.dumps(state, ensure_ascii=False, default=str))

    def biggest(v: Any, path: Tuple = ()) -> Tuple[int, Tuple]:
        if isinstance(v, str):
            return (len(v), path)
        if isinstance(v, dict):
            return max((biggest(x, path + (k,)) for k, x in v.items()), default=(0, path))
        if isinstance(v, list):
            return max((biggest(x, path + (i,)) for i, x in enumerate(v)), default=(0, path))
        return (0, path)

    def setp(v: Any, path: Tuple, val: Any) -> None:
        for k in path[:-1]:
            v = v[k]
        v[path[-1]] = val

    def getp(v: Any, path: Tuple) -> Any:
        for k in path:
            v = v[k]
        return v

    for _ in range(50):
        if len(json.dumps(state, ensure_ascii=False, default=str)) <= budget:
            break
        n, path = biggest(state)
        if n < 2000 or not path:
            break
        setp(state, path, _digest(getp(state, path), force=True))
        logger.info("jev agent: state over budget; replaced a %s-char value at %s with its digest", f"{n:,}", "/".join(map(str, path)))
    return state


def _goal_terms(goal: str) -> List[str]:
    """Distinctive words in the goal worth looking for inside a large result: names,
    capitalised words, numbers, emails."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}|\d{2,}|[\w.+-]+@[\w.-]+", goal or "")
    stop = {"the", "and", "for", "with", "from", "that", "this", "what", "which", "who", "how", "many", "did",
            "does", "check", "find", "look", "please", "tamid", "sjba", "form", "sheet", "responses", "fall",
            "spring", "his", "her", "their", "she", "him", "they", "are", "was", "were", "you", "can", "get",
            "list", "all", "out", "into", "about", "give", "show", "tell", "see", "new", "member", "application"}
    out = []
    for w in words:
        lw = w.lower()
        if lw in stop or lw in out or re.fullmatch(r"(19|20)\d\d", lw):
            continue                                   # years match every dated row: noise
        out.append(lw)
    # Relative days become the forms a sheet or calendar row actually carries:
    # "tomorrow" -> "10/1", "oct 1", "thursday", "thu".
    lowered = (goal or "").lower()
    from datetime import timedelta
    now = datetime.now(TZ)
    rel = {"today": 0, "tonight": 0, "tomorrow": 1, "yesterday": -1}
    for word, delta in rel.items():
        if word in lowered:
            d = now + timedelta(days=delta)
            for form in (f"{d.month}/{d.day}", d.strftime("%b %-d").lower(), d.strftime("%A").lower(), d.strftime("%a").lower(),
                         d.strftime("%Y-%m-%d")):
                if form not in out:
                    out.append(form)
    return out[:14]


_CURRENT_GOAL: List[str] = []   # set per run so digests can show the lines that matter


def _digest(result: Any, force: bool = False) -> Any:
    """A structural breakdown of a tool result: shape, size, keys, counts, header
    lines, how many ids and addresses it holds. Used only when a judgment state
    would otherwise overload Jev (see `_fit`)."""
    t = result if isinstance(result, str) else _text(result)
    if not force and len(t) <= _LARGE:
        return result
    lines = t.splitlines()
    d: Dict[str, Any] = {"size": f"{len(t):,} characters, {len(lines):,} lines", "kind": "text"}
    try:
        j = json.loads(t)
        d["kind"] = "json"
        if isinstance(j, dict):
            d["keys"] = list(j)[:40]
            for k, v in j.items():
                if isinstance(v, list):
                    d[f"{k}_count"] = len(v)
                    if v and isinstance(v[0], dict):
                        d[f"{k}_item_keys"] = list(v[0])[:30]
        elif isinstance(j, list):
            d["items"] = len(j)
            if j and isinstance(j[0], dict):
                d["item_keys"] = list(j[0])[:30]
    except Exception:
        d["first_lines"] = [ln[:200] for ln in lines[:5]]
    d["emails_found"] = len(set(_EMAIL_RE.findall(t)))
    d["ids_found"] = len(set(_LABELED_ID_RE.findall(t)))
    notes = [ln.strip() for ln in lines if ln.strip().startswith("NOTE:")]
    if notes:
        d["notes"] = notes                      # what the agent learned about this result (tabs, etc.)
    if _CURRENT_GOAL:
        hits = [ln.strip()[:400] for ln in lines if any(t in ln.lower() for t in _CURRENT_GOAL)]
        if hits:
            d["lines_matching_the_goal"] = hits[:25]   # the rows that mention the names/terms asked about
            d["matching_line_count"] = len(hits)
    d["note"] = "large result; its values are offered as candidates after a narrowing search"
    return d


def _split(text: str, parts: int) -> List[str]:
    """Split on line boundaries into about `parts` pieces (rows/records stay whole)."""
    lines = text.splitlines(keepends=True)
    if len(lines) < parts * 2:
        n = max(1, len(text) // parts)
        return [text[i:i + n] for i in range(0, len(text), n)]
    per = max(1, len(lines) // parts)
    return ["".join(lines[i:i + per]) for i in range(0, len(lines), per)]


async def narrow(judge: "Judge", text: str, need: str, goal: str, with_score: bool = False) -> Any:
    """Binary-search style: split `text` into parts small enough to judge, ask Jev in
    parallel which part contains what `need` describes, descend into the best one,
    repeat until a leaf small enough to extract from. log(n) rounds, one request per
    part per round."""
    cur = text
    rounds = 0
    best_p = 1.0
    while len(cur) > _LEAF and rounds < 12:
        n = max(2, -(-len(cur) // _PART))
        parts = [p for p in _split(cur, n) if p.strip()]
        if len(parts) < 2:
            break
        async def score(part: str) -> float:
            a = await judge.ask({"goal": goal, "looking_for": need, "part": part},
                                {"q": {"type": "noul", "instructions": {
                                    "question": "Does `part` contain the information described by `looking_for` (the row, entry, "
                                                "record or value that `goal` needs)?",
                                    "criteria": {"true": "It is in this part.", "false": "It is not in this part."}}}})
            return float((a.get("q") or {}).get("noul") or 0.0)
        probs = await asyncio.gather(*[score(p) for p in parts])
        best = max(range(len(parts)), key=lambda i: probs[i])
        logger.info("jev agent: narrowing %s chars -> part %d/%d (p=%.2f)", f"{len(cur):,}", best + 1, len(parts), probs[best])
        best_p = min(best_p, probs[best])
        cur = parts[best]
        rounds += 1
    return (cur, best_p) if with_score else cur


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


def _seen(step: Dict[str, Any]) -> Any:
    """What a judgment sees of a step: the narrowed evidence for a large result,
    the whole result otherwise. The full result is kept for prose and the report."""
    ev = step.get("evidence")
    if ev:
        return {"evidence_relevant_to_goal": ev, "note": f"narrowed from a {len(_text(step['result'])):,}-character result"}
    return step["result"]


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


_TIME_RE = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([ap])\.?m\.?\b", re.I)
_MD_RE = re.compile(r"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?\s*(\d{1,2})/(\d{1,2})\b|\b(\d{1,2})/(\d{1,2})\b", re.I)


def _timed_candidates(text: str, role: str, now: Optional[datetime] = None) -> Dict[str, str]:
    """Specific moments named in the text (a slot like '7:30 PM' on a day like 'Thu 10/1'
    or 'tomorrow'), as ISO datetimes in New York, labelled for Jev to choose from. For an
    END bound each start also offers +30 and +60 minutes, the usual slot lengths."""
    now = now or datetime.now(TZ)
    text = text or ""
    days: List[datetime] = []
    low = text.lower()
    for word, delta in (("today", 0), ("tonight", 0), ("tomorrow", 1), ("yesterday", -1)):
        if word in low:
            days.append((now + timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0))
    for m in _MD_RE.finditer(text):
        mo, da = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        try:
            d = now.replace(month=int(mo), day=int(da), hour=0, minute=0, second=0, microsecond=0)
            if d < now - timedelta(days=200):
                d = d.replace(year=d.year + 1)
            if d not in days:
                days.append(d)
        except ValueError:
            continue
    times: List[Tuple[int, int]] = []
    for m in _TIME_RE.finditer(text):
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
        h = h % 12 + (12 if ap == "p" else 0)
        if (h, mi) not in times:
            times.append((h, mi))
    out: Dict[str, str] = {}
    for d in days[:4]:
        for h, mi in times[:24]:
            start = d.replace(hour=h, minute=mi)
            label = start.strftime("%a %-m/%-d %-I:%M %p")
            if role == "end":
                for mins in (30, 60):
                    e = start + timedelta(minutes=mins)
                    out[e.isoformat()] = f"{label} + {mins} min = {e.strftime('%-I:%M %p')}"
            else:
                out[start.isoformat()] = label
    return out


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
        """Run one request. The LangGraph engine (jevgraph) is the default: the same
        judgments and helpers, with the state and the two parallel fan-outs managed
        by the graph. GATE_AGENT_GRAPH=0 runs the plain loop below instead."""
        if (os.getenv("GATE_AGENT_GRAPH") or "1").strip().lower() not in ("0", "false", "no"):
            try:
                from . import jevgraph
            except ImportError as exc:
                logger.warning("jev agent: langgraph unavailable (%s); using the plain loop", exc)
            else:
                return await jevgraph.run(self, req, context)
        return await self.run_loop(req, context)

    async def run_loop(self, req: Request, context: str = "") -> Dict[str, Any]:
        if sha256(req.prompt) != req.prompt_sha256:
            return {"ok": False, "error": "prompt hash mismatch; refused to run", "tool_calls": []}
        judge, prose = Judge(), Prose(self.router)
        allowed = tools_for(req.scopes)
        started = time.time()
        steps: List[Dict[str, Any]] = []
        facts: Dict[str, Any] = {"request": req.original_message, "task_context": context,
                                 "background": req.prompt if req.prompt != req.original_message else "", **_now_facts(),
                                 "accounts": {"org_google": ORG_ACCOUNT, "owner_google": OWNER_ACCOUNT}}
        try:
            async with ToolBox(self.inkbox_server, self.mcp_config) as box:
                tool_options: Dict[str, str] = {}
                for name in allowed:
                    try:
                        tool_options[name] = TOOL_PURPOSE.get(name) or TOOL_DESCRIPTIONS.get(name) or (await box.schema(name))["description"] or name
                    except ValueError:
                        continue  # a Claude built-in (WebSearch, WebFetch): not callable here
                    except Exception as exc:
                        logger.warning("jev agent: cannot describe %s: %s", name, exc)
                p_done = 0.0
                partial: Optional[float] = None   # set when reads ran out before confidence was reached
                excluded: set = set()   # tools that have nothing new to give this run

                for step in range(self.max_steps):
                    state = {"goal": req.original_message, "task_context": context,
                             "background": facts.get("background") or "", **_now_facts(),
                             "steps_so_far": [{"tool": s["tool"], "args": s["args"], "ok": s["ok"],
                                               "result": _seen(s)} for s in steps]}
                    state = _fit(state)
                    if steps:
                        p_done = await judge.yes(state, {
                            "question": "Can `goal` now be answered, or is the action it asks for complete, WITH CONFIDENCE "
                                        "from `steps_so_far` alone: is every fact the answer needs already in the evidence?",
                            "criteria": {"true": "Everything the goal asked for has been done or, for a question, "
                                                 "the retrieved results contain what is needed to answer it. Counting, "
                                                 "filtering or comparing rows that are already retrieved is NOT a "
                                                 "further tool call; the answer is written from the results. An empty "
                                                 "result from the right place also answers the question.",
                                         "false": "Something the goal asked for has not happened yet, or the results "
                                                  "retrieved so far do not contain the needed information and a "
                                                  "different call is needed."}})
                        logger.info("jev agent step %d: p(done)=%.2f", step + 1, p_done)
                        if p_done >= DONE_MIN:
                            break
                    options = {k: v for k, v in tool_options.items() if k not in excluded} or dict(tool_options)
                    if len(steps) >= 2 and steps[-1]["tool"] == steps[-2]["tool"] and len(options) > 1:
                        # Two calls of the same tool in a row: the next move must be a different
                        # one (open a result, read a sheet), not a third search.
                        options.pop(steps[-1]["tool"], None)
                    options[GIVE_UP] = "The goal cannot be achieved with these tools or the information available."
                    useful = await self._useful_reads(judge, state, options)
                    choice, conf, probs = await judge.choose(
                        state,
                        {"question": "What is the single best next step toward `goal`?",
                         "rules": ["The gateway delivers the findings to the person who asked; never send them the answer.",
                                   "Read before you write. A wrong first pick or an empty lookup means try the next likely one.",
                                   f"Choose {GIVE_UP} only when no tool here can still help."]},
                        options)
                    logger.info("jev agent step %d: %s (conf %.2f)", step + 1, choice, conf)
                    if choice is None and probs:
                        # Not done and no clear pick: narrow to the likeliest few and ask
                        # again, framed as "what do we TRY next". Giving up needs two
                        # unsure answers in a row, not one.
                        top3 = [k for k, _ in sorted(probs.items(), key=lambda kv: -kv[1]) if k != GIVE_UP][:3]
                        if top3:
                            narrowed = {k: options[k] for k in top3 if k in options}
                            narrowed[GIVE_UP] = "Nothing in this list could possibly move the goal forward."
                            choice, conf, probs2 = await judge.choose(
                                state, {"question": "The goal is NOT met yet. Of these, which is the most useful next attempt "
                                                    "(a different query, a different tab or record, a different tool)?",
                                        "rules": ["Prefer trying something over giving up whenever a tool here can still help.",
                                                  "A lookup that returned the wrong thing means: try it again differently."]},
                                narrowed, min_p=0.34)
                            logger.info("jev agent step %d (retry): %s (conf %.2f)", step + 1, choice, conf)
                            if choice == GIVE_UP:
                                choice = None
                    if choice is None:
                        logger.info("jev agent: unsure; top options %s", sorted(probs.items(), key=lambda kv: -kv[1])[:5])
                        if steps and steps[-1]["ok"] and not is_write_tool(steps[-1]["tool"]) and p_done >= DONE_MIN:
                            # Nothing stands out after a read and the goal is plausibly met:
                            # what was gathered is the answer (the gate's reply model phrases it).
                            logger.info("jev agent: no clear next step after reads (p_done %.2f); finishing", p_done)
                            break
                        if steps and all(s["ok"] and not is_write_tool(s["tool"]) for s in steps if s["ok"]) and any(s["ok"] for s in steps):
                            # Reads exhausted: deliver what was gathered with the confidence stated.
                            logger.info("jev agent: reads exhausted at confidence %.2f; delivering findings as partial", p_done)
                            partial = p_done
                            break
                        return self._status(False, f"insufficient evidence to answer with confidence (done {p_done:.2f}); "
                                                   "no further useful step found", steps, judge, prose, started, probs=probs)
                    if choice == GIVE_UP:
                        if steps and any(s["ok"] for s in steps) and all(not is_write_tool(s["tool"]) for s in steps if s["ok"]):
                            if p_done < DONE_MIN:
                                logger.info("jev agent: give_up after reads at confidence %.2f; delivering findings as partial", p_done)
                                partial = p_done
                            break
                        if steps and p_done >= DONE_MIN and all(s["ok"] and not is_write_tool(s["tool"]) for s in steps):
                            # Everything so far was a successful read and the goal is plausibly met:
                            # what was found IS the answer.
                            # The gateway delivers it to the requester; nothing needs sending.
                            logger.info("jev agent: give_up after reads only; finishing with what was found")
                            break
                        return self._status(False, "gave up: goal not achievable with the granted tools", steps, judge, prose, started)
                    args = await self._fill_args(box, judge, prose, choice, facts, steps, req)
                    if args is None:
                        # The call cannot be made yet (an id not in evidence, a value not determinable):
                        # a failed attempt, and the loop goes on to collect what is missing.
                        steps.append({"tool": choice, "args": {}, "ok": False,
                                      "result": "ERROR: this call needs a value (an id or item) that no gathered result "
                                                "contains yet; look the item up first"})
                        if sum(1 for s in steps if s["tool"] == choice and not s["ok"]) >= 3:
                            return self._status(False, f"could not determine arguments for {choice}", steps, judge, prose, started)
                        continue
                    if sends_to_requester(choice, args, self.protected + [req.sender, req.chat_id]):
                        # The gateway delivers the answer; a send to the requester would duplicate it.
                        logger.info("jev agent: refusing %s to the requester; finishing with findings", choice)
                        break
                    if is_destructive(choice, args):
                        about = _mentions(steps, args)
                        if not about:
                            # The target is not described by anything gathered: never ask the owner to
                            # confirm a blind delete. Count it as a failed attempt and keep collecting.
                            logger.info("jev agent: %s target not in evidence; looking things up first", choice.split("__")[-1])
                            steps.append({"tool": choice, "args": args, "ok": False,
                                          "result": "ERROR: the target of this change is not identified in any gathered result; "
                                                    "look the item up (list or search it) before changing it"})
                            continue
                        # Nothing is deleted, cancelled or replaced without the owner's yes. The run
                        # stops here with the exact call and what it refers to; the gate asks Aaron
                        # and performs this one call on "yes".
                        st = self._status(False, "confirmation required", steps, judge, prose, started)
                        st["confirm"] = {"tool": choice, "args": args, "about": about}
                        logger.info("jev agent: %s needs the owner's confirmation; pausing", choice.split("__")[-1])
                        return st
                    if any(s["ok"] and s["tool"] == choice and s["args"] == args for s in steps) and p_done < DONE_MIN:
                        # Same call again: the judge wants this tool but with something different
                        # (another tab, another record). Refill avoiding the values just used.
                        again = await self._fill_args(box, judge, prose, choice, facts, steps, req, avoid=args)
                        if again is not None and again != args:
                            logger.info("jev agent: re-filled %s to avoid a repeat: %s", choice.split("__")[-1],
                                        json.dumps({k: v for k, v in again.items() if k != "user_google_email"})[:200])
                            args = again
                    if any(s["ok"] and s["tool"] == choice and s["args"] == args for s in steps):
                        if p_done >= DONE_MIN:
                            # The best next move is one already made and the goal is plausibly met: done.
                            logger.info("jev agent: would repeat %s with identical arguments; treating as done", choice)
                            break
                        # Not plausibly done and this tool has nothing new to give: try something else.
                        logger.info("jev agent: would repeat %s identically at p_done %.2f; excluding it and re-picking", choice, p_done)
                        excluded.add(choice)
                        continue
                    # Collect in parallel: the pick plus any other read tool Jev rated useful now.
                    batch: List[Tuple[str, Dict[str, Any]]] = [(choice, args)]
                    extras = [t for t, pu in sorted(useful.items(), key=lambda kv: -kv[1])
                              if pu >= PARALLEL_MIN and t != choice and not is_write_tool(t) and t not in excluded][:PARALLEL_MAX]
                    if extras and not is_write_tool(choice):
                        extra_args = await asyncio.gather(*[self._fill_args(box, judge, prose, t, facts, steps, req) for t in extras])
                        for t, a in zip(extras, extra_args):
                            if a is not None and not any(s["ok"] and s["tool"] == t and s["args"] == a for s in steps):
                                batch.append((t, a))
                        if len(batch) > 1:
                            logger.info("jev agent step %d: also collecting %s", step + 1, [t.split("__")[-1] for t, _ in batch[1:]])
                    outcomes = await asyncio.gather(*[self._call(box, judge, t, a, req) for t, a in batch])
                    for (t, a), (ok_call, result, evidence, err) in zip(batch, outcomes):
                        if ok_call:
                            steps.append({"tool": t, "args": a, "result": result, "ok": True, "evidence": evidence})
                            logger.info("jev agent step %d: %s(%s) -> %s", step + 1, t.split("__")[-1],
                                        json.dumps({k: v for k, v in a.items() if k != "user_google_email"}, ensure_ascii=False)[:300],
                                        " ".join(_text(result).split())[:300])
                            facts[f"result_of_{t.split('__')[-1]}_{len(steps)}"] = evidence or _text(result)
                        else:
                            steps.append({"tool": t, "args": a, "result": f"ERROR: {err}", "ok": False})
                            logger.info("jev agent step %d: %s(%s) -> ERROR %s", step + 1, t.split("__")[-1],
                                        json.dumps({k: v for k, v in a.items() if k != "user_google_email"}, ensure_ascii=False)[:300],
                                        " ".join(str(err).split())[:300])
                            same_call_failed = sum(1 for s in steps if s["tool"] == t and s["args"] == a and not s["ok"])
                            total_failed = sum(1 for s in steps if s["tool"] == t and not s["ok"])
                            if t == choice and (same_call_failed >= 2 or total_failed >= 3):
                                # Different arguments are trial and error; the same call failing
                                # twice, or three failures of one tool, means the tool is not the way.
                                return self._status(False, f"{choice} keeps failing: {err}", steps, judge, prose, started)
                else:
                    return self._status(False, "step limit reached", steps, judge, prose, started)
        except Exception as exc:
            logger.exception("jev agent failed for request %s", req.id)
            return self._status(False, str(exc), steps, judge, prose, started)
        return self._status(True, "", steps, judge, prose, started)

    async def _fill_args(self, box: ToolBox, judge: Judge, prose: Prose, tool: str, facts: Dict[str, Any],
                         steps: List[Dict[str, Any]], req: Request,
                         avoid: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
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
                 "known_facts": dict(facts),
                 "steps_so_far": [{"tool": s["tool"], "ok": s["ok"], "result": _seen(s)} for s in steps]}
        state = _fit(state)
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
                # Specific moments named in the goal, the conversation and the evidence (a
                # 7:30 PM slot on Thu 10/1) are offered as values of their own: a write that
                # creates or moves something must land on the slot, not on the day's bounds.
                timed_src = "\n".join([req.original_message, str(facts.get("background") or ""),
                                       *(_text(s["result"])[:4000] for s in steps[-4:])])
                timed = _timed_candidates(timed_src, role)
                for iso, label in list(timed.items())[:24]:
                    opts[f"at::{iso}"] = f"exactly {label}"
                if is_write_tool(tool) and timed:
                    for k in list(ranges):
                        opts[k] = opts[k] + " (a whole-day bound; wrong for a timed slot)"
                opts["write_new"] = "None of these; a specific date or time must be composed from the goal."
                questions[f"range::{name}"] = {"type": "choice", "instructions":
                                               f"`arguments.{name}` is the {role} of a time range or a timed event. "
                                               f"Which value does `goal` mean for THIS call?",
                                               "criteria": opts}
                continue
            cs = await self._candidates(judge, name, str(spec.get("description") or ""), typ, facts, steps, req)
            if not cs and typ == "string" and not _id_like(name):
                # Select instead of generate: phrases already in the message and in
                # small results are offered as choices; DeepSeek only on "write new".
                # Never for ids: a phrase is not an id.
                cs = _span_candidates(req.original_message, steps)
            if avoid and name in avoid and len(cs) > 1:
                cs = [c for c in cs if c != avoid[name]]      # the value just used is not offered again
            variable_args = [k for k in props if k not in _FIXED_ARGS]
            tried = _tried_values(steps, tool, name, sole=True) if len(variable_args) == 1 else []
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
                    args[name] = pick[len("at::"):] if pick.startswith("at::") else _date_ranges()[pick][_date_role(name, desc) or "start"]
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
                if pick is None and name in required:
                    # No candidate clears the bar but one is needed: take the likeliest.
                    # A wrong pick is cheaper than stopping; the loop retries differently.
                    pick = _top(f"pick::{name}", 0.0)
                    logger.info("jev agent: %s.%s taken as best guess (%s)", short, name, pick)
                if pick and pick != "write_new":
                    args[name] = _coerce(cands[name][int(pick[1:])], typ)
                    continue
                if pick is None and name not in required:
                    continue
            if _id_like(name):
                # Ids are selected from evidence, never written. None in evidence yet means
                # this call cannot be made until a lookup has found the item.
                if name in required:
                    logger.info("jev agent: %s.%s has no id in evidence; a lookup must come first", short, name)
                    return None
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
                if _id_like(name) and not _in_evidence(text, steps, facts):
                    # An id must be selected from evidence, never written: a made-up id
                    # deletes the wrong thing or nothing. Leave it unfilled.
                    logger.info("jev agent: rejected written id %s.%s=%r (not in evidence)", short, name, text)
                    if name in required:
                        return None
                    continue
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

    async def _useful_reads(self, judge: "Judge", state: Any, options: Dict[str, str]) -> Dict[str, float]:
        """One request: for every allowed read tool, would calling it NOW add information
        the goal needs? Lets the agent collect from several sources at once."""
        reads = {t: d for t, d in options.items() if t != GIVE_UP and not is_write_tool(t)}
        if len(reads) < 2:
            return {}
        questions = {f"use::{t}": {"type": "noul", "instructions": {
            "tool": d, "question": "Would calling this tool now add information that `goal` needs and that "
                                   "`steps_so_far` does not already contain?"},
            "criteria": {"true": "It would add needed, not-yet-gathered information.",
                         "false": "It is irrelevant to the goal, or its information is already gathered."}}
            for t, d in reads.items()}
        answers = await judge.ask(state, questions)
        return {t: float((answers.get(f"use::{t}") or {}).get("noul") or 0.0) for t in reads}

    async def _call(self, box: ToolBox, judge: "Judge", tool: str, args: Dict[str, Any], req: Request) -> Tuple[bool, Any, Optional[str], str]:
        """One tool call plus its reading: (ok, result, evidence, error)."""
        try:
            result = await box.call(tool, args)
            if _looks_like_error(result):
                # Many MCP tools report failures as ordinary text. Count it as a failed step.
                return False, None, None, (result if isinstance(result, str) else _text(result))
            text = _text(result)
            evidence = None
            if len(text) > _LARGE:
                # Read the result: reduce it to the part that bears on the goal, once, so every
                # later judgment sees rows and facts instead of a size digest.
                leaf, best_p = await narrow(judge, text, "the information `goal` asks for", req.original_message, with_score=True)
                evidence = leaf if best_p >= 0.3 else (
                    f"(This result does not appear to contain what the goal needs: the most relevant part scored {best_p:.2f}.) "
                    f"Closest part:\n{leaf}")
            return True, result, evidence, ""
        except Exception as exc:
            return False, None, None, str(exc)

    async def perform(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run one confirmed call and nothing else."""
        try:
            async with ToolBox(self.inkbox_server, self.mcp_config) as box:
                result = await box.call(tool, args)
            if _looks_like_error(result):
                return {"ok": False, "error": _text(result), "summary": _text(result), "raw": _text(result), "tool_calls": [tool], "wrote": False, "engine": "jev"}
            short = tool.split("__")[-1]
            raw = (f"WRITES PERFORMED: {short}({json.dumps({k: v for k, v in args.items() if k not in _FIXED_ARGS}, ensure_ascii=False)})\n"
                   f"- {short}:\n{_text(result)}\nSTATUS: OK")
            return {"ok": True, "summary": raw, "raw": raw, "tool_calls": [tool], "wrote": True, "engine": "jev"}
        except Exception as exc:
            logger.exception("jev agent: confirmed call %s failed", tool)
            return {"ok": False, "error": str(exc), "summary": str(exc), "raw": str(exc), "tool_calls": [tool], "wrote": False, "engine": "jev"}

    async def _candidates(self, judge: "Judge", name: str, desc: str, typ: str, facts: Dict[str, Any],
                          steps: List[Dict[str, Any]], req: Request) -> List[Any]:
        """Candidate values for one argument. Small sources are scanned whole; a large
        result is first narrowed to the part that holds what this argument needs,
        so a 300-row sheet yields the right row's values, not 300 of them. Newest
        results are scanned first and the list is capped at what a Choice accepts."""
        need = f"{name}: {desc}".strip(": ")
        small: List[Any] = []
        large: List[str] = []
        for st in reversed(steps):
            r = st["result"]
            t = r if isinstance(r, str) else _text(r)
            (large if len(t) > _BUDGET else small).append(t)
        for k, v in facts.items():
            if not k.startswith("result_of_"):
                small.append(v)
        small_facts = {"request": req.original_message}
        out = _candidate_values(name, desc, typ, small_facts, [{"result": src} for src in small])
        if not out and large:
            # Only when nothing small holds a candidate is a huge result worth narrowing.
            for t in large:
                leaf = await narrow(judge, t, need, req.original_message)
                out += _candidate_values(name, desc, typ, {}, [{"result": leaf}])
        return out[:_MAX_CANDIDATES]

    @staticmethod
    def _status(ok: bool, error: str, steps: List[Dict[str, Any]], judge: Judge, prose: Prose,
                started: float, probs: Optional[Dict[str, float]] = None, partial: Optional[float] = None) -> Dict[str, Any]:
        lines = [f"{'Done' if ok else 'Failed'} via Jev agent in {time.time() - started:.1f}s "
                 f"({len(steps)} tool call(s), {judge.calls} Jev judgment(s), {prose.calls} prose call(s))."]
        if partial is not None:
            lines.append(f"CONFIDENCE: {partial:.2f}. The lookups available were exhausted before the answer was certain; "
                         "the reply must say what the results show and what could not be established.")
        writes = [s for s in steps if s["ok"] and is_write_tool(s["tool"])]
        lines.append("WRITES PERFORMED: " + ("; ".join(
            f"{s['tool'].split('__')[-1]}({json.dumps({k: v for k, v in s['args'].items() if k not in _FIXED_ARGS}, ensure_ascii=False)})"
            for s in writes) if writes else "none. Nothing was sent, created, changed or deleted."))
        # Two renderings. `raw` carries every result in full (the record, the owner's
        # findings file). `summary` is what the reply writer and the grounding judge read:
        # a result small enough is given whole; a large one is given as the evidence the
        # agent's own goal-directed narrowing extracted from it, plus every line that
        # names a term of the goal, with its size stated. Relevance selection, not a cut.
        full, brief = [], []
        for s in steps:
            r = s["result"] if isinstance(s["result"], str) else json.dumps(s["result"], ensure_ascii=False, default=str)
            full.append(f"- {s['tool'].split('__')[-1]}:\n{r}")
            if len(r) <= _REPORT_WHOLE:
                brief.append(f"- {s['tool'].split('__')[-1]}:\n{r}")
                continue
            part = [f"- {s['tool'].split('__')[-1]}: ({len(r):,} characters; the parts relevant to the goal follow)"]
            ev = s.get("evidence")
            if ev:
                part.append(ev if isinstance(ev, str) else json.dumps(ev, ensure_ascii=False, default=str))
            if _CURRENT_GOAL:
                hits = [ln.strip() for ln in r.splitlines() if any(t in ln.lower() for t in _CURRENT_GOAL)]
                if hits:
                    part.append("lines naming the goal's terms:\n" + "\n".join(hits[:60]))
            if len(part) == 1:
                part.append("\n".join(r.splitlines()[:40]))
            part.append(f"(this result has {len(r.splitlines()):,} lines in all; the rest were not shown here, "
                        "not missing from the source)")
            brief.append("\n".join(part))
        tail = ([f"Reason: {error}"] if error else []) + ["STATUS: OK" if ok else "STATUS: FAILED"]
        raw = "\n".join(lines + full + tail)
        summary = "\n".join(lines + brief + tail)
        wrote = any(s["ok"] and is_write_tool(s["tool"]) for s in steps)
        return {"ok": ok, "summary": summary, "raw": raw, "tool_calls": [s["tool"] for s in steps], "wrote": wrote, "partial": partial,
                "steps": [{"tool": s["tool"], "args": s["args"], "ok": s["ok"]} for s in steps],
                "error": error or None, "engine": "jev", "jev_calls": judge.calls, "prose_calls": prose.calls,
                "seconds": round(time.time() - started, 1), "probs": probs}



_WRITE_RE = re.compile(r"(send|create|update|modify|delete|manage|append|publish|place_call|move|set_|replace|import|insert|format|resize|run_script|reply|complete|fail"
                       r"|browser_click|browser_type|browser_fill_form|browser_select_option|browser_press_key|browser_drag|browser_drop|browser_file_upload|browser_handle_dialog)", re.I)


_DESTRUCTIVE_RE = re.compile(os.getenv("GATE_CONFIRM_TOOLS") or r"(delete|remove|trash|cancel|replace|clear|purge|publish|deploy|redeploy|restart|stop|submit_assignment|post_discussion|reply_discussion|bulk_env)", re.I)


def _mentions(steps: List[Dict[str, Any]], args: Dict[str, Any]) -> List[str]:
    """Earlier result lines that mention one of the argument values: what the call
    is about, in the words the tools used (the event title and time, the row)."""
    vals = [str(v) for k, v in args.items() if k not in _FIXED_ARGS and isinstance(v, (str, int)) and len(str(v)) >= 6]
    out: List[str] = []
    for st in steps:
        t = st.get("evidence") or st["result"]
        t = t if isinstance(t, str) else _text(t)
        for ln in t.splitlines():
            if any(v in ln for v in vals) and ln.strip() not in out:
                out.append(ln.strip())
    return out[:10]


def is_destructive(name: str, args: Dict[str, Any]) -> bool:
    """A write the owner must confirm first: deleting, removing, cancelling, replacing.
    Also manage_event/update with a delete-like action argument."""
    short = name.split("__")[-1]
    if _DESTRUCTIVE_RE.search(short):
        return True
    act = str(args.get("action") or "").lower()
    return bool(act) and bool(_DESTRUCTIVE_RE.search(act))


def is_write_tool(name: str) -> bool:
    """Whether a tool changes the world. A successful write must never be
    retried by the Claude fallback (double email, duplicate event)."""
    return bool(_WRITE_RE.search(name.split("__")[-1]))


_LABELED_ID_RE = re.compile(r"(?:\b(?:ID|id|Id)|[A-Za-z_]+(?:Id|ID|_id))\"?\s*[:=]\s*\"?([^\s\"'|,)>}\]]+)")
_ID_RE = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{16,}\b")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?1?[ (.-]*\d{3}[ ).-]*\d{3}[ .-]*\d{4}")


_TAB_RE = re.compile(r'^\s*-\s+"([^"]+)"\s+\(ID:\s*\d+\)\s*\|\s*Size:', re.M)


def _sheet_tabs(info_text: str) -> List[str]:
    """Tab names out of a get_spreadsheet_info result."""
    return [m for m in _TAB_RE.findall(info_text or "")]


_SPAN_STOP = {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "is", "are", "was", "it", "this", "that",
              "what", "which", "who", "how", "many", "did", "do", "does", "check", "find", "look", "up", "please", "me",
              "my", "his", "her", "their", "he", "she", "they", "see", "get", "tell", "show", "give", "list", "out", "with",
              "from", "into", "about", "can", "you", "i", "we", "be", "there", "if", "so", "now", "just"}


def _in_evidence(value: str, steps: List[Dict[str, Any]], facts: Dict[str, Any]) -> bool:
    """Whether a value appears verbatim in any tool result or fact gathered so far."""
    v = (value or "").strip()
    if len(v) < 6:
        return False
    for st in steps:
        if v in _text(st.get("result")):
            return True
    return any(v in _text(x) for k, x in facts.items() if k not in ("request",))


def _id_like(name: str) -> bool:
    n = name.lower()
    return n.endswith("_id") or n.endswith("id") and len(n) <= 12 or n in ("id", "page_token", "token", "cursor")


def _span_candidates(message: str, steps: List[Dict[str, Any]], limit: int = 120) -> List[str]:
    """Phrases a text argument could be, taken from words already in play: quoted
    strings, runs of capitalised words, 1-5 word phrases of the message (not made
    only of stop words), and short cell-like values from small results."""
    out: List[str] = []

    def add(v: str) -> None:
        v = " ".join(v.split()).strip(" ,.;:!?\"'")
        if 2 <= len(v) <= 80 and v.lower() not in {o.lower() for o in out}:
            out.append(v)

    for q in re.findall(r"\"([^\"]{2,80})\"|'([^']{2,80})'", message or ""):
        add(q[0] or q[1])
    for m in re.findall(r"(?:[A-Z][\w'&-]*)(?:\s+[A-Z][\w'&-]*)+", message or ""):
        add(m)
    words = re.findall(r"[\w'&.@-]+", message or "")
    for n in (5, 4, 3, 2, 1):
        for i in range(0, max(0, len(words) - n + 1)):
            gram = words[i:i + n]
            if all(w.lower() in _SPAN_STOP for w in gram):
                continue
            if gram[0].lower() in _SPAN_STOP or gram[-1].lower() in _SPAN_STOP:
                continue
            add(" ".join(gram))
            if len(out) >= limit:
                return out[:limit]
    for st in steps[-3:]:
        t = st.get("evidence") or st.get("result")
        t = t if isinstance(t, str) else _text(t)
        if len(t) > _LARGE:
            continue
        for cell in re.findall(r"'([^']{2,60})'|\"([^\"]{2,60})\"", t)[:80]:
            add(cell[0] or cell[1])
            if len(out) >= limit:
                break
    return out[:limit]


def _candidate_values(name: str, desc: str, typ: str, facts: Dict[str, Any], steps: List[Dict[str, Any]]) -> List[Any]:
    """Values already in play that could fill this argument: ids, addresses, phones,
    and short strings from tool results. Jev selects among them."""
    blob = "\n".join(_text(v) for v in facts.values()) + "\n" + "\n".join(_text(s["result"]) for s in steps)
    lname = (name + " " + desc).lower()
    out: List[Any] = []
    if name in ("range_name", "range", "sheet_name", "tab", "worksheet", "sheet"):
        tabs = _sheet_tabs(blob)
        tabs += re.findall(r'tabs: ((?:"[^"]+"(?:, )?)+)', blob) and [t for grp in re.findall(r'tabs: ((?:"[^"]+"(?:, )?)+)', blob) for t in re.findall(r'"([^"]+)"', grp)] or []
        seen_t: List[str] = []
        for t in tabs:
            if t not in seen_t:
                seen_t.append(t)
        out += [f"'{t}'!A1:Z1000" for t in seen_t]
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
