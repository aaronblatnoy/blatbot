"""Suggestions: a few times a day Blatbot looks at Aaron's recent context and offers at
most two concrete things it could do for him right now, or stays quiet.

Aaron, 2026-11-27: "i like how claude duet sends me like suggestions of things. having
that for blatbot would be cool" / "like the proactive nature of it is nice".

How it works:

* A pass runs at the times in SUGGESTION_TIMES (New York time) while GATE_SUGGESTIONS is
  on. Each slot is claimed once in `suggestion_passes`, so a restart never repeats it; a
  slot missed by more than PASS_WINDOW_S is skipped, not caught up hours late.
* The pass reads, in code, what Aaron did with Blatbot lately: his recent messages, his
  requests that failed, expired, were declined or are waiting on him, open follow-ups,
  schedules, the long-term memory graph, and what was already suggested. One model call
  with no tools turns that into zero, one or two suggestions. Nothing is done by the pass.
* Each suggestion reserves a request id (state "suggested", never "pending", so it never
  takes part in a bare yes or in the 24 hour approval expiry) and is sent to Aaron alone,
  on his most recent surface, as outbox kind "proactive" with the usual "#id yes / no".
* "#id yes" turns that request into an ordinary owner request with his standing scopes
  and runs it. "#id no" dismisses it; a dismissed or accepted suggestion is never made
  again (the ledger is checked by meaning, not only by exact text). Unanswered ones
  expire after a day without a text.
* Quiet by default: no pass while two suggestions are still unanswered, no text when the
  model finds nothing worth his time, and never more than MAX_PER_PASS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from . import ownertext
from . import settings as _settings

logger = logging.getLogger(__name__)

TZ = ZoneInfo("America/New_York")
MAX_PER_PASS = 2
MAX_UNANSWERED = 2              # this many still open: the pass stays quiet
PASS_WINDOW_S = 45 * 60         # a slot later than this after its time is skipped
EXPIRE_S = 24 * 3600            # an unanswered suggestion lapses after this
LOOKBACK_S = 7 * 24 * 3600      # how far back requests are read
MESSAGE_LOOKBACK_S = 72 * 3600  # how far back messages are read
REPEAT_WINDOW_S = 90 * 24 * 3600
SIMILAR = 0.6                   # token overlap at which two suggestions are the same one
SUGGESTED = "suggested"         # the request state that reserves a suggestion's #id
SHARED_TASK_TITLE = "Blatbot suggestions"

SCHEMA = """
CREATE TABLE IF NOT EXISTS suggestions (
  id INTEGER PRIMARY KEY,
  pass_slot TEXT NOT NULL,
  text TEXT NOT NULL,
  task TEXT NOT NULL,
  why TEXT NOT NULL DEFAULT '',
  source_refs TEXT NOT NULL DEFAULT '[]',
  dedupe_key TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending',
  created_at REAL NOT NULL,
  decided_at REAL,
  request_id INTEGER
);
CREATE INDEX IF NOT EXISTS suggestions_status ON suggestions(status, created_at);
CREATE TABLE IF NOT EXISTS suggestion_passes (
  slot TEXT PRIMARY KEY,
  started_at REAL NOT NULL,
  finished_at REAL,
  outcome TEXT NOT NULL DEFAULT 'running',
  sent INTEGER NOT NULL DEFAULT 0,
  note TEXT NOT NULL DEFAULT ''
);
"""
SCHEMA_PG = (SCHEMA.replace("INTEGER PRIMARY KEY", "BIGINT PRIMARY KEY")
             .replace("request_id INTEGER", "request_id BIGINT")
             .replace(" REAL", " DOUBLE PRECISION"))

YES_WORDS = {"yes", "y", "ok", "okay", "sure", "go", "run", "approve", "do it", "yes please", "yep", "yeah"}
NO_WORDS = {"no", "n", "nah", "nope", "skip", "drop", "dismiss", "not now", "no thanks", "pass", "reject",
            "decline", "cancel"}
_REPLY_RE = re.compile(r"^\s*#(\d+)\s*[:\-]?\s*(.*?)\s*[.!]*\s*$", re.S)
_STOP = {"the", "a", "an", "to", "for", "of", "and", "or", "on", "in", "your", "you", "i", "it", "that",
         "this", "with", "from", "about", "can", "could", "me", "my", "is", "be", "now", "so", "up", "at"}

SYSTEM = (
    "You are Blatbot, Aaron's always-on assistant. A few times a day you look at what he has "
    "been doing with you and offer at most two concrete things you could do for him right now, "
    "the way a sharp chief of staff would. You do not act; you only propose.\n\n"
    "A good suggestion is specific, grounded in something in the context (a request that failed "
    "or expired, a thread he left hanging, a follow-up that is due, a schedule that keeps failing, "
    "something he said he wanted), and doable with the systems listed. It saves him time today. "
    "A bad one is generic ('want me to check your email?'), repeats one already made, guesses at "
    "facts not in the context, or needs someone else's yes.\n\n"
    "Staying quiet is the right answer most of the time. If nothing clears that bar, return no "
    "suggestions.\n\n"
    "Return JSON only, with this shape:\n"
    '{"suggestions": [{"text": "...", "task": "...", "why": "...", "refs": ["..."]}], '
    '"quiet_reason": "..."}\n'
    "text: what you would do, to Aaron, one or two short sentences, first person, plain, no "
    "question mark at the end, no greeting, no markdown, no em dashes. Name the thing.\n"
    "task: the instruction you would run if he says yes, written as Aaron asking you, complete "
    "enough to run on its own.\n"
    "why: one short line on what in the context prompted it.\n"
    "refs: the context items it rests on, like 'request #625' or 'follow-up #12'.\n"
    "quiet_reason: when there are no suggestions, why.")


# -- storage ------------------------------------------------------------------

def _ensure(store: Any) -> bool:
    if getattr(store, "_suggestions_ready", False):
        return True
    try:
        with store._lock:
            store._db.executescript(SCHEMA_PG if store._is_pg else SCHEMA)
            store._db.commit()
        store._suggestions_ready = True
        return True
    except Exception as exc:
        try:
            store._db.rollback()
        except Exception:
            pass
        logger.warning("[suggest] storage unavailable: %s", exc)
        return False


def _rows(store: Any, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    with store._lock:
        return [dict(r) for r in store._db.execute(sql, params).fetchall()]


def _write(store: Any, sql: str, params: Tuple[Any, ...] = ()) -> int:
    with store._lock:
        cur = store._db.execute(sql, params)
        store._db.commit()
        return int(getattr(cur, "rowcount", 0) or 0)


def get(store: Any, sid: int) -> Optional[Dict[str, Any]]:
    if not _ensure(store):
        return None
    rows = _rows(store, "SELECT * FROM suggestions WHERE id=?", (int(sid),))
    return rows[0] if rows else None


def recent(store: Any, limit: int = 50) -> List[Dict[str, Any]]:
    if not _ensure(store):
        return []
    return _rows(store, "SELECT * FROM suggestions ORDER BY created_at DESC LIMIT ?", (int(limit),))


def passes(store: Any, limit: int = 20) -> List[Dict[str, Any]]:
    if not _ensure(store):
        return []
    return _rows(store, "SELECT * FROM suggestion_passes ORDER BY started_at DESC LIMIT ?", (int(limit),))


def pending(store: Any) -> List[Dict[str, Any]]:
    if not _ensure(store):
        return []
    return _rows(store, "SELECT * FROM suggestions WHERE status='pending' ORDER BY created_at")


def _set_status(store: Any, sid: int, status: str, request_id: Optional[int] = None) -> bool:
    """Move a pending suggestion on. False when it had already been decided."""
    return _write(store, "UPDATE suggestions SET status=?, decided_at=?, request_id=COALESCE(?, request_id) "
                         "WHERE id=? AND status='pending'",
                  (status, time.time(), request_id, int(sid))) > 0


def _claim_slot(store: Any, slot: str) -> bool:
    return _write(store, "INSERT OR IGNORE INTO suggestion_passes(slot, started_at) VALUES(?, ?)",
                  (slot, time.time())) > 0


def _finish_slot(store: Any, slot: str, outcome: str, sent: int = 0, note: str = "") -> None:
    try:
        _write(store, "UPDATE suggestion_passes SET finished_at=?, outcome=?, sent=?, note=? WHERE slot=?",
               (time.time(), outcome, int(sent), note[:500], slot))
    except Exception as exc:
        logger.warning("[suggest] could not record pass %s: %s", slot, exc)


# -- schedule -----------------------------------------------------------------

def enabled() -> bool:
    return bool(_settings.get("GATE_SUGGESTIONS"))


def times() -> List[Tuple[int, int]]:
    raw = str(_settings.get("SUGGESTION_TIMES") or "")
    out: List[Tuple[int, int]] = []
    for part in re.split(r"[,\s;]+", raw):
        m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", part.strip().lower())
        if not m:
            continue
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if m.group(3) == "pm" and hour < 12:
            hour += 12
        if m.group(3) == "am" and hour == 12:
            hour = 0
        if 0 <= hour < 24 and 0 <= minute < 60:
            out.append((hour, minute))
    return sorted(set(out))


def due_slot(now: Optional[float] = None) -> Optional[str]:
    """The slot whose time has come within the window, as '2026-11-27T09:00', or None."""
    local = datetime.fromtimestamp(time.time() if now is None else now, TZ)
    for hour, minute in times():
        at = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if timedelta(0) <= local - at < timedelta(seconds=PASS_WINDOW_S):
            return at.strftime("%Y-%m-%dT%H:%M")
    return None


async def loop(manager: Any) -> None:
    while True:
        await asyncio.sleep(60)
        try:
            if not enabled():
                continue
            expire_old(manager.store)
            slot = due_slot()
            if slot:
                await run_pass(manager, slot)
        except Exception as exc:
            logger.warning("[suggest] loop failed: %s", exc)


def expire_old(store: Any, now: Optional[float] = None) -> int:
    if not _ensure(store):
        return 0
    cutoff = (time.time() if now is None else now) - EXPIRE_S
    old = _rows(store, "SELECT id FROM suggestions WHERE status='pending' AND created_at<?", (cutoff,))
    for row in old:
        if _set_status(store, row["id"], "expired"):
            try:
                req = store.get_request(int(row["id"]))
                if req is not None and req.state == SUGGESTED:
                    store.set_state(req.id, "expired")
            except Exception as exc:
                logger.debug("[suggest] could not expire request #%s: %s", row["id"], exc)
    return len(old)


# -- context ------------------------------------------------------------------

def _ago(ts: Any, now: float) -> str:
    try:
        secs = max(0, now - float(ts))
    except (TypeError, ValueError):
        return ""
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def _short(text: Any, limit: int = 220) -> str:
    return ownertext.one_line(text, limit)


def _owner_request(manager: Any, req: Any) -> bool:
    try:
        return bool(manager._is_owner_request(req)) or bool(manager._asked_by_owner(req))
    except Exception:
        return False


async def gather(manager: Any, now: Optional[float] = None) -> Dict[str, Any]:
    """Everything the pass may read, as short text blocks plus the raw counts."""
    from .manager import OWNER_THREAD
    store = manager.store
    now = time.time() if now is None else now
    blocks: Dict[str, List[str]] = {}

    # His recent conversation with Blatbot, across his own threads.
    msgs: List[Tuple[float, str]] = []
    try:
        threads = [OWNER_THREAD] + [c for c in store.owner_threads() if c != OWNER_THREAD][:4]
        for cid in threads:
            route = store.thread_route(cid) or {}
            meta = route.get("meta") or {}
            group = meta.get("group_title") if (meta.get("conversation_kind") == "group" or meta.get("group_title")) else ""
            for m in store.history(cid, 30):
                if now - float(m.get("created_at") or 0) > MESSAGE_LOOKBACK_S:
                    continue
                if m.get("kind") not in ("inbound", "outbound"):
                    continue
                if m["kind"] == "outbound":
                    who = "Blatbot" + (f" in {group}" if group else "")
                else:
                    who = f"In {group}" if group else "Aaron"
                text = ownertext.strip_group_tag(m["text"])
                msgs.append((float(m["created_at"]), f"{_ago(m['created_at'], now)} {who}: {_short(text, 260)}"))
    except Exception as exc:
        logger.warning("[suggest] could not read messages: %s", exc)
    msgs.sort()
    blocks["recent messages"] = [line for _, line in msgs[-40:]]

    # Requests that did not end well, or are waiting on him.
    unfinished: List[str] = []
    try:
        for req in store.recent_requests(limit=200):
            if now - float(req.created_at or 0) > LOOKBACK_S or req.state == SUGGESTED:
                continue
            if not _owner_request(manager, req):
                continue
            status = req.status or {}
            if req.state in ("failed", "expired", "rejected") or (req.state == "pending"):
                note = status.get("error") or status.get("result_line") or status.get("summary") or ""
                unfinished.append(f"request #{req.id} {req.state} {_ago(req.created_at, now)}: "
                                  f"{_short(req.summary, 160)}" + (f" ({_short(note, 140)})" if note else ""))
    except Exception as exc:
        logger.warning("[suggest] could not read requests: %s", exc)
    blocks["requests that failed, expired, were declined or wait on Aaron"] = unfinished[:15]

    try:
        from .hosttools import followup_time
        blocks["open follow-ups"] = [f"follow-up #{f['id']} due {followup_time(f['due_at'])}: {_short(f['note'], 160)}"
                                     for f in store.pending_followups()][:10]
    except Exception as exc:
        logger.debug("[suggest] no follow-ups: %s", exc)
        blocks["open follow-ups"] = []

    sched: List[str] = []
    try:
        for s in store.schedules(["active", "paused"]):
            last = store.last_schedule_run(s.id) or {}
            sched.append(f"schedule #{s.id} {s.state}: {_short(s.title, 120)}"
                         + (f", last run {last.get('outcome')}" if last.get("outcome") else ""))
    except Exception as exc:
        logger.debug("[suggest] no schedules: %s", exc)
    blocks["schedules"] = sched[:12]

    made: List[str] = []
    for row in recent(store, 25):
        made.append(f"#{row['id']} {row['status']} {_ago(row['created_at'], now)}: {_short(row['text'], 160)}")
    blocks["already suggested (never repeat these)"] = made

    memory: List[str] = []
    try:
        from . import graphmem
        if graphmem.enabled() and msgs:
            query = " ".join(line.split(": ", 1)[-1] for _, line in msgs[-8:])[:600]
            facts = await graphmem.recall_facts(query, OWNER_THREAD, True)
            memory = [_short(f.get("fact") or f.get("text") or "", 200) for f in facts][:10]
    except Exception as exc:
        logger.debug("[suggest] memory unavailable: %s", exc)
    blocks["long-term memory"] = [m for m in memory if m]

    try:
        from .scopes import available_systems
        blocks["systems Blatbot can use"] = [f"{k}: {v}" for k, v in available_systems().items()]
    except Exception:
        blocks["systems Blatbot can use"] = []
    return blocks


def render(blocks: Dict[str, List[str]]) -> str:
    from .router import now_line
    parts = [f"Now: {now_line()}"]
    for title, lines in blocks.items():
        parts.append(f"--- {title.upper()} ---\n" + ("\n".join(lines) if lines else "(none)"))
    return "\n\n".join(parts)


def has_signal(blocks: Dict[str, List[str]]) -> bool:
    keys = ("recent messages", "requests that failed, expired, were declined or wait on Aaron",
            "open follow-ups")
    return any(blocks.get(k) for k in keys)


# -- the model call -------------------------------------------------------------

async def ask_model(system: str, prompt: str) -> str:
    """One Claude turn, no tools. Kept apart so tests and the harness can replace it."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query
    model = str(_settings.get("GATE_FAST_MODEL") or "sonnet")
    options = ClaudeAgentOptions(model=model, system_prompt=system, allowed_tools=[], tools=[],
                                 max_turns=1, setting_sources=[], strict_mcp_config=True)
    out: List[str] = []
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            out.extend(b.text for b in msg.content if isinstance(b, TextBlock))
    return "\n".join(out).strip()


def parse(raw: str) -> Tuple[List[Dict[str, Any]], str]:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return [], "no JSON in the answer"
    try:
        data = json.loads(m.group(0))
    except Exception:
        return [], "the answer was not valid JSON"
    out = []
    for item in data.get("suggestions") or []:
        if not isinstance(item, dict):
            continue
        say = ownertext.clean(item.get("text") or "")
        task = " ".join(str(item.get("task") or "").split())
        if not say or not task:
            continue
        refs = [str(r)[:80] for r in (item.get("refs") or []) if str(r).strip()][:6]
        out.append({"text": say[:600], "task": task[:2000], "why": _short(item.get("why"), 300), "refs": refs})
    return out, " ".join(str(data.get("quiet_reason") or "").split())[:300]


# -- dedupe -------------------------------------------------------------------

def key_of(text: str) -> str:
    words = re.findall(r"[a-z0-9#]+", (text or "").lower())
    return " ".join(sorted({w for w in words if w not in _STOP and len(w) > 1}))


def _overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_repeat(store: Any, text: str, task: str, now: Optional[float] = None) -> bool:
    now = time.time() if now is None else now
    key, task_key = key_of(text), key_of(task)
    for row in _rows(store, "SELECT text, task, dedupe_key FROM suggestions WHERE created_at>?",
                     (now - REPEAT_WINDOW_S,)):
        if (row["dedupe_key"] == key or _overlap(row["dedupe_key"], key) >= SIMILAR
                or _overlap(key_of(row["task"]), task_key) >= SIMILAR):
            return True
    return False


# -- the pass -----------------------------------------------------------------

def _owner_home(manager: Any) -> Tuple[str, str, str]:
    """(chat_id, mode, sender) of Aaron's most recent direct thread with Blatbot."""
    from .manager import OWNER_THREAD
    store = manager.store
    try:
        for row in store.thread_routes(store.owner_threads()):
            meta = row.get("meta") or {}
            if meta.get("conversation_kind") == "group" or meta.get("group_title"):
                continue
            return row["chat_id"], row["mode"] or "imessage", str(meta.get("sender") or "Aaron")
    except Exception as exc:
        logger.debug("[suggest] no owner thread: %s", exc)
    return OWNER_THREAD, "imessage", "Aaron"


def _shared_task(store: Any) -> int:
    rows = _rows(store, "SELECT id FROM tasks WHERE title=? ORDER BY id LIMIT 1", (SHARED_TASK_TITLE,))
    if rows:
        return int(rows[0]["id"])
    task = store.create_task(SHARED_TASK_TITLE, [])
    try:
        _write(store, "UPDATE tasks SET state='done' WHERE id=?", (int(task["id"]),))
    except Exception:
        pass
    return int(task["id"])


def message_for(sid: int, text: str) -> str:
    return ownertext.compose("ask", "Something I can do for you", body=text,
                             handle=ownertext.reply_handle(sid))


async def run_pass(manager: Any, slot: str, *, model: Optional[Callable[[str, str], Awaitable[str]]] = None,
                   now: Optional[float] = None) -> Dict[str, Any]:
    """One pass for one slot. Returns {"outcome": sent|quiet|skipped|failed, "sent": [...ids], "note"}."""
    store = manager.store
    if not _ensure(store):
        return {"outcome": "failed", "sent": [], "note": "storage unavailable"}
    if not _claim_slot(store, slot):
        return {"outcome": "skipped", "sent": [], "note": "slot already ran"}
    now = time.time() if now is None else now
    try:
        expire_old(store, now)
        open_ones = pending(store)
        if len(open_ones) >= MAX_UNANSWERED:
            note = f"{len(open_ones)} suggestions still unanswered"
            _finish_slot(store, slot, "quiet", 0, note)
            logger.info("[suggest] %s quiet: %s", slot, note)
            return {"outcome": "quiet", "sent": [], "note": note}
        blocks = await gather(manager, now)
        if not has_signal(blocks):
            _finish_slot(store, slot, "quiet", 0, "nothing recent to go on")
            return {"outcome": "quiet", "sent": [], "note": "nothing recent to go on"}
        raw = await (model or ask_model)(SYSTEM, render(blocks))
        items, quiet_reason = parse(raw)
        room = MAX_PER_PASS - len(open_ones) if open_ones else MAX_PER_PASS
        fresh = []
        for item in items:
            if len(fresh) >= room:
                break
            if is_repeat(store, item["text"], item["task"], now) or any(
                    _overlap(key_of(item["text"]), key_of(f["text"])) >= SIMILAR for f in fresh):
                continue
            fresh.append(item)
        if not fresh:
            note = quiet_reason or ("every idea was a repeat" if items else "nothing worth sending")
            _finish_slot(store, slot, "quiet", 0, note)
            logger.info("[suggest] %s quiet: %s", slot, note)
            return {"outcome": "quiet", "sent": [], "note": note}
        sent_ids = await _deliver(manager, slot, fresh)
        _finish_slot(store, slot, "sent" if sent_ids else "failed", len(sent_ids),
                     "" if sent_ids else "delivery failed")
        return {"outcome": "sent" if sent_ids else "failed", "sent": sent_ids, "note": ""}
    except Exception as exc:
        logger.warning("[suggest] pass %s failed: %s", slot, exc)
        _finish_slot(store, slot, "failed", 0, str(exc))
        return {"outcome": "failed", "sent": [], "note": str(exc)}


async def _deliver(manager: Any, slot: str, items: List[Dict[str, Any]]) -> List[int]:
    from . import outcomes
    store = manager.store
    chat_id, mode, sender = _owner_home(manager)
    task_id = _shared_task(store)
    dest = outcomes._to_owner(manager, None, None)
    sent: List[int] = []
    for item in items:
        req = store.create_request(
            chat_id=chat_id, sender=sender, sender_name="Aaron", mode=mode, subject="",
            original_message=item["task"], summary=f"Suggestion: {_short(item['text'], 140)}",
            scopes=[], prompt=item["task"], state=SUGGESTED, task_id=task_id)
        store.set_state(req.id, SUGGESTED, status={"suggestion_id": req.id, "why": item["why"]})
        _write(store, "INSERT INTO suggestions(id, pass_slot, text, task, why, source_refs, dedupe_key, "
                      "status, created_at) VALUES(?,?,?,?,?,?,?,'pending',?)",
               (req.id, slot, item["text"], item["task"], item["why"], json.dumps(item["refs"]),
                key_of(item["text"]), time.time()))
        text = message_for(req.id, item["text"])
        try:
            await manager.send(dest["target"], text, dest["mode"], dest["meta"], chat_id=dest["record"],
                               kind="proactive")
            sent.append(req.id)
            logger.info("[suggest] %s sent #%s to Aaron on %s", slot, req.id, dest["mode"])
        except Exception as exc:
            logger.warning("[suggest] #%s could not be sent: %s", req.id, exc)
            _set_status(store, req.id, "failed")
            store.set_state(req.id, "failed", status={"suggestion_id": req.id, "error": str(exc)})
    return sent


# -- his answer ---------------------------------------------------------------

def _word(rest: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", rest.lower()).split())


async def handle_reply(manager: Any, session: Any, text: str) -> bool:
    """Aaron's "#id yes" / "#id no" to a suggestion, and "suggest now". True when consumed."""
    t = (text or "").strip()
    if t.lower().rstrip(".!") in ("suggest now", "suggestions now", "any suggestions", "any suggestions?"):
        slot = "manual-" + datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S")
        out = await run_pass(manager, slot)
        if out["outcome"] != "sent":
            await manager.send_to_approver("Nothing worth suggesting right now.")
        return True
    m = _REPLY_RE.match(t)
    if not m:
        return False
    if not _ensure(manager.store):
        return False
    row = get(manager.store, int(m.group(1)))
    if row is None:
        return False
    word = _word(m.group(2))
    if word in YES_WORDS:
        decision = "yes"
    elif word in NO_WORDS:
        decision = "no"
    else:
        return False
    out = await decide(manager, int(row["id"]), decision, session=session)
    if out.get("reply"):
        await manager.send_to_approver(out["reply"])
    return True


async def decide(manager: Any, sid: int, decision: str, *, session: Any = None,
                 actor: str = "Aaron") -> Dict[str, Any]:
    """The one place a suggestion is answered, from a text or the console."""
    store = manager.store
    if decision not in ("yes", "no"):
        return {"ok": False, "state": "", "error": f"#{sid} needs a yes or a no", "reply": ""}
    row = get(store, sid)
    if row is None:
        return {"ok": False, "state": "", "error": f"no suggestion #{sid}", "reply": ""}
    if row["status"] != "pending":
        said = {"accepted": "is already under way", "dismissed": "was dismissed",
                "expired": "lapsed unanswered", "failed": "never reached you"}.get(row["status"], row["status"])
        return {"ok": False, "state": row["status"], "error": f"#{sid} {said}",
                "reply": f"#{sid} {said}."}
    if decision == "no":
        if _set_status(store, sid, "dismissed"):
            try:
                store.set_state(sid, "rejected")
            except Exception:
                pass
        return {"ok": True, "state": "dismissed", "error": "", "reply": f"Dismissed #{sid}. I will not suggest it again."}
    req = store.get_request(sid)
    if req is None:
        return {"ok": False, "state": "", "error": f"no request #{sid}", "reply": ""}
    if not _set_status(store, sid, "accepted", request_id=sid):
        return {"ok": False, "state": "", "error": f"#{sid} was already answered", "reply": ""}
    from .scopes import BLATBOT_ADMIN_SCOPE, SCOPES
    scopes = [sc for sc in SCOPES if not sc.startswith("vault:")] + [BLATBOT_ADMIN_SCOPE]
    if session is not None:
        # The work answers him where he said yes.
        try:
            sender = str((getattr(session, "reply_meta", None) or {}).get("sender") or req.sender)
            _write(store, "UPDATE requests SET chat_id=?, mode=?, sender=? WHERE id=?",
                   (session.record_id, session.mode or req.mode, sender, sid))
        except Exception as exc:
            logger.debug("[suggest] could not move #%s to the reply thread: %s", sid, exc)
    task = store.create_task(_short(row["text"], 150), [(req.sender, "Aaron")])
    store.link_request_task(sid, int(task["id"]))
    store.set_scopes(sid, scopes)
    store.set_state(sid, "approved", status={"suggestion_id": sid, "why": row.get("why") or ""})
    fresh = store.get_request(sid)
    manager.task_note(fresh, "approved", f"{actor} said yes to the suggestion: {row['text']}", state="running")
    asyncio.create_task(manager.execute(fresh))
    return {"ok": True, "state": "approved", "error": "", "reply": ""}


# -- console -----------------------------------------------------------------

def console_view(store: Any, limit: int = 100) -> Dict[str, Any]:
    rows = []
    for r in recent(store, limit):
        try:
            refs = json.loads(r.get("source_refs") or "[]")
        except Exception:
            refs = []
        rows.append({"id": r["id"], "text": r["text"], "task": r["task"], "why": r.get("why") or "",
                     "refs": refs, "status": r["status"], "created_at": r["created_at"],
                     "decided_at": r.get("decided_at"), "request_id": r.get("request_id")})
    return {"enabled": enabled(), "times": [f"{h}:{m:02d}" for h, m in times()],
            "pending": sum(1 for r in rows if r["status"] == "pending"),
            "rows": rows, "passes": passes(store, 20)}
