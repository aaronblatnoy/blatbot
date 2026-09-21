"""SQLite persistence for threads, messages, and requests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
  chat_id TEXT PRIMARY KEY,
  state TEXT NOT NULL DEFAULT 'idle',
  mode TEXT,
  meta_json TEXT,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT NOT NULL,
  kind TEXT NOT NULL,            -- inbound | outbound | system
  mode TEXT,
  text TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT NOT NULL,
  sender TEXT,
  sender_name TEXT,
  mode TEXT,
  subject TEXT,
  original_message TEXT NOT NULL,
  summary TEXT NOT NULL,
  scopes_json TEXT NOT NULL,
  prompt TEXT NOT NULL,
  prompt_sha256 TEXT NOT NULL,
  state TEXT NOT NULL,           -- pending|approved|running|done|rejected|expired|failed
  revision INTEGER NOT NULL DEFAULT 0,
  status_json TEXT,
  raw_output TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS requests_state ON requests(state);
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT NOT NULL,             -- normalized counterpart: email | phone digits | lowercase name
  display TEXT NOT NULL,         -- how to show the counterpart
  title TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'open',  -- open | waiting_aaron | running | done | failed
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_key ON tasks(key, updated_at);
CREATE TABLE IF NOT EXISTS task_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  kind TEXT NOT NULL,            -- inbound | outbound | request | approved | rejected | done | failed | expired | note
  chat_id TEXT,
  request_id INTEGER,
  text TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS task_events_task ON task_events(task_id, id);
"""

TASK_REOPEN_AFTER = 7 * 24 * 3600   # a finished task older than this starts a fresh one
TASK_MEMORY_AGE = 21 * 24 * 3600    # finished tasks older than this are not shown


def task_key(raw: str) -> str:
    """Normalize a counterpart identifier so email, phone and name variants collide."""
    v = (raw or "").strip().lower()
    if "@" in v:
        return v
    digits = "".join(ch for ch in v if ch.isdigit())
    if len(digits) >= 10:
        return digits[-10:]
    return " ".join(v.split())


def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 48 * 3600:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Request:
    id: int
    chat_id: str
    sender: str
    sender_name: str
    mode: str
    subject: str
    original_message: str
    summary: str
    scopes: List[str]
    prompt: str
    prompt_sha256: str
    state: str
    revision: int
    status: Optional[Dict[str, Any]]
    raw_output: Optional[str]
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Request":
        return cls(
            id=r["id"], chat_id=r["chat_id"], sender=r["sender"] or "",
            sender_name=r["sender_name"] or "", mode=r["mode"] or "",
            subject=r["subject"] or "", original_message=r["original_message"],
            summary=r["summary"], scopes=json.loads(r["scopes_json"]),
            prompt=r["prompt"], prompt_sha256=r["prompt_sha256"], state=r["state"],
            revision=r["revision"],
            status=json.loads(r["status_json"]) if r["status_json"] else None,
            raw_output=r["raw_output"], created_at=r["created_at"], updated_at=r["updated_at"],
        )


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            cols = {r["name"] for r in self._db.execute("PRAGMA table_info(requests)")}
            if "task_id" not in cols:
                self._db.execute("ALTER TABLE requests ADD COLUMN task_id INTEGER")
            self._db.commit()

    # -- threads ---------------------------------------------------------
    def thread_state(self, chat_id: str) -> str:
        with self._lock:
            r = self._db.execute("SELECT state FROM threads WHERE chat_id=?", (chat_id,)).fetchone()
        return r["state"] if r else "idle"

    def set_thread(self, chat_id: str, state: str, mode: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO threads(chat_id,state,mode,meta_json,updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(chat_id) DO UPDATE SET state=excluded.state, "
                "mode=COALESCE(NULLIF(excluded.mode,''),threads.mode), "
                "meta_json=COALESCE(excluded.meta_json,threads.meta_json), updated_at=excluded.updated_at",
                (chat_id, state, mode, json.dumps(meta) if meta is not None else None, time.time()),
            )
            self._db.commit()

    def thread_route(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Last known channel + reply meta for a thread (to message them later)."""
        with self._lock:
            r = self._db.execute("SELECT mode, meta_json FROM threads WHERE chat_id=?", (chat_id,)).fetchone()
        if not r or not r["meta_json"]:
            return None
        return {"mode": r["mode"], "meta": json.loads(r["meta_json"])}

    # -- messages --------------------------------------------------------
    def add_message(self, chat_id: str, kind: str, text: str, mode: str = "") -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO messages(chat_id,kind,mode,text,created_at) VALUES(?,?,?,?,?)",
                (chat_id, kind, mode, text, time.time()),
            )
            self._db.commit()

    def history(self, chat_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT kind,mode,text,created_at FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- requests --------------------------------------------------------
    def create_request(self, *, chat_id: str, sender: str, sender_name: str, mode: str, subject: str,
                       original_message: str, summary: str, scopes: List[str], prompt: str,
                       state: str) -> Request:
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO requests(chat_id,sender,sender_name,mode,subject,original_message,summary,"
                "scopes_json,prompt,prompt_sha256,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (chat_id, sender, sender_name, mode, subject, original_message, summary,
                 json.dumps(scopes), prompt, sha256(prompt), state, now, now),
            )
            self._db.commit()
            rid = cur.lastrowid
        return self.get_request(rid)  # type: ignore[return-value]

    def get_request(self, rid: int) -> Optional[Request]:
        with self._lock:
            r = self._db.execute("SELECT * FROM requests WHERE id=?", (rid,)).fetchone()
        return Request.from_row(r) if r else None

    def pending(self) -> List[Request]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM requests WHERE state='pending' ORDER BY id").fetchall()
        return [Request.from_row(r) for r in rows]

    def pending_for_thread(self, chat_id: str) -> Optional[Request]:
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM requests WHERE chat_id=? AND state IN ('pending','approved','running') ORDER BY id DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
        return Request.from_row(r) if r else None

    def set_state(self, rid: int, state: str, *, status: Optional[Dict[str, Any]] = None,
                  raw_output: Optional[str] = None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE requests SET state=?, status_json=COALESCE(?,status_json), "
                "raw_output=COALESCE(?,raw_output), updated_at=? WHERE id=?",
                (state, json.dumps(status) if status is not None else None, raw_output, time.time(), rid),
            )
            self._db.commit()

    def revise_prompt(self, rid: int, prompt: str) -> Request:
        with self._lock:
            self._db.execute(
                "UPDATE requests SET prompt=?, prompt_sha256=?, revision=revision+1, state='pending', updated_at=? WHERE id=?",
                (prompt, sha256(prompt), time.time(), rid),
            )
            self._db.commit()
        return self.get_request(rid)  # type: ignore[return-value]

    # -- tasks -----------------------------------------------------------
    def task_for(self, raw_key: str, display: str, title: str = "") -> Dict[str, Any]:
        """Find the live task for a counterpart, or start one. Never raises."""
        key = task_key(raw_key)
        now = time.time()
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM tasks WHERE key=? ORDER BY updated_at DESC LIMIT 1", (key,)
            ).fetchone()
            if r and (r["state"] not in ("done", "failed") or now - r["updated_at"] < TASK_REOPEN_AFTER):
                if display and display != r["display"] and "@" not in r["display"]:
                    self._db.execute("UPDATE tasks SET display=? WHERE id=?", (display, r["id"]))
                    self._db.commit()
                    r = self._db.execute("SELECT * FROM tasks WHERE id=?", (r["id"],)).fetchone()
                return dict(r)
            cur = self._db.execute(
                "INSERT INTO tasks(key,display,title,state,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (key, display or raw_key, title or f"Thread with {display or raw_key}", "open", now, now),
            )
            self._db.commit()
            r = self._db.execute("SELECT * FROM tasks WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(r)

    def task_event(self, task_id: int, kind: str, text: str, *, chat_id: str = "",
                   request_id: Optional[int] = None, state: Optional[str] = None,
                   title: Optional[str] = None) -> None:
        text = (text or "").strip()
        if len(text) > 700:
            text = text[:700] + "..."
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO task_events(task_id,kind,chat_id,request_id,text,created_at) VALUES(?,?,?,?,?,?)",
                (task_id, kind, chat_id, request_id, text, now),
            )
            sets, args = ["updated_at=?"], [now]
            if state:
                sets.append("state=?"); args.append(state)
            if title:
                sets.append("title=?"); args.append(title)
            args.append(task_id)
            self._db.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", args)
            self._db.commit()

    def link_request_task(self, rid: int, task_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE requests SET task_id=? WHERE id=?", (task_id, rid))
            self._db.commit()

    def task_id_for_request(self, rid: int) -> Optional[int]:
        with self._lock:
            r = self._db.execute("SELECT task_id FROM requests WHERE id=?", (rid,)).fetchone()
        return int(r["task_id"]) if r and r["task_id"] else None

    def task_ids_for_chat(self, chat_id: str, limit: int = 3) -> List[int]:
        """Tasks this thread has touched, most recent first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT task_id, MAX(id) AS last FROM task_events WHERE chat_id=? GROUP BY task_id ORDER BY last DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [int(r["task_id"]) for r in rows]

    def recent_tasks(self, limit: int = 12) -> List[Dict[str, Any]]:
        """Live tasks plus recently finished ones, most recently touched first."""
        cutoff = time.time() - TASK_MEMORY_AGE
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM tasks WHERE state NOT IN ('done','failed') OR updated_at>? ORDER BY updated_at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def task_with_events(self, task_id: int, limit: int = 8) -> Optional[Dict[str, Any]]:
        with self._lock:
            t = self._db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not t:
                return None
            ev = self._db.execute(
                "SELECT kind,chat_id,request_id,text,created_at FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT ?",
                (task_id, limit),
            ).fetchall()
        d = dict(t)
        d["events"] = [dict(e) for e in reversed(ev)]
        return d

    def task_memory_for_task(self, task_id: int, now: Optional[float] = None) -> str:
        """One task's ledger, for the executor's context."""
        t = self.task_with_events(task_id, limit=12)
        if not t:
            return ""
        now = now or time.time()
        lines = [f"Task T{t['id']} | {t['display']} | {t['title']} | state: {t['state']}"]
        for e in t["events"]:
            rid = f" (#{e['request_id']})" if e.get("request_id") else ""
            lines.append(f"  - {_age(now - e['created_at'])} {e['kind']}{rid}: {e['text']}")
        return "\n".join(lines)

    def task_memory(self, chat_id: str, *, is_approver: bool, now: Optional[float] = None) -> str:
        """Plain-text ledger the router reads before anything else.

        For a counterpart's thread: their task(s). For Aaron: every live task plus
        the ones his thread touched, so a follow-up hours later lands on the record."""
        now = now or time.time()
        ids: List[int] = self.task_ids_for_chat(chat_id)
        if is_approver:
            for t in self.recent_tasks():
                if t["id"] not in ids:
                    ids.append(t["id"])
        blocks: List[str] = []
        for tid in ids[:12]:
            t = self.task_with_events(tid)
            if not t:
                continue
            head = f"Task T{t['id']} | {t['display']} | {t['title']} | state: {t['state']}"
            lines = [head]
            for e in t["events"]:
                age = now - e["created_at"]
                when = _age(age)
                rid = f" (#{e['request_id']})" if e.get("request_id") else ""
                lines.append(f"  - {when} {e['kind']}{rid}: {e['text']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "(no tasks on record)"

    def expire_older_than(self, seconds: float) -> List[Request]:
        cutoff = time.time() - seconds
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM requests WHERE state='pending' AND created_at<?", (cutoff,)
            ).fetchall()
            self._db.execute(
                "UPDATE requests SET state='expired', updated_at=? WHERE state='pending' AND created_at<?",
                (time.time(), cutoff),
            )
            self._db.commit()
        return [Request.from_row(r) for r in rows]
