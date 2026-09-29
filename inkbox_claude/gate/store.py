"""SQLite persistence for threads, messages, and requests."""

from __future__ import annotations

import hashlib
import json
import logging
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
CREATE TABLE IF NOT EXISTS mail_ids (
  message_id TEXT PRIMARY KEY,   -- RFC Message-ID of a mail we have seen
  root TEXT NOT NULL             -- Message-ID of the first mail in its reply chain
);
CREATE TABLE IF NOT EXISTS thread_links (
  thread_key TEXT NOT NULL,      -- e.g. email:<provider thread id>
  chat_id TEXT NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (thread_key, chat_id)
);
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
  key TEXT NOT NULL DEFAULT '',  -- legacy (pre-participants) counterpart key; unused for new tasks
  display TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL,           -- what the task is
  state TEXT NOT NULL DEFAULT 'open',  -- open | waiting_aaron | running | done | failed | closed
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_key ON tasks(key, updated_at);
CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state, updated_at);
CREATE TABLE IF NOT EXISTS task_participants (
  task_id INTEGER NOT NULL,
  key TEXT NOT NULL,             -- one way to reach the person: email | phone digits | lowercase name
  display TEXT NOT NULL,
  person_id TEXT NOT NULL DEFAULT '',  -- contact id when known; groups several keys as one person
  PRIMARY KEY (task_id, key)
);
CREATE INDEX IF NOT EXISTS task_participants_key ON task_participants(key);
CREATE TABLE IF NOT EXISTS people (
  person_id TEXT NOT NULL,       -- contact id
  key TEXT NOT NULL,             -- every known email / phone / name of that person
  display TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL,
  PRIMARY KEY (person_id, key)
);
CREATE INDEX IF NOT EXISTS people_key ON people(key);
CREATE VIRTUAL TABLE IF NOT EXISTS task_fts USING fts5(
  task_id UNINDEXED, title, summary, people, events, requests, dates,
  tokenize='porter unicode61'
);
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

TASK_MEMORY_AGE = 21 * 24 * 3600    # finished tasks older than this are not shown
OPEN_STATES = ("open", "waiting_aaron", "running")


@dataclass
class Person:
    """One human, however they reached us: a contact id plus every email, phone and
    name we know for them. Built from the Inkbox contact record on each message."""
    person_id: str          # contact id, or "" when the sender is unknown
    display: str
    keys: List[str]         # normalized, deduplicated, primary first

    @classmethod
    def from_contact(cls, contact: Optional[Dict[str, Any]], sender: str = "",
                     name: str = "") -> "Person":
        contact = contact if isinstance(contact, dict) else {}
        raw: List[str] = [sender]
        raw += [str(x) for x in (contact.get("emails") or [])]
        raw += [str(x) for x in (contact.get("phones") or [])]
        display = name or str(contact.get("name") or "").strip() or sender
        if display and "@" not in display and not any(ch.isdigit() for ch in display):
            raw.append(display)  # a real name is a key too
        keys: List[str] = []
        for r in raw:
            k = task_key(r)
            if k and k not in keys:
                keys.append(k)
        return cls(person_id=str(contact.get("id") or ""), display=display, keys=keys)


logger = logging.getLogger(__name__)


class TaskRequired(RuntimeError):
    """Raised when a request would be created without a task. The gateway
    enforces that every request is written to a task; this is the backstop."""


def task_key(raw: str) -> str:
    """Normalize a counterpart identifier so email, phone and name variants collide."""
    v = (raw or "").strip().lower()
    if "@" in v:
        return v
    digits = "".join(ch for ch in v if ch.isdigit())
    if len(digits) >= 10:
        return digits[-10:]
    return " ".join(v.split())


_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def extract_dates(text: str, anchor: Optional[float] = None) -> List[str]:
    """Pull calendar dates out of free text as ISO strings (YYYY-MM-DD), in text order.

    Understands 9/22, 9/22/26, 2026-09-22, Sep 22, September 22 2026, Tue 9/22.
    Two-digit and missing years resolve to the anchor's year."""
    import re
    from datetime import datetime
    year = datetime.fromtimestamp(anchor or time.time()).year
    found: List[Any] = []  # (position, iso)
    t = text or ""
    for m in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t):
        found.append((m.start(), f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"))
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t):
        mo, d, y = int(m[1]), int(m[2]), m[3]
        if 1 <= mo <= 12 and 1 <= d <= 31:
            yy = int(y) if y else year
            if yy < 100:
                yy += 2000
            found.append((m.start(), f"{yy:04d}-{mo:02d}-{d:02d}"))
    for m in re.finditer(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))?\b", t, re.I):
        mo = _MONTHS[m[1].lower()[:3]]
        d = int(m[2])
        yy = int(m[3]) if m[3] else year
        if 1 <= d <= 31:
            found.append((m.start(), f"{yy:04d}-{mo:02d}-{d:02d}"))
    out: List[str] = []
    for _, iso in sorted(found):
        if iso not in out:
            out.append(iso)
    return out


def _fts_query(text: str) -> str:
    """Turn free text into an FTS5 query: quoted phrases kept, OR honoured, every
    other word required, with prefix matching so 'cater' finds 'caterer'."""
    import re
    parts: List[str] = []
    for tok in re.findall(r'"[^"]+"|\S+', text or ""):
        if tok.startswith('"') and tok.endswith('"') and len(tok) > 2:
            parts.append(tok)
        elif tok.upper() == "OR":
            parts.append("OR")
        else:
            w = re.sub(r"[^\w@.'-]", "", tok)
            if len(w) > 1:
                parts.append('"' + w.replace('"', '') + '"*')
    return " ".join(parts) or '""'


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
    inbound_id: Optional[int] = None

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
            inbound_id=(int(r["inbound_id"]) if "inbound_id" in r.keys() and r["inbound_id"] else None),
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
            if "inbound_id" not in cols:
                self._db.execute("ALTER TABLE requests ADD COLUMN inbound_id INTEGER")
            mcols = {r["name"] for r in self._db.execute("PRAGMA table_info(messages)")}
            if "reply_to" not in mcols:
                # Every outbound is linked to the inbound it answers and typed ack|answer,
                # so "one answer per message" can be enforced, not just intended.
                self._db.execute("ALTER TABLE messages ADD COLUMN reply_to INTEGER")
                self._db.execute("ALTER TABLE messages ADD COLUMN role TEXT")
                self._db.execute("CREATE INDEX IF NOT EXISTS messages_reply_to ON messages(reply_to)")
            tcols = {r["name"] for r in self._db.execute("PRAGMA table_info(tasks)")}
            if "summary" not in tcols:
                self._db.execute("ALTER TABLE tasks ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
            pcols = {r["name"] for r in self._db.execute("PRAGMA table_info(task_participants)")}
            if "person_id" not in pcols:
                self._db.execute("ALTER TABLE task_participants ADD COLUMN person_id TEXT NOT NULL DEFAULT ''")
            self._db.execute("CREATE INDEX IF NOT EXISTS task_participants_person ON task_participants(person_id)")
            self._migrate_person_keyed_tasks()
            self._db.commit()
            n_tasks = self._db.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
            n_fts = self._db.execute("SELECT COUNT(*) AS n FROM task_fts").fetchone()["n"]
        if n_fts < n_tasks:
            self.reindex_all()

    def _migrate_person_keyed_tasks(self) -> None:
        """One-time: tasks used to be keyed to a person. Turn each key into a
        participant row, and give tasks that were only a conversation a real title."""
        rows = self._db.execute(
            "SELECT t.id, t.key, t.display, t.title FROM tasks t "
            "WHERE t.key<>'' AND NOT EXISTS (SELECT 1 FROM task_participants p WHERE p.task_id=t.id)"
        ).fetchall()
        for r in rows:
            key = str(r["key"])
            if key.startswith("aaron:"):
                continue  # an errand with no counterpart: no participant
            self._db.execute(
                "INSERT OR IGNORE INTO task_participants(task_id,key,display) VALUES(?,?,?)",
                (r["id"], key, r["display"] or key),
            )
        # Every request belongs to a task. Attach any orphan to a task titled from its summary.
        now = time.time()
        for r in self._db.execute("SELECT id, summary, state, created_at FROM requests WHERE task_id IS NULL").fetchall():
            cur = self._db.execute(
                "INSERT INTO tasks(key,display,title,state,created_at,updated_at) VALUES('','',?,?,?,?)",
                (r["summary"] or f"Request #{r['id']}",
                 "done" if r["state"] == "done" else ("failed" if r["state"] in ("failed", "expired", "rejected") else "open"),
                 r["created_at"], now),
            )
            self._db.execute("UPDATE requests SET task_id=? WHERE id=?", (cur.lastrowid, r["id"]))
            self._db.execute(
                "INSERT INTO task_events(task_id,kind,chat_id,request_id,text,created_at) VALUES(?,?,?,?,?,?)",
                (cur.lastrowid, "note", "", r["id"], "Attached during the task-model migration.", now),
            )
        # Conversation-only "tasks" (no request ever attached) are closed, not deleted.
        self._db.execute(
            "UPDATE tasks SET state='closed' WHERE title LIKE 'Thread with %' "
            "AND state NOT IN ('done','failed','closed') "
            "AND NOT EXISTS (SELECT 1 FROM task_events e WHERE e.task_id=tasks.id AND e.kind='request')"
        )

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
    def add_message(self, chat_id: str, kind: str, text: str, mode: str = "",
                    reply_to: Optional[int] = None, role: Optional[str] = None) -> int:
        """Store a message and return its id. Outbound messages carry `reply_to`
        (the inbound they respond to) and `role` ('ack' or 'answer')."""
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO messages(chat_id,kind,mode,text,created_at,reply_to,role) VALUES(?,?,?,?,?,?,?)",
                (chat_id, kind, mode, text, time.time(), reply_to, role),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def responses_to(self, inbound_id: int) -> List[str]:
        """Roles of the outbound messages already sent for one inbound message."""
        if not inbound_id:
            return []
        with self._lock:
            rows = self._db.execute("SELECT role FROM messages WHERE reply_to=? AND kind='outbound' ORDER BY id",
                                    (inbound_id,)).fetchall()
        return [r["role"] or "answer" for r in rows]

    def history(self, chat_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT kind,mode,text,created_at FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- threads that span conversations (an email thread with several senders) --
    def thread_root(self, *, message_id: str = "", in_reply_to: str = "", references: Optional[List[str]] = None) -> str:
        """Resolve an email to the root of its reply chain and remember it. Returns ""
        when the mail carries no RFC ids at all."""
        refs = [r for r in (references or []) if r]
        if not (message_id or in_reply_to or refs):
            return ""
        with self._lock:
            root = ""
            for cand in ([in_reply_to] if in_reply_to else []) + list(reversed(refs)):
                r = self._db.execute("SELECT root FROM mail_ids WHERE message_id=?", (cand,)).fetchone()
                if r:
                    root = r["root"]; break
            if not root:
                root = refs[0] if refs else (in_reply_to or message_id)
            for mid in [message_id, in_reply_to] + refs:
                if mid:
                    self._db.execute("INSERT OR IGNORE INTO mail_ids(message_id, root) VALUES(?,?)", (mid, root))
            self._db.commit()
        return root

    def link_thread(self, thread_key: str, chat_id: str) -> None:
        if not thread_key or not chat_id:
            return
        with self._lock:
            self._db.execute(
                "INSERT INTO thread_links(thread_key,chat_id,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(thread_key,chat_id) DO UPDATE SET updated_at=excluded.updated_at",
                (thread_key, chat_id, time.time()),
            )
            self._db.commit()

    def thread_chats(self, thread_key: str) -> List[str]:
        """Every conversation that has taken part in this thread."""
        if not thread_key:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT chat_id FROM thread_links WHERE thread_key=? ORDER BY updated_at", (thread_key,)
            ).fetchall()
        return [r["chat_id"] for r in rows]

    def thread_history(self, thread_key: str, exclude_chat: str = "", limit: int = 12) -> List[Dict[str, Any]]:
        """Recent messages from the OTHER participants of a thread, oldest first, each
        labelled with who it was from, so a reply can be read in context."""
        chats = [c for c in self.thread_chats(thread_key) if c != exclude_chat]
        if not chats:
            return []
        marks = ",".join("?" * len(chats))
        with self._lock:
            rows = self._db.execute(
                f"SELECT chat_id,kind,mode,text,created_at FROM messages WHERE chat_id IN ({marks}) "
                f"AND kind IN ('inbound','outbound') ORDER BY id DESC LIMIT ?", (*chats, limit),
            ).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            d["kind"] = f"{r['kind']} (other participant {r['chat_id'][:8]})"
            out.append(d)
        return out

    def task_ids_for_thread(self, thread_key: str, limit: int = 5) -> List[int]:
        chats = self.thread_chats(thread_key)
        if not chats:
            return []
        marks = ",".join("?" * len(chats))
        with self._lock:
            rows = self._db.execute(
                f"SELECT task_id, MAX(id) AS last FROM task_events WHERE chat_id IN ({marks}) "
                f"GROUP BY task_id ORDER BY last DESC LIMIT ?", (*chats, limit),
            ).fetchall()
        return [int(r["task_id"]) for r in rows]

    # -- requests --------------------------------------------------------
    def create_request(self, *, chat_id: str, sender: str, sender_name: str, mode: str, subject: str,
                       original_message: str, summary: str, scopes: List[str], prompt: str,
                       state: str, task_id: int, inbound_id: Optional[int] = None) -> Request:
        """Create a request. ``task_id`` is mandatory: a request is always part of a task."""
        if not task_id:
            raise TaskRequired("a request must belong to a task")
        now = time.time()
        with self._lock:
            if not self._db.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
                raise TaskRequired(f"task T{task_id} does not exist")
            cur = self._db.execute(
                "INSERT INTO requests(chat_id,sender,sender_name,mode,subject,original_message,summary,"
                "scopes_json,prompt,prompt_sha256,state,created_at,updated_at,task_id,inbound_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (chat_id, sender, sender_name, mode, subject, original_message, summary,
                 json.dumps(scopes), prompt, sha256(prompt), state, now, now, task_id, inbound_id),
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

    RUN_STALE_S = 15 * 60  # longer than any executor timeout: a run this old died without reporting

    def running_for_thread(self, chat_id: str) -> Optional[Request]:
        """The request currently executing on this thread, if any. A run older than
        RUN_STALE_S is not running any more whatever its row says (a restart mid-run
        leaves 'running' behind); it is marked failed here so it never blocks a thread."""
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM requests WHERE chat_id=? AND state IN ('approved','running') ORDER BY id DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
            if r and time.time() - float(r["updated_at"]) > self.RUN_STALE_S:
                self._db.execute("UPDATE requests SET state='failed', updated_at=? WHERE id=?", (time.time(), r["id"]))
                self._db.commit()
                logger.warning("[store] request #%s had been 'running' since %s; marked failed (stale)", r["id"], r["updated_at"])
                return None
        return Request.from_row(r) if r else None

    def inbound_since(self, chat_id: str, since: float) -> List[Dict[str, Any]]:
        """Inbound messages ({id, text}) on this thread after a point in time, oldest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, text FROM messages WHERE chat_id=? AND kind='inbound' AND created_at>? ORDER BY id",
                (chat_id, since),
            ).fetchall()
        return [{"id": int(r["id"]), "text": r["text"]} for r in rows]

    def set_state(self, rid: int, state: str, *, status: Optional[Dict[str, Any]] = None,
                  raw_output: Optional[str] = None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE requests SET state=?, status_json=COALESCE(?,status_json), "
                "raw_output=COALESCE(?,raw_output), updated_at=? WHERE id=?",
                (state, json.dumps(status) if status is not None else None, raw_output, time.time(), rid),
            )
            self._db.commit()
            tid = self._db.execute("SELECT task_id FROM requests WHERE id=?", (rid,)).fetchone()
        if tid and tid["task_id"]:
            self.reindex_task(int(tid["task_id"]))

    def revise_prompt(self, rid: int, prompt: str) -> Request:
        with self._lock:
            self._db.execute(
                "UPDATE requests SET prompt=?, prompt_sha256=?, revision=revision+1, state='pending', updated_at=? WHERE id=?",
                (prompt, sha256(prompt), time.time(), rid),
            )
            self._db.commit()
        return self.get_request(rid)  # type: ignore[return-value]

    # -- tasks -----------------------------------------------------------
    def create_task(self, title: str, participants: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Start a task. ``participants`` is a list of (raw_key, display) pairs or bare strings."""
        now = time.time()
        title = " ".join((title or "").split())[:160] or "Untitled task"
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO tasks(key,display,title,state,created_at,updated_at) VALUES('','',?,?,?,?)",
                (title, "open", now, now),
            )
            tid = int(cur.lastrowid)
            self._db.commit()
        for p in participants or []:
            raw, display = (p if isinstance(p, (tuple, list)) else (p, p))
            self.add_participant(tid, str(raw), str(display or raw))
        self.reindex_task(tid)
        return self.get_task(tid)  # type: ignore[return-value]

    def remember_person(self, person: "Person") -> None:
        """Record every key we know for a contact, so later lookups by any of them match."""
        if not person.person_id:
            return
        now = time.time()
        with self._lock:
            for k in person.keys:
                self._db.execute(
                    "INSERT INTO people(person_id,key,display,updated_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(person_id,key) DO UPDATE SET display=CASE WHEN excluded.display<>'' "
                    "THEN excluded.display ELSE people.display END, updated_at=excluded.updated_at",
                    (person.person_id, k, person.display, now),
                )
            self._db.commit()

    def person_keys(self, raw_key: str) -> List[str]:
        """All keys that belong to the same person as ``raw_key`` (itself included)."""
        key = task_key(raw_key)
        if not key:
            return []
        with self._lock:
            pid = self._db.execute("SELECT person_id FROM people WHERE key=?", (key,)).fetchone()
            if not pid:
                pid = self._db.execute(
                    "SELECT person_id FROM task_participants WHERE key=? AND person_id<>''", (key,)
                ).fetchone()
            if not pid:
                return [key]
            rows = self._db.execute(
                "SELECT key FROM people WHERE person_id=? UNION SELECT key FROM task_participants WHERE person_id=?",
                (pid["person_id"], pid["person_id"]),
            ).fetchall()
        keys = [r["key"] for r in rows]
        return keys if key in keys else [key] + keys

    def add_participant(self, task_id: int, who: Any, display: str = "") -> None:
        """Attach a person to a task. ``who`` is a Person, or a bare email/phone/name."""
        if isinstance(who, Person):
            person = who
        else:
            person = Person(person_id="", display=display or str(who), keys=[task_key(str(who))])
            # A bare key may already be known as part of a contact.
            with self._lock:
                r = self._db.execute("SELECT person_id, display FROM people WHERE key=?", (person.keys[0],)).fetchone()
            if r:
                person = Person(person_id=r["person_id"], display=display or r["display"] or str(who),
                                keys=self.person_keys(person.keys[0]))
        self.remember_person(person)
        with self._lock:
            for k in person.keys:
                if not k:
                    continue
                self._db.execute(
                    "INSERT INTO task_participants(task_id,key,display,person_id) VALUES(?,?,?,?) "
                    "ON CONFLICT(task_id,key) DO UPDATE SET "
                    "display=CASE WHEN excluded.display<>'' AND instr(excluded.display,'@')=0 THEN excluded.display ELSE task_participants.display END, "
                    "person_id=CASE WHEN excluded.person_id<>'' THEN excluded.person_id ELSE task_participants.person_id END",
                    (task_id, k, person.display or k, person.person_id),
                )
            self._db.commit()
        self.reindex_task(task_id)

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            t = self._db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not t:
                return None
            ps = self._db.execute(
                "SELECT key, display, person_id FROM task_participants WHERE task_id=? ORDER BY rowid", (task_id,)
            ).fetchall()
        d = dict(t)
        # One entry per person: keys sharing a person_id are grouped.
        people: Dict[str, Dict[str, Any]] = {}
        for p in ps:
            gid = p["person_id"] or f"key:{p['key']}"
            entry = people.setdefault(gid, {"key": p["key"], "display": p["display"], "person_id": p["person_id"], "keys": []})
            entry["keys"].append(p["key"])
            if p["display"] and "@" not in p["display"] and not any(ch.isdigit() for ch in p["display"]):
                entry["display"] = p["display"]
        d["participants"] = list(people.values())
        return d

    def reindex_task(self, task_id: int) -> None:
        """Rebuild the search document for one task from everything on it."""
        with self._lock:
            t = self._db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not t:
                self._db.execute("DELETE FROM task_fts WHERE task_id=?", (task_id,))
                self._db.commit()
                return
            ps = self._db.execute("SELECT key, display FROM task_participants WHERE task_id=?", (task_id,)).fetchall()
            ev = self._db.execute("SELECT kind, text, created_at FROM task_events WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
            rq = self._db.execute("SELECT summary, prompt, state, status_json, raw_output FROM requests WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
            people = " ".join(f"{p['display']} {p['key']}" for p in ps)
            events = " ".join(f"{e['kind']}: {e['text']}" for e in ev)
            reqs: List[str] = []
            for r in rq:
                res = ""
                if r["status_json"]:
                    try:
                        st = json.loads(r["status_json"]); res = str(st.get("summary") or st.get("error") or "")
                    except Exception:
                        res = ""
                reqs.append(f"{r['state']}: {r['summary']} | {r['prompt']} | {res}")
            requests_text = " ".join(reqs)
            dates: List[str] = []
            for txt, when in [(t["title"], t["created_at"]), (t["summary"], t["updated_at"])] + \
                             [(e["text"], e["created_at"]) for e in ev] + [(r["prompt"], t["created_at"]) for r in rq]:
                for d in extract_dates(txt or "", when):
                    if d not in dates:
                        dates.append(d)
            self._db.execute("DELETE FROM task_fts WHERE task_id=?", (task_id,))
            self._db.execute(
                "INSERT INTO task_fts(task_id,title,summary,people,events,requests,dates) VALUES(?,?,?,?,?,?,?)",
                (task_id, t["title"] or "", t["summary"] or "", people, events, requests_text, " ".join(dates)),
            )
            self._db.commit()

    def reindex_all(self) -> int:
        with self._lock:
            ids = [int(r["id"]) for r in self._db.execute("SELECT id FROM tasks").fetchall()]
        for tid in ids:
            self.reindex_task(tid)
        return len(ids)

    def set_task_summary(self, task_id: int, summary: str) -> None:
        """The router's own plain-language description of where the task stands."""
        summary = " ".join((summary or "").split())[:400]
        if not summary:
            return
        with self._lock:
            self._db.execute("UPDATE tasks SET summary=? WHERE id=?", (summary, task_id))
            self._db.commit()
        self.reindex_task(task_id)

    def set_task_state(self, task_id: int, state: str) -> None:
        with self._lock:
            self._db.execute("UPDATE tasks SET state=?, updated_at=? WHERE id=?", (state, time.time(), task_id))
            self._db.commit()
        self.reindex_task(task_id)

    def tasks_for_person(self, raw_key: str, *, open_only: bool = True, limit: int = 8) -> List[Dict[str, Any]]:
        """Tasks this person is a participant of, most recently touched first."""
        keys = self.person_keys(raw_key)
        if not keys:
            return []
        cond = "AND t.state IN ('open','waiting_aaron','running')" if open_only else ""
        marks = ",".join("?" * len(keys))
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT t.id, t.updated_at FROM tasks t JOIN task_participants p ON p.task_id=t.id "
                f"WHERE p.key IN ({marks}) {cond} ORDER BY t.updated_at DESC LIMIT ?", (*keys, limit),
            ).fetchall()
        return [self.get_task(int(r["id"])) for r in rows]  # type: ignore[misc]

    def open_tasks(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM tasks WHERE state IN ('open','waiting_aaron','running') "
                "ORDER BY updated_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self.get_task(int(r["id"])) for r in rows]  # type: ignore[misc]

    def recent_tasks(self, limit: int = 12) -> List[Dict[str, Any]]:
        """Live tasks plus recently finished ones, most recently touched first."""
        cutoff = time.time() - TASK_MEMORY_AGE
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM tasks WHERE (state IN ('open','waiting_aaron','running') OR updated_at>?) "
                "AND state<>'closed' ORDER BY updated_at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [self.get_task(int(r["id"])) for r in rows]  # type: ignore[misc]

    def query_tasks(self, *, text: str = "", participant: str = "", states: Optional[List[str]] = None,
                    touched_within_days: Optional[float] = None, created_within_days: Optional[float] = None,
                    has_participants: Optional[bool] = None, chat_id: str = "", ids: Optional[List[int]] = None,
                    date_from: Optional[str] = None, date_to: Optional[str] = None,
                    any_of: Optional[List[Dict[str, Any]]] = None, limit: int = 10) -> Dict[str, Any]:
        """Filter tasks by any combination of criteria and rank the matches.

        text: full-text search (stemmed, prefix-matched) over EVERYTHING on a task:
              title, summary, participants, every event, every request's summary,
              prompt and result. Supports quoted phrases and OR.
        participant: a person by email, phone or name; expands to all their contacts.
        states: e.g. ["open","waiting_aaron"]; "live" means all non-finished states.
        touched_within_days / created_within_days: recency windows.
        has_participants: True = tasks about someone; False = tasks with nobody.
        chat_id: tasks this conversation has logged events on.
        date_from / date_to: ISO dates; matches tasks that mention a calendar date
              in that range anywhere (meeting dates, deadlines).
        any_of: a list of sub-filters (same fields); a task matches if it matches ANY.
        Returns {"tasks": [...], "total": n} where total counts all matches before limit."""
        if any_of:
            seen: Dict[int, Dict[str, Any]] = {}
            total = 0
            for sub in any_of:
                r = self.query_tasks(**{k: v for k, v in sub.items() if k != "any_of"}, limit=50)
                for t in r["tasks"]:
                    seen.setdefault(int(t["id"]), t)
            ordered = sorted(seen.values(), key=lambda t: (0 if t["state"] in OPEN_STATES else 1, -t["updated_at"]))
            return {"tasks": ordered[:max(1, min(int(limit), 50))], "total": len(ordered)}

        where: List[str] = []
        args: List[Any] = []
        now = time.time()
        if states:
            expanded: List[str] = []
            for st in states:
                expanded += list(OPEN_STATES) if st == "live" else [st]
            where.append(f"t.state IN ({','.join('?' * len(expanded))})"); args += expanded
        else:
            where.append("t.state<>'closed'")
        if touched_within_days is not None:
            where.append("t.updated_at>=?"); args.append(now - float(touched_within_days) * 86400)
        if created_within_days is not None:
            where.append("t.created_at>=?"); args.append(now - float(created_within_days) * 86400)
        if has_participants is True:
            where.append("EXISTS (SELECT 1 FROM task_participants p WHERE p.task_id=t.id)")
        elif has_participants is False:
            where.append("NOT EXISTS (SELECT 1 FROM task_participants p WHERE p.task_id=t.id)")
        if participant:
            keys = self.person_keys(participant)
            if keys:
                where.append(f"EXISTS (SELECT 1 FROM task_participants p WHERE p.task_id=t.id AND "
                             f"(p.key IN ({','.join('?' * len(keys))}) OR lower(p.display)=?))")
                args += keys + [participant.strip().lower()]
            else:
                where.append("0")
        if chat_id:
            where.append("EXISTS (SELECT 1 FROM task_events e WHERE e.task_id=t.id AND e.chat_id=?)"); args.append(chat_id)
        if ids:
            where.append(f"t.id IN ({','.join('?' * len(ids))})"); args += [int(i) for i in ids]
        if date_from or date_to:
            # the dates column holds ISO dates; a task matches if any date falls in range
            lo, hi = (date_from or "0000-01-01"), (date_to or "9999-12-31")
            where.append("EXISTS (SELECT 1 FROM task_fts f WHERE f.task_id=t.id AND f.dates<>'' AND EXISTS ("
                         "SELECT 1 FROM json_each('[\"' || replace(f.dates,' ','\",\"') || '\"]') d "
                         "WHERE d.value BETWEEN ? AND ?))")
            args += [lo, hi]
        fts_join, rank_col = "", "0 AS rank"
        if (text or "").strip():
            fts_join = "JOIN task_fts f ON f.task_id=t.id"
            where.append("task_fts MATCH ?")
            args.append(_fts_query(text))
            rank_col = "bm25(task_fts, 0.0, 10.0, 6.0, 4.0, 2.0, 2.0, 1.0) AS rank"
        sql_where = " AND ".join(where) if where else "1"
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) AS n FROM tasks t {fts_join} WHERE {sql_where}", args).fetchone()["n"]
            rows = self._db.execute(
                f"SELECT t.id, t.updated_at, t.state, {rank_col} FROM tasks t {fts_join} WHERE {sql_where} "
                f"ORDER BY rank, CASE WHEN t.state IN ('open','waiting_aaron','running') THEN 0 ELSE 1 END, t.updated_at DESC LIMIT ?",
                (*args, max(1, min(int(limit), 50))),
            ).fetchall()
        return {"tasks": [self.task_with_events(int(r["id"]), limit=6) for r in rows], "total": int(total)}

    def render_tasks(self, tasks: List[Dict[str, Any]], now: Optional[float] = None) -> str:
        now = now or time.time()
        blocks: List[str] = []
        for t in tasks:
            if not t:
                continue
            lines = [self._task_head(t)]
            for e in t.get("events") or []:
                rid = f" (#{e['request_id']})" if e.get("request_id") else ""
                lines.append(f"  - {_age(now - e['created_at'])} {e['kind']}{rid}: {e['text']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def task_for(self, raw_key: str, display: str, title: str = "") -> Dict[str, Any]:
        """Compatibility: the most recent open task this person is on, or a new one."""
        found = self.tasks_for_person(raw_key, open_only=True, limit=1)
        if found:
            return found[0]
        return self.create_task(title or f"Thread with {display or raw_key}", [(raw_key, display or raw_key)])

    def is_participant(self, task_id: int, raw_key: str) -> bool:
        keys = self.person_keys(raw_key)
        if not keys:
            return False
        marks = ",".join("?" * len(keys))
        with self._lock:
            r = self._db.execute(
                f"SELECT 1 FROM task_participants WHERE task_id=? AND key IN ({marks}) LIMIT 1", (task_id, *keys)
            ).fetchone()
        return bool(r)

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
        self.reindex_task(task_id)

    def link_request_task(self, rid: int, task_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE requests SET task_id=? WHERE id=?", (task_id, rid))
            self._db.commit()

    def task_id_for_request(self, rid: int) -> Optional[int]:
        with self._lock:
            r = self._db.execute("SELECT task_id FROM requests WHERE id=?", (rid,)).fetchone()
        return int(r["task_id"]) if r and r["task_id"] else None

    def scopes_for_task(self, task_id: int) -> List[str]:
        """Every scope granted to any earlier request on this task, most recent first.
        A follow-up ("try again", "yes do that") inherits these."""
        with self._lock:
            rows = self._db.execute("SELECT scopes_json FROM requests WHERE task_id=? ORDER BY id DESC", (task_id,)).fetchall()
        out: List[str] = []
        for r in rows:
            for sc in json.loads(r["scopes_json"] or "[]"):
                if sc not in out:
                    out.append(sc)
        return out

    def task_ids_for_chat(self, chat_id: str, limit: int = 3) -> List[int]:
        """Tasks this thread has touched, most recent first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT task_id, MAX(id) AS last FROM task_events WHERE chat_id=? GROUP BY task_id ORDER BY last DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [int(r["task_id"]) for r in rows]

    def task_with_events(self, task_id: int, limit: int = 8) -> Optional[Dict[str, Any]]:
        d = self.get_task(task_id)
        if not d:
            return None
        with self._lock:
            ev = self._db.execute(
                "SELECT kind,chat_id,request_id,text,created_at FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT ?",
                (task_id, limit),
            ).fetchall()
        d["events"] = [dict(e) for e in reversed(ev)]
        return d

    @staticmethod
    def _task_head(t: Dict[str, Any]) -> str:
        who = ", ".join(p["display"] for p in t.get("participants") or []) or "no one in particular"
        head = f"Task T{t['id']} | {t['title']} | with: {who} | state: {t['state']}"
        if t.get("summary"):
            head += f"\n  Where it stands: {t['summary']}"
        return head

    def task_memory_for_task(self, task_id: int, now: Optional[float] = None) -> str:
        """One task's ledger, for the executor's context."""
        t = self.task_with_events(task_id, limit=12)
        if not t:
            return ""
        now = now or time.time()
        lines = [self._task_head(t)]
        for e in t["events"]:
            rid = f" (#{e['request_id']})" if e.get("request_id") else ""
            lines.append(f"  - {_age(now - e['created_at'])} {e['kind']}{rid}: {e['text']}")
        return "\n".join(lines)

    def task_memory(self, chat_id: str, *, is_approver: bool, now: Optional[float] = None,
                    person: str = "") -> str:
        """Plain-text ledger the router reads before anything else.

        For a person's thread: every task they are on (open first), plus tasks this
        thread touched. For Aaron: every live task, then recently finished ones."""
        now = now or time.time()
        ids: List[int] = []
        if person:
            for t in self.tasks_for_person(person, open_only=False, limit=6):
                ids.append(int(t["id"]))
        for tid in self.task_ids_for_chat(chat_id):
            if tid not in ids:
                ids.append(tid)
        if is_approver:
            for t in self.recent_tasks():
                if t["id"] not in ids:
                    ids.append(int(t["id"]))
        blocks: List[str] = []
        for tid in ids[:14]:
            t = self.task_with_events(tid)
            if not t or t["state"] == "closed":
                continue
            lines = [self._task_head(t)]
            for e in t["events"]:
                rid = f" (#{e['request_id']})" if e.get("request_id") else ""
                lines.append(f"  - {_age(now - e['created_at'])} {e['kind']}{rid}: {e['text']}")
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
