"""SQLite persistence for threads, messages, and requests."""

from __future__ import annotations

import hashlib
import json
import re
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pgdb import PGConnection, is_postgres_target

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
CREATE TABLE IF NOT EXISTS schedules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT NOT NULL,
  task_id INTEGER NOT NULL,
  title TEXT NOT NULL,
  prompt TEXT NOT NULL,
  prompt_sha256 TEXT NOT NULL,
  kind TEXT NOT NULL,
  cron TEXT,
  run_at REAL,
  timezone TEXT NOT NULL DEFAULT 'America/New_York',
  scopes_json TEXT NOT NULL,
  report_mode TEXT NOT NULL DEFAULT 'always',
  state TEXT NOT NULL DEFAULT 'proposed',
  next_run REAL,
  last_run REAL,
  last_request_id INTEGER,
  run_count INTEGER NOT NULL DEFAULT 0,
  max_runs INTEGER NOT NULL DEFAULT 20,
  deadline REAL,
  notes TEXT NOT NULL DEFAULT '[]',
  revision INTEGER NOT NULL DEFAULT 0,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  last_outcome TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS schedules_due ON schedules(state, next_run);
CREATE TABLE IF NOT EXISTS persons (
  id TEXT PRIMARY KEY,            -- stable generated id, never derived from a contact
  display TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL DEFAULT '',
  scopes TEXT NOT NULL DEFAULT '[]',
  note TEXT NOT NULL DEFAULT '',
  merged_into TEXT,               -- set when this person was linked (merged) into another
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS persons_merged ON persons(merged_into);
CREATE TABLE IF NOT EXISTS contacts (
  id TEXT PRIMARY KEY,
  person_id TEXT,                 -- NULL = unlinked: seen in traffic, not attached to anyone
  kind TEXT NOT NULL,              -- email | phone | telegram | imessage | name | other
  value TEXT NOT NULL,             -- normalised (the same normalisation as the old trust key)
  raw_value TEXT NOT NULL DEFAULT '',  -- exactly as entered/seen, never truncated
  source TEXT NOT NULL DEFAULT '',     -- manual | sync | traffic | migration
  last_seen REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0,
  UNIQUE(kind, value)
);
CREATE INDEX IF NOT EXISTS contacts_person ON contacts(person_id);
CREATE TABLE IF NOT EXISTS contact_kinds (
  kind TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  hint TEXT NOT NULL DEFAULT '',
  addable INTEGER NOT NULL DEFAULT 1,
  sort_order INTEGER NOT NULL DEFAULT 0
);
"""

# Postgres gets the FINAL current shape directly (every column every sqlite
# ALTER/migrate step below would otherwise add), since a Postgres database
# only ever starts from the migration script's already-migrated data -- the
# ALTER-TABLE-if-column-missing dance and _migrate_person_keyed_tasks /
# _migrate_people_model below exist solely to carry an OLD sqlite file
# forward through this codebase's history; they do not apply to a database
# that starts empty and is filled by the migration script in one shot.
#
# task_fts is no longer an FTS5 virtual table: it is a normal table plus a
# generated tsvector column `doc` with a GIN index. query_tasks() branches on
# self._is_pg to query it with websearch_to_tsquery/ts_rank_cd instead of
# FTS5 MATCH/bm25 -- see the `_is_pg` branch there for the documented
# difference in ranking and matching behaviour.
SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS threads (
  chat_id TEXT PRIMARY KEY,
  state TEXT NOT NULL DEFAULT 'idle',
  mode TEXT,
  meta_json TEXT,
  updated_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id BIGSERIAL PRIMARY KEY,
  chat_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  mode TEXT,
  text TEXT NOT NULL,
  created_at DOUBLE PRECISION NOT NULL,
  reply_to BIGINT,
  role TEXT
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
CREATE INDEX IF NOT EXISTS messages_reply_to ON messages(reply_to);
CREATE TABLE IF NOT EXISTS mail_ids (
  message_id TEXT PRIMARY KEY,
  root TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS thread_links (
  thread_key TEXT NOT NULL,
  chat_id TEXT NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (thread_key, chat_id)
);
CREATE TABLE IF NOT EXISTS requests (
  id BIGSERIAL PRIMARY KEY,
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
  state TEXT NOT NULL,
  revision INTEGER NOT NULL DEFAULT 0,
  status_json TEXT,
  raw_output TEXT,
  created_at DOUBLE PRECISION NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL,
  task_id BIGINT,
  inbound_id BIGINT,
  schedule_id BIGINT,
  schedule_kind TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS requests_state ON requests(state);
CREATE TABLE IF NOT EXISTS tasks (
  id BIGSERIAL PRIMARY KEY,
  key TEXT NOT NULL DEFAULT '',
  display TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'open',
  created_at DOUBLE PRECISION NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_key ON tasks(key, updated_at);
CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state, updated_at);
CREATE TABLE IF NOT EXISTS task_participants (
  task_id BIGINT NOT NULL,
  key TEXT NOT NULL,
  display TEXT NOT NULL,
  person_id TEXT NOT NULL DEFAULT '',
  seq BIGSERIAL,
  PRIMARY KEY (task_id, key)
);
CREATE INDEX IF NOT EXISTS task_participants_key ON task_participants(key);
CREATE INDEX IF NOT EXISTS task_participants_person ON task_participants(person_id);
CREATE TABLE IF NOT EXISTS people (
  person_id TEXT NOT NULL,
  key TEXT NOT NULL,
  display TEXT NOT NULL DEFAULT '',
  updated_at DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (person_id, key)
);
CREATE INDEX IF NOT EXISTS people_key ON people(key);
CREATE TABLE IF NOT EXISTS task_fts (
  task_id BIGINT NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  summary TEXT NOT NULL DEFAULT '',
  people TEXT NOT NULL DEFAULT '',
  events TEXT NOT NULL DEFAULT '',
  requests TEXT NOT NULL DEFAULT '',
  dates TEXT NOT NULL DEFAULT '',
  doc tsvector GENERATED ALWAYS AS (
    to_tsvector('english', regexp_replace(
      coalesce(title,'') || ' ' || coalesce(summary,'') || ' ' || coalesce(people,'') || ' ' ||
      coalesce(events,'') || ' ' || coalesce(requests,'') || ' ' || coalesce(dates,''),
      '[^[:alnum:]]+', ' ', 'g'))
  ) STORED
);
CREATE INDEX IF NOT EXISTS task_fts_task ON task_fts(task_id);
CREATE INDEX IF NOT EXISTS task_fts_doc ON task_fts USING GIN(doc);
CREATE TABLE IF NOT EXISTS task_events (
  id BIGSERIAL PRIMARY KEY,
  task_id BIGINT NOT NULL,
  kind TEXT NOT NULL,
  chat_id TEXT,
  request_id BIGINT,
  text TEXT NOT NULL,
  created_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS task_events_task ON task_events(task_id, id);
CREATE TABLE IF NOT EXISTS schedules (
  id BIGSERIAL PRIMARY KEY,
  chat_id TEXT NOT NULL,
  task_id BIGINT NOT NULL,
  title TEXT NOT NULL,
  prompt TEXT NOT NULL,
  prompt_sha256 TEXT NOT NULL,
  kind TEXT NOT NULL,
  cron TEXT,
  run_at DOUBLE PRECISION,
  timezone TEXT NOT NULL DEFAULT 'America/New_York',
  scopes_json TEXT NOT NULL,
  report_mode TEXT NOT NULL DEFAULT 'always',
  state TEXT NOT NULL DEFAULT 'proposed',
  next_run DOUBLE PRECISION,
  last_run DOUBLE PRECISION,
  last_request_id BIGINT,
  run_count INTEGER NOT NULL DEFAULT 0,
  max_runs INTEGER NOT NULL DEFAULT 20,
  deadline DOUBLE PRECISION,
  notes TEXT NOT NULL DEFAULT '[]',
  revision INTEGER NOT NULL DEFAULT 0,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  last_outcome TEXT NOT NULL DEFAULT '',
  created_at DOUBLE PRECISION NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS schedules_due ON schedules(state, next_run);
CREATE TABLE IF NOT EXISTS persons (
  id TEXT PRIMARY KEY,
  display TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL DEFAULT '',
  scopes TEXT NOT NULL DEFAULT '[]',
  note TEXT NOT NULL DEFAULT '',
  merged_into TEXT,
  created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
  updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS persons_merged ON persons(merged_into);
CREATE TABLE IF NOT EXISTS contacts (
  id TEXT PRIMARY KEY,
  person_id TEXT,
  kind TEXT NOT NULL,
  value TEXT NOT NULL,
  raw_value TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT '',
  last_seen DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
  updated_at DOUBLE PRECISION NOT NULL DEFAULT 0,
  UNIQUE(kind, value)
);
CREATE INDEX IF NOT EXISTS contacts_person ON contacts(person_id);
CREATE TABLE IF NOT EXISTS roles (
  name TEXT PRIMARY KEY, scopes TEXT NOT NULL DEFAULT '[]',
  note TEXT NOT NULL DEFAULT '', updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS trust (
  key TEXT PRIMARY KEY, person TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL DEFAULT '', scopes TEXT NOT NULL DEFAULT '[]',
  note TEXT NOT NULL DEFAULT '', updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS synced_people (
  key TEXT PRIMARY KEY, person TEXT NOT NULL DEFAULT '',
  channels TEXT NOT NULL DEFAULT '[]', last_seen DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS settings (
  name TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '',
  updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS contact_kinds (
  kind TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  hint TEXT NOT NULL DEFAULT '',
  addable BOOLEAN NOT NULL DEFAULT true,
  sort_order INTEGER NOT NULL DEFAULT 0
);
"""

# The ways a person can be reached, in console display order. Lives in the
# contact_kinds table (seeded here if empty) so the console API reads it from
# the database instead of a hard-coded list; see Store.contact_kinds().
_CONTACT_KINDS_SEED = [
    ("email", "Email", "name@example.com", True, 0),
    ("phone", "Phone", "+1 555 010 0001", True, 1),
    ("telegram", "Telegram", "Telegram user id (digits)", True, 2),
    ("imessage", "iMessage", "phone or Apple ID email", True, 3),
    ("other", "Other", "handle", True, 4),
    ("name", "Name", "", False, 5),
]

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


def _pg_tsquery(text: str) -> str:
    """The same search as _fts_query, written as a Postgres tsquery: quoted phrases
    kept, OR honoured, every other word required, each word matched as a prefix. Text
    is indexed with punctuation turned into spaces, so an address such as eve@venue.com
    is the words eve, venue, com in order, and a search for venue.com finds it."""
    import re
    terms: List[str] = []
    pending_or = False
    for tok in re.findall(r'"[^"]+"|\S+', text or ""):
        if tok.upper() == "OR" and terms:
            pending_or = True
            continue
        words = [w for w in re.split(r"[^0-9A-Za-z\u00C0-\uFFFF]+", tok) if w]
        if not words:
            continue
        quoted = tok.startswith('"') and tok.endswith('"') and len(tok) > 2
        if quoted:
            term = "(" + " <-> ".join(words) + ")"
        else:
            if len("".join(words)) < 2:
                continue
            term = "(" + " <-> ".join(words[:-1] + [words[-1] + ":*"]) + ")"
        if terms:
            terms.append("|" if pending_or else "&")
        terms.append(term)
        pending_or = False
    return " ".join(terms)


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
    schedule_id: Optional[int] = None
    schedule_kind: str = ""

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
            schedule_id=(int(r["schedule_id"]) if "schedule_id" in r.keys() and r["schedule_id"] else None),
            schedule_kind=(str(r["schedule_kind"] or "") if "schedule_kind" in r.keys() else ""),
        )


@dataclass
class Schedule:
    id: int
    chat_id: str
    task_id: int
    title: str
    prompt: str
    prompt_sha256: str
    kind: str
    cron: str
    run_at: Optional[float]
    timezone: str
    scopes: List[str]
    report_mode: str
    state: str
    next_run: Optional[float]
    last_run: Optional[float]
    last_request_id: Optional[int]
    run_count: int
    max_runs: int
    deadline: Optional[float]
    notes: List[str]
    revision: int
    consecutive_failures: int
    last_outcome: str
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Schedule":
        return cls(
            id=int(r["id"]), chat_id=str(r["chat_id"]), task_id=int(r["task_id"]),
            title=str(r["title"]), prompt=str(r["prompt"]), prompt_sha256=str(r["prompt_sha256"]),
            kind=str(r["kind"]), cron=str(r["cron"] or ""),
            run_at=float(r["run_at"]) if r["run_at"] is not None else None,
            timezone=str(r["timezone"]), scopes=json.loads(r["scopes_json"] or "[]"),
            report_mode=str(r["report_mode"]), state=str(r["state"]),
            next_run=float(r["next_run"]) if r["next_run"] is not None else None,
            last_run=float(r["last_run"]) if r["last_run"] is not None else None,
            last_request_id=int(r["last_request_id"]) if r["last_request_id"] else None,
            run_count=int(r["run_count"]), max_runs=int(r["max_runs"]),
            deadline=float(r["deadline"]) if r["deadline"] is not None else None,
            notes=list(json.loads(r["notes"] or "[]")), revision=int(r["revision"]),
            consecutive_failures=int(r["consecutive_failures"]), last_outcome=str(r["last_outcome"] or ""),
            created_at=float(r["created_at"]), updated_at=float(r["updated_at"]),
        )


def role_names(value: Any) -> List[str]:
    """The roles held by one trust row. A person can hold several; they are kept in the
    one `role` column separated by commas."""
    out: List[str] = []
    for part in str(value or "").split(","):
        name = " ".join(part.split()).lower()
        if name and name not in out:
            out.append(name)
    return out


def _trust_key(value: str) -> str:
    """One handle, normalised. A phone is its digits, anything else is lowercase text, so
    the same person matches whether they arrive as +1 (555) 010-0001 or 15550100001."""
    v = " ".join(str(value or "").split()).lower()
    digits = re.sub(r"\D", "", v)
    if len(digits) >= 10 and not any(c.isalpha() for c in v):
        return digits[-10:]
    return v


class Store:
    def __init__(self, path: str):
        self._is_pg = is_postgres_target(path)
        if self._is_pg:
            self._db = PGConnection(path)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            if self._is_pg:
                # Postgres always gets the final current shape in one shot --
                # see SCHEMA_PG's docstring-comment above for why the
                # ALTER-TABLE-if-missing dance and the two one-time sqlite
                # migrations below do not apply here.
                self._db.executescript(SCHEMA_PG)
                self._seed_contact_kinds_locked()
                self._db.commit()
            else:
                self._db.executescript(SCHEMA)
                cols = {r["name"] for r in self._db.execute("PRAGMA table_info(requests)")}
                if "task_id" not in cols:
                    self._db.execute("ALTER TABLE requests ADD COLUMN task_id INTEGER")
                if "inbound_id" not in cols:
                    self._db.execute("ALTER TABLE requests ADD COLUMN inbound_id INTEGER")
                if "schedule_id" not in cols:
                    self._db.execute("ALTER TABLE requests ADD COLUMN schedule_id INTEGER")
                if "schedule_kind" not in cols:
                    self._db.execute("ALTER TABLE requests ADD COLUMN schedule_kind TEXT NOT NULL DEFAULT ''")
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
                self._migrate_people_model()
                self._seed_contact_kinds_locked()
                self._db.commit()
            n_tasks = self._db.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
            n_fts = self._db.execute("SELECT COUNT(*) AS n FROM task_fts").fetchone()["n"]
        if n_fts < n_tasks:
            self.reindex_all()

    def _seed_contact_kinds_locked(self) -> None:
        """Idempotent: insert the fixed contact-kind rows if the table is
        empty. Assumes self._lock is already held and contact_kinds exists."""
        row = self._db.execute("SELECT COUNT(*) AS n FROM contact_kinds").fetchone()
        if int(row["n"]) > 0:
            return
        for kind, label, hint, addable, order in _CONTACT_KINDS_SEED:
            self._db.execute(
                "INSERT INTO contact_kinds(kind,label,hint,addable,sort_order) VALUES(?,?,?,?,?)",
                (kind, label, hint, addable, order),
            )

    def contact_kinds(self) -> List[Dict[str, Any]]:
        """The ways a person can be reached, in console display order. Read
        from the database (contact_kinds table) instead of a hard-coded list."""
        with self._lock:
            rows = self._db.execute(
                "SELECT kind, label, hint, addable FROM contact_kinds ORDER BY sort_order, kind"
            ).fetchall()
        return [{"kind": r["kind"], "label": r["label"], "hint": r["hint"],
                 "addable": bool(r["addable"])} for r in rows]

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

    def mark_owner_thread(self, chat_id: str, mode: str = "") -> None:
        """Remember that this thread is the owner's. Notices have to reach his record even
        when no session for him is live in memory, which is the usual case overnight."""
        if not chat_id:
            return
        # json_set is sqlite-only; Postgres's jsonb_set has a similar shape
        # but needs the target cast to jsonb and the new value as a jsonb
        # literal, and returns jsonb (meta_json is stored as TEXT -- see the
        # migration note on keeping JSON columns as text -- so the result is
        # cast back to text to match).
        if self._is_pg:
            set_owner_expr = (
                "jsonb_set(COALESCE(threads.meta_json,'{}')::jsonb, '{owner}', '1', true)::text"
            )
        else:
            set_owner_expr = "json_set(COALESCE(threads.meta_json,'{}'),'$.owner',1)"
        with self._lock:
            self._db.execute(
                "INSERT INTO threads(chat_id,state,mode,meta_json,updated_at) VALUES(?,?,?,?,?) "
                f"ON CONFLICT(chat_id) DO UPDATE SET meta_json={set_owner_expr}, "
                "mode=COALESCE(excluded.mode, threads.mode), updated_at=excluded.updated_at",
                (chat_id, "idle", mode or None, '{"owner": 1}', time.time()),
            )
            self._db.commit()

    def owner_threads(self) -> List[str]:
        """Every thread known to be the owner's, newest first."""
        with self._lock:
            if self._is_pg:
                rows = self._db.execute(
                    "SELECT chat_id FROM threads WHERE "
                    "(COALESCE(meta_json,'{}')::jsonb->>'owner') IN ('1','true') "
                    "ORDER BY updated_at DESC"
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT chat_id FROM threads WHERE json_extract(COALESCE(meta_json,'{}'),'$.owner')=1 "
                    "ORDER BY updated_at DESC"
                ).fetchall()
        return [str(r["chat_id"]) for r in rows]

    # -- who may act without asking ------------------------------------------
    def _ensure_trust(self) -> None:
        with self._lock:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS roles (
                  name TEXT PRIMARY KEY,
                  scopes TEXT NOT NULL DEFAULT '[]',
                  note TEXT NOT NULL DEFAULT '',
                  updated_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS trust (
                  key TEXT PRIMARY KEY,
                  person TEXT NOT NULL DEFAULT '',
                  role TEXT NOT NULL DEFAULT '',
                  scopes TEXT NOT NULL DEFAULT '[]',
                  note TEXT NOT NULL DEFAULT '',
                  updated_at REAL NOT NULL DEFAULT 0
                );
            """)
            self._db.commit()

    def set_role(self, name: str, scopes: List[str], note: str = "") -> None:
        """A named level of trust and the scopes it may use without asking the owner."""
        self._ensure_trust()
        with self._lock:
            self._db.execute(
                "INSERT INTO roles(name,scopes,note,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET scopes=excluded.scopes, note=excluded.note, "
                "updated_at=excluded.updated_at",
                (name.strip().lower(), json.dumps(sorted(set(scopes))), note, time.time()))
            self._db.commit()

    def drop_role(self, name: str) -> None:
        self._ensure_trust()
        with self._lock:
            self._db.execute("DELETE FROM roles WHERE name=?", (name.strip().lower(),))
            self._db.commit()

    def roles(self) -> List[Dict[str, Any]]:
        self._ensure_trust()
        with self._lock:
            rows = self._db.execute("SELECT * FROM roles ORDER BY name").fetchall()
        return [{"name": r["name"], "scopes": json.loads(r["scopes"] or "[]"), "note": r["note"]} for r in rows]

    def set_trust(self, key: str, *, person: str = "", role: str = "", scopes: Optional[List[str]] = None,
                  note: str = "") -> None:
        """Legacy single-handle grant, now a thin view over the person model: the handle's
        contact is found or created, the person it is attached to (creating one if the
        handle was unlinked) gets the role/scopes/note/display. Kept for every old caller
        (console upsert_person, role-member add/remove) so they go through the person
        without needing their own rewrite."""
        norm = self.normalize_contact_value(key)
        kind = self.guess_contact_kind(key)
        with self._lock:
            existing = self._db.execute(
                "SELECT * FROM contacts WHERE kind=? AND value=?", (kind, norm)
            ).fetchone()
            now = time.time()
            if existing and existing["person_id"]:
                pid = self._resolve_person(existing["person_id"])
            else:
                pid = self._new_id("p")
                self._db.execute(
                    "INSERT INTO persons(id,display,role,scopes,note,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?)", (pid, person or key, "", "[]", "", now, now))
                if existing:
                    self._db.execute("UPDATE contacts SET person_id=?, updated_at=? WHERE id=?",
                                      (pid, now, existing["id"]))
                else:
                    self._db.execute(
                        "INSERT INTO contacts(id,person_id,kind,value,raw_value,source,last_seen,"
                        "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (self._new_id("c"), pid, kind, norm, key, "manual", now, now, now))
            self._db.execute(
                "UPDATE persons SET display=?, role=?, scopes=?, note=?, updated_at=? WHERE id=?",
                (person or self._person_row(pid).get("display") or key, ", ".join(role_names(role)),
                 json.dumps(sorted(set(scopes or []))), note, now, pid))
            self._db.commit()

    def drop_trust(self, key: str) -> None:
        """Legacy single-handle removal: detach and delete that one contact. If it was
        the person's only contact, the person goes too; otherwise the person (and its
        remaining contacts) stays, with role/scopes left as they were."""
        norm = self.normalize_contact_value(key)
        kind = self.guess_contact_kind(key)
        with self._lock:
            existing = self._db.execute(
                "SELECT * FROM contacts WHERE kind=? AND value=?", (kind, norm)
            ).fetchone()
            if existing is None:
                return
            pid = existing["person_id"]
            self._db.execute("DELETE FROM contacts WHERE id=?", (existing["id"],))
            if pid:
                remaining = self._db.execute(
                    "SELECT COUNT(*) AS n FROM contacts WHERE person_id=?", (pid,)
                ).fetchone()["n"]
                if remaining == 0:
                    self._db.execute("DELETE FROM persons WHERE id=?", (pid,))
                    self._db.execute("UPDATE persons SET merged_into=NULL WHERE merged_into=?", (pid,))
            self._db.commit()

    def trusted(self) -> List[Dict[str, Any]]:
        """One row per contact of every live (non-merged) person, in the old trust-row
        shape plus person_id/contact_id so the console can group them back into people."""
        with self._lock:
            rows = self._db.execute(
                "SELECT c.id AS contact_id, c.person_id AS person_id, c.kind AS kind, c.value AS value, "
                "p.display AS person, p.role AS role, p.scopes AS scopes, p.note AS note "
                "FROM contacts c JOIN persons p ON p.id = c.person_id "
                "WHERE c.person_id IS NOT NULL AND p.merged_into IS NULL "
                "ORDER BY p.display, c.value"
            ).fetchall()
        return [{"key": r["value"], "person": r["person"], "role": r["role"] or "",
                 "scopes": json.loads(r["scopes"] or "[]"), "note": r["note"] or "",
                 "person_id": r["person_id"], "contact_id": r["contact_id"], "contact_kind": r["kind"]}
                for r in rows]

    def trust_for(self, keys: List[str]) -> Dict[str, Any]:
        """What this sender may do without asking, via resolve_handles. Zero or
        conflicting matches both grant nothing -- that is the safe default for an
        unknown sender, and the deliberate choice when two different people's handles
        land on the same message."""
        result = self.resolve_handles(keys)
        return {"role": result["role"], "scopes": result["scopes"], "person": result["person"]}

    def known_people(self) -> List[Dict[str, Any]]:
        """Let the console show known contacts before the owner grants them anything."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key, MAX(display) AS person FROM people GROUP BY key ORDER BY person, key"
            ).fetchall()
        merged: Dict[str, Dict[str, Any]] = {
            r["key"]: {"key": r["key"], "person": r["person"] or "", "channels": [], "last_seen": 0.0}
            for r in rows
        }
        for row in self.synced_people():
            entry = merged.setdefault(row["key"], {
                "key": row["key"], "person": "", "channels": [], "last_seen": 0.0,
            })
            if row.get("person") and (not entry["person"] or len(row["person"]) > len(entry["person"])):
                entry["person"] = row["person"]
            entry["channels"] = sorted(set(entry.get("channels") or []) | set(row.get("channels") or []))
            entry["last_seen"] = max(entry.get("last_seen") or 0.0, row.get("last_seen") or 0.0)
        return sorted(merged.values(), key=lambda r: ((r.get("person") or "").lower(), r["key"]))

    def _ensure_synced_people(self) -> None:
        with self._lock:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS synced_people (
                  key TEXT PRIMARY KEY,
                  person TEXT NOT NULL DEFAULT '',
                  channels TEXT NOT NULL DEFAULT '[]',
                  last_seen REAL NOT NULL DEFAULT 0
                );
            """)
            self._db.commit()

    def remember_synced_people(self, rows: List[Dict[str, Any]]) -> None:
        """Merge directory rows from an Inkbox sync. Never touches trust or roles."""
        self._ensure_synced_people()
        # Scalar multi-arg MAX(a,b) is sqlite-only; Postgres needs GREATEST(a,b).
        last_seen_expr = ("GREATEST(synced_people.last_seen, excluded.last_seen)" if self._is_pg
                          else "MAX(synced_people.last_seen, excluded.last_seen)")
        with self._lock:
            for row in rows:
                self._db.execute(
                    "INSERT INTO synced_people(key,person,channels,last_seen) VALUES(?,?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "person=CASE WHEN length(excluded.person) > length(synced_people.person) "
                    "THEN excluded.person ELSE synced_people.person END, "
                    "channels=excluded.channels, "
                    f"last_seen={last_seen_expr}",
                    (row["key"], row.get("person") or "", json.dumps(row.get("channels") or []),
                     float(row.get("last_seen") or 0)),
                )
            self._db.commit()

    def synced_people(self) -> List[Dict[str, Any]]:
        self._ensure_synced_people()
        with self._lock:
            rows = self._db.execute("SELECT * FROM synced_people").fetchall()
        return [{"key": r["key"], "person": r["person"],
                 "channels": json.loads(r["channels"] or "[]"), "last_seen": r["last_seen"]} for r in rows]

    # -- person-centric trust model --------------------------------------
    # One human is a `person` row (stable id, display name, role, scopes, note).
    # Any number of `contacts` (email/phone/telegram/imessage/name/other) point at
    # a person; a contact with person_id=NULL was seen but never attached to anyone
    # ("unlinked"). Role and scopes live ONLY on the person. The old trust/roles
    # tables are left in place untouched for rollback; trusted()/set_trust()/
    # drop_trust()/trust_for() below are now thin views over persons+contacts so
    # every existing caller (manager.trust_for, the console role/people views, the
    # people sync) goes through the person without having to change its own code.

    @staticmethod
    def _new_id(prefix: str) -> str:
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def guess_contact_kind(value: str) -> str:
        v = (value or "").strip()
        if v.lower().startswith("telegram:"):
            return "telegram"
        if "@" in v:
            return "email"
        digits = re.sub(r"\D", "", v)
        if len(digits) >= 7 and not any(c.isalpha() for c in v):
            return "phone"
        if not v:
            return "other"
        return "name"

    @staticmethod
    def normalize_contact_value(value: str) -> str:
        """Same normalisation the old trust key used, so a migrated contact collides
        with the same value a live message arrives on."""
        v = str(value or "").strip()
        if v.lower().startswith("telegram:"):
            v = v.split(":", 1)[1]
        return _trust_key(v)

    @classmethod
    def contact_value(cls, kind: str, value: str) -> str:
        return cls.normalize_contact_value(value)

    def _persons_cols(self) -> set:
        return {r["name"] for r in self._db.execute("PRAGMA table_info(persons)")}

    def _resolve_person(self, person_id: Optional[str]) -> Optional[str]:
        """Follow merged_into to the live person a (possibly merged) id now points at."""
        seen = set()
        pid = person_id
        while pid and pid not in seen:
            seen.add(pid)
            row = self._db.execute("SELECT merged_into FROM persons WHERE id=?", (pid,)).fetchone()
            if row is None:
                return None
            if not row["merged_into"]:
                return pid
            pid = row["merged_into"]
        return None

    def _person_row(self, person_id: str) -> Optional[Dict[str, Any]]:
        r = self._db.execute("SELECT * FROM persons WHERE id=?", (person_id,)).fetchone()
        if r is None:
            return None
        return {
            "id": r["id"], "display": r["display"], "role": r["role"],
            "scopes": json.loads(r["scopes"] or "[]"), "note": r["note"],
            "merged_into": r["merged_into"], "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }

    def _contact_rows(self, person_id: str) -> List[Dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM contacts WHERE person_id=? ORDER BY kind, value", (person_id,)
        ).fetchall()
        return [self._contact_dict(r) for r in rows]

    @staticmethod
    def _contact_dict(r: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": r["id"], "person_id": r["person_id"], "kind": r["kind"], "value": r["value"],
            "raw_value": r["raw_value"], "source": r["source"], "last_seen": r["last_seen"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
        }

    def create_person(self, *, display: str = "", role: str = "", scopes: Optional[List[str]] = None,
                       note: str = "") -> Dict[str, Any]:
        pid = self._new_id("p")
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO persons(id,display,role,scopes,note,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (pid, display.strip(), ", ".join(role_names(role)),
                 json.dumps(sorted(set(scopes or []))), note, now, now),
            )
            self._db.commit()
        return self.get_person(pid)

    def update_person(self, person_id: str, *, display: Optional[str] = None, role: Optional[str] = None,
                       scopes: Optional[List[str]] = None, note: Optional[str] = None) -> Dict[str, Any]:
        live = self._resolve_person(person_id)
        if live is None:
            raise KeyError(f"no such person: {person_id}")
        fields, values = [], []
        if display is not None:
            fields.append("display=?"); values.append(display.strip())
        if role is not None:
            fields.append("role=?"); values.append(", ".join(role_names(role)))
        if scopes is not None:
            fields.append("scopes=?"); values.append(json.dumps(sorted(set(scopes))))
        if note is not None:
            fields.append("note=?"); values.append(note)
        fields.append("updated_at=?"); values.append(time.time())
        with self._lock:
            self._db.execute(f"UPDATE persons SET {', '.join(fields)} WHERE id=?", (*values, live))
            self._db.commit()
        return self.get_person(live)

    def delete_person(self, person_id: str) -> bool:
        live = self._resolve_person(person_id)
        if live is None:
            return False
        with self._lock:
            self._db.execute("UPDATE contacts SET person_id=NULL, updated_at=? WHERE person_id=?",
                              (time.time(), live))
            self._db.execute("DELETE FROM persons WHERE id=?", (live,))
            self._db.execute("UPDATE persons SET merged_into=NULL WHERE merged_into=?", (live,))
            self._db.commit()
        return True

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        live = self._resolve_person(person_id)
        if live is None:
            return None
        row = self._person_row(live)
        if row is None:
            return None
        row["contacts"] = self._contact_rows(live)
        return row

    def list_people(self) -> List[Dict[str, Any]]:
        """Every live person with their contacts, newest-created last; merged-away
        persons are not listed (follow merged_into to find who absorbed them)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM persons WHERE merged_into IS NULL ORDER BY display, id"
            ).fetchall()
            out = [self.get_person(r["id"]) for r in rows]
        return [p for p in out if p is not None]

    def unlinked_contacts(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM contacts WHERE person_id IS NULL ORDER BY kind, value"
            ).fetchall()
        return [self._contact_dict(r) for r in rows]

    def find_contact(self, kind: str, value: str) -> Optional[Dict[str, Any]]:
        norm = self.normalize_contact_value(value)
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM contacts WHERE kind=? AND value=?", (kind, norm)
            ).fetchone()
        return self._contact_dict(r) if r else None

    def _upsert_contact_locked(self, kind: str, raw_value: str, *, person_id: Optional[str],
                                source: str, last_seen: Optional[float]) -> Dict[str, Any]:
        """Core of upsert_contact, assuming self._lock is already held. If the contact
        already belongs to a DIFFERENT person than `person_id` asks for, nothing is
        moved -- the existing owner is reported back. person_id=None (a sync or live
        traffic call) never detaches a contact a human already assigned; it only
        refreshes last_seen/source, or creates a new UNLINKED contact."""
        norm = self.normalize_contact_value(raw_value)
        now = time.time()
        seen = float(last_seen) if last_seen is not None else now
        live_person = self._resolve_person(person_id) if person_id else None
        existing = self._db.execute(
            "SELECT * FROM contacts WHERE kind=? AND value=?", (kind, norm)
        ).fetchone()
        if existing is None:
            cid = self._new_id("c")
            self._db.execute(
                "INSERT INTO contacts(id,person_id,kind,value,raw_value,source,last_seen,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (cid, live_person, kind, norm, raw_value, source, seen, now, now),
            )
            self._db.commit()
            return {"ok": True, "conflict": False, "contact": self.get_contact(cid)}
        existing_pid = existing["person_id"]
        if live_person and existing_pid and existing_pid != live_person:
            return {"ok": False, "conflict": True, "existing_person_id": existing_pid,
                    "contact": self._contact_dict(existing)}
        new_pid = existing_pid or live_person  # attach if it was unlinked
        last_seen_expr = "GREATEST(last_seen, ?)" if self._is_pg else "MAX(last_seen, ?)"
        self._db.execute(
            "UPDATE contacts SET person_id=?, raw_value=?, source=?, "
            f"last_seen={last_seen_expr}, updated_at=? WHERE id=?",
            (new_pid, raw_value or existing["raw_value"], source or existing["source"],
             seen, now, existing["id"]),
        )
        self._db.commit()
        return {"ok": True, "conflict": False, "contact": self.get_contact(existing["id"])}

    def upsert_contact(self, kind: str, raw_value: str, *, person_id: Optional[str] = None,
                        source: str = "manual", last_seen: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            return self._upsert_contact_locked(kind, raw_value, person_id=person_id,
                                                source=source, last_seen=last_seen)

    def get_contact(self, contact_id: str) -> Optional[Dict[str, Any]]:
        r = self._db.execute("SELECT * FROM contacts WHERE id=?", (contact_id,)).fetchone()
        return self._contact_dict(r) if r else None

    def add_contact(self, person_id: str, kind: str, value: str, *, source: str = "manual") -> Dict[str, Any]:
        live = self._resolve_person(person_id)
        if live is None:
            raise KeyError(f"no such person: {person_id}")
        return self.upsert_contact(kind, value, person_id=live, source=source)

    def remove_contact(self, contact_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM contacts WHERE id=?", (contact_id,))
            self._db.commit()
        return cur.rowcount > 0

    def move_contact(self, contact_id: str, person_id: Optional[str]) -> Dict[str, Any]:
        """Reassign one contact to a person (or to None, to unlink it). Always an
        explicit action -- never inferred from a name or a shared role."""
        live = self._resolve_person(person_id) if person_id else None
        if person_id and live is None:
            raise KeyError(f"no such person: {person_id}")
        with self._lock:
            cur = self._db.execute(
                "UPDATE contacts SET person_id=?, updated_at=? WHERE id=?",
                (live, time.time(), contact_id),
            )
            self._db.commit()
        if cur.rowcount == 0:
            raise KeyError(f"no such contact: {contact_id}")
        return self.get_contact(contact_id)

    def link_people(self, keep_id: str, merge_id: str) -> Dict[str, Any]:
        """Merge merge_id into keep_id: every contact moves to keep_id, and merge_id is
        marked merged_into keep_id (not deleted -- its original role/scopes/note stay
        visible for the owner to review). No permission union happens: keep_id's role
        and scopes are exactly what they were before."""
        keep_live = self._resolve_person(keep_id)
        merge_live = self._resolve_person(merge_id)
        if keep_live is None:
            raise KeyError(f"no such person: {keep_id}")
        if merge_live is None:
            raise KeyError(f"no such person: {merge_id}")
        if keep_live == merge_live:
            raise ValueError("cannot link a person to themself")
        before = self.get_person(merge_live)
        now = time.time()
        with self._lock:
            self._db.execute(
                "UPDATE contacts SET person_id=?, updated_at=? WHERE person_id=?",
                (keep_live, now, merge_live),
            )
            self._db.execute(
                "UPDATE persons SET merged_into=?, updated_at=? WHERE id=?",
                (keep_live, now, merge_live),
            )
            self._db.commit()
        return {"ok": True, "kept": self.get_person(keep_live), "merged_person_was": before}

    def resolve_handles(self, handles: List[str]) -> Dict[str, Any]:
        """What role/scopes a message on ANY of these handles should get. Zero matching
        persons -> nothing. Exactly one -> that person's role+scopes. More than one
        DISTINCT person -> a real conflict (two different people's handles landed on one
        message): grant neither, and log it loudly so it gets noticed."""
        # A display name is whatever the sender typed, and a Telegram id is ten digits like a
        # phone number, so a match is made on kind as well as value: a telegram contact only
        # from a handle the gateway stamped "telegram:", an email or phone contact only from
        # a bare address or number, and a name contact never.
        wanted = set()
        for h in handles:
            h = str(h or "").strip()
            if not h:
                continue
            kind = self.guess_contact_kind(h)
            if kind in ("email", "phone", "telegram"):
                wanted.add((kind, self.contact_value(kind, h)))
        if not wanted:
            return {"role": "", "scopes": [], "person": "", "person_id": "", "conflict": False}
        clause = " OR ".join("(kind=? AND value=?)" for _ in wanted)
        args = [x for pair in sorted(wanted) for x in pair]
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT person_id FROM contacts WHERE ({clause}) AND person_id IS NOT NULL",
                args,
            ).fetchall()
        person_ids = {self._resolve_person(r["person_id"]) for r in rows}
        person_ids.discard(None)
        if not person_ids:
            return {"role": "", "scopes": [], "person": "", "person_id": "", "conflict": False}
        if len(person_ids) > 1:
            logger.warning(
                "resolve_handles: handles %s resolve to %d different people (%s) -- granting neither",
                handles, len(person_ids), sorted(person_ids),
            )
            return {"role": "", "scopes": [], "person": "", "person_id": "", "conflict": True}
        pid = next(iter(person_ids))
        person = self._person_row(pid) or {}
        roles = {r["name"]: json.loads(r["scopes"] or "[]")
                 for r in self._db.execute("SELECT * FROM roles").fetchall()}
        role = person.get("role") or ""
        role_scopes: List[str] = []
        for name in role_names(role):
            role_scopes += roles.get(name, [])
        scopes = sorted(set(person.get("scopes") or []) | set(role_scopes))
        return {"role": role, "scopes": scopes, "person": person.get("display") or "",
                "person_id": pid, "conflict": False}

    def _migrate_people_model(self) -> None:
        """One-time, idempotent: build persons+contacts from the legacy trust / people /
        synced_people tables. Never drops or mutates those legacy tables; safe to re-run
        (guarded by a settings flag, and every write below is itself an upsert on a
        UNIQUE(kind,value) contact or a PK persons.id)."""
        # Called from __init__ while self._lock is already held: create the legacy
        # tables directly (no re-entrant lock acquisition) instead of calling the
        # lazy _ensure_* helpers, which each try to take the lock themselves.
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS roles (
              name TEXT PRIMARY KEY, scopes TEXT NOT NULL DEFAULT '[]',
              note TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS trust (
              key TEXT PRIMARY KEY, person TEXT NOT NULL DEFAULT '',
              role TEXT NOT NULL DEFAULT '', scopes TEXT NOT NULL DEFAULT '[]',
              note TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS synced_people (
              key TEXT PRIMARY KEY, person TEXT NOT NULL DEFAULT '',
              channels TEXT NOT NULL DEFAULT '[]', last_seen REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS settings (
              name TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '',
              updated_at REAL NOT NULL DEFAULT 0
            );
        """)
        flag = self._db.execute(
            "SELECT value FROM settings WHERE name='people_model_migrated_v1'"
        ).fetchone()
        if flag is not None:
            return
        now = time.time()
        value_to_person: Dict[str, str] = {}
        telegram_keys = set()
        try:
            for r in self._db.execute("SELECT key, channels FROM synced_people").fetchall():
                if json.loads(r["channels"] or "[]") == ["telegram"]:
                    telegram_keys.add(self.normalize_contact_value(r["key"]))
        except sqlite3.OperationalError:
            pass

        def kind_of(key: str) -> str:
            if self.normalize_contact_value(key) in telegram_keys:
                return "telegram"
            return self.guess_contact_kind(key)

        # 1) trust rows -> one person per DISTINCT NON-EMPTY display name, carrying every
        # key, role and scope of that group onto the one person; a row with no display
        # name becomes its own person (nothing to group it by). This is reading the OLD
        # one-row-per-handle table, where a shared display name across two rows was
        # Aaron's only way of recording "these are the same person" -- it is the one-time
        # migration's job to not throw that fact away. It is not the ongoing rule for the
        # new model: resolve_handles and the console's "link people" action never merge
        # by display name once persons/contacts exist, exactly as decided for this model.
        trust_groups: Dict[str, List[sqlite3.Row]] = {}
        trust_order: List[str] = []
        for r in self._db.execute("SELECT * FROM trust ORDER BY key").fetchall():
            name = " ".join(str(r["person"] or "").split()).strip().lower()
            group_id = name or f"__key__{r['key']}"
            if group_id not in trust_groups:
                trust_groups[group_id] = []
                trust_order.append(group_id)
            trust_groups[group_id].append(r)
        for group_id in trust_order:
            members = trust_groups[group_id]
            display = next((m["person"] for m in members if m["person"]), members[0]["key"])
            roles_union: List[str] = []
            scopes_union: set = set()
            note = ""
            for m in members:
                for rn in role_names(m["role"]):
                    if rn not in roles_union:
                        roles_union.append(rn)
                scopes_union |= set(json.loads(m["scopes"] or "[]"))
                if m["note"] and not note:
                    note = m["note"]
            pid = self._new_id("p")
            self._db.execute(
                "INSERT INTO persons(id,display,role,scopes,note,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (pid, display, ", ".join(roles_union), json.dumps(sorted(scopes_union)), note, now, now),
            )
            for m in members:
                key = m["key"]
                norm = self.normalize_contact_value(key)
                kind = kind_of(key)
                self._db.execute(
                    "INSERT OR IGNORE INTO contacts(id,person_id,kind,value,raw_value,source,"
                    "last_seen,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (self._new_id("c"), pid, kind, norm, key, "migration", now, now, now),
                )
                value_to_person[norm] = pid

        # 2) people table: (person_id, key) pairs grouped by the Inkbox person_id. If any
        # of a group's keys already matches a person from step 1, that group's remaining
        # keys attach to that SAME person (an explicit, data-grounded link: the two
        # sources already agreed these keys are one person). Otherwise a new person is
        # made for the group.
        groups: Dict[str, List[sqlite3.Row]] = {}
        for r in self._db.execute("SELECT * FROM people").fetchall():
            groups.setdefault(r["person_id"], []).append(r)
        for _inkbox_pid, members in groups.items():
            norms = [(self.normalize_contact_value(m["key"]), m) for m in members]
            target = next((value_to_person[n] for n, _m in norms if n in value_to_person), None)
            if target is None:
                best_display = max((m["display"] or "" for _n, m in norms), key=len, default="")
                target = self._new_id("p")
                self._db.execute(
                    "INSERT INTO persons(id,display,role,scopes,note,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (target, best_display, "", "[]", "", now, now),
                )
            for norm, m in norms:
                if norm in value_to_person:
                    continue  # already attached in step 1
                kind = kind_of(m["key"])
                self._db.execute(
                    "INSERT OR IGNORE INTO contacts(id,person_id,kind,value,raw_value,source,"
                    "last_seen,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (self._new_id("c"), target, kind, norm, m["key"], "migration", now, now, now),
                )
                value_to_person[norm] = target

        # 3) synced_people: attach only where the key ALREADY belongs to a person
        # established by steps 1/2 (just a last_seen/source refresh); a key seen only in
        # the sync becomes an unlinked contact, never a new person.
        for r in self._db.execute("SELECT * FROM synced_people").fetchall():
            norm = self.normalize_contact_value(r["key"])
            kind = kind_of(r["key"])
            pid = value_to_person.get(norm)
            self._db.execute(
                "INSERT OR IGNORE INTO contacts(id,person_id,kind,value,raw_value,source,"
                "last_seen,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (self._new_id("c"), pid, kind, norm, r["key"], "migration", r["last_seen"] or 0, now, now),
            )

        self._db.execute(
            "INSERT INTO settings(name,value,updated_at) VALUES('people_model_migrated_v1','1',?) "
            "ON CONFLICT(name) DO NOTHING",
            (now,),
        )

    def thread_counterparts(self) -> List[Dict[str, Any]]:
        """Every handle with a thread in this database, for the sync to union in even when
        the Inkbox directory API did not surface it."""
        # A thread's chat_id is a conversation id, not a person. The handle the person
        # wrote from is the sender recorded on the thread; threads without one are skipped.
        with self._lock:
            if self._is_pg:
                # No json_valid() in Postgres; filter to parseable JSON objects
                # with a regex (well-formed enough to rule out plain strings)
                # instead of a try/cast per row.
                rows = self._db.execute(
                    "SELECT DISTINCT (meta_json::jsonb->>'sender') AS handle, mode AS channel "
                    "FROM threads WHERE meta_json IS NOT NULL AND meta_json ~ '^\\s*\\{' "
                    "AND (meta_json::jsonb->>'sender') IS NOT NULL"
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT DISTINCT json_extract(meta_json, '$.sender') AS handle, mode AS channel "
                    "FROM threads WHERE meta_json IS NOT NULL AND json_valid(meta_json)"
                ).fetchall()
        return [{"handle": r["handle"], "name": "", "channel": r["channel"] or "thread"}
                for r in rows if str(r["handle"] or "").strip()]

    def console_overview(self) -> Dict[str, Any]:
        """Keep dashboard totals accurate without loading entire operational tables."""
        self._ensure_trust()
        with self._lock:
            request_rows = self._db.execute(
                "SELECT state, COUNT(*) AS n FROM requests GROUP BY state"
            ).fetchall()
            task_rows = self._db.execute(
                "SELECT state, COUNT(*) AS n FROM tasks WHERE state<>'closed' GROUP BY state"
            ).fetchall()
            # One entry per person: a trusted person (persons/contacts) counts once no
            # matter how many contacts they have; an untrusted directory contact counts
            # by its own key.
            people = self._db.execute(
                "SELECT COUNT(*) AS n FROM ("
                "  SELECT person_id AS k FROM contacts WHERE person_id IS NOT NULL"
                "  UNION"
                "  SELECT key FROM people WHERE key NOT IN "
                "    (SELECT value FROM contacts WHERE person_id IS NOT NULL)"
                ")"
            ).fetchone()["n"]
            roles = self._db.execute("SELECT COUNT(*) AS n FROM roles").fetchone()["n"]
        return {
            "people": int(people),
            "roles": int(roles),
            "requests": {str(r["state"]): int(r["n"]) for r in request_rows},
            "tasks": {str(r["state"]): int(r["n"]) for r in task_rows},
        }

    def interrupted_request_count(self) -> int:
        """Surface restart-interrupted work separately from ordinary failures."""
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM requests WHERE state='failed' "
                "AND (status_json LIKE '%interrupted by a restart%' "
                "OR status_json LIKE '%interrupted by restart%')"
            ).fetchone()
        return int(row["n"])

    def recent_requests(self, state: str = "", limit: int = 100) -> List[Request]:
        """Give the console a bounded, newest-first request timeline."""
        limit = max(1, min(int(limit), 500))
        with self._lock:
            if state:
                rows = self._db.execute(
                    "SELECT * FROM requests WHERE state=? ORDER BY id DESC LIMIT ?",
                    (state, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [Request.from_row(r) for r in rows]

    @classmethod
    def trusted_key(cls, key: str) -> str:
        """Let permission editors identify the normalized row they just changed, in
        the same normalisation `trusted()` reports contacts under (a Telegram handle's
        `telegram:` prefix is stripped, same as every other contact value)."""
        return cls.normalize_contact_value(key)

    # -- knobs the owner turns -------------------------------------------------
    def _ensure_settings(self) -> None:
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS settings ("
                             "name TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '', "
                             "updated_at REAL NOT NULL DEFAULT 0)")
            self._db.commit()

    def settings(self) -> Dict[str, str]:
        """Every value the owner has set, overriding the environment."""
        self._ensure_settings()
        with self._lock:
            rows = self._db.execute("SELECT name, value FROM settings").fetchall()
        return {str(r["name"]): str(r["value"]) for r in rows}

    def set_setting(self, name: str, value: str) -> None:
        self._ensure_settings()
        with self._lock:
            self._db.execute(
                "INSERT INTO settings(name,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (name, str(value), time.time()))
            self._db.commit()

    def clear_setting(self, name: str) -> None:
        """Hand one knob back to the environment, or to its built-in default."""
        self._ensure_settings()
        with self._lock:
            self._db.execute("DELETE FROM settings WHERE name=?", (name,))
            self._db.commit()

    def set_scopes(self, rid: int, scopes: List[str]) -> None:
        """Widen one request's scopes after the run reported which tool it was missing. The
        prompt is untouched, so its hash still checks: what changed is what it may reach."""
        with self._lock:
            self._db.execute("UPDATE requests SET scopes_json=?, updated_at=? WHERE id=?",
                             (json.dumps(sorted(set(scopes))), time.time(), int(rid)))
            self._db.commit()

    def requests_in_state(self, state: str) -> List[Request]:
        """Every request sitting in one state. Used at startup to find runs the process
        died in the middle of."""
        with self._lock:
            rows = self._db.execute("SELECT * FROM requests WHERE state=? ORDER BY id", (state,)).fetchall()
        return [Request.from_row(r) for r in rows]

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
                       state: str, task_id: int, inbound_id: Optional[int] = None,
                       schedule_id: Optional[int] = None, schedule_kind: str = "") -> Request:
        """Create a request. ``task_id`` is mandatory: a request is always part of a task."""
        if not task_id:
            raise TaskRequired("a request must belong to a task")
        now = time.time()
        with self._lock:
            if not self._db.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
                raise TaskRequired(f"task T{task_id} does not exist")
            cur = self._db.execute(
                "INSERT INTO requests(chat_id,sender,sender_name,mode,subject,original_message,summary,"
                "scopes_json,prompt,prompt_sha256,state,created_at,updated_at,task_id,inbound_id,schedule_id,schedule_kind) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (chat_id, sender, sender_name, mode, subject, original_message, summary,
                 json.dumps(scopes), prompt, sha256(prompt), state, now, now, task_id, inbound_id,
                 schedule_id, schedule_kind),
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

    # -- schedules ------------------------------------------------------
    def create_schedule(self, *, chat_id: str, task_id: int, title: str, prompt: str,
                        kind: str, scopes: List[str], timezone: str, report_mode: str,
                        cron: str = "", run_at: Optional[float] = None,
                        next_run: Optional[float] = None, max_runs: int = 20,
                        deadline: Optional[float] = None) -> Schedule:
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO schedules(chat_id,task_id,title,prompt,prompt_sha256,kind,cron,run_at,timezone,"
                "scopes_json,report_mode,state,next_run,max_runs,deadline,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,'proposed',?,?,?,?,?)",
                (chat_id, task_id, title, prompt, sha256(prompt), kind, cron or None, run_at, timezone,
                 json.dumps(sorted(set(scopes))), report_mode, next_run, max_runs, deadline, now, now),
            )
            self._db.commit()
        return self.get_schedule(int(cur.lastrowid))  # type: ignore[return-value]

    def get_schedule(self, schedule_id: int) -> Optional[Schedule]:
        with self._lock:
            row = self._db.execute("SELECT * FROM schedules WHERE id=?", (int(schedule_id),)).fetchone()
        return Schedule.from_row(row) if row else None

    def schedules(self, states: Optional[List[str]] = None) -> List[Schedule]:
        with self._lock:
            if states:
                marks = ",".join("?" for _ in states)
                rows = self._db.execute(
                    f"SELECT * FROM schedules WHERE state IN ({marks}) ORDER BY id DESC", states).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM schedules ORDER BY id DESC").fetchall()
        return [Schedule.from_row(r) for r in rows]

    def due_schedules(self, now: float) -> List[Schedule]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM schedules WHERE state='active' AND next_run IS NOT NULL "
                "AND next_run<=? ORDER BY next_run,id", (now,)).fetchall()
        return [Schedule.from_row(r) for r in rows]

    def request_running_for_schedule(self, schedule_id: int) -> Optional[Request]:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM requests WHERE schedule_id=? AND state IN ('pending','approved','running') "
                "ORDER BY id DESC LIMIT 1", (int(schedule_id),)).fetchone()
        return Request.from_row(row) if row else None

    def pending_requests_for_schedule(self, schedule_id: int) -> List[Request]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM requests WHERE schedule_id=? AND state='pending' ORDER BY id",
                (int(schedule_id),)).fetchall()
        return [Request.from_row(row) for row in rows]

    def update_schedule(self, schedule_id: int, **changes: Any) -> Schedule:
        allowed = {"title", "prompt", "prompt_sha256", "kind", "cron", "run_at", "timezone", "scopes",
                   "scopes_json", "report_mode", "state", "next_run", "last_run", "last_request_id",
                   "run_count", "max_runs", "deadline", "notes", "revision", "consecutive_failures",
                   "last_outcome"}
        bad = set(changes) - allowed
        if bad:
            raise ValueError(f"unknown schedule columns: {sorted(bad)}")
        if not changes:
            return self.get_schedule(schedule_id)  # type: ignore[return-value]
        if "prompt" in changes:
            changes["prompt_sha256"] = sha256(str(changes["prompt"]))
        if "scopes" in changes:
            changes["scopes_json"] = json.dumps(sorted(set(changes.pop("scopes"))))
        if "notes" in changes and not isinstance(changes["notes"], str):
            changes["notes"] = json.dumps(changes["notes"], ensure_ascii=False)
        changes["updated_at"] = time.time()
        sets = ", ".join(f"{name}=?" for name in changes)
        with self._lock:
            self._db.execute(f"UPDATE schedules SET {sets} WHERE id=?", (*changes.values(), int(schedule_id)))
            self._db.commit()
        return self.get_schedule(schedule_id)  # type: ignore[return-value]

    def delete_schedule(self, schedule_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM schedules WHERE id=?", (int(schedule_id),))
            self._db.commit()
        return bool(cur.rowcount)

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
        """Record every key we know for a contact, so later lookups by any of them match.
        Also upserts each key into the new contacts table (person_id=None, source=
        'traffic'): a handle already attached to someone just gets last_seen refreshed;
        a brand-new handle shows up as an unlinked contact. Never creates a person and
        never detaches a contact a human already assigned."""
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
                kind = self.guess_contact_kind(k)
                self._upsert_contact_locked(kind, k, person_id=None, source="traffic", last_seen=now)
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
        # instr(a,b) is sqlite-only; Postgres's strpos(a,b) has the same
        # "0 means not found" semantics, so only the function name changes.
        no_at_fn = "strpos" if self._is_pg else "instr"
        with self._lock:
            for k in person.keys:
                if not k:
                    continue
                self._db.execute(
                    "INSERT INTO task_participants(task_id,key,display,person_id) VALUES(?,?,?,?) "
                    "ON CONFLICT(task_id,key) DO UPDATE SET "
                    f"display=CASE WHEN excluded.display<>'' AND {no_at_fn}(excluded.display,'@')=0 THEN excluded.display ELSE task_participants.display END, "
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
            # sqlite orders by its implicit rowid (insertion order); Postgres
            # has no equivalent, so task_participants carries an explicit
            # `seq` serial column in SCHEMA_PG for the same ordering.
            order_col = "seq" if self._is_pg else "rowid"
            ps = self._db.execute(
                f"SELECT key, display, person_id FROM task_participants WHERE task_id=? ORDER BY {order_col}",
                (task_id,),
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

    def set_task_title(self, task_id: int, title: str) -> None:
        """Rename a task (the reply writer named it after it was created from the message)."""
        title = " ".join((title or "").split())[:160]
        if not title:
            return
        with self._lock:
            self._db.execute("UPDATE tasks SET title=? WHERE id=?", (title, task_id))
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
                r = self.query_tasks(**{k: v for k, v in sub.items() if k not in ("any_of", "limit")}, limit=50)
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
            # the dates column holds space-separated ISO dates; a task matches if
            # any date falls in range. sqlite splits the string via a json_each
            # hack; Postgres has a direct equivalent in regexp_split_to_table.
            lo, hi = (date_from or "0000-01-01"), (date_to or "9999-12-31")
            if self._is_pg:
                where.append("EXISTS (SELECT 1 FROM task_fts f WHERE f.task_id=t.id AND f.dates<>'' AND EXISTS ("
                             "SELECT 1 FROM regexp_split_to_table(f.dates, ' ') AS d(value) "
                             "WHERE d.value BETWEEN ? AND ?))")
            else:
                where.append("EXISTS (SELECT 1 FROM task_fts f WHERE f.task_id=t.id AND f.dates<>'' AND EXISTS ("
                             "SELECT 1 FROM json_each('[\"' || replace(f.dates,' ','\",\"') || '\"]') d "
                             "WHERE d.value BETWEEN ? AND ?))")
            args += [lo, hi]
        fts_join, rank_col = "", "0 AS rank"
        # rank_col's own placeholder (Postgres only) must come FIRST in the
        # rows query's param list, since it sits in the SELECT clause before
        # WHERE -- it is never used by the COUNT query, which has no rank_col.
        rank_args: List[Any] = []
        if (text or "").strip():
            fts_join = "JOIN task_fts f ON f.task_id=t.id"
            if self._is_pg:
                tsq = _pg_tsquery(text)
                if tsq:
                    where.append("f.doc @@ to_tsquery('english', ?)")
                    args.append(tsq)
                    # Rank is negated so the existing "ORDER BY rank" (ascending = best
                    # first, bm25's convention) still puts the best match first under
                    # ts_rank_cd, where higher is better.
                    rank_col = "-ts_rank_cd(f.doc, to_tsquery('english', ?)) AS rank"
                    rank_args.append(tsq)
                else:
                    where.append("1=0")
            else:
                where.append("task_fts MATCH ?")
                args.append(_fts_query(text))
                rank_col = "bm25(task_fts, 0.0, 10.0, 6.0, 4.0, 2.0, 2.0, 1.0) AS rank"
        sql_where = " AND ".join(where) if where else "1"
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) AS n FROM tasks t {fts_join} WHERE {sql_where}", args).fetchone()["n"]
            rows = self._db.execute(
                f"SELECT t.id, t.updated_at, t.state, {rank_col} FROM tasks t {fts_join} WHERE {sql_where} "
                f"ORDER BY rank, CASE WHEN t.state IN ('open','waiting_aaron','running') THEN 0 ELSE 1 END, t.updated_at DESC LIMIT ?",
                (*rank_args, *args, max(1, min(int(limit), 50))),
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
        # Whole, always. The ledger is the only record of what happened on a task, and a
        # cut here is permanent: no later reader can recover what was removed.
        text = (text or "").strip()
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
