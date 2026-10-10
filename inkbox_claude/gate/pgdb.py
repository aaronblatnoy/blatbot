"""Thin Postgres compatibility shim so Store (built around sqlite3's shape)
can run against Postgres via psycopg 3 with minimal changes to store.py's
call sites.

Design: a single guarded connection (matches store.py's existing
threading.Lock model -- no pool, no behaviour change there), a hybrid dict
row factory so `row["col"]`, `row[0]`, `row.keys()`, and `dict(row)` all
behave like sqlite3.Row, a cursor wrapper that fakes sqlite3's `lastrowid` via an
auto-appended `RETURNING id` for the handful of tables that actually have a
serial id column, and a small SQL-text translator for the few sqlite-only
constructs store.py's call sites use directly (`?` placeholders,
`INSERT OR IGNORE`, `executescript`).

Everything this module does NOT handle generically -- json_extract/json_each,
FTS5 MATCH/bm25, scalar multi-arg MAX(), PRAGMA table_info, ORDER BY rowid --
is handled by explicit `if self._is_pg:` branches in store.py itself. There
are few enough of those call sites that naming them directly is safer than a
blanket regex SQL rewriter would be.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

try:
    import psycopg
except ImportError:  # psycopg not installed; only sqlite targets can run
    psycopg = None


class _HybridRow(dict):
    """A row that behaves like sqlite3.Row: `row["col"]` AND `row[0]` (by
    position) both work, plus dict(row)/row.keys() since it IS a dict.
    store.py itself only indexes rows by column name, but at least one test
    reads a row positionally (`fetchone()[0]`), so plain dict_row rows
    (string keys only) are not quite a drop-in match for sqlite3.Row."""

    __slots__ = ("_values",)

    def __init__(self, keys, values):
        super().__init__(zip(keys, values))
        self._values = values

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


def _hybrid_row_factory(cursor):
    desc = cursor.description
    keys = [c.name for c in desc] if desc else []

    def make_row(values):
        return _HybridRow(keys, values)

    return make_row


def is_postgres_target(path: str) -> bool:
    return isinstance(path, str) and (
        path.startswith("postgresql://") or path.startswith("postgres://")
    )


# Tables whose `id` column is a serial/identity primary key -- the only ones
# where sqlite3's cur.lastrowid was ever read by store.py.
SERIAL_ID_TABLES = {"messages", "requests", "tasks", "task_events", "schedules"}

_OR_IGNORE_RE = re.compile(r"(?i)\bINSERT\s+OR\s+IGNORE\s+INTO\b")
_INSERT_TABLE_RE = re.compile(r"(?is)^\s*INSERT\s+(?:OR\s+IGNORE\s+)?INTO\s+(\w+)")


def _translate_sql(sql: str) -> str:
    """Rewrite the sqlite-flavoured bits of a single SQL statement that are
    safe to handle generically for every caller."""
    had_or_ignore = bool(_OR_IGNORE_RE.search(sql))
    sql = _OR_IGNORE_RE.sub("INSERT INTO", sql)
    # '?' -> '%s'. No SQL text in store.py embeds a literal '?' inside a
    # string literal, so a blanket replace is safe for this codebase.
    sql = sql.replace("?", "%s")
    if had_or_ignore and "ON CONFLICT" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    return sql


class PGCursor:
    """Wraps a psycopg cursor to fake sqlite3.Cursor's `lastrowid` and
    forward `rowcount` (psycopg already has it; this just keeps the
    attribute name store.py expects)."""

    __slots__ = ("_cur", "lastrowid")

    def __init__(self, cur):
        self._cur = cur
        self.lastrowid: Optional[int] = None

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __iter__(self):
        return iter(self._cur)


class PGConnection:
    """A single guarded psycopg connection, shaped like sqlite3.Connection for
    the subset of the API store.py calls: execute, executescript, commit,
    close. Not thread-safe on its own -- callers hold Store._lock exactly as
    they do today around the sqlite3 connection, so this never needs its own
    lock or a pool."""

    def __init__(self, dsn: str):
        if psycopg is None:
            raise RuntimeError(
                "psycopg is not installed; cannot open a postgresql:// Store target "
                "(pip install 'psycopg[binary]' in this venv)"
            )
        self._conn = psycopg.connect(dsn, autocommit=False, row_factory=_hybrid_row_factory)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> PGCursor:
        table_match = _INSERT_TABLE_RE.match(sql)
        auto_returning = bool(
            table_match
            and table_match.group(1).lower() in SERIAL_ID_TABLES
            and "RETURNING" not in sql.upper()
        )
        translated = _translate_sql(sql)
        if auto_returning:
            translated = translated.rstrip().rstrip(";") + " RETURNING id"
        cur = self._conn.cursor()
        cur.execute(translated, tuple(params) if params else None)
        wrapped = PGCursor(cur)
        if auto_returning:
            row = cur.fetchone()
            wrapped.lastrowid = row["id"] if row else None
        return wrapped

    def executescript(self, script: str) -> None:
        """psycopg has no sqlite-style executescript; split on ';' at
        statement boundaries and run each non-empty statement. store.py's
        scripts are plain DDL/DML with no semicolons inside string
        literals, so a naive split is safe here."""
        with self._conn.cursor() as cur:
            for stmt in script.split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
