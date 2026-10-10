#!/usr/bin/env python3
"""Copy a live Blatbot gate.db (sqlite) into a Postgres database/schema with
the same ids, re-runnably and without ever writing to the sqlite file.

Usage:
    .venv/bin/python inkbox_claude/gate/migrate_to_postgres.py \\
        --sqlite /home/aaron/blatbot/gate.db \\
        --pg-dsn postgresql://blatbot:***@10.0.1.15:5432/blatbot_rehearsal

Safety:
  - the sqlite side is opened with the sqlite3 URI `mode=ro` (read-only at
    the driver level, independent of file permissions) -- nothing here can
    write to gate.db;
  - the Postgres side is created fresh by Store(dsn) (SCHEMA_PG, idempotent
    CREATE TABLE IF NOT EXISTS), then every row is upserted with an explicit
    id/PK via ON CONFLICT ... DO UPDATE, so re-running this script against
    the same target is safe and converges rather than duplicating rows;
  - every table in SCHEMA_PG is copied with EXACTLY the same column values
    and ids (ids are preserved, not reassigned, because messages/requests/
    tasks/task_events/schedules are cross-referenced by id via foreign-key-
    shaped columns like task_id/request_id/reply_to); each serial table's
    sequence is advanced past the max copied id afterwards so the next live
    insert does not collide;
  - `doc` (task_fts) and `seq` (task_participants) are Postgres-generated
    columns and are never written directly -- they are produced by Postgres
    itself as a side effect of inserting the other columns.

Report: per-table row counts on both sides, and a sha256 checksum of every
text/TEXT column's concatenated values per table (order-independent: sorted
before hashing) so a silent truncation or encoding change would show up as a
mismatched hash even when the row count matches.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import psycopg

# (table, ordered columns as they exist in BOTH sqlite's SCHEMA and
# Postgres's SCHEMA_PG, primary key columns, serial-id column or None)
TABLES: List[Tuple[str, List[str], List[str], str | None]] = [
    ("threads", ["chat_id", "state", "mode", "meta_json", "updated_at"], ["chat_id"], None),
    ("messages", ["id", "chat_id", "kind", "mode", "text", "created_at", "reply_to", "role"], ["id"], "id"),
    ("mail_ids", ["message_id", "root"], ["message_id"], None),
    ("thread_links", ["thread_key", "chat_id", "updated_at"], ["thread_key", "chat_id"], None),
    ("requests", ["id", "chat_id", "sender", "sender_name", "mode", "subject", "original_message",
                  "summary", "scopes_json", "prompt", "prompt_sha256", "state", "revision",
                  "status_json", "raw_output", "created_at", "updated_at", "task_id", "inbound_id",
                  "schedule_id", "schedule_kind"], ["id"], "id"),
    ("tasks", ["id", "key", "display", "title", "summary", "state", "created_at", "updated_at"],
     ["id"], "id"),
    ("task_participants", ["task_id", "key", "display", "person_id"], ["task_id", "key"], None),
    ("people", ["person_id", "key", "display", "updated_at"], ["person_id", "key"], None),
    ("task_fts", ["task_id", "title", "summary", "people", "events", "requests", "dates"],
     [], None),  # no PK/unique in either schema; cleared and reinserted whole
    ("task_events", ["id", "task_id", "kind", "chat_id", "request_id", "text", "created_at"],
     ["id"], "id"),
    ("schedules", ["id", "chat_id", "task_id", "title", "prompt", "prompt_sha256", "kind", "cron",
                   "run_at", "timezone", "scopes_json", "report_mode", "state", "next_run",
                   "last_run", "last_request_id", "run_count", "max_runs", "deadline", "notes",
                   "revision", "consecutive_failures", "last_outcome", "created_at", "updated_at"],
     ["id"], "id"),
    ("persons", ["id", "display", "role", "scopes", "note", "merged_into", "created_at", "updated_at"],
     ["id"], None),
    ("contacts", ["id", "person_id", "kind", "value", "raw_value", "source", "last_seen",
                  "created_at", "updated_at"], ["id"], None),
    ("roles", ["name", "scopes", "note", "updated_at"], ["name"], None),
    ("trust", ["key", "person", "role", "scopes", "note", "updated_at"], ["key"], None),
    ("synced_people", ["key", "person", "channels", "last_seen"], ["key"], None),
    ("settings", ["name", "value", "updated_at"], ["name"], None),
    ("contact_kinds", ["kind", "label", "hint", "addable", "sort_order"], ["kind"], None),
]


def _sqlite_ro(path: str) -> sqlite3.Connection:
    uri = f"file:{Path(path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists_sqlite(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (table,)
    ).fetchone()
    return row is not None


def _checksum(rows: Sequence[sqlite3.Row], columns: Sequence[str]) -> str:
    """Order-independent hash of every text-ish column's values, so a
    truncation or mis-encoding shows up even when the row count matches."""
    parts = []
    for r in rows:
        parts.append("\x1f".join(
            "" if r[c] is None else str(r[c]) for c in columns
        ))
    parts.sort()
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "surrogatepass"))
        h.update(b"\x1e")
    return h.hexdigest()


def migrate(sqlite_path: str, pg_dsn: str) -> int:
    # Import here so this script can also be run standalone with just
    # psycopg installed, without requiring the rest of the package on sys.path
    # for a --check-only run against an already-migrated target.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from inkbox_claude.gate.store import Store  # noqa: E402  (ensures SCHEMA_PG is applied)

    Store(pg_dsn)  # idempotent: creates SCHEMA_PG + seeds contact_kinds if empty

    src = _sqlite_ro(sqlite_path)
    dst = psycopg.connect(pg_dsn, autocommit=False)

    print(f"{'table':<20} {'sqlite_rows':>12} {'pg_rows_before':>15}", flush=True)
    results = []
    with dst.cursor() as cur:
        for table, columns, pk, serial_col in TABLES:
            if not _table_exists_sqlite(src, table):
                print(f"{table:<20} {'(missing in sqlite)':>12}")
                continue
            order_col = "rowid" if table in ("task_fts",) else (pk[0] if pk else "rowid")
            try:
                rows = src.execute(f"SELECT * FROM {table} ORDER BY {order_col}").fetchall()
            except sqlite3.OperationalError:
                rows = src.execute(f"SELECT * FROM {table}").fetchall()
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            pg_before = cur.fetchone()[0]
            print(f"{table:<20} {len(rows):>12} {pg_before:>15}", flush=True)

            if table == "task_fts":
                # No stable PK on either side; this table is fully derived
                # from tasks/task_participants/task_events/requests by
                # reindex_task(), so the migration clears and reinserts it
                # wholesale rather than row-by-row upserting.
                cur.execute("DELETE FROM task_fts")
                for r in rows:
                    cur.execute(
                        "INSERT INTO task_fts(task_id,title,summary,people,events,requests,dates) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        tuple(r[c] for c in columns),
                    )
            else:
                placeholders = ",".join(["%s"] * len(columns))
                collist = ",".join(columns)
                if pk:
                    pkcols = ",".join(pk)
                    updates = ",".join(f"{c}=EXCLUDED.{c}" for c in columns if c not in pk)
                    conflict = f"ON CONFLICT ({pkcols}) DO UPDATE SET {updates}" if updates else \
                               f"ON CONFLICT ({pkcols}) DO NOTHING"
                else:
                    conflict = ""
                sql = f"INSERT INTO {table}({collist}) VALUES ({placeholders}) {conflict}"
                for r in rows:
                    cur.execute(sql, tuple(r[c] for c in columns))

            if serial_col:
                cur.execute(f"SELECT COALESCE(MAX({serial_col}), 0) FROM {table}")
                max_id = cur.fetchone()[0]
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence(%s, %s), %s, true)",
                    (table, serial_col, max(max_id, 1)),
                )

            cur.execute(f"SELECT COUNT(*) FROM {table}")
            pg_after = cur.fetchone()[0]
            checksum = _checksum(rows, [c for c in columns if c not in ("addable", "revision",
                                                                         "run_count", "max_runs",
                                                                         "consecutive_failures")])
            results.append((table, len(rows), pg_after, checksum))

    dst.commit()
    src.close()
    dst.close()

    print()
    print(f"{'table':<20} {'sqlite_rows':>12} {'pg_rows_after':>14} {'text_checksum':>18}")
    ok = True
    for table, n_src, n_dst, checksum in results:
        flag = "" if n_src == n_dst else "  <-- MISMATCH"
        if n_src != n_dst:
            ok = False
        print(f"{table:<20} {n_src:>12} {n_dst:>14} {checksum[:16]:>18}{flag}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqlite", required=True, help="path to the gate.db to copy FROM (read-only)")
    ap.add_argument("--pg-dsn", required=True, help="postgresql:// URL to copy INTO")
    args = ap.parse_args()
    return migrate(args.sqlite, args.pg_dsn)


if __name__ == "__main__":
    raise SystemExit(main())
