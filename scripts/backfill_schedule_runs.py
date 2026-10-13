#!/usr/bin/env python3
"""Backfill schedule_runs rows for one schedule's past requests (ones from
before the schedule_runs table existed, or ones a crash never recorded).

Usage:
  python scripts/backfill_schedule_runs.py --db /path/to/gate.db --schedule-id 7 --dry-run
  python scripts/backfill_schedule_runs.py --db /path/to/gate.db --schedule-id 7

--dry-run prints how many rows it would create without writing anything.
Never point this at the live database from this working copy -- the
coordinator runs it there separately.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inkbox_claude.gate.store import Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to the sqlite database file")
    parser.add_argument("--schedule-id", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="count what would be created without writing it")
    args = parser.parse_args()

    store = Store(args.db)
    schedule = store.get_schedule(args.schedule_id)
    if schedule is None:
        print(f"no such schedule: {args.schedule_id}")
        return 1

    if args.dry_run:
        with store._lock:
            existing = {int(r["request_id"]) for r in store._db.execute(
                "SELECT request_id FROM schedule_runs WHERE schedule_id=? AND request_id IS NOT NULL",
                (args.schedule_id,)).fetchall()}
            rows = store._db.execute(
                "SELECT id FROM requests WHERE schedule_id=? AND state IN ('done','failed') ORDER BY id",
                (args.schedule_id,)).fetchall()
        missing = [r for r in rows if int(r["id"]) not in existing]
        print(f"would create {len(missing)} schedule_runs row(s) for schedule {args.schedule_id} "
              f"({schedule.title!r})")
        return 0

    created = store.backfill_schedule_runs(args.schedule_id)
    print(f"created {created} schedule_runs row(s) for schedule {args.schedule_id} ({schedule.title!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
