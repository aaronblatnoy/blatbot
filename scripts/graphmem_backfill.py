#!/usr/bin/env python3
"""Send every message already in the gate store to the Graphiti memory graph,
oldest first, through the same episode shape live messages use.

Resumable: the id of the last message sent is kept in --state, so a second run
continues where the first stopped and never sends a message twice.

  set -a; . ~/blatbot/.env; set +a
  .venv/bin/python scripts/graphmem_backfill.py            # everything not yet sent
  .venv/bin/python scripts/graphmem_backfill.py --limit 20 # a first small batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inkbox_claude.gate import graphmem  # noqa: E402
from inkbox_claude.gate.store import Store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("GATE_PG_URL") or os.getenv("GATE_DB_PATH") or "")
    ap.add_argument("--state", default=str(Path.home() / "blatbot" / "graphmem_backfill.state"))
    ap.add_argument("--until-id", type=int, default=0, help="stop after this message id (0 = no bound)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.2, help="seconds between messages")
    ap.add_argument("--max-queued", type=int, default=2, help="wait while the graph has more than this many queued")
    args = ap.parse_args()
    if not graphmem.enabled():
        print("GRAPHITI_URL is not set", file=sys.stderr)
        return 2
    if not args.db:
        print("no store: pass --db or set GATE_PG_URL", file=sys.stderr)
        return 2
    state = Path(args.state)
    last = int(state.read_text().strip() or 0) if state.exists() else 0
    store = Store(args.db)
    sql = "SELECT id,chat_id,kind,mode,text,created_at FROM messages WHERE id>?"
    params: list = [last]
    if args.until_id:
        sql += " AND id<=?"
        params.append(args.until_id)
    sql += " ORDER BY id"
    with store._lock:
        rows = [dict(r) for r in store._db.execute(sql, tuple(params)).fetchall()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} message(s) to send, starting after id {last}", flush=True)
    sent = 0
    speakers: dict = {}
    for r in rows:
        if str(r["text"] or "").strip():
            # Keep the graph's queue short: a live message must never wait behind the
            # whole history, and a restart of the service loses at most a few episodes.
            while True:
                try:
                    with urllib.request.urlopen(graphmem.base_url() + "/healthcheck", timeout=10) as resp:
                        if int(json.loads(resp.read()).get("queued") or 0) <= args.max_queued:
                            break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(3)
            cid = r["chat_id"]
            if cid not in speakers:
                speakers[cid] = store.memory_speaker(cid)
            body = graphmem.episode(cid, r["kind"], r["text"], r["mode"] or "",
                                    created_at=r["created_at"], message_id=r["id"],
                                    speaker=speakers[cid] if r["kind"] == "inbound" else "")
            for attempt in range(5):
                try:
                    graphmem._post("/messages", body, graphmem.WRITE_TIMEOUT)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 4:
                        print(f"stopped at id {r['id']}: {exc}", file=sys.stderr)
                        return 1
                    time.sleep(2 * (attempt + 1))
            sent += 1
            time.sleep(args.pause)
        state.write_text(str(r["id"]))
        if sent and sent % 100 == 0:
            print(f"  sent {sent} (through id {r['id']})", flush=True)
    print(f"done: sent {sent}, last id {rows[-1]['id'] if rows else last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
