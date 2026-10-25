"""Read-only payload measurement. No gateway, send tools or execution path is loaded.

Run with the service environment and --chat-id for a stored conversation.
The only external call is the ledger relevance judgment to TypeSafe.
"""
import argparse
import asyncio
import json
import logging
import os
import time
import threading
import sqlite3
from inkbox_claude.gate.store import Store
from inkbox_claude.gate.pgdb import PGConnection, is_postgres_target
from inkbox_claude.gate.taskpick import TaskPicker
from inkbox_claude.gate.ledger import render_index


async def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chat-id', required=True)
    parser.add_argument('--log-file')
    parser.add_argument('--contains', default='')
    parser.add_argument('--latest-task-message', action='store_true')
    args = parser.parse_args()
    target = os.environ.get('GATE_PG_URL') or os.environ['GATE_DB_PATH']
    # Bypass Store's boot migrations and role reconciliation: only read existing data.
    store = Store.__new__(Store)
    store._is_pg = is_postgres_target(target)
    store._lock = threading.Lock()
    if store._is_pg:
        store._db = PGConnection(target)
        store._db.execute('SET TRANSACTION READ ONLY')
    else:
        store._db = sqlite3.connect('file:' + target + '?mode=ro', uri=True)
        store._db.row_factory = sqlite3.Row
    history = store.history(args.chat_id, limit=100)
    latest = next(row for row in reversed(history) if row['kind'] == 'inbound' and args.contains.lower() in row['text'].lower())
    prior = history[:history.index(latest)]
    before = store.task_memory(args.chat_id, is_approver=True)
    start = time.monotonic()
    result = store.ledger_index(is_approver=True)
    index = render_index(result['tasks'])
    if args.latest_task_message:
        task_id = result['tasks'][0]['id']
        row = store._db.execute("SELECT text FROM task_events WHERE task_id=? AND kind='inbound' ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
        latest = {'text': row['text']}
        prior = []
    pick = await TaskPicker().judge_ledger_tree(message=latest['text'], history=prior, entries=result['tasks'])
    if pick['ids'] is None:
        raise RuntimeError('Judgment unavailable; no measurement produced')
    after = '\n\n'.join(store.task_memory_for_task(tid) for tid in pick['ids']) or '(no tasks on record)'
    metrics = {'before_chars': len(before), 'index_tasks': len(result['tasks']), 'index_chars': len(index),
               'picked': [f'T{tid}' for tid in pick['ids']], 'after_chars': len(after),
               'dropped_stale': result['dropped'], 'timings': pick['timings'],
               'total_seconds': round(time.monotonic() - start, 3)}
    print(json.dumps(metrics))
    if args.log_file:
        log = logging.getLogger('ledger_replay')
        log.setLevel(logging.INFO)
        handler = logging.FileHandler(args.log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        log.addHandler(handler)
        log.info('read-only stored-message replay: %s', json.dumps(metrics))


if __name__ == '__main__':
    asyncio.run(main())
