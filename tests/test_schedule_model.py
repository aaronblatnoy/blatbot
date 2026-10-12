"""Tests for the schedule data contract: the new schedules columns, schedule_runs,
messages.schedule_id, and the generic kind/state migration. Uses only throwaway
sqlite databases -- no live credentials, no real gate.db, no network."""

from __future__ import annotations

import json
import time

from inkbox_claude.gate.store import Store


def make_store(tmp_path, name="gate.db"):
    return Store(str(tmp_path / name))


def make_schedule(store, **overrides):
    kwargs = dict(
        chat_id="chat-1", task_id=store.create_task("Some watch")["id"],
        title="Team total watch", prompt="check the made-up team page",
        kind="cron", scopes=["web_read"], timezone="America/New_York",
        report_mode="always",
    )
    kwargs.update(overrides)
    return store.create_schedule(**kwargs)


# ---------------------------------------------------------------------------
# New columns round-trip
# ---------------------------------------------------------------------------

def test_new_columns_round_trip_on_create(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(
        store, cadence_kind="every", cadence={"minutes": 240}, report_every_time=True,
        until=1999999999.0, origin_mode="telegram", origin_chat_id="tg-42",
    )
    assert sched.cadence_kind == "every"
    assert sched.cadence == {"minutes": 240}
    assert sched.report_every_time is True
    assert sched.until == 1999999999.0
    assert sched.origin_mode == "telegram"
    assert sched.origin_chat_id == "tg-42"
    assert sched.last_note == ""

    fetched = store.get_schedule(sched.id)
    assert fetched.cadence_kind == "every"
    assert fetched.cadence == {"minutes": 240}
    assert fetched.report_every_time is True
    assert fetched.until == 1999999999.0
    assert fetched.origin_mode == "telegram"
    assert fetched.origin_chat_id == "tg-42"


def test_create_schedule_defaults_origin_chat_id_to_chat_id(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    assert sched.origin_chat_id == "chat-1"
    assert sched.origin_mode == ""


def test_new_columns_round_trip_on_update(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    updated = store.update_schedule(
        sched.id, cadence_kind="daily", cadence={"time": "08:00"},
        report_every_time=False, until=None, last_note="total was $4,027.39",
    )
    assert updated.cadence_kind == "daily"
    assert updated.cadence == {"time": "08:00"}
    assert updated.report_every_time is False
    assert updated.until is None
    assert updated.last_note == "total was $4,027.39"


def test_existing_create_update_parameters_still_work_untouched(tmp_path):
    """Old callers that never mention the new fields must see unchanged behaviour."""
    store = make_store(tmp_path)
    sched = make_schedule(
        store, kind="once", cron="", run_at=1700000000.0, next_run=1700000000.0,
        max_runs=5, deadline=1800000000.0,
    )
    assert sched.kind == "once"
    assert sched.run_at == 1700000000.0
    assert sched.next_run == 1700000000.0
    assert sched.max_runs == 5
    assert sched.deadline == 1800000000.0
    assert sched.cadence_kind == ""
    assert sched.cadence == {}

    updated = store.update_schedule(sched.id, state="active", run_count=3)
    assert updated.state == "active"
    assert updated.run_count == 3


def test_update_schedule_rejects_unknown_column(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    try:
        store.update_schedule(sched.id, not_a_real_column="x")
    except ValueError as e:
        assert "not_a_real_column" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ended_state_is_accepted(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    updated = store.update_schedule(sched.id, state="ended", last_outcome="stopped by owner")
    assert updated.state == "ended"
    assert updated.last_outcome == "stopped by owner"


# ---------------------------------------------------------------------------
# schedule_runs: create, finish, list newest first, last
# ---------------------------------------------------------------------------

def test_schedule_run_create_and_finish_round_trip(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    started = time.time()
    run_id = store.create_schedule_run(sched.id, request_id=None, started_at=started)
    assert isinstance(run_id, int) and run_id > 0

    long_status = "x" * 5000  # final_status must never be cut, however long
    store.finish_schedule_run(
        run_id, finished_at=started + 12.0, complete=True, outcome="complete",
        result_line="total went from $4,027.39 to $4,922.45.",
        final_status=long_status, denied=[{"tool": "mcp__x__write", "reason": "scope"}],
        note="watch for a drop next time", changed=True, cost_usd=0.0423,
    )
    run = store.last_schedule_run(sched.id)
    assert run["id"] == run_id
    assert run["complete"] is True
    assert run["outcome"] == "complete"
    assert run["result_line"] == "total went from $4,027.39 to $4,922.45."
    assert run["final_status"] == long_status
    assert len(run["final_status"]) == 5000
    assert run["denied"] == [{"tool": "mcp__x__write", "reason": "scope"}]
    assert run["note"] == "watch for a drop next time"
    assert run["changed"] is True
    assert run["cost_usd"] == 0.0423


def test_schedule_runs_list_newest_first_and_limit(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    ids = []
    for i in range(3):
        rid = store.create_schedule_run(sched.id, request_id=None, started_at=1000.0 + i)
        store.finish_schedule_run(
            rid, finished_at=1001.0 + i, complete=True, outcome="complete",
            result_line=f"run {i}", final_status="", denied=[], note="", changed=False,
            cost_usd=None,
        )
        ids.append(rid)

    runs = store.schedule_runs(sched.id)
    assert [r["id"] for r in runs] == list(reversed(ids))

    limited = store.schedule_runs(sched.id, limit=2)
    assert [r["id"] for r in limited] == list(reversed(ids))[:2]


def test_last_schedule_run_none_when_no_runs(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    assert store.last_schedule_run(sched.id) is None
    assert store.schedule_runs(sched.id) == []


def test_schedule_run_cost_usd_nullable(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    rid = store.create_schedule_run(sched.id, request_id=None, started_at=time.time())
    store.finish_schedule_run(
        rid, finished_at=time.time(), complete=False, outcome="failed",
        result_line="", final_status="timed out", denied=None, note="", changed=False,
        cost_usd=None,
    )
    run = store.last_schedule_run(sched.id)
    assert run["cost_usd"] is None
    assert run["denied"] == []


# ---------------------------------------------------------------------------
# Message tagging and schedule_for_message
# ---------------------------------------------------------------------------

def test_message_schedule_id_round_trip_and_lookup(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store)
    msg_id = store.add_message("chat-1", "outbound", "Team total watch: no change.",
                               schedule_id=sched.id)
    found = store.schedule_for_message(msg_id)
    assert found is not None
    assert found.id == sched.id


def test_schedule_for_message_none_when_untagged(tmp_path):
    store = make_store(tmp_path)
    msg_id = store.add_message("chat-1", "outbound", "unrelated notice")
    assert store.schedule_for_message(msg_id) is None


def test_schedule_for_message_none_for_unknown_message(tmp_path):
    store = make_store(tmp_path)
    assert store.schedule_for_message(999999) is None


def test_add_message_existing_callers_unaffected(tmp_path):
    """reply_to/role still work exactly as before when schedule_id is not passed."""
    store = make_store(tmp_path)
    inbound_id = store.add_message("chat-1", "inbound", "ping")
    out_id = store.add_message("chat-1", "outbound", "pong", reply_to=inbound_id, role="answer")
    assert store.responses_to(inbound_id) == ["answer"]
    assert store.schedule_for_message(out_id) is None


# ---------------------------------------------------------------------------
# open_schedule_proposal
# ---------------------------------------------------------------------------

def test_open_schedule_proposal_returns_the_proposed_one(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store, chat_id="chat-2")
    assert sched.state == "proposed"
    found = store.open_schedule_proposal("chat-2")
    assert found is not None
    assert found.id == sched.id


def test_open_schedule_proposal_none_once_activated(tmp_path):
    store = make_store(tmp_path)
    sched = make_schedule(store, chat_id="chat-3")
    store.update_schedule(sched.id, state="active")
    assert store.open_schedule_proposal("chat-3") is None


def test_open_schedule_proposal_none_for_other_thread(tmp_path):
    store = make_store(tmp_path)
    make_schedule(store, chat_id="chat-4")
    assert store.open_schedule_proposal("chat-5") is None


def test_open_schedule_proposal_picks_newest_when_several(tmp_path):
    store = make_store(tmp_path)
    make_schedule(store, chat_id="chat-6", title="older one")
    newer = make_schedule(store, chat_id="chat-6", title="newer one")
    found = store.open_schedule_proposal("chat-6")
    assert found.id == newer.id


# ---------------------------------------------------------------------------
# Migration: old kind/state rows -> the new cadence model
# ---------------------------------------------------------------------------

def _seed_legacy_db(path, rows):
    """Build a throwaway sqlite file shaped like the OLD (pre-cadence) schedules
    table -- none of the new columns exist yet -- and insert the given legacy
    rows. Opening it with Store(path) then exercises the same ALTER-column-if-
    missing + migration path a real old gate.db would go through. Mirrors how
    the people-model migration's own tests seed legacy fixture tables directly
    (see test_people_model.py's test_migration_from_legacy_fixture_tables)."""
    import sqlite3
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE schedules (
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
        CREATE TABLE tasks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          key TEXT NOT NULL DEFAULT '', display TEXT NOT NULL DEFAULT '',
          title TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'open',
          created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
    """)
    now = time.time()
    ids = []
    for row in rows:
        conn.execute(
            "INSERT INTO tasks(key,display,title,state,created_at,updated_at) "
            "VALUES('','','legacy watch','open',?,?)", (now, now),
        )
        task_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        cur = conn.execute(
            "INSERT INTO schedules(chat_id,task_id,title,prompt,prompt_sha256,kind,cron,run_at,"
            "timezone,scopes_json,report_mode,state,notes,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("chat-legacy", task_id, "legacy watch", "do the thing", "sha",
             row["kind"], row.get("cron") or None, row.get("run_at"), "America/New_York", "[]",
             row.get("report_mode", "always"), row["state"], row.get("notes", "[]"), now, now),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    return ids


def test_migration_maps_cron_kind_directly(tmp_path):
    path = tmp_path / "gate.db"
    [sid] = _seed_legacy_db(path, [{"kind": "cron", "state": "active", "cron": "0 */4 * * *"}])
    store = Store(str(path))
    sched = store.get_schedule(sid)
    assert sched.cadence_kind == "cron"
    assert sched.cadence == {"expr": "0 */4 * * *"}
    assert sched.state == "active"  # active is not one of the states the migration rewrites


def test_migration_maps_once_kind_directly(tmp_path):
    path = tmp_path / "gate.db"
    [sid] = _seed_legacy_db(path, [{"kind": "once", "state": "done", "run_at": 1700000000.0}])
    store = Store(str(path))
    sched = store.get_schedule(sid)
    assert sched.cadence_kind == "once"
    assert sched.cadence.get("at")
    assert sched.state == "ended"


def test_migration_continue_with_no_cadence_pauses_with_reason(tmp_path):
    path = tmp_path / "gate.db"
    [sid] = _seed_legacy_db(path, [{"kind": "continue", "state": "active"}])
    store = Store(str(path))
    sched = store.get_schedule(sid)
    assert sched.cadence_kind == ""
    assert sched.state == "paused"
    assert sched.last_outcome == "needs a cadence"


def test_migration_maps_old_states_to_ended(tmp_path):
    path = tmp_path / "gate.db"
    old_states = ("done", "failed", "rejected")
    ids = _seed_legacy_db(path, [{"kind": "cron", "state": s, "cron": "0 9 * * *"} for s in old_states])
    store = Store(str(path))
    for sid, old_state in zip(ids, old_states):
        assert store.get_schedule(sid).state == "ended", f"{old_state} should migrate to ended"


def test_migration_report_mode_always_maps_to_report_every_time(tmp_path):
    path = tmp_path / "gate.db"
    sid_always, sid_changed = _seed_legacy_db(path, [
        {"kind": "cron", "state": "active", "cron": "0 9 * * *", "report_mode": "always"},
        {"kind": "cron", "state": "active", "cron": "0 9 * * *", "report_mode": "changed"},
    ])
    store = Store(str(path))
    assert store.get_schedule(sid_always).report_every_time is True
    assert store.get_schedule(sid_changed).report_every_time is False


def test_migration_last_note_takes_last_model_note(tmp_path):
    path = tmp_path / "gate.db"
    notes = json.dumps([
        {"kind": "model", "text": "first note"},
        {"kind": "operational", "text": "ignored, not a model note"},
        {"kind": "model", "text": "most recent model note"},
    ])
    [sid] = _seed_legacy_db(path, [{"kind": "cron", "state": "active", "cron": "0 9 * * *", "notes": notes}])
    store = Store(str(path))
    assert store.get_schedule(sid).last_note == "most recent model note"


def test_migration_running_twice_changes_nothing(tmp_path):
    """Store.__init__ runs the migration once on open; opening the same file a
    second time must be a pure no-op (same convention as the people-model
    migration's own idempotency test)."""
    path = tmp_path / "gate.db"
    [sid] = _seed_legacy_db(path, [{"kind": "continue", "state": "active"}])
    store1 = Store(str(path))
    first = store1.get_schedule(sid)
    store2 = Store(str(path))
    second = store2.get_schedule(sid)
    assert first.state == second.state == "paused"
    assert first.last_outcome == second.last_outcome == "needs a cadence"
    assert first.cadence_kind == second.cadence_kind == ""
    assert first.updated_at == second.updated_at  # untouched the second time: no-op, not a rewrite
