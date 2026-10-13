"""store.backfill_schedule_runs: filling in schedule_runs rows for a schedule's
old requests that predate the schedule_runs table, from the request rows
themselves."""

from __future__ import annotations

from inkbox_claude.gate.store import Store


def _make_schedule(store, task_id):
    return store.create_schedule(
        chat_id="owner", task_id=task_id, title="Old watch", prompt="check a page",
        kind="every", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="every", cadence={"minutes": 60}, origin_mode="imessage", origin_chat_id="owner",
    )


def test_backfill_creates_rows_from_old_requests(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    task = store.create_task("backfill test")
    schedule = _make_schedule(store, task["id"])
    done = store.create_request(
        chat_id="owner", sender="owner", sender_name="owner", mode="imessage", subject="",
        original_message="", summary="", scopes=["web"], prompt="check a page", state="pending",
        task_id=task["id"], schedule_id=schedule.id)
    store.set_state(done.id, "done", status={"result_line": "total is $12.00"})
    failed = store.create_request(
        chat_id="owner", sender="owner", sender_name="owner", mode="imessage", subject="",
        original_message="", summary="", scopes=["web"], prompt="check a page", state="pending",
        task_id=task["id"], schedule_id=schedule.id)
    store.set_state(failed.id, "failed", status={"error": "the page did not load"})

    created = store.backfill_schedule_runs(schedule.id)
    assert created == 2

    runs = {r["request_id"]: r for r in store.schedule_runs(schedule.id)}
    assert runs[done.id]["outcome"] == "complete"
    assert runs[done.id]["result_line"] == "total is $12.00"
    assert runs[failed.id]["outcome"] == "failed"
    assert runs[failed.id]["result_line"] == "the page did not load"


def test_backfill_is_idempotent_and_skips_requests_already_backed_by_a_run(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    task = store.create_task("backfill test")
    schedule = _make_schedule(store, task["id"])
    req = store.create_request(
        chat_id="owner", sender="owner", sender_name="owner", mode="imessage", subject="",
        original_message="", summary="", scopes=["web"], prompt="check a page", state="pending",
        task_id=task["id"], schedule_id=schedule.id)
    store.set_state(req.id, "done", status={"result_line": "ok"})

    # A real run already recorded this request: backfill must not double it.
    run_id = store.create_schedule_run(schedule.id, req.id, 1000.0)
    store.finish_schedule_run(run_id, finished_at=1001.0, complete=True, outcome="complete",
                              result_line="ok", final_status="ok", denied=[], note="", changed=False,
                              cost_usd=None)

    assert store.backfill_schedule_runs(schedule.id) == 0
    assert len(store.schedule_runs(schedule.id)) == 1

    # A pending (never-finished) request is left alone -- only done/failed count.
    pending = store.create_request(
        chat_id="owner", sender="owner", sender_name="owner", mode="imessage", subject="",
        original_message="", summary="", scopes=["web"], prompt="check a page", state="pending",
        task_id=task["id"], schedule_id=schedule.id)
    assert store.backfill_schedule_runs(schedule.id) == 0
    assert len(store.schedule_runs(schedule.id)) == 1

    # Running it again after nothing changed still creates nothing.
    assert store.backfill_schedule_runs(schedule.id) == 0
