"""Tests for GET /console/api/overview (api.overview): the Home page's single
source of stuck / running / done-today / coming-up rows, built with made-up
rows against the real store (not faked)."""

from __future__ import annotations

import time

import json as _json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from inkbox_claude.console import api
from inkbox_claude.console import routes as console_routes
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.store import Store


def make_store(tmp_path):
    return Store(str(tmp_path / "overview.db"))


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.schedules = sched_mod.ScheduleService(self)
        self.notices = []

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule.id, text))

    async def execute(self, req):
        self.store.set_state(req.id, "running")


class FakeGateway:
    def __init__(self, manager):
        self.sessions = manager


def _make_schedule(store, *, title="Team total watch", prompt="check the team page",
                   scopes=None, cadence_kind="every", cadence=None, state="active",
                   report_every_time=False, timezone="America/New_York", **extra):
    task_id = store.create_task(title)["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title=title, prompt=prompt, kind=cadence_kind,
        scopes=list(scopes or ["web"]), timezone=timezone, report_mode="changed",
        cadence_kind=cadence_kind, cadence=cadence or {"minutes": 240},
        report_every_time=report_every_time, origin_mode="console", origin_chat_id="owner",
    )
    if state != "proposed":
        schedule = store.update_schedule(schedule.id, state=state, next_run=time.time() + 3600, **extra)
    return schedule


def _add_run(store, schedule_id, *, outcome="complete", result_line="done",
             final_status="", denied=None, note="", changed=True, cost_usd=0.01,
             finished_at=None):
    started = (finished_at or time.time()) - 60
    run_id = store.create_schedule_run(schedule_id, None, started)
    store.finish_schedule_run(
        run_id, finished_at=finished_at or (started + 55), complete=(outcome == "complete"),
        outcome=outcome, result_line=result_line, final_status=final_status or result_line,
        denied=denied or [], note=note, changed=changed, cost_usd=cost_usd,
    )
    return run_id


def _make_request(store, *, title_task=None, summary="Approve the donor mailing list.",
                   original_message="Can you approve the donor mailing list?",
                   state="pending", sender_name="Jordan", schedule_id=None,
                   status=None):
    task_id = title_task or store.create_task(summary)["id"]
    request = store.create_request(
        chat_id="owner", sender="jordan@example.test", sender_name=sender_name, mode="console",
        subject="", original_message=original_message, summary=summary, scopes=[], prompt=summary,
        state=state, task_id=task_id, schedule_id=schedule_id,
    )
    if status is not None:
        store._db.execute("UPDATE requests SET status_json=? WHERE id=?",
                          (_json.dumps(status), request.id))
        store._db.commit()
    return store.get_request(request.id)


def _backdate(store, request_id, *, created_at=None, updated_at=None):
    created_at = created_at if created_at is not None else time.time()
    updated_at = updated_at if updated_at is not None else created_at
    store._db.execute(
        "UPDATE requests SET created_at=?, updated_at=? WHERE id=?",
        (created_at, updated_at, request_id),
    )
    store._db.commit()


# ---------------------------------------------------------------------------
# "Nothing is stuck" only when all three stuck sources are empty.
# ---------------------------------------------------------------------------

def test_nothing_stuck_when_no_schedule_request_or_proposal_is_waiting(tmp_path):
    store = make_store(tmp_path)
    _make_schedule(store, title="Fine schedule", state="active")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert result["stuck"] == []
    assert result["counts"]["stuck"] == 0


def test_a_pending_request_alone_makes_it_stuck(tmp_path):
    store = make_store(tmp_path)
    _make_request(store, summary="Approve the donor mailing list.")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert len(result["stuck"]) == 1
    row = result["stuck"][0]
    assert row["kind"] == "approval"
    assert "donor mailing list" in row["sentence"]
    actions = {a["action"] for a in row["actions"]}
    assert actions == {"approve", "decline"}
    assert row["actions"][0]["target"]["endpoint"] == "requests/decide"


# ---------------------------------------------------------------------------
# Each kind of stuck item, with the right sentence and actions.
# ---------------------------------------------------------------------------

def test_paused_schedule_with_a_reason_is_stuck_with_resume_and_stop(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Weekly backup check", cadence_kind="weekly",
                              cadence={"day": "mon", "time": "06:00"}, state="paused",
                              last_outcome="it is not allowed to look up and edit contacts")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    rows = [r for r in result["stuck"] if r["id"] == schedule.id]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "schedule"
    assert row["sentence"] == "Weekly backup check: it is not allowed to look up and edit contacts."
    actions = {a["action"] for a in row["actions"]}
    assert "resume" in actions and "stop" in actions


def test_paused_schedule_with_no_cadence_gets_edit_timing_action(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Half-built schedule", state="proposed")
    store.update_schedule(schedule.id, state="paused", cadence_kind="", cadence={},
                          last_outcome="needs a cadence")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == schedule.id)
    actions = {a["action"] for a in row["actions"]}
    assert "edit_timing" in actions
    assert "resume" not in actions  # nothing to resume into until it has a timing


def test_active_schedule_whose_last_run_did_not_complete_is_stuck(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Donor tracker sync",
                              prompt="Check the donor page and use inkbox_lookup_contact to update it.",
                              state="active")
    _add_run(store, schedule.id, outcome="incomplete", result_line="The total did change (+$200).",
             denied=[{"tool": "mcp__inkbox__inkbox_lookup_contact", "reason": "scope"}])

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == schedule.id)
    assert row["kind"] == "schedule"
    assert "The total did change" in row["sentence"]


def test_schedule_proposal_waiting_for_yes_is_stuck_with_approve_and_reject(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Monday brief", state="proposed")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == schedule.id)
    assert row["kind"] == "approval"
    actions = {a["action"] for a in row["actions"]}
    assert actions == {"approve", "reject"}
    assert row["actions"][0]["target"]["endpoint"] == "schedules/approve"


def test_a_failed_request_in_the_last_day_is_stuck(tmp_path):
    store = make_store(tmp_path)
    request = _make_request(store, summary="Wire the vendor invoice.", state="failed",
                            status={"error": "the vendor portal timed out"})
    _backdate(store, request.id, created_at=time.time() - 3600, updated_at=time.time() - 3600)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["kind"] == "request" and r["id"] == request.id)
    assert "vendor portal timed out" in row["sentence"]
    assert row["actions"] == []  # no retry endpoint exists


def test_failed_request_older_than_a_day_is_not_stuck(tmp_path):
    store = make_store(tmp_path)
    request = _make_request(store, summary="Old failed thing.", state="failed",
                            status={"error": "old failure"})
    _backdate(store, request.id, created_at=time.time() - 90000, updated_at=time.time() - 90000)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert request.id not in [r["id"] for r in result["stuck"] if r["kind"] == "request"]


# ---------------------------------------------------------------------------
# A failed request superseded by a later success on the same task is not stuck.
# ---------------------------------------------------------------------------

def test_failed_request_superseded_by_later_success_same_task_is_not_stuck(tmp_path):
    store = make_store(tmp_path)
    task_id = store.create_task("Donor list approval")["id"]
    failed = _make_request(store, title_task=task_id, summary="Approve the donor list.",
                           state="failed", status={"error": "it ran out of time"})
    _backdate(store, failed.id, created_at=time.time() - 1000, updated_at=time.time() - 1000)
    retried = _make_request(store, title_task=task_id, summary="Approve the donor list.", state="done")
    _backdate(store, retried.id, created_at=time.time() - 500, updated_at=time.time() - 500)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert failed.id not in [r["id"] for r in result["stuck"] if r["kind"] == "request"]


def test_failed_request_not_superseded_without_a_later_success_is_still_stuck(tmp_path):
    store = make_store(tmp_path)
    task_id = store.create_task("Donor list approval")["id"]
    failed = _make_request(store, title_task=task_id, summary="Approve the donor list.",
                           state="failed", status={"error": "it ran out of time"})
    _backdate(store, failed.id, created_at=time.time() - 1000, updated_at=time.time() - 1000)
    another_failure = _make_request(store, title_task=task_id, summary="Approve the donor list.",
                                    state="failed", status={"error": "it ran out of time again"})
    _backdate(store, another_failure.id, created_at=time.time() - 500, updated_at=time.time() - 500)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert failed.id in [r["id"] for r in result["stuck"] if r["kind"] == "request"]


# ---------------------------------------------------------------------------
# A hand-off prompt is never shown; the owner's original message stands in.
# ---------------------------------------------------------------------------

def test_handoff_prompt_summary_is_replaced_by_the_owners_original_message(tmp_path):
    store = make_store(tmp_path)
    request = _make_request(
        store,
        summary="Live phone call with an unknown caller about a billing dispute that needs context.",
        original_message="Can you call the billing company back about the overcharge?",
        state="pending",
    )

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == request.id)
    assert "Live phone call with" not in row["sentence"]
    assert "overcharge" in row["sentence"]


def test_sentences_are_whole_never_cut(tmp_path):
    store = make_store(tmp_path)
    long_summary = "Approve this: " + ("x" * 400)
    request = _make_request(store, summary=long_summary, original_message=long_summary, state="pending")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == request.id)
    assert long_summary in row["sentence"]


# ---------------------------------------------------------------------------
# running / done_today / coming_up / counts / sources.
# ---------------------------------------------------------------------------

def test_running_request_and_running_schedule_both_appear(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Live sync", state="active")
    sched_request = _make_request(store, summary="running the sync", state="running",
                                  schedule_id=schedule.id)
    plain_request = _make_request(store, summary="Draft the weekly memo.", state="running")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    ids = {(r["kind"], r["id"]) for r in result["running"]}
    assert ("schedule", schedule.id) in ids
    assert ("request", plain_request.id) in ids
    assert result["counts"]["running"] == len(result["running"]) == 2


def test_done_today_includes_finished_requests_and_schedule_runs_since_midnight(tmp_path):
    store = make_store(tmp_path)
    request = _make_request(store, summary="Draft the weekly memo.", state="done")
    schedule = _make_schedule(store, title="Team total watch", state="active")
    _add_run(store, schedule.id, outcome="complete", result_line="Total went up by $500.")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    kinds_ids = {(r["kind"], r["id"]) for r in result["done_today"]}
    assert ("request", request.id) in kinds_ids
    assert ("schedule", schedule.id) in kinds_ids
    by_id = {(r["kind"], r["id"]): r for r in result["done_today"]}
    assert by_id[("request", request.id)]["outcome"] == "finished"
    assert by_id[("schedule", schedule.id)]["outcome"] == "finished"
    assert result["counts"]["done_today"] == len(result["done_today"])


def test_done_today_excludes_runs_finished_before_todays_midnight(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Old run", state="active")
    _add_run(store, schedule.id, outcome="complete", result_line="yesterday's result",
             finished_at=time.time() - 200000)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert schedule.id not in [r["id"] for r in result["done_today"] if r["kind"] == "schedule"]


def test_coming_up_lists_the_next_three_schedule_firings_only(tmp_path):
    store = make_store(tmp_path)
    now = time.time()
    for i, title in enumerate(["First", "Second", "Third", "Fourth"]):
        s = _make_schedule(store, title=title, state="active")
        store.update_schedule(s.id, next_run=now + (i + 1) * 3600)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert [c["title"] for c in result["coming_up"]] == ["First", "Second", "Third"]
    assert all(c["next_run_in_seconds"] > 0 for c in result["coming_up"])


def test_counts_match_the_section_lengths(tmp_path):
    store = make_store(tmp_path)
    _make_request(store, summary="Approve this.", state="pending")
    _make_schedule(store, title="Running one", state="active")
    result = api.overview(FakeGateway(FakeManager(store)), store)
    assert result["counts"] == {
        "stuck": len(result["stuck"]), "running": len(result["running"]),
        "done_today": len(result["done_today"]),
    }


def test_sources_are_present_and_describe_what_the_page_is_built_from(tmp_path):
    store = make_store(tmp_path)
    _make_request(store, summary="Approve this.", state="pending")
    result = api.overview(FakeGateway(FakeManager(store)), store)
    labels = {s["label"] for s in result["sources"]}
    assert labels == {"Requests", "Schedules and their runs"}
    for source in result["sources"]:
        assert source["detail"]


# ---------------------------------------------------------------------------
# The schedule state label helper: never "Waiting".
# ---------------------------------------------------------------------------

def test_active_schedule_with_nothing_running_is_labeled_active_not_waiting(tmp_path):
    assert api._state_label(_FakeSchedule(state="active"), running=False) == "Active"
    assert api._state_label(_FakeSchedule(state="active"), running=True) == "Running"


class _FakeSchedule:
    def __init__(self, state, last_outcome=""):
        self.state = state
        self.last_outcome = last_outcome


# ---------------------------------------------------------------------------
# The Origin gate applies to /console/api/overview like every other route.
# ---------------------------------------------------------------------------

def _test_app(manager):
    app = web.Application()
    app["console.gateway"] = FakeGateway(manager)
    r = console_routes
    app.router.add_get("/console/api/overview", r._endpoint(r.get_overview))
    return app


@pytest.mark.asyncio
async def test_overview_route_serves_the_contract_shape():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        manager = FakeManager(Store(f"{tmp}/overview.db"))
        _make_request(manager.store, summary="Approve this.", state="pending")
        async with TestClient(TestServer(_test_app(manager))) as client:
            resp = await client.get("/console/api/overview")
            assert resp.status == 200
            body = await resp.json()
            assert set(body.keys()) == {"stuck", "running", "done_today", "coming_up", "counts", "sources"}


@pytest.mark.asyncio
async def test_overview_route_is_gated_by_the_tailnet_origin_check_like_the_others():
    from inkbox_claude.console import access
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        manager = FakeManager(Store(f"{tmp}/overview.db"))
        app = _test_app(manager)
        app.middlewares.append(access.tailnet_gate)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/console/api/overview")
            # Same as every other console route: a loopback test-client peer is
            # not a tailnet address, so the gate refuses it before the handler
            # (and its store lookup) ever runs.
            assert resp.status == 404
