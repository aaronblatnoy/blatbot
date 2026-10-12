"""Tests for the schedules half of the console API: the list/runs read models in
inkbox_claude/console/api.py, the create/edit/action/approve command wrappers, and
the Origin access gate on the mutation routes in inkbox_claude/console/routes.py.

Runs against the real store, the real cadence.py and schedprop.py (not faked --
those packages are already built in this tree), and a fake manager that stands in
for the gate's Manager/ScheduleService wiring. Made-up titles and prompts only, no
real schedule content."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from inkbox_claude.console import api
from inkbox_claude.console import routes as console_routes
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.store import Store


def make_store(tmp_path):
    return Store(str(tmp_path / "console.db"))


class FakeManager:
    """Enough of the real gate Manager for ScheduleService and the console API
    functions under test: a store, a bound ScheduleService, and the notice/execute
    seams ScheduleService calls into."""

    def __init__(self, store):
        self.store = store
        self.schedules = sched_mod.ScheduleService(self)
        self.notices = []

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule.id, text))
        return self.store.add_message(
            schedule.origin_chat_id or "owner", "outbound", text,
            schedule.origin_mode or "console", role="notice", schedule_id=schedule.id,
        )

    async def execute(self, req):
        self.store.set_state(req.id, "running")


class FakeGateway:
    """Enough of the real gateway for routes._manager() to find the store/service."""

    def __init__(self, manager):
        self.sessions = manager


def make_manager(tmp_path):
    store = make_store(tmp_path)
    return FakeManager(store)


def _run(coro):
    return asyncio.run(coro)


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
        import time
        schedule = store.update_schedule(schedule.id, state=state, next_run=time.time() + 3600, **extra)
    return schedule


def _add_run(store, schedule_id, *, outcome="complete", result_line="done",
             final_status="", denied=None, note="", changed=True, cost_usd=0.01):
    import time
    run_id = store.create_schedule_run(schedule_id, None, time.time() - 60)
    store.finish_schedule_run(
        run_id, finished_at=time.time() - 55, complete=(outcome == "complete"),
        outcome=outcome, result_line=result_line, final_status=final_status or result_line,
        denied=denied or [], note=note, changed=changed, cost_usd=cost_usd,
    )
    return run_id


def _test_app(manager):
    """A minimal app exposing only the schedule routes through the identical
    _endpoint(..., mutation=True) wrapper routes.register() uses -- without the
    tailnet_gate middleware, which rejects every request from a test client
    (loopback is never a tailnet address) before the Origin check ever runs."""
    app = web.Application()
    app["console.gateway"] = FakeGateway(manager)
    r = console_routes
    app.router.add_get("/console/api/schedules", r._endpoint(r.get_schedules))
    app.router.add_get("/console/api/schedules/{id}/runs", r._endpoint(r.get_schedule_runs))
    app.router.add_post("/console/api/schedules", r._endpoint(r.post_schedules, mutation=True))
    app.router.add_post("/console/api/schedules/edit", r._endpoint(r.post_schedules_edit, mutation=True))
    app.router.add_post("/console/api/schedules/action", r._endpoint(r.post_schedules_action, mutation=True))
    app.router.add_post("/console/api/schedules/approve", r._endpoint(r.post_schedules_approve, mutation=True))
    return app


# ---------------------------------------------------------------------------
# GET /console/api/schedules: shape, every state, missing capability, a last
# run that could not finish.
# ---------------------------------------------------------------------------

def test_schedule_list_shapes_every_state_and_top_level_fields(tmp_path):
    manager = make_manager(tmp_path)
    store = manager.store

    active = _make_schedule(store, title="Team total watch", state="active")
    _add_run(store, active.id, outcome="complete", result_line="Total went from $4,027.39 to $4,922.45.")

    proposed = _make_schedule(store, title="Monday brief", state="proposed", scopes=["web"])

    incomplete = _make_schedule(
        store, title="Donor tracker sync",
        prompt="Check the donor page and use inkbox_lookup_contact to update the tracker.",
        scopes=["web"], state="active",
    )
    _add_run(store, incomplete.id, outcome="incomplete", result_line="The total did change (+$200).",
             denied=[{"tool": "mcp__inkbox__inkbox_lookup_contact", "reason": "scope"}])

    paused = _make_schedule(store, title="Weekly backup check", cadence_kind="weekly",
                            cadence={"day": "mon", "time": "06:00"}, state="paused")

    ended = _make_schedule(store, title="October invoice reminder", cadence_kind="once",
                           cadence={"at": "2026-10-01T09:00:00"}, state="ended",
                           last_outcome="stopped")

    result = api.schedule_list(store, manager.schedules)
    by_title = {s["title"]: s for s in result["schedules"]}

    assert by_title["Team total watch"]["state_label"] == "Waiting"  # active, no run in flight
    assert by_title["Monday brief"]["state_label"] == "Needs your yes"
    assert by_title["Monday brief"]["needs_approval"] is True
    assert by_title["Weekly backup check"]["state_label"] == "Paused"
    assert by_title["October invoice reminder"]["state_label"] == "Stopped"

    donor = by_title["Donor tracker sync"]
    assert donor["missing_capabilities"] == [{"scope": "contacts", "label": "look up and edit contacts"}]
    assert donor["last_run"]["outcome"] == "incomplete"
    assert donor["last_run"]["outcome_label"] == "Could not finish"
    assert donor["last_run"]["denied"] == [
        {"tool": "mcp__inkbox__inkbox_lookup_contact", "label": "look up and edit contacts"}
    ]

    team = by_title["Team total watch"]
    assert team["cadence_text"] == "every 4 hours"
    assert team["capabilities"] == [{"scope": "web", "label": "read public web pages"}]
    assert team["last_run"]["outcome_label"] == "Finished"
    assert team["next_run_text"] is not None
    assert team["next_run_in_seconds"] is not None

    assert result["paused_all"] is False
    kinds = {k["kind"] for k in result["cadence_kinds"]}
    assert kinds == {"once", "every", "daily", "weekdays", "weekly", "cron"}
    assert {"value": False, "label": "Only when something changed or it could not finish"} in result["report_options"]
    assert {"value": True, "label": "Every time it runs"} in result["report_options"]
    assert {w["value"] for w in result["weekdays"]} == {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    # Every capability option carries a real label, not the bare scope name.
    for opt in result["capability_options"]:
        assert opt["label"] and opt["label"] != opt["scope"]
        assert isinstance(opt["tools"], list)


def test_no_scope_name_is_ever_the_only_description_of_a_capability(tmp_path):
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, scopes=["web", "contacts"])
    view = api.schedule_list(manager.store, manager.schedules)
    row = next(s for s in view["schedules"] if s["id"] == schedule.id)
    for cap in row["capabilities"]:
        assert cap["label"] and cap["label"] != cap["scope"]
    for opt in view["capability_options"]:
        assert opt["label"] and opt["label"] != opt["scope"]


# ---------------------------------------------------------------------------
# GET /console/api/schedules/<id>/runs
# ---------------------------------------------------------------------------

def test_runs_endpoint_returns_a_long_final_status_whole(tmp_path):
    manager = make_manager(tmp_path)
    store = manager.store
    schedule = _make_schedule(store)
    long_status = "it kept going. " * 400  # several thousand characters
    assert len(long_status) > 3000
    _add_run(store, schedule.id, final_status=long_status)

    result = api.schedule_runs_view(store, schedule.id)

    assert len(result["runs"]) == 1
    assert result["runs"][0]["final_status"] == long_status
    assert result["runs"][0]["started_text"] is not None


def test_runs_endpoint_newest_first(tmp_path):
    manager = make_manager(tmp_path)
    store = manager.store
    schedule = _make_schedule(store)
    first = _add_run(store, schedule.id, result_line="first")
    second = _add_run(store, schedule.id, result_line="second")

    result = api.schedule_runs_view(store, schedule.id)

    assert [r["id"] for r in result["runs"]] == [second, first]


def test_runs_endpoint_unknown_schedule_returns_none(tmp_path):
    manager = make_manager(tmp_path)
    assert api.schedule_runs_view(manager.store, 999) is None


# ---------------------------------------------------------------------------
# POST /console/api/schedules (create) and /schedules/edit
# ---------------------------------------------------------------------------

def test_create_schedule_returns_needs_approval_and_stores_proposed(tmp_path):
    manager = make_manager(tmp_path)

    result = _run(api.create_schedule(manager, {
        "title": "Team total watch", "prompt": "check the team page and tell me when it changes",
        "cadence_kind": "every", "cadence": {"minutes": 240}, "scopes": ["web"],
        "report_every_time": False,
    }))

    assert result["needs_approval"] is True
    assert result["applied"] is False
    assert result["message"]
    assert result["schedule"]["state"] == "proposed"
    stored = manager.store.get_schedule(result["schedule"]["id"])
    assert stored.state == "proposed"
    assert stored.cadence_kind == "every"


def test_create_schedule_rejects_unknown_scope(tmp_path):
    manager = make_manager(tmp_path)
    with pytest.raises(api.ValidationError):
        _run(api.create_schedule(manager, {
            "title": "x", "prompt": "do a thing", "cadence_kind": "once",
            "cadence": {"at": "2026-11-01T09:00:00"}, "scopes": ["not_a_real_scope"],
        }))


def test_create_schedule_rejects_invalid_cadence(tmp_path):
    manager = make_manager(tmp_path)
    with pytest.raises(api.ValidationError):
        _run(api.create_schedule(manager, {
            "title": "x", "prompt": "do a thing", "cadence_kind": "every",
            "cadence": {"minutes": 1}, "scopes": ["web"],  # below the 5-minute floor
        }))


def test_edit_cadence_applies_at_once(tmp_path):
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="active")

    result = _run(api.edit_schedule(manager, {
        "id": schedule.id, "cadence_kind": "daily", "cadence": {"time": "08:00"},
    }))

    assert result["applied"] is True
    assert result["needs_approval"] is False
    assert result["schedule"]["cadence_kind"] == "daily"
    assert manager.store.get_schedule(schedule.id).cadence_kind == "daily"


def test_edit_prompt_needs_approval(tmp_path):
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="active")

    result = _run(api.edit_schedule(manager, {
        "id": schedule.id, "prompt": "check the team page and also look at the archive",
    }))

    assert result["applied"] is False
    assert result["needs_approval"] is True
    assert result["message"]
    assert manager.store.get_schedule(schedule.id).state == "proposed"


def test_edit_rejects_invalid_cadence(tmp_path):
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="active")
    with pytest.raises(api.ValidationError):
        _run(api.edit_schedule(manager, {"id": schedule.id, "cadence_kind": "weekly", "cadence": {}}))


# ---------------------------------------------------------------------------
# POST /console/api/schedules/action and /approve, via the real HTTP routes so
# the 409/Origin behavior is exercised end to end. An explicit allow-listed
# origin is used for the "allowed" requests rather than trying to match the
# test server's own dynamic host:port as a same-origin request would.
# ---------------------------------------------------------------------------

ALLOWED_ORIGIN = "https://console.example.com"


@pytest.mark.asyncio
async def test_action_valid_pauses_an_active_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="active")
    app = _test_app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/console/api/schedules/action",
            json={"id": schedule.id, "action": "pause"},
            headers={"Origin": ALLOWED_ORIGIN},
        )
        body = await resp.json()
        assert resp.status == 200
        assert body["schedule"]["state"] == "paused"


@pytest.mark.asyncio
async def test_action_invalid_for_state_returns_409(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="paused")
    app = _test_app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/console/api/schedules/action",
            json={"id": schedule.id, "action": "run_now"},  # run_now is not valid while paused
            headers={"Origin": ALLOWED_ORIGIN},
        )
        body = await resp.json()
        assert resp.status == 409
        assert "error" in body


@pytest.mark.asyncio
async def test_approve_activates_a_proposed_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="proposed")
    app = _test_app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/console/api/schedules/approve",
            json={"id": schedule.id, "approve": True},
            headers={"Origin": ALLOWED_ORIGIN},
        )
        body = await resp.json()
        assert resp.status == 200
        assert body["schedule"]["state"] == "active"


@pytest.mark.asyncio
async def test_reject_ends_a_proposed_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="proposed")
    app = _test_app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/console/api/schedules/approve",
            json={"id": schedule.id, "approve": False},
            headers={"Origin": ALLOWED_ORIGIN},
        )
        body = await resp.json()
        assert resp.status == 200
        assert body["schedule"]["state"] == "ended"


@pytest.mark.asyncio
async def test_approve_is_behind_the_same_origin_gate_as_every_other_console_write(tmp_path, monkeypatch):
    """The approve endpoint activates unattended work and must sit behind the
    identical cross-origin check every other console mutation uses -- a request
    whose Origin is not on the allow-list is refused before it reaches
    ScheduleService.decide, exactly like any other mutation route."""
    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="proposed")
    app = _test_app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/console/api/schedules/approve",
            json={"id": schedule.id, "approve": True},
            headers={"Origin": "https://evil.example.com"},
        )
        assert resp.status == 403
        body = await resp.json()
        assert "error" in body
    # The schedule must be untouched: the rejected request never reached decide().
    assert manager.store.get_schedule(schedule.id).state == "proposed"


@pytest.mark.asyncio
async def test_approve_with_no_origin_header_at_all_is_also_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    manager = make_manager(tmp_path)
    schedule = _make_schedule(manager.store, state="proposed")
    app = _test_app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/console/api/schedules/approve", json={"id": schedule.id, "approve": True})
        assert resp.status == 403
    assert manager.store.get_schedule(schedule.id).state == "proposed"
