"""The one-step repair console endpoints: GET .../repair returns the plan,
POST .../repair applies it -- against the real store/schedprop and the same
_test_app harness test_console_schedules.py uses, so these run through the
identical Origin gate as every other console write."""

from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from inkbox_claude.console import api
from inkbox_claude.console import routes as console_routes
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.store import Store

ALLOWED_ORIGIN = "https://console.example.com"

STUCK_PROMPT = (
    "Check the team fundraising page every 4 hours and tell me the total.\n\n"
    "When done, call schedule_continue with delay_minutes 240 to run again.\n\n"
    "If the total changed, call inkbox_update_contact with the new total."
)


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.schedules = sched_mod.ScheduleService(self)
        self.notices = []

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule.id, text))
        return self.store.add_message(schedule.origin_chat_id or "owner", "outbound", text,
                                      schedule.origin_mode or "console", role="notice",
                                      schedule_id=schedule.id)


class FakeGateway:
    def __init__(self, manager):
        self.sessions = manager


def _test_app(manager):
    app = web.Application()
    app["console.gateway"] = FakeGateway(manager)
    r = console_routes
    app.router.add_get("/console/api/schedules/{id}/repair", r._endpoint(r.get_schedule_repair))
    app.router.add_post("/console/api/schedules/repair", r._endpoint(r.post_schedules_repair, mutation=True))
    return app


def _make_stuck(store, *, prompt=STUCK_PROMPT, scopes=("web",)):
    task_id = store.create_task("repair console test")["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Team total watch", prompt=prompt,
        kind="", scopes=list(scopes), timezone="America/New_York", report_mode="changed",
        cadence_kind="", cadence={}, origin_mode="console", origin_chat_id="owner",
    )
    return store.update_schedule(schedule.id, state="paused", last_outcome="needs a cadence")


def test_get_repair_plan_returns_the_plan_shape(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    schedule = _make_stuck(store)
    app = _test_app(manager)

    async def go():
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(f"/console/api/schedules/{schedule.id}/repair")
            assert resp.status == 200
            return await resp.json()

    body = asyncio.run(go())
    assert body["needed"] is True
    assert body["timing"] == {"cadence_kind": "every", "cadence": {"minutes": 240}, "text": "every 4 hours"}
    assert body["add_capabilities"] == [{"scope": "contacts", "label": "look up and edit contacts"}]
    assert len(body["prompt_changes"]) == 1


def test_get_repair_plan_404_for_unknown_schedule(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    app = _test_app(manager)

    async def go():
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/console/api/schedules/9999/repair")
            return resp.status

    assert asyncio.run(go()) == 404


def test_post_repair_applies_and_sends_one_proposal(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    schedule = _make_stuck(store)
    app = _test_app(manager)

    async def go():
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/console/api/schedules/repair", json={"id": schedule.id},
                headers={"Origin": ALLOWED_ORIGIN},
            )
            assert resp.status == 200
            return await resp.json()

    body = asyncio.run(go())
    assert body["needs_approval"] is True
    assert body["schedule"]["state"] == "proposed"
    assert len(manager.notices) == 1


def test_post_repair_is_behind_the_same_origin_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    schedule = _make_stuck(store)
    app = _test_app(manager)

    async def go():
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/console/api/schedules/repair", json={"id": schedule.id})
            return resp.status

    assert asyncio.run(go()) == 403
