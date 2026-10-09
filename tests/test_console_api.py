"""Tests for the console's roles/people API logic, the Inkbox people sync engine, and
the tailnet + origin access control. Uses only throwaway sqlite databases and a fake
Inkbox client -- no live credentials, no real gate.db, no network."""

from __future__ import annotations

import pytest
from aiohttp import web

from inkbox_claude.console import access, api, events, sync
from inkbox_claude.gate.store import Store


def make_store(tmp_path):
    return Store(str(tmp_path / "console.db"))


# ---------------------------------------------------------------------------
# Roles: members, scope kinds, the send warning, validation
# ---------------------------------------------------------------------------

def test_role_with_members_and_scope_kinds(tmp_path):
    store = make_store(tmp_path)
    store.set_role("comms", ["inbox_read", "email_send", "sms_send"], note="can send")
    store.set_trust("mia@example.edu", person="Mia", role="comms")
    store.set_trust("+15551230000", person="Theo", role="comms")

    rows = api.roles(store)
    role = next(r for r in rows if r["name"] == "comms")
    assert role["people_count"] == 2
    assert {m["key"] for m in role["members"]} == {"mia@example.edu", Store.trusted_key("+15551230000")}
    assert {m["person"] for m in role["members"]} == {"Mia", "Theo"}
    read_names = {s["name"] for s in role["read_scopes"]}
    write_names = {s["name"] for s in role["write_scopes"]}
    assert "inbox_read" in read_names
    assert {"email_send", "sms_send"} <= write_names
    for s in role["write_scopes"]:
        assert s["kind"] in ("write", "send")
    assert "without Aaron seeing them first" in role["warning"]


def test_role_without_send_scope_has_no_warning(tmp_path):
    store = make_store(tmp_path)
    store.set_role("read-only", ["inbox_read"], note="")
    role = next(r for r in api.roles(store) if r["name"] == "read-only")
    assert role["warning"] == ""


def test_add_and_remove_role_member(tmp_path):
    store = make_store(tmp_path)
    store.set_role("ops", ["inbox_read"])

    role = api.add_role_member(store, {"name": "ops", "key": "new@example.com", "person": "New Person"})
    assert any(m["key"] == "new@example.com" for m in role["members"])

    role = api.remove_role_member(store, {"name": "ops", "key": "new@example.com"})
    assert all(m["key"] != "new@example.com" for m in role["members"])
    # Removing a member clears the role, not the person's whole trust row.
    trusted = next(r for r in store.trusted() if r["key"] == "new@example.com")
    assert trusted["role"] == ""


def test_add_member_to_unknown_role_is_rejected(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(api.ValidationError):
        api.add_role_member(store, {"name": "ghost-role", "key": "x@example.com"})


def test_upsert_role_rejects_unknown_scope(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(api.ValidationError):
        api.upsert_role(store, {"name": "bad", "scopes": ["not_a_real_scope"]})


def test_upsert_role_rejects_empty_name(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(api.ValidationError):
        api.upsert_role(store, {"name": "   ", "scopes": []})


# ---------------------------------------------------------------------------
# People: effective scopes, counts, derived kind flags
# ---------------------------------------------------------------------------

def test_people_scope_counts_and_send_flag(tmp_path):
    store = make_store(tmp_path)
    store.set_role("comms", ["inbox_read", "email_send"])
    store.set_trust("mia@example.edu", person="Mia", role="comms")

    rows = api.people(store)
    row = next(r for r in rows if r["key"] == "mia@example.edu")
    assert row["scope_counts"]["read"] == 1
    assert row["scope_counts"]["write"] == 1
    assert row["has_send"] is True


# ---------------------------------------------------------------------------
# Inkbox people sync: paging, dedupe, idempotence, never touches trust/roles
# ---------------------------------------------------------------------------

class FakeInkboxClient:
    def __init__(self):
        now = 1_700_000_000.0
        self._imessage = [
            {"handle": "+15551230000", "name": "Theo Park", "last_seen": now},
        ]
        self._sms = [
            {"handle": "+15551230000", "name": "Theo", "last_seen": now - 10},  # same person, shorter name
            {"handle": "+15559990000", "name": "", "last_seen": now - 500},
        ]
        self._email = [
            {"handle": "dana@example.com", "name": "Dana Lee", "last_seen": now - 2000},
        ]
        self._calls = []
        self._contacts = [
            {"handle": "dana@example.com", "name": "Dana Lee", "last_seen": now - 2000},
        ]

    def imessage_conversations(self):
        return self._imessage

    def sms_threads(self):
        return self._sms

    def email_threads(self):
        return self._email

    def calls(self):
        return self._calls

    def contacts(self):
        return self._contacts


def test_sync_people_pages_dedupes_and_merges(tmp_path):
    store = make_store(tmp_path)
    client = FakeInkboxClient()

    result = sync.sync_people(store, client)
    assert result["ok"] is True
    # Theo (imessage+sms dedup), Dana (email+contact dedup), and the nameless sms-only
    # handle -- three distinct people out of five rows across the channels.
    assert result["people"] == 3

    people = api.people(store)
    theo = next(r for r in people if r["key"] == store.trusted_key("+15551230000"))
    assert theo["person"] == "Theo Park"  # the longer name wins
    assert set(theo["channels"]) >= {"imessage", "sms"}

    dana = next(r for r in people if r["key"] == store.trusted_key("dana@example.com"))
    assert set(dana["channels"]) >= {"email", "contact"}

    # A third handle with no name anywhere still shows up.
    assert any(r["key"] == store.trusted_key("+15559990000") for r in people)


def test_sync_people_never_touches_trust_or_roles(tmp_path):
    store = make_store(tmp_path)
    store.set_role("ops", ["inbox_read"])
    store.set_trust("mia@example.edu", person="Mia", role="ops")
    before_trust = store.trusted()
    before_roles = store.roles()

    sync.sync_people(store, FakeInkboxClient())

    assert store.trusted() == before_trust
    assert store.roles() == before_roles
    # A freshly synced person has no role and no scopes.
    synced = next(r for r in api.people(store) if r["key"] == store.trusted_key("dana@example.com"))
    assert synced["role"] == ""
    assert synced["effective_scopes"] == []


def test_sync_people_is_idempotent(tmp_path):
    store = make_store(tmp_path)
    client = FakeInkboxClient()
    first = sync.sync_people(store, client)
    second = sync.sync_people(store, client)
    assert first["people"] == second["people"]
    assert len(api.people(store)) == first["people"]


def test_sync_records_last_sync_time_and_counts(tmp_path):
    store = make_store(tmp_path)
    sync.sync_people(store, FakeInkboxClient())
    info = sync.last_sync(store)
    assert info["at"] is not None
    assert info["counts"].get("imessage") == 1
    assert info["counts"].get("sms") == 2


def test_sync_excludes_owner_and_bot_handles(tmp_path):
    store = make_store(tmp_path)
    store.set_setting("owner_phone", "+15551230000")

    class ClientWithOwner(FakeInkboxClient):
        pass

    sync.sync_people(store, ClientWithOwner())
    people = api.people(store)
    assert not any(r["key"] == store.trusted_key("+15551230000") for r in people)


# ---------------------------------------------------------------------------
# Access control: tailnet peer required, loopback-through-tunnel refused,
# origin allow-list for CORS, preflight answered.
# ---------------------------------------------------------------------------

def test_is_tailnet_peer():
    assert access.is_tailnet_peer("100.64.0.10") is True
    assert access.is_tailnet_peer("100.64.0.1") is True
    assert access.is_tailnet_peer("127.0.0.1") is False
    assert access.is_tailnet_peer("192.168.1.163") is False
    assert access.is_tailnet_peer(None) is False
    assert access.is_tailnet_peer("not-an-ip") is False


def test_origin_allowed_requires_explicit_list(monkeypatch):
    monkeypatch.delenv("CONSOLE_ALLOWED_ORIGINS", raising=False)
    assert access.origin_allowed("https://console.example.com") is False

    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", "https://console.example.com, https://other.example.com")
    assert access.origin_allowed("https://console.example.com") is True
    assert access.origin_allowed("https://other.example.com") is True
    assert access.origin_allowed("https://evil.example.com") is False


@pytest.mark.asyncio
async def test_tailnet_gate_blocks_loopback_and_public_peers():
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    async def handler(_request):
        return web.json_response({"ok": True})

    app = web.Application(middlewares=[access.tailnet_gate])
    app.router.add_get("/console/api/health", handler)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/console/api/health")
        # aiohttp's test client connects over loopback, which is not a tailnet
        # address, so even a request that looks locally-sourced is refused -- the
        # same case as the public tunnel terminating on localhost.
        assert resp.status == 404


@pytest.mark.asyncio
async def test_cors_gate_answers_preflight_for_allowed_origin(monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", "https://console.example.com")

    async def handler(_request):
        return web.json_response({"ok": True})

    app = web.Application(middlewares=[access.cors_gate])
    app.router.add_get("/console/api/health", handler)
    app.router.add_route("OPTIONS", "/console/api/health", lambda r: web.Response(status=204))

    async with TestClient(TestServer(app)) as client:
        resp = await client.options(
            "/console/api/health",
            headers={"Origin": "https://console.example.com",
                     "Access-Control-Request-Method": "GET"},
        )
        assert resp.status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "https://console.example.com"

        resp = await client.options(
            "/console/api/health",
            headers={"Origin": "https://evil.example.com",
                     "Access-Control-Request-Method": "GET"},
        )
        assert "Access-Control-Allow-Origin" not in resp.headers


@pytest.mark.asyncio
async def test_event_stream_sets_cors_header_for_allowed_origin(monkeypatch):
    """Regression test: an SSE response is already prepared (headers flushed) by the
    time cors_gate's middleware tries to add Access-Control-Allow-Origin after
    `await handler(request)` returns, so a browser EventSource from an allowed
    cross-origin frontend would otherwise be blocked by CORS. event_stream must set
    the header itself, before `response.prepare()`."""
    from aiohttp.test_utils import TestClient, TestServer

    monkeypatch.setenv("CONSOLE_ALLOWED_ORIGINS", "https://console.example.com")

    app = web.Application()
    app.router.add_get("/console/events", events.event_stream)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/console/events", headers={"Origin": "https://console.example.com"}
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "https://console.example.com"
        resp.close()

        resp = await client.get("/console/events", headers={"Origin": "https://evil.example.com"})
        assert "Access-Control-Allow-Origin" not in resp.headers
        resp.close()


# ---------------------------------------------------------------------------
# Home: the dashboard's single shaped endpoint
# ---------------------------------------------------------------------------

class FakeGateway:
    """Enough of the real gateway for api.health()/api.home() to read."""

    def __init__(self, started_at=1_700_000_000.0, public_url="", tunnel=None):
        self._console_started_at = started_at
        self._public_url = public_url
        self._tunnel = tunnel


def test_home_empty_state(tmp_path):
    store = make_store(tmp_path)
    gateway = FakeGateway()

    result = api.home(gateway, store)

    assert result["needs_you"] == {"count": 0, "oldest": None}
    assert result["right_now"]["running_count"] == 0
    assert result["right_now"]["running"] == []
    assert result["right_now"]["gateway_up"] is True
    assert result["recent_activity"] == []
    assert result["coming_up"] == {"schedules_paused": False, "next": []}
    assert result["people"]["total"] == 0
    assert result["people"]["with_role"] == 0
    assert result["people"]["roles"] == []
    assert result["people"]["last_sync"] == {"at": None, "counts": {}, "failed": []}
    assert result["recent_contacts"] == []
    # No message bodies or secrets anywhere in the shape.
    assert "original_message" not in str(result["needs_you"])


def test_home_needs_you_is_the_oldest_pending_request(tmp_path):
    store = make_store(tmp_path)
    gateway = FakeGateway()
    task_id = store.create_task("Help Alex")["id"]
    first = store.create_request(
        chat_id="c1", sender="a@example.com", sender_name="Alex", mode="email",
        subject="", original_message="please do X", summary="Do X", scopes=[],
        prompt="do X", state="pending", task_id=task_id,
    )
    store.create_request(
        chat_id="c2", sender="b@example.com", sender_name="Bo", mode="sms",
        subject="", original_message="please do Y", summary="Do Y", scopes=[],
        prompt="do Y", state="pending", task_id=task_id,
    )

    result = api.home(gateway, store)

    assert result["needs_you"]["count"] == 2
    assert result["needs_you"]["oldest"]["id"] == first.id
    assert result["needs_you"]["oldest"]["summary"] == "Do X"
    assert "original_message" not in result["needs_you"]["oldest"]


def test_home_people_and_roles_summary(tmp_path):
    store = make_store(tmp_path)
    gateway = FakeGateway()
    store.set_role("comms", ["inbox_read", "email_send"], note="")
    store.set_trust("mia@example.edu", person="Mia", role="comms")
    store.set_trust("stranger@example.com", person="Stranger", role="")

    result = api.home(gateway, store)

    assert result["people"]["total"] == 2
    assert result["people"]["with_role"] == 1
    roles_by_name = {r["name"]: r["members"] for r in result["people"]["roles"]}
    assert roles_by_name["comms"] == 1


def test_home_coming_up_lists_only_active_future_schedules(tmp_path):
    import time as _time

    store = make_store(tmp_path)
    gateway = FakeGateway()
    task_id = store.create_task("Weekly report")["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Weekly report", prompt="do the report",
        kind="recurring", scopes=[], timezone="America/New_York", report_mode="always",
        cron="0 9 * * 1-5", next_run=_time.time() + 3600,
    )
    store.update_schedule(schedule.id, state="active")
    # A paused schedule with a future run must not appear in "coming up".
    other_task = store.create_task("Paused job")["id"]
    paused = store.create_schedule(
        chat_id="owner", task_id=other_task, title="Paused job", prompt="do the thing",
        kind="once", scopes=[], timezone="America/New_York", report_mode="always",
        run_at=_time.time() + 7200, next_run=_time.time() + 7200,
    )
    store.update_schedule(paused.id, state="paused")

    result = api.home(gateway, store)

    assert result["coming_up"]["schedules_paused"] is False
    assert len(result["coming_up"]["next"]) == 1
    assert result["coming_up"]["next"][0]["title"] == "Weekly report"
    assert "cron" not in result["coming_up"]["next"][0]  # worded cadence, not raw cron


def test_home_recent_contacts_excludes_people_never_seen(tmp_path):
    store = make_store(tmp_path)
    gateway = FakeGateway()
    store.remember_synced_people([
        {"key": "+15551230000", "person": "Theo", "channels": ["sms"], "last_seen": 1_700_000_000.0},
        {"key": "dana@example.com", "person": "Dana", "channels": [], "last_seen": 0.0},
    ])

    result = api.home(gateway, store)

    keys = {c["key"] for c in result["recent_contacts"]}
    assert "+15551230000" in keys
    assert "dana@example.com" not in keys
    for contact in result["recent_contacts"]:
        assert "message" not in contact and "text" not in contact


def test_thread_counterparts_are_senders_not_conversation_ids(tmp_path):
    from inkbox_claude.gate.store import Store
    store = Store(str(tmp_path / "gate.db"))
    store.set_thread("9b9bd364-1b71-4cf2-b3e5-2e747793ce25", "idle", mode="imessage", meta={"sender": "+15551230000"})
    store.set_thread("owner", "idle", mode="imessage", meta={})
    handles = {row["handle"] for row in store.thread_counterparts()}
    assert handles == {"+15551230000"}


def test_task_detail_returns_every_ledger_event(tmp_path):
    from inkbox_claude.console import api
    from inkbox_claude.gate.store import Store
    store = Store(str(tmp_path / "gate.db"))
    task = store.create_task("long running")
    for n in range(150):
        store.task_event(task["id"], "note", f"event {n}", chat_id="c")
    detail = api.task(store, task["id"])
    assert len(detail["events"]) == 150
    assert detail["events"][0]["text"] == "event 0" and detail["events"][-1]["text"] == "event 149"
