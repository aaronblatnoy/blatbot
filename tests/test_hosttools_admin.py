"""Tests for Blatbot's own-settings admin tools (gate/hosttools.py ADMIN_*):
commander-only read/change of console-exposed settings, role scopes, person
roles, schedules, and chat modes. Every write goes through the same
console/api.py function the web console calls. Uses only throwaway sqlite
databases and a fake manager -- no live credentials, no network."""

from __future__ import annotations

import asyncio
import time

import pytest

from inkbox_claude.console import api as console_api
from inkbox_claude.gate import hosttools
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate import settings as gate_settings
from inkbox_claude.gate.scopes import SCOPES, tools_for, BLATBOT_ADMIN_SCOPE, BLATBOT_ADMIN_TOOLS
from inkbox_claude.gate.store import Store


def make_store(tmp_path):
    return Store(str(tmp_path / "gate.db"))


class FakeManager:
    """Enough of the real gate Manager for ScheduleService and the admin tools
    under test: a store, a bound ScheduleService, and the notice/execute seams
    ScheduleService calls into (same shape as tests/test_console_schedules.py)."""

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


def make_manager(tmp_path):
    store = make_store(tmp_path)
    # settings.py reads/writes through a module-global binding (the same one
    # GateSessionManager.__init__ sets up for real), rebound here to this
    # test's own throwaway store so admin_get/set_setting round-trip for real.
    gate_settings.bind(store)
    gate_settings.invalidate()
    return FakeManager(store)


def run(coro):
    return asyncio.run(coro)


def _make_schedule(store, *, title="Corgi NYC digest", prompt="summarize the chat",
                   scopes=None, cadence_kind="every", cadence=None, state="active"):
    task_id = store.create_task(title)["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title=title, prompt=prompt, kind=cadence_kind,
        scopes=list(scopes or ["web"]), timezone="America/New_York", report_mode="changed",
        cadence_kind=cadence_kind, cadence=cadence or {"minutes": 240},
        origin_mode="console", origin_chat_id="owner",
    )
    if state != "proposed":
        schedule = store.update_schedule(schedule.id, state=state, next_run=time.time() + 3600)
    return schedule


# ---------------------------------------------------------------------------
# Scope wiring: blatbot_admin is never a key any role can be given, and its
# tools are only in tools_for() output when the scope itself is present.
# ---------------------------------------------------------------------------

def test_blatbot_admin_scope_is_never_in_the_registry():
    assert BLATBOT_ADMIN_SCOPE not in SCOPES


def test_tools_for_grants_admin_tools_only_with_the_scope():
    some_other_scope = next(iter(SCOPES))
    without = tools_for([some_other_scope])
    assert not (set(BLATBOT_ADMIN_TOOLS) & set(without))
    granted = tools_for([BLATBOT_ADMIN_SCOPE])
    assert set(BLATBOT_ADMIN_TOOLS) <= set(granted)


def test_role_scope_endpoints_refuse_the_admin_scope_name(tmp_path):
    store = make_store(tmp_path)
    store.set_role("corgi teammate", [], note="")
    with pytest.raises(console_api.ValidationError):
        console_api.add_role_scope(store, {"name": "corgi teammate", "scope": BLATBOT_ADMIN_SCOPE})


# ---------------------------------------------------------------------------
# admin_list_settings / admin_get_setting / admin_set_setting
# ---------------------------------------------------------------------------

def test_admin_list_and_get_setting_happy_path(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_list_settings({}, m, True))
    assert "GATE_CONFIRM_SENDS" in out
    out2 = run(hosttools.admin_get_setting({"key": "GATE_CONFIRM_SENDS"}, m, True))
    assert "GATE_CONFIRM_SENDS" in out2


def test_admin_set_setting_changes_value_and_reports_old_and_new(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_set_setting({"key": "GATE_CONFIRM_SENDS", "value": "false"}, m, True))
    assert "->" in out
    confirm = run(hosttools.admin_get_setting({"key": "GATE_CONFIRM_SENDS"}, m, True))
    assert "False" in confirm


def test_admin_set_setting_refuses_unknown_name(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_set_setting({"key": "NOT_A_REAL_SETTING", "value": "1"}, m, True))
    assert out.startswith("ERROR")


def test_admin_get_setting_refuses_unknown_name(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_get_setting({"key": "NOT_A_REAL_SETTING"}, m, True))
    assert out.startswith("ERROR")


def test_admin_settings_refused_for_non_commander(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_list_settings({}, m, False))
    assert "commander" in out.lower()
    out2 = run(hosttools.admin_set_setting({"key": "GATE_CONFIRM_SENDS", "value": "0"}, m, False))
    assert "commander" in out2.lower()


# ---------------------------------------------------------------------------
# admin_list_roles / admin_set_role_scopes
# ---------------------------------------------------------------------------

def test_admin_list_roles_happy_path(tmp_path):
    m = make_manager(tmp_path)
    m.store.set_role("corgi teammate", ["pension_filings_read"], note="corgi work")
    out = run(hosttools.admin_list_roles({}, m, True))
    assert "corgi teammate" in out


def test_admin_set_role_scopes_happy_path(tmp_path):
    m = make_manager(tmp_path)
    m.store.set_role("corgi teammate", ["pension_filings_read"], note="corgi work")
    scope_names = [n for n in list(SCOPES)[:2]]
    out = run(hosttools.admin_set_role_scopes({"role": "corgi teammate", "scopes": scope_names}, m, True))
    assert "->" in out
    row = next(r for r in console_api.roles(m.store) if r["name"] == "corgi teammate")
    assert sorted(row["scopes"]) == sorted(scope_names)
    assert row["note"] == "corgi work"   # note preserved, not wiped


def test_admin_set_role_scopes_refuses_commander_role(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_set_role_scopes({"role": "commander", "scopes": []}, m, True))
    assert out.startswith("ERROR")


def test_admin_set_role_scopes_refuses_unknown_scope(tmp_path):
    m = make_manager(tmp_path)
    m.store.set_role("corgi teammate", [], note="")
    out = run(hosttools.admin_set_role_scopes(
        {"role": "corgi teammate", "scopes": ["not_a_real_scope"]}, m, True))
    assert out.startswith("ERROR")


def test_admin_set_role_scopes_refuses_unknown_role(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_set_role_scopes({"role": "no such role", "scopes": []}, m, True))
    assert out.startswith("ERROR")


def test_admin_roles_refused_for_non_commander(tmp_path):
    m = make_manager(tmp_path)
    m.store.set_role("corgi teammate", [], note="")
    out = run(hosttools.admin_list_roles({}, m, False))
    assert "commander" in out.lower()
    out2 = run(hosttools.admin_set_role_scopes({"role": "corgi teammate", "scopes": []}, m, False))
    assert "commander" in out2.lower()


# ---------------------------------------------------------------------------
# admin_set_person_role
# ---------------------------------------------------------------------------

def test_admin_set_person_role_happy_path(tmp_path):
    m = make_manager(tmp_path)
    m.store.set_role("corgi teammate", [], note="")
    person = m.store.create_person(display="Mark Morrison")
    m.store.add_contact(person["id"], "email", "mark@example.com")
    out = run(hosttools.admin_set_person_role(
        {"person": "Mark Morrison", "role": "corgi teammate"}, m, True))
    assert "->" in out
    row = next(p for p in console_api.people_directory(m.store)["people"] if p["id"] == person["id"])
    assert "corgi teammate" in row["role"]


def test_admin_set_person_role_refuses_commander_change(tmp_path):
    m = make_manager(tmp_path)
    m.store.set_approver_handles(telegram_id="700000111", phone="5550100111")
    person = m.store.create_person(display="Sam Parker")
    m.store.add_contact(person["id"], "telegram", "telegram:700000111")
    m.store.add_contact(person["id"], "phone", "5550100111")
    assert m.store.commander_person_id() == person["id"]
    out = run(hosttools.admin_set_person_role({"person": "Sam Parker", "role": "general"}, m, True))
    assert out.startswith("ERROR")


def test_admin_set_person_role_refuses_ambiguous_match(tmp_path):
    m = make_manager(tmp_path)
    m.store.set_role("general", [], note="")
    p1 = m.store.create_person(display="Alex Lee")
    m.store.add_contact(p1["id"], "email", "alex.lee@example.com")
    p2 = m.store.create_person(display="Alex Kim")
    m.store.add_contact(p2["id"], "email", "alex.kim@example.com")
    out = run(hosttools.admin_set_person_role({"person": "alex", "role": "general"}, m, True))
    assert out.startswith("ERROR")
    assert "more than one person" in out


def test_admin_set_person_role_refuses_unknown_person(tmp_path):
    m = make_manager(tmp_path)
    m.store.set_role("general", [], note="")
    out = run(hosttools.admin_set_person_role({"person": "Nobody Here", "role": "general"}, m, True))
    assert out.startswith("ERROR")


def test_admin_set_person_role_refused_for_non_commander(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_set_person_role({"person": "Anyone", "role": "general"}, m, False))
    assert "commander" in out.lower()


# ---------------------------------------------------------------------------
# admin_schedule
# ---------------------------------------------------------------------------

def test_admin_schedule_pause_and_resume_happy_path(tmp_path):
    m = make_manager(tmp_path)
    schedule = _make_schedule(m.store)
    out = run(hosttools.admin_schedule({"action": "pause", "schedule": schedule.title}, m, True))
    assert "->" in out
    assert m.store.get_schedule(schedule.id).state == "paused"
    out2 = run(hosttools.admin_schedule({"action": "resume", "schedule": str(schedule.id)}, m, True))
    assert "->" in out2
    assert m.store.get_schedule(schedule.id).state == "active"


def test_admin_schedule_refuses_unsupported_action(tmp_path):
    m = make_manager(tmp_path)
    schedule = _make_schedule(m.store)
    out = run(hosttools.admin_schedule({"action": "stop", "schedule": schedule.title}, m, True))
    assert out.startswith("ERROR")


def test_admin_schedule_refuses_unmatched_name(tmp_path):
    m = make_manager(tmp_path)
    _make_schedule(m.store)
    out = run(hosttools.admin_schedule({"action": "pause", "schedule": "no such schedule"}, m, True))
    assert out.startswith("ERROR")


def test_admin_schedule_refused_for_non_commander(tmp_path):
    m = make_manager(tmp_path)
    schedule = _make_schedule(m.store)
    out = run(hosttools.admin_schedule({"action": "pause", "schedule": schedule.title}, m, False))
    assert "commander" in out.lower()
    assert m.store.get_schedule(schedule.id).state == "active"   # nothing changed


# ---------------------------------------------------------------------------
# admin_set_chat_mode -- delegates to store.set_response_mode when it exists
# ---------------------------------------------------------------------------

def test_admin_set_chat_mode_happy_path_when_store_supports_it(tmp_path):
    m = make_manager(tmp_path)
    if not hasattr(m.store, "set_response_mode"):
        pytest.skip("per-chat response modes not in this store build")
    out = run(hosttools.admin_set_chat_mode({"chat": "chat-123", "mode": "mention"}, m, True))
    assert "->" in out
    assert m.store.get_response_mode("chat-123") == "mention"


def test_admin_set_chat_mode_refused_for_non_commander(tmp_path):
    m = make_manager(tmp_path)
    out = run(hosttools.admin_set_chat_mode({"chat": "chat-123", "mode": "mention"}, m, False))
    assert "commander" in out.lower()


def test_admin_set_chat_mode_gracefully_declines_when_store_lacks_it(tmp_path):
    m = make_manager(tmp_path)

    # A bare stand-in with no response-mode method at all, as if this landed
    # before the per-chat response-mode store methods did.
    class _NoModeStore:
        pass

    m.store = _NoModeStore()
    out = run(hosttools.admin_set_chat_mode({"chat": "chat-123", "mode": "mention"}, m, True))
    assert "not available" in out.lower() or "not in the console" in out.lower()


# ---------------------------------------------------------------------------
# The prompt clause: the commander exception is present and still bounded.
# ---------------------------------------------------------------------------

def test_executor_system_prompt_has_commander_exception():
    from inkbox_claude.gate import executor
    assert "Commander exception" in executor.EXECUTOR_SYSTEM
    assert "admin_" in executor.EXECUTOR_SYSTEM
    assert "never reveals a token" in executor.EXECUTOR_SYSTEM
    assert "Commander exception" in executor.SCHEDULE_EXECUTOR_SYSTEM


def test_admin_tool_names_match_the_host_prefix_executor_gates_on():
    for name in BLATBOT_ADMIN_TOOLS:
        assert name.startswith("mcp__host__admin_")
