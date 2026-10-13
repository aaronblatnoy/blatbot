"""The one-step repair of a schedule migrated from the old continue-kind model:
no timing, a prompt that still calls the removed schedule_continue tool, and a
prompt that names a tool outside its granted scopes. Against the real store
and the real schedprop module (not faked) so missing_capabilities and the
labels read as the owner would actually see them."""

from __future__ import annotations

import time

import pytest

from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.store import Store

STUCK_PROMPT = (
    "Check the team fundraising page every 4 hours and tell me the total.\n\n"
    "When done, call schedule_continue with delay_minutes 240 to run again.\n\n"
    "If the total changed, call inkbox_update_contact with the new total."
)

HEALTHY_PROMPT = "Check the team fundraising page and tell me the total."


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.notices = []

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule, text))


@pytest.fixture
def env(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    service = sched_mod.ScheduleService(manager)
    task = store.create_task("repair test")
    return store, manager, service, task["id"]


def _make_stuck(store, task_id, *, prompt=STUCK_PROMPT, scopes=("web",)):
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Team total watch", prompt=prompt,
        kind="", scopes=list(scopes), timezone="America/New_York", report_mode="changed",
        cadence_kind="", cadence={}, origin_mode="imessage", origin_chat_id="owner",
    )
    return store.update_schedule(schedule.id, state="paused", last_outcome="needs a cadence")


def test_plan_recovers_timing_capability_and_dead_tool(env):
    store, manager, service, task_id = env
    schedule = _make_stuck(store, task_id)
    plan = service.repair_plan(schedule.id)
    assert plan["needed"] is True
    assert plan["timing"] == {"cadence_kind": "every", "cadence": {"minutes": 240}, "text": "every 4 hours"}
    assert plan["add_capabilities"] == [{"scope": "contacts", "label": "look up and edit contacts"}]
    assert len(plan["prompt_changes"]) == 1
    assert "schedule_continue" in plan["prompt_changes"][0]["remove"]
    assert "schedule_continue" not in plan["new_prompt"]
    assert plan["sentence"] == (
        "It needs a timing (every 4 hours, from its old instructions), "
        "permission to look up and edit contacts, and its instructions mention "
        "a tool that no longer exists."
    )


def test_plan_with_only_missing_timing(env):
    store, manager, service, task_id = env
    schedule = _make_stuck(store, task_id, prompt=HEALTHY_PROMPT, scopes=("web",))
    plan = service.repair_plan(schedule.id)
    assert plan["needed"] is True
    assert plan["timing"] is None
    assert plan["add_capabilities"] == []
    assert plan["prompt_changes"] == []
    assert plan["sentence"] == "It needs a timing you need to choose."


def test_healthy_schedule_needs_no_repair(env):
    store, manager, service, task_id = env
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title="Fine already", prompt=HEALTHY_PROMPT,
        kind="every", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="every", cadence={"minutes": 240}, origin_mode="imessage", origin_chat_id="owner",
    )
    schedule = store.update_schedule(schedule.id, state="active", next_run=time.time() + 60)
    plan = service.repair_plan(schedule.id)
    assert plan == {"needed": False, "sentence": "", "timing": None, "add_capabilities": [],
                    "prompt_changes": [], "new_prompt": HEALTHY_PROMPT}


@pytest.mark.asyncio
async def test_apply_full_plan_proposes_then_activates(env):
    store, manager, service, task_id = env
    schedule = _make_stuck(store, task_id)
    result = await service.repair_apply(schedule.id, owner=True)
    assert result["needs_approval"] is True
    proposed = result["schedule"]
    assert proposed.state == "proposed"
    assert proposed.cadence_kind == "every"
    assert proposed.cadence["minutes"] == 240
    assert "contacts" in proposed.scopes
    assert "schedule_continue" not in proposed.prompt
    assert len(manager.notices) == 1

    before = time.time()
    activated = await service.decide(proposed.id, True)
    assert activated.state == "active"
    assert activated.cadence_kind == "every"
    assert "contacts" in activated.scopes
    assert activated.next_run is not None
    assert before + 239 * 60 <= activated.next_run <= before + 241 * 60


@pytest.mark.asyncio
async def test_apply_timing_only_resumes_at_once(env):
    store, manager, service, task_id = env
    schedule = _make_stuck(store, task_id, prompt=HEALTHY_PROMPT, scopes=("web",))
    result = await service.repair_apply(
        schedule.id, owner=True, timing={"cadence_kind": "every", "cadence": {"minutes": 240}})
    assert result["needs_approval"] is False
    resumed = result["schedule"]
    assert resumed.state == "active"
    assert resumed.cadence_kind == "every"
    assert resumed.next_run is not None


@pytest.mark.asyncio
async def test_apply_refuses_when_timing_missing_and_none_given(env):
    store, manager, service, task_id = env
    schedule = _make_stuck(store, task_id, prompt="Do a thing with no stated timing anywhere.",
                           scopes=("web",))
    with pytest.raises(ValueError, match="timing"):
        await service.repair_apply(schedule.id, owner=True)


@pytest.mark.asyncio
async def test_apply_refused_for_non_owner(env):
    store, manager, service, task_id = env
    schedule = _make_stuck(store, task_id)
    with pytest.raises(PermissionError):
        await service.repair_apply(schedule.id, owner=False)
