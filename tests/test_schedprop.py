"""schedprop.py: turning the owner's words into a schedule proposal, and the
scopes a proposal gets -- chosen deterministically, never by Jev."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate import schedprop
from inkbox_claude.gate.router import RouterOutput, RouterRequest, ScheduleProposal
from inkbox_claude.gate.scopes import SCOPES


class _FakeRouter:
    def __init__(self):
        self.next = RouterOutput(reply=None, request=None)

    async def plan_query(self, **kwargs):
        return None

    async def route(self, **kwargs):
        return self.next.model_copy(deep=True)

    async def phrase(self, instruction, facts):
        return "Before I do this, I need your yes: remove the named item."


class _FakeExecutor:
    def __init__(self, result=None):
        self.result = result or {"ok": True, "summary": "looked it up", "raw": "looked it up",
                                 "tool_calls": ["mcp__x__read"], "wrote": False}
        self.ran = []

    async def run(self, req, **kwargs):
        self.ran.append((req, kwargs))
        return dict(self.result)


def manager(tmp_path, result=None):
    """A real GateSessionManager with only the router and executor faked -- the
    manager block under test (the ScheduleProposal -> schedprop.capabilities_for
    -> schedules.propose path) is exercised for real."""
    sent = []

    async def send_fn(target, text, mode, meta):
        sent.append((target, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id="conv-aaron", approver_phone="")
    out = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                                store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    out.router = _FakeRouter()
    out.executor = _FakeExecutor(result)
    out.task_picker = None
    return out, sent


class RecordingSchedules:
    """Stand-in for ScheduleService.propose, so these manager-block tests check
    exactly what manager.py passes it without also exercising the real store."""

    def __init__(self):
        self.calls = []

    async def propose(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id=len(self.calls), **kwargs)

    async def command(self, text: str, *, owner: bool) -> bool:
        return False


class RaisingScopePicker:
    """A Jev scope picker that fails the test if it is ever consulted: schedule
    permission is chosen with no probability judgment at all (SPEC.md's
    "Permission" section)."""

    async def judge_scopes_tree(self, **kwargs):
        raise AssertionError("Jev must never be consulted on a schedule code path")


def new_proposal(cadence_kind="once", **changes):
    spec = {"title": "Scheduled check", "prompt": "Check the roster", "cadence_kind": cadence_kind}
    if cadence_kind == "once":
        spec["cadence"] = {"at": "2026-11-01T09:00:00"}
    elif cadence_kind == "every":
        spec["cadence"] = {"minutes": 240}
    elif cadence_kind == "daily":
        spec["cadence"] = {"time": "08:00"}
    spec.update(changes)
    return spec


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_every_scope_in_the_registry_has_a_label():
    for name in SCOPES:
        got = schedprop.label(name)
        assert got, f"{name} has no label"


def test_no_two_scopes_share_a_label():
    seen = {}
    for name in SCOPES:
        lbl = schedprop.label(name)
        assert lbl not in seen, f"{name} and {seen.get(lbl)} share the label {lbl!r}"
        seen[lbl] = name


def test_vault_scope_label_is_generated_from_its_folder_name():
    # A dynamically discovered vault knowledge scope has no entry in
    # scopes.yaml at all; it is not even guaranteed to exist right now, so the
    # label is tested directly rather than through a live SCOPES lookup.
    assert schedprop.label("vault:projects/widgets") == "read your notes on projects/widgets"


def test_labels_joins_one_two_and_three_or_more_with_commas_and_and():
    assert schedprop.labels(["web"]) == schedprop.label("web")
    assert schedprop.labels(["web", "contacts"]) == f"{schedprop.label('web')} and {schedprop.label('contacts')}"
    joined = schedprop.labels(["web", "contacts", "host_read"])
    assert joined == (f"{schedprop.label('web')}, {schedprop.label('contacts')}, "
                       f"and {schedprop.label('host_read')}")


# ---------------------------------------------------------------------------
# capabilities_for
# ---------------------------------------------------------------------------

def test_capabilities_for_includes_named_scopes():
    assert schedprop.capabilities_for("check it", ["web"], [], "") == ["web"]


def test_capabilities_for_matches_a_synonym_in_his_own_words():
    got = schedprop.capabilities_for("watch the page", [], [], "update the contact note if it changes")
    assert "contacts" in got


def test_capabilities_for_does_not_match_a_bare_common_word():
    assert schedprop.capabilities_for("watch the page", [], [], "the service") == []


def test_capabilities_for_matches_a_tool_named_literally_in_the_prompt():
    got = schedprop.capabilities_for("Then call inkbox_create_contact for the tracker.", [], [], "")
    assert "contacts" in got


def test_capabilities_for_includes_the_routers_suggested_scopes():
    assert schedprop.capabilities_for("check it", [], ["calendar"], "") == ["calendar"]


def test_capabilities_for_dedupes_and_orders_by_registry():
    got = schedprop.capabilities_for(
        "Then call inkbox_create_contact for the tracker.", ["web"], ["contacts"], "edit contacts please")
    assert got == [name for name in SCOPES if name in {"web", "contacts"}]


def test_capabilities_for_refuses_an_unknown_named_scope():
    with pytest.raises(schedprop.UnknownScopes) as exc:
        schedprop.capabilities_for("check it", ["not_a_real_scope"], [], "")
    assert exc.value.names == ["not_a_real_scope"]


def test_capabilities_for_refuses_an_unknown_suggested_scope():
    with pytest.raises(schedprop.UnknownScopes):
        schedprop.capabilities_for("check it", [], ["not_a_real_scope"], "")


def test_missing_capabilities_names_a_tool_named_prompt_scope_not_granted():
    prompt = "Then call inkbox_create_contact for the tracker."
    assert schedprop.missing_capabilities(prompt, []) == ["contacts"]
    assert schedprop.missing_capabilities(prompt, ["contacts"]) == []


# ---------------------------------------------------------------------------
# format_proposal / format_change
# ---------------------------------------------------------------------------

def _schedule(**kwargs):
    base = dict(title="Team total watch", cadence_kind="every", cadence={"minutes": 240},
                timezone="America/New_York", scopes=["web", "contacts"],
                report_every_time=False, until=None)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_format_proposal_matches_the_specs_example_shape(monkeypatch):
    monkeypatch.setattr(schedprop, "_cadence_text", lambda s: "every 4 hours")
    monkeypatch.setattr(schedprop, "_first_run_text", lambda s: "2:30 PM today")
    schedule = _schedule()
    text = schedprop.format_proposal(schedule)
    assert text == (
        'Set up "Team total watch": every 4 hours, starting 2:30 PM today. '
        'It will read public web pages and look up and edit contacts. '
        'You hear from it only when something changed or it could not finish. '
        'Reply yes to start, or tell me what to change.'
    )
    assert "cron" not in text
    assert "#" not in text
    assert "*" not in text
    assert "every\"" not in text and "'every'" not in text  # no raw cadence_kind word as a token


def test_format_proposal_every_time_reporting(monkeypatch):
    monkeypatch.setattr(schedprop, "_cadence_text", lambda s: "once")
    monkeypatch.setattr(schedprop, "_first_run_text", lambda s: "9:00 AM tomorrow")
    schedule = _schedule(cadence_kind="once", cadence={"at": "2026-11-01T09:00:00"}, report_every_time=True)
    text = schedprop.format_proposal(schedule)
    assert "You hear from it every time it runs." in text
    assert "only when something changed" not in text


def test_format_proposal_with_until_adds_the_date(monkeypatch):
    monkeypatch.setattr(schedprop, "_cadence_text", lambda s: "every 4 hours")
    monkeypatch.setattr(schedprop, "_first_run_text", lambda s: "2:30 PM today")
    monkeypatch.setattr(schedprop, "_until_text", lambda s: "November 1")
    schedule = _schedule(until=1793000000.0)
    text = schedprop.format_proposal(schedule)
    assert "until November 1" in text


def test_format_proposal_contains_no_scope_names():
    import unittest.mock as mock
    with mock.patch.object(schedprop, "_cadence_text", return_value="daily at 8:00 AM"), \
         mock.patch.object(schedprop, "_first_run_text", return_value="8:00 AM tomorrow"):
        # Scope identifiers (underscored, machine-shaped) must not leak into the
        # text, even though the plain-English label can of course use an
        # ordinary word that happens to also be part of a scope's name ("web").
        schedule = _schedule(cadence_kind="daily", cadence={"time": "08:00"},
                             scopes=["browser_act", "coolify_control"])
        text = schedprop.format_proposal(schedule)
    for scope_name in schedule.scopes:
        assert scope_name not in text


def test_format_change_cadence_only(monkeypatch):
    monkeypatch.setattr(schedprop, "_cadence_text", lambda s: "daily at 8:00 AM")
    monkeypatch.setattr(schedprop, "_first_run_text", lambda s: "8:00 AM tomorrow")
    schedule = _schedule(title="Team total watch", cadence_kind="daily", cadence={"time": "08:00"})
    text = schedprop.format_change(schedule, {"cadence_kind": "daily", "cadence": {"time": "08:00"}})
    assert text == ("Team total watch will run daily at 8:00 AM, next 8:00 AM tomorrow. "
                     "Reply yes to confirm.")


def test_format_change_added_capabilities_only(monkeypatch):
    schedule = _schedule(title="Team total watch")
    text = schedprop.format_change(schedule, {"scopes": ["contacts"]})
    assert text == "Team total watch will also be allowed to look up and edit contacts. Reply yes to confirm."


def test_format_change_both_cadence_and_capabilities(monkeypatch):
    monkeypatch.setattr(schedprop, "_cadence_text", lambda s: "daily at 8:00 AM")
    monkeypatch.setattr(schedprop, "_first_run_text", lambda s: "8:00 AM tomorrow")
    schedule = _schedule(title="Team total watch")
    text = schedprop.format_change(schedule, {"cadence_kind": "daily", "scopes": ["contacts"]})
    assert "Team total watch will run daily at 8:00 AM, next 8:00 AM tomorrow. Reply yes to confirm." in text
    assert "Team total watch will also be allowed to look up and edit contacts. Reply yes to confirm." in text


# ---------------------------------------------------------------------------
# Router schema
# ---------------------------------------------------------------------------

def test_router_schedule_proposal_accepts_the_new_fields():
    sp = ScheduleProposal(title="t", prompt="p", cadence_kind="every", cadence={"minutes": 240},
                          timezone="America/New_York", report_every_time=True, until="2026-12-01",
                          scopes=["web"], start_now=True)
    assert sp.cadence_kind == "every"
    assert sp.cadence == {"minutes": 240}


def test_router_schedule_proposal_rejects_the_old_kind_field():
    with pytest.raises(Exception):
        ScheduleProposal(title="t", prompt="p", kind="once", run_at="2026-11-01T09:00:00")


def test_router_schedule_proposal_rejects_an_unknown_cadence_kind():
    with pytest.raises(Exception):
        ScheduleProposal(title="t", prompt="p", cadence_kind="continue", cadence={})


# ---------------------------------------------------------------------------
# Manager block
# ---------------------------------------------------------------------------


def test_manager_calls_propose_with_the_new_keyword_arguments_and_origin_thread(tmp_path):
    m, _ = manager(tmp_path)
    m.schedules = RecordingSchedules()
    schedule = ScheduleProposal(**new_proposal("every", title="Team total watch",
                                               prompt="Check the page.", scopes=["web"], start_now=True))
    session = m.get("aaron-origin-route")

    async def decide(**kwargs):
        return RouterOutput(reply=None, task="new", task_title="Team total watch", schedule=schedule), None

    session.decide = decide
    m.task_picker = RaisingScopePicker()
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(session.handle_inbound("set it up", "imessage", meta))
    call = m.schedules.calls[0]
    assert call["title"] == "Team total watch"
    assert call["prompt"] == "Check the page."
    assert call["cadence_kind"] == "every"
    assert call["cadence"] == {"minutes": 240}
    assert call["start_now"] is True
    assert call["owner"] is True
    assert call["origin_mode"] == "imessage"
    # origin_chat_id carries the real address to reach him on later -- the
    # session's own chat_id and reply meta, not the owner-thread record_id (which
    # is not an address any channel can send to).
    import json
    origin = json.loads(call["origin_chat_id"])
    assert origin["chat_id"] == session.chat_id == "aaron-origin-route"
    assert origin["meta"] == meta


def test_manager_relays_a_propose_valueerror_as_the_reply(tmp_path):
    m, sent = manager(tmp_path)

    class FailingSchedules(RecordingSchedules):
        async def propose(self, **kwargs):
            raise ValueError("every must be at least 5 minutes")

    m.schedules = FailingSchedules()
    schedule = ScheduleProposal(**new_proposal("every", title="Too fast", prompt="Check it.", scopes=["web"]))
    session = m.get("aaron-fail-route")

    async def decide(**kwargs):
        return RouterOutput(reply=None, task="new", task_title="Too fast", schedule=schedule), None

    session.decide = decide
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(session.handle_inbound("every minute please", "imessage", meta))
    assert any("every must be at least 5 minutes" in row[1] for row in sent)


def test_manager_never_calls_the_jev_picker_for_a_schedule(tmp_path):
    m, _ = manager(tmp_path)
    m.schedules = RecordingSchedules()
    schedule = ScheduleProposal(**new_proposal("daily", title="Monday brief", prompt="Check it.", scopes=["web"]))
    session = m.get("aaron-no-jev-route")

    async def decide(**kwargs):
        return RouterOutput(reply=None, task="new", task_title="Monday brief", schedule=schedule), None

    session.decide = decide
    m.task_picker = RaisingScopePicker()  # raises AssertionError if ever called
    meta = {"sender": "+15550100001", "conversation_id": "conv-aaron"}
    asyncio.run(session.handle_inbound("set it up", "imessage", meta))
    assert len(m.schedules.calls) == 1



# ---------------------------------------------------------------------------
# capabilities_from_permission(): deterministic grant from his own permission
# words, never from the router's suggestion (item D)
# ---------------------------------------------------------------------------

def test_permission_grants_only_the_named_scope():
    got = schedprop.capabilities_from_permission("let it edit contacts")
    assert got == ["contacts"]


def test_permission_excludes_a_negated_clause_after_but():
    got = schedprop.capabilities_from_permission("let it edit contacts but don't send email")
    assert got == ["contacts"]
    assert "email_send" not in got


def test_permission_excludes_a_clause_with_no_permission_phrase():
    # "update the contact note" alone, without "let it"/"allow it to"/etc, is
    # not a permission grant -- unlike capabilities_for's proposal-time
    # matching, add_capability only ever widens on an explicit grant.
    assert schedprop.capabilities_from_permission("update the contact note") == []


def test_permission_vague_phrase_narrows_to_denied_scopes():
    got = schedprop.capabilities_from_permission("let it do that", denied_scopes=["contacts"])
    assert got == ["contacts"]


def test_permission_fix_that_narrows_to_denied_scopes_without_let_it():
    got = schedprop.capabilities_from_permission("fix that", denied_scopes=["contacts", "web"])
    assert got == [name for name in SCOPES if name in {"contacts", "web"}]


def test_permission_vague_phrase_with_no_prior_denial_grants_nothing():
    assert schedprop.capabilities_from_permission("let it do that", denied_scopes=None) == []


def test_every_scopes_own_label_grants_exactly_that_scope():
    # The recovery phrase the bot itself dictates ("Reply 'let it <label>'")
    # must work for every scope in the registry, including a discovered vault
    # knowledge scope that has no entry in scopes.yaml at all.
    from inkbox_claude.gate import scopes as scopes_mod
    scopes_mod.refresh_vault_scopes(force=True)
    assert any(scopes_mod.SCOPE_SYSTEM.get(n) == "vault" for n in scopes_mod.SCOPES), \
        "expected at least one discovered vault scope to test against"
    for name in scopes_mod.SCOPES:
        lbl = schedprop.label(name)
        got = schedprop.capabilities_from_permission(f"let it {lbl}")
        assert got == [name], f"scope {name!r} label {lbl!r} -> {got!r}"


def test_overlapping_labels_longest_match_wins():
    # "look up and edit contacts" (contacts) and a shorter label that is a
    # substring of it must not both match; only the longest wins.
    from inkbox_claude.gate.scopes import SCOPES
    assert "contacts" in SCOPES
    got = schedprop.capabilities_from_permission("let it look up and edit contacts")
    assert got == ["contacts"]


def test_permission_no_words_at_all_grants_nothing():
    assert schedprop.capabilities_from_permission("") == []
