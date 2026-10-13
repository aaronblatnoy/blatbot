"""Tests for the roles-summary view (GET /console/api/roles/summary and friends):
the plain-English "who can do what" model that replaces raw scope names on the
Roles page. Uses only a throwaway sqlite Store -- no live gate.db, no network."""

from __future__ import annotations

import re

import pytest

from inkbox_claude.console import api
from inkbox_claude.gate.scopes import ASK, SCOPES
from inkbox_claude.gate.store import Store

_SCOPE_SHAPED = re.compile(r"(_read\b|_write\b|_send\b|_control\b|vault:)")


def make_store(tmp_path):
    return Store(str(tmp_path / "console.db"))


def _all_text_fields(summary):
    texts = []
    for role in summary["roles"]:
        texts.append(role["can_text"])
        texts.append(role["description_text"])
        texts.append(role["notes"]["text"])
        for sy in role["systems"]:
            texts.append(sy["system_label"])
            texts.append(sy["level_label"])
            if sy["send_label"]:
                texts.append(sy["send_label"])
            texts.extend(sy["phrases"])
        for f in role["notes"]["folders"]:
            texts.append(f["label"])
    for sy in summary["systems"]:
        texts.append(sy["system_label"])
        if sy["send_label"]:
            texts.append(sy["send_label"])
        for lv in sy["levels"]:
            texts.append(lv["label"])
    for f in summary["note_folders"]:
        texts.append(f["label"])
    return texts


def test_no_scope_name_shaped_text_anywhere(tmp_path):
    store = make_store(tmp_path)
    store.set_role("comms", ["inbox_read", "email_send", "sms_send", "contacts"], note="")
    store.set_trust("mia@example.edu", person="Mia", role="comms")
    summary = api.roles_summary(store)
    for text in _all_text_fields(summary):
        assert not _SCOPE_SHAPED.search(text or ""), text


def test_every_scope_has_a_unique_nonempty_ask():
    asks = {}
    for name in SCOPES:
        phrase = ASK.get(name)
        if phrase is None and name.startswith("vault:"):
            folder = name.split(":", 1)[1].replace("/", " / ")
            phrase = f"read your notes on {folder}"
        assert phrase, f"scope {name} has no ask phrase"
        assert phrase not in asks.values() or asks.get(phrase) == name, (
            f"scope {name} shares an ask phrase with {[k for k, v in asks.items() if v == phrase]}"
        )
        asks[phrase] = name


def test_level_derivation_none_read_change_custom(tmp_path):
    store = make_store(tmp_path)
    store.set_role("nothing", [])
    store.set_role("reader", ["tamid_drive_read"])
    store.set_role("changer", ["tamid_drive_read", "tamid_drive_write"])
    store.set_role("custom-role", ["contacts"])  # blatbot_inkbox read set is {inbox_read}; contacts changes things

    summary = api.roles_summary(store)
    by_name = {r["name"]: r for r in summary["roles"]}

    def level(role_name, system):
        return next(sy for sy in by_name[role_name]["systems"] if sy["system"] == system)["level"]

    assert level("nothing", "tamid_workspace") == "none"
    assert level("reader", "tamid_workspace") == "read"
    assert level("changer", "tamid_workspace") == "change"
    assert level("custom-role", "blatbot_inkbox") == "custom"
    custom_sys = next(sy for sy in by_name["custom-role"]["systems"] if sy["system"] == "blatbot_inkbox")
    assert custom_sys["phrases"] == [ASK["contacts"]]


def test_set_level_read_grants_exactly_read_set_and_leaves_other_systems_alone(tmp_path):
    store = make_store(tmp_path)
    store.set_role("r1", ["stern_calendar", "brightspace_read", "brightspace_act"])
    result = api.roles_set_level(store, {"role": "r1", "system": "aaron_stern", "level": "read"})
    row = next(r for r in store.roles() if r["name"] == "r1")
    # stern_calendar carries manage_event (a change tool), so "Read" no longer
    # keeps it -- selecting Read must not silently also grant change. The
    # brightspace scopes are a different system (nyu_brightspace); this call
    # only ever touches aaron_stern's own scopes.
    assert set(row["scopes"]) == {"stern_email_read", "brightspace_read", "brightspace_act"}
    sy = next(s for s in result["systems"] if s["system"] == "aaron_stern")
    assert sy["level"] == "read"
    brightspace_sy = next(s for s in result["systems"] if s["system"] == "nyu_brightspace")
    assert brightspace_sy["level"] == "change"


def test_set_level_change_then_none_removes_scopes(tmp_path):
    store = make_store(tmp_path)
    store.set_role("r2", [])
    api.roles_set_level(store, {"role": "r2", "system": "tamid_workspace", "level": "change"})
    row = next(r for r in store.roles() if r["name"] == "r2")
    assert set(row["scopes"]) == {"tamid_drive_read", "tamid_drive_write"}
    api.roles_set_level(store, {"role": "r2", "system": "tamid_workspace", "level": "none"})
    row = next(r for r in store.roles() if r["name"] == "r2")
    assert "tamid_drive_read" not in row["scopes"]
    assert "tamid_drive_write" not in row["scopes"]


def test_set_send_is_independent_of_level(tmp_path):
    store = make_store(tmp_path)
    store.set_role("r3", ["inbox_read", "contacts"])
    result = api.roles_set_send(store, {"role": "r3", "system": "blatbot_inkbox", "on": True})
    sy = next(s for s in result["systems"] if s["system"] == "blatbot_inkbox")
    assert sy["send"] is True
    # inbox_read + contacts is this system is whole read+change set now that
    # contacts (a create/update tool) no longer silently counts as read.
    assert sy["level"] == "change"
    result = api.roles_set_send(store, {"role": "r3", "system": "blatbot_inkbox", "on": False})
    sy = next(s for s in result["systems"] if s["system"] == "blatbot_inkbox")
    assert sy["send"] is False
    # level is unaffected by the send toggle either way: inbox_read + contacts was
    # already this system's whole read+change set before, during and after it.
    assert sy["level"] == "change"


def test_set_notes_none_some_all(tmp_path):
    store = make_store(tmp_path)
    store.set_role("r4", [])
    result = api.roles_set_notes(store, {"role": "r4", "level": "all", "folders": []})
    assert result["notes"]["level"] == "all"
    row = next(r for r in store.roles() if r["name"] == "r4")
    assert "vault_read" in row["scopes"]

    result = api.roles_set_notes(store, {"role": "r4", "level": "none", "folders": []})
    assert result["notes"]["level"] == "none"
    row = next(r for r in store.roles() if r["name"] == "r4")
    assert "vault_read" not in row["scopes"]


def test_locked_role_rejects_all_three_write_endpoints(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(api.ValidationError):
        api.roles_set_level(store, {"role": "commander", "system": "tamid_workspace", "level": "read"})
    with pytest.raises(api.ValidationError):
        api.roles_set_send(store, {"role": "commander", "system": "blatbot_inkbox", "on": True})
    with pytest.raises(api.ValidationError):
        api.roles_set_notes(store, {"role": "commander", "level": "all", "folders": []})


def test_unknown_role_system_level_return_plain_sentence_errors(tmp_path):
    store = make_store(tmp_path)
    store.set_role("r5", [])
    with pytest.raises(api.ValidationError):
        api.roles_set_level(store, {"role": "ghost", "system": "tamid_workspace", "level": "read"})
    with pytest.raises(api.ValidationError):
        api.roles_set_level(store, {"role": "r5", "system": "not-a-system", "level": "read"})
    with pytest.raises(api.ValidationError):
        api.roles_set_level(store, {"role": "r5", "system": "tamid_workspace", "level": "not-a-level"})


def test_who_can_matches_phrase_system_label_and_empty_query(tmp_path):
    store = make_store(tmp_path)
    store.set_role("comms", ["email_send"])
    store.set_trust("mia@example.edu", person="Mia", role="comms")

    by_phrase = api.roles_who_can(store, "send email")
    names = {m["name"] for match in by_phrase["matches"] for m in match["roles"]}
    assert "comms" in names
    member_names = {
        person["name"]
        for match in by_phrase["matches"]
        for role in match["roles"]
        for person in role["members"]
    }
    assert "Mia" in member_names

    by_system_label = api.roles_who_can(store, "inbox")
    assert any("comms" == m["name"] for match in by_system_label["matches"] for m in match["roles"])

    empty = api.roles_who_can(store, "")
    assert empty["matches"] == []

    no_match = api.roles_who_can(store, "zzz-nonexistent-phrase-zzz")
    assert no_match["matches"] == []


def test_origin_gate_rejects_bad_or_missing_origin():
    from aiohttp import web

    from inkbox_claude.console import access, routes

    app = web.Application()
    request = type(
        "FakeRequest",
        (),
        {"headers": {"Origin": "https://evil.example.com"}, "host": "console.example.org"},
    )()
    with pytest.raises(web.HTTPForbidden):
        routes._same_origin(request)

    request_missing = type("FakeRequest2", (), {"headers": {}, "host": "console.example.org"})()
    with pytest.raises(web.HTTPForbidden):
        routes._same_origin(request_missing)


def test_person_can_text_is_union_of_role_phrases(tmp_path):
    store = make_store(tmp_path)
    store.set_role("a", ["inbox_read"])
    store.set_role("b", ["stern_calendar"])
    person = store.create_person(display="Test Person", role="a, b", scopes=[], note="")
    model = api.people_directory(store)
    row = next(p for p in model["people"] if p["id"] == person["id"])
    assert "can_text" in row
    assert ASK["inbox_read"] in row["can_text"]
    assert ASK["stern_calendar"] in row["can_text"]
    assert "role_badges" in row
    names = {r["name"] for r in row["role_badges"]}
    assert {"a", "b"} <= names


# ---------------------------------------------------------------------------
# Follow-up: actions list, delete preview, singular/plural, custom phrases
# ---------------------------------------------------------------------------

def test_role_actions_locked_role_has_no_rename_or_delete(tmp_path):
    store = make_store(tmp_path)
    store.set_role("ops", ["inbox_read"])
    summary = api.roles_summary(store)
    by_name = {r["name"]: r for r in summary["roles"]}

    assert "rename" in by_name["ops"]["actions"]
    assert "delete" in by_name["ops"]["actions"]

    commander = by_name["commander"]
    assert "delete" not in commander["actions"]
    assert "rename" not in commander["actions"]


def test_delete_preview_singular_and_plural_and_locked_and_unknown(tmp_path):
    store = make_store(tmp_path)
    store.set_role("solo", ["inbox_read"])
    store.set_trust("mia@example.edu", person="Mia", role="solo")
    preview = api.roles_delete_preview(store, "solo")
    assert preview["member_count"] == 1
    assert "1 person" in preview["text"]
    assert "1 people" not in preview["text"]
    assert "They keep their other roles." in preview["text"]

    store.set_role("duo", ["inbox_read"])
    store.set_trust("mia@example.edu", person="Mia", role="duo")
    store.set_trust("theo@example.edu", person="Theo", role="duo")
    preview2 = api.roles_delete_preview(store, "duo")
    assert preview2["member_count"] == 2
    assert "2 people" in preview2["text"]

    with pytest.raises(api.ValidationError):
        api.roles_delete_preview(store, "commander")
    with pytest.raises(api.ValidationError):
        api.roles_delete_preview(store, "no-such-role")


def test_affects_text_uses_singular_person_for_one_member(tmp_path):
    store = make_store(tmp_path)
    store.set_role("solo2", [])
    store.set_trust("mia@example.edu", person="Mia", role="solo2")
    result = api.roles_set_level(store, {"role": "solo2", "system": "tamid_workspace", "level": "read"})
    assert "1 person" in result["affects_text"]
    assert "1 people" not in result["affects_text"]


def test_sources_have_no_file_paths_or_table_names_and_include_counts(tmp_path):
    store = make_store(tmp_path)
    store.set_role("ops2", ["inbox_read"])
    store.set_trust("mia@example.edu", person="Mia", role="ops2")
    summary = api.roles_summary(store)
    for source in summary["sources"]:
        detail = source["detail"]
        assert ".py" not in detail
        assert ".yaml" not in detail
        assert "table" not in detail.lower()
        assert any(ch.isdigit() for ch in detail), detail


def test_custom_level_phrases_are_nonempty_and_scope_name_free(tmp_path):
    # blatbot_inkbox's read set is {inbox_read} now that contacts (a create/update
    # tool) no longer silently counts as read; holding just contacts is a genuine
    # partial slice of that system's read+change set, so it reads as "custom".
    store = make_store(tmp_path)
    store.set_role("partial", ["contacts"])
    summary = api.roles_summary(store)
    role = next(r for r in summary["roles"] if r["name"] == "partial")
    sy = next(s for s in role["systems"] if s["system"] == "blatbot_inkbox")
    assert sy["level"] == "custom"
    assert sy["phrases"], "custom level must list what it currently holds"
    for phrase in sy["phrases"]:
        assert not _SCOPE_SHAPED.search(phrase), phrase


_CHANGING_TOOL_SHAPE = re.compile(
    r"(^|_)(create|update|delete|manage|send|write)(_|$)", re.IGNORECASE)
# Not "post": this registry has a legitimate read-only analytics tool named
# ig_post_insights (reads how a past post performed), which "post" alone as a
# trigger word would wrongly flag -- genuine posting tools are always under a
# separate, not-read-flagged scope (tamid_instagram_post etc), never one this
# test would be checking in the first place.


def test_every_systems_read_set_contains_no_scope_with_a_changing_tool():
    """R2: a role at "Read" for any system must never silently also be able to
    change something. _system_scope_sets's "read" bucket is built only from
    wording.truly_read_only, so no scope in it should carry a tool whose name
    looks like create/update/delete/manage/send/post/write."""
    sets_by_system = api._system_scope_sets()
    for system, sets in sets_by_system.items():
        for scope in sets["read"]:
            tools = [str(t).rsplit("__", 1)[-1] for t in (SCOPES.get(scope) or {}).get("tools") or []]
            bad = [t for t in tools if _CHANGING_TOOL_SHAPE.search(t)]
            assert not bad, f"{system}/{scope} is in the read set but has changing tool(s): {bad}"


def test_every_system_with_a_changing_scope_offers_read_and_change():
    """R2: a system with anything in its change set must offer the "Read and
    change" level -- never an implicit, unlabeled escalation."""
    sets_by_system = api._system_scope_sets()
    for system, sets in sets_by_system.items():
        if sets["change"]:
            levels = {lv["value"] for lv in api._system_level_labels(sets)}
            assert "change" in levels, f"{system} has change scopes but no 'change' level offered"


def test_tamid_calendar_scope_is_in_change_not_read():
    """The specific case the coordinator named: the TAMID calendar system's one
    scope carries manage_event, so it must never be silently granted by "Read"."""
    sets_by_system = api._system_scope_sets()
    tamid_calendar_sets = sets_by_system.get("tamid_calendar")
    assert tamid_calendar_sets is not None
    assert "calendar" not in tamid_calendar_sets["read"]
    assert "calendar" in tamid_calendar_sets["change"]
