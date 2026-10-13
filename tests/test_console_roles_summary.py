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
    store.set_role("custom-role", ["inbox_read"])  # blatbot_inkbox read set is {inbox_read, contacts}

    summary = api.roles_summary(store)
    by_name = {r["name"]: r for r in summary["roles"]}

    def level(role_name, system):
        return next(sy for sy in by_name[role_name]["systems"] if sy["system"] == system)["level"]

    assert level("nothing", "tamid_workspace") == "none"
    assert level("reader", "tamid_workspace") == "read"
    assert level("changer", "tamid_workspace") == "change"
    assert level("custom-role", "blatbot_inkbox") == "custom"
    custom_sys = next(sy for sy in by_name["custom-role"]["systems"] if sy["system"] == "blatbot_inkbox")
    assert custom_sys["phrases"] == [ASK["inbox_read"]]


def test_set_level_read_grants_exactly_read_set_and_leaves_other_systems_alone(tmp_path):
    store = make_store(tmp_path)
    store.set_role("r1", ["stern_calendar", "brightspace_read", "brightspace_act"])
    result = api.roles_set_level(store, {"role": "r1", "system": "aaron_stern", "level": "read"})
    row = next(r for r in store.roles() if r["name"] == "r1")
    assert set(row["scopes"]) == {"stern_calendar", "stern_email_read", "brightspace_read", "brightspace_act"}
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
    assert sy["level"] == "read"
    result = api.roles_set_send(store, {"role": "r3", "system": "blatbot_inkbox", "on": False})
    sy = next(s for s in result["systems"] if s["system"] == "blatbot_inkbox")
    assert sy["send"] is False
    assert sy["level"] == "read"


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
