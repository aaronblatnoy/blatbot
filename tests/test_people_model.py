"""Tests for the person-centric trust model: persons, contacts, resolution,
explicit linking, migration from the legacy trust/people/synced_people tables,
and the new console/api endpoints. No network."""

from __future__ import annotations

import json
import tempfile
import time
import os

import pytest

from inkbox_claude.gate.store import Store
from inkbox_claude.console import api as console_api


@pytest.fixture
def store():
    path = tempfile.mktemp(suffix=".db")
    s = Store(path)
    yield s
    try:
        os.remove(path)
    except OSError:
        pass


def test_one_person_several_contacts_any_resolves_to_same_role(store):
    store.set_role("partner", ["calendar"], "")
    p = store.create_person(display="Sam Parker", role="partner", scopes=["tamid_drive_read"])
    store.add_contact(p["id"], "telegram", "telegram:5550100001")
    store.add_contact(p["id"], "email", "sean@example.com")
    store.add_contact(p["id"], "phone", "+1 (555) 010-0001")

    for handle in ("telegram:5550100001", "sean@example.com", "5550100001", "+15550100001"):
        result = store.trust_for([handle])
        assert result["role"] == "partner"
        assert result["person"] == "Sam Parker"
        assert set(result["scopes"]) == {"calendar", "tamid_drive_read"}


def test_two_different_people_on_one_message_grants_neither(store):
    p1 = store.create_person(display="A", role="", scopes=["calendar"])
    p2 = store.create_person(display="B", role="", scopes=["sjba_site_read"])
    store.add_contact(p1["id"], "email", "a@example.com")
    store.add_contact(p2["id"], "phone", "+15555550000")

    result = store.resolve_handles(["a@example.com", "+15555550000"])
    assert result["conflict"] is True
    assert result["role"] == ""
    assert result["scopes"] == []
    assert result["person"] == ""

    # trust_for (what the gate actually calls) must also grant nothing.
    legacy = store.trust_for(["a@example.com", "+15555550000"])
    assert legacy["scopes"] == []
    assert legacy["role"] == ""


def test_unknown_handle_grants_nothing(store):
    result = store.trust_for(["nobody@example.com"])
    assert result == {"role": "", "scopes": [], "person": ""}


def test_linking_does_not_union_permissions(store):
    keep = store.create_person(display="Keep", role="partner", scopes=["calendar"])
    other = store.create_person(display="Other", role="", scopes=["sjba_site_write", "tamid_social"])
    store.add_contact(keep["id"], "email", "keep@example.com")
    store.add_contact(other["id"], "email", "other@example.com")

    result = store.link_people(keep["id"], other["id"])
    assert result["ok"] is True
    # The surviving person's role/scopes are untouched.
    kept = store.get_person(keep["id"])
    assert kept["role"] == "partner"
    assert kept["scopes"] == ["calendar"]
    # The other person's original scopes are preserved in the response for review,
    # not applied to anyone.
    assert set(result["merged_person_was"]["scopes"]) == {"sjba_site_write", "tamid_social"}

    # Both handles now resolve to the keeper's permissions only.
    for handle in ("keep@example.com", "other@example.com"):
        got = store.trust_for([handle])
        assert got["role"] == "partner"
        assert got["scopes"] == ["calendar"]


def test_add_contact_already_owned_reports_conflict_without_moving(store):
    p1 = store.create_person(display="A")
    p2 = store.create_person(display="B")
    store.add_contact(p1["id"], "email", "shared@example.com")
    result = store.upsert_contact("email", "shared@example.com", person_id=p2["id"], source="manual")
    assert result["ok"] is False
    assert result["conflict"] is True
    assert result["existing_person_id"] == p1["id"]
    # Still attached to p1, not moved.
    assert store.get_contact(result["contact"]["id"])["person_id"] == p1["id"]


def test_move_contact_explicit_reassignment(store):
    p1 = store.create_person(display="A")
    p2 = store.create_person(display="B", role="partner", scopes=["calendar"])
    store.set_role("partner", [], "")
    added = store.add_contact(p1["id"], "email", "x@example.com")
    contact_id = added["contact"]["id"]
    store.move_contact(contact_id, p2["id"])
    assert store.get_contact(contact_id)["person_id"] == p2["id"]
    result = store.trust_for(["x@example.com"])
    assert result["person"] == "B"


def test_sync_never_moves_a_human_assigned_contact(store):
    p = store.create_person(display="Human Assigned")
    store.add_contact(p["id"], "email", "fixed@example.com", source="manual")
    # A sync pass sees the same handle again with no person_id -- must not detach it.
    store.upsert_contact("email", "fixed@example.com", person_id=None, source="sync")
    assert store.get_contact(store.find_contact("email", "fixed@example.com")["id"])["person_id"] == p["id"]


def test_sync_creates_unlinked_contact_for_new_handle(store):
    before = len(store.unlinked_contacts())
    store.upsert_contact("phone", "+15551230000", person_id=None, source="sync")
    after = store.unlinked_contacts()
    assert len(after) == before + 1
    assert any(c["value"].endswith("1230000") or "1230000" in c["value"] for c in after)


def test_sync_run_twice_is_idempotent_in_row_counts(store):
    rows = [{"key": "person@example.com", "person": "Person", "channels": ["email"], "last_seen": 1.0}]

    class FakeClient:
        def contacts(self):
            return rows

    from inkbox_claude.console import sync as console_sync
    console_sync.sync_people(store, FakeClient())
    n1 = len(store.unlinked_contacts())
    console_sync.sync_people(store, FakeClient())
    n2 = len(store.unlinked_contacts())
    assert n1 == n2 == 1


def test_migration_from_legacy_fixture_tables():
    path = tempfile.mktemp(suffix=".db")
    import sqlite3
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE trust (key TEXT PRIMARY KEY, person TEXT, role TEXT, scopes TEXT, note TEXT, updated_at REAL);
        CREATE TABLE people (person_id TEXT, key TEXT, display TEXT, updated_at REAL, PRIMARY KEY(person_id,key));
        CREATE TABLE synced_people (key TEXT PRIMARY KEY, person TEXT, channels TEXT, last_seen REAL);
    """)
    conn.execute("INSERT INTO trust VALUES ('telegram:111','Sam Parker','partner','[\"calendar\"]','',0)")
    conn.execute("INSERT INTO people VALUES ('ib1','telegram:111','Sam Parker',0)")
    conn.execute("INSERT INTO people VALUES ('ib1','sean@example.com','Sam Parker',0)")
    conn.execute("INSERT INTO people VALUES ('ib2','someone@else.com','Someone Else',0)")
    conn.execute("INSERT INTO synced_people VALUES ('sean@example.com','Sam Parker','[\"email\"]',5)")
    conn.execute("INSERT INTO synced_people VALUES ('neverlinked@example.com','Stranger','[\"email\"]',9)")
    conn.commit()
    conn.close()

    s = Store(path)  # migration runs in __init__
    people_rows = s.list_people()
    unlinked = s.unlinked_contacts()

    sean = next(p for p in people_rows if p["display"] == "Sam Parker")
    assert {c["value"] for c in sean["contacts"]} == {"111", "sean@example.com"}
    assert sean["role"] == "partner"

    someone = next(p for p in people_rows if p["display"] == "Someone Else")
    assert {c["value"] for c in someone["contacts"]} == {"someone@else.com"}

    assert any(c["value"] == "neverlinked@example.com" for c in unlinked)
    os.remove(path)


def test_migration_is_idempotent_rerun():
    path = tempfile.mktemp(suffix=".db")
    s1 = Store(path)
    s1.set_trust("a@example.com", person="A", role="", scopes=[], note="")
    before = len(s1.list_people())
    # Re-open the same file: migration must be a no-op the second time.
    s2 = Store(path)
    after = len(s2.list_people())
    assert before == after
    os.remove(path)


# -- console/api endpoints ---------------------------------------------------

def test_api_create_update_remove_person(store):
    created = console_api.create_person_endpoint(store, {"display": "New", "role": "", "note": ""})
    assert created["display"] == "New"
    updated = console_api.update_person_endpoint(store, {"id": created["id"], "role": "partner"})
    assert updated["role"] == "partner"
    removed = console_api.remove_person_endpoint(store, {"id": created["id"]})
    assert removed["ok"] is True
    assert store.get_person(created["id"]) is None


def test_api_add_move_remove_contact(store):
    p1 = console_api.create_person_endpoint(store, {"display": "A"})
    p2 = console_api.create_person_endpoint(store, {"display": "B"})
    added = console_api.add_contact_endpoint(store, {"person_id": p1["id"], "kind": "email", "value": "x@example.com"})
    assert added["ok"] is True
    contact_id = added["contact"]["id"]
    moved = console_api.move_contact_endpoint(store, {"contact_id": contact_id, "person_id": p2["id"]})
    assert moved["contact"]["person_id"] == p2["id"]
    removed = console_api.remove_contact_endpoint(store, {"contact_id": contact_id})
    assert removed["ok"] is True
    assert store.get_contact(contact_id) is None


def test_api_add_contact_conflict_reported(store):
    p1 = console_api.create_person_endpoint(store, {"display": "A"})
    p2 = console_api.create_person_endpoint(store, {"display": "B"})
    console_api.add_contact_endpoint(store, {"person_id": p1["id"], "kind": "email", "value": "shared@example.com"})
    result = console_api.add_contact_endpoint(store, {"person_id": p2["id"], "kind": "email", "value": "shared@example.com"})
    assert result["ok"] is False
    assert result["conflict"] is True
    assert result["existing_person_id"] == p1["id"]


def test_api_link_people(store):
    p1 = console_api.create_person_endpoint(store, {"display": "Keep", "role": "partner", "scopes": []})
    p2 = console_api.create_person_endpoint(store, {"display": "Merge", "scopes": []})
    result = console_api.link_people_endpoint(store, {"keep_id": p1["id"], "merge_id": p2["id"]})
    assert result["ok"] is True
    directory = console_api.people_directory(store)
    ids = {p["id"] for p in directory["people"]}
    assert p1["id"] in ids
    assert p2["id"] not in ids


def test_api_people_directory_shape(store):
    p = console_api.create_person_endpoint(store, {"display": "A"})
    console_api.add_contact_endpoint(store, {"person_id": p["id"], "kind": "email", "value": "a@example.com"})
    store.upsert_contact("phone", "+15550000000", person_id=None, source="sync")
    directory = console_api.people_directory(store)
    assert any(person["id"] == p["id"] for person in directory["people"])
    assert any(c["value"].endswith("0000000") for c in directory["unlinked"])


def test_api_validation_errors(store):
    with pytest.raises(console_api.ValidationError):
        console_api.add_contact_endpoint(store, {"person_id": "nope", "kind": "bogus", "value": "x"})
    with pytest.raises(console_api.ValidationError):
        console_api.update_person_endpoint(store, {"id": "nope", "display": "x"})
    with pytest.raises(console_api.ValidationError):
        console_api.link_people_endpoint(store, {"keep_id": "nope", "merge_id": "alsonope"})


def test_permissions_never_resolve_from_a_name_or_across_kinds(tmp_path):
    from inkbox_claude.gate.store import Store
    store = Store(str(tmp_path / "gate.db"))
    store.set_role("board", ["calendar"])
    person = store.create_person(display="Pat Example", role="board")
    pid = person["id"]
    store.add_contact(pid, "telegram", "5550100001")
    store.add_contact(pid, "name", "Pat Example")
    store.add_contact(pid, "email", "pat@example.com")
    # the stamped Telegram id resolves; the same digits arriving as a phone number do not
    assert store.trust_for(["telegram:5550100001"])["role"] == "board"
    assert store.trust_for(["+1 (651) 231-9697"])["role"] == ""
    assert store.trust_for(["5550100001"])["role"] == ""
    # a display name is typed by the sender and proves nothing
    assert store.trust_for(["Pat Example"])["role"] == ""
    assert store.trust_for(["pat@example.com", "Pat Example"])["role"] == "board"


def test_migration_groups_two_trust_rows_sharing_a_display_name_into_one_person():
    """The exact live shape this was built for: Sam Parker had two OLD one-row-per-
    handle trust rows (phone, telegram), same display name, same comma-joined roles.
    The migration must fold them into ONE person with both contacts and both roles,
    not two persons who happen to share a name."""
    path = tempfile.mktemp(suffix=".db")
    import sqlite3
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE trust (key TEXT PRIMARY KEY, person TEXT, role TEXT, scopes TEXT, note TEXT, updated_at REAL);
        CREATE TABLE people (person_id TEXT, key TEXT, display TEXT, updated_at REAL, PRIMARY KEY(person_id,key));
        CREATE TABLE synced_people (key TEXT PRIMARY KEY, person TEXT, channels TEXT, last_seen REAL);
        CREATE TABLE roles (name TEXT PRIMARY KEY, scopes TEXT, note TEXT, updated_at REAL);
    """)
    conn.execute(
        "INSERT INTO trust VALUES ('5550100001','Sam Parker','partner, tamid board member','[]','',0)"
    )
    conn.execute(
        "INSERT INTO trust VALUES ('telegram:7000000001','Sam Parker','partner, tamid board member','[]','',0)"
    )
    conn.execute("INSERT INTO roles VALUES ('partner','[\"pension_filings_read\"]','',0)")
    conn.execute("INSERT INTO roles VALUES ('tamid board member','[]','',0)")
    conn.commit()
    conn.close()

    s = Store(path)
    people_rows = s.list_people()
    seans = [p for p in people_rows if p["display"] == "Sam Parker"]
    assert len(seans) == 1
    sean = seans[0]
    assert {c["value"] for c in sean["contacts"]} == {"5550100001", "7000000001"}
    assert {c["kind"] for c in sean["contacts"]} == {"phone", "telegram"}
    assert set(sean["role"].split(", ")) == {"partner", "tamid board member"}

    assert s.trust_for(["5550100001"])["scopes"] == ["pension_filings_read"]
    assert s.trust_for(["telegram:7000000001"])["scopes"] == ["pension_filings_read"]

    # Idempotent re-open does not re-split or duplicate the person.
    s2 = Store(path)
    assert len([p for p in s2.list_people() if p["display"] == "Sam Parker"]) == 1
    os.remove(path)


def test_person_can_hold_several_roles_and_either_contact_resolves_them(store):
    store.set_role("partner", ["pension_filings_read"], "")
    store.set_role("tamid board member", [], "")
    store.set_trust("5550100001", person="Sam Parker", role="partner, tamid board member")
    pid = store.find_contact("phone", "5550100001")["person_id"]
    store.add_contact(pid, "telegram", "7000000001")

    assert store.trust_for(["5550100001"])["scopes"] == ["pension_filings_read"]
    assert store.trust_for(["telegram:7000000001"])["scopes"] == ["pension_filings_read"]
    person = store.get_person(pid)
    assert set(person["role"].split(", ")) == {"partner", "tamid board member"}


def test_remove_one_role_leaves_the_others_and_other_contacts_alone(store):
    store.set_role("partner", ["pension_filings_read"], "")
    store.set_role("tamid board member", [], "")
    store.set_trust("5550100001", person="Sam Parker", role="partner, tamid board member")
    pid = store.find_contact("phone", "5550100001")["person_id"]
    store.add_contact(pid, "telegram", "7000000001")

    console_api.remove_role_member(store, {"name": "partner", "key": "telegram:7000000001"})

    person = store.get_person(pid)
    assert person["role"] == "tamid board member"
    assert {c["value"] for c in person["contacts"]} == {"5550100001", "7000000001"}


# -- scope breakdown and per-role scope add/remove ---------------------------

def test_add_and_remove_role_scope(store):
    store.set_role("ops", ["inbox_read"], note="n")
    role = console_api.add_role_scope(store, {"name": "ops", "scope": "calendar"})
    assert set(role["scopes"]) == {"inbox_read", "calendar"}
    assert role["note"] == "n"

    role = console_api.remove_role_scope(store, {"name": "ops", "scope": "inbox_read"})
    assert role["scopes"] == ["calendar"]


def test_add_role_scope_rejects_unknown_scope_or_role(store):
    store.set_role("ops", [])
    with pytest.raises(console_api.ValidationError):
        console_api.add_role_scope(store, {"name": "ops", "scope": "not_a_real_scope"})
    with pytest.raises(console_api.ValidationError):
        console_api.add_role_scope(store, {"name": "ghost", "scope": "inbox_read"})


def test_scopes_breakdown_groups_by_system_and_shows_roles_holding_each(store):
    store.set_role("ops", ["inbox_read"])

    flat = console_api.scopes(store)
    inbox = next(s for s in flat if s["name"] == "inbox_read")
    assert inbox["read"] is True
    assert "ops" in inbox["roles"]
    assert isinstance(inbox["tools"], list) and inbox["tools"]
    assert inbox["description"]

    breakdown = console_api.scopes_breakdown(store)
    assert isinstance(breakdown, list) and breakdown
    all_names = {s["name"] for group in breakdown for s in group["scopes"]}
    assert "inbox_read" in all_names
    for group in breakdown:
        for s in group["scopes"]:
            assert s["group"] == group["system"]


def test_scopes_without_a_store_still_works_but_has_no_roles():
    flat = console_api.scopes()
    assert flat and all(s["roles"] == [] for s in flat)


def test_roles_page_and_people_list_show_sean_once_not_twice(store):
    """The Roles page member list, its people_count, and the flat people() list must
    all show one entry for a person with two contacts -- never one row per contact."""
    store.set_role("partner", ["pension_filings_read"], "")
    store.set_role("tamid board member", [], "")
    store.set_trust("5550100001", person="Sam Parker", role="partner, tamid board member")
    pid = store.find_contact("phone", "5550100001")["person_id"]
    store.add_contact(pid, "telegram", "7000000001")

    roles = console_api.roles(store)
    by_name = {r["name"]: r for r in roles}
    assert by_name["partner"]["people_count"] == 1
    assert by_name["tamid board member"]["people_count"] == 1
    assert set(by_name["partner"]["members"][0]["keys"]) == {"5550100001", "7000000001"}

    people = console_api.people(store)
    seans = [p for p in people if p.get("person") == "Sam Parker"]
    assert len(seans) == 1
    assert set(seans[0]["keys"]) == {"5550100001", "7000000001"}
