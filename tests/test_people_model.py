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
    assert {c["value"] for c in sean["contacts"]} == {"telegram:111", "sean@example.com"}
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
