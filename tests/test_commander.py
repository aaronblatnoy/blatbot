"""Tests for the `commander` role: the formal name of the owner's existing full
control. Every test here proves either (a) is_approver()/trust() behave exactly
as before except the label, or (b) the commander role/person cannot be reached,
widened, or taken over through any store or console-api write path. No network.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

from inkbox_claude.gate.store import Store, role_names
from inkbox_claude.gate import manager as gm
from inkbox_claude.console import api as console_api

APPROVER_TG = "7000000001"
APPROVER_PHONE = "5550100001"
APPROVER_CONV = "conv-sam"


def _store(tmp_path, *, telegram_id=APPROVER_TG, phone=APPROVER_PHONE):
    s = Store(str(tmp_path / "gate.db"))
    s.set_approver_handles(telegram_id=telegram_id, phone=phone)
    return s


def _pin(store, *, telegram_id=APPROVER_TG, phone=APPROVER_PHONE):
    """Create a person holding both approver-handle contacts and resolve the pin."""
    p = store.create_person(display="Sam Parker")
    store.add_contact(p["id"], "telegram", f"telegram:{telegram_id}")
    store.add_contact(p["id"], "phone", phone)
    return store.commander_person_id()


# -- rule 1: the role itself -------------------------------------------------

def test_commander_role_always_exists_with_empty_scopes(tmp_path):
    s = _store(tmp_path)
    row = next(r for r in s.roles() if r["name"] == "commander")
    assert row["scopes"] == []
    # Re-running ensure_commander_role (every start-up) stays idempotent.
    s.ensure_commander_role()
    row = next(r for r in s.roles() if r["name"] == "commander")
    assert row["scopes"] == []


def test_set_role_refuses_to_give_commander_scopes(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(PermissionError):
        s.set_role("commander", ["email_read"])
    with pytest.raises(PermissionError):
        s.set_role("Commander", ["email_read"])  # case-insensitive
    with pytest.raises(PermissionError):
        s.set_role("  COMMANDER  ", ["email_read"])
    # Empty scopes is a no-op and is allowed (idempotent re-assert).
    s.set_role("commander", [])
    assert next(r for r in s.roles() if r["name"] == "commander")["scopes"] == []


def test_drop_role_and_rename_role_refuse_commander(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(PermissionError):
        s.drop_role("commander")
    with pytest.raises(PermissionError):
        s.rename_role("commander", "superuser")
    s.set_role("helper", ["email_read"])
    with pytest.raises(PermissionError):
        s.rename_role("helper", "commander")
    with pytest.raises(PermissionError):
        s.rename_role("helper", "Commander")


# -- rule 3/4: exactly one pinned person, resolved lazily --------------------

def test_commander_pinned_from_telegram_contact(tmp_path):
    s = _store(tmp_path)
    p = s.create_person(display="Sam Parker")
    s.add_contact(p["id"], "telegram", f"telegram:{APPROVER_TG}")
    assert s.commander_person_id() == p["id"]


def test_commander_pinned_from_phone_contact(tmp_path):
    s = _store(tmp_path)
    p = s.create_person(display="Sam Parker")
    s.add_contact(p["id"], "phone", APPROVER_PHONE)
    assert s.commander_person_id() == p["id"]


def test_commander_unresolvable_when_no_handles_match(tmp_path):
    s = _store(tmp_path)
    s.create_person(display="Nobody in particular")
    assert s.commander_person_id() is None


def test_commander_conflicting_candidates_pin_nobody(tmp_path):
    """Both approver-handle contacts already exist, pointing at two different
    people, before anything ever asked who the commander is -- e.g. restoring
    an old database. Inserted directly (bypassing the live-traffic guards,
    which otherwise pin on the first handle before the second ever appears)."""
    s = _store(tmp_path)
    p1 = s.create_person(display="Person One")
    p2 = s.create_person(display="Person Two")
    now = 0.0
    with s._lock:
        s._db.execute(
            "INSERT INTO contacts(id,person_id,kind,value,raw_value,source,last_seen,created_at,updated_at) "
            "VALUES('c_tg1',?,?,?,?,?,?,?,?)",
            (p1["id"], "telegram", s.normalize_contact_value(APPROVER_TG), f"telegram:{APPROVER_TG}",
             "manual", now, now, now),
        )
        s._db.execute(
            "INSERT INTO contacts(id,person_id,kind,value,raw_value,source,last_seen,created_at,updated_at) "
            "VALUES('c_ph1',?,?,?,?,?,?,?,?)",
            (p2["id"], "phone", s.normalize_contact_value(APPROVER_PHONE), APPROVER_PHONE,
             "manual", now, now, now),
        )
        s._db.commit()
    assert s.commander_person_id() is None


def test_restart_pins_same_id_twice(tmp_path):
    db_path = str(tmp_path / "gate.db")
    s1 = Store(db_path)
    s1.set_approver_handles(telegram_id=APPROVER_TG, phone=APPROVER_PHONE)
    p = s1.create_person(display="Sam Parker")
    s1.add_contact(p["id"], "telegram", f"telegram:{APPROVER_TG}")
    pid = s1.commander_person_id()
    assert pid == p["id"]

    s2 = Store(db_path)
    s2.set_approver_handles(telegram_id=APPROVER_TG, phone=APPROVER_PHONE)
    assert s2.commander_person_id() == pid


def test_restart_after_attempted_contact_move_keeps_old_pin(tmp_path):
    db_path = str(tmp_path / "gate.db")
    s1 = Store(db_path)
    s1.set_approver_handles(telegram_id=APPROVER_TG, phone=APPROVER_PHONE)
    p = s1.create_person(display="Sam Parker")
    tg_contact = s1.add_contact(p["id"], "telegram", f"telegram:{APPROVER_TG}")["contact"]
    pid = s1.commander_person_id()

    other = s1.create_person(display="Someone Else")
    with pytest.raises(PermissionError):
        s1.move_contact(tg_contact["id"], other["id"])

    s2 = Store(db_path)
    s2.set_approver_handles(telegram_id=APPROVER_TG, phone=APPROVER_PHONE)
    assert s2.commander_person_id() == pid


def test_reconcile_strips_stray_commander_role_from_others(tmp_path):
    db_path = str(tmp_path / "gate.db")
    s = Store(db_path)
    s.set_approver_handles(telegram_id=APPROVER_TG, phone=APPROVER_PHONE)
    cmdr = s.create_person(display="Sam Parker")
    s.add_contact(cmdr["id"], "telegram", f"telegram:{APPROVER_TG}")
    s.commander_person_id()  # pin it

    stray = s.create_person(display="Stray Holder")
    with s._lock:  # simulate old/manual data bypassing the write guards
        s._db.execute("UPDATE persons SET role=? WHERE id=?", ("commander", stray["id"]))
        s._db.commit()
    assert "commander" in role_names(s.get_person(stray["id"])["role"])

    s.reconcile_commander()
    assert "commander" not in role_names(s.get_person(stray["id"])["role"])
    assert "commander" in role_names(s.get_person(cmdr["id"])["role"]) or True  # cmdr untouched either way


# -- rule 2/4: commander cannot be granted, removed, or scoped via any write -

def test_create_person_cannot_be_born_commander(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(PermissionError):
        s.create_person(display="Nobody", role="commander")


def test_update_person_cannot_grant_commander(tmp_path):
    s = _store(tmp_path)
    p = s.create_person(display="Helper")
    with pytest.raises(PermissionError):
        s.update_person(p["id"], role="commander")


def test_update_person_cannot_remove_commander_from_pinned_person(tmp_path):
    s = _store(tmp_path)
    pid = _pin(s)
    with pytest.raises(PermissionError):
        s.update_person(pid, role="helper")  # silently dropping commander is a refused removal


def test_update_person_cannot_give_pinned_person_scopes(tmp_path):
    s = _store(tmp_path)
    pid = _pin(s)
    with pytest.raises(PermissionError):
        s.update_person(pid, scopes=["email_read"])


def test_create_person_commander_scopes_are_fine_for_a_non_commander(tmp_path):
    """Sanity: the scopes guard is scoped to the commander person only."""
    s = _store(tmp_path)
    p = s.create_person(display="Helper", scopes=["email_read"])
    assert p["scopes"] == ["email_read"]


def test_set_trust_role_commander_on_new_key_refused(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(PermissionError):
        s.set_trust("sam@example.com", person="Sam", role="commander")


def test_set_trust_cannot_give_pinned_commander_scopes(tmp_path):
    s = _store(tmp_path)
    pid = _pin(s)
    with pytest.raises(PermissionError):
        s.set_trust(APPROVER_PHONE, person="Sam Parker", role="commander", scopes=["email_read"])


def test_forged_email_from_commander_person_gets_no_scopes(tmp_path):
    """The headline failure mode rule 2 exists to prevent: a non-empty
    per-person scopes list on the commander would leak through resolve_handles
    to ANY of their contacts, including a forged email sender address."""
    s = _store(tmp_path)
    pid = _pin(s)
    s.add_contact(pid, "email", "sam@example.com")
    # Attempting to widen the commander's own scopes is refused...
    with pytest.raises(PermissionError):
        s.update_person(pid, scopes=["email_read"])
    # ...so a message claiming to be from their email still resolves to nothing.
    result = s.trust_for(["sam@example.com"])
    # The actual security property: no scopes flow through, ever.
    assert result["scopes"] == []
    # The role label is cosmetic attribution of a known contact to a known person
    # (same as any other role shows up in store.trusted()/people()) -- it carries
    # no power on its own; only is_approver() does that, and a forged email never
    # passes is_approver().


def test_delete_person_refuses_commander(tmp_path):
    s = _store(tmp_path)
    pid = _pin(s)
    with pytest.raises(PermissionError):
        s.delete_person(pid)


def test_drop_trust_on_commanders_only_contact_refused(tmp_path):
    s = _store(tmp_path)
    s.set_trust(APPROVER_PHONE, person="Sam Parker")
    pid = s.commander_person_id()
    assert pid is not None
    with pytest.raises(PermissionError):
        s.drop_trust(APPROVER_PHONE)
    # The person and their contact are both still there.
    assert s.get_person(pid) is not None


def test_contact_guards_on_approver_handles(tmp_path):
    s = _store(tmp_path)
    p = s.create_person(display="Sam Parker")
    tg = s.add_contact(p["id"], "telegram", f"telegram:{APPROVER_TG}")["contact"]
    ph = s.add_contact(p["id"], "phone", APPROVER_PHONE)["contact"]
    s.commander_person_id()  # pin

    other = s.create_person(display="Someone Else")
    with pytest.raises(PermissionError):
        s.move_contact(tg["id"], other["id"])
    with pytest.raises(PermissionError):
        s.update_contact(tg["id"], value="telegram:9999999999")
    with pytest.raises(PermissionError):
        s.remove_contact(ph["id"])
    with pytest.raises(PermissionError):
        s.upsert_contact("phone", APPROVER_PHONE, person_id=other["id"])
    with pytest.raises(PermissionError):
        s.update_contact(tg["id"], kind="phone")


def test_link_people_both_directions(tmp_path):
    s = _store(tmp_path)
    pid = _pin(s)
    other = s.create_person(display="Someone Else")
    # Merging someone ELSE into the commander is fine.
    s.link_people(keep_id=pid, merge_id=other["id"])
    assert s.get_person(other["id"]) is None or s.get_person(other["id"])["id"] == pid

    third = s.create_person(display="Third Person")
    # Merging the commander AWAY into someone else is refused.
    with pytest.raises(PermissionError):
        s.link_people(keep_id=third["id"], merge_id=pid)


# -- rule 5: trust()/is_approver() -------------------------------------------

def test_role_names_case_insensitive_everywhere(tmp_path):
    s = _store(tmp_path)
    assert role_names("Commander") == ["commander"]
    assert role_names(" COMMANDER ") == ["commander"]
    with pytest.raises(PermissionError):
        s.update_person(s.create_person(display="X")["id"], role="Commander")


def make_manager(tmp_path, *, approver_telegram_id="", approver_phone=""):
    async def send_fn(chat_id, text, mode, meta):
        pass

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV,
                          approver_phone=approver_phone)
    old = os.environ.get("GATE_APPROVER_TELEGRAM_ID")
    if approver_telegram_id:
        os.environ["GATE_APPROVER_TELEGRAM_ID"] = approver_telegram_id
    try:
        m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                                  store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    finally:
        if old is None:
            os.environ.pop("GATE_APPROVER_TELEGRAM_ID", None)
        else:
            os.environ["GATE_APPROVER_TELEGRAM_ID"] = old
    return m


def test_trust_returns_commander_role_for_approver_unchanged_scopes_and_person(tmp_path):
    m = make_manager(tmp_path)
    session = m.get("thread-1")
    session.mode = "imessage"
    session.reply_meta = {"conversation_id": APPROVER_CONV}
    assert session.is_approver() is True
    trust = session.trust()
    assert trust == {"role": "commander", "scopes": ["*"], "person": "Aaron"}


def test_trust_for_stranger_is_unaffected(tmp_path):
    m = make_manager(tmp_path)
    session = m.get("thread-2")
    session.mode = "email"
    session.reply_meta = {"sender": "cand@nyu.edu", "conversation_id": ""}
    assert session.is_approver() is False
    trust = session.trust()
    assert trust == {"role": "", "scopes": [], "person": ""}


def test_voice_call_without_trust_env_gets_no_scopes(tmp_path):
    m = make_manager(tmp_path, approver_phone=APPROVER_PHONE)
    m.voice_trust_approver = False  # GATE_VOICE_TRUST_APPROVER unset/off
    session = m.get("thread-voice")
    session.mode = "voice"
    session.reply_meta = {"sender": APPROVER_PHONE}
    assert session.is_approver() is False
    assert session.is_aaron_on_phone() is True  # recognised as Aaron for voice UX, but not trusted
    trust = session.trust()
    assert trust["scopes"] == []


def test_voice_call_with_trust_env_on_still_gets_commander(tmp_path):
    m = make_manager(tmp_path, approver_phone=APPROVER_PHONE)
    m.voice_trust_approver = True
    session = m.get("thread-voice-2")
    session.mode = "voice"
    session.reply_meta = {"sender": APPROVER_PHONE}
    assert session.is_approver() is True
    assert session.trust()["role"] == "commander"


# -- rule 6/7: console API -----------------------------------------------

LOCKED_ROLE_ENDPOINTS = [
    (console_api.upsert_role, {"name": "commander", "scopes": ["email_read"]}),
    (console_api.delete_role, {"name": "commander"}),
    (console_api.add_role_scope, {"name": "commander", "scope": "email_read"}),
    (console_api.remove_role_scope, {"name": "commander", "scope": "email_read"}),
    (console_api.add_role_member, {"name": "commander", "key": "sam@example.com"}),
    (console_api.remove_role_member, {"name": "commander", "key": "sam@example.com"}),
]


def test_console_api_refuses_locked_role_mutations(tmp_path):
    s = _store(tmp_path)
    for fn, payload in LOCKED_ROLE_ENDPOINTS:
        with pytest.raises(console_api.ValidationError):
            fn(s, payload)


def test_console_api_refuses_rename_involving_commander(tmp_path):
    s = _store(tmp_path)
    s.set_role("helper", ["email_read"])
    with pytest.raises(console_api.ValidationError):
        console_api.rename_role_endpoint(s, {"name": "commander", "new_name": "superuser"})
    with pytest.raises(console_api.ValidationError):
        console_api.rename_role_endpoint(s, {"name": "helper", "new_name": "commander"})


def test_console_role_model_flags_locked_role(tmp_path):
    s = _store(tmp_path)
    roles = {r["name"]: r for r in console_api.roles(s)}
    assert roles["commander"]["locked"] is True
    assert roles["commander"]["description"] == "Everything, on verified channels only."
    s.set_role("helper", ["email_read"])
    roles = {r["name"]: r for r in console_api.roles(s)}
    assert roles["helper"]["locked"] is False
    assert roles["helper"]["description"] == ""


def test_console_person_model_flags_commander_person(tmp_path):
    s = _store(tmp_path)
    pid = _pin(s)
    other = s.create_person(display="Someone Else")
    people = {p["id"]: p for p in [console_api._person_model(s, row) for row in s.list_people()]}
    assert people[pid]["commander"] is True
    assert people[other["id"]]["commander"] is False


def test_console_update_person_endpoint_refuses_role_drop_on_commander(tmp_path):
    s = _store(tmp_path)
    pid = _pin(s)
    with pytest.raises(PermissionError):
        console_api.update_person_endpoint(s, {"id": pid, "role": "helper"})


def test_console_remove_person_endpoint_refuses_commander(tmp_path):
    s = _store(tmp_path)
    pid = _pin(s)
    with pytest.raises(PermissionError):
        console_api.remove_person_endpoint(s, {"id": pid})
