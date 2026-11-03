"""People auto-indexing: whoever Blatbot meets gets one People record, a directory
entry only, and Aaron is told about it once, as the record now stands. No network."""
from __future__ import annotations

import asyncio
import sqlite3
import types

import pytest

from inkbox_claude import gateway, telegram
from inkbox_claude.gate import peopleindex
from inkbox_claude.gate.store import Store
from test_gate import make_manager, stranger_meta

ROOM = "FellowSIp of the Ring"


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "gate.db"))


def _stanley(store, **kw):
    return store.index_person(display="Stanley Rabkin",
                              handles=[("telegram", "8804662817"), ("telegram_username", "@stanrab")],
                              met_in=kw.pop("met_in", ROOM), **kw)


def _kinds(person):
    return {(c["kind"], c["value"]) for c in person["contacts"]}


def test_someone_new_gets_a_record_with_every_handle_and_nothing_to_do(store):
    store.set_role("partner", ["calendar"], "")
    res = _stanley(store, note="TAMID senior advisor")
    assert res["action"] == "added"
    p = store.get_person(res["person"]["id"])
    assert p["display"] == "Stanley Rabkin"
    assert _kinds(p) == {("telegram", "8804662817"), ("telegram_username", "@stanrab")}
    assert p["met_in"] == [ROOM] and p["first_seen"] > 0 and p["note"] == "TAMID senior advisor"
    # A directory entry only: no role, no scope, so a message from him grants nothing.
    assert p["role"] == "" and p["scopes"] == []
    assert store.trust_for(["telegram:8804662817"]) == {"role": "", "scopes": [], "person": "Stanley Rabkin"}


def test_the_same_person_again_is_the_same_record_and_nothing_new(store):
    first = _stanley(store)["person"]["id"]
    again = _stanley(store, met_in="Aaron & Stanley, Blatbot")
    assert again["action"] == "unchanged" and again["person"]["id"] == first
    assert len(store.list_people()) == 1
    # Where else they were met is kept, but is not a change worth a text.
    assert store.get_person(first)["met_in"] == [ROOM, "Aaron & Stanley, Blatbot"]
    assert {c["source"] for c in store.get_person(first)["contacts"]} == {f"met in {ROOM}"}


def test_a_new_handle_extends_the_record(store):
    pid = _stanley(store)["person"]["id"]
    res = store.index_person(display="Stanley Rabkin",
                             handles=[("telegram", "8804662817"), ("phone", "+1 (555) 010-0042")],
                             met_in=ROOM)
    assert res["action"] == "updated" and res["person"]["id"] == pid
    assert ("phone", "5550100042") in _kinds(store.get_person(pid))


def test_an_introduction_before_they_speak_becomes_their_record(store):
    intro = store.index_person(display="Stanley", met_in=ROOM, note="He runs TAMID")
    assert intro["action"] == "added" and intro["person"]["contacts"] == []
    spoke = _stanley(store)
    assert spoke["person"]["id"] == intro["person"]["id"]
    p = store.get_person(intro["person"]["id"])
    assert p["display"] == "Stanley Rabkin" and p["note"] == "He runs TAMID"
    assert len(store.list_people()) == 1


def test_an_introduction_after_they_spoke_adds_the_note_to_their_record(store):
    pid = _stanley(store)["person"]["id"]
    res = store.index_person(display="Stanley", met_in="Telegram DM", note="Our senior advisor")
    assert res["action"] == "updated" and res["person"]["id"] == pid
    assert store.get_person(pid)["note"] == "Our senior advisor"
    assert store.get_person(pid)["display"] == "Stanley Rabkin"


def test_a_same_name_with_a_different_account_is_not_merged(store):
    _stanley(store)
    other = store.index_person(display="Stanley Rabkin", handles=[("telegram", "1111111111")], met_in=ROOM)
    assert other["action"] == "added"
    assert len(store.list_people()) == 2


def test_handles_of_two_people_are_never_merged(store):
    a = store.index_person(display="A", handles=[("email", "a@example.com")], met_in="email")
    b = store.index_person(display="B", handles=[("phone", "+15555550000")], met_in="SMS")
    res = store.index_person(display="AB", handles=[("email", "a@example.com"), ("phone", "+15555550000")])
    assert res["action"] == "conflict"
    assert {p["id"] for p in store.list_people()} == {a["person"]["id"], b["person"]["id"]}


def test_the_commander_is_never_indexed(store):
    store.set_approver_handles(telegram_id="6752193595", phone="+15550100001")
    assert store.index_person(display="Aaron B", handles=[("telegram", "6752193595")])["action"] == "owner"
    assert store.index_person(display="Aaron", handles=[("phone", "+1 555 010 0001")])["action"] == "owner"
    assert store.list_people() == [] or all(p["first_seen"] == 0 for p in store.list_people())


def test_an_old_database_gets_the_new_columns(tmp_path):
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE persons (id TEXT PRIMARY KEY, display TEXT NOT NULL DEFAULT '',
        role TEXT NOT NULL DEFAULT '', scopes TEXT NOT NULL DEFAULT '[]', note TEXT NOT NULL DEFAULT '',
        merged_into TEXT, created_at REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0);
        INSERT INTO persons(id, display) VALUES ('p_old', 'Old Friend');""")
    db.commit(); db.close()
    s = Store(path)
    p = s.get_person("p_old")
    if p is not None:       # the sqlite migration keeps hand-made rows
        assert p["met_in"] == [] and p["first_seen"] == 0
    assert any(k["kind"] == "telegram_username" for k in s.contact_kinds())
    assert s.people_to_announce() == []


# -- the text to Aaron ---------------------------------------------------------

def _owner_texts(sent):
    return [t for _, t, *_ in sent]


def test_the_notice_states_the_record_once_and_never_again(tmp_path):
    m, sent = make_manager(tmp_path)
    m.index_person_seen(display="Stanley Rabkin",
                        handles=[("telegram", "8804662817"), ("telegram_username", "@stanrab")],
                        met_in=ROOM, note="Aaron's TAMID senior advisor.")
    assert asyncio.run(m.flush_people_notices()) == 1
    assert _owner_texts(sent) == [
        "Stanley Rabkin is in People: Telegram @stanrab, met in FellowSIp of the Ring. "
        "Note: Aaron's TAMID senior advisor."]
    # Indexed again, flushed again, after a restart: nothing more.
    m.index_person_seen(display="Stanley Rabkin", handles=[("telegram", "8804662817")], met_in=ROOM)
    assert asyncio.run(m.flush_people_notices()) == 0
    m2, sent2 = make_manager(tmp_path)
    assert asyncio.run(m2.flush_people_notices()) == 0 and sent2 == []


def test_only_handle_is_the_id_so_the_id_is_said(tmp_path):
    m, sent = make_manager(tmp_path)
    m.index_person_seen(display="Stanley Rabkin", handles=[("telegram", "8804662817")],
                        met_in=ROOM)
    m.index_person_seen(display="Stanley Rabkin", handles=[("telegram", "8804662817")],
                        met_in="Aaron & Stanley, Blatbot")
    asyncio.run(m.flush_people_notices())
    assert _owner_texts(sent) == [
        "Stanley Rabkin is in People: Telegram id 8804662817, met in FellowSIp of the Ring "
        "and in Aaron & Stanley, Blatbot."]
    for text in _owner_texts(sent):
        assert "—" not in text and "**" not in text


def test_a_later_handle_is_one_text_naming_only_what_is_new(tmp_path):
    m, sent = make_manager(tmp_path)
    m.index_person_seen(display="Stanley Rabkin", handles=[("telegram", "8804662817")], met_in=ROOM)
    asyncio.run(m.flush_people_notices())
    m.index_person_seen(display="Stanley Rabkin", handles=[("telegram", "8804662817"),
                                                           ("phone", "+1 555 010 0042")], met_in=ROOM)
    asyncio.run(m.flush_people_notices())
    asyncio.run(m.flush_people_notices())
    assert _owner_texts(sent)[1:] == ["Stanley Rabkin in People now also has phone +1 555 010 0042."]


def test_several_people_at_once_are_one_text(tmp_path):
    m, sent = make_manager(tmp_path)
    m.index_person_seen(display="Dana Reyes", handles=[("telegram", "5550100011")], met_in="Board")
    m.index_person_seen(display="Lee Kim", handles=[("telegram", "5550100012"),
                                                   ("telegram_username", "@leekim")], met_in="Board")
    assert asyncio.run(m.flush_people_notices()) == 2
    assert len(sent) == 1
    assert sent[0][1].splitlines() == ["Dana Reyes is in People: Telegram id 5550100011, met in Board.",
                                       "Lee Kim is in People: Telegram @leekim, met in Board."]


def test_a_record_made_by_hand_is_told_only_what_indexing_added(tmp_path):
    m, sent = make_manager(tmp_path)
    p = m.store.create_person(display="Sean Parnell", note="Corgi")
    m.store.add_contact(p["id"], "email", "sean@example.com")
    asyncio.run(m.flush_people_notices())
    assert sent == []           # nothing indexing touched
    m.index_person_seen(display="Sean Parnell", handles=[("email", "sean@example.com")], met_in="email")
    asyncio.run(m.flush_people_notices())
    assert sent == []           # touched, nothing new
    m.index_person_seen(display="", handles=[("email", "sean@example.com"), ("phone", "+15550100077")])
    asyncio.run(m.flush_people_notices())
    assert _owner_texts(sent) == ["Sean Parnell in People now also has phone +15550100077."]


def test_an_email_from_someone_new_indexes_them(tmp_path):
    m, sent = make_manager(tmp_path)
    asyncio.run(m.get("c1").handle_inbound("hello there", "email", stranger_meta()))
    people = [p for p in m.store.list_people() if p["first_seen"]]
    assert [p["display"] for p in people] == ["Cand Idate"]
    assert ("email", "cand@nyu.edu") in _kinds(people[0])
    assert people[0]["role"] == "" and people[0]["scopes"] == []
    sent.clear()
    asyncio.run(m.flush_people_notices())
    assert _owner_texts(sent) == ["Cand Idate is in People: email cand@nyu.edu, met over email."]


def test_mail_from_a_mailbox_nobody_writes_from_is_not_indexed(tmp_path):
    m, sent = make_manager(tmp_path)
    meta = dict(stranger_meta(), sender="no-reply@accounts.example.com", to="no-reply@accounts.example.com")
    asyncio.run(m.get("c2").handle_inbound("Your code is 123456", "email", meta))
    assert not [p for p in m.store.list_people() if p["first_seen"]]


# -- reading who was met ---------------------------------------------------------

@pytest.mark.parametrize("text,addressed,want", [
    ("blatbot, this is Stanley Rabkin, he runs X", True, ("Stanley Rabkin", "He runs X")),
    ("blatbot meet stanley", True, ("Stanley", "")),
    ("say hi to Stanley", True, ("Stanley", "")),
    ("blatbot meet stanley rabkin he is our advisor", True, ("Stanley Rabkin", "He is our advisor")),
    ("this is great", True, None),
    ("blatbot this is stanley", True, None),          # "this is" needs the name capitalised
    ("say hi to everyone", True, None),
    ("blatbot what's stanley's role in tamid?", True, None),
    ("meet Stanley", False, None),                    # not aimed at Blatbot
])
def test_an_introduction_is_read_from_aarons_words(text, addressed, want):
    assert peopleindex.parse_introduction(text, addressed=addressed) == want


def test_mailboxes_nobody_writes_from():
    assert peopleindex.is_automated_email("noreply@x.com")
    assert peopleindex.is_automated_email("notifications+abc@github.com")
    assert not peopleindex.is_automated_email("stanley.rabkin@nyu.edu")


def test_telegram_updates_carry_the_username_and_new_members():
    u = telegram.parse_update({"update_id": 1, "message": {
        "message_id": 2, "text": "hi", "chat": {"id": -100, "type": "group", "title": ROOM},
        "from": {"id": 8804662817, "first_name": "Stanley", "last_name": "Rabkin", "username": "stanrab"}}})
    assert u["username"] == "stanrab" and u["is_bot"] is False
    joined = telegram.parse_new_members({"update_id": 3, "message": {
        "message_id": 4, "chat": {"id": -100, "type": "group", "title": ROOM},
        "new_chat_members": [{"id": 1, "first_name": "Lee", "username": "leekim"},
                             {"id": 2, "first_name": "Shire", "is_bot": True}]}})
    assert joined["group_title"] == ROOM
    assert joined["members"] == [{"id": "1", "name": "Lee", "username": "leekim", "is_bot": False},
                                 {"id": "2", "name": "Shire", "username": "", "is_bot": True}]
    assert telegram.parse_new_members({"message": {"text": "hi", "chat": {"id": -1}}}) is None


@pytest.fixture
def gw(tmp_path, monkeypatch):
    monkeypatch.setattr(telegram, "approver_id", lambda: "6752193595")
    calls = []
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    g.sessions = types.SimpleNamespace(
        index_person_seen=lambda **kw: calls.append(("seen", kw)),
        index_introduction=lambda text, **kw: calls.append(("intro", dict(kw, text=text))))
    return g, calls


def test_a_telegram_group_sender_is_indexed_whether_or_not_blatbot_answers(gw):
    g, calls = gw
    g._index_telegram_people({"chat_id": "-100", "from_id": "8804662817", "name": "Stanley Rabkin",
                              "username": "stanrab", "text": "how did you make that dashboard",
                              "is_group": True, "group_title": ROOM, "addressed": False},
                             is_commander=False)
    assert calls == [("seen", {"display": "Stanley Rabkin", "met_in": ROOM, "handles": [
        ("telegram", "8804662817"), ("telegram_username", "@stanrab")]})]


def test_aaron_in_a_group_is_read_for_an_introduction_never_indexed(gw):
    g, calls = gw
    g._index_telegram_people({"chat_id": "-100", "from_id": "6752193595", "name": "Aaron B",
                              "text": "blatbot this is Stanley", "is_group": True,
                              "group_title": ROOM, "addressed": True}, is_commander=True)
    assert calls == [("intro", {"text": "blatbot this is Stanley", "addressed": True, "met_in": ROOM})]


def test_new_group_members_are_indexed_but_not_bots_or_aaron(gw):
    g, calls = gw
    g._index_telegram_members({"chat_id": "-100", "group_title": ROOM, "members": [
        {"id": "1", "name": "Lee", "username": "leekim", "is_bot": False},
        {"id": "2", "name": "Shire", "username": "", "is_bot": True},
        {"id": "6752193595", "name": "Aaron", "username": "", "is_bot": False}]})
    assert calls == [("seen", {"display": "Lee", "met_in": ROOM, "handles": [
        ("telegram", "1"), ("telegram_username", "@leekim")]})]
