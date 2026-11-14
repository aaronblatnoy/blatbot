"""Long-term graph memory: off by default, fails open, and keeps each chat's facts
to that chat unless the reader is the owner."""

import asyncio

from inkbox_claude.gate import graphmem


def test_off_without_url(monkeypatch):
    monkeypatch.delenv("GRAPHITI_URL", raising=False)
    calls = []
    monkeypatch.setattr(graphmem, "_post", lambda *a, **k: calls.append(a))
    graphmem.record("tg:1", "inbound", "hello")
    assert graphmem.search("hello", "tg:1", True) == []
    assert asyncio.run(graphmem.recall("hello", "tg:1", True)) == ""
    assert calls == []


def test_group_id_is_safe():
    assert graphmem.group_for("tg:-100123") == "blatbot_tg_-100123"
    assert graphmem.group_for("a@b.com") == "blatbot_a_b_com"
    assert graphmem.group_for("") == "blatbot_unknown"


def test_episode_shape_and_clip():
    e = graphmem.episode("owner", "inbound", "x" * 9000, "telegram", created_at=0, message_id=7)
    m = e["messages"][0]
    assert e["group_id"] == "blatbot_owner"
    assert m["role_type"] == "user" and m["name"] == "msg-7"
    assert m["content"].endswith("[clipped]") and len(m["content"]) < 6100
    assert m["timestamp"].startswith("1970-01-01")
    assert graphmem.episode("owner", "outbound", "hi")["messages"][0]["role_type"] == "assistant"
    assert graphmem.episode("owner", "system", "note")["messages"][0]["role_type"] == "system"


def test_non_owner_reads_only_their_own_chat(monkeypatch):
    monkeypatch.setenv("GRAPHITI_URL", "http://graph.test")
    sent = []

    def fake_post(path, payload, timeout):
        sent.append((path, payload))
        return {"facts": [{"fact": "Sam prefers email", "valid_at": "2026-09-20T00:00:00Z"}]}

    monkeypatch.setattr(graphmem, "_post", fake_post)
    graphmem.search("what does Sam prefer", "tg:5", False)
    graphmem.search("what does Sam prefer", "owner", True)
    assert sent[0] == ("/search", {"query": "what does Sam prefer", "max_facts": graphmem.MAX_FACTS,
                                   "group_ids": ["blatbot_tg_5"]})
    assert sent[1][1]["group_ids"] is None


def test_recall_renders_and_fails_open(monkeypatch):
    monkeypatch.setenv("GRAPHITI_URL", "http://graph.test")
    monkeypatch.setattr(graphmem, "_post", lambda *a, **k: {"facts": [
        {"fact": "Sam prefers email", "valid_at": "2026-09-20T00:00:00Z"},
        {"fact": "Sam prefers email", "valid_at": "2026-09-20T00:00:00Z"},
        {"fact": "Sam worked at Acme", "valid_at": "2025-01-01T00:00:00Z", "invalid_at": "2026-03-01T00:00:00Z"},
    ]})
    text = asyncio.run(graphmem.recall("Sam", "owner", True))
    assert text.startswith(graphmem.HEADER)
    assert text.count("Sam prefers email") == 1
    assert "(since 2026-09-20)" in text and "no longer current" in text

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(graphmem, "_post", boom)
    assert asyncio.run(graphmem.recall("Sam", "owner", True)) == ""


def test_record_never_raises(monkeypatch):
    monkeypatch.setenv("GRAPHITI_URL", "http://graph.test")
    monkeypatch.setattr(graphmem, "_ensure_worker", lambda: None)
    graphmem.record("tg:1", "inbound", "remember the gate code is 4411", "telegram", message_id=3)
    item = graphmem._queue.get_nowait()
    assert item["messages"][0]["content"] == "remember the gate code is 4411"
    graphmem.record("tg:1", "inbound", "   ")
    assert graphmem._queue.empty()


def test_speaker_from_thread_meta():
    assert graphmem.speaker_from_meta("tg:-1", '{"group_title": "Corgi  NYC", "sender": "x"}') == \
        "a member of the group chat Corgi NYC"
    assert graphmem.speaker_from_meta("abc", {"sender": "sam@example.com"}) == "sam@example.com"
    assert graphmem.speaker_from_meta("owner", {"sender": "+15550000000"}) == ""
    assert graphmem.speaker_from_meta("abc", "not json") == ""
    e = graphmem.episode("abc", "inbound", "hi", speaker="sam@example.com")
    assert e["messages"][0]["role"] == "sam@example.com"


def test_message_ids_come_from_episode_names():
    facts = [{"fact": "a", "episodes": ["msg-12", "msg-7"]}, {"fact": "b", "episodes": ["msg-7", "msg-1794523875000", "x"]},
             {"fact": "c"}]
    assert graphmem.message_ids(facts) == [12, 7]


def test_tasks_for_messages_ranked_by_hits(tmp_path):
    from inkbox_claude.gate.store import Store
    store = Store(str(tmp_path / "gate.db"))
    with store._lock:
        for rid, inbound, task in ((1, 10, 5), (2, 11, 6), (3, 12, 6), (4, 13, None)):
            store._db.execute(
                "INSERT INTO requests(id, chat_id, sender, mode, original_message, summary, scopes_json, prompt, prompt_sha256, state, "
                "created_at, updated_at, task_id, inbound_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, "c", "s", "imessage", "m", "s", "[]", "p", "x", "done", 0, 0, task, inbound))
        for mid, chat, at in ((20, "c", 1000.0), (21, "c", 5000.0), (22, "other", 1000.0)):
            store._db.execute("INSERT INTO messages(id, chat_id, kind, mode, text, created_at) VALUES(?,?,?,?,?,?)",
                              (mid, chat, "outbound", "imessage", "t", at))
        store._db.execute("INSERT INTO task_events(task_id, kind, chat_id, text, created_at) VALUES(?,?,?,?,?)",
                          (7, "done", "c", "t", 1000.5))
        store._db.commit()
    assert store.task_ids_for_messages([10, 11, 12, 13, 99]) == [6, 5]
    # A reply is tied to the task active in its own chat at that moment, and to nothing otherwise.
    assert store.task_ids_for_messages([20, 21, 22]) == [7]
    assert store.task_ids_for_messages([]) == []


# ---- who may read which memory groups --------------------------------------

def _people_store(tmp_path):
    """Owner (commander), Ann (role "friend", phone + email + telegram), Bob (no role), a group
    chat both Ann and Bob are in, and a private chat per channel."""
    from inkbox_claude.gate.store import Store
    store = Store(str(tmp_path / "gate.db"))
    owner = store.create_person(display="Aaron")
    store.add_contact(owner["id"], "phone", "+14075550100")
    store._ensure_settings()
    with store._lock:  # pin the owner's person record the way start-up does
        store._db.execute("INSERT INTO settings(name,value,updated_at) VALUES('commander_person_id',?,0)", (owner["id"],))
        store._db.commit()
    ann = store.create_person(display="Ann", role="friend")
    store.add_contact(ann["id"], "phone", "+12015550111")
    store.add_contact(ann["id"], "email", "ann@example.com")
    store.add_contact(ann["id"], "telegram", "555111")
    bob = store.create_person(display="Bob")
    store.add_contact(bob["id"], "phone", "+13475550122")
    store.set_role("friend", [])
    for chat, mode, meta in (
            ("im-ann", "imessage", {"sender": "+12015550111", "conversation_kind": "direct"}),
            ("mail-ann", "email", {"sender": "ann@example.com"}),
            ("tg:555111", "telegram", {"sender": "telegram:555111"}),
            ("im-bob", "imessage", {"sender": "+13475550122"}),
            ("tg:-77", "telegram", {"sender": "telegram:555111", "conversation_kind": "group", "group_title": "Club"}),
            ("owner", "telegram", {"sender": "telegram:1", "owner": 1}),
            ("im-owner-dm", "imessage", {"sender": "+14075550100", "owner": 1})):
        store.set_thread(chat, "idle", mode, meta)
    return store, ann, bob


def _groups(store, chat, handles, *, approver=False, group=False):
    return graphmem.readable_groups(store, chat, is_approver=approver, is_group=group, handles=handles)


def test_owner_reads_every_group(tmp_path):
    store, _, _ = _people_store(tmp_path)
    assert _groups(store, "owner", ["+14075550100"], approver=True) is None


def test_person_recalls_across_their_own_channels(tmp_path):
    store, _, _ = _people_store(tmp_path)
    got = _groups(store, "mail-ann", ["ann@example.com"])
    assert got[0] == "blatbot_mail-ann"
    assert set(got) == {"blatbot_im-ann", "blatbot_mail-ann", "blatbot_tg_555111"}
    # Telegram resolves the same person from the stamped id alone.
    assert set(_groups(store, "tg:555111", ["telegram:555111"])) == set(got)


def test_stranger_and_unrelated_person_see_only_their_own_chat(tmp_path):
    store, _, _ = _people_store(tmp_path)
    assert _groups(store, "im-new", ["+19995550000"]) == ["blatbot_im-new"]
    # Bob is on record but has no chats of Ann's, and Ann's chats never reach him.
    assert _groups(store, "im-bob", ["+13475550122"]) == ["blatbot_im-bob"]


def test_group_chat_facts_do_not_reach_a_dm_and_dm_facts_do_not_reach_the_group(tmp_path):
    store, _, _ = _people_store(tmp_path)
    assert "blatbot_tg_-77" not in _groups(store, "im-ann", ["+12015550111"])
    # Inside the group only the group is read, even for Ann who has a role and private chats.
    assert _groups(store, "tg:-77", ["telegram:555111"], group=True) == ["blatbot_tg_-77"]


def test_role_grants_extra_groups(tmp_path):
    store, _, _ = _people_store(tmp_path)
    store.set_role("friend", ["memory:chat:im-bob", "memory:groups"])
    got = set(_groups(store, "im-ann", ["+12015550111"]))
    assert {"blatbot_im-bob", "blatbot_tg_-77", "blatbot_im-ann"} <= got
    assert "blatbot_owner" not in got and "blatbot_im-owner-dm" not in got
    store.set_role("friend", ["memory:all"])
    assert _groups(store, "im-ann", ["+12015550111"]) is None


def test_memory_entries_are_not_tool_scopes(tmp_path):
    store, _, _ = _people_store(tmp_path)
    store.set_role("friend", ["memory:all"])
    who = store.resolve_handles(["+12015550111"])
    assert who["memory"] == ["memory:all"] and who["scopes"] == []


def test_commander_contact_without_approval_gets_only_its_own_chat(tmp_path):
    store, _, _ = _people_store(tmp_path)
    # A forged or unverified contact that resolves to the owner's record is not the owner.
    assert _groups(store, "im-x", ["+14075550100"]) == ["blatbot_im-x"]


def test_unresolvable_identity_falls_back_to_own_chat(tmp_path):
    class Broken:
        def resolve_handles(self, handles):
            raise RuntimeError("db down")
    assert graphmem.readable_groups(Broken(), "c1", is_approver=False, is_group=False, handles=["x"]) == ["blatbot_c1"]
    store, _, _ = _people_store(tmp_path)
    assert _groups(store, "c2", []) == ["blatbot_c2"]


def test_conflicting_handles_fall_back_to_own_chat(tmp_path):
    store, _, _ = _people_store(tmp_path)
    assert _groups(store, "im-ann", ["+12015550111", "+13475550122"]) == ["blatbot_im-ann"]


def test_search_sends_groups_and_drops_facts_from_other_groups(monkeypatch):
    monkeypatch.setenv("GRAPHITI_URL", "http://graph.test")
    sent = []

    def fake_post(path, payload, timeout):
        sent.append(payload)
        return {"facts": [{"fact": "mine", "group_id": "blatbot_a"},
                          {"fact": "theirs", "group_id": "blatbot_z"},
                          {"fact": "no group"}]}

    monkeypatch.setattr(graphmem, "_post", fake_post)
    out = graphmem.search("q", "a", False, groups=["blatbot_a", "blatbot_b"])
    assert sent[0]["group_ids"] == ["blatbot_a", "blatbot_b"]
    assert [f["fact"] for f in out] == ["mine"]
    # The owner (all groups) keeps every fact.
    assert len(graphmem.search("q", "owner", True, groups=None)) == 3


def test_roles_accept_only_well_formed_memory_entries(tmp_path):
    from inkbox_claude.console import api as console_api
    from inkbox_claude.gate.store import Store
    assert graphmem.valid_memory_grant("memory:all") and graphmem.valid_memory_grant("memory:groups")
    assert graphmem.valid_memory_grant("memory:chat:tg:-77")
    assert not graphmem.valid_memory_grant("memory:chat:") and not graphmem.valid_memory_grant("memory:everything")
    store = Store(str(tmp_path / "gate.db"))
    store.set_role("friend", [])
    console_api.upsert_role(store, {"name": "friend", "scopes": ["memory:groups"], "note": ""})
    assert next(r for r in store.roles() if r["name"] == "friend")["scopes"] == ["memory:groups"]
    import pytest
    with pytest.raises(console_api.ValidationError):
        console_api.upsert_role(store, {"name": "friend", "scopes": ["memory:everything"], "note": ""})
