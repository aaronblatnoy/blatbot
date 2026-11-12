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
