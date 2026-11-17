"""The console's read-only Memory and Pages views, on throwaway stores and folders."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from inkbox_claude.console import api
from inkbox_claude.gate.store import Store


def test_memory_view_counts_messages_tasks_notes_and_instructions(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    store.add_message("+15550001111", "inbound", "hello there", "sms")
    store.add_message("+15550001111", "outbound", "hi", "sms")
    store.add_message("tg:-100", "inbound", "group note", "telegram")
    store.create_task("Book the room", participants=["+15550001111"])
    standing = tmp_path / "standing.md"
    standing.write_text("Be brief.\n\nNo emojis.\n", "utf-8")
    vault = tmp_path / "vault"
    (vault / "Areas").mkdir(parents=True)
    (vault / "Areas" / "Infra.md").write_text("x", "utf-8")
    (vault / "Home.md").write_text("x", "utf-8")
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "hidden.md").write_text("x", "utf-8")
    gateway = SimpleNamespace(cfg=SimpleNamespace(gate_standing_path=str(standing)))

    view = api.memory_view(gateway, store, notes_root=vault)

    assert view["instructions"]["paragraphs"] == ["Be brief.", "No emojis."]
    assert view["messages"]["by_kind"] == {"inbound": 2, "outbound": 1}
    assert view["messages"]["chats"] == 2
    assert sum(view["messages"]["per_day"]) == 3
    assert view["tasks"]["by_state"].get("open") == 1
    assert view["tasks"]["latest"][0]["title"] == "Book the room"
    assert view["notes"]["total"] == 2
    assert {s["name"] for s in view["notes"]["sections"]} == {"Areas", "Loose notes"}
    assert "enabled" in view["graph"]


def test_memory_view_without_instructions_or_vault(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    gateway = SimpleNamespace(cfg=SimpleNamespace(gate_standing_path=str(tmp_path / "missing.md")))
    view = api.memory_view(gateway, store, notes_root=tmp_path / "no-vault")
    assert view["instructions"]["available"] is False
    assert view["notes"]["available"] is False
    assert view["messages"]["chats"] == 0


def test_pages_view_lists_live_pages_newest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_PAGES_DIR", str(tmp_path))
    now = time.time()
    good_old, good_new, expired = "a" * 43, "b" * 43, "c" * 43
    for token, created, expires in ((good_old, now - 100, now + 1000), (good_new, now - 10, now + 1000),
                                    (expired, now - 50, now - 1)):
        (tmp_path / f"{token}.html").write_text("<p>x</p>", "utf-8")
        (tmp_path / f"{token}.json").write_text(json.dumps({"title": token[:3], "created": created, "expires": expires}), "utf-8")
    (tmp_path / "not-a-token.json").write_text("{}", "utf-8")

    view = api.pages_view()

    assert [p["title"] for p in view["pages"]] == ["bbb", "aaa"]
    assert view["pages"][0]["url"].endswith(f"/p/{good_new}")
    assert view["pages"][0]["bytes"] > 0
