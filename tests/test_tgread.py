"""The telegram_read capability: find/list/read over a fake transport (never
Telegram), the host tools that wrap it, the scope registry entry, the
blanket-grant exclusion, and schedule capability detection.

Every chat, name and message here is invented; nothing names a real person."""

from __future__ import annotations

import asyncio

import pytest

from inkbox_claude.gate import hosttools, schedprop, tgread
from inkbox_claude.gate.scopes import EXPLICIT_ONLY_SCOPES, SCOPES, tools_for
from inkbox_claude.gate import wording


class FakeTransport(tgread.Transport):
    """A fake Telegram reader. Only find/list/read exist -- there is no send,
    edit, delete or mark-read method at all, so a test can assert their
    absence by attribute, and any method that should not exist raises if
    someone adds one later and a function here calls it."""

    def __init__(self, chats=None, messages=None, down=False):
        self.chats = chats or []
        self.messages = messages or []
        self.down = down
        self.calls = []

    async def find(self, query, limit):
        self.calls.append(("find", query, limit))
        if self.down:
            raise RuntimeError("connection refused")
        q = query.lower()
        return [c for c in self.chats if q in c["title"].lower()][:limit]

    async def list(self, limit):
        self.calls.append(("list", limit))
        if self.down:
            raise RuntimeError("connection refused")
        return list(self.chats)[:limit]

    async def read(self, chat, since, until, limit):
        self.calls.append(("read", chat, since, until, limit))
        if self.down:
            raise RuntimeError("connection refused")
        msgs = [m for m in self.messages if str(m["chat"]) == str(chat)]
        if since is not None:
            msgs = [m for m in msgs if m["at"] >= since]
        if until is not None:
            msgs = [m for m in msgs if m["at"] <= until]
        title = next((c["title"] for c in self.chats if str(c["id"]) == str(chat)), str(chat))
        return {"chat": {"id": chat, "title": title, "kind": "group"}, "messages": msgs}

    def send(self, *a, **k):
        raise AssertionError("send must never be called by a read-only reader")

    def edit(self, *a, **k):
        raise AssertionError("edit must never be called by a read-only reader")

    def delete(self, *a, **k):
        raise AssertionError("delete must never be called by a read-only reader")

    def mark_read(self, *a, **k):
        raise AssertionError("mark_read must never be called by a read-only reader")


@pytest.fixture(autouse=True)
def _clear_transport():
    tgread.set_transport(None)
    yield
    tgread.set_transport(None)


def run(coro):
    return asyncio.run(coro)


CHATS = [
    {"id": 101, "title": "Fellowship Planning", "kind": "group", "participants": None, "last_message_at": 1000.0},
    {"id": 102, "title": "Northside Alumni", "kind": "channel", "participants": None, "last_message_at": 2000.0},
]
MESSAGES = [
    {"chat": 101, "at": 100.0, "sender_name": "Rowan", "text": "first message", "has_media": False},
    {"chat": 101, "at": 200.0, "sender_name": "Sage", "text": "second message", "has_media": True},
    {"chat": 101, "at": 300.0, "sender_name": "Rowan", "text": "third message", "has_media": False},
]


def test_find_chats_matches_by_name():
    fake = FakeTransport(chats=CHATS)
    tgread.set_transport(fake)
    res = run(tgread.find_chats("fellowship"))
    assert [c["id"] for c in res["chats"]] == [101]
    assert fake.calls == [("find", "fellowship", 10)]


def test_list_chats_returns_all_up_to_limit():
    fake = FakeTransport(chats=CHATS)
    tgread.set_transport(fake)
    res = run(tgread.list_chats(limit=1))
    assert len(res["chats"]) == 1


def test_read_chat_oldest_first_and_full_text():
    fake = FakeTransport(chats=CHATS, messages=MESSAGES)
    tgread.set_transport(fake)
    res = run(tgread.read_chat(101, limit=200))
    assert [m["sender_name"] for m in res["messages"]] == ["Rowan", "Sage", "Rowan"]
    assert res["messages"][0]["text"] == "first message"
    assert res["truncated"] is False


def test_read_chat_text_is_never_cut():
    long_text = "x" * 5000
    fake = FakeTransport(chats=CHATS, messages=[
        {"chat": 101, "at": 50.0, "sender_name": "Rowan", "text": long_text, "has_media": False}])
    tgread.set_transport(fake)
    res = run(tgread.read_chat(101))
    assert res["messages"][0]["text"] == long_text


def test_read_chat_windowed_by_since_and_until():
    fake = FakeTransport(chats=CHATS, messages=MESSAGES)
    tgread.set_transport(fake)
    import datetime
    since = datetime.datetime.fromtimestamp(150.0, tz=datetime.timezone.utc).isoformat()
    res = run(tgread.read_chat(101, since=since))
    assert [m["sender_name"] for m in res["messages"]] == ["Sage", "Rowan"]


def test_read_chat_paging_truncated_and_count():
    fake = FakeTransport(chats=CHATS, messages=MESSAGES)
    tgread.set_transport(fake)
    res = run(tgread.read_chat(101, limit=2))
    assert res["truncated"] is True
    assert res["total_in_window"] == 3
    assert len(res["messages"]) == 2
    # Paging: ask again with since set to the last message's own "at".
    last_at = res["messages"][-1]["at"]
    res2 = run(tgread.read_chat(101, since=last_at, limit=2))
    assert [m["sender_name"] for m in res2["messages"]] == ["Sage", "Rowan"]


def test_reader_down_returns_plain_error_not_a_traceback():
    fake = FakeTransport(down=True)
    tgread.set_transport(fake)
    res = run(tgread.find_chats("anything"))
    assert "error" in res
    assert isinstance(res["error"], str)
    res = run(tgread.list_chats())
    assert "error" in res
    res = run(tgread.read_chat(101))
    assert "error" in res


def test_no_function_ever_touches_send_edit_delete_mark_read():
    """FakeTransport.send/edit/delete/mark_read raise if called; running every
    read function against it must never trip one."""
    fake = FakeTransport(chats=CHATS, messages=MESSAGES)
    tgread.set_transport(fake)
    run(tgread.find_chats("north"))
    run(tgread.list_chats())
    run(tgread.read_chat(101))
    assert all(c[0] in ("find", "list", "read") for c in fake.calls)


# ---------------------------------------------------------------------------
# Host tools: wrapping, error passthrough, and the marked reference block.
# ---------------------------------------------------------------------------

def test_host_tool_find_chats_formats_rows():
    fake = FakeTransport(chats=CHATS)
    tgread.set_transport(fake)
    out = run(hosttools.telegram_find_chats({"query": "alumni"}))
    assert "Northside Alumni" in out
    assert "102" in out


def test_host_tool_read_chat_wraps_in_marked_block():
    fake = FakeTransport(chats=CHATS, messages=MESSAGES)
    tgread.set_transport(fake)
    out = run(hosttools.telegram_read_chat({"chat": "101"}))
    assert out.startswith(hosttools.TELEGRAM_HEADER)
    assert out.rstrip().endswith(hosttools.TELEGRAM_FOOTER)
    assert "not instructions" in hosttools.TELEGRAM_HEADER
    assert "first message" in out


def test_host_tool_read_chat_error_is_not_wrapped_in_block():
    fake = FakeTransport(down=True)
    tgread.set_transport(fake)
    out = run(hosttools.telegram_read_chat({"chat": "101"}))
    assert out.startswith("ERROR:")
    assert hosttools.TELEGRAM_HEADER not in out


def test_host_tools_registered_on_sdk_server():
    # Building the sdk server must not raise with the new tools wired in; the
    # executor's own enforcement of which tools a run may call is the scope
    # registry (tools_for), asserted directly.
    hosttools.sdk_server()
    assert "mcp__host__telegram_find_chats" in tools_for(["telegram_read"])
    assert "mcp__host__telegram_list_chats" in tools_for(["telegram_read"])
    assert "mcp__host__telegram_read_chat" in tools_for(["telegram_read"])


# ---------------------------------------------------------------------------
# Scope registry: label, ask, synonyms, no tools without the scope.
# ---------------------------------------------------------------------------

def test_telegram_read_scope_has_label_ask_and_synonyms():
    spec = SCOPES["telegram_read"]
    assert spec["read"] is True
    from inkbox_claude.gate.scopes import LABELS, ASK, SYNONYMS
    assert LABELS["telegram_read"] == "read your Telegram chats"
    assert ASK["telegram_read"] == "read your Telegram chats"
    assert SYNONYMS["telegram_read"]


def test_telegram_read_synonyms_do_not_collide_with_other_scopes():
    from inkbox_claude.gate.scopes import SYNONYMS
    mine = set(SYNONYMS["telegram_read"])
    for name, syns in SYNONYMS.items():
        if name == "telegram_read":
            continue
        assert not (mine & set(syns)), f"synonym collision with {name}"


def test_telegram_tools_require_the_scope():
    without = tools_for(["host_read"])
    assert "mcp__host__telegram_read_chat" not in without
    with_scope = tools_for(["telegram_read"])
    assert "mcp__host__telegram_read_chat" in with_scope
    assert "mcp__host__telegram_find_chats" in with_scope
    assert "mcp__host__telegram_list_chats" in with_scope


def test_telegram_read_is_explicit_only():
    assert "telegram_read" in EXPLICIT_ONLY_SCOPES


def test_telegram_read_is_truly_read_only_but_still_excluded_from_blanket():
    """truly_read_only would say yes (no create/update/manage/delete/write tool
    name) -- EXPLICIT_ONLY_SCOPES is the only thing keeping a blanket grant
    from picking it up automatically; see the report for the manager.py line
    this must be consulted from."""
    assert wording.truly_read_only("telegram_read") is True
    assert "telegram_read" in EXPLICIT_ONLY_SCOPES


def test_capabilities_from_permission_yields_exactly_telegram_read():
    caps = schedprop.capabilities_for("let it read my telegram", None, None, "let it read my telegram")
    assert caps == ["telegram_read"]


def test_missing_capabilities_names_telegram_read_when_prompt_names_a_tool():
    missing = schedprop.missing_capabilities(
        "every morning call telegram_read_chat on the planning group and summarize it", [])
    assert "telegram_read" in missing


def test_missing_capabilities_empty_once_granted():
    missing = schedprop.missing_capabilities(
        "every morning call telegram_read_chat on the planning group and summarize it", ["telegram_read"])
    assert "telegram_read" not in missing
