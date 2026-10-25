"""Per-chat response mode (natural | mention | silent): the roommode helpers, the
Store's read/write/list methods, and the gateway's enforcement + chat-command
handling. Uses only throwaway sqlite databases -- no live credentials, no network."""

from __future__ import annotations

import os
import types

import pytest

from inkbox_claude import gateway, telegram
from inkbox_claude.gate import roommode
from inkbox_claude.gate.store import Store


def make_store(tmp_path):
    return Store(str(tmp_path / "gate.db"))


# ---------------------------------------------------------------------------
# roommode.py: pure helpers
# ---------------------------------------------------------------------------

def test_normalize_mode_accepts_known_values_only():
    assert roommode.normalize_mode("Mention") == "mention"
    assert roommode.normalize_mode(" SILENT ") == "silent"
    assert roommode.normalize_mode("") == ""
    assert roommode.normalize_mode("loud") == ""
    assert roommode.normalize_mode(None) == ""


def test_effective_mode_defaults_unset_to_mention():
    assert roommode.effective_mode("") == "mention"
    assert roommode.effective_mode(None) == "mention"
    assert roommode.effective_mode("bogus") == "mention"
    assert roommode.effective_mode("natural") == "natural"
    assert roommode.effective_mode("silent") == "silent"


def test_addressed_by_name_matches_only_at_message_start():
    assert roommode.addressed_by_name("blatbot, can you check this")
    assert roommode.addressed_by_name("Blat do this for me")
    assert roommode.addressed_by_name("@blatbot what's up")
    assert not roommode.addressed_by_name("hey is blatbot around")
    assert not roommode.addressed_by_name("blatbots are not a thing")
    assert not roommode.addressed_by_name("just chatting here")


def test_parse_mode_command_self_chat():
    assert roommode.parse_mode_command("blatbot mode mention") == (None, "mention")
    assert roommode.parse_mode_command("Blatbot Mode Silent.") == (None, "silent")
    assert roommode.parse_mode_command("  blatbot   mode   natural  ") == (None, "natural")


def test_parse_mode_command_named_chat():
    assert roommode.parse_mode_command("blatbot mode Corgi NYC Team Chat mention") == (
        "Corgi NYC Team Chat", "mention")


def test_parse_mode_command_rejects_other_text():
    assert roommode.parse_mode_command("blatbot mode loud") is None
    assert roommode.parse_mode_command("good morning everyone") is None
    assert roommode.parse_mode_command("") is None


# ---------------------------------------------------------------------------
# Store: get/set/list/find
# ---------------------------------------------------------------------------

def test_get_response_mode_unset_is_empty_string(tmp_path):
    store = make_store(tmp_path)
    assert store.get_response_mode("tg:-100") == ""


def test_set_and_get_response_mode_roundtrips(tmp_path):
    store = make_store(tmp_path)
    store.set_response_mode("tg:-100", "silent")
    assert store.get_response_mode("tg:-100") == "silent"
    store.set_response_mode("tg:-100", "Natural")
    assert store.get_response_mode("tg:-100") == "natural"


def test_set_response_mode_rejects_unknown_mode(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.set_response_mode("tg:-100", "loud")


def test_set_response_mode_does_not_disturb_other_thread_fields(tmp_path):
    store = make_store(tmp_path)
    store.set_thread("tg:-100", "idle", "telegram", {"conversation_kind": "group", "group_title": "Board"})
    store.set_response_mode("tg:-100", "silent")
    route = store.thread_route("tg:-100")
    assert route["mode"] == "telegram"
    assert route["meta"]["group_title"] == "Board"
    assert store.get_response_mode("tg:-100") == "silent"


def test_known_group_chats_lists_groups_with_effective_mode(tmp_path):
    store = make_store(tmp_path)
    store.set_thread("tg:-1", "idle", "telegram", {"conversation_kind": "group", "group_title": "Corgi NYC Team Chat"})
    store.set_thread("tg:-2", "idle", "telegram", {"conversation_kind": "group", "group_title": "Dev Chat"})
    store.set_thread("tg:555", "idle", "telegram", {"conversation_kind": "direct"})
    store.set_response_mode("tg:-2", "silent")

    groups = {g["chat_id"]: g for g in store.known_group_chats()}
    assert set(groups) == {"tg:-1", "tg:-2"}
    assert groups["tg:-1"]["response_mode"] == "mention"  # never set -> product default
    assert groups["tg:-1"]["title"] == "Corgi NYC Team Chat"
    assert groups["tg:-1"]["channel"] == "telegram"
    assert groups["tg:-2"]["response_mode"] == "silent"


def test_known_group_chats_member_count_from_history(tmp_path):
    store = make_store(tmp_path)
    store.set_thread("tg:-3", "idle", "telegram", {"conversation_kind": "group", "group_title": "Board"})
    store.add_message("tg:-3", "inbound",
                      "[telegram group: Board | from: Dana Reyes | your reply goes to the whole group]\nhi", "telegram")
    store.add_message("tg:-3", "inbound", "Lee Kim: overheard line", "telegram")
    store.add_message("tg:-3", "inbound", "Dana Reyes: a second message", "telegram")
    groups = {g["chat_id"]: g for g in store.known_group_chats()}
    assert groups["tg:-3"]["member_count"] == 2  # Dana Reyes, Lee Kim -- not recounted


def test_find_group_chat_by_name_exact_match(tmp_path):
    store = make_store(tmp_path)
    store.set_thread("tg:-1", "idle", "telegram", {"conversation_kind": "group", "group_title": "Corgi NYC Team Chat"})
    found = store.find_group_chat_by_name("corgi nyc team chat")
    assert found["chat_id"] == "tg:-1"


def test_find_group_chat_by_name_unique_substring(tmp_path):
    store = make_store(tmp_path)
    store.set_thread("tg:-1", "idle", "telegram", {"conversation_kind": "group", "group_title": "Corgi NYC Team Chat"})
    found = store.find_group_chat_by_name("Team Chat")
    assert found["chat_id"] == "tg:-1"


def test_find_group_chat_by_name_ambiguous_returns_none(tmp_path):
    store = make_store(tmp_path)
    store.set_thread("tg:-1", "idle", "telegram", {"conversation_kind": "group", "group_title": "Corgi NYC Team Chat"})
    store.set_thread("tg:-2", "idle", "telegram", {"conversation_kind": "group", "group_title": "Corgi NYC Dev"})
    assert store.find_group_chat_by_name("Corgi NYC") is None


def test_find_group_chat_by_name_unknown_returns_none(tmp_path):
    store = make_store(tmp_path)
    assert store.find_group_chat_by_name("Nobody's Chat") is None


# ---------------------------------------------------------------------------
# Gateway enforcement: _group_mode_blocks_reply (iMessage/SMS path)
# ---------------------------------------------------------------------------

@pytest.fixture
def gw(tmp_path):
    store = make_store(tmp_path)
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    g.sessions = types.SimpleNamespace(store=store)
    return g


def test_silent_mode_blocks_every_message(gw):
    gw.sessions.store.set_response_mode("sms:conv-1", "silent")
    assert gw._group_mode_blocks_reply("sms:conv-1", "blatbot please help") is True
    assert gw._group_mode_blocks_reply("sms:conv-1", "just chatting") is True


def test_mention_mode_blocks_unaddressed_allows_addressed(gw):
    gw.sessions.store.set_response_mode("sms:conv-2", "mention")
    assert gw._group_mode_blocks_reply("sms:conv-2", "just chatting") is True
    assert gw._group_mode_blocks_reply("sms:conv-2", "blatbot please help") is False
    assert gw._group_mode_blocks_reply("sms:conv-2", "Blat, what's the status") is False


def test_natural_mode_never_blocks(gw):
    gw.sessions.store.set_response_mode("sms:conv-3", "natural")
    assert gw._group_mode_blocks_reply("sms:conv-3", "just chatting") is False
    assert gw._group_mode_blocks_reply("sms:conv-3", "blatbot please help") is False


def test_default_mode_for_a_never_configured_group_is_mention(gw):
    # No set_response_mode call at all -- a brand new group chat nobody has tuned.
    assert gw._group_mode_blocks_reply("sms:conv-new", "just chatting") is True
    assert gw._group_mode_blocks_reply("sms:conv-new", "blatbot please help") is False


def test_no_store_available_fails_open_to_natural():
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    g.sessions = None
    assert g._group_mode_blocks_reply("sms:conv-x", "just chatting") is False


# ---------------------------------------------------------------------------
# Telegram-specific mention detection paths (username, reply-to, name-anywhere)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def telegram_bot_identity(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "blatbot_bot")
    monkeypatch.setenv("TELEGRAM_BOT_NAME", "Blatbot")


def _group_update(text, reply_to_bot=False, update_id=1):
    msg = {
        "message_id": "1", "chat": {"id": -100, "type": "group", "title": "Board"},
        "from": {"id": 42, "first_name": "Pat"}, "text": text,
    }
    if reply_to_bot:
        msg["reply_to_message"] = {"from": {"is_bot": True, "username": "blatbot_bot"}}
    return {"update_id": update_id, "message": msg}


def test_addressed_by_telegram_username_mention(telegram_bot_identity):
    u = telegram.parse_update(_group_update("hey @blatbot_bot can you check this"))
    assert u["is_group"] is True
    assert u["addressed"] is True


def test_addressed_by_telegram_reply_to_bot(telegram_bot_identity):
    u = telegram.parse_update(_group_update("what about this one", reply_to_bot=True))
    assert u["addressed"] is True


def test_addressed_by_telegram_name_in_text(telegram_bot_identity):
    u = telegram.parse_update(_group_update("blatbot what do you think"))
    assert u["addressed"] is True


def test_telegram_group_chatter_is_not_addressed(telegram_bot_identity):
    u = telegram.parse_update(_group_update("anyone free for lunch"))
    assert u["is_group"] is True
    assert u["addressed"] is False


# ---------------------------------------------------------------------------
# The "blatbot mode ..." chat command
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_command_sets_this_chat_mode_and_confirms(gw):
    replies = []

    async def reply(message):
        replies.append(message)

    handled = await gw._maybe_handle_mode_command(
        "blatbot mode silent", channel="telegram", is_commander=True,
        self_chat_id="tg:-9", self_label="Test Group", reply=reply)
    assert handled is True
    assert gw.sessions.store.get_response_mode("tg:-9") == "silent"
    assert replies == ["Set this chat's mode to silent."]
    assert "--" not in replies[0]


@pytest.mark.asyncio
async def test_command_from_non_commander_is_ignored(gw):
    replies = []

    async def reply(message):
        replies.append(message)

    handled = await gw._maybe_handle_mode_command(
        "blatbot mode silent", channel="telegram", is_commander=False,
        self_chat_id="tg:-9", self_label="Test Group", reply=reply)
    assert handled is False
    assert gw.sessions.store.get_response_mode("tg:-9") == ""
    assert replies == []


@pytest.mark.asyncio
async def test_command_sets_a_named_chat_from_a_dm(gw):
    gw.sessions.store.set_thread("tg:-5179779107", "idle", "telegram",
                                 {"conversation_kind": "group", "group_title": "Corgi NYC Team Chat"})
    replies = []

    async def reply(message):
        replies.append(message)

    handled = await gw._maybe_handle_mode_command(
        "blatbot mode Corgi NYC Team Chat mention", channel="telegram", is_commander=True,
        self_chat_id="tg:555", self_label="DM", reply=reply)
    assert handled is True
    assert gw.sessions.store.get_response_mode("tg:-5179779107") == "mention"
    assert replies == ["Set Corgi NYC Team Chat's mode to mention."]


@pytest.mark.asyncio
async def test_command_for_unknown_chat_name_replies_not_found_and_changes_nothing(gw):
    replies = []

    async def reply(message):
        replies.append(message)

    handled = await gw._maybe_handle_mode_command(
        "blatbot mode Nonexistent Chat silent", channel="telegram", is_commander=True,
        self_chat_id="tg:555", self_label="DM", reply=reply)
    assert handled is True
    assert replies == ["I could not find a group chat called Nonexistent Chat."]


@pytest.mark.asyncio
async def test_non_command_text_is_not_handled(gw):
    replies = []

    async def reply(message):
        replies.append(message)

    handled = await gw._maybe_handle_mode_command(
        "good morning team", channel="telegram", is_commander=True,
        self_chat_id="tg:-9", self_label="Test Group", reply=reply)
    assert handled is False
    assert replies == []


# ---------------------------------------------------------------------------
# Console API: groups_list / groups_set_mode
# ---------------------------------------------------------------------------

def test_groups_list_returns_known_group_chats(tmp_path):
    from inkbox_claude.console import api as console_api
    store = make_store(tmp_path)
    store.set_thread("tg:-1", "idle", "telegram", {"conversation_kind": "group", "group_title": "Board"})
    rows = console_api.groups_list(store)
    assert any(r["chat_id"] == "tg:-1" and r["response_mode"] == "mention" for r in rows)


def test_groups_set_mode_updates_store_and_returns_list(tmp_path):
    from inkbox_claude.console import api as console_api
    store = make_store(tmp_path)
    store.set_thread("tg:-1", "idle", "telegram", {"conversation_kind": "group", "group_title": "Board"})
    result = console_api.groups_set_mode(store, {"chat_id": "tg:-1", "response_mode": "silent"})
    assert result["ok"] is True
    assert store.get_response_mode("tg:-1") == "silent"
    row = next(g for g in result["groups"] if g["chat_id"] == "tg:-1")
    assert row["response_mode"] == "silent"


def test_groups_set_mode_rejects_unknown_mode(tmp_path):
    from inkbox_claude.console import api as console_api
    store = make_store(tmp_path)
    with pytest.raises(console_api.ValidationError):
        console_api.groups_set_mode(store, {"chat_id": "tg:-1", "response_mode": "loud"})


def test_groups_set_mode_requires_chat_id(tmp_path):
    from inkbox_claude.console import api as console_api
    store = make_store(tmp_path)
    with pytest.raises(console_api.ValidationError):
        console_api.groups_set_mode(store, {"response_mode": "mention"})
