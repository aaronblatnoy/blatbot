"""react_message: an emoji reaction on the Telegram message that started the run."""

import asyncio
import types

import pytest

from inkbox_claude import telegram
from inkbox_claude.gate import outcomes


def test_reaction_set_and_cleaning():
    assert telegram.clean_reaction("\U0001F525") == "\U0001F525"
    assert telegram.clean_reaction("❤️") == "❤"
    with pytest.raises(ValueError):
        telegram.clean_reaction("\U0001F4A3")


def test_react_goes_to_latest_inbound_in_this_chat_only(monkeypatch):
    calls = []

    async def fake_react(chat, mid, emoji):
        calls.append((chat, mid, emoji))

    monkeypatch.setattr(telegram, "react", fake_react)
    telegram.remember_inbound("111", "42")
    telegram.remember_inbound("222", "99")
    req = types.SimpleNamespace(chat_id="tg:111", id=1)
    out = asyncio.run(outcomes.react_message({"emoji": "\U0001F44D"}, None, req))
    assert out == "Reacted." and calls == [("111", "42", "\U0001F44D")]
    out = asyncio.run(outcomes.react_message({"emoji": "\U0001F44D", "message_id": "40"}, None, req))
    assert calls[-1] == ("111", "40", "\U0001F44D")


def test_react_refuses_non_telegram_and_bad_emoji():
    imsg = types.SimpleNamespace(chat_id="imessage:+15551112222", id=2)
    assert "Telegram" in asyncio.run(outcomes.react_message({"emoji": "\U0001F44D"}, None, imsg))
    tg = types.SimpleNamespace(chat_id="tg:111", id=3)
    telegram.remember_inbound("111", "7")
    assert "allows" in asyncio.run(outcomes.react_message({"emoji": "\U0001F4A3"}, None, tg))
