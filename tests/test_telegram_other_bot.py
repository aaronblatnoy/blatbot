"""A group message aimed at a different bot is never Blatbot's: not judged, not filed as a
task, not turned into an approval request. Replays request #504's message
("@Shire_analyst_bot add this to my gcal") through the real Telegram handler with sending
stubbed. Throwaway sqlite stores only."""

from __future__ import annotations

import asyncio
import time
import types
from types import SimpleNamespace

import pytest

from inkbox_claude import gateway, telegram
from inkbox_claude.gate.store import Store

SECRET = "s3cret"
CHAT = -5275613007
MSG_504 = "@Shire_analyst_bot add this to my gcal"


def _update(text, *, reply_to_blatbot=False, from_id=8804662817, uid=1):
    msg = {"message_id": 7, "date": int(time.time()), "chat": {"id": CHAT, "type": "supergroup",
           "title": "FellowSIp of the Ring"},
           "from": {"id": from_id, "first_name": "Stanley", "last_name": "Rabkin"}, "text": text}
    if reply_to_blatbot:
        msg["reply_to_message"] = {"message_id": 6, "text": "Moved: ...",
                                   "from": {"id": 1, "is_bot": True, "username": "bh_blatbot"}}
    return {"update_id": uid, "message": msg}


async def _post(g, body):
    resp = await g._handle_telegram(_Req(body))
    await asyncio.sleep(0)  # an addressed message is dispatched on a background task
    return resp


class _Req:
    def __init__(self, body):
        self.headers = {telegram.SECRET_HEADER: SECRET}
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def gw(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "bh_blatbot")
    monkeypatch.setenv("TELEGRAM_BOT_NAME", "Blatbot")
    monkeypatch.setattr(telegram, "enabled", lambda: True)
    monkeypatch.setattr(telegram, "webhook_secret", lambda: SECRET)
    monkeypatch.setattr(telegram, "approver_id", lambda: "999")
    store = Store(str(tmp_path / "gate.db"))
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    g.sessions = SimpleNamespace(store=store, get=lambda key: None)
    g._tg_recent = {}
    g._is_duplicate = lambda key: False
    g._index_telegram_people = lambda *a, **k: None
    g._telegram_known_name = lambda u: u
    g.dispatched, g.judged = [], []

    async def dispatch(u):
        g.dispatched.append(u)

    async def judge(u):
        g.judged.append(u)

    async def command(text, **kw):
        return False

    async def stale(u):
        return None

    g._telegram_dispatch = dispatch
    g._telegram_group_maybe = judge
    g._maybe_handle_mode_command = command
    g._telegram_stale_age = lambda u: None
    return g


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [False, True])
async def test_request_504_message_is_never_blatbots(gw, reply):
    await _post(gw, _update(MSG_504, reply_to_blatbot=reply))
    assert gw.dispatched == [] and gw.judged == []
    store = gw.sessions.store
    assert store.pending() == []
    assert store.requests_in_state("pending") == []
    # Heard and recorded in the room's history, nothing more.
    assert any("Shire_analyst_bot" in (r.get("text") or "") for r in store.history(f"tg:{CHAT}", 10))


@pytest.mark.asyncio
async def test_other_bot_message_from_the_owner_is_also_not_blatbots(gw):
    await _post(gw, (_update(MSG_504, from_id=999)))
    assert gw.dispatched == [] and gw.judged == []


@pytest.mark.asyncio
async def test_naming_blatbot_alongside_another_bot_is_still_blatbots(gw):
    await _post(gw, (_update("@bh_blatbot ask @Shire_analyst_bot about it")))
    assert len(gw.dispatched) == 1
    await _post(gw, (_update("Blatbot, and @Shire_analyst_bot too", uid=2)))
    assert len(gw.dispatched) == 2


@pytest.mark.asyncio
async def test_reply_to_blatbot_without_another_bot_is_still_blatbots(gw):
    await _post(gw, (_update("add this to my gcal", reply_to_blatbot=True)))
    assert len(gw.dispatched) == 1


@pytest.mark.asyncio
async def test_mentioning_a_person_does_not_silence_blatbot_in_a_reply(gw):
    await _post(gw, (_update("@aaron_b see this", reply_to_blatbot=True)))
    assert len(gw.dispatched) == 1


@pytest.mark.asyncio
async def test_judgment_path_also_refuses_it_even_when_the_writer_model_would_say_yes(gw):
    # _telegram_group_maybe itself must not reach the judge or the fallback.
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    called = []

    class Picker:
        async def judge_group_reply(self, **kw):
            called.append("judge")
            return {"should_reply": True, "p": 1.0}

    g.sessions = types.SimpleNamespace(store=gw.sessions.store, task_picker=Picker(), router=None)
    g._tg_recent = {}
    g._telegram_dispatch = lambda u: called.append("dispatch")
    await g._telegram_group_maybe({"chat_id": str(CHAT), "from_id": "7", "name": "S", "text": MSG_504,
                                   "group_title": "G"})
    assert called == []


def test_for_other_bot_rules(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "bh_blatbot")
    monkeypatch.setenv("TELEGRAM_BOT_NAME", "Blatbot")
    assert telegram.for_other_bot(MSG_504)
    assert telegram.for_other_bot("hey @Some_Other_BOT do it")
    assert telegram.for_other_bot("/start@other_bot")
    assert not telegram.for_other_bot("@bh_blatbot do it")
    assert not telegram.for_other_bot("blatbot please ping @shire_analyst_bot")
    assert not telegram.for_other_bot("email me at a@robot.com")
    assert not telegram.for_other_bot("@aaron_b hi")
    assert not telegram.for_other_bot("add this to my gcal")
