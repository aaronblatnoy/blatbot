"""A Telegram message delivered long after it was sent (a backlog released after an
outage) is stored but not acted on. Throwaway sqlite stores only; nothing is sent."""

from __future__ import annotations

import asyncio
import time
import types
from types import SimpleNamespace

import pytest

from inkbox_claude import gateway, telegram
from inkbox_claude.gate import manager as gm
from inkbox_claude.gate.store import Store

SECRET = "s3cret"


def _update(text, *, age_s, group=False, chat_id=None, from_id=111):
    chat = {"id": chat_id or (-42 if group else from_id), "type": "supergroup" if group else "private"}
    if group:
        chat["title"] = "Test Group"
    return {"update_id": int(time.time() * 1000) % 10**9 + int(age_s),
            "message": {"message_id": 7, "date": int(time.time() - age_s), "chat": chat,
                        "from": {"id": from_id, "first_name": "Sam"}, "text": text}}


class _Req:
    def __init__(self, body):
        self.headers = {telegram.SECRET_HEADER: SECRET}
        self._body = body

    async def json(self):
        return self._body


class _Session:
    def __init__(self):
        self.filed = []

    async def file_inbound(self, text, mode, meta):
        self.filed.append((text, mode, meta))
        return 1


@pytest.fixture
def gw(tmp_path, monkeypatch):
    monkeypatch.setattr(telegram, "enabled", lambda: True)
    monkeypatch.setattr(telegram, "webhook_secret", lambda: SECRET)
    monkeypatch.setattr(telegram, "approver_id", lambda: "999")
    monkeypatch.delenv("GATE_TELEGRAM_STALE_MIN", raising=False)
    store = Store(str(tmp_path / "gate.db"))
    session = _Session()
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    g.sessions = SimpleNamespace(store=store, get=lambda key: session)
    g._tg_recent = {}
    g._is_duplicate = lambda key: False
    g._index_telegram_people = lambda *a, **k: None
    g._telegram_known_name = lambda u: u
    g.dispatched, g.judged, g.commands = [], [], []

    async def dispatch(u):
        g.dispatched.append(u)

    async def judge(u):
        g.judged.append(u)

    async def command(text, **kw):
        g.commands.append(text)
        return False

    g._telegram_dispatch = dispatch
    g._telegram_group_maybe = judge
    g._maybe_handle_mode_command = command
    g.session = session
    return g


async def _post(g, body):
    resp = await g._handle_telegram(_Req(body))
    await asyncio.sleep(0)  # let the background dispatch task run
    return resp


def test_parse_update_carries_the_send_time():
    u = telegram.parse_update(_update("hi", age_s=0))
    assert abs(u["date"] - time.time()) < 5
    assert telegram.parse_update({"message": {"chat": {"id": 1}, "text": "x"}})["date"] == 0


@pytest.mark.asyncio
async def test_fresh_message_is_handled_as_before(gw):
    await _post(gw, _update("what's on friday", age_s=30))
    assert [u["text"] for u in gw.dispatched] == ["what's on friday"]
    assert gw.commands == ["what's on friday"]
    assert gw.session.filed == []


@pytest.mark.asyncio
async def test_stale_direct_message_is_filed_and_not_acted_on(gw, caplog):
    caplog.set_level("INFO")
    await _post(gw, _update("book me a table", age_s=11 * 60))
    assert gw.dispatched == [] and gw.judged == [] and gw.commands == []
    assert len(gw.session.filed) == 1
    text, mode, meta = gw.session.filed[0]
    assert text == "book me a table" and mode == "telegram"
    assert meta["conversation_kind"] == "direct" and meta["sender"] == "telegram:111"
    assert sum("filed, not acted on" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_stale_group_message_is_stored_in_the_group_record_only(gw):
    await _post(gw, _update("blatbot who is free tomorrow", age_s=3600, group=True))
    assert gw.dispatched == [] and gw.judged == [] and gw.commands == []
    rows = gw.sessions.store.history("tg:-42", limit=5)
    assert [r["text"] for r in rows] == ["Sam: blatbot who is free tomorrow"]
    assert rows[0]["kind"] == "inbound"


@pytest.mark.asyncio
async def test_setting_changes_the_limit(gw, monkeypatch):
    monkeypatch.setenv("GATE_TELEGRAM_STALE_MIN", "60")
    await _post(gw, _update("twenty minutes late", age_s=20 * 60))
    assert len(gw.dispatched) == 1 and gw.session.filed == []
    await _post(gw, _update("two hours late", age_s=2 * 3600))
    assert len(gw.dispatched) == 1 and len(gw.session.filed) == 1
    monkeypatch.setenv("GATE_TELEGRAM_STALE_MIN", "0")  # off
    await _post(gw, _update("a day late", age_s=86400))
    assert len(gw.dispatched) == 2


def _manager(tmp_path):
    sent = []

    async def send_fn(chat_id, text, mode, meta):
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id="approver-conv")
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))

    class NoRouter:
        async def route(self, **kw):
            raise AssertionError("a filed message must not reach the reply writer")

    m.router = NoRouter()
    return m, sent


@pytest.mark.asyncio
async def test_session_files_without_a_reply_or_a_request(tmp_path):
    m, sent = _manager(tmp_path)
    s = m.get("tg:555")
    before = (s.mode, s.reply_meta)
    meta = {"sender": "telegram:555", "to": "555", "conversation_kind": "direct", "contact": {"name": "Sam"}}
    mid = await s.file_inbound("  can you check my calendar  ", "telegram", meta)
    assert mid is not None
    rows = m.store.history("tg:555", limit=5)
    assert [r["text"] for r in rows] == ["can you check my calendar"]
    assert sent == []
    assert (s.mode, s.reply_meta) == before
