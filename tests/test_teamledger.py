"""Ledger mode (inkbox_claude/teamledger.py): Gondroid's features as switches on the gateway.

Replays Telegram updates through the real handler with every Telegram and brain call stubbed:
messages are recorded in gondroid.py's wording, only Aaron and Stanley get answers, nothing is sent
to any other chat, shadow mode sends and writes nothing. Throwaway sqlite stores and folders only."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from inkbox_claude import gateway, teamledger, telegram
from inkbox_claude.gate import graphmem
from inkbox_claude.gate.store import Store

SECRET = "s3cret"
OWNER, STANLEY, STRANGER = "999", "8804662817", "123"
GROUP, OTHER_GROUP = -5275613007, -4000000001


def _update(text, *, chat=GROUP, from_id=STANLEY, uid=1, edited=False, kind="supergroup"):
    msg = {"message_id": uid + 100, "date": int(time.time()),
           "chat": {"id": chat, "type": kind, "title": "FellowSIp of the Ring"} if kind != "private"
           else {"id": int(from_id), "type": "private"},
           "from": {"id": int(from_id), "first_name": "Stanley", "last_name": "Rabkin", "username": "srabkin"},
           "text": text}
    return {"update_id": uid, ("edited_message" if edited else "message"): msg}


class _Req:
    def __init__(self, body):
        self.headers = {telegram.SECRET_HEADER: SECRET}
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "gondroid_bot")
    monkeypatch.setenv("TELEGRAM_BOT_NAME", "Gondroid")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("GATE_APPROVER_TELEGRAM_ID", OWNER)
    monkeypatch.setenv("TELEGRAM_LEDGER", "1")
    monkeypatch.setenv("LEDGER_FILES", "1")
    monkeypatch.setenv("TELEGRAM_ANSWER_ONLY", STANLEY)
    monkeypatch.setenv("LEDGER_STATE_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("INKBOX_CLAUDE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TELEGRAM_SHADOW", raising=False)
    monkeypatch.setattr(telegram, "webhook_secret", lambda: SECRET)
    monkeypatch.setattr(teamledger, "_lib", None)
    calls = {"brain": [], "tg": []}

    async def real_tg(method, payload, timeout=20.0):
        calls["tg"].append((method, payload))
        if method == "getChatMember":
            return {"status": "member" if payload["chat_id"] == str(GROUP) else "left"}
        return {"message_id": 1}

    async def real_brain_add(content, title):
        calls["brain"].append(content)

    # gondroid.py's own Telegram and brain calls are replaced as it loads, so the shadow and
    # send-guard wrappers teamledger puts around them are the real ones under test.
    import importlib.util
    loaded = {}

    real_spec = importlib.util.spec_from_file_location

    def spec(name, path):
        s = real_spec(name, path)
        loader = s.loader
        real_exec = loader.exec_module

        def exec_module(m):
            real_exec(m)
            m.tg = real_tg
            m.brain_add = real_brain_add
            loaded["m"] = m
        loader.exec_module = exec_module
        return s

    monkeypatch.setattr(importlib.util, "spec_from_file_location", spec)
    m = teamledger.lib()
    m.ME.update({"id": 42, "username": "gondroid_bot"})
    telegram.set_send_guard(teamledger.may_send_to)
    yield SimpleNamespace(mod=m, calls=calls, tmp=tmp_path)
    telegram.set_send_guard(None)
    teamledger._lib = None


@pytest.fixture
def gw(env, tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    g.sessions = SimpleNamespace(store=store, get=lambda key: None)
    g._tg_recent = {}
    g._tg_answered = {}
    g._is_duplicate = lambda key: False
    g._index_telegram_people = lambda *a, **k: None
    g._telegram_known_name = lambda u: u
    g.dispatched = []

    async def dispatch(u):
        g.dispatched.append(u)

    async def nothing(*a, **k):
        return False

    g._telegram_dispatch = dispatch
    g._telegram_group_maybe = dispatch
    g._maybe_handle_mode_command = nothing
    g._telegram_stale_age = lambda u: None
    g._inbox_enabled = lambda: False
    g.store = store
    return g


async def _run(g, body):
    resp = await g._handle_telegram(_Req(body))
    for _ in range(5):
        await asyncio.sleep(0)
    await asyncio.gather(*list(teamledger._tasks), return_exceptions=True)
    # Some other test files swap gateway.web for a stub whose json_response returns the dict.
    return resp if isinstance(resp, dict) else json.loads(resp.text)


def test_every_group_message_is_recorded_in_gondroid_wording(gw, env):
    out = asyncio.run(_run(gw, _update("the pricing call moved to Friday", from_id=STRANGER)))
    assert out.get("ledger") is True and gw.dispatched == []          # a stranger is never answered
    assert len(env.calls["brain"]) == 1
    line = env.calls["brain"][0]
    assert line.startswith("On ") and 'in Telegram group "FellowSIp of the Ring", Stanley Rabkin (@srabkin) said: the pricing call moved to Friday' in line
    rows = [json.loads(x) for x in (env.tmp / "ledger" / "transcripts" / f"{GROUP}.jsonl").read_text().splitlines()]
    assert rows[-1]["text"] == "the pricing call moved to Friday"


def test_untrusted_group_is_ignored_entirely(gw, env):
    asyncio.run(_run(gw, _update("@gondroid_bot hi", chat=OTHER_GROUP)))
    assert env.calls["brain"] == [] and gw.dispatched == []


def test_answerer_mention_in_trusted_group_is_answered(gw, env):
    asyncio.run(_run(gw, _update("@gondroid_bot what did we decide on pricing?")))
    assert [u["text"] for u in gw.dispatched] == ["@gondroid_bot what did we decide on pricing?"]
    assert gw.dispatched[0]["addressed"] is True


def test_group_message_without_handle_is_recorded_not_answered(gw, env):
    # Blatbot would treat its plain name as addressed; Gondroid answers only an @mention.
    asyncio.run(_run(gw, _update("gondroid what did we decide?")))
    assert gw.dispatched == [] and len(env.calls["brain"]) == 1


def test_edit_is_recorded_never_answered(gw, env):
    asyncio.run(_run(gw, _update("@gondroid_bot what time?", edited=True)))
    assert gw.dispatched == [] and "edited a message to say" in env.calls["brain"][0]


def test_dm_from_stranger_is_ignored_and_from_answerer_is_answered(gw, env):
    asyncio.run(_run(gw, _update("hello", from_id=STRANGER, kind="private", uid=2)))
    assert gw.dispatched == [] and env.calls["brain"] == []
    asyncio.run(_run(gw, _update("what was said about hiring?", from_id=STANLEY, kind="private", uid=3)))
    assert len(gw.dispatched) == 1 and "a Telegram DM with Gondroid" in env.calls["brain"][0]


def test_file_request_is_answered_by_the_ledger(gw, env):
    m = env.mod
    m.save_file("PRD v2.pdf", b"%PDF-1.4 test", "FILEID", "FellowSIp of the Ring", "Aaron", "2026-11-27 10:00 UTC")
    out = asyncio.run(_run(gw, _update("@gondroid_bot send me the PRD")))
    assert out.get("file") is True and gw.dispatched == []
    assert ("sendDocument", {"chat_id": str(GROUP), "document": "FILEID", "reply_to_message_id": 101,
                             "allow_sending_without_reply": True}) in env.calls["tg"]


def test_send_guard_refuses_any_other_chat(env):
    with pytest.raises(telegram.SendRefused):
        telegram.check_send(STRANGER, "sendMessage")
    with pytest.raises(telegram.SendRefused):
        telegram.check_send(str(OTHER_GROUP), "sendMessage")
    telegram.check_send(STANLEY, "sendMessage")
    telegram.check_send(OWNER, "sendMessage")


def test_shadow_sends_and_writes_nothing(gw, env, monkeypatch):
    monkeypatch.setenv("TELEGRAM_SHADOW", "1")
    m = env.mod
    m.save_file("deck.pptx", b"x", "FID2", "FellowSIp of the Ring", "Aaron", "2026-11-27 10:00 UTC")
    asyncio.run(_run(gw, _update("@gondroid_bot send me the deck", uid=4)))
    assert env.calls["brain"] == []                                  # brain write captured, not made
    assert not [c for c in env.calls["tg"] if not c[0].startswith("get")]
    shadow = env.tmp / "home" / "shadow"
    assert "deck" in (shadow / "telegram.jsonl").read_text()
    assert "send me the deck" in (shadow / "brain.jsonl").read_text()
    sent = []

    async def boom(*a, **k):
        sent.append(a)
        raise AssertionError("must not reach Telegram")
    monkeypatch.setattr(telegram.httpx, "AsyncClient", boom)
    asyncio.run(telegram.send_message(STANLEY, "hi"))                # the gateway's own sends too
    assert sent == []


def test_poll_is_never_on_in_shadow(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("TELEGRAM_POLL", "1")
    monkeypatch.delenv("TELEGRAM_SHADOW", raising=False)
    assert telegram.poll_enabled()
    monkeypatch.setenv("TELEGRAM_SHADOW", "1")
    assert not telegram.poll_enabled()


def test_ledger_off_by_default(monkeypatch):
    monkeypatch.delenv("TELEGRAM_LEDGER", raising=False)
    assert not teamledger.enabled()


def test_do_brain_backend_reads_the_team_brain_and_never_writes(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "do_brain")
    monkeypatch.setenv("DO_BRAIN_KEY", "k")
    monkeypatch.delenv("GRAPHITI_URL", raising=False)
    asked = []

    def fake(path, payload, timeout):
        asked.append(path)
        if path == "/api/search":
            return {"facts": [{"fact": "Pricing call moved to Friday", "valid_at": "2026-11-27"}]}
        return {"memories": [{"content": "On 2026-11-27 Aaron said: ship it", "created_at": "2026-11-27"}]}
    monkeypatch.setattr(graphmem, "_brain_request", fake)
    assert graphmem.enabled()
    facts = graphmem.search("pricing", "tg:1", False, groups=[])
    assert [f["fact"] for f in facts] == ["Pricing call moved to Friday", "Recent team brain entry: On 2026-11-27 Aaron said: ship it"]
    graphmem.record("tg:1", "inbound", "hello")
    assert graphmem._queue.qsize() == 0 and asked == ["/api/search", "/api/recent?limit=20"]


def test_owner_messages_go_to_telegram_without_an_imessage_line(monkeypatch):
    from inkbox_claude.gate.manager import GateSessionManager
    sent = []

    async def send(target, text, mode, meta, **kw):
        sent.append((target, mode, kw.get("kind")))
    m = SimpleNamespace(approver_conv="", approver_telegram_id=OWNER, send=send)
    asyncio.run(GateSessionManager.send_to_approver(m, "a question for Aaron", kind="approval"))
    assert sent == [(f"tg:{OWNER}", "telegram", "approval")]
    monkeypatch.setenv("GATE_OWNER_NOTICES", "0")
    asyncio.run(GateSessionManager.send_to_approver(m, "how Stanley's question went", kind="notice"))
    assert len(sent) == 1                                       # notices off for this instance
    m2 = SimpleNamespace(approver_conv="CONV", approver_telegram_id=OWNER, send=send)
    asyncio.run(GateSessionManager.send_to_approver(m2, "x", kind="notice"))
    assert sent[-1] == ("imessage:CONV", "imessage", "notice")  # Blatbot: unchanged
