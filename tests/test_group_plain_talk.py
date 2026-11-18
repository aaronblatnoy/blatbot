"""Group plain talk: someone who is not Aaron, in a group, gets a tool-less reply with no
approval; anything needing a tool, lookup or action still goes to Aaron as before."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate.router import Router, RouterOutput, RouterRequest

APPROVER_CONV = "conv-aaron"
OWNER_TG = "6752193595"


class PlainRouter:
    def __init__(self, plain: str):
        self.plain = plain
        self.plain_calls = []
        self.calls = []
        self.next = RouterOutput(reply=None, request=RouterRequest(
            prompt="Check Aaron's calendar", scopes=["calendar"], summary="check calendar"))

    async def plain_talk(self, **kw):
        self.plain_calls.append(kw)
        return self.plain

    async def plan_query(self, **kw):
        return None

    async def route(self, **kw):
        self.calls.append(kw)
        return self.next.model_copy(deep=True)


def _manager(tmp_path, plain):
    sent = []

    async def send_fn(chat_id, text, mode, meta):
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV)
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    m.router = PlainRouter(plain)
    m.executor = SimpleNamespace(ran=[], run=None)
    m.approver_telegram_id = OWNER_TG
    return m, sent


def _group_meta(user_id="4242"):
    return {"sender": f"telegram:{user_id}", "to": "-100", "telegram_chat_id": "-100",
            "telegram_user_id": user_id, "conversation_kind": "group", "group_title": "Team"}


def _run_group(m, text, user_id="4242"):
    s = m.get("tg:-100")
    s.room = "--- WHERE YOU ARE ---\nThis is a group conversation on Telegram."
    asyncio.run(s.handle_inbound(text, "telegram", _group_meta(user_id)))
    return s


def test_plain_reply_in_group_goes_out_with_no_approval(tmp_path, monkeypatch):
    monkeypatch.delenv("GATE_GROUP_PLAIN_TALK", raising=False)
    m, sent = _manager(tmp_path, "Ha, fair point. Mondays are rough.")
    _run_group(m, "blatbot mondays are the worst right")
    room = [t for chat, t, mode, _ in sent if chat == "tg:-100"]
    assert room == ["Ha, fair point. Mondays are rough."]
    assert m.store.pending() == [], "no approval request for plain talk"
    assert not [t for _, t, mode, _ in sent if mode == "imessage" and t.startswith("◆ ")]
    assert m.router.calls == [], "the usual router never ran"
    hist = m.store.history("tg:-100", limit=10)
    assert any(r["kind"] == "outbound" and "Mondays" in r["text"] for r in hist)
    assert any(r["kind"] == "system" and "Plain talk" in r["text"] for r in hist)
    call = m.router.plain_calls[0]
    assert "WHERE YOU ARE" in call["room"] and call["message"] == "blatbot mondays are the worst right"
    assert all(r.get("kind") in ("inbound", "outbound") for r in call["history"])


def test_sentinel_falls_through_to_the_approval_flow(tmp_path, monkeypatch):
    monkeypatch.delenv("GATE_GROUP_PLAIN_TALK", raising=False)
    m, sent = _manager(tmp_path, Router.PLAIN_TALK_SENTINEL)
    _run_group(m, "blatbot is aaron free tuesday?")
    assert len(m.router.plain_calls) == 1
    assert len(m.store.pending()) == 1, "the request waits for Aaron as before"
    room = [t for chat, t, mode, _ in sent if chat == "tg:-100"]
    assert Router.PLAIN_TALK_SENTINEL not in " ".join(room)
    assert any(mode == "imessage" and t.startswith("◆ ") for _, t, mode, _ in sent)


def test_plain_talk_error_falls_through(tmp_path, monkeypatch):
    monkeypatch.delenv("GATE_GROUP_PLAIN_TALK", raising=False)
    m, sent = _manager(tmp_path, "unused")

    async def boom(**kw):
        raise httpx.ConnectError("down")
    m.router.plain_talk = boom
    _run_group(m, "blatbot is aaron free tuesday?")
    assert len(m.store.pending()) == 1


def test_dm_from_non_owner_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("GATE_GROUP_PLAIN_TALK", raising=False)
    m, sent = _manager(tmp_path, "should never be sent")
    meta = {"sender": "cand@nyu.edu", "to": "cand@nyu.edu", "subject": "Re: chat",
            "contact": {"id": "c1", "name": "Cand Idate", "notes": "applicant"}}
    asyncio.run(m.get("c1").handle_inbound("hello there", "email", meta))
    assert m.router.plain_calls == []
    assert len(m.store.pending()) == 1
    assert "should never be sent" not in " ".join(t for _, t, *_ in sent)


def test_owner_in_group_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("GATE_GROUP_PLAIN_TALK", raising=False)
    m, sent = _manager(tmp_path, "should never be sent")
    _run_group(m, "blatbot hi", user_id=OWNER_TG)
    assert m.router.plain_calls == []


def test_setting_off_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_GROUP_PLAIN_TALK", "0")
    m, sent = _manager(tmp_path, "should never be sent")
    _run_group(m, "blatbot mondays are the worst right")
    assert m.router.plain_calls == []
    assert len(m.store.pending()) == 1
    assert "should never be sent" not in " ".join(t for _, t, *_ in sent)


def test_plain_talk_call_has_zero_tools(monkeypatch):
    bodies = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Sure thing."}}]}

    async def post(self, url, headers=None, json=None, **kw):
        bodies.append(json)
        return Resp()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    r = Router(api_key="k", model="m")
    out = asyncio.run(r.plain_talk(room="--- WHERE YOU ARE ---", history=[{"kind": "inbound", "text": "Sam: hi"}],
                                   message="hey blatbot", sender="Sam"))
    assert out == "Sure thing."
    assert len(bodies) == 1
    body = bodies[0]
    assert set(body) <= {"model", "messages", "temperature"}, body.keys()
    assert not any(k in body for k in ("tools", "functions", "tool_choice", "function_call"))
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_sentinel_anywhere_in_output_counts_as_sentinel(monkeypatch):
    r = Router(api_key="k", model="m")

    async def chat(messages, json_mode=True):
        return "NEEDS_AARON."
    r._chat = chat
    assert asyncio.run(r.plain_talk(room="", history=[], message="find my file")) == Router.PLAIN_TALK_SENTINEL
