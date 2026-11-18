"""Writes to other people need a second, exact approval: a message a request would send
to anyone but Aaron is held, shown to him word for word with its own #id, and only
"#id yes" sends exactly that text, once. GATE_HOLD_SENDS_TO_OTHERS=0 restores the
earlier behaviour."""

from __future__ import annotations

import asyncio

import pytest

from inkbox_claude.gate import permhook
from inkbox_claude.gate.router import RouterOutput, RouterRequest

from test_gate import approver_meta, make_manager, stranger_meta
import test_group_plain_talk as gpt


@pytest.fixture(autouse=True)
def hold_on(monkeypatch):
    monkeypatch.setenv("GATE_HOLD_SENDS_TO_OTHERS", "1")


ANSWER = "Tuesday 6pm works. See you then."


def _approved_stranger_request(m, sent):
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="P1", scopes=["web"], summary="book chat"))
    asyncio.run(m.get("c1").handle_inbound("can we do tue 6pm?", "email", stranger_meta()))
    rid = m.store.pending()[0].id
    sent.clear()
    m.router.next = RouterOutput(reply=ANSWER, request=None)
    m.router.next_note = RouterOutput(reply=ANSWER, request=None)

    async def go():
        await m.get("aaron").handle_inbound("yes", "imessage", approver_meta())
        await asyncio.sleep(0.1)
    asyncio.run(go())
    return rid


def _held(m):
    return [r for r in m.store.pending() if (r.status or {}).get("held_message")]


def test_answer_to_a_non_owner_is_held_with_the_exact_text(tmp_path):
    m, sent = make_manager(tmp_path)
    rid = _approved_stranger_request(m, sent)
    assert m.store.get_request(rid).state == "done"
    assert not [x for x in sent if x[2] == "email"], "nothing reached the stranger"
    held = _held(m)
    assert len(held) == 1
    h = held[0].status["held_message"]
    assert h["text"] == ANSWER and h["mode"] == "email" and h["for_request"] == rid
    prompts = [t for _, t, mode, _ in sent if mode == "imessage" and f"#{held[0].id}" in t]
    assert prompts and ANSWER in prompts[0] and "To: Cand Idate" in prompts[0] and "cand@nyu.edu" not in prompts[0]


def test_yes_sends_exactly_that_text_once_and_no_sends_nothing(tmp_path):
    m, sent = make_manager(tmp_path)
    _approved_stranger_request(m, sent)
    hid = _held(m)[0].id
    sent.clear()

    async def go(text):
        await m.get("aaron").handle_inbound(text, "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go(f"#{hid} yes"))
    asyncio.run(go(f"#{hid} yes"))
    to_stranger = [t for chat, t, mode, _ in sent if mode == "email"]
    assert to_stranger == [ANSWER]
    assert m.store.get_request(hid).state == "done"

    m2, sent2 = make_manager(tmp_path / "two")
    _approved_stranger_request(m2, sent2)
    hid2 = _held(m2)[0].id
    sent2.clear()

    async def go2():
        await m2.get("aaron").handle_inbound(f"#{hid2} no", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go2())
    assert not [x for x in sent2 if x[2] == "email"]
    assert m2.store.get_request(hid2).state == "rejected"


def test_a_send_to_aaron_is_not_held(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Rate",
                                 request=RouterRequest(prompt="compute it", scopes=["web"], summary="Rate"))
    m.router.next_note = RouterOutput(reply="Done: 47%.", task="T1")

    async def go():
        await m.get("aaron").handle_inbound("what's my rate?", "imessage", approver_meta())
        await asyncio.sleep(0.1)
    asyncio.run(go())
    assert "Done: 47%." in [t for _, t, *_ in sent]
    assert _held(m) == []


def test_group_plain_talk_is_not_held(tmp_path, monkeypatch):
    monkeypatch.delenv("GATE_GROUP_PLAIN_TALK", raising=False)
    m, sent = gpt._manager(tmp_path, "Ha, fair point. Mondays are rough.")
    gpt._run_group(m, "blatbot mondays are the worst right")
    assert [t for chat, t, mode, _ in sent if chat == "tg:-100"] == ["Ha, fair point. Mondays are rough."]
    assert m.store.pending() == []


def test_tool_send_to_someone_else_is_deferred_even_for_an_approved_stranger_request(tmp_path):
    m, _ = make_manager(tmp_path)
    t = m.store.create_task("x")
    req = m.store.create_request(chat_id="c1", sender="cand@nyu.edu", sender_name="Cand", mode="email",
                                 subject="", original_message="m", summary="s", scopes=["email_send"],
                                 prompt="p", state="approved", task_id=t["id"])
    tool = "mcp__inkbox__inkbox_send_email"
    decision, _, kind = permhook.decide(req, tool, {"to": "someone@else.com", "body": "hi"}, [tool],
                                        from_owner=False, protected=[], manager=m, confirm_sends=True,
                                        owner_handle=lambda r, p: False)
    assert (decision, kind) == ("defer", "send")


def test_setting_off_restores_the_old_behaviour(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_HOLD_SENDS_TO_OTHERS", "0")
    m, sent = make_manager(tmp_path)
    _approved_stranger_request(m, sent)
    assert _held(m) == []
    assert [t for chat, t, mode, _ in sent if mode == "email"] == [ANSWER]
