"""Held messages, 2026-11-18 live defects: a bare yes answers the newest held message
(no new request), a run whose send is held waits instead of failing and Aaron gets only
the prompt, and Aaron's own question asked in a group is answered there with no second
yes while sends anywhere else stay held."""

from __future__ import annotations

import asyncio

import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate import outcomes, permhook, runguard

from test_gate import approver_meta, make_manager

AARON = "+15550100001"
GROUP = "whatsapp:3c5b6491-8adc-4f8c-9757-c883a32b09dc"
GROUP_META = {"conversation_kind": "group", "group_title": "Presidents of Tamid & Entourage",
              "conversation_id": "3c5b6491-8adc-4f8c-9757-c883a32b09dc"}


@pytest.fixture(autouse=True)
def hold_on(monkeypatch):
    monkeypatch.setenv("GATE_HOLD_SENDS_TO_OTHERS", "1")
    monkeypatch.setenv("GATE_RUN_OUTCOMES", "1")


def _manager(tmp_path):
    m, sent = make_manager(tmp_path)
    m.approver_phone = AARON
    return m, sent


def _request(m, *, chat_id, sender, mode, state="running", summary="s"):
    t = m.store.create_task("x")
    return m.store.create_request(chat_id=chat_id, sender=sender, sender_name="A", mode=mode, subject="",
                                  original_message="m", summary=summary, scopes=["web"], prompt="p",
                                  state=state, task_id=t["id"])


def _held(m):
    return [r for r in m.store.pending() if (r.status or {}).get("held_message")]


async def _send_as(m, req, target, text, meta):
    token = runguard.bind(req.id)
    try:
        return await m.send(target, text, "whatsapp", dict(meta), chat_id=target, record=False)
    finally:
        runguard.unbind(token)


# -- defect 1: a bare yes answers the newest held message ----------------------------

def test_bare_yes_sends_the_newest_held_message_and_names_the_others(tmp_path):
    m, sent = _manager(tmp_path)
    origin = _request(m, chat_id="sean-chat", sender="sean@x.com", mode="email", state="done")

    async def hold_two():
        await m.hold_message(origin, GROUP, "First answer.", "whatsapp", dict(GROUP_META), chat_id=GROUP)
        await m.hold_message(origin, GROUP, "Second answer.", "whatsapp", dict(GROUP_META), chat_id=GROUP)
    asyncio.run(hold_two())
    older, newest = sorted(_held(m), key=lambda r: r.id)
    before = len(m.store.pending())
    sent.clear()

    async def go():
        await m.get("aaron").handle_inbound("Yes", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())

    assert [t for chat, t, mode, _ in sent if chat == GROUP] == ["Second answer."]
    assert m.store.get_request(newest.id).state == "done"
    assert m.store.get_request(older.id).state == "pending"
    to_aaron = [t for _, t, mode, _ in sent if mode == "imessage"]
    label = newest.status["held_message"]["label"]
    assert to_aaron == [f"Sent #{newest.id} to {label}. #{older.id} is still waiting for your yes or no."]
    assert len(m.store.pending()) == before - 1, "no new request was created"


def test_bare_no_discards_the_newest_held_message(tmp_path):
    m, sent = _manager(tmp_path)
    origin = _request(m, chat_id="sean-chat", sender="sean@x.com", mode="email", state="done")
    asyncio.run(m.hold_message(origin, GROUP, "Only answer.", "whatsapp", dict(GROUP_META), chat_id=GROUP))
    hid = _held(m)[0].id
    sent.clear()

    async def go():
        await m.get("aaron").handle_inbound("no", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert not [x for x in sent if x[0] == GROUP]
    assert m.store.get_request(hid).state == "rejected"
    assert [t for _, t, mode, _ in sent if mode == "imessage"] == [f"Discarded #{hid}. Nothing was sent."]


# -- defect 2: a run whose send is held waits, and Aaron gets only the prompt ----------

def _fake_dest(manager, req, to, surface):
    return {"target": GROUP, "mode": "whatsapp", "meta": dict(GROUP_META), "record": GROUP,
            "who": "the Presidents of Tamid & Entourage group", "requester": False, "owner": False}


def test_send_message_tool_is_let_through_to_be_held_not_deferred(tmp_path, monkeypatch):
    m, _ = _manager(tmp_path)
    monkeypatch.setattr(outcomes, "resolve", _fake_dest)
    req = _request(m, chat_id=gm.OWNER_THREAD, sender=AARON, mode="imessage")
    decision, _, _ = permhook.decide(req, outcomes.SEND_TOOL, {"to": "Presidents", "text": "hi"},
                                     [outcomes.SEND_TOOL], from_owner=True, protected=[], manager=m,
                                     confirm_sends=True, owner_handle=lambda r, p: False)
    assert decision == "allow"


def test_held_tool_sends_end_the_run_waiting_and_yes_finishes_it(tmp_path, monkeypatch):
    m, sent = _manager(tmp_path)
    monkeypatch.setattr(outcomes, "resolve", _fake_dest)
    req = _request(m, chat_id=gm.OWNER_THREAD, sender=AARON, mode="imessage", state="approved",
                   summary="Reply to Sean in the group")
    log: dict = {}

    async def run(r, **kw):
        token = runguard.bind(r.id)
        try:
            # Two held sends in one run: no limit of one.
            a = await outcomes.send_message({"to": "Presidents", "text": "Sean, here it is."}, m, r, log)
            b = await outcomes.send_message({"to": "Presidents", "text": "And the deck too."}, m, r, log)
        finally:
            runguard.unbind(token)
        assert a.startswith("Held for Aaron's yes.") and b.startswith("Held for Aaron's yes.")
        return {"ok": False, "summary": "The send was blocked. Stopping and waiting for direction.",
                "raw": "", "tool_calls": [outcomes.SEND_TOOL], "outcome": "held",
                "held_sends": log["held_sends"], "sends": []}
    m.executor.run = lambda r, **kw: run(r, **kw)
    sent.clear()
    asyncio.run(m.execute(m.store.get_request(req.id)))

    assert not [x for x in sent if x[0] == GROUP], "nothing reached the group"
    to_aaron = [t for _, t, mode, _ in sent if mode == "imessage"]
    held = sorted(_held(m), key=lambda r: r.id)
    assert len(held) == 2 and len(to_aaron) == 2
    assert all(t.endswith(f"#{h.id} yes / no") for t, h in zip(to_aaron, held))
    assert not any("blocked" in t or "failed" in t.lower() for t in to_aaron)
    assert m.store.get_request(req.id).state == "waiting"

    async def say(text):
        await m.get("aaron").handle_inbound(text, "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(say(f"#{held[0].id} yes"))
    assert m.store.get_request(req.id).state == "waiting", "one held message is still open"
    asyncio.run(say(f"#{held[1].id} yes"))
    assert [t for chat, t, *_ in sent if chat == GROUP] == ["Sean, here it is.", "And the deck too."]
    assert m.store.get_request(req.id).state == "done"


# -- defect 3: Aaron's own question in a group is answered there without a second yes --

def test_aarons_question_in_a_group_is_answered_there_without_a_second_yes(tmp_path):
    m, sent = _manager(tmp_path)
    req = _request(m, chat_id=GROUP, sender=f"whatsapp:{AARON}", mode="whatsapp")
    sent.clear()
    out = asyncio.run(_send_as(m, req, GROUP, "Jacob is on the Fund track.", GROUP_META))
    assert out is not gm.HELD_FOR_AARON
    assert [t for chat, t, *_ in sent if chat == GROUP] == ["Jacob is on the Fund track."]
    assert _held(m) == []


def test_aarons_group_run_sending_anywhere_else_is_still_held(tmp_path):
    m, sent = _manager(tmp_path)
    req = _request(m, chat_id=GROUP, sender=f"whatsapp:{AARON}", mode="whatsapp")
    out = asyncio.run(_send_as(m, req, "whatsapp:+15550199999", "Hi Sean.", {"name": "Sean Parnell"}))
    assert out is gm.HELD_FOR_AARON
    assert not [x for x in sent if x[0] == "whatsapp:+15550199999"]
    assert len(_held(m)) == 1


def test_someone_elses_question_in_the_group_is_still_held(tmp_path):
    m, sent = _manager(tmp_path)
    req = _request(m, chat_id=GROUP, sender="whatsapp:+15550177777", mode="whatsapp")
    out = asyncio.run(_send_as(m, req, GROUP, "Here you go.", GROUP_META))
    assert out is gm.HELD_FOR_AARON
    assert not [x for x in sent if x[0] == GROUP]
