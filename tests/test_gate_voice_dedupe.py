"""One request per ask: voice transcript fragments of one utterance must not become
several requests, and the dedupe guard must hold both at the manager layer (voice
hand-off, _route_and_act) and at the store layer (create_request itself, closing the
race between the guard's read and the insert). Reuses the fakes from test_gate.py."""

from __future__ import annotations

import asyncio
import threading

import pytest

from inkbox_claude.gate.manager import OWNER_THREAD
from inkbox_claude.gate.router import RouterOutput, RouterRequest
from inkbox_claude.gate.store import Store

from test_gate import (
    AARON_PHONE,
    FakeRouter,
    approver_meta,
    make_manager,
    phone_manager,
    stranger_meta,
    voice_meta,
)


def _always_ungrounded_picker():
    class Picker:
        async def judge_grounded(self, **kw):
            return {"grounded": False, "p": 0.1, "reason": "unsupported"}
    return Picker()


# -- voice: one request per ask -------------------------------------------------

def test_three_voice_fragments_within_22s_create_one_request_and_attach_the_rest(tmp_path):
    """The realtime voice model can hand one spoken request to voice_consult as several
    transcript fragments a few seconds apart. The first fragment's request must still be
    open (approved/running) when the later fragments arrive -- simulated here by holding
    the executor until all three fragments have been sent."""
    m, sent = phone_manager(tmp_path, trust=True)
    hold = asyncio.Event()

    async def run(req, context="", prior_work="", model=None, **kw):
        m.executor.ran.append(req.id)
        await hold.wait()
        return {"ok": True, "summary": "did it", "tool_calls": ["mcp__tamid-drive__manage_event"], "raw": "did it"}
    m.executor.run = run

    m.router.next = RouterOutput(reply=None, request=RouterRequest(
        prompt="list every definition", scopes=["tamid_drive_read"], summary="list definitions"))

    async def go():
        s = m.get("p1")
        t1 = asyncio.create_task(s.voice_consult("give me a list of every single definition", voice_meta()))
        await asyncio.sleep(0)  # let fragment 1 reach create_request + the blocked run
        out2 = await s.voice_consult("and every formula, text me", voice_meta())
        out3 = await s.voice_consult("Got it.", voice_meta())
        hold.set()
        out1 = await t1
        return out1, out2, out3

    out1, out2, out3 = asyncio.run(go())

    # Exactly one request exists on the whole store, however many fragments arrived.
    total = m.store._db.execute("SELECT COUNT(*) AS c FROM requests").fetchone()["c"]
    assert total == 1
    assert m.store.get_request(2) is None and m.store.get_request(3) is None
    assert m.executor.ran == [1]  # the executor only ever ran once

    # The later fragments were folded in as context, not dropped: the thread's own
    # record carries all three fragments as inbound text.
    history = m.store.history(OWNER_THREAD, limit=20)
    inbound_texts = [h["text"] for h in history if h["kind"] == "inbound"]
    assert "and every formula, text me" in inbound_texts
    assert "Got it." in inbound_texts
    assert out2 and out3  # each fragment still got some spoken reply


def test_voice_fragment_while_request_pending_not_yet_approved_creates_none(tmp_path):
    """A non-approver caller: the first fragment's request waits on Aaron ('pending');
    a second fragment must not start a second one either."""
    m, sent = phone_manager(tmp_path, trust=True)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(
        prompt="book it", scopes=["calendar"], summary="book chat"))
    s = m.get("p1")
    out1 = asyncio.run(s.voice_consult("book me tuesday", voice_meta("+12125550100")))
    assert len(m.store.pending()) == 1
    out2 = asyncio.run(s.voice_consult("at six please", voice_meta("+12125550100")))
    assert len(m.store.pending()) == 1  # still just the one, already-covered path
    assert out2


# -- text message while a request is approved-but-not-yet-running --------------

def test_text_message_while_request_approved_not_running_creates_none(tmp_path):
    m, sent = make_manager(tmp_path)
    task = m.store.create_task("investigate", [("cand@nyu.edu", "Cand Idate")])
    r = m.store.create_request(
        chat_id="c1", sender="cand@nyu.edu", sender_name="Cand Idate", mode="email", subject="",
        original_message="first ask", summary="original ask", scopes=["web"], prompt="p",
        state="approved", task_id=task["id"])
    m.router.next = RouterOutput(reply="Still working on it.", request=None)

    asyncio.run(m.get("c1").handle_inbound("any update?", "email", stranger_meta()))

    assert m.store.get_request(r.id).state == "approved"
    assert m.store.get_request(2) is None
    total = m.store._db.execute("SELECT COUNT(*) AS c FROM requests").fetchone()["c"]
    assert total == 1
    assert any(x[1] == "Still working on it." for x in sent)
    events = m.store.task_with_events(task["id"])["events"]
    assert any("any update?" in (e.get("body") or e.get("text") or "") for e in events)


# -- the race itself, closed in the store -----------------------------------------

def test_concurrent_create_request_calls_for_one_thread_yield_one_request(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    t = s.create_task("investigate")
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = s.create_request(chat_id="c1", sender="cand@nyu.edu", sender_name="", mode="email",
                             subject="", original_message="m", summary="s", scopes=["web"],
                             prompt="p", state="approved", task_id=t["id"])
        results.append(r.id)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(set(results)) == 1
    total = s._db.execute("SELECT COUNT(*) AS c FROM requests").fetchone()["c"]
    assert total == 1


def test_create_request_dedupe_window_expires(tmp_path):
    """Past the dedupe window, a genuinely new ask on the same thread gets its own
    request -- this is a dedupe guard against a race, not a lock on the thread."""
    s = Store(str(tmp_path / "g.db"))
    t = s.create_task("investigate")
    r1 = s.create_request(chat_id="c1", sender="x", sender_name="", mode="email", subject="",
                          original_message="m1", summary="s1", scopes=["web"], prompt="p1",
                          state="approved", task_id=t["id"])
    # Backdate it past the window.
    s._db.execute("UPDATE requests SET created_at=created_at-200 WHERE id=?", (r1.id,))
    s._db.commit()
    r2 = s.create_request(chat_id="c1", sender="x", sender_name="", mode="email", subject="",
                          original_message="m2", summary="s2", scopes=["web"], prompt="p2",
                          state="approved", task_id=t["id"])
    assert r2.id != r1.id


# -- scheduled runs never block, and are never blocked ----------------------------

def test_schedule_requests_never_block_and_are_never_blocked(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    t = s.create_task("recurring report")
    sched_req = s.create_request(chat_id="c1", sender="sched", sender_name="", mode="email", subject="",
                                 original_message="m1", summary="scheduled run", scopes=["web"],
                                 prompt="p1", state="approved", task_id=t["id"],
                                 schedule_id=1, schedule_kind="report")
    # A scheduled run "in flight" on this thread does not block a fresh owner request.
    owner_req = s.create_request(chat_id="c1", sender="owner", sender_name="", mode="imessage", subject="",
                                 original_message="m2", summary="owner ask", scopes=["web"],
                                 prompt="p2", state="approved", task_id=t["id"])
    assert owner_req.id != sched_req.id

    # And an owner request now open does not block a fresh scheduled run either.
    sched_req2 = s.create_request(chat_id="c1", sender="sched", sender_name="", mode="email", subject="",
                                  original_message="m3", summary="scheduled run 2", scopes=["web"],
                                  prompt="p3", state="approved", task_id=t["id"],
                                  schedule_id=2, schedule_kind="report")
    assert sched_req2.id not in (sched_req.id, owner_req.id)

    # running_for_thread (the guard both call sites use) only ever sees the owner
    # request -- schedule requests are excluded from "busy" both ways.
    running = s.running_for_thread("c1")
    assert running is not None and running.id == owner_req.id


# -- the fallback wording never speaks about Aaron in the third person to Aaron ---

def test_fallback_unverified_caller_from_owners_number_gets_plain_first_person_wording(tmp_path):
    """GATE_VOICE_TRUST_APPROVER is off: caller ID can be spoofed, so is_approver() is
    correctly False from Aaron's own number. The fallback wording still must not call
    him "Aaron" in the third person -- and he must gain no owner power from this."""
    m, sent = phone_manager(tmp_path, trust=False)
    m.task_picker = _always_ungrounded_picker()
    s = m.get("p1")
    s.mode, s.reply_meta = "voice", voice_meta()  # sender == AARON_PHONE

    assert s.is_approver() is False  # untrusted number: no owner power
    assert s._is_owner_person() is True  # but it is recognisably his wording case

    out = asyncio.run(s.ground_reply(
        "Here is a guess.", "Result:\nsome raw tool output",
        [{"kind": "inbound", "text": "what happened?"}], "", "", s.is_approver()))

    assert out == "I don't have an answer for that yet."
    assert "Aaron" not in out
    # No owner power leaked through the wording decision.
    assert s.is_approver() is False
    assert s.trust().get("scopes") == []


def test_fallback_contact_resolving_to_commander_gets_plain_wording_too(tmp_path):
    """A different channel entirely (no phone number involved): a sender whose contact
    record resolves to the commander's person entry is still recognised for wording,
    still without is_approver() or trust() granting anything."""
    m, sent = make_manager(tmp_path)
    m.store.set_approver_handles(phone=AARON_PHONE)
    person = m.store.create_person(display="Aaron", role="", scopes=[])
    m.store.add_contact(person["id"], "phone", AARON_PHONE)
    assert m.store.commander_person_id() == person["id"]
    m.task_picker = _always_ungrounded_picker()

    s = m.get("voicecall1")
    s.mode = "voice"
    s.reply_meta = {"sender": AARON_PHONE, "contact": {"id": "c9", "name": "Unverified"}}
    assert s.is_approver() is False
    assert s.is_aaron_on_phone() is False  # m.approver_phone is unset in this manager
    assert s._is_owner_person() is True  # resolved via the contact, not the phone check

    out = asyncio.run(s.ground_reply(
        "Here is a guess.", "Result:\nsome raw tool output",
        [{"kind": "inbound", "text": "what happened?"}], "", "", s.is_approver()))

    assert out == "I don't have an answer for that yet."
    assert s.is_approver() is False
    assert s.trust().get("scopes") == []  # commander role scopes are always empty


def test_fallback_stranger_still_gets_the_existing_third_person_line(tmp_path):
    m, sent = make_manager(tmp_path)
    m.task_picker = _always_ungrounded_picker()
    s = m.get("c1")
    s.mode, s.reply_meta = "email", stranger_meta()

    assert s.is_approver() is False
    assert s._is_owner_person() is False

    out = asyncio.run(s.ground_reply(
        "Here is a guess.", "Result:\nsome raw tool output",
        [{"kind": "inbound", "text": "what happened?"}], "", "", s.is_approver()))

    assert out == ("I do not have an answer ready for that right now. I have let Aaron "
                   "know, and he will follow up if there is more to share.")
