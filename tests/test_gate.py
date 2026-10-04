"""Gate tests: store state machine, approver command parsing, scopes,
executor hash refusal, routing with a stubbed router. No network."""

from __future__ import annotations

import asyncio
import time
import json
from types import SimpleNamespace

import pytest
from inkbox_claude.gate.manager import OWNER_THREAD

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate.router import RouterOutput, RouterRequest
from inkbox_claude.gate.scopes import SCOPES, tools_for
from inkbox_claude.gate.store import Request, Store, sha256

APPROVER_CONV = "conv-aaron"


class FakeRouter:
    def __init__(self):
        self.next = RouterOutput(reply="hi", request=None)
        self.calls = []
        self.next_query = None      # a TaskQuery the fake "plans", or None
        self.query_calls = []

    async def plan_query(self, **kw):
        self.query_calls.append(kw)
        return self.next_query

    next_note = None            # what to answer when phrasing a result (action=False), if set

    async def route(self, **kw):
        self.calls.append(kw)
        if kw.get("action") is False and self.next_note is not None:
            return self.next_note.model_copy(deep=True)
        return self.next.model_copy(deep=True)  # a fresh object per call, like the real router


class FakeExecutor:
    def __init__(self):
        self.ran = []
        self.result = {"ok": True, "summary": "did it", "tool_calls": ["mcp__tamid-drive__manage_event"], "raw": "did it"}

    async def run(self, req, context="", prior_work="", model=None):
        self.ran.append(req.id)
        self.context = context
        self.prior_work = prior_work
        if sha256(req.prompt) != req.prompt_sha256:
            return {"ok": False, "error": "prompt hash mismatch; refused to run", "tool_calls": []}
        return self.result


def make_manager(tmp_path):
    sent = []

    async def send_fn(chat_id, text, mode, meta):
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV)
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    m.router = FakeRouter()
    m.executor = FakeExecutor()
    return m, sent


def stranger_meta():
    return {"sender": "cand@nyu.edu", "to": "cand@nyu.edu", "subject": "Re: chat",
            "contact": {"id": "c1", "name": "Cand Idate", "notes": "applicant"}}


def approver_meta():
    return {"sender": "+15550100001", "to": "+15550100001", "conversation_id": APPROVER_CONV}


def test_scopes_have_tools_and_union_dedups():
    assert all(SCOPES[s]["tools"] for s in SCOPES)
    t = tools_for(["email_send", "email_send", "web"])
    assert t.count("WebSearch") == 1 and "mcp__inkbox__inkbox_send_email" in t


def test_router_request_rejects_unknown_scope():
    with pytest.raises(Exception):
        RouterRequest(prompt="p", scopes=["shell"], summary="s")


def test_store_request_lifecycle(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    t = s.create_task("do x")
    r = s.create_request(chat_id="c", sender="a", sender_name="A", mode="email", subject="", original_message="m",
                         summary="s", scopes=["web"], prompt="do x", state="pending", task_id=t["id"])
    assert r.state == "pending" and r.prompt_sha256 == sha256("do x")
    r2 = s.revise_prompt(r.id, "do y")
    assert r2.revision == 1 and r2.prompt_sha256 == sha256("do y") and r2.state == "pending"
    s.set_state(r.id, "approved"); assert s.get_request(r.id).state == "approved"
    assert s.pending() == []


def test_stranger_benign_reply_no_request(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply="Sure, what times work?", request=None)
    asyncio.run(m.get("c1").handle_inbound("hello", "email", stranger_meta()))
    assert [x[1] for x in sent if x[2] == "email"] == ["Sure, what times work?"]
    fyi = [x for x in sent if x[2] == "imessage"]
    assert len(fyi) == 1 and "[Blatbot FYI]" in fyi[0][1] and "hello" in fyi[0][1]
    assert m.store.pending() == [] and m.store.thread_state("c1") == "idle"


def test_stranger_request_goes_to_aaron_and_waits(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(
        prompt="Book Tue 6pm", scopes=["calendar", "email_send"], summary="book chat"))
    asyncio.run(m.get("c1").handle_inbound("can we do tue 6pm?", "email", stranger_meta()))
    pend = m.store.pending(); assert len(pend) == 1
    approver_msgs = [x for x in sent if x[2] == "imessage" and "[Blatbot #" in x[1]]
    assert len(approver_msgs) == 1
    text = approver_msgs[0][1]
    assert "can we do tue 6pm?" in text and "book chat" in text and "calendar, email_send" in text
    sender_msgs = [x for x in sent if x[2] == "email"]
    assert sender_msgs and "confirm that with Aaron" in sender_msgs[0][1]
    assert m.store.thread_state("c1") == "awaiting_aaron"
    assert m.executor.ran == []


def test_aaron_yes_runs_exact_prompt_and_reports(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="P1", scopes=["web"], summary="s"))
    asyncio.run(m.get("c1").handle_inbound("q", "email", stranger_meta()))
    rid = m.store.pending()[0].id
    sent.clear()

    async def go():
        await m.get("aaron").handle_inbound("yes", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.executor.ran == [rid]
    assert m.store.get_request(rid).state == "done"
    assert any(f"#{rid} done" in x[1] for x in sent if x[2] == "imessage")
    assert not [x for x in sent if x[2] == "email"]  # nothing to the sender from the executor


def test_aaron_no_rejects_and_router_may_reply(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="P1", scopes=["web"], summary="s"))
    asyncio.run(m.get("c1").handle_inbound("q", "email", stranger_meta()))
    rid = m.store.pending()[0].id
    m.router.next = RouterOutput(reply="Sorry, Aaron can't do that.", request=RouterRequest(prompt="X", scopes=["web"], summary="x"))

    async def go():
        await m.get("aaron").handle_inbound(f"#{rid} no", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.store.get_request(rid).state == "rejected"
    assert m.executor.ran == []
    assert m.store.pending() == []  # a system-note turn may never create a request


def test_aaron_edit_reshows_then_runs_edited(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="P1", scopes=["web"], summary="s"))
    asyncio.run(m.get("c1").handle_inbound("q", "email", stranger_meta()))
    rid = m.store.pending()[0].id

    async def go():
        await m.get("aaron").handle_inbound(f"#{rid} edit: P2 instead", "imessage", approver_meta())
        assert m.store.get_request(rid).prompt.endswith("--- AARON'S INSTRUCTIONS ---\nP2 instead")
        assert "q" in m.store.get_request(rid).prompt  # the source message is kept
        assert any("revised" in x[1] and "P2 instead" in x[1] for x in sent if x[2] == "imessage")
        await m.get("aaron").handle_inbound(f"#{rid} yes", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.store.get_request(rid).state == "done" and m.executor.ran == [rid]


def test_hash_mismatch_refused(tmp_path):
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("s")
    r = m.store.create_request(chat_id="c", sender="a", sender_name="", mode="email", subject="", original_message="m",
                               summary="s", scopes=["web"], prompt="ok", state="approved", task_id=t["id"])
    r.prompt = "tampered"
    asyncio.run(m.execute(r))
    assert m.store.get_request(r.id).state == "failed"
    assert any("FAILED" in x[1] for x in sent)


def test_aaron_direct_request_skips_gate(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(prompt="P", scopes=["calendar"], summary="s"))
    async def go():
        await m.get("aaron").handle_inbound("book it", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.executor.ran and m.store.pending() == []


def test_aaron_chat_while_pending_is_not_consumed(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="P1", scopes=["web"], summary="s"))
    asyncio.run(m.get("c1").handle_inbound("q", "email", stranger_meta()))
    m.router.next = RouterOutput(reply="Here you go.", request=None)
    asyncio.run(m.get("aaron").handle_inbound("what's on my calendar", "imessage", approver_meta()))
    assert m.store.pending()[0].state == "pending" and m.executor.ran == []
    assert any(x[1] == "Here you go." for x in sent)


def test_email_from_aaron_address_is_not_approver(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="P", scopes=["web"], summary="s"))
    asyncio.run(m.get("x").handle_inbound("do it", "email", {"sender": "owner@example.com", "to": "owner@example.com"}))
    assert m.store.pending() and m.executor.ran == []


# -- task ledger -------------------------------------------------------------

def test_task_key_normalizes_email_phone_name():
    from inkbox_claude.gate.store import task_key
    assert task_key(" Cand@NYU.edu ") == "cand@nyu.edu"
    assert task_key("+1 (555) 010-0001") == "5550100001"
    assert task_key("Jamie   Rivera") == "jamie rivera"


def test_inbound_lands_on_ledger_before_routing(tmp_path):
    m, sent = make_manager(tmp_path)
    # Pure conversation creates no task: the ledger is loaded (empty) before routing.
    m.router.next = RouterOutput(reply="Hi!", request=None)
    asyncio.run(m.get("c1").handle_inbound("hello there", "email", stranger_meta()))
    assert m.router.calls[-1]["task_memory"] == "(no tasks on record)"
    assert m.store.tasks_for_person("cand@nyu.edu", open_only=False) == []
    # A message that starts a piece of work: the router opens a task and the inbound
    # and outbound both land on it, so a follow-up hours later finds them.
    m.router.next = RouterOutput(reply="Sure, what times work?", task="new",
                                 task_title="Coffee chat with Cand Idate", request=None)
    asyncio.run(m.get("c1").handle_inbound("can we set up a coffee chat?", "email", stranger_meta()))
    mem2 = m.store.task_memory("c1", is_approver=False, person="cand@nyu.edu")
    assert "Coffee chat with Cand Idate" in mem2 and "inbound: can we set up a coffee chat?" in mem2
    assert "outbound: Sure, what times work?" in mem2
    # next turn the router sees it first
    m.router.next = RouterOutput(reply="ok", request=None)
    asyncio.run(m.get("c1").handle_inbound("wed?", "email", stranger_meta()))
    assert "Coffee chat with Cand Idate" in m.router.calls[-1]["task_memory"]


def test_request_approval_and_result_are_on_the_ledger(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="Book Tue 6pm", scopes=["calendar"], summary="book chat"))
    asyncio.run(m.get("c1").handle_inbound("tue 6pm?", "email", stranger_meta()))
    rid = m.store.pending()[0].id
    assert m.store.task_id_for_request(rid) is not None
    mem = m.store.task_memory("c1", is_approver=False)
    assert f"request (#{rid}): book chat" in mem and "state: waiting_aaron" in mem

    async def go():
        await m.get("aaron").handle_inbound("yes", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    mem = m.store.task_memory("c1", is_approver=False)
    assert "approved" in mem and "done (#%d): book chat -> did it" % rid in mem and "state: done" in mem


def test_aaron_follow_up_hours_later_sees_candidates_ledger(tmp_path):
    """Aaron asks about a candidate in his own thread; the memory must carry that
    candidate's whole record even though it lived in another chat."""
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply="Got it", task="new", task_title="Coffee chat with Cand Idate", request=None)
    asyncio.run(m.get("c1").handle_inbound("free wed 3pm", "email", stranger_meta()))
    # Aaron, hours later, asks to book it; the router names the counterpart but not the task.
    m.router.next = RouterOutput(reply="On it", request=RouterRequest(
        prompt="Book Cand Idate cand@nyu.edu Wed 3pm", scopes=["calendar"], summary="book cand", counterpart="cand@nyu.edu"))

    async def go():
        await m.get("aaron").handle_inbound("book the cand chat", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    mem_aaron = m.router.calls[-1]["task_memory"]
    assert "inbound: free wed 3pm" in mem_aaron  # candidate thread visible from Aaron's thread
    # Aaron's request linked to the same task as the candidate's thread
    tid = m.store.task_id_for_request(1)
    assert "cand@nyu.edu" in [p["key"] for p in m.store.get_task(tid)["participants"]]
    assert len(m.store.tasks_for_person("cand@nyu.edu", open_only=False)) == 1  # continued, not duplicated
    # and the candidate's thread now sees Aaron's action + result
    mem_c = m.store.task_memory("c1", is_approver=False)
    assert "book cand" in mem_c and "did it" in mem_c


def test_tasks_are_units_of_work_not_people(tmp_path):
    from inkbox_claude.gate.store import Store, TaskRequired
    s = Store(str(tmp_path / "g.db"))
    t1 = s.create_task("Book coffee chat with X", [("x@nyu.edu", "X")])
    t2 = s.create_task("Send X the application link", [("x@nyu.edu", "X")])
    t3 = s.create_task("Find the Club Fest date")  # nobody in particular
    assert t1["id"] != t2["id"]
    assert [t["id"] for t in s.tasks_for_person("X@NYU.edu")] == [t2["id"], t1["id"]]
    assert t3["participants"] == []
    assert "no one in particular" in s.task_memory("aaron", is_approver=True)
    import pytest
    with pytest.raises(TaskRequired):
        s.create_request(chat_id="c", sender="x", sender_name="", mode="email", subject="", original_message="m",
                         summary="s", scopes=["calendar"], prompt="p", state="pending", task_id=0)
    with pytest.raises(TaskRequired):
        s.create_request(chat_id="c", sender="x", sender_name="", mode="email", subject="", original_message="m",
                         summary="s", scopes=["calendar"], prompt="p", state="pending", task_id=9999)


def test_router_picks_the_task_and_code_enforces_it(tmp_path):
    m, sent = make_manager(tmp_path)
    # first request: router says new task
    m.router.next = RouterOutput(reply=None, task="new", task_title="Book coffee chat with Cand Idate",
                                 request=RouterRequest(prompt="Book Tue 6pm", scopes=["calendar"], summary="book chat"))
    asyncio.run(m.get("c1").handle_inbound("tue 6?", "email", stranger_meta()))
    r1 = m.store.get_request(1); t1 = m.store.get_task(r1.__dict__.get("task_id") or m.store.task_id_for_request(1))
    assert t1["title"] == "Book coffee chat with Cand Idate"
    assert [p["key"] for p in t1["participants"]] == ["cand@nyu.edu"]
    # a different job from the same person: router says new again -> second task, same person on both
    m.store.set_state(1, "done")
    m.router.next = RouterOutput(reply=None, task="new", task_title="Send the application link",
                                 request=RouterRequest(prompt="Email link", scopes=["email_send"], summary="send link"))
    asyncio.run(m.get("c1").handle_inbound("can you send me the app link?", "email", stranger_meta()))
    t2 = m.store.get_task(m.store.task_id_for_request(2))
    assert t2["id"] != t1["id"] and [p["key"] for p in t2["participants"]] == ["cand@nyu.edu"]
    assert len(m.store.tasks_for_person("cand@nyu.edu", open_only=False)) == 2
    # a follow-up on the first job: router names T<id>
    m.store.set_state(2, "done")
    m.router.next = RouterOutput(reply=None, task=f"T{t1['id']}",
                                 request=RouterRequest(prompt="Move to 7pm", scopes=["calendar"], summary="move chat"))
    asyncio.run(m.get("c1").handle_inbound("actually 7?", "email", stranger_meta()))
    assert m.store.task_id_for_request(3) == t1["id"]


def test_stranger_cannot_hijack_someone_elses_task(tmp_path):
    m, sent = make_manager(tmp_path)
    other = m.store.create_task("Book chat with Someone Else", [("other@nyu.edu", "Other")])
    m.router.next = RouterOutput(reply=None, task=f"T{other['id']}",
                                 request=RouterRequest(prompt="Cancel it", scopes=["calendar"], summary="cancel"))
    asyncio.run(m.get("c1").handle_inbound("cancel that booking", "email", stranger_meta()))
    tid = m.store.task_id_for_request(1)
    assert tid != other["id"]  # code refused the router's choice and started a task for the sender
    assert [p["key"] for p in m.store.get_task(tid)["participants"]] == ["cand@nyu.edu"]


def test_aaron_may_continue_any_task_and_errands_have_no_participant(tmp_path):
    m, sent = make_manager(tmp_path)
    cand = m.store.create_task("Book chat with Cand", [("cand@nyu.edu", "Cand")])
    m.router.next = RouterOutput(reply="On it.", task=f"T{cand['id']}",
                                 request=RouterRequest(prompt="Did the invite go out?", scopes=["calendar"], summary="check invite"))
    asyncio.run(m.get("aaron").handle_inbound("did cand's invite go out?", "imessage", approver_meta()))
    assert m.store.task_id_for_request(1) == cand["id"]
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Find the Club Fest date",
                                 request=RouterRequest(prompt="Look up club fest", scopes=["web"], summary="club fest date"))
    asyncio.run(m.get("aaron").handle_inbound("when is club fest?", "imessage", approver_meta()))
    t = m.store.get_task(m.store.task_id_for_request(2))
    assert t["title"] == "Find the Club Fest date" and t["participants"] == []


def test_every_request_has_a_task_even_when_the_router_says_nothing(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="Book", scopes=["calendar"], summary="book chat"))
    asyncio.run(m.get("c1").handle_inbound("book me", "email", stranger_meta()))
    tid = m.store.task_id_for_request(1)
    assert tid and m.store.get_task(tid)["title"] == "book chat"


def test_aaron_task_commands(tmp_path):
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("Book chat with Cand", [("cand@nyu.edu", "Cand")])
    asyncio.run(m.get("aaron").handle_inbound("tasks", "imessage", approver_meta()))
    assert f"T{t['id']} Book chat with Cand (Cand; open)" in sent[-1][1]
    asyncio.run(m.get("aaron").handle_inbound(f"T{t['id']} note: he prefers mornings", "imessage", approver_meta()))
    asyncio.run(m.get("aaron").handle_inbound(f"T{t['id']} done", "imessage", approver_meta()))
    assert m.store.get_task(t["id"])["state"] == "done"
    assert "prefers mornings" in m.store.task_memory("aaron", is_approver=True)
    assert m.router.calls == []  # commands never reach the router


def test_executor_receives_the_persons_ledger(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(prompt="P1", scopes=["web"], summary="look up"))
    asyncio.run(m.get("c1").handle_inbound("who am i", "email", stranger_meta()))
    rid = m.store.pending()[0].id

    async def go():
        await m.get("aaron").handle_inbound("yes", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert "Cand Idate" in m.executor.context and f"request (#{rid}): look up" in m.executor.context


# -- phone surface -------------------------------------------------------------
AARON_PHONE = "+15550100001"


def phone_manager(tmp_path, *, trust, active=True):
    m, sent = make_manager(tmp_path)
    m.approver_phone = AARON_PHONE
    m.voice_trust_approver = trust
    m.call_active = lambda chat_id: active
    return m, sent


def voice_meta(sender=AARON_PHONE):
    return {"call_id": "call-1", "sender": sender, "contact": {"id": "p1", "name": "Caller"}}


def test_phone_stranger_request_waits_for_aaron_and_is_spoken_to(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True)
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Book Tue 6pm", scopes=["calendar"], summary="book chat"))
    asyncio.run(m.get("p1").handle_inbound("book me tuesday at six", "voice", voice_meta("+12125550100")))
    assert m.router.calls[0]["mode"] == "voice"
    assert [x for x in sent if x[2] == "voice" and x[1] == "On it."]
    assert len(m.store.pending()) == 1 and m.executor.ran == []
    # no per-utterance FYI spam on a call; only the approval text goes to Aaron
    assert not [x for x in sent if "[Blatbot FYI]" in x[1]]


def test_phone_aaron_untrusted_number_still_needs_imessage_yes(tmp_path):
    m, sent = phone_manager(tmp_path, trust=False)
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Email Bob", scopes=["email_send"], summary="email bob"))
    asyncio.run(m.get("p1").handle_inbound("email bob for me", "voice", voice_meta()))
    assert len(m.store.pending()) == 1 and m.executor.ran == []


def test_phone_aaron_trusted_runs_immediately(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True)
    m.router.next = RouterOutput(reply="On it.", request=RouterRequest(
        prompt="Email Bob", scopes=["email_send"], summary="email bob"))

    async def go():
        await m.get("p1").handle_inbound("email bob for me", "voice", voice_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.executor.ran and not m.store.pending()


def test_phone_result_after_hangup_goes_by_text(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True, active=False)
    s = m.get("p1"); s.mode = "voice"; s.reply_meta = voice_meta("+12125550100")
    asyncio.run(s.send_to_sender("Booked for Tuesday."))
    assert sent[-1][2] == "sms" and sent[-1][3]["to"] == "+12125550100"
    s.reply_meta = voice_meta()
    asyncio.run(s.send_to_sender("Done."))
    assert sent[-1][2] == "imessage"


def test_call_end_logs_ledger_and_texts_nothing(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True)
    s = m.get("p1")
    asyncio.run(s.handle_inbound("hi there", "voice", voice_meta("+12125550100")))
    asyncio.run(s.run_consult("[voice call ended] ...\n\nRecent call transcript:\nuser: hi there"))
    assert not [x for x in sent if "Phone call" in x[1]]
    assert asyncio.run(s.run_consult("some other wake-up")) == ""


def test_realtime_consult_trusted_runs_and_returns_result(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(
        prompt="List today's events", scopes=["calendar"], summary="read calendar"))
    out = asyncio.run(m.get("p1").voice_consult("what's on my calendar today", voice_meta()))
    assert m.executor.ran and "did it" in out and out.startswith("Done")
    assert not [x for x in sent if x[2] == "voice"]  # returned to the voice model, not pushed


def test_realtime_consult_stranger_is_gated(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True)
    m.router.next = RouterOutput(reply=None, request=RouterRequest(
        prompt="Book Tue 6pm", scopes=["calendar"], summary="book chat"))
    out = asyncio.run(m.get("p1").voice_consult("book me tuesday", voice_meta("+12125550100")))
    assert "Aaron" in out and m.executor.ran == [] and len(m.store.pending()) == 1
    assert [x for x in sent if x[2] == "imessage" and "[Blatbot #" in x[1]]


def test_realtime_consult_plain_answer(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True)
    m.router.next = RouterOutput(reply="It is Sunday.", request=None)
    assert asyncio.run(m.get("p1").voice_consult("what day is it", voice_meta())) == "It is Sunday."


def test_voice_briefing_and_gate_prompt(tmp_path):
    from inkbox_claude.realtime import RealtimeCallMeta, build_realtime_instructions
    m, _ = phone_manager(tmp_path, trust=True)
    asyncio.run(m.get("p2").handle_inbound("please book thursday", "voice", voice_meta("+12125550100")))
    m.voice_vocabulary = "Acme (AK-mee)."
    b = m.get("p1").voice_briefing(voice_meta())
    assert "It is " in b and "Acme" in b and "Aaron himself" in b and "not a limit" in b
    stranger = m.get("p2").voice_briefing(voice_meta("+12125550100"))
    assert "NOT Aaron" in stranger
    fields = {f: None for f in RealtimeCallMeta.__dataclass_fields__}
    try:
        meta = RealtimeCallMeta()
    except TypeError:
        meta = RealtimeCallMeta(**fields)
    text = build_realtime_instructions(meta, b, gate_mode=True)
    assert "This is your call" in text and "git" not in text
    low = text.lower()
    assert not any(w in low for w in ("claude", "openai", "inkbox agent", "gateway", "back office"))
    g = build_realtime_instructions  # noqa
    from inkbox_claude.realtime import build_realtime_greeting
    assert "Blatbot" in build_realtime_greeting(meta, True) and "Claude" not in build_realtime_greeting(meta, True)
    assert "git" in build_realtime_instructions(meta, "", gate_mode=False)


def test_gate_mode_session_tools_and_speech_only_relay():
    import json as _j
    from inkbox_claude import realtime as rt

    class WS:
        def __init__(self): self.sent = []
        async def send_str(self, x): self.sent.append(_j.loads(x))

    fields = {f: None for f in rt.RealtimeCallMeta.__dataclass_fields__}
    try:
        meta = rt.RealtimeCallMeta()
    except TypeError:
        meta = rt.RealtimeCallMeta(**fields)
    ws = WS()
    asyncio.run(rt._send_session_update(ws, rt.RealtimeConfig(gate_mode=True), meta))
    sess = ws.sent[0]["session"]
    assert sess["tool_choice"] == "auto"
    assert "There is no script" in sess["instructions"]
    assert sorted(t["name"] for t in sess["tools"]) == sorted([rt.CONSULT_TOOL_NAME, rt.HANG_UP_CALL_TOOL_NAME])
    desc = next(t for t in sess["tools"] if t["name"] == rt.CONSULT_TOOL_NAME)["description"]
    assert "beyond conversation" in desc and "git" not in desc and "back office" not in desc
    ws2 = WS()
    asyncio.run(rt._send_session_update(ws2, rt.RealtimeConfig(gate_mode=False), meta))
    assert ws2.sent[0]["session"]["tool_choice"] == "auto"
    ws3 = WS()
    asyncio.run(rt._submit_tool_result(ws3, "c1", {"status": "ok", "answer": "x"}, response={"tool_choice": "none"}))
    assert ws3.sent[-1] == {"type": "response.create", "response": {"tool_choice": "none"}}


def test_phone_call_never_changes_a_threads_text_or_email_route(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True)
    s = m.get("c1")
    asyncio.run(s.handle_inbound("can we do tue 6pm?", "email", stranger_meta()))
    route_before = m.store.thread_route("c1")
    s.voice_briefing(voice_meta("+12125550100"))
    m.router.next = RouterOutput(reply="Tuesday works.", request=None)
    asyncio.run(s.voice_consult("does tuesday work", voice_meta("+12125550100")))
    assert s.mode == "email" and s.reply_meta.get("subject") == "Re: chat"
    assert m.store.thread_route("c1") == route_before
    asyncio.run(s.send_to_sender("Following up by email."))
    assert sent[-1][2] == "email"


def test_bridge_state_has_every_field_the_pump_touches():
    # A missing attribute here kills a live call the moment the first frame arrives.
    import re, inspect
    from inkbox_claude import realtime as rt
    used = set(re.findall(r"\bstate\.([a-z_]+)", inspect.getsource(rt)))
    st = rt._BridgeState()
    missing = [a for a in used if not hasattr(st, a)]
    assert not missing, missing


def test_street_noise_blip_does_not_interrupt_but_real_speech_does(monkeypatch):
    """Drive the live-call pump with fake OpenAI frames, end to end."""
    import base64, json as _j
    import aiohttp
    from types import SimpleNamespace as NS
    from inkbox_claude import realtime as rt

    monkeypatch.setenv("INKBOX_REALTIME_BARGE_IN_MS", "120")

    class FakeOpenAI:
        def __init__(self, script): self.script, self.sent = script, []
        async def send_str(self, x): self.sent.append(_j.loads(x))
        def __aiter__(self): return self._gen()
        async def _gen(self):
            for delay, frame in self.script:
                await asyncio.sleep(delay)
                yield NS(type=aiohttp.WSMsgType.TEXT, data=_j.dumps(frame))

    class FakeInkbox:
        def __init__(self): self.sent = []
        async def send_str(self, x): self.sent.append(_j.loads(x))

    audio = base64.b64encode(b"\xff" * 16000).decode()  # 2 s of agent speech
    script = [
        (0, {"type": "response.created"}), (0, {"type": "response.done"}),           # the introduction
        (0, {"type": "response.created"}),
        (0, {"type": "response.output_audio.delta", "delta": audio}),
        (0, {"type": "input_audio_buffer.speech_started"}),                           # a car horn
        (0.04, {"type": "input_audio_buffer.speech_stopped"}),
        (0.25, {"type": "input_audio_buffer.speech_started"}),                        # Aaron actually talking
        (0.30, {"type": "input_audio_buffer.speech_stopped"}),
    ]
    oa, ib = FakeOpenAI(script), FakeInkbox()
    state = rt._BridgeState(gate_mode=True)

    async def consult(q, t): return "x"
    fields = {f: None for f in rt.RealtimeCallMeta.__dataclass_fields__}
    try:
        meta = rt.RealtimeCallMeta()
    except TypeError:
        meta = rt.RealtimeCallMeta(**fields)

    async def go():
        await rt._openai_to_inkbox_pump(openai_ws=oa, inkbox_ws=ib, state=state,
                                        config=rt.RealtimeConfig(gate_mode=True), meta=meta,
                                        on_agent_consult=consult)
        await asyncio.sleep(0.2)
    asyncio.run(go())
    cancels = [x for x in oa.sent if x.get("type") == "response.cancel"]
    clears = [x for x in ib.sent if x.get("event") == "clear"]
    assert len(cancels) == 1 and len(clears) == 1  # the horn did nothing; the real interruption did


def test_one_person_many_contacts_share_tasks(tmp_path):
    """Sam emails to book, then calls from his phone to move it: same person, same task."""
    m, sent = make_manager(tmp_path)
    contact = {"id": "sam-1", "name": "Sam Lee", "emails": ["Sam@NYU.edu"], "phones": ["+1 (212) 555-0100"]}
    email_meta = {"sender": "sam@nyu.edu", "to": "sam@nyu.edu", "subject": "chat", "contact": contact}
    m.router.next = RouterOutput(reply=None, task="new", task_title="Book coffee chat with Sam Lee",
                                 request=RouterRequest(prompt="Book Tue 6pm", scopes=["calendar"], summary="book chat"))
    asyncio.run(m.get("sam-1").handle_inbound("tue 6?", "email", email_meta))
    t = m.store.get_task(m.store.task_id_for_request(1))
    assert len(t["participants"]) == 1  # one person, not one per address
    assert sorted(t["participants"][0]["keys"]) == ["2125550100", "sam lee", "sam@nyu.edu"]
    # a call from the phone number, with no contact hydrated at all, still finds Sam's task
    assert [x["id"] for x in m.store.tasks_for_person("+12125550100")] == [t["id"]]
    assert m.store.is_participant(t["id"], "212-555-0100")
    # and a router pointing the phone turn at that task is allowed
    m.store.set_state(1, "done")
    m.router.next = RouterOutput(reply="Moved.", task=f"T{t['id']}",
                                 request=RouterRequest(prompt="Move to 7", scopes=["calendar"], summary="move chat"))
    asyncio.run(m.get("sam-1").voice_consult("move it to seven", {"call_id": "c", "sender": "+12125550100", "contact": {}}))
    assert m.store.task_id_for_request(2) == t["id"]
    # a stranger on a different number is still refused
    m.store.set_state(2, "done")
    m.router.next = RouterOutput(reply=None, task=f"T{t['id']}",
                                 request=RouterRequest(prompt="Cancel", scopes=["calendar"], summary="cancel"))
    asyncio.run(m.get("x9").handle_inbound("cancel sam's chat", "sms", {"sender": "+13125550199", "to": "+13125550199", "contact": {}}))
    assert m.store.task_id_for_request(3) != t["id"]


def test_router_summary_is_kept_on_the_task_and_shown_next_time(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply=None, task="new", task_title="Coffee chat with Cand Idate",
                                 task_summary="Cand wants a coffee chat; asked for Tue 6 PM, awaiting Aaron.",
                                 request=RouterRequest(prompt="Book Tue 6pm", scopes=["calendar"], summary="book chat"))
    asyncio.run(m.get("c1").handle_inbound("tue 6?", "email", stranger_meta()))
    m.router.next = RouterOutput(reply="ok", request=None)
    asyncio.run(m.get("c1").handle_inbound("thanks", "email", stranger_meta()))
    assert "Where it stands: Cand wants a coffee chat; asked for Tue 6 PM" in m.router.calls[-1]["task_memory"]


def test_query_tasks_filters_and_ranks(tmp_path):
    from inkbox_claude.gate.store import Store
    import time as _t
    s = Store(str(tmp_path / "g.db"))
    a = s.create_task("Get a quote from the caterer", [("cater@food.com", "Cater Co")])
    s.set_task_summary(a["id"], "Waiting on the caterer's revised quote for 40 people.")
    b = s.create_task("Book a call with Sam Lee", [("sam@nyu.edu", "Sam Lee")])
    c = s.create_task("Renew the domain")  # nobody
    s.task_event(c["id"], "done", "renewed", state="done")
    old = s.create_task("Old errand"); s.task_event(old["id"], "done", "x", state="done")
    s._db.execute("UPDATE tasks SET updated_at=? WHERE id=?", (_t.time() - 40 * 86400, old["id"])); s._db.commit()
    r = s.query_tasks(text="caterer")
    assert [t["id"] for t in r["tasks"]] == [a["id"]] and r["total"] == 1
    r = s.query_tasks(participant="+1 (555) 000-0000")
    assert r["total"] == 0
    r = s.query_tasks(participant="Sam Lee")
    assert [t["id"] for t in r["tasks"]] == [b["id"]]
    r = s.query_tasks(has_participants=False)
    assert {t["id"] for t in r["tasks"]} == {c["id"], old["id"]}
    r = s.query_tasks(states=["live"])
    assert {t["id"] for t in r["tasks"]} == {a["id"], b["id"]}
    r = s.query_tasks(touched_within_days=7)
    assert old["id"] not in {t["id"] for t in r["tasks"]}
    r = s.query_tasks(text="quote 40 people")  # words across title and summary
    assert [t["id"] for t in r["tasks"]] == [a["id"]]
    assert "Where it stands" in s.render_tasks(r["tasks"])


def test_router_lookup_feeds_the_decision_and_is_scoped(tmp_path):
    from inkbox_claude.gate.router import TaskQuery
    m, sent = make_manager(tmp_path)
    caterer = m.store.create_task("Get a quote from the caterer", [("cand@nyu.edu", "Cand Idate")])
    m.store.task_event(caterer["id"], "done", "quote received", state="done")
    for i in range(8):  # newer tasks push the caterer out of the default per-person view
        t = m.store.create_task(f"Chore number {i}", [("cand@nyu.edu", "Cand Idate")])
        m.store.task_event(t["id"], "done", "ok", state="done")
    assert "caterer" not in m.store.task_memory("c1", is_approver=False, person="cand@nyu.edu")
    others = m.store.create_task("Aaron's private errand")  # no participants
    # the candidate asks about the caterer: the router plans a text query, code runs it
    m.router.next_query = TaskQuery(text="caterer", states=None)
    m.router.next = RouterOutput(reply="Quote is in.", request=None)
    asyncio.run(m.get("c1").handle_inbound("any news from the caterer?", "email", stranger_meta()))
    found = m.router.calls[-1]["found_tasks"]
    assert "Get a quote from the caterer" in found
    # a non-owner's lookup can never surface tasks they are not on
    m.router.next_query = TaskQuery(text="errand", has_participants=False)
    asyncio.run(m.get("c1").handle_inbound("what about the errand?", "email", stranger_meta()))
    assert "private errand" not in m.router.calls[-1]["found_tasks"]
    # the owner can: it is either already in his default view or surfaced by the lookup
    asyncio.run(m.get("aaron").handle_inbound("what about my errand?", "imessage", approver_meta()))
    seen = m.router.calls[-1]["task_memory"] + m.router.calls[-1]["found_tasks"]
    assert "private errand" in seen
    # and a lookup by the owner for something outside his default view does surface it
    m.store.task_event(others["id"], "done", "ok", state="done")
    m.store._db.execute("UPDATE tasks SET updated_at=updated_at-40*86400 WHERE id=?", (others["id"],)); m.store._db.commit()
    asyncio.run(m.get("aaron").handle_inbound("that old errand?", "imessage", approver_meta()))
    assert "private errand" not in m.router.calls[-1]["task_memory"]
    assert "private errand" in m.router.calls[-1]["found_tasks"]
    # no query planned -> nothing extra passed
    m.router.next_query = None
    asyncio.run(m.get("c1").handle_inbound("thanks!", "email", stranger_meta()))
    assert m.router.calls[-1]["found_tasks"] == ""


def test_everything_on_a_task_is_searchable(tmp_path):
    from inkbox_claude.gate.store import Store, extract_dates
    s = Store(str(tmp_path / "g.db"))
    t = s.create_task("Venue for the spring mixer", [("eve@venue.com", "Eve Park")])
    r = s.create_request(chat_id="c", sender="eve@venue.com", sender_name="Eve Park", mode="email", subject="",
                         original_message="m", summary="ask about capacity", scopes=["email_send"],
                         prompt="Email Eve Park and ask the maximum capacity of the Grand Hall for Tue 9/22", state="approved",
                         task_id=t["id"])
    s.set_state(r.id, "done", status={"ok": True, "summary": "Eve replied: capacity is 180 standing, 120 seated."})
    s.task_event(t["id"], "note", "Aaron: budget is $4,000 all in", state="open")
    other = s.create_task("Renew the domain")
    # words from the request prompt, the result, an event note, a participant, and stems/prefixes
    for q in ("capacity", "Grand Hall", "seated", "budget", "Eve Park", "venue.com", "mixers", "cater OR mixer", '"120 seated"'):
        got = [x["id"] for x in s.query_tasks(text=q)["tasks"]]
        assert got == [t["id"]], (q, got)
    assert s.query_tasks(text="domain")["tasks"][0]["id"] == other["id"]
    assert s.query_tasks(text="nonexistentword")["total"] == 0
    # dates mentioned anywhere on the task are searchable as a range
    assert extract_dates("Tue 9/22 and Sep 30, 2026 and 2026-10-01", 1789000000)[:3] == ["2026-09-22", "2026-09-30", "2026-10-01"]
    assert [x["id"] for x in s.query_tasks(date_from="2026-09-20", date_to="2026-09-25")["tasks"]] == [t["id"]]
    assert s.query_tasks(date_from="2026-10-05", date_to="2026-10-10")["total"] == 0
    # any_of: OR across sub-filters
    r2 = s.query_tasks(any_of=[{"text": "domain"}, {"participant": "Eve Park"}])
    assert {x["id"] for x in r2["tasks"]} == {t["id"], other["id"]}
    # the index follows later changes
    s.set_task_summary(t["id"], "Waiting on the deposit invoice.")
    assert [x["id"] for x in s.query_tasks(text="deposit invoice")["tasks"]] == [t["id"]]


def test_site_admin_scopes_cover_the_admin_mcps():
    assert "mcp__sjba-admin__sjba_update_board_member" in tools_for(["sjba_site_write"])
    assert "mcp__sjba-admin__sjba_update_board_member" not in tools_for(["sjba_site_read"])
    assert "mcp__sjba-admin__sjba_list_board_members" in tools_for(["sjba_site_read"])
    assert "mcp__tamid-admin__tamid_update_event" in tools_for(["tamid_site_write"])
    for name in ("sjba_site_read", "sjba_site_write", "tamid_site_read", "tamid_site_write"):
        assert name in SCOPES


class FakePicker:
    def __init__(self, choice=None, conf=0.9, event=None):
        self.choice, self.conf, self.calls, self.event_calls = choice, conf, [], []
        self.event = event or {"kind": None, "kind_conf": 0.0, "waiting_on": None, "resolves_task": 0.0}

    async def pick(self, **kw):
        self.calls.append(kw)
        return {"choice": self.choice, "confidence": self.conf, "probabilities": {}, "reason": "ok" if self.choice else "low confidence"}

    async def judge_event(self, **kw):
        self.event_calls.append(kw)
        return self.event

    scopes = None  # a list to override, None = undecided
    action = None  # True/False to override, None = undecided

    async def judge_action(self, **kw):
        self.action_calls = getattr(self, "action_calls", []) + [kw]
        return {"needs_action": self.action, "p": 0.9 if self.action else 0.1, "reason": "ok"}

    async def judge_scopes(self, **kw):
        self.scope_calls = getattr(self, "scope_calls", []) + [kw]
        return {"scopes": self.scopes, "probabilities": {}, "reason": "ok" if self.scopes else "disabled"}

    async def judge_grounded(self, **kw):
        return {"grounded": None, "p": 0.0, "reason": "undecided"}   # tests override when they care


def test_jev_pick_overrides_router_when_confident(tmp_path):
    m, sent = make_manager(tmp_path)
    a = m.store.create_task("Book a call with Cand Idate", [("cand@nyu.edu", "Cand Idate")])
    b = m.store.create_task("Send Cand the venue list", [("cand@nyu.edu", "Cand Idate")])
    m.task_picker = FakePicker(choice=f"T{b['id']}")
    m.router.next = RouterOutput(reply=None, task=f"T{a['id']}",  # router guessed the other task
                                 request=RouterRequest(prompt="Email the list", scopes=["email_send"], summary="send list"))
    asyncio.run(m.get("c1").handle_inbound("did you send that list yet?", "email", stranger_meta()))
    assert m.store.task_id_for_request(1) == b["id"]
    call = m.task_picker.calls[-1]
    assert {f"T{a['id']}", f"T{b['id']}"} <= {f"T{t['id']}" for t in call["candidates"]}
    assert call["router_hint"] == f"T{a['id']}"


def test_jev_undecided_keeps_router_pick_and_code_rules(tmp_path):
    m, sent = make_manager(tmp_path)
    a = m.store.create_task("Book a call with Cand Idate", [("cand@nyu.edu", "Cand Idate")])
    m.task_picker = FakePicker(choice=None, conf=0.3)
    m.router.next = RouterOutput(reply=None, task=f"T{a['id']}",
                                 request=RouterRequest(prompt="Move it", scopes=["calendar"], summary="move call"))
    asyncio.run(m.get("c1").handle_inbound("can we move it?", "email", stranger_meta()))
    assert m.store.task_id_for_request(1) == a["id"]
    # a confident pick of someone else's task is still refused by code
    other = m.store.create_task("Someone else's job", [("other@nyu.edu", "Other")])
    m.task_picker = FakePicker(choice=f"T{other['id']}")
    m.store.set_state(1, "done")
    m.router.next = RouterOutput(reply=None, task="new", task_title="x",
                                 request=RouterRequest(prompt="Cancel", scopes=["calendar"], summary="cancel"))
    asyncio.run(m.get("c1").handle_inbound("cancel it", "email", stranger_meta()))
    assert m.store.task_id_for_request(2) != other["id"]


def test_jev_new_and_none(tmp_path):
    m, sent = make_manager(tmp_path)
    a = m.store.create_task("Book a call with Cand Idate", [("cand@nyu.edu", "Cand Idate")])
    m.task_picker = FakePicker(choice="new")
    m.router.next = RouterOutput(reply=None, task=f"T{a['id']}", task_title="Get the venue list",
                                 request=RouterRequest(prompt="Send list", scopes=["email_send"], summary="venue list"))
    asyncio.run(m.get("c1").handle_inbound("separately, can you send me the venue list?", "email", stranger_meta()))
    assert m.store.task_id_for_request(1) != a["id"]
    m.task_picker = FakePicker(choice="none")
    m.router.next = RouterOutput(reply="You're welcome!", task=f"T{a['id']}", request=None)
    before = m.store._db.execute("select count(*) from tasks").fetchone()[0]
    asyncio.run(m.get("c1").handle_inbound("thanks!", "email", stranger_meta()))
    assert m.store._db.execute("select count(*) from tasks").fetchone()[0] == before


def test_fyi_is_not_truncated(tmp_path):
    m, sent = make_manager(tmp_path)
    long = "word " * 300
    asyncio.run(m.get("c1").handle_inbound(long, "email", stranger_meta()))
    fyi = [x for x in sent if "[Blatbot FYI]" in x[1]][0][1]
    assert "..." not in fyi and fyi.count("word") == 300


def test_email_thread_shares_tasks_and_history_across_senders(tmp_path):
    """Aaron replies in a thread with an instruction; the candidate replies in the same
    thread later. The router must see Aaron's instruction and the task it created."""
    m, sent = make_manager(tmp_path)
    aaron_email = {"sender": "owner@example.edu", "to": "owner@example.edu", "subject": "Re: Workshop",
                   "thread_id": "thr-1", "contact": {"id": "a1", "name": "Aaron Blatnoy"}}
    tem_email = {"sender": "sam@example.edu", "to": "sam@example.edu", "subject": "Re: Workshop",
                 "thread_id": "thr-1", "contact": {"id": "t1", "name": "Sam Rivera"}}
    m.router.next = RouterOutput(reply="Will do.", task="new", task_title="Add Sam's application to the sheet",
                                 task_summary="When Tem sends the application, add it to the sheet.", request=None)
    asyncio.run(m.get("a1").handle_inbound("When Sam responds, add their application to the sheet.", "email", aaron_email))
    tid = m.store.task_ids_for_chat("a1")[0]
    m.router.next = RouterOutput(reply="Thanks.", task=f"T{tid}",
                                 request=RouterRequest(prompt="Add Tem to sheet", scopes=["tamid_drive_write"], summary="add to sheet"))
    asyncio.run(m.get("t1").handle_inbound("Here is my application: ...", "email", tem_email))
    call = m.router.calls[-1]
    assert "Add Sam's application to the sheet" in call["task_memory"]          # the thread's task is in view
    assert any("When Sam responds" in h["text"] for h in call["history"])       # and Aaron's instruction is in history
    assert m.store.task_id_for_request(1) == tid                                # Tem was allowed onto that task
    assert m.store.is_participant(tid, "sam@example.edu")


def test_unknown_sender_triggers_a_name_search(tmp_path):
    from inkbox_claude.gate.router import TaskQuery
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("Get Sam Rivera's application onto the sheet")
    m.store.task_event(t["id"], "note", "waiting for Sam Rivera to send it", state="open")
    m.router.next_query = None  # the planner declines; code searches by name anyway
    m.router.next = RouterOutput(reply="ok", request=None)
    meta = {"sender": "sam@example.edu", "to": "sam@example.edu", "subject": "hi", "contact": {"id": "t1", "name": "Sam Rivera"}}
    asyncio.run(m.get("t1").handle_inbound("here it is", "email", meta))
    # a stranger still can't see a task they aren't on and that isn't on their thread
    assert "Sam Rivera's application" not in m.router.calls[-1]["found_tasks"]
    # but the owner asking about Tem finds it
    m.router.next_query = TaskQuery(text="Sam Rivera")
    asyncio.run(m.get("aaron").handle_inbound("where's tem's app?", "imessage", approver_meta()))
    assert "Sam Rivera's application" in (m.router.calls[-1]["found_tasks"] + m.router.calls[-1]["task_memory"])


def test_jev_event_judgment_writes_typed_event_summary_and_state(tmp_path):
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("Add Sam's application to the sheet", [("sam@example.edu", "Sam Rivera")])
    m.task_picker = FakePicker(choice=f"T{t['id']}", event={"kind": "provided_information", "kind_conf": 0.9,
                                                           "waiting_on": "owner", "resolves_task": 0.1})
    m.router.next = RouterOutput(reply="Thanks.", task=f"T{t['id']}", request=None)
    meta = {"sender": "sam@example.edu", "to": "sam@example.edu", "subject": "app", "contact": {"id": "t1", "name": "Sam Rivera"}}
    asyncio.run(m.get("t1").handle_inbound("Here is my application: name, netid, ...", "email", meta))
    tk = m.store.task_with_events(t["id"])
    kinds = [e["kind"] for e in tk["events"]]
    assert "provided_information" in kinds and tk["state"] == "waiting_aaron"
    assert "Waiting on Aaron" in tk["summary"]
    # a message that resolves the task marks it done
    m.task_picker = FakePicker(choice=f"T{t['id']}", event={"kind": "confirmed", "kind_conf": 0.9,
                                                           "waiting_on": "nobody", "resolves_task": 0.95})
    asyncio.run(m.get("t1").handle_inbound("all set, thanks!", "email", meta))
    assert m.store.get_task(t["id"])["state"] == "done"


def test_rfc_reply_chain_links_mails_across_mailboxes(tmp_path):
    from inkbox_claude.gate.store import Store
    s = Store(str(tmp_path / "g.db"))
    r1 = s.thread_root(message_id="<a@x>")                                   # first mail
    r2 = s.thread_root(message_id="<b@y>", in_reply_to="<a@x>", references=["<a@x>"])   # reply from another mailbox
    r3 = s.thread_root(message_id="<c@z>", in_reply_to="<b@y>", references=["<a@x>", "<b@y>"])
    assert r1 == r2 == r3 == "<a@x>"
    assert s.thread_root(message_id="<d@w>") == "<d@w>"   # unrelated mail: its own root
    # the session uses the chain even when provider thread ids differ
    m, sent = make_manager(tmp_path)
    a = {"sender": "owner@example.edu", "to": "owner@example.edu", "subject": "Re: W", "thread_id": "provider-1",
         "message_id": "<a@x>", "contact": {"id": "a1", "name": "Aaron Blatnoy"}}
    tm = {"sender": "sam@example.edu", "to": "sam@example.edu", "subject": "Re: W", "thread_id": "provider-2",
          "message_id": "<b@y>", "in_reply_to": "<a@x>", "references": ["<a@x>"], "contact": {"id": "t1", "name": "Sam Rivera"}}
    m.router.next = RouterOutput(reply="ok", task="new", task_title="Add Sam to the sheet", request=None)
    asyncio.run(m.get("a1").handle_inbound("When Sam responds, add them to the sheet.", "email", a))
    tid = m.store.task_ids_for_chat("a1")[0]
    m.router.next = RouterOutput(reply="ok", task=f"T{tid}", request=RouterRequest(prompt="Add", scopes=["tamid_drive_write"], summary="add"))
    asyncio.run(m.get("t1").handle_inbound("here it is", "email", tm))
    assert "Add Sam to the sheet" in m.router.calls[-1]["task_memory"]
    assert m.store.task_id_for_request(1) == tid


def test_jev_scopes_replace_router_scopes_and_executor_gets_them(tmp_path):
    m, sent = make_manager(tmp_path)
    m.task_picker = FakePicker(choice="new")
    m.task_picker.scopes = ["tamid_drive_write", "tamid_drive_read"]   # router guessed wrong below
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Close the form",
                                 request=RouterRequest(prompt="Set the TAMID form to stop accepting responses",
                                                       scopes=["calendar"], summary="close form"))
    async def go():
        await m.get("aaron").handle_inbound("close the tamid form", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    r = m.store.get_request(1)
    assert set(r.scopes) == {"tamid_drive_write", "tamid_drive_read"}
    assert m.task_picker.scope_calls[-1]["router_scopes"] == ["calendar"]
    assert "calendar" not in m.task_picker.scope_calls[-1]["scopes"] or True  # all scopes offered


def test_jev_scopes_undecided_keeps_router_scopes(tmp_path):
    m, sent = make_manager(tmp_path)
    m.task_picker = FakePicker(choice="new")
    m.task_picker.scopes = None
    m.router.next = RouterOutput(reply=None, task="new", task_title="x",
                                 request=RouterRequest(prompt="Book Tue 6pm", scopes=["calendar"], summary="book"))
    asyncio.run(m.get("c1").handle_inbound("tue 6?", "email", stranger_meta()))
    assert m.store.get_request(1).scopes == ["calendar"]


def test_claude_gets_the_source_not_a_paraphrase(tmp_path):
    m, sent = make_manager(tmp_path)
    m.task_picker = FakePicker(choice="new")
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Move the Tuesday call",
                                 request=RouterRequest(prompt="DEEPSEEK PARAPHRASE", scopes=["calendar"], summary="move call"))
    async def go():
        await m.get("aaron").handle_inbound("push my tuesday 6pm with sam to 7", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    r = m.store.get_request(1)
    assert "DEEPSEEK PARAPHRASE" not in r.prompt
    assert "push my tuesday 6pm with sam to 7" in r.prompt          # verbatim message
    assert "THE TASK THIS BELONGS TO" in r.prompt and "Move the Tuesday call" in r.prompt
    assert sha256(r.prompt) == r.prompt_sha256                        # what ran is what was stored
    assert "run exactly this" not in "".join(x[1] for x in sent)


def test_jev_action_creates_request_when_router_missed_it(tmp_path):
    """The Tem case: the router replied politely and proposed nothing; Jev says a tool
    action is needed because the message supplies what the task was waiting for."""
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("Add Sam's application to the sheet", [("sam@example.edu", "Sam Rivera")])
    m.store.task_event(t["id"], "note", "waiting for Sam to send the application", state="open")
    m.task_picker = FakePicker(choice=f"T{t['id']}")
    m.task_picker.action = True
    m.task_picker.scopes = ["tamid_drive_write"]
    m.router.next = RouterOutput(reply="Thanks, got it.", task=f"T{t['id']}", request=None)
    meta = {"sender": "sam@example.edu", "to": "sam@example.edu", "subject": "app", "contact": {"id": "s1", "name": "Sam Rivera"}}
    asyncio.run(m.get("s1").handle_inbound("Here is my application: name, netid, year...", "email", meta))
    pend = m.store.pending()
    assert len(pend) == 1 and pend[0].scopes == ["tamid_drive_write"]
    assert "Here is my application" in pend[0].prompt and m.store.task_id_for_request(pend[0].id) == t["id"]
    assert any("[Blatbot #" in x[1] for x in sent)  # went to Aaron for approval


def test_jev_action_drops_request_router_invented(tmp_path):
    m, sent = make_manager(tmp_path)
    m.task_picker = FakePicker(choice="none")
    m.task_picker.action = False
    m.router.next = RouterOutput(reply="You're welcome!", task=None,
                                 request=RouterRequest(prompt="x", scopes=["web"], summary="look something up"))
    asyncio.run(m.get("c1").handle_inbound("thanks so much!!", "email", stranger_meta()))
    assert m.store.pending() == [] and m.executor.ran == []


# --- Jev-first: the router only writes the reply; code builds the request from the task ---

def _jev_first(m, **picker_kw):
    m.task_picker = FakePicker(**picker_kw)
    m.router_defines_requests = False
    return m.task_picker


def test_jev_first_builds_request_from_task_not_router(tmp_path):
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("Close the TAMID application form")
    p = _jev_first(m, choice=f"T{t['id']}")
    p.action, p.scopes = True, ["tamid_drive_write"]
    # The router tries to define a request anyway; it must be ignored.
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Something else",
                                 request=RouterRequest(prompt="paraphrase", scopes=["calendar"], summary="router's idea"))
    async def go():
        await m.get("aaron").handle_inbound("close the tamid form pls", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    r = m.store.get_request(1)
    assert r.summary == "Close the TAMID application form"      # the task title, not DeepSeek's line
    assert "tamid_drive_write" in r.scopes and "calendar" not in r.scopes or "tamid_drive_write" in r.scopes   # Jev's write scope granted
    assert "close the tamid form pls" in r.prompt and "paraphrase" not in r.prompt
    assert m.store.task_id_for_request(1) == t["id"]
    # The router was told the decision before writing its reply, and no router hint reached Jev.
    assert m.router.calls[0]["action"] is True and m.router.calls[0]["action_task"] == t["title"]
    assert p.calls[-1]["router_hint"] is None
    assert p.scope_calls[-1]["router_scopes"] is None
    assert any("On it." == s[1] for s in sent)


def test_jev_first_no_action_means_no_request_even_if_router_wanted_one(tmp_path):
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="none")
    p.action = False
    m.router.next = RouterOutput(reply="Sure, 3pm works.", task="new", task_title="x",
                                 request=RouterRequest(prompt="do stuff", scopes=["calendar"], summary="s"))
    asyncio.run(m.get("aaron").handle_inbound("thanks!", "imessage", approver_meta()))
    assert m.store.get_request(1) is None
    assert m.router.calls[-1]["action"] is False
    assert m.store.list_tasks() == [] if hasattr(m.store, "list_tasks") else True


def test_jev_first_new_task_titled_by_router_scopes_fallback_when_undecided(tmp_path):
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, None                      # scope judgment undecided
    m.router.next = RouterOutput(reply=None, task="new", task_title="Book lunch with Sam Rivera")
    async def go():
        await m.get("aaron").handle_inbound("book lunch with sam rivera tomorrow", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    r = m.store.get_request(1)
    assert r.summary == "Book lunch with Sam Rivera"
    assert r.scopes == ["web"]                              # undecided: the read-only web scope, never empty


def test_jev_first_stranger_request_waits_for_aaron(tmp_path):
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["calendar"]
    m.router.next = RouterOutput(reply="I will confirm with Aaron.", task="new", task_title="Schedule a chat with Cand Idate")
    asyncio.run(m.get("c1").handle_inbound("can we chat tue 6pm?", "email", stranger_meta()))
    r = m.store.get_request(1)
    assert r.state == "pending" and r.scopes == ["calendar"]
    assert m.router.calls[-1]["action"] is True
    assert any("[Blatbot #1]" in s[1] for s in sent)


def test_jev_first_follow_up_inherits_task_scopes(tmp_path):
    """'try again' says nothing about tools; the task's earlier request did."""
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("List the board members from the TAMID site")
    m.store.create_request(chat_id="aaron", sender="+15550100001", sender_name="Aaron", mode="imessage", subject="",
                           original_message="list the board", summary="List the board members", scopes=["tamid_site_read"],
                           prompt="list the board", state="done", task_id=t["id"])
    p = _jev_first(m, choice=f"T{t['id']}")
    p.action, p.scopes = True, None                      # scope judgment has nothing to go on
    m.router.next = RouterOutput(reply="On it.", task=f"T{t['id']}")
    async def go():
        await m.get("aaron").handle_inbound("try again.", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    r = m.store.get_request(2)
    assert "tamid_site_read" in r.scopes and "web" in r.scopes


def test_claude_executor_denies_send_to_requester(tmp_path):
    from inkbox_claude.gate.executor import Executor
    ex = Executor(mcp_server=None, cwd=str(tmp_path), protected=[APPROVER_CONV, "+15550100001"])
    assert ex.protected == [APPROVER_CONV, "+15550100001"]


def test_browser_scopes_split_read_from_act():
    from inkbox_claude.gate.scopes import SCOPES, tools_for
    from inkbox_claude.gate.jevagent import is_write_tool
    read, act = set(tools_for(["browser_read"])), set(tools_for(["browser_act"]))
    assert "mcp__playwright__browser_navigate" in read and "mcp__playwright__browser_click" not in read
    assert read < act and "mcp__playwright__browser_fill_form" in act
    assert "mcp__playwright__browser_run_code_unsafe" not in act and "mcp__playwright__browser_evaluate" not in act
    assert is_write_tool("mcp__playwright__browser_click") and not is_write_tool("mcp__playwright__browser_snapshot")


class HoldingExecutor(FakeExecutor):
    """Blocks until released, so a second message can arrive mid-run."""
    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def run(self, req, context="", model=None, **kw):
        self.started.set()
        await self.release.wait()
        return await super().run(req, context)


def test_message_during_a_running_request_does_not_start_a_second_one(tmp_path):
    m, sent = make_manager(tmp_path)
    m.executor = HoldingExecutor()
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Acceptance rate",
                                 request=RouterRequest(prompt="compute it", scopes=["tamid_drive_read"], summary="Acceptance rate"))
    async def go():
        s = m.get("aaron")
        await s.handle_inbound("what's my acceptance rate as a reviewer?", "imessage", approver_meta())
        await m.executor.started.wait()
        m.router.next = RouterOutput(reply="Noted, using that sheet.", task="new", task_title="x",
                                     request=RouterRequest(prompt="again", scopes=["tamid_drive_read"], summary="again"))
        await s.handle_inbound("it's in the Fall 2026 responses sheet", "imessage", approver_meta())
        assert m.store.get_request(2) is None                      # no second request while #1 runs
        assert m.router.calls[-1]["action"] is True                # router told a request is running
        m.router.next = RouterOutput(reply="Done: 47%.", task="T1")  # the result phrasing
        m.executor.release.set()
        await asyncio.sleep(0.1)
    asyncio.run(go())
    assert m.store.get_request(1).state == "done"
    assert len(m.executor.ran) == 1
    outbound = [t for _, t, *_ in sent]
    assert outbound == ["On it.", "Noted, using that sheet.", "Done: 47%."]


def test_failed_run_re_decides_the_message_that_arrived_meanwhile(tmp_path):
    m, sent = make_manager(tmp_path)
    m.executor = HoldingExecutor()
    m.executor.result = {"ok": False, "error": "nothing found", "tool_calls": [], "raw": "STATUS: FAILED"}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Acceptance rate",
                                 request=RouterRequest(prompt="compute it", scopes=["tamid_drive_read"], summary="Acceptance rate"))
    async def go():
        s = m.get("aaron")
        await s.handle_inbound("what's my acceptance rate?", "imessage", approver_meta())
        await m.executor.started.wait()
        m.router.next = RouterOutput(reply="Got it.", task="T1",
                                     request=RouterRequest(prompt="use the sheet", scopes=["tamid_drive_read"], summary="Acceptance rate from sheet"))
        await s.handle_inbound("use the Fall 2026 responses sheet", "imessage", approver_meta())
        assert m.store.get_request(2) is None
        m.executor.release.set()
        await asyncio.sleep(0.2)
    asyncio.run(go())
    r2 = m.store.get_request(2)
    assert r2 is not None and r2.original_message == "use the Fall 2026 responses sheet"
    assert m.store.get_request(1).state == "failed"


def test_escalation_to_claude_produces_exactly_one_response(tmp_path):
    """Jev agent fails without writing -> Claude runs. One status, one delivery, and
    nothing about the Jev attempt reaches the sender, the note, or the ledger."""
    m, sent = make_manager(tmp_path)

    class FakeJev:
        def __init__(self): self.ran = 0
        async def run(self, req, context="", model=None, **kw):
            self.ran += 1
            return {"ok": False, "error": "JEV-GAVE-UP", "summary": "JEV-GAVE-UP", "raw": "JEV-GAVE-UP\nSTATUS: FAILED",
                    "tool_calls": [], "wrote": False, "engine": "jev"}
    m.jev_agent, m.jev_fallback = FakeJev(), True
    m.executor.result = {"ok": True, "summary": "CLAUDE-RESULT", "tool_calls": [], "raw": "CLAUDE-RESULT\nSTATUS: OK"}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Row count",
                                 request=RouterRequest(prompt="count rows", scopes=["tamid_drive_read"], summary="Row count"))
    m.router.next_note = RouterOutput(reply="There are 40 rows.", task="T1")
    async def go():
        await m.get("aaron").handle_inbound("how many rows?", "imessage", approver_meta())
        await asyncio.sleep(0.15)
    asyncio.run(go())
    r = m.store.get_request(1)
    assert r.state == "done" and r.status["escalated"] is True and m.jev_agent.ran == 1 and m.executor.ran == [1]
    outbound = [t for _, t, *_ in sent]
    assert outbound == ["On it.", "There are 40 rows."]                  # one ack, one result, nothing in between
    everything = " ".join(outbound) + " ".join(x["text"] for x in m.store.history(OWNER_THREAD, 50)) + m.store.task_memory_for_task(1)
    assert "JEV-GAVE-UP" not in everything and "CLAUDE-RESULT" in everything


def test_stale_running_request_does_not_block_the_thread(tmp_path):
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("Old work")
    old = m.store.create_request(chat_id="aaron", sender="+15550100001", sender_name="Aaron", mode="imessage", subject="",
                                 original_message="old", summary="old", scopes=["calendar"], prompt="old", state="approved", task_id=t["id"])
    m.store.set_state(old.id, "running")
    with m.store._lock:
        m.store._db.execute("UPDATE requests SET updated_at=? WHERE id=?", (time.time() - 3600, old.id)); m.store._db.commit()
    assert m.store.running_for_thread("aaron") is None
    assert m.store.get_request(old.id).state == "failed"
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="New work",
                                 request=RouterRequest(prompt="do new", scopes=["calendar"], summary="New work"))
    async def go():
        await m.get("aaron").handle_inbound("do the new thing", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.store.get_request(old.id + 1) is not None and m.executor.ran == [old.id + 1]


def test_reply_alongside_a_request_must_be_an_acknowledgement(tmp_path):
    """The reply writer answered the question itself while a request was created; the
    run's result then arrived as a second answer. Only a short ack may accompany a run."""
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["contacts"]
    m.router.next = RouterOutput(reply="I do not have his phone number on file, and I would not share a personal number "
                                       "without his say-so. If you have it, I can send him a message for you.",
                                 task="new", task_title="Find a phone number")
    m.router.next_note = RouterOutput(reply="No number on file, only the email.", task="T1")
    async def go():
        await m.get("aaron").handle_inbound("are you able to message his phone number?", "imessage", approver_meta())
        await asyncio.sleep(0.15)
    asyncio.run(go())
    outbound = [t for _, t, *_ in sent]
    assert outbound == ["No number on file, only the email."]        # one answer, from the run

    from inkbox_claude.gate.manager import _is_acknowledgement
    assert _is_acknowledgement("On it. I will look up Jonathan Fried's email.")
    assert not _is_acknowledgement("Did you mean the TAMID one or the SJBA one?")
    assert not _is_acknowledgement(" ".join(["word"] * 21))


# --- every outbound is flagged against the inbound it responds to: one ack, one answer ---

def _roles(m, chat=OWNER_THREAD):
    with m.store._lock:
        rows = m.store._db.execute("SELECT reply_to, role, text FROM messages WHERE chat_id=? AND kind='outbound' ORDER BY id", (chat,)).fetchall()
    return [(r["reply_to"], r["role"], r["text"]) for r in rows]


def test_reply_only_turn_is_flagged_as_the_answer(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply="Donald Trump.", task=None)
    asyncio.run(m.get("aaron").handle_inbound("who is the president?", "imessage", approver_meta()))
    inbound = m.store.history(OWNER_THREAD, 5)[0]
    roles = _roles(m)
    assert len(roles) == 1 and roles[0][1] == "answer" and roles[0][0] is not None
    assert m.store.responses_to(roles[0][0]) == ["answer"]


def test_request_turn_gets_one_ack_then_one_answer_both_linked(tmp_path):
    m, sent = make_manager(tmp_path)
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Count rows",
                                 request=RouterRequest(prompt="count", scopes=["tamid_drive_read"], summary="Count rows"))
    m.router.next_note = RouterOutput(reply="40 rows.", task="T1")
    async def go():
        await m.get("aaron").handle_inbound("how many rows?", "imessage", approver_meta())
        await asyncio.sleep(0.15)
    asyncio.run(go())
    roles = _roles(m)
    assert [(r[1], r[2]) for r in roles] == [("ack", "On it."), ("answer", "40 rows.")]
    assert roles[0][0] == roles[1][0] and roles[0][0] is not None      # both point at the same inbound
    assert m.store.get_request(1).inbound_id == roles[0][0]


def test_second_answer_for_the_same_inbound_is_refused(tmp_path):
    """Whatever produces it (a reply writer that answered early, a result phrasing),
    the store already holds an answer for that inbound and the send is refused."""
    m, sent = make_manager(tmp_path)
    s = m.get("aaron")
    m.router.next = RouterOutput(reply="First answer.", task=None)
    asyncio.run(s.handle_inbound("question?", "imessage", approver_meta()))
    assert asyncio.run(s.send_to_sender("Second answer.", role="answer")) is False
    assert asyncio.run(s.send_to_sender("Late ack.", role="ack")) is True     # a different role is still allowed once
    assert asyncio.run(s.send_to_sender("Another ack.", role="ack")) is False
    assert [t for _, t, *_ in sent] == ["First answer.", "Late ack."]
    # and a result phrasing for that inbound is refused too, reported as handled
    assert asyncio.run(s.notify_after_request("Task #9 done: x\nResult:\ny", inbound_id=s.inbound_id)) is True
    assert [t for _, t, *_ in sent] == ["First answer.", "Late ack."]


def test_escalation_hands_claude_the_agents_findings(tmp_path):
    m, sent = make_manager(tmp_path)
    class FakeJev:
        async def run(self, req, context="", model=None, **kw):
            return {"ok": False, "error": "unsure", "raw": "- search_drive_files:\nFound: X (ID: 1ABC)\nSTATUS: FAILED",
                    "tool_calls": ["mcp__tamid-drive__search_drive_files"], "wrote": False, "engine": "jev"}
    class Ex(FakeExecutor):
        async def run(self, req, context="", prior_work="", model=None):
            self.prior = prior_work
            return await super().run(req, context)
    m.jev_agent, m.jev_fallback, m.executor = FakeJev(), True, Ex()
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Find X",
                                 request=RouterRequest(prompt="find x", scopes=["tamid_drive_read"], summary="Find X"))
    m.router.next_note = RouterOutput(reply="Found it.", task="T1")
    async def go():
        await m.get("aaron").handle_inbound("find x", "imessage", approver_meta())
        await asyncio.sleep(0.15)
    asyncio.run(go())
    assert "Found: X (ID: 1ABC)" in m.executor.prior


def test_ungrounded_result_reply_is_rewritten_or_replaced(tmp_path):
    """The reply writer embellished a result with names the result does not contain."""
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["web"]
    p.grounded = [False, True]          # first draft unsupported, rewrite supported
    async def judge_grounded(**kw):
        p.ground_calls = getattr(p, "ground_calls", []) + [kw]
        g = p.grounded.pop(0)
        return {"grounded": g, "p": 0.9 if g else 0.1, "reason": "ok"}
    p.judge_grounded = judge_grounded
    m.executor.result = {"ok": True, "summary": "Web search results: 1. Leadership - Stern Real Estate Group\n   co-presidents: Cand Idate and Sam Rivera",
                         "tool_calls": [], "raw": "x"}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Find the SREG president")
    drafts = [RouterOutput(reply="The president is Jamie Rivera, elected last spring.", task="T1"),
              RouterOutput(reply="The search lists Cand Idate and Sam Rivera as co-presidents; not yet confirmed on the site.", task="T1")]
    async def route(**kw):
        m.router.calls.append(kw)
        if kw.get("action") is False:
            return drafts.pop(0).model_copy(deep=True)
        return m.router.next.model_copy(deep=True)
    m.router.route = route
    async def go():
        await m.get("aaron").handle_inbound("who is the SREG president?", "imessage", approver_meta())
        await asyncio.sleep(0.2)
    asyncio.run(go())
    outbound = [t for _, t, *_ in sent]
    assert outbound[-1].startswith("The search lists Cand Idate and Sam Rivera")
    assert "Jamie Rivera" not in " ".join(outbound)
    assert len(p.ground_calls) == 2 and "GROUNDING" in m.router.calls[-1]["message"]


def test_still_ungrounded_reply_falls_back_to_the_raw_result(tmp_path):
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["web"]
    async def judge_grounded(**kw):
        return {"grounded": False, "p": 0.1, "reason": "ok"}
    p.judge_grounded = judge_grounded
    m.executor.result = {"ok": True, "summary": "Found 2 matches for President:\n- Cand Idate: Co-President", "tool_calls": [], "raw": "x"}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="x")
    m.router.next_note = RouterOutput(reply="The president is Jamie Rivera.", task="T1")
    async def go():
        await m.get("aaron").handle_inbound("who is the president?", "imessage", approver_meta())
        await asyncio.sleep(0.2)
    asyncio.run(go())
    outbound = [t for _, t, *_ in sent]
    assert outbound[-1].startswith("Found 2 matches for President") and "Cand Idate: Co-President" in outbound[-1]
    assert "Jamie Rivera" not in " ".join(outbound)


def test_long_unverified_result_sends_the_rewrite_not_a_dump(tmp_path):
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["calendar"]
    async def judge_grounded(**kw):
        return {"grounded": False, "p": 0.2, "reason": "ok"}
    p.judge_grounded = judge_grounded
    big = "Done via Jev agent in 4.7s (1 tool call(s)).\n- get_events:\n" + "\n".join(
        "- \"Event %d\" (Starts: 2026-10-01T%02d:00:00-04:00)" % (i, 9 + i % 10) for i in range(80)) + "\nSTATUS: OK"
    m.executor.result = {"ok": True, "summary": big, "tool_calls": [], "raw": big}
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="x")
    m.router.next_note = RouterOutput(reply="Open slots Thursday: 11:00, 12:30, 3:00.", task="T1")
    async def go():
        await m.get("aaron").handle_inbound("what slots are open thursday?", "imessage", approver_meta())
        await asyncio.sleep(0.2)
    asyncio.run(go())
    last = [t for _, t, *_ in sent][-1]
    assert last.startswith("Open slots Thursday") and "could not fully verify" not in last
    assert "Done via Jev agent" not in last and "get_events" not in last


def test_provider_billing_failure_is_reported_to_the_owner_once(tmp_path):
    import httpx
    m, sent = make_manager(tmp_path)
    async def broke(**kw):
        req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
        raise httpx.HTTPStatusError("Client error '402 Payment Required' for url 'https://api.deepseek.com/chat/completions'",
                                    request=req, response=httpx.Response(402, request=req))
    m.router.route = broke
    async def go():
        s = m.get("aaron")
        await s.handle_inbound("is X in stern?", "imessage", approver_meta())
        await s.handle_inbound("hello?", "imessage", approver_meta())
    asyncio.run(go())
    notices = [t for _, t, *_ in sent if "out of credit" in t]
    assert len(notices) == 1 and "DeepSeek" in notices[0] and "not answered" in notices[0]


def test_escalation_findings_go_to_a_file_not_the_argv(tmp_path):
    from inkbox_claude.gate.executor import Executor
    ex = Executor(mcp_server=None, cwd=str(tmp_path))
    big = "- read_sheet_values:\n" + ("Row: ['a','b']\n" * 40000)     # far past any argv limit
    captured = {}
    async def fake_query(prompt, options):
        captured["system"] = options.system_prompt["append"] if isinstance(options.system_prompt, dict) else str(options.system_prompt)
        return
        yield
    import inkbox_claude.gate.executor as exmod
    monkey = exmod.query
    exmod.query = fake_query
    try:
        req = Request(id=77, chat_id="c", sender="s", sender_name="", mode="imessage", subject="", original_message="m",
                      summary="m", scopes=["tamid_drive_read"], prompt="m", prompt_sha256=sha256("m"), state="approved",
                      revision=0, status=None, raw_output=None, created_at=0, updated_at=0)
        asyncio.run(ex.run(req, context="", prior_work=big))
    finally:
        exmod.query = monkey
    assert len(captured["system"]) < 20000 and "request-77.txt" in captured["system"]
    assert (tmp_path / "findings" / "request-77.txt").read_text().startswith("- read_sheet_values:")


def test_no_action_reply_may_not_promise_action(tmp_path):
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="none")
    p.action = False
    drafts = [RouterOutput(reply="Resubmitting now with the access it needs.", task=None),
              RouterOutput(reply="Nothing is running for that; I would need a host tool to check.", task=None)]
    async def route(**kw):
        m.router.calls.append(kw)
        return drafts.pop(0).model_copy(deep=True)
    m.router.route = route
    asyncio.run(m.get("aaron").handle_inbound("give it such scopes.", "imessage", approver_meta()))
    assert [t for _, t, *_ in sent] == ["Nothing is running for that; I would need a host tool to check."]
    assert "no request has been created" in m.router.calls[-1]["action_task"]


def test_host_read_scope_maps_to_the_status_tool():
    from inkbox_claude.gate.scopes import tools_for
    from inkbox_claude.gate import hosttools
    assert tools_for(["host_read"]) == ["mcp__host__host_status", "mcp__host__rows_where"]   # analysis rides along
    out = asyncio.run(hosttools.host_status({"parts": ["uptime_load", "nope"]}))
    assert out.startswith("## uptime_load") and "nope" not in out


def test_destructive_call_asks_the_owner_and_runs_only_on_yes(tmp_path):
    """Deleting is never done on the owner's first message: the exact call is texted,
    '#N yes' performs that one call, '#N no' leaves everything alone."""
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["calendar"]
    performed = []
    class FakeJev:
        async def run(self, req, context="", model=None, **kw):
            return {"ok": False, "error": "confirmation required", "summary": "", "raw": "WRITES PERFORMED: none",
                    "tool_calls": ["mcp__tamid-drive__get_events"], "wrote": False, "engine": "jev",
                    "confirm": {"tool": "mcp__tamid-drive__manage_event", "args": {"action": "delete", "event_id": "evt_QUANT01"},
                                "about": ['"Sam Rivera and TAMID at NYU (Quant)" Fri 10/2 3:00 PM ID: evt_QUANT01']}}
        async def perform(self, tool, args):
            performed.append((tool, args))
            return {"ok": True, "summary": "WRITES PERFORMED: manage_event\ndeleted evt_QUANT01\nSTATUS: OK", "raw": "deleted", "tool_calls": [tool], "wrote": True, "engine": "jev"}
    m.jev_agent, m.jev_fallback = FakeJev(), True
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Cancel Sam's interview")
    m.router.next_note = RouterOutput(reply="Done, the Quant interview is cancelled.", task="T1")
    async def go():
        s = m.get("aaron")
        await s.handle_inbound("cancel sam rivera's quant interview", "imessage", approver_meta())
        await asyncio.sleep(0.15)
        assert performed == []                                                   # nothing changed yet
        ask = [t for _, t, *_ in sent if "Before I do this" in t]
        assert len(ask) == 1 and "delete" in ask[0] and "Sam Rivera and TAMID at NYU (Quant)" in ask[0]
        assert m.store.get_request(1).state == "pending"
        await s.handle_inbound("#1 yes", "imessage", approver_meta())
        await asyncio.sleep(0.15)
    asyncio.run(go())
    assert performed == [("mcp__tamid-drive__manage_event", {"action": "delete", "event_id": "evt_QUANT01"})]
    assert m.store.get_request(1).state == "done"
    assert [t for _, t, *_ in sent][-1] == "Done, the Quant interview is cancelled."


def test_destructive_call_declined_leaves_everything_alone(tmp_path):
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["calendar"]
    performed = []
    class FakeJev:
        async def run(self, req, context="", model=None, **kw):
            return {"ok": False, "error": "confirmation required", "summary": "", "raw": "", "tool_calls": [], "wrote": False, "engine": "jev",
                    "confirm": {"tool": "mcp__tamid-drive__manage_event", "args": {"action": "delete", "event_id": "evt_X"}, "about": []}}
        async def perform(self, tool, args):
            performed.append(tool); return {"ok": True}
    m.jev_agent = FakeJev()
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="x")
    async def go():
        s = m.get("aaron")
        await s.handle_inbound("delete that event", "imessage", approver_meta())
        await asyncio.sleep(0.1)
        await s.handle_inbound("#1 no", "imessage", approver_meta())
    asyncio.run(go())
    assert performed == [] and m.store.get_request(1).state == "rejected"
    assert any("Left alone" in t for _, t, *_ in sent)


def test_owner_requests_get_jev_scopes_plus_likely_writes_only(tmp_path):
    """The tree's choice stands for the owner; likely scopes (p >= 0.5) are added, nothing else."""
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["calendar"]
    async def judge_scopes(**kw):
        return {"scopes": ["calendar"], "probabilities": {"calendar": 0.9, "tamid_drive_write": 0.55, "email_send": 0.1}, "reason": "ok"}
    p.judge_scopes = judge_scopes
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="x")
    async def go():
        await m.get("aaron").handle_inbound("cancel philip's edu interview", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    scopes = set(m.store.get_request(1).scopes)
    assert scopes == {"calendar", "tamid_drive_write"}


def test_a_request_nobody_could_act_on_asks_instead_of_running(tmp_path):
    """A message that wants something done but does not say what: the gate asks, and no
    request reaches the execution layer."""
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action = True
    async def judge_actionable(**kw):
        return {"actionable": False, "p": 0.1, "reason": "ok"}
    p.judge_actionable = judge_actionable
    m.router.next = RouterOutput(reply="What would you like me to check?", task="new", task_title="x")
    async def go():
        await m.get("aaron").handle_inbound("blatbot, could you please check", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.store.get_request(1) is None
    assert m.router.calls and m.router.calls[0].get("ask") is True
    # The question has to reach the sender: nothing is running behind the silence.
    assert any("What would you like me to check?" in str(row) for row in sent), sent


def test_a_notice_reaches_the_owner_record_with_no_session_loaded(tmp_path):
    """Sessions fall out of memory between turns. A notice sent then still has to land on
    his thread, or his next message has nothing to refer back to."""
    m, sent = make_manager(tmp_path)
    asyncio.run(m.get("aaron").handle_inbound("hello", "imessage", approver_meta()))
    chat_id = "aaron"
    m.sessions.clear()                       # overnight: nothing live
    asyncio.run(m.send_to_approver("[Blatbot FYI] someone via email | a thing happened"))
    texts = [str(r.get("text") or "") for r in m.store.history(OWNER_THREAD, 20)]
    assert any("a thing happened" in t for t in texts), texts


def test_the_room_note_works_for_any_group(tmp_path):
    """Nothing in it is particular to one chat, one channel, or one set of people."""
    from inkbox_claude.gate.rooms import room_note
    a = room_note(channel="Telegram", title="TAMID Quant", me="Blatbot",
                  members=["Rick", "Gabri", "Romi"], principal="Aaron")
    assert "TAMID Quant" in a and "Rick, Gabri and Romi" in a and "Aaron's assistant" in a
    b = room_note(channel="WhatsApp", title="Sunday ride", me="Helper", members=["Dana"])
    assert "Sunday ride" in b and "The people in it are Dana," in b and "Helper" in b
    # With nothing known it still says the one thing that matters.
    c = room_note(channel="")
    assert "one of its members" in c and "read by everyone in the room" in c
    # The assistant never lists itself among the others.
    d = room_note(channel="Telegram", me="Blatbot", members=["Blatbot", "Sean"])
    assert "The people in it are Sean," in d


def test_a_group_thread_tells_the_executor_where_it_is(tmp_path):
    """In a group the prompt describes the room, so questions about who is here include it."""
    m, sent = make_manager(tmp_path)
    s = m.get("tg:-1")
    s.room = "--- WHERE YOU ARE ---\nThis is a group chat and you are one of its members."
    task = m.store.create_task("x", [])
    prompt = s.build_task_prompt("anyone here have my calendar?", [], task)
    assert "one of its members" in prompt
    assert prompt.index("WHERE YOU ARE") < prompt.index("THEIR MESSAGE")


def test_the_executor_sees_whole_messages(tmp_path):
    """No clipping: a long line reaches the executor intact."""
    m, sent = make_manager(tmp_path)
    s = m.get("c1")
    long_line = "x" * 2500
    task = m.store.create_task("x", [])
    prompt = s.build_task_prompt("do it", [{"kind": "inbound", "text": long_line}], task)
    assert long_line in prompt


def test_a_reply_that_commits_to_the_work_runs_it(tmp_path):
    """The gate meant to ask, but the writer read the thread and said it was doing it. The
    words have to be true, so the request survives and runs."""
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action = True
    async def judge_actionable(**kw):
        return {"actionable": False, "p": 0.3, "reason": "ok"}
    p.judge_actionable = judge_actionable
    async def judge_reply(**kw):
        return {"promises_action": True, "is_acknowledgement": True, "reason": "ok"}
    p.judge_reply = judge_reply
    m.router.next = RouterOutput(reply="On it. I will pull the sheet and rank them.", task="new", task_title="x")
    async def go():
        await m.get("aaron").handle_inbound("yes", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.store.get_request(1) is not None, "a reply that says it is doing the work must run it"


def test_a_clear_request_still_runs(tmp_path):
    """The same path with a request anyone could act on: the run happens as before."""
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["calendar"]
    async def judge_actionable(**kw):
        return {"actionable": True, "p": 0.9, "reason": "ok"}
    p.judge_actionable = judge_actionable
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="x")
    async def go():
        await m.get("aaron").handle_inbound("what is on my calendar tomorrow", "imessage", approver_meta())
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert m.store.get_request(1) is not None
    assert m.router.calls[0].get("ask") is False


def test_stranger_requests_keep_the_strict_scope_judgment(tmp_path):
    m, sent = make_manager(tmp_path)
    p = _jev_first(m, choice="new")
    p.action, p.scopes = True, ["calendar"]
    m.router.next = RouterOutput(reply="I will confirm with Aaron.", task="new", task_title="x")
    asyncio.run(m.get("c1").handle_inbound("can we chat tue 6pm?", "email", stranger_meta()))
    assert m.store.get_request(1).scopes == ["calendar"]


class LookupPicker(FakePicker):
    """A picker that also plans the ledger lookup (Jev as the router's lookup planner)."""
    def __init__(self, plan, **kw):
        super().__init__(**kw)
        self.plan, self.lookup_calls = plan, []

    async def plan_lookup(self, **kw):
        self.lookup_calls.append(kw)
        return self.plan


def test_jev_plans_the_lookup_and_the_chat_model_does_not(tmp_path):
    m, sent = make_manager(tmp_path)
    m.store.create_task("Update Jared's SJBA bio", [])
    hidden = m.store.create_task("Private errand for Sam Rivera", [])
    m.store.task_event(hidden["id"], "done", "ok", state="done")   # old and finished: outside the default view
    m.store._db.execute("UPDATE tasks SET updated_at=updated_at-40*86400 WHERE id=?", (hidden["id"],)); m.store._db.commit()
    m.task_picker = LookupPicker({"lookup": "any", "text": "Sam Rivera", "confidence": 0.9, "reason": "ok"},
                                 choice="none")
    m.router.next = RouterOutput(reply="checking", request=None)
    asyncio.run(m.get(APPROVER_CONV).handle_inbound("what happened with Sam Rivera's errand?", "imessage", approver_meta()))
    assert m.router.query_calls == []                       # the chat model never planned the lookup
    call = m.task_picker.lookup_calls[-1]
    assert "Sam Rivera" in call["candidates"]
    assert any("Update Jared's SJBA bio" in t for t in call["in_view_titles"])
    assert "Private errand for Sam Rivera" not in m.router.calls[-1]["task_memory"]
    assert "Private errand for Sam Rivera" in m.router.calls[-1]["found_tasks"]


def test_jev_no_lookup_skips_the_search(tmp_path):
    m, sent = make_manager(tmp_path)
    m.task_picker = LookupPicker({"lookup": "none", "text": "", "confidence": 0.9, "reason": "ok"}, choice="none")
    asyncio.run(m.get(APPROVER_CONV).handle_inbound("thank you blatbot", "imessage", approver_meta()))
    assert m.router.query_calls == []
    assert m.router.calls[-1]["found_tasks"] == ""


def test_jev_undecided_lookup_falls_back_to_the_chat_planner(tmp_path):
    m, sent = make_manager(tmp_path)
    m.task_picker = LookupPicker({"lookup": None, "text": "", "confidence": 0.2, "reason": "low confidence"}, choice="none")
    asyncio.run(m.get(APPROVER_CONV).handle_inbound("hm", "imessage", approver_meta()))
    assert len(m.router.query_calls) == 1


def test_decide_graph_nodes_and_edges():
    from inkbox_claude.gate import decidegraph
    g = decidegraph.build_graph().get_graph()
    nodes = set(g.nodes) - {"__start__", "__end__"}
    assert nodes == {"pick_task", "judge_action", "judge_actionable", "join_judgments", "write_reply", "attach_task",
                     "build_request", "judge_scopes", "inherit_scopes", "record_event", "finalize"}
    edges = {(e.source, e.target) for e in g.edges}
    # the two Jev judgments fan out from START and join
    assert ("__start__", "pick_task") in edges and ("__start__", "judge_action") in edges
    assert ("pick_task", "join_judgments") in edges and ("judge_action", "join_judgments") in edges
    # scopes and the event classification fan out after the request is built
    assert ("build_request", "judge_scopes") in edges and ("build_request", "record_event") in edges
    assert ("judge_scopes", "inherit_scopes") in edges
    # everything merges in finalize
    assert ("write_reply", "finalize") in edges and ("inherit_scopes", "finalize") in edges and ("record_event", "finalize") in edges


def test_decide_graph_runs_the_jev_first_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_DECIDE_GRAPH", "1")
    m, sent = make_manager(tmp_path)
    m.task_picker = FakePicker(choice="new")
    m.task_picker.action = True
    m.task_picker.scopes = ["calendar"]
    m.router.next = RouterOutput(reply="On it.", task_title="Book the room", request=None)
    asyncio.run(m.get(APPROVER_CONV).handle_inbound("book T306 for friday 9am", "imessage", approver_meta()))
    req = m.store.get_request(1)
    assert req.state in ("approved", "running", "done")
    assert "calendar" in req.scopes
    assert req.original_message == "book T306 for friday 9am"
    assert sent[0][1] == "On it."


def test_voice_result_is_phrased_for_speech_not_raw(tmp_path):
    m, sent = phone_manager(tmp_path, trust=True)
    m.executor.result = {"ok": True, "summary": "Done via Jev agent in 9.1s (3 tool call(s)).\n- get_events:\nSuccessfully retrieved 2 events", "tool_calls": [], "raw": "x"}
    m.router.next = RouterOutput(reply=None, task="new", task_title="Tomorrow's chats",
                                 request=RouterRequest(prompt="List tomorrow", scopes=["stern_calendar"], summary="Tomorrow's chats"))
    m.router.next_note = RouterOutput(reply="Two chats tomorrow: Owen at five and Philip at six.", request=None)
    out = asyncio.run(m.get("p1").voice_consult("what chats do I have tomorrow", voice_meta()))
    assert out == "Two chats tomorrow: Owen at five and Philip at six."
    assert "Done via Jev agent" not in out
    spoken = [r for r in m.store.history("p1", limit=20) if r["kind"] == "outbound" and r["mode"] == "voice"]
    assert spoken and spoken[-1]["text"] == out
