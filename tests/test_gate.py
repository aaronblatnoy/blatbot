"""Gate tests: store state machine, approver command parsing, scopes,
executor hash refusal, routing with a stubbed router. No network."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate.router import RouterOutput, RouterRequest
from inkbox_claude.gate.scopes import SCOPES, tools_for
from inkbox_claude.gate.store import Store, sha256

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

    async def route(self, **kw):
        self.calls.append(kw)
        return self.next


class FakeExecutor:
    def __init__(self):
        self.ran = []
        self.result = {"ok": True, "summary": "did it", "tool_calls": ["mcp__tamid-drive__manage_event"], "raw": "did it"}

    async def run(self, req, context=""):
        self.ran.append(req.id)
        self.context = context
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
    assert "can we do tue 6pm?" in text and "Book Tue 6pm" in text and "calendar, email_send" in text
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
        assert m.store.get_request(rid).prompt == "P2 instead"
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
