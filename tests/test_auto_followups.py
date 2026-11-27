"""The gate's own follow-ups: a reminder on a request waiting on Aaron, and a
check-back on a run that passed the live timeout. No network."""

from __future__ import annotations

import asyncio
import time

from test_gate import make_manager


def _request(m, state="running"):
    task = m.store.create_task("Long job")
    m.store.set_thread("imessage:person", "idle", mode="imessage", meta={"conversation_id": "person"})
    return m.store.create_request(
        chat_id="imessage:person", sender="person", sender_name="Person", mode="imessage",
        subject="", original_message="Find the file", summary="Find the file", scopes=["host_read"],
        prompt="Find the file", state=state, task_id=task["id"])


def _pending(m, req_id, kind):
    return [r for r in m.store.console_followups() if r["request_id"] == req_id
            and r["kind"] == kind and r["status"] == "pending"]


def test_ask_sets_one_reminder_that_fires_with_the_handle_and_answer_cancels(tmp_path):
    m, sent = make_manager(tmp_path)
    req = _request(m)
    status = {"ok": False, "ask": {"question": "Which folder?", "kind": "input"},
              "deferred": {"kind": "ask", "id": "call-1", "session_id": "s"}}

    asyncio.run(m.handle_ask(req, status))
    asyncio.run(m.handle_ask(m.store.get_request(req.id), status))
    rows = _pending(m, req.id, "remind")
    assert len(rows) == 1
    assert rows[0]["due_at"] < time.time() + 24 * 3600 - 3 * 3600

    sent.clear()
    assert m.store.mark_followup_fired(rows[0]["id"])
    asyncio.run(m._run_followup(m.store.get_followup(rows[0]["id"])))
    texts = [t for _c, t, _m, _meta in sent]
    assert len(texts) == 1 and f"Still waiting on you (#{req.id})" in texts[0]
    assert f"#{req.id} and your answer" in texts[0] and "Which folder?" in texts[0]
    assert m.store.get_followup(rows[0]["id"])["result"] == "Reminded Aaron."

    # A second wait arms a fresh reminder; answering retires it.
    m._arm_reminder(m.store.get_request(req.id))
    assert len(_pending(m, req.id, "remind")) == 1
    m._cancel_auto_followups(req.id, ["remind"])
    assert _pending(m, req.id, "remind") == []


def test_reminder_says_nothing_once_the_request_was_answered(tmp_path):
    m, sent = make_manager(tmp_path)
    req = _request(m, state="pending")
    fid = m._arm_reminder(req)
    m.store.set_state(req.id, "done", status={"ok": True})
    sent.clear()
    assert m.store.mark_followup_fired(fid)
    asyncio.run(m._run_followup(m.store.get_followup(fid)))
    assert sent == []


def test_timed_out_run_gets_a_checkback_that_tells_the_person_when_it_was_lost(tmp_path):
    m, sent = make_manager(tmp_path)
    req = _request(m, state="approved")
    m.executor.result = {"ok": False, "timed_out": True, "error": "timed out after 60s",
                         "summary": "Still working on that one. I'll text you the answer.",
                         "final_status": "Still working on that one. I'll text you the answer.",
                         "tool_calls": []}
    asyncio.run(m.execute(m.store.get_request(req.id)))
    rows = _pending(m, req.id, "checkback")
    assert len(rows) == 1

    # Still running in the background: check again, say nothing.
    m.executor.late_runs = {req.id}
    sent.clear()
    assert m.store.mark_followup_fired(rows[0]["id"])
    asyncio.run(m._run_followup(m.store.get_followup(rows[0]["id"])))
    assert sent == []
    again = _pending(m, req.id, "checkback")
    assert len(again) == 1 and again[0]["id"] != rows[0]["id"]

    # Lost (a restart cut it off): close it and tell the person.
    m.executor.late_runs = set()
    assert m.store.mark_followup_fired(again[0]["id"])
    asyncio.run(m._run_followup(m.store.get_followup(again[0]["id"])))
    assert len(sent) == 1 and sent[0][0] == "imessage:person"
    assert f"stopped before it finished, so I have no answer for it yet (#{req.id})" in sent[0][1]
    done = m.store.get_request(req.id)
    assert done.state == "failed" and not done.status.get("timed_out")


def test_late_result_retires_the_checkback(tmp_path):
    m, _sent = make_manager(tmp_path)
    req = _request(m)
    m.store.set_state(req.id, "failed", status={"ok": False, "timed_out": True})
    m._arm_checkback(m.store.get_request(req.id))
    assert len(_pending(m, req.id, "checkback")) == 1
    asyncio.run(m.deliver_late_result(m.store.get_request(req.id),
                                      {"ok": True, "summary": "Found it.", "tool_calls": []}))
    assert _pending(m, req.id, "checkback") == []


def test_auto_followups_do_not_use_the_runs_three_slots(tmp_path):
    from inkbox_claude.gate import hosttools

    m, _sent = make_manager(tmp_path)
    req = _request(m, state="approved")
    m._arm_reminder(req)
    m._arm_checkback(req)
    for n in range(3):
        assert "I will check back" in hosttools.set_followup({"in_minutes": 5 + n, "note": f"n{n}"}, m, req)
