"""Suggestion passes: at most two offers to Aaron, quiet when there is nothing, "#id yes"
runs it as his own request, "#id no" dismisses it for good."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate import suggestions as sg

APPROVER_CONV = "conv-aaron"


def _manager(tmp_path):
    sent = []

    async def send_fn(chat_id, text, mode, meta):
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV)
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    m.store.add_message(gm.OWNER_THREAD, "inbound", "make the orgcomm doc on my stern drive", "imessage")
    return m, sent


def _model(items, quiet=""):
    async def call(system, prompt):
        call.prompts.append(prompt)
        return json.dumps({"suggestions": items, "quiet_reason": quiet})
    call.prompts = []
    return call


IDEA = {"text": "I can redo the orgcomm Google Doc on your Stern Drive that failed Sunday.",
        "task": "Make the orgcomm deliverable a Google Doc on my Stern Drive.", "why": "request failed",
        "refs": ["request #625"]}
IDEA2 = {"text": "I can draft the reply to Sean about the NDA he asked for Tuesday.",
         "task": "Draft a reply to Sean Parnell with the NDA he asked for.", "why": "open thread",
         "refs": []}


def test_pass_sends_numbered_offers_to_aaron_only(tmp_path):
    async def run():
        m, sent = _manager(tmp_path)
        model = _model([IDEA, IDEA2, {**IDEA, "text": "third one"}])
        out = await sg.run_pass(m, "2026-11-27T09:00", model=model)
        assert out["outcome"] == "sent" and len(out["sent"]) == 2
        texts = [t for _, t, _, _ in sent]
        assert len(texts) == 2
        for sid, text in zip(out["sent"], texts):
            assert text.endswith(f"#{sid} yes / no")
            assert "[" not in text and "STATUS" not in text
            assert m.store.get_request(sid).state == sg.SUGGESTED
        assert all(target == f"imessage:{APPROVER_CONV}" for target, _, _, _ in sent)
        rows = m.store._db.execute("SELECT kind FROM outbox").fetchall()
        assert {r["kind"] for r in rows} == {"proactive"}
        # The same slot never runs twice.
        again = await sg.run_pass(m, "2026-11-27T09:00", model=model)
        assert again["outcome"] == "skipped" and len(sent) == 2
        # Suggested requests are not pending: a bare yes cannot pick them up.
        assert m.store.pending() == []
    asyncio.run(run())


def test_pass_stays_quiet(tmp_path):
    async def run():
        m, sent = _manager(tmp_path)
        out = await sg.run_pass(m, "s1", model=_model([], "nothing new"))
        assert out == {"outcome": "quiet", "sent": [], "note": "nothing new"}
        # Two still unanswered: the next pass does not even ask the model.
        await sg.run_pass(m, "s2", model=_model([IDEA, IDEA2]))
        model = _model([{**IDEA, "text": "Something else entirely about calendars", "task": "calendar"}])
        out = await sg.run_pass(m, "s3", model=model)
        assert out["outcome"] == "quiet" and model.prompts == []
        assert len(sent) == 2
    asyncio.run(run())


def test_dismissed_is_never_repeated(tmp_path):
    async def run():
        m, sent = _manager(tmp_path)
        out = await sg.run_pass(m, "s1", model=_model([IDEA]))
        sid = out["sent"][0]
        res = await sg.decide(m, sid, "no")
        assert res["state"] == "dismissed"
        assert m.store.get_request(sid).state == "rejected"
        again = await sg.decide(m, sid, "no")
        assert again["reply"] == f"#{sid} was dismissed."
        reworded = {**IDEA, "text": "I can redo the orgcomm Google Doc on your Stern Drive, it failed Sunday."}
        out = await sg.run_pass(m, "s2", model=_model([reworded]))
        assert out["outcome"] == "quiet" and len(sent) == 1
    asyncio.run(run())


def test_yes_runs_it_as_his_own_request(tmp_path):
    async def run():
        m, sent = _manager(tmp_path)
        ran = []

        async def fake_execute(req, notify=True, **kw):
            ran.append(req)
            return ""
        m.execute = fake_execute
        out = await sg.run_pass(m, "s1", model=_model([IDEA]))
        sid = out["sent"][0]
        session = SimpleNamespace(record_id=gm.OWNER_THREAD, mode="imessage", reply_meta={"sender": "+15550001111"})
        assert await sg.handle_reply(m, session, f"#{sid} yes") is True
        await asyncio.sleep(0)
        req = m.store.get_request(sid)
        assert req.state == "approved" and req.scopes and req.prompt == IDEA["task"]
        assert ran and ran[0].id == sid
        assert sg.get(m.store, sid)["status"] == "accepted"
        # Not a suggestion id: left for the ordinary commands.
        assert await sg.handle_reply(m, session, "#99999 yes") is False
    asyncio.run(run())


def test_console_decide_goes_through_the_suggestion(tmp_path):
    async def run():
        m, sent = _manager(tmp_path)
        out = await sg.run_pass(m, "s1", model=_model([IDEA]))
        sid = out["sent"][0]
        res = await m.decide_request(sid, "no", actor="console")
        assert res["ok"] and res["state"] == "dismissed"
        view = sg.console_view(m.store)
        assert view["rows"][0]["status"] == "dismissed" and view["passes"]
    asyncio.run(run())


def test_due_slot_and_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv("SUGGESTION_TIMES", "9:00, 1pm, 18:00")
    at = datetime(2026, 11, 27, 13, 10, tzinfo=sg.TZ).timestamp()
    assert sg.due_slot(at) == "2026-11-27T13:00"
    assert sg.due_slot(at + 3600) is None

    async def run():
        m, sent = _manager(tmp_path)
        out = await sg.run_pass(m, "s1", model=_model([IDEA]))
        sid = out["sent"][0]
        assert sg.expire_old(m.store, time.time() + sg.EXPIRE_S + 5) == 1
        assert sg.get(m.store, sid)["status"] == "expired"
        assert m.store.get_request(sid).state == "expired"
    asyncio.run(run())
