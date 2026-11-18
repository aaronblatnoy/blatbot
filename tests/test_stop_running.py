"""Aaron's "no" stops a running request: the run is cancelled, the request is marked
cancelled, nothing more is sent for it, and it is never resumed after a restart."""

from __future__ import annotations

import asyncio

from inkbox_claude.gate import runguard, wording
from inkbox_claude.gate.router import RouterOutput, RouterRequest

from test_gate import HoldingExecutor, approver_meta, make_manager


def _start_running(m):
    m.executor = HoldingExecutor()
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="Write to the group",
                                 request=RouterRequest(prompt="message the group", scopes=["web"],
                                                       summary="Write to the group"))


def test_parse_stop_words():
    for w in ("no", "No", "NO!", "stop", "Stop.", "cancel", "abort", "nvm", " nooo "):
        assert wording.parse_stop(w) == (True, None), w
    assert wording.parse_stop("#520 no") == (True, 520)
    assert wording.parse_stop("stop #520") == (True, 520)
    for w in ("no thanks, use the other sheet", "stop by at 5", "know", "yes"):
        assert wording.parse_stop(w)[0] is False, w


def test_plain_no_during_a_running_request_cancels_it_and_nothing_more_is_sent(tmp_path):
    m, sent = make_manager(tmp_path)
    _start_running(m)

    async def go():
        s = m.get("aaron")
        await s.handle_inbound("tell the group the plan", "imessage", approver_meta())
        await m.executor.started.wait()
        await s.handle_inbound("NO", "imessage", approver_meta())
        m.router.next = RouterOutput(reply="Done, told them.", task="T1")
        m.executor.release.set()
        await asyncio.sleep(0.1)
    asyncio.run(go())
    req = m.store.get_request(1)
    assert req.state == "rejected" and req.status.get("cancelled") is True
    assert m.store.request_cancelled(1)
    outbound = [t for _, t, *_ in sent]
    assert outbound == ["On it.", "Stopped #1. Nothing had been sent."]


def test_mid_score_judgment_asks_instead_of_killing(tmp_path):
    m, sent = make_manager(tmp_path)
    _start_running(m)

    async def fake_preflight(*a, **k):
        return {"event": {"kind": None, "raw_kind": "declined_or_cancelled", "kind_conf": 0.62},
                "lookup_inputs": {}}

    async def go():
        s = m.get("aaron")
        await s.handle_inbound("tell the group the plan", "imessage", approver_meta())
        await m.executor.started.wait()
        m.task_picker = object()
        m.router_defines_requests = False
        s.run_preflight = fake_preflight
        async def fake_memory(*a, **k):
            s._ledger_memory = ""
            return ""
        s.indexed_task_memory = fake_memory
        s.lookup_tasks = lambda *a, **k: _done("")
        s._ledger_memory = ""
        await s.handle_inbound("hmm maybe hold off on that", "imessage", approver_meta())
        assert m.store.get_request(1).state == "running"
        m.router.next = RouterOutput(reply="Done.", task="T1")
        m.executor.release.set()
        await asyncio.sleep(0.1)
    asyncio.run(go())
    assert m.store.get_request(1).state == "done"
    assert wording.ask_stop_line(1) in [t for _, t, *_ in sent]


async def _done(value):
    return value


def test_cancelled_request_is_not_resumed_on_startup(tmp_path):
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("x")
    r = m.store.create_request(chat_id="c1", sender="a", sender_name="A", mode="imessage", subject="",
                               original_message="m", summary="s", scopes=["web"], prompt="p",
                               state="running", task_id=t["id"])
    m.store.cancel_request(r.id)
    # Nothing later may move it out of the cancelled state, and startup leaves it alone.
    m.store.set_state(r.id, "running")
    assert m.store.get_request(r.id).state == "rejected"
    asyncio.run(m.recover_interrupted())
    assert m.store.get_request(r.id).state == "rejected"
    assert m.store.running_for_thread("c1") is None
    assert asyncio.run(m.execute(m.store.get_request(r.id))) == ""
    assert sent == [] and m.executor.ran == []


def test_send_attempted_after_cancel_is_refused(tmp_path):
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("x")
    r = m.store.create_request(chat_id="c1", sender="a", sender_name="A", mode="imessage", subject="",
                               original_message="m", summary="s", scopes=["web"], prompt="p",
                               state="running", task_id=t["id"])

    async def go():
        token = runguard.bind(r.id)
        try:
            await m.send("whatsapp:group-1", "first", "whatsapp", {})
            runguard.cancel(r.id)
            m.store.cancel_request(r.id)
            await m.send("whatsapp:group-1", "second", "whatsapp", {})
        finally:
            runguard.unbind(token)
    asyncio.run(go())
    assert [x[1] for x in sent] == ["first"]
    assert runguard.sent(r.id) == ["whatsapp to whatsapp:group-1"]
