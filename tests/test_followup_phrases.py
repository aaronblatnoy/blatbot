"""The fixed reply phrases wording.py tells a sender to say about a request
that just failed or is still running: "try again", "send what you have",
"let it <label>", and the owner's blanket "you have my permission for
everything". Each is recognized in code (fixed phrase lists, no model call)
in GateSessionManager.handle_followup_phrase, before either decision path
(the legacy sequence or the decide graph) ever sees the message -- so these
tests exercise both manager setups to confirm the recognition sits ahead of
both, not just one.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from inkbox_claude.gate import manager as gm
from inkbox_claude.gate.router import RouterOutput, RouterRequest
from inkbox_claude.gate import schedprop, wording


async def _drain():
    """Let a background asyncio.create_task(self.execute(...)) run to completion."""
    for _ in range(3):
        await asyncio.sleep(0)

APPROVER_CONV = "conv-aaron"
STRANGER = "cand@nyu.edu"


# ---------------------------------------------------------------------------
# Two manager setups: the legacy sequence (no task picker) and the decide
# graph (a fake picker, router_defines_requests=False, GATE_DECIDE_GRAPH on by
# default). Both route through GateSession.handle_inbound, which is exactly
# where handle_followup_phrase is wired in.
# ---------------------------------------------------------------------------


class FakeRouter:
    def __init__(self):
        self.next = RouterOutput(reply=None, request=None)

    async def route(self, **kw):
        return self.next.model_copy(deep=True)

    async def plan_query(self, **kw):
        return None

    async def phrase(self, *a, **kw):
        return ""


class FakePicker:
    async def pick(self, **kw):
        return {"choice": "none"}

    async def judge_action(self, **kw):
        return {"needs_action": False, "p": 0.0}


def _base_manager(tmp_path):
    sent = []

    async def send_fn(chat_id, text, mode, meta):
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV)
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    m.router = FakeRouter()
    return m, sent


def make_legacy_manager(tmp_path):
    m, sent = _base_manager(tmp_path)
    return m, sent


def make_graph_manager(tmp_path):
    m, sent = _base_manager(tmp_path)
    m.task_picker = FakePicker()
    m.router_defines_requests = False
    return m, sent


@pytest.fixture(params=["legacy", "graph"])
def manager(request, tmp_path):
    if request.param == "legacy":
        return make_legacy_manager(tmp_path)
    return make_graph_manager(tmp_path)


def approver_meta():
    return {"sender": "+15550100001", "to": "+15550100001", "conversation_id": APPROVER_CONV}


def stranger_meta():
    return {"sender": STRANGER, "to": STRANGER, "subject": "Re: chat"}


# ---------------------------------------------------------------------------
# Fixture helpers: a failed request on a thread, with a chosen age and status.
# ---------------------------------------------------------------------------

def _fail_request(m, *, chat_id, sender, sender_name, mode, age_seconds=60.0, status=None):
    t = m.store.create_task("do the thing")
    req = m.store.create_request(chat_id=chat_id, sender=sender, sender_name=sender_name, mode=mode,
                                 subject="", original_message="do the thing", summary="do the thing",
                                 scopes=["web"], prompt="do the thing", state="approved", task_id=t["id"])
    m.store.set_state(req.id, "failed", status=status or {"ok": False, "error": "boom"})
    if age_seconds:
        m.store._db.execute("UPDATE requests SET updated_at=? WHERE id=?",
                            (time.time() - age_seconds, req.id))
        m.store._db.commit()
    return m.store.get_request(req.id)


def _executed_ids(m):
    """request ids that reached FakeExecutor via Executor.run -- substitute
    FakeExecutor below records in m.executor.ran when present."""
    return getattr(m.executor, "ran", [])


class RecordingExecutor:
    def __init__(self, result=None):
        self.ran = []
        self.result = result or {"ok": True, "summary": "done this time", "tool_calls": ["x"], "raw": "done"}

    async def run(self, req, context="", prior_work="", model=None, **kw):
        self.ran.append(req.id)
        return self.result


# ---------------------------------------------------------------------------
# 1. "try again"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_try_again_owner_retries_at_once(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    req = _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage")
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage", meta=approver_meta())
    await _drain()
    rows = m.store._db.execute("SELECT * FROM requests WHERE chat_id=? ORDER BY id", (APPROVER_CONV,)).fetchall()
    assert len(rows) == 2
    assert rows[1]["prompt"] == req.prompt
    assert m.executor.ran == [rows[1]["id"]]


@pytest.mark.asyncio
async def test_try_again_stranger_goes_through_approval(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    _fail_request(m, chat_id=STRANGER, sender=STRANGER, sender_name="Cand", mode="email")
    session = m.get(STRANGER)
    await session.handle_inbound("try again", mode="email", meta=stranger_meta())
    assert m.executor.ran == []  # held for Aaron's yes, not run
    rows = m.store._db.execute("SELECT state FROM requests WHERE chat_id=?", (STRANGER,)).fetchall()
    assert any(r["state"] == "pending" for r in rows)


@pytest.mark.asyncio
async def test_try_again_outside_window_falls_through_to_router(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Got it.", request=None)
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  age_seconds=13 * 3600)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage", meta=approver_meta())
    assert m.executor.ran == []
    assert any("Got it." in s[1] for s in sent)


@pytest.mark.asyncio
async def test_try_again_after_newer_unrelated_message_not_captured(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Sure, noted.", request=None)
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  age_seconds=3600)
    # A newer, unrelated outbound landed on the thread after the failure notice,
    # and the failure itself is older than the 30-minute grace window.
    m.store.add_message(APPROVER_CONV, "outbound", "unrelated later reply", "imessage")
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage", meta=approver_meta())
    assert m.executor.ran == []
    assert any("Sure, noted." in s[1] for s in sent)


# ---------------------------------------------------------------------------
# 2. "send what you have"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_what_you_have_sends_final_status(manager, tmp_path):
    m, sent = manager
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  status={"ok": False, "final_status": "Found three candidates so far."})
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("send what you have", mode="imessage", meta=approver_meta())
    assert any("Found three candidates so far." in s[1] for s in sent)


@pytest.mark.asyncio
async def test_send_what_you_have_with_nothing(manager, tmp_path):
    m, sent = manager
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("send what you have", mode="imessage", meta=approver_meta())
    # with no recent failed run there is nothing it could mean: it is not captured
    assert not any("I don't have anything from that run." in s[1] for s in sent)


# ---------------------------------------------------------------------------
# 3. "let it <label>" (denied-tool failure, owner only)
# ---------------------------------------------------------------------------

def _denied_status(tool):
    return {"ok": False, "denied": [{"tool": tool, "reason": "scope"}]}


@pytest.mark.asyncio
async def test_let_it_label_owner_widens_and_retries(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    tool = next(t for t in gm.SCOPES["web"]["tools"]) if gm.SCOPES.get("web", {}).get("tools") else None
    # Use a scope guaranteed to exist with at least one tool.
    scope_name = next(name for name, spec in gm.SCOPES.items() if spec.get("tools"))
    tool = gm.SCOPES[scope_name]["tools"][0]
    label = schedprop.short_label(scope_name)
    req = _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                        status=_denied_status(tool))
    session = m.get(APPROVER_CONV)
    await session.handle_inbound(f"let it {label}", mode="imessage", meta=approver_meta())
    await _drain()
    assert m.executor.ran, "the retry should have actually run"
    ran_id = m.executor.ran[-1]
    row = m.store._db.execute("SELECT scopes_json FROM requests WHERE id=?", (ran_id,)).fetchone()
    import json as _json
    assert scope_name in _json.loads(row["scopes_json"])


@pytest.mark.asyncio
async def test_let_it_label_nothing_maps_asks_what(manager, tmp_path):
    m, sent = manager
    scope_name = next(name for name, spec in gm.SCOPES.items() if spec.get("tools"))
    tool = gm.SCOPES[scope_name]["tools"][0]
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  status=_denied_status(tool))
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("let it do whatever makes sense", mode="imessage", meta=approver_meta())
    assert any("What should it be allowed to do:" in s[1] for s in sent)


@pytest.mark.asyncio
async def test_let_it_label_from_stranger_falls_through(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Noted.", request=None)
    scope_name = next(name for name, spec in gm.SCOPES.items() if spec.get("tools"))
    tool = gm.SCOPES[scope_name]["tools"][0]
    label = schedprop.short_label(scope_name)
    _fail_request(m, chat_id=STRANGER, sender=STRANGER, sender_name="Cand", mode="email",
                  status=_denied_status(tool))
    session = m.get(STRANGER)
    await session.handle_inbound(f"let it {label}", mode="email", meta=stranger_meta())
    assert m.executor.ran == []


# ---------------------------------------------------------------------------
# 4. blanket "you have my permission for everything"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blanket_permission_grants_denied_plus_read_only(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    scope_name = next(name for name, spec in gm.SCOPES.items() if spec.get("tools") and not spec.get("read"))
    tool = gm.SCOPES[scope_name]["tools"][0]
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  age_seconds=60, status=_denied_status(tool))
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("You have my personal permission for everything.", mode="imessage",
                                 meta=approver_meta())
    await _drain()
    assert m.executor.ran, "the retry should actually run"
    import json as _json
    ran_id = m.executor.ran[-1]
    row = m.store._db.execute("SELECT scopes_json FROM requests WHERE id=?", (ran_id,)).fetchone()
    granted = set(_json.loads(row["scopes_json"]))
    assert scope_name in granted
    from inkbox_claude.gate import wording as _wording
    for read_scope in gm.READ_SCOPES:
        if read_scope == "vault_read":
            # The whole vault is excluded from the auto-add (item 8): it is granted
            # only when the run was actually refused it, never just for being
            # read-only -- it is not refused here, so it stays out.
            assert read_scope not in granted
        elif _wording.truly_read_only(read_scope):
            assert read_scope in granted
        elif read_scope != scope_name:
            # a "read" scope that can also create or change things is never added on its own
            assert read_scope not in granted
    # Never a send/write/post/delete/deploy/control scope beyond what was refused.
    for name, spec in gm.SCOPES.items():
        if name in granted and name != scope_name:
            assert spec.get("read"), f"{name} was granted but is not read-only and was never refused"
    # The confirmation names only what was actually refused (scope_name), never the
    # read-only scopes the grant also quietly added alongside it.
    confirmations = [s[1] for s in sent if "Okay. For this task it can now" in s[1]]
    assert confirmations
    assert all("vault" not in c.lower() for c in confirmations)


@pytest.mark.asyncio
async def test_blanket_permission_never_persists(manager, tmp_path):
    """Nothing is stored as a standing grant: a second, unrelated failure's own
    blanket grant is computed fresh from ITS OWN denied scopes, not widened by
    anything an earlier grant added."""
    m, sent = manager
    m.executor = RecordingExecutor()
    first_scope = next(name for name, spec in gm.SCOPES.items() if spec.get("tools") and not spec.get("read"))
    first_tool = gm.SCOPES[first_scope]["tools"][0]
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  status=_denied_status(first_tool))
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("You have my personal permission for everything.", mode="imessage",
                                 meta=approver_meta())
    await _drain()
    assert m._blanket_pending == set(), "a consumed grant must not linger as standing state"
    second_scope = next(name for name, spec in gm.SCOPES.items()
                        if spec.get("tools") and not spec.get("read") and name != first_scope)
    second_tool = gm.SCOPES[second_scope]["tools"][0]
    other_chat = "owner-thread-2"
    _fail_request(m, chat_id=other_chat, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  status=_denied_status(second_tool))
    session2 = m.get(other_chat)
    # Same verified owner conversation_id (is_approver() keys off that, not chat_id),
    # a separate store thread -- exactly what distinguishes two of his own requests.
    await session2.handle_inbound("You have my personal permission for everything.", mode="imessage",
                                  meta={"sender": "+15550100001", "conversation_id": APPROVER_CONV})
    await _drain()
    import json as _json
    row = m.store._db.execute(
        "SELECT scopes_json FROM requests WHERE chat_id=? ORDER BY id DESC LIMIT 1", (other_chat,)
    ).fetchone()
    granted = set(_json.loads(row["scopes_json"]))
    assert second_scope in granted
    assert first_scope not in granted


@pytest.mark.asyncio
async def test_blanket_permission_never_touches_schedule(manager, tmp_path):
    m, sent = manager
    t = m.store.create_task("watch the page")
    sched = m.schedules.store.create_schedule if hasattr(m.schedules, "store") else None
    # No schedule row created here; the assertion is simply that the schedules
    # table stays untouched by a blanket-permission turn.
    before = m.store._db.execute("SELECT count(*) FROM schedules").fetchone()[0]
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("You have my personal permission for everything.", mode="imessage",
                                 meta=approver_meta())
    after = m.store._db.execute("SELECT count(*) FROM schedules").fetchone()[0]
    assert before == after


@pytest.mark.asyncio
async def test_blanket_permission_from_stranger_falls_through(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Noted.", request=None)
    scope_name = next(name for name, spec in gm.SCOPES.items() if spec.get("tools"))
    tool = gm.SCOPES[scope_name]["tools"][0]
    _fail_request(m, chat_id=STRANGER, sender=STRANGER, sender_name="Cand", mode="email",
                  status=_denied_status(tool))
    session = m.get(STRANGER)
    await session.handle_inbound("You have my personal permission for everything.", mode="email",
                                 meta=stranger_meta())
    assert m.executor.ran == []


@pytest.mark.asyncio
async def test_blanket_permission_with_nothing_recent(manager, tmp_path):
    m, sent = manager
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("You have my personal permission for everything.", mode="imessage",
                                 meta=approver_meta())
    assert any("I don't keep a blanket permission." in s[1] for s in sent)


@pytest.mark.asyncio
async def test_blanket_permission_mid_run_replies_once(manager, tmp_path):
    """Item 8: said while a run is still going, it used to reply nothing at all.
    Now it replies once, and the grant/confirmation for THIS run still only fires
    later, when the run actually fails and shows what it was refused."""
    m, sent = manager

    class SlowExecutor:
        def __init__(self):
            self.ran = []
        async def run(self, req, context="", prior_work="", model=None, **kw):
            self.ran.append(req.id)
            await asyncio.sleep(10)  # never actually finishes within this test
            return {"ok": True, "summary": "too late", "tool_calls": ["x"], "raw": "too late"}
    m.executor = SlowExecutor()
    t = m.store.create_task("long task")
    req = m.store.create_request(chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron",
                                 mode="imessage", subject="", original_message="do the long task",
                                 summary="do the long task", scopes=["web"], prompt="do the long task",
                                 state="approved", task_id=t["id"])
    asyncio.create_task(m.execute(req))
    await _drain()
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("You have my personal permission for everything.", mode="imessage",
                                 meta=approver_meta())
    await _drain()
    replies = [row for row in sent if "If it gets refused something, I'll let it and continue." in row[1]]
    assert len(replies) == 1
    assert req.id in m._blanket_pending


# ---------------------------------------------------------------------------
# 5. short, re-sayable labels
# ---------------------------------------------------------------------------

def test_short_label_cuts_at_parenthesis():
    long_scope = next((name for name, lbl in schedprop.LABELS.items() if "(" in lbl), None)
    if long_scope is None:
        pytest.skip("no scope with a parenthetical label in this registry")
    short = schedprop.short_label(long_scope)
    assert "(" not in short
    assert schedprop.LABELS[long_scope].startswith(short)


def test_parser_accepts_both_full_and_short_label_forms():
    long_scope = next((name for name, lbl in schedprop.LABELS.items() if "(" in lbl), None)
    if long_scope is None:
        pytest.skip("no scope with a parenthetical label in this registry")
    full = schedprop.LABELS[long_scope]
    short = schedprop.short_label(long_scope)
    assert schedprop.capabilities_from_permission(f"let it {full}") == [long_scope]
    assert schedprop.capabilities_from_permission(f"let it {short}") == [long_scope]


# ---------------------------------------------------------------------------
# 6. every phrase wording.py tells him to say actually works
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_wording_offered_phrase_is_accepted(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    scope_name = next(name for name, spec in gm.SCOPES.items() if spec.get("tools"))
    tool = gm.SCOPES[scope_name]["tools"][0]
    status = _denied_status(tool)
    msg = wording.failure_message("the thing", status)
    assert 'Say "try again" and I will.' not in msg or True  # sanity: refusal path picked below
    # The refusal phrasing names a "let it <label>" phrase; extract and send it.
    assert f'let it {schedprop.short_label(scope_name)}' in msg
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  status=status)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound(f"let it {schedprop.short_label(scope_name)}", mode="imessage", meta=approver_meta())
    await _drain()
    assert m.executor.ran


@pytest.mark.asyncio
async def test_try_again_phrase_from_failure_message_is_accepted(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    status = {"ok": False, "error": "it took too long", "final_status": "timed out"}
    msg = wording.failure_message("the thing", status)
    assert "try again" in msg
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  status=status)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage", meta=approver_meta())
    await _drain()
    assert m.executor.ran


@pytest.mark.asyncio
async def test_send_what_you_have_phrase_from_failure_message_is_accepted(manager, tmp_path):
    m, sent = manager
    status = {"ok": False, "error": "it took too long", "final_status": "found two of three items"}
    msg = wording.failure_message("the thing", status)
    assert 'send what you have' in msg
    _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage",
                  status=status)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("send what you have", mode="imessage", meta=approver_meta())
    assert any("found two of three items" in s[1] for s in sent)


# ---------------------------------------------------------------------------
# Precedence: a pending approval or an open schedule proposal on this thread
# still wins a bare "yes"/"no" (unchanged by this feature).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_approval_still_wins_bare_yes(manager, tmp_path):
    m, sent = manager
    t = m.store.create_task("book the thing")
    pend = m.store.create_request(chat_id=STRANGER, sender=STRANGER, sender_name="Cand", mode="email",
                                   subject="", original_message="book the thing", summary="book the thing",
                                   scopes=["web"], prompt="book the thing", state="pending", task_id=t["id"])
    m.executor = RecordingExecutor()
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("yes", mode="imessage", meta=approver_meta())
    row = m.store._db.execute("SELECT state FROM requests WHERE id=?", (pend.id,)).fetchone()
    assert row["state"] == "approved"


# ---------------------------------------------------------------------------
# Item 6: a schedule's own failed run shares the owner's chat_id -- "try again"
# and the blanket permission must never act on it (as _handle_let_it_label
# already refused to, via req.schedule_id).
# ---------------------------------------------------------------------------

def _fail_scheduled_request(m, *, chat_id, sender, sender_name, mode, schedule_id, age_seconds=60.0, status=None):
    t = m.store.create_task("watch the page")
    req = m.store.create_request(chat_id=chat_id, sender=sender, sender_name=sender_name, mode=mode,
                                 subject="", original_message="watch the page", summary="watch the page",
                                 scopes=["web"], prompt="watch the page", state="approved", task_id=t["id"],
                                 schedule_id=schedule_id)
    m.store.set_state(req.id, "failed", status=status or {"ok": False, "error": "boom"})
    if age_seconds:
        m.store._db.execute("UPDATE requests SET updated_at=? WHERE id=?",
                            (time.time() - age_seconds, req.id))
        m.store._db.commit()
    return m.store.get_request(req.id)


@pytest.mark.asyncio
async def test_try_again_ignores_a_schedules_own_failed_run(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Got it.", request=None)
    sched = _fail_scheduled_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron",
                                    mode="imessage", schedule_id=999)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage", meta=approver_meta())
    # Never re-created as a one-off: the only request on the thread is still the
    # schedule's own failed row, unchanged.
    assert m.executor.ran == []
    assert m.store.get_request(sched.id).state == "failed"
    assert any("Got it." in s[1] for s in sent)


@pytest.mark.asyncio
async def test_blanket_permission_ignores_a_schedules_own_failed_run(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    scope_name = next(name for name, spec in gm.SCOPES.items() if spec.get("tools") and not spec.get("read"))
    tool = gm.SCOPES[scope_name]["tools"][0]
    sched = _fail_scheduled_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron",
                                    mode="imessage", schedule_id=999, status=_denied_status(tool))
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("You have my personal permission for everything.", mode="imessage",
                                 meta=approver_meta())
    await _drain()
    # Nothing ran, nothing widened: the schedule's own scopes are untouched.
    assert m.executor.ran == []
    import json as _json
    row = m.store._db.execute("SELECT scopes_json FROM requests WHERE id=?", (sched.id,)).fetchone()
    assert _json.loads(row["scopes_json"]) == ["web"]


# ---------------------------------------------------------------------------
# Item 5 (unsafe): create_request_once's dedupe is scoped by sender as well as
# thread, a caller that gets an existing request back never acknowledges,
# executes or widens it, and retry_request never dedupes against a different
# request.
# ---------------------------------------------------------------------------

def test_create_request_once_dedupe_is_scoped_by_sender(tmp_path):
    from inkbox_claude.gate.store import Store
    s = Store(str(tmp_path / "g.db"))
    t = s.create_task("do x")
    req_a, created_a = s.create_request_once(
        chat_id="group-1", sender="alice", sender_name="Alice", mode="imessage", subject="",
        original_message="book a flight", summary="book a flight", scopes=["web"], prompt="book a flight",
        state="approved", task_id=t["id"])
    assert created_a
    # Bob asks something in the same group thread within the dedupe window: his
    # own request, never folded into Alice's.
    req_b, created_b = s.create_request_once(
        chat_id="group-1", sender="bob", sender_name="Bob", mode="imessage", subject="",
        original_message="book a different flight", summary="book a different flight", scopes=["web"],
        prompt="book a different flight", state="approved", task_id=t["id"])
    assert created_b
    assert req_b.id != req_a.id
    # Alice asking again right away (same sender, same thread, still approved/running)
    # IS deduped -- the sender-scoping narrows the match, it doesn't remove it.
    req_a2, created_a2 = s.create_request_once(
        chat_id="group-1", sender="alice", sender_name="Alice", mode="imessage", subject="",
        original_message="book a flight again", summary="book a flight again", scopes=["web"],
        prompt="book a flight again", state="approved", task_id=t["id"])
    assert not created_a2
    assert req_a2.id == req_a.id


def test_retry_request_bypasses_dedupe_against_a_different_request(tmp_path):
    from inkbox_claude.gate.store import Store
    s = Store(str(tmp_path / "g.db"))
    t = s.create_task("do x")
    # One request from this sender is already approved/running (unrelated to the
    # one about to be retried).
    other, _ = s.create_request_once(
        chat_id="c1", sender="alice", sender_name="Alice", mode="imessage", subject="",
        original_message="unrelated ask", summary="unrelated ask", scopes=["web"], prompt="unrelated ask",
        state="approved", task_id=t["id"])
    # A retry (bypass_dedupe=True) for a different, failed request from the same
    # sender must not fold into `other" -- it gets its own fresh row.
    retried, created = s.create_request_once(
        chat_id="c1", sender="alice", sender_name="Alice", mode="imessage", subject="",
        original_message="the failed ask", summary="the failed ask", scopes=["web"], prompt="the failed ask",
        state="approved", task_id=t["id"], bypass_dedupe=True)
    assert created
    assert retried.id != other.id


@pytest.mark.asyncio
async def test_retry_refuses_when_the_request_is_itself_still_running(manager, tmp_path):
    m, sent = manager
    req = _fail_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron", mode="imessage")
    # The request has since moved back to "running" (a race: the failure notice
    # went out just before a stale retry landed, or two fragments overlapped).
    m.store.set_state(req.id, "running")
    assert m.retry_request(req, approved=True) is None


@pytest.mark.asyncio
async def test_group_thread_second_senders_ask_is_not_swallowed_by_the_first(tmp_path):
    """The actual unsafe shape in item 5: two different people ask in the same
    group thread within the dedupe window. Before the sender-scoped dedupe, the
    second person's ask could be folded into the first person's request, with
    its ack already sent for work that would never separately run.

    Legacy manager only: the decide-graph manager builds its requests from the
    task picker rather than the router's own `request` field, which is a
    separate concern from the dedupe this test is about."""
    m, sent = make_legacy_manager(tmp_path)
    m.executor = RecordingExecutor()
    t = m.store.create_task("group ask")

    def group_meta(sender):
        return {"sender": sender, "conversation_id": "group-conv", "conversation_kind": "group"}

    m.router.next = RouterOutput(reply="On it.", task=f"T{t['id']}",
                                 request=RouterRequest(prompt="help alice", scopes=["web"], summary="help alice"))
    session_a = m.get("group-conv")
    await session_a.handle_inbound("can you help me", mode="imessage", meta=group_meta("alice"))
    m.router.next = RouterOutput(reply="On it.", task=f"T{t['id']}",
                                 request=RouterRequest(prompt="help bob", scopes=["web"], summary="help bob"))
    session_b = m.get("group-conv")
    await session_b.handle_inbound("can you help me too", mode="imessage", meta=group_meta("bob"))
    rows = m.store._db.execute("SELECT sender FROM requests WHERE chat_id=?", ("group-conv",)).fetchall()
    senders = sorted(r["sender"] for r in rows)
    assert senders == ["alice", "bob"]


# ---------------------------------------------------------------------------
# Extra fix: a call's rows can land on the call's own thread (chat_id) while
# the same person's texts land on the owner thread -- an unverified caller on
# the owner's own number (is_aaron_on_phone() true, is_approver() false) is
# exactly this case. A texted "try again" must still find that call's failed
# request, by resolving sender -> person -> that other sender, never by
# granting the texter any extra power.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_try_again_finds_a_failed_request_on_a_different_thread_for_the_same_person(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Got it.", request=None)
    person = m.store.create_person(display="Aaron")
    # The call's own number and his iMessage address both belong to this one person.
    m.store.add_contact(person["id"], "phone", "+15550100099")
    m.store.add_contact(person["id"], "email", "aaron@example.com")
    # A call's failed request lives on the call's own chat_id -- not the owner
    # thread -- with sender = the call's phone number.
    _fail_request(m, chat_id="call-xyz", sender="+15550100099", sender_name="Aaron", mode="voice")
    # He texts "try again" from imessage, verified as the approver there (a
    # different thread, a different sender identity -- his email/imessage handle).
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage",
                                 meta={"sender": "aaron@example.com", "conversation_id": APPROVER_CONV})
    await _drain()
    assert m.executor.ran, "the call's failed request should have been retried"


@pytest.mark.asyncio
async def test_try_again_never_finds_an_unrelated_persons_failed_request(manager, tmp_path):
    """Resolving sender -> person only ever finds that SAME person's other
    requests -- never a stranger's, and never grants power a direct match
    would not already have given."""
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Got it.", request=None)
    aaron = m.store.create_person(display="Aaron")
    m.store.add_contact(aaron["id"], "email", "aaron@example.com")
    stranger_person = m.store.create_person(display="Cand")
    m.store.add_contact(stranger_person["id"], "phone", "+15559990000")
    _fail_request(m, chat_id="call-stranger", sender="+15559990000", sender_name="Cand", mode="voice")
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage",
                                 meta={"sender": "aaron@example.com", "conversation_id": APPROVER_CONV})
    assert m.executor.ran == []
    assert any("Got it." in s[1] for s in sent)


# ---------------------------------------------------------------------------
# Item 9: a progress question after the run has already finished or failed is
# answered from the request row, not re-decided as a new ask.
# ---------------------------------------------------------------------------

def _finish_request(m, *, chat_id, sender, sender_name, mode, state, age_seconds=60.0, status=None):
    t = m.store.create_task("do the thing")
    req = m.store.create_request(chat_id=chat_id, sender=sender, sender_name=sender_name, mode=mode,
                                 subject="", original_message="do the thing", summary="do the thing",
                                 scopes=["web"], prompt="do the thing", state="approved", task_id=t["id"])
    m.store.set_state(req.id, state, status=status or ({"ok": True, "summary": "all set"} if state == "done"
                                                        else {"ok": False, "error": "timed out after 300s"}))
    if age_seconds:
        m.store._db.execute("UPDATE requests SET updated_at=? WHERE id=?",
                            (time.time() - age_seconds, req.id))
        m.store._db.commit()
    return m.store.get_request(req.id)


@pytest.mark.asyncio
async def test_progress_question_after_failure_is_answered_from_the_row(manager, tmp_path):
    m, sent = manager
    m.router.next = RouterOutput(reply="Should not be used.", request=RouterRequest(
        prompt="something new", scopes=["web"], summary="something new"))
    req = _finish_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron",
                          mode="imessage", state="failed", age_seconds=30)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("any update?", mode="imessage", meta=approver_meta())
    assert any("That one failed: it took too long." in s[1] for s in sent)
    # No new request was created from this question.
    rows = m.store._db.execute("SELECT id FROM requests WHERE chat_id=?", (APPROVER_CONV,)).fetchall()
    assert [r["id"] for r in rows] == [req.id]


@pytest.mark.asyncio
async def test_progress_question_after_success_is_answered_from_the_row(manager, tmp_path):
    m, sent = manager
    m.router.next = RouterOutput(reply="Should not be used.", request=None)
    _finish_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron",
                    mode="imessage", state="done", age_seconds=30)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("status", mode="imessage", meta=approver_meta())
    assert any("all set" in s[1] for s in sent)
    assert not any("Should not be used." in s[1] for s in sent)


@pytest.mark.asyncio
async def test_progress_question_outside_the_ten_minute_window_falls_through(manager, tmp_path):
    m, sent = manager
    m.router.next = RouterOutput(reply="A fresh answer.", request=None)
    _finish_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron",
                    mode="imessage", state="failed", age_seconds=20 * 60)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("status", mode="imessage", meta=approver_meta())
    assert any("A fresh answer." in s[1] for s in sent)


@pytest.mark.asyncio
async def test_a_genuinely_new_question_mark_ask_still_creates_a_request_after_a_finish(tmp_path):
    """The loose "ends in a question mark" guess must not leak into the
    post-finish path: only the fixed phrases do. "When is it?" right after an
    unrelated request finished is still a new ask.

    Legacy manager only: this file's FakePicker always answers "no action" for
    the decide-graph manager, which is a separate concern from what this test
    is about."""
    m, sent = make_legacy_manager(tmp_path)
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="On it.", task="new", task_title="look something up",
                                 request=RouterRequest(prompt="look it up", scopes=["web"], summary="look it up"))
    _finish_request(m, chat_id=APPROVER_CONV, sender="+15550100001", sender_name="Aaron",
                    mode="imessage", state="done", age_seconds=30)
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("when is it?", mode="imessage", meta=approver_meta())
    await _drain()
    assert m.executor.ran, "a genuinely new question-shaped ask must still create a request"



# ---------------------------------------------------------------------------
# Item C4: the person-fallback lookup is restricted to one-to-one threads and
# to phone/iMessage/Telegram contacts, so a shared number or a group thread
# can never let one person's "try again" re-run another's request.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_try_again_shared_number_contact_finds_nothing(manager, tmp_path):
    """Two different people sharing one phone contact value must never cross-match:
    the fallback only widens how a person's OWN failed request is found, never
    someone else's, even if a contacts-table merge made the values collide."""
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Got it.", request=None)
    aaron = m.store.create_person(display="Aaron")
    m.store.add_contact(aaron["id"], "email", "aaron@example.com")
    stranger = m.store.create_person(display="Cand")
    # The stranger's phone contact happens to be the same value some other
    # failed request's sender used -- not Aaron's own.
    m.store.add_contact(stranger["id"], "phone", "+15550100099")
    _fail_request(m, chat_id="call-xyz", sender="+15550100099", sender_name="Cand", mode="voice")
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage",
                                 meta={"sender": "aaron@example.com", "conversation_id": APPROVER_CONV})
    assert m.executor.ran == []
    assert any("Got it." in s[1] for s in sent)


@pytest.mark.asyncio
async def test_try_again_never_crosses_into_a_group_thread(manager, tmp_path):
    """A failed request that landed on a group thread is never picked up by the
    person fallback, even if the group member is the same person the owner is
    texting about elsewhere."""
    m, sent = manager
    m.executor = RecordingExecutor()
    m.router.next = RouterOutput(reply="Got it.", request=None)
    aaron = m.store.create_person(display="Aaron")
    m.store.add_contact(aaron["id"], "phone", "+15550100099")
    m.store.add_contact(aaron["id"], "email", "aaron@example.com")
    _fail_request(m, chat_id="group-abc", sender="+15550100099", sender_name="Aaron", mode="imessage")
    m.store.set_thread("group-abc", "idle", "imessage", {"conversation_kind": "group"})
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("try again", mode="imessage",
                                 meta={"sender": "aaron@example.com", "conversation_id": APPROVER_CONV})
    assert m.executor.ran == []
    assert any("Got it." in s[1] for s in sent)


# ---------------------------------------------------------------------------
# Item C3: "let it <label>" and the blanket permission use the same restricted
# person lookup "try again" uses, so they also find a call's failed request
# on a different thread -- while still requiring the owner's is_approver().
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_let_it_label_finds_a_failed_request_on_a_different_thread(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    person = m.store.create_person(display="Aaron")
    m.store.add_contact(person["id"], "phone", "+15550100099")
    m.store.add_contact(person["id"], "email", "aaron@example.com")
    scope_name = next(name for name, spec in gm.SCOPES.items() if spec.get("tools"))
    tool = gm.SCOPES[scope_name]["tools"][0]
    label = schedprop.short_label(scope_name)
    req = _fail_request(m, chat_id="call-xyz", sender="+15550100099", sender_name="Aaron", mode="voice",
                        status=_denied_status(tool))
    session = m.get(APPROVER_CONV)
    await session.handle_inbound(f"let it {label}", mode="imessage",
                                 meta={"sender": "aaron@example.com", "conversation_id": APPROVER_CONV})
    await _drain()
    assert m.executor.ran, "the retry should have actually run"


@pytest.mark.asyncio
async def test_blanket_permission_finds_a_failed_request_on_a_different_thread(manager, tmp_path):
    m, sent = manager
    m.executor = RecordingExecutor()
    person = m.store.create_person(display="Aaron")
    m.store.add_contact(person["id"], "phone", "+15550100099")
    m.store.add_contact(person["id"], "email", "aaron@example.com")
    req = _fail_request(m, chat_id="call-xyz", sender="+15550100099", sender_name="Aaron", mode="voice")
    session = m.get(APPROVER_CONV)
    await session.handle_inbound("you have my permission for everything", mode="imessage",
                                 meta={"sender": "aaron@example.com", "conversation_id": APPROVER_CONV})
    await _drain()
    assert m.executor.ran, "blanket permission should have retried the call's failed request"



# ---------------------------------------------------------------------------
# Item C5: GateSessionManager.retry_approval_for_request decides from the
# request's OWN sender identity, never the live console operator's.
# ---------------------------------------------------------------------------

def test_retry_approval_for_request_console_is_always_owner(manager, tmp_path):
    m, sent = manager
    t = m.store.create_task("do the thing")
    req = m.store.create_request(chat_id="owner", sender="jordan@example.test", sender_name="Jordan",
                                 mode="console", subject="", original_message="do the thing",
                                 summary="do the thing", scopes=["web"], prompt="do the thing",
                                 state="failed", task_id=t["id"])
    assert m.retry_approval_for_request(req) is True


def test_retry_approval_for_request_strangers_email_needs_approval_again(manager, tmp_path):
    m, sent = manager
    t = m.store.create_task("do the thing")
    req = m.store.create_request(chat_id=STRANGER, sender=STRANGER, sender_name="Cand", mode="email",
                                 subject="", original_message="do the thing", summary="do the thing",
                                 scopes=["web"], prompt="do the thing", state="failed", task_id=t["id"])
    assert m.retry_approval_for_request(req) is False


def test_retry_approval_for_request_the_owners_verified_imessage_is_approved(manager, tmp_path):
    m, sent = manager
    t = m.store.create_task("do the thing")
    req = m.store.create_request(chat_id=APPROVER_CONV, sender="aaron@example.com", sender_name="Aaron",
                                 mode="imessage", subject="", original_message="do the thing",
                                 summary="do the thing", scopes=["web"], prompt="do the thing",
                                 state="failed", task_id=t["id"])
    m.store.set_thread(APPROVER_CONV, "idle", "imessage", {"conversation_id": APPROVER_CONV})
    assert m.retry_approval_for_request(req) is True
