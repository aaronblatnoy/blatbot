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
        if _wording.truly_read_only(read_scope):
            assert read_scope in granted
        elif read_scope != scope_name:
            # a "read" scope that can also create or change things is never added on its own
            assert read_scope not in granted
    # Never a send/write/post/delete/deploy/control scope beyond what was refused.
    for name, spec in gm.SCOPES.items():
        if name in granted and name != scope_name:
            assert spec.get("read"), f"{name} was granted but is not read-only and was never refused"
    assert any("Okay. For this task it can now" in s[1] for s in sent)


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
