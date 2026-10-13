"""Tests for GET /console/api/overview (api.overview): the Home page's single
source of stuck / running / done-today / coming-up rows, built with made-up
rows against the real store (not faked)."""

from __future__ import annotations

import time

import json as _json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from inkbox_claude.console import api
from inkbox_claude.console import routes as console_routes
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate.store import Store


def make_store(tmp_path):
    return Store(str(tmp_path / "overview.db"))


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.schedules = sched_mod.ScheduleService(self)
        self.notices = []

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule.id, text))

    async def execute(self, req):
        self.store.set_state(req.id, "running")


class FakeGateway:
    def __init__(self, manager):
        self.sessions = manager


def _make_schedule(store, *, title="Team total watch", prompt="check the team page",
                   scopes=None, cadence_kind="every", cadence=None, state="active",
                   report_every_time=False, timezone="America/New_York", **extra):
    task_id = store.create_task(title)["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title=title, prompt=prompt, kind=cadence_kind,
        scopes=list(scopes or ["web"]), timezone=timezone, report_mode="changed",
        cadence_kind=cadence_kind, cadence=cadence or {"minutes": 240},
        report_every_time=report_every_time, origin_mode="console", origin_chat_id="owner",
    )
    if state != "proposed":
        schedule = store.update_schedule(schedule.id, state=state, next_run=time.time() + 3600, **extra)
    return schedule


def _add_run(store, schedule_id, *, outcome="complete", result_line="done",
             final_status="", denied=None, note="", changed=True, cost_usd=0.01,
             finished_at=None):
    started = (finished_at or time.time()) - 60
    run_id = store.create_schedule_run(schedule_id, None, started)
    store.finish_schedule_run(
        run_id, finished_at=finished_at or (started + 55), complete=(outcome == "complete"),
        outcome=outcome, result_line=result_line, final_status=final_status or result_line,
        denied=denied or [], note=note, changed=changed, cost_usd=cost_usd,
    )
    return run_id


def _make_request(store, *, title_task=None, summary="Approve the donor mailing list.",
                   original_message="Can you approve the donor mailing list?",
                   state="pending", sender_name="Jordan", schedule_id=None,
                   status=None, chat_id="owner"):
    task_id = title_task or store.create_task(summary)["id"]
    request = store.create_request(
        chat_id=chat_id, sender="jordan@example.test", sender_name=sender_name, mode="console",
        subject="", original_message=original_message, summary=summary, scopes=[], prompt=summary,
        state=state, task_id=task_id, schedule_id=schedule_id,
    )
    if status is not None:
        store._db.execute("UPDATE requests SET status_json=? WHERE id=?",
                          (_json.dumps(status), request.id))
        store._db.commit()
    return store.get_request(request.id)


def _backdate(store, request_id, *, created_at=None, updated_at=None):
    created_at = created_at if created_at is not None else time.time()
    updated_at = updated_at if updated_at is not None else created_at
    store._db.execute(
        "UPDATE requests SET created_at=?, updated_at=? WHERE id=?",
        (created_at, updated_at, request_id),
    )
    store._db.commit()


# ---------------------------------------------------------------------------
# "Nothing is stuck" only when all three stuck sources are empty.
# ---------------------------------------------------------------------------

def test_nothing_stuck_when_no_schedule_request_or_proposal_is_waiting(tmp_path):
    store = make_store(tmp_path)
    _make_schedule(store, title="Fine schedule", state="active")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert result["stuck"] == []
    assert result["counts"]["stuck"] == 0


def test_a_pending_request_alone_makes_it_stuck(tmp_path):
    store = make_store(tmp_path)
    _make_request(store, summary="Approve the donor mailing list.")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert len(result["stuck"]) == 1
    row = result["stuck"][0]
    assert row["kind"] == "approval"
    assert "donor mailing list" in row["sentence"]
    actions = {a["action"] for a in row["actions"]}
    assert actions == {"approve", "decline"}
    assert row["actions"][0]["target"]["endpoint"] == "requests/decide"


# ---------------------------------------------------------------------------
# Each kind of stuck item, with the right sentence and actions.
# ---------------------------------------------------------------------------

def test_paused_schedule_with_a_reason_is_stuck_with_resume_and_stop(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Weekly backup check", cadence_kind="weekly",
                              cadence={"day": "mon", "time": "06:00"}, state="paused",
                              last_outcome="it is not allowed to look up and edit contacts")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    rows = [r for r in result["stuck"] if r["id"] == schedule.id]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "schedule"
    assert row["sentence"] == "Weekly backup check: it is not allowed to look up and edit contacts."
    actions = {a["action"] for a in row["actions"]}
    assert "resume" in actions and "stop" in actions


def test_paused_schedule_with_no_cadence_gets_edit_timing_action(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Half-built schedule", state="proposed")
    store.update_schedule(schedule.id, state="paused", cadence_kind="", cadence={},
                          last_outcome="needs a cadence")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == schedule.id)
    actions = {a["action"] for a in row["actions"]}
    assert "edit_timing" in actions
    assert "resume" not in actions  # nothing to resume into until it has a timing


def test_active_schedule_whose_last_run_did_not_complete_is_stuck(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Donor tracker sync",
                              prompt="Check the donor page and use inkbox_lookup_contact to update it.",
                              state="active")
    _add_run(store, schedule.id, outcome="incomplete", result_line="The total did change (+$200).",
             denied=[{"tool": "mcp__inkbox__inkbox_lookup_contact", "reason": "scope"}])

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == schedule.id)
    assert row["kind"] == "schedule"
    # The run's own RESULT sentence is normalized (schedules._normalize_reason),
    # not shown verbatim/capitalized as the run wrote it.
    assert "the total did change" in row["sentence"]


def test_schedule_proposal_waiting_for_yes_is_stuck_with_approve_and_reject(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Monday brief", state="proposed")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == schedule.id)
    assert row["kind"] == "approval"
    actions = {a["action"] for a in row["actions"]}
    assert actions == {"approve", "reject"}
    assert row["actions"][0]["target"]["endpoint"] == "schedules/approve"


def test_a_failed_request_in_the_last_day_is_stuck(tmp_path):
    store = make_store(tmp_path)
    request = _make_request(store, summary="Wire the vendor invoice.", state="failed",
                            status={"error": "the vendor portal timed out"})
    _backdate(store, request.id, created_at=time.time() - 3600, updated_at=time.time() - 3600)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["kind"] == "request" and r["id"] == request.id)
    # Never the raw error string -- wording.failure_in_words' plain clause for a
    # timeout-shaped error.
    assert "vendor portal timed out" not in row["sentence"]
    assert row["sentence"] == "Wire the vendor invoice.: it took too long."
    assert row["actions"] == []  # no retry endpoint exists


def test_failed_request_older_than_a_day_is_not_stuck(tmp_path):
    store = make_store(tmp_path)
    request = _make_request(store, summary="Old failed thing.", state="failed",
                            status={"error": "old failure"})
    _backdate(store, request.id, created_at=time.time() - 90000, updated_at=time.time() - 90000)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert request.id not in [r["id"] for r in result["stuck"] if r["kind"] == "request"]


# ---------------------------------------------------------------------------
# A failed request superseded by a later success on the same task is not stuck.
# ---------------------------------------------------------------------------

def test_failed_request_superseded_by_later_success_same_task_is_not_stuck(tmp_path):
    store = make_store(tmp_path)
    task_id = store.create_task("Donor list approval")["id"]
    failed = _make_request(store, title_task=task_id, summary="Approve the donor list.",
                           state="failed", status={"error": "it ran out of time"})
    _backdate(store, failed.id, created_at=time.time() - 1000, updated_at=time.time() - 1000)
    retried = _make_request(store, title_task=task_id, summary="Approve the donor list.", state="done")
    _backdate(store, retried.id, created_at=time.time() - 500, updated_at=time.time() - 500)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert failed.id not in [r["id"] for r in result["stuck"] if r["kind"] == "request"]


def test_failed_request_not_superseded_without_a_later_success_is_still_stuck(tmp_path):
    store = make_store(tmp_path)
    task_id = store.create_task("Donor list approval")["id"]
    failed = _make_request(store, title_task=task_id, summary="Approve the donor list.",
                           state="failed", status={"error": "it ran out of time"})
    _backdate(store, failed.id, created_at=time.time() - 1000, updated_at=time.time() - 1000)
    another_failure = _make_request(store, title_task=task_id, summary="Approve the donor list.",
                                    state="failed", status={"error": "it ran out of time again"})
    _backdate(store, another_failure.id, created_at=time.time() - 500, updated_at=time.time() - 500)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    # Two failures on the same task, neither followed by a success, are the same
    # ask -- collapsed into one row, reported from the latest attempt.
    request_rows = [r for r in result["stuck"] if r["kind"] == "request"]
    assert len(request_rows) == 1
    assert request_rows[0]["id"] == another_failure.id
    assert failed.id not in [r["id"] for r in request_rows]


def test_failed_requests_on_different_tasks_and_threads_stay_separate(tmp_path):
    store = make_store(tmp_path)
    first = _make_request(store, summary="Wire the vendor invoice.", state="failed",
                          status={"error": "timed out"})
    store._db.execute("UPDATE requests SET chat_id=? WHERE id=?", ("thread-a", first.id))
    store._db.commit()
    _backdate(store, first.id, created_at=time.time() - 3600, updated_at=time.time() - 3600)
    second = _make_request(store, summary="Draft the weekly memo.", state="failed",
                           status={"error": "timed out"})
    store._db.execute("UPDATE requests SET chat_id=? WHERE id=?", ("thread-b", second.id))
    store._db.commit()
    _backdate(store, second.id, created_at=time.time() - 1800, updated_at=time.time() - 1800)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    ids = {r["id"] for r in result["stuck"] if r["kind"] == "request"}
    assert ids == {first.id, second.id}


# ---------------------------------------------------------------------------
# A hand-off prompt is never shown; the owner's original message stands in.
# ---------------------------------------------------------------------------

def _realistic_voice_transcript(phone="+15550123456", ask="sign up for the pottery course"):
    """The exact shape live.py's build_delegation_query hands off: a preamble
    naming the caller, an "Earlier in the call" section, then "What was just
    said (act on this):" with alternating blatbot:/caller: lines -- made-up
    phone number and ask, same section headings and line prefixes as the
    real thing."""
    preamble = (
        f"Live phone call with {phone}. Blatbot's voice side has handed this off because the caller needs "
        "something done or looked up. The lines below are speech-recognition transcript, so expect misheard "
        "words, unfinished phrases and later corrections; trust the latest version of what they said."
    )
    earlier = "blatbot: Hi, this is Blatbot, Aaron's assistant.\ncaller: Hey, I had a question."
    fresh = (
        f"blatbot: What can I help with?\ncaller: I want to {ask}.\n"
        "blatbot: Let me check on that for you."
    )
    return "\n".join([
        preamble, "", "Earlier in the call (already handled, for context only):", earlier, "",
        "What was just said (act on this):", fresh, "",
        "Work out what the caller is asking for right now and handle it.",
    ])


def test_handoff_prompt_summary_is_replaced_by_the_caller_own_words(tmp_path):
    store = make_store(tmp_path)
    transcript = _realistic_voice_transcript()
    cut_summary = transcript[:90]  # reproduces the historical 90-character-cut bug exactly
    request = _make_request(store, summary=cut_summary, original_message=transcript, state="pending")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == request.id)
    assert "Live phone call with" not in row["sentence"]
    assert "+15550123456" not in row["sentence"]
    assert "pottery course" in row["sentence"]
    assert "blatbot:" not in row["sentence"].lower()
    assert "caller:" not in row["sentence"].lower()


def test_sentences_are_whole_never_cut(tmp_path):
    store = make_store(tmp_path)
    long_summary = "Approve this: " + ("x" * 400)
    request = _make_request(store, summary=long_summary, original_message=long_summary, state="pending")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    row = next(r for r in result["stuck"] if r["id"] == request.id)
    assert long_summary in row["sentence"]


# ---------------------------------------------------------------------------
# Realistic shapes from the real database (coordinator's merged-backend pass):
# a 90-char-cut hand-off summary, the JSON buffer-size error, and a schedule
# run with a first-person RESULT line. Every sentence everywhere is scanned
# for the exact forbidden substrings the coordinator named.
# ---------------------------------------------------------------------------

_FORBIDDEN_SUBSTRINGS = ("Live phone call", "blatbot:", "caller:", "JSON", "bytes", "\n")
_PHONE_RE_FOR_TEST = __import__("re").compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")


def _assert_no_forbidden_text(sentence):
    for bad in _FORBIDDEN_SUBSTRINGS:
        assert bad not in sentence, f"{bad!r} found in sentence: {sentence!r}"
    assert not _PHONE_RE_FOR_TEST.search(sentence), f"a phone number found in sentence: {sentence!r}"


def test_realistic_voice_hand_off_request_produces_a_clean_sentence_everywhere(tmp_path):
    store = make_store(tmp_path)
    transcript = _realistic_voice_transcript(phone="+15557654321", ask="reschedule the orientation session")
    cut_summary = transcript[:90]

    # One copy pending (Stuck/approval), one running, one failed with the exact
    # JSON buffer-size error the coordinator quoted, one done today -- each on
    # its own thread, since the store's own dedupe (same thread, approved/
    # running, within its window) would otherwise fold these made-up fixture
    # rows into one another.
    pending = _make_request(store, summary=cut_summary, original_message=transcript, state="pending",
                            chat_id="voice-1")
    running = _make_request(store, summary=cut_summary, original_message=transcript, state="running",
                            chat_id="voice-2")
    failed = _make_request(store, summary=cut_summary, original_message=transcript, state="failed",
                           chat_id="voice-3",
                           status={"error": "Failed to decode JSON: JSON message exceeded maximum "
                                            "buffer size of 1048576 bytes."})
    done = _make_request(store, summary=cut_summary, original_message=transcript, state="done",
                         chat_id="voice-4",
                         status={"summary": "Rescheduled the orientation session to Friday."})

    result = api.overview(FakeGateway(FakeManager(store)), store)

    all_sentences = (
        [r["sentence"] for r in result["stuck"]]
        + [r["sentence"] for r in result["running"]]
        + [r["sentence"] for r in result["done_today"]]
    )
    assert all_sentences  # the fixture actually produced rows to check
    for sentence in all_sentences:
        _assert_no_forbidden_text(sentence)

    pending_row = next(r for r in result["stuck"] if r["id"] == pending.id)
    assert "orientation session" in pending_row["sentence"]

    running_row = next(r for r in result["running"] if r["id"] == running.id)
    assert "orientation session" in running_row["sentence"]

    failed_row = next(r for r in result["stuck"] if r["id"] == failed.id)
    assert failed_row["sentence"] == "I want to reschedule the orientation session.: it ran out of room handling something too large."

    done_row = next(r for r in result["done_today"] if r["id"] == done.id)
    assert "Rescheduled the orientation session to Friday" in done_row["sentence"]


def test_schedule_with_first_person_failed_run_is_stuck_once_with_normalized_reason(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Donor tracker sync", state="active")
    # The schedule's own run also creates a request row (schedule_id set) -- this
    # must never also appear as a second, separate Stuck "request" row.
    task_id = store.create_task(schedule.title)["id"]
    run_request = store.create_request(
        chat_id="owner", sender="owner", sender_name="Blatbot", mode="schedule", subject="",
        original_message="", summary="Checking the donor tracker.", scopes=["web"],
        prompt="check the donor tracker", state="failed", task_id=task_id, schedule_id=schedule.id,
    )
    store.set_state(run_request.id, "failed", status={"error": "could not find the tracker page"})
    _add_run(store, schedule.id, outcome="failed",
             result_line="I could not find the donor tracker page.")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    schedule_rows = [r for r in result["stuck"] if r["id"] == schedule.id]
    assert len(schedule_rows) == 1
    assert schedule_rows[0]["sentence"] == "Donor tracker sync: it could not find the donor tracker page."
    assert run_request.id not in [r["id"] for r in result["stuck"] if r["kind"] == "request"]
    assert run_request.id not in [r["id"] for r in result["done_today"] if r["kind"] == "request"]


def test_stuck_schedule_run_is_not_also_listed_under_done_today(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Inbox triage", state="active")
    _add_run(store, schedule.id, outcome="incomplete", result_line="The total did change again (+$200).")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert any(r["id"] == schedule.id for r in result["stuck"])
    assert not any(r["id"] == schedule.id and r["kind"] == "schedule" for r in result["done_today"])


# ---------------------------------------------------------------------------
# running / done_today / coming_up / counts / sources.
# ---------------------------------------------------------------------------

def test_running_request_and_running_schedule_both_appear(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Live sync", state="active")
    sched_request = _make_request(store, summary="running the sync", state="running",
                                  schedule_id=schedule.id)
    plain_request = _make_request(store, summary="Draft the weekly memo.", state="running")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    ids = {(r["kind"], r["id"]) for r in result["running"]}
    assert ("schedule", schedule.id) in ids
    assert ("request", plain_request.id) in ids
    assert result["counts"]["running"] == len(result["running"]) == 2


def test_done_today_includes_finished_requests_and_schedule_runs_since_midnight(tmp_path):
    store = make_store(tmp_path)
    request = _make_request(store, summary="Draft the weekly memo.", state="done")
    schedule = _make_schedule(store, title="Team total watch", state="active")
    _add_run(store, schedule.id, outcome="complete", result_line="Total went up by $500.")

    result = api.overview(FakeGateway(FakeManager(store)), store)

    kinds_ids = {(r["kind"], r["id"]) for r in result["done_today"]}
    assert ("request", request.id) in kinds_ids
    assert ("schedule", schedule.id) in kinds_ids
    by_id = {(r["kind"], r["id"]): r for r in result["done_today"]}
    assert by_id[("request", request.id)]["outcome"] == "finished"
    assert by_id[("schedule", schedule.id)]["outcome"] == "finished"
    assert result["counts"]["done_today"] == len(result["done_today"])


def test_done_today_excludes_runs_finished_before_todays_midnight(tmp_path):
    store = make_store(tmp_path)
    schedule = _make_schedule(store, title="Old run", state="active")
    _add_run(store, schedule.id, outcome="complete", result_line="yesterday's result",
             finished_at=time.time() - 200000)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert schedule.id not in [r["id"] for r in result["done_today"] if r["kind"] == "schedule"]


def test_coming_up_lists_the_next_three_schedule_firings_only(tmp_path):
    store = make_store(tmp_path)
    now = time.time()
    for i, title in enumerate(["First", "Second", "Third", "Fourth"]):
        s = _make_schedule(store, title=title, state="active")
        store.update_schedule(s.id, next_run=now + (i + 1) * 3600)

    result = api.overview(FakeGateway(FakeManager(store)), store)

    assert [c["title"] for c in result["coming_up"]] == ["First", "Second", "Third"]
    assert all(c["next_run_in_seconds"] > 0 for c in result["coming_up"])


def test_counts_match_the_section_lengths(tmp_path):
    store = make_store(tmp_path)
    _make_request(store, summary="Approve this.", state="pending")
    _make_schedule(store, title="Running one", state="active")
    result = api.overview(FakeGateway(FakeManager(store)), store)
    assert result["counts"] == {
        "stuck": len(result["stuck"]), "running": len(result["running"]),
        "done_today": len(result["done_today"]),
    }


def test_sources_are_present_and_describe_what_the_page_is_built_from(tmp_path):
    store = make_store(tmp_path)
    _make_request(store, summary="Approve this.", state="pending")
    result = api.overview(FakeGateway(FakeManager(store)), store)
    labels = {s["label"] for s in result["sources"]}
    assert labels == {"Requests", "Schedules and their runs"}
    for source in result["sources"]:
        assert source["detail"]


# ---------------------------------------------------------------------------
# The schedule state label helper: never "Waiting".
# ---------------------------------------------------------------------------

def test_active_schedule_with_nothing_running_is_labeled_active_not_waiting(tmp_path):
    assert api._state_label(_FakeSchedule(state="active"), running=False) == "Active"
    assert api._state_label(_FakeSchedule(state="active"), running=True) == "Running"


class _FakeSchedule:
    def __init__(self, state, last_outcome=""):
        self.state = state
        self.last_outcome = last_outcome


# ---------------------------------------------------------------------------
# The Origin gate applies to /console/api/overview like every other route.
# ---------------------------------------------------------------------------

def _test_app(manager):
    app = web.Application()
    app["console.gateway"] = FakeGateway(manager)
    r = console_routes
    app.router.add_get("/console/api/overview", r._endpoint(r.get_overview))
    return app


@pytest.mark.asyncio
async def test_overview_route_serves_the_contract_shape():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        manager = FakeManager(Store(f"{tmp}/overview.db"))
        _make_request(manager.store, summary="Approve this.", state="pending")
        async with TestClient(TestServer(_test_app(manager))) as client:
            resp = await client.get("/console/api/overview")
            assert resp.status == 200
            body = await resp.json()
            assert set(body.keys()) == {"stuck", "running", "done_today", "coming_up", "counts", "sources"}


@pytest.mark.asyncio
async def test_overview_route_is_gated_by_the_tailnet_origin_check_like_the_others():
    from inkbox_claude.console import access
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        manager = FakeManager(Store(f"{tmp}/overview.db"))
        app = _test_app(manager)
        app.middlewares.append(access.tailnet_gate)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/console/api/overview")
            # Same as every other console route: a loopback test-client peer is
            # not a tailnet address, so the gate refuses it before the handler
            # (and its store lookup) ever runs.
            assert resp.status == 404
