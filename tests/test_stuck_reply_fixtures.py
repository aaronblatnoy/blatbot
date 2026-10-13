"""Fixtures shaped like the real-data rows the coordinator flagged (made-up
content throughout): a schedule needing repair, an active schedule whose
last run failed with a first-person result line, an individual call request
that failed on a buffer-size error, and four call requests on one thread
that all failed with the exact session-limit error within a few minutes of
each other (one with a one-word, transcript-like summary). Exercises
overview.stuck_rows (Home) and schedcmd.stuck_reply (chat) against the SAME
fixtures, and checks the forbidden shapes the coordinator named: a raw
transcript, "something broke" for a usage limit, "the the", and a cut word.
"""

from __future__ import annotations

import re
import time

import pytest

from inkbox_claude.gate import overview as overview_mod
from inkbox_claude.gate import schedcmd
from inkbox_claude.gate import schedules as sched_mod
from inkbox_claude.gate import wording
from inkbox_claude.gate.router import ScheduleCommand
from inkbox_claude.gate.store import Store

BUFFER_ERROR = "A tool result exceeded the maximum buffer size"


def _format_reset(ts):
    """"3:10pm (America/New_York)"-style text for `ts`, used to build a
    session-limit error whose reset is always shortly after the fixture's
    own failure times, regardless of what day or hour the suite happens to
    run in."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("America/New_York")
    dt = datetime.fromtimestamp(ts, tz)
    return dt.strftime("%-I:%M%p").lower() + " (America/New_York)"


def _session_limit_error(reset_ts):
    return ("Claude Code returned an error result: You've hit your session limit "
            f"· resets {_format_reset(reset_ts)} (exit code: 1)")
CALL_HANDOFF_PREFIX = "live phone call with a caller\n\nWhat was just said (act on this):\ncaller: "


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.notices = []
        self.schedules = sched_mod.ScheduleService(self)

    async def send_schedule_notice(self, schedule, text):
        self.notices.append((schedule, text))
        return self.store.add_message(schedule.origin_chat_id or "owner", "outbound", text,
                                      schedule.origin_mode or "imessage", role="notice",
                                      schedule_id=schedule.id)


class FakeSession:
    def __init__(self, chat_id="owner", approver=True, inbound_id=None):
        self.chat_id = chat_id
        self.record_id = chat_id
        self.inbound_id = inbound_id
        self._approver = approver

    def is_approver(self):
        return self._approver


def _command(action, target=None, **changes):
    return ScheduleCommand(action=action, target=target, changes=changes)


def _backdate(store, request_id, *, created_at, updated_at=None):
    updated_at = created_at if updated_at is None else updated_at
    store._db.execute("UPDATE requests SET created_at=?, updated_at=? WHERE id=?",
                      (created_at, updated_at, request_id))
    store._db.commit()


@pytest.fixture
def env(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    manager = FakeManager(store)
    return store, manager


def _needs_repair_schedule(store, *, title="Tamid project watch"):
    task_id = store.create_task(title)["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title=title,
        prompt=("Check the tamid signal page every 4 hours and tell me if it changed.\n\n"
                "When done, call schedule_continue with delay_minutes 240 to run again.\n\n"
                "If it changed, call mcp__tamid-drive__legacy_tool to update the sheet."),
        kind="", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="", cadence={}, origin_mode="imessage", origin_chat_id="owner")
    return store.update_schedule(schedule.id, state="paused", last_outcome="needs a cadence")


def _active_schedule_with_failed_run(store, *, title="Donor site check"):
    task_id = store.create_task(title)["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title=title,
        prompt="Check the donor page and call inkbox_update_contact to update the tracker note.",
        kind="every", scopes=["web"], timezone="America/New_York", report_mode="changed",
        cadence_kind="every", cadence={"minutes": 60}, origin_mode="imessage", origin_chat_id="owner")
    schedule = store.update_schedule(schedule.id, state="active", next_run=time.time() + 60)
    run_id = store.create_schedule_run(schedule.id, None, time.time() - 120)
    store.finish_schedule_run(
        run_id, finished_at=time.time() - 60, complete=False, outcome="incomplete",
        result_line="I could not update the tracker note, so nothing changed on my end.",
        final_status="blocked", denied=[{"tool": "mcp__inkbox__inkbox_update_contact", "reason": "scope"}],
        note="", changed=True, cost_usd=None)
    return schedule


def _buffer_failed_call_request(store, *, when):
    task_id = store.create_task("buffer ask")["id"]
    text = CALL_HANDOFF_PREFIX + "Can you pull the whole site map for me?"
    req = store.create_request(
        chat_id="call-thread-1", sender="+15550100100", sender_name="", mode="voice",
        subject="", original_message=text, summary=text,
        scopes=["web"], prompt="pull the site map", state="pending", task_id=task_id)
    store.set_state(req.id, "failed", status={"error": BUFFER_ERROR})
    _backdate(store, req.id, created_at=when, updated_at=when)
    return store.get_request(req.id)


def _session_limit_call_requests(store, *, chat_id="call-thread-2", start, reset_ts=None):
    """Four requests from one call, all failing with the exact session-limit
    error, a minute apart (within the same clock hour) -- one with a
    one-word, transcript-like summary (what a real "Again." utterance looks
    like). The reset time is always a few minutes after the last failure,
    computed from `start` rather than hard-coded, so the fixture behaves the
    same whatever time of day the suite runs."""
    reset_ts = start + 600 if reset_ts is None else reset_ts
    error = _session_limit_error(reset_ts)
    summaries = [
        CALL_HANDOFF_PREFIX + "Could you check the one pager format and some other things for me?",
        CALL_HANDOFF_PREFIX + "I am Aaron um okay where can we check the GitHub.",
        CALL_HANDOFF_PREFIX + "Again.",
        CALL_HANDOFF_PREFIX + "Walk me through the steps you took.",
    ]
    out = []
    for i, summary in enumerate(summaries):
        task_id = store.create_task(f"call ask {i}")["id"]
        req = store.create_request(
            chat_id=chat_id, sender="+15550100200", sender_name="", mode="voice",
            subject="", original_message=summary, summary=summary, scopes=["web"],
            prompt=summary, state="pending", task_id=task_id)
        store.set_state(req.id, "failed", status={"error": error})
        when = start + i * 60
        _backdate(store, req.id, created_at=when, updated_at=when)
        out.append(store.get_request(req.id))
    return out


def _minute_rounded(ts):
    """The reset text format drops seconds, so the epoch round-trips back
    through _format_reset only to the minute -- round the same way the
    fixture's error text itself does, for an exact expiry comparison."""
    return ts - (ts % 60)


def _assert_clean(text):
    low = text.lower()
    for bad in ("what was just said", "caller:", "live phone call"):
        assert bad not in low, f"{bad!r} found in: {text!r}"
    assert not re.search(r"\bthe the\b", low), f"'the the' found in: {text!r}"
    for part in text.split("."):
        if "usage limit" in part:
            assert "something broke" not in part, f"'something broke' found alongside a usage limit: {text!r}"


# ---------------------------------------------------------------------------
# The combined scenario: before the fix this read as six stuck things with a
# raw transcript and "something broke on my side" for a session limit; after
# the fix it is four things, the session-limit calls collapsed into one.
# ---------------------------------------------------------------------------

def test_combined_scenario_home_rows(env, monkeypatch):
    store, manager = env
    now = time.time()
    monkeypatch.setattr(overview_mod.time, "time", lambda: now)
    _needs_repair_schedule(store)
    _active_schedule_with_failed_run(store)
    buffer_req = _buffer_failed_call_request(store, when=now - 300)
    cluster_start = now - 600
    cluster_reset = now + 300
    cluster_reqs = _session_limit_call_requests(store, start=cluster_start, reset_ts=cluster_reset)

    rows = overview_mod.stuck_rows(store, manager.schedules)
    assert len(rows) == 4

    kinds = [r["kind"] for r in rows]
    assert kinds == ["schedule", "schedule", "request", "request_cluster"]

    repair_row, failed_schedule_row, buffer_row, cluster_row = rows
    assert repair_row["sentence"].startswith("Tamid project watch: ")
    after_colon = repair_row["sentence"].split(": ", 1)[1]
    assert after_colon[:1].islower() or not after_colon[:1].isalpha()

    assert failed_schedule_row["sentence"].startswith("Donor site check: ")
    assert "I could not update" not in failed_schedule_row["sentence"]

    assert buffer_row["sentence"] == (
        f"the request from the {wording.call_time_text(buffer_req.created_at)} call: "
        "it ran out of room handling something too large.")
    assert buffer_row["say"] == "try again"

    assert cluster_row["say"] is None
    assert cluster_row["actions"] == []
    when = wording.call_time_text(cluster_reqs[0].created_at)
    assert cluster_row["sentence"] == (
        f"4 requests from the {when} call failed: "
        f"it had hit a usage limit until {_format_reset(cluster_reset)}.")
    for forbidden in ("Could you check the one pager", "I am Aaron", "Again.", "Walk me through"):
        assert forbidden not in cluster_row["sentence"]

    for row in rows:
        _assert_clean(row["sentence"])


@pytest.mark.asyncio
async def test_combined_scenario_chat_reply(env, monkeypatch):
    store, manager = env
    now = time.time()
    monkeypatch.setattr(overview_mod.time, "time", lambda: now)
    monkeypatch.setattr(schedcmd.time, "time", lambda: now)
    _needs_repair_schedule(store)
    _active_schedule_with_failed_run(store)
    _buffer_failed_call_request(store, when=now - 300)
    _session_limit_call_requests(store, start=now - 600, reset_ts=now + 300)

    reply = await schedcmd.handle(manager, FakeSession(), _command("stuck"))

    assert reply.startswith("Four things are stuck. ")
    lines = reply[len("Four things are stuck. "):].split("\n")
    assert len(lines) == 4
    assert lines[0].startswith("Tamid project watch: ")
    assert '"repair the tamid one"' in lines[0]
    assert lines[1].startswith("Donor site check: ")
    assert '"let the donor one' in lines[1]
    assert "the request from the" in lines[2] and "call:" in lines[2]
    assert '"try the' in lines[2] and "one again" in lines[2]
    assert lines[3] == "And 1 more on the console."
    _assert_clean(reply)


def test_cluster_disappears_from_stuck_after_its_reset_time_passes(env, monkeypatch):
    store, manager = env
    start = time.time() - 600
    reset_epoch = _minute_rounded(start + 600)
    _session_limit_call_requests(store, start=start, reset_ts=reset_epoch)

    monkeypatch.setattr(overview_mod.time, "time", lambda: reset_epoch - 60)
    rows_before = overview_mod.stuck_rows(store, manager.schedules)
    assert any(r["kind"] == "request_cluster" for r in rows_before)

    monkeypatch.setattr(overview_mod.time, "time", lambda: reset_epoch + 60)
    rows_after = overview_mod.stuck_rows(store, manager.schedules)
    assert not any(r["kind"] == "request_cluster" for r in rows_after)

    done = overview_mod.done_today_rows(store, now=reset_epoch + 60)
    # the four identical failures surface under Done today as one line, not four
    collapsed = [d for d in done if d["sentence"].startswith("4 requests failed: it had hit a usage limit")]
    assert len(collapsed) == 1
    assert collapsed[0]["actions"] == []


def test_cluster_with_no_reset_time_expires_after_two_hours(env, monkeypatch):
    store, manager = env
    start = time.time() - 600
    task_id = store.create_task("no reset")["id"]
    text = CALL_HANDOFF_PREFIX + "Hello."
    req = store.create_request(
        chat_id="call-thread-3", sender="+15550100300", sender_name="", mode="voice",
        subject="", original_message=text, summary=text, scopes=["web"], prompt="hello",
        state="pending", task_id=task_id)
    store.set_state(req.id, "failed", status={"error": "You've hit your session limit, try later"})
    _backdate(store, req.id, created_at=start, updated_at=start)

    monkeypatch.setattr(overview_mod.time, "time", lambda: start + 3600)
    rows_within = overview_mod.stuck_rows(store, manager.schedules)
    assert any(r["kind"] == "request_cluster" for r in rows_within)

    monkeypatch.setattr(overview_mod.time, "time", lambda: start + 2.1 * 3600)
    rows_past = overview_mod.stuck_rows(store, manager.schedules)
    assert not any(r["kind"] == "request_cluster" for r in rows_past)


# ---------------------------------------------------------------------------
# Every quoted "say" phrase a reply offers really does the thing it claims,
# run back through the real schedule-command handler.
# ---------------------------------------------------------------------------

_SAY_RE = re.compile(r'"([^"]+)"')


@pytest.mark.asyncio
async def test_every_quoted_phrase_in_the_stuck_reply_does_what_it_says(env, monkeypatch):
    store, manager = env
    now = time.time()
    monkeypatch.setattr(overview_mod.time, "time", lambda: now)
    monkeypatch.setattr(schedcmd.time, "time", lambda: now)
    repair_schedule = _needs_repair_schedule(store)
    failed_schedule = _active_schedule_with_failed_run(store)

    reply = await schedcmd.handle(manager, FakeSession(), _command("stuck"))
    phrases = _SAY_RE.findall(reply)
    assert phrases, "expected at least one quoted phrase"
    for phrase in phrases:
        assert phrase, "a reply never offers an empty quoted phrase"
        assert not re.search(r"\bthe the\b", phrase.lower())

    repair_phrase = next(p for p in phrases if p.startswith("repair "))
    found = schedcmd.resolve(store, repair_phrase[len("repair "):], "owner", None)
    assert found is not None and getattr(found, "id", None) == repair_schedule.id

    let_it_phrase = next(p for p in phrases if p.startswith("let "))
    # "let the donor one <verb phrase>" still resolves to the schedule by name.
    target_words = let_it_phrase.split()
    assert "donor" in target_words
    found2 = schedcmd.resolve(store, "the donor one", "owner", None)
    assert found2 is not None and getattr(found2, "id", None) == failed_schedule.id
    reply2 = await schedcmd.handle(manager, FakeSession(), _command("add_capability", target="the donor one"))
    assert reply2
