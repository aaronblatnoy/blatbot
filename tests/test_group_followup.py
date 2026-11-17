"""A follow-up set up in a Telegram group (Aaron, 2026-11-17 1:15 AM):

    "blatbot starting 9am today follow up every 30minutes until he gives us his email"

The stated start sets the first run, the runs post in that group (no iMessage
scope), the stop condition ends it, Aaron hears only when it ends, and his bare
"yes" in the group starts it -- with TypeSafe down, since nothing here asks it.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from inkbox_claude.gate import schedprop
from tests.test_schedule_flow import _propose_via_manager, make_manager

TZ = ZoneInfo("America/New_York")
GROUP = "FellowSIp of the Ring"
TAG = (f"[telegram group: {GROUP} | from: Aaron Blatnoy | your reply goes to the whole group; "
       "answer only what was asked of you]\n")
SAID = "blatbot starting 9am today follow up every 30minutes until he gives us his email"
META = {"telegram_chat_id": "-5275613007", "sender": "telegram:1", "telegram_user_id": "1",
        "to": "-5275613007", "conversation_kind": "group", "group_title": GROUP}
PROMPT = ("In the Telegram group chat \"FellowSIp of the Ring\", send a short message to Stanley Rabkin "
          "asking for his email address for today's 12:30pm meeting invite. End with a RESULT line and a "
          "STATUS line (OK, FAILED, or STOP).")


def _at(hour, minute=0, day=17):
    return datetime(2026, 11, day, hour, minute, tzinfo=TZ).timestamp()


def test_parse_start_reads_his_stated_start():
    now = _at(1, 15)
    assert schedprop.parse_start(SAID, "America/New_York", now) == _at(9)
    assert schedprop.parse_start(TAG + SAID, "America/New_York", now) == _at(9)
    assert schedprop.parse_start("start at 7:30 tomorrow", "America/New_York", now) == _at(7, 30, 18)
    assert schedprop.parse_start("starting 9 today", "America/New_York", now) == _at(9)
    assert schedprop.parse_start("check every hour", "America/New_York", now) is None
    assert schedprop.parse_start("results from 2 people", "America/New_York", now) is None
    # Already passed today: not a start time at all.
    assert schedprop.parse_start("starting 1am today", "America/New_York", now) is None


def test_parse_stop_when_reads_a_condition_not_a_date():
    assert schedprop.parse_stop_when(TAG + SAID) == "he gives us his email"
    assert schedprop.parse_stop_when("check every hour until she replies.") == "she replies"
    assert schedprop.parse_stop_when("every day until Friday") is None
    assert schedprop.parse_stop_when("every hour until 5pm") is None
    assert schedprop.parse_stop_when("every hour until tomorrow") is None
    assert schedprop.parse_stop_when("every hour") is None


def test_group_scopes_drop_unasked_send_channels():
    scopes = schedprop.group_scopes(["imessage_send", "telegram_read"], SAID, PROMPT, META)
    assert scopes == ["telegram_read"]
    # Outside a group nothing changes.
    assert schedprop.group_scopes(["imessage_send"], SAID, PROMPT, {}) == ["imessage_send"]
    # He asked for the channel himself: kept.
    kept = schedprop.group_scopes(["imessage_send"], "send an imessage to him every hour", PROMPT, META)
    assert "imessage_send" in kept


async def _propose_group_followup(m, monkeypatch, now):
    monkeypatch.setattr(time, "time", lambda: now)
    await _propose_via_manager(
        m, title="Follow up with Stanley for his email", prompt=PROMPT, said=TAG + SAID,
        cadence_kind="every", cadence={"minutes": 30}, scopes=["imessage_send", "telegram_read"],
        session_mode="telegram", session_chat_id="tg:-5275613007", session_meta=dict(META))
    return m.store.schedules(states=["proposed"])[0]


@pytest.mark.asyncio
async def test_group_followup_proposal_and_yes_in_the_group(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_APPROVER_TELEGRAM_ID", "1")
    m, sent = make_manager(tmp_path)
    now = _at(1, 15)
    schedule = await _propose_group_followup(m, monkeypatch, now)

    assert schedule.scopes == ["telegram_read"]
    assert schedule.report_mode == "end"
    assert not schedule.report_every_time
    assert schedule.cadence.get("anchor") == _at(9)
    assert schedule.cadence.get("stop_when") == "he gives us his email"
    assert "STATUS: STOP" in schedule.prompt and "he gives us his email" in schedule.prompt
    target, text, mode, _meta = sent[-1]
    assert (target, mode) == ("tg:-5275613007", "telegram")
    assert "starting 9:00 AM today" in text
    assert "Stops when: he gives us his email, or it gives up after 20 tries" in text
    assert f"post in the {GROUP} group" in text
    assert "iMessage" not in text
    assert "You hear from it: only when it ends" in text
    assert "every time it runs" not in text

    # His bare "yes" in the group, behind the bridge's routing tag, starts it --
    # even with another request pending on a different chat.
    m.store.create_request(chat_id="owner", sender="x", sender_name="x", mode="imessage", subject="",
                           original_message="other", summary="other", scopes=["web"], prompt="p",
                           state="pending", task_id=m.store.create_task("other")["id"])
    m.store.create_request(chat_id="owner", sender="x", sender_name="x", mode="imessage", subject="",
                           original_message="other2", summary="other2", scopes=["web"], prompt="p",
                           state="pending", task_id=m.store.create_task("other2")["id"])
    session = m.get("tg:-5275613007")
    await session.handle_inbound(TAG + "yes", mode="telegram", meta=dict(META))
    active = m.store.get_schedule(schedule.id)
    assert active.state == "active"
    assert active.next_run == _at(9)
    assert len(m.store.schedules()) == 1  # no duplicate proposal
    assert "First check 9:00 AM" in sent[-1][1]


@pytest.mark.asyncio
async def test_group_followup_stop_ends_it_and_tells_aaron_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_APPROVER_TELEGRAM_ID", "1")
    m, sent = make_manager(tmp_path)
    schedule = await _propose_group_followup(m, monkeypatch, _at(1, 15))
    monkeypatch.setattr(time, "time", lambda: _at(1, 16))
    active = await m.schedules.decide(schedule.id, True)
    owner_notices = []

    async def to_approver(text, **kw):
        owner_notices.append(text)
    m.send_to_approver = to_approver

    def result(line, stop=False):
        return {"ok": True, "result_line": line, "final_status": "STATUS: STOP" if stop else "checked",
                "note": "", "denied": [], "tool_calls": ["mcp__host__send_message"], "held": False,
                "started_at": _at(9), "finished_at": _at(9, 1), "cost_usd": None,
                **({"stop": True} if stop else {})}

    # A run that asked again: no notice to anyone beyond its own post.
    before = len(sent)
    m.executor.results.append(result("Asked Stanley for his email again."))
    monkeypatch.setattr(time, "time", lambda: _at(9))
    await m.schedules.tick(now=active.next_run)
    for _ in range(20):
        await asyncio.sleep(0)
    assert m.store.get_schedule(schedule.id).state == "active"
    assert owner_notices == []
    assert len(sent) == before

    # He answered: the run's STATUS: STOP ends it, and Aaron hears once.
    run = m.store.create_request(chat_id=schedule.chat_id, sender="Aaron", sender_name="Aaron",
                                 mode="schedule", subject="", original_message=schedule.prompt,
                                 summary=schedule.title, scopes=["telegram_read"], prompt=schedule.prompt,
                                 state="approved", task_id=schedule.task_id, schedule_id=schedule.id)
    m.store.create_schedule_run(schedule.id, run.id, _at(9, 30))
    await m.schedules.finished(run, result("Stanley gave his email, stanley@example.com.", stop=True))
    ended = m.store.get_schedule(schedule.id)
    assert ended.state == "ended" and ended.next_run is None
    assert len(owner_notices) == 1 and "stanley@example.com" in owner_notices[0]


@pytest.mark.asyncio
async def test_group_followup_gives_up_after_max_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_APPROVER_TELEGRAM_ID", "1")
    m, _sent = make_manager(tmp_path)
    schedule = await _propose_group_followup(m, monkeypatch, _at(1, 15))
    await m.schedules.decide(schedule.id, True)
    m.store.update_schedule(schedule.id, run_count=schedule.max_runs)
    owner_notices = []

    async def to_approver(text, **kw):
        owner_notices.append(text)
    m.send_to_approver = to_approver
    run = m.store.create_request(chat_id=schedule.chat_id, sender="Aaron", sender_name="Aaron",
                                 mode="schedule", subject="", original_message=schedule.prompt,
                                 summary=schedule.title, scopes=["telegram_read"], prompt=schedule.prompt,
                                 state="approved", task_id=schedule.task_id, schedule_id=schedule.id)
    m.store.create_schedule_run(schedule.id, run.id, _at(18, 30))
    await m.schedules.finished(run, {"ok": True, "result_line": "Asked again.", "final_status": "ok",
                                     "note": "", "denied": [], "tool_calls": [], "held": False,
                                     "started_at": _at(18, 30), "finished_at": _at(18, 31),
                                     "cost_usd": None})
    ended = m.store.get_schedule(schedule.id)
    assert ended.state == "ended"
    assert len(owner_notices) == 1 and "gave up after 20 tries" in owner_notices[0]
