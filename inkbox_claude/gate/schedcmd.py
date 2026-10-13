"""Plain-language schedule management in chat, second version.

Two jobs: answering an open schedule proposal with a bare yes/no before the
router ever sees the message (`answer_proposal`, called from
manager.py's handle_approver_command), and turning the router's
`schedule_command` into a short plain reply (`resolve` + `handle`).

No Jev anywhere in this module. Scope names, "#<id>" and markdown never reach
the owner through anything this module returns; "S<id>" is accepted only as
an INPUT handle (see resolve), never produced in a reply.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

from . import cadence as cadence_module
from . import schedprop
from .store import Schedule

# A small fixed list, not a model: Aaron's own close variants of yes/no on a
# schedule proposal. Checked whole-message, case-insensitive, whitespace
# collapsed -- "yes" inside a longer sentence is not a bare answer.
YES_PHRASES = {"yes", "y", "yep", "yeah", "yup", "yes please", "yes do it"}
NO_PHRASES = {"no", "n", "nope", "nah", "no thanks", "cancel that"}


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def is_yes(text: str) -> bool:
    return _norm(text) in YES_PHRASES


def is_no(text: str) -> bool:
    return _norm(text) in NO_PHRASES


def _field(command: Any, name: str, default: Any = None) -> Any:
    """command is the router's ScheduleCommand, which may arrive as the real
    pydantic model or (in tests) a plain dict -- read either the same way."""
    if isinstance(command, dict):
        return command.get(name, default)
    return getattr(command, name, default)


def _open_proposals(store: Any, chat_id: str) -> List[Schedule]:
    return [s for s in store.schedules(states=["proposed"]) if s.chat_id == chat_id]


_BARE_ANSWER_WINDOW_S = 15 * 60


def _captures_bare_answer(store: Any, chat_id: str, schedule: Schedule) -> bool:
    """A bare yes/no on this thread answers `schedule`'s proposal only when its
    own notice is still the most recent outbound message on the thread, or it
    went out within the last 15 minutes. Otherwise too much may have happened
    since, so a bare yes/no is not captured at all (item G) -- the message
    goes on to the router as today."""
    with store._lock:
        prop_row = store._db.execute(
            "SELECT created_at FROM messages WHERE chat_id=? AND schedule_id=? AND kind='outbound' "
            "ORDER BY created_at DESC LIMIT 1", (chat_id, schedule.id)).fetchone()
        if prop_row is None:
            return False
        sent_at = float(prop_row["created_at"])
        latest_row = store._db.execute(
            "SELECT created_at FROM messages WHERE chat_id=? AND kind='outbound' "
            "ORDER BY created_at DESC LIMIT 1", (chat_id,)).fetchone()
    is_latest = latest_row is None or float(latest_row["created_at"]) <= sent_at + 1e-6
    return is_latest or (time.time() - sent_at) <= _BARE_ANSWER_WINDOW_S


_BARE_ANSWER_WINDOW_S = 15 * 60


def _captures_bare_answer(store: Any, chat_id: str, schedule: Schedule) -> bool:
    """A bare yes/no on this thread answers 's proposal only when its
    own notice is still the most recent outbound message on the thread, or it
    went out within the last 15 minutes. Otherwise too much may have happened
    since, so a bare yes/no is not captured at all (item G) -- the message
    goes on to the router as today."""
    with store._lock:
        prop_row = store._db.execute(
            "SELECT created_at FROM messages WHERE chat_id=? AND schedule_id=? AND kind='outbound' "
            "ORDER BY created_at DESC LIMIT 1", (chat_id, schedule.id)).fetchone()
        if prop_row is None:
            return False
        sent_at = float(prop_row["created_at"])
        latest_row = store._db.execute(
            "SELECT created_at FROM messages WHERE chat_id=? AND kind='outbound' "
            "ORDER BY created_at DESC LIMIT 1", (chat_id,)).fetchone()
    is_latest = latest_row is None or float(latest_row["created_at"]) <= sent_at + 1e-6
    return is_latest or (time.time() - sent_at) <= _BARE_ANSWER_WINDOW_S


# ---------------------------------------------------------------------------
# Answering a bare yes/no before the router ever sees the message
# ---------------------------------------------------------------------------

async def answer_proposal(manager: Any, session: Any, text: str) -> Optional[str]:
    """If `text` is a bare yes/no and exactly one schedule proposal is open on
    this thread, answer it through ScheduleService.decide and return the reply
    to send (empty string when decide() already sent its own notice and there
    is nothing more to say). Returns None when nothing here applies -- text is
    not a plain yes/no, or no proposal is open -- so the caller falls through
    to the router. Call this only AFTER checking that no pending request on
    Aaron's plate already claims this reply; that precedence is the caller's
    job, not this function's."""
    yes, no = is_yes(text), is_no(text)
    if not yes and not no:
        return None
    chat_id = getattr(session, "record_id", None) or getattr(session, "chat_id", "")
    proposals = _open_proposals(manager.store, chat_id)
    proposals = [p for p in proposals if _captures_bare_answer(manager.store, chat_id, p)]
    if not proposals:
        return None
    if len(proposals) > 1:
        titles = " or ".join(f'"{s.title}"' for s in proposals)
        return f"Which one: {titles}?"
    await manager.schedules.decide(proposals[0].id, yes)
    return ""


# ---------------------------------------------------------------------------
# resolve(): which schedule he means
# ---------------------------------------------------------------------------

_HANDLE_RE = re.compile(r"^s(\d+)$", re.I)
_RECENT_NOTICE_WINDOW_S = 12 * 3600

# Words that carry no identifying content when he refers to a schedule ("the
# team one", "that thing") -- dropped before matching against titles.
_FILLER_WORDS = {
    "the", "a", "an", "one", "that", "this", "it", "thing", "schedule",
    "job", "check", "monitor", "my", "please",
}


def _word_matches_title_word(word: str, title_word: str) -> bool:
    if word == title_word:
        return True
    return len(title_word) >= 4 and title_word.startswith(word)


def _schedule_matches_words(schedule: Schedule, words: List[str]) -> bool:
    title_words = [w for w in re.split(r"\s+", schedule.title.lower()) if w]
    return bool(words) and all(
        any(_word_matches_title_word(w, tw) for tw in title_words) for w in words)


def _recent_notice_schedule(store: Any, chat_id: str) -> Optional[Schedule]:
    """The schedule of the most recent schedule notice or proposal on this
    thread within the last 12 hours, if there is exactly one such schedule.
    No store method already returns this, so this reads the messages table
    directly (same lock store itself uses) rather than add one to store.py,
    which is another package's file."""
    if not chat_id:
        return None
    cutoff = time.time() - _RECENT_NOTICE_WINDOW_S
    with store._lock:
        rows = store._db.execute(
            "SELECT schedule_id FROM messages WHERE chat_id=? AND schedule_id IS NOT NULL "
            "AND created_at>=? ORDER BY created_at DESC",
            (chat_id, cutoff),
        ).fetchall()
    ids: List[int] = []
    for r in rows:
        sid = int(r["schedule_id"])
        if sid not in ids:
            ids.append(sid)
    if len(ids) == 1:
        return store.get_schedule(ids[0])
    return None


def _message_text(store: Any, message_id: Optional[int]) -> str:
    if not message_id:
        return ""
    with store._lock:
        row = store._db.execute("SELECT text FROM messages WHERE id=?", (int(message_id),)).fetchone()
    return str(row["text"]) if row else ""


def resolve(store: Any, text_target: Optional[str], chat_id: str,
            reply_to_message_id: Optional[int]) -> Union[Schedule, List[Schedule], None]:
    """Which schedule the owner means, in this fixed order:
    1. the schedule the replied-to message is tagged with;
    2. an explicit "S<id>" handle;
    3. an exact title match, case-insensitive;
    4. a unique match where every word of the target appears in the title;
    5. when target is empty: the schedule of the most recent schedule notice
       or proposal on this thread within the last 12 hours, if there is
       exactly one; else the only non-ended schedule on this thread, if there
       is exactly one.
    Returns one Schedule, a list of candidates when ambiguous (title or
    partial match hits more than one), or None when nothing matches."""
    if reply_to_message_id:
        tagged = store.schedule_for_message(reply_to_message_id)
        if tagged is not None:
            return tagged
    target = (text_target or "").strip()
    if target:
        m = _HANDLE_RE.match(target)
        if m:
            found = store.get_schedule(int(m.group(1)))
            if found is not None:
                return found
        all_schedules = store.schedules()
        exact = [s for s in all_schedules if s.title.strip().lower() == target.lower()]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return exact
        words = [w for w in re.split(r"\s+", target.lower()) if w]
        words = [w for w in words if w not in _FILLER_WORDS]
        if words:
            partial = [s for s in all_schedules if _schedule_matches_words(s, words)]
            if len(partial) == 1:
                return partial[0]
            if len(partial) > 1:
                return partial
            return None
        # Nothing was left after dropping filler words ("that one"): treat the
        # target as if he had named none, and fall through to the empty-target
        # rules below instead of reporting no match.
    recent = _recent_notice_schedule(store, chat_id)
    if recent is not None:
        return recent
    active = [s for s in store.schedules() if s.chat_id == chat_id and s.state != "ended"]
    if len(active) == 1:
        return active[0]
    if len(active) > 1:
        return active
    return None


# ---------------------------------------------------------------------------
# Wording helpers
# ---------------------------------------------------------------------------

def _clock_text(ts: Optional[float], timezone_name: str) -> str:
    """"2:30 PM", with the day added when it is not today, in the given zone."""
    if ts is None:
        return "soon"
    tz = ZoneInfo(timezone_name or "America/New_York")
    dt = datetime.fromtimestamp(ts, tz)
    now_local = datetime.fromtimestamp(time.time(), tz)
    clock = dt.strftime("%-I:%M %p")
    if dt.date() != now_local.date():
        return f"{dt.strftime('%a %b %-d')}, {clock}"
    return clock


_COUNT_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
               7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def _count_word(n: int) -> str:
    return _COUNT_WORDS.get(n, str(n))


def _last_result_phrase(last: Dict[str, Any]) -> str:
    if last.get("outcome") == "complete" and not last.get("changed"):
        return "found no change"
    rl = str(last.get("result_line") or "").strip()
    return rl if rl else "ran"


def _schedule_line(store: Any, schedule: Schedule) -> str:
    try:
        cadence_text = cadence_module.describe(schedule.cadence_kind, schedule.cadence, schedule.timezone)
    except ValueError:
        cadence_text = "no timing set yet"  # a row migrated from the old self-scheduling kind
    bits = [f"{schedule.title}, {cadence_text}"]
    last = store.last_schedule_run(schedule.id)
    if last is not None and last.get("finished_at") is not None:
        when = _clock_text(last["finished_at"], schedule.timezone)
        bits.append(f"last check {when} {_last_result_phrase(last)}")
    if schedule.state == "active" and schedule.next_run is not None:
        bits.append(f"next {_clock_text(schedule.next_run, schedule.timezone)}")
    elif schedule.state == "paused":
        bits.append("paused")
    elif schedule.state == "proposed":
        bits.append("waiting on your yes")
    return ", ".join(bits) + "."


def _list_sentence(store: Any) -> str:
    rows = store.schedules()
    if not rows:
        return "Nothing is scheduled."
    live = [s for s in rows if s.state != "ended"]
    ended = [s for s in rows if s.state == "ended"]
    sentences: List[str] = []
    if live:
        sentences.append(f"{_count_word(len(live))} thing{'s' if len(live) != 1 else ''}.")
        sentences.extend(_schedule_line(store, s) for s in live)
    if ended:
        titles = ", ".join(s.title for s in ended)
        sentences.append(f"{titles} {'is' if len(ended) == 1 else 'are'} stopped.")
    return " ".join(sentences) if sentences else "Nothing is scheduled."


def _last_sentence(store: Any, schedule: Schedule) -> str:
    runs = store.schedule_runs(schedule.id, limit=2)  # newest first
    if not runs:
        return f"{schedule.title} has not run yet."
    last = runs[0]
    when = _clock_text(last.get("finished_at") or last.get("started_at"), schedule.timezone)
    outcome = last.get("outcome")
    if outcome == "complete":
        line = f"{when}: {last.get('result_line') or 'all clear'}."
    elif outcome == "incomplete":
        reason = last.get("result_line") or "it was missing a capability it needed"
        line = f"{when}: could not finish. {reason}"
        if not line.endswith("."):
            line += "."
    else:
        reason = last.get("result_line") or last.get("final_status") or "it failed"
        line = f"{when}: could not finish: {reason}."
    if len(runs) > 1:
        prev = runs[1]
        if (prev.get("result_line"), prev.get("note")) != (last.get("result_line"), last.get("note")):
            prev_when = _clock_text(prev.get("finished_at") or prev.get("started_at"), schedule.timezone)
            line += f" Before that, {prev_when}: {prev.get('result_line') or 'no result recorded'}."
    return line


# ---------------------------------------------------------------------------
# handle(): the router's schedule_command -> a plain reply
# ---------------------------------------------------------------------------

def _state_error_reply(schedule: Schedule, action: str) -> str:
    """A plain sentence for an action the schedule's current state refuses --
    never the raw "<action> is not valid for a schedule that is <state>
    (valid actions: ...)" message (item L)."""
    if schedule.state == "proposed":
        return f"{schedule.title} has not started yet; say yes or no first."
    if schedule.state == "paused" and action == "pause":
        return f"{schedule.title} is already paused."
    if schedule.state == "active" and action == "resume":
        return f"{schedule.title} is already running."
    if schedule.state == "ended":
        return f'{schedule.title} has ended. Say "start it again" to bring it back.'
    return f"{schedule.title} can't do that right now."


def _friendly_action_error(schedule: Schedule, exc: ValueError) -> str:
    """Map a service ValueError from schedules.action/edit to a plain owner
    sentence. Timing errors (CadenceError) never say "cadence"; a missing
    capability is named in words, not as "scope"."""
    msg = str(exc)
    if isinstance(exc, cadence_module.CadenceError):
        if "at least 5 minutes" in msg:
            return "It has to run at least every 5 minutes."
        return "I didn't understand that timing."
    if msg.startswith("missing capability:"):
        return f"It isn't allowed to {msg.split(':', 1)[1].strip()} yet."
    # "already has a run in flight", "needs to be told how often to run
    # first", "That time has already passed." -- already plain sentences.
    return msg


async def _handle_simple_action(manager: Any, schedule: Schedule, action: str) -> str:
    if action == "stop" and schedule.state == "proposed":
        # A proposal has never run; "stop it" on one means the same as "no" --
        # reject it rather than report it as already stopped (item G).
        await manager.schedules.decide(schedule.id, False)
        return ""
    real_action = action
    if action == "resume" and schedule.state == "ended":
        real_action = "start_again"  # a resumed ended schedule has to restart, not just unpause
    valid = manager.schedules.valid_actions(schedule)
    if real_action not in valid:
        return _state_error_reply(schedule, action)
    try:
        updated = await manager.schedules.action(schedule.id, real_action, owner=True)
    except ValueError as exc:
        return _friendly_action_error(schedule, exc)
    if action == "pause":
        return f"Paused {updated.title}."
    if action == "resume":
        when = _clock_text(updated.next_run, updated.timezone)
        return f"{updated.title} is running again, next check {when}."
    if action == "run_now":
        return f"Running {updated.title} now."
    if action == "stop":
        return f'Stopped {updated.title}. Say "start it again" if you want it back.'
    return f"{updated.title} updated."


async def _handle_change(manager: Any, schedule: Schedule, changes: Dict[str, Any]) -> str:
    was_proposed = schedule.state == "proposed"
    edit_changes: Dict[str, Any] = {}
    for key in ("cadence_kind", "cadence", "report_every_time", "title"):
        if key in changes:
            edit_changes[key] = changes[key]
    if "until" in changes:
        # The router hands until back as an ISO date/datetime string in the
        # schedule's own time zone; the store column is a numeric epoch. Reuse
        # manager's own parser (the same one schedule creation uses) rather
        # than duplicate it, so the two paths cannot drift.
        from . import manager as manager_module
        try:
            edit_changes["until"] = manager_module._parse_until(changes["until"], schedule.timezone)
        except manager_module.UntilParseError as exc:
            return str(exc)
    if changes.get("prompt_addition"):
        edit_changes["prompt"] = (schedule.prompt + " " + str(changes["prompt_addition"]).strip()).strip()
    added_scopes: List[str] = []
    if changes.get("scopes"):
        added_scopes = [s for s in changes["scopes"] if s not in schedule.scopes]
        if added_scopes:
            edit_changes["scopes"] = list(schedule.scopes) + added_scopes
    if not edit_changes:
        return "I don't see a change to make there."
    try:
        result = await manager.schedules.edit(schedule.id, owner=True, changes=edit_changes)
    except ValueError as exc:
        return _friendly_action_error(schedule, exc)
    updated = result["schedule"]
    if result["needs_approval"]:
        display_changes = dict(edit_changes)
        if added_scopes:
            display_changes["scopes"] = added_scopes
        return schedprop.format_change(updated, display_changes)
    if was_proposed:
        # Revising a proposal that has not started yet resends the whole thing
        # for a fresh yes, rather than announcing an "applied" change.
        return schedprop.format_proposal(updated)
    cadence_text = cadence_module.describe(updated.cadence_kind, updated.cadence, updated.timezone)
    when = _clock_text(updated.next_run, updated.timezone)
    return f"Changed. {updated.title} now runs {cadence_text}, next {when}."


async def _handle_add_capability(manager: Any, schedule: Schedule, changes: Dict[str, Any],
                                 said_text: str) -> str:
    # The router's own changes["scopes"] suggestion is NEVER granted here --
    # that would make whatever the model lists permission at once (item D).
    # Only his own permission words, optionally narrowed to what the last run
    # was actually denied, can widen an already-active schedule's scopes.
    store = manager.store
    last = store.last_schedule_run(schedule.id)
    denied_names: List[str] = []
    if last:
        for item in last.get("denied") or []:
            tool = item.get("tool") if isinstance(item, dict) else item
            for name in manager.schedules._scopes_for_tool(str(tool)):
                if name not in denied_names:
                    denied_names.append(name)
    mapped = schedprop.capabilities_from_permission(said_text or "", denied_names)
    added = [s for s in mapped if s not in schedule.scopes]
    if not added:
        if denied_names:
            return f"What should it be allowed to do: {schedprop.labels(sorted(denied_names))}?"
        return "What should it be allowed to do?"
    new_scopes = list(schedule.scopes) + added
    updated = store.update_schedule(schedule.id, scopes=new_scopes)
    reply = f"Done. It can now {schedprop.labels(added)}."
    last = store.last_schedule_run(schedule.id)
    if last is not None and last.get("outcome") in ("incomplete", "failed") and \
            "run_now" in manager.schedules.valid_actions(updated):
        reply += " Running the check again now."
        await manager.schedules.action(updated.id, "run_now", owner=True)
    return reply


async def handle(manager: Any, session: Any, command: Any) -> str:
    """The plain reply to a schedule_command: list/last/change/pause/resume/
    run_now/stop/add_capability. Only the owner may manage schedules; anyone
    else gets the same refusal a request they are not allowed to make would
    get today."""
    if not session.is_approver():
        return "Only Aaron can manage schedules."
    store = manager.store
    action = str(_field(command, "action") or "").strip().lower()
    if action == "list":
        return _list_sentence(store)
    target_text = _field(command, "target")
    changes = dict(_field(command, "changes") or {})
    chat_id = getattr(session, "record_id", None) or getattr(session, "chat_id", "")
    reply_to = getattr(session, "inbound_id", None)
    found = resolve(store, target_text, chat_id, reply_to)
    if isinstance(found, list):
        if not found:
            return "I don't have a schedule like that. " + _list_sentence(store)
        titles = " or ".join(f'"{s.title}"' for s in found)
        return f"Which one: {titles}?"
    if found is None:
        return "I don't have a schedule like that. " + _list_sentence(store)
    schedule = found
    if action == "last":
        return _last_sentence(store, schedule)
    if action == "change":
        return await _handle_change(manager, schedule, changes)
    if action == "add_capability":
        said_text = _message_text(store, reply_to)
        return await _handle_add_capability(manager, schedule, changes, said_text)
    if action == "approve":
        await manager.schedules.decide(schedule.id, True)
        return ""
    if action == "reject":
        await manager.schedules.decide(schedule.id, False)
        return ""
    if action in {"pause", "resume", "run_now", "stop"}:
        return await _handle_simple_action(manager, schedule, action)
    return "I don't understand that schedule command."
