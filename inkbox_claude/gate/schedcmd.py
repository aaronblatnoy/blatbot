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
from . import overview as overview_mod
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


# ---------------------------------------------------------------------------
# Answering a bare yes/no before the router ever sees the message
# ---------------------------------------------------------------------------

async def answer_proposal(manager: Any, session: Any, text: str) -> Optional[str]:
    """If `text` is a bare yes/no and exactly one schedule proposal is open on
    this thread, answer it through ScheduleService.decide and return the reply
    to send (empty string when decide() already sent its own notice and there
    is nothing more to say). Returns None when nothing here applies at all --
    text is not a plain yes/no, or no proposal is open on this thread -- so
    the caller falls through to the router.

    A proposal IS open on this thread but fails the latest-outbound-or-15-
    minutes window (item 5): never falls through silently in that case --
    too much may have happened since for a bare yes/no to mean it, but he is
    still owed an answer, not nothing. Instead this asks him to say so
    explicitly ("yes, start it" / "no, drop it"), which schedcmd.resolve()
    then finds as the open proposal on the thread even with no target named.

    Call this only AFTER checking that no pending request on Aaron's plate
    already claims this reply; that precedence is the caller's job, not this
    function's."""
    yes, no = is_yes(text), is_no(text)
    if not yes and not no:
        return None
    chat_id = getattr(session, "record_id", None) or getattr(session, "chat_id", "")
    all_open = _open_proposals(manager.store, chat_id)
    if not all_open:
        return None
    proposals = [p for p in all_open if _captures_bare_answer(manager.store, chat_id, p)]
    if not proposals:
        if len(all_open) == 1:
            p = all_open[0]
            if yes:
                return f'Do you mean start "{p.title}"? Say "yes, start it".'
            return f'Do you mean drop "{p.title}"? Say "no, drop it".'
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
    # With no target named, an open proposal on this thread wins before the
    # 12-hour recent-notice rule: "yes, start it"/"no, drop it" (and any other
    # untargeted approve/reject) should resolve to the thing he is actually
    # being asked about, not to whatever else was last mentioned.
    open_props = _open_proposals(store, chat_id)
    if len(open_props) == 1:
        return open_props[0]
    if len(open_props) > 1:
        return open_props
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
    """The last run's outcome in plain words -- never "Waiting", and never
    the raw outcome string (finished / could not finish / could not run /
    failed). Detail (the actual result sentence) belongs to the "last"
    question (_last_sentence), not this one-line summary."""
    outcome = last.get("outcome")
    if outcome == "complete":
        return "found no change" if not last.get("changed") else "finished"
    if outcome == "incomplete":
        return "could not finish"
    if outcome == "blocked":
        return "could not run"
    if outcome == "failed":
        return "failed"
    return str(last.get("result_line") or "").strip() or "ran"


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
        reason = (schedule.last_outcome or "").strip()
        bits.append(f"paused: {reason}" if reason else "paused")
    elif schedule.state == "proposed":
        bits.append("not started yet")
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
    sentence. The mapping itself (schedprop.friendly_error) is shared with
    manager.py's proposal-time errors, so an unknown scope or an unparseable
    cadence is worded the same way no matter which path hit it."""
    return schedprop.friendly_error(exc)


async def _handle_simple_action(manager: Any, schedule: Schedule, action: str) -> str:
    if action == "stop" and schedule.state == "proposed":
        # A proposal has never run; "stop it" on one means the same as "no" --
        # reject it rather than report it as already stopped (item G).
        await manager.schedules.decide(schedule.id, False)
        return ""
    real_action = action
    if action == "resume" and schedule.state == "ended":
        real_action = "start_again"  # a resumed ended schedule has to restart, not just unpause
    if action == "run_now" and schedule.state == "paused":
        # "run it now"/"try it now"/"try again" after a blocked run's fix: a
        # paused schedule's valid_actions never includes run_now on its own, but
        # this is exactly what answering a blocked run's question means -- resume
        # it first (same cadence it already has), then fire it immediately. If it
        # is blocked again, the existing three-strikes/blocked-outcome handling
        # pauses it again with its own notice; this reply is only the first ack.
        try:
            schedule = await manager.schedules.action(schedule.id, "resume", owner=True)
        except ValueError as exc:
            return _friendly_action_error(schedule, exc)
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
    # A bare reply that is exactly one of the labels the refusal just offered
    # ("What should it be allowed to do: <labels>?") grants that scope on its
    # own, with no permission phrase needed. The last run having been refused
    # that scope is what makes the prior question (and this answer) about it.
    bare = _norm(said_text or "")
    label_match = [n for n in denied_names if _norm(schedprop.label(n)) == bare]
    mapped = label_match or schedprop.capabilities_from_permission(said_text or "", denied_names)
    added = [s for s in mapped if s not in schedule.scopes]
    if not added:
        if last is not None and last.get("outcome") == "blocked" and schedule.state == "paused":
            # Not a permission grant -- he is telling it where the thing is
            # or what to change ("it's the Telegram group called X"). Record
            # it for the next run and wait for an explicit "run it now"; a
            # blocked schedule never auto-runs off a reply alone (item 3).
            note_text = (said_text or "").strip()
            addition = f"The owner says: {note_text}" if note_text else "The owner replied, but said nothing."
            existing_note = (schedule.last_note or "").strip()
            new_note = f"{existing_note}\n{addition}" if existing_note else addition
            store.update_schedule(schedule.id, last_note=new_note)
            return 'Got it. Say "run it now" to try with that.'
        if denied_names:
            return f"What should it be allowed to do: {schedprop.labels(sorted(denied_names))}?"
        return "What should it be allowed to do?"
    new_scopes = list(schedule.scopes) + added
    updated = store.update_schedule(schedule.id, scopes=new_scopes)
    last = store.last_schedule_run(schedule.id)
    # A blocked or incomplete schedule's last run is exactly what granting
    # this capability was meant to fix: resume it (a blocked schedule sits
    # "paused") and run it right now, so he gets one further message -- the
    # run's own result or a fresh notice -- instead of this "Done..." reply
    # plus a second message once the run lands. No rerun follows (nothing
    # was actually blocking it, or it is not in a runnable state) -> this
    # reply is the only thing he gets, so it says what changed.
    should_rerun = last is not None and last.get("outcome") in ("incomplete", "blocked", "failed")
    if should_rerun:
        valid = manager.schedules.valid_actions(updated)
        if updated.state == "paused" and "resume" in valid:
            updated = await manager.schedules.action(updated.id, "resume", owner=True)
            valid = manager.schedules.valid_actions(updated)
        if "run_now" in valid:
            await manager.schedules.action(updated.id, "run_now", owner=True)
            return ""
    return f"Done. It can now {schedprop.labels(added)}."



# ---------------------------------------------------------------------------
# status / stuck / today (item 2): plain answers to "what are you doing",
# built from the SAME row builders console/api.py's /console/api/overview
# uses (gate/overview.py) -- no model, no ids, no scope names, no markdown,
# and never a second, independently-guessed diagnosis of the same schedule.
# A small fixed phrase list is matched whole-message before the router ever
# runs (manager.py's decide()); the router can also still route here via
# schedule_command for a differently-worded question.
# ---------------------------------------------------------------------------

_FIXED_PHRASES = {
    "what are you working on": "status",
    "what are you doing": "status",
    "anything running": "status",
    "what's stuck": "stuck",
    "whats stuck": "stuck",
    "what needs me": "stuck",
    "what's waiting on me": "stuck",
    "whats waiting on me": "stuck",
    "what did you do today": "today",
    "what happened today": "today",
    "repair it": "repair",
}


def match_fixed_phrase(text: str) -> Optional[str]:
    """One of "status"/"stuck"/"today"/"repair" when `text` is (whole-message,
    case-insensitive, question mark optional) one of Aaron's own common exact
    phrasings for these -- matched here, before the router ever sees the
    message, so the answer never depends on the model. Named forms ("repair
    the pledge one", "let the donor one do that") are not here: they carry a
    schedule name, so they go to the router, which resolve() then matches the
    same way it already matches any other targeted schedule command."""
    norm = _norm(text).rstrip("?")
    return _FIXED_PHRASES.get(norm)


_NAMING_FILLER_WORDS = {
    "the", "a", "an", "one", "that", "this", "it", "thing", "schedule",
    "job", "check", "monitor", "my", "please", "and", "for", "from", "with",
    "your", "you", "request",
}


def _title_words(title: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", (title or "").lower()) if w]


def _distinguishing_word(title: str, other_titles: List[str]) -> Optional[str]:
    """A word of 4 or more letters in `title` that appears in no other title
    in `other_titles` and is not a filler word -- the one word code can use
    to tell this stuck thing apart from the rest of the reply without
    quoting its whole title. None when no such word exists."""
    other_words: set = set()
    for other in other_titles:
        other_words.update(_title_words(other))
    for word in _title_words(title):
        if len(word) >= 4 and word not in _NAMING_FILLER_WORDS and word not in other_words:
            return word
    return None


def _named_say(say: str, word: str) -> str:
    """`say` reworded to name which stuck thing it answers by its
    distinguishing word ("repair it" -> "repair the pledge one"; "let it
    edit contacts" -> "let the donor one edit contacts"; "let it do that" ->
    "let the donor one do that"; "try again" -> "try the tracker one
    again")."""
    if say == "repair it":
        return f"repair the {word} one"
    if say.startswith("let it "):
        return f"let the {word} one " + say[len("let it "):]
    if say == "try again":
        return f"try the {word} one again"
    return say


def _named_say_by_title(say: str, title: str) -> str:
    """`say` reworded to name a stuck thing by its full title in quotes, used
    when no single word distinguishes it from the rest of the reply (item 4:
    never "the the", never an empty quoted phrase)."""
    if say == "repair it":
        return f'repair "{title}"'
    if say.startswith("let it "):
        return f'let "{title}" ' + say[len("let it "):]
    if say == "try again":
        return f'try "{title}" again'
    return say


def status_reply(manager: Any) -> str:
    store = manager.store
    running = overview_mod.running_rows(store)
    nxt = overview_mod.next_scheduled_row(store)
    if running:
        parts = [f"{row['sentence']} (started {row['started_text']})" for row in running]
        reply = f"{_count_word(len(parts))} thing{'s' if len(parts) != 1 else ''} running. " + " ".join(parts)
    else:
        reply = "Nothing is running."
    if nxt:
        reply += f" Next: {nxt['title']}, {nxt['next_run_text']}."
    return reply


def _stuck_sayable(manager: Any) -> List[Dict[str, Any]]:
    """Stuck rows a chat reply can say anything about at all: every row except
    a plain approval/proposal (kind "approval"), which already has its own
    yes/no mechanism, so is left to "list"/the open-proposal flow instead. A
    collapsed usage-limit group has no say (nothing can be said to fix a
    session limit) but still belongs in the reply -- it is kind
    "request_cluster", not "approval"."""
    rows = overview_mod.stuck_rows(manager.store, manager.schedules)
    return [r for r in rows if r.get("say") or r.get("kind") == "request_cluster"]


def _stuck_tail(item: Dict[str, Any]) -> str:
    """The "Say ..." phrase that follows one stuck item's sentence, or "" when
    nothing can be said about it at all (a collapsed usage-limit group). A
    one-off request can never be "stopped" -- only a schedule can -- so a
    request's tail never offers "stop it"."""
    say = item.get("say")
    stoppable = item.get("kind") != "request" and item.get("kind") != "request_cluster"
    if not say:
        return ""
    if say == "stop it":
        return f'Say "{say}".'
    if stoppable:
        return f'Say "{say}" or "stop it".'
    return f'Say "{say}".'


def _stuck_tail_named(item: Dict[str, Any], other_titles: List[str]) -> str:
    """Item 4's naming rule: a distinguishing word of 4+ letters not shared
    with any other title shown in this reply, or -- when none exists -- the
    full title in quotes. Never "the the", never an empty quoted phrase, and
    never "stop the <word> one" for a request (see _stuck_tail)."""
    say = item.get("say")
    stoppable = item.get("kind") != "request" and item.get("kind") != "request_cluster"
    if not say:
        return ""
    word = _distinguishing_word(item["title"], other_titles)
    if word:
        named_say = _named_say(say, word)
        stop_phrase = f'stop the {word} one'
    else:
        named_say = _named_say_by_title(say, item["title"])
        stop_phrase = f'stop "{item["title"]}"'
    if say == "stop it":
        return f'Say "{named_say}".'
    if stoppable:
        return f'Say "{named_say}" or "{stop_phrase}".'
    return f'Say "{named_say}".'


_STUCK_REPLY_SHOWN = 3


def stuck_reply(manager: Any) -> str:
    items = _stuck_sayable(manager)
    if not items:
        return "Nothing is stuck."
    if len(items) == 1:
        it = items[0]
        tail = _stuck_tail(it)
        sentence = f"{it['sentence']} {tail}".strip() if tail else it["sentence"]
        return f"One thing is stuck. {sentence}"
    shown = items[:_STUCK_REPLY_SHOWN]
    all_titles = [it["title"] for it in shown]
    lines = []
    for i, it in enumerate(shown):
        other_titles = all_titles[:i] + all_titles[i + 1:]
        tail = _stuck_tail_named(it, other_titles)
        line = f"{it['sentence']} {tail}".strip() if tail else it["sentence"]
        lines.append(line)
    reply = f"{_count_word(len(items))} things are stuck. " + "\n".join(lines)
    if len(items) > _STUCK_REPLY_SHOWN:
        reply += f"\nAnd {len(items) - _STUCK_REPLY_SHOWN} more on the console."
    return reply


def today_reply(manager: Any) -> str:
    items = overview_mod.done_today_rows(manager.store)
    if not items:
        return "Nothing finished today."
    return " ".join(it["sentence"] for it in items)


async def _handle_repair(manager: Any, schedule: Schedule, changes: Dict[str, Any]) -> str:
    """"repair it"/"repair the <title> one"/"fix the <title> one": apply
    ScheduleService.repair_plan() for this schedule. A timing already in
    `changes` (the router filling it in from the same message, "repair it,
    every 4 hours") or recovered by the plan itself is applied at once, same
    as repair_apply always does; with neither, this asks how often it should
    run instead of guessing, and tags that question to the schedule (via a
    schedule notice, not the generic reply) so a bare timing answered back on
    this thread ("every 4 hours") resolves to it the same way a reply to any
    other schedule notice does."""
    plan = manager.schedules.repair_plan(schedule.id)
    if not plan["needed"]:
        return f"{schedule.title} does not need repair."
    timing = None
    if "cadence_kind" in changes and "cadence" in changes:
        timing = {"cadence_kind": changes["cadence_kind"], "cadence": changes["cadence"]}
    if plan["timing"] is None and timing is None:
        await manager.send_schedule_notice(
            schedule, f"How often should {schedule.title} run? For example, every 4 hours.")
        return ""
    try:
        result = await manager.schedules.repair_apply(schedule.id, owner=True, timing=timing)
    except ValueError as exc:
        return _friendly_action_error(schedule, exc)
    return result["message"]


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
    if action == "status":
        return status_reply(manager)
    if action == "stuck":
        return stuck_reply(manager)
    if action == "today":
        return today_reply(manager)
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
    if action == "repair":
        return await _handle_repair(manager, schedule, changes)
    if action == "approve":
        await manager.schedules.decide(schedule.id, True)
        return ""
    if action == "reject":
        await manager.schedules.decide(schedule.id, False)
        return ""
    if action in {"pause", "resume", "run_now", "stop"}:
        return await _handle_simple_action(manager, schedule, action)
    return "I don't understand that schedule command."
