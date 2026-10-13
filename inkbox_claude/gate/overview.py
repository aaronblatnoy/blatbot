"""The owner's Home overview, shared between the console (/console/api/overview)
and chat (schedcmd.status/stuck/today): one set of row builders so the two
surfaces can never disagree about what is running, what is stuck, or what
finished today.

Each stuck row carries a "say" field: the one short phrase (never a scope
name, never an id) that fixes it back in chat, or None when only the console
can fix it (an approval/proposal already has its own yes/no). The console
ignores "say"; schedcmd uses it to build the chat reply, and schedcmd names
each thing by a plain word from its title when more than one stuck thing
needs its own phrase.

Nothing here reads raw first-person run text into a sentence without going
through sched_mod._normalize_reason or wording.failure_in_words -- a run's
own "I could not find..." never reaches the owner as the scheduler's own
words, on either surface."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from . import schedprop, schedules as sched_mod, wording
from .schedules import DEFAULT_TIMEZONE
from .store import Request, Store

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def _schedule_time_text(ts: Optional[float], timezone_name: str) -> Optional[str]:
    """"Mon Oct 12, 2:33 AM EDT" in the schedule's own time zone. None in, None out."""
    if ts is None:
        return None
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        tz = ZoneInfo(timezone_name or DEFAULT_TIMEZONE)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)
    return datetime.fromtimestamp(ts, tz).strftime("%a %b %-d, %-I:%M %p %Z")


def _cadence_text(schedule: Any) -> str:
    from . import cadence as cadence_mod
    try:
        return cadence_mod.describe(schedule.cadence_kind, schedule.cadence, schedule.timezone)
    except cadence_mod.CadenceError:
        return "No timing set yet"  # a legacy/paused row with no cadence model yet


def _last_non_skipped_run(store: Store, schedule_id: int) -> Optional[Dict[str, Any]]:
    for row in store.schedule_runs(schedule_id):
        if row.get("outcome") != "skipped":
            return row
    return None


def _redact_contacts(text: str, store: Store) -> str:
    """Replace a phone number or email address in `text` with the person's name,
    when the store can resolve that contact. Anything it cannot resolve (a
    stranger's number, a made-up test value) is left exactly as it was."""
    if not text:
        return text

    def _sub(match: Any) -> str:
        value = match.group(0)
        try:
            kind = store.guess_contact_kind(value)
            contact = store.find_contact(kind, value)
        except Exception:
            contact = None
        if contact and contact.get("person"):
            return str(contact["person"])
        return value

    return _PHONE_RE.sub(_sub, _EMAIL_RE.sub(_sub, text))


def _first_sentence(text: str) -> str:
    for piece in _SENTENCE_SPLIT_RE.split((text or "").strip()):
        if piece.strip():
            return piece.strip()
    return ""


def _looks_like_bare_identifier(text: str) -> bool:
    """A sender "name" that is really just a phone number or an email address --
    never shown to the owner as who is asking."""
    t = (text or "").strip()
    if not t:
        return True
    return bool(_EMAIL_RE.fullmatch(t) or _PHONE_RE.fullmatch(t))


def _request_who(request: Request, store: Store) -> str:
    """A real name for whoever this request is from, or "" when all the store has
    is a bare phone number or email -- callers drop the "who said this" framing
    entirely rather than ever show that raw identifier."""
    for candidate in (request.sender_name, request.sender):
        name = _redact_contacts((candidate or "").strip(), store)
        if name and not _looks_like_bare_identifier(name):
            return name
    return ""


def _request_title_and_plain(store: Store, request: Request) -> "tuple[str, str]":
    """(title, plain) for a request, from wording.plain_title: never the voice
    hand-off's internal preamble, a "blatbot:"/"caller:" line, a phone number,
    or text cut by length (see gate/wording.py)."""
    plain = _redact_contacts(wording.plain_title(store, request), store)
    title = _first_sentence(plain) or plain or "A request"
    return title, plain


def _request_result_text(request: Request) -> str:
    """What happened on a finished request, for Done today: its own result text
    if the executor recorded one, else its plain ask (there is nothing better
    to show)."""
    status = request.status or {}
    text = str(status.get("summary") or status.get("error") or request.raw_output or "").strip()
    return _first_sentence(text) or text


def _request_row(request: Request, store: Store, *, kind: str, since_field: str,
                  sentence: str, title: str, actions: List[Dict[str, Any]],
                  say: Optional[str] = None) -> Dict[str, Any]:
    ts = request.updated_at if since_field == "updated_at" else request.created_at
    return {
        "kind": kind, "id": request.id, "title": title, "sentence": sentence,
        since_field: _schedule_time_text(ts, DEFAULT_TIMEZONE), "actions": actions, "say": say,
    }


def _stuck_pending_requests(store: Store) -> List[Dict[str, Any]]:
    out = []
    for request in store.pending():
        if request.schedule_id:
            continue  # a schedule's own run, represented by its Stuck schedule row
        who = _request_who(request, store)
        title, plain = _request_title_and_plain(store, request)
        if plain:
            sentence = f"{who} is asking: {plain}" if who else f"Waiting on your yes: {plain}."
        else:
            sentence = f"{who} is waiting on your yes." if who else "Waiting on your yes."
        actions = [
            {"action": "approve", "label": "Approve",
             "target": {"endpoint": "requests/decide", "id": request.id, "decision": "yes"}},
            {"action": "decline", "label": "Decline",
             "target": {"endpoint": "requests/decide", "id": request.id, "decision": "no"}},
        ]
        # An approval already has its own yes/no mechanism (a bare reply on this
        # thread) -- "say" is only for the phrases status/stuck/today themselves
        # teach, which that mechanism is not.
        out.append(_request_row(request, store, kind="approval", since_field="since_text",
                                 sentence=sentence, title=title, actions=actions, say=None))
    return out


_USAGE_LIMIT_DEFAULT_EXPIRY_HOURS = 2.0


def _failed_request_candidates(store: Store, now: float) -> List[Dict[str, Any]]:
    """Non-usage-limit failed one-off requests still worth their own Stuck row:
    failed within the last 6 hours, and either refused a permission or are the
    most recent request on their own thread. A usage-limit failure is never
    reported here -- it is never an individual row at all, see
    _usage_limit_groups/_usage_limit_cluster_rows below. Not a schedule's own
    run either (that is represented by the schedule's Stuck row instead)."""
    cutoff = now - 6 * 3600.0
    recent = store.recent_requests(limit=500)
    latest_created_by_chat: Dict[str, float] = {}
    for request in recent:
        if not request.chat_id:
            continue
        prior = latest_created_by_chat.get(request.chat_id)
        if prior is None or request.created_at > prior:
            latest_created_by_chat[request.chat_id] = request.created_at
    out = []
    for request in recent:
        if request.schedule_id or request.state != "failed" or request.updated_at < cutoff:
            continue
        status = request.status or {}
        if wording.failure_category(status) == "usage_limit":
            continue
        is_latest_on_thread = bool(request.chat_id) and request.created_at >= latest_created_by_chat.get(
            request.chat_id, request.created_at)
        was_refused = bool(status.get("denied"))
        if not (was_refused or is_latest_on_thread):
            continue
        title, _plain = _request_title_and_plain(store, request)
        out.append({"request": request, "title": title})
    return out


def _stuck_failed_requests(store: Store, now: float) -> List[Dict[str, Any]]:
    """One row per still-actionable failed ask (see _failed_request_candidates).
    The sentence is always "<plain title>: <why, in plain words>." --
    wording.failure_in_words never returns the raw error string or a run's own
    first-person text, so this is already clean on both surfaces. "say" is
    "try again": the one phrase a failed one-off task can actually be retried
    with."""
    out = []
    for item in _failed_request_candidates(store, now):
        request, title = item["request"], item["title"]
        reason = wording.failure_in_words(request.status or {})
        sentence = f"{title}: {reason}."
        actions = [{"action": "retry", "label": "Try again",
                    "target": {"endpoint": "requests/retry", "id": request.id}}]
        out.append(_request_row(request, store, kind="request", since_field="since_text",
                                 sentence=sentence, title=title, actions=actions, say="try again"))
    return out


def _usage_limit_groups(store: Store) -> List[Dict[str, Any]]:
    """Failed one-off requests whose own thread hit a usage limit, grouped by
    thread and by the clock hour they failed in -- what collapses several
    failed requests from the same call into one Stuck/Done-today row instead
    of one per request. Each group carries its own expiry: the parsed reset
    time if the error named one, else _USAGE_LIMIT_DEFAULT_EXPIRY_HOURS after
    its latest member's failure -- past that, the group is no longer Stuck at
    all (it only still shows under Done today, as plain individual rows)."""
    recent = store.recent_requests(limit=500)
    groups: Dict[Any, List[Request]] = {}
    for request in recent:
        if request.schedule_id or request.state != "failed":
            continue
        if wording.failure_category(request.status or {}) != "usage_limit":
            continue
        bucket = int(request.created_at // 3600)
        groups.setdefault((request.chat_id, bucket), []).append(request)
    out = []
    for reqs in groups.values():
        reqs = sorted(reqs, key=lambda r: r.created_at)
        latest = reqs[-1]
        reset_text = wording.usage_limit_reset_text(latest.status or {})
        latest_failed_at = max(r.updated_at for r in reqs)
        expiry = _usage_limit_expiry(reset_text, latest_failed_at)
        out.append({"reqs": reqs, "reset_text": reset_text, "expiry": expiry})
    return out


def _usage_limit_expiry(reset_text: Optional[str], latest_failed_at: float) -> float:
    if reset_text:
        parsed = _parse_reset_epoch(reset_text, latest_failed_at)
        if parsed is not None:
            return parsed
    return latest_failed_at + _USAGE_LIMIT_DEFAULT_EXPIRY_HOURS * 3600.0


def _parse_reset_epoch(reset_text: str, ref_ts: float) -> Optional[float]:
    """The next occurrence (at or after `ref_ts`) of the clock time `reset_text`
    names, in whatever time zone it carries in parentheses (else the default
    zone). None when the text names no parseable clock time at all."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*\(([^)]*)\))?", reset_text.strip(), re.I)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    tzname = m.group(4) or DEFAULT_TIMEZONE
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    try:
        tz = ZoneInfo(tzname)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)
    ref = datetime.fromtimestamp(ref_ts, tz)
    candidate = ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if (ref - candidate).total_seconds() > 3600:
        candidate = candidate + timedelta(days=1)
    return candidate.timestamp()


def _usage_limit_cluster_rows(store: Store, now: float) -> List[Dict[str, Any]]:
    """One Stuck row per still-live usage-limit group: "<n> requests from the
    <time> call failed: it had hit a usage limit until <reset>." (or "...
    around <time> failed: ..." when the thread is not a phone call). No
    action and no say -- nothing can be said to fix a session limit."""
    out = []
    for group in _usage_limit_groups(store):
        if now > group["expiry"]:
            continue
        reqs = group["reqs"]
        latest = reqs[-1]
        is_call = wording.is_call_request(latest)
        when = wording.call_time_text(reqs[0].created_at)
        count = len(reqs)
        plural = "" if count == 1 else "s"
        clause = (f"it had hit a usage limit until {group['reset_text']}" if group["reset_text"]
                  else "it had hit a usage limit")
        if is_call and when:
            sentence = f"{count} request{plural} from the {when} call failed: {clause}."
        elif when:
            sentence = f"{count} request{plural} from around {when} failed: {clause}."
        else:
            sentence = f"{count} request{plural} failed: {clause}."
        out.append({
            "kind": "request_cluster", "id": latest.id, "title": wording.plain_title(store, latest),
            "sentence": sentence, "since_text": _schedule_time_text(latest.updated_at, DEFAULT_TIMEZONE),
            "actions": [], "say": None,
        })
    return out


def _usage_limit_stuck_request_ids(store: Store, now: float) -> "set[int]":
    ids: "set[int]" = set()
    for group in _usage_limit_groups(store):
        if now <= group["expiry"]:
            ids.update(r.id for r in group["reqs"])
    return ids


def _stuck_schedule_proposals(store: Store) -> List[Dict[str, Any]]:
    out = []
    for schedule in store.schedules(states=["proposed"]):
        sentence = f"{schedule.title} wants your yes to start: {_cadence_text(schedule)}."
        actions = [
            {"action": "approve", "label": "Approve",
             "target": {"endpoint": "schedules/approve", "id": schedule.id, "approve": True}},
            {"action": "reject", "label": "Reject",
             "target": {"endpoint": "schedules/approve", "id": schedule.id, "approve": False}},
        ]
        out.append({
            "kind": "approval", "id": schedule.id, "title": schedule.title, "sentence": sentence,
            "since_text": _schedule_time_text(schedule.created_at, schedule.timezone), "actions": actions,
            "say": None,  # a proposal already answers to a bare yes/no on its own thread
        })
    return out


def _schedule_fix_actions(schedule: Any, *, paused: bool) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    missing = schedprop.missing_capabilities(schedule.prompt, schedule.scopes)
    if missing:
        merged = list(dict.fromkeys(list(schedule.scopes) + missing))
        actions.append({
            "action": "add_capability", "label": f"Let it {schedprop.labels(missing)}",
            "target": {"endpoint": "schedules/edit", "id": schedule.id, "scopes": merged},
        })
    if paused:
        actions.append({"action": "resume", "label": "Resume",
                         "target": {"endpoint": "schedules/action", "id": schedule.id, "action": "resume"}})
    else:
        # Still active, merely had a failed/incomplete last run: offer to try it
        # again right now rather than only Stop, which is all an active schedule
        # used to be offered here.
        actions.append({"action": "run_now", "label": "Run now",
                         "target": {"endpoint": "schedules/action", "id": schedule.id, "action": "run_now"}})
    actions.append({"action": "stop", "label": "Stop",
                    "target": {"endpoint": "schedules/action", "id": schedule.id, "action": "stop"}})
    return actions


def _capability_say(missing: List[str]) -> Optional[str]:
    """The one short "let it ..." phrase a stuck schedule's missing capability
    can be fixed with in chat: the short label for exactly one, or the vague
    "let it do that" for several (schedprop.capabilities_from_permission
    already accepts that vague form, narrowed to what was actually denied)."""
    if not missing:
        return None
    if len(missing) == 1:
        return f"let it {schedprop.short_label(missing[0])}"
    return "let it do that"


def _stuck_schedules(store: Store, service: Any) -> List[Dict[str, Any]]:
    """Schedules paused with a reason, or active schedules whose last run did
    not complete (so the owner sees trouble before it gets to three strikes
    and pauses itself). Exactly one row per schedule: the reason always goes
    through sched_mod._normalize_reason, so a run's own first-person RESULT
    text ("I could not find ...") never shows up as the scheduler's words.
    A schedule repair_plan() finds something to fix for gets a "repair" action
    first, the plan's own sentence in place of the generic reason, and
    say="repair it" -- always checked before the plain missing-capability
    case, so a schedule that needs a full repair is never *also* reported as
    merely missing one capability (the two would disagree about the fix)."""
    out = []
    repair_action = lambda schedule: {"action": "repair", "label": "Repair and resume",
                                       "target": {"endpoint": "schedules/repair", "id": schedule.id}}
    for schedule in store.schedules(states=["paused"]):
        plan = service.repair_plan(schedule.id)
        if plan["needed"]:
            why = plan["sentence"]
            actions = [repair_action(schedule),
                       {"action": "stop", "label": "Stop",
                        "target": {"endpoint": "schedules/action", "id": schedule.id, "action": "stop"}}]
            out.append({
                "kind": "schedule", "id": schedule.id, "title": schedule.title,
                "sentence": f"{schedule.title}: {why}",
                "since_text": _schedule_time_text(schedule.updated_at, schedule.timezone), "actions": actions,
                "say": "repair it",
            })
            continue
        why = sched_mod._normalize_reason(schedule.last_outcome) if schedule.last_outcome else "it stopped running"
        missing = schedprop.missing_capabilities(schedule.prompt, schedule.scopes)
        actions = _schedule_fix_actions(schedule, paused=True)
        out.append({
            "kind": "schedule", "id": schedule.id, "title": schedule.title,
            "sentence": f"{schedule.title}: {why}.",
            "since_text": _schedule_time_text(schedule.updated_at, schedule.timezone), "actions": actions,
            "say": _capability_say(missing),
        })
    for schedule in store.schedules(states=["active"]):
        last = _last_non_skipped_run(store, schedule.id)
        if not last or last.get("outcome") == "complete":
            continue
        why = sched_mod._normalize_reason(last.get("result_line") or "")
        missing = schedprop.missing_capabilities(schedule.prompt, schedule.scopes)
        out.append({
            "kind": "schedule", "id": schedule.id, "title": schedule.title,
            "sentence": f"{schedule.title}: {why}.",
            "since_text": _schedule_time_text(last.get("finished_at") or last.get("started_at"), schedule.timezone),
            "actions": _schedule_fix_actions(schedule, paused=False),
            "say": _capability_say(missing),
        })
    return out


def _running_items(store: Store) -> List[Dict[str, Any]]:
    out = []
    for request in store.recent_requests(state="running", limit=50):
        schedule = store.get_schedule(request.schedule_id) if request.schedule_id else None
        status = request.status or {}
        latest_step = ""
        calls = status.get("tool_calls") or []
        if calls:
            last = calls[-1]
            latest_step = str(last.get("tool") or last.get("name") or "") if isinstance(last, dict) else str(last)
        if schedule is not None:
            title = schedule.title
            sentence = f"{schedule.title} is running now."
            kind = "schedule"
        else:
            who = _request_who(request, store)
            title, plain = _request_title_and_plain(store, request)
            if plain:
                sentence = f"{who}: {plain}" if who else plain
            else:
                sentence = f"{who}'s request is running." if who else "A request is running."
            kind = "request"
        out.append({
            "kind": kind, "id": request.id, "title": title, "sentence": sentence,
            "started_text": _schedule_time_text(request.created_at, DEFAULT_TIMEZONE),
            "latest_step": latest_step, "actions": [],
        })
    return out


def _midnight_epoch(timezone_name: str, now: float) -> float:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        tz = ZoneInfo(timezone_name or DEFAULT_TIMEZONE)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)
    local = datetime.fromtimestamp(now, tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


def _done_today_items(store: Store, now: float) -> List[Dict[str, Any]]:
    """Finished things, and things that failed with no fix available -- never an
    item that is also listed under Stuck. A schedule's non-complete run is
    always represented there instead (while it is active, or by the paused
    schedule once three strikes pause it), so only its complete runs land
    here; a failed request currently flagged Stuck is excluded by id, so only
    a failure outside the 24-hour window or already superseded shows up here,
    with the same never-raw-error wording Stuck uses. A complete schedule
    run's own result line is passed through _normalize_reason too, so its
    first-person RESULT text never shows up as the scheduler's own words."""
    cutoff = _midnight_epoch(DEFAULT_TIMEZONE, now)
    stuck_failed_ids = ({c["request"].id for c in _failed_request_candidates(store, now)}
                        | _usage_limit_stuck_request_ids(store, now))
    items: List[Dict[str, Any]] = []
    for request in store.recent_requests(limit=500):
        if request.schedule_id or request.state not in ("done", "failed", "rejected"):
            continue
        if request.updated_at < cutoff or request.id in stuck_failed_ids:
            continue
        who = _request_who(request, store)
        title, plain = _request_title_and_plain(store, request)
        if request.state == "done":
            outcome = "finished"
            result = _request_result_text(request) or plain or "Finished."
        elif request.state == "failed":
            outcome = "failed"
            result = wording.failure_in_words(request.status or {}) + "."
        else:
            outcome = "could_not_finish"
            result = _request_result_text(request) or "Declined."
        sentence = f"{who}: {result}" if who else f"{title}: {result}"
        items.append({
            "kind": "request", "id": request.id, "title": title, "sentence": sentence,
            "outcome": outcome, "time_text": _schedule_time_text(request.updated_at, DEFAULT_TIMEZONE),
            "actions": [], "_ts": request.updated_at,
        })
    for schedule in store.schedules():
        for run in store.schedule_runs(schedule.id):
            finished = run.get("finished_at")
            if finished is None or finished < _midnight_epoch(schedule.timezone, now):
                continue
            if run.get("outcome") != "complete":
                continue  # any other outcome is represented under Stuck instead, never here too
            result = sched_mod._normalize_reason(
                run.get("result_line") or run.get("final_status") or "")
            if not result or result == "it did not say why":
                result = "no result recorded"
            items.append({
                "kind": "schedule", "id": schedule.id, "title": schedule.title,
                "sentence": f"{schedule.title}: {result}.",
                "outcome": "finished", "time_text": _schedule_time_text(finished, schedule.timezone),
                "actions": [], "_ts": finished,
            })
    items.sort(key=lambda row: row["_ts"], reverse=True)
    for row in items:
        del row["_ts"]
    return items


def _coming_up_items(store: Store, now: float) -> List[Dict[str, Any]]:
    """The next few active schedules due to fire -- each flagged with
    last_run_failed when its last run did not complete, so one is never shown
    here looking perfectly healthy while it is also sitting in Stuck."""
    upcoming = sorted(
        (s for s in store.schedules(states=["active"]) if s.next_run and s.next_run >= now),
        key=lambda s: s.next_run,
    )[:3]
    out = []
    for s in upcoming:
        last = _last_non_skipped_run(store, s.id)
        last_run_failed = bool(last and last.get("outcome") not in ("complete", "skipped"))
        out.append({
            "id": s.id, "title": s.title, "cadence_text": _cadence_text(s),
            "next_run_text": _schedule_time_text(s.next_run, s.timezone),
            "next_run_in_seconds": s.next_run - now,
            "last_run_failed": last_run_failed,
            "note": "last run could not finish" if last_run_failed else None,
        })
    return out


def _overview_sources(store: Store) -> List[Dict[str, Any]]:
    requests = store.recent_requests(limit=1)
    schedule_rows = store.schedules()
    newest_run_ts: Optional[float] = None
    for schedule in schedule_rows:
        last = _last_non_skipped_run(store, schedule.id)
        if last and last.get("finished_at"):
            if newest_run_ts is None or last["finished_at"] > newest_run_ts:
                newest_run_ts = last["finished_at"]
    request_count = len(store.recent_requests(limit=100000))
    requests_detail = (
        f"{request_count} total, newest {_schedule_time_text(requests[0].updated_at, DEFAULT_TIMEZONE)}"
        if requests else f"{request_count} total, none yet"
    )
    schedules_detail = (
        f"{len(schedule_rows)} total, newest run {_schedule_time_text(newest_run_ts, DEFAULT_TIMEZONE)}"
        if newest_run_ts is not None else f"{len(schedule_rows)} total, no runs yet"
    )
    return [
        {"label": "Requests", "detail": requests_detail},
        {"label": "Schedules and their runs", "detail": schedules_detail},
    ]


def stuck_rows(store: Store, service: Any) -> List[Dict[str, Any]]:
    """Every stuck row, in the console's own order: schedules first (repair or
    plain), then pending approvals, then failed one-off asks, then collapsed
    usage-limit groups (no action, no say -- a session limit is never fixed
    by saying anything), then schedule proposals waiting on a yes."""
    now = time.time()
    return (
        _stuck_schedules(store, service) + _stuck_pending_requests(store)
        + _stuck_failed_requests(store, now) + _usage_limit_cluster_rows(store, now)
        + _stuck_schedule_proposals(store)
    )


def running_rows(store: Store) -> List[Dict[str, Any]]:
    return _running_items(store)


def done_today_rows(store: Store, now: Optional[float] = None) -> List[Dict[str, Any]]:
    return _done_today_items(store, time.time() if now is None else now)


def next_scheduled_row(store: Store, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    rows = _coming_up_items(store, time.time() if now is None else now)
    return rows[0] if rows else None


def overview(gateway: Any, store: Store) -> Dict[str, Any]:
    """Everything the Home page's four sections need, already sentenced,
    sorted and counted -- nothing is encoded in the front end beyond section
    headings and button chrome."""
    now = time.time()
    stuck = stuck_rows(store, gateway.sessions.schedules)
    running = _running_items(store)
    done_today = _done_today_items(store, now)
    coming_up = _coming_up_items(store, now)
    return {
        "stuck": stuck, "running": running, "done_today": done_today, "coming_up": coming_up,
        "counts": {"stuck": len(stuck), "running": len(running), "done_today": len(done_today)},
        "sources": _overview_sources(store),
    }
