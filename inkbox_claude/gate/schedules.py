"""Blatbot schedules, second version: the scheduler core.

A schedule is approved once, then fires itself on a cadence computed by code
(no model ever picks its own next run). Scope names, cron strings and "S<id>"
jargon never reach the owner here -- that is schedprop's and schedcmd's job;
this module calls through two thin seams (`_labels`, `_missing`, and
`_proposal_text`) so a prompt naming a tool outside its scopes is caught
before anything runs, and so notices read in plain words."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import cadence as cadence_module
from . import settings
from .scopes import READ_SCOPES, SCOPES, scope_for_tool, tools_for
from .store import Request, Schedule, sha256

logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "America/New_York"

# Actions schedule.action() itself handles. "edit", "approve" and "reject" are
# real actions the console/chat can take, but they go through edit()/decide()
# instead -- valid_actions() lists them too, so a caller sees the whole menu.
_ACTION_NAMES = {"pause", "resume", "run_now", "stop", "start_again"}
_ACTIONS_BY_STATE: Dict[str, set] = {
    "proposed": {"approve", "reject"},
    "active": {"pause", "stop", "run_now"},
    # A paused schedule also offers run_now (item: console parity with chat's
    # "run it now" on a paused schedule) -- action() resumes it first, same as
    # the chat path, rather than firing a run that will not actually happen.
    "paused": {"resume", "stop", "run_now"},
    "ended": {"start_again"},
}


def names_tool(prompt: str, tool: str) -> str:
    """The tool's short name (the part after the last `__`) when the prompt names it
    literally, else "". Short names that are ordinary single words ("logs", "service",
    "database") are never matched: a prompt using the word is not naming the tool."""
    short = tool.rsplit("__", 1)[-1]
    if "_" not in short and short.lower() == short:
        return ""
    return short if re.search(r"\b" + re.escape(short) + r"\b", prompt) else ""


def run_outcome(status: Dict[str, Any]) -> str:
    """complete / incomplete / blocked / failed, from the executor's status
    contract. A run that was denied a tool or had a call held is incomplete
    even when the model's own STATUS line said OK -- the model does not get
    the last word on whether its own run was complete.

    blocked is its own outcome, distinct from both: the run itself said so
    (STATUS: BLOCKED, parsed by schedrun.parse_closing into status["blocked"]),
    or every tool call it tried was refused and none at all succeeded --
    meaning the same capabilities it has now could never finish this job, so
    retrying on the current cadence would only fail identically forever.
    incomplete stays for a run that did part of its job (some tool calls
    denied/held, but at least one succeeded)."""
    if status.get("blocked"):
        return "blocked"
    if status.get("error"):
        return "failed"
    denied = status.get("denied") or []
    held = status.get("held")
    if denied and "tool_results" in status and not status.get("tool_results"):
        return "blocked"
    if denied or held:
        return "incomplete"
    if status.get("ok"):
        tool_results = status.get("tool_results")
        if tool_results and not any(r.get("ok") for r in tool_results):
            return "incomplete"
        if _looks_like_could_not(str(status.get("result_line") or "")):
            return "incomplete"
        return "complete"
    return "failed"


_COULD_NOT_PHRASES = (
    "i could not", "i couldn't", "could not", "couldn't",
    "i was unable to", "unable to", "i can't", "i cannot",
)


def _looks_like_could_not(result_line: str) -> bool:
    """True when a STATUS: OK run's own RESULT line opens by admitting it could not
    do the job (item C11): a status of ok with a result like this is not a genuine
    "nothing new" -- the model is reporting a failure under a false OK. Checked
    against the start of the text only, so a result that merely mentions "could not"
    partway through (describing something else) does not trip this."""
    text = (result_line or "").strip().lower()
    if not text:
        return False
    return any(text.startswith(p) for p in _COULD_NOT_PHRASES)


# Rewriting the run's own first-person "I" so it never sits next to the
# scheduler's own first-person "I" in the same notice (item 2). Checked
# longest-phrase-first so "i was unable to" is not shadowed by a shorter
# prefix that happens to also match.
_I_REWRITES = [
    ("i was unable to", "it was unable to"),
    ("i could not", "it could not"),
    ("i couldn't", "it couldn't"),
    ("i cannot", "it cannot"),
    ("i can't", "it can't"),
]

# First words common enough to be an ordinary sentence opener rather than a
# capitalized name ("Telegram group", "Aaron's calendar"): only these get
# lower-cased when normalize_reason otherwise would read oddly mid-sentence.
_LOWER_STARTERS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "there",
    "no", "not", "we", "you", "they", "he", "she", "when", "after", "before",
    "once", "each", "every", "if", "since", "because", "while", "so", "and",
    "but", "i", "i'm", "i've", "i'd", "it's",
}


def _normalize_reason(line: str) -> str:
    """The run's own RESULT sentence, reworded to read as the scheduler's own
    words: trailing punctuation and whitespace stripped, a leading "I could
    not"/"I can't"/etc rewritten to "it ...", and the first letter lower-cased
    unless it looks like a capitalized name. Never returns "", ".", " ." or
    anything ending in "..": a blank or punctuation-only line falls back to
    "it could not finish the scheduled task"."""
    text = (line or "").strip().rstrip(". \t")
    if not text:
        return "it could not finish the scheduled task"
    low = text.lower()
    for prefix, repl in _I_REWRITES:
        if low.startswith(prefix):
            text = repl + text[len(prefix):]
            break
    first_word = re.match(r"[A-Za-z']+", text)
    word = first_word.group(0).lower() if first_word else ""
    if text[:1].isupper() and word in _LOWER_STARTERS:
        text = text[0].lower() + text[1:]
    text = text.strip().rstrip(".")
    return text or "it could not finish the scheduled task"


def classify_failure(status: Dict[str, Any], timezone_name: str = DEFAULT_TIMEZONE,
                     now: Optional[float] = None) -> Dict[str, Any]:
    """Persist a stable failure class and a human explanation with each run."""
    from . import wording, schedprop
    if run_outcome(status) == "complete":
        return {"failure_class": "", "failure_reason": "", "reset_at": None}
    category = wording.failure_category(status)
    reset_at = None
    if category == "usage_limit":
        kind = "usage_limit"
        reset = wording.usage_limit_reset_text(status)
        if reset:
            reset_at = _parse_reset(reset, timezone_name, time.time() if now is None else now)
        reason = "it hit a usage limit" + (f" until {reset}" if reset else "")
    elif status.get("denied"):
        kind = "missing_permission"
        scopes = []
        for item in status["denied"]:
            tool = item.get("tool", "") if isinstance(item, dict) else str(item)
            scope = scope_for_tool(tool)
            label = schedprop.labels([scope]) if scope else tool.rsplit("__", 1)[-1].replace("_", " ")
            if label and label not in scopes:
                scopes.append(label)
        reason = "it needs permission to " + (" and ".join(scopes) or "use the required tools")
    elif category in ("network", "timeout", "buffer") or status.get("error") or any(
            not r.get("ok") for r in status.get("tool_results") or []):
        kind, reason = "tool_network_error", "a tool or network error interrupted the run"
    else:
        kind, reason = "could_not_finish", "it could not finish the scheduled task"
    return {"failure_class": kind, "failure_reason": reason, "reset_at": reset_at}


def _parse_reset(text: str, timezone_name: str, now: float) -> Optional[float]:
    """Provider reset dates use the schedule zone unless a zone is supplied."""
    zone = re.search(r"\(([A-Za-z_]+/[A-Za-z_/]+)\)", text)
    try:
        tz = ZoneInfo(zone.group(1) if zone else timezone_name)
    except (KeyError, ValueError):
        tz = ZoneInfo(timezone_name)
    local = datetime.fromtimestamp(now, tz)
    clean = re.sub(r"\s*\([^)]*\).*", "", text).strip().rstrip(".")
    try:
        dt = datetime.fromisoformat(clean.replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=tz)).timestamp()
    except ValueError:
        pass
    clean = re.sub(r"\s+", " ", clean.replace(",", " ")).strip()
    for fmt in ("%b %d %I%p", "%b %d %I:%M%p", "%I:%M%p", "%I%p",
                "%b %d %I %p", "%b %d %I:%M %p", "%I:%M %p", "%I %p"):
        try:
            parsed = datetime.strptime(clean, fmt)
        except ValueError:
            continue
        dated = "%b" in fmt
        try:
            dt = local.replace(month=parsed.month if dated else local.month,
                               day=parsed.day if dated else local.day, hour=parsed.hour,
                               minute=parsed.minute, second=0, microsecond=0)
        except ValueError:
            continue
        if dt.timestamp() <= now:
            dt = dt.replace(year=dt.year + 1) if dated else dt + timedelta(days=1)
        return dt.timestamp()
    return None


def failure_reason(run: Dict[str, Any]) -> str:
    classified = run if run.get("failure_class") else classify_failure(run)
    return _normalize_reason(run.get("result_line") or run.get("final_status")
                             or classified.get("failure_reason") or "it could not finish the scheduled task")


def _first_sentence(text: str) -> str:
    """The first sentence of a longer status block, whole and never cut mid-word; a
    text with no sentence punctuation at all is just its first line."""
    text = (text or "").strip()
    if not text:
        return ""
    m = re.search(r"(.+?[.!?])(\s|$)", text, re.S)
    if m:
        return m.group(1).strip()
    return text.splitlines()[0].strip()


def _title_for_sentence(title: str) -> str:
    """A schedule's title, trailing punctuation stripped, so composing "<title>
    could not finish: ..." never collides a title that already ends in a period
    or colon with the sentence's own one right after it."""
    return (title or "").rstrip(".!?:;,")


def _ot():
    """ownertext, imported late (it is a leaf module, but schedules is imported early)."""
    from . import ownertext
    return ownertext


def _sentence(text: str) -> str:
    """A clause as a sentence of its own: capital first letter, closing full stop."""
    t = " ".join(str(text or "").split())
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".!?" else t + "."


def _shown(ts: Optional[float], timezone_name: str) -> str:
    """"2:30 PM", with the day added when it is not today, in the schedule's zone."""
    if ts is None:
        return "soon"
    tz = ZoneInfo(timezone_name)
    dt = datetime.fromtimestamp(ts, tz)
    now_local = datetime.fromtimestamp(time.time(), tz)
    clock = dt.strftime("%-I:%M %p")
    if dt.date() != now_local.date():
        return f"{dt.strftime('%a %b %-d')}, {clock}"
    return clock


_REPAIR_DELAY_MINUTES_RE = re.compile(r"\bdelay_minutes\s+(\d+)\b")
_REPAIR_EVERY_UNIT_RE = re.compile(r"\bevery\s+(\d+)\s+(minute|minutes|hour|hours)\b", re.IGNORECASE)

# A tool-shaped identifier (word_word, the shape every registered tool's short
# name takes) immediately followed by a call-like "(" or "with", or preceded
# by "call" -- the two phrasings the old continue-kind instructions used
# ("call schedule_continue with delay_minutes 240", "schedule_continue(...)").
_REPAIR_TOOL_CALL_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]*)\b\s*(?:\(|with\b)")
_REPAIR_CALL_PREFIX_RE = re.compile(r"\bcall\s+([a-zA-Z][a-zA-Z0-9_]*)\b")


def _repair_known_tool_names() -> "set":
    """Every tool the registry actually grants, by its short name (after the
    last `__`) and its full name -- a prompt naming anything outside this set
    in a call-shaped phrase is naming a tool that does not exist."""
    names = set()
    for info in SCOPES.values():
        for tool in info.get("tools") or []:
            names.add(tool)
            names.add(tool.rsplit("__", 1)[-1])
    return names


# Scopes a schedule Aaron asked for himself can start with on the spot: reading,
# and messaging (a scheduled run's send to anyone but Aaron is held for his yes
# regardless, see executor.run). Anything that writes to a sheet, a site, a
# calendar or a deployment is still proposed first.
SELF_SERVE_SCOPES = frozenset({"imessage_send", "sms_send", "email_send"})


def self_serve(scopes: List[str]) -> bool:
    """Whether every capability in `scopes` is one a self-serve schedule may start
    with unasked. An empty list is not self-serve: it is nothing at all."""
    return bool(scopes) and all(s in READ_SCOPES or s in SELF_SERVE_SCOPES for s in scopes)


def starts_unasked(cadence_kind: str, scopes: List[str]) -> bool:
    """A one-time, self-serve ask ("remind me in 12 hours") starts without a proposal.
    Recurring work keeps the proposal: its timing and reach are worth one look before
    it runs unattended for weeks."""
    return cadence_kind == "once" and self_serve(scopes)


def _repair_recover_timing(prompt: str) -> Optional[Dict[str, Any]]:
    """A cadence recovered from the prompt's own old instructions: a
    `delay_minutes N` or "every N minutes/hours" phrase gives `every` N
    minutes (hours converted). None when the prompt states no timing at all."""
    m = _REPAIR_DELAY_MINUTES_RE.search(prompt or "")
    if m:
        return {"cadence_kind": "every", "cadence": {"minutes": int(m.group(1))}}
    m = _REPAIR_EVERY_UNIT_RE.search(prompt or "")
    if m:
        n = int(m.group(1))
        if m.group(2).lower().startswith("hour"):
            n *= 60
        return {"cadence_kind": "every", "cadence": {"minutes": n}}
    return None


def _repair_dead_tool_names(paragraph: str, known: "set") -> List[str]:
    found: List[str] = []
    for pattern in (_REPAIR_TOOL_CALL_RE, _REPAIR_CALL_PREFIX_RE):
        for m in pattern.finditer(paragraph):
            name = m.group(1)
            if "_" in name and name not in known and name not in found:
                found.append(name)
    return found


def _repair_prompt_paragraphs(prompt: str) -> "tuple[List[Dict[str, str]], str]":
    """Every paragraph of `prompt` that instructs calling a tool not in the
    registry, with why, and the prompt that remains once they are cut.
    Paragraphs are split on a blank line, matching how these old prompts were
    written (one instruction block per paragraph)."""
    known = _repair_known_tool_names()
    paragraphs = re.split(r"\n\s*\n", prompt or "")
    changes: List[Dict[str, str]] = []
    kept: List[str] = []
    for para in paragraphs:
        dead = _repair_dead_tool_names(para, known)
        if dead:
            changes.append({
                "remove": para.strip(),
                "why": f"it tells the run to call {dead[0]}, which no longer exists",
            })
        else:
            kept.append(para)
    return changes, "\n\n".join(kept)


def _insert_before_reply_yes(text: str, extra: str) -> str:
    """Item C10: `extra` (one sentence, no leading/trailing space) inserted right
    before the proposal's closing "Reply yes ..." sentence, or appended at the end
    when that sentence is not found (never silently dropped). A blank `extra`
    leaves `text` untouched."""
    if not extra:
        return text
    extra = extra.strip()
    marker = "Reply yes"
    idx = text.rfind(marker)
    if idx == -1:
        return f"{text} {extra}".strip()
    if text[:idx].endswith("\n\n"):
        # The house format (ownertext.py) puts the reply line in its own paragraph.
        return f"{text[:idx]}{extra}\n\n{text[idx:]}"
    return f"{text[:idx]}{extra} {text[idx:]}"


def _repair_join(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _ends_only(schedule: Schedule) -> bool:
    """A follow-up: Aaron hears from it only when it ends (see schedprop.is_follow_up)."""
    return str(getattr(schedule, "report_mode", "") or "") == "end"


class ScheduleService:
    def __init__(self, manager: Any):
        self.manager = manager
        self.store = manager.store

    # -- the seams other packages fill in (schedprop.py does not exist in this
    # tree; tests replace these by setting an attribute on the gate package so
    # `from . import schedprop` resolves to a fake) --------------------------
    def _proposal_text(self, schedule: Schedule) -> str:
        from . import schedprop
        return schedprop.format_proposal(schedule)

    def _labels(self, scopes: List[str]) -> str:
        from . import schedprop
        return schedprop.labels(scopes)

    def _missing(self, prompt: str, scopes: List[str]) -> List[str]:
        from . import schedprop
        return schedprop.missing_capabilities(prompt, scopes)

    def _default_report_every_time(self, prompt: str, cadence_kind: str) -> bool:
        from . import schedprop
        return schedprop.default_report_every_time(prompt, cadence_kind)

    def _capability_match(self, text: str) -> Optional[str]:
        from . import schedprop
        return schedprop.capability_match(text)

    async def _notice(self, schedule: Schedule, text: str, *, kind: str = "schedule") -> None:
        try:
            await self.manager.send_schedule_notice(schedule, text, kind=kind)
        except TypeError as exc:
            if "unexpected keyword argument 'kind'" not in str(exc):
                raise
            # Older manager stand-ins still have the two-argument send method.
            await self.manager.send_schedule_notice(schedule, text)

    async def _owner_notice(self, text: str) -> None:
        try:
            await self.manager.send_to_approver(text, kind="schedule")
        except TypeError as exc:
            if "unexpected keyword argument 'kind'" not in str(exc):
                raise
            await self.manager.send_to_approver(text)

    def _scopes_for_tool(self, tool: str) -> List[str]:
        best = scope_for_tool(tool)
        return [best] if best else []

    def _run_row_for_request(self, schedule_id: int, request_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if request_id is None:
            return None
        for row in self.store.schedule_runs(schedule_id):
            if row.get("request_id") == request_id:
                return row
        return None

    # -- creating and deciding -------------------------------------------
    async def propose(self, *, owner: bool, chat_id: str, task_id: int, title: str, prompt: str,
                      cadence_kind: str, cadence: Dict[str, Any], timezone: Optional[str] = None,
                      report_every_time: Optional[bool] = None, until: Optional[float] = None,
                      scopes: List[str], start_now: bool = False, origin_mode: str = "",
                      origin_chat_id: str = "", auto_start: bool = False,
                      start_at: Optional[float] = None, stop_when: Optional[str] = None) -> Schedule:
        """Create a schedule. It is proposed to Aaron for his yes, unless `auto_start`:
        then it is activated at once and he hears what was set up instead. The caller
        decides `auto_start` (see self_serve); the service never widens it.

        `start_at` is the first run he stated ("starting 9am"): an `every` cadence is
        anchored on it, any other repeating cadence fires no earlier. `stop_when` makes
        it a follow-up: the condition is stated in its prompt as each run's first
        check (the run ends it with STATUS: STOP), and he hears only when it ends."""
        if not owner:
            raise PermissionError("only Aaron can create or change schedules")
        cadence_module.validate(cadence_kind, cadence)
        timezone_name = timezone or DEFAULT_TIMEZONE
        ZoneInfo(timezone_name)
        if cadence_kind == "once" and cadence_module.next_run(cadence_kind, cadence, timezone_name,
                                                              time.time()) is None:
            raise ValueError("That time has already passed.")
        missing = self._missing(prompt, scopes)
        if missing:
            raise ValueError(f"missing capability: {self._labels(missing)}")
        report_every = (bool(report_every_time) if report_every_time is not None
                        else self._default_report_every_time(prompt, cadence_kind))
        stored_cadence = dict(cadence or {})
        stored_cadence["start_now"] = bool(start_now)
        if start_at is not None and start_at > time.time() and cadence_kind != "once":
            stored_cadence["start_now"] = False
            if cadence_kind == "every":
                stored_cadence["anchor"] = float(start_at)
            else:
                stored_cadence["not_before"] = float(start_at)
        report_mode = "changed"
        stop_when = " ".join(str(stop_when or "").split())
        if stop_when:
            from . import schedprop
            stored_cadence["stop_when"] = stop_when
            prompt = schedprop.follow_up_prompt(prompt, stop_when)
            report_mode, report_every = "end", False
        schedule = self.store.create_schedule(
            chat_id=chat_id, task_id=task_id, title=title, prompt=prompt,
            kind=cadence_kind, scopes=scopes, timezone=timezone_name, report_mode=report_mode,
            cadence_kind=cadence_kind, cadence=stored_cadence, report_every_time=report_every,
            until=until, origin_mode=origin_mode, origin_chat_id=origin_chat_id or chat_id,
        )
        if auto_start:
            # His own reminder or check, in his own words, able to do nothing beyond
            # reading and messaging him: it starts now. "Remind me in 12 hours" once
            # sat as a "reply yes" he never saw until its time had passed.
            logger.info("[schedules] %s self-serve: started without a proposal: %s", schedule.id, title)
            return await self.decide(schedule.id, True)
        logger.info("[schedules] %s proposed: %s", schedule.id, title)
        await self._notice(schedule, self._proposal_text(schedule), kind="approval")
        return schedule

    @staticmethod
    def self_serve(scopes: List[str]) -> bool:
        return self_serve(scopes)

    @staticmethod
    def starts_unasked(cadence_kind: str, scopes: List[str]) -> bool:
        return starts_unasked(cadence_kind, scopes)

    async def decide(self, schedule_id: int, approve: bool) -> Schedule:
        schedule = self.store.get_schedule(schedule_id)
        if schedule is None:
            raise ValueError(f"no such schedule: {schedule_id}")
        if schedule.state != "proposed":
            raise ValueError(f"{schedule.title} is not waiting for an answer")
        now = time.time()
        was_edit = self._edit_revert(schedule) is not None
        if not approve:
            revert = self._edit_revert(schedule)
            if revert is not None and revert.get("state") in ("active", "paused"):
                # A rejected change to an existing, already-approved schedule puts it
                # back as it was; only a brand-new proposal ends when he says no.
                state = revert.get("state")
                next_run = (cadence_module.next_run(schedule.cadence_kind, schedule.cadence,
                                                    schedule.timezone, now) if state == "active" else None)
                updated = self.store.update_schedule(
                    schedule_id, prompt=revert["prompt"], scopes=revert["scopes"], state=state,
                    next_run=next_run, notes=[], last_outcome="a change was rejected; kept as it was")
                logger.info("[schedules] %s edit rejected; reverted", schedule_id)
                await self._notice(updated, f"Left {updated.title} as it was.", kind="reply")
                return updated
            # Either a brand-new proposal, or an edit snapshot whose own pre-edit state
            # was "proposed" (he never actually said yes to it even once): either way
            # there is no approved version to revert to, so this ends rather than
            # becoming a "paused" schedule that resume could restart unapproved.
            updated = self.store.update_schedule(schedule_id, state="ended", next_run=None,
                                                 last_outcome="rejected")
            logger.info("[schedules] %s rejected", schedule_id)
            await self._notice(updated, f"Rejected {updated.title}.", kind="reply")
            return updated
        cadence = dict(schedule.cadence or {})
        start_now = bool(cadence.pop("start_now", False))
        if cadence != (schedule.cadence or {}):
            self.store.update_schedule(schedule_id, cadence=cadence)
            schedule = self.store.get_schedule(schedule_id)  # type: ignore[assignment]
        if start_now:
            next_run = now
        else:
            next_run = cadence_module.next_run(schedule.cadence_kind, schedule.cadence,
                                               schedule.timezone, now)
        if schedule.cadence_kind == "every" and "anchor" not in (schedule.cadence or {}):
            anchored = dict(schedule.cadence)
            anchored["anchor"] = next_run
            self.store.update_schedule(schedule_id, cadence=anchored)
            schedule = self.store.get_schedule(schedule_id)  # type: ignore[assignment]
        updated = self.store.update_schedule(schedule_id, state="active", next_run=next_run, notes=[])
        logger.info("[schedules] %s activated, first run %s", schedule_id, next_run)
        await self._notice(updated, f"Changed {updated.title}. Next check {_shown(next_run, updated.timezone)}."
                           if was_edit else self._started_text(updated, next_run), kind="reply")
        return updated

    def _started_text(self, schedule: Schedule, next_run: Optional[float]) -> str:
        when = _shown(next_run, schedule.timezone)
        if schedule.cadence_kind == "once":
            return f"Scheduled for {when}."
        words = cadence_module.describe(schedule.cadence_kind, schedule.cadence, schedule.timezone)
        return f"Started. First check {when}, then {words}."

    def is_proposal_request(self, req: Request) -> bool:
        """The new model never routes a proposal through the requests table (it is
        a schedule in state 'proposed' with the owner answering plain yes/no on its
        thread), so a request is never, itself, the proposal. Kept so older call
        sites that still check this before touching a request never misfire."""
        return False

    async def decide_proposal(self, req: Request, decision: str, note: str = "") -> Dict[str, Any]:
        """Back-compat entry point: the single activation path is decide()."""
        schedule_id = req.schedule_id or 0
        d = decision.strip().lower()
        if d in {"yes", "y", "ok", "approve", "run", "go"}:
            updated = await self.decide(schedule_id, True)
            return {"ok": True, "state": updated.state, "error": ""}
        if d in {"no", "n", "stop", "drop", "reject", "decline"}:
            updated = await self.decide(schedule_id, False)
            return {"ok": True, "state": updated.state, "error": ""}
        return {"ok": False, "state": req.state, "error": f"unknown decision {decision!r}"}

    def proposal_expired(self, req: Request) -> None:
        """Back-compat: an old-style proposal that still lived as a 'pending' request
        row (one with a request to expire at all). The new model proposes with no
        request row, so this is dead in practice unless such a row exists; see
        expire_stale_proposals for the real path."""
        if not req.schedule_id:
            return
        schedule = self.store.get_schedule(req.schedule_id)
        if schedule is None or schedule.state != "proposed":
            return
        updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                             last_outcome="not approved in time")
        return updated

    def _edit_revert(self, schedule: Schedule) -> Optional[Dict[str, Any]]:
        if not schedule.notes:
            return None
        try:
            data = json.loads(schedule.notes[-1])
        except (TypeError, ValueError, IndexError):
            return None
        return data.get("_edit_revert") if isinstance(data, dict) else None

    async def expire_stale_proposals(self, max_age: float, now: Optional[float] = None) -> None:
        """A schedule stuck in 'proposed' with no answer for longer than max_age (the
        24-hour rule). A fresh proposal just ends. An edit proposal (one with a
        pre-edit snapshot stashed by edit()) reverts to what it was before the edit
        when that was 'active', or is left 'paused' with a plain reason when it was
        not (nothing to safely resume)."""
        now = time.time() if now is None else now
        for schedule in self.store.schedules(states=["proposed"]):
            revert = self._edit_revert(schedule)
            if (revert is None and schedule.cadence_kind == "once"
                    and cadence_module.next_run("once", schedule.cadence, schedule.timezone, now) is None):
                # A one-time proposal whose moment has gone by unanswered is over
                # now, not 24 hours later: there is nothing left to approve.
                updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                     last_outcome="its time passed before it was approved")
                logger.info("[schedules] %s once-proposal expired: its time passed", schedule.id)
                await self._notice(updated, _ot().compose(
                    "timer", f"{_title_for_sentence(updated.title)} did not run",
                    body="It was never approved and its time has passed."))
                continue
            if now - schedule.updated_at < max_age:
                continue
            if revert is None:
                updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                     last_outcome="not approved in time")
                logger.info("[schedules] %s proposal expired unanswered", schedule.id)
                await self._notice(updated, _ot().compose(
                    "timer", f"{_title_for_sentence(updated.title)} was not set up",
                    body="It was not approved in time."))
                continue
            if revert.get("state") == "active":
                next_run = cadence_module.next_run(schedule.cadence_kind, schedule.cadence,
                                                   schedule.timezone, now)
                updated = self.store.update_schedule(
                    schedule.id, prompt=revert["prompt"], scopes=revert["scopes"], state="active",
                    next_run=next_run, notes=[], last_outcome="an edit expired unanswered; kept as it was")
                logger.info("[schedules] %s edit proposal expired; reverted to the previous version", schedule.id)
                await self._notice(updated, _ot().compose(
                    "timer", f"{_title_for_sentence(updated.title)} is unchanged",
                    body="That change expired unanswered, so I kept it as it was."))
            elif revert.get("state") == "paused":
                updated = self.store.update_schedule(
                    schedule.id, prompt=revert["prompt"], scopes=revert["scopes"], state="paused",
                    next_run=None, notes=[],
                    last_outcome="an edit expired unanswered and could not resume automatically")
                logger.info("[schedules] %s edit proposal expired; left paused", schedule.id)
                await self._notice(updated, _ot().compose(
                    "failed", f"{_title_for_sentence(updated.title)} is paused",
                    body="That change expired unanswered."))
            else:
                # The pre-edit snapshot was itself "proposed" -- he never said yes to
                # this schedule even once, so there is nothing approved to revert to
                # or pause; ending it (not pausing) keeps resume from ever restarting
                # something he never approved.
                updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                     last_outcome="never approved; expired unanswered")
                logger.info("[schedules] %s edit proposal expired; never approved, ended", schedule.id)
                await self._notice(updated, _ot().compose(
                    "timer", f"{_title_for_sentence(updated.title)} was not set up",
                    body="It was not approved in time."))

    async def held(self, req: Request, status: Dict[str, Any]) -> None:
        """A scheduled run's call is parked for the owner's yes. Record the run's
        own (still-blank) row as incomplete right away, so it does not sit blank
        while he decides -- without touching consecutive_failures or next_run,
        both already settled at fire time, and without sending a notice: the
        confirmation text ask_confirmation sends is the only message this
        moment gets."""
        if not req.schedule_id:
            return
        schedule = self.store.get_schedule(req.schedule_id)
        if schedule is None:
            return
        confirm = status.get("confirm") or {}
        tool = str(confirm.get("tool") or "")
        what = tool.split("__")[-1].replace("_", " ") or "do this"
        self._record_held_outcome(schedule.id, req.id, outcome="incomplete",
                                  result_line=f"waiting for your yes to {what}",
                                  final_status=str(status.get("final_status") or ""))

    async def held_resolved(self, req: Request, status: Dict[str, Any]) -> None:
        """The owner said yes and the held call ran. Record the run as complete
        -- the call that was waiting is the whole of what this run needed to
        do -- without routing through finished(): a held run was already
        excluded from the three-strikes count and already got its one notice
        at hold time, so completing it sends nothing more."""
        if not req.schedule_id:
            return
        schedule = self.store.get_schedule(req.schedule_id)
        if schedule is None:
            return
        result_line = str(status.get("summary") or status.get("error") or "performed.")
        self._record_held_outcome(schedule.id, req.id, outcome="complete", result_line=result_line,
                                  final_status=str(status.get("final_status") or result_line))

    async def held_ended(self, req: Request, reason: str) -> None:
        if not req.schedule_id:
            return
        schedule = self.store.get_schedule(req.schedule_id)
        if schedule is None:
            return
        if reason == "declined":
            outcome = "The held call was declined."
        elif reason == "expired":
            outcome = "The held call expired unanswered."
        else:
            raise ValueError(f"unknown held call outcome: {reason}")
        self._record_held_outcome(schedule.id, req.id, outcome="incomplete", result_line=outcome,
                                  final_status=outcome)
        changes: Dict[str, Any] = {"last_outcome": outcome}
        if schedule.cadence_kind == "once":
            changes.update(state="ended", next_run=None)
        updated = self.store.update_schedule(schedule.id, **changes)
        await self._notice(updated, _ot().compose("timer", _title_for_sentence(updated.title),
                                                  body=_sentence(outcome)))

    def _record_held_outcome(self, schedule_id: int, request_id: Optional[int], *, outcome: str,
                             result_line: str, final_status: str) -> None:
        """Write or update a held call's schedule_runs row. Used both the moment a
        call is parked (outcome="incomplete") and the moment it is resolved
        (complete, or still incomplete after a decline/expiry) -- never finished(),
        so a held run never counts toward the three-strikes pause and its
        resolution never fires a second notice."""
        now = time.time()
        run_row = self._run_row_for_request(schedule_id, request_id)
        run_id = run_row["id"] if run_row else self.store.create_schedule_run(schedule_id, request_id, now)
        self.store.finish_schedule_run(run_id, finished_at=now, complete=(outcome == "complete"),
                                       outcome=outcome, result_line=result_line, final_status=final_status,
                                       denied=[], note="", changed=False, cost_usd=None)

    # -- editing and acting ------------------------------------------------
    async def edit(self, schedule_id: int, *, owner: bool, changes: Dict[str, Any],
                   extra_proposal_note: str = "") -> Dict[str, Any]:
        if not owner:
            raise PermissionError("only Aaron can create or change schedules")
        old = self.store.get_schedule(schedule_id)
        if old is None:
            raise ValueError(f"no such schedule: {schedule_id}")
        immediate: Dict[str, Any] = {}
        if "title" in changes:
            immediate["title"] = changes["title"]
        if "cadence_kind" in changes or "cadence" in changes:
            kind = changes.get("cadence_kind", old.cadence_kind)
            cadence = dict(changes.get("cadence", old.cadence) or {})
            # Saving a title/report change alongside the SAME every-N-minutes interval
            # should not restart the firing window: keep the old anchor rather than
            # letting the next_run recompute below treat this as a brand new interval.
            old_cadence = old.cadence or {}
            if (kind == "every" and old.cadence_kind == "every" and "anchor" in old_cadence
                    and cadence.get("minutes") == old_cadence.get("minutes")):
                cadence["anchor"] = old_cadence["anchor"]
            cadence_module.validate(kind, cadence)
            if kind == "once" and cadence_module.next_run(kind, cadence, old.timezone, time.time()) is None:
                raise ValueError("That time has already passed.")
            immediate["cadence_kind"] = kind
            immediate["cadence"] = cadence
        if "report_every_time" in changes:
            immediate["report_every_time"] = bool(changes["report_every_time"])
        if "until" in changes:
            immediate["until"] = changes["until"]
        if immediate and old.state == "active":
            immediate["next_run"] = cadence_module.next_run(
                immediate.get("cadence_kind", old.cadence_kind),
                immediate.get("cadence", old.cadence), old.timezone, time.time())
        updated = old
        message = ""
        if immediate:
            updated = self.store.update_schedule(schedule_id, **immediate)
            message = f"{updated.title} updated."
        needs_approval = "prompt" in changes or "scopes" in changes
        if needs_approval:
            prompt = changes.get("prompt", old.prompt)
            scopes = changes.get("scopes", old.scopes)
            missing = self._missing(prompt, scopes)
            if missing:
                raise ValueError(f"missing capability: {self._labels(missing)}")
            # Stashed so an unanswered edit can be reverted rather than left half-applied
            # if it expires (expire_stale_proposals below). `notes` is otherwise unread
            # after the cadence-model migration, so it is free for this bookkeeping.
            revert = {"prompt": old.prompt, "scopes": old.scopes, "state": old.state}
            updated = self.store.update_schedule(
                schedule_id, prompt=prompt, scopes=scopes, state="proposed",
                revision=old.revision + 1, notes=[json.dumps({"_edit_revert": revert})])
            await self._notice(updated, _insert_before_reply_yes(
                self._proposal_text(updated), extra_proposal_note), kind="approval")
            message = f"Updated proposal sent for {updated.title}."
        if immediate or needs_approval:
            updated = self.store.update_schedule(schedule_id, failure_reset_at=time.time(),
                                                 retry_at=None, consecutive_failures=0, last_outcome="",
                                                 revision=old.revision + 1)
        logger.info("[schedules] %s edited (needs_approval=%s)", schedule_id, needs_approval)
        return {"schedule": updated, "applied": immediate, "needs_approval": needs_approval,
                "message": message}

    def valid_actions(self, schedule: Schedule) -> List[str]:
        actions = set(_ACTIONS_BY_STATE.get(schedule.state, set()))
        actions.add("edit")
        return sorted(actions)

    async def action(self, schedule_id: int, action: str, *, owner: bool) -> Schedule:
        if not owner:
            raise PermissionError("only Aaron can create or change schedules")
        schedule = self.store.get_schedule(schedule_id)
        if schedule is None:
            raise ValueError(f"no such schedule: {schedule_id}")
        action = action.strip().lower().replace(" ", "_")
        valid = self.valid_actions(schedule)
        if action not in _ACTION_NAMES or action not in valid:
            raise ValueError(f"{action} is not valid for a schedule that is {schedule.state} "
                             f"(valid actions: {', '.join(valid)})")
        now = time.time()
        if action == "pause":
            # Owner-initiated, three-strikes, or migration: any route to 'paused' clears
            # next_run so a paused schedule never shows a time it will not actually fire.
            return self.store.update_schedule(schedule_id, state="paused", next_run=None)
        if action == "run_now":
            if self.store.request_running_for_schedule(schedule_id) is not None:
                raise ValueError(f"{schedule.title} already has a run in flight; wait for it to finish")
            if schedule.state == "paused":
                # Same resume-then-run behavior the chat path uses: a paused
                # schedule has to be running again before "run it now" means
                # anything, or the fired run would never actually happen.
                if not schedule.cadence_kind:
                    raise ValueError(f"{schedule.title} needs to be told how often to run first")
                missing = self._missing(schedule.prompt, schedule.scopes)
                if missing:
                    raise ValueError(f"missing capability: {self._labels(missing)}")
                self.store.update_schedule(schedule_id, state="active", consecutive_failures=0,
                                           failure_reset_at=now, retry_at=None, last_outcome="", revision=schedule.revision + 1)
            updated = self.store.update_schedule(schedule_id, next_run=now, retry_at=None)
            gateway = getattr(self.manager, "gateway", None)
            try:
                if gateway is not None and settings.get("GATE_INBOX"):
                    # A manual run waits behind the owner's current inbox turn.
                    await self.tick(now=now)
            except Exception as exc:
                logger.warning("[schedules] run now will use the next scheduler pass: %s", exc)
            return self.store.get_schedule(schedule_id) or updated
        if action == "stop":
            return self.store.update_schedule(schedule_id, state="ended", next_run=None,
                                              last_outcome="stopped")
        if action in ("resume", "start_again"):
            if not schedule.cadence_kind:
                raise ValueError(f"{schedule.title} needs to be told how often to run first")
            missing = self._missing(schedule.prompt, schedule.scopes)
            if missing:
                raise ValueError(f"missing capability: {self._labels(missing)}")
            next_run = cadence_module.next_run(schedule.cadence_kind, schedule.cadence,
                                               schedule.timezone, now)
            return self.store.update_schedule(schedule_id, state="active", next_run=next_run,
                                              consecutive_failures=0, failure_reset_at=now,
                                              retry_at=None, last_outcome="", revision=schedule.revision + 1)
        raise ValueError(f"unknown action: {action}")

    # -- one-step repair --------------------------------------------------
    def repair_plan(self, schedule_id: int) -> Dict[str, Any]:
        """What would make a stuck schedule runnable again, worked out by code:
        a timing when none is set (recovered from its old instructions when
        they state one), the capabilities its prompt needs but does not have,
        and any paragraph instructing a tool that no longer exists. Returns
        `needed=False` when nothing here applies -- a healthy schedule has
        nothing to repair."""
        from . import schedprop
        schedule = self.store.get_schedule(schedule_id)
        if schedule is None:
            raise ValueError(f"no such schedule: {schedule_id}")
        sentence_parts: List[str] = []
        timing: Optional[Dict[str, Any]] = None
        if not schedule.cadence_kind:
            recovered = _repair_recover_timing(schedule.prompt)
            if recovered is not None:
                text = cadence_module.describe(recovered["cadence_kind"], recovered["cadence"], schedule.timezone)
                timing = {"cadence_kind": recovered["cadence_kind"], "cadence": recovered["cadence"], "text": text}
                sentence_parts.append(f"a timing ({text}, from its old instructions)")
            else:
                sentence_parts.append("a timing you need to choose")
        missing = schedprop.missing_capabilities(schedule.prompt, schedule.scopes)
        add_capabilities = [{"scope": name, "label": schedprop.label(name)} for name in missing]
        if add_capabilities:
            sentence_parts.append(f"permission to {schedprop.labels(missing)}")
        prompt_changes, new_prompt = _repair_prompt_paragraphs(schedule.prompt)
        if prompt_changes:
            sentence_parts.append("its instructions mention a tool that no longer exists")
        needed = bool(timing or (not schedule.cadence_kind) or add_capabilities or prompt_changes)
        # cadence_kind missing with no recovered timing still counts as needed
        # (the sentence already says so above); only the "nothing at all to
        # fix" case returns needed=False.
        if not sentence_parts:
            return {"needed": False, "sentence": "", "timing": None, "add_capabilities": [],
                    "prompt_changes": [], "new_prompt": schedule.prompt}
        sentence = f"It needs {_repair_join(sentence_parts)}."
        return {"needed": True, "sentence": sentence, "timing": timing,
                "add_capabilities": add_capabilities, "prompt_changes": prompt_changes,
                "new_prompt": new_prompt}

    async def repair_apply(self, schedule_id: int, *, owner: bool,
                           timing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a repair_plan() in one step: sets the timing (the plan's own
        recovered one, or the one passed in when the plan had none), adds the
        missing capabilities, and replaces the prompt. Only the timing and
        capabilities/prompt change -- nothing else about the schedule is
        touched. If only the timing was missing it applies at once and
        resumes; a prompt or capability change needs his yes, same as edit()."""
        if not owner:
            raise PermissionError("only Aaron can create or change schedules")
        plan = self.repair_plan(schedule_id)
        if not plan["needed"]:
            schedule = self.store.get_schedule(schedule_id)
            return {"schedule": schedule, "applied": {}, "needs_approval": False,
                    "message": f"{schedule.title} does not need repair."}
        chosen_timing = plan["timing"] or timing
        if chosen_timing is None and not self.store.get_schedule(schedule_id).cadence_kind:
            raise ValueError("It needs a timing and none was given.")
        changes: Dict[str, Any] = {}
        if chosen_timing is not None:
            changes["cadence_kind"] = chosen_timing["cadence_kind"]
            changes["cadence"] = chosen_timing["cadence"]
        add_scopes = [c["scope"] for c in plan["add_capabilities"]]
        if add_scopes or plan["prompt_changes"]:
            schedule = self.store.get_schedule(schedule_id)
            changes["scopes"] = sorted(set(schedule.scopes) | set(add_scopes))
            changes["prompt"] = plan["new_prompt"]
        extra_note = ("I will also drop the part of its instructions that mentions a "
                     "tool that no longer exists." if plan["prompt_changes"] else "")
        result = await self.edit(schedule_id, owner=owner, changes=changes, extra_proposal_note=extra_note)
        if not result["needs_approval"] and result["schedule"].state == "paused":
            # Only the timing changed: no prompt/scope change means edit() never
            # moves the schedule out of 'paused' on its own (edit() only flips
            # next_run for an already-active schedule) -- resume it here so a
            # timing-only repair really does apply and resume in one step.
            resumed = await self.action(schedule_id, "resume", owner=owner)
            result = {"schedule": resumed, "applied": result["applied"], "needs_approval": False,
                      "message": f"{resumed.title} repaired and resumed."}
        return result

    # -- firing --------------------------------------------------------------
    async def tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        if str(self.store.settings().get("GATE_SCHEDULES_PAUSED", "0")).lower() in {"1", "true", "yes", "on"}:
            return
        for schedule in self.store.due_schedules(now):
            try:
                await self._fire(schedule, now)
            except Exception:
                logger.exception("schedule %s failed during its tick", schedule.id)

    async def _fire(self, schedule: Schedule, now: float) -> None:
        if schedule.until is not None and now >= schedule.until:
            updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                 last_outcome="reached its end date")
            logger.info("[schedules] %s ended: reached its end date", schedule.id)
            await self._notice(updated, _ot().compose(
                "timer", f"{_title_for_sentence(updated.title)} has ended",
                body="It reached its end date and will not run again."))
            return
        running = self.store.request_running_for_schedule(schedule.id)
        if running is not None:
            run_id = self.store.create_schedule_run(schedule.id, None, now)
            self.store.finish_schedule_run(run_id, finished_at=now, complete=False, outcome="skipped",
                                           result_line="", final_status="", denied=[], note="",
                                           changed=False, cost_usd=None)
            next_run = cadence_module.next_run(schedule.cadence_kind, schedule.cadence,
                                               schedule.timezone, now)
            self.store.update_schedule(schedule.id, next_run=next_run)
            logger.info("[schedules] %s skipped a firing: previous run still in flight", schedule.id)
            await self._notify_held_skip_once(schedule, running, now)
            return
        owner_scopes = [scope for scope in SCOPES if not scope.startswith("vault:")]
        full_scopes = list(dict.fromkeys(list(schedule.scopes) + owner_scopes))
        req = self.store.create_request(
            chat_id=schedule.chat_id, sender="Aaron", sender_name="Aaron", mode="schedule", subject="",
            original_message=schedule.prompt, summary=schedule.title, scopes=full_scopes,
            prompt=schedule.prompt, state="approved", task_id=schedule.task_id, schedule_id=schedule.id,
            schedule_kind=schedule.cadence_kind,
        )
        # The revision this run started from: finished() refuses to act on a signal
        # from an older revision if the owner edited the schedule while it ran.
        self.store.set_state(req.id, "approved", status={"schedule_revision": schedule.revision,
                                                          "usage_retry": schedule.retry_at is not None})
        next_run = cadence_module.next_run(schedule.cadence_kind, schedule.cadence, schedule.timezone, now)
        req = self.store.get_request(req.id)
        run_id = self.store.create_schedule_run(schedule.id, req.id, now)
        self.store.update_schedule(schedule.id, next_run=next_run, last_run=now, last_request_id=req.id,
                                   run_count=schedule.run_count + 1, retry_at=None)
        self.store.task_event(schedule.task_id, "request", f"Scheduled run: {schedule.title}",
                              chat_id=schedule.chat_id, request_id=req.id, state="running")
        logger.info("[schedules] %s fired: request #%s", schedule.id, req.id)
        # The occurrence is recorded before its event, so a later tick cannot
        # reserve the same run again while the worker waits for this chat.
        queue_timer = getattr(self.manager, "queue_timer", None)
        if not (queue_timer and queue_timer(
                schedule.chat_id, f"schedule:{schedule.id}:{schedule.next_run or now:.6f}",
                {"schedule_id": schedule.id, "request_id": req.id, "run_id": run_id})):
            asyncio.create_task(self._run_fired(req))

    async def _notify_held_skip_once(self, schedule: Schedule, running: Request, now: float) -> None:
        """A run parked waiting for the owner's yes (a held send) blocks every
        later firing silently otherwise -- tell him, but only the first time a
        firing is skipped for this held call (item 6), not on every one of
        what could be many ticks over the up-to-24-hours it may sit there."""
        if running.state != "pending":
            return
        status = running.status or {}
        if status.get("held_skip_notified"):
            return
        confirm = status.get("confirm") or {}
        about = confirm.get("about") or []
        what = ", ".join(str(a) for a in about) if about else \
            str(confirm.get("tool", "")).split("__")[-1].replace("_", " ") or "do that"
        when = _shown(running.updated_at, schedule.timezone)
        from . import ownertext
        await self._notice(schedule, ownertext.compose(
            "held", f"{_title_for_sentence(schedule.title)} is waiting for your yes",
            facts=[("To", what), ("Since", when)],
            body="It skipped a run while it waits. Say yes or no to that first.",
            handle=ownertext.reply_handle(running.id)), kind="approval")
        self.store.set_state(running.id, running.state, status={**status, "held_skip_notified": True})

    async def _run_fired(self, req: Request, *, raise_errors: bool = False) -> None:
        """Execute the reserved request and record failures before the inbox finishes.

        A raised error must still fill the run row, update the failure count, and
        send the usual notice. The inbox also needs the error to mark its event failed.
        """
        try:
            await self.manager.execute(req)
        except Exception as exc:
            logger.exception("[schedules] run #%s for schedule %s raised", req.id, req.schedule_id)
            run_row = self._run_row_for_request(req.schedule_id, req.id)
            if not run_row or run_row.get("finished_at") is None:
                self.store.set_state(req.id, "failed", status={"error": str(exc)})
                await self.finished(req, {"ok": False, "error": str(exc), "result_line": "",
                                          "final_status": f"the run raised an exception: {exc}",
                                          "denied": [], "note": "", "tool_calls": [],
                                          "finished_at": time.time()})
            if raise_errors:
                raise
        else:
            if raise_errors:
                finished = self.store.get_request(req.id)
                status = (finished.status or {}) if finished else {}
                if status.get("ok") is False:
                    raise RuntimeError(str(status.get("error") or "scheduled run failed"))

    # -- restart recovery --------------------------------------------------
    async def interrupted(self, req: Request, *, already_replied: bool = False) -> None:
        """A scheduled request stranded mid-run by a process restart. Recorded as its
        own outcome ('interrupted'), never routed through finished()/run_outcome: an
        interruption is not a failure of the model's work, so it must not count toward
        the three-strikes pause, and the schedule stays active with its next_run
        intact. Re-fired once immediately only when that next run is not already
        fairly close, or when there is no regular next run to wait for (a one-time
        schedule, or a repeating one whose next_run was already consumed). A run
        that already replied is closed silently and never re-fired."""
        if not req.schedule_id:
            return
        schedule = self.store.get_schedule(req.schedule_id)
        if schedule is None:
            return
        now = time.time()
        run_row = self._run_row_for_request(schedule.id, req.id)
        run_id = run_row["id"] if run_row else self.store.create_schedule_run(schedule.id, req.id, now)
        self.store.finish_schedule_run(run_id, finished_at=now, complete=False, outcome="interrupted",
                                       result_line="", final_status="interrupted by a restart", denied=[],
                                       note="", changed=False, cost_usd=None)
        logger.info("[schedules] %s run #%s recorded as interrupted by a restart", schedule.id, run_id)
        if already_replied:
            return
        await self._notice(schedule, _ot().compose(
            "failed", f"{_title_for_sentence(schedule.title)} was interrupted by a restart",
            tail=f"Next check: {_shown(schedule.next_run, schedule.timezone)}"))
        if schedule.state != "active":
            return
        refire = schedule.cadence_kind == "once" or schedule.next_run is None
        if not refire:
            interval = None
            try:
                probe = cadence_module.next_run(schedule.cadence_kind, schedule.cadence,
                                                schedule.timezone, now)
                probe2 = (cadence_module.next_run(schedule.cadence_kind, schedule.cadence,
                                                  schedule.timezone, probe)
                          if probe is not None else None)
                if probe is not None and probe2 is not None:
                    interval = probe2 - probe
            except cadence_module.CadenceError:
                interval = None
            if interval and (schedule.next_run - now) > interval / 2:
                refire = True
        if refire:
            self.store.update_schedule(schedule.id, next_run=now)
            logger.info("[schedules] %s re-fired immediately after the restart", schedule.id)

    # -- finishing a run -------------------------------------------------
    async def finished(self, req: Request, status: Dict[str, Any]) -> bool:
        if not req.schedule_id:
            return False
        schedule = self.store.get_schedule(req.schedule_id)
        if schedule is None:
            return True
        classification = classify_failure(status, schedule.timezone, status.get("finished_at"))
        outcome = run_outcome(status)
        if classification["failure_class"] == "usage_limit":
            outcome = "failed"
        complete = outcome == "complete"
        finished_at = status.get("finished_at") or time.time()
        result_line = str(status.get("result_line") or "")
        final_status = str(status.get("final_status") or status.get("summary") or status.get("error") or "")
        denied = list(status.get("denied") or [])
        note = str(status.get("note") or "")
        # The newest schedule_runs row by id is THIS run's own (still-blank) row,
        # created by _fire before the executor was awaited -- the previous run to
        # compare against for "changed" is the next one back, not that row.
        run_row = self._run_row_for_request(schedule.id, req.id)
        if run_row and run_row.get("finished_at") is not None:
            return True  # Completion callbacks may be delivered more than once.
        recent = self.store.schedule_runs(schedule.id, limit=2)
        prev = next((r for r in recent if r.get("id") != (run_row or {}).get("id")), None)
        # The model's own CHANGED: yes/no (schedrun.parse_closing) is the source of truth
        # when it said one; the old text-comparison heuristic is only a fallback for a run
        # whose closing text never matched the contract (no CHANGED line at all). A first
        # run always notifies, regardless of what either says.
        reported_changed = status.get("changed")
        if prev is None:
            changed = True
        elif reported_changed is not None:
            changed = bool(reported_changed)
        else:
            changed = (prev.get("result_line"), prev.get("note")) != (result_line, note)
        # The run already posted its own message where this schedule reports, or chose to
        # send nothing (gate/outcomes.py): a successful run then gets no second notice.
        told_itself = complete and (bool(status.get("answered_requester")) or status.get("outcome") == "quiet")
        run_id = run_row["id"] if run_row else self.store.create_schedule_run(schedule.id, req.id, finished_at)
        self.store.finish_schedule_run(run_id, finished_at=finished_at, complete=complete, outcome=outcome,
                                       result_line=result_line, final_status=final_status, denied=denied,
                                       note=note, changed=changed, cost_usd=status.get("cost_usd"), **classification)
        logger.info("[schedules] %s run #%s finished: outcome=%s", schedule.id, run_id, outcome)
        stored_status = (self.store.get_request(req.id).status or {})
        started_revision = status.get("schedule_revision")
        if started_revision is None:
            started_revision = (req.status or {}).get("schedule_revision", stored_status.get("schedule_revision"))
        stale = schedule.state != "active" or (
            started_revision is not None and started_revision != schedule.revision)
        if stale:
            self.store.task_event(schedule.task_id, "note",
                                  f"Run finished but {schedule.title} had moved on ({schedule.state}); "
                                  "not reactivated.", chat_id=schedule.chat_id)
            # The owner paused or edited this schedule while the run was in flight. Its
            # result still happened and is still worth telling him -- just with no "next
            # run" line (there may not be one any more) and no change to what he set.
            run_dict = {"outcome": outcome, "result_line": result_line, "final_status": final_status,
                       "denied": denied, "note": note, "changed": changed, "cost_usd": status.get("cost_usd"),
                       "needs": str(status.get("needs") or ""), "error": status.get("error")}
            send = ((not complete) or schedule.report_every_time or changed) and not told_itself
            if send and not _ends_only(schedule):
                await self._notice(schedule, self.notice_text(schedule, run_dict, None))
            return True
        if classification["failure_class"] == "usage_limit":
            retry = None
            if not status.get("usage_retry", (req.status or {}).get("usage_retry", stored_status.get("usage_retry"))):
                reset = classification["reset_at"]
                # A missing reset uses the next cadence; once-only jobs get one delayed retry.
                retry = min(t for t in (reset, schedule.next_run) if t is not None) if (
                    reset is not None or schedule.next_run is not None) else finished_at + 3600
            updated = self.store.update_schedule(schedule.id, retry_at=retry,
                                                 next_run=retry if retry is not None else schedule.next_run,
                                                 last_outcome=classification["failure_reason"])
            await self._notice(updated, _ot().compose(
                "failed", f"{_title_for_sentence(schedule.title)} could not finish",
                body=_sentence(classification["failure_reason"]),
                tail=(f"Next try: {_shown(retry, schedule.timezone)}, automatically"
                      if retry is not None else "Waiting for the next scheduled run.")))
            return True
        if note:
            self.store.update_schedule(schedule.id, last_note=note)
        run_dict = {"outcome": outcome, "result_line": result_line, "final_status": final_status,
                   "denied": denied, "note": note, "changed": changed, "cost_usd": status.get("cost_usd"),
                       "needs": str(status.get("needs") or ""), "error": status.get("error")}
        if outcome == "blocked":
            # Blocked never counts toward the three-strikes pause and is never
            # retried on its own -- retrying with the same tools could not
            # succeed, so there is nothing a later tick could do that this run
            # did not already rule out. It pauses immediately, with next_run
            # cleared, and stays paused until the owner grants a capability
            # ("let it <label>") or tells it where to find the thing.
            updated = self.store.update_schedule(schedule.id, state="paused", next_run=None,
                                                 last_outcome=failure_reason({**status, **classification}))
            logger.info("[schedules] %s paused: blocked (%s)", schedule.id, result_line or "no reason given")
            await self._owner_notice(self.notice_text(updated, run_dict, None))
            return True
        stop = bool(status.get("stop"))
        failures = 0 if complete else schedule.consecutive_failures + 1
        if stop:
            updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                 last_outcome=result_line or "stopped itself",
                                                 consecutive_failures=0)
            logger.info("[schedules] %s ended: STATUS: STOP", schedule.id)
            text = _ot().compose(
                "timer", f"{_title_for_sentence(updated.title)} has ended",
                body=_sentence(result_line or "it stopped itself"), tail="It will not run again.")
            if _ends_only(updated):
                # A follow-up's end is its report, and it is for Aaron, not the room.
                await self._owner_notice(text)
            else:
                await self._notice(updated, text)
            return True
        if _ends_only(schedule) and schedule.cadence_kind != "once" and schedule.run_count >= schedule.max_runs:
            # A follow-up whose condition never came true gives up after max_runs tries.
            updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                 consecutive_failures=failures,
                                                 last_outcome=f"gave up after {schedule.run_count} tries")
            logger.info("[schedules] %s ended: gave up after %s tries", schedule.id, schedule.run_count)
            stop_text = (getattr(schedule, "cadence", None) or {}).get("stop_when") or "what it was waiting for"
            await self._owner_notice(_ot().compose(
                "timer", f"{_title_for_sentence(updated.title)} gave up after {schedule.run_count} tries",
                body=_sentence(f"{_normalize_reason(str(stop_text))} has not happened"),
                tail=_sentence(f"Last run: {result_line or self._shown_text(run_dict)}")))
            return True
        if failures >= 3:
            updated = self.store.update_schedule(schedule.id, state="paused", next_run=None,
                                                 consecutive_failures=failures, last_outcome=failure_reason({**status, **classification}))
            logger.info("[schedules] %s paused: three consecutive runs that could not finish", schedule.id)
            paused_run = dict(run_dict)
            paused_run["paused"] = True
            await self._owner_notice(self.notice_text(updated, paused_run, None))
            return True
        extra: Dict[str, Any] = {}
        if schedule.cadence_kind == "once":
            extra.update(state="ended", next_run=None)
        updated = self.store.update_schedule(schedule.id, consecutive_failures=failures,
                                             last_outcome=result_line, **extra)
        send = ((not complete) or schedule.report_every_time or changed) and not told_itself
        if _ends_only(updated):
            # A follow-up reports when it ends (above), not on each run.
            send = False
        if told_itself:
            logger.info("[schedules] %s run #%s %s; no notice added", schedule.id, run_id,
                        "posted its own message" if status.get("answered_requester") else "chose to send nothing")
        if send:
            await self._notice(updated, self.notice_text(updated, run_dict, updated.next_run))
        return True

    # -- notice text -------------------------------------------------------
    @staticmethod
    def _shown_text(run: Dict[str, Any]) -> str:
        """The one sentence a complete-but-RESULT-less run shows: the first sentence of
        final_status (never the whole narration, never blank)."""
        return _first_sentence(str(run.get("final_status") or "")) or "it finished with nothing to report"

    @staticmethod
    def _failure_reason(run: Dict[str, Any]) -> str:
        """result_line (normalized, see _normalize_reason) -> wording.failure_in_words
        for an executor/provider error (never the raw error string) -> the first
        sentence of final_status. Never degenerates to the bare word "failed": one
        of these three always has something to say, even if it is only "it could
        not finish". Only the result_line branch is normalized here: failure_in_words
        already returns plain, owner-facing clauses on its own, and the final_status
        fallback is kept verbatim (including capitalization) as today."""
        result_line = str(run.get("result_line") or "").strip()
        if result_line:
            return _normalize_reason(result_line)
        if str(run.get("error") or "").strip():
            from . import wording
            return wording.failure_in_words(run)
        final_status = _first_sentence(str(run.get("final_status") or ""))
        if final_status:
            return final_status
        return "it could not finish"

    @staticmethod
    def _blocked_reason(run: Dict[str, Any]) -> str:
        """What a blocked run could not do, in the scheduler's own words: its
        RESULT sentence normalized, or its NEEDS text, or the plain fallback."""
        result_line = str(run.get("result_line") or "").strip()
        if result_line:
            return _normalize_reason(result_line)
        needs = str(run.get("needs") or "").strip()
        if needs:
            return _normalize_reason(needs)
        return "it did not give a reason"

    def notice_text(self, schedule: Schedule, run: Dict[str, Any], next_run: Optional[float]) -> str:
        """A schedule run's notice in the house format (ownertext.py): what happened to
        which schedule on the first line, then why, then what he can say."""
        from . import ownertext
        title = _title_for_sentence(schedule.title)
        if run.get("paused"):
            return ownertext.compose(
                "failed", f"{title} is paused",
                body=f"Three runs in a row could not finish. {_sentence(failure_reason(run))}",
                tail='Say "run now" when it is fixed.')
        outcome = run.get("outcome")
        result_line = run.get("result_line") or ""
        if outcome == "complete":
            facts = []
            if schedule.cadence_kind != "once" and next_run is not None:
                facts.append(("Next check", _shown(next_run, schedule.timezone)))
            return ownertext.compose("timer", title, body=_sentence(result_line or self._shown_text(run)),
                                     tail="\n".join(f"{k}: {v}" for k, v in facts))
        if outcome == "incomplete" and not run.get("denied"):
            return ownertext.compose("failed", f"{title} could not finish",
                                     body=_sentence(failure_reason(run)), tail='Say "run now" to try again.')
        if outcome == "incomplete":
            scope_names: List[str] = []
            for item in run.get("denied") or []:
                tool = item.get("tool") if isinstance(item, dict) else item
                for name in self._scopes_for_tool(str(tool)):
                    if name not in scope_names:
                        scope_names.append(name)
            if not scope_names:
                # A denied tool that belongs to no known scope has no label to
                # offer back ("let it do part of this" names nothing real);
                # the vague-approval phrase add_capability already understands
                # is the only sound recovery to suggest.
                return ownertext.compose(
                    "failed", f"{title} could not finish",
                    body=f"It is not allowed to do part of this. {_sentence(result_line)}".strip(),
                    tail='Say "let it do that" and I will run it again, or "stop it".')
            labels = self._labels(sorted(scope_names))
            return ownertext.compose(
                "failed", f"{title} could not finish",
                body=(_sentence(result_line) if "not allowed" in result_line.lower()
                      else f"It is not allowed to {labels}. {_sentence(result_line)}".strip()),
                tail=f'Say "let it {labels}" and I will run it again, or "stop it".')
        if outcome == "blocked":
            reason = self._blocked_reason(run)
            match = self._capability_match(str(run.get("needs") or "")) or self._capability_match(result_line)
            # Offering what it already has would send him round in a circle.
            if match and match not in (schedule.scopes or []):
                fix = f'Say "let it {self._labels([match])}" and I will run it now, or "stop it".'
            else:
                fix = 'Tell me where to find it or what to change, or say "stop it".'
            return ownertext.compose("failed", f"{title} could not run",
                                     body=f"{_sentence(reason)} I won't retry on my own.", tail=fix)
        if outcome == "failed":
            when = _shown(next_run, schedule.timezone) if next_run is not None else "soon"
            reason = self._failure_reason(run)
            tail = f"Next try: {when}"
            if schedule.consecutive_failures == 2:
                tail += "\nOne more miss and I will pause it."
            return ownertext.compose("failed", f"{title} could not finish", body=_sentence(reason), tail=tail)
        return ownertext.compose("timer", title, body=_sentence(result_line))
