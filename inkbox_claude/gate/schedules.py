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
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from . import cadence as cadence_module
from .scopes import SCOPES, tools_for
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
    "paused": {"resume", "stop"},
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
    """complete / incomplete / failed, from the executor's status contract. A run
    that was denied a tool or had a call held is incomplete even when the model's
    own STATUS line said OK -- the model does not get the last word on whether its
    own run was complete."""
    if status.get("error"):
        return "failed"
    if status.get("denied") or status.get("held"):
        return "incomplete"
    if status.get("ok"):
        return "complete"
    return "failed"


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

    async def _notice(self, schedule: Schedule, text: str) -> None:
        await self.manager.send_schedule_notice(schedule, text)

    def _scopes_for_tool(self, tool: str) -> List[str]:
        return [name for name, info in SCOPES.items() if tool in (info.get("tools") or [])]

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
                      origin_chat_id: str = "") -> Schedule:
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
                        else cadence_kind == "once")
        stored_cadence = dict(cadence or {})
        stored_cadence["start_now"] = bool(start_now)
        schedule = self.store.create_schedule(
            chat_id=chat_id, task_id=task_id, title=title, prompt=prompt,
            kind=cadence_kind, scopes=scopes, timezone=timezone_name, report_mode="changed",
            cadence_kind=cadence_kind, cadence=stored_cadence, report_every_time=report_every,
            until=until, origin_mode=origin_mode, origin_chat_id=origin_chat_id or chat_id,
        )
        logger.info("[schedules] %s proposed: %s", schedule.id, title)
        await self._notice(schedule, self._proposal_text(schedule))
        return schedule

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
                await self._notice(updated, f"Left {updated.title} as it was.")
                return updated
            # Either a brand-new proposal, or an edit snapshot whose own pre-edit state
            # was "proposed" (he never actually said yes to it even once): either way
            # there is no approved version to revert to, so this ends rather than
            # becoming a "paused" schedule that resume could restart unapproved.
            updated = self.store.update_schedule(schedule_id, state="ended", next_run=None,
                                                 last_outcome="rejected")
            logger.info("[schedules] %s rejected", schedule_id)
            await self._notice(updated, f"Rejected {updated.title}.")
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
                           if was_edit else self._started_text(updated, next_run))
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
            if now - schedule.updated_at < max_age:
                continue
            revert = self._edit_revert(schedule)
            if revert is None:
                updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                     last_outcome="not approved in time")
                logger.info("[schedules] %s proposal expired unanswered", schedule.id)
                await self._notice(updated, f"{updated.title} was not approved in time, so it was not set up.")
                continue
            if revert.get("state") == "active":
                next_run = cadence_module.next_run(schedule.cadence_kind, schedule.cadence,
                                                   schedule.timezone, now)
                updated = self.store.update_schedule(
                    schedule.id, prompt=revert["prompt"], scopes=revert["scopes"], state="active",
                    next_run=next_run, notes=[], last_outcome="an edit expired unanswered; kept as it was")
                logger.info("[schedules] %s edit proposal expired; reverted to the previous version", schedule.id)
                await self._notice(updated, f"{updated.title}: that change expired unanswered, so I kept it as it was.")
            elif revert.get("state") == "paused":
                updated = self.store.update_schedule(
                    schedule.id, prompt=revert["prompt"], scopes=revert["scopes"], state="paused",
                    next_run=None, notes=[],
                    last_outcome="an edit expired unanswered and could not resume automatically")
                logger.info("[schedules] %s edit proposal expired; left paused", schedule.id)
                await self._notice(updated, f"{updated.title} is paused: that change expired unanswered.")
            else:
                # The pre-edit snapshot was itself "proposed" -- he never said yes to
                # this schedule even once, so there is nothing approved to revert to
                # or pause; ending it (not pausing) keeps resume from ever restarting
                # something he never approved.
                updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                     last_outcome="never approved; expired unanswered")
                logger.info("[schedules] %s edit proposal expired; never approved, ended", schedule.id)
                await self._notice(updated, f"{updated.title} was not approved in time, so it was not set up.")

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
        await self._notice(updated, f"{updated.title}: {outcome}")

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
    async def edit(self, schedule_id: int, *, owner: bool, changes: Dict[str, Any]) -> Dict[str, Any]:
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
            await self._notice(updated, self._proposal_text(updated))
            message = f"Updated proposal sent for {updated.title}."
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
            return self.store.update_schedule(schedule_id, next_run=now)
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
                                              consecutive_failures=0)
        raise ValueError(f"unknown action: {action}")

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
            await self._notice(updated, f"{updated.title} has reached its end date and will not run again.")
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
        req = self.store.create_request(
            chat_id=schedule.chat_id, sender="Aaron", sender_name="Aaron", mode="schedule", subject="",
            original_message=schedule.prompt, summary=schedule.title, scopes=list(schedule.scopes),
            prompt=schedule.prompt, state="approved", task_id=schedule.task_id, schedule_id=schedule.id,
            schedule_kind=schedule.cadence_kind,
        )
        # The revision this run started from: finished() refuses to act on a signal
        # from an older revision if the owner edited the schedule while it ran.
        self.store.set_state(req.id, "approved", status={"schedule_revision": schedule.revision})
        next_run = cadence_module.next_run(schedule.cadence_kind, schedule.cadence, schedule.timezone, now)
        self.store.create_schedule_run(schedule.id, req.id, now)
        self.store.update_schedule(schedule.id, next_run=next_run, last_run=now, last_request_id=req.id,
                                   run_count=schedule.run_count + 1)
        self.store.task_event(schedule.task_id, "request", f"Scheduled run: {schedule.title}",
                              chat_id=schedule.chat_id, request_id=req.id, state="running")
        logger.info("[schedules] %s fired: request #%s", schedule.id, req.id)
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
        await self._notice(schedule, f'{schedule.title} is waiting for your yes to {what} since {when}. '
                                     'Say yes or no to that first.')
        self.store.set_state(running.id, running.state, status={**status, "held_skip_notified": True})

    async def _run_fired(self, req: Request) -> None:
        """manager.execute() runs detached (nothing here awaits it), so an exception
        inside it would otherwise vanish: the request stays "running" forever (stale,
        so later firings are skipped by request_running_for_schedule), the run row
        this firing created stays blank, and the owner hears nothing. Catch it, log
        it, and route it through finished() like any other outcome so the row is
        filled in, the schedule's consecutive-failure count moves, and he gets a
        notice."""
        try:
            await self.manager.execute(req)
        except Exception as exc:
            logger.exception("[schedules] run #%s for schedule %s raised", req.id, req.schedule_id)
            self.store.set_state(req.id, "failed", status={"error": str(exc)})
            await self.finished(req, {"ok": False, "error": str(exc), "result_line": "",
                                      "final_status": f"the run raised an exception: {exc}",
                                      "denied": [], "note": "", "tool_calls": [],
                                      "finished_at": time.time()})

    # -- restart recovery --------------------------------------------------
    async def interrupted(self, req: Request) -> None:
        """A scheduled request stranded mid-run by a process restart. Recorded as its
        own outcome ('interrupted'), never routed through finished()/run_outcome: an
        interruption is not a failure of the model's work, so it must not count toward
        the three-strikes pause, and the schedule stays active with its next_run
        intact. Re-fired once immediately only when that next run is not already
        fairly close, or when there is no regular next run to wait for (a one-time
        schedule, or a repeating one whose next_run was already consumed)."""
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
        await self._notice(schedule, f"{schedule.title} was interrupted by a restart. "
                                      f"Next check {_shown(schedule.next_run, schedule.timezone)}.")
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
        outcome = run_outcome(status)
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
        run_id = run_row["id"] if run_row else self.store.create_schedule_run(schedule.id, req.id, finished_at)
        self.store.finish_schedule_run(run_id, finished_at=finished_at, complete=complete, outcome=outcome,
                                       result_line=result_line, final_status=final_status, denied=denied,
                                       note=note, changed=changed, cost_usd=status.get("cost_usd"))
        logger.info("[schedules] %s run #%s finished: outcome=%s", schedule.id, run_id, outcome)
        started_revision = status.get("schedule_revision")
        if started_revision is None:
            started_revision = (req.status or {}).get("schedule_revision")
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
                       "error": status.get("error")}
            send = (not complete) or schedule.report_every_time or changed
            if send:
                await self._notice(schedule, self.notice_text(schedule, run_dict, None))
            return True
        if note:
            self.store.update_schedule(schedule.id, last_note=note)
        run_dict = {"outcome": outcome, "result_line": result_line, "final_status": final_status,
                   "denied": denied, "note": note, "changed": changed, "cost_usd": status.get("cost_usd"),
                       "error": status.get("error")}
        stop = bool(status.get("stop"))
        failures = 0 if complete else schedule.consecutive_failures + 1
        if stop:
            updated = self.store.update_schedule(schedule.id, state="ended", next_run=None,
                                                 last_outcome=result_line or "stopped itself",
                                                 consecutive_failures=0)
            logger.info("[schedules] %s ended: STATUS: STOP", schedule.id)
            await self._notice(updated, f"{updated.title}: {result_line or 'it stopped itself.'} "
                                        "It will not run again.")
            return True
        if failures >= 3:
            updated = self.store.update_schedule(schedule.id, state="paused", next_run=None,
                                                 consecutive_failures=failures, last_outcome=result_line)
            logger.info("[schedules] %s paused: three consecutive runs that could not finish", schedule.id)
            paused_run = dict(run_dict)
            paused_run["paused"] = True
            await self._notice(updated, self.notice_text(updated, paused_run, None))
            return True
        extra: Dict[str, Any] = {}
        if schedule.cadence_kind == "once":
            extra.update(state="ended", next_run=None)
        updated = self.store.update_schedule(schedule.id, consecutive_failures=failures,
                                             last_outcome=result_line, **extra)
        send = (not complete) or schedule.report_every_time or changed
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
        """result_line -> the run's own error in plain words -> the first sentence of
        final_status. Never degenerates to the bare word "failed": one of these three
        always has something to say, even if it is only "it could not finish"."""
        result_line = str(run.get("result_line") or "").strip()
        if result_line:
            return result_line
        err = str(run.get("error") or "").strip()
        if err:
            low = err.lower()
            if "timed out" in low or "timeout" in low:
                return "it ran out of time"
            return err
        final_status = _first_sentence(str(run.get("final_status") or ""))
        if final_status:
            return final_status
        return "it could not finish"

    def notice_text(self, schedule: Schedule, run: Dict[str, Any], next_run: Optional[float]) -> str:
        if run.get("paused"):
            return (f'{schedule.title} is paused after three runs that could not finish. '
                    'Say "start it again" when it is fixed.')
        outcome = run.get("outcome")
        result_line = run.get("result_line") or ""
        if outcome == "complete":
            text = f"{schedule.title}: {result_line or self._shown_text(run)}"
            if schedule.cadence_kind != "once" and next_run is not None:
                text += f" Next check {_shown(next_run, schedule.timezone)}."
            return text
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
                return (f'{schedule.title} could not finish: it is not allowed to do part of this. '
                        f'{result_line} Reply "let it do that" to try again.')
            labels = self._labels(sorted(scope_names))
            return (f"{schedule.title} could not finish: it is not allowed to {labels}. {result_line} "
                    f'Reply "let it {labels}" to fix that, or "stop it".')
        if outcome == "failed":
            remaining = max(0, 3 - schedule.consecutive_failures)
            when = _shown(next_run, schedule.timezone) if next_run is not None else "soon"
            reason = self._failure_reason(run)
            return (f"{schedule.title} could not finish: {reason}. I will try again at {when}. "
                    f"After {remaining} more misses I will pause it and tell you.")
        return f"{schedule.title}: {result_line}"
