"""Approved schedules as a source of ordinary gate requests."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import cron
from .store import Request, Schedule, sha256

logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "America/New_York"
MIN_CONTINUE_DELAY = 5 * 60
DEFAULT_MAX_RUNS = 20
DEFAULT_DEADLINE = 7 * 24 * 3600
REPORT_MODES = {"always", "changed"}
KINDS = {"once", "recurring", "continue"}
OWNER_INSTRUCTIONS = "--- AARON'S INSTRUCTIONS ---"


def _timestamp(value: Any, timezone_name: str = DEFAULT_TIMEZONE) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid date and time: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone_name))
    return dt.timestamp()


def _shown(when: Optional[float], timezone_name: str) -> str:
    if when is None:
        return "not set"
    return datetime.fromtimestamp(when, ZoneInfo(timezone_name)).strftime("%A, %B %-d, %Y at %-I:%M %p %Z")


def cadence(schedule: Schedule) -> str:
    if schedule.kind == "once":
        return f"once on {_shown(schedule.run_at, schedule.timezone)}"
    if schedule.kind == "continue":
        first = _shown(schedule.run_at or schedule.next_run, schedule.timezone)
        return f"continuing work, first run {first}"
    return f"{cron.describe(schedule.cron)} in {schedule.timezone}"


def next_three(schedule: Schedule, now: Optional[float] = None) -> List[float]:
    now = time.time() if now is None else now
    if schedule.kind == "recurring":
        return cron.next_fires(schedule.cron, now, schedule.timezone, 3)
    first = schedule.next_run if schedule.next_run is not None else schedule.run_at
    return [first] if first is not None and first >= now else []


class ScheduleService:
    def __init__(self, manager: Any):
        self.manager = manager
        self.store = manager.store

    def validate(self, spec: Dict[str, Any], *, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        kind = str(spec.get("kind") or "").strip().lower()
        if kind not in KINDS:
            raise ValueError("kind must be once, recurring, or continue")
        timezone_name = str(spec.get("timezone") or DEFAULT_TIMEZONE).strip()
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {timezone_name}") from exc
        title = " ".join(str(spec.get("title") or "").split())
        prompt = str(spec.get("prompt") or "").strip()
        if not title or not prompt:
            raise ValueError("title and prompt are required")
        report_mode = str(spec.get("report_mode") or "always").strip().lower()
        if report_mode not in REPORT_MODES:
            raise ValueError("report_mode must be always or changed")
        expression = str(spec.get("cron") or "").strip()
        run_at = _timestamp(spec.get("run_at"), timezone_name)
        if kind == "recurring":
            cron.parse(expression)
            next_run = cron.next_fire(expression, now, timezone_name)
        else:
            if kind == "once" and run_at is None:
                raise ValueError("a once schedule needs run_at")
            next_run = run_at if run_at is not None else now
        max_runs = int(spec.get("max_runs") or DEFAULT_MAX_RUNS)
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        deadline = _timestamp(spec.get("deadline"), timezone_name)
        if kind == "continue" and deadline is None:
            deadline = now + DEFAULT_DEADLINE
        if kind == "continue" and deadline is not None and deadline <= now:
            raise ValueError("deadline must be in the future")
        return {"kind": kind, "timezone": timezone_name, "title": title, "prompt": prompt,
                "report_mode": report_mode, "cron": expression, "run_at": run_at,
                "next_run": next_run, "max_runs": max_runs, "deadline": deadline}

    async def propose(self, *, owner: bool, chat_id: str, task_id: int, spec: Dict[str, Any],
                      scopes: List[str], mode: str = "imessage") -> Schedule:
        if not owner:
            raise PermissionError("only Aaron can create or change schedules")
        data = self.validate(spec)
        schedule = self.store.create_schedule(chat_id=chat_id, task_id=task_id, scopes=scopes, **data)
        await self._new_proposal(schedule, mode=mode)
        return schedule

    async def _new_proposal(self, schedule: Schedule, mode: str = "imessage") -> Request:
        req = self.store.create_request(
            chat_id=schedule.chat_id, sender="Aaron", sender_name="Aaron", mode=mode, subject="",
            original_message=schedule.prompt, summary=f"Approve schedule: {schedule.title}",
            scopes=schedule.scopes, prompt=schedule.prompt, state="pending", task_id=schedule.task_id,
            schedule_id=schedule.id, schedule_kind=schedule.kind,
        )
        self.store.set_state(req.id, "pending", status={"schedule_proposal": True, "schedule_id": schedule.id})
        self.store.update_schedule(schedule.id, last_request_id=req.id)
        self.store.task_event(schedule.task_id, "request", f"Schedule proposed: {schedule.title}",
                              chat_id=schedule.chat_id, request_id=req.id, state="waiting_aaron")
        await self.manager.send_to_approver(self.format_proposal(self.store.get_schedule(schedule.id), req.id))
        return self.store.get_request(req.id)  # type: ignore[return-value]

    def format_proposal(self, schedule: Schedule, request_id: int) -> str:
        fires = next_three(schedule)
        shown = "\n".join(f"  {_shown(t, schedule.timezone)}" for t in fires) or "  none"
        extra = ""
        if schedule.kind == "continue":
            extra = (f"\nRun limit: {schedule.max_runs}\n"
                     f"Deadline: {_shown(schedule.deadline, schedule.timezone)}")
        return (
            f"[Blatbot #{request_id}] Schedule proposal\n\n"
            f"Title: {schedule.title}\n"
            f"Cadence: {cadence(schedule)}\n"
            f"Next three fire times:\n{shown}\n\n"
            f"Exact prompt:\n{schedule.prompt}\n\n"
            f"Scopes: {', '.join(schedule.scopes)}\n"
            f"Report mode: {schedule.report_mode}{extra}\n\n"
            f"Reply \"#{request_id} yes\" to activate it or \"#{request_id} no\" to reject it."
        )

    def is_proposal_request(self, req: Request) -> bool:
        return bool(req.schedule_id and (req.status or {}).get("schedule_proposal"))

    async def decide_proposal(self, req: Request, decision: str, note: str = "") -> Dict[str, Any]:
        schedule = self.store.get_schedule(req.schedule_id or 0)
        if schedule is None:
            return {"ok": False, "state": req.state, "error": "the schedule no longer exists"}
        d = decision.strip().lower()
        if d in {"yes", "y", "ok", "approve", "run", "go"}:
            now = time.time()
            if schedule.kind == "recurring":
                next_run = cron.next_fire(schedule.cron, now, schedule.timezone)
            elif schedule.kind == "continue":
                next_run = max(now, schedule.run_at or now)
            else:
                next_run = schedule.run_at
            if schedule.kind == "once" and (next_run is None or next_run < now):
                next_run = now
            self.store.update_schedule(schedule.id, state="active", next_run=next_run,
                                       deadline=(schedule.deadline or now + DEFAULT_DEADLINE)
                                       if schedule.kind == "continue" else schedule.deadline)
            self.store.set_state(req.id, "done", status={"schedule_proposal": True, "approved": True,
                                                          "schedule_id": schedule.id})
            self.store.task_event(schedule.task_id, "approved", f"Schedule S{schedule.id} activated.",
                                  chat_id=schedule.chat_id, request_id=req.id, state="open")
            await self.manager.send_to_approver(f"Schedule S{schedule.id} is active: {schedule.title}")
            return {"ok": True, "state": "active", "error": ""}
        if d in {"no", "n", "stop", "drop", "reject", "decline"}:
            self.store.update_schedule(schedule.id, state="rejected", next_run=None)
            self.store.set_state(req.id, "rejected")
            self.store.task_event(schedule.task_id, "rejected", f"Schedule S{schedule.id} rejected.",
                                  chat_id=schedule.chat_id, request_id=req.id, state="open")
            await self.manager.send_to_approver(f"Rejected schedule S{schedule.id}: {schedule.title}")
            return {"ok": True, "state": "rejected", "error": ""}
        if d == "edit":
            if not note.strip():
                return {"ok": False, "state": req.state, "error": "an edit needs instructions"}
            base = schedule.prompt.split(f"\n\n{OWNER_INSTRUCTIONS}")[0].rstrip()
            prompt = f"{base}\n\n{OWNER_INSTRUCTIONS}\n{note.strip()}"
            edited = await self.edit(schedule.id, owner=True, changes={"prompt": prompt})
            self.store.set_state(req.id, "rejected")
            return {"ok": True, "state": edited.state, "error": ""}
        return {"ok": False, "state": req.state, "error": f"unknown decision {decision!r}"}

    async def edit(self, schedule_id: int, *, owner: bool, changes: Dict[str, Any]) -> Schedule:
        if not owner:
            raise PermissionError("only Aaron can create or change schedules")
        old = self.store.get_schedule(schedule_id)
        if old is None:
            raise ValueError(f"no schedule S{schedule_id}")
        for request in self.store.pending_requests_for_schedule(schedule_id):
            if self.is_proposal_request(request):
                self.store.set_state(request.id, "rejected")
        spec = {"title": old.title, "prompt": old.prompt, "kind": old.kind, "cron": old.cron,
                "run_at": old.run_at, "timezone": old.timezone, "report_mode": old.report_mode,
                "max_runs": old.max_runs, "deadline": old.deadline}
        spec.update({k: v for k, v in changes.items() if k != "scopes"})
        data = self.validate(spec)
        scopes = changes.get("scopes", old.scopes)
        updated = self.store.update_schedule(schedule_id, **data, scopes=scopes, state="proposed",
                                             revision=old.revision + 1, consecutive_failures=0)
        await self._new_proposal(updated)
        return self.store.get_schedule(schedule_id)  # type: ignore[return-value]

    async def action(self, schedule_id: int, action: str, *, owner: bool) -> Schedule:
        if not owner:
            raise PermissionError("only Aaron can create or change schedules")
        schedule = self.store.get_schedule(schedule_id)
        if schedule is None:
            raise ValueError(f"no schedule S{schedule_id}")
        action = action.strip().lower().replace(" ", "_")
        now = time.time()
        if action == "pause":
            if schedule.state != "active":
                raise ValueError(f"schedule S{schedule.id} is {schedule.state}; it can only be paused while active")
            return self.store.update_schedule(schedule.id, state="paused")
        if action == "resume":
            if schedule.state not in {"paused", "failed"}:
                raise ValueError(f"schedule S{schedule.id} is {schedule.state}; only paused or failed schedules can resume")
            if schedule.kind == "recurring":
                next_run = cron.next_fire(schedule.cron, now, schedule.timezone)
            else:
                next_run = max(now, schedule.next_run or schedule.run_at or now)
            return self.store.update_schedule(schedule.id, state="active", next_run=next_run,
                                              consecutive_failures=0)
        if action == "run_now":
            if schedule.state != "active":
                raise ValueError(f"schedule S{schedule.id} is {schedule.state}; it can only run now while active")
            return self.store.update_schedule(schedule.id, next_run=now)
        if action == "delete":
            for request in self.store.pending_requests_for_schedule(schedule.id):
                self.store.set_state(request.id, "rejected")
            self.store.delete_schedule(schedule.id)
            return schedule
        raise ValueError("action must be pause, resume, run_now, or delete")

    def proposal_expired(self, req: Request) -> None:
        if not req.schedule_id:
            return
        schedule = self.store.get_schedule(req.schedule_id)
        if schedule is None or schedule.state != "proposed":
            return
        self.store.update_schedule(
            schedule.id, state="rejected", next_run=None,
            last_outcome="The proposal expired unanswered.",
        )

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
        changes: Dict[str, Any] = {"last_outcome": outcome}
        if schedule.kind in {"once", "continue"}:
            changes.update(state="done", next_run=None)
        self.store.update_schedule(schedule.id, **changes)
        await self.manager.send_to_approver(
            f"Schedule S{schedule.id} run ended: {schedule.title}. {outcome}")

    async def tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        if str(self.store.settings().get("GATE_SCHEDULES_PAUSED", "0")).lower() in {"1", "true", "yes", "on"}:
            return
        for schedule in self.store.due_schedules(now):
            try:
                await self._fire(schedule, now)
            except Exception:
                logger.exception("schedule S%s failed during its tick", schedule.id)

    async def _fire(self, schedule: Schedule, now: float) -> None:
        if sha256(schedule.prompt) != schedule.prompt_sha256:
            self.store.update_schedule(schedule.id, state="failed", next_run=None,
                                       last_outcome="Prompt hash mismatch; refused to run.")
            await self.manager.send_to_approver(
                f"Schedule S{schedule.id} failed its prompt hash check and did not run: {schedule.title}")
            return
        if schedule.kind == "continue":
            if schedule.run_count >= schedule.max_runs:
                self.store.update_schedule(schedule.id, state="done", next_run=None,
                                           last_outcome="Stopped at the approved run limit.")
                await self.manager.send_to_approver(f"Schedule S{schedule.id} stopped at its run limit: {schedule.title}")
                return
            if schedule.deadline is not None and now >= schedule.deadline:
                self.store.update_schedule(schedule.id, state="done", next_run=None,
                                           last_outcome="Stopped at the approved deadline.")
                await self.manager.send_to_approver(f"Schedule S{schedule.id} reached its deadline: {schedule.title}")
                return
        running = self.store.request_running_for_schedule(schedule.id)
        if running is not None:
            note = f"Skipped a firing because request #{running.id} was still running."
            notes = list(schedule.notes) + [note]
            next_run = None if schedule.kind == "once" else (
                cron.next_fire(schedule.cron, now, schedule.timezone) if schedule.kind == "recurring"
                else now + MIN_CONTINUE_DELAY)
            self.store.update_schedule(schedule.id, notes=notes, next_run=next_run,
                                       state="done" if schedule.kind == "once" else schedule.state,
                                       last_outcome=note)
            self.store.task_event(schedule.task_id, "note", note, chat_id=schedule.chat_id)
            return
        prompt = schedule.prompt
        if schedule.kind == "continue" and schedule.notes:
            prompt += "\n\n--- PROGRESS NOTES FROM EARLIER RUNS ---\n" + "\n\n".join(schedule.notes)
        req = self.store.create_request(
            chat_id=schedule.chat_id, sender="Aaron", sender_name="Aaron", mode="schedule", subject="",
            original_message=schedule.prompt, summary=schedule.title, scopes=list(schedule.scopes), prompt=prompt,
            state="approved", task_id=schedule.task_id, schedule_id=schedule.id, schedule_kind=schedule.kind,
        )
        if schedule.kind == "recurring":
            next_run = cron.next_fire(schedule.cron, now, schedule.timezone)
        else:
            next_run = None
        self.store.update_schedule(schedule.id, next_run=next_run, last_run=now, last_request_id=req.id,
                                   run_count=schedule.run_count + 1)
        self.store.task_event(schedule.task_id, "request", f"Scheduled run #{schedule.run_count + 1}: {schedule.title}",
                              chat_id=schedule.chat_id, request_id=req.id, state="running")
        asyncio.create_task(self.manager.execute(req))

    async def finished(self, req: Request, status: Dict[str, Any]) -> bool:
        """Update the schedule and report. Returns True when normal delivery is replaced."""
        if not req.schedule_id or self.is_proposal_request(req):
            return False
        schedule = self.store.get_schedule(req.schedule_id)
        if schedule is None:
            return True
        ok = bool(status.get("ok"))
        failures = 0 if ok else schedule.consecutive_failures + 1
        result = str(status.get("summary") or status.get("error") or "")
        changes: Dict[str, Any] = {"consecutive_failures": failures, "last_outcome": result}
        if schedule.kind == "once":
            changes.update(state="done" if ok else "failed", next_run=None)
        elif schedule.kind == "continue":
            signal = status.get("continue") or {}
            if signal:
                delay = int(signal.get("delay_minutes") or 0) * 60
                note = str(signal.get("note") or "").strip()
                if delay < MIN_CONTINUE_DELAY:
                    ok = False
                    failures = schedule.consecutive_failures + 1
                    result = "The continue tool requested a delay shorter than 5 minutes."
                    changes.update(consecutive_failures=failures, last_outcome=result, state="failed", next_run=None)
                else:
                    notes = list(schedule.notes) + [note]
                    next_run = time.time() + delay
                    if schedule.run_count >= schedule.max_runs or (schedule.deadline and next_run > schedule.deadline):
                        changes.update(notes=notes, state="done", next_run=None)
                    else:
                        changes.update(notes=notes, state="active", next_run=next_run)
            else:
                changes.update(state="done" if ok else "failed", next_run=None)
        if failures >= 3:
            changes.update(state="paused", next_run=None)
        updated = self.store.update_schedule(schedule.id, **changes)
        if failures >= 3:
            await self.manager.send_to_approver(
                f"Schedule S{schedule.id} paused after three consecutive failed runs: {schedule.title}")
        from .jevagent import is_write_tool
        steps = status.get("steps") or []
        successful_writes = any(
            bool(step.get("ok")) and step.get("tool") != "mcp__host__schedule_continue"
            and is_write_tool(str(step.get("tool") or ""))
            for step in steps if isinstance(step, dict)
        )
        if not steps and ok:
            successful_writes = any(
                tool != "mcp__host__schedule_continue" and is_write_tool(str(tool))
                for tool in status.get("tool_calls") or []
            )
        final_continue = schedule.kind == "continue" and updated.state in {"done", "failed"}
        final_failed_once = schedule.kind == "once" and not ok
        if (schedule.report_mode == "always" or successful_writes or not ok
                or final_continue or final_failed_once):
            word = "done" if ok else "failed"
            await self.manager.send_to_approver(
                f"Schedule S{schedule.id} run {schedule.run_count} {word}: {schedule.title}\n{result}")
        return True

    async def command(self, text: str, *, owner: bool) -> bool:
        import re
        t = " ".join((text or "").strip().split())
        if t.lower() in {"list schedules", "schedules", "schedule list"}:
            if not owner:
                return False
            rows = self.store.schedules()
            lines = ["Schedules:"] + [f"S{s.id} {s.title} ({s.state}; {cadence(s)})" for s in rows]
            await self.manager.send_to_approver("\n".join(lines) if rows else "No schedules.")
            return True
        if t.lower() in {"pause all schedules", "pause schedules"}:
            if not owner:
                return False
            self.store.set_setting("GATE_SCHEDULES_PAUSED", "1")
            await self.manager.send_to_approver("All schedules are paused.")
            return True
        if t.lower() in {"resume all schedules", "resume schedules"}:
            if not owner:
                return False
            self.store.set_setting("GATE_SCHEDULES_PAUSED", "0")
            await self.manager.send_to_approver("All schedules may fire again.")
            return True
        # The word "schedule" or the S prefix is required, so "delete 5" about something
        # else is never read as a schedule command.
        m = re.match(r"^(pause|resume|delete|run now)\s+(?:schedule\s+s?|s)(\d+)$", t, re.I)
        if not m:
            return False
        if not owner:
            return False
        action = m.group(1).lower().replace(" ", "_")
        try:
            schedule = await self.action(int(m.group(2)), action, owner=True)
        except ValueError as exc:
            await self.manager.send_to_approver(str(exc))
            return True
        await self.manager.send_to_approver(f"Schedule S{schedule.id} {action.replace('_', ' ')}: {schedule.title}")
        return True
