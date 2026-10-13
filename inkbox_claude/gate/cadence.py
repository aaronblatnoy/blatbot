"""Cadence model for schedules, second version: one of six fixed kinds (once,
every, daily, weekdays, weekly, cron), the next-fire computation, validation,
and a plain-words description. The owner never sees cron strings or kind
plumbing -- this is the one place that turns a cadence into a time or words."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import cron

KINDS = {"once", "every", "daily", "weekdays", "weekly", "cron"}
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_WEEKDAY_NAMES = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
                  "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}


class CadenceError(ValueError):
    pass


def _zone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise CadenceError(f"unknown timezone: {timezone_name}") from exc


def _parse_time(text: str) -> Any:
    try:
        hour_text, minute_text = str(text).strip().split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except (ValueError, AttributeError) as exc:
        raise CadenceError(f"invalid time of day: {text!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise CadenceError(f"invalid time of day: {text!r}")
    return hour, minute


def _parse_once_at(text: str) -> datetime:
    try:
        return datetime.fromisoformat(str(text))
    except (ValueError, TypeError) as exc:
        raise CadenceError(f"invalid date and time: {text!r}") from exc


def validate(cadence_kind: str, cadence: Dict[str, Any]) -> None:
    """Raises CadenceError on anything malformed. Does not need a timezone: that
    is only consulted when computing an actual next-run time."""
    cadence = cadence or {}
    if cadence_kind not in KINDS:
        raise CadenceError(f"cadence_kind must be one of {sorted(KINDS)}")
    if cadence_kind == "once":
        if not cadence.get("at"):
            raise CadenceError("a once cadence needs 'at'")
        _parse_once_at(cadence["at"])
    elif cadence_kind == "every":
        try:
            minutes = int(cadence.get("minutes"))
        except (TypeError, ValueError) as exc:
            raise CadenceError("an every cadence needs an integer 'minutes'") from exc
        if minutes < 5:
            raise CadenceError("an every cadence must be at least 5 minutes")
    elif cadence_kind in ("daily", "weekdays"):
        _parse_time(cadence.get("time") or "")
    elif cadence_kind == "weekly":
        day = str(cadence.get("day") or "").strip().lower()
        if day not in WEEKDAYS:
            raise CadenceError(f"a weekly cadence needs 'day' to be one of {WEEKDAYS}")
        _parse_time(cadence.get("time") or "")
    elif cadence_kind == "cron":
        cron.parse(str(cadence.get("expr") or ""))


def _next_time_of_day(after_local: datetime, hour: int, minute: int) -> datetime:
    candidate = after_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= after_local:
        candidate += timedelta(days=1)
    return candidate


def next_run(cadence_kind: str, cadence: Dict[str, Any], timezone_name: str,
            after: Optional[float]) -> Optional[float]:
    """First firing strictly after `after` (epoch seconds), or None when the cadence
    has none left (a past `once`). `every` is anchored on `cadence.get('anchor')`
    when present -- the schedule's first run -- so repeated calls do not drift by
    however long each run took; without an anchor it is one interval after `after`."""
    validate(cadence_kind, cadence)
    tz = _zone(timezone_name)
    cadence = cadence or {}
    if cadence_kind == "once":
        at = _parse_once_at(cadence["at"])
        if at.tzinfo is None:
            at = at.replace(tzinfo=tz)
        ts = at.timestamp()
        if after is not None and ts <= after:
            return None
        return ts
    if cadence_kind == "cron":
        return cron.next_fire(str(cadence["expr"]), after if after is not None else 0.0, timezone_name)
    if cadence_kind == "every":
        minutes = int(cadence["minutes"])
        step = minutes * 60
        anchor = cadence.get("anchor")
        if anchor is None:
            # No anchor (a cadence set by an edit, or not yet started): one full
            # interval from now. Never `after` itself, which would make the schedule
            # due again on every tick.
            return (after if after is not None else 0.0) + step
        base = float(anchor)
        if after is None:
            return base
        if base > after:
            return base
        # Anchored arithmetic, not "after + step": a run that takes 10 minutes on a
        # 4-hour cadence must not push the next fire 10 minutes later every time.
        periods = int((after - base) // step) + 1
        return base + periods * step
    local_after = datetime.fromtimestamp(after, tz) if after is not None else datetime.fromtimestamp(0, tz)
    if cadence_kind == "daily":
        hour, minute = _parse_time(cadence["time"])
        local = _next_time_of_day(local_after, hour, minute)
        return local.timestamp()
    if cadence_kind == "weekdays":
        hour, minute = _parse_time(cadence["time"])
        local = _next_time_of_day(local_after, hour, minute)
        while local.weekday() >= 5:  # 5=Sat, 6=Sun
            local += timedelta(days=1)
        return local.timestamp()
    if cadence_kind == "weekly":
        hour, minute = _parse_time(cadence["time"])
        target_weekday = WEEKDAYS.index(str(cadence["day"]).strip().lower())
        local = _next_time_of_day(local_after, hour, minute)
        while local.weekday() != target_weekday:
            local += timedelta(days=1)
        return local.timestamp()
    raise CadenceError(f"unknown cadence_kind: {cadence_kind!r}")


def _clock(hour: int, minute: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    shown = hour % 12 or 12
    return f"{shown}:{minute:02d} {suffix}"


def describe(cadence_kind: str, cadence: Dict[str, Any], timezone_name: str) -> str:
    """Plain words for the owner. Never a cron string or a raw field name except
    the cron fallback, which is cron's own job to word."""
    validate(cadence_kind, cadence)
    cadence = cadence or {}
    if cadence_kind == "once":
        tz = _zone(timezone_name)
        at = _parse_once_at(cadence["at"])
        if at.tzinfo is None:
            at = at.replace(tzinfo=tz)
        return f"once on {at.strftime('%a %b %-d at %-I:%M %p')}"
    if cadence_kind == "every":
        minutes = int(cadence["minutes"])
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"every {hours} hour{'s' if hours != 1 else ''}"
        return f"every {minutes} minutes"
    if cadence_kind == "daily":
        hour, minute = _parse_time(cadence["time"])
        return f"daily at {_clock(hour, minute)}"
    if cadence_kind == "weekdays":
        hour, minute = _parse_time(cadence["time"])
        return f"weekdays at {_clock(hour, minute)}"
    if cadence_kind == "weekly":
        hour, minute = _parse_time(cadence["time"])
        day = str(cadence["day"]).strip().lower()
        return f"{_WEEKDAY_NAMES[day]}s at {_clock(hour, minute)}"
    if cadence_kind == "cron":
        return cron.describe(str(cadence["expr"]))
    raise CadenceError(f"unknown cadence_kind: {cadence_kind!r}")
