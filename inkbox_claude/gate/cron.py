"""Small five-field cron parser and timezone-aware next-fire calculation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import FrozenSet, Iterable, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class CronError(ValueError):
    pass


@dataclass(frozen=True)
class CronSpec:
    minute: FrozenSet[int]
    hour: FrozenSet[int]
    day: FrozenSet[int]
    month: FrozenSet[int]
    weekday: FrozenSet[int]
    day_any: bool
    weekday_any: bool


def _field(text: str, lo: int, hi: int, *, weekday: bool = False) -> FrozenSet[int]:
    values = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise CronError("empty cron list item")
        base, slash, step_text = part.partition("/")
        try:
            step = int(step_text) if slash else 1
        except ValueError as exc:
            raise CronError(f"invalid cron step: {part}") from exc
        if step < 1:
            raise CronError("cron steps must be positive")
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            left, right = base.split("-", 1)
            try:
                start, end = int(left), int(right)
            except ValueError as exc:
                raise CronError(f"invalid cron range: {part}") from exc
            if start > end:
                raise CronError(f"cron range runs backwards: {part}")
        else:
            try:
                start = end = int(base)
            except ValueError as exc:
                raise CronError(f"invalid cron value: {part}") from exc
        if start < lo or end > hi:
            raise CronError(f"cron value outside {lo}-{hi}: {part}")
        values.update(range(start, end + 1, step))
    if weekday and 7 in values:
        values.remove(7)
        values.add(0)
    return frozenset(values)


def parse(expression: str) -> CronSpec:
    parts = " ".join((expression or "").split()).split(" ")
    if len(parts) != 5:
        raise CronError("cron must have five fields: minute hour day month weekday")
    minute, hour, day, month, weekday = parts
    spec = CronSpec(
        _field(minute, 0, 59), _field(hour, 0, 23), _field(day, 1, 31),
        _field(month, 1, 12), _field(weekday, 0, 7, weekday=True),
        day == "*", weekday == "*",
    )
    if not spec.day_any and spec.weekday_any:
        longest = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                   7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        if not any(value <= longest[month_value]
                   for value in spec.day for month_value in spec.month):
            raise CronError("cron day never occurs in the selected month")
    return spec


def _day_matches(spec: CronSpec, local: datetime) -> bool:
    if local.month not in spec.month:
        return False
    dom = local.day in spec.day
    cron_weekday = (local.weekday() + 1) % 7
    dow = cron_weekday in spec.weekday
    if spec.day_any and spec.weekday_any:
        return True
    if spec.day_any:
        return dow
    if spec.weekday_any:
        return dom
    return dom or dow


def _matches(spec: CronSpec, local: datetime) -> bool:
    if local.minute not in spec.minute or local.hour not in spec.hour:
        return False
    return _day_matches(spec, local)


def next_fire(expression: str, after: float, timezone_name: str = "America/New_York") -> float:
    """First firing strictly after ``after``.

    UTC-minute iteration means nonexistent local times never appear. During a fall-back
    hour, fold=1 is ignored so one local minute fires only once.
    """
    spec = parse(expression)
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise CronError(f"unknown timezone: {timezone_name}") from exc
    cursor = datetime.fromtimestamp(after, timezone.utc).replace(second=0, microsecond=0)
    cursor += timedelta(minutes=1)
    limit = cursor + timedelta(days=366 * 8)
    while cursor <= limit:
        local = cursor.astimezone(tz)
        if not _day_matches(spec, local):
            next_midnight = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            advanced = next_midnight.astimezone(timezone.utc)
            cursor = max(cursor + timedelta(minutes=1), advanced)
            continue
        if local.fold == 0 and _matches(spec, local):
            return cursor.timestamp()
        cursor += timedelta(minutes=1)
    raise CronError("no cron firing found within eight years")


def next_fires(expression: str, after: float, timezone_name: str, count: int = 3) -> List[float]:
    out: List[float] = []
    cursor = after
    for _ in range(max(0, count)):
        cursor = next_fire(expression, cursor, timezone_name)
        out.append(cursor)
    return out


def _join(values: Iterable[int]) -> str:
    return ", ".join(str(v) for v in sorted(values))


def describe(expression: str) -> str:
    """Render a stored cron expression in plain words without trusting model wording."""
    spec = parse(expression)
    raw = " ".join(expression.split())
    minute, hour, day, month, weekday = raw.split(" ")
    if minute == "*" and hour == "*" and day == month == weekday == "*":
        return "every minute"
    if minute.startswith("*/") and hour == "*" and day == month == weekday == "*":
        return f"every {int(minute[2:])} minutes"
    if len(spec.minute) == 1 and len(spec.hour) == 1:
        h, m = next(iter(spec.hour)), next(iter(spec.minute))
        suffix = "AM" if h < 12 else "PM"
        shown = h % 12 or 12
        clock = f"{shown}:{m:02d} {suffix}"
        if day == month == weekday == "*":
            return f"every day at {clock}"
        if day == month == "*" and weekday != "*":
            names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            selected = [names[value] for value in sorted(spec.weekday)]
            days = selected[0] if len(selected) == 1 else ", ".join(selected[:-1]) + f" and {selected[-1]}"
            return f"every {days} at {clock}"
        if month == weekday == "*" and len(spec.day) == 1:
            return f"on day {next(iter(spec.day))} of every month at {clock}"
    return (f"cron {raw} (minutes {_join(spec.minute)}; hours {_join(spec.hour)}; "
            f"days {_join(spec.day)}; months {_join(spec.month)}; weekdays {_join(spec.weekday)})")
