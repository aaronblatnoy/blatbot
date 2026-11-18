"""The one shape for every reply Aaron gets to an admin or status command by text.

Every reply built by admincmd.py, accesscmd.py and schedcmd.py, and every held change
or status line Blatbot sends him, passes through finish() before it goes out, so the
rules below hold in one place whatever wrote the words:

* The answer comes first, in one line. A list follows only when it helps: one item per
  line, the same shape on every line, at most LIST_CAP lines, then "and 12 more" with
  where to see the rest.
* Plain text. No decorative symbols at all (no shapes, bars, box lines, arrows used as
  ornament, bullets, middle dots or emoji), no markdown marks, no em dashes.
* People, schedules, settings and systems by name. Never a raw id, UUID, hash, scope
  slug, snake_case name, JSON or a machine timestamp. The only number shown is a
  request handle like "#521" when he needs it to answer.

when() and span() turn timestamps and durations into words ("today 9:00 AM",
"Mon Nov 23 9:00 AM", "3 hours"); listing() builds the lead-line-plus-list shape.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any, Iterable, List, Optional, Sequence

LIST_CAP = 8

# Decorative characters: arrows, box drawing, block elements, geometric shapes, misc
# symbols, dingbats, misc symbols and arrows, emoji, and the dot/bullet separators.
ORNAMENT_RE = re.compile(
    "[←-⇿─-╿▀-▟■-◿☀-⛿✀-➿"
    "⬀-⯿\U0001f000-\U0001faff·•‣⁃∙⋅️]")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
HEX_ID_RE = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{12,}\b", re.I)
SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b")
_MD_RE = re.compile(r"(\*\*|__|`)")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_LIST_MARK_RE = re.compile(r"^(\s*)[-*]\s+", re.M)
_DASH_RE = re.compile(r"[ \t]*[—―][ \t]*")
_EN_DASH_RE = re.compile(r"[ \t]+–[ \t]+")
_HANDLE_LINE_RE = re.compile(r"^#\d+\b")


def _snake_words(match: "re.Match[str]") -> str:
    word = match.group(0)
    try:
        from . import schedprop
        label = schedprop.label(word)
        if label and label != word and "_" not in label:
            return label
    except Exception:  # noqa: BLE001 - a plain-words fallback is fine
        pass
    return word.replace("_", " ")


def scrub(text: Any) -> str:
    """Text with every forbidden mark removed: ornaments, markdown, em dashes, raw ids,
    snake_case names and machine timestamps. Line breaks are kept."""
    out = str(text or "").replace("\r\n", "\n")
    out = ORNAMENT_RE.sub("", out)
    out = _MD_RE.sub("", out)
    out = _HEADER_RE.sub("", out)
    out = _LIST_MARK_RE.sub(r"\1", out)
    out = _DASH_RE.sub(", ", out)
    out = _EN_DASH_RE.sub(", ", out)
    out = UUID_RE.sub("", out)
    out = HEX_ID_RE.sub("", out)
    out = ISO_TS_RE.sub(lambda m: _iso_words(m.group(0)), out)
    # snake_case outside URLs only (a link is the one place an underscore may stay).
    out = " ".join(SNAKE_RE.sub(_snake_words, part) if "://" not in part else part
                   for part in out.split(" "))
    lines = [re.sub(r"[ \t]{2,}", " ", ln).strip().replace(" ,", ",").replace(" .", ".")
             for ln in out.split("\n")]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _iso_words(raw: str) -> str:
    try:
        return when(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return raw


def finish(text: Any, *, cap: int = 14) -> str:
    """The reply as it goes out: scrubbed, and no wall of text. A reply longer than
    `cap` lines keeps its first lines and its reply handle line, with "and N more"."""
    out = scrub(text)
    lines = out.split("\n")
    real = [ln for ln in lines if ln.strip()]
    if len(real) <= cap:
        return out
    handle = lines[-1] if _HANDLE_LINE_RE.match(lines[-1].strip()) else ""
    body = lines[:-1] if handle else lines
    kept: List[str] = []
    count = 0
    for ln in body:
        if ln.strip():
            count += 1
        if count > cap - 1:
            break
        kept.append(ln)
    dropped = len([ln for ln in body if ln.strip()]) - len([ln for ln in kept if ln.strip()])
    kept.append(f"and {dropped} more.")
    if handle:
        kept += ["", handle]
    return "\n".join(kept).strip()


def listing(lead: str, items: Sequence[Any], *, total: Optional[int] = None,
            rest: str = "", empty: str = "", cap: int = LIST_CAP) -> str:
    """A lead line, then at most `cap` items one per line, then "and N more" and where
    to see them (`rest`, e.g. "on the console's Inbox page")."""
    rows = [" ".join(str(i).split()) for i in items if str(i or "").strip()]
    total = len(rows) if total is None else total
    if not rows:
        return scrub(empty or lead)
    shown = rows[:cap]
    out = [lead.rstrip()] + shown
    extra = total - len(shown)
    if extra > 0:
        out.append(f"and {extra} more" + (f" {rest.strip()}" if rest else "") + ".")
    return scrub("\n".join(out))


def count(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _local(ts: float, timezone_name: str = "") -> datetime:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone_name or _default_tz())
        return datetime.fromtimestamp(float(ts), tz)
    except Exception:  # noqa: BLE001
        return datetime.fromtimestamp(float(ts))


def _default_tz() -> str:
    import os
    return os.getenv("GATE_TIMEZONE") or "America/New_York"


def _clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def when(ts: Any, timezone_name: str = "", *, now: Optional[float] = None) -> str:
    """A moment in words: "today 9:00 AM", "yesterday 4:10 PM", "tomorrow 8:00 AM",
    "Mon Nov 23 9:00 AM". "" for no time."""
    try:
        value = float(ts)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    dt = _local(value, timezone_name)
    today = _local(now if now is not None else time.time(), timezone_name).date()
    day = dt.date()
    if day == today:
        return f"today {_clock(dt)}"
    if day == today - timedelta(days=1):
        return f"yesterday {_clock(dt)}"
    if day == today + timedelta(days=1):
        return f"tomorrow {_clock(dt)}"
    return f"{dt.strftime('%a %b')} {dt.day} {_clock(dt)}"


def span(seconds: Any) -> str:
    """A duration in words: "40 seconds", "12 minutes", "3 hours", "2 days"."""
    try:
        s = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return ""
    if s < 90:
        return count(s, "second")
    if s < 90 * 60:
        return count(round(s / 60), "minute")
    if s < 48 * 3600:
        return count(round(s / 3600), "hour")
    return count(round(s / 86400), "day")


def and_list(items: Iterable[Any], word: str = "and") -> str:
    parts = [str(i) for i in items if str(i or "").strip()]
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return f"{parts[0]} {word} {parts[1]}"
    return ", ".join(parts[:-1]) + f", {word} {parts[-1]}"


GROUP_REFUSAL = "I only change or show my setup in our direct chat. Ask me there."
NOT_OWNER_REFUSAL = "Only Aaron can see or change my setup. Nothing was changed."
