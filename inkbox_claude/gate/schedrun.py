"""Pure helpers for scheduled-run prompts and closing-line parsing.

Kept free of the SDK and the store so they are trivial to unit test; the
executor (executor.py) is the only caller.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

_START_FMT = "%A, %B %d, %Y, %-I:%M %p %Z"


def start_time_line(started_at: float, timezone: str) -> str:
    """'This run started <weekday, date, time zone-abbrev> (<zone name>).' in the
    schedule's own time zone, not the server's. A moment that is one calendar day
    in UTC can be the previous or next day here; that is the point of saying it in
    local time at all."""
    local = datetime.fromtimestamp(started_at, ZoneInfo(timezone))
    return f"This run started {local.strftime(_START_FMT)} ({timezone})."


def build_prompt(prompt: str, started_at: float, timezone: str, note: Optional[str] = None) -> str:
    """The user prompt text for one scheduled run: the start-time line, then the
    schedule's own prompt, then the previous run's note (if any). No task ledger
    is ever added here; a scheduled run carries none."""
    parts = [start_time_line(started_at, timezone), prompt]
    if note and note.strip():
        parts.append("NOTE FROM THE PREVIOUS RUN:\n" + note.strip())
    return "\n\n".join(parts)


def parse_closing(text: str) -> Dict[str, Any]:
    """Pull RESULT:/NOTE:/STATUS: out of a run's closing text.

    Returns {"result_line", "note", "status" (None|"OK"|"FAILED"|"STOP"), "stop",
    "final_status"} where final_status is the text with those three blocks
    removed, kept whole (never cut) however long it runs."""
    lines = text.splitlines()
    headers: List[tuple] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("RESULT:"):
            headers.append((i, "RESULT"))
        elif s.startswith("NOTE:"):
            headers.append((i, "NOTE"))
        elif s.startswith("STATUS:"):
            headers.append((i, "STATUS"))
        elif s.startswith("CHANGED:"):
            headers.append((i, "CHANGED"))

    status_idx = None
    status = None
    for i, kind in headers:
        if kind == "STATUS":
            status_idx = i
    if status_idx is not None:
        val = lines[status_idx].strip()[len("STATUS:"):].strip().upper()
        if val in ("OK", "FAILED", "STOP"):
            status = val

    result_idx = None
    for i, kind in headers:
        if kind == "RESULT" and (status_idx is None or i < status_idx):
            result_idx = i
    note_idx = None
    for i, kind in headers:
        if kind == "NOTE" and (status_idx is None or i < status_idx):
            note_idx = i

    def _block(start_idx: Optional[int], header: str, stop_idx: Optional[int]) -> str:
        if start_idx is None:
            return ""
        end = stop_idx if stop_idx is not None else len(lines)
        chunk = lines[start_idx:end]
        first = chunk[0].strip()[len(header):].strip()
        rest = chunk[1:]
        return "\n".join([first] + rest).strip()

    result_end = note_idx if (note_idx is not None and note_idx > (result_idx or -1)) else status_idx
    result_line = _block(result_idx, "RESULT:", result_end)
    note = _block(note_idx, "NOTE:", status_idx)

    changed_idx = None
    for i, kind in headers:
        if kind == "CHANGED":
            changed_idx = i
    changed = None
    if changed_idx is not None:
        val = lines[changed_idx].strip()[len("CHANGED:"):].strip().lower()
        if val in ("yes", "no"):
            changed = val == "yes"

    remove_idxs = set()
    if result_idx is not None:
        remove_idxs.update(range(result_idx, result_end if result_end is not None else len(lines)))
    if note_idx is not None:
        remove_idxs.update(range(note_idx, status_idx if status_idx is not None else len(lines)))
    if status_idx is not None:
        remove_idxs.add(status_idx)
    if changed_idx is not None:
        remove_idxs.add(changed_idx)
    final_status = "\n".join(ln for i, ln in enumerate(lines) if i not in remove_idxs).strip()

    return {
        "result_line": result_line,
        "note": note,
        "status": status,
        "stop": status == "STOP",
        "final_status": final_status,
        "changed": changed,
    }
