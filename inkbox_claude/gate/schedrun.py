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
    """Pull RESULT:/NOTE:/NEEDS:/STATUS: out of a run's closing text.

    Returns {"result_line", "note", "needs", "status" (None|"OK"|"FAILED"|"STOP"|
    "BLOCKED"), "stop", "blocked", "final_status"} where final_status is the
    text with those blocks removed, kept whole (never cut) however long it
    runs. NEEDS is the run's own plain words for what would make the work
    possible -- only meaningful when status is BLOCKED, but parsed whenever
    present so a caller never has to guess at its absence."""
    lines = text.splitlines()
    headers: List[tuple] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("RESULT:"):
            headers.append((i, "RESULT"))
        elif s.startswith("NOTE:"):
            headers.append((i, "NOTE"))
        elif s.startswith("NEEDS:"):
            headers.append((i, "NEEDS"))
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
        if val in ("OK", "FAILED", "STOP", "BLOCKED"):
            status = val

    # RESULT/NOTE/NEEDS all appear (when present) before STATUS; each block
    # runs from its own header to whichever of the other two headers (or
    # STATUS) comes next, so a block never swallows a header that follows it.
    pre_status = [(i, kind) for i, kind in headers
                  if kind in ("RESULT", "NOTE", "NEEDS") and (status_idx is None or i < status_idx)]
    pre_status.sort(key=lambda t: t[0])

    def _next_boundary(idx: int) -> Optional[int]:
        for j, _ in pre_status:
            if j > idx:
                return j
        return status_idx

    def _block(header: str) -> str:
        for idx, kind in pre_status:
            if kind == header:
                end = _next_boundary(idx)
                chunk = lines[idx:end] if end is not None else lines[idx:]
                first = chunk[0].strip()[len(header) + 1:].strip()
                rest = chunk[1:]
                return "\n".join([first] + rest).strip()
        return ""

    result_line = _block("RESULT")
    note = _block("NOTE")
    needs = _block("NEEDS")

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
    for idx, kind in pre_status:
        end = _next_boundary(idx)
        remove_idxs.update(range(idx, end if end is not None else len(lines)))
    if status_idx is not None:
        remove_idxs.add(status_idx)
    if changed_idx is not None:
        remove_idxs.add(changed_idx)
    final_status = "\n".join(ln for i, ln in enumerate(lines) if i not in remove_idxs).strip()

    return {
        "result_line": result_line,
        "note": note,
        "needs": needs,
        "status": status,
        "stop": status == "STOP",
        "blocked": status == "BLOCKED",
        "final_status": final_status,
        "changed": changed,
    }
