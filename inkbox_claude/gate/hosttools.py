"""Read-only view of the machine Blatbot runs on. Fixed commands only, no shell:
the owner can ask what is running without giving any model a terminal."""

from __future__ import annotations

import ast
import asyncio
import re
from typing import Any, Dict, List, Optional

COMMANDS: Dict[str, list] = {
    "containers": ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"],
    "services": ["systemctl", "--user", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"],
    "system_services": ["systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"],
    "uptime_load": ["uptime"],
    "disk": ["df", "-h", "/", "/home"],
    "memory": ["free", "-h"],
    "gpus": ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
}

DESCRIPTION = ("WHAT IS RUNNING ON THIS SERVER (the machine Blatbot itself runs on, called black-sky): running "
               "containers, running services, uptime and load, disk, memory, GPUs. Read-only; pick which parts.")

SCHEMA = {"type": "object", "properties": {
    "parts": {"type": "array", "items": {"type": "string", "enum": list(COMMANDS)},
              "description": "which of: " + ", ".join(COMMANDS) + ". Omit for all."}}}


async def _run(cmd: list) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
        text = out.decode("utf-8", "replace").strip()
        return text or "(nothing)"
    except FileNotFoundError:
        return f"({cmd[0]} is not installed here)"
    except Exception as exc:
        return f"(failed: {exc})"


async def host_status(args: Dict[str, Any]) -> str:
    parts = [p for p in (args.get("parts") or list(COMMANDS)) if p in COMMANDS]
    outs = await asyncio.gather(*[_run(COMMANDS[p]) for p in parts])
    return "\n\n".join(f"## {p}\n{o}" for p, o in zip(parts, outs))


def sdk_server() -> Any:
    """The same tool as an in-process MCP server for the Claude Code executor."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("host_status", DESCRIPTION, SCHEMA)
    async def _host_status(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": await host_status(args)}]}

    return create_sdk_mcp_server(name="host", version="1.0.0", tools=[_host_status])


# ---------------------------------------------------------------------------
# rows_where: counting and filtering over a result already gathered. Code, not a
# model: a count over 71 rows is exact, and the reply writer gets the rows that
# matter instead of guessing from a slice.
# ---------------------------------------------------------------------------

ROWS_WHERE_DESCRIPTION = (
    "COUNT OR FILTER ROWS of a result already gathered (a sheet read, a calendar list, a roster, search hits): "
    "name the result (its `result_key`) and the text the rows must contain; optionally a column name for sheet "
    "rows. Returns the exact count, the total, and the matching rows. Use this instead of counting by eye."
)
ROWS_WHERE_SCHEMA = {"type": "object", "required": ["result_key", "contains"], "properties": {
    "result_key": {"type": "string", "description": "which gathered result to look in, e.g. result_of_read_sheet_values_3"},
    "contains": {"type": "string", "description": "text a row must contain (case-insensitive); a name, a date like 10/1, a word"},
    "column": {"type": "string", "description": "for sheet rows: only match inside this column (header name); omit to match anywhere"},
    "also_contains": {"type": "string", "description": "a second text the same row must also contain; omit if none"},
}}

_ROW_RE = re.compile(r"^\s*Row\s+(\d+):\s*(\[.*\])\s*$")


def _parse_row(line: str) -> Optional[List[str]]:
    m = _ROW_RE.match(line)
    if not m:
        return None
    try:
        v = ast.literal_eval(m.group(2))
        return [str(x) for x in v] if isinstance(v, list) else None
    except Exception:
        return None


def rows_where(args: Dict[str, Any], facts: Dict[str, Any]) -> str:
    key = str(args.get("result_key") or "").strip()
    needle = str(args.get("contains") or "").strip().lower()
    also = str(args.get("also_contains") or "").strip().lower()
    column = str(args.get("column") or "").strip().lower()
    if key not in facts:
        close = [k for k in facts if k.startswith("result_of_")]
        return f"ERROR: no gathered result named {key!r}. Gathered results: {', '.join(close) or '(none)'}"
    text = facts[key]
    text = text if isinstance(text, str) else str(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rows = [(ln, _parse_row(ln)) for ln in lines]
    sheet = [r for _, r in rows if r is not None]
    header = sheet[0] if sheet else None
    col_idx = None
    if column and header:
        for i, hname in enumerate(header):
            if column == hname.strip().lower() or column in hname.strip().lower():
                col_idx = i
                break
        if col_idx is None:
            return f"ERROR: no column named {column!r}. Columns: {header}"
    def hit(ln: str, cells: Optional[List[str]]) -> bool:
        if cells is not None and col_idx is not None:
            field = cells[col_idx].lower() if col_idx < len(cells) else ""
            return needle in field and (not also or also in ln.lower())
        low = ln.lower()
        return needle in low and (not also or also in low)
    if sheet:
        body = [(ln, cells) for ln, cells in rows if cells is not None and cells is not header]
        matches = [ln for ln, cells in body if hit(ln, cells)]
        total = len(body)
        kind = "sheet rows (header excluded)"
    else:
        candidates = [ln for ln in lines if ln.lstrip().startswith(("- ", "* ", "•")) or re.match(r"^\s*\d+[.)]", ln)] or lines
        matches = [ln for ln in candidates if hit(ln, None)]
        total = len(candidates)
        kind = "lines"
    where = f" in column {column!r}" if col_idx is not None else ""
    out = [f"{len(matches)} of {total} {kind} contain {needle!r}{where}" + (f" and {also!r}" if also else "") + "."]
    out += [ln.strip()[:400] for ln in matches[:80]]
    if len(matches) > 80:
        out.append(f"... and {len(matches) - 80} more matching rows")
    return "\n".join(out)
