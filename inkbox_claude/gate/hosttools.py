"""Read-only view of the machine Blatbot runs on. Fixed commands only, no shell:
the owner can ask what is running without giving any model a terminal."""

from __future__ import annotations

import ast
import asyncio
import os
import re
from pathlib import Path
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


def sdk_server(continue_state: Optional[Dict[str, Any]] = None) -> Any:
    """The same tool as an in-process MCP server for the Claude Code executor."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("host_status", DESCRIPTION, SCHEMA)
    async def _host_status(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": await host_status(args)}]}

    @tool("vault_search", VAULT_SEARCH_DESCRIPTION, VAULT_SEARCH_SCHEMA)
    async def _vault_search(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": vault_search(args)}]}

    @tool("vault_read", VAULT_READ_DESCRIPTION, VAULT_READ_SCHEMA)
    async def _vault_read(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": vault_read(args)}]}

    @tool("vault_list", VAULT_LIST_DESCRIPTION, VAULT_LIST_SCHEMA)
    async def _vault_list(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": vault_list(args)}]}

    tools = [_host_status, _vault_search, _vault_read, _vault_list]
    if continue_state is not None:
        @tool("schedule_continue", SCHEDULE_CONTINUE_DESCRIPTION, SCHEDULE_CONTINUE_SCHEMA)
        async def _schedule_continue(args: Dict[str, Any]) -> Dict[str, Any]:
            text = schedule_continue(args)
            continue_state.clear()
            continue_state.update({"delay_minutes": int(args["delay_minutes"]), "note": str(args["note"]).strip()})
            return {"content": [{"type": "text", "text": text}]}

        tools.append(_schedule_continue)
    return create_sdk_mcp_server(name="host", version="1.0.0", tools=tools)


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

SCHEDULE_CONTINUE_DESCRIPTION = (
    "SET THE NEXT WAKE-UP for this continuing scheduled job and leave a progress note. "
    "Call this only when more work remains. If it is not called, the continuing schedule ends."
)
SCHEDULE_CONTINUE_SCHEMA = {
    "type": "object", "required": ["delay_minutes", "note"], "properties": {
        "delay_minutes": {"type": "integer", "minimum": 5,
                          "description": "whole minutes from now until the next run; at least 5"},
        "note": {"type": "string", "description": "progress so far and what the next run should continue"},
    },
}


def schedule_continue(args: Dict[str, Any]) -> str:
    try:
        delay = int(args.get("delay_minutes"))
    except (TypeError, ValueError) as exc:
        raise ValueError("delay_minutes must be a whole number") from exc
    note = str(args.get("note") or "").strip()
    if delay < 5:
        raise ValueError("the next wake-up must be at least 5 minutes away")
    if not note:
        raise ValueError("a progress note is required")
    return f"Next wake-up requested in {delay} minutes. Progress note:\n{note}"

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


# ---------------------------------------------------------------------------
# The vault: Aaron's Obsidian second brain, a folder of markdown notes synced to
# this machine. Read-only here, and confined to the folder: a path that resolves
# outside it is refused.
# ---------------------------------------------------------------------------

VAULT_SEARCH_DESCRIPTION = (
    "SEARCH AARON'S SECOND BRAIN (his Obsidian vault of notes: projects, areas, people, decisions, preferences, "
    "daily notes) for notes whose title or text contains every word of the query. Returns each matching note's "
    "path and its matching lines. Read-only. Follow with vault_read for a whole note."
)
VAULT_SEARCH_SCHEMA = {"type": "object", "required": ["query"], "properties": {
    "query": {"type": "string", "description": "words a note must contain (case-insensitive, all of them); a name, a project, a topic"},
    "folder": {"type": "string", "description": "only look under this vault folder, e.g. Projects/Blatbot; omit for the whole vault"},
}}
VAULT_READ_DESCRIPTION = (
    "READ ONE NOTE IN FULL from Aaron's second brain (his Obsidian vault): give the note's path as vault_search "
    "or vault_list printed it, or its title. Read-only."
)
VAULT_READ_SCHEMA = {"type": "object", "required": ["path"], "properties": {
    "path": {"type": "string", "description": "the note's path inside the vault, e.g. Projects/Blatbot.md, or its title"},
}}
VAULT_LIST_DESCRIPTION = (
    "LIST THE NOTES in Aaron's second brain (his Obsidian vault), all of them or those under one folder. "
    "Returns paths. Read-only."
)
VAULT_LIST_SCHEMA = {"type": "object", "properties": {
    "folder": {"type": "string", "description": "a vault folder, e.g. Projects or Daily; omit for the whole vault"},
}}


def vault_dir() -> Path:
    return Path(os.getenv("BLATBOT_VAULT_DIR") or (Path.home() / "vault")).resolve()


def _vault_path(rel: str) -> Path:
    """A path inside the vault, or ValueError. Dot-directories (.obsidian, .git) are not notes."""
    root = vault_dir()
    p = (root / str(rel or "").strip().lstrip("/")).resolve()
    if p != root and root not in p.parents:
        raise ValueError("that path is outside the vault")
    if any(part.startswith(".") for part in p.relative_to(root).parts):
        raise ValueError("that path is not a note")
    return p


def _vault_notes(folder: str = "") -> List[Path]:
    base = _vault_path(folder)
    if not base.is_dir():
        raise ValueError(f"no vault folder named {folder!r}")
    root = vault_dir()
    return sorted(p for p in base.rglob("*.md")
                  if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts))


def vault_list(args: Dict[str, Any]) -> str:
    try:
        notes = _vault_notes(str(args.get("folder") or ""))
    except ValueError as exc:
        return f"ERROR: {exc}"
    root = vault_dir()
    return "\n".join([f"{len(notes)} notes"] + [str(p.relative_to(root)) for p in notes])


def vault_read(args: Dict[str, Any]) -> str:
    rel = str(args.get("path") or "").strip()
    if not rel:
        return "ERROR: a note path is required"
    root = vault_dir()
    try:
        p = _vault_path(rel)
        if not p.is_file() and not rel.lower().endswith(".md"):
            p = _vault_path(rel + ".md")
        if not p.is_file():
            # A bare title: every note with that file name, wherever it sits.
            want = Path(rel).name.lower().removesuffix(".md")
            hits = [n for n in _vault_notes() if n.stem.lower() == want]
            if len(hits) > 1:
                return "Several notes have that title; read one by path:\n" + "\n".join(str(n.relative_to(root)) for n in hits)
            if not hits:
                return f"ERROR: no note at {rel!r}. Use vault_search or vault_list to find it."
            p = hits[0]
    except ValueError as exc:
        return f"ERROR: {exc}"
    return f"# {p.relative_to(root)}\n\n" + p.read_text(encoding="utf-8", errors="replace")


def vault_search(args: Dict[str, Any]) -> str:
    words = [w for w in str(args.get("query") or "").lower().split() if w]
    if not words:
        return "ERROR: a query is required"
    try:
        notes = _vault_notes(str(args.get("folder") or ""))
    except ValueError as exc:
        return f"ERROR: {exc}"
    root = vault_dir()
    out: List[str] = []
    found = 0
    for p in notes:
        rel = str(p.relative_to(root))
        text = p.read_text(encoding="utf-8", errors="replace")
        hay = rel.lower() + "\n" + text.lower()
        if not all(w in hay for w in words):
            continue
        found += 1
        out.append(f"## {rel}")
        out += [f"{i}: {ln.strip()}" for i, ln in enumerate(text.splitlines(), 1)
                if any(w in ln.lower() for w in words)]
    return "\n".join([f"{found} of {len(notes)} notes contain {' '.join(words)!r}"] + out)
