"""Fixed tool scopes. A scope is the only unit of capability the executor
accepts. The global deny list (executor workdir .claude/settings.json) is
applied by Claude Code on top of every scope and cannot be widened here."""

from __future__ import annotations

import os
from typing import Any, Dict, List

# Which Google accounts the workspace tools act as. Set in the environment; the
# defaults are placeholders so no real address lives in the source.
ORG_ACCOUNT = os.getenv("GATE_ORG_GOOGLE_ACCOUNT") or "org@example.org"
OWNER_ACCOUNT = os.getenv("GATE_OWNER_GOOGLE_ACCOUNT") or "owner@example.edu"

# ---------------------------------------------------------------------------
# The registry. scopes.yaml is the single source: systems (the Jev scope tree),
# scopes (name, description, read/write, tools). A scope lists its tools
# explicitly, or names a server plus include/exclude glob patterns that
# `inkbox-claude scopes sync` resolves against the live server into
# scopes.resolved.json (committed, so runtime never connects to resolve).
# ---------------------------------------------------------------------------

import fnmatch
import json
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
REGISTRY_PATH = Path(os.getenv("GATE_SCOPES_FILE") or _HERE / "scopes.yaml")
RESOLVED_PATH = REGISTRY_PATH.with_name("scopes.resolved.json")


def load_registry() -> Dict[str, object]:
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_resolved() -> Dict[str, List[str]]:
    """Server -> full tool names, as last synced. Empty when never synced."""
    return {k: list(v) for k, v in _load_resolved_raw().items()}


def _load_resolved_raw() -> Dict[str, Dict[str, str]]:
    """Server -> {tool name: description}. Accepts the older list-only format."""
    try:
        with open(RESOLVED_PATH, encoding="utf-8") as fh:
            raw = json.load(fh) or {}
    except FileNotFoundError:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for server, v in raw.items():
        out[server] = {t: "" for t in v} if isinstance(v, list) else {str(t): str(d or "") for t, d in v.items()}
    return out


# Tool descriptions captured at sync time, so a run can describe its tools to the
# planner without launching a single MCP server. Empty string = not captured.
TOOL_DESCRIPTIONS: Dict[str, str] = {t: d for srv in _load_resolved_raw().values() for t, d in srv.items() if d}


def match_tools(server: str, available: List[str], include: List[str], exclude: List[str]) -> List[str]:
    """Resolve short-name glob patterns against a server's tool list; keeps order of `include`."""
    out: List[str] = []
    for pat in include:
        for t in available:
            short = t.split("__")[-1]
            if fnmatch.fnmatchcase(short, pat) and not any(fnmatch.fnmatchcase(short, ex) for ex in exclude or []) and t not in out:
                out.append(t)
    return out


def _expand(text: str) -> str:
    return (text or "").replace("{ORG_ACCOUNT}", ORG_ACCOUNT).replace("{OWNER_ACCOUNT}", OWNER_ACCOUNT)


def build_scopes(reg: Dict[str, object], resolved: Dict[str, List[str]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for name, spec in (reg.get("scopes") or {}).items():  # type: ignore[union-attr]
        spec = dict(spec)
        tools = list(spec.get("tools") or [])
        if not tools and spec.get("server"):
            avail = resolved.get(str(spec["server"]), [])
            tools = match_tools(str(spec["server"]), avail, list(spec.get("include") or []), list(spec.get("exclude") or []))
        out[name] = {"description": _expand(str(spec.get("description") or name)), "tools": tools,
                     "read": bool(spec.get("read")), "server": spec.get("server"),
                     "include": list(spec.get("include") or []), "exclude": list(spec.get("exclude") or [])}
    return out


_REGISTRY = load_registry()
SCOPES: Dict[str, Dict[str, object]] = build_scopes(_REGISTRY, load_resolved())
READ_SCOPES: List[str] = [k for k, v in SCOPES.items() if v.get("read")]
WHERE_THINGS_LIVE: List[str] = list(_REGISTRY.get("where_things_live") or [])  # type: ignore[arg-type]

# The Jev scope tree, derived from the registry's systems.
SCOPE_TREE_SYSTEMS: Dict[str, str] = {k: _expand(str(v.get("description") or k)) for k, v in (_REGISTRY.get("systems") or {}).items()}  # type: ignore[union-attr]
SCOPE_TREE_ALWAYS: Dict[str, List[str]] = {k: list(v.get("always") or []) for k, v in (_REGISTRY.get("systems") or {}).items()}  # type: ignore[union-attr]
SCOPE_TREE_LEAVES: Dict[str, Dict[str, object]] = {
    k: {leaf: (str(l.get("question") or leaf), list(l.get("grants") or [])) for leaf, l in (v.get("leaves") or {}).items()}
    for k, v in (_REGISTRY.get("systems") or {}).items()  # type: ignore[union-attr]
}


# ---------------------------------------------------------------------------
# Purposes for tools without a hand-written line: a template that names WHERE the
# tool acts (which account, site or system) before what it does, since telling
# the website admin from the public web, or TAMID's calendar from Aaron's, is
# the judgment that goes wrong when a tool is described only by its verb.
# ---------------------------------------------------------------------------

_SERVER_HOME = {
    "tamid-admin": "the TAMID at NYU WEBSITE ADMIN (the club's own site: its board list with positions and bios, events, members, settings, contact-form submissions, newsletter signups)",
    "sjba-admin": "the SJBA WEBSITE ADMIN (the Stern Jewish Business Association's own site: board list with positions and bios, events, members, settings, contact-form submissions, newsletter signups)",
    "tamid-drive": f"TAMID's GOOGLE WORKSPACE ({ORG_ACCOUNT}: the club's Drive, Sheets, Docs, Forms, Gmail, calendar)",
    "stern-drive": f"AARON's OWN NYU STERN GOOGLE ACCOUNT ({OWNER_ACCOUNT}: his personal calendar, Gmail and Drive, not the club's)",
    "tamid-instagram": "TAMID's INSTAGRAM account",
    "tamid-linkedin": "TAMID's LINKEDIN page",
    "analytics-admin": "TAMID's GOOGLE ANALYTICS property (website traffic configuration)",
    "brightspace": "AARON's NYU BRIGHTSPACE (the university course site: his courses, assignments, grades, discussions)",
    "coolify": "the BLACK-SKY COOLIFY (Aaron's self-hosted PaaS: deployed apps, services, databases, deployments, logs)",
    "coolify-vox": "the VOX COOLIFY instance (a second self-hosted PaaS)",
    "playwright": "the PUBLIC WEB in a headless browser (only for things outside Aaron's own systems)",
    "inkbox": "BLATBOT's OWN INKBOX (its mailbox, phone line, iMessage and address book)",
    "host": "the BLACK-SKY SERVER itself",
}
_VERBS = {
    "list": "LIST", "get": "READ ONE", "search": "SEARCH", "read": "READ", "create": "CREATE", "update": "CHANGE",
    "delete": "DELETE", "replace": "REPLACE", "modify": "CHANGE", "append": "ADD ROWS TO", "insert": "INSERT INTO",
    "send": "SEND", "publish": "PUBLISH", "stage": "STAGE (prepare, not yet public)", "check": "CHECK", "manage": "MANAGE",
    "query": "QUERY", "mark": "MARK", "submit": "SUBMIT", "post": "POST", "reply": "REPLY IN", "pin": "PIN",
    "dismiss": "DISMISS", "download": "DOWNLOAD", "deploy": "DEPLOY", "restart": "RESTART", "stop": "STOP",
    "diagnose": "DIAGNOSE", "find": "FIND", "validate": "VALIDATE", "archive": "ARCHIVE", "grant": "GRANT",
    "revoke": "REVOKE", "set": "SET", "import": "IMPORT", "copy": "COPY", "export": "EXPORT", "format": "FORMAT",
}
_PREFIXES = ("tamid_", "sjba_", "bs_", "ga_", "ig_", "li_", "inkbox_")


def purpose_for(name: str) -> str:
    """The hand-written purpose, else a templated one naming the tool's home and verb,
    else the server's own description, else the name."""
    if name in TOOL_PURPOSE:
        return TOOL_PURPOSE[name]
    parts = name.split("__")
    server, short = (parts[1], parts[2]) if len(parts) >= 3 else ("", name)
    home = _SERVER_HOME.get(server)
    bare = short
    for p in _PREFIXES:
        if bare.startswith(p):
            bare = bare[len(p):]
            break
    words = bare.split("_")
    verb = _VERBS.get(words[0]) if words else None
    obj = " ".join(words[1:]) if verb else " ".join(words)
    desc = (TOOL_DESCRIPTIONS.get(name) or "").strip()
    desc = (desc[:160] + ("..." if len(desc) > 160 else "")) if desc else ""
    if home:
        head = f"{verb} {obj}".strip() if verb else short.replace("_", " ")
        return f"{head.upper() if verb else head}: on {home}." + (f" {desc}" if desc else "")
    return desc or name


def _scope_system() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for sysname in SCOPE_TREE_SYSTEMS:
        for sc in SCOPE_TREE_ALWAYS.get(sysname, []):
            m.setdefault(sc, sysname)
        for _, (_, grants) in SCOPE_TREE_LEAVES.get(sysname, {}).items():
            for sc in grants:
                m.setdefault(sc, sysname)
    return m


SCOPE_SYSTEM: Dict[str, str] = _scope_system()
TOOL_SYSTEM: Dict[str, str] = {}
for _sc, _v in SCOPES.items():
    for _t in _v["tools"]:  # type: ignore[union-attr]
        TOOL_SYSTEM.setdefault(_t, SCOPE_SYSTEM.get(_sc, "other"))


def tools_by_system(tools: List[str]) -> Dict[str, List[str]]:
    """Group tool names by the registry system they belong to (order preserved)."""
    out: Dict[str, List[str]] = {}
    for t in tools:
        out.setdefault(TOOL_SYSTEM.get(t, "other"), []).append(t)
    return out


def registry_problems() -> List[str]:
    """Consistency checks: every granted scope exists, every scope has tools."""
    problems: List[str] = []
    for sysname, leaves in SCOPE_TREE_LEAVES.items():
        for leaf, (_, grants) in leaves.items():
            for g in grants:
                if g not in SCOPES:
                    problems.append(f"system {sysname} leaf {leaf} grants unknown scope {g}")
        for g in SCOPE_TREE_ALWAYS.get(sysname, []):
            if g not in SCOPES:
                problems.append(f"system {sysname} always-grants unknown scope {g}")
    for name in SCOPES:
        if name not in SCOPE_SYSTEM:
            problems.append(f"scope {name} belongs to no system in the tree")
    for name, v in SCOPES.items():
        if not v["tools"]:
            problems.append(f"scope {name} has no tools" + (" (run `inkbox-claude scopes sync`)" if v.get("server") else ""))
    return problems


ALWAYS_TOOLS = ["mcp__host__rows_where"]     # analysis over gathered results: granted with any scope


def tools_for(scopes: List[str]) -> List[str]:
    out: List[str] = []
    for s in scopes:
        for t in SCOPES.get(s, {"tools": []})["tools"]:  # type: ignore[union-attr,index]
            if t not in out:
                out.append(t)
    if out:
        out += [t for t in ALWAYS_TOOLS if t not in out]
    return out


_SEND_TOOLS = ("inkbox_send_imessage", "inkbox_send_sms", "inkbox_send_email")


def _norm(v: str) -> str:
    v = (v or "").strip().lower()
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits[-10:] if digits and "@" not in v and len(digits) >= 7 else v


def sends_to_requester(tool_name: str, args: dict, protected: list) -> bool:
    """True when a send tool is aimed at the person who asked (their address, phone
    or conversation). The gateway delivers results itself; an executor sending the
    answer as well means the requester gets it twice."""
    short = tool_name.split("__")[-1]
    if short not in _SEND_TOOLS:
        return False
    keys = {_norm(p) for p in protected if p}
    targets = []
    to = args.get("to")
    targets += to if isinstance(to, list) else ([to] if to else [])
    if args.get("conversation_id"):
        targets.append(args["conversation_id"])
    return any(_norm(str(t)) in keys for t in targets)


# What each tool is FOR, in the words a task uses. Shown to the judgment that picks
# the next tool instead of the servers' own (often long, implementation-flavoured)
# descriptions. Tools not listed fall back to the server's description.
TOOL_PURPOSE: Dict[str, str] = {
    "mcp__host__rows_where": "COUNT OR FILTER ROWS of a result ALREADY GATHERED (a sheet read, a calendar list, search hits) by a text they contain, optionally within one column. Exact counts; the matching rows come back. Use it after a read, never instead of one.",
    "mcp__host__vault_search": "SEARCH AARON'S SECOND BRAIN (his Obsidian vault of notes) for notes containing every word of a query: what he has written down about a project, a person, a decision, a preference. Read-only.",
    "mcp__host__vault_read": "READ ONE VAULT NOTE IN FULL by the path vault_search or vault_list printed, or by title. Read-only.",
    "mcp__host__vault_list": "LIST THE NOTES in Aaron's vault, all or under one folder (Projects, Areas, Daily, Reference, Inbox, Archive). Read-only.",
    "mcp__host__host_status": "WHAT IS RUNNING ON BLACK-SKY (the server Blatbot runs on): containers, services, uptime, disk, memory, GPUs. Read-only.",
    # web
    "mcp__playwright__browser_search": "SEARCH THE PUBLIC WEB (last resort for facts that live OUTSIDE Aaron's own systems): a person's LinkedIn or employer, an outside organization's leadership, an article, a public page. Not for anything that lives in TAMID/SJBA sheets, forms, rosters, calendars or inboxes; use those tools first.",
    "mcp__playwright__browser_navigate": "OPEN A PUBLIC WEB PAGE by URL in the browser (a search result, a profile, an article). Only for things online; follow with browser_snapshot or browser_find to read it.",
    "mcp__playwright__browser_snapshot": "READ THE CURRENT PAGE: the full text and links of the page that is open.",
    "mcp__playwright__browser_find": "FIND TEXT ON THE CURRENT PAGE: locate a name, number or phrase on the open page and read around it.",
    "mcp__playwright__browser_wait_for": "WAIT for the page to finish loading or for text to appear (only after a page opened blank).",
    "mcp__playwright__browser_navigate_back": "GO BACK to the previous page.",
    "mcp__playwright__browser_click": "CLICK a button or link on the open page.",
    "mcp__playwright__browser_type": "TYPE into a field on the open page.",
    "mcp__playwright__browser_fill_form": "FILL a form's fields on the open page.",
    # Google Drive / Sheets / Docs / Forms (TAMID account)
    "mcp__tamid-drive__search_drive_files": "FIND A FILE in TAMID Drive by name or words: sheets, docs, forms, folders. Returns names and ids. Use before opening anything whose id is unknown.",
    "mcp__tamid-drive__list_drive_items": "LIST the files inside one Drive folder by folder id.",
    "mcp__tamid-drive__list_spreadsheets": "LIST recent spreadsheets in TAMID Drive with their ids.",
    "mcp__tamid-drive__get_spreadsheet_info": "SHEET STRUCTURE: the tabs of a spreadsheet, their names and row/column counts, by spreadsheet id. Answers 'how many rows'.",
    "mcp__tamid-drive__read_sheet_values": "READ A SHEET'S CELLS: the rows of a spreadsheet tab (a roster, responses, a tracker) by spreadsheet id and range.",
    "mcp__tamid-drive__get_drive_file_content": "READ A WHOLE NON-SPREADSHEET FILE'S text by file id (a doc, a PDF). NEVER for a spreadsheet or a form: read_sheet_values and list_form_responses give those as rows; this would return an unreadable export.",
    "mcp__tamid-drive__get_doc_content": "READ A GOOGLE DOC's text by document id.",
    "mcp__tamid-drive__search_docs": "FIND A GOOGLE DOC by words in its name.",
    "mcp__tamid-drive__get_form": "FORM DEFINITION: a Google Form's title, questions and settings, by form id.",
    "mcp__tamid-drive__list_form_responses": "WHO ANSWERED A FORM: every response to a Google Form (respondent email, answers, time), by form id. Answers 'who filled it out', 'who has not responded'.",
    "mcp__tamid-drive__get_form_response": "ONE FORM RESPONSE in full, by form id and response id.",
    "mcp__tamid-drive__get_events": "TAMID CALENDAR EVENTS: list or search events on the TAMID calendar (a query word, a date range, or an event id). Returns titles, times, attendees, ids.",
    "mcp__tamid-drive__list_calendars": "WHICH CALENDARS the TAMID account has and their ids (needed before get_events on a non-primary calendar).",
    "mcp__tamid-drive__manage_event": "CREATE, UPDATE or DELETE an event on the TAMID calendar.",
    "mcp__tamid-drive__search_gmail_messages": "SEARCH THE TAMID INBOX: find emails by sender, words, or date (Gmail search syntax). Returns message ids and headers. Use to find someone's email address or what they sent.",
    "mcp__tamid-drive__get_gmail_message_content": "READ ONE EMAIL in the TAMID inbox in full, by message id.",
    "mcp__tamid-drive__get_gmail_thread_content": "READ A WHOLE EMAIL THREAD in the TAMID inbox, by thread id.",
    "mcp__tamid-drive__send_gmail_message": "SEND AN EMAIL from the TAMID account to someone else.",
    # Stern account
    "mcp__stern-drive__get_events": "AARON'S STERN CALENDAR: list or search his events (a query word, a date range). Returns titles, times, ids.",
    "mcp__stern-drive__list_calendars": "WHICH CALENDARS Aaron's Stern account has and their ids.",
    "mcp__stern-drive__manage_event": "CREATE, UPDATE or DELETE an event on Aaron's Stern calendar.",
    "mcp__stern-drive__search_gmail_messages": "SEARCH AARON'S STERN INBOX: find emails by sender, words, or date. Returns message ids and headers.",
    "mcp__stern-drive__get_gmail_message_content": "READ ONE EMAIL in Aaron's Stern inbox in full, by message id.",
    "mcp__stern-drive__send_gmail_message": "SEND AN EMAIL from Aaron's Stern account to someone else.",
    # Inkbox (Blatbot's own mailbox, phone, contacts)
    "mcp__inkbox__inkbox_get_contact": "LOOK UP A CONTACT in Blatbot's address book by name, email or phone: returns their email, phone and notes. First stop for 'what is X's email / number'.",
    "mcp__inkbox__inkbox_list_contacts": "LIST Blatbot's contacts (names, emails, phones).",
    "mcp__inkbox__inkbox_list_emails": "LIST recent emails in Blatbot's own mailbox.",
    "mcp__inkbox__inkbox_send_email": "SEND AN EMAIL from Blatbot's mailbox to someone else.",
    "mcp__inkbox__inkbox_send_sms": "SEND A TEXT (SMS) from Blatbot's number to someone else.",
    "mcp__inkbox__inkbox_send_imessage": "SEND AN IMESSAGE from Blatbot's line to someone else.",
    # site admin
    "mcp__tamid-admin__tamid_list_board_members": "TAMID BOARD ROSTER from the website: every board member with name, title, email, bio.",
    "mcp__tamid-admin__tamid_list_members": "TAMID MEMBER ROSTER from the website: club members with names and emails.",
    "mcp__tamid-admin__tamid_list_events": "TAMID EVENTS listed on the website.",
    "mcp__tamid-admin__tamid_list_site_config": "TAMID WEBSITE SETTINGS: application open/closed, deadlines, labels.",
    "mcp__sjba-admin__sjba_list_board_members": "SJBA BOARD ROSTER from the website: every board member with name, title, email, bio.",
    "mcp__sjba-admin__sjba_list_events": "SJBA EVENTS listed on the website.",
    "mcp__sjba-admin__sjba_list_upcoming_events": "SJBA UPCOMING EVENTS on the website.",
}
