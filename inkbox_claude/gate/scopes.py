"""Fixed tool scopes. A scope is the only unit of capability the executor
accepts. The global deny list (executor workdir .claude/settings.json) is
applied by Claude Code on top of every scope and cannot be widened here."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .. import identity as _identity

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


def _default_registry_path() -> Path:
    """SCOPES_PATH if set, else the legacy GATE_SCOPES_FILE alias, else the
    private registry at ~/blatbot/config/scopes.yaml if it exists, else the
    packaged generic example. Never the real personal registry by default:
    that file lives outside the repo and is named explicitly via env."""
    env_path = os.getenv("SCOPES_PATH") or os.getenv("GATE_SCOPES_FILE")
    if env_path:
        return Path(env_path).expanduser()
    home_registry = Path.home() / "blatbot" / "config" / "scopes.yaml"
    if home_registry.exists():
        return home_registry
    return _HERE / "scopes.example.yaml"


REGISTRY_PATH = _default_registry_path()
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

# Owner-facing labels and plain-language synonyms (schedprop.py), loaded straight
# from scopes.yaml so the one registry stays the single source for both the Jev
# scope tree and schedule proposal wording.
LABELS: Dict[str, str] = {
    name: str(spec["label"]) for name, spec in (_REGISTRY.get("scopes") or {}).items()  # type: ignore[union-attr]
    if spec.get("label")
}
SYNONYMS: Dict[str, List[str]] = {
    name: list(spec.get("synonyms") or []) for name, spec in (_REGISTRY.get("scopes") or {}).items()  # type: ignore[union-attr]
}
ASK: Dict[str, str] = {
    name: str(spec["ask"]) for name, spec in (_REGISTRY.get("scopes") or {}).items()  # type: ignore[union-attr]
    if spec.get("ask")
}
WHERE_THINGS_LIVE: List[str] = list(_REGISTRY.get("where_things_live") or [])  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Registry-driven personal content (servers, tool purposes, account rules,
# judgment wordlists, speech hints, stopwords, system labels). Every key here
# is optional: when the registry omits it, code falls back to a small
# generic default so the package runs with no personal data baked in. The
# real deployment's private registry fills these with the owner's content
# verbatim, so the assembled strings stay byte-identical to what used to be
# hardcoded.
# ---------------------------------------------------------------------------

_REGISTRY_SERVERS: Dict[str, Dict[str, object]] = dict(_REGISTRY.get("servers") or {})  # type: ignore[union-attr]
SERVERS: Dict[str, Dict[str, object]] = {
    name: {
        "home": _expand(str(spec.get("home") or "")),
        "label": str(spec.get("label") or ""),
        "google_account": str(spec.get("google_account") or ""),
    }
    for name, spec in _REGISTRY_SERVERS.items()
}


def google_account_for(server: str) -> str:
    """Which Google account a server's tools act as: ORG_ACCOUNT when the
    registry marks that server's google_account "org", else OWNER_ACCOUNT.
    Matches the old hardcoded behaviour: the org's workspace server (marked org) and every other
    Google-Workspace-backed server (owner)."""
    return ORG_ACCOUNT if SERVERS.get(server, {}).get("google_account") == "org" else OWNER_ACCOUNT


# Generic default account sentence for executor.py's two prompts, used only
# when the registry has no `account_rules` key.
_DEFAULT_ACCOUNT_RULES = (
    "every org-Google-Workspace tool takes user_google_email={ORG_ACCOUNT}; "
    "every owner-Google-Workspace tool takes user_google_email={OWNER_ACCOUNT} (the owner's own calendar and mail)."
)
ACCOUNT_RULES: str = _expand(str(_REGISTRY.get("account_rules") or _DEFAULT_ACCOUNT_RULES))

# Generic default capability list for taskpick.py's judge_action "the_assistant_can"
# list, used only when the registry has no `capabilities` key.
_DEFAULT_CAPABILITIES: List[str] = [
    "read and send messages on the assistant's own inbox",
    "look things up on the public web",
    "read and write files in the owner's second brain (notes vault)",
]
CAPABILITIES: List[str] = list(_REGISTRY.get("capabilities") or _DEFAULT_CAPABILITIES)  # type: ignore[arg-type]

# Generic default for taskpick.py's "count_as_yes" judgment wordlist.
_DEFAULT_COUNT_AS_YES: List[str] = []
COUNT_AS_YES: List[str] = list(_REGISTRY.get("count_as_yes") or _DEFAULT_COUNT_AS_YES)  # type: ignore[arg-type]

# router.py speech-recognition proper-noun hints. Empty default: with no
# registry, no personal proper-noun hints are assembled.
SPEECH_HINTS: str = str(_REGISTRY.get("speech_hints") or "")

# Personal stopwords (manager.py / jevagent.py), unioned into the code's own
# generic stopword sets.
STOPWORDS: List[str] = list(_REGISTRY.get("stopwords") or [])  # type: ignore[arg-type]

# tool name short-prefixes stripped before matching a verb in purpose_for();
# the generic ones (not tied to a club/org name) stay in code, the registry's
# full tuple (its own order preserved) replaces this default when present.
_DEFAULT_TOOL_PREFIXES = ("bs_", "ga_", "ig_", "li_", "inkbox_")
TOOL_PREFIXES: tuple = tuple(_REGISTRY.get("tool_prefixes") or _DEFAULT_TOOL_PREFIXES)  # type: ignore[arg-type]

# Scopes a registry entry marks `explicit_only: true`: never added by a blanket
# "permission for everything" grant or any other automatic widening, no matter
# how read-only their tools are (see wording.truly_read_only / manager's
# _blanket_grant_scopes). A scope lands here only when it reads something as
# broad and personal as an owner's own message history -- granted the same way
# any other scope is, just never for free.
EXPLICIT_ONLY_SCOPES: set = {
    name for name, spec in (_REGISTRY.get("scopes") or {}).items()  # type: ignore[union-attr]
    if spec.get("explicit_only")
}

# The Jev scope tree, derived from the registry's systems.
SCOPE_TREE_SYSTEMS: Dict[str, str] = {k: _expand(str(v.get("description") or k)) for k, v in (_REGISTRY.get("systems") or {}).items()}  # type: ignore[union-attr]
SCOPE_TREE_ALWAYS: Dict[str, List[str]] = {k: list(v.get("always") or []) for k, v in (_REGISTRY.get("systems") or {}).items()}  # type: ignore[union-attr]
SCOPE_TREE_LEAVES: Dict[str, Dict[str, object]] = {
    k: {leaf: (str(l.get("question") or leaf), list(l.get("grants") or [])) for leaf, l in (v.get("leaves") or {}).items()}
    for k, v in (_REGISTRY.get("systems") or {}).items()  # type: ignore[union-attr]
}

# console/api.py's short console labels for a system (_SYSTEM_SHORT_LABEL).
# A system entry's `label` key wins over whatever code defines for the same
# system name; systems the registry doesn't label fall back to code's own
# generic defaults there.
SYSTEM_LABELS: Dict[str, str] = {
    k: str(v["label"]) for k, v in (_REGISTRY.get("systems") or {}).items() if v.get("label")  # type: ignore[union-attr]
}


# ---------------------------------------------------------------------------
# Purposes for tools without a hand-written line: a template that names WHERE the
# tool acts (which account, site or system) before what it does, since telling
# the website admin from the public web, or the org's calendar from the owner's,
# is the judgment that goes wrong when a tool is described only by its verb.
# The per-server home text for personal/org servers (club sites, the owner's
# school/Google accounts, the owner's own PaaS) lives in the registry
# (servers.<name>.home); only the domain-neutral entries that make sense with
# no club/org at all stay here, built from the identity values.
# ---------------------------------------------------------------------------

_SERVER_HOME = {
    "playwright": f"the PUBLIC WEB in a headless browser (only for things outside {_identity.OWNER_FIRST_NAME}'s own systems)",
    "inkbox": f"{_identity.ASSISTANT_NAME_UPPER_POSSESSIVE} OWN INKBOX (its mailbox, phone line, iMessage and address book)",
    "host": f"the {_identity.SERVER_NAME_UPPER} SERVER itself",
}
_SERVER_HOME.update({name: str(spec.get("home") or "") for name, spec in SERVERS.items() if spec.get("home")})
_VERBS = {
    "list": "LIST", "get": "READ ONE", "search": "SEARCH", "read": "READ", "create": "CREATE", "update": "CHANGE",
    "delete": "DELETE", "replace": "REPLACE", "modify": "CHANGE", "append": "ADD ROWS TO", "insert": "INSERT INTO",
    "send": "SEND", "publish": "PUBLISH", "stage": "STAGE (prepare, not yet public)", "check": "CHECK", "manage": "MANAGE",
    "query": "QUERY", "mark": "MARK", "submit": "SUBMIT", "post": "POST", "reply": "REPLY IN", "pin": "PIN",
    "dismiss": "DISMISS", "download": "DOWNLOAD", "deploy": "DEPLOY", "restart": "RESTART", "stop": "STOP",
    "diagnose": "DIAGNOSE", "find": "FIND", "validate": "VALIDATE", "archive": "ARCHIVE", "grant": "GRANT",
    "revoke": "REVOKE", "set": "SET", "import": "IMPORT", "copy": "COPY", "export": "EXPORT", "format": "FORMAT",
}
_PREFIXES = TOOL_PREFIXES


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



# ---------------------------------------------------------------------------
# Knowledge scopes: discovered from the vault's own folder structure (see
# vaultscopes.py), not a list typed here. Merged into SCOPES/SCOPE_SYSTEM so the
# existing role storage, console scope validation and the scopes breakdown
# endpoint carry them with no change of their own: a knowledge scope is a scope
# like any other, grouped under the system name "vault". vault_read (above)
# keeps meaning the whole vault and is unaffected by this.
# ---------------------------------------------------------------------------

VAULT_TOOLS = ["mcp__host__vault_search", "mcp__host__vault_read", "mcp__host__vault_list"]
VAULT_SYSTEM = "vault"


def refresh_vault_scopes(force: bool = False) -> None:
    """Recompute the knowledge scopes from the vault folder and merge them into
    SCOPES/SCOPE_SYSTEM in place (mutated, not reassigned, so every module that
    imported the dict already sees the update). Safe to call often: discovery
    itself is cached against the folder's own mtimes."""
    from . import vaultscopes
    try:
        discovered = vaultscopes.discover_cached(force=force)
    except Exception:
        return
    stale = [name for name, system in SCOPE_SYSTEM.items() if system == VAULT_SYSTEM and name not in discovered]
    for name in stale:
        SCOPES.pop(name, None)
        SCOPE_SYSTEM.pop(name, None)
    for name, info in discovered.items():
        SCOPES[name] = {
            "description": info.description or info.title,
            "tools": list(VAULT_TOOLS),
            "read": True,
            "server": None,
            "include": [],
            "exclude": [],
            "vault_paths": list(info.prefixes),
            "vault_title": info.title,
        }
        SCOPE_SYSTEM[name] = VAULT_SYSTEM


refresh_vault_scopes()

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


def blocks_requester_send(req, tool_name: str, args: dict, protected: list) -> bool:
    """True when a send must be refused because the gateway already delivers the result
    to that person. A scheduled run is exempt: the owner approved the schedule, its prompt
    and its send scopes, so a message it sends him is the job itself, not a duplicate answer."""
    if getattr(req, "schedule_id", None):
        return False
    return sends_to_requester(tool_name, args, list(protected) + [req.sender, req.chat_id])


# What each tool is FOR, in the words a task uses. Shown to the judgment that picks
# the next tool instead of the servers' own (often long, implementation-flavoured)
# descriptions. Tools not listed fall back to the server's description. Only the
# domain-neutral entries (the owner's own vault/host/web tools, built from the
# identity values; the assistant's own inbox tools, built from ASSISTANT_NAME)
# stay in code. Entries that name a specific club/org account (a club's workspace,
# the owner's school account, ...) live in the registry's `tool_purpose` key and are merged in below,
# winning over anything code defines for the same tool name.
_CODE_TOOL_PURPOSE: Dict[str, str] = {
    "mcp__host__rows_where": "COUNT OR FILTER ROWS of a result ALREADY GATHERED (a sheet read, a calendar list, search hits) by a text they contain, optionally within one column. Exact counts; the matching rows come back. Use it after a read, never instead of one.",
    "mcp__host__vault_search": f"SEARCH {_identity.OWNER_FIRST_UPPER}'S SECOND BRAIN (his Obsidian vault of notes) for notes containing every word of a query: what he has written down about a project, a person, a decision, a preference. Read-only.",
    "mcp__host__vault_read": "READ ONE VAULT NOTE IN FULL by the path vault_search or vault_list printed, or by title. Read-only.",
    "mcp__host__vault_list": f"LIST THE NOTES in {_identity.OWNER_FIRST_NAME}'s vault, all or under one folder (Projects, Areas, Daily, Reference, Inbox, Archive). Read-only.",
    "mcp__host__host_status": f"WHAT IS RUNNING ON {_identity.SERVER_NAME_UPPER} (the server {_identity.ASSISTANT_NAME} runs on): containers, services, uptime, disk, memory, GPUs. Read-only.",
    # web
    # Generic default; the private registry's tool_purpose carries the owner's own wording.
    "mcp__playwright__browser_search": f"SEARCH THE PUBLIC WEB (last resort for facts that live OUTSIDE {_identity.OWNER_FIRST_NAME}'s own systems): a person's LinkedIn or employer, an outside organization's leadership, an article, a public page. Not for anything that lives in the owner's own sheets, forms, rosters, calendars or inboxes; use those tools first.",
    "mcp__playwright__browser_navigate": "OPEN A PUBLIC WEB PAGE by URL in the browser (a search result, a profile, an article). Only for things online; follow with browser_snapshot or browser_find to read it.",
    "mcp__playwright__browser_snapshot": "READ THE CURRENT PAGE: the full text and links of the page that is open.",
    "mcp__playwright__browser_find": "FIND TEXT ON THE CURRENT PAGE: locate a name, number or phrase on the open page and read around it.",
    "mcp__playwright__browser_wait_for": "WAIT for the page to finish loading or for text to appear (only after a page opened blank).",
    "mcp__playwright__browser_navigate_back": "GO BACK to the previous page.",
    "mcp__playwright__browser_click": "CLICK a button or link on the open page.",
    "mcp__playwright__browser_type": "TYPE into a field on the open page.",
    "mcp__playwright__browser_fill_form": "FILL a form's fields on the open page.",
    # Inkbox (the assistant's own mailbox, phone, contacts)
    "mcp__inkbox__inkbox_get_contact": f"LOOK UP A CONTACT in {_identity.ASSISTANT_NAME}'s address book by name, email or phone: returns their email, phone and notes. First stop for 'what is X's email / number'.",
    "mcp__inkbox__inkbox_list_contacts": f"LIST {_identity.ASSISTANT_NAME}'s contacts (names, emails, phones).",
    "mcp__inkbox__inkbox_list_emails": f"LIST recent emails in {_identity.ASSISTANT_NAME}'s own mailbox.",
    "mcp__inkbox__inkbox_send_email": f"SEND AN EMAIL from {_identity.ASSISTANT_NAME}'s mailbox to someone else.",
    "mcp__inkbox__inkbox_send_sms": f"SEND A TEXT (SMS) from {_identity.ASSISTANT_NAME}'s number to someone else.",
    "mcp__inkbox__inkbox_send_imessage": f"SEND AN IMESSAGE from {_identity.ASSISTANT_NAME}'s line to someone else.",
}
TOOL_PURPOSE: Dict[str, str] = {**_CODE_TOOL_PURPOSE, **{k: str(v) for k, v in (_REGISTRY.get("tool_purpose") or {}).items()}}  # type: ignore[union-attr]


def scope_for_tool(tool: str) -> Optional[str]:
    """The one permission to ask for when this tool was refused. Several scopes can carry
    the same tool (every notes folder carries the notes search, for one); asking for all
    of them is noise. A whole-system scope beats a folder-level one, and among those the
    narrowest, so the ask is the smallest grant that lets the tool run."""
    owners = [name for name, spec in SCOPES.items() if tool in (spec.get("tools") or [])]
    if not owners:
        return None
    return min(owners, key=lambda n: (":" in n, len(SCOPES[n].get("tools") or []), n))


# ---------------------------------------------------------------------------
# blatbot_admin: Blatbot's own settings (see gate/hosttools.py ADMIN_* tools).
# Deliberately NOT a key in scopes.yaml/SCOPES: add_role_scope/upsert_role
# validate scope names against SCOPES, so no role can ever be granted this
# scope through the console or any admin tool -- the owner's standing-scope
# grant in manager.py._standing_scopes (which appends BLATBOT_ADMIN_SCOPE only
# for the verified commander) is the one and only path to it. tools_for below
# is redefined, not edited in place, to keep this addition small and separate
# from the generated registry above.
# ---------------------------------------------------------------------------

BLATBOT_ADMIN_SCOPE = "blatbot_admin"
BLATBOT_ADMIN_TOOLS: List[str] = [
    "mcp__host__admin_list_settings",
    "mcp__host__admin_get_setting",
    "mcp__host__admin_set_setting",
    "mcp__host__admin_list_roles",
    "mcp__host__admin_set_role_scopes",
    "mcp__host__admin_set_person_role",
    "mcp__host__admin_schedule",
    "mcp__host__admin_set_chat_mode",
]

_generated_tools_for = tools_for


def tools_for(scopes: List[str]) -> List[str]:
    out = _generated_tools_for(scopes)
    if BLATBOT_ADMIN_SCOPE in (scopes or []):
        out += [t for t in BLATBOT_ADMIN_TOOLS if t not in out]
    return out
