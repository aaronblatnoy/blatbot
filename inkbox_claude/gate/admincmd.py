"""Everything the web console's pages can do, from Aaron's own chat.

The console has Home, Requests, People, Roles, Scopes, Tasks, Schedules, Group chats,
Health and Settings. Settings, a person's role and access, a group chat's mode and
schedule management were already chat commands (accesscmd.py, schedcmd.py). This module
adds the rest and one friendly entry point: ask about an area ("settings", "roles",
"schedules", "group chats", "people", "requests", or just "help") and Blatbot answers
with what the area is, what is set now, and the exact phrases to type.

Code, not a model: match() recognises the phrasing before any router or Jev call, and
handle() calls the same console/api.py functions the web console's routes call, so the
two surfaces cannot behave differently. Owner only (the caller checks is_approver, as
for every other admin command). Anything that removes something or widens a lot of
people's access is not done at once: it becomes a numbered hold ("#417 yes / #417 no"),
and asking for the same thing twice reuses the same hold.

Replies are written by wording.py, state what is true now (never "I just did"), name
old and new values on a change, and say the phrase that undoes it.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from . import accesscmd, settings as gate_settings, wording
from .store import role_names

logger = logging.getLogger(__name__)

# Settings shown in the guide, in this order (name, example value for a number setting).
_GUIDE_SETTINGS = ["GATE_GROUP_REPLY_MIN", "GATE_CONFIRM_SENDS", "GATE_SCHEDULES_PAUSED", "GATE_FAST_MODEL"]

_AREA_WORDS = {
    "settings": r"(?:settings?|preferences|options|config(?:uration)?)",
    "roles": r"(?:roles?|permissions?|access)",
    "schedules": r"(?:schedules?|scheduled jobs?|jobs|recurring tasks?|reminders?)",
    "groups": r"(?:group ?chats?|groups?|rooms?|chats)",
    "people": r"(?:people|contacts|the people list|my people)",
    "requests": r"(?:requests?|approvals?|pending|approvals? queue)",
}

_GUIDE_PATTERNS = []
for _area, _words in _AREA_WORDS.items():
    _GUIDE_PATTERNS.append((_area, re.compile(
        r"^(?:(?:show|open|give) me |tell me about |what are |what about )?(?:my |your |the )?" + _words + r"$", re.I)))
    _GUIDE_PATTERNS.append((_area, re.compile(
        r"^(?:help|guide|how do i|how to|what can i (?:do|change|set|control)|what can you (?:change|do)) ?(?:with|for|about|in|on|from)? ?(?:my |your |the )?"
        + _words + r"(?: help| guide)?$", re.I)))
    _GUIDE_PATTERNS.append((_area, re.compile(r"^" + _words + r" (?:help|guide|menu)$", re.I)))
_GUIDE_PATTERNS += [
    ("settings", re.compile(r"^what can i (?:change|set|control|turn on|turn off)(?: in (?:your |the )?settings)?$", re.I)),
    ("settings", re.compile(r"^what (?:can|do) (?:you|i) (?:change|have|control)(?: in settings)?$", re.I)),
    ("overview", re.compile(r"^(?:help|admin help|settings help|menu|what can i (?:do|control|change) (?:here|from here|by text|in chat)|"
                            r"what (?:else )?can you (?:do|change) for me|what can i ask you to change|admin|options)$", re.I)),
]


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

_NAME_RE = r"(?P<name>[A-Za-z][A-Za-z0-9' \-]{0,38}?)"

# (action, regex). Earlier wins. Named groups: name, role, target, ...
_PATTERNS: List[Any] = [(a, re.compile(p, re.I)) for a, p in [
    ("role_create", r"^(?:create|make|add|start) (?:me )?(?:a |an |a new |an new |the )?(?:new )?role (?:called|named|for) " + _NAME_RE + r"$"),
    ("role_create", r"^(?:create|make|add) (?:a |an )?(?:new )?" + _NAME_RE + r" role$"),
    ("role_create", r"^new role[: ] ?" + _NAME_RE + r"$"),
    ("schedules_pause_all", r"^pause (?:all|every|all of|all of my|all of your) (?:my |your )?schedules?$"),
    ("schedules_resume_all", r"^(?:resume|unpause|restart|turn back on) (?:all|every|all of|all of my|all of your) (?:my |your )?schedules?$"),
    ("roles_list", r"^(?:list|show|what are)(?: me)?(?: all)?(?: the| my| your)? roles\??$"),
    ("roles_list", r"^what roles (?:are there|do (?:we|you|i) have|exist)\??$"),
    ("role_show", r"^what (?:can|does|do) (?:the |a )?(?P<role>.+?) role (?:do|have|allow)\??$"),
    ("role_show", r"^(?:show|what is|what's|describe) (?:the )?(?P<role>.+?) role\??$"),
]]


def clean(text: str) -> str:
    return accesscmd.clean(text).rstrip("?").strip()


def match(text: str, store: Any) -> Optional[Dict[str, Any]]:
    """One of Aaron's admin phrasings as {action, ...}, or None. An action is claimed only
    when the part that says WHAT is real (an area word, a role that exists), so ordinary
    conversation never reaches it."""
    t = clean(text)
    if not t or len(t) > 200:
        return None
    for area, pattern in _GUIDE_PATTERNS:
        if pattern.match(t):
            return {"action": "guide", "area": area}
    for action, pattern in _PATTERNS:
        m = pattern.match(t)
        if not m:
            continue
        g = {k: (v.strip() if isinstance(v, str) else v) for k, v in m.groupdict().items()}
        if action in ("role_show",):
            if accesscmd.resolve_role(store, g.get("role") or "") is None:
                continue
        if action == "role_create":
            name = (g.get("name") or "").strip()
            if not name or name.lower() in ("new", "a", "an", "the") or " role" in name.lower():
                continue
        return {"action": action, **g}
    return None


# ---------------------------------------------------------------------------
# guides
# ---------------------------------------------------------------------------

def _label_lower(row: Dict[str, Any]) -> str:
    return " ".join(str(row.get("label") or row["name"]).split()).lower()


def _setting_rows_by_name() -> Dict[str, Dict[str, Any]]:
    return {r["name"]: r for r in gate_settings.describe()}


def _other_bool_word(value: Any) -> str:
    return "off" if wording.setting_value_text(value, "bool") == "on" else "on"


def setting_example(row: Dict[str, Any]) -> str:
    """One phrase that changes this setting to something else it accepts."""
    label = _label_lower(row)
    kind = row.get("kind")
    if kind == "bool":
        return f"turn {_other_bool_word(row.get('value'))} {label}"
    if kind == "number":
        cur = float(row.get("value") or 0)
        lo, hi = float(row.get("min") or 0), float(row.get("max") or 1)
        step = float(row.get("step") or 0.1)
        target = min(hi, cur + step) if cur + step <= hi else max(lo, cur - step)
        return f"set the {label} to {wording.setting_value_text(target, 'number')}"
    if kind == "choice":
        other = next((c for c in (row.get("choices") or []) if c != row.get("value")), row.get("value"))
        return f"set the {label} to {other}"
    return f"set the {label} to something else"


def guide_settings() -> str:
    rows = _setting_rows_by_name()
    picked = [rows[n] for n in _GUIDE_SETTINGS if n in rows]
    state = [wording.setting_line(r.get("label") or r["name"], r.get("value"), r.get("kind") or "") for r in picked]
    examples = [setting_example(r) for r in picked]
    examples.append(f"what is the {_label_lower(picked[0])}" if picked else "show all my settings")
    examples.append("show all my settings")
    return wording.guide("settings", state, examples)


def _role_rows(store: Any) -> List[Dict[str, Any]]:
    from ..console import api as console_api
    return console_api.roles(store)


def guide_roles(store: Any) -> str:
    rows = _role_rows(store)
    state = []
    for r in rows:
        n = int(r.get("people_count") or 0)
        if r.get("locked"):
            state.append(f"{r['name']}: you alone")
        else:
            state.append(f"{r['name']}: " + ("nobody" if n == 0 else f"{n} person" if n == 1 else f"{n} people"))
    editable = [r["name"] for r in rows if not r.get("locked")]
    sample = editable[0] if editable else "interns"
    person = next((p.get("display") for p in store.list_people()
                   if p.get("display") and not accesscmd._is_commander(store, p)), "Sean Parnell")
    examples = [
        "create a role called interns",
        f"what can the {sample} role do",
        f"make {person} a {sample}",
        f"who are the {sample}",
        "show all roles",
    ]
    return wording.guide("roles", state, examples)


def guide_groups(store: Any) -> str:
    groups = store.known_group_chats()
    counts: Dict[str, int] = {}
    for g in groups:
        counts[g.get("response_mode") or g.get("mode") or "mention"] = counts.get(g.get("response_mode") or g.get("mode") or "mention", 0) + 1
    state = [f"{len(groups)} group chats"] + [f"{n} on {mode}" for mode, n in sorted(counts.items())] if groups else ["No group chats yet."]
    title = (groups[0]["title"] if groups else "this chat")
    examples = [f"what mode is {title} in", f"set {title} to silent", f"set {title} to mention",
                f"set {title} to natural", "silent mode"]
    return wording.guide("groups", [", ".join(state)] if groups else state, examples)


def guide_people(store: Any) -> str:
    people = [p for p in store.list_people() if p.get("display")]
    name = next((p["display"] for p in people if not accesscmd._is_commander(store, p)), "Sean Parnell")
    return wording.guide("people", [f"{len(people)} people on the list"], [
        f"what can {name} do", f"set {name}'s role to general", "who are the general",
        f"let {name} read the TAMID calendar", "who has TAMID calendar access"])


def guide_schedules(store: Any) -> str:
    rows = store.schedules()
    counts: Dict[str, int] = {}
    for s in rows:
        counts[s.state] = counts.get(s.state, 0) + 1
    state = [", ".join(f"{n} {st}" for st, n in sorted(counts.items())) or "No schedules yet."]
    return wording.guide("schedules", state, [
        "what are you working on", "what's stuck", "what did you do today",
        "pause every schedule", "resume every schedule"])


def guide_requests(store: Any) -> str:
    n = len(store.pending())
    state = [f"{n} waiting on you" if n else "Nothing is waiting on you."]
    return wording.guide("requests", state, ["what needs me", "#417 yes", "#417 no", "#417 edit: make it shorter",
                                              "try again"])


def guide_for(area: str, store: Any, manager: Any = None) -> str:
    if area == "settings":
        return guide_settings()
    if area == "roles":
        return guide_roles(store)
    if area == "groups":
        return guide_groups(store)
    if area == "people":
        return guide_people(store)
    if area == "schedules":
        return guide_schedules(store)
    if area == "requests":
        return guide_requests(store)
    if area == "overview":
        return wording.guide_overview(["settings", "roles", "schedules", "group chats", "people", "requests"])
    return wording.guide(area, [], [])


# ---------------------------------------------------------------------------
# role create / list / show
# ---------------------------------------------------------------------------

def _members(store: Any, name: str) -> int:
    from ..console import api as console_api
    row = next((r for r in console_api.roles(store) if r["name"] == name), None)
    return int((row or {}).get("people_count") or 0)


def _role_row(store: Any, name: str) -> Optional[Dict[str, Any]]:
    return next((r for r in store.roles() if r["name"] == name), None)


def role_create(store: Any, name_text: str) -> str:
    from ..console import api as console_api
    name = " ".join(str(name_text or "").lower().replace("’", "'").split())
    if not name or len(name.split()) > 4 or not re.match(r"^[a-z][a-z0-9' \-]*$", name):
        return wording.role_name_invalid(name_text)
    if name == accesscmd.COMMANDER_ROLE:
        return wording.COMMANDER_NOT_FROM_CHAT
    existing = _role_row(store, name)
    if existing is not None:
        return wording.role_line(name, existing.get("scopes") or [], _members(store, name))
    try:
        console_api.upsert_role(store, {"name": name, "scopes": [], "note": ""})
    except console_api.ValidationError as exc:
        return str(exc)
    line = wording.role_line(name, [], 0, created=True)
    return line + "\n" + wording.undo_line(f"delete the {name} role")


def roles_list(store: Any) -> str:
    lines = []
    for r in _role_rows(store):
        if r.get("locked"):
            lines.append(f"{r['name']}: full control; you alone")
        else:
            lines.append(wording.role_line(r["name"], r.get("scopes") or [], int(r.get("people_count") or 0)).rstrip("."))
    return "\n".join(lines)


def role_show(store: Any, role_text: str) -> str:
    name = accesscmd.resolve_role(store, role_text)
    if name is None:
        return wording.role_unknown(role_text, [r["name"] for r in store.roles() if r["name"] != accesscmd.COMMANDER_ROLE])
    if name == accesscmd.COMMANDER_ROLE:
        return wording.commander_line("You")
    row = _role_row(store, name)
    return wording.role_line(name, (row or {}).get("scopes") or [], _members(store, name))


# ---------------------------------------------------------------------------
# pause or resume every schedule (the console's one global switch)
# ---------------------------------------------------------------------------

def schedules_global(ctx: Any, paused: bool) -> str:
    from ..console import api as console_api
    manager = getattr(ctx, "manager", None)
    if manager is None:
        return "I can't reach the schedules from here."
    was = bool(gate_settings.get("GATE_SCHEDULES_PAUSED"))
    if was != paused:
        console_api.schedules_global(manager, {"paused": paused})
        gate_settings.invalidate()
    return wording.schedules_global_line(paused, was=was if was != paused else None)


# ---------------------------------------------------------------------------
# the one entry point
# ---------------------------------------------------------------------------

async def handle(ctx: Any, cmd: Dict[str, Any]) -> str:
    """The reply to an admin command. ctx carries store, chat_id, here and (when the
    caller has them) manager and directory_client."""
    store = ctx.store
    action = cmd["action"]
    if action == "guide":
        return guide_for(cmd.get("area") or "overview", store, getattr(ctx, "manager", None))
    if action in ("schedules_pause_all", "schedules_resume_all"):
        return schedules_global(ctx, action == "schedules_pause_all")
    if action == "role_create":
        return role_create(store, cmd.get("name") or "")
    if action == "roles_list":
        return roles_list(store)
    if action == "role_show":
        return role_show(store, cmd.get("role") or "")
    return wording.ABILITY_UNKNOWN
