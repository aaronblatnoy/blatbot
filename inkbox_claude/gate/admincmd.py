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

import json
import logging
import re
from types import SimpleNamespace
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

_VALUE = r"(?P<value>[\w.+\-]+@[\w.\-]+\.\w+|\+?\d[\d\-\s().]{6,}\d|@\w{3,})"
_NAME_RE = r"(?P<name>[A-Za-z][A-Za-z0-9' \-]{0,38}?)"

# (action, regex). Earlier wins. Named groups: name, role, target, ...
_PATTERNS: List[Any] = [(a, re.compile(p, re.I)) for a, p in [
    ("role_create", r"^(?:create|make|add|start) (?:me )?(?:a |an |a new |an new |the )?(?:new )?role (?:called|named|for) " + _NAME_RE + r"$"),
    ("role_create", r"^(?:create|make|add) (?:a |an )?(?:new )?" + _NAME_RE + r" role$"),
    ("role_create", r"^new role[: ] ?" + _NAME_RE + r"$"),
    ("schedules_pause_all", r"^pause (?:all|every|all of|all of my|all of your) (?:my |your )?schedules?$"),
    ("schedules_resume_all", r"^(?:resume|unpause|restart|turn back on) (?:all|every|all of|all of my|all of your) (?:my |your )?schedules?$"),
    ("role_rename", r"^rename (?:the )?(?P<role>.+?) role to (?P<name>.+)$"),
    ("role_delete", r"^(?:delete|remove|drop) (?:the )?(?P<role>.+?) role$"),
    ("role_level", r"^set (?:the )?(?P<role>.+?) role'?s? (?P<system>.+?) (?:access )?to (?P<level>none|read|read only|change|full)$"),
    ("role_send", r"^(?:let|allow) (?:the )?(?P<role>.+?) role send (?:as|from|with|through|on|in) (?P<system>.+)$"),
    ("role_unsend", r"^(?:stop|don't let|do not let) (?:the )?(?P<role>.+?) role (?:from )?send(?:ing)? (?:as|from|with|through|on|in) (?P<system>.+)$"),
    ("role_remove", r"^(?:take|remove|revoke) (?P<abilities>.+?) (?:away )?(?:from|out of|off) (?:the )?(?P<role>.+?) role$"),
    ("role_remove", r"^(?:the )?(?P<role>.+?) role (?:can no longer|can't|cannot|may not) (?P<abilities>.+)$"),
    ("role_add", r"^(?:let|allow) (?:the )?(?P<role>.+?) role (?:to )?(?P<abilities>(?:read|see|look|use|manage|edit|change|check|access|send|post|query|deploy|view|open|operate|act|write|update|search|book|schedule|text|email|message|control|run|pull|remember|recall)\b.+)$"),
    ("role_add", r"^give (?:the )?(?P<role>.+?) role (?:access to |permission to )?(?P<abilities>.+?)(?: access)?$"),
    ("role_add", r"^add (?P<abilities>.+?) to (?:the )?(?P<role>.+?) role$"),
    ("person_merge", r"^(?:merge|combine|link) (?P<person>.+?) (?:into|with) (?P<to>.+)$"),
    ("person_remove", r"^(?:delete|remove) (?P<person>.+?) from (?:the )?(?:people|people list|contacts)$"),
    ("person_remove", r"^delete (?:the )?person (?P<person>.+)$"),
    ("people_sync", r"^(?:refresh|sync|update|pull)(?: the)?(?: people| contacts)?(?: list)?(?: from inkbox| with inkbox)$"),
    ("people_sync", r"^(?:refresh|sync|update)(?: the)? (?:people|contacts)(?: list)?$"),
    ("people_list", r"^(?:list|show)(?: me)?(?: all)?(?: of)?(?: the| my)? people$"),
    ("people_list", r"^who(?:'s| is| are) (?:in|on) (?:the )?(?:people|people list)\??$"),
    ("contact_add", r"^add (?:the |a )?(?:(?P<kind>email|e-mail|phone|number|telegram|imessage|sms|handle)(?: address| number)? )?" + _VALUE + r" to (?P<person>.+)$"),
    ("contact_move", r"^move (?:the |a )?(?:(?P<kind>email|e-mail|phone|number|telegram|imessage|sms|handle)(?: address| number)? )?" + _VALUE + r" from (?P<person>.+?) to (?P<to>.+)$"),
    ("contact_remove", r"^(?:remove|delete|take off) (?:the |a )?(?:(?P<kind>email|e-mail|phone|number|telegram|imessage|sms|handle)(?: address| number)? )?" + _VALUE + r" (?:from|off) (?P<person>.+)$"),
    ("person_note", r"^(?:set|change|update) (?P<person>.+?)'s note to (?P<note>.+)$"),
    ("person_create", r"^add (?P<name>[A-Za-z][A-Za-z'.\- ]{1,40}?) to (?:the )?(?:people|people list|contacts)$"),
    ("person_create", r"^(?:add|create|make) (?:a |an )?(?:new )?person (?:called |named )?(?P<name>[A-Za-z][A-Za-z'.\- ]{1,40})$"),
    ("person_rename", r"^rename (?:the )?(?:person )?(?P<person>.+?) to (?P<name>.+)$"),
    ("groups_list", r"^(?:list|show)(?: me)?(?: all)?(?: of)?(?: the| my| your)? group ?chats$"),
    ("groups_list", r"^what (?:groups|group chats|rooms) are you in\??$"),
    ("requests_list", r"^(?:show|list)(?: me)?(?: all)?(?: the| my)? (?P<which>pending|failed|waiting) requests$"),
    ("requests_list", r"^what(?:'s| is| are) (?:pending|waiting)(?: on me| for me| for my approval)?\??$"),
    ("request_retry", r"^(?:retry|try again on|try) (?:request )?#?(?P<id>\d+)(?: again)?$"),
    ("health", r"^(?:health|health check|how are you doing|how(?:'s| is) blatbot(?: doing)?|are you ok(?:ay)?|are you healthy)\??$"),
    ("schedule_do", r"^(?P<sact>pause|resume|stop) (?:the )?(?P<target>.+?)(?: schedule| job)?$"),
    ("schedule_do", r"^(?P<sact>run) (?:the )?(?P<target>.+?)(?: schedule| job)? now$"),
    ("roles_list", r"^(?:list|show|what are)(?: me)?(?: all)?(?: the| my| your)? roles\??$"),
    ("roles_list", r"^what roles (?:are there|do (?:we|you|i) have|exist)\??$"),
    ("role_show", r"^what (?:can|does|do) (?:the |a )?(?P<role>.+?) role (?:do|have|allow)\??$"),
    ("role_show", r"^(?:show|what is|what's|describe) (?:the )?(?P<role>.+?) role\??$"),
]]


def clean(text: str) -> str:
    return accesscmd.clean(text).rstrip("?").strip()


def _norm_person_word(text: Any) -> str:
    return " ".join(str(text or "").split()).lower()


def _one_schedule(store: Any, target_text: str) -> Any:
    """The one schedule his words name, or None (nothing, or more than one): a schedule
    command is claimed only for a schedule that really exists."""
    from . import schedcmd
    try:
        found = schedcmd.resolve(store, target_text, "", None)
    except Exception:  # noqa: BLE001
        return None
    return found if found is not None and not isinstance(found, list) else None


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
        if action in ("role_show", "role_rename", "role_delete", "role_level", "role_send", "role_unsend",
                      "role_remove", "role_add"):
            if accesscmd.resolve_role(store, g.get("role") or "") is None:
                continue
        if action in ("role_add", "role_remove"):
            if not _abilities_for(g.get("abilities") or "", strict=True):
                continue
        if action in ("role_level", "role_send", "role_unsend") and _resolve_system(g.get("system") or "") is None:
            continue
        if action == "person_rename" and (g.get("person") or "").lower().endswith(" role"):
            continue
        if action in ("person_merge", "person_remove", "person_rename", "person_note", "contact_add",
                      "contact_remove", "contact_move"):
            if _norm_person_word(g.get("person")) in accesscmd._PRONOUNS or not g.get("person"):
                continue
        if action == "schedule_do":
            sact = (g.get("sact") or "").lower()
            found = _one_schedule(store, g.get("target") or "")
            if found is None:
                continue
            return {"action": "schedule_do", "sact": {"run": "run_now"}.get(sact, sact), "target": g.get("target")}
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
        f"let the {sample} role read the TAMID calendar",
        f"make {person} a {sample}",
        f"what can the {sample} role do",
        f"rename the {sample} role to volunteers",
        f"delete the {sample} role",
    ]
    return wording.guide("roles", state, examples)


def guide_groups(store: Any) -> str:
    groups = store.known_group_chats()
    counts: Dict[str, int] = {}
    for g in groups:
        counts[g.get("response_mode") or g.get("mode") or "mention"] = counts.get(g.get("response_mode") or g.get("mode") or "mention", 0) + 1
    state = [f"{len(groups)} group chats"] + [f"{n} on {mode}" for mode, n in sorted(counts.items())] if groups else ["No group chats yet."]
    title = (groups[0]["title"] if groups else "this chat")
    examples = ["show group chats", f"what mode is {title} in", f"set {title} to silent",
                f"set {title} to mention", f"set {title} to natural"]
    return wording.guide("groups", [", ".join(state)] if groups else state, examples)


def guide_people(store: Any) -> str:
    people = [p for p in store.list_people() if p.get("display")]
    name = next((p["display"] for p in people if not accesscmd._is_commander(store, p)), "Sean Parnell")
    return wording.guide("people", [f"{len(people)} people on the list"], [
        "show people", f"what can {name} do", f"add the email sean@example.com to {name}",
        f"rename {name} to {name} Jr", f"set {name}'s role to general", "refresh people from Inkbox"])


def guide_schedules(store: Any) -> str:
    rows = store.schedules()
    counts: Dict[str, int] = {}
    for s in rows:
        counts[s.state] = counts.get(s.state, 0) + 1
    state = [", ".join(f"{n} {st}" for st, n in sorted(counts.items())) or "No schedules yet."]
    title = rows[0].title if rows else "daily check"
    return wording.guide("schedules", state, [
        "what are you working on", "what's stuck", "what did you do today",
        f"pause the {title} schedule", f"run the {title} schedule now", "pause every schedule"])


def guide_requests(store: Any) -> str:
    n = len(store.pending())
    state = [f"{n} waiting on you" if n else "Nothing is waiting on you."]
    return wording.guide("requests", state, ["show pending requests", "#417 yes", "#417 no",
                                              "#417 edit: make it shorter", "show failed requests", "retry #417"])


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
# resolving his words: abilities (with memory grants), systems
# ---------------------------------------------------------------------------

def _abilities_for(text: str, *, strict: bool = False) -> List[str]:
    """Scope names his words name, plus the memory grants ("remember everything" ->
    memory:all, "remember group chats" -> memory:groups)."""
    found = list(accesscmd.resolve_abilities(text, strict=strict))
    low = accesscmd._norm(text)
    if re.search(r"\b(?:remember|recall)\b.*\b(?:everything|all)\b", low):
        found.append("memory:all")
    elif re.search(r"\b(?:remember|recall)\b.*\bgroup(?: chats)?\b", low):
        found.append("memory:groups")
    return list(dict.fromkeys(found))


def _resolve_system(text: str) -> Optional[str]:
    from ..console import api as console_api
    want = accesscmd._norm(re.sub(r"^(?:the|my|your)\s+", "", accesscmd._norm(text)))
    if not want:
        return None
    sets = console_api._system_scope_sets()
    for key in sets:
        label = accesscmd._norm(console_api._system_label(key))
        if want in (key, label, key.replace("_", " ")):
            return key
    hits = [k for k in sets if want in accesscmd._norm(console_api._system_label(k))
            or want in k.replace("_", " ")]
    return hits[0] if len(hits) == 1 else None


def _scope_words(scopes: List[str]) -> str:
    """Scopes for a sentence: registry labels, and plain words for memory grants."""
    plain = {"memory:all": "remember everything", "memory:groups": "remember group chats"}
    parts = [plain.get(sc) or (f"remember the chat {sc.split(':', 2)[2]}" if sc.startswith("memory:chat:")
                               else None) for sc in scopes]
    named = [sc for sc, w in zip(scopes, parts) if w is None]
    out = ([wording.schedprop.labels(named)] if named else []) + [w for w in parts if w]
    return wording._and_list(out)


def _role_state(store: Any, name: str, *, old_scopes: Any = None, created: bool = False) -> str:
    row = _role_row(store, name) or {}
    scopes = row.get("scopes") or []
    plain = [sc for sc in scopes if not sc.startswith("memory:")]
    line = wording.role_line(name, plain, _members(store, name), created=created,
                             old_scopes=None if old_scopes is None else [sc for sc in old_scopes
                                                                         if not sc.startswith("memory:")])
    mem = [sc for sc in scopes if sc.startswith("memory:")]
    if mem:
        line = line[:-1] + "; " + _scope_words(mem) + "."
    return line


# ---------------------------------------------------------------------------
# holds: "#417 yes / #417 no" for anything that removes or widens a lot
# ---------------------------------------------------------------------------

def open_hold(ctx: Any, kind: str, payload: Dict[str, Any], *, ask: str) -> str:
    """Park one change behind a numbered yes/no, as a request in the same queue the
    console's Requests page and the "#N yes" handler already use. Asking again for the
    same change returns the same hold (one request per distinct change)."""
    store = ctx.store
    sig = json.dumps([kind, payload], sort_keys=True)
    for req in store.pending():
        hold = (req.status or {}).get("admin_hold") or {}
        if hold.get("sig") == sig:
            return wording.hold_line(ask, req.id)
    task = store.create_task(f"Admin: {ask}"[:150])
    req, _created = store.create_request_once(
        chat_id=ctx.chat_id, sender="owner", sender_name="Aaron", mode="admin", subject="Admin change",
        original_message=f"admin hold {sig}", summary=ask, scopes=[], prompt=f"admin hold {sig}",
        state="pending", task_id=task["id"], bypass_dedupe=True)
    store.set_state(req.id, "pending", status={"admin_hold": {"kind": kind, "payload": payload, "sig": sig,
                                                              "ask": ask}})
    return wording.hold_line(ask, req.id)


async def apply_hold(manager: Any, hold: Dict[str, Any]) -> str:
    """Do the held change, after his yes. Same console/api functions as the console."""
    from ..console import api as console_api
    store = manager.store
    kind, pl = hold.get("kind"), hold.get("payload") or {}
    try:
        if kind == "role_delete":
            name = pl["role"]
            members = [m.get("person") or "" for m in console_api._role_members(store, name)]
            console_api.delete_role(store, {"name": name})
            return wording.role_gone_line(name, members)
        if kind == "role_grant":
            name = pl["role"]
            old = (_role_row(store, name) or {}).get("scopes") or []
            for sc in pl["scopes"]:
                console_api.add_role_scope(store, {"name": name, "scope": sc})
            return _role_state(store, name, old_scopes=old) + "\n" + wording.undo_line(
                f"take {pl['words']} away from the {name} role")
        if kind == "person_remove":
            console_api.remove_person_endpoint(store, {"id": pl["id"]})
            return wording.person_removed_line(pl["display"])
        if kind == "people_merge":
            console_api.link_people_endpoint(store, {"keep_id": pl["keep"], "merge_id": pl["merge"]})
            return wording.people_merged_line(pl["keep_name"], pl["merge_name"])
        if kind == "schedule_stop":
            schedule = await manager.schedules.action(int(pl["id"]), "stop", owner=True)
            return wording.schedule_line(schedule.title, "ended")
    except (console_api.ValidationError, PermissionError, KeyError, ValueError) as exc:
        return wording.change_refused(str(exc))
    return wording.change_refused("that change is not one I know how to apply")


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------

def _role_or_error(store: Any, role_text: str):
    name = accesscmd.resolve_role(store, role_text)
    if name is None:
        return None, wording.role_unknown(role_text, [r["name"] for r in store.roles()
                                                       if r["name"] != accesscmd.COMMANDER_ROLE])
    if name == accesscmd.COMMANDER_ROLE:
        return None, wording.COMMANDER_NOT_FROM_CHAT
    return name, None


def role_rename(store: Any, role_text: str, new_text: str) -> str:
    from ..console import api as console_api
    name, err = _role_or_error(store, role_text)
    if err:
        return err
    new = " ".join(str(new_text or "").lower().split())
    if not new or len(new.split()) > 4 or not re.match(r"^[a-z][a-z0-9' \-]*$", new):
        return wording.role_name_invalid(new_text)
    if new == name:
        return _role_state(store, name)
    if _role_row(store, new) is not None:
        return wording.role_name_taken(new)
    try:
        console_api.rename_role_endpoint(store, {"name": name, "new_name": new})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    return _role_state(store, new)[:-1] + f" (was called {name}).\n" + wording.undo_line(
        f"rename the {new} role to {name}")


def role_delete(ctx: Any, role_text: str) -> str:
    from ..console import api as console_api
    name, err = _role_or_error(ctx.store, role_text)
    if err:
        return err
    members = [m.get("person") or "" for m in console_api._role_members(ctx.store, name)]
    return open_hold(ctx, "role_delete", {"role": name}, ask=wording.hold_role_delete(name, members))


def role_scopes(ctx: Any, role_text: str, abilities_text: str, *, add: bool) -> str:
    from ..console import api as console_api
    store = ctx.store
    name, err = _role_or_error(store, role_text)
    if err:
        return err
    wanted = _abilities_for(abilities_text)
    if not wanted:
        return wording.ABILITY_UNKNOWN
    row = _role_row(store, name) or {}
    old = list(row.get("scopes") or [])
    words = " ".join(str(abilities_text).split())
    if add:
        todo = [sc for sc in wanted if sc not in old]
        if not todo:
            return _role_state(store, name)
        wide = [sc for sc in todo if sc.startswith("memory:")
                or (sc in console_api.SCOPES and console_api._scope_kind(sc, console_api.SCOPES[sc]) == "send")]
        if wide:
            return open_hold(ctx, "role_grant", {"role": name, "scopes": todo, "words": words},
                             ask=wording.hold_role_grant(name, _scope_words(todo), _members(store, name)))
        for sc in todo:
            console_api.add_role_scope(store, {"name": name, "scope": sc})
        return _role_state(store, name, old_scopes=old) + "\n" + wording.undo_line(
            f"take {words} away from the {name} role")
    todo = [sc for sc in wanted if sc in old]
    if not todo:
        return _role_state(store, name)
    for sc in todo:
        console_api.remove_role_scope(store, {"name": name, "scope": sc})
    return _role_state(store, name, old_scopes=old) + "\n" + wording.undo_line(
        f"give the {name} role access to {words}")


def role_level(store: Any, role_text: str, system_text: str, level_text: str) -> str:
    from ..console import api as console_api
    name, err = _role_or_error(store, role_text)
    if err:
        return err
    system = _resolve_system(system_text)
    level = {"read only": "read", "full": "change"}.get(accesscmd._norm(level_text), accesscmd._norm(level_text))
    old = (_role_row(store, name) or {}).get("scopes") or []
    try:
        console_api.roles_set_level(store, {"role": name, "system": system, "level": level})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    label = console_api._system_label(system)
    return _role_state(store, name, old_scopes=old) + "\n" + wording.undo_line(
        f"set the {name} role's {label.lower()} to none")


def role_send(store: Any, role_text: str, system_text: str, on: bool) -> str:
    from ..console import api as console_api
    name, err = _role_or_error(store, role_text)
    if err:
        return err
    system = _resolve_system(system_text)
    old = (_role_row(store, name) or {}).get("scopes") or []
    try:
        console_api.roles_set_send(store, {"role": name, "system": system, "on": on})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    label = console_api._system_label(system).lower()
    undo = (f"stop the {name} role sending from {label}" if on else f"let the {name} role send from {label}")
    return _role_state(store, name, old_scopes=old) + "\n" + wording.undo_line(undo)


# ---------------------------------------------------------------------------
# people
# ---------------------------------------------------------------------------

def _person(store: Any, text: str):
    return accesscmd._one_person(store, text)


def people_list(store: Any) -> str:
    people = [p for p in store.list_people() if p.get("display")]
    if not people:
        return "No one is in People yet."
    shown = [p["display"] for p in people[:25]]
    more = f", and {len(people) - 25} more" if len(people) > 25 else ""
    return f"{len(people)} people: " + ", ".join(shown) + more + "."


def person_create(store: Any, name_text: str) -> str:
    from ..console import api as console_api
    name = " ".join(str(name_text or "").split())
    if not name:
        return wording.person_name_missing()
    same = [p for p in store.list_people() if accesscmd._norm(p.get("display")) == accesscmd._norm(name)]
    if same:
        return wording.person_line_plain(same[0].get("display") or name, new=False)
    console_api.create_person_endpoint(store, {"display": name, "role": accesscmd.BASELINE_ROLE})
    return wording.person_line_plain(name, new=True) + "\n" + wording.undo_line(f"delete {name} from people")


def person_rename(store: Any, person_text: str, new_text: str) -> str:
    from ..console import api as console_api
    person, err = _person(store, person_text)
    if err:
        return err
    new = " ".join(str(new_text or "").split())
    if not new:
        return wording.person_name_missing()
    old = person.get("display") or ""
    if old == new:
        return wording.person_line_plain(new, new=False)
    console_api.update_person_endpoint(store, {"id": person["id"], "display": new})
    return wording.person_line_plain(new, new=False, was=old) + "\n" + wording.undo_line(f"rename {new} to {old}")


def person_note(store: Any, person_text: str, note: str) -> str:
    from ..console import api as console_api
    person, err = _person(store, person_text)
    if err:
        return err
    old = person.get("note") or ""
    new = " ".join(str(note or "").split())
    if new != old:
        console_api.update_person_endpoint(store, {"id": person["id"], "note": new})
    return wording.person_note_line(person.get("display") or "", new, was=old if new != old else None)


def _kind_for(store: Any, kind_text: str, value: str) -> str:
    kind = accesscmd._norm(kind_text).replace("e-mail", "email")
    kind = {"number": "phone", "handle": "telegram"}.get(kind, kind)
    if kind:
        return kind
    if "@" in value and "." in value.split("@")[-1]:
        return "email"
    if value.startswith("@"):
        return "telegram"
    return "phone"


def contact_add(store: Any, person_text: str, kind_text: str, value: str) -> str:
    from ..console import api as console_api
    person, err = _person(store, person_text)
    if err:
        return err
    kind = _kind_for(store, kind_text, value)
    try:
        res = console_api.add_contact_endpoint(store, {"person_id": person["id"], "kind": kind, "value": value})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    if res.get("conflict"):
        return wording.contact_conflict(value, res.get("existing_person_display") or "someone else")
    shown = (res.get("contact") or {}).get("value") or value
    return wording.contact_line(person.get("display") or "", kind, shown) + "\n" + wording.undo_line(
        f"remove the {kind} {value} from {person.get('display')}")


def _find_contact(person: Dict[str, Any], value: str) -> Optional[Dict[str, Any]]:
    want = re.sub(r"\D", "", value) if re.search(r"\d{5,}", value) else accesscmd._norm(value)
    for c in person.get("contacts") or []:
        have = re.sub(r"\D", "", str(c.get("value") or "")) if want.isdigit() else accesscmd._norm(c.get("value"))
        if have == want or accesscmd._norm(c.get("raw_value")) == accesscmd._norm(value):
            return c
    return None


def contact_remove(store: Any, person_text: str, kind_text: str, value: str) -> str:
    from ..console import api as console_api
    person, err = _person(store, person_text)
    if err:
        return err
    contact = _find_contact(person, value)
    if contact is None:
        return wording.contact_absent(person.get("display") or "", value)
    console_api.remove_contact_endpoint(store, {"contact_id": contact["id"]})
    kind = contact.get("kind") or _kind_for(store, kind_text, value)
    return wording.contact_gone_line(person.get("display") or "", kind, value) + "\n" + wording.undo_line(
        f"add the {kind} {value} to {person.get('display')}")


def contact_move(store: Any, person_text: str, to_text: str, kind_text: str, value: str) -> str:
    from ..console import api as console_api
    src, err = _person(store, person_text)
    if err:
        return err
    dst, err = _person(store, to_text)
    if err:
        return err
    contact = _find_contact(src, value)
    if contact is None:
        done = _find_contact(dst, value)
        if done is not None:
            return wording.contact_line(dst.get("display") or "", done.get("kind") or "", value)
        return wording.contact_absent(src.get("display") or "", value)
    console_api.move_contact_endpoint(store, {"contact_id": contact["id"], "person_id": dst["id"]})
    return wording.contact_line(dst.get("display") or "", contact.get("kind") or "", value) + "\n" + wording.undo_line(
        f"move {value} from {dst.get('display')} to {src.get('display')}")


def person_remove(ctx: Any, person_text: str) -> str:
    person, err = _person(ctx.store, person_text)
    if err:
        return err
    if accesscmd._is_commander(ctx.store, person):
        return wording.commander_fixed(person.get("display") or person_text)
    n = len(person.get("contacts") or [])
    return open_hold(ctx, "person_remove", {"id": person["id"], "display": person.get("display") or ""},
                     ask=wording.hold_person_remove(person.get("display") or "", n))


def person_merge(ctx: Any, person_text: str, to_text: str) -> str:
    a, err = _person(ctx.store, person_text)
    if err:
        return err
    b, err = _person(ctx.store, to_text)
    if err:
        return err
    if a["id"] == b["id"]:
        return wording.people_merged_line(b.get("display") or "", a.get("display") or "")
    if accesscmd._is_commander(ctx.store, a):
        return wording.commander_fixed(a.get("display") or person_text)
    return open_hold(ctx, "people_merge", {"keep": b["id"], "merge": a["id"], "keep_name": b.get("display") or "",
                                           "merge_name": a.get("display") or ""},
                     ask=wording.hold_people_merge(a.get("display") or "", b.get("display") or ""))


async def people_sync(ctx: Any) -> str:
    import asyncio
    from ..console import api as console_api
    client = getattr(ctx, "directory_client", None)
    if client is None:
        return wording.change_refused("I can't reach Inkbox's contact list from here")
    before = len(ctx.store.list_people())
    try:
        result = await asyncio.to_thread(console_api.people_sync, ctx.store, client)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[admin] people sync failed: %s", exc)
        return wording.change_refused("Inkbox did not answer")
    after = len(ctx.store.list_people())
    return wording.people_synced_line(after, after - before, result)


# ---------------------------------------------------------------------------
# groups, requests, schedules by name, health
# ---------------------------------------------------------------------------

def groups_list(store: Any) -> str:
    groups = store.known_group_chats()
    if not groups:
        return "I am not in any group chats yet."
    return "\n".join(f"{g['title']}: {g.get('response_mode') or g.get('mode') or 'mention'}"
                     for g in groups[:15]) + (f"\nand {len(groups) - 15} more" if len(groups) > 15 else "")


def requests_list(store: Any, which: str = "pending") -> str:
    which = (which or "pending").lower()
    if which in ("pending", "waiting"):
        rows = store.pending()
        verb = "Waiting on you"
    else:
        rows = store.recent_requests(state="failed")[:8]
        verb = "Failed"
    if not rows:
        return "Nothing is waiting on you." if which != "failed" else "Nothing has failed lately."
    lines = [f"#{r.id}: {' '.join(str(r.summary or '').split())[:90]}" for r in rows[:8]]
    tail = ('Reply "#N yes" or "#N no" to answer one.' if verb == "Waiting on you"
            else 'Say "retry #N" to try one again.')
    return f"{verb}:\n" + "\n".join(lines) + "\n" + tail


async def request_retry(ctx: Any, request_id: str) -> str:
    from ..console import api as console_api
    if getattr(ctx, "manager", None) is None:
        return wording.change_refused("I can't reach the request queue from here")
    try:
        res = await console_api.retry_request_endpoint(ctx.manager, {"id": int(request_id)})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    return res.get("message") or "Retrying."


async def schedule_do(ctx: Any, cmd: Dict[str, Any]) -> str:
    from . import schedcmd
    session = getattr(ctx, "session", None)
    schedule = _one_schedule(ctx.store, cmd.get("target") or "")
    if schedule is None:
        return "I don't have a schedule like that. Say schedules to see them."
    if cmd["sact"] == "stop":
        return open_hold(ctx, "schedule_stop", {"id": schedule.id},
                         ask=wording.hold_schedule_stop(schedule.title))
    if session is None or ctx.manager is None:
        return wording.change_refused("I can't reach the schedules from here")
    return await schedcmd.handle(ctx.manager, session, SimpleNamespace(
        action=cmd["sact"], target=cmd.get("target"), changes={}))


def health(ctx: Any) -> str:
    from ..console import api as console_api
    gateway = getattr(ctx.manager, "gateway", None) if getattr(ctx, "manager", None) else None
    data = console_api.health(gateway or SimpleNamespace(), ctx.store)
    return wording.health_line(data, pending=len(ctx.store.pending()))


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
    sync_actions = {
        "role_rename": lambda: role_rename(store, cmd.get("role") or "", cmd.get("name") or ""),
        "role_delete": lambda: role_delete(ctx, cmd.get("role") or ""),
        "role_add": lambda: role_scopes(ctx, cmd.get("role") or "", cmd.get("abilities") or "", add=True),
        "role_remove": lambda: role_scopes(ctx, cmd.get("role") or "", cmd.get("abilities") or "", add=False),
        "role_level": lambda: role_level(store, cmd.get("role") or "", cmd.get("system") or "", cmd.get("level") or ""),
        "role_send": lambda: role_send(store, cmd.get("role") or "", cmd.get("system") or "", True),
        "role_unsend": lambda: role_send(store, cmd.get("role") or "", cmd.get("system") or "", False),
        "people_list": lambda: people_list(store),
        "person_create": lambda: person_create(store, cmd.get("name") or ""),
        "person_rename": lambda: person_rename(store, cmd.get("person") or "", cmd.get("name") or ""),
        "person_note": lambda: person_note(store, cmd.get("person") or "", cmd.get("note") or ""),
        "person_remove": lambda: person_remove(ctx, cmd.get("person") or ""),
        "person_merge": lambda: person_merge(ctx, cmd.get("person") or "", cmd.get("to") or ""),
        "contact_add": lambda: contact_add(store, cmd.get("person") or "", cmd.get("kind") or "", cmd.get("value") or ""),
        "contact_remove": lambda: contact_remove(store, cmd.get("person") or "", cmd.get("kind") or "", cmd.get("value") or ""),
        "contact_move": lambda: contact_move(store, cmd.get("person") or "", cmd.get("to") or "", cmd.get("kind") or "", cmd.get("value") or ""),
        "groups_list": lambda: groups_list(store),
        "requests_list": lambda: requests_list(store, cmd.get("which") or "pending"),
        "health": lambda: health(ctx),
    }
    if action in sync_actions:
        return sync_actions[action]()
    if action == "people_sync":
        return await people_sync(ctx)
    if action == "request_retry":
        return await request_retry(ctx, cmd.get("id") or "0")
    if action == "schedule_do":
        return await schedule_do(ctx, cmd)
    if action in ("schedules_pause_all", "schedules_resume_all"):
        return schedules_global(ctx, action == "schedules_pause_all")
    if action == "role_create":
        return role_create(store, cmd.get("name") or "")
    if action == "roles_list":
        return roles_list(store)
    if action == "role_show":
        return role_show(store, cmd.get("role") or "")
    return wording.ABILITY_UNKNOWN
