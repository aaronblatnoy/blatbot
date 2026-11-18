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
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from . import accesscmd, chatreply, settings as gate_settings, wording
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
    # The console's read-only pages, by name or in his own words (console parity, 2026-11-18).
    ("home_show", r"^(?:overview|dashboard|home|how are things|what(?:'s| is) going on|give me the rundown)$"),
    ("inbox_show", r"^(?:show (?:me )?(?:my |the |your )?)?(?:event )?inbox$"),
    ("inbox_show", r"^what(?:'s| is| came) (?:in (?:the |your |my )?inbox|in lately|in today)$"),
    ("followups_show", r"^(?:show (?:me )?(?:my |the |your )?)?(?:follow[- ]?ups?|timers)$"),
    ("followups_show", r"^what are you waiting (?:on|for)$"),
    ("memory_show", r"^(?:show (?:me )?(?:your |the |my )?)?memory$"),
    ("memory_show", r"^(?:what do you remember|what(?:'s| is) in your memory)$"),
    ("knowledge_show", r"^who can (?:see|read) (?:my |your |the )?(?:notes|memory|what)$"),
    ("pages_show", r"^(?:show (?:me )?(?:your |the |my )?)?(?:published )?pages$"),
    ("pages_show", r"^what pages have you (?:published|made)$"),
    ("tasks_show", r"^(?:(?:show|list)(?: me)?(?: my| your| the)?(?: recent)? )?tasks$"),
    ("schedules_show", r"^(?:show|list)(?: me)?(?: all)?(?: of)?(?: my| your| the)? schedules$"),
    ("schedule_runs", r"^when did (?:the )?(?P<target>.+?)(?: schedule| job)? (?:last )?run$"),
    ("schedule_runs", r"^(?:show )?(?:the )?(?:runs|run history) (?:of|for) (?:the )?(?P<target>.+?)(?: schedule| job)?$"),
    ("request_dismiss", r"^(?:dismiss|clear) (?:request )?#?(?P<id>\d+)$"),
    ("systems_show", r"^(?:show (?:me )?(?:your |the )?)?(?:scopes|systems|tools|connections)$"),
    ("systems_show", r"^what (?:systems|tools|apps) (?:can you use|do you have|are you connected to)$"),
    # Console parity, second pass (2026-11-18): the rest of the routes by text.
    ("task_show", r"^(?:show |open |pull up )?(?:me )?(?:the )?(?:task|t) ?#?(?P<id>\d+)$"),
    ("roles_summary", r"^(?:who can do what|roles summary|summary of (?:the |my |your )?roles|what can each role do)$"),
    ("roles_who_can", r"^which roles? (?:can|may|is allowed to|are allowed to) (?P<q>.+)$"),
    ("followups_show", r"^(?:what |which )?(?:follow[- ]?ups?|timers) (?:are )?(?:due|coming up|pending)(?: today| soon)?$"),
    ("followup_cancel", r"^(?:cancel|stop|drop|delete) (?:the )?(?:follow[- ]?up|timer) #?(?P<id>\d+)$"),
    ("followup_snooze", r"^(?:snooze|push back|delay|postpone) (?:the )?(?:follow[- ]?up|timer) #?(?P<id>\d+)"
                        r"(?: (?:by|for) (?P<amount>\d+|an?|one) (?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?))?$"),
    ("schedule_report", r"^(?P<how>report every (?:time|run)|only report changes) (?:for|on) (?:the )?(?P<target>.+?)(?: schedule| job)?$"),
    ("role_notes", r"^set (?:the )?(?P<role>.+?) role'?s? notes? (?:access )?to (?P<level>none|nothing|all|everything)$"),
    ("role_memory", r"^set (?:the )?(?P<role>.+?) role'?s? memory (?:access )?to (?P<level>own|their own|own chats|groups|group chats|all|everything)$"),
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
    ("contact_update", r"^(?:change|update|replace) (?P<person>.+?)'s (?:(?P<kind>email|e-mail|phone|number|telegram|imessage|sms|handle)(?: address| number)? )?(?:from )?"
     + _VALUE + r" (?:to|with) " + _VALUE.replace("?P<value>", "?P<new>") + r"$"),
    ("person_note", r"^(?:set|change|update) (?P<person>.+?)'s note to (?P<note>.+)$"),
    ("person_create", r"^add (?P<name>[A-Za-z][A-Za-z'.\- ]{1,40}?) to (?:the )?(?:people|people list|contacts)$"),
    ("person_create", r"^(?:add|create|make) (?:a |an )?(?:new )?person (?:called |named )?(?P<name>[A-Za-z][A-Za-z'.\- ]{1,40})$"),
    ("schedule_rename", r"^rename (?:the )?(?P<target>.+?) (?:schedule|job) to (?P<name>.+)$"),
    ("person_rename", r"^rename (?:the )?(?:person )?(?P<person>.+?) to (?P<name>.+)$"),
    ("groups_list", r"^(?:list|show)(?: me)?(?: all)?(?: of)?(?: the| my| your)? group ?chats$"),
    ("groups_list", r"^what (?:groups|group chats|rooms) are you in\??$"),
    ("requests_list", r"^(?:show|list)(?: me)?(?: all)?(?: the| my)? (?P<which>pending|failed|waiting) requests$"),
    ("requests_list", r"^what(?:'s| is| are) (?:pending|waiting)(?: on me| for me| for my approval)?\??$"),
    ("request_retry", r"^(?:retry|try again on|try) (?:request )?#?(?P<id>\d+)(?: again)?$"),
    ("outbox_retry", r"^retry outbox #?(?P<id>\d+)$"),
    ("outbox_list", r"^outbox$"),
    ("health", r"^(?:health|health check|how are you doing|how(?:'s| is) blatbot(?: doing)?|are you ok(?:ay)?|are you healthy)\??$"),
    ("schedule_do", r"^(?P<sact>pause|resume|stop|repair|fix) (?:the )?(?P<target>.+?)(?: schedule| job)?$"),
    ("schedule_do", r"^(?P<sact>run) (?:the )?(?P<target>.+?)(?: schedule| job)? now$"),
    ("roles_list", r"^(?:list|show|what are)(?: me)?(?: all)?(?: the| my| your)? roles\??$"),
    ("roles_list", r"^what roles (?:are there|do (?:we|you|i) have|exist)\??$"),
    ("role_show", r"^what (?:can|does|do) (?:the |a )?(?P<role>.+?) role (?:do|have|allow)\??$"),
    ("role_show", r"^(?:show|what is|what's|describe) (?:the )?(?P<role>.+?) role\??$"),
    ("person_show", r"^(?:show|show me|look up|pull up|who is|who's) (?P<person>[A-Za-z][A-Za-z'.\- ]{1,40})$"),
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
                      "role_remove", "role_add", "role_notes", "role_memory"):
            if accesscmd.resolve_role(store, g.get("role") or "") is None:
                continue
        if action in ("role_add", "role_remove"):
            if not _abilities_for(g.get("abilities") or "", strict=True):
                continue
        if action in ("role_level", "role_send", "role_unsend") and _resolve_system(g.get("system") or "") is None:
            continue
        if action == "person_rename" and (g.get("person") or "").lower().endswith((" role", " schedule", " job")):
            continue
        if action in ("person_merge", "person_remove", "person_rename", "person_note", "contact_add", "contact_update",
                      "contact_remove", "contact_move"):
            if _norm_person_word(g.get("person")) in accesscmd._PRONOUNS or not g.get("person"):
                continue
        if action in ("schedule_runs", "schedule_rename", "schedule_report") and _one_schedule(
                store, g.get("target") or "") is None:
            continue
        if action == "person_show":
            if _norm_person_word(g.get("person")) in accesscmd._PRONOUNS or not accesscmd.find_people(
                    store, g.get("person") or ""):
                continue
        if action == "schedule_do":
            sact = (g.get("sact") or "").lower()
            found = _one_schedule(store, g.get("target") or "")
            if found is None:
                continue
            return {"action": "schedule_do", "sact": {"run": "run_now", "fix": "repair"}.get(sact, sact), "target": g.get("target")}
        if action == "role_create":
            name = (g.get("name") or "").strip()
            if not name or name.lower() in ("new", "a", "an", "the") or " role" in name.lower():
                continue
        return {"action": action, **g}
    return None


# Phrases anyone might say in ordinary conversation ("help", "how are you doing", "who
# is Sean", "what do you remember"). From someone other than Aaron these go on to the
# usual flow, where that person's own role decides what any run may touch. Every other
# admin phrase from a non-owner gets one plain refusal (refuse_for_non_owner).
_CONVERSATIONAL = {"guide", "health", "home_show", "person_show", "memory_show", "inbox_show",
                   "followups_show", "tasks_show", "pages_show"}
_CONVERSATIONAL_ACCESS = {"show_person"}


def refuse_for_non_owner(text: str, store: Any) -> bool:
    """True when a non-owner's message is one of the owner-only admin phrases, so the
    caller answers with a refusal instead of routing it."""
    try:
        cmd = match(text, store)
        if cmd is not None:
            return cmd["action"] not in _CONVERSATIONAL
        access = accesscmd.match(text, store)
        return access is not None and access.get("action") not in _CONVERSATIONAL_ACCESS
    except Exception:  # noqa: BLE001 - a matcher error never blocks an ordinary message
        logger.exception("[admin] non-owner phrase check failed")
        return False


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
        "who can do what",
        "which roles can read the TAMID calendar",
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
        f"change {name}'s email from sean@example.com to sean@corgi.nyc",
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
        f"pause the {title} schedule", f"run the {title} schedule now", "pause every schedule",
        f"when did the {title} schedule last run", f"rename the {title} schedule to morning digest",
        f"report every time for the {title} schedule", f"fix the {title} schedule"])


def guide_requests(store: Any) -> str:
    n = len(store.pending())
    state = [f"{n} waiting on you" if n else "Nothing is waiting on you."]
    return wording.guide("requests", state, ["show pending requests", "#417 yes", "#417 no",
                                              "#417 edit: make it shorter", "show failed requests", "retry #417"])


# "What can I do from here": one line per area, one phrase each.
_AREAS_BY_TEXT = [
    ("Status", "what's going on"),
    ("Requests", "what's pending"),
    ("People", "show people"),
    ("Roles and access", "who has access to my calendar"),
    ("Schedules", "show schedules"),
    ("Group chats", "show group chats"),
    ("Settings", "turn off group chatter"),
    ("Inbox, outbox and timers", "what's in the inbox"),
    ("Tasks", "task 12"),
    ("Follow-up timers", "snooze timer #3 by 2 hours"),
    ("Memory and pages", "what do you remember"),
    ("Health", "are you ok"),
]


def guide_everything(store: Any = None) -> str:
    rows = list(_AREAS_BY_TEXT)
    if store is not None:
        name = next((p.get("display") for p in store.list_people()
                     if p.get("display") and not accesscmd._is_commander(store, p)), None)
        if name:
            rows[2] = ("People", f"show {name.split()[0] if len(accesscmd.find_people(store, name.split()[0])) == 1 else name}")
        live = [s for s in store.schedules() if s.state == "active"]
        if live:
            rows[4] = ("Schedules", f"pause the {live[0].title} schedule")
    return chatreply.scrub("Everything the console shows, you can see or change here. One example each:\n"
                           + "\n".join(f"{area}: {phrase}" for area, phrase in rows)
                           + "\nSay an area's name for more.")


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
        return guide_everything(store)
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
        if kind == "role_memory":
            name = pl["role"]
            old = set((_role_row(store, name) or {}).get("scopes") or [])
            was = "all" if "memory:all" in old else "groups" if "memory:groups" in old else "own"
            console_api.roles_set_memory(store, {"role": name, "level": pl["level"]})
            return memory_line(name, pl["level"], was=was if was != pl["level"] else None)
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
    return chatreply.listing(f"{chatreply.count(len(people), 'person', 'people')} in People:",
                             [p["display"] for p in people], rest="on the console's People page")


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
    return chatreply.listing(f"I am in {chatreply.count(len(groups), 'group chat')}:",
                             [f"{g['title']}: {g.get('response_mode') or g.get('mode') or 'mention'}" for g in groups],
                             rest="on the console's Group chats page")


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
    lines = [f"#{r.id}: {_one(r.summary, 90)}" for r in rows]
    tail = ('Reply "#N yes" or "#N no" to answer one.' if verb == "Waiting on you"
            else 'Say "retry #N" to try one again, or "dismiss #N" to clear it.')
    return chatreply.listing(f"{verb}:", lines, rest="on the console's Requests page", cap=7) + "\n" + tail


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


def outbox_list(store: Any) -> str:
    from ..console import api as console_api
    cutoff = time.time() - 24 * 3600
    rows = [r for r in console_api.outbox_view(store, 200)["rows"]
            if r["status"] in ("queued", "held", "failed") and float(r.get("created_at") or 0) >= cutoff]
    if not rows:
        return "Nothing is queued, held or failed in the last 24 hours."
    words = {"queued": "queued", "held": "held for your yes", "failed": "failed"}
    to = [r["to"] if not re.match(r"^(telegram|tg|imessage|sms|voice|whatsapp):", str(r["to"])) else "someone"
          for r in rows]
    items = [f"#{r['id']} to {t}, {words.get(r['status'], r['status'])}: {_one(r['preview'], 60)}"
             for r, t in zip(rows, to)]
    return chatreply.listing(f"{chatreply.count(len(rows), 'message')} not sent yet in the last 24 hours:", items,
                             rest="on the console's Outbox page")


def outbox_retry(store: Any, row_id: str) -> str:
    return (f"Outbox #{row_id} is queued to try again." if store.retry_outbox(int(row_id))
            else f"Outbox #{row_id} was not a failed message.")


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
        "outbox_list": lambda: outbox_list(store),
        "outbox_retry": lambda: outbox_retry(store, cmd.get("id") or "0"),
        "contact_update": lambda: contact_update(store, cmd),
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
    if action in _PAGE_ACTIONS:
        return await _PAGE_ACTIONS[action](ctx, cmd)
    return wording.ABILITY_UNKNOWN


# ---------------------------------------------------------------------------
# the rest of the console, by text (2026-11-18): every reply comes from the same
# console/api.py function the page reads, shaped by chatreply (lead line, short list)
# ---------------------------------------------------------------------------

def _gateway(ctx: Any) -> Any:
    manager = getattr(ctx, "manager", None)
    return getattr(manager, "gateway", None) or SimpleNamespace()


async def home_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    data = console_api.home(_gateway(ctx), ctx.store)
    needs = int(data["needs_you"]["count"])
    running = data["right_now"]["running"]
    lead = (f"{chatreply.count(needs, 'thing')} waiting on you, {int(data['right_now']['running_count'])} running."
            if needs or running else "Nothing is waiting on you and nothing is running.")
    items = [f"Running: {_one(r.get('summary'))}" for r in running[:3]]
    oldest = data["needs_you"].get("oldest")
    if oldest:
        items.insert(0, f"Waiting longest: {_one(oldest.get('summary'))} (#{oldest.get('id')})")
    nxt = (data.get("coming_up") or {}).get("next") or []
    if (data.get("coming_up") or {}).get("schedules_paused"):
        items.append("Every schedule is paused.")
    elif nxt:
        items.append(f"Next schedule: {nxt[0]['title']}, {chatreply.when(nxt[0].get('next_run'))}")
    ppl = data.get("people") or {}
    items.append(f"People: {ppl.get('total', 0)}; roles: {len(ppl.get('roles') or [])}")
    if not data["right_now"].get("tunnel_connected"):
        items.append("My public link is not connected.")
    return chatreply.listing(lead, items)


def _one(text: Any, limit: int = 80) -> str:
    out = " ".join(str(text or "").split())
    from .ownertext import strip_group_tag
    out = strip_group_tag(out)
    return out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


async def inbox_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    data = console_api.inbox_view(ctx.store, 200)
    c = data.get("counts") or {}
    waiting, running, failed = int(c.get("pending", 0)), int(c.get("running", 0)), int(c.get("failed", 0))
    lead = f"Inbox: {waiting} waiting, {running} being handled, {failed} failed."
    items = []
    for lane in data.get("lanes") or []:
        ev = (lane.get("events") or [{}])[0]
        state = {"pending": "waiting", "running": "being handled", "done": "done", "failed": "failed"}.get(
            ev.get("status"), ev.get("status") or "")
        items.append(f"{lane.get('label') or 'Someone'}: {ev.get('kind_label') or 'Message'}, {state}, "
                     f"{chatreply.when(ev.get('created_at'))}")
    return chatreply.listing(lead, items, total=len(data.get("lanes") or []),
                             rest="on the console's Inbox page", empty="The inbox is empty.")


async def followups_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    data = console_api.followups_view(ctx.store, 100)
    if not data.get("available", True):
        return "Follow-up timers are not set up on this server."
    pending = [r for r in data.get("rows") or [] if r.get("status") == "pending"]
    if not pending:
        return "No follow-up timers are waiting."
    items = [f"#{r['id']}: {_one(r.get('note') or 'Check back', 60)}, for {r.get('for') or 'you'}, "
             f"{chatreply.when(r.get('due_at'))}" for r in pending]
    return (chatreply.listing(f"{chatreply.count(len(pending), 'follow-up timer')} waiting:", items,
                              rest="on the console's Timers page")
            + '\nSay "snooze timer #N by 1 hour" or "cancel timer #N".')


async def memory_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    data = console_api.memory_view(_gateway(ctx), ctx.store)
    notes = data.get("notes") or {}
    graph = data.get("graph") or {}
    msgs = data.get("messages") or {}
    tasks = data.get("tasks") or {}
    items = []
    instr = data.get("instructions") or {}
    if instr.get("available"):
        items.append(f"Standing instructions: {chatreply.count(int(instr.get('paragraph_count') or 0), 'paragraph')}")
    if notes.get("available"):
        top = ", ".join(sec["name"] for sec in (notes.get("sections") or [])[:3])
        items.append(f"Notes I can read: {notes.get('total', 0)}" + (f", mostly {top}" if top else ""))
    total_msgs = msgs.get("total") if isinstance(msgs, dict) else None
    if total_msgs is not None:
        items.append(f"Stored messages: {total_msgs}")
    open_tasks = tasks.get("open") if isinstance(tasks, dict) else None
    if open_tasks is not None:
        items.append(f"Open tasks: {open_tasks}")
    items.append("Long-term memory: " + ("on" if graph.get("enabled") else "off")
                 + (f", {graph.get('pending_writes')} facts still being saved" if graph.get("pending_writes") else ""))
    return chatreply.listing("Here is what I keep between conversations.", items)


async def knowledge_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    rows = console_api.memory_access(ctx.store)
    items = [f"{r['part']}: {r['readers']}" for r in rows]
    return chatreply.listing("Who can see what I keep:", items)


async def pages_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    data = console_api.pages_view(200)
    pages = data.get("pages") or []
    items = [f"{p['title']}, {chatreply.when(p.get('created_at'))}" for p in pages]
    return chatreply.listing(f"{chatreply.count(len(pages), 'page')} published, kept {data.get('days')} days.",
                             items, rest="on the console's Pages page", empty="No pages are published right now.")


_TASK_STATE_WORDS = {"open": "open", "running": "running", "waiting_aaron": "waiting on you", "done": "done",
                     "failed": "failed", "closed": "closed"}


async def tasks_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    rows = console_api.tasks(ctx.store)
    live = [r for r in rows if r.get("state") in ("open", "running", "waiting_aaron")]
    items = [f"{_one(r.get('title'), 70)}: {_TASK_STATE_WORDS.get(r.get('state'), str(r.get('state') or '').replace('_', ' '))}"
             for r in (live or rows)]
    lead = (f"{chatreply.count(len(live), 'task')} open." if live else "No open tasks. The most recent:")
    return chatreply.listing(lead, items, rest="on the console's Tasks page", empty="No tasks yet.")


async def schedules_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    manager = getattr(ctx, "manager", None)
    if manager is None:
        return "I can't reach the schedules from here."
    data = console_api.schedule_list(ctx.store, manager.schedules)
    rows = [r for r in data["schedules"] if r["state"] != "ended"]
    items = []
    for r in rows:
        nxt = chatreply.when(r.get("next_run"), r.get("timezone") or "")
        items.append(f"{r['title']}: {r.get('cadence_text') or ''}, {str(r.get('state_label') or r['state']).lower()}"
                     + (f", next {nxt}" if nxt else ""))
    lead = ("Every schedule is paused. " if data.get("paused_all") else "") + f"{chatreply.count(len(rows), 'schedule')}:"
    return chatreply.listing(lead, items, rest="on the console's Schedules page", empty="No schedules yet.")


async def schedule_runs(ctx: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    schedule = _one_schedule(ctx.store, cmd.get("target") or "")
    if schedule is None:
        return "I don't have a schedule like that. Say show schedules to see them."
    data = console_api.schedule_runs_view(ctx.store, schedule.id) or {"runs": []}
    runs = data["runs"]
    if not runs:
        return f"{schedule.title} has not run yet."
    items = []
    for r in runs:
        line = f"{chatreply.when(r.get('started_at'), schedule.timezone)}: {r.get('outcome_label') or 'Ran'}"
        if r.get("duration_seconds") is not None:
            line += f", took {chatreply.span(r['duration_seconds'])}"
        if r.get("failure_reason"):
            line += f", {_one(r['failure_reason'], 60)}"
        items.append(line)
    return chatreply.listing(f"{schedule.title} has run {chatreply.count(len(runs), 'time')}. Latest first:",
                             items, rest="on the console's Schedules page")


async def request_dismiss(ctx: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    manager = getattr(ctx, "manager", None)
    if manager is None:
        return wording.change_refused("I can't reach the request queue from here")
    rid = int(cmd.get("id") or 0)
    req = ctx.store.get_request(rid)
    if req is not None and req.state == "failed" and (req.status or {}).get("dismissed"):
        return f"#{rid} is cleared from the failed list."
    try:
        await console_api.dismiss_request_endpoint(manager, {"id": rid})
    except console_api.ValidationError:
        if req is None:
            return f"There is no request #{rid}."
        return f"#{rid} is {req.state}, not failed, so there is nothing to clear."
    return f"#{rid} is cleared from the failed list.\n" + wording.undo_line(f"retry #{rid}")


async def systems_show(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    rows = console_api.scopes_breakdown(ctx.store)
    items = []
    for r in rows:
        name = console_api._system_label(r.get("system") or "")
        n = len(r.get("scopes") or [])
        if not n:
            continue
        items.append(f"{name}" + (f": {chatreply.count(int(n), 'ability', 'abilities')}" if n else ""))
    return chatreply.listing(f"I can use {chatreply.count(len(items), 'system')}:", items, total=len(items),
                             rest="on the console's Scopes page", empty="I have no systems connected.")


def _level_word(level: str, table: Dict[str, str]) -> str:
    return table.get(accesscmd._norm(level), accesscmd._norm(level))


async def role_notes(ctx: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    name, err = _role_or_error(ctx.store, cmd.get("role") or "")
    if err:
        return err
    level = _level_word(cmd.get("level") or "", {"nothing": "none", "everything": "all"})
    old = (_role_row(ctx.store, name) or {}).get("scopes") or []
    try:
        console_api.roles_set_notes(ctx.store, {"role": name, "level": level})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    shown = "every note folder" if level == "all" else "no notes"
    undo = f"set the {name} role's notes to {'none' if level == 'all' else 'all'}"
    head = f"The {name} role can read {shown}."
    if set(old) == set((_role_row(ctx.store, name) or {}).get("scopes") or []):
        return head
    return head + "\n" + wording.undo_line(undo)


async def role_memory(ctx: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    name, err = _role_or_error(ctx.store, cmd.get("role") or "")
    if err:
        return err
    level = _level_word(cmd.get("level") or "", {"their own": "own", "own chats": "own", "group chats": "groups",
                                                  "everything": "all"})
    old = set((_role_row(ctx.store, name) or {}).get("scopes") or [])
    was = "all" if "memory:all" in old else "groups" if "memory:groups" in old else "own"
    if level in ("all", "groups") and was != level:
        words = {"all": "remember every conversation", "groups": "remember every group chat"}[level]
        return open_hold(ctx, "role_memory", {"role": name, "level": level},
                         ask=wording.hold_role_grant(name, words, _members(ctx.store, name)))
    if was != level:
        try:
            console_api.roles_set_memory(ctx.store, {"role": name, "level": level})
        except console_api.ValidationError as exc:
            return wording.change_refused(str(exc))
    return memory_line(name, level, was=was if was != level else None)


def memory_line(name: str, level: str, *, was: Optional[str] = None) -> str:
    reach = {"own": "only their own conversations", "groups": "their own conversations plus every group chat",
             "all": "every conversation"}[level]
    line = f"The {name} role remembers {reach}."
    if was:
        line += "\n" + wording.undo_line(f"set the {name} role's memory to {was}")
    return line


async def person_show(ctx: Any, cmd: Dict[str, Any]) -> str:
    person, err = accesscmd._one_person(ctx.store, cmd.get("person") or "")
    if err:
        return err
    line = accesscmd.show_person(ctx.store, person.get("display") or cmd.get("person") or "")
    contacts = [c for c in person.get("contacts") or [] if c.get("value")]
    kinds = {"email": "Email", "phone": "Phone", "telegram": "Telegram", "imessage": "iMessage", "sms": "Text",
             "whatsapp": "WhatsApp"}
    items = [f"{kinds.get(c.get('kind'), str(c.get('kind') or 'Contact').title())}: {c.get('value')}"
             for c in contacts if not re.match(r"^(telegram|tg|imessage|sms|voice):\d", str(c.get("value")))]
    if person.get("note"):
        items.append(f"Note: {_one(person['note'], 120)}")
    return chatreply.listing(line, items)


# ---------------------------------------------------------------------------
# console parity, second pass (2026-11-18): a task by number, the roles summary and
# "which roles can", a contact's value, a schedule's name and reporting, repair, and
# follow-up timers. Same console/api.py (and store) calls the console's routes make.
# ---------------------------------------------------------------------------

_EVENT_WORDS = {"inbound": "Message", "request": "Asked", "done": "Done", "note": "Note", "result": "Result",
                "failed": "Failed", "rejected": "Turned down", "outbound": "Sent"}


async def task_show(ctx: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    tid = int(cmd.get("id") or 0)
    row = console_api.task(ctx.store, tid)
    if not row:
        return f"There is no task {tid}. Say tasks to see the open ones."
    state = _TASK_STATE_WORDS.get(row.get("state"), str(row.get("state") or "").replace("_", " "))
    lead = f"Task {tid}: {_one(row.get('title'), 90)}. It is {state}."
    items = []
    if row.get("summary"):
        items.append(f"Where it stands: {_one(row['summary'], 160)}")
    for ev in (row.get("events") or [])[-5:]:
        items.append(f"{_EVENT_WORDS.get(ev.get('kind'), str(ev.get('kind') or 'Event').title())}, "
                     f"{chatreply.when(ev.get('created_at'))}: {_one(ev.get('text'), 90)}")
    tail = (f"\nTo close it, say: T{tid} done" if row.get("state") in ("open", "running", "waiting_aaron")
            else f"\nTo reopen it, say: T{tid} reopen")
    return chatreply.listing(lead, items, rest="on the console's Tasks page") + tail


async def roles_summary(ctx: Any, _cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    data = console_api.roles_summary(ctx.store)
    items = []
    for r in data.get("roles") or []:
        if r.get("locked"):
            items.append(f"{r['name']}: full control; you alone")
            continue
        who = r.get("member_count") or 0
        can = " ".join(str(r.get("can_text") or "nothing without asking").split()).rstrip(".")
        items.append(f"{r['name']}, {chatreply.count(int(who), 'person', 'people')}: {_one(can, 140)}")
    return chatreply.listing("Who can do what:", items, rest="on the console's Roles page", empty="No roles yet.")


async def roles_who_can(ctx: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    q = " ".join(str(cmd.get("q") or "").split())
    data = console_api.roles_who_can(ctx.store, q)
    items = []
    for m in data.get("matches") or []:
        holders = []
        for r in m.get("roles") or []:
            names = [x.get("name") for x in r.get("members") or [] if x.get("name")]
            holders.append(r["name"] + (f" ({', '.join(names[:4])})" if names and r["name"] != "You" else ""))
        items.append(f"{_one(m.get('phrase'), 70)}: {', '.join(holders)}")
    return chatreply.listing(f"Roles that can {q}:", items, rest="on the console's Roles page",
                             empty=f"No role can {q}. Only you can.")


def contact_update(store: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    person, err = _person(store, cmd.get("person") or "")
    if err:
        return err
    old, new = cmd.get("value") or "", cmd.get("new") or ""
    contact = _find_contact(person, old)
    who = person.get("display") or ""
    if contact is None:
        if _find_contact(person, new) is not None:
            return wording.contact_line(who, _kind_for(store, cmd.get("kind") or "", new), new)
        return wording.contact_absent(who, old)
    try:
        res = console_api.update_contact_endpoint(store, {"contact_id": contact["id"], "value": new})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    if res.get("conflict"):
        return wording.contact_conflict(new, res.get("existing_person_display") or "someone else")
    kind = contact.get("kind") or _kind_for(store, cmd.get("kind") or "", new)
    shown = (res.get("contact") or {}).get("value") or new
    return (wording.contact_line(who, kind, shown)[:-1] + f" (was {contact.get('value') or old}).\n"
            + wording.undo_line(f"change {who}'s {kind} from {new} to {old}"))


async def schedule_rename(ctx: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    schedule = _one_schedule(ctx.store, cmd.get("target") or "")
    if schedule is None:
        return "I don't have a schedule like that. Say show schedules to see them."
    new = " ".join(str(cmd.get("name") or "").split()).strip("\"'")
    if not new:
        return "Say the new name, like: rename the daily check schedule to morning check"
    if new == schedule.title:
        return f"The schedule is called {new}."
    if getattr(ctx, "manager", None) is None:
        return wording.change_refused("I can't reach the schedules from here")
    try:
        res = await console_api.edit_schedule(ctx.manager, {"id": schedule.id, "title": new})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    if res.get("needs_approval"):
        return res.get("message") or "That change needs your yes first."
    return (f"The schedule is called {new} (was {schedule.title}).\n"
            + wording.undo_line(f"rename the {new} schedule to {schedule.title}"))


async def schedule_report(ctx: Any, cmd: Dict[str, Any]) -> str:
    from ..console import api as console_api
    schedule = _one_schedule(ctx.store, cmd.get("target") or "")
    if schedule is None:
        return "I don't have a schedule like that. Say show schedules to see them."
    every = str(cmd.get("how") or "").lower().startswith("report every")
    was_every = bool(getattr(schedule, "report_every_time", False))
    words = {True: "reports after every run", False: "reports only when something changed"}
    if every == was_every:
        return f"{schedule.title} {words[every]}."
    if getattr(ctx, "manager", None) is None:
        return wording.change_refused("I can't reach the schedules from here")
    try:
        res = await console_api.edit_schedule(ctx.manager, {"id": schedule.id, "report_every_time": every})
    except console_api.ValidationError as exc:
        return wording.change_refused(str(exc))
    if res.get("needs_approval"):
        return res.get("message") or "That change needs your yes first."
    undo = (f"only report changes for the {schedule.title} schedule" if every
            else f"report every time for the {schedule.title} schedule")
    return f"{schedule.title} {words[every]} (was: {words[was_every].split(' ', 1)[1]}).\n" + wording.undo_line(undo)


def _followup_row(store: Any, fid: int) -> Optional[Dict[str, Any]]:
    try:
        return store.get_followup(fid)
    except Exception:  # noqa: BLE001 - a store without the table
        return None


async def followup_cancel(ctx: Any, cmd: Dict[str, Any]) -> str:
    fid = int(cmd.get("id") or 0)
    row = ctx.store.cancel_followup(fid)
    if row is None:
        old = _followup_row(ctx.store, fid)
        if old is None:
            return f"There is no follow-up timer #{fid}. Say timers to see them."
        return f"Follow-up timer #{fid} is {old.get('status')}, so there is nothing to cancel."
    ctx.store.task_event(row["task_id"], "note", f"Follow-up #{row['id']} cancelled: {row['note']}",
                         chat_id=getattr(ctx, "chat_id", "") or "", state=None)
    return (f"Follow-up timer #{fid} is cancelled; it will not run. It was set to {_one(row.get('note'), 80)}, "
            f"{chatreply.when(row.get('due_at'))}.")


_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


async def followup_snooze(ctx: Any, cmd: Dict[str, Any]) -> str:
    fid = int(cmd.get("id") or 0)
    amount = str(cmd.get("amount") or "1").lower()
    n = 1 if amount in ("a", "an", "one") else int(amount)
    unit = str(cmd.get("unit") or "hour").lower()
    seconds = n * _UNIT_SECONDS.get(unit[0], 3600)
    if seconds < 5 * 60 or seconds > 30 * 86400:
        return "Nothing changed: snooze a timer by at least 5 minutes and at most 30 days."
    row = _followup_row(ctx.store, fid)
    if row is None:
        return f"There is no follow-up timer #{fid}. Say timers to see them."
    if row.get("status") != "pending":
        return f"Follow-up timer #{fid} is {row.get('status')}, so there is nothing to snooze."
    old_due = float(row["due_at"])
    new_due = max(old_due, time.time()) + seconds
    updated = ctx.store.snooze_followup(fid, new_due)
    if updated is None:
        return f"Follow-up timer #{fid} already ran or was cancelled, so there is nothing to snooze."
    ctx.store.task_event(row["task_id"], "note", f"Follow-up #{fid} moved to {chatreply.when(new_due)}: {row['note']}",
                         chat_id=getattr(ctx, "chat_id", "") or "", state=None)
    return (f"Follow-up timer #{fid} runs {chatreply.when(new_due)} (was {chatreply.when(old_due)}).\n"
            f"To cancel it instead, say: cancel timer #{fid}")


_PAGE_ACTIONS = {
    "home_show": home_show, "inbox_show": inbox_show, "followups_show": followups_show,
    "memory_show": memory_show, "knowledge_show": knowledge_show, "pages_show": pages_show,
    "tasks_show": tasks_show, "schedules_show": schedules_show, "schedule_runs": schedule_runs,
    "request_dismiss": request_dismiss, "systems_show": systems_show, "role_notes": role_notes,
    "role_memory": role_memory, "person_show": person_show,
    "task_show": task_show, "roles_summary": roles_summary, "roles_who_can": roles_who_can,
    "schedule_rename": schedule_rename, "schedule_report": schedule_report,
    "followup_cancel": followup_cancel, "followup_snooze": followup_snooze,
}
