"""Who may do what, and how freely Blatbot speaks in a room, changed and read back
from Aaron's own chat in one message, in any of his chats: a person's role, what a
person may do without asking him, who holds some access, and a group chat's
response mode.

Code, not a model. The common phrasings are matched here before the router or Jev
ever see the message (match()), so "make Sean a corgi teammate" can never be judged
a lookup or small talk; anything worded differently arrives as the router's
access_command, and both end in handle(). Every change goes through the same store
methods and console events the web console uses, and every reply is one line from
wording.py that states the resulting state, never the action: asking for a state
that already holds changes nothing and returns the same line without "(was ...)",
and a read-back returns that same line too.

Reversible configuration runs at once when Aaron asks for it. Deleting a person or
a role is not offered here at all; that stays on the console, behind its own
confirmation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import roommode, schedprop, wording
from .scopes import SCOPE_SYSTEM, SCOPES
from .store import role_names

logger = logging.getLogger(__name__)

COMMANDER_ROLE = "commander"
BASELINE_ROLE = "general"

# Words that point at a person but never name one in People.
_PRONOUNS = {"you", "u", "i", "me", "we", "us", "it", "they", "them", "he", "him", "she", "her",
             "blatbot", "yourself", "everyone", "anyone", "somebody", "someone", "people"}
# Words in "Corgi access" style phrases that say nothing about which system is meant.
_FILLER = {"access", "the", "to", "a", "an", "all", "of", "his", "her", "their", "its", "read", "see",
           "use", "and", "any", "everything", "permission", "permissions", "tools", "stuff", "things",
           "rights", "with", "for", "on", "in", "from", "away", "anymore", "now", "please", "more"}


# ---------------------------------------------------------------------------
# resolving his words
# ---------------------------------------------------------------------------

def _norm(text: Any) -> str:
    return " ".join(str(text or "").replace("’", "'").split()).strip().lower()


def find_people(store: Any, text: str) -> List[Dict[str, Any]]:
    """Live people matching `text`: an exact display name or contact value wins
    outright; otherwise every person whose display name starts with those words or
    holds them as whole words ("Stanley" finds "Stanley Rabkin")."""
    want = _norm(text)
    if not want:
        return []
    people = store.list_people()
    exact = [p for p in people if _norm(p.get("display")) == want
             or any(_norm(c.get("value")) == want or _norm(c.get("raw_value")) == want
                    for c in (p.get("contacts") or []))]
    if exact:
        return exact
    pattern = re.compile(r"(?:^|\s)" + re.escape(want) + r"(?:\s|$)")
    return [p for p in people if pattern.search(_norm(p.get("display")))]


def resolve_role(store: Any, text: str) -> Optional[str]:
    """The role his words name: exact, a plural ("corgi teammates"), or the one role
    whose name contains them ("teammate" -> "corgi teammate"). None when nothing or
    more than one role fits."""
    want = re.sub(r"^(?:an?|the)\s+", "", _norm(text))
    want = re.sub(r"\s+role$", "", want)
    if not want:
        return None
    names = [r["name"] for r in store.roles()]
    for candidate in (want, want[:-1] if want.endswith("s") else ""):
        if candidate and candidate in names:
            return candidate
    hits = [n for n in names if want in n or (want.endswith("s") and want[:-1] in n)]
    return hits[0] if len(hits) == 1 else None


def resolve_abilities(text: str, *, strict: bool = False) -> List[str]:
    """Scopes his own words name: a scope's label or synonym ("read the TAMID
    calendar"), else every scope of the systems a distinctive word names ("Corgi
    access" -> every scope declared under a corgi system). Registry order. strict
    (match()) takes the system fallback only when he said "access", so "who has my
    calendar invite" is never read as a question about the TAMID calendar."""
    found = schedprop.scopes_named(text or "")
    if not found and (not strict or re.search(r"\baccess\b", _norm(text))):
        words = [w for w in re.findall(r"[a-z0-9]+", _norm(text)) if w not in _FILLER and len(w) >= 3]
        systems = {sys for sys in set(SCOPE_SYSTEM.values())
                   if any(w in sys.split("_") for w in words)}
        found = [name for name, sys in SCOPE_SYSTEM.items() if sys in systems]
    wanted = set(found)
    return [name for name in SCOPES if name in wanted]


def _effective(store: Any, person: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    roles_by_name = {r["name"]: r.get("scopes") or [] for r in store.roles()}
    held = role_names(person.get("role"))
    scopes = set(person.get("scopes") or [])
    for name in held:
        scopes |= set(roles_by_name.get(name, []))
    return held, sorted(scopes)


def _is_commander(store: Any, person: Dict[str, Any]) -> bool:
    return person.get("id") == store.commander_person_id() or COMMANDER_ROLE in role_names(person.get("role"))


def person_line(store: Any, person: Dict[str, Any], *, old_roles: Optional[List[str]] = None,
                old_scopes: Optional[List[str]] = None) -> str:
    """The one line that states what this person is and may do, now."""
    if _is_commander(store, person):
        return wording.commander_line(person.get("display") or "Aaron")
    roles, scopes = _effective(store, person)
    return wording.access_line(person.get("display") or "", roles, scopes,
                               old_roles=old_roles, old_scopes=old_scopes)


def _one_person(store: Any, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    matches = find_people(store, text)
    if not matches:
        return None, wording.person_not_found(text)
    if len(matches) > 1:
        return None, wording.person_ambiguous(text, [m.get("display") or "" for m in matches])
    return matches[0], None


def _publish(kind: str, payload: Dict[str, Any]) -> None:
    try:
        from ..console import events
        events.publish(kind, payload)
    except Exception:  # noqa: BLE001 -- the console being absent never blocks a change
        logger.debug("console event %s not published", kind)


# ---------------------------------------------------------------------------
# applying a command
# ---------------------------------------------------------------------------

def _write_person(store: Any, person: Dict[str, Any], *, roles: List[str], scopes: List[str]) -> Dict[str, Any]:
    before_roles = role_names(person.get("role"))
    before_scopes = sorted(set(person.get("scopes") or []))
    if roles == before_roles and sorted(set(scopes)) == before_scopes:
        return person
    row = store.update_person(person["id"], role=", ".join(roles), scopes=sorted(set(scopes)))
    _publish("people.changed", {"kind": "person", "id": person["id"], "action": "updated"})
    logger.info("[access] %s: roles %s -> %s, own scopes %s -> %s", person["id"], before_roles, roles,
                before_scopes, sorted(set(scopes)))
    return row


def change_roles(store: Any, person_text: str, role_text: str, how: str) -> str:
    """how: "add" (make X a R), "remove" (X is no longer a R), or "set" (X's role is R)."""
    person, err = _one_person(store, person_text)
    if err:
        return err
    if _is_commander(store, person):
        return wording.commander_fixed(person.get("display") or person_text)
    role = resolve_role(store, role_text)
    if role is None:
        return wording.role_unknown(role_text, [r["name"] for r in store.roles() if r["name"] != COMMANDER_ROLE])
    if role == COMMANDER_ROLE:
        return wording.COMMANDER_NOT_FROM_CHAT
    old_roles, old_scopes = _effective(store, person)
    if how == "set" or role == BASELINE_ROLE:
        roles = [role]
    elif how == "remove":
        roles = [r for r in old_roles if r != role] or [BASELINE_ROLE]
    else:
        roles = [r for r in old_roles if r != BASELINE_ROLE]
        if role not in roles:
            roles.append(role)
    row = _write_person(store, person, roles=roles, scopes=list(person.get("scopes") or []))
    return person_line(store, row, old_roles=old_roles, old_scopes=old_scopes)


def change_abilities(store: Any, person_text: str, abilities_text: str, grant: bool,
                     scopes: Optional[List[str]] = None) -> str:
    """Grant: the person may now use these without asking (held on the person, beside
    whatever their roles give). Revoke: taken off the person, and any role of theirs
    that gives one of them is taken off too, so the access is actually gone."""
    person, err = _one_person(store, person_text)
    if err:
        return err
    wanted = [s for s in (scopes or []) if s in SCOPES] or resolve_abilities(abilities_text)
    if not wanted:
        return wording.ABILITY_UNKNOWN
    if _is_commander(store, person):
        return wording.commander_fixed(person.get("display") or person_text)
    old_roles, old_scopes = _effective(store, person)
    own = set(person.get("scopes") or [])
    roles = list(old_roles)
    if grant:
        own |= {s for s in wanted if s not in old_scopes}
    else:
        own -= set(wanted)
        roles_by_name = {r["name"]: set(r.get("scopes") or []) for r in store.roles()}
        roles = [r for r in roles if not (roles_by_name.get(r, set()) & set(wanted))] or [BASELINE_ROLE]
    row = _write_person(store, person, roles=roles, scopes=sorted(own))
    return person_line(store, row, old_roles=old_roles, old_scopes=old_scopes)


def show_person(store: Any, person_text: str) -> str:
    person, err = _one_person(store, person_text)
    return err or person_line(store, person)


def who_has(store: Any, words: str) -> str:
    """Everyone who holds a role his words name, or any scope they name."""
    people = [p for p in store.list_people() if not _is_commander(store, p)]
    role = resolve_role(store, words)
    if role is not None and role != COMMANDER_ROLE:
        holders = [p.get("display") or "" for p in people if role in role_names(p.get("role"))]
        return wording.who_has_line(role, holders)
    wanted = resolve_abilities(words)
    if not wanted:
        return wording.ABILITY_UNKNOWN
    holders = [p.get("display") or "" for p in people if set(_effective(store, p)[1]) & set(wanted)]
    subject = schedprop.labels(wanted) if len(wanted) <= 2 else " ".join(_norm(words).split())
    return wording.who_has_line(subject, holders)


def _chat_target(store: Any, chat_text: str, here: Tuple[str, str]) -> Tuple[Optional[str], str]:
    words = _norm(chat_text)
    if words in ("", "here", "this chat", "this group", "this room", "this group chat", "this one"):
        return here
    match = store.find_group_chat_by_name(chat_text)
    if match is None:
        bare = re.sub(r"^the\s+", "", words)
        bare = re.sub(r"\s+(?:group chat|group|chat|room)$", "", bare)
        match = store.find_group_chat_by_name(bare) if bare and bare != words else None
    if match is None:
        return None, chat_text
    return match["chat_id"], match["title"]


def set_mode(store: Any, chat_text: str, mode: str, here: Tuple[str, str]) -> str:
    mode = roommode.normalize_mode("silent" if _norm(mode) == "quiet" else mode)
    if not mode:
        return wording.MODE_UNKNOWN
    chat_id, title = _chat_target(store, chat_text, here)
    if not chat_id:
        return wording.chat_not_found(title)
    stored = store.get_response_mode(chat_id)
    old = roommode.effective_mode(stored)
    if stored != mode:
        store.set_response_mode(chat_id, mode)
        _publish("groups.changed", {"chat_id": chat_id, "response_mode": mode})
    return wording.mode_line(title, mode, old_mode=old)


def show_mode(store: Any, chat_text: str, here: Tuple[str, str]) -> str:
    chat_id, title = _chat_target(store, chat_text, here)
    if not chat_id:
        return wording.chat_not_found(title)
    stored = store.get_response_mode(chat_id)
    return wording.mode_line(title, roommode.effective_mode(stored), default=not stored)


# ---------------------------------------------------------------------------
# Blatbot's own settings (the console's Settings page), in the same line shape
# ---------------------------------------------------------------------------

def _setting_rows() -> List[Dict[str, Any]]:
    from . import settings as gate_settings
    return gate_settings.describe()


def resolve_setting(text: str, *, strict: bool = False) -> Optional[Dict[str, Any]]:
    """A setting by its console name (GATE_CONFIRM_SENDS), its label, or the one
    setting whose label or name holds every distinctive word he used ("speaking
    bar", "confirm sends"). strict (match(), before any model has looked) needs at
    least two such words, so "what's on my schedule" is never read as the "Pause
    every schedule" setting."""
    want = _norm(text)
    if not want:
        return None
    rows = _setting_rows()
    for r in rows:
        if want in (r["name"].lower(), _norm(r.get("label"))):
            return r
    words = [w for w in re.findall(r"[a-z0-9]+", want) if w not in _FILLER and w not in ("your", "my", "setting")
             and len(w) >= 3]
    if not words or (strict and len(words) < 2):
        return None

    def holds(r: Dict[str, Any]) -> bool:
        hay = set(re.findall(r"[a-z0-9]+", _norm(r.get("label")))) | set(r["name"].lower().split("_"))
        return all(any(h.startswith(w) or w.startswith(h) and len(h) >= 4 for h in hay) for w in words)

    hits = [r for r in rows if holds(r)]
    return hits[0] if len(hits) == 1 else None


def setting_line_for(row: Dict[str, Any], *, old_value: Any = None, changed: bool = False) -> str:
    return wording.setting_line(row.get("label") or row["name"], row.get("value"), row.get("kind") or "",
                                old_value=old_value, changed=changed, restart=bool(row.get("restart")))


def show_settings() -> str:
    """Every setting, one line each, stacked: the same lines a change replies with."""
    return "\n".join(setting_line_for(r) for r in _setting_rows())


def show_setting(text: str) -> str:
    row = resolve_setting(text)
    return setting_line_for(row) if row else wording.setting_unknown_guided(text)


def set_setting(store: Any, text: str, value: Any) -> str:
    row = resolve_setting(text)
    if row is None:
        return wording.setting_unknown_guided(text)
    if (row.get("kind") == "bool") and _norm(value) in ("on", "off", "true", "false", "yes", "no", "1", "0"):
        value = _norm(value) in ("on", "true", "yes", "1")
    from ..console import api as console_api
    old = row.get("value")
    try:
        from . import settings as gate_settings
        if gate_settings.coerce(row["name"], value) != gate_settings.coerce(row["name"], old):
            console_api.settings_set(store, {"name": row["name"], "value": value})
    except (ValueError, console_api.ValidationError):
        from . import admincmd
        return wording.setting_invalid(row.get("label") or row["name"], row, admincmd.setting_example(row))
    new_row = next((r for r in _setting_rows() if r["name"] == row["name"]), row)
    return setting_line_for(new_row, old_value=old, changed=True)


def handle(store: Any, cmd: Any, *, here: Tuple[str, str] = ("", "This chat")) -> str:
    """The reply to an access command, plus the phrase that undoes it when it changed
    something (see _undo_for). The one-line state itself comes from _handle."""
    before = _before_state(store, cmd, here)
    reply = _handle(store, cmd, here=here)
    undo = _undo_for(store, cmd, before, reply)
    return f"{reply}\n{wording.undo_line(undo)}" if undo else reply


def _before_state(store: Any, cmd: Any, here: Tuple[str, str]) -> Dict[str, Any]:
    """What the command is about to change, read before it runs, so the undo phrase can
    name the old value."""
    action = cmd.action
    out: Dict[str, Any] = {}
    if action == "set_mode":
        chat_id, title = _chat_target(store, cmd.chat or "", here)
        if chat_id:
            out["mode"] = roommode.effective_mode(store.get_response_mode(chat_id))
            out["title"] = title
    elif action == "set_setting":
        row = resolve_setting(getattr(cmd, "setting", None) or "")
        if row is not None:
            out["row"] = dict(row)
    elif action in ("add_role", "remove_role", "set_role"):
        person, _err = _one_person(store, cmd.person or "")
        if person is not None:
            out["person"] = person.get("display") or ""
            out["roles"] = list(_effective(store, person)[0])
    return out


def _undo_for(store: Any, cmd: Any, before: Dict[str, Any], reply: str) -> str:
    """The phrase that puts things back, only when this very message changed something."""
    action = cmd.action
    if action == "set_mode" and before.get("mode") and f"(was {before['mode']})" in reply:
        return f"set {before['title']} to {before['mode']}"
    if action == "set_setting" and before.get("row") and "(was " in reply:
        row = before["row"]
        label = " ".join(str(row.get("label") or row["name"]).split()).lower()
        old = row.get("value")
        if row.get("kind") == "bool":
            return f"turn {'on' if wording.setting_value_text(old, 'bool') == 'on' else 'off'} {label}"
        return f"set the {label} to {wording.setting_value_text(old, row.get('kind') or '')}"
    if action in ("add_role", "remove_role", "set_role") and before.get("roles") is not None and "(was " in reply:
        old = before["roles"]
        if len(old) == 1:
            return f"set {before['person']}'s role to {old[0]}"
        return " and ".join(f"make {before['person']} a {r}" for r in old[:2])
    return ""


def _handle(store: Any, cmd: Any, *, here: Tuple[str, str] = ("", "This chat")) -> str:
    """The one-line reply to an access command (router.AccessCommand or the shape
    match() builds). `here` is (chat_id, title) of the chat the message came in on,
    for "this chat"."""
    action = cmd.action
    if action in ("add_role", "remove_role", "set_role"):
        return change_roles(store, cmd.person or "", cmd.role or "", action.split("_", 1)[0])
    if action in ("grant", "revoke"):
        return change_abilities(store, cmd.person or "", cmd.abilities or "", action == "grant")
    if action == "show_person":
        return show_person(store, cmd.person or "")
    if action == "who_has":
        return who_has(store, cmd.abilities or cmd.role or "")
    if action == "set_mode":
        return set_mode(store, cmd.chat or "", cmd.mode or "", here)
    if action == "show_mode":
        return show_mode(store, cmd.chat or "", here)
    if action == "show_settings":
        return show_settings()
    if action == "show_setting":
        return show_setting(getattr(cmd, "setting", None) or "")
    if action == "set_setting":
        return set_setting(store, getattr(cmd, "setting", None) or "", getattr(cmd, "value", None))
    return wording.ABILITY_UNKNOWN


# ---------------------------------------------------------------------------
# match(): his common phrasings, before any model sees the message
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"^\s*\[[^\]\n]*\]\s*\n?")
_LEAD_RE = re.compile(r"^(?:@\w+[\s,:]*|(?:hey|ok|okay|so)\s+|blatbot[\s,:]+)+", re.I)
_TAIL_RE = re.compile(r"[\s.!]*(?:,?\s*please)?[\s.!]*$", re.I)
_MODES = r"(?P<mode>natural|mention|silent|quiet)"
_ABILITY_VERB = (r"(?:read|see|look|use|manage|edit|change|check|access|send|post|query|deploy|view|open|"
                 r"operate|act|write|update|search|book|schedule|text|email|message|control|run|pull)")

# (action, regex). Earlier patterns win. Named groups: person, role, abilities, chat, mode.
_PATTERNS: List[Tuple[str, re.Pattern]] = [(a, re.compile(p, re.I)) for a, p in [
    ("show_person", r"^what (?:can|could|may) (?P<person>.+?) do(?: now| without asking(?: me)?)?\??$"),
    ("show_person", r"^what (?:access|permissions?|roles?|scopes) does (?P<person>.+?) have\??$"),
    ("show_person", r"^what(?:'s| is) (?P<person>.+?)'s (?:role|access|permissions?)(?: now)?\??$"),
    ("show_person", r"^(?:show|check) (?:me )?(?P<person>.+?)'s (?:role|access|permissions?)$"),
    ("show_mode", r"^what(?:'s| is) (?:the |your )?mode (?:in|for|of) (?P<chat>.+?)\??$"),
    ("show_mode", r"^what mode (?:is|are you in) (?P<chat>.+?)(?: in| on)?\??$"),
    ("who_has", r"^who (?:has|can|gets|is allowed to|may) (?P<abilities>.+?)\??$"),
    ("who_has", r"^who (?:are|is)(?: the| a)? (?P<role>.+?)\??$"),
    ("revoke", r"^take (?P<abilities>.+?) away from (?P<person>.+?)$"),
    ("revoke", r"^take (?:away )?(?P<person>.+?)'s (?P<abilities>.+?)(?: away)?$"),
    ("revoke", r"^(?:revoke|remove) (?P<abilities>.+?) from (?P<person>.+?)$"),
    ("revoke", r"^(?P<person>.+?) (?:can no longer|shouldn't|should not|may not|is not allowed to) "
               r"(?P<abilities>.+?)(?: anymore| any more)?$"),
    ("set_role", r"^(?:set|change|switch) (?P<person>.+?)'s role to (?:an? |the )?(?P<role>.+?)$"),
    ("remove_role", r"^(?P<person>.+?) is no longer an? (?P<role>.+?)$"),
    ("remove_role", r"^(?:remove|take) (?P<person>.+?) (?:off|out of|from) (?:the )?(?P<role>.+?)$"),
    ("add_role", r"^make (?P<person>.+) (?:an?|the|one of the) (?P<role>[^,]+?)$"),
    ("add_role", r"^make (?P<person>.+?) (?P<role>[^,]+?)$"),
    ("add_role", r"^make (?P<person>.+) (?P<role>\S+)$"),
    ("add_role", r"^add (?P<person>.+?) to (?:the )?(?P<role>.+?)$"),
    ("grant", r"^(?:let|allow) (?P<person>.+?) (?:to )?(?P<abilities>" + _ABILITY_VERB + r"\b.+?)$"),
    ("grant", r"^give (?P<person>.+?) (?:access to|permission to) (?P<abilities>.+?)$"),
    ("grant", r"^give (?P<person>.+?) (?P<abilities>.+?) access$"),
    ("set_mode", r"^(?:set|put|switch|change) (?P<chat>.+?) (?:to|on|in|into) " + _MODES + r"(?: mode)?$"),
    ("set_mode", r"^(?:go|be|stay) " + _MODES + r"(?: (?:in|on) (?P<chat>.+?))?$"),
    ("set_mode", r"^" + _MODES + r" mode(?: (?:in|for|on) (?P<chat>.+?))?$"),
    ("show_settings", r"^(?:show|list|what are|tell me)(?: me)? (?:your|blatbot's|the|all|all your|all of your) "
                      r"settings\??$"),
    ("show_settings", r"^(?:show|list)(?: me)?(?: all)?(?: of)?(?: my| your)? settings(?: please)?$"),
    ("set_setting", r"^(?:set|change|put) (?:the |your )?(?P<setting>.+?) to (?P<value>.+?)$"),
    ("set_setting", r"^turn (?P<value>on|off) (?:the |your )?(?P<setting>.+?)$"),
    ("set_setting", r"^turn (?:the |your )?(?P<setting>.+?) (?P<value>on|off)$"),
    ("show_setting", r"^what(?:'s| is) (?:the |your )?(?P<setting>.+?)(?: set to| at| now)?\??$"),
]]


def clean(text: str) -> str:
    """The message without a group tag line, a leading "blatbot," or @mention, and
    trailing punctuation or "please"."""
    t = _TAG_RE.sub("", str(text or ""), count=1)
    t = " ".join(t.replace("’", "'").split())
    t = _LEAD_RE.sub("", t)
    return _TAIL_RE.sub("", t).strip()


def match(text: str, store: Any) -> Optional[Dict[str, Optional[str]]]:
    """One of Aaron's common access phrasings, as {action, person, role, abilities,
    chat, mode}, or None. Claimed only when the part that says WHAT is real: a role
    that exists, words that name a scope or system, a known mode. A person is never
    required to resolve here (an unknown name gets its own plain answer), but a
    pronoun ("what can you do?") is never taken for one."""
    t = clean(text)
    if not t or len(t) > 200:
        return None
    for action, pattern in _PATTERNS:
        m = pattern.match(t)
        if not m:
            continue
        g = {k: (v.strip() if isinstance(v, str) else v) for k, v in m.groupdict().items()}
        person = g.get("person")
        if person is not None and (_norm(person) in _PRONOUNS or not person):
            continue
        if action in ("add_role", "remove_role", "set_role"):
            if resolve_role(store, g.get("role") or "") is None:
                continue
        elif action in ("grant", "revoke"):
            if not resolve_abilities(g.get("abilities") or "", strict=True):
                continue
        elif action in ("set_setting", "show_setting"):
            if resolve_setting(g.get("setting") or "", strict=True) is None:
                continue
        elif action == "who_has":
            words = g.get("abilities") or g.get("role") or ""
            if resolve_role(store, words) is None and not resolve_abilities(words, strict=True):
                continue
        return {"action": action, "person": person, "role": g.get("role"),
                "abilities": g.get("abilities"), "chat": g.get("chat"), "mode": g.get("mode"),
                "setting": g.get("setting"), "value": g.get("value")}
    return None
