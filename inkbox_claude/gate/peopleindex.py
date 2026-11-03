"""Auto-indexing people: when Blatbot meets someone, they get a People record of
their own, and Aaron hears about it once.

"Meets" is any of three things, each decided in code, never by a model:
  - Aaron introduces someone in words ("blatbot, this is Stanley Rabkin, he runs X",
    "meet Stanley", "say hi to Stanley"): parse_introduction below.
  - someone with no record sends a message on any channel (gateway.py for Telegram,
    manager.handle_inbound for email, SMS and iMessage).
  - a new member is added to a Telegram group Blatbot is in (telegram.parse_new_members).

Indexing is a directory entry only. It writes a name, the handles actually seen, where
they were met, when, and the introduction as a note; it never writes a role or a scope
(store.index_person has no parameter for either).

What Aaron is told is a statement of the record as it now stands, never an event ("is
in People", not "I just added"). What he was last told is stored on the person
(persons.told), so the same record never produces a second text, a restart loses
nothing, and a later handle produces exactly one text naming only what is new.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Words that end a name in an introduction: "meet stanley he runs tamid" names Stanley.
_NAME_STOP = {
    "he", "she", "they", "him", "her", "them", "who", "whos", "who's", "from", "is", "my",
    "our", "your", "the", "a", "an", "and", "to", "of", "at", "in", "on", "for", "with",
    "please", "pls", "here", "there", "today", "now", "too", "also", "everyone", "guys",
    "blatbot", "he's", "she's", "they're", "hes", "shes", "its", "it's", "it", "this",
    "that", "be", "will", "can", "you", "i", "me", "we", "us", "was", "has", "have",
    "runs", "works", "just", "new", "senior", "junior",
}
# Words that are never a name on their own: "this is great", "say hi to everyone".
_NOT_A_NAME = _NAME_STOP | {
    "great", "good", "fine", "awesome", "amazing", "cool", "nice", "crazy", "wild", "huge",
    "ok", "okay", "perfect", "wrong", "right", "true", "false", "not", "so", "very", "what",
    "why", "how", "all", "everybody", "everything", "something", "nothing", "insane",
    "hilarious", "funny", "sick", "big", "bad", "done", "broken", "working", "fire",
    "important", "urgent", "weird", "interesting", "exactly", "correct", "me", "mine",
    "going", "gonna", "about", "like", "really", "actually", "probably", "definitely",
}
_TRIGGER_RE = re.compile(
    r"(?:^|[\s,.;:!?-])(?P<trigger>this\s+is|meet|say\s+(?:hi|hello)\s+to|introducing|"
    r"i(?:'d|\s+would)\s+like\s+you\s+to\s+meet|(?:i\s+)?want\s+you\s+to\s+meet)\s+"
    r"(?P<rest>.+)$", re.IGNORECASE | re.DOTALL)
_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")
_AUTOMATED_LOCAL_RE = re.compile(
    r"^(?:no-?reply|do-?not-?reply|donotreply|notifications?|notify|mailer-daemon|postmaster|"
    r"bounces?|alerts?|news(?:letter)?s?|updates?|digest|marketing|billing|receipts?|"
    r"invoices?|calendar-notification|support|help|info|hello|team|admin|system|automated|"
    r"robot|bot|service|accounts?|security|verify|verification)(?:[+._-].*)?$", re.IGNORECASE)


def parse_introduction(text: str, *, addressed: bool) -> Optional[Tuple[str, str]]:
    """(name, note) when `text` introduces someone to Blatbot, else None.

    The name is up to three words after the trigger, ending at punctuation or a word
    that starts the rest of the sentence ("he", "who", "from" ...); the note is what
    the sentence says after the name ("he runs X"), "" when it says nothing more.

    "this is X" is everyday speech, so it only counts when the message is aimed at
    Blatbot and the name was typed capitalised: "blatbot, this is Stanley" yes, "this
    is great" never. The other triggers are introductions by their wording alone but
    still need the message aimed at Blatbot."""
    raw = " ".join(str(text or "").split())
    if not raw or not addressed:
        return None
    m = _TRIGGER_RE.search(raw)
    if not m:
        return None
    trigger = " ".join(m.group("trigger").lower().split())
    rest = m.group("rest")
    words = re.split(r"(\s+)", rest)
    name: List[str] = []
    consumed = 0
    for w in words:
        consumed += len(w)
        if not w.strip():
            continue
        core = w.strip()
        ends = core[-1] in ",.;:!?)"
        token = core.rstrip(",.;:!?)").strip("\"'(")
        if token.endswith("'s") or token.endswith("’s"):
            break
        if not token or not _NAME_TOKEN_RE.match(token) or token.lower() in _NAME_STOP:
            consumed -= len(w)
            break
        name.append(token)
        if ends or len(name) == 3:
            break
    if not name or name[0].lower() in _NOT_A_NAME:
        return None
    if trigger == "this is" and not name[0][0].isupper():
        return None
    display = " ".join(n[:1].upper() + n[1:] for n in name)
    note = rest[consumed:].strip(" ,.;:-")
    note = note[:1].upper() + note[1:] if note else ""
    return display, note


def is_automated_email(address: str) -> bool:
    """A mailbox no person writes from (noreply@, notifications@, billing@ ...). Mail
    from one is not someone Blatbot met."""
    local = str(address or "").strip().lower().split("@", 1)[0]
    return bool(local) and bool(_AUTOMATED_LOCAL_RE.match(local))


def same_name(a: str, b: str) -> bool:
    """One display name is the other with words added at the end: "Stanley" and
    "Stanley Rabkin" are the same name, "Stan" and "Stanley" are not."""
    ta, tb = str(a or "").casefold().split(), str(b or "").casefold().split()
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return long_[:len(short)] == short


# -- what Aaron is told -------------------------------------------------------

def _handle_text(kind: str, raw: str, value: str) -> str:
    shown = raw or value
    if kind == "telegram":
        return f"Telegram id {value}"
    if kind == "telegram_username":
        return f"Telegram {shown if shown.startswith('@') else '@' + shown}"
    if kind == "phone":
        return f"phone {shown}"
    if kind == "email":
        return f"email {shown}"
    if kind == "imessage":
        return f"iMessage {shown}"
    return shown


def state_of(person: Dict[str, Any]) -> Dict[str, Any]:
    """The part of a record Aaron is told about: name, reachable handles, note. Where
    they were met is said once, in the first text, and is not a material change."""
    handles = sorted({f"{c['kind']}:{c['value']}" for c in person.get("contacts") or []
                      if c.get("kind") != "name"})
    return {"display": person.get("display") or "", "handles": handles,
            "note": (person.get("note") or "").strip()}


def _handles_shown(contacts: List[Dict[str, Any]], only: Optional[Iterable[str]] = None,
                   known: Optional[Iterable[str]] = None) -> List[str]:
    """Handles as Aaron reads them. A Telegram id is left out when the person also has
    a Telegram username (the id is a number he has never seen), unless it is the only
    handle there is."""
    rows = [c for c in contacts if c.get("kind") != "name"]
    if only is not None:
        wanted = set(only)
        rows = [c for c in rows if f"{c['kind']}:{c['value']}" in wanted]
    all_kinds = {c["kind"] for c in contacts} | {h.split(":", 1)[0] for h in (known or [])}
    out = []
    for c in rows:
        if c["kind"] == "telegram" and "telegram_username" in all_kinds and len(rows) > 1:
            continue
        out.append(_handle_text(c["kind"], c.get("raw_value") or "", c["value"]))
    return out


# Where someone was met when it was not a group: the channel itself.
CHANNEL_PLACES = {"Telegram DM": "on Telegram", "email": "over email", "SMS": "over SMS",
                  "iMessage": "on iMessage"}


def _place_text(place: str) -> str:
    """ "in FellowSIp of the Ring" for a group, "over email" for a channel."""
    return CHANNEL_PLACES.get(place) or f"in {place}"


def _join(items: List[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def notice_line(person: Dict[str, Any], told: Optional[Dict[str, Any]]) -> str:
    """One sentence stating a record as it stands, or "" when nothing material is new
    since `told` (what Aaron was last told; None when he was never told)."""
    now = state_of(person)
    name = now["display"] or "Someone"
    contacts = person.get("contacts") or []
    if told is None:
        parts = [f"{name} is in People"]
        handles = _handles_shown(contacts)
        detail = []
        if handles:
            detail.append(_join(handles))
        places = [_place_text(p) for p in person.get("met_in") or [] if p]
        if places:
            detail.append("met " + _join(places))
        line = parts[0] + (": " + ", ".join(detail) if detail else "") + "."
        if now["note"]:
            line += f" Note: {now['note']}"
            line += "" if line.endswith((".", "!", "?")) else "."
        return line
    new_handles = [h for h in now["handles"] if h not in set(told.get("handles") or [])]
    renamed = bool(told.get("display")) and told.get("display") != now["display"]
    new_note = now["note"] and now["note"] != (told.get("note") or "")
    if not (new_handles or renamed or new_note):
        return ""
    head = f"{name} (was {told['display']})" if renamed else name
    line = f"{head} in People"
    shown = _handles_shown(contacts, only=new_handles, known=told.get("handles"))
    if shown:
        line += " now also has " + _join(shown)
    line += "."
    if new_note:
        line += f" Note: {now['note']}"
        line += "" if line.endswith((".", "!", "?")) else "."
    return line


def told_json(person: Dict[str, Any]) -> str:
    return json.dumps(state_of(person), sort_keys=True)
