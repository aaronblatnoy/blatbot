"""Turning the owner's words into a schedule proposal he can judge, and the
permission that comes with it -- chosen without any probability judgment.

A scope is granted to a schedule only for one of three reasons, all deterministic
and grounded in something that already happened: he named it, his own message
used a synonym for it, or the schedule's prompt names one of its tools literally.
Jev is never called from this module.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from .schedules import names_tool
from .scopes import LABELS, SCOPES, SYNONYMS


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def label(scope: str) -> str:
    """The owner-facing verb phrase for a scope ("read public web pages"). A
    dynamically discovered vault knowledge scope (name "vault:<folder>") has no
    entry in scopes.yaml, so its label is generated here from its folder name."""
    found = LABELS.get(scope)
    if found:
        return found
    if scope.startswith("vault:"):
        return f"read your notes on {scope.split(':', 1)[1]}"
    # Fallback for a scope with no written label and no vault pattern: a scope
    # that is in the registry but was never given one, which registry_problems()
    # style checks are expected to catch during review, not here.
    return scope.replace("_", " ")


def labels(scopes: Iterable[str]) -> str:
    """Scope labels joined in words: one, "X and Y", or "X, Y, and Z" for three
    or more."""
    ls = [label(s) for s in scopes]
    if not ls:
        return ""
    if len(ls) == 1:
        return ls[0]
    if len(ls) == 2:
        return f"{ls[0]} and {ls[1]}"
    return ", ".join(ls[:-1]) + ", and " + ls[-1]


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class UnknownScopes(ValueError):
    """Raised by capabilities_for when named_scopes or suggested_scopes contains
    a name that is not in the registry, so the caller can refuse the proposal
    with a plain reply instead of silently dropping or granting it."""

    def __init__(self, names: List[str]):
        self.names = names
        super().__init__(f"unknown scope(s): {', '.join(names)}")


def _said_scopes(said_text: str) -> List[str]:
    """Scope names whose synonym appears in the owner's own words, whole-phrase
    and case-insensitive. Synonyms are checked longest-first and a match
    consumes its span of the text, so a shorter synonym that is only a
    substring of an already-matched longer one is not also counted."""
    text = said_text or ""
    remaining = text.lower()
    candidates = [(len(syn), name, syn.lower()) for name, syns in SYNONYMS.items() for syn in syns]
    # Every scope's own label is always a valid phrase for that scope too (the
    # recovery phrase the bot itself dictates, "let it <label>", must work for
    # every scope, including a discovered vault knowledge scope with no entry
    # in scopes.yaml). Where a label and a synonym overlap, the longest match
    # wins via the same sort below.
    candidates += [(len(label(name).lower()), name, label(name).lower()) for name in SCOPES]
    candidates.sort(key=lambda t: -t[0])
    found: List[str] = []
    for _, name, syn in candidates:
        # A hand-written \b fails at either end of a phrase that ends or
        # starts on punctuation (several labels do, e.g. "...(submit, post,
        # mark complete)"): \b needs a word character on exactly one side,
        # and two adjacent non-word characters are never a boundary. These
        # lookarounds require a non-word character (or string edge) outside
        # the phrase instead, which also still blocks a word being matched
        # mid-token (no match of "cat" inside "category").
        m = re.search(r"(?<![\w])" + re.escape(syn) + r"(?![\w])", remaining)
        if not m:
            continue
        if name not in found:
            found.append(name)
        start, end = m.span()
        remaining = remaining[:start] + (" " * (end - start)) + remaining[end:]
    return found


def _prompt_tool_scopes(prompt: str) -> List[str]:
    """Scopes whose own tools are named literally, by short name, in a prompt."""
    found: List[str] = []
    for name, info in SCOPES.items():
        for tool in info.get("tools") or []:
            if names_tool(prompt, tool):
                if name not in found:
                    found.append(name)
                break
    return found


def capabilities_for(prompt: str, named_scopes: Optional[List[str]], suggested_scopes: Optional[List[str]],
                      said_text: str) -> List[str]:
    """The de-duplicated union, in registry order, of: scopes he named
    explicitly, scopes his own words name by synonym, scopes whose tools the
    prompt names literally, and the scopes the router suggests the work needs
    (a suggestion only -- it becomes permission only with his yes).

    A scope he named himself (named_scopes) that is not in the registry
    always refuses the proposal with UnknownScopes -- that is his own typed
    or spoken word for something that does not exist. A scope the router
    merely *suggests* that is not in the registry (the model hallucinating a
    name) is dropped silently as long as some other, real capability is
    still matched from any source; it is refused only when nothing real is
    left to propose at all, so the caller can fall back to "tell me what it
    should be allowed to do" rather than naming a made-up scope back to him."""
    named_unknown = sorted({n for n in (named_scopes or []) if n not in SCOPES})
    if named_unknown:
        raise UnknownScopes(named_unknown)
    valid_suggested = [n for n in (suggested_scopes or []) if n in SCOPES]
    matched = set(named_scopes or []) | set(_said_scopes(said_text)) | set(_prompt_tool_scopes(prompt)) | set(valid_suggested)
    if not matched:
        suggested_unknown = sorted({n for n in (suggested_scopes or []) if n not in SCOPES})
        if suggested_unknown:
            raise UnknownScopes(suggested_unknown)
    return [name for name in SCOPES if name in matched]


def friendly_error(exc: Exception) -> str:
    """Plain owner-facing text for a scope/cadence service error -- never the
    words "scope", "cadence" or "cadence_kind". Shared by the proposal path
    (manager.py) and the schedule-management path (schedcmd.py) so an error
    from either is worded the same way."""
    if isinstance(exc, UnknownScopes):
        return ('I can\'t tell what it should be allowed to do. Tell me in plain words, '
                'like "read public web pages".')
    from . import cadence as cadence_mod
    msg = str(exc)
    if isinstance(exc, cadence_mod.CadenceError):
        if "at least 5 minutes" in msg:
            return "It has to run at least every 5 minutes."
        return "I didn't understand that timing."
    if msg.startswith("missing capability:"):
        return f"It isn't allowed to {msg.split(':', 1)[1].strip()} yet."
    # "already has a run in flight", "needs to be told how often to run
    # first", "That time has already passed." -- already plain sentences.
    return msg


# Phrases that open a permission grant ("let it edit contacts") and words that
# close one off within the same clause ("but don't send email"). Used only by
# capabilities_from_permission -- the mid-run, no-fresh-approval widening path
# (item D) -- never by capabilities_for, whose said_text is read at proposal
# time where a plain description of the work ("update the contact note") is
# not itself a permission grant.
_PERMISSION_PHRASES = [
    "let it", "allow it to", "it can", "it may", "give it", "it should be able to",
]
_NEGATION_WORDS = ["don't", "do not", "not", "never", "without", "except", "but not", "can't", "cannot",
                   "won't", "shouldn't", "mustn't", "no longer", "stop", "isn't allowed", "may not"]
# A permission phrase must stand as its own words: "it can" inside "it can't" or
# "it cannot" is a refusal, not a grant.
_PERMISSION_RE = re.compile(
    r"\b(?:let it|allow it to|it can|it may|give it|it should be able to)\b(?!'t|not| not)")
_VAGUE_PHRASES = ["do that", "fix that", "fix it"]
# Comma is deliberately NOT a split point: several scope labels are
# themselves comma-separated lists ("read TAMID's Drive, Sheets, Docs and
# Forms"), and splitting on every comma would strand the "let it" phrase in
# the first fragment and leave the rest unmatched. A negated tail is still
# cut off below via _NEGATION_WORDS ("not"/"without"/etc), comma or no comma.
_CLAUSE_SPLIT_RE = re.compile(r"[.;]| but | except | without ")


def _permitted_scopes_from_words(said_text: str) -> List[str]:
    """Scopes named by a synonym inside a clause that a permission phrase
    opens, never inside a clause (or the part of one) that a negation word
    closes off first."""
    text = (said_text or "").lower().replace("\u2019", "'")
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
    found: List[str] = []
    for clause in clauses:
        if not _PERMISSION_RE.search(clause):
            continue
        cut = len(clause)
        for neg in _NEGATION_WORDS:
            # Word-boundary search: plain substring search would find "not"
            # inside "notes" (as in the label "read your notes") and wrongly
            # truncate the clause before the scope name ever appears.
            m = re.search(r"\b" + re.escape(neg) + r"\b", clause)
            idx = m.start() if m else -1
            if idx != -1:
                cut = min(cut, idx)
        for name in _said_scopes(clause[:cut]):
            if name not in found:
                found.append(name)
    return [name for name in SCOPES if name in found]


def capabilities_from_permission(said_text: str, denied_scopes: Optional[List[str]] = None) -> List[str]:
    """What a mid-run "let it ..."/"it can't ... yet" exchange grants (item D):
    only scopes matched from a permission clause in his own words, never the
    router's suggestion. When nothing matches but his words are a vague
    approval ("let it do that", "fix that") and a previous run was refused
    something, narrow to exactly those denied scopes; otherwise grant
    nothing, so the caller can ask what it should be allowed to do."""
    found = _permitted_scopes_from_words(said_text)
    if found:
        return found
    text = (said_text or "").lower().replace("\u2019", "'")
    refused = re.search(r"\bno\b", text) or any(
        re.search(r"\b" + re.escape(neg) + r"\b", text) for neg in _NEGATION_WORDS)
    if denied_scopes and not refused and any(phrase in text for phrase in _VAGUE_PHRASES):
        denied = set(denied_scopes)
        return [name for name in SCOPES if name in denied]
    return []


def capability_match(text: str) -> Optional[str]:
    """The first known scope whose label or a synonym appears in `text` (a
    run's own NEEDS or RESULT sentence), or None. Used to decide whether a
    blocked run's notice can offer a concrete "let it <label>" fix, or must
    fall back to asking the owner where the thing is -- never guesses a
    capability that is not actually in the registry."""
    found = _said_scopes(text or "")
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Report-every-time default (item 4): a digest hears every time by default, a
# watch only on change, unless the router states a value explicitly (that
# check happens in the caller -- this function is only ever consulted when no
# explicit value was given).
# ---------------------------------------------------------------------------

_DIGEST_WORDS = [
    "summary", "summarize", "summarise", "recap", "digest", "brief", "briefing",
    "report", "roundup", "send me", "text me", "tell me every", "each morning",
    "every morning",
]
_WATCH_WORDS = ["when ", "if ", "changes", "alert me", "let me know if"]


def default_report_every_time(text: str, cadence_kind: str) -> bool:
    """True (hear from it every time) when the owner's words or the schedule's
    prompt read as a digest; False (only on change) when they read as a watch.
    Neither matching, a one-time schedule defaults to every time (there is
    only ever one run to hear about); a repeating one defaults to only on
    change, per the specification's default for a watch."""
    low = (text or "").lower()
    if any(w in low for w in _DIGEST_WORDS):
        return True
    if any(w in low for w in _WATCH_WORDS):
        return False
    return cadence_kind == "once"


def missing_capabilities(prompt: str, scopes: Iterable[str]) -> List[str]:
    """Scopes whose tools the prompt names literally but that are not in
    `scopes` -- what a proposal or an activation attempt is missing."""
    granted = set(scopes)
    out: List[str] = []
    for name, info in SCOPES.items():
        if name in granted:
            continue
        for tool in info.get("tools") or []:
            if names_tool(prompt, tool):
                out.append(name)
                break
    return out


# ---------------------------------------------------------------------------
# Wording seams onto cadence.py (owned by another package; not built yet in
# this tree). Tests replace these two functions directly.
# ---------------------------------------------------------------------------

def _cadence_text(schedule: Any) -> str:
    from . import cadence as cadence_mod
    return cadence_mod.describe(schedule.cadence_kind, schedule.cadence, schedule.timezone)


def _first_run_text(schedule: Any) -> str:
    from . import cadence as cadence_mod
    when = cadence_mod.next_run(schedule.cadence_kind, schedule.cadence, schedule.timezone, time.time())
    return _format_time(when, schedule.timezone)


def _format_time(when: Optional[float], timezone_name: str) -> str:
    if when is None:
        return "an unknown time"
    tz = ZoneInfo(timezone_name or "America/New_York")
    dt = datetime.fromtimestamp(when, tz)
    now = datetime.now(tz)
    clock = dt.strftime("%-I:%M %p")
    if dt.date() == now.date():
        return f"{clock} today"
    if dt.date() == now.date() + timedelta(days=1):
        return f"{clock} tomorrow"
    return f"{dt.strftime('%A, %B %-d')}, {clock}"


def _until_text(schedule: Any) -> str:
    return _format_time(schedule.until, schedule.timezone).replace(" today", "").replace(" tomorrow", "")


# ---------------------------------------------------------------------------
# What he reads
# ---------------------------------------------------------------------------

def format_proposal(schedule: Any) -> str:
    """The proposal message, in the specification's voice. No scope names, no
    cadence kind, no cron string, no "#<id>" and no markdown."""
    cadence_text = _cadence_text(schedule)
    first_run = _first_run_text(schedule)
    sentence = f'Set up "{schedule.title}": {cadence_text}, starting {first_run}'
    if getattr(schedule, "until", None):
        sentence += f", until {_until_text(schedule)}"
    if schedule.scopes:
        sentence += ". It will " + labels(schedule.scopes) + "."
    else:
        # Belt and suspenders: manager.py refuses to propose a schedule with
        # no capabilities at all, but format_proposal must never read "It
        # will ." if that guard is ever bypassed (an edit path, a test).
        sentence += ". It will not be able to use any tools."
    hears = "every time it runs" if schedule.report_every_time else "only when something changed or it could not finish"
    sentence += f" You hear from it {hears}."
    sentence += " Reply yes to start, or tell me what to change."
    return sentence


def format_change(schedule: Any, changes: Dict[str, Any]) -> str:
    """The message after an edit: the new cadence if it changed, and/or newly
    added capabilities. Each half asks for his yes on its own."""
    parts: List[str] = []
    if any(k in changes for k in ("cadence_kind", "cadence", "until", "report_every_time")):
        parts.append(f"{schedule.title} will run {_cadence_text(schedule)}, next {_first_run_text(schedule)}. "
                     "Reply yes to confirm.")
    added = changes.get("scopes")
    if added:
        parts.append(f"{schedule.title} will also be allowed to {labels(added)}. Reply yes to confirm.")
    return " ".join(parts)
