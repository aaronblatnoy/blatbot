"""Knobs the owner can turn, and where their values come from.

A setting is read from the database first and the environment second, so a change made in
the console takes effect on the next turn without a restart, while anything set in the
environment still works as it always did. The registry below is the whole surface: a knob
not named here cannot be changed from the console, which keeps the dangerous ones out of
reach of a mis-click.
"""

from __future__ import annotations

from inkbox_claude.instance import ASSISTANT as _ASSISTANT, ASSISTANT_UPPER as _ASSISTANT_UPPER, ASSISTANT_EMAIL as _ASSISTANT_EMAIL, SERVER_NAME as _SERVER_NAME  # noqa: F401
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# name, kind, default, group, label, help
KNOBS: List[Dict[str, Any]] = [
    # -- when it speaks -------------------------------------------------------
    {"name": "GATE_INBOX", "kind": "bool", "default": True, "group": "Speaking",
     "label": "Use the message inbox",
     "help": "Save new messages before handling them and combine messages sent close together. "
             "Turn this off to handle each new message as it arrives."},
    {"name": "INBOX_SETTLE_SECONDS", "kind": "number", "default": 1.0, "min": 0.0,
     "max": 60.0, "step": 0.5, "group": "Speaking",
     "label": "Seconds to wait for more messages",
     "help": "Wait this long after the newest message before answering a conversation."},
    {"name": "GATE_GROUP_REPLY_MIN", "kind": "number", "default": 0.45, "min": 0.0, "max": 1.0,
     "step": 0.05, "group": "Speaking",
     "label": "Bar for speaking in a group unprompted",
     "help": "In a group chat, how sure it must be that a message not naming it is still its "
             "to answer. Lower means it joins in more; higher means it stays out unless named."},
    {"name": "GATE_GROUP_PLAIN_TALK", "kind": "bool", "default": True, "group": "Speaking",
     "label": "Chat in groups without asking me",
     "help": "In a group chat, answer plain conversation from someone else straight away, with no "
             "tools and nothing beyond what the room can already see. Anything that needs a tool, "
             "a lookup, memory or an action still goes through your yes. Turn this off to handle "
             "group messages the usual way."},
    {"name": "GATE_ACTIONABLE_MIN", "kind": "number", "default": 0.4, "min": 0.0, "max": 1.0,
     "step": 0.05, "group": "Speaking",
     "label": "Bar for acting without asking what you meant",
     "help": "How clear a request must be before it goes to the execution layer. Below this it "
             "asks you one question instead of running and failing."},
    {"name": "GATE_SLOW_ACK_AFTER_S", "kind": "number", "default": 7, "min": 0, "max": 60,
     "step": 1, "group": "Speaking",
     "label": "Seconds before it says it is working",
     "help": "How long a run may take in silence before it sends a short line so you know it "
             "heard you."},
    {"name": "GATE_TELEGRAM_STALE_MIN", "kind": "number", "default": 10, "min": 0, "max": 1440,
     "step": 1, "group": "Speaking",
     "label": "Minutes after which a late Telegram message is only filed",
     "help": "After an outage Telegram delivers every message it held. One sent longer ago than "
             "this is kept in the history and memory but starts nothing and gets no reply. "
             "0 turns the check off."},
    {"name": "GATE_ASSISTANT_ROLE", "kind": "text", "default": "", "group": "Speaking",
     "label": "Who it is in a room",
     "help": "One or two sentences describing its standing, used when it judges whether a "
             "group message is its to answer. Blank uses the built-in description."},

    # -- what it may do on its own -------------------------------------------
    {"name": "GATE_CONFIRM_SENDS", "kind": "bool", "default": True, "group": "Safety",
     "label": "Show me anything it would send",
     "help": "A message to another person that you asked for is held and shown to you first. "
             "Turning this off lets your own requests send without a second look."},
    {"name": "GATE_HOLD_SENDS_TO_OTHERS", "kind": "bool", "default": True, "group": "Safety",
     "label": "Second yes for every message to someone else",
     "help": "Any message a request would send to anyone but you, on any channel, is held and "
             "shown to you word for word with its own #id yes / no, even when the request was "
             "already approved and even when you asked for it. Turning this off restores the "
             "earlier rules."},
    {"name": "GATE_VOICE_TRUST_APPROVER", "kind": "bool", "default": False, "group": "Safety",
     "label": "Trust my voice on the phone",
     "help": "Caller ID can be faked, so by default a spoken request from your number still "
             "waits for a yes by text. Turn this on to act on calls from your number directly."},
    {"name": "GATE_SCHEDULES_PAUSED", "kind": "bool", "default": False, "group": "Safety",
     "label": "Pause every schedule",
     "help": "Stops new scheduled runs from starting. Existing runs keep going."},
    {"name": "GATE_FOLLOWUPS", "kind": "bool", "default": True, "group": "Safety",
     "label": "Let runs set follow-up timers",
     "help": "Lets ordinary runs check back later on work they already have permission to do."},
    {"name": "GATE_ACCESS_ASKS", "kind": "bool", "default": True, "group": "Safety",
     "label": "Let runs ask for access",
     "help": "Lets a run pause and ask for a missing system, then continue when access is decided."},
    {"name": "GATE_RUN_OUTCOMES", "kind": "bool", "default": True, "group": "Safety",
     "label": "Runs choose how they end",
     "help": "A run answers with its own send, says nothing needs sending, or asks you. Off: the run's closing text is delivered as before."},
    {"name": "GATE_PERSON_INBOX", "kind": "bool", "default": True, "group": "Safety",
     "label": "One inbox per person",
     "help": "Messages from one person on several surfaces are handled one at a time. Off: one inbox per chat."},
    {"name": "GATE_ASK_AARON", "kind": "bool", "default": True, "group": "Safety",
     "label": "Let runs ask you a question",
     "help": "A run that needs a detail or a decision only you can give asks you and continues with your answer."},
    {"name": "GATE_NATIVE_DEFER", "kind": "bool", "default": True, "group": "Safety",
     "label": "Pause on the exact tool call",
     "help": "Save a call while it waits for your answer, then continue that same run."},
    {"name": "GATE_AUTO_MODE", "kind": "bool", "default": True, "group": "Safety",
     "label": "Use Claude Code auto mode",
     "help": f"Let Claude Code screen risky actions on top of {_ASSISTANT}'s own permissions."},
    {"name": "SESSION_MAX_AGE_HOURS", "kind": "number", "default": 24, "min": 0,
     "max": 8760, "step": 1, "group": "Engine",
     "label": "Hours to keep a saved session",
     "help": "Start fresh when a saved run is older than this."},

    # -- formatted pages ------------------------------------------------------
    {"name": "GATE_HTML_PAGES_AUDIENCE", "kind": "choice", "default": "owner",
     "choices": ["off", "owner", "everyone"], "group": "Pages",
     "label": "Who gets a formatted web page for a long answer",
     "help": "A long or structured answer can go out as a web page with a short text and its link. "
             "owner means only replies to you; everyone widens it to every reply; off turns pages off."},
    {"name": "GATE_HTML_PAGE_DAYS", "kind": "number", "default": 30, "min": 1, "max": 365,
     "step": 1, "group": "Pages",
     "label": "Days a page link keeps working",
     "help": "After this many days the link returns not found and the page is deleted."},

    # -- how it runs ----------------------------------------------------------
    {"restart": True, "name": "GATE_EXECUTOR", "kind": "choice", "default": "claude", "choices": ["claude", "jev"],
     "group": "Engine", "label": "Which engine does the work",
     "help": "claude is Claude Code, slower and broader. jev is the typed-judgment agent, "
             "faster and cheaper but narrower."},
    {"name": "GATE_STRONG_MODEL", "kind": "choice", "default": "opus",
     "choices": ["opus", "sonnet", "haiku"], "group": "Engine",
     "label": "Model for hard requests", "help": "Used when a request is judged to need it."},
    {"name": "GATE_FAST_MODEL", "kind": "choice", "default": "sonnet",
     "choices": ["opus", "sonnet", "haiku"], "group": "Engine",
     "label": "Model for ordinary requests", "help": "Used for everything else."},
    {"restart": True, "name": "GATE_TASK_PICKER", "kind": "choice", "default": "jev", "choices": ["jev", "router"],
     "group": "Engine", "label": "Who picks which task a message belongs to",
     "help": "jev uses typed judgments; router asks the writing model."},
    {"restart": True, "name": "GATE_DECIDE_GRAPH", "kind": "bool", "default": True, "group": "Engine",
     "label": "Use the decision graph",
     "help": "Off falls back to the older straight-line path. A way out if the graph misbehaves."},
    {"restart": True, "name": "GATE_EXECUTOR_FALLBACK", "kind": "bool", "default": True, "group": "Engine",
     "label": "Fall back to Claude when the fast engine gives up",
     "help": "Only applies when the engine above is jev."},

    # -- the vault (second brain) --------------------------------------------
    {"name": "GATE_CONTEXT_BUDGET_CHARS", "kind": "number", "default": 8000, "min": 1000, "max": 200000,
     "step": 500, "group": "Vault",
     "label": "Total context a run carries, in characters",
     "help": "The task ledger and the vault attachment together are cut to roughly this many "
             "characters before a run starts: the most recent and the matching ledger events, and "
             "the vault sections/bullets that match the request, rather than every whole note and "
             "the full task history. Raise it if answers are missing context the task actually "
             "has; lower it to speed runs up further."},
    {"name": "GATE_VAULT_ATTACH_BUDGET_CHARS", "kind": "number", "default": 60000, "min": 2000, "max": 2000000,
     "step": 1000, "group": "Vault",
     "label": "Hard ceiling on the vault attachment alone, in characters",
     "help": "A second, looser cap on top of the total context budget above: the vault attachment "
             "never exceeds this regardless of how little the ledger used. Notes beyond either "
             "budget are listed by path instead of attached, so the run can still open them itself."},
]

_BY_NAME = {k["name"]: k for k in KNOBS}
_lock = threading.Lock()
_cache: Dict[str, Any] = {}
_cache_at = 0.0
_TTL = 2.0          # a change in the console shows up on the next turn, not the next restart
_store = None


def bind(store: Any) -> None:
    """Point the settings at the database. Called once when the gate comes up."""
    global _store
    _store = store


def _db_values() -> Dict[str, str]:
    global _cache, _cache_at
    now = time.time()
    with _lock:
        if _store is not None and now - _cache_at > _TTL:
            try:
                _cache = _store.settings()
                _cache_at = now
            except Exception:
                logger.debug("[settings] could not read overrides", exc_info=True)
        return dict(_cache)


def raw(name: str) -> Optional[str]:
    """The value as text: the database first, then the environment, then nothing."""
    v = _db_values().get(name)
    if v is not None and str(v).strip() != "":
        return str(v)
    env = os.getenv(name)
    return env if env is not None and env.strip() != "" else None


def get(name: str, default: Any = None) -> Any:
    """One setting, typed by its registry entry. Unknown names fall back to the environment."""
    spec = _BY_NAME.get(name)
    v = raw(name)
    if spec is None:
        return v if v is not None else default
    if v is None:
        return spec["default"] if default is None else default
    kind = spec["kind"]
    try:
        if kind == "bool":
            return str(v).strip().lower() in ("1", "true", "yes", "on")
        if kind == "number":
            f = float(v)
            return int(f) if float(spec["default"]).is_integer() and f.is_integer() else f
        if kind == "choice":
            return v if v in spec["choices"] else spec["default"]
        return v
    except Exception:
        return spec["default"]


def describe() -> List[Dict[str, Any]]:
    """Every knob with where its value is coming from, for the console."""
    db = _db_values()
    out = []
    for spec in KNOBS:
        name = spec["name"]
        source = "console" if db.get(name) not in (None, "") else (
            "environment" if os.getenv(name) not in (None, "") else "default")
        out.append({**spec, "value": get(name), "source": source,
                    "restart": bool(spec.get("restart"))})
    return out


def known(name: str) -> bool:
    return name in _BY_NAME


def coerce(name: str, value: Any) -> str:
    """Validate a value against its registry entry and return what to store."""
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"{name} is not a setting the console may change")
    kind = spec["kind"]
    if kind == "bool":
        return "1" if (value is True or str(value).strip().lower() in ("1", "true", "yes", "on")) else "0"
    if kind == "number":
        f = float(value)
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and f < lo or hi is not None and f > hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")
        return str(f)
    if kind == "choice":
        v = str(value)
        if v not in spec["choices"]:
            raise ValueError(f"{name} must be one of {', '.join(spec['choices'])}")
        return v
    return str(value)


def invalidate() -> None:
    global _cache_at
    with _lock:
        _cache_at = 0.0
