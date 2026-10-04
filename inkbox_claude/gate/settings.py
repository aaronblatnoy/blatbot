"""Knobs the owner can turn, and where their values come from.

A setting is read from the database first and the environment second, so a change made in
the console takes effect on the next turn without a restart, while anything set in the
environment still works as it always did. The registry below is the whole surface: a knob
not named here cannot be changed from the console, which keeps the dangerous ones out of
reach of a mis-click.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# name, kind, default, group, label, help
KNOBS: List[Dict[str, Any]] = [
    # -- when it speaks -------------------------------------------------------
    {"name": "GATE_GROUP_REPLY_MIN", "kind": "number", "default": 0.45, "min": 0.0, "max": 1.0,
     "step": 0.05, "group": "Speaking",
     "label": "Bar for speaking in a group unprompted",
     "help": "In a group chat, how sure it must be that a message not naming it is still its "
             "to answer. Lower means it joins in more; higher means it stays out unless named."},
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
    {"name": "GATE_ASSISTANT_ROLE", "kind": "text", "default": "", "group": "Speaking",
     "label": "Who it is in a room",
     "help": "One or two sentences describing its standing, used when it judges whether a "
             "group message is its to answer. Blank uses the built-in description."},

    # -- what it may do on its own -------------------------------------------
    {"name": "GATE_CONFIRM_SENDS", "kind": "bool", "default": True, "group": "Safety",
     "label": "Show me anything it would send",
     "help": "A message to another person that you asked for is held and shown to you first. "
             "Turning this off lets your own requests send without a second look."},
    {"name": "GATE_VOICE_TRUST_APPROVER", "kind": "bool", "default": False, "group": "Safety",
     "label": "Trust my voice on the phone",
     "help": "Caller ID can be faked, so by default a spoken request from your number still "
             "waits for a yes by text. Turn this on to act on calls from your number directly."},

    # -- how it runs ----------------------------------------------------------
    {"name": "GATE_EXECUTOR", "kind": "choice", "default": "claude", "choices": ["claude", "jev"],
     "group": "Engine", "label": "Which engine does the work",
     "help": "claude is Claude Code, slower and broader. jev is the typed-judgment agent, "
             "faster and cheaper but narrower."},
    {"name": "GATE_STRONG_MODEL", "kind": "choice", "default": "opus",
     "choices": ["opus", "sonnet", "haiku"], "group": "Engine",
     "label": "Model for hard requests", "help": "Used when a request is judged to need it."},
    {"name": "GATE_FAST_MODEL", "kind": "choice", "default": "sonnet",
     "choices": ["opus", "sonnet", "haiku"], "group": "Engine",
     "label": "Model for ordinary requests", "help": "Used for everything else."},
    {"name": "GATE_TASK_PICKER", "kind": "choice", "default": "jev", "choices": ["jev", "router"],
     "group": "Engine", "label": "Who picks which task a message belongs to",
     "help": "jev uses typed judgments; router asks the writing model."},
    {"name": "GATE_DECIDE_GRAPH", "kind": "bool", "default": True, "group": "Engine",
     "label": "Use the decision graph",
     "help": "Off falls back to the older straight-line path. A way out if the graph misbehaves."},
    {"name": "GATE_EXECUTOR_FALLBACK", "kind": "bool", "default": True, "group": "Engine",
     "label": "Fall back to Claude when the fast engine gives up",
     "help": "Only applies when the engine above is jev."},
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
        out.append({**spec, "value": get(name), "source": source})
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
