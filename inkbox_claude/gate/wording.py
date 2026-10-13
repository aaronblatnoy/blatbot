"""Plain-language wording for a finished request's result message.

A failed run's status dict (as the executor returns it: ``ok``, ``error``,
``denied``, ``held``, ``final_status``, ``tool_calls``, ...) is never shown to
a person as-is -- its ``error``/``raw`` text is an SDK or provider string, not
something a person asked for. This module turns that status into one plain
clause people can read, plus one next step, and assembles the whole result
message a person receives. The raw text stays in the request row and the log;
nothing here ever returns it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import schedprop
from . import scopes as _scopes

_TOOL_SYSTEM_NAMES: Dict[str, str] = {
    "tamid-admin": "the TAMID website admin",
    "sjba-admin": "the SJBA website admin",
    "tamid-drive": "TAMID's Google Workspace",
    "stern-drive": "Aaron's Stern Google account",
    "tamid-instagram": "TAMID's Instagram",
    "tamid-linkedin": "TAMID's LinkedIn",
    "analytics-admin": "TAMID's Google Analytics",
    "brightspace": "Brightspace",
    "coolify": "Coolify",
    "coolify-vox": "the Vox Coolify instance",
    "playwright": "the web",
    "inkbox": "Blatbot's own inbox",
    "host": "the server",
    "advancedmd": "AdvancedMD",
    "availity": "Availity",
    "quickbooks": "QuickBooks",
    "dol": "the pension filings database",
    "runpod": "RunPod",
    "google-workspace": "Google Workspace",
}

_NETWORK_WORDS = (
    "connection", "network", "unreachable", "could not resolve", "name resolution",
    "dns", "connection refused", "reset by peer", "ssl", "certificate", "handshake",
    "bad gateway", "gateway timeout", "service unavailable", "502", "503", "504",
    "econnrefused", "econnreset", "enotfound",
)
_LOGIN_WORDS = (
    "unauthorized", "authentication failed", "auth failed", "401", "403",
    "not logged in", "no longer logged in", "session expired", "token expired",
    "invalid credentials", "forbidden", "login failed", "re-login",
)


def _tool_scope(tool: str) -> Optional[str]:
    for name, spec in _scopes.SCOPES.items():
        if tool in (spec.get("tools") or []):
            return name
    return None


def _system_for_tool(tool: str) -> Optional[str]:
    parts = str(tool).split("__")
    server = parts[1] if len(parts) >= 3 else (parts[0] if parts else "")
    return _TOOL_SYSTEM_NAMES.get(server)


def _denied_tools(status: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in status.get("denied") or []:
        tool = item.get("tool") if isinstance(item, dict) else item
        if tool:
            out.append(str(tool))
    return out


def _denied_label(status: Dict[str, Any]) -> str:
    """The owner-facing verb phrase for whatever was refused, from the scope(s)
    that own the denied tool(s). Falls back to a vague phrase when the tool
    belongs to no known scope (nothing real to name)."""
    scope_names: List[str] = []
    for tool in _denied_tools(status):
        sc = _tool_scope(tool)
        if sc and sc not in scope_names:
            scope_names.append(sc)
    if scope_names:
        return schedprop.labels(scope_names)
    return "do part of this"


def _is_buffer_error(text: str) -> bool:
    low = text.lower()
    return "buffer size" in low or "exceeded maximum buffer" in low


def _is_timeout(text: str) -> bool:
    low = text.lower()
    return "timed out" in low or "timeout" in low


def _is_network_or_login(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in _NETWORK_WORDS) or any(w in low for w in _LOGIN_WORDS)


def _classify(status: Dict[str, Any]) -> str:
    """One category for a failed run: 'buffer', 'timeout', 'refusal', 'network',
    or 'other'. Order matters: a specific error signature (buffer, timeout) wins
    over a refusal or network guess even if denied/held also happen to be set."""
    text = f"{status.get('error') or ''} {status.get('final_status') or ''}"
    if _is_buffer_error(text):
        return "buffer"
    if _is_timeout(text):
        return "timeout"
    if _denied_tools(status) or status.get("held"):
        return "refusal"
    if _is_network_or_login(text):
        return "network"
    return "other"


def failure_in_words(status: Dict[str, Any]) -> str:
    """One plain clause for why a run failed. Never the raw error string."""
    category = _classify(status)
    if category == "buffer":
        return "it ran out of room handling something too large"
    if category == "timeout":
        return "it took too long"
    if category == "refusal":
        return f"it is not allowed to {_denied_label(status)}"
    if category == "network":
        for tool in reversed(status.get("tool_calls") or []):
            system = _system_for_tool(tool)
            if system:
                return f"it could not reach {system}"
        return "a service it needs did not respond"
    return "something broke on my side"


def next_step(status: Dict[str, Any]) -> str:
    """One next step the person can say to move the request forward."""
    if _classify(status) == "refusal":
        return f'Say "let it {_denied_label(status)}" and I will finish.'
    return 'Say "try again" and I will.'


def has_partial_findings(status: Dict[str, Any]) -> bool:
    """Whether the failed run left behind real narration worth offering, beyond
    the bare error itself."""
    final = str(status.get("final_status") or "").strip()
    return bool(final) and final != "(no status text)"


def failure_message(summary: str, status: Dict[str, Any]) -> str:
    """The whole message a person receives for a failed request: never
    'That failed.', never an exception or SDK string, never markdown."""
    msg = f"I couldn't finish {summary}: {failure_in_words(status)}. {next_step(status)}"
    if has_partial_findings(status):
        msg += ' I have part of it; say "send what you have".'
    return msg


def success_message(result: str) -> str:
    """The whole message for a finished request that succeeded: the result as
    is, without a leading 'Done.' when the result already reads as a sentence
    (starts capitalized, as the executor's own closing text normally does)."""
    text = (result or "").strip()
    if not text:
        return "Done."
    if text[0].isupper():
        return text
    return f"Done. {text}"



# ---------------------------------------------------------------------------
# A message arriving while a request is still running: "is it working?" gets a
# status line built here, in code, never a model call; anything else keeps the
# existing follow-up behaviour but with a fixed line instead of a promise.
# ---------------------------------------------------------------------------

_FIXED_PROGRESS_PHRASES = {
    "are you working on this", "are you working on it", "status", "how's it going",
    "hows it going", "did you start", "any update", "any updates",
}
# A verb that names a new ask: a question-mark message built around one of these is
# a new request, not a progress check, even without a fixed phrase match.
_NEW_TASK_VERBS = (
    "book", "schedule", "send", "email", "text", "call", "cancel", "find", "look up",
    "set up", "create", "add", "remove", "delete", "move", "update", "change", "buy",
    "order", "pay", "write", "draft", "check on",
)


def is_progress_question(body: str) -> bool:
    """Whether an inbound message, arriving while a request runs, is asking about
    progress rather than giving a new instruction. Code-only, on purpose: no model
    call decides this."""
    raw = (body or "").strip()
    if not raw:
        return False
    bare = raw.rstrip("?!. ").lower()
    if bare in _FIXED_PROGRESS_PHRASES:
        return True
    if raw.endswith("?") and not any(v in bare for v in _NEW_TASK_VERBS):
        return True
    return False


NOT_PROMISING_FOLLOWUP = "Noted. I will pick that up when this finishes."


def progress_status_line(started_at: float, latest_step: Optional[str], summary: str) -> str:
    """'Yes. Started <time>, <latest step>.' -- the latest step from the run's own
    recorded tool calls when any are available, else 'still working on <summary>'."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    started = datetime.fromtimestamp(started_at, ZoneInfo("America/New_York")).strftime("%-I:%M %p")
    step = latest_step or f"still working on {summary}"
    return f"Yes. Started {started}, {step}."


def latest_step_from_tool_calls(tool_calls: Optional[List[str]]) -> Optional[str]:
    """A plain phrase for the last tool call a running request made, if the run has
    recorded any yet (today, nothing streams tool calls into a running request's row,
    so this is normally None; the caller falls back to 'still working on <summary>')."""
    if not tool_calls:
        return None
    short = str(tool_calls[-1]).split("__")[-1].replace("_", " ").strip()
    return f"now working on {short}" if short else None



# ---------------------------------------------------------------------------
# "try again" / "send what you have", said about a request that just failed.
# Recognizing the words and building the reply text live here; re-creating the
# request itself is `GateSessionManager.retry_request` in manager.py (it needs
# the store), since this module stays a pure-text module with no store access.
# ---------------------------------------------------------------------------

_TRY_AGAIN_PHRASES = {"try again", "try it again", "retry", "please try again", "run it again"}
_SEND_WHAT_YOU_HAVE_PHRASES = {
    "send what you have", "send me what you have", "send it", "share what you have",
    "give me what you have",
}

RETRY_WINDOW_HOURS = 12


def is_try_again(body: str) -> bool:
    return (body or "").strip().rstrip("?!. ").lower() in _TRY_AGAIN_PHRASES


def is_send_what_you_have(body: str) -> bool:
    return (body or "").strip().rstrip("?!. ").lower() in _SEND_WHAT_YOU_HAVE_PHRASES


def within_retry_window(failed_at: Optional[float], now: float, hours: float = RETRY_WINDOW_HOURS) -> bool:
    """Whether a failure notice sent at `failed_at` is still within the window a
    "try again" or "send what you have" reply to it is honored."""
    if not failed_at:
        return False
    return 0 <= (now - failed_at) <= hours * 3600


def send_what_you_have_text(status: Dict[str, Any]) -> str:
    """The failed run's final status text, in full -- never summarized, never
    re-worded. The one place this module intentionally returns raw run text,
    because the person explicitly asked to see it."""
    text = str(status.get("final_status") or status.get("raw") or status.get("summary") or "").strip()
    return text or "There is nothing more to send; the run left no findings."
