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

import re

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
_USAGE_LIMIT_WORDS = (
    "429", "too many requests", "session limit", "weekly limit", "usage limit", "rate limit", "quota", "overloaded", "credit balance",
)
_RESET_TIME_RE = re.compile(
    r"resets?\s+((?:[A-Za-z]{3}\s+\d{1,2},?\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?(?:\s*\([^)]*\))?)", re.I)


def _is_usage_limit(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in _USAGE_LIMIT_WORDS) or "resets " in low


def _reset_time(text: str) -> Optional[str]:
    """The reset time exactly as the error gave it, e.g. "3:10pm (America/New_York)",
    or None when the error names no reset time at all."""
    m = _RESET_TIME_RE.search(text or "")
    return m.group(1).strip().rstrip(".") if m else None


def _tool_scope(tool: str) -> Optional[str]:
    return _scopes.scope_for_tool(tool)


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
        # The short, re-sayable form: "act in Brightspace", never the parenthetical
        # detail ("...submit, post, mark complete)") -- a person asked to say this
        # phrase back ("let it <label>") must be able to say it whole.
        return schedprop.short_labels(scope_names)
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


def is_usage_limit_failure(status: Dict[str, Any]) -> bool:
    """Whether a failed run's status classifies as a usage/session/rate-limit,
    quota, overload or exhausted-credit failure -- public so a caller deciding
    whether to suppress a repeat notice (item 7's "told once, not once per
    ask") does not need to reach into _classify."""
    return _classify(status) == "usage_limit"


def _classify(status: Dict[str, Any]) -> str:
    """One category for a failed run: 'usage_limit', 'buffer', 'timeout', 'refusal',
    'network', or 'other'. Order matters: usage_limit is checked first (an exhausted
    session has nothing to do with a buffer or a denied tool, even if the error text
    happens to also mention one), then a specific error signature (buffer, timeout)
    wins over a refusal or network guess even if denied/held also happen to be set."""
    text = f"{status.get('error') or ''} {status.get('final_status') or ''}"
    if _is_usage_limit(text):
        return "usage_limit"
    if _is_buffer_error(text):
        return "buffer"
    if _is_timeout(text):
        return "timeout"
    if _denied_tools(status) or status.get("held"):
        return "refusal"
    if _is_network_or_login(text):
        return "network"
    return "other"


def failure_category(status: Dict[str, Any]) -> str:
    """Public name for _classify: one category for a failed run -- 'usage_limit',
    'buffer', 'timeout', 'refusal', 'network', or 'other'. Shared by every place
    that needs to tell a usage-limit failure apart from the rest (overview.py's
    Stuck collapsing, in particular), so there is exactly one classifier."""
    return _classify(status)


def usage_limit_reset_text(status: Dict[str, Any]) -> Optional[str]:
    """The reset time exactly as a usage-limit error gave it (e.g. "3:10pm
    (America/New_York)"), or None when the error named no reset time."""
    if status.get("reset_time"):
        return str(status["reset_time"])
    text = f"{status.get('error') or ''} {status.get('final_status') or ''}"
    return _reset_time(text)


def failure_in_words(status: Dict[str, Any]) -> str:
    """One plain clause for why a run failed. Never the raw error string."""
    category = _classify(status)
    if category == "usage_limit":
        reset = usage_limit_reset_text(status)
        if reset:
            return f"it had hit a usage limit until {reset}"
        return "it had hit a usage limit"
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


def run_own_explanation(status: Dict[str, Any]) -> str:
    """What a run said for itself when it stopped on its own: no exception, no
    refused tool, no limit -- it ran, could not go on, and explained. Its closing
    text, whole, with markdown marks and the STATUS line removed. "" otherwise."""
    if status.get("error") or _classify(status) != "other":
        return ""
    final = str(status.get("final_status") or "").strip()
    if not final or final == "(no status text)":
        return ""
    lines = [ln for ln in final.splitlines() if not ln.strip().upper().startswith("STATUS:")]
    text = "\n".join(lines).strip()
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def failure_message(summary: str, status: Dict[str, Any]) -> str:
    """The whole message a person receives for a failed request: never
    'That failed.', never an exception or SDK string, never markdown."""
    if _classify(status) == "usage_limit":
        # Its own case, entirely: no "I couldn't finish <summary>" framing, no
        # partial-findings offer, and never "try again" as an immediate fix (the
        # session that would run it is the thing that is out).
        text = f"{status.get('error') or ''} {status.get('final_status') or ''}"
        reset = _reset_time(text)
        if reset:
            clause = f"I've hit a usage limit and can't run anything until {reset}."
        else:
            clause = "I've hit a usage limit and can't run anything right now."
        return f"{clause} I'll need you to try again after that."
    own = run_own_explanation(status)
    if own:
        # The run itself stopped and said why (a name it could not match, a detail
        # it needs). That account is the answer; "something broke" would hide it.
        return own
    msg = f"I couldn't finish {summary}: {failure_in_words(status)}. {next_step(status)}"
    if has_partial_findings(status):
        msg += ' I have part of it; say "send what you have".'
    return msg


def same_limit_message(status: Dict[str, Any]) -> str:
    """Item C8: the one short line said or sent when a second, different ask fails
    on the same usage-limit outage a thread was already told about once within the
    suppression window. Not nothing (the old behavior), and not the full
    failure_message again -- a short pointer back to the same limit, with the reset
    time when the status gives one."""
    reset = usage_limit_reset_text(status)
    if reset:
        return f"Same limit. I can't run that until {reset}."
    return "Same limit. I can't run that right now."


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


def is_fixed_progress_phrase(body: str) -> bool:
    """Whether `body` is one of the fixed, unambiguous progress phrases ("status",
    "any update?", ...) -- never the looser "ends in a question mark" guess
    is_progress_question also makes. Used once nothing is running any more: after
    a request has already finished, a short question-shaped message is just as
    likely to be a genuinely new ask ("when is it?", "actually 7?") as a check-in,
    so only an explicit, fixed phrase is read as "about the thing that just
    finished" -- the loose guess stays scoped to the one place it is safe: while
    a request is actually running and nothing else could be meant."""
    raw = (body or "").strip()
    if not raw:
        return False
    return raw.rstrip("?!. ").lower() in _FIXED_PROGRESS_PHRASES


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
    "send what you have", "send me what you have", "share what you have",
    "give me what you have",
}

RETRY_WINDOW_HOURS = 12


def is_try_again(body: str) -> bool:
    return (body or "").strip().rstrip("?!. ").lower() in _TRY_AGAIN_PHRASES


def is_send_what_you_have(body: str) -> bool:
    return (body or "").strip().rstrip("?!. ").lower() in _SEND_WHAT_YOU_HAVE_PHRASES


# A small fixed list, whole-message match: the owner granting blanket permission
# for whatever the currently failed/running request needs, never a standing grant
# (see GateSessionManager._handle_blanket_permission in manager.py).
_EVERYTHING_PHRASES = {
    "you have my permission for everything",
    "you have my personal permission for everything",
    "you have permission for everything",
    "do whatever you need",
    "you're allowed to do all of it",
    "you are allowed to do all of it",
    "full permission",
    "go ahead with everything",
}


def is_blanket_permission(body: str) -> bool:
    return (body or "").strip().rstrip("?!. ").lower().replace("’", "'") in _EVERYTHING_PHRASES


def denied_scopes(status: Dict[str, Any]) -> List[str]:
    """Scope names behind each tool a failed run's status denied, de-duplicated --
    what "let it <label>" (item 3) and a blanket grant (item 4) narrow to when
    nothing more specific is named."""
    out: List[str] = []
    for tool in _denied_tools(status):
        sc = _tool_scope(tool)
        if sc and sc not in out:
            out.append(sc)
    return out


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


# ---------------------------------------------------------------------------
# Naming a request to the owner, and building a plain fallback title/summary
# for one, from a raw inbound message or an already-created request.
#
# A live phone call hands a request off with its whole transcript as
# original_message (build_delegation_query in live.py) -- a preamble, prior
# turns for context, then "What was just said (act on this):" followed by the
# caller's and Blatbot's own lines. Nothing here ever shows that preamble, a
# "blatbot:" line, a phone number, or text force-cut by length: when there is
# no real summary yet, the caller's own words stand in for one.
# ---------------------------------------------------------------------------

_HANDOFF_PREFIXES = ("live phone call with",)
_WHAT_WAS_SAID_MARKER = "What was just said (act on this):"
_CALLER_PREFIX = "caller:"


def _is_handoff_shaped(summary: str) -> bool:
    return (summary or "").strip().lower().startswith(_HANDOFF_PREFIXES)


def _is_cut_of(summary: str, original: str) -> bool:
    """Whether `summary` looks like a hard character slice of `original` rather
    than a real summary someone wrote: a strict, shorter prefix that does not
    end on a sentence boundary and stops mid-word (the next character in the
    source is itself a word character). A short summary that genuinely ends a
    sentence, or that just happens to be a prefix followed by a space or
    punctuation, is never flagged."""
    summary = (summary or "").strip()
    original = (original or "").strip()
    if not summary or not original or summary == original or len(summary) >= len(original):
        return False
    if not original.startswith(summary):
        return False
    if summary[-1:] in ".!?":
        return False
    next_char = original[len(summary):len(summary) + 1]
    return next_char.isalnum() or next_char == "'"


def caller_lines(transcript: str) -> List[str]:
    """The caller's own words from a voice hand-off transcript: every "caller:"
    line under "What was just said (act on this):", or every "caller:" line in
    the whole transcript when that section is missing. Never a "blatbot:" line,
    never the preamble above it."""
    text = transcript or ""
    if _WHAT_WAS_SAID_MARKER in text:
        text = text.split(_WHAT_WAS_SAID_MARKER, 1)[1]
    out: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith(_CALLER_PREFIX):
            said = line[len(_CALLER_PREFIX):].strip()
            if said:
                out.append(said)
    return out


def fallback_title_from_message(message: str) -> str:
    """A plain task/request title built in code from a raw inbound message,
    never cut by length: the caller's own words when the message is a voice
    hand-off transcript (see caller_lines), else the message itself,
    whitespace-collapsed and whole."""
    text = " ".join((message or "").split())
    if _is_handoff_shaped(text):
        lines = caller_lines(message)
        if lines:
            return " ".join(lines)
        return "a phone call"
    return text or "Untitled task"


def looks_like_fallback_title(title: Optional[str], message: str) -> bool:
    """Whether `title` is still exactly the code-built fallback above (never a
    real title a person or the router wrote), so a better title found later is
    free to replace it."""
    if title in (None, "", "Untitled task"):
        return True
    return title == fallback_title_from_message(message)


def _call_time_text(created_at: Any, timezone_name: str = "America/New_York") -> str:
    """"10:12 AM" for a call today, "Oct 13, 10:12 AM" otherwise; "" when unknown."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone_name)
        at = datetime.fromtimestamp(float(created_at), tz)
    except (TypeError, ValueError, OSError):
        return ""
    clock = at.strftime("%I:%M %p").lstrip("0")
    if at.date() == datetime.now(tz).date():
        return clock
    return f"{at.strftime('%b')} {at.day}, {clock}"


_MAX_TITLE_LEN = 80


def is_call_request(req: Any) -> bool:
    """Whether `req` came from a phone call: its mode is voice, or its
    original message is a call hand-off (see _is_handoff_shaped). Either
    one is enough -- a call request is never named by its summary, whatever
    that summary says, because the summary of a call is often a speech
    transcript."""
    mode = str(getattr(req, "mode", "") or "")
    original = (getattr(req, "original_message", "") or "").strip()
    return mode == "voice" or _is_handoff_shaped(original) or _is_handoff_shaped(
        (getattr(req, "summary", "") or "").strip())


def _call_title(req: Any) -> str:
    when = _call_time_text(getattr(req, "created_at", None))
    return f"the request from the {when} call" if when else "the request from a call"


def _non_call_title(summary: str, created_at: Any) -> str:
    """A plain title for a non-call request's summary: the summary itself
    when it is short and a single sentence; otherwise its first sentence
    alone, if that sentence is 80 characters or fewer; otherwise a request
    is named by when it was asked rather than cut mid-sentence."""
    text = summary.strip()
    sentences = [s for s in _SENTENCE_PIECE_RE.split(text) if s.strip()]
    one_sentence = len(sentences) <= 1
    if len(text) <= _MAX_TITLE_LEN and one_sentence:
        return text
    first = (sentences[0].strip() if sentences else text)
    if first and len(first) <= _MAX_TITLE_LEN:
        return first
    when = _call_time_text(created_at)
    return f"a request from {when}" if when else "a request"


_SENTENCE_PIECE_RE = re.compile(r"(?<=[.!?])\s+")


def call_time_text(created_at: Any, timezone_name: str = "America/New_York") -> str:
    """Public wrapper for _call_time_text: "10:12 AM" for today, "Oct 13,
    10:12 AM" otherwise, "" when unknown. Used wherever a call is named by
    when it happened outside of a single request's own title (overview.py's
    usage-limit Stuck/Done-today collapsing, in particular)."""
    return _call_time_text(created_at, timezone_name)


def plain_title(store: Any, req: Any) -> str:
    """The request named to the owner, anywhere a request is named (overview
    rows, failure/success messages, the progress reply): a call request is
    always named by when it was asked (see is_call_request), whatever its
    summary says, because a speech transcript is never a title. A non-call
    request is named by its own summary when that already reads as plain
    text (see _non_call_title); a long or multi-sentence summary is named
    by its first sentence alone, or by when it was asked when even that
    sentence runs long -- never cut mid-sentence.

    `store` is accepted for parity with the rest of the console's
    contact-aware wording (and in case a future hand-off shape ever needs a
    contact lookup); plain_title itself never needs one; it only ever reads
    the caller's own spoken words or the request's own summary."""
    if is_call_request(req):
        return _call_title(req)
    summary = (getattr(req, "summary", "") or "").strip()
    original = (getattr(req, "original_message", "") or "").strip()
    if summary and not _is_cut_of(summary, original):
        return _non_call_title(summary, getattr(req, "created_at", None))
    # No usable summary at all: fall back to the call-shaped naming so a
    # request with nothing better named still gets a plain, whole title.
    return _call_title(req)


_CHANGING_TOOL_RE = re.compile(r"(^|_)(create|update|delete|manage|remove|write)(_|$)")


def truly_read_only(scope: str) -> bool:
    """A scope the registry marks `read` whose tools also cannot change anything.
    Some `read` scopes carry a create/update/manage tool (contacts, the calendars);
    a blanket "permission for everything" never adds those on its own."""
    from .scopes import READ_SCOPES, SCOPES
    if scope not in READ_SCOPES:
        return False
    tools = [t.rsplit("__", 1)[-1] for t in (SCOPES.get(scope) or {}).get("tools") or []]
    return not any(_CHANGING_TOOL_RE.search(t) for t in tools)
