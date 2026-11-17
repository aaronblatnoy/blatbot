"""One decision for every Claude Code tool call, including pre-allowed tools."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .jevagent import is_destructive, is_outbound_message, recipients_of
from .scopes import blocks_requester_send, scope_for_tool
from . import hosttools
from . import outcomes


# Claude Code's own bookkeeping, not an action on anything: ToolSearch only loads the
# description of a tool the run may then try to call, and that call is decided here like
# any other. Refusing it leaves a run unable to see the tools it was given.
HARNESS_TOOLS = frozenset({"ToolSearch"})


def decide(req: Any, name: str, args: Dict[str, Any], allowed: Iterable[str], *,
           from_owner: bool, protected: list[str], manager: Any,
           confirm_sends: bool, owner_handle: Any) -> tuple[str, str, str]:
    """Return (allow, deny, or defer), a plain reason, and the hold kind."""
    allowed_set = set(allowed)
    if name in HARNESS_TOOLS:
        return "allow", "", ""
    if name.startswith("mcp__host__admin_") and not from_owner:
        return "deny", "Only Aaron, as the verified commander, can change Blatbot's own settings.", "commander_only"
    if name not in allowed_set:
        scope = scope_for_tool(name)
        refused = set((req.status or {}).get("access_refused") or [])
        if scope and scope in refused:
            return "deny", "Aaron already refused that access for this task. Finish with what you have.", "scope"
        if scope and "mcp__host__request_access" in allowed_set:
            return ("deny", f'{name} needs {scope}, which is outside this task\'s scope. '
                    f'Call request_access with access ["{scope}"] and one sentence saying why. '
                    "Then stop and say you are waiting. Do not try another route to the same data.", "scope")
        return "deny", f"{name} is outside this task's scope; do not retry it.", "scope"
    if name in (outcomes.QUIET_TOOL, outcomes.REACH_TOOL):
        # Ending quietly and looking up where someone can be reached touch nothing.
        return "allow", "", ""
    if name == outcomes.ASK_TOOL:
        status = req.status or {}
        if not outcomes.ask_enabled():
            return "deny", "Asking Aaron is off. Finish with what you have and say plainly what is missing.", "ask"
        if int(status.get("ask_count") or 0) >= outcomes.ASK_LIMIT:
            return "deny", "You have already asked Aaron twice for this task. Finish with what you have.", "ask"
        if not str(args.get("question") or "").strip():
            return "deny", "Give the question in one or two plain sentences.", "ask"
        if args.get("kind") not in ("permission", "input"):
            return "deny", 'kind must be "permission" (a yes or no) or "input" (a detail or a choice).', "ask"
        # Always deferred: the run ends here with its session saved, Aaron is texted the
        # question, and his answer resumes this exact call (manager.handle_ask).
        return "defer", "Asking Aaron.", "ask"
    if name == outcomes.SEND_TOOL:
        try:
            dest = outcomes.resolve(manager, req, args.get("to"), args.get("surface"))
        except ValueError:
            # The tool itself says what was wrong with the address; nothing is sent.
            return "allow", "", ""
        if dest.get("requester") or dest.get("owner"):
            # Answering the person who asked, or telling Aaron himself, needs no sign-off.
            return "allow", "", ""
        if req.schedule_id or confirm_sends:
            return "defer", f"Ask Aaron before sending to {outcomes.describe(dest)}.", "send"
        return "allow", "", ""
    if blocks_requester_send(req, name, args, protected):
        if outcomes.SEND_TOOL in allowed_set:
            # One way to answer the person who asked, so it is recorded as the answer and
            # the gateway does not answer them a second time.
            return ("deny", 'Answer the person who asked with send_message, to "requester". It goes '
                    "back on the surface they wrote from.", "requester")
        return "deny", "Do not message the requester; the gateway delivers your result. Put the answer in your final status and stop.", "requester"
    if name == "mcp__host__request_access":
        if int((req.status or {}).get("access_ask_count") or 0) >= 2:
            return "deny", "You have already asked twice; finish with what you have.", "access"
        choices = hosttools.access_choices(req)
        names = args.get("access")
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            return "deny", "Give one or more exact access names.", "access"
        if any(n in (req.status or {}).get("access_refused", []) for n in names):
            return "deny", "Aaron already refused that access for this task. Finish with what you have.", "access"
        if any(n not in choices for n in names):
            return "deny", "Unknown access name. Finish with what you have.", "access"
        if not str(args.get("why") or "").strip():
            return "deny", "Say in one sentence why this task needs that access.", "access"
        # An access ask always ends the run here. What this person's role and their own
        # grants already cover is given at once and the same session continues with those
        # tools loaded; only what is left over goes to Aaron (manager.handle_access_ask).
        # Granting inside the run would hand over tools whose servers this run never loaded.
        return "defer", "Checking this access before continuing.", "access"
    if is_outbound_message(name, args):
        recips = recipients_of(args)
        owner_only = bool(recips) and all(owner_handle(r, protected) for r in recips)
        if (req.schedule_id or confirm_sends) and not owner_only:
            who = ", ".join(recips) or "(no recipient in the call)"
            return "defer", f"Ask Aaron before sending to {who}.", "send"
    if is_destructive(name, args):
        return "defer", "Ask Aaron before deleting, cancelling or replacing anything.", "destructive"
    return "allow", "", ""


def hook_result(decision: str, reason: str) -> Dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": decision,
                                   "permissionDecisionReason": reason}}
