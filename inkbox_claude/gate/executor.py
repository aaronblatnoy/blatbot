"""Deterministic executor: one approved prompt -> one Claude Code run.

Uses the Agent SDK one-shot query so the in-process Inkbox MCP server is
available, with allowed_tools fixed by scope and every other tool denied by
the hook. The project deny list in the exec workdir is enforced by Claude
Code underneath.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from .router import now_line
from .scopes import ORG_ACCOUNT, OWNER_ACCOUNT, SCOPES, blocks_requester_send, tools_for
from . import hosttools
from . import schedrun
from .store import Request, sha256

logger = logging.getLogger(__name__)


def vault_prefixes_for_scopes(scopes: List[str]) -> Optional[List[str]]:
    """The vault-relative path prefixes a run's granted scopes allow the vault
    tools to see; None means unrestricted. The whole-vault vault_read scope (the
    owner's usual grant when the second_brain system opens, and the fallback when
    the knowledge traversal did not run) always means None: everything. Anyone
    else's run with specific vault:* knowledge scopes and no vault_read is bounded
    to those scopes' own prefixes; a run with no vault scope at all is bounded to
    nothing, but in that case the vault tools are not in its allowed_tools either,
    so the restriction here is defence in depth, not the only gate."""
    if "vault_read" in (scopes or []):
        return None
    prefixes: List[str] = []
    for name in scopes or []:
        spec = SCOPES.get(name)
        if spec and spec.get("vault_paths"):
            for p in spec["vault_paths"]:
                if p not in prefixes:
                    prefixes.append(p)
    if not prefixes and not any(str(s).startswith("vault:") for s in (scopes or [])):
        # No vault scope of any kind was granted; vault tools were never handed to
        # this run, so the prefix list here is moot. Returning None would instead
        # open the whole vault to a run that was never meant to touch it at all.
        return []
    return prefixes

EXECUTOR_SYSTEM = (
    "You are running ONE approved task for Blatbot, Executive Assistant to Aaron Blatnoy. "
    "You are given the sender's message verbatim, the recent conversation, and the task's record. "
    "Read them and do what the message calls for using only the tools you have, then stop. Do not ask "
    "questions. Do not send anything the message and task record do not call for. If an essential "
    "detail is missing, stop and report what is missing. When something must be found in a document, sheet, "
    "calendar or inbox: search, open the most likely result, and if it is not the right one open the next "
    "likely ones or search again with different words before concluding it does not exist. "
    "The gateway delivers your status to the person "
    "who asked: NEVER text, iMessage or email the requester with the answer; put it in your final status.\n"
    f"Accounts: every tamid-drive tool takes user_google_email={ORG_ACCOUNT}; every stern-drive tool "
    f"takes user_google_email={OWNER_ACCOUNT} (Aaron's own calendar and mail). Never any other "
    "address. The Inkbox mailbox is blatbot@inkboxmail.com. Timezone America/New_York.\n"
    "Anything you write to another person (email, text, calendar invite) is from Blatbot, Aaron's "
    "assistant. Never mention Claude, Claude Code, Anthropic, AI tooling, prompts or how the task was run, "
    "and never sign as anything but Blatbot.\n"
    "If a tool asks for authorization or a browser, stop and report it as a failure.\n"
    "Finish with a short plain-text status of what you did (ids, times) or what failed and why, "
    "and make the very last line exactly STATUS: OK or STATUS: FAILED."
)

# The schedule system prompt replaces the ordinary closing-line rule with the
# three-line contract P1 parses: RESULT is the one sentence a notice is built
# from, NOTE carries forward only what the next run needs, STATUS adds STOP
# (the run's own request to end its schedule) to OK/FAILED.
SCHEDULE_EXECUTOR_SYSTEM = (
    "You are running ONE scheduled, unattended run for Blatbot, Executive Assistant to Aaron Blatnoy, inside "
    "an approved schedule. Do what the schedule's prompt calls for using only the tools you have, then stop. Do "
    "not ask questions; nobody is watching this run live. If an essential detail is missing, stop and report "
    "what is missing. When something must be found in a document, sheet, calendar or inbox: search, open the "
    "most likely result, and if it is not the right one open the next likely ones or search again with "
    "different words before concluding it does not exist. The gateway delivers your result to Aaron: NEVER "
    "text, iMessage or email him with it yourself unless the schedule's own prompt asks you to message him; "
    "put it in RESULT instead.\n"
    f"Accounts: every tamid-drive tool takes user_google_email={ORG_ACCOUNT}; every stern-drive tool "
    f"takes user_google_email={OWNER_ACCOUNT} (Aaron's own calendar and mail). Never any other address. "
    "The Inkbox mailbox is blatbot@inkboxmail.com. Timezone America/New_York.\n"
    "Anything you write to another person (email, text, calendar invite) is from Blatbot, Aaron's assistant. "
    "Never mention Claude, Claude Code, Anthropic, AI tooling, prompts or how the task was run, and never sign "
    "as anything but Blatbot.\n"
    "If a tool asks for authorization or a browser, stop and report it as a failure.\n"
    "End with exactly these closing lines, in this order, and nothing after them:\n"
    "RESULT: <one plain sentence for Aaron, no markdown, no lists, no bold>\n"
    "NOTE: <what the next run of this schedule needs to know, or omit this line entirely>\n"
    "STATUS: OK\n"
    "CHANGED: yes or CHANGED: no\n"
    "Use STATUS: FAILED instead of STATUS: OK when the run could not finish what it was asked. Use "
    "STATUS: STOP instead when this schedule should not run again (its job is done for good, or it asked to "
    "end itself); STOP still needs a RESULT line saying why. CHANGED is yes when something Aaron asked this "
    "schedule to watch has changed since the previous run's NOTE, or when this is the first run ever; "
    "CHANGED is no otherwise. CHANGED is always the last line."
)

def _norm_handle(v: str) -> str:
    """Same normalization the gate already uses to recognize the owner's own
    phone/iMessage/Telegram handle (scopes._norm): digits-only for anything that is
    not an email, last 10 digits for a phone, case-folded otherwise."""
    v = (v or "").strip().lower()
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits[-10:] if digits and "@" not in v and len(digits) >= 7 else v


def _is_owner_handle(value: str, protected: List[str]) -> bool:
    """True when  (a recipient a scheduled run would send to) is one of the
    handles the gate already treats as Aaron's own: his iMessage conversation id, his
    phone, his Telegram id -- exactly the  list the executor already holds,
    nothing newly invented here (a bare email is never treated as his unless the gate
    already does so elsewhere)."""
    norm = _norm_handle(value)
    if not norm:
        return False
    return norm in {_norm_handle(p) for p in protected if p}


ORDINARY_DESTRUCTIVE_DENIAL = (
    "Deleting, cancelling or replacing anything requires Aaron's confirmation, which only the gateway can ask "
    "for. Report exactly what you would change (name, time, id) in your status and stop."
)


def destructive_policy(req: Request, tool_name: str, input_data: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], str]:
    if not req.schedule_id:
        return None, ORDINARY_DESTRUCTIVE_DENIAL
    return ({"tool": tool_name, "args": dict(input_data or {})},
            "Deleting, cancelling or replacing anything requires Aaron's confirmation, "
            "which the gateway will ask for. Stop now.")


class Executor:
    def __init__(self, *, mcp_server: Any, cwd: str, model: str = "sonnet", timeout_s: float = 600.0,
                 protected: Optional[List[str]] = None):
        self.mcp_server = mcp_server
        self.cwd = cwd
        self.model = model
        self.timeout_s = timeout_s
        # Addresses/conversations the gateway itself replies to (the owner's).
        self.protected = list(protected or [])

    allow_send = False        # set for one run, after the owner has read the draft and said yes
    held_send: Optional[Dict[str, Any]] = None   # the call that was stopped, for the gate to show him

    _sessions: Dict[str, str] = {}      # thread -> the Claude session that thread is having

    async def run(self, req: Request, context: str = "", prior_work: str = "", model: Optional[str] = None,
                  from_owner: bool = False, timezone: str = "America/New_York",
                  note: Optional[str] = None) -> Dict[str, Any]:
        """Run an approved request. Returns a status dict; never raises.

        `context` is the task ledger for the person this request concerns. It is
        appended to the system prompt as read-only background; the task text
        itself (req.prompt) is what was approved and is hash-checked unchanged.

        `timezone` and `note` matter only for a scheduled run (req.schedule_id):
        `timezone` is the schedule's own time zone, used for the run's start-time
        line; `note` is the previous run's note, carried forward once. Neither is
        used, and no task ledger is added, for an ordinary request."""
        if sha256(req.prompt) != req.prompt_sha256:
            return {"ok": False, "error": "prompt hash mismatch; refused to run", "tool_calls": []}
        started_at = time.time()
        vault_prefixes = vault_prefixes_for_scopes(req.scopes)
        allowed = tools_for(req.scopes)
        allowed_set = set(allowed)   # what this task may touch at all, send tools included
        if req.schedule_id:
            # Every scheduled write reaches the hook. The schedule pre-approves ordinary
            # writes and sends, while delete-like calls are parked for Aaron's yes.
            from .jevagent import is_write_tool as _is_write
            allowed = [tool for tool in allowed if not _is_write(tool)]
        tool_calls: List[str] = []
        tool_use_names: Dict[str, str] = {}
        tool_results: List[Dict[str, Any]] = []
        texts: List[str] = []
        denied_steps: List[Dict[str, Any]] = []
        # The key this run's Claude Code session resumes under. A scheduled run never
        # resumes any session at all -- not the owner's thread, not even its own prior
        # runs' session -- and its own session id is never stored for a later run to
        # pick up: every run of a schedule starts cold, with only the carried `note`
        # (not a reused conversation) linking it to the last one. An ordinary request
        # still resumes its thread's session as before.
        session_key = None if req.schedule_id else req.chat_id

        self.held_send = None
        held_destructive: Optional[Dict[str, Any]] = None
        # A message the owner asked to have sent is him speaking to someone through the
        # assistant, and he sees it before it goes. A request that came from someone else
        # was already read and approved by him in full, so answering it needs nothing more.
        from .settings import get as _setting
        confirm_sends = bool(from_owner) and bool(_setting("GATE_CONFIRM_SENDS"))
        # A scheduled run's sends are routed through the hook too, but never on a blanket
        # from_owner basis: can_use below decides each one by its own recipients, letting a
        # send to Aaron himself through and holding anything to a third party (or anything
        # whose recipients cannot be told at all).
        if confirm_sends or req.schedule_id:
            # A tool listed in allowed_tools is pre-approved and the permission hook is never
            # consulted for it. Sends are therefore taken off that list, which is what routes
            # them through the hook so they can be held and shown to him first.
            from .jevagent import is_outbound_message as _out
            allowed = [t for t in allowed if not _out(t, {})]

        async def can_use(tool_name: str, input_data: Dict[str, Any], context: Any):
            if tool_name not in allowed_set:
                denied_steps.append({"tool": tool_name, "args": dict(input_data or {}), "ok": False, "denied": "scope"})
                return PermissionResultDeny(message=f"{tool_name} is outside this task's scope; do not retry it.")
            from .jevagent import is_destructive, is_outbound_message, recipients_of
            if is_outbound_message(tool_name, input_data or {}):
                if req.schedule_id:
                    recips = recipients_of(input_data or {})
                    if recips and all(_is_owner_handle(r, self.protected) for r in recips):
                        pass   # every recipient is Aaron himself; the schedule's own job, not a hold
                    else:
                        who = ", ".join(recips) or "(no recipient in the call)"
                        self.held_send = {"tool": tool_name, "args": dict(input_data or {}), "to": who}
                        return PermissionResultDeny(message=(
                            "A scheduled run only sends on its own to Aaron himself. This would have "
                            f"reached: {who}. Do not try another tool or another route to send it. Put the "
                            "whole thing in your final status, exactly as you would send it: every "
                            "recipient, every cc, the subject, and the full text, then stop."))
                elif confirm_sends and not self.allow_send:
                    who = ", ".join(recipients_of(input_data or {})) or "(no recipient in the call)"
                    self.held_send = {"tool": tool_name, "args": dict(input_data or {}), "to": who}
                    return PermissionResultDeny(message=(
                        "Nothing goes out to another person without Aaron seeing it first, including when he "
                        f"asked for it. This would have reached: {who}. Do not try another tool or another "
                        "route to send it. Put the whole thing in your final status, exactly as you would send "
                        "it: every recipient, every cc, the subject, and the full text, then stop."))
            if is_destructive(tool_name, input_data or {}):
                nonlocal held_destructive
                held_destructive, message = destructive_policy(req, tool_name, input_data or {})
                return PermissionResultDeny(message=message)
            if blocks_requester_send(req, tool_name, input_data or {}, self.protected):
                # Enforced, not just prompted: the gateway delivers the result to whoever asked.
                return PermissionResultDeny(message="Do not message the requester; the gateway delivers your "
                                                    "result. Put the answer in your final status and stop.")
            return PermissionResultAllow()

        if req.schedule_id:
            system_append = SCHEDULE_EXECUTOR_SYSTEM + f"\nNow: {now_line()}. Resolve 'today', 'tomorrow', weekday names and relative dates from this.\n"
            prompt_text = schedrun.build_prompt(req.prompt, started_at, timezone, note)
        else:
            system_append = EXECUTOR_SYSTEM + f"\nNow: {now_line()}. Resolve 'today', 'tomorrow', weekday names and relative dates from this.\n"
            prompt_text = req.prompt
        if context.strip():
            system_append += (
                "\n\nTASK LEDGER (read-only background on this person; the task text below is what to do):\n"
                + context.strip()
            )
        if prior_work.strip():
            # The system prompt travels as a command-line argument to the CLI, so large
            # findings go to a file Claude reads; only a short head rides in the prompt.
            findings_dir = os.path.join(self.cwd, "findings")
            os.makedirs(findings_dir, exist_ok=True)
            path = os.path.join(findings_dir, f"request-{req.id}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(prior_work)
            head = prior_work.strip()
            head = head if len(head) <= 6000 else head[:6000] + "\n... (continues in the file)"
            system_append += (
                "\n\nWORK ALREADY DONE on this task by a faster agent before you (its tool calls and their FULL "
                f"results; it stopped because it could not decide the next step). The complete record is in {path}; "
                "Read it before calling any tool. Do not repeat those calls. Continue from them. Head of the record:\n" + head
            )
        options = ClaudeAgentOptions(
            cwd=self.cwd,
            model=model or self.model,
            system_prompt={"type": "preset", "preset": "claude_code", "append": system_append},
            setting_sources=["user", "project"],
            permission_mode="default",
            allowed_tools=allowed,
            mcp_servers={"inkbox": self.mcp_server, "host": hosttools.sdk_server(vault_prefixes=vault_prefixes)},
            # One conversation per thread, continued rather than restarted, for an ordinary
            # request: what was looked up an hour ago is still in view and a follow-up does
            # not arrive as a stranger's first sentence. A scheduled run never resumes:
            # session_key is None for it, so resume is always None too.
            resume=(self._sessions.get(session_key) if session_key is not None else None),
            can_use_tool=can_use,
            max_turns=30,
        )

        closing_texts: List[str] = []

        async def _go():
            final: Optional[Any] = None
            # Consume the generator to completion; returning from inside the
            # loop leaves the SDK's async generator half-closed.
            async for msg in query(prompt=prompt_text, options=options):
                if isinstance(msg, AssistantMessage):
                    saw_tool_use = False
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            texts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(block.name)
                            tool_use_names[block.id] = block.name
                            saw_tool_use = True
                    if saw_tool_use:
                        # A text block landed in the same turn as a tool call; it is
                        # narration, not the closing status. Only text after the LAST
                        # tool call in the whole run is the run's final status.
                        closing_texts.clear()
                    else:
                        closing_texts.extend(block.text for block in msg.content if isinstance(block, TextBlock))
                elif isinstance(msg, UserMessage):
                    # Only a call that actually ran produces a ToolResultBlock; a denied
                    # or held call never reaches here, so tool_results never includes one.
                    content = msg.content if isinstance(msg.content, list) else []
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            name = tool_use_names.get(block.tool_use_id)
                            if name:
                                tool_results.append({"tool": name, "ok": not bool(block.is_error)})
                elif isinstance(msg, ResultMessage):
                    sid = getattr(msg, "session_id", None)
                    if sid and session_key is not None:
                        self._sessions[session_key] = sid
                    final = msg
            return final

        try:
            result = await asyncio.wait_for(_go(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            finished_at = time.time()
            return {"ok": False, "error": f"timed out after {int(self.timeout_s)}s", "tool_calls": tool_calls,
                    "raw": "\n".join(texts), "stop": False, "result_line": "", "final_status": "",
                    "note": "", "denied": [{"tool": d["tool"], "reason": d.get("denied", "scope")}
                                           for d in denied_steps],
                    "held": bool(self.held_send or held_destructive), "tool_results": tool_results,
                    "started_at": started_at, "finished_at": finished_at, "cost_usd": None}
        except Exception as exc:  # noqa: BLE001
            logger.exception("executor failed for request %s", req.id)
            finished_at = time.time()
            return {"ok": False, "error": str(exc), "tool_calls": tool_calls, "raw": "\n".join(texts),
                    "stop": False, "result_line": "", "final_status": "", "note": "",
                    "denied": [{"tool": d["tool"], "reason": d.get("denied", "scope")} for d in denied_steps],
                    "held": bool(self.held_send or held_destructive), "tool_results": tool_results,
                    "started_at": started_at, "finished_at": finished_at, "cost_usd": None}

        finished_at = time.time()
        raw = "\n".join(texts).strip()
        is_error = bool(getattr(result, "is_error", False)) if result is not None else False
        closing_full = "\n".join(closing_texts).strip()
        parsed = schedrun.parse_closing(closing_full)
        if parsed["status"] is not None:
            ok = parsed["status"] in ("OK", "STOP")
            is_error = is_error or parsed["status"] == "FAILED"
        else:
            # No RESULT/NOTE/STATUS contract matched (the ordinary, non-schedule system
            # prompt just asks for a last line of STATUS: OK or STATUS: FAILED): fall
            # back to the legacy check on the raw closing text.
            last = raw.strip().splitlines()[-1].strip().upper() if raw.strip() else ""
            if last.startswith("STATUS:"):
                is_error = is_error or ("FAILED" in last)
            elif raw.lower().startswith("task failed") or "STATUS: FAILED" in raw.upper():
                is_error = True
            ok = not is_error
        # final_status is the run's own closing status -- the text that arrived after
        # the last tool call, with the RESULT:/NOTE:/STATUS: lines stripped. raw (the
        # full step-by-step narration) stays available for the console and the record,
        # but a notice to the owner is built from final_status, never raw.
        final_status = parsed["final_status"] if closing_full else (raw if raw else "(no status text)")
        out = {
            "ok": ok,
            "stop": bool(parsed["stop"]),
            "result_line": parsed["result_line"],
            "note": parsed["note"],
            "changed": parsed.get("changed"),
            "engine": f"claude:{model or self.model}",
            "summary": raw if raw else "(no status text)",
            "final_status": final_status,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "denied": [{"tool": d["tool"], "reason": d.get("denied", "scope")} for d in denied_steps],
            "held": bool(self.held_send or held_destructive),
            "raw": raw,
            "steps": denied_steps,
            "started_at": started_at,
            "finished_at": finished_at,
            "cost_usd": getattr(result, "total_cost_usd", None) if result is not None else None,
        }
        if self.held_send is not None:
            # Something was going to be said to someone else. It is not sent, it is shown.
            out["confirm"] = {"tool": self.held_send["tool"], "args": self.held_send["args"],
                              "about": [f"It would go to: {self.held_send['to']}"], "kind": "send"}
            out["ok"] = False
        elif held_destructive is not None:
            out["confirm"] = {"tool": held_destructive["tool"], "args": held_destructive["args"],
                              "about": [raw or req.original_message], "kind": "destructive"}
            out["ok"] = False
        return out
