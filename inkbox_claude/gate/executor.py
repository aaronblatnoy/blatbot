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
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from .router import now_line
from .scopes import ORG_ACCOUNT, OWNER_ACCOUNT, sends_to_requester, tools_for
from . import hosttools
from .store import Request, sha256

logger = logging.getLogger(__name__)

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
                  from_owner: bool = False) -> Dict[str, Any]:
        """Run an approved request. Returns a status dict; never raises.

        `context` is the task ledger for the person this request concerns. It is
        appended to the system prompt as read-only background; the task text
        itself (req.prompt) is what was approved and is hash-checked unchanged."""
        if sha256(req.prompt) != req.prompt_sha256:
            return {"ok": False, "error": "prompt hash mismatch; refused to run", "tool_calls": []}
        allowed = tools_for(req.scopes)
        if req.schedule_kind == "continue":
            allowed.append("mcp__host__schedule_continue")
        allowed_set = set(allowed)   # what this task may touch at all, send tools included
        if req.schedule_id:
            # Every scheduled write reaches the hook. The schedule pre-approves ordinary
            # writes and sends, while delete-like calls are parked for Aaron's yes.
            from .jevagent import is_write_tool as _is_write
            allowed = [tool for tool in allowed if not _is_write(tool)]
        tool_calls: List[str] = []
        texts: List[str] = []

        protected = self.protected + [req.sender, req.chat_id]
        self.held_send = None
        held_destructive: Optional[Dict[str, Any]] = None
        continue_state: Optional[Dict[str, Any]] = {} if req.schedule_kind == "continue" else None
        # A message the owner asked to have sent is him speaking to someone through the
        # assistant, and he sees it before it goes. A request that came from someone else
        # was already read and approved by him in full, so answering it needs nothing more.
        from .settings import get as _setting
        confirm_sends = bool(from_owner) and bool(_setting("GATE_CONFIRM_SENDS"))
        if confirm_sends:
            # A tool listed in allowed_tools is pre-approved and the permission hook is never
            # consulted for it. Sends are therefore taken off that list, which is what routes
            # them through the hook so they can be held and shown to him first.
            from .jevagent import is_outbound_message as _out
            allowed = [t for t in allowed if not _out(t, {})]

        async def can_use(tool_name: str, input_data: Dict[str, Any], context: Any):
            if tool_name not in allowed_set:
                return PermissionResultDeny(message=f"{tool_name} is outside this task's scope; do not retry it.")
            from .jevagent import is_destructive, is_outbound_message, recipients_of
            if confirm_sends and is_outbound_message(tool_name, input_data or {}) and not self.allow_send:
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
            if sends_to_requester(tool_name, input_data or {}, protected):
                # Enforced, not just prompted: the gateway delivers the result to whoever asked.
                return PermissionResultDeny(message="Do not message the requester; the gateway delivers your "
                                                    "result. Put the answer in your final status and stop.")
            return PermissionResultAllow()

        system_append = EXECUTOR_SYSTEM + f"\nNow: {now_line()}. Resolve 'today', 'tomorrow', weekday names and relative dates from this.\n"
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
            mcp_servers={"inkbox": self.mcp_server, "host": hosttools.sdk_server(continue_state)},
            # One conversation per thread, continued rather than restarted. A request is a
            # turn in it, so what was looked up an hour ago is still in view and a follow-up
            # does not arrive as a stranger's first sentence.
            resume=self._sessions.get(req.chat_id),
            can_use_tool=can_use,
            max_turns=30,
        )

        async def _go():
            final: Optional[Any] = None
            # Consume the generator to completion; returning from inside the
            # loop leaves the SDK's async generator half-closed.
            async for msg in query(prompt=req.prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            texts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(block.name)
                elif isinstance(msg, ResultMessage):
                    sid = getattr(msg, "session_id", None)
                    if sid:
                        self._sessions[req.chat_id] = sid
                    final = msg
            return final

        try:
            result = await asyncio.wait_for(_go(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"timed out after {int(self.timeout_s)}s", "tool_calls": tool_calls,
                    "raw": "\n".join(texts)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("executor failed for request %s", req.id)
            return {"ok": False, "error": str(exc), "tool_calls": tool_calls, "raw": "\n".join(texts)}

        raw = "\n".join(texts).strip()
        is_error = bool(getattr(result, "is_error", False)) if result is not None else False
        last = raw.strip().splitlines()[-1].strip().upper() if raw.strip() else ""
        if last.startswith("STATUS:"):
            is_error = is_error or ("FAILED" in last)
        elif raw.lower().startswith("task failed") or "STATUS: FAILED" in raw.upper():
            is_error = True
        out = {
            "ok": not is_error,
            "engine": f"claude:{model or self.model}",
            "summary": raw if raw else "(no status text)",
            "tool_calls": tool_calls,
            "raw": raw,
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
        if continue_state:
            out["continue"] = dict(continue_state)
        return out
