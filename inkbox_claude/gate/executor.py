"""Deterministic executor: one approved prompt -> one Claude Code run.

Uses the Agent SDK one-shot query so the in-process Inkbox MCP server is
available, with allowed_tools fixed by scope and every other tool denied by
the hook. The project deny list in the exec workdir is enforced by Claude
Code underneath.
"""

from __future__ import annotations

import asyncio
import logging
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
from .store import Request, sha256

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM = (
    "You are running ONE approved task for Blatbot, Executive Assistant to Aaron Blatnoy. "
    "You are given the sender's message verbatim, the recent conversation, and the task's record. "
    "Read them and do what the message calls for using only the tools you have, then stop. Do not ask "
    "questions. Do not send anything the message and task record do not call for. If an essential "
    "detail is missing, stop and report what is missing. The gateway delivers your status to the person "
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


class Executor:
    def __init__(self, *, mcp_server: Any, cwd: str, model: str = "sonnet", timeout_s: float = 600.0,
                 protected: Optional[List[str]] = None):
        self.mcp_server = mcp_server
        self.cwd = cwd
        self.model = model
        self.timeout_s = timeout_s
        # Addresses/conversations the gateway itself replies to (the owner's).
        self.protected = list(protected or [])

    async def run(self, req: Request, context: str = "") -> Dict[str, Any]:
        """Run an approved request. Returns a status dict; never raises.

        `context` is the task ledger for the person this request concerns. It is
        appended to the system prompt as read-only background; the task text
        itself (req.prompt) is what was approved and is hash-checked unchanged."""
        if sha256(req.prompt) != req.prompt_sha256:
            return {"ok": False, "error": "prompt hash mismatch; refused to run", "tool_calls": []}
        allowed = tools_for(req.scopes)
        allowed_set = set(allowed)
        tool_calls: List[str] = []
        texts: List[str] = []

        protected = self.protected + [req.sender, req.chat_id]

        async def can_use(tool_name: str, input_data: Dict[str, Any], context: Any):
            if tool_name not in allowed_set:
                return PermissionResultDeny(message=f"{tool_name} is outside this task's scope; do not retry it.")
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
        options = ClaudeAgentOptions(
            cwd=self.cwd,
            model=self.model,
            system_prompt={"type": "preset", "preset": "claude_code", "append": system_append},
            setting_sources=["user", "project"],
            permission_mode="default",
            allowed_tools=allowed,
            mcp_servers={"inkbox": self.mcp_server},
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
        return {
            "ok": not is_error,
            "summary": raw[-1500:] if raw else "(no status text)",
            "tool_calls": tool_calls,
            "raw": raw,
            "cost_usd": getattr(result, "total_cost_usd", None) if result is not None else None,
        }
