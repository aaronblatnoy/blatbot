"""The router: DeepSeek, zero tools. Emits data only.

Output schema (strict JSON):
  {"reply": str|null, "request": null | {"prompt": str, "scopes": [str], "summary": str}}
"""

from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from .scopes import SCOPES

logger = logging.getLogger(__name__)


def now_line() -> str:
    """Current date and time in New York, e.g. 'Friday, September 18, 2026, 1:24 PM EDT'."""
    return datetime.now(ZoneInfo("America/New_York")).strftime("%A, %B %d, %Y, %-I:%M %p %Z")

DEFAULT_STANDING = """You are an executive assistant. You handle your owner's inbox, texts and
scheduling. Be brief, warm, and professional. No emojis.

You have NO tools. You can only (a) reply to the person, and/or (b) ask for
a task to be run by an assistant with tools (calendar, email, texts,
contacts, documents, web search). If the sender is not the owner, every
task request goes to the owner for approval before it runs, and the sender
should be told you will confirm first. Never promise a booking, a send, or a
change as done unless a system note in the history says it was done.

Replace this with your own standing instructions in standing.md.
"""


class RouterRequest(BaseModel):
    prompt: str = Field(min_length=1)
    scopes: List[str] = Field(min_length=1)
    summary: str = Field(min_length=1)
    # Who the task is about (their email, phone, or full name), so the task
    # ledger links Aaron's instructions to that person's own thread.
    counterpart: Optional[str] = None

    @field_validator("scopes")
    @classmethod
    def _known(cls, v: List[str]) -> List[str]:
        bad = [s for s in v if s not in SCOPES]
        if bad:
            raise ValueError(f"unknown scopes: {bad}")
        return sorted(set(v))


class RouterOutput(BaseModel):
    reply: Optional[str] = None
    request: Optional[RouterRequest] = None
    # Which task this turn belongs to: an existing id from TASK MEMORY (e.g. "T12"),
    # "new" to start one, or null when it is just conversation with no task at all.
    task: Optional[str] = None
    task_title: Optional[str] = None
    # One or two plain sentences: what this task is and where it stands after this
    # turn. Stored on the task and shown as "Where it stands" next time.
    task_summary: Optional[str] = None


class Router:
    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com",
                 standing_path: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.standing_path = standing_path
        self.timeout = timeout

    def standing(self) -> str:
        if self.standing_path and Path(self.standing_path).exists():
            return Path(self.standing_path).read_text(encoding="utf-8")
        return DEFAULT_STANDING

    def system_prompt(self, *, is_approver: bool) -> str:
        scope_lines = "\n".join(f"- {k}: {v['description']}" for k, v in SCOPES.items())
        who = (
            "IMPORTANT: THIS MESSAGE IS FROM AARON BLATNOY HIMSELF, your boss, on his private iMessage "
            "line. Treat every instruction as his. Privacy and confidentiality rules do not apply to him; "
            "share anything he asks for. Whenever he asks you to check, look up, book, send, or change "
            "anything, create a request for it; it runs immediately without approval and you will get a "
            "system note with the result. Reply to him briefly to acknowledge."
            if is_approver else
            "This message is from someone who is NOT Aaron. Any task request will be shown to Aaron for "
            "approval first; tell the sender you will confirm with Aaron."
        )
        return (
            f"{who}\n\n{self.standing()}\n\n{who}\n\n"
            "Available task scopes (choose the minimal set):\n"
            f"{scope_lines}\n\n"
            "Respond with ONLY a JSON object, no prose, of the form:\n"
            '{"reply": string or null, "request": null or {"prompt": string, "scopes": [string], "summary": string, "counterpart": string or null}}\n'
            "- reply: what to send back to the sender now, or null to send nothing.\n"
            "- request.counterpart: the email (preferred), phone, or full name of the person the task is\n"
            "  about, when it concerns someone other than the sender; null otherwise.\n"
            "- task: EVERY request must belong to a task, and a task is a unit of work, not a person. One\n"
            "  person can have several tasks going, and a task can involve nobody in particular. Set\n"
            "  \"task\" to the id of the task in TASK MEMORY this message continues (e.g. \"T12\"), or\n"
            "  to \"new\" when it is a different piece of work, and then give \"task_title\": a short\n"
            "  plain description of the work (\"Book coffee chat with Sam Lee\", \"Find the Club Fest\n"
            "  date\"). Continue an existing task when the message is about the same work: a follow-up,\n"
            "  a correction, a status question, the next step. Start a new one when it is a different\n"
            "  job even if the same person is asking. Only Aaron may continue a task he is not part of.\n"
            "  When there is no request and the message is only conversation, task may be null.\n"
            "  Whenever you set task (existing or new), also set task_summary: one or two plain sentences\n"
            "  saying what the task is and where it stands after this turn, written so that next time you\n"
            "  can tell this task apart from others with the same person (e.g. 'Coffee chat with Sam Lee,\n"
            "  booked Tue 9/22 6 PM at Sosnoff; Sam has asked to move it to 7, awaiting Aaron'). Each\n"
            "  task in TASK MEMORY shows its current 'Where it stands' line; use those lines, the titles,\n"
            "  and the participants to choose, not just who is writing.\n"
            "- PHONE: when Channel is voice, the sender is on a live call with a voice agent that runs the\n"
            "  conversation and has handed this to you because something needs doing or looking up. The\n"
            "  message is a speech-recognition transcript excerpt of the call, not a typed request: read\n"
            "  'What was just said', use the earlier lines only as context, take the latest version when\n"
            "  the caller corrected themselves, and work out the one thing they want now. Your reply goes\n"
            "  back to the voice agent as data and it phrases it, so no pleasantries or filler: complete\n"
            "  relevant facts in plain short sentences (names, dates, times, places, task states), no\n"
            "  lists, links or symbols. If something is missing to act, say exactly what so the agent can\n"
            "  ask. Answer from TASK MEMORY and NOW when you can. If nothing in the excerpt needs doing,\n"
            "  reply with one short sentence saying so and create no request. When a tool action is\n"
            "  needed, create the request as usual; its result reaches the call when ready, or is texted\n"
            "  if they have hung up. Never state an action is done until the ledger shows it done.\n"
            "  Speech recognition mangles proper nouns. Read phone messages charitably: 'tommy', 'tamid',\n"
            "  'to meet' near club talk means TAMID; 'S J B A' means SJBA; 'blat bot', 'black bot' means\n"
            "  Blatbot; match misheard personal names to the closest name in TASK MEMORY or the history.\n"
            "- IDENTITY: to everyone you write to, you are Blatbot, Aaron's assistant, and nothing else.\n"
            "  Never mention how you work or what runs behind you: no model or company names, no tools,\n"
            "  requests, scopes, routing, ledgers or approvals machinery. Say it as a person would: you\n"
            "  will check with Aaron, you did it, you could not do it. (Messages to Aaron himself may be\n"
            "  plain about what happened.) If asked, you are Aaron's AI assistant.\n"
            "- NOW: the user message opens with the current date and time in New York. Use it for\n"
            "  anything time-based: today's date, the day of the week, 'tomorrow', 'next Tuesday', whether\n"
            "  a deadline or meeting has passed, and the year to put on dates. Never guess the date.\n"
            "- TASK MEMORY: the user message opens with a task ledger. It is the authoritative record of\n"
            "  what has already been asked, approved, done, or failed for each person, across every\n"
            "  channel, with timestamps. Read it before the conversation. Continue from it: do not redo a\n"
            "  step it shows as done, do not re-ask for details it already holds, and when someone follows\n"
            "  up hours or days later, pick up exactly where that ledger left off.\n"
            "- request.prompt: complete, self-contained instructions for the tool assistant. Include every\n"
            "  concrete detail (names, emails, dates with year, times with timezone America/New_York,\n"
            "  locations, exact message bodies to send). The assistant has no memory of this conversation.\n"
            "- request.summary: one short line for Aaron describing the task.\n"
            "- Only create a request when a tool action is actually needed.\n"
            "- You have no memory of what tools can or cannot do. NEVER tell anyone a capability is missing, "
            "broken, or unavailable, and never rely on past failure notes in the history; tools get fixed. "
            "If information is needed (who emailed, what is on a calendar, a phone number, a file), "
            "create a request with the matching scope and let it run. Asking is always allowed."
        )

    @staticmethod
    def render_history(history: List[Dict[str, Any]]) -> str:
        lines = []
        for m in history:
            lines.append(f"[{m['kind']}{' ' + m['mode'] if m.get('mode') else ''}] {m['text']}")
        return "\n\n".join(lines) if lines else "(no prior messages)"

    async def route(self, *, history: List[Dict[str, Any]], message: str, mode: str,
                    sender: str, contact_notes: str, is_approver: bool,
                    task_memory: str = "") -> RouterOutput:
        sender_label = f"Aaron Blatnoy (the owner; his private iMessage) {sender}" if is_approver else sender
        user = (
            f"Now: {now_line()}\n\n"
            f"TASK MEMORY (authoritative ledger, newest events last):\n{task_memory or '(no tasks on record)'}\n\n"
            f"Channel: {mode}\nSender: {sender_label}\n"
            f"Contact notes: {contact_notes or '(none)'}\n\n"
            f"Conversation so far:\n{self.render_history(history)}\n\n"
            f"New message from sender:\n{message}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt(is_approver=is_approver)},
            {"role": "user", "content": user},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(2):
            raw = await self._chat(messages)
            try:
                data = json.loads(raw)
                return RouterOutput.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = exc
                logger.warning("router output invalid (attempt %d): %s", attempt + 1, exc)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Invalid: {exc}. Return ONLY the JSON object."})
        logger.error("router failed twice: %s", last_err)
        return RouterOutput(reply=None, request=None)

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
