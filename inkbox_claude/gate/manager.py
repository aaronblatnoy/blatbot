"""Gate session manager: drop-in for SessionManager when INKBOX_MODE=gate.

Interface the gateway uses:
  get(chat_id, system_prompt_extra="") -> session with
      handle_inbound(text, mode, meta), run_consult(...), run_consult_detailed(...), close()
  close_all()
"""

from __future__ import annotations

from inkbox_claude.instance import OWNER_NAME as _OWNER_NAME, ASSISTANT as _ASSISTANT, ASSISTANT_UPPER as _ASSISTANT_UPPER, ASSISTANT_EMAIL as _ASSISTANT_EMAIL, SERVER_NAME as _SERVER_NAME  # noqa: F401
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .executor import Executor
from . import settings as _settings
from .jevagent import JevAgent, enabled as jev_agent_enabled
from .taskpick import TaskPicker, enabled as jev_enabled
from .router import Router, RouterOutput, RouterRequest, ScheduleCommand
from .scopes import SCOPES, READ_SCOPES, EXPLICIT_ONLY_SCOPES, BLATBOT_ADMIN_SCOPE, LABELS, ASK
from .store import Person, Request, Store, TaskRequired, task_key
from .schedules import ScheduleService, starts_unasked
from . import schedprop
from . import wording
from . import peopleindex
from . import accesscmd, admincmd, chatreply
from . import graphmem
from . import knowledge
from . import outcomes
from . import ownertext
from . import runguard

# Manager.send's answer for a message held for Aaron's yes instead of sent.
HELD_FOR_AARON = object()

logger = logging.getLogger(__name__)
_MISSING = object()
_OUTBOX_HELD = object()

SendFn = Callable[[str, str, str, Dict[str, Any]], Awaitable[Any]]

HANDOFF_LINE = (
    "Thanks! I'll confirm that with Aaron and get back to you shortly.\n\n"
    f"{_ASSISTANT}\nExecutive Assistant to Aaron Blatnoy"
)
_MODE_LABEL = {
    "imessage": "iMessage", "sms": "text", "email": "email",
    "voice": "phone call", "telegram": "Telegram", "whatsapp": "WhatsApp",
}
EXPIRE_SECONDS = 24 * 3600
# Item C2: how long a call's failed ask is still "the same ask" for folding purposes
# on timing alone -- long enough to cover the tail end of one spoken turn, short
# enough that a later fragment on the same task but a genuinely different ask (the
# other test of _is_same_ask_as_failed, caller-words continuation) is the one doing
# the real work past this window.
CALL_ASK_FOLD_WINDOW_S = 45
VOICE_FAST_PATH_MAX_CHARS = 600
# Only messages nobody asked for are paced. A notice or a late result answers a request
# someone made, so it always goes at once.
OUTBOX_PROACTIVE_KINDS = {"schedule", "proactive"}
OUTBOX_BACKOFF = (30, 120, 600)


def _normalize_caller_words(text: str) -> str:
    """The caller's own words from a voice hand-off transcript (wording.caller_lines),
    joined and normalized (whitespace collapsed, lower-cased) for a containment
    comparison (item C2). Falls back to the whole message, same normalization, when
    the text is not hand-off shaped or has no caller lines at all -- a plain spoken
    fragment compares on its own words rather than always failing to match."""
    lines = wording.caller_lines(text or "")
    words = " ".join(lines) if lines else (text or "")
    return " ".join(words.split()).lower()


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CANDIDATE_STOP = {"the", "and", "for", "with", "from", "that", "this", "what", "which", "who", "how", "many", "did",
                   "does", "check", "find", "look", "please", "his", "her", "their", "she", "him", "they", "are", "was",
                   "were", "you", "can", "get", "all", "out", "into", "about", "give", "show", "tell", "see", "have",
                   "has", "had", "when", "where", "why", "yes", "okay", "sure", "thanks", "blatbot", "tamid", "sjba",
                   "stern", "nyu", "today", "tomorrow", "tonight", "week", "any", "some", "there", "here", "just", "also"}
_YES = {"yes", "y", "1", "ok", "approve", "run", "go"}
_NO = {"no", "n", "3", "deny", "reject", "drop", "cancel"}


def is_inbox_command(text: str) -> bool:
    """Keep commands on their own turns so a burst cannot change their meaning."""
    from . import wording, schedprop, roommode
    t = (text or "").strip()
    low = t.lower()
    return bool(
        low in (_YES | _NO | {"tasks", "task list", "open tasks", "what's open", "whats open"})
        or re.match(r"^#?\d+\s*[:\-]?\s*(?:yes|no|y|n|ok|approve|run|go|deny|reject|drop|cancel|edit\s*:.*)$", t, re.I | re.S)
        # "#12 <anything>" may be his answer to a run's question (ask_aaron).
        or re.match(r"^#\d+\s*[:\-]?\s*\S.*$", t, re.S)
        or re.match(r"^t\d+\s*[:\-]?\s*(?:done|close|closed|reopen|open|note\s*:.*)$", t, re.I | re.S)
        or re.match(r"^edit\s*:.+", t, re.I | re.S)
        or wording.is_try_again(t) or wording.is_send_what_you_have(t)
        or wording.is_blanket_permission(t) or schedprop.looks_like_permission_grant(t)
        or roommode.parse_mode_command(t) is not None
    )


def _clean_result(text: str) -> str:
    """Drop the STATUS line and trailing whitespace from an executor result."""
    lines = [ln for ln in text.strip().splitlines() if not ln.strip().upper().startswith("STATUS:")]
    return "\n".join(lines).strip()


def _request_id_from_note(note: str) -> Optional[int]:
    match = re.match(r"Task #(\d+)\b", note or "")
    return int(match.group(1)) if match else None


def _log_answer_path(request_id: Optional[int], path: str, started: float) -> None:
    if request_id is not None:
        logger.info("[gate] #%s answer path: %s in %.1fs",
                    request_id, path, time.monotonic() - started)


DEFAULT_VOICE_VOCABULARY = (
    f"{_ASSISTANT} (BLAT-bot, this assistant). Add the names, places and terms your callers use, with pronunciation "
    "hints, by setting GATE_VOICE_VOCABULARY."
)


def _names_a_task(choice: Optional[str]) -> bool:
    c = (choice or "").strip().lower()
    return c == "new" or (c.startswith("t") and c[1:].isdigit())


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")[-10:]


def _strip_quoted(text: str) -> str:
    """Drop quoted reply history from an email body."""
    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if re.match(r"^On .{5,120} wrote:\s*$", s) or s.startswith("-----Original Message"):
            break
        if s.startswith(">"):
            continue
        out.append(ln)
    return "\n".join(out).strip()


class UntilParseError(ValueError):
    """The router's `until` was not a date or datetime this code can parse."""


def _parse_until(until_text: Optional[str], timezone_name: Optional[str]) -> Optional[float]:
    """The router hands back `until` as an ISO date ("2026-12-31") or datetime
    string, in the schedule's own time zone -- never an epoch. A bare date means
    the end of that day. Raises UntilParseError on anything else, so the caller
    can answer in words instead of letting a schedule store a sentinel-string
    "epoch" or crashing later when schedprop tries to format it as a time."""
    text = (until_text or "").strip()
    if not text:
        return None
    tz = ZoneInfo(timezone_name or "America/New_York")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise UntilParseError(f"I didn't understand the end date {until_text!r}.") from exc
    if dt.tzinfo is None:
        if "T" not in text and len(text) <= 10:
            # A bare date: the schedule runs through the end of that day.
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)
        dt = dt.replace(tzinfo=tz)
    return dt.timestamp()


OWNER_THREAD = "owner"   # the one thread for the person the assistant works for
OWNER_NAME = _OWNER_NAME


class GateSession:
    def __init__(self, chat_id: str, manager: "GateSessionManager"):
        self.chat_id = chat_id
        self.m = manager
        self.mode = "email"
        self.reply_meta: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._turn_task_id: Optional[int] = None  # task the current turn's inbound was logged on
        self._found_task_ids: List[int] = []      # tasks surfaced by this turn's lookup
        self._typing: Optional[asyncio.Task] = None
        self.deferred_ack: Optional[str] = None   # the writer's "on it" line, sent only if the run is slow
        self.room: str = ""                       # set when this thread is a group chat: who is in it
        # Item 4: how many times a voice fragment has already folded into THIS
        # call's already-failed ask, keyed by (call_id, failed request id) so a
        # different call from the same person never shares the count. In-memory
        # only -- it resets with the session, which is the documented fallback
        # ("the voice session object's lifetime") when no call_id is available.
        self._call_ask_replies: Dict[Tuple[Optional[str], int], int] = {}

    @property
    def record_id(self) -> str:
        """Where this conversation is written down. A surface is how a message arrived, not
        who it is with: email, iMessage, SMS, Telegram and the phone are one conversation
        with the owner, so they share one thread. chat_id stays the routing key, so a reply
        still goes back out the way it came."""
        if self.is_approver() and (self.reply_meta or {}).get("conversation_kind") != "group":
            return OWNER_THREAD
        # A group is its own conversation even when he is the one talking: what is said in
        # front of other people does not belong in his private record, and a reply there
        # must not be written from it.
        return self.chat_id

    # -- typing indicator (iMessage) --------------------------------------
    def typing_start(self) -> None:
        """Show "Blatbot is typing" while a message is decided or its request runs.
        iMessage only; the indicator expires on Apple's side, so it is re-sent every
        few seconds until typing_stop."""
        if self.m.typing_fn is None:
            return
        if self.mode == "imessage" and not (self.reply_meta or {}).get("conversation_id"):
            return
        if self.mode not in ("imessage", "telegram"):
            return
        if self._typing is not None and not self._typing.done():
            return
        meta = dict(self.reply_meta)

        mode = self.mode

        async def pulse() -> None:
            try:
                while True:
                    await self.m.typing_fn(self.chat_id, mode, meta)
                    await asyncio.sleep(4)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("[gate %s] typing pulse ended", self.chat_id, exc_info=True)

        self._typing = asyncio.create_task(pulse(), name=f"typing-{self.chat_id}")

    def typing_stop(self) -> None:
        t, self._typing = self._typing, None
        if t is not None and not t.done():
            t.cancel()

    # -- identity ----------------------------------------------------------
    def is_approver(self) -> bool:
        conv = str((self.reply_meta or {}).get("conversation_id") or "")
        if self.mode == "imessage" and bool(conv) and conv == self.m.approver_conv:
            return True
        # Telegram: the numeric user id Telegram stamps on every message. It cannot be
        # set by the sender, so it identifies Aaron as reliably as his iMessage thread.
        if self.mode == "telegram" and self.m.approver_telegram_id:
            return str((self.reply_meta or {}).get("telegram_user_id") or "") == self.m.approver_telegram_id
        # WhatsApp: the sender number is the WhatsApp account's own, verified when it was
        # registered and set by Inkbox on the event, not by the sender, so Aaron's number
        # identifies him on WhatsApp the way his user id does on Telegram.
        if self.mode == "whatsapp":
            return bool(self.m.approver_phone) and _digits(self._sender()) == _digits(self.m.approver_phone)
        # Phone: caller ID can be spoofed, so Aaron's number only counts as the
        # approver when GATE_VOICE_TRUST_APPROVER=1. Otherwise his spoken requests
        # go to his iMessage for a yes like anyone else's.
        return (self.mode == "voice" and self.m.voice_trust_approver
                and bool(self.m.approver_phone) and _digits(self._sender()) == _digits(self.m.approver_phone))

    def is_aaron_on_phone(self) -> bool:
        return (self.mode == "voice" and bool(self.m.approver_phone)
                and _digits(self._sender()) == _digits(self.m.approver_phone))

    def _is_owner_person(self) -> bool:
        """Who this session is talking to, for WORDING ONLY -- never scopes, approval
        power, or anything is_approver() gates. True for the verified approver, for
        Aaron calling from his own number even when GATE_VOICE_TRUST_APPROVER is off
        (caller ID can be spoofed, which is exactly why that case is not is_approver()),
        and for a sender on any channel, including voice, whose contact resolves to the
        commander's person entry. Both caller ID and a contact record can be forged, so
        nothing here may ever stand in for is_approver() -- it only changes which of two
        equally honest sentences gets said to someone who cannot be proven to be Aaron."""
        if self.is_approver() or self.is_aaron_on_phone():
            return True
        try:
            pid = self.m.store.commander_person_id()
            if not pid:
                return False
            res = self.m.store.resolve_handles(self.handles())
            return bool(pid) and not res.get("conflict") and res.get("person_id") == pid
        except Exception:
            logger.exception("[gate %s] could not resolve owner identity for fallback wording", self.chat_id)
            return False

    def _sender(self) -> str:
        return str((self.reply_meta or {}).get("sender") or self.chat_id)

    def _known_person(self) -> Dict[str, Any]:
        """Who the sender is in Aaron's people records, matched on any handle the message
        carries (phone, email, Telegram id), the same match that decides their permissions.
        Empty when no record matches, or when the handles point at two different people."""
        if self.is_approver():
            return {"person": "Aaron Blatnoy", "role": "owner"}
        try:
            found = self.m.store.resolve_handles(self.handles())
        except Exception:
            logger.exception("[gate] could not look up who the sender is")
            return {}
        if found.get("conflict") or not str(found.get("person") or "").strip():
            return {}
        return {"person": str(found["person"]).strip(), "role": str(found.get("role") or "").strip()}

    def _sender_name(self) -> str:
        """The sender's name: the person on record first, whatever the channel calls them
        ("SP", a bare number, an email display name) only when nobody on record matches."""
        known = self._known_person().get("person")
        return known or self._channel_name()

    def _sender_label(self) -> str:
        """Who the sender is, said plainly enough that a run never has to guess."""
        if self.is_approver():
            return "Aaron Blatnoy (the owner)"
        known = self._known_person()
        if known:
            return known["person"] + (f" ({known['role']})" if known.get("role") else "")
        name = self._channel_name()
        shown = f"{name}, " if name else ""
        return f"{shown}someone who is not in Aaron's people records"

    def _channel_name(self) -> str:
        """The name the channel itself attached to the message."""
        c = (self.reply_meta or {}).get("contact") or {}
        if isinstance(c, dict):
            name = str(c.get("name") or "").strip()
            if not name:
                name = " ".join(str(c.get(k) or "") for k in ("given_name", "family_name")).strip()
            return name
        return ""

    def in_group(self) -> bool:
        """This turn arrived in a room with several people in it (gate/knowledge.py)."""
        return knowledge.meta_is_group(self.reply_meta or {})

    def reads_all_tasks(self) -> bool:
        """Whether this turn's ledger may show every task. Aaron in his own chats only: in a
        group, whoever speaks, the ledger shows the tasks that group itself worked on, since
        everything the turn reads can end up in a reply the whole group sees."""
        return self.is_approver() and not self.in_group()

    def ledger_person(self) -> str:
        """Whose own tasks join the ledger: the sender's in a direct chat, nobody's in a group."""
        return "" if self.in_group() else self._sender()

    def memory_groups(self) -> Optional[List[str]]:
        """The long-term memory groups this turn's reader may search (None = all, owner only).
        The rule lives in graphmem.readable_groups; this only hands it who is asking."""
        meta = self.reply_meta or {}
        return graphmem.readable_groups(
            self.m.store, self.record_id, is_approver=self.is_approver(),
            is_group=knowledge.meta_is_group(meta), handles=self.handles())

    def thread_key(self) -> str:
        """The email thread this message belongs to. Prefers the RFC reply chain
        (Message-ID / In-Reply-To / References), which survives across mailboxes;
        falls back to the provider's thread id."""
        if self.mode != "email":
            return ""
        meta = self.reply_meta or {}
        root = self.m.store.thread_root(
            message_id=str(meta.get("message_id") or "").strip(),
            in_reply_to=str(meta.get("in_reply_to") or "").strip(),
            references=[str(x) for x in (meta.get("references") or [])],
        )
        if root:
            return f"email:{root}"
        tid = str(meta.get("thread_id") or "").strip()
        return f"email:{tid}" if tid else ""

    def person(self) -> Person:
        """The sender as one person: contact id plus every email/phone/name we know."""
        c = (self.reply_meta or {}).get("contact")
        p = Person.from_contact(c if isinstance(c, dict) else None, self._sender(), self._channel_name())
        self.m.store.remember_person(p)
        return p

    def handles(self) -> List[str]:
        """Every way this sender reaches the assistant on this turn. Trust is held against
        handles rather than a person record, because a handle is what a message carries."""
        meta = self.reply_meta or {}
        c = meta.get("contact") if isinstance(meta.get("contact"), dict) else {}
        out = [self._sender(), str(meta.get("telegram_user_id") or ""), str((c or {}).get("id") or ""),
               str((c or {}).get("name") or ""), self._channel_name()]
        tg = str(meta.get("telegram_user_id") or "")
        if self.mode == "telegram":
            # Telegram stamps the numeric id; nothing else on a Telegram message is proof of
            # who sent it, and the bare id would read as a phone number.
            return [f"telegram:{tg}"] if tg else []
        return [h for h in out if str(h or "").strip() and str(h) != tg]

    def trust(self) -> Dict[str, Any]:
        """What this sender may have done without asking Aaron. Nothing, unless he has put
        them in the table: everyone starts at no permissions. `commander` is the formal
        name of what the approver already is; it changes only the label here, never the
        check above (is_approver()) or the scopes, and the person name stays the existing
        literal rather than a store lookup -- nothing about is_approver() reads the store."""
        if self.is_approver():
            return {"role": "commander", "scopes": ["*"], "person": "Aaron"}
        try:
            return self.m.store.trust_for(self.handles())
        except Exception:
            logger.exception("[gate] could not read trust; treating as untrusted")
            return {"role": "", "scopes": [], "person": ""}

    def _contact_notes(self) -> str:
        c = (self.reply_meta or {}).get("contact") or {}
        notes = str(c.get("notes") or "") if isinstance(c, dict) else ""
        return f"{self.room}\n\n{notes}".strip() if self.room else notes

    # -- task ledger -----------------------------------------------------------
    def task(self) -> Optional[Dict[str, Any]]:
        """The most recent open task this thread's person is on, if any. Used to
        log conversation onto the task it most likely concerns. Never creates one."""
        if self.is_approver():
            return None
        found = self.m.store.tasks_for_person(self._sender(), open_only=True, limit=1)
        return found[0] if found else None

    def _ensure_inbound_on_task(self, task: Dict[str, Any], body: str) -> None:
        """The inbound was logged on the person's most recent open task at the top of the
        turn. If the router routed this turn to a different (or new) task, log it there."""
        if self._turn_task_id != task["id"]:
            self.m.store.task_event(task["id"], "inbound", body, chat_id=self.record_id)
            self._turn_task_id = task["id"]

    async def jev_event(self, task: Dict[str, Any], body: str, history: List[Dict[str, Any]],
                        out: RouterOutput, result: Optional[Dict[str, Any]] = None) -> None:
        """Second judgment: what this message did to the task it was assigned to.
        Writes a typed event line, updates the summary and, when confident, the state."""
        picker = self.m.task_picker
        if picker is None:
            return
        approver = self.is_approver()
        res = result if result is not None else await picker.judge_event(
            message=body, task=task, history=history,
            sender_label="Aaron, the owner" if approver else (self._sender_name() or self._sender()))
        kind = res.get("kind")
        if not kind or kind == "conversation_only":
            return
        who = "Aaron" if approver else (self._sender_name() or self._sender())
        label = kind.replace("_", " ")
        line = f"{who} {label}: {body.strip()}"
        new_state: Optional[str] = None
        if res.get("resolves_task", 0) >= 0.8 and out.request is None:
            new_state = "done"
        elif res.get("waiting_on") == "owner" and out.request is None:
            new_state = "waiting_aaron"
        elif task.get("state") in ("done", "failed") and kind in ("asked_for_something", "changed_or_corrected"):
            new_state = "open"
        self.m.store.task_event(task["id"], kind, line, chat_id=self.record_id, state=new_state)
        waiting = res.get("waiting_on")
        if waiting in ("owner", "other", "nobody"):
            people = ", ".join(p.get("display") or p.get("key") or "" for p in task.get("participants") or [])
            self.m.store.set_task_waiting_on(task["id"], {
                "owner": "Aaron", "other": people or "other participant", "nobody": "nobody"}[waiting])
        if not out.task_summary:
            # The router gave no summary this turn: keep the ledger honest with a short one.
            wo = res.get("waiting_on")
            tail = {"owner": "Waiting on Aaron.", "other": f"Waiting on {who}.", "nobody": "Nothing outstanding."}.get(wo or "", "")
            self.m.store.set_task_summary(task["id"], f"Latest: {who} {label}. {tail}".strip())

    def prepare_lookup_inputs(self, message: str, in_view: str) -> Dict[str, Any]:
        from .jevagent import _span_candidates
        emails = _EMAIL.findall(message)
        spans = list(emails)
        for c in _span_candidates(message, [], limit=40):
            for v in (re.sub(r"['\u2019]s\b", "", c), c):
                if v and v not in spans:
                    spans.append(v)
        titles: List[str] = []
        for line in in_view.splitlines():
            if line.startswith("Task T"):
                titles.append(" ".join(line[len("Task "):].split(" | ")[:2]))
            elif re.match(r"^T\d+\s*\|", line):
                titles.append(" ".join(part.strip() for part in line.split("|")[:2]))
        label = "Aaron, the owner" if self.is_approver() else (self._sender_name() or self._sender())
        return {"emails": emails, "spans": spans, "titles": titles, "label": label}

    async def _plan_query_safely(self, history: List[Dict[str, Any]], message: str,
                                 in_view: str) -> Any:
        try:
            return await self.m.router.plan_query(history=history, message=message, in_view=in_view,
                                                  is_approver=self.is_approver())
        except Exception:
            logger.exception("[gate %s] task query planning failed", self.chat_id)
            return None

    async def lookup_tasks(self, history: List[Dict[str, Any]], message: str, in_view: str,
                           plan: Any = _MISSING, fallback_query: Any = _MISSING,
                           lookup_inputs: Optional[Dict[str, Any]] = None) -> str:
        """Query pass: the router emits a structured filter, code runs it. Non-owners
        are confined to tasks they are on; the owner can query everything."""
        q: Any = None
        planned = False
        picker = self.m.task_picker
        if picker is not None and hasattr(picker, "plan_lookup"):
            # Jev decides whether the ledger is searched and for which phrase (taken from
            # words in the message). The chat model plans only when Jev is undecided.
            from .router import TaskQuery
            lookup_inputs = lookup_inputs or self.prepare_lookup_inputs(message, in_view)
            emails = lookup_inputs["emails"]
            if plan is _MISSING:
                plan = await picker.plan_lookup(
                    message=message, history=history, sender_label=lookup_inputs["label"],
                    in_view_titles=lookup_inputs["titles"], candidates=lookup_inputs["spans"])
            if isinstance(plan, BaseException):
                plan = {"lookup": None, "text": "", "confidence": 0.0, "reason": f"error: {plan}"}
            if plan.get("lookup") == "none":
                return ""
            if plan.get("lookup") in ("live", "any"):
                planned = True
                text = plan.get("text") or ""
                participant = text if text in emails and self.reads_all_tasks() else None
                q = TaskQuery(text=None if participant else (text or None), participant=participant,
                              states=["live"] if plan["lookup"] == "live" else None, limit=10)
        if not planned:
            q = (await self._plan_query_safely(history, message, in_view)
                 if fallback_query is _MISSING else fallback_query)
            if isinstance(q, BaseException):
                q = None
        self._found_task_ids = []
        if (q is None and in_view.strip() in ("", "(no tasks on record)") and not self.is_approver()
                and not self.in_group()):
            # Nobody on record for this sender: that is exactly when to search for them by name.
            from .router import TaskQuery
            q = TaskQuery(text=(self._sender_name() or "").strip() or None, states=["live"], limit=10)
            if not q.text:
                q = None
        if q is None:
            return ""
        def to_kw(tq: Any) -> Dict[str, Any]:
            kw: Dict[str, Any] = dict(text=tq.text or "", states=tq.states, touched_within_days=tq.touched_within_days,
                                      created_within_days=tq.created_within_days, has_participants=tq.has_participants,
                                      chat_id=self.record_id if tq.this_conversation else "",
                                      date_from=tq.date_from, date_to=tq.date_to, limit=min(tq.limit, 3))
            if self.reads_all_tasks():
                kw["participant"] = tq.participant or ""
            elif self.in_group():
                # A group sees the tasks it worked on itself, never anyone's own tasks.
                kw["chat_id"] = self.record_id
                kw["participant"] = ""
                kw["has_participants"] = None
            else:
                thread_ids = self.m.store.task_ids_for_thread(self.thread_key())
                if thread_ids:
                    # their own tasks, or tasks living on the email thread they are replying in
                    kw["any_of"] = [dict(kw, participant=self._sender(), has_participants=None),
                                    dict(kw, ids=thread_ids, has_participants=None)]
                    kw["participant"] = ""
                else:
                    kw["participant"] = self._sender()  # a non-owner only ever sees their own tasks
                kw["has_participants"] = None
            if tq.any_of:
                kw["any_of"] = [to_kw(sub) for sub in tq.any_of]
            return kw
        res = self.m.store.query_tasks(**to_kw(q))
        self._found_task_ids = [int(t["id"]) for t in res["tasks"]]
        shown = [t for t in res["tasks"] if f"Task T{t['id']} " not in in_view]
        logger.info("[gate %s] task query -> %d match(es), %d new to the router", self.chat_id, res["total"], len(shown))
        if not shown:
            return ""
        if hasattr(self, "_ledger_selected_ids"):
            # Lookup hits share the body budget with the initial tree selection.
            budget = max(3, len(self._ledger_selected_ids))
            matched = set(self._found_task_ids)
            kept = sorted(self._ledger_selected_ids, key=lambda tid: tid not in matched)
            kept = kept[:max(0, budget - len(shown))]
            self._found_task_ids = [int(t["id"]) for t in shown]
            self._ledger_selected_ids = kept
            self._ledger_candidates = [t for t in self._ledger_candidates if t["id"] in kept]
            self._ledger_memory = "\n\n".join(self.m.store.task_memory_for_task(tid) for tid in kept) or "(no tasks on record)"
        text = self.m.store.render_tasks(shown)
        if res["total"] > len(res["tasks"]):
            text += f"\n\n({res['total'] - len(res['tasks'])} more matched; narrow the lookup to see them)"
        return text

    def candidate_tasks(self, message: str = "") -> List[Dict[str, Any]]:
        """Tasks the message could belong to, as a retrieval union, never a recency slice:
        this chat's recent tasks, every live task, tasks of people the message names,
        full-text hits on the message's own words, and the lookup's finds. Scored by how
        many signals agree plus recency, capped at 25 cards for the pick."""
        if hasattr(self, "_ledger_candidates"):
            candidates = list(self._ledger_candidates)
            existing = {t["id"] for t in candidates}
            for tid in self._found_task_ids:
                if tid not in existing:
                    task = self.m.store.get_task(tid)
                    if task:
                        candidates.append(task)
            return candidates
        approver = self.reads_all_tasks()
        group = self.in_group()
        score: Dict[int, float] = {}
        order: List[int] = []

        def add(tid: int, pts: float) -> None:
            if tid not in score:
                order.append(tid)
            score[tid] = score.get(tid, 0.0) + pts

        if not approver and not group:
            for t in self.m.store.tasks_for_person(self._sender(), open_only=False, limit=6):
                add(int(t["id"]), 3.0)
        for i, tid in enumerate(self.m.store.task_ids_for_chat(self.record_id)):
            add(tid, 3.0 - i * 0.5)                                   # locality: what this chat was on
        for tid in self.m.store.task_ids_for_thread(self.thread_key()):
            add(tid, 1.5)
        if approver:
            for t in self.m.store.recent_tasks(limit=40):
                add(int(t["id"]), 1.0 if t["state"] in ("open", "waiting_aaron", "running") else 0.3)
        text = message or ""
        # people the message names: exact participant matches
        people = set(_EMAIL.findall(text)) | set(re.findall(r"(?:[A-Z][a-z'-]+)(?:\s+[A-Z][a-z'-]+)+", text))
        for who in list(people)[:6]:
            for t in self.m.store.tasks_for_person(who, open_only=False, limit=6):
                if approver or self.m.store.is_participant(int(t["id"]), self._sender()):
                    add(int(t["id"]), 2.5)
        # the message's own words, OR'd, against everything on every task
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}|\d{1,2}/\d{1,2}", text)
                 if w.lower() not in _CANDIDATE_STOP][:12]
        if words:
            try:
                kw: Dict[str, Any] = {"text": " OR ".join(words), "limit": 10}
                if group:
                    kw["chat_id"] = self.record_id
                elif not approver:
                    kw["participant"] = self._sender()
                for t in self.m.store.query_tasks(**kw)["tasks"]:
                    add(int(t["id"]), 2.0)
            except Exception:
                logger.debug("[gate %s] candidate text search failed", self.chat_id, exc_info=True)
        for tid in self._found_task_ids:
            add(tid, 2.0)
        ranked = sorted(order, key=lambda tid: -score[tid])
        if group:
            # Names in the message can match anyone's tasks; a group keeps only its own.
            own = set(self.m.store.task_ids_for_chat(self.record_id, limit=200))
            ranked = [tid for tid in ranked if tid in own]
        out: List[Dict[str, Any]] = []
        for tid in ranked[:25]:
            t = self.m.store.task_with_events(tid, limit=4)
            if t and t["state"] != "closed":
                out.append(t)
        return out

    async def jev_pick(self, message: str, history: List[Dict[str, Any]], out: RouterOutput) -> None:
        """Replace the router's task choice with Jev's typed judgment when confident.
        The router's own answer stays as the fallback and as a hint to Jev."""
        picker = self.m.task_picker
        if picker is None:
            return
        fast_id = getattr(self, "_ledger_fast_id", None)
        if fast_id is not None:
            out.task = f"T{fast_id}"
            return
        cands = self.candidate_tasks()
        approver = self.is_approver()
        res = await picker.pick(
            message=message, history=history, candidates=cands,
            sender_label="Aaron, the owner" if approver else (self._sender_name() or self._sender()),
            router_hint=out.task, proposed_title=out.task_title or (out.request.summary if out.request else ""))
        choice = res.get("choice")
        if choice is None:
            logger.info("[gate %s] jev undecided (%s); keeping router pick %r", self.chat_id, res.get("reason"), out.task)
            return
        if choice == "none":
            if out.request is None:
                out.task = None  # pure conversation: no task this turn
            return
        out.task = choice

    def _peek_task(self, out: RouterOutput) -> Optional[Dict[str, Any]]:
        c = (out.task or "").strip().lower()
        if c.startswith("t") and c[1:].isdigit():
            return self.m.store.get_task(int(c[1:]))
        found = self.m.store.tasks_for_person(self._sender(), open_only=True, limit=1) if not self.is_approver() else []
        return found[0] if found else None

    def build_task_prompt(self, body: str, history: List[Dict[str, Any]], task: Dict[str, Any]) -> str:
        """The prompt Claude runs: the sender's own words, the recent conversation, and
        the task's ledger. No model paraphrases the request. Claude reads the source."""
        who = self._sender_label()
        addr = self._sender()
        # Whole messages, not clipped ones: a half-read line is worse than a missing one,
        # and the window matches what the judgments upstream were given.
        convo = "\n".join(f"[{m.get('kind')}] {str(m.get('text') or '')}" for m in history[-20:]) or "(none)"
        ledger = self.m.store.task_memory_for_task(int(task["id"])) or "(new task, no events yet)"
        recalled = str(getattr(self, "_graph_memory", "") or "")
        recalled = f"--- {recalled}\n\n" if recalled else ""
        subject = str((self.reply_meta or {}).get("subject") or "").strip()
        return (
            (f"{self.room}\n\n" if self.room else "")
            + f"A message arrived over {self.mode} from {who} ({addr}).\n"
            + (f"Subject: {subject}\n" if subject else "")
            + f"\n--- THEIR MESSAGE (verbatim) ---\n{body.strip()}\n"
            + _what_it_answers(body, history)
            + _what_it_supplies(body, _last_failed_run(self.m.store, task))
            + "\n"
            f"--- RECENT CONVERSATION WITH THEM ---\n{convo}\n\n"
            f"--- THE TASK THIS BELONGS TO ---\n{ledger}\n\n{recalled}"
            "Work out what they are asking for, or what this message makes possible on the task (for example, "
            "information the task was waiting for), and do it with your tools. A bare reference in the message "
            "(this, that, it, the list, the report) points at the nearest earlier message in the conversation "
            "above that fits, most often the one just before it, from either side; a block of text pasted into "
            "the conversation is content to use as it stands, not something to go and look up. Do only what the "
            "message and the task call for; do not invent extra steps. If something essential is missing, stop "
            "and say exactly what is missing instead of guessing.\n\n"
            + ("This came from a room. Your closing status is the material for what is said back in that room "
               "(the gateway posts it, reworded lightly if at all): write it as the message itself, addressed to "
               "whoever it is for, and nothing else: no narration of what you decided or which channel you "
               "chose.\n\n" if self.room else "")
            + "The task record is background, not an answer. An earlier run on this task answered an earlier "
            "question, and the world has moved since: do the work again now and report what you find this time. "
            "Never close by saying the answer was already given, or that no action is needed because the record "
            "holds one. If you genuinely cannot run the lookup, say that, rather than repeating what the record "
            "says as though you had just checked."
        )

    def _schedules_context(self) -> str:
        """Compact 'title: state' lines for the router's prompt, non-ended schedules
        first -- only for the verified owner, and only when at least one schedule row
        exists. Anyone else, or an empty store, gets "" (the router default): the
        router must never be handed schedule data for a sender who cannot see it."""
        if not self.is_approver():
            return ""
        rows = self.m.store.schedules()
        if not rows:
            return ""
        rows = sorted(rows, key=lambda s: s.state == "ended")
        return "\n".join(f"{s.title}: {s.state}" for s in rows)

    async def decide(self, *, body: str, message: str, prior: List[Dict[str, Any]], mode: str,
                     memory: str, found: str,
                     preflight: Optional[Dict[str, Any]] = None) -> "Tuple[RouterOutput, Optional[Dict[str, Any]]]":
        """One turn's decisions: which task, whether to act, the reply, the request.

        Jev-first flow (default when the picker is on): Jev picks the task and judges
        whether a tool action is needed BEFORE the router runs; the router is told the
        decision and only writes the reply and the task title/summary; the request is
        built by code from the task record (summary = the task title, scopes = Jev's
        judgment over the source prompt). No model paraphrases the request.

        Legacy flow (picker off, or GATE_ROUTER_DEFINES_REQUEST=1): the router proposes
        the request and Jev checks task, action and scopes afterwards."""
        approver = self.is_approver()
        if approver:
            from . import schedcmd
            fixed_action = schedcmd.match_fixed_phrase(message)
            if fixed_action is not None:
                # One of Aaron's own common exact phrasings for status/stuck/today
                # (schedcmd._FIXED_PHRASES): answered without the router or any
                # model call at all, on either flow below.
                out = RouterOutput(schedule_command=ScheduleCommand(action=fixed_action))
                return out, None
        picker = self.m.task_picker
        if self.m.router_defines_requests or picker is None:
            out = await self.m.router.route(history=prior, message=message, mode=mode, sender=self._sender(),
                                            contact_notes=self._contact_notes(), is_approver=approver,
                                            task_memory=memory, found_tasks=found,
                                            schedules_context=self._schedules_context())
            await self.jev_pick(message, prior, out)
            await self.jev_action(message, prior, out, self._peek_task(out))
            task = None
            if out.request is not None or _names_a_task(out.task):
                task = self.m.resolve_task(self, out, message=body)
                self._ensure_inbound_on_task(task, body if mode != "voice" else f"(phone) {body}")
                if out.request is not None:
                    out.request.prompt = self.build_task_prompt(body, prior, task)
                    await self.jev_scopes(out)
                await self.jev_event(task, body, prior, out)
            return out, task

        # --- Jev first ---
        if (os.getenv("GATE_DECIDE_GRAPH") or "1").strip().lower() not in ("0", "false", "no"):
            try:
                from . import decidegraph
            except ImportError as exc:
                logger.warning("[gate %s] langgraph unavailable (%s); deciding in sequence", self.chat_id, exc)
            else:
                return await decidegraph.decide(self, body=body, message=message, prior=prior, mode=mode,
                                                memory=memory, found=found, preflight=preflight)
        label = "Aaron, the owner" if approver else (self._sender_name() or self._sender())
        fast_id = getattr(self, "_ledger_fast_id", None)
        pick = ({"choice": f"T{fast_id}"} if fast_id is not None else
                await picker.pick(message=message, history=prior, candidates=self.candidate_tasks(),
                                  sender_label=label, router_hint=None, proposed_title=""))
        choice = pick.get("choice")
        task_choice = None if choice in (None, "none") else choice
        peek = self._peek_task(RouterOutput(task=task_choice))
        act = await picker.judge_action(message=message, history=prior, task=peek, sender_label=label,
                                        router_said_action=None)
        needs_action = act.get("needs_action")
        if needs_action is None:
            p_act = float(act.get("p") or 0.0)
            # Same tie-break as the decide graph: the exact midpoint leans to action for
            # the owner (cheap to over-act), and away from it for anyone else (a tie
            # would otherwise become an approval request nobody can resolve).
            needs_action = p_act >= 0.4 if approver else p_act > 0.5
        logger.info("[gate %s] jev-first: task=%s action=%s (p=%.2f)", self.chat_id, task_choice, needs_action,
                    float(act.get("p") or 0.0))
        out = await self.m.router.route(history=prior, message=message, mode=mode, sender=self._sender(),
                                        contact_notes=self._contact_notes(), is_approver=approver,
                                        task_memory=memory, found_tasks=found, action=needs_action,
                                        action_task=(peek or {}).get("title") or "",
                                        schedules_context=self._schedules_context())
        out.request = None  # the router never defines the request on this path
        if not needs_action and out.reply and _promises_action(out.reply):
            # No request exists, so a reply that says "on it" / "submitting now" would be a lie.
            logger.info("[gate %s] reply promised an action but none was decided; rewriting", self.chat_id)
            out2 = await self.m.router.route(history=prior, message=message, mode=mode, sender=self._sender(),
                                             contact_notes=self._contact_notes(), is_approver=approver,
                                             task_memory=memory, found_tasks=found, action=False,
                                             action_task="(NOTE: no request has been created and none will run for this "
                                                         "message; do not say you are doing, submitting or checking anything. "
                                                         "Answer from what is known, or ask what is needed.)",
                                             schedules_context=self._schedules_context())
            out.reply = out2.reply if out2.reply and not _promises_action(out2.reply) else None
        if needs_action and out.reply and not _is_acknowledgement(out.reply):
            # A request will run and its result will be delivered. A reply that answers
            # the question now would make two answers; only a short acknowledgement passes.
            logger.info("[gate %s] dropped a %d-word reply written alongside a request; result will follow",
                        self.chat_id, len(out.reply.split()))
            out.reply = None
        if task_choice is not None:
            out.task = task_choice            # Jev's pick wins; router's stands only when Jev abstained
        elif choice == "none" and not needs_action:
            out.task = None
        task = None
        if needs_action or _names_a_task(out.task):
            task = self.m.resolve_task(self, out, message=body)
            self._ensure_inbound_on_task(task, body if mode != "voice" else f"(phone) {body}")
            if needs_action:
                from .router import RouterRequest
                emails = [e for e in _EMAIL.findall(message) if e.lower() != self._sender().lower()]
                out.request = RouterRequest(
                    prompt=self.build_task_prompt(body, prior, task),
                    scopes=["web"],
                    summary=(ownertext.strip_group_tag(task.get("title") or "") or wording.fallback_title_from_message(message)),
                    counterpart=emails[0] if (approver and emails) else None)
                await self.jev_scopes(out, strict=True)
                await self.jev_knowledge(out)
                # A follow-up on a task keeps the tools its earlier requests had: "try again"
                # carries no words for the scope judgment, but the task history does.
                inherited = [sc for sc in self.m.store.scopes_for_task(task["id"]) if sc in SCOPES]
                if inherited:
                    merged = list(out.request.scopes) + [sc for sc in inherited if sc not in out.request.scopes]
                    if merged != out.request.scopes:
                        logger.info("[gate %s] scopes: +%s inherited from T%s", self.chat_id,
                                    [sc for sc in inherited if sc not in out.request.scopes], task["id"])
                    out.request.scopes = merged
            await self.jev_event(task, body, prior, out)
        return out, task

    async def jev_action(self, message: str, history: List[Dict[str, Any]], out: RouterOutput,
                         task: Optional[Dict[str, Any]]) -> None:
        """Jev decides whether a tool action is needed. If yes and the router proposed
        none, synthesise a request from the source; if no and the router proposed one,
        drop it. Undecided keeps the router's call."""
        picker = self.m.task_picker
        if picker is None:
            return
        res = await picker.judge_action(
            message=message, history=history, task=task,
            sender_label="Aaron, the owner" if self.is_approver() else (self._sender_name() or self._sender()),
            router_said_action=out.request is not None)
        na = res.get("needs_action")
        if na is None:
            return
        if na and out.request is None:
            from .router import RouterRequest
            title = (task or {}).get("title") or (out.task_title or "") or wording.fallback_title_from_message(message)
            out.request = RouterRequest(prompt="(built from source)", scopes=["web"], summary=f"Act on: {title}")
            logger.info("[gate %s] jev: action needed (p=%.2f); router proposed none", self.chat_id, res["p"])
        elif not na and out.request is not None:
            logger.info("[gate %s] jev: no action needed (p=%.2f); dropping router request %r",
                        self.chat_id, res["p"], out.request.summary)
            out.request = None

    def _standing_scopes(self) -> List[str]:
        """Everything this sender may use without asking: every tool scope for the owner,
        their role's scopes for a member with one, nothing for anyone else (who still
        goes through the picker and the owner's approval). The owner's grant also
        includes blatbot_admin -- Blatbot's own settings (gate/hosttools.py ADMIN_*
        tools) -- which is never a key in SCOPES at all, so no role can ever pick it
        up: this line is the only place it is ever granted."""
        if self.is_approver():
            return [sc for sc in SCOPES if not sc.startswith("vault:")] + [BLATBOT_ADMIN_SCOPE]
        try:
            role_scopes = list(self.trust().get("scopes") or [])
        except Exception:
            logger.exception("[gate %s] could not read the sender's role", self.chat_id)
            return []
        return [sc for sc in SCOPES if sc in role_scopes]

    async def jev_scopes(self, out: RouterOutput, strict: bool = False,
                         owner_generous: bool = True,
                         result: Optional[Dict[str, Any]] = None) -> None:
        """Replace the router's scope list with Jev's judgment when it yields a
        non-empty set. Empty or unavailable: the router's list stands. Code never
        lets this widen beyond the fixed scope map, since pydantic already
        validated names and the executor only ever grants mapped tools."""
        if out.request is None:
            return
        standing = self._standing_scopes()
        picker = self.m.task_picker
        if self.is_approver():
            if picker is None or not hasattr(picker, "judge_scopes_tree"):
                out.request.scopes = standing
                logger.info("[gate %s] scopes granted (owner, everything): %s", self.chat_id, standing)
                return
            res = result if result is not None else await picker.judge_scopes_tree(
                prompt=out.request.prompt, summary=out.request.summary, generous=True)
            chosen = res.get("scopes")
            if chosen is None:
                out.request.scopes = standing
                logger.info("[gate %s] scopes granted (owner, everything): %s", self.chat_id, standing)
                return
            probs = res.get("probabilities") or {}
            chosen = list(dict.fromkeys(chosen + [k for k, v in probs.items() if v >= 0.5]))
            out.request.scopes = chosen
            logger.info("[gate %s] scopes granted (owner, tree): %s", self.chat_id, chosen)
            return
        if picker is None:
            if standing:
                out.request.scopes = list(dict.fromkeys(list(out.request.scopes) + standing))
            return
        res = result if result is not None else {"scopes": None}
        if result is None and strict and hasattr(picker, "judge_scopes_tree"):
            # Systems first, then read/write and channel within each: a few options per question.
            res = await picker.judge_scopes_tree(
                prompt=out.request.prompt, summary=out.request.summary,
                generous=self.is_approver() and owner_generous,
            )
        if res.get("scopes") is None:
            res = await picker.judge_scopes(
                prompt=out.request.prompt, summary=out.request.summary,  # prompt is the source-built one by now
                scopes={k: str(v["description"]) for k, v in SCOPES.items()},
                router_scopes=None if strict else list(out.request.scopes))
        chosen = res.get("scopes") or []
        probs = res.get("probabilities") or {}
        if strict:
            # Generous by design: reads cost nothing (destructive calls need the owner's
            # yes, sends to the requester are refused), so the owner's requests get every
            # read scope, and writes are granted on a lower bar than the strict judgment.
            if self.is_approver() and owner_generous:
                # The tree already chose with the owner's full picture; every extra read scope
                # only widens the agent's option list and pulls in sources the goal never named.
                chosen = list(dict.fromkeys(chosen + [k for k, v in probs.items() if v >= 0.5]))
            if not chosen:
                limit = 2 if owner_generous else 1
                top = [k for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:limit] if v >= 0.25]
                chosen = top or ["web"]
            logger.info("[gate %s] scopes granted: %s", self.chat_id, chosen)
        if not chosen:
            logger.info("[gate %s] scope judgment undecided (%s); keeping router scopes %s",
                        self.chat_id, res.get("reason"), out.request.scopes)
            if standing:
                out.request.scopes = list(dict.fromkeys(list(out.request.scopes) + standing))
            return
        if standing:
            # A member with a role has everything that role allows on every run. What the
            # judgment adds beyond the role is the part that still waits for the owner.
            chosen = list(dict.fromkeys(list(chosen) + standing))
        if set(chosen) != set(out.request.scopes):
            logger.info("[gate %s] scopes: router %s -> jev %s", self.chat_id, out.request.scopes, chosen)
        out.request.scopes = chosen

    async def jev_knowledge(self, out: RouterOutput,
                            result: Optional[Dict[str, Any]] = None) -> None:
        """Which notes in the vault this request needs, chosen by the same top-down
        traversal as the tool scope tree (TaskPicker.judge_vault_tree): fast, run in
        the decision step rather than inside the executor. The chosen knowledge
        scopes join the request's ordinary scope list -- the same list, the same
        'vault:' names the roles and the console scope picker already carry, under
        the same permission intersection every other scope goes through. Jev chooses
        relevance only; what the sender may see is still decided by that
        intersection afterwards, never by this judgment. Unavailable or nothing
        cleared the bar: today's behaviour stands (whichever vault scope, if any,
        the tool scope tree itself granted)."""
        picker = self.m.task_picker
        if picker is None or out.request is None or not hasattr(picker, "judge_vault_tree"):
            return
        if self.in_group():
            # A group's runs get no vault notes up front, whoever asked (gate/knowledge.py).
            return
        res = result if result is not None else await picker.judge_vault_tree(
            prompt=out.request.prompt, summary=out.request.summary)
        chosen = res.get("scopes")
        if chosen is None:
            logger.info("[gate %s] vault tree unavailable (%s); leaving vault scopes to the tool tree",
                        self.chat_id, res.get("reason"))
            return
        if chosen and not self.is_approver():
            # Relevance never widens what someone else may see, and never turns a request
            # their role already covers into one that waits for approval: only knowledge
            # scopes their roles carry are added. Asking for the owner's notes outright
            # still goes through the tool scope tree and its approval, as before.
            allowed = set(self.trust().get("scopes") or [])
            chosen = [sc for sc in chosen if sc in allowed]
        if chosen:
            merged = list(out.request.scopes) + [sc for sc in chosen if sc not in out.request.scopes]
            if merged != out.request.scopes:
                logger.info("[gate %s] knowledge scopes: +%s (%d notes chosen)", self.chat_id,
                            [sc for sc in chosen if sc not in out.request.scopes], len(res.get("notes") or []))
            out.request.scopes = merged
            out.request.knowledge_notes = list(res.get("notes") or [])

    def apply_knowledge_rule(self, out: RouterOutput) -> None:
        """The last step before a request is stored: in a group, whoever asked, no vault
        scope is kept (gate/knowledge.py). The run can still ask for one exact folder,
        which only Aaron's yes grants."""
        if out.request is None or not self.in_group():
            return
        dropped = knowledge.group_dropped(out.request.scopes or [])
        if dropped:
            out.request.scopes = knowledge.strip_for_group(out.request.scopes or [])
            logger.info("[gate %s] group chat: dropped %s (a group gets no vault notes unasked)",
                        self.chat_id, dropped)
        if getattr(out.request, "knowledge_notes", None):
            out.request.knowledge_notes = []

    def prepare_task_index(self, history: List[Dict[str, Any]], message: str) -> Dict[str, Any]:
        from .ledger import render_index, continuation_id
        start = time.monotonic()
        result = self.m.store.ledger_index(
            is_approver=self.reads_all_tasks(), person=self.ledger_person(), chat_id=self.record_id,
            thread_ids=self.m.store.task_ids_for_thread(self.thread_key()))
        entries = result["tasks"]
        index = render_index(entries)
        self._ledger_fast_id = continuation_id(
            message, entries, self.m.store.task_ids_for_chat(self.record_id), history)
        timings = {"index": round(time.monotonic() - start, 3)}
        return {"start": start, "result": result, "entries": entries, "index": index, "timings": timings}

    async def run_preflight(self, body: str, message: str, history: List[Dict[str, Any]],
                            prepared: Dict[str, Any]) -> Dict[str, Any]:
        """Launch every source-only judgment together and return inert results.

        Nothing here mutates the ledger, request, or task. The ordinary consumers below
        apply the winning results after the fan-out, and ignore speculative fallbacks.
        """
        picker = self.m.task_picker
        entries = list(prepared["entries"])
        lookup_inputs = self.prepare_lookup_inputs(message, prepared["index"])
        label = lookup_inputs["label"]
        likely_task = entries[0] if entries else None
        summary = wording.fallback_title_from_message(message)
        convo = "\n".join(f"[{m.get('kind')}] {str(m.get('text') or '')}" for m in history[-20:])
        source_prompt = message if not convo else f"{message}\n\nRecent conversation:\n{convo}"

        calls: List[Tuple[str, Awaitable[Any]]] = []
        # With graph memory on, the task records are picked from what the graph recalls;
        # the scored picker only runs later if the graph matched no task.
        if (self._ledger_fast_id is None and picker is not None and hasattr(picker, "judge_ledger_tree")
                and not graphmem.enabled()):
            calls.append(("ledger", picker.judge_ledger_tree(
                message=message, history=history, entries=entries)))
        if picker is not None and hasattr(picker, "plan_lookup"):
            calls.append(("lookup", picker.plan_lookup(
                message=message, history=history, sender_label=label,
                in_view_titles=lookup_inputs["titles"], candidates=lookup_inputs["spans"])))
        calls.append(("query", self._plan_query_safely(history, message, prepared["index"])))
        if self._ledger_fast_id is None and picker is not None and hasattr(picker, "pick"):
            calls.append(("pick", picker.pick(
                message=message, history=history, candidates=entries, sender_label=label,
                router_hint=None, proposed_title="")))
        if picker is not None and hasattr(picker, "judge_action"):
            calls.append(("action", picker.judge_action(
                message=message, history=history, task=likely_task, sender_label=label,
                router_said_action=None)))
        if picker is not None and hasattr(picker, "judge_actionable"):
            calls.append(("actionable", picker.judge_actionable(
                message=message, history=history, task=likely_task)))
        if picker is not None and hasattr(picker, "judge_vault_tree"):
            calls.append(("vault", picker.judge_vault_tree(prompt=source_prompt, summary=summary)))
        if picker is not None and hasattr(picker, "judge_event") and likely_task is not None:
            calls.append(("event", picker.judge_event(
                message=body, task=likely_task, history=history, sender_label=label)))
        if picker is not None and hasattr(picker, "judge_complexity"):
            calls.append(("complexity", picker.judge_complexity(
                prompt=source_prompt, summary=summary, scopes=["web"])))
        if (not self.is_approver() and picker is not None
                and hasattr(picker, "judge_scopes_tree")):
            calls.append(("scopes", picker.judge_scopes_tree(
                prompt=source_prompt, summary=summary, generous=False)))

        started = time.monotonic()
        values = await asyncio.gather(*(call for _, call in calls), return_exceptions=True)
        elapsed = time.monotonic() - started
        defaults: Dict[str, Any] = {
            "ledger": {"ids": None, "timings": {}, "reason": "unavailable"},
            "lookup": {"lookup": None, "text": "", "confidence": 0.0, "reason": "unavailable"},
            "query": None,
            "pick": {"choice": None, "confidence": 0.0, "probabilities": {}, "reason": "unavailable"},
            "action": {"needs_action": None, "p": 0.0, "reason": "unavailable"},
            "actionable": {"actionable": None, "p": 0.0, "reason": "unavailable"},
            "vault": {"scopes": None, "notes": [], "probabilities": {}, "reason": "unavailable", "timings": {}},
            "event": {"kind": None, "kind_conf": 0.0, "waiting_on": None,
                      "resolves_task": 0.0, "reason": "unavailable"},
            "complexity": {"complex": None, "p": 0.0, "reason": "unavailable"},
            "scopes": {"scopes": None, "probabilities": {}, "reason": "unavailable"},
        }
        results: Dict[str, Any] = {}
        for (name, _), value in zip(calls, values):
            if isinstance(value, BaseException):
                logger.warning("[gate %s] preflight %s failed: %s", self.chat_id, name, value)
                value = defaults[name]
            results[name] = value
        results["lookup_inputs"] = lookup_inputs
        results["event_task_id"] = likely_task.get("id") if likely_task else None
        logger.info("[gate %s] preflight_total: %.3fs (n=%d calls)", self.chat_id, elapsed, len(calls))
        return results

    async def indexed_task_memory(self, history: List[Dict[str, Any]], message: str,
                                  prepared: Optional[Dict[str, Any]] = None,
                                  pick: Any = _MISSING) -> str:
        prepared = prepared or self.prepare_task_index(history, message)
        start = prepared["start"]
        result = prepared["result"]
        entries = prepared["entries"]
        index = prepared["index"]
        timings = dict(prepared["timings"])
        picker = self.m.task_picker
        graph_ids: List[int] = []
        if self._ledger_fast_id is None and getattr(self, "_recall", None) is not None:
            # Tasks the recalled facts were learned in, limited to tasks this reader may see.
            self._graph_facts = await self._recall
            open_ids = {t["id"] for t in entries}
            graph_ids = [tid for tid in self.m.store.task_ids_for_messages(
                graphmem.message_ids(self._graph_facts)) if tid in open_ids][:3]
        if self._ledger_fast_id is not None:
            ids = [self._ledger_fast_id]
            timings["fast_path"] = True
        elif graph_ids:
            ids = graph_ids
            timings["graph"] = True
        elif picker is not None and hasattr(picker, "judge_ledger_tree"):
            if pick is _MISSING:
                pick = await picker.judge_ledger_tree(message=message, history=history, entries=entries)
            if isinstance(pick, BaseException):
                pick = {"ids": None, "timings": {}, "reason": f"error: {pick}"}
            ids = pick.get("ids")
            timings.update(pick.get("timings") or {})
            if ids is None:
                ids = [t["id"] for t in entries[:3]]
        else:
            ids = [t["id"] for t in entries[:3]]
        allowed = {t["id"] for t in entries}
        ids = list(dict.fromkeys(tid for tid in ids if tid in allowed))
        self._ledger_selected_ids = ids
        self._ledger_candidates = [t for tid in ids for t in entries if t["id"] == tid]
        self._found_task_ids = []
        load_start = time.monotonic()
        blocks = [self.m.store.task_memory_for_task(tid) for tid in ids]
        self._ledger_memory = "\n\n".join(block for block in blocks if block) or "(no tasks on record)"
        timings["bodies"] = round(time.monotonic() - load_start, 3)
        timings["total"] = round(time.monotonic() - start, 3)
        self._ledger_log = (len(entries), len(index), timings, result["dropped"], start)
        return self._ledger_memory

    def log_ledger_selection(self, memory: str, found: str) -> None:
        count, chars, timings, dropped, start = self._ledger_log
        timings = dict(timings, total=round(time.monotonic() - start, 3))
        ids = list(dict.fromkeys(self._ledger_selected_ids + self._found_task_ids))
        logger.info("[gate %s] ledger index: %d tasks, %d chars -> picked %s (%d chars) timings=%s dropped_stale=%d",
                    self.chat_id, count, chars, [f"T{tid}" for tid in ids],
                    len(memory) + len(found), timings, dropped)

    def task_memory(self) -> str:
        if hasattr(self, "_ledger_memory"):
            return self._ledger_memory
        # Synchronous callers have no message to judge. Keep a bounded metadata fallback.
        result = self.m.store.ledger_index(is_approver=self.reads_all_tasks(), person=self.ledger_person(),
                                          chat_id=self.record_id,
                                          thread_ids=self.m.store.task_ids_for_thread(self.thread_key()))
        return "\n\n".join(self.m.store.task_memory_for_task(t["id"])
                             for t in result["tasks"][:3]) or "(no tasks on record)"

    # -- gateway interface ---------------------------------------------------
    async def handle_inbound(self, text: str, mode: str, meta: Dict[str, Any],
                             *, raise_errors: bool = False) -> None:
        async with self._lock:
            self.__dict__.pop("_ledger_memory", None)
            self.__dict__.pop("_ledger_candidates", None)
            self.__dict__.pop("_ledger_fast_id", None)
            self.mode = mode
            self.reply_meta = dict(meta or {})
            self.m.store.set_thread(self.chat_id, "routing", mode, self.reply_meta_for_store())
            if self.is_approver():
                # Remembered on disk: notices have to land here even when no session is loaded.
                self.m.store.mark_owner_thread(self.chat_id, mode)
            body = _strip_quoted(text) if mode == "email" else text.strip()
            self.inbound_id = self.m.store.add_message(self.record_id, "inbound", body, mode)
            self._index_sender()
            self.deferred_ack = None
            self.typing_start()
            if self.thread_key():
                self.m.store.link_thread(self.thread_key(), self.record_id)
            # First thing every turn: put this message on the person's task ledger,
            # so the router starts from the record, not from a 20-message window.
            t = self.task()
            self._turn_task_id = t["id"] if t is not None else None
            if t is not None:
                self.m.store.task_event(t["id"], "inbound", body, chat_id=self.record_id)
            try:
                if self.mode != "voice" and await self._maybe_stop_running(body):
                    return
                if self.is_approver() and self.mode != "voice" and await self.m.handle_approver_command(self, body):
                    return
                if self.mode != "voice" and await self.m.handle_followup_phrase(self, body):
                    return
                if not self.is_approver() and self.mode != "voice":
                    # Deterministic FYI to Aaron for every inbound from anyone else.
                    await self.m.notify_inbound(self, body)
                await self._route_and_act(body)
            except Exception as exc:
                logger.exception("[gate %s] turn failed", self.chat_id)
                await self.m.report_outage(exc, self.chat_id)
                if raise_errors:
                    raise
            finally:
                self.typing_stop()
                if self.m.store.thread_state(self.chat_id) == "routing":
                    self.m.store.set_thread(self.chat_id, "idle")

    async def file_inbound(self, text: str, mode: str, meta: Dict[str, Any]) -> Optional[int]:
        """Keep a message in this conversation's history, and so in long-term memory,
        without acting on it: no task, no request, no reply. For messages that arrive
        too late to answer (a backlog delivered after an outage). The meta is used only
        to find where the conversation is written down; the session's own routing is
        left as it was, so a reply already on its way still goes where it was going."""
        async with self._lock:
            saved = (self.mode, self.reply_meta)
            try:
                self.mode = mode
                self.reply_meta = dict(meta or {})
                body = _strip_quoted(text) if mode == "email" else (text or "").strip()
                if not body:
                    return None
                return self.m.store.add_message(self.record_id, "inbound", body, mode)
            finally:
                self.mode, self.reply_meta = saved

    _INDEX_PLACES = {"email": "email", "sms": "SMS", "imessage": "iMessage", "whatsapp": "WhatsApp"}

    def _index_sender(self) -> None:
        """Someone writing on email, SMS or iMessage gets a People record if they have
        none (peopleindex.py). Telegram is indexed in gateway.py, where group members
        who never reach a session are seen too. Calls, the owner and mail from a
        mailbox no person writes from (noreply@ ...) are not indexed."""
        label = self._INDEX_PLACES.get(self.mode)
        if label is None or self.is_approver() or self._is_owner_person():
            return
        meta = self.reply_meta or {}
        sender = str(meta.get("sender") or "").strip()
        if not sender or (self.mode == "email" and peopleindex.is_automated_email(sender)):
            return
        kind = Store.guess_contact_kind(sender)
        if kind not in ("email", "phone"):
            return
        contact = meta.get("contact") if isinstance(meta.get("contact"), dict) else {}
        name = str((contact or {}).get("name") or "").strip()
        group = str(meta.get("group_title") or "").strip()
        place = (f"{group} ({label})" if group else f"a {label} group") if (
            meta.get("conversation_kind") == "group") else label
        self.m.index_person_seen(display=name, handles=[(kind, sender)], met_in=place)

    def reply_meta_for_store(self) -> Dict[str, Any]:
        keep = {k: v for k, v in (self.reply_meta or {}).items()
                if k in ("to", "sender", "subject", "thread_id", "conversation_id", "conversation_kind",
                         "message_id", "in_reply_to", "references",
                         # A group's display name, so the console's group-chats list (and
                         # find_group_chat_by_name for the "blatbot mode <chat name> ..."
                         # command) has a real name instead of the bare chat_id.
                         "group_title")}
        return keep

    async def run_consult(self, *args: Any, **kwargs: Any) -> str:
        """Gateway wake-ups. The only one the gate acts on is the end of a phone
        call: it closes the call on the ledger. No transcript is texted to Aaron;
        anything a caller asked for already reached him as an approval request. Live call turns go through handle_inbound like any channel."""
        prompt = str(args[0]) if args else ""
        if "[voice call ended]" not in prompt and "[call_ended]" not in prompt:
            return ""
        convo = prompt.split("call transcript:", 1)[-1].strip() if "call transcript:" in prompt.lower() else ""
        if not convo:
            m = re.search(r"(?is)call transcript:\s*(.*)", prompt)
            convo = m.group(1).strip() if m else ""
        try:
            t = self.task()
            if t is not None:
                self.m.store.task_event(t["id"], "call_ended", "Phone call ended.", chat_id=self.record_id)
            self.m.store.add_message(self.record_id, "system", "Phone call ended.")
        except Exception:
            logger.exception("[gate %s] call-ended hook failed", self.chat_id)
        return ""

    def voice_briefing(self, meta: Dict[str, Any]) -> str:
        """Snapshot handed to the voice model when a call connects."""
        from .router import now_line
        prev = (self.mode, self.reply_meta)
        self.mode, self.reply_meta = "voice", dict(meta or {})
        try:
            trusted = self.is_approver()
            sender, sender_name = self._sender(), self._sender_name()
        finally:
            self.mode, self.reply_meta = prev  # a call never changes the thread's text/email channel
        who = ("You're talking to Aaron himself. He's your boss and you know him well; be yourself. Whatever he "
               "asks for gets done right away. He usually calls while walking around New York City, so expect "
               "traffic, sirens, wind and other people's voices on the line. Ignore all of that: only respond to "
               "Aaron. Don't react to stray words that clearly aren't meant for you, don't comment on the noise, "
               "and if you only caught part of what he said, ask him to repeat that one part rather than guess. "
               "Keep answers short and front-load the important bit, since he may miss the end of a long sentence." if trusted else
               f"The caller is NOT Aaron ({sender_name or 'unknown'}, {sender}). Be helpful and "
               "discreet: share nothing about Aaron's schedule, tasks or other people beyond what concerns this "
               "caller. Anything they ask to have done needs Aaron's confirmation first; consult handles that.")
        notes = ""
        try:
            prev2 = (self.mode, self.reply_meta)
            self.mode, self.reply_meta = "voice", dict(meta or {})
            try:
                if trusted:
                    import time as _t
                    now = _t.time()
                    rows = self.m.store.recent_tasks(limit=14)
                    out = []
                    for r in rows:
                        title = str(r["title"] or "")
                        if title.startswith("Thread with") or title == "Untitled task" or r["state"] == "closed":
                            continue  # a bare conversation, not a task
                        hrs = (now - r["updated_at"]) / 3600
                        age = f"{int(hrs * 60)} min ago" if hrs < 1 else (f"{int(hrs)} hours ago" if hrs < 48 else f"{int(hrs / 24)} days ago")
                        state = {"open": "in progress", "waiting_aaron": "waiting on Aaron's okay",
                                 "running": "running now", "done": "done", "failed": "didn't work"}.get(r["state"], r["state"])
                        line = f"- {title} ({state}, {age})"
                        stands = " ".join(str(r.get("summary") or "").split())
                        if stands and not stands.startswith("Latest: Aaron asked for something"):
                            line += f"\n    where it stands: {stands[:280]}"
                        # The last result on the task, so facts already found can be said without a hand-off.
                        full = self.m.store.task_with_events(int(r["id"]), limit=8) or {}
                        for e in reversed(full.get("events") or []):
                            if e.get("kind") == "done":
                                res = " ".join(str(e.get("text") or "").split(" -> ", 1)[-1].split())
                                res = res.split("WRITES PERFORMED", 1)[0].split("Done via Jev agent", 1)[0].strip()
                                if res:
                                    # Conclusions come last in a result; the head is narration.
                                    line += f"\n    last result: {('... ' + res[-320:]) if len(res) > 320 else res}"
                                break
                        out.append(line)
                    notes = ("Recent things you've been working on, newest first, with what was found. Answer from "
                             "these directly when they already hold the answer; hand off only for something newer "
                             "or not here. This is background, not a menu and not a limit on what you can do:\n"
                             + "\n".join(out))
                else:
                    notes = "What's on record with this caller:\n" + self.task_memory()
            finally:
                self.mode, self.reply_meta = prev2
        except Exception:
            logger.exception("[gate %s] briefing notes failed", self.chat_id)
        return (
            "NOTES FOR THIS CALL\n"
            f"It is {now_line()}.\n{who}\n\n"
            f"Names you'll hear, said as written: {self.m.voice_vocabulary}\n\n"
            f"{notes}\n"
            "These notes were true when the call started. Talk from them freely; check for anything newer "
            "or more detailed."
        )

    def _is_same_ask_as_failed(self, failed: Request, body: str) -> bool:
        """Item C2: a repeated call fragment only folds into an already-failed request
        when it is genuinely the same ask, not merely the same task within a loose
        5-minute window (which used to swallow a different ask on the same topic).
        Same ask means either: this fragment arrived within CALL_ASK_FOLD_WINDOW_S
        (45s) of the failed request's own inbound, so it reads as the tail end of
        the same spoken turn; or the two fragments' caller words are a continuation
        of the same utterance -- after normalizing whitespace and case, one set of
        "what was just said" caller lines contains, or is contained in, the other.
        Anything else is a new ask and is never folded."""
        if (time.time() - failed.updated_at) <= CALL_ASK_FOLD_WINDOW_S:
            return True
        failed_said = _normalize_caller_words(failed.original_message)
        this_said = _normalize_caller_words(body)
        if not failed_said or not this_said:
            return False
        return failed_said in this_said or this_said in failed_said

    async def _fold_into_failed_ask(self, task_id: int, call_id: Optional[str],
                                    body: str = "") -> Optional[str]:
        """Item 4: when a voice fragment's ask is the SAME ask (same task) as a
        request that already failed on this thread -- the realtime voice model
        can hand off several fragments of one spoken request a few seconds apart,
        and a run that fails in a few seconds leaves nothing "running" for the
        existing fold (above) to catch -- a further fragment never starts a
        second run of it. The failure is spoken once, then a short "still stuck"
        line at most once more, then nothing further for this ask. Returns None
        when this is not that case (a genuinely new ask proceeds as normal).

        Item C9: the "still stuck" line is spoken, not texted -- the caller never
        actually gets the failure in writing unless he hangs up right after hearing
        it (and send_to_sender's hang-up path only fires for the CURRENT fragment's
        own inbound, not the earlier failed request). So the moment that line is
        said, the failure message is also delivered once to a text channel -- the
        same single-channel choice _run_for_call uses after hang-up (owner's own
        primary when this is Aaron, otherwise back to the number that called) --
        and the failed request is marked answered so that delivery never repeats,
        whether this fold fires again or the request is later handled at hang-up."""
        failed = self.m.store.last_failed_for_thread(self.chat_id, exclude_scheduled=True)
        if failed is None:
            return None
        failed_task_id = self.m.store.task_id_for_request(failed.id)
        if failed_task_id != task_id:
            return None
        if not self._is_same_ask_as_failed(failed, body):
            return None
        key = (call_id, failed.id)
        count = self._call_ask_replies.get(key, 0)
        self._call_ask_replies[key] = count + 1
        if count == 0:
            text = wording.failure_message(wording.plain_title(self.m.store, failed), failed.status or {})
        elif count == 1:
            text = "Still stuck on that. I'll text you."
            already = failed.inbound_id and "answer" in self.m.store.responses_to(failed.inbound_id)
            if not already:
                failure_text = wording.failure_message(wording.plain_title(self.m.store, failed), failed.status or {})
                if self.is_aaron_on_phone():
                    await self.m.send_to_approver(failure_text)
                else:
                    num = self._sender()
                    await self.m.send(f"sms:{num}", failure_text, "sms", {"to": num, "sender": num}, record=False)
                if failed.inbound_id:
                    self.m.store.add_message(self.record_id, "outbound", failure_text, "sms",
                                             reply_to=failed.inbound_id, role="answer")
        else:
            text = ""
        if text:
            self.m.store.add_message(self.record_id, "outbound", text, "voice")
        return text or "Got it."

    async def voice_consult(self, query: str, meta: Dict[str, Any]) -> str:
        """One consult from the realtime voice model during a live call.

        Same path as any inbound message (ledger, router, gate), except the
        answer is returned for the voice model to say instead of being sent.
        Aaron's trusted number runs the request now and hears the result;
        anyone else's request goes to Aaron for a yes and they are told so."""
        req = None
        call_meta = dict(meta or {})
        async with self._lock:
            prev = (self.mode, self.reply_meta)
            self.mode, self.reply_meta = "voice", dict(call_meta)
            self.m.store.set_thread(self.chat_id, "routing")  # state only: keep their text/email route
            body = (query or "").strip()
            self.inbound_id = self.m.store.add_message(self.record_id, "inbound", body, "voice")
            t = self.task()
            self._turn_task_id = t["id"] if t is not None else None
            if t is not None:
                self.m.store.task_event(t["id"], "inbound", f"(phone) {body}", chat_id=self.record_id)
            try:
                approver = self.is_approver()
                history = self.m.store.history(self.record_id, limit=20)
                # Long-term memory for a spoken turn too; runs alongside the ledger pick.
                recall = asyncio.ensure_future(graphmem.recall_facts(body, self.record_id, approver, self.memory_groups()))
                self._recall = recall if graphmem.enabled() else None
                memory = await self.indexed_task_memory(history[:-1], body)
                found = await self.lookup_tasks(history[:-1], body, memory)
                self._graph_memory = graphmem.block(await recall)
                self._recall = None
                if self._graph_memory:
                    memory = f"{memory}\n\n{self._graph_memory}"
                memory = self._ledger_memory
                self.log_ledger_selection(memory, found)
                # One request per ask, including from the trusted caller: the realtime
                # voice model hands this consult one transcript fragment at a time, so
                # one spoken request can arrive as several calls a few seconds apart.
                # running_for_thread already covers both 'approved' and 'running' (and
                # already excludes schedule requests), so a fragment that lands while
                # the first fragment's request is still open folds in as more context on
                # that request -- the same "meanwhile" handling an ordinary follow-up
                # message gets in _route_and_act -- instead of becoming a second request.
                running = self.m.store.running_for_thread(self.chat_id)
                if running is not None:
                    logger.info("[gate %s] request #%s still open; folding this voice fragment into it "
                                "instead of starting another", self.chat_id, running.id)
                    if t is not None:
                        self._ensure_inbound_on_task(t, body)
                    out = await self.m.router.route(
                        history=history[:-1], message=body, mode="voice", sender=self._sender(),
                        contact_notes=self._contact_notes(), is_approver=approver,
                        task_memory=memory, found_tasks=found, action=True,
                        action_task=(t or {}).get("title") or running.summary,
                    )
                    reply = out.reply or "Got it."
                    self.m.store.add_message(self.record_id, "outbound", reply, "voice")
                    if t is not None:
                        self.m.store.task_event(t["id"], "outbound", f"(phone) {reply}", chat_id=self.record_id)
                    return reply
                # Item A5: checking for the fold BEFORE calling decide() here would save
                # a model round on a certain dedupe hit, but it cannot be done safely:
                # the running_for_thread check just above already covers every case
                # where create_request_once's content-based dedupe would also fire --
                # running_for_thread is broader while a request remains open, so it has
                # already folded this fragment through the branch above and returned.
                # The only way create_request_once's
                # dedupe still fires below is a race in the time decide() itself takes
                # (another fragment's request lands mid-call) -- a window too narrow to
                # pre-check meaningfully, so the model round still happens; only the
                # reply said on that rare hit is fixed, just below.
                out, task = await self.decide(body=body, message=body, prior=history[:-1], mode="voice",
                                              memory=memory, found=found)
                if task is not None and out.request is not None:
                    # Item 4: this ask's task may be the same one a request already
                    # failed under on this call -- never start a second run of it.
                    folded = await self._fold_into_failed_ask(task["id"], call_meta.get("call_id"), body)
                    if folded is not None:
                        if t is not None:
                            self._ensure_inbound_on_task(t, body)
                        return folded
                reply = out.reply or ""
                if reply:
                    self.m.store.add_message(self.record_id, "outbound", reply, "voice")
                    if t is not None:
                        self.m.store.task_event(t["id"], "outbound", f"(phone) {reply}", chat_id=self.record_id)
                if out.request is None:
                    return reply or "I don't have anything on that."
                if self.m.store.pending_for_thread(self.chat_id, sender=self._sender()) and not approver:
                    return reply or "I already have a request waiting on Aaron for you."
                assert task is not None
                req, created = self.m.store.create_request_once(
                    chat_id=self.chat_id, sender=self._sender(), sender_name=self._sender_name(), mode="voice",
                    subject="", original_message=body, summary=out.request.summary,
                    scopes=out.request.scopes, prompt=out.request.prompt,
                    state="approved" if approver else "pending", task_id=task["id"], inbound_id=self.inbound_id)
                if not created:
                    # Another copy of this same ask already has a recent request on this
                    # thread: never execute a second run of it, and never treat this
                    # fragment as having just created or widened anything.
                    logger.info("[gate %s] voice fragment deduped onto existing request #%s; "
                                "not running it a second time", self.chat_id, req.id)
                    # Item A5: nothing new was created, so the fixed line is said --
                    # never the model's own acknowledgement (already stored above as
                    # `reply`, which is a separate, pre-existing inconsistency this
                    # does not attempt to fix: reordering that storage risks losing the
                    # model's genuinely-new reply on the common, non-dedupe path).
                    return "Got it."
                self.m.remember_knowledge_notes(req.id, getattr(out.request, "knowledge_notes", None))
                self.m.record_request_on_task(req, task, self)
                if not approver:
                    self.m.store.set_thread(self.chat_id, "awaiting_aaron")
                    await self.m.send_approval_text(req)
                    return reply or "I need to confirm that with Aaron first. I will follow up by text once he answers."
            except Exception:
                logger.exception("[gate %s] voice consult failed", self.chat_id)
                return "Something went wrong on my end. Please try that again."
            finally:
                if self.m.store.thread_state(self.chat_id) == "routing":
                    self.m.store.set_thread(self.chat_id, "idle")
                aaron_call = self.is_aaron_on_phone()
                caller = self._sender()
                self.mode, self.reply_meta = prev
        # Trusted caller: run it now, outside the lock, and hand back the result to speak.
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._run_for_call(req, aaron_call, caller, call_meta)), timeout=75)
        except asyncio.TimeoutError:
            return "That is still running. I will text you the result as soon as it finishes."

    async def phrase_result_for_call(self, req: Request, text: str, call_meta: Dict[str, Any], ok: bool) -> str:
        """Turn a finished request's result into what the voice model says. A failure is
        already the whole message wording.py built for it (`execute()` returns it as-is
        with notify=False); it is spoken exactly as written, never handed to a model that
        can rewrite it into something the grounding check then rejects. A short success
        is spoken directly from the run's own narration; only a long success goes through
        the reply writer and grounding check. `ok` comes from the request's own status,
        never from sniffing the text for a leading "Done"."""
        if not ok:
            return text
        direct_started = time.monotonic()
        cleaned = _result_bodies(text)
        if cleaned and len(cleaned) <= VOICE_FAST_PATH_MAX_CHARS:
            _log_answer_path(req.id, "voice-direct", direct_started)
            return " ".join(cleaned.split())
        note = f"Task #{req.id} done: {wording.plain_title(self.m.store, req)}\nResult:\n{text}"
        async with self._lock:
            prev = (self.mode, self.reply_meta)
            self.mode, self.reply_meta = "voice", dict(call_meta or {})
            try:
                approver = self.is_approver()
                history = self.m.store.history(self.record_id, limit=20)
                memory = self.task_memory()
                answer_started = time.monotonic()
                out = await self.m.router.route(
                    history=history, mode="voice", sender=self._sender(), contact_notes=self._contact_notes(),
                    is_approver=approver, task_memory=memory, found_tasks="", action=False,
                    message=("(No new message. The task result above just arrived DURING A PHONE CALL. Write what "
                             "the voice assistant should now say to the caller: one to three short spoken sentences, "
                             "the answer or outcome first, names and times as words, no lists, no links, no symbols, "
                             "no internal details. If the result is only partial, say plainly what was and was not "
                             "found. Do not create a request.)"),
                    action_task=wording.plain_title(self.m.store, req))
                out.request = None
                reply = await self.ground_reply(
                    out.reply, note, history, memory, "", approver,
                    answer_started=answer_started, answer_request_id=req.id)
            except Exception:
                logger.exception("[gate %s] phrasing a call result failed", self.chat_id)
                reply = None
            finally:
                self.mode, self.reply_meta = prev
        reply = " ".join((reply or "").split())
        if reply and not reply.startswith(("Done via", "Failed via")):
            return reply
        # Fallback: the tool bodies only, never the agent's header.
        body = _result_bodies(text.split("Result:", 1)[-1] if "Result:" in text else text)
        return " ".join(body.split())[:700] or text

    async def _run_for_call(self, req: Request, aaron_call: bool, caller: str,
                            call_meta: Dict[str, Any]) -> str:
        text = await self.m.execute(req, notify=False)
        fresh = self.m.store.get_request(req.id)
        ok = bool((fresh.status or {}).get("ok")) if fresh is not None else True
        text = await self.phrase_result_for_call(req, text, call_meta, ok)
        # Every row this call writes -- the ones above during voice_consult and these
        # closing ones -- belong on the same thread record: the call's own identity
        # (mode "voice", this call's meta) decides record_id the same way it did then,
        # so the split-record bug (inbound on the owner thread, outbound on chat_id)
        # cannot happen here even though mode/reply_meta were restored to the prior
        # channel's values the moment the lock was released in voice_consult.
        async with self._lock:
            prev = (self.mode, self.reply_meta)
            self.mode, self.reply_meta = "voice", dict(call_meta or {})
            rid = self.record_id
            still_on_call = self.m.call_active(self.chat_id)
            if still_on_call and text:
                # Still on the call: this is spoken, not sent, so it is recorded here as
                # the one row for this result -- the only place that happens. A blank
                # text (a usage-limit failure already told once within the window --
                # item 7) means nothing is spoken and nothing is recorded here either.
                self.m.store.add_message(rid, "outbound", text, "voice")
            self.mode, self.reply_meta = prev
        if still_on_call:
            return text
        if not text:
            return ""
        # They hung up first: deliver on exactly one channel -- the owner's own primary
        # (send_to_approver, iMessage) when it was Aaron, or back to the number that
        # called otherwise -- and let that one send record the one outbound row. Two
        # earlier bugs both traced here: this function used to store a "voice" row for
        # a result that was never spoken (above), then send() recorded a second row for
        # whichever channel it actually went out on, then a third explicit add_message
        # duplicated (or mislabeled) that same row again.
        if aaron_call:
            await self.m.send_to_approver(text)
        else:
            await self.m.send(f"sms:{caller}", text, "sms", {"to": caller, "sender": caller},
                              chat_id=rid)
        return text

    async def run_consult_detailed(self, *args: Any, **kwargs: Any) -> Any:
        return await self.run_consult()

    async def close(self) -> None:
        return None

    # -- core ----------------------------------------------------------------
    def _here(self) -> Tuple[str, str]:
        """(chat_id, title) of the chat this session is: what "this chat" means in an
        access command."""
        title = " ".join(str((self.reply_meta or {}).get("group_title") or "").split())
        return self.chat_id, title or "This chat"

    PLAIN_TALK_TIMEOUT_S = 20.0

    def _running_to_stop(self, request_id: Optional[int]) -> Optional[Request]:
        """The running request a stop from Aaron means: the one he named, else the one
        running on this thread, else (in his own chat) the only one running anywhere."""
        if request_id is not None:
            req = self.m.store.get_request(request_id)
            return req if req is not None and req.state in ("approved", "running") and not req.schedule_id else None
        req = self.m.store.running_for_thread(self.chat_id)
        if req is not None:
            return req
        if self.is_approver():
            running = [r for r in self.m.store.requests_in_state("running") if not r.schedule_id]
            if len(running) == 1:
                return running[0]
        return None

    async def _maybe_stop_running(self, body: str) -> bool:
        """Aaron's exact stop words while one of his approved requests runs: stop it now,
        by code, before any lookup or model sees the message."""
        if not (self.is_approver() or self._is_owner_person()):
            return False
        matched, request_id = wording.parse_stop(body)
        if not matched:
            return False
        req = self._running_to_stop(request_id)
        if req is None:
            return False
        line = await self.m.stop_request(req)
        await self.send_to_sender(line, role="answer")
        return True

    async def _group_plain_talk(self, body: str) -> bool:
        """Group plain talk. Someone who is not Aaron says something in a group that the
        room's mode already let through: one model call with NO tools wired at all
        (Router.plain_talk posts messages only), seeing only the voice instructions, the
        room note and this room's own recent messages. A plain reply goes out with no
        approval; the sentinel, an error or a timeout returns False and the turn carries
        on exactly as before. Returns whether a reply was sent."""
        if self.is_approver() or self._is_owner_person() or not self.in_group() or self.mode == "voice":
            return False
        if not bool(_settings.get("GATE_GROUP_PLAIN_TALK")):
            return False
        if not (body or "").strip():
            return False
        # A run already going in this room, or a check-in on one, is the existing flow's to answer.
        if self.m.store.running_for_thread(self.chat_id) is not None or wording.is_fixed_progress_phrase(body):
            return False
        sentinel = Router.PLAIN_TALK_SENTINEL
        try:
            recent = [m for m in self.m.store.history(self.record_id, limit=20)
                      if m.get("kind") in ("inbound", "outbound")]
            reply = await asyncio.wait_for(
                self.m.router.plain_talk(room=self.room, history=recent[:-1], message=body,
                                         sender=self._sender_name() or ""),
                timeout=self.PLAIN_TALK_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001  (timeout or provider error: the usual flow takes it)
            logger.warning("[gate %s] plain talk failed (%s); using the usual flow", self.chat_id,
                           type(exc).__name__)
            return False
        reply = wording.plain_dashes(" ".join(str(reply or "").split()))
        if not reply or reply == sentinel or sentinel in reply:
            logger.info("[gate %s] plain talk: needs Aaron; using the usual flow", self.chat_id)
            return False
        sent = await self.send_to_sender(reply, role="answer")
        if sent:
            logger.info("[gate %s] plain talk: replied in the room without approval (inbound #%s, %d words)",
                        self.chat_id, self.inbound_id, len(reply.split()))
            self.m.store.add_message(self.record_id, "system",
                                     "Plain talk: answered in the room without Aaron's approval, no tools used.")
        return sent

    async def _route_and_act(self, body: str, system_note: Optional[str] = None) -> bool:
        approver = self.is_approver()
        if not system_note and approver:
            # Aaron changing or asking who may do what, a room's mode, or a setting, in
            # one of his own common phrasings: answered by code in one state line, before
            # any lookup, model or running request can turn it into something else.
            admin = admincmd.match(body, self.m.store)
            access = accesscmd.match(body, self.m.store) if admin is None else None
            if (admin is not None or access is not None) and self.in_group() and not (
                    access is not None and access.get("action") in ("set_mode", "show_mode")):
                # Admin output never lands in a group: one short line, nothing changed.
                logger.info("[gate %s] admin command refused in a group", self.chat_id)
                return await self.send_to_sender(chatreply.GROUP_REFUSAL, role="answer")
            if admin is not None:
                from types import SimpleNamespace
                logger.info("[gate %s] admin command: %s", self.chat_id, admin["action"])
                ctx = SimpleNamespace(store=self.m.store, chat_id=self.record_id, here=self._here(),
                                      manager=self.m, session=self,
                                      directory_client=getattr(self.m, "directory_client", None))
                reply = await admincmd.handle(ctx, admin)
                return await self.send_to_sender(chatreply.finish(reply), role="answer")
            if access is not None:
                from types import SimpleNamespace
                logger.info("[gate %s] access command: %s", self.chat_id, access["action"])
                reply = accesscmd.handle(self.m.store, SimpleNamespace(**access), here=self._here())
                return await self.send_to_sender(chatreply.finish(reply), role="answer")
        if not system_note and not approver and admincmd.refuse_for_non_owner(body, self.m.store):
            # Someone else using one of Aaron's admin phrases ("delete the interns role",
            # "show pending requests"): one plain refusal, nothing changed, and no router or
            # model run that could be talked into doing it another way.
            logger.info("[gate %s] admin phrase from a non-owner refused", self.chat_id)
            return await self.send_to_sender(chatreply.NOT_OWNER_REFUSAL, role="answer")
        if not system_note and await self._group_plain_talk(body):
            return True
        history = self.m.store.thread_history(self.thread_key(), exclude_chat=self.record_id) + \
            self.m.store.history(self.record_id, limit=20)
        if system_note:
            prior, message = history, "(No new message from the sender. A system note with a task result was just added above. Write the reply the sender should receive: for Aaron, give him the answer or outcome clearly and naturally, in his voice preference (brief, no emojis); for anyone else, tell them only what concerns them, without internal details. Reply null only if there is truly nothing to say. Do not create a request.)"
        else:
            prior, message = history[:-1], body  # the new inbound is already stored last
        # Long-term memory: facts from the graph that match this message, however old.
        # Started now so it runs alongside the ledger work below; "" when off or slow.
        self._graph_memory = ""
        recall = None if system_note or not graphmem.enabled() else asyncio.ensure_future(
            graphmem.recall_facts(body, self.record_id, approver, self.memory_groups()))
        self._recall = recall
        preflight: Optional[Dict[str, Any]] = None
        if system_note:
            # A result belongs to the task just completed, even when it has left the live index.
            ids = self.m.store.task_ids_for_chat(self.record_id, limit=3)
            memory = "\n\n".join(self.m.store.task_memory_for_task(tid) for tid in ids) or "(no tasks on record)"
        else:
            prepared = self.prepare_task_index(prior, message)
            graph_on = (os.getenv("GATE_DECIDE_GRAPH") or "1").strip().lower() not in ("0", "false", "no")
            if self.m.task_picker is not None and not self.m.router_defines_requests and graph_on:
                preflight = await self.run_preflight(body, message, prior, prepared)
                memory = await self.indexed_task_memory(
                    prior, message, prepared=prepared, pick=preflight.get("ledger", _MISSING))
                found = await self.lookup_tasks(
                    prior, message, memory,
                    plan=preflight.get("lookup", _MISSING),
                    fallback_query=preflight.get("query", _MISSING),
                    lookup_inputs=preflight["lookup_inputs"])
            else:
                memory = await self.indexed_task_memory(prior, message, prepared=prepared)
                found = await self.lookup_tasks(prior, message, memory)
        found = "" if system_note else found
        if not system_note:
            memory = self._ledger_memory
            self.log_ledger_selection(memory, found)
        if recall is not None:
            self._graph_memory = graphmem.block(await recall)
            self._recall = None
            if self._graph_memory:
                memory = f"{memory}\n\n{self._graph_memory}"
        task: Optional[Dict[str, Any]] = None
        running = None if system_note else self.m.store.running_for_thread(self.chat_id)
        if running is not None:
            # One request at a time per thread. The new message is already on the
            # thread and the task; when the running request ends, execute() re-reads
            # what arrived meanwhile (and re-decides it if the run failed). Creating a
            # second request here is what made the assistant answer twice.
            logger.info("[gate %s] request #%s still running; not starting another for this message",
                        self.chat_id, running.id)
            if approver or self._is_owner_person():
                event = (preflight or {}).get("event") or {}
                kind = event.get("raw_kind") or event.get("kind")
                conf = float(event.get("kind_conf") or 0.0)
                if kind == "declined_or_cancelled" and conf >= 0.8:
                    logger.info("[gate %s] judged a cancellation (%.2f); stopping #%s", self.chat_id, conf, running.id)
                    line = await self.m.stop_request(running)
                    return await self.send_to_sender(line, role="answer")
                if kind == "declined_or_cancelled" and conf >= 0.5 and running.id not in self.m._stop_asked:
                    self.m._stop_asked.add(running.id)
                    return await self.send_to_sender(wording.ask_stop_line(running.id), role="answer")
            t = self._peek_task(RouterOutput(task=f"T{self.m.store.task_id_for_request(running.id)}"))
            if t:
                self._ensure_inbound_on_task(t, body)
            # Code only, no model call: a progress question gets a status line built from
            # the request row; anything else keeps today's behaviour of becoming a
            # follow-up for when the run ends, but with a fixed line, never a model's
            # promise about what it is doing.
            if wording.is_progress_question(body):
                reply = self._progress_reply(running)
            else:
                reply = wording.NOT_PROMISING_FOLLOWUP
            if reply:
                await self.send_to_sender(reply, role="ack")
            return bool(reply)
        if not system_note and wording.is_fixed_progress_phrase(body):
            # Nothing is running, but something on this thread finished or failed
            # recently enough that a fixed, unambiguous check-in phrase ("status",
            # "any update?") almost certainly means IT, not a brand new ask: answer
            # from that row and stop here. Only the fixed phrases, never the looser
            # "ends in a question mark" guess -- that guess is safe only while a
            # request is still running and nothing else could be meant; once it is
            # over, a question-shaped message is just as likely a genuinely new ask.
            finished = self.m.store.last_finished_for_thread(self.chat_id)
            if finished is not None:
                t = self._peek_task(RouterOutput(task=f"T{self.m.store.task_id_for_request(finished.id)}"))
                if t:
                    self._ensure_inbound_on_task(t, body)
                reply = self._finished_progress_reply(finished)
                if reply:
                    await self.send_to_sender(reply, role="ack")
                return bool(reply)
        if system_note:
            answer_started = time.monotonic()
            answer_request_id = _request_id_from_note(system_note)
            rewrite_task = asyncio.create_task(
                self._grounding_rewrite(prior, memory, found, approver))
            try:
                out: RouterOutput = await self.m.router.route(
                    history=prior, message=message, mode=self.mode, sender=self._sender(),
                    contact_notes=self._contact_notes(), is_approver=approver,
                    task_memory=memory, found_tasks=found, action=False,
                )
                out.request = None
                out.reply = await self.ground_reply(
                    out.reply, system_note, prior, memory, found, approver,
                    rewrite_task=rewrite_task, answer_started=answer_started,
                    answer_request_id=answer_request_id)
            finally:
                if not rewrite_task.done():
                    rewrite_task.cancel()
                await asyncio.gather(rewrite_task, return_exceptions=True)
        else:
            out, task = await self.decide(body=body, message=message, prior=prior, mode=self.mode,
                                          memory=memory, found=found, preflight=preflight)
        if not system_note and out.schedule is not None:
            if not approver:
                out.request = None
                out.schedule = None
                return await self.send_to_sender("Only Aaron can create or change a schedule.", role="answer")
            if task is None:
                task = self.m.store.create_task(out.schedule.title)
                self._ensure_inbound_on_task(task, body)
            # Permission is chosen with no probability judgment at all: scopes he
            # named, scopes his own words name by synonym, scopes the prompt names
            # its own tools by, and the router's suggestion -- never Jev.
            said = schedprop.strip_group_tag(message)
            try:
                scopes = schedprop.capabilities_for(
                    out.schedule.prompt, [], out.schedule.scopes, said)
                # Set up in a group: its runs post in that group, which needs no scope.
                scopes = schedprop.group_scopes(scopes, said, out.schedule.prompt, self.reply_meta)
            except schedprop.UnknownScopes as exc:
                out.request = None
                out.schedule = None
                return await self.send_to_sender(schedprop.friendly_error(exc), role="answer")
            if not scopes:
                # Nothing named it, nothing in his words matched a synonym or
                # label, the prompt names no tool literally, and the router
                # suggested no scope the work needs: a proposal with no
                # capabilities would read "It will ." and be refused every
                # tool on its first run. Ask instead of proposing.
                out.request = None
                out.schedule = None
                return await self.send_to_sender(
                    "Tell me what it should be allowed to do, for example read public web pages.",
                    role="answer")
            try:
                until_epoch = _parse_until(out.schedule.until, out.schedule.timezone)
            except UntilParseError as exc:
                out.request = None
                out.schedule = None
                return await self.send_to_sender(str(exc), role="answer")
            # A start or stop he stated is read from his own words first, then from the
            # router's fields: "starting 9am today" must never become "starting 1:45 AM".
            start_at = schedprop.parse_start(body, out.schedule.timezone)
            if start_at is None and out.schedule.start_at:
                try:
                    start_dt = datetime.fromisoformat(out.schedule.start_at.strip())
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=ZoneInfo(out.schedule.timezone or "America/New_York"))
                    start_at = start_dt.timestamp()
                except (ValueError, TypeError):
                    start_at = None
            stop_when = (out.schedule.stop_when or "").strip() or schedprop.parse_stop_when(body)
            try:
                await self.m.schedules.propose(
                    owner=True, chat_id=self.record_id, task_id=int(task["id"]),
                    title=out.schedule.title, prompt=out.schedule.prompt,
                    cadence_kind=out.schedule.cadence_kind, cadence=out.schedule.cadence,
                    timezone=out.schedule.timezone, report_every_time=out.schedule.report_every_time,
                    until=until_epoch, scopes=scopes, start_now=out.schedule.start_now,
                    auto_start=starts_unasked(out.schedule.cadence_kind, scopes) and not stop_when,
                    start_at=start_at, stop_when=stop_when or None,
                    origin_mode=self.mode,
                    origin_chat_id=json.dumps({"chat_id": self.chat_id, "meta": self.reply_meta or {}}),
                )
            except ValueError as exc:
                out.request = None
                out.schedule = None
                return await self.send_to_sender(schedprop.friendly_error(exc), role="answer")
            return True
        if not system_note and getattr(out, "schedule_command", None) is not None:
            if not approver:
                out.request = None
                out.schedule_command = None
                return await self.send_to_sender("Only Aaron can manage schedules.", role="answer")
            from . import schedcmd
            reply = await schedcmd.handle(self.m, self, out.schedule_command)
            return await self.send_to_sender(chatreply.finish(reply), role="answer")
        if not system_note and getattr(out, "access_command", None) is not None:
            out.request = None
            if not approver:
                out.access_command = None
                return await self.send_to_sender("Only Aaron can change who can do what.", role="answer")
            if self.in_group() and getattr(out.access_command, "action", "") not in ("set_mode", "show_mode"):
                return await self.send_to_sender(chatreply.GROUP_REFUSAL, role="answer")
            reply = accesscmd.handle(self.m.store, out.access_command, here=self._here())
            return await self.send_to_sender(chatreply.finish(reply), role="answer")
        if out.request is None:
            if out.reply:
                return await self.send_to_sender(out.reply, role="answer")
            return False
        # Validate scopes (pydantic already rejected unknown ones).
        if self.m.store.pending_for_thread(self.chat_id, sender=self._sender()) and not approver:
            self.m.store.add_message(self.record_id, "system", "A request is already awaiting Aaron; new request not created.")
            if out.reply:
                return await self.send_to_sender(out.reply, role="ack")
            return False
        assert task is not None
        self.apply_knowledge_rule(out)
        # Someone Aaron has trusted for exactly these scopes does not need him woken up.
        # Anyone else, and any scope outside what they were given, still waits for his yes.
        trusted = False
        if not approver:
            t = self.trust()
            allowed = set(t.get("scopes") or [])
            trusted = bool(allowed) and set(out.request.scopes or []) <= allowed
            if trusted:
                logger.info("[gate %s] %s is trusted for %s; running without asking Aaron",
                            self.chat_id, t.get("person") or self._sender(), sorted(out.request.scopes))
        req, created = self.m.store.create_request_once(
            chat_id=self.chat_id, sender=self._sender(), sender_name=self._sender_name(), mode=self.mode,
            subject=str(self.reply_meta.get("subject") or ""), original_message=body,
            summary=ownertext.strip_group_tag(out.request.summary) or out.request.summary,
            scopes=out.request.scopes, prompt=out.request.prompt,
            state="approved" if (approver or trusted) else "pending", task_id=task["id"],
            inbound_id=self.inbound_id,
        )
        if not created:
            # Another copy of this ask on this thread already has a recent request:
            # its ack is never sent, it is never executed again, and its scopes are
            # never widened -- all three
            # belong to whichever turn actually created the row this one found.
            logger.info("[gate %s] request deduped onto #%s for sender %s; not acknowledging or running again",
                        self.chat_id, req.id, self._sender())
            return False
        # Alongside a request the reply is an acknowledgement, sent only now that
        # this turn is known to own a genuinely new request.
        sent_reply = await self.send_to_sender(out.reply, role="ack") if out.reply else False
        self.m.remember_knowledge_notes(req.id, getattr(out.request, "knowledge_notes", None))
        if preflight is not None:
            self.m.remember_complexity(req.id, preflight.get("complexity"))
        self.m.record_request_on_task(req, task, self)
        if approver or trusted:
            # Run after this turn releases the session lock; execute() will
            # re-enter the session to phrase the result.
            asyncio.create_task(self.m.execute(req))
            return True
        self.m.store.set_thread(self.chat_id, "awaiting_aaron")
        await self.m.send_approval_text(req)
        if not sent_reply:
            await self.send_to_sender(HANDOFF_LINE, role="ack")
        return True

    inbound_id: Optional[int] = None   # the inbound message this turn is responding to

    def _finished_progress_reply(self, req: Request) -> str:
        """The answer to a progress question ("did you start?", "any update?") that
        arrives after the request it asks about has already finished or failed --
        read straight from the row, never by deciding the message as a new ask."""
        status = req.status or {}
        if req.state == "done":
            return wording.success_message(_clean_result(status.get("summary") or ""))
        return f"That one failed: {wording.failure_in_words(status)}."

    def _progress_reply(self, running: Request) -> str:
        """The status line for a message that asked about the still-running request
        named by `running`. Re-reads the request fresh: it may have finished or failed
        in the moment between running_for_thread's check and this turn being handled,
        and that gets the same wording a finished request's result ever gets, not a
        claim that it is still going."""
        fresh = self.m.store.get_request(running.id) or running
        fresh_status = fresh.status or {}
        if fresh.state == "done":
            return wording.success_message(_clean_result(fresh_status.get("summary") or ""))
        if fresh.state == "failed":
            return wording.failure_message(wording.plain_title(self.m.store, fresh), fresh_status)
        step = wording.latest_step_from_tool_calls(fresh_status.get("tool_calls"))
        return wording.progress_status_line(fresh.created_at, step, fresh.summary)


    async def send_to_sender(self, text: str, role: str = "answer") -> bool:
        """Deliver to the sender. Enforced: per inbound message at most one 'ack' and
        one 'answer' go out. A second of either is refused here, whatever produced it.
        Returns whether it was sent."""
        given = self.m.store.responses_to(self.inbound_id or 0)
        if self.inbound_id and role in given:
            logger.warning("[gate %s] refused a second %s for inbound #%s: %r", self.chat_id, role, self.inbound_id, text)
            return False
        if self.mode == "voice" and not self.m.call_active(self.chat_id):
            # The call ended before this was ready: deliver by text instead, on exactly
            # one channel -- the owner's own primary (send_to_approver) when it was
            # Aaron, or back to the number that called otherwise -- and record exactly
            # one row for it. send_to_approver already records its own row (imessage);
            # recording a second one below under mode "voice" (never spoken) was the
            # voice+imessage double-row bug.
            if self.is_aaron_on_phone():
                await self.m.send_to_approver(text, kind=getattr(self, "_outbox_kind", "reply"))
            else:
                num = self._sender()
                await self.m.send(f"sms:{num}", text, "sms", {"to": num, "sender": num},
                                  record=False, kind=getattr(self, "_outbox_kind", "reply"))
                self.m.store.add_message(self.record_id, "outbound", text, "sms", reply_to=self.inbound_id, role=role)
        else:
            self.typing_stop()
            result = await self.m.send(self.chat_id, text, self.mode, self.reply_meta, record=False,
                                       kind=getattr(self, "_outbox_kind", "reply"),
                                       hold_exempt=(role == "ack"))
            if result is HELD_FOR_AARON:
                # Not sent: held for Aaron's yes on the exact words (manager.hold_message).
                self.m.store.add_message(self.record_id, "system",
                                         f"Reply held for Aaron's yes before it goes out: {text}")
                return True
            self.m.store.add_message(self.record_id, "outbound", text, self.mode, reply_to=self.inbound_id, role=role)
        t = self.task()
        if t is not None:
            self.m.store.task_event(t["id"], "outbound", text, chat_id=self.record_id)
        return True

    async def _grounding_rewrite(self, prior: List[Dict[str, Any]], memory: str,
                                 found: str, approver: bool) -> RouterOutput:
        strict = ("(GROUNDING: your previous draft stated something the task result does not support. Fill "
                  "\"reasoning\" first: what the result establishes, what follows from it by ordinary inference, "
                  "what stays unknown. Then write the reply as a conclusion from that: answer the question directly, "
                  "name only what the result contains or what plainly follows from it, mark inferences as inferences, "
                  "and say what stays unknown. Do not create a request.)")
        return await self.m.router.route(
            history=prior + [{"kind": "system", "text": strict}], message=strict, mode=self.mode, sender=self._sender(),
            contact_notes=self._contact_notes(), is_approver=approver, task_memory=memory, found_tasks=found, action=False)

    async def ground_reply(self, reply: Optional[str], note: str, prior: List[Dict[str, Any]], memory: str,
                           found: str, approver: bool, *, rewrite_task: Optional[asyncio.Task] = None,
                           answer_started: Optional[float] = None,
                           answer_request_id: Optional[int] = None) -> Optional[str]:
        """A result reply may only state what the results support. Jev checks the
        draft against the tool output; an unsupported draft is rewritten once with
        the results as the only source, and if still unsupported, replaced by a
        plain statement of what was and was not found. Never a confident guess."""
        picker = self.m.task_picker
        if not reply or picker is None or "Result:" not in note:
            if rewrite_task is not None:
                if not rewrite_task.done():
                    rewrite_task.cancel()
                await asyncio.gather(rewrite_task, return_exceptions=True)
            return reply
        started = answer_started if answer_started is not None else time.monotonic()
        request_id = answer_request_id if answer_request_id is not None else _request_id_from_note(note)
        results = note.split("Result:", 1)[1].strip()
        question = ""
        for m in reversed(prior):
            if m.get("kind") == "inbound":
                question = str(m.get("text") or "")
                break
        res = await picker.judge_grounded(reply=reply, results=results, question=question)
        if res.get("grounded") is not False:
            if rewrite_task is not None:
                if not rewrite_task.done():
                    rewrite_task.cancel()
                await asyncio.gather(rewrite_task, return_exceptions=True)
            _log_answer_path(request_id, "grounded", started)
            return reply
        logger.info("[gate %s] reply not grounded (p=%.2f); rewriting from the results only", self.chat_id, res["p"])
        out2 = await rewrite_task if rewrite_task is not None else await self._grounding_rewrite(
            prior, memory, found, approver)
        reply2 = out2.reply or reply
        res2 = await picker.judge_grounded(reply=reply2, results=results, question=question)
        p2 = float(res2.get("p") or 0.0)
        if res2.get("grounded") is not False or p2 >= 0.35:
            # A rewrite from the results only, near or over the line: send it. Derived
            # answers (free slots from a list of events, a count from rows) score low
            # on literal support even when they are right.
            _log_answer_path(request_id, "rewritten", started)
            return reply2
        # Twice unsupported. Sending it anyway is the one thing worse than saying nothing:
        # it is a confident answer the check already judged to be unbacked.
        logger.warning("[gate %s] rewrite still not grounded (p=%.2f); refusing to pass it off",
                       self.chat_id, p2)
        _log_answer_path(request_id, "refused", started)
        if approver:
            # For the verified Aaron: hand over what the run actually produced, and say
            # plainly that it did not answer the question.
            body = _result_bodies(results)
            if body.strip() and len(body) <= 1500:
                return body
            # Too long to hand over raw, and dumping tool output is not an answer either.
            return ("I could not put an answer together from what came back, and I am not going to "
                    "guess at one. The full result is on the task. Tell me to try again and I will "
                    "run it fresh.")
        if self._is_owner_person():
            # Looks like Aaron (his number, or a contact record that resolves to him)
            # but not verified -- is_approver() stays False and no scope or power widens.
            # He still hears it in the first person, about himself, not a note that
            # someone else "let him know": there is no one else here to tell.
            return "I don't have an answer for that yet."
        # Someone else entirely never sees the run's own words: no mention of tools,
        # scopes, approval or why it failed. Aaron gets the full result separately (the
        # note this call was given, in full, on his own thread).
        return ("I do not have an answer ready for that right now. I have let Aaron "
                "know, and he will follow up if there is more to share.")

    async def handle_followup(self, body: str, inbound_id: Optional[int] = None) -> bool:
        """Re-decide a message that was received while a request was running. The
        message is already stored on the thread, so this only routes and acts."""
        async with self._lock:
            if inbound_id:
                self.inbound_id = inbound_id
            return await self._route_and_act(body)

    async def notify_failure(self, message: str, note: str, inbound_id: Optional[int] = None) -> bool:
        """A failed request's message to the person is always the wording.py text, exactly
        as written -- never phrased or rewritten by a model (the raw status in `note`
        still lands on the ledger/log, just never in front of a person). Same per-inbound
        at-most-one-answer guarantee as notify_after_request."""
        async with self._lock:
            if inbound_id:
                self.inbound_id = inbound_id
            self.m.store.add_message(self.record_id, "system", note)
            if self.inbound_id and "answer" in self.m.store.responses_to(self.inbound_id):
                logger.warning("[gate %s] inbound #%s already answered; failure result not delivered again",
                               self.chat_id, self.inbound_id)
                return True
            # "Still working, I'll text you the answer" promises an answer; it is not one.
            # Filed as an answer it made the real result, when the run finished, look like
            # a second answer to the same message, and the result was never sent.
            role = "ack" if message == wording.still_working_message() else "answer"
            return await self.send_to_sender(message, role=role)

    async def notify_after_request(self, note: str, inbound_id: Optional[int] = None) -> bool:
        """Phrase a request's result for the sender. The result is THE answer to the
        inbound that raised the request; if that inbound already has an answer the
        send is refused (returns True: handled, nothing more to deliver)."""
        async with self._lock:
            if inbound_id:
                self.inbound_id = inbound_id
            self.m.store.add_message(self.record_id, "system", note)
            if self.inbound_id and "answer" in self.m.store.responses_to(self.inbound_id):
                logger.warning("[gate %s] inbound #%s already answered; result of a request not delivered again",
                               self.chat_id, self.inbound_id)
                return True
            return await self._route_and_act("", system_note=note)


ACK_MAX_WORDS = 20


_PROMISE_RE = re.compile(r"\b(on it|i will (check|pull|look|submit|resubmit|run|send|book|pull)|submitting|resubmitting|"
                         r"checking now|pulling|looking (it |that )?up now|running it now|let me (check|pull|look))\b", re.I)


def _promises_action(text: str) -> bool:
    """A reply that announces work the assistant is about to do."""
    return bool(_PROMISE_RE.search(text or ""))


def _result_bodies(results: str) -> str:
    """Tool output only: drop the agent's own header, tool labels and status line."""
    keep = []
    for ln in (results or "").splitlines():
        t = ln.strip()
        if t.startswith(("Done via Jev agent", "Failed via Jev agent", "STATUS:", "Reason:")):
            continue
        if re.fullmatch(r"- [a-z_]+:", t):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()


def _is_acknowledgement(text: str) -> bool:
    """Short, forward-looking, no facts: "On it, pulling that now." Anything longer,
    or containing a question, is an answer and is not sent while a request runs."""
    t = (text or "").strip()
    return 0 < len(t.split()) <= ACK_MAX_WORDS and "?" not in t


_SHORT_REPLY = re.compile(r"^\W*(the former|the latter|the first|the second|yes|yeah|yep|sure|ok|okay|"
                          r"go ahead|do it|please do|both|all of them|that one|this one|number \d+|\d+)"
                          r"\W*(please|thanks|thank you|pls)?\W*$", re.I)


def _what_it_answers(body: str, history: List[Dict[str, Any]]) -> str:
    """A message like "the former please" carries none of its own meaning. The question it
    answers is the last thing the assistant said, so it is quoted beside it: without that,
    the only thing left to read is the task record, which describes what was asked before."""
    if not _SHORT_REPLY.match(" ".join((body or "").split())):
        return ""
    for row in reversed(history or []):
        if row.get("kind") == "outbound":
            asked = " ".join(str(row.get("text") or "").split())
            if asked:
                return (f"\nThis is an answer to the last thing you said to them, which was:\n"
                        f"  {asked}\nAct on what that question offered, not on anything older.\n")
    return ""


def _last_failed_run(store: Any, task: Optional[Dict[str, Any]]) -> str:
    """What the task's most recent run said when it failed, or "" when its last run
    succeeded or nothing has run. Read from the store, since the task dict a turn
    carries does not always come with its events."""
    tid = int((task or {}).get("id") or 0)
    if not tid:
        return ""
    try:
        full = store.task_with_events(tid, limit=12) or {}
    except Exception:
        return ""
    for e in reversed(full.get("events") or []):
        kind = e.get("kind")
        if kind == "failed":
            return " ".join(str(e.get("text") or "").split())
        if kind == "done":
            return ""
    return ""


def _what_it_supplies(body: str, last_failed: str) -> str:
    """The last run on this task stopped short, and here is a new message that is not a
    bare yes or no. Most often it is the sender handing over what that run said was
    missing: the report pasted in, the address, the name. Said so beside the message,
    so the run reads it as the answer to its own failure rather than as a fresh
    remark to acknowledge. (A short reply is _what_it_answers' case, not this one.)"""
    if not last_failed:
        return ""
    if _SHORT_REPLY.match(" ".join((body or "").split())):
        return ""
    said = last_failed if len(last_failed) <= 500 else last_failed[:500] + "..."
    return ("\nThe last run on this task stopped short. It said:\n"
            f"  {said}\n"
            "If this message supplies what that run lacked (content pasted in, a document, a name, an "
            "address, a date, a choice), it is the answer to that failure: carry out the original request "
            "with it now rather than acknowledging it. If it only asks whether the work was done, answer "
            "that plainly and say what is now in hand.\n")


def _owner_request(manager: Any, req: Request) -> bool:
    """Whether a request came from the person the assistant works for. Read through the
    manager when it can answer, so a stand-in without the machinery simply says no."""
    fn = getattr(manager, "_is_owner_request", None)
    try:
        return bool(fn(req)) if callable(fn) else False
    except Exception:
        return False


_ACTION_WORDS = ("delete", "remove", "trash", "cancel", "replace", "clear", "purge", "archive",
                 "move", "create", "update", "add", "restart", "stop", "deploy", "publish", "submit")


def _derive_action_word(short: str, shown: Dict[str, Any]) -> str:
    """A plain English verb for a held destructive call, when one is plainly there:
    the call's own `action` argument (the common "manage_X" shape -- see is_destructive
    in jevagent.py, which already reads this same field) when it's a single recognized
    word, else a recognized verb token in the tool's own short name. Empty when neither
    names one -- the caller falls back to its own wording rather than guessing."""
    act = str(shown.get("action") or "").strip().lower()
    if act in _ACTION_WORDS:
        return act
    tokens = short.lower().split("_")
    for word in _ACTION_WORDS:
        if word in tokens:
            return word
    return ""


def _draft_text(args: Dict[str, Any]) -> str:
    """One outbound message, laid out as the person reading it would want to check it:
    who it reaches first, then what it says, whole and unedited."""
    def val(*keys: str) -> str:
        for k in keys:
            v = args.get(k)
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v if str(x).strip())
            v = str(v or "").strip()
            if v:
                return v
        return ""
    lines = []
    for label, keys in (("To", ("to", "recipient", "recipients", "to_email", "email", "phone", "number")),
                        ("Cc", ("cc",)), ("Bcc", ("bcc",)), ("Subject", ("subject", "title"))):
        got = val(*keys)
        if got:
            lines.append(f"{label}: {got}")
    body = val("body", "message", "text", "content", "caption")
    out = "\n".join(lines)
    if body:
        out += ("\n\n" if out else "") + body
    return out or "There's a message to send, but I couldn't lay out its parts plainly."


class GateSessionManager:
    def __init__(self, *, cfg: Any, send_fn: SendFn, mcp_server: Any, identity_info: Dict[str, str],
                 store_path: str, exec_cwd: str, standing_path: Optional[str] = None,
                 typing_fn: Optional[Callable[..., Awaitable[None]]] = None):
        self.cfg = cfg
        self.send_fn = send_fn
        self.typing_fn = typing_fn            # iMessage typing indicator, same signature as send_fn minus text
        self.identity_info = identity_info
        self.store = Store(store_path)
        # Knobs read the database from here on, so a change in the console takes effect on
        # the next turn rather than the next restart.
        _settings.bind(self.store)
        self.router = Router(api_key=cfg.deepseek_api_key, model=cfg.deepseek_model, standing_path=standing_path)
        from .. import telegram as _tg
        # Every handle the gate already treats as the owner's own: the protected list a
        # scheduled run's sends are checked against is exactly this, nothing wider.
        protected = [str(cfg.approver_imessage_conversation_id or ""), str(getattr(cfg, "approver_phone", "") or ""),
                     str(_tg.approver_id() or "")]
        # manager=self on all three: lets the executor/agent reach manager.store and
        # manager.schedules for Blatbot's own ADMIN_* tools (gate/hosttools.py). Every
        # one of those calls still checks is_commander itself before doing anything.
        self.executor = Executor(mcp_server=mcp_server, cwd=exec_cwd, model=cfg.claude_model or "sonnet",
                                 protected=protected, manager=self)
        # GATE_EXECUTOR=jev runs requests through the typed-judgment agent
        # (Jev picks tools, code fills arguments, DeepSeek writes prose only when
        # needed); Claude Code stays as the fallback unless GATE_EXECUTOR_FALLBACK=none.
        self.jev_agent: Optional[JevAgent] = JevAgent(inkbox_server=mcp_server, router=self.router,
                                                      protected=protected, manager=self) if jev_agent_enabled() else None
        # One confirmed call has to be runnable whoever the executor is: when the owner has
        # read a draft and said yes, the same call goes out, and nothing else does.
        self.caller: JevAgent = self.jev_agent or JevAgent(inkbox_server=mcp_server, router=self.router,
                                                           protected=protected, manager=self)
        self.jev_fallback = (os.getenv("GATE_EXECUTOR_FALLBACK") or "claude").strip().lower() != "none"
        logger.info("[gate] executor: %s", "jev agent (fallback %s)" % ("claude" if self.jev_fallback else "none") if self.jev_agent else "claude code")
        self.task_picker: Optional[TaskPicker] = TaskPicker() if jev_enabled() else None
        # With Jev on, the request is built by code from the task record (title, ledger,
        # Jev scopes); DeepSeek only writes replies. GATE_ROUTER_DEFINES_REQUEST=1 restores
        # the older flow where the router proposes the request and Jev checks it.
        self.router_defines_requests = self.task_picker is None or \
            str(os.getenv("GATE_ROUTER_DEFINES_REQUEST") or "").strip().lower() in ("1", "true", "yes")
        logger.info("[gate] task picker: %s", "jev (TypeSafe)" if self.task_picker else "router")
        self.approver_conv = cfg.approver_imessage_conversation_id
        # Aaron can stop a running request: the asyncio task each top-level run is on,
        # by request id, and the requests he has already been asked "stop #N?" about.
        self._run_tasks: Dict[int, asyncio.Task] = {}
        self._stop_asked: set = set()
        runguard.reset()
        runguard.set_store_check(self.store.request_cancelled)
        self.approver_phone = str(getattr(cfg, "approver_phone", "") or "")
        self.approver_telegram_id = _tg.approver_id()
        # The commander role's one legitimate assignment: tell the store the owner's two
        # verified handles (never logged, never read from the environment by the store
        # itself) and have it resolve/pin who holds the role, then reconcile stray data.
        self.store.set_approver_handles(telegram_id=self.approver_telegram_id or "",
                                        phone=self.approver_phone or "")
        try:
            self.store.reconcile_commander()
        except Exception:
            logger.exception("[gate] commander reconciliation failed at start-up")
        self.voice_vocabulary = str(os.getenv("GATE_VOICE_VOCABULARY") or DEFAULT_VOICE_VOCABULARY).strip()
        # read per turn, not captured here: the console can change it between messages
        self._voice_trust_override: Optional[bool] = None
        self.sessions: Dict[str, GateSession] = {}
        # request id -> the notes the vault traversal chose, most relevant first. Held in
        # memory only: after a restart a run falls back to its knowledge scopes' own notes.
        self._knowledge_notes: Dict[int, List[str]] = {}
        # request id -> the source-only model-complexity judgment made in the turn's
        # preflight. Scheduled/restarted work has no entry and judges at execution time.
        self._complexity_results: Dict[int, Dict[str, Any]] = {}
        # Request ids a blanket "you have my permission for everything" (item 4) was
        # said about while still running: applied to that request's own retry, once,
        # the moment it actually fails and shows what it was refused. Held in memory
        # only -- a restart loses an in-flight blanket grant, same as other in-flight
        # judgments here (_knowledge_notes, _voice_trust_override).
        self._blanket_pending: set = set()
        self.schedules = ScheduleService(self)
        # People auto-indexing: one pending notice at a time, so everyone met within
        # PEOPLE_NOTICE_DELAY goes to Aaron in one text. What he was told lives on
        # disk (persons.told); this task is only the timer.
        self._people_notice_task: Optional[asyncio.Task] = None
        self._people_notice_lock = asyncio.Lock()
        self._expiry_task: Optional[asyncio.Task] = None
        self._schedule_task: Optional[asyncio.Task] = None
        self._followup_task: Optional[asyncio.Task] = None
        self._outbox_task: Optional[asyncio.Task] = None
        self._outbox_wake = asyncio.Event()
        self._outbox_waiters: Dict[int, asyncio.Future] = {}
        self._outbox_retry_policy: Dict[int, bool] = {}
        self._outbox_payloads: Dict[int, Dict[str, Any]] = {}
        self._outbox_target_tasks: Dict[str, asyncio.Task] = {}
        self._outbox_dirty_targets: set[str] = set()
        self._outbox_dispatch_lock = asyncio.Lock()
        if self._outbox_enabled():
            try:
                interrupted = self.store.fail_interrupted_outbox()
                if interrupted:
                    logger.warning("[gate] outbox marked %d interrupted sends failed; not retried", interrupted)
                stale = self.store.fail_old_queued_outbox(time.time() - self._outbox_max_age())
                if stale:
                    logger.warning("[gate] outbox marked %d old queued sends failed", stale)
            except Exception as exc:
                logger.warning("[gate] outbox startup recovery failed: %s", exc)
        try:
            self._expiry_task = asyncio.get_running_loop().create_task(self._expiry_loop())
            self._schedule_task = asyncio.get_running_loop().create_task(self._schedule_loop())
            self._followup_task = asyncio.get_running_loop().create_task(self._followup_loop())
            if self._outbox_enabled():
                self._outbox_task = asyncio.get_running_loop().create_task(self._outbox_loop())
        except RuntimeError:
            pass

    def call_active(self, chat_id: str) -> bool:
        gw = getattr(self.send_fn, "__self__", None)
        return chat_id in (getattr(gw, "_active_call_ws", None) or {})

    def get(self, chat_id: str, system_prompt_extra: str = "") -> GateSession:
        s = self.sessions.get(chat_id)
        if s is None:
            s = GateSession(chat_id, self)
            self.sessions[chat_id] = s
        return s

    async def close_all(self) -> None:
        if self._expiry_task:
            self._expiry_task.cancel()
        if self._schedule_task:
            self._schedule_task.cancel()
        if self._followup_task:
            self._followup_task.cancel()
        if self._outbox_task:
            self._outbox_task.cancel()
        for task in self._outbox_target_tasks.values():
            task.cancel()

    # -- approver channel ----------------------------------------------------
    @property
    def voice_trust_approver(self) -> bool:
        """Whether a call from the owner's number acts directly. Read when the call happens,
        so turning it off in the console applies to the next call rather than the next boot."""
        if self._voice_trust_override is not None:
            return self._voice_trust_override
        return bool(_settings.get("GATE_VOICE_TRUST_APPROVER"))

    @voice_trust_approver.setter
    def voice_trust_approver(self, value: bool) -> None:
        """Set directly only by tests, which need a fixed answer rather than a live one."""
        self._voice_trust_override = bool(value)

    async def send(self, target: str, text: str, mode: str, meta: Dict[str, Any], *,
                   chat_id: str = "", role: str = "notice", reply_to: Optional[int] = None,
                   record: bool = True, kind: str = "reply",
                   retry_on_failure: bool = True, hold_exempt: bool = False) -> Any:
        """Save a message and wait for the dispatcher to deliver it.
        Callers that write their own reply record pass record=False.

        The one place every gateway send passes: a send made on behalf of a running
        request (runguard) to anyone but Aaron is not sent but held for his exact yes
        (GATE_HOLD_SENDS_TO_OTHERS); hold_exempt is for the short holding notice only."""
        text = wording.plain_dashes(text)
        if mode != "email":
            # The name-and-title signature is for email only. The gateway strips it at
            # send too; stripping here keeps the outbox row, the held copy and the
            # recorded message identical to what actually goes out.
            from ..signoff import strip_signoff
            text = strip_signoff(text)
        if self._is_owner_target(target, meta):
            # Aaron, 2026-11-18: no decorative symbols in anything Blatbot sends him.
            text = chatreply.ORNAMENT_RE.sub("", text).replace(" \n", "\n")
        running_for = runguard.current()
        if running_for is not None and runguard.is_cancelled(running_for):
            # A send made on behalf of a request Aaron has stopped: refused, right here,
            # whatever produced it.
            logger.warning("[gate] refused a %s send to %s: request #%s was stopped by Aaron",
                           mode, target, running_for)
            return None
        if (running_for is not None and not hold_exempt and mode != "voice"
                and bool(_settings.get("GATE_HOLD_SENDS_TO_OTHERS"))
                and not self._is_owner_target(target, meta)):
            origin = self.store.get_request(running_for)
            if (origin is not None and not origin.schedule_id
                    and not self._answers_owner_in_place(origin, target, chat_id)):
                await self.hold_message(origin, target, text, mode, meta, chat_id=chat_id, kind=kind)
                return HELD_FOR_AARON
        result = None
        if not self._outbox_enabled() or mode == "voice":
            result = await self.send_fn(target, text, mode, meta)
        else:
            row_id = None
            try:
                row_id = self.store.add_outbox(chat_id or self._threads_for(target)[0], target,
                                               mode, text, meta, kind)
                loop = asyncio.get_running_loop()
                if self._outbox_task is None or self._outbox_task.done() or self._outbox_task.get_loop() is not loop:
                    # A manager made outside a running loop still needs its worker.
                    self._outbox_wake = asyncio.Event()
                    self._outbox_dispatch_lock = asyncio.Lock()
                    self._outbox_target_tasks = {}
                    self._outbox_task = loop.create_task(self._outbox_loop())
                waiter = loop.create_future()
                self._outbox_waiters[row_id] = waiter
                self._outbox_retry_policy[row_id] = retry_on_failure
                self._outbox_payloads[row_id] = {
                    "id": row_id, "target": target, "text": text, "mode": mode,
                    "meta_json": json.dumps(meta), "kind": kind,
                }
                self._outbox_dirty_targets.add(target)
                self._outbox_wake.set()
            except Exception as exc:
                logger.warning("[gate] outbox unavailable, sending directly: %s", exc)
                if row_id is not None:
                    try:
                        self.store.abandon_outbox(row_id)
                    except Exception:
                        pass
                result = await self.send_fn(target, text, mode, meta)
            else:
                # The dispatcher always answers. This ceiling only stops a turn from hanging
                # forever if it somehow does not; the row stays with the dispatcher, so
                # nothing is sent twice. A timeout raised by the channel itself comes
                # through the waiter untouched.
                finished, _ = await asyncio.wait({waiter}, timeout=90)
                if not finished:
                    self._outbox_waiters.pop(row_id, None)
                    logger.warning("[gate] outbox row %s got no delivery result in 90 s", row_id)
                    raise TimeoutError("the message was queued but its delivery was not confirmed")
                result = waiter.result()
                if result is _OUTBOX_HELD:
                    return None
        if running_for is not None and mode != "voice":
            runguard.record_sent(f"{mode} to {target}", running_for)
        if not record:
            return result      # the caller writes its own line, with the turn's reply_to and role
        for cid in self._threads_for(target, chat_id):
            try:
                self.store.add_message(cid, "outbound", text, mode, reply_to=reply_to, role=role)
            except Exception:
                logger.debug("[gate] could not record an outbound on %s", cid, exc_info=True)
        return result

    @staticmethod
    def _outbox_enabled() -> bool:
        return os.getenv("GATE_OUTBOX", "1").strip().lower() not in ("0", "false", "no")

    @staticmethod
    def _proactive_limit() -> int:
        try:
            return max(0, int(os.getenv("OUTBOX_PROACTIVE_PER_HOUR", "6")))
        except ValueError:
            return 6

    @staticmethod
    def _outbox_max_age() -> float:
        try:
            return max(0.0, float(os.getenv("OUTBOX_MAX_AGE_SECONDS", "600")))
        except ValueError:
            return 600.0

    @staticmethod
    def _outbox_transient(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status = (getattr(exc, "status", None) or getattr(exc, "status_code", None)
                  or getattr(response, "status_code", None) or getattr(response, "status", None))
        try:
            status = int(status)
        except (TypeError, ValueError):
            match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
            status = int(match.group(1)) if match else None
        if status is not None:
            return status == 429 or status >= 500
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        message = str(exc).lower()
        return any(word in message for word in ("timeout", "timed out", "connection", "rate limit", "network"))

    async def _deliver_outbox(self, row_id: int, *, retry_on_failure: bool = True,
                              row: Optional[Dict[str, Any]] = None) -> Any:
        row = row or self.store.get_outbox(row_id)
        if row is None:
            return
        try:
            # An answer, and a question that is waiting on his yes or no, always go out:
            # dropping a repeated ask as a duplicate would leave a request waiting on a
            # question he never received.
            if row["kind"] not in ("reply", "approval"):
                try:
                    recent = self.store.recent_outbox_sends(row["target"], time.time() - 600, row["mode"])
                except Exception as exc:
                    logger.warning("[gate] outbox duplicate check unavailable: %s", exc)
                    recent = []
                if any(other["text"] == row["text"] and other["last_error"] != "duplicate suppressed"
                       for other in recent):
                    try:
                        self.store.set_outbox_state(row_id, "sent", error="duplicate suppressed")
                    except Exception as exc:
                        logger.warning("[gate] outbox could not record duplicate: %s", exc)
                    return
            result = await self.send_fn(row["target"], row["text"], row["mode"],
                                        json.loads(row["meta_json"] or "{}"))
        except Exception as exc:
            attempts = int(row["attempts"])
            transient = retry_on_failure and self._outbox_transient(exc) and attempts < 4
            delay = OUTBOX_BACKOFF[min(attempts - 1, 2)] if transient else 0
            try:
                self.store.set_outbox_state(row_id, "queued" if transient else "failed",
                                            error=str(exc), not_before=time.time() + delay)
            except Exception as store_exc:
                logger.warning("[gate] outbox could not record send failure: %s", store_exc)
            raise
        else:
            provider_id = None
            if isinstance(result, dict):
                provider_id = result.get("message_id") or result.get("id")
            try:
                self.store.set_outbox_state(row_id, "sent", provider_message_id=str(provider_id) if provider_id else None)
            except Exception as exc:
                logger.warning("[gate] outbox could not record sent message: %s", exc)
            return result

    async def _outbox_loop(self) -> None:
        while True:
            try:
                await self._dispatch_outbox()
            except Exception as exc:
                logger.warning("[gate] outbox dispatcher failed: %s", exc)
                # If the store cannot be read, waiting callers still need a reply.
                for row_id in list(self._outbox_waiters):
                    try:
                        row = self._outbox_payloads.get(row_id)
                        active = self._outbox_target_tasks.get(row["target"]) if row else None
                        if row and (active is None or active.done()):
                            await self._fallback_outbox(row)
                    except Exception as fallback_exc:
                        logger.warning("[gate] outbox fallback failed: %s", fallback_exc)
            try:
                await asyncio.wait_for(self._outbox_wake.wait(), timeout=15)
            except asyncio.TimeoutError:
                pass
            self._outbox_wake.clear()

    def _finish_outbox(self, row_id: int, result: Any = None, error: Optional[Exception] = None) -> None:
        self._outbox_retry_policy.pop(row_id, None)
        self._outbox_payloads.pop(row_id, None)
        waiter = self._outbox_waiters.pop(row_id, None)
        if waiter is not None and not waiter.done():
            if error is not None:
                waiter.set_exception(error)
            else:
                waiter.set_result(result)

    async def _fallback_outbox(self, row: Dict[str, Any]) -> None:
        """Keep a broken queue from silencing an otherwise available channel."""
        try:
            self.store.abandon_outbox(row["id"])
        except Exception:
            pass
        try:
            result = await self.send_fn(row["target"], row["text"], row["mode"],
                                        json.loads(row["meta_json"] or "{}"))
        except Exception as exc:
            self._finish_outbox(row["id"], error=exc)
        else:
            self._finish_outbox(row["id"], result=result)

    async def _dispatch_outbox(self) -> None:
        async with self._outbox_dispatch_lock:
            rows = self.store.due_outbox(time.time())
            recipients: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                recipients.setdefault(row["target"], []).append(row)
            held: Dict[str, List[str]] = {}
            for target, mode in self.store.held_recipients():
                held.setdefault(target, []).append(mode)
            # Keep scanning while a slow channel send is in progress. One task owns
            # each person until that person's queued and held sends are done.
            for target in recipients.keys() | held.keys():
                active = self._outbox_target_tasks.get(target)
                if active is not None and not active.done():
                    continue
                self._outbox_dirty_targets.discard(target)
                task = asyncio.create_task(self._dispatch_target_and_held(
                    target, recipients.get(target, []), held.get(target, [])))
                self._outbox_target_tasks[target] = task
                task.add_done_callback(lambda done, who=target: self._outbox_target_done(who, done))

    def _outbox_target_done(self, target: str, task: asyncio.Task) -> None:
        if self._outbox_target_tasks.get(target) is task:
            self._outbox_target_tasks.pop(target, None)
        progressed = False
        if not task.cancelled():
            try:
                progressed = task.result()
            except Exception as exc:
                logger.warning("[gate] outbox delivery for %s failed: %s", target, exc)
        if progressed or target in self._outbox_dirty_targets:
            self._outbox_wake.set()

    async def _dispatch_target_and_held(self, target: str, rows: List[Dict[str, Any]],
                                        modes: List[str]) -> bool:
        completed = await self._dispatch_target(rows)
        held_sent = await self._dispatch_held_target(target, modes)
        if len(rows) >= 100 and completed:
            self._outbox_wake.set()
        return bool(completed or held_sent)

    async def _dispatch_target(self, rows: List[Dict[str, Any]]) -> int:
        completed = 0
        for row in rows:
            row_id = row["id"]
            try:
                # A paced message is still saved, but its caller is free to continue.
                if row["kind"] in OUTBOX_PROACTIVE_KINDS and self._proactive_limit() > 0:
                    recent = self.store.recent_outbox_sends(row["target"], time.time() - 3600, row["mode"])
                    if sum(r["kind"] in OUTBOX_PROACTIVE_KINDS and r["last_error"] != "duplicate suppressed"
                           for r in recent) >= self._proactive_limit() or self.store.has_held_outbox(row["target"], row["mode"]):
                        if self.store.hold_outbox(row_id):
                            self._finish_outbox(row_id, result=_OUTBOX_HELD)
                            completed += 1
                        continue
                claimed = self.store.claim_outbox(row_id)
            except Exception as exc:
                logger.warning("[gate] outbox unavailable, sending directly: %s", exc)
                await self._fallback_outbox(row)
                completed += 1
                continue
            if not claimed:
                # An older retry or another worker owns this person's next send.
                break
            try:
                result = await self._deliver_outbox(
                    row_id, retry_on_failure=self._outbox_retry_policy.get(row_id, True))
            except Exception as exc:
                self._finish_outbox(row_id, error=exc)
                logger.warning("[gate] outbox send %s failed: %s", row_id, exc)
                completed += 1
                if self.store.get_outbox(row_id)["status"] == "queued":
                    break  # A queued retry must go before later messages to this person.
            else:
                self._finish_outbox(row_id, result=result)
                completed += 1
        return completed

    async def _dispatch_held_target(self, target: str, modes: List[str]) -> bool:
        sent = False
        for mode in modes:
            try:
                sent = await self._dispatch_held(target, mode) or sent
            except Exception as exc:
                logger.warning("[gate] held outbox delivery failed: %s", exc)
        return sent

    async def _dispatch_held(self, target: str, mode: str) -> bool:
        recent = self.store.recent_outbox_sends(target, time.time() - 3600, mode)
        if self._proactive_limit() > 0 and sum(
                row["kind"] in OUTBOX_PROACTIVE_KINDS and row["last_error"] != "duplicate suppressed"
                for row in recent) >= self._proactive_limit():
            return False
        sent_texts = {row["text"] for row in self.store.recent_outbox_sends(target, time.time() - 600, mode)
                      if row["last_error"] != "duplicate suppressed"}
        held = self.store.held_outbox(target, mode)
        if any(row["not_before"] > time.time() for row in held):
            return False      # Wait for the oldest failed batch so all held words stay together.
        claimed = []
        for row in held:
            if not self.store.claim_outbox(row["id"], "held",
                                           allow_prior_sending=bool(claimed)):
                break
            claimed.append(row)
        if not claimed:
            return False
        fresh = []
        for row in claimed:
            if row["text"] in sent_texts:
                self.store.set_outbox_state(row["id"], "sent", error="duplicate suppressed")
            else:
                fresh.append(row)
        claimed = fresh
        if not claimed:
            return True
        text = "\n\n".join(row["text"] for row in claimed)
        try:
            result = await self.send_fn(target, text, mode, json.loads(claimed[0]["meta_json"] or "{}"))
        except Exception as exc:
            for row in claimed:
                attempts = int(row["attempts"]) + 1
                transient = self._outbox_transient(exc) and attempts < 4
                delay = OUTBOX_BACKOFF[min(attempts - 1, 2)] if transient else 0
                self.store.set_outbox_state(row["id"], "held" if transient else "failed",
                                            error=str(exc), not_before=time.time() + delay)
            raise
        else:
            provider_id = (result.get("message_id") or result.get("id")) if isinstance(result, dict) else None
            for row in claimed:
                self.store.set_outbox_state(row["id"], "sent", provider_message_id=str(provider_id) if provider_id else None)
        return True

    def _threads_for(self, target: str, chat_id: str = "") -> List[str]:
        """Where an outbound belongs. The caller's record when it named one, the owner's
        single thread for anything addressed to him, and otherwise the target itself so the
        message is written down somewhere rather than nowhere."""
        if chat_id:
            return [chat_id]
        conv = str(self.approver_conv or "")
        return [OWNER_THREAD] if (conv and conv in target) else [target]

    async def recover_interrupted(self) -> None:
        """A run that was in flight when the process died leaves the sender watching a
        typing indicator that never resolves. On startup, say so on the thread it was for,
        and offer to run it again. It is not retried on its own: a half-finished run may
        already have changed something, and doing it twice is worse than asking."""
        try:
            # "running" is a crash while the executor was mid-call. An "approved" row
            # stuck in that state is only ever a schedule request (an ordinary request
            # goes pending -> approved -> running in one call, with no gap a crash can
            # land in) -- store.request_running_for_schedule treats an "approved" row as
            # still in flight forever, so a missed one here means every later firing of
            # that schedule silently skips. Non-schedule "approved" rows are left alone.
            queued_timer_requests = self.store.active_timer_request_ids()
            queued_followups = self.store.active_timer_followup_ids()
            replied_requests, replied_followups = self.store.replied_timer_refs()
            running = self.store.requests_in_state("running")
            for req in running:
                if (req.status or {}).get("followup_id") in queued_followups:
                    # The inbox will retry this follow-up. Retire the partial
                    # request without telling the chat it was lost.
                    self.store.set_state(req.id, "failed", status={"ok": False,
                                         "error": "interrupted; inbox will retry"})
                elif req.id in replied_requests or (req.status or {}).get("followup_id") in replied_followups:
                    # A reply went out before the crash. Close the partial run
                    # without running it twice or sending another notice.
                    self.store.set_state(req.id, "failed", status={"ok": False,
                                         "error": "interrupted after replying"})
                    if req.schedule_id:
                        await self.schedules.interrupted(req, already_replied=True)
                    elif (req.status or {}).get("followup_id"):
                        self.store.finish_followup(int(req.status["followup_id"]),
                                                   "Interrupted after replying.")
            stranded = [r for r in running
                        if r.id not in queued_timer_requests and
                        r.id not in replied_requests and
                        (r.status or {}).get("followup_id") not in queued_followups and
                        (r.status or {}).get("followup_id") not in replied_followups] + [
                r for r in self.store.requests_in_state("approved") if r.schedule_id
                and r.id not in queued_timer_requests and r.id not in replied_requests
            ]
        except Exception:
            logger.exception("[gate] could not look for interrupted runs")
            return
        for req in stranded:
            try:
                self.store.set_state(req.id, "failed",
                                     status={"ok": False, "error": "interrupted by a restart",
                                             "summary": "interrupted by a restart", "raw": "", "tool_calls": []})
                self.store.set_thread(req.chat_id, "idle")
                self.task_note(req, "failed", f"Interrupted by a restart: {req.summary}", state="open")
                if req.schedule_id:
                    # A restart is not a failed run: schedules.interrupted() records its own
                    # outcome, does not count toward the three-strikes pause, and re-fires the
                    # schedule once immediately if that is warranted -- finished()/run_outcome
                    # would wrongly treat this as a failure and pause after three restarts.
                    await self.schedules.interrupted(req)
                    logger.info("[gate] reported interrupted scheduled request #%s", req.id)
                    continue
                line = (f"That one stopped when I restarted, so it never finished: {req.summary}. "
                        f"Nothing was reported back. Say the word and I will run it again.")
                session = self.get(req.chat_id)
                if not await session.notify_after_request(line, inbound_id=req.inbound_id):
                    await self.send_to_approver(ownertext.compose(
                        "failed", f"Stopped by a restart (#{req.id})",
                        facts=[("Asked", ownertext.strip_group_tag(req.summary))],
                        body="It never finished and nothing was reported back.",
                        tail="Say the word and I will run it again."), kind="notice")
                logger.info("[gate] told %s about interrupted request #%s", req.chat_id, req.id)
            except Exception:
                logger.exception("[gate] could not report interrupted request #%s", req.id)

    async def send_to_approver(self, text: str, *, kind: str = "reply") -> None:
        conv = self.approver_conv
        await self.send(f"imessage:{conv}", text, "imessage", {"conversation_id": conv}, kind=kind,
                        hold_exempt=True)

    def _is_owner_target(self, target: str, meta: Optional[Dict[str, Any]]) -> bool:
        """Whether a send reaches Aaron and nobody else. A group never counts as him."""
        meta = meta or {}
        if meta.get("conversation_kind") == "group" or meta.get("group_title"):
            return False
        conv = str(self.approver_conv or "")
        if conv and (conv in str(target) or str(meta.get("conversation_id") or "") == conv):
            return True
        sess = self.sessions.get(target)
        if sess is not None:
            try:
                if sess.is_approver() and not sess.in_group():
                    return True
            except Exception:
                pass
        from .executor import _is_owner_handle
        protected = list(getattr(self.executor, "protected", []) or [])
        value = str(target).split(":", 1)[-1]
        if protected and (_is_owner_handle(value, protected) or _is_owner_handle(str(target), protected)):
            return True
        digits = re.sub(r"\D", "", value)
        if self.approver_phone and digits and digits[-10:] == re.sub(r"\D", "", self.approver_phone)[-10:]:
            return True
        if self.approver_telegram_id and value == str(self.approver_telegram_id):
            return True
        return False

    def _asked_by_owner(self, req: Request) -> bool:
        """Whether Aaron himself wrote the message this request came from, wherever he
        wrote it (his own thread, or a group he is in). Judged from the request's own
        sender with the same verified identities is_approver() uses; a phone call never
        counts (caller ID can be spoofed)."""
        if self._is_owner_request(req):
            return True
        if req.mode == "voice":
            return False
        value = str(req.sender or "").split(":", 1)[-1].strip()
        if not value:
            return False
        if req.mode == "telegram":
            return bool(self.approver_telegram_id) and value == str(self.approver_telegram_id)
        if self.approver_phone and _digits(value) and _digits(value) == _digits(self.approver_phone):
            return True
        try:
            from .executor import _is_owner_handle
            protected = list(getattr(self.executor, "protected", []) or [])
            return bool(protected) and _is_owner_handle(value, protected)
        except Exception:
            return False

    def _answers_owner_in_place(self, origin: Request, target: str, chat_id: str = "") -> bool:
        """Aaron asked in a chat and the reply goes back to that same chat: he is reading
        it there, so it needs no second yes. A send to any other chat or person is still
        held."""
        if not self._asked_by_owner(origin):
            return False
        here = str(origin.chat_id or "")
        return bool(here) and here in (str(target or ""), str(chat_id or ""))

    def _recipient_label(self, target: str, meta: Dict[str, Any]) -> str:
        """Who a message goes to, by name only: the group title or the person's name.
        A chat id, a UUID or a provider handle is never shown; a phone number or email
        address only when no name is known at all."""
        contact = meta.get("contact") if isinstance(meta.get("contact"), dict) else {}
        name = str(meta.get("group_title") or meta.get("name") or contact.get("name") or "").strip()
        if not name:
            try:
                route = self.store.thread_route(str(target)) or {}
                rmeta = dict(route.get("meta") or {})
                name = str(rmeta.get("group_title") or rmeta.get("name") or "").strip()
            except Exception:
                name = ""
        if name:
            return " ".join(name.split())
        address = str(meta.get("to") or meta.get("conversation_id") or target).strip()
        address = address.split(":", 1)[-1] if re.match(r"^[a-z_]+:", address) else address
        if chatreply.UUID_RE.search(address) or chatreply.HEX_ID_RE.search(address):
            return "someone"
        return address

    async def hold_message(self, origin: Request, target: str, text: str, mode: str,
                           meta: Dict[str, Any], *, chat_id: str = "", kind: str = "reply") -> Request:
        """Park one outbound message as its own pending request and show Aaron exactly what
        would go: channel, recipient, full text, attachments. "#id yes" sends that text,
        unchanged, once; "#id no" discards it."""
        meta = dict(meta or {})
        label = self._recipient_label(target, meta)
        files = []
        for key in ("attachments", "media", "files", "media_paths", "attachment_paths"):
            v = meta.get(key)
            for item in (v if isinstance(v, (list, tuple)) else [v] if v else []):
                name = os.path.basename(str(item.get("name") or item.get("path") or item.get("url") or "")
                                        if isinstance(item, dict) else str(item))
                if name:
                    files.append(name)
        held = {"target": target, "text": text, "mode": mode, "meta": meta, "chat_id": chat_id,
                "kind": kind, "for_request": origin.id, "label": label, "attachments": files}
        req, _ = self.store.create_request_once(
            chat_id=origin.chat_id, sender=origin.sender, sender_name=origin.sender_name, mode=origin.mode,
            subject=origin.subject or "", original_message=f"Held message from #{origin.id} to {label}",
            summary=f"Send to {label}", scopes=[], prompt=json.dumps(held, sort_keys=True, ensure_ascii=False),
            state="pending", task_id=self.store.task_id_for_request(origin.id), bypass_dedupe=True)
        self.store.set_state(req.id, "pending", status={"held_message": held})
        logger.info("[gate] #%s: %s message to %s held for Aaron as #%s", origin.id, mode, target, req.id)
        facts = [("To", label), ("On", ownertext.SURFACE.get(str(mode or "").lower(), str(mode or "")))]
        if files:
            facts.append(("Attachments", ", ".join(files)))
        await self.send_to_approver(ownertext.compose(
            "held", "A message waiting for your yes", facts=facts,
            body=f"Nothing has been sent. This is exactly what would go out:\n\n{text}",
            handle=ownertext.reply_handle(req.id)), kind="approval")
        return req

    async def _decide_held_message(self, req: Request, d: str, actor: str) -> Dict[str, Any]:
        held = (req.status or {}).get("held_message") or {}
        if d in ("yes", "y", "ok", "approve", "run", "go", "send"):
            self.store.set_state(req.id, "approved")
            try:
                await self.send(held["target"], held["text"], held["mode"], dict(held.get("meta") or {}),
                                chat_id=held.get("chat_id") or "", kind=held.get("kind") or "reply",
                                role="answer", hold_exempt=True)
            except Exception as exc:
                logger.warning("[gate] held message #%s could not be sent: %s", req.id, exc)
                self.store.set_state(req.id, "failed", status={**(req.status or {}), "error": str(exc)})
                await self.send_to_approver(f"#{req.id} could not be sent: {exc}")
                return {"ok": False, "state": "failed", "error": str(exc)}
            self.store.set_state(req.id, "done", status={**(req.status or {}), "sent_by": actor})
            self.task_note(req, "done", f"{actor} approved the held message to {held.get('label')}", state="open")
            self._finish_waiting_origin(held)
            await self.send_to_approver(f"Sent #{req.id} to {held.get('label')}.{self._still_waiting_line(req.id)}")
            return {"ok": True, "state": "done", "error": ""}
        if d in ("no", "n", "stop", "drop", "reject", "decline", "discard", "deny", "cancel"):
            self.store.set_state(req.id, "rejected")
            self.task_note(req, "rejected", f"{actor} discarded the held message to {held.get('label')}", state="open")
            self._finish_waiting_origin(held)
            await self.send_to_approver(f"Discarded #{req.id}. Nothing was sent.{self._still_waiting_line(req.id)}")
            return {"ok": True, "state": "rejected", "error": ""}
        return {"ok": False, "error": f"#{req.id} needs a yes or a no", "state": req.state}

    def _open_held(self) -> List[Request]:
        """Held messages still waiting for Aaron's yes or no, oldest first."""
        return [p for p in self.store.pending() if (p.status or {}).get("held_message")]

    def _still_waiting_line(self, answered_id: int) -> str:
        """After one held message is answered: the others still waiting, by handle."""
        rest = [p for p in self._open_held() if p.id != answered_id]
        if not rest:
            return ""
        handles = chatreply.and_list([f"#{p.id}" for p in rest])
        verb = "is" if len(rest) == 1 else "are"
        return f" {handles} {verb} still waiting for your yes or no."

    def _finish_waiting_origin(self, held: Dict[str, Any]) -> None:
        """The run that wrote a held message ended as waiting on it. Once none of its held
        messages are open any more, that request is done."""
        try:
            origin = self.store.get_request(int(held.get("for_request") or 0))
        except Exception:
            origin = None
        if origin is None or origin.state != "waiting":
            return
        if any(int((p.status or {}).get("held_message", {}).get("for_request") or 0) == origin.id
               for p in self._open_held()):
            return
        self.store.set_state(origin.id, "done", status={**(origin.status or {}), "ok": True})
        self.task_note(origin, "done", "The held message was answered.", state="open")

    async def send_schedule_notice(self, schedule: Any, text: str, *, kind: str = "schedule") -> int:
        """Every schedule notice and proposal (schedules.py's only way to speak)
        goes through here: send the same way an ordinary owner reply on that channel
        is sent (the real chat_id and reply meta captured at proposal time, packed into
        origin_chat_id as JSON by the caller), falling back to the owner's iMessage
        thread when there is no real origin (console-created, or an old plain-string
        row) or that send fails. The stored row always lands on the owner's own
        thread, tagged with schedule_id, so a bare reply there is understood as
        answering this schedule regardless of which channel it went out on."""
        raw = str(getattr(schedule, "origin_chat_id", "") or "")
        mode = str(getattr(schedule, "origin_mode", "") or "")
        target = raw
        meta: Dict[str, Any] = {}
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                target = str(parsed.get("chat_id") or "")
                meta = dict(parsed.get("meta") or {})
            except Exception:
                target = raw
        sent = False
        if target and mode and mode != "console":
            try:
                if self._outbox_enabled():
                    await self.send(target, text, mode, meta, record=False, kind=kind,
                                    retry_on_failure=False)
                else:
                    await self.send_fn(target, text, mode, meta)
                sent = True
            except Exception:
                logger.exception("[gate] schedule %s notice failed on %s/%s; falling back to the owner's thread",
                                 getattr(schedule, "id", "?"), mode, target)
        if not sent:
            conv = self.approver_conv
            mode = "imessage"
            target = f"imessage:{conv}"
            if self._outbox_enabled():
                await self.send(target, text, mode, {"conversation_id": conv}, record=False,
                                kind=kind)
            else:
                await self.send_fn(target, text, mode, {"conversation_id": conv})
        # Said in a group chat, it is written down in that group's own record, so a
        # bare "yes" there finds the proposal it answers.
        record = OWNER_THREAD
        schedule_chat = str(getattr(schedule, "chat_id", "") or "")
        if sent and meta.get("conversation_kind") == "group" and schedule_chat and schedule_chat != OWNER_THREAD:
            record = schedule_chat
        return self.store.add_message(record, "outbound", text, mode, role="notice",
                                      schedule_id=getattr(schedule, "id", None))

    _outage_notified: Dict[str, float] = {}

    async def report_outage(self, exc: BaseException, chat_id: str) -> None:
        """A turn died on an infrastructure failure the owner has to fix (an API out
        of credit, a bad key, a provider down). Say so once per hour per cause on
        the owner's thread instead of failing silently."""
        text = str(exc)
        status = ""
        m = re.search(r"'(\d{3}) [^']+' for url '(https?://[^/']+)", text)
        if m:
            status, host = m.group(1), m.group(2)
        else:
            host = ""
        if status not in ("401", "402", "403", "429", "500", "502", "503") and "Payment" not in text:
            return
        key = f"{host}:{status}"
        now = time.time()
        if now - self._outage_notified.get(key, 0) < 3600:
            return
        self._outage_notified[key] = now
        what = {"402": "is out of credit (402 Payment Required)", "401": "rejected the API key (401)",
                "403": "refused the request (403)", "429": "is rate limiting (429)"}.get(status, f"returned {status}")
        who = "DeepSeek" if "deepseek" in host else ("TypeSafe" if "typesafe" in host else host or "a provider")
        try:
            await self.send_to_approver(ownertext.compose(
                "failed", f"{who} {what}",
                body="I cannot finish turns until that is fixed. Your last message was not answered."),
                kind="notice")
        except Exception:
            logger.exception("[gate] could not deliver the outage notice")

    # -- people auto-indexing (peopleindex.py) -----------------------------
    PEOPLE_NOTICE_DELAY = 20.0

    def index_person_seen(self, *, display: str = "", handles: Optional[List[Tuple[str, str]]] = None,
                          met_in: str = "", note: str = "") -> Dict[str, Any]:
        """Record someone Blatbot met (store.index_person: a directory entry, never a
        role or a scope) and, when that made or materially changed a record, queue
        the one notice to Aaron. Never raises: a failed index is logged and the
        message it came with is handled exactly as before."""
        try:
            res = self.store.index_person(display=display, handles=handles or [], met_in=met_in, note=note)
        except Exception:
            logger.exception("[gate] people index failed for %s", display or "a sender")
            return {"action": "error", "person": None}
        if res.get("action") in ("added", "updated"):
            logger.info("[gate] people index: %s %s (met in %s)", res["action"],
                        (res.get("person") or {}).get("display") or "?", met_in or "?")
            self._queue_people_notice()
        return res

    def index_introduction(self, text: str, *, addressed: bool, met_in: str) -> Optional[Dict[str, Any]]:
        """Aaron introducing someone in words ("blatbot, this is Stanley Rabkin, he runs
        X"): their name, and the rest of his sentence as the note. None when the
        message is not an introduction (peopleindex.parse_introduction decides)."""
        parsed = peopleindex.parse_introduction(text, addressed=addressed)
        if parsed is None:
            return None
        name, note = parsed
        return self.index_person_seen(display=name, met_in=met_in, note=note)

    def _queue_people_notice(self) -> None:
        if self._people_notice_task is not None and not self._people_notice_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return      # no loop yet: the expiry loop's flush picks it up
        self._people_notice_task = loop.create_task(self._people_notice_after_delay())

    async def _people_notice_after_delay(self) -> None:
        await asyncio.sleep(self.PEOPLE_NOTICE_DELAY)
        try:
            await self.flush_people_notices()
        except Exception:
            logger.exception("[gate] people notice failed; the next flush retries it")

    async def flush_people_notices(self) -> int:
        """Tell Aaron, in one text, about every indexed record that differs from what
        he was last told: one line each, stating the record as it stands. A record is
        marked told only after the text went out, so a failed send is retried and a
        sent one is never repeated, across restarts too. Returns the lines sent."""
        async with self._people_notice_lock:
            lines: List[str] = []
            seen: List[Dict[str, Any]] = []
            for person, told in self.store.people_to_announce():
                line = peopleindex.notice_line(person, told)
                if line:
                    lines.append(line)
                seen.append(person)
            if lines:
                await self.send_to_approver(peopleindex.notice_text(lines), kind="notice")
            for person in seen:
                self.store.mark_person_told(person["id"], person)
            return len(lines)

    def result_notice(self, req: Request, title: str, message: str, *, ok: bool,
                      late: bool = False, same_limit: bool = False, own: bool = False) -> str:
        """How a finished request reads on Aaron's phone: what happened and for whom in
        the first line, what was asked and where on the label lines, then the result."""
        group, tagged_from, _ = ownertext.split_group_tag(req.original_message or "")
        who = "" if own else ownertext.person(req.sender_name or tagged_from, req.sender)
        if same_limit:
            head = "Could not run"
            message = "Same usage limit as before."
        elif ok:
            head = "Finished late" if late else "Done"
        else:
            head = "Could not finish" + (", finished late" if late else "")
        if who:
            head += f" for {who}"
        head += f" (#{req.id})"
        facts = [("Asked", ownertext.strip_group_tag(title))]
        if not own:
            facts.append(("Where", ownertext.where(req.mode, group)))
        return ownertext.compose("done" if ok else "failed", head, facts=facts, body=message)

    def owner_in_group(self, session: GateSession) -> bool:
        """True when this thread is a group chat Aaron is in himself. There is no
        membership roster anywhere, so this is what the record shows: he has written in
        that group (his own messages are stored there under his name)."""
        meta = session.reply_meta or {}
        if meta.get("conversation_kind") != "group":
            return False
        own = re.sub(r"\D", "", str(getattr(self, "approver_phone", "") or ""))[-10:]
        if own and any(re.sub(r"\D", "", str(n))[-10:] == own for n in (meta.get("participants") or [])):
            return True
        first = OWNER_NAME.split()[0].lower()
        try:
            names = self.store._group_member_names(session.record_id)
        except Exception:
            logger.debug("[gate] could not read who is in %s", session.chat_id, exc_info=True)
            return False
        return any(str(n).split()[0].lower() == first for n in names if str(n).strip())

    def fyi_text(self, session: GateSession, body: str) -> str:
        """The copy Aaron gets of a message someone else sent: who wrote, where, the
        message itself, and that nothing is needed from him (an ask, if one is needed,
        comes separately). Never the run's routing note or a raw sender id."""
        meta = session.reply_meta or {}
        group, tagged_from, message = ownertext.split_group_tag(body)
        group = group or str(meta.get("group_title") or "")
        who = ownertext.person(session._sender_name() or tagged_from, session._sender())
        subject = str(meta.get("subject") or "").strip()
        return ownertext.compose(
            "fyi", f"{who} wrote to {_ASSISTANT}",
            facts=[("Where", ownertext.where(session.mode, group)), ("Subject", subject)],
            body=ownertext.quote(message, limit=0),
            tail=("Nothing needed from you. I answer in the group only when it is meant for me."
                  if group or meta.get("conversation_kind") == "group" else
                  "Nothing needed from you. I am answering it, and anything that needs your yes "
                  "comes as its own text."))

    async def notify_inbound(self, session: GateSession, body: str) -> None:
        if not bool(_settings.get("GATE_FYI_INBOUND")):
            # Aaron, 2026-11-18: no copy of every message people send it. He is looped in
            # only when a run needs a tool or memory the sender has no permission for.
            return
        meta = session.reply_meta or {}
        if meta.get("conversation_kind") == "group" or knowledge.meta_is_group(meta):
            # Group chatter is never copied to Aaron. He hears about a group only
            # when something needs his yes, which comes as its own text.
            return
        if self.owner_in_group(session):
            return
        try:
            await self.send_to_approver(self.fyi_text(session, body), kind="notice")
        except Exception:
            logger.exception("[gate] FYI to approver failed")

    def _approval_reply_text(self, req: Request) -> str:
        return ownertext.reply_handle(req.id, edit=True)

    def format_request(self, req: Request, revised: bool = False) -> str:
        group, tagged_from, original = ownertext.split_group_tag(req.original_message or "")
        _, _, summary = ownertext.split_group_tag(req.summary or "")
        name = ownertext.person(req.sender_name or tagged_from, req.sender)
        facts = [("Where", ownertext.where(req.mode, group)), ("Wants", summary),
                 ("Subject", req.subject or "")]
        if req.scopes:
            # Scope labels are already verb phrases ("look up and manage your Stern
            # calendar"), so this reads as one sentence after "It would".
            facts.append(("It would", schedprop.short_labels(req.scopes)))
        marker = "--- AARON'S INSTRUCTIONS ---"
        if marker in req.prompt:
            facts.append(("Your instructions", req.prompt.split(marker, 1)[1].strip()))
        quoted = ownertext.quote(original) if original and " ".join(original.split()) != " ".join(summary.split()) else ""
        head = f"{name} needs your yes" + (", revised" if revised else "") + f" (#{req.id})"
        return ownertext.compose("ask", head, facts=facts, body=quoted,
                                 handle=self._approval_reply_text(req))

    def resolve_task(self, session: GateSession, out: RouterOutput, message: str = "") -> Dict[str, Any]:
        """Decide which task a request belongs to. Always returns a task.

        The router proposes: an existing id ("T12"), "new" with a title, or nothing.
        Code checks the proposal: a non-owner may only continue a task they are on;
        an unknown or missing choice falls back to the sender's most recent open task
        (for a non-owner) or to a new task titled from the request summary. Whatever
        happens, the request ends up on exactly one task. This cannot be skipped."""
        req = out.request
        # A task is titled by the router, else by its request summary, else by the message
        # that started it. Never a placeholder, and never cut by length: a voice hand-off's
        # `message` is its whole call transcript, so the fallback here pulls the caller's own
        # words out of it (wording.fallback_title_from_message) rather than slicing the raw
        # transcript -- that slice used to land, verbatim and cut mid-word, as both the task
        # title and (via the "else by its request summary" rule two lines down, when the
        # router proposed none) the request's own summary.
        fallback = wording.fallback_title_from_message(message)
        summary = ownertext.strip_group_tag(req.summary if req else "") or fallback or "Untitled task"
        sender = session._sender()
        sender_name = session._sender_name() or sender
        approver = session.is_approver()
        counterpart = ((req.counterpart if req else None) or "").strip()
        choice = (out.task or "").strip().lower()

        chosen: Optional[Dict[str, Any]] = None
        if choice.startswith("t") and choice[1:].isdigit():
            cand = self.store.get_task(int(choice[1:]))
            if cand and cand["state"] != "closed":
                allowed = (approver or self.store.is_participant(cand["id"], sender)
                           or cand["id"] in self.store.task_ids_for_thread(session.thread_key()))
                if allowed:
                    chosen = cand
                else:
                    logger.warning("[gate] %s tried to continue task T%s they are not on; starting a new one",
                                   sender, cand["id"])
        if chosen is None and choice == "":
            # No explicit choice: continue the most recent open task of the person this
            # concerns (the sender, or the named counterpart when Aaron is asking).
            person = counterpart if (approver and counterpart) else ("" if approver else sender)
            found = self.store.tasks_for_person(person, open_only=True, limit=1) if person else []
            if found:
                chosen = found[0]
        if chosen is None:
            # Only the person's own words: never the group marker or its policy lines.
            title = ownertext.strip_group_tag((out.task_title or "").strip() or summary) or fallback
            if title == "Untitled task" and fallback:
                title = fallback
            chosen = self.store.create_task(title)
            logger.info("[gate] new task T%s: %s", chosen["id"], title)

        if out.task_summary:
            self.store.set_task_summary(chosen["id"], out.task_summary)
        # Participants: the sender (unless it is Aaron) and any named counterpart.
        if not approver:
            self.store.add_participant(chosen["id"], session.person())
        if counterpart:
            self.store.add_participant(chosen["id"], counterpart, counterpart)
        else:
            for addr in _EMAIL.findall(req.prompt if req else ""):
                if task_key(addr) != task_key(sender):
                    self.store.add_participant(chosen["id"], addr, addr)
                    break
        return self.store.get_task(chosen["id"])  # type: ignore[return-value]

    def record_request_on_task(self, req: Request, task: Dict[str, Any], session: GateSession) -> None:
        state = "running" if req.state == "approved" else "waiting_aaron"
        self.store.task_event(task["id"], "request", f"{req.summary} [scopes: {', '.join(req.scopes)}]",
                              chat_id=session.chat_id, request_id=req.id, state=state,
                              title=req.summary if str(task["title"]).startswith("Thread with") else None)

    def task_note(self, req: Request, kind: str, text: str, state: Optional[str] = None) -> None:
        tid = self.store.task_id_for_request(req.id)
        if tid:
            self.store.task_event(tid, kind, text, chat_id=req.chat_id, request_id=req.id, state=state)

    async def send_approval_text(self, req: Request, revised: bool = False) -> None:
        await self.send_to_approver(self.format_request(req, revised), kind="approval")

    async def handle_task_command(self, session: GateSession, text: str) -> bool:
        """Aaron's task commands: "tasks" lists open work; "T12 done" / "T12 close" /
        "T12 reopen" change state; "T12 note: ..." adds a note. Returns True if consumed."""
        t = (text or "").strip()
        if t.lower() in ("followups", "what are you waiting on", "what are you waiting on?"):
            from .hosttools import followup_time
            try:
                rows = self.store.pending_followups()
                lines = ["Pending follow-ups:"]
                for row in rows:
                    task = self.store.get_task(row["task_id"])
                    lines.append(f"#{row['id']} T{row['task_id']} {followup_time(row['due_at'])}: "
                                 f"{row['note']} ({task['title'] if task else 'task'})")
                await self.send_to_approver("\n".join(lines) if rows else "No pending follow-ups.")
            except Exception as exc:
                logger.warning("[gate] follow-up list failed: %s", exc)
                await self.send_to_approver("I could not list follow-ups right now.")
            return True
        match = re.fullmatch(r"cancel followup\s+#?(\d+)", t, re.I)
        if match:
            try:
                row = self.store.cancel_followup(int(match.group(1)))
                if row:
                    self.store.task_event(row["task_id"], "note",
                                          f"Follow-up #{row['id']} cancelled: {row['note']}",
                                          chat_id=session.chat_id)
                await self.send_to_approver(f"Follow-up #{match.group(1)} cancelled." if row else
                                            f"No pending follow-up #{match.group(1)}.")
            except Exception as exc:
                logger.warning("[gate] follow-up cancellation failed: %s", exc)
                await self.send_to_approver("I could not cancel that follow-up right now.")
            return True
        if t.lower() in ("tasks", "task list", "open tasks", "what's open", "whats open"):
            rows = self.store.open_tasks(limit=25)
            if not rows:
                await self.send_to_approver("No open tasks.")
                return True
            lines = ["Open tasks:"]
            for r in rows:
                who = ", ".join(p["display"] for p in r["participants"]) or "no one"
                st = {"waiting_aaron": "waiting on you", "running": "running"}.get(r["state"], r["state"])
                lines.append(f"T{r['id']} {r['title']} ({who}; {st})")
            await self.send_to_approver("\n".join(lines))
            return True
        m = re.match(r"^\s*t(\d+)\s*[:\-]?\s*(done|close|closed|reopen|open|note\s*:\s*(.+))\s*$", t, re.S | re.I)
        if not m:
            return False
        tid = int(m.group(1))
        task = self.store.get_task(tid)
        if not task:
            await self.send_to_approver(f"No task T{tid}.")
            return True
        verb = m.group(2).lower()
        if verb.startswith("note"):
            self.store.task_event(tid, "note", f"Aaron: {m.group(3).strip()}", chat_id=session.chat_id)
            await self.send_to_approver(f"Noted on T{tid}.")
        elif verb in ("done",):
            self.store.task_event(tid, "done", "Aaron marked this done.", chat_id=session.chat_id, state="done")
            await self.send_to_approver(f"T{tid} marked done: {task['title']}")
        elif verb in ("close", "closed"):
            self.store.task_event(tid, "note", "Aaron closed this task.", chat_id=session.chat_id, state="closed")
            await self.send_to_approver(f"T{tid} closed: {task['title']}")
        else:
            self.store.task_event(tid, "note", "Aaron reopened this task.", chat_id=session.chat_id, state="open")
            await self.send_to_approver(f"T{tid} reopened: {task['title']}")
        return True

    def _is_owner_request(self, req: Request) -> bool:
        """Did this come from the person the assistant works for? His own requests send in
        his name, so he reads anything that would go out. A request from anyone else was
        already shown to him and approved before it ran, and its answer goes back to them.

        Item C1: judged from the REQUEST's own identity (its chat_id, and its mode plus
        sender for a call), never from whatever mode the live session for that chat_id
        happens to be in right now. A session's mode is a transient, per-turn value --
        set to "voice" only for the duration of a call and restored to a text default
        once it ends -- so by the time an approved call request finishes (voice not
        trusted as the approver, so the approval itself happened later, on text), the
        session for its chat_id no longer reads as voice, or as the owner, at all. This
        grants no power: is_approver()/GATE_VOICE_TRUST_APPROVER already decided, at
        request-creation time, whether the request ran at all -- this only decides how
        its result is worded and where it goes once it is already finished."""
        if req.chat_id == OWNER_THREAD or req.chat_id in self._approver_chat_ids():
            return True
        return bool(req.mode == "voice" and self.approver_phone
                   and _digits(req.sender) == _digits(self.approver_phone))

    async def ask_confirmation(self, req: Request, status: Dict[str, Any]) -> None:
        """A destructive call is waiting. Park the request as pending with the exact call
        stored, and text the owner what it is. '#N yes' performs that one call."""
        c = status["confirm"]
        self.store.set_state(req.id, "pending", status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "awaiting_aaron")
        if req.schedule_id:
            await self.schedules.held(req, status)
        short = c["tool"].split("__")[-1]
        shown = {k: v for k, v in c["args"].items() if k != "user_google_email"}
        about = "\n".join(f"  {ln}" for ln in c.get("about") or []) or "  (no further detail found)"
        self.task_note(req, "request", f"Waiting on Aaron to confirm: {short} {json.dumps(shown, ensure_ascii=False)}", state="waiting_aaron")
        schedule_title = ""
        if req.schedule_id:
            sch = self.store.get_schedule(req.schedule_id)
            schedule_title = str(getattr(sch, "title", "") or "") if sch is not None else ""
        if c.get("kind") == "send":
            # Not a summary of a message: the message. He approves the words that will go out.
            full_text = ownertext.compose(
                "held", "A message waiting for your yes",
                facts=[("Schedule", schedule_title)],
                body=f"Nothing has been sent. This is what would go out:\n\n{_draft_text(shown)}",
                handle=ownertext.reply_handle(req.id))
        else:
            plain = await self._phrase_confirmation(req, short, shown, c.get("about") or [])
            plain = re.sub(r"^Before I do this, I need your yes[.:]?\s*", "", plain).strip()
            if plain:
                plain = plain[0].upper() + plain[1:]
            full_text = ownertext.compose(
                "ask", f"Before I do this, I need your yes (#{req.id})",
                facts=[("Schedule", schedule_title)], body=plain,
                tail="Nothing has happened yet.", handle=ownertext.reply_handle(req.id))
        schedule = self.store.get_schedule(req.schedule_id) if req.schedule_id else None
        if schedule is not None:
            # A held call that belongs to a scheduled run asks on that
            # schedule's own channel, not always the owner's iMessage -- the
            # "#<id> yes" reply mechanics themselves are unchanged.
            await self.send_schedule_notice(schedule, full_text, kind="approval")
        else:
            await self.send_to_approver(full_text, kind="approval")

    async def handle_ask(self, req: Request, status: Dict[str, Any]) -> None:
        """A run asked Aaron something (ask_aaron, always deferred). Park the request with
        its saved session and text him the question; his "#id <answer>" comes back
        through the inbox to decide_request, which resumes that same call with it."""
        ask = status.get("ask") or {}
        fresh = self.store.get_request(req.id) or req
        before = dict(fresh.status or {})
        saved = {**before, **status, "ask_count": int(before.get("ask_count") or 0) + 1}
        saved.pop("deferred_decision", None)
        self.store.set_state(req.id, "pending", status=saved, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "awaiting_aaron")
        question = " ".join(str(ask.get("question") or "").split())
        kind = "permission" if ask.get("kind") == "permission" else "input"
        owner = _owner_request(self, req)
        self.task_note(req, "request", f"Asked Aaron: {question}", state="waiting_aaron")
        await self.send_to_approver(outcomes.ask_text(req, question, kind, owner), kind="approval")
        if not owner:
            try:
                await self.get(req.chat_id).send_to_sender(HANDOFF_LINE, role="access_wait")
            except Exception as exc:
                logger.warning("[gate] #%s could not send the waiting notice: %s", req.id, exc)

    async def _ended_by_its_own_choice(self, req: Request, status: Dict[str, Any], session: "GateSession",
                                       title: str, message: str, from_aaron: bool) -> bool:
        """The run itself chose how it ended (gate/outcomes.py). True when that choice
        already covered the person who asked, so nothing more is delivered to them: it
        answered them with send_message, or it decided nothing needed sending. A run that
        chose none of the three is the safety net, and its closing text is delivered as
        before so nobody is left in silence."""
        chosen = status.get("outcome")
        if chosen is None or not outcomes.enabled():
            return False
        if chosen == "none":
            logger.info("[gate] #%s ended without choosing how (no send, no nothing_to_send, no ask); "
                        "delivering its closing text", req.id)
            return False
        if not (status.get("answered_requester") or chosen == "quiet"):
            logger.info("[gate] #%s sent to others but did not answer the person who asked; "
                        "delivering its closing text", req.id)
            return False
        ok = bool(status.get("ok"))
        try:
            if chosen == "quiet" and not status.get("answered_requester"):
                logger.info("[gate] #%s chose to send nothing: %s", req.id, status.get("quiet_reason") or "")
            elif not from_aaron:
                # Someone else's request: Aaron approved it, so he still hears how it went,
                # in the words that actually went out rather than the run's note to itself.
                said = "\n".join(f"Sent to {s.get('to')} on {outcomes.SURFACE_WORDS.get(s.get('surface'), s.get('surface'))}: "
                                 f"{s.get('text') or ''}" for s in status.get("sends") or [])
                await self.send_to_approver(self.result_notice(req, title, said or message, ok=ok),
                                            kind="notice")
        finally:
            session.typing_stop()
        return True

    async def handle_access_ask(self, req: Request, status: Dict[str, Any]) -> Optional[str]:
        """Grant what this sender already has, then park only the remaining access for Aaron."""
        if status.get("deferred"):
            status = dict(status)
            status["access_ask_count"] = int((req.status or {}).get("access_ask_count") or 0) + 1
        ask = status.get("access_request") or {}
        wanted = [s for s in ask.get("scopes", []) if s in SCOPES and s not in req.scopes]
        if not wanted:
            saved = {**((self.store.get_request(req.id) or req).status or {}), **status}
            saved.pop("access_request", None)
            self.store.set_state(req.id, "running", status=saved)
            return "Access granted. Continue the task from where you stopped; do not repeat finished steps."
        try:
            # Use the request's sender, since the live chat session may now belong
            # to a later message from someone else in the same conversation.
            session = self.get(req.chat_id)
            handles = session.handles() if session._sender() == req.sender else [req.sender]
            role_scopes = set(self.store.trust_for(handles).get("scopes") or [])
        except Exception as exc:
            logger.warning("[gate] #%s could not read role access: %s", req.id, exc)
            role_scopes = set()
        in_group = knowledge.chat_is_group(self.store, req.chat_id)
        covered = [s for s in wanted if not (in_group and knowledge.is_vault_scope(s)) and (
                   s in role_scopes or (_owner_request(self, req) and s not in EXPLICIT_ONLY_SCOPES))]
        uncovered = [s for s in wanted if s not in covered]
        if covered:
            self.store.set_scopes(req.id, list(req.scopes) + covered)
            self.task_note(req, "approved", "Granted access to " + ", ".join(LABELS.get(s, s) for s in covered) + ".")
        if uncovered:
            saved = {**(self.store.get_request(req.id).status or {}), **status,
                     "access_request": {**ask, "uncovered": uncovered}}
            self.store.set_state(req.id, "pending", status=saved, raw_output=status.get("raw"))
            self.store.set_thread(req.chat_id, "awaiting_aaron")
            labels = ", ".join(ASK.get(s) or LABELS.get(s, s) for s in uncovered).replace("\u2014", ":")
            who = (req.sender_name or req.sender or "Someone").replace("\u2014", ":")
            summary = (req.summary or "the task").replace("\u2014", ":")
            why = str(ask.get("why") or "No reason given").replace("\u2014", ":")
            self.task_note(req, "request", f"Waiting on Aaron for access to {labels}.", state="waiting_aaron")
            group, tagged_from, _ = ownertext.split_group_tag(req.original_message or "")
            await self.send_to_approver(ownertext.compose(
                "ask", f"Needs more access (#{req.id})",
                facts=[("From", ownertext.person(req.sender_name or tagged_from, req.sender)),
                       ("Where", ownertext.where(req.mode, group)),
                       ("Task", ownertext.strip_group_tag(summary)),
                       ("Also needs to", labels), ("Why", why)],
                handle=ownertext.reply_handle(req.id)),
                kind="approval")
            if not _owner_request(self, req):
                try:
                    await self.get(req.chat_id).send_to_sender(HANDOFF_LINE, role="access_wait")
                except Exception as exc:
                    logger.warning("[gate] #%s could not send access wait notice: %s", req.id, exc)
            return None
        saved = {**(self.store.get_request(req.id).status or {}), **status}
        saved.pop("access_request", None)
        self.store.set_state(req.id, "running", status=saved)
        return "Access granted: " + ", ".join(LABELS.get(s, s) for s in covered) + ". Continue the task from where you stopped; do not repeat finished steps."

    async def _phrase_confirmation(self, req: Request, short: str, shown: Dict[str, Any], about: List[str]) -> str:
        """One or two plain sentences saying what the change would do, for the owner."""
        try:
            text = await self.router.phrase(
                "Aaron's assistant is about to make a change and must ask him first. In one or two short plain "
                "sentences, say exactly what would happen, naming the thing (title, date, time, person) in words, "
                "no ids, no JSON, no links, no tool names. Start with 'Before I do this, I need your yes:' and end "
                "with a full stop. Nothing has happened yet; say so only if it helps.",
                {"what_he_asked": req.original_message, "the_change": {"tool": short, "arguments": shown},
                 "what_it_refers_to": about[:6]})
            text = " ".join((text or "").split())
            if text:
                return text
        except Exception:
            logger.debug("[gate] confirmation phrasing failed", exc_info=True)
        items = [a.strip() for a in about if a.strip()]
        if not items:
            return "Before I do this, I need your yes. I don't have more detail on what this affects."
        detail = " ".join(items)
        verb = _derive_action_word(short, shown)
        if verb:
            sentence = f"{verb} {detail}"
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            return f"Before I do this, I need your yes: {sentence}"
        return f"Before I do this, I need your yes. {detail}"

    async def perform_confirmed(self, req: Request) -> None:
        """The owner said yes: run exactly the stored call, then finish the request."""
        c = (req.status or {}).get("confirm") or {}
        if runguard.is_cancelled(req.id):
            logger.info("[gate] #%s was stopped by Aaron; not performing its held call", req.id)
            return
        self.store.set_state(req.id, "running")
        if not c:
            status = {"ok": False, "error": "no confirmed call on record", "summary": "no confirmed call on record", "raw": "", "tool_calls": []}
        else:
            # He has read it and said yes. That exact call runs, once, and nothing else.
            if c["tool"] == outcomes.SEND_TOOL:
                said = await outcomes.send_message(c["args"], self, req)
                sent = said.startswith(("Sent to", "Queued for"))
                status = {"ok": sent, "summary": said, "raw": said, "tool_calls": [c["tool"]],
                          **({} if sent else {"error": said})}
            else:
                caller = self.jev_agent or self.caller
                status = await caller.perform(c["tool"], c["args"])
        ok = bool(status.get("ok"))
        self.store.set_state(req.id, "done" if ok else "failed", status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "idle")
        result = _clean_result(status.get("summary") or status.get("error") or "")
        self.task_note(req, "done" if ok else "failed", f"{req.summary} -> {result}",
                       state="open" if req.schedule_id else ("done" if ok else "failed"))
        if req.schedule_id:
            # A held call's resolution is the whole of what this run needed to do; it was
            # already excluded from the three-strikes count and already got its one notice
            # when it was parked, so this goes through held_resolved, never finished().
            await self.schedules.held_resolved(req, status)
            return
        if await self.schedules.finished(req, status):
            return
        session = self.get(req.chat_id)
        note = f"Task #{req.id} {'done' if ok else 'failed'}: {req.summary}\nResult:\n{result}"
        replied = await session.notify_after_request(note, inbound_id=req.inbound_id)
        if not replied:
            await self.send_to_approver(self.result_notice(req, req.summary, result, ok=ok,
                                                           own=self._is_owner_request(req)))

    async def decide_request(self, request_id: int, decision: str, *, actor: str = "aaron",
                            source: str = "imessage", note: str = "") -> Dict[str, Any]:
        """The one place a pending request is answered, whatever surface the answer came in
        on. A text reply and a click in the console both land here, so the rules about what
        may run, and the record of who said so, cannot drift apart.

        decision: "yes" runs it, "no" drops it, "edit" revises it with `note`.
        Returns {"ok", "state", "error"}. Deciding twice is refused, not repeated.
        """
        req = self.store.get_request(int(request_id))
        if req is None:
            return {"ok": False, "error": f"no request #{request_id}", "state": ""}
        if req.state in ("approved", "running") and not req.schedule_id and \
                (decision or "").strip().lower() in ("no", "n", "stop", "cancel", "abort", "drop", "reject", "decline"):
            # "#520 no" while #520 runs: stop it, from a text or the console alike.
            line = await self.stop_request(req, by=actor)
            await self.send_to_approver(line)
            return {"ok": True, "state": "rejected", "error": ""}
        if req.state not in ("pending",):
            # Someone already answered this one. Say so rather than acting again.
            return {"ok": False, "error": f"#{req.id} is already {req.state}", "state": req.state}
        if (req.status or {}).get("held_message"):
            return await self._decide_held_message(req, (decision or "").strip().lower(), actor)
        hold = (req.status or {}).get("admin_hold")
        if hold:
            # A change parked by the chat admin commands (admincmd.py): his yes does exactly
            # that change, once; his no leaves everything as it was. Same answer from the
            # console's Requests page and from a text, since both land here.
            d0 = (decision or "").strip().lower()
            if d0 in ("yes", "y", "ok", "approve", "run", "go"):
                from . import admincmd
                reply = chatreply.finish(await admincmd.apply_hold(self, hold))
                self.store.set_state(req.id, "done", status={**(req.status or {}), "result": reply})
                self.task_note(req, "done", reply, state="done")
                await self.send_to_approver(reply)
                return {"ok": True, "state": "done", "error": ""}
            if d0 in ("no", "n", "stop", "drop", "reject", "decline"):
                self.store.set_state(req.id, "rejected")
                self.task_note(req, "rejected", f"Left alone: {hold.get('ask')}", state="done")
                await self.send_to_approver(f"Left alone. Nothing was changed for #{req.id}.")
                return {"ok": True, "state": "rejected", "error": ""}
            return {"ok": False, "error": f"#{req.id} needs a yes or a no", "state": req.state}
        if self.schedules.is_proposal_request(req):
            return await self.schedules.decide_proposal(req, decision, note)
        d = (decision or "").strip().lower()
        deferred = (req.status or {}).get("deferred") or {}
        if deferred.get("kind") == "ask":
            # Aaron's answer to a run's question. Whatever he wrote after "#id" is the
            # answer; a yes or no is the answer to a permission ask.
            answer = (" ".join(note.split()) if d in ("answer", "edit") else
                      "yes" if d in _YES or d in ("yes", "approve") else
                      "no" if d in _NO or d in ("no", "stop", "decline") else " ".join(str(decision or "").split()))
            if not answer:
                return {"ok": False, "error": f"#{req.id} needs an answer", "state": req.state}
            status = dict(req.status or {})
            status["deferred_decision"] = {"id": deferred["id"], "allow": True, "answer": answer}
            self.task_note(req, "approved", f"Aaron answered: {answer}", state="running")
            self.store.set_state(req.id, "approved", status=status)
            fresh = self.store.get_request(req.id) or req
            saved_session = deferred.get("session_id") or ""
            saved_at = float(status.get("session_saved_at") or 0)
            if (saved_at and time.time() - saved_at >
                    float(_settings.get("SESSION_MAX_AGE_HOURS")) * 3600):
                saved_session = ""
            asyncio.create_task(self.execute(fresh, continuation="continue", resume_session=saved_session))
            return {"ok": True, "state": "approved", "error": ""}
        if deferred and d != "edit":
            if d not in ("yes", "y", "ok", "approve", "run", "go",
                         "no", "n", "stop", "drop", "reject", "decline"):
                return {"ok": False, "error": f"#{req.id} needs a yes or a no", "state": req.state}
            allowed = d in ("yes", "y", "ok", "approve", "run", "go")
            status = dict(req.status or {})
            status["deferred_decision"] = {"id": deferred["id"], "allow": allowed}
            if deferred.get("kind") == "access":
                ask = status.pop("access_request", {}) or {}
                names = list(ask.get("uncovered") or ask.get("scopes") or [])
                labels = ", ".join(LABELS.get(s, s) for s in names)
                if allowed:
                    self.store.set_scopes(req.id, list(req.scopes) + names)
                    self.task_note(req, "approved", f"Aaron granted access to {labels}.", state="running")
                else:
                    status["access_refused"] = list(dict.fromkeys(
                        list(status.get("access_refused") or []) + names))
                    self.task_note(req, "rejected", f"Aaron refused access to {labels}.", state="running")
            else:
                self.task_note(req, "approved" if allowed else "rejected",
                               f"Aaron {'approved' if allowed else 'refused'} the held {deferred.get('kind') or 'call'}: {req.summary}",
                               state="running")
            self.store.set_state(req.id, "approved", status=status)
            if req.schedule_id and not allowed and deferred.get("kind") in ("send", "destructive"):
                await self.schedules.held_ended(req, "declined")
            fresh = self.store.get_request(req.id) or req
            saved_session = deferred.get("session_id") or ""
            saved_at = float(status.get("session_saved_at") or 0)
            if (saved_at and time.time() - saved_at >
                    float(_settings.get("SESSION_MAX_AGE_HOURS")) * 3600):
                saved_session = ""
            asyncio.create_task(self.execute(fresh, continuation="continue",
                                             resume_session=saved_session))
            return {"ok": True, "state": "approved", "error": ""}
        access_ask = (req.status or {}).get("access_request") or {}
        if access_ask and d != "edit":
            if d not in ("yes", "y", "ok", "approve", "run", "go", "no", "n", "stop", "drop", "reject", "decline"):
                return {"ok": False, "error": f"#{req.id} needs a yes or a no", "state": req.state}
            try:
                allowed = d in ("yes", "y", "ok", "approve", "run", "go")
                uncovered = list(access_ask.get("uncovered") or [])
                labels = ", ".join(LABELS.get(s, s) for s in uncovered).replace("\u2014", ":")
                status = dict(req.status or {})
                status.pop("access_request", None)
                if allowed:
                    self.store.set_scopes(req.id, list(req.scopes) + uncovered)
                    self.task_note(req, "approved", f"Aaron granted access to {labels}.", state="running")
                    prompt = f"Access granted: {labels}. Continue the task from where you stopped; do not repeat finished steps."
                else:
                    status["access_refused"] = list(dict.fromkeys(list(status.get("access_refused") or []) + uncovered))
                    self.task_note(req, "rejected", f"Aaron refused access to {labels}.", state="running")
                    prompt = f"Aaron said no to {labels}. Finish with what you have, or say plainly what could not be done."
                self.store.set_state(req.id, "approved", status=status)
                fresh = self.store.get_request(req.id) or req
                asyncio.create_task(self.execute(fresh, continuation=prompt, resume_session=status.get("session_id")))
                return {"ok": True, "state": "approved", "error": ""}
            except Exception as exc:
                logger.warning("[gate] #%s could not decide access: %s", req.id, exc)
                return {"ok": False, "state": req.state, "error": f"I could not decide access for #{req.id} right now."}
        confirm = (req.status or {}).get("confirm") or {}
        if d in ("yes", "y", "ok", "approve", "run", "go"):
            if confirm:
                self.store.add_message(OWNER_THREAD, "system",
                                       f"Confirmed the held call on request #{req.id} ({source}).")
                self.task_note(req, "approved", f"{actor} confirmed the change: {req.summary}", state="running")
                asyncio.create_task(self.perform_confirmed(req))
                return {"ok": True, "state": "running", "error": ""}
            self.store.set_state(req.id, "approved")
            self.store.add_message(OWNER_THREAD, "system", f"Approved request #{req.id} ({source}).")
            self.task_note(req, "approved", f"{actor} approved: {req.summary}", state="running")
            asyncio.create_task(self.execute(self.store.get_request(req.id)))  # type: ignore[arg-type]
            return {"ok": True, "state": "approved", "error": ""}
        if d in ("no", "n", "stop", "drop", "reject", "decline"):
            self.store.set_state(req.id, "rejected")
            self.store.set_thread(req.chat_id, "idle")
            kind = "rejected" if confirm else "declined"
            self.task_note(req, "rejected", f"{actor} {kind}: {req.summary}", state="open")
            if confirm:
                if req.schedule_id:
                    await self.schedules.held_ended(req, "declined")
                else:
                    await self.send_to_approver(f"Left alone. Nothing was changed for #{req.id}.")
            else:
                await self.send_to_approver(f"Dropped #{req.id}.")
                asyncio.create_task(self.get(req.chat_id).notify_after_request(
                    f"Aaron declined the task: {req.summary}. Nothing was done. "
                    f"Reply to the sender briefly if appropriate."))
            return {"ok": True, "state": "rejected", "error": ""}
        if d == "edit":
            if not note.strip():
                return {"ok": False, "error": "an edit needs instructions", "state": req.state}
            if access_ask or deferred:
                clean = dict(req.status or {})
                clean.pop("access_request", None)
                clean.pop("deferred", None)
                clean.pop("deferred_decision", None)
                clean.pop("confirm", None)
                self.store.set_state(req.id, "pending", status=clean)
            base = req.prompt.split("\n\n--- AARON'S INSTRUCTIONS ---")[0]
            new_req = self.store.revise_prompt(req.id, f"{base}\n\n--- AARON'S INSTRUCTIONS ---\n{note.strip()}")
            await self.send_approval_text(new_req, revised=True)
            return {"ok": True, "state": "pending", "error": ""}
        return {"ok": False, "error": f"unknown decision {decision!r}", "state": req.state}

    async def handle_approver_command(self, session: GateSession, text: str) -> bool:
        """Parse Aaron's reply as a gate command. Returns True if consumed."""
        if await self.handle_task_command(session, text):
            return True
        pending = self.store.pending()
        # In a group the bridge puts its routing tag in front of what he wrote; a
        # command is read from his own words. A bare yes/no there answers only what
        # is open in that group, never a pending request from another chat.
        from . import schedprop
        t = schedprop.strip_group_tag(text)
        if (session.reply_meta or {}).get("conversation_kind") == "group":
            here = getattr(session, "record_id", None) or getattr(session, "chat_id", "")
            explicit = re.match(r"^\s*#?(\d+)\b", t)
            if not explicit:
                pending = [p for p in pending if p.chat_id == here]
        else:
            # A bare yes/no in his own chat, where every held message's prompt is asked,
            # answers the most recently asked one, exactly like "#id yes". Never a new
            # request (2026-11-18: his "Yes" to #531 started #532 instead of sending it).
            bare = t.strip().lower().rstrip(".!")
            held = [p for p in pending if (p.status or {}).get("held_message")]
            if held and bare in (_YES | _NO):
                newest = max(held, key=lambda p: p.id)
                out = await self.decide_request(newest.id, "yes" if bare in _YES else "no",
                                                actor="Aaron", source=session.mode or "imessage")
                if not out.get("ok") and out.get("error"):
                    await self.send_to_approver(out["error"])
                return True
        m = re.match(r"^\s*#?(\d+)\s*[:\-]?\s*(.*)$", t, re.S)
        req: Optional[Request] = None
        rest = t
        if m and any(p.id == int(m.group(1)) for p in pending):
            req = next(p for p in pending if p.id == int(m.group(1)))
            rest = m.group(2).strip()
        elif len(pending) == 1 and t.lower() in (_YES | _NO):
            # A bare yes/no with exactly one pending request used to resolve it no
            # matter which thread it was on -- wrong when THIS thread also has an
            # open schedule proposal, which could then be skipped for an unrelated
            # request elsewhere. Only grab the pending request here when this
            # thread has no open proposal to disambiguate against; otherwise leave
            # req unset so an open proposal on this thread (below) or the explicit
            # disambiguation question can take it instead.
            from . import schedcmd
            chat_id = getattr(session, "record_id", None) or getattr(session, "chat_id", "")
            this_thread_proposals = schedcmd._open_proposals(self.store, chat_id)
            if this_thread_proposals:
                if pending[0].chat_id == chat_id:
                    # Both an open proposal and a pending approval on this same
                    # thread: genuinely ambiguous. Ask, resolve neither.
                    other = (pending[0].summary or pending[0].original_message or "").strip()
                    await session.send_to_sender(
                        f'Yes to which: starting "{this_thread_proposals[0].title}", or {other}?',
                        role="answer")
                    return True
                # The pending request is on a different thread; this thread's own
                # proposal (handled further below) answers the bare yes/no instead.
            else:
                req = pending[0]
        elif len(pending) == 1 and re.match(r"^edit\s*:\s*.+$", t, re.I | re.S):
            from . import schedcmd
            chat_id = getattr(session, "record_id", None) or getattr(session, "chat_id", "")
            if not schedcmd._open_proposals(self.store, chat_id):
                req = pending[0]
        elif len(pending) >= 2:
            # Two or more ordinary requests are pending (on any thread): with only
            # one pending a bare yes/no is unambiguous (handled above), but with
            # several there is no single "the" pending request to resolve, so an
            # open proposal on THIS thread must not be silently approved underneath
            # that ambiguity -- ask instead (item G). Checked with schedcmd's own
            # narrow yes/no vocabulary, since that is what would otherwise answer
            # the proposal below.
            from . import schedcmd
            chat_id = getattr(session, "record_id", None) or getattr(session, "chat_id", "")
            this_thread_proposals = schedcmd._open_proposals(self.store, chat_id)
            if this_thread_proposals and (schedcmd.is_yes(t) or schedcmd.is_no(t)):
                await session.send_to_sender(
                    f'Yes to which: starting "{this_thread_proposals[0].title}", '
                    f'or one of {len(pending)} pending things?', role="answer")
                return True
            if t.lower() in (_YES | _NO):
                # Several held things and a bare yes/no: never hand the word to the
                # router as a new task (that is what swallowed "Yes" on 2026-11-18).
                # Say which ones are open so he can answer "#id yes".
                lines = [f"#{p.id} {(p.summary or p.original_message or '').strip()[:80]}"
                         for p in pending[:6]]
                await session.send_to_sender(
                    "Yes to which? Reply with the number:\n" + "\n".join(lines), role="answer")
                return True
        if req is not None:
            # A run's question to him (ask_aaron): any text after "#id" is his answer. Every
            # other kind of pending request keeps "yes", "no" and "edit: ..." as they were.
            if ((req.status or {}).get("deferred") or {}).get("kind") == "ask" and rest:
                out = await self.decide_request(req.id, "answer", actor="Aaron",
                                                source=session.mode or "imessage", note=rest)
                if not out.get("ok") and out.get("error"):
                    await self.send_to_approver(out["error"])
                return True
            # Parsing stops here: what the words mean is decided in one place, so a text and a
            # click in the console cannot drift apart.
            low = rest.lower()
            if low in _YES or low in _NO:
                out = await self.decide_request(req.id, "yes" if low in _YES else "no",
                                                actor="Aaron", source=session.mode or "imessage")
                if not out.get("ok") and out.get("error"):
                    await self.send_to_approver(out["error"])
                return True
            em = re.match(r"^edit\s*:\s*(.+)$", rest, re.S | re.I)
            if em:
                await self.decide_request(req.id, "edit", actor="Aaron",
                                          source=session.mode or "imessage", note=em.group(1).strip())
                return True
            return False
        # No pending request on his plate claims this reply (that precedence is
        # exactly the block above, unchanged): a bare yes/no may instead be
        # answering an open schedule proposal on this thread. Everything else
        # about managing an existing schedule (S<id>, titles, pause/resume/...)
        # goes through the router's schedule_command instead of a regex here.
        from . import schedcmd
        reply = await schedcmd.answer_proposal(self, session, t)
        if reply is not None:
            if reply:
                await session.send_to_sender(reply, role="answer")
            return True
        return False

    # -- follow-up phrases on a one-off request (never a schedule) -------------
    # The fixed phrases wording.failure_message tells any sender to say about a
    # request that just failed or is still running. Recognized here, in code,
    # before the router ever sees the message -- after the precedence above
    # (a pending approval or an open schedule proposal on this thread still
    # wins a bare "yes"/"no"), for owner and non-owner senders alike.

    async def handle_followup_phrase(self, session: "GateSession", text: str) -> bool:
        if await self._handle_try_again(session, text):
            return True
        if await self._handle_send_what_you_have(session, text):
            return True
        if session.is_approver():
            # "let it <label>" and the blanket grant are powers only the verified
            # owner has; for anyone else these fall through to the router like any
            # other message (a stranger cannot widen what a request may do).
            if await self._handle_let_it_label(session, text):
                return True
            if await self._handle_blanket_permission(session, text):
                return True
        return False

    def _retry_approval(self, session: "GateSession", req: Request) -> bool:
        """Whether a retry of `req` runs at once: the owner's own always does;
        anyone else's runs at once only if they are trusted for exactly the
        scopes the original request already carried -- the same rule a brand
        new request from them is held to."""
        if session.is_approver():
            return True
        t = session.trust()
        allowed = set(t.get("scopes") or [])
        return bool(allowed) and set(req.scopes or []) <= allowed

    def retry_approval_for_request(self, req: Request) -> bool:
        """The same question _retry_approval answers for a live chat sender, asked
        instead from the request's OWN standing -- used by a retry that has no
        asking sender at all (the console's "try again" button). A console-made
        request is always the owner's own (nothing else can use the console). For
        every other request this rebuilds the identity its original sender had --
        never the console operator's -- so a stranger's request that needed the
        owner's approval the first time still needs it on a retry."""
        if req.mode == "console":
            return True
        probe = GateSession(req.chat_id, self)
        probe.mode = req.mode
        route = self.store.thread_route(req.chat_id)
        meta = dict((route or {}).get("meta") or {})
        meta["sender"] = req.sender
        probe.reply_meta = meta
        if probe.is_approver():
            return True
        t = probe.trust()
        allowed = set(t.get("scopes") or [])
        return bool(allowed) and set(req.scopes or []) <= allowed

    async def _handle_try_again(self, session: "GateSession", text: str) -> bool:
        if not wording.is_try_again(text):
            return False
        chat_id = session.chat_id
        # A schedule's own failed run shares the owner's chat_id; "try again" said
        # about something else entirely must never re-create the schedule's prompt
        # as a one-off (_handle_let_it_label already excludes it the same way).
        # last_failed_for_thread_or_person also finds a failed request that landed
        # on a DIFFERENT thread record for the same person -- a call from the
        # owner's number that never verified as the approver writes its rows to
        # its own chat_id, while his texts land on the owner thread; this is a
        # lookup only, it grants no owner power.
        req = self.store.last_failed_for_thread_or_person(chat_id, session._sender(), exclude_scheduled=True)
        if req is None:
            return False
        now = time.time()
        if not wording.within_retry_window(req.updated_at, now):
            return False
        last_out = self.store.last_outbound_at(chat_id)
        # The failure notice is still the last word on the thread (no newer,
        # unrelated outbound landed after it), or it is recent enough (30 min)
        # that "try again" plainly still means it even if something else was
        # said meanwhile.
        is_latest = last_out is None or last_out <= req.updated_at + 1
        if not (is_latest or (now - req.updated_at) <= 1800):
            return False
        approved = self._retry_approval(session, req)
        new_req = self.retry_request(req, approved=approved, inbound_id=session.inbound_id)
        if new_req is None:
            # The request being retried is itself still approved/running (a race:
            # it had not actually finished failing when this phrase arrived). Never
            # dedupe a retry against a different request -- there is simply nothing
            # to retry yet.
            return True
        if approved:
            asyncio.create_task(self.execute(new_req))
        else:
            self.store.set_thread(chat_id, "awaiting_aaron")
            await self.send_approval_text(new_req)
        return True

    async def _handle_send_what_you_have(self, session: "GateSession", text: str) -> bool:
        if not wording.is_send_what_you_have(text):
            return False
        req = self.store.last_failed_for_thread_or_person(session.chat_id, session._sender())
        if req is None or not wording.within_retry_window(req.updated_at, time.time()):
            return False   # nothing recent it could mean; the message goes on as usual
        text_out = wording.send_what_you_have_text(req.status or {})
        await session.send_to_sender(text_out or "I don't have anything from that run.", role="answer")
        return True

    async def _handle_let_it_label(self, session: "GateSession", text: str) -> bool:
        if not schedprop.looks_like_permission_grant(text):
            return False
        # Same restricted person lookup "try again" uses (item C4): a call's
        # failed request can live on a different thread record than the texted
        # reply granting it permission. is_approver() still gates everything
        # below exactly as it did -- this only widens how the request is found.
        req = self.store.last_failed_for_thread_or_person(session.chat_id, session._sender())
        if req is None or req.schedule_id:
            return False
        denied = wording.denied_scopes(req.status or {})
        if not denied or not wording.within_retry_window(req.updated_at, time.time()):
            return False
        mapped = schedprop.capabilities_from_permission(text, denied)
        if not mapped:
            await session.send_to_sender(
                f"What should it be allowed to do: {schedprop.short_labels(denied)}?", role="answer")
            return True
        new_scopes = [sc for sc in mapped if sc not in req.scopes]
        grown = self.retry_request(req, approved=True, inbound_id=session.inbound_id)
        if grown is None:
            # Already running again (a race); nothing left to widen or start.
            return True
        if new_scopes:
            self.store.set_scopes(grown.id, list(req.scopes) + new_scopes)
            grown = self.store.get_request(grown.id)
        asyncio.create_task(self.execute(grown))
        return True

    async def _handle_blanket_permission(self, session: "GateSession", text: str) -> bool:
        if not wording.is_blanket_permission(text):
            return False
        chat_id = session.chat_id
        now = time.time()
        # Never a schedule's own failed run: the blanket grant would otherwise widen
        # the SCHEDULE's standing scopes from an unrelated "permission for everything".
        # Same restricted person lookup as "try again" (item C4): a call and the
        # text granting blanket permission can land on different thread records.
        failed = self.store.last_failed_for_thread_or_person(chat_id, session._sender(), exclude_scheduled=True)
        if failed is not None and wording.within_retry_window(failed.updated_at, now, hours=0.5):
            await self._grant_blanket_and_retry(session, failed)
            return True
        running = self.store.running_for_thread(chat_id)
        if running is not None:
            self._blanket_pending.add(running.id)
            await session.send_to_sender(
                "Okay. If it gets refused something, I'll let it and continue.", role="answer")
            return True
        await session.send_to_sender(
            "I don't keep a blanket permission. Tell me what to do and I'll ask for anything it needs.",
            role="answer")
        return True

    def _blanket_grant_scopes(self, req: Request, status: Dict[str, Any]) -> List[str]:
        """Every scope this run was refused, plus every read-only scope in the
        registry -- never a scope that sends, posts, writes, deletes, deploys or
        controls anything unless it was actually refused. "read-only" is the
        `read` field scopes.yaml gives each scope (exposed as scopes.READ_SCOPES);
        where a scope carries no such field it is treated as not read-only.

        Every vault scope -- vault_read (the WHOLE vault of Aaron's own notes) and
        each vault:<folder> -- is excluded from the auto-add: none is granted just
        for being read-only, only when the run was actually refused that exact one.
        The vault can hold anything he has ever written down, so a blanket
        "permission for everything" must not open it, folder by folder, as a side
        effect of a run that never asked for it (gate/knowledge.py)."""
        denied = set(wording.denied_scopes(status))
        granted = [sc for sc in SCOPES if sc in denied or
                  (wording.truly_read_only(sc) and not knowledge.blanket_excluded(sc)
                   and sc not in EXPLICIT_ONLY_SCOPES)]
        return [sc for sc in granted if sc not in (req.scopes or [])]

    async def _grant_blanket_and_retry(self, session: "GateSession", req: Request) -> None:
        new_scopes = self._blanket_grant_scopes(req, req.status or {})
        grown = self.retry_request(req, approved=True, inbound_id=session.inbound_id)
        if grown is None:
            # Already running again (a race); nothing left to widen or start.
            return
        if new_scopes:
            self.store.set_scopes(grown.id, list(req.scopes) + new_scopes)
            grown = self.store.get_request(grown.id)
            # Named here: only what the run had actually been refused, never every
            # read-only scope the grant also quietly added alongside it.
            denied_now = [sc for sc in wording.denied_scopes(req.status or {}) if sc in new_scopes]
            if denied_now:
                await session.send_to_sender(
                    f"Okay. For this task it can now {schedprop.short_labels(denied_now)}. Continuing.",
                    role="answer")
            else:
                # Only read-only scopes were auto-added (nothing was actually denied
                # to name); still confirm, without claiming a capability was refused.
                await session.send_to_sender("Okay. Continuing.", role="answer")
        asyncio.create_task(self.execute(grown))

    # -- executor --------------------------------------------------------------
    async def _grant_what_it_asked_for(self, req: Request, status: Dict[str, Any], context: str,
                                       model: Optional[str]) -> Dict[str, Any]:
        """A run that stops because it was not given the tool it needed has diagnosed itself
        better than any judgment could. The scopes were a guess made before the work began;
        this is the work reporting what the guess missed.

        For the owner's own requests the named scope is granted and the run repeated once.
        For anyone else the scopes are what he approved, so it is reported rather than widened."""
        if (status.get("confirm") or status.get("access_request") or status.get("ok")
                or "mcp__host__request_access" in (status.get("tool_calls") or [])):
            return status
        said = f"{status.get('summary') or ''}\n{status.get('error') or ''}\n{status.get('raw') or ''}"
        refused = set((req.status or {}).get("access_refused") or [])
        wanted = [sc for sc in SCOPES if sc not in req.scopes and sc not in refused and re.search(
            r"\b" + re.escape(sc) + r"\b", said)]
        if not wanted or not _owner_request(self, req) or req.schedule_id or (req.status or {}).get("followup_id"):
            return status
        grown = list(req.scopes) + wanted
        logger.info("[gate] #%s stopped for want of %s; granting and running again",
                    req.id, ", ".join(wanted))
        self.store.set_scopes(req.id, grown)
        again = self.store.get_request(req.id) or req
        out = await self.executor.run(again, context=context, model=model,
                                      from_owner=True)
        out["widened_scopes"] = wanted
        return out

    def _answered_from_attached_notes(self, req: Request) -> bool:
        """True when this request was given vault notes in its prompt (knowledge scopes
        with a non-empty attachment), so an answer without a tool call can be real."""
        if not any(str(sc).startswith("vault:") for sc in (req.scopes or [])):
            return False
        try:
            return bool(self.vault_attachment_for(req))
        except Exception:
            logger.exception("[gate] could not rebuild the vault attachment for #%s", req.id)
            return False

    async def _insist_it_actually_ran(self, req: Request, status: Dict[str, Any], context: str,
                                      model: Optional[str]) -> Dict[str, Any]:
        """A request exists because a tool was needed. Reporting success without having
        called one means the work was not done, whatever the words say.

        The usual way this happens: the task record holds the answer an earlier run gave,
        and that reads as the question already being settled. The record is evidence about
        the past, not a substitute for looking now. One retry says so in as many words; a
        second refusal to act is reported as a failure rather than dressed up as a result."""
        if status.get("confirm") or status.get("access_request") or not status.get("ok") or status.get("tool_calls"):
            return status
        if self._answered_from_attached_notes(req):
            # His own notes were attached to this run's prompt and the answer came from
            # them: nothing needed looking up, so no tool call is the right outcome.
            return status
        if not req.scopes and not req.schedule_id and not (req.status or {}).get("followup_id"):
            # Nothing was granted to look anything up with: the ask is thinking (names,
            # a draft, an estimate) and the words are the work. Insisting on a tool here
            # threw away finished answers and reported them as failures.
            return status
        logger.warning("[gate] #%s reported success with no tool call; insisting it runs", req.id)
        insist = (context + "\n\n--- THIS RUN ---\nThe last attempt returned without using a tool and "
                  "said the work was already done. It was not. Whatever the record above says was "
                  "answered before, answer it again now by looking, and report what you find this "
                  "time. If a tool you need is not available to you, say which one and stop.")
        run_kwargs: Dict[str, Any] = {}
        if req.schedule_id:
            sched = self.store.get_schedule(req.schedule_id)
            if sched is not None:
                run_kwargs = {"timezone": sched.timezone, "note": sched.last_note}
        again = await self.executor.run(req, context=insist, model=model,
                                        from_owner=_owner_request(self, req),
                                        **run_kwargs)
        if again.get("tool_calls") or again.get("confirm"):
            return again
        return {**again, "ok": False,
                "error": "finished without doing anything: it read the task record as the answer",
                "summary": (again.get("summary") or "") + "\n\n(Nothing was actually looked up. "
                            "The task record was treated as the answer.)"}

    async def _run_executor(self, req: Request, context: str, *, continuation: str = "",
                            resume_session: Optional[str] = None) -> Dict[str, Any]:
        """Run one request and return exactly ONE status object. Claude Code is an
        escalation inside this call, not a second execution: nothing is delivered,
        noted, or phrased for the Jev attempt when it escalates; only the final
        status leaves this function. Executors themselves never message anyone who
        asked (denied at the tool call).

        Scheduled and follow-up runs take the Claude Code branch so the closing
        text can say whether anything changed. The Jev agent has no time zone,
        no carried-forward note, no
        scheduled-run prompt, no RESULT/NOTE/STATUS closing-line contract and
        no third-party-send hold, all of which the scheduled-run contract
        (schedrun.py, schedules.py) requires."""
        if continuation or not self.jev_agent or req.schedule_id or (req.status or {}).get("followup_id"):
            model = await self._pick_model(req)
            run_kwargs: Dict[str, Any] = {}
            if req.schedule_id:
                sched = self.store.get_schedule(req.schedule_id)
                if sched is not None:
                    run_kwargs = {"timezone": sched.timezone, "note": sched.last_note}
            if continuation:
                run_kwargs["continuation"] = continuation if resume_session else ""
                run_kwargs["resume_session"] = resume_session or ""
                if not resume_session and (req.status or {}).get("raw"):
                    run_kwargs["prior_work"] = str(req.status["raw"])
            status = await self.executor.run(req, context=context, model=model,
                                             from_owner=_owner_request(self, req), **run_kwargs)
            if status.get("quota_error"):
                return await self._run_quota_fallback(req, context, status)
            if continuation:
                return status
            if (status.get("access_granted") and
                    any(d.get("reason") == "scope" for d in status.get("denied") or [])):
                # A newly covered MCP server is loaded by the next run. Keep the
                # completed work in view so Claude can continue from the refusal.
                fresh = self.store.get_request(req.id) or req
                status = await self.executor.run(
                    fresh, context=context, prior_work=status.get("raw") or "",
                    model=model, from_owner=_owner_request(self, fresh), resume_session="")
            if status.get("deferred"):
                return status
            status = await self._grant_what_it_asked_for(req, status, context, model)
            return await self._insist_it_actually_ran(req, status, context, model)
        status = await self.jev_agent.run(req, context=context)
        logger.info("[gate] jev agent #%s: ok=%s steps=%s jev=%s prose=%s %ss",
                    req.id, status.get("ok"), status.get("tool_calls"), status.get("jev_calls"),
                    status.get("prose_calls"), status.get("seconds"))
        if status.get("confirm") or status.get("access_request") or not self.jev_fallback:
            return status
        if status.get("ok"):
            # Fast is only worth it when right: a draft answer that the grounding judge
            # cannot support in the findings sends the request on to Claude Code, with
            # the findings, instead of a "could not" reaching the sender.
            if await self._findings_answer_the_goal(req, status):
                return status
            if status.get("wrote"):
                return status           # a write happened; never redo it
            logger.info("[gate] jev agent #%s finished but its findings do not answer the goal; escalating", req.id)
        if status.get("wrote"):
            logger.info("[gate] jev agent #%s failed after a write; not falling back", req.id)
            return status
        logger.info("[gate] jev agent gave up on #%s (%s); falling back to claude code", req.id, status.get("error"))
        findings = status.get("raw") or ""
        fallback = await self.executor.run(req, context=context, prior_work=findings,
                                           from_owner=_owner_request(self, req))
        if fallback.get("quota_error"):
            return await self._run_quota_fallback(req, context, fallback)
        fallback["escalated"] = True
        fallback["jev_attempt"] = {k: status.get(k) for k in ("error", "tool_calls", "jev_calls", "prose_calls", "seconds", "steps", "raw")}
        return fallback

    async def _run_quota_fallback(self, req: Request, context: str,
                                  quota_status: Dict[str, Any]) -> Dict[str, Any]:
        """Use the always-constructed Jev caller while Claude Code is cooling down."""
        if os.getenv("TYPESAFE_API_KEY") and getattr(self, "caller", None) is not None:
            try:
                status = await self.caller.run(req, context=context)
                if status.get("ok") or status.get("tool_calls"):
                    status["quota_fallback"] = True
                    return status
            except Exception:  # noqa: BLE001
                logger.exception("[gate] jev quota fallback failed for request #%s", req.id)
        reset_time = quota_status.get("reset_time")
        when = f" until {reset_time}" if reset_time else " until it resets"
        message = f"{_ASSISTANT}'s model quota is exhausted{when}; try again after that."
        return {
            **quota_status,
            "ok": False,
            "quota_error": True,
            "error": message,
            "summary": message,
            "final_status": message,
        }

    SLOW_ACK_AFTER_S = 7.0      # the default; the live value is read per run below

    async def _ack_if_slow(self, session: "GateSession", req: Request) -> None:
        """A request that is still running after a few seconds gets one short, casual line
        so the wait does not read as silence. A fast one gets its answer and nothing else."""
        try:
            await asyncio.sleep(float(_settings.get("GATE_SLOW_ACK_AFTER_S") or self.SLOW_ACK_AFTER_S))
        except asyncio.CancelledError:
            return
        if self.store.get_request(req.id).state != "running":
            return
        line, session.deferred_ack = session.deferred_ack, None
        if not line:
            return                      # the writer gave no acknowledgement; the typing bubble is enough
        try:
            async with session._lock:
                prev = session.inbound_id
                session.inbound_id = req.inbound_id or prev
                try:
                    await session.send_to_sender(line, role="ack")
                finally:
                    session.inbound_id = prev
            session.typing_start()           # the send stopped the pulse; the run continues
        except Exception:
            logger.debug("[gate] slow ack failed", exc_info=True)

    STRONG_MODEL = "opus"       # defaults; the live values are read per request below
    FAST_MODEL = "sonnet"

    async def _pick_model(self, req: Request) -> str:
        """Claude Code runs the request; Jev decides which Claude: the strongest model for
        work that combines sources, changes many things, or writes at length; the fast one
        for a lookup or a single plain change."""
        picker = self.task_picker
        if picker is None or not hasattr(picker, "judge_complexity"):
            return _settings.get("GATE_FAST_MODEL") or self.FAST_MODEL
        res = self._complexity_results.pop(int(req.id), None)
        if res is None:
            res = await picker.judge_complexity(prompt=req.prompt, summary=req.summary, scopes=list(req.scopes))
        model = (_settings.get("GATE_STRONG_MODEL") or self.STRONG_MODEL) if res.get("complex") \
            else (_settings.get("GATE_FAST_MODEL") or self.FAST_MODEL)
        logger.info("[gate] #%s model: %s (p(complex)=%.2f)", req.id, model, float(res.get("p") or 0.0))
        return model

    async def _findings_answer_the_goal(self, req: Request, status: Dict[str, Any]) -> bool:
        """Draft the answer from the findings and let the grounding judge say whether the
        findings support it. Undecided counts as yes (no judge, or too large to judge)."""
        picker = self.task_picker
        if picker is None or not hasattr(picker, "judge_grounded"):
            return True
        summary = status.get("summary") or ""
        try:
            draft = await self.router.phrase(
                "Answer the question below from the findings, in one to three plain sentences. If the findings do "
                "not contain the answer, say exactly what they do and do not show.",
                {"question": req.original_message, "findings": summary[-20000:]})
            res = await picker.judge_grounded(reply=draft, results=summary, question=req.original_message)
        except Exception:
            logger.debug("[gate] findings check failed", exc_info=True)
            return True
        p = float(res.get("p") or 0.0)
        ok = res.get("grounded") is not False and (res.get("grounded") is not None or True)
        answered = (res.get("grounded") is None) or p >= 0.5
        logger.info("[gate] jev agent #%s findings check: p(answered)=%.2f -> %s", req.id, p, "keep" if answered else "escalate")
        status["draft_answer"] = draft
        return answered

    def remember_knowledge_notes(self, request_id: int, notes: Optional[List[str]]) -> None:
        if notes:
            self._knowledge_notes[int(request_id)] = list(notes)
            while len(self._knowledge_notes) > 500:
                self._knowledge_notes.pop(next(iter(self._knowledge_notes)))

    def remember_complexity(self, request_id: int, result: Any) -> None:
        if isinstance(result, dict):
            self._complexity_results[int(request_id)] = dict(result)
            while len(self._complexity_results) > 500:
                self._complexity_results.pop(next(iter(self._complexity_results)))

    def vault_attachment_for(self, req: Request, *, ledger_chars: int = 0) -> str:
        """The owner's vault notes this approved request's knowledge scopes cover,
        as a delimited block for the run's context. Computed fresh from req.scopes
        at run time (never from req.prompt, which is hash-locked from approval), so
        it cannot be widened by anything that happened between approval and running:
        the scopes it reads here are exactly the ones Aaron's approval (or his own
        trust) already covers. Holding the whole-vault vault_read scope as well changes
        only what the vault tools may open afterwards, not what is attached here.

        A note is attached whole only up to vaultscopes.SMALL_NOTE_CHARS; past that
        it is cut to the sections/bullets that match the request (see
        vaultscopes.build_relevant_attachment), so a run's vault attachment carries
        the relevant paragraphs of six notes rather than all six of them whole.
        `ledger_chars` is how much of the run's shared context budget the task
        ledger already used, so the vault gets whatever is left of it."""
        names = [sc for sc in (req.scopes or []) if str(sc).startswith("vault:")]
        if not names:
            return ""
        from . import vaultscopes
        from . import settings as _settings
        root = vaultscopes.vault_dir()
        discovered = vaultscopes.discover_cached()
        covered: List[str] = []
        for name in names:
            info = discovered.get(name)
            if info is None:
                continue
            for p in vaultscopes.notes_under(root, info):
                rel = str(p.relative_to(root))
                if rel not in covered:
                    covered.append(rel)
        if not covered:
            return ""
        # The notes the traversal chose come first, most relevant first, and only those
        # the granted knowledge scopes cover. Without that list (a restart since the
        # decision) every note the scopes cover is offered, in the vault's own order.
        chosen = [n for n in self._knowledge_notes.get(int(req.id), []) if n in covered]
        notes = chosen or covered
        if req.mode == "voice":
            notes = notes[:2]
        total_budget = int(_settings.get("GATE_CONTEXT_BUDGET_CHARS") or 8000)
        ceiling = int(_settings.get("GATE_VAULT_ATTACH_BUDGET_CHARS") or 60000)
        budget = max(500, min(ceiling, total_budget - max(0, ledger_chars)))
        attachment = vaultscopes.build_relevant_attachment(
            root, notes, prompt=req.prompt or "", summary=req.summary or "", budget_chars=budget)
        if attachment.overflow or attachment.dropped_units:
            logger.info("[gate] #%s vault attachment: %d notes attached (%d chars), %d over budget "
                        "listed by path, sections trimmed in %s",
                        req.id, len(attachment.included), len(attachment.block),
                        len(attachment.overflow),
                        {p: f"{d}/{t}" for p, (d, t) in attachment.dropped_units.items()} or "none")
        return attachment.block

    def retry_request(self, req: Request, *, approved: bool,
                      inbound_id: Optional[int] = None) -> Optional[Request]:
        """Re-create a failed request exactly as it was asked: the same prompt,
        scopes and task, through the store's own create_request_once -- never a
        shortcut around approval. The caller recognizes "try again" on an inbound
        message (wording.is_try_again, within wording.within_retry_window of the
        failure notice) and decides `approved` the same way any new request's
        approval is decided for that sender; when it is False, the caller still owes
        this fresh pending request whatever a new pending request normally gets
        (an approval text to the owner), the same as request creation always has.

        Bypasses the sender-scoped dedupe entirely (a retry recreates a request for
        the exact sender `req` already belongs to, so that dedupe would otherwise
        fold the retry into some other approved/running request of theirs that
        happened to race it, or even into `req` itself before its own "failed" state
        had committed). In its place: refuses -- returns None -- when `req` itself
        is, right now, already approved or running. That is the one case a retry
        must never hand back a second, concurrent run of the same work; the caller
        treats None as "nothing to retry yet" and does not acknowledge, execute, or
        widen scopes for anything."""
        task_id = self.store.task_id_for_request(req.id)
        if not task_id:
            raise TaskRequired(f"request #{req.id} has no task to retry under")
        fresh = self.store.get_request(req.id) or req
        if fresh.state in ("approved", "running"):
            logger.info("[gate] retry of #%s refused: it is already %s", req.id, fresh.state)
            return None
        new_req, _created = self.store.create_request_once(
            chat_id=req.chat_id, sender=req.sender, sender_name=req.sender_name, mode=req.mode,
            subject=req.subject, original_message=req.original_message, summary=req.summary,
            scopes=list(req.scopes), prompt=req.prompt, state="approved" if approved else "pending",
            # A retry is started by a new message ("try again", "let it ..."): its result
            # answers that message. Tied to the original ask, which already got the
            # failure, the result would be dropped as a second answer.
            task_id=task_id, inbound_id=inbound_id or req.inbound_id, bypass_dedupe=True)
        return new_req

    async def deliver_late_result(self, req: Request, status: Dict[str, Any]) -> None:
        """A run that outran its live timeout has now actually finished. Deliver the
        real result the same way execute() would have, once, on whatever channel is
        still owed an answer -- notify_after_request/notify_failure already refuse to
        send a second answer to an inbound that already has one, so this is safe even
        if the fast timeout path already told the person something in the meantime."""
        fresh = self.store.get_request(req.id)
        req = fresh or req
        if status.get("deferred") and status["deferred"].get("kind") == "ask":
            try:
                await self.handle_ask(req, status)
                return
            except Exception as exc:
                logger.warning("[gate] #%s ask could not be sent: %s", req.id, exc)
                status = {**status, "ok": False, "error": "the question for Aaron could not be sent"}
                status.pop("deferred", None)
        if status.get("deferred") and status["deferred"].get("kind") == "access":
            try:
                next_prompt = await self.handle_access_ask(req, status)
            except Exception as exc:
                logger.warning("[gate] #%s access decision failed: %s", req.id, exc)
                status.pop("deferred", None)
                status.pop("access_request", None)
                next_prompt = False
            if next_prompt is None:
                return
            if next_prompt is not False:
                fresh = self.store.get_request(req.id) or req
                saved = dict(fresh.status or {})
                saved["deferred_decision"] = {"id": status["deferred"]["id"], "allow": True}
                self.store.set_state(req.id, "approved", status=saved)
                fresh = self.store.get_request(req.id) or req
                asyncio.create_task(self.execute(fresh, continuation="continue",
                                                 resume_session=status["deferred"].get("session_id") or ""))
                return
        if status.get("access_request"):
            try:
                next_prompt = await self.handle_access_ask(req, status)
            except Exception as exc:
                logger.warning("[gate] #%s access decision failed: %s", req.id, exc)
                status.pop("access_request", None)
                next_prompt = False
            if next_prompt is not None:
                if next_prompt is not False:
                    fresh = self.store.get_request(req.id) or req
                    saved_session = (fresh.status or {}).get("session_id") or status.get("session_id")
                    asyncio.create_task(self.execute(fresh,
                                                     continuation=next_prompt if saved_session else "",
                                                     resume_session=saved_session))
            if next_prompt is not False:
                return
        if status.get("confirm"):
            await self.ask_confirmation(req, status)
            return
        ok = bool(status.get("ok"))
        self.store.set_state(req.id, "done" if ok else "failed", status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "idle")
        if req.schedule_id and (req.status or {}).get("deferred"):
            if ((req.status or {}).get("deferred_decision") or {}).get("allow"):
                await self.schedules.held_resolved(req, status)
            return
        title = wording.plain_title(self.store, req)
        result = (status.get("summary") or status.get("error") or "").strip()
        message = wording.success_message(result) if ok else wording.failure_message(title, status)
        outcome = "done" if ok else "failed"
        note = (f"Task #{req.id} {outcome} (after running past the live-call budget in the background): "
                f"{req.summary}\nResult:\n{result}")
        self.task_note(req, "done" if ok else "failed", f"{req.summary} -> {result}",
                       state="done" if ok else "failed")
        session = self.get(req.chat_id)
        if (not (req.status or {}).get("followup_id") and
                await self._ended_by_its_own_choice(req, status, session, title, message,
                                                    self._is_owner_request(req))):
            return
        previous_kind = getattr(session, "_outbox_kind", "reply")
        session._outbox_kind = "late_result"
        try:
            if ok:
                delivered = await session.notify_after_request(note, inbound_id=req.inbound_id)
            else:
                delivered = await session.notify_failure(message, note, inbound_id=req.inbound_id)
            if not delivered:
                await self.send_to_approver(self.result_notice(req, title, message, ok=ok, late=True,
                                                               own=self._is_owner_request(req)),
                                            kind="late_result")
        except Exception:
            logger.exception("[gate] #%s late delivery failed", req.id)
        finally:
            session._outbox_kind = previous_kind

    async def _fallback_deferred(self, req: Request, context: str) -> Optional[Dict[str, Any]]:
        """A lost Claude session uses the older single-call path or starts access work afresh."""
        deferred = (req.status or {}).get("deferred") or {}
        decision = (req.status or {}).get("deferred_decision") or {}
        try:
            self.executor._sessions.pop(req.chat_id, None)
            self.store.drop_session(req.chat_id)
        except Exception as exc:
            logger.warning("[gate] #%s could not clear saved session: %s", req.id, exc)
        if deferred.get("kind") in ("send", "destructive"):
            if decision.get("allow"):
                await self.perform_confirmed(req)
            else:
                self.store.set_state(req.id, "rejected")
                self.store.set_thread(req.chat_id, "idle")
                if req.schedule_id:
                    await self.schedules.held_ended(req, "declined")
                else:
                    await self.send_to_approver(f"Left alone. Nothing was changed for #{req.id}.")
            return None
        fresh = self.store.get_request(req.id) or req
        saved = dict(fresh.status or {})
        saved.pop("deferred", None)
        saved.pop("deferred_decision", None)
        saved.pop("access_request", None)
        self.store.set_state(req.id, "running", status=saved)
        fresh = self.store.get_request(req.id) or req
        model = await self._pick_model(fresh)
        again = ""
        if deferred.get("kind") == "ask" and decision.get("answer"):
            # The saved run is gone, so the task starts over with his answer in hand.
            again = (f"{fresh.prompt}\n\nYou already asked Aaron: {(deferred.get('input') or {}).get('question') or ''}\n"
                     f"Aaron said: {decision['answer']}\nUse that answer; do not ask it again.")
        return await self.executor.run(fresh, context=context, model=model,
                                       from_owner=_owner_request(self, fresh), resume_session="",
                                       continuation=again)

    async def execute(self, req: Request, notify: bool = True, *, continuation: str = "",
                      resume_session: Optional[str] = None) -> str:
        """Run an approved request. With notify=False the caller delivers the
        result itself (a live phone call reads it aloud) and gets it back.

        The run is bound to its request id (runguard) so every send and write made for
        it can check, right before acting, that Aaron has not stopped it; the top-level
        task is remembered so stop_request() can cancel it outright."""
        if runguard.is_cancelled(req.id):
            logger.info("[gate] #%s was stopped by Aaron; not running it", req.id)
            return ""
        token = runguard.bind(req.id)
        task = asyncio.current_task()
        outer = req.id not in self._run_tasks and task is not None
        if outer:
            self._run_tasks[req.id] = task  # type: ignore[assignment]
        try:
            return await self._execute(req, notify, continuation=continuation,
                                       resume_session=resume_session)
        finally:
            runguard.unbind(token)
            if outer and self._run_tasks.get(req.id) is task:
                self._run_tasks.pop(req.id, None)

    async def stop_request(self, req: Request, *, by: str = "aaron") -> str:
        """Aaron said stop. Mark it cancelled first (every later send or write for it is
        refused from this instant), cancel the task it runs on, and return the one line
        for him: stopped, and what had already gone out, if anything."""
        runguard.cancel(req.id)
        self.store.cancel_request(req.id, by=by)
        self.store.set_thread(req.chat_id, "idle")
        task = self._run_tasks.pop(req.id, None)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        try:
            self.get(req.chat_id).typing_stop()
        except Exception:
            logger.debug("[gate] typing stop after #%s failed", req.id, exc_info=True)
        try:
            self.task_note(req, "rejected", f"Aaron stopped it while it ran: {req.summary}", state="open")
        except Exception:
            logger.debug("[gate] could not note the stop of #%s", req.id, exc_info=True)
        sent = runguard.sent(req.id)
        logger.warning("[gate] #%s stopped by %s while %s; already sent: %s", req.id, by, req.state, sent or "nothing")
        return wording.stopped_line(req.id, sent)

    async def _execute(self, req: Request, notify: bool = True, *, continuation: str = "",
                       resume_session: Optional[str] = None) -> str:
        self.store.set_state(req.id, "running")
        # The revision this run started from, as _fire recorded it on the request's own
        # status -- the `req` object here can be the one created before _fire wrote that
        # status (stale in memory), so re-read it from the store rather than trust req.status.
        schedule_revision = (req.status or {}).get("schedule_revision")
        if schedule_revision is None and req.schedule_id:
            fresh = self.store.get_request(req.id)
            schedule_revision = (fresh.status or {}).get("schedule_revision") if fresh else None
        fresh = self.store.get_request(req.id) if req.schedule_id else None
        usage_retry = ((fresh.status or {}).get("usage_retry", False) if fresh else False)
        session = self.get(req.chat_id)
        slow_ack: Optional[asyncio.Task] = None
        if notify and req.mode in ("imessage", "telegram"):
            session.typing_start()
            slow_ack = asyncio.create_task(self._ack_if_slow(session, req))
        tid = self.store.task_id_for_request(req.id)
        ledger_info: Dict[str, int] = {}
        if tid:
            ledger_budget = max(1500, int(int(_settings.get("GATE_CONTEXT_BUDGET_CHARS") or 8000) * 0.4))
            context, ledger_info = self.store.ledger_context_for_execution(
                tid, req.original_message or req.prompt or "", budget_chars=ledger_budget)
        else:
            context = ""
        ledger_chars = len(context)
        vault_block = self.vault_attachment_for(req, ledger_chars=ledger_chars)
        if vault_block:
            context = f"{context}\n\n{vault_block}" if context else vault_block
        logger.info("[gate] executing #%s with ledger T%s (%d chars, pool %s dropped %s matched %s) "
                    "vault attachment %d chars",
                    req.id, tid, ledger_chars,
                    ledger_info.get("pool"), ledger_info.get("dropped"), ledger_info.get("matched"),
                    len(vault_block))
        if continuation and (req.status or {}).get("deferred_decision") and not resume_session:
            try:
                status = await self._fallback_deferred(req, context)
            except Exception as exc:
                logger.warning("[gate] #%s could not continue held call: %s", req.id, exc)
                status = {"ok": False, "error": "could not continue the saved call", "tool_calls": []}
            if status is None:
                if slow_ack is not None:
                    slow_ack.cancel()
                session.typing_stop()
                return ""
        else:
            status = None
        try:
            if status is not None:
                pass
            elif continuation:
                status = await self._run_executor(req, context, continuation=continuation,
                                                  resume_session=resume_session)
            else:
                status = await self._run_executor(req, context)
        finally:
            if slow_ack is not None and not slow_ack.done():
                slow_ack.cancel()
            if not notify:
                session.typing_stop()
        if runguard.is_cancelled(req.id):
            # Aaron stopped it while the executor ran: nothing is recorded, phrased or sent.
            session.typing_stop()
            return ""
        if status.get("resume_failed") and (req.status or {}).get("deferred_decision"):
            try:
                status = await self._fallback_deferred(req, context)
            except Exception as exc:
                logger.warning("[gate] #%s could not continue held call: %s", req.id, exc)
                status = {"ok": False, "error": "could not continue the saved call", "tool_calls": []}
            if status is None:
                return ""
        if status.get("deferred") and status["deferred"].get("kind") == "ask":
            session.typing_stop()
            try:
                await self.handle_ask(req, status)
                return "I'm checking with Aaron and will get back to you."
            except Exception as exc:
                # Nobody is left waiting on a question that never went out.
                logger.warning("[gate] #%s ask could not be sent: %s", req.id, exc)
                status = {**status, "ok": False, "error": "the question for Aaron could not be sent"}
                status.pop("deferred", None)
        if status.get("deferred") and status["deferred"].get("kind") == "access":
            session.typing_stop()
            try:
                next_prompt = await self.handle_access_ask(req, status)
            except Exception as exc:
                logger.warning("[gate] #%s access decision failed: %s", req.id, exc)
                status.pop("deferred", None)
                status.pop("access_request", None)
                next_prompt = False
            if next_prompt is None:
                return "I'm checking with Aaron and will get back to you."
            if next_prompt is not False:
                fresh = self.store.get_request(req.id) or req
                saved = dict(fresh.status or {})
                saved["deferred_decision"] = {"id": status["deferred"]["id"], "allow": True}
                self.store.set_state(req.id, "approved", status=saved)
                fresh = self.store.get_request(req.id) or req
                return await self.execute(fresh, notify=notify, continuation="continue",
                                          resume_session=status["deferred"].get("session_id") or "")
        if status.get("access_request"):
            session.typing_stop()
            try:
                next_prompt = await self.handle_access_ask(req, status)
            except Exception as exc:
                logger.warning("[gate] #%s access decision failed: %s", req.id, exc)
                status.pop("access_request", None)
                next_prompt = False
            if next_prompt is None:
                return "I'm checking with Aaron and will get back to you."
            if next_prompt is not False:
                fresh = self.store.get_request(req.id) or req
                saved_session = (fresh.status or {}).get("session_id") or status.get("session_id")
                return await self.execute(fresh, notify=notify,
                                          continuation=next_prompt if saved_session else "",
                                          resume_session=saved_session)
        if status.get("outcome") == "held" and not req.schedule_id:
            # The run's message to someone else is held for Aaron's yes and he has been
            # shown it with its own "#id yes / no". That prompt is the only thing he gets:
            # the run is not a failure, it waits, and his yes sends it and finishes it.
            session.typing_stop()
            status["ok"] = True
            self.store.set_state(req.id, "waiting", status=status, raw_output=status.get("raw"))
            self.store.set_thread(req.chat_id, "idle")
            self.task_note(req, "request", "Waiting on Aaron's yes for the held message.", state="waiting_aaron")
            logger.info("[gate] #%s ended waiting on %d held message(s)", req.id, len(status.get("held_sends") or []))
            return ""
        if status.get("confirm"):
            session.typing_stop()
            await self.ask_confirmation(req, status)
            return "I need your yes before I change anything; I have texted you what it is."
        ok = bool(status.get("ok"))
        if schedule_revision is not None:
            status["schedule_revision"] = schedule_revision
            status["usage_retry"] = usage_retry
        self.store.set_state(req.id, "done" if ok else "failed",
                             status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "idle")
        if (req.status or {}).get("followup_id"):
            # The timer owns delivery. This run is already approved, but it has no
            # inbound message to answer and must not close the ongoing task.
            self.task_note(req, "done" if ok else "failed",
                           f"Follow-up checked: {status.get('result_line') or status.get('final_status') or status.get('error') or ''}",
                           state="open")
            return ""
        if req.id in self._blanket_pending:
            # "You have my permission for everything" was said about this request
            # while it was still running (item 4): it applies to this request's
            # own retry, once, the moment it actually fails and shows what it was
            # refused. A run that succeeds leaves the grant unconsumed and moot.
            self._blanket_pending.discard(req.id)
            if not ok:
                await self._grant_blanket_and_retry(session, self.store.get_request(req.id) or req)
                session.typing_stop()
                return ""
        outcome = "done" if ok else "failed"
        result = _clean_result(status.get("summary") or status.get("error") or "")
        title = wording.plain_title(self.store, req)
        # The message a person actually reads for this result: the cleaned result on
        # success, or a plain failure message (never the raw SDK/provider error, never
        # "That failed.") on failure. `result`/`note` below stay as they were -- the
        # grounding fallback downstream (ground_reply) still works from the run's own
        # findings -- only the text actually sent or returned here uses `message`.
        message = wording.success_message(result) if ok else wording.failure_message(title, status)
        # Item 7's remainder: a usage-limit failure is told once, not once per ask.
        # If this thread already heard one within the last 10 minutes, this one
        # still fails the same way but says nothing further -- the raw status and
        # the plain-worded message both still land on the ledger/log via `note`
        # below, just never a second time in front of a person.
        told_already = False
        if not ok and wording.is_usage_limit_failure(status):
            prior = self.store.recent_failed_statuses_for_thread(
                req.chat_id, window_seconds=600, exclude_request_id=req.id)
            told_already = any(wording.is_usage_limit_failure(st) for st in prior)
        self.task_note(req, "done" if ok else "failed", f"{req.summary} -> {result}",
                       state="open" if req.schedule_id else ("done" if ok else "failed"))
        if req.schedule_id and (req.status or {}).get("deferred"):
            if ((req.status or {}).get("deferred_decision") or {}).get("allow"):
                await self.schedules.held_resolved(req, status)
            session.typing_stop()
            return ""
        if await self.schedules.finished(req, status):
            # A message the owner sent while this ran already got its own turn when it
            # arrived (store.running_for_thread ignores schedule requests, so it was never
            # held). Nothing further to do with it here.
            session.typing_stop()
            return ""
        if not notify:
            self.store.add_message(req.chat_id, "system", f"Task #{req.id} {outcome}: {req.summary}\nResult:\n{result}")
            if told_already:
                return wording.same_limit_message(status)
            return message
        from_aaron = self._is_owner_request(req)
        if await self._ended_by_its_own_choice(req, status, session, title, message, from_aaron):
            return ""
        # Item C1: a request born from a call by the owner's own number, with voice not
        # trusted as the approver, is approved later on text -- by the time this runs,
        # the session for req.chat_id (the call's own chat_id) no longer reads as a live
        # call or as the owner (self.mode has long since reset). Its result still has to
        # reach him, once, on a channel that actually exists: his own text channel,
        # first person, never the "[Blatbot #N ...]" framing a stranger's request gets,
        # and never a send through this stale session (which has no voice channel open
        # any more and may hold a bare contact id instead of a real one).
        from_owner_call = (from_aaron and req.mode == "voice"
                          and req.chat_id not in self._approver_chat_ids() and req.chat_id != OWNER_THREAD)
        note = f"Task #{req.id} {outcome}: {req.summary}\nResult:\n{result}"
        # Messages live on the person's record, which for the owner spans every surface.
        meanwhile = self.store.inbound_since(self.get(req.chat_id).record_id, req.created_at)
        if from_owner_call:
            # Delivered once, directly, in the first person (wording.py's own text
            # already reads that way), never through this stale session's
            # mode-dependent send_to_sender and never with the "[Blatbot #N ...]"
            # third-person framing a stranger's request gets.
            record_id = self.get(req.chat_id).record_id
            self.store.add_message(record_id, "system", note)
            already = req.inbound_id and "answer" in self.store.responses_to(req.inbound_id)
            if not already:
                text_out = wording.same_limit_message(status) if (told_already and not ok) else message
                await self.send_to_approver(text_out)
                if req.inbound_id:
                    self.store.add_message(record_id, "outbound", text_out, "imessage",
                                           reply_to=req.inbound_id, role="answer")
            session.typing_stop()
            return ""
        if from_aaron:
            if not ok:
                # A failed request's message is always wording.py's own text, on every
                # branch: notify_failure sends `message` exactly as written and never
                # hands it to the router/model (which is how the raw status error used
                # to leak through, and how a correctly-worded failure line got rewritten
                # into something the grounding check then threw away). The raw status
                # still lands on the ledger/log via `note`.
                if told_already:
                    same_limit = wording.same_limit_message(status)
                    replied = await session.notify_failure(same_limit, note, inbound_id=req.inbound_id)
                    if not replied:
                        await self.send_to_approver(same_limit)
                else:
                    replied = await session.notify_failure(message, note, inbound_id=req.inbound_id)
                    if not replied:
                        await self.send_to_approver(self.result_notice(req, title, message, ok=False, own=True))
                if meanwhile:
                    # Aaron said more while it ran (a hint, a correction): the failure
                    # above is the answer to the request itself; this is a separate,
                    # later turn for whatever he said after that.
                    logger.info("[gate] #%s failed with %d message(s) received meanwhile; handling the latest",
                                req.id, len(meanwhile))
                    try:
                        await session.handle_followup(meanwhile[-1]["text"], inbound_id=meanwhile[-1]["id"])
                    except Exception:
                        logger.exception("[gate] follow-up after #%s failed", req.id)
                session.typing_stop()
                return ""
            # Aaron asked for it himself: let the router phrase the answer.
            try:
                replied = await session.notify_after_request(note, inbound_id=req.inbound_id)
                if not replied and session.room:
                    # He asked in a room, with the people it concerns in it: the answer
                    # belongs there, in the run's own words. Not in his DMs, addressed to
                    # someone else ("Stanley, here's Aaron's repo" landing on his iMessage).
                    logger.info("[gate] #%s: no phrased reply for the room; posting the result there", req.id)
                    replied = await session.send_to_sender(message, role="answer")
                if not replied:
                    await self.send_to_approver(self.result_notice(req, title, message, ok=True, own=True))
            finally:
                session.typing_stop()
            return ""
        # Someone else's request: short status to Aaron, and the router may
        # tell the sender the outcome -- except on failure, which is always
        # wording.py's own text, never the router's rewrite of it.
        # Aaron is in this group himself and reads the result there: a copy in his own
        # thread is noise.
        in_room = self.owner_in_group(session)
        if not ok:
            if told_already:
                same_limit = wording.same_limit_message(status)
                if not in_room:
                    await self.send_to_approver(self.result_notice(req, title, "", ok=False, same_limit=True),
                                                kind="notice")
                await session.notify_failure(same_limit, note, inbound_id=req.inbound_id)
            else:
                if not in_room:
                    await self.send_to_approver(self.result_notice(req, title, message, ok=False), kind="notice")
                await session.notify_failure(message, note, inbound_id=req.inbound_id)
            if meanwhile:
                # Someone was told "I will pick that up when this finishes". It has
                # finished, so pick it up.
                logger.info("[gate] #%s failed with %d message(s) received meanwhile; handling the latest",
                            req.id, len(meanwhile))
                try:
                    await session.handle_followup(meanwhile[-1]["text"], inbound_id=meanwhile[-1]["id"])
                except Exception:
                    logger.exception("[gate] follow-up after #%s failed", req.id)
            session.typing_stop()
            return ""
        try:
            if not in_room:
                await self.send_to_approver(self.result_notice(req, title, message, ok=ok), kind="notice")
            await session.notify_after_request(note, inbound_id=req.inbound_id)
        finally:
            session.typing_stop()
        return ""

    def _approver_chat_ids(self) -> set:
        """The owner's record. One thread, whatever surface a message came in on, so it no
        longer depends on which session happens to be loaded."""
        out = {OWNER_THREAD}
        out.update(cid for cid, sess in self.sessions.items() if sess.is_approver())
        return out

    # -- expiry ----------------------------------------------------------------
    async def _expire_pending(self) -> None:
        # New-model proposals are schedule rows in state "proposed", not requests, so
        # they are never in store.expire_older_than's result; they expire here too.
        await self.schedules.expire_stale_proposals(EXPIRE_SECONDS)
        for req in self.store.expire_older_than(EXPIRE_SECONDS):
            self.store.set_thread(req.chat_id, "idle")
            if self.schedules.is_proposal_request(req):
                self.schedules.proposal_expired(req)
                await self.send_to_approver(
                    self._expired_text(req), kind="notice")
            elif req.schedule_id and (req.status or {}).get("confirm"):
                await self.schedules.held_ended(req, "expired")
            else:
                await self.send_to_approver(
                    self._expired_text(req), kind="notice")
            self.store.add_message(req.chat_id, "system", f"Task #{req.id} expired unanswered: {req.summary}")
            self.task_note(req, "expired", f"Expired after 24h with no answer from Aaron: {req.summary}", state="open")

    @staticmethod
    def _expired_text(req: Request) -> str:
        group, tagged_from, _ = ownertext.split_group_tag(req.original_message or req.summary or "")
        who = ownertext.person(req.sender_name or tagged_from, req.sender)
        return ownertext.compose(
            "timer", f"Expired with no answer (#{req.id})",
            facts=[("From", who), ("Where", ownertext.where(req.mode, group)),
                   ("Asked", ownertext.strip_group_tag(req.summary))],
            tail="Nothing was done. It waited 24 hours for your yes.")

    async def _expiry_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            try:
                await self._expire_pending()
            except Exception:
                logger.exception("expiry loop error")
            try:
                # Catches up a notice whose timer a restart cut off.
                await self.flush_people_notices()
            except Exception:
                logger.exception("people notice flush error")

    async def _schedule_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                await self.schedules.tick()
            except Exception:
                logger.exception("schedule loop error")

    def queue_timer(self, chat_id: str, source_event_id: str, meta: Dict[str, Any]) -> bool:
        """Put a ready timer behind this chat's earlier events when the inbox is on."""
        try:
            if not _settings.get("GATE_INBOX"):
                return False
            gateway = getattr(self, "gateway", None)
            if gateway is None:
                raise RuntimeError("inbox is unavailable")
            ensure = getattr(gateway, "_ensure_inbox", None)
            inbox = ensure() if callable(ensure) else getattr(gateway, "inbox", None)
            if inbox is None:
                raise RuntimeError("inbox is unavailable")
            inbox.submit(chat_id, "timer", source_event_id, "timer", "", meta, kind="timer")
            return True
        except Exception as exc:
            logger.warning("[gate] timer inbox unavailable; running directly: %s", exc)
            return False

    async def handle_timer_event(self, event: Dict[str, Any]) -> None:
        meta = json.loads(event["meta_json"])
        if meta.get("schedule_id"):
            req = self.store.get_request(int(meta["request_id"]))
            if req is None:
                raise ValueError("scheduled request is missing")
            await self.schedules._run_fired(req, raise_errors=True)
        elif meta.get("followup_id"):
            row = self.store.get_followup(int(meta["followup_id"]))
            if row is None:
                raise ValueError("follow-up is missing")
            await self._run_followup(row, raise_errors=True)
        else:
            raise ValueError("timer type is missing")

    async def _followup_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                if _settings.get("GATE_FOLLOWUPS") and self.store.followups_available:
                    for row in self.store.due_followups():
                        # Claim before starting any work. A restart cannot repeat it.
                        if self.store.mark_followup_fired(row["id"]):
                            if not self.queue_timer(row["chat_id"], f"followup:{row['id']}",
                                                    {"followup_id": row["id"]}):
                                asyncio.create_task(self._run_followup(row))
            except Exception as exc:
                logger.warning("[gate] follow-up loop failed: %s", exc)

    async def _run_followup(self, row: Dict[str, Any], *, raise_errors: bool = False) -> None:
        from .hosttools import followup_time
        request = None
        run_failed = False
        try:
            original = self.store.get_request(row["request_id"])
            if original is None:
                raise ValueError("original request is missing")
            scopes = json.loads(row["scopes_json"])
            # The saved approval is the ceiling even if the task gained scopes later.
            scopes = [scope for scope in scopes if scope in original.scopes]
            if not scopes:
                raise ValueError("original scopes are missing")
            prompt = (f"Follow-up you set on {followup_time(row['created_at'])}: {row['note']}\n\n"
                      "Check the task ledger. Report CHANGED: yes if there is something worth telling "
                      "the chat, or CHANGED: no if nothing changed. Explain the result in plain words.")
            request = self.store.create_request(
                chat_id=row["chat_id"], sender=row["requester"], sender_name=original.sender_name,
                mode=original.mode, subject=original.subject, original_message=prompt,
                summary=f"Follow-up: {row['note']}", scopes=scopes, prompt=prompt,
                state="approved", task_id=row["task_id"])
            self.store.set_state(request.id, "approved", status={"followup_id": row["id"]})
            request = self.store.get_request(request.id)
            self.store.task_event(row["task_id"], "request", f"Follow-up fired: {row['note']}",
                                  chat_id=row["chat_id"], request_id=request.id, state="running")
            await self.execute(request, notify=False)
            finished = self.store.get_request(request.id)
            status = (finished.status or {}) if finished else {}
            run_failed = status.get("ok") is False
            if finished and finished.state == "pending":
                self.store.finish_followup(row["id"], "Waiting for confirmation.")
                return
            result = str(status.get("result_line") or status.get("final_status") or
                         status.get("summary") or status.get("error") or "").strip()
            self.store.finish_followup(row["id"], result, failed=not bool(status.get("ok")))
            if outcomes.enabled() and status.get("ok") and (
                    status.get("answered_requester") or status.get("outcome") == "quiet"):
                # It answered the chat itself, or chose to stay quiet: nothing more goes out.
                if status.get("outcome") == "quiet" and not status.get("sends"):
                    self.store.task_event(row["task_id"], "note", "Follow-up found nothing to send.",
                                          chat_id=row["chat_id"], request_id=request.id, state="open")
                return
            if status.get("ok") and (status.get("changed") is False or re.search(
                    r"\b(nothing changed|no change|nothing to report|no update)\b", result, re.I)):
                self.store.task_event(row["task_id"], "note", "Follow-up found nothing changed.",
                                      chat_id=row["chat_id"], request_id=request.id, state="open")
                return
            if not result and status.get("ok"):
                return
            message = result if status.get("ok") else wording.failure_message(
                wording.plain_title(self.store, request), status)
            route = self.store.thread_route(row["chat_id"])
            mode = (route or {}).get("mode") or original.mode
            if mode == "voice" and not self._is_owner_request(original) and len(_digits(row["requester"])) >= 7:
                number = row["requester"]
                await self.send(f"sms:{number}", message, "sms", {"to": number, "sender": number},
                                chat_id=row["chat_id"])
            elif mode == "voice" or not route:
                await self.send_to_approver(message)
            else:
                await self.send(row["chat_id"], message, mode, route["meta"], chat_id=row["chat_id"])
        except Exception as exc:
            logger.warning("[gate] follow-up #%s failed: %s", row["id"], exc)
            try:
                self.store.finish_followup(row["id"], str(exc), failed=True)
                self.store.task_event(row["task_id"], "failed", f"Follow-up failed: {exc}",
                                      chat_id=row["chat_id"], request_id=request.id if request else None,
                                      state="open")
            except Exception as record_exc:
                logger.warning("[gate] follow-up #%s failure recording failed: %s", row["id"], record_exc)
            if raise_errors:
                raise
        if raise_errors and run_failed:
            raise RuntimeError(str(status.get("error") or "follow-up run failed"))
