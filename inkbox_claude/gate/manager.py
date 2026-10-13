"""Gate session manager: drop-in for SessionManager when INKBOX_MODE=gate.

Interface the gateway uses:
  get(chat_id, system_prompt_extra="") -> session with
      handle_inbound(text, mode, meta), run_consult(...), run_consult_detailed(...), close()
  close_all()
"""

from __future__ import annotations

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
from .scopes import SCOPES, READ_SCOPES, EXPLICIT_ONLY_SCOPES
from .store import Person, Request, Store, TaskRequired, task_key
from .schedules import ScheduleService
from . import schedprop
from . import wording

logger = logging.getLogger(__name__)

SendFn = Callable[[str, str, str, Dict[str, Any]], Awaitable[Any]]

HANDOFF_LINE = (
    "Thanks! I'll confirm that with Aaron and get back to you shortly.\n\n"
    "Blatbot\nExecutive Assistant to Aaron Blatnoy"
)
EXPIRE_SECONDS = 24 * 3600
# Item C2: how long a call's failed ask is still "the same ask" for folding purposes
# on timing alone -- long enough to cover the tail end of one spoken turn, short
# enough that a later fragment on the same task but a genuinely different ask (the
# other test of _is_same_ask_as_failed, caller-words continuation) is the one doing
# the real work past this window.
CALL_ASK_FOLD_WINDOW_S = 45


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


def _clean_result(text: str) -> str:
    """Drop the STATUS line and trailing whitespace from an executor result."""
    lines = [ln for ln in text.strip().splitlines() if not ln.strip().upper().startswith("STATUS:")]
    return "\n".join(lines).strip()


DEFAULT_VOICE_VOCABULARY = (
    "Blatbot (BLAT-bot, this assistant). Add the names, places and terms your callers use, with pronunciation "
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

    def _sender_name(self) -> str:
        c = (self.reply_meta or {}).get("contact") or {}
        if isinstance(c, dict):
            name = str(c.get("name") or "").strip()
            if not name:
                name = " ".join(str(c.get(k) or "") for k in ("given_name", "family_name")).strip()
            return name
        return ""

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
        p = Person.from_contact(c if isinstance(c, dict) else None, self._sender(), self._sender_name())
        self.m.store.remember_person(p)
        return p

    def handles(self) -> List[str]:
        """Every way this sender reaches the assistant on this turn. Trust is held against
        handles rather than a person record, because a handle is what a message carries."""
        meta = self.reply_meta or {}
        c = meta.get("contact") if isinstance(meta.get("contact"), dict) else {}
        out = [self._sender(), str(meta.get("telegram_user_id") or ""), str((c or {}).get("id") or ""),
               str((c or {}).get("name") or ""), self._sender_name()]
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
                        out: RouterOutput) -> None:
        """Second judgment: what this message did to the task it was assigned to.
        Writes a typed event line, updates the summary and, when confident, the state."""
        picker = self.m.task_picker
        if picker is None:
            return
        approver = self.is_approver()
        res = await picker.judge_event(
            message=body, task=task, history=history,
            sender_label="Aaron, the owner" if approver else (self._sender_name() or self._sender()))
        kind = res.get("kind")
        if not kind or kind == "conversation_only":
            return
        who = "Aaron" if approver else (self._sender_name() or self._sender())
        label = kind.replace("_", " ")
        line = f"{who} {label}: {body.strip()[:300]}"
        new_state: Optional[str] = None
        if res.get("resolves_task", 0) >= 0.8 and out.request is None:
            new_state = "done"
        elif res.get("waiting_on") == "owner" and out.request is None:
            new_state = "waiting_aaron"
        elif task.get("state") in ("done", "failed") and kind in ("asked_for_something", "changed_or_corrected"):
            new_state = "open"
        self.m.store.task_event(task["id"], kind, line, chat_id=self.record_id, state=new_state)
        if not out.task_summary:
            # The router gave no summary this turn: keep the ledger honest with a short one.
            wo = res.get("waiting_on")
            tail = {"owner": "Waiting on Aaron.", "other": f"Waiting on {who}.", "nobody": "Nothing outstanding."}.get(wo or "", "")
            self.m.store.set_task_summary(task["id"], f"Latest: {who} {label}. {tail}".strip())

    async def lookup_tasks(self, history: List[Dict[str, Any]], message: str, in_view: str) -> str:
        """Query pass: the router emits a structured filter, code runs it. Non-owners
        are confined to tasks they are on; the owner can query everything."""
        q: Any = None
        planned = False
        picker = self.m.task_picker
        if picker is not None and hasattr(picker, "plan_lookup"):
            # Jev decides whether the ledger is searched and for which phrase (taken from
            # words in the message). The chat model plans only when Jev is undecided.
            from .jevagent import _span_candidates
            from .router import TaskQuery
            emails = _EMAIL.findall(message)
            spans = list(emails)
            for c in _span_candidates(message, [], limit=40):
                for v in (re.sub(r"['\u2019]s\b", "", c), c):     # "Sam Rivera's" also offers "Sam Rivera"
                    if v and v not in spans:
                        spans.append(v)
            titles = [" ".join(ln[len("Task "):].split(" | ")[:2]) for ln in in_view.splitlines() if ln.startswith("Task T")]
            label = "Aaron, the owner" if self.is_approver() else (self._sender_name() or self._sender())
            plan = await picker.plan_lookup(message=message, history=history, sender_label=label,
                                            in_view_titles=titles, candidates=spans)
            if plan.get("lookup") == "none":
                return ""
            if plan.get("lookup") in ("live", "any"):
                planned = True
                text = plan.get("text") or ""
                participant = text if text in emails and self.is_approver() else None
                q = TaskQuery(text=None if participant else (text or None), participant=participant,
                              states=["live"] if plan["lookup"] == "live" else None, limit=10)
        if not planned:
            try:
                q = await self.m.router.plan_query(history=history, message=message, in_view=in_view,
                                                   is_approver=self.is_approver())
            except Exception:
                logger.exception("[gate %s] task query planning failed", self.chat_id)
                return ""
        self._found_task_ids = []
        if q is None and in_view.strip() in ("", "(no tasks on record)") and not self.is_approver():
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
                                      date_from=tq.date_from, date_to=tq.date_to, limit=tq.limit)
            if self.is_approver():
                kw["participant"] = tq.participant or ""
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
        text = self.m.store.render_tasks(shown)
        if res["total"] > len(res["tasks"]):
            text += f"\n\n({res['total'] - len(res['tasks'])} more matched; narrow the lookup to see them)"
        return text

    def candidate_tasks(self, message: str = "") -> List[Dict[str, Any]]:
        """Tasks the message could belong to, as a retrieval union, never a recency slice:
        this chat's recent tasks, every live task, tasks of people the message names,
        full-text hits on the message's own words, and the lookup's finds. Scored by how
        many signals agree plus recency, capped at 25 cards for the pick."""
        approver = self.is_approver()
        score: Dict[int, float] = {}
        order: List[int] = []

        def add(tid: int, pts: float) -> None:
            if tid not in score:
                order.append(tid)
            score[tid] = score.get(tid, 0.0) + pts

        if not approver:
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
                if not approver:
                    kw["participant"] = self._sender()
                for t in self.m.store.query_tasks(**kw)["tasks"]:
                    add(int(t["id"]), 2.0)
            except Exception:
                logger.debug("[gate %s] candidate text search failed", self.chat_id, exc_info=True)
        for tid in self._found_task_ids:
            add(tid, 2.0)
        ranked = sorted(order, key=lambda tid: -score[tid])
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
        who = "Aaron Blatnoy (the owner)" if self.is_approver() else (self._sender_name() or self._sender())
        addr = self._sender()
        # Whole messages, not clipped ones: a half-read line is worse than a missing one,
        # and the window matches what the judgments upstream were given.
        convo = "\n".join(f"[{m.get('kind')}] {str(m.get('text') or '')}" for m in history[-20:]) or "(none)"
        ledger = self.m.store.task_memory_for_task(int(task["id"])) or "(new task, no events yet)"
        subject = str((self.reply_meta or {}).get("subject") or "").strip()
        return (
            (f"{self.room}\n\n" if self.room else "")
            + f"A message arrived over {self.mode} from {who} ({addr}).\n"
            + (f"Subject: {subject}\n" if subject else "")
            + f"\n--- THEIR MESSAGE (verbatim) ---\n{body.strip()}\n"
            + _what_it_answers(body, history)
            + "\n"
            f"--- RECENT CONVERSATION WITH THEM ---\n{convo}\n\n"
            f"--- THE TASK THIS BELONGS TO ---\n{ledger}\n\n"
            "Work out what they are asking for, or what this message makes possible on the task (for example, "
            "information the task was waiting for), and do it with your tools. Do only what the message and the "
            "task call for; do not invent extra steps. If something essential is missing, stop and say exactly "
            "what is missing instead of guessing.\n\n"
            "The task record is background, not an answer. An earlier run on this task answered an earlier "
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
                     memory: str, found: str) -> "Tuple[RouterOutput, Optional[Dict[str, Any]]]":
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
                                                memory=memory, found=found)
        label = "Aaron, the owner" if approver else (self._sender_name() or self._sender())
        pick = await picker.pick(message=message, history=prior, candidates=self.candidate_tasks(),
                                 sender_label=label, router_hint=None, proposed_title="")
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
                    summary=str(task.get("title") or wording.fallback_title_from_message(message)),
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

    async def jev_scopes(self, out: RouterOutput, strict: bool = False,
                         owner_generous: bool = True) -> None:
        """Replace the router's scope list with Jev's judgment when it yields a
        non-empty set. Empty or unavailable: the router's list stands. Code never
        lets this widen beyond the fixed scope map, since pydantic already
        validated names and the executor only ever grants mapped tools."""
        picker = self.m.task_picker
        if picker is None or out.request is None:
            return
        res = {"scopes": None}
        if strict and hasattr(picker, "judge_scopes_tree"):
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
            return
        if set(chosen) != set(out.request.scopes):
            logger.info("[gate %s] scopes: router %s -> jev %s", self.chat_id, out.request.scopes, chosen)
        out.request.scopes = chosen

    async def jev_knowledge(self, out: RouterOutput) -> None:
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
        res = await picker.judge_vault_tree(prompt=out.request.prompt, summary=out.request.summary)
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

    def task_memory(self) -> str:
        mem = self.m.store.task_memory(self.record_id, is_approver=self.is_approver(),
                                       person="" if self.is_approver() else self._sender())
        extra = []
        for tid in self.m.store.task_ids_for_thread(self.thread_key()):
            if f"Task T{tid} " not in mem:
                t = self.m.store.task_with_events(tid)
                if t and t["state"] != "closed":
                    extra.append(t)
        if extra:
            block = self.m.store.render_tasks(extra)
            mem = block if mem == "(no tasks on record)" else mem + "\n\n" + block
        return mem

    # -- gateway interface ---------------------------------------------------
    async def handle_inbound(self, text: str, mode: str, meta: Dict[str, Any]) -> None:
        async with self._lock:
            self.mode = mode
            self.reply_meta = dict(meta or {})
            self.m.store.set_thread(self.chat_id, "routing", mode, self.reply_meta_for_store())
            if self.is_approver():
                # Remembered on disk: notices have to land here even when no session is loaded.
                self.m.store.mark_owner_thread(self.chat_id, mode)
            body = _strip_quoted(text) if mode == "email" else text.strip()
            self.inbound_id = self.m.store.add_message(self.record_id, "inbound", body, mode)
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
            finally:
                self.typing_stop()
                if self.m.store.thread_state(self.chat_id) == "routing":
                    self.m.store.set_thread(self.chat_id, "idle")

    def reply_meta_for_store(self) -> Dict[str, Any]:
        keep = {k: v for k, v in (self.reply_meta or {}).items()
                if k in ("to", "sender", "subject", "thread_id", "conversation_id", "conversation_kind",
                         "message_id", "in_reply_to", "references")}
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
                    notes = "What's on record with this caller:\n" + self.task_memory()[-1500:]
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
                memory = self.task_memory()
                logger.info("[gate %s] ledger loaded (voice consult): %d chars", self.chat_id, len(memory))
                found = await self.lookup_tasks(history[:-1], body, memory)
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
                # where create_request_once's own dedupe (same chat_id AND sender,
                # approved/running, within the window) would also fire -- that check
                # is thread-wide (any sender), this one is sender-scoped, so whenever
                # it would hit, running_for_thread already folded this fragment through
                # the branch above and returned. The only way create_request_once's
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
                    # Another fragment of this same ask, from this same caller, already
                    # has a request approved or running on this thread within the dedupe
                    # window: never execute a second run of it, and never treat this
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
        can rewrite it into something the grounding check then rejects. Only a real
        success result -- the run's own narration -- goes through the reply writer and
        grounding check. `ok` comes from the request's own status, never from sniffing
        the text for a leading "Done"."""
        if not ok:
            return text
        note = f"Task #{req.id} done: {wording.plain_title(self.m.store, req)}\nResult:\n{text}"
        async with self._lock:
            prev = (self.mode, self.reply_meta)
            self.mode, self.reply_meta = "voice", dict(call_meta or {})
            try:
                approver = self.is_approver()
                history = self.m.store.history(self.record_id, limit=20)
                memory = self.task_memory()
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
                reply = await self.ground_reply(out.reply, note, history, memory, "", approver)
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
    async def _route_and_act(self, body: str, system_note: Optional[str] = None) -> bool:
        approver = self.is_approver()
        history = self.m.store.thread_history(self.thread_key(), exclude_chat=self.record_id) + \
            self.m.store.history(self.record_id, limit=20)
        if system_note:
            prior, message = history, "(No new message from the sender. A system note with a task result was just added above. Write the reply the sender should receive: for Aaron, give him the answer or outcome clearly and naturally, in his voice preference (brief, no emojis); for anyone else, tell them only what concerns them, without internal details. Reply null only if there is truly nothing to say. Do not create a request.)"
        else:
            prior, message = history[:-1], body  # the new inbound is already stored last
        memory = self.task_memory()
        logger.info("[gate %s] ledger loaded: %d task(s), %d chars", self.chat_id,
                    memory.count("\nTask T") + (1 if memory.startswith("Task T") else 0), len(memory))
        found = "" if system_note else await self.lookup_tasks(prior, message, memory)
        task: Optional[Dict[str, Any]] = None
        running = None if system_note else self.m.store.running_for_thread(self.chat_id)
        if running is not None:
            # One request at a time per thread. The new message is already on the
            # thread and the task; when the running request ends, execute() re-reads
            # what arrived meanwhile (and re-decides it if the run failed). Creating a
            # second request here is what made the assistant answer twice.
            logger.info("[gate %s] request #%s still running; not starting another for this message",
                        self.chat_id, running.id)
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
            out: RouterOutput = await self.m.router.route(
                history=prior, message=message, mode=self.mode, sender=self._sender(),
                contact_notes=self._contact_notes(), is_approver=approver,
                task_memory=memory, found_tasks=found, action=False,
            )
            out.request = None
            out.reply = await self.ground_reply(out.reply, system_note, prior, memory, found, approver)
        else:
            out, task = await self.decide(body=body, message=message, prior=prior, mode=self.mode,
                                          memory=memory, found=found)
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
            try:
                scopes = schedprop.capabilities_for(
                    out.schedule.prompt, [], out.schedule.scopes, message)
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
            try:
                await self.m.schedules.propose(
                    owner=True, chat_id=self.record_id, task_id=int(task["id"]),
                    title=out.schedule.title, prompt=out.schedule.prompt,
                    cadence_kind=out.schedule.cadence_kind, cadence=out.schedule.cadence,
                    timezone=out.schedule.timezone, report_every_time=out.schedule.report_every_time,
                    until=until_epoch, scopes=scopes, start_now=out.schedule.start_now,
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
            return await self.send_to_sender(reply, role="answer")
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
            summary=out.request.summary, scopes=out.request.scopes, prompt=out.request.prompt,
            state="approved" if (approver or trusted) else "pending", task_id=task["id"],
            inbound_id=self.inbound_id,
        )
        if not created:
            # Another message from this same sender on this thread already has a
            # request approved or running within the dedupe window: the ack for a
            # second one is never sent (there is no second run to acknowledge), it
            # is never executed again, and its scopes are never widened -- all three
            # belong to whichever turn actually created the row this one found.
            logger.info("[gate %s] request deduped onto #%s for sender %s; not acknowledging or running again",
                        self.chat_id, req.id, self._sender())
            return False
        # Alongside a request the reply is an acknowledgement, sent only now that
        # this turn is known to own a genuinely new request.
        sent_reply = await self.send_to_sender(out.reply, role="ack") if out.reply else False
        self.m.remember_knowledge_notes(req.id, getattr(out.request, "knowledge_notes", None))
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
            return wording.failure_message(fresh.summary, fresh_status)
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
                await self.m.send_to_approver(text)
            else:
                num = self._sender()
                await self.m.send(f"sms:{num}", text, "sms", {"to": num, "sender": num}, record=False)
                self.m.store.add_message(self.record_id, "outbound", text, "sms", reply_to=self.inbound_id, role=role)
        else:
            self.typing_stop()
            await self.m.send(self.chat_id, text, self.mode, self.reply_meta, record=False)
            self.m.store.add_message(self.record_id, "outbound", text, self.mode, reply_to=self.inbound_id, role=role)
        t = self.task()
        if t is not None:
            self.m.store.task_event(t["id"], "outbound", text, chat_id=self.record_id)
        return True

    async def ground_reply(self, reply: Optional[str], note: str, prior: List[Dict[str, Any]], memory: str,
                           found: str, approver: bool) -> Optional[str]:
        """A result reply may only state what the results support. Jev checks the
        draft against the tool output; an unsupported draft is rewritten once with
        the results as the only source, and if still unsupported, replaced by a
        plain statement of what was and was not found. Never a confident guess."""
        picker = self.m.task_picker
        if not reply or picker is None or "Result:" not in note:
            return reply
        results = note.split("Result:", 1)[1].strip()
        question = ""
        for m in reversed(prior):
            if m.get("kind") == "inbound":
                question = str(m.get("text") or "")
                break
        res = await picker.judge_grounded(reply=reply, results=results, question=question)
        if res.get("grounded") is not False:
            return reply
        logger.info("[gate %s] reply not grounded (p=%.2f); rewriting from the results only", self.chat_id, res["p"])
        strict = ("(GROUNDING: your previous draft stated something the task result does not support. Fill "
                  "\"reasoning\" first: what the result establishes, what follows from it by ordinary inference, "
                  "what stays unknown. Then write the reply as a conclusion from that: answer the question directly, "
                  "name only what the result contains or what plainly follows from it, mark inferences as inferences, "
                  "and say what stays unknown. Do not create a request.)")
        out2 = await self.m.router.route(
            history=prior + [{"kind": "system", "text": strict}], message=strict, mode=self.mode, sender=self._sender(),
            contact_notes=self._contact_notes(), is_approver=approver, task_memory=memory, found_tasks=found, action=False)
        reply2 = out2.reply or reply
        res2 = await picker.judge_grounded(reply=reply2, results=results, question=question)
        p2 = float(res2.get("p") or 0.0)
        if res2.get("grounded") is not False or p2 >= 0.35:
            # A rewrite from the results only, near or over the line: send it. Derived
            # answers (free slots from a list of events, a count from rows) score low
            # on literal support even when they are right.
            return reply2
        # Twice unsupported. Sending it anyway is the one thing worse than saying nothing:
        # it is a confident answer the check already judged to be unbacked.
        logger.warning("[gate %s] rewrite still not grounded (p=%.2f); refusing to pass it off",
                       self.chat_id, p2)
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
            return await self.send_to_sender(message, role="answer")

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


def _owner_request(manager: Any, req: Request) -> bool:
    """Whether a request came from the person the assistant works for. Read through the
    manager when it can answer, so a stand-in without the machinery simply says no."""
    fn = getattr(manager, "_is_owner_request", None)
    try:
        return bool(fn(req)) if callable(fn) else False
    except Exception:
        return False


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
    rest = {k: v for k, v in args.items()
            if k not in {"to", "recipient", "recipients", "to_email", "email", "phone", "number", "cc", "bcc",
                         "subject", "title", "body", "message", "text", "content", "caption", "user_google_email"}}
    out = "\n".join(lines)
    if body:
        out += ("\n\n" if out else "") + body
    if rest:
        out += ("\n\n" if out else "") + json.dumps(rest, ensure_ascii=False)
    return out or json.dumps(args, ensure_ascii=False)


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
        self.executor = Executor(mcp_server=mcp_server, cwd=exec_cwd, model=cfg.claude_model or "sonnet",
                                 protected=protected)
        # GATE_EXECUTOR=jev runs requests through the typed-judgment agent
        # (Jev picks tools, code fills arguments, DeepSeek writes prose only when
        # needed); Claude Code stays as the fallback unless GATE_EXECUTOR_FALLBACK=none.
        self.jev_agent: Optional[JevAgent] = JevAgent(inkbox_server=mcp_server, router=self.router,
                                                      protected=protected) if jev_agent_enabled() else None
        # One confirmed call has to be runnable whoever the executor is: when the owner has
        # read a draft and said yes, the same call goes out, and nothing else does.
        self.caller: JevAgent = self.jev_agent or JevAgent(inkbox_server=mcp_server, router=self.router,
                                                           protected=protected)
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
        # Request ids a blanket "you have my permission for everything" (item 4) was
        # said about while still running: applied to that request's own retry, once,
        # the moment it actually fails and shows what it was refused. Held in memory
        # only -- a restart loses an in-flight blanket grant, same as other in-flight
        # judgments here (_knowledge_notes, _voice_trust_override).
        self._blanket_pending: set = set()
        self.schedules = ScheduleService(self)
        self._expiry_task: Optional[asyncio.Task] = None
        self._schedule_task: Optional[asyncio.Task] = None
        try:
            self._expiry_task = asyncio.get_running_loop().create_task(self._expiry_loop())
            self._schedule_task = asyncio.get_running_loop().create_task(self._schedule_loop())
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
                   record: bool = True) -> None:
        """Every outbound message goes through here, and every one of them is written to a
        thread. What was said is what can be referred to later, so a message that was sent
        but never recorded leaves the next turn reading a conversation that did not happen."""
        await self.send_fn(target, text, mode, meta)
        if not record:
            return      # the caller writes its own line, with the turn's reply_to and role
        for cid in self._threads_for(target, chat_id):
            try:
                self.store.add_message(cid, "outbound", text, mode, reply_to=reply_to, role=role)
            except Exception:
                logger.debug("[gate] could not record an outbound on %s", cid, exc_info=True)

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
            stranded = self.store.requests_in_state("running") + [
                r for r in self.store.requests_in_state("approved") if r.schedule_id
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
                    await self.send_to_approver(f"[Blatbot #{req.id}] {line}")
                logger.info("[gate] told %s about interrupted request #%s", req.chat_id, req.id)
            except Exception:
                logger.exception("[gate] could not report interrupted request #%s", req.id)

    async def send_to_approver(self, text: str) -> None:
        conv = self.approver_conv
        await self.send(f"imessage:{conv}", text, "imessage", {"conversation_id": conv})

    async def send_schedule_notice(self, schedule: Any, text: str) -> int:
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
                await self.send_fn(target, text, mode, meta)
                sent = True
            except Exception:
                logger.exception("[gate] schedule %s notice failed on %s/%s; falling back to the owner's thread",
                                 getattr(schedule, "id", "?"), mode, target)
        if not sent:
            conv = self.approver_conv
            mode = "imessage"
            target = f"imessage:{conv}"
            await self.send_fn(target, text, mode, {"conversation_id": conv})
        return self.store.add_message(OWNER_THREAD, "outbound", text, mode, role="notice",
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
            await self.send_to_approver(f"Heads up: {who} {what}. I cannot finish turns until that is fixed. "
                                        f"Your last message was not answered.")
        except Exception:
            logger.exception("[gate] could not deliver the outage notice")

    async def notify_inbound(self, session: GateSession, body: str) -> None:
        who = session._sender_name()
        sender = session._sender()
        who = f"{who} ({sender})" if who else sender
        subj = str((session.reply_meta or {}).get("subject") or "").strip()
        snippet = body.strip()  # the whole message; Aaron asked for no truncation
        head = f"[Blatbot FYI] {who} via {session.mode}"
        if subj:
            head += f" | {subj}"
        try:
            await self.send_to_approver(f"{head}\n{snippet}")
        except Exception:
            logger.exception("[gate] FYI to approver failed")

    def format_request(self, req: Request, revised: bool = False) -> str:
        who = f"{req.sender_name} ({req.sender})" if req.sender_name else req.sender
        subj = f"\nSubject: {req.subject}" if req.subject else ""
        head = f"[Blatbot #{req.id}{' revised' if revised else ''}] {who} via {req.mode}{subj}"
        marker = "--- AARON'S INSTRUCTIONS ---"
        yours = ""
        if marker in req.prompt:
            yours = "Your instructions: " + req.prompt.split(marker, 1)[1].strip() + "\n\n"
        return (
            f"{head}\n\n--- Their message ---\n{req.original_message}\n\n"
            f"--- Blatbot wants ---\n{req.summary}\nTools: {', '.join(req.scopes)}\n\n"
            "Claude will read their message and the task record and act on it with only those tools.\n"
            + yours +
            f"Reply \"#{req.id} yes\" to run, \"#{req.id} no\" to drop, "
            f"or \"#{req.id} edit: <instructions>\" to add instructions (you will see it again before it runs). "
            "A bare yes/no works when this is the only one pending."
        )

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
        summary = (req.summary if req else "") or fallback or "Untitled task"
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
            title = (out.task_title or "").strip() or summary
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
        await self.send_to_approver(self.format_request(req, revised))

    async def handle_task_command(self, session: GateSession, text: str) -> bool:
        """Aaron's task commands: "tasks" lists open work; "T12 done" / "T12 close" /
        "T12 reopen" change state; "T12 note: ..." adds a note. Returns True if consumed."""
        t = (text or "").strip()
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
        already shown to him and approved before it ran, and its answer goes back to them."""
        return req.chat_id == OWNER_THREAD or req.chat_id in self._approver_chat_ids()

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
        if c.get("kind") == "send":
            # Not a summary of a message: the message. He approves the words that will go out.
            plain = f"[Blatbot #{req.id}] Nothing has been sent. This is what would go out:\n\n{_draft_text(shown)}"
        else:
            plain = await self._phrase_confirmation(req, short, shown, c.get("about") or [])
        full_text = f"{plain}\nReply \"#{req.id} yes\" to do it or \"#{req.id} no\" to leave it alone."
        schedule = self.store.get_schedule(req.schedule_id) if req.schedule_id else None
        if schedule is not None:
            # A held call that belongs to a scheduled run asks on that
            # schedule's own channel, not always the owner's iMessage -- the
            # "#<id> yes" reply mechanics themselves are unchanged.
            await self.send_schedule_notice(schedule, full_text)
        else:
            await self.send_to_approver(full_text)

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
                return f"[Blatbot #{req.id}] {text}"
        except Exception:
            logger.debug("[gate] confirmation phrasing failed", exc_info=True)
        return (f"[Blatbot #{req.id}] Before I do this, I need your yes.\n"
                f"Action: {short} {json.dumps(shown, ensure_ascii=False)}\nIt refers to:\n"
                + ("\n".join(f"  {ln}" for ln in about) or "  (no further detail found)"))

    async def perform_confirmed(self, req: Request) -> None:
        """The owner said yes: run exactly the stored call, then finish the request."""
        c = (req.status or {}).get("confirm") or {}
        self.store.set_state(req.id, "running")
        if not c:
            status = {"ok": False, "error": "no confirmed call on record", "summary": "no confirmed call on record", "raw": "", "tool_calls": []}
        else:
            # He has read it and said yes. That exact call runs, once, and nothing else.
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
        note = f"Task #{req.id} {'done' if ok else 'FAILED'}: {req.summary}\nResult:\n{result}"
        replied = await session.notify_after_request(note, inbound_id=req.inbound_id)
        if not replied:
            await self.send_to_approver(f"{'Done' if ok else 'Failed'}: {req.summary}\n{result}")

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
        if req.state not in ("pending",):
            # Someone already answered this one. Say so rather than acting again.
            return {"ok": False, "error": f"#{req.id} is already {req.state}", "state": req.state}
        if self.schedules.is_proposal_request(req):
            return await self.schedules.decide_proposal(req, decision, note)
        d = (decision or "").strip().lower()
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
        t = text.strip()
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
        if req is not None:
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
        new_req = self.retry_request(req, approved=approved)
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
        grown = self.retry_request(req, approved=True)
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

        vault_read (the WHOLE vault of Aaron's own notes) is excluded from the
        auto-add: it is never granted on its own just for being read-only, only
        when the run was actually refused it -- the vault can hold anything he has
        ever written down, so a blanket "permission for everything" must not open
        all of it as a side effect of a run that never asked for it."""
        denied = set(wording.denied_scopes(status))
        granted = [sc for sc in SCOPES if sc in denied or
                  (wording.truly_read_only(sc) and sc != "vault_read"
                   and sc not in EXPLICIT_ONLY_SCOPES)]
        return [sc for sc in granted if sc not in (req.scopes or [])]

    async def _grant_blanket_and_retry(self, session: "GateSession", req: Request) -> None:
        new_scopes = self._blanket_grant_scopes(req, req.status or {})
        grown = self.retry_request(req, approved=True)
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
        if status.get("confirm") or status.get("ok"):
            return status
        said = f"{status.get('summary') or ''}\n{status.get('error') or ''}\n{status.get('raw') or ''}"
        wanted = [sc for sc in SCOPES if sc not in req.scopes and re.search(
            r"\b" + re.escape(sc) + r"\b", said)]
        if not wanted or not _owner_request(self, req) or req.schedule_id:
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

    async def _insist_it_actually_ran(self, req: Request, status: Dict[str, Any], context: str,
                                      model: Optional[str]) -> Dict[str, Any]:
        """A request exists because a tool was needed. Reporting success without having
        called one means the work was not done, whatever the words say.

        The usual way this happens: the task record holds the answer an earlier run gave,
        and that reads as the question already being settled. The record is evidence about
        the past, not a substitute for looking now. One retry says so in as many words; a
        second refusal to act is reported as a failure rather than dressed up as a result."""
        if status.get("confirm") or not status.get("ok") or status.get("tool_calls"):
            return status
        logger.warning("[gate] #%s reported success with no tool call; insisting it runs", req.id)
        print("DEBUG insist: status=", status)
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

    async def _run_executor(self, req: Request, context: str) -> Dict[str, Any]:
        """Run one request and return exactly ONE status object. Claude Code is an
        escalation inside this call, not a second execution: nothing is delivered,
        noted, or phrased for the Jev attempt when it escalates; only the final
        status leaves this function. Executors themselves never message anyone who
        asked (denied at the tool call).

        A scheduled run always takes the Claude Code branch, GATE_EXECUTOR=jev
        or not: the Jev agent has no time zone, no carried-forward note, no
        scheduled-run prompt, no RESULT/NOTE/STATUS closing-line contract and
        no third-party-send hold, all of which the scheduled-run contract
        (schedrun.py, schedules.py) requires."""
        if not self.jev_agent or req.schedule_id:
            model = await self._pick_model(req)
            run_kwargs: Dict[str, Any] = {}
            if req.schedule_id:
                sched = self.store.get_schedule(req.schedule_id)
                if sched is not None:
                    run_kwargs = {"timezone": sched.timezone, "note": sched.last_note}
            status = await self.executor.run(req, context=context, model=model,
                                             from_owner=_owner_request(self, req),
                                             **run_kwargs)
            status = await self._grant_what_it_asked_for(req, status, context, model)
            return await self._insist_it_actually_ran(req, status, context, model)
        status = await self.jev_agent.run(req, context=context)
        logger.info("[gate] jev agent #%s: ok=%s steps=%s jev=%s prose=%s %ss",
                    req.id, status.get("ok"), status.get("tool_calls"), status.get("jev_calls"),
                    status.get("prose_calls"), status.get("seconds"))
        if status.get("confirm") or not self.jev_fallback:
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
        fallback["escalated"] = True
        fallback["jev_attempt"] = {k: status.get(k) for k in ("error", "tool_calls", "jev_calls", "prose_calls", "seconds", "steps", "raw")}
        return fallback

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

    def vault_attachment_for(self, req: Request) -> str:
        """The owner's vault notes this approved request's knowledge scopes cover,
        as a delimited block for the run's context. Computed fresh from req.scopes
        at run time (never from req.prompt, which is hash-locked from approval), so
        it cannot be widened by anything that happened between approval and running:
        the scopes it reads here are exactly the ones Aaron's approval (or his own
        trust) already covers. Holding the whole-vault vault_read scope as well changes
        only what the vault tools may open afterwards, not what is attached here."""
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
        budget = int(_settings.get("GATE_VAULT_ATTACH_BUDGET_CHARS") or 60000)
        attachment = vaultscopes.build_attachment(root, notes, budget_chars=budget)
        if attachment.overflow:
            logger.info("[gate] #%s vault attachment: %d notes attached, %d over budget listed by path",
                        req.id, len(attachment.included), len(attachment.overflow))
        return attachment.block

    def retry_request(self, req: Request, *, approved: bool) -> Optional[Request]:
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
            task_id=task_id, inbound_id=req.inbound_id, bypass_dedupe=True)
        return new_req

    async def execute(self, req: Request, notify: bool = True) -> str:
        """Run an approved request. With notify=False the caller delivers the
        result itself (a live phone call reads it aloud) and gets it back."""
        self.store.set_state(req.id, "running")
        # The revision this run started from, as _fire recorded it on the request's own
        # status -- the `req` object here can be the one created before _fire wrote that
        # status (stale in memory), so re-read it from the store rather than trust req.status.
        schedule_revision = (req.status or {}).get("schedule_revision")
        if schedule_revision is None and req.schedule_id:
            fresh = self.store.get_request(req.id)
            schedule_revision = (fresh.status or {}).get("schedule_revision") if fresh else None
        session = self.get(req.chat_id)
        slow_ack: Optional[asyncio.Task] = None
        if notify and req.mode in ("imessage", "telegram"):
            session.typing_start()
            slow_ack = asyncio.create_task(self._ack_if_slow(session, req))
        tid = self.store.task_id_for_request(req.id)
        context = self.store.task_memory_for_task(tid) if tid else ""
        vault_block = self.vault_attachment_for(req)
        if vault_block:
            context = f"{context}\n\n{vault_block}" if context else vault_block
        logger.info("[gate] executing #%s with ledger T%s (%d chars, vault attachment %d chars)",
                    req.id, tid, len(context), len(vault_block))
        try:
            status = await self._run_executor(req, context)
        finally:
            if slow_ack is not None and not slow_ack.done():
                slow_ack.cancel()
            if not notify:
                session.typing_stop()
        if status.get("confirm"):
            session.typing_stop()
            await self.ask_confirmation(req, status)
            return "I need your yes before I change anything; I have texted you what it is."
        ok = bool(status.get("ok"))
        if schedule_revision is not None:
            status["schedule_revision"] = schedule_revision
        self.store.set_state(req.id, "done" if ok else "failed",
                             status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "idle")
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
        outcome = "done" if ok else "FAILED"
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
        from_aaron = session.is_approver() or session.is_aaron_on_phone() or req.chat_id in self._approver_chat_ids()
        note = f"Task #{req.id} {outcome}: {req.summary}\nResult:\n{result}"
        # Messages live on the person's record, which for the owner spans every surface.
        meanwhile = self.store.inbound_since(self.get(req.chat_id).record_id, req.created_at)
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
                        await self.send_to_approver(f"Failed: {title}\n{message}")
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
                if not replied:
                    await self.send_to_approver(f"Done: {title}\n{message}")
            finally:
                session.typing_stop()
            return ""
        # Someone else's request: short status to Aaron, and the router may
        # tell the sender the outcome -- except on failure, which is always
        # wording.py's own text, never the router's rewrite of it.
        if not ok:
            if told_already:
                same_limit = wording.same_limit_message(status)
                await self.send_to_approver(f"[Blatbot #{req.id} FAILED, same limit] {title}")
                await session.notify_failure(same_limit, note, inbound_id=req.inbound_id)
            else:
                await self.send_to_approver(f"[Blatbot #{req.id} FAILED] {title}\n{message}")
                await session.notify_failure(message, note, inbound_id=req.inbound_id)
            session.typing_stop()
            return ""
        try:
            await self.send_to_approver(f"[Blatbot #{req.id} {outcome}] {title}\n{message}")
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
                    f"[Blatbot #{req.id}] expired after 24h with no answer: {req.summary}")
            elif req.schedule_id and (req.status or {}).get("confirm"):
                await self.schedules.held_ended(req, "expired")
            else:
                await self.send_to_approver(
                    f"[Blatbot #{req.id}] expired after 24h with no answer: {req.summary}")
            self.store.add_message(req.chat_id, "system", f"Task #{req.id} expired unanswered: {req.summary}")
            self.task_note(req, "expired", f"Expired after 24h with no answer from Aaron: {req.summary}", state="open")

    async def _expiry_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            try:
                await self._expire_pending()
            except Exception:
                logger.exception("expiry loop error")

    async def _schedule_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                await self.schedules.tick()
            except Exception:
                logger.exception("schedule loop error")
