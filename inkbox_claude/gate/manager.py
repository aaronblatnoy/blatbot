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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .executor import Executor
from .jevagent import JevAgent, enabled as jev_agent_enabled
from .taskpick import TaskPicker, enabled as jev_enabled
from .router import Router, RouterOutput
from .scopes import READ_SCOPES, SCOPES
from .store import Person, Request, Store, TaskRequired, task_key

logger = logging.getLogger(__name__)

SendFn = Callable[[str, str, str, Dict[str, Any]], Awaitable[Any]]

HANDOFF_LINE = (
    "Thanks! I'll confirm that with Aaron and get back to you shortly.\n\n"
    "Blatbot\nExecutive Assistant to Aaron Blatnoy"
)
EXPIRE_SECONDS = 24 * 3600

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
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


class GateSession:
    def __init__(self, chat_id: str, manager: "GateSessionManager"):
        self.chat_id = chat_id
        self.m = manager
        self.mode = "email"
        self.reply_meta: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._turn_task_id: Optional[int] = None  # task the current turn's inbound was logged on
        self._found_task_ids: List[int] = []      # tasks surfaced by this turn's lookup

    # -- identity ----------------------------------------------------------
    def is_approver(self) -> bool:
        conv = str((self.reply_meta or {}).get("conversation_id") or "")
        if self.mode == "imessage" and bool(conv) and conv == self.m.approver_conv:
            return True
        # Phone: caller ID can be spoofed, so Aaron's number only counts as the
        # approver when GATE_VOICE_TRUST_APPROVER=1. Otherwise his spoken requests
        # go to his iMessage for a yes like anyone else's.
        return (self.mode == "voice" and self.m.voice_trust_approver
                and bool(self.m.approver_phone) and _digits(self._sender()) == _digits(self.m.approver_phone))

    def is_aaron_on_phone(self) -> bool:
        return (self.mode == "voice" and bool(self.m.approver_phone)
                and _digits(self._sender()) == _digits(self.m.approver_phone))

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

    def _contact_notes(self) -> str:
        c = (self.reply_meta or {}).get("contact") or {}
        return str(c.get("notes") or "") if isinstance(c, dict) else ""

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
            self.m.store.task_event(task["id"], "inbound", body, chat_id=self.chat_id)
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
        self.m.store.task_event(task["id"], kind, line, chat_id=self.chat_id, state=new_state)
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
                                      chat_id=self.chat_id if tq.this_conversation else "",
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

    def candidate_tasks(self) -> List[Dict[str, Any]]:
        """The task objects behind the ledger text the router saw this turn."""
        ids: List[int] = []
        approver = self.is_approver()
        if not approver:
            for t in self.m.store.tasks_for_person(self._sender(), open_only=False, limit=6):
                ids.append(int(t["id"]))
        for tid in self.m.store.task_ids_for_chat(self.chat_id):
            if tid not in ids:
                ids.append(tid)
        if approver:
            for t in self.m.store.recent_tasks():
                if t["id"] not in ids:
                    ids.append(int(t["id"]))
        ids += [i for i in self.m.store.task_ids_for_thread(self.thread_key()) if i not in ids]
        ids += [i for i in self._found_task_ids if i not in ids]
        out: List[Dict[str, Any]] = []
        for tid in ids[:40]:
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
        convo = "\n".join(f"[{m.get('kind')}] {str(m.get('text') or '')[:600]}" for m in history[-8:]) or "(none)"
        ledger = self.m.store.task_memory_for_task(int(task["id"])) or "(new task, no events yet)"
        subject = str((self.reply_meta or {}).get("subject") or "").strip()
        return (
            f"A message arrived over {self.mode} from {who} ({addr}).\n"
            + (f"Subject: {subject}\n" if subject else "")
            + f"\n--- THEIR MESSAGE (verbatim) ---\n{body.strip()}\n\n"
            f"--- RECENT CONVERSATION WITH THEM ---\n{convo}\n\n"
            f"--- THE TASK THIS BELONGS TO ---\n{ledger}\n\n"
            "Work out what they are asking for, or what this message makes possible on the task (for example, "
            "information the task was waiting for), and do it with your tools. Do only what the message and the "
            "task call for; do not invent extra steps. If something essential is missing, stop and say exactly "
            "what is missing instead of guessing."
        )

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
        picker = self.m.task_picker
        if self.m.router_defines_requests or picker is None:
            out = await self.m.router.route(history=prior, message=message, mode=mode, sender=self._sender(),
                                            contact_notes=self._contact_notes(), is_approver=approver,
                                            task_memory=memory, found_tasks=found)
            await self.jev_pick(message, prior, out)
            await self.jev_action(message, prior, out, self._peek_task(out))
            task = None
            if out.request is not None or _names_a_task(out.task):
                task = self.m.resolve_task(self, out)
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
            needs_action = float(act.get("p") or 0.0) >= 0.5
        logger.info("[gate %s] jev-first: task=%s action=%s (p=%.2f)", self.chat_id, task_choice, needs_action,
                    float(act.get("p") or 0.0))
        out = await self.m.router.route(history=prior, message=message, mode=mode, sender=self._sender(),
                                        contact_notes=self._contact_notes(), is_approver=approver,
                                        task_memory=memory, found_tasks=found, action=needs_action,
                                        action_task=(peek or {}).get("title") or "")
        out.request = None  # the router never defines the request on this path
        if not needs_action and out.reply and _promises_action(out.reply):
            # No request exists, so a reply that says "on it" / "submitting now" would be a lie.
            logger.info("[gate %s] reply promised an action but none was decided; rewriting", self.chat_id)
            out2 = await self.m.router.route(history=prior, message=message, mode=mode, sender=self._sender(),
                                             contact_notes=self._contact_notes(), is_approver=approver,
                                             task_memory=memory, found_tasks=found, action=False,
                                             action_task="(NOTE: no request has been created and none will run for this "
                                                         "message; do not say you are doing, submitting or checking anything. "
                                                         "Answer from what is known, or ask what is needed.)")
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
            task = self.m.resolve_task(self, out)
            self._ensure_inbound_on_task(task, body if mode != "voice" else f"(phone) {body}")
            if needs_action:
                from .router import RouterRequest
                emails = [e for e in _EMAIL.findall(message) if e.lower() != self._sender().lower()]
                out.request = RouterRequest(prompt=self.build_task_prompt(body, prior, task),
                                            scopes=["web"], summary=str(task.get("title") or message[:100])[:160],
                                            counterpart=emails[0] if (approver and emails) else None)
                await self.jev_scopes(out, strict=True)
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
            title = (task or {}).get("title") or (out.task_title or "") or message[:80]
            out.request = RouterRequest(prompt="(built from source)", scopes=["web"], summary=f"Act on: {title}"[:120])
            logger.info("[gate %s] jev: action needed (p=%.2f); router proposed none", self.chat_id, res["p"])
        elif not na and out.request is not None:
            logger.info("[gate %s] jev: no action needed (p=%.2f); dropping router request %r",
                        self.chat_id, res["p"], out.request.summary)
            out.request = None

    async def jev_scopes(self, out: RouterOutput, strict: bool = False) -> None:
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
            res = await picker.judge_scopes_tree(prompt=out.request.prompt, summary=out.request.summary)
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
            if self.is_approver():
                chosen = list(dict.fromkeys(chosen + READ_SCOPES + [k for k, v in probs.items() if v >= 0.35]))
            elif not chosen:
                top = [k for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:2] if v >= 0.25]
                chosen = top or ["web"]
            logger.info("[gate %s] scopes granted: %s", self.chat_id, chosen)
        if not chosen:
            logger.info("[gate %s] scope judgment undecided (%s); keeping router scopes %s",
                        self.chat_id, res.get("reason"), out.request.scopes)
            return
        if set(chosen) != set(out.request.scopes):
            logger.info("[gate %s] scopes: router %s -> jev %s", self.chat_id, out.request.scopes, chosen)
        out.request.scopes = chosen

    def task_memory(self) -> str:
        mem = self.m.store.task_memory(self.chat_id, is_approver=self.is_approver(),
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
            body = _strip_quoted(text) if mode == "email" else text.strip()
            self.inbound_id = self.m.store.add_message(self.chat_id, "inbound", body, mode)
            if self.thread_key():
                self.m.store.link_thread(self.thread_key(), self.chat_id)
            # First thing every turn: put this message on the person's task ledger,
            # so the router starts from the record, not from a 20-message window.
            t = self.task()
            self._turn_task_id = t["id"] if t is not None else None
            if t is not None:
                self.m.store.task_event(t["id"], "inbound", body, chat_id=self.chat_id)
            try:
                if self.is_approver() and self.mode != "voice" and await self.m.handle_approver_command(self, body):
                    return
                if not self.is_approver() and self.mode != "voice":
                    # Deterministic FYI to Aaron for every inbound from anyone else.
                    await self.m.notify_inbound(self, body)
                await self._route_and_act(body)
            except Exception as exc:
                logger.exception("[gate %s] turn failed", self.chat_id)
                await self.m.report_outage(exc, self.chat_id)
            finally:
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
                self.m.store.task_event(t["id"], "call_ended", "Phone call ended.", chat_id=self.chat_id)
            self.m.store.add_message(self.chat_id, "system", "Phone call ended.")
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
                    rows = self.m.store.recent_tasks(limit=20)
                    out = []
                    for r in rows:
                        title = str(r["title"] or "")
                        if title.startswith("Thread with") or r["state"] == "closed":
                            continue  # a bare conversation, not a task
                        hrs = (now - r["updated_at"]) / 3600
                        age = f"{int(hrs * 60)} min ago" if hrs < 1 else (f"{int(hrs)} hours ago" if hrs < 48 else f"{int(hrs / 24)} days ago")
                        state = {"open": "in progress", "waiting_aaron": "waiting on Aaron's okay",
                                 "running": "running now", "done": "done", "failed": "didn't work"}.get(r["state"], r["state"])
                        out.append(f"- {title} ({state}, {age})")
                    notes = ("Recent things you've been working on, newest first. This is background, not a "
                             "menu and not a limit on what you can do:\n" + "\n".join(out))
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

    async def voice_consult(self, query: str, meta: Dict[str, Any]) -> str:
        """One consult from the realtime voice model during a live call.

        Same path as any inbound message (ledger, router, gate), except the
        answer is returned for the voice model to say instead of being sent.
        Aaron's trusted number runs the request now and hears the result;
        anyone else's request goes to Aaron for a yes and they are told so."""
        req = None
        async with self._lock:
            prev = (self.mode, self.reply_meta)
            self.mode, self.reply_meta = "voice", dict(meta or {})
            self.m.store.set_thread(self.chat_id, "routing")  # state only: keep their text/email route
            body = (query or "").strip()
            self.inbound_id = self.m.store.add_message(self.chat_id, "inbound", body, "voice")
            t = self.task()
            self._turn_task_id = t["id"] if t is not None else None
            if t is not None:
                self.m.store.task_event(t["id"], "inbound", f"(phone) {body}", chat_id=self.chat_id)
            try:
                approver = self.is_approver()
                history = self.m.store.history(self.chat_id, limit=20)
                memory = self.task_memory()
                logger.info("[gate %s] ledger loaded (voice consult): %d chars", self.chat_id, len(memory))
                found = await self.lookup_tasks(history[:-1], body, memory)
                out, task = await self.decide(body=body, message=body, prior=history[:-1], mode="voice",
                                              memory=memory, found=found)
                reply = out.reply or ""
                if reply:
                    self.m.store.add_message(self.chat_id, "outbound", reply, "voice")
                    if t is not None:
                        self.m.store.task_event(t["id"], "outbound", f"(phone) {reply}", chat_id=self.chat_id)
                if out.request is None:
                    return reply or "I don't have anything on that."
                if self.m.store.pending_for_thread(self.chat_id) and not approver:
                    return reply or "I already have a request waiting on Aaron for you."
                assert task is not None
                req = self.m.store.create_request(
                    chat_id=self.chat_id, sender=self._sender(), sender_name=self._sender_name(), mode="voice",
                    subject="", original_message=body, summary=out.request.summary,
                    scopes=out.request.scopes, prompt=out.request.prompt,
                    state="approved" if approver else "pending", task_id=task["id"], inbound_id=self.inbound_id)
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
            return await asyncio.wait_for(asyncio.shield(self._run_for_call(req, aaron_call, caller)), timeout=75)
        except asyncio.TimeoutError:
            return "That is still running. I will text you the result as soon as it finishes."

    async def _run_for_call(self, req: Request, aaron_call: bool, caller: str) -> str:
        text = await self.m.execute(req, notify=False)
        if not self.m.call_active(self.chat_id):
            # They hung up first: deliver by text, without touching the thread's channel.
            if aaron_call:
                await self.m.send_to_approver(text)
            else:
                await self.m.send_fn(f"sms:{caller}", text, "sms", {"to": caller, "sender": caller})
            self.m.store.add_message(self.chat_id, "outbound", text, "sms")
        return text

    async def run_consult_detailed(self, *args: Any, **kwargs: Any) -> Any:
        return await self.run_consult()

    async def close(self) -> None:
        return None

    # -- core ----------------------------------------------------------------
    async def _route_and_act(self, body: str, system_note: Optional[str] = None) -> bool:
        approver = self.is_approver()
        history = self.m.store.thread_history(self.thread_key(), exclude_chat=self.chat_id) + \
            self.m.store.history(self.chat_id, limit=20)
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
            out = await self.m.router.route(
                history=prior, message=message, mode=self.mode, sender=self._sender(),
                contact_notes=self._contact_notes(), is_approver=approver,
                task_memory=memory, found_tasks=found, action=True,
                action_task=(t or {}).get("title") or running.summary,
            )
            out.request = None
            if out.reply:
                await self.send_to_sender(out.reply, role="ack")
            return bool(out.reply)
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
        sent_reply = False
        if out.reply:
            # Alongside a request the reply is an acknowledgement; on its own, or when
            # phrasing a result, it is THE answer to that inbound message.
            sent_reply = await self.send_to_sender(out.reply, role="ack" if out.request is not None else "answer")
        if out.request is None:
            return sent_reply
        # Validate scopes (pydantic already rejected unknown ones).
        if self.m.store.pending_for_thread(self.chat_id) and not approver:
            self.m.store.add_message(self.chat_id, "system", "A request is already awaiting Aaron; new request not created.")
            return sent_reply
        assert task is not None
        req = self.m.store.create_request(
            chat_id=self.chat_id, sender=self._sender(), sender_name=self._sender_name(), mode=self.mode,
            subject=str(self.reply_meta.get("subject") or ""), original_message=body,
            summary=out.request.summary, scopes=out.request.scopes, prompt=out.request.prompt,
            state="approved" if approver else "pending", task_id=task["id"], inbound_id=self.inbound_id,
        )
        self.m.record_request_on_task(req, task, self)
        if approver:
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

    async def send_to_sender(self, text: str, role: str = "answer") -> bool:
        """Deliver to the sender. Enforced: per inbound message at most one 'ack' and
        one 'answer' go out. A second of either is refused here, whatever produced it.
        Returns whether it was sent."""
        given = self.m.store.responses_to(self.inbound_id or 0)
        if self.inbound_id and role in given:
            logger.warning("[gate %s] refused a second %s for inbound #%s: %r", self.chat_id, role, self.inbound_id, text)
            return False
        if self.mode == "voice" and not self.m.call_active(self.chat_id):
            # The call ended before this was ready: deliver by text instead.
            if self.is_aaron_on_phone():
                await self.m.send_to_approver(text)
            else:
                num = self._sender()
                await self.m.send_fn(f"sms:{num}", text, "sms", {"to": num, "sender": num})
        else:
            await self.m.send_fn(self.chat_id, text, self.mode, self.reply_meta)
        self.m.store.add_message(self.chat_id, "outbound", text, self.mode, reply_to=self.inbound_id, role=role)
        t = self.task()
        if t is not None:
            self.m.store.task_event(t["id"], "outbound", text, chat_id=self.chat_id)
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
        logger.warning("[gate %s] rewrite still not grounded (p=%.2f); sending it with a caveat", self.chat_id, p2)
        body = _result_bodies(results)
        if len(body) <= 1500:
            return body                                   # short enough to show as is
        return reply2 + "\n\nI could not fully verify that against the data I pulled; tell me if it looks off."

    async def handle_followup(self, body: str, inbound_id: Optional[int] = None) -> bool:
        """Re-decide a message that was received while a request was running. The
        message is already stored on the thread, so this only routes and acts."""
        async with self._lock:
            if inbound_id:
                self.inbound_id = inbound_id
            return await self._route_and_act(body)

    async def notify_after_request(self, note: str, inbound_id: Optional[int] = None) -> bool:
        """Phrase a request's result for the sender. The result is THE answer to the
        inbound that raised the request; if that inbound already has an answer the
        send is refused (returns True: handled, nothing more to deliver)."""
        async with self._lock:
            if inbound_id:
                self.inbound_id = inbound_id
            self.m.store.add_message(self.chat_id, "system", note)
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


class GateSessionManager:
    def __init__(self, *, cfg: Any, send_fn: SendFn, mcp_server: Any, identity_info: Dict[str, str],
                 store_path: str, exec_cwd: str, standing_path: Optional[str] = None):
        self.cfg = cfg
        self.send_fn = send_fn
        self.identity_info = identity_info
        self.store = Store(store_path)
        self.router = Router(api_key=cfg.deepseek_api_key, model=cfg.deepseek_model, standing_path=standing_path)
        protected = [str(cfg.approver_imessage_conversation_id or ""), str(getattr(cfg, "approver_phone", "") or "")]
        self.executor = Executor(mcp_server=mcp_server, cwd=exec_cwd, model=cfg.claude_model or "sonnet",
                                 protected=protected)
        # GATE_EXECUTOR=jev runs requests through the typed-judgment agent
        # (Jev picks tools, code fills arguments, DeepSeek writes prose only when
        # needed); Claude Code stays as the fallback unless GATE_EXECUTOR_FALLBACK=none.
        self.jev_agent: Optional[JevAgent] = JevAgent(inkbox_server=mcp_server, router=self.router,
                                                      protected=protected) if jev_agent_enabled() else None
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
        self.voice_vocabulary = str(os.getenv("GATE_VOICE_VOCABULARY") or DEFAULT_VOICE_VOCABULARY).strip()
        self.voice_trust_approver = str(os.getenv("GATE_VOICE_TRUST_APPROVER") or "").strip().lower() in ("1", "true", "yes")
        self.sessions: Dict[str, GateSession] = {}
        self._expiry_task: Optional[asyncio.Task] = None
        try:
            self._expiry_task = asyncio.get_running_loop().create_task(self._expiry_loop())
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

    # -- approver channel ----------------------------------------------------
    async def send_to_approver(self, text: str) -> None:
        conv = self.approver_conv
        await self.send_fn(f"imessage:{conv}", text, "imessage", {"conversation_id": conv})

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

    def resolve_task(self, session: GateSession, out: RouterOutput) -> Dict[str, Any]:
        """Decide which task a request belongs to. Always returns a task.

        The router proposes: an existing id ("T12"), "new" with a title, or nothing.
        Code checks the proposal: a non-owner may only continue a task they are on;
        an unknown or missing choice falls back to the sender's most recent open task
        (for a non-owner) or to a new task titled from the request summary. Whatever
        happens, the request ends up on exactly one task. This cannot be skipped."""
        req = out.request
        summary = (req.summary if req else "") or "Untitled task"
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

    async def ask_confirmation(self, req: Request, status: Dict[str, Any]) -> None:
        """A destructive call is waiting. Park the request as pending with the exact call
        stored, and text the owner what it is. '#N yes' performs that one call."""
        c = status["confirm"]
        self.store.set_state(req.id, "pending", status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "awaiting_aaron")
        short = c["tool"].split("__")[-1]
        shown = {k: v for k, v in c["args"].items() if k != "user_google_email"}
        about = "\n".join(f"  {ln}" for ln in c.get("about") or []) or "  (no further detail found)"
        self.task_note(req, "request", f"Waiting on Aaron to confirm: {short} {json.dumps(shown, ensure_ascii=False)}", state="waiting_aaron")
        await self.send_to_approver(
            f"[Blatbot #{req.id}] Before I do this, I need your yes.\n"
            f"Action: {short} {json.dumps(shown, ensure_ascii=False)}\n"
            f"It refers to:\n{about}\n"
            f"Reply \"#{req.id} yes\" to do it or \"#{req.id} no\" to leave it alone. Nothing has been changed yet.")

    async def perform_confirmed(self, req: Request) -> None:
        """The owner said yes: run exactly the stored call, then finish the request."""
        c = (req.status or {}).get("confirm") or {}
        self.store.set_state(req.id, "running")
        if self.jev_agent is None or not c:
            status = {"ok": False, "error": "no confirmed call on record", "summary": "no confirmed call on record", "raw": "", "tool_calls": []}
        else:
            status = await self.jev_agent.perform(c["tool"], c["args"])
        ok = bool(status.get("ok"))
        self.store.set_state(req.id, "done" if ok else "failed", status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "idle")
        result = _clean_result(status.get("summary") or status.get("error") or "")
        self.task_note(req, "done" if ok else "failed", f"{req.summary} -> {result}", state="done" if ok else "failed")
        session = self.get(req.chat_id)
        note = f"Task #{req.id} {'done' if ok else 'FAILED'}: {req.summary}\nResult:\n{result}"
        replied = await session.notify_after_request(note, inbound_id=req.inbound_id)
        if not replied:
            await self.send_to_approver(f"{'Done' if ok else 'Failed'}: {req.summary}\n{result}")

    async def handle_approver_command(self, session: GateSession, text: str) -> bool:
        """Parse Aaron's reply as a gate command. Returns True if consumed."""
        if await self.handle_task_command(session, text):
            return True
        pending = self.store.pending()
        if not pending:
            return False
        t = text.strip()
        m = re.match(r"^\s*#?(\d+)\s*[:\-]?\s*(.*)$", t, re.S)
        req: Optional[Request] = None
        rest = t
        if m and any(p.id == int(m.group(1)) for p in pending):
            req = next(p for p in pending if p.id == int(m.group(1)))
            rest = m.group(2).strip()
        elif len(pending) == 1 and t.lower() in (_YES | _NO):
            req = pending[0]
        if req is None:
            return False
        low = rest.lower()
        if low in _YES and (req.status or {}).get("confirm"):
            self.store.add_message(session.chat_id, "system", f"Aaron confirmed the pending change on request #{req.id}.")
            asyncio.create_task(self.perform_confirmed(req))
            return True
        if low in _NO and (req.status or {}).get("confirm"):
            self.store.set_state(req.id, "rejected")
            self.store.set_thread(req.chat_id, "idle")
            self.task_note(req, "rejected", f"Aaron declined the change: {req.summary}", state="open")
            await self.send_to_approver(f"Left alone. Nothing was changed for #{req.id}.")
            return True
        if low in _YES:
            self.store.set_state(req.id, "approved")
            self.store.add_message(session.chat_id, "system", f"Approved request #{req.id}.")
            self.task_note(req, "approved", f"Aaron approved: {req.summary}", state="running")
            asyncio.create_task(self.execute(self.store.get_request(req.id)))  # type: ignore[arg-type]
            return True
        if low in _NO:
            self.store.set_state(req.id, "rejected")
            self.store.set_thread(req.chat_id, "idle")
            self.task_note(req, "rejected", f"Aaron declined: {req.summary}", state="open")
            await self.send_to_approver(f"Dropped #{req.id}.")
            asyncio.create_task(self.get(req.chat_id).notify_after_request(
                f"Aaron declined the task: {req.summary}. Nothing was done. Reply to the sender briefly if appropriate."))
            return True
        em = re.match(r"^edit\s*:\s*(.+)$", rest, re.S | re.I)
        if em:
            note = em.group(1).strip()
            base = req.prompt.split("\n\n--- AARON'S INSTRUCTIONS ---")[0]
            new = self.store.revise_prompt(req.id, f"{base}\n\n--- AARON'S INSTRUCTIONS ---\n{note}")
            await self.send_approval_text(new, revised=True)
            return True
        return False

    # -- executor --------------------------------------------------------------
    async def _run_executor(self, req: Request, context: str) -> Dict[str, Any]:
        """Run one request and return exactly ONE status object. Claude Code is an
        escalation inside this call, not a second execution: nothing is delivered,
        noted, or phrased for the Jev attempt when it escalates; only the final
        status leaves this function. Executors themselves never message anyone who
        asked (denied at the tool call)."""
        if not self.jev_agent:
            return await self.executor.run(req, context=context)
        status = await self.jev_agent.run(req, context=context)
        logger.info("[gate] jev agent #%s: ok=%s steps=%s jev=%s prose=%s %ss",
                    req.id, status.get("ok"), status.get("tool_calls"), status.get("jev_calls"),
                    status.get("prose_calls"), status.get("seconds"))
        if status.get("ok") or status.get("confirm") or not self.jev_fallback:
            return status
        if status.get("wrote"):
            logger.info("[gate] jev agent #%s failed after a write; not falling back", req.id)
            return status
        logger.info("[gate] jev agent gave up on #%s (%s); falling back to claude code", req.id, status.get("error"))
        findings = status.get("raw") or ""
        fallback = await self.executor.run(req, context=context, prior_work=findings)
        fallback["escalated"] = True
        fallback["jev_attempt"] = {k: status.get(k) for k in ("error", "tool_calls", "jev_calls", "prose_calls", "seconds", "steps", "raw")}
        return fallback

    async def execute(self, req: Request, notify: bool = True) -> str:
        """Run an approved request. With notify=False the caller delivers the
        result itself (a live phone call reads it aloud) and gets it back."""
        self.store.set_state(req.id, "running")
        tid = self.store.task_id_for_request(req.id)
        context = self.store.task_memory_for_task(tid) if tid else ""
        logger.info("[gate] executing #%s with ledger T%s (%d chars)", req.id, tid, len(context))
        status = await self._run_executor(req, context)
        if status.get("confirm"):
            await self.ask_confirmation(req, status)
            return "I need your yes before I change anything; I have texted you what it is."
        ok = bool(status.get("ok"))
        self.store.set_state(req.id, "done" if ok else "failed",
                             status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "idle")
        outcome = "done" if ok else "FAILED"
        result = _clean_result(status.get("summary") or status.get("error") or "")
        self.task_note(req, "done" if ok else "failed", f"{req.summary} -> {result}",
                       state="done" if ok else "failed")
        if not notify:
            self.store.add_message(req.chat_id, "system", f"Task #{req.id} {outcome}: {req.summary}\nResult:\n{result}")
            return f"{'Done' if ok else 'That failed'}. {result}"
        session = self.get(req.chat_id)
        from_aaron = session.is_approver() or session.is_aaron_on_phone() or req.chat_id in self._approver_chat_ids()
        note = f"Task #{req.id} {outcome}: {req.summary}\nResult:\n{result}"
        meanwhile = self.store.inbound_since(req.chat_id, req.created_at)
        if from_aaron:
            if not ok and meanwhile:
                # The run failed and Aaron said more while it ran (a hint, a correction):
                # decide that latest message now, as one fresh turn. Exactly one reply.
                logger.info("[gate] #%s failed with %d message(s) received meanwhile; re-deciding the latest",
                            req.id, len(meanwhile))
                self.store.add_message(req.chat_id, "system", note)
                try:
                    await session.handle_followup(meanwhile[-1]["text"], inbound_id=meanwhile[-1]["id"])
                except Exception:
                    logger.exception("[gate] follow-up after #%s failed", req.id)
                return ""
            # Aaron asked for it himself: let the router phrase the answer.
            replied = await session.notify_after_request(note, inbound_id=req.inbound_id)
            if not replied:
                await self.send_to_approver(f"{'Done' if ok else 'Failed'}: {req.summary}\n{result}")
            return ""
        # Someone else's request: short status to Aaron, and the router may
        # tell the sender the outcome (never Claude's raw text).
        await self.send_to_approver(f"[Blatbot #{req.id} {outcome}] {req.summary}\n{result}")
        await session.notify_after_request(note, inbound_id=req.inbound_id)
        return ""

    def _approver_chat_ids(self) -> set:
        out = set()
        for cid, sess in self.sessions.items():
            if sess.is_approver():
                out.add(cid)
        return out

    # -- expiry ----------------------------------------------------------------
    async def _expiry_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            try:
                for req in self.store.expire_older_than(EXPIRE_SECONDS):
                    self.store.set_thread(req.chat_id, "idle")
                    await self.send_to_approver(f"[Blatbot #{req.id}] expired after 24h with no answer: {req.summary}")
                    self.store.add_message(req.chat_id, "system", f"Task #{req.id} expired unanswered: {req.summary}")
                    self.task_note(req, "expired", f"Expired after 24h with no answer from Aaron: {req.summary}", state="open")
            except Exception:
                logger.exception("expiry loop error")
