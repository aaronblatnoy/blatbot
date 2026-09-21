"""Gate session manager: drop-in for SessionManager when INKBOX_MODE=gate.

Interface the gateway uses:
  get(chat_id, system_prompt_extra="") -> session with
      handle_inbound(text, mode, meta), run_consult(...), run_consult_detailed(...), close()
  close_all()
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .executor import Executor
from .router import Router, RouterOutput
from .scopes import SCOPES
from .store import Request, Store, task_key

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

    def _contact_notes(self) -> str:
        c = (self.reply_meta or {}).get("contact") or {}
        return str(c.get("notes") or "") if isinstance(c, dict) else ""

    # -- task ledger -----------------------------------------------------------
    def task(self) -> Optional[Dict[str, Any]]:
        """The live task for this thread's counterpart. Aaron's own thread has none."""
        if self.is_approver():
            return None
        name = self._sender_name()
        return self.m.store.task_for(self._sender(), name or self._sender())

    def task_memory(self) -> str:
        return self.m.store.task_memory(self.chat_id, is_approver=self.is_approver())

    # -- gateway interface ---------------------------------------------------
    async def handle_inbound(self, text: str, mode: str, meta: Dict[str, Any]) -> None:
        async with self._lock:
            self.mode = mode
            self.reply_meta = dict(meta or {})
            self.m.store.set_thread(self.chat_id, "routing", mode, self.reply_meta_for_store())
            body = _strip_quoted(text) if mode == "email" else text.strip()
            self.m.store.add_message(self.chat_id, "inbound", body, mode)
            # First thing every turn: put this message on the person's task ledger,
            # so the router starts from the record, not from a 20-message window.
            t = self.task()
            if t is not None:
                self.m.store.task_event(t["id"], "inbound", body, chat_id=self.chat_id,
                                        state="open" if t["state"] in ("done", "failed") else None)
            try:
                if self.is_approver() and self.mode != "voice" and await self.m.handle_approver_command(self, body):
                    return
                if not self.is_approver() and self.mode != "voice":
                    # Deterministic FYI to Aaron for every inbound from anyone else.
                    await self.m.notify_inbound(self, body)
                await self._route_and_act(body)
            except Exception:
                logger.exception("[gate %s] turn failed", self.chat_id)
            finally:
                if self.m.store.thread_state(self.chat_id) == "routing":
                    self.m.store.set_thread(self.chat_id, "idle")

    def reply_meta_for_store(self) -> Dict[str, Any]:
        keep = {k: v for k, v in (self.reply_meta or {}).items()
                if k in ("to", "sender", "subject", "thread_id", "conversation_id", "conversation_kind")}
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
                        if title.startswith("Thread with"):
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
            self.m.store.add_message(self.chat_id, "inbound", body, "voice")
            t = self.task()
            if t is not None:
                self.m.store.task_event(t["id"], "inbound", f"(phone) {body}", chat_id=self.chat_id,
                                        state="open" if t["state"] in ("done", "failed") else None)
            try:
                approver = self.is_approver()
                history = self.m.store.history(self.chat_id, limit=20)
                memory = self.task_memory()
                logger.info("[gate %s] ledger loaded (voice consult): %d chars", self.chat_id, len(memory))
                out = await self.m.router.route(
                    history=history[:-1], message=body, mode="voice", sender=self._sender(),
                    contact_notes=self._contact_notes(), is_approver=approver, task_memory=memory)
                reply = out.reply or ""
                if reply:
                    self.m.store.add_message(self.chat_id, "outbound", reply, "voice")
                    if t is not None:
                        self.m.store.task_event(t["id"], "outbound", f"(phone) {reply}", chat_id=self.chat_id)
                if out.request is None:
                    return reply or "I don't have anything on that."
                if self.m.store.pending_for_thread(self.chat_id) and not approver:
                    return reply or "I already have a request waiting on Aaron for you."
                req = self.m.store.create_request(
                    chat_id=self.chat_id, sender=self._sender(), sender_name=self._sender_name(), mode="voice",
                    subject="", original_message=body, summary=out.request.summary,
                    scopes=out.request.scopes, prompt=out.request.prompt,
                    state="approved" if approver else "pending")
                self.m.link_task(req, self, out.request.counterpart)
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
        history = self.m.store.history(self.chat_id, limit=20)
        if system_note:
            prior, message = history, "(No new message from the sender. A system note with a task result was just added above. Write the reply the sender should receive: for Aaron, give him the answer or outcome clearly and naturally, in his voice preference (brief, no emojis); for anyone else, tell them only what concerns them, without internal details. Reply null only if there is truly nothing to say. Do not create a request.)"
        else:
            prior, message = history[:-1], body  # the new inbound is already stored last
        memory = self.task_memory()
        logger.info("[gate %s] ledger loaded: %d task(s), %d chars", self.chat_id,
                    memory.count("\nTask T") + (1 if memory.startswith("Task T") else 0), len(memory))
        out: RouterOutput = await self.m.router.route(
            history=prior, message=message, mode=self.mode, sender=self._sender(),
            contact_notes=self._contact_notes(), is_approver=approver,
            task_memory=memory,
        )
        if system_note and out.request is not None:
            out.request = None
        if out.reply:
            await self.send_to_sender(out.reply)
        if out.request is None:
            return bool(out.reply)
        # Validate scopes (pydantic already rejected unknown ones).
        if self.m.store.pending_for_thread(self.chat_id) and not approver:
            self.m.store.add_message(self.chat_id, "system", "A request is already awaiting Aaron; new request not created.")
            return bool(out.reply)
        req = self.m.store.create_request(
            chat_id=self.chat_id, sender=self._sender(), sender_name=self._sender_name(), mode=self.mode,
            subject=str(self.reply_meta.get("subject") or ""), original_message=body,
            summary=out.request.summary, scopes=out.request.scopes, prompt=out.request.prompt,
            state="approved" if approver else "pending",
        )
        self.m.link_task(req, self, out.request.counterpart)
        if approver:
            # Run after this turn releases the session lock; execute() will
            # re-enter the session to phrase the result.
            asyncio.create_task(self.m.execute(req))
            return True
        self.m.store.set_thread(self.chat_id, "awaiting_aaron")
        await self.m.send_approval_text(req)
        if not out.reply:
            await self.send_to_sender(HANDOFF_LINE)
        return True

    async def send_to_sender(self, text: str) -> None:
        if self.mode == "voice" and not self.m.call_active(self.chat_id):
            # The call ended before this was ready: deliver by text instead.
            if self.is_aaron_on_phone():
                await self.m.send_to_approver(text)
            else:
                num = self._sender()
                await self.m.send_fn(f"sms:{num}", text, "sms", {"to": num, "sender": num})
        else:
            await self.m.send_fn(self.chat_id, text, self.mode, self.reply_meta)
        self.m.store.add_message(self.chat_id, "outbound", text, self.mode)
        t = self.task()
        if t is not None:
            self.m.store.task_event(t["id"], "outbound", text, chat_id=self.chat_id)

    async def notify_after_request(self, note: str) -> bool:
        """Run the router once with a system note so it can reply to the sender.

        Returns True if a reply was sent."""
        async with self._lock:
            self.m.store.add_message(self.chat_id, "system", note)
            return await self._route_and_act("", system_note=note)


class GateSessionManager:
    def __init__(self, *, cfg: Any, send_fn: SendFn, mcp_server: Any, identity_info: Dict[str, str],
                 store_path: str, exec_cwd: str, standing_path: Optional[str] = None):
        self.cfg = cfg
        self.send_fn = send_fn
        self.identity_info = identity_info
        self.store = Store(store_path)
        self.router = Router(api_key=cfg.deepseek_api_key, model=cfg.deepseek_model, standing_path=standing_path)
        self.executor = Executor(mcp_server=mcp_server, cwd=exec_cwd, model=cfg.claude_model or "sonnet")
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

    async def notify_inbound(self, session: GateSession, body: str) -> None:
        who = session._sender_name()
        sender = session._sender()
        who = f"{who} ({sender})" if who else sender
        subj = str((session.reply_meta or {}).get("subject") or "").strip()
        snippet = re.sub(r"\s+", " ", body).strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
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
        return (
            f"{head}\n\n--- Their message ---\n{req.original_message}\n\n"
            f"--- Blatbot wants ---\n{req.summary}\nScopes: {', '.join(req.scopes)}\n\n"
            f"--- Exact prompt Claude will run ---\n{req.prompt}\n\n"
            f"Reply \"#{req.id} yes\" to run exactly this, \"#{req.id} no\" to drop, "
            f"or \"#{req.id} edit: <new prompt>\" to replace it (you will see it again before it runs). "
            "A bare yes/no works when this is the only one pending."
        )

    def link_task(self, req: Request, session: GateSession, counterpart: Optional[str]) -> None:
        """Attach a request to the task ledger of the person it concerns."""
        try:
            if not session.is_approver():
                t = session.task()
            else:
                who = (counterpart or "").strip() or next(iter(_EMAIL.findall(req.prompt)), "")
                if not who:
                    who = f"aaron:{req.id}"  # a personal errand with no counterpart
                t = self.store.task_for(who, counterpart or who, title=req.summary)
            if t is None:
                return
            self.store.link_request_task(req.id, t["id"])
            state = "running" if req.state == "approved" else "waiting_aaron"
            self.store.task_event(t["id"], "request", f"{req.summary} [scopes: {', '.join(req.scopes)}]",
                                  chat_id=session.chat_id, request_id=req.id, state=state,
                                  title=req.summary if t["title"].startswith("Thread with") else None)
        except Exception:
            logger.exception("[gate] task link failed for request %s", req.id)

    def task_note(self, req: Request, kind: str, text: str, state: Optional[str] = None) -> None:
        tid = self.store.task_id_for_request(req.id)
        if tid:
            self.store.task_event(tid, kind, text, chat_id=req.chat_id, request_id=req.id, state=state)

    async def send_approval_text(self, req: Request, revised: bool = False) -> None:
        await self.send_to_approver(self.format_request(req, revised))

    async def handle_approver_command(self, session: GateSession, text: str) -> bool:
        """Parse Aaron's reply as a gate command. Returns True if consumed."""
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
            new = self.store.revise_prompt(req.id, em.group(1).strip())
            await self.send_approval_text(new, revised=True)
            return True
        return False

    # -- executor --------------------------------------------------------------
    async def execute(self, req: Request, notify: bool = True) -> str:
        """Run an approved request. With notify=False the caller delivers the
        result itself (a live phone call reads it aloud) and gets it back."""
        self.store.set_state(req.id, "running")
        tid = self.store.task_id_for_request(req.id)
        context = self.store.task_memory_for_task(tid) if tid else ""
        logger.info("[gate] executing #%s with ledger T%s (%d chars)", req.id, tid, len(context))
        status = await self.executor.run(req, context=context)
        ok = bool(status.get("ok"))
        self.store.set_state(req.id, "done" if ok else "failed",
                             status=status, raw_output=status.get("raw"))
        self.store.set_thread(req.chat_id, "idle")
        outcome = "done" if ok else "FAILED"
        result = _clean_result(status.get("summary") or status.get("error") or "")
        self.task_note(req, "done" if ok else "failed", f"{req.summary} -> {result[:500]}",
                       state="done" if ok else "failed")
        if not notify:
            self.store.add_message(req.chat_id, "system", f"Task #{req.id} {outcome}: {req.summary}\nResult:\n{result}")
            return f"{'Done' if ok else 'That failed'}. {result[:600]}"
        session = self.get(req.chat_id)
        from_aaron = session.is_approver() or session.is_aaron_on_phone() or req.chat_id in self._approver_chat_ids()
        note = f"Task #{req.id} {outcome}: {req.summary}\nResult:\n{result}"
        if from_aaron:
            # Aaron asked for it himself: let the router phrase the answer.
            replied = await session.notify_after_request(note)
            if not replied:
                await self.send_to_approver(f"{'Done' if ok else 'Failed'}: {req.summary}\n{result[:600]}")
            return ""
        # Someone else's request: short status to Aaron, and the router may
        # tell the sender the outcome (never Claude's raw text).
        await self.send_to_approver(f"[Blatbot #{req.id} {outcome}] {req.summary}\n{result[:500]}")
        await session.notify_after_request(note)
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
