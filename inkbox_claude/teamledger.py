"""Ledger mode: Gondroid's features, as switches on this gateway (all off by default).

Gondroid is the Definite Optimists team's Telegram ledger. Run as an instance of this gateway
(deploy/instances/gondroid.yaml) it keeps doing exactly what deploy/gondroid/gondroid.py did, by
calling that file's own functions, while its answers go through the gate like Blatbot's:

  TELEGRAM_LEDGER=1        every text message and file in a trusted Telegram chat goes to the team
                           brain (POST /api/add with DO_BRAIN_KEY), with who said it, where and when,
                           in gondroid.py's exact wording; files are kept (originals + index.json) and
                           their text written in parts. A group is trusted when the owner
                           (GATE_APPROVER_TELEGRAM_ID) is a member of it; a DM only from an answerer.
  TELEGRAM_ANSWER_ONLY     comma-separated Telegram user ids that may get an answer, the owner always
                           included. With it set, a message from anyone else is recorded and never
                           routed, and nothing is ever sent to a chat that is not an answerer's DM or a
                           trusted group (the send guard in telegram.py). A group message is answered
                           only when it @mentions the bot's handle (gondroid.py's rule); an edit is
                           recorded, never answered.
  LEDGER_FILES=1           "send me the <name>" from an answerer posts the stored file in that chat.
  LEDGER_INBOX=1           other bots' messages and files, which Telegram never delivers to a bot, are
                           read from <state>/inbox where deploy/gondroid/bridge.py puts them.
  LEDGER_STATE_DIR         where trust.json, files/, transcripts/ and inbox/ live
                           (default $INKBOX_CLAUDE_HOME/ledger).
  TELEGRAM_SHADOW=1        shadow: everything runs, but nothing is written to the brain and nothing is
                           sent to Telegram; both are appended to $INKBOX_CLAUDE_HOME/shadow/{brain,telegram}.jsonl
                           so a run can be compared with the live deployment it shadows.

Only the answer is new: gondroid.py's own `compose` (one Claude call over brain search and the
chat transcript) is replaced by the gate (router, runs, follow-ups, steering, outcomes), so every
Blatbot upgrade reaches Gondroid. The team brain is the gate's long-term memory there too
(MEMORY_BACKEND=do_brain in gate/graphmem.py).
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set

from . import telegram

logger = logging.getLogger(__name__)

_LIB_PATH = Path(__file__).resolve().parents[1] / "deploy" / "gondroid" / "gondroid.py"
_lib: Any = None
_tasks: Set["asyncio.Task[Any]"] = set()


def _flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _flag("TELEGRAM_LEDGER") and telegram.enabled()


def files_enabled() -> bool:
    return enabled() and _flag("LEDGER_FILES")


def inbox_enabled() -> bool:
    return enabled() and _flag("LEDGER_INBOX")


def shadow() -> bool:
    return telegram.shadow()


def answerers() -> Set[str]:
    """Who may get an answer: TELEGRAM_ANSWER_ONLY plus the owner. Empty = no restriction."""
    ids = {x.strip() for x in (os.getenv("TELEGRAM_ANSWER_ONLY") or "").split(",") if x.strip()}
    if ids and telegram.approver_id():
        ids.add(telegram.approver_id())
    return ids


def state_dir() -> str:
    d = (os.getenv("LEDGER_STATE_DIR") or "").strip()
    if not d:
        d = os.path.join(os.getenv("INKBOX_CLAUDE_HOME") or os.path.expanduser("~/.inkbox-claude"), "ledger")
    return os.path.expanduser(d)


def _shadow_write(kind: str, row: Dict[str, Any]) -> None:
    telegram.shadow_record(kind, row)


def lib() -> Any:
    """gondroid.py as a library, configured for this instance. Loaded once."""
    global _lib
    if _lib is not None:
        return _lib
    spec = importlib.util.spec_from_file_location("gondroid_ledger", str(_LIB_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("deploy/gondroid/gondroid.py is missing")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    st = state_dir()
    mod.TOKEN = telegram.token()
    mod.OWNER = telegram.approver_id()
    mod.ALLOWED = answerers() or ({mod.OWNER} if mod.OWNER else set())
    mod.STATE = os.path.join(st, "trust.json")
    mod.FILES_DIR = os.path.join(st, "files")
    mod.FILES_INDEX = os.path.join(mod.FILES_DIR, "index.json")
    mod.TX_DIR = os.path.join(st, "transcripts")
    mod.INBOX = os.path.join(st, "inbox")
    for d in (st, mod.FILES_DIR, mod.TX_DIR, mod.INBOX):
        os.makedirs(d, exist_ok=True)
    real_tg = mod.tg

    async def tg(method: str, payload: Dict[str, Any], timeout: float = 20.0) -> Any:
        if not telegram.is_read_method(method):
            if shadow():
                _shadow_write("telegram", {"method": method, "payload": payload})
                return {"message_id": 0, "shadow": True}
            telegram.check_send(str(payload.get("chat_id") or ""), method)
        return await real_tg(method, payload, timeout)

    mod.tg = tg
    real_send_document = mod.send_document

    async def send_document(to_chat: str, stored: str, reply_to: Optional[int] = None) -> None:
        if shadow():
            _shadow_write("telegram", {"method": "sendDocument", "payload": {"chat_id": to_chat, "stored": stored,
                                                                              "reply_to": reply_to}})
            return
        telegram.check_send(str(to_chat), "sendDocument")
        await real_send_document(to_chat, stored, reply_to)

    mod.send_document = send_document
    real_brain_add = mod.brain_add

    async def brain_add(content: str, title: str) -> None:
        if shadow():
            _shadow_write("brain", {"title": title[:120], "content": content})
            return
        await real_brain_add(content, title)

    mod.brain_add = brain_add
    _lib = mod
    return mod


def is_trusted_group_cached(chat_id: str) -> bool:
    """The trust a group last got from getChatMember, without asking Telegram again."""
    try:
        g = lib().load()["groups"].get(str(chat_id))
    except Exception:  # noqa: BLE001
        return False
    return bool(g and g.get("ok"))


def may_send_to(chat_id: str) -> bool:
    """The send guard: an answerer's own DM, or a trusted group."""
    chat_id = str(chat_id or "")
    return chat_id in answerers() or (chat_id.startswith("-") and is_trusted_group_cached(chat_id))


async def start() -> None:
    """Know the bot's own handle (for @mentions and its own files), install the send guard,
    and start the bridge inbox when it is on."""
    mod = lib()
    try:
        mod.ME.update(await telegram.get_me())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ledger] getMe failed: %s", exc)
    if answerers():
        telegram.set_send_guard(may_send_to)
    logger.info("[ledger] on as @%s; answerers: %d; files: %s; bridge inbox: %s; shadow: %s; state %s",
                mod.ME.get("username"), len(answerers()), files_enabled(), inbox_enabled(), shadow(), state_dir())
    if inbox_enabled():
        _spawn(mod.inbox_loop(), "ledger-inbox")


def _spawn(coro: Any, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _tasks.add(task)

    def done(t: "asyncio.Task[Any]") -> None:
        _tasks.discard(t)
        if not t.cancelled() and t.exception():
            logger.error("[ledger] %s failed: %r", name, t.exception())
    task.add_done_callback(done)


async def take(update: Dict[str, Any]) -> Dict[str, Any]:
    """Record one Telegram update the way gondroid.py's handle() does (brain entry, transcript
    row, file kept and read, a replied-to bot message taken in), in the background. Returns at once
    with what the gateway needs to decide whether the message may go on to be answered:
    {"trusted", "answerable", "is_group", "file_request"}."""
    mod = lib()
    msg = update.get("message") or update.get("edited_message")
    out = {"trusted": False, "answerable": False, "is_group": False, "file_request": False}
    if not msg:
        return out
    chat, frm = msg["chat"], msg.get("from") or {}
    chat_id, from_id = str(chat["id"]), str(frm.get("id") or "")
    is_group = chat["type"] in ("group", "supergroup")
    out["is_group"] = is_group
    if is_group:
        if not await mod.group_trusted(chat_id):
            logger.info("[ledger] group %s untrusted; ignored", chat_id)
            return out
    elif from_id not in mod.ALLOWED:
        logger.info("[ledger] DM from someone not allowed; ignored")
        return out
    out["trusted"] = True
    text = (msg.get("text") or msg.get("caption") or "").strip()
    doc = mod.doc_of(msg)
    if not doc:
        pass
    elif (doc.get("mime_type") or "").startswith(("audio/", "video/")) or (doc.get("file_name") or "").lower().endswith(mod.MEDIA_EXT):
        doc = None
    if doc and not text:
        text = f'[sent the file "{doc.get("file_name") or "file"}"]'
    if not text:
        return out
    name = " ".join(x for x in (frm.get("first_name"), frm.get("last_name")) if x) or frm.get("username") or from_id
    from datetime import datetime, timezone
    when = datetime.fromtimestamp(msg.get("edit_date") or msg["date"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    who = name + (f" (@{frm['username']})" if frm.get("username") else "")
    where = f'Telegram group "{chat.get("title") or chat_id}"' if is_group else "a Telegram DM with Gondroid"
    edited = "edited_message" in update
    title = chat.get("title") or "DM"
    await mod.ingest_bot_reply(msg, chat_id, where, title)
    mod.tx_append(chat_id, {"id": msg["message_id"], "ts": when, "who": name, "from_id": from_id, "text": text[:4000],
                            "reply_to": (msg.get("reply_to_message") or {}).get("message_id"),
                            "file": (doc or {}).get("file_name"), "edited": edited})

    async def record() -> None:
        try:
            await mod.brain_add(f"On {when} in {where}, {who} {'edited a message to say' if edited else 'said'}: {text}",
                                f"Telegram {title}: {name} {when}")
            logger.info("[ledger] recorded msg %s in %s", msg["message_id"], chat_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("[ledger] brain add failed (msg %s in %s): %s", msg["message_id"], chat_id, exc)

    async def record_doc() -> None:
        sink: Dict[str, str] = {}
        try:
            logger.info("[ledger] %s (msg %s in %s)", await mod.record_file(doc, who, where, when, title, None, sink),
                        msg["message_id"], chat_id)
            if sink.get("text"):
                mod.tx_set_file_text(chat_id, msg["message_id"], sink["text"])
        except Exception as exc:  # noqa: BLE001
            logger.error("[ledger] file read failed (msg %s in %s): %s", msg["message_id"], chat_id, type(exc).__name__)

    _spawn(record(), f"ledger-record-{chat_id}")
    if doc and not edited:
        _spawn(record_doc(), f"ledger-file-{chat_id}")
    if not edited and from_id in mod.ALLOWED and mod.addressed(msg, text):
        out["answerable"] = True
        q = re.sub(r"@" + re.escape(mod.ME.get("username") or "~"), "", text, flags=re.I).strip(" ,:")
        out["file_request"] = files_enabled() and bool(q) and mod.wants_file(q)
    return out


async def file_request(update: Dict[str, Any]) -> bool:
    """gondroid.py's "send me the <name>": True when it answered (the file, a "which one?" list,
    or "I don't have a stored file that matches that"), False to let the gate answer instead."""
    mod = lib()
    msg = update.get("message") or {}
    chat_id, from_id = str(msg["chat"]["id"]), str((msg.get("from") or {}).get("id") or "")
    is_group = msg["chat"]["type"] in ("group", "supergroup")
    text = (msg.get("text") or msg.get("caption") or "").strip()
    q = re.sub(r"@" + re.escape(mod.ME.get("username") or "~"), "", text, flags=re.I).strip(" ,:")
    try:
        return bool(await mod.file_request(q, chat_id, from_id, is_group, msg["message_id"]))
    except Exception as exc:  # noqa: BLE001
        logger.error("[ledger] file request failed: %s", exc)
        return False
