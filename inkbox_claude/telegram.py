"""Telegram as a Blatbot channel.

A Telegram **bot** (created with BotFather) is the counterpart here, not Aaron's
own account: people message @<bot>, Telegram POSTs each update to this gateway's
``/telegram`` route, and replies go back through the Bot API. The same gate runs
as for iMessage or email, so a stranger's request still waits for Aaron's yes.

Configuration (environment):
  TELEGRAM_BOT_TOKEN        from BotFather; absent means the channel is off
  TELEGRAM_WEBHOOK_SECRET   optional; generated and stored if unset. Telegram
                            echoes it in X-Telegram-Bot-Api-Secret-Token, which
                            is what proves an update really came from Telegram
  GATE_APPROVER_TELEGRAM_ID Aaron's numeric Telegram user id (from @userinfobot)
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from typing import Any, Callable, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

API = "https://api.telegram.org"
MAX_LENGTH = 4096           # Telegram's own per-message cap
SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"
WEBHOOK_PATH = "/telegram"


def token() -> str:
    return (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()


def enabled() -> bool:
    return bool(token())


def approver_id() -> str:
    return str(os.getenv("GATE_APPROVER_TELEGRAM_ID") or "").strip()


def webhook_secret() -> str:
    """The shared secret Telegram echoes back on every update. Generated once
    and kept in the environment for this process if it was not configured."""
    s = (os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip()
    if not s:
        s = secrets.token_urlsafe(24)
        os.environ["TELEGRAM_WEBHOOK_SECRET"] = s
        logger.info("[telegram] generated a webhook secret for this run; set TELEGRAM_WEBHOOK_SECRET to pin it")
    return s


# ---- shadow mode and the send guard ----------------------------------------------------------
# TELEGRAM_SHADOW=1: every Bot API call that would change anything in Telegram (send, react,
# typing, webhook) is appended to <INKBOX_CLAUDE_HOME>/shadow/telegram.jsonl instead of made, so an
# instance can run beside the live one on the same traffic and be compared. Reads still go out.
# A send guard (set by teamledger for an instance that answers only some people) is asked before
# any message, photo, file or reaction goes to a chat; it raises for a chat it does not allow.
_READ_PREFIXES = ("get",)
_send_guard: Optional[Callable[[str], bool]] = None


def shadow() -> bool:
    return (os.getenv("TELEGRAM_SHADOW") or "").strip().lower() in {"1", "true", "yes", "on"}


def is_read_method(method: str) -> bool:
    return method.startswith(_READ_PREFIXES)


def shadow_record(kind: str, row: Dict[str, Any], base: str = "") -> None:
    base = base or os.path.join(os.path.expanduser(os.getenv("INKBOX_CLAUDE_HOME") or "~/.inkbox-claude"), "shadow")
    try:
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, f"{kind}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.time(), **row}, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("[telegram] shadow record failed: %s", exc)


def set_send_guard(fn: Optional[Callable[[str], bool]]) -> None:
    global _send_guard
    _send_guard = fn


class SendRefused(RuntimeError):
    pass


def check_send(chat_id: str, method: str) -> None:
    """Raise SendRefused when the guard does not allow this chat."""
    if _send_guard is not None and method not in ("deleteWebhook", "setWebhook") and not _send_guard(str(chat_id)):
        logger.warning("[telegram] %s to chat %s refused: not an allowed chat for this instance", method, chat_id)
        raise SendRefused(f"telegram {method} to chat {chat_id} refused for this instance")


def _intercept(method: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Shadow or guard a write call. Returns a stand-in result when the call must not be made."""
    if is_read_method(method):
        return None
    if shadow():
        shadow_record("telegram", {"method": method, "payload": payload})
        return {"message_id": 0, "shadow": True}
    check_send(str(payload.get("chat_id") or ""), method)
    return None


async def _call(method: str, payload: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    stand_in = _intercept(method, payload)
    if stand_in is not None:
        return stand_in
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{API}/bot{token()}/{method}", json=payload)
        r.raise_for_status()
        data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"telegram {method} failed: {str(data)[:200]}")
    return data.get("result") or {}


async def send_message(chat_id: str, text: str) -> None:
    """One message to a chat, split at Telegram's length cap on line breaks."""
    text = (text or "").strip()
    if not text:
        return
    chunks = []
    while len(text) > MAX_LENGTH:
        cut = text.rfind("\n", 0, MAX_LENGTH)
        cut = cut if cut > MAX_LENGTH // 2 else MAX_LENGTH
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    chunks.append(text)
    for part in chunks:
        await _call("sendMessage", {"chat_id": chat_id, "text": part, "disable_web_page_preview": True})


async def send_photo(chat_id: str, path: str, caption: str = "") -> None:
    """One local image as a Telegram photo (a document when it is too big or too tall for a photo)."""
    from pathlib import Path as _P
    p = _P(path)
    if _intercept("sendPhoto", {"chat_id": str(chat_id), "path": str(path), "caption": caption}) is not None:
        return
    data = p.read_bytes()
    form = {"chat_id": str(chat_id)}
    if caption:
        form["caption"] = caption[:1000]
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{API}/bot{token()}/sendPhoto", data=form, files={"photo": (p.name, data)})
        if r.status_code == 400 or len(data) > 10 * 1024 * 1024:
            r = await client.post(f"{API}/bot{token()}/sendDocument", data=form, files={"document": (p.name, data)})
        r.raise_for_status()
        if not r.json().get("ok"):
            raise RuntimeError(f"telegram sendPhoto failed: {str(r.json())[:200]}")


# ---- emoji reactions -------------------------------------------------------------------------
# Telegram lets a bot react with these (the free set). Compared without the emoji variation selector.
REACTIONS = set(
    "\U0001F44D \U0001F44E \u2764 \U0001F525 \U0001F970 \U0001F44F \U0001F601 \U0001F914 \U0001F92F \U0001F631 "
    "\U0001F622 \U0001F389 \U0001F929 \U0001F64F \U0001F44C \U0001F54A \U0001F921 \U0001F60D \U0001F433 \U0001F4AF "
    "\U0001F923 \u26A1 \U0001F3C6 \U0001F494 \U0001F928 \U0001F610 \U0001F4CB \U0001F60E \U0001F440 \U0001F648 "
    "\U0001F607 \U0001F91D \U0001FAE1 \U0001F5FF \U0001F192 \U0001F984 \U0001F618 \U0001F917 \U0001F913 \U0001F47B "
    "\U0001F62D \U0001F4A9 \U0001F91D".split()
) - {"\U0001F4CB"}
INBOUND: Dict[str, str] = {}      # chat id -> message id of the latest message people sent there


def remember_inbound(chat_id: str, message_id: str) -> None:
    if chat_id and message_id:
        INBOUND[str(chat_id)] = str(message_id)
        if len(INBOUND) > 2000:
            INBOUND.pop(next(iter(INBOUND)))


def clean_reaction(emoji: str) -> str:
    """The emoji Telegram accepts, or raises ValueError listing what is allowed."""
    e = (emoji or "").strip().replace("\ufe0f", "")
    if e not in REACTIONS:
        raise ValueError("Telegram only allows these reactions: " + " ".join(sorted(REACTIONS)))
    return e


async def react(chat_id: str, message_id: str, emoji: str) -> None:
    """Put an emoji reaction on one message (setMessageReaction); an empty emoji clears it."""
    reaction = [{"type": "emoji", "emoji": clean_reaction(emoji)}] if emoji else []
    if _intercept("setMessageReaction", {"chat_id": str(chat_id), "message_id": message_id, "reaction": reaction}) is not None:
        return
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(f"{API}/bot{token()}/setMessageReaction",
                              json={"chat_id": str(chat_id), "message_id": int(message_id), "reaction": reaction})
        if r.status_code >= 400 or not r.json().get("ok"):
            raise RuntimeError(f"telegram setMessageReaction failed: {r.text[:200]}")


async def send_typing(chat_id: str) -> None:
    """The "typing..." indicator; it lapses after ~5 s, so it is re-sent."""
    await _call("sendChatAction", {"chat_id": chat_id, "action": "typing"}, timeout=10.0)


async def remember_username(me: Dict[str, Any]) -> None:
    """Keep the bot's own @name, so a group mention can be recognised."""
    if me.get("username"):
        os.environ["TELEGRAM_BOT_USERNAME"] = str(me["username"])
    if me.get("first_name"):
        os.environ["TELEGRAM_BOT_NAME"] = str(me["first_name"])


async def set_webhook(public_url: str) -> Dict[str, Any]:
    """Point Telegram at this gateway. Called at startup once the tunnel is up."""
    url = public_url.rstrip("/") + WEBHOOK_PATH
    await _call("setWebhook", {"url": url, "secret_token": webhook_secret(),
                               "allowed_updates": ["message", "edited_message"],
                               "drop_pending_updates": False})
    me = await _call("getMe", {})
    await remember_username(me)
    logger.info("[telegram] webhook set: %s -> @%s", url, me.get("username") or "?")
    return me


async def get_me() -> Dict[str, Any]:
    """The bot's own identity; also kept for @mention matching (remember_username)."""
    me = await _call("getMe", {})
    await remember_username(me)
    return me


def poll_enabled() -> bool:
    """TELEGRAM_POLL=1: take updates by long polling (getUpdates) instead of a webhook, for an
    instance with no public URL. Never in shadow mode: one poller per bot token at a time, and
    the token belongs to the deployment being shadowed."""
    return enabled() and not shadow() and (os.getenv("TELEGRAM_POLL") or "").strip().lower() in {"1", "true", "yes", "on"}


async def get_updates(offset: int, timeout_s: int = 50) -> list:
    async with httpx.AsyncClient(timeout=timeout_s + 20) as client:
        r = await client.post(f"{API}/bot{token()}/getUpdates",
                              json={"offset": offset, "timeout": timeout_s,
                                    "allowed_updates": ["message", "edited_message"]})
        data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"telegram getUpdates failed: {str(data.get('description'))[:200]}")
    return list(data.get("result") or [])


def _names_the_bot(text: str, msg: Dict[str, Any]) -> bool:
    """Was this group message aimed at the bot? Its @username, its plain name as a word
    (people type "blatbot, do x", not the handle), a slash command, or a reply to something
    the bot itself said."""
    low = text.lower()
    handle = (os.getenv("TELEGRAM_BOT_USERNAME") or "").lstrip("@").lower()
    if handle and f"@{handle}" in low:
        return True
    if text.startswith("/"):
        return True
    reply_to = (msg.get("reply_to_message") or {}).get("from") or {}
    if reply_to.get("is_bot") and str(reply_to.get("username") or "").lower() == handle:
        return True
    for name in (os.getenv("TELEGRAM_BOT_NAME") or "", handle):
        name = name.strip().lower()
        if name and re.search(r"\b" + re.escape(name) + r"\b", low):
            return True
    return False


_MENTION = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9_]{2,31})")
_SLASH_TARGET = re.compile(r"(?<!\w)/\w{1,32}@([A-Za-z][A-Za-z0-9_]{2,31})")


def names_blatbot(text: str) -> bool:
    """The text itself calls Blatbot: its @username, or its plain name as a word. A reply
    to one of Blatbot's messages or a slash command is not a call by name."""
    low = text.lower()
    handle = (os.getenv("TELEGRAM_BOT_USERNAME") or "").lstrip("@").lower()
    if handle and f"@{handle}" in low:
        return True
    for name in (os.getenv("TELEGRAM_BOT_NAME") or "", handle):
        name = name.strip().lower()
        if name and re.search(r"\b" + re.escape(name) + r"\b", low):
            return True
    return False


def for_other_bot(text: str) -> bool:
    """Is this message plainly aimed at a different bot? It @mentions another bot's handle
    (Telegram bot usernames always end in "bot") and never calls Blatbot by handle or name.
    Deterministic on purpose: it must hold when every judgment service is down. It holds
    even when the message is a reply to something Blatbot said, because typing another
    bot's @handle in that reply is the person choosing who to talk to."""
    handle = (os.getenv("TELEGRAM_BOT_USERNAME") or "").lstrip("@").lower()
    others = [m for m in _MENTION.findall(text or "") + _SLASH_TARGET.findall(text or "")
              if m.lower().endswith("bot") and m.lower() != handle]
    return bool(others) and not names_blatbot(text)


_MIME_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/heic": ".heic",
             "image/gif": ".gif", "application/pdf": ".pdf", "text/csv": ".csv", "text/plain": ".txt"}


def media_of(msg: Dict[str, Any]) -> list:
    """The readable files on one message: its photo (largest size) and/or document."""
    out = []
    photos = [p for p in (msg.get("photo") or []) if isinstance(p, dict) and p.get("file_id")]
    if photos:
        best = max(photos, key=lambda p: int(p.get("width") or 0) * int(p.get("height") or 0))
        out.append({"kind": "photo", "file_id": best["file_id"],
                    "file_unique_id": best.get("file_unique_id") or best["file_id"],
                    "name": "photo.jpg", "size": int(best.get("file_size") or 0)})
    doc = msg.get("document")
    if isinstance(doc, dict) and doc.get("file_id"):
        name = str(doc.get("file_name") or "file")
        if "." not in name:
            name += _MIME_EXT.get(str(doc.get("mime_type") or ""), "")
        out.append({"kind": "file", "file_id": doc["file_id"],
                    "file_unique_id": doc.get("file_unique_id") or doc["file_id"],
                    "name": name, "size": int(doc.get("file_size") or 0)})
    return out


def _display(frm: Dict[str, Any]) -> str:
    name = " ".join(str(frm.get(k) or "").strip() for k in ("first_name", "last_name")).strip()
    return name or str(frm.get("username") or "")


def reply_of(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The message this one replies to, as Telegram attached it (text, caption, files)."""
    r = msg.get("reply_to_message")
    if not isinstance(r, dict):
        return None
    frm = r.get("from") or {}
    return {"message_id": str(r.get("message_id") or ""), "name": _display(frm),
            "from_id": str(frm.get("id") or ""), "username": str(frm.get("username") or ""),
            "is_bot": bool(frm.get("is_bot")), "date": _unix(r.get("date")),
            "text": str(r.get("text") or r.get("caption") or "").strip(), "media": media_of(r)}


async def download(file_id: str) -> bytes:
    """One file's bytes through the Bot API (bots may download up to 20 MB). The URL
    carries the token, so it is never logged."""
    info = await _call("getFile", {"file_id": file_id})
    path = info.get("file_path")
    if not path:
        raise RuntimeError("telegram gave no file path")
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(f"{API}/file/bot{token()}/{path}")
        r.raise_for_status()
        return r.content


def parse_update(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What the gateway needs from one message update, or None when there is nothing
    to act on. In a group the bot now sees every message (privacy mode off), so the
    dict says whether this one was actually addressed to it. A photo or file with no
    caption is a message too: its `media` is read before anything acts on it."""
    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    frm = msg.get("from") or {}
    text = str(msg.get("text") or msg.get("caption") or "").strip()
    chat_id = str(chat.get("id") or "")
    media = media_of(msg)
    if not chat_id or (not text and not media):
        return None
    name = " ".join(str(frm.get(k) or "").strip() for k in ("first_name", "last_name")).strip()
    if not name:
        name = str(frm.get("username") or "")
    kind = str(chat.get("type") or "private")
    is_group = kind in ("group", "supergroup")
    other_bot = is_group and for_other_bot(text)
    addressed = not is_group or (not other_bot and _names_the_bot(text, msg))
    return {"update_id": str(update.get("update_id") or ""),
            "message_id": str(msg.get("message_id") or ""),
            "chat_id": chat_id, "from_id": str(frm.get("id") or ""), "name": name, "text": text,
            "is_group": is_group, "group_title": str(chat.get("title") or ""), "addressed": addressed,
            "for_other_bot": other_bot,
            "username": str(frm.get("username") or ""), "is_bot": bool(frm.get("is_bot")),
            "media": media, "reply_to": reply_of(msg),
            # Unix seconds the message was sent (or last edited); 0 when Telegram gave none.
            "date": _unix(msg.get("edit_date") or msg.get("date"))}


def _unix(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def parse_new_members(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The people just added to a group (Telegram's new_chat_members service message,
    which carries no text and so is not a parse_update message), or None when this
    update is not that."""
    msg = update.get("message") or {}
    joined = msg.get("new_chat_members") or []
    chat = msg.get("chat") or {}
    if not joined or not chat.get("id"):
        return None
    members = []
    for m in joined:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        name = " ".join(str(m.get(k) or "").strip() for k in ("first_name", "last_name")).strip()
        members.append({"id": str(m["id"]), "name": name or str(m.get("username") or ""),
                        "username": str(m.get("username") or ""), "is_bot": bool(m.get("is_bot"))})
    return {"chat_id": str(chat["id"]), "group_title": str(chat.get("title") or ""),
            "members": members}
