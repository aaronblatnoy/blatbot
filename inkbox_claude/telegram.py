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

import logging
import os
import re
import secrets
from typing import Any, Dict, Optional, Tuple

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


async def _call(method: str, payload: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
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


def parse_update(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What the gateway needs from one message update, or None when there is nothing
    to act on. In a group the bot now sees every message (privacy mode off), so the
    dict says whether this one was actually addressed to it."""
    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    frm = msg.get("from") or {}
    text = str(msg.get("text") or msg.get("caption") or "").strip()
    chat_id = str(chat.get("id") or "")
    if not chat_id or not text:
        return None
    name = " ".join(str(frm.get(k) or "").strip() for k in ("first_name", "last_name")).strip()
    if not name:
        name = str(frm.get("username") or "")
    kind = str(chat.get("type") or "private")
    is_group = kind in ("group", "supergroup")
    addressed = not is_group or _names_the_bot(text, msg)
    return {"chat_id": chat_id, "from_id": str(frm.get("id") or ""), "name": name, "text": text,
            "is_group": is_group, "group_title": str(chat.get("title") or ""), "addressed": addressed}
