"""Inkbox-native WhatsApp for Blatbot.

Ported from upstream inkbox-ai/claude-code-plugin (#84, #85), cut down to what
Blatbot's gate needs. Upstream keeps its own channel stores, Companion grants and
quote-recovery journals; Blatbot routes every inbound through the gate instead
(gateway._inbox_or_direct), the same way iMessage arrives, so those layers are
left out. What is kept from upstream: the event names, the shape checks on an
incoming message, mention rendering, refreshed media, and the send call with a
stable idempotency key so a retried reply never goes out twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

EVENTS = [
    "whatsapp.received",
    "whatsapp.sent",
    "whatsapp.delivered",
    "whatsapp.delivery_failed",
    "whatsapp.delivery_unconfirmed",
    "whatsapp.reaction_received",
]

MAX_LENGTH = 4000
_E164 = re.compile(r"\+[1-9][0-9]{6,14}")

logger = logging.getLogger(__name__)


def field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def unsupported(exc) -> bool:
    """An older API may reject the WhatsApp subscription family without blocking other channels."""
    detail = str(getattr(exc, "detail", exc)).lower()
    return (
        (isinstance(exc, ValueError) or getattr(exc, "status_code", None) in {400, 422})
        and "whatsapp" in detail
        and any(word in detail for word in ("event", "prefix", "unknown", "unsupported"))
    )


def require_method(identity, name):
    method = getattr(identity, name, None)
    if not callable(method):
        raise RuntimeError("WhatsApp requires Inkbox SDK 0.7.29 or newer")
    return method


def is_group(value) -> bool:
    return field(value, "is_group") is True


def enabled(identity) -> bool:
    return bool(getattr(identity, "whatsapp_enabled", False))


def _valid_numbers(numbers) -> bool:
    return isinstance(numbers, list) and bool(numbers) and all(
        isinstance(n, str) and _E164.fullmatch(n) for n in numbers)


async def render_mentions(gateway, text: str, mentions) -> str:
    """Replace @<digits> mention tokens with the person's name (or number)."""
    replacements: Dict[str, str] = {}
    for item in mentions or []:
        if not isinstance(item, dict):
            continue
        token = item.get("text")
        if not isinstance(token, str) or not re.fullmatch(r"@[0-9]+", token):
            continue
        if item.get("is_agent") is True:
            replacements[token] = "@" + str(gateway.cfg.identity)
            continue
        phone = item.get("phone_number")
        if not isinstance(phone, str) or not _E164.fullmatch(phone):
            continue
        name = ""
        try:
            contact = await gateway._resolve_contact_full(kind="phone", value=phone)
            name = str(field(contact, "name") or "") if contact else ""
        except Exception:
            pass
        replacements[token] = "@" + (" ".join(name.split())[:80] or phone)
    for token in sorted(replacements, key=len, reverse=True):
        text = re.sub(re.escape(token) + r"(?![0-9])", replacements[token], text)
    return text


async def with_media(gateway, message: Dict[str, Any], text: str) -> str:
    """Refresh each attachment's temporary URL and download it like other channels."""
    media = message.get("media") or []
    if not media:
        return text
    refreshed = []
    for item in media:
        media_id = item.get("id") if isinstance(item, dict) else None
        if not media_id or item.get("status") == "unavailable":
            text += f"\n[WhatsApp attachment {media_id or 'unknown'} unavailable]"
            continue
        try:
            fresh = await asyncio.to_thread(require_method(gateway._identity, "get_whatsapp_media"), media_id)
        except Exception:
            text += f"\n[WhatsApp attachment {media_id} unavailable; do not guess its contents]"
            continue
        if field(fresh, "status") == "ready" and field(fresh, "url"):
            refreshed.append({"url": field(fresh, "url"), "content_type": field(fresh, "content_type"),
                              "size": field(fresh, "size")})
        else:
            text += f"\n[WhatsApp attachment {media_id} unavailable]"
    if not refreshed:
        return text
    return await gateway._with_media(text, refreshed, prefix=f"whatsapp-{message.get('id', '')}")


def group_prompt(body: str, *, sender: str, conversation_id: str, participants: List[str],
                 group_name: str, contact_marker_text: str) -> str:
    marker = " ".join(part for part in [
        f"[inkbox:group_whatsapp conversation_id={conversation_id}",
        f"from={sender}",
        f"group={group_name!r}" if group_name else None,
        f"participants={','.join(participants)}" if participants else None,
        "reply_mode=conversation_id",
        f"| {contact_marker_text}]",
    ] if part)
    policy = "\n".join([
        "Group WhatsApp response policy: you receive every message in this group so you can track context.",
        "Reply only when the latest message clearly addresses this Inkbox agent, asks it to act, or a visible answer would be expected from the agent.",
        "Treat ordinary group chatter as context only.",
        "If no visible reply is warranted, return exactly [SILENT].",
    ])
    return "\n".join(part for part in [marker, policy, body] if part)


def mode_blocks(gateway, chat_id: str, text: str, *, mentioned: bool) -> bool:
    """The room's response mode, as for iMessage groups, except that a native
    WhatsApp @mention of Blatbot counts as being addressed (silent still wins)."""
    if not gateway._group_mode_blocks_reply(chat_id, text):
        return False
    if not mentioned:
        return True
    try:
        from .gate import roommode
        store = gateway.sessions.store
        return roommode.effective_mode(store.get_response_mode(chat_id)) == "silent"
    except Exception:
        return True


async def receive(gateway, envelope: Dict[str, Any]):
    from aiohttp import web

    message = (envelope.get("data") or {}).get("message") or {}
    message_id = str(message.get("id") or "").strip()
    key = f"whatsapp:{message_id}" if message_id else ""
    if gateway._dedup_begin(key):
        return web.json_response({"ok": True, "deduped": True})
    try:
        response = await _receive_once(gateway, envelope)
    except Exception:
        gateway._dedup_rollback(key)
        raise
    gateway._dedup_commit(key)
    return response


async def _receive_once(gateway, envelope: Dict[str, Any]):
    from aiohttp import web
    from .prompts import contact_marker

    # A message can arrive in the first second after start, before the sessions
    # exist. Wait for them briefly; if they never come, answer 503 so the delivery
    # is retried instead of failing with an error and losing the message.
    for _ in range(60):
        if gateway.sessions is not None:
            break
        await asyncio.sleep(0.25)
    else:
        logger.warning("[whatsapp] message arrived before startup finished; asked for a retry")
        return web.json_response({"ok": False, "error": "starting"}, status=503)

    data = envelope.get("data") or {}
    message = data.get("message") or {}
    message_id = str(message.get("id") or "").strip()
    conversation_id = str(message.get("conversation_id") or "").strip()
    sender = str(message.get("sender_number") or "").strip()
    participants = message.get("participants") or []
    group = is_group(message)
    identity = gateway._identity
    if (
        not message_id or not conversation_id
        or message.get("direction") != "inbound"
        or str(message.get("agent_identity_id") or "") != str(field(identity, "id", ""))
        or not _valid_numbers(participants)
        or sender not in participants
        or sender == message.get("local_number")
        or (not group and (len(participants) != 1 or sender != message.get("remote_number")))
    ):
        return web.json_response({"ok": True, "ignored": "unsupported-message"})
    if not gateway._sender_allowed(sender):
        return web.json_response({"ok": True, "ignored": "sender-not-allowed"})
    raw = str(message.get("content") or "")
    media = message.get("media") or []
    if not raw.strip() and not media:
        return web.json_response({"ok": True, "ignored": "empty"})

    text = (await render_mentions(gateway, raw, message.get("mentions"))).strip()
    body = await with_media(gateway, message, text)
    if message.get("reply_to_message_id"):
        body += f"\n[replying to WhatsApp message {message['reply_to_message_id']}]"
    contact = await gateway._resolve_contact_full(kind="phone", value=sender)
    memories = gateway._webhook_contact_memories(
        data, resolved_contact_id=(contact or {}).get("id", ""), allow_sole_fallback=True,
    ) if gateway.cfg.contact_memories_enabled and not group else []
    agent_identity = None if (contact or group) else gateway._single_agent_identity(
        gateway._webhook_list(data, "agent_identities", "agentIdentities", "identity_agents"))
    group_name = str(message.get("group_name") or "").strip()
    if group:
        body = group_prompt(body, sender=sender, conversation_id=conversation_id,
                            participants=participants, group_name=group_name,
                            contact_marker_text=contact_marker(contact))
    thread_key = gateway._thread_key("whatsapp", conversation_id)
    # A group is one shared context for everyone in it, so the conversation is
    # the chat; a direct chat keeps its per-contact chat like iMessage.
    chat_id = thread_key if group else gateway._chat_key(
        data, sender, thread_key, contact=contact, allow_webhook_contact=False)
    meta: Dict[str, Any] = {
        "conversation_id": conversation_id,
        "sender": sender,
        "contact": contact,
        "agent_identity": agent_identity,
        "contact_memories": memories,
        "conversation_kind": "group" if group else "direct",
        "message_id": message_id,
        "participants": [str(n) for n in participants] if group else [],
    }
    if group:
        if group_name:
            meta["group_title"] = group_name
        if mode_blocks(gateway, chat_id, text, mentioned=message.get("agent_mentioned") is True):
            gateway._overhear_group_text(chat_id, "whatsapp", sender, text)
            return web.json_response({"ok": True, "mode": "blocked"})
    session = gateway.sessions.get(chat_id)
    gateway._set_room(session, chat_id=chat_id, channel="WhatsApp", is_group=group,
                      title=group_name, members=list(participants) if group else [])
    if group and gateway._inbox_enabled():
        meta["_inbox_members"] = list(participants)
    await gateway._inbox_or_direct(chat_id, "whatsapp", message_id, "whatsapp", body, meta)
    return web.json_response({"ok": True})


def send_key(identity_id: str, target: str, source: str, text: str) -> str:
    raw = json.dumps([identity_id, target, source, text], sort_keys=True).encode()
    return "blatbot-wa-" + hashlib.sha256(raw).hexdigest()


async def send(gateway, chat_id: str, content: str, meta: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Send a reply (or a tool's message) on WhatsApp. Returns {"id"} for tool sends."""
    from .prompts import strip_markdown

    text = content if meta.get("_inkbox_tool") else strip_markdown(content)
    if len(text) > MAX_LENGTH:
        raise ValueError(f"WhatsApp text must be at most {MAX_LENGTH} characters (got {len(text)})")
    conversation_id = str(meta.get("conversation_id") or "").strip()
    if not conversation_id and not meta.get("_inkbox_tool") and str(chat_id).startswith("whatsapp:"):
        conversation_id = str(chat_id).split(":", 1)[1]
    recipient = str(meta.get("whatsapp_to") or "").strip()
    if not conversation_id and not recipient:
        raise ValueError(f"No WhatsApp conversation id for chat {chat_id}")
    identity = gateway._identity
    kwargs: Dict[str, Any] = {"text": text}
    if conversation_id:
        kwargs["conversation_id"] = conversation_id
    else:
        kwargs["to"] = recipient
    if meta.get("whatsapp_media_ids"):
        kwargs["media_ids"] = list(meta["whatsapp_media_ids"])
    # A reply to one inbound message has a stable key, so a retried send is the
    # same send; a tool message without a source gets a fresh one.
    source = str(meta.get("outbox_id") or meta.get("message_id") or uuid.uuid4().hex)
    kwargs["idempotency_key"] = send_key(str(field(identity, "id", "")), conversation_id or recipient,
                                         source, text)
    msg = await asyncio.to_thread(require_method(identity, "send_whatsapp"), **kwargs)
    return {"id": str(field(msg, "id", ""))} if meta.get("_inkbox_tool") else None


def record_status(gateway, envelope: Dict[str, Any]) -> None:
    """Delivery notifications are logged only; they never start model work."""
    message = (envelope.get("data") or {}).get("message") or {}
    if envelope.get("event_type") == "whatsapp.delivery_failed":
        logger.warning("[bridge] WhatsApp message %s was not delivered (%s)",
                       message.get("id"), message.get("delivery_state") or message.get("status"))
    else:
        logger.debug("[bridge] WhatsApp %s for %s", envelope.get("event_type"), message.get("id"))
