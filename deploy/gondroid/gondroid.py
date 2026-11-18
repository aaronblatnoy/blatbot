"""Gondroid: the Definite Optimists team's Telegram ledger. One file, two jobs.

1. Every text message in a trusted chat is written to the Definite Optimists brain
   (POST /api/add, Gondroid's own brain key) with who said it, where and when.
2. A message aimed at Gondroid (a DM, an @mention, its name, or a reply to it) is a question:
   the brain is searched and a short answer goes back to that same chat. Nothing else is sent.

The brain is Gondroid's only memory. The only local state is which chats and people are trusted:
  * a group is trusted when the owner (OWNER_TELEGRAM_ID) is a member of it (getChatMember);
  * a person is trusted when they are the owner or have written in a trusted group;
  * DMs are answered and recorded only for trusted people. Everything else is ignored.

Updates arrive by long polling (getUpdates), so no public URL is needed.

Environment: TELEGRAM_BOT_TOKEN, OWNER_TELEGRAM_ID, DO_BRAIN_KEY, DO_BRAIN_API_URL (optional),
CLAUDE_CODE_OAUTH_TOKEN + LEDGER_MODEL (answers; default "sonnet"), STATE_DIR (default /data).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gondroid")
# httpx logs every request URL at INFO, and Telegram URLs carry the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
OWNER = os.environ.get("OWNER_TELEGRAM_ID", "").strip()
BRAIN = (os.environ.get("DO_BRAIN_API_URL") or
         "https://definite-optimists-brain.aaronblatnoy.workers.dev").rstrip("/")
BRAIN_HEADERS = {"Authorization": "Bearer " + os.environ.get("DO_BRAIN_KEY", "").strip(),
                 "User-Agent": "gondroid", "Content-Type": "application/json"}
MODEL = os.environ.get("LEDGER_MODEL", "sonnet").strip()
STATE = os.path.join(os.environ.get("STATE_DIR", "/data"), "trust.json")
MEMBER = ("creator", "administrator", "member", "restricted")
ME: Dict[str, Any] = {}
RECENT: Dict[str, List[str]] = {}

SYSTEM = (
    "You are Gondroid, the Definite Optimists team's ledger. Your memory is the team brain, and the "
    "material in the message is everything you know. Answer the question from that material only. "
    "Be plain and short: a sentence or a few lines, no headers, no preamble, no em dashes. Say who "
    "said what and when when it helps. If the material does not hold the answer, say you don't have "
    "that recorded. Never invent anything.")


# ---------- Telegram ----------
async def tg(method: str, payload: Dict[str, Any], timeout: float = 20.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"https://api.telegram.org/bot{TOKEN}/{method}", json=payload)
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"{method}: {data.get('description')}")
    return data.get("result")


async def send(chat_id: str, text: str, reply_to: Optional[int] = None) -> None:
    for i in range(0, len(text), 4000):
        p: Dict[str, Any] = {"chat_id": chat_id, "text": text[i:i + 4000], "disable_web_page_preview": True}
        if reply_to and i == 0:
            p["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
        await tg("sendMessage", p)


# ---------- trust ----------
def load() -> Dict[str, Any]:
    try:
        d = json.load(open(STATE))
    except Exception:
        d = {}
    d.setdefault("groups", {})
    d.setdefault("people", {})
    return d


def save(d: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE + ".tmp", "w") as f:
        json.dump(d, f)
    os.replace(STATE + ".tmp", STATE)


async def group_trusted(chat_id: str) -> bool:
    d = load()
    g = d["groups"].get(chat_id)
    if g and (g["ok"] or time.time() - g["checked"] < 600):
        return g["ok"]
    ok = False
    try:
        m = await tg("getChatMember", {"chat_id": chat_id, "user_id": int(OWNER)})
        ok = m.get("status") in MEMBER
    except Exception as exc:
        log.info("getChatMember %s: %s", chat_id, exc)
    d = load()
    d["groups"][chat_id] = {"ok": ok, "checked": time.time()}
    save(d)
    return ok


# ---------- brain ----------
async def brain_add(content: str, title: str) -> None:
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(BRAIN + "/api/add", headers=BRAIN_HEADERS,
                         json={"content": content, "title": title[:120], "source": "telegram"})
        r.raise_for_status()


async def brain_context(question: str) -> str:
    lines = ["FACTS FROM THE BRAIN:"]
    async with httpx.AsyncClient(timeout=30.0) as c:
        try:
            r = await c.post(BRAIN + "/api/search", headers=BRAIN_HEADERS, json={"query": question, "limit": 20})
            for f in r.json().get("facts") or []:
                lines.append(f"- {f.get('fact')} (from {f.get('valid_at') or '?'}"
                             + (f", ended {f['invalid_at']}" if f.get("invalid_at") else "") + ")")
        except Exception as exc:
            log.error("brain search: %s", exc)
        lines.append("\nNEWEST MEMORIES (raw):")
        try:
            r = await c.get(BRAIN + "/api/recent?limit=40", headers=BRAIN_HEADERS)
            for m in r.json().get("memories") or []:
                lines.append(f"- {str(m.get('content') or '')[:600]}")
        except Exception as exc:
            log.error("brain recent: %s", exc)
    return "\n".join(lines)


async def compose(question: str, context: str, recent: List[str]) -> str:
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query
    prompt = f"{context}\n\nTHIS CHAT, LAST MESSAGES:\n" + "\n".join(recent[-10:]) + f"\n\nQUESTION: {question}"
    opts = ClaudeAgentOptions(model=MODEL, system_prompt=SYSTEM, allowed_tools=[],
                              mcp_servers={}, strict_mcp_config=True, max_turns=1, setting_sources=[])
    out: List[str] = []
    async for msg in query(prompt=prompt, options=opts):
        if isinstance(msg, AssistantMessage):
            out += [b.text for b in msg.content if isinstance(b, TextBlock)]
    return "\n".join(out).strip()


# ---------- one message ----------
def addressed(msg: Dict[str, Any], text: str) -> bool:
    if msg["chat"]["type"] == "private":
        return True
    handle = (ME.get("username") or "").lower()
    low = text.lower()
    reply_from = (msg.get("reply_to_message") or {}).get("from") or {}
    return (f"@{handle}" in low or re.search(r"\bgondroid\b", low) is not None
            or str(reply_from.get("id")) == str(ME.get("id")))


async def handle(update: Dict[str, Any]) -> None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat, frm = msg["chat"], msg.get("from") or {}
    chat_id, from_id = str(chat["id"]), str(frm.get("id") or "")
    is_group = chat["type"] in ("group", "supergroup")
    if is_group:
        if not await group_trusted(chat_id):
            log.info("group %s untrusted; ignored", chat_id)
            return
    elif from_id != OWNER and from_id not in load()["people"]:
        log.info("DM from untrusted %s; ignored", from_id)
        return
    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not text:
        return
    name = " ".join(x for x in (frm.get("first_name"), frm.get("last_name")) if x) or frm.get("username") or from_id
    if is_group and from_id and from_id not in load()["people"]:
        d = load()
        d["people"][from_id] = {"name": name, "username": frm.get("username")}
        save(d)
    when = datetime.fromtimestamp(msg.get("edit_date") or msg["date"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    who = name + (f" (@{frm['username']})" if frm.get("username") else "")
    where = f'Telegram group "{chat.get("title") or chat_id}"' if is_group else "a Telegram DM with Gondroid"
    edited = "edited_message" in update
    RECENT.setdefault(chat_id, []).append(f"[{when}] {name}: {text}")
    del RECENT[chat_id][:-20]

    async def record():
        try:
            await brain_add(f"On {when} in {where}, {who} {'edited a message to say' if edited else 'said'}: {text}",
                            f"Telegram {chat.get('title') or 'DM'}: {name} {when}")
            log.info("recorded msg %s in %s", msg["message_id"], chat_id)
        except Exception as exc:
            log.error("brain add failed (msg %s in %s): %s", msg["message_id"], chat_id, exc)

    async def answer():
        q = re.sub(r"@" + re.escape(ME.get("username") or "~"), "", text, flags=re.I).strip(" ,:")
        if q.lower() in ("/start", "/help", ""):
            await send(chat_id, "I'm Gondroid. I keep a record of this chat in the team brain. "
                                "Ask me anything that was said or saved and I'll look it up.")
            return
        try:
            await tg("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except Exception:
            pass
        try:
            reply = await asyncio.wait_for(compose(q, await brain_context(q), RECENT[chat_id]), 120)
        except Exception as exc:
            log.error("answer failed: %s", exc)
            reply = ""
        await send(chat_id, reply or "I couldn't put an answer together just now. Try again in a minute.",
                   msg["message_id"] if is_group else None)
        log.info("answered msg %s in %s", msg["message_id"], chat_id)

    jobs = [record()]
    if not edited and addressed(msg, text):
        jobs.append(answer())
    await asyncio.gather(*jobs)


async def main() -> None:
    ME.update(await tg("getMe", {}))
    await tg("deleteWebhook", {"drop_pending_updates": False})
    log.info("gondroid up as @%s; owner set: %s; reads all group messages: %s",
             ME.get("username"), bool(OWNER), ME.get("can_read_all_group_messages"))
    offset = 0
    alive = os.path.join(os.path.dirname(STATE), "alive")
    while True:
        try:
            open(alive, "w").close()  # the container health check reads this file's age
        except OSError:
            pass
        try:
            updates = await tg("getUpdates", {"offset": offset, "timeout": 50,
                                              "allowed_updates": ["message", "edited_message"]}, timeout=70)
        except Exception as exc:
            log.error("getUpdates: %s", exc)
            await asyncio.sleep(5)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            task = asyncio.create_task(handle(u))
            task.add_done_callback(lambda t: t.exception() and log.error("update failed: %r", t.exception()))


if __name__ == "__main__":
    asyncio.run(main())
