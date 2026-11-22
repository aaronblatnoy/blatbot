"""What a Telegram chat holds, read for a run: the files and photos sent in it, the
message a request replies to, and the chat's recent messages from everyone.

Files: photos, images, PDFs (text layer; scans through the vision model), .docx, .pptx,
.xlsx/.xls/.csv and text files. The extraction itself is Gondroid's
(deploy/gondroid/gondroid.py: ``extract``), imported from the same repo rather than
written twice; images go to the local Ollama vision model (qwen2.5vl:7b at
127.0.0.1:11434 by default, BLATBOT_OLLAMA_URL / BLATBOT_VISION_MODEL to change).
What a file reads as becomes the text of its message, so it is stored with the chat
like any message and reaches long-term memory tagged with that chat (store.add_message
-> graphmem.record), where the per-fact privacy already applies.

Recent messages: the Bot API never delivers other bots' messages and has no history
call, so for a GROUP chat the window is read through Aaron's own Telegram account
(read-only, ~/blatbot/telegram-user, a fresh subprocess per read, the same transport
gate/tgread.py uses). Only group chats are read that way: a bot DM's chat id is the
person's user id, which in Aaron's account is his own private DM with them, so DMs use
only what Blatbot itself stored for that chat. Nothing here sends, edits or marks read.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

FILE_MAX_BYTES = 20 * 1024 * 1024      # the most a bot may download
FILE_CHARS = int(os.getenv("BLATBOT_FILE_CHARS") or 8000)   # per file, in a run and in memory
WINDOW_MESSAGES = 40
WINDOW_MIN_MESSAGES = 20               # always this many, however old: a follow-up hours later still has its context
WINDOW_SECONDS = 6 * 3600
# Photos and images are kept on disk so the model that answers sees the pixels, not only
# the local vision model's description (which once called a guilloche "dividers").
MEDIA_DIR = Path(os.getenv("BLATBOT_MEDIA_DIR") or (Path.home() / "blatbot" / "media" / "telegram"))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
IMAGE_MARK = "image file: "
_IMAGE_RE = re.compile(r"image file: (/\S+?\.(?:jpg|jpeg|png|webp|gif))", re.IGNORECASE)
DOWNLOADS_PER_WINDOW = 3               # files read through Aaron's account per context build
TZ = ZoneInfo("America/New_York")

_GONDROID = Path(__file__).resolve().parents[1] / "deploy" / "gondroid" / "gondroid.py"
_reader: Any = None


def _gondroid() -> Any:
    """Gondroid's module, loaded once by path (it is a deploy script, not a package)."""
    global _reader
    if _reader is None:
        spec = importlib.util.spec_from_file_location("gondroid_reader", str(_GONDROID))
        if spec is None or spec.loader is None:
            raise RuntimeError("the file reader (deploy/gondroid/gondroid.py) is missing")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.OLLAMA = (os.getenv("BLATBOT_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
        mod.VISION_MODEL = os.getenv("BLATBOT_VISION_MODEL") or "qwen2.5vl:7b"
        mod.BEDROCK_KEY = ""   # Blatbot's images stay on the local model, whatever the environment holds
        _reader = mod
    return _reader


def _cap(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= FILE_CHARS:
        return text
    return text[:FILE_CHARS].rstrip() + f"\n(truncated: the file reads as {len(text)} characters; the first {FILE_CHARS} are shown)"


async def read_bytes(name: str, data: bytes) -> str:
    """The text of one file, capped. Never raises: a file that cannot be read says so."""
    if len(data) > FILE_MAX_BYTES:
        return "(too large to read)"
    try:
        text = await _gondroid().extract(name or "file", data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[files] could not read %s: %s: %s", name, type(exc).__name__, exc)
        return f"(could not be read: {type(exc).__name__})"
    return _cap(text) or "(no readable text)"


def is_image(kind: str, name: str = "") -> bool:
    return kind == "photo" or Path(name or "").suffix.lower() in IMAGE_EXTS


def save_image(chat_id: str, key: str, name: str, data: bytes, at: float = 0.0) -> str:
    """Keep an image under MEDIA_DIR/<chat>/ and return its path ("" when it cannot be
    written). The file's time is the message's time, so "recent images in this chat"
    is a directory listing that survives restarts."""
    try:
        ext = Path(name or "").suffix.lower()
        ext = ext if ext in IMAGE_EXTS else ".jpg"
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(key or "img"))[:80] or "img"
        d = MEDIA_DIR / re.sub(r"[^0-9A-Za-z_-]", "_", str(chat_id or "unknown"))
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{safe}{ext}"
        if not path.exists():
            path.write_bytes(data)
        if at:
            os.utime(path, (float(at), float(at)))
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[files] could not keep image %s: %s", name or key, exc)
        return ""


def image_paths(text: str) -> List[str]:
    """Image files named in a text (a message, a prompt), in order, once each, that exist."""
    out: List[str] = []
    for p in _IMAGE_RE.findall(text or ""):
        if p not in out and os.path.isfile(p):
            out.append(p)
    return out


def mentions_unread_image(text: str) -> bool:
    """A photo or image is in the text but its pixels are not on disk (download failed)."""
    t = text or ""
    return ("[photo," in t or "[photo]" in t) and not image_paths(t)


def recent_images(chat_id: str, within_s: float = 900.0) -> List[str]:
    """Images kept for this chat whose message time is within `within_s`, newest first."""
    d = MEDIA_DIR / re.sub(r"[^0-9A-Za-z_-]", "_", str(chat_id or ""))
    try:
        now = time.time()
        files = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS and now - p.stat().st_mtime <= within_s]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def label(kind: str, name: str = "") -> str:
    if kind == "photo":
        return "photo"
    return f"file {name}" if name else "file"


def render(kind: str, name: str, read: str, path: str = "") -> str:
    """How a file appears inside a message's text. An image kept on disk carries its path
    (the run that answers attaches it as an image); the description is the local vision
    model's and is marked as such."""
    if path:
        return (f"[{label(kind, name)}, {IMAGE_MARK}{path}\n"
                f"automatic description (a small local model; can be wrong, the image itself is what counts):\n{read}\n]")
    if is_image(kind, name):
        return f"[{label(kind, name)}, the image itself could not be read; automatic description:\n{read}\n]"
    return f"[{label(kind, name)}, contents as read:\n{read}\n]"


# -- files delivered by the Bot API ----------------------------------------------------

_reads: Dict[str, "asyncio.Future[Tuple[str, str]]"] = {}
_by_time: Dict[Tuple[str, int], str] = {}   # (chat_id, unix second) -> rendered files of that message


async def _read_bot_file(item: Dict[str, Any], chat_id: str = "", at: float = 0.0) -> Tuple[str, str]:
    """(text as read, kept image path or "")."""
    from . import telegram
    if int(item.get("size") or 0) > FILE_MAX_BYTES:
        return "(too large to read)", ""
    try:
        data = await telegram.download(item["file_id"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("[files] download failed (%s): %s", item.get("kind"), type(exc).__name__)
        return f"(could not be downloaded: {type(exc).__name__})", ""
    path = ""
    if is_image(item.get("kind") or "", item.get("name") or ""):
        path = save_image(chat_id, str(item.get("file_unique_id") or item.get("file_id") or ""),
                          item.get("name") or "", data, at)
    return await read_bytes(item.get("name") or "file", data), path


async def read_media(media: List[Dict[str, Any]], chat_id: str = "", at: float = 0.0) -> str:
    """All files on one Bot API message, rendered. One read per file however many
    messages ask (the photo and a reply to it share the same read)."""
    parts = []
    for item in media or []:
        key = str(item.get("file_unique_id") or item.get("file_id") or "")
        fut = _reads.get(key)
        if fut is None:
            fut = asyncio.ensure_future(_read_bot_file(item, chat_id, at))
            _reads[key] = fut
            if len(_reads) > 500:
                for k in list(_reads)[:100]:
                    _reads.pop(k, None)
        try:
            read, path = await asyncio.shield(fut)
        except Exception as exc:  # noqa: BLE001
            read, path = f"(could not be read: {type(exc).__name__})", ""
        parts.append(render(item.get("kind") or "file", item.get("name") or "", read, path))
    return "\n".join(parts)


def remember_read(chat_id: str, date: int, rendered: str) -> None:
    if chat_id and date and rendered:
        _by_time[(str(chat_id), int(date))] = rendered
        if len(_by_time) > 2000:
            for k in list(_by_time)[:500]:
                _by_time.pop(k, None)


def when(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=TZ).strftime("%a %H:%M")
    except Exception:  # noqa: BLE001
        return "?"


def reply_block(reply: Dict[str, Any], rendered_files: str) -> str:
    """The message a request replies to, whole."""
    who = reply.get("name") or reply.get("username") or "someone"
    if reply.get("is_bot") and reply.get("username"):
        who = f"{who} (@{reply['username']}, a bot)"
    body = (reply.get("text") or "").strip()
    lines = [f"--- THE MESSAGE THIS REPLIES TO: {who}, {when(reply.get('date') or 0)} ---"]
    if body:
        lines.append(body)
    if rendered_files:
        lines.append(rendered_files)
    if not body and not rendered_files:
        lines.append("(no text)")
    lines.append("--- END OF THE REPLIED-TO MESSAGE; the request below is about it ---")
    return "\n".join(lines)


# -- the chat's recent messages ----------------------------------------------------------

_CONTEXT_SCRIPT = r"""
import sys, json, asyncio, os
import server

async def main():
    a = json.loads(sys.argv[1])
    c = await server.client()
    ent = await server._entity(c, a["chat"])
    since = float(a["since"])
    known = set(int(x) for x in a.get("known") or [])
    budget = int(a.get("downloads") or 0)
    ddir = a.get("dir")
    out = []
    keep = int(a.get("keep") or 0)
    async for m in c.iter_messages(ent, limit=int(a["limit"])):
        ts = m.date.timestamp()
        if ts < since and len(out) >= keep:
            break
        s = m.sender
        name = " ".join(x for x in [getattr(s, "first_name", None), getattr(s, "last_name", None)] if x) \
            or getattr(s, "title", None) or getattr(s, "username", None) or "?"
        row = {"id": m.id, "at": ts, "name": name, "is_bot": bool(getattr(s, "bot", False)),
               "username": getattr(s, "username", None) or "", "text": m.message or "", "media": ""}
        if m.media:
            row["media"] = "photo" if m.photo else ("file" if m.document else "media")
            f = getattr(m, "file", None)
            row["file_name"] = (getattr(f, "name", None) or "") if f else ""
            size = (getattr(f, "size", 0) or 0) if f else 0
            if (ddir and budget > 0 and int(ts) not in known and row["media"] in ("photo", "file")
                    and size <= 20 * 1024 * 1024):
                d = os.path.join(ddir, str(m.id))
                os.makedirs(d, exist_ok=True)
                try:
                    p = await m.download_media(file=d + "/")
                    if p:
                        row["path"] = p
                        budget -= 1
                except Exception as e:
                    row["download_error"] = type(e).__name__
        out.append(row)
    print(json.dumps({"messages": out}))

asyncio.run(main())
"""


async def _account_window(chat_id: str, since: float, downloads: int) -> List[Dict[str, Any]]:
    """The group's messages since `since`, newest first, through Aaron's account, with up
    to `downloads` files fetched that Blatbot has not already read."""
    from .gate import tgread
    transport = tgread.SubprocessTransport()
    known = [t for (c, t) in _by_time if c == str(chat_id)]
    tmp = tempfile.mkdtemp(prefix="blatbot-files-")
    try:
        data = await asyncio.wait_for(transport._run(_CONTEXT_SCRIPT, {
            "chat": int(chat_id), "since": since, "limit": WINDOW_MESSAGES, "known": known,
            "keep": WINDOW_MIN_MESSAGES,
            "downloads": downloads, "dir": tmp}), 120)
        rows = list(data.get("messages") or [])
        for r in rows:
            p = r.pop("path", None)
            if not p:
                continue
            try:
                raw = Path(p).read_bytes()
            except OSError:
                continue
            name = r.get("file_name") or Path(p).name
            kept = (save_image(chat_id, f"m{r.get('id')}", name, raw, float(r.get("at") or 0))
                    if is_image(r.get("media") or "", name) else "")
            r["read"] = render(r.get("media") or "file", r.get("file_name") or "", await read_bytes(name, raw), kept)
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _store_lines(store: Any, chat_key: str, since: float, before: float) -> List[str]:
    lines: List[str] = []
    bot = os.getenv("TELEGRAM_BOT_NAME") or "Blatbot"
    rows = [r for r in store.history(chat_key, WINDOW_MESSAGES + 10)
            if r.get("kind") in ("inbound", "outbound")
            and not (float(r.get("created_at") or 0) and float(r.get("created_at") or 0) >= before)]
    for i, row in enumerate(rows):
        at = float(row.get("created_at") or 0)
        if at and at < since and i < len(rows) - WINDOW_MIN_MESSAGES:
            continue
        text = str(row.get("text") or "").strip()
        if text.startswith("[telegram group:"):
            head, _, rest = text.partition("]")
            who = ""
            for part in head.split("|"):
                if part.strip().startswith("from:"):
                    who = part.split(":", 1)[1].strip()
            text = f"{who}: {rest.strip()}" if who else rest.strip()
        if row.get("kind") == "outbound":
            text = f"{bot}: {text}"
        if text:
            lines.append(f"{when(at)} {text}" if at else text)
    return lines[-WINDOW_MESSAGES:]


async def recent_chat(chat_id: str, *, is_group: bool, store: Any, before: float = 0.0) -> str:
    """The last ~40 messages or ~6 hours of THIS chat (whichever is smaller, but never fewer
    than the last 20 however old), oldest first,
    with names, times and files as read. Before `before` (the message being answered,
    shown separately). "" when there is nothing."""
    now = time.time()
    before = before or now + 1
    since = max(now - WINDOW_SECONDS, 0)
    lines: List[str] = []
    if is_group and str(chat_id).startswith("-"):
        try:
            rows = await _account_window(chat_id, since, DOWNLOADS_PER_WINDOW)
            for r in reversed(rows):
                at = float(r.get("at") or 0)
                if at >= before:
                    continue
                who = r.get("name") or "?"
                if r.get("is_bot") and r.get("username"):
                    who = f"{who} (@{r['username']}, bot)"
                files = r.get("read") or _by_time.get((str(chat_id), int(at)), "")
                if r.get("media") and not files:
                    files = f"[{label(r['media'], r.get('file_name') or '')}, not read]"
                text = (r.get("text") or "").strip()
                body = "\n".join(x for x in (files, text) if x)
                lines.append(f"{when(at)} {who}: {body}")
            lines = lines[-WINDOW_MESSAGES:]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[files] could not read the chat through Aaron's account (%s); using Blatbot's own record",
                           type(exc).__name__)
            lines = []
    if not lines and store is not None:
        try:
            lines = _store_lines(store, f"tg:{chat_id}", since, before)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[files] could not read the stored chat: %s", exc)
    if not lines:
        return ""
    return ("--- RECENT MESSAGES IN THIS CHAT (everyone, people and bots, oldest first; "
            "only this chat) ---\n" + "\n".join(lines) + "\n--- END OF RECENT MESSAGES ---")
