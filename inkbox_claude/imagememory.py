"""Images that arrive as attachments (SMS/MMS, iMessage, email), described by the LOCAL
vision model and written into the chat's history, and so into long-term memory.

The gateway saves inbound attachments to disk and appends a note listing them
(media.inbound_media_note: "- <path> (<content type>)"). When a message carrying such a
note is stored, every image in it is handed to the local Ollama vision model
(qwen2.5vl:7b, through chatfiles.read_bytes, the same reader Telegram photos use) in a
background task, one image at a time, so a reply is never held up by it. The
description is stored as a "system" message in that same chat (store.add_message), which
queues it for the memory graph (graphmem.record) tagged with that chat, so the per-fact
privacy that applies to the chat's messages applies to the image too.

Telegram photos are not handled here: they are read before routing and their text is
already part of the stored message (gateway._telegram_read_then_route).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".gif")
_NOTE_LINE = re.compile(r"^- (/\S+) \(([^)]*)\)\s*$", re.M)
_CHANNEL = {"sms": "SMS", "imessage": "iMessage", "email": "email", "whatsapp": "WhatsApp"}
_one_at_a_time: Optional[asyncio.Semaphore] = None
_tasks: set = set()


def image_paths(body: str) -> List[Tuple[str, str]]:
    """(path, content type) of each image the attachments note in `body` lists."""
    if "[Attachments received" not in (body or ""):
        return []
    out = []
    for path, ctype in _NOTE_LINE.findall(body):
        if (ctype or "").lower().startswith("image/") or path.lower().endswith(IMAGE_EXT):
            out.append((path, ctype))
    return out


def schedule(store: Any, record_id: str, body: str, mode: str) -> None:
    """Describe and record the images listed in `body`, in the background. Never raises,
    never waits."""
    try:
        paths = image_paths(body)
        if not paths:
            return
        task = asyncio.get_running_loop().create_task(_describe_all(store, record_id, paths, mode),
                                                      name=f"image-memory-{record_id}")
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    except Exception as exc:  # noqa: BLE001 - memory must never break storing a message
        logger.warning("[images] could not schedule image reading for %s: %s", record_id, exc)


def line(name: str, sender: str, channel: str, read: str) -> str:
    who = f" sent by {sender}" if sender else ""
    return f'Image "{name}"{who} over {channel}, as read by the local vision model.\n{read}'


async def _describe_all(store: Any, record_id: str, paths: List[Tuple[str, str]], mode: str) -> None:
    global _one_at_a_time
    from . import chatfiles
    if _one_at_a_time is None:
        _one_at_a_time = asyncio.Semaphore(1)   # Ollama runs one request at a time on its GPU
    channel = _CHANNEL.get(mode, mode or "a message")
    try:
        sender = store.memory_speaker(record_id) if hasattr(store, "memory_speaker") else ""
    except Exception:  # noqa: BLE001
        sender = ""
    for path, _ctype in paths:
        p = Path(path)
        try:
            data = await asyncio.to_thread(p.read_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[images] could not open %s: %s", p.name, type(exc).__name__)
            continue
        async with _one_at_a_time:
            t0 = time.monotonic()
            read = await chatfiles.read_bytes(p.name, data)
            took = time.monotonic() - t0
        if read.startswith("(") and read.endswith(")"):
            logger.warning("[images] %s in %s not described: %s", p.name, record_id, read)
            continue
        try:
            store.add_message(record_id, "system", line(p.name, sender, channel, read))
            logger.info("[images] described %s in %s in %.1fs (%d chars)", p.name, record_id, took, len(read))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[images] could not store the description of %s: %s", p.name, exc)
