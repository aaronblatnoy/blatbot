"""Gondroid: the Definite Optimists team's Telegram ledger. One file, two jobs.

1. Every text message in a trusted chat is written to the Definite Optimists brain
   (POST /api/add, Gondroid's own brain key) with who said it, where and when.
2. A message aimed at Gondroid (a DM, or in a group an @mention of its handle, nothing else) is a question:
   the brain is searched and a short answer goes back to that same chat. Nothing else is sent.

The brain is Gondroid's only memory. The only local state is which groups are trusted:
  * a group is trusted when the owner (OWNER_TELEGRAM_ID) is a member of it (getChatMember);
    every text message in a trusted group is recorded, whoever wrote it;
  * only the people in GONDROID_ALLOWED_USER_IDS (comma-separated Telegram user ids; the owner
    is always included) can DM Gondroid or get an answer from it. Anyone else's DMs, mentions
    and replies are ignored silently (group messages are still recorded).

Files: a document or photo sent in a trusted chat is downloaded, its text is extracted and written to
the brain in parts, each part naming the file, who sent it, where and when. Read: PDF (text layer;
scans are read by the local vision model), Word .docx, PowerPoint .pptx (slide text and notes),
Excel .xlsx/.xls, any text file (.txt, .md, .csv, .json), and images (png, jpg, webp, heic, gif, photos)
via the local Ollama vision model (OLLAMA_URL, qwen2.5vl:7b): it transcribes the visible text and
describes the picture. Audio and video are not read. After a file is ingested an allowed sender gets
one reply line: file name, parts written, stored.
The original bytes are also kept in STATE_DIR/files/<YYYYMMDD-HHMMSS>-<name> with an index.json
(original name, Telegram file_id, sha256, size, chat, sender, date); every brain entry for the file
carries a "Stored file: <stored name> (telegram file_id <id>)" line. An allowed user who asks
Gondroid to send/give/share a file gets it back by DM (never to anyone else). Files over 20 MB
cannot be downloaded by a bot and are not kept.

Updates arrive by long polling (getUpdates), so no public URL is needed.

Environment: TELEGRAM_BOT_TOKEN, OWNER_TELEGRAM_ID, GONDROID_ALLOWED_USER_IDS, DO_BRAIN_KEY, DO_BRAIN_API_URL (optional),
CLAUDE_CODE_OAUTH_TOKEN + LEDGER_MODEL (answers; default "sonnet"), STATE_DIR (default /data).
"""

from __future__ import annotations

import asyncio
import hashlib
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
ALLOWED = {x.strip() for x in os.environ.get("GONDROID_ALLOWED_USER_IDS", "").split(",") if x.strip()} | ({OWNER} if OWNER else set())
BRAIN = (os.environ.get("DO_BRAIN_API_URL") or
         "https://definite-optimists-brain.aaronblatnoy.workers.dev").rstrip("/")
BRAIN_HEADERS = {"Authorization": "Bearer " + os.environ.get("DO_BRAIN_KEY", "").strip(),
                 "User-Agent": "gondroid", "Content-Type": "application/json"}
MODEL = os.environ.get("LEDGER_MODEL", "sonnet").strip()
OLLAMA = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b").strip()
STATE = os.path.join(os.environ.get("STATE_DIR", "/data"), "trust.json")
FILES_DIR = os.path.join(os.environ.get("STATE_DIR", "/data"), "files")
FILES_INDEX = os.path.join(FILES_DIR, "index.json")
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


# ---------- files ----------
FILE_MAX_BYTES = 20 * 1024 * 1024   # the most the Telegram Bot API lets a bot download
PART_CHARS, MAX_PARTS = 3000, 60
SHEET_ROWS, SCAN_PAGES = 2000, 10


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".gif")
MEDIA_EXT = (".mp3", ".m4a", ".wav", ".ogg", ".oga", ".opus", ".flac", ".aac", ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
VISION_PROMPT = ("Transcribe all visible text in this image exactly as written, in reading order. "
                 "Then, on a new line starting with 'Picture:', describe what the image shows in two or three sentences.")


def file_text(name: str, data: bytes) -> str:
    import io
    low = name.lower()
    if low.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(data)).pages)
    if low.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        rows = ["\t".join(c.text for c in r.cells) for t in d.tables for r in t.rows]
        return "\n".join([p.text for p in d.paragraphs] + rows)
    if low.endswith(".pptx"):
        from pptx import Presentation
        out = []
        for i, sl in enumerate(Presentation(io.BytesIO(data)).slides, 1):
            lines = []
            for sh in sl.shapes:
                if sh.has_text_frame and sh.text_frame.text.strip():
                    lines.append(sh.text_frame.text.strip())
                if getattr(sh, "has_table", False) and sh.has_table:
                    lines += ["\t".join(c.text for c in r.cells) for r in sh.table.rows]
            if sl.has_notes_slide and sl.notes_slide.notes_text_frame.text.strip():
                lines.append("Notes: " + sl.notes_slide.notes_text_frame.text.strip())
            if lines:
                out.append(f"Slide {i}:\n" + "\n".join(lines))
        return "\n\n".join(out)
    if low.endswith(".xlsx") or low.endswith(".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets[:20]:
            rows = []
            for row in ws.iter_rows(values_only=True, max_row=SHEET_ROWS):
                if any(v is not None for v in row):
                    rows.append("\t".join("" if v is None else str(v) for v in row))
            if rows:
                out.append(f"Sheet {ws.title}:\n" + "\n".join(rows))
        return "\n\n".join(out)
    if low.endswith(".xls"):
        import xlrd
        wb = xlrd.open_workbook(file_contents=data)
        out = []
        for ws in wb.sheets()[:20]:
            rows = ["\t".join(str(v) for v in ws.row_values(r)) for r in range(min(ws.nrows, SHEET_ROWS))]
            if rows:
                out.append(f"Sheet {ws.name}:\n" + "\n".join(rows))
        return "\n\n".join(out)
    if b"\x00" in data[:4000]:
        raise ValueError("not a text file")
    return data.decode("utf-8", errors="replace")


def to_jpeg(data: bytes) -> bytes:
    """Any image (png, jpg, webp, heic, gif) as a JPEG of at most 1600px, which the vision model reads."""
    import io
    from PIL import Image
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((1600, 1600))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=88)
    return out.getvalue()


def pdf_page_images(data: bytes) -> List[bytes]:
    import io
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    out = []
    for i in range(min(len(pdf), SCAN_PAGES)):
        buf = io.BytesIO()
        pdf[i].render(scale=2).to_pil().convert("RGB").save(buf, "JPEG", quality=88)
        out.append(buf.getvalue())
    return out


async def vision(jpeg: bytes) -> str:
    """Ask the local Ollama vision model to transcribe and describe one image."""
    import base64
    async with httpx.AsyncClient(timeout=240.0) as c:
        r = await c.post(OLLAMA + "/api/chat", json={
            "model": VISION_MODEL, "stream": False, "keep_alive": "5m",
            "messages": [{"role": "user", "content": VISION_PROMPT, "images": [base64.b64encode(jpeg).decode()]}]})
        r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


async def extract(name: str, data: bytes) -> str:
    """Text of any readable file; images and scanned PDFs go through the local vision model."""
    low = name.lower()
    if low.endswith(IMAGE_EXT):
        return await vision(await asyncio.to_thread(to_jpeg, data))
    text = (await asyncio.to_thread(file_text, name, data)).strip()
    if not text and low.endswith(".pdf"):
        pages = await asyncio.to_thread(pdf_page_images, data)
        text = "\n\n".join(f"Page {i}:\n{await vision(j)}" for i, j in enumerate(pages, 1))
    return text.strip()


def parts_of(text: str) -> List[str]:
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        while len(para) > PART_CHARS:
            if cur:
                out.append(cur); cur = ""
            out.append(para[:PART_CHARS]); para = para[PART_CHARS:]
        if cur and len(cur) + len(para) + 2 > PART_CHARS:
            out.append(cur); cur = ""
        cur = (cur + "\n\n" + para).strip() if para else cur
    if cur:
        out.append(cur)
    return out


def safe_name(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(name or "file")).strip("._") or "file"
    return n[:100]


def load_index() -> Dict[str, Any]:
    try:
        return json.load(open(FILES_INDEX))
    except Exception:
        return {}


def save_file(name: str, data: bytes, file_id: str, chat: str, sender: str, when: str) -> str:
    """Keep the original bytes under FILES_DIR and record them in index.json. Returns the stored name."""
    os.makedirs(FILES_DIR, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + safe_name(name)
    stored, n = stem, 1
    while os.path.exists(os.path.join(FILES_DIR, stored)):
        n += 1
        stored = f"{stem}.{n}"
    with open(os.path.join(FILES_DIR, stored + ".tmp"), "wb") as f:
        f.write(data)
    os.replace(os.path.join(FILES_DIR, stored + ".tmp"), os.path.join(FILES_DIR, stored))
    idx = load_index()
    idx[stored] = {"original_name": name, "file_id": file_id, "sha256": hashlib.sha256(data).hexdigest(),
                   "size": len(data), "chat": chat, "sender": sender, "date": when}
    with open(FILES_INDEX + ".tmp", "w") as f:
        json.dump(idx, f, indent=1)
    os.replace(FILES_INDEX + ".tmp", FILES_INDEX)
    return stored


STOP = set("""me the a an please file files pdf doc docx document documents copy of that this us to you can could would
send give share forward resend attach again back i my our it one for from and gondroid sent we want need get
latest last newest most recent""".split())


def find_files(query: str, extra_stored: Optional[List[str]] = None) -> List[str]:
    """Best-matching stored names (newest first). Name match against index.json first; if none,
    fall back to stored names that brain search results mention (extra_stored)."""
    idx = load_index()
    toks = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in STOP]
    newest = sorted(idx, reverse=True)
    if not toks:
        return newest[:5]
    def score(k: str) -> int:
        nm = (idx[k].get("original_name") or "").lower()
        return sum(1 for t in toks if t in nm)
    scored = {k: score(k) for k in idx}
    best = max(scored.values(), default=0)
    if best > 0:
        return sorted((k for k, v in scored.items() if v == best), reverse=True)
    return sorted({k for k in (extra_stored or []) if k in idx}, reverse=True)


SEND_RE = re.compile(r"\b(send|give|share|forward|resend|attach)\b", re.I)
FILE_WORD_RE = re.compile(r"\b(files?|pdf|docx?|documents?|attachments?|deck|spreadsheet|sheet|csv|prd)\b", re.I)


def wants_file(q: str) -> bool:
    return bool(SEND_RE.search(q)) and bool(load_index())


async def send_document(to_chat: str, stored: str) -> None:
    ent = load_index()[stored]
    try:
        await tg("sendDocument", {"chat_id": to_chat, "document": ent["file_id"]}, timeout=60)
        return
    except Exception as exc:
        log.info("sendDocument by file_id failed (%s); uploading stored bytes", exc)
    with open(os.path.join(FILES_DIR, stored), "rb") as f:
        data = f.read()
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"https://api.telegram.org/bot{TOKEN}/sendDocument", data={"chat_id": to_chat},
                         files={"document": (ent.get("original_name") or stored, data)})
    if not r.json().get("ok"):
        raise RuntimeError("sendDocument upload: " + str(r.json().get("description")))


async def file_request(q: str, chat_id: str, from_id: str, is_group: bool, reply_to: Optional[int]) -> bool:
    """Handle "send me the PRD". Returns True when it answered. Files only ever go to the asker's own DM."""
    extra: List[str] = []
    if not find_files(q) or not [t for t in re.findall(r"[a-z0-9]+", q.lower()) if t not in STOP]:
        try:
            ctx = await brain_context(q)
            extra = re.findall(r"Stored file: (\S+?)(?: \(|\s|$)", ctx)
        except Exception:
            pass
    found = find_files(q, extra)
    idx = load_index()
    if not found:
        if FILE_WORD_RE.search(q):
            await send(chat_id, "I don't have a stored file that matches that.", reply_to if is_group else None)
            return True
        return False
    if len(found) > 1:
        lines = [f"- {idx[k]['original_name']} ({idx[k]['date']}, from {idx[k]['sender']})" for k in found[:5]]
        await send(chat_id, "Which one do you want?\n" + "\n".join(lines) + "\nTell me the name and I'll send it.",
                   reply_to if is_group else None)
        return True
    try:
        await send_document(from_id, found[0])
    except Exception as exc:
        log.error("file send failed: %s", exc)
        await send(chat_id, "I couldn't send it to you. Message me directly once (DM me /start), then ask again.",
                   reply_to if is_group else None)
        return True
    if is_group:
        await send(chat_id, f"Sent {idx[found[0]]['original_name']} to you in a DM.", reply_to)
    log.info("sent stored file %s to %s", found[0], from_id)
    return True


async def record_file(doc: Dict[str, Any], who: str, where: str, when: str, chat_title: str, notify: Optional[str] = None) -> str:
    """Download one Telegram document, keep the original bytes, and write its text to the brain. Returns a one-line outcome."""
    name = doc.get("file_name") or "file"
    if (doc.get("file_size") or 0) > FILE_MAX_BYTES:
        if notify:
            await send(notify, f'I can\'t keep "{name}": Telegram only lets a bot download files up to 20 MB.')
        return f"file {name} is over 20 MB, too large for a bot to download; not kept"
    info = await tg("getFile", {"file_id": doc["file_id"]})
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.get(f"https://api.telegram.org/file/bot{TOKEN}/{info['file_path']}")
        r.raise_for_status()
    stored = await asyncio.to_thread(save_file, name, r.content, doc["file_id"], chat_title, who, when)
    ref = f"Stored file: {stored} (telegram file_id {doc['file_id']})"
    try:
        text = await extract(name, r.content)
    except Exception as exc:
        log.info("no text from %s: %s", name, type(exc).__name__)
        text = ""
    if not text:
        await brain_add(f'File "{name}", sent by {who} in {where} on {when}. Its text could not be read, but the original is kept. {ref}',
                        f"File {name} ({chat_title})")
        if notify:
            await send(notify, f'{name}: nothing readable in it, 0 parts written to the brain. The file is stored.')
        return f"file {name}: kept as {stored}, no readable text"
    parts = parts_of(text)
    kept = parts[:MAX_PARTS]
    for i, part in enumerate(kept, 1):
        await brain_add(f'File "{name}" (part {i} of {len(kept)}), sent by {who} in {where} on {when}. {ref}. Its text:\n{part}',
                        f"File {name} part {i}/{len(kept)} ({chat_title})")
    if notify:
        await send(notify, f"{name}: {len(kept)} part(s) written to the brain, file stored.")
    return f"file {name}: kept as {stored}, {len(kept)} part(s) written" + (f", {len(parts) - len(kept)} more left out (too long)" if len(parts) > len(kept) else "")


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
    return bool(handle) and re.search(r"@" + re.escape(handle) + r"\b", text.lower()) is not None


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
    elif from_id not in ALLOWED:
        log.info("DM from untrusted %s; ignored", from_id)
        return
    text = (msg.get("text") or msg.get("caption") or "").strip()
    doc = msg.get("document")
    if not doc and msg.get("photo"):
        ph = max(msg["photo"], key=lambda x: x.get("file_size") or x.get("width", 0))
        doc = {"file_id": ph["file_id"], "file_size": ph.get("file_size"), "file_name": f"photo-{msg['message_id']}.jpg"}
    unread = None
    if not doc:
        for k in ("voice", "audio", "video", "video_note", "animation"):
            if msg.get(k):
                unread = k.replace("_", " ")
    elif (doc.get("mime_type") or "").startswith(("audio/", "video/")) or (doc.get("file_name") or "").lower().endswith(MEDIA_EXT):
        unread, doc = "audio or video", None
    if unread and from_id in ALLOWED and not "edited_message" in update:
        try:
            await send(chat_id, f"I don't read {unread} files. Only documents, spreadsheets, slides and images.")
        except Exception as exc:
            log.error("unread-media reply failed: %s", exc)
    if doc and not text:
        text = f'[sent the file "{doc.get("file_name") or "file"}"]'
    if not text:
        return
    name = " ".join(x for x in (frm.get("first_name"), frm.get("last_name")) if x) or frm.get("username") or from_id
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
        if wants_file(q):
            try:
                if await file_request(q, chat_id, from_id, is_group, msg["message_id"]):
                    return
            except Exception as exc:
                log.error("file request failed: %s", exc)
        try:
            reply = await asyncio.wait_for(compose(q, await brain_context(q), RECENT[chat_id]), 120)
        except Exception as exc:
            log.error("answer failed: %s", exc)
            reply = ""
        await send(chat_id, reply or "I couldn't put an answer together just now. Try again in a minute.",
                   msg["message_id"] if is_group else None)
        log.info("answered msg %s in %s", msg["message_id"], chat_id)

    async def record_doc():
        try:
            log.info("%s (msg %s in %s)", await record_file(doc, who, where, when, chat.get("title") or "DM",
                                                            chat_id if from_id in ALLOWED else None),
                     msg["message_id"], chat_id)
        except Exception as exc:
            log.error("file read failed (msg %s in %s): %s", msg["message_id"], chat_id, type(exc).__name__)

    jobs = [record()]
    if doc and not edited:
        jobs.append(record_doc())
    if not edited and from_id in ALLOWED and addressed(msg, text):
        jobs.append(answer())
    await asyncio.gather(*jobs)


async def main() -> None:
    ME.update(await tg("getMe", {}))
    await tg("deleteWebhook", {"drop_pending_updates": False})
    log.info("gondroid up as @%s; owner set: %s; allowlist loaded: %d user(s); reads all group messages: %s",
             ME.get("username"), bool(OWNER), len(ALLOWED), ME.get("can_read_all_group_messages"))
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
