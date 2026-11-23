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

Transcript: every message seen in a trusted chat (sender, time, text or caption, reply-to id, text read from files and
photos) is also kept in STATE_DIR/transcripts/<chat_id>.jsonl (last 200 per chat). An answer prompt carries the replied-to
message in full (its photo or file is read on the spot if not yet ingested), the last 40 transcript lines of that same chat
only, and the brain results.

Files: a document or photo sent in a trusted chat is downloaded, its text is extracted and written to
the brain in parts, each part naming the file, who sent it, where and when. Read: PDF (text layer;
scans are read by the local vision model), Word .docx, PowerPoint .pptx (slide text and notes),
Excel .xlsx/.xls, any text file (.txt, .md, .csv, .json), and images (png, jpg, webp, heic, gif, photos)
via a vision model: the Bedrock one (BEDROCK_VISION_KEY, BEDROCK_VISION_MODEL, default
qwen.qwen3-vl-235b-a22b-instruct) when its key is set, else local Ollama (OLLAMA_URL, qwen2.5vl:7b). It
describes the picture and transcribes the visible text; an image goes to the brain as one entry,
'Image "<name or photo>" sent by X in Y on Z. What it shows: ... Text in it: ...'. Audio and video are not read. After a file is ingested an allowed sender gets
one reply line: file name, parts written, stored.
The original bytes are also kept in STATE_DIR/files/<YYYYMMDD-HHMMSS>-<name> with an index.json
(original name, Telegram file_id, sha256, size, chat, sender, date); every brain entry for the file
carries a "Stored file: <stored name> (telegram file_id <id>)" line. An allowed user who asks
Gondroid to send/give/share a file gets it posted in the chat they asked in. Files over 20 MB
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
# Images go to a Bedrock vision model (OpenAI-compatible chat/completions) when a key is set; without one, to Ollama.
BEDROCK_URL = (os.environ.get("BEDROCK_VISION_URL") or "https://bedrock-mantle.us-east-2.api.aws/v1").rstrip("/")
BEDROCK_KEY = os.environ.get("BEDROCK_VISION_KEY", "").strip()
BEDROCK_MODEL = os.environ.get("BEDROCK_VISION_MODEL", "qwen.qwen3-vl-235b-a22b-instruct").strip()
STATE = os.path.join(os.environ.get("STATE_DIR", "/data"), "trust.json")
FILES_DIR = os.path.join(os.environ.get("STATE_DIR", "/data"), "files")
FILES_INDEX = os.path.join(FILES_DIR, "index.json")
MEMBER = ("creator", "administrator", "member", "restricted")
ME: Dict[str, Any] = {}
TX_DIR = os.path.join(os.environ.get("STATE_DIR", "/data"), "transcripts")
TX_KEEP = 200        # messages kept per chat
TX_PROMPT = 40       # messages put in an answer prompt
TX_FILE_CAP = 8000   # characters of a file's text kept in a transcript row

SYSTEM = (
    "You are Gondroid, the Definite Optimists team's ledger. The material in the message is everything you know. "
    "It has up to three blocks. 'MESSAGE BEING REPLIED TO' is the exact message the asker replied to, with the text "
    "read out of its photo or file; when it is there, the question is usually about it, so read it first. "
    "'CURRENT CONVERSATION' is the last messages of this very chat in order with names and times: it is what is "
    "being talked about right now, and the newest lines are the most relevant. 'FACTS FROM THE BRAIN' is older "
    "team memory, searched by topic; it may lag by a few minutes, so trust the conversation block for anything recent. "
    "Answer the question from that material only. "
    "Be plain and short: a sentence or a few lines, no headers, no preamble, no em dashes. Say who "
    "said what and when when it helps. If the material does not hold the answer, say you don't have "
    "that recorded. Never invent anything. Telegram does not show you what other bots (such as Samwise) post "
    "unless someone replies to it, so when 'that document' or 'that message' has no match in the conversation "
    "block, do not guess from older memory: say you can't see it and ask them to reply to it and tag you, which "
    "lets you read and keep it.")


# ---------- per-chat transcript ----------
# One file per chat, /data/transcripts/<chat_id>.jsonl, last TX_KEEP rows. A chat only ever reads its own file.
# Limit: Telegram's Bot API never delivers other bots' messages to a bot, so a message another bot posts in a
# group is not in the transcript. It can still be read when someone replies to it, because the reply update
# carries the replied-to message (reply_to_message) in full.
def tx_path(chat_id: str) -> str:
    return os.path.join(TX_DIR, re.sub(r"[^0-9A-Za-z_-]", "_", chat_id) + ".jsonl")


def tx_rows(chat_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(tx_path(chat_id), encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def tx_write(chat_id: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(TX_DIR, exist_ok=True)
    tmp = tx_path(chat_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[-TX_KEEP:]))
    os.replace(tmp, tx_path(chat_id))


def tx_append(chat_id: str, row: Dict[str, Any]) -> None:
    """No await inside, so on the single event loop each append/rewrite is atomic."""
    try:
        os.makedirs(TX_DIR, exist_ok=True)
        with open(tx_path(chat_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["id"] % 20 == 0 or len(tx_rows(chat_id)) > TX_KEEP + 40:
            tx_write(chat_id, tx_rows(chat_id))
    except OSError as exc:
        log.error("transcript write failed: %s", exc)


def tx_set_file_text(chat_id: str, message_id: int, text: str) -> None:
    try:
        rows = tx_rows(chat_id)
        for r in rows:
            if r.get("id") == message_id:
                r["file_text"] = text[:TX_FILE_CAP]
        tx_write(chat_id, rows)
    except OSError as exc:
        log.error("transcript update failed: %s", exc)


def tx_line(r: Dict[str, Any], file_cap: int = 1500) -> str:
    head = f"[{r.get('ts')}] {r.get('who')} (message {r.get('id')}" + (f", reply to {r['reply_to']}" if r.get("reply_to") else "") + ")"
    out = f"{head}: {r.get('text') or ''}"
    if r.get("file_text"):
        out += f"\n   [text read from the file or photo: {r['file_text'][:file_cap]}]"
    return out


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
VISION_PROMPT = ("Read this image. Answer in exactly this form, nothing else:\n"
                 "What it shows: <two or three plain sentences describing the image>\n"
                 "Text in it: <all visible text exactly as written, in reading order, or 'none'>")


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
    """Describe one image and transcribe its text: the Bedrock vision model when BEDROCK_VISION_KEY is set,
    otherwise the local Ollama vision model."""
    import base64
    if BEDROCK_KEY:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=120.0) as c:
            r = await c.post(BEDROCK_URL + "/chat/completions", headers={"Authorization": "Bearer " + BEDROCK_KEY}, json={
                "model": BEDROCK_MODEL, "max_tokens": 1500, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]}]})
            r.raise_for_status()
        out = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        log.info("image read by %s in %.1fs (%d chars)", BEDROCK_MODEL, time.monotonic() - t0, len(out))
        return out.strip()
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
        segs = []
        for i, j in enumerate(pages, 1):
            segs.append(f"Page {i}:\n{await vision(j)}")
        text = "\n\n".join(segs)
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


async def send_document(to_chat: str, stored: str, reply_to: Optional[int] = None) -> None:
    ent = load_index()[stored]
    extra = {"reply_to_message_id": reply_to, "allow_sending_without_reply": True} if reply_to else {}
    try:
        await tg("sendDocument", {"chat_id": to_chat, "document": ent["file_id"], **extra}, timeout=60)
        return
    except Exception as exc:
        log.info("sendDocument by file_id failed (%s); uploading stored bytes", exc)
    with open(os.path.join(FILES_DIR, stored), "rb") as f:
        data = f.read()
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"https://api.telegram.org/bot{TOKEN}/sendDocument", data={"chat_id": to_chat, **{k: str(v) for k, v in extra.items()}},
                         files={"document": (ent.get("original_name") or stored, data)})
    if not r.json().get("ok"):
        raise RuntimeError("sendDocument upload: " + str(r.json().get("description")))


async def file_request(q: str, chat_id: str, from_id: str, is_group: bool, reply_to: Optional[int]) -> bool:
    """Handle "send me the PRD". Returns True when it answered. The file is posted in the chat it was asked in
    (a trusted group or the asker's DM); only allowlisted people get here."""
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
        await send_document(chat_id, found[0], reply_to if is_group else None)
    except Exception as exc:
        log.error("file send failed: %s", exc)
        await send(chat_id, f"I couldn't post {idx[found[0]]['original_name']} here just now. Try again in a minute.",
                   reply_to if is_group else None)
        return True
    log.info("sent stored file %s to chat %s (asked by %s)", found[0], chat_id, from_id)
    return True


async def record_file(doc: Dict[str, Any], who: str, where: str, when: str, chat_title: str, notify: Optional[str] = None,
                      sink: Optional[Dict[str, str]] = None) -> str:
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
        return f"file {name}: kept as {stored}, no readable text"
    if sink is not None:
        sink["text"] = text
    if name.lower().endswith(IMAGE_EXT):
        label = "photo" if name.startswith("photo-") else name
        await brain_add(f'Image "{label}" sent by {who} in {where} on {when}. {text[:6000]}\n{ref}',
                        f"Image {name} ({chat_title})")
        return f"image {name}: kept as {stored}, description written"
    parts = parts_of(text)
    kept = parts[:MAX_PARTS]
    for i, part in enumerate(kept, 1):
        await brain_add(f'File "{name}" (part {i} of {len(kept)}), sent by {who} in {where} on {when}. {ref}. Its text:\n{part}',
                        f"File {name} part {i}/{len(kept)} ({chat_title})")
    return f"file {name}: kept as {stored}, {len(kept)} part(s) written" + (f", {len(parts) - len(kept)} more left out (too long)" if len(parts) > len(kept) else "")


def doc_of(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The file or photo of a Telegram message, shaped like record_file wants it."""
    doc = msg.get("document")
    if doc and (doc.get("mime_type") or "").startswith("image/") and not (doc.get("file_name") or "").lower().endswith(IMAGE_EXT):
        # An image sent as a document without an image extension is still read as an image.
        doc = dict(doc, file_name=(doc.get("file_name") or f"image-{msg['message_id']}") + "." + doc["mime_type"].split("/", 1)[1].split(";")[0])
    if not doc and msg.get("photo"):
        ph = max(msg["photo"], key=lambda x: x.get("file_size") or x.get("width", 0))
        doc = {"file_id": ph["file_id"], "file_size": ph.get("file_size"), "file_name": f"photo-{msg['message_id']}.jpg"}
    return doc


async def read_doc(doc: Dict[str, Any]) -> str:
    """Download and read a file now (not stored, not written to the brain). Used for a replied-to file not yet ingested."""
    name = doc.get("file_name") or "file"
    if (doc.get("file_size") or 0) > FILE_MAX_BYTES:
        return ""
    info = await tg("getFile", {"file_id": doc["file_id"]})
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.get(f"https://api.telegram.org/file/bot{TOKEN}/{info['file_path']}")
        r.raise_for_status()
    return (await extract(name, r.content))[:12000]


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


async def reply_block(msg: Dict[str, Any], chat_id: str) -> str:
    """The message the asker replied to, in full, with the text of its photo or file."""
    rep_msg = msg.get("reply_to_message")
    if not rep_msg:
        return ""
    sender = rep_msg.get("from") or {}
    name = " ".join(x for x in (sender.get("first_name"), sender.get("last_name")) if x) or sender.get("username") or "unknown"
    if sender.get("is_bot"):
        name += " (a bot)"
    when = datetime.fromtimestamp(rep_msg.get("date") or 0, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = (rep_msg.get("text") or rep_msg.get("caption") or "").strip()
    row = next((r for r in tx_rows(chat_id) if r.get("id") == rep_msg.get("message_id")), None)
    file_text = (row or {}).get("file_text") or ""
    doc = doc_of(rep_msg)
    note = ""
    if doc and not file_text:
        try:
            file_text = await asyncio.wait_for(read_doc(doc), 110)
            log.info("read replied-to file now (msg %s in %s): %d chars", rep_msg.get("message_id"), chat_id, len(file_text))
        except Exception as exc:
            log.error("replied-to file read failed: %s", type(exc).__name__)
            note = " (it has a photo or file I could not read just now)"
    out = f"[{when}] {name} (message {rep_msg.get('message_id')}){note}: {text or '(no text)'}"
    if doc:
        out += f"\nFile or photo: {doc.get('file_name') or 'photo'}\nText read from it:\n{file_text or '(nothing readable)'}"
    return out


async def ingest_bot_reply(msg: Dict[str, Any], chat_id: str, where: str, chat_title: str) -> None:
    """Telegram never delivers another bot's messages, so the only time Gondroid sees one is when someone replies to
    it. Take that chance: put the bot's message in the transcript and the brain, and keep its file like any other."""
    rep = msg.get("reply_to_message") or {}
    sender = rep.get("from") or {}
    if not sender.get("is_bot") or sender.get("id") == ME.get("id"):
        return
    if any(r.get("id") == rep.get("message_id") for r in tx_rows(chat_id)):
        return
    doc = doc_of(rep)
    text = (rep.get("text") or rep.get("caption") or "").strip() or (f'[sent the file "{doc.get("file_name") or "file"}"]' if doc else "")
    if not text:
        return
    name = " ".join(x for x in (sender.get("first_name"), sender.get("last_name")) if x) or sender.get("username") or "a bot"
    who = name + (f" (@{sender['username']})" if sender.get("username") else "") + " (a bot)"
    when = datetime.fromtimestamp(rep.get("date") or 0, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tx_append(chat_id, {"id": rep["message_id"], "ts": when, "who": name + " (a bot)", "from_id": str(sender.get("id") or ""),
                        "text": text[:4000], "reply_to": None, "file": (doc or {}).get("file_name"), "edited": False})
    try:
        await brain_add(f"On {when} in {where}, {who} said: {text}", f"Telegram {chat_title}: {name} {when}")
        if doc and not (doc.get("mime_type") or "").startswith(("audio/", "video/")):
            sink: Dict[str, str] = {}
            log.info("bot file via reply: %s (msg %s in %s)", await asyncio.wait_for(
                record_file(doc, who, where, when, chat_title, None, sink), 150), rep["message_id"], chat_id)
            if sink.get("text"):
                tx_set_file_text(chat_id, rep["message_id"], sink["text"])
    except Exception as exc:
        log.error("bot reply ingest failed (msg %s in %s): %s", rep.get("message_id"), chat_id, type(exc).__name__)


def transcript_block(chat_id: str) -> str:
    return "\n".join(tx_line(r) for r in tx_rows(chat_id)[-TX_PROMPT:])


async def compose(question: str, context: str, replied: str = "", transcript: str = "") -> str:
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query
    prompt = ""
    if replied:
        prompt += f"MESSAGE BEING REPLIED TO (the asker replied to this message):\n{replied}\n\n"
    prompt += f"CURRENT CONVERSATION (this chat, oldest first, newest last):\n{transcript or '(nothing yet)'}\n\n{context}\n\nQUESTION: {question}"
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
    doc = doc_of(msg)
    unread = None
    if not doc:
        for k in ("voice", "audio", "video", "video_note", "animation"):
            if msg.get(k):
                unread = k.replace("_", " ")
    elif (doc.get("mime_type") or "").startswith(("audio/", "video/")) or (doc.get("file_name") or "").lower().endswith(MEDIA_EXT):
        unread, doc = "audio or video", None
    if unread:
        log.info("not read: %s (msg %s in %s)", unread, msg.get("message_id"), chat_id)
    if doc and not text:
        text = f'[sent the file "{doc.get("file_name") or "file"}"]'
    if not text:
        return
    name = " ".join(x for x in (frm.get("first_name"), frm.get("last_name")) if x) or frm.get("username") or from_id
    when = datetime.fromtimestamp(msg.get("edit_date") or msg["date"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    who = name + (f" (@{frm['username']})" if frm.get("username") else "")
    where = f'Telegram group "{chat.get("title") or chat_id}"' if is_group else "a Telegram DM with Gondroid"
    edited = "edited_message" in update
    await ingest_bot_reply(msg, chat_id, where, chat.get("title") or "DM")
    tx_append(chat_id, {"id": msg["message_id"], "ts": when, "who": name, "from_id": from_id, "text": text[:4000],
                        "reply_to": (msg.get("reply_to_message") or {}).get("message_id"),
                        "file": (doc or {}).get("file_name"), "edited": edited})

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
            replied = await reply_block(msg, chat_id)
            reply = await asyncio.wait_for(compose(q, await brain_context(q), replied, transcript_block(chat_id)), 120)
        except Exception as exc:
            log.error("answer failed: %s", exc)
            reply = ""
        await send(chat_id, reply or "I couldn't put an answer together just now. Try again in a minute.",
                   msg["message_id"] if is_group else None)
        log.info("answered msg %s in %s", msg["message_id"], chat_id)

    async def record_doc():
        sink: Dict[str, str] = {}
        try:
            log.info("%s (msg %s in %s)", await record_file(doc, who, where, when, chat.get("title") or "DM",
                                                            None, sink),
                     msg["message_id"], chat_id)
            if sink.get("text"):
                tx_set_file_text(chat_id, msg["message_id"], sink["text"])
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
