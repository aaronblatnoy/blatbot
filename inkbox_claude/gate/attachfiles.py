"""Files a send_message can attach on any surface: a local path, a public URL, or both.

A URL is downloaded first and the file itself is what gets sent, so a public PDF
lands as a real document, never as a link. Local paths are limited to the folders
Blatbot keeps its own files in (plus BLATBOT_ATTACH_ROOTS), so a send can never
carry a secret off the machine.
"""

import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Any, List
from urllib.parse import unquote, urlparse

import httpx

FILES_DIR = Path(os.getenv("BLATBOT_ATTACH_DIR") or (Path.home() / "blatbot" / "media" / "attachments"))
MAX_FILES = 5
MAX_BYTES = 45 * 1024 * 1024   # under Telegram's 50 MB bot upload cap


def _roots() -> List[Path]:
    from . import imagetool
    roots = [FILES_DIR, imagetool.GENERATED_DIR]
    try:
        from .. import chatfiles
        roots.append(chatfiles.MEDIA_DIR)
    except Exception:
        pass
    for extra in (os.getenv("BLATBOT_ATTACH_ROOTS") or "").split(os.pathsep):
        if extra.strip():
            roots.append(Path(extra.strip()))
    return [r.expanduser().resolve() for r in roots]


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(unquote(name or "")).name).strip(" .")
    return name[:120] or "file"


def _name_from(resp: "httpx.Response", url: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd, re.I)
    name = _safe_name(m.group(1)) if m else _safe_name(urlparse(url).path.rsplit("/", 1)[-1])
    if "." not in name:
        ext = mimetypes.guess_extension((resp.headers.get("content-type") or "").split(";")[0].strip()) or ""
        name += ext
    return name


async def download(url: str) -> str:
    """Fetch one http(s) URL into FILES_DIR and return the local path."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"{url} is not an http(s) link.")
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (compatible; document-fetch)"}) as client:
        async with client.stream("GET", url) as r:
            if r.status_code >= 400:
                raise ValueError(f"Could not download {url}: HTTP {r.status_code}.")
            ctype = (r.headers.get("content-type") or "").lower()
            if ctype.startswith("text/html"):
                raise ValueError(f"{url} is a web page, not a file (content type {ctype.split(';')[0]}). "
                                 "Find the direct link to the document.")
            data = bytearray()
            async for chunk in r.aiter_bytes():
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise ValueError(f"{url} is larger than {MAX_BYTES // (1024 * 1024)} MB.")
            name = _name_from(r, url)
    if not data:
        raise ValueError(f"{url} returned an empty file.")
    path = FILES_DIR / f"{uuid.uuid4().hex[:8]}-{name}"
    path.write_bytes(bytes(data))
    return str(path)


def local_path(p: str) -> str:
    """A local file under an allowed folder, as an absolute path, else ValueError."""
    q = Path(str(p)).expanduser().resolve()
    if not any(q == r or r in q.parents for r in _roots()):
        raise ValueError(f"{p} is not in a folder files can be sent from. Pass a link to the file, or a path "
                         "the other tools returned.")
    if not q.is_file():
        raise ValueError(f"{p} is not a file.")
    if q.stat().st_size > MAX_BYTES:
        raise ValueError(f"{p} is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if q.stat().st_size == 0:
        raise ValueError(f"{p} is empty.")
    return str(q)


async def resolve_files(items: Any) -> List[str]:
    """Local paths for every file item (a path or an http(s) URL). Raises ValueError with a plain reason."""
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        raise ValueError("files must be a list of paths or links.")
    if len(items) > MAX_FILES:
        raise ValueError(f"Attach at most {MAX_FILES} files per message.")
    out: List[str] = []
    for it in items:
        it = str(it).strip()
        if not it:
            continue
        try:
            out.append(await download(it) if re.match(r"^https?://", it, re.I) else local_path(it))
        except httpx.HTTPError as exc:
            raise ValueError(f"Could not download {it}: {exc}") from exc
    return out
