"""Formatted HTML pages: blatbot answers a long or structured request with a page and sends
a short text containing the link.

A page is one file on disk under <state dir>/pages, named by a long random token, with a
small sidecar file holding the title and the expiry. The gateway serves it at
/p/<token>. A wrong, malformed or expired token is a plain 404, and nothing lists the
directory. The page body is written by a model, so it is rebuilt from an allowlist of tags
and attributes before it is stored, and it is served with a strict Content-Security-Policy.
"""

from __future__ import annotations

import html as _html
import json
import logging
import os
import re
import secrets
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

logger = logging.getLogger(__name__)

PAGE_PATH_PREFIX = "/p/"
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
MAX_BODY_CHARS = 400_000
MAX_TITLE_CHARS = 140

# ---------------------------------------------------------------------------
# Sanitiser: a fixed list of tags and attributes. Anything else is dropped; the text
# inside a dropped tag is kept, except for tags whose content is code or markup that must
# never reach the page at all.
# ---------------------------------------------------------------------------

ALLOWED_TAGS = {
    "h1", "h2", "h3", "h4", "p", "br", "hr", "ul", "ol", "li", "table", "thead", "tbody",
    "tfoot", "tr", "th", "td", "caption", "a", "strong", "b", "em", "i", "u", "code", "pre",
    "blockquote", "span", "div", "small", "sup", "sub", "mark", "del", "ins", "dl", "dt", "dd",
}
VOID_TAGS = {"br", "hr"}
DROP_WITH_CONTENT = {"script", "style", "iframe", "frame", "frameset", "object", "embed", "form",
                     "template", "noscript", "svg", "math", "head", "title", "textarea", "select",
                     "button", "audio", "video", "canvas", "applet", "xmp", "noembed", "noframes"}
DROP_VOID = {"embed"}      # dropped tags that never have a closing tag
ALLOWED_ATTRS = {
    "a": {"href", "title"},
    "th": {"colspan", "rowspan", "scope", "class"},
    "td": {"colspan", "rowspan", "class"},
    "ol": {"start"},
    "span": {"class"},
    "div": {"class"},
    "p": {"class"},
    "table": {"class"},
    "li": {"class"},
}
ALLOWED_CLASSES = {"note", "lede", "muted", "num", "tag", "good", "warn", "bad", "callout", "kv"}
SAFE_HREF = re.compile(r"^(https?://|mailto:|tel:|#)", re.I)


def _clean_href(value: str) -> Optional[str]:
    # Browsers ignore control characters and whitespace inside a scheme, so remove them
    # before looking at it: "java\tscript:" must not slip past.
    v = re.sub(r"[\x00-\x20\x7f-\x9f]+", "", value or "")
    if not v or not SAFE_HREF.match(v):
        return None
    return v[:2000]


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.stack: List[str] = []
        self.skip = 0                 # depth inside a dropped-with-content tag
        self.skip_tags: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if self.skip:
            if tag in DROP_WITH_CONTENT and tag not in DROP_VOID:
                self.skip_tags.append(tag)
                self.skip += 1
            return
        if tag in DROP_WITH_CONTENT:
            if tag not in DROP_VOID:
                self.skip_tags.append(tag)
                self.skip = 1
            return
        if tag not in ALLOWED_TAGS:
            return
        allowed = ALLOWED_ATTRS.get(tag, set())
        parts = []
        href_seen = False
        for name, value in attrs:
            name = (name or "").lower()
            if name not in allowed:
                continue
            value = value or ""
            if name == "href":
                href = _clean_href(value)
                if href is None:
                    continue
                value, href_seen = href, True
            elif name == "class":
                kept = [c for c in value.split() if c in ALLOWED_CLASSES]
                if not kept:
                    continue
                value = " ".join(kept)
            elif name in ("colspan", "rowspan", "start"):
                if not value.strip().isdigit() or len(value.strip()) > 3:
                    continue
                value = value.strip()
            elif name == "scope":
                if value not in ("row", "col", "rowgroup", "colgroup"):
                    continue
            else:
                value = value[:300]
            parts.append(f' {name}="{_html.escape(value, quote=True)}"')
        if tag == "a":
            if href_seen:
                parts.append(' rel="noopener noreferrer nofollow" target="_blank"')
            else:
                return                # an anchor with no safe link is just its text
        self.out.append(f"<{tag}{''.join(parts)}>")
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip:
            if tag in DROP_WITH_CONTENT and tag in self.skip_tags:
                # close the innermost matching dropped tag
                idx = len(self.skip_tags) - 1 - self.skip_tags[::-1].index(tag)
                del self.skip_tags[idx:]
                self.skip = len(self.skip_tags)
            return
        if tag not in ALLOWED_TAGS or tag in VOID_TAGS:
            return
        if tag == "a" and "a" not in self.stack:
            return
        if tag in self.stack:
            while self.stack:
                top = self.stack.pop()
                self.out.append(f"</{top}>")
                if top == tag:
                    break

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        self.out.append(_html.escape(data, quote=False))

    def handle_comment(self, data: str) -> None:
        return

    def handle_decl(self, decl: str) -> None:
        return

    def handle_pi(self, data: str) -> None:
        return

    def unknown_decl(self, data: str) -> None:
        return

    def close_all(self) -> str:
        while self.stack:
            self.out.append(f"</{self.stack.pop()}>")
        return "".join(self.out)


def sanitize_html(body: str) -> str:
    """Rebuild `body` from the allowlist. Never raises; on a parser failure the text is
    escaped whole, so the worst case is a page of plain text."""
    body = (body or "")[:MAX_BODY_CHARS]
    parser = _Sanitizer()
    try:
        parser.feed(body)
        parser.close()
        return parser.close_all()
    except Exception:
        logger.exception("[pages] sanitiser failed; escaping the whole body")
        return "<p>" + _html.escape(body) + "</p>"


# ---------------------------------------------------------------------------
# The page: inline CSS only, no script, no external request.
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#0b1220;--panel:#111a2e;--ink:#e8ecf5;--muted:#9aa6bf;--line:rgba(255,255,255,.09);--accent:#6ea0ff;--accent-soft:rgba(110,160,255,.14);--good:#6ee7b7;--warn:#fbbf77;--bad:#fca5a5;--r:8px;--shadow:0 1px 2px rgba(0,0,0,.35),0 8px 24px rgba(0,0,0,.28)}
@media (prefers-color-scheme:light){:root{--bg:#f4f6fb;--panel:#fff;--ink:#0f172a;--muted:#5b6679;--line:rgba(15,23,42,.10);--accent:#2b5fd9;--accent-soft:rgba(43,95,217,.10);--good:#047857;--warn:#b45309;--bad:#b91c1c;--shadow:0 1px 2px rgba(15,23,42,.06),0 8px 24px rgba(15,23,42,.07)}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:760px;margin:0 auto;padding:28px 16px 64px}
header{padding:8px 0 22px;border-bottom:1px solid var(--line);margin-bottom:24px}
header .brand{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:600}
header h1{margin:8px 0 6px;font-size:28px;line-height:1.2;letter-spacing:-.02em;font-weight:650}
header time{color:var(--muted);font-size:14px}
main{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);padding:22px 20px;overflow-wrap:anywhere}
main>:first-child{margin-top:0}main>:last-child{margin-bottom:0}
h1,h2,h3,h4{line-height:1.3;letter-spacing:-.01em}
main h1{font-size:24px;margin:1.6em 0 .5em}main h2{font-size:20px;margin:1.6em 0 .5em;padding-bottom:.35em;border-bottom:1px solid var(--line)}
main h3{font-size:17px;margin:1.4em 0 .4em}main h4{font-size:15px;margin:1.2em 0 .3em;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
p{margin:.8em 0}ul,ol{padding-left:1.35em;margin:.8em 0}li{margin:.3em 0}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid var(--accent-soft)}a:hover{border-bottom-color:var(--accent)}
hr{border:0;border-top:1px solid var(--line);margin:1.6em 0}
blockquote{margin:1em 0;padding:.2em 0 .2em 14px;border-left:3px solid var(--accent);color:var(--muted)}
code{font:13.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--accent-soft);padding:.1em .4em;border-radius:4px}
pre{font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--bg);border:1px solid var(--line);border-radius:var(--r);padding:12px 14px;overflow-x:auto}
pre code{background:none;padding:0}
table{width:100%;border-collapse:collapse;margin:1em 0;font-size:15px;display:block;overflow-x:auto}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600;white-space:nowrap}
tbody tr:hover td{background:var(--accent-soft)}
caption{caption-side:top;text-align:left;color:var(--muted);font-size:13px;padding-bottom:6px}
.num{text-align:right;font-variant-numeric:tabular-nums}
.lede{font-size:18px;color:var(--ink)}.muted,small{color:var(--muted)}
.note,.callout{border:1px solid var(--line);border-left:3px solid var(--accent);background:var(--accent-soft);border-radius:var(--r);padding:10px 14px;margin:1em 0}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border:1px solid var(--line);border-radius:999px;color:var(--muted)}
.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
footer{margin-top:20px;color:var(--muted);font-size:12px;text-align:center}
@media (max-width:480px){.wrap{padding:18px 16px 48px}header h1{font-size:23px}main{padding:16px 14px}table{font-size:14px}th,td{padding:8px 9px}}
@media print{body{background:#fff;color:#000}main{box-shadow:none;border-color:#ccc}}
"""

CSP = ("default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; script-src 'none'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'")


def _title(title: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip())[:MAX_TITLE_CHARS]
    return t or "Blatbot"


def render_page(title: str, body_html: str, created: float) -> str:
    t = _html.escape(_title(title))
    when = time.strftime("%B %-d, %Y", time.localtime(created))
    iso = time.strftime("%Y-%m-%d", time.localtime(created))
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta name=\"robots\" content=\"noindex,nofollow,noarchive\">"
        "<meta name=\"referrer\" content=\"no-referrer\">"
        f"<title>{t}</title><style>{CSS}</style></head><body><div class=\"wrap\">"
        f"<header><div class=\"brand\">Blatbot</div><h1>{t}</h1><time datetime=\"{iso}\">{when}</time></header>"
        f"<main>{body_html}</main></div></body></html>\n"
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_public_base = ""


def set_public_base(url: str) -> None:
    """The gateway's public address, told to us once it is known (tunnel up or configured)."""
    global _public_base
    _public_base = (url or "").strip().rstrip("/")


def public_base() -> str:
    return _public_base or (os.getenv("INKBOX_PUBLIC_URL") or "").strip().rstrip("/")


def pages_dir() -> Path:
    from ..config import state_dir
    root = os.getenv("GATE_PAGES_DIR")
    path = Path(root) if root else state_dir() / "pages"
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _ttl_days() -> float:
    from . import settings
    try:
        return max(1.0, float(settings.get("GATE_HTML_PAGE_DAYS") or 30))
    except Exception:
        return 30.0


def audience() -> str:
    from . import settings
    try:
        return str(settings.get("GATE_HTML_PAGES_AUDIENCE") or "owner")
    except Exception:
        return "owner"


def allowed_for(from_owner: bool) -> bool:
    """Whether a reply to this person may use a page: the owner always (unless pages are
    off), anyone else only once the setting is widened to everyone."""
    mode = audience()
    if mode == "everyone":
        return True
    return mode == "owner" and bool(from_owner)


def purge_expired(now: Optional[float] = None) -> int:
    now = time.time() if now is None else now
    removed = 0
    try:
        for meta in pages_dir().glob("*.json"):
            try:
                data = json.loads(meta.read_text("utf-8"))
                if float(data.get("expires", 0)) <= now:
                    meta.with_suffix(".html").unlink(missing_ok=True)
                    meta.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                continue
    except Exception:
        logger.exception("[pages] purge failed")
    return removed


def store_page(title: str, body_html: str, *, now: Optional[float] = None,
               ttl_days: Optional[float] = None) -> Dict[str, Any]:
    """Sanitise, wrap and write one page. Returns token, path, url and expiry."""
    now = time.time() if now is None else now
    days = _ttl_days() if ttl_days is None else float(ttl_days)
    clean = sanitize_html(body_html)
    if not re.sub(r"<[^>]+>|\s", "", clean):
        raise ValueError("the page has no content once unsafe markup is removed")
    doc = render_page(title, clean, now)
    token = secrets.token_urlsafe(32)
    d = pages_dir()
    purge_expired(now)
    expires = now + days * 86400
    (d / f"{token}.html").write_text(doc, "utf-8")
    (d / f"{token}.json").write_text(json.dumps({"title": _title(title), "created": now, "expires": expires}), "utf-8")
    path = f"{PAGE_PATH_PREFIX}{token}"
    base = public_base()
    return {"token": token, "path": path, "url": f"{base}{path}" if base else path, "expires": expires}


def delete_page(token: str) -> bool:
    if not TOKEN_RE.match(token or ""):
        return False
    d = pages_dir()
    existed = (d / f"{token}.html").exists()
    (d / f"{token}.html").unlink(missing_ok=True)
    (d / f"{token}.json").unlink(missing_ok=True)
    return existed


def load_page(token: str, *, now: Optional[float] = None) -> Optional[str]:
    """The stored document, or None for a malformed, unknown or expired token."""
    if not TOKEN_RE.match(token or ""):
        return None
    now = time.time() if now is None else now
    d = pages_dir()
    try:
        meta = json.loads((d / f"{token}.json").read_text("utf-8"))
        if float(meta.get("expires", 0)) <= now:
            (d / f"{token}.html").unlink(missing_ok=True)
            (d / f"{token}.json").unlink(missing_ok=True)
            return None
        return (d / f"{token}.html").read_text("utf-8")
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Robots-Tag": "noindex, nofollow, noarchive",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "private, no-store",
    "X-Frame-Options": "DENY",
}


async def handle_page(request: web.Request) -> web.StreamResponse:
    doc = load_page(request.match_info.get("token", ""))
    if doc is None:
        return web.Response(status=404, text="Not found", content_type="text/plain",
                            headers={"X-Robots-Tag": "noindex", "Cache-Control": "no-store"})
    return web.Response(text=doc, content_type="text/html", charset="utf-8", headers=SECURITY_HEADERS)


def register(app: web.Application) -> None:
    app.router.add_get(PAGE_PATH_PREFIX + "{token}", handle_page)


# ---------------------------------------------------------------------------
# The host tool
# ---------------------------------------------------------------------------

PUBLISH_PAGE_DESCRIPTION = (
    "MAKE A FORMATTED WEB PAGE for a long or structured answer (a report, a comparison, a table of results, "
    "a list of many items) and get back its link. Pass a short title and the page body as HTML using only "
    "h2, h3, p, ul, ol, li, table, thead, tbody, tr, th, td, a, strong, em, code, pre, blockquote, hr, br. "
    "No scripts, styles, forms, images or iframes (they are removed). Never put a secret, a password or a "
    "private note in a page. The link works for about a month. After this, your closing status is a short "
    "plain-text summary of two or three lines plus the link on its own line."
)
PUBLISH_PAGE_SCHEMA = {"type": "object", "required": ["title", "html"], "properties": {
    "title": {"type": "string", "description": "short page title, under 100 characters"},
    "html": {"type": "string", "description": "the page body as HTML, headings and tables included, no <html> or <body> wrapper"}}}

PAGE_GUIDANCE = (
    "Pages: when the answer is long (more than about 12 lines) or structured (a report, a comparison, a table, "
    "many results), call make_page with the full answer as HTML and then reply with only a short plain-text "
    "summary and the link it returns. Short answers stay plain text and never use make_page. Never put a "
    "secret or password in a page.\n"
)


async def publish_page(args: Dict[str, Any], allowed: bool) -> str:
    if not allowed:
        return "Pages are not enabled for this reply. Answer in plain text."
    title = str((args or {}).get("title") or "").strip()
    body = str((args or {}).get("html") or "")
    if not body.strip():
        return "No page was made: the html was empty."
    try:
        info = store_page(title, body)
    except ValueError as exc:
        return f"No page was made: {exc}."
    except Exception:
        logger.exception("[pages] could not store a page")
        return "No page was made: it could not be saved. Answer in plain text."
    days = max(1, int(round((info["expires"] - time.time()) / 86400)))
    return f"Page ready: {info['url']} (works for {days} days). Put this link in your reply."


def urls_in(content: Any) -> List[str]:
    """Page links found in a tool result's content (a string or a list of text blocks)."""
    if isinstance(content, list):
        text = " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(getattr(c, "text", c)) for c in content)
    else:
        text = str(content or "")
    return re.findall(r"(?:https?://[^\s)]+)?" + re.escape(PAGE_PATH_PREFIX) + r"[A-Za-z0-9_-]{32,64}", text)
