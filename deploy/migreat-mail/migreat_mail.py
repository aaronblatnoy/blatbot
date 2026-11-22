"""migreat-mail: funnels every migreat.us Workspace mailbox into the Definite Optimists brain.

Every POLL_SECONDS (default 300): list the domain's users (Admin SDK, impersonating DIRECTORY_ADMIN), and for each
user read new Gmail messages with a domain-wide-delegated service account (gmail.readonly) and write each one to the
brain (POST /api/add, brain agent key `migreat-mail`) as one episode. First run backfills the whole mailbox.
New users are picked up automatically on the next poll (their whole mailbox is backfilled).

State (SQLite, /data/state.db): one row per message already written, keyed by RFC Message-ID (falls back to
mailbox:gmail-id), so a message that sits in two mailboxes is written once. Nothing is stored in git.

Environment: SA_JSON (service account key JSON text) or SA_FILE, DO_BRAIN_KEY, DO_BRAIN_API_URL (optional),
WORKSPACE_DOMAIN (migreat.us), DIRECTORY_ADMIN (aaron@migreat.us), POLL_SECONDS, MAX_BODY (8000),
BRAIN_PACE_SECONDS (0.4 pause between writes), STATE_DIR (/data), DRY_RUN=1 to only count.
"""
import base64
import email.utils
import html
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

import httpx
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("migreat-mail")

DOMAIN = os.environ.get("WORKSPACE_DOMAIN", "migreat.us")
ADMIN = os.environ.get("DIRECTORY_ADMIN", "aaron@migreat.us")
POLL = int(os.environ.get("POLL_SECONDS", "300"))
MAX_BODY = int(os.environ.get("MAX_BODY", "8000"))
PACE = float(os.environ.get("BRAIN_PACE_SECONDS", "0.4"))
STATE_DIR = os.environ.get("STATE_DIR", "/data")
DRY = os.environ.get("DRY_RUN") == "1"
BRAIN = (os.environ.get("DO_BRAIN_API_URL") or "https://definite-optimists-brain.aaronblatnoy.workers.dev").rstrip("/")
BRAIN_HEADERS = {"Authorization": "Bearer " + os.environ.get("DO_BRAIN_KEY", "").strip(),
                 "User-Agent": "migreat-mail", "Content-Type": "application/json"}
GMAIL = "https://gmail.googleapis.com/gmail/v1/users/"
DIRECTORY = "https://admin.googleapis.com/admin/directory/v1/users"
SA_INFO = json.loads(os.environ["SA_JSON"]) if os.environ.get("SA_JSON") else json.load(open(os.environ["SA_FILE"]))

db = sqlite3.connect(os.path.join(STATE_DIR, "state.db"))
db.execute("CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, mailbox TEXT, at TEXT)")
db.execute("CREATE TABLE IF NOT EXISTS mailbox (user TEXT PRIMARY KEY, last_ms INTEGER, backfilled INTEGER DEFAULT 0)")
db.commit()

_sessions = {}


def session(subject: str, scopes: list) -> AuthorizedSession:
    key = (subject, tuple(scopes))
    if key not in _sessions:
        creds = service_account.Credentials.from_service_account_info(SA_INFO, scopes=scopes, subject=subject)
        _sessions[key] = AuthorizedSession(creds)
    return _sessions[key]


def get(s: AuthorizedSession, url: str, **params):
    for attempt in range(5):
        r = s.get(url, params=params, timeout=60)
        if r.status_code in (429, 500, 502, 503) or (r.status_code == 403 and "rateLimit" in r.text):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def domain_users() -> list:
    s = session(ADMIN, ["https://www.googleapis.com/auth/admin.directory.user.readonly"])
    out, token = [], None
    while True:
        p = {"domain": DOMAIN, "maxResults": 200, "showDeleted": "false"}
        if token:
            p["pageToken"] = token
        d = get(s, DIRECTORY, **p)
        out += [u["primaryEmail"] for u in d.get("users", []) if not u.get("suspended")]
        token = d.get("nextPageToken")
        if not token:
            return out


def strip_html(h: str) -> str:
    h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h)
    h = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", h)
    return html.unescape(re.sub(r"<[^>]+>", " ", h))


def body_text(payload: dict) -> tuple:
    plain, htm, files = [], [], []

    def walk(p):
        mt = p.get("mimeType", "")
        data = (p.get("body") or {}).get("data")
        if p.get("filename"):
            files.append(p["filename"])
        elif data and mt == "text/plain":
            plain.append(base64.urlsafe_b64decode(data + "==").decode("utf-8", "replace"))
        elif data and mt == "text/html":
            htm.append(base64.urlsafe_b64decode(data + "==").decode("utf-8", "replace"))
        for c in p.get("parts") or []:
            walk(c)

    walk(payload)
    text = "\n".join(plain) if plain else strip_html("\n".join(htm))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text, files


def brain_add(content: str, title: str) -> None:
    for attempt in range(6):
        try:
            r = httpx.post(BRAIN + "/api/add", headers=BRAIN_HEADERS, timeout=60,
                           json={"content": content, "title": title[:120], "source": "migreat-mail"})
            if r.status_code in (429, 500, 502, 503, 524):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return
        except httpx.TransportError:
            time.sleep(2 ** attempt)
    raise RuntimeError("brain write failed after retries")


def ingest_message(user: str, m: dict) -> str:
    """returns 'written' or 'dup'"""
    hd = {h["name"].lower(): h["value"] for h in m["payload"].get("headers", [])}
    key = (hd.get("message-id") or "").strip() or f"{user}:{m['id']}"
    if db.execute("SELECT 1 FROM seen WHERE key=?", (key,)).fetchone():
        return "dup"
    text, files = body_text(m["payload"])
    date = hd.get("date", "")
    try:
        date = email.utils.parsedate_to_datetime(date).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        date = datetime.fromtimestamp(int(m["internalDate"]) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = hd.get("subject", "(no subject)")
    head = [f"Email in the {user} mailbox ({DOMAIN}).", f"Date: {date}", f"From: {hd.get('from', '')}",
            f"To: {hd.get('to', '')}"]
    if hd.get("cc"):
        head.append(f"Cc: {hd['cc']}")
    head.append(f"Subject: {subject}")
    if files:
        head.append("Attachments: " + ", ".join(files[:20]))
    content = "\n".join(head) + "\n\n" + text[:MAX_BODY]
    if not DRY:
        brain_add(content, f"Email {user}: {subject}")
        time.sleep(PACE)
    if not DRY:
        db.execute("INSERT OR IGNORE INTO seen VALUES (?,?,?)", (key, user, datetime.now(timezone.utc).isoformat()))
        db.commit()
    return "written"


def poll_user(user: str) -> dict:
    s = session(user, ["https://www.googleapis.com/auth/gmail.readonly"])
    row = db.execute("SELECT last_ms, backfilled FROM mailbox WHERE user=?", (user,)).fetchone()
    q = ""
    if row and row[1]:
        # one day of overlap; the seen table makes re-reading harmless
        q = "after:%d" % max(0, int(row[0] / 1000) - 86400)
    ids, token = [], None
    while True:
        p = {"maxResults": 500, "includeSpamTrash": "false"}
        if q:
            p["q"] = q
        if token:
            p["pageToken"] = token
        d = get(s, GMAIL + user + "/messages", **p)
        ids += [x["id"] for x in d.get("messages", [])]
        token = d.get("nextPageToken")
        if not token:
            break
    stats = {"listed": len(ids), "written": 0, "dup": 0, "errors": 0}
    newest = row[0] if row else 0
    for mid in reversed(ids):  # oldest first
        try:
            m = get(s, GMAIL + user + "/messages/" + mid, format="full")
            newest = max(newest, int(m["internalDate"]))
            stats[ingest_message(user, m)] += 1
        except Exception as exc:
            stats["errors"] += 1
            log.warning("%s message %s failed: %r", user, mid, exc)
    if stats["errors"] == 0 and not DRY:
        db.execute("INSERT OR REPLACE INTO mailbox VALUES (?,?,1)", (user, newest))
        db.commit()
    return stats


def main() -> None:
    log.info("migreat-mail starting: domain=%s admin=%s poll=%ss dry=%s", DOMAIN, ADMIN, POLL, DRY)
    while True:
        try:
            users = domain_users()
            log.info("mailboxes: %s", users)
            for u in users:
                st = poll_user(u)
                log.info("%s: %s", u, st)
        except Exception as exc:
            log.error("poll failed: %r", exc)
        open(os.path.join(STATE_DIR, "alive"), "w").write(str(time.time()))
        if DRY:
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
