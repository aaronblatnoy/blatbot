"""dj-khaled: posts "Anotha one" to a Telegram group, as the DJ Khaled bot, every time a Migreat outreach thread gets a reply.

Watches Aaron's NYU Stern Gmail (he is cc'd on every Migreat outreach). Every POLL_SECONDS (default 60) it lists inbox
messages newer than the moment the hook first went live and, for each one not seen before, posts once if it is a reply
on a thread that includes a migreat.us address and the sender is not migreat.us, not Aaron's Stern account, and not a
bounce/auto-reply/list message. Never fires for mail that arrived before go-live (no backfill) or twice for one Message-ID.

Environment: STERN_OAUTH_JSON (authorized-user JSON with a gmail.readonly refresh token; text), DJKHALED_BOT_TOKEN,
DJKHALED_CHAT_ID, POLL_SECONDS (60), STATE_DIR (/data), MIGREAT_DOMAIN (migreat.us),
OWN_ADDRESSES (comma list, default amb10503@stern.nyu.edu,aaronblatnoy@stern.nyu.edu), DRY_RUN=1 to log instead of post.
"""
import email.utils
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dj-khaled")

DOMAIN = os.environ.get("MIGREAT_DOMAIN", "migreat.us").lower()
OWN = {a.strip().lower() for a in os.environ.get(
    "OWN_ADDRESSES", "amb10503@stern.nyu.edu,aaronblatnoy@stern.nyu.edu").split(",") if a.strip()}
POLL = int(os.environ.get("POLL_SECONDS", "60"))
STATE_DIR = os.environ.get("STATE_DIR", "/data")
DRY = os.environ.get("DRY_RUN") == "1"
TEXT = "Anotha one \U0001F3B5"
GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me/"
HEADERS = ["From", "To", "Cc", "Subject", "In-Reply-To", "References", "Message-ID", "Auto-Submitted", "Precedence",
           "List-Id", "List-Unsubscribe", "Return-Path", "X-Autoreply", "X-Autorespond", "X-Auto-Response-Suppress"]
AUTO_LOCAL = re.compile(r"^(mailer-daemon|postmaster|no-?reply|do-?not-?reply|bounces?|notifications?)([+._-].*)?$", re.I)


def addrs(value: str) -> list:
    return [a.lower() for _, a in email.utils.getaddresses([value or ""]) if "@" in a]


def is_domain(addr: str) -> bool:
    host = addr.rsplit("@", 1)[-1]
    return host == DOMAIN or host.endswith("." + DOMAIN)


def hdrs(m: dict) -> dict:
    return {h["name"].lower(): h["value"] for h in m.get("payload", {}).get("headers", [])}


def sender_ok(hd: dict) -> bool:
    """The sender is a real outside person: not migreat.us, not Aaron's Stern account, not a bot/auto sender."""
    senders = addrs(hd.get("from", ""))
    if not senders:
        return False
    a = senders[0]
    if is_domain(a) or a in OWN or AUTO_LOCAL.match(a.rsplit("@", 1)[0]):
        return False
    if hd.get("auto-submitted", "no").strip().lower() not in ("", "no"):
        return False
    if any(hd.get(k) for k in ("x-autoreply", "x-autorespond", "x-auto-response-suppress", "list-id", "list-unsubscribe")):
        return False
    if hd.get("precedence", "").strip().lower() in ("bulk", "junk", "list", "auto_reply"):
        return False
    if "return-path" in hd and re.sub(r"[\s<>]", "", hd["return-path"]) == "":
        return False
    return True


def is_reply(hd: dict) -> bool:
    return bool(hd.get("in-reply-to", "").strip() or hd.get("references", "").strip()
                or re.match(r"\s*re\s*:", hd.get("subject", ""), re.I))


def has_migreat(hd: dict) -> bool:
    return any(is_domain(a) for k in ("from", "to", "cc") for a in addrs(hd.get(k, "")))


def should_fire(m: dict, thread_msgs: list, live_ms: int) -> bool:
    if int(m["internalDate"]) < live_ms:
        return False
    if {"SENT", "DRAFT", "SPAM", "TRASH"} & set(m.get("labelIds", [])):
        return False
    hd = hdrs(m)
    return sender_ok(hd) and is_reply(hd) and any(has_migreat(hdrs(t)) for t in thread_msgs)


class App:
    def __init__(self):
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2.credentials import Credentials
        info = json.loads(os.environ["STERN_OAUTH_JSON"])
        creds = Credentials.from_authorized_user_info(info, ["https://www.googleapis.com/auth/gmail.readonly"])
        self.s = AuthorizedSession(creds)
        self.token = os.environ["DJKHALED_BOT_TOKEN"].strip()
        self.chat = os.environ["DJKHALED_CHAT_ID"].strip()
        self.db = sqlite3.connect(os.path.join(STATE_DIR, "state.db"))
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, fired INTEGER, telegram_id INTEGER, at TEXT)")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('live_ms', ?)", (str(int(time.time() * 1000)),))
        self.db.commit()
        self.live_ms = int(self.db.execute("SELECT v FROM meta WHERE k='live_ms'").fetchone()[0])

    def get(self, path: str, **params):
        for attempt in range(5):
            r = self.s.get(GMAIL + path, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def post(self) -> int:
        if DRY:
            log.info("DRY_RUN: would post %r", TEXT)
            return 0
        r = httpx.post(f"https://api.telegram.org/bot{self.token}/sendMessage", timeout=20,
                       json={"chat_id": self.chat, "text": TEXT})
        r.raise_for_status()
        return r.json()["result"]["message_id"]

    def poll(self) -> None:
        q = "in:inbox after:%d" % max(0, self.live_ms // 1000 - 3600)
        ids, token = [], None
        while True:
            p = {"q": q, "maxResults": 100}
            if token:
                p["pageToken"] = token
            d = self.get("messages", **p)
            ids += [(x["id"], x["threadId"]) for x in d.get("messages", [])]
            token = d.get("nextPageToken")
            if not token:
                break
        for gid, tid in reversed(ids):
            try:
                self.check(gid, tid)
            except Exception as exc:
                log.warning("message %s failed: %r", gid, exc)

    def check(self, gid: str, tid: str) -> None:
        m = self.get("messages/" + gid, format="metadata", metadataHeaders=HEADERS)
        key = hdrs(m).get("message-id", "").strip() or "gmail:" + gid
        if self.db.execute("SELECT 1 FROM seen WHERE key=?", (key,)).fetchone():
            return
        thread = self.get("threads/" + tid, format="metadata", metadataHeaders=HEADERS).get("messages", [m])
        fire = should_fire(m, thread, self.live_ms)
        tg = None
        if fire:
            tg = self.post()  # raises on failure, so the message is retried next poll
            log.info("Anotha one posted (telegram message_id=%s) for %s", tg, key)
        else:
            log.info("skip %s", key)
        self.db.execute("INSERT OR IGNORE INTO seen VALUES (?,?,?,?)",
                        (key, int(fire), tg, datetime.now(timezone.utc).isoformat()))
        self.db.commit()


def main() -> None:
    app = App()
    log.info("dj-khaled ARMED: watching Stern inbox, live since %s ms, poll=%ss dry=%s", app.live_ms, POLL, DRY)
    while True:
        try:
            app.poll()
        except Exception as exc:
            log.error("poll failed: %r", exc)
        open(os.path.join(STATE_DIR, "alive"), "w").write(str(time.time()))
        time.sleep(POLL)


if __name__ == "__main__":
    main()
