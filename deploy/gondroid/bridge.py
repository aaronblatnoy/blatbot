"""Gondroid bridge: gives Gondroid the messages and files other bots post in its groups.

Telegram's Bot API never delivers one bot's messages to another bot, so Gondroid cannot see what Samwise or Blatbot
post. This runs on black-sky beside the telegram-user tools, signed in as a person (the same Telethon session), and
only READS: every few seconds it lists new messages in each group Gondroid keeps a transcript for, takes the ones
sent by bots other than Gondroid, and writes each (with its file) into the Gondroid container's /data/inbox, where
gondroid.py's inbox_loop keeps it. It never sends, marks read, or changes anything in Telegram.

Run: systemd user unit gondroid-bridge (env from ~/blatbot/telegram-user/.env).

Gondroid as a gateway instance (deploy/instances/gondroid.yaml) runs on the host, not in a container:
set BRIDGE_LEDGER_DIR to its ledger state folder (~/gondroid/ledger) and the bridge reads the trusted
groups from chats.json there and writes into inbox/ there directly. Unset, it uses the container.
"""
import asyncio, json, logging, os, subprocess, time
from datetime import datetime, timedelta, timezone

from telethon import TelegramClient

SESSION = os.path.expanduser(os.getenv("TELEGRAM_SESSION") or "~/blatbot/telegram-user/aaron.session")
STATE = os.path.expanduser("~/gondroid-bridge/state.json")
MAX_BYTES = 50 * 1024 * 1024
SKIP = "gondroid_bot"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bridge")


def sh(args, data=None) -> str:
    return subprocess.run(args, input=data, capture_output=True, check=True, timeout=120).stdout.decode()


LEDGER_DIR = os.path.expanduser(os.getenv("BRIDGE_LEDGER_DIR") or "")


def container() -> str:
    if LEDGER_DIR:
        return "host"
    return (sh(["docker", "ps", "--filter", "name=gondroid-", "--format", "{{.Names}}"]).split() or [""])[0]


def put(c: str, name: str, data: bytes) -> None:
    if LEDGER_DIR:
        d = os.path.join(LEDGER_DIR, "inbox")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name + ".tmp"), "wb") as f:
            f.write(data)
        os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))
        return
    sh(["docker", "exec", "-i", c, "sh", "-c", f'mkdir -p /data/inbox && cat > "/data/inbox/{name}.tmp" && mv "/data/inbox/{name}.tmp" "/data/inbox/{name}"'], data)


def group_chats(c: str) -> list:
    """The groups to read: the ones Gondroid keeps a transcript for (negative ids)."""
    if LEDGER_DIR:
        names = os.listdir(os.path.join(LEDGER_DIR, "transcripts")) if os.path.isdir(os.path.join(LEDGER_DIR, "transcripts")) else []
    else:
        names = sh(["docker", "exec", c, "ls", "/data/transcripts"]).split() if c else []
    return [int(n[:-6]) for n in names if n.endswith(".jsonl") and n.startswith("-")]


async def main() -> None:
    client = TelegramClient(SESSION, int(os.environ["TELEGRAM_API_ID"]), os.environ["TELEGRAM_API_HASH"], receive_updates=False)
    await client.connect()
    if not await client.is_user_authorized():
        raise SystemExit("not logged in")
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {}
    bots, titles = {}, {}
    log.info("bridge up")
    while True:
        try:
            c = container()
            chats = group_chats(c)
            for chat in chats:
                key, last = str(chat), state.get(str(chat))
                if last is None:   # first time: the last two days
                    since = datetime.now(timezone.utc) - timedelta(days=2)
                    msgs = [m async for m in client.iter_messages(chat, limit=600) if m.date >= since]
                else:
                    msgs = [m async for m in client.iter_messages(chat, min_id=last, limit=200)]
                if key not in titles:
                    titles[key] = getattr(await client.get_entity(chat), "title", key)
                for m in sorted(msgs, key=lambda x: x.id):
                    sid = m.sender_id
                    if sid not in bots:
                        s = await m.get_sender()
                        bots[sid] = (bool(getattr(s, "bot", False)), " ".join(x for x in (getattr(s, "first_name", None), getattr(s, "last_name", None)) if x),
                                     getattr(s, "username", None) or "")
                    is_bot, name, username = bots[sid]
                    if is_bot and username.lower() != SKIP:
                        item = {"chat_id": chat, "chat_title": titles[key], "uid": m.id, "from_id": sid, "name": name or username or "a bot",
                                "username": username, "date": int(m.date.timestamp()), "text": m.message or ""}
                        f, mime = m.file, (m.file.mime_type or "") if m.file else ""
                        if f and not mime.startswith(("audio/", "video/")) and not m.sticker and (f.size or 0) <= MAX_BYTES:
                            data = await m.download_media(bytes)
                            if data:
                                item["file_name"] = f.name or (f"photo-u{m.id}.jpg" if m.photo else f"file-u{m.id}{f.ext or ''}")
                                item["blob"] = f"{chat}_{m.id:012d}.bin"
                                put(c, item["blob"], data)
                        if item["text"] or item.get("blob"):
                            put(c, f"{chat}_{m.id:012d}.json", json.dumps(item).encode())
                            log.info("passed message %s in %s from %s%s", m.id, chat, username or name, " with a file" if item.get("blob") else "")
                    state[key] = m.id
                    os.makedirs(os.path.dirname(STATE), exist_ok=True)
                    json.dump(state, open(STATE + ".tmp", "w"))
                    os.replace(STATE + ".tmp", STATE)
                if last is None and not msgs:
                    state[key] = 0
        except Exception as exc:
            log.error("pass failed: %s %s", type(exc).__name__, str(exc)[:300])
            await asyncio.sleep(20)
        await asyncio.sleep(5)


asyncio.run(main())
