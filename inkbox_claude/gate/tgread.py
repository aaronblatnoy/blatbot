"""Read-only reads of Aaron's own Telegram account (his Telethon user session),
for a gate run holding the `telegram_read` scope.

This module never logs in, sends, edits, deletes or marks anything read. The
real work happens in the existing standalone client at
`~/blatbot/telegram-user/server.py` (owned outside this package; not edited
here). That module's own `read_chat` tool formats a reply for a person to read
and truncates each message to 400 characters -- this reader must never cut
text -- so instead of calling its TOOLS registry, the subprocess transport
below imports server.py and calls its `client()`/`_entity()`/`_kind()` helpers
directly, the same way server.py's own functions do, just without the
truncation. No send/edit/delete/mark-read call is ever made or even available
through the three functions here.

Session safety: Telethon's sqlite session file (`aaron.session`) must not be
opened by two processes at once. server.py's own `cli.py` already runs as a
fresh subprocess per call -- connect, one read, exit -- so this transport does
the same thing rather than holding a second long-lived client open next to a
live server.py process. The lock below only serializes this process's own
concurrent calls; it is not a cross-process guarantee (see the report for
what a live rollout should check)."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_TIMEOUT_SECONDS = 30

TELEGRAM_USER_DIR = Path(os.getenv("BLATBOT_TELEGRAM_USER_DIR") or (Path.home() / "blatbot" / "telegram-user"))


class ReaderError(Exception):
    """A plain, expected failure of the Telegram reader (down, timed out, not
    set up, chat not found) -- callers turn this into an {"error": ...} dict,
    never an uncaught traceback."""


class Transport:
    """What find_chats/list_chats/read_chat need from a Telegram reader.
    Strictly read-only: no method here sends, edits, deletes, marks read, or
    logs in. Production uses SubprocessTransport (below); tests supply a fake
    that raises on any method outside this contract."""

    async def find(self, query: str, limit: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def list(self, limit: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def read(self, chat: Any, since: Optional[float], until: Optional[float],
                    limit: int) -> Dict[str, Any]:
        raise NotImplementedError


_transport: Optional[Transport] = None


def set_transport(t: Optional[Transport]) -> None:
    """Tests inject a fake here. Left unset in production: importing this
    module never touches Telegram, and the real transport is only built the
    first time a call actually needs it."""
    global _transport
    _transport = t


def _get_transport() -> Transport:
    global _transport
    if _transport is None:
        _transport = SubprocessTransport()
    return _transport


def _parse_when(v: Optional[str]) -> Optional[float]:
    if not v:
        return None
    dt = datetime.fromisoformat(str(v))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


async def _call(coro: Any) -> Any:
    try:
        return await asyncio.wait_for(coro, DEFAULT_TIMEOUT_SECONDS)
    except ReaderError:
        raise
    except asyncio.TimeoutError:
        raise ReaderError("the Telegram reader timed out")
    except Exception as exc:  # noqa: BLE001 -- any transport failure becomes a plain error
        raise ReaderError(f"the Telegram reader is unavailable: {type(exc).__name__}: {exc}")


async def find_chats(query: str, limit: int = 10) -> Dict[str, Any]:
    """Chats (groups, channels, people) matching `query` by name. Read-only."""
    try:
        rows = await _call(_get_transport().find(str(query or ""), int(limit or 10)))
    except ReaderError as exc:
        return {"error": str(exc)}
    return {"chats": list(rows)[: int(limit or 10)]}


async def list_chats(limit: int = 30) -> Dict[str, Any]:
    """The most recently active chats, read-only."""
    try:
        rows = await _call(_get_transport().list(int(limit or 30)))
    except ReaderError as exc:
        return {"error": str(exc)}
    return {"chats": list(rows)[: int(limit or 30)]}


async def read_chat(chat: Any, since: Optional[str] = None, until: Optional[str] = None,
                     limit: int = 200, tz: str = DEFAULT_TIMEZONE) -> Dict[str, Any]:
    """Messages in one chat, oldest first, windowed by `since`/`until` (ISO
    datetimes). When more than `limit` messages exist in the window,
    `truncated` is set true with the count still in the window; page by
    calling again with `since` set to the last message's `at`. Text is never
    cut. `tz` is the owner's time zone to stamp each message's `at` with."""
    try:
        since_ts = _parse_when(since)
        until_ts = _parse_when(until)
    except ValueError:
        return {"error": "since and until must be ISO datetimes"}
    try:
        raw = await _call(_get_transport().read(chat, since_ts, until_ts, int(limit or 200)))
    except ReaderError as exc:
        return {"error": str(exc)}
    if isinstance(raw, dict) and raw.get("error"):
        return {"error": str(raw["error"])}
    try:
        zone = ZoneInfo(tz)
    except Exception:
        zone = ZoneInfo(DEFAULT_TIMEZONE)
    all_msgs = sorted((raw.get("messages") or []), key=lambda m: float(m["at"]))
    total = len(all_msgs)
    want = int(limit or 200)
    shown = all_msgs[:want]
    messages = []
    for m in shown:
        at = datetime.fromtimestamp(float(m["at"]), tz=timezone.utc).astimezone(zone).isoformat()
        messages.append({
            "at": at,
            "sender_name": m.get("sender_name") or "?",
            "text": m.get("text") or "",
            "has_media": bool(m.get("has_media")),
        })
    return {
        "chat": raw.get("chat") or {},
        "messages": messages,
        "truncated": total > len(shown),
        "total_in_window": total,
    }


# ---------------------------------------------------------------------------
# Production transport: a fresh subprocess per call into telegram-user's own
# venv, calling server.py's connect/resolve helpers directly (never its TOOLS
# registry, which truncates read_chat's text). Never started unless a call
# actually asks for one -- importing this module touches nothing.
# ---------------------------------------------------------------------------

_FIND_SCRIPT = r"""
import sys, json, asyncio
import server

_KIND = {"dm": "person", "bot": "person", "group": "group", "channel": "channel"}

async def main():
    args = json.loads(sys.argv[1])
    c = await server.client()
    q = str(args.get("query") or "").lower()
    limit = int(args.get("limit") or 10)
    out = []
    for archived in (False, True):
        async for d in c.iter_dialogs(archived=archived):
            if q and q not in (d.name or "").lower():
                continue
            out.append({
                "id": d.id,
                "title": d.name or "",
                "kind": _KIND.get(server._kind(d), "person"),
                "participants": None,
                "last_message_at": d.date.timestamp() if d.date else None,
            })
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    print(json.dumps({"chats": out}))

asyncio.run(main())
"""

_LIST_SCRIPT = r"""
import sys, json, asyncio
import server

_KIND = {"dm": "person", "bot": "person", "group": "group", "channel": "channel"}

async def main():
    args = json.loads(sys.argv[1])
    c = await server.client()
    limit = int(args.get("limit") or 30)
    out = []
    async for d in c.iter_dialogs(limit=limit):
        out.append({
            "id": d.id,
            "title": d.name or "",
            "kind": _KIND.get(server._kind(d), "person"),
            "participants": None,
            "last_message_at": d.date.timestamp() if d.date else None,
        })
    print(json.dumps({"chats": out}))

asyncio.run(main())
"""

_READ_SCRIPT = r"""
import sys, json, asyncio, datetime
import server

_KIND = {"dm": "person", "bot": "person", "group": "group", "channel": "channel"}

async def main():
    args = json.loads(sys.argv[1])
    c = await server.client()
    chat = args["chat"]
    since = args.get("since")
    until = args.get("until")
    limit = int(args.get("limit") or 200)
    ent = await server._entity(c, chat)
    title, kind = str(chat), "person"
    async for d in c.iter_dialogs():
        if await c.get_input_entity(d.entity) == ent or str(d.id) == str(chat):
            title, kind = d.name or title, _KIND.get(server._kind(d), "person")
            break
    offset_date = (datetime.datetime.fromtimestamp(float(until), tz=datetime.timezone.utc)
                   if until else None)
    msgs = []
    async for m in c.iter_messages(ent, limit=max(limit * 4, limit + 1), offset_date=offset_date):
        ts = m.date.timestamp()
        if since and ts < float(since):
            break
        who = (getattr(m.sender, "first_name", None) or getattr(m.sender, "title", None)
               or getattr(m.sender, "username", None) or "?")
        msgs.append({"at": ts, "sender_name": who, "text": m.text or m.message or "",
                     "has_media": bool(m.media)})
        if len(msgs) >= limit + 1:
            break
    print(json.dumps({"chat": {"id": chat, "title": title, "kind": kind}, "messages": msgs}))

asyncio.run(main())
"""


class SubprocessTransport(Transport):
    def __init__(self, telegram_user_dir: Optional[Path] = None):
        self.dir = telegram_user_dir or TELEGRAM_USER_DIR
        self.python = self.dir / ".venv" / "bin" / "python"
        self._lock = asyncio.Lock()

    def _env(self) -> Dict[str, str]:
        """telegram-user's own .env (API id/hash/session path) merged over this
        process's environment. Never printed or logged, never ~/blatbot/.env."""
        env = dict(os.environ)
        try:
            for line in (self.dir / ".env").read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
        except OSError:
            pass
        return env

    async def _run(self, script: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.python.exists():
            raise ReaderError("the Telegram reader is not set up on this server")
        async with self._lock:
            proc = await asyncio.create_subprocess_exec(
                str(self.python), "-c", script, json.dumps(args),
                cwd=str(self.dir), env=self._env(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), DEFAULT_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                proc.kill()
                raise ReaderError("the Telegram reader timed out")
        if proc.returncode != 0:
            msg = err.decode("utf-8", "replace").strip()[:200] or "unknown error"
            raise ReaderError(f"the Telegram reader is unavailable: {msg}")
        try:
            return json.loads(out.decode("utf-8", "replace"))
        except ValueError:
            raise ReaderError("the Telegram reader returned something unreadable")

    async def find(self, query: str, limit: int) -> List[Dict[str, Any]]:
        data = await self._run(_FIND_SCRIPT, {"query": query, "limit": limit})
        return list(data.get("chats") or [])

    async def list(self, limit: int) -> List[Dict[str, Any]]:
        data = await self._run(_LIST_SCRIPT, {"limit": limit})
        return list(data.get("chats") or [])

    async def read(self, chat: Any, since: Optional[float], until: Optional[float],
                    limit: int) -> Dict[str, Any]:
        return await self._run(_READ_SCRIPT, {"chat": chat, "since": since, "until": until, "limit": limit})
