"""Rehearsal-only console backend: mounts the REAL routes.register() against a
REAL Store (sqlite or postgresql://), with NO seeding (so it reflects whatever
the Store already holds), and a gateway stand-in that cannot send or receive
any real message, open the Inkbox tunnel, register a webhook, or run
schedules -- only ScheduleService exists (routes.py reaches schedules through
`gateway.sessions.schedules`), and send_to_approver is a no-op.

Binds 127.0.0.1 only. Never call a mutating endpoint against this process;
it is for GET-only comparison between two snapshots of the same data.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

REPO = Path(os.environ.get("REHEARSAL_BACKEND_REPO", str(Path.home() / "blatbot" / "worktrees" / "pg-migration")))
sys.path.insert(0, str(REPO))

from aiohttp import web  # noqa: E402

from inkbox_claude.console import routes  # noqa: E402
from inkbox_claude.gate.schedules import ScheduleService  # noqa: E402
from inkbox_claude.gate.store import Store  # noqa: E402


class _NoDirectoryClient:
    """Never calls the real Inkbox API."""

    def imessage_conversations(self) -> Iterable[Dict[str, Any]]:
        return []

    def sms_threads(self) -> Iterable[Dict[str, Any]]:
        return []

    def email_threads(self) -> Iterable[Dict[str, Any]]:
        return []

    def calls(self) -> Iterable[Dict[str, Any]]:
        return []

    def contacts(self) -> Iterable[Dict[str, Any]]:
        return []


class _ReadOnlyIntentSessionManager:
    """Only .store and .schedules are read by the GET routes this rehearsal
    is used for. decide_request/send_to_approver exist so routes.py does not
    crash if something unexpected calls them, but they never run anything:
    this harness opens no messaging channel, webhook, tunnel, or scheduler
    loop at all -- ScheduleService.propose()/action() are also never called
    here, only GET /console/api/schedules (a plain read)."""

    def __init__(self, store: Store) -> None:
        self.store = store

    async def decide_request(self, *a, **kw) -> Dict[str, Any]:
        return {"ok": False, "error": "rehearsal instance: mutations are disabled", "state": ""}

    async def send_to_approver(self, text: str) -> None:
        return None


class RehearsalGateway:
    def __init__(self, store: Store) -> None:
        self.sessions = _ReadOnlyIntentSessionManager(store)
        self.sessions.schedules = ScheduleService(self.sessions)
        self.inkbox_directory_client = _NoDirectoryClient()
        self._console_started_at = time.time()
        self._public_url = ""  # no tunnel
        self._tunnel = None


async def run(host: str, port: int, db_path: str) -> None:
    os.environ["CONSOLE_ALLOWED_ORIGINS"] = ""
    store = Store(db_path)
    gateway = RehearsalGateway(store)

    app = web.Application()
    routes.register(app, gateway)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    # Never print db_path: for a postgresql:// target it carries the
    # password in plain text. "sqlite" / "postgres" is all a log needs.
    kind = "postgres" if db_path.startswith(("postgresql://", "postgres://")) else "sqlite"
    print(f"READY rehearsal backend http://{host}:{port} target_kind={kind}", flush=True)
    while True:
        await asyncio.sleep(3600)


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8794
    db_path = sys.argv[3]
    asyncio.run(run(host, port, db_path))


if __name__ == "__main__":
    main()
