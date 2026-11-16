"""Durable inbound events, drained one conversation at a time."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict

from .manager import GateSession, is_inbox_command
from . import settings

logger = logging.getLogger(__name__)


class Inbox:
    def __init__(self, store: Any, session_factory: Callable[..., Any] | None = None,
                 record_id_for_event: Callable[[Dict[str, Any]], str] | None = None,
                 command_handler: Callable[[Dict[str, Any]], Awaitable[bool]] | None = None):
        self.store = store
        self.session_factory = session_factory
        self.record_id_for_event = record_id_for_event
        self.command_handler = command_handler
        self.tasks: Dict[str, asyncio.Task] = {}
        self._wakes: Dict[str, asyncio.Event] = {}
        self._closed = False

    def submit(self, chat_id: str, source: str, source_event_id: str, mode: str,
               body: str, meta: Dict[str, Any], session_factory: Callable[..., Any] | None = None) -> str:
        if self._closed:
            raise RuntimeError("inbox is closed")
        if session_factory is None and self.session_factory is None:
            raise RuntimeError("inbox session factory is unavailable")
        # Provider ids are the durable dedup key. An absent id gets a fresh key.
        event_id = self.store.insert_inbox_event(chat_id, source, source_event_id or f"missing:{time.time_ns()}",
                                                 mode, body, meta)
        if event_id is None:
            return "duplicate"
        try:
            self.wake(chat_id, session_factory)
        except Exception:
            # The gateway handles this delivery directly if a worker cannot start.
            self.store.finish_inbox_event(event_id, "failed", "inbox worker could not start")
            raise
        return "queued"

    def wake(self, chat_id: str, session_factory: Callable[..., Any] | None = None) -> None:
        if session_factory is not None:
            self.session_factory = session_factory
        if self.session_factory is None:
            raise RuntimeError("inbox session factory is unavailable")
        event = self._wakes.setdefault(chat_id, asyncio.Event())
        event.set()
        task = self.tasks.get(chat_id)
        if task is None or task.done():
            self.tasks[chat_id] = asyncio.create_task(self._drain(chat_id), name=f"inbox-{chat_id}")

    async def _drain(self, chat_id: str) -> None:
        healthy = True
        try:
            while True:
                pending = self.store.list_pending_inbox(chat_id)
                if not pending:
                    return
                # A fresh message extends the quiet period; later arrivals during a
                # turn stay pending for the next pass.
                try:
                    settle = max(0.0, float(settings.get("INBOX_SETTLE_SECONDS")))
                except Exception as exc:
                    logger.warning("[inbox] settle setting unavailable; using 1 second: %s", exc)
                    settle = 1.0
                delay = max(0.0, pending[-1]["created_at"] + settle - time.time())
                if delay:
                    wake = self._wakes[chat_id]
                    wake.clear()
                    try:
                        await asyncio.wait_for(wake.wait(), timeout=delay)
                        continue
                    except asyncio.TimeoutError:
                        pass
                pending = self.store.list_pending_inbox(chat_id)
                if not pending:
                    continue
                if pending[-1]["created_at"] + settle > time.time():
                    continue
                batch = []
                for event in pending:
                    if event["attempts"] > 3:
                        self.store.finish_inbox_event(event["id"], "failed", "too many attempts")
                    else:
                        batch.append(event)
                groups = []
                for event in batch:
                    try:
                        command = (bool(json.loads(event["meta_json"]).get("_inbox_mode_command"))
                                   or is_inbox_command(event["body"]))
                    except Exception as exc:
                        logger.warning("[inbox] command check failed; handling normally: %s", exc)
                        command = False
                    if (not groups or command or groups[-1][1] or
                            groups[-1][0][-1]["mode"] != event["mode"]):
                        groups.append(([event], command))
                    else:
                        groups[-1][0].append(event)
                for events, _command in groups:
                    # Claim only the turn about to run. A crash after an earlier
                    # reply must leave later turns pending for recovery.
                    claimed = [event for event in events if self.store.claim_inbox_event(event["id"])]
                    if not claimed:
                        continue
                    events = claimed
                    newest = events[-1]
                    try:
                        meta = json.loads(newest["meta_json"])
                        body = "\n\n".join(event["body"] for event in events)
                        handled = bool(self.command_handler and meta.get("_inbox_mode_command")
                                       and await self.command_handler(newest))
                        if not handled:
                            session = self.session_factory(chat_id, newest["mode"], meta)
                            if isinstance(session, GateSession):
                                await session.handle_inbound(body, newest["mode"], meta, raise_errors=True)
                            else:
                                await session.handle_inbound(body, newest["mode"], meta)
                    except Exception as exc:
                        logger.warning("[inbox] chat %s turn failed: %s", chat_id, exc)
                        for event in events:
                            self.store.finish_inbox_event(event["id"], "failed", str(exc))
                    else:
                        for event in events:
                            self.store.finish_inbox_event(event["id"], "done")
        except asyncio.CancelledError:
            healthy = False
            raise
        except Exception as exc:
            healthy = False
            logger.warning("[inbox] chat %s worker stopped: %s", chat_id, exc)
        finally:
            if self.tasks.get(chat_id) is asyncio.current_task():
                self.tasks.pop(chat_id, None)
                # A submit may have arrived while the worker was finishing.
                if healthy and not self._closed:
                    try:
                        if self.store.list_pending_inbox(chat_id):
                            self.wake(chat_id)
                    except Exception as exc:
                        logger.warning("[inbox] chat %s restart failed: %s", chat_id, exc)

    def recover(self) -> None:
        for event in self.store.recover_inbox_events(self.record_id_for_event):
            logger.info("[inbox] replaying event %s in chat %s", event["id"], event["chat_id"])
            self.wake(event["chat_id"])

    async def close(self) -> None:
        self._closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
