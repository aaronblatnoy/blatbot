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
                 command_handler: Callable[[Dict[str, Any]], Awaitable[bool]] | None = None,
                 timer_handler: Callable[[Dict[str, Any]], Awaitable[None]] | None = None,
                 external_handler: Callable[[Dict[str, Any]], Awaitable[None]] | None = None):
        self.store = store
        self.session_factory = session_factory
        self.record_id_for_event = record_id_for_event
        self.command_handler = command_handler
        self.timer_handler = timer_handler
        self.external_handler = external_handler
        self.tasks: Dict[str, asyncio.Task] = {}
        self._wakes: Dict[str, asyncio.Event] = {}
        self._closed = False

    def submit(self, chat_id: str, source: str, source_event_id: str, mode: str,
               body: str, meta: Dict[str, Any], session_factory: Callable[..., Any] | None = None,
               kind: str = "message") -> str:
        if self._closed:
            raise RuntimeError("inbox is closed")
        if session_factory is None and self.session_factory is None:
            raise RuntimeError("inbox session factory is unavailable")
        # Provider ids are the durable dedup key. An absent id gets a fresh key.
        lane = self.lane_for(chat_id, mode, meta, kind)
        event_id = self.store.insert_inbox_event(chat_id, source, source_event_id or f"missing:{time.time_ns()}",
                                                 mode, body, meta, kind=kind, lane=lane)
        if event_id is None:
            return "duplicate"
        try:
            self.wake(lane, session_factory)
        except Exception:
            # The gateway handles this delivery directly if a worker cannot start.
            self.store.finish_inbox_event(event_id, "failed", "inbox worker could not start")
            raise
        return "queued"

    def lane_for(self, chat_id: str, mode: str, meta: Dict[str, Any], kind: str = "message") -> str:
        """Which inbox an event belongs to. One inbox per person: the owner's messages
        on every surface, and timers on his thread, resolve to the same record and so
        share one worker. Anyone else's direct chats, and timers on them, share the inbox
        of the one person on record their handles name, so a person writing on two
        surfaces is still handled one event at a time. Anything that cannot be resolved
        keeps its own chat."""
        lane = chat_id
        try:
            if self.record_id_for_event is not None:
                lane = str(self.record_id_for_event({
                    "chat_id": chat_id, "mode": mode, "kind": kind,
                    "meta_json": json.dumps(meta, default=str)}) or chat_id)
        except Exception as exc:
            logger.warning("[inbox] could not resolve the inbox for %s; using the chat: %s", chat_id, exc)
            return chat_id
        if lane != chat_id or not settings.get("GATE_PERSON_INBOX"):
            return lane
        try:
            route_meta = dict(meta or {})
            route_mode = mode
            if kind != "message" or not (route_meta.get("sender") or route_meta.get("telegram_user_id")):
                # A timer carries no sender; the chat it belongs to does.
                route = self.store.thread_route(chat_id) or {}
                route_meta, route_mode = dict(route.get("meta") or {}), str(route.get("mode") or "")
            if route_meta.get("conversation_kind") == "group" or route_meta.get("group_title"):
                return lane     # a group is everyone in it, not one person
            probe = GateSession(chat_id, None)  # type: ignore[arg-type]
            probe.mode, probe.reply_meta = route_mode, route_meta
            found = self.store.resolve_handles(probe.handles())
            if found.get("person_id") and not found.get("conflict"):
                return f"person:{found['person_id']}"
        except Exception as exc:
            logger.warning("[inbox] could not find the person for %s; using the chat: %s", chat_id, exc)
        return lane

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
                # Timers and webhooks stand alone and do not wait for a message burst.
                message_head = []
                for event in pending:
                    if event.get("kind", "message") != "message":
                        break
                    message_head.append(event)
                settle_until = message_head[-1]["created_at"] + settle if message_head else 0.0
                delay = max(0.0, settle_until - time.time())
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
                message_head = []
                for event in pending:
                    if event.get("kind", "message") != "message":
                        break
                    message_head.append(event)
                if message_head and message_head[-1]["created_at"] + settle > time.time():
                    continue
                batch = []
                for event in pending:
                    if event["attempts"] > 3:
                        self.store.finish_inbox_event(event["id"], "failed", "too many attempts")
                    else:
                        batch.append(event)
                groups = []
                for event in batch:
                    if event.get("kind", "message") != "message":
                        groups.append(([event], True))
                        continue
                    try:
                        command = (bool(json.loads(event["meta_json"]).get("_inbox_mode_command"))
                                   or is_inbox_command(event["body"]))
                    except Exception as exc:
                        logger.warning("[inbox] command check failed; handling normally: %s", exc)
                        command = False
                    if (not groups or command or groups[-1][1] or
                            groups[-1][0][-1]["mode"] != event["mode"] or
                            groups[-1][0][-1]["chat_id"] != event["chat_id"]):
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
                        kind = newest.get("kind", "message")
                        if kind == "timer":
                            if self.timer_handler is None:
                                raise RuntimeError("timer handler is unavailable")
                            await self.timer_handler(newest)
                        elif kind == "external":
                            if self.external_handler is None:
                                raise RuntimeError("external handler is unavailable")
                            await self.external_handler(newest)
                        else:
                            handled = bool(self.command_handler and meta.get("_inbox_mode_command")
                                           and await self.command_handler(newest))
                            if not handled:
                                # The lane may hold several of this person's chats; a turn
                                # runs in, and replies to, the chat its own events came from.
                                session = self.session_factory(newest["chat_id"], newest["mode"], meta)
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
            self.wake(event.get("lane") or event["chat_id"])

    async def close(self) -> None:
        self._closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
