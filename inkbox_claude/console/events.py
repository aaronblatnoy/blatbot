"""Small in-process server-sent-event fanout for console refresh hints."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Set

from aiohttp import web

from . import access

_QUEUE_SIZE = 32
_HEARTBEAT_SECONDS = 20
_subscribers: Set[asyncio.Queue[Dict[str, Any]]] = set()


def publish(kind: str, payload: Any) -> None:
    """Publish a refresh hint, evicting subscribers that cannot keep up."""
    event = {"kind": str(kind), "payload": payload}
    slow = []
    for queue in tuple(_subscribers):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            slow.append(queue)
    for queue in slow:
        _subscribers.discard(queue)


async def event_stream(request: web.Request) -> web.StreamResponse:
    """Hold one bounded subscription open and heartbeat it through proxies.

    CORS headers for this endpoint must be set here, before `prepare()`, rather than
    left to the `cors_gate` middleware: once this streaming response is prepared its
    headers are already flushed to the client, so the middleware's own
    `resp.headers[...] = ...` after `await handler(request)` returns has no effect for
    a request that actually opened the stream (it only helped the OPTIONS preflight).
    A browser EventSource for an allowed cross-origin frontend would otherwise be
    blocked by CORS even though the frontend's origin is on the allow-list.
    """
    origin = request.headers.get("Origin", "")
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    if origin and access.origin_allowed(origin):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
    response = web.StreamResponse(status=200, headers=headers)
    await response.prepare(request)
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_SIZE)
    _subscribers.add(queue)
    try:
        await response.write(b": connected\n\n")
        while queue in _subscribers:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                await response.write(b": heartbeat\n\n")
                continue
            body = json.dumps(event["payload"], ensure_ascii=False, separators=(",", ":"))
            frame = f"event: {event['kind']}\ndata: {body}\n\n".encode("utf-8")
            await response.write(frame)
    except (ConnectionResetError, BrokenPipeError, RuntimeError, asyncio.CancelledError):
        pass
    finally:
        _subscribers.discard(queue)
    return response
