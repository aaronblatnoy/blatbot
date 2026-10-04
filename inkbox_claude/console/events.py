"""Small in-process server-sent-event fanout for console refresh hints."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Set

from aiohttp import web

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
    """Hold one bounded subscription open and heartbeat it through proxies."""
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
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
