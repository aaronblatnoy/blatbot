"""Access control for /console/api/*: the network peer must be on the tailnet, and (for
the handful of browser-origin checks that matter) the Origin must be on an explicit
allow-list. This closes the hole where the whole console -- reads and writes -- was
reachable at the public tunnel hostname with mutations guarded only by an Origin header
a script can set to anything.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Iterable, Optional

from aiohttp import web

TAILNET = ipaddress.ip_network("100.64.0.0/10")


def is_tailnet_peer(remote: Optional[str]) -> bool:
    if not remote:
        return False
    try:
        addr = ipaddress.ip_address(remote)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr in TAILNET


def peer_address(request: web.Request) -> Optional[str]:
    """The literal socket peer. Deliberately ignores X-Forwarded-For and similar headers,
    which a client controls -- a request that actually arrived through the public tunnel
    must not be able to claim a tailnet address by setting a header, including a request
    whose immediate TCP peer is 127.0.0.1 because it came in through a local reverse proxy
    terminating the public hostname."""
    peername = request.transport.get_extra_info("peername") if request.transport else None
    if not peername:
        return None
    return peername[0]


def allowed_origins() -> Iterable[str]:
    raw = os.getenv("CONSOLE_ALLOWED_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def origin_allowed(origin: str) -> bool:
    allow = list(allowed_origins())
    if not allow:
        # No allow-list configured: fall back to same-origin-only, the previous behavior,
        # rather than silently accepting anything.
        return False
    return origin in allow


@web.middleware
async def tailnet_gate(request: web.Request, handler):
    """404, not 403: a box reachable from the public tunnel should look like nothing is
    there at all for this path, not like a locked door worth trying to pick."""
    if not request.path.startswith("/console/api") and request.path != "/console/events":
        return await handler(request)
    peer = peer_address(request)
    if not is_tailnet_peer(peer):
        raise web.HTTPNotFound()
    return await handler(request)


@web.middleware
async def cors_gate(request: web.Request, handler):
    path_is_console_api = request.path.startswith("/console/api") or request.path == "/console/events"
    origin = request.headers.get("Origin", "")
    if request.method == "OPTIONS" and path_is_console_api:
        resp = web.Response(status=204)
        if origin and origin_allowed(origin):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Vary"] = "Origin"
        return resp
    resp = await handler(request)
    if path_is_console_api and origin and origin_allowed(origin):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
    return resp
