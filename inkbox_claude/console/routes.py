"""aiohttp routes for the private console."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict
from urllib.parse import urlsplit

from aiohttp import web

from . import api, events

_STATIC = Path(__file__).resolve().parent / "static"


def _json(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def _manager(request: web.Request) -> Any:
    gateway = request.app["console.gateway"]
    manager = getattr(gateway, "sessions", None)
    if manager is None or getattr(manager, "store", None) is None:
        raise web.HTTPServiceUnavailable(
            text='{"error":"gate is not ready"}', content_type="application/json"
        )
    return manager


def _same_origin(request: web.Request) -> None:
    origin = request.headers.get("Origin", "")
    parsed = urlsplit(origin)
    if (
        not origin
        or parsed.scheme not in {"http", "https"}
        or parsed.netloc.lower() != request.host.lower()
    ):
        raise web.HTTPForbidden(
            text='{"error":"cross-origin mutation rejected"}', content_type="application/json"
        )


async def _payload(request: web.Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise api.ValidationError("body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise api.ValidationError("body must be a JSON object")
    return body


def _endpoint(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]], *, mutation: bool = False
) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    async def wrapped(request: web.Request) -> web.StreamResponse:
        try:
            if mutation:
                _same_origin(request)
            return await handler(request)
        except api.ValidationError as exc:
            return _json({"error": str(exc)}, status=400)

    return wrapped


def _asset_version(name: str) -> str:
    """A short fingerprint of an asset's current bytes, used to bust browser caches."""
    path = _STATIC / name
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    except OSError:
        return "0"


async def index(_request: web.Request) -> web.StreamResponse:
    """Serve the shell with its assets stamped by content.

    The stylesheet and script are referenced with the hash of what they currently hold, so
    a deploy changes their URLs and no browser can answer from a copy of the old ones. The
    shell itself must never be cached, or it would keep handing out the stamps it was
    built with.
    """
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    for name in ("console.css", "console.js"):
        html = html.replace(f"/console/{name}", f"/console/{name}?v={_asset_version(name)}")
    return web.Response(text=html, content_type="text/html",
                        headers={"Cache-Control": "no-store"})


async def get_overview(request: web.Request) -> web.StreamResponse:
    manager = _manager(request)
    return _json(api.overview(request.app["console.gateway"], manager.store))


async def get_people(request: web.Request) -> web.StreamResponse:
    return _json(api.people(_manager(request).store))


async def post_people(request: web.Request) -> web.StreamResponse:
    return _json(api.upsert_person(_manager(request).store, await _payload(request)))


async def post_people_delete(request: web.Request) -> web.StreamResponse:
    return _json(api.delete_person(_manager(request).store, await _payload(request)))


async def get_roles(request: web.Request) -> web.StreamResponse:
    return _json(api.roles(_manager(request).store))


async def post_roles(request: web.Request) -> web.StreamResponse:
    return _json(api.upsert_role(_manager(request).store, await _payload(request)))


async def post_roles_delete(request: web.Request) -> web.StreamResponse:
    return _json(api.delete_role(_manager(request).store, await _payload(request)))


async def get_scopes(request: web.Request) -> web.StreamResponse:
    _manager(request)
    return _json(api.scopes())


async def get_requests(request: web.Request) -> web.StreamResponse:
    return _json(api.requests(_manager(request).store, request.query.get("state", "")))


async def post_request_decision(request: web.Request) -> web.StreamResponse:
    result = await api.decide_request(_manager(request), await _payload(request))
    return _json(result, status=200 if result.get("ok") else 409)


async def get_tasks(request: web.Request) -> web.StreamResponse:
    return _json(api.tasks(_manager(request).store))


async def get_task(request: web.Request) -> web.StreamResponse:
    try:
        task_id = int(request.match_info["id"])
    except (KeyError, ValueError):
        raise api.ValidationError("task id must be an integer") from None
    row = api.task(_manager(request).store, task_id)
    return _json(row) if row else _json({"error": f"no task {task_id}"}, status=404)


async def get_health(request: web.Request) -> web.StreamResponse:
    manager = _manager(request)
    return _json(api.health(request.app["console.gateway"], manager.store))


async def get_events(request: web.Request) -> web.StreamResponse:
    _manager(request)
    return await events.event_stream(request)


def _asset(name: str, content_type: str):
    async def handler(_request: web.Request) -> web.StreamResponse:
        path = _STATIC / name
        if not path.is_file():
            raise web.HTTPNotFound()
        # Revalidate every time: the stamped URL makes a hit cheap, and a console that
        # serves yesterday's stylesheet is worse than one that asks.
        return web.FileResponse(path, headers={"Content-Type": content_type,
                                               "Cache-Control": "no-cache"})
    return handler


async def get_settings(request: web.Request) -> web.StreamResponse:
    return _json(api.settings_list(_manager(request).store))


async def post_settings(request: web.Request) -> web.StreamResponse:
    return _json(api.settings_set(_manager(request).store, await _payload(request)))


def register(app: web.Application, gateway: Any) -> None:
    """Mount the console without assuming the gate is initialized yet."""
    gateway._console_started_at = getattr(gateway, "_console_started_at", time.time())
    app["console.gateway"] = gateway
    app.router.add_get("/console", index)
    app.router.add_get("/console/", index)
    # One page per screen, each with its own URL. The shell decides what to render from the
    # path, so every one of these has to reach it: typed in, reloaded on, linked to.
    for page in ("permissions", "tasks", "health", "settings", "requests"):
        app.router.add_get(f"/console/{page}", index)
        app.router.add_get(f"/console/{page}/", index)
    app.router.add_get("/console/tasks/{task_id}", index)
    app.router.add_get("/console/api/overview", _endpoint(get_overview))
    app.router.add_get("/console/api/people", _endpoint(get_people))
    app.router.add_post("/console/api/people", _endpoint(post_people, mutation=True))
    app.router.add_post("/console/api/people/delete", _endpoint(post_people_delete, mutation=True))
    app.router.add_get("/console/api/roles", _endpoint(get_roles))
    app.router.add_post("/console/api/roles", _endpoint(post_roles, mutation=True))
    app.router.add_post("/console/api/roles/delete", _endpoint(post_roles_delete, mutation=True))
    app.router.add_get("/console/api/scopes", _endpoint(get_scopes))
    app.router.add_get("/console/api/requests", _endpoint(get_requests))
    app.router.add_post(
        "/console/api/requests/decide", _endpoint(post_request_decision, mutation=True)
    )
    app.router.add_get("/console/api/tasks", _endpoint(get_tasks))
    app.router.add_get("/console/api/tasks/{id}", _endpoint(get_task))
    app.router.add_get("/console/api/health", _endpoint(get_health))
    app.router.add_get("/console/api/settings", _endpoint(get_settings))
    app.router.add_post("/console/api/settings", _endpoint(post_settings, mutation=True))
    app.router.add_get("/console/events", get_events)
    app.router.add_static("/console/static/", _STATIC, show_index=False)
    # The page asks for its own assets next to itself; serve them there too, so the markup
    # does not have to know where the package keeps them.
    for name, kind in (("console.css", "text/css"), ("console.js", "application/javascript")):
        app.router.add_get(f"/console/{name}", _asset(name, kind))
