"""aiohttp routes for the private console API.

The console's HTML/CSS/JS now live in the separate blatbot-console frontend repo and
are served by their own tiny static server. This module serves only /console/api/*
and the /console/events stream -- no page routes, no static asset routes.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Dict
from urllib.parse import urlsplit

from aiohttp import web

from . import access, api, events


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
    """A mutation must come from the console's own origin (the historical behavior,
    kept as a second layer) or from an origin named in CONSOLE_ALLOWED_ORIGINS -- the
    separate frontend's own origin, now that it is served by its own app rather than
    by this gateway. The tailnet_gate middleware in access.py is the first layer and
    runs on every /console/api/* request, read or write, before this one does."""
    origin = request.headers.get("Origin", "")
    parsed = urlsplit(origin)
    same_origin = bool(origin) and parsed.scheme in {"http", "https"} and parsed.netloc.lower() == request.host.lower()
    if same_origin or (origin and access.origin_allowed(origin)):
        return
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


async def get_overview(request: web.Request) -> web.StreamResponse:
    manager = _manager(request)
    return _json(api.overview(request.app["console.gateway"], manager.store))


async def get_home(request: web.Request) -> web.StreamResponse:
    manager = _manager(request)
    return _json(api.home(request.app["console.gateway"], manager.store))


async def get_people(request: web.Request) -> web.StreamResponse:
    store = _manager(request).store
    return _json({
        "people": api.people(store),
        "count": api.people_count(store),
        "last_sync": api.people_last_sync(store),
    })


async def post_people(request: web.Request) -> web.StreamResponse:
    return _json(api.upsert_person(_manager(request).store, await _payload(request)))


async def post_people_delete(request: web.Request) -> web.StreamResponse:
    return _json(api.delete_person(_manager(request).store, await _payload(request)))


async def post_people_sync(request: web.Request) -> web.StreamResponse:
    gateway = request.app["console.gateway"]
    client = getattr(gateway, "inkbox_directory_client", None)
    if client is None:
        return _json({"error": "no Inkbox directory client configured"}, status=503)
    import asyncio

    # api.people_sync pages Inkbox's SDK synchronously (blocking HTTP calls);
    # run the whole thing off the event loop so one sync can't stall every other
    # console request (or webhook delivery) for its duration.
    result = await asyncio.to_thread(api.people_sync, _manager(request).store, client)
    return _json(result)


# -- person directory endpoints ---------------------------------------------
async def get_people_directory(request: web.Request) -> web.StreamResponse:
    return _json(api.people_directory(_manager(request).store))


async def post_people_create(request: web.Request) -> web.StreamResponse:
    return _json(api.create_person_endpoint(_manager(request).store, await _payload(request)))


async def post_people_update(request: web.Request) -> web.StreamResponse:
    return _json(api.update_person_endpoint(_manager(request).store, await _payload(request)))


async def post_people_remove(request: web.Request) -> web.StreamResponse:
    return _json(api.remove_person_endpoint(_manager(request).store, await _payload(request)))


async def post_people_contact_add(request: web.Request) -> web.StreamResponse:
    return _json(api.add_contact_endpoint(_manager(request).store, await _payload(request)))


async def post_people_contact_remove(request: web.Request) -> web.StreamResponse:
    return _json(api.remove_contact_endpoint(_manager(request).store, await _payload(request)))


async def post_people_contact_move(request: web.Request) -> web.StreamResponse:
    return _json(api.move_contact_endpoint(_manager(request).store, await _payload(request)))


async def post_people_link(request: web.Request) -> web.StreamResponse:
    return _json(api.link_people_endpoint(_manager(request).store, await _payload(request)))


async def get_roles(request: web.Request) -> web.StreamResponse:
    return _json(api.roles(_manager(request).store))


async def post_roles(request: web.Request) -> web.StreamResponse:
    return _json(api.upsert_role(_manager(request).store, await _payload(request)))


async def post_roles_delete(request: web.Request) -> web.StreamResponse:
    return _json(api.delete_role(_manager(request).store, await _payload(request)))


async def post_roles_members_add(request: web.Request) -> web.StreamResponse:
    return _json(api.add_role_member(_manager(request).store, await _payload(request)))


async def post_roles_members_remove(request: web.Request) -> web.StreamResponse:
    return _json(api.remove_role_member(_manager(request).store, await _payload(request)))


async def post_roles_scopes_add(request: web.Request) -> web.StreamResponse:
    return _json(api.add_role_scope(_manager(request).store, await _payload(request)))


async def post_roles_scopes_remove(request: web.Request) -> web.StreamResponse:
    return _json(api.remove_role_scope(_manager(request).store, await _payload(request)))


async def get_scopes(request: web.Request) -> web.StreamResponse:
    store = _manager(request).store
    return _json(api.scopes(store))


async def get_scopes_breakdown(request: web.Request) -> web.StreamResponse:
    store = _manager(request).store
    return _json(api.scopes_breakdown(store))


async def get_requests(request: web.Request) -> web.StreamResponse:
    return _json(api.requests(_manager(request).store, request.query.get("state", "")))


async def post_request_decision(request: web.Request) -> web.StreamResponse:
    result = await api.decide_request(_manager(request), await _payload(request))
    return _json(result, status=200 if result.get("ok") else 409)


async def get_tasks(request: web.Request) -> web.StreamResponse:
    return _json(api.tasks(_manager(request).store))


async def get_schedules(request: web.Request) -> web.StreamResponse:
    return _json(api.schedules(_manager(request).store))


async def post_schedules(request: web.Request) -> web.StreamResponse:
    return _json(await api.create_schedule(_manager(request), await _payload(request)))


async def post_schedules_edit(request: web.Request) -> web.StreamResponse:
    return _json(await api.edit_schedule(_manager(request), await _payload(request)))


async def post_schedules_action(request: web.Request) -> web.StreamResponse:
    return _json(await api.schedule_action(_manager(request), await _payload(request)))


async def post_schedules_global(request: web.Request) -> web.StreamResponse:
    return _json(api.schedules_global(_manager(request).store, await _payload(request)))


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


async def get_settings(request: web.Request) -> web.StreamResponse:
    return _json(api.settings_list(_manager(request).store))


async def post_settings(request: web.Request) -> web.StreamResponse:
    return _json(api.settings_set(_manager(request).store, await _payload(request)))


def register(app: web.Application, gateway: Any) -> None:
    """Mount only the console API and event stream. Page rendering lives in the
    separate blatbot-console frontend repo now."""
    gateway._console_started_at = getattr(gateway, "_console_started_at", time.time())
    app["console.gateway"] = gateway
    # Page rendering (including /console/schedules) lives in the separate blatbot-console
    # frontend repo now; this gateway mounts only the JSON API and the event stream, gated
    # to the tailnet.
    app.middlewares.append(access.tailnet_gate)
    app.middlewares.append(access.cors_gate)
    app.router.add_get("/console/api/overview", _endpoint(get_overview))
    app.router.add_get("/console/api/home", _endpoint(get_home))
    app.router.add_get("/console/api/people", _endpoint(get_people))
    app.router.add_post("/console/api/people", _endpoint(post_people, mutation=True))
    app.router.add_post("/console/api/people/delete", _endpoint(post_people_delete, mutation=True))
    app.router.add_post("/console/api/people/sync", _endpoint(post_people_sync, mutation=True))
    app.router.add_get("/console/api/people/directory", _endpoint(get_people_directory))
    app.router.add_post("/console/api/people/create", _endpoint(post_people_create, mutation=True))
    app.router.add_post("/console/api/people/update", _endpoint(post_people_update, mutation=True))
    app.router.add_post("/console/api/people/remove", _endpoint(post_people_remove, mutation=True))
    app.router.add_post("/console/api/people/contact/add", _endpoint(post_people_contact_add, mutation=True))
    app.router.add_post("/console/api/people/contact/remove", _endpoint(post_people_contact_remove, mutation=True))
    app.router.add_post("/console/api/people/contact/move", _endpoint(post_people_contact_move, mutation=True))
    app.router.add_post("/console/api/people/link", _endpoint(post_people_link, mutation=True))
    app.router.add_get("/console/api/roles", _endpoint(get_roles))
    app.router.add_post("/console/api/roles", _endpoint(post_roles, mutation=True))
    app.router.add_post("/console/api/roles/delete", _endpoint(post_roles_delete, mutation=True))
    app.router.add_post("/console/api/roles/members/add", _endpoint(post_roles_members_add, mutation=True))
    app.router.add_post("/console/api/roles/members/remove", _endpoint(post_roles_members_remove, mutation=True))
    app.router.add_post("/console/api/roles/scopes/add", _endpoint(post_roles_scopes_add, mutation=True))
    app.router.add_post("/console/api/roles/scopes/remove", _endpoint(post_roles_scopes_remove, mutation=True))
    app.router.add_get("/console/api/scopes", _endpoint(get_scopes))
    app.router.add_get("/console/api/scopes/breakdown", _endpoint(get_scopes_breakdown))
    app.router.add_get("/console/api/requests", _endpoint(get_requests))
    app.router.add_post(
        "/console/api/requests/decide", _endpoint(post_request_decision, mutation=True)
    )
    app.router.add_get("/console/api/tasks", _endpoint(get_tasks))
    app.router.add_get("/console/api/tasks/{id}", _endpoint(get_task))
    app.router.add_get("/console/api/schedules", _endpoint(get_schedules))
    app.router.add_post("/console/api/schedules", _endpoint(post_schedules, mutation=True))
    app.router.add_post("/console/api/schedules/edit", _endpoint(post_schedules_edit, mutation=True))
    app.router.add_post("/console/api/schedules/action", _endpoint(post_schedules_action, mutation=True))
    app.router.add_post("/console/api/schedules/global", _endpoint(post_schedules_global, mutation=True))
    app.router.add_get("/console/api/health", _endpoint(get_health))
    app.router.add_get("/console/api/settings", _endpoint(get_settings))
    app.router.add_post("/console/api/settings", _endpoint(post_settings, mutation=True))
    app.router.add_get("/console/events", get_events)
    for path in (
        "/console/api/overview", "/console/api/home", "/console/api/people", "/console/api/people/sync",
        "/console/api/people/directory", "/console/api/people/create", "/console/api/people/update",
        "/console/api/people/remove", "/console/api/people/contact/add", "/console/api/people/contact/remove",
        "/console/api/people/contact/move", "/console/api/people/link",
        "/console/api/roles", "/console/api/roles/delete", "/console/api/roles/members/add",
        "/console/api/roles/members/remove", "/console/api/roles/scopes/add", "/console/api/roles/scopes/remove",
        "/console/api/scopes", "/console/api/scopes/breakdown", "/console/api/requests",
        "/console/api/requests/decide", "/console/api/tasks", "/console/api/health",
        "/console/api/settings",
    ):
        app.router.add_route("OPTIONS", path, _options)
    app.router.add_route("OPTIONS", "/console/events", _options)


async def _options(request: web.Request) -> web.StreamResponse:
    # Actual CORS headers are added by the cors_gate middleware; this handler just
    # needs to exist so aiohttp's router has a 2xx response for OPTIONS to decorate.
    return web.Response(status=204)
