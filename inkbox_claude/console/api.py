"""Framework-free console read models and command wrappers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..gate.scopes import SCOPES, SCOPE_SYSTEM
from ..gate.store import Request, Store
from ..gate.schedules import cadence as schedule_cadence, next_three as schedule_next_three
from . import events

REQUEST_STATES = frozenset(
    {"pending", "approved", "running", "done", "rejected", "expired", "failed"}
)


def settings_list(_store: Any) -> List[Dict[str, Any]]:
    """Every knob the owner may turn, with its value and where that value came from."""
    from ..gate import settings as gate_settings
    return gate_settings.describe()


def settings_set(store: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Set one knob, or hand it back by sending a null value. The registry decides what is
    allowed: a name it does not know, or a value out of range, is refused."""
    from ..gate import settings as gate_settings
    name = _required_text(payload.get("name"), "name")
    if not gate_settings.known(name):
        raise ValidationError(f"{name} is not a setting the console may change")
    if payload.get("value", None) is None:
        store.clear_setting(name)
    else:
        try:
            store.set_setting(name, gate_settings.coerce(name, payload.get("value")))
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
    gate_settings.invalidate()
    events.publish("settings.changed", {"name": name})
    return {"ok": True, "settings": gate_settings.describe()}


class ValidationError(ValueError):
    """A command payload is invalid and should become an HTTP 400."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} is required")
    return value.strip()


def _text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    return value.strip()


def _scopes(value: Any) -> List[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValidationError("scopes must be an array of strings")
    scopes = list(dict.fromkeys(item.strip() for item in value if item.strip()))
    unknown = sorted(set(scopes) - set(SCOPES))
    if unknown:
        raise ValidationError(f"unknown scope(s): {', '.join(unknown)}")
    return scopes


def _request_model(request: Request) -> Dict[str, Any]:
    status = request.status or {}
    result = status.get("summary") or status.get("error") or request.raw_output or ""
    return {
        "id": request.id,
        "sender": request.sender_name or request.sender,
        "sender_key": request.sender,
        "surface": request.mode,
        "summary": request.summary,
        "scopes": list(request.scopes),
        "original_message": request.original_message,
        "result": result,
        "state": request.state,
        "subject": request.subject,
        "created_at": request.created_at,
        "updated_at": request.updated_at,
        "revision": request.revision,
    }


def people(store: Store) -> List[Dict[str, Any]]:
    roles_by_name = {row["name"]: row for row in store.roles()}
    by_key: Dict[str, Dict[str, Any]] = {
        row["key"]: {
            "key": row["key"],
            "person": row.get("person") or row["key"],
            "role": "",
            "scopes": [],
            "note": "",
            "channels": row.get("channels") or [],
            "last_seen": row.get("last_seen") or 0.0,
        }
        for row in store.known_people()
    }
    for row in store.trusted():
        merged = dict(by_key.get(row["key"], {}))
        merged.update(row)
        by_key[row["key"]] = merged
    out = []
    for row in by_key.values():
        role_scopes = roles_by_name.get(row.get("role") or "", {}).get("scopes", [])
        effective = sorted(set(role_scopes) | set(row.get("scopes") or []))
        row["effective_scopes"] = effective
        read_count = sum(1 for name in effective if _scope_kind(name, SCOPES.get(name, {})) == "read")
        write_count = sum(1 for name in effective if _scope_kind(name, SCOPES.get(name, {})) in ("write", "send"))
        row["scope_counts"] = {"read": read_count, "write": write_count, "total": len(effective)}
        row["has_send"] = any(_scope_kind(name, SCOPES.get(name, {})) == "send" for name in effective)
        out.append(row)
    return sorted(out, key=lambda row: ((row.get("person") or "").lower(), row["key"]))


def people_count(store: Store) -> int:
    return len(people(store))


def people_sync(store: Store, client: Any) -> Dict[str, Any]:
    from . import sync as _sync
    result = _sync.sync_people(store, client)
    events.publish("people.synced", result)
    return result


def people_last_sync(store: Store) -> Dict[str, Any]:
    from . import sync as _sync
    return _sync.last_sync(store)


def upsert_person(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    key = _required_text(payload.get("key"), "key")
    scopes = _scopes(payload.get("scopes"))
    store.set_trust(
        key,
        person=_text(payload.get("person"), "person"),
        role=_text(payload.get("role"), "role"),
        scopes=scopes,
        note=_text(payload.get("note"), "note"),
    )
    row = next(item for item in people(store) if item["key"] == store.trusted_key(key))
    events.publish("permissions.changed", {"kind": "person", "key": row["key"]})
    return row


def delete_person(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    key = _required_text(payload.get("key"), "key")
    store.drop_trust(key)
    events.publish("permissions.changed", {"kind": "person", "key": key, "deleted": True})
    return {"ok": True, "key": key}


def _scope_kind(name: str, spec: Dict[str, Any]) -> str:
    """read | write | send. Everything in SCOPES is read or write already (`read` bool);
    a scope is "send" instead of plain "write" when it can make Blatbot speak as itself
    (its tools end in _send, or its own name says so) -- that is the one category the
    console calls out with a standing warning, because a person who holds it can have
    Blatbot send a message Aaron never saw."""
    if spec.get("read"):
        return "read"
    tools = [str(t).lower() for t in (spec.get("tools") or [])]
    if "send" in name.lower() or any(t.endswith("_send") or "_send_" in t for t in tools):
        return "send"
    return "write"


def _scope_detail(name: str) -> Dict[str, Any]:
    spec = SCOPES.get(name, {})
    return {
        "name": name,
        "purpose": spec.get("purpose") or spec.get("description") or name,
        "tools_count": len(spec.get("tools") or []),
        "group": SCOPE_SYSTEM.get(name, "other"),
        "kind": _scope_kind(name, spec),
    }


_SEND_WARNING = (
    "People in this role can have Blatbot send messages without Aaron seeing them first."
)


def _role_members(store: Store, name: str) -> List[Dict[str, Any]]:
    return [
        {"key": row["key"], "person": row.get("person") or row["key"]}
        for row in store.trusted()
        if (row.get("role") or "") == name
    ]


def _role_model(store: Store, row: Dict[str, Any], people_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    scope_names = row.get("scopes") or []
    details = [_scope_detail(name) for name in scope_names]
    read_scopes = [d for d in details if d["kind"] == "read"]
    write_scopes = [d for d in details if d["kind"] in ("write", "send")]
    has_send = any(d["kind"] == "send" for d in details)
    members = [
        {"key": r["key"], "person": r.get("person") or r["key"]}
        for r in (people_rows if people_rows is not None else store.trusted())
        if (r.get("role") or "") == row["name"]
    ]
    return {
        "name": row["name"],
        "note": row.get("note") or "",
        "scopes": scope_names,
        "read_scopes": read_scopes,
        "write_scopes": write_scopes,
        "members": members,
        "people_count": len(members),
        "warning": _SEND_WARNING if has_send else "",
    }


def roles(store: Store) -> List[Dict[str, Any]]:
    people_rows = store.trusted()
    return [_role_model(store, row, people_rows) for row in store.roles()]


def upsert_role(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _required_text(payload.get("name"), "name").lower()
    store.set_role(name, _scopes(payload.get("scopes")), _text(payload.get("note"), "note"))
    row = next(item for item in roles(store) if item["name"] == name)
    events.publish("permissions.changed", {"kind": "role", "name": name})
    return row


def delete_role(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _required_text(payload.get("name"), "name").lower()
    store.drop_role(name)
    events.publish("permissions.changed", {"kind": "role", "name": name, "deleted": True})
    return {"ok": True, "name": name}


def add_role_member(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _required_text(payload.get("name"), "name").lower()
    key = _required_text(payload.get("key"), "key")
    if not any(row["name"] == name for row in store.roles()):
        raise ValidationError(f"no such role: {name}")
    existing = next((row for row in store.trusted() if row["key"] == store.trusted_key(key)), None)
    person = _text(payload.get("person"), "person") or (existing or {}).get("person", "") or key
    scopes = (existing or {}).get("scopes") or []
    note = (existing or {}).get("note") or ""
    store.set_trust(key, person=person, role=name, scopes=scopes, note=note)
    events.publish("permissions.changed", {"kind": "role", "name": name, "member_added": key})
    return next(item for item in roles(store) if item["name"] == name)


def remove_role_member(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _required_text(payload.get("name"), "name").lower()
    key = _required_text(payload.get("key"), "key")
    existing = next((row for row in store.trusted() if row["key"] == store.trusted_key(key)), None)
    if existing and (existing.get("role") or "") == name:
        store.set_trust(key, person=existing.get("person") or "", role="",
                         scopes=existing.get("scopes") or [], note=existing.get("note") or "")
    events.publish("permissions.changed", {"kind": "role", "name": name, "member_removed": key})
    return next(item for item in roles(store) if item["name"] == name)


def scopes() -> List[Dict[str, Any]]:
    return [_scope_detail(name) for name in SCOPES]


def requests(store: Store, state: str = "") -> List[Dict[str, Any]]:
    state = (state or "").strip().lower()
    if state and state not in REQUEST_STATES:
        raise ValidationError(f"unknown request state: {state}")
    return [_request_model(row) for row in store.recent_requests(state=state)]


async def decide_request(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("id")
    if isinstance(request_id, bool):
        raise ValidationError("id must be an integer")
    try:
        request_id = int(request_id)
    except (TypeError, ValueError):
        raise ValidationError("id must be an integer") from None
    decision = _required_text(payload.get("decision"), "decision").lower()
    if decision not in {"yes", "no", "edit"}:
        raise ValidationError("decision must be yes, no, or edit")
    note = _text(payload.get("note"), "note")
    if decision == "edit" and not note:
        raise ValidationError("an edit needs a note")
    result = await manager.decide_request(
        request_id, decision, actor="aaron", source="console", note=note
    )
    if result.get("ok"):
        events.publish(
            "request.state_changed",
            {"id": request_id, "state": result.get("state"), "decision": decision},
        )
    return dict(result)


def tasks(store: Store) -> List[Dict[str, Any]]:
    return [dict(row) for row in store.recent_tasks(limit=50)]


def schedules(store: Store) -> Dict[str, Any]:
    rows = []
    for schedule in store.schedules():
        rows.append({
            "id": schedule.id, "task_id": schedule.task_id, "title": schedule.title,
            "prompt": schedule.prompt, "kind": schedule.kind, "cron": schedule.cron,
            "run_at": schedule.run_at, "timezone": schedule.timezone,
            "scopes": list(schedule.scopes), "report_mode": schedule.report_mode,
            "state": schedule.state, "next_run": schedule.next_run,
            "next_three": schedule_next_three(schedule), "last_run": schedule.last_run,
            "last_outcome": schedule.last_outcome, "run_count": schedule.run_count,
            "max_runs": schedule.max_runs, "deadline": schedule.deadline,
            "notes": list(schedule.notes), "revision": schedule.revision,
            "cadence": schedule_cadence(schedule),
        })
    paused = str(store.settings().get("GATE_SCHEDULES_PAUSED", "0")).lower() in {"1", "true", "yes", "on"}
    return {"global_paused": paused, "schedules": rows}


async def create_schedule(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    title = _required_text(payload.get("title"), "title")
    prompt = _required_text(payload.get("prompt"), "prompt")
    task_id = payload.get("task_id")
    if task_id in (None, ""):
        task_id = manager.store.create_task(title)["id"]
    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        raise ValidationError("task_id must be an integer") from None
    spec = {k: payload.get(k) for k in ("kind", "cron", "run_at", "timezone", "report_mode",
                                        "max_runs", "deadline")}
    spec.update(title=title, prompt=prompt)
    try:
        schedule = await manager.schedules.propose(owner=True, chat_id="owner", task_id=task_id,
                                                   spec=spec, scopes=_scopes(payload.get("scopes")))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("schedules.changed", {"id": schedule.id, "state": schedule.state})
    return schedules(manager.store)


async def edit_schedule(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        schedule_id = int(payload.get("id"))
    except (TypeError, ValueError):
        raise ValidationError("id must be an integer") from None
    changes = {k: payload[k] for k in ("title", "prompt", "kind", "cron", "run_at", "timezone",
                                                  "report_mode", "max_runs", "deadline", "scopes") if k in payload}
    if "scopes" in changes:
        changes["scopes"] = _scopes(changes["scopes"])
    try:
        schedule = await manager.schedules.edit(schedule_id, owner=True, changes=changes)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("schedules.changed", {"id": schedule.id, "state": schedule.state})
    return schedules(manager.store)


async def schedule_action(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        schedule_id = int(payload.get("id"))
    except (TypeError, ValueError):
        raise ValidationError("id must be an integer") from None
    action = _required_text(payload.get("action"), "action")
    try:
        schedule = await manager.schedules.action(schedule_id, action, owner=True)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("schedules.changed", {"id": schedule.id, "action": action})
    return schedules(manager.store)


def schedules_global(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    paused = payload.get("paused")
    if not isinstance(paused, bool):
        raise ValidationError("paused must be true or false")
    store.set_setting("GATE_SCHEDULES_PAUSED", "1" if paused else "0")
    events.publish("schedules.changed", {"global_paused": paused})
    return schedules(store)


def task(store: Store, task_id: int) -> Optional[Dict[str, Any]]:
    # The ledger is shown whole. The limit is only a guard against a runaway table.
    row = store.task_with_events(int(task_id), limit=100000)
    return dict(row) if row else None


def _tail_errors(path: Path, *, max_bytes: int = 64 * 1024, limit: int = 30) -> List[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]
    interesting = [
        line for line in lines
        if " ERROR " in line or " CRITICAL " in line or "Traceback (most recent call last)" in line
    ]
    return interesting[-limit:]


def health(gateway: Any, store: Store) -> Dict[str, Any]:
    started_at = float(getattr(gateway, "_console_started_at", time.time()))
    state_dir = Path(os.getenv("INKBOX_CLAUDE_HOME") or (Path.home() / ".inkbox-claude"))
    public_url = str(getattr(gateway, "_public_url", "") or "")
    tunnel = getattr(gateway, "_tunnel", None)
    return {
        "tunnel": {
            "connected": bool(public_url or tunnel),
            "url": public_url,
            "mode": "configured" if public_url and tunnel is None else "managed",
        },
        "uptime_seconds": max(0, int(time.time() - started_at)),
        "last_restart": started_at,
        "interrupted_runs": store.interrupted_request_count(),
        "recent_errors": _tail_errors(state_dir / "gateway.log"),
    }


def overview(gateway: Any, store: Store) -> Dict[str, Any]:
    waiting = [_request_model(row) for row in store.pending()]
    return {
        "counts": store.console_overview(),
        "health": health(gateway, store),
        "waiting": waiting,
    }


def _request_summary(request: Request) -> Dict[str, Any]:
    """A trimmed request shape for the home page: no original message, no raw tool
    output -- just what a calm overview needs to show and link to."""
    return {
        "id": request.id,
        "sender": request.sender_name or request.sender,
        "surface": request.mode,
        "summary": request.summary or "",
        "state": request.state,
        "created_at": request.created_at,
        "updated_at": request.updated_at,
    }


def home(gateway: Any, store: Store) -> Dict[str, Any]:
    """Everything the Home page shows, already shaped, counted, sorted and worded.
    No message bodies and no secrets: summaries, names, handles, channels, states,
    counts and timestamps only."""
    gateway_health = health(gateway, store)

    pending = store.pending()  # oldest first
    oldest = pending[0] if pending else None
    needs_you = {
        "count": len(pending),
        "oldest": _request_summary(oldest) if oldest else None,
    }

    running = store.recent_requests(state="running", limit=50)
    right_now = {
        "running_count": len(running),
        "running": [_request_summary(r) for r in running[:5]],
        "gateway_up": True,  # this response itself proves the gate answered
        "tunnel_connected": bool(gateway_health["tunnel"]["connected"]),
    }

    recent = store.recent_requests(limit=8)  # newest first
    recent_activity = [_request_summary(r) for r in recent]

    schedules_data = schedules(store)
    now = time.time()
    upcoming = sorted(
        (
            {
                "id": s["id"],
                "title": s["title"],
                "cadence": s["cadence"],
                "next_run": s["next_run"],
                "state": s["state"],
            }
            for s in schedules_data["schedules"]
            if s.get("next_run") and s["next_run"] >= now and s["state"] == "active"
        ),
        key=lambda s: s["next_run"],
    )[:5]
    coming_up = {
        "schedules_paused": schedules_data["global_paused"],
        "next": upcoming,
    }

    people_rows = people(store)
    role_rows = roles(store)
    people_summary = {
        "total": len(people_rows),
        "with_role": sum(1 for row in people_rows if row.get("role")),
        "roles": [{"name": r["name"], "members": r["people_count"]} for r in role_rows],
        "last_sync": people_last_sync(store),
    }

    contacts = sorted(
        (row for row in people_rows if (row.get("last_seen") or 0) and row["key"] not in {"", None}),
        key=lambda row: -(row.get("last_seen") or 0),
    )[:6]
    recent_contacts = [
        {
            "key": row["key"],
            "person": row.get("person") or row["key"],
            "channels": row.get("channels") or [],
            "last_seen": row.get("last_seen") or 0,
        }
        for row in contacts
    ]

    return {
        "needs_you": needs_you,
        "right_now": right_now,
        "recent_activity": recent_activity,
        "coming_up": coming_up,
        "people": people_summary,
        "recent_contacts": recent_contacts,
    }
