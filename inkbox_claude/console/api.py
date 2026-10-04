"""Framework-free console read models and command wrappers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..gate.scopes import SCOPES, SCOPE_SYSTEM
from ..gate.store import Request, Store
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
        }
        for row in store.known_people()
    }
    for row in store.trusted():
        by_key[row["key"]] = dict(row)
    out = []
    for row in by_key.values():
        role_scopes = roles_by_name.get(row.get("role") or "", {}).get("scopes", [])
        row["effective_scopes"] = sorted(set(role_scopes) | set(row.get("scopes") or []))
        out.append(row)
    return sorted(out, key=lambda row: ((row.get("person") or "").lower(), row["key"]))


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


def roles(store: Store) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for person in store.trusted():
        role = person.get("role") or ""
        if role:
            counts[role] = counts.get(role, 0) + 1
    return [dict(row, people_count=counts.get(row["name"], 0)) for row in store.roles()]


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


def scopes() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "purpose": spec.get("purpose") or spec.get("description") or name,
            "tools_count": len(spec.get("tools") or []),
            "group": SCOPE_SYSTEM.get(name, "other"),
        }
        for name, spec in SCOPES.items()
    ]


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


def task(store: Store, task_id: int) -> Optional[Dict[str, Any]]:
    row = store.task_with_events(int(task_id), limit=100)
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
