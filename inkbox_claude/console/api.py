"""Framework-free console read models and command wrappers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..gate.scopes import SCOPES, SCOPE_SYSTEM
from ..gate.store import Request, Store, role_names
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
    """One entry per person. `store.trusted()` is one row per CONTACT (a trusted
    person with two phones is two rows there); this groups those back into one row
    per person_id, with every contact they hold listed under `keys`. A directory
    contact nobody has linked to a person yet still stands alone, under its own key,
    until the owner attaches or promotes it."""
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
            "person_id": "",
        }
        for row in store.known_people()
    }
    if hasattr(store, "find_contact") and hasattr(store, "guess_contact_kind"):
        # Attach a person_id where the key matches an already-linked contact, so the
        # console can group several contacts of one person on this flat view too.
        for key, row in by_key.items():
            contact = store.find_contact(store.guess_contact_kind(key), key)
            if contact and contact.get("person_id"):
                row["person_id"] = contact["person_id"]
    for row in store.trusted():
        merged = dict(by_key.get(row["key"], {}))
        merged.update(row)
        by_key[row["key"]] = merged

    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in by_key.values():
        pid = row.get("person_id") or f"__key__{row['key']}"
        if pid not in grouped:
            g = dict(row)
            g["keys"] = [row["key"]]
            grouped[pid] = g
            order.append(pid)
        else:
            g = grouped[pid]
            g["keys"].append(row["key"])
            g["channels"] = sorted(set(g.get("channels") or []) | set(row.get("channels") or []))
            g["last_seen"] = max(g.get("last_seen") or 0.0, row.get("last_seen") or 0.0)

    out = []
    for pid in order:
        row = grouped[pid]
        role_scopes = [s for name in role_names(row.get("role"))
                       for s in roles_by_name.get(name, {}).get("scopes", [])]
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


# -- person directory: full CRUD over persons + contacts --------------------
# This is the model the People page's management UI drives. The flat
# upsert_person/delete_person pair above stays for the legacy single-handle shape;
# everything here operates on person ids and contact ids directly.

_CONTACT_KINDS = {"email", "phone", "telegram", "imessage", "name", "other"}


def _person_model(store: Store, row: Dict[str, Any]) -> Dict[str, Any]:
    roles_by_name = {r["name"]: r for r in store.roles()}
    role_scopes = [s for name in role_names(row.get("role"))
                   for s in roles_by_name.get(name, {}).get("scopes", [])]
    effective = sorted(set(role_scopes) | set(row.get("scopes") or []))
    return {
        "id": row["id"],
        "display": row.get("display") or "",
        "role": row.get("role") or "",
        "roles": role_names(row.get("role")),
        "scopes": row.get("scopes") or [],
        "effective_scopes": effective,
        "note": row.get("note") or "",
        "contacts": row.get("contacts") or [],
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def people_directory(store: Store) -> Dict[str, Any]:
    """Every live person with their contacts, plus every contact seen in traffic or
    sync that has not been attached to anyone yet."""
    return {
        "people": [_person_model(store, p) for p in store.list_people()],
        "unlinked": store.unlinked_contacts(),
    }


def create_person_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    display = _text(payload.get("display"), "display")
    role = _text(payload.get("role"), "role")
    scopes = _scopes(payload.get("scopes") or [])
    note = _text(payload.get("note"), "note")
    row = store.create_person(display=display, role=role, scopes=scopes, note=note)
    events.publish("people.changed", {"kind": "person", "id": row["id"], "action": "created"})
    return _person_model(store, row)


def update_person_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    person_id = _required_text(payload.get("id"), "id")
    changes: Dict[str, Any] = {}
    if "display" in payload:
        changes["display"] = _text(payload.get("display"), "display")
    if "role" in payload:
        changes["role"] = _text(payload.get("role"), "role")
    if "scopes" in payload:
        changes["scopes"] = _scopes(payload.get("scopes") or [])
    if "note" in payload:
        changes["note"] = _text(payload.get("note"), "note")
    try:
        row = store.update_person(person_id, **changes)
    except KeyError as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("people.changed", {"kind": "person", "id": person_id, "action": "updated"})
    return _person_model(store, row)


def remove_person_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    person_id = _required_text(payload.get("id"), "id")
    ok = store.delete_person(person_id)
    events.publish("people.changed", {"kind": "person", "id": person_id, "action": "deleted"})
    return {"ok": ok, "id": person_id}


def _validate_contact_kind(kind: Any) -> str:
    kind = _required_text(kind, "kind").lower()
    if kind not in _CONTACT_KINDS:
        raise ValidationError(f"unknown contact kind: {kind}")
    return kind


def add_contact_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    person_id = _required_text(payload.get("person_id"), "person_id")
    kind = _validate_contact_kind(payload.get("kind"))
    value = _required_text(payload.get("value"), "value")
    result = store.upsert_contact(kind, value, person_id=person_id, source="manual")
    if result["conflict"]:
        events.publish("people.changed", {"kind": "contact", "action": "conflict",
                                           "contact_id": result["contact"]["id"]})
        return {"ok": False, "conflict": True, "existing_person_id": result["existing_person_id"],
                "contact": result["contact"]}
    events.publish("people.changed", {"kind": "contact", "person_id": person_id, "action": "added"})
    return {"ok": True, "conflict": False, "contact": result["contact"]}


def remove_contact_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    contact_id = _required_text(payload.get("contact_id"), "contact_id")
    ok = store.remove_contact(contact_id)
    events.publish("people.changed", {"kind": "contact", "contact_id": contact_id, "action": "removed"})
    return {"ok": ok, "contact_id": contact_id}


def move_contact_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    contact_id = _required_text(payload.get("contact_id"), "contact_id")
    person_id = payload.get("person_id") or None
    if person_id is not None and not isinstance(person_id, str):
        raise ValidationError("person_id must be a string or null")
    try:
        row = store.move_contact(contact_id, person_id)
    except KeyError as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("people.changed", {"kind": "contact", "contact_id": contact_id, "action": "moved",
                                       "person_id": person_id})
    return {"ok": True, "contact": row}


def link_people_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    keep_id = _required_text(payload.get("keep_id"), "keep_id")
    merge_id = _required_text(payload.get("merge_id"), "merge_id")
    try:
        result = store.link_people(keep_id, merge_id)
    except (KeyError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("people.changed", {"kind": "person", "action": "linked",
                                       "keep_id": keep_id, "merge_id": merge_id})
    return result


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


def _scope_detail(name: str, *, roles_holding: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    spec = SCOPES.get(name, {})
    return {
        "name": name,
        "description": spec.get("description") or name,
        "purpose": spec.get("purpose") or spec.get("description") or name,
        "tools": list(spec.get("tools") or []),
        "tools_count": len(spec.get("tools") or []),
        "group": SCOPE_SYSTEM.get(name, "other"),
        "kind": _scope_kind(name, spec),
        "read": _scope_kind(name, spec) == "read",
        "roles": sorted((roles_holding or {}).get(name, [])),
    }


_SEND_WARNING = (
    "People in this role can have Blatbot send messages without Aaron seeing them first."
)


def _dedupe_members_by_person(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`store.trusted()` is one row per CONTACT; a role's member list must be one row
    per PERSON, with every contact they hold shown inside that one entry -- never a
    second entry for the same person because they have a second phone or Telegram id."""
    by_person: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in rows:
        person_id = r.get("person_id") or r["key"]  # legacy rows with no person_id stand alone
        if person_id not in by_person:
            by_person[person_id] = {"key": r["key"], "person": r.get("person") or r["key"], "keys": []}
            order.append(person_id)
        by_person[person_id]["keys"].append(r["key"])
    return [by_person[pid] for pid in order]


def _role_members(store: Store, name: str) -> List[Dict[str, Any]]:
    rows = [row for row in store.trusted() if name in role_names(row.get("role"))]
    return _dedupe_members_by_person(rows)


def _role_model(store: Store, row: Dict[str, Any], people_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    scope_names = row.get("scopes") or []
    details = [_scope_detail(name) for name in scope_names]
    read_scopes = [d for d in details if d["kind"] == "read"]
    write_scopes = [d for d in details if d["kind"] in ("write", "send")]
    has_send = any(d["kind"] == "send" for d in details)
    matching = [
        r for r in (people_rows if people_rows is not None else store.trusted())
        if row["name"] in role_names(r.get("role"))
    ]
    members = _dedupe_members_by_person(matching)
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


def add_role_scope(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add one scope to a role without touching the rest of its scopes."""
    name = _required_text(payload.get("name"), "name").lower()
    scope_name = _required_text(payload.get("scope"), "scope")
    role = next((r for r in store.roles() if r["name"] == name), None)
    if role is None:
        raise ValidationError(f"no such role: {name}")
    if scope_name not in SCOPES:
        raise ValidationError(f"unknown scope: {scope_name}")
    scopes = sorted(set(role.get("scopes") or []) | {scope_name})
    store.set_role(name, scopes, role.get("note") or "")
    row = next(item for item in roles(store) if item["name"] == name)
    events.publish("permissions.changed", {"kind": "role", "name": name, "scope_added": scope_name})
    return row


def remove_role_scope(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove one scope from a role without touching the rest of its scopes."""
    name = _required_text(payload.get("name"), "name").lower()
    scope_name = _required_text(payload.get("scope"), "scope")
    role = next((r for r in store.roles() if r["name"] == name), None)
    if role is None:
        raise ValidationError(f"no such role: {name}")
    scopes = [s for s in (role.get("scopes") or []) if s != scope_name]
    store.set_role(name, scopes, role.get("note") or "")
    row = next(item for item in roles(store) if item["name"] == name)
    events.publish("permissions.changed", {"kind": "role", "name": name, "scope_removed": scope_name})
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
    held = role_names((existing or {}).get("role"))
    store.set_trust(key, person=person, role=", ".join(held + [name]), scopes=scopes, note=note)
    events.publish("permissions.changed", {"kind": "role", "name": name, "member_added": key})
    return next(item for item in roles(store) if item["name"] == name)


def remove_role_member(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _required_text(payload.get("name"), "name").lower()
    key = _required_text(payload.get("key"), "key")
    existing = next((row for row in store.trusted() if row["key"] == store.trusted_key(key)), None)
    if existing and name in role_names(existing.get("role")):
        kept = [r for r in role_names(existing.get("role")) if r != name]
        store.set_trust(key, person=existing.get("person") or "", role=", ".join(kept),
                         scopes=existing.get("scopes") or [], note=existing.get("note") or "")
    events.publish("permissions.changed", {"kind": "role", "name": name, "member_removed": key})
    return next(item for item in roles(store) if item["name"] == name)


def _roles_holding_scope(store: Store) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for role in store.roles():
        for scope_name in role.get("scopes") or []:
            out.setdefault(scope_name, []).append(role["name"])
    return out


def scopes(store: Optional[Store] = None) -> List[Dict[str, Any]]:
    """Every scope in the registry, flat. `store` is optional so this can still be
    called without one; pass it to get which roles currently hold each scope."""
    roles_holding = _roles_holding_scope(store) if store is not None else {}
    return [_scope_detail(name, roles_holding=roles_holding) for name in SCOPES]


def scopes_breakdown(store: Optional[Store] = None) -> List[Dict[str, Any]]:
    """Every scope grouped by the system it belongs to (`SCOPE_TREE_SYSTEMS` in
    scopes.py / `systems` in scopes.yaml), for the console's scopes page and for the
    per-role scope picker -- the same grouped list drives both."""
    from ..gate.scopes import SCOPE_TREE_SYSTEMS
    flat = scopes(store)
    by_system: Dict[str, List[Dict[str, Any]]] = {}
    for detail in flat:
        by_system.setdefault(detail["group"], []).append(detail)
    out = []
    for system_name in sorted(set(SCOPE_TREE_SYSTEMS) | set(by_system)):
        out.append({
            "system": system_name,
            "description": SCOPE_TREE_SYSTEMS.get(system_name, ""),
            "scopes": sorted(by_system.get(system_name, []), key=lambda d: d["name"]),
        })
    return out


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
