"""Framework-free console read models and command wrappers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..gate.scopes import SCOPES, SCOPE_SYSTEM
from ..gate.store import Request, Store, role_names
from ..gate import cadence as cadence_mod, schedprop
from ..gate.schedules import DEFAULT_TIMEZONE
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

def _person_display(store: Store, person_id: Optional[str]) -> str:
    if not person_id:
        return ""
    row = store.get_person(person_id)
    return (row or {}).get("display") or person_id


def _person_model(store: Store, row: Dict[str, Any]) -> Dict[str, Any]:
    roles_by_name = {r["name"]: r for r in store.roles()}
    role_scopes = [s for name in role_names(row.get("role"))
                   for s in roles_by_name.get(name, {}).get("scopes", [])]
    effective = sorted(set(role_scopes) | set(row.get("scopes") or []))
    is_commander = bool(row.get("id")) and row["id"] == store.commander_person_id()
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
        # The front end acts only on this flag, never on a role name: it is this
        # one person, their "commander" role membership cannot be changed here,
        # and their delete/merge-away controls must not be offered.
        "commander": is_commander,
    }


def people_directory(store: Store) -> Dict[str, Any]:
    """Every live person with their contacts, plus every contact seen in traffic or
    sync that has not been attached to anyone yet."""
    return {
        "people": [_person_model(store, p) for p in store.list_people()],
        "unlinked": store.unlinked_contacts(),
        # The ways a person can be reached, in the order the console offers
        # them, read from the contact_kinds table (store.py seeds it once at
        # startup) rather than a hard-coded list -- the console holds no list
        # of its own: labels and input hints come from here. "name" exists as
        # a stored kind for display names seen in traffic but is not
        # something a person is reached on, so it cannot be added by hand.
        "contact_kinds": store.contact_kinds(),
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


def _validate_contact_kind(store: Store, kind: Any, *, require_addable: bool = False) -> str:
    kind = _required_text(kind, "kind").lower()
    known = store.known_contact_kind(kind)
    if known is None:
        raise ValidationError(f"unknown contact kind: {kind}")
    if require_addable and not known["addable"]:
        raise ValidationError(f"{kind} is not a kind that can be added by hand")
    return kind


def add_contact_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    person_id = _required_text(payload.get("person_id"), "person_id")
    kind = _validate_contact_kind(store, payload.get("kind"), require_addable=True)
    value = _required_text(payload.get("value"), "value")
    result = store.upsert_contact(kind, value, person_id=person_id, source="manual")
    if result["conflict"]:
        events.publish("people.changed", {"kind": "contact", "action": "conflict",
                                           "contact_id": result["contact"]["id"]})
        return {"ok": False, "conflict": True, "existing_person_id": result["existing_person_id"],
                "existing_person_display": _person_display(store, result["existing_person_id"]),
                "contact": result["contact"]}
    events.publish("people.changed", {"kind": "contact", "person_id": person_id, "action": "added"})
    return {"ok": True, "conflict": False, "contact": result["contact"]}


def update_contact_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    contact_id = _required_text(payload.get("contact_id"), "contact_id")
    kind = _validate_contact_kind(store, payload.get("kind"), require_addable=True) if payload.get("kind") else None
    value = _text(payload.get("value"), "value") if "value" in payload else None
    if value == "":
        raise ValidationError("value is required")
    try:
        result = store.update_contact(contact_id, kind=kind, value=value)
    except KeyError as exc:
        raise ValidationError(str(exc)) from exc
    if result["conflict"]:
        return {"ok": False, "conflict": True, "existing_person_id": result["existing_person_id"],
                "existing_person_display": _person_display(store, result["existing_person_id"]),
                "contact": result["contact"]}
    events.publish("people.changed", {"kind": "contact", "contact_id": contact_id, "action": "updated"})
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


LOCKED_ROLES = frozenset({"commander"})
_LOCKED_ROLE_DESCRIPTION = {
    "commander": "Everything, on verified channels only.",
}


def _require_unlocked_role(name: str) -> None:
    """Member add/remove, delete, rename and scope changes are refused for a
    locked (system) role at the API layer -- a clear 400 message -- as well as
    inside the store itself, which is the layer that cannot be bypassed."""
    if name in LOCKED_ROLES:
        raise ValidationError(f"{name} is a built-in role and cannot be changed here")


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
    locked = row["name"] in LOCKED_ROLES
    return {
        "name": row["name"],
        "note": row.get("note") or "",
        "scopes": scope_names,
        "read_scopes": read_scopes,
        "write_scopes": write_scopes,
        "members": members,
        "people_count": len(members),
        "warning": _SEND_WARNING if has_send else "",
        # The front end acts only on this flag (and the text below), never on
        # the role's name: locked means not editable, not deletable, and its
        # membership cannot be changed from here.
        "locked": locked,
        "description": _LOCKED_ROLE_DESCRIPTION.get(row["name"], "") if locked else "",
    }


def roles(store: Store) -> List[Dict[str, Any]]:
    people_rows = store.trusted()
    return [_role_model(store, row, people_rows) for row in store.roles()]


def upsert_role(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _required_text(payload.get("name"), "name").lower()
    _require_unlocked_role(name)
    store.set_role(name, _scopes(payload.get("scopes")), _text(payload.get("note"), "note"))
    row = next(item for item in roles(store) if item["name"] == name)
    events.publish("permissions.changed", {"kind": "role", "name": name})
    return row


def rename_role_endpoint(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    old_name = _required_text(payload.get("name"), "name").lower()
    new_name = _required_text(payload.get("new_name"), "new_name").lower()
    _require_unlocked_role(old_name)
    _require_unlocked_role(new_name)
    try:
        store.rename_role(old_name, new_name)
    except (KeyError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc
    row = next(item for item in roles(store) if item["name"] == new_name)
    events.publish("permissions.changed", {"kind": "role", "name": new_name, "renamed_from": old_name})
    return row


def add_role_scope(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add one scope to a role without touching the rest of its scopes."""
    name = _required_text(payload.get("name"), "name").lower()
    _require_unlocked_role(name)
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
    _require_unlocked_role(name)
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
    _require_unlocked_role(name)
    store.drop_role(name)
    events.publish("permissions.changed", {"kind": "role", "name": name, "deleted": True})
    return {"ok": True, "name": name}


def add_role_member(store: Store, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _required_text(payload.get("name"), "name").lower()
    _require_unlocked_role(name)
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
    _require_unlocked_role(name)
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
    from ..gate.scopes import SCOPE_TREE_ALWAYS, SCOPE_TREE_LEAVES, SCOPE_TREE_SYSTEMS
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
            # the decision tree the gate walks: scopes granted whenever the system is
            # picked, then one branch per question with the scopes a yes grants
            "always": list(SCOPE_TREE_ALWAYS.get(system_name, [])),
            "leaves": [{"name": leaf, "question": question, "grants": list(grants)}
                       for leaf, (question, grants) in SCOPE_TREE_LEAVES.get(system_name, {}).items()],
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


# ---------------------------------------------------------------------------
# Schedules: list/create/edit/action/approve + run history, shaped exactly to
# the "Console API contract" section of SPEC.md. Nothing here is encoded in a
# front end -- cadence kinds, capability options, report wording options and
# weekday names are all served from here so a console can render its forms
# with nothing hardcoded.
# ---------------------------------------------------------------------------

CADENCE_KIND_FIELDS: Dict[str, List[str]] = {
    "once": ["at"], "every": ["minutes"], "daily": ["time"],
    "weekdays": ["time"], "weekly": ["day", "time"], "cron": ["expr"],
}
CADENCE_KIND_LABELS: Dict[str, str] = {
    "once": "Once", "every": "Every N minutes or hours", "daily": "Every day",
    "weekdays": "Weekdays", "weekly": "Weekly", "cron": "Cron (advanced)",
}
WEEKDAY_OPTIONS: List[Dict[str, str]] = [
    {"value": "mon", "label": "Monday"}, {"value": "tue", "label": "Tuesday"},
    {"value": "wed", "label": "Wednesday"}, {"value": "thu", "label": "Thursday"},
    {"value": "fri", "label": "Friday"}, {"value": "sat", "label": "Saturday"},
    {"value": "sun", "label": "Sunday"},
]
OUTCOME_LABELS: Dict[str, str] = {
    "complete": "Finished", "incomplete": "Could not finish", "failed": "Failed",
    "interrupted": "Interrupted", "skipped": "Skipped",
}
# Plain words for the handful of US zones the owner actually uses; anything
# else falls back to the last path segment of the IANA name ("Europe/London"
# -> "London") rather than guessing at a word that may be wrong.
_TIMEZONE_LABELS: Dict[str, str] = {
    "America/New_York": "Eastern", "America/Chicago": "Central",
    "America/Denver": "Mountain", "America/Los_Angeles": "Pacific",
    "America/Anchorage": "Alaska", "Pacific/Honolulu": "Hawaii", "UTC": "UTC",
}
# ended-state last_outcome values the owner caused directly, versus any other
# text that led a schedule to stop (reached its until date, the model's own
# STATUS: STOP, an unanswered proposal expiring). There is no separate "why"
# column in the data model, so this is an inference from that free-text field.
_OWNER_STOPPED_OUTCOMES = {"stopped", "rejected"}


def _timezone_label(timezone_name: str) -> str:
    if timezone_name in _TIMEZONE_LABELS:
        return _TIMEZONE_LABELS[timezone_name]
    return timezone_name.rsplit("/", 1)[-1].replace("_", " ")


def _schedule_time_text(ts: Optional[float], timezone_name: str) -> Optional[str]:
    """"Mon Oct 12, 2:33 AM EDT" in the schedule's own time zone. None in, None out."""
    if ts is None:
        return None
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        tz = ZoneInfo(timezone_name or DEFAULT_TIMEZONE)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)
    return datetime.fromtimestamp(ts, tz).strftime("%a %b %-d, %-I:%M %p %Z")


def _cadence_text(schedule: Any) -> str:
    try:
        return cadence_mod.describe(schedule.cadence_kind, schedule.cadence, schedule.timezone)
    except cadence_mod.CadenceError:
        return "not set"  # a legacy/paused row with no cadence model yet


def _capability_list(scopes: Any) -> List[Dict[str, str]]:
    return [{"scope": s, "label": schedprop.label(s)} for s in scopes]


def _scope_for_tool(tool: str) -> Optional[str]:
    for name, info in SCOPES.items():
        if tool in (info.get("tools") or []):
            return name
    return None


def _denied_list(denied: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in denied or []:
        tool = str(item.get("tool")) if isinstance(item, dict) else str(item)
        scope = _scope_for_tool(tool)
        out.append({"tool": tool, "label": schedprop.label(scope) if scope else tool})
    return out


def _run_view(run: Dict[str, Any], timezone_name: str) -> Dict[str, Any]:
    started = run.get("started_at")
    finished = run.get("finished_at")
    duration = (finished - started) if (started is not None and finished is not None) else None
    outcome = str(run.get("outcome") or "")
    return {
        "id": run["id"], "started_at": started, "started_text": _schedule_time_text(started, timezone_name),
        "duration_seconds": duration, "outcome": outcome,
        "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
        "result_line": run.get("result_line") or "", "final_status": run.get("final_status") or "",
        "denied": _denied_list(run.get("denied")), "note": run.get("note") or "",
        "cost_usd": run.get("cost_usd"), "request_id": run.get("request_id"),
    }


def _last_non_skipped_run(store: Store, schedule_id: int) -> Optional[Dict[str, Any]]:
    for row in store.schedule_runs(schedule_id):
        if row.get("outcome") != "skipped":
            return row
    return None


def _state_label(schedule: Any, running: bool) -> str:
    if schedule.state == "proposed":
        return "Needs your yes"
    if schedule.state == "active":
        return "Running" if running else "Waiting"
    if schedule.state == "paused":
        return "Paused"
    if schedule.state == "ended":
        return "Stopped" if schedule.last_outcome in _OWNER_STOPPED_OUTCOMES else "Ended"
    return schedule.state


def _paused_all(store: Store) -> bool:
    return str(store.settings().get("GATE_SCHEDULES_PAUSED", "0")).lower() in {"1", "true", "yes", "on"}


def _schedule_view(store: Store, service: Any, schedule: Any) -> Dict[str, Any]:
    running = store.request_running_for_schedule(schedule.id) is not None
    missing = schedprop.missing_capabilities(schedule.prompt, schedule.scopes)
    last = _last_non_skipped_run(store, schedule.id)
    # A schedule that is not active has no next run, whatever is still sitting in the
    # next_run column (paused/ended clear it on their own route in, but this is belt
    # and suspenders: the console never shows a time that will not actually fire).
    next_run = schedule.next_run if schedule.state == "active" else None
    return {
        "id": schedule.id, "title": schedule.title, "prompt": schedule.prompt,
        "state": schedule.state, "state_label": _state_label(schedule, running),
        "cadence_kind": schedule.cadence_kind, "cadence": schedule.cadence,
        "cadence_text": _cadence_text(schedule), "timezone": schedule.timezone,
        "timezone_label": _timezone_label(schedule.timezone),
        "next_run": next_run, "next_run_text": _schedule_time_text(next_run, schedule.timezone),
        "next_run_in_seconds": (next_run - time.time()) if next_run is not None else None,
        "report_every_time": schedule.report_every_time,
        "report_text": ("Every time it runs" if schedule.report_every_time
                        else "Only when something changed or it could not finish"),
        "until": schedule.until,
        "capabilities": _capability_list(schedule.scopes),
        "missing_capabilities": _capability_list(missing),
        "last_run": _run_view(last, schedule.timezone) if last else None,
        "needs_approval": schedule.state == "proposed",
        "approval_request_id": None,
        "actions": service.valid_actions(schedule),
    }


def schedule_list(store: Store, service: Any) -> Dict[str, Any]:
    rows = [_schedule_view(store, service, schedule) for schedule in store.schedules()]
    capability_options = [
        {"scope": name, "label": schedprop.label(name), "tools": list(info.get("tools") or [])}
        for name, info in SCOPES.items()
    ]
    cadence_kinds = [
        {"kind": kind, "label": CADENCE_KIND_LABELS[kind], "fields": CADENCE_KIND_FIELDS[kind]}
        for kind in ("once", "every", "daily", "weekdays", "weekly", "cron")
    ]
    report_options = [
        {"value": False, "label": "Only when something changed or it could not finish"},
        {"value": True, "label": "Every time it runs"},
    ]
    return {
        "schedules": rows, "paused_all": _paused_all(store), "cadence_kinds": cadence_kinds,
        "capability_options": capability_options, "report_options": report_options,
        "weekdays": WEEKDAY_OPTIONS,
    }


def schedule_runs_view(store: Store, schedule_id: int) -> Optional[Dict[str, Any]]:
    schedule = store.get_schedule(schedule_id)
    if schedule is None:
        return None
    rows = store.schedule_runs(schedule.id)
    return {"runs": [_run_view(r, schedule.timezone) for r in rows]}


def _valid_timezone(value: Any) -> Optional[str]:
    text = _text(value, "timezone")
    if not text:
        return None
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        ZoneInfo(text)
    except ZoneInfoNotFoundError as exc:
        raise ValidationError(f"unknown timezone: {text}") from exc
    return text


def _until(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError("until must be a number (epoch seconds) or null") from None


def _cadence_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cadence = payload.get("cadence")
    if not isinstance(cadence, dict):
        raise ValidationError("cadence must be an object")
    return cadence


class ScheduleActionConflict(ValueError):
    """An action/approve call that is not available for a schedule's current
    state. Kept distinct from ValidationError (malformed input) because the
    console contract requires HTTP 409 here, not 400."""


async def create_schedule(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    title = _required_text(payload.get("title"), "title")
    prompt = _required_text(payload.get("prompt"), "prompt")
    cadence_kind = _required_text(payload.get("cadence_kind"), "cadence_kind")
    cadence = _cadence_payload(payload)
    try:
        cadence_mod.validate(cadence_kind, cadence)
    except cadence_mod.CadenceError as exc:
        raise ValidationError(str(exc)) from exc
    timezone = _valid_timezone(payload.get("timezone"))
    report_every_time = bool(payload.get("report_every_time"))
    until = _until(payload.get("until"))
    scopes = _scopes(payload.get("scopes"))
    task_id = manager.store.create_task(title)["id"]
    try:
        schedule = await manager.schedules.propose(
            owner=True, chat_id="owner", task_id=task_id, title=title, prompt=prompt,
            cadence_kind=cadence_kind, cadence=cadence, timezone=timezone,
            report_every_time=report_every_time, until=until, scopes=scopes,
            origin_mode="console", origin_chat_id="owner",
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("schedules.changed", {"id": schedule.id, "state": schedule.state})
    message = schedprop.format_proposal(schedule)
    return {"schedule": _schedule_view(manager.store, manager.schedules, schedule),
            "applied": False, "needs_approval": True, "message": message}


async def edit_schedule(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        schedule_id = int(payload.get("id"))
    except (TypeError, ValueError):
        raise ValidationError("id must be an integer") from None
    changes: Dict[str, Any] = {}
    if "title" in payload:
        changes["title"] = _required_text(payload.get("title"), "title")
    if "cadence_kind" in payload or "cadence" in payload:
        old = manager.store.get_schedule(schedule_id)
        if old is None:
            raise ValidationError(f"no such schedule: {schedule_id}")
        kind = payload.get("cadence_kind", old.cadence_kind)
        cadence = payload.get("cadence", old.cadence)
        if not isinstance(cadence, dict):
            raise ValidationError("cadence must be an object")
        try:
            cadence_mod.validate(kind, cadence)
        except cadence_mod.CadenceError as exc:
            raise ValidationError(str(exc)) from exc
        changes["cadence_kind"] = kind
        changes["cadence"] = cadence
    if "report_every_time" in payload:
        changes["report_every_time"] = bool(payload.get("report_every_time"))
    if "until" in payload:
        changes["until"] = _until(payload.get("until"))
    if "prompt" in payload:
        changes["prompt"] = _required_text(payload.get("prompt"), "prompt")
    if "scopes" in payload:
        changes["scopes"] = _scopes(payload.get("scopes"))
    try:
        result = await manager.schedules.edit(schedule_id, owner=True, changes=changes)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("schedules.changed", {"id": result["schedule"].id, "state": result["schedule"].state})
    return {"schedule": _schedule_view(manager.store, manager.schedules, result["schedule"]),
            "applied": bool(result["applied"]), "needs_approval": bool(result["needs_approval"]),
            "message": result["message"]}


async def schedule_action(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        schedule_id = int(payload.get("id"))
    except (TypeError, ValueError):
        raise ValidationError("id must be an integer") from None
    action = _required_text(payload.get("action"), "action")
    try:
        schedule = await manager.schedules.action(schedule_id, action, owner=True)
    except ValueError as exc:
        raise ScheduleActionConflict(str(exc)) from exc
    events.publish("schedules.changed", {"id": schedule.id, "action": action})
    return {"schedule": _schedule_view(manager.store, manager.schedules, schedule),
            "message": f"{action.replace('_', ' ')} done."}


def schedules_global(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """The one remaining global knob: pause/resume every schedule's firing at once.
    Not part of the Console API contract's enumerated endpoints, but paused_all in
    the schedules list reads this setting, and the console keeps a toggle for it."""
    paused = payload.get("paused")
    if not isinstance(paused, bool):
        raise ValidationError("paused must be true or false")
    manager.store.set_setting("GATE_SCHEDULES_PAUSED", "1" if paused else "0")
    events.publish("schedules.changed", {"global_paused": paused})
    return schedule_list(manager.store, manager.schedules)


async def approve_schedule(manager: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        schedule_id = int(payload.get("id"))
    except (TypeError, ValueError):
        raise ValidationError("id must be an integer") from None
    approve = payload.get("approve")
    if not isinstance(approve, bool):
        raise ValidationError("approve must be true or false")
    try:
        schedule = await manager.schedules.decide(schedule_id, approve)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    events.publish("schedules.changed", {"id": schedule.id, "state": schedule.state})
    message = "Started." if approve else "Not started."
    return {"schedule": _schedule_view(manager.store, manager.schedules, schedule), "message": message}


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

    # schedule_list() builds the full console view (actions, capabilities, run
    # history) via ScheduleService; Home only needs the worded cadence and the
    # next few firings, so it reads the store directly rather than paying for
    # that whole shape.
    active_schedules = store.schedules(states=["active"])
    now = time.time()
    upcoming = sorted(
        (
            {
                "id": s.id,
                "title": s.title,
                "cadence": _cadence_text(s),
                "next_run": s.next_run,
                "state": s.state,
            }
            for s in active_schedules
            if s.next_run and s.next_run >= now
        ),
        key=lambda s: s["next_run"],
    )[:5]
    coming_up = {
        "schedules_paused": _paused_all(store),
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
