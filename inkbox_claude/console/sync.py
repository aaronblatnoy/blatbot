"""People sync: pull every counterpart Blatbot has ever talked to via Inkbox into the
console's people list, without ever touching trust or role assignment.

There is no single "list every conversation" call in the Inkbox API; a sync pages
through each channel the API exposes and folds the results together. The channel
paging itself is isolated behind the `InkboxDirectoryClient` protocol below so this
module can be exercised with a fake client in tests and, when a real paging client
exists in this package, wired to it without changing the merge logic.

Dedupe uses the exact same handle normalization trust keys use (`_trust_key`), so a
synced person and a trusted person always collide on the same row instead of
appearing twice under slightly different spellings of the same handle.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Protocol


class InkboxDirectoryClient(Protocol):
    """Whatever can page through Inkbox's channels. A fake implements this for tests;
    the real client (once one pages channels rather than just sending/receiving a single
    conversation) implements it for a live sync. No method here ever mutates anything."""

    def imessage_conversations(self) -> Iterable[Dict[str, Any]]:
        ...

    def sms_threads(self) -> Iterable[Dict[str, Any]]:
        ...

    def email_threads(self) -> Iterable[Dict[str, Any]]:
        ...

    def calls(self) -> Iterable[Dict[str, Any]]:
        ...

    def contacts(self) -> Iterable[Dict[str, Any]]:
        ...


_CHANNEL_METHODS = (
    ("imessage", "imessage_conversations"),
    ("sms", "sms_threads"),
    ("email", "email_threads"),
    ("call", "calls"),
    ("contact", "contacts"),
)


def _owner_and_bot_handles(settings: Dict[str, str]) -> set:
    """Trust-key-normalized handles to exclude: the owner's own numbers/addresses and
    Blatbot's own mailbox/number/iMessage line, so the directory never lists the owner
    or the bot as someone the bot has "spoken to". Normalized the same way trust keys
    are, so "+1 555-123-0000" and "5551230000" exclude the same handle."""
    from ..gate.store import _trust_key

    keys = (
        "owner_imessage_handle", "owner_phone", "owner_email",
        "blatbot_email", "blatbot_phone", "blatbot_imessage_handle",
    )
    out = set()
    for key in keys:
        value = (settings.get(key) or "").strip()
        if value:
            out.add(_trust_key(value))
    return out


def sync_people(store: Any, client: InkboxDirectoryClient) -> Dict[str, Any]:
    """Page every channel, dedupe by trust-key, and merge into the people directory.

    Never writes to the trust or roles tables: a synced person who is not already
    trusted ends up with no role and no scopes, exactly like any other unknown
    contact the console already shows.
    """
    from ..gate.store import _trust_key  # local import: keeps this module import-light

    exclude = _owner_and_bot_handles(store.settings() if hasattr(store, "settings") else {})
    merged: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}
    failed: List[str] = []
    now = time.time()

    for channel, method_name in _CHANNEL_METHODS:
        method = getattr(client, method_name, None)
        if method is None:
            continue
        try:
            rows = list(method() or [])
        except Exception:
            # One channel's paging blowing up (a flaky upstream page, a channel the
            # identity has no access to, whatever) must not take down the whole sync —
            # every other channel still gets collected and merged.
            failed.append(channel)
            continue
        counts[channel] = len(rows)
        for row in rows:
            handle = str(row.get("handle") or row.get("key") or "").strip()
            if not handle:
                continue
            key = _trust_key(handle)
            if key in exclude:
                continue
            name = str(row.get("name") or row.get("display") or "").strip()
            last_seen = float(row.get("last_seen") or 0)
            entry = merged.setdefault(key, {
                "key": key, "person": "", "channels": [], "last_seen": 0.0,
            })
            if name and (not entry["person"] or len(name) > len(entry["person"])):
                entry["person"] = name
            if channel not in entry["channels"]:
                entry["channels"].append(channel)
            entry["last_seen"] = max(entry["last_seen"], last_seen)

    # Union in every counterpart with a thread in the gate database, by the same key.
    if hasattr(store, "thread_counterparts"):
        for row in store.thread_counterparts() or []:
            handle = str(row.get("handle") or "").strip()
            if not handle:
                continue
            key = _trust_key(handle)
            if key in exclude:
                continue
            entry = merged.setdefault(key, {
                "key": key, "person": "", "channels": [], "last_seen": 0.0,
            })
            name = str(row.get("name") or "").strip()
            if name and (not entry["person"] or len(name) > len(entry["person"])):
                entry["person"] = name
            channel = str(row.get("channel") or "thread")
            if channel not in entry["channels"]:
                entry["channels"].append(channel)
            entry["last_seen"] = max(entry["last_seen"], float(row.get("last_seen") or 0))

    if hasattr(store, "remember_synced_people"):
        store.remember_synced_people(list(merged.values()))
    if hasattr(store, "set_setting"):
        store.set_setting("people_sync_last_at", str(now))
        store.set_setting("people_sync_counts", _dumps(counts))
        store.set_setting("people_sync_failed", _dumps(failed))

    return {
        "ok": True, "synced_at": now, "people": len(merged), "counts": counts,
        "failed": failed,
    }


def _dumps(value: Any) -> str:
    import json
    return json.dumps(value)


def last_sync(store: Any) -> Dict[str, Any]:
    settings = store.settings() if hasattr(store, "settings") else {}
    at = settings.get("people_sync_last_at")
    counts_raw = settings.get("people_sync_counts")
    failed_raw = settings.get("people_sync_failed")
    counts: Dict[str, int] = {}
    failed: List[str] = []
    if counts_raw:
        import json
        try:
            counts = json.loads(counts_raw)
        except ValueError:
            counts = {}
    if failed_raw:
        import json
        try:
            failed = json.loads(failed_raw)
        except ValueError:
            failed = []
    return {"at": float(at) if at else None, "counts": counts, "failed": failed}
