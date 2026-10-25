"""Compact task metadata and conservative thread continuation detection."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from .scopes import SCOPE_TREE_SYSTEMS, SCOPE_TREE_LEAVES, SCOPE_TREE_ALWAYS


def index_entry(row):
    title = row["title"]
    words = set(re.findall(r"[a-z0-9]+", title.lower()))
    systems = set()
    scopes = set(json.loads(row.get("scopes_json") or "[]"))
    for name in SCOPE_TREE_SYSTEMS:
        tokens = set(name.split("_")) - {"public", "workspace", "read", "write", "brain"}
        if words & tokens:
            systems.add(name)
    if not systems:
        for name in SCOPE_TREE_SYSTEMS:
            granted = set(SCOPE_TREE_ALWAYS.get(name, []))
            for _, leaf_scopes in SCOPE_TREE_LEAVES.get(name, {}).values():
                granted.update(leaf_scopes)
            if granted & scopes:
                systems.add(name)
        # Owner requests grant everything: this is authorization, not topic evidence.
        if len(systems) > 3:
            systems.clear()
    categories = sorted(systems) or ["general"]
    waiting = ("Aaron" if row["state"] == "waiting_aaron" else
               "assistant" if row["state"] == "running" else row.get("waiting_on") or "unspecified")
    date = datetime.fromtimestamp(row["updated_at"], timezone.utc).date().isoformat()
    return {**row, "categories": categories, "waiting_on": waiting,
            "line": f"T{row['id']} | {title} | {','.join(categories)} | {date} | {row['state']} | waiting: {waiting}"}


def render_index(entries):
    return "\n".join(t["line"] for t in entries)


def continuation_id(message, entries, recent_ids, history):
    """Only explicit IDs or unambiguous short continuations of a recent task."""
    allowed = {t["id"] for t in entries}
    ids = {int(x) for x in re.findall(r"\bT(\d+)\b", message, re.I)}
    if len(ids) == 1 and ids <= allowed:
        return next(iter(ids))
    if ids or not recent_ids or recent_ids[0] not in allowed or not history:
        return None
    # Require a recent conversation as well as a recent task; never guess from a greeting.
    last_time = history[-1].get("created_at", 0)
    import time
    if not last_time or time.time() - float(last_time) > 1800:
        return None
    if re.fullmatch(r"\s*(?:yes(?: please)?|go ahead|do that|try again|retry|continue|what(?:'s| is) the status|any update)[.!? ]*", message, re.I):
        return recent_ids[0]
    return None
