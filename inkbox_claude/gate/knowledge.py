"""One rule for Aaron's private knowledge: the vault (his Obsidian second brain) and
Blatbot's long-term memory (gate/graphmem.py).

Who reads what:

  Aaron, in his own chats      everything: every vault folder, every memory group.
  anyone else, in a direct chat what their role or their own record grants: `vault:<folder>`
                                entries for notes, `memory:` entries for memory beyond their
                                own conversations (which they always reach). The whole vault
                                (`vault_read`) only with Aaron's yes for that one task.
  a group chat, whoever speaks  the group's own conversation and nothing else: no vault
                                notes, no other chat's memory, no task from outside the
                                group. Others are present and there is no membership list
                                to prove each of them may see more, so Aaron speaking in a
                                group gets exactly what Stanley speaking there gets. A
                                vault folder reaches a group run only when Aaron says yes
                                to that exact folder for that one task.

Writes: every stored message is filed in the memory group of the chat it was said in
(graphmem.record), so a person's conversation is never filed under Aaron's private
group, and Aaron's private chat is never filed under anyone else's. Nothing writes to
the vault: its tools are read-only.

This module only answers the questions; graphmem.readable_groups, the manager's ledger
reads and its request-scope step call it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

WHOLE_VAULT = "vault_read"
FOLDER_PREFIX = "vault:"


def is_vault_scope(name: Any) -> bool:
    n = str(name or "")
    return n == WHOLE_VAULT or n.startswith(FOLDER_PREFIX)


def meta_is_group(meta: Any) -> bool:
    """A thread's stored metadata says it is a room with several people in it."""
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except Exception:  # noqa: BLE001
            return False
    if not isinstance(meta, dict):
        return False
    return meta.get("conversation_kind") == "group" or bool(meta.get("group_title"))


def chat_is_group(store: Any, chat_id: str) -> bool:
    """Is this stored chat a group? Unknown chats are treated as direct chats; callers that
    must fail closed check the live message's own metadata as well."""
    if not chat_id or store is None:
        return False
    try:
        route = store.thread_route(chat_id) or {}
    except Exception:  # noqa: BLE001
        return False
    return meta_is_group(route.get("meta") or {})


def strip_for_group(scopes: Iterable[str]) -> List[str]:
    """A group run's scopes without any vault scope. Long-term memory stays: the
    memory_search tool in a group reads that group's own conversation only."""
    return [s for s in scopes if not is_vault_scope(s)]


def group_dropped(scopes: Iterable[str]) -> List[str]:
    return [s for s in scopes if is_vault_scope(s)]


def blanket_excluded(name: str) -> bool:
    """Scopes a "yes to everything" never adds by itself: every vault scope. Each folder is
    Aaron's own writing; it is opened only when a run was refused that exact folder."""
    return is_vault_scope(name)


def askable_vault_folders(scopes_map: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(n for n in scopes_map if n.startswith(FOLDER_PREFIX))


def folder_title(name: str, scopes_map: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """The folder a `vault:` scope covers, as Aaron's vault names it: 'Projects / Corgi'."""
    spec = (scopes_map or {}).get(name) or {}
    paths = spec.get("vault_paths") or []
    if paths:
        p = str(paths[-1]).rstrip("/")
        if p.lower().endswith(".md"):
            p = p[:-3]
        return " / ".join(part for part in p.split("/") if part)
    return " / ".join(part.replace("-", " ").title() for part in name[len(FOLDER_PREFIX):].split("/"))
