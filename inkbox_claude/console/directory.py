"""Real `InkboxDirectoryClient` implementation, backed by the `inkbox` SDK.

Each method pages one Inkbox channel to exhaustion and yields plain dicts shaped
the way `console.sync.sync_people` expects: `{"handle", "name", "last_seen"}`.

Paging contract per channel, read from `inkbox/agent_identity.py` and
`inkbox/contacts/resources/contacts.py` in the installed `inkbox==0.5.14` SDK:

- `AgentIdentity.list_imessage_conversations(limit=, offset=)` and
  `AgentIdentity.list_text_conversations(limit=, offset=)` return a plain
  `list[...]` (no cursor/has_more field), so exhaustion is detected by the
  classic offset/limit convention: keep raising `offset` by `limit` until a
  page comes back shorter than `limit` requested.
- `AgentIdentity.list_calls(limit=, offset=)` pages the same way.
- `AgentIdentity.iter_emails(page_size=)` already returns an `Iterator[Message]`
  that pages internally; it is exhausted simply by iterating it to the end.
- `Inkbox.contacts.list(limit=, offset=)` (org-wide, not identity-scoped) pages
  the same offset/limit way as calls/conversations.

This module never writes to Inkbox and never mutates trust/role state — it only
reads and yields rows for `sync_people` to merge.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator

_PAGE = 100


def _ts(value: Any) -> float:
    """Epoch seconds for a `datetime`/`None`/already-numeric value."""
    if value is None:
        return 0.0
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except (OverflowError, OSError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class InkboxSDKDirectoryClient:
    """Pages every Inkbox channel this agent identity (plus the org's contacts
    resource) exposes, for the console's "people Blatbot has spoken to" sync.

    `inkbox` is the connected `inkbox.Inkbox` client, `identity` the
    `inkbox.agent_identity.AgentIdentity` the gateway logged in as.
    """

    def __init__(self, inkbox: Any, identity: Any) -> None:
        self._inkbox = inkbox
        self._identity = identity

    # -- iMessage ---------------------------------------------------------

    def imessage_conversations(self) -> Iterable[Dict[str, Any]]:
        offset = 0
        while True:
            page = self._identity.list_imessage_conversations(
                limit=_PAGE, offset=offset, include_groups=True,
            )
            if not page:
                break
            for conv in page:
                last_seen = _ts(getattr(conv, "latest_message_at", None))
                participants = getattr(conv, "participants", None) or []
                remote = getattr(conv, "remote_number", None)
                handles = participants if participants else ([remote] if remote else [])
                for handle in handles:
                    if not handle:
                        continue
                    yield {"handle": str(handle), "name": "", "last_seen": last_seen}
            if len(page) < _PAGE:
                break
            offset += _PAGE

    # -- SMS ----------------------------------------------------------------

    def sms_threads(self) -> Iterable[Dict[str, Any]]:
        offset = 0
        while True:
            page = self._identity.list_text_conversations(
                limit=_PAGE, offset=offset, include_groups=True,
            )
            if not page:
                break
            for conv in page:
                last_seen = _ts(getattr(conv, "latest_message_at", None))
                participants = getattr(conv, "participants", None) or []
                remote = getattr(conv, "remote_phone_number", None)
                handles = participants if participants else ([remote] if remote else [])
                for handle in handles:
                    if not handle:
                        continue
                    yield {"handle": str(handle), "name": "", "last_seen": last_seen}
            if len(page) < _PAGE:
                break
            offset += _PAGE

    # -- Email ----------------------------------------------------------------

    def email_threads(self) -> Iterable[Dict[str, Any]]:
        mine = {
            str(a).strip().lower()
            for a in (getattr(getattr(self._identity, "mailbox", None), "email_address", None),)
            if a
        }
        messages: Iterator[Any] = self._identity.iter_emails(page_size=_PAGE)
        for msg in messages:
            last_seen = _ts(getattr(msg, "created_at", None))
            direction = str(getattr(msg, "direction", "") or "")
            from_address = getattr(msg, "from_address", None)
            to_addresses = getattr(msg, "to_addresses", None) or []
            counterparts = []
            if direction.endswith("inbound") or direction == "in":
                if from_address and str(from_address).strip().lower() not in mine:
                    counterparts.append(from_address)
            else:
                counterparts.extend(
                    a for a in to_addresses if a and str(a).strip().lower() not in mine
                )
            if not counterparts and from_address and str(from_address).strip().lower() not in mine:
                counterparts.append(from_address)
            for addr in counterparts:
                yield {"handle": str(addr), "name": "", "last_seen": last_seen}

    # -- Calls ----------------------------------------------------------------

    def calls(self) -> Iterable[Dict[str, Any]]:
        offset = 0
        while True:
            page = self._identity.list_calls(limit=_PAGE, offset=offset)
            if not page:
                break
            for call in page:
                remote = getattr(call, "remote_phone_number", None)
                if not remote:
                    continue
                last_seen = _ts(
                    getattr(call, "ended_at", None)
                    or getattr(call, "started_at", None)
                    or getattr(call, "created_at", None)
                )
                yield {"handle": str(remote), "name": "", "last_seen": last_seen}
            if len(page) < _PAGE:
                break
            offset += _PAGE

    # -- Org-wide contacts ------------------------------------------------

    def contacts(self) -> Iterable[Dict[str, Any]]:
        contacts_resource = getattr(self._inkbox, "contacts", None)
        if contacts_resource is None:
            return
        offset = 0
        while True:
            page = contacts_resource.list(limit=_PAGE, offset=offset)
            if not page:
                break
            for contact in page:
                name = str(getattr(contact, "preferred_name", None) or "").strip()
                last_seen = _ts(getattr(contact, "updated_at", None) or getattr(contact, "created_at", None))
                for email in getattr(contact, "emails", None) or []:
                    value = getattr(email, "value", None)
                    if value:
                        yield {"handle": str(value), "name": name, "last_seen": last_seen}
                for phone in getattr(contact, "phones", None) or []:
                    value = getattr(phone, "value", None)
                    if value:
                        yield {"handle": str(value), "name": name, "last_seen": last_seen}
            if len(page) < _PAGE:
                break
            offset += _PAGE


def build_directory_client(inkbox: Any, identity: Any) -> InkboxSDKDirectoryClient:
    """Factory the gateway calls once, at startup, after `identity` is logged in."""
    return InkboxSDKDirectoryClient(inkbox, identity)
