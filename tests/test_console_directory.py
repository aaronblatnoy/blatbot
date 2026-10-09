"""Tests for the real `InkboxSDKDirectoryClient`, against fake objects shaped
exactly like the `inkbox` SDK's real return types (IMessageConversationSummary,
TextConversationSummary, PhoneCall, Message, Contact/ContactEmail/ContactPhone),
not raw dicts — so a signature or field-name drift in the real SDK would break
these tests the same way it would break the real sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from inkbox_claude.console.directory import InkboxSDKDirectoryClient
from inkbox_claude.console import directory as directory_module
from inkbox_claude.console.sync import sync_people
from inkbox_claude.gate.store import Store


def make_store(tmp_path):
    return Store(str(tmp_path / "gate.db"))


def _dt(seconds_from_epoch: float) -> datetime:
    return datetime.fromtimestamp(seconds_from_epoch, tz=timezone.utc)


@dataclass
class FakeIMessageConversationSummary:
    id: str
    remote_number: str | None
    latest_message_at: datetime | None = None
    participants: list[str] | None = None
    is_group: bool = False


@dataclass
class FakeTextConversationSummary:
    remote_phone_number: str | None
    latest_message_at: datetime | None = None
    participants: list[str] | None = None
    is_group: bool = False


@dataclass
class FakePhoneCall:
    remote_phone_number: str | None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    created_at: datetime | None = None


@dataclass
class FakeMessage:
    direction: str
    from_address: str | None
    to_addresses: list[str] = field(default_factory=list)
    created_at: datetime | None = None


@dataclass
class FakeContactEmail:
    value: str


@dataclass
class FakeContactPhone:
    value: str


@dataclass
class FakeContact:
    preferred_name: str | None
    emails: list[FakeContactEmail] = field(default_factory=list)
    phones: list[FakeContactPhone] = field(default_factory=list)
    updated_at: datetime | None = None


class FakeContactsResource:
    def __init__(self, contacts: list[FakeContact]):
        self._contacts = contacts

    def list(self, *, limit: int, offset: int):
        return self._contacts[offset:offset + limit]


class FakeInkbox:
    def __init__(self, contacts: list[FakeContact]):
        self.contacts = FakeContactsResource(contacts)


class FakeIdentity:
    """Mimics `inkbox.agent_identity.AgentIdentity`'s paging surface exactly."""

    def __init__(
        self,
        imessage_pages=None,
        sms_pages=None,
        calls_pages=None,
        emails=None,
        mailbox_address: str | None = "bot@example.com",
        raise_on=None,
    ):
        self._imessage = imessage_pages or []
        self._sms = sms_pages or []
        self._calls = calls_pages or []
        self._emails = emails or []
        self.mailbox = type("Mailbox", (), {"email_address": mailbox_address})()
        self._raise_on = raise_on or set()

    def list_imessage_conversations(self, *, limit, offset, include_groups=True):
        if "imessage" in self._raise_on:
            raise RuntimeError("imessage upstream exploded")
        return self._imessage[offset:offset + limit]

    def list_text_conversations(self, *, limit, offset, include_groups=True):
        if "sms" in self._raise_on:
            raise RuntimeError("sms upstream exploded")
        return self._sms[offset:offset + limit]

    def list_calls(self, *, limit, offset):
        if "call" in self._raise_on:
            raise RuntimeError("calls upstream exploded")
        return self._calls[offset:offset + limit]

    def iter_emails(self, *, page_size):
        if "email" in self._raise_on:
            raise RuntimeError("email upstream exploded")
        return iter(self._emails)


@pytest.fixture(autouse=True)
def small_page(monkeypatch):
    # Force multi-page exhaustion with small fixtures instead of needing 100+ rows.
    monkeypatch.setattr(directory_module, "_PAGE", 2)


def test_pages_imessage_to_exhaustion_across_several_pages():
    now = 1_700_000_000.0
    convos = [
        FakeIMessageConversationSummary(id=str(i), remote_number=f"+1555000{i:04d}", latest_message_at=_dt(now + i))
        for i in range(5)  # 5 rows, page size 2 -> 3 pages (2, 2, 1)
    ]
    identity = FakeIdentity(imessage_pages=convos)
    client = InkboxSDKDirectoryClient(FakeInkbox([]), identity)
    rows = list(client.imessage_conversations())
    assert len(rows) == 5
    assert {r["handle"] for r in rows} == {c.remote_number for c in convos}


def test_imessage_group_conversation_yields_each_participant():
    convo = FakeIMessageConversationSummary(
        id="g1", remote_number=None, is_group=True,
        participants=["+15550001111", "+15550002222"],
        latest_message_at=_dt(1_700_000_000.0),
    )
    identity = FakeIdentity(imessage_pages=[convo])
    client = InkboxSDKDirectoryClient(FakeInkbox([]), identity)
    rows = list(client.imessage_conversations())
    assert {r["handle"] for r in rows} == {"+15550001111", "+15550002222"}


def test_sms_pages_to_exhaustion():
    now = 1_700_000_000.0
    convos = [
        FakeTextConversationSummary(remote_phone_number=f"+1777000{i:04d}", latest_message_at=_dt(now + i))
        for i in range(4)
    ]
    identity = FakeIdentity(sms_pages=convos)
    client = InkboxSDKDirectoryClient(FakeInkbox([]), identity)
    rows = list(client.sms_threads())
    assert len(rows) == 4


def test_calls_channel_empty_yields_nothing():
    identity = FakeIdentity(calls_pages=[])
    client = InkboxSDKDirectoryClient(FakeInkbox([]), identity)
    assert list(client.calls()) == []


def test_calls_pages_and_uses_remote_number_and_timestamp():
    now = 1_700_000_000.0
    calls = [
        FakePhoneCall(remote_phone_number=f"+1888000{i:04d}", ended_at=_dt(now + i))
        for i in range(3)
    ]
    identity = FakeIdentity(calls_pages=calls)
    client = InkboxSDKDirectoryClient(FakeInkbox([]), identity)
    rows = list(client.calls())
    assert len(rows) == 3
    assert all(r["last_seen"] > 0 for r in rows)


def test_email_excludes_own_mailbox_and_uses_direction_for_counterpart():
    now = 1_700_000_000.0
    messages = [
        FakeMessage(direction="inbound", from_address="alice@example.com", to_addresses=["bot@example.com"], created_at=_dt(now)),
        FakeMessage(direction="outbound", from_address="bot@example.com", to_addresses=["bob@example.com"], created_at=_dt(now + 1)),
    ]
    identity = FakeIdentity(emails=messages, mailbox_address="bot@example.com")
    client = InkboxSDKDirectoryClient(FakeInkbox([]), identity)
    rows = list(client.email_threads())
    handles = {r["handle"] for r in rows}
    assert handles == {"alice@example.com", "bob@example.com"}
    assert "bot@example.com" not in handles


def test_contacts_yields_one_row_per_email_and_phone():
    contacts = [
        FakeContact(
            preferred_name="Dana Lee",
            emails=[FakeContactEmail(value="dana@example.com")],
            phones=[FakeContactPhone(value="+15552223333")],
            updated_at=_dt(1_700_000_000.0),
        ),
    ]
    client = InkboxSDKDirectoryClient(FakeInkbox(contacts), FakeIdentity())
    rows = list(client.contacts())
    handles = {r["handle"] for r in rows}
    assert handles == {"dana@example.com", "+15552223333"}
    assert all(r["name"] == "Dana Lee" for r in rows)


def test_contacts_pages_to_exhaustion():
    contacts = [
        FakeContact(preferred_name=f"Person {i}", emails=[FakeContactEmail(value=f"p{i}@example.com")])
        for i in range(5)
    ]
    client = InkboxSDKDirectoryClient(FakeInkbox(contacts), FakeIdentity())
    rows = list(client.contacts())
    assert len(rows) == 5


def test_one_channel_raising_is_reported_failed_others_still_sync(tmp_path):
    now = 1_700_000_000.0
    identity = FakeIdentity(
        imessage_pages=[FakeIMessageConversationSummary(id="1", remote_number="+15550001111", latest_message_at=_dt(now))],
        sms_pages=[FakeTextConversationSummary(remote_phone_number="+15559990000", latest_message_at=_dt(now))],
        calls_pages=[],
        emails=[],
        raise_on={"sms"},
    )
    client = InkboxSDKDirectoryClient(FakeInkbox([]), identity)
    store = make_store(tmp_path)
    result = sync_people(store, client)
    assert result["failed"] == ["sms"]
    assert "sms" not in result["counts"]
    assert result["people"] == 1  # the imessage row still merged in
    synced = store.synced_people()
    assert any(row.get("key") == "5550001111" for row in synced)
