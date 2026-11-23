"""WhatsApp through Blatbot's gate: inbound routing, rooms, owner recognition, sends."""

import asyncio
import json
import types

import pytest

from inkbox_claude import gateway, whatsapp
from inkbox_claude.config import BridgeConfig
from inkbox_claude.gate.manager import GateSession

IDENTITY = "11111111-1111-1111-1111-111111111111"
CONV = "22222222-2222-2222-2222-222222222222"
GROUP = "33333333-3333-3333-3333-333333333333"
LOCAL = "+15550000229"


class _Session:
    def __init__(self):
        self.inbound = []
        self.room = ""

    async def handle_inbound(self, text, mode, meta):
        self.inbound.append((text, mode, meta))


class _Store:
    def __init__(self, mode="mention"):
        self.mode = mode
        self.messages = []

    def get_response_mode(self, chat_id):
        return self.mode

    def add_message(self, chat_id, direction, text, channel, **kwargs):
        self.messages.append((chat_id, direction, text, channel))


class _Sessions:
    def __init__(self, store=None):
        self.by_id = {}
        self.store = store

    def get(self, chat_id):
        return self.by_id.setdefault(chat_id, _Session())


def _gw(store=None):
    gw = gateway.InkboxGateway(BridgeConfig(require_signature=False, allow_all_users=True))
    gw.sessions = _Sessions(store)
    gw._inkbox = types.SimpleNamespace(contacts=types.SimpleNamespace(lookup=lambda **kw: []))
    sent = []

    def send_whatsapp(**kwargs):
        sent.append(kwargs)
        return types.SimpleNamespace(id="wa-out-1")

    gw._identity = types.SimpleNamespace(id=IDENTITY, whatsapp_enabled=True, send_whatsapp=send_whatsapp)
    gw.sent = sent
    return gw


def _direct(text="hello", sender="+15551112222", mid="m1"):
    return {"event_type": "whatsapp.received", "id": "evt-" + mid, "data": {"message": {
        "id": mid, "direction": "inbound", "agent_identity_id": IDENTITY, "conversation_id": CONV,
        "sender_number": sender, "remote_number": sender, "local_number": LOCAL,
        "participants": [sender], "is_group": False, "content": text,
    }}}


def _group(text, sender="+15551112222", mid="g1", mentioned=False):
    members = [sender, "+15553334444"]
    return {"event_type": "whatsapp.received", "id": "evt-" + mid, "data": {"message": {
        "id": mid, "direction": "inbound", "agent_identity_id": IDENTITY, "conversation_id": GROUP,
        "sender_number": sender, "remote_number": None, "local_number": LOCAL,
        "participants": members, "is_group": True, "group_name": "Weekend plans",
        "content": text, "agent_mentioned": mentioned,
    }}}


def test_direct_message_reaches_its_session_once():
    gw = _gw()
    first = asyncio.run(whatsapp.receive(gw, _direct()))
    second = asyncio.run(whatsapp.receive(gw, _direct()))
    assert json.loads(first.text)["ok"] is True
    assert json.loads(second.text)["deduped"] is True
    session = gw.sessions.by_id[f"whatsapp:{CONV}"]
    assert len(session.inbound) == 1
    body, mode, meta = session.inbound[0]
    assert (body, mode) == ("hello", "whatsapp")
    assert meta["sender"] == "+15551112222"
    assert meta["conversation_id"] == CONV
    assert meta["conversation_kind"] == "direct"
    assert session.room == ""


def test_outbound_and_foreign_identity_messages_are_ignored():
    gw = _gw()
    event = _direct()
    event["data"]["message"]["direction"] = "outbound"
    assert json.loads(asyncio.run(whatsapp.receive(gw, event)).text)["ignored"] == "unsupported-message"
    event = _direct(mid="m2")
    event["data"]["message"]["agent_identity_id"] = "someone-else"
    assert json.loads(asyncio.run(whatsapp.receive(gw, event)).text)["ignored"] == "unsupported-message"
    assert gw.sessions.by_id == {}


def test_group_message_sets_the_room_and_is_framed_as_a_group():
    gw = _gw(_Store(mode="natural"))
    asyncio.run(whatsapp.receive(gw, _group("anyone up for dinner?")))
    session = gw.sessions.by_id[f"whatsapp:{GROUP}"]
    body, mode, meta = session.inbound[0]
    assert mode == "whatsapp"
    assert body.startswith("[inkbox:group_whatsapp ")
    assert meta["conversation_kind"] == "group"
    assert meta["group_title"] == "Weekend plans"
    assert "WhatsApp" in session.room and "Weekend plans" in session.room


def test_group_in_mention_mode_overhears_unaddressed_chatter():
    store = _Store(mode="mention")
    gw = _gw(store)
    response = asyncio.run(whatsapp.receive(gw, _group("see you at 8")))
    assert json.loads(response.text)["mode"] == "blocked"
    assert gw.sessions.by_id == {}
    assert store.messages == [(f"whatsapp:{GROUP}", "inbound", "+15551112222: see you at 8", "whatsapp")]


def test_native_mention_counts_as_addressed_but_silent_still_wins():
    gw = _gw(_Store(mode="mention"))
    asyncio.run(whatsapp.receive(gw, _group("@15550000229 book it", mentioned=True)))
    assert len(gw.sessions.by_id[f"whatsapp:{GROUP}"].inbound) == 1
    gw = _gw(_Store(mode="silent"))
    response = asyncio.run(whatsapp.receive(gw, _group("@15550000229 book it", mid="g2", mentioned=True)))
    assert json.loads(response.text)["mode"] == "blocked"


def test_reply_goes_to_the_conversation_with_a_stable_key():
    gw = _gw()
    meta = {"conversation_id": CONV, "message_id": "m1"}
    asyncio.run(gw.send_to_contact("contact-1", "**Done.**", "whatsapp", meta))
    asyncio.run(gw.send_to_contact("contact-1", "**Done.**", "whatsapp", meta))
    assert gw.sent[0]["conversation_id"] == CONV
    assert gw.sent[0]["text"] == "Done."
    assert gw.sent[0]["idempotency_key"] == gw.sent[1]["idempotency_key"]


def test_too_long_reply_is_refused_before_sending():
    gw = _gw()
    with pytest.raises(ValueError):
        asyncio.run(whatsapp.send(gw, "c", "x" * 4001, {"conversation_id": CONV}))
    assert gw.sent == []


def test_owner_is_recognised_by_his_number_only_on_whatsapp():
    manager = types.SimpleNamespace(approver_phone="+1 (555) 999-0000", approver_conv="",
                                    approver_telegram_id="", voice_trust_approver=False)
    session = GateSession("whatsapp:x", manager)
    session.mode = "whatsapp"
    session.reply_meta = {"sender": "+15559990000", "conversation_kind": "direct"}
    assert session.is_approver()
    session.reply_meta = {"sender": "+15551112222"}
    assert not session.is_approver()
    session.mode = "sms"
    session.reply_meta = {"sender": "+15559990000"}
    assert not session.is_approver()


def test_unsupported_subscription_errors_are_recognised():
    assert whatsapp.unsupported(ValueError("Unknown event type prefix whatsapp"))
    assert not whatsapp.unsupported(ValueError("network down"))


def test_whatsapp_is_a_send_message_surface_never_picked_by_default():
    from inkbox_claude.gate import outcomes
    assert outcomes.surface_of("WhatsApp") == "whatsapp"
    assert outcomes.surface_of("wa") == "whatsapp"
    assert outcomes.SURFACE_ORDER[-1] == "whatsapp"
    person = {"display": "Sam", "contacts": [{"kind": "phone", "raw_value": "+15551112222"}]}
    options = outcomes._person_options(None, person)
    assert options["whatsapp"]["mode"] == "whatsapp"
    assert options["whatsapp"]["meta"]["whatsapp_to"] == "+15551112222"


def test_tool_send_to_a_number_uses_to():
    gw = _gw()
    result = asyncio.run(whatsapp.send(gw, "whatsapp:+15551112222", "hi there",
                                       {"whatsapp_to": "+15551112222", "_inkbox_tool": True}))
    assert result == {"id": "wa-out-1"}
    assert gw.sent[0]["to"] == "+15551112222" and "conversation_id" not in gw.sent[0]


def test_whatsapp_send_scope_is_registered():
    from inkbox_claude.gate import scopes
    assert "mcp__inkbox__inkbox_send_whatsapp" in scopes.SCOPES["whatsapp_send"]["tools"]
    assert scopes.sends_to_requester("mcp__inkbox__inkbox_send_whatsapp",
                                     {"to": "+1 555 111 2222"}, ["+15551112222"])


def test_generated_image_is_uploaded_and_attached_dry_run(tmp_path, monkeypatch):
    """Stubbed transport: a render_image PNG is uploaded, its media id rides on the send."""
    from inkbox_claude.gate import imagetool
    monkeypatch.setattr(imagetool, "GENERATED_DIR", tmp_path)
    png = tmp_path / "sheet-1.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    gw = _gw()
    uploads = []

    def upload_whatsapp_media(**kwargs):
        uploads.append(kwargs)
        return types.SimpleNamespace(id="media-1")

    gw._identity.upload_whatsapp_media = upload_whatsapp_media
    paths = imagetool.allowed_image_paths([str(png)])
    meta = {"whatsapp_to": "+15551112222", "_inkbox_tool": True, "generated_images": paths}
    asyncio.run(gw.send_to_contact("whatsapp:+15551112222", "here are the options", "whatsapp", meta))
    assert uploads[0]["filename"] == "sheet-1.png" and uploads[0]["content_type"] == "image/png"
    assert uploads[0]["recipient_number"] == "+15551112222"
    assert gw.sent[0]["media_ids"] == ["media-1"] and gw.sent[0]["to"] == "+15551112222"
