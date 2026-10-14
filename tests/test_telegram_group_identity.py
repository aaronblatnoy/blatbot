"""A group message carries who the sender actually is, and the room knows its members."""
import types

import pytest

from inkbox_claude import gateway
from inkbox_claude.gate.store import Store


@pytest.fixture
def gw(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    p = store.create_person(display="Dana Reyes", role="board")
    store.add_contact(p["id"], "telegram", "telegram:5550100001")
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    g.sessions = types.SimpleNamespace(store=store)
    return g


def test_linked_telegram_sender_is_named_as_the_person(gw):
    u = {"chat_id": "-100", "from_id": "5550100001", "name": "DR", "text": "hi"}
    assert gw._telegram_known_name(u)["name"] == "Dana Reyes"


def test_unknown_telegram_sender_keeps_their_display_name(gw):
    u = {"chat_id": "-100", "from_id": "5550100009", "name": "Pat", "text": "hi"}
    assert gw._telegram_known_name(u)["name"] == "Pat"


def test_room_members_come_from_the_framing_not_the_bracket(gw):
    store = gw.sessions.store
    store.add_message("tg:-100", "inbound",
                      "[telegram group: Board | from: Dana Reyes | your reply goes to the whole group]\nhello", "telegram")
    store.add_message("tg:-100", "inbound", "Lee Kim: overheard line", "telegram")
    names = gw._seen_in("tg:-100")
    assert "Dana Reyes" in names and "Lee Kim" in names
    assert not any(n.startswith("[") for n in names)
