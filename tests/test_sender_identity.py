"""Whatever the channel calls a sender, Blatbot names them from the people records."""
from tests.test_gate import make_manager


def _session(m, chat, mode, meta):
    s = m.get(chat)
    s.mode, s.reply_meta = mode, meta
    return s


def _known(m):
    p = m.store.create_person(display="Dana Reyes", role="board")
    for kind, value in (("telegram", "telegram:5550100001"), ("phone", "+15550100002"),
                        ("email", "dana@example.org")):
        m.store.add_contact(p["id"], kind, value)


def test_known_person_is_named_on_every_channel(tmp_path):
    m, _ = make_manager(tmp_path)
    _known(m)
    cases = [
        ("tg:1", "telegram", {"sender": "telegram:5550100001", "telegram_user_id": "5550100001",
                              "contact": {"name": "DR"}}),
        ("sms:2", "sms", {"sender": "+15550100002", "to": "+15550100002"}),
        ("im:3", "imessage", {"sender": "+15550100002", "to": "+15550100002", "contact": {"name": "D"}}),
        ("em:4", "email", {"sender": "dana@example.org", "to": "dana@example.org",
                           "contact": {"name": "dreyes"}}),
        ("call:5", "voice", {"sender": "+15550100002", "to": "+15550100002"}),
    ]
    for chat, mode, meta in cases:
        s = _session(m, chat, mode, meta)
        assert s._sender_name() == "Dana Reyes", mode
        assert s._sender_label() == "Dana Reyes (board)", mode


def test_unknown_sender_is_said_to_be_unknown(tmp_path):
    m, _ = make_manager(tmp_path)
    s = _session(m, "sms:9", "sms", {"sender": "+15550100099", "to": "+15550100099",
                                     "contact": {"name": "Pat"}})
    assert s._sender_name() == "Pat"
    assert s._sender_label() == "Pat, someone who is not in Aaron's people records"
