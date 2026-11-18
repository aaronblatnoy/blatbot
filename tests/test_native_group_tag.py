from inkbox_claude.gate import ownertext, schedprop, wording

RAW = (
    "[inkbox:group_whatsapp conversation_id=3c5b6491-8adc-4f8c-9757-c883a32b09dc from=+16512319697 "
    "group='Presidents of Tamid & Entourage' participants=+12014218285,+14078088771 "
    "reply_mode=conversation_id | contact=unknown_in_inkbox]\n"
    "Group WhatsApp response policy: you receive every message in this group so you can track context.\n"
    "Reply only when the latest message clearly addresses this Inkbox agent, asks it to act, or a visible answer would be expected from the agent.\n"
    "Treat ordinary group chatter as context only.\n"
    "If no visible reply is warranted, return exactly [SILENT].\n"
    "Why do you think they said that to us?"
)


def test_native_marker_gives_only_the_persons_words():
    group, sender, message = ownertext.split_group_tag(RAW)
    assert group == "Presidents of Tamid & Entourage"
    assert sender == "+16512319697"
    assert message == "Why do you think they said that to us?"


def test_nothing_internal_survives_in_titles():
    for out in (ownertext.strip_group_tag(RAW), schedprop.strip_group_tag(RAW),
                wording.fallback_title_from_message(RAW)):
        assert out == "Why do you think they said that to us?"
        assert "inkbox:" not in out and "conversation_id" not in out and "+1" not in out


def test_imessage_marker_and_plain_text_still_work():
    im = RAW.replace("group_whatsapp", "group_imessage").replace("Group WhatsApp", "Group iMessage").replace(
        "group='Presidents of Tamid & Entourage' ", "")
    assert ownertext.split_group_tag(im) == ("", "+16512319697", "Why do you think they said that to us?")
    assert ownertext.split_group_tag("hello there") == ("", "", "hello there")
    assert ownertext.split_group_tag("[telegram group: Quant | from: Rick | x] hi") == ("Quant", "Rick", "hi")
