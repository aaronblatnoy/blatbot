"""2026-11-18 live defects: the held-message prompt shows names, not ids; titles,
summaries and memory carry only the person's words, never the group marker; the
email-only signature is stripped before a non-email message is stored or sent."""

from __future__ import annotations

import asyncio

import pytest

from inkbox_claude.gate import graphmem, ownertext, wording
from inkbox_claude.gate.manager import HANDOFF_LINE

from test_gate import make_manager

GROUP = "whatsapp:3c5b6491-8adc-4f8c-9757-c883a32b09dc"
GROUP_META = {"conversation_kind": "group", "group_title": "Presidents of Tamid & Entourage",
              "conversation_id": "3c5b6491-8adc-4f8c-9757-c883a32b09dc"}
MARKED = (
    "[inkbox:group_whatsapp conversation_id=3c5b6491-8adc-4f8c-9757-c883a32b09dc from=+14075550101 "
    "group='Presidents of Tamid & Entourage' participants=+12015550102,+14075550101 "
    "reply_mode=conversation_id | contact_id=8a931c27-94bb-485e-8e8a-6b4fa08f32f6 contact_phones=['+14075550101']]\n"
    "Group WhatsApp response policy: you receive every message in this group so you can track context.\n"
    "Reply only when the latest message clearly addresses this Inkbox agent, asks it to act, or a visible "
    "answer would be expected from the agent.\n"
    "Treat ordinary group chatter as context only.\n"
    "If no visible reply is warranted, return exactly [SILENT].\n"
    "blatbot change my role from Director of Quant to Director of Attendance Management"
)
WORDS = "blatbot change my role from Director of Quant to Director of Attendance Management"


@pytest.fixture(autouse=True)
def hold_on(monkeypatch):
    monkeypatch.setenv("GATE_HOLD_SENDS_TO_OTHERS", "1")


# -- defect 4: the held prompt -------------------------------------------------------

def test_held_prompt_names_the_recipient_and_shows_no_ids(tmp_path):
    m, sent = make_manager(tmp_path)
    t = m.store.create_task("x")
    origin = m.store.create_request(chat_id="c1", sender="a@b.com", sender_name="A", mode="email", subject="",
                                    original_message="m", summary="s", scopes=[], prompt="p", state="running",
                                    task_id=t["id"])
    sent.clear()
    held = asyncio.run(m.hold_message(origin, GROUP, "Jacob is on the Fund track.", "whatsapp",
                                      dict(GROUP_META), chat_id=GROUP))
    prompt = [t for _, t, mode, _ in sent if mode == "imessage"][-1]
    assert prompt == (
        "A message waiting for your yes\n\n"
        "To: Presidents of Tamid & Entourage\nOn: WhatsApp\n\n"
        "Nothing has been sent. This is exactly what would go out:\n\nJacob is on the Fund track.\n\n"
        f"#{held.id} yes / no")
    assert m.store.get_request(held.id).summary == "Send to Presidents of Tamid & Entourage"


def test_recipient_without_a_name_never_shows_a_uuid(tmp_path):
    m, _ = make_manager(tmp_path)
    assert m._recipient_label(GROUP, {"conversation_id": "3c5b6491-8adc-4f8c-9757-c883a32b09dc"}) == "someone"
    assert m._recipient_label("whatsapp:+14075550199", {"to": "+14075550199"}) == "+14075550199"
    assert m._recipient_label("x", {"contact": {"name": "Sean Parnell"}}) == "Sean Parnell"


# -- defect 5: titles and summaries ----------------------------------------------------

def test_marker_with_bracketed_lists_is_stripped_whole():
    title, _, words = ownertext.split_group_tag(MARKED)
    assert words == WORDS and title == "Presidents of Tamid & Entourage"
    assert wording.fallback_title_from_message(MARKED) == WORDS


def test_a_task_title_from_a_marked_message_is_only_the_words(tmp_path):
    from inkbox_claude.gate.router import RouterOutput, RouterRequest
    m, _ = make_manager(tmp_path)
    s = m.get(GROUP)
    out = RouterOutput(reply=None, task="new", request=RouterRequest(prompt="p", scopes=["web"], summary=MARKED))
    task = m.resolve_task(s, out, message=MARKED)
    assert task["title"] == WORDS


# -- defect 6: memory gets only the words ---------------------------------------------

def test_memory_episode_carries_only_the_words_and_the_group_name():
    body = graphmem.episode(GROUP, "inbound", MARKED, mode="whatsapp", message_id=7, speaker="Aaron")
    msg = body["messages"][0]
    assert msg["content"] == WORDS
    assert "4075550101" not in str(body) and "policy" not in msg["content"]
    assert msg["role"] == "Aaron"
    assert "Presidents of Tamid & Entourage" in msg["source_description"]


# -- defect 7: the email-only signature ------------------------------------------------

def test_signature_is_stripped_before_a_whatsapp_message_is_stored_and_sent(tmp_path):
    m, sent = make_manager(tmp_path)
    sent.clear()
    asyncio.run(m.send("whatsapp:+14075550199", HANDOFF_LINE, "whatsapp", {"to": "+14075550199"},
                       chat_id="c-wa", role="answer"))
    expected = "Thanks! I'll confirm that with Aaron and get back to you shortly."
    assert [t for chat, t, *_ in sent if chat == "whatsapp:+14075550199"] == [expected]
    stored = [r["text"] for r in m.store.history("c-wa", limit=5) if r["kind"] == "outbound"]
    assert stored == [expected]


def test_email_keeps_the_signature(tmp_path):
    m, sent = make_manager(tmp_path)
    sent.clear()
    asyncio.run(m.send("cand@nyu.edu", HANDOFF_LINE, "email", {"to": "cand@nyu.edu"}, chat_id="c-mail"))
    assert [t for chat, t, *_ in sent if chat == "cand@nyu.edu"] == [HANDOFF_LINE]
