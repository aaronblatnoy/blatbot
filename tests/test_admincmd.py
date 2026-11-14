"""Aaron's console, from his chat (admincmd.py): the friendly guides, settings changes that
say old value, new value and how to undo, and role creation. Throwaway sqlite databases
only; nothing here touches a live setting, role or person."""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

from inkbox_claude.console import api as console_api
from inkbox_claude.gate import accesscmd, admincmd, schedcmd, wording
from inkbox_claude.gate import settings as gate_settings
from inkbox_claude.gate.store import Store

from test_gate import approver_meta, make_manager, stranger_meta  # noqa: E402


def make_store(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    gate_settings.bind(store)
    gate_settings.invalidate()
    return store


def run(coro):
    return asyncio.run(coro)


def ctx_for(store, manager=None):
    return SimpleNamespace(store=store, chat_id="aaron", here=("aaron", "This chat"), manager=manager,
                           session=None, directory_client=None)


def say(store, text, manager=None):
    """What Aaron's message would be answered with, through the same two matchers the
    gate uses (admin first, then access), or None when neither claims it."""
    cmd = admincmd.match(text, store)
    if cmd is not None:
        return run(admincmd.handle(ctx_for(store, manager), cmd))
    access = accesscmd.match(text, store)
    if access is not None:
        return accesscmd.handle(store, SimpleNamespace(**access), here=("aaron", "This chat"))
    return None


def understood(store, text):
    """True when some matcher in the gate claims this exact phrase."""
    return (admincmd.match(text, store) is not None or accesscmd.match(text, store) is not None
            or schedcmd.match_fixed_phrase(text) is not None)


def seed(store):
    store.set_role("corgi teammate", ["corgi_read"], "")
    store.set_role("general", [], "")
    store.set_trust("telegram:111", person="Stanley Rabkin", role="general")


def plain_ok(text):
    assert "—" not in text and "--" not in text, "no dashes used as punctuation"
    assert not re.search(r"[\U0001F300-\U0001FAFF☀-➿]", text), "no emoji"
    assert "GATE_" not in text, "no internal key names"
    assert len(text) <= 900, "short enough for one iMessage"


# ---------------------------------------------------------------------------
# the guides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase,area", [
    ("settings", "settings"), ("Settings?", "settings"), ("help with settings", "settings"),
    ("what can I change", "settings"), ("roles", "roles"), ("help with roles", "roles"),
    ("schedules", "schedules"), ("group chats", "groups"), ("help with group chats", "groups"),
    ("people", "people"), ("requests", "requests"), ("help", "overview"), ("blatbot, settings please", "settings"),
])
def test_general_and_area_questions_open_the_right_guide(tmp_path, phrase, area):
    store = make_store(tmp_path)
    cmd = admincmd.match(phrase, store)
    assert cmd == {"action": "guide", "area": area}


def test_ordinary_sentences_are_not_taken_for_a_guide(tmp_path):
    store = make_store(tmp_path)
    for text in ("what's on my schedule today", "my settings are weird, can you look into the laptop?",
                 "tell the people in the group I'm late", "help me write an email", "roles in the movie"):
        assert admincmd.match(text, store) is None


def test_settings_guide_shows_current_values_and_exact_phrases(tmp_path):
    store = make_store(tmp_path)
    reply = say(store, "settings")
    assert reply.startswith("Settings are how I behave")
    assert "Bar for speaking in a group unprompted: 0.45." in reply
    assert "Show me anything it would send: on." in reply
    plain_ok(reply)
    examples = reply.split("Type any of these:\n", 1)[1].splitlines()
    assert len(examples) >= 3 and len(examples) <= 6
    for phrase in examples:
        assert understood(store, phrase), f"the guide offers a phrase nothing understands: {phrase!r}"


def test_settings_guide_example_changes_the_value_it_names(tmp_path):
    store = make_store(tmp_path)
    reply = say(store, "settings")
    phrase = next(p for p in reply.splitlines() if p.startswith("set the bar for speaking"))
    changed = say(store, phrase)
    assert changed.startswith("Bar for speaking in a group unprompted: 0.5 (was 0.45).")
    assert gate_settings.get("GATE_GROUP_REPLY_MIN") == 0.5


@pytest.mark.parametrize("area", ["roles", "groups", "people", "schedules", "requests", "overview"])
def test_every_guide_is_plain_and_its_phrases_work(tmp_path, area):
    store = make_store(tmp_path)
    seed(store)
    store.set_response_mode("tg:-9", "natural")
    reply = admincmd.guide_for(area, store)
    plain_ok(reply)
    block = reply.split("Type any of these:\n", 1)
    assert len(block) == 2
    for phrase in block[1].splitlines():
        if phrase.startswith("#") or phrase == "try again":
            continue  # approval handles and retry belong to the request path, not match()
        assert understood(store, phrase), f"{area}: nothing understands {phrase!r}"


def test_roles_guide_lists_roles_with_counts(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    reply = admincmd.guide_for("roles", store)
    assert "corgi teammate: nobody" in reply
    assert "general: 1 person" in reply
    assert "make Stanley Rabkin a corgi teammate" in reply


# ---------------------------------------------------------------------------
# changing one setting: old value, new value, the phrase that undoes it
# ---------------------------------------------------------------------------

def test_a_setting_change_states_old_and_new_and_how_to_undo(tmp_path):
    store = make_store(tmp_path)
    reply = say(store, "set the bar for speaking in a group unprompted to 0.7")
    assert reply == ("Bar for speaking in a group unprompted: 0.7 (was 0.45).\n"
                     "To undo, say: set the bar for speaking in a group unprompted to 0.45")
    assert gate_settings.get("GATE_GROUP_REPLY_MIN") == 0.7
    again = say(store, "set the bar for speaking in a group unprompted to 0.7")
    assert again == "Bar for speaking in a group unprompted: 0.7.", "asking twice changes nothing and adds no undo"
    undone = say(store, "set the bar for speaking in a group unprompted to 0.45")
    assert undone.startswith("Bar for speaking in a group unprompted: 0.45 (was 0.7).")
    assert gate_settings.get("GATE_GROUP_REPLY_MIN") == 0.45


def test_a_switch_change_undoes_with_the_opposite_switch(tmp_path):
    store = make_store(tmp_path)
    reply = say(store, "turn off show me anything it would send")
    assert reply == ("Show me anything it would send: off (was on).\n"
                     "To undo, say: turn on show me anything it would send")
    say(store, "turn on show me anything it would send")
    assert gate_settings.get("GATE_CONFIRM_SENDS") is True


def test_an_invalid_value_says_what_is_allowed_and_what_to_type(tmp_path):
    store = make_store(tmp_path)
    too_high = say(store, "set the bar for speaking in a group unprompted to 7")
    assert too_high == ("Bar for speaking in a group unprompted takes a number from 0 to 1. "
                        "Try: set the bar for speaking in a group unprompted to 0.5")
    words = say(store, "set the bar for speaking in a group unprompted to loud")
    assert words.startswith("Bar for speaking in a group unprompted takes a number from 0 to 1.")
    choice = say(store, "set the model for hard requests to gpt")
    assert choice == ("Model for hard requests takes one of opus, sonnet, or haiku. "
                      "Try: set the model for hard requests to sonnet")
    assert gate_settings.get("GATE_GROUP_REPLY_MIN") == 0.45, "nothing changed"


def test_an_unknown_setting_points_back_to_the_guide(tmp_path):
    store = make_store(tmp_path)
    reply = accesscmd.set_setting(store, "the flux capacitor", "9")
    assert reply == "I don't have a setting called the flux capacitor. Say settings and I will list what I have."


def test_show_all_settings_still_lists_every_setting_one_per_line(tmp_path):
    store = make_store(tmp_path)
    reply = say(store, "show all my settings")
    assert reply.count("\n") >= 8
    assert "Bar for speaking in a group unprompted: 0.45." in reply.splitlines()


# ---------------------------------------------------------------------------
# roles: create, list, show
# ---------------------------------------------------------------------------

def test_create_a_role_applies_once_and_says_how_to_undo(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    reply = say(store, "create a role called interns")
    assert reply == ("interns: asks you before doing anything; nobody holds it (new role).\n"
                     "To undo, say: delete the interns role")
    assert any(r["name"] == "interns" for r in store.roles())
    again = say(store, "create a role called interns")
    assert again == "interns: asks you before doing anything; nobody holds it."
    assert len([r for r in store.roles() if r["name"] == "interns"]) == 1


def test_create_a_role_refuses_bad_names(tmp_path):
    store = make_store(tmp_path)
    assert say(store, "create a role called commander") == wording.COMMANDER_NOT_FROM_CHAT
    long_name = say(store, "create a role called one two three four five")
    assert long_name == wording.role_name_invalid("one two three four five")
    assert "Try: create a role called interns" in long_name


def test_role_show_and_list_use_labels_not_ids(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    assert say(store, "what can the corgi teammate role do") == (
        "corgi teammate: can look up Corgi Command (the pension CRM); nobody holds it.")
    listed = say(store, "show all roles")
    assert "corgi teammate: can look up Corgi Command (the pension CRM); nobody holds it" in listed
    assert "commander: full control; you alone" in listed
    assert "corgi_read" not in listed


def test_role_commands_reach_the_gate_only_for_aaron(tmp_path):
    m, sent = make_manager(tmp_path)
    m.store.set_role("general", [], "")
    asyncio.run(m.get("aaron").handle_inbound("create a role called interns", "imessage", approver_meta()))
    assert [r["name"] for r in m.store.roles()].count("interns") == 1
    assert sent[-1][1].startswith("interns: asks you before doing anything; nobody holds it (new role).")
    before = [r["name"] for r in m.store.roles()]
    asyncio.run(m.get("c1").handle_inbound("create a role called spies", "email", stranger_meta()))
    assert [r["name"] for r in m.store.roles()] == before


def test_pause_and_resume_every_schedule_are_one_message_each(tmp_path):
    m, sent = make_manager(tmp_path)
    gate_settings.bind(m.store)
    gate_settings.invalidate()
    s = m.get("aaron")
    asyncio.run(s.handle_inbound("pause every schedule", "imessage", approver_meta()))
    asyncio.run(s.handle_inbound("pause every schedule", "imessage", approver_meta()))
    asyncio.run(s.handle_inbound("resume every schedule", "imessage", approver_meta()))
    texts = [t for _, t, *_ in sent]
    assert texts == [
        "Every schedule is paused (was running).\nTo undo, say: resume every schedule",
        "Every schedule is paused.",
        "Every schedule is running (was paused).\nTo undo, say: pause every schedule",
    ]
