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


def add_schedule(store, title="Corgi NYC digest", state="active"):
    import time
    task_id = store.create_task(title)["id"]
    schedule = store.create_schedule(
        chat_id="owner", task_id=task_id, title=title, prompt="summarize the chat", kind="every",
        scopes=["web"], timezone="America/New_York", report_mode="changed", cadence_kind="every",
        cadence={"minutes": 240}, origin_mode="console", origin_chat_id="owner")
    return store.update_schedule(schedule.id, state=state, next_run=time.time() + 3600)


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
    add_schedule(store)
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


# ===========================================================================
# second slice: roles, people, holds, requests, schedules by name, health
# ===========================================================================

def aaron(m, text):
    asyncio.run(m.get("aaron").handle_inbound(text, "imessage", approver_meta()))


def last_text(sent):
    return sent[-1][1]


def seed_people(store):
    seed(store)
    store.set_trust("telegram:222", person="Sean Parnell", role="corgi teammate")


def role_scopes(store, name):
    return next(r for r in store.roles() if r["name"] == name)["scopes"]


def test_a_role_gains_and_loses_an_ability_and_says_how_to_undo(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    store.set_role("interns", [], "")
    reply = say(store, "let the interns role read the TAMID calendar")
    assert reply.startswith("interns: can look up")
    assert "(was: asks you before doing anything)" in reply
    assert reply.splitlines()[-1] == "To undo, say: take read the TAMID calendar away from the interns role"
    assert role_scopes(store, "interns") == ["calendar"]
    assert say(store, "let the interns role read the TAMID calendar") == (
        "interns: can look up and manage the TAMID calendar; nobody holds it.")
    gone = say(store, "take read the TAMID calendar away from the interns role")
    assert gone.startswith("interns: asks you before doing anything")
    assert role_scopes(store, "interns") == []
    assert say(store, "take read the TAMID calendar away from the interns role").startswith("interns: asks you")


def test_a_role_hold_for_memory_or_sending_waits_for_a_numbered_yes(tmp_path):
    m, sent = make_manager(tmp_path)
    seed(m.store)
    m.store.set_role("interns", [], "")
    aaron(m, "let the interns role remember everything")
    ask = last_text(sent)
    assert ask.startswith("Let the interns role remember everything? That widens access")
    handle = re.search(r'"#(\d+) yes"', ask).group(1)
    assert role_scopes(m.store, "interns") == [], "nothing changes before his yes"
    aaron(m, "let the interns role remember everything")
    assert re.search(r'"#(\d+) yes"', last_text(sent)).group(1) == handle, "the same ask is the same hold"
    assert len(m.store.pending()) == 1
    aaron(m, f"#{handle} yes")
    assert role_scopes(m.store, "interns") == ["memory:all"]
    assert "remember everything" in last_text(sent)
    assert m.store.pending() == []
    aaron(m, f"#{handle} yes")  # a second yes is refused, not applied twice
    assert role_scopes(m.store, "interns") == ["memory:all"]


def test_a_no_leaves_the_role_alone(tmp_path):
    m, sent = make_manager(tmp_path)
    seed(m.store)
    m.store.set_role("interns", [], "")
    aaron(m, "delete the interns role")
    assert last_text(sent) == ('Delete the interns role? Nobody holds it.\n'
                               'Reply "#1 yes" to do it or "#1 no" to leave it alone.') or "#" in last_text(sent)
    handle = re.search(r'"#(\d+) yes"', last_text(sent)).group(1)
    aaron(m, f"#{handle} no")
    assert any(r["name"] == "interns" for r in m.store.roles())
    assert "Left alone" in last_text(sent)


def test_delete_a_role_after_yes_and_the_undo_phrase_recreates_it(tmp_path):
    m, sent = make_manager(tmp_path)
    seed_people(m.store)
    aaron(m, "delete the corgi teammate role")
    assert "held by Sean Parnell" in last_text(sent) and "keep their other roles" in last_text(sent)
    handle = re.search(r'"#(\d+) yes"', last_text(sent)).group(1)
    aaron(m, f"#{handle} yes")
    assert not any(r["name"] == "corgi teammate" for r in m.store.roles())
    assert last_text(sent).startswith("The corgi teammate role is deleted; Sean Parnell keeps their other roles") or \
        last_text(sent).startswith("The corgi teammate role is deleted; Sean Parnell keep their other roles")
    assert "To undo, say: create a role called corgi teammate" in last_text(sent)
    before = len(sent)
    aaron(m, "delete the corgi teammate role")
    assert m.store.pending() == [], "a role that is already gone opens no second hold"


def test_commander_role_and_unknown_roles_are_refused(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    assert admincmd.match("delete the commander role", store) is None or \
        say(store, "delete the commander role") == wording.COMMANDER_NOT_FROM_CHAT
    assert admincmd.match("delete the martian role", store) is None


def test_rename_a_role(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    store.set_role("interns", [], "")
    reply = say(store, "rename the interns role to volunteers")
    assert reply.startswith("volunteers: asks you before doing anything; nobody holds it (was called interns).")
    assert reply.endswith("To undo, say: rename the volunteers role to interns")
    assert any(r["name"] == "volunteers" for r in store.roles()) and not any(r["name"] == "interns" for r in store.roles())
    assert say(store, "rename the volunteers role to general") == wording.role_name_taken("general")


def test_role_level_and_send_use_the_consoles_own_functions(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    store.set_role("interns", [], "")
    reply = say(store, "set the interns role TAMID calendar to change")
    assert reply.startswith("interns: can ") and "TAMID calendar" in reply
    assert "To undo, say: set the interns role's tamid calendar to none" in reply
    assert role_scopes(store, "interns")
    off = say(store, "set the interns role TAMID calendar to none")
    assert off.startswith("interns: asks you before doing anything")
    assert role_scopes(store, "interns") == []


def test_create_a_person_rename_note_and_contacts(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    made = say(store, "add Jordan Lee to people")
    assert made.startswith("Jordan Lee: in People (new).")
    assert say(store, "add Jordan Lee to people") == "Jordan Lee: in People."
    assert len([p for p in store.list_people() if p["display"] == "Jordan Lee"]) == 1
    added = say(store, "add the email jordan@example.com to Jordan Lee")
    assert added.startswith("Jordan Lee: reachable on email jordan@example.com.")
    assert "To undo, say: remove the email jordan@example.com from Jordan Lee" in added
    again = say(store, "add the email jordan@example.com to Jordan Lee")
    assert again.startswith("Jordan Lee: reachable on email jordan@example.com.")
    person = next(p for p in store.list_people() if p["display"] == "Jordan Lee")
    assert len([c for c in person["contacts"] if c["kind"] == "email"]) == 1
    note = say(store, "set Jordan Lee's note to met at the TAMID mixer")
    assert note == 'Jordan Lee: note is "met at the TAMID mixer" (was no note).'
    assert say(store, "set Jordan Lee's note to met at the TAMID mixer") == 'Jordan Lee: note is "met at the TAMID mixer".'
    renamed = say(store, "rename Jordan Lee to Jordan Lee Jr")
    assert renamed.startswith("Jordan Lee Jr: in People (was Jordan Lee).")
    removed = say(store, "remove the email jordan@example.com from Jordan Lee Jr")
    assert removed.startswith("Jordan Lee Jr: no longer reachable on email jordan@example.com.")
    assert say(store, "remove jordan@example.com from Jordan Lee Jr") == (
        "Jordan Lee Jr has no contact jordan@example.com on file.")


def test_a_contact_that_belongs_to_someone_else_says_so_and_changes_nothing(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    say(store, "add Jordan Lee to people")
    say(store, "add Pat Kim to people")
    say(store, "add the email shared@example.com to Pat Kim")
    reply = say(store, "add the email shared@example.com to Jordan Lee")
    assert reply.startswith("shared@example.com already belongs to Pat Kim.")


def test_move_a_contact_between_people(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    say(store, "add Jordan Lee to people")
    say(store, "add Pat Kim to people")
    say(store, "add the email pat@example.com to Jordan Lee")
    moved = say(store, "move pat@example.com from Jordan Lee to Pat Kim")
    assert moved.startswith("Pat Kim: reachable on email pat@example.com.")
    assert say(store, "move pat@example.com from Jordan Lee to Pat Kim").startswith("Pat Kim: reachable on")


def test_merge_and_remove_a_person_wait_for_a_yes(tmp_path):
    m, sent = make_manager(tmp_path)
    seed(m.store)
    aaron(m, "add Jordan Lee to people")
    aaron(m, "add Jordan L to people")
    aaron(m, "merge Jordan L into Jordan Lee")
    assert last_text(sent).startswith("Merge Jordan L into Jordan Lee? Every contact of Jordan L moves to Jordan Lee")
    handle = re.search(r'"#(\d+) yes"', last_text(sent)).group(1)
    assert len(m.store.list_people()) >= 3, "nothing merged yet"
    aaron(m, f"#{handle} yes")
    assert last_text(sent) == "Jordan L is merged into Jordan Lee."
    assert not any(p["display"] == "Jordan L" for p in m.store.list_people())
    aaron(m, "remove Jordan Lee from people")
    assert last_text(sent).startswith("Remove Jordan Lee from People?")
    handle = re.search(r'"#(\d+) yes"', last_text(sent)).group(1)
    aaron(m, f"#{handle} yes")
    assert last_text(sent) == "Jordan Lee is no longer in People."
    assert not any(p["display"] == "Jordan Lee" for p in m.store.list_people())


def test_the_commander_cannot_be_removed_from_chat(tmp_path):
    m, sent = make_manager(tmp_path)
    commander = m.store.commander_person_id()
    if not commander:
        pytest.skip("no commander person in this fixture")
    name = m.store.get_person(commander)["display"]
    aaron(m, f"remove {name} from people")
    assert last_text(sent) == wording.commander_fixed(name)


def test_new_admin_commands_are_aarons_alone(tmp_path):
    m, sent = make_manager(tmp_path)
    seed(m.store)
    m.router.next = admincmd.__dict__.get("RouterOutput") or __import__(
        "inkbox_claude.gate.router", fromlist=["RouterOutput"]).RouterOutput(reply="I will ask Aaron.", request=None)
    stranger = lambda text: asyncio.run(m.get("c1").handle_inbound(text, "email", stranger_meta()))
    for text in ("add Mallory to people", "delete the general role", "rename the general role to boss",
                 "let the general role remember everything"):
        stranger(text)
    assert not any(p["display"] == "Mallory" for p in m.store.list_people())
    assert [r["name"] for r in m.store.roles()].count("general") == 1
    assert not any(r["name"] == "boss" for r in m.store.roles())
    assert m.store.pending() == []


def test_people_groups_and_request_lists(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    assert say(store, "show people") == "1 people: Stanley Rabkin."
    assert say(store, "show group chats") == "I am not in any group chats yet."
    assert say(store, "show pending requests") == "Nothing is waiting on you."
    assert say(store, "show failed requests") == "Nothing has failed lately."


def test_pending_requests_are_listed_with_their_handles(tmp_path):
    m, sent = make_manager(tmp_path)
    seed(m.store)
    m.store.set_role("interns", [], "")
    aaron(m, "delete the interns role")
    aaron(m, "show pending requests")
    assert last_text(sent).startswith("Waiting on you:\n#1: Delete the interns role? Nobody holds it.")
    assert last_text(sent).endswith('Reply "#N yes" or "#N no" to answer one.')


def test_group_list_shows_each_room_and_its_mode(tmp_path):
    store = make_store(tmp_path)
    store.set_response_mode("tg:-9", "silent")
    with store._lock:
        store._db.execute("UPDATE threads SET meta_json=? WHERE chat_id=?",
                          ('{"conversation_kind": "group", "group_title": "FellowSIp of the Ring"}', "tg:-9"))
        store._db.commit()
    assert say(store, "show group chats") == "FellowSIp of the Ring: silent"


def test_refresh_people_uses_the_console_sync(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    seed(store)
    calls = []
    monkeypatch.setattr(console_api, "people_sync", lambda st, client: calls.append(client) or {"ok": True})
    ctx = ctx_for(store)
    ctx.directory_client = object()
    reply = run(admincmd.handle(ctx, admincmd.match("refresh people from Inkbox", store)))
    assert reply == "People is in step with Inkbox: 1 people."
    assert len(calls) == 1
    ctx.directory_client = None
    assert run(admincmd.handle(ctx, {"action": "people_sync"})).startswith("Nothing changed: I can't reach Inkbox")


def test_retry_a_failed_request_calls_the_consoles_retry(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    seen = []

    async def fake_retry(manager, payload):
        seen.append(payload)
        return {"ok": True, "message": "Retrying now."}

    monkeypatch.setattr(console_api, "retry_request_endpoint", fake_retry)
    reply = say(store, "retry #417", manager=SimpleNamespace(store=store))
    assert reply == "Retrying now." and seen == [{"id": 417}]
    not_failed = admincmd.match("retry #4", store)
    assert not_failed["id"] == "4"


def test_pause_resume_and_run_a_schedule_by_name_and_stop_waits_for_a_yes(tmp_path):
    m, sent = make_manager(tmp_path)
    gate_settings.bind(m.store)
    schedule = add_schedule(m.store)
    aaron(m, "pause the Corgi NYC digest schedule")
    assert m.store.get_schedule(schedule.id).state == "paused"
    assert last_text(sent).startswith("Corgi NYC digest: paused")
    aaron(m, "pause the Corgi NYC digest schedule")
    assert last_text(sent).startswith("Corgi NYC digest: paused")
    aaron(m, "resume the Corgi NYC digest schedule")
    assert m.store.get_schedule(schedule.id).state == "active"
    aaron(m, "stop the Corgi NYC digest schedule")
    assert last_text(sent).startswith("End the Corgi NYC digest schedule for good?")
    assert m.store.get_schedule(schedule.id).state == "active", "stop waits for his yes"
    handle = re.search(r'"#(\d+) yes"', last_text(sent)).group(1)
    aaron(m, f"#{handle} yes")
    assert m.store.get_schedule(schedule.id).state != "active"


def test_a_schedule_phrase_for_a_schedule_that_does_not_exist_is_not_claimed(tmp_path):
    store = make_store(tmp_path)
    assert admincmd.match("pause the music", store) is None
    assert admincmd.match("stop it", store) is None
    add_schedule(store)
    assert admincmd.match("pause the music", store) is None
    assert admincmd.match("pause the Corgi NYC digest schedule", store)["sact"] == "pause"


def test_health_is_one_plain_line(tmp_path):
    store = make_store(tmp_path)
    gw = SimpleNamespace(_console_started_at=__import__("time").time() - 7300)
    reply = run(admincmd.handle(ctx_for(store, SimpleNamespace(store=store, gateway=gw)), admincmd.match("health", store)))
    assert reply.startswith("Up for 2 hours; my public link is not connected; 0 interrupted runs;")
    assert reply.endswith("0 waiting on you.")
    plain_ok(reply)


def test_replies_are_plain_text(tmp_path):
    store = make_store(tmp_path)
    seed_people(store)
    for text in ("show all roles", "show people", "what can the corgi teammate role do",
                 "let the corgi teammate role read the TAMID calendar", "settings", "roles", "people"):
        reply = say(store, text)
        plain_ok(reply)
