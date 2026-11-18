"""Console parity by text (2026-11-18): every console page Aaron can also read or change
by texting Blatbot, and every reply in the one clean shape (chatreply.py). Throwaway
sqlite databases only; nothing is sent to anyone."""

from __future__ import annotations

import re
import time

import pytest

from inkbox_claude.gate import chatreply, ownertext, wording

from test_admincmd import add_schedule, make_store, run, say, seed  # noqa: E402

ORNAMENT = re.compile("[←-⇿─-➿⬀-⯿\U0001f000-\U0001faff·•]")
RAW = [
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),  # UUID
    re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{12,}\b", re.I),               # hex id
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),                                      # snake_case
    re.compile(r"[{}\[\]]|\w+=\w+|\"\w+\":"),                                              # JSON / repr
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),                                     # machine time
    re.compile(r"\b1[6-9]\d{8}(?:\.\d+)?\b"),                                              # unix time
    re.compile(r"(?:telegram|tg|imessage|sms):\d"),                                        # provider id
    re.compile(r"\*\*|__|`|^#+ ", re.M),                                                   # markdown
    re.compile("—"),                                                                  # em dash
]


def clean_reply(text):
    assert text, "a reply"
    out = chatreply.finish(text)
    assert not ORNAMENT.search(out), out
    for pat in RAW:
        assert not pat.search(out), (pat.pattern, out)
    assert len([ln for ln in out.split("\n") if ln.strip()]) <= 14, out
    return out


def world(tmp_path):
    store = make_store(tmp_path)
    seed(store)
    store.set_trust("telegram:222", person="Sean Parnell", role="corgi teammate")
    add_schedule(store, "Corgi NYC digest")
    return store


EVERY_PHRASE = [
    "help", "what can I do from here", "overview", "what's going on", "inbox", "what's in the inbox",
    "follow-ups", "what are you waiting on", "memory", "what do you remember", "who can see my notes",
    "pages", "tasks", "show schedules", "when did the corgi digest last run", "show tools",
    "show people", "show group chats", "show pending requests", "show failed requests", "roles",
    "show the corgi teammate role", "show Sean", "what can Sean Parnell do", "settings", "show all my settings",
    "outbox", "people", "schedules", "requests",
]


@pytest.mark.parametrize("phrase", EVERY_PHRASE)
def test_every_console_read_is_one_clean_reply(tmp_path, phrase):
    store = world(tmp_path)
    reply = say(store, phrase)
    assert reply is not None, f"not understood: {phrase}"
    clean_reply(reply)


def test_what_can_i_do_lists_areas_with_one_example_each(tmp_path):
    out = clean_reply(say(world(tmp_path), "what can I do from here"))
    lines = out.split("\n")
    assert lines[0].startswith("Everything the console shows")
    assert any(ln.startswith("People: ") for ln in lines)
    assert any(ln.startswith("Schedules: ") for ln in lines)


def test_show_person_names_role_and_contacts(tmp_path):
    out = clean_reply(say(world(tmp_path), "show Sean"))
    assert out.startswith("Sean Parnell")


def test_ambiguous_person_asks_which_by_name(tmp_path):
    store = world(tmp_path)
    store.set_trust("telegram:333", person="Sean Lee", role="general")
    out = clean_reply(say(store, "show Sean"))
    assert "Sean Lee" in out and "Sean Parnell" in out


def test_schedule_runs_by_title(tmp_path):
    out = clean_reply(say(world(tmp_path), "when did the corgi digest last run"))
    assert out == "Corgi NYC digest has not run yet."


def test_role_memory_widening_asks_once_then_idempotent(tmp_path):
    store = world(tmp_path)
    first = clean_reply(say(store, "set the corgi teammate role's memory to everything"))
    again = clean_reply(say(store, "set the corgi teammate role's memory to everything"))
    assert first == again and re.search(r"#\d+ yes / no$", first)
    own = clean_reply(say(store, "set the corgi teammate role's memory to own"))
    assert own == "The corgi teammate role remembers only their own conversations."


def test_role_notes_states_result_and_reads_the_same_twice(tmp_path):
    store = world(tmp_path)
    first = clean_reply(say(store, "set the corgi teammate role's notes to all"))
    again = clean_reply(say(store, "set the corgi teammate role's notes to all"))
    assert first.split("\n")[0] == again == "The corgi teammate role can read every note folder."


def test_long_list_is_capped_with_and_more(tmp_path):
    store = world(tmp_path)
    for i in range(20):
        store.set_trust(f"telegram:9{i:03d}", person=f"Person Number{chr(65 + i)}", role="general")
    out = clean_reply(say(store, "show people"))
    assert re.search(r"and \d+ more on the console's People page\.$", out)
    assert len(out.split("\n")) <= 10


def test_house_format_has_no_marks():
    text = ownertext.compose("failed", "Could not finish for Aaron Blatnoy (#513)", facts=[("Asked", "x")],
                             body="It stopped.", handle=ownertext.reply_handle(513))
    assert text.startswith("Could not finish for Aaron Blatnoy (#513)")
    assert not ORNAMENT.search(text)
    assert not ORNAMENT.search(wording.hold_line("Delete the interns role?", 7))


def test_finish_strips_ornaments_and_machine_bits():
    out = chatreply.finish("▲ Done · **ok** — at 2026-11-18T09:00:00 for corgi_read")
    assert not ORNAMENT.search(out) and "**" not in out and "—" not in out
    assert "2026-11-18T" not in out and "corgi_read" not in out


def test_when_and_span_are_words():
    now = time.time()
    assert chatreply.when(now, now=now).startswith("today ")
    assert chatreply.when(now - 86400, now=now).startswith("yesterday ")
    assert chatreply.span(3 * 3600) == "3 hours"


def test_group_chat_refuses_admin_in_one_line(tmp_path):
    """In a group, Aaron's admin phrasing gets one short line and changes nothing."""
    import asyncio
    from test_gate import approver_meta, make_manager
    m, sent = make_manager(tmp_path)
    seed(m.store)
    meta = approver_meta()
    meta.update({"conversation_kind": "group", "group_title": "Board"})
    asyncio.run(m.get("aaron-group").handle_inbound("create a role called interns", "imessage", meta))
    assert sent[-1][1] == chatreply.GROUP_REFUSAL
    assert not any(r["name"] == "interns" for r in m.store.roles())


def test_direct_chat_admin_reply_goes_out_clean(tmp_path):
    from test_admincmd import aaron, last_text
    from test_gate import make_manager
    m, sent = make_manager(tmp_path)
    seed(m.store)
    aaron(m, "what can I do from here")
    clean_reply(last_text(sent))
    aaron(m, "inbox")
    assert last_text(sent) in ("The inbox is empty.",) or last_text(sent).startswith("Inbox: ")
