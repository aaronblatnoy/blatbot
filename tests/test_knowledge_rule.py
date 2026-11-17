"""One rule for Aaron's notes and Blatbot's memory (gate/knowledge.py): a group chat gets
its own conversation only, whoever speaks there, Aaron included; a blanket yes never
opens a note folder; a run can ask for one folder by name; the console shows who
reaches what per role and per person."""

import asyncio
from types import SimpleNamespace

from inkbox_claude.gate import graphmem, knowledge
from inkbox_claude.gate.router import RouterOutput, RouterRequest

from tests.test_graphmem import _people_store
from tests.test_vault_knowledge import _manager_with_vault


GROUP_META = {"conversation_kind": "group", "group_title": "Club", "telegram_user_id": "1",
              "sender": "telegram:1", "telegram_chat_id": "-77"}


def _aaron_in_group(m):
    m.approver_telegram_id = "1"
    m.store.set_thread("tg:-77", "idle", "telegram", GROUP_META)
    s = m.get("tg:-77")
    s.mode, s.reply_meta = "telegram", dict(GROUP_META)
    return s


# ---------------------------------------------------------------- memory recall


def test_owner_in_a_group_recalls_only_that_group(tmp_path):
    store, _, _ = _people_store(tmp_path)
    got = graphmem.readable_groups(store, "tg:-77", is_approver=True, is_group=True, handles=["telegram:1"])
    assert got == ["blatbot_tg_-77"]
    # His own chats still read everything.
    assert graphmem.readable_groups(store, "owner", is_approver=True, is_group=False, handles=[]) is None


def test_memory_search_in_a_group_run_reads_only_that_group_even_from_the_owner(tmp_path):
    store, _, _ = _people_store(tmp_path)
    req = SimpleNamespace(chat_id="tg:-77", sender="telegram:1")
    assert graphmem.groups_for_request(store, req, True) == ["blatbot_tg_-77"]
    owner_req = SimpleNamespace(chat_id="owner", sender="telegram:1")
    assert graphmem.groups_for_request(store, owner_req, True) is None


# ---------------------------------------------------------------- the session in a group


def test_session_in_a_group_reads_no_outside_tasks(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    s = _aaron_in_group(m)
    assert s.is_approver() and s.in_group()
    assert not s.reads_all_tasks() and s.ledger_person() == ""
    assert s.memory_groups() == ["blatbot_tg_-77"]
    private = m.store.create_task(title="Aaron's private thing")
    m.store.task_event(private["id"], "inbound", "private detail", chat_id="owner")
    own = m.store.create_task(title="Club meeting")
    m.store.task_event(own["id"], "inbound", "when do we meet", chat_id="tg:-77")
    ids = [t["id"] for t in s.prepare_task_index([], "anything")["entries"]]
    assert own["id"] in ids and private["id"] not in ids
    # The same owner in his own chat still sees both.
    dm = m.get("conv-aaron")
    dm.mode, dm.reply_meta = "imessage", {"conversation_id": "conv-aaron"}
    assert dm.reads_all_tasks()
    ids = [t["id"] for t in dm.prepare_task_index([], "anything")["entries"]]
    assert {own["id"], private["id"]} <= set(ids)


def test_group_request_keeps_no_vault_scope_whoever_asks(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    s = _aaron_in_group(m)
    out = RouterOutput(request=RouterRequest(prompt="p", summary="s",
                                             scopes=["web", "vault_read", "vault:areas/garden-shed", "memory_read"]))
    out.request.knowledge_notes = ["Areas/Garden Shed.md"]
    s.apply_knowledge_rule(out)
    assert sorted(out.request.scopes) == ["memory_read", "web"]
    assert out.request.knowledge_notes == []


def test_group_turn_attaches_no_notes(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    s = _aaron_in_group(m)
    called = []

    class Picker:
        async def judge_vault_tree(self, **kw):
            called.append(kw)
            return {"scopes": ["vault:areas/garden-shed"], "notes": ["Areas/Garden Shed.md"]}

    m.task_picker = Picker()
    out = RouterOutput(request=RouterRequest(prompt="p", summary="s", scopes=["web"]))
    asyncio.run(s.jev_knowledge(out))
    assert out.request.scopes == ["web"] and not called


def test_direct_chat_request_keeps_its_vault_scopes(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    dm = m.get("conv-aaron")
    dm.mode, dm.reply_meta = "imessage", {"conversation_id": "conv-aaron"}
    out = RouterOutput(request=RouterRequest(prompt="p", summary="s", scopes=["vault_read"]))
    dm.apply_knowledge_rule(out)
    assert out.request.scopes == ["vault_read"]


# ---------------------------------------------------------------- grants and asks


def test_a_blanket_yes_never_opens_a_note_folder(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    req = SimpleNamespace(scopes=["web"])
    added = m._blanket_grant_scopes(req, {})
    assert added and not [s for s in added if knowledge.is_vault_scope(s)]
    # ...unless the run was refused that exact folder.
    refused = {"denied": [{"scope": "vault:areas/garden-shed"}]}
    from inkbox_claude.gate import wording
    if wording.denied_scopes(refused):
        assert "vault:areas/garden-shed" in m._blanket_grant_scopes(req, refused)


def test_a_run_can_ask_for_one_folder_by_name_but_never_for_memory(tmp_path, monkeypatch):
    _m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    from inkbox_claude.gate import hosttools
    choices = hosttools.access_choices(SimpleNamespace(scopes=["web"], status={}))
    assert choices["vault:areas/garden-shed"] == "read your notes in Areas / Garden Shed"
    assert choices["vault_read"] == "read all of your notes"
    assert not [n for n in choices if n.startswith("memory:")]
    # A run that already holds the whole vault is not offered its folders again.
    whole = hosttools.access_choices(SimpleNamespace(scopes=["vault_read"], status={}))
    assert not [n for n in whole if n.startswith("vault:")]


def test_a_role_folder_is_not_auto_granted_in_a_group(tmp_path, monkeypatch):
    m, sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    m.store.set_role("gardener", ["vault:areas/garden-shed"])
    p = m.store.create_person(display="Gina", role="gardener")
    m.store.add_contact(p["id"], "telegram", "222")
    m.store.set_thread("tg:-88", "idle", "telegram", {**GROUP_META, "sender": "telegram:222"})
    task = m.store.create_task(title="shed")
    req = m.store.create_request(chat_id="tg:-88", sender="telegram:222", sender_name="Gina", mode="telegram",
                                 subject="", original_message="m", summary="shed", scopes=["web"],
                                 prompt="p", state="running", task_id=task["id"])
    status = {"access_request": {"scopes": ["vault:areas/garden-shed"], "why": "needs the shed notes"}}
    out = asyncio.run(m.handle_access_ask(req, status))
    assert out is None                                    # parked for Aaron's yes
    assert "vault:areas/garden-shed" not in (m.store.get_request(req.id).scopes or [])
    assert any("Garden Shed" in str(x[1]) for x in sent)  # the ask names the exact folder


# ---------------------------------------------------------------- console


def test_console_shows_notes_and_memory_per_role_and_person(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    from inkbox_claude.console import api
    m.store.set_role("gardener", ["vault:areas/garden-shed"])
    p = m.store.create_person(display="Gina", role="gardener")
    m.store.add_contact(p["id"], "phone", "+12015550199")
    api.roles_set_memory(m.store, {"role": "gardener", "level": "groups"})
    role = next(r for r in api.roles_summary(m.store)["roles"] if r["name"] == "gardener")
    assert role["memory"]["level"] == "groups"
    assert role["notes"]["level"] == "some"
    assert "memory:groups" in next(r for r in m.store.roles() if r["name"] == "gardener")["scopes"]
    view = api.knowledge_access(m.store)
    gina = next(x for x in view["people"] if x["name"] == "Gina")
    assert gina["notes"]["folders"] == ["read your notes in Areas / Garden Shed"]
    assert gina["memory"]["level"] == "groups" and gina["via"] == ["Gardener"]
    api.roles_set_memory(m.store, {"role": "gardener", "level": "own"})
    assert not [s for s in next(r for r in m.store.roles() if r["name"] == "gardener")["scopes"]
                if s.startswith("memory:")]
    parts = {p["part"]: p for p in api.memory_access(m.store)}
    assert "Never in a group chat" in parts["User notes"]["readers"]


def test_console_routes_register(tmp_path, monkeypatch):
    """The whole console registers on a real aiohttp app (a broken route once took it down)."""
    from aiohttp import web
    from inkbox_claude.console import routes
    app = web.Application()
    routes.register(app, SimpleNamespace())
    paths = {r.resource.canonical for r in app.router.routes() if r.resource is not None}
    assert {"/console/api/knowledge", "/console/api/roles/set_memory", "/console/api/roles/set_notes"} <= paths
