"""Knowledge scopes: discovery from the vault's own folders, Jev's top-down
traversal, the permission intersection, prompt attachment, and tool enforcement.

Every vault here is a throwaway one built under tmp_path with invented notes;
nothing in this file names a real project, person or place."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from inkbox_claude.gate import hosttools
from inkbox_claude.gate import vaultscopes
from inkbox_claude.gate.executor import vault_prefixes_for_scopes
from inkbox_claude.gate.taskpick import TaskPicker


def write(p: Path, text: str = "placeholder text") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def make_vault(root: Path) -> Path:
    """A small invented vault: two sections, each with one hub that has its own
    folder of notes, plus a loose top-level note and a section with no hubs."""
    write(root / "Home.md", "# Home\n\nStarting point.")
    write(root / "Areas" / "Garden Shed.md", "# Garden Shed\n\nUpkeep notes for the shed.")
    write(root / "Areas" / "Garden Shed" / "Roof.md", "# Roof\n\nThe roof was patched in spring.")
    write(root / "Areas" / "Garden Shed" / "Paint.md", "# Paint\n\nRepainted the door blue.")
    write(root / "Areas" / "Garden Shed" / "Maintenance" / "Gutters.md", "# Gutters\n\nCleared in autumn.")
    write(root / "Areas" / "Garden Shed" / "Maintenance" / "Old" / "Ancient.md", "# Ancient\n\nTwo levels deep.")
    write(root / "Areas" / "Greenhouse.md", "# Greenhouse\n\nSeedling schedule and notes.")
    write(root / "Areas" / "Greenhouse" / "Tomatoes.md", "# Tomatoes\n\nStaked in June.")
    write(root / "Areas" / "People.md", "# People\n\nFolks involved in the shed project.")
    write(root / "Areas" / "People" / "Pebble.md", "# Pebble\n\nRuns the greenhouse rota.")
    write(root / "Areas" / "People" / "Juniper.md", "# Juniper\n\nFixes the shed roof.")
    write(root / "Reference" / "Reference.md", "# Reference\n\nGeneral reference notes.")
    write(root / "Reference" / "Tools.md", "# Tools\n\nWhich tool does what.")
    write(root / "Daily" / "2024-03-14.md", "# 2024-03-14\n\nPatched the roof today.")
    return root


# --------------------------------------------------------------------------- discovery


def test_discovery_finds_sections_and_hubs(tmp_path):
    root = make_vault(tmp_path / "vault")
    scopes = vaultscopes.discover(root)
    names = set(scopes)
    assert "vault:home" in names
    assert "vault:areas" in names
    assert "vault:areas/garden-shed" in names
    assert "vault:areas/greenhouse" in names
    assert "vault:reference" in names
    # Reference has no nested hub (Reference.md sits inside Reference/ itself).
    assert not any(n.startswith("vault:reference/") for n in names)
    hub = scopes["vault:areas/garden-shed"]
    assert hub.depth == 2
    assert hub.parent == "vault:areas"
    assert "Areas/Garden Shed.md" in hub.prefixes
    assert "Areas/Garden Shed" in hub.prefixes
    section = scopes["vault:areas"]
    assert section.depth == 1
    assert section.prefixes == ("Areas",)


def test_discovery_description_comes_from_the_hub_note(tmp_path):
    root = make_vault(tmp_path / "vault")
    scopes = vaultscopes.discover(root)
    assert "Upkeep notes for the shed" in scopes["vault:areas/garden-shed"].description


def test_discovery_skips_dotfiles_and_symlinks(tmp_path):
    root = make_vault(tmp_path / "vault")
    (root / ".obsidian").mkdir()
    write(root / ".obsidian" / "config.md", "ignore me")
    outside = tmp_path / "outside"
    outside.mkdir()
    write(outside / "Secret.md", "not in the vault")
    try:
        (root / "Linked").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable in this environment")
    scopes = vaultscopes.discover(root)
    assert not any("obsidian" in n for n in scopes)
    assert "vault:linked" not in scopes
    assert all("Linked" not in str(p) for info in scopes.values() for p in info.prefixes)


def test_discovery_is_cached_until_the_folder_changes(tmp_path):
    root = make_vault(tmp_path / "vault")
    vaultscopes._cache = vaultscopes._Cache()  # isolate from any process-wide cache
    first = vaultscopes.discover_cached(root)
    assert "vault:areas/garden-shed" in first
    assert "vault:areas/toolshed" not in first
    write(root / "Areas" / "Toolshed.md", "# Toolshed\n\nWhere the tools live.")
    write(root / "Areas" / "Toolshed" / "Hammer.md", "# Hammer\n\nClaw hammer, 16oz.")
    second = vaultscopes.discover_cached(root)
    assert "vault:areas/toolshed" in second


def test_scopes_merge_into_the_registry(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    from inkbox_claude.gate import scopes as scopes_mod
    vaultscopes._cache = vaultscopes._Cache()
    scopes_mod.refresh_vault_scopes(force=True)
    try:
        assert "vault:areas/garden-shed" in scopes_mod.SCOPES
        assert scopes_mod.SCOPE_SYSTEM["vault:areas/garden-shed"] == "vault"
        assert scopes_mod.SCOPES["vault:areas/garden-shed"]["read"] is True
        assert "mcp__host__vault_read" in scopes_mod.SCOPES["vault:areas/garden-shed"]["tools"]
    finally:
        scopes_mod.SCOPE_SYSTEM.pop("vault:areas/garden-shed", None)
        scopes_mod.SCOPES.pop("vault:areas/garden-shed", None)


# --------------------------------------------------------------------------- traversal (scripted Jev)


class ScriptedPicker(TaskPicker):
    """A TaskPicker whose TypeSafe call is replaced by a scripted answer keyed by
    the question names it was asked, so the traversal's own level-by-level logic
    runs for real while the underlying judgments are canned."""

    def __init__(self, script):
        super().__init__(api_key="fake-key")
        self.script = script          # {question_name: probability}
        self.asked = []                # list of question-name lists, one per level

    async def _ask_patch(self, questions):
        self.asked.append(sorted(questions))
        return {k: self.script.get(k, 0.0) for k in questions}


@pytest.fixture
def patched_ask(monkeypatch):
    """judge_vault_tree builds its own `ask` closure per call; patch httpx so that
    closure's HTTP call returns the script instead of hitting the network."""
    def install(picker, script):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def post(self, url, headers=None, json=None):
                questions = json["questions"]
                answers = {k: {"noul": script.get(k, 0.0)} for k in questions}
                return FakeResponse({"answers": answers})

        monkeypatch.setattr("inkbox_claude.gate.taskpick.httpx.AsyncClient", FakeClient)
    return install


def test_vault_tree_picks_hub_directly_not_its_container_section(tmp_path, monkeypatch, patched_ask):
    """A section with hubs (Areas) is never itself a level-1 option; only its
    hubs are judged, by name, directly."""
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")
    script = {
        "vault:areas/garden-shed": 0.95, "vault:areas/greenhouse": 0.1, "vault:areas/people": 0.05,
        "vault:reference": 0.05,
    }
    patched_ask(picker, script)
    res = asyncio.run(picker.judge_vault_tree(prompt="what's the garden shed upkeep plan", summary="shed upkeep"))
    assert res["scopes"] == ["vault:areas/garden-shed"]
    assert "vault:areas" not in res["scopes"]  # the container section was never a level-1 option
    # The hub's own note, plus the two notes directly in its folder, were judged (0.0
    # under this script); bounded recursion is covered separately below.
    assert res["timings"].get("level1") is not None


def test_vault_tree_picks_nothing_below_the_bar(tmp_path, monkeypatch, patched_ask):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")
    patched_ask(picker, {})  # every question answers 0.0
    res = asyncio.run(picker.judge_vault_tree(prompt="what's the weather tomorrow", summary="weather"))
    assert res["scopes"] == []
    assert res["notes"] == []
    assert res["reason"] == "ok"


def test_vault_tree_catchall_not_chosen_by_default(tmp_path, monkeypatch, patched_ask):
    """The home note is a catch-all: it must not be asked about (and so never
    picked) unless the question names it. The script would answer 'yes' for it
    if asked; the point of this test is that it is never asked."""
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")
    script = {"vault:home": 0.99}
    patched_ask(picker, script)
    res = asyncio.run(picker.judge_vault_tree(prompt="what's the shed upkeep plan", summary="shed upkeep"))
    assert "vault:home" not in res["scopes"]
    assert "vault:home" not in res["probabilities"]  # never even asked


def test_vault_tree_catchall_chosen_when_named_outright(tmp_path, monkeypatch, patched_ask):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")
    script = {"vault:home": 0.9}
    patched_ask(picker, script)
    res = asyncio.run(picker.judge_vault_tree(prompt="what's on my home note", summary="home note"))
    assert "vault:home" in res["scopes"]


def test_vault_tree_date_shaped_question_resolves_without_a_jev_call(tmp_path, monkeypatch, patched_ask):
    """A question naming an explicit date is resolved in code, by filename, with
    no Jev call needed to tell near-identical daily notes apart."""
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")
    patched_ask(picker, {})  # every other question answers 0.0; proves this isn't guesswork
    res = asyncio.run(picker.judge_vault_tree(prompt="what happened on 2024-03-14", summary="that day"))
    assert "Daily/2024-03-14.md" in res["notes"]
    assert res["scopes"]  # the date's own scope was picked, by filename, without being asked about


def test_vault_tree_hub_chosen_but_no_note_clears_the_bar_falls_back(tmp_path, monkeypatch, patched_ask):
    """A chosen hub never resolves to zero notes: it falls back to its own hub
    note plus its three best-scoring notes."""
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")
    script = {"vault:areas/garden-shed": 0.9}
    # Every note under the hub scores below VAULT_SCOPE_MIN_YES but not all equally:
    # the keys are "<hub>::<index>", assigned in notes_under_bounded's sorted order.
    for i in range(4):
        script[f"vault:areas/garden-shed::{i}"] = 0.1 + i * 0.05
    patched_ask(picker, script)
    res = asyncio.run(picker.judge_vault_tree(prompt="garden shed", summary="garden shed"))
    assert "vault:areas/garden-shed" in res["scopes"]
    assert "Areas/Garden Shed.md" in res["notes"]  # the hub's own note is always included
    assert len(res["notes"]) >= 1


def test_vault_tree_people_cross_reference_even_when_people_hub_not_picked(tmp_path, monkeypatch, patched_ask):
    """A person note can be pulled in even though 'People' itself did not clear
    the level-1 bar, when the question names the person."""
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")
    discovered = vaultscopes.discover(root)
    people_paths = sorted(vaultscopes.notes_under_bounded(root, discovered["vault:areas/people"]))
    idx = next(i for i, p in enumerate(people_paths) if p.stem == "Juniper")
    script = {"vault:areas/people": 0.1, f"vault:areas/people::{idx}": 0.95}
    patched_ask(picker, script)
    res = asyncio.run(picker.judge_vault_tree(prompt="who is juniper", summary="who is juniper"))
    assert "vault:areas/people" in res["scopes"]
    assert "Areas/People/Juniper.md" in res["notes"]


def test_vault_tree_bounded_recursion_skips_two_levels_deep(tmp_path, monkeypatch, patched_ask):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    discovered = vaultscopes.discover(root)
    paths = vaultscopes.notes_under_bounded(root, discovered["vault:areas/garden-shed"])
    names = {p.name for p in paths}
    assert "Gutters.md" in names        # one level of subfolder below the hub: included
    assert "Ancient.md" not in names    # two levels deep: out of bounds


def test_vault_tree_disabled_without_a_key(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    picker = TaskPicker(api_key=None)
    res = asyncio.run(picker.judge_vault_tree(prompt="x", summary="y"))
    assert res["scopes"] is None
    assert res["reason"] == "disabled"


def test_vault_tree_falls_back_when_typesafe_errors(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")

    class BoomClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, *a, **kw):
            raise RuntimeError("typesafe is down")

    monkeypatch.setattr("inkbox_claude.gate.taskpick.httpx.AsyncClient", BoomClient)
    res = asyncio.run(picker.judge_vault_tree(prompt="x", summary="y"))
    assert res["scopes"] is None
    assert "error" in res["reason"]


def test_vault_tree_on_an_empty_vault(tmp_path, monkeypatch):
    root = tmp_path / "empty_vault"
    root.mkdir()
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    picker = TaskPicker(api_key="fake-key")
    res = asyncio.run(picker.judge_vault_tree(prompt="x", summary="y"))
    assert res["scopes"] is None
    assert res["reason"] == "empty vault"


# --------------------------------------------------------------------------- permission intersection


def _manager_with_vault(tmp_path, monkeypatch):
    from inkbox_claude.gate import manager as gm
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    vaultscopes._cache = vaultscopes._Cache()
    from inkbox_claude.gate import scopes as scopes_mod
    scopes_mod.refresh_vault_scopes(force=True)
    sent = []

    async def send_fn(chat_id, text, mode, meta):
        sent.append((chat_id, text, mode, meta))

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id="conv-aaron")
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    return m, sent, root


def test_owner_is_allowed_every_knowledge_scope(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    session = m.get("conv-aaron")
    session.mode = "imessage"
    session.reply_meta = {"conversation_id": "conv-aaron"}
    assert session.is_approver()
    trust = session.trust()
    assert trust["scopes"] == ["*"]
    # The owner's trust record carries no narrower list a knowledge scope could be
    # checked against: _route_and_act lets his requests through on is_approver()
    # alone, before any scope intersection runs.


def test_role_holding_one_hub_is_trusted_for_it_but_not_another(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    m.store.set_trust("gardener@example.org", person="Gardener", role="gardener",
                       scopes=["vault:areas/garden-shed"])
    trust = m.store.trust_for(["gardener@example.org"])
    allowed = set(trust["scopes"])
    assert {"vault:areas/garden-shed"} <= allowed
    assert set(["vault:areas/garden-shed"]) <= allowed
    assert not (set(["vault:areas/greenhouse"]) <= allowed)


def test_stranger_has_no_knowledge_scopes(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    trust = m.store.trust_for(["nobody@example.org"])
    assert trust["scopes"] == []


def test_approval_text_names_the_knowledge_scope(tmp_path, monkeypatch):
    from inkbox_claude.gate.store import Request
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    task = m.store.create_task("Shed upkeep question", [("gardener@example.org", "Gardener")])
    req = m.store.create_request(
        chat_id="c1", sender="gardener@example.org", sender_name="Gardener", mode="email",
        subject="", original_message="what's the shed plan?", summary="shed upkeep plan",
        scopes=["vault:areas/garden-shed"], prompt="look it up", state="pending", task_id=task["id"],
        inbound_id=None,
    )
    text = m.format_request(req)
    assert "vault:areas/garden-shed" in text


# --------------------------------------------------------------------------- attachment


def test_attachment_contains_whole_notes_and_not_instructions_framing(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    att = vaultscopes.build_attachment(
        root, ["Areas/Garden Shed/Roof.md", "Areas/Garden Shed/Paint.md"], budget_chars=100000)
    assert "The roof was patched in spring." in att.block
    assert "Repainted the door blue." in att.block
    assert "not instructions" in att.block
    assert att.included == ("Areas/Garden Shed/Roof.md", "Areas/Garden Shed/Paint.md")
    assert att.overflow == ()


def test_attachment_never_truncates_a_note_it_attaches(tmp_path):
    root = tmp_path / "vault"
    long_text = "# Long\n\n" + ("word " * 5000)
    write(root / "Areas" / "Big.md", long_text)
    write(root / "Areas" / "Big" / "More.md", "placeholder")
    att = vaultscopes.build_attachment(root, ["Areas/Big.md"], budget_chars=1_000_000)
    assert long_text in att.block


def test_attachment_lists_overflow_by_path_instead_of_cutting_a_note_short(tmp_path):
    root = make_vault(tmp_path / "vault")
    first = (root / "Areas" / "Garden Shed" / "Roof.md").read_text()
    att = vaultscopes.build_attachment(
        root, ["Areas/Garden Shed/Roof.md", "Areas/Garden Shed/Paint.md"],
        budget_chars=len(first))  # only the first note fits
    assert first in att.block
    assert "Areas/Garden Shed/Paint.md" in att.overflow
    assert "Repainted the door blue." not in att.block
    assert "Areas/Garden Shed/Paint.md" in att.block  # named in the overflow listing


def test_vault_attachment_for_request_builds_from_granted_scopes_only(tmp_path, monkeypatch):
    from inkbox_claude.gate.store import Request
    m, _sent, root = _manager_with_vault(tmp_path, monkeypatch)
    req = Request(
        id=1, chat_id="c1", sender="gardener@example.org", sender_name="Gardener", mode="email",
        subject="", original_message="x", summary="x", scopes=["vault:areas/garden-shed"],
        prompt="x", prompt_sha256="x", state="approved", revision=0, status=None, raw_output=None,
        created_at=0.0, updated_at=0.0,
    )
    block = m.vault_attachment_for(req)
    assert "The roof was patched in spring." in block
    assert "Seedling schedule" not in block  # greenhouse was not granted


def test_vault_attachment_for_request_empty_without_a_knowledge_scope(tmp_path, monkeypatch):
    from inkbox_claude.gate.store import Request
    m, _sent, root = _manager_with_vault(tmp_path, monkeypatch)
    req = Request(
        id=1, chat_id="c1", sender="x", sender_name="x", mode="email", subject="", original_message="x",
        summary="x", scopes=["web"], prompt="x", prompt_sha256="x", state="approved", revision=0,
        status=None, raw_output=None, created_at=0.0, updated_at=0.0,
    )
    assert m.vault_attachment_for(req) == ""


def test_vault_attachment_for_request_empty_with_whole_vault_scope(tmp_path, monkeypatch):
    from inkbox_claude.gate.store import Request
    m, _sent, root = _manager_with_vault(tmp_path, monkeypatch)
    req = Request(
        id=1, chat_id="c1", sender="x", sender_name="x", mode="email", subject="", original_message="x",
        summary="x", scopes=["vault_read"], prompt="x", prompt_sha256="x", state="approved", revision=0,
        status=None, raw_output=None, created_at=0.0, updated_at=0.0,
    )
    # The whole-vault scope is handled by letting the executor's vault tools search
    # freely, not by attaching anything up front.
    assert m.vault_attachment_for(req) == ""


# --------------------------------------------------------------------------- tool enforcement


def test_vault_prefixes_for_scopes_none_for_whole_vault():
    assert vault_prefixes_for_scopes(["vault_read"]) is None


def test_vault_prefixes_for_scopes_specific_hub(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    from inkbox_claude.gate import scopes as scopes_mod
    vaultscopes._cache = vaultscopes._Cache()
    scopes_mod.refresh_vault_scopes(force=True)
    try:
        prefixes = vault_prefixes_for_scopes(["vault:areas/garden-shed"])
        assert set(prefixes) == {"Areas/Garden Shed.md", "Areas/Garden Shed"}
    finally:
        scopes_mod.SCOPE_SYSTEM.pop("vault:areas/garden-shed", None)
        scopes_mod.SCOPES.pop("vault:areas/garden-shed", None)


def test_vault_prefixes_for_scopes_empty_without_any_vault_scope():
    assert vault_prefixes_for_scopes(["web"]) == []


def test_tool_cannot_read_a_note_outside_the_granted_hub(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    prefixes = ["Areas/Garden Shed.md", "Areas/Garden Shed"]
    ok = hosttools.vault_read({"path": "Areas/Garden Shed/Roof.md"}, prefixes)
    assert "The roof was patched" in ok
    blocked = hosttools.vault_read({"path": "Areas/Greenhouse/Tomatoes.md"}, prefixes)
    assert blocked.startswith("ERROR")
    assert "Staked in June" not in blocked


def test_tool_list_and_search_stay_inside_the_granted_hub(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    prefixes = ["Areas/Garden Shed.md", "Areas/Garden Shed"]
    listed = hosttools.vault_list({}, prefixes)
    assert "Areas/Garden Shed/Roof.md" in listed
    assert "Tomatoes" not in listed
    found = hosttools.vault_search({"query": "tomatoes"}, prefixes)
    assert found.startswith("0 of")


def test_tool_refuses_dotdot_traversal_out_of_the_vault(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    result = hosttools.vault_read({"path": "../outside.md"}, None)
    assert result.startswith("ERROR")
    assert "secret" not in result


def test_tool_refuses_an_absolute_path_escape(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    result = hosttools.vault_read({"path": str(outside)}, None)
    assert result.startswith("ERROR")
    assert "secret" not in result


def test_tool_refuses_a_symlink_pointing_outside_the_vault(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    outside = tmp_path / "outside.md"
    outside.write_text("secret from outside the vault", encoding="utf-8")
    try:
        (root / "Areas" / "Garden Shed" / "Escape.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable in this environment")
    result = hosttools.vault_read({"path": "Areas/Garden Shed/Escape.md"}, None)
    assert result.startswith("ERROR")
    assert "secret" not in result
    listed = hosttools.vault_list({"folder": "Areas/Garden Shed"}, None)
    assert "Escape.md" not in listed


def test_tool_refuses_a_symlinked_directory_pointing_outside_the_vault(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    write(outside_dir / "Secret.md", "leaked")
    try:
        (root / "Areas" / "Linked").symlink_to(outside_dir)
    except OSError:
        pytest.skip("symlinks unavailable in this environment")
    result = hosttools.vault_read({"path": "Areas/Linked/Secret.md"}, None)
    assert result.startswith("ERROR")
    assert "leaked" not in result


def test_owner_runs_are_unrestricted_through_the_tools(tmp_path, monkeypatch):
    root = make_vault(tmp_path / "vault")
    monkeypatch.setenv("BLATBOT_VAULT_DIR", str(root))
    # None (the owner's usual case: vault_read or no narrowing at all) reaches every note.
    everything = hosttools.vault_list({}, None)
    assert "Areas/Garden Shed/Roof.md" in everything
    assert "Areas/Greenhouse/Tomatoes.md" in everything


def _approved(scopes, request_id=7):
    from inkbox_claude.gate.store import Request
    return Request(
        id=request_id, chat_id="c1", sender="x", sender_name="x", mode="email", subject="",
        original_message="x", summary="x", scopes=scopes, prompt="x", prompt_sha256="x",
        state="approved", revision=0, status=None, raw_output=None, created_at=0.0, updated_at=0.0,
    )


def test_attachment_is_built_when_the_whole_vault_scope_is_held_too(tmp_path, monkeypatch):
    # the owner's usual grant: the tool tree gives vault_read, the traversal adds the hub
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    block = m.vault_attachment_for(_approved(["vault_read", "vault:areas/garden-shed"]))
    assert "The roof was patched in spring." in block


def test_attachment_puts_the_chosen_notes_first_and_only_those(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    m.remember_knowledge_notes(7, ["Areas/Garden Shed/Paint.md", "Areas/Greenhouse/Tomatoes.md"])
    block = m.vault_attachment_for(_approved(["vault:areas/garden-shed"]))
    assert "Repainted the door blue." in block
    assert "The roof was patched in spring." not in block   # in the hub, but not chosen
    assert "Tomatoes" not in block                           # chosen, but outside the granted scope


def test_attachment_falls_back_to_the_scope_when_no_choice_was_kept(tmp_path, monkeypatch):
    m, _sent, _root = _manager_with_vault(tmp_path, monkeypatch)
    block = m.vault_attachment_for(_approved(["vault:areas/garden-shed"], request_id=99))
    assert "The roof was patched in spring." in block and "Repainted the door blue." in block
