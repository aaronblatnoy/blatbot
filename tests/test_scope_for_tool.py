"""A refused tool asks for one permission, not every scope that happens to carry it."""
from inkbox_claude.gate.scopes import SCOPES, scope_for_tool
from inkbox_claude.gate import wording


def test_notes_search_asks_for_one_notes_permission():
    owners = [n for n, s in SCOPES.items() if "mcp__host__vault_search" in (s.get("tools") or [])]
    assert len(owners) > 1
    assert scope_for_tool("mcp__host__vault_search") == "vault_read"


def test_unknown_tool_asks_for_nothing():
    assert scope_for_tool("mcp__nowhere__nothing") is None


def test_failure_label_is_short_for_a_notes_refusal():
    status = {"ok": False, "denied": [{"tool": "mcp__host__vault_search", "reason": "scope"},
                                      {"tool": "mcp__host__telegram_read_chat", "reason": "scope"}]}
    assert wording.denied_scopes(status) == ["vault_read", "telegram_read"]
    assert "notes on" not in wording._denied_label(status)
