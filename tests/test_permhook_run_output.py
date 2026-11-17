"""Request #506's denied calls, replayed through the real permission decision."""

import os
from types import SimpleNamespace

from inkbox_claude.gate import permhook
from inkbox_claude.gate.scopes import tools_for

SESSION = "591a37c7-d388-47c2-8dbc-00f1b5640ca3"


def _decide(name, args, scopes, hook_input):
    allowed = set(tools_for(scopes))
    req = SimpleNamespace(status={}, schedule_id=2, id=506)
    return permhook.decide(req, name, args, allowed, from_owner=False, protected=[], manager=None,
                           confirm_sends=False, owner_handle=lambda *a: False,
                           read_dirs=permhook.run_read_dirs(hook_input, allowed))[0]


def test_request_506_calls(tmp_path, monkeypatch):
    home = tmp_path
    pw = home / ".playwright-mcp"
    monkeypatch.setattr(permhook, "PLAYWRIGHT_OUTPUT_DIR", str(pw))
    proj = home / ".claude" / "projects" / "-home-aaron-blatbot-exec"
    results = proj / SESSION / "tool-results"
    results.mkdir(parents=True)
    pw.mkdir()
    snap = results / "mcp-playwright-browser_snapshot-1794898689915.txt"
    snap.write_text("x")
    page = pw / "page-2026-11-17T06-57-58-096Z.yml"
    page.write_text("x")
    (home / ".env").write_text("SECRET=1")
    hook_input = {"session_id": SESSION, "transcript_path": str(proj / f"{SESSION}.jsonl")}
    scopes = ["browser_act", "browser_read", "web"]

    assert _decide("Read", {"file_path": str(page)}, scopes, hook_input) == "allow"
    assert _decide("Grep", {"pattern": "Total Raised", "path": str(snap)}, scopes, hook_input) == "allow"
    assert _decide("Read", {"file_path": str(snap), "offset": 1, "limit": 600}, scopes, hook_input) == "allow"
    assert _decide("Glob", {"pattern": "*.txt", "path": str(results)}, scopes, hook_input) == "allow"
    assert _decide("mcp__playwright__browser_network_request", {"index": 3}, scopes, hook_input) == "allow"

    # Everything else on the box stays out of reach.
    assert _decide("Read", {"file_path": str(home / ".env")}, scopes, hook_input) == "deny"
    assert _decide("Read", {"file_path": str(results / ".." / ".." / ".." / ".." / ".env")},
                   scopes, hook_input) == "deny"
    assert _decide("Read", {"file_path": "tool-results/x.txt"}, scopes, hook_input) == "deny"
    assert _decide("Grep", {"pattern": "SECRET"}, scopes, hook_input) == "deny"
    assert _decide("Grep", {"pattern": "SECRET", "path": str(home)}, scopes, hook_input) == "deny"
    assert _decide("Glob", {"pattern": "../../../*", "path": str(results)}, scopes, hook_input) == "deny"
    other = proj / "another-session" / "tool-results" / "a.txt"
    assert _decide("Read", {"file_path": str(other)}, scopes, hook_input) == "deny"
    link = results / "escape.txt"
    os.symlink(home / ".env", link)
    assert _decide("Read", {"file_path": str(link)}, scopes, hook_input) == "deny"

    # The Playwright folder only for a run that holds a Playwright tool.
    assert _decide("Read", {"file_path": str(page)}, ["tamid_drive_read"], hook_input) == "deny"
    assert _decide("Read", {"file_path": str(snap)}, ["tamid_drive_read"], hook_input) == "allow"
