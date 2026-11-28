import asyncio
import json
import os
import time

from inkbox_claude.gate import ccsessions, hosttools


def _line(**kw):
    return json.dumps(kw)


def _fake_claude(tmp_path, busy=True):
    root = tmp_path / ".claude"
    proj = root / "projects" / "-home-aaron-demo"
    proj.mkdir(parents=True)
    (root / "sessions").mkdir()
    sid = "abcd1234-0000-0000-0000-000000000000"
    lines = [
        _line(type="user", timestamp="2026-11-28T07:00:00.000Z", cwd="/home/aaron/demo", entrypoint="cli",
              message={"role": "user", "content": "fix the login bug"}),
        _line(type="user", timestamp="2026-11-28T07:00:01.000Z",
              message={"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}),
        _line(type="user", timestamp="2026-11-28T07:00:02.000Z",
              message={"role": "user", "content": "<system-reminder>ignore</system-reminder>"}),
        _line(type="assistant", timestamp="2026-11-28T07:00:03.000Z",
              message={"role": "assistant", "content": [{"type": "text", "text": "Fixed it. token=abcdef123456 was removed."}]}),
    ]
    (proj / f"{sid}.jsonl").write_text("\n".join(lines) + "\n")
    (root / "sessions" / f"{os.getpid()}.json").write_text(json.dumps(
        {"pid": os.getpid(), "sessionId": sid, "cwd": "/home/aaron/demo", "name": "demo-1",
         "status": "busy" if busy else "idle", "startedAt": int(time.time() * 1000)}))
    return root, sid


def test_snapshot_reads_latest_ask_gist_and_state(tmp_path, monkeypatch):
    root, sid = _fake_claude(tmp_path)
    monkeypatch.setenv("CCSESSIONS_CLAUDE_DIR", str(root))
    snap = ccsessions.snapshot("black-sky", hours=1)
    s = snap["sessions"][0]
    assert s["id"] == sid and s["state"] == "working" and s["process_alive"]
    assert s["latest_ask"] == "fix the login bug"          # tool results and harness text are not asks
    assert "abcdef123456" not in s["gist"] and "Fixed it" in s["gist"]   # secrets masked


def test_stale_mac_snapshot_is_reported_as_last_seen(tmp_path, monkeypatch):
    root, _ = _fake_claude(tmp_path)
    monkeypatch.setenv("CCSESSIONS_CLAUDE_DIR", str(root))
    monkeypatch.setenv("CCSESSIONS_DIR", str(tmp_path / "state"))
    old = time.time() - 3 * 3600
    mac = {"machine": "Mac", "generated_at": ccsessions._iso(old), "sessions": [{
        "id": "feedbeef-1", "name": "", "title": "Deck", "cwd": "/Users/a", "project": "~", "automated": False,
        "state": "working", "process_alive": True, "last_activity": ccsessions._iso(old),
        "latest_ask": "build the deck", "gist": "on it", "last_tool": "", "first_ask": "", "asks": [], "turns": []}]}
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "mac.json").write_text(json.dumps(mac))
    text = ccsessions.report({"hours": 6})
    assert "Mac last seen at" in text and "not current" in text
    assert "[Mac]" in text and "[black-sky]" in text
    assert "no snapshot" not in text


def test_no_mac_snapshot_says_so(tmp_path, monkeypatch):
    root, _ = _fake_claude(tmp_path)
    monkeypatch.setenv("CCSESSIONS_CLAUDE_DIR", str(root))
    monkeypatch.setenv("CCSESSIONS_DIR", str(tmp_path / "nowhere"))
    assert "no snapshot has arrived" in ccsessions.report({"machine": "all"})


def test_tool_is_commander_only(monkeypatch):
    monkeypatch.setenv("CLAUDE_SESSIONS_ENABLED", "1")
    assert asyncio.run(hosttools.claude_sessions({}, False)) == hosttools.CLAUDE_SESSIONS_REFUSAL


def test_tool_is_off_unless_the_instance_enables_it(monkeypatch):
    monkeypatch.delenv("CLAUDE_SESSIONS_ENABLED", raising=False)
    assert not ccsessions.enabled()
    assert "not switched on" in asyncio.run(hosttools.claude_sessions({}, True))
    for name, want in (("blatbot", True), ("gondroid", False)):
        text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "deploy", "instances", name + ".yaml")).read()
        assert ("CLAUDE_SESSIONS_ENABLED: \"%d\"" % int(want)) in text


def test_scope_is_in_blatbot_but_not_gondroid():
    from inkbox_claude.gate import scopes
    assert "claude_sessions_read" in scopes.SCOPES
    assert "mcp__host__claude_sessions" in scopes.tools_for(["claude_sessions_read"])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gond = open(os.path.join(root, "deploy", "instances", "gondroid", "scopes.yaml")).read()
    assert "claude_sessions" not in gond
