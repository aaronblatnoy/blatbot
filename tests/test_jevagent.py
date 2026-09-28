"""The Jev agent: typed judgments pick tools, code fills arguments, prose only on demand."""
import asyncio
import os

import pytest

from inkbox_claude.gate import jevagent
from inkbox_claude.gate.jevagent import JevAgent, ToolBox, is_write_tool, _candidate_values, _coerce
from inkbox_claude.gate.store import Request, sha256


def _req(msg="add lunch with Sam Rivera tomorrow at noon", scopes=("tamid_calendar_write",), prompt=None):
    prompt = prompt or msg
    return Request(id=1, chat_id="c1", sender="s", sender_name="Aaron", mode="imessage", subject="",
                   original_message=msg, summary=msg, scopes=list(scopes), prompt=prompt,
                   prompt_sha256=sha256(prompt), state="approved", revision=0, status=None, raw_output=None, created_at=0.0, updated_at=0.0)


class FakeBox:
    """Stands in for ToolBox: fixed schemas, records calls."""
    def __init__(self, schemas, results=None, fail=()):
        self.schemas, self.results, self.fail, self.calls = schemas, results or {}, set(fail), []

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return None

    async def schema(self, name):
        return self.schemas.get(name, {"description": "", "schema": {}})

    async def call(self, name, args):
        self.calls.append((name, args))
        if name in self.fail:
            raise RuntimeError("boom")
        return self.results.get(name, "ok")


class ScriptedJudge:
    """Answers Choice questions from a script; Noul answers: 'done' questions say
    yes once the scripted picks are exhausted, other Nouls return `yes`."""
    def __init__(self, picks, yes=0.9):
        self.picks, self.yes_p, self.calls = list(picks), yes, 0

    async def choose(self, state, question, options, min_p=None):
        self.calls += 1
        want = self.picks.pop(0)
        assert want in options, (want, list(options))
        return want, 0.95, {want: 0.95}

    async def yes(self, state, question):
        self.calls += 1
        if "fully achieved" in str(question):
            return 0.9 if not self.picks else 0.1
        return self.yes_p

    async def ask(self, state, questions):
        """Batched argument questions: supply:: -> yes_p, pick:: -> next scripted pick."""
        self.calls += 1
        out = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                out[qid] = {"noul": self.yes_p}
            else:
                want = self.picks.pop(0)
                assert want in q["criteria"], (want, list(q["criteria"]))
                out[qid] = {"choice": want, "probabilities": {want: 0.95}}
        return out


class ScriptedProse:
    def __init__(self):
        self.calls, self.asks = 0, []

    async def write(self, instruction, facts):
        self.calls += 1
        self.asks.append(instruction)
        return "2026-09-29T12:00:00-04:00" if "start" in instruction else "Lunch with Sam Rivera"


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    ag = JevAgent(inkbox_server={"type": "sdk", "name": "inkbox", "instance": None}, router=None, mcp_config={}, max_steps=5)
    return ag


def _patch(monkeypatch, box, judge, prose, allowed):
    monkeypatch.setattr(jevagent, "ToolBox", lambda *a, **k: box)
    monkeypatch.setattr(jevagent, "Judge", lambda: judge)
    monkeypatch.setattr(jevagent, "Prose", lambda router: prose)
    monkeypatch.setattr(jevagent, "tools_for", lambda scopes: allowed)


def test_enabled_requires_env(monkeypatch):
    monkeypatch.delenv("GATE_EXECUTOR", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    assert not jevagent.enabled()
    monkeypatch.setenv("GATE_EXECUTOR", "jev")
    assert jevagent.enabled()
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert not jevagent.enabled()


def test_calendar_write_fills_args_and_finishes(agent, monkeypatch):
    create = "mcp__tamid-drive__manage_event"
    schemas = {create: {"description": "Create a calendar event", "schema": {
        "type": "object", "required": ["user_google_email", "action", "summary", "start_time"],
        "properties": {"user_google_email": {"type": "string"}, "action": {"type": "string", "enum": ["create", "update"]},
                       "summary": {"type": "string", "description": "title"},
                       "start_time": {"type": "string", "description": "start"},
                       "description": {"type": "string"}}}}}
    box = FakeBox(schemas, results={create: "Event created id=abc"})
    # step 1 tool pick, then action enum, then summary write_new (no candidates -> straight to prose), start write_new, then done
    judge = ScriptedJudge([create, "create"], yes=0.1)  # optional description skipped (0.1)
    prose = ScriptedProse()
    _patch(monkeypatch, box, judge, prose, [create])
    st = asyncio.run(agent.run(_req(), context="T1 lunch"))
    assert st["ok"] and st["engine"] == "jev"
    assert box.calls[0][0] == create
    args = box.calls[0][1]
    assert args["user_google_email"] == jevagent.ORG_ACCOUNT
    assert args["action"] == "create" and args["summary"] == "Lunch with Sam Rivera"
    assert args["start_time"].startswith("2026-09-29T12:00")
    assert "description" not in args
    assert prose.calls == 2 and st["prose_calls"] == 2
    assert "STATUS: OK" in st["raw"] and st["wrote"] is True


def test_candidate_id_is_selected_not_written(agent, monkeypatch):
    lst, get = "mcp__tamid-drive__get_events", "mcp__tamid-drive__delete_event"
    schemas = {lst: {"description": "List events", "schema": {"type": "object", "properties": {"user_google_email": {"type": "string"}}, "required": []}},
               get: {"description": "Delete an event", "schema": {"type": "object", "required": ["event_id"],
                                                                 "properties": {"user_google_email": {"type": "string"}, "event_id": {"type": "string"}}}}}
    box = FakeBox(schemas, results={lst: "id=evt_ABCDEFGHIJKLMNOP123 Lunch; id=evt_ZZZZZZZZZZZZZZZZ999 Dentist"})
    judge = ScriptedJudge([lst, get, "c1"])  # c1 = the second id (Dentist)
    prose = ScriptedProse()
    _patch(monkeypatch, box, judge, prose, [lst, get])
    st = asyncio.run(agent.run(_req("cancel the dentist")))
    assert st["ok"]
    assert box.calls[1][1]["event_id"] == "evt_ZZZZZZZZZZZZZZZZ999"
    assert prose.calls == 0


def test_give_up_and_unsure_fail_without_side_effects(agent, monkeypatch):
    t = "mcp__inkbox__inkbox_send_email"
    box = FakeBox({t: {"description": "send", "schema": {}}})
    _patch(monkeypatch, box, ScriptedJudge(["give_up"]), ScriptedProse(), [t])
    st = asyncio.run(agent.run(_req("do a thing")))
    assert not st["ok"] and "gave up" in st["error"] and box.calls == [] and st["wrote"] is False

    class Unsure(ScriptedJudge):
        async def choose(self, state, question, options, min_p=None):
            return None, 0.3, {}
    _patch(monkeypatch, box, Unsure([]), ScriptedProse(), [t])
    st = asyncio.run(agent.run(_req("do a thing")))
    assert not st["ok"] and "unsure" in st["error"]


def test_tool_failing_twice_stops(agent, monkeypatch):
    t = "mcp__tamid-drive__get_events"
    box = FakeBox({t: {"description": "list", "schema": {"type": "object", "properties": {}, "required": []}}}, fail=[t])
    _patch(monkeypatch, box, ScriptedJudge([t, t]), ScriptedProse(), [t])
    st = asyncio.run(agent.run(_req("what's on")))
    assert not st["ok"] and "failed twice" in st["error"] and len(box.calls) == 2


def test_hash_mismatch_refused(agent):
    r = _req(); r.prompt = "tampered"
    st = asyncio.run(agent.run(r))
    assert not st["ok"] and "hash" in st["error"]


def test_helpers():
    assert is_write_tool("mcp__inkbox__inkbox_send_email") and is_write_tool("mcp__tamid-drive__manage_event")
    assert not is_write_tool("mcp__tamid-drive__get_events") and not is_write_tool("mcp__stern-drive__read_sheet_values")
    facts = {"request": "email sam.rivera@example.org and jamie@example.org"}
    assert _candidate_values("to", "recipient email", "string", facts, []) == ["sam.rivera@example.org", "jamie@example.org"]
    assert _coerce("3", "integer") == 3 and _coerce("a, b", "array") == ["a", "b"] and _coerce('"x"', "string") == "x"


def test_manager_fallback_rules(monkeypatch):
    """Manager falls back to Claude only when the jev agent failed WITHOUT writing."""
    from inkbox_claude.gate import manager as mg

    class M:  # minimal stand-in with the method under test
        _run_executor = mg.GateSessionManager._run_executor
        def __init__(self, jev_status, fallback=True):
            self.jev_fallback = fallback
            self.jev_agent = type("A", (), {"run": staticmethod(lambda req, context: _aw(jev_status))})()
            self.executor = type("E", (), {"run": staticmethod(lambda req, context: _aw({"ok": True, "summary": "claude did it"}))})()

    async def _aw(v): return v
    req = _req()
    st = asyncio.run(M({"ok": False, "error": "gave up", "wrote": False})._run_executor(req, ""))
    assert st["ok"] and st["summary"] == "claude did it" and st["jev_attempt"]["error"] == "gave up"
    st = asyncio.run(M({"ok": False, "error": "x", "wrote": True})._run_executor(req, ""))
    assert not st["ok"]
    st = asyncio.run(M({"ok": False, "error": "x", "wrote": False}, fallback=False)._run_executor(req, ""))
    assert not st["ok"]


def test_give_up_after_reads_finishes_with_findings(agent, monkeypatch):
    lst = "mcp__stern-drive__get_events"
    box = FakeBox({lst: {"description": "list", "schema": {"type": "object", "properties": {}, "required": []}}},
                  results={lst: "6 events"})
    j = ScriptedJudge([lst, "give_up"])
    async def never_done(state, question):
        return 0.4
    j.yes = never_done
    _patch(monkeypatch, box, j, ScriptedProse(), [lst, "mcp__inkbox__inkbox_send_imessage"])
    st = asyncio.run(agent.run(_req("list my coffee chats last week")))
    assert st["ok"] and st["tool_calls"] == [lst] and "6 events" in st["raw"]


def test_inkbox_server_call_across_mcp_versions():
    from inkbox_claude.gate.jevagent import _call_server

    class Entry:
        def __init__(self): self.handler = self.h
        async def h(self, ctx, params): return {"got": params}

    class New:  # method-keyed entries with (ctx, params) handlers
        def get_request_handler(self, m): return Entry() if m == "tools/list" else None

    class Req:
        def __init__(self, method, params=None): self.method, self.params = method, params

    class Old:  # request-class keyed callables
        request_handlers = {}
    async def old_h(req): return type("R", (), {"root": {"old": req.method}})()
    Old.request_handlers[Req] = old_h

    assert asyncio.run(_call_server(New(), "tools/list", Req, {"a": 1})) == {"got": {"a": 1}}
    assert asyncio.run(_call_server(Old(), "x", Req, None)) == {"old": "x"}
    with pytest.raises(RuntimeError):
        asyncio.run(_call_server(New(), "tools/call", Req, None))
