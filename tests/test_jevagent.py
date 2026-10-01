"""The Jev agent: typed judgments pick tools, code fills arguments, prose only on demand."""
import asyncio
import json
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
        if set(options) == {"answer", "act"}:
            want = "act" if (self.picks and jevagent.is_write_tool(self.picks[0])) else "answer"
            return want, 0.95, {want: 0.95}
        want = self.picks.pop(0)
        assert want in options, (want, list(options))
        return want, 0.95, {want: 0.95}

    async def yes(self, state, question):
        self.calls += 1
        if "WITH CONFIDENCE" in str(question):
            return 0.9 if not self.picks else 0.1
        return self.yes_p

    async def ask(self, state, questions):
        """Batched argument questions: supply:: -> yes_p, pick:: -> next scripted pick."""
        self.calls += 1
        out = {}
        consumed = None
        for qid, q in questions.items():
            if q["type"] == "choice" and qid.startswith("tool::"):
                # the tool tree asks every system for its pick at once; only the system holding
                # the next scripted pick answers with it, the others say none_of_these
                want = self.picks[0] if self.picks else None
                if want in q["criteria"]:
                    out[qid] = {"choice": want, "probabilities": {want: 0.95}}
                    consumed = want
                else:
                    out[qid] = {"choice": "none_of_these", "probabilities": {"none_of_these": 0.9}}
                continue
            if q["type"] == "noul" and qid.startswith("sys::"):
                # the tool tree: a system is the next step iff the next scripted pick lives in it
                here = set(q["instructions"].get("tools_here") or [])
                want = self.picks[0].split("__")[-1] if self.picks else ""
                out[qid] = {"noul": 0.95 if want in here else 0.05}
            elif q["type"] == "noul":
                out[qid] = {"noul": 0.0 if qid.startswith("use::") else self.yes_p}   # no parallel extras unless a test asks
            elif (not self.picks or self.picks[0] not in q["criteria"]) and "write_new" in q["criteria"]:
                # a span/candidate question this script did not plan for: let prose write it
                out[qid] = {"choice": "write_new", "probabilities": {"write_new": 0.95}}
            else:
                want = self.picks.pop(0)
                assert want in q["criteria"], (want, list(q["criteria"]))
                out[qid] = {"choice": want, "probabilities": {want: 0.95}}
        if consumed is not None and self.picks and self.picks[0] == consumed:
            self.picks.pop(0)
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
    monkeypatch.setenv("JEV_AGENT_COMPOSE", "jev")     # the scripted fakes drive Jev selection; compose has its own test
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
    judge = ScriptedJudge([create, "create", "write_new"], yes=0.1)  # enum, then start_time: no named range fits noon tomorrow
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
    assert prose.calls == 3 and st["prose_calls"] == 3        # item extraction + summary + start_time
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
    # a delete pauses for the owner's yes; the selected id is in the pending call
    assert not st["ok"] and st["confirm"]["args"]["event_id"] == "evt_ZZZZZZZZZZZZZZZZ999"
    assert [c[0] for c in box.calls] == [lst] and prose.calls == 1   # only the item extraction; the id was selected


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
    assert not st["ok"] and "insufficient evidence" in st["error"]      # nothing gathered: nothing to deliver


def test_tool_failing_twice_stops(agent, monkeypatch):
    t = "mcp__tamid-drive__get_events"
    box = FakeBox({t: {"description": "list", "schema": {"type": "object", "properties": {}, "required": []}}}, fail=[t])
    _patch(monkeypatch, box, ScriptedJudge([t, t]), ScriptedProse(), [t])
    st = asyncio.run(agent.run(_req("what's on")))
    assert not st["ok"] and "keeps failing" in st["error"] and len(box.calls) == 2


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
            self.executor = type("E", (), {"run": staticmethod(lambda req, context, prior_work="": _aw({"ok": True, "summary": "claude did it"}))})()

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
    async def confident(state, question):
        return 0.8 if "WITH CONFIDENCE" in str(question) else 0.0
    j.yes = confident
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


def test_date_range_bounds_are_picked_not_written(agent, monkeypatch):
    from inkbox_claude.gate.jevagent import _date_ranges
    get = "mcp__stern-drive__get_events"
    schemas = {get: {"description": "events", "schema": {"type": "object", "required": ["time_min", "time_max"],
                                                        "properties": {"user_google_email": {"type": "string"},
                                                                       "time_min": {"type": "string", "description": "RFC3339 start"},
                                                                       "time_max": {"type": "string", "description": "RFC3339 end"}}}}}
    box = FakeBox(schemas, results={get: "6 events"})
    # the lone read is collected without a tool choice; the script holds only the two date picks
    judge = ScriptedJudge(["last_week", "last_week"])
    prose = ScriptedProse()
    _patch(monkeypatch, box, judge, prose, [get])
    st = asyncio.run(agent.run(_req("list my coffee chats from last week")))
    assert st["ok"] and prose.calls == 0
    args = box.calls[0][1]
    r = _date_ranges()["last_week"]
    assert args["time_min"] == r["start"] and args["time_max"] == r["end"]
    assert args["time_min"] < args["time_max"]


def test_sends_to_requester_guard():
    from inkbox_claude.gate.scopes import sends_to_requester
    prot = ["e3b9cc0b-conv", "+1 (407) 808-8771"]
    assert sends_to_requester("mcp__inkbox__inkbox_send_imessage", {"conversation_id": "e3b9cc0b-conv", "body": "x"}, prot)
    assert sends_to_requester("mcp__inkbox__inkbox_send_sms", {"to": "14078088771"}, prot)
    assert sends_to_requester("mcp__inkbox__inkbox_send_email", {"to": ["Cand@nyu.edu"]}, ["cand@nyu.edu"])
    assert not sends_to_requester("mcp__inkbox__inkbox_send_sms", {"to": "+15550100001"}, prot)
    assert not sends_to_requester("mcp__inkbox__inkbox_get_imessage_conversation", {"conversation_id": "e3b9cc0b-conv"}, prot)


def test_agent_refuses_to_message_requester(agent, monkeypatch):
    lst, send = "mcp__stern-drive__get_events", "mcp__inkbox__inkbox_send_sms"
    schemas = {lst: {"description": "list", "schema": {"type": "object", "properties": {}, "required": []}},
               send: {"description": "send sms", "schema": {"type": "object", "required": ["to", "body"],
                                                            "properties": {"to": {"type": "string", "description": "phone"}, "body": {"type": "string"}}}}}
    box = FakeBox(schemas, results={lst: "6 events"})
    judge = ScriptedJudge([lst, send, "c2"])           # c2 = the requester's own number (c0, c1 are the account emails)
    prose = ScriptedProse()
    _patch(monkeypatch, box, judge, prose, [lst, send])
    r = _req("text me at 407-808-8771 my coffee chats from last week"); r.sender = "+14078088771"
    st = asyncio.run(agent.run(r))
    assert st["ok"] and [c[0] for c in box.calls] == [lst], box.calls     # the send never happened


def test_slim_state_only_replaces_oversized_strings():
    from inkbox_claude.gate.jevagent import _slim_state, _LARGE
    st = {"goal": "g", "known_facts": {"a": "x" * (_LARGE + 1), "b": "small"}, "steps": ["s -> " + "y" * (_LARGE + 5)]}
    out, n = _slim_state(st)
    assert n == 2 and out["known_facts"]["b"] == "small" and out["goal"] == "g"
    assert "character tool result" in out["known_facts"]["a"] and "character tool result" in out["steps"][0]


def test_judge_retries_slim_on_too_large(monkeypatch):
    from inkbox_claude.gate import jevagent
    j = jevagent.Judge()
    seen = []
    async def post(state, questions):
        seen.append(state)
        if len(seen) == 1:
            raise jevagent.JudgeTooLarge("max_tokens_exceeded")
        return {"q": {"noul": 0.9}}
    j._post = post
    big = {"goal": "g", "known_facts": {"r": "z" * (jevagent._LARGE + 1)}}
    ans = asyncio.run(j.ask(big, {"q": {"type": "noul", "instructions": "?"}}))
    assert ans == {"q": {"noul": 0.9}} and len(seen) == 2 and "character tool result" in seen[1]["known_facts"]["r"]

    async def boom(state, questions): raise RuntimeError("down")
    j._post = boom
    assert asyncio.run(j.ask({"goal": "g"}, {"q": {"type": "noul", "instructions": "?"}})) == {}


def test_wrong_first_document_leads_to_the_next_candidate(agent, monkeypatch):
    """First sheet opened is not the one; the second call must not be offered it again."""
    search, info = "mcp__tamid-drive__search_drive_files", "mcp__tamid-drive__get_spreadsheet_info"
    schemas = {search: {"description": "search", "schema": {"type": "object", "required": ["query"],
                                                            "properties": {"user_google_email": {"type": "string"}, "query": {"type": "string"}}}},
               info: {"description": "sheet info", "schema": {"type": "object", "required": ["spreadsheet_id"],
                                                              "properties": {"user_google_email": {"type": "string"}, "spreadsheet_id": {"type": "string"}}}}}
    box = FakeBox(schemas, results={search: 'Found: "Coffee Chat Tracker" (ID: 1AAAAAAAAAAAAAAAAAAAAAAAA1) | "New Member App Fall 2026 (Responses)" (ID: 1BBBBBBBBBBBBBBBBBBBBBBBB2)',
                                    info: "sheet: 3 tabs"})
    # picks: search, then info with c0 (wrong sheet), then info again: only ONE candidate left, so c0 is now the other id
    judge = ScriptedJudge([search, info, "c0", info, "c0"])
    class Never(ScriptedJudge):
        async def yes(self, state, question):
            if "WITH CONFIDENCE" in str(question):
                return 0.9 if not self.picks else 0.1
            return 0.9
    judge = Never([search, info, "c0", info, "c0"])
    prose = ScriptedProse()
    _patch(monkeypatch, box, judge, prose, [search, info])
    st = asyncio.run(agent.run(_req("how many rows are in the fall 2026 application responses sheet?")))
    assert st["ok"]
    ids = [c[1]["spreadsheet_id"] for c in box.calls if c[0] == info]
    assert ids == ["1AAAAAAAAAAAAAAAAAAAAAAAA1", "1BBBBBBBBBBBBBBBBBBBBBBBB2"]


def test_prose_is_told_which_queries_already_failed():
    from inkbox_claude.gate.jevagent import _tried_values
    steps = [{"tool": "t", "args": {"query": "acceptance rate"}, "ok": True, "result": "0 results"},
             {"tool": "t", "args": {"query": "reviewer"}, "ok": True, "result": "0 results"},
             {"tool": "other", "args": {"query": "x"}, "ok": True, "result": ""}]
    assert _tried_values(steps, "t", "query") == ["acceptance rate", "reviewer"]
    assert _tried_values(steps, "t", "nope") == []
    multi = [{"tool": "r", "args": {"user_google_email": "x", "spreadsheet_id": "S1"}, "ok": True, "result": ""},
             {"tool": "r", "args": {"user_google_email": "x", "spreadsheet_id": "S2", "range_name": "Tab!A1"}, "ok": True, "result": ""}]
    assert _tried_values(multi, "r", "spreadsheet_id", sole=True) == ["S1"]      # S2 was read with a tab: still offerable


def test_error_text_results_count_as_failures():
    from inkbox_claude.gate.jevagent import _looks_like_error
    assert _looks_like_error("Error calling tool 'search_drive_files': API error in search_drive_files: <HttpError 400 ...>")
    assert _looks_like_error("### Error\nBrowser is not installed")
    assert _looks_like_error('API returned "Invalid value"')
    assert not _looks_like_error("Successfully listed 5 calendars")
    assert not _looks_like_error('{"success": true, "count": 0, "data": []}')
    assert not _looks_like_error("Found 10 messages matching 'error report'")
    assert _looks_like_error("### Page\n- Page URL: https://www.linkedin.com/authwall?trk=x\n- Page Title: Sign Up | LinkedIn")
    assert not _looks_like_error("### Page\n- Page URL: https://www.nyutamid.org/our-board\n- Page Title: Executive Board")


def test_error_result_makes_agent_retry_with_other_args(agent, monkeypatch):
    search = "mcp__tamid-drive__search_drive_files"
    schemas = {search: {"description": "search", "schema": {"type": "object", "required": ["query"],
                                                            "properties": {"user_google_email": {"type": "string"}, "query": {"type": "string"}}}}}
    class Box(FakeBox):
        async def call(self, name, args):
            self.calls.append((name, args))
            return "Error calling tool: HttpError 400 bad corpora" if len(self.calls) == 1 else "Found: X (ID: 1CCCCCCCCCCCCCCCCCCCCCCC3)"
    box = Box(schemas)
    class P(ScriptedProse):
        async def write(self, instruction, facts):
            self.calls += 1; self.asks.append(instruction)
            return "second query" if "DIFFERENT" in instruction else "first query"
    judge = ScriptedJudge([search, search])
    prose = P()
    _patch(monkeypatch, box, judge, prose, [search])
    st = asyncio.run(agent.run(_req("find the mentorship sheet")))
    assert st["ok"] and [c[1]["query"] for c in box.calls] == ["first query", "second query"]
    assert st["steps"][0]["ok"] is False and st["steps"][1]["ok"] is True


def test_third_consecutive_same_tool_is_not_offered(agent, monkeypatch):
    search, info = "mcp__tamid-drive__search_drive_files", "mcp__tamid-drive__get_spreadsheet_info"
    schemas = {search: {"description": "search", "schema": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}},
               info: {"description": "info", "schema": {"type": "object", "required": ["spreadsheet_id"], "properties": {"spreadsheet_id": {"type": "string"}}}}}
    box = FakeBox(schemas, results={search: "Found: Sheet (ID: 1DDDDDDDDDDDDDDDDDDDDDDD4)", info: "3 tabs, 40 rows"})
    seen_options = []
    class J(ScriptedJudge):
        async def choose(self, state, question, options, min_p=None):
            seen_options.append(sorted(options))
            return await super().choose(state, question, options, min_p)
        async def yes(self, state, question):
            if "WITH CONFIDENCE" in str(question):
                return 0.9 if not self.picks else 0.1
            return 0.9
    class P(ScriptedProse):
        async def write(self, instruction, facts):
            self.calls += 1
            return "q%d" % self.calls
    judge = J([search, search, info, "c0"])
    _patch(monkeypatch, box, judge, P(), [search, info])
    st = asyncio.run(agent.run(_req("how many rows in the sheet")))
    assert st["ok"] and [c[0] for c in box.calls] == [search, search, info]
    assert search not in seen_options[2]          # third step: search was withheld


def test_unsure_after_reads_with_low_done_score_fails_over(agent, monkeypatch):
    lst = "mcp__stern-drive__get_events"
    box = FakeBox({lst: {"description": "list", "schema": {"type": "object", "properties": {}, "required": []}}}, results={lst: "nothing relevant"})
    class Unsure(ScriptedJudge):
        async def choose(self, state, question, options, min_p=None):
            if self.picks:
                return await super().choose(state, question, options, min_p)
            return None, 0.2, {}
        async def yes(self, state, question):
            return 0.05
    _patch(monkeypatch, box, Unsure([lst]), ScriptedProse(), [lst])
    st = asyncio.run(agent.run(_req("book the thing")))
    assert st["ok"] and st["partial"] is not None and "CONFIDENCE:" in st["raw"]


def test_fit_leaves_states_under_budget_untouched_and_digests_only_when_over():
    from inkbox_claude.gate.jevagent import _fit, _BUDGET
    mid = {"goal": "g", "steps_so_far": [{"result": "x" * 40000}]}          # big, but under budget
    assert _fit(mid) == mid
    assert _fit(mid, reserve=50000) != mid                                    # the questions count too
    over = {"goal": "g", "steps_so_far": [{"result": "y" * 90000}, {"result": "z" * 90000}]}
    fitted = _fit(over)
    assert fitted["goal"] == "g" and len(json.dumps(fitted)) <= _BUDGET
    results = [st["result"] for st in fitted["steps_so_far"]]
    assert any(isinstance(r, dict) for r in results)                         # the largest got digested
    assert any(r == "y" * 90000 or r == "z" * 90000 for r in results) or all(isinstance(r, dict) for r in results)


def test_digest_breaks_large_results_down_and_passes_small_ones_through():
    from inkbox_claude.gate.jevagent import _digest, _LARGE
    assert _digest("Successfully listed 5 calendars") == "Successfully listed 5 calendars"
    rows = [{"id": "row%04d_ABCDEFGHIJKLMNOP" % i, "email": "p%d@example.org" % i, "name": "Person %d" % i} for i in range(1500)]
    big_json = json.dumps({"success": True, "count": len(rows), "data": rows})
    assert len(big_json) > _LARGE
    d = _digest(big_json)
    assert d["kind"] == "json" and d["data_count"] == 1500 and "email" in d["data_item_keys"] and d["emails_found"] == 1500
    big_text = "\n".join("row %d | p%d@example.org | Person %d" % (i, i, i) for i in range(3000))
    d = _digest(big_text)
    assert d["kind"] == "text" and len(d["first_lines"]) == 5 and "3,000 lines" in d["size"]
    assert len(json.dumps(d)) < 2000


class NeedleJudge:
    """Scores a part high only if it contains NEEDLE."""
    def __init__(self): self.calls = 0
    async def ask(self, state, questions):
        self.calls += 1
        part = state.get("part", "")
        return {"q": {"noul": 0.95 if "NEEDLE" in part else 0.05}}


def test_narrow_finds_the_part_with_the_needle_in_log_rounds():
    from inkbox_claude.gate.jevagent import narrow, _LEAF
    lines = ["row %05d | filler filler filler filler filler filler filler filler | x%05d@example.org" % (i, i) for i in range(6000)]
    lines[4321] = "row 04321 | NEEDLE Cand Idate | cand@example.org"
    text = "\n".join(lines)
    j = NeedleJudge()
    leaf = asyncio.run(narrow(j, text, "the row for Cand Idate", "email cand idate"))
    assert "NEEDLE" in leaf and len(leaf) <= _LEAF * 1.2
    assert j.calls <= 40                       # a handful of parallel rounds, not one call per row


def test_agent_selects_from_a_narrowed_large_result_and_never_judges_raw_text(agent, monkeypatch):
    from inkbox_claude.gate.jevagent import _LARGE
    lst, get = "mcp__tamid-drive__read_sheet_values", "mcp__tamid-drive__get_form_response"
    schemas = {lst: {"description": "read", "schema": {"type": "object", "properties": {}, "required": []}},
               get: {"description": "get one response", "schema": {"type": "object", "required": ["response_id"],
                                                                   "properties": {"response_id": {"type": "string", "description": "the response id"}}}}}
    from inkbox_claude.gate.jevagent import _BUDGET
    rows = ["row %05d | ID: resp_%05dAAAAAAAAAAAAAA | filler filler filler filler filler filler" % (i, i) for i in range(4000)]
    rows[2500] = "row 02500 | ID: resp_NEEDLE0000000000000 | Cand Idate"
    big = "\n".join(rows)
    assert len(big) > _BUDGET
    box = FakeBox(schemas, results={lst: big, get: "the response"})
    seen_states = []
    class J(ScriptedJudge):
        async def ask(self, state, questions):
            seen_states.append(state)
            if "part" in state:
                return {"q": {"noul": 0.95 if "NEEDLE" in state["part"] else 0.05}}
            return await super().ask(state, questions)
        async def choose(self, state, question, options, min_p=None):
            seen_states.append(state)
            return await super().choose(state, question, options, min_p)
        async def yes(self, state, question):
            seen_states.append(state)
            if "WITH CONFIDENCE" in str(question):
                return 0.9 if len(box.calls) >= 2 else 0.1     # done once the response itself was read
            return 0.9
        async def ask(self, state, questions):  # batched argument questions: pick the candidate holding the needle
            seen_states.append(state)
            if "part" in state:
                return {"q": {"noul": 0.95 if "NEEDLE" in state["part"] else 0.05}}
            out = {}
            for qid, q in questions.items():
                if q["type"] == "noul":
                    out[qid] = {"noul": 0.9}
                else:
                    hit = [k for k, v in q["criteria"].items() if "NEEDLE" in json.dumps(v)] or ["write_new"]
                    out[qid] = {"choice": hit[0], "probabilities": {hit[0]: 0.97}}
            return out
    judge = J([])                      # both reads are collected on usefulness, no tool choice
    _patch(monkeypatch, box, judge, ScriptedProse(), [lst, get])
    st = asyncio.run(agent.run(_req("open cand idate's application response")))
    assert st["ok"] and box.calls[1][1]["response_id"] == "resp_NEEDLE0000000000000"
    for state in seen_states:
        for v in json.loads(json.dumps(state)).values() if isinstance(state, dict) else []:
            pass
        assert len(json.dumps(state, default=str)) <= _BUDGET * 1.05, "an over-budget state reached a judgment"
    assert big in st["raw"]                                          # the full result still reaches the report


def _strings(v):
    if isinstance(v, str): return [v]
    if isinstance(v, dict): return [s for x in v.values() for s in _strings(x)]
    if isinstance(v, list): return [s for x in v for s in _strings(x)]
    return []


def test_browser_search_reads_results_off_the_search_page():
    from inkbox_claude.gate.jevagent import ToolBox
    snap = ('### Page\n- Page URL: http://127.0.0.1:8888/search?q=x&format=json\n### Snapshot\n```yaml\n- generic [active] [ref=e1]: ' + json.dumps(json.dumps({"results": [
        {"title": "Sean Parnell - Student at New York University | LinkedIn", "url": "https://www.linkedin.com/in/seanparnelll",
         "content": "Sean Parnell. Student at New York University.  TAMID at NYU Co-President."},
        {"title": "Executive Board - TAMID Group at NYU", "url": "https://www.nyutamid.org/our-board", "content": ""}]})) + '\n```')
    class Box(ToolBox):
        def __init__(self): self.calls = []
        async def call(self, name, args):
            self.calls.append((name, args))
            return "" if name.endswith("navigate") else snap
    b = Box()
    out = asyncio.run(b._browser_search("Sean Parnell TAMID NYU"))
    assert b.calls[0][0].endswith("browser_navigate") and "q=Sean%20Parnell%20TAMID%20NYU" in b.calls[0][1]["url"]
    assert "1. Sean Parnell - Student at New York University | LinkedIn" in out and "url: https://www.linkedin.com/in/seanparnelll" in out
    assert "Co-President" in out and "127.0.0.1" not in out and "2. Executive Board" in out


def test_choose_accepts_a_clear_leader_among_many_options():
    from inkbox_claude.gate.jevagent import Judge
    j = Judge()
    async def ask(state, questions):
        return {"q": {"choice": "a", "probabilities": {"a": 0.3, "b": 0.12, "c": 0.1, "d": 0.1, "e": 0.1, "f": 0.1}}}
    j.ask = ask
    pick, p, _ = asyncio.run(j.choose({}, "?", {k: k for k in "abcdef"}))
    assert pick == "a" and p == 0.3                      # under the 0.45 floor, but 2.5x the runner-up
    async def ask2(state, questions):
        return {"q": {"choice": "a", "probabilities": {"a": 0.3, "b": 0.25, "c": 0.1, "d": 0.1, "e": 0.1, "f": 0.1}}}
    j.ask = ask2
    assert asyncio.run(j.choose({}, "?", {k: k for k in "abcdef"}))[0] is None   # no clear leader


def test_ids_are_found_under_camelcase_and_json_keys():
    from inkbox_claude.gate.jevagent import _candidate_values
    blob = ('{"formId": "1OGQhnPoO9OzdYqzUgdtBcxYdeEdeLjYGTzE7", "responses": [{"responseId": "ACYDBNhX0m9vQ", "respondentEmail": "a@nyu.edu"}]}'
            ' spreadsheet_id=18FY8UeEMQsKqYUHlrm7 Event ID: evt_abc123def456')
    ids = _candidate_values("response_id", "", "string", {"request": "x"}, [{"result": blob}])
    assert "ACYDBNhX0m9vQ" in ids and "1OGQhnPoO9OzdYqzUgdtBcxYdeEdeLjYGTzE7" in ids and "18FY8UeEMQsKqYUHlrm7" in ids and "evt_abc123def456" in ids





def test_multi_argument_tools_keep_offering_a_value_read_once(agent, monkeypatch):
    """A spreadsheet read once (no range) must stay offerable for a read with a range."""
    read = "mcp__tamid-drive__read_sheet_values"
    schemas = {read: {"description": "read", "schema": {"type": "object", "required": ["spreadsheet_id"],
                                                        "properties": {"user_google_email": {"type": "string"}, "spreadsheet_id": {"type": "string"},
                                                                       "range_name": {"type": "string", "description": "range"}}}}}
    box = FakeBox(schemas, results={read: "Successfully read 2 rows: Row 1: ['x'] (ID: 1AAAAAAAAAAAAAAAAAAAAAAAA1)"})
    class J(ScriptedJudge):
        async def ask(self, state, questions):
            self.calls += 1
            out = {}
            for qid, q in questions.items():
                if q["type"] == "noul":
                    out[qid] = {"noul": 0.9 if qid == "supply::range_name" and len(box.calls) >= 1 else 0.2}
                else:
                    opts = q["criteria"]
                    want = next((k for k, v in opts.items() if "1AAAA" in json.dumps(v)), None) or "write_new"
                    out[qid] = {"choice": want, "probabilities": {want: 0.95}}
            return out
        async def yes(self, state, question):
            if "WITH CONFIDENCE" in str(question):
                return 0.9 if not self.picks else 0.1
            return 0.9
    class P(ScriptedProse):
        async def write(self, instruction, facts):
            self.calls += 1
            return "'Tab2'!A1:Z10" if "range" in instruction else "UNKNOWN"
    judge = J([read, read])
    _patch(monkeypatch, box, judge, P(), [read])
    st = asyncio.run(agent.run(_req("read sheet 1AAAAAAAAAAAAAAAAAAAAAAAA1 tab2")))
    ids = [c[1].get("spreadsheet_id") for c in box.calls]
    assert st["ok"] and len(ids) >= 2 and set(ids) == {"1AAAAAAAAAAAAAAAAAAAAAAAA1"}


def test_large_result_is_read_into_evidence_before_judging(agent, monkeypatch):
    """A huge sheet read becomes a small piece of evidence about the goal; the done
    judgment and the report both work from it."""
    from inkbox_claude.gate.jevagent import _LARGE
    read = "mcp__tamid-drive__read_sheet_values"
    schemas = {read: {"description": "read", "schema": {"type": "object", "properties": {}, "required": []}}}
    rows = ["Row %d: ['p%d@nyu.edu', 'Person %d', 'CAS', 'filler filler filler filler']" % (i, i, i) for i in range(4000)]
    rows[2222] = "Row 2222: ['ci@nyu.edu', 'Cand Idate', 'Stern', 'NEEDLE']"
    big = "\n".join(rows); assert len(big) > _LARGE
    box = FakeBox(schemas, results={read: big})
    seen = []
    class J(ScriptedJudge):
        async def ask(self, state, questions):
            seen.append(state)
            if "part" in state:
                return {"q": {"noul": 0.95 if "NEEDLE" in state["part"] else 0.05}}
            return await super().ask(state, questions)
        async def yes(self, state, question):
            seen.append(state)
            if "WITH CONFIDENCE" in str(question):
                return 0.9 if any("NEEDLE" in json.dumps(st) for st in [state]) else 0.1
            return 0.9
    _patch(monkeypatch, box, J([read]), ScriptedProse(), [read])
    st = asyncio.run(agent.run(_req("which school is Cand Idate in? check the responses sheet")))
    assert st["ok"] and st["tool_calls"] == [read]
    done_states = [x for x in seen if "steps_so_far" in x]
    assert done_states and "NEEDLE" in json.dumps(done_states[-1]) and len(json.dumps(done_states[-1])) < 20000
    assert big in st["raw"]


def test_text_arguments_are_selected_from_spans_before_being_written(agent, monkeypatch):
    from inkbox_claude.gate.jevagent import _span_candidates
    spans = _span_candidates("check tamid coffee chat tracking form for fall 2026 and see how many coffee chats Max Levy did.", [])
    assert "Max Levy" in spans and "coffee chat tracking form" in spans and "the" not in spans
    search = "mcp__tamid-drive__search_drive_files"
    schemas = {search: {"description": "search", "schema": {"type": "object", "required": ["query"],
                                                            "properties": {"query": {"type": "string", "description": "words to search for"}}}}}
    box = FakeBox(schemas, results={search: "Found: Coffee Chat Tracker (ID: 1AAAAAAAAAAAAAAAAAAAAAAAA1)"})
    class J(ScriptedJudge):
        async def ask(self, state, questions):
            self.calls += 1
            out = {}
            for qid, q in questions.items():
                if q["type"] == "noul":
                    out[qid] = {"noul": 0.9}
                else:
                    want = next(k for k, v in q["criteria"].items() if json.dumps(v) == json.dumps({"value": "coffee chat tracking form"}))
                    out[qid] = {"choice": want, "probabilities": {want: 0.9}}
            return out
        async def yes(self, state, question):
            return 0.9
    prose = ScriptedProse()
    _patch(monkeypatch, box, J([search]), prose, [search])
    st = asyncio.run(agent.run(_req("check tamid coffee chat tracking form for fall 2026 and see how many coffee chats Max Levy did.")))
    assert st["ok"] and box.calls[0][1]["query"] == "coffee chat tracking form" and prose.calls == 0


def test_agent_pauses_before_a_destructive_call(agent, monkeypatch):
    from inkbox_claude.gate.jevagent import is_destructive
    assert is_destructive("mcp__tamid-drive__manage_event", {"action": "delete"}) and is_destructive("mcp__sjba-admin__sjba_delete_event", {})
    assert not is_destructive("mcp__tamid-drive__manage_event", {"action": "create"}) and not is_destructive("mcp__tamid-drive__get_events", {})
    find, change = "mcp__tamid-drive__get_events", "mcp__tamid-drive__manage_event"
    schemas = {find: {"description": "events", "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}},
               change: {"description": "change", "schema": {"type": "object", "required": ["action", "event_id"],
                                                            "properties": {"action": {"type": "string", "enum": ["create", "update", "delete"]},
                                                                           "event_id": {"type": "string"}}}}}
    box = FakeBox(schemas, results={find: '- "Sam Rivera and TAMID at NYU (Quant)" (Starts: 2026-10-02T15:00) ID: evt_QUANT000000000001', change: "deleted"})
    class J(ScriptedJudge):
        async def yes(self, state, question):
            return 0.1 if "WITH CONFIDENCE" in str(question) else 0.9
    _patch(monkeypatch, box, J([find, change, "delete", "c0"]), ScriptedProse(), [find, change])
    st = asyncio.run(agent.run(_req("cancel sam rivera's quant interview")))
    assert not st["ok"] and st["confirm"]["tool"] == change and st["confirm"]["args"]["event_id"] == "evt_QUANT000000000001"
    assert "Sam Rivera and TAMID at NYU (Quant)" in st["confirm"]["about"][0]
    assert [c[0] for c in box.calls] == [find] and "WRITES PERFORMED: none" in st["raw"]


def test_useful_reads_are_collected_in_parallel_and_land_in_the_state(agent, monkeypatch):
    """Two read tools rated useful run together with the pick; the next judgment sees all three results."""
    a, b, c = "mcp__tamid-drive__get_events", "mcp__tamid-drive__search_gmail_messages", "mcp__tamid-admin__tamid_list_board_members"
    schemas = {t: {"description": t, "schema": {"type": "object", "properties": {}, "required": []}} for t in (a, b, c)}
    box = FakeBox(schemas, results={a: "EVENTS", b: "MAILS", c: "ROSTER"})
    seen = []
    class J(ScriptedJudge):
        async def ask(self, state, questions):
            if any(q.startswith("use::") for q in questions):
                return {q: {"noul": 0.9} for q in questions}          # everything is useful now
            return await super().ask(state, questions)
        async def yes(self, state, question):
            seen.append(state)
            if "WITH CONFIDENCE" in str(question):
                return 0.9 if len([s for s in state.get("steps_so_far", [])]) >= 3 else 0.1
            return 0.9
    _patch(monkeypatch, box, J([a]), ScriptedProse(), [a, b, c])
    st = asyncio.run(agent.run(_req("were they sent interviews and did they book?")))
    assert st["ok"] and sorted(c0 for c0, _ in box.calls) == sorted([a, b, c])
    done_state = [x for x in seen if "steps_so_far" in x][-1]
    assert {s["tool"] for s in done_state["steps_so_far"]} == {a, b, c}


def test_run_ends_only_with_confidence(agent, monkeypatch):
    lst = "mcp__stern-drive__get_events"
    box = FakeBox({lst: {"description": "list", "schema": {"type": "object", "properties": {}, "required": []}}}, results={lst: "6 events"})
    class J(ScriptedJudge):
        async def choose(self, state, question, options, min_p=None):
            if self.picks:
                return await super().choose(state, question, options, min_p)
            return None, 0.2, {}
        async def yes(self, state, question):
            return 0.5 if "WITH CONFIDENCE" in str(question) else 0.0     # plausible is not enough
    _patch(monkeypatch, box, J([lst]), ScriptedProse(), [lst])
    st = asyncio.run(agent.run(_req("what's on my calendar")))
    # reads exhausted: findings are delivered, with the confidence stated for the reply writer
    assert st["ok"] and st["partial"] == 0.5 and "CONFIDENCE: 0.50" in st["raw"]


def test_blind_delete_is_never_offered_for_confirmation(agent, monkeypatch):
    """No event was looked up: a delete with a written id is refused and the agent
    goes and lists events; the confirmation names the real event."""
    find, change = "mcp__tamid-drive__get_events", "mcp__tamid-drive__manage_event"
    schemas = {find: {"description": "events", "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}},
               change: {"description": "change", "schema": {"type": "object", "required": ["action", "event_id"],
                                                            "properties": {"action": {"type": "string", "enum": ["create", "update", "delete"]},
                                                                           "event_id": {"type": "string"}}}}}
    box = FakeBox(schemas, results={find: '- "Sam Rivera and TAMID at NYU (Quant)" (Starts: 2026-10-02T09:00) ID: evt_REAL000000000000001'})
    class P(ScriptedProse):
        async def write(self, instruction, facts):
            self.calls += 1
            return "3sq" if "event_id" in instruction else "Sam Rivera"
    class J(ScriptedJudge):
        async def yes(self, state, question):
            return 0.1 if "WITH CONFIDENCE" in str(question) else 0.9
    # a write is never offered while facts are still missing: the lookup comes first, then the
    # delete with the real id; the written id "3sq" from prose never reaches the call
    judge = J([find, change, "delete", "c0"])
    _patch(monkeypatch, box, judge, P(), [find, change])
    st = asyncio.run(agent.run(_req("cancel sam rivera's 9am interview")))
    assert not st["ok"] and st.get("confirm"), st.get("error")
    assert st["confirm"]["args"]["event_id"] == "evt_REAL000000000000001"
    assert st["confirm"]["about"] and "Sam Rivera" in st["confirm"]["about"][0]
    assert [c[0] for c in box.calls] == [find]                                   # nothing deleted, one lookup
    assert "3sq" not in json.dumps(st["confirm"])


def test_arguments_are_composed_by_prose_and_ids_validated(agent, monkeypatch):
    """One prose call composes every argument; a composed id must exist in evidence."""
    monkeypatch.setenv("JEV_AGENT_COMPOSE", "deepseek")
    lst, get = "mcp__tamid-drive__get_events", "mcp__tamid-drive__delete_event"
    schemas = {lst: {"description": "List events", "schema": {"type": "object", "properties": {"user_google_email": {"type": "string"}, "query": {"type": "string"}}, "required": []}},
               get: {"description": "Delete an event", "schema": {"type": "object", "required": ["event_id"],
                                                                 "properties": {"user_google_email": {"type": "string"}, "event_id": {"type": "string"}}}}}
    box = FakeBox(schemas, results={lst: "id=evt_ABCDEFGHIJKLMNOP123 Lunch; id=evt_ZZZZZZZZZZZZZZZZ999 Dentist"})
    class P(ScriptedProse):
        async def write(self, instruction, facts):
            self.calls += 1
            if "arguments_schema" in json.dumps(facts, default=str):
                if facts["tool"] == "get_events":
                    return '{"query": "dentist"}'
                return '{"event_id": "evt_MADEUP0000000000000"}' if self.calls == 2 else '{"event_id": "evt_ZZZZZZZZZZZZZZZZ999"}'
            return "[]"
    judge = ScriptedJudge([lst, get, get])
    _patch(monkeypatch, box, judge, P(), [lst, get])
    st = asyncio.run(agent.run(_req("cancel the dentist")))
    assert box.calls[0] == (lst, {"user_google_email": jevagent.ORG_ACCOUNT, "query": "dentist"})
    # the invented id was rejected (lookup-first), the real one reached the confirmation
    assert not st["ok"] and st.get("confirm") and st["confirm"]["args"]["event_id"] == "evt_ZZZZZZZZZZZZZZZZ999"
