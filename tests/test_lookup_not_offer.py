"""A direct factual question is looked up, not answered with an offer to look it up,
including when the typed judgments (TypeSafe) are failing. Throwaway stores, fake
models; nothing is sent and no request is executed (decide() only builds it)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inkbox_claude.gate import decidegraph
from inkbox_claude.gate import manager as gm
from inkbox_claude.gate.router import Router as GateRouter, RouterOutput

APPROVER_CONV = "conv-aaron"
OFFER = ("I don't have that in front of me, so I can't name TAMID's senior advisor off the top of my "
         "head. Say the word and I'll look it up and report back here.")
QUESTION = "Blatbot? Do you know who TAMID's senior advisor is?"


class Picker:
    def __init__(self, action):
        self.action = action

    async def pick(self, **kw):
        return {"choice": "none"}

    async def judge_action(self, **kw):
        return dict(self.action)

    # TypeSafe is down in these tests: every scope judgment comes back empty.
    async def judge_scopes_tree(self, **kw):
        return {"scopes": None, "probabilities": {}}

    async def judge_scopes(self, **kw):
        return {"scopes": None, "probabilities": {}}

    def __getattr__(self, name):
        # Every other typed judgment fails the way a 402 does: no reading at all.
        if name.startswith("judge_"):
            async def unavailable(**kw):
                return {"reason": "error: Client error '402 Payment Required'"}
            return unavailable
        raise AttributeError(name)


class Router:
    def __init__(self, reply, needs=None):
        self.reply, self.needs, self.fallback_calls = reply, needs, 0

    async def route(self, **kw):
        return RouterOutput(reply=self.reply if not kw.get("action") else "On it.", request=None)

    async def judge_needs_action(self, **kw):
        self.fallback_calls += 1
        if self.needs == "boom":
            raise AssertionError("the fallback must not run when the judgment answered")
        return self.needs

    async def plan_query(self, **kw):
        return None


def manager(tmp_path, picker, router):
    async def send_fn(*a):
        raise AssertionError("nothing is sent in these tests")

    cfg = SimpleNamespace(deepseek_api_key="x", deepseek_model="m", claude_model="sonnet",
                          approver_imessage_conversation_id=APPROVER_CONV)
    m = gm.GateSessionManager(cfg=cfg, send_fn=send_fn, mcp_server=None, identity_info={},
                              store_path=str(tmp_path / "gate.db"), exec_cwd=str(tmp_path))
    m.jev_agent = None
    m.caller = None
    m.task_picker = picker
    m.router = router
    m.router_defines_requests = False
    return m


def session(m, chat_id, sender):
    s = m.get(chat_id)
    s.mode = "imessage"
    s.reply_meta = {"sender": sender, "conversation_id": chat_id}
    return s


async def decide(s, preflight=None):
    return await decidegraph.decide(s, body=QUESTION, message=QUESTION, prior=[], mode="imessage",
                                    memory="", found="", preflight=preflight or {})


ERROR_402 = {"needs_action": None, "p": 0.0, "reason": "error: Client error '402 Payment Required'"}


@pytest.mark.asyncio
async def test_failed_judgment_asks_the_writer_model_and_the_owner_gets_the_lookup(tmp_path):
    router = Router(OFFER, needs=True)
    m = manager(tmp_path, Picker(ERROR_402), router)
    s = session(m, APPROVER_CONV, "+15550100001")
    assert s.is_approver()
    out, _task = await decide(s, preflight={"action": dict(ERROR_402)})
    assert router.fallback_calls == 1
    assert out.request is not None and "senior advisor" in out.request.prompt
    assert out.reply is None or "look it up" not in out.reply.lower()


@pytest.mark.asyncio
async def test_failed_judgment_with_no_fallback_answer_still_never_sends_the_owner_an_offer(tmp_path):
    router = Router(OFFER, needs=None)
    m = manager(tmp_path, Picker(ERROR_402), router)
    s = session(m, APPROVER_CONV, "+15550100001")
    out, _task = await decide(s)
    assert out.request is not None
    assert out.reply is None


@pytest.mark.asyncio
async def test_a_real_no_from_the_judgment_does_not_call_the_fallback(tmp_path):
    router = Router("Not that I know of.", needs="boom")
    m = manager(tmp_path, Picker({"needs_action": False, "p": 0.1, "reason": "ok"}), router)
    s = session(m, APPROVER_CONV, "+15550100001")
    out, _task = await decide(s)
    assert router.fallback_calls == 0
    assert out.request is None and out.reply == "Not that I know of."


@pytest.mark.asyncio
async def test_offer_to_someone_else_creates_no_request(tmp_path):
    # Permission rules unchanged: an offer written to anyone but the owner is not
    # turned into a run on their behalf.
    router = Router(OFFER, needs=False)
    m = manager(tmp_path, Picker(ERROR_402), router)
    s = session(m, "conv-stranger", "+15550199999")
    assert not s.is_approver()
    out, _task = await decide(s)
    assert out.request is None


@pytest.mark.asyncio
async def test_failed_judgment_for_someone_else_follows_the_normal_request_rules(tmp_path):
    # The fallback answers yes: the request is built exactly as a yes from the typed
    # judgment would build it, and for a non-owner that path ends at Aaron's approval.
    router = Router("Let me confirm with Aaron.", needs=True)
    m = manager(tmp_path, Picker(ERROR_402), router)
    s = session(m, "conv-stranger", "+15550199999")
    out, _task = await decide(s)
    assert router.fallback_calls == 1
    assert out.request is not None


def test_offer_wording_is_recognised():
    assert decidegraph.offers_lookup(OFFER)
    assert decidegraph.offers_lookup("I can look it up and report back here if you want.")
    assert decidegraph.offers_lookup("Want me to check?")
    assert not decidegraph.offers_lookup("I can't find that anywhere.")
    assert not decidegraph.offers_lookup("You can look it up on the site.")
    assert not decidegraph.offers_lookup("Thanks, noted.")


@pytest.mark.asyncio
async def test_router_fallback_reads_the_models_yes_no(monkeypatch):
    r = GateRouter.__new__(GateRouter)

    async def yes(messages, json_mode=True):
        return '{"needs_action": true}'

    async def boom(messages, json_mode=True):
        raise RuntimeError("down")

    monkeypatch.setattr(r, "_chat", yes)
    assert await r.judge_needs_action(history=[], message="who is the senior advisor?") is True
    monkeypatch.setattr(r, "_chat", boom)
    assert await r.judge_needs_action(history=[], message="who is the senior advisor?") is None
