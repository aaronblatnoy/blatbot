"""When the typed group judgment fails, the writer model decides whether to speak."""
import json
import types

import pytest

from inkbox_claude import gateway
from inkbox_claude.gate.router import Router
from inkbox_claude.gate.store import Store


class FailingPicker:
    async def judge_group_reply(self, **kw):
        return {"should_reply": None, "p": 0.0, "reason": "error: 402 Payment Required"}


class RaisingPicker:
    async def judge_group_reply(self, **kw):
        raise RuntimeError("boom")


class RealPicker:
    async def judge_group_reply(self, **kw):
        return {"should_reply": False, "p": 0.1, "reason": "ok"}


def _router(answer_for):
    r = Router(api_key="k", model="m")

    async def chat(messages, json_mode=True):
        user = messages[-1]["content"]
        v = answer_for(user)
        if v == "fail":
            raise RuntimeError("model down")
        return json.dumps({"should_reply": v})
    r._chat = chat
    return r


def _gw(tmp_path, picker, router):
    store = Store(str(tmp_path / "gate.db"))
    g = gateway.InkboxGateway.__new__(gateway.InkboxGateway)
    g.sessions = types.SimpleNamespace(store=store, task_picker=picker, router=router)
    g._tg_recent = {}
    g.dispatched, g.overheard = [], []

    async def dispatch(u):
        g.dispatched.append(u["text"])

    g._telegram_dispatch = dispatch
    g._telegram_overhear = lambda u: g.overheard.append(u["text"])
    return g


def _u(text):
    return {"chat_id": "-1", "from_id": "7", "name": "Aaron", "text": text, "group_title": "Team"}


def _by_text(user):
    return "look it up" in user.lower()


@pytest.mark.asyncio
async def test_follow_up_is_answered_when_typed_judgment_fails(tmp_path):
    g = _gw(tmp_path, FailingPicker(), _router(_by_text))
    await g._telegram_group_maybe(_u("Look it up on tamids website"))
    assert g.dispatched == ["Look it up on tamids website"] and not g.overheard


@pytest.mark.asyncio
async def test_chatter_stays_quiet_when_typed_judgment_fails(tmp_path):
    g = _gw(tmp_path, FailingPicker(), _router(_by_text))
    await g._telegram_group_maybe(_u("Okie dokie thanks"))
    assert not g.dispatched and g.overheard == ["Okie dokie thanks"]


@pytest.mark.asyncio
async def test_both_checks_failing_stays_quiet(tmp_path):
    g = _gw(tmp_path, FailingPicker(), _router(lambda u: "fail"))
    await g._telegram_group_maybe(_u("Look it up on tamids website"))
    assert not g.dispatched and g.overheard


@pytest.mark.asyncio
async def test_picker_exception_also_falls_back(tmp_path):
    g = _gw(tmp_path, RaisingPicker(), _router(_by_text))
    await g._telegram_group_maybe(_u("Look it up on tamids website"))
    assert g.dispatched


@pytest.mark.asyncio
async def test_real_typed_answer_is_used_as_before(tmp_path):
    # A real "no" from the typed judgment is not overridden by the fallback.
    g = _gw(tmp_path, RealPicker(), _router(lambda u: True))
    await g._telegram_group_maybe(_u("Look it up on tamids website"))
    assert not g.dispatched and g.overheard
