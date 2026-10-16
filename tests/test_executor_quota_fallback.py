import asyncio
import json
import time
from types import SimpleNamespace

from inkbox_claude.gateway import InkboxGateway
from inkbox_claude.gate.executor import Executor, is_quota_error, quota_reset_time
from inkbox_claude.gate.manager import GateSessionManager
from inkbox_claude.gate.store import Request, sha256
from inkbox_claude.gate.wording import usage_limit_reset_text


def _request() -> Request:
    prompt = "check the calendar"
    return Request(
        id=1, chat_id="owner", sender="Aaron", sender_name="Aaron", mode="imessage",
        subject="", original_message=prompt, summary=prompt, scopes=[], prompt=prompt,
        prompt_sha256=sha256(prompt), state="approved", revision=0, status=None,
        raw_output=None, created_at=0, updated_at=0,
    )


def test_quota_error_detector_matches_only_quota_phrase_families():
    assert is_quota_error("You have hit your weekly limit")
    assert is_quota_error("RATE LIMIT exceeded")
    assert is_quota_error("The service is temporarily Overloaded")
    assert not is_quota_error("The calendar tool timed out")
    assert not is_quota_error("The result limit was hit")


def test_quota_reset_time_parser():
    assert quota_reset_time("Limit reached; resets 3:10pm (America/New_York).") == \
        "3:10pm (America/New_York)"
    assert quota_reset_time("The service is overloaded") is None
    assert usage_limit_reset_text({"reset_time": "3:10pm (America/New_York)"}) == \
        "3:10pm (America/New_York)"


def test_active_quota_cooldown_skips_sdk_launch(tmp_path, monkeypatch):
    launched = False

    async def should_not_launch(*args, **kwargs):
        nonlocal launched
        launched = True
        if False:
            yield None

    monkeypatch.setattr("inkbox_claude.gate.executor.query", should_not_launch)
    executor = Executor(mcp_server=None, cwd=str(tmp_path))
    executor._quota_until = time.time() + 60
    executor._quota_reset_text = "3:10pm (America/New_York)"

    status = asyncio.run(executor.run(_request()))

    assert status["quota_error"] is True
    assert status["reset_time"] == "3:10pm (America/New_York)"
    assert status["tool_calls"] == []
    assert launched is False


def test_manager_uses_caller_when_claude_reports_quota(monkeypatch):
    class Claude:
        async def run(self, *args, **kwargs):
            return {"ok": False, "quota_error": True, "reset_time": "4pm", "tool_calls": []}

    class Caller:
        async def run(self, req, context=""):
            return {"ok": True, "tool_calls": ["calendar_read"], "summary": "Done via Jev."}

    manager = GateSessionManager.__new__(GateSessionManager)
    manager.executor = Claude()
    manager.caller = Caller()
    manager.jev_agent = None
    manager._pick_model = lambda req: _async_value("sonnet")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    status = asyncio.run(manager._run_executor(_request(), "context"))

    assert status["ok"] is True
    assert status["quota_fallback"] is True


def test_health_includes_executor_quota_state():
    gateway = InkboxGateway.__new__(InkboxGateway)
    gateway.cfg = SimpleNamespace(identity="blatbot")
    gateway.sessions = SimpleNamespace(
        executor=SimpleNamespace(quota_state=lambda: {
            "state": "exhausted", "reset_time": "4pm",
        })
    )

    response = asyncio.run(gateway._handle_health(None))

    assert json.loads(response.text)["quota"] == {
        "state": "exhausted", "reset_time": "4pm",
    }


async def _async_value(value):
    return value
