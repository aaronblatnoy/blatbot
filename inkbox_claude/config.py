"""Shared Inkbox Claude Code bridge configuration helpers."""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from .realtime import (
    DEFAULT_MODEL as REALTIME_DEFAULT_MODEL,
    DEFAULT_VOICE as REALTIME_DEFAULT_VOICE,
    RealtimeConfig,
)

# Empty means "do not override"; the Inkbox SDK owns its API default.
INKBOX_BASE_URL_DEFAULT = ""
INKBOX_WS_PATH = "/phone/media/ws"

USER_AGENT_NAME = "inkbox-claude-code"
DISTRIBUTION_NAME = "claude-code-plugin"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8767
DEFAULT_WEBHOOK_PATH = "/webhook"


class VoiceStack(str, Enum):
    """Supported phone-call voice stacks."""

    INKBOX_VOICE_AI = "inkbox_voice_ai"
    OPENAI_REALTIME = "openai_realtime"
    INKBOX_TTS_STT = "inkbox_tts_stt"

# Tools Claude Code may run without texting the human first. Everything
# else (Bash, Write, Edit, ...) escalates over the active channel.
DEFAULT_AUTO_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "Task",
    "NotebookRead",
]


def state_dir() -> Path:
    """~/.inkbox-claude (or INKBOX_CLAUDE_HOME)."""
    return Path(os.getenv("INKBOX_CLAUDE_HOME") or (Path.home() / ".inkbox-claude"))


def call_contexts_dir() -> Path:
    """Directory where ``inkbox_place_call`` stashes per-call context."""
    root = Path(os.getenv("INKBOX_CLAUDE_HOME") or (Path.home() / ".inkbox-claude"))
    path = root / "call_contexts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str) -> List[str]:
    raw = os.getenv(name) or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class BridgeConfig:
    api_key: str = ""
    identity: str = ""
    signing_key: str = ""
    base_url: str = INKBOX_BASE_URL_DEFAULT
    public_url: str = ""
    tunnel_name: str = ""
    home_channel: str = ""
    allowed_users: List[str] = field(default_factory=list)
    allow_all_users: bool = False
    require_signature: bool = True
    # Leave webhook subscriptions alone on start. For deployments that
    # provision them ahead of time, where the destination is fixed or this
    # API key may not change it; they must already point at this bridge.
    skip_webhook_reconcile: bool = False
    # Wake the agent on unrecognised/unverified external webhooks (default
    # off: only registered, signature-verified sources get through).
    external_events_enabled: bool = False
    contact_memories_enabled: bool = True
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # Claude Code side
    project_dir: str = ""
    # E.164 phone that receives every permission/poll escalation instead of
    # the remote party. Empty = stock behaviour (ask the sender themselves).
    approver_phone: str = ""
    # Existing iMessage conversation id with the approver. When set, approval
    # requests go over iMessage; otherwise they fall back to SMS.
    approver_imessage_conversation_id: str = ""
    # Cross-session approval tickets (stranger session texts the approver and
    # waits). Off by default: strangers are refused and hand off instead.
    approver_tickets_enabled: bool = False
    # "sessions" = Claude Code per contact (stock). "gate" = DeepSeek router +
    # approval gate + deterministic Claude executor (inkbox_claude.gate).
    mode: str = "sessions"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    gate_db_path: str = ""
    gate_exec_dir: str = ""
    gate_standing_path: str = ""
    claude_model: str = ""
    permission_timeout_s: float = 600.0
    auto_allowed_tools: List[str] = field(default_factory=lambda: list(DEFAULT_AUTO_ALLOWED_TOOLS))
    voice_stack: VoiceStack = VoiceStack.INKBOX_TTS_STT
    voice_stack_invalid_value: str = ""
    voice_ai_authority_mode: str = "contact_scoped"
    voicemail_detection: str = "enabled"
    # OpenAI Realtime voice (off unless the wizard validated a key)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)


def inkbox_base_url_kwargs(base_url: str | None = None) -> Dict[str, str]:
    normalized = str(base_url or "").strip()
    return {"base_url": normalized} if normalized else {}


@lru_cache(maxsize=1)
def plugin_user_agent() -> str:
    """Identifies this plugin ahead of the SDK's own ``User-Agent`` token."""
    try:
        version = importlib.metadata.version(DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"{USER_AGENT_NAME}/{version}"


def inkbox_client_kwargs(api_key: str, base_url: str | None = None) -> Dict[str, str]:
    return {
        "api_key": api_key,
        "user_agent_prefix": plugin_user_agent(),
        **inkbox_base_url_kwargs(base_url),
    }


_BLATBOT_VOICE = (
    "You are Blatbot, Aaron Blatnoy's executive assistant, answering his assistant line. You're a guy in "
    "your late twenties: relaxed, quick, dry sense of humor, low-key confident. Talk like a real person on "
    "the phone, not like a customer service agent."
)
# Delivery steering written to flatten Realtime's performance. GPT-Live drops it (see live.py):
# it fought the chosen voice's natural character and made Cedar sound like a different voice.
_BLATBOT_DELIVERY = (
    " Your voice is grounded and plain: chest voice, low "
    "in your range, steady pitch, relaxed pace. Think of a calm guy talking to a friend across a table, not a "
    "host, a concierge or a narrator. Keep the energy a notch below the caller's. No brightness, no smile in "
    "the voice, no breathiness, no drawn-out vowels, no melodic swoops, and sentences end flat or falling, never "
    "rising. Dry humor stays dry: say the funny thing in the same flat tone as everything else."
)


def _read_realtime_config() -> RealtimeConfig:
    """Build the Realtime voice config from the env.

    The API key falls back to OPENAI_API_KEY so an operator who already
    exports one doesn't have to re-enter it. Realtime stays disabled unless
    INKBOX_REALTIME_ENABLED is truthy.

    Returns:
        RealtimeConfig: Resolved settings; ``enabled`` False leaves calls on
        the Inkbox STT/TTS path.
    """
    api_key = str(os.getenv("INKBOX_REALTIME_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    return RealtimeConfig(
        enabled=env_flag("INKBOX_REALTIME_ENABLED", False) and bool(api_key),
        api_key=api_key,
        model=str(os.getenv("INKBOX_REALTIME_MODEL") or REALTIME_DEFAULT_MODEL).strip(),
        voice=str(os.getenv("INKBOX_REALTIME_VOICE") or REALTIME_DEFAULT_VOICE).strip(),
        fallback_to_inkbox_stt_tts=env_flag("INKBOX_REALTIME_FALLBACK_TO_INKBOX_STT_TTS", True),
        additional_instructions=str(os.getenv("INKBOX_REALTIME_INSTRUCTIONS") or (_BLATBOT_VOICE + _BLATBOT_DELIVERY)).strip(),
    )


def resolve_voice_stack(
    value: Any,
    *,
    realtime_enabled: Any = None,
    realtime_api_key: str = "",
) -> tuple[VoiceStack, str]:
    """Resolve the canonical stack while preserving pre-selector installs."""
    normalized = str(value or "").strip().lower()
    if normalized:
        try:
            return VoiceStack(normalized), ""
        except ValueError:
            return VoiceStack.INKBOX_TTS_STT, normalized
    if realtime_enabled is not None:
        enabled = str(realtime_enabled).strip().lower() in {"auto", "1", "true", "yes", "on"}
        if not enabled:
            return VoiceStack.INKBOX_TTS_STT, ""
    if realtime_api_key:
        return VoiceStack.OPENAI_REALTIME, ""
    return VoiceStack.INKBOX_TTS_STT, ""


def read_config(extra: Dict[str, Any] | None = None) -> BridgeConfig:
    extra = extra or {}
    realtime = _read_realtime_config()
    voice_stack, invalid_voice_stack = resolve_voice_stack(
        extra.get("voice_stack") or os.getenv("INKBOX_VOICE_STACK"),
        realtime_enabled=os.getenv("INKBOX_REALTIME_ENABLED"),
        realtime_api_key=realtime.api_key,
    )
    return BridgeConfig(
        api_key=str(extra.get("api_key") or os.getenv("INKBOX_API_KEY") or "").strip(),
        identity=str(extra.get("identity") or os.getenv("INKBOX_IDENTITY") or "").strip(),
        signing_key=str(extra.get("signing_key") or os.getenv("INKBOX_SIGNING_KEY") or "").strip(),
        base_url=str(extra.get("base_url") or os.getenv("INKBOX_BASE_URL") or INKBOX_BASE_URL_DEFAULT).strip(),
        public_url=str(extra.get("public_url") or os.getenv("INKBOX_PUBLIC_URL") or "").strip(),
        tunnel_name=str(extra.get("tunnel_name") or os.getenv("INKBOX_TUNNEL_NAME") or "").strip(),
        home_channel=str(os.getenv("INKBOX_HOME_CHANNEL") or extra.get("home_channel") or "").strip(),
        allowed_users=_csv_env("INKBOX_ALLOWED_USERS"),
        allow_all_users=env_flag("INKBOX_ALLOW_ALL_USERS", False),
        require_signature=env_flag("INKBOX_REQUIRE_SIGNATURE", True),
        skip_webhook_reconcile=env_flag("INKBOX_SKIP_WEBHOOK_RECONCILE", False),
        external_events_enabled=env_flag("INKBOX_EXTERNAL_EVENTS_ENABLED", False),
        contact_memories_enabled=env_flag("INKBOX_CONTACT_MEMORIES_ENABLED", True),
        host=str(os.getenv("INKBOX_BRIDGE_HOST") or DEFAULT_HOST).strip(),
        port=int(os.getenv("INKBOX_BRIDGE_PORT") or DEFAULT_PORT),
        project_dir=str(os.getenv("CLAUDE_PROJECT_DIR") or extra.get("project_dir") or os.getcwd()).strip(),
        approver_phone=str(os.getenv("INKBOX_APPROVER_PHONE") or extra.get("approver_phone") or "").strip(),
        approver_imessage_conversation_id=str(
            os.getenv("INKBOX_APPROVER_IMESSAGE_CONVERSATION_ID")
            or extra.get("approver_imessage_conversation_id")
            or ""
        ).strip(),
        approver_tickets_enabled=str(os.getenv("INKBOX_APPROVER_TICKETS") or "").strip().lower() in ("1", "true", "yes"),
        mode=str(os.getenv("INKBOX_MODE") or "sessions").strip().lower(),
        deepseek_api_key=str(os.getenv("DEEPSEEK_API_KEY") or "").strip(),
        deepseek_model=str(os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip(),
        gate_db_path=str(os.getenv("GATE_DB_PATH") or str(state_dir() / "gate.db")).strip(),
        gate_exec_dir=str(os.getenv("GATE_EXEC_DIR") or str(state_dir() / "exec")).strip(),
        gate_standing_path=str(os.getenv("GATE_STANDING_PATH") or str(state_dir() / "standing.md")).strip(),
        claude_model=str(os.getenv("CLAUDE_MODEL") or extra.get("claude_model") or "").strip(),
        permission_timeout_s=float(os.getenv("INKBOX_PERMISSION_TIMEOUT_S") or 600.0),
        auto_allowed_tools=_csv_env("INKBOX_AUTO_ALLOWED_TOOLS") or list(DEFAULT_AUTO_ALLOWED_TOOLS),
        voice_stack=voice_stack,
        voice_stack_invalid_value=invalid_voice_stack,
        voice_ai_authority_mode=str(
            extra.get("voice_ai_authority_mode")
            or os.getenv("INKBOX_VOICE_AI_AUTHORITY_MODE")
            or "contact_scoped"
        ).strip().lower(),
        voicemail_detection=str(
            extra.get("voicemail_detection")
            or os.getenv("INKBOX_VOICEMAIL_DETECTION")
            or "enabled"
        ).strip().lower(),
        realtime=realtime,
    )
