"""Per-chat response mode for group conversations: how freely Blatbot may speak into
a room it did not start, decided per chat rather than once for every channel.

Three modes:
  natural  - today's behavior: Jev, or the model's own judgment inside a turn, decides
             whether an unaddressed message is Blatbot's to answer.
  mention  - only replies when a message names it, is a reply to its own last message,
             or (on Telegram) @mentions its username. Nothing else reaches Jev or the
             model: an unaddressed message in a mention-mode room is recorded, never
             judged.
  silent   - never replies, regardless of whether it was addressed. Still recorded for
             context and digests.

DMs are never gated by this: a direct thread is always natural, since there is no
room to decide not to speak into (see callers in gateway.py, which only look this up
for a group chat).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

MODES = ("natural", "mention", "silent")
DEFAULT_GROUP_MODE = "mention"


def normalize_mode(value: Optional[str]) -> str:
    """A stored mode value, lowercased and trimmed, or "" when it is unset or not one
    of the three known modes -- never a silent typo-to-loudest-mode fallback."""
    v = (value or "").strip().lower()
    return v if v in MODES else ""


def effective_mode(stored: Optional[str]) -> str:
    """What a chat's mode actually is right now: the explicit value if one was ever
    set on it, otherwise the product default for a group nobody has tuned yet."""
    return normalize_mode(stored) or DEFAULT_GROUP_MODE


_NAME_AT_START_RE = re.compile(r"^\s*@?(blatbot|blat)\b", re.IGNORECASE)


def addressed_by_name(text: str) -> bool:
    """A case-insensitive match of Blatbot's name at the very start of the message --
    "blatbot, can you..." or "blat do x" -- the channel-agnostic half of mention
    detection, used as-is for iMessage/SMS (no @mentions or reply-to there) and
    alongside Telegram's own username/reply-to checks."""
    return bool(_NAME_AT_START_RE.match(str(text or "")))


_MODE_WORDS = "|".join(MODES)
_SELF_MODE_RE = re.compile(rf"^\s*blatbot\s+mode\s+({_MODE_WORDS})\s*[.!]?\s*$", re.IGNORECASE)
_NAMED_MODE_RE = re.compile(rf"^\s*blatbot\s+mode\s+(.+?)\s+({_MODE_WORDS})\s*[.!]?\s*$", re.IGNORECASE)


def parse_mode_command(text: str) -> Optional[Tuple[Optional[str], str]]:
    """Parse a commander mode-setting command, case-insensitively.

    "blatbot mode mention" -> (None, "mention"): sets the mode of the chat the
    message was sent in.
    "blatbot mode Corgi NYC Team Chat mention" -> ("Corgi NYC Team Chat", "mention"):
    sets the named chat's mode, for sending from a DM.

    Returns None when `text` is not one of these two shapes at all.
    """
    text = str(text or "").strip()
    m = _SELF_MODE_RE.match(text)
    if m:
        return None, m.group(1).lower()
    m = _NAMED_MODE_RE.match(text)
    if m:
        name = " ".join(m.group(1).split())
        if name:
            return name, m.group(2).lower()
    return None
