"""The assistant's signature belongs on email only.

Texts, iMessage, WhatsApp and Telegram messages never carry the name-and-title
sign-off. ``strip_signoff`` removes it from the end of a message; the gateway
calls it for every send that is not an email, so a fixed notice or a model reply
that ends with the signature goes out without it.
"""

from __future__ import annotations

import os
import re

_TITLE = "Executive Assistant to Aaron Blatnoy"


def _pattern() -> "re.Pattern[str]":
    name = re.escape((os.getenv("ASSISTANT_NAME") or "Blatbot").strip() or "Blatbot")
    return re.compile(
        r"(?:\n|^)[ \t]*(?:(?:best|thanks|thank you|regards|best regards|sincerely|cheers)[,!.]?[ \t]*\n[ \t]*)?"
        r"(?:[-\u2013\u2014~][ \t]*)?(?:" + name + r"[ \t]*[,/|\n-]?[ \t]*)?"
        + re.escape(_TITLE) + r"\.?\s*$",
        re.IGNORECASE,
    )


def strip_signoff(text: str) -> str:
    """``text`` without a trailing name-and-title signature. Anything else is untouched."""
    if not text or _TITLE.lower() not in text.lower():
        return text
    out = _pattern().sub("", text).rstrip()
    return out if out else text
