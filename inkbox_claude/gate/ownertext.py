"""The one house format for every structured text Blatbot sends Aaron.

A structured text is one Blatbot writes about something (a task finished, a run
needs his yes, someone wrote, a schedule missed), as opposed to a run's own
free-form answer. Every one of them is built here, so they all look the same on
his phone:

    <what this is, in a few words>

    Label: short value
    Label: short value

    The body: the message, the result, the question.

    #472 yes / no

* The first line is the header: a few plain words that say what the text is. No
  symbols or marks of any kind. There are no bracket tags,
  no ALL CAPS words and no "STATUS:" lines.
* Label lines carry who, what, where and when, only when they help.
* The reply handle, when there is one, is the last line on its own, so it can be
  copied or typed straight back ("#472 yes"). The inbound parser reads "#<id> yes",
  "#<id> no", "#<id> edit: ..." and "#<id> <answer>", which is what the handle shows.
* Plain text only: no markdown (iMessage and SMS show asterisks as asterisks), no
  em dashes, no raw ids such as "telegram:8804662817" and no internal names.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple

# The kinds of structured text. Aaron, 2026-11-18: "adding random squares and
# triangles is not what i'm looking for". No kind carries a symbol any more; the
# headline's own words say what the text is.
DONE = FAILED = ASK = HELD = FYI = TIMER = PEOPLE = WORKING = ""

MARKS = {
    "done": DONE, "failed": FAILED, "ask": ASK, "held": HELD, "fyi": FYI,
    "timer": TIMER, "people": PEOPLE, "working": WORKING,
}

SURFACE = {
    "imessage": "iMessage", "sms": "text", "email": "email", "voice": "phone call",
    "telegram": "Telegram", "whatsapp": "WhatsApp", "console": "the console",
}

_GROUP_TAG_RE = re.compile(r"^\s*\[(telegram|imessage|sms|whatsapp) group:([^\]]*)\]\s*", re.I)
_DASH_RE = re.compile(r"[ \t]*[—―][ \t]*")
_EN_DASH_SPACED_RE = re.compile(r"[ \t]+–[ \t]+")
_SPACES_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def clean(text: object) -> str:
    """Text as it may appear in a structured text: no em dashes (a comma instead),
    no trailing spaces, at most one blank line in a row."""
    out = str(text or "").replace("\r\n", "\n").strip()
    if not out:
        return ""
    out = _DASH_RE.sub(", ", out)
    out = _EN_DASH_SPACED_RE.sub(", ", out)
    out = out.replace(", ,", ",").replace(",.", ".")
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    return _BLANKS_RE.sub("\n\n", out).strip()


def one_line(text: object, limit: int = 0) -> str:
    """A value for a label line or a header: one line, single spaced, optionally cut
    at a word boundary with an ellipsis."""
    out = _SPACES_RE.sub(" ", " ".join(clean(text).split()))
    if limit and len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return out


def split_group_tag(text: object) -> Tuple[str, str, str]:
    """A group message as stored for a run starts with a routing note:
    "[telegram group: <title> | from: <name> | your reply goes to ...]". That note is
    for the run, never for Aaron. Returns (group title, sender name, the message alone);
    the first two are "" when the text carries no note."""
    raw = str(text or "")
    m = _GROUP_TAG_RE.match(raw)
    if not m:
        return "", "", raw.strip()
    parts = [p.strip() for p in m.group(2).split("|")]
    title = parts[0] if parts else ""
    sender = ""
    for p in parts[1:]:
        if p.lower().startswith("from:"):
            sender = p.split(":", 1)[1].strip()
    return title, sender, raw[m.end():].strip()


def strip_group_tag(text: object) -> str:
    return split_group_tag(text)[2]


def where(mode: Optional[str], group: str = "") -> str:
    """Where something happened, in words: "Telegram, FellowSIp of the Ring" or "email"."""
    surface = SURFACE.get(str(mode or "").lower(), str(mode or "").strip())
    group = one_line(group)
    if group:
        return f"{surface} group {group}" if surface else f"group {group}"
    return surface


def person(name: Optional[str], handle: Optional[str] = None) -> str:
    """A person by name. A raw provider id ("telegram:8804662817") is never shown;
    an email address or a phone number is, when there is no name at all."""
    name = one_line(name)
    if name:
        return name
    handle = str(handle or "").strip()
    if not handle or re.match(r"^(telegram|tg|imessage|sms|voice):", handle, re.I):
        return "Someone"
    return handle


def reply_handle(req_id: object, *, edit: bool = False, answer: bool = False) -> str:
    """The reply line: what to type back, with the request number."""
    if answer:
        return f"#{req_id} and your answer"
    if edit:
        return f"#{req_id} yes / no / edit: what to change"
    return f"#{req_id} yes / no"


def quote(text: object, limit: int = 600) -> str:
    """A message someone wrote, shown in quotes, cut at a word when it is long."""
    body = clean(text)
    if not body:
        return ""
    if limit and len(body) > limit:
        body = body[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return f"“{body}”"


def bullets(items: Iterable[object]) -> str:
    return "\n".join(one_line(i) for i in items if one_line(i))


def compose(kind: str, headline: str, *, facts: Sequence[Tuple[str, object]] = (),
            body: object = "", handle: str = "", tail: object = "") -> str:
    """Build one structured text in the house format.

    kind     one of MARKS ("done", "failed", "ask", "held", "fyi", "timer", "people",
             "working"). Kept for the callers; it adds nothing to the text.
    headline what this text is, in a few words. One line.
    facts    (label, value) pairs shown as short "Label: value" lines; empty values
             are dropped.
    body     the message itself: a result, a question, a quoted message.
    tail     one closing line (what he can do, or that nothing is needed).
    handle   the reply line, e.g. reply_handle(472); always the last line.
    """
    head = one_line(headline).rstrip(".:")
    blocks = [head]
    fact_lines = []
    for label, value in facts:
        value = one_line(value)
        if value:
            fact_lines.append(f"{label}: {value}")
    if fact_lines:
        blocks.append("\n".join(fact_lines))
    body_text = clean(body)
    if body_text:
        blocks.append(body_text)
    tail_text = clean(tail)
    if tail_text:
        blocks.append(tail_text)
    if handle:
        blocks.append(handle.strip())
    return "\n\n".join(blocks)
