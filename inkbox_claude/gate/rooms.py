"""What it means to be in a group chat, for any channel.

A room note is a description of a situation, not a rule about when to speak: who is
here, who can hear what is said, and that the assistant is one of the members rather
than a service the others are calling. The gate decides whether to speak; this decides
what the assistant understands about where it is standing while it does.
"""

from typing import Iterable, List, Optional


def _names(members: Optional[Iterable[str]], me: str) -> List[str]:
    """Everyone known to be in the room, the assistant last, without repeats."""
    out: List[str] = []
    for name in (members or []):
        name = " ".join(str(name or "").split())
        if name and name != me and name not in out:
            out.append(name)
    return out


def room_note(*, channel: str, title: str = "", me: str = "", members: Optional[Iterable[str]] = None,
              principal: str = "") -> str:
    """The description a group thread carries into every prompt. Everything is optional:
    with no title or membership it still says the one thing that matters, which is that
    this is a room with other people in it and the assistant is one of them."""
    me = " ".join(str(me or "the assistant").split())
    others = _names(members, me)
    named = f' called "{" ".join(title.split())}"' if title else ""
    who = ""
    if others:
        listed = others[0] if len(others) == 1 else ", ".join(others[:-1]) + " and " + others[-1]
        who = f" The people in it are {listed}, along with you."
    elif title:
        who = " Several people are in it, along with you."
    mine = (f" You are here as {principal}'s assistant, and they are one of the people in the room."
            if principal else "")
    return (
        "--- WHERE YOU ARE ---\n"
        f"This is a group conversation on {channel or 'a messaging channel'}{named}.{who} "
        f"You, {me}, are one of its members, not an outside service the others are calling.{mine}\n"
        "Everything you say here is read by everyone in the room, and you see everything they say "
        "to each other, including messages that were never meant for you.\n"
        "Because you are a member, a question about the room includes you: who here can do "
        "something, whether anyone has access to a thing, whether someone could check it. Answer "
        "for yourself first and say plainly what you have and what you do not, before saying what "
        "you cannot see about the others.\n"
        "Every message says who sent it. Different people want different things, and a request "
        "from one of them does not become another's. Anything private to the person you work for "
        "is private to them, and in here anything you say is said to all of them at once."
    )
