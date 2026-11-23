"""How a run ends, and where its messages go.

A run ends one of three ways: it sends a message (send_message), it decides nothing
needs to be sent (nothing_to_send), or it stops to ask Aaron (ask_aaron, or any other
call the permission hook defers). This module holds the tools for the first two and
the answer side of the third, plus the one place that decides which surface a message
goes out on: iMessage, text message and email through Inkbox, or Telegram through the
bot. Every send still goes through the permission hook and the outbox.
"""

from __future__ import annotations

from inkbox_claude.instance import ASSISTANT as _ASSISTANT, ASSISTANT_UPPER as _ASSISTANT_UPPER, ASSISTANT_EMAIL as _ASSISTANT_EMAIL, SERVER_NAME as _SERVER_NAME  # noqa: F401
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SEND_TOOL = "mcp__host__send_message"
QUIET_TOOL = "mcp__host__nothing_to_send"
REACH_TOOL = "mcp__host__where_to_reach"
ASK_TOOL = "mcp__host__ask_aaron"
OUTCOME_TOOLS = (SEND_TOOL, QUIET_TOOL, REACH_TOOL)
ALL_TOOLS = OUTCOME_TOOLS + (ASK_TOOL,)

# The order a surface is picked in when nothing says which one a person uses.
# WhatsApp is last: its line cannot open a conversation with a new contact, so it is
# used when asked for, or to answer a WhatsApp conversation that already exists.
SURFACE_ORDER = ("imessage", "telegram", "sms", "email", "whatsapp")
SURFACE_WORDS = {"imessage": "iMessage", "telegram": "Telegram", "sms": "text message", "email": "email",
                 "whatsapp": "WhatsApp"}
_SURFACE_ALIASES = {
    "imessage": "imessage", "i message": "imessage", "apple": "imessage", "blue": "imessage",
    "telegram": "telegram", "tg": "telegram",
    "sms": "sms", "text": "sms", "text message": "sms", "texts": "sms", "mms": "sms", "phone": "sms",
    "email": "email", "e-mail": "email", "mail": "email",
    "whatsapp": "whatsapp", "whats app": "whatsapp", "wa": "whatsapp",
}
_OWNER_WORDS = {"aaron", "aaron blatnoy", "me", "owner", "the owner", "aaron b"}
_REQUESTER_WORDS = {"", "requester", "the requester", "sender", "the sender", "them",
                    "the person who asked", "whoever asked", "reply", "this chat", "here"}

ASK_LIMIT = 2


def enabled() -> bool:
    from .settings import get
    try:
        return bool(get("GATE_RUN_OUTCOMES"))
    except Exception:
        return False


def ask_enabled() -> bool:
    from .settings import get
    try:
        return bool(get("GATE_ASK_AARON"))
    except Exception:
        return False


SEND_MESSAGE_DESCRIPTION = (
    f"SEND A MESSAGE. This is how a run delivers anything to anyone, on any surface {_ASSISTANT} has: "
    "iMessage, text message, email, Telegram or WhatsApp. to is \"requester\" (the person who asked, answered on "
    "the surface they wrote from: this is how you give them your answer), \"aaron\", a person's name, or a "
    "group chat's name. surface is optional (imessage, telegram, sms, email, whatsapp): without it the surface they "
    f"last used with {_ASSISTANT} is picked. subject is for email only. The answer says where it went. "
    f"Posting in a Telegram group {_ASSISTANT} is in is done here too: to is the group's name. It needs no extra "
    "access; the telegram_* tools only read Aaron's personal account and have nothing to do with sending."
)
SEND_MESSAGE_SCHEMA = {"type": "object", "required": ["to", "text"], "properties": {
    "to": {"type": "string", "description": "\"requester\", \"aaron\", a person's name, or a group chat's name"},
    "text": {"type": "string", "description": "the whole message, plain text"},
    "surface": {"type": "string", "description": "optional: imessage, telegram, sms, email or whatsapp"},
    "subject": {"type": "string", "description": "email subject, only for email"},
    "images": {"type": "array", "items": {"type": "string"},
               "description": "PNG paths returned by render_image to attach (up to 4), on any surface"}}}
NOTHING_TO_SEND_DESCRIPTION = (
    "END WITHOUT SENDING ANYTHING. Call this when this run has nothing worth telling anyone (for example a "
    "check found nothing new). It is recorded on the task with your reason and nothing is sent. Then stop."
)
NOTHING_TO_SEND_SCHEMA = {"type": "object", "required": ["reason"], "properties": {
    "reason": {"type": "string", "description": "one plain sentence: why nothing needs to be sent"}}}
WHERE_TO_REACH_DESCRIPTION = (
    "WHERE CAN SOMEONE BE REACHED: lists the surfaces on record for a person or group chat (iMessage, text "
    "message, email, Telegram, WhatsApp) and which one send_message would use. Read-only."
)
WHERE_TO_REACH_SCHEMA = {"type": "object", "required": ["person"], "properties": {
    "person": {"type": "string", "description": "\"requester\", \"aaron\", a person's name or a group chat's name"}}}
ASK_AARON_DESCRIPTION = (
    "ASK AARON. When an essential detail is missing, or a decision only Aaron can make is needed, ask him "
    "here instead of stopping with \"missing X\". kind \"input\" asks for a detail or a choice; kind "
    "\"permission\" asks a yes or no. The run pauses and continues with his answer as this tool's result. "
    "For access to a system you were not given, use request_access instead. At most two asks per task."
)
ASK_AARON_SCHEMA = {"type": "object", "required": ["question", "kind"], "properties": {
    "question": {"type": "string", "description": "the question, in one or two plain sentences"},
    "kind": {"type": "string", "enum": ["permission", "input"]}}}


def surface_of(word: Any) -> Optional[str]:
    """A surface name from whatever words a run used, or None for none given."""
    w = " ".join(str(word or "").lower().replace("_", " ").split())
    if not w:
        return None
    if w in _SURFACE_ALIASES:
        return _SURFACE_ALIASES[w]
    raise ValueError(f'"{word}" is not a surface. Use imessage, telegram, sms, email or whatsapp.')


def _e164(raw: str) -> str:
    digits = re.sub(r"\D", "", str(raw or ""))
    if len(digits) == 10:
        return "+1" + digits
    return "+" + digits if digits else ""


def _owner_request(manager: Any, req: Any) -> bool:
    try:
        return bool(manager._is_owner_request(req))
    except Exception:
        return False


def _is_group(meta: Dict[str, Any]) -> bool:
    return (meta or {}).get("conversation_kind") == "group" or bool((meta or {}).get("group_title"))


def _dest(target: str, mode: str, meta: Dict[str, Any], record: str, who: str, *,
          requester: bool = False, owner: bool = False, group: bool = False) -> Dict[str, Any]:
    return {"target": target, "mode": mode, "meta": dict(meta or {}), "record": record or target,
            "who": who, "requester": requester, "owner": owner, "group": group}


def _route_dest(chat_id: str, mode: str, meta: Dict[str, Any], who: str, record: str = "",
                **flags: Any) -> Dict[str, Any]:
    """A destination that replies into a conversation Blatbot already has."""
    meta = dict(meta or {})
    if mode == "telegram":
        tid = str(meta.get("telegram_chat_id") or meta.get("to") or chat_id.removeprefix("tg:"))
        meta = {"telegram_chat_id": tid}
        chat_id = chat_id if chat_id.startswith("tg:") else f"tg:{tid}"
    return _dest(chat_id, mode, meta, record or chat_id, who, **flags)


# -- who is "aaron" ------------------------------------------------------------

def _owner_surfaces(manager: Any) -> Dict[str, Dict[str, Any]]:
    from .manager import OWNER_THREAD
    out: Dict[str, Dict[str, Any]] = {}
    conv = str(getattr(manager, "approver_conv", "") or "")
    if conv:
        out["imessage"] = _dest(f"imessage:{conv}", "imessage", {"conversation_id": conv}, OWNER_THREAD,
                                "Aaron", owner=True)
    tg = str(getattr(manager, "approver_telegram_id", "") or "")
    if tg:
        out["telegram"] = _dest(f"tg:{tg}", "telegram", {"telegram_chat_id": tg}, OWNER_THREAD,
                                "Aaron", owner=True)
    phone = _e164(getattr(manager, "approver_phone", "") or "")
    if phone:
        out["sms"] = _dest(f"sms:{phone}", "sms", {"to": phone}, OWNER_THREAD, "Aaron", owner=True)
        out["whatsapp"] = _dest(f"whatsapp:{phone}", "whatsapp", {"whatsapp_to": phone}, OWNER_THREAD,
                                "Aaron", owner=True)
    try:
        pid = manager.store.commander_person_id()
        person = manager.store.get_person(pid) if pid else None
        emails = [c for c in (person or {}).get("contacts") or [] if c.get("kind") == "email"]
        if emails:
            addr = str(emails[0].get("raw_value") or emails[0].get("value"))
            out["email"] = _dest(f"email:{addr}", "email", {"to": addr, "_inkbox_tool": True},
                                 OWNER_THREAD, "Aaron", owner=True)
    except Exception as exc:
        logger.debug("[outcomes] no email on record for Aaron: %s", exc)
    return out


def _owner_recent(manager: Any) -> Optional[str]:
    """The surface Aaron last used with Blatbot."""
    try:
        for row in manager.store.thread_routes(manager.store.owner_threads()):
            if row["mode"] in SURFACE_ORDER and not _is_group(row["meta"]):
                return row["mode"]
    except Exception as exc:
        logger.debug("[outcomes] could not read Aaron's recent surface: %s", exc)
    return None


def _to_owner(manager: Any, req: Any, surface: Optional[str]) -> Dict[str, Any]:
    options = _owner_surfaces(manager)
    if surface:
        if surface not in options:
            raise ValueError(f"Aaron has no {SURFACE_WORDS[surface]} on record. "
                             f"He can be reached on: {_words(options)}.")
        return options[surface]
    # The conversation he is in right now, when this run is answering him.
    if req is not None and _owner_request(manager, req) and req.mode in options and not req.schedule_id:
        return options[req.mode]
    recent = _owner_recent(manager)
    if recent in options:
        return options[recent]
    for s in SURFACE_ORDER:
        if s in options:
            return options[s]
    raise ValueError("Aaron has no surface on record.")


def _words(options: Dict[str, Any]) -> str:
    names = [SURFACE_WORDS[s] for s in SURFACE_ORDER if s in options]
    return ", ".join(names) if names else "nothing"


# -- the person who asked --------------------------------------------------------

def _schedule_origin(manager: Any, req: Any) -> Optional[Dict[str, Any]]:
    try:
        sched = manager.store.get_schedule(req.schedule_id)
    except Exception:
        sched = None
    if sched is None:
        return None
    mode = str(getattr(sched, "origin_mode", "") or "")
    raw = str(getattr(sched, "origin_chat_id", "") or "")
    if not mode or mode == "console" or not raw or mode not in SURFACE_ORDER:
        return None
    target, meta = raw, {}
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            target = str(parsed.get("chat_id") or "")
            meta = dict(parsed.get("meta") or {})
        except Exception:
            return None
    if not target:
        return None
    title = " ".join(str(meta.get("group_title") or "").split())
    who = f'the "{title}" group' if title else "Aaron"
    from .manager import OWNER_THREAD
    group = _is_group(meta)
    return _route_dest(target, mode, meta, who, record=target if group else OWNER_THREAD,
                       requester=True, owner=not group, group=group)


def _to_requester(manager: Any, req: Any, surface: Optional[str]) -> Dict[str, Any]:
    if req is None:
        raise ValueError("This run has no one who asked; name who the message is for.")
    if req.schedule_id:
        origin = _schedule_origin(manager, req)
        if origin is not None and (surface is None or surface == origin["mode"]):
            return origin
        dest = _to_owner(manager, req, surface)
        dest["requester"] = True
        return dest
    owner = _owner_request(manager, req)
    route = None
    try:
        route = manager.store.thread_route(req.chat_id)
    except Exception:
        route = None
    meta = dict((route or {}).get("meta") or {})
    mode = req.mode or (route or {}).get("mode") or ""
    try:
        session = manager.get(req.chat_id)
        if session.mode == mode and session.reply_meta:
            meta.update(session.reply_meta)
    except Exception:
        pass
    group = _is_group(meta)
    from .manager import OWNER_THREAD
    if owner and not group and (mode not in SURFACE_ORDER or (surface and surface != mode)
                                or req.chat_id == OWNER_THREAD):
        # Aaron asked on a call, or wants it somewhere else: his own surfaces.
        dest = _to_owner(manager, req, surface)
        dest["requester"] = True
        return dest
    if mode == "voice" or (surface and surface != mode):
        if group:
            raise ValueError("The person asked in a group chat; answer there without naming a surface.")
        person = _person_for_handles(manager, [req.sender])
        if person is not None:
            dest = _to_person(manager, req, person, surface or ("sms" if mode == "voice" else None))
            dest["requester"] = True
            return dest
        number = _e164(req.sender) if len(re.sub(r"\D", "", req.sender or "")) >= 7 else ""
        if (surface or "sms") == "sms" and number:
            return _dest(f"sms:{number}", "sms", {"to": number, "sender": number}, req.chat_id,
                         req.sender_name or "the person who asked", requester=True)
        raise ValueError("The person who asked can only be answered where they wrote from.")
    from .manager import OWNER_THREAD
    record = OWNER_THREAD if (owner and not group) else req.chat_id
    title = " ".join(str(meta.get("group_title") or "").split())
    who = (f'the "{title}" group' if group and title else
           "Aaron" if owner else (req.sender_name or "the person who asked"))
    return _route_dest(req.chat_id, mode, meta, who, record=record, requester=True,
                       owner=owner and not group, group=group)


# -- anyone else -----------------------------------------------------------------

def _person_for_handles(manager: Any, handles: List[str]) -> Optional[Dict[str, Any]]:
    try:
        found = manager.store.resolve_handles([h for h in handles if h])
        if found.get("person_id") and not found.get("conflict"):
            return manager.store.get_person(found["person_id"])
    except Exception:
        pass
    return None


def _find_person(manager: Any, name: str) -> Optional[Dict[str, Any]]:
    store = manager.store
    kind = store.guess_contact_kind(name)
    if kind in ("email", "phone", "telegram"):
        try:
            contact = store.find_contact(kind, name)
            if contact and contact.get("person_id"):
                return store.get_person(contact["person_id"])
        except Exception:
            pass
        return None
    needle = " ".join(name.lower().split())
    people = [p for p in store.list_people() if str(p.get("display") or "").strip()]
    exact = [p for p in people if " ".join(str(p["display"]).lower().split()) == needle]
    if len(exact) == 1:
        return exact[0]
    first = [p for p in people if str(p["display"]).lower().split()[:1] == needle.split()[:1]
             and (len(needle.split()) == 1 or needle in str(p["display"]).lower())]
    if len(first) == 1:
        return first[0]
    contains = [p for p in people if needle and needle in str(p["display"]).lower()]
    if len(contains) == 1:
        return contains[0]
    if len(exact) + len(first) + len(contains) > 1:
        names = sorted({str(p["display"]) for p in exact + first + contains})
        raise ValueError(f'"{name}" matches more than one person: {", ".join(names)}. Use the full name.')
    return None


def _person_options(manager: Any, person: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every surface this person can be sent to, built from their contacts."""
    who = str(person.get("display") or "them")
    out: Dict[str, Dict[str, Any]] = {}
    for c in person.get("contacts") or []:
        kind, raw = c.get("kind"), str(c.get("raw_value") or c.get("value") or "")
        if kind == "phone" and _e164(raw):
            phone = _e164(raw)
            out.setdefault("imessage", _dest(f"imessage:{phone}", "imessage", {"imessage_to": phone},
                                             f"imessage:{phone}", who))
            out.setdefault("sms", _dest(f"sms:{phone}", "sms", {"to": phone}, f"sms:{phone}", who))
            out.setdefault("whatsapp", _dest(f"whatsapp:{phone}", "whatsapp", {"whatsapp_to": phone},
                                             f"whatsapp:{phone}", who))
        elif kind == "telegram" and raw:
            tid = raw.split(":", 1)[1] if raw.lower().startswith("telegram:") else raw
            out.setdefault("telegram", _dest(f"tg:{tid}", "telegram", {"telegram_chat_id": tid},
                                             f"tg:{tid}", who))
        elif kind == "email" and "@" in raw:
            out.setdefault("email", _dest(f"email:{raw}", "email", {"to": raw, "_inkbox_tool": True},
                                          f"email:{raw}", who))
    return out


def _person_recent(manager: Any, person: Dict[str, Any]) -> List[Dict[str, Any]]:
    """This person's own conversations with Blatbot, most recent first."""
    try:
        chats = manager.store.memory_chats_for_person(person["id"])
        return [r for r in manager.store.thread_routes(chats)
                if r["mode"] in SURFACE_ORDER and not _is_group(r["meta"])]
    except Exception as exc:
        logger.debug("[outcomes] could not read recent chats for %s: %s", person.get("id"), exc)
        return []


def _to_person(manager: Any, req: Any, person: Dict[str, Any], surface: Optional[str]) -> Dict[str, Any]:
    who = str(person.get("display") or "them")
    try:
        if person.get("id") and person["id"] == manager.store.commander_person_id():
            return _to_owner(manager, req, surface)
    except Exception:
        pass
    recent = _person_recent(manager, person)
    options = _person_options(manager, person)
    for row in recent:
        # A conversation they already have with Blatbot is replied into as it is.
        if surface is None or row["mode"] == surface:
            if row["mode"] == "imessage" and not row["meta"].get("conversation_id") and surface is None:
                continue
            return _route_dest(row["chat_id"], row["mode"], row["meta"], who)
    if surface:
        if surface not in options:
            raise ValueError(f"{who} has no {SURFACE_WORDS[surface]} on record. "
                             f"They can be reached on: {_words(options)}.")
        return options[surface]
    for s in SURFACE_ORDER:
        if s in options:
            return options[s]
    raise ValueError(f"There is no way to reach {who} on record.")


def _asked_in_group(manager: Any, req: Any) -> Optional[Dict[str, Any]]:
    """The group chat a one-off request was asked in, or None when it was not asked in one."""
    if req is None or getattr(req, "schedule_id", None):
        return None
    try:
        mine = _to_requester(manager, req, None)
    except Exception:
        return None
    return mine if mine.get("group") else None


def resolve(manager: Any, req: Any, to: Any, surface: Any = None) -> Dict[str, Any]:
    """Where one message goes. Raises ValueError with a plain sentence when it cannot."""
    if manager is None:
        raise ValueError("Sending is unavailable for this run.")
    want = surface_of(surface)
    name = " ".join(str(to or "").split())
    low = name.lower()
    if low in _REQUESTER_WORDS:
        dest = _to_requester(manager, req, want)
    elif low in _OWNER_WORDS:
        dest = _to_owner(manager, req, want)
    else:
        group = None
        try:
            group = manager.store.find_group_chat_by_name(name)
        except Exception:
            group = None
        if group is not None:
            mode = str(group.get("channel") or "")
            if want and want != mode:
                raise ValueError(f'The "{group["title"]}" group is on {SURFACE_WORDS.get(mode, mode)}; '
                                 "leave surface out to post there.")
            route = manager.store.thread_route(group["chat_id"]) or {}
            dest = _route_dest(group["chat_id"], mode, route.get("meta") or {},
                               f'the "{group["title"]}" group', group=True)
            # Asked in a group chat, the run answers in that group chat. Which group a
            # message lands in is never the model's pick: a different group named here is
            # replaced by the one the request came from (2026-11-18: an answer went to
            # another group). From Aaron's own chat a named group is still honoured.
            home = _asked_in_group(manager, req)
            if home is not None and home["target"] != dest["target"]:
                logger.info("[outcomes] #%s named %s; sent where it was asked, %s",
                            getattr(req, "id", "?"), dest["who"], home["who"])
                dest = home
        else:
            person = _find_person(manager, name)
            if person is None:
                kind = manager.store.guess_contact_kind(name)
                if kind == "email":
                    dest = _dest(f"email:{name}", "email", {"to": name, "_inkbox_tool": True},
                                 f"email:{name}", name)
                elif kind == "phone" and _e164(name):
                    phone = _e164(name)
                    dest = (_dest(f"sms:{phone}", "sms", {"to": phone}, f"sms:{phone}", name)
                            if want == "sms" else
                            _dest(f"whatsapp:{phone}", "whatsapp", {"whatsapp_to": phone},
                                  f"whatsapp:{phone}", name)
                            if want == "whatsapp" else
                            _dest(f"imessage:{phone}", "imessage", {"imessage_to": phone},
                                  f"imessage:{phone}", name))
                else:
                    try:
                        titles = [g["title"] for g in manager.store.known_group_chats()][:12]
                    except Exception:
                        titles = []
                    known = (f" Group chats {_ASSISTANT} can post in: " + "; ".join(titles) + "."
                             if titles else "")
                    raise ValueError(f'No person or group chat called "{name}" is on record.{known} '
                                     "Use one of those names exactly, or a person's name.")
                if want and dest["mode"] != want and not (want == "email" and kind == "email"):
                    raise ValueError(f"{name} can only be reached on {SURFACE_WORDS[dest['mode']]}.")
            else:
                dest = _to_person(manager, req, person, want)
    if dest["mode"] not in SURFACE_ORDER:
        raise ValueError(f"{dest['who']} cannot be sent a message on {dest['mode'] or 'that surface'}.")
    # The person who asked is answered however they are named.
    if req is not None and not dest["requester"]:
        try:
            mine = _to_requester(manager, req, None)
            if mine["target"] == dest["target"] or (mine.get("owner") and dest.get("owner")
                                                     and not mine.get("group")):
                dest["requester"] = True
        except Exception:
            pass
    return dest


def describe(dest: Dict[str, Any]) -> str:
    return f"{dest['who']} on {SURFACE_WORDS.get(dest['mode'], dest['mode'])}"


# -- the tools ---------------------------------------------------------------------

def _task_id(manager: Any, req: Any) -> Optional[int]:
    try:
        return manager.store.task_id_for_request(req.id) if req is not None else None
    except Exception:
        return None


async def send_message(args: Dict[str, Any], manager: Any, request: Any,
                       log: Optional[Dict[str, Any]] = None) -> str:
    if not enabled() or manager is None or request is None:
        return "Sending is unavailable for this run. Put the message in your final status."
    text = str(args.get("text") or "").strip()
    if not text and not args.get("images"):
        return "Give the text of the message."
    try:
        dest = resolve(manager, request, args.get("to"), args.get("surface"))
    except ValueError as exc:
        return str(exc)
    meta = dict(dest["meta"])
    if args.get("images"):
        from . import imagetool
        try:
            meta["generated_images"] = imagetool.allowed_image_paths(args.get("images"))
        except ValueError as exc:
            return str(exc)
    subject = str(args.get("subject") or "").strip()
    if dest["mode"] == "email":
        if subject:
            meta["subject"] = subject
            meta["_inkbox_tool"] = True
        elif meta.get("_inkbox_tool") and not meta.get("subject"):
            meta["subject"] = f"A message from {_ASSISTANT}"
    kind = ("schedule" if request.schedule_id else
            "proactive" if (request.status or {}).get("followup_id") else "reply")
    try:
        result = await manager.send(dest["target"], text, dest["mode"], meta, chat_id=dest["record"],
                                    record=False, kind=kind)
    except Exception as exc:
        logger.warning("[outcomes] #%s send to %s failed: %s", request.id, describe(dest), exc)
        return (f"Could not send to {describe(dest)}: {exc}. Try another surface, "
                "or say plainly in your final status that it could not be sent.")
    from .manager import HELD_FOR_AARON
    if result is HELD_FOR_AARON:
        # Not sent: shown to Aaron word for word with its own "#id yes / no". Nothing is
        # recorded as said until he says yes.
        try:
            tid = _task_id(manager, request)
            if tid:
                manager.store.task_event(tid, "note", f"Held for Aaron's yes: to {describe(dest)}: {text}",
                                         chat_id=dest["record"], request_id=request.id)
        except Exception as exc:
            logger.warning("[outcomes] #%s could not record a held send: %s", request.id, exc)
        if log is not None:
            log.setdefault("held_sends", []).append({"to": dest["who"], "surface": dest["mode"],
                                                     "target": dest["target"], "text": text[:600]})
        logger.info("[outcomes] #%s held a message to %s for Aaron's yes", request.id, describe(dest))
        return ("Held for Aaron's yes. He has been shown this exact message and it goes out when he says "
                "yes. This is expected, not an error: do not retry it or try another route. If there is "
                "another message to send, send it the same way; otherwise stop now.")
    held = result is None and manager._outbox_enabled() and dest["mode"] != "voice"
    answered = bool(request.inbound_id) and "answer" in manager.store.responses_to(request.inbound_id)
    role = "answer" if (dest["requester"] and request.inbound_id and not answered
                        and not request.schedule_id) else "notice"
    try:
        manager.store.add_message(dest["record"], "outbound", text, dest["mode"],
                                  reply_to=request.inbound_id if dest["requester"] else None, role=role)
        tid = _task_id(manager, request)
        if tid:
            manager.store.task_event(tid, "outbound", f"To {describe(dest)}: {text}",
                                     chat_id=dest["record"], request_id=request.id)
    except Exception as exc:
        logger.warning("[outcomes] #%s could not record a send: %s", request.id, exc)
    if log is not None:
        log.setdefault("sends", []).append({"to": dest["who"], "surface": dest["mode"],
                                            "requester": bool(dest["requester"]),
                                            "target": dest["target"], "text": text[:600]})
    logger.info("[outcomes] #%s sent to %s%s", request.id, describe(dest),
                " (the person who asked)" if dest["requester"] else "")
    if held:
        return f"Queued for {describe(dest)}; it goes out shortly with the next messages."
    return f"Sent to {describe(dest)}."


def nothing_to_send(args: Dict[str, Any], manager: Any, request: Any,
                    log: Optional[Dict[str, Any]] = None) -> str:
    if not enabled():
        return "Put your result in your final status."
    reason = " ".join(str(args.get("reason") or "").split()) or "nothing new to report"
    if log is not None:
        log["quiet"] = reason
    try:
        tid = _task_id(manager, request)
        if tid:
            manager.store.task_event(tid, "note", f"Decided nothing needed sending: {reason}",
                                     chat_id=getattr(request, "chat_id", ""), request_id=request.id)
    except Exception as exc:
        logger.warning("[outcomes] could not record a quiet run: %s", exc)
    logger.info("[outcomes] #%s ended quietly: %s", getattr(request, "id", "?"), reason)
    return "Noted. Nothing will be sent. Stop now."


def where_to_reach(args: Dict[str, Any], manager: Any, request: Any) -> str:
    name = " ".join(str(args.get("person") or "").split())
    try:
        chosen = resolve(manager, request, name, None)
    except ValueError as exc:
        return str(exc)
    low = name.lower()
    options: Dict[str, Any] = {}
    recent: List[str] = []
    try:
        if chosen.get("owner") or low in _OWNER_WORDS:
            options = _owner_surfaces(manager)
            r = _owner_recent(manager)
            recent = [r] if r else []
        elif chosen.get("group"):
            options = {chosen["mode"]: chosen}
        elif low not in _REQUESTER_WORDS:
            person = _find_person(manager, name)
            if person is not None:
                options = _person_options(manager, person)
                recent = [row["mode"] for row in _person_recent(manager, person)]
    except ValueError:
        pass
    if chosen["mode"] not in options:
        options[chosen["mode"]] = chosen
    listed = ", ".join(SURFACE_WORDS[s] + (" (used most recently)" if recent and s == recent[0] else "")
                       for s in SURFACE_ORDER if s in options)
    return (f"{chosen['who']} can be reached on: {listed}. "
            f"send_message without a surface would use {SURFACE_WORDS[chosen['mode']]}.")


def ask_answer(manager: Any, request: Any) -> str:
    """The result of ask_aaron once he has answered (the hook only lets it run then)."""
    try:
        fresh = manager.store.get_request(request.id) if manager is not None and request is not None else None
        status = (fresh.status or {}) if fresh is not None else {}
        decided = status.get("deferred_decision") or {}
        deferred = status.get("deferred") or {}
        if deferred.get("kind") == "ask" and decided.get("id") == deferred.get("id") and decided.get("answer"):
            return f"Aaron said: {decided['answer']}"
    except Exception as exc:
        logger.warning("[outcomes] could not read Aaron's answer: %s", exc)
    return "Asking Aaron is unavailable for this run. Finish with what you have and say plainly what is missing."


def ask_text(req: Any, question: str, kind: str, owner: bool) -> str:
    """The text Aaron gets for one ask, in the house format (ownertext.py)."""
    from . import ownertext
    q = " ".join(str(question or "").replace("\u2014", ", ").split())
    if q and q[-1] not in ".?!":
        q += "."
    if owner:
        head = f"Your task has a question (#{req.id})"
        facts = [("Task", ownertext.strip_group_tag(getattr(req, "summary", "") or ""))]
    else:
        group, tagged_from, _ = ownertext.split_group_tag(getattr(req, "original_message", "") or "")
        who = ownertext.person(getattr(req, "sender_name", "") or tagged_from, getattr(req, "sender", ""))
        head = f"{who}'s task has a question (#{req.id})"
        facts = [("Task", ownertext.strip_group_tag(getattr(req, "summary", "") or "")),
                 ("Where", ownertext.where(getattr(req, "mode", ""), group))]
    handle = ownertext.reply_handle(req.id, answer=kind != "permission")
    return ownertext.compose("ask", head, facts=facts, body=q, handle=handle)
