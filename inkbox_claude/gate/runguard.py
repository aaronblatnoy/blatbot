"""Which request the current asyncio task is running for, and whether Aaron stopped it.

manager.execute() binds the request id for the whole run. Every send and every
write-type tool call made on its behalf asks `is_cancelled()` immediately before it
acts and refuses when the answer is yes. The answer comes from an in-process set
(filled the moment Aaron says stop) and, as a backstop, from the store (a request
in the cancelled state), so a stop survives anything that bypasses this process's
memory.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("gate_request_id", default=None)
_cancelled: set = set()
_sent: Dict[int, List[str]] = {}
_store_check: Optional[Callable[[int], bool]] = None


def bind(request_id: int) -> contextvars.Token:
    return _current.set(int(request_id))


def unbind(token: contextvars.Token) -> None:
    try:
        _current.reset(token)
    except (ValueError, RuntimeError):
        pass


def current() -> Optional[int]:
    return _current.get()


def reset() -> None:
    """Forget in-memory stops and sends (a new manager is a new process's worth of state;
    the store still answers for anything stopped before)."""
    _cancelled.clear()
    _sent.clear()


def set_store_check(fn: Optional[Callable[[int], bool]]) -> None:
    global _store_check
    _store_check = fn


def cancel(request_id: int) -> None:
    _cancelled.add(int(request_id))


def is_cancelled(request_id: Optional[int] = None) -> bool:
    rid = current() if request_id is None else request_id
    if rid is None:
        return False
    if int(rid) in _cancelled:
        return True
    if _store_check is not None:
        try:
            return bool(_store_check(int(rid)))
        except Exception:  # noqa: BLE001  (a store blip must not block or allow by accident)
            logger.debug("[runguard] store check failed for #%s", rid, exc_info=True)
    return False


def refusal(request_id: Optional[int] = None) -> str:
    rid = current() if request_id is None else request_id
    return f"Aaron stopped request #{rid}. Do nothing more. Stop now."


def record_sent(what: str, request_id: Optional[int] = None) -> None:
    rid = current() if request_id is None else request_id
    if rid is None or not what:
        return
    items = _sent.setdefault(int(rid), [])
    if what not in items:
        items.append(what)


def sent(request_id: int) -> List[str]:
    return list(_sent.get(int(request_id), []))
