"""Keep the Telegram bot token out of the log.

The HTTP client library logs every request URL at INFO, and a Telegram Bot API URL carries the
token in its path (``https://api.telegram.org/bot<id>:<secret>/getUpdates``). This filter
rewrites ``bot<digits>:<token chars>`` to ``bot<redacted>`` in a record's message, arguments
and traceback text, whichever library logged it.
"""
from __future__ import annotations

import logging
import re

_TOKEN = re.compile(r"bot[0-9]+:[A-Za-z0-9_-]+")
REDACTED = "bot<redacted>"


def redact(text: str) -> str:
    return _TOKEN.sub(REDACTED, text)


class TokenRedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            if _TOKEN.search(message):
                record.msg = redact(message)
                record.args = ()
            if record.exc_info and not record.exc_text:
                text = logging.Formatter().formatException(record.exc_info)
                if _TOKEN.search(text):
                    record.exc_text = redact(text)
            elif record.exc_text:
                record.exc_text = redact(record.exc_text)
        except Exception:  # noqa: BLE001  never let redaction break logging
            pass
        return True


_FILTER = TokenRedactFilter()


def install() -> None:
    """Put the filter on every handler of the root logger, and on every handler that is
    attached to any named logger, so no output path is missed."""
    loggers = [logging.getLogger()] + [lg for lg in logging.Logger.manager.loggerDict.values()
                                       if isinstance(lg, logging.Logger)]
    for lg in loggers:
        for handler in lg.handlers:
            if _FILTER not in handler.filters:
                handler.addFilter(_FILTER)
