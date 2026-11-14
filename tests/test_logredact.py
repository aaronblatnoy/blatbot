import io
import logging

from inkbox_claude import logredact

# Made up, and assembled at run time so no token-shaped text sits in the repository.
TOKEN = "123456789" + ":" + "A" * 35


def _logger(name):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    lg = logging.getLogger(name)
    lg.handlers[:] = [handler]
    lg.propagate = False
    lg.setLevel(logging.INFO)
    logredact.install()
    return lg, stream


def test_token_in_message_arguments_is_masked():
    lg, out = _logger("t.args")
    lg.info('HTTP Request: POST https://api.telegram.org/bot%s/getUpdates "HTTP/1.1 200 OK"', TOKEN)
    text = out.getvalue()
    assert TOKEN not in text and "123456789" not in text
    assert "https://api.telegram.org/bot<redacted>/getUpdates" in text


def test_token_in_preformatted_message_is_masked():
    lg, out = _logger("t.msg")
    lg.info("GET https://api.telegram.org/bot%s/sendMessage" % TOKEN)
    assert TOKEN not in out.getvalue() and "bot<redacted>/sendMessage" in out.getvalue()


def test_token_in_traceback_is_masked():
    lg, out = _logger("t.exc")
    try:
        raise RuntimeError(f"failed https://api.telegram.org/bot{TOKEN}/getMe")
    except RuntimeError:
        lg.exception("boom")
    assert TOKEN not in out.getvalue() and "bot<redacted>/getMe" in out.getvalue()


def test_other_records_are_unchanged():
    lg, out = _logger("t.plain")
    lg.info("robot 42: ready, bot count %d, url https://example.com/bot/status", 3)
    assert out.getvalue() == "INFO t.plain: robot 42: ready, bot count 3, url https://example.com/bot/status\n"


def test_malformed_format_does_not_break_logging():
    lg, out = _logger("t.bad")
    lg.info("needs two %s %s", "only-one")  # logging prints its own error; redaction must not raise
