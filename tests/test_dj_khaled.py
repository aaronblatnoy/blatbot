import importlib.util
import os
import sys

os.environ.setdefault("STATE_DIR", "/tmp")
spec = importlib.util.spec_from_file_location("dj_khaled", os.path.join(os.path.dirname(__file__), "..", "deploy", "dj-khaled", "dj_khaled.py"))
dj = importlib.util.module_from_spec(spec)
sys.modules["dj_khaled"] = dj
spec.loader.exec_module(dj)


def msg(frm, to="aaron@migreat.us", cc="amb10503@stern.nyu.edu", subject="Re: hello", irt="<a@x>", ms=2000, labels=("INBOX",), **extra):
    h = [{"name": "From", "value": frm}, {"name": "To", "value": to}, {"name": "Cc", "value": cc}, {"name": "Subject", "value": subject}]
    if irt:
        h.append({"name": "In-Reply-To", "value": irt})
    h += [{"name": k, "value": v} for k, v in extra.items()]
    return {"internalDate": str(ms), "labelIds": list(labels), "payload": {"headers": h}}


def fire(m, thread=None, live=1000):
    return dj.should_fire(m, thread or [m], live)


def test_external_reply_fires():
    assert fire(msg("Sofia <sofia@gmail.com>"))
    assert fire(msg("Aaron <aaronblatnoy@gmail.com>"))


def test_subject_only_reply_fires():
    assert fire(msg("a@acme.com", irt=None))


def test_not_a_reply():
    assert not fire(msg("a@acme.com", subject="hello", irt=None))


def test_no_migreat_on_thread():
    assert not fire(msg("a@acme.com", to="me@other.com"))


def test_migreat_earlier_in_thread_counts():
    first = msg("aaron@migreat.us", to="a@acme.com")
    reply = msg("a@acme.com", to="amb10503@stern.nyu.edu", cc="")
    assert fire(reply, [first, reply])


def test_own_and_internal_senders():
    assert not fire(msg("amb10503@stern.nyu.edu"))
    assert not fire(msg("Aaron <aaronblatnoy@stern.nyu.edu>"))
    assert not fire(msg("aaron@migreat.us"))
    assert not fire(msg("dev@mail.migreat.us"))


def test_sent_label_and_backfill():
    assert not fire(msg("a@acme.com", labels=("SENT",)))
    assert not fire(msg("a@acme.com", ms=999))


def test_bounces_and_autoreplies():
    assert not fire(msg("Mail Delivery <mailer-daemon@googlemail.com>"))
    assert not fire(msg("noreply@acme.com"))
    assert not fire(msg("a@acme.com", **{"Auto-Submitted": "auto-replied"}))
    assert not fire(msg("a@acme.com", Precedence="bulk"))
    assert not fire(msg("a@acme.com", **{"List-Id": "<x.acme.com>"}))
    assert not fire(msg("a@acme.com", **{"Return-Path": "<>"}))
    assert fire(msg("a@acme.com", **{"Auto-Submitted": "no"}))
