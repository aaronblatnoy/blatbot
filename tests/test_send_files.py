"""send_message can attach documents (a PDF from a link or a path) on every surface."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from inkbox_claude import gateway, telegram
from inkbox_claude.config import BridgeConfig
from inkbox_claude.gate import attachfiles, outcomes

PDF = b"%PDF-1.4 fake"


_REAL = httpx.AsyncClient


def _mock_httpx(monkeypatch, module, handler, calls=None):
    real = _REAL

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)
    monkeypatch.setattr(module.httpx, "AsyncClient", factory)


@pytest.fixture(autouse=True)
def _dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(attachfiles, "FILES_DIR", tmp_path / "att")
    monkeypatch.setenv("BLATBOT_ATTACH_ROOTS", str(tmp_path / "ok"))
    (tmp_path / "ok").mkdir()


def test_url_is_downloaded_to_a_real_file(monkeypatch):
    def handler(req):
        return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
    _mock_httpx(monkeypatch, attachfiles, handler)
    paths = asyncio.run(attachfiles.resolve_files(["https://www.uscis.gov/sites/default/files/document/forms/i-485.pdf"]))
    assert len(paths) == 1 and paths[0].endswith("i-485.pdf")
    assert open(paths[0], "rb").read() == PDF


def test_web_page_missing_and_oversize_urls_are_refused(monkeypatch):
    _mock_httpx(monkeypatch, attachfiles, lambda r: httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"}))
    with pytest.raises(ValueError, match="web page"):
        asyncio.run(attachfiles.resolve_files(["https://x.gov/page"]))
    _mock_httpx(monkeypatch, attachfiles, lambda r: httpx.Response(404))
    with pytest.raises(ValueError, match="404"):
        asyncio.run(attachfiles.resolve_files(["https://x.gov/a.pdf"]))


def test_local_path_only_from_allowed_folders(tmp_path):
    good = tmp_path / "ok" / "form.pdf"
    good.write_bytes(PDF)
    assert asyncio.run(attachfiles.resolve_files([str(good)])) == [str(good.resolve())]
    bad = tmp_path / "secret.env"
    bad.write_text("KEY=1")
    with pytest.raises(ValueError, match="not in a folder"):
        asyncio.run(attachfiles.resolve_files([str(bad)]))
    with pytest.raises(ValueError, match="at most"):
        asyncio.run(attachfiles.resolve_files([str(good)] * 6))


def test_telegram_send_document_posts_the_file(tmp_path, monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["body"] = req.content
        return httpx.Response(200, json={"ok": True, "result": {}})
    monkeypatch.setattr(telegram, "token", lambda: "T")
    _mock_httpx(monkeypatch, telegram, handler)
    f = tmp_path / "ok" / "a.pdf"
    f.write_bytes(PDF)
    asyncio.run(telegram.send_document("-100", str(f)))
    assert seen["url"].endswith("/botT/sendDocument") and PDF in seen["body"] and b"a.pdf" in seen["body"]


def test_telegram_send_document_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(telegram, "token", lambda: "T")
    _mock_httpx(monkeypatch, telegram, lambda r: httpx.Response(400, json={"ok": False, "description": "bad"}))
    f = tmp_path / "ok" / "a.pdf"
    f.write_bytes(PDF)
    with pytest.raises(RuntimeError, match="sendDocument failed"):
        asyncio.run(telegram.send_document("-100", str(f)))


class _Identity:
    def __init__(self):
        self.texts, self.imsgs, self.mails, self.uploads = [], [], [], []

    def upload_imessage_media(self, content, filename, content_type):
        self.uploads.append((filename, content_type))
        return SimpleNamespace(media_url=f"https://media/{filename}")

    def send_text(self, **kw):
        self.texts.append(kw)
        return SimpleNamespace(id="1")

    def send_imessage(self, **kw):
        self.imsgs.append(kw)
        return SimpleNamespace(id="2")

    def send_email(self, **kw):
        self.mails.append(kw)
        return SimpleNamespace(id="3")


def _gw(identity):
    gw = gateway.InkboxGateway(BridgeConfig(require_signature=False, identity="claude"))
    gw._inkbox = SimpleNamespace(get_identity=lambda _i: identity)
    return gw


def _file(tmp_path):
    f = tmp_path / "ok" / "form.pdf"
    f.write_bytes(PDF)
    return str(f)


def test_every_channel_carries_the_file(tmp_path, monkeypatch):
    f = _file(tmp_path)
    ident = _Identity()
    gw = _gw(ident)
    meta = {"_inkbox_tool": True, "attach_files": [f]}
    asyncio.run(gw.send_to_contact("c", "here", "sms", {**meta, "to": "+15551234567"}))
    assert ident.texts[0]["media_urls"] == ["https://media/form.pdf"]
    asyncio.run(gw.send_to_contact("c", "here", "imessage", {**meta, "conversation_id": "9"}))
    assert ident.imsgs[0]["media_urls"] == ["https://media/form.pdf"]
    asyncio.run(gw.send_to_contact("c", "here", "email", {**meta, "to": "a@b.co", "subject": "s"}))
    att = ident.mails[0]["attachments"][0]
    assert att["filename"] == "form.pdf" and att["content_type"] == "application/pdf"
    docs = []

    async def fake_doc(chat, path, caption=""):
        docs.append((chat, path))

    async def fake_msg(chat, text):
        pass
    monkeypatch.setattr(gateway.telegram, "send_document", fake_doc)
    monkeypatch.setattr(gateway.telegram, "send_message", fake_msg)
    asyncio.run(gw.send_to_contact("tg:-100", "here", "telegram", {**meta, "telegram_chat_id": "-100"}))
    assert docs == [("-100", f)]


def test_whatsapp_uploads_with_the_file_type(tmp_path):
    from inkbox_claude import whatsapp
    f = _file(tmp_path)
    seen = {}

    class _I:
        id = "i"

        def upload_whatsapp_media(self, **kw):
            seen.update(kw)
            return SimpleNamespace(id="m1")

        def send_whatsapp(self, **kw):
            seen["send"] = kw
            return SimpleNamespace(id="w")
    gw = SimpleNamespace(_identity=_I())
    asyncio.run(whatsapp.send(gw, "whatsapp:c1", "hi", {"_inkbox_tool": True, "conversation_id": "c1", "attach_files": [f]}))
    assert seen["content_type"] == "application/pdf" and seen["send"]["media_ids"] == ["m1"]


def test_send_message_tool_attaches_resolved_files_and_reports_bad_ones(tmp_path, monkeypatch):
    f = _file(tmp_path)
    sent = {}

    class _Mgr:
        store = SimpleNamespace(task_id_for_request=lambda i: None, responses_to=lambda i: [],
                                add_message=lambda *a, **k: None)

        def _outbox_enabled(self):
            return False

        async def send(self, target, text, mode, meta, **kw):
            sent.update(meta=meta, mode=mode)
            return {"id": "1"}

    monkeypatch.setattr(outcomes, "enabled", lambda: True)
    monkeypatch.setattr(outcomes, "resolve", lambda m, r, to, surface: {
        "meta": {"telegram_chat_id": "-100"}, "mode": "telegram", "target": "tg:-100", "record": "tg:-100",
        "who": "group", "requester": False})
    monkeypatch.setattr(outcomes, "describe", lambda d: "the group")
    req = SimpleNamespace(id=1, schedule_id=None, status={}, inbound_id=None)
    out = asyncio.run(outcomes.send_message({"to": "group", "text": "the forms", "files": [f]}, _Mgr(), req))
    assert out.startswith("Sent") and sent["meta"]["attach_files"] == [str(__import__("pathlib").Path(f).resolve())]
    out = asyncio.run(outcomes.send_message({"to": "group", "text": "x", "files": ["/etc/passwd"]}, _Mgr(), req))
    assert out.startswith("Not sent:")
    assert "files" in outcomes.SEND_MESSAGE_SCHEMA["properties"]
    assert "never say you cannot send files" in outcomes.SEND_MESSAGE_DESCRIPTION
