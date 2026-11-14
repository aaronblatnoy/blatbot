"""Formatted HTML pages: stored, served, expired, sanitised, owner-only."""

import asyncio
import re
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from inkbox_claude.gate import pages, settings


@pytest.fixture(autouse=True)
def page_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_PAGES_DIR", str(tmp_path / "pages"))
    monkeypatch.delenv("GATE_HTML_PAGES_AUDIENCE", raising=False)
    monkeypatch.delenv("GATE_HTML_PAGE_DAYS", raising=False)
    settings.invalidate()
    pages.set_public_base("https://blatbot.example.test")
    yield
    pages.set_public_base("")
    settings.invalidate()


def run(coro):
    return asyncio.run(coro)


async def _client():
    app = web.Application()
    pages.register(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_page_is_stored_and_served_with_security_headers():
    info = pages.store_page("Q3 report", "<h2>Totals</h2><table><tr><th>A</th><td>1</td></tr></table>")
    assert info["url"].startswith("https://blatbot.example.test/p/")
    assert len(info["token"]) >= 32

    async def go():
        c = await _client()
        try:
            r = await c.get(info["path"])
            body = await r.text()
            return r.status, dict(r.headers), body
        finally:
            await c.close()

    status, headers, body = run(go())
    assert status == 200
    assert "<title>Q3 report</title>" in body
    assert "<h2>Totals</h2>" in body and "<table>" in body
    csp = headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "script-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert "noindex" in headers["X-Robots-Tag"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Content-Type"].startswith("text/html")
    # self-contained: no external request of any kind
    assert not re.search(r"(src|href)=\"https?://", body.split("<main>")[0])
    assert "<script" not in body and "@import" not in body and "url(" not in body


def test_wrong_and_malformed_paths_are_404_and_nothing_lists_the_directory():
    pages.store_page("x", "<p>hello</p>")

    async def go():
        c = await _client()
        try:
            out = []
            for path in ("/p/" + "a" * 43, "/p/short", "/p/..%2f..%2fetc%2fpasswd", "/p/", "/p"):
                out.append((await c.get(path)).status)
            return out
        finally:
            await c.close()

    assert run(go()) == [404, 404, 404, 404, 404]


def test_expired_page_is_404_and_deleted():
    t0 = time.time()
    info = pages.store_page("old", "<p>hello</p>", now=t0 - 40 * 86400, ttl_days=30)
    assert pages.load_page(info["token"], now=t0 - 11 * 86400) is not None
    assert pages.load_page(info["token"], now=t0) is None
    assert not (pages.pages_dir() / f"{info['token']}.html").exists()


def test_expiry_default_comes_from_the_setting(monkeypatch):
    monkeypatch.setenv("GATE_HTML_PAGE_DAYS", "7")
    settings.invalidate()
    now = 1_000_000.0
    info = pages.store_page("t", "<p>x</p>", now=now)
    assert info["expires"] == now + 7 * 86400


def test_purge_removes_only_expired():
    now = time.time()
    old = pages.store_page("old", "<p>a</p>", now=now - 100 * 86400, ttl_days=1)
    new = pages.store_page("new", "<p>b</p>", now=now)
    assert not (pages.pages_dir() / f"{old['token']}.html").exists()
    assert (pages.pages_dir() / f"{new['token']}.html").exists()


def test_sanitiser_strips_script_handlers_iframes_forms_and_js_links():
    dirty = (
        '<h2 onclick="steal()">Title</h2>'
        '<script>alert(1)</script>'
        '<p onmouseover="x()" style="position:fixed">ok <b>bold</b></p>'
        '<a href="javascript:alert(1)">bad</a>'
        '<a href="  JaVa\tScRiPt:alert(2)">bad2</a>'
        '<a href="data:text/html;base64,AAAA">bad3</a>'
        '<a href="https://example.com/x" onclick="y()">good</a>'
        '<iframe src="https://evil.example"></iframe>'
        '<form action="https://evil.example"><input name="pw"><button>go</button></form>'
        '<img src="https://evil.example/p.png" onerror="z()">'
        '<style>body{background:url(https://evil.example)}</style>'
        '<svg><script>1</script></svg>'
        '<object data="x"></object><embed src="x">'
        '<!-- comment --><table><tr><td>cell</td></tr></table>'
    )
    clean = pages.sanitize_html(dirty)
    low = clean.lower()
    for bad in ("<script", "onclick", "onmouseover", "onerror", "javascript:", "<iframe", "<form", "<input",
                "<button", "<img", "<style", "<svg", "<object", "<embed", "evil.example", "data:text", "alert(", "position:fixed", "style="):
        assert bad not in low, bad
    assert "<h2>Title</h2>" in clean
    assert "<b>bold</b>" in clean
    assert '<a href="https://example.com/x"' in clean and 'rel="noopener noreferrer nofollow"' in clean
    assert "bad" in clean            # the text of a dropped link survives
    assert "<td>cell</td>" in clean


def test_sanitiser_closes_open_tags_and_escapes_text():
    clean = pages.sanitize_html("<ul><li>one<li>two & <three>")
    assert clean.count("<ul>") == clean.count("</ul>") == 1
    assert clean.count("<li>") == clean.count("</li>")
    assert "&amp;" in clean and "<three>" not in clean


def test_page_with_only_unsafe_content_is_refused():
    with pytest.raises(ValueError):
        pages.store_page("x", "<script>alert(1)</script>")


def test_title_is_escaped():
    info = pages.store_page('<img src=x onerror=1>"A"', "<p>b</p>")
    doc = pages.load_page(info["token"])
    assert "<img" not in doc and "&lt;img" in doc


def test_owner_only_by_default_and_setting_widens_it(monkeypatch):
    assert pages.allowed_for(True) is True
    assert pages.allowed_for(False) is False
    monkeypatch.setenv("GATE_HTML_PAGES_AUDIENCE", "everyone")
    settings.invalidate()
    assert pages.allowed_for(False) is True
    monkeypatch.setenv("GATE_HTML_PAGES_AUDIENCE", "off")
    settings.invalidate()
    assert pages.allowed_for(True) is False


def test_tool_refuses_for_non_owner_and_returns_link_for_owner():
    refused = run(pages.publish_page({"title": "t", "html": "<p>x</p>"}, allowed=False))
    assert "not enabled" in refused and "/p/" not in refused
    assert list(pages.pages_dir().glob("*.html")) == []
    ok = run(pages.publish_page({"title": "t", "html": "<p>x</p>"}, allowed=True))
    assert "https://blatbot.example.test/p/" in ok
    assert len(list(pages.pages_dir().glob("*.html"))) == 1
    assert pages.urls_in(ok) and pages.urls_in([{"text": ok}]) == pages.urls_in(ok)


def test_tool_is_registered_on_the_host_server_and_not_destructive():
    from inkbox_claude.gate import hosttools, jevagent
    server = hosttools.sdk_server(is_commander=True)
    assert server is not None
    assert not jevagent.is_destructive("mcp__host__make_page", {})
    assert not jevagent.is_write_tool("mcp__host__make_page")


def test_settings_registry_has_the_knobs():
    assert settings.known("GATE_HTML_PAGES_AUDIENCE") and settings.known("GATE_HTML_PAGE_DAYS")
    assert settings.coerce("GATE_HTML_PAGES_AUDIENCE", "everyone") == "everyone"
    with pytest.raises(ValueError):
        settings.coerce("GATE_HTML_PAGES_AUDIENCE", "nobody")
