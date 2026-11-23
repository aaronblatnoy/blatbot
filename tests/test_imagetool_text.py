"""render_image: <text> in a bundled font is converted to exact outlines."""

import asyncio
import re

import pytest

pytest.importorskip("fontTools")

from inkbox_claude.gate import imagetool as it


def _svg(inner):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120" viewBox="0 0 400 120">{inner}</svg>'


def test_bundled_font_text_becomes_paths_and_unknown_font_is_left_alone():
    out = it.outline_text(_svg('<text x="20" y="80" font-family="Libre Baskerville" font-size="48" '
                               'font-style="italic" fill="#fff">Migreat</text>'))
    assert "<text" not in out and "<path" in out and 'fill="#fff"' in out
    keep = _svg('<text x="1" y="9" font-family="Comic Sans">hi</text>')
    assert it.outline_text(keep) == keep


@pytest.mark.parametrize("family", ["Libre Baskerville", "Newsreader", "Inter"])
def test_each_bundled_family_measures_and_italic_differs(family):
    a, wa = it.text_to_path("Migreat", family.lower(), 48)
    b, wb = it.text_to_path("Migreat", family.lower(), 48, italic=True)
    assert wa > 100 and a.startswith("M") and b
    assert (a != b) or family == "Inter"


def test_anchor_middle_and_weight_and_tspan():
    out = it.outline_text(_svg('<text x="200" y="80" text-anchor="middle" font-family="Newsreader" '
                               'font-size="40" font-weight="700">AB<tspan fill="red">C</tspan></text>'))
    xs = [float(m) for m in re.findall(r'translate\(([-0-9.]+) ', out)]
    assert len(xs) == 2 and xs[0] < 200 and 'fill="red"' in out


def test_render_uses_the_exact_font(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "GENERATED_DIR", tmp_path)
    res = asyncio.run(it.render({"svg": _svg('<rect width="400" height="120" fill="#10243f"/>'
                                             '<text x="20" y="80" font-family="Libre Baskerville" '
                                             'font-size="48" fill="#f3ead8">Migreat</text>'), "name": "t"}))
    assert res["image"] and res["text"].startswith("Rendered")
