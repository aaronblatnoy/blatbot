"""render_image: the model draws an image as SVG code, this renders it to PNG.

No image-generation model is involved. The model writes the SVG (hand-built shapes,
flat-shaded polygons, thin guilloche curves, text in a real font); this tool checks the
SVG is inert, rasterizes it with headless Chromium (falling back to rsvg-convert),
optionally lays several variants out on one labeled contact sheet with small-size
previews, saves the PNGs under ~/blatbot/media/generated/ and hands the picture back to
the model so it can look at it, fix what is off and render again. Sending is the
send_message tool's ``images`` argument (gate/outcomes.py).
"""

from __future__ import annotations

import asyncio
import glob
import io
import os
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GENERATED_DIR = Path(os.getenv("BLATBOT_GENERATED_DIR") or (Path.home() / "blatbot" / "media" / "generated"))
MAX_SVG_BYTES = 2_000_000
MAX_VARIANTS = 12
MAX_SIDE = 4096
DEFAULT_SIDE = 1024
RENDER_TIMEOUT_S = 30.0
MAX_PNG_BYTES = 8_000_000
SEE_MAX_SIDE = 1568          # what the model is shown: its own picture, never larger

_ALLOWED_HREF = re.compile(r"^(#|data:image/(png|jpeg|jpg|gif|webp);base64,)", re.IGNORECASE)
_BAD_URL = re.compile(r"url\(\s*['\"]?\s*(?!#)", re.IGNORECASE)
_BAD_CSS = re.compile(r"@import|expression\s*\(|javascript:|file:|https?:|ftp:|//\w", re.IGNORECASE)
_NS_SVG = "http://www.w3.org/2000/svg"

DESCRIPTION = (
    "DRAW AN IMAGE AS CODE AND RENDER IT TO A PNG you can then look at. You write the SVG yourself "
    "(hand-built vector shapes; 3D looks are flat-shaded polygons; guilloche is thin low-opacity sine/rosette "
    "curves you compute; lettering is <text> in a real font such as Liberation Serif or Liberation Sans). No image "
    "model is used. Pass svg for one image, or variants (a list of {id, svg, caption}) for a labeled CONTACT SHEET: "
    "every variant shown large with 64 px and 32 px previews under it, on a light gray backdrop. The result is the "
    "saved PNG path(s) AND the picture itself so you can check it: look, fix what is off, render again. To send "
    "it, pass the PNG path in send_message's images. External links, scripts and files inside the SVG are refused."
)
SCHEMA = {"type": "object", "properties": {
    "svg": {"type": "string", "description": "one complete <svg> document (needs width/height or a viewBox)"},
    "variants": {"type": "array", "description": "several SVGs for a contact sheet",
                 "items": {"type": "object", "required": ["id", "svg"], "properties": {
                     "id": {"type": "string", "description": "short label such as ic-01"},
                     "svg": {"type": "string"},
                     "caption": {"type": "string", "description": "optional second line under the id"}}}},
    "name": {"type": "string", "description": "file name stem for the saved PNG (letters, digits, dashes)"},
    "width": {"type": "integer", "description": "output width for a single svg (default from the SVG, max 4096)"},
    "height": {"type": "integer", "description": "output height for a single svg"},
    "tile": {"type": "integer", "description": "contact sheet: size of the large view in px (default 200)"},
    "preview_sizes": {"type": "array", "items": {"type": "integer"},
                      "description": "contact sheet: small previews under each variant (default [64, 32])"},
}}


class SvgRefused(ValueError):
    pass


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def sanitize(svg: str) -> str:
    """The SVG text if it is inert; raises SvgRefused saying why when it is not."""
    svg = (svg or "").strip()
    if not svg:
        raise SvgRefused("the svg is empty")
    if len(svg.encode("utf-8")) > MAX_SVG_BYTES:
        raise SvgRefused(f"the svg is larger than {MAX_SVG_BYTES // 1000} KB; simplify it")
    if re.search(r"<!DOCTYPE|<!ENTITY|<\?xml-stylesheet", svg, re.IGNORECASE):
        raise SvgRefused("DOCTYPE, ENTITY and stylesheet instructions are not allowed")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise SvgRefused(f"the svg is not valid XML: {exc}")
    if _local(root.tag) != "svg":
        raise SvgRefused("the root element must be <svg>")
    for el in root.iter():
        name = _local(el.tag)
        if name in ("script", "foreignobject", "iframe", "object", "embed", "audio", "video", "animate", "set"):
            raise SvgRefused(f"<{name}> is not allowed")
        if name == "style" and _BAD_CSS.search(el.text or ""):
            raise SvgRefused("a <style> block loads something external or runs code")
        for key, val in el.attrib.items():
            k = key.rsplit("}", 1)[-1].lower()
            if k.startswith("on"):
                raise SvgRefused(f"event attribute {k} is not allowed")
            if k == "href" and not _ALLOWED_HREF.match(val.strip()):
                raise SvgRefused("href may only point at #ids or an embedded data:image")
            if "url(" in val.lower() and _BAD_URL.search(val):
                raise SvgRefused("url() may only point at an #id inside the svg")
            if k == "style" and _BAD_CSS.search(val):
                raise SvgRefused("a style attribute loads something external")
    return svg


def _size_of(svg: str) -> Tuple[int, int]:
    root = ET.fromstring(svg)

    def num(v: Optional[str]) -> Optional[float]:
        m = re.match(r"^\s*([0-9.]+)\s*(px)?\s*$", v or "")
        return float(m.group(1)) if m else None
    w, h = num(root.get("width")), num(root.get("height"))
    if not (w and h):
        vb = (root.get("viewBox") or "").replace(",", " ").split()
        if len(vb) == 4:
            try:
                w, h = float(vb[2]), float(vb[3])
            except ValueError:
                w = h = None
    if not (w and h):
        raise SvgRefused("the svg needs a width and height or a viewBox")
    return int(w), int(h)


def _fit(w: int, h: int, want_w: Optional[int], want_h: Optional[int]) -> Tuple[int, int]:
    if want_w and not want_h:
        want_h = round(h * want_w / w)
    elif want_h and not want_w:
        want_w = round(w * want_h / h)
    elif not (want_w and want_h):
        k = DEFAULT_SIDE / max(w, h)
        want_w, want_h = round(w * k), round(h * k)
    want_w, want_h = max(16, min(MAX_SIDE, int(want_w))), max(16, min(MAX_SIDE, int(want_h)))
    return want_w, want_h


def _chrome() -> Optional[str]:
    found = sorted(glob.glob(str(Path.home() / ".cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell")))
    return found[-1] if found else None


async def _run(cmd: List[str]) -> Tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=RENDER_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        raise SvgRefused(f"rendering took longer than {int(RENDER_TIMEOUT_S)} s; simplify the svg")
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace")[-400:]


async def rasterize(svg: str, width: int, height: int) -> bytes:
    """PNG bytes of an already-sanitized SVG at exactly width x height."""
    svg = sanitize(svg)
    with tempfile.TemporaryDirectory(prefix="blatbot-svg-") as td:
        src, png = Path(td) / "in.svg", Path(td) / "out.png"
        root = ET.fromstring(svg)
        w0, h0 = _size_of(svg)
        # Scale through the viewBox so the drawing fills the requested size.
        if not root.get("viewBox"):
            svg = re.sub(r"<svg\b", f'<svg viewBox="0 0 {w0} {h0}"', svg, count=1)
        svg = re.sub(r"(<svg\b[^>]*?)\swidth=(\"[^\"]*\"|'[^']*')", r"\1", svg, count=1)
        svg = re.sub(r"(<svg\b[^>]*?)\sheight=(\"[^\"]*\"|'[^']*')", r"\1", svg, count=1)
        svg = re.sub(r"<svg\b", f'<svg width="{width}" height="{height}"', svg, count=1)
        src.write_text(svg, encoding="utf-8")
        chrome = _chrome()
        code, out = (1, "")
        if chrome:
            code, out = await _run([chrome, "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                                    "--force-device-scale-factor=1", "--disable-remote-fonts", "--default-background-color=00000000",
                                    '--host-resolver-rules=MAP * ~NOTFOUND',
                                    f"--window-size={width},{height}", f"--screenshot={png}", src.as_uri()])
        if code != 0 or not png.exists():
            rsvg = shutil.which("rsvg-convert")
            if not rsvg:
                raise SvgRefused(f"no rasterizer is available ({out.strip()})")
            png.unlink(missing_ok=True)
            code, out = await _run([rsvg, "-w", str(width), "-h", str(height), "--unlimited", str(src), "-o", str(png)])
            if code != 0 or not png.exists():
                raise SvgRefused(f"the svg could not be rendered: {out.strip()}")
        data = png.read_bytes()
    if len(data) > MAX_PNG_BYTES:
        raise SvgRefused("the rendered PNG is too large; use a smaller size")
    return data


def _stem(name: Any, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name or "")).strip("-")[:40]
    return f"{s or fallback}-{time.strftime('%H%M%S')}"


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/liberation/LiberationSans-%s.ttf" % ("Bold" if bold else "Regular"),
              "/usr/share/fonts/TTF/LiberationSans-%s.ttf" % ("Bold" if bold else "Regular")):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _contact_sheet(tiles: List[Dict[str, Any]], tile: int, sizes: List[int]) -> bytes:
    """tiles: {id, caption, big: PIL image}. Light gray backdrop, large view, small previews, labels."""
    from PIL import Image, ImageDraw
    bg, ink, dim = (233, 233, 236), (30, 30, 34), (110, 110, 118)
    cols = min(3, len(tiles)) or 1
    rows = (len(tiles) + cols - 1) // cols
    pad, label_h = 24, 44
    prev_h = (max(sizes) + 28) if sizes else 0
    cell_w = max(tile, sum(sizes) + 14 * len(sizes)) + pad
    cell_h = tile + label_h + prev_h + pad
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), bg)
    d = ImageDraw.Draw(sheet)
    f_id, f_cap, f_sz = _font(18, True), _font(14), _font(12)
    for i, t in enumerate(tiles):
        x0 = pad + (i % cols) * cell_w
        y0 = pad + (i // cols) * cell_h
        d.text((x0, y0), str(t["id"]), font=f_id, fill=ink)
        if t.get("caption"):
            d.text((x0, y0 + 22), str(t["caption"])[:48], font=f_cap, fill=dim)
        big = t["big"].convert("RGBA")
        view = big.resize((tile, round(big.height * tile / big.width)) if big.width >= big.height
                          else (round(big.width * tile / big.height), tile), Image.LANCZOS)
        sheet.paste(view, (x0, y0 + label_h), view)
        x = x0
        for s in sizes:
            small = big.resize((s, round(big.height * s / big.width)) if big.width >= big.height
                               else (round(big.width * s / big.height), s), Image.LANCZOS)
            sheet.paste(small, (x, y0 + label_h + tile + 10), small)
            d.text((x, y0 + label_h + tile + 12 + s), f"{s} px", font=f_sz, fill=dim)
            x += s + 14
    buf = io.BytesIO()
    sheet.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def shown_to_model(png: bytes) -> bytes:
    """The picture handed back to the model, no larger than SEE_MAX_SIDE on a side."""
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    if max(im.size) <= SEE_MAX_SIDE:
        return png
    im.thumbnail((SEE_MAX_SIDE, SEE_MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


async def render(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run the tool. Returns {"text": str, "image": bytes | None}; refusals come back as text, never raise."""
    from PIL import Image
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        variants = args.get("variants") or []
        if variants:
            if not isinstance(variants, list) or len(variants) > MAX_VARIANTS:
                raise SvgRefused(f"variants must be a list of at most {MAX_VARIANTS}")
            tile = max(96, min(480, int(args.get("tile") or 200)))
            sizes = [max(8, min(256, int(s))) for s in (args.get("preview_sizes") or [64, 32])][:4]
            stem = _stem(args.get("name"), "sheet")
            tiles, lines = [], []

            async def one(v: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
                svg = sanitize(v.get("svg"))
                w, h = _size_of(svg)
                bw, bh = _fit(w, h, None, None)
                return v, await rasterize(svg, bw, bh)
            done = await asyncio.gather(*[one(v) for v in variants], return_exceptions=True)
            for v, res in zip(variants, done):
                if isinstance(res, Exception):
                    raise SvgRefused(f"variant {v.get('id')}: {res}")
                vid, png = str(v.get("id") or f"v{len(tiles) + 1}"), res[1]
                path = GENERATED_DIR / f"{stem}-{re.sub(r'[^A-Za-z0-9_-]+', '-', vid)}.png"
                path.write_bytes(png)
                lines.append(f"{vid}: {path}")
                tiles.append({"id": vid, "caption": v.get("caption"), "big": Image.open(io.BytesIO(png))})
            sheet = _contact_sheet(tiles, tile, sizes)
            spath = GENERATED_DIR / f"{stem}.png"
            spath.write_bytes(sheet)
            text = (f"Contact sheet saved: {spath}  (send this one with send_message images). "
                    f"Large view {tile} px, previews {', '.join(str(s) for s in sizes)} px.\n"
                    "Each variant's own PNG:\n" + "\n".join(lines) +
                    "\nThe sheet is attached below: look at it, fix what is off at the small sizes, render again.")
            return {"text": text, "image": shown_to_model(sheet)}
        svg = sanitize(args.get("svg"))
        w, h = _size_of(svg)
        ow, oh = _fit(w, h, args.get("width"), args.get("height"))
        png = await rasterize(svg, ow, oh)
        path = GENERATED_DIR / f"{_stem(args.get('name'), 'image')}.png"
        path.write_bytes(png)
        return {"text": f"Rendered {ow}x{oh} PNG: {path}\nThe picture is attached below: look at it, fix what is off, "
                        "render again. To send it, pass this path in send_message's images.",
                "image": shown_to_model(png)}
    except SvgRefused as exc:
        return {"text": f"Not rendered: {exc}", "image": None}
    except Exception as exc:  # a bad argument type or an unreadable PNG
        return {"text": f"Not rendered: {exc}", "image": None}


def allowed_image_paths(paths: Any, limit: int = 4) -> List[str]:
    """The generated PNGs a send may attach: only files directly under GENERATED_DIR."""
    out: List[str] = []
    base = GENERATED_DIR.resolve()
    for p in (paths or [])[:limit] if isinstance(paths, list) else []:
        q = Path(str(p)).expanduser().resolve()
        if q.parent != base or q.suffix.lower() not in (".png", ".jpg", ".jpeg") or not q.is_file():
            raise ValueError(f"{p} is not an image made by render_image; attach only paths it returned.")
        out.append(str(q))
    return out
