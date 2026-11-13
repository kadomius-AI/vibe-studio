"""Text rendering to transparent images (captions, headlines, thumbnails, ad/listing overlays).

Markup: wrap words in *asterisks* to paint them in the accent color.
"""
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from .env import font as font_path

_cache = {}


def get_font(name, size):
    key = (name, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(font_path(name), size)
    return _cache[key]


def hex_rgb(h, alpha=255):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def _tokens(line):
    """Split a line into (word, accented) tokens."""
    out = []
    for m in re.finditer(r"\*([^*]+)\*|(\S+)", line):
        if m.group(1):
            out += [(w, True) for w in m.group(1).split()]
        else:
            out.append((m.group(2), False))
    return out


def wrap(text, fnt, max_w):
    lines = []
    for para in text.split("\n"):
        toks, cur = _tokens(para), []
        for t in toks:
            trial = cur + [t]
            w = fnt.getlength(" ".join(x[0] for x in trial))
            if cur and w > max_w:
                lines.append(cur)
                cur = [t]
            else:
                cur = trial
        if cur:
            lines.append(cur)
    return lines


def render_text(text, *, font="Montserrat-ExtraBold.ttf", size=72, color="#FFFFFF", accent="#FFD400",
                max_width=900, align="center", stroke=0, stroke_color="#000000", shadow=True,
                line_gap=1.12, box=None, box_pad=(36, 22), box_radius=28, upper=False,
                highlight_index=None, highlight_color=None, highlight_scale=1.0):
    """Return an RGBA PIL image containing the styled text block.

    highlight_index: index of one word (counting all words) drawn in highlight_color (karaoke captions).
    box: background panel color like "#000000B4" (hex with optional alpha) or None.
    """
    if upper:
        text = text.upper()
    fnt = get_font(font, size)
    lines = wrap(text, fnt, max_width)
    space = fnt.getlength(" ")
    asc, desc = fnt.getmetrics()
    lh = int((asc + desc) * line_gap)
    widths = [sum(fnt.getlength(w) for w, _ in ln) + space * (len(ln) - 1) for ln in lines]
    W = int(max(widths or [1])) + 2 * stroke + 40
    H = lh * len(lines) + 2 * stroke + 30
    pad_x, pad_y = (box_pad if box else (0, 0))
    canvas = Image.new("RGBA", (W + 2 * pad_x, H + 2 * pad_y), (0, 0, 0, 0))
    txt = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(txt)
    idx = 0
    y = pad_y + stroke + 10
    for ln, lw in zip(lines, widths):
        if align == "center":
            x = (canvas.width - lw) / 2
        elif align == "right":
            x = canvas.width - pad_x - stroke - 20 - lw
        else:
            x = pad_x + stroke + 20
        for w, acc in ln:
            fill = accent if acc else color
            if highlight_index is not None and idx == highlight_index:
                fill = highlight_color or accent
            d.text((x, y), w, font=fnt, fill=hex_rgb(fill) if isinstance(fill, str) else fill,
                   stroke_width=stroke, stroke_fill=hex_rgb(stroke_color))
            x += fnt.getlength(w) + space
            idx += 1
        y += lh
    if box:
        h = box.lstrip("#")
        rgba = hex_rgb(h[:6], int(h[6:8], 16) if len(h) == 8 else 220)
        bb = txt.getbbox() or (0, 0, canvas.width, canvas.height)
        ImageDraw.Draw(canvas).rounded_rectangle(
            [bb[0] - pad_x, bb[1] - pad_y, bb[2] + pad_x, bb[3] + pad_y], radius=box_radius, fill=rgba)
    if shadow:
        a = np.array(txt.split()[-1])
        sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sh.putalpha(Image.fromarray((a * 0.75).astype("uint8")))
        sh = sh.filter(ImageFilter.GaussianBlur(max(3, size // 14)))
        canvas = Image.alpha_composite(canvas, sh.transform(sh.size, Image.AFFINE, (1, 0, -3, 0, 1, -5)))
    canvas = Image.alpha_composite(canvas, txt)
    bb = canvas.getbbox()
    return canvas.crop(bb) if bb else canvas


def paste(frame_bgr, rgba, x, y, opacity=1.0):
    """Alpha-blend an RGBA PIL image onto a BGR numpy frame at (x, y) (top-left), in place."""
    if rgba is None or opacity <= 0:
        return frame_bgr
    arr = np.asarray(rgba, dtype=np.float32)
    h, w = arr.shape[:2]
    H, W = frame_bgr.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + w, W), min(y0 + h, H)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    src = arr[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    alpha = (src[..., 3:4] / 255.0) * opacity
    rgb = src[..., 2::-1]  # RGB -> BGR
    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    frame_bgr[y1:y2, x1:x2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    return frame_bgr
