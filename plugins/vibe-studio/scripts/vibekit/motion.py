"""Spec-driven motion videos from still images (ads, listing videos).

Spec (JSON):
{
  "size": [1080, 1920], "fps": 30, "music": "optional.mp3", "music_db": -6,
  "brand": {"bg": "#111111", "accent": "#FFD400", "text": "#FFFFFF"},
  "scenes": [
    {"type": "image",   "src": "photo.jpg", "duration": 2.4, "motion": "zoom_in|zoom_out|pan_left|pan_right|pan_up|pan_down|none",
     "texts": [{"text": "...", "style": "hook|caption|price|lower|cta|badge|title|small", "at": 0.0, "until": null, "y": 0.2}]},
    {"type": "product", "src": "product.png", "duration": 3.0, "bg": "#hex or 'blur'", "cutout": true, "texts": [...]},
    {"type": "card",    "duration": 2.0, "bg": "#hex", "logo": "optional.png", "texts": [...]}
  ]
}
Every scene starts with a short zoom "punch" for energy; text pops in. Words in *asterisks* use the accent color.
"""
import json, subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .env import FFMPEG
from .text import render_text, paste, hex_rgb

TEXT_STYLES = {
    "hook":    dict(font="Anton.ttf", size=118, upper=True, stroke=8, y=0.20),
    "caption": dict(font="Montserrat-ExtraBold.ttf", size=66, box="#000000B0", y=0.74),
    "price":   dict(font="Anton.ttf", size=170, upper=True, stroke=8, y=0.20, accent_all=True),
    "lower":   dict(font="Montserrat-SemiBold.ttf", size=52, box="#000000A0", y=0.86),
    "cta":     dict(font="Anton.ttf", size=104, upper=True, y=0.78, cta=True),
    "badge":   dict(font="Montserrat-ExtraBold.ttf", size=58, upper=True, y=0.10, badge=True),
    "title":   dict(font="PlayfairDisplay.ttf", size=104, y=0.44, shadow_only=True),
    "small":   dict(font="Montserrat-SemiBold.ttf", size=44, upper=True, y=0.55, shadow_only=True),
}


def _load(src):
    im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(src)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def _cover_box(iw, ih, W, H, scale):
    s = max(W / iw, H / ih) * scale
    return iw * s, ih * s, s


def _image_frames(img, W, H, n, motion):
    """Ken-Burns style frames from a still, sub-pixel smooth via affine warps."""
    ih, iw = img.shape[:2]
    bgr = img[..., :3]
    base = 1.12
    for k in range(n):
        p = k / max(1, n - 1)
        e = p * p * (3 - 2 * p)  # ease in-out
        z, dx, dy = base, 0.0, 0.0
        if motion == "zoom_in":
            z = base + 0.12 * e
        elif motion == "zoom_out":
            z = base + 0.12 * (1 - e)
        elif motion == "pan_left":
            dx = 0.5 - e
        elif motion == "pan_right":
            dx = e - 0.5
        elif motion == "pan_up":
            dy = 0.5 - e
        elif motion == "pan_down":
            dy = e - 0.5
        punch = 1 + 0.05 * max(0.0, 1 - k / 6)      # quick zoom punch at the cut
        sw, sh, s = _cover_box(iw, ih, W, H, z * punch)
        ox = (W - sw) / 2 + dx * (sw - W)
        oy = (H - sh) / 2 + dy * (sh - H)
        M = np.float32([[s, 0, ox], [0, s, oy]])
        yield cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _contain_frames(img, W, H, n, motion):
    """Whole photo visible (scaled to width) with a slow move, over a dark blurred copy of itself."""
    bgr = img[..., :3]
    back = cv2.GaussianBlur(next(_image_frames(img, W, H, 1, "none")), (0, 0), 35)
    back = (back * 0.45).astype(np.uint8)
    ih, iw = bgr.shape[:2]
    for k in range(n):
        p = k / max(1, n - 1)
        e = p * p * (3 - 2 * p)
        z = 1.0 + (0.06 * e if motion != "zoom_out" else 0.06 * (1 - e))
        punch = 1 + 0.04 * max(0.0, 1 - k / 6)
        s = W / iw * 1.04 * z * punch
        ox = (W - iw * s) / 2 + (({"pan_left": 0.5 - e, "pan_right": e - 0.5}.get(motion, 0)) * iw * s * 0.08)
        oy = (H - ih * s) / 2
        M = np.float32([[s, 0, ox], [0, s, oy]])
        f = back.copy()
        fg = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = cv2.warpAffine(np.ones((ih, iw), np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        yield (fg * m + f * (1 - m)).astype(np.uint8)


def _bg_canvas(W, H, bg, img=None):
    if bg == "blur" and img is not None:
        f = next(_image_frames(img, W, H, 1, "none"))
        f = cv2.GaussianBlur(f, (0, 0), 30)
        return (f * 0.55).astype(np.uint8)
    c = np.array(hex_rgb(bg or "#111111")[:3][::-1], np.float32)
    grad = np.linspace(1.15, 0.55, H, dtype=np.float32)[:, None, None]
    return np.clip(np.ones((H, W, 3), np.float32) * c * grad, 0, 255).astype(np.uint8)


def _product_frames(img, W, H, n, bg, cutout=True):
    from .segment import product_mask
    rgba = None
    if img.shape[2] == 4:
        rgba = img
    elif cutout:
        m, cov = product_mask(img)
        if 0.05 < cov < 0.9:
            rgba = np.dstack([img, (m * 255).astype(np.uint8)])
    if rgba is None:  # no clean cut-out: present the photo as a rounded card
        h, w = img.shape[:2]
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius=int(min(w, h) * 0.06), fill=255)
        rgba = np.dstack([img, np.array(mask)])
    canvas = _bg_canvas(W, H, bg, img[..., :3])
    ph, pw = rgba.shape[:2]
    target_w = W * 0.82
    s0 = min(target_w / pw, H * 0.56 / ph)
    shadow = cv2.GaussianBlur((rgba[..., 3] > 0).astype(np.float32), (0, 0), 25)
    for k in range(n):
        p = k / max(1, n - 1)
        s = s0 * (0.92 + 0.08 * min(1, p * 3)) * (1 + 0.03 * p)
        bob = np.sin(p * np.pi * 2) * H * 0.006
        w2, h2 = int(pw * s), int(ph * s)
        prod = cv2.resize(rgba, (w2, h2), interpolation=cv2.INTER_AREA)
        sh = cv2.resize(shadow, (w2, h2))
        f = canvas.copy()
        x, y = (W - w2) // 2, int(H * 0.46 - h2 / 2 + bob)
        # soft drop shadow
        sy = min(H, y + int(h2 * 0.06) + h2) - (y + int(h2 * 0.06))
        if sy > 0 and 0 <= x and x + w2 <= W:
            region = f[y + int(h2 * 0.06):y + int(h2 * 0.06) + sy, x:x + w2].astype(np.float32)
            region *= (1 - 0.45 * sh[:sy, :, None])
            f[y + int(h2 * 0.06):y + int(h2 * 0.06) + sy, x:x + w2] = region.astype(np.uint8)
        pil = Image.fromarray(cv2.cvtColor(prod, cv2.COLOR_BGRA2RGBA))
        paste(f, pil, x, y)
        yield f


def _text_img(t, brand, W):
    st = dict(TEXT_STYLES.get(t.get("style", "caption"), TEXT_STYLES["caption"]))
    scale = W / 1080
    accent = brand.get("accent", "#FFD400")
    color = brand.get("text", "#FFFFFF")
    txt = t["text"]
    if st.pop("accent_all", False):
        color = accent
    kw = dict(font=st["font"], size=int(st["size"] * scale), upper=st.get("upper", False),
              stroke=int(st.get("stroke", 0) * scale), color=color, accent=accent, max_width=int(W * 0.86))
    if st.get("cta"):
        return render_text(txt, box=accent + "FF", box_pad=(44, 20), **{**kw, "color": brand.get("cta_text", "#000000")})
    if st.get("badge"):
        return render_text(txt, box=accent + "FF", box_pad=(26, 12), box_radius=40,
                           **{**kw, "color": brand.get("cta_text", "#000000"), "shadow": False})
    if st.get("box"):
        return render_text(txt, box=st["box"], **kw)
    return render_text(txt, **kw)


def render_spec(spec, out):
    if isinstance(spec, (str, Path)):
        base = Path(spec).parent
        spec = json.loads(Path(spec).read_text())
    else:
        base = Path(".")
    W, H = spec.get("size", [1080, 1920])
    fps = spec.get("fps", 30)
    brand = spec.get("brand", {})
    total = sum(s.get("duration", 2.5) for s in spec["scenes"])
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    music = spec.get("music")
    if music:
        mp = (base / music) if not Path(music).is_absolute() else Path(music)
        cmd += ["-i", str(mp), "-filter_complex",
                f"[1:a]atrim=0:{total:.2f},afade=t=in:d=0.3,afade=t=out:st={max(0, total - 1.2):.2f}:d=1.2,"
                f"volume={spec.get('music_db', -6)}dB,loudnorm=I=-14:TP=-1.5,aresample=48000[a]",
                "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for sc in spec["scenes"]:
        n = max(1, int(round(sc.get("duration", 2.5) * fps)))
        typ = sc.get("type", "image")
        if typ == "card":
            canvas = _bg_canvas(W, H, sc.get("bg", brand.get("bg", "#111111")))
            if sc.get("logo"):
                lg = Image.open(base / sc["logo"]).convert("RGBA")
                lg.thumbnail((int(W * 0.5), int(H * 0.16)))
                paste(canvas, lg, (W - lg.width) / 2, H * 0.24)
            frames = (canvas.copy() for _ in range(n))
        elif typ == "product":
            frames = _product_frames(_load(base / sc["src"]), W, H, n, sc.get("bg", brand.get("bg", "#111111")),
                                     sc.get("cutout", True))
        elif sc.get("fit") == "contain":
            frames = _contain_frames(_load(base / sc["src"]), W, H, n, sc.get("motion", "zoom_in"))
        else:
            frames = _image_frames(_load(base / sc["src"]), W, H, n, sc.get("motion", "zoom_in"))
        texts = [(t, _text_img(t, brand, W)) for t in sc.get("texts", [])]
        for k, f in enumerate(frames):
            ts = k / fps
            for t, im in texts:
                a = t.get("at", 0.0)
                b = t.get("until") or 1e9
                if a <= ts < b:
                    age = ts - a
                    pop = min(1.0, age / 0.18)
                    s = 0.86 + 0.14 * pop
                    img = im if s >= 0.999 else im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
                    y = t.get("y", TEXT_STYLES.get(t.get("style", "caption"), {}).get("y", 0.5))
                    x = {"left": W * 0.07, "right": W * 0.93 - img.width}.get(t.get("align"), (W - img.width) / 2)
                    paste(f, img, x, y * H - img.height / 2, opacity=pop)
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed for " + str(out))
    return str(out)
