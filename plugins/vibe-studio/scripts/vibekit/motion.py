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
    "small":   dict(font="Montserrat-SemiBold.ttf", size=54, upper=True, y=0.55, shadow_only=True),
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
        s = W / iw * 1.3 * z * punch
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
    from .segment import product_cutout
    rgba = None
    if img.shape[2] == 4:
        rgba = img
    elif cutout:
        m, cov = product_cutout(img)
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


def _text_kw(t, brand, W):
    st = dict(TEXT_STYLES.get(t.get("style", "caption"), TEXT_STYLES["caption"]))
    scale = W / 1080
    accent = brand.get("accent", "#FFD400")
    color = brand.get("text", "#FFFFFF")
    if st.pop("accent_all", False):
        color = accent
    kw = dict(font=st["font"], size=int(st["size"] * scale), upper=st.get("upper", False),
              stroke=int(st.get("stroke", 0) * scale), color=color, accent=accent, max_width=int(W * 0.86))
    if st.get("cta"):
        kw.update(box=accent + "FF", box_pad=(44, 20), color=brand.get("cta_text", "#000000"))
    elif st.get("badge"):
        kw.update(box=accent + "FF", box_pad=(26, 12), box_radius=40, color=brand.get("cta_text", "#000000"), shadow=False)
    elif st.get("box"):
        kw.update(box=st["box"])
    return kw


class _Text:
    """A text overlay with pop-in or word-by-word ('anim': 'words') animation."""

    def __init__(self, t, brand, W, H):
        self.t, self.W, self.H = t, W, H
        self.kw = _text_kw(t, brand, W)
        self.words = len(t["text"].replace("*", "").split())
        self.kinetic = t.get("anim") == "words" and self.words > 1
        self.step = t.get("word_secs", 0.13)
        self.cache = {}
        self.y = t.get("y", TEXT_STYLES.get(t.get("style", "caption"), {}).get("y", 0.5))

    def img(self, k):
        if k not in self.cache:
            self.cache[k] = render_text(self.t["text"], visible_words=k if self.kinetic else None, **self.kw)
        return self.cache[k]

    def draw(self, f, ts):
        a, b = self.t.get("at", 0.0), self.t.get("until") or 1e9
        if not (a <= ts < b):
            return
        age = ts - a
        if self.kinetic:
            k = min(self.words, 1 + int(age / self.step))
            im = self.img(k)
            pop = 1.0
        else:
            im = self.img(None)
            pop = min(1.0, age / 0.16)
            s = 0.82 + 0.18 * (1 - (1 - pop) ** 3)
            if s < 0.999:
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        W, H = self.W, self.H
        x = {"left": W * 0.07, "right": W * 0.93 - im.width}.get(self.t.get("align"), (W - im.width) / 2)
        paste(f, im, x, self.y * H - im.height / 2, opacity=pop)


# ---------------- transitions ----------------
def _shift(img, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _hblur(img, k):
    k = int(k)
    if k < 3:
        return img
    ker = np.ones((1, k), np.float32) / k
    return cv2.filter2D(img, -1, ker)


def _zoom(img, s):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _zoom_blur(img, s, amount):
    acc = img.astype(np.float32)
    n = 4
    for i in range(1, n + 1):
        acc += _zoom(img, 1 + amount * i / n).astype(np.float32)
    return _zoom((acc / (n + 1)).astype(np.uint8), s)


def transition(kind, tail, head):
    """Replace the last frames of scene A (tail) and first frames of scene B (head) with a transition."""
    k = len(tail)
    out = []
    W = tail[0].shape[1]
    if kind == "whip":
        for i in range(2 * k):
            p = (i + 0.5) / (2 * k)
            e = p * p * (3 - 2 * p)
            a = tail[min(i, k - 1)] if i < k else tail[-1]
            b = head[max(0, i - k)] if i >= k else head[0]
            off = -e * W
            fa, fb = _shift(a, off), _shift(b, off + W)
            mix = fa.copy()
            x = int(W + off)
            mix[:, max(0, x):] = fb[:, max(0, x):]
            out.append(_hblur(mix, 18 + 140 * np.sin(np.pi * p)))
    elif kind == "zoom":
        for i in range(2 * k):
            p = (i + 0.5) / (2 * k)
            if i < k:
                out.append(_zoom_blur(tail[i], 1 + 0.5 * p * 2, 0.12 * p * 2))
            else:
                q = (i - k + 0.5) / k
                out.append(_zoom_blur(head[i - k], 1.35 - 0.35 * q, 0.12 * (1 - q)))
    elif kind == "flash":
        for i in range(2 * k):
            p = (i + 0.5) / (2 * k)
            src = tail[i] if i < k else head[i - k]
            w = 1 - abs(p - 0.5) * 2
            out.append(cv2.addWeighted(src, 1 - w * 0.95, np.full_like(src, 255), w * 0.95, 0))
    elif kind == "fade":
        for i in range(2 * k):
            p = (i + 0.5) / (2 * k)
            a = tail[min(i, k - 1)]
            b = head[max(0, i - k)] if i >= k else head[0]
            out.append(cv2.addWeighted(a, 1 - p, b, p, 0))
    else:  # cut
        out = list(tail) + list(head)
    return out


TRANSITION_SFX = {"whip": ("whoosh_short", -8), "zoom": ("whoosh", -9), "flash": ("hit", -7), "fade": (None, 0)}
AUTO_CYCLE = ["zoom", "whip", "flash", "whip"]


def _scene_frames(sc, W, H, n, base, brand):
    typ = sc.get("type", "image")
    if typ == "card":
        canvas = _bg_canvas(W, H, sc.get("bg", brand.get("bg", "#111111")))
        if sc.get("logo"):
            lg = Image.open(base / sc["logo"]).convert("RGBA")
            lg.thumbnail((int(W * 0.5), int(H * 0.16)))
            paste(canvas, lg, (W - lg.width) / 2, H * 0.24)
        return [_zoom(canvas, 1.0 + 0.04 * k / max(1, n - 1)) for k in range(n)]
    if typ == "product":
        return list(_product_frames(_load(base / sc["src"]), W, H, n, sc.get("bg", brand.get("bg", "#111111")),
                                    sc.get("cutout", True)))
    img = _load(base / sc["src"])
    motion = sc.get("motion", "zoom_in")
    if motion.startswith("parallax"):
        from .depth import parallax_frames
        move = motion.split("_", 1)[1] if "_" in motion else "push"
        return list(parallax_frames(img[..., :3], W, H, n, move=move, strength=sc.get("strength", 1.0)))
    if sc.get("fit") == "contain":
        return list(_contain_frames(img, W, H, n, motion))
    return list(_image_frames(img, W, H, n, motion))


def _resolve(base, p):
    return Path(p) if Path(p).is_absolute() else base / p


def render_spec(spec, out):
    """Render a spec to MP4 with transitions, voiceover, synced captions, music (ducked) and sound effects."""
    from . import sound
    if isinstance(spec, (str, Path)):
        base = Path(spec).parent
        spec = json.loads(Path(spec).read_text())
    else:
        base = Path(".")
    W, H = spec.get("size", [1080, 1920])
    fps = spec.get("fps", 30)
    brand = spec.get("brand", {})
    scenes = spec["scenes"]
    durs = [s.get("duration", 2.5) for s in scenes]

    # ---- voiceover: generate and fit scene timing to it ----
    vo_audio, vo_words, vo_start = None, None, spec.get("voiceover", {}).get("start", 0.35) if spec.get("voiceover") else 0
    if spec.get("voiceover"):
        from .voice import speak
        vo = spec["voiceover"]
        vo_audio, vo_words, vo_len = speak(vo["text"], vo.get("voice", "warm_female"), vo.get("speed", 1.0))
        need = vo_start + vo_len + spec.get("tail", 1.4)
        if sum(durs) < need or spec.get("fit_to_voice", True):
            f = need / sum(durs)
            durs = [d * f for d in durs]

    # ---- music: file, or generate an original track ----
    music_audio = None
    m = spec.get("music")
    if m:
        if isinstance(m, dict) and m.get("prompt"):
            import hashlib
            secs = m.get("seconds") or round(sum(durs) + 2)
            key = hashlib.md5(f"{m['prompt']}|{m.get('seed', 7)}|{secs}".encode()).hexdigest()[:10]
            path = base / f"music_{key}.wav"
            if not path.exists():
                gen = Path(__file__).resolve().parent.parent / "music_gen.py"
                py = Path.home() / ".local/share/vibekit/acestep-venv/bin/python"
                subprocess.run([str(py), str(gen), "--prompt", m["prompt"], "--seconds", str(secs),
                                "--seed", str(m.get("seed", 7)), "--out", str(path)], check=True)
            music_audio = sound.load(path)
        else:
            music_audio = sound.load(_resolve(base, m if isinstance(m, str) else m["file"]))
        if spec.get("beat_sync", True) and music_audio is not None:
            bt, _ = sound.beats(music_audio)
            durs = sound.snap_durations(durs, bt)

    # ---- frames with transitions ----
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    tmp_wav = Path(str(out) + ".mix.wav")
    has_audio = bool(vo_audio is not None or music_audio is not None or spec.get("sfx", True))
    if has_audio:
        cmd += ["-i", str(tmp_wav), "-map", "0:v", "-map", "1:a", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]

    from .captions import CaptionTrack
    vo_caps = None
    if vo_words and spec.get("voice_captions", True):
        shifted = [{"word": w["word"], "start": w["start"] + vo_start, "end": w["end"] + vo_start} for w in vo_words]
        vo_caps = CaptionTrack(shifted, W, H, style=spec.get("caption_style", "punch"), y=spec.get("caption_y", 0.70),
                               highlight=spec.get("caption_highlight", brand.get("accent")))

    frames_out = []          # (global frame index) handled by writer
    sfx_events = []
    state = {"i": 0}
    vid = None

    def write(f):
        t = state["i"] / fps
        if vo_caps:
            c = vo_caps.at(t)
            if c:
                paste(f, *c)
        vid.stdin.write(np.ascontiguousarray(f).tobytes())
        state["i"] += 1

    # audio must exist before ffmpeg reads it: render video frames to a temp file first, then mux
    tmp_vid = Path(str(out) + ".video.mp4")
    vcmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", str(tmp_vid)]
    vid = subprocess.Popen(vcmd, stdin=subprocess.PIPE)
    K = 5
    prev_tail = None
    default_tr = spec.get("transition", "auto")
    for i, (sc, d) in enumerate(zip(scenes, durs)):
        n = max(2 * K + 2, int(round(d * fps)))
        frames = _scene_frames(sc, W, H, n, base, brand)
        texts = [_Text(t, brand, W, H) for t in sc.get("texts", [])]
        for k, f in enumerate(frames):
            for tx in texts:
                tx.draw(f, k / fps)
        # text pop sounds on big text
        for t in sc.get("texts", []):
            if t.get("style") in ("hook", "price", "cta", "badge") and spec.get("sfx", True):
                sfx_events.append(((state["i"] + (0 if prev_tail is None else K)) / fps + t.get("at", 0.0), "pop", -16))
        if prev_tail is not None:
            kind = sc.get("transition", default_tr)
            if kind == "auto":
                kind = AUTO_CYCLE[(i - 1) % len(AUTO_CYCLE)]
            sfx_name, db = TRANSITION_SFX.get(kind, (None, 0))
            if sfx_name and spec.get("sfx", True):
                sfx_events.append((state["i"] / fps + K / fps - 0.18, sfx_name, db))
            for f in transition(kind, prev_tail, frames[:K]):
                write(f)
            frames = frames[K:]
        if i == 0 and spec.get("sfx", True):
            sfx_events.append((0.0, "hit", -9))          # impact on the hook
        if i == len(scenes) - 2 and spec.get("sfx", True) and len(scenes) > 2:
            sfx_events.append(((state["i"] + len(frames)) / fps - 1.0, "riser", -14))   # build into the end card
        if i < len(scenes) - 1:
            for f in frames[:-K]:
                write(f)
            prev_tail = frames[-K:]
        else:
            for f in frames:
                write(f)
    vid.stdin.close()
    if vid.wait() != 0:
        raise RuntimeError("ffmpeg failed while rendering frames")
    total = state["i"] / fps

    # ---- audio mix ----
    if has_audio:
        voice_track = None
        if vo_audio is not None:
            voice_track = np.concatenate([np.zeros((int(vo_start * sound.SR), 2), np.float32), vo_audio])
        mixed = sound.mix(total, voice=voice_track, music=music_audio, sfx_events=sfx_events if spec.get("sfx", True) else (),
                          music_db=spec.get("music_db", -15 if vo_audio is not None else -9))
        sound.save_wav(tmp_wav, mixed)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(tmp_vid), "-i", str(tmp_wav), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", "-shortest", str(out)], check=True)
        tmp_wav.unlink(missing_ok=True)
        tmp_vid.unlink(missing_ok=True)
    else:
        tmp_vid.rename(out)
    return str(out)
