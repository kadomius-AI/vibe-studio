#!/usr/bin/env python3
"""/thumbnail — a video in, 3 ready-to-upload YouTube thumbnails (1280x720) out.

  thumbnail.py candidates VIDEO_OR_URL --out DIR [--n 12]
      Scores frames (face size + confidence, sharpness, exposure) and saves the best, spread over the
      video, plus candidates_sheet.jpg (numbered) for Claude to look at and pick from.

  thumbnail.py make --frame IMG --text "BIG *WORDS*" --style face|split|result --out FILE.png
                    [--frame2 IMG] [--labels "BEFORE|AFTER"] [--bg blur|dark|#hex] [--accent #FFD400]
                    [--side right|left]
      face   : subject cut out large on one side, 2-4 huge words on the other (classic emotion thumbnail)
      split  : two frames side by side with an arrow (before/after, then/now, contrast)
      result : full-bleed frame zoomed on the subject + bold text banner (object/result focus)
      --face-edit beard|stubble|shave|custom [--face-prompt "..."]: generative face edit (own face / consent only)

  thumbnail.py compare A.png B.png C.png --out compare.png
      Side-by-side at full size and at real phone size (~246px wide) to check readability.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from vibekit import media
from vibekit.faces import FaceFinder
from vibekit.segment import person_mask, to_rgba
from vibekit.text import render_text, hex_rgb

TW, TH = 1280, 720


def candidates(video, out, n=12):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if str(video).startswith("http"):
        video, _ = media.ingest(str(video), media.new_project(out, "source"))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    step = max(0.5, dur / 300)
    ff, scored, t = FaceFinder(), [], 0.5
    while t < dur - 0.3:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(cv2.resize(gray, (640, 360)), cv2.CV_64F).var()
        expo = 1 - abs(gray.mean() - 128) / 128
        faces = ff.find(f)
        fs = 0.0
        if faces:
            x, y, fw, fh, sc = faces[0]
            fs = sc * min(1.0, (fw * fh) / (w * h) * 12)
        score = 2.0 * fs + 0.6 * min(1.0, sharp / 300) + 0.4 * expo
        scored.append((score, t))
        t += step
    cap.release()
    picks = []
    for s, tt in sorted(scored, reverse=True):
        if all(abs(tt - p) > max(3.0, dur / (n * 3)) for _, p in picks):
            picks.append((s, tt))
        if len(picks) >= n:
            break
    picks.sort(key=lambda x: x[1])
    cap = cv2.VideoCapture(str(video))
    files, tiles = [], []
    for k, (s, tt) in enumerate(picks, 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000)
        ok, f = cap.read()
        p = out / f"cand_{k:02d}_{tt:.1f}s.jpg"
        cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        files.append({"n": k, "time": round(tt, 1), "score": round(s, 2), "file": str(p)})
        tile = cv2.resize(f, (384, int(384 * f.shape[0] / f.shape[1])))
        cv2.putText(tile, f"#{k} {tt:.0f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(tile, f"#{k} {tt:.0f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(tile)
    cap.release()
    rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4]))) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(out / "candidates_sheet.jpg"), np.vstack(rows))
    return {"sheet": str(out / "candidates_sheet.jpg"), "frames": files}


def _cover(im, w, h):
    s = max(w / im.width, h / im.height)
    im = im.resize((int(im.width * s + 0.5), int(im.height * s + 0.5)), Image.LANCZOS)
    l, t = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((l, t, l + w, t + h))


def _background(frame_pil, bg):
    if bg.startswith("#"):
        a = Image.new("RGB", (TW, TH), hex_rgb(bg)[:3])
        g = Image.linear_gradient("L").rotate(90).resize((TW, TH))
        dark = Image.new("RGB", (TW, TH), (0, 0, 0))
        return Image.composite(a, Image.blend(a, dark, 0.45), g)
    b = _cover(frame_pil, TW, TH).filter(ImageFilter.GaussianBlur(18))
    b = ImageEnhance.Color(b).enhance(1.4)
    return ImageEnhance.Brightness(b).enhance(0.45 if bg == "dark" else 0.7)


def _outline(rgba, px=10, color=(255, 255, 255)):
    a = np.array(rgba.split()[-1])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    grown = cv2.dilate(a, k)
    pad = Image.new("RGBA", rgba.size, color + (0,))
    pad.putalpha(Image.fromarray(grown))
    glow = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    glow.putalpha(Image.fromarray(cv2.GaussianBlur(grown, (0, 0), px * 1.6)))
    out = Image.alpha_composite(glow, pad)
    return Image.alpha_composite(out, rgba)


def make(frame, text, style, out, frame2=None, labels=None, bg="blur", accent="#FFD400", side="right",
         face_edit=None, face_prompt=None):
    bgr = cv2.imread(str(frame))
    if face_edit:   # generative face edit (own face / with consent only)
        from vibekit.faceedit import edit_image
        bgr, _, _ = edit_image(bgr, face_edit, face_prompt)
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    canvas = _background(pil, bg).convert("RGBA")
    big = dict(font="Anton.ttf", color="#FFFFFF", accent=accent, stroke=10, upper=True, shadow=True)
    if style == "face":
        pm = person_mask(bgr)
        if not bg.startswith("#"):
            # rebuild the background without the person (no blurry "ghost" behind the cut-out)
            sm = cv2.resize(bgr, (480, int(480 * bgr.shape[0] / bgr.shape[1])))
            hole = cv2.dilate((cv2.resize(pm, (sm.shape[1], sm.shape[0])) > 0.3).astype(np.uint8) * 255,
                              np.ones((25, 25), np.uint8))
            clean = cv2.inpaint(sm, hole, 12, cv2.INPAINT_TELEA)
            canvas = _background(Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)), bg).convert("RGBA")
        # frame head-and-shoulders around the main face, so the face is big and desks/props are left out
        fb = FaceFinder().find(bgr)
        if fb:
            fx, fy, fw, fh = fb[0][:4]
            h0, w0 = bgr.shape[:2]
            x0, x1 = int(max(0, fx + fw / 2 - fw * 2.1)), int(min(w0, fx + fw / 2 + fw * 2.1))
            y0, y1 = int(max(0, fy - fh * 0.75)), int(min(h0, fy + fh * 3.2))
            pm = pm.copy()
            keep = np.zeros_like(pm)
            keep[y0:y1, x0:x1] = 1
            # fade the bottom edge so the cut-out ends softly at the chest
            ramp = np.clip((y1 - np.arange(h0)) / max(1, fh * 0.6), 0, 1)[:, None]
            pm = pm * keep * ramp
        person = _outline(to_rgba(bgr, pm))
        s = TH * 1.02 / person.height
        person = person.resize((int(person.width * s), int(person.height * s)), Image.LANCZOS)
        if person.width > TW * 0.6:
            s2 = TW * 0.6 / person.width
            person = person.resize((int(person.width * s2), int(person.height * s2)), Image.LANCZOS)
        px = TW - person.width + 20 if side == "right" else -20
        canvas.alpha_composite(person, (px, TH - person.height + 10))
        txt = render_text(text, size=150, max_width=int(TW * 0.5), align="left", **big)
        tx = 50 if side == "right" else TW - txt.width - 50
        canvas.alpha_composite(txt, (tx, max(20, (TH - txt.height) // 2)))
    elif style == "split":
        b2 = cv2.imread(str(frame2 or frame))
        L = _cover(pil, TW // 2, TH)
        L = ImageEnhance.Color(L).enhance(0.25)
        L = ImageEnhance.Brightness(L).enhance(0.75)
        R = _cover(Image.fromarray(cv2.cvtColor(b2, cv2.COLOR_BGR2RGB)), TW // 2, TH)
        R = ImageEnhance.Color(R).enhance(1.35)
        canvas = Image.new("RGBA", (TW, TH))
        canvas.paste(L, (0, 0)); canvas.paste(R, (TW // 2, 0))
        d = ImageDraw.Draw(canvas)
        d.rectangle([TW // 2 - 5, 0, TW // 2 + 5, TH], fill=(255, 255, 255, 255))
        cx, cy = TW // 2, TH // 2
        d.polygon([(cx - 70, cy - 60), (cx + 30, cy - 60), (cx + 30, cy - 105), (cx + 110, cy), (cx + 30, cy + 105),
                   (cx + 30, cy + 60), (cx - 70, cy + 60)], fill=hex_rgb(accent), outline=(0, 0, 0, 255), width=6)
        if labels:
            a, b = (labels.split("|") + [""])[:2]
            la = render_text(a, size=86, max_width=560, **big)
            lb = render_text(b, size=86, max_width=560, **{**big, "color": accent})
            canvas.alpha_composite(la, ((TW // 2 - la.width) // 2, TH - la.height - 30))
            canvas.alpha_composite(lb, (TW // 2 + (TW // 2 - lb.width) // 2, TH - lb.height - 30))
        if text:
            tt = render_text(text, size=110, max_width=int(TW * 0.9), **big)
            canvas.alpha_composite(tt, ((TW - tt.width) // 2, 24))
    else:  # result
        ff = FaceFinder().find(bgr)
        h, w = bgr.shape[:2]
        cx, cy = (ff[0][0] + ff[0][2] / 2, ff[0][1] + ff[0][3] / 2) if ff else (w / 2, h / 2)
        zw = w / 1.35; zh = zw * TH / TW
        x0 = int(min(max(cx - zw / 2, 0), w - zw)); y0 = int(min(max(cy - zh * 0.4, 0), h - zh))
        z = pil.crop((x0, y0, int(x0 + zw), int(y0 + zh))).resize((TW, TH), Image.LANCZOS)
        z = ImageEnhance.Contrast(ImageEnhance.Color(z).enhance(1.3)).enhance(1.12)
        canvas = z.convert("RGBA")
        vg = Image.radial_gradient("L").resize((TW, TH)).point(lambda v: int(v * 0.55))
        canvas = Image.composite(Image.new("RGBA", (TW, TH), (0, 0, 0, 255)), canvas, vg)
        tt = render_text(text, size=130, max_width=int(TW * 0.9), box="#000000C8", box_pad=(30, 14), **big)
        canvas.alpha_composite(tt, ((TW - tt.width) // 2, TH - tt.height - 28))
    canvas.convert("RGB").save(out, quality=95)
    return str(out)


def compare(files, out):
    ims = [Image.open(f).convert("RGB").resize((TW // 2, TH // 2)) for f in files]
    small = [Image.open(f).convert("RGB").resize((246, 138)) for f in files]
    W = len(ims) * (TW // 2 + 20) + 20
    sheet = Image.new("RGB", (W, TH // 2 + 220), (24, 24, 24))
    for i, (a, b) in enumerate(zip(ims, small)):
        x = 20 + i * (TW // 2 + 20)
        sheet.paste(a, (x, 20)); sheet.paste(b, (x, TH // 2 + 50))
    sheet.save(out)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates"); c.add_argument("video"); c.add_argument("--out", required=True); c.add_argument("--n", type=int, default=12)
    m = sub.add_parser("make")
    for k in ("--frame", "--text", "--style", "--out"):
        m.add_argument(k, required=k != "--text", default="" if k == "--text" else None)
    m.add_argument("--frame2"); m.add_argument("--labels"); m.add_argument("--bg", default="blur")
    m.add_argument("--accent", default="#FFD400"); m.add_argument("--side", default="right")
    m.add_argument("--face-edit", choices=["beard", "stubble", "shave", "custom"],
                   help="generative face edit on the person (own face or with consent only)")
    m.add_argument("--face-prompt", help="custom description for --face-edit custom")
    k = sub.add_parser("compare"); k.add_argument("files", nargs="+"); k.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "candidates":
        print(json.dumps(candidates(a.video, a.out, a.n), indent=1))
    elif a.cmd == "make":
        print(make(a.frame, a.text, a.style, a.out, a.frame2, a.labels, a.bg, a.accent, a.side, a.face_edit, a.face_prompt))
    else:
        print(compare(a.files, a.out))


if __name__ == "__main__":
    main()
