"""Measure a reference video's editing style as numbers, so it can be matched within limits (not copied).

Profile:
  pacing    — cuts per minute, words per minute, pause share (silence between words)
  color     — face exposure, levels, white balance offsets, saturation (see grade.measure)
  framing   — median face height and face centre (as fractions of the frame)
  captions  — rough caption band position if burned-in text is visible (optional; Gemini when a key is set)
Deterministic by default; an optional vision-model pass (Gemini, GEMINI_API_KEY) can describe caption style and
transitions, recorded as notes in the plan. Never required.
"""
import os
import cv2
import numpy as np

from .faces import FaceFinder
from . import grade, media
from .env import probe


def _frames(video, n=24):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def cut_rate(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step, i, prev, cuts = max(1, int(fps / 5)), 0, None, 0
    while cap.grab():
        if i % step == 0:
            ok, f = cap.retrieve()
            h = cv2.calcHist([cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(h, h)
            if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > 0.45:
                cuts += 1
            prev = h
        i += 1
    cap.release()
    return round(cuts / max(i / fps / 60, 1e-6), 1)


def caption_band(frames):
    """Where burned-in text tends to sit (fraction of height), via edge density of bright/dark text strokes."""
    rows = []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (270, 480)) if f.shape[0] > f.shape[1] else cv2.resize(f, (480, 270)), cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 120, 240)
        rows.append(e.mean(1))
    prof = np.mean(rows, 0)
    prof = np.convolve(prof, np.ones(9) / 9, mode="same")
    k = int(np.argmax(prof[int(len(prof) * 0.45):])) + int(len(prof) * 0.45)
    strong = prof[k] > 2.5 * np.median(prof)
    return round(k / len(prof), 3) if strong else None


def profile(video, words=None):
    fr = _frames(video)
    ff = FaceFinder()
    boxes, fh, fx, fy = [], [], [], []
    for f in fr:
        fs = ff.find(f)
        b = fs[0][:4] if fs else None
        boxes.append(b)
        if b is not None:
            fh.append(b[3] / f.shape[0]); fx.append((b[0] + b[2] / 2) / f.shape[1]); fy.append((b[1] + b[3] / 2) / f.shape[0])
    info = probe(video)
    sil = media.silences(video, min_len=0.35) if info["has_audio"] else []
    prof = {
        "size": [info["width"], info["height"]], "fps": round(info["fps"], 3), "duration": round(info["duration"], 2),
        "pacing": {"cuts_per_min": cut_rate(video),
                   "pause_share": round(sum(e - s for s, e in sil) / max(info["duration"], 1), 3),
                   "words_per_min": round(len(words) / max(info["duration"] / 60, 1e-6), 1) if words else None},
        "color": grade.measure(fr, boxes),
        "framing": {"face_height": round(float(np.median(fh)), 3) if fh else None,
                    "face_x": round(float(np.median(fx)), 3) if fx else None,
                    "face_y": round(float(np.median(fy)), 3) if fy else None},
        "captions": {"band_y": caption_band(fr)},
        "vision_notes": None,
    }
    if os.environ.get("GEMINI_API_KEY"):
        prof["vision_notes"] = _gemini_notes(fr[:6])
    return prof


def _gemini_notes(frames):  # pragma: no cover - needs a key
    """Optional: ask Gemini to describe caption style / transitions from a few frames. Free-tier key only."""
    try:
        import base64, json, urllib.request
        parts = [{"text": "These are frames from a short-form video. In JSON, describe: caption_style (font weight, case, "
                          "colors, position), framing, color_look, on_screen_graphics. Be brief."}]
        for f in frames:
            ok, buf = cv2.imencode(".jpg", cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1]))))
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf).decode()}})
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            data=json.dumps({"contents": [{"parts": parts}]}).encode(),
            headers={"content-type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        return r["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:  # noqa: BLE001
        return f"vision notes unavailable: {e}"
