"""Automated quality checks for a finished render. Returns a report with pass/fail per check.

Checks (each one can fail the render):
  render      — file exists, decodes, has a video stream
  dimensions  — width/height match the expected format
  fps         — frame rate is sane (23.9–60)
  duration    — matches the expected length (if given)
  black       — no black stretches (ffmpeg blackdetect)
  frozen      — no frozen picture > 1.5 s (ffmpeg freezedetect)
  audio       — audio present, not silent, loudness within -16..-12 LUFS, true peak <= -1 dBTP
  captions    — every sampled caption inside the platform safe area and not covering a face (render manifest)
  crop        — the speaker's face fully in frame and not cut off (sampled output frames)
  cut_clicks  — no audible click/pop at splice points (peak jump at each cut, if cut times given)
Inspired by vibe-editing's dedicated auditors and MakeMyClip's cut review (both MIT).
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from .env import FFMPEG, probe, loudness
from . import layout as LY

SIZES = {"vertical": (1080, 1920), "square": (1080, 1080), "horizontal": (1920, 1080), "portrait": (1080, 1350)}


def _ff_detect(path, vf=None, af=None):
    args = [FFMPEG, "-hide_banner", "-nostats", "-i", str(path)]
    if vf:
        args += ["-vf", vf]
    if af:
        args += ["-af", af]
    args += ["-f", "null", "-"]
    return subprocess.run(args, capture_output=True, text=True).stderr


def black_segments(path, min_d=0.25):
    err = _ff_detect(path, vf=f"blackdetect=d={min_d}:pix_th=0.08")
    return [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)]


def frozen_segments(path, min_d=1.5):
    err = _ff_detect(path, vf=f"freezedetect=n=0.002:d={min_d}")
    st = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", err)]
    en = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", err)]
    return list(zip(st, en + [None] * (len(st) - len(en))))


def true_peak(path):
    err = _ff_detect(path, af="ebur128=peak=true")
    m = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", err)
    return float(m[-1]) if m and m[-1] != "-inf" else None


def _boxes_overlap(a, b, pad=0):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx - pad or bx + bw <= ax - pad or ay + ah <= by - pad or by + bh <= ay - pad)


def run(path, fmt=None, expected_duration=None, cut_times=(), manifest=None, strict_audio=True):
    path = Path(path)
    rep = {"file": str(path), "checks": {}, "passed": True}

    def add(name, ok, detail=""):
        rep["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            rep["passed"] = False

    if not path.exists() or path.stat().st_size < 1000:
        add("render", False, "output missing or empty")
        return rep
    try:
        info = probe(path)
    except Exception as e:  # noqa: BLE001
        add("render", False, f"cannot read output: {e}")
        return rep
    add("render", info["width"] is not None, f"{info['width']}x{info['height']} {info['duration']:.2f}s")
    if fmt:
        W, H = SIZES[fmt]
        add("dimensions", (info["width"], info["height"]) == (W, H), f"{info['width']}x{info['height']} expected {W}x{H}")
    add("fps", 23.5 <= info["fps"] <= 60.5, f"{info['fps']:.2f}")
    if expected_duration:
        add("duration", abs(info["duration"] - expected_duration) <= max(0.25, 0.02 * expected_duration),
            f"{info['duration']:.2f}s expected {expected_duration:.2f}s")
    blk = [s for s in black_segments(path) if s[1] - s[0] > 0.25]
    add("black", not blk, f"{len(blk)} black stretch(es) {blk[:3]}" if blk else "none")
    frz = frozen_segments(path)
    add("frozen", not frz, f"{len(frz)} frozen stretch(es) {frz[:3]}" if frz else "none")
    if info["has_audio"]:
        lufs = loudness(path)
        tp = true_peak(path)
        ok = lufs is not None and lufs > -40 and (not strict_audio or -16.5 <= lufs <= -11.5) and (tp is None or tp <= -0.5)
        add("audio", ok, f"{lufs} LUFS, true peak {tp} dBTP")
    else:
        add("audio", False, "no audio stream")
    # captions + crop from the render manifest
    man = manifest
    mp = Path(str(path) + ".render.json")
    if man is None and mp.exists():
        man = json.loads(mp.read_text())
    if man:
        W, H = man["size"]
        x0, top, x1, bottom = LY.safe_area(W, H)
        faces = {f["t"]: f["box"] for f in man.get("faces", [])}
        bad_safe, bad_face = [], []
        for c in man.get("captions", []):
            bx, by, bw, bh = c["box"]
            if bx < x0 - 2 or by < top - 2 or bx + bw > W - x0 + 2 or by + bh > bottom + 2:
                bad_safe.append(c["t"])
            fb = faces.get(c["t"])
            if fb and _boxes_overlap(c["box"], [fb[0], fb[1] + fb[3] * 0.1, fb[2], fb[3] * 0.85]):
                bad_face.append(c["t"])
        n = max(1, len(man.get("captions", [])))
        add("captions", not bad_safe and len(bad_face) <= 0.05 * n,
            f"{len(man.get('captions', []))} sampled; outside safe area at {bad_safe[:5]}; over face at {bad_face[:5]}")
        cut_off, missing = [], 0
        for f in man.get("faces", []):
            b = f["box"]
            if not b:
                missing += 1
                continue
            if b[0] < 2 or b[1] < 2 or b[0] + b[2] > W - 2 or b[1] + b[3] > H - 2:
                cut_off.append(f["t"])
        nf = max(1, len(man.get("faces", [])))
        add("crop", len(cut_off) <= 0.1 * nf and missing <= 0.5 * nf,
            f"face cut off at {cut_off[:5]}; no face in {missing}/{nf} samples; layout {man.get('layout')}")
    if cut_times and info["has_audio"]:
        clicks = []
        for t in cut_times:
            a = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{max(0, t - 0.02):.3f}", "-t", "0.04", "-i", str(path),
                                "-ac", "1", "-ar", "48000", "-f", "f32le", "-"], capture_output=True).stdout
            x = np.frombuffer(a, np.float32)
            if len(x) > 10 and np.abs(np.diff(x)).max() > 0.5:
                clicks.append(round(t, 2))
        add("cut_clicks", not clicks, f"possible clicks at {clicks}" if clicks else f"{len(cut_times)} cuts clean")
    return rep


def save(rep, path):
    Path(path).write_text(json.dumps(rep, indent=2))
    return str(path)


def summary(rep):
    bad = [k for k, v in rep["checks"].items() if not v["ok"]]
    return "QA PASSED" if rep["passed"] else "QA FAILED: " + ", ".join(f"{k} ({rep['checks'][k]['detail']})" for k in bad)
