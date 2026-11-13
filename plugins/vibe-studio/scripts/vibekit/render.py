"""One pass over a clip: reframe, restore, grade, captions + headline, then mix audio and encode once.

Layouts:
  crop  — face-tracked crop (smooth, snaps at scene cuts), optional punch-in zooms
  fill  — the whole source frame over a blurred, darkened copy of itself (for close-up webcam / wide shots
          where a crop would be too tight or lose context); captions and headline go in the free bands
  auto  — picks crop or fill from the face size

Quality: video is encoded once (raw frames -> H.264 CRF 17), source frame rate kept (capped at 60),
audio is mixed in float and muxed with stream copy for the video. A render manifest (.render.json) records
caption and face boxes per sampled frame for automated QA.
"""
import json
import os
import subprocess
import cv2
import numpy as np
from pathlib import Path

from .env import FFMPEG, probe
from .text import render_text, paste
from . import faces as F
from . import layout as LY
from .captions import CaptionTrack
from .log import get as _log, ffmpeg as _ffmpeg, FFmpegError

SIZES = {"vertical": (1080, 1920), "square": (1080, 1080), "horizontal": (1920, 1080), "portrait": (1080, 1350)}
log = _log("render")


def _crop(frame, box, W, H, sr=None, restorer=None):
    """Cut box=(x,y,w,h) from frame and fit to W x H; optionally super-resolve + restore faces at native size."""
    x, y, cw, ch = box
    if sr is not None and cw < W * 0.75:
        nw, nh = max(8, int(round(cw))), max(8, int(round(ch)))
        Mn = cv2.getAffineTransform(np.float32([[x, y], [x + cw, y], [x, y + ch]]), np.float32([[0, 0], [nw, 0], [0, nh]]))
        native = cv2.warpAffine(frame, Mn, (nw, nh), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        big = sr(native)
        if restorer is not None:
            big = restorer(big)
        return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA if big.shape[1] > W else cv2.INTER_LANCZOS4)
    M = cv2.getAffineTransform(np.float32([[x, y], [x + cw, y], [x, y + ch]]), np.float32([[0, 0], [W, 0], [0, H]]))
    out = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return restorer(out) if restorer is not None else out


def _fill(frame, W, H, geom, sr=None, restorer=None, box=None):
    """The picture (4:3 framed around the face) over a dark, soft backdrop."""
    fx, fy, fw, fh = geom
    if box is not None:                      # frame the picture around the face (full source height)
        x, y, cw, ch = box
        frame_c = frame[int(y):int(y + ch), int(x):int(x + cw)]
    else:
        frame_c = frame
    back = cv2.resize(frame, (int(H * frame.shape[1] / frame.shape[0]), H), interpolation=cv2.INTER_AREA)
    x0 = (back.shape[1] - W) // 2
    back = back[:, x0:x0 + W]
    back = cv2.GaussianBlur(cv2.resize(back, (W // 8, H // 8)), (0, 0), 5)
    back = cv2.cvtColor(cv2.resize(back, (W, H)), cv2.COLOR_BGR2HSV)
    back[..., 1] = (back[..., 1] * 0.6).astype(np.uint8)
    back = (cv2.cvtColor(back, cv2.COLOR_HSV2BGR) * 0.28).astype(np.uint8)   # dark + soft: no distracting face
    fg = frame_c
    if sr is not None and frame_c.shape[1] < fw * 0.75:
        fg = sr(frame_c)
    if restorer is not None:
        fg = restorer(fg)
    fg = cv2.resize(fg, (fw, fh), interpolation=cv2.INTER_AREA if fg.shape[1] > fw else cv2.INTER_LANCZOS4)
    out = back
    y0, y1 = max(0, fy), min(H, fy + fh)
    out[y0:y1, fx:fx + fw] = fg[y0 - fy:y1 - fy]
    # soft shadow lines above/below the picture
    for yy, sgn in ((y0, -1), (y1, 1)):
        for k in range(1, 18):
            r = yy + sgn * k
            if 0 <= r < H:
                out[r] = (out[r] * (0.55 + 0.45 * k / 18)).astype(np.uint8)
    return out


def render(src, out, *, fmt="vertical", words=None, caption_style="bold", caption_y=None,
           headline=None, headline_secs=3.0, zoom=1.0, eye_y=0.38, prefer_x=None, follow_face=True,
           filters=(), music=None, music_db=-22, loudnorm=True, crf=17, punch=True, sfx=True, pre_filters=(),
           upscale=False, restore_faces=0.0, layout="auto", fps_out=None, brand=None):
    info = probe(src)
    W, H = SIZES[fmt]
    aspect = W / H
    tr = F.track(src, prefer_x=prefer_x) if follow_face else {
        "fps": info["fps"], "n": int(info["duration"] * info["fps"]) + 1, "w": info["width"], "h": info["height"],
        "samples": [], "cuts": []}
    fps = tr["fps"]
    fps_out = fps_out or min(60.0, fps)

    # ---- layout decision ----
    fh = [s[3] / info["height"] for s in tr["samples"] if s[3]]
    face_frac = float(np.median(fh)) if fh else None
    chosen = layout if layout in ("crop", "fill") else LY.choose_layout(face_frac, info["width"], info["height"], W, H, 1.0)
    if chosen == "crop" and face_frac and zoom > 1.0 and face_frac * zoom > LY.MAX_FACE_IN_CROP:
        new_zoom = max(1.0, LY.MAX_FACE_IN_CROP / face_frac)       # too tight with this zoom: zoom out, keep the crop
        log.info("zoom %.2f -> %.2f so the face isn't cropped too tight", zoom, new_zoom)
        zoom = new_zoom
    geom = LY.fill_geometry(info["width"], info["height"], W, H) if chosen == "fill" else None
    log.info("layout: %s (face height %.0f%% of source)", chosen, 100 * face_frac if face_frac else -1)
    if chosen == "fill":
        boxes = F.crop_path(tr, out_aspect=geom[2] / geom[3], zoom=1.0, eye_y=0.45)
    else:
        boxes = F.crop_path(tr, out_aspect=aspect, zoom=zoom, eye_y=eye_y)

    # ---- AI detail restoration ----
    sr = restorer = None
    if upscale:
        from .restore import upscale as sr
    if restore_faces > 0:
        from .restore import FaceRestorer
        restorer = FaceRestorer(restore_faces)

    logo = None
    if brand:
        from . import brand as BR
        caption_style = BR.caption_style(brand)
        if caption_y is None and brand.get("y") is not None:
            caption_y = brand["y"]
        logo = BR.logo_overlay(brand, W, H)
    caps = CaptionTrack(words, W, H, style=caption_style, y=caption_y) if words else None
    head = None
    if headline:
        head = render_text(headline.replace("|", "\n"), font="Montserrat-ExtraBold.ttf", size=int(70 * W / 1080),
                           max_width=int(W * 0.84), box=(brand or {}).get("headline_box") or "#0C0C0ED0", stroke=0,
                           box_pad=(40, 24))
    n_frames = len(boxes)
    zoom_mul = np.ones(n_frames, np.float32)
    punch_times = []
    if punch and words and chosen == "crop":
        last, level = -9.0, 0
        for k, w in enumerate(words):
            prev_end = words[k - 1]["word"].strip()[-1:] if k else "."
            if prev_end in ".!?" and w["start"] - last > 2.8 and w["start"] > 0.8:
                level = 1 - level
                punch_times.append((w["start"], level))
                last = w["start"]
        for t0, lv in punch_times:
            zoom_mul[int(t0 * fps):] = 1.12 if lv else 1.0

    tmp_vid = str(out) + ".video.mp4"
    errf = open(str(out) + ".ffmpeg.log", "w")
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", f"{fps:.5f}", "-i", "-", "-vf", f"fps={fps_out:.5f}", "-c:v", "libx264", "-preset", "slow",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high", tmp_vid]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
    cap = cv2.VideoCapture(str(src))
    ff = F.FaceFinder()
    manifest = {"layout": chosen, "size": [W, H], "fps": fps_out, "captions": [], "faces": [], "headline": None}
    face_box = None
    head_box = None
    if head is not None:
        hy = int(H * LY.SAFE["top"]) + int(H * 0.01) if chosen == "crop" or not geom else LY.fill_headline_y(head.height, H, geom)
        head_box = ((W - head.width) // 2, hy, head.width, head.height)
        manifest["headline"] = list(head_box)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            for pf in pre_filters:
                frame = pf(frame, t)
            if chosen == "fill":
                out_f = _fill(frame, W, H, geom, sr, restorer, box=boxes[min(i, n_frames - 1)])
            else:
                x, y, cw, ch = boxes[min(i, n_frames - 1)]
                z = zoom_mul[min(i, n_frames - 1)]
                if z != 1.0:
                    cx, cy = x + cw / 2, y + ch * eye_y
                    cw, ch = cw / z, ch / z
                    x = min(max(cx - cw / 2, 0), info["width"] - cw)
                    y = min(max(cy - ch * eye_y, 0), info["height"] - ch)
                out_f = _crop(frame, (x, y, cw, ch), W, H, sr, restorer)
            for flt in filters:
                out_f = flt(out_f, t)
            if i % 5 == 0:                       # where is the face in the OUTPUT frame? (for caption placement + QA)
                fs = ff.find(out_f)
                if chosen == "fill" and geom:        # only faces inside the picture count (not the soft backdrop)
                    fs = [f for f in fs if geom[1] <= f[1] + f[3] / 2 <= geom[1] + geom[3]]
                face_box = tuple(int(v) for v in fs[0][:4]) if fs else None
                if i % 15 == 0:
                    manifest["faces"].append({"t": round(t, 2), "box": list(face_box) if face_box else None})
            if logo is not None and not (head is not None and t < headline_secs and logo[2] < H * 0.25):
                paste(out_f, *logo)            # a top logo waits until the headline is gone (no overlap)
            if head is not None and t < headline_secs:
                op = min(1.0, (headline_secs - t) / 0.35, (t + 0.05) / 0.2)
                paste(out_f, head, head_box[0], head_box[1], opacity=op)
            if caps:
                c = caps.at(t)
                if c:
                    img, cx_, _ = c
                    avoid = [head_box] if (head is not None and t < headline_secs) else []
                    if chosen == "fill" and geom:     # keep captions off the picture's face area entirely
                        avoid.append((0, geom[1], W, geom[3]) if face_box is None else None)
                    pref = caption_y if caption_y is not None else (
                        min(LY.SAFE["bottom"] - 0.05, (geom[1] + geom[3]) / H + 0.08) if chosen == "fill" else caps.y)
                    if chosen == "fill" and geom and caption_y is None:
                        cy_ = LY.fill_caption_y(img.height, H, geom)
                    else:
                        cy_ = LY.caption_y(img.height, W, H, face_box, preferred=pref, avoid=[a for a in avoid if a])
                    paste(out_f, img, cx_, cy_)
                    if i % 15 == 0:
                        manifest["captions"].append({"t": round(t, 2), "box": [int(cx_), int(cy_), img.width, img.height]})
            proc.stdin.write(out_f.tobytes())
            i += 1
    finally:
        cap.release()
        proc.stdin.close()
        rc = proc.wait()
        errf.close()
    if rc != 0:
        msg = Path(str(out) + ".ffmpeg.log").read_text()[-1500:]
        raise FFmpegError("video encode failed: " + msg)
    os.remove(str(out) + ".ffmpeg.log")

    # ---- audio: original sound + ducked music + subtle sound effects, mixed once ----
    from . import sound
    total = i / fps
    bed = sound.load(src) if info["has_audio"] else None
    mus = sound.load(music) if music else None
    events = [(t0 - 0.05, "swish_up", -24) for t0, _ in punch_times] if sfx else []
    if head is not None and sfx:
        events.append((0.0, "pop", -20))
    manifest.update({"duration": round(total, 3), "has_audio": bool(bed is not None or mus is not None or events)})
    Path(str(out) + ".render.json").write_text(json.dumps(manifest))
    if bed is None and mus is None and not events:
        os.replace(tmp_vid, out)
        return out
    mixed = sound.mix(total, bed=bed, music=mus, sfx_events=events, music_db=music_db)
    wav = str(out) + ".mix.wav"
    sound.save_wav(wav, mixed)
    af = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"] if loudnorm else []
    _ffmpeg(["-y", "-i", tmp_vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", *af, "-c:a", "aac",
             "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-shortest", str(out)], what="mux audio")
    os.remove(wav)
    os.remove(tmp_vid)
    return out
