"""One pass over a clip: reframe (follow the face), apply look filters, burn captions + headline,
normalize loudness to -14 LUFS and encode for social (H.264 / AAC / 30 fps / faststart)."""
import subprocess
import cv2
import numpy as np
from .env import FFMPEG, probe
from .text import render_text, paste
from . import faces as F
from .captions import CaptionTrack

SIZES = {"vertical": (1080, 1920), "square": (1080, 1080), "horizontal": (1920, 1080), "portrait": (1080, 1350)}


def render(src, out, *, fmt="vertical", words=None, caption_style="bold", caption_y=None,
           headline=None, headline_secs=3.0, zoom=1.0, eye_y=0.38, prefer_x=None, follow_face=True,
           filters=(), music=None, music_db=-22, loudnorm=True, crf=18, punch=True, sfx=True, pre_filters=()):
    info = probe(src)
    W, H = SIZES[fmt]
    aspect = W / H
    tr = F.track(src, prefer_x=prefer_x) if follow_face else {
        "fps": info["fps"], "n": int(info["duration"] * info["fps"]) + 1, "w": info["width"], "h": info["height"],
        "samples": [], "cuts": []}
    boxes = F.crop_path(tr, out_aspect=aspect, zoom=zoom, eye_y=eye_y)
    caps = CaptionTrack(words, W, H, style=caption_style, y=caption_y) if words else None
    head = None
    if headline:
        head = render_text(headline.replace("|", "\n"), font="Montserrat-ExtraBold.ttf", size=int(64 * W / 1080),
                           max_width=int(W * 0.86), box="#0C0C0EC8", stroke=0)
    fps = tr["fps"]
    n_frames = len(boxes)
    # punch-in zooms: jump tighter at new sentences / emphasis (every few seconds), back out at the next one
    zoom_mul = np.ones(n_frames, np.float32)
    punch_times = []
    if punch and words:
        last, level = -9.0, 0
        for k, w in enumerate(words):
            prev_end = words[k - 1]["word"].strip()[-1:] if k else "."
            if prev_end in ".!?" and w["start"] - last > 2.8 and w["start"] > 0.8:
                level = 1 - level
                punch_times.append((w["start"], level))
                last = w["start"]
        for t0, lv in punch_times:
            zoom_mul[int(t0 * fps):] = 1.12 if lv else 1.0
    import tempfile, os
    tmp_vid = str(out) + ".video.mp4"
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:.5f}",
           "-i", "-", "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-profile:v", "high", tmp_vid]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(src))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for pf in pre_filters:          # full-frame edits (e.g. face edits) before reframing
            frame = pf(frame, i / fps)
        x, y, cw, ch = boxes[min(i, len(boxes) - 1)]
        z = zoom_mul[min(i, n_frames - 1)]
        if z != 1.0:  # tighten around the face line, stay inside the frame
            cx, cy = x + cw / 2, y + ch * eye_y
            cw2, ch2 = cw / z, ch / z
            x = min(max(cx - cw2 / 2, 0), info["width"] - cw2)
            y = min(max(cy - ch2 * eye_y, 0), info["height"] - ch2)
            cw, ch = cw2, ch2
        M = cv2.getAffineTransform(
            np.float32([[x, y], [x + cw, y], [x, y + ch]]),
            np.float32([[0, 0], [W, 0], [0, H]]))
        out_f = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        t = i / fps
        for flt in filters:
            out_f = flt(out_f, t)
        if caps:
            c = caps.at(t)
            if c:
                paste(out_f, *c)
        if head is not None and t < headline_secs:
            op = min(1.0, (headline_secs - t) / 0.35)
            paste(out_f, head, (W - head.width) / 2, int(H * 0.085), opacity=op)
        proc.stdin.write(out_f.tobytes())
        i += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding " + str(out))
    # ---- audio: original sound + ducked music + subtle sound effects ----
    from . import sound
    total = i / fps
    bed = sound.load(src) if info["has_audio"] else None
    mus = None
    if music:
        mus = sound.load(music)
    events = [(t0 - 0.05, "swish_up", -24) for t0, _ in punch_times] if sfx else []
    if head is not None and sfx:
        events.append((0.0, "pop", -20))
    if bed is None and mus is None and not events:
        os.replace(tmp_vid, out)
        return out
    mixed = sound.mix(total, bed=bed, music=mus, sfx_events=events, music_db=music_db)
    wav = str(out) + ".mix.wav"
    sound.save_wav(wav, mixed)
    af = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"] if loudnorm else []
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", tmp_vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    *af, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)], check=True)
    os.remove(wav); os.remove(tmp_vid)
    return out
