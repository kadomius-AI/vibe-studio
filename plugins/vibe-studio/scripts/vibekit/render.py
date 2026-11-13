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
           filters=(), music=None, music_db=-22, loudnorm=True, crf=18):
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
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:.5f}",
           "-i", "-"]
    inputs = 1
    if info["has_audio"]:
        cmd += ["-i", str(src)]
        inputs += 1
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        inputs += 1
    af = []
    if info["has_audio"] and music:
        af.append(f"[2:a]volume={music_db}dB,afade=t=in:d=0.6[m];[1:a][m]amix=inputs=2:duration=first:normalize=0[mix]")
        alabel = "[mix]"
    elif info["has_audio"]:
        alabel = "1:a"
    elif music:
        alabel = f"{inputs - 1}:a"
    else:
        alabel = None
    if alabel and loudnorm:
        src_lbl = alabel if alabel.startswith("[") else f"[{alabel}]"
        af.append(f"{src_lbl}loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
        alabel = "[aout]"
    if af:
        cmd += ["-filter_complex", ";".join(af)]
    cmd += ["-map", "0:v"]
    if alabel:
        cmd += ["-map", alabel if alabel.startswith("[") else alabel, "-c:a", "aac", "-b:a", "192k"]
        if not info["has_audio"]:
            cmd += ["-shortest"]
    cmd += ["-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-profile:v", "high", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(src))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x, y, cw, ch = boxes[min(i, len(boxes) - 1)]
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
    return out
