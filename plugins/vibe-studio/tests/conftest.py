"""Shared test fixtures: small synthetic videos built at test time (no real footage in the repo)."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
FIX = Path(__file__).parent / "fixtures"

from vibekit.env import FFMPEG  # noqa: E402


def _write_video(path, frames, fps=30, tone_db=-20, seconds=None):
    h, w = frames[0].shape[:2]
    raw = path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    secs = seconds or len(frames) / fps
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(raw), "-f", "lavfi", "-i",
                    f"sine=frequency=220:duration={secs}", "-filter_complex", f"[1:a]volume={tone_db}dB[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(path)], check=True)
    raw.unlink()
    return path


def _face_frames(w, h, n, face_scale):
    face = cv2.imread(str(FIX / "face.jpg"))
    out = []
    for k in range(n):
        t = k / 30
        bg = np.full((h, w, 3), (70, 80, 95), np.uint8)
        s = int(h * face_scale * (1 + 0.03 * np.sin(t * 2)))
        f = cv2.resize(face, (s, s))
        x = int(w / 2 - s / 2 + 0.05 * w * np.sin(t * 1.5))
        y = int(h * 0.12)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + s), min(h, y + s)
        bg[y0:y1, x0:x1] = f[y0 - y:y1 - y, x0 - x:x1 - x]
        out.append(bg)
    return out


@pytest.fixture(scope="session")
def wide_talking_head(tmp_path_factory):
    """4 s, 1280x720, face ~40% of height, moving gently, with a tone track."""
    p = tmp_path_factory.mktemp("fx") / "wide.mp4"
    return _write_video(p, _face_frames(1280, 720, 120, 0.55))


@pytest.fixture(scope="session")
def webcam_closeup(tmp_path_factory):
    """4 s, 640x360 low-res close-up (face ~70% of height): should trigger the 'fill' layout."""
    p = tmp_path_factory.mktemp("fx") / "webcam.mp4"
    return _write_video(p, _face_frames(640, 360, 120, 0.95))


@pytest.fixture(scope="session")
def black_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("fx") / "black.mp4"
    return _write_video(p, [np.zeros((360, 640, 3), np.uint8)] * 90)


@pytest.fixture()
def words():
    """Fake word timings (as transcription would give)."""
    text = "This is a test of the captions. They should never cover the face or leave the safe area."
    out, t = [], 0.3
    for w in text.split():
        out.append({"word": w, "start": round(t, 2), "end": round(t + 0.22, 2)})
        t += 0.28
    return out
