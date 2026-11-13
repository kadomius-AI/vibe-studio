"""Logging and a safe FFmpeg runner with readable errors.

Every run logs to the console (short) and to ~/.local/share/vibekit/logs/vibe.log (full detail).
"""
import logging
import subprocess
import time
from pathlib import Path

from .env import DATA, FFMPEG

_LOG_DIR = DATA / "logs"
_configured = False


def get(name="vibekit"):
    global _configured
    if not _configured:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger("vibekit")
        root.setLevel(logging.DEBUG)
        fh = logging.FileHandler(_LOG_DIR / "vibe.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("· %(message)s"))
        root.addHandler(fh)
        root.addHandler(ch)
        root.propagate = False
        _configured = True
    return logging.getLogger(name if name.startswith("vibekit") else f"vibekit.{name}")


class FFmpegError(RuntimeError):
    """FFmpeg failed. The message carries the last lines of FFmpeg's own explanation."""


def ffmpeg(args, *, what="ffmpeg", input_bytes=None, capture=False, check=True):
    """Run FFmpeg with logging. args excludes the binary. Returns CompletedProcess."""
    log = get("ffmpeg")
    cmd = [FFMPEG, "-hide_banner"] + (["-nostdin"] if input_bytes is None else []) + list(map(str, args))
    t0 = time.time()
    log.debug("%s: %s", what, " ".join(map(str, cmd)))
    r = subprocess.run(cmd, input=input_bytes, capture_output=True)
    dt = time.time() - t0
    if r.returncode != 0 and check:
        tail = "\n".join(r.stderr.decode(errors="replace").strip().splitlines()[-8:])
        log.error("%s failed after %.1fs:\n%s", what, dt, tail)
        raise FFmpegError(f"{what} failed: {tail}")
    log.debug("%s done in %.1fs", what, dt)
    return r


def encoder_args(crf=17, preset="medium", fps=None):
    """High-quality, widely compatible H.264 settings for final deliverables."""
    a = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-movflags", "+faststart"]
    if fps:
        a = ["-r", f"{fps:.5f}"] + a
    return a


LOSSLESS = ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p"]  # intermediates: no quality loss
