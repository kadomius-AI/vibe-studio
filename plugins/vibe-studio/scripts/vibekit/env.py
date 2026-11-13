"""Shared paths, tool discovery and one-time model downloads for the Vibe plugins."""
import os, shutil, subprocess, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent.parent                      # plugins/<name>/
FONTS = PLUGIN / "assets" / "fonts"
DATA = Path(os.environ.get("VIBEKIT_HOME", Path.home() / ".local" / "share" / "vibekit"))
MODELS = DATA / "models"

# Small, openly licensed models fetched on first use (never committed to git).
MODEL_URLS = {
    # OpenCV Zoo YuNet face detector (MIT)
    "yunet": ("face_detection_yunet_2023mar.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    # Google MediaPipe selfie segmenter (Apache-2.0)
    "selfie": ("selfie_segmenter.tflite",
               "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"),
}


def font(name="Montserrat-ExtraBold.ttf"):
    p = FONTS / name
    if not p.exists():
        raise FileNotFoundError(f"Font {name} not found in {FONTS}")
    return str(p)


def model_path(key):
    fname, url = MODEL_URLS[key]
    p = MODELS / fname
    if not p.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(p)
    return str(p)


def _find(tool):
    for cand in (f"/opt/homebrew/bin/{tool}", f"/usr/local/bin/{tool}", shutil.which(tool)):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError(f"{tool} is not installed. On a Mac: brew install ffmpeg")


FFMPEG = _find("ffmpeg")
FFPROBE = _find("ffprobe")
# mlx-whisper and yt-dlp call ffmpeg by name
os.environ["PATH"] = str(Path(FFMPEG).parent) + os.pathsep + os.environ.get("PATH", "")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def probe(path):
    import json
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    fps = 30.0
    if v and v.get("avg_frame_rate", "0/0") != "0/0":
        n, d = v["avg_frame_rate"].split("/")
        fps = float(n) / float(d) if float(d) else 30.0
    return {"width": v and v["width"], "height": v and v["height"], "fps": fps,
            "duration": float(j["format"].get("duration", 0)), "has_audio": a is not None}


def loudness(path):
    """Integrated loudness (LUFS) of a file's audio."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    import re
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None
