"""End-to-end ad render from a generated product photo (no voice/music: offline and fast)."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_ad_scaffold_and_render(tmp_path):
    img = np.full((1200, 1000, 3), 235, np.uint8)
    cv2.rectangle(img, (380, 300), (620, 1000), (60, 120, 200), -1)
    cv2.rectangle(img, (440, 200), (560, 300), (30, 30, 30), -1)
    prod = tmp_path / "product.jpg"
    cv2.imwrite(str(prod), img)
    od = tmp_path / "ads"
    r = subprocess.run([sys.executable, str(SCRIPTS / "ad.py"), "scaffold", "--product", str(prod), "--name", "Test Bottle",
                        "--benefit", "always cold water", "--features", "Steel|Leak proof|24h cold", "--no-voice",
                        "--no-music", "--out", str(od)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run([sys.executable, str(SCRIPTS / "ad.py"), "render", str(od / "ad_2.json")], capture_output=True, text=True)
    qa = json.loads((od / "ad_2.mp4.qa.json").read_text())
    for k in ("render", "dimensions", "fps", "black", "audio"):
        assert qa["checks"][k]["ok"], (k, qa["checks"][k], r.stderr[-1500:])
