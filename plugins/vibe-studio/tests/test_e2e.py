"""End-to-end: render real videos from fixtures and run the automated QA on them."""
import json
from pathlib import Path

import pytest

from vibekit import checks, render as R


def test_qa_catches_black_and_wrong_size(black_video):
    rep = checks.run(black_video, fmt="vertical")
    assert not rep["passed"]
    assert not rep["checks"]["black"]["ok"]
    assert not rep["checks"]["dimensions"]["ok"]


def test_qa_catches_caption_outside_safe_area(wide_talking_head):
    man = {"size": [1080, 1920], "layout": "crop", "faces": [{"t": 0.0, "box": [300, 300, 400, 500]}],
           "captions": [{"t": 0.0, "box": [100, 1800, 800, 120]}]}
    rep = checks.run(wide_talking_head, manifest=man, strict_audio=False)
    assert not rep["checks"]["captions"]["ok"]


def test_render_vertical_crop_passes_qa(tmp_path, wide_talking_head, words):
    out = tmp_path / "v.mp4"
    R.render(wide_talking_head, out, fmt="vertical", words=words, headline="test|headline", punch=False, sfx=False)
    man = json.loads(Path(str(out) + ".render.json").read_text())
    assert man["layout"] == "crop"
    rep = checks.run(out, fmt="vertical", manifest=man, strict_audio=False)
    for k in ("render", "dimensions", "fps", "black", "captions", "crop"):
        assert rep["checks"][k]["ok"], (k, rep["checks"][k])


def test_render_webcam_uses_fill_layout(tmp_path, webcam_closeup, words):
    out = tmp_path / "w.mp4"
    R.render(webcam_closeup, out, fmt="vertical", words=words, punch=False, sfx=False)
    man = json.loads(Path(str(out) + ".render.json").read_text())
    assert man["layout"] == "fill"
    rep = checks.run(out, fmt="vertical", manifest=man, strict_audio=False)
    for k in ("render", "dimensions", "black", "captions"):
        assert rep["checks"][k]["ok"], (k, rep["checks"][k])


def test_render_keeps_audio_in_sync(tmp_path, wide_talking_head):
    from vibekit.env import probe
    out = tmp_path / "a.mp4"
    R.render(wide_talking_head, out, fmt="horizontal", punch=False, sfx=False)
    a, b = probe(wide_talking_head), probe(out)
    assert b["has_audio"] and abs(a["duration"] - b["duration"]) < 0.1
