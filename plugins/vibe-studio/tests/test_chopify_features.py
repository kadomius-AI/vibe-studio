"""Brand presets, moment scoring and the batch queue (features adapted from Chopify, MIT)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vibekit import brand as BR, moments
from vibekit.env import FFMPEG

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


# ---------------- brand presets ----------------
def test_brand_validation_messages():
    with pytest.raises(BR.BrandError, match="colour"):
        BR.validate({"highlight": "yellow"})
    with pytest.raises(BR.BrandError, match="between"):
        BR.validate({"size": 999})
    with pytest.raises(BR.BrandError, match="Unknown brand fields"):
        BR.validate({"sparkles": True})
    with pytest.raises(BR.BrandError, match="not found"):
        BR.validate({"font": "Comic Sans.ttf"})


def test_brand_layering(tmp_path, monkeypatch):
    monkeypatch.setattr(BR, "BRAND_DIR", tmp_path / "brands")
    BR.save("acme", {"base": "punch", "highlight": "#39FF88", "size": 90})
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "brand.json").write_text(json.dumps({"size": 70}))           # project override wins
    b = BR.resolve("acme", proj)
    assert b["base"] == "punch" and b["highlight"] == "#39FF88" and b["size"] == 70
    st = BR.caption_style(b)
    assert st["font"] == "Anton.ttf" and st["size"] == 70 and st["highlight"] == "#39FF88"


def test_logo_stays_in_safe_area(tmp_path):
    from PIL import Image
    p = tmp_path / "logo.png"
    Image.new("RGBA", (400, 200), (255, 0, 0, 255)).save(p)
    for pos in ("top-left", "top-right", "bottom-left", "bottom-right"):
        im, x, y = BR.logo_overlay(BR.validate({"logo": str(p), "logo_position": pos}) | {"logo_width": 170}, 1080, 1920)
        assert 0 <= x and x + im.width <= 1080 and 0.08 * 1920 <= y and y + im.height <= 0.78 * 1920


# ---------------- moment scoring ----------------
def _words(text, start=0.0, wps=2.6):
    out, t = [], start
    for w in text.split():
        out.append({"word": w, "start": round(t, 2), "end": round(t + 0.3, 2)})
        t += 1 / wps
    return out


def test_text_score_prefers_hooks_and_payoffs():
    good = moments.sentences(_words("Here's the thing nobody tells you about money. Most people quit right before it works. "
                                    "That's why the lesson is simple: never stop at the hard part."))
    bad = moments.sentences(_words("Welcome back to the channel, um, so, uh, this is, like, you know, a video. "
                                   "And it, um, uh, is what it is I guess."))
    assert moments.text_score(good)[0] > moments.text_score(bad)[0] + 0.15


def test_audio_reaction_scores_higher(tmp_path):
    """A burst of loud audio (laughter/applause stand-in) should raise the audio score of that window."""
    p = tmp_path / "a.wav"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=40",
                    "-f", "lavfi", "-i", "anoisesrc=d=40:a=0.8", "-filter_complex",
                    "[0:a]volume=-26dB[s];[1:a]volume='if(between(t,28,34),1,0)':eval=frame[n];[s][n]amix=inputs=2:normalize=0",
                    str(p)], check=True)
    tl = moments.audio_timeline(p)
    calm, _ = moments.audio_score(tl, 2, 22)
    loud, parts = moments.audio_score(tl, 20, 38)
    assert loud > calm + 0.2 and parts["bursts"] > 0.3


def test_find_returns_non_overlapping_ranked(wide_talking_head):
    w = _words("Here's the thing nobody tells you. The secret is consistency. That's why it works. " * 2, start=0.2, wps=6)
    found = moments.find(wide_talking_head, w, min_len=1.0, max_len=3.5, top=4)
    assert found and all(found[i]["score"] >= found[i + 1]["score"] for i in range(len(found) - 1))
    for i, a in enumerate(found):
        for b in found[i + 1:]:
            assert a["end"] <= b["start"] or a["start"] >= b["end"]


# ---------------- batch queue ----------------
def test_batch_continues_after_a_failure(tmp_path, wide_talking_head):
    q = tmp_path / "queue.txt"
    q.write_text(f"{wide_talking_head}\n/definitely/missing/video.mp4\n")
    r = subprocess.run([sys.executable, str(SCRIPTS / "clips.py"), "batch", str(q), "--root", str(tmp_path / "projects"),
                        "--min", "1", "--max", "3"], capture_output=True, text=True, timeout=900)
    rep = json.loads((tmp_path / "batch_report.json").read_text())
    assert [i["status"] for i in rep["items"]] == ["ok", "failed"], r.stderr[-1500:]
    assert "FileNotFoundError" in rep["items"][1]["error"]


def test_outros_are_not_top_moments():
    """Regression: a sign-off ('hope you have an amazing Thursday') must score below real content."""
    outro = moments.sentences(_words("That's just been my experience and hopefully that provides value for you. "
                                     "So otherwise, hope you have an amazing Thursday, catch you on the flip side."))
    content = moments.sentences(_words("The worst thing in the world is not having any flow. "
                                       "Once I get flow, I build offers on top of it, which is why I make more money."))
    assert moments.text_score(content)[0] > moments.text_score(outro)[0] + 0.1
