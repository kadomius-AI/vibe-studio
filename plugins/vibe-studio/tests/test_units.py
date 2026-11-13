"""Fast unit tests for the shared engine (no rendering)."""
import json

import cv2
import numpy as np
import pytest

from vibekit import grade, layout, captions, plan, sound


# ---------------- grade ----------------
def _dark_portrait():
    img = cv2.imread(str(__import__("conftest").FIX / "face.jpg"))
    return (img * 0.45).astype(np.uint8)          # simulate an underexposed webcam


def test_grade_brightens_dark_face_within_bounds():
    img = _dark_portrait()
    box = (128, 100, 256, 300)
    st = grade.measure([img], [box])
    g = grade.Grade(st, {"face_L": 166, "sat": 0.32})
    out = g(img)
    lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[100:400, 128:384, 0].mean()
    lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[100:400, 128:384, 0].mean()
    assert lab_out > lab_in + 15, "dark face should get brighter"
    p = g.params()
    assert grade.BOUNDS["gamma"][0] <= p["gamma"] <= grade.BOUNDS["gamma"][1]
    assert abs(p["wb_a"]) <= grade.BOUNDS["wb"] and abs(p["wb_b"]) <= grade.BOUNDS["wb"]


def test_grade_never_crushes_blacks_or_clips_whites():
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (64, 1))
    img = cv2.merge([ramp, ramp, ramp])
    g = grade.Grade(grade.measure([img]), None)
    out = cv2.cvtColor(g(img), cv2.COLOR_BGR2LAB)[..., 0]
    assert out.min() >= 5 and out.max() <= 250


def test_grade_keeps_skin_hue():
    """Blue-cast room + a warm skin patch: white balance should fix the room but barely move the skin hue."""
    img = np.full((400, 400, 3), (150, 120, 105), np.uint8)          # bluish-grey wall (BGR)
    img[120:320, 140:260] = (95, 125, 175)                            # typical skin tone (BGR)
    g = grade.Grade(grade.measure([img], [(140, 120, 120, 200)]), {"face_L": 170, "sat": 0.4})
    out = g(img)
    def hue(im, ys, xs):
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)[ys, xs]
        return np.degrees(np.arctan2(lab[..., 2].mean() - 128, lab[..., 1].mean() - 128))
    skin = (slice(150, 300), slice(160, 240))
    assert abs(hue(out, *skin) - hue(img, *skin)) < 6, "skin hue should barely move"
    wall = (slice(10, 80), slice(10, 80))
    chroma = lambda im: float(np.hypot(*(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)[wall][..., 1:].reshape(-1, 2).mean(0) - 128)))
    assert chroma(out) < chroma(img), "the blue cast on the wall should be reduced"


# ---------------- layout ----------------
def test_choose_layout():
    assert layout.choose_layout(0.25, 1920, 1080, 1080, 1920) == "crop"
    assert layout.choose_layout(0.5, 640, 360, 1080, 1920) == "fill"      # webcam close-up
    assert layout.choose_layout(None, 1920, 1080, 1080, 1920) == "fill"
    assert layout.choose_layout(0.6, 1920, 1080, 1920, 1080) == "crop"    # same aspect: no crop problem


def test_caption_never_overlaps_face_and_stays_safe():
    W, H = 1080, 1920
    x0, top, x1, bottom = layout.safe_area(W, H)
    for face in [(300, 1100, 480, 520), (300, 400, 480, 600), (200, 900, 680, 700)]:
        y = layout.caption_y(140, W, H, face, preferred=0.70)
        assert top <= y and y + 140 <= bottom
        fx, fy, fw, fh = face
        if not (fy <= top + 10 and fy + fh >= bottom - 10):
            assert y + 140 <= fy or y >= fy + fh, f"caption at {y} overlaps face {face}"


# ---------------- captions ----------------
def test_chunking_breaks_on_punctuation_and_size(words=None):
    w = [{"word": x, "start": i * 0.3, "end": i * 0.3 + 0.2} for i, x in enumerate("one two three four. five six".split())]
    ch = captions.chunk(w, max_words=3)
    assert [len(c) for c in ch] == [3, 1, 2]


def test_caption_track_timing(words):
    tr = captions.CaptionTrack(words, 1080, 1920)
    assert tr.at(words[0]["start"] + 0.05) is not None
    assert tr.at(words[-1]["end"] + 5) is None
    img, x, y = tr.at(words[0]["start"] + 0.2)
    assert img.width <= 1080 * 0.9


# ---------------- plan ----------------
def test_plan_roundtrip_and_change_detection(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 5000)
    p = plan.new("test", {"source": src}, {"segments": [[0, 1.5]], "gain": np.float32(1.25)})
    path = plan.save(p, tmp_path / "p.json")
    back = plan.load(path)
    assert back["decisions"]["segments"] == [[0, 1.5]] and back["decisions"]["gain"] == pytest.approx(1.25)
    src.write_bytes(b"y" * 5000)
    with pytest.raises(ValueError):
        plan.load(path)


# ---------------- sound ----------------
def test_music_ducks_under_voice():
    sr = sound.SR
    voice = np.zeros((sr * 4, 2), np.float32)
    voice[sr:2 * sr] = 0.3 * np.sin(np.linspace(0, 2 * np.pi * 300, sr))[:, None]
    music = np.full((sr * 4, 2), 0.1, np.float32)
    mixed = sound.mix(4.0, voice=voice, music=music, music_db=0, duck_db=-12)
    quiet_music = np.abs(mixed[int(3.0 * sr):int(3.2 * sr)]).mean()
    under_voice = np.abs(mixed[int(1.4 * sr):int(1.6 * sr)] - voice[int(1.4 * sr):int(1.6 * sr)]).mean()
    assert under_voice < quiet_music * 0.5


def test_snap_durations_lands_on_beats():
    beats = [i * 0.5 for i in range(40)]
    d = sound.snap_durations([2.2, 1.9, 2.6], beats)
    t = np.cumsum(d)
    assert all(abs((x / 0.5) - round(x / 0.5)) < 1e-6 for x in t)


def test_fill_captions_stay_in_one_band():
    """Different caption heights must not flip between 'below the picture' and 'inside it'."""
    W, H = 1080, 1920
    geom = layout.fill_geometry(640, 360, W, H)
    centers = {layout.fill_caption_y(h, H, geom) + h / 2 for h in (96, 110, 124, 138, 150)}
    assert max(centers) - min(centers) <= 1
    assert min(layout.fill_caption_y(h, H, geom) for h in (96, 150)) >= geom[1] + geom[3]   # under the picture
    assert layout.fill_headline_y(150, H, geom) + 150 <= geom[1]                            # headline above it


def test_zoom_does_not_force_fill_layout():
    """A studio shot (face 28% of height) stays a crop even if a big zoom was requested."""
    assert layout.choose_layout(0.28, 1920, 1080, 1080, 1920, 1.0) == "crop"
