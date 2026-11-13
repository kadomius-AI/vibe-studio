#!/usr/bin/env python3
"""/ad — a product photo in, 5 short vertical video ads + ad copy out.

  ad.py prep PRODUCT.jpg --out DIR
      Tries a clean cut-out (cutout.png + preview), reports how clean it is and the product's main colors.

  ad.py scaffold --product PRODUCT.jpg --name "Glow Serum" --benefit "glass skin in 7 days"
                 --problem "dull, tired skin" --features "Vitamin C|No fragrance|Dermatologist tested"
                 [--price "$29"] [--offer "20% off this week"] [--cta "Shop now"] [--accent #FFD400] [--bg #111111]
                 [--photos extra1.jpg,extra2.jpg] [--music track.mp3] --out DIR
      Writes 5 ad specs (ad_1..ad_5.json), one per proven structure, plus ad_copy.md. Claude then edits
      the wording in each spec to be specific and true before rendering.

  ad.py render DIR            renders every ad_*.json in DIR to ad_*.mp4 and a contact sheet for each
  ad.py render SPEC.json      renders one spec
"""
import argparse, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from vibekit.motion import render_spec
from vibekit.segment import product_cutout
from vibekit import qa

STRUCTURES = ["problem_solution", "bold_claim", "offer", "pov", "reasons"]


def prep(img, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(img))
    m, cov = product_cutout(bgr)
    rgba = np.dstack([bgr, (m * 255).astype(np.uint8)])
    cv2.imwrite(str(out / "cutout.png"), rgba)
    prev = bgr.copy(); prev[m < 0.5] = (prev[m < 0.5] * 0.25).astype(np.uint8)
    cv2.imwrite(str(out / "cutout_preview.jpg"), prev)
    px = bgr[m > 0.5].reshape(-1, 3).astype(np.float32)
    colors = []
    if len(px) > 50:
        _, labels, centers = cv2.kmeans(px[:: max(1, len(px) // 20000)], 4, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=4)
        for i in np.argsort(-counts):
            b, g, r = centers[i]
            colors.append("#%02X%02X%02X" % (int(r), int(g), int(b)))
    return {"cutout": str(out / "cutout.png"), "preview": str(out / "cutout_preview.jpg"),
            "coverage": round(cov, 3), "clean": 0.05 < cov < 0.9, "product_colors": colors}


def _t(text, style, at=0.0, until=None, **kw):
    d = {"text": text, "style": style, "at": at}
    if until is not None:
        d["until"] = until
    d.update(kw)
    return d


def scaffold(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prod = Path(a.product)
    shutil.copy2(prod, out / ("product" + prod.suffix.lower()))
    p = "product" + prod.suffix.lower()
    photos = []
    for i, ph in enumerate([x for x in (a.photos or "").split(",") if x]):
        dst = out / f"photo_{i + 1}{Path(ph).suffix.lower()}"
        shutil.copy2(ph, dst); photos.append(dst.name)
    feats = [f for f in (a.features or "").split("|") if f] or ["Feature one", "Feature two", "Feature three"]
    brand = {"bg": a.bg, "accent": a.accent, "text": "#FFFFFF", "cta_text": "#000000"}
    name, ben, prob = a.name, a.benefit, a.problem or "the same old problem"
    price_line = a.offer or (f"Only {a.price}" if a.price else "Limited time")
    cta = a.cta or "Shop now"
    life = (lambda i: {"type": "image", "src": photos[i % len(photos)], "motion": "zoom_in"}) if photos else None

    def end_card(d=2.6):
        return {"type": "card", "duration": d, "bg": a.bg,
                "texts": [_t(f"*{name}*", "hook", y=0.36), _t(price_line, "caption", 0.2, y=0.52), _t(cta, "cta", 0.4)]}

    specs = {
        "problem_solution": [
            {**(life(0) if life else {"type": "product", "src": p, "bg": "blur"}), "duration": 2.6,
             "texts": [_t(f"Tired of {prob}?", "hook")]},
            {"type": "product", "src": p, "duration": 3.0, "texts": [_t(f"Meet *{name}*", "hook"), _t(ben, "caption", 0.6)]},
            {"type": "product", "src": p, "duration": 3.4, "bg": a.bg,
             "texts": [_t(feats[0], "badge", 0.0, y=0.14), _t(feats[1 % len(feats)], "badge", 0.8, y=0.21),
                       _t(feats[2 % len(feats)], "badge", 1.6, y=0.28)]},
            end_card()],
        "bold_claim": [
            {"type": "product", "src": p, "duration": 3.0, "texts": [_t(f"*{ben}*", "hook", y=0.18)]},
            {"type": "product", "src": p, "duration": 3.0, "bg": "blur", "texts": [_t(f"No, really. {feats[0]}.", "caption")]},
            *([{**life(1), "duration": 2.4, "texts": [_t(feats[1 % len(feats)], "caption")]}] if life else []),
            end_card()],
        "offer": [
            {"type": "card", "duration": 1.6, "bg": a.accent, "texts": [_t("STOP SCROLLING", "hook", y=0.5, **{})]},
            {"type": "product", "src": p, "duration": 3.0, "texts": [_t(f"*{name}*", "hook"), _t(ben, "caption", 0.5)]},
            {"type": "product", "src": p, "duration": 3.0, "bg": "blur",
             "texts": [_t(a.price or price_line, "price"), _t(a.offer or "This week only", "caption", 0.5)]},
            end_card(2.4)],
        "pov": [
            {**(life(0) if life else {"type": "product", "src": p, "bg": "blur"}), "duration": 3.2,
             "texts": [_t(f"POV: you finally found {ben}", "caption", y=0.24)]},
            {"type": "product", "src": p, "duration": 3.2, "texts": [_t(f"It's *{name}*", "caption", y=0.24)]},
            {"type": "product", "src": p, "duration": 3.0, "bg": "blur",
             "texts": [_t(f"{feats[0]} + {feats[1 % len(feats)]}", "caption", y=0.24)]},
            end_card(2.2)],
        "reasons": [
            {"type": "product", "src": p, "duration": 2.4, "texts": [_t(f"3 reasons you need *{name}*", "hook")]},
            *[{"type": "product", "src": p, "duration": 2.2, "bg": "blur" if i % 2 else a.bg,
               "texts": [_t(f"{i + 1}", "price", y=0.2), _t(f, "caption", 0.2, y=0.76)]} for i, f in enumerate(feats[:3])],
            end_card(2.4)],
    }
    feats3 = (feats + feats + feats)[:3]
    vo_scripts = {
        "problem_solution": f"Tired of {prob}? Meet {name}. {ben}. {feats3[0]}, {feats3[1]}, and {feats3[2]}. {price_line}. {cta}.",
        "bold_claim": f"{ben}. That's what {name} does. {feats3[0]}. {cta}.",
        "offer": f"Stop scrolling. This is {name}. {ben}. {price_line}. {cta}.",
        "pov": f"P O V: you finally found {ben}. It's {name}. {feats3[0]} and {feats3[1]}. {cta}.",
        "reasons": f"Three reasons you need {name}. One: {feats3[0]}. Two: {feats3[1]}. Three: {feats3[2]}. {cta}.",
    }
    music_prompt = a.music_prompt or "modern upbeat pop, punchy drums, bright synth plucks, confident, 118 bpm, instrumental"
    made = []
    for i, key in enumerate(STRUCTURES, 1):
        scenes = specs[key]
        if not a.no_voice:
            # the voice carries the words: keep big on-screen hooks/prices/CTAs, let synced captions do the rest
            for sc in scenes:
                sc["texts"] = [t for t in sc.get("texts", []) if t.get("style") != "caption"]
                for t in sc["texts"]:
                    if t.get("style") == "hook":
                        t["anim"] = "words"
        spec = {"structure": key, "size": [1080, 1920], "fps": 30, "brand": brand, "scenes": scenes,
                "transition": "auto", "sfx": True, "caption_style": "punch", "caption_y": 0.74}
        if not a.no_voice:
            spec["voiceover"] = {"text": vo_scripts[key], "voice": a.voice, "speed": 1.05}
        if a.music:
            shutil.copy2(a.music, out / ("music" + Path(a.music).suffix)); spec["music"] = "music" + Path(a.music).suffix
        elif not a.no_music:
            spec["music"] = {"prompt": music_prompt, "seed": 11, "seconds": 30}
        (out / f"ad_{i}.json").write_text(json.dumps(spec, indent=1))
        made.append(str(out / f"ad_{i}.json"))
    copy = [f"# Ad copy — {name}\n"]
    hooks = {"problem_solution": f"Tired of {prob}?", "bold_claim": ben, "offer": "Stop scrolling 👇",
             "pov": f"POV: you finally found {ben}", "reasons": f"3 reasons you need {name}"}
    for i, key in enumerate(STRUCTURES, 1):
        copy.append(f"## Ad {i} — {key}\n- **Primary text:** {hooks[key]} {name}: {ben}. {feats[0]}.\n"
                    f"- **Headline:** {name} — {ben}\n- **CTA button:** {cta}\n")
    (out / "ad_copy.md").write_text("\n".join(copy))
    return {"specs": made, "copy": str(out / "ad_copy.md")}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prep"); pr.add_argument("image"); pr.add_argument("--out", required=True)
    s = sub.add_parser("scaffold")
    for k in ("--product", "--name", "--benefit", "--out"):
        s.add_argument(k, required=True)
    for k in ("--problem", "--features", "--price", "--offer", "--cta", "--photos", "--music"):
        s.add_argument(k)
    s.add_argument("--accent", default="#FFD400"); s.add_argument("--bg", default="#111111")
    s.add_argument("--voice", default="confident_male"); s.add_argument("--music-prompt")
    s.add_argument("--no-voice", action="store_true"); s.add_argument("--no-music", action="store_true")
    r = sub.add_parser("render"); r.add_argument("target")
    a = ap.parse_args()
    if a.cmd == "prep":
        print(json.dumps(prep(a.image, a.out), indent=1))
    elif a.cmd == "scaffold":
        print(json.dumps(scaffold(a), indent=1))
    else:
        t = Path(a.target)
        specs = sorted(t.glob("ad_*.json")) if t.is_dir() else [t]
        res = []
        for sp in specs:
            out = sp.with_suffix(".mp4")
            render_spec(sp, out)
            res.append(qa.report(out))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
