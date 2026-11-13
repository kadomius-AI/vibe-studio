#!/usr/bin/env python3
"""/clips — long video in, captioned vertical clips out.

  clips.py prepare SOURCE [--name NAME] [--root DIR]
      Copy/download the footage into a project and transcribe it (offline).
      Prints the project folder and the transcript (.txt) Claude reads to choose moments.

  clips.py render PROJECT --segments '[[s,e],...]' --name CLIP_NAME
                  [--headline "Line 1|Line 2"] [--headline-secs 3] [--style bold|clean|punch]
                  [--format vertical|square|portrait|horizontal] [--zoom 1.0] [--eye-y 0.38]
                  [--prefer-x 0.25] [--no-captions] [--music FILE] [--words FILE]
      Cut (snapped to silence) -> follow the face -> captions -> headline -> -14 LUFS -> 20_DELIVER/NAME.mp4

  clips.py moments PROJECT [--min 20] [--max 60] [--top 8]
      Shortlist the best moments: transcript cues (hook, payoff, self-contained, quotable, emotion...)
      + audio reactions (laughs/applause/bursts, energy build-up) + visual energy. Writes 10_WORK/moments.json.
      Claude reads the reasons and makes the final pick (scores are a shortlist, not a verdict).

  clips.py batch QUEUE.txt [--root DIR] [--auto-render 3] [--brand NAME] [--min 20 --max 60]
      Many videos: one file path or link per line (or a JSON list). Each gets its own project, transcript and
      moments shortlist; --auto-render renders the top N per video. Failures are recorded and the batch continues.
      Writes batch_report.json.

  clips.py brand save NAME --font Anton.ttf --highlight "#39FF88" [--logo logo.png] [...]
  clips.py brand list | clips.py brand show NAME
      Reusable brand looks (captions + logo), applied with --brand NAME on render/batch.

  clips.py check FILE
      Specs, loudness and a contact sheet of stills (for visual QA).
"""
import argparse, json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vibekit import media, render as R, qa


def batch(a):
    """Process a queue of videos; each item is isolated so one failure doesn't stop the rest."""
    import subprocess, traceback
    from vibekit import moments
    from vibekit.log import get as getlog
    log = getlog("batch")
    qp = Path(a.queue).expanduser()
    raw = qp.read_text().strip()
    items = json.loads(raw) if raw.startswith("[") else [l.strip() for l in raw.splitlines() if l.strip() and not l.startswith("#")]
    report = {"queue": str(qp), "items": []}
    for n, src in enumerate(items, 1):
        item = {"source": src, "status": "ok"}
        log.info("[%d/%d] %s", n, len(items), src)
        try:
            tmp = media.new_project(Path(a.root).expanduser(), f"incoming-{n}")
            path, title = media.ingest(src, tmp)
            proj = Path(a.root).expanduser() / media.slug(title)
            if not proj.exists():
                tmp.rename(proj)
            else:
                proj = tmp
            path = next(f for f in sorted((proj / "00_SOURCE").iterdir()) if f.suffix.lower() in media.VIDEO_EXT)
            words = media.transcribe(path, proj / "10_WORK" / "transcript.json")["words"]
            found = moments.find(path, words, a.min, a.max, 8)
            (proj / "10_WORK" / "moments.json").write_text(json.dumps(found, indent=1))
            item.update({"project": str(proj), "title": title, "moments": len(found),
                         "top": [{k: m[k] for k in ("start", "end", "score", "reasons")} for m in found[:3]]})
            renders = []
            for k, m in enumerate(found[:a.auto_render], 1):
                cmd = [sys.executable, __file__, "render", str(proj), "--segments", json.dumps([[m["start"], m["end"]]]),
                       "--name", f"moment-{k}", "--format", a.format, "--allow-qa-fail"]
                if a.brand:
                    cmd += ["--brand", a.brand]
                r = subprocess.run(cmd, capture_output=True, text=True)
                out = json.loads(r.stdout[r.stdout.index("{"):]) if "{" in r.stdout else {"error": r.stderr[-800:]}
                renders.append({"moment": k, "file": out.get("file"), "qa": out.get("qa"), "error": out.get("error")})
            item["renders"] = renders
        except Exception as e:  # noqa: BLE001 - one bad item must not stop the batch
            item.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
            log.error("item failed: %s\n%s", src, traceback.format_exc())
        report["items"].append(item)
    out = qp.with_name("batch_report.json")
    out.write_text(json.dumps(report, indent=1))
    ok = sum(i["status"] == "ok" for i in report["items"])
    print(json.dumps({"report": str(out), "ok": ok, "failed": len(items) - ok}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("source"); p.add_argument("--name"); p.add_argument("--root", default="~/Documents/Vibe Projects")
    r = sub.add_parser("render"); r.add_argument("project"); r.add_argument("--segments", required=True); r.add_argument("--name", required=True)
    r.add_argument("--headline"); r.add_argument("--headline-secs", type=float, default=3.0)
    r.add_argument("--style", default="bold"); r.add_argument("--format", default="vertical")
    r.add_argument("--zoom", type=float, default=1.0); r.add_argument("--eye-y", type=float, default=0.38)
    r.add_argument("--prefer-x", type=float); r.add_argument("--no-captions", action="store_true")
    r.add_argument("--music"); r.add_argument("--words", help="corrected clip-relative words JSON to use for captions")
    r.add_argument("--caption-y", type=float)
    r.add_argument("--no-punch", action="store_true", help="no punch-in zooms")
    r.add_argument("--no-sfx", action="store_true", help="no sound effects")
    r.add_argument("--music-prompt", help="generate an original music bed, e.g. 'lofi hip hop, mellow, 85 bpm, instrumental'")
    r.add_argument("--layout", choices=["auto", "crop", "fill"], default="auto",
                   help="crop = face-tracked crop; fill = whole frame over a blurred backdrop; auto picks by face size")
    r.add_argument("--upscale", choices=["auto", "on", "off"], default="auto", help="AI upscaling for footage under 720p")
    r.add_argument("--restore-faces", type=float, default=None, help="GFPGAN face restoration 0..1 (auto 0.6 under 720p)")
    r.add_argument("--allow-qa-fail", action="store_true", help="exit 0 even when automated QA fails")
    r.add_argument("--brand", help="saved brand name or brand .json (a project brand.json is layered on top)")
    c = sub.add_parser("check"); c.add_argument("file")
    mo = sub.add_parser("moments"); mo.add_argument("project"); mo.add_argument("--min", type=float, default=20)
    mo.add_argument("--max", type=float, default=60); mo.add_argument("--top", type=int, default=8)
    b = sub.add_parser("batch"); b.add_argument("queue"); b.add_argument("--root", default="~/Documents/Vibe Projects")
    b.add_argument("--auto-render", type=int, default=0); b.add_argument("--brand")
    b.add_argument("--min", type=float, default=20); b.add_argument("--max", type=float, default=60)
    b.add_argument("--format", default="vertical")
    br = sub.add_parser("brand"); br.add_argument("action", choices=["save", "list", "show"]); br.add_argument("name", nargs="?")
    for k, t in (("--base", str), ("--font", str), ("--size", int), ("--color", str), ("--highlight", str), ("--stroke", int),
                 ("--box", str), ("--max-words", int), ("--y", float), ("--accent", str), ("--logo", str),
                 ("--logo-position", str), ("--logo-width", int), ("--music-prompt", str), ("--headline-box", str)):
        br.add_argument(k, type=t)
    br.add_argument("--upper", action=argparse.BooleanOptionalAction, default=None)
    a = ap.parse_args()

    if a.cmd == "prepare":
        tmp = media.new_project(Path(a.root).expanduser(), a.name or "incoming")
        src, title = media.ingest(a.source, tmp)
        proj = tmp
        if not a.name:  # name the project after the video title
            target = Path(a.root).expanduser() / media.slug(title)
            if not target.exists():
                tmp.rename(target)
                proj = target
                src = target / "00_SOURCE" / src.name
        data = media.transcribe(src, proj / "10_WORK" / "transcript.json")
        info = media.probe(src)
        print(json.dumps({"project": str(proj), "source": str(src), "title": title,
                          "duration": round(info["duration"], 1), "size": [info["width"], info["height"]],
                          "transcript": str(proj / "10_WORK" / "transcript.txt"), "words": len(data["words"])}, indent=1))

    elif a.cmd == "render":
        proj = Path(a.project).expanduser()
        src = next(f for f in sorted((proj / "00_SOURCE").iterdir()) if f.suffix.lower() in media.VIDEO_EXT)
        words = json.loads((proj / "10_WORK" / "transcript.json").read_text())["words"]
        segs = json.loads(a.segments) if a.segments.strip().startswith("[") else json.loads(Path(a.segments).read_text())
        work = proj / "10_WORK" / media.slug(a.name)
        work.mkdir(parents=True, exist_ok=True)
        rel, dur = media.cut(src, segs, work / "cut.mp4", words)
        (work / "words.json").write_text(json.dumps(rel, indent=0))
        if a.words:
            rel = json.loads(Path(a.words).read_text())
        out = proj / "20_DELIVER" / f"{media.slug(a.name)}.mp4"
        music = a.music
        if a.music_prompt and not music:
            import subprocess
            music = str(work / "music.wav")
            subprocess.run([str(Path.home() / ".local/share/vibekit/acestep-venv/bin/python"),
                            str(Path(__file__).resolve().parent / "music_gen.py"), "--prompt", a.music_prompt,
                            "--seconds", str(round(dur + 2)), "--out", music], check=True)
        from vibekit import checks, plan as vplan, restore as _r, brand as BR  # noqa: F401
        brand_cfg = BR.resolve(a.brand, proj) if (a.brand or (proj / "brand.json").exists()) else None
        if brand_cfg and brand_cfg.get("music_prompt") and not a.music and not a.music_prompt:
            a.music_prompt = brand_cfg["music_prompt"]
        low_res = media.probe(work / "cut.mp4")["height"] < 720
        restore = a.restore_faces if a.restore_faces is not None else (0.6 if low_res else 0.0)
        R.render(work / "cut.mp4", out, fmt=a.format, words=None if a.no_captions else rel,
                 caption_style=a.style, caption_y=a.caption_y, headline=a.headline, headline_secs=a.headline_secs,
                 zoom=a.zoom, eye_y=a.eye_y, prefer_x=a.prefer_x, music=music,
                 punch=not a.no_punch, sfx=not a.no_sfx, layout=a.layout,
                 upscale=(a.upscale == "on") or (a.upscale == "auto" and low_res), restore_faces=restore,
                 brand=brand_cfg)
        man = json.loads(Path(str(out) + ".render.json").read_text())
        cut_points = list(np.cumsum([e - s for s, e in segs])[:-1]) if len(segs) > 1 else []
        rep = checks.run(out, fmt=a.format, expected_duration=dur, cut_times=cut_points, manifest=man)
        pl = vplan.new("clips", {"source": src})
        pl["decisions"] = {"args": {k: v for k, v in vars(a).items() if k != "cmd"}, "segments": segs, "brand": brand_cfg,
                           "layout": man["layout"], "restore_faces": restore, "words_file": str(work / "words.json")}
        pl["outputs"] = {"video": str(out), "qa": str(out) + ".qa.json"}
        vplan.save(pl, str(out) + ".plan.json")
        checks.save(rep, str(out) + ".qa.json")
        res = qa.report(out)
        res.update({"qa": checks.summary(rep), "plan": str(out) + ".plan.json"})
        print(json.dumps(res, indent=1))
        if not rep["passed"] and not a.allow_qa_fail:
            sys.exit(2)

    elif a.cmd == "check":
        print(json.dumps(qa.report(Path(a.file)), indent=1))

    elif a.cmd == "moments":
        from vibekit import moments
        proj = Path(a.project).expanduser()
        src = next(f for f in sorted((proj / "00_SOURCE").iterdir()) if f.suffix.lower() in media.VIDEO_EXT)
        words = json.loads((proj / "10_WORK" / "transcript.json").read_text())["words"]
        found = moments.find(src, words, a.min, a.max, a.top)
        (proj / "10_WORK" / "moments.json").write_text(json.dumps(found, indent=1))
        print(json.dumps([{k: m[k] for k in ("start", "end", "duration", "score", "reasons")} | {"text": m["text"][:140]}
                          for m in found], indent=1))

    elif a.cmd == "batch":
        batch(a)

    elif a.cmd == "brand":
        from vibekit import brand as BR
        if a.action == "list":
            print(json.dumps(BR.list_brands()))
        elif a.action == "show":
            print(json.dumps(BR.resolve(a.name), indent=1))
        else:
            keys = {"base": a.base, "font": a.font, "size": a.size, "color": a.color, "highlight": a.highlight,
                    "stroke": a.stroke, "box": a.box, "max_words": a.max_words, "y": a.y, "accent": a.accent,
                    "logo": a.logo, "logo_position": a.logo_position, "logo_width": a.logo_width,
                    "music_prompt": a.music_prompt, "headline_box": a.headline_box, "upper": a.upper}
            print(BR.save(a.name, {k: v for k, v in keys.items() if v is not None}))


if __name__ == "__main__":
    main()
