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

  clips.py check FILE
      Specs, loudness and a contact sheet of stills (for visual QA).
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vibekit import media, render as R, qa


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
    c = sub.add_parser("check"); c.add_argument("file")
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
        R.render(work / "cut.mp4", out, fmt=a.format, words=None if a.no_captions else rel,
                 caption_style=a.style, caption_y=a.caption_y, headline=a.headline, headline_secs=a.headline_secs,
                 zoom=a.zoom, eye_y=a.eye_y, prefer_x=a.prefer_x, music=a.music)
        print(json.dumps(qa.report(out), indent=1))

    elif a.cmd == "check":
        print(json.dumps(qa.report(Path(a.file)), indent=1))


if __name__ == "__main__":
    main()
