---
name: clips
description: >
  Turn a long video (local file, YouTube or any video link) into finished, captioned, face-tracked
  vertical clips (9:16 Reels/Shorts/TikTok). Use for "make clips", "cut this up", "shorts from this",
  "clip this podcast/interview/webinar", "vertical clips", "vibe edit".
---

# /clips — long video in, captioned vertical clips out

Scripts live two folders up from this skill: `../../scripts/`. Always run them through
`../../scripts/run.sh` (it sets everything up automatically the first time; first run takes a few minutes).
The user is not technical: run every command yourself and explain in plain words.

## Steps
1. **Bring it in + transcribe** (offline, free):
   `run.sh clips.py prepare "<file or link>"`  → prints the project folder and transcript path.
   Projects go to `~/Documents/Vibe Projects/<title>/` (00_SOURCE, 10_WORK, 20_DELIVER). Never delete source footage.
2. **Read the transcript** (`10_WORK/transcript.txt`, timestamped) and pick the strongest moments (rules below).
   A clip may join several segments: `[[start,end],[start,end]]` in seconds.
3. **Render each clip:**
   `run.sh clips.py render "<project>" --segments '[[12.4,40.1]]' --name "<short-name>" --headline "Line 1|Line 2"`
   Options: `--style bold|clean|punch`, `--format vertical|square|portrait|horizontal`, `--zoom 1.0–2.0` (tighter on the face),
   `--eye-y 0.30–0.45`, `--prefer-x 0.25` (follow the left person), `--music file.mp3`, `--no-captions`,
   `--words fixed.json` (corrected caption words).
4. **Check your work:** open the printed `contact_sheet` image and look at it. Fix and re-render if the face is cut off,
   captions cover the face, or burned-in text from the source shows. Report duration and loudness (target about −14 LUFS).
5. Show the user the files in `20_DELIVER/`.

## Choosing moments (the playbook)
- **Open on the hook.** The first sentence must make someone stop: a bold claim, a surprising number, a question, conflict.
  Never open on "so", "um", throat-clearing or setup.
- **One idea per clip, with a payoff.** Setup → tension → payoff. End on the punchline, not after it.
- **Length:** 20–60 s for Reels/Shorts; up to 90 s only if every second earns it.
- **Join segments** to remove rambling, but never cut mid-word or mid-thought; the engine snaps cuts to silence.
- **Remove filler and repeats** by splitting segments around them.
- Prefer moments with emotion, stories, specific numbers, contrarian opinions, and clear advice.
- 3–5 clips per long video is a good default. Name each clip after its hook.

## Captions and headline
- Captions are word-by-word with the spoken word highlighted. `bold` = social default, `clean` = premium/brand,
  `punch` = loud ads.
- Transcription can mishear names and invent words. Read `10_WORK/<clip>/words.json`; if needed, fix spellings,
  save a copy and re-render with `--words`.
- Headline (first ~3 s, top of frame): 3–7 words that frame the clip ("my mentor's #1|leadership rule:").
  Never cover the face: if it does, lower `--eye-y` or zoom less.

## Quality bar
1080×1920, 30 fps, H.264/AAC, loudness about −14 LUFS, face fully in frame, captions readable on a phone,
no burned-in source text, clean first and last word.
