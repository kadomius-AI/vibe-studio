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

## Pro finishing (automatic)
- **Punch-in zooms:** the frame jumps ~12% tighter at new sentences (every ~3 s) with a soft swish; `--no-punch` to disable.
- **Sound effects** are original (synthesized), free to use anywhere; `--no-sfx` to disable.
- **Music bed:** `--music-prompt "cinematic emotional piano, soft pads, 80 bpm, instrumental"` generates an ORIGINAL,
  royalty-free track (ACE-Step, ~20 s to make) and ducks it under the speaker. Or `--music file.mp3` (user-licensed).
  Match the mood: motivational = cinematic piano/strings; business = minimal deep house; story = warm lofi. Always "instrumental".

## Layout, restoration and look (automatic)
- `--layout auto|crop|fill`. Auto picks **crop** (face-tracked 9:16) when the face fits, and **fill** (the picture framed 4:3
  around the face over a dark soft backdrop, headline above, captions below) for close-up webcam or wide shots
  where a crop would be too tight.
- Under 720p: Real-ESRGAN upscaling plus GFPGAN face restoration (`--upscale auto|on|off`, `--restore-faces 0..1`).
- Captions are placed per frame: inside the platform safe area (top 8%, bottom 22%, sides 6–10% reserved) and never over the face.

## Edit plans and automated QA (every render)
- `<output>.plan.json`: every decision (segments, layout, grade numbers, restoration, captions, music) plus input fingerprints.
  Edit it and re-render exactly (`glowup.py render-plan PLAN.json`), or keep it as a record.
- `<output>.qa.json`: automated checks: render, dimensions, fps, duration, black or frozen frames, audio present
  and loud enough (-16..-12 LUFS, true peak <= -0.5 dBTP), captions inside the safe area and off the face, the face not cut off,
  and no clicks at cuts. **The command exits with code 2 if QA fails.** Read the failed check, fix it, and re-render.
  Only use `--allow-qa-fail` when the user accepts the issue.

## Find moments, batch many videos, brand presets (ideas from Chopify, MIT)
- **Shortlist:** `run.sh clips.py moments "<project>" --min 20 --max 60 --top 8`. Every candidate gets a 0–10 score
  from transcript cues (hook, payoff, self-contained, quotable, emotion, controversy, humour, minus filler and
  intros/sign-offs), audio reactions (laughs, applause, energy building to a peak) and visual energy, with reasons.
  **The score is a shortlist: read the top candidates and make the call yourself.** Never auto-pick an intro or outro.
- **Batch:** `run.sh clips.py batch queue.txt --auto-render 3 --brand NAME` (one file or link per line). Each video
  gets its own project; failures are recorded in `batch_report.json` and the batch continues.
- **Brand presets:** `run.sh clips.py brand save NAME --base punch --highlight "#39FF88" --logo logo.png
  --logo-position top-left`, then `--brand NAME` on render or batch. A `brand.json` in a project folder overrides
  single fields for that project. Bad values are rejected with a clear message.
