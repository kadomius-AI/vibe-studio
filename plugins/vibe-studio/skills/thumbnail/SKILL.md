---
name: thumbnail
description: >
  Make 3 ready-to-upload YouTube thumbnails (1280x720) from a video file or link, in three tested styles
  (big face + emotion, before/after split, result focus). Use for "make a thumbnail", "thumbnails for this video",
  "YouTube cover", "A/B test thumbnails".
---

# /thumbnail — video in, 3 thumbnails out

Run scripts through `../../scripts/run.sh` (auto-setup on first run). The user is not technical: do every step yourself.

## Steps
1. **Understand the video.** If it has speech and you need the topic, run `run.sh clips.py prepare "<video>"` and skim the
   transcript. Decide the single reason someone would click (curiosity, result, conflict, transformation).
2. **Find frames:** `run.sh thumbnail.py candidates "<video or link>" --out "<folder>/thumbs"` and LOOK at
   `candidates_sheet.jpg`. Pick frames with a clear, expressive face (surprise, joy, worry), eyes open, sharp, well lit.
   For "split", pick two frames that contrast (stressed vs happy, before vs after).
3. **Make 3 different styles** (one of each, so the user can A/B test):
   - `run.sh thumbnail.py make --frame <img> --text "NEVER *VENT* DOWN" --style face --out A_face.png`
   - `run.sh thumbnail.py make --frame <img1> --frame2 <img2> --labels "BEFORE|AFTER" --text "ONE *RULE*" --style split --out B_split.png`
   - `run.sh thumbnail.py make --frame <img> --text "HONEST > *SMART*" --style result --out C_result.png`
   Options: `--accent #FFD400` (highlight color), `--bg blur|dark|#hex`, `--side left|right` (where the person sits).
4. **Check readability:** `run.sh thumbnail.py compare A_face.png B_split.png C_result.png --out compare.png` and look
   at the small row (real phone size). If words are hard to read small, cut words or increase contrast and redo.
5. Deliver the 3 PNGs and tell the user to upload all three in YouTube Studio → video → **Test & Compare**,
   let it run a couple of days, and keep the winner.

## Rules from research on top thumbnails
- **Expressive face wins** in most niches; neutral faces underperform. Eye contact helps.
- **3–5 words max**, huge and high-contrast. The words must ADD to the title (curiosity, stakes, number), never repeat it.
- **One accent word** in a contrasting color (`*word*`). Avoid full sentences and small text.
- **High contrast and saturation** beat clean minimal looks in 2026; subject clearly separated from background.
- **Readable at phone size** (about 246 px wide) is the real test.
- **Always make variants and test**; patterns are starting points, the channel's own data decides.
- Never use misleading claims or someone else's likeness without rights.
