---
name: ad
description: >
  Turn a product photo (plus optional lifestyle photos) into 5 short vertical video ads (9:16, ~9–12 s) with
  captions, motion and an end card, plus ready-to-paste ad copy. Use for "make ads", "video ads for my product",
  "Meta/TikTok/Reels ads", "UGC-style ad", "vibe ad".
---

# /ad — product photo in, 5 video ads out

Run scripts through `../../scripts/run.sh` (auto-setup on first run). The user is not technical: do every step yourself.

## Steps
1. **Ask only what you can't infer** (one question at a time): product name, the #1 benefit, the problem it solves,
   2–3 true features, price/offer if any, call to action, brand colors. Never invent claims, prices, reviews or results.
2. **Prep the photo:** `run.sh ad.py prep product.jpg --out "<folder>/ads"` → look at `cutout_preview.jpg`.
   If the cut-out is messy, keep the photo as a card (automatic) or ask for a photo on a plain background.
   `product_colors` suggests a matching accent color.
3. **Scaffold 5 ads** (one per proven structure):
   `run.sh ad.py scaffold --product product.jpg --name "…" --benefit "…" --problem "…" --features "A|B|C" --price "$29" --offer "…" --cta "Shop now" --accent "#hex" --bg "#hex" [--photos a.jpg,b.jpg] [--music track.mp3] --out "<folder>/ads"`
4. **Rewrite the words** in each `ad_N.json` so they are specific, punchy and true (rules below). Keep each text short.
5. **Render:** `run.sh ad.py render "<folder>/ads"` → `ad_1.mp4 … ad_5.mp4`. Look at each `ad_N_sheet.png`; fix
   overlapping text, unreadable words or a bad cut-out, then re-render.
6. Deliver the 5 videos + `ad_copy.md` (primary text, headline, CTA per ad). Suggest testing all 5 and scaling the winner.

## The 5 structures
1. **Problem → solution:** "Tired of X?" → meet the product → 3 benefits → offer + CTA.
2. **Bold claim:** the result as the hook → proof/feature → CTA.
3. **Offer:** pattern-interrupt ("STOP SCROLLING") → product → price/offer reveal → CTA.
4. **POV / UGC caption style:** "POV: you finally found…" in native-looking captions → product → CTA.
5. **Reasons listicle:** "3 reasons you need X" → numbered reasons → CTA.

## Rules from research on top-performing short-form ads
- **The first 1.5–3 seconds decide everything.** Lead with the problem, the result or a surprising line, never a logo.
- **Show the product early** (within ~2 s) unless the structure is problem-first.
- **Design for sound off:** every key message is on screen as text.
- **One idea per screen,** few words, large type, high contrast; captions in the middle/lower-middle, clear of UI.
- **Hook → body → CTA** within 9–20 s; the CTA sounds like a recommendation, not a shout.
- **Make several hooks and test**; judge by 3-second view rate (hook) and hold rate, not by taste.
- Use only true claims and real reviews. Music must be licensed for ads (user-supplied); otherwise leave silent so
  the user can add a platform-licensed track.
