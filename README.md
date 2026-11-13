# Vibe Studio

Clips, thumbnails and ads from your content — just by talking to Claude Code.

| Say this | You get |
|---|---|
| "Make clips from this video: <link or file>" | Captioned, face-tracked vertical clips (Reels / Shorts / TikTok) |
| "Make thumbnails for this video" | 3 YouTube thumbnails in 3 styles, ready to A/B test |
| "Make ads for this product" + a photo | 5 short vertical video ads + ready-to-paste ad copy |

Everything runs on your own computer. No API keys, no accounts. First use installs what it needs (a few minutes).

## Install (in Claude Code)
```
/plugin marketplace add kadomius-AI/vibe-studio
/plugin install vibe-studio@vibe-studio
```
Then just ask in plain English, or use `/vibe-studio:clips`, `/vibe-studio:thumbnail`, `/vibe-studio:ad`.

Needs: macOS or Linux, Homebrew (`ffmpeg` is installed automatically), Python 3.11–3.13.

## Where things go
`~/Documents/Vibe Projects/<project>/` → `00_SOURCE` (your footage, never deleted), `10_WORK`, `20_DELIVER` (finished files).

## Credits
Fonts: Montserrat, Anton, Playfair Display, Inter (SIL Open Font License 1.1, https://openfontlicense.org).
Models downloaded on first use: OpenCV YuNet face detector (MIT), MediaPipe selfie segmenter (Apache-2.0),
Whisper speech recognition (MIT).
