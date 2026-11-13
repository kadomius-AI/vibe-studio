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

## Built with (all free and open source, runs on your Mac)
| Job | Framework / model | License |
|---|---|---|
| Video in/out, cutting, loudness | FFmpeg, OpenCV | LGPL/GPL tool, Apache-2.0 |
| Face tracking | OpenCV YuNet | MIT |
| Transcription + word timing | Whisper (mlx-whisper on Apple Silicon) | MIT |
| Voiceover with word timing | Kokoro-82M | Apache-2.0 |
| Original music | ACE-Step | Apache-2.0 |
| Beat detection | librosa | ISC |
| Sound effects | synthesized in code (original) | — |
| Voice isolation | Demucs (htdemucs) | MIT |
| 3D parallax on photos | Depth Anything V2 Small (PyTorch + transformers) | Apache-2.0 |
| Product cut-outs | Segment Anything (SAM ViT-B) | Apache-2.0 |
| Person cut-outs / backgrounds | MediaPipe selfie segmenter | Apache-2.0 |
| Face edits (beard / stubble / shave, consent only) | SDXL Inpainting via diffusers + MediaPipe Face Landmarker | CreativeML Open RAIL++-M / Apache-2.0 |
| AI upscaling of old low-res footage | Real-ESRGAN x4 via spandrel | BSD-3 / MIT |
| Graphics and text | Pillow + NumPy | MIT-CMU / BSD |
| AI runtime | PyTorch (Apple GPU), MLX | BSD / MIT |
No API keys, no accounts, no payments. First use downloads about 10–15 GB of models once.

Fonts: Montserrat, Anton, Playfair Display, Inter — SIL Open Font License 1.1.
