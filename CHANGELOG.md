# Changelog

## 0.5.0 (improve/v0.5)
- **Moment shortlist** (`clips.py moments`): transcript cues + audio reactions + visual energy, with reasons
  (adapted from Chopify, MIT; sign-off/intro penalty added after a real test ranked an outro first).
- **Batch queue** (`clips.py batch`): isolated projects per video, continues past failures, `batch_report.json`.
- **Brand presets** (`clips.py brand`, `--brand`): layered built-in -> brand -> project, validated, logo kept in the safe area.
- **Shared engine (vibekit 0.5)**: one source in vibe-studio, synced by `tools/sync_engine.py`, with a hash manifest and a
  test that fails on local drift.
- **Quality**: lossless intermediates (one lossy encode only), source frame rate kept (≤60), CRF 17 / preset slow,
  30 ms audio fades at cuts (from video-use).
- **Look**: bounded auto-grade (face-based exposure, white balance on neutral pixels, skin-hue protection, black floor,
  light luminance sharpening). Bounded, measurement-driven approach from video-use's auto-grade (MIT).
- **Restoration**: Real-ESRGAN general x4 v3 upscaling and GFPGAN v1.4 face restoration (spandrel) for footage under 720p.
- **Layout**: auto crop vs fill (4:3 picture over a dark backdrop) by face size; QA ignores backdrop faces.
- **Captions**: safe-area aware, never over the face (per-frame face check), pop-in animation; phrase-aware breaks
  (autobroll's pause/clause rules, MIT).
- **Edit plans**: `*.plan.json` with input fingerprints; `glowup.py render-plan` re-renders exactly.
- **Automated QA**: `*.qa.json` (render, size, fps, duration, black/frozen frames, loudness/true peak, caption safe
  area and face overlap, face cut-off, clicks at cuts; ideas from vibe-editing's auditors and MakeMyClip's cut review).
  Commands exit 2 when QA fails.
- **Logging**: `~/.local/share/vibekit/logs/vibe.log`; FFmpeg errors are reported with FFmpeg's own explanation.
- **Tests**: unit + end-to-end render tests on generated fixtures (`python -m pytest plugins/*/tests`).
- Glow Up: the reference is measured into a style profile (pacing, color, framing, caption band); the set rebuild is
  now opt-in (`--match-scale`).
