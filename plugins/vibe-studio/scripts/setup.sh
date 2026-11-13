#!/bin/bash
# One-time setup: ffmpeg check + a private Python environment with the editing libraries.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PLUGIN="$(basename "$(dirname "$HERE")")"
VENV="${VIBEKIT_HOME:-$HOME/.local/share/vibekit}/venv"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v ffmpeg >/dev/null; then
  if command -v brew >/dev/null; then echo "Installing ffmpeg…"; brew install ffmpeg;
  else echo "ffmpeg is missing. Install Homebrew (https://brew.sh) then: brew install ffmpeg"; exit 1; fi
fi
PY=""
for c in python3.12 python3.11 python3.13 python3; do
  if command -v $c >/dev/null && $c -c 'import sys; sys.exit(0 if sys.version_info[:2] in [(3,11),(3,12),(3,13)] else 1)'; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  if command -v brew >/dev/null; then brew install python@3.12; PY=python3.12; else echo "Python 3.11–3.13 is required."; exit 1; fi
fi
command -v espeak-ng >/dev/null || { command -v brew >/dev/null && brew install espeak-ng; }   # voice pronunciation helper
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$HERE/requirements.txt"
if [ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ]; then "$VENV/bin/pip" install -q mlx-whisper; else "$VENV/bin/pip" install -q faster-whisper; fi
# Original music generator (ACE-Step, Apache-2.0) lives in its own environment (different library versions)
ACE="${VIBEKIT_HOME:-$HOME/.local/share/vibekit}/acestep-venv"
if [ ! -x "$ACE/bin/python" ]; then
  echo "Installing the music generator (one time, a few minutes)…"
  "$PY" -m venv "$ACE" && "$ACE/bin/pip" install -q --upgrade pip &&   "$ACE/bin/pip" install -q "git+https://github.com/ace-step/ACE-Step.git@1bee4c9f5b43" soundfile || echo "Music generator not installed; videos will use your own music files."
fi
touch "$VENV/.ok-$PLUGIN"
echo "Vibe setup complete ($PLUGIN)."
