#!/bin/bash
# Runs a Vibe script inside its private Python environment, creating it on first use.
# Usage: run.sh clips.py prepare VIDEO ...
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VIBEKIT_HOME:-$HOME/.local/share/vibekit}/venv"
if [ ! -x "$VENV/bin/python" ] || [ ! -f "$VENV/.ok-$(basename "$(dirname "$HERE")")" ]; then
  "$HERE/setup.sh" >&2
fi
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec "$VENV/bin/python" "$HERE/$@"
