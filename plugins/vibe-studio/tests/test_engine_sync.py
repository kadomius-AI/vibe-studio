"""The shared engine must match its manifest: edit it in vibe-studio, then run tools/sync_engine.py."""
import hashlib
import json
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]


def test_engine_matches_manifest():
    man = json.loads((PLUGIN / "scripts" / "vibekit" / "MANIFEST.json").read_text())
    for rel, h in man["files"].items():
        f = PLUGIN / rel
        assert f.exists(), f"missing engine file {rel}"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == h, (
            f"{rel} differs from the shared engine. Make engine changes in vibe-studio and run tools/sync_engine.py.")
    now = sorted(str(p.relative_to(PLUGIN)) for p in (PLUGIN / "scripts" / "vibekit").rglob("*.py"))
    assert now == sorted(man["files"]), "engine files were added/removed without syncing"
