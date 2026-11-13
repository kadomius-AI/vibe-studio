#!/usr/bin/env python3
"""Copy the shared engine from vibe-studio (the source of truth) into the sibling plugin repos and stamp a manifest.

  python3 tools/sync_engine.py ../vibe-listing ../vibe-glowup

Copies: scripts/vibekit/, scripts/clips.py, scripts/music_gen.py, scripts/run.sh, scripts/setup.sh,
scripts/requirements.txt, assets/fonts/, tests/conftest.py, tests/test_units.py, tests/test_e2e.py, tests/fixtures/.
Writes scripts/vibekit/MANIFEST.json (file hashes + engine version + source commit) in every repo, and each repo's
tests/test_engine_sync.py fails if its engine copy was edited locally instead of here.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "plugins" / "vibe-studio"
SHARED = ["scripts/vibekit", "scripts/clips.py", "scripts/music_gen.py", "scripts/run.sh", "scripts/setup.sh",
          "scripts/requirements.txt", "assets/fonts", "tests/conftest.py", "tests/test_units.py", "tests/test_e2e.py",
          "tests/test_engine_sync.py", "tests/test_chopify_features.py", "tests/fixtures"]


def engine_hashes(plugin_dir):
    out = {}
    for f in sorted((plugin_dir / "scripts" / "vibekit").rglob("*.py")):
        out[str(f.relative_to(plugin_dir))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def stamp(plugin_dir, commit):
    sys.path.insert(0, str(SRC / "scripts"))
    from vibekit.plan import ENGINE_VERSION
    m = {"engine": ENGINE_VERSION, "source": "vibe-studio", "commit": commit, "files": engine_hashes(plugin_dir)}
    (plugin_dir / "scripts" / "vibekit" / "MANIFEST.json").write_text(json.dumps(m, indent=1))


def main(targets):
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp(SRC, commit)
    for t in targets:
        repo = Path(t).resolve()
        dst = repo / "plugins" / repo.name
        if not dst.is_dir():
            sys.exit(f"{dst} is not a plugin folder")
        for rel in SHARED:
            s, d = SRC / rel, dst / rel
            if not s.exists():
                continue
            if s.is_dir():
                shutil.copytree(s, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
        stamp(dst, commit)
        print(f"synced engine -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1:])
