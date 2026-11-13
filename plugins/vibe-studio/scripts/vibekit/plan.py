"""Structured JSON edit plans: every AI/automatic decision is written down so it can be inspected,
edited by hand or by Claude, and re-rendered exactly.

A plan looks like:
{
  "schema": "vibekit.plan/1", "tool": "glowup", "created": "...", "engine": "0.5.0",
  "inputs": {"source": {"path": "...", "sha256": "..."}, "reference": {...}},
  "decisions": {"segments": [[s, e], ...], "layout": "crop", "grade": {...}, "captions": {...}, ...},
  "outputs": {"video": "...", "qa": "..."}
}
"""
import datetime as _dt
import hashlib
import json
from pathlib import Path

ENGINE_VERSION = "0.5.0"
SCHEMA = "vibekit.plan/1"


def file_sha256(path, limit_mb=64):
    """Fingerprint a media file (first + last `limit_mb` MB, plus size) so plans detect changed inputs quickly."""
    p = Path(path)
    h = hashlib.sha256()
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(limit_mb << 20))
        if size > 2 * (limit_mb << 20):
            f.seek(-(limit_mb << 20), 2)
            h.update(f.read())
    return h.hexdigest()


def new(tool, inputs, decisions=None):
    return {
        "schema": SCHEMA,
        "tool": tool,
        "engine": ENGINE_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {k: ({"path": str(v), "sha256": file_sha256(v)} if v else None) for k, v in inputs.items()},
        "decisions": decisions or {},
        "outputs": {},
    }


def save(plan, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, default=_jsonable))
    return str(path)


def load(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a vibekit plan (schema {plan.get('schema')!r}).")
    for k, v in (plan.get("inputs") or {}).items():
        if v and Path(v["path"]).exists() and file_sha256(v["path"]) != v["sha256"]:
            raise ValueError(f"Input '{k}' changed since the plan was made: {v['path']}")
    return plan


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
