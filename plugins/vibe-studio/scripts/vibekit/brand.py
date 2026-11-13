"""Reusable brand presets: a caption look + logo saved once, applied to every render.

Layering (same idea as Chopify's preset system, MIT, (c) 2026 Mehbul Islam):
    built-in caption style  ->  brand file  ->  per-project override (brand.json in the project folder)
Later layers only override the fields they set. Every value is validated with a clear message.

Brand files live in ~/.config/vibekit/brands/<name>.json (or pass a path to any .json file).
"""
import json
import re
from pathlib import Path

from .captions import STYLES

BRAND_DIR = Path.home() / ".config" / "vibekit" / "brands"
HEX = re.compile(r"^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$")

FIELDS = {
    # caption look
    "base": ("enum", set(STYLES)),            # start from a built-in style: bold | clean | punch
    "font": ("font", None),                   # a font file name in assets/fonts (e.g. Anton.ttf)
    "size": ("int", (36, 160)),               # caption font size at 1080 px wide
    "color": ("hex", None),
    "highlight": ("hex", None),               # the spoken word
    "stroke": ("int", (0, 14)),
    "box": ("hex_or_none", None),             # caption background panel, e.g. #000000AA
    "upper": ("bool", None),
    "max_words": ("int", (1, 6)),
    "y": ("float", (0.12, 0.78)),             # caption centre, fraction of height (kept in the safe area)
    # headline + extras
    "headline_box": ("hex_or_none", None),
    "accent": ("hex", None),                  # thumbnails / ads accent colour
    # logo
    "logo": ("path_or_none", None),
    "logo_position": ("enum", {"top-left", "top-right", "bottom-left", "bottom-right"}),
    "logo_width": ("int", (40, 600)),
    "logo_opacity": ("float", (0.1, 1.0)),
    # defaults for generated music
    "music_prompt": ("str_or_none", None),
}
DEFAULTS = {"base": "bold", "logo": None, "logo_position": "top-right", "logo_width": 170, "logo_opacity": 0.9,
            "headline_box": "#0C0C0ED0", "music_prompt": None}


class BrandError(ValueError):
    pass


def _check(key, value, base_dir):
    kind, rng = FIELDS[key]
    if kind == "enum" and value not in rng:
        raise BrandError(f"{key} must be one of: {', '.join(sorted(rng))} (got {value!r}).")
    if kind == "int" and (isinstance(value, bool) or not isinstance(value, int) or not rng[0] <= value <= rng[1]):
        raise BrandError(f"{key} must be a whole number between {rng[0]} and {rng[1]} (got {value!r}).")
    if kind == "float" and (isinstance(value, bool) or not isinstance(value, (int, float)) or not rng[0] <= value <= rng[1]):
        raise BrandError(f"{key} must be a number between {rng[0]} and {rng[1]} (got {value!r}).")
    if kind == "bool" and not isinstance(value, bool):
        raise BrandError(f"{key} must be true or false.")
    if kind in ("hex", "hex_or_none") and not (value is None and kind == "hex_or_none") and not (
            isinstance(value, str) and HEX.match(value)):
        raise BrandError(f"{key} must be a colour like #FFD400 or #000000AA (got {value!r}).")
    if kind == "font":
        from .env import FONTS
        if not (FONTS / value).exists():
            have = ", ".join(sorted(p.name for p in FONTS.iterdir() if p.suffix in (".ttf", ".otf")))
            raise BrandError(f"font {value!r} not found. Available: {have}")
    if kind == "path_or_none" and value is not None:
        p = Path(value).expanduser()
        p = p if p.is_absolute() else Path(base_dir) / p
        if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise BrandError(f"logo must be an existing PNG/JPEG/WebP image (got {value!r}).")
        return str(p.resolve())
    if kind == "str_or_none" and value is not None and not isinstance(value, str):
        raise BrandError(f"{key} must be text.")
    return value


def validate(data, base_dir="."):
    if not isinstance(data, dict):
        raise BrandError("A brand must be a JSON object.")
    unknown = set(data) - set(FIELDS) - {"name"}
    if unknown:
        raise BrandError(f"Unknown brand fields: {', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(FIELDS))}.")
    return {k: _check(k, v, base_dir) for k, v in data.items() if k != "name" and v is not None or k == "logo"}


def _find(name_or_path):
    p = Path(str(name_or_path)).expanduser()
    if p.suffix == ".json" and p.exists():
        return p
    q = BRAND_DIR / f"{name_or_path}.json"
    if q.exists():
        return q
    raise BrandError(f"Brand {name_or_path!r} not found. Saved brands: {', '.join(list_brands()) or 'none yet'}.")


def resolve(name_or_path=None, project_dir=None):
    """Merge built-in style <- brand <- project override. Returns a flat dict of settings."""
    out = dict(DEFAULTS)
    layers = []
    if name_or_path:
        p = _find(name_or_path)
        layers.append((json.loads(p.read_text(encoding="utf-8-sig")), p.parent))
    if project_dir and (Path(project_dir) / "brand.json").exists():
        pp = Path(project_dir) / "brand.json"
        layers.append((json.loads(pp.read_text(encoding="utf-8-sig")), pp.parent))
    for data, base in layers:
        out.update(validate(data, base))
    return out


def caption_style(brand):
    """The caption style dict CaptionTrack uses: built-in base style with brand overrides."""
    st = dict(STYLES[brand.get("base", "bold")])
    for k in ("font", "size", "color", "highlight", "stroke", "box", "upper", "max_words", "y"):
        if brand.get(k) is not None:
            st[k] = brand[k]
    return st


def save(name, data):
    if not re.match(r"^[\w-]{1,40}$", name):
        raise BrandError("Brand name: letters, numbers, - and _ only.")
    clean = validate(data)
    BRAND_DIR.mkdir(parents=True, exist_ok=True)
    p = BRAND_DIR / f"{name}.json"
    p.write_text(json.dumps({"name": name, **clean}, indent=2))
    return str(p)


def list_brands():
    return sorted(p.stem for p in BRAND_DIR.glob("*.json")) if BRAND_DIR.exists() else []


def logo_overlay(brand, W, H):
    """(RGBA image, x, y) for the brand logo, or None."""
    if not brand.get("logo"):
        return None
    from PIL import Image
    im = Image.open(brand["logo"]).convert("RGBA")
    w = int(brand.get("logo_width", 170) * W / 1080)
    im = im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)
    if brand.get("logo_opacity", 1.0) < 1:
        a = im.split()[-1].point(lambda v: int(v * brand["logo_opacity"]))
        im.putalpha(a)
    m = int(0.045 * W)
    top, bottom = int(0.08 * H) + m // 2, int(0.78 * H) - im.height - m // 2     # inside the platform safe area
    pos = brand.get("logo_position", "top-right")
    x = m if "left" in pos else W - im.width - m
    y = top if pos.startswith("top") else bottom
    return im, x, y
