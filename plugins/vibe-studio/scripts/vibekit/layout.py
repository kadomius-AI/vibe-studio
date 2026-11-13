"""Frame layout rules: platform safe zones, crop-vs-fill decisions, and face-aware caption placement.

Safe zones (9:16 Reels/TikTok/Shorts): the bottom ~22% and the right edge hold platform UI
(caption text, buttons), the top ~8% holds the status/progress bar. Captions and titles stay inside.
"""
SAFE = {"top": 0.08, "bottom": 0.78, "left": 0.06, "right": 0.90}      # fractions of W/H (vertical)
SAFE_WIDE = {"top": 0.06, "bottom": 0.90, "left": 0.06, "right": 0.94}  # horizontal / square
MAX_FACE_IN_CROP = 0.36   # face taller than this share of the frame = too tight -> use the "fill" layout


def safe_area(W, H):
    s = SAFE if H > W else SAFE_WIDE
    return int(s["left"] * W), int(s["top"] * H), int(s["right"] * W), int(s["bottom"] * H)


def choose_layout(face_h_frac_src, src_w, src_h, out_w, out_h, zoom=1.0):
    """'crop' when a face-tracked crop frames the speaker well, else 'fill' (whole frame over a blurred backdrop).

    face_h_frac_src: median face height / source height (None if no face)."""
    if out_w / out_h >= src_w / src_h:          # output is as wide as the source: plain scaling, no crop issues
        return "crop"
    if face_h_frac_src is None:
        return "fill"                            # nothing to follow: keep the whole picture
    crop_h_frac = 1.0 / zoom                     # crop uses the full source height (divided by zoom)
    face_in_out = face_h_frac_src / crop_h_frac
    return "crop" if face_in_out <= MAX_FACE_IN_CROP else "fill"


FILL_ASPECT = 1.1     # the picture in the "fill" layout: near-square around the face; clean bands above/below for text


def fill_geometry(src_w, src_h, W, H, aspect=FILL_ASPECT):
    """Where the picture sits inside a tall canvas in the 'fill' layout: (x, y, w, h)."""
    aspect = min(aspect, src_w / src_h)
    w = W
    h = int(round(W / aspect))
    y = int(H * 0.43 - h / 2)                    # above centre: a fixed caption band fits below
    return 0, max(int(H * SAFE["top"]) + int(H * 0.095), y), w, h


CAPTION_BAND = 0.075     # fixed caption band height (fraction of H): position never jumps between words


def fill_caption_y(img_h, H, geom):
    """Captions centred in a fixed band under the picture (decided once, so they never jump)."""
    _, gy, _, gh = geom
    band_top = gy + gh + int(H * 0.012)
    band_h = int(H * CAPTION_BAND)
    if band_top + band_h > int(H * SAFE["bottom"]):          # no room under the picture: fixed band inside its bottom
        band_top = gy + gh - band_h - int(H * 0.012)
    return int(band_top + (band_h - img_h) / 2)


def fill_headline_y(img_h, H, geom):
    """Headline in the band above the picture."""
    return max(int(H * SAFE["top"]), geom[1] - img_h - int(H * 0.015))


def caption_y(img_h, W, H, face_box=None, preferred=0.70, avoid=()):
    """Top y (pixels) for a caption image of height img_h: inside the safe area, never over the face.

    face_box: (x, y, w, h) in output pixels. avoid: other boxes (e.g. a headline) to keep clear of."""
    x0, top, x1, bottom = safe_area(W, H)
    pad = int(H * 0.015)
    y = int(preferred * H - img_h / 2)
    blocks = [b for b in ([face_box] if face_box is not None else []) + list(avoid) if b is not None]

    def hits(yy):
        for bx, by, bw, bh in blocks:
            # faces: protect from forehead to chin + a little; overlap test on the vertical axis only
            if yy < by + bh + pad and yy + img_h > by - pad:
                return True
        return False

    y = min(max(y, top), bottom - img_h)
    if not hits(y):
        return y
    # try below the lowest blocking box, then above the highest
    below = max(by + bh for bx, by, bw, bh in blocks) + pad
    if below + img_h <= bottom:
        return below
    above = min(by for bx, by, bw, bh in blocks) - pad - img_h
    if above >= top:
        return above
    return bottom - img_h                       # no clean spot: lowest safe position (overlap logged by QA)
