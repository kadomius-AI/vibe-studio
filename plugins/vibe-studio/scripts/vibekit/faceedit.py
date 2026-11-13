"""Generative face edits (beard, stubble, clean-shaven, custom) on photos and talking-head video.

Photos: MediaPipe face landmarks -> precise region mask -> SDXL Inpainting (diffusers) -> blended back.
Video (experimental): edit one keyframe, then carry the edit to every frame by warping it along the face
landmarks (piecewise-affine), keeping each frame's real mouth so speech stays natural.

Only for your own face or people who have agreed. Never to impersonate or mislead about who someone is.
"""
import cv2
import numpy as np
from .env import model_path

FACE_OVAL_LOWER = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
NOSE_BOTTOM = 2
CHEEK_L, CHEEK_R = 205, 425          # mid-cheek points (beard line)

PRESETS = {  # prompt, region, extra negative, strength
    "beard":   ("a man with a thick, full, well-groomed natural beard and mustache, realistic facial hair, sharp detail, natural skin",
                "lower", "patchy, streaks, lines, scratches", 0.95),
    "stubble": ("a man with short dark stubble and five o'clock shadow, realistic facial hair, natural skin texture",
                "lower", "long beard, streaks, lines", 0.9),
    "shave":   ("freshly clean-shaven man, bare skin on chin jaw and upper lip, natural matte skin with visible pores and fine texture, realistic photo",
                "lower", "beard, stubble, mustache, facial hair, hair, shadow, streaks", 0.99),
}
NEGATIVE = "cartoon, painting, illustration, blurry, deformed, distorted face, extra mouth, extra teeth, plastic skin, artifacts"

_lm_image = _lm_video = _pipe = None


# ---------------- landmarks ----------------
def _landmarker(video=False):
    global _lm_image, _lm_video
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
    if video and _lm_video is None or not video and _lm_image is None:
        opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model_path("facemesh")),
                                            running_mode=mode, num_faces=1)
        lm = vision.FaceLandmarker.create_from_options(opts)
        if video:
            _lm_video = lm
        else:
            _lm_image = lm
    return _lm_video if video else _lm_image


def landmarks(bgr, ts_ms=None):
    """(478, 2) pixel landmarks of the main face, or None."""
    import mediapipe as mp
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    res = _landmarker(ts_ms is not None).detect_for_video(img, int(ts_ms)) if ts_ms is not None else _landmarker().detect(img)
    if not res.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)


# ---------------- masks ----------------
def region_mask(shape, L, region="lower", keep_mouth=True):
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    face_h = np.linalg.norm(L[10] - L[152])
    if region in ("lower", "chin"):
        jaw = L[FACE_OVAL_LOWER].copy()
        # beards grow under the jaw line onto the upper neck
        center = L[FACE_OVAL_LOWER].mean(0)
        down = np.array([0, 1], np.float32)
        # push the jaw outline out (sides) and down (neck) so edge stubble is covered too
        jaw = center + (jaw - center) * 1.06 + down * face_h * (0.11 if region == "lower" else 0.08)
        top_y = L[NOSE_BOTTOM][1] + face_h * 0.01
        if region == "chin":
            sel = [i for i, k in enumerate(FACE_OVAL_LOWER) if k in (365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136)]
            pts = np.vstack([[[L[61][0] - face_h * 0.09, top_y]], jaw[sel],
                             [[L[291][0] + face_h * 0.09, top_y]]])
        else:
            cheeks = np.array([[L[CHEEK_R][0], top_y], [L[CHEEK_L][0], top_y]], np.float32)
            pts = np.vstack([jaw, cheeks[::-1]])
        cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    elif region == "face":
        oval = [10, 338, 297, 332, 284, 251, 389, 356] + FACE_OVAL_LOWER + [162, 21, 54, 103, 67, 109]
        cv2.fillPoly(m, [L[oval].astype(np.int32)], 255)
    if keep_mouth and region != "face":
        lips = L[LIPS_OUTER].astype(np.int32)
        cv2.fillPoly(m, [lips], 0)
    k = max(3, int(face_h * 0.04)) | 1
    return cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255.0


# ---------------- generative inpainting ----------------
def _pipeline():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForInpainting
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if dev == "cuda" else torch.float32     # fp32 on Apple GPU avoids black/NaN images
        _pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                          torch_dtype=dtype, use_safetensors=True)
        _pipe = _pipe.to(dev)
        _pipe.set_progress_bar_config(disable=True)
    return _pipe


def inpaint(bgr, mask, prompt, negative=NEGATIVE, steps=30, strength=0.93, guidance=7.5, seed=7, blur=0):
    """SDXL inpaint the masked area (mask 0..1) at 1024 px; returns the full-size edited BGR image."""
    import torch
    from PIL import Image
    pipe = _pipeline()
    h, w = bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (1024, 1024)), cv2.COLOR_BGR2RGB))
    msk = Image.fromarray((cv2.resize(mask, (1024, 1024)) * 255).astype(np.uint8))
    if blur:
        msk = pipe.mask_processor.blur(msk, blur_factor=blur)
    g = torch.Generator("cpu").manual_seed(seed)
    out = pipe(prompt=prompt, negative_prompt=negative, image=img, mask_image=msk, num_inference_steps=steps,
               strength=strength, guidance_scale=guidance, generator=g).images[0]
    return cv2.resize(cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_LANCZOS4)


def _face_crop_box(L, shape, scale=2.3):
    h, w = shape[:2]
    x0, y0 = L.min(0)
    x1, y1 = L.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + (y1 - y0) * 0.12
    s = max(x1 - x0, y1 - y0) * scale
    s = min(s, w, h)
    bx = int(np.clip(cx - s / 2, 0, w - s)); by = int(np.clip(cy - s / 2, 0, h - s))
    return bx, by, int(s)


def _match_color(src, ref, mask):
    """Shift src colors (LAB mean) toward ref inside a ring around the mask so the edit blends with the lighting."""
    ring = (cv2.dilate((mask > 0.1).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & (mask < 0.05)
    if ring.sum() < 50:
        return src
    a = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = b[ring].mean(0) - a[ring].mean(0)
    a += d * np.array([0.6, 0.8, 0.8], np.float32)
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def edit_image(bgr, edit="beard", prompt=None, seed=7, steps=30, strength=None):
    """Edit one photo. edit: beard|stubble|shave or 'custom' with a prompt. Returns (edited, mask, landmarks)."""
    return _edit_one(bgr, edit, prompt, seed, steps, strength)


def _edit_one(bgr, edit, prompt, seed, steps, strength):
    L = landmarks(bgr)
    if L is None:
        raise ValueError("No face found.")
    p, region, neg_extra, st = PRESETS.get(edit, (prompt or "", "lower", "", 0.93))
    if prompt:
        p = prompt
    bx, by, s = _face_crop_box(L, bgr.shape, scale=1.75)     # tighter crop = more pixels for the beard area
    crop = bgr[by:by + s, bx:bx + s].copy()
    Lc = L - [bx, by]
    m = region_mask(crop.shape, Lc, region)
    src = crop
    if edit == "shave":
        # remove existing hair first: fill the area with smooth surrounding skin so the model can't copy stubble
        hole = (m > 0.2).astype(np.uint8) * 255
        filled = cv2.inpaint(crop, hole, max(5, s // 60), cv2.INPAINT_TELEA)
        filled = cv2.bilateralFilter(filled, 15, 60, 15)
        src = (filled * m[..., None] + crop * (1 - m[..., None])).astype(np.uint8)
        st = 0.8                                  # repaint with natural skin texture on top
    gen = inpaint(src, m, p, negative=NEGATIVE + ", " + neg_extra, seed=seed, steps=steps,
                  strength=strength or st, blur=8)
    gen = _match_color(gen, crop, m)
    # final blend uses a slightly wider, softer edge so no seams show
    mb = cv2.GaussianBlur(m, (0, 0), max(2, s / 160))
    blended = (gen * mb[..., None] + crop * (1 - mb[..., None])).astype(np.uint8)
    out = bgr.copy()
    out[by:by + s, bx:bx + s] = blended
    full_mask = np.zeros(bgr.shape[:2], np.float32)
    full_mask[by:by + s, bx:bx + s] = mb
    return out, full_mask, L


# ---------------- video: carry a keyframe edit along the face ----------------
WARP_IDS = sorted(set(FACE_OVAL_LOWER + LIPS_OUTER + [1, 2, 4, 5, 6, 98, 327, 205, 425, 50, 280, 116, 345, 13, 14, 78, 308]))


def _tri_warp(src, dst_img, s_pts, d_pts, tris, alpha_src=None):
    """Piecewise-affine warp of src into dst_img (in place) over triangles."""
    for t in tris:
        s, d = s_pts[t], d_pts[t]
        r = cv2.boundingRect(d.astype(np.float32))
        x, y, w, h = r
        if w <= 1 or h <= 1 or x < 0 or y < 0 or x + w > dst_img.shape[1] or y + h > dst_img.shape[0]:
            continue
        M = cv2.getAffineTransform(s.astype(np.float32), (d - [x, y]).astype(np.float32))
        patch = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        tm = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(tm, (d - [x, y]).astype(np.int32), 1.0, cv2.LINE_AA)
        roi = dst_img[y:y + h, x:x + w]
        if patch.ndim == 3:
            roi[:] = patch * tm[..., None] + roi * (1 - tm[..., None])
        else:
            roi[:] = patch * tm + roi * (1 - tm)


class VideoFaceEditor:
    """Apply a generative face edit to every frame of a talking-head clip (experimental)."""

    def __init__(self, key_frame, edit="beard", prompt=None, seed=7):
        self.edited, self.mask, L = edit_image(key_frame, edit, prompt, seed=seed)
        self.key = key_frame
        self.Lk = L[WARP_IDS]
        rect = (0, 0, key_frame.shape[1], key_frame.shape[0])
        sub = cv2.Subdiv2D(rect)
        for p in self.Lk:
            sub.insert((float(np.clip(p[0], 0, rect[2] - 1)), float(np.clip(p[1], 0, rect[3] - 1))))
        idx = {tuple(np.round(p, 1)): i for i, p in enumerate(self.Lk)}
        tris = []
        for t in sub.getTriangleList():
            pts = [tuple(np.round((t[0], t[1]), 1)), tuple(np.round((t[2], t[3]), 1)), tuple(np.round((t[4], t[5]), 1))]
            ids = [self._nearest(p) for p in pts]
            if len(set(ids)) == 3:
                tris.append(ids)
        self.tris = np.array(tris, np.int32)
        self.prev = None
        self.ms = 0

    def _nearest(self, p):
        return int(np.argmin(np.linalg.norm(self.Lk - np.array(p), axis=1)))

    def __call__(self, frame, t):
        self.ms += 34
        L = landmarks(frame, ts_ms=self.ms)
        if L is None:
            return frame
        Lt = L[WARP_IDS]
        self.prev = Lt if self.prev is None or np.abs(Lt - self.prev).mean() > 25 else 0.6 * Lt + 0.4 * self.prev
        Lt = self.prev
        warped = frame.copy()
        _tri_warp(self.edited, warped, self.Lk, Lt, self.tris)
        wm = np.zeros(frame.shape[:2], np.float32)
        _tri_warp(self.mask, wm, self.Lk, Lt, self.tris)
        # keep this frame's real mouth (speech) and match this frame's lighting
        lips = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(lips, [L[LIPS_OUTER].astype(np.int32)], 255)
        wm *= 1 - cv2.GaussianBlur(lips, (9, 9), 0).astype(np.float32) / 255
        warped = _match_color(warped, frame, wm)
        return (warped * wm[..., None] + frame * (1 - wm[..., None])).astype(np.uint8)
