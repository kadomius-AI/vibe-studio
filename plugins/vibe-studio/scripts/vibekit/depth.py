"""2.5D parallax camera moves on still photos using Depth Anything V2 Small (Apache-2.0).

Near things move more than far things, so a photo feels like a real camera move.
"""
import cv2
import numpy as np

_pipe = None


def depth_map(bgr):
    """0..1 depth (1 = near) at image size."""
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
    from PIL import Image
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (int(w * 768 / max(w, h)), int(h * 768 / max(w, h)))) if max(w, h) > 768 else bgr
    res = _pipe(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    d = np.array(res["predicted_depth"]).astype(np.float32).squeeze()
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    d = (d - np.percentile(d, 2)) / max(1e-6, np.percentile(d, 98) - np.percentile(d, 2))
    d = np.clip(d, 0, 1)
    # soften depth edges so the warp doesn't tear around objects
    return cv2.GaussianBlur(d, (0, 0), max(2, w / 300))


def parallax_frames(bgr, W, H, n, move="push", strength=1.0, depth=None):
    """Yield n frames (W x H) of a 2.5D camera move over the photo.
    move: push (dolly in), pull (dolly out), left, right, up, orbit."""
    if depth is None:
        depth = depth_map(bgr)
    ih, iw = bgr.shape[:2]
    # cover-fit the photo (a bit oversized so edges never show)
    s0 = max(W / iw, H / ih) * 1.16
    img = cv2.resize(bgr, (int(iw * s0), int(ih * s0)), interpolation=cv2.INTER_CUBIC)
    dep = cv2.resize(depth, (img.shape[1], img.shape[0]))
    ox, oy = (img.shape[1] - W) / 2, (img.shape[0] - H) / 2
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # depth sampled at output grid (approximate backward warp)
    dgrid = cv2.remap(dep, xs + ox, ys + oy, cv2.INTER_LINEAR)
    cx, cy = W / 2, H * 0.48
    amp = 0.055 * strength
    for k in range(n):
        p = k / max(1, n - 1)
        e = p * p * (3 - 2 * p)
        dx = dy = 0.0
        z = 0.0
        if move == "push":
            z = amp * 1.2 * e
        elif move == "pull":
            z = amp * 1.2 * (1 - e)
        elif move == "left":
            dx = amp * W * (0.5 - e)
        elif move == "right":
            dx = amp * W * (e - 0.5)
        elif move == "up":
            dy = amp * H * 0.6 * (0.5 - e)
        elif move == "orbit":
            dx = amp * W * (e - 0.5); z = amp * 0.6 * e
        # per-pixel scale/shift weighted by nearness
        scale = 1 + z * (0.35 + 0.65 * dgrid)
        mx = cx + (xs - cx) / scale - dx * (0.25 + 0.75 * dgrid) + ox
        my = cy + (ys - cy) / scale - dy * (0.25 + 0.75 * dgrid) + oy
        yield cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
