"""Cut people (MediaPipe selfie segmenter) or products (OpenCV GrabCut) out of images/frames."""
import cv2
import numpy as np
from .env import model_path

_seg = None


def person_mask(bgr):
    """Soft 0..1 mask of the person/people in a BGR frame."""
    global _seg
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    if _seg is None:
        opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=model_path("selfie")),
                                            output_confidence_masks=True)
        _seg = vision.ImageSegmenter.create_from_options(opts)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = _seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    m = res.confidence_masks[0].numpy_view().astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]))
    return np.clip(m, 0, 1)


class MaskSmoother:
    """Temporal smoothing so video masks don't flicker."""

    def __init__(self, alpha=0.6):
        self.prev, self.alpha = None, alpha

    def __call__(self, m):
        self.prev = m if self.prev is None else self.alpha * m + (1 - self.alpha) * self.prev
        return self.prev


def product_mask(bgr, iters=6):
    """Mask for a product photo, assuming the product is roughly centered on a plainer background."""
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.04), int(h * 0.04), int(w * 0.92), int(h * 0.92))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    # keep the largest blob, soften the edge
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8))
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    coverage = float(m.mean())
    return m, coverage


def to_rgba(bgr, mask):
    """BGR + mask -> RGBA PIL image cropped to the subject."""
    from PIL import Image
    rgba = np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (mask * 255).astype(np.uint8)])
    im = Image.fromarray(rgba, "RGBA")
    bb = im.getbbox()
    return im.crop(bb) if bb else im
