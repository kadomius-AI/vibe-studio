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
        # reset on camera cuts (mask changes a lot) so no "ghost" of the previous shot lingers
        if self.prev is None or self.prev.shape != m.shape or float(np.abs(m - self.prev).mean()) > 0.08:
            self.prev = m
        else:
            self.prev = self.alpha * m + (1 - self.alpha) * self.prev
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


_sam = None


def _sam_load():
    global _sam
    if _sam is None:
        import torch
        from transformers import SamModel, SamProcessor
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _sam = (SamProcessor.from_pretrained("facebook/sam-vit-base"),
                SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval(), dev)
    return _sam


def sam_mask(bgr, box, points=None):
    """Segment Anything (Apache-2.0): best object mask inside box=[x0,y0,x1,y1] (+ optional positive points)."""
    import torch
    from PIL import Image
    proc, model, dev = _sam_load()
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    kw = {"input_boxes": [[list(map(float, box))]]}
    if points:
        kw["input_points"] = [[[list(map(float, p)) for p in points]]]
    inp = proc(img, return_tensors="pt", **kw)
    sizes = (inp["original_sizes"], inp["reshaped_input_sizes"])
    inp = {k: (v.float() if v.is_floating_point() else v).to(dev) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, multimask_output=True)
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), *sizes)[0][0].numpy().astype(np.float32)
    scores = out.iou_scores.cpu()[0, 0].numpy()
    bx = (box[2] - box[0]) * (box[3] - box[1]) / (bgr.shape[0] * bgr.shape[1])
    best, best_s = None, -1
    for m, s in zip(masks, scores):
        cov = m.mean()
        if 0.02 < cov < min(0.95, bx * 0.97) and s > best_s:   # an object, not the whole box
            best, best_s = m, s
    return best, float(best_s)


def product_cutout(bgr):
    """Best-quality product mask: GrabCut finds the product, SAM refines the edge. Returns (mask, coverage)."""
    rough, cov = product_mask(bgr)
    if not (0.03 < cov < 0.9):
        return rough, cov
    ys, xs = np.where(rough > 0.5)
    h, w = bgr.shape[:2]
    pad = 0.04 * max(w, h)
    box = [max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad), min(h, ys.max() + pad)]
    try:
        m, s = sam_mask(bgr, box, points=[((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)])
    except Exception:
        m, s = None, 0
    if m is None or s < 0.8:
        return rough, cov
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    return m, float(m.mean())


def to_rgba(bgr, mask):
    """BGR + mask -> RGBA PIL image cropped to the subject."""
    from PIL import Image
    rgba = np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (mask * 255).astype(np.uint8)])
    im = Image.fromarray(rgba, "RGBA")
    bb = im.getbbox()
    return im.crop(bb) if bb else im
