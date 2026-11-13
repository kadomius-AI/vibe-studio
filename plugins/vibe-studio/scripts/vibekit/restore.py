"""Detail restoration for old / low-resolution footage, on the Apple GPU.

- Upscaling: Real-ESRGAN general x4 v3 (BSD-3) — small and fast, good on compressed video.
- Faces: GFPGAN v1.4 (Apache-2.0) on landmark-aligned 512px crops, blended back softly so identity
  and expression stay true to the original (strength < 1).
Both loaded with spandrel (MIT); weights download once from the official GitHub releases.
"""
import urllib.request
import cv2
import numpy as np
from .env import MODELS

WEIGHTS = {
    "sr": ("realesr-general-x4v3.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
    "face": ("GFPGANv1.4.pth", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"),
}
# FFHQ 5-point template for 512px aligned faces (right eye, left eye, nose, right mouth, left mouth)
FFHQ_512 = np.array([[192.98138, 239.94708], [318.90277, 240.19360], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
_models = {}


def _model(key):
    if key not in _models:
        import torch
        from spandrel import ModelLoader
        name, url = WEIGHTS[key]
        p = MODELS / name
        if not p.exists():
            MODELS.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, p.with_suffix(".part"))
            p.with_suffix(".part").rename(p)
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _models[key] = (ModelLoader().load_from_file(str(p)).to(dev).eval(), dev)
    return _models[key]


def _run(key, bgr):
    import torch
    m, dev = _model(key)
    x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    with torch.no_grad():
        y = m(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    return np.ascontiguousarray(y.squeeze(0).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()[..., ::-1])


def upscale(bgr):
    """4x super-resolution."""
    return _run("sr", bgr)


class FaceRestorer:
    """Restore the main face(s) in a frame. strength 0..1 blends restored over original."""

    def __init__(self, strength=0.65, max_faces=2):
        from .faces import FaceFinder
        self.ff = FaceFinder(min_score=0.7)
        self.strength = float(np.clip(strength, 0, 1))
        self.max_faces = max_faces
        self.det = None
        mask = np.zeros((512, 512), np.float32)
        cv2.ellipse(mask, (256, 290), (190, 235), 0, 0, 360, 1, -1)
        self.mask512 = cv2.GaussianBlur(mask, (0, 0), 22)

    def _landmarks5(self, frame):
        """YuNet gives 5 landmarks per face: right eye, left eye, nose, right mouth, left mouth."""
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        ff = self.ff
        ff.find(small)  # ensures detector exists with right size
        _, faces = ff.det.detect(small)
        out = []
        if faces is not None:
            for f in sorted(faces, key=lambda r: -r[2] * r[3])[: self.max_faces]:
                if f[-1] < 0.7 or f[3] / scale < 40:
                    continue
                out.append(f[4:14].reshape(5, 2) / scale)
        return out

    def __call__(self, frame, t=0.0):
        if self.strength <= 0:
            return frame
        out = frame
        for lm in self._landmarks5(frame):
            M, _ = cv2.estimateAffinePartial2D(lm.astype(np.float32), FFHQ_512, method=cv2.LMEDS)
            if M is None:
                continue
            crop = cv2.warpAffine(frame, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            fixed = _run("face", crop)
            if fixed.shape[:2] != (512, 512):
                fixed = cv2.resize(fixed, (512, 512))
            blended = cv2.addWeighted(fixed, self.strength, crop, 1 - self.strength, 0)
            Minv = cv2.invertAffineTransform(M)
            h, w = frame.shape[:2]
            back = cv2.warpAffine(blended, Minv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            m = cv2.warpAffine(self.mask512, Minv, (w, h), flags=cv2.INTER_LINEAR)[..., None]
            out = (back * m + out * (1 - m)).astype(np.uint8)
        return out
