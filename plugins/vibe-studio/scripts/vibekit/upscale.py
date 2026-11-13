"""AI upscaling for low-resolution footage: Real-ESRGAN x4 (BSD-3) loaded with spandrel (MIT), on the Apple GPU."""
import numpy as np
from .env import MODELS
import urllib.request

WEIGHTS = ("RealESRGAN_x4plus.pth",
           "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from spandrel import ModelLoader
        p = MODELS / WEIGHTS[0]
        if not p.exists():
            MODELS.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(WEIGHTS[1], p.with_suffix(".part"))
            p.with_suffix(".part").rename(p)
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _model = (ModelLoader().load_from_file(str(p)).to(dev).eval(), dev)
    return _model


def upscale(bgr):
    """4x upscale a BGR uint8 image (keep inputs small: crop first)."""
    import torch
    m, dev = _load()
    x = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    with torch.no_grad():
        y = m(x)
    y = y.squeeze(0).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(y[..., ::-1])
