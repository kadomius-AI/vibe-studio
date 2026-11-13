"""Bounded automatic color correction (inspired by video-use's bounded auto-grade, MIT).

Principles:
- Exposure is set from the FACE, not the whole frame (a dark room should not darken the speaker).
- White balance is measured on near-neutral, non-skin pixels and applied gently; skin gets a smaller share.
- Contrast stretches levels within limits; shadows are never crushed (black floor), highlights never clipped hard.
- Saturation moves toward a target within limits; skin saturation is protected.
- Sharpening is light, on brightness only, and gentler on faces.
- Every correction is clamped, so bad measurements can't produce extreme looks.
"""
import cv2
import numpy as np

# 8-bit OpenCV LAB: L 0..255, a/b centred on 128
FACE_L_TARGET = (150, 182)      # well-exposed skin, light to deep skin tones
BOUNDS = dict(gamma=(0.6, 1.25), stretch=(1.0, 1.22), wb=6.0, sat=(0.9, 1.22), sharpen=(0.0, 0.6))


def skin_mask(bgr):
    """Fast skin-likelihood mask (YCrCb range + blur), 0..1."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (0, 135, 85), (255, 180, 135)).astype(np.float32) / 255
    return cv2.GaussianBlur(m, (0, 0), 3)


def measure(frames, faces=None):
    """Stats used for grading. faces: list (per frame) of (x,y,w,h) or None."""
    Ls, As, Bs, sats, faceL, chromas = [], [], [], [], [], []
    for i, f in enumerate(frames):
        small = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        sk = skin_mask(small) > 0.5
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        hue = np.degrees(np.arctan2(lab[..., 2] - 128, lab[..., 1] - 128))
        skin_hue = (hue > 20) & (hue < 85) & (chroma > 6)          # warm, skin-like colours are never "neutral"
        neutral = (chroma < 18) & ~sk & ~skin_hue & (lab[..., 0] > 40) & (lab[..., 0] < 235)
        Ls.append(lab[..., 0].ravel())
        if neutral.sum() > 200:
            As.append(lab[..., 1][neutral]); Bs.append(lab[..., 2][neutral])
        sats.append(hsv[..., 1].ravel())
        chromas.append(float(np.median(chroma)))
        fb = faces[i] if faces and i < len(faces) else None
        if fb is not None:
            s = small.shape[1] / f.shape[1]
            x, y, w, h = [int(v * s) for v in fb]
            x0, y0 = max(0, x + w // 4), max(0, y + h // 4)
            roi = lab[y0:y0 + h // 2, x0:x0 + w // 2, 0]
            if roi.size:
                faceL.append(float(np.median(roi)))
    L = np.concatenate(Ls)
    return {
        "L_p1": float(np.percentile(L, 1)), "L_p99": float(np.percentile(L, 99)), "L_mean": float(L.mean()),
        "wb_a": float(np.median(np.concatenate(As))) - 128 if As else 0.0,
        "wb_b": float(np.median(np.concatenate(Bs))) - 128 if Bs else 0.0,
        "sat": float(np.concatenate(sats).mean() / 255),
        "face_L": float(np.median(faceL)) if faceL else None,
        "chroma": float(np.median(chromas)) if chromas else 10.0,
    }


class Grade:
    """Callable per-frame grade with parameters recorded for the edit plan."""

    def __init__(self, src_stats, ref_stats=None, strength=1.0, sharpen=0.35):
        s, r = src_stats, ref_stats or {}
        clamp = lambda v, lo_hi: float(np.clip(v, *lo_hi))
        # exposure from the face (fallback: whole-frame mean), toward the reference face level if known
        lo, hi = FACE_L_TARGET
        target = clamp(r.get("face_L") or 166, (lo, hi))
        cur = s.get("face_L") or (s["L_mean"] + 25)
        # exponent p maps the face level cur -> target on a 0..1 scale: (cur/255)^p = target/255  (p<1 brightens)
        g = np.log(max(target, 1) / 255) / np.log(max(min(cur, 254), 1) / 255)
        self.gamma = clamp(1 + (g - 1) * strength, BOUNDS["gamma"])
        # levels stretch, never crushing shadows: black floor 6, white ceiling 248
        span = max(30.0, s["L_p99"] - s["L_p1"])
        self.stretch = clamp(1 + ((230.0 / span) - 1) * 0.6 * strength, BOUNDS["stretch"])
        self.black = float(s["L_p1"])
        # white balance: pull neutral pixels toward grey
        # near-monochrome / sepia footage: its colour cast is probably the look, so correct it only lightly
        wb_k = 0.7 * strength * (0.3 if s.get("chroma", 10) < 5 else 1.0)
        self.wb_a = clamp(-s["wb_a"] * wb_k, (-BOUNDS["wb"], BOUNDS["wb"]))
        self.wb_b = clamp(-s["wb_b"] * wb_k, (-BOUNDS["wb"], BOUNDS["wb"]))
        # saturation toward the reference (or a pleasant default)
        tsat = r.get("sat") or 0.32
        self.sat = clamp(1 + (tsat / max(s["sat"], 0.05) - 1) * 0.5 * strength, BOUNDS["sat"])
        self.sharpen = clamp(sharpen, BOUNDS["sharpen"])
        self._lut = self._build_lut()

    def _build_lut(self):
        x = np.arange(256, dtype=np.float32)
        y = (x - self.black) * self.stretch + self.black * 0.55       # stretch, lift the floor slightly
        y = np.clip(y, 0, 255) / 255
        y = np.power(y, self.gamma) * 255                               # exponent <1 brightens
        y = 6 + y * (248 - 6) / 255                                      # protect blacks and whites
        # soft S-curve for a touch of "pop" without crushing
        t = y / 255
        s = t * t * (3 - 2 * t)                                          # smoothstep S-curve
        y = 255 * (t + 0.12 * (s - t))                                   # 12% of the way: gentle mid-contrast
        return np.clip(y, 0, 255).astype(np.uint8)

    def params(self):
        return {k: round(getattr(self, k), 4) for k in ("gamma", "stretch", "black", "wb_a", "wb_b", "sat", "sharpen")}

    def __call__(self, frame, t=0.0):
        sk = skin_mask(frame)[..., None]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] = cv2.LUT(lab[..., 0].astype(np.uint8), self._lut).astype(np.float32)
        # white balance (skin gets 40%)
        k = 1 - 0.6 * sk[..., 0]
        lab[..., 1] += self.wb_a * k
        lab[..., 2] += self.wb_b * k
        # saturation via chroma scaling (skin moves only half way)
        sf = 1 + (self.sat - 1) * (1 - 0.5 * sk[..., 0])
        lab[..., 1] = 128 + (lab[..., 1] - 128) * sf
        lab[..., 2] = 128 + (lab[..., 2] - 128) * sf
        if self.sharpen > 0:
            L = lab[..., 0]
            blur = cv2.GaussianBlur(L, (0, 0), 1.3)
            detail = L - blur
            detail[np.abs(detail) < 1.5] = 0                         # don't sharpen noise
            amt = self.sharpen * (1 - 0.5 * sk[..., 0])              # gentler on skin
            lab[..., 0] = L + amt * detail
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out
