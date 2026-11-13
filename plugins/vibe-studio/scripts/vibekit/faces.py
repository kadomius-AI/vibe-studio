"""Face finding and a smooth 'virtual camera' that follows the speaker when reframing."""
import cv2
import numpy as np
from .env import model_path


class FaceFinder:
    def __init__(self, min_score=0.6):
        self.det = None
        self.min_score = min_score

    def find(self, frame):
        """Return faces as (x, y, w, h, score) in frame pixels, biggest first."""
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        sh, sw = small.shape[:2]
        if self.det is None:
            self.det = cv2.FaceDetectorYN.create(model_path("yunet"), "", (sw, sh), self.min_score, 0.3, 50)
        self.det.setInputSize((sw, sh))
        _, faces = self.det.detect(small)
        out = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = (f[:4] / scale).tolist()
                out.append((x, y, fw, fh, float(f[-1])))
        out.sort(key=lambda r: r[2] * r[3], reverse=True)
        return out


def track(video, sample_every=0.2, cut_threshold=0.45, prefer_x=None):
    """Sample the video, find the main face over time and detect hard scene cuts.

    Returns dict(fps, n, w, h, samples=[(t, cx, cy, fh) or (t, None...)], cuts=[t...]).
    prefer_x: 0..1 horizontal hint; picks the face closest to it (e.g. 0.25 = left speaker).
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps * sample_every)))
    ff = FaceFinder()
    samples, cuts, prev_hist, last = [], [], None, None
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = i / fps
            small = cv2.resize(frame, (160, 90))
            hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > cut_threshold:
                cuts.append(t)
                last = None
            prev_hist = hist
            faces = ff.find(frame)
            pick = None
            if faces:
                if prefer_x is not None:
                    pick = min(faces, key=lambda f: abs((f[0] + f[2] / 2) / W - prefer_x))
                elif last is not None:
                    big = faces[0][2] * faces[0][3]
                    near = [f for f in faces if f[2] * f[3] > 0.45 * big]
                    pick = min(near, key=lambda f: abs(f[0] + f[2] / 2 - last))
                else:
                    pick = faces[0]
            if pick:
                cx, cy = pick[0] + pick[2] / 2, pick[1] + pick[3] / 2
                last = cx
                samples.append((t, cx, cy, pick[3]))
            else:
                samples.append((t, None, None, None))
        i += 1
    cap.release()
    return {"fps": fps, "n": n or i, "w": W, "h": H, "samples": samples, "cuts": cuts}


def crop_path(tr, out_aspect=9 / 16, zoom=1.0, eye_y=0.38, smooth_s=0.8, deadzone=0.04):
    """Per-frame crop boxes (x, y, cw, ch) that follow the face smoothly and snap at scene cuts."""
    W, H, fps, n = tr["w"], tr["h"], tr["fps"], tr["n"]
    ch = H / zoom
    cw = ch * out_aspect
    if cw > W:                      # source narrower than target: limit by width
        cw = W / zoom
        ch = cw / out_aspect
    times = np.arange(n) / fps
    s = tr["samples"]
    st = np.array([x[0] for x in s]) if s else np.array([0.0])
    sx = np.array([x[1] if x[1] is not None else np.nan for x in s]) if s else np.array([np.nan])
    sy = np.array([x[2] if x[2] is not None else np.nan for x in s]) if s else np.array([np.nan])
    bounds = [0.0] + list(tr["cuts"]) + [times[-1] + 1 if len(times) else 1]
    cx = np.full(n, W / 2.0)
    cy = np.full(n, H / 2.0)
    for a, b in zip(bounds[:-1], bounds[1:]):
        fm = (times >= a) & (times < b)
        sm = (st >= a) & (st < b) & ~np.isnan(sx)
        if not fm.any():
            continue
        if sm.any():
            # robust per-scene path: interpolate, then heavy moving average + deadzone
            px = np.interp(times[fm], st[sm], sx[sm])
            py = np.interp(times[fm], st[sm], sy[sm])
            k = max(1, int(smooth_s * fps)) | 1
            if len(px) > k:
                ker = np.ones(k) / k
                px = np.convolve(np.pad(px, k // 2, mode="edge"), ker, mode="valid")
                py = np.convolve(np.pad(py, k // 2, mode="edge"), ker, mode="valid")
            # deadzone: hold position until the subject moves noticeably
            hx, hy = px[0], py[0]
            for j in range(len(px)):
                if abs(px[j] - hx) > deadzone * cw:
                    hx += (px[j] - hx) - np.sign(px[j] - hx) * deadzone * cw
                if abs(py[j] - hy) > deadzone * ch:
                    hy += (py[j] - hy) - np.sign(py[j] - hy) * deadzone * ch
                px[j], py[j] = hx, hy
            cx[fm], cy[fm] = px, py
    x = np.clip(cx - cw / 2, 0, W - cw)
    y = np.clip(cy - eye_y * ch, 0, H - ch)
    return [(float(a), float(b), float(cw), float(ch)) for a, b in zip(x, y)]
