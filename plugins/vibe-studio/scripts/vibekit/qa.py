"""Quick quality report: specs, loudness and a contact sheet of stills Claude can look at."""
import cv2
import numpy as np
from pathlib import Path
from .env import probe, loudness


def contact_sheet(video, out_png, n=6, height=480):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tiles = []
    for k in range(n):
        idx = int((k + 0.5) * total / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height))
        cv2.putText(f, f"{idx / fps:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(f, f"{idx / fps:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(f)
    cap.release()
    if tiles:
        cv2.imwrite(str(out_png), np.hstack(tiles))
    return str(out_png)


def report(path):
    path = Path(path)
    info = probe(path)
    sheet = contact_sheet(path, path.with_name(path.stem + "_sheet.png"))
    return {"file": str(path), "size": [info["width"], info["height"]], "fps": round(info["fps"], 2),
            "duration": round(info["duration"], 2), "lufs": loudness(path) if info["has_audio"] else None,
            "contact_sheet": sheet}
