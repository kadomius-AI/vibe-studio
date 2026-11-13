"""Getting footage in, transcribing it, and cutting clean multi-segment clips."""
import json, re, shutil, subprocess
from pathlib import Path
import numpy as np
from .env import FFMPEG, run, probe

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "project"


def new_project(root, name):
    p = Path(root).expanduser() / slug(name)
    for d in ("00_SOURCE", "10_WORK", "20_DELIVER"):
        (p / d).mkdir(parents=True, exist_ok=True)
    return p


def ingest(source, project):
    """Copy a local file or download a URL into 00_SOURCE. Never moves or deletes the original."""
    dst_dir = Path(project) / "00_SOURCE"
    if re.match(r"https?://", source):
        import yt_dlp
        opts = {"outtmpl": str(dst_dir / "%(id)s.%(ext)s"), "quiet": True, "noprogress": True,
                "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
                "merge_output_format": "mp4"}
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(source, download=True)
            path = Path(y.prepare_filename(info)).with_suffix(".mp4")
            if not path.exists():
                path = next(dst_dir.glob(f"{info['id']}.*"))
        return path, info.get("title", path.stem)
    src = Path(source).expanduser()
    if not src.exists():
        raise FileNotFoundError(source)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst, src.stem


def transcribe(video, out_json):
    """Word-level transcript, offline (Apple Silicon: mlx-whisper; else faster-whisper)."""
    out_json = Path(out_json)
    if out_json.exists():
        return json.loads(out_json.read_text())
    words, segs = [], []
    try:
        import mlx_whisper
        r = mlx_whisper.transcribe(str(video), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                   word_timestamps=True, verbose=None)
        for s in r["segments"]:
            segs.append({"start": s["start"], "end": s["end"], "text": s["text"].strip()})
            for w in s.get("words", []):
                words.append({"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])})
    except ImportError:
        from faster_whisper import WhisperModel
        m = WhisperModel("small", compute_type="int8")
        it, _ = m.transcribe(str(video), word_timestamps=True)
        for s in it:
            segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
            for w in s.words or []:
                words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    # drop zero-length "words" (whisper hallucination tell)
    words = [w for w in words if w["end"] - w["start"] > 0.01 and w["word"]]
    data = {"words": words, "segments": segs}
    out_json.write_text(json.dumps(data, indent=0))
    txt = "\n".join(f"[{int(s['start']//60):02d}:{s['start']%60:05.2f} -> {int(s['end']//60):02d}:{s['end']%60:05.2f}] {s['text']}"
                    for s in segs)
    out_json.with_suffix(".txt").write_text(txt)
    return data


def _audio(video):
    pcm = subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768


def quietest(a, t, win=0.12, sr=16000):
    """Time of the quietest 20 ms window within +/- win seconds of t (clean splice point)."""
    lo, hi = int(max(0, t - win) * sr), int((t + win) * sr)
    hop, fl = int(0.005 * sr), int(0.02 * sr)
    best, bt = 9.0, t
    for i in range(lo, max(lo + 1, hi - fl), hop):
        seg = a[i:i + fl]
        if len(seg) == 0:
            break
        e = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        if e < best:
            best, bt = e, (i + fl / 2) / sr
    return bt


def cut(video, segments, out, words=None):
    """Cut [[start,end],...] from video, snapping each edge to silence, with tiny audio fades.
    Returns clip-relative words (if words given) and the duration."""
    a = _audio(video)
    snapped = [(quietest(a, s), quietest(a, e)) for s, e in segments]
    parts, labels = [], ""
    has_audio = probe(video)["has_audio"]
    for i, (s, e) in enumerate(snapped):
        d = e - s
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        if has_audio:
            parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
                         f"afade=t=in:st=0:d=0.012,afade=t=out:st={max(0, d - 0.012):.3f}:d=0.012[a{i}]")
            labels += f"[v{i}][a{i}]"
        else:
            labels += f"[v{i}]"
    n = len(snapped)
    fc = ";".join(parts) + (f";{labels}concat=n={n}:v=1:a=1[v][a]" if has_audio else f";{labels}concat=n={n}:v=1:a=0[v]")
    maps = ["-map", "[v]"] + (["-map", "[a]"] if has_audio else [])
    run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-filter_complex", fc, *maps,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
         *(["-c:a", "pcm_s16le"] if has_audio else []), str(out)])
    rel, off = [], 0.0
    for s, e in snapped:
        for w in words or []:
            if w["start"] >= s - 0.02 and w["end"] <= e + 0.05:
                rel.append({"word": w["word"], "start": round(max(0, w["start"] - s) + off, 3),
                            "end": round(min(e, w["end"]) - s + off, 3)})
        off += e - s
    return rel, off


def silences(video, noise_db=-35, min_len=0.35):
    """List of (start, end) silent stretches."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(video), "-af",
                        f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"],
                       capture_output=True, text=True)
    st = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", r.stderr)]
    en = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", r.stderr)]
    return list(zip(st, en))
