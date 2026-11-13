"""Sound design: original synthesized SFX (no licensing issues), beat detection, voice-ducked mixing.

All audio is float32 stereo at 48 kHz internally.
"""
import subprocess
from pathlib import Path
import numpy as np
from .env import FFMPEG

SR = 48000
_rng = np.random.default_rng(7)


# ---------------- synthesized sound effects ----------------
def _env(n, attack=0.05, release=0.6, shape=2.0):
    t = np.linspace(0, 1, n, dtype=np.float32)
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    r = np.clip((1 - t) / max(release, 1e-4), 0, 1) ** shape
    return a * r


def _svf_bandpass(x, f_start, f_end, q=0.9):
    """Time-varying state-variable band-pass (smooth frequency sweep)."""
    n = len(x)
    f = np.geomspace(f_start, f_end, n).astype(np.float32) if f_start > 0 and f_end > 0 else np.linspace(f_start, f_end, n)
    g = 2 * np.sin(np.pi * np.clip(f, 20, SR / 3) / SR)
    damp = 1 / q
    low = band = 0.0
    out = np.empty(n, np.float32)
    for i in range(n):
        high = x[i] - low - damp * band
        band += g[i] * high
        low += g[i] * band
        out[i] = band
    return out


def _stereo(m, pan_from=0.0, pan_to=0.0):
    p = np.linspace(pan_from, pan_to, len(m), dtype=np.float32)
    l = m * np.cos((p + 1) * np.pi / 4)
    r = m * np.sin((p + 1) * np.pi / 4)
    return np.stack([l, r], 1)


def _norm(x, peak=0.9):
    m = np.abs(x).max() or 1.0
    return (x / m * peak).astype(np.float32)


def sfx(kind, dur=None):
    """Return a stereo float32 array for: whoosh, whoosh_short, riser, hit, boom, pop, click, swish_up."""
    if kind in ("whoosh", "whoosh_short"):
        d = dur or (0.55 if kind == "whoosh" else 0.32)
        n = int(SR * d)
        noise = _rng.standard_normal(n).astype(np.float32)
        half = n // 2
        sweep = np.concatenate([_svf_bandpass(noise[:half], 350, 3200, 1.4), _svf_bandpass(noise[half:], 3200, 600, 1.4)])
        return _stereo(_norm(sweep * _env(n, 0.45, 0.55, 1.6)), -0.6, 0.6)
    if kind == "swish_up":
        n = int(SR * (dur or 0.4))
        noise = _rng.standard_normal(n).astype(np.float32)
        return _stereo(_norm(_svf_bandpass(noise, 500, 6000, 2.0) * _env(n, 0.7, 0.3)), 0, 0)
    if kind == "riser":
        d = dur or 1.4
        n = int(SR * d)
        t = np.arange(n) / SR
        noise = _svf_bandpass(_rng.standard_normal(n).astype(np.float32), 300, 7000, 2.5)
        tone = np.sin(2 * np.pi * np.cumsum(np.geomspace(180, 900, n)) / SR).astype(np.float32)
        env = (t / d) ** 2.2
        return _stereo(_norm((0.7 * noise + 0.3 * tone) * env * _env(n, 0.01, 0.04)), 0, 0)
    if kind in ("hit", "boom"):
        d = dur or (0.7 if kind == "hit" else 1.6)
        n = int(SR * d)
        t = np.arange(n) / SR
        f = 48 + 90 * np.exp(-t * 18)                       # pitch drop = punchy "808" thump
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (5.5 if kind == "hit" else 2.2))
        click = _rng.standard_normal(n) * np.exp(-t * 90) * 0.5
        sig = np.tanh(2.2 * (body + click)).astype(np.float32)
        return _stereo(_norm(sig), 0, 0)
    if kind == "pop":
        n = int(SR * (dur or 0.11))
        t = np.arange(n) / SR
        sig = np.sin(2 * np.pi * np.cumsum(np.geomspace(1300, 520, n)) / SR) * np.exp(-t * 38)
        return _stereo(_norm(sig.astype(np.float32), 0.6), 0, 0)
    if kind == "click":
        n = int(SR * 0.03)
        t = np.arange(n) / SR
        sig = _rng.standard_normal(n) * np.exp(-t * 400)
        return _stereo(_norm(sig.astype(np.float32), 0.5), 0, 0)
    raise ValueError(kind)


# ---------------- io ----------------
def load(path, sr=SR):
    """Decode any audio/video file to stereo float32 at sr."""
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def save_wav(path, audio, sr=SR):
    import soundfile as sf
    sf.write(str(path), np.clip(audio, -1, 1), sr, subtype="PCM_16")
    return str(path)


# ---------------- beats ----------------
def beats(audio_or_path):
    """Beat times (s) and tempo (BPM) of a music track."""
    import librosa
    if isinstance(audio_or_path, (str, Path)):
        y, sr = librosa.load(str(audio_or_path), sr=22050, mono=True)
    else:
        y = librosa.resample(audio_or_path.mean(1), orig_sr=SR, target_sr=22050)
        sr = 22050
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    return librosa.frames_to_time(frames, sr=sr).tolist(), float(np.atleast_1d(tempo)[0])


def snap_durations(durations, beat_times, min_dur=0.9):
    """Adjust scene durations so every cut lands on a beat (keeps total length close)."""
    if not beat_times or len(beat_times) < 4:
        return durations
    bt = np.array(beat_times)
    out, t = [], 0.0
    for d in durations:
        target = t + d
        cands = bt[bt >= t + min_dur]
        end = float(cands[np.argmin(np.abs(cands - target))]) if len(cands) else target
        out.append(round(end - t, 3))
        t = end
    return out


# ---------------- mixing ----------------
def _env_follow(x, win=0.05):
    m = np.abs(x).mean(1) if x.ndim == 2 else np.abs(x)
    k = max(1, int(SR * win))
    return np.convolve(m, np.ones(k) / k, mode="same")


def mix(duration, voice=None, music=None, sfx_events=(), music_db=-14.0, duck_db=-11.0, voice_db=0.0,
        bed=None, bed_db=0.0):
    """Mix a timeline. voice/music/bed: stereo arrays (start at 0). sfx_events: [(time_s, kind_or_array, gain_db)].
    Music is ducked under the voice (and under the bed audio, e.g. a speaker's original sound)."""
    n = int(SR * duration)
    out = np.zeros((n, 2), np.float32)

    def add(sig, start=0, db=0.0):
        if sig is None:
            return
        s = int(start * SR)
        if s >= n:
            return
        seg = sig[: n - s] * (10 ** (db / 20))
        out[s:s + len(seg)] += seg

    lead = np.zeros(n, np.float32)
    for src, db in ((voice, voice_db), (bed, bed_db)):
        if src is not None:
            add(src, 0, db)
            e = _env_follow(src[:n])
            lead[:len(e)] = np.maximum(lead[:len(e)], e)
    if music is not None:
        m = music
        if len(m) < n:  # loop music to fill
            m = np.tile(m, (int(np.ceil(n / len(m))), 1))
        m = m[:n].copy()
        speaking = lead > (lead.max() * 0.06 if lead.max() > 0 else 1)
        g = np.where(speaking, 10 ** (duck_db / 20), 1.0).astype(np.float32)
        k = int(SR * 0.25)
        g = np.convolve(g, np.ones(k) / k, mode="same")          # smooth ducking
        fade = np.ones(n, np.float32)
        fl = min(n, int(SR * 1.5))
        fade[-fl:] = np.linspace(1, 0, fl)
        fi = min(n, int(SR * 0.3))
        fade[:fi] *= np.linspace(0, 1, fi)
        out += m * (10 ** (music_db / 20)) * (g * fade)[:, None]
    for t, kind, db in sfx_events:
        sig = sfx(kind) if isinstance(kind, str) else kind
        add(sig, max(0.0, t), db)
    peak = np.abs(out).max()
    if peak > 0.98:
        out *= 0.98 / peak
    return out


def isolate_voice(path):
    """Separate the speaking voice from music/noise with Demucs htdemucs (MIT). Returns stereo float32 @48k."""
    import torch
    import librosa
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    m = get_model("htdemucs").eval()
    a = load(path, sr=m.samplerate)
    wav = torch.from_numpy(a.T.copy())[None]
    ref = wav.mean(1, keepdim=True)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        srcs = apply_model(m, (wav - mean) / std, device=dev, split=True, overlap=0.25, progress=False)[0] * std + mean
    v = srcs[m.sources.index("vocals")].numpy()
    if m.samplerate != SR:
        v = np.stack([librosa.resample(ch, orig_sr=m.samplerate, target_sr=SR) for ch in v])
    return v.T.astype(np.float32)
