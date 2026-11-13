"""Find and score the best clip candidates in a long video: words + audio reactions + visual energy.

Adapted from Chopify (MIT, (c) 2026 Mehbul Islam — https://github.com/mehbul/chopify):
the cue word lists, text-feature weights, and the per-second audio "evidence" idea (loudness rising to a
peak, bursts such as laughs/applause, spectral change). Re-implemented compactly for vibekit.

The score is a SHORTLIST tool: Claude reads the top candidates (with their reasons) and makes the final call.
Output: moments.json = [{start, end, duration, score, text, reasons, parts}, ...] best first, non-overlapping.
"""
import math
import re
import subprocess

import numpy as np

from .env import FFMPEG

# ---- cue lists (from Chopify, MIT) ----
FILLER = {"um", "uh", "uhh", "umm", "erm", "er", "hmm", "mmm", "mm", "ah", "eh"}
HOOK = ("the secret", "nobody tells you", "here's the thing", "here is the thing", "what most people", "the truth is",
        "here's why", "here is why", "imagine", "the problem is", "the biggest mistake", "let me tell you", "you need to",
        "here's how", "here is how", "the trick", "the key", "listen", "story", "never", "the worst", "the best")
PAYOFF = ("so", "therefore", "that is why", "that's why", "the result", "it turned out", "in the end", "changed", "finally",
          "which means", "the lesson", "turned out", "which is why")
EMOTION = ("love", "hate", "afraid", "scared", "amazing", "terrible", "proud", "embarrassed", "angry", "excited",
           "grateful", "hurts", "cried", "dream", "hope", "believe")
CONTROVERSY = ("i disagree", "unpopular opinion", "hot take", "controversial", "stop doing", "don't do", "do not do",
               "overrated", "underrated", "everyone is wrong", "myth", "lie", "scam")
HUMOUR = ("haha", "lol", "laugh", "joke", "funny", "hilarious", "kidding", "ridiculous")
INSIGHT = ("because", "the reason", "how it works", "step one", "first", "second", "third", "the way", "what happens",
           "if you", "the difference", "framework", "system", "lesson", "learned", "the point")
STAKES = ("lose", "losing", "lost", "risk", "money", "business", "life", "everything", "nothing", "fail", "failure",
          "broke", "fired", "quit", "survive")
META_OPEN = ("welcome back", "welcome to", "in this video", "today we", "before we start", "quick intro")
META_OUT = ("thanks for watching", "subscribe", "like and comment", "hit the bell", "see you next time", "sponsor",
            # spoken sign-offs (added after a real test picked an outro as the top moment)
            "have an amazing", "have a great", "have a good", "catch you", "see you guys", "on the flip side",
            "hope that helps", "hope that provides", "hopefully that", "that's it for", "that's all for", "until next time",
            "talk soon", "peace out", "signing off")
ANAPHORIC = ("this", "that", "it", "he", "she", "they", "those", "these", "and", "but", "so")
EARLIER = ("as i said", "like i mentioned", "earlier", "last time", "previous video", "as you know")
TEXT_WEIGHTS = {"hook": 0.18, "payoff": 0.16, "self_contained": 0.12, "arc": 0.12, "energy": 0.11,
                "quotable": 0.10, "emotion": 0.08, "controversy": 0.07, "humour": 0.06}


def _hits(low, cues):
    return sum(1 for c in cues if f" {c} " in low or low.startswith(c + " "))


def sentences(words, gap=1.2):
    """Group words into sentences on punctuation or long pauses."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        end_p = re.search(r"[.!?…][\"')\]]?$", w["word"].strip())
        nxt_gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 9
        if end_p or nxt_gap > gap:
            out.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": " ".join(x["word"] for x in cur),
                        "complete": bool(end_p)})
            cur = []
    if cur:
        out.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": " ".join(x["word"] for x in cur),
                    "complete": False})
    return out


def text_score(sents):
    """0..1 text score and a dict of parts for a window of sentences."""
    text = " ".join(s["text"] for s in sents)
    low = " " + re.sub(r"[^\w' ]", " ", text.lower()) + " "
    first = " " + re.sub(r"[^\w' ]", " ", sents[0]["text"].lower()) + " "
    last = " " + re.sub(r"[^\w' ]", " ", sents[-1]["text"].lower()) + " "
    toks = low.split()
    dur = max(0.1, sents[-1]["end"] - sents[0]["start"])
    filler = sum(t in FILLER for t in toks) / max(1, len(toks))
    first_tok = first.split()[:1]
    parts = {
        "hook": min(1.0, 0.5 * _hits(first, HOOK) + 0.35 * ("?" in sents[0]["text"]) +
                0.25 * bool(re.search(r"\b\d[\d,.$%]*\b", sents[0]["text"]))),
        "payoff": min(1.0, 0.5 * _hits(last, PAYOFF) + 0.3 * _hits(low, INSIGHT) / 3 + 0.3 * sents[-1]["complete"]),
        "self_contained": 1.0 - 0.5 * bool(first_tok and first_tok[0] in ANAPHORIC) - 0.5 * min(1, _hits(low, EARLIER))
                          - 0.6 * min(1, _hits(first, META_OPEN)) - 0.6 * min(1, _hits(low, META_OUT)),
        "arc": 1.0 if sents[-1]["complete"] and len(sents) >= 2 else 0.4,
        "energy": min(1.0, max(0.0, (len(toks) / dur - 1.6) / 1.6)),
        "quotable": min(1.0, 0.4 * (2 <= len(last.split()) <= 14) + 0.3 * _hits(low, STAKES) / 2 +
                        0.3 * any(w in low for w in (" not ", " but ", " instead ", " never "))),
        "emotion": min(1.0, _hits(low, EMOTION) / 2),
        "controversy": min(1.0, _hits(low, CONTROVERSY)),
        "humour": min(1.0, _hits(low, HUMOUR)),
    }
    parts = {k: round(max(0.0, min(1.0, v)), 3) for k, v in parts.items()}
    score = sum(TEXT_WEIGHTS[k] * v for k, v in parts.items()) - 0.6 * filler
    if _hits(low, META_OUT) or _hits(first, META_OPEN):
        score *= 0.5                       # intros/sign-offs are never the clip, however warm they sound
    return max(0.0, min(1.0, score)), parts, text


# ---- audio evidence (Chopify's idea: per-second loudness timeline) ----
def audio_timeline(video):
    """Per-second loudness (dB) and spectral change of the audio track."""
    pcm = subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    if a.size < 16000:
        return None
    n = a.size // 16000
    sec = a[:n * 16000].reshape(n, 16000)
    rms = 20 * np.log10(np.sqrt((sec ** 2).mean(1)) + 1e-9)
    frames = a[:n * 16000].reshape(n * 20, 800)
    spec = np.abs(np.fft.rfft(frames * np.hanning(800), axis=1))
    flux = np.r_[0, (np.maximum(spec[1:] - spec[:-1], 0).sum(1) / (spec[1:].sum(1) + 1e-9))]
    flux = flux.reshape(n, 20).mean(1)
    return {"rms": rms, "flux": flux}


def audio_score(tl, start, end):
    """0..1: does the audio rise to a peak / contain bursts (laughs, applause, shouts) in this window?"""
    if tl is None:
        return 0.5, {}
    rms, flux = tl["rms"], tl["flux"]
    lo, hi = max(0, int(start)), min(len(rms), max(int(start) + 1, int(math.ceil(end))))
    w = rms[lo:hi]
    if w.size == 0:
        return 0.5, {}
    base = float(np.median(rms))
    ss = lambda x: float(np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1)))
    peak_rise = ss((w.max() - base) / 18.0)
    burst = ss((w > base + 6).mean() / 0.25)
    p50, p90 = np.percentile(flux, [50, 90])
    spectral = ss((np.percentile(flux[lo:hi], 90) - p50) / max(p90 - p50, 1e-6))
    k = max(1, int(len(w) * 0.3))
    arc = ss((w[k:].mean() - w[:k].mean()) / 15.0) if len(w) > 2 else 0.0
    score = 0.3 * peak_rise + 0.25 * burst + 0.25 * spectral + 0.2 * arc
    return score, {"peak_rise": round(peak_rise, 3), "bursts": round(burst, 3), "spectral": round(spectral, 3),
                   "build_up": round(arc, 3)}


# ---- visual energy (2 fps frame differences) ----
def visual_timeline(video, fps=2):
    import cv2
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(src_fps / fps))
    prev, out, i = None, [], 0
    while cap.grab():
        if i % step == 0:
            ok, f = cap.retrieve()
            g = cv2.cvtColor(cv2.resize(f, (96, 54)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            out.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
            prev = g
        i += 1
    cap.release()
    return np.array(out), fps


def visual_score(vt, start, end):
    if vt is None:
        return 0.5
    v, fps = vt
    seg = v[int(start * fps):int(end * fps) + 1]
    if seg.size == 0 or v.size == 0:
        return 0.5
    return float(np.clip((np.percentile(seg, 75) - np.median(v)) / (np.percentile(v, 95) - np.median(v) + 1e-6), 0, 1))


def find(video, words, min_len=20, max_len=60, top=8, weights=(0.70, 0.20, 0.10)):
    """Candidate windows of whole sentences, scored 0..10, best first, non-overlapping."""
    sents = sentences(words)
    tl = audio_timeline(video)
    vt = visual_timeline(video)
    cands = []
    for i in range(len(sents)):
        for j in range(i, len(sents)):
            dur = sents[j]["end"] - sents[i]["start"]
            if dur > max_len:
                break
            if dur < min_len:
                continue
            win = sents[i:j + 1]
            ts, tparts, text = text_score(win)
            a, aparts = audio_score(tl, win[0]["start"], win[-1]["end"])
            v = visual_score(vt, win[0]["start"], win[-1]["end"])
            total = 10 * (weights[0] * ts + weights[1] * a + weights[2] * v)
            reasons = [k for k, val in sorted(tparts.items(), key=lambda kv: -kv[1] * TEXT_WEIGHTS[kv[0]]) if val >= 0.5][:3]
            if aparts.get("bursts", 0) > 0.5:
                reasons.append("audio reaction (laugh/applause/burst)")
            if aparts.get("build_up", 0) > 0.5:
                reasons.append("energy builds to a peak")
            cands.append({"start": round(win[0]["start"], 2), "end": round(win[-1]["end"], 2), "duration": round(dur, 1),
                          "score": round(total, 2), "text": text, "reasons": reasons,
                          "parts": {"text": round(ts, 3), "audio": round(a, 3), "visual": round(v, 3), **tparts, **aparts}})
    cands.sort(key=lambda c: -c["score"])
    chosen = []
    for c in cands:
        if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in chosen):
            chosen.append(c)
        if len(chosen) >= top:
            break
    return chosen
