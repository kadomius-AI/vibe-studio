"""Free, local voiceovers with Kokoro-82M (Apache-2.0) + exact word timings for synced captions."""
import numpy as np

VOICES = {
    # American
    "warm_female": "af_heart", "bright_female": "af_bella", "calm_female": "af_nicole",
    "deep_male": "am_fenrir", "friendly_male": "am_michael", "confident_male": "am_puck",
    # British
    "british_female": "bf_emma", "british_male": "bm_george",
}
_pipes = {}


import re

SAY = [  # how abbreviations should be spoken
    (r"\bAve\b\.?", "Avenue"), (r"\bSt\b\.?(?=\s*,|\s*$)", "Street"), (r"\bRd\b\.?", "Road"), (r"\bDr\b\.?(?=\s*,|\s*$)", "Drive"),
    (r"\bBlvd\b\.?", "Boulevard"), (r"\bLn\b\.?", "Lane"), (r"\bCt\b\.?", "Court"), (r"\bApt\b\.?", "Apartment"),
    (r"\bNJ\b", "New Jersey"), (r"\bNY\b", "New York"), (r"\bCA\b", "California"), (r"\bFL\b", "Florida"), (r"\bTX\b", "Texas"),
    (r"\bsq\.? ?ft\b", "square feet"), (r"\bBD\b", "bedrooms"), (r"\bBA\b", "baths"), (r"\bDM\b", "message"),
    (r"(\d+)\.5 baths", r"\1 and a half baths"), (r"\bw/", "with"), (r"&", "and"), (r"\bPOV\b", "P O V"),
]


def speakable(text):
    for pat, rep in SAY:
        text = re.sub(pat, rep, text)
    return text


def speak(text, voice="warm_female", speed=1.0):
    """Return (stereo float32 audio @48k, words=[{word,start,end}], seconds)."""
    from kokoro import KPipeline
    import librosa
    text = speakable(text)
    vid = VOICES.get(voice, voice)
    lang = "b" if vid.startswith("b") else "a"
    if lang not in _pipes:
        _pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    chunks, words, offset = [], [], 0.0
    for r in _pipes[lang](text, voice=vid, speed=speed):
        a = r.audio.numpy() if hasattr(r.audio, "numpy") else np.asarray(r.audio)
        for t in (getattr(r, "tokens", None) or []):
            if t.start_ts is not None and t.text.strip() and any(ch.isalnum() for ch in t.text):
                words.append({"word": t.text + (t.whitespace and "" or ""), "start": offset + float(t.start_ts),
                              "end": offset + float(t.end_ts if t.end_ts is not None else t.start_ts + 0.2)})
            elif words and t.text.strip() and not any(ch.isalnum() for ch in t.text):
                words[-1]["word"] += t.text          # keep punctuation attached to the previous word
        chunks.append(a.astype(np.float32))
        offset += len(a) / 24000
    mono = np.concatenate(chunks) if chunks else np.zeros(24000, np.float32)
    mono48 = librosa.resample(mono, orig_sr=24000, target_sr=48000)
    stereo = np.stack([mono48, mono48], 1).astype(np.float32)
    return stereo, words, len(mono48) / 48000
