"""Word-by-word captions: group words into short lines and highlight the word being spoken."""
import re
from .text import render_text

STYLES = {
    # bold social style: white words, the spoken word in yellow
    "bold": dict(font="Montserrat-ExtraBold.ttf", size=82, color="#FFFFFF", highlight="#FFD400",
                 stroke=7, upper=False, max_words=3, y=0.66),
    # clean premium style for luxury / brand content
    "clean": dict(font="Montserrat-SemiBold.ttf", size=64, color="#FFFFFF", highlight="#F2D9A0",
                  stroke=0, upper=False, max_words=4, y=0.72, box="#00000088"),
    # loud all-caps style for ads and hooks
    "punch": dict(font="Anton.ttf", size=96, color="#FFFFFF", highlight="#39FF88",
                  stroke=6, upper=True, max_words=2, y=0.62),
}


def chunk(words, max_words=3, max_gap=0.45, max_chars=18):
    """Group words into caption chunks. words: [{word,start,end}] (seconds, clip-relative)."""
    chunks, cur = [], []
    for w in words:
        txt = w["word"].strip()
        if not txt:
            continue
        brk = False
        if cur:
            gap = w["start"] - cur[-1]["end"]
            chars = sum(len(x["word"]) + 1 for x in cur) + len(txt)
            brk = (len(cur) >= max_words or gap > max_gap or chars > max_chars
                   or re.search(r"[.!?,;:]$", cur[-1]["word"].strip()))
        if brk:
            chunks.append(cur)
            cur = []
        cur.append({"word": txt, "start": w["start"], "end": w["end"]})
    if cur:
        chunks.append(cur)
    return chunks


class CaptionTrack:
    """Pre-renders caption images and returns the right one for any time t."""

    def __init__(self, words, frame_w, frame_h, style="bold", y=None, highlight=None):
        st = dict(STYLES.get(style, STYLES["bold"]))
        if highlight:
            st["highlight"] = highlight
        self.st = st
        self.W, self.H = frame_w, frame_h
        self.y = y if y is not None else st["y"]
        scale = frame_w / 1080
        self.kw = dict(font=st["font"], size=int(st["size"] * scale), color=st["color"],
                       stroke=int(st["stroke"] * scale), upper=st["upper"], max_width=int(frame_w * 0.84),
                       highlight_color=st["highlight"], box=st.get("box"))
        self.chunks = chunk(words, max_words=st["max_words"])
        self.cache = {}
        # display window per chunk: from first word start to next chunk start (max 0.6s hold)
        self.windows = []
        for i, c in enumerate(self.chunks):
            end = c[-1]["end"] + 0.25
            if i + 1 < len(self.chunks):
                end = min(end, self.chunks[i + 1][0]["start"])
            self.windows.append((c[0]["start"], max(end, c[-1]["end"])))

    def at(self, t):
        for i, (a, b) in enumerate(self.windows):
            if a <= t < b:
                c = self.chunks[i]
                active = 0
                for j, w in enumerate(c):
                    if t >= w["start"]:
                        active = j
                key = (i, active)
                if key not in self.cache:
                    text = " ".join(w["word"].replace("*", "") for w in c)
                    self.cache[key] = render_text(text, highlight_index=active, **self.kw)
                img = self.cache[key]
                x = (self.W - img.width) / 2
                y = self.y * self.H - img.height / 2
                return img, x, y
        return None
