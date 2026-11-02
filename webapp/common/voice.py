"""Text-to-speech with WORD timings (so subtitles can follow the voice word by word).

Preferred engine: edge-tts  -> neural voices + a timestamp for every spoken word (needs internet).
Fallback engine:  gTTS      -> no timestamps, so word times are ESTIMATED from word length and
                               the real audio duration (close, but not exact).
"""
import asyncio
import re
import subprocess

from common.video_io import ffmpeg_exe

VOICES = {"en": "en-US-AriaNeural", "ar": "ar-EG-SalmaNeural"}   # ar-EG = Egyptian Arabic
_PAUSE = re.compile(r"[.!?,;:\u061F\u060C\u061B]$")


def audio_duration(path):
    """Duration in seconds, read from `ffmpeg -i` output."""
    exe = ffmpeg_exe()
    err = subprocess.run([exe, "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if not m:
        raise RuntimeError(f"cannot read audio duration of {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def estimate_word_times(words, total, lead=0.15):
    """Spread `total` seconds over the words by length; punctuation adds a small pause."""
    weights = [len(w) + (4 if _PAUSE.search(w) else 0) + 1 for w in words]
    span = max(total - lead, 0.1)
    unit = span / (sum(weights) or 1)
    out, t = [], lead
    for w, wt in zip(words, weights):
        out.append((t, t + wt * unit, w))
        t += wt * unit
    return out


def _align(display_words, boundaries, total):
    """Map the narrative's own words (with punctuation) onto the engine's word timestamps."""
    n, m = len(display_words), len(boundaries)
    if m == 0:
        return estimate_word_times(display_words, total)
    starts = [boundaries[min(m - 1, round(i * m / n))][0] for i in range(n)]
    out = []
    for i, w in enumerate(display_words):
        end = starts[i + 1] if i + 1 < n else boundaries[-1][1]
        out.append((starts[i], max(end, starts[i] + 0.05), w))
    return out


async def _edge_stream(text, voice, out_path):
    import edge_tts
    try:
        com = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    except TypeError:                       # older edge-tts: word boundaries are the default
        com = edge_tts.Communicate(text, voice)
    bounds = []
    with open(out_path, "wb") as f:
        async for ch in com.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                bounds.append((ch["offset"] / 1e7, (ch["offset"] + ch["duration"]) / 1e7, ch["text"]))
    return bounds


def synthesize(text, lang, out_path):
    """Returns dict(audio, duration, words=[(start,end,word)], engine, exact, note)."""
    words = text.split()
    note = ""
    try:
        bounds = asyncio.run(_edge_stream(text, VOICES.get(lang, VOICES["en"]), out_path))
        dur = audio_duration(out_path)
        if bounds:
            return {"audio": str(out_path), "duration": dur, "words": _align(words, bounds, dur),
                    "engine": "edge-tts", "exact": True, "note": ""}
        note = "edge-tts returned no word timestamps; word timing is estimated."
        return {"audio": str(out_path), "duration": dur, "words": estimate_word_times(words, dur),
                "engine": "edge-tts", "exact": False, "note": note}
    except Exception as e:
        note = f"edge-tts unavailable ({type(e).__name__}); used gTTS with estimated word timing."
    from gtts import gTTS
    gTTS(text=text, lang=lang).save(str(out_path))
    dur = audio_duration(out_path)
    return {"audio": str(out_path), "duration": dur, "words": estimate_word_times(words, dur),
            "engine": "gTTS", "exact": False, "note": note}
