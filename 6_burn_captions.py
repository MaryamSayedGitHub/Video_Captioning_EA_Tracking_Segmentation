import os
import re

try:
    from moviepy import TextClip, CompositeVideoClip
except ImportError:
    from moviepy.editor import TextClip, CompositeVideoClip

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _ARABIC_SUPPORT = True
except ImportError:
    _ARABIC_SUPPORT = False


def prepare_arabic_text(text):
    """
    Arabic letters must be reshaped (joined into their connected forms) and
    reordered (right-to-left) before being drawn as an image, otherwise they
    render disconnected and in the wrong order.
    Requires: pip install arabic-reshaper python-bidi
    """
    if not _ARABIC_SUPPORT:
        raise ImportError(
            "Arabic captions need the 'arabic-reshaper' and 'python-bidi' packages. "
            "Install them with: uv add arabic-reshaper python-bidi"
        )
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def split_into_sentences(text):
    """Splits narration text into sentence-sized chunks for subtitle display."""
    parts = re.split(r"(?<=[.!?\u061F\u06D4])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def find_default_font():
    """
    Tries a few common Windows font paths that support Arabic glyphs.
    You can override this by passing font_path explicitly to add_narrative_captions().
    """
    candidates = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No default font found. Pass font_path explicitly to add_narrative_captions() "
        "-- e.g. the path to a .ttf file that supports Arabic, such as Noto Naskh Arabic."
    )


def add_narrative_captions(video, narrative_text, duration, language="ar", font_path=None):
    """
    Overlays the narration text as timed subtitles at the bottom of the video,
    one sentence at a time, with each sentence's display time proportional to
    its length so the pacing roughly follows the spoken narration.
    """
    sentences = split_into_sentences(narrative_text)
    if not sentences:
        return video

    if font_path is None:
        font_path = find_default_font()

    total_chars = sum(len(s) for s in sentences) or 1
    clips = [video]
    current_time = 0.0

    for sentence in sentences:
        display_text = prepare_arabic_text(sentence) if language == "ar" else sentence

        seg_duration = duration * (len(sentence) / total_chars)
        seg_duration = max(seg_duration, 1.0)
        seg_duration = min(seg_duration, max(duration - current_time, 0.1))

        txt_clip = TextClip(
            font=font_path,
            text=display_text,
            font_size=32,
            color="white",
            bg_color="black",
            method="caption",
            size=(int(video.w * 0.9), None),
        )
        txt_clip = (
            txt_clip
            .with_position(("center", "bottom"))
            .with_start(current_time)
            .with_duration(seg_duration)
        )
        clips.append(txt_clip)
        current_time += seg_duration

    return CompositeVideoClip(clips)