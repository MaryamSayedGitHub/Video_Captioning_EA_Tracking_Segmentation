"""Caption strip drawn on video frames (PIL, so Arabic can be shaped). Used by Full analysis."""
import os

import numpy as np


def _font_path():
    for p in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            return p
    return None


class CaptionBar:
    """Renders a caption strip with PIL (so Arabic can be shaped) and caches it per text."""

    def __init__(self, segments, width, rtl=False):
        self.segments = segments  # [(start_s, end_s, text)]
        self.width = width
        self.rtl = rtl
        self._cache = {}

    def _render(self, text):
        from PIL import Image, ImageDraw, ImageFont
        if self.rtl:
            import arabic_reshaper
            from bidi.algorithm import get_display
            text = get_display(arabic_reshaper.reshape(text))
        size = max(16, self.width // 38)
        fp = _font_path()
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
        img = Image.new("RGBA", (self.width, size * 2 + 12), (0, 0, 0, 160))
        d = ImageDraw.Draw(img)
        # naive wrap into at most 2 lines
        words, lines, cur = text.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if d.textlength(t, font=font) > self.width - 30 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
        for i, line in enumerate(lines[:2]):
            tw = d.textlength(line, font=font)
            d.text(((self.width - tw) / 2, 6 + i * (size + 2)), line, font=font, fill=(255, 255, 255, 255))
        return np.array(img)

    def draw(self, frame, t):
        text = next((s[2] for s in self.segments if s[0] <= t < s[1]), None)
        if not text:
            return frame
        if text not in self._cache:
            self._cache[text] = self._render(text)
        bar = self._cache[text]
        h = bar.shape[0]
        y0 = frame.shape[0] - h
        if y0 < 0:
            return frame
        a = bar[..., 3:4] / 255.0
        roi = frame[y0:, :bar.shape[1]]
        frame[y0:, :bar.shape[1]] = (roi * (1 - a) + bar[..., [2, 1, 0]] * a).astype(np.uint8)
        return frame


def segments_from_captions(captions_with_time, duration):
    """[(ts, text)] keyframe captions -> [(start, end, text)] each lasting until the next keyframe."""
    out = []
    for i, (ts, text) in enumerate(captions_with_time):
        end = captions_with_time[i + 1][0] if i + 1 < len(captions_with_time) else duration + 1
        out.append((ts, end, text))
    return out
