"""Burn word-by-word (karaoke style) subtitles onto a video and mix in a voice-over.

Pure OpenCV + PIL + ffmpeg (no moviepy), so it does not depend on how the source file was muxed.
The word being spoken is highlighted; words already spoken are white; coming words are dimmed.
"""
import os
import subprocess

import cv2
import numpy as np

from common.video_io import ffmpeg_exe, to_h264


def _font_path(rtl):
    cands = ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


class WordSubtitles:
    def __init__(self, words, width, height, rtl=False, max_lines=2):
        from PIL import ImageFont, Image, ImageDraw
        self.words, self.W, self.H, self.rtl = words, width, height, rtl
        self.size = max(22, int(height / 22))
        fp = _font_path(rtl)
        self.font = ImageFont.truetype(fp, self.size) if fp else ImageFont.load_default()
        self._draw = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        if rtl:
            import arabic_reshaper
            from bidi.algorithm import get_display
            self._shape = lambda w: get_display(arabic_reshaper.reshape(w))
        else:
            self._shape = lambda w: w
        self.disp = [self._shape(w[2]) for w in words]
        self.wid = [self._draw.textlength(d, font=self.font) for d in self.disp]
        self.space = self._draw.textlength(" ", font=self.font)
        self.pages = self._paginate(max_lines)
        self.starts = np.array([w[0] for w in words])
        self._cache = {}

    def _paginate(self, max_lines):
        """Group words into pages of at most `max_lines` lines that fit the video width."""
        limit = self.W * 0.92
        pages, lines, cur, cw = [], [], [], 0.0
        for i, w in enumerate(self.wid):
            add = w + (self.space if cur else 0)
            if cur and cw + add > limit:
                lines.append(cur); cur, cw = [], 0.0
                add = w
                if len(lines) == max_lines:
                    pages.append(lines); lines = []
            cur.append(i); cw += add
        if cur:
            lines.append(cur)
        if lines:
            pages.append(lines)
        return pages

    def _page_of(self, wi):
        for pi, lines in enumerate(self.pages):
            if wi <= lines[-1][-1]:
                return pi
        return len(self.pages) - 1

    def _render(self, pi, cur):
        from PIL import Image, ImageDraw
        lines = self.pages[pi]
        lh = int(self.size * 1.45)
        h = lh * len(lines) + int(self.size * 0.6)
        img = Image.new("RGBA", (self.W, h), (0, 0, 0, 175))
        d = ImageDraw.Draw(img)
        for li, line in enumerate(lines):
            total = sum(self.wid[i] for i in line) + self.space * (len(line) - 1)
            x = (self.W + total) / 2 if self.rtl else (self.W - total) / 2
            y = int(self.size * 0.3) + li * lh
            for i in line:
                w = self.wid[i]
                x0 = x - w if self.rtl else x
                if i == cur:
                    d.rounded_rectangle([x0 - 6, y - 2, x0 + w + 6, y + self.size + 8], 8, fill=(255, 196, 0, 255))
                    col = (20, 20, 20, 255)
                elif i < cur:
                    col = (255, 255, 255, 255)
                else:
                    col = (190, 190, 190, 255)
                d.text((x0, y), self.disp[i], font=self.font, fill=col)
                x = x0 - self.space if self.rtl else x + w + self.space
        return np.array(img)

    def bar_at(self, t):
        if t < self.starts[0] - 0.3:
            return None
        cur = max(int(np.searchsorted(self.starts, t, side="right")) - 1, 0)
        if t > self.words[-1][1] + 0.6:
            return None
        key = (self._page_of(cur), cur)
        if key not in self._cache:
            self._cache[key] = self._render(*key)
        return self._cache[key]

    def draw(self, frame, t):
        bar = self.bar_at(t)
        if bar is None:
            return frame
        h = bar.shape[0]
        y0 = self.H - h - int(self.H * 0.04)
        if y0 < 0:
            return frame
        a = bar[..., 3:4] / 255.0
        roi = frame[y0:y0 + h, :bar.shape[1]]
        frame[y0:y0 + h, :bar.shape[1]] = (roi * (1 - a) + bar[..., [2, 1, 0]] * a).astype(np.uint8)
        return frame


def burn(video_path, words, out_path, audio_path=None, audio_duration=0.0, rtl=False):
    """Write `out_path` (H.264 mp4) = video + word-by-word subtitles (+ voice-over if given).

    If the voice-over is longer than the video, the last frame is held so the speech is not cut.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_dur = n / fps if n > 0 else 0
    total = max(vid_dur, audio_duration + 0.4)
    subs = WordSubtitles(words, W, H, rtl=rtl)

    raw = str(out_path) + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    last, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            if last is None or i / fps >= total:
                break
            frame = last.copy()                      # hold the last frame while the voice finishes
        else:
            last = frame.copy()
        vw.write(subs.draw(frame, i / fps))
        i += 1
    cap.release(); vw.release()

    if audio_path:
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", raw, "-i", str(audio_path),
               "-map", "0:v:0", "-map", "1:a:0", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(out_path)]
        subprocess.run(cmd, check=True)
    else:
        to_h264(raw, out_path)
    os.remove(raw)
    return out_path
