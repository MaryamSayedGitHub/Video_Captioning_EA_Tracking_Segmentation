"""STEP 2 - From model output to usable masks.

Ultralytics gives each mask as a polygon (result.masks.xy). We measure how much of the
frame each class covers, per frame, so we can plot it over time.
"""
from collections import defaultdict

import cv2
import numpy as np


def polygons(result):
    return list(result.masks.xy) if result.masks is not None else []


class AreaLog:
    """Share of the frame (%) covered by each class, per frame, plus instance and confidence counts."""

    def __init__(self, n_frames):
        self.n = max(n_frames, 1)
        self.area = defaultdict(lambda: np.zeros(self.n, dtype=np.float32))
        self.instances = defaultdict(int)
        self.conf = defaultdict(list)

    def update(self, frame_idx, polys, cls, conf, names, width, height):
        for i, k in enumerate(cls):
            name = names[int(k)]
            self.instances[name] += 1
            self.conf[name].append(float(conf[i]))
            if i < len(polys) and len(polys[i]) >= 3 and frame_idx < self.n:
                self.area[name][frame_idx] += 100.0 * cv2.contourArea(polys[i].astype(np.float32)) / (width * height)
