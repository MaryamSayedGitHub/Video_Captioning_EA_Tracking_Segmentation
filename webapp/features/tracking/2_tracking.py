"""STEP 2 - Tracking (ByteTrack / BoT-SORT).

What it does: turns each frame's result into track records and keeps short trails.

ByteTrack : Kalman filter + Hungarian matching on IoU. Second pass uses LOW-confidence boxes,
            so partly hidden objects keep their ID. No appearance network. Fast.
BoT-SORT  : same family, adds camera-motion compensation (and optional ReID).
Trackers are not trained, so there is no tracker loss - only the detector has one.
"""
from collections import defaultdict, deque

import numpy as np

TRACKERS = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}


def frame_detections(result, n_names=None):
    """Result -> (xyxy, conf, cls, ids). ids = -1 when the tracker has not assigned one yet."""
    b = result.boxes
    n = 0 if b is None else len(b)
    if n == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int)
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    ids = b.id.cpu().numpy().astype(int) if b.id is not None else -np.ones(n, int)
    return xyxy, conf, cls, ids


class TrackLog:
    """Collects (frame, id, x1, y1, x2, y2, conf, cls) records and the recent trail per ID."""

    def __init__(self, trail_len=40):
        self.records = []
        self.trails = defaultdict(lambda: deque(maxlen=trail_len))

    def update(self, frame_idx, xyxy, conf, cls, ids):
        alive = set()
        for box, c, k, tid in zip(xyxy, conf, cls, ids):
            if tid < 0:
                continue
            self.records.append((frame_idx, int(tid), *map(float, box), float(c), int(k)))
            self.trails[int(tid)].append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            alive.add(int(tid))
        for tid in [t for t in self.trails if t not in alive]:   # object left: drop its trail
            del self.trails[tid]
