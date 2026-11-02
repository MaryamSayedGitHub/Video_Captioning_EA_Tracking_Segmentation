"""STEP 3 - Analytics: pure-numpy analytics over tracking results (no model needed, fully testable).

A "track record" is a tuple: (frame_idx, track_id, x1, y1, x2, y2, conf, cls_id).
"""
from collections import defaultdict

import cv2
import numpy as np


def group_by_track(records):
    tracks = defaultdict(list)
    for fr, tid, x1, y1, x2, y2, conf, cls in records:
        tracks[int(tid)].append((int(fr), x1, y1, x2, y2, conf, int(cls)))
    for t in tracks.values():
        t.sort(key=lambda r: r[0])
    return tracks


def _majority_class(rows):
    classes = [r[6] for r in rows]
    return max(set(classes), key=classes.count)


def summarize(records, names, fps, n_frames, width, height, line_y_ratio=0.5):
    """Everything the dashboard needs, as a JSON-serialisable dict."""
    tracks = group_by_track(records)
    fps = float(fps) if fps else 25.0
    duration = n_frames / fps if n_frames else 0.0

    per_track = []
    class_counts = defaultdict(int)
    trajectories = {}
    line_y = height * line_y_ratio
    crossings = {"down": defaultdict(int), "up": defaultdict(int)}

    for tid, rows in tracks.items():
        cls = _majority_class(rows)
        cname = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
        first, last = rows[0][0], rows[-1][0]
        centers = np.array([[(r[1] + r[3]) / 2, (r[2] + r[4]) / 2] for r in rows])
        path_len = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()) if len(rows) > 1 else 0.0
        # Line crossing: sign change of (cy - line_y) between consecutive detections
        sign = np.sign(centers[:, 1] - line_y)
        for a, b in zip(sign[:-1], sign[1:]):
            if a < 0 < b:
                crossings["down"][cname] += 1
            elif a > 0 > b:
                crossings["up"][cname] += 1
        class_counts[cname] += 1
        per_track.append({
            "id": tid, "class": cname,
            "first_s": round(first / fps, 2), "last_s": round(last / fps, 2),
            "dwell_s": round((last - first + 1) / fps, 2),
            "frames_seen": len(rows),
            "path_px": round(path_len, 1),
            "avg_conf": round(float(np.mean([r[5] for r in rows])), 3),
        })
        step = max(1, len(centers) // 60)  # keep trajectories light
        trajectories[tid] = {"class": cname, "pts": centers[::step].round(1).tolist()}

    per_track.sort(key=lambda t: t["first_s"])

    # Active objects per frame -> per-second timeline (mean of frames in that second)
    active = np.zeros(max(n_frames, 1), dtype=np.int32)
    for rows in tracks.values():
        active[rows[0][0]: rows[-1][0] + 1] += 1
    seconds = int(np.ceil(len(active) / fps))
    timeline = [int(round(active[int(s * fps): int((s + 1) * fps)].mean())) if len(active[int(s * fps): int((s + 1) * fps)]) else 0
                for s in range(seconds)]

    return {
        "video": {"fps": round(fps, 2), "frames": n_frames, "duration_s": round(duration, 2),
                  "width": width, "height": height},
        "totals": {"unique_objects": len(tracks), "by_class": dict(class_counts),
                   "max_simultaneous": int(active.max()) if len(active) else 0},
        "line_crossing": {"line_y_ratio": line_y_ratio,
                          "down": dict(crossings["down"]), "up": dict(crossings["up"])},
        "tracks": per_track,
        "timeline_per_second": timeline,
        "trajectories": trajectories,
        "quality_proxies": quality_proxies(tracks, fps, active),
    }


def quality_proxies(tracks, fps, active):
    """Ground-truth-free hints about tracker health. NOT a replacement for IDF1/MOTA/HOTA.

    - id_churn: unique IDs / max simultaneous objects. ~1 is ideal; much higher means
      the tracker keeps re-issuing IDs (ID switches / fragmentation).
    - short_track_pct: tracks living < 1 second (typically flicker / false positives).
    - avg_track_s: mean life of a track.
    """
    if not tracks:
        return {"id_churn": 0, "short_track_pct": 0, "avg_track_s": 0}
    lifetimes = [(rows[-1][0] - rows[0][0] + 1) / fps for rows in tracks.values()]
    mx = max(int(active.max()), 1)
    return {
        "id_churn": round(len(tracks) / mx, 2),
        "short_track_pct": round(100 * float(np.mean([l < 1.0 for l in lifetimes])), 1),
        "avg_track_s": round(float(np.mean(lifetimes)), 2),
    }


def heatmap_image(records, width, height, bg_frame=None, grid=64):
    """Density of object centres, blended over a background frame. Returns BGR uint8."""
    acc = np.zeros((height, width), dtype=np.float32)
    for fr, tid, x1, y1, x2, y2, conf, cls in records:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= cx < width and 0 <= cy < height:
            acc[cy, cx] += 1
    k = max(15, (min(width, height) // grid) | 1)
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=k, sigmaY=k)
    if acc.max() > 0:
        acc = acc / acc.max()
    color = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if bg_frame is None:
        return color
    bg = cv2.resize(bg_frame, (width, height))
    alpha = (0.25 + 0.65 * acc)[..., None]
    return (bg * (1 - alpha) + color * alpha).astype(np.uint8)


def to_mot_txt(records, path):
    """MOTChallenge format: frame,id,x,y,w,h,conf,-1,-1,-1 (frame is 1-based)."""
    with open(path, "w") as f:
        for fr, tid, x1, y1, x2, y2, conf, cls in sorted(records, key=lambda r: (r[0], r[1])):
            f.write(f"{fr + 1},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1,-1\n")
