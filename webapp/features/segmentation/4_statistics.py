"""STEP 4 - Dashboard statistics for segmentation (describes the output; it is NOT accuracy)."""
import numpy as np


def summarize(log, n_done, fps):
    classes = {}
    for name, arr in log.area.items():
        used = arr[:n_done]
        classes[name] = {
            "instances": int(log.instances[name]),
            "mean_area_pct": round(float(used.mean()), 2),
            "max_area_pct": round(float(used.max()), 2),
            "avg_conf": round(float(np.mean(log.conf[name])), 3) if log.conf[name] else 0,
        }
    step = max(1, n_done // 120)          # keep the timeline light (<= ~120 points)
    return {"classes": classes,
            "area_timeline": {n: [round(float(v), 2) for v in a[:n_done:step]] for n, a in log.area.items()},
            "timeline_step_s": round(step / fps, 2)}
