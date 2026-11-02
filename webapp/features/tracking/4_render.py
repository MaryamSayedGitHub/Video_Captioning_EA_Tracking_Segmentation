"""STEP 4 - Drawing the tracking result on the video (boxes, ID labels, trails, heatmap)."""
import cv2
import numpy as np

# 12 clearly different BGR colours; colour = track ID % 12, so an object keeps its colour
PALETTE = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241), (182, 89, 155),
           (156, 188, 26), (34, 126, 230), (173, 68, 142), (94, 73, 52), (96, 174, 39),
           (43, 57, 192), (133, 160, 22)]


def color_for(i):
    return PALETTE[int(i) % len(PALETTE)]


def draw_tracks(frame, xyxy, cls, ids, names, trails):
    for box, k, tid in zip(xyxy, cls, ids):
        c = color_for(tid if tid >= 0 else k)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, max(2, frame.shape[0] // 400))
        label = names[k] + (f" #{tid}" if tid >= 0 else "")
        fs = max(0.6, frame.shape[0] / 1080 * 1.0)          # label size grows with the video resolution
        th_ = max(1, int(round(fs * 2)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), c, -1)
        cv2.putText(frame, label, (x1 + 5, max(th + 4, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th_, cv2.LINE_AA)
    for tid, pts in trails.items():
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, np.int32)], False, color_for(tid), 2, cv2.LINE_AA)
    return frame
