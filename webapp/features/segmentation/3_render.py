"""STEP 3 - Drawing masks: transparent fill + outline, colour per object."""
import cv2
import numpy as np

PALETTE = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241), (182, 89, 155),
           (156, 188, 26), (34, 126, 230), (173, 68, 142), (94, 73, 52), (96, 174, 39),
           (43, 57, 192), (133, 160, 22)]


def color_for(i):
    return PALETTE[int(i) % len(PALETTE)]


def draw_masks(frame, polys, cls, ids=None, alpha=0.45):
    """colour = track ID if given (stable per object), otherwise class id."""
    if not polys:
        return frame
    keys = [(ids[i] if ids is not None and ids[i] >= 0 else cls[i]) for i in range(len(polys))]
    overlay = frame.copy()
    for poly, k in zip(polys, keys):
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [np.round(poly).astype(np.int32)], color_for(k))
    out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    for poly, k in zip(polys, keys):
        if len(poly) >= 3:
            cv2.polylines(out, [np.round(poly).astype(np.int32)], True, color_for(k), 2)
    return out


def draw_labels(frame, xyxy, cls, names):
    for box, k in zip(xyxy, cls):
        x1, y1 = int(box[0]), int(box[1])
        label = names[int(k)]
        fs = max(0.6, frame.shape[0] / 1080 * 1.0)          # label size grows with the video resolution
        th_ = max(1, int(round(fs * 2)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color_for(k), -1)
        cv2.putText(frame, label, (x1 + 5, max(th + 4, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th_, cv2.LINE_AA)
    return frame
