"""STEP 1 - Detection (YOLOv8).

What it does: loads a YOLOv8 detector and streams the video through it.
Ultralytics runs detection AND tracking in one call (model.track), so this file starts the
stream and the next file (2_tracking.py) interprets the track IDs.

Model: YOLOv8 = CSPDarknet(C2f) backbone + SPPF, PAN-FPN neck, decoupled anchor-free head.
Loss (training): CIoU + DFL for boxes, BCE for classes. Metric: mAP@50 / mAP@50-95.
"""
SIZES = ("n", "s", "m", "l")        # n = fastest, l = most accurate


def load_detector(size="s"):
    from ultralytics import YOLO       # imported here so the web app starts without it
    assert size in SIZES, size
    return YOLO(f"yolov8{size}.pt")    # weights are downloaded automatically on first use


def stream(model, video_path, tracker_yaml, conf=0.3):
    """Yield one ultralytics Result per frame, with tracker IDs attached.

    persist=True keeps the tracker state between frames (otherwise IDs reset every frame).
    stream=True processes one frame at a time, so memory stays low for long videos.
    """
    return model.track(source=str(video_path), tracker=tracker_yaml, persist=True,
                       stream=True, conf=conf, verbose=False)
