"""STEP 1 - Segmentation model (YOLOv8-seg).

What it does: loads the instance-segmentation model and streams the video through it.
If a tracker is given it also tracks (used by the Full analysis feature).

YOLOv8-seg = the YOLOv8 detector + a mask branch (YOLACT idea):
  * proto head  -> 32 prototype masks for the whole image (1/4 resolution)
  * detection head also predicts 32 mask coefficients per object
  * mask of one object = sigmoid(coefficients @ prototypes), cropped to its box
Loss: box (CIoU + DFL) + class (BCE) + mask (binary cross-entropy). Trained on COCO masks.
"""
SIZES = ("n", "s", "m", "l")


def load_model(size="s"):
    from ultralytics import YOLO
    assert size in SIZES, size
    return YOLO(f"yolov8{size}-seg.pt")


def stream(model, video_path, conf=0.3, tracker_yaml=None):
    kw = dict(source=str(video_path), stream=True, conf=conf, verbose=False)
    if tracker_yaml:
        return model.track(tracker=tracker_yaml, persist=True, **kw)
    return model.predict(**kw)
