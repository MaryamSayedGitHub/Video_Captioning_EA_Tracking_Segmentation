"""SEGMENTATION pipeline: video -> YOLOv8-seg -> masks -> statistics -> annotated video.

Steps: 1_segmentation_model.py  2_masks.py  3_render.py  4_statistics.py  5_evaluation.py
"""
import time
from pathlib import Path

import cv2

from common.loader import step
from common.video_io import to_h264

HERE = Path(__file__).parent
model_step, masks = step(HERE, "1_segmentation_model.py"), step(HERE, "2_masks.py")
render, stats = step(HERE, "3_render.py"), step(HERE, "4_statistics.py")


def run(video_path, out_dir, size="s", conf=0.3, progress=None, **_):
    progress = progress or (lambda f, m="": None)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = model_step.load_model(size)
    names = model.names
    log = masks.AreaLog(n_frames)
    raw = out_dir / "_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    t0, i = time.time(), 0
    for r in model_step.stream(model, video_path, conf):
        polys = masks.polygons(r)
        b = r.boxes
        n = 0 if b is None else len(b)
        xyxy = b.xyxy.cpu().numpy() if n else []
        cls = b.cls.cpu().numpy().astype(int) if n else []
        confs = b.conf.cpu().numpy() if n else []
        log.update(i, polys, cls, confs, names, W, H)
        frame = render.draw_masks(r.orig_img.copy(), polys, cls)
        writer.write(render.draw_labels(frame, xyxy, cls, names))
        i += 1
        if i % 5 == 0:
            progress(min(0.95, i / max(n_frames, 1)), f"frame {i}/{n_frames}")
    writer.release()
    elapsed, done = time.time() - t0, max(i, 1)

    progress(0.96, "encoding video")
    to_h264(raw, out_dir / "annotated.mp4"); raw.unlink(missing_ok=True)
    progress(1.0, "done")
    return {"mode": "segment", "weights": f"yolov8{size}-seg.pt", "conf": conf,
            "runtime": {"seconds": round(elapsed, 1), "fps_processing": round(done / elapsed, 1)},
            "video_file": "annotated.mp4",
            "video": {"fps": round(fps, 2), "frames": done, "duration_s": round(done / fps, 2), "width": W, "height": H},
            "segmentation": stats.summarize(log, done, fps)}
