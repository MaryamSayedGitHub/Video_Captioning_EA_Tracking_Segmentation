"""TRACKING pipeline: video -> detect -> track -> analytics -> annotated video + dashboard data.

Steps (read them in order):
  1_detection.py   2_tracking.py   3_analytics.py   4_render.py   5_evaluation.py
"""
import time
from pathlib import Path

import cv2

from common.loader import step
from common.video_io import to_h264

HERE = Path(__file__).parent
detection, tracking = step(HERE, "1_detection.py"), step(HERE, "2_tracking.py")
analytics, render = step(HERE, "3_analytics.py"), step(HERE, "4_render.py")


def run(video_path, out_dir, tracker="bytetrack", size="s", conf=0.3, progress=None, **_):
    progress = progress or (lambda f, m="": None)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("Could not read the video.")

    model = detection.load_detector(size)
    names = model.names
    log = tracking.TrackLog()
    raw = out_dir / "_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    t0, i = time.time(), 0
    for result in detection.stream(model, video_path, tracking.TRACKERS[tracker], conf):
        xyxy, c, cls, ids = tracking.frame_detections(result)
        log.update(i, xyxy, c, cls, ids)
        writer.write(render.draw_tracks(result.orig_img.copy(), xyxy, cls, ids, names, log.trails))
        i += 1
        if i % 5 == 0:
            progress(min(0.95, i / max(n_frames, 1)), f"frame {i}/{n_frames}")
    writer.release()
    elapsed, done = time.time() - t0, max(i, 1)

    progress(0.96, "encoding video")
    to_h264(raw, out_dir / "annotated.mp4"); raw.unlink(missing_ok=True)

    summary = analytics.summarize(log.records, names, fps, done, W, H)
    analytics.to_mot_txt(log.records, out_dir / "tracks_mot.txt")
    cv2.imwrite(str(out_dir / "heatmap.jpg"), analytics.heatmap_image(log.records, W, H, bg_frame=first))
    progress(1.0, "done")
    return {"mode": "track", "tracker": tracker, "weights": f"yolov8{size}.pt", "conf": conf,
            "runtime": {"seconds": round(elapsed, 1), "fps_processing": round(done / elapsed, 1)},
            "video_file": "annotated.mp4", "heatmap_file": "heatmap.jpg", "mot_file": "tracks_mot.txt", **summary}
