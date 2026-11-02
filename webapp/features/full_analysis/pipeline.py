"""FULL ANALYSIS pipeline: one video with masks + tracked IDs + captions, and one dashboard.

Nothing new is invented here: it re-uses the step files of the other features.
  segmentation/1  YOLOv8-seg model (tracking switched on)     segmentation/2  masks + area log
  tracking/2      track log and trails                        segmentation/3  mask drawing
  tracking/3      analytics                                   tracking/4      ID labels + trails
  segmentation/4  class statistics                            captioning      CLIP -> BLIP keyframe captions
"""
import time
from pathlib import Path

import cv2

from common.caption_bar import CaptionBar, segments_from_captions
from common.loader import step
from common.video_io import to_h264

FEATURES = Path(__file__).resolve().parent.parent
seg_model, seg_masks = step(FEATURES / "segmentation", "1_segmentation_model.py"), step(FEATURES / "segmentation", "2_masks.py")
seg_render, seg_stats = step(FEATURES / "segmentation", "3_render.py"), step(FEATURES / "segmentation", "4_statistics.py")
trk_tracking, trk_analytics = step(FEATURES / "tracking", "2_tracking.py"), step(FEATURES / "tracking", "3_analytics.py")
trk_render = step(FEATURES / "tracking", "4_render.py")


def run(video_path, out_dir, tracker="bytetrack", size="s", conf=0.3, with_captions=True, progress=None, **_):
    progress = progress or (lambda f, m="": None)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("Could not read the video.")
    duration = n_frames / fps

    bar, keyframe_captions, base = None, None, 0.0
    if with_captions:                                   # CLIP + BLIP from the captioning feature
        progress(0.02, "captioning keyframes (CLIP + BLIP)")
        cap_pipe = step(FEATURES / "captioning", "pipeline.py")
        _, _, caps, _ = cap_pipe.run_vision_stage(video_path, progress=lambda f, m="": progress(0.02 + 0.28 * f, m))
        segs = segments_from_captions(caps, duration)
        bar = CaptionBar(segs, W)
        keyframe_captions = [{"t": round(s[0], 2), "caption": s[2]} for s in segs]
        base = 0.3

    model = seg_model.load_model(size)
    names = model.names
    log, areas = trk_tracking.TrackLog(), seg_masks.AreaLog(n_frames)
    raw = out_dir / "_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    t0, i = time.time(), 0
    for r in seg_model.stream(model, video_path, conf, tracker_yaml=trk_tracking.TRACKERS[tracker]):
        xyxy, c, cls, ids = trk_tracking.frame_detections(r)
        polys = seg_masks.polygons(r)
        log.update(i, xyxy, c, cls, ids)
        areas.update(i, polys, cls, c, names, W, H)
        frame = seg_render.draw_masks(r.orig_img.copy(), polys, cls, ids)
        frame = trk_render.draw_tracks(frame, xyxy, cls, ids, names, log.trails)
        if bar:
            frame = bar.draw(frame, i / fps)
        writer.write(frame)
        i += 1
        if i % 5 == 0:
            progress(base + (0.95 - base) * min(1, i / max(n_frames, 1)), f"frame {i}/{n_frames}")
    writer.release()
    elapsed, done = time.time() - t0, max(i, 1)

    progress(0.96, "encoding video")
    to_h264(raw, out_dir / "annotated.mp4"); raw.unlink(missing_ok=True)

    summary = trk_analytics.summarize(log.records, names, fps, done, W, H)
    trk_analytics.to_mot_txt(log.records, out_dir / "tracks_mot.txt")
    cv2.imwrite(str(out_dir / "heatmap.jpg"), trk_analytics.heatmap_image(log.records, W, H, bg_frame=first))
    progress(1.0, "done")
    res = {"mode": "full", "tracker": tracker, "weights": f"yolov8{size}-seg.pt", "conf": conf,
           "runtime": {"seconds": round(elapsed, 1), "fps_processing": round(done / elapsed, 1)},
           "video_file": "annotated.mp4", "heatmap_file": "heatmap.jpg", "mot_file": "tracks_mot.txt",
           "segmentation": seg_stats.summarize(areas, done, fps), **summary}
    if keyframe_captions:
        res["keyframe_captions"] = keyframe_captions
    return res
