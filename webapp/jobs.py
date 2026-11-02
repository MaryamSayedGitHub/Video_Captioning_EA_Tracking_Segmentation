"""Background jobs: run a feature pipeline in a thread, report progress, store the result."""
import json
import threading
import traceback
from pathlib import Path

from common.loader import step

FEATURES = Path(__file__).parent / "features"
LOCK, GPU_LOCK = threading.Lock(), threading.Lock()   # GPU_LOCK: one heavy job at a time
JOBS = {}


def get(job_id):
    with LOCK:
        return dict(JOBS[job_id]) if job_id in JOBS else None


def create(job_id, kind):
    with LOCK:
        JOBS[job_id] = {"id": job_id, "kind": kind, "status": "queued", "progress": 0, "message": "queued", "result": None}


def _set(job_id, **kw):
    with LOCK:
        JOBS[job_id].update(kw)


def run(job_id, kind, video_path, out_dir, opts):
    """kind = captioning | tracking | segmentation | analysis -> features/<folder>/pipeline.py"""
    folder = {"captioning": "captioning", "tracking": "tracking",
              "segmentation": "segmentation", "analysis": "full_analysis"}[kind]
    try:
        with GPU_LOCK:
            _set(job_id, status="running")
            pipeline = step(FEATURES / folder, "pipeline.py")
            res = pipeline.run(video_path, out_dir,
                               progress=lambda f, m="": _set(job_id, progress=round(f, 3), message=m), **opts)
            (Path(out_dir) / "result.json").write_text(json.dumps(res))
            _set(job_id, status="done", progress=1.0, message="done", result=res)
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", message=f"{type(e).__name__}: {e}")
