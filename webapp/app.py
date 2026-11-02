"""Web app entry point. Only wires pages to features - all the vision work lives in features/.

Run:  uv run python webapp/app.py   ->  http://127.0.0.1:5000

  features/captioning     CLIP keyframes + BLIP captions   (uses the project's 1_..6_ scripts)
  features/tracking       YOLOv8 detection + ByteTrack / BoT-SORT + analytics
  features/segmentation   YOLOv8-seg masks + statistics
  features/full_analysis  all of the above in one video
"""
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

import jobs

BASE = Path(__file__).resolve().parent
UPLOADS, OUTPUTS = BASE / "uploads", BASE / "outputs"
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

PAGES = {
    "captioning":   {"title": "Captioning",   "blurb": "CLIP picks the keyframes, BLIP describes them. Get English and Arabic captioned videos."},
    "tracking":     {"title": "Tracking",     "blurb": "Detect, follow and count every object. Includes the analytics dashboard."},
    "segmentation": {"title": "Segmentation", "blurb": "Pixel-accurate masks for every object, with per-class area statistics."},
    "analysis":     {"title": "Full analysis", "blurb": "Segmentation, tracking and captions in one video, with one dashboard."},
}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@app.route("/")
def home():
    return render_template("home.html", pages=PAGES)


@app.route("/<page>")
def page(page):
    if page not in PAGES:
        abort(404)
    return render_template(f"{page}.html", pages=PAGES, current=page, meta=PAGES[page])


@app.route("/api/start/<kind>", methods=["POST"])
def start(kind):
    if kind not in PAGES:
        abort(404)
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="Choose a video file first."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in VIDEO_EXT:
        return jsonify(error="Unsupported file type. Use mp4, mov, avi, mkv or webm."), 400
    job_id = uuid.uuid4().hex[:10]
    path = UPLOADS / f"{job_id}{ext}"
    f.save(path)
    form = request.form
    opts = {  # every pipeline ignores the options it does not use
        "tracker": form.get("tracker", "bytetrack"), "size": form.get("size", "s"),
        "conf": float(form.get("conf", 0.3)), "with_captions": "with_captions" in form,
        "languages": [l for l in form.getlist("languages") if l in ("en", "ar")] or ["en", "ar"],
        "fps_sample": float(form.get("fps_sample", 2)),
        "similarity_threshold": float(form.get("similarity_threshold", 0.96)),
    }
    jobs.create(job_id, kind)
    threading.Thread(target=jobs.run, args=(job_id, kind, path, OUTPUTS / job_id, opts), daemon=True).start()
    return jsonify(job_id=job_id)


@app.route("/api/job/<job_id>")
def job(job_id):
    j = jobs.get(job_id)
    if j is None:
        abort(404)
    return jsonify(j)


@app.route("/files/<job_id>/<path:name>")
def files(job_id, name):
    if not job_id.isalnum():
        abort(404)
    return send_from_directory(OUTPUTS / job_id, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
