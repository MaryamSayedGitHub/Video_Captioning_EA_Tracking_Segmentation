# Web app

The numbered scripts in the project root (1_Frame_Extraction.py ... 6_burn_captions.py) are untouched.

## Run
    uv add flask ultralytics          # optional: uv add motmetrics
    uv run python webapp/app.py       # http://127.0.0.1:5000

First run downloads YOLOv8 weights automatically. Arabic needs GROQ_API_KEY in .env.

## Layout - one folder per feature, numbered steps, like the captioning project
    webapp/
      app.py                      routes only (pages -> features)
      jobs.py                     runs a feature in a background thread
      common/                     loader.py, video_io.py, caption_bar.py
      features/
        captioning/    pipeline.py  5_evaluation.py   (steps 1-4, 6 are the root scripts)
        tracking/      1_detection  2_tracking  3_analytics  4_render  5_evaluation  pipeline.py
        segmentation/  1_segmentation_model  2_masks  3_render  4_statistics  5_evaluation  pipeline.py
        full_analysis/ pipeline.py  (re-uses the tracking + segmentation steps)
      templates/ static/          the four pages
    docs/                         pipeline diagrams + PDF (make_diagrams.py, make_pdf.py)
