"""CAPTIONING pipeline: wraps the project's existing pipeline (steps 1-4 + 6).

Re-uses her modules unchanged (loaded by core.loader):
  1_Frame_Extraction  -> 2_Keyframe_Selection_CLIP -> 3_Captioning_BLIP
  -> 4_Collect_Captions (Groq) -> 6_burn_captions
The TTS / voice-over steps (5, Final_Video_with_voice_over) are intentionally not used here.
"""
import os
import time
from pathlib import Path

from common.loader import load_root as load, step
from common.video_io import to_h264

HERE = Path(__file__).parent
evaluation = step(HERE, "5_evaluation.py")


def run_vision_stage(video_path, fps_sample=2, similarity_threshold=0.96, progress=None):
    """Language-independent part: frames -> CLIP keyframes -> BLIP captions."""
    progress = progress or (lambda f, m="": None)
    fe, ks, cp = load("1_Frame_Extraction.py"), load("2_Keyframe_Selection_CLIP.py"), load("3_Captioning_BLIP.py")
    timings = {}

    t = time.time(); progress(0.05, "extracting frames")
    frames = fe.extract_frames(video_path, fps_sample=fps_sample)
    timings["frame_extraction_s"] = round(time.time() - t, 2)

    t = time.time(); progress(0.15, "CLIP keyframe selection")
    keyframes = ks.select_keyframes(frames, similarity_threshold=similarity_threshold)
    timings["clip_selection_s"] = round(time.time() - t, 2)

    t = time.time(); progress(0.45, "BLIP captioning")
    captions = cp.caption_keyframes(keyframes)
    timings["blip_captioning_s"] = round(time.time() - t, 2)
    return frames, keyframes, captions, timings


def _burn(video_path, narrative, language, out_path):
    """Burn the narrative onto the video with her 6_burn_captions, keep original audio."""
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip
    bc = load("6_burn_captions.py")
    clip = VideoFileClip(str(video_path))
    captioned = bc.add_narrative_captions(clip, narrative, duration=clip.duration, language=language)
    raw = str(out_path) + ".raw.mp4"
    captioned.write_videofile(raw, codec="libx264", audio_codec="aac", logger=None)
    clip.close()
    to_h264(raw, out_path)
    os.remove(raw)


def run(video_path, out_dir, languages=("en", "ar"), fps_sample=2, similarity_threshold=0.96,
        with_clip_score=True, progress=None, **_):
    progress = progress or (lambda f, m="": None)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    frames, keyframes, captions, timings = run_vision_stage(
        video_path, fps_sample, similarity_threshold, progress)

    cc = load("4_Collect_Captions.py")
    result = {"keyframe_captions": [{"t": round(ts, 2), "caption": c} for ts, c in captions],
              "videos": {}, "narratives": {}, "notes": []}

    for i, lang in enumerate(languages):
        progress(0.6 + 0.35 * i / max(len(languages), 1), f"narrative + captions ({lang})")
        try:
            narrative = cc.build_narrative(captions, language=lang)
        except Exception as e:  # missing GROQ key / network
            if lang == "en":
                narrative = " ".join(c for _, c in cc.denoise_captions(captions))
                result["notes"].append(f"Groq unavailable ({e}); English uses the joined BLIP captions.")
            else:
                result["notes"].append(f"Skipped '{lang}': needs the Groq LLM ({e}).")
                continue
        result["narratives"][lang] = narrative
        name = f"captioned_{lang}.mp4"
        _burn(video_path, narrative, lang, out_dir / name)
        result["videos"][lang] = name

    stats = evaluation.caption_stats(len(frames), len(keyframes), captions)
    if with_clip_score and keyframes:
        mean, per = evaluation.clip_scores(keyframes, captions)
        stats["clip_score_mean"] = round(mean, 3)
        stats["clip_score_per_keyframe"] = per
    stats["timings"] = timings
    stats["settings"] = {"fps_sample": fps_sample, "similarity_threshold": similarity_threshold}
    result["metrics"] = stats
    progress(1.0, "done")
    return result
