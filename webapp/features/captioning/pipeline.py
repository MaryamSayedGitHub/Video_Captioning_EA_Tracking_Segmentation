"""CAPTIONING pipeline: wraps the project's existing pipeline (steps 1-4 + 6).

Re-uses her modules unchanged (loaded by core.loader):
  1_Frame_Extraction  -> 2_Keyframe_Selection_CLIP -> 3_Captioning_BLIP
  -> 4_Collect_Captions (Groq) -> 6_burn_captions
The TTS / voice-over steps (5, Final_Video_with_voice_over) are intentionally not used here.
"""
import time
from pathlib import Path

from common.loader import load_root as load, step
from common import subtitles, voice

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


def run(video_path, out_dir, languages=("en", "ar"), fps_sample=2, similarity_threshold=0.96,
        with_clip_score=True, voiceover=True, progress=None, **_):
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
        rtl = lang == "ar"
        audio_path, dur, info = None, 0.0, None
        if voiceover:
            try:
                info = voice.synthesize(narrative, lang, out_dir / f"voiceover_{lang}.mp3")
                audio_path, dur, words = info["audio"], info["duration"], info["words"]
                if info["note"]:
                    result["notes"].append(f"[{lang}] {info['note']}")
            except Exception as e:                       # no internet / TTS failure: keep the subtitles
                result["notes"].append(f"[{lang}] voice-over failed ({type(e).__name__}: {e}); subtitles only.")
        if info is None:
            vdur = frames[-1][0] if frames else 10.0
            words = voice.estimate_word_times(narrative.split(), max(vdur, 4.0))
        subtitles.burn(video_path, words, out_dir / name, audio_path=audio_path, audio_duration=dur, rtl=rtl)
        result["videos"][lang] = name
        if info:
            result.setdefault("voice", {})[lang] = {"engine": info["engine"], "exact_word_timing": info["exact"]}

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
