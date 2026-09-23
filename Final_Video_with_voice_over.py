"""
Final_Video_with_voice_over.py
--------------------------------
This file runs all the project stages in order:
1) Frame extraction
2) Keyframe selection (CLIP)
3) Caption generation (BLIP)
4) Merging captions into a connected narrative (Groq LLM)
5) Converting the narrative to speech (TTS)
6) Merging the audio with the video as a voiceover

Usage:
    python Final_Video_with_voice_over.py <video_path> [output_path]
"""

import os
import sys
import importlib.util

try:
    # moviepy >= 2.0
    from moviepy import VideoFileClip, AudioFileClip
except ImportError:
    # moviepy < 2.0 (legacy import path)
    from moviepy.editor import VideoFileClip, AudioFileClip


# ---------------------------------------------------------------------------
# Load files whose names start with a digit (can't be imported normally in Python)
# ---------------------------------------------------------------------------
def load_module(file_name, module_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File {file_name} was not found in the same folder as {os.path.basename(__file__)}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frame_extraction = load_module("1_Frame_Extraction.py", "frame_extraction")
keyframe_selection = load_module("2_Keyframe_Selection_CLIP.py", "keyframe_selection")
captioning = load_module("3_Captioning_BLIP.py", "captioning")
collect_captions = load_module("4_Collect_Captions.py", "collect_captions")
tts_module = load_module("5_TTS_Text_to_Speec.py", "tts_module")
burn_captions = load_module("6_Burn_Captions.py", "burn_captions")


# ---------------------------------------------------------------------------
# Merge the audio with the video
# ---------------------------------------------------------------------------
def merge_audio_video(video_path, audio_path, output_path, narrative_text=None, language="ar"):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # If the narration turns out longer than the video, trim it to the video's length
    # If it's shorter, the video will stay silent for the remainder (adjust if you want to loop the audio)
    if audio.duration > video.duration:
        if hasattr(audio, "subclipped"):
            # moviepy >= 2.0
            audio = audio.subclipped(0, video.duration)
        else:
            # moviepy < 2.0 (legacy method name)
            audio = audio.subclip(0, video.duration)

    caption_duration = min(audio.duration, video.duration)

    if narrative_text:
        video = burn_captions.add_narrative_captions(
            video, narrative_text, duration=caption_duration, language=language
        )

    if hasattr(video, "with_audio"):
        # moviepy >= 2.0
        final_video = video.with_audio(audio)
    else:
        # moviepy < 2.0 (legacy method name)
        final_video = video.set_audio(audio)
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )

    video.close()
    audio.close()

    print(f"[Merge] Final video saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# The full pipeline — generates one output video per language
# ---------------------------------------------------------------------------
def run_pipeline(
    video_path,
    output_path="output_with_voiceover.mp4",
    fps_sample=2,
    similarity_threshold=0.96,
    languages=("ar", "en"),
):
    print("=== 1) Frame Extraction ===")
    frames = frame_extraction.extract_frames(video_path, fps_sample=fps_sample)

    print("\n=== 2) Keyframe Selection (CLIP) ===")
    keyframes = keyframe_selection.select_keyframes(
        frames, similarity_threshold=similarity_threshold
    )

    print("\n=== 3) Captioning (BLIP) ===")
    captions_with_time = captioning.caption_keyframes(keyframes)

    # Everything above this line (frame extraction, keyframe selection, captioning)
    # is language-independent, so it only needs to run once.
    # Everything below (narrative, TTS, merge) runs once per requested language.

    base, ext = os.path.splitext(output_path)
    final_paths = {}

    for language in languages:
        print(f"\n=== 4) Collect Captions (Groq LLM) — language: {language} ===")
        narrative = collect_captions.build_narrative(captions_with_time, language=language)
        print(f"\n--- Narrative ({language}) ---\n{narrative}\n")

        print(f"=== 5) Text-to-Speech — language: {language} ===")
        audio_path = tts_module.text_to_speech(
            narrative, output_path=f"voiceover_{language}.mp3", lang=language
        )

        print(f"\n=== 6) Merge Audio + Video (+ burned-in captions) — language: {language} ===")
        lang_output_path = f"{base}_{language}{ext}"
        final_path = merge_audio_video(
            video_path, audio_path, lang_output_path,
            narrative_text=narrative, language=language,
        )
        final_paths[language] = final_path

    print("\n✅ Done! Final videos:")
    for language, path in final_paths.items():
        print(f"  [{language}] {path}")

    return final_paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Final_Video_with_voice_over.py <video_path> [output_path]")
        print("Generates one output video per language (Arabic and English) by default.")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "output_with_voiceover.mp4"

    run_pipeline(input_video, output_video, languages=("ar", "en"))