# Video Caption 🎬

An automated pipeline that takes a raw video, extracts and selects the most meaningful keyframes, generates AI captions for them, turns those captions into a narrated script, converts the script to speech, burns captions onto the video, and merges everything into a final voiced-over video — in multiple languages (Arabic & English).

## ✨ Features

- **Frame Extraction** — pulls frames from the input video.
- **Keyframe Selection (CLIP)** — filters extracted frames down to the most representative/diverse keyframes using CLIP embeddings.
- **Image Captioning (BLIP)** — generates a natural-language caption for each keyframe using the BLIP model.
- **Caption Collection** — aggregates all keyframe captions into a coherent narrative/script.
- **Text-to-Speech** — converts the generated narrative into voice-over audio (Arabic & English).
- **Caption Burning** — overlays subtitles/captions onto the video.
- **Final Merge** — combines the original video, burned captions, and generated voice-over into a final multi-language output video.

## 📁 Project Structure

```
Video_Caption/
├── 1_Frame_Extraction.py          # Extract frames from the input video
├── 2_Keyframe_Selection_CLIP.py   # Select best keyframes using CLIP
├── 3_Captioning_BLIP.py           # Generate captions per keyframe using BLIP
├── 4_Collect_Captions.py          # Combine captions into a full narrative
├── 5_TTS_Text_to_Speec.py         # Text-to-speech generation
├── 6_burn_captions.py             # Burn captions onto the video
├── Final_Video_with_voice_over.py # Merge audio + video + captions (main pipeline)
├── check_groq.py                  # Utility script to test Groq API connectivity
├── Experiment_1/ , Experiment_2/  # Experiment outputs/notebooks
├── src/                           # Shared source/helper modules
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration (uv)
├── uv.lock                        # Locked dependency versions
└── .env                           # API keys / environment variables (not committed)
```

## ⚙️ Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- A Groq API key (or equivalent) for LLM-based caption/narrative generation
- FFmpeg installed and available on PATH (required by MoviePy)

## 🚀 Installation

```bash
git clone https://github.com/USERNAME/Video_Caption.git
cd Video_Caption
uv sync
```

Create a `.env` file in the project root with your API key(s):

```
GROQ_API_KEY=your_api_key_here
```

## ▶️ Usage

Run the full pipeline on an input video:

```bash
uv run Final_Video_with_voice_over.py Video_Sample_Input.mp4 output_final.mp4
```

This will generate one output video per configured language, e.g.:

```
✅ Done! Final videos:
  [ar] output_final_ar.mp4
  [en] output_final_en.mp4
```

You can also run each pipeline stage individually for debugging/experimentation:

```bash
uv run 1_Frame_Extraction.py
uv run 2_Keyframe_Selection_CLIP.py
uv run 3_Captioning_BLIP.py
uv run 4_Collect_Captions.py
uv run 5_TTS_Text_to_Speec.py
uv run 6_burn_captions.py
```

## 📤 Output

- `output_final_<lang>.mp4` — final captioned & narrated video per language
- `voiceover_<lang>.mp3` — generated voice-over audio per language

## 📝 Notes

- Large media files (`.mp4`, `.mp3`) and the `.env` file are excluded from version control — see `.gitignore`.
- Add a `LICENSE` file if you plan to open-source this project.

## 📄 License

This project is currently unlicensed. Add a license of your choice (MIT, Apache-2.0, etc.) if needed.