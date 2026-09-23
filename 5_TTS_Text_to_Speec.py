from gtts import gTTS


def text_to_speech(text, output_path="voiceover.mp3", lang="ar"):
    if not text or not text.strip():
        raise ValueError("Empty text passed to TTS")

    tts = gTTS(text=text, lang=lang)
    tts.save(output_path)
    print(f"[TTS] Saved voiceover to {output_path}")
    return output_path