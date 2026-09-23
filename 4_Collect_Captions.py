import os
import re
from collections import Counter
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # if python-dotenv isn't installed, fall back to the regular environment

_groq_client = None

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "on", "in", "at", "of",
    "and", "with", "to", "there", "this", "that",
}


def _get_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or "your-groq-api-key" in api_key:
            raise ValueError(
                "GROQ_API_KEY is missing or still the placeholder. "
                "Open your .env file and put your real key from console.groq.com/keys"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _content_words(text):
    """Extracts the meaningful words from a caption, ignoring articles/prepositions."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def denoise_captions(captions_with_time, overlap_threshold=0.4):
    """
    BLIP occasionally misreads a frame and produces a caption that contradicts
    the rest of the scene (e.g. "praying" or "a woman" appearing once among
    many captions describing "a man and a girl sitting on a bench"). Exact
    string matching alone misses paraphrased hallucinations (same wrong idea,
    different wording), so this compares captions by their meaningful word
    overlap against the majority caption instead. Any caption sharing too
    few content words with the majority is treated as noise and replaced.
    Runs locally, no extra API calls or cost.
    """
    if not captions_with_time:
        return captions_with_time

    texts = [cap for _, cap in captions_with_time]
    counts = Counter(texts)

    # If every caption is unique, we can't establish a majority — leave as is
    if max(counts.values()) < 2:
        return captions_with_time

    majority_text = counts.most_common(1)[0][0]
    majority_words = _content_words(majority_text)

    denoised = []
    for ts, cap in captions_with_time:
        if cap == majority_text:
            denoised.append((ts, cap))
            continue

        cap_words = _content_words(cap)
        union = cap_words | majority_words
        overlap = len(cap_words & majority_words) / len(union) if union else 1.0

        if overlap < overlap_threshold:
            print(
                f'[Denoise] Dropped outlier caption at {ts:.1f}s: "{cap}" '
                f'-> replaced with majority: "{majority_text}" (overlap={overlap:.2f})'
            )
            denoised.append((ts, majority_text))
        else:
            denoised.append((ts, cap))

    return denoised


def build_narrative(captions_with_time, language="ar", model="openai/gpt-oss-20b"):
    """
    Merges the individual captions (each with a timestamp) into one connected
    voiceover narration using the Groq LLM, with a single API call regardless
    of how many captions there are.
    """
    if not captions_with_time:
        return ""

    captions_with_time = denoise_captions(captions_with_time)
    caption_list = "\n".join(f"[{ts:.1f}s] {cap}" for ts, cap in captions_with_time)

    if language == "ar":
        system_prompt = (
            "You are a voiceover scriptwriter. Turn this chronological list of frame "
            "captions into a natural, connected narration suitable for a voiceover, "
            "written in simple Modern Standard Arabic. Be strictly literal and factual: "
            "describe ONLY what is explicitly stated in the captions below. Do not "
            "invent, assume, guess, or add any people, objects, actions, emotions, or "
            "moods that are not explicitly mentioned. Do not mention timestamps or "
            "repeat the same idea more than once. Keep it concise and clear, suitable "
            "for being read aloud, and return only the narration text with no preamble."
        )
    else:
        system_prompt = (
            "You are a voiceover scriptwriter. Turn this chronological list of frame "
            "captions into a natural, connected narration suitable for a voiceover. "
            "Be strictly literal and factual: describe ONLY what is explicitly stated "
            "in the captions below. Do not invent, assume, guess, or add any people, "
            "objects, actions, emotions, or moods that are not explicitly mentioned. "
            "Do not mention timestamps or repeat the same idea more than once. "
            "Keep it concise and return only the narration text."
        )

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": caption_list},
        ],
        temperature=0.1,
    )
    narrative = response.choices[0].message.content.strip()
    print("[Collect Captions] Narrative built via Groq")
    return narrative