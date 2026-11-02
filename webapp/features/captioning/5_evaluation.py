"""STEP 5 - Evaluation of captioning (reference-free, so no human captions are needed).

compression_ratio : frames sampled / keyframes kept        (what CLIP removed)
CLIPScore         : 2.5 * max(cos(image, caption), 0)       (image-text agreement)
redundancy        : word overlap of neighbouring captions   (lower = more diverse)
With human reference captions you could also report CIDEr, BLEU-4, METEOR, ROUGE-L, SPICE.
"""
import numpy as np

from common.loader import load_root


def _jaccard_words(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def caption_stats(n_frames_sampled, n_keyframes, captions):
    """Proxy metrics for the keyframe + caption stage.

    compression_ratio : frames_sampled / keyframes   (how much CLIP removed)
    redundancy        : mean word-Jaccard of consecutive captions (lower = more diverse)
    unique_caption_pct: share of distinct captions
    """
    texts = [c for _, c in captions]
    red = float(np.mean([_jaccard_words(a, b) for a, b in zip(texts[:-1], texts[1:])])) if len(texts) > 1 else 0.0
    return {
        "frames_sampled": n_frames_sampled,
        "keyframes": n_keyframes,
        "compression_ratio": round(n_frames_sampled / max(n_keyframes, 1), 2),
        "redundancy": round(red, 3),
        "unique_caption_pct": round(100 * len(set(texts)) / max(len(texts), 1), 1),
    }


def clip_scores(keyframes, captions):
    """Reference-free CLIPScore = 2.5 * max(cos(image, text), 0) per keyframe, using the
    project's own CLIP model. Returns (mean, per_frame_list)."""
    import torch
    clip_mod = load_root("2_Keyframe_Selection_CLIP.py")
    model, processor = clip_mod._load_clip()
    scores = []
    for (ts, frame), (_, text) in zip(keyframes, captions):
        img = clip_mod.get_clip_embedding(frame)
        tin = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            t = model.get_text_features(**tin)
            t = t.pooler_output if hasattr(t, "pooler_output") else t
        t = t / t.norm(dim=-1, keepdim=True)
        scores.append(2.5 * max(float((img @ t.T).item()), 0.0))
    return (float(np.mean(scores)) if scores else 0.0), [round(s, 3) for s in scores]
