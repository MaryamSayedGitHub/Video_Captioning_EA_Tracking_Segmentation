import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


def get_clip_embedding(frame):
    model, processor = _load_clip()
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR -> RGB, fresh contiguous PIL image
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        output = model.get_image_features(**inputs)

    # Some transformers versions return an object you need to extract the tensor from, instead of returning it directly
    if hasattr(output, "image_embeds"):
        emb = output.image_embeds
    elif hasattr(output, "pooler_output"):
        emb = output.pooler_output
    elif torch.is_tensor(output):
        emb = output
    else:
        raise TypeError(f"Unexpected output type from get_image_features: {type(output)}")

    return emb / emb.norm(dim=-1, keepdim=True)


def select_keyframes(frames, similarity_threshold=0.95):
    """
    Removes near-duplicate frames and keeps only frames where the scene actually changes.
    Higher similarity_threshold = keeps more frames (more accurate but slower)
    Lower similarity_threshold = keeps fewer frames (faster but may miss detail)
    """
    if not frames:
        return []

    keyframes = [frames[0]]
    last_emb = get_clip_embedding(frames[0][1])

    for ts, frame in frames[1:]:
        emb = get_clip_embedding(frame)
        sim = (emb @ last_emb.T).item()
        if sim < similarity_threshold:
            keyframes.append((ts, frame))
            last_emb = emb

    print(f"[Keyframe Selection] Reduced {len(frames)} -> {len(keyframes)} keyframes")
    return keyframes