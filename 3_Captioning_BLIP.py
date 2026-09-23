from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

_blip_model = None
_blip_processor = None


def _load_blip():
    global _blip_model, _blip_processor
    if _blip_model is None:
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_model, _blip_processor


def generate_caption(frame):
    model, processor = _load_blip()
    image = Image.fromarray(frame[..., ::-1])  # BGR -> RGB
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30, num_beams=5, repetition_penalty=1.2)
    return processor.decode(out[0], skip_special_tokens=True)


def caption_keyframes(keyframes):
    """
    Takes a list of (timestamp, frame) and returns a list of (timestamp, caption)
    """
    results = []
    for ts, frame in keyframes:
        caption = generate_caption(frame)
        results.append((ts, caption))
        print(f"[Captioning] {ts:.1f}s -> {caption}")
    return results