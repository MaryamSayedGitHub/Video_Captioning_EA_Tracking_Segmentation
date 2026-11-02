"""Draws the three pipeline diagrams (captioning, tracking, segmentation) as PNG files."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, PAPER, CARD, LINE = "#12202f", "#5b6b7b", "#f4f6f8", "#ffffff", "#c9d2db"
W, H = 20, 11.5


def canvas(title, subtitle, accent):
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, H - 1.55), W, 1.55, color=INK))
    ax.add_patch(plt.Rectangle((0, H - 1.55), 0.28, 1.55, color=accent))
    ax.text(0.7, H - 0.75, title, color="white", fontsize=30, fontweight="bold", va="center")
    ax.text(0.7, H - 1.28, subtitle, color="#b7c4d1", fontsize=15, va="center")
    return fig, ax


def box(ax, x, y, w, h, head, body="", color="#3498db", fs=12.5, hs=14.5, fill=CARD):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=color, lw=2.4))
    ax.add_patch(plt.Rectangle((x, y + h - 0.5), w, 0.5, color=color))
    ax.text(x + w / 2, y + h - 0.25, head, color="white", fontsize=hs, fontweight="bold", ha="center", va="center")
    ax.text(x + 0.14, y + h - 0.65, body, color=INK, fontsize=fs, va="top", linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22, lw=2.4, color=color))


def pipeline(ax, steps, y, h, color):
    n = len(steps); gap = 0.42; left = 0.5
    bw = (W - 2 * left - gap * (n - 1)) / n
    for i, (head, body) in enumerate(steps):
        x = left + i * (bw + gap)
        box(ax, x, y, bw, h, head, body, color=color, fs=12)
        if i < n - 1:
            arrow(ax, x + bw + 0.03, y + h / 2, x + bw + gap - 0.03, y + h / 2)


def panel(ax, x, y, w, h, head, lines, color, fs=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=CARD, ec=LINE, lw=1.6))
    ax.add_patch(plt.Rectangle((x, y + h - 0.08), w, 0.08, color=color))
    ax.text(x + 0.2, y + h - 0.42, head, fontsize=16, fontweight="bold", color=INK, va="center")
    ax.text(x + 0.2, y + h - 0.85, "\n".join(lines), fontsize=fs, color=INK, va="top", linespacing=1.5)


def section_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=13, color=MUTED, fontweight="bold")


# ------------------------------------------------------------------ 1. captioning
def captioning():
    c = "#e74c3c"
    fig, ax = canvas("Pipeline 1 · Video Captioning (CLIP + BLIP)",
                     "Video in, English and Arabic captioned videos out. Keyframes first, captions second.", c)
    section_label(ax, 0.5, 8.95, "PIPELINE")
    pipeline(ax, [
        ("1 Video input", "mp4 / mov\nOpenCV reads frames"),
        ("2 Frame sampling", "2 frames / second\n(fps_sample = 2)"),
        ("3 CLIP keyframes", "Embed each frame\n(512-d, L2-normalised)\nKeep if cosine sim to\nlast kept frame < 0.96"),
        ("4 BLIP captions", "One caption per keyframe\nBeam search, 5 beams\nmax 30 new tokens"),
        ("5 LLM narrative", "Denoise duplicates, then\nGroq LLM merges into a\nstory in EN / AR"),
        ("6 Burn captions", "Arabic reshaped + bidi\nCaptions drawn on video\n(moviepy)"),
        ("7 Output", "captioned_en.mp4\ncaptioned_ar.mp4\n+ metrics JSON"),
    ], 6.2, 2.55, c)

    section_label(ax, 0.5, 5.75, "MODELS: WHAT THEY DO, WHY, AND HOW THEY ARE BUILT")
    panel(ax, 0.5, 0.45, 6.55, 5.15, "CLIP ViT-B/32  (keyframe selection)", [
        "Why: raw pixel differences fire on lighting and",
        "noise. CLIP embeddings compare meaning, so a",
        "frame is kept only when the scene really changes.",
        "",
        "Architecture: two encoders, one shared space",
        "  Image: ViT-B/32, 32x32 patches, 12 layers",
        "  Text: 12-layer Transformer",
        "  Both project to a 512-d vector",
        "",
        "Loss: symmetric InfoNCE (contrastive).",
        "  Matching image-text pairs pulled together,",
        "  all other pairs in the batch pushed apart.",
        "Trained on 400M image-text pairs.",
    ], c, fs=12)
    panel(ax, 7.2, 0.45, 6.55, 5.15, "BLIP base  (caption generation)", [
        "Why: strong captions from a pretrained model,",
        "no training needed. Fits our use: short, factual",
        "descriptions of each keyframe.",
        "",
        "Architecture: Multimodal mixture of Encoder-Decoder",
        "  ViT image encoder",
        "  Text decoder with cross-attention to the image",
        "  (we use the image-grounded decoder)",
        "",
        "Losses in pretraining: ITC (contrastive) + ITM",
        "  (match / no match) + LM (next-word prediction).",
        "CapFilt: a captioner writes synthetic captions,",
        "  a filter removes noisy ones, then retrain.",
    ], c, fs=12)
    panel(ax, 13.9, 0.45, 5.6, 5.15, "Evaluation", [
        "Without reference captions (what we run):",
        "  Compression ratio = frames / keyframes",
        "  CLIPScore = 2.5 * max(cos(img, caption), 0)",
        "  Redundancy = word overlap of neighbour captions",
        "  Unique captions %, per-stage runtime",
        "",
        "With human reference captions (if available):",
        "  BLEU-4, METEOR, ROUGE-L",
        "  CIDEr (main COCO caption metric)",
        "  SPICE (objects, attributes, relations)",
        "",
        "CLIP stage itself: report keyframe recall",
        "against hand-marked scene changes.",
    ], c, fs=12)
    fig.savefig("pipeline_1_captioning.png"); plt.close(fig)


# ------------------------------------------------------------------ 2. tracking
def tracking():
    c = "#3498db"
    fig, ax = canvas("Pipeline 2 · Object Tracking",
                     "Detect every object, keep one ID per object through the video, then analyse what happened.", c)
    section_label(ax, 0.5, 8.95, "PIPELINE")
    pipeline(ax, [
        ("1 Video input", "Read frame by frame\n(streamed, low memory)"),
        ("2 YOLOv8 detect", "Boxes + class + score\nconf >= 0.3\nCOCO: 80 classes"),
        ("3 Predict", "Kalman filter predicts\nwhere each existing\ntrack should be now"),
        ("4 Associate", "IoU cost matrix\nHungarian matching\nByteTrack: 2 passes\n(high, then low score)"),
        ("5 Update tracks", "Matched: update state\nUnmatched: kept alive for\na buffer, then dropped\nNew: start a new ID"),
        ("6 Analytics", "Unique count per class\nDwell time, centre-line\ncrossings, heatmap,\ntrajectories"),
        ("7 Output", "Annotated video\n+ dashboard\n+ MOT txt for scoring"),
    ], 6.2, 2.55, c)

    section_label(ax, 0.5, 5.75, "MODELS: WHAT THEY DO, WHY, AND HOW THEY ARE BUILT")
    panel(ax, 0.5, 0.45, 6.55, 5.15, "YOLOv8  (detector)", [
        "Why: fast single-stage detector, good accuracy,",
        "sizes n/s/m/l let us trade speed for accuracy.",
        "",
        "Architecture:",
        "  Backbone: CSPDarknet with C2f blocks + SPPF",
        "  Neck: PAN-FPN (top-down + bottom-up fusion)",
        "  Head: decoupled (separate box / class branches),",
        "  anchor-free, 3 scales (strides 8, 16, 32)",
        "",
        "Loss: CIoU + DFL (box), BCE (class).",
        "  Task-Aligned Assigner picks positive samples.",
        "Detector metric: mAP@50 and mAP@50-95.",
    ], c, fs=12)
    panel(ax, 7.2, 0.45, 6.55, 5.15, "ByteTrack  (default) / BoT-SORT", [
        "Why ByteTrack: keeps low-confidence boxes in a",
        "second matching pass, so occluded objects keep",
        "their ID. Very fast; no appearance network.",
        "",
        "Building blocks: Kalman filter (motion model)",
        "  + Hungarian algorithm (assignment)",
        "  + IoU distance",
        "",
        "Why BoT-SORT (option): adds camera-motion",
        "  compensation, so it holds up when the camera",
        "  moves. Can add ReID appearance features.",
        "No tracker-specific loss: trackers are not trained.",
    ], c, fs=12)
    panel(ax, 13.9, 0.45, 5.6, 5.15, "Evaluation (needs ground truth)", [
        "Standard MOT metrics on MOT-format files:",
        "  MOTA = 1 - (FN + FP + IDSW) / GT",
        "    detection-heavy",
        "  IDF1 = identity-consistency over time",
        "    association-heavy",
        "  HOTA = sqrt(DetA * AssA), balances both",
        "  ID switches, fragmentations",
        "",
        "Tools: motmetrics (MOTA, IDF1), TrackEval (HOTA).",
        "Benchmarks: MOT17, MOT20.",
        "",
        "No ground truth? The dashboard shows ID churn,",
        "short-track % and mean track length as hints.",
    ], c, fs=12)
    fig.savefig("pipeline_2_tracking.png"); plt.close(fig)


# ------------------------------------------------------------------ 3. segmentation
def segmentation():
    c = "#2ecc71"
    fig, ax = canvas("Pipeline 3 · Instance Segmentation",
                     "A pixel mask for every object, not just a box. Optionally merged with tracking and captions.", c)
    section_label(ax, 0.5, 8.95, "PIPELINE")
    pipeline(ax, [
        ("1 Video input", "Read frame by frame"),
        ("2 Backbone + neck", "YOLOv8-seg:\nC2f backbone, SPPF,\nPAN-FPN features"),
        ("3 Two outputs", "Per object: box, class,\n32 mask coefficients.\nPer image: 32 prototype\nmasks (1/4 resolution)"),
        ("4 Build masks", "mask = sigmoid(\ncoefficients x prototypes)\nthen crop to the box"),
        ("5 Render", "Coloured transparent\nmask + outline per object\n(colour = object ID)"),
        ("6 Statistics", "Instances per class\nMean / max frame area\nArea over time"),
        ("7 Output", "Segmented video\n+ dashboard\n(Full analysis adds IDs\nand captions)"),
    ], 6.2, 2.55, c)

    section_label(ax, 0.5, 5.75, "MODELS: WHAT THEY DO, WHY, AND HOW THEY ARE BUILT")
    panel(ax, 0.5, 0.45, 6.55, 5.15, "YOLOv8-seg  (instance segmentation)", [
        "Why: one pass, real-time, and it shares the same",
        "backbone as our tracker's detector. Mask R-CNN",
        "is two-stage and slower on video.",
        "",
        "Architecture (YOLACT-style mask branch):",
        "  Same backbone + neck as YOLOv8 detection",
        "  Proto head: 32 prototype masks for the image",
        "  Detection head also predicts 32 coefficients",
        "  per object. Mask = coefficients x prototypes.",
        "",
        "Loss: box (CIoU + DFL) + class (BCE)",
        "  + mask (binary cross-entropy on the mask).",
        "Trained on COCO instance masks.",
    ], c, fs=12)
    panel(ax, 7.2, 0.45, 6.55, 5.15, "Why not the alternatives", [
        "Mask R-CNN: accurate, but two-stage and slower.",
        "",
        "SAM / SAM 2: segments anything and tracks masks,",
        "  but is heavy and has no class labels by itself.",
        "  A good upgrade path, not the real-time default.",
        "",
        "Semantic segmentation (U-Net, DeepLab): labels",
        "  each pixel by class but cannot separate two",
        "  people standing together. We need instances",
        "  so each object can also get a tracking ID.",
    ], c, fs=12)
    panel(ax, 13.9, 0.45, 5.6, 5.15, "Evaluation (needs ground truth)", [
        "Instance-level:",
        "  Mask mAP@50 and mAP@50-95 (COCO style)",
        "Per mask:",
        "  Mask IoU = overlap / union",
        "  Dice = 2 * overlap / (sum of areas)",
        "Boundary quality (optional): boundary F-score",
        "",
        "Ultralytics: model.val(data=...) reports mask mAP.",
        "Our code: segmentation/5_evaluation.py (mask_iou, dice).",
        "",
        "No ground truth? The dashboard shows confidence",
        "and area statistics per class. They describe",
        "the output but are not accuracy.",
    ], c, fs=12)
    fig.savefig("pipeline_3_segmentation.png"); plt.close(fig)


if __name__ == "__main__":
    captioning(); tracking(); segmentation(); print("done")
