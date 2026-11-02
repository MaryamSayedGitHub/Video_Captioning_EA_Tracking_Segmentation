"""Architecture figures for the study PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, LINE = "#12202f", "#5b6b7b", "#c9d2db"
C = dict(red="#e74c3c", blue="#3498db", green="#2ecc71", violet="#9b59b6", amber="#f1c40f", grey="#ecf0f3")


def fig(w=11, h=5.2):
    f = plt.figure(figsize=(w, h), dpi=170, facecolor="white")
    ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return f, ax


def bx(ax, x, y, w, h, text, fc="#ecf0f3", ec=INK, fs=10, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def ar(ax, x1, y1, x2, y2, c=INK, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.6, color=c, linestyle=ls))


def title(ax, t, w):
    ax.text(0.25, 4.95 if w else 0, t, fontsize=14, fontweight="bold", color=INK)


# ---------------------------------------------------------------- CLIP
f, ax = fig()
ax.text(0.25, 4.9, "CLIP ViT-B/32: two encoders, one shared 512-d space", fontsize=14, fontweight="bold", color=INK)
bx(ax, 0.3, 3.3, 1.5, 0.9, "Image\n224 x 224 x 3", C["grey"])
bx(ax, 2.2, 3.3, 1.7, 0.9, "Split into 32x32\npatches: 7x7 = 49\n+ [CLS] = 50 tokens", C["grey"], fs=8.5)
bx(ax, 4.3, 3.3, 2.1, 0.9, "ViT Transformer\n12 layers, width 768\n12 heads", C["red"], tc="white", bold=True, fs=9)
bx(ax, 6.8, 3.3, 1.5, 0.9, "Linear\n768 -> 512", C["grey"])
bx(ax, 8.7, 3.3, 1.9, 0.9, "L2 normalise\nimage vector v", C["amber"])
for a, b in [(1.8, 2.2), (3.9, 4.3), (6.4, 6.8), (8.3, 8.7)]:
    ar(ax, a, 3.75, b, 3.75)
bx(ax, 0.3, 1.7, 1.5, 0.9, "Text\n'a man walks'", C["grey"])
bx(ax, 2.2, 1.7, 1.7, 0.9, "BPE tokens\n<SOT> ... <EOT>\nmax 77", C["grey"], fs=8.5)
bx(ax, 4.3, 1.7, 2.1, 0.9, "Transformer (causal)\n12 layers, width 512\n8 heads", C["blue"], tc="white", bold=True, fs=9)
bx(ax, 6.8, 1.7, 1.5, 0.9, "Take <EOT>\nLinear 512->512", C["grey"], fs=9)
bx(ax, 8.7, 1.7, 1.9, 0.9, "L2 normalise\ntext vector t", C["amber"])
for a, b in [(1.8, 2.2), (3.9, 4.3), (6.4, 6.8), (8.3, 8.7)]:
    ar(ax, a, 2.15, b, 2.15)
ar(ax, 9.65, 3.3, 9.65, 2.6, c=C["red"])
bx(ax, 2.2, 0.25, 8.4, 0.9, "Training: batch of N pairs -> N x N cosine matrix x exp(learnable temperature)\n"
   "diagonal = true pairs. Cross-entropy over rows (image->text) and columns (text->image), averaged.", "#fff6d6", fs=9)
f.savefig("fig_clip.png"); plt.close(f)

# ---------------------------------------------------------------- BLIP
f, ax = fig()
ax.text(0.25, 4.9, "BLIP: Multimodal mixture of Encoder-Decoder (MED)", fontsize=14, fontweight="bold", color=INK)
bx(ax, 0.3, 1.2, 2.0, 2.6, "Image encoder\nViT-B/16\n\n(patch 16,\n12 layers,\nwidth 768)", C["red"], tc="white", bold=True, fs=10)
cols = [(3.6, "1  Text encoder\n(unimodal)\n\nBERT-like,\nbidirectional\nself-attention\n\nLoss: ITC", C["blue"]),
        (6.1, "2  Image-grounded\ntext encoder\n\nbidirectional SA\n+ cross-attention\nto image\n\nLoss: ITM", C["green"]),
        (8.6, "3  Image-grounded\ntext DECODER\n\ncausal SA\n+ cross-attention\nto image\n\nLoss: LM  (captioning)", C["violet"])]
for x, t, c in cols:
    bx(ax, x, 1.2, 2.2, 2.6, t, c, tc="white", bold=True, fs=9)
ax.plot([1.3, 1.3, 7.2, 7.2], [3.8, 4.35, 4.35, 3.82], color=C["red"], lw=1.8)
ax.plot([9.7, 9.7], [4.35, 3.82], color=C["red"], lw=1.8); ax.plot([7.2, 9.7], [4.35, 4.35], color=C["red"], lw=1.8)
ar(ax, 7.2, 3.95, 7.2, 3.8, c=C["red"]); ar(ax, 9.7, 3.95, 9.7, 3.8, c=C["red"])
ax.text(2.0, 4.45, "image tokens feed the cross-attention of blocks 2 and 3 (block 1 sees text only)", fontsize=8.5, color=C["red"])
ax.text(3.6, 0.82, "Shares weights with 2 & 3\nexcept self-attention layers", fontsize=8.5, color=MUTED)
bx(ax, 0.3, 0.1, 10.4, 0.55, "Our project uses ONLY block 3 (decoder) at inference: image -> ViT -> decoder generates the caption word by word (beam search, 5 beams).",
   "#fff6d6", fs=9.5)
f.savefig("fig_blip.png"); plt.close(f)

# ---------------------------------------------------------------- YOLOv8
f, ax = fig(11, 5.6)
ax.text(0.25, 5.3, "YOLOv8 detection architecture (input 640 x 640)", fontsize=14, fontweight="bold", color=INK)
ax.text(0.3, 4.85, "BACKBONE (CSPDarknet + C2f)", fontsize=10, color=MUTED, fontweight="bold")
bb = [("Conv s2\nP1/2", 0.3), ("Conv s2 + C2f\nP2/4", 1.75), ("Conv s2 + C2f\nP3/8", 3.2), ("Conv s2 + C2f\nP4/16", 4.65), ("Conv s2 + C2f\n+ SPPF  P5/32", 6.1)]
for t, x in bb:
    bx(ax, x, 3.7, 1.3, 0.95, t, C["red"], tc="white", bold=True, fs=8.2)
for i in range(4):
    ar(ax, bb[i][1] + 1.3, 4.17, bb[i + 1][1], 4.17)
ax.text(0.3, 3.35, "NECK (PAN-FPN)", fontsize=9.5, color=MUTED, fontweight="bold")
nk = [("P3 out\n(small objects)\nstride 8", 3.2), ("P4 out\n(medium)\nstride 16", 4.65), ("P5 out\n(large)\nstride 32", 6.1)]
for t, x in nk:
    bx(ax, x, 2.1, 1.3, 0.95, t, C["blue"], tc="white", bold=True, fs=8.2)
for (t, x) in nk:
    ar(ax, x + 0.65, 3.7, x + 0.65, 3.05, c=MUTED, ls="--")
ar(ax, 4.5, 2.35, 4.65, 2.35, c=C["blue"]); ar(ax, 5.95, 2.35, 6.1, 2.35, c=C["blue"])
ar(ax, 4.65, 2.75, 4.5, 2.75, c=C["blue"]); ar(ax, 6.1, 2.75, 5.95, 2.75, c=C["blue"])
ax.text(0.3, 1.7, "HEAD (decoupled, anchor-free) - same head on every scale", fontsize=9.5, color=MUTED, fontweight="bold")
bx(ax, 8.1, 2.9, 2.7, 1.0, "Box branch: 2 convs ->\n4 x 16 numbers\n(distribution per side)", C["green"], tc="white", bold=True, fs=8.5)
bx(ax, 8.1, 1.65, 2.7, 1.0, "Class branch: 2 convs ->\n80 scores (sigmoid)\nno objectness score", C["violet"], tc="white", bold=True, fs=8.5)
ar(ax, 7.4, 2.6, 8.1, 3.3, c=INK); ar(ax, 7.4, 2.4, 8.1, 2.15, c=INK)
bx(ax, 0.3, 0.1, 10.5, 1.4, "Neck: top-down path (upsample + concat + C2f), then bottom-up path (strided conv + concat + C2f).\nDecoding: each grid cell predicts distances left/top/right/bottom to its box edges (anchor-free).\n"
   "Each distance = expected value of a 16-bin distribution (that is what DFL trains).  Then confidence filter + NMS.\n"
   "Training loss = 7.5 x CIoU + 1.5 x DFL (box)  +  0.5 x BCE (class).  Positive cells chosen by the Task-Aligned Assigner.",
   "#fff6d6", fs=9)
f.savefig("fig_yolo.png"); plt.close(f)

# ---------------------------------------------------------------- YOLOv8-seg
f, ax = fig(11, 5.2)
ax.text(0.25, 4.9, "YOLOv8-seg: detection head + prototype masks", fontsize=14, fontweight="bold", color=INK)
bx(ax, 0.3, 2.7, 1.9, 1.1, "Backbone + PAN-FPN\n(same as detection)", C["red"], tc="white", bold=True, fs=9)
bx(ax, 3.0, 3.5, 2.6, 0.95, "Detection head (per cell)\nbox + class + 32 coefficients", C["green"], tc="white", bold=True, fs=8.8)
bx(ax, 3.0, 2.0, 2.6, 0.95, "Proto head (from P3)\n-> 32 prototype masks,\n160x160 (1/4 resolution)", C["violet"], tc="white", bold=True, fs=8.8)
ar(ax, 2.2, 3.4, 3.0, 3.95); ar(ax, 2.2, 3.0, 3.0, 2.5)
bx(ax, 6.4, 3.5, 1.9, 0.95, "After NMS: per object\nc = (c1 ... c32)", C["grey"], fs=9)
bx(ax, 6.4, 2.0, 1.9, 0.95, "Prototypes\nP = (P1 ... P32)", C["grey"], fs=9)
ar(ax, 5.6, 3.97, 6.4, 3.97); ar(ax, 5.6, 2.47, 6.4, 2.47)
bx(ax, 8.9, 2.75, 2.0, 1.2, "mask logits =\nc1 P1 + ... + c32 P32\nsigmoid > 0.5\ncrop to the box", C["amber"], fs=9)
ar(ax, 8.3, 3.97, 8.9, 3.6); ar(ax, 8.3, 2.47, 8.9, 3.1)
bx(ax, 0.3, 0.2, 10.6, 1.35, "Why it is fast: ONE pass. Two objects only need two small coefficient vectors; the expensive mask features (prototypes) are computed once per image.\n"
   "Training loss = box (CIoU + DFL) + class (BCE) + mask (per-pixel BCE between sigmoid(c . P) and the true mask, inside the box,\n"
   "normalised by the box area). Metrics: mask mAP@50 and mAP@50-95, mask IoU, Dice.", "#fff6d6", fs=9)
f.savefig("fig_yoloseg.png"); plt.close(f)

# ---------------------------------------------------------------- ByteTrack
f, ax = fig(11, 5.4)
ax.text(0.25, 5.1, "ByteTrack: what happens on every frame", fontsize=14, fontweight="bold", color=INK)
bx(ax, 0.3, 3.8, 1.9, 0.8, "YOLO detections\n(boxes + scores)", C["green"], tc="white", bold=True, fs=9)
bx(ax, 2.7, 4.0, 2.3, 0.6, "score >= high thr  -> HIGH", C["grey"], fs=9)
bx(ax, 2.7, 3.2, 2.3, 0.6, "low thr <= score < high -> LOW", C["grey"], fs=9)
ar(ax, 2.2, 4.2, 2.7, 4.3); ar(ax, 2.2, 4.0, 2.7, 3.5)
bx(ax, 0.3, 2.0, 1.9, 0.8, "Existing tracks\nKalman PREDICT\nnew position", C["blue"], tc="white", bold=True, fs=8.5)
bx(ax, 5.5, 3.85, 2.5, 0.75, "Pass 1: match tracks <-> HIGH\ncost = 1 - IoU, Hungarian", C["amber"], fs=8.8)
bx(ax, 5.5, 2.85, 2.5, 0.75, "Pass 2: leftover tracks <-> LOW\n(recovers occluded objects)", C["amber"], fs=8.8)
ar(ax, 5.0, 4.3, 5.5, 4.2); ar(ax, 5.0, 3.5, 5.5, 3.25)
ar(ax, 2.2, 2.5, 5.5, 3.95, c=C["blue"], ls="--")
ar(ax, 6.75, 3.85, 6.75, 3.6, c=INK)
bx(ax, 8.5, 3.85, 2.4, 0.75, "Matched: Kalman UPDATE\n(same ID)", C["grey"], fs=9)
bx(ax, 8.5, 2.95, 2.4, 0.75, "Unmatched HIGH detection\n-> NEW track (new ID)", C["grey"], fs=9)
bx(ax, 8.5, 2.05, 2.4, 0.75, "Unmatched track: keep 'lost'\nfor track_buffer frames,\nthen delete", C["grey"], fs=8.5)
ar(ax, 8.0, 4.2, 8.5, 4.2); ar(ax, 8.0, 3.3, 8.5, 3.3); ar(ax, 8.0, 3.1, 8.5, 2.4)
bx(ax, 0.3, 0.2, 10.6, 1.4, "Kalman filter state = box position + size + their velocities (8 numbers). Predict: where should the object be now?  Update: blend the prediction\n"
   "with the measured box, weighted by how much each is trusted. Hungarian algorithm: the assignment with the lowest TOTAL cost.\n"
   "BoT-SORT = same loop + camera-motion compensation (so predictions stay right when the camera moves) + optional ReID appearance.",
   "#fff6d6", fs=9)
f.savefig("fig_bytetrack.png"); plt.close(f)
print("figs ok")
