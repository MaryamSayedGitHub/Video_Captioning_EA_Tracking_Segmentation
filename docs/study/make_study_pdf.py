from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, PageBreak,
                                Table, TableStyle, KeepTogether, CondPageBreak)

D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", D + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", D + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#12202f"); MUTED = colors.HexColor("#5b6b7b"); LINE = colors.HexColor("#c9d2db")
ACC = colors.HexColor("#e74c3c")
PW, PH = A4
FW = PW - 3.4 * cm

H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=19, leading=23, textColor=INK, spaceBefore=0, spaceAfter=6)
H2 = ParagraphStyle("h2", keepWithNext=1, fontName="DV-B", fontSize=12.2, leading=15.5, textColor=INK, spaceBefore=9, spaceAfter=3)
H3 = ParagraphStyle("h3", keepWithNext=1, fontName="DV-B", fontSize=10.2, leading=13, textColor=ACC, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("b", fontName="DV", fontSize=9.4, leading=13, textColor=INK, leftIndent=12, bulletIndent=1, spaceAfter=1.8)
P = ParagraphStyle("p", fontName="DV", fontSize=9.4, leading=13, textColor=INK, spaceAfter=3)
SM = ParagraphStyle("sm", parent=P, fontSize=8.2, leading=11, textColor=MUTED)
F = ParagraphStyle("f", fontName="DVM", fontSize=8.6, leading=12.2, textColor=INK)
CELL = ParagraphStyle("c", fontName="DV", fontSize=8.3, leading=10.8, textColor=INK)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="DV-B")


def bl(items): return [Paragraph(t, B, bulletText="•") for t in items]


def formula(lines):
    t = Table([[Paragraph("<br/>".join(lines), F)]], colWidths=[FW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef2f6")), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, ACC)]))
    return [t, Spacer(1, 4)]


def example(title, lines):
    inner = [Paragraph(f"<b>Worked example: {title}</b>", P)] + [Paragraph(l, P) for l in lines]
    t = Table([[inner]], colWidths=[FW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff6d6")), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return [t, Spacer(1, 5)]


def tbl(rows, widths, head=True):
    data = [[Paragraph(str(c), CELLB if (head and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf0")))
    t.setStyle(TableStyle(st)); return [t, Spacer(1, 5)]


def fig(path, h_ratio):
    return [Image(path, width=FW, height=FW * h_ratio), Spacer(1, 4)]


def deco(c, doc):
    c.saveState()
    c.setFillColor(INK); c.rect(0, PH - 0.45 * cm, PW, 0.45 * cm, fill=1, stroke=0)
    c.setFont("DV", 7.5); c.setFillColor(MUTED)
    c.drawString(1.7 * cm, 0.8 * cm, "Video Intelligence: study notes for the interview")
    c.drawRightString(PW - 1.7 * cm, 0.8 * cm, str(doc.page))
    c.restoreState()


doc = BaseDocTemplate("Study_Notes_Models_Losses_Metrics.pdf", pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                      topMargin=1.3 * cm, bottomMargin=1.4 * cm, title="Video Intelligence: models, losses, metrics",
                      author="Maryam Sayed")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(1.7 * cm, 1.4 * cm, FW, PH - 2.7 * cm, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=deco)])
S = []
def H(t): S.append(PageBreak()); S.append(Paragraph(t, H1))

# ======================================================= 0. map
S += [Paragraph("Video Intelligence: every model, loss and metric", H1),
      Paragraph("Study notes to keep beside you in the interview. Concepts, architecture and hand calculations. No code.", P), Spacer(1, 4)]
S.append(Paragraph("How to use these notes", H2))
S += bl(["Section 1 is the map: what is in the project, what is new, what each model is for.",
         "Sections 2 to 7 take one model or topic each: <b>what it solves, architecture, loss, metrics, worked numbers</b>.",
         "Section 8 lists every setting we chose and why. Section 9 is a question bank with short answers.",
         "Numbers marked <i>(from papers / docs)</i> are published values; check the source if you quote them exactly."])
S.append(Paragraph("1. The map", H2))
S += tbl([
    ["Stage", "Model / method", "Role", "Trained loss", "Metrics we use or can use"],
    ["Keyframes", "CLIP ViT-B/32", "Embed frames; keep a frame only if the scene changed", "Symmetric InfoNCE (contrastive)", "Compression ratio; keyframe recall (needs labels)"],
    ["Captions", "BLIP base (ViT-B/16 + text decoder)", "Describe each keyframe", "LM loss (fine-tune); ITC + ITM + LM (pre-train)", "CLIPScore, redundancy, unique %; with references: BLEU, METEOR, ROUGE-L, CIDEr, SPICE"],
    ["Narrative", "LLM on Groq (gpt-oss-20b)", "Merge captions into one story, EN / AR", "Next-token cross-entropy (not trained by us)", "Not evaluated (text rewriting only)"],
    ["Detection", "YOLOv8 (n/s/m/l)", "Find objects in each frame", "CIoU + DFL (box), BCE (class)", "mAP@50, mAP@50-95, precision, recall"],
    ["Tracking", "ByteTrack or BoT-SORT (Kalman + Hungarian)", "Keep one ID per object over time", "None (not trained)", "MOTA, IDF1, HOTA, ID switches; proxies without labels"],
    ["Segmentation", "YOLOv8-seg", "Pixel mask per object", "Box + class + mask BCE", "Mask mAP, mask IoU, Dice"],
    ["Analytics", "Counting, dwell time, line crossing, heatmap", "Explain what happened", "n/a", "ID churn, short-track %, mean track length"],
], [2.0 * cm, 3.3 * cm, 3.5 * cm, 3.5 * cm, FW - 12.3 * cm])
S.append(Paragraph("What is new compared with the original project", H2))
S += tbl([
    ["Original (kept as is)", "New (added)"],
    ["Frame extraction, CLIP keyframes, BLIP captions, Groq narrative, burned EN / AR captions. TTS and voice-over stay in the folder but are out of scope.",
     "Tracking (YOLOv8 + ByteTrack / BoT-SORT) with analytics dashboard; instance segmentation (YOLOv8-seg); a Full analysis video that merges masks, IDs and captions; "
     "evaluation (caption proxies, MOT export, mask IoU / Dice); a Flask web app with four pages and background jobs."],
], [FW / 2, FW / 2])
S.append(Paragraph("Honest status: all models are pretrained (COCO for YOLO, web image-text pairs for CLIP, BLIP's own data). Nothing is trained or fine-tuned by us. "
                   "No ground-truth labels exist for our videos, so accuracy numbers (IDF1, MOTA, HOTA, mask mAP) are not yet measured. "
                   "The web app exports files so they can be.", P))

# ======================================================= 2. CLIP
H("2. CLIP ViT-B/32 (keyframe selection)")
S.append(Paragraph("What problem it solves", H2))
S += bl(["A video at 2 fps still has many near-identical frames. Captioning all of them is slow and gives repeated sentences.",
         "Pixel differences react to lighting, noise and camera shake. We want a change in <b>meaning</b>.",
         "CLIP maps an image to a 512-number vector where similar content gives similar vectors. Cosine similarity of two frames = how alike their content is."])
S += fig("fig_clip.png", 5.2 / 11)
S.append(Paragraph("Architecture, step by step", H2))
S += bl(["<b>Image side (ViT-B/32):</b> 224×224 image cut into 32×32 patches = 7×7 = 49 patches. Each patch is flattened and linearly projected to 768 numbers. "
         "A learnable [CLS] token is added, so 50 tokens. Position embeddings are added. 12 Transformer layers (self-attention + MLP), 12 heads. "
         "The [CLS] output goes through a linear layer to 512 numbers.",
         "<b>Text side:</b> byte-pair tokens, max 77, start and end tokens. 12-layer Transformer, width 512, 8 heads, causal attention. "
         "The vector at the end token goes through a linear layer to 512 numbers.",
         "Both vectors are L2-normalised (length 1). Then cosine similarity = dot product.",
         "Why '32' matters: larger patches mean fewer tokens, so B/32 is the fastest CLIP ViT. B/16 is more accurate but about 4× the tokens.",
         "Trained on about 400 million image-text pairs from the web <i>(from papers / docs)</i>."])
S.append(Paragraph("Loss: symmetric InfoNCE (contrastive)", H2))
S += bl(["Take a batch of N matched (image, text) pairs. Compute the N×N matrix of cosine similarities, multiply by exp(τ) where τ is a learnable temperature.",
         "Row i = image i against all N texts. The correct answer is column i. Cross-entropy on that row pushes the true pair up and the other N-1 down.",
         "Do the same for columns (text to image). Average both directions.",
         "Intuition: it is an N-way multiple-choice question 'which text belongs to this image?' asked in both directions."])
S += formula(["L_img->txt = -(1/N) Σ_i log( exp(s_ii) / Σ_j exp(s_ij) )",
              "L_txt->img = -(1/N) Σ_i log( exp(s_ii) / Σ_j exp(s_ji) )",
              "L = ( L_img->txt + L_txt->img ) / 2        s_ij = exp(τ) · cos(v_i, t_j)"])
S += example("InfoNCE with N = 2", [
    "Scaled similarity matrix (rows = images, columns = texts): [[4, 1], [0, 3]]. Diagonal = true pairs.",
    "Image 1: softmax of [4, 1] gives e<super>4</super>/(e<super>4</super>+e<super>1</super>) = 54.60/57.32 = 0.953, loss = -ln 0.953 = 0.049. Image 2: row [0, 3] gives 20.09/21.09 = 0.953, loss 0.049.",
    "Text 1: column [4, 0] gives 54.60/55.60 = 0.982, loss 0.018. Text 2: column [1, 3] gives 20.09/22.80 = 0.881, loss 0.127.",
    "L_img->txt = 0.049, L_txt->img = (0.018 + 0.127)/2 = 0.073, so L = (0.049 + 0.073)/2 = <b>0.061</b>. Text 2 hurts most because its column has a wrong text scoring 1 vs the true 3."])
S.append(Paragraph("How we use CLIP in the project", H2))
S += bl(["Embed each sampled frame (2 per second). Keep frame 0. For every next frame compute cos(frame, <b>last kept</b> frame). Keep it if cosine < 0.96, then it becomes the new reference.",
         "Higher threshold = more keyframes (more detail, slower). Lower = fewer keyframes.",
         "Comparing with the last <b>kept</b> frame (not the previous frame) stops slow drift: many tiny changes can add up to a big one."])
S += example("keyframe rule with threshold 0.96", [
    "Frames f0..f5. f0 kept. Cosine to f0: f1 = 0.99 skip, f2 = 0.97 skip, f3 = 0.93 <b>keep</b> (reference is now f3). Cosine to f3: f4 = 0.98 skip, f5 = 0.94 <b>keep</b>.",
    "Keyframes = f0, f3, f5. Compression ratio = 6 frames / 3 keyframes = <b>2.0</b>."])
S.append(Paragraph("Metrics for this stage", H2))
S += bl(["<b>Compression ratio</b> = frames sampled / keyframes kept. Tells how much redundancy was removed. Not a quality score by itself.",
         "<b>Keyframe recall / precision</b> (needs labels): hand-mark the true scene changes. Recall = marked changes that have a keyframe within ±1 s. Precision = keyframes that match a marked change.",
         "Weak points: 0.96 is a chosen value, not learned. CLIP may ignore small but important changes (a person entering far away)."])

# ======================================================= 3. BLIP
H("3. BLIP base (caption generation)")
S.append(Paragraph("What problem it solves", H2))
S += bl(["Turns a keyframe into one sentence. We use the public checkpoint <i>Salesforce/blip-image-captioning-base</i>: a BLIP model with a ViT-B image encoder, fine-tuned for captioning.",
         "BLIP's idea: web image-text pairs are noisy. It cleans them by letting the model write its own captions and filter bad ones (CapFilt), then trains on the cleaner data."])
S += fig("fig_blip.png", 5.2 / 11)
S.append(Paragraph("Architecture: the MED (Multimodal mixture of Encoder-Decoder)", H2))
S += bl(["<b>Image encoder:</b> a ViT with 16×16 patches, 12 layers, width 768. Output = one vector per patch.",
         "<b>1. Unimodal text encoder:</b> BERT-style, bidirectional self-attention, no image. Used for the contrastive loss (ITC).",
         "<b>2. Image-grounded text encoder:</b> same, plus a <b>cross-attention</b> layer in every block that lets text tokens look at image tokens. Used for match / no-match (ITM).",
         "<b>3. Image-grounded text decoder:</b> bidirectional self-attention replaced by <b>causal</b> self-attention (a word sees only earlier words) plus cross-attention. Used for generation (LM). <b>This is the part our project runs.</b>",
         "Weight sharing: text encoder and decoder share all weights except the self-attention layers, which saves parameters.",
         "Cross-attention in one line: queries come from the text, keys and values come from the image patches, so each word decides which image regions to read."])
S.append(Paragraph("Losses in pre-training (three, one per block)", H2))
S += bl(["<b>ITC (image-text contrastive):</b> same idea as CLIP's InfoNCE, on the unimodal encoders. BLIP uses a momentum encoder and soft targets (momentum distillation) to cope with noisy pairs.",
         "<b>ITM (image-text matching):</b> binary classifier on the image-grounded encoder: does this text match this image? Trained with <b>hard negatives</b>: texts that look similar by ITC but are wrong.",
         "<b>LM (language modelling):</b> the decoder predicts the next word given the image and previous words. Cross-entropy, with label smoothing 0.1 <i>(from the paper)</i>.",
         "<b>Fine-tuning for captioning (the checkpoint we use):</b> LM loss only, on COCO captions, with a prompt such as 'a picture of'."])
S += formula(["L_LM = - Σ_t log p( w_t | w_<t , image )",
              "L_ITM = - [ y log p(match) + (1 - y) log(1 - p(match)) ]"])
S += example("LM loss for 'a man walks'", [
    "The decoder gives these probabilities for the true next words: a = 0.40, man = 0.50, walks = 0.25, end = 0.60.",
    "Loss = -(ln 0.40 + ln 0.50 + ln 0.25 + ln 0.60) = 0.916 + 0.693 + 1.386 + 0.511 = 3.51, per word 0.88. Training lowers this over many images."])
S.append(Paragraph("CapFilt: bootstrapping clean data", H2))
S += bl(["Start with web pairs (noisy) and human COCO pairs (clean). Fine-tune a <b>captioner</b> (decoder) and a <b>filter</b> (ITM encoder) on COCO.",
         "The captioner writes a synthetic caption for each web image. The filter removes bad web captions and bad synthetic ones.",
         "Pre-train a fresh model on clean web + synthetic + human pairs. Result: better downstream captioning and retrieval."])
S.append(Paragraph("Inference in our project (settings from the code)", H2))
S += bl(["Image resized to the model's input size, ViT encodes it once.",
         "Decoder generates left to right. <b>Beam search with 5 beams</b>: keep the 5 best partial sentences at every step, choose the best finished one.",
         "<b>max_new_tokens = 30</b> limits length. <b>repetition_penalty = 1.2</b> lowers the score of words already used, to avoid 'a man a man a man'.",
         "Beam search is deterministic: same frame, same caption. Nucleus sampling would be more varied but less stable."])
S.append(Paragraph("Metrics: with and without human captions", H2))
S += bl(["<b>Without references (what we run):</b> CLIPScore, redundancy, unique-caption %.",
         "<b>With references (if you had human captions):</b> BLEU, METEOR, ROUGE-L, CIDEr, SPICE. Details below."])
S += formula(["CLIPScore = 2.5 · max( cos(image_vec, caption_vec), 0 )",
              "Redundancy = mean over neighbouring caption pairs of  |words(a) ∩ words(b)| / |words(a) ∪ words(b)|",
              "Unique % = 100 · (number of distinct captions) / (number of captions)"])
S += example("CLIPScore", ["Cosine between a keyframe and its caption = 0.31, so CLIPScore = 2.5 × 0.31 = <b>0.775</b>. Average over all keyframes for the video score.",
                           "Reading it: higher = the caption matches the image better in CLIP's view. It does not check grammar, Arabic quality or facts that CLIP cannot see."])
S.append(Paragraph("Reference-based caption metrics (know how each is computed)", H2))
S += tbl([
    ["Metric", "Idea", "Formula in words", "Weakness"],
    ["BLEU-n", "n-gram precision", "Geometric mean of clipped n-gram precisions × brevity penalty BP = exp(1 - r/c) if candidate length c < reference length r", "Ignores meaning and recall; synonyms score 0"],
    ["METEOR", "Unigram match with stems and synonyms", "F-mean (recall weighted more) × (1 - fragmentation penalty)", "Needs language resources; weak for Arabic"],
    ["ROUGE-L", "Longest common subsequence", "F-score of LCS/len(candidate) and LCS/len(reference)", "Order only, no meaning"],
    ["CIDEr", "TF-IDF weighted n-gram cosine", "Average over n=1..4 of cosine between TF-IDF vectors of candidate and references, ×10", "Needs several references; main COCO caption metric"],
    ["SPICE", "Scene-graph match", "F-score over matched objects, attributes, relations parsed from both captions", "Depends on the parser; slow"],
], [1.7 * cm, 3.3 * cm, 7.0 * cm, FW - 12.0 * cm])
S += example("BLEU-2 and ROUGE-L", [
    "Candidate: 'a man walks on the street' (6 words). Reference: 'a man is walking on the street' (7 words).",
    "Unigram matches: a, man, on, the, street = 5, so p1 = 5/6. Bigrams: 'a man', 'on the', 'the street' match = 3 of 5, so p2 = 3/5.",
    "Brevity penalty: candidate shorter than reference, BP = exp(1 - 7/6) = 0.847. <b>BLEU-2 = 0.847 × sqrt(5/6 × 3/5) = 0.847 × 0.707 = 0.599.</b>",
    "ROUGE-L: longest common subsequence = 'a man on the street' = 5. P = 5/6, R = 5/7, F1 = 2PR/(P+R) = <b>0.769</b>."])

# ======================================================= 4. YOLOv8
H("4. YOLOv8 (detection)")
S.append(Paragraph("What problem it solves", H2))
S += bl(["Finds every object in a frame: a box, a class and a confidence score. Tracking needs these boxes as input.",
         "Single-stage: one forward pass gives all boxes. Faster than two-stage detectors (Faster R-CNN) that first propose regions.",
         "Sizes n, s, m, l (and x). We default to <b>s</b>: a balanced choice. Larger = more accurate and slower."])
S += fig("fig_yolo.png", 5.6 / 11)
S.append(Paragraph("Architecture, step by step", H2))
S += bl(["<b>Backbone (CSPDarknet with C2f):</b> a stack of strided 3×3 convolutions that halve the resolution (strides 2, 4, 8, 16, 32) while adding channels. "
         "<b>C2f block</b> = split the features in two, pass one part through several small bottleneck blocks, <b>concatenate every intermediate output</b>, then a 1×1 conv. "
         "More gradient paths and feature reuse than a plain chain, at low cost.",
         "<b>SPPF</b> (spatial pyramid pooling, fast): three max-pools of size 5 applied one after another, all outputs concatenated. Gives a large receptive field cheaply.",
         "<b>Neck (PAN-FPN):</b> features from strides 8, 16, 32 are fused. Top-down path (upsample deep features, concat with shallower ones) gives small layers semantic strength. "
         "Bottom-up path (strided conv, concat) passes precise location information back up. Output: three feature maps (P3, P4, P5).",
         "<b>Head (decoupled, anchor-free):</b> separate small conv branches for box and class (classification and localisation want different features). "
         "No anchors: each grid cell directly predicts the distance to the four box edges. No objectness score in v8: confidence = the class score.",
         "<b>Box as a distribution:</b> each edge distance is predicted as a probability distribution over 16 bins; the value used = expected value. This is what DFL trains.",
         "Why three scales: P3 (stride 8) sees small objects, P5 (stride 32) sees large ones.",
         "<b>Post-processing:</b> drop boxes under the confidence threshold (we use 0.3), then NMS (non-maximum suppression) removes duplicate boxes of the same object."])
S.append(Paragraph("Training loss (three terms)", H2))
S += bl(["<b>Box loss = CIoU loss.</b> IoU alone gives no gradient when boxes do not overlap. CIoU adds a centre-distance term and an aspect-ratio term.",
         "<b>DFL (distribution focal loss)</b>: trains the 16-bin distribution so that the probability mass sits on the two bins around the true distance.",
         "<b>Class loss = BCE</b> (binary cross-entropy) per class, each class independent (sigmoid, not softmax).",
         "<b>Task-Aligned Assigner (TAL)</b> decides which grid cells are 'positives' for each ground-truth box, using a score that combines predicted class score and IoU: t = s<super>α</super> · u<super>β</super> (defaults α = 0.5, β = 6, top-10 candidates, check your installed config). "
         "It aligns classification confidence with localisation quality.",
         "Total = 7.5 × box + 0.5 × class + 1.5 × DFL (default gains)."])
S += formula(["IoU = |A ∩ B| / |A ∪ B|",
              "L_CIoU = 1 - IoU + ρ²(b, b_gt)/c² + α·v",
              "v = (4/π²) · ( arctan(w_gt/h_gt) - arctan(w/h) )²        α = v / (1 - IoU + v)",
              "L_DFL = - [ (y_{i+1} - y)·log S_i + (y - y_i)·log S_{i+1} ]        y_i ≤ y ≤ y_{i+1}",
              "L_BCE = - [ y·log p + (1 - y)·log(1 - p) ]"])
S += example("CIoU", [
    "Predicted box A = (0,0)-(10,10). True box B = (5,5)-(15,15). Intersection 5×5 = 25. Union = 100 + 100 - 25 = 175. IoU = 25/175 = 0.143.",
    "Centre distance squared ρ² = 5² + 5² = 50. Smallest box enclosing both is 15×15, so c² = 15² + 15² = 450. ρ²/c² = 0.111.",
    "Same aspect ratio, so v = 0. <b>L_CIoU = 1 - 0.143 + 0.111 = 0.968.</b> If the boxes did not overlap, IoU = 0 but the distance term would still give a gradient."])
S += example("DFL", [
    "True distance y = 3.4 lies between bins 3 and 4. Weights: bin 3 gets (4 - 3.4) = 0.6, bin 4 gets 0.4. Model's softmax: S<sub>3</sub> = 0.5, S<sub>4</sub> = 0.3.",
    "L_DFL = -(0.6·ln 0.5 + 0.4·ln 0.3) = 0.6×0.693 + 0.4×1.204 = <b>0.898</b>. Lowest when all mass sits on bins 3 and 4 in the 60 / 40 ratio."])
S.append(Paragraph("Metrics: precision, recall, AP, mAP", H2))
S += bl(["A detection is a <b>TP</b> if its class is right and IoU with an unmatched true box ≥ the IoU threshold; otherwise <b>FP</b>. Unmatched true boxes are <b>FN</b>.",
         "<b>Precision</b> = TP/(TP+FP): of what I predicted, how much is right. <b>Recall</b> = TP/(TP+FN): of what exists, how much I found.",
         "<b>AP</b> for one class: sort detections by confidence, walk down the list, record precision and recall, take the area under the (interpolated) precision-recall curve.",
         "<b>mAP@50</b> = mean AP over classes at IoU 0.5. <b>mAP@50-95</b> = also average over IoU thresholds 0.50, 0.55, ..., 0.95 (COCO standard). It rewards tight boxes.",
         "Published COCO val numbers for YOLOv8 detection, mAP@50-95: n 37.3, s 44.9, m 50.2, l 52.9; parameters about 3.2M, 11.2M, 25.9M, 43.7M <i>(from Ultralytics docs; verify)</i>."])
S += example("AP from a ranked list", [
    "4 true objects. Five detections by confidence: TP, TP, FP, TP, FP.",
    "After each: (precision, recall) = (1/1, 1/4), (2/2, 2/4), (2/3, 2/4), (3/4, 3/4), (3/5, 3/4).",
    "Interpolated precision = highest precision at the same or higher recall: recall 0.25 → 1.0, 0.50 → 1.0, 0.75 → 0.75.",
    "AP = 0.25×1.0 + 0.25×1.0 + 0.25×0.75 + 0 (recall never reaches 1.0) = <b>0.6875</b>. COCO samples 101 recall points; the idea is the same."])

# ======================================================= 5. YOLOv8-seg
H("5. YOLOv8-seg (instance segmentation)")
S.append(Paragraph("What problem it solves", H2))
S += bl(["A box includes background. A <b>mask</b> marks exactly the pixels of each object, and keeps two touching people as two objects (<b>instance</b> segmentation).",
         "Semantic segmentation (U-Net, DeepLab) gives one label per pixel but cannot separate two people. Instance masks can get their own tracking ID.",
         "Alternatives: Mask R-CNN (accurate, two-stage, slower), SAM / SAM 2 (segments anything, heavy, no class names by itself)."])
S += fig("fig_yoloseg.png", 5.2 / 11)
S.append(Paragraph("Architecture: what is added to the detector", H2))
S += bl(["Same backbone and PAN-FPN as detection.",
         "<b>Proto head:</b> takes the highest-resolution neck feature (P3) and upsamples to 1/4 of the input (160×160 for 640 input). Output: <b>32 prototype masks</b> for the whole image. "
         "Think of them as 32 reusable 'building-block' masks, for example 'left side', 'edges', 'top region'.",
         "<b>Detection head</b> now also outputs <b>32 mask coefficients</b> per object (besides box and class).",
         "<b>Mask of one object</b> = sigmoid( c1·P1 + c2·P2 + ... + c32·P32 ), keep pixels above 0.5, <b>crop to the predicted box</b>. It is a weighted mix of the building blocks.",
         "Cost: one matrix multiplication per object. The heavy part (prototypes) is computed once per image, so cost barely grows with the number of objects.",
         "Idea comes from YOLACT. Limit: mask resolution is 1/4 of the input, so thin details and exact boundaries are approximate."])
S.append(Paragraph("Loss", H2))
S += bl(["Box (CIoU + DFL) and class (BCE) exactly as in detection.",
         "<b>Mask loss:</b> binary cross-entropy between the predicted mask (sigmoid of c·P) and the true mask, computed inside the box and normalised by the box area so that big objects do not dominate.",
         "Trained on COCO instance masks (80 classes)."])
S += example("one mask pixel", [
    "At a pixel, two prototype values P1 = 0.9, P2 = 0.2, other coefficients 0. Coefficients c1 = 4, c2 = -4. Logit = 4×0.9 + (-4)×0.2 = 2.8. sigmoid(2.8) = 0.94 → the pixel belongs to the object.",
    "True mask says 1: BCE = -ln 0.94 = 0.062 (small). If the true label were 0: BCE = -ln(0.06) = 2.8 (large)."])
S.append(Paragraph("Metrics", H2))
S += bl(["<b>Mask IoU</b> = overlap of predicted and true mask / their union. <b>Dice</b> = 2 × overlap / (sum of both areas). Dice = 2·IoU/(1 + IoU), so Dice is always ≥ IoU.",
         "<b>Mask mAP@50 and @50-95:</b> same procedure as box mAP, but 'IoU' is the <b>mask</b> IoU.",
         "Published COCO val mask mAP@50-95: n 30.5, s 36.8, m 40.8, l 42.6 <i>(from Ultralytics docs; verify)</i>.",
         "Not used here: pixel accuracy and mIoU (for semantic segmentation), boundary F-score (edge quality)."])
S += example("IoU and Dice", [
    "Predicted mask 50 pixels, true mask 70 pixels, overlap 40. Union = 50 + 70 - 40 = 80. <b>IoU = 40/80 = 0.50.</b>",
    "<b>Dice = 2×40/(50+70) = 80/120 = 0.667.</b> Check: 2×0.5/(1+0.5) = 0.667."])
S.append(Paragraph("What our dashboard shows (descriptive, not accuracy)", H2))
S += bl(["Instances per class, average confidence, mean and max <b>area share</b> = mask pixels of that class / frame pixels × 100, per frame, plotted over time.",
         "These describe the output. Without ground-truth masks we cannot report mask mAP or IoU."])

# ======================================================= 6. tracking
H("6. Tracking: ByteTrack and BoT-SORT")
S.append(Paragraph("What problem it solves", H2))
S += bl(["Detection treats every frame alone. Counting unique objects or measuring time on screen needs the <b>same object to keep the same ID</b> across frames.",
         "Tracking-by-detection: detect every frame, then <b>associate</b> new detections with existing tracks.",
         "Trackers are not neural networks that we train. They are algorithms (Kalman filter + matching). Only the detector has a loss."])
S += fig("fig_bytetrack.png", 5.4 / 11)
S.append(Paragraph("Building block 1: the Kalman filter (motion prediction)", H2))
S += bl(["State = box centre, box size and the velocity of each (8 numbers). Assumes roughly constant velocity between frames.",
         "<b>Predict:</b> push the state forward one frame; uncertainty P grows (we know less after moving on).",
         "<b>Update:</b> when a measurement (matched box) arrives, correct the prediction. The <b>Kalman gain K</b> decides how much to trust the measurement vs the prediction: "
         "high measurement noise → small K, trust the prediction; high prediction uncertainty → large K, trust the measurement.",
         "Why it helps: the prediction tells where to look for the object next frame, and bridges frames where the detector misses it."])
S += formula(["Predict:   x' = F x          P' = F P Fᵀ + Q",
              "Update:    K = P' Hᵀ ( H P' Hᵀ + R )⁻¹      x = x' + K ( z - H x' )      P = ( I - K H ) P'"])
S += example("1-D Kalman update", [
    "Predicted position x' = 10, uncertainty P' = 4. Measurement z = 12 with noise R = 1 (H = 1).",
    "K = 4/(4 + 1) = 0.8. New position = 10 + 0.8×(12 - 10) = <b>11.6</b>. New uncertainty = (1 - 0.8)×4 = <b>0.8</b>. The detector is trusted more than the prediction, and the estimate is more certain than either."])
S.append(Paragraph("Building block 2: matching with IoU and the Hungarian algorithm", H2))
S += bl(["Build a cost matrix: rows = predicted track boxes, columns = detections, cost = 1 - IoU.",
         "The <b>Hungarian algorithm</b> finds the one-to-one assignment with the lowest <b>total</b> cost (better than greedy picking, which can steal a good match from another track).",
         "Matches with cost above a threshold are rejected (the boxes do not overlap enough)."])
S += example("assignment", [
    "Cost matrix (1 - IoU): T1: [D1 0.2, D2 0.9], T2: [D1 0.8, D2 0.3].",
    "Option A: T1-D1 + T2-D2 = 0.2 + 0.3 = <b>0.5</b>. Option B: T1-D2 + T2-D1 = 0.9 + 0.8 = 1.7. The algorithm picks A."])
S.append(Paragraph("ByteTrack (our default): the two-pass idea", H2))
S += bl(["Older trackers (SORT) throw away low-confidence detections. But a partly hidden person often gets a <b>low score</b>, not zero.",
         "<b>Pass 1:</b> match all tracks with high-score detections. <b>Pass 2:</b> match the <b>unmatched tracks</b> with low-score detections. This recovers occluded objects without creating false tracks.",
         "Unmatched high-score detections start <b>new tracks</b>. Unmatched tracks are kept 'lost' for <b>track_buffer</b> frames (default about 30; check bytetrack.yaml in your Ultralytics install), then deleted.",
         "No appearance (ReID) network, so it is very fast. Weakness: two similar objects that cross or a long occlusion can swap IDs.",
         "Ultralytics call: model.track with persist=True (keep tracker state between frames) and the tracker config file."])
S.append(Paragraph("BoT-SORT (option) and the older trackers", H2))
S += tbl([
    ["Tracker", "Motion", "Appearance", "Distinctive idea", "Trade-off"],
    ["SORT", "Kalman", "No", "Kalman + Hungarian on IoU", "Fast; many ID switches"],
    ["DeepSORT", "Kalman", "Yes: CNN embedding (128-d)", "Cosine appearance distance, matching cascade, gating", "Fewer switches; slower"],
    ["ByteTrack", "Kalman", "No", "Uses low-score boxes in a second pass", "Fast and strong; can swap IDs after long occlusion"],
    ["BoT-SORT", "Kalman", "Optional ReID", "Camera-motion compensation (aligns frames) + IoU/ReID fusion", "Best when the camera moves; slightly slower"],
    ["OC-SORT", "Kalman, observation-centric", "No", "Re-updates using observations after occlusion, handles non-linear motion", "Good for crowds and dance-like motion"],
], [2.0 * cm, 2.4 * cm, 3.0 * cm, 5.4 * cm, FW - 12.8 * cm])
S.append(Paragraph("Metrics for tracking (need ground truth in MOT format)", H2))
S += bl(["Ground truth = for every frame, the true boxes with identity numbers. The tracker output is compared by matching boxes with IoU ≥ 0.5.",
         "<b>MOTA</b> counts three error types: FN (missed objects), FP (false boxes) and IDSW (ID switches). Dominated by detection quality; can be negative.",
         "<b>MOTP</b> = average localisation quality (overlap) of matched boxes.",
         "<b>IDF1</b>: match whole <b>trajectories</b> to true identities over the entire video, then compute an F1 over identity-correct detections. Focuses on 'does an object keep one ID?'",
         "<b>HOTA</b> = sqrt(DetA × AssA), averaged over IoU thresholds 0.05 to 0.95. <b>DetA</b> = detection accuracy = TP/(TP+FN+FP). <b>AssA</b> = association accuracy: for every matched detection, how well its predicted track and true track agree over time. "
         "It balances detecting well and associating well, and avoids MOTA's detection bias.",
         "Also: <b>ID switches</b>, <b>fragmentations</b>, <b>MT / ML</b> (mostly tracked / mostly lost trajectories ≥ 80% / ≤ 20% covered), FPS.",
         "Tools: py-motmetrics (MOTA, IDF1) and TrackEval (HOTA). Standard benchmarks: MOT17, MOT20."])
S += formula(["MOTA = 1 - ( FN + FP + IDSW ) / GT",
              "IDP = IDTP/(IDTP + IDFP)     IDR = IDTP/(IDTP + IDFN)     IDF1 = 2·IDTP / ( 2·IDTP + IDFP + IDFN )",
              "DetA = TP / (TP + FN + FP)       HOTA_α = sqrt( DetA_α · AssA_α )       HOTA = mean over α"])
S += example("MOTA and IDF1", [
    "Ground truth has 100 object-frames. The tracker missed 8 (FN = 8), produced 5 false boxes (FP = 5) and switched an ID twice (IDSW = 2).",
    "<b>MOTA = 1 - (8 + 5 + 2)/100 = 0.85.</b>",
    "The tracker output 100 - 8 + 5 = 97 boxes. Suppose 84 of them carry the correct identity: IDTP = 84, IDFN = 100 - 84 = 16, IDFP = 97 - 84 = 13.",
    "<b>IDF1 = 2×84 / (2×84 + 13 + 16) = 168/197 = 0.853.</b> A tracker with identical detections but constant ID switches would keep MOTA high but score a much lower IDF1."])
S += example("HOTA", [
    "TP = 80, FN = 20, FP = 10: DetA = 80/110 = 0.727. Suppose AssA = 0.90. HOTA at this threshold = sqrt(0.727 × 0.90) = <b>0.809</b>. The final HOTA averages many thresholds."])
S.append(Paragraph("Metrics without ground truth (what our dashboard shows)", H2))
S += bl(["<b>ID churn</b> = unique IDs / maximum objects visible at once. Close to 1 = healthy. 3.0 means the tracker issued 3× more IDs than needed: it keeps losing and re-creating objects.",
         "<b>Short-track %</b> = tracks lasting under 1 second. High = flicker or false detections.",
         "<b>Mean track length</b> in seconds.",
         "These are <b>hints</b>. A video with many short-lived objects also has high churn, so they cannot replace IDF1 or HOTA."])
S.append(Paragraph("Analytics we compute from tracks", H2))
S += bl(["<b>Unique count per class</b> = number of distinct IDs (class = the most frequent class seen for that ID).",
         "<b>Dwell time</b> = (last frame - first frame + 1) / fps.",
         "<b>Line crossing</b>: a horizontal line at the middle of the frame. An object crosses when its centre's side of the line changes between two detections (up to down or down to up).",
         "<b>Heatmap</b>: add one count at each object centre in every frame, blur with a Gaussian, colour-map, overlay on the first frame.",
         "<b>Trajectory</b>: the sequence of centres of one ID."])
S += example("dwell time and churn", [
    "An object first seen at frame 50 and last seen at frame 199 in a 25 fps video: (199 - 50 + 1)/25 = <b>6.0 s</b>.",
    "A video where at most 4 objects appear together but 12 IDs were issued: churn = 12/4 = <b>3.0</b>."])

# ======================================================= 7. LLM + web app + full analysis
H("7. Remaining parts of the project")
S.append(Paragraph("LLM narrative (Groq, gpt-oss-20b)", H2))
S += bl(["Takes the per-keyframe captions (after a de-duplication step) and writes one connected narrative in English or Arabic.",
         "A decoder-only Transformer language model (mixture-of-experts: about 21B parameters in total, about 3.6B active per token <i>(from the model card; verify)</i>), "
         "trained with next-token cross-entropy. We only call it through an API.",
         "Risk to mention: an LLM can add details that are not in the captions. Arabic output depends on it, so Arabic quality is not evaluated by our metrics."])
S.append(Paragraph("Full analysis feature", H2))
S += bl(["One pass of YOLOv8-seg with tracking switched on: masks + IDs from the same detections, so they always agree.",
         "BLIP keyframe captions are drawn as a caption bar on each frame between their timestamps (English).",
         "One dashboard: tracking analytics + segmentation area statistics + caption list."])
S.append(Paragraph("Web app design", H2))
S += bl(["Flask serves four pages. An upload starts a <b>background thread</b>; the browser polls a job endpoint for progress. A lock allows one heavy job at a time.",
         "Output video is re-encoded to <b>H.264</b> with ffmpeg because browsers cannot play OpenCV's default mp4v.",
         "Each feature is a folder with numbered steps (like the original captioning project) and a pipeline file that runs them in order."])

# ======================================================= 8. settings
H("8. Every setting we chose, and why")
S += tbl([
    ["Setting", "Value", "Why / effect"],
    ["Frame sampling", "2 per second", "Enough to catch scene changes; far cheaper than all frames"],
    ["CLIP keyframe threshold", "cosine < 0.96", "Higher = more keyframes, lower = fewer; chosen empirically, not learned"],
    ["Reference frame for the rule", "last kept frame", "Prevents slow drift missing a big total change"],
    ["BLIP decoding", "beam search 5, max 30 new tokens, repetition penalty 1.2", "Stable, short, non-repetitive captions"],
    ["Detector size", "YOLOv8 s (n / m / l selectable)", "Balanced accuracy vs speed; n for speed, m / l for accuracy"],
    ["Detection confidence", "0.3", "Lower finds more objects but more false positives; higher the opposite"],
    ["Tracker", "ByteTrack (BoT-SORT selectable)", "Fast and strong; BoT-SORT if the camera moves"],
    ["Trail length", "40 positions", "Only for drawing"],
    ["Line crossing", "horizontal line at 50% height", "Simple default; the position is a parameter in the analytics code"],
    ["Mask opacity", "0.45", "Visualisation only"],
    ["Output encoding", "H.264, yuv420p", "Plays in every browser"],
], [4.0 * cm, 5.2 * cm, FW - 9.2 * cm])
S.append(Paragraph("Known limits (say them yourself first)", H2))
S += bl(["Everything is pretrained: only COCO's 80 classes are detected and segmented; no fine-tuning on our domain.",
         "No ground truth, so IDF1 / MOTA / HOTA / mask mAP / CIDEr are not measured. The app exports a MOT file; the evaluation helper computes MOTA and IDF1 with py-motmetrics; HOTA needs TrackEval.",
         "ByteTrack has no appearance model: ID swaps after long occlusions or crossings.",
         "The keyframe threshold and detection confidence are manual choices.",
         "Arabic captions come from an LLM working on English captions, so BLIP's errors pass through.",
         "The numbers a video gives (counts, dwell time) are only as good as the detector."])

# ======================================================= 9. Q&A
H("9. Question bank with short answers")
qa = [
 ("Why CLIP and not frame differencing or histograms?", "They react to pixels (light, noise). CLIP compares meaning, so a keyframe appears only when the scene content changes."),
 ("What does CLIP's loss do?", "Symmetric InfoNCE: in a batch, each image must pick its own text among N, and each text its own image. Matching pairs get high cosine, the rest low."),
 ("Why compare with the last kept frame?", "Comparing consecutive frames misses slow drift: each step looks similar but the sum is a new scene."),
 ("How does BLIP differ from CLIP?", "CLIP only embeds and compares (no text generation). BLIP adds a cross-attention decoder that generates text, plus ITC, ITM and LM losses."),
 ("What is cross-attention in BLIP?", "Queries from the text tokens, keys and values from the image patches. Each word chooses which image regions to read."),
 ("What is CapFilt?", "A captioner writes synthetic captions for web images; a filter removes noisy ones; the model is retrained on the cleaned set."),
 ("Beam search vs sampling?", "Beam keeps the 5 best partial sentences, deterministic and fluent. Sampling is random and more varied but less stable."),
 ("How do you evaluate captions without references?", "CLIPScore (image-text cosine), plus redundancy and unique-caption %. With references you would add CIDEr, BLEU, METEOR, ROUGE-L, SPICE."),
 ("Why YOLOv8?", "Single-stage, fast, anchor-free, good accuracy, sizes n to l, and the same family gives both detection and segmentation."),
 ("What is C2f?", "A split-transform-concatenate block: part of the features passes through several bottlenecks, all intermediate outputs are concatenated. Richer gradient flow at low cost."),
 ("What is anchor-free?", "No predefined box shapes. Each cell predicts distances to the four box edges directly."),
 ("What is DFL?", "A loss that trains each edge distance as a 16-bin probability distribution; the box edge is its expected value. It handles ambiguous boundaries better than one number."),
 ("Why CIoU?", "IoU gives zero gradient for non-overlapping boxes. CIoU adds centre distance and aspect ratio so the loss is always informative."),
 ("What is mAP@50-95?", "Mean over classes of AP, averaged over IoU thresholds 0.50 to 0.95. Rewards tight boxes, stricter than mAP@50."),
 ("How does the segmentation mask get built?", "32 prototype masks per image, 32 coefficients per object. Mask = sigmoid(coefficients · prototypes), cropped to the box."),
 ("Instance vs semantic segmentation?", "Semantic labels pixels by class only; instance also separates objects of the same class."),
 ("Why not SAM?", "Heavier, no class labels on its own. It is a good next step, for example to refine masks from YOLO boxes."),
 ("Why track?", "To count unique objects and measure time. Detection alone counts one person many times."),
 ("What does the Kalman filter do?", "Predicts where each object will be next frame and blends that with the measurement according to their uncertainties."),
 ("Why Hungarian instead of greedy?", "It minimises the total cost over all pairs; greedy can take a good match away from another track."),
 ("What makes ByteTrack special?", "A second matching pass with low-confidence detections, recovering partly hidden objects."),
 ("BoT-SORT vs ByteTrack?", "BoT-SORT adds camera-motion compensation and optional ReID. Better with moving cameras, a bit slower."),
 ("MOTA vs IDF1 vs HOTA?", "MOTA: error counting, detection-heavy. IDF1: identity consistency over time. HOTA: geometric mean of detection and association accuracy, balanced."),
 ("How would you evaluate your tracker?", "Label 20-30 seconds in MOT format (or use MOT17), run the pipeline, export tracks_mot.txt, compute MOTA / IDF1 with py-motmetrics and HOTA with TrackEval."),
 ("What if there are many ID switches?", "Raise detection quality (larger model), raise confidence or track buffer, use BoT-SORT with ReID, or DeepSORT-style appearance matching."),
 ("What would you do next?", "Label a short clip and report IDF1 / HOTA and mask IoU; fine-tune the detector on our domain; try SAM 2 for mask tracking; add reference captions for CIDEr."),
 ("Biggest weakness of the project?", "Pretrained only and no ground truth, so no accuracy numbers on our own videos yet. Say it first, then show the plan above."),
]
for q, a in qa:
    S.append(KeepTogether([Paragraph(f"<b>{q}</b>", P), Paragraph(a, B), Spacer(1, 2)]))

S.append(Spacer(1, 8))
S.append(Paragraph("Numbers to fill in after you run the app", H2))
S += tbl([["Item", "Your value"], ["Frames sampled / keyframes / compression ratio", ""], ["Mean CLIPScore", ""],
          ["Detector size, hardware, processing fps", ""], ["Unique objects / ID churn / short-track %", ""],
          ["IDF1 / MOTA / HOTA on a labelled clip (optional)", ""], ["Mask mAP or IoU on a labelled clip (optional)", ""]], [FW * 0.65, FW * 0.35])
doc.build(S)
print("ok")
