from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
                                PageBreak, Table, TableStyle, KeepTogether)

INK = colors.HexColor("#12202f"); MUTED = colors.HexColor("#5b6b7b")
ACC = {"cap": colors.HexColor("#e74c3c"), "trk": colors.HexColor("#3498db"), "seg": colors.HexColor("#2ecc71"), "gen": colors.HexColor("#12202f")}
PW, PH = landscape(A4)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=INK, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=INK, spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("b", fontName="Helvetica", fontSize=11, leading=14.5, textColor=INK, leftIndent=13, bulletIndent=2, spaceAfter=2.2)
P = ParagraphStyle("p", fontName="Helvetica", fontSize=11, leading=14.5, textColor=MUTED)
SM = ParagraphStyle("sm", parent=P, fontSize=9.5, leading=12.5)
CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=10.2, leading=13, textColor=INK)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold")


def bullets(items):
    return [Paragraph(t, B, bulletText="•") for t in items]


def two_col(left_title, left, right_title, right, color):
    def col(title, items):
        return [Paragraph(f'<font color="{color.hexval().replace("0x", "#")}">■</font> {title}', H2)] + bullets(items)
    t = Table([[col(left_title, left), col(right_title, right)]], colWidths=[(PW - 3.2 * cm) / 2] * 2)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
    return t


def table(rows, widths, head=True, color=INK):
    data = [[Paragraph(str(c), CELLB if (head and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d2db")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if head:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf0"))]
    t.setStyle(TableStyle(st))
    return t


def page_deco(c, doc):
    c.saveState()
    c.setFillColor(INK); c.rect(0, PH - 0.55 * cm, PW, 0.55 * cm, fill=1, stroke=0)
    c.setFont("Helvetica", 8.5); c.setFillColor(MUTED)
    c.drawString(1.6 * cm, 0.8 * cm, "Video Intelligence - interview notes")
    c.drawRightString(PW - 1.6 * cm, 0.8 * cm, f"{doc.page}")
    c.restoreState()


doc = BaseDocTemplate("Interview_Notes_Video_Intelligence.pdf", pagesize=landscape(A4),
                      leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                      title="Video Intelligence - interview notes", author="Maryam Sayed")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(1.6 * cm, 1.4 * cm, PW - 3.2 * cm, PH - 2.8 * cm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page_deco)])
S = []
FW = PW - 3.2 * cm


def diagram(path):
    return Image(path, width=FW, height=FW * 1725 / 3000)

# ---------------- 1. overview
S += [Paragraph("Video Intelligence", H1),
      Paragraph("Captioning, tracking and segmentation of video, one web app. Notes for tomorrow's discussion.", P), Spacer(1, 10)]
S += [two_col("The 30-second pitch", [
        "I extended my CLIP + BLIP video captioning project into a full video-understanding app.",
        "Three features, each its own page: <b>captioning</b>, <b>tracking</b>, <b>segmentation</b>.",
        "A fourth page merges all three into one annotated video with one dashboard.",
        "The web app only calls my original pipeline files. I did not rewrite them.",
        "Voice-over (TTS) stays in the repo but is not part of the vision work."],
      "What I built on top", [
        "<b>Tracking:</b> YOLOv8 + ByteTrack / BoT-SORT, unique counts, dwell time, line crossings, heatmap, trajectories.",
        "<b>Segmentation:</b> YOLOv8-seg masks, per-class area statistics over time.",
        "<b>Evaluation:</b> proxy metrics for captioning, MOT export for IDF1 / MOTA / HOTA, mask IoU / Dice.",
        "<b>Web app:</b> Flask, background jobs with a progress bar, Chart.js dashboards."], ACC["gen"]),
      Spacer(1, 8), Paragraph("How the pieces connect", H2)]
S.append(table([
    ["Page", "Input", "Models", "Output"],
    ["Captioning", "video", "CLIP ViT-B/32, BLIP base, Groq LLM", "EN + AR captioned videos, caption metrics"],
    ["Tracking", "video", "YOLOv8 + ByteTrack (or BoT-SORT)", "boxes + IDs + trails, dashboard, MOT file"],
    ["Segmentation", "video", "YOLOv8-seg", "masks, class area dashboard"],
    ["Full analysis", "video", "YOLOv8-seg + tracker + BLIP captions", "one video + one dashboard"],
], [3.6 * cm, 2.4 * cm, 9.2 * cm, FW - 15.2 * cm]))
S += [Spacer(1, 6), Paragraph("Where the code is (webapp/features)", H2)]
S.append(table([
    ["Folder", "Steps, in order"],
    ["captioning", "your root files 1_Frame_Extraction, 2_Keyframe_Selection_CLIP, 3_Captioning_BLIP, 4_Collect_Captions, 6_burn_captions; pipeline.py wires them; 5_evaluation.py"],
    ["tracking", "1_detection, 2_tracking, 3_analytics, 4_render, 5_evaluation, pipeline.py"],
    ["segmentation", "1_segmentation_model, 2_masks, 3_render, 4_statistics, 5_evaluation, pipeline.py"],
    ["full_analysis", "pipeline.py only: re-uses the steps above"],
], [3.6 * cm, FW - 3.6 * cm]))
S.append(PageBreak())

# ---------------- pipelines
def pipeline_pages(title, img, color, say, asked, trap):
    S.append(diagram(img)); S.append(PageBreak())
    S.append(Paragraph(title, H1))
    S.append(two_col("What I say", say, "If they push", asked, color))
    S.append(Spacer(1, 6))
    S.append(Paragraph("Honest limits (say them before they ask)", H2))
    S.extend(bullets(trap))
    S.append(PageBreak())

pipeline_pages("Captioning: say it in one breath", "pipeline_1_captioning.png", ACC["cap"], [
    "Sample 2 frames per second, embed each with CLIP.",
    "Keep a frame only if its cosine similarity to the last kept frame is below 0.96.",
    "BLIP captions each keyframe (beam search, 5 beams).",
    "An LLM merges the captions into a short story in English and Arabic.",
    "Captions are burned onto the video.",
    "<b>Why CLIP for selection:</b> it compares meaning, not pixels, so lighting changes do not trigger a keyframe.",
    "<b>Why BLIP:</b> strong pretrained captioner, no training needed."],
  [
    "<b>CLIP loss:</b> symmetric InfoNCE. Matching pairs pulled together, other pairs in the batch pushed apart.",
    "<b>BLIP losses:</b> ITC + ITM + LM. CapFilt cleans noisy web captions.",
    "<b>Threshold trade-off:</b> higher = more keyframes, more detail, slower.",
    "<b>Metric with no references:</b> CLIPScore = 2.5 * max(cos(image, caption), 0).",
    "<b>Metric with references:</b> CIDEr (main), BLEU-4, METEOR, ROUGE-L, SPICE.",
    "<b>Compression ratio</b> shows how much redundancy CLIP removed."],
  [
    "CLIPScore rates image-text agreement, not fluency or Arabic quality.",
    "Arabic is produced by the LLM from English captions, so it inherits BLIP's mistakes.",
    "BLIP base describes scenes, not identities or fine actions.",
    "I have no human reference captions, so CIDEr / BLEU are not reported."])

pipeline_pages("Tracking: say it in one breath", "pipeline_2_tracking.png", ACC["trk"], [
    "YOLOv8 finds objects in every frame.",
    "A Kalman filter predicts where each known object should be.",
    "Hungarian matching on IoU links detections to tracks, so each object keeps one ID.",
    "<b>ByteTrack</b> also uses low-confidence boxes in a second pass, so partly hidden objects keep their ID.",
    "<b>BoT-SORT</b> adds camera-motion compensation for moving cameras.",
    "Analytics on top: unique count per class, time on screen, centre-line crossings, heatmap, trajectories."],
  [
    "<b>SORT:</b> Kalman + Hungarian, no appearance. Fast, many ID switches.",
    "<b>DeepSORT:</b> adds a ReID appearance embedding. Fewer switches, slower.",
    "<b>ByteTrack:</b> SORT-style plus the low-score second pass.",
    "<b>MOTA:</b> detection-heavy (misses, false positives, ID switches).",
    "<b>IDF1:</b> how consistently an identity is kept over time.",
    "<b>HOTA:</b> sqrt(DetA * AssA), balances detection and association.",
    "Trackers have no loss: they are not trained. The detector is."],
  [
    "Without ground truth I cannot report MOTA / IDF1 / HOTA. The app exports MOT files so they can be computed.",
    "Dashboard 'tracker health' (ID churn, short tracks) are hints, not accuracy.",
    "Counts are only as good as the detector: unseen objects are not counted.",
    "ByteTrack has no appearance model, so identities can swap after long occlusions."])

pipeline_pages("Segmentation: say it in one breath", "pipeline_3_segmentation.png", ACC["seg"], [
    "YOLOv8-seg is single-stage: one pass gives boxes, classes and masks.",
    "The network outputs 32 prototype masks for the image and 32 coefficients per object.",
    "<b>Mask = sigmoid(coefficients x prototypes)</b>, cropped to the box.",
    "Same backbone family as my tracker's detector, so both run fast together.",
    "Dashboard shows instances per class and the share of the frame each class covers over time."],
  [
    "<b>Instance vs semantic:</b> instance separates two people standing together; semantic does not.",
    "<b>Why not Mask R-CNN:</b> two-stage, slower on video.",
    "<b>Why not SAM:</b> heavy, no class labels by itself. Good upgrade path.",
    "<b>Loss:</b> box (CIoU + DFL) + class (BCE) + mask (binary cross-entropy).",
    "<b>Metrics:</b> mask mAP@50 and 50-95, mask IoU, Dice."],
  [
    "Pretrained on COCO's 80 classes. Unusual objects are not segmented.",
    "No ground-truth masks for my videos, so no mask mAP / IoU numbers yet.",
    "Masks are per frame. Flicker is possible; tracking IDs help but do not smooth masks.",
    "Mask resolution is limited by the prototype size (1/4 of the input)."])

# ---------------- evaluation cheat sheet
S += [Paragraph("Evaluation cheat sheet", H1), Spacer(1, 4)]
S.append(table([
    ["Stage", "Metric", "In one line", "Needs"],
    ["CLIP keyframes", "Compression ratio", "frames sampled / keyframes kept", "nothing"],
    ["BLIP captions", "CLIPScore", "2.5 * max(cos(image, caption), 0)", "nothing"],
    ["BLIP captions", "CIDEr, BLEU-4, METEOR, ROUGE-L, SPICE", "overlap with human captions", "reference captions"],
    ["Detector", "mAP@50, mAP@50-95", "box quality across IoU thresholds", "labelled boxes"],
    ["Tracker", "MOTA", "1 - (FN + FP + ID switches) / GT", "MOT ground truth"],
    ["Tracker", "IDF1", "F1 of identity matches over time", "MOT ground truth"],
    ["Tracker", "HOTA", "sqrt(detection accuracy * association accuracy)", "MOT ground truth"],
    ["Segmentation", "Mask mAP, mask IoU, Dice", "overlap of predicted and true masks", "labelled masks"],
], [3.6 * cm, 6.4 * cm, 10.2 * cm, FW - 20.2 * cm]))
S += [Spacer(1, 8), Paragraph("How to get real tracking numbers", H2)]
S += bullets([
    "Download a short sequence with ground truth (for example from MOT17), or hand-label 20-30 seconds of your own video.",
    "Run the app on it and download <b>tracks_mot.txt</b>.",
    "<b>MOTA, IDF1, ID switches:</b> webapp/features/tracking/5_evaluation.py, evaluate_mot(gt_path, pred_path) (needs <i>pip install motmetrics</i>).",
    "<b>HOTA:</b> official TrackEval on the same two files."])
S.append(PageBreak())

# ---------------- numbers to fill + interview lines
S += [Paragraph("Numbers to fill in after your runs", H1)]
S.append(table([
    ["Item", "Your value", "Where to read it"],
    ["Frames sampled / keyframes / compression ratio", "", "Captioning page, metrics row"],
    ["Mean CLIPScore", "", "Captioning page"],
    ["Processing speed (fps), model size, hardware", "", "Tracking / Full analysis page"],
    ["Unique objects, ID churn, short tracks %", "", "Tracking dashboard"],
    ["IDF1 / MOTA / HOTA on a labelled clip (optional)", "", "motmetrics + TrackEval"],
    ["Mask mAP or IoU on a labelled clip (optional)", "", "model.val() or segmentation/5_evaluation.py"],
], [10 * cm, 5 * cm, FW - 15 * cm]))
S += [Spacer(1, 8), Paragraph("Likely questions, short answers", H2)]
S.append(table([
    ["Question", "Answer"],
    ["Why track at all?", "Counting unique objects needs identity across frames; detection alone counts the same person many times."],
    ["Why ByteTrack?", "Strong accuracy with no extra network. Its second pass recovers low-confidence, occluded objects."],
    ["When BoT-SORT?", "Moving camera, or when appearance features are needed."],
    ["What if IDs switch a lot?", "Raise confidence, increase track buffer, use BoT-SORT with ReID, or use a larger detector."],
    ["How do you know it works?", "Honest answer: proxy metrics on my videos, plus the MOT export to compute IDF1 / MOTA / HOTA against labels."],
    ["Biggest weakness?", "Everything is pretrained on COCO. No custom training or ground-truth evaluation yet."],
    ["Next step?", "Label a short clip, report IDF1 / HOTA; fine-tune the detector on my domain; try SAM 2 for mask tracking."],
], [5.2 * cm, FW - 5.2 * cm]))
doc.build(S)
print("pdf ok")
