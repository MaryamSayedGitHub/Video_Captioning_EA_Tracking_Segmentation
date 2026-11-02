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
    c.drawString(1.7 * cm, 0.8 * cm, "Generative models and U-Net: study notes")
    c.drawRightString(PW - 1.7 * cm, 0.8 * cm, str(doc.page))
    c.restoreState()



def make_doc(name):
    doc = BaseDocTemplate(name, pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.3 * cm, bottomMargin=1.4 * cm,
                          title="U-Net, VAE, GAN and Stable Diffusion: study notes", author="Maryam Sayed")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(1.7 * cm, 1.4 * cm, FW, PH - 2.7 * cm, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=deco)])
    return doc


S = []
def H(t): S.append(PageBreak()); S.append(Paragraph(t, H1))
def sec(t): S.append(Paragraph(t, H2))
def sub(t): S.append(Paragraph(t, H3))
def para(t): S.append(Paragraph(t, P))
def bullets(items): S.extend(bl(items))
def form(lines): S.extend(formula(lines))
def ex(title, lines): S.extend(example(title, lines))
def table(rows, widths): S.extend(tbl(rows, widths))
def figure(path, ratio): S.extend(fig(path, ratio))
def link(url, text=None): return f'<a href="{url}" color="#1a5fb4">{text or url}</a>'
