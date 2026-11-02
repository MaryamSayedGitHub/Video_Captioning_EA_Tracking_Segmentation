import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

INK, MUTED, LINE = "#12202f", "#5b6b7b", "#c9d2db"
C = dict(red="#e74c3c", blue="#3498db", green="#2ecc71", violet="#9b59b6", amber="#f1c40f", grey="#ecf0f3", teal="#1abc9c", orange="#e67e22")


def fig(w=11, h=5.2):
    f = plt.figure(figsize=(w, h), dpi=170, facecolor="white")
    ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return f, ax


def bx(ax, x, y, w, h, text, fc="#ecf0f3", ec=INK, fs=9.5, bold=False, tc=INK, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def ar(ax, x1, y1, x2, y2, c=INK, ls="-", lw=1.6, ms=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, linestyle=ls))


def ttl(ax, t, h):
    ax.text(0.25, h - 0.3, t, fontsize=14, fontweight="bold", color=INK, va="center")


# ============================================================ U-Net
f, ax = fig(11, 5.6)
ttl(ax, "U-Net: encoder, bottleneck, decoder, and skip connections", 5.6)
levels = [(64, 3.5, 1.3), (128, 2.8, 1.1), (256, 2.2, 0.9), (512, 1.7, 0.7)]
xs_enc = [0.5, 1.7, 2.9, 4.1]
def stack(x, y, hh, c1, c2):
    ax.add_patch(Rectangle((x, y), 0.5, hh, fc=c1, ec=INK, lw=1.3)); ax.add_patch(Rectangle((x + 0.5, y), 0.22, hh, fc=c2, ec=INK, lw=1.3))
for (ch, y, hh), x in zip(levels, xs_enc):
    stack(x, y, hh, C["blue"], "#9ecbea"); ax.text(x + 0.36, y - 0.22, str(ch), ha="center", fontsize=8.5)
for i in range(3):
    ar(ax, xs_enc[i] + 0.78, levels[i][1] + 0.15, xs_enc[i + 1] - 0.03, levels[i + 1][1] + levels[i + 1][2] - 0.05, c=C["red"], lw=1.4, ms=11)
ax.text(1.5, 4.95, "max-pool 2x2 (halve H, W)", fontsize=8.5, color=C["red"], ha="left")
stack(5.2, 1.15, 0.5, C["violet"], "#d0a8e0"); ax.text(5.56, 0.93, "1024 (bottleneck)", ha="center", fontsize=8.5)
ar(ax, 4.9, 1.8, 5.2, 1.6, c=C["red"], lw=1.4, ms=11)
dec = [(512, 1.7, 0.7), (256, 2.2, 0.9), (128, 2.8, 1.1), (64, 3.5, 1.3)]
xs_dec = [6.3, 7.5, 8.7, 9.9]
for (ch, y, hh), x in zip(dec, xs_dec):
    stack(x, y, hh, C["green"], "#a7e8c4"); ax.text(x + 0.36, y - 0.22, str(ch), ha="center", fontsize=8.5)
for i in range(3):
    ar(ax, xs_dec[i] + 0.78, dec[i][1] + dec[i][2] - 0.05, xs_dec[i + 1] - 0.03, dec[i + 1][1] + 0.15, c=C["orange"], lw=1.4, ms=11)
ar(ax, 5.95, 1.5, 6.27, 1.85, c=C["orange"], lw=1.4, ms=11)
ax.text(8.2, 0.8, "up-conv 2x2 (double H, W)", fontsize=8.5, color=C["orange"], ha="center")
for xe, xd, (ch, y, hh) in zip(xs_enc, xs_dec[::-1], levels):
    ax.annotate("", xy=(xd - 0.02, y + hh * 0.55), xytext=(xe + 0.76, y + hh * 0.55), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, ls="--"))
bx(ax, 9.75, 4.95, 1.0, 0.4, "1x1 conv\n-> classes", C["amber"], fs=7)
bx(ax, 0.3, 0.05, 10.4, 0.65, "Dashed arrows = skip connections: copy encoder features, concatenate with decoder features.\nEach block = two 3x3 conv + ReLU. Each level down: half the size, double the channels.", "#fff6d6", fs=8.6)
f.savefig("fig_unet.png"); plt.close(f)

# ============================================================ U-Net variants (4 panels)
f, axs = plt.subplots(2, 2, figsize=(11, 7.4), dpi=170, facecolor="white")
f.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.04, hspace=0.12)
for a in axs.ravel():
    a.set_xlim(0, 10); a.set_ylim(0, 6.4); a.axis("off")


def node(a, x, y, t="", fc=C["blue"], r=0.34):
    a.add_patch(Circle((x, y), r, fc=fc, ec=INK, lw=1.2)); a.text(x, y, t, ha="center", va="center", fontsize=7.5, color="white" if fc not in (C["amber"], C["grey"]) else INK)


def lnk(a, p, q, c=INK, ls="-", lw=1.3):
    a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=lw, color=c, linestyle=ls, shrinkA=7, shrinkB=7))


def panel(a, title, sub):
    a.text(0.2, 6.15, title, fontsize=11.5, fontweight="bold", color=INK, va="center")
    a.text(0.2, 5.7, sub, fontsize=8.3, color=MUTED, va="center")


# (a) plain U-Net
a = axs[0, 0]; panel(a, "(a) U-Net (2015)", "skip = copy encoder map straight to decoder")
ys = [4.7, 3.5, 2.3, 1.1]
for i, y in enumerate(ys): node(a, 1.2 + i * 0.0 + 0.4 * 0, y, f"E{i+1}", C["blue"]); node(a, 8.6, y, f"D{i+1}", C["green"])
for i in range(3):
    lnk(a, (1.2, ys[i]), (1.2, ys[i + 1]), C["red"]); lnk(a, (8.6, ys[i + 1]), (8.6, ys[i]), C["orange"])
for i, y in enumerate(ys): lnk(a, (1.2, y), (8.6, y), INK, "--")
node(a, 4.9, 0.55, "B", C["violet"]); lnk(a, (1.2, ys[3]), (4.9, 0.55), C["red"]); lnk(a, (4.9, 0.55), (8.6, ys[3]), C["orange"])
a.text(4.9, 4.95, "concat", fontsize=8, color=INK, ha="center")

# (b) Attention U-Net
a = axs[0, 1]; panel(a, "(b) Attention U-Net (2018)", "skip passes through an attention gate (AG) first")
for i, y in enumerate(ys): node(a, 1.2, y, f"E{i+1}", C["blue"]); node(a, 8.6, y, f"D{i+1}", C["green"])
for i in range(3):
    lnk(a, (1.2, ys[i]), (1.2, ys[i + 1]), C["red"]); lnk(a, (8.6, ys[i + 1]), (8.6, ys[i]), C["orange"])
for i, y in enumerate(ys):
    node(a, 4.9, y, "AG", C["amber"], 0.38); lnk(a, (1.2, y), (4.9, y), INK, "--"); lnk(a, (4.9, y), (8.6, y), INK, "--")
    if i < 3: lnk(a, (8.6, ys[i + 1] + 0.0), (5.2, y - 0.25), C["violet"], ":", 1.0)
a.text(4.9, 5.05, "gate signal g from the coarser decoder level\nre-weights each pixel of the skip features", fontsize=7.8, color=C["violet"], ha="center")

# (c) U-Net++
a = axs[1, 0]; panel(a, "(c) U-Net++ (2018)", "nested dense skip pathways X(i,j), deep supervision")
cols = [1.3, 3.3, 5.3, 7.3]
rows = [4.6, 3.3, 2.0, 0.7]
for j in range(4):
    for i in range(4 - j):
        node(a, cols[j] + 0.0, rows[i], f"{i}.{j}", C["blue"] if j == 0 else C["green"], 0.3)
for j in range(1, 4):
    for i in range(4 - j):
        lnk(a, (cols[j - 1], rows[i]), (cols[j], rows[i]), INK, "--", 1.0)
        lnk(a, (cols[j - 1], rows[i + 1]), (cols[j], rows[i]), C["orange"], "-", 1.0)
for i in range(3): lnk(a, (cols[0], rows[i]), (cols[0], rows[i + 1]), C["red"], "-", 1.0)
a.text(8.0, 4.55, "outputs from\nX(0,1..4)\nsupervised\n(deep supervision)", fontsize=7.6, color=C["violet"], va="center")
a.text(5.0, 0.15, "every decoder node sees all earlier nodes of its level + upsampled node from below", fontsize=7.8, color=INK, ha="center")

# (d) TransUNet
a = axs[1, 1]; panel(a, "(d) TransUNet (2021)", "CNN features -> Transformer in the bottleneck -> U-Net decoder")
for i, y in enumerate(ys[:3]): node(a, 1.2, y, f"E{i+1}", C["blue"]); node(a, 8.6, y, f"D{i+1}", C["green"])
for i in range(2):
    lnk(a, (1.2, ys[i]), (1.2, ys[i + 1]), C["red"]); lnk(a, (8.6, ys[i + 1]), (8.6, ys[i]), C["orange"])
for i, y in enumerate(ys[:3]): lnk(a, (1.2, y), (8.6, y), INK, "--")
a.add_patch(FancyBboxPatch((3.3, 0.5), 3.2, 1.35, boxstyle="round,pad=0,rounding_size=0.06", fc=C["violet"], ec=INK, lw=1.4))
a.text(4.9, 1.2, "flatten to tokens\n12 x Transformer layers\n(global self-attention)", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
lnk(a, (1.2, ys[2]), (3.3, 1.3), C["red"]); lnk(a, (6.5, 1.3), (8.6, ys[2]), C["orange"])
a.text(4.9, 5.0, "CNN keeps local detail via skips;\nTransformer adds global context", fontsize=7.8, color=INK, ha="center")
f.savefig("fig_unet_variants.png"); plt.close(f)

# ============================================================ VAE
f, ax = fig(11, 5.2)
ttl(ax, "VAE: encode to a distribution, sample, decode", 5.2)
bx(ax, 0.3, 2.9, 1.3, 0.9, "Input x\n(image)", C["grey"])
bx(ax, 2.1, 2.7, 1.9, 1.3, "Encoder q(z|x)\nCNN", C["blue"], tc="white", bold=True)
bx(ax, 4.6, 3.55, 1.5, 0.6, "mean  mu", C["amber"]); bx(ax, 4.6, 2.55, 1.5, 0.6, "log-variance", C["amber"])
bx(ax, 6.6, 2.7, 2.0, 1.3, "z = mu + sigma * eps\neps ~ N(0, I)\n(reparameterisation)", "#fff6d6", fs=8.8)
bx(ax, 9.0, 2.7, 1.8, 1.3, "Decoder p(x|z)\nCNN", C["green"], tc="white", bold=True)
ar(ax, 1.6, 3.35, 2.1, 3.35); ar(ax, 4.0, 3.5, 4.6, 3.85); ar(ax, 4.0, 3.1, 4.6, 2.85)
ar(ax, 6.1, 3.85, 6.6, 3.6); ar(ax, 6.1, 2.85, 6.6, 3.1); ar(ax, 8.6, 3.35, 9.0, 3.35)
bx(ax, 6.9, 4.45, 1.4, 0.5, "eps (random)", C["grey"], fs=8.5); ar(ax, 7.6, 4.45, 7.6, 4.0, ls="--")
bx(ax, 9.0, 1.45, 1.8, 0.7, "Output x_hat", C["grey"]); ar(ax, 9.9, 2.7, 9.9, 2.15)
bx(ax, 0.3, 0.2, 5.0, 1.0, "Loss part 1: KL( q(z|x) || N(0, I) )\nkeeps the latent space smooth and sampleable", "#e8f4fc", fs=8.8)
bx(ax, 5.8, 0.2, 5.0, 1.0, "Loss part 2: reconstruction  ||x - x_hat||^2  (or BCE)\nmakes the decoded image look like the input", "#e9f9ef", fs=8.8)
bx(ax, 0.3, 1.45, 5.9, 0.75, "Why not sample z directly? Sampling blocks gradients.\nWriting z = mu + sigma*eps moves the randomness into eps, so gradients reach mu and sigma.", "#fff6d6", fs=8.5)
f.savefig("fig_vae.png"); plt.close(f)

# ============================================================ GAN
f, ax = fig(11, 5.0)
ttl(ax, "GAN: a forger (G) and a detective (D) trained against each other", 5.0)
bx(ax, 0.3, 3.1, 1.5, 0.9, "noise z\n~ N(0, I)", C["grey"])
bx(ax, 2.3, 2.9, 2.0, 1.3, "Generator G\n(upsampling\nnetwork)", C["red"], tc="white", bold=True)
bx(ax, 4.8, 3.1, 1.5, 0.9, "Fake image\nG(z)", C["grey"])
bx(ax, 4.8, 1.6, 1.5, 0.9, "Real image x\n(from data)", C["grey"])
bx(ax, 7.0, 2.5, 2.0, 1.5, "Discriminator D\n(classifier)", C["blue"], tc="white", bold=True)
bx(ax, 9.5, 2.9, 1.3, 0.9, "real or\nfake?\nD(.) in (0,1)", "#fff6d6", fs=8.5)
ar(ax, 1.8, 3.55, 2.3, 3.55); ar(ax, 4.3, 3.55, 4.8, 3.55); ar(ax, 6.3, 3.55, 7.0, 3.5); ar(ax, 6.3, 2.05, 7.0, 2.8); ar(ax, 9.0, 3.3, 9.5, 3.3)
bx(ax, 0.3, 0.2, 5.2, 1.0, "D update: raise D(real), lower D(fake)\n(plain binary classification)", "#e8f4fc", fs=9)
bx(ax, 5.8, 0.2, 5.0, 1.0, "G update: make D(fake) high\n(gradient flows back THROUGH D into G)", "#fdeaea", fs=9)
ax.text(0.3, 1.55, "Alternate: a few D steps, one G step.\nAt the ideal end, D outputs 0.5 everywhere\n(cannot tell real from fake).", fontsize=8.8, color=INK, va="center")
f.savefig("fig_gan.png"); plt.close(f)
print("a ok")
