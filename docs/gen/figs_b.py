import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("figs_a.py").read().split("# ============================================================ U-Net\n")[0])  # shared helpers

# ============================================================ GAN math plots
f, axs = plt.subplots(1, 3, figsize=(13, 3.9), dpi=170, facecolor="white")
f.subplots_adjust(wspace=0.32, left=0.05, right=0.99, top=0.9, bottom=0.3)
d = np.linspace(0.001, 0.999, 400)
ax = axs[0]
ax.plot(d, d, color=C["red"], lw=2.4, label="saturating: min log(1 - D(G(z)))")
ax.plot(d, 1 - d, color=C["green"], lw=2.4, label="non-saturating: min -log D(G(z))")
ax.set_xlabel("D(G(z))  (how real D thinks the fake is)", fontsize=9); ax.set_ylabel("|gradient| w.r.t. D's logit", fontsize=9)
ax.set_title("Generator gradient at the start of training", fontsize=10, fontweight="bold", loc="left")
ax.annotate("early training: D easily\nspots fakes (D~0.01)\nred gradient ~ 0.01\ngreen gradient ~ 0.99", xy=(0.01, 0.99), xytext=(0.28, 0.6), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK))
ax.legend(fontsize=7.5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.25)); ax.grid(alpha=.25)

ax = axs[1]
th = np.linspace(-2, 2, 401)
ax.plot(th, np.abs(th), color=C["blue"], lw=2.4, label="Wasserstein W(P0, P_theta) = |theta|")
jsd = np.where(np.abs(th) < 1e-9, 0, np.log(2)); ax.plot(th, jsd, color=C["red"], lw=2.4, label="JS divergence (constant ln 2 = 0.69)")
ax.set_xlabel("theta: distance between real and fake distributions", fontsize=9); ax.set_ylabel("value", fontsize=9)
ax.set_title("Why Wasserstein gives a usable gradient", fontsize=10, fontweight="bold", loc="left")
ax.legend(fontsize=7.5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.25)); ax.grid(alpha=.25)

ax = axs[2]
rng = np.random.default_rng(3)
ang = np.arange(8) * 2 * np.pi / 8; centers = np.c_[2 * np.cos(ang), 2 * np.sin(ang)]
real = np.vstack([c + 0.12 * rng.standard_normal((60, 2)) for c in centers])
coll = centers[0] + 0.12 * rng.standard_normal((480, 2))
part = np.vstack([centers[i] + 0.12 * rng.standard_normal((120, 2)) for i in (0, 1, 2, 3)])
ax.scatter(real[:, 0], real[:, 1], s=4, c=C["blue"], label="real data: 8 modes")
ax.scatter(coll[:, 0] + 0.0, coll[:, 1] + 0.0, s=4, c=C["red"], alpha=.7, label="full collapse: 1 mode")
ax.scatter(part[:, 0] * 0.0 + part[:, 0] + 0.0, part[:, 1] - 0.0, s=0, c="none")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Mode collapse (schematic: 8 Gaussians)", fontsize=10, fontweight="bold", loc="left")
ax.legend(fontsize=7.5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.02), ncol=1)
f.savefig("fig_gan_plots.png"); plt.close(f)

# ============================================================ diffusion forward demo
n = 96
yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
img = 0.25 + 0.5 * xx
r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
img = np.where(r2 < 0.09, 0.9, img)
img = np.where((np.abs(xx - 0.5) < 0.04) & (np.abs(yy - 0.5) < 0.3), 0.15, img)
x0 = img * 2 - 1
rng = np.random.default_rng(0); eps = rng.standard_normal(x0.shape)
abars = [1.0, 0.8, 0.5, 0.2, 0.05, 0.0]
f, axs = plt.subplots(1, 6, figsize=(13, 2.7), dpi=170, facecolor="white")
for a, ab in zip(axs, abars):
    xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
    a.imshow(np.clip((xt + 1) / 2, 0, 1), cmap="gray", vmin=0, vmax=1); a.axis("off")
    a.set_title(f"abar = {ab:g}\n" + ("clean x0" if ab == 1 else "pure noise" if ab == 0 else f"{np.sqrt(ab):.2f}*x0 + {np.sqrt(1-ab):.2f}*eps"), fontsize=8)
f.suptitle("Forward diffusion: x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps   (real computation on a toy image)", fontsize=10, fontweight="bold", x=0.01, ha="left")
f.subplots_adjust(top=0.68, bottom=0.02, wspace=0.05, left=0.01, right=0.99)
f.savefig("fig_diffusion_forward.png"); plt.close(f)

# ============================================================ Stable Diffusion pipeline
f, ax = fig(11, 6.2)
ttl(ax, "Stable Diffusion v1.x: text-to-image at inference", 6.2)
bx(ax, 0.3, 4.2, 1.9, 0.9, "Prompt\n'a red fox in snow'", C["grey"], fs=8.5)
bx(ax, 2.7, 4.1, 2.3, 1.1, "CLIP ViT-L/14\ntext encoder (frozen)\n123M params", C["blue"], tc="white", bold=True, fs=8.5)
bx(ax, 5.5, 4.2, 2.1, 0.9, "77 tokens x 768\n(text embeddings)", "#fff6d6", fs=8.5)
ar(ax, 2.2, 4.65, 2.7, 4.65); ar(ax, 5.0, 4.65, 5.5, 4.65)
bx(ax, 0.3, 2.4, 1.9, 0.9, "Random noise\nlatent 4 x 64 x 64", C["grey"], fs=8.5)
bx(ax, 2.7, 2.05, 3.3, 1.6, "U-Net (860M params)\npredicts the noise in the latent\nrepeated T = 20-50 steps\n(scheduler: DDIM, Euler, DPM++ ...)", C["red"], tc="white", bold=True, fs=8.5)
ar(ax, 2.2, 2.85, 2.7, 2.85)
ar(ax, 6.55, 4.2, 4.6, 3.65, c=C["blue"]); ax.text(5.7, 3.85, "cross-attention", fontsize=8, color=C["blue"])
ar(ax, 4.35, 2.05, 4.35, 1.65, c=INK); ar(ax, 4.35, 1.65, 1.2, 1.65, c=INK, ls="--") if False else None
ax.annotate("", xy=(1.25, 2.4), xytext=(1.25, 1.75), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4)); ax.plot([1.25, 4.35], [1.75, 1.75], color=INK, lw=1.4)
ax.text(2.3, 1.55, "loop: subtract a bit of predicted noise, repeat", fontsize=8, color=INK)
bx(ax, 6.6, 2.4, 1.9, 0.9, "Clean latent\n4 x 64 x 64", C["grey"], fs=8.5); ar(ax, 6.0, 2.85, 6.6, 2.85)
bx(ax, 8.9, 2.2, 1.9, 1.3, "VAE decoder\n(8x upsampling)", C["green"], tc="white", bold=True, fs=8.5); ar(ax, 8.5, 2.85, 8.9, 2.85)
bx(ax, 8.9, 0.5, 1.9, 0.9, "Image\n3 x 512 x 512", C["grey"], fs=8.5); ar(ax, 9.85, 2.2, 9.85, 1.4)
bx(ax, 6.2, 0.3, 2.5, 1.2, "Training (not at inference):\nimage -> VAE encoder -> latent\n-> add noise -> U-Net must\npredict that noise (MSE)", "#e8f4fc", fs=7.8)
bx(ax, 0.3, 0.3, 5.5, 0.9, "Classifier-free guidance: run U-Net twice per step (with and without the prompt)\nnoise = uncond + s * (cond - uncond),  s ~ 7.5 pushes the image toward the prompt", "#fdeaea", fs=8)
f.savefig("fig_sd_pipeline.png"); plt.close(f)

# ============================================================ SD U-Net internals
f, ax = fig(11, 6.4)
ttl(ax, "Inside the Stable Diffusion U-Net (v1.x)", 6.4)
res = [("64x64\n320 ch", 4.6), ("32x32\n640 ch", 3.7), ("16x16\n1280 ch", 2.8), ("8x8\n1280 ch", 1.9)]
for i, (t, y) in enumerate(res):
    bx(ax, 0.3, y, 1.9, 0.75, t, C["blue"], tc="white", bold=True, fs=8.2)
    bx(ax, 2.4, y, 1.1, 0.75, "ResBlock\n+ Attn" if i < 3 else "ResBlock\n(no attn)", C["grey"], fs=7.8)
    bx(ax, 6.4, y, 1.1, 0.75, "ResBlock\n+ Attn" if i < 3 else "ResBlock\n(no attn)", C["grey"], fs=7.8)
    bx(ax, 7.7, y, 1.9, 0.75, t, C["green"], tc="white", bold=True, fs=8.2)
    ax.annotate("", xy=(6.35, y + 0.37), xytext=(3.55, y + 0.37), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, ls="--"))
for i in range(3):
    ar(ax, 1.25, res[i][1], 1.25, res[i + 1][1] + 0.75, c=C["red"], lw=1.2, ms=10)
    ar(ax, 8.65, res[i + 1][1] + 0.75, 8.65, res[i][1], c=C["orange"], lw=1.2, ms=10)
bx(ax, 3.2, 0.95, 3.0, 0.65, "Middle block: ResBlock + Attn + ResBlock", C["violet"], tc="white", bold=True, fs=8)
ax.text(1.25, 5.55, "down (encoder)", fontsize=8.5, color=C["red"], ha="center"); ax.text(8.65, 5.55, "up (decoder)", fontsize=8.5, color=C["orange"], ha="center")
bx(ax, 9.85, 3.0, 1.05, 2.0, "Timestep t\nsinusoidal\n-> MLP\nadded in\nevery\nResBlock", C["amber"], fs=7.5)
bx(ax, 0.3, 0.1, 10.4, 0.65, "Attention block (SpatialTransformer) at each level: (1) self-attention over image positions, (2) CROSS-attention: queries = image, keys/values = the 77x768 text embeddings, (3) feed-forward.", "#fff6d6", fs=8.2)
f.savefig("fig_sd_unet.png"); plt.close(f)
print("b ok")
