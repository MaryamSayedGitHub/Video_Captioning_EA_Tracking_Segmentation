from kit import *

# ---------------------------------------------------------------- cover / map
S.append(Paragraph("U-Net, VAE, GAN and Stable Diffusion", H1))
para("Study notes for the interview: how each model works, its losses, how to evaluate it, what goes wrong and how to fix it. Concepts and hand calculations, no code.")
sec("How to read these notes")
bullets(["Every model follows the same order: <b>the problem it solves → intuition → architecture → loss → evaluation → failure modes and fixes → worked numbers</b>.",
         "Chapter 1 U-Net and all its variants. Chapter 2 VAE. Chapter 3 GAN (loss types, evaluation, mode collapse). Chapter 4 Stable Diffusion. Chapter 5 comparison and question bank. Chapter 6 sources.",
         "Numbers marked <i>(from papers / docs)</i> are published values. Verify them if you quote them exactly.",
         "<b>About images:</b> my work environment cannot download pictures from the internet, so the figures here are drawn by me. "
         "Chapter 6 lists direct links to the original paper figures so you can open them next to these notes."])
sec("The big map")
table([
    ["Model", "Task", "Core idea", "Loss", "How to evaluate"],
    ["U-Net", "Segmentation (also the denoiser inside diffusion)", "Encoder-decoder with skip connections: context from the encoder, precise location from the skips", "Cross-entropy, Dice, IoU, focal, Tversky, combinations", "Dice, IoU, sensitivity / specificity, Hausdorff distance"],
    ["VAE", "Generation + learned latent space", "Encode to a distribution, sample, decode; keep the latent space smooth with a KL term", "Reconstruction + KL (the negative ELBO)", "ELBO / bits per dim, reconstruction PSNR / SSIM / LPIPS, FID of samples"],
    ["GAN", "Sharp image generation", "Two networks compete: generator forges, discriminator detects", "Minimax, non-saturating, LSGAN, hinge, Wasserstein (+GP), R1", "FID, IS, KID, precision / recall, density / coverage"],
    ["Stable Diffusion", "Text-to-image", "Learn to remove noise in a compressed latent space, guided by text", "MSE on predicted noise (epsilon), in latent space", "FID, CLIP score, human preference, prompt-adherence benchmarks"],
], [2.0 * cm, 2.6 * cm, 4.6 * cm, 3.3 * cm, FW - 12.5 * cm])
sec("How they connect")
bullets(["<b>U-Net inside Stable Diffusion:</b> the denoiser of Stable Diffusion is a U-Net with time embedding and cross-attention.",
         "<b>U-Net inside Pix2Pix (a GAN):</b> the generator is a U-Net, the discriminator judges 70×70 patches.",
         "<b>VAE inside Stable Diffusion:</b> a KL-regularised autoencoder compresses images 8× in each side so that diffusion runs on small latents.",
         "<b>GAN ideas inside the VAE of Stable Diffusion:</b> the autoencoder is trained with an adversarial patch loss to avoid blur."])

# ================================================================ 1 U-Net
H("1. U-Net and its variants")
sec("1.1 The problem it solves")
bullets(["<b>Classification</b> gives one label per image. <b>Segmentation</b> gives one label per pixel. The output has the same height and width as the input.",
         "Two needs pull in opposite directions: <b>context</b> (what is this object? needs a large field of view, so deep and downsampled layers) and <b>precise location</b> (where exactly is the boundary? needs high resolution).",
         "Earlier approach: run a classifier on patches around every pixel. Slow and wasteful because neighbouring patches overlap.",
         "FCN (2014) made the network fully convolutional but its output was coarse. <b>U-Net (2015, Ronneberger, Fischer, Brox)</b> fixed this with a symmetric decoder and skip connections.",
         "Original goal: biomedical images, where labelled data is scarce (the paper used only 30 training images for an EM segmentation challenge) and boundaries between touching cells matter."])
sec("1.2 Intuition")
para("Think of reading a map. The encoder zooms <b>out</b> step by step: you lose detail but you see the whole city and understand what each area is. The decoder zooms <b>back in</b>, but it would "
     "have forgotten the street details. So at every zoom level it is handed the <b>original sharp tile</b> from the way down (the skip connection). The decoder combines 'what it is' (from below) with 'exactly where' (from the skip).")
sec("1.3 Architecture of the original U-Net")
figure("fig_unet.png", 5.6 / 11)
bullets(["<b>Contracting path (encoder):</b> repeat 4 times: two 3×3 convolutions + ReLU, then 2×2 max-pooling (stride 2). Channels double each time: 64, 128, 256, 512.",
         "<b>Bottleneck:</b> two 3×3 convolutions with 1024 channels. The receptive field is largest here: each pixel 'sees' a large part of the image.",
         "<b>Expanding path (decoder):</b> repeat 4 times: 2×2 up-convolution (transposed convolution) that doubles height and width and halves channels, <b>concatenate</b> with the matching encoder feature map, then two 3×3 convolutions + ReLU.",
         "<b>Output:</b> a 1×1 convolution maps the 64 channels to one score per class per pixel (softmax or sigmoid).",
         "<b>Original details <i>(from the paper)</i>:</b> 23 convolutional layers; 3×3 convolutions were <b>unpadded</b>, so a 572×572 input gave a 388×388 output (borders are lost); large images are segmented with an <b>overlap-tile</b> strategy with mirrored borders. "
         "Modern implementations use 'same' padding so input and output sizes match. About 31 million parameters for the standard 64-base version (commonly cited).",
         "<b>Why concatenate and not add?</b> Concatenation keeps both sources separate and lets the next convolution learn how to mix them. Residual-style networks add instead (ResNet-like skips). They are different ideas: U-Net skips cross between encoder and decoder, ResNet skips jump inside a block."])
sec("1.4 What makes U-Net train well on small data")
bullets(["<b>Heavy augmentation:</b> random elastic deformations (the paper's key trick), rotations, shifts, intensity changes. Teaches invariance with few images.",
         "<b>Skip connections</b> give a short gradient path to early layers and give the decoder fine detail, so less must be learned from scratch.",
         "<b>Weighted loss</b> forces the network to separate touching cells (explained next).",
         "<b>Weight initialisation</b> with Gaussian of std sqrt(2/N) (He-style) keeps activations balanced, which matters with no batch normalisation in the original (modern versions add batch or instance norm)."])

sec("1.5 Loss functions for segmentation")
para("Notation: p = predicted probability of foreground for a pixel, g = ground truth (0 or 1), summed over all pixels. TP, FP, FN are true positives, false positives and false negatives counted on pixels.")
sub("Cross-entropy and the original weighted version")
form(["BCE = - (1/N) Σ [ g·log p + (1 - g)·log(1 - p) ]",
      "Original U-Net: E = Σ_x w(x) · log p_l(x)(x)      (softmax cross-entropy weighted per pixel)",
      "w(x) = w_c(x) + w0 · exp( - (d1(x) + d2(x))² / (2σ²) )       w0 = 10, σ ≈ 5 pixels"])
bullets(["w_c balances class frequencies. d1 and d2 are the distances to the nearest and second-nearest cell border. The exponential term is large <b>in the thin gap between two touching cells</b>, so errors there are punished heavily.",
         "Plain cross-entropy treats every pixel equally. With 2% foreground a network that predicts 'all background' gets 98% accuracy and a low loss: a <b>class imbalance</b> problem."])
ex("border weight", ["A pixel lies 3 px from one cell and 4 px from another. d1 + d2 = 7, so exponent = -49/(2×25) = -0.98, exp = 0.375. With w_c = 1: <b>w = 1 + 10×0.375 = 4.75</b>. A pixel far from borders gets w ≈ 1. The gap pixel counts almost 5× more."])
sub("Dice loss (overlap-based, handles imbalance)")
form(["Dice = 2 Σ p·g / ( Σ p + Σ g )          Dice loss = 1 - Dice",
      "Soft Dice uses the probabilities p directly, so it is differentiable. IoU (Jaccard) loss = 1 - Σ p·g / ( Σ p + Σ g - Σ p·g )"])
bullets(["It measures <b>overlap relative to object size</b>, so a small object matters as much as a big one. No class-weight tuning.",
         "Weakness: gradients can be unstable when the target is tiny or empty (denominator near 0). It is usually combined with cross-entropy: <b>BCE + Dice</b> gives stable gradients from BCE and overlap focus from Dice.",
         "Introduced for segmentation by V-Net (2016); nnU-Net uses Dice + cross-entropy by default."])
ex("BCE + Dice on 4 pixels", ["Predictions p = [0.9, 0.8, 0.2, 0.1], truth g = [1, 1, 0, 0].",
    "BCE = -(ln 0.9 + ln 0.8 + ln 0.8 + ln 0.9)/4 = (0.105 + 0.223 + 0.223 + 0.105)/4 = <b>0.164</b>.",
    "Dice = 2×(0.9 + 0.8)/(2.0 + 2) = 3.4/4 = 0.85, so Dice loss = <b>0.15</b>. Combined = 0.314."])
sub("Focal loss (focus on hard pixels)")
form(["FL = - α_t · (1 - p_t)^γ · log p_t        p_t = p if g = 1 else 1 - p        γ = 2 is typical"])
ex("focal factor", ["Easy pixel p_t = 0.9: CE = 0.105, factor (0.1)² = 0.01, <b>FL = 0.001</b>. Hard pixel p_t = 0.1: CE = 2.303, factor (0.9)² = 0.81, <b>FL = 1.865</b>.",
    "Easy pixels are almost ignored (100× smaller) while hard pixels keep most of their loss. Good when most pixels are easy background."])
sub("Tversky loss (control false positives vs false negatives)")
form(["Tversky = TP / ( TP + α·FN + β·FP )        α = β = 0.5 gives Dice;  α > β punishes missed pixels more"])
para("In medical work missing a tumour (FN) is often worse than a false alarm, so α = 0.7, β = 0.3 is common. Other losses worth naming: <b>Lovász-softmax</b> (directly optimises IoU), "
     "<b>boundary loss</b> (distance-based, helps thin structures), <b>combo loss</b> (weighted BCE + Dice), deep supervision (loss at several decoder levels).")
table([
    ["Loss", "Best when", "Watch out for"],
    ["Cross-entropy", "Balanced classes; stable baseline", "Imbalance: predicts background"],
    ["Weighted CE", "Touching objects; known class frequencies", "Needs weight maps / tuning"],
    ["Dice / IoU", "Small foreground, imbalance", "Noisy gradients; use with CE"],
    ["Focal", "Many easy pixels, few hard", "Extra hyper-parameters γ, α"],
    ["Tversky", "FN and FP have different cost", "Tune α, β"],
    ["Boundary / Lovász", "Thin structures, direct metric optimisation", "Harder to train from scratch"],
], [3.2 * cm, 7.2 * cm, FW - 10.4 * cm])

sec("1.6 Evaluation metrics")
form(["Dice = 2TP / (2TP + FP + FN)               IoU = TP / (TP + FP + FN)               Dice = 2·IoU / (1 + IoU)",
      "Precision = TP/(TP+FP)    Recall (sensitivity) = TP/(TP+FN)    Specificity = TN/(TN+FP)    Pixel accuracy = (TP+TN)/all"])
bullets(["<b>Dice and IoU</b> measure overlap. Dice ≥ IoU always. Report mean over images and over classes (and say how you average).",
         "<b>Pixel accuracy is misleading</b> with imbalance (see example).",
         "<b>Hausdorff distance (HD, HD95):</b> the largest distance from a point on one boundary to the nearest point on the other. HD95 uses the 95th percentile so a single outlier does not dominate. Measures boundary errors that overlap metrics hide.",
         "<b>Average surface distance (ASSD):</b> mean boundary distance. <b>Volume difference</b> matters in clinical use (tumour size).",
         "<b>Per-case vs pooled:</b> compute Dice per image then average. Pooling all pixels hides failures on small cases.",
         "<b>Instance-level</b> metrics (if separate objects matter): panoptic quality, F1 at IoU 0.5 for each object."])
ex("why accuracy misleads", ["1000 pixels. TP = 80, FP = 20, FN = 10, TN = 890.",
    "Pixel accuracy = (80 + 890)/1000 = <b>0.97</b> looks great. Dice = 160/(160 + 20 + 10) = <b>0.842</b>. IoU = 80/110 = <b>0.727</b>. Precision = 0.80, recall = 0.889, specificity = 0.978.",
    "The foreground is only 9% of the pixels, so accuracy is dominated by easy background."])

sec("1.7 The U-Net family: what each variant changes")
table([
    ["Variant (year)", "What changes", "Why", "Cost / trade-off"],
    ["U-Net (2015)", "Baseline: conv encoder, conv decoder, concat skips", "Context + precise location", "Skips pass raw low-level features, including noise"],
    ["3D U-Net (2016)", "All operations in 3D (3D conv, pooling, up-conv)", "Volumes: CT, MRI", "Much more memory; patches or 2.5D often used"],
    ["V-Net (2016)", "3D, residual blocks, Dice loss", "Volumetric segmentation with imbalance", "Memory heavy"],
    ["Residual U-Net (ResUNet)", "ResNet-style residual units inside blocks", "Deeper nets train more easily", "Slightly more compute"],
    ["R2U-Net (2018)", "Recurrent + residual conv units", "Better feature accumulation", "More compute, modest gains"],
    ["Dense U-Net", "DenseNet blocks (each layer sees all earlier maps)", "Feature reuse, fewer parameters per feature", "High memory for concatenations"],
    ["Attention U-Net (2018)", "Attention gates on the skip paths", "Suppress irrelevant regions, focus on target (pancreas)", "Small extra cost; gains largest for small organs"],
    ["U-Net++ (2018)", "Nested, dense skip pathways + deep supervision", "Close the semantic gap between encoder and decoder maps; prune at inference", "More parameters and training memory"],
    ["UNet 3+ (2020)", "Full-scale skip connections (every decoder level sees all scales)", "Use fine and coarse detail together", "Fewer parameters than U-Net++, more complex wiring"],
    ["nnU-Net (2018 / 2021)", "Not a new block: a <b>self-configuring pipeline</b> (patch size, depth, normalisation, loss, ensembling decided from the dataset)", "Most of the gain in practice comes from correct configuration", "Needs compute for 2D / 3D / cascade runs; strong default baseline"],
    ["TransUNet (2021)", "CNN features → Transformer encoder in the bottleneck → U-Net decoder", "Global context that convolutions lack", "Needs pretraining; heavier"],
    ["Swin-UNet (2021)", "Pure Swin Transformer U-shape (patch merging down, patch expanding up)", "Long-range modelling with local windows", "Data hungry"],
    ["UNETR (2021)", "3D ViT encoder; features from layers 3, 6, 9, 12 feed a CNN decoder via skips", "Global context for volumes", "Very heavy; data hungry"],
    ["UNeXt (2022)", "Convolutions early, tokenised MLP in the bottleneck", "Light and fast", "Lower capacity"],
], [3.2 * cm, 5.0 * cm, 4.4 * cm, FW - 12.6 * cm])
figure("fig_unet_variants.png", 7.4 / 11)
sub("Attention U-Net in detail")
bullets(["Problem: the raw skip features carry everything, including irrelevant background. The decoder receives them blindly.",
         "An <b>attention gate</b> takes the skip feature x (fine detail) and a <b>gating signal g</b> from the coarser decoder level (it knows what the target roughly is). "
         "It produces a map α between 0 and 1 for each pixel and multiplies the skip features by α. Pixels the network considers irrelevant are suppressed before concatenation.",
         "This is <b>additive attention</b> computed with small 1×1 convolutions. It is a local gating, not the global self-attention of Transformers."])
form(["q = ψᵀ · ReLU( W_x·x + W_g·g + b_g ) + b_ψ         α = sigmoid(q)         x_hat = α ⊙ x   (fed to the decoder)"])
sub("U-Net++ in detail")
bullets(["Skip connections in U-Net join feature maps that are <b>semantically far apart</b>: a very low-level encoder map meets a high-level decoder map. U-Net++ fills the gap with intermediate convolution blocks.",
         "Node X(i, j): level i, j-th step along the nested path. X(i, 0) is the encoder. X(i, j) = conv( concat of all X(i, k) for k &lt; j , upsampled X(i+1, j-1) ). Every node therefore receives <b>all earlier nodes at its level</b>.",
         "<b>Deep supervision:</b> a loss is attached to the outputs X(0,1) ... X(0,4). At inference you can <b>prune</b>: use only a shallower output for a faster model.",
         "Reported gains: consistent improvements across several medical datasets <i>(from the paper)</i>; the price is more parameters and memory."])
sub("nnU-Net: why it matters")
bullets(["Observation: many 'new architectures' lose against a well-configured plain U-Net. nnU-Net automates the configuration: it looks at the dataset (image sizes, spacing, class frequencies) and sets patch size, batch size, network depth, resampling, normalisation, loss (Dice + cross-entropy), augmentation and post-processing.",
         "It trains 2D, 3D full-resolution and 3D cascade variants, picks or ensembles the best by cross-validation, and has won many segmentation challenges.",
         "Interview line: <b>'before changing the architecture, make sure the training recipe is right'</b>."])
sub("Transformer-based U-Nets")
bullets(["Convolutions see only a local neighbourhood. Stacked layers grow the field of view slowly, but a Transformer relates <b>every position to every other</b> in one layer (global context).",
         "<b>TransUNet:</b> ResNet features are flattened into tokens, a ViT processes them, then a U-Net decoder upsamples using CNN skips. Keeps local precision from the CNN skips.",
         "<b>Swin-UNet:</b> shifted-window attention gives linear cost in image size. Downsampling = patch merging, upsampling = patch expanding.",
         "<b>UNETR:</b> a plain ViT as encoder on 3D patches; its intermediate layers (3, 6, 9, 12) are reshaped into feature maps and passed to a CNN decoder.",
         "Trade-off: Transformers need more data or pretraining. On small medical datasets a tuned U-Net / nnU-Net is often still the best choice."])

sec("1.8 Practical issues and fixes")
table([
    ["Problem", "Why it happens", "Fix"],
    ["Predicts only background", "Class imbalance", "Dice / focal / Tversky, class weights, oversample foreground patches"],
    ["Merged touching objects", "No boundary awareness", "Weighted loss (original paper), boundary loss, instance post-processing"],
    ["Overfits with few images", "Too many parameters, tiny dataset", "Strong augmentation (elastic), pretrained encoder, dropout, smaller U-Net, nnU-Net recipe"],
    ["Border artefacts", "Unpadded conv or tiling", "Same padding, overlapping tiles, blending"],
    ["Out of memory (3D)", "Large volumes", "Patch-based training, mixed precision, 2.5D, gradient checkpointing"],
    ["Blurry or noisy edges", "Low-resolution output, strong smoothing", "Deeper skips, deep supervision, test-time augmentation, CRF / refinement"],
    ["Good Dice but bad boundaries", "Overlap metric hides edge error", "Also report HD95, use boundary loss"],
    ["Poor on new scanner", "Domain shift", "Intensity augmentation, normalisation per image, fine-tune on target domain"],
], [3.6 * cm, 4.6 * cm, FW - 8.2 * cm])
sec("1.9 Where else U-Net appears")
bullets(["<b>Diffusion models:</b> the denoiser takes a noisy image and a timestep and outputs the noise. Same shape in and out, so a U-Net is natural. Adds timestep embedding and attention (Chapter 4).",
         "<b>Pix2Pix:</b> U-Net generator with a PatchGAN discriminator; skip connections let low-level structure (edges) pass straight from input to output.",
         "<b>Denoising, super-resolution, inpainting, optical flow:</b> any task where output is an image aligned with the input."])
