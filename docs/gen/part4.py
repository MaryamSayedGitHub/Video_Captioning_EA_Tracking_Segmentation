from kit import *

H("4. Stable Diffusion")
sec("4.1 The problem it solves")
bullets(["Generate an image that matches a <b>text prompt</b>, with high quality and diversity.",
         "Earlier diffusion models (DDPM, ADM, Imagen) ran in <b>pixel space</b>: a U-Net on 512×512×3 images, hundreds of steps. Powerful but extremely expensive.",
         "<b>Latent Diffusion (Rombach et al., 2022)</b> idea: most pixels carry imperceptible detail. First <b>compress</b> the image with an autoencoder, then run diffusion only on the small latent. Stable Diffusion is the open model built on this idea."])
para("Compression example: a 512×512×3 image has 786,432 numbers. Its latent is 64×64×4 = 16,384 numbers, <b>48× fewer</b>. Self-attention at the 64×64 level compares 4,096 positions: 4,096² ≈ 16.8 million pairs per head. "
     "In pixel space (262,144 positions) it would be about 6.9×10¹⁰ pairs. This is why latent diffusion made high-resolution text-to-image practical.")
sec("4.2 Diffusion from scratch (intuition first)")
para("Take a photograph and add a little random noise, repeatedly, until only static remains. That is easy and needs no learning. Diffusion models learn the <b>reverse</b>: given a noisy image, predict the noise that was added, subtract a bit of it, and repeat. "
     "Starting from pure static and repeating this removal many times yields a new image. Analogy: a sculptor revealing a statue by removing material step by step; the analogy is limited because here the 'sculptor' is a trained network and the removal is random-guided.")
figure("fig_diffusion_forward.png", 2.7 / 13)
sub("Forward process (fixed, no learning)")
form(["q(x_t | x_{t-1}) = N( sqrt(1 - β_t) · x_{t-1},  β_t · I )          small noise β_t added at each of T = 1000 steps",
      "α_t = 1 - β_t          ᾱ_t = α_1 · α_2 · ... · α_t",
      "x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε          ε ~ N(0, I)          (jump to any step t in one shot)"])
bullets(["ᾱ_t falls from 1 (clean) to near 0 (pure noise). Stable Diffusion uses a <b>scaled-linear</b> schedule of β from 0.00085 to 0.012 over 1000 steps <i>(from the model configuration)</i>.",
         "Known subtlety: with this schedule ᾱ_T is small but not exactly 0, so a trace of the original image survives at the last step; this is one reason SD v1 models struggle to produce very dark or very bright images, and 'zero terminal SNR' rescaling is a recommended fix."])
ex("one noising step", ["ᾱ_t = 0.64, so sqrt(ᾱ_t) = 0.8 and sqrt(1 - ᾱ_t) = 0.6. One pixel x_0 = 1.0 with noise ε = 0.5: x_t = 0.8×1.0 + 0.6×0.5 = <b>1.1</b>.",
    "A perfect noise predictor would give back x_0 = (x_t - 0.6 × ε)/0.8 = (1.1 - 0.3)/0.8 = <b>1.0</b>. This identity is how the model's noise prediction is turned into an image estimate."])
sub("Training (what the network learns)")
bullets(["Pick a clean training sample x_0 and a random timestep t. Sample noise ε. Build x_t with the formula above.",
         "The network ε_θ(x_t, t) predicts the noise. Loss = mean squared error between true and predicted noise. That is all.",
         "Why predicting the noise works: it is equivalent to learning the <b>score</b> (the gradient of the log-density) of the noisy data, which tells in which direction the data becomes more probable. The loss is also a re-weighted form of the ELBO."])
form(["L_simple = E_{x_0, ε, t} [ || ε - ε_θ( x_t, t ) ||² ]          Stable Diffusion:  E [ || ε - ε_θ( z_t, t, τ(y) ) ||² ]   (z = latent, τ(y) = text embedding)"])
sub("Sampling (generation)")
form(["DDPM step:  x_{t-1} = 1/sqrt(α_t) · ( x_t - (1 - α_t)/sqrt(1 - ᾱ_t) · ε_θ(x_t, t) ) + σ_t · z",
      "DDIM step (deterministic):  x0_hat = ( x_t - sqrt(1 - ᾱ_t) · ε_θ ) / sqrt(ᾱ_t);   x_{t-1} = sqrt(ᾱ_{t-1}) · x0_hat + sqrt(1 - ᾱ_{t-1}) · ε_θ"])
bullets(["DDPM needs the full 1000 steps because each step is a small stochastic update. <b>DDIM</b> reinterprets the same trained model as a deterministic process, so 20 to 50 steps are enough, and the same starting noise always gives the same image.",
         "Modern samplers (Euler, Euler ancestral, DPM-Solver++, UniPC) are better ODE / SDE solvers that reach good quality in 15 to 30 steps. Distilled models (LCM, turbo, lightning) go down to 1 to 4 steps."])
table([
    ["Sampler", "Deterministic?", "Typical steps", "Note"],
    ["DDPM", "No", "1000", "Original; slow"],
    ["DDIM", "Yes (η = 0)", "20 to 50", "Fast; reproducible; enables inversion"],
    ["Euler / Euler a", "Euler: yes, 'a' (ancestral): no", "20 to 40", "Ancestral adds fresh noise each step: more variety, does not converge to one image"],
    ["DPM-Solver++ / UniPC", "Mostly yes", "15 to 25", "High-order solvers, good quality in few steps"],
    ["LCM / Turbo / Lightning", "Distilled", "1 to 8", "Student trained to skip steps; some loss in diversity / detail"],
], [3.5 * cm, 4.2 * cm, 2.6 * cm, FW - 10.3 * cm])

sec("4.3 Architecture of Stable Diffusion v1.x")
figure("fig_sd_pipeline.png", 6.2 / 11)
para("The system has <b>three separately trained parts</b>: the VAE (compress / decompress), the text encoder (understand the prompt) and the U-Net (the generator). Roughly 860M (U-Net) + 123M (text encoder) + 84M (VAE) parameters <i>(from the model card and paper; the VAE size is approximate)</i>.")
sub("Part 1: the VAE (autoencoder)")
bullets(["Encoder downsamples 8× in each side: 512×512×3 → 64×64×4 latent. Decoder reverses it. It is a convolutional autoencoder with residual blocks and some attention at the lowest resolution.",
         "Trained <b>once, before and separately from</b> the diffusion model, with: reconstruction loss (L1 / L2), <b>LPIPS perceptual loss</b> (sharp textures instead of blur), <b>patch-based adversarial loss</b> (a GAN discriminator on image patches) and a tiny KL term (weight about 1e-6) that only keeps the latent scale reasonable. Details of this are in Chapter 2.",
         "Latents are multiplied by a <b>scale factor 0.18215</b> before diffusion so that they have about unit variance (the diffusion noise assumes that scale).",
         "Limits: the VAE is a bottleneck for fine detail. Small text, faces at low resolution and tiny structures can already be damaged by 8× compression (hence the improved VAEs in SDXL and SD3)."])
sub("Part 2: the text encoder")
bullets(["<b>CLIP ViT-L/14 text Transformer, frozen</b> (v1.x). The prompt is tokenised and padded to 77 tokens; the output is a sequence of 77 vectors of size 768. Not a single vector: every token keeps its own embedding so the U-Net can attend to individual words.",
         "Frozen because it already understands language and images jointly (trained contrastively, see CLIP in the other notes). Limit: CLIP is weak at counting, spatial relations and rendering text, which shows in generated images.",
         "Prompts longer than 77 tokens are truncated (or chunked by tools)."])
sub("Part 3: the U-Net denoiser")
figure("fig_sd_unet.png", 6.4 / 11)
bullets(["Input: noisy latent z_t (4×64×64), timestep t, text embeddings. Output: predicted noise, same shape as the input (4×64×64). Same in and out shape is why a U-Net fits.",
         "<b>Levels:</b> resolutions 64, 32, 16, 8 with channels 320, 640, 1280, 1280 (multipliers 1, 2, 4, 4). Each level has ResNet blocks (2 on the way down, 3 on the way up); attention blocks at the first three levels and in the middle block <i>(from the original configuration)</i>.",
         "<b>Timestep embedding:</b> t becomes a sinusoidal vector (like Transformer positions), passes through a small MLP and is added inside every ResNet block, so every block knows how noisy the input is.",
         "<b>Self-attention</b> inside the attention block lets all positions in the latent exchange information (global layout). <b>Cross-attention</b> injects the text.",
         "Skip connections (concatenation) as in any U-Net."])
form(["Cross-attention:  Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d) ) · V          Q = W_Q · (image features),   K = W_K · τ(y),   V = W_V · τ(y)"])
bullets(["Queries come from the image latent (4,096 positions at the 64×64 level), keys and values come from the 77 text tokens. The attention matrix is 4,096 × 77: <b>each image position decides which words matter to it</b>. This is where 'a red fox' makes the fox region red."])
sub("Classifier-free guidance (CFG)")
bullets(["Training: with probability about 10% the text is replaced by an empty prompt, so one network learns both conditional and unconditional prediction.",
         "Sampling: run the U-Net twice per step (with the prompt and with the empty prompt; in practice batched together) and <b>extrapolate</b> away from the unconditional prediction toward the conditional one.",
         "Guidance scale s (default about 7.5). s = 1: plain conditional. Higher s: follows the prompt more strongly but with less diversity and, if too high, oversaturated, artefact-prone images. Cost: doubles the U-Net work per step."])
form(["ε_guided = ε_uncond + s · ( ε_cond - ε_uncond )"])
ex("guidance", ["For one latent value: ε_uncond = 0.20, ε_cond = 0.50, s = 7.5. ε_guided = 0.20 + 7.5×(0.30) = <b>2.45</b>. The guided prediction is far outside the range of either prediction: the model is pushed strongly in the 'what the prompt adds' direction.",
    "Cost of one image: 50 steps × 2 passes = 100 U-Net evaluations (batched as 50 passes with batch 2)."])
sub("Training data and cost")
bullets(["Trained on a subset of <b>LAION-5B</b> (hundreds of millions of image-text pairs filtered by resolution, aesthetic score and watermark probability), first at 256×256 and then at 512×512 <i>(from the model description)</i>.",
         "Training cost was in the order of hundreds of thousands of GPU-hours. We use pretrained weights and adapt them (LoRA, DreamBooth, textual inversion) instead of training from scratch."])
sec("4.4 What you can build on top")
table([
    ["Technique", "What it does", "How"],
    ["Image-to-image (SDEdit)", "Edit or restyle a given image", "Encode the image, add noise up to step t_start (strength), then denoise with a new prompt"],
    ["Inpainting / outpainting", "Fill a masked region", "Extra input channels for the mask and masked image; or blend latents during sampling"],
    ["ControlNet", "Control layout (edges, pose, depth)", "A trainable copy of the encoder takes the control image; its outputs are added to the frozen U-Net via zero-initialised convolutions"],
    ["LoRA", "Cheap fine-tuning of style or subject", "Train small low-rank matrices added to the attention weights (a few MB instead of GB)"],
    ["DreamBooth", "Teach a specific subject / person", "Fine-tune with a few images and a rare token; prior-preservation loss avoids forgetting"],
    ["Textual inversion", "Learn a new word", "Optimise only a new token embedding; the model stays frozen"],
    ["IP-Adapter", "Image prompts", "Extra cross-attention from image embeddings"],
    ["Negative prompt", "Steer away from things", "Used in place of the empty prompt in CFG"],
], [3.5 * cm, 4.0 * cm, FW - 7.5 * cm])
sec("4.5 Versions and what changed")
table([
    ["Model", "U-Net", "Text encoder(s)", "Other key changes"],
    ["SD 1.4 / 1.5", "860M parameters", "CLIP ViT-L/14, context dim 768", "512×512, latent 4×64×64; epsilon prediction"],
    ["SD 2.0 / 2.1", "865M parameters", "OpenCLIP ViT-H, context dim 1024", "768×768 model uses v-prediction; new training data filtering"],
    ["SDXL (2023)", "2.6B parameters; transformer blocks per level [0, 2, 10]; drops the lowest 8× level", "OpenCLIP ViT-bigG + CLIP ViT-L, context dim 2048 (817M parameters total)", "Size and crop conditioning; multi-aspect training near 1024×1024; improved VAE trained from scratch; optional <b>refiner</b> model on the first 200 noise scales"],
    ["SD 3 (2024)", "<b>MMDiT</b> Transformer replaces the U-Net; 450M to 8B parameters in the study", "Two CLIP models + T5 (4.7B)", "Separate weights for image and text tokens joined in attention; <b>rectified flow</b> training; improved autoencoder"],
], [2.6 * cm, 4.5 * cm, 4.2 * cm, FW - 11.3 * cm])
bullets(["<i>Numbers for SDXL and SD3 from the SDXL paper and the Stability AI SD3 research page: 2.6B vs 860M U-Net, text encoders 817M total, 8B model, 24GB VRAM / 34 s for 1024×1024 with 50 steps (the company's early unoptimised test).</i>",
         "<b>Why SDXL's U-Net moves compute to lower resolutions:</b> it removes the 8× level and puts 10 transformer blocks at the 4× level, because attention is cheaper on small grids and the semantics matter more there.",
         "<b>Why SD3 changes the backbone:</b> Transformers scale predictably with size, and joint attention over text and image tokens (with separate weights per modality) improves prompt understanding and text rendering over cross-attention alone.",
         "<b>Rectified flow</b> connects data and noise by a straight line, x_t = (1 - t)·x_0 + t·ε, and trains the network to predict the constant velocity (ε - x_0). Straighter paths allow fewer sampling steps."])
form(["Rectified flow:  x_t = (1 - t) x_0 + t ε          loss = || v_θ(x_t, t) - ( ε - x_0 ) ||²          v-prediction (SD 2.x):  v = sqrt(ᾱ_t)·ε - sqrt(1 - ᾱ_t)·x_0"])
sec("4.6 How to evaluate Stable Diffusion")
bullets(["<b>FID</b> on a standard prompt set (for example COCO captions) against real images: overall realism and diversity. Same cautions as in Chapter 3.",
         "<b>CLIP score:</b> cosine similarity between the CLIP embedding of the image and of the prompt (a prompt-adherence proxy). Limits: CLIP is blind to counting and relations, and models can be tuned to the metric.",
         "<b>Human preference:</b> side-by-side comparisons, ELO scores; learned preference models (PickScore, HPS, ImageReward) approximate this.",
         "<b>Compositional benchmarks:</b> GenEval, T2I-CompBench, DrawBench measure objects, counts, colours, positions and attributes.",
         "<b>Diversity:</b> LPIPS between images from the same prompt, precision / recall in feature space.",
         "<b>Also:</b> latency per image and steps, memory use, safety and bias audits, memorisation checks (does it reproduce training images?).",
         "<b>Autoencoder alone:</b> rFID, PSNR, SSIM of reconstruction (SDXL paper reports rFID 4.4 for its VAE, 5.0 for the SD 1.x VAE <i>(from the SDXL paper)</i>)."])
sec("4.7 Failure modes and fixes")
table([
    ["Problem", "Cause", "Fix"],
    ["Wrong object counts, bad spatial relations", "CLIP text encoder and cross-attention limits", "Stronger text encoders (T5, SDXL, SD3), layout control (ControlNet), prompt engineering"],
    ["Garbled text in images", "Weak text encoding + latent compression", "SD3-style joint attention, T5 encoder"],
    ["Hands and fine detail", "Small structures lost in 8× latent, data variety", "Bigger / better VAE, higher resolution, refiner, inpainting"],
    ["Washed-out or mid-grey images", "Noise schedule does not reach zero SNR", "Zero-terminal-SNR rescaling, v-prediction"],
    ["Over-saturated, harsh images", "CFG scale too high", "Lower guidance (5 to 8), dynamic thresholding"],
    ["Slow generation", "Many steps × 2 passes for CFG", "Better samplers, distilled models, fewer steps, FP16 + efficient attention"],
    ["Memorised training images, bias", "Training data duplicates and web biases", "Deduplication, safety filters, careful fine-tuning data"],
    ["Mode collapse", "Not a typical issue for diffusion", "Diffusion models cover the data distribution well; low diversity usually means too high CFG or over-distilled models"],
], [4.0 * cm, 5.0 * cm, FW - 9.0 * cm])

# ================================================================ 5 comparison and Q&A
H("5. Comparison and question bank")
sec("5.1 When to use which")
table([
    ["Need", "Choose", "Reason"],
    ["Pixel-accurate labels in an image", "U-Net / nnU-Net", "Dense prediction with precise boundaries and small data"],
    ["Compact learned representation, anomaly detection", "VAE", "Explicit encoder, likelihood bound, smooth latent"],
    ["Fast, sharp generation with a narrow domain (faces)", "GAN (StyleGAN)", "Single pass; high fidelity; editable latent space"],
    ["Flexible, diverse, text-guided generation", "Diffusion (Stable Diffusion)", "Stable training, great coverage, easy conditioning"],
    ["Pre-compress images for another generator", "VAE / VQGAN", "Makes the second model cheap (latent diffusion)"],
], [5.5 * cm, 4.0 * cm, FW - 9.5 * cm])
sec("5.2 Question bank with short answers")
qa = [
 ("Why does U-Net have skip connections?", "The decoder loses spatial detail through pooling. Skips pass high-resolution encoder features to the decoder so it can localise boundaries while using the deep features for context."),
 ("Skip by concatenation vs addition?", "Concatenation keeps both sources and lets the next layer learn how to mix them. Addition (ResNet) forces the same channel count and mixes them directly."),
 ("Why Dice loss instead of cross-entropy?", "Dice measures overlap relative to object size, so it is not dominated by the large background. Often combined with cross-entropy for stable gradients."),
 ("Dice vs IoU?", "Same idea, Dice = 2·IoU/(1 + IoU). Dice is always at least as large as IoU and is more forgiving; IoU punishes errors more."),
 ("Why can pixel accuracy mislead?", "With 9% foreground, predicting mostly background still gives 90%+ accuracy. Use Dice, IoU, sensitivity."),
 ("What does an attention gate do?", "It computes a 0 to 1 map from the skip features and a gating signal from the coarser decoder level, and multiplies the skip features by it, suppressing irrelevant regions."),
 ("U-Net++ vs U-Net?", "Nested dense skip pathways reduce the semantic gap between encoder and decoder maps and add deep supervision. More parameters, better accuracy in many cases, can be pruned at inference."),
 ("What is nnU-Net?", "A self-configuring pipeline that sets patch size, depth, loss and training recipe from the dataset. It shows that the recipe matters more than new blocks."),
 ("Why Transformers in U-Net?", "Global context that convolutions lack. But they need more data or pretraining, so on small datasets a tuned CNN U-Net often wins."),
 ("What is the reparameterisation trick?", "Write z = μ + σ·ε with ε ~ N(0, I). The sample becomes a differentiable function of μ and σ, so gradients pass through."),
 ("What is the ELBO?", "A lower bound on log p(x): expected reconstruction log-likelihood minus KL between the encoder's distribution and the prior. Maximising it trains the VAE."),
 ("Why are VAE images blurry?", "A Gaussian decoder with MSE predicts the average of plausible outputs, and the KL term makes posteriors overlap. Fix with perceptual / adversarial loss or by using the VAE only as a compressor."),
 ("What is posterior collapse and how do you fix it?", "The encoder ignores x and q(z|x) equals the prior; KL is near 0. Fix with KL annealing, free bits, weaker decoder, lower β, or discrete latents (VQ-VAE)."),
 ("β-VAE?", "Multiplies the KL by β > 1 to encourage disentangled latents, at the cost of reconstruction quality."),
 ("VQ-VAE vs VAE?", "Discrete codebook latent instead of Gaussian, trained with codebook and commitment losses and a straight-through gradient; no KL, so no posterior collapse, but codebook collapse is possible."),
 ("What is the GAN objective doing?", "With an optimal discriminator the generator minimises the Jensen-Shannon divergence between data and model distributions."),
 ("Why the non-saturating generator loss?", "The minimax loss has vanishing gradient when D rejects fakes confidently. -log D(G(z)) gives strong gradients exactly then."),
 ("Why Wasserstein?", "It stays informative when real and fake distributions do not overlap, so gradients do not vanish, and the critic loss correlates with quality."),
 ("How does WGAN-GP enforce the Lipschitz constraint?", "It penalises the critic's gradient norm to be 1 at random points between real and fake samples (λ = 10), replacing weight clipping."),
 ("What is mode collapse?", "The generator produces only a few kinds of output. Caused by each sample being judged independently, and by G chasing whatever D currently likes."),
 ("How do you detect and fix mode collapse?", "Detect: sample grids, interpolation, low LPIPS diversity, low recall, mode counting. Fix: WGAN-GP or hinge, spectral norm, minibatch-stddev or packing, unrolling, TTUR, augmentation, diversity regularisers, conditioning."),
 ("What is FID and what are its limits?", "Fréchet distance between Gaussians fitted to Inception features of real and generated images. Biased for few samples, ImageNet-specific, blind to some failures; use ≥10k samples and a standard implementation."),
 ("Inception Score vs FID?", "IS only looks at generated images (class confidence and class variety). FID compares against real data, so it detects missing diversity and shifts better."),
 ("Why diffusion instead of GAN?", "Stable training, excellent coverage of modes, flexible conditioning. Price: many sampling steps (mitigated by better samplers and distillation)."),
 ("What does the network predict in diffusion?", "The noise ε added to the clean sample (sometimes v or x_0). The loss is MSE between true and predicted noise."),
 ("Why diffuse in latent space?", "A 48× smaller representation makes both training and sampling far cheaper while keeping perceptual quality, using an autoencoder trained once."),
 ("How does text enter Stable Diffusion?", "A frozen CLIP text encoder gives 77 token embeddings; cross-attention layers in the U-Net use them as keys and values with image features as queries."),
 ("What is classifier-free guidance?", "Train with random prompt dropout, then at sampling extrapolate: ε_uncond + s(ε_cond - ε_uncond). Larger s follows the prompt more but reduces diversity."),
 ("DDPM vs DDIM?", "Same trained model. DDPM samples stochastically with ~1000 steps, DDIM uses a deterministic update with 20 to 50 steps."),
 ("What changed in SDXL and SD3?", "SDXL: 3× larger U-Net, two text encoders, size/crop conditioning, refiner, new VAE. SD3: Transformer (MMDiT) with joint text-image attention, rectified flow, three text encoders."),
 ("What is LoRA?", "Low-rank updates added to attention weights, trained instead of the full model: small files, fast fine-tuning."),
]
for q, a in qa:
    S.append(KeepTogether([Paragraph(f"<b>{q}</b>", P), Paragraph(a, B), Spacer(1, 2)]))

# ================================================================ 6 sources
H("6. Sources and original figures")
para("Open these to see the <b>original figures</b> from the papers (U-Net Figure 1, Attention U-Net Figure 2, U-Net++ Figure 1, the VAE and GAN diagrams, the Latent Diffusion architecture figure, the SDXL and SD3 diagrams). "
     "Links point to arXiv abstract pages; check licences before reusing figures in your own material.")
refs = [
 ("U-Net", "Ronneberger, Fischer, Brox 2015", "https://arxiv.org/abs/1505.04597"),
 ("3D U-Net", "Çiçek et al. 2016", "https://arxiv.org/abs/1606.06650"),
 ("V-Net", "Milletari et al. 2016", "https://arxiv.org/abs/1606.04797"),
 ("Attention U-Net", "Oktay et al. 2018", "https://arxiv.org/abs/1804.03999"),
 ("U-Net++", "Zhou et al. 2018", "https://arxiv.org/abs/1807.10165"),
 ("UNet 3+", "Huang et al. 2020", "https://arxiv.org/abs/2004.08790"),
 ("nnU-Net", "Isensee et al. 2018", "https://arxiv.org/abs/1809.10486"),
 ("TransUNet", "Chen et al. 2021", "https://arxiv.org/abs/2102.04306"),
 ("Swin-UNet", "Cao et al. 2021", "https://arxiv.org/abs/2105.05537"),
 ("UNETR", "Hatamizadeh et al. 2021", "https://arxiv.org/abs/2103.10504"),
 ("U-Net review (variants, theory)", "Siddique et al. 2020", "https://arxiv.org/abs/2011.01118"),
 ("U-Net review (2025)", "Comprehensive review", "https://arxiv.org/abs/2502.06895"),
 ("VAE", "Kingma and Welling 2013", "https://arxiv.org/abs/1312.6114"),
 ("VQ-VAE", "van den Oord et al. 2017", "https://arxiv.org/abs/1711.00937"),
 ("VQGAN", "Esser et al. 2021", "https://arxiv.org/abs/2012.09841"),
 ("Lagging inference and posterior collapse", "He et al. 2019", "https://arxiv.org/abs/1901.05534"),
 ("GAN", "Goodfellow et al. 2014", "https://arxiv.org/abs/1406.2661"),
 ("DCGAN", "Radford et al. 2015", "https://arxiv.org/abs/1511.06434"),
 ("Wasserstein GAN", "Arjovsky et al. 2017", "https://arxiv.org/abs/1701.07875"),
 ("WGAN-GP", "Gulrajani et al. 2017", "https://arxiv.org/abs/1704.00028"),
 ("LSGAN", "Mao et al. 2016", "https://arxiv.org/abs/1611.04076"),
 ("Spectral normalisation", "Miyato et al. 2018", "https://arxiv.org/abs/1802.05957"),
 ("Unrolled GAN", "Metz et al. 2016", "https://arxiv.org/abs/1611.02163"),
 ("Pix2Pix", "Isola et al. 2016", "https://arxiv.org/abs/1611.07004"),
 ("CycleGAN", "Zhu et al. 2017", "https://arxiv.org/abs/1703.10593"),
 ("Progressive GAN", "Karras et al. 2017", "https://arxiv.org/abs/1710.10196"),
 ("StyleGAN", "Karras et al. 2018", "https://arxiv.org/abs/1812.04948"),
 ("BigGAN", "Brock et al. 2018", "https://arxiv.org/abs/1809.11096"),
 ("FID and TTUR", "Heusel et al. 2017", "https://arxiv.org/abs/1706.08500"),
 ("Inception Score (Improved techniques)", "Salimans et al. 2016", "https://arxiv.org/abs/1606.03498"),
 ("KID", "Binkowski et al. 2018", "https://arxiv.org/abs/1801.01401"),
 ("Improved precision and recall", "Kynkäänniemi et al. 2019", "https://arxiv.org/abs/1904.06991"),
 ("Pros and cons of GAN evaluation measures", "Borji 2018", "https://arxiv.org/abs/1802.03446"),
 ("DDPM", "Ho et al. 2020", "https://arxiv.org/abs/2006.11239"),
 ("DDIM", "Song et al. 2020", "https://arxiv.org/abs/2010.02502"),
 ("Latent Diffusion (Stable Diffusion)", "Rombach et al. 2021", "https://arxiv.org/abs/2112.10752"),
 ("Classifier-free guidance", "Ho and Salimans 2022", "https://arxiv.org/abs/2207.12598"),
 ("ControlNet", "Zhang et al. 2023", "https://arxiv.org/abs/2302.05543"),
 ("LoRA", "Hu et al. 2021", "https://arxiv.org/abs/2106.09685"),
 ("SDXL", "Podell et al. 2023", "https://arxiv.org/abs/2307.01952"),
 ("Stable Diffusion 3 (rectified flow transformers)", "Esser et al. 2024", "https://arxiv.org/abs/2403.03206"),
 ("DiT", "Peebles and Xie 2022", "https://arxiv.org/abs/2212.09748"),
 ("Stable Diffusion 3 research page", "Stability AI", "https://stability.ai/news/stable-diffusion-3-research-paper"),
]
rows = [["Topic", "Authors / year", "Link"]] + [[a, b, link(u)] for a, b, u in refs]
table(rows, [5.0 * cm, 4.2 * cm, FW - 9.2 * cm])
para("<i>Facts confirmed during this session by reading the SDXL paper, the Stability AI SD3 research page and a review of U-Net variants. The arXiv identifiers above are from memory of the standard references; if a link does not open, search the title.</i>")
