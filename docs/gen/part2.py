from kit import *

H("2. Variational Autoencoder (VAE)")
sec("2.1 The problem it solves")
bullets(["We want to <b>generate new data</b> (images, molecules, sounds) that look like the training data, and also to have a compact <b>latent code</b> that captures what matters.",
         "A plain <b>autoencoder</b> (encoder → code → decoder, trained to reconstruct) compresses well but cannot generate: its latent space has holes. Pick a random code and the decoder outputs garbage, because it only learned to decode the specific codes that training images produced.",
         "A VAE reshapes the latent space so that <b>nearby codes decode to similar images and random codes from a simple distribution decode to realistic images</b>."])
sec("2.2 Intuition")
para("Instead of mapping an image to one point, the encoder maps it to a <b>small cloud</b> (a Gaussian: centre μ and spread σ). During training we sample a point from the cloud and decode it. "
     "Because clouds from similar images overlap, the decoder learns that nearby points mean similar things, which fills the holes. A second force (the KL term) pulls all clouds toward one standard bell-shaped region, "
     "so at generation time we can sample from that region and decode.")
para("Analogy and its limit: it is like describing a face by sliders. The cloud says 'about this setting, give or take'. The analogy stops being exact because the sliders are learned, not named, and several sliders may mix several properties.")
sec("2.3 Architecture")
figure("fig_vae.png", 5.2 / 11)
bullets(["<b>Encoder q(z|x):</b> a CNN (or MLP) that outputs two vectors: the mean μ and the log-variance log σ². (It predicts the log of the variance so the value can be any real number and σ² stays positive.)",
         "<b>Sampling with the reparameterisation trick:</b> z = μ + σ ⊙ ε with ε ~ N(0, I). The randomness is moved to ε, which does not depend on the network weights, so backpropagation flows through μ and σ.",
         "<b>Decoder p(x|z):</b> a CNN that maps z back to image space (transposed convolutions or upsampling). Its output is the mean of a distribution over pixels (Gaussian, or Bernoulli for binary images).",
         "<b>Generation:</b> throw away the encoder. Sample z ~ N(0, I), decode.",
         "<b>Interpolation:</b> encode two images, walk linearly between their μ, decode each point: a smooth morph. This works because the latent space has no holes."])
sec("2.4 The loss: the negative ELBO")
para("We want to maximise the likelihood p(x) of the training data but it needs an integral over all z that is intractable. The VAE maximises a <b>lower bound</b> on log p(x) called the ELBO (evidence lower bound).")
form(["log p(x)  ≥  E_q(z|x)[ log p(x|z) ]  -  KL( q(z|x) || p(z) )          (ELBO)",
      "Loss = - ELBO = Reconstruction loss  +  KL term          p(z) = N(0, I)"])
sub("Where the ELBO comes from (short derivation)")
bullets(["Start from log p(x) = log ∫ p(x|z) p(z) dz. Multiply and divide by q(z|x) inside the integral.",
         "By Jensen's inequality (log of an average ≥ average of the log) we move the log inside: log p(x) ≥ E_q[ log p(x|z) + log p(z) - log q(z|x) ].",
         "Rearranging gives reconstruction term minus KL( q(z|x) || p(z) ).",
         "The gap between log p(x) and the ELBO is exactly KL( q(z|x) || p(z|x) ): how far the encoder's guess is from the true posterior. Better encoder → tighter bound."])
sub("The two terms")
bullets(["<b>Reconstruction:</b> with a Gaussian decoder it is ||x - x_hat||² (mean squared error up to a constant). With Bernoulli pixels it is binary cross-entropy. 'Make the output look like the input.'",
         "<b>KL:</b> penalises encoder clouds that differ from N(0, I). 'Keep the latent space organised and sampleable.'",
         "They <b>fight</b>: reconstruction wants distinct, narrow, informative clouds (a lot of information in z); KL wants everything to look the same (no information). The balance decides the behaviour."])
form(["KL( N(μ, σ²) || N(0, 1) ) per dimension = - 1/2 · ( 1 + log σ² - μ² - σ² )          sum over latent dimensions"])
ex("closed-form KL for one latent dimension", [
    "μ = 0, σ² = 1 (already the prior): KL = -½(1 + 0 - 0 - 1) = <b>0</b>.",
    "μ = 1, σ² = 1 (shifted): KL = -½(1 + 0 - 1 - 1) = <b>0.5</b>. Shifting the centre costs.",
    "μ = 0, σ² = 4 (too wide): KL = -½(1 + ln 4 - 0 - 4) = -½(1 + 1.386 - 4) = <b>0.807</b>.",
    "μ = 2, σ² = 0.25 (informative, narrow): KL = -½(1 + ln 0.25 - 4 - 0.25) = <b>2.32</b>. Information in z costs KL: this is the price of encoding 'a lot' in a dimension."])
sub("Why the KL term has a real meaning")
bullets(["Average KL over the dataset is an upper bound of the <b>bits of information</b> that z carries about x. KL close to 0 means z carries almost nothing.",
         "Per-dimension KL tells you which latent dimensions are <b>active</b> (KL clearly above 0) and which are unused."])
sec("2.5 Training in practice")
bullets(["Loss per image = reconstruction (sum over pixels) + β · KL (sum over latent dimensions). Standard VAE has β = 1.",
         "Adam optimiser, learning rate around 1e-3 for small models. Normalise pixels to [0, 1] and use BCE, or to [-1, 1] and use MSE.",
         "<b>Scale matters:</b> if reconstruction is averaged over pixels but KL summed over dimensions, the balance is wrong. Keep the same convention (sum or mean) for both, or tune β.",
         "Monitor <b>reconstruction and KL separately</b>, not only the total."])
sec("2.6 Known problems, their causes and fixes")
sub("Blurry samples")
bullets(["Cause: the Gaussian decoder with MSE predicts the <b>average</b> of all images compatible with a code, and an average of sharp images is blurry. Also KL pushes the clouds to overlap, mixing different images.",
         "Fixes: perceptual (LPIPS) loss instead of pixel loss; add an <b>adversarial loss</b> (VAE-GAN, VQGAN, Stable Diffusion's autoencoder); more expressive decoder; hierarchical VAEs (NVAE); or use the VAE only as a compressor and generate with a diffusion model in its latent space (Stable Diffusion)."])
sub("Posterior collapse")
bullets(["<b>What:</b> the encoder ignores the input: q(z|x) ≈ p(z) = N(0, I) for every x. KL ≈ 0. The decoder generates the 'average' output regardless of z. Samples all look alike or generic.",
         "<b>Why:</b> if the decoder is powerful enough to model the data on its own (autoregressive decoders such as LSTMs or PixelCNN, very strong decoders), using z is unnecessary and costs KL, so the optimiser sets z aside. It is also a local optimum early in training when the encoder is poor.",
         "<b>How to detect:</b> total KL near 0; per-dimension KL all near 0 (no 'active units'); decoding different z gives the same output; mutual information between x and z near 0.",
         "<b>Fixes:</b> (1) <b>KL annealing / warm-up</b>: start β at 0 and increase to 1 over the first epochs, so the model first learns to use z. (2) <b>Free bits</b>: do not penalise KL below a small threshold λ per dimension. (3) <b>Weaken the decoder</b> (smaller, word dropout in text, limited context). "
         "(4) <b>Lower β</b> (below 1) trading sampling quality for information. (5) Train the encoder more aggressively than the decoder at the start. (6) Use a <b>discrete latent</b> (VQ-VAE), which has no KL term to collapse. (7) Add skip paths from z to every decoder layer."])
sub("Holes and mismatch between aggregate posterior and prior")
bullets(["Even with a good KL, the average of all encoder clouds may not match N(0, I) exactly, so some prior samples land in regions the decoder has never seen. Fixes: more expressive priors (mixtures, flows, learned prior), hierarchical VAEs, or a second generative model on the latents (as in Stable Diffusion and VQ-VAE + Transformer)."])
sec("2.7 Important variants")
table([
    ["Variant", "Idea", "Loss change", "Why use it"],
    ["β-VAE (2017)", "Multiply the KL by β > 1", "Recon + β·KL", "Pushes toward <b>disentangled</b> factors (one latent per factor of variation) at the cost of blurrier output"],
    ["Conditional VAE", "Feed a label or condition to encoder and decoder", "Same ELBO, conditioned", "Controlled generation (digit class, text)"],
    ["VQ-VAE (2017)", "Latent is <b>discrete</b>: pick the nearest vector from a learned codebook", "Recon + codebook loss + commitment loss", "No posterior collapse; sharper; works with autoregressive priors (images, audio)"],
    ["VQ-VAE-2", "Hierarchy of discrete latents", "Same", "High-resolution images"],
    ["VQGAN (2021)", "VQ-VAE + perceptual + adversarial loss; Transformer prior", "Recon (perceptual) + GAN + codebook", "Sharp reconstructions; basis for latent generative models"],
    ["Hierarchical VAE (NVAE, VDVAE)", "Many layers of latents, coarse to fine", "ELBO with a KL per layer", "Competitive likelihoods and sharper samples"],
    ["KL-regularised AE (Stable Diffusion)", "Tiny KL weight (about 1e-6) + LPIPS + patch adversarial loss", "Recon + LPIPS + GAN + very small KL", "A <b>compressor</b>, not a generator: 8× smaller in each side, little blur"],
], [3.3 * cm, 4.6 * cm, 3.7 * cm, FW - 11.6 * cm])
sub("VQ-VAE in a bit more detail")
bullets(["Encoder output z_e is replaced by the nearest of K codebook vectors e_k. The decoder receives the code vector.",
         "The nearest-neighbour step is not differentiable, so the gradient is copied from the decoder input straight to the encoder output (<b>straight-through estimator</b>).",
         "Loss = ||x - x_hat||² + ||sg[z_e] - e||² + β · ||z_e - sg[e]||² where sg is stop-gradient. The second term moves codebook vectors toward encoder outputs, the third (commitment, β ≈ 0.25) keeps the encoder from drifting away from the codes.",
         "<b>Codebook collapse</b> (only a few codes used): fix with EMA codebook updates, k-means initialisation, restarting dead codes, smaller codebook dimension, or better normalisation."])
sec("2.8 How to evaluate a VAE")
bullets(["<b>ELBO / negative log-likelihood</b> on held-out data, often given in <b>bits per dimension</b> (nats divided by (number of dimensions × ln 2)). Lower is better. Report the reconstruction and KL parts separately.",
         "<b>Reconstruction quality:</b> MSE, PSNR, SSIM, LPIPS (perceptual) between input and reconstruction. For a compressor (Stable Diffusion's VAE) also <b>rFID</b> (FID of reconstructions).",
         "<b>Sample quality and diversity:</b> FID of samples drawn from the prior against real data (see Chapter 3).",
         "<b>Latent space health:</b> number of active units (dimensions with variance of μ across data above a threshold), KL per dimension, interpolation smoothness, latent traversals by eye.",
         "<b>Disentanglement</b> (β-VAE style work): MIG, DCI, FactorVAE score, which need datasets with known factors.",
         "<b>Downstream:</b> accuracy of a classifier trained on the latent codes."])
ex("bits per dimension", ["A binarised-MNIST VAE reports a negative ELBO of 90 nats per image. Dimensions = 28×28 = 784. Bits per dimension = 90/(784 × ln 2) = 90/543.4 = <b>0.166</b>.",
    "Comparing models on different data needs this normalisation. Remember that ELBO is a bound: the true likelihood is at least as good."])
sec("2.9 VAE versus plain autoencoder in one table")
table([
    ["", "Autoencoder", "VAE"],
    ["Encoder output", "One point", "Mean and variance (a cloud)"],
    ["Latent space", "May have holes", "Smooth and organised by the KL term"],
    ["Generation", "Not reliable", "Sample from N(0, I) and decode"],
    ["Loss", "Reconstruction", "Reconstruction + KL (negative ELBO)"],
    ["Typical issue", "Overfits to identity mapping", "Blur, posterior collapse"],
], [3.2 * cm, 6.5 * cm, FW - 9.7 * cm])
