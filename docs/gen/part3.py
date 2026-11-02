from kit import *

H("3. Generative Adversarial Networks (GANs)")
sec("3.1 The problem it solves")
bullets(["Generate realistic samples (images, video, audio) from a simple random vector, <b>without writing down a likelihood</b>.",
         "VAEs optimise pixel-wise reconstruction, which averages and blurs. A GAN replaces 'is the pixel close?' with a <b>learned judge</b> that asks 'does this look real?', which rewards sharp, realistic detail.",
         "Goodfellow et al. 2014. Dominated image generation until diffusion models (2020 to 2022)."])
sec("3.2 Intuition")
para("A counterfeiter (<b>generator G</b>) makes fake banknotes. A detective (<b>discriminator D</b>) tries to tell fake from real. Each improves because of the other. When training works, the fakes become so good that the detective guesses 50/50. "
     "Where the analogy breaks: the generator never sees the real notes. It only learns from the detective's feedback (the gradient through D).")
figure("fig_gan.png", 5.0 / 11)
sec("3.3 Architecture")
bullets(["<b>Generator:</b> noise vector z (often 100 to 512 numbers from N(0, I)) → fully connected → reshape → a stack of upsampling blocks (transposed convolution or upsample + convolution) → image. Last activation tanh (images in [-1, 1]).",
         "<b>Discriminator:</b> an ordinary image classifier: strided convolutions that downsample, ending in one score (probability of real, or an unbounded score in WGAN).",
         "<b>DCGAN guidelines (2015)</b> that made GANs train: strided convolutions instead of pooling, batch normalisation, no fully connected layers in the body, ReLU in G and LeakyReLU in D.",
         "<b>Training loop:</b> (1) train D on a batch of real and a batch of fake (G frozen). (2) train G so that D calls its fakes real (D frozen, but gradients flow <b>through</b> D to G). Repeat. WGAN-GP trains D 5 times per G step."])
sec("3.4 The objective and what it really optimises")
form(["min_G max_D  V(D, G) = E_x~p_data [ log D(x) ] + E_z~p_z [ log( 1 - D(G(z)) ) ]",
      "For a fixed G the best discriminator is  D*(x) = p_data(x) / ( p_data(x) + p_g(x) )",
      "Plugging D* in:  V(D*, G) = - log 4 + 2 · JSD( p_data || p_g )          minimum -log 4 = -1.386 when p_g = p_data (then D* = 0.5 everywhere)"])
bullets(["So an ideal GAN minimises the <b>Jensen-Shannon divergence</b> between real and generated distributions.",
         "<b>Problem:</b> real images and generated images live on thin low-dimensional surfaces inside pixel space and usually <b>do not overlap</b>. When two distributions do not overlap, JSD is the constant ln 2 no matter how far apart they are, so there is <b>no useful gradient</b> (see the plot below). This is the theoretical root of unstable GAN training."])
figure("fig_gan_plots.png", 3.9 / 13)
para("Left: why the generator uses a different loss than the textbook one. Middle: why Wasserstein distance helps. Right: schematic of mode collapse (a made-up 2-D example with 8 Gaussians). Next sections explain each.")

sec("3.5 Loss functions, one by one")
sub("1. Original (saturating) GAN loss")
form(["D loss:  - [ log D(x) + log( 1 - D(G(z)) ) ]          G loss:  + log( 1 - D(G(z)) )   (minimise)"])
sub("2. Non-saturating GAN loss (what people actually use)")
form(["G loss:  - log D(G(z))"])
bullets(["Early in training D rejects fakes confidently: D(G(z)) ≈ 0. For the saturating loss the gradient with respect to D's logit is proportional to D(G(z)) ≈ 0, so <b>G learns almost nothing exactly when it needs to learn most</b> (vanishing gradient). The non-saturating loss has gradient proportional to 1 - D(G(z)) ≈ 1."])
ex("saturation with numbers", [
    "D(x) = 0.9 on a real image, D(G(z)) = 0.2 on a fake. D loss = -(ln 0.9 + ln 0.8) = 0.105 + 0.223 = <b>0.329</b>. Non-saturating G loss = -ln 0.2 = <b>1.609</b>. Saturating G loss = ln 0.8 = -0.223.",
    "Early training: D(G(z)) = 0.01. Gradient w.r.t. the logit a (D = sigmoid(a)): saturating = D = <b>0.01</b>, non-saturating = 1 - D = <b>0.99</b>. About a 100× difference."])
sub("3. Least-squares GAN (LSGAN)")
form(["D loss:  ½ E[ (D(x) - 1)² ] + ½ E[ D(G(z))² ]          G loss:  ½ E[ (D(G(z)) - 1)² ]"])
para("Replaces the log loss by squared error. Fakes that are on the correct side of the boundary but far from the real data still receive gradient (no saturation), which gives more stable training and slightly sharper images.")
ex("LSGAN", ["D(x) = 0.9, D(G(z)) = 0.2: D loss = ½(0.9 - 1)² + ½(0.2)² = 0.005 + 0.02 = <b>0.025</b>. G loss = ½(0.2 - 1)² = <b>0.32</b>."])
sub("4. Hinge loss (SAGAN, BigGAN)")
form(["D loss:  E[ max(0, 1 - D(x)) ] + E[ max(0, 1 + D(G(z))) ]          G loss:  - E[ D(G(z)) ]       (D outputs raw scores, not probabilities)"])
para("D is only pushed until real scores exceed +1 and fake scores fall below -1; beyond that margin there is no more gradient, so D does not become overconfident.")
ex("hinge", ["Real score 0.7, fake score -0.3: real term max(0, 1 - 0.7) = 0.3; fake term max(0, 1 + (-0.3)) = 0.7. D loss = <b>1.0</b>. If the real score were 1.5 and the fake -1.5, D loss = <b>0</b>."])
sub("5. Wasserstein GAN (WGAN) and WGAN-GP")
form(["W(P_r, P_g) = sup over 1-Lipschitz f of  E_x~P_r[ f(x) ] - E_x~P_g[ f(x) ]          (Kantorovich-Rubinstein duality)",
      "Critic loss:  E[ f(G(z)) ] - E[ f(x) ]          Generator loss:  - E[ f(G(z)) ]",
      "WGAN-GP adds:  λ · E_x_hat[ ( || ∇ f(x_hat) ||₂ - 1 )² ]      x_hat = random point on the line between a real and a fake sample;  λ = 10"])
bullets(["The 'discriminator' becomes a <b>critic</b> that outputs an unbounded score (no sigmoid). The loss approximates the <b>Earth Mover (Wasserstein-1) distance</b>: the minimum 'work' to move the generated distribution onto the real one.",
         "<b>Why it is better:</b> Wasserstein distance decreases smoothly as distributions approach, even when they do not overlap (middle plot). Gradients stay informative, and the critic loss <b>correlates with sample quality</b>, so you can watch it as a progress signal (a rare property for GANs).",
         "The critic must be <b>1-Lipschitz</b>. WGAN enforced it by clipping weights to [-0.01, 0.01] (crude, causes poor capacity use). <b>WGAN-GP</b> instead penalises the gradient norm at points between real and fake samples to be 1. Settings from the paper: λ = 10, 5 critic steps per generator step, Adam with lr 1e-4, β1 = 0, β2 = 0.9. Do not use batch norm in the critic (it couples samples in a batch); use layer norm.",
         "<b>Spectral normalisation (SN-GAN)</b> is a cheaper way to enforce Lipschitz continuity: divide each weight matrix by its largest singular value (estimated by power iteration)."])
ex("WGAN numbers", ["Critic scores: real 2.1, fake -0.4. Critic loss = -(2.1 - (-0.4)) = <b>-2.5</b>, so the estimated Wasserstein distance ≈ 2.5.",
    "Gradient penalty: gradient norm at an interpolated point = 1.5. Penalty = 10 × (1.5 - 1)² = <b>2.5</b>. A norm of exactly 1 gives 0."])
ex("Wasserstein vs JSD", ["Real distribution: a point mass at 0. Fake distribution: a point mass at θ. For any θ ≠ 0 the JSD is the constant ln 2 = 0.693 (no signal about direction). The Wasserstein distance is |θ|: shrinking θ always reduces it, so the generator has a gradient everywhere."])
sub("6. Relativistic GAN")
para("Asks 'is this real sample <b>more realistic than</b> the average fake?' instead of 'is it real?'. Often more stable and gives better FID in practice (used in several modern GANs).")
sub("7. Regularisers and auxiliary losses")
form(["R1 penalty:  γ/2 · E_x~p_data [ || ∇ D(x) ||² ]          (StyleGAN2: lazy regularisation, γ depends on resolution)",
      "Pix2Pix:  G* = argmin_G max_D  L_cGAN(G, D) + λ · E || y - G(x) ||₁          λ = 100",
      "CycleGAN cycle consistency:  E || F(G(x)) - x ||₁ + E || G(F(y)) - y ||₁          λ = 10",
      "Feature matching: match the mean of an intermediate D feature of fakes to that of reals. Perceptual loss: distance in VGG / LPIPS feature space."])
bullets(["<b>R1</b> penalises D's gradient on real data, stopping D from becoming too sharp around real samples; it makes training locally convergent.",
         "<b>L1 reconstruction</b> (Pix2Pix) anchors the output to the paired target so the GAN term only has to add sharpness.",
         "<b>Cycle consistency</b> makes unpaired translation (horse ↔ zebra) possible: translating there and back must give the original."])
table([
    ["Loss", "D output", "Strength", "Weakness"],
    ["Minimax / non-saturating", "Probability", "Simple, works with care", "Vanishing gradients; unstable"],
    ["LSGAN", "Score", "No saturation; stable", "Can over-penalise confident correct samples"],
    ["Hinge", "Score", "Margin; popular in big models", "Needs normalisation tricks"],
    ["WGAN", "Score (critic)", "Meaningful loss curve; gradients everywhere", "Weight clipping hurts"],
    ["WGAN-GP", "Score (critic)", "Stable; no clipping", "Slower (gradient of gradient); no batch norm"],
    ["Relativistic", "Score difference", "Stable, good FID", "Slightly more complex"],
    ["R1 / R2 penalty", "Added to any loss", "Local convergence", "Weight must be tuned"],
], [3.3 * cm, 2.6 * cm, 5.6 * cm, FW - 11.5 * cm])

sec("3.6 Major GAN families (what each adds)")
table([
    ["Model", "Key idea", "Why it matters"],
    ["DCGAN (2015)", "Convolutional architecture rules", "First stable image GAN recipe"],
    ["Conditional GAN (2014)", "Feed a label to G and D", "Controlled generation"],
    ["InfoGAN", "Maximise mutual information between a code and the output", "Unsupervised disentangled controls"],
    ["Pix2Pix (2016)", "U-Net generator, PatchGAN discriminator, cGAN + L1", "Paired image translation"],
    ["CycleGAN (2017)", "Two generators + cycle consistency", "Unpaired translation"],
    ["SRGAN / ESRGAN", "Perceptual + adversarial loss", "Realistic super-resolution"],
    ["ProGAN (2017)", "Grow resolution progressively; minibatch standard deviation; pixel norm", "First 1024×1024 faces; the std layer fights mode collapse"],
    ["SAGAN (2018)", "Self-attention + spectral norm + hinge", "Long-range structure"],
    ["BigGAN (2018)", "Large batches, class conditioning, truncation trick", "High-fidelity ImageNet; shows scale helps"],
    ["StyleGAN 1 / 2 / 3", "Mapping network, style injection, noise inputs; weight demodulation; alias-free", "State-of-the-art faces; controllable and disentangled latents"],
], [3.3 * cm, 7.0 * cm, FW - 10.3 * cm])

sec("3.7 Why GAN training is hard")
bullets(["<b>Two networks, one moving target:</b> each loss depends on the other network, so the 'landscape' changes every step. It is a game, not a single minimisation; simultaneous gradient descent can <b>cycle</b> or oscillate instead of converging.",
         "<b>Vanishing gradients:</b> a D that is too good gives G nothing to learn from.",
         "<b>Non-overlapping distributions</b> (JSD plateau).",
         "<b>Sensitivity</b> to architecture, learning rates, normalisation and initial conditions.",
         "<b>No single loss that tells quality:</b> the loss values of G and D are not a quality measure (except WGAN critic). You must judge with samples and metrics (Section 3.9)."])

sec("3.8 Mode collapse")
sub("What it is")
bullets(["The generator produces <b>only a few kinds of output</b> (low diversity) even though the real data has many. In the 8-Gaussian toy: real data has 8 clusters, the generator covers only 1 or 2.",
         "<b>Full collapse:</b> the same image for any z. <b>Partial collapse / mode dropping:</b> some classes or styles are missing (for example only certain digits or faces without glasses). <b>Intra-mode collapse:</b> variety is lost inside a mode.",
         "It is not the same as non-convergence: collapsed models can have a 'good-looking' loss and sharp images that are all alike."])
sub("Why it happens (the mechanism)")
bullets(["G is rewarded for fooling D, not for covering the data. If one output is the most convincing to the current D, G moves all its mass there (the maximum-seeking behaviour of the min-max order).",
         "D, seeing only individual samples, can punish that point but cannot see that <b>all</b> samples are identical. D then pushes G away from the point; G jumps to another point that D currently likes. The cycle repeats: <b>whack-a-mole</b> instead of covering all modes.",
         "Each sample is judged <b>independently</b>: nothing in the plain objective measures the diversity of a batch.",
         "Contributing factors: overpowered or underpowered D, vanishing gradients (JSD plateau), poor learning-rate balance, small or unbalanced datasets, a very low-capacity or badly regularised G, and too aggressive training of G on a stale D."])
sub("How to detect it")
bullets(["<b>Look at a large grid of samples</b> from different z. Near-duplicates, or the same pose / colour, are the signature.",
         "<b>Interpolate</b> between z values. If the output barely changes or jumps between a few images, suspect collapse.",
         "<b>Recall (improved precision / recall) or coverage:</b> low recall with high precision = good-looking but not diverse (see 3.9).",
         "<b>Mode counting</b> on synthetic benchmarks: 8 Gaussians (how many are covered), <b>Stacked MNIST</b> (1000 modes: how many of the 1000 digit triples appear; plus KL between generated and true mode frequencies).",
         "<b>Diversity measures:</b> mean pairwise LPIPS or SSIM across generated samples (low LPIPS = collapse), number of statistically different bins (NDB), classifier-based class histogram of the generated images.",
         "<b>Training signs:</b> D loss near 0 while G loss rises, G loss oscillating strongly, FID that stops improving or gets worse while images look sharp."])
sub("How to fix it (ordered by how I would try them)")
table([
    ["Remedy", "How it helps", "Used in"],
    ["Switch the objective to WGAN-GP, or LSGAN / hinge", "Gradients stay informative when distributions do not overlap; reduces over-confident D", "WGAN-GP, SAGAN, BigGAN"],
    ["Spectral normalisation in D (and G)", "Limits how fast D can change, avoids a D that is too sharp", "SN-GAN, SAGAN, BigGAN"],
    ["Minibatch discrimination / minibatch standard deviation layer", "D sees <b>statistics of the whole batch</b>, so a batch of identical fakes is easy to spot, and G is forced to diversify", "Improved-GAN, ProGAN, StyleGAN"],
    ["Unrolled GAN", "G is updated using D <b>after</b> several future D steps, so G cannot simply exploit the current D", "Metz et al. 2016"],
    ["Two time-scale update rule (TTUR)", "Different learning rates for D and G (for example D 4e-4, G 1e-4) so D tracks G and training converges", "SAGAN, BigGAN"],
    ["R1 / R2 gradient penalty", "Stabilises D near the data, local convergence", "StyleGAN2"],
    ["Packing (PacGAN)", "D classifies <b>several samples at once</b> (packed together), so lack of variety is penalised", "PacGAN"],
    ["Feature matching", "G matches batch-level feature statistics instead of only fooling D", "Improved-GAN"],
    ["Mode-seeking or diversity regulariser", "Directly penalises two different z mapping to similar images (maximise image distance / latent distance)", "MSGAN"],
    ["Multiple generators or mixtures", "Each generator covers a different mode, extra loss to keep them apart", "MGAN, MAD-GAN"],
    ["Add a reconstruction / encoder (VAE-GAN, VEEGAN, BiGAN)", "The encoder forces every real sample to have a place in latent space", "VEEGAN, BiGAN"],
    ["Data augmentation for D (DiffAugment, ADA) and more data", "Stops D from memorising a small dataset, which otherwise pushes G to a few safe outputs", "StyleGAN2-ADA"],
    ["Label smoothing, instance noise", "Softens D's targets so it does not become overconfident", "Improved-GAN"],
    ["Conditioning on labels; truncation trick for sampling", "Conditioning guarantees class coverage; truncation trades diversity for quality at sampling time, so it is a trade-off to report", "BigGAN"],
    ["Better balance of D and G capacity and steps", "A D that is too strong gives vanishing gradients; too weak gives no signal", "General practice"],
], [4.3 * cm, 8.5 * cm, FW - 12.8 * cm])
bullets(["<b>Checklist when I see collapse:</b> (1) confirm with a sample grid and a diversity / recall measure; (2) check data (balanced, enough, preprocessed to [-1, 1]); (3) move to hinge or WGAN-GP with spectral norm; (4) add minibatch-stddev layer; (5) tune learning rates (TTUR) and D / G balance; "
         "(6) add augmentation for D; (7) if it still fails, consider a diffusion model, which does not suffer from mode collapse by construction (it is trained with a likelihood-style objective that covers the whole data distribution).",
         "<b>Honest note:</b> no fix is guaranteed. Mode collapse is reduced, not eliminated, and often a combination is needed."])

sec("3.9 Evaluating GANs")
para("There is no likelihood, so we compare <b>sets</b> of samples. Good metrics should reward both <b>fidelity</b> (each sample looks real) and <b>diversity</b> (all modes covered).")
table([
    ["Metric", "What it measures", "Formula idea", "Weakness"],
    ["Inception Score (IS)", "Quality and class diversity", "exp( E_x KL( p(y|x) || p(y) ) ) with an ImageNet classifier", "Needs ImageNet-like content; ignores real data; can be gamed"],
    ["FID", "Distance between real and generated feature distributions", "Fit Gaussians to Inception features; Fréchet distance", "Gaussian assumption; biased with few samples; ImageNet features"],
    ["KID", "Same idea without Gaussian assumption", "Unbiased squared MMD with a cubic polynomial kernel", "Higher variance; slower"],
    ["Precision / Recall (improved)", "Fidelity and diversity separately", "k-NN balls around real / fake features; fraction of fakes inside real manifold, and reals inside fake manifold", "Sensitive to outliers and k"],
    ["Density / Coverage", "Robust version of precision / recall", "Counts how many real-neighbourhoods contain each fake", "Needs feature extractor"],
    ["LPIPS diversity", "Variety of generated outputs", "Mean pairwise LPIPS between samples", "Does not measure realism"],
    ["Human study", "Perceived realism", "Two-alternative forced choice, ELO", "Costly, subjective"],
], [3.0 * cm, 3.8 * cm, 5.6 * cm, FW - 12.4 * cm])
form(["FID = || μ_r - μ_g ||² + Tr( Σ_r + Σ_g - 2 (Σ_r Σ_g)^(1/2) )          (μ, Σ of 2048-d Inception-v3 pool features; lower is better)",
      "IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )          (between 1 and the number of classes; higher is better)",
      "KID = unbiased MMD²  with k(a, b) = ( aᵀb / d + 1 )³"])
ex("FID in one dimension", ["Real features: mean 0, std 1. Generated: mean 1, std 2. FID = (1 - 0)² + (1 + 4 - 2×1×2) = 1 + 1 = <b>2</b>. The first part measures the shift of the mean, the second the mismatch of the spread. A collapsed generator has a too-small spread, which raises the second term."])
ex("Inception Score, two classes", ["Every generated image is classified confidently (p(y|x) is one-hot) and the two classes appear equally often: p(y) = [0.5, 0.5]. KL(one-hot || uniform) = ln 2 for every image, so IS = exp(ln 2) = <b>2</b> (the maximum for 2 classes).",
    "If the generator collapsed to one class: p(y|x) = p(y) = one-hot, KL = 0, IS = exp(0) = <b>1</b> (the minimum)."])
ex("precision and recall", ["Generate 100 samples: 80 fall inside the real-data manifold, so precision = <b>0.80</b>. Take 100 real samples: only 30 fall inside the generated manifold, so recall = <b>0.30</b>.",
    "High precision with low recall is exactly the signature of <b>mode dropping</b>: the images are good but many kinds of real image are never produced. FID alone would only show one mixed number."])
bullets(["<b>Practical rules for FID:</b> use at least 10,000 samples (often 50,000), the same preprocessing for real and fake, the same Inception network and resize method (use a standard implementation such as Clean-FID), and report the real-data subset size. FID depends on the Inception network trained on ImageNet, so for medical or very different domains consider domain features, and do not compare numbers across papers with different protocols.",
         "<b>Use several metrics together:</b> FID (overall), precision / recall or density / coverage (separating quality from diversity), and look at samples.",
         "<b>For conditional models:</b> add class-conditional accuracy of a classifier on generated images, and per-class FID."])
sec("3.10 GAN versus VAE versus diffusion (preview)")
table([
    ["", "VAE", "GAN", "Diffusion"],
    ["Sample sharpness", "Blurry (pixel loss)", "Sharp", "Sharp"],
    ["Diversity / mode coverage", "Good", "Risk of mode collapse", "Very good"],
    ["Training", "Stable", "Unstable, needs care", "Stable"],
    ["Sampling speed", "One pass (fast)", "One pass (fast)", "Many steps (slow, improving)"],
    ["Likelihood", "Lower bound (ELBO)", "None", "Lower bound (training objective)"],
    ["Latent space", "Explicit, encoder available", "No encoder by default", "Noise levels; inversion possible"],
], [3.8 * cm, 4.2 * cm, 4.2 * cm, FW - 12.2 * cm])
