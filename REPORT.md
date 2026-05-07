# REPORT.md — Generative Deep Learning on Fashion-MNIST

**Course:** SWE012 — Deep Learning with Python
**University:** İstinye University
**Instructor:** Asst. Prof. Dr. Yiğit Bekir Kaya
**Project:** Project 2 (Final)
**Scope:** Weeks 9 (CNNs), 12 (RNN/LSTM/GRU), 13 (Autoencoders, VAEs, GANs)
**Dataset:** Fashion-MNIST (54,000 train / 6,000 validation / 10,000 test)
**Framework:** PyTorch

---

## 1. Project Overview & T-Model Design

This project follows the same **T-Model** approach used in Project 1: one core topic explored in depth (the vertical bar of the T) with all other weekly topics covered as breadth (the horizontal bar). For Project 2 the depth axis shifts to **Generative Models**, because they are the most novel concept introduced in the second half of the semester and the only one that ties classification, sequence modelling and reconstruction together into a single experimental story.

- **DEPTH (Core, Wk 13):** Generative Models — Vanilla AE → Denoising AE → Variational Autoencoder → Deep Convolutional GAN → (bonus) Diffusion Model.
- **BREADTH (Wk 9):** Convolutional Neural Networks — LeNet-style baseline, modern CNN with BatchNorm + Dropout + Augmentation, and a ResNet-style network with skip connections. The same convolutional building blocks are then *re-used* as encoders/decoders for the generative models, so the breadth axis materially supports the depth axis.
- **BREADTH (Wk 12):** Sequence Modelling — Vanilla RNN, LSTM, GRU and a Bidirectional LSTM autoencoder, treating each image as a length-28 sequence of row vectors. This deliberately under-specified inductive bias makes the comparative story (CNN vs RNN for image data) sharper.
- **BONUS:** Tiny Diffusion Model (forward/reverse process). Included in line with the instructor's note that *bonus points are awarded for topics not yet covered, provided the syllabus topics are thoroughly addressed first*.

**Why we chose this topic over alternatives.** A focused VAE-only or DCGAN-only project would have been narrower than the T-Model the instructor encouraged in Project 1. Conversely, a "build one of everything" project would be the broad/superficial pattern the instructor explicitly warned against. The Generative-Models depth lets us *visibly compare* every Week-9/12/13 building block on the same dataset and the same evaluation pipeline, while still letting the convolutional and recurrent breadth axes carry independent weight in the report.

---

## 2. Dataset Description & Quality Assessment

### Fashion-MNIST
- **Size:** 70,000 grayscale images (60,000 train / 10,000 test), each 28×28.
- **Classes:** 10 clothing categories — T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle-boot — approximately balanced.
- **Curation:** Hand-curated by Zalando Research from product images; near-zero label noise.
- **Preprocessing:** `transforms.ToTensor()` everywhere; an additional `Normalize((0.5,), (0.5,))` is applied to map images to `[-1, 1]` for the DCGAN (matching the `Tanh` output of its generator). Random horizontal flips and 28×28 random crops with 2-pixel padding are used for the modern CNN classifier only.

### Why Fashion-MNIST instead of MNIST or CIFAR-10
- It is a drop-in MNIST replacement (same shape, same loader API) with substantially higher inter-class similarity. Logistic regression caps at ~84% (vs ~92% on MNIST), so a *real* CNN is required to reach the upper-90s.
- Unlike CIFAR-10 it is small enough that a VAE, a DCGAN, *and* an RNN-AE can all train on a CPU laptop in a few minutes — essential for an end-to-end notebook.
- Class-balanced (6,000 per class) — no resampling needed.
- Visually informative — generated samples are easy to evaluate qualitatively, which is critical when comparing VAE / DCGAN / Diffusion side by side.

### Data split
- 54,000 train / 6,000 validation (split with `torch.Generator().manual_seed(42)` for reproducibility) / 10,000 test.
- The validation set is used for early-stopping decisions and cross-model comparison; the test set is reserved for the final classification confusion matrix.
- Reproducibility: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `torch.backends.cudnn.deterministic = True`.

---

## 3. Models Implemented (12 total)

| # | Model | Family | Topic | Why included |
|---|---|---|---|---|
| 1 | LeNet-style CNN | CNN classifier | Wk 9 | Classic Conv→Pool baseline, no normalization, no augmentation |
| 2 | ModernCNN | CNN classifier | Wk 9 | BatchNorm, dropout, AdamW, augmentation, strided conv replaces pooling |
| 3 | ResNet-Mini | CNN classifier | Wk 9 | Residual / skip connections, deeper but trainable |
| 4 | Conv-AE | Autoencoder | Wk 13 | Undercomplete bottleneck, MSE reconstruction |
| 5 | Denoising AE (DAE) | Autoencoder | Wk 13 | Same architecture, trained on noise-corrupted inputs |
| 6 | Variational AE (VAE) | Generative | Wk 13 | ELBO, reparameterization trick, KL regularizer |
| 7 | DCGAN | Generative | Wk 13 | Two-player minimax, BatchNorm-stabilized G/D, label smoothing |
| 8 | RNN-AE | Sequence | Wk 12 | Vanilla RNN encoder–decoder over image rows |
| 9 | LSTM-AE | Sequence | Wk 12 | Same shape, gated cell |
| 10 | GRU-AE | Sequence | Wk 12 | Same shape, smaller gated cell |
| 11 | BiLSTM-AE | Sequence | Wk 12 | Bidirectional encoder demonstration |
| 12 | Tiny Diffusion (bonus) | Generative | Bonus | Forward/reverse process, time-conditioned UNet |

---

## 4. Hyperparameter Tuning Rationale

Hyperparameters were chosen to make comparisons **fair**, not to maximise raw accuracy. The notebook uses small epoch budgets so the entire pipeline is reproducible end-to-end on a laptop; each value below is a published default or a deliberate choice we documented in the notebook.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Batch size | 128 | Standard for 28×28 grayscale, fits comfortably on CPU/GPU. |
| Optimizer (CNN baseline) | SGD lr=0.01 | Matches Project 1 — keeps training dynamics interpretable. |
| Optimizer (modern CNN, AE, VAE) | AdamW / Adam lr=1e-3 | Default for fast, stable convergence. AdamW decouples weight decay from the gradient step. |
| Optimizer (DCGAN) | Adam lr=2e-4, β₁=0.5, β₂=0.999 | DCGAN paper — vanilla β₁=0.9 destabilises adversarial training. |
| Weight decay (modern CNN) | 1e-4 | Standard mild L2; large enough to bite on Fashion-MNIST without underfitting. |
| Dropout | 0.3 (head) | Conservative; Fashion-MNIST is small but the modern CNN has 100k+ params. |
| Augmentation | RandomHorizontalFlip + RandomCrop(padding=2) | Cheap, label-preserving augmentations that improve CNN generalisation. |
| AE / VAE latent dim | 32 / 20 | Small enough to force compression, large enough to support reconstruction. |
| AE noise σ (DAE) | 0.3 | High enough to clearly corrupt the input, low enough that reconstruction remains feasible. |
| VAE β | 1.0 | Standard ELBO weight; β > 1 gives β-VAE-style disentanglement which is out of scope. |
| DCGAN ngf / ndf | 64 | Radford et al.'s default for 28-pixel inputs scaled down. |
| DCGAN label smoothing | real=0.9, fake=0.0 | One-sided smoothing — prevents the discriminator from becoming over-confident. |
| RNN hidden size | 64 | Matches the AE latent dimension order of magnitude — fair comparison. |
| Diffusion T | 200 | Small enough for 3-epoch training, large enough to demonstrate the noise schedule. |
| Diffusion β schedule | linear 1e-4 → 2e-2 | DDPM default. |
| Seed | 42 | Reproducibility across all 12 experiments. |

### What we did NOT tune (and why)
- **Adam β₂** stays at 0.999 — well established, rarely needs adjustment.
- **VAE β** stays at 1.0 — sweeping β would shift the project into β-VAE disentanglement, which is a different research question.
- **DCGAN spectral normalization / gradient penalty** were left out to keep the focus on textbook DCGAN behaviour and stability tricks (one-sided smoothing, β₁=0.5).
- **Architecture sizes** were chosen once and held constant across families — this isolates the *type* of architecture as the variable.

---

## 5. Methodology Comparison & Why We Chose Specific Approaches

### 5.1 Convolution vs Recurrence vs Adversarial — picking the right inductive bias

The single most important design choice in the project was **which inductive bias matches the data**. Fashion-MNIST is image data with strong local spatial structure, so:

- **Convolutional architectures win on classification** (~91–94% val acc with the modern CNN/ResNet-Mini vs ~88% for LeNet) because translation equivariance and local connectivity are exactly the right priors.
- **Convolutional architectures also win on reconstruction quality** — the Conv-AE produces visibly sharper reconstructions than any RNN-AE at comparable parameter counts. The same architecture re-used inside VAE and DCGAN gives those models their generative power.
- **Recurrent architectures still work** but trail in MSE — they have no spatial prior. The lesson is methodological: *RNNs are the right tool for sequence-structured data; using them on natively spatial data is a deliberate (and informative) handicap.*

### 5.2 Within CNNs — LeNet vs Modern vs ResNet
- **LeNet-style** baseline is intentionally weak: average pooling, no BN, no augmentation. It establishes a floor.
- **Modern CNN** introduces the four Week-9 best-practice ingredients (BatchNorm, Dropout, Augmentation, strided convolutions instead of pooling) — and they collectively buy 3–5 points of accuracy.
- **ResNet-Mini** adds residual / skip connections so we can stack 6 blocks without optimisation collapse. This is the same vanishing-gradient story we already documented in Project 1 with sigmoid MLPs — only now the fix is architectural rather than activation-based.

### 5.3 Vanilla AE vs Denoising AE
Both share an identical encoder/decoder. The difference is that the DAE is trained on `(x + σε, x)` pairs. This forces the DAE to map *noisy points* in input space back to the *data manifold* — a more demanding task that yields representations more robust to perturbation. We use σ = 0.3 (Gaussian, then clamped to [0,1]) — high enough that corruption is visually obvious, low enough that reconstruction remains tractable.

### 5.4 Vanilla AE vs VAE
A vanilla AE has two structural problems for generation: the latent space has no probabilistic structure, and latent codes are deterministic. The VAE fixes both by encoding to a distribution `q(z|x) = N(μ(x), σ²(x))` and adding a KL regulariser that pulls every per-sample posterior toward `N(0, I)`. Sampling from the prior then yields realistic images, and linear interpolation between two posteriors gives smooth visual transitions — both are demonstrated in the notebook (`10_vae_samples.png`, `11_vae_interp.png`).

The VAE objective is the negative ELBO:

$$\mathcal{L}_\text{VAE}(x) = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + D_\text{KL}(q_\phi(z|x)\,\|\,\mathcal{N}(0,I))$$

We use the closed-form Gaussian KL `−½ Σ (1 + log σ² − μ² − σ²)` and a per-pixel BCE reconstruction term. The reparameterisation trick `z = μ + σ ⊙ ε` keeps the sampler differentiable.

### 5.5 VAE vs DCGAN
This is the headline trade-off of Week 13:

| Aspect | VAE | DCGAN |
|---|---|---|
| Objective | Maximise ELBO (likelihood lower bound) | Two-player minimax (no likelihood) |
| Latent prior | Explicit `N(0, I)` | Implicit, defined by training |
| Sample sharpness | Slightly blurry — Gaussian decoder + averaging | Sharper, but mode-collapse risk |
| Training stability | Very stable, monotonic loss | Notoriously unstable, needs tricks |
| Latent interpolation | Smooth and meaningful | Smooth in practice, no theoretical guarantee |
| Mode coverage | Good (KL discourages mode-dropping) | Poor without intervention |
| Use cases | Representation learning, anomaly detection | Visually convincing samples, super-resolution |

**Stability tricks we used in the DCGAN:** Adam β₁ = 0.5, BatchNorm in both networks (except first/last layers), LeakyReLU(0.2) in D and ReLU in G, Tanh output, and one-sided real-label smoothing (real → 0.9). We did *not* use spectral normalization or gradient penalty — they would belong to a WGAN/SN-GAN extension and are noted as future work to keep the focus on textbook DCGAN.

### 5.6 RNN vs LSTM vs GRU
- **Vanilla RNN** suffers visibly on the 28-step row-sequence — its hidden state cannot preserve enough information to reconstruct fine details, exactly the vanishing-gradient story from theory.
- **LSTM** has input/forget/output gates and a separate cell-state path, designed precisely to mitigate vanishing gradients.
- **GRU** merges the gates and removes the separate cell state — ~25% fewer parameters than LSTM at the same hidden size, with comparable empirical performance on sequences this short.
- **Bidirectional LSTM** lets the encoder see both top-down and bottom-up context. We left the *decoder* unidirectional because true bidirectional decoding is non-causal and unusual for autoencoding.

### 5.7 Diffusion (Bonus)
Diffusion models train a denoiser that, given a noisy image and a step index, predicts the noise that was added. Sampling reverses the noise schedule iteratively. We use the textbook DDPM formulation (Ho et al., 2020) with a tiny UNet (~80k params), 200 forward steps, and a linear β schedule. Even with this minimal budget the samples are recognisable — which is the whole motivation behind diffusion's recent success.

---

## 6. Methods Executed Simultaneously

Each experiment changes at most one variable from the family-baseline so causal attribution is preserved. Methods that are routinely *combined within a single experiment* are:

| Experiment | Simultaneous methods |
|---|---|
| ModernCNN | BatchNorm + Dropout + RandomHorizontalFlip + RandomCrop + AdamW(weight_decay) + strided convolutions |
| ResNet-Mini | Above + Residual/skip connections + global average pooling |
| DAE | Vanilla AE + per-batch Gaussian noise corruption |
| VAE | Conv encoder + reparameterisation trick + KL regulariser + Adam |
| DCGAN | Strided D + Transposed-conv G + BatchNorm + LeakyReLU + Adam(β₁=0.5) + one-sided label smoothing |
| BiLSTM-AE | Bidirectional encoder + unidirectional decoder + Adam |
| Diffusion | Linear β schedule + sinusoidal time embedding + tiny UNet + ε-prediction MSE loss |

We deliberately avoid stacking too many techniques into a single ablation — the rule throughout Project 1 was "change one variable per experiment", and we keep that discipline here.

---

## 7. Performance Comparison

The table below shows a final summary of each model. CNN classifiers report *validation accuracy*; AE-family models report *MSE reconstruction loss*; the VAE reports *negative ELBO*. The DCGAN and Diffusion model are evaluated qualitatively only — sample quality is the relevant metric and there is no single scalar.

| Model | Family | Topic | Final metric | Qualitative behaviour |
|---|---|---|---|---|
| LeNet | CNN | Wk 9 | ~88% val_acc | Floor baseline; no BN/augmentation |
| ModernCNN | CNN | Wk 9 | ~91–93% val_acc | BN + augmentation closes most of the gap to ResNet |
| ResNet-Mini | CNN | Wk 9 | ~92–94% val_acc | Skip connections enable depth |
| Conv-AE | AE | Wk 13 | ~0.012–0.020 MSE | Sharp reconstructions, no generative ability |
| Denoising-AE | AE | Wk 13 | ~0.015–0.025 MSE | Slightly higher loss but more robust representations |
| VAE | Generative | Wk 13 | ~150–180 ELBO | Slightly blurry samples, smooth latent interpolation |
| DCGAN | Generative | Wk 13 | Stable losses | Sharper samples, occasional mode focus on bags/sneakers |
| RNN-AE | Sequence | Wk 12 | Highest MSE | Worst reconstruction — vanishing gradients |
| LSTM-AE | Sequence | Wk 12 | Mid MSE | Significantly better than vanilla RNN |
| GRU-AE | Sequence | Wk 12 | ≈ LSTM MSE | Same quality with ~25% fewer params |
| BiLSTM-AE | Sequence | Wk 12 | Slightly < LSTM | Bidirectional encoding shaves a small amount of MSE |
| Diffusion (bonus) | Generative | Bonus | Noise-pred MSE | Recognisable samples even with 3 epochs / 80k params |

(Exact numbers depend on the random seed and the epoch budget used at run-time; the notebook prints them to the cells and writes a CSV to `figures/results_table.csv`.)

---

## 8. Visualizations Generated

All figures are written to `DL_Project_2/figures/` by the notebook.

| File | Content | Section |
|---|---|---|
| `01_dataset_samples.png` | 10 sample images, one per class | Wk 9 dataset |
| `02_class_distribution.png` | Class-count bar chart | Wk 9 dataset |
| `03_cnn_comparison.png` | Loss + val-acc curves for the 3 CNN classifiers | Wk 9 |
| `04_filters_lenet.png` | First-layer convolutional filters of LeNet | Wk 9 |
| `05_filters_modern.png` | First-layer convolutional filters of ModernCNN | Wk 9 |
| `06_confusion_resnet.png` | ResNet-Mini test confusion matrix | Wk 9 |
| `07_ae_recon.png` | Vanilla Conv-AE reconstructions | Wk 13 |
| `08_dae_recon.png` | Denoising AE reconstructions (with noisy intermediate row) | Wk 13 |
| `09_ae_tsne.png` | t-SNE of the AE latent space, colour by class | Wk 13 |
| `10_vae_samples.png` | 64 samples from `z ~ N(0, I)` | Wk 13 |
| `11_vae_interp.png` | Linear interpolation between two test-set codes | Wk 13 |
| `12_vae_tsne.png` | t-SNE of VAE posterior μ, colour by class | Wk 13 |
| `13_dcgan_loss_and_samples.png` | DCGAN G/D losses and final-epoch samples | Wk 13 |
| `14_dcgan_progression.png` | Fixed-z samples across all training epochs | Wk 13 |
| `15_rnn_loss.png` | Validation reconstruction loss for RNN/LSTM/GRU/BiLSTM | Wk 12 |
| `16_rnn_reconstructions.png` | Side-by-side reconstructions for the RNN family | Wk 12 |
| `17_diffusion_samples.png` | Tiny diffusion model samples | Bonus |
| `results_table.csv` | Programmatically generated results summary | All |

---

## 9. Weekly Topic Coverage Checklist

| Week | Topic | Covered? | Where |
|---|---|---|---|
| **Wk 9** | Convolution operation (kernel, stride, padding) | ✅ | LeNet / Modern / ResNet code |
| | Average vs strided pooling | ✅ | LeNet uses AvgPool; ModernCNN uses strided conv |
| | BatchNorm in CNNs | ✅ | ModernCNN + ResBlock |
| | Dropout in CNN heads | ✅ | ModernCNN + ResNet-Mini head |
| | Data augmentation | ✅ | RandomHorizontalFlip + RandomCrop |
| | Skip / residual connections | ✅ | ResBlock |
| | Filter visualisation | ✅ | `04_filters_lenet.png`, `05_filters_modern.png` |
| | Transposed convolutions vs upsample+conv | ✅ | DCGAN uses ConvTranspose; Conv-AE uses Upsample+Conv (with documented rationale) |
| | Confusion-matrix error analysis | ✅ | ResNet-Mini test confusion |
| **Wk 12** | Vanilla RNN cell (`h_t = tanh(Wh+Wx+b)`) | ✅ | RNN-AE |
| | LSTM (input/forget/output gates, cell state) | ✅ | LSTM-AE |
| | GRU (reset/update gates) | ✅ | GRU-AE |
| | Vanishing gradients in long sequences | ✅ | RNN vs LSTM/GRU comparison |
| | Encoder–decoder sequence architecture | ✅ | All four RNN-AEs |
| | Bidirectional encoders | ✅ | BiLSTM-AE |
| | Sequence reconstruction loss | ✅ | MSE per row |
| **Wk 13** | Undercomplete autoencoders | ✅ | Conv-AE |
| | Denoising autoencoders | ✅ | DAE with σ = 0.3 Gaussian noise |
| | Latent space visualisation (t-SNE) | ✅ | `09_ae_tsne.png`, `12_vae_tsne.png` |
| | Variational autoencoders (ELBO) | ✅ | VAE |
| | Reparameterisation trick | ✅ | `VAE.reparameterize` |
| | KL divergence (closed form) | ✅ | `vae_loss_fn` |
| | Latent interpolation | ✅ | `11_vae_interp.png` |
| | GAN minimax objective | ✅ | DCGAN |
| | DCGAN architectural rules | ✅ | BN, LeakyReLU/ReLU, Tanh output, no pooling |
| | One-sided label smoothing for stability | ✅ | DCGAN training loop |
| | Mode-collapse discussion | ✅ | VAE vs DCGAN comparison table |
| **Bonus** | Diffusion forward / reverse process | ✅ | Tiny DDPM section |
| | Time-conditioned UNet | ✅ | `TinyUNet` |
| | Sinusoidal time embeddings | ✅ | `TimeEmbed` module |

Note: Per the instructor's email of 25 April, **Transformers / Attention Mechanisms are excluded** from the project scope.

---

## 10. Repository Structure

```
DL_Project_2/
├── README.md
├── REPORT.md                    # This file
├── requirements.txt
├── notebooks/
│   └── Generative_Deep_Learning_Fashion_MNIST.ipynb
├── responsibilities/
│   ├── 220901755.md             # Theoretical foundations + Wk 9 CNN baselines
│   ├── 229910141.md             # Modern CNN + ResNet-Mini + filter visualisation
│   ├── 229910158.md             # Convolutional Autoencoders (vanilla + denoising)
│   ├── 2309011036.md            # VAE + DCGAN (Wk 13 depth)
│   └── 2309011053.md            # RNN/LSTM/GRU + Bonus diffusion + integration
└── figures/                     # Generated by the notebook
```

---

## 11. Conclusions

1. **Architecture matters most when matched to the data.** CNNs dominate both classification *and* reconstruction on Fashion-MNIST because their inductive biases (translation equivariance, local connectivity, weight sharing) match the data. RNNs reconstruct, but at a clear quality cost — the lesson is methodological.
2. **Generative depth ladder works as predicted.** Vanilla AE → DAE → VAE → DCGAN → Diffusion. Each step adds a distinct mechanism: noise robustness → probabilistic latent → adversarial sharpness → iterative denoising.
3. **VAE vs DCGAN is the canonical stability/quality trade-off.** VAE training is stable and reproducible; DCGAN training is delicate but produces sharper samples. Both still have a place in modern systems.
4. **Skip connections rescue depth.** ResNet-Mini trains stably at six residual blocks; the same depth without skips would suffer from the same vanishing-gradient story we already documented with sigmoid MLPs in Project 1.
5. **GRU ≈ LSTM at a smaller cost** on 28-step sequences; vanilla RNN visibly under-fits, validating the gating motivation from Week 12.
6. **Diffusion is the natural next step** — even our 80k-parameter, 3-epoch toy model produces recognisable samples, which is why diffusion has dominated generative-modelling research since 2020.

### Design recommendations distilled from this study

| Need | Pick |
|---|---|
| Image classification | ResNet-style CNN with BN, augmentation, AdamW |
| Image compression / denoising | Convolutional Autoencoder (DAE if input is noisy) |
| Smooth, controllable latent space | VAE |
| Sharpest possible samples | DCGAN or Diffusion |
| Sequence input | LSTM or GRU; avoid vanilla RNN past ~10 timesteps |
| Best of both worlds (research path) | Diffusion |
