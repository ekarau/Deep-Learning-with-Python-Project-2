# Technical Design Notes

Engineering companion to `PAPER.md`: architectural rationale, training
protocol, ablation matrix, and compute budget.

---

## 1. Architectural decisions

### Block 1 — 3D-CNN encoder/decoder

ACDC volumes carry voxel-level spatial structure. To preserve inter-slice
context we use 3D rather than 2D convolutions, in the U-Net encoder–decoder
topology (Ronneberger 2015). A five-stage encoder with channel widths
`{32, 64, 128, 256, 320}` yields a bottleneck whose receptive field spans
the full volume.

### Block 2 — Convolutional autoencoder

100 training patients is a small-data regime. We pretrain the encoder as
a denoising autoencoder (Vincent et al. 2008) and transfer its weights to
the segmentation network. Loss: MSE. Corruptions: Gaussian noise σ = 0.1
and voxel dropout p = 0.2.

### Block 3 — ConvLSTM

The cardiac cycle in cine MRI is a genuine temporal signal (T ≈ 25-30
frames). Vanilla RNNs suffer vanishing gradients at this horizon.
ConvLSTM (Shi et al. 2015) solves this via gating while preserving
spatial structure through 3D convolutional gates.

### Block 4 — Attention gates

Plain skip connections are agnostic to which encoder voxels are relevant
for the current decoder query. An additive attention gate (Oktay et al.
2018) produces a soft mask `α ∈ [0, 1]` that re-weights the skip feature
before concatenation.

### Block 5 — VAE diagnosis branch

Pooled encoder features are mapped to a Gaussian latent `(μ, log σ²)`.
The reparameterisation trick (Kingma & Welling 2014) enables
backpropagation through sampling. A KL term pulls the latent toward
N(0, I); an MLP head outputs a 5-class disease prediction.

---

## 2. Training protocol

- 5-fold patient-stratified cross-validation.
- Adam, lr = 1e-4, 5-epoch warm-up + cosine annealing, 100 epochs.
- Effective batch size 4 (batch 2 × gradient accumulation 2).
- Early stopping on validation Dice with patience 20.
- Headline number averaged over 3 random seeds (42, 123, 7).

### Regularisation stack

- Dropout `p = 0.2` on decoder feature maps
- L₂ weight decay `1e-5`
- InstanceNorm throughout (batch = 2 makes BatchNorm noisy)
- 3D augmentation: random affine, Gaussian noise, flip, elastic deformation

---

## 3. Ablation matrix

Each scenario uses the same 5-fold split, the same seed, and the same
epoch budget — exactly one component changes at a time.

| # | Variant | Removed | Expected Δ Dice |
|---|---|---|---|
| A0 | Full model | — | reference |
| A1 | − ConvLSTM | temporal modelling | ≈ −0.04 |
| A2 | − AE pretraining | self-supervised init | ≈ −0.03 |
| A3 | − Attention | gated skip refinement | ≈ −0.02 |
| A4 | − VAE | probabilistic latent | seg ≈ 0, diagnosis ↓ |
| A5 | 2D-CNN | 3D spatial context | ≈ −0.07 |
| A6 | LSTM → GRU | gate complexity | small |
| A7 | − Augmentation | regularisation | ≈ −0.06 |

Auxiliary sweeps: optimiser (SGD / Nesterov / RMSProp / Adam),
initialisation (He / Xavier / Zero), regularisation components.

---

## 4. Compute budget (Colab T4)

| Stage | Time |
|---|---|
| Download + preprocessing | 30 min |
| AE pretraining (50 epochs) | ≈ 2 h |
| Full model (100 epochs) | ≈ 3-4 h |
| 7 ablations (5 owners in parallel) | wall-clock ≈ 3 h |
| Optimiser / init / regularisation sweeps | ≈ 8 h |
| **Total active compute** | **≈ 16-20 h** |
