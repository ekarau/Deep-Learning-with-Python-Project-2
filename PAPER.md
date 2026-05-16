# Self-Supervised 3D-CNN + ConvLSTM + VAE for Cine Cardiac MRI Segmentation and Diagnosis: An Ablation Study on ACDC

**Authors:** [Student 1], [Student 2], [Student 3], [Student 4], [Student 5]
**Affiliation:** Department of Software Engineering, İstinye University, İstanbul, Türkiye
**Course:** SWE012 — Deep Learning with Python (Spring 2026)
**Instructor:** Asst. Prof. Dr. Yiğit Bekir Kaya

---

## Abstract

We present a unified deep architecture for joint segmentation and diagnosis on the Automated Cardiac Diagnosis Challenge (ACDC) dataset (Bernard et al. 2018, *IEEE TMI*). The model integrates five architectural blocks — a 3D convolutional encoder, a convolutional autoencoder used for self-supervised pretraining, a convolutional long short-term memory (ConvLSTM) module that operates along the cardiac time axis, attention-gated skip connections, and a variational autoencoder for disease classification — each motivated by a specific property of cine cardiac MRI. We additionally report a seven-scenario ablation that isolates the empirical contribution of every block, characterise the impact of regularisation choices (dropout, batch normalisation, L₂ penalty, early stopping, data augmentation), and benchmark four optimisers (SGD with momentum, Nesterov, RMSProp, Adam) under matched compute budgets. On 100 patients (5-fold patient-stratified cross-validation), the full model attains a Dice score of [TBD] on left-ventricle, [TBD] on right-ventricle, and [TBD] on myocardium, while reaching [TBD] balanced accuracy on the five-class diagnosis task.

**Keywords:** cardiac MRI segmentation, ConvLSTM, autoencoder pretraining, variational autoencoder, ablation study, ACDC.

---

## 1. Introduction

Automated analysis of cine cardiac magnetic resonance (cMRI) sequences has clinical value for the quantification of ventricular volumes, ejection fraction, and the discrimination of cardiac pathologies. The ACDC benchmark (Bernard et al. 2018) provides 150 patient cine cMRI scans stratified into five diagnostic subgroups (normal, previous myocardial infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy, abnormal right ventricle) with expert annotations of left ventricle, right ventricle, and myocardium at end-diastole and end-systole.

Three observations shape the architecture we propose:

1. **Voxel-level spatial structure** of cMRI demands convolutional inductive biases — we adopt a 3D convolutional encoder–decoder backbone in the U-Net family (Ronneberger et al. 2015).
2. **Cine acquisition** provides a genuine temporal signal (≈25–30 phases per cycle); however, vanilla recurrent units suffer vanishing gradients over such horizons (Hochreiter & Schmidhuber 1997). We therefore use ConvLSTM (Shi et al. 2015), which preserves spatial structure while gating temporal flow.
3. **Small labelled set** (100 training patients) motivates self-supervised pretraining. We pretrain the encoder as part of a denoising convolutional autoencoder (Vincent et al. 2008) on the same volumes, then transfer weights to the segmentation task.

We further include attention-gated skip connections (Oktay et al. 2018) to focus the decoder on relevant regions, and a variational autoencoder (Kingma & Welling 2014) on encoder features to support probabilistic five-class diagnosis.

**Contributions.** (i) A coherent five-block architecture in which every block is justified by a property of cine cMRI rather than added arbitrarily. (ii) A seven-scenario ablation isolating the contribution of each block. (iii) A regularisation and optimisation study aligned with the methodological topics of SWE012.

---

## 2. Related Work

[TBD — to be filled by Student 5. Cover: U-Net (Ronneberger 2015), nnU-Net (Isensee 2021), ConvLSTM (Shi 2015), Attention U-Net (Oktay 2018), VAE (Kingma & Welling 2014), Denoising AE (Vincent 2008), ACDC challenge participants (Bernard 2018 §IV).]

---

## 3. Methods

### 3.1 Dataset and preprocessing

ACDC provides 100 training and 50 test patients, each with cine cMRI sequences and manual segmentations at end-diastole (ED) and end-systole (ES). The label map contains four classes — background (0), right-ventricle cavity (1), myocardium (2), and left-ventricle cavity (3).

**Preprocessing pipeline.** (i) Resample all volumes to isotropic 1.25 × 1.25 × 10 mm spacing using MONAI's `Spacingd`. (ii) Centre-crop to 224 × 224 in-plane and pad/trim slice axis to 16. (iii) Intensity-normalise per-volume via z-score on non-zero voxels. (iv) Apply 3D data augmentation at training time: random affine (rotation ±15°, scaling 0.9–1.1), Gaussian noise (σ ≤ 0.05), elastic deformation, random crop.

### 3.2 Architecture

We define the full model `F(x) = (ŷ_seg, ŷ_cls)` as the composition of five blocks.

**Block 1 — 3D-CNN encoder `E`.** Five-stage 3D residual encoder with channel widths {32, 64, 128, 256, 320}. Each stage: two `Conv3d → InstanceNorm → LeakyReLU` units with residual skip; downsampling via strided convolution. Receptive field at the bottleneck spans the full volume.

**Block 2 — Convolutional autoencoder `AE`.** Mirror decoder `D_AE` reconstructs the input volume. Trained in Stage A with denoising objective: input is corrupted by Gaussian noise (σ = 0.1) and dropout (p = 0.2) on voxels, and `D_AE(E(x̃))` is regressed to `x` with mean squared error. The encoder weights are then transferred to Stage B.

**Block 3 — ConvLSTM `R`.** Operates on the bottleneck feature map per time frame. Cell follows Shi et al. (2015): all gates (`i, f, o, g`) are 3D convolutions, preserving spatial structure. Hidden state is initialised to zero. Output sequence is fed to the segmentation decoder.

**Block 4 — Attention-gated skip connections.** At every decoder level, the upsampled feature is combined with the corresponding encoder skip through an additive attention gate (Oktay et al. 2018) producing a soft mask `α ∈ [0, 1]` that re-weights the skip feature.

**Block 5 — Variational autoencoder `V`.** Pooled encoder features are mapped to a Gaussian latent `(μ, log σ²)`. Reparameterised samples feed a small classifier MLP producing a 5-class disease prediction. Loss adds KL divergence to a standard normal prior plus cross-entropy.

### 3.3 Training and optimisation

Loss for segmentation is focal soft-Dice combined with cross-entropy, weighted as `L = 0.5·DiceFocal + 0.5·CE`. Loss for diagnosis is `L_cls = CE + β·KL` with `β = 1e-3`. Total loss is `L_seg + λ·L_cls` with `λ = 0.1`.

We train with Adam (lr = 1e-4, β = (0.9, 0.999)), warm-up of 5 epochs followed by cosine annealing over 100 epochs. Effective batch size 4 (gradient accumulation of 2 micro-batches of 2). Early stopping with patience 20 epochs on validation Dice.

### 3.4 Regularisation

Dropout (p = 0.2) on decoder feature maps; weight decay (L₂) `1e-5`; batch / instance normalisation throughout; on-the-fly data augmentation (§3.1); early stopping on validation Dice; ensemble of three random seeds for the headline number.

### 3.5 Evaluation

Five-fold patient-stratified cross-validation. Metrics: Dice per class, mean Dice, Hausdorff distance at the 95th percentile (HD95), balanced accuracy for diagnosis. Reported numbers are mean ± standard deviation across folds.

---

## 4. Experiments

### 4.1 Compute environment

All experiments were run on Google Colab T4 (16 GB VRAM) with PyTorch 2.x and MONAI 1.3.x.

### 4.2 Main results

*Placeholder — to be filled after experiments.*

| Method | LV Dice ↑ | RV Dice ↑ | Myo Dice ↑ | Mean ↑ | HD95 ↓ |
|---|---|---|---|---|---|
| 3D-UNet (baseline) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Full model (ours) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 4.3 Ablation study

We ablate one architectural choice at a time, keeping everything else fixed. All ablations are run on the same 5-fold split with seed 42.

| # | Variant | Mean Dice ↑ | Δ vs. full |
|---|---|---|---|
| A0 | Full model | [TBD] | — |
| A1 | − ConvLSTM (single frame) | [TBD] | [TBD] |
| A2 | − AE pretraining (random init) | [TBD] | [TBD] |
| A3 | − Attention gates (plain skip) | [TBD] | [TBD] |
| A4 | − VAE branch (deterministic classifier) | [TBD] | [TBD] |
| A5 | 2D-CNN encoder (slice-wise) | [TBD] | [TBD] |
| A6 | LSTM → GRU | [TBD] | [TBD] |
| A7 | − Data augmentation | [TBD] | [TBD] |

### 4.4 Optimiser comparison

| Optimiser | Best val Dice | Epochs to convergence |
|---|---|---|
| SGD + momentum (0.9) | [TBD] | [TBD] |
| SGD + Nesterov | [TBD] | [TBD] |
| RMSProp | [TBD] | [TBD] |
| Adam (default) | [TBD] | [TBD] |

### 4.5 Initialisation study

| Init | Final val Dice | Comment |
|---|---|---|
| He (Kaiming) | [TBD] | recommended for ReLU |
| Xavier (Glorot) | [TBD] | symmetric activations |
| Zero | [TBD] | failure mode — symmetry not broken |

### 4.6 Regularisation study

| Configuration | Val Dice | Train–Val gap |
|---|---|---|
| No regularisation | [TBD] | [TBD] |
| + Dropout (0.2) | [TBD] | [TBD] |
| + L₂ (1e-5) | [TBD] | [TBD] |
| + BatchNorm | [TBD] | [TBD] |
| + Augmentation | [TBD] | [TBD] |
| All (final) | [TBD] | [TBD] |

---

## 5. Discussion

[TBD — to be filled after results. Anchor points: (i) which block carries the largest empirical weight; (ii) bias–variance trade-off observed in regularisation sweeps; (iii) where the model fails (likely apex slices); (iv) limitations — small dataset, no domain shift evaluation.]

---

## 6. Conclusion

[TBD — single paragraph summarising contributions and one-sentence forward-look (e.g., extension to multi-centre M&Ms, transfer to multi-parametric prostate MRI).]

---

## References

1. O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?" in *IEEE Transactions on Medical Imaging*, vol. 37, no. 11, pp. 2514–2525, Nov. 2018. doi: 10.1109/TMI.2018.2837502
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
3. Isensee, F., Jaeger, P. F., Kohl, S. A., et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211.
4. Shi, X., Chen, Z., Wang, H., et al. (2015). Convolutional LSTM network: a machine learning approach for precipitation nowcasting. *NeurIPS*.
5. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
6. Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). Extracting and composing robust features with denoising autoencoders. *ICML*.
7. Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.
8. Oktay, O., Schlemper, J., Le Folgoc, L., et al. (2018). Attention U-Net: learning where to look for the pancreas. *MIDL*.
9. Kingma, D. P., & Ba, J. (2014). Adam: a method for stochastic optimization. *ICLR*.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## Appendix A — Architectural diagram

[Insert architecture figure from `figures/architecture.png` once finalised.]

## Appendix B — Hyperparameter grid

| Hyperparameter | Searched values | Selected |
|---|---|---|
| Learning rate | {1e-3, 5e-4, 1e-4, 5e-5} | 1e-4 |
| Batch size | {2, 4, 8} | 4 |
| Dropout | {0.0, 0.1, 0.2, 0.3, 0.5} | 0.2 |
| Weight decay | {0, 1e-6, 1e-5, 1e-4} | 1e-5 |
| LR schedule | {step, cosine, plateau} | cosine |
| ConvLSTM hidden | {32, 64, 128} | 64 |
| VAE latent dim | {16, 32, 64} | 32 |
