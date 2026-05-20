# Technical Design Notes

This report is the engineering companion to `PAPER.md`. It summarises the implemented architecture, training plan, ablation matrix, and current project status.

---

## 1. Architecture

### Block 1 - 3D-CNN encoder/decoder

The baseline model is a 3D U-Net-style encoder-decoder. It uses volumetric convolutions so that neighbouring MRI slices contribute to the segmentation decision. This is the strongest completed model in the current results, reaching 0.8576 mean validation Dice on fold 0.

### Block 2 - Denoising autoencoder

The autoencoder was designed to pretrain the encoder in a self-supervised way. Inputs are corrupted with Gaussian noise and voxel dropout, then reconstructed with MSE loss. The ablation result is negative but informative: removing this pretraining raises Dice from 0.7412 to 0.8555, suggesting objective mismatch between reconstruction and segmentation.

### Block 3 - ConvLSTM

ConvLSTM is included to model the cine cardiac cycle while preserving spatial feature maps. The current training logs mainly evaluate single-frame ED/ES behaviour, so the full temporal benefit remains a future experiment.

### Block 4 - Attention gates

Attention gates re-weight encoder skip features before decoder concatenation. In the completed fold-0 ablation, removing attention changes Dice from 0.7412 to 0.7244.

### Block 5 - VAE diagnosis branch

The VAE branch maps pooled encoder features to a latent distribution for diagnosis. In segmentation Dice, removing the VAE changes the score only slightly, from 0.7412 to 0.7364.

---

## 2. Training Protocol

- Dataset: ACDC cardiac MRI.
- Optimiser: Adam.
- Learning rate: 1e-4.
- Epochs: 100.
- Fold/seed completed: fold 0, seed 42.
- Regularisation: dropout, weight decay, InstanceNorm, augmentation, early stopping.
- Main metric: foreground mean validation Dice.

---

## 3. Current Results

| Run | Mean validation Dice |
|---|---:|
| 3D U-Net baseline | **0.8576** |
| Full model | 0.7412 |
| No autoencoder pretraining | 0.8555 |
| No ConvLSTM | 0.7272 |
| No attention | 0.7244 |
| No VAE | 0.7364 |
| No augmentation | 0.7411 |

The main conclusion is that the 3D U-Net baseline is currently the safest operating point. The full model is implemented, but it does not outperform the baseline under the completed fold-0 experiment.

---

## 4. Compute Budget

The command below is the normal Colab training entry point:

```bash
python -m src.training.train --config configs/baseline.yaml --fold 1 --data-root data/training
```

A full 100-epoch run can take approximately 5-7 hours depending on the GPU. Extra folds and optimiser sweeps are therefore optional improvements rather than blocking deliverables.

---

## 5. Submission Status

The repository contains code, configs, ablation CSV results, figures, a paper draft, responsibility files, and a PowerPoint presentation. The local copy is not a Git repository because it appears to be extracted from a ZIP; pushing changes requires copying these files into a cloned Git repository or re-cloning the repository and applying the modified files there.

