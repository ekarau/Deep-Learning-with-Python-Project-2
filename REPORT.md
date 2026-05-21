# Technical Report

This report documents how the project addresses the course methodologies: convolutional neural networks, recurrent modelling with LSTM-style gates, autoencoder models, hyperparameter tuning, regularisation, simultaneous model integration, and empirical comparison.

---

## 1. Dataset Selection

We use the Automated Cardiac Diagnosis Challenge (ACDC) cardiac MRI dataset. It is sourced from a published research benchmark rather than a toy dataset or a generic Kaggle dataset. This is appropriate for the course project because it contains real medical images, voxel-level segmentation labels, and clinically meaningful diagnostic categories.

The task is to segment the right ventricle, myocardium, and left ventricle from cine cardiac MRI. The dataset is challenging because the inputs are volumetric, cardiac structures are small relative to the background, and the cine acquisition contains temporal information from the cardiac cycle.

Dataset rationale:

- It is not MNIST/Fashion-MNIST or another overly common benchmark.
- It is linked to a research paper, which supports the dataset-quality bonus.
- It naturally motivates CNN, recurrent, and autoencoder components.
- It supports both segmentation and diagnosis-oriented modelling.

---

## 2. Methodologies Applied

### 2.1 Convolutional Neural Network: 3D U-Net Baseline

The CNN component is a 3D U-Net-style encoder-decoder. We selected 3D convolutions because cardiac MRI volumes contain inter-slice anatomical continuity. A 2D slice-wise model would ignore part of this spatial context.

Role in the project:

- Extract volumetric spatial features.
- Downsample the image into a compact bottleneck representation.
- Decode the representation back into voxel-level class predictions.

The 3D U-Net baseline is the strongest completed model, reaching 0.8576 mean validation Dice on fold 0.

### 2.2 Recurrent Modelling: ConvLSTM

The recurrent component is a ConvLSTM block. It was included because cine MRI has a temporal cardiac cycle. Standard fully connected LSTMs lose spatial layout, while ConvLSTM keeps spatial feature maps and applies gates through convolution.

Role in the project:

- Model temporal dependencies across cardiac phases.
- Preserve spatial structure while using LSTM-style gating.
- Address vanishing-gradient issues more effectively than a plain RNN.

The completed fold-0 ablation shows that removing ConvLSTM gives 0.7272 Dice compared with 0.7412 for the full model.

### 2.3 Autoencoder: Denoising Autoencoder Pretraining

The autoencoder component is a denoising 3D convolutional autoencoder. It corrupts inputs using noise/dropout and reconstructs the clean volume. The encoder weights can then initialise the segmentation model.

Role in the project:

- Learn self-supervised representations from the same MRI volumes.
- Improve robustness in a small labelled-data regime.
- Provide the required AE methodology in a coherent medical-imaging setting.

The result was informative but negative: removing autoencoder pretraining increases Dice from 0.7412 to 0.8555. This suggests that reconstruction pretraining was not aligned with sharp-boundary segmentation in the completed run.

### 2.4 Attention Gates

Attention gates are added to skip connections. They re-weight encoder features before concatenation with decoder features, helping the decoder focus on relevant cardiac regions instead of passing all skip activations unchanged.

This is an additional block beyond the required three methodologies.

### 2.5 Variational Autoencoder Diagnosis Branch

The VAE branch maps pooled encoder features into a probabilistic latent distribution. It uses the reparameterisation trick and KL regularisation, then feeds a classifier for diagnosis-oriented prediction.

This is another additional block, bringing the project to five distinct model blocks:

1. 3D-CNN encoder-decoder
2. Denoising autoencoder
3. ConvLSTM
4. Attention gates
5. VAE diagnosis branch

---

## 3. Simultaneous Integration

The full model integrates all blocks in one architecture:

```text
Input MRI volume
  -> 3D-CNN encoder
  -> optional ConvLSTM temporal processing
  -> attention-gated decoder skip connections
  -> segmentation logits
  -> VAE diagnosis branch from bottleneck features
```

The denoising autoencoder is used as a pretraining stage, then its encoder weights are transferred to the full segmentation/diagnosis model. Segmentation and diagnosis losses are optimised together when the VAE branch is enabled.

---

## 4. Hyperparameters and Tuning

Selected hyperparameters:

| Hyperparameter | Selected value | Rationale |
|---|---:|---|
| Optimiser | Adam | Stable and fast convergence for noisy medical-image training |
| Learning rate | 1e-4 | Conservative value for 3D models and small batch sizes |
| Epochs | 100 | Enough for convergence within Colab compute limits |
| Effective batch size | 4 | Fits GPU memory while using gradient accumulation |
| Dropout | 0.2 | Reduces overfitting in a small-data regime |
| Weight decay | 1e-5 | L2 regularisation without dominating the loss |
| Scheduler | warm-up + cosine annealing | Stabilises early training and gradually lowers LR |
| VAE latent dim | 32 | Compact latent representation for diagnosis branch |

The codebase supports additional optimiser choices: SGD, Nesterov SGD, RMSProp, Adam, and AdamW. It also supports ablation configs for removing individual architectural components. Because each 100-epoch run can take several hours on Colab, the completed table reports fold 0 with seed 42.

---

## 5. Regularisation Techniques

Regularisation methods used:

- Denoising autoencoder corruption for representation learning.
- Dropout in decoder feature maps.
- L2 weight decay.
- InstanceNorm, preferred over BatchNorm because the effective batch size is small.
- Data augmentation through affine/noise/deformation-style transforms.
- Early stopping based on validation Dice.
- Patient-level fold splitting to reduce leakage risk.

The no-augmentation ablation reaches 0.7411, which is nearly identical to the full model in fold 0. This should be interpreted cautiously because only fold 0 is reported.

---

## 6. Evaluation Criteria

The main metric is foreground mean Dice score for segmentation. Dice is appropriate because the target is voxel-level overlap between predicted and ground-truth cardiac structures.

The project also discusses HD95 and balanced accuracy as relevant metrics, but the completed CSV records the best validation mean Dice for the available fold-0 runs.

---

## 7. Performance Comparison

| Run | Mean validation Dice |
|---|---:|
| 3D U-Net baseline | **0.8576** |
| No autoencoder pretraining | 0.8555 |
| Full model | 0.7412 |
| No augmentation | 0.7411 |
| No VAE | 0.7364 |
| No ConvLSTM | 0.7272 |
| No attention | 0.7244 |

Interpretation:

- The plain 3D U-Net baseline is currently the safest operating point.
- The full model is implemented but does not outperform the baseline in the completed fold-0 run.
- The most important ablation is autoencoder pretraining: removing it nearly restores baseline performance.
- This suggests objective mismatch between denoising reconstruction and sharp-boundary segmentation.

---

## 8. Compute Budget and Limitations

Example Colab command:

```bash
python -m src.training.train --config configs/baseline.yaml --fold 1 --data-root data/training
```

A full 100-epoch run can take approximately 5-7 hours depending on the GPU. The reported results are therefore presented as completed fold-0 experiments rather than full five-fold averages.

Limitations:

- Not all five folds are completed.
- A5 2D-CNN and A6 GRU are configured as optional extra runs.
- Full cine-sequence ingestion through ConvLSTM should be expanded in future work.

---

## 9. Submission Artifacts

The repository includes:

- `PAPER.md`: conference-style explanation.
- `REPORT.md`: methodology and evaluation report.
- `presentation.pptx`: 3-minute presentation deck.
- `responsibilities/`: student-number-named responsibility files.
- `ablation/results/all_runs.csv`: completed ablation results.
- `figures/`: generated plots and MRI visualisations.
- `src/`: implementation code.
