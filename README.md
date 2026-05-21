# Cardiac MRI Segmentation and Diagnosis on ACDC

This repository contains a deep-learning project for cardiac MRI segmentation on the ACDC benchmark. The task is to segment the right ventricle, myocardium, and left ventricle, with an additional diagnosis-oriented branch in the full model.

The main write-up is in `PAPER.md`, the engineering notes are in `REPORT.md`, and the 3-minute presentation deck is `presentation.pptx`.

---

## Current Result Summary

| Run | Fold | Seed | Best validation mean Dice |
|---|---:|---:|---:|
| 3D U-Net baseline | 0 | 42 | **0.8576** |
| Full five-block model | 0 | 42 | 0.7412 |
| No autoencoder pretraining | 0 | 42 | 0.8555 |
| No ConvLSTM | 0 | 42 | 0.7272 |
| No attention | 0 | 42 | 0.7244 |
| No VAE | 0 | 42 | 0.7364 |
| No augmentation | 0 | 42 | 0.7411 |

The strongest completed operating point is the plain 3D U-Net baseline.

---

## Repository Layout

```text
.
|-- PAPER.md
|-- REPORT.md
|-- requirements.txt
|-- ablation/
|   `-- results/all_runs.csv
|-- configs/
|   |-- baseline.yaml
|   |-- full_model.yaml
|   `-- ablation/
|-- presentation.pptx
|-- docs/
|-- figures/
|-- notebooks/
|-- references/
|-- responsibilities/
`-- src/
    |-- data/
    |-- models/
    |-- training/
    `-- utils/
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Colab or Linux:

```bash
pip install -r requirements.txt
```

---

## Training

Example baseline command:

```bash
python -m src.training.train --config configs/baseline.yaml --fold 1 --data-root data/training
```

A full 100-epoch run can take several hours on Colab, so extra folds are optional unless more GPU time is available.

---

## Citation

Any use of the ACDC database should cite:

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?", IEEE Transactions on Medical Imaging, 37(11), 2514-2525, 2018.
