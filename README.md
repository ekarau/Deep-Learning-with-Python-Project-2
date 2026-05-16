# DL Project 1 — Cardiac MRI Segmentation on ACDC

**Course:** SWE012 — Deep Learning with Python · İstinye University · Spring 2026
**Instructor:** Asst. Prof. Dr. Yiğit Bekir Kaya
**Dataset:** [ACDC — Automated Cardiac Diagnosis Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/) (Bernard et al. 2018, *IEEE TMI*)
**Framework:** PyTorch + MONAI
**Hardware:** Google Colab (Free / Pro)
**Seed:** 42

---

## What this repository is

A 3D + temporal cardiac MRI segmentation and diagnosis pipeline that integrates the three architectural blocks required by the course (CNN, RNN/LSTM, Autoencoder) plus two bonus blocks (VAE, Attention) in a *methodologically justified* way — i.e. each block is included because the data structure demands it, not as a stack-everything exercise.

A full conference-paper-style methods write-up lives in [`PAPER.md`](PAPER.md). This README is the engineering landing page (how to run, file map, team responsibilities).

---

## Targeted bonus points (per course rubric, max +60)

| Bonus | Mechanism | Status |
|---|---|---|
| +15 — Research-paper dataset | ACDC published in *IEEE TMI* 2018 (Bernard et al.) | ✅ Locked in |
| +15 — 5+ architectural blocks | 3D-CNN, Conv-AE, ConvLSTM, Attention, VAE | ✅ Planned |
| +15 — Ablation study | 7-scenario ablation matrix, paired with 3 random seeds | 🔧 In progress |
| +15 — Conference-paper-style documentation | `PAPER.md` — Abstract → References | 🔧 In progress |

---

## Architecture overview

```
Input: cine cardiac MRI volume  (B × 1 × D × H × W × T)
                │
                ▼
   ┌────────────────────────────────┐
   │ Stage A — Self-supervised      │
   │ Convolutional Autoencoder      │  ← Week 13
   │ (denoising pretraining)        │
   └────────────────────────────────┘
                │
                ▼  pretrained encoder weights
   ┌────────────────────────────────┐
   │ Stage B — Segmentation network │
   │  • 3D-CNN encoder              │  ← Week 9
   │  • ConvLSTM over time axis     │  ← Week 10
   │  • Attention skip connections  │  ← bonus
   │  • 3D-CNN decoder              │
   └────────────────────────────────┘
                │
                ├──► Segmentation head → LV / RV / Myo masks
                │
                ▼
   ┌────────────────────────────────┐
   │ Stage C — Diagnosis branch     │
   │  • VAE on encoder features     │  ← bonus
   │  • 5-class classifier          │  NOR/MINF/DCM/HCM/ARV
   └────────────────────────────────┘
```

Each block's justification, in one line:

| Block | Why it must be there |
|---|---|
| 3D-CNN | Voxel-wise spatial features; respects 3D anatomy. |
| Conv-AE | Self-supervised pretraining mitigates the small-data regime (100 patients). |
| ConvLSTM | Cardiac cycle is genuinely temporal; gates solve vanishing gradient over T frames. |
| Attention | Encoder-decoder skip refinement; focuses on relevant regions. |
| VAE | Probabilistic latent for disease classification + uncertainty. |

Full mathematical justification: see `PAPER.md` §3.

---

## Repository layout

```
DL_Project_1_ACDC/
├── README.md                 ← you are here
├── PAPER.md                  ← conference-style methods write-up (+15 doc bonus)
├── REPORT.md                 ← detailed technical report (Turkish-friendly)
├── requirements.txt
├── .gitignore
├── data/                     ← ACDC training/testing (gitignored)
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_baseline_3D_UNet.ipynb
│   ├── 03_full_model.ipynb
│   └── 04_ablation_studies.ipynb
├── src/
│   ├── data/                 ← Dataset + transforms + splits
│   ├── models/               ← cnn3d, convlstm, autoencoder, vae, full_model
│   ├── training/             ← train loop, losses, metrics
│   └── utils/
├── configs/
│   ├── baseline.yaml
│   ├── full_model.yaml
│   └── ablation/             ← one YAML per ablation scenario
├── ablation/
│   └── results/              ← CSV outputs + figures
├── responsibilities/         ← per-student task contracts
├── figures/                  ← generated plots for PAPER.md
├── checkpoints/              ← saved weights (gitignored)
├── logs/                     ← TensorBoard logs (gitignored)
└── docs/
    └── presentation.md       ← 3-minute presentation script
```

---

## Setup

### Option A — Google Colab (recommended)

Open the notebook of interest, then:

```python
!pip install -q monai[all] nibabel itk tensorboard pyyaml
!git clone <repo-url> && cd DL_Project_1_ACDC
```

### Option B — Local

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Data download

1. Register at https://www.creatis.insa-lyon.fr/Challenge/acdc/
2. Download `training.zip` and `testing.zip`.
3. Extract into `data/` such that you have `data/training/patient001/...`.
4. Run `notebooks/00_EDA.ipynb` for a sanity check.

---

## Team responsibilities

See `responsibilities/` — one file per student with their concrete task contract.

| Student | Primary | Files |
|---|---|---|
| 1 | Data + preprocessing + EDA | `src/data/*`, `00_EDA.ipynb`, `01_preprocessing.ipynb` |
| 2 | 3D-CNN baseline + decoder | `src/models/cnn3d.py`, `02_baseline_3D_UNet.ipynb` |
| 3 | ConvLSTM integration + Attention | `src/models/convlstm.py`, `src/models/attention.py` |
| 4 | Autoencoder pretraining + VAE | `src/models/autoencoder.py`, `src/models/vae.py` |
| 5 | Ablation studies + PAPER.md + presentation | `04_ablation_studies.ipynb`, `PAPER.md`, `docs/` |

---

## Reproducibility

| | |
|---|---|
| Random seed | 42 (numpy, torch, torch.cuda) |
| Cross-validation | 5-fold patient-stratified |
| Determinism | `torch.backends.cudnn.deterministic = True` |
| Logging | TensorBoard + per-experiment CSV row in `ablation/results/` |

---

## Citation

If you reference ACDC, cite:

```bibtex
@article{bernard2018deep,
  title={Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: Is the problem solved?},
  author={Bernard, Olivier and Lalande, Alain and Zotti, Clement and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={37}, number={11}, pages={2514--2525},
  year={2018},
  doi={10.1109/TMI.2018.2837502}
}
```
