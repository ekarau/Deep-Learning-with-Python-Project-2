# Team Responsibilities

Five-person split. Each student owns one major axis and contributes to ablation runs.

| Student | Primary axis | Files |
|---|---|---|
| Student 1 | Data + preprocessing + EDA + augmentation ablation | `src/data/*`, `notebooks/00_EDA.ipynb`, `notebooks/01_preprocessing.ipynb` |
| Student 2 | 3D-CNN backbone + decoder + 2D-CNN ablation | `src/models/cnn3d.py`, `notebooks/02_baseline_3D_UNet.ipynb` |
| Student 3 | ConvLSTM + Attention + GRU ablation | `src/models/convlstm.py`, `src/models/attention.py` |
| Student 4 | Conv-Autoencoder pretraining + VAE diagnosis | `src/models/autoencoder.py`, `src/models/vae.py` |
| Student 5 | Ablation aggregation + `PAPER.md` + 3-minute presentation | `notebooks/04_ablation_studies.ipynb`, `PAPER.md`, `docs/presentation.md` |

Each student fills their own `student_<no>.md` with:
- their primary deliverables
- the conference paper sections they wrote
- the ablation runs they executed

This file is what the instructor sees to attribute work.
