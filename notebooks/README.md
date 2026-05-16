# Notebooks

| File | Owner | Purpose |
|---|---|---|
| `00_EDA.ipynb` | Student 1 | Patient discovery, distribution plots, slice grid, K-fold sanity check |
| `01_preprocessing.ipynb` | Student 1 | Visualise MONAI transform pipeline before/after |
| `02_baseline_3D_UNet.ipynb` | Student 2 | Train baseline (no LSTM, no AE, no VAE) — anchor for ablation |
| `03_full_model.ipynb` | Student 4 | Stage A AE pretraining → Stage B segmentation + diagnosis fine-tune |
| `04_ablation_studies.ipynb` | Student 5 | Aggregate `ablation/results/all_runs.csv` into the tables of PAPER.md §4 |

All notebooks start with a Colab/local detection cell and assume `PROJECT_ROOT` points at this repo (mounted on Drive in Colab).
