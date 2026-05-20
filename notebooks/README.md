# Notebooks

| File | Owner | Purpose |
|---|---|---|
| `00_EDA.ipynb` | Selvinaz Sayın | Patient discovery, distribution plots, slice grid, K-fold sanity check |
| `01_preprocessing.ipynb` | Selvinaz Sayın | Visualise MONAI transform pipeline before/after |
| `02_baseline_3D_UNet.ipynb` | Ege Karaurgan | Train baseline (no LSTM, no AE, no VAE) - anchor for ablation |
| `03_full_model.ipynb` | Mehmet Emin Akkaya | Stage A AE pretraining to Stage B segmentation + diagnosis fine-tune |
| `04_ablation_studies.ipynb` | Bayram Selim Yılmaz | Aggregate `ablation/results/all_runs.csv` into the tables of PAPER.md |

All notebooks start with a Colab/local detection cell and assume `PROJECT_ROOT` points at this repo (mounted on Drive in Colab).
