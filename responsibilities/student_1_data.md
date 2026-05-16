# Student 1 — Data Pipeline Owner

**Name:** [Adınız Soyadınız]
**Student No:** [Öğrenci numarası]
**Primary axis:** Data, preprocessing, EDA, augmentation ablation (A7).

## Deliverables

- [ ] `src/data/dataset.py` — `ACDCPatient`, `discover_patients`, `ACDCDataset`, K-fold splits.
- [ ] `src/data/transforms.py` — MONAI train + validation transform pipelines.
- [ ] `notebooks/00_EDA.ipynb` — distribution of disease groups, volume sizes, voxel spacing histogram, class pixel ratios.
- [ ] `notebooks/01_preprocessing.ipynb` — visualise transform pipeline before/after on 3 patients.
- [ ] Ablation A7 (`configs/ablation/A7_no_augmentation.yaml`) — 5 folds × 100 epochs.
- [ ] Figure for `PAPER.md` §3.1: 2×3 grid of axial slices + ground truth overlays.

## Conference paper sections to author

- §3.1 Dataset and preprocessing
- Appendix B partial — preprocessing hyperparameter table

## Presentation slot

- Slide 1 (0:00–0:20) of the 3-minute presentation: problem + dataset overview.

## Key technical points to internalise (for Q&A)

- Why z-score on non-zero voxels (background bias).
- Why isotropic resampling matters in ACDC (slice-thickness 5–10 mm).
- Bias-variance link from augmentation ablation.
