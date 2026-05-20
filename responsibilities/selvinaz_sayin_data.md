# Selvinaz Sayın - Data Pipeline Owner

**Name:** Selvinaz Sayın  
**Student No:** 220901755  
**Primary axis:** Data, preprocessing, EDA, augmentation ablation (A7).

## Deliverables

- [x] `src/data/dataset.py` - ACDC patient discovery, dataset wrapper, and K-fold split support.
- [x] `src/data/transforms.py` - MONAI-style train and validation transform pipelines.
- [x] `notebooks/00_EDA.ipynb` - EDA notebook.
- [x] `notebooks/00_EDA_colab.ipynb` - Colab-ready EDA notebook.
- [x] EDA figures in `figures/`.
- [x] Ablation A7 config: `configs/ablation/A7_no_augmentation.yaml`.
- [x] A7 result recorded in `ablation/results/all_runs.csv`: 0.7411 mean Dice on fold 0.

## Paper / Presentation Contribution

- Dataset and preprocessing explanation.
- Slide 1: problem, dataset, and clinical motivation.

## Q&A Notes

- Z-score should be computed on non-zero voxels to avoid background bias.
- 3D preprocessing matters because ACDC volumes have inter-slice structure.
- A7 did not change fold-0 Dice meaningfully, so it should be discussed cautiously.
