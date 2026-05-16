# Ablation Studies

Targets the **+15 ablation bonus** from the course rubric. Each scenario isolates
the empirical contribution of one architectural decision while keeping
everything else fixed (same 5-fold split, same seed=42, same epoch budget,
same optimiser).

## Scenarios

| # | Variant | What is removed | Hypothesis (Δ Mean Dice) | Owner | Status |
|---|---|---|---|---|---|
| A0 | Full model | — (baseline) | reference | Student 5 | ⬜ |
| A1 | − ConvLSTM | temporal block | −0.04 to −0.06 | Student 3 | ⬜ |
| A2 | − AE pretraining | self-supervised init | −0.02 to −0.03 | Student 4 | ⬜ |
| A3 | − Attention | gated skip refinement | −0.01 to −0.02 | Student 3 | ⬜ |
| A4 | − VAE branch | probabilistic latent | minimal seg Δ, large diag Δ | Student 4 | ⬜ |
| A5 | 2D-CNN backbone | 3D spatial context | **−0.06 to −0.08** (largest) | Student 2 | ⬜ |
| A6 | LSTM → GRU | gate mechanism | small Δ (research interest) | Student 3 | ⬜ |
| A7 | − Augmentation | data augmentation | −0.05 to −0.07 | Student 1 | ⬜ |

Each owner runs `python -m src.training.train --config configs/ablation/A{N}_*.yaml --fold {0..4}`.

## Auxiliary studies (also feed §4.4–4.6 of PAPER.md)

### Optimiser comparison (A8 group)

| Run | Optimiser | LR |
|---|---|---|
| A8a | SGD + momentum (0.9) | 1e-2 |
| A8b | SGD + Nesterov | 1e-2 |
| A8c | RMSProp | 1e-4 |
| A8d | Adam | 1e-4 |

### Initialisation study (A9 group)

| Run | Init |
|---|---|
| A9a | He (Kaiming) — default |
| A9b | Xavier (Glorot) |
| A9c | Zero (failure-mode demo) |

### Regularisation sweep (A10 group)

| Run | Configuration |
|---|---|
| A10a | no regularisation |
| A10b | + Dropout (0.2) |
| A10c | + L₂ (1e-5) |
| A10d | + BatchNorm |
| A10e | + Augmentation |
| A10f | all combined (final) |

## Output protocol

Every run appends one row to `ablation/results/all_runs.csv`:

```
run_id, scenario, fold, seed, val_dice_mean, val_dice_lv, val_dice_rv, val_dice_myo,
val_hd95, val_balanced_acc, train_time_min, gpu_peak_mb
```

A small script (`src/utils/aggregate_ablation.py`, future deliverable) reads this
CSV, computes mean ± std across folds, and emits the markdown table that lands
in `PAPER.md` §4.3.

## Statistical convention

- Mean ± std across 5 folds.
- Bold = best in column.
- Δ column = (full model) − (variant), so positive Δ means the removed block helped.
- 3 random seeds for A0 only; ablations on seed 42 to stay within compute budget.
