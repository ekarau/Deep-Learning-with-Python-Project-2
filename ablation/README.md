# Ablation Studies

This folder documents the completed ablation results and the optional extra scenarios.

## Completed Results

The current CSV contains fold 0, seed 42, 100-epoch runs.

| Scenario | Owner | Best validation mean Dice | Status |
|---|---|---:|---|
| 3D U-Net baseline | Ege Karaurgan | **0.8576** | done |
| Full model | Bayram Selim Yilmaz | 0.7412 | done |
| A1 - no ConvLSTM | Vedat Efe Gezer | 0.7272 | done |
| A2 - no autoencoder pretraining | Mehmet Emin Akkaya | 0.8555 | done |
| A3 - no attention gates | Vedat Efe Gezer | 0.7244 | done |
| A4 - no VAE branch | Mehmet Emin Akkaya | 0.7364 | done |
| A7 - no augmentation | Selvinaz Sayin | 0.7411 | done |

## Interpretation

The strongest completed result is the plain 3D U-Net baseline. The most important ablation is A2: removing autoencoder pretraining almost recovers the baseline result. This suggests that the reconstruction objective used during denoising autoencoder pretraining was not aligned with the segmentation objective.

## Optional Extra Scenarios

These are configured but not required for the current submission because each full run is expensive on Colab.

| Scenario | Owner | Purpose |
|---|---|---|
| A5 - 2D-CNN backbone | Ege Karaurgan | Test the value of 3D inter-slice context |
| A6 - GRU instead of LSTM | Vedat Efe Gezer | Test recurrent gate complexity |

## CSV Format

`ablation/results/all_runs.csv` currently uses:

```text
run_id,scenario,fold,seed,best_val_dice_mean,epochs_trained
```

Use:

```bash
python -m src.utils.aggregate_ablation --csv ablation/results/all_runs.csv
```

to print a Markdown summary table.
