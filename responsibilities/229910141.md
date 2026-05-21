# Ege Karaurgan - CNN Backbone Owner

**Name:** Ege Karaurgan  
**Student No:** 229910141  
**Primary axis:** 3D-CNN encoder/decoder and baseline U-Net.

## Deliverables

- [x] `src/models/cnn3d.py` - 3D encoder and decoder.
- [x] `configs/baseline.yaml` - baseline training configuration.
- [x] Baseline result recorded in `ablation/results/all_runs.csv`: 0.8576 mean Dice on fold 0.
- [x] `figures/baseline_curves.png` - baseline training curves.
- Optional extra: complete A5 2D-CNN implementation/run if extra compute time is available.

## Paper / Presentation Contribution

- 3D-CNN backbone explanation.
- Main result: 3D U-Net baseline is the strongest completed run.

## Q&A Notes

- 3D convolution preserves inter-slice context.
- InstanceNorm is preferred for small batch sizes.
- The baseline result is the safest claim in the current project.
