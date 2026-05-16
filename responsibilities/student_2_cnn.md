# Student 2 — CNN Backbone Owner

**Name:** [Adınız Soyadınız]
**Student No:** [Öğrenci numarası]
**Primary axis:** 3D-CNN encoder/decoder, baseline U-Net, 2D-CNN ablation (A5).

## Deliverables

- [ ] `src/models/cnn3d.py` — `Encoder3D`, `Decoder3D` (already scaffolded; verify shapes).
- [ ] `src/models/cnn2d_ablation.py` — slice-wise 2D-CNN variant for A5.
- [ ] `notebooks/02_baseline_3D_UNet.ipynb` — train baseline (no LSTM, no AE, no VAE) and report Dice per class.
- [ ] Ablation A5 (`configs/ablation/A5_2d_cnn.yaml`) — 5 folds × 100 epochs.
- [ ] Receptive-field diagram for `PAPER.md` Appendix A.

## Conference paper sections to author

- §3.2 (Block 1 paragraph — 3D-CNN encoder + decoder)
- §4.2 main results table (baseline column)
- §4.5 initialisation study (A9 group)

## Presentation slot

- Slide 2 (0:20–1:00) of the 3-minute presentation: architecture + 5-block justification — backbone portion.

## Key technical points to internalise (for Q&A)

- Receptive field arithmetic (kernel × depth × stride compound effect).
- Why InstanceNorm over BatchNorm here (small batch size = 2 on Colab).
- He vs Xavier vs Zero init experimental evidence.
- 2D vs 3D context impact from A5 numbers.
