# Student 3 — Recurrent + Attention Owner

**Name:** [Adınız Soyadınız]
**Student No:** [Öğrenci numarası]
**Primary axis:** ConvLSTM module, attention gates, GRU ablation (A6), LSTM ablation (A1), attention ablation (A3).

## Deliverables

- [ ] `src/models/convlstm.py` — `ConvLSTMCell`, `ConvLSTM` (scaffolded; verify gradient flow over long sequences).
- [ ] `src/models/attention.py` — `AttentionGate` (scaffolded; verify spatial alignment after upsampling).
- [ ] `src/models/gru_ablation.py` — ConvGRU variant for A6.
- [ ] Ablation A1 (`configs/ablation/A1_no_convlstm.yaml`).
- [ ] Ablation A3 (`configs/ablation/A3_no_attention.yaml`).
- [ ] Ablation A6 (`configs/ablation/A6_gru.yaml`).
- [ ] Vanishing-gradient illustration figure for `PAPER.md` §3.2 (gradient norms vs depth).

## Conference paper sections to author

- §3.2 (Block 3 + Block 4 paragraphs — ConvLSTM, attention)
- §4.3 ablation table (rows A1, A3, A6 narrative)

## Presentation slot

- Slide 2 (0:20–1:00) of the 3-minute presentation: architecture — temporal + attention justification.

## Key technical points to internalise (for Q&A)

- Why LSTM gates solve vanishing gradient (multiplicative path through cell state).
- ConvLSTM vs fully-connected LSTM (parameter scaling, spatial preservation).
- Additive attention gate formulation (sigmoid(ψ(σ(θ·skip + φ·gate)))).
- Why GRU sometimes matches LSTM despite fewer gates (A6 evidence).
