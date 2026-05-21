# Vedat Efe Gezer - Recurrent + Attention Owner

**Name:** Vedat Efe Gezer  
**Student No:** 229910158  
**Primary axis:** ConvLSTM, attention gates, and related ablations.

## Deliverables

- [x] `src/models/convlstm.py` - ConvLSTM implementation.
- [x] `src/models/attention.py` - attention gate implementation.
- [x] Ablation A1 config: `configs/ablation/A1_no_convlstm.yaml`.
- [x] Ablation A3 config: `configs/ablation/A3_no_attention.yaml`.
- [x] A1 result recorded: 0.7272 mean Dice on fold 0.
- [x] A3 result recorded: 0.7244 mean Dice on fold 0.
- Optional extra: complete A6 GRU implementation/run if extra compute time is available.

## Paper / Presentation Contribution

- ConvLSTM and attention explanation.
- Ablation interpretation for A1 and A3.

## Q&A Notes

- LSTM gates help preserve gradient flow through a temporal sequence.
- ConvLSTM keeps spatial structure because gates are convolutional.
- Current temporal claims should be conservative because full cine-sequence evaluation is not fully completed.
