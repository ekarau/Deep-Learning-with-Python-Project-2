# Student 5 — Ablation Aggregation + Paper + Presenter

**Name:** [Adınız Soyadınız]
**Student No:** [Öğrenci numarası]
**Primary axis:** Ablation aggregation, conference-paper write-up, 3-minute presentation.

## Deliverables

- [ ] `src/utils/aggregate_ablation.py` — read `ablation/results/all_runs.csv`, produce markdown tables for `PAPER.md`.
- [ ] `notebooks/04_ablation_studies.ipynb` — visualise all ablation results (bar plots with error bars, paired comparison).
- [ ] `PAPER.md` — §1, §2, §5, §6, References, Abstract (in collaboration with whole group).
- [ ] `docs/presentation.md` — full 3-minute script, time-coded.
- [ ] Final architecture figure (`figures/architecture.png`).
- [ ] Final results figure (`figures/main_results.png`).
- [ ] Submission package: GitHub URL + PDF/PPTX presentation to Blackboard.

## Conference paper sections to author

- §1 Introduction
- §2 Related Work (annotated bibliography from `notebooks/lit_review.ipynb` if needed)
- §4 Experiments — assemble all tables from per-owner results
- §5 Discussion
- §6 Conclusion
- References

## Presentation slot

- **Presenter for the full 3 minutes.** Per instructor's email: only one person presents, reading from notes = 0 in the fluency component.

## Key technical points to internalise (for Q&A)

Required jargon coverage (instructor rubric, 40% of presentation grade):

- Bias-variance tradeoff — augmentation ablation A7 evidence.
- Vanishing/exploding gradient — LSTM justification (A1).
- Ill-conditioning, saddle points — momentum / Nesterov motivation.
- AdaGrad / RMSProp / Adam — optimiser study A8.
- He vs Xavier init — A9.
- L1/L2 penalties, dropout (probabilistic interpretation) — regularisation A10.
- Early stopping, batch norm.
- Ensemble methods — final result uses 3-seed ensemble.

## Time-coded presentation script (target 3:00 ± 0:15)

| Time | Topic | Slide |
|---|---|---|
| 0:00 – 0:20 | Problem + ACDC dataset + clinical motivation | 1 |
| 0:20 – 1:00 | 5-block architecture + per-block justification | 2 |
| 1:00 – 1:40 | Optimiser / regularisation / initialisation choices | 3 |
| 1:40 – 2:30 | Ablation table — 2 most-important deltas | 4 |
| 2:30 – 3:00 | Conclusion + limitations + forward look | 5 |
