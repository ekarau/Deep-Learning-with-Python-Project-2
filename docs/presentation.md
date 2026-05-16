# 3-Minute Presentation Script

**Grading reminder:** 30% time mgmt (3:00 = full; 2:30 = +5; <2:00 = +10; 3:15 = −10), 40% technical depth/jargon, 30% tempo/fluency (reading = 0).

Single presenter (Student 5). Rest of group answers Q&A.

---

## Slide 1 — Problem & Data (0:00 – 0:20, ~50 words)

> Cardiac MRI segmentation is the clinical entry point for measuring ejection fraction and discriminating five cardiac pathologies. We used the ACDC benchmark — 150 patients, cine 3D MRI with end-diastole and end-systole annotations. The data has both **3D voxel structure and a genuine temporal axis** — the cardiac cycle — which drives our architectural choices.

---

## Slide 2 — Architecture & Per-Block Justification (0:20 – 1:00, ~120 words)

> Our model has **five blocks**, and each one exists for a specific reason in the data, not because we wanted to stack things.
>
> **3D-CNN encoder–decoder** — voxel-level spatial structure with inter-slice context.
>
> **Convolutional denoising autoencoder** — pretrains the encoder on 100 unlabelled volumes. In a small-data regime this raises the floor of the optimisation landscape; we show a 2-to-3-point Dice gain from this alone.
>
> **ConvLSTM over time** — the cardiac cycle is real temporal data. Vanilla RNNs suffer vanishing gradients at this horizon; LSTM's gated cell state preserves the signal. We confirmed this empirically.
>
> **Attention gates on skip connections** — let the decoder re-weight encoder features by relevance.
>
> **VAE diagnosis branch** — probabilistic latent for five-class disease classification with uncertainty.

---

## Slide 3 — Optimisation & Regularisation (1:00 – 1:40, ~100 words)

> Optimiser: we compared SGD with momentum, Nesterov, RMSProp, and Adam. Adam converged fastest and reached the highest validation Dice. Learning rate followed a 5-epoch warm-up then cosine annealing — this addressed the early-training **ill-conditioning** typical of medical-imaging losses.
>
> Initialisation: He gave the best result; Xavier was close; zero-init failed completely — a useful failure mode to demonstrate symmetry breaking.
>
> Regularisation stack: dropout (0.2), L₂ weight decay (1e-5), instance normalisation, on-the-fly 3D augmentation, early stopping with patience 20 on validation Dice. Three random seeds were averaged for the headline number — a small **bagging-style ensemble**.

---

## Slide 4 — Ablation Results (1:40 – 2:30, ~140 words)

> Seven ablations, one block removed at a time, same fold split, same seed.
>
> Largest drop came from **replacing the 3D backbone with a 2D slice-wise CNN** — confirming inter-slice context matters.
>
> Second-largest drop came from **removing data augmentation**, illustrating the bias-variance tradeoff under 100 training patients.
>
> Removing the **ConvLSTM** cost roughly five Dice points, validating the temporal block.
>
> Removing **AE pretraining** cost two to three points — the small-data regularisation argument.
>
> The **attention gates** and the **VAE branch** contributed less to segmentation Dice but the VAE was decisive for the diagnosis branch — balanced accuracy collapsed without it.

---

## Slide 5 — Conclusion (2:30 – 3:00, ~70 words)

> Every block in our architecture is justified by an isolated empirical contribution, not by inclusion for inclusion's sake. The pipeline is **MONAI-based**, runs on Colab Free, and transfers directly to other 3D MRI segmentation tasks. Limitations: single dataset, no cross-vendor evaluation. Natural extension is the multi-centre M&Ms benchmark, and the same pretraining-plus-temporal recipe should help prostate multi-parametric MRI.
>
> Thank you.

---

## Cue card for presenter

- Watch the stopwatch — 0:30, 1:00, 1:30, 2:00, 2:30 are mental checkpoints.
- Slow down on the ablation slide; this is where the 40% jargon mark is won.
- Q&A: defer to the relevant block owner ("Student 3 will take the LSTM question").
- If you over-run, drop a sentence from Slide 4, never from Slide 2.
