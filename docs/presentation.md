# 3-Minute Presentation Script

**Target:** 3:00 minutes.  
**Deck:** `docs/presentation.pptx`, 6 slides.  
**Presenter:** one speaker; the rest of the team answers Q&A.

This script matches the 6-slide version of the deck. The talk should stay concise: dataset, architecture, result, ablation, conclusion.

---

## Slide 1 - Title (0:00-0:15)

Good morning. Our project is cardiac MRI segmentation on the ACDC benchmark. We built a five-block pipeline with a 3D U-Net backbone, denoising autoencoder pretraining, ConvLSTM, attention gates, and a VAE diagnosis head. The main point of the project is not only the model, but the ablation study that tells us which parts actually helped.

---

## Slide 2 - Problem (0:15-0:40)

ACDC contains 150 cine cardiac MRI patients across five phenotypes: normal, myocardial infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy, and abnormal right ventricle. The task is voxel-wise segmentation of background, right ventricle, myocardium, and left ventricle. The difficulty is that we only have 100 labelled training patients, so the model needs good inductive bias and regularisation.

---

## Slide 3 - Architecture (0:40-1:15)

The architecture has five blocks. The 3D U-Net preserves inter-slice anatomy. The denoising autoencoder provides self-supervised encoder pretraining. ConvLSTM is included for cine temporal modelling while keeping spatial feature maps. Attention gates re-weight skip features, and the VAE diagnosis head adds a probabilistic latent space. The optimisation uses Dice plus cross-entropy for segmentation, and classification cross-entropy plus KL for the VAE branch.

---

## Slide 4 - Result (1:15-1:50)

The strongest completed model is the 3D U-Net baseline. On fold 0 with seed 42 and 100 epochs, it reaches 0.8576 mean validation Dice. The full five-block model reaches 0.7412, and the no-autoencoder-pretraining variant reaches 0.8555. So the best completed operating point is the simpler discriminative baseline.

---

## Slide 5 - Ablation Analysis (1:50-2:35)

The key ablation is A2: removing autoencoder pretraining changes the score from 0.7412 to 0.8555. This almost fully recovers the baseline gap. Our interpretation is objective mismatch. Denoising reconstruction encourages smoother, lower-frequency features, while cardiac segmentation needs sharp boundaries, especially around the myocardium. This is why the ablation is useful: it prevents us from claiming that more blocks automatically produce a better model.

---

## Slide 6 - Conclusion (2:35-3:00)

What we can claim is clear: the project implements all required methodologies, uses a research-paper dataset, and includes an ablation study. What we cannot claim yet is full five-fold stability or full cine-sequence temporal superiority. The next step would be completing all folds, feeding the full cine sequence to ConvLSTM, and testing cross-centre generalisation.

Thank you.

---

## Cue Card

- Strongest number: 3D U-Net baseline, 0.8576 Dice.
- Full model: 0.7412 Dice.
- Key ablation: no AE pretraining, 0.8555 Dice.
- Main explanation: reconstruction pretraining and segmentation have mismatched objectives.
- Do not oversell ConvLSTM; say full temporal testing remains future work.

