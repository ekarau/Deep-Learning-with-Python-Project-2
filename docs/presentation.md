# 3-Minute Presentation Script

**Target:** 3:00 minutes.  
**Deck:** `docs/presentation.pptx`, 8 slides.  
**Presenter:** one speaker; the rest of the team answers Q&A.

The deck has 8 slides, but the talk is still 3 minutes. Treat slides 2-5 as quick technical setup and spend the most time on slides 6-8.

---

## Slide 1 - Title (0:00-0:10)

Good morning. Our project is cardiac MRI segmentation and diagnosis on the ACDC benchmark. We implemented a five-block deep-learning pipeline: 3D U-Net, denoising autoencoder, ConvLSTM, attention gates, and a VAE diagnosis branch.

---

## Slide 2 - Clinical Context and Data (0:10-0:30)

The dataset contains cine cardiac MRI from 150 patients across five cardiac phenotypes. The segmentation targets are right ventricle, myocardium, and left ventricle. The important data property is that this is both volumetric and temporal: each patient has 3D anatomy and a cardiac cycle of roughly 28 frames.

---

## Slide 3 - Architecture (0:30-0:55)

Each block has a data-driven motivation. The 3D-CNN keeps inter-slice context. The autoencoder was added for self-supervised pretraining. ConvLSTM targets the temporal cardiac cycle. Attention gates re-weight skip connections, and the VAE branch gives a probabilistic diagnosis latent.

---

## Slide 4 - Mathematical Foundations (0:55-1:15)

The core operators are standard deep-learning tools applied to medical imaging. 3D convolution gives local spatial feature extraction, ConvLSTM preserves gradient flow through time with gates, and the VAE uses reparameterisation plus KL regularisation. The training loss combines Dice, cross-entropy, diagnosis loss, and KL.

---

## Slide 5 - Training Protocol (1:15-1:35)

Training uses Adam with learning rate 1e-4, warm-up, cosine annealing, dropout, weight decay, InstanceNorm, augmentation, and early stopping. The completed results are fold 0, seed 42, and 100 epochs. More folds are configured, but each run takes several hours on Colab.

---

## Slide 6 - Main Result (1:35-1:55)

The best completed model is the plain 3D U-Net baseline. It reaches 0.8576 mean validation Dice on fold 0. This is our safest operating result and the number we should emphasise as the strongest completed segmentation performance.

---

## Slide 7 - Ablation Analysis (1:55-2:40)

The most important finding is that the full five-block model does not beat the baseline. The full model reaches 0.7412, while removing autoencoder pretraining recovers performance to 0.8555. That means the denoising reconstruction objective probably learned features that are too smooth for sharp cardiac boundaries, especially myocardium. So the ablation does its job: it prevents us from claiming that more modules automatically mean better segmentation.

---

## Slide 8 - Discussion and Conclusion (2:40-3:00)

The conclusion is honest and defensible. We built the full pipeline, but the clean 3D U-Net is currently the strongest operating point. The main limitations are single-fold reporting and incomplete full-sequence temporal evaluation. Future work is to finish five-fold validation, feed full cine sequences to ConvLSTM, and test cross-centre generalisation.

Thank you.

---

## Cue Card

- Do not claim the full model beats the baseline.
- Strongest result: baseline 3D U-Net, 0.8576 Dice.
- Key ablation: no autoencoder pretraining, 0.8555 Dice.
- Main interpretation: objective mismatch between reconstruction and segmentation.
- If time is short, compress slides 3-5 and protect slide 7.

