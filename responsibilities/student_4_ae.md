# Student 4 — Autoencoder + VAE Owner

**Name:** [Adınız Soyadınız]
**Student No:** [Öğrenci numarası]
**Primary axis:** Convolutional autoencoder pretraining, VAE diagnosis branch, pretraining ablation (A2), VAE ablation (A4).

## Deliverables

- [ ] `src/models/autoencoder.py` — `ConvAutoencoder3D` denoising AE (scaffolded; verify reconstruction quality).
- [ ] `src/models/vae.py` — `VAEHead` (scaffolded; verify KL term scale).
- [ ] `notebooks/03_full_model.ipynb` — Stage A (AE pretraining) → Stage B (segmentation fine-tune).
- [ ] Ablation A2 (`configs/ablation/A2_no_pretraining.yaml`).
- [ ] Ablation A4 (`configs/ablation/A4_no_vae.yaml`).
- [ ] Reconstruction quality figure for `PAPER.md` §3.2 (original / corrupted / recon side by side).
- [ ] Latent space figure: t-SNE / UMAP of `mu` coloured by disease group.

## Conference paper sections to author

- §3.2 (Block 2 + Block 5 paragraphs — Conv-AE, VAE)
- §4.3 ablation table (rows A2, A4 narrative)
- §3.3 diagnosis loss derivation

## Presentation slot

- Slide 3 (1:00–1:40) of the 3-minute presentation: optimiser/regularisation/init choices — AE pretraining link.

## Key technical points to internalise (for Q&A)

- Reparameterisation trick (z = μ + σ ⊙ ε, ε ~ N(0, I)) and why it enables backprop through sampling.
- KL divergence between approximate posterior and N(0, I) — closed form.
- Denoising AE vs vanilla AE (why noise robustness yields better features).
- Why pretraining helps in low-data regime (loss landscape initialisation).
