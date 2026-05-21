# Mehmet Emin Akkaya - Autoencoder + VAE Owner

**Name:** Mehmet Emin Akkaya  
**Student No:** 2309011036  
**Primary axis:** Denoising autoencoder pretraining and VAE diagnosis branch.

## Deliverables

- [x] `src/models/autoencoder.py` - denoising 3D autoencoder.
- [x] `src/models/vae.py` - VAE head.
- [x] `src/training/pretrain_ae.py` - pretraining entry point.
- [x] Ablation A2 config: `configs/ablation/A2_no_pretraining.yaml`.
- [x] Ablation A4 config: `configs/ablation/A4_no_vae.yaml`.
- [x] A2 result recorded: 0.8555 mean Dice on fold 0.
- [x] A4 result recorded: 0.7364 mean Dice on fold 0.

## Paper / Presentation Contribution

- Autoencoder, VAE, and objective-mismatch discussion.
- Key finding: removing autoencoder pretraining almost recovers baseline performance.

## Q&A Notes

- VAE uses the reparameterisation trick: z = mu + sigma * epsilon.
- KL divergence regularises the latent distribution.
- Denoising pretraining did not help segmentation in the completed run, likely because reconstruction and boundary segmentation have different objectives.
