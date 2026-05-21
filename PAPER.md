# Self-Supervised 3D-CNN + ConvLSTM + VAE for Cine Cardiac MRI Segmentation and Diagnosis: An Ablation Study on ACDC

**Authors:** Selvinaz Sayın (220901755), Ege Karaurgan (229910141), Vedat Efe Gezer (229910158), Mehmet Emin Akkaya (2309011036), Bayram Selim Yılmaz (2309011053)  
**Affiliation:** Department of Software Engineering, Istinye University, Istanbul, Turkiye  
**Course:** SWE012 - Deep Learning with Python (Spring 2026)  
**Instructor:** Asst. Prof. Dr. Yigit Bekir Kaya

---

## Abstract

We present a unified deep architecture for joint segmentation and diagnosis on the Automated Cardiac Diagnosis Challenge (ACDC) dataset. The system combines a 3D convolutional encoder-decoder, denoising autoencoder pretraining, a ConvLSTM temporal block, attention-gated skip connections, and a variational autoencoder diagnosis branch. We evaluate these components with controlled ablation experiments on fold 0 using seed 42. The best operating point in the completed experiments is the plain 3D U-Net baseline, which reaches 0.8576 mean validation Dice. The full five-block model reaches 0.7412 mean validation Dice. This negative result is useful: it shows that adding advanced modules can hurt segmentation when the auxiliary objectives are not aligned with the discriminative target. In particular, removing autoencoder pretraining recovers performance to 0.8555 Dice, suggesting that reconstruction pretraining over-smoothed features needed for thin myocardial boundaries.

**Keywords:** cardiac MRI segmentation, 3D U-Net, ConvLSTM, autoencoder pretraining, variational autoencoder, ablation study, ACDC.

---

## 1. Introduction

Automated analysis of cine cardiac magnetic resonance imaging (cMRI) supports ventricular volume measurement, ejection-fraction estimation, and cardiac pathology assessment. The ACDC benchmark provides 150 patient cine cMRI cases across five diagnostic groups: normal, previous myocardial infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy, and abnormal right ventricle. Expert labels are available for the right ventricle, myocardium, and left ventricle at end-diastole and end-systole.

Three properties motivate our design. First, cMRI is volumetric; a 3D encoder-decoder can preserve inter-slice context better than slice-wise 2D processing. Second, cine MRI contains a temporal cardiac cycle, motivating recurrent modelling through ConvLSTM. Third, the labelled training set is small, so self-supervised pretraining and regularisation are attractive. We therefore implement a five-block model and compare it against simpler variants.

The main contribution of this project is not only the architecture, but the ablation-based interpretation. The experiments show that the clean 3D U-Net is currently the strongest segmentation model in this repository, while the full five-block model requires further tuning before it can outperform the baseline.

---

## 2. Related Work

U-Net is the dominant encoder-decoder pattern for biomedical segmentation because skip connections recover spatial detail lost during downsampling. For cardiac MRI, the 3D variant is especially relevant because adjacent slices contain anatomical continuity that 2D models may ignore.

nnU-Net demonstrated that careful preprocessing, normalisation, augmentation, and training protocols can be as important as architectural novelty. This lesson is central to our results: the baseline 3D U-Net outperformed the more complex full model, suggesting that a well-tuned simple model can be stronger than a loosely integrated complex one.

ConvLSTM extends LSTM gating to spatial feature maps by replacing fully connected gates with convolutions. It is appropriate for cine MRI because the cardiac cycle has ordered temporal structure and spatial anatomy must be preserved. Attention U-Net introduced attention gates on skip connections, allowing the decoder to suppress irrelevant encoder features before concatenation.

Denoising autoencoders learn robust representations by reconstructing clean inputs from corrupted inputs. VAEs learn a probabilistic latent space using the reparameterisation trick and KL regularisation. In this project, these ideas are used for encoder pretraining and disease classification, but the ablation results show that reconstruction-oriented pretraining does not automatically improve segmentation.

---

## 3. Methods

### 3.1 Dataset and preprocessing

ACDC provides 100 training and 50 testing patients. The label map contains four classes: background, right-ventricle cavity, myocardium, and left-ventricle cavity. The preprocessing pipeline resamples volumes, normalises intensity by z-score on non-zero voxels, crops or pads volumes to a fixed spatial size, and applies training-time augmentation.

The default training augmentation includes random affine transformation, Gaussian noise, elastic deformation, and random flipping. These transformations are intended to reduce overfitting under the small-data regime.

### 3.2 Architecture

The full model combines five blocks:

1. **3D-CNN encoder-decoder:** a volumetric U-Net-style backbone using Conv3d, InstanceNorm, and LeakyReLU blocks. The block exploits the three architectural priors of convolution: sparse connectivity (each output sees only a local voxel neighbourhood through a 3x3x3 kernel), parameter sharing (the same kernel is applied at every spatial position so the same feature detector works everywhere in the volume), and translation equivariance (a shifted input produces a correspondingly shifted feature map). The encoder uses five stages with channel widths {32, 64, 128, 256, 320} and strided downsampling, so the receptive field grows hierarchically with depth and deeper layers integrate evidence from broader anatomical regions.
2. **Denoising autoencoder pretraining:** the encoder is first trained to reconstruct clean volumes from corrupted inputs. Without corruption the network could fall into the identity trap, learning a trivial copy that drives the reconstruction loss to zero but yields no useful representation. We apply Gaussian noise (sigma = 0.1) together with voxel dropout (p = 0.2) so the encoder must learn invariances over local perturbations rather than memorise the input. The bottleneck is undercomplete relative to the raw voxel grid, so the encoder must discard noise dimensions to minimise the MSE reconstruction loss.
3. **ConvLSTM temporal block:** bottleneck features can be processed across cine frames when temporal input is provided. The same convolutional update is applied at every time step (parameter sharing across time), and training relies on backpropagation through the unfolded time graph, also known as BPTT. The fundamental fix LSTM brings over a vanilla RNN is that the cell state evolves through addition (c_t = f * c_(t-1) + i * g) instead of repeated multiplication, which prevents the exponential collapse of gradients across the cardiac cycle. Spatial structure is preserved because all four gates (input, forget, output, candidate) are implemented as 3D convolutions rather than fully connected layers.
4. **Attention-gated skip connections:** decoder skip features are re-weighted by additive attention gates.
5. **VAE diagnosis branch:** pooled bottleneck features are mapped to a latent distribution and used for five-class diagnosis.

### 3.3 Training and optimisation

Segmentation uses a combined soft-Dice and cross-entropy loss. The VAE branch adds classification cross-entropy and a KL-divergence term. The default optimiser is Adam with learning rate 1e-4, five warm-up epochs, cosine annealing, effective batch size 4, and early stopping based on validation Dice.

### 3.4 Evaluation

The repository is configured for five-fold patient-stratified cross-validation. The completed results currently report fold 0 with seed 42 and 100 epochs. The primary metric is mean validation Dice across the foreground segmentation classes.

---

## 4. Experiments

### 4.1 Compute environment

Experiments were designed for Google Colab GPUs with PyTorch 2.x and MONAI 1.3.x. A full 100-epoch training run can take several hours depending on the available GPU.

### 4.2 Main results

| Method | Fold | Seed | Best validation mean Dice |
|---|---:|---:|---:|
| 3D U-Net baseline | 0 | 42 | **0.8576** |
| Full five-block model | 0 | 42 | 0.7412 |

The baseline is the strongest completed run. This is an important empirical finding rather than a failure: it shows that the added blocks need better integration, especially the pretraining objective.

### 4.3 Ablation study

| Variant | Description | Best validation mean Dice | Delta vs full model |
|---|---|---:|---:|
| Baseline | Plain 3D U-Net | **0.8576** | +0.1164 |
| Full model | 3D-CNN + AE + ConvLSTM + attention + VAE | 0.7412 | 0.0000 |
| A1 | No ConvLSTM | 0.7272 | -0.0140 |
| A2 | No autoencoder pretraining | 0.8555 | +0.1143 |
| A3 | No attention gates | 0.7244 | -0.0168 |
| A4 | No VAE branch | 0.7364 | -0.0048 |
| A7 | No augmentation | 0.7411 | -0.0001 |

A2 is the key ablation. Removing denoising autoencoder pretraining almost recovers the baseline score, which suggests that the reconstruction task learned features that were not ideal for fine myocardial segmentation. ConvLSTM, attention, VAE, and augmentation show smaller changes in the completed fold-0 results.

### 4.4 Optimiser, initialisation, and regularisation

The codebase supports Adam, SGD, Nesterov SGD, and RMSProp. The completed CSV records Adam-based training runs. Additional optimiser and initialisation sweeps remain optional because each 100-epoch run is expensive on Colab. The report therefore treats these sweeps as configured experiments rather than final averaged claims.

Regularisation is implemented through dropout, weight decay, InstanceNorm, data augmentation, and early stopping. The A7 result shows that removing augmentation did not change fold-0 validation Dice in the current run, but this should not be over-generalised without all five folds.

---

## 5. Discussion

The central result is that the simplest completed model, the 3D U-Net baseline, outperforms the full five-block model. This matters because deep-learning projects often add modules for conceptual reasons without verifying whether each module improves the target metric. The ablation table shows that the largest issue is autoencoder pretraining: when it is removed, validation Dice increases from 0.7412 to 0.8555.

A plausible explanation is objective mismatch. Denoising reconstruction encourages smooth and globally consistent features, while cardiac segmentation depends on sharp boundaries and thin structures such as myocardium. If pretraining is not carefully tuned, it can initialise the encoder toward features that are less discriminative for boundary localisation.

The ConvLSTM result should also be interpreted carefully. The training script supports temporal input, but the completed fold-0 runs appear to operate mainly on single-frame ED/ES volumes. Therefore, the current ablation does not fully test the value of cine-cycle modelling. A stronger future experiment should feed the full temporal sequence into the ConvLSTM path.

The main limitation is that the reported table is based on fold 0 rather than completed five-fold cross-validation. Because medical-image datasets are small, fold-to-fold variation can be meaningful. The current results are valid as a project checkpoint and ablation demonstration, but final research claims would require all folds and ideally external validation on a multi-centre dataset.

---

## 6. Conclusion

This project implements a complete ACDC cardiac MRI segmentation pipeline with 3D U-Net, denoising autoencoder pretraining, ConvLSTM, attention gates, and VAE-based diagnosis. The strongest completed segmentation result is the 3D U-Net baseline with 0.8576 mean Dice on fold 0. The ablation study shows that model complexity alone does not guarantee improvement: in this setting, removing autoencoder pretraining nearly restores baseline performance. Future work should complete five-fold validation, feed full cine sequences to the temporal branch, and test cross-centre generalisation on datasets such as M&Ms.

---

## References

1. O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?", IEEE Transactions on Medical Imaging, 37(11), 2514-2525, 2018.
2. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI, 2015.
3. F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods, 2021.
4. X. Shi, Z. Chen, H. Wang, D. Yeung, W. Wong, and W. Woo, "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting", NeurIPS, 2015.
5. P. Vincent, H. Larochelle, Y. Bengio, and P. Manzagol, "Extracting and Composing Robust Features with Denoising Autoencoders", ICML, 2008.
6. D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes", ICLR, 2014.
7. O. Oktay, J. Schlemper, L. Le Folgoc, et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL, 2018.
8. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR, 2015.

---

## Appendix A - Hyperparameters

| Hyperparameter | Selected value |
|---|---:|
| Learning rate | 1e-4 |
| Batch size | 4 effective |
| Epochs | 100 |
| Dropout | 0.2 |
| Weight decay | 1e-5 |
| LR schedule | warm-up + cosine |
| VAE latent dim | 32 |
