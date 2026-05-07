# Generative Deep Learning on Fashion-MNIST

Project 2 of SWE012 — Deep Learning with Python. Covers Weeks 9 (CNNs), 12 (RNN/LSTM/GRU), and 13 (Autoencoders, VAEs, GANs).

---

## Course Info

| | |
|---|---|
| Course | SWE012 — Deep Learning with Python |
| University | İstinye University |
| Instructor | Asst. Prof. Dr. Yiğit Bekir Kaya |
| Dataset | Fashion-MNIST (60k train / 10k test, 28×28 grayscale, 10 classes) |
| Framework | PyTorch |
| Seed | 42 |

---

## Notebook Structure

| § | Section | Topic |
|---|---|---|
| 0 | Setup, imports, reproducibility | — |
| 1 | Fashion-MNIST loading + EDA | — |
| 2 | Generic training loop helpers | — |
| 3 | **A.0** CNN foundations (from-scratch) | Week 9 |
| 4 | **A.1** CNN classifiers on Fashion-MNIST | Week 9 |
| 5 | **B** Convolutional autoencoders | Week 13 |
| 6 | **C** Variational autoencoder | Week 13 |
| 7 | **D** DCGAN | Week 13 |
| 8 | **E** RNN / LSTM / GRU sequence autoencoders | Week 12 |
| 9 | **F** Tiny diffusion model (bonus) | Bonus |
| 10 | Comprehensive comparison | — |
| 11 | Conclusions | — |

---

## Section A.0 — CNN Foundations (Week 9)

| § | Topic |
|---|---|
| A.0.1 | FC vs CNN parameter scaling |
| A.0.2 | 2D convolution from scratch (NumPy + PyTorch verify) |
| A.0.3 | Hand-crafted kernels: Sobel, Laplacian, Gabor |
| A.0.4 | Padding & stride geometry (Valid / Same / Full) |
| A.0.5 | Max / average pooling from scratch |
| A.0.6 | Translation equivariance vs invariance |
| A.0.7 | Receptive field growth |
| A.0.8 | Three Pillars summary |
| A.0.9 | Architecture history (LeNet → ResNet) |
| A.0.10 | V1 / Hubel-Wiesel / Gabor connection |

---

## Models Trained

| # | Model | Family | Section |
|---|---|---|---|
| 1 | LeNet-style CNN | CNN | A.1 |
| 2 | Modern CNN (BN + Dropout + Augmentation) | CNN | A.1 |
| 3 | ResNet-Mini | CNN | A.1 |
| 4 | Convolutional Autoencoder | Generative | B |
| 5 | Denoising Autoencoder | Generative | B |
| 6 | Variational Autoencoder | Generative | C |
| 7 | DCGAN | Generative | D |
| 8 | RNN Autoencoder | Sequence | E |
| 9 | LSTM Autoencoder | Sequence | E |
| 10 | GRU Autoencoder | Sequence | E |
| 11 | Bidirectional LSTM Autoencoder | Sequence | E |
| 12 | Tiny Diffusion Model | Generative | F |

---

## Repository Structure

```
DL_Project_2/
├── README.md
├── REPORT.md
├── requirements.txt
├── notebooks/
│   └── Generative_Deep_Learning_Fashion_MNIST.ipynb
├── responsibilities/
│   ├── 220901755.md
│   ├── 229910141.md
│   ├── 229910158.md
│   ├── 2309011036.md
│   └── 2309011053.md
└── figures/
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Generative_Deep_Learning_Fashion_MNIST.ipynb
```
