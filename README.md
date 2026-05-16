# Cardiac MRI Segmentation and Diagnosis on ACDC

A 3D + temporal deep-learning pipeline for joint cardiac MRI segmentation
(left ventricle, right ventricle, myocardium) and five-class disease
diagnosis on the Automated Cardiac Diagnosis Challenge (ACDC) benchmark.

The full methods write-up lives in [`PAPER.md`](PAPER.md).

---

## Architecture

| Block | Role |
|---|---|
| 3D-CNN encoder–decoder | Voxel-level spatial features (U-Net family backbone) |
| Convolutional autoencoder | Self-supervised denoising pretraining of the encoder |
| ConvLSTM | Temporal modelling over the cardiac cycle |
| Attention gates | Re-weighted encoder–decoder skip connections |
| Variational autoencoder | Probabilistic latent for disease classification |

---

## Repository layout

```
.
├── PAPER.md                  Methods write-up
├── REPORT.md                 Technical notes
├── requirements.txt
├── data/                     ACDC training/testing (gitignored)
├── references/               Cited papers
├── notebooks/                EDA, preprocessing, training, ablation
├── src/
│   ├── data/                 Dataset, transforms, K-fold splits
│   ├── models/               cnn3d, autoencoder, convlstm, attention, vae, full_model
│   ├── training/             Losses and metrics
│   └── utils/                Reproducibility helpers
├── configs/                  YAML configs for baseline, full model, and ablations
├── ablation/                 Ablation matrix and aggregated results
├── responsibilities/         Per-author task contracts
├── docs/                     Presentation script
└── figures/                  Generated plots
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

1. Register and download from <https://www.creatis.insa-lyon.fr/Challenge/acdc/>.
2. Extract `training.zip` and `testing.zip` into `data/`.
3. Run `notebooks/00_EDA.ipynb` for a sanity check.

### Training

```bash
python -m src.training.train --config configs/baseline.yaml --fold 0
python -m src.training.train --config configs/full_model.yaml --fold 0
```

Ablation scenarios are run from `configs/ablation/A1..A7_*.yaml`.

---

## Citation

If you use this code or the ACDC dataset, please cite:

```bibtex
@article{bernard2018deep,
  title   = {Deep learning techniques for automatic {MRI} cardiac multi-structures
             segmentation and diagnosis: Is the problem solved?},
  author  = {Bernard, Olivier and Lalande, Alain and Zotti, Clement and
             Cervenansky, Frederick and Yang, Xin and Heng, Pheng-Ann and
             Cetin, Irem and Lekadir, Karim and Camara, Oscar and
             Gonzalez Ballester, Miguel A. and Sanroma, Gerard and
             Napel, Sandy and Petersen, Steffen and Tziritas, Georgios and
             Grinias, Elias and Khened, Mahendra and Kollerathu, Varghese A. and
             Krishnamurthi, Ganapathy and Roh{\'e}, Marc-Michel and
             Pennec, Xavier and Sermesant, Maxime and Isensee, Fabian and
             J{\"a}ger, Paul and Maier-Hein, Klaus H. and Full, Peter M. and
             Wolf, Ivo and Engelhardt, Sandy and Baumgartner, Christian F. and
             Koch, Lisa M. and Wolterink, Jelmer M. and I{\v{s}}gum, Ivana and
             Jang, Yeonggul and Hong, Yoonmi and Patravali, Jay and
             Jain, Shubham and Humbert, Olivier and Jodoin, Pierre-Marc},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {37},
  number  = {11},
  pages   = {2514--2525},
  year    = {2018},
  doi     = {10.1109/TMI.2018.2837502}
}
```

---

## License

Code released under the MIT License. The ACDC dataset is distributed under
its own terms — see the
[ACDC challenge page](https://www.creatis.insa-lyon.fr/Challenge/acdc/) for
details.
