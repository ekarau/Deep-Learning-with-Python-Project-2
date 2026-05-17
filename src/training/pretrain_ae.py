"""
Stage A — Denoising convolutional autoencoder pretraining.

Trains ConvAutoencoder3D on ACDC volumes with MSE reconstruction loss
under Gaussian noise + voxel dropout corruption. The resulting encoder
weights are loaded by `src.training.train` when `pretraining.enabled`
is set in the full_model config.

Usage:
    python -m src.training.pretrain_ae --epochs 50 --out checkpoints/ae_pretrain.pt
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from monai.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import ACDCDataset, discover_patients
from src.data.transforms import build_train_transforms
from src.models.autoencoder import ConvAutoencoder3D
from src.utils.seeding import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--voxel-dropout", type=float, default=0.2)
    parser.add_argument("--spatial-size", nargs=3, type=int, default=[224, 224, 16])
    parser.add_argument("--out", default="checkpoints/ae_pretrain.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    seed_all(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}  out={args.out}")

    # --- Data: same dataset as segmentation, but we only need the image ---
    patients = discover_patients(args.data_root)
    print(f"[data] {len(patients)} patients (using all for pretraining — no train/val split)")

    train_tx = build_train_transforms(spatial_size=tuple(args.spatial_size))
    ds = ACDCDataset.from_patients(patients, transform=train_tx, include_es=True)
    loader = DataLoader(ds._ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # --- Model ---
    model = ConvAutoencoder3D(
        in_channels=1,
        noise_std=args.noise_std,
        voxel_dropout=args.voxel_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] AE params={n_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss()

    # --- Output paths ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "ae_tb"
    writer = SummaryWriter(log_dir)
    metrics_csv = out_path.parent / "ae_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "lr", "elapsed_s"])

    # --- Loop ---
    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            x = batch["image"].to(device)
            recon, _ = model(x)
            loss = mse(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"[ep {epoch:3d}] mse={avg:.5f}  lr={lr:.2e}  t={elapsed:.0f}s")
        writer.add_scalar("train/mse", avg, epoch)
        writer.add_scalar("opt/lr", lr, epoch)
        with metrics_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, avg, lr, elapsed])

        # Save best by reconstruction loss
        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "model": model.state_dict(), "mse": avg}, out_path)

    writer.close()
    print(f"[done] best train mse={best_loss:.5f}  saved -> {out_path}")


if __name__ == "__main__":
    main()
