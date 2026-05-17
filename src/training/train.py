"""
Segmentation training entry point.

Usage:
    python -m src.training.train --config configs/baseline.yaml --fold 0
    python -m src.training.train --config configs/full_model.yaml --fold 0
    python -m src.training.train --config configs/ablation/A1_no_convlstm.yaml --fold 0

Outputs:
    checkpoints/<experiment_name>/fold_<k>/
        best.pt              best checkpoint by val Dice
        last.pt              most recent checkpoint
        tb/                  TensorBoard logs
        metrics.csv          per-epoch train/val metrics
    ablation/results/all_runs.csv
        one row appended per completed run (for aggregation)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from monai.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import ACDCDataset, build_splits, discover_patients
from src.data.transforms import build_train_transforms, build_val_transforms
from src.models.full_model import FullModel
from src.training.config import load_config
from src.training.losses import combined_segmentation_loss, diagnosis_loss
from src.training.metrics import dice_per_class
from src.utils.seeding import seed_all


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["name"].lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict, total_epochs: int):
    name = cfg.get("name", "cosine").lower()
    warmup = cfg.get("warmup_epochs", 0)

    if name == "cosine":
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup)
    elif name == "step":
        cos = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 30), gamma=0.1)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    else:
        return None

    if warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[warmup])
    return cos


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-root", default=None, help="Override config data.root")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}  experiment={cfg['experiment_name']}  fold={args.fold}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_root = args.data_root or cfg["data"]["root"]
    patients = discover_patients(data_root)
    print(f"[data] found {len(patients)} patients")

    splits = build_splits(patients, n_splits=cfg.get("n_folds", 5), seed=cfg.get("seed", 42))
    tr_idx, va_idx = splits[args.fold]
    train_patients = [patients[i] for i in tr_idx]
    val_patients = [patients[i] for i in va_idx]
    print(f"[data] train={len(train_patients)}  val={len(val_patients)}")

    spacing = tuple(cfg["data"]["spacing"])
    spatial_size = tuple(cfg["data"]["spatial_size"])
    train_tx = build_train_transforms(spacing=spacing, spatial_size=spatial_size)
    val_tx = build_val_transforms(spacing=spacing, spatial_size=spatial_size)

    train_ds = ACDCDataset.from_patients(train_patients, transform=train_tx)
    val_ds = ACDCDataset.from_patients(val_patients, transform=val_tx)

    train_loader = DataLoader(train_ds._ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds._ds, batch_size=1, shuffle=False, num_workers=2)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    mcfg = cfg["model"]
    model = FullModel(
        in_channels=mcfg["in_channels"],
        n_seg_classes=mcfg["n_seg_classes"],
        n_disease_classes=mcfg["n_disease_classes"],
        channels=tuple(mcfg["channels"]),
        lstm_hidden=mcfg.get("lstm_hidden", 64),
        vae_latent_dim=mcfg.get("vae_latent_dim", 32),
        use_convlstm=mcfg.get("use_convlstm", False),
        use_attention=mcfg.get("use_attention", False),
        use_vae=mcfg.get("use_vae", False),
        dropout=mcfg.get("dropout", 0.2),
    ).to(device)

    # Optional encoder pretraining
    pre = cfg.get("pretraining", {})
    if pre.get("enabled"):
        ckpt_path = Path(pre["ae_checkpoint"])
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_pretrained_encoder(ckpt["model"])
            print(f"[init] loaded pretrained encoder from {ckpt_path}")
        else:
            print(f"[warn] pretraining.enabled but {ckpt_path} not found — training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M  blocks="
          f"{'CNN+' if True else ''}"
          f"{'LSTM+' if mcfg.get('use_convlstm') else ''}"
          f"{'ATT+' if mcfg.get('use_attention') else ''}"
          f"{'VAE' if mcfg.get('use_vae') else ''}")

    # ------------------------------------------------------------------
    # Optim / sched
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model.parameters(), cfg["train"]["optimizer"])
    epochs = args.epochs or cfg["train"]["epochs"]
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler", {}), epochs)
    grad_accum = cfg["train"].get("grad_accum", 1)

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir) / cfg["experiment_name"] / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    metrics_csv = out_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_dice_mean", "val_dice_rv",
            "val_dice_myo", "val_dice_lv", "lr", "elapsed_s",
        ])

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    best_dice = 0.0
    patience = 0
    es_patience = cfg["train"]["early_stopping"]["patience"]
    diag_w = cfg["train"].get("loss", {}).get("diagnosis_weight", 0.1)

    t0 = time.time()
    for epoch in range(epochs):
        # ----- train -----
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            out = model(x)

            loss = combined_segmentation_loss(out["seg_logits"], y, n_classes=mcfg["n_seg_classes"])
            if mcfg.get("use_vae"):
                cls = batch["class"].to(device).long()
                cls_loss, _ = diagnosis_loss(out["logits"], cls, out["mu"], out["logvar"])
                loss = loss + diag_w * cls_loss

            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader), 1)

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # ----- val -----
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                out = model(x)
                dices.append(dice_per_class(out["seg_logits"], y, n_classes=mcfg["n_seg_classes"]).cpu())
        d = torch.stack(dices).mean(dim=0)              # [4]
        d_overall = d[1:].mean().item()                  # exclude bg

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(d_overall)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"[ep {epoch:3d}] loss={tr_loss:.4f}  val_dice={d_overall:.4f}  "
              f"(RV={d[1]:.3f} MYO={d[2]:.3f} LV={d[3]:.3f})  lr={lr:.2e}  t={elapsed:.0f}s")

        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("val/dice_mean", d_overall, epoch)
        writer.add_scalar("val/dice_rv", d[1].item(), epoch)
        writer.add_scalar("val/dice_myo", d[2].item(), epoch)
        writer.add_scalar("val/dice_lv", d[3].item(), epoch)
        writer.add_scalar("opt/lr", lr, epoch)
        with metrics_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, d_overall, d[1].item(), d[2].item(), d[3].item(), lr, elapsed])

        # ----- checkpoint + early stop -----
        torch.save({"epoch": epoch, "model": model.state_dict()}, out_dir / "last.pt")
        if d_overall > best_dice:
            best_dice = d_overall
            patience = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_dice_mean": best_dice},
                       out_dir / "best.pt")
        else:
            patience += 1
            if patience >= es_patience:
                print(f"[stop] early stopping at epoch {epoch} (no improvement {es_patience}ep)")
                break

    writer.close()
    print(f"[done] best val_dice_mean={best_dice:.4f}  total_time={time.time()-t0:.0f}s")

    # ------------------------------------------------------------------
    # Append to ablation results CSV
    # ------------------------------------------------------------------
    abl_csv = Path("ablation/results/all_runs.csv")
    abl_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not abl_csv.exists()
    with abl_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_id", "scenario", "fold", "seed", "best_val_dice_mean", "epochs_trained"])
        w.writerow([
            f"{cfg['experiment_name']}_fold{args.fold}",
            cfg["experiment_name"],
            args.fold,
            cfg.get("seed", 42),
            f"{best_dice:.4f}",
            epoch + 1,
        ])


if __name__ == "__main__":
    main()
