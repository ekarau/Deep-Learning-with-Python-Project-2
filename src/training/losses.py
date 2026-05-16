"""
Loss functions.

Segmentation:
    DiceFocalLoss handles class imbalance (background ≫ tumour-like classes).
    Adding cross-entropy stabilises early training.

Diagnosis:
    Cross-entropy + KL term coming from the VAE block.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def combined_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = 4,
    focal_weight: float = 0.5,
):
    """
    logits: (B, C, D, H, W)
    targets: (B, 1, D, H, W) integer class index

    Returns weighted sum of DiceFocalLoss + cross-entropy.
    """
    from monai.losses import DiceFocalLoss

    dice_focal = DiceFocalLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_focal=1.0,
    )
    ce = nn.CrossEntropyLoss()

    targets_long = targets.squeeze(1).long()
    loss_df = dice_focal(logits, targets)
    loss_ce = ce(logits, targets_long)
    return focal_weight * loss_df + (1.0 - focal_weight) * loss_ce


def diagnosis_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
):
    ce = nn.CrossEntropyLoss()
    cls = ce(logits, targets)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return cls + beta * kl, dict(ce=cls.detach(), kl=kl.detach())
