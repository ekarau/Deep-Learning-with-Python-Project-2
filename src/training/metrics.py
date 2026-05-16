"""Segmentation + classification metrics."""
from __future__ import annotations

import torch


def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_classes: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    pred:   (B, C, D, H, W) softmax logits OR argmax class map (B, 1, D, H, W)
    target: (B, 1, D, H, W) integer class index

    Returns: 1D tensor of length n_classes — Dice per class (background incl.)
    """
    if pred.shape[1] > 1:
        pred = pred.argmax(dim=1, keepdim=True)
    target = target.long()

    dices = []
    for c in range(n_classes):
        p_c = (pred == c).float()
        t_c = (target == c).float()
        inter = (p_c * t_c).sum()
        denom = p_c.sum() + t_c.sum()
        dices.append((2 * inter + eps) / (denom + eps))
    return torch.stack(dices)


def hausdorff95(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Wrapper around MONAI's HD95 metric. See notebook for invocation."""
    from monai.metrics import HausdorffDistanceMetric
    metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    metric(y_pred=pred, y=target)
    return metric.aggregate()


def balanced_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    from sklearn.metrics import balanced_accuracy_score
    pred = logits.argmax(dim=1).cpu().numpy()
    return torch.tensor(balanced_accuracy_score(targets.cpu().numpy(), pred))
