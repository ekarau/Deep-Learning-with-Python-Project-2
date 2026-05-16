"""
Block 4 — Attention Gate for skip connections.

Motivation (PAPER.md §3.2):
    Plain U-Net skip concatenation is uninformed about which encoder
    voxels are relevant for the current decoder query. The attention
    gate (Oktay et al. 2018) produces a soft mask alpha in [0, 1]
    derived from skip + gating signal that re-weights the skip
    features before concatenation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Additive attention gate for 3D U-Net skip connections."""

    def __init__(self, in_channels_skip: int, in_channels_gate: int, hidden: int) -> None:
        super().__init__()
        self.theta = nn.Conv3d(in_channels_skip, hidden, kernel_size=1)
        self.phi = nn.Conv3d(in_channels_gate, hidden, kernel_size=1)
        self.psi = nn.Conv3d(hidden, 1, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        # gating is the upsampled decoder feature; matches `skip` spatially after up.
        g = self.phi(gating)
        s = self.theta(skip)
        att = self.sigmoid(self.psi(self.act(s + g)))   # (B, 1, D, H, W)
        return skip * att
