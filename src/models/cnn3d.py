"""
Block 1 — 3D-CNN encoder/decoder (U-Net-style backbone).

Motivation (PAPER.md §3.2):
    Voxel-wise spatial structure of cine cMRI demands convolutional
    inductive biases. We adopt the U-Net family encoder–decoder topology
    (Ronneberger 2015) operating in 3D so that inter-slice context is
    preserved at every level.

The encoder also exposes `skips` for attention-gated skip connections,
and a `bottleneck` tensor consumed by the ConvLSTM and VAE blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_c, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_c, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout3d(dropout))
    return nn.Sequential(*layers)


class Encoder3D(nn.Module):
    """Five-stage 3D residual encoder. Returns bottleneck + 4 skip features."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 320),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        prev = in_channels
        for i, c in enumerate(channels):
            self.stages.append(_conv_block(prev, c, dropout=dropout if i > 0 else 0.0))
            if i < len(channels) - 1:
                self.downs.append(nn.Conv3d(c, c, kernel_size=2, stride=2))
            prev = c

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skips.append(x)
                x = self.downs[i](x)
        bottleneck = x
        return bottleneck, skips


class Decoder3D(nn.Module):
    """Mirror decoder. Builds one attention gate per level if `use_attention=True`."""

    def __init__(
        self,
        out_channels: int = 4,
        channels: tuple[int, ...] = (320, 256, 128, 64, 32),
        dropout: float = 0.0,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_attention = use_attention
        self.ups = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.attentions = nn.ModuleList() if use_attention else None

        for i in range(len(channels) - 1):
            in_c, out_c = channels[i], channels[i + 1]
            self.ups.append(nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2))
            # Concatenated input is upsample (out_c) + skip (out_c) → 2*out_c
            self.stages.append(_conv_block(out_c * 2, out_c, dropout=dropout))
            if use_attention:
                # Local import to avoid a circular dependency at module load.
                from .attention import AttentionGate
                self.attentions.append(
                    AttentionGate(in_channels_skip=out_c, in_channels_gate=out_c, hidden=max(out_c // 2, 1))
                )

        self.head = nn.Conv3d(channels[-1], out_channels, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """`skips` is the list returned by Encoder3D (encoder order, shallowest first)."""
        x = bottleneck
        for i, (up, stage) in enumerate(zip(self.ups, self.stages)):
            x = up(x)
            skip = skips[-(i + 1)]  # take deepest skip first
            if self.use_attention:
                skip = self.attentions[i](skip=skip, gating=x)
            x = torch.cat([x, skip], dim=1)
            x = stage(x)
        return self.head(x)
