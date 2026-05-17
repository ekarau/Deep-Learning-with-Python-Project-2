"""
Block 2 — Convolutional Autoencoder (denoising pretraining).

Motivation (PAPER.md §3.2):
    100 training patients is a small-data regime. We pretrain the
    encoder as part of a denoising convolutional autoencoder
    (Vincent et al. 2008): input is corrupted by Gaussian noise and
    voxel dropout, encoder–decoder is trained to reconstruct the clean
    volume with MSE. The pretrained encoder weights are then
    transferred to the segmentation network in Stage B.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnn3d import Encoder3D, Decoder3D


class ConvAutoencoder3D(nn.Module):
    """3D denoising autoencoder. Shares the same encoder topology as the segmentation net."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 320),
        noise_std: float = 0.1,
        voxel_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = Encoder3D(in_channels=in_channels, channels=channels)
        # Reconstruction decoder mirrors encoder, output channel == input channel.
        self.recon_decoder = Decoder3D(
            out_channels=in_channels,
            channels=tuple(reversed(channels)),
            use_attention=False,
        )
        self.noise_std = noise_std
        self.voxel_dropout = voxel_dropout

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.training and self.voxel_dropout > 0:
            mask = torch.rand_like(x) > self.voxel_dropout
            x = x * mask
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, bottleneck)."""
        x_noisy = self.corrupt(x)
        bottleneck, skips = self.encoder(x_noisy)
        recon = self.recon_decoder(bottleneck, skips)
        return recon, bottleneck
