"""
Full architecture: 3D-CNN encoder + ConvLSTM + Attention decoder + VAE diagnosis branch.

Stage A (separate script): pretrain the encoder via ConvAutoencoder3D.
Stage B (this module):     fine-tune full model for segmentation + diagnosis.

Ablation toggles (constructor flags):
    use_convlstm:   include the ConvLSTM temporal block
    use_attention:  use attention gates on skip connections
    use_vae:        include the VAE diagnosis branch
    pretrained_encoder_state: state_dict of a ConvAutoencoder3D encoder
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnn3d import Encoder3D, Decoder3D
from .convlstm import ConvLSTM
from .attention import AttentionGate
from .vae import VAEHead


class FullModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_seg_classes: int = 4,
        n_disease_classes: int = 5,
        channels: tuple[int, ...] = (32, 64, 128, 256, 320),
        lstm_hidden: int = 64,
        vae_latent_dim: int = 32,
        use_convlstm: bool = True,
        use_attention: bool = True,
        use_vae: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_convlstm = use_convlstm
        self.use_attention = use_attention
        self.use_vae = use_vae

        self.encoder = Encoder3D(in_channels=in_channels, channels=channels, dropout=dropout)

        if use_convlstm:
            self.lstm = ConvLSTM(channels=channels[-1], hidden=lstm_hidden)
        else:
            self.lstm = None

        self.decoder = Decoder3D(
            out_channels=n_seg_classes,
            channels=tuple(reversed(channels)),
            dropout=dropout,
            use_attention=use_attention,
        )

        if use_vae:
            self.vae = VAEHead(
                in_channels=channels[-1],
                latent_dim=vae_latent_dim,
                n_classes=n_disease_classes,
            )
        else:
            self.vae = None

    def load_pretrained_encoder(self, state_dict: dict) -> None:
        """Load encoder weights from a ConvAutoencoder3D state_dict."""
        encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in state_dict.items() if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        return missing, unexpected

    def forward(self, x: torch.Tensor, t_seq: torch.Tensor | None = None) -> dict:
        """
        x:      (B, C, D, H, W)                      — single frame (ED or ES)
        t_seq:  (B, T, C, D, H, W) optional sequence — used only if use_convlstm
        """
        bottleneck, skips = self.encoder(x)

        if self.use_convlstm and t_seq is not None:
            # Per-frame encode then ConvLSTM over time
            b, t, c_in, d, h, w = t_seq.shape
            t_seq_flat = t_seq.view(b * t, c_in, d, h, w)
            feats_flat, _ = self.encoder(t_seq_flat)
            feats = feats_flat.view(b, t, *feats_flat.shape[1:])
            feats_t = self.lstm(feats)
            bottleneck = feats_t[:, -1]  # use last time step as gated summary

        seg_logits = self.decoder(bottleneck, skips)

        out = dict(seg_logits=seg_logits)
        if self.use_vae:
            out.update(self.vae(bottleneck))
        return out
