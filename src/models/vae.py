"""
Block 5 — Variational Autoencoder head for diagnosis branch.

Motivation (PAPER.md §3.2):
    Pooled encoder features are mapped to a Gaussian latent (mu, log_var).
    Reparameterised samples feed a small MLP for 5-class disease
    classification. KL divergence regularises the latent toward a
    standard normal prior (Kingma & Welling 2014). This block introduces
    the probabilistic / Bayesian element required by the course rubric.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VAEHead(nn.Module):
    """Maps encoder bottleneck → 5-class diagnosis with VAE latent."""

    def __init__(
        self,
        in_channels: int = 320,
        latent_dim: int = 32,
        n_classes: int = 5,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(in_channels, latent_dim)
        self.fc_logvar = nn.Linear(in_channels, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, bottleneck: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.flatten(self.pool(bottleneck))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterise(mu, logvar)
        logits = self.classifier(z)
        return dict(logits=logits, mu=mu, logvar=logvar, z=z)
