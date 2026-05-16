"""
Block 3 — Convolutional LSTM over the cardiac time axis.

Motivation (PAPER.md §3.2):
    Cine MRI is a genuine temporal signal (~25-30 phases per cardiac
    cycle). Vanilla recurrent units suffer vanishing gradients over
    such horizons (Hochreiter & Schmidhuber 1997). ConvLSTM
    (Shi et al. 2015) preserves spatial structure via convolutional
    gates while gating temporal flow.

Operates on bottleneck feature maps with shape (B, T, C, D, H, W).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single 3D ConvLSTM cell. Spatial structure preserved by 3D convolutions on gates."""

    def __init__(self, channels: int, hidden: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.hidden = hidden
        # i, f, o, g — 4 gates in one conv for efficiency
        self.conv = nn.Conv3d(channels + hidden, 4 * hidden, kernel_size=kernel_size, padding=pad)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            b, _, d, h, w = x.shape
            zeros = x.new_zeros(b, self.hidden, d, h, w)
            state = (zeros, zeros)
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """ConvLSTM over a sequence of bottleneck feature volumes."""

    def __init__(self, channels: int, hidden: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(channels, hidden, kernel_size)
        self.project = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq:  (B, T, C, D, H, W)
        out:  (B, T, C, D, H, W)  — same shape, gated over time

        Use the last hidden state if you want a single-frame summary,
        or `out[:, -1]` etc. downstream.
        """
        b, t, c, d, h, w = seq.shape
        state = None
        outputs: list[torch.Tensor] = []
        for tt in range(t):
            x = seq[:, tt]
            h_next, c_next = self.cell(x, state)
            state = (h_next, c_next)
            outputs.append(self.project(h_next))
        return torch.stack(outputs, dim=1)
