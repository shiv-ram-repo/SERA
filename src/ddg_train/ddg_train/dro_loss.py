"""
dro_loss.py
===========
CVaR (Conditional Value at Risk) distributionally-robust wrapper around
the base per-example regression loss.

Motivation for the paper: the imbalance and OOD-generalization framing
both reduce to "don't let the dominant, easy examples dominate the
gradient." BMC addresses this via distribution-aware reweighting in
label space. CVaR-DRO addresses it directly in LOSS space: at each step,
optimize only the worst-alpha fraction of per-example losses (the tail
the model handles worst). This is a principled, single-hyperparameter
alternative that subsumes the imbalance motivation under distributional
robustness (Levy et al. 2020; Sagawa et al. 2020).

L_CVaR(alpha) = mean( top-alpha-fraction of per-example losses )

alpha=1.0 recovers standard mean loss. alpha=0.5 optimizes the worst
half each step. Typically alpha in [0.3, 0.7].

This wraps a per-example loss tensor; combine with BMC/siamese/OOD-margin
in the same total objective, or use alone in place of the base regression
loss.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CVaRLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        self.alpha = float(alpha)

    def forward(self, per_example_loss: torch.Tensor):
        """per_example_loss: (N,) tensor of non-reduced losses."""
        n = per_example_loss.numel()
        if n == 0:
            return torch.zeros((), device=per_example_loss.device)
        if self.alpha >= 1.0:
            return per_example_loss.mean()
        k = max(1, int(round(self.alpha * n)))
        topk, _ = torch.topk(per_example_loss, k, largest=True, sorted=False)
        return topk.mean()


class GroupDRO(nn.Module):
    """Group-DRO: maintain per-group weights that upweight the worst group.

    Groups here can be proteins (group_id per example). Weights follow an
    exponentiated-gradient update on the per-group mean loss. Use when you
    want worst-PROTEIN robustness rather than worst-example.
    """

    def __init__(self, n_groups: int, step_size: float = 0.01):
        super().__init__()
        self.register_buffer("weights", torch.ones(n_groups) / n_groups)
        self.step_size = float(step_size)

    def forward(self, per_example_loss: torch.Tensor, group_id: torch.Tensor):
        device = per_example_loss.device
        w = self.weights.to(device)
        groups = torch.unique(group_id)
        group_losses = torch.zeros_like(w)
        for g in groups:
            m = group_id == g
            if m.any():
                group_losses[g] = per_example_loss[m].mean()
        # exponentiated-gradient update on group weights
        with torch.no_grad():
            neww = w * torch.exp(self.step_size * group_losses)
            neww = neww / (neww.sum() + 1e-12)
            self.weights = neww.detach().cpu()
        return (self.weights.to(device) * group_losses).sum()
