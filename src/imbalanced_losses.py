

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Balanced MSE (Batch-based Monte Carlo variant — BMC)
# --------------------------------------------------------------------------- #
class BMCLoss(nn.Module):
    """
    Balanced MSE via Batch-based Monte Carlo (Ren et al. 2022).

    Treats regression as Gaussian classification over the batch labels.
    The noise sigma is learnable (initialised to `init_noise_sigma`) and
    optimised alongside model parameters.

    Empirically equivalent to weighting samples by inverse local density,
    but requires no pre-computed histogram.
    """

    def __init__(self, init_noise_sigma: float = 1.0):
        super().__init__()
        # Learnable log_sigma for numerical stability.
        self.noise_log_sigma = nn.Parameter(
            torch.tensor(math.log(init_noise_sigma), dtype=torch.float32)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.reshape(-1, 1)            # (N, 1)
        target = target.reshape(-1, 1)          # (N, 1)
        sigma2 = (self.noise_log_sigma.exp() ** 2)

        # logits[i, j] = -||pred_i - target_j||^2 / (2 sigma^2)
        logits = -((pred - target.t()) ** 2) / (2.0 * sigma2)

        # Cross-entropy where the "correct" class for sample i is i itself.
        labels = torch.arange(pred.size(0), device=pred.device)
        loss = F.cross_entropy(logits, labels)

        # Detach the sigma^2 factor so it doesn't double-count in backward.
        return loss * (2.0 * sigma2).detach()


# --------------------------------------------------------------------------- #
# Label Distribution Smoothing (LDS) — pre-computed weights
# --------------------------------------------------------------------------- #
def compute_lds_weights(
    labels: np.ndarray,
    bin_size: float = 0.1,
    kernel: str = "gaussian",
    sigma: float = 2.0,
    truncate: float = 4.0,
    eps: float = 1e-6,
) -> dict:
    """
    Compute per-sample LDS weights from a 1-D array of training labels.

    Returns a dict with:
        - 'bin_edges'      : np.ndarray of histogram bin edges
        - 'smoothed_density': np.ndarray of smoothed densities (one per bin)
        - 'weights'        : np.ndarray of per-sample weights (same len as labels)

    Use `weights` to scale per-sample loss inside the training loop.
    """
    labels = np.asarray(labels, dtype=np.float64).ravel()
    y_min, y_max = labels.min() - bin_size, labels.max() + bin_size
    n_bins = int(np.ceil((y_max - y_min) / bin_size))
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Raw histogram (length = n_bins).
    hist, _ = np.histogram(labels, bins=bin_edges)
    hist = hist.astype(np.float64)

    # 1-D Gaussian convolution. np.convolve(mode='same') returns max(M,N),
    # so we trim explicitly to len(hist).
    half = int(np.ceil(truncate * sigma / bin_size))
    grid = np.arange(-half, half + 1) * bin_size
    if kernel == "gaussian":
        kern = np.exp(-(grid ** 2) / (2.0 * sigma ** 2))
    elif kernel == "triang":
        kern = np.maximum(0.0, 1.0 - np.abs(grid) / (sigma * 3.0))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    kern /= kern.sum()
    smoothed_full = np.convolve(hist, kern, mode="full")
    # 'full' length = len(hist) + len(kern) - 1; centre slice matches hist.
    start = (len(kern) - 1) // 2
    smoothed = smoothed_full[start:start + len(hist)]
    smoothed = np.maximum(smoothed, eps)
    assert smoothed.shape == hist.shape, (smoothed.shape, hist.shape)

    # Inverse-frequency weights, normalised to mean 1 over occupied bins.
    inv = 1.0 / smoothed
    occupied = hist > 0
    if occupied.sum() > 0:
        inv /= inv[occupied].mean()

    # Map each label to its bin and look up weight.
    bin_idx = np.clip(np.digitize(labels, bin_edges) - 1, 0, n_bins - 1)
    weights = inv[bin_idx].astype(np.float32)

    return dict(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        smoothed_density=smoothed,
        weights=weights,
    )


class LDSWeightedHuber(nn.Module):
    """
    Huber loss with per-sample LDS weights passed in at call time.

    Usage:
        loss_fn = LDSWeightedHuber(beta=1.0)
        # weights[i] is the LDS weight for sample i in the batch
        loss = loss_fn(pred, target, weights)
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        elementwise = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        if weights is None:
            return elementwise.mean()
        return (elementwise * weights).mean()
