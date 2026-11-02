
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ======================================================================= #
# (1) BCAS — Bias-Corrected Anti-Symmetric loss
# ======================================================================= #
class BCASLoss(nn.Module):
    """
    L_BCAS = alpha * [mean(f_fwd + f_rev)]^2 + beta * mean[(f_fwd + f_rev)^2]

    Designed to be a drop-in replacement for the original siamese term.
    The first (bias) term and the second (variance) term are exposed as
    separate scalars in the returned dict for logging.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta  = float(beta)

    def forward(self,
                ddg_fwd: torch.Tensor,    # (K,) — forward ΔΔG predictions
                ddg_rev: torch.Tensor,    # (K,) — reverse ΔΔG predictions
                valid: torch.Tensor,      # (K,) bool mask, optional
                ):
        # Apply validity mask
        if valid is not None:
            ddg_fwd = ddg_fwd[valid]
            ddg_rev = ddg_rev[valid]
        if ddg_fwd.numel() == 0:
            zero = torch.zeros((), device=ddg_fwd.device)
            return zero, {"bcas_bias": 0.0, "bcas_var": 0.0,
                          "bcas_mean_sum": 0.0}

        # Forward + reverse sum per mutation
        s = ddg_fwd + ddg_rev            # (K,)

        # (a) Squared mean of the sum → systematic bias
        mean_s = s.mean()
        bias_term = mean_s ** 2

        # (b) Mean of the squared sum → per-mutation variance about zero
        var_term = (s ** 2).mean()

        total = self.alpha * bias_term + self.beta * var_term

        return total, {
            "bcas_bias":     float(bias_term.item()),
            "bcas_var":      float(var_term.item()),
            "bcas_mean_sum": float(mean_s.item()),    # signed; for diagnostics
        }


# ======================================================================= #
# (2) OOD-margin loss — input-noise consistency regularization
# ======================================================================= #
class OODMarginLoss(nn.Module):
    """
    Apply small Gaussian noise to the per-position feature representation
    and penalize the difference between the original and perturbed
    predictions. This is a first-order consistency regularizer that
    encourages stable predictions under representation drift.

    The noise is applied to `muted_id_representation` AFTER the encoder
    pass, so we only re-run the MLP head — no extra encoder cost.

    Caller is responsible for running:
        1. Clean forward pass via the standard model → ddg_clean, also
           populating batch["muted_id_representation"]
        2. Calling .forward(model, batch, ddg_clean, valid) here, which
           internally adds noise and re-runs the MLP head only.
    """

    def __init__(self, sigma: float = 0.1, weight: float = 0.5,
                 n_samples: int = 1):
        super().__init__()
        self.sigma     = float(sigma)
        self.weight    = float(weight)
        self.n_samples = int(n_samples)

    def forward(self, model, batch: dict,
                ddg_clean: torch.Tensor,   # (K,) clean prediction
                valid: torch.Tensor,
                ):
        """
        Returns (loss_scalar, diagnostics_dict).
        """
        # The clean encoder representation. Populated by the prior model(batch)
        # call by the parent MultimodalDDG forward.
        rep = batch.get("muted_id_representation", None)
        if rep is None:
            return (torch.zeros((), device=ddg_clean.device),
                    {"ood_margin_loss": 0.0, "ood_margin_skipped": True})

        # We want to re-run the MLP head only, on noisy rep.
        mlp = getattr(model, "mlp", None)
        if mlp is None:
            return (torch.zeros((), device=ddg_clean.device),
                    {"ood_margin_loss": 0.0, "ood_margin_skipped": True})

        # MultimodalDDG gathers at WT and MUT positions from append_tensors.
        # We need to reproduce that gather.
        append_tensors = batch.get("append_tensors", None)
        if append_tensors is None:
            return (torch.zeros((), device=ddg_clean.device),
                    {"ood_margin_loss": 0.0, "ood_margin_skipped": True})

        wt_onehot  = append_tensors[..., :21].float()
        mut_onehot = append_tensors[..., 21:42].float()

        total = torch.zeros((), device=ddg_clean.device)
        last_diff_norm = 0.0

        for _ in range(self.n_samples):
            # Build noisy rep: same shape as clean
            noise = torch.randn_like(rep) * self.sigma

            noisy_batch = dict(batch)
            noisy_batch["muted_id_representation"] = rep + noise

            # Re-run the MLP head (cheap)
            pred = mlp(noisy_batch)              # (1, K, 21)
            score = pred.squeeze(0)              # (K, 21)
            score_wt  = (score * wt_onehot).sum(-1)
            score_mut = (score * mut_onehot).sum(-1)
            ddg_noisy = score_mut - score_wt     # (K,)

            # Squared difference between clean and noisy predictions
            diff = ddg_noisy - ddg_clean.detach()
            if valid is not None:
                diff = diff[valid]
            if diff.numel() == 0:
                continue
            term = (diff ** 2).mean()
            total = total + term
            last_diff_norm = float(diff.abs().mean().item())

        total = total / max(self.n_samples, 1)

        return total, {
            "ood_margin_loss":     float(total.item()),
            "ood_margin_diff":     last_diff_norm,
            "ood_margin_skipped":  False,
        }
