"""
custom_losses.py
================
Two candidate losses for improving OOD generalization on protein
stability prediction, designed to complement BMC + siamese in
configurations E, F, G of the seed-42 ablation suite.

(1) BCAS (Bias-Corrected Anti-Symmetric loss)
    -----------------------------------------
    Motivation:
        Our anti-symmetry diagnostic shows that the existing siamese
        loss (E[(f_fwd + f_rev)^2]) does NOT eliminate the systematic
        forward-reverse bias on Ssym. Every config we tested has a
        signed batch-mean (f_fwd + f_rev) of order +/-0.3 kcal/mol.
        The standard siamese loss minimizes the *variance* of the sum,
        not its mean: a model that consistently mis-predicts by a fixed
        offset has zero variance contribution but non-zero bias.

    Loss:
        L_BCAS = alpha * [mean_batch(f_fwd + f_rev)]^2
               + beta  * mean_batch[(f_fwd + f_rev)^2]

        First term is the squared *batch-level mean* of the
        forward+reverse sum. This directly penalizes the signed bias.
        Second term is the original siamese variance penalty.

        Defaults: alpha=1.0, beta=0.5 (same as previous siamese_weight).

    Replaces the siamese loss in the training objective. Other loss
    components (BMC) unchanged.

(2) OOD-margin loss (input-noise consistency regularization)
    --------------------------------------------------------
    Motivation:
        We observe a clear in-distribution / OOD trade-off: BMC-based
        configurations gain on S669 (+0.031) but lose on Megascale-test
        (-0.022). We hypothesize that this trade-off is partly because
        the model overfits in-distribution feature statistics. Adding
        a consistency-regularization term on small noise perturbations
        of the per-position features encourages the model to produce
        stable predictions under representation drift, which is the
        first-order approximation of what an OOD sample looks like.

    Loss:
        Sample delta ~ N(0, sigma^2 * I), shape (1, K, 1408).
        L_margin = (1/K) * sum_k [ (f_pred_clean - f_pred_noisy)_k ]^2

        Where f_pred_noisy is computed by re-running the MLP head only
        on (muted_id_representation + delta), reusing the encoder output.

    Defaults: sigma=0.1, weight=0.5.

    Both losses are off in TrainConfig by default; enable with
    `use_bcas=True` (which automatically disables `use_siamese`) and/or
    `use_ood_margin=True`.
"""

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
