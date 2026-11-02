"""
novel_losses.py
===============
New loss variants for the resubmission's methodological contribution.
All are drop-in replacements for OODMarginLoss and share its call
signature: forward(model, batch, ddg_clean, valid) -> (loss, diag).

The methodological upgrade over the isotropic OOD-margin loss: instead
of perturbing every feature dimension with the same scalar sigma, we
perturb each dimension in proportion to how much that dimension actually
shifts between the training distribution and unseen proteins. This turns
the consistency regularizer from a generic noise injection into a
targeted defense along the directions that carry distribution shift.

Three variants, increasing in adaptivity:

  1. VarianceScaledOODMargin
     Per-dimension noise scaled by the empirical std of the fused feature
     representation on the training set. Directions with large natural
     variance get proportionally larger perturbations. The per-dimension
     std vector is estimated once from a sample of training batches
     (see estimate_feature_std) and passed in at construction.

  2. LearnableOODMargin
     Per-dimension log-sigma is a free parameter vector, learned jointly
     with the network. A small penalty keeps the sigmas from collapsing
     to zero (which would trivially minimize the consistency loss). This
     lets the model discover which directions most need stabilizing.

  3. ShiftAlignedOODMargin
     Per-dimension noise scaled by the measured train->OOD shift vector
     (e.g., |mean_OOD - mean_train| per dimension, from the representation
     -shift analysis). Perturbs precisely the directions along which real
     OOD proteins differ. Requires a shift vector at construction.

All keep the encoder pass shared (head-only re-run), so per-step cost is
the same as the isotropic version.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _gather_ddg(score, wt_onehot, mut_onehot):
    """Reproduce the SPURS gather: ddG = score_mut - score_wt."""
    score_wt = (score * wt_onehot).sum(-1)
    score_mut = (score * mut_onehot).sum(-1)
    return score_mut - score_wt


def _rerun_head(model, batch, rep_noisy):
    """Re-run the MLP head only on a perturbed representation."""
    mlp = getattr(model, "mlp", None)
    if mlp is None:
        return None
    noisy_batch = dict(batch)
    noisy_batch["muted_id_representation"] = rep_noisy
    pred = mlp(noisy_batch)          # (1, K, 21)
    return pred.squeeze(0)           # (K, 21)


class _BaseMargin(nn.Module):
    """Shared plumbing for the OOD-margin variants."""

    def __init__(self, weight: float = 0.5, n_samples: int = 1):
        super().__init__()
        self.weight = float(weight)
        self.n_samples = int(n_samples)

    def _noise(self, rep):
        raise NotImplementedError

    def _extra_penalty(self, device):
        return torch.zeros((), device=device)

    def forward(self, model, batch, ddg_clean, valid):
        rep = batch.get("muted_id_representation", None)
        if rep is None:
            return (torch.zeros((), device=ddg_clean.device),
                    {"ood_margin_loss": 0.0, "ood_margin_skipped": True})
        at = batch.get("append_tensors", None)
        if at is None or getattr(model, "mlp", None) is None:
            return (torch.zeros((), device=ddg_clean.device),
                    {"ood_margin_loss": 0.0, "ood_margin_skipped": True})

        wt_onehot = at[..., :21].float()
        mut_onehot = at[..., 21:42].float()

        total = torch.zeros((), device=ddg_clean.device)
        last = 0.0
        for _ in range(self.n_samples):
            rep_noisy = rep + self._noise(rep)
            score = _rerun_head(model, batch, rep_noisy)
            if score is None:
                continue
            ddg_noisy = _gather_ddg(score, wt_onehot, mut_onehot)
            diff = ddg_noisy - ddg_clean.detach()
            if valid is not None:
                diff = diff[valid]
            if diff.numel() == 0:
                continue
            total = total + (diff ** 2).mean()
            last = float(diff.abs().mean().item())
        total = total / max(self.n_samples, 1)
        total = total + self._extra_penalty(ddg_clean.device)
        return total, {"ood_margin_loss": float(total.item()),
                       "ood_margin_diff": last, "ood_margin_skipped": False}


class VarianceScaledOODMargin(_BaseMargin):
    """Per-dimension noise scaled by the training-set feature std.

    feature_std: 1-D tensor of length D (=1408), the empirical std of the
    fused per-position representation over a sample of training data.
    base_sigma multiplies the whole vector.
    """

    def __init__(self, feature_std: torch.Tensor, base_sigma: float = 0.10,
                 weight: float = 0.5, n_samples: int = 1):
        super().__init__(weight, n_samples)
        std = feature_std.float()
        std = std / (std.mean() + 1e-8)          # normalize so mean scale = 1
        self.register_buffer("scale", std * base_sigma)

    def _noise(self, rep):
        # rep: (1, K, D). scale: (D,). Broadcast over positions.
        return torch.randn_like(rep) * self.scale.to(rep.device)


class LearnableOODMargin(_BaseMargin):
    """Per-dimension log-sigma learned jointly with the network.

    A small L2 penalty on (target_log_sigma - log_sigma) keeps the sigmas
    from collapsing to zero, which would trivially zero the consistency
    loss. target corresponds to the isotropic base_sigma.
    """

    def __init__(self, dim: int = 1408, base_sigma: float = 0.10,
                 weight: float = 0.5, n_samples: int = 1,
                 collapse_penalty: float = 1e-3):
        super().__init__(weight, n_samples)
        import math
        init = math.log(base_sigma)
        self.log_sigma = nn.Parameter(torch.full((dim,), init))
        self.register_buffer("target_log_sigma", torch.tensor(init))
        self.collapse_penalty = float(collapse_penalty)

    def _noise(self, rep):
        sigma = torch.exp(self.log_sigma).to(rep.device)   # (D,)
        return torch.randn_like(rep) * sigma

    def _extra_penalty(self, device):
        # keep sigmas near the target scale on average
        dev = (self.log_sigma - self.target_log_sigma) ** 2
        return self.collapse_penalty * dev.mean().to(device)


class ShiftAlignedOODMargin(_BaseMargin):
    """Per-dimension noise scaled by the measured train->OOD shift vector.

    shift_vec: 1-D tensor length D, e.g. |mean_OOD - mean_train| per
    dimension from the representation-shift analysis. Perturbs precisely
    the directions along which unseen proteins actually differ.
    """

    def __init__(self, shift_vec: torch.Tensor, base_sigma: float = 0.10,
                 weight: float = 0.5, n_samples: int = 1):
        super().__init__(weight, n_samples)
        s = shift_vec.float().abs()
        s = s / (s.mean() + 1e-8)
        self.register_buffer("scale", s * base_sigma)

    def _noise(self, rep):
        return torch.randn_like(rep) * self.scale.to(rep.device)


@torch.no_grad()
def estimate_feature_std(model, loader, device, max_batches: int = 50):
    """Estimate per-dimension std of the fused representation over training data.

    Runs the model forward on up to max_batches, collects the
    `muted_id_representation` populated during the forward pass, and
    returns a length-D std vector. Use its output to construct
    VarianceScaledOODMargin.
    """
    model.eval()
    sums = None
    sqs = None
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        _ = model(batch)
        rep = batch.get("muted_id_representation", None)
        if rep is None:
            continue
        flat = rep.reshape(-1, rep.shape[-1])     # (N, D)
        if sums is None:
            sums = flat.sum(0)
            sqs = (flat ** 2).sum(0)
        else:
            sums += flat.sum(0)
            sqs += (flat ** 2).sum(0)
        count += flat.shape[0]
    if count == 0 or sums is None:
        log.warning("estimate_feature_std collected no representations.")
        return torch.ones(1408)
    mean = sums / count
    var = sqs / count - mean ** 2
    std = torch.sqrt(var.clamp(min=1e-8))
    return std.cpu()
