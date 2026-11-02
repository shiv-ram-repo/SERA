"""
feature_mixup.py
================
Manifold-mixup consistency in the fused feature space, as a
better-motivated relative of the isotropic noise-consistency loss.

Instead of perturbing a single mutation's representation with random
noise, we interpolate between the per-position representations of two
different mutations and ask the head's prediction on the interpolate to
match the same interpolation of the two clean predictions:

    rep_mix = lam * rep_i + (1 - lam) * rep_j
    target  = lam * ddg_i + (1 - lam) * ddg_j
    L_mix   = ( head(rep_mix) - target )^2

This encourages the head to behave smoothly (locally linearly) along the
feature-space paths between real mutations, which is a stronger and more
data-grounded regularizer than isotropic noise: the perturbation
directions are exactly the directions between real examples, not random.
Connects to manifold mixup (Verma et al. 2019) and gives a principled
answer to "why is a consistency loss the right regularizer here."

Shares the encoder pass (head-only re-run), same per-step cost profile as
the noise-consistency loss.
"""

from __future__ import annotations
import torch
import torch.nn as nn


def _gather_ddg(score, wt_onehot, mut_onehot):
    score_wt = (score * wt_onehot).sum(-1)
    score_mut = (score * mut_onehot).sum(-1)
    return score_mut - score_wt


class FeatureMixupConsistency(nn.Module):
    def __init__(self, alpha: float = 2.0, weight: float = 0.5):
        """alpha parameterizes the Beta(alpha, alpha) distribution for lam."""
        super().__init__()
        self.alpha = float(alpha)
        self.weight = float(weight)

    def forward(self, model, batch, ddg_clean, valid):
        rep = batch.get("muted_id_representation", None)
        at = batch.get("append_tensors", None)
        mlp = getattr(model, "mlp", None)
        if rep is None or at is None or mlp is None:
            return (torch.zeros((), device=ddg_clean.device),
                    {"mixup_loss": 0.0, "mixup_skipped": True})

        # Work on valid mutations only
        wt_onehot = at[..., :21].float()
        mut_onehot = at[..., 21:42].float()

        # rep: (1, K, D). Flatten position dim to (K, D).
        rep_k = rep.squeeze(0)                       # (K, D)
        K = rep_k.shape[0]
        if K < 2:
            return (torch.zeros((), device=ddg_clean.device),
                    {"mixup_loss": 0.0, "mixup_skipped": True})

        # Random permutation to pick partner j for each i
        perm = torch.randperm(K, device=rep_k.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(rep_k.device)

        rep_mix = lam * rep_k + (1 - lam) * rep_k[perm]     # (K, D)
        rep_mix = rep_mix.unsqueeze(0)                      # (1, K, D)

        noisy_batch = dict(batch)
        noisy_batch["muted_id_representation"] = rep_mix
        score = mlp(noisy_batch).squeeze(0)                # (K, 21)
        ddg_mix = _gather_ddg(score, wt_onehot, mut_onehot)  # (K,)

        target = lam * ddg_clean.detach() + (1 - lam) * ddg_clean.detach()[perm]
        diff = ddg_mix - target
        if valid is not None:
            diff = diff[valid]
        if diff.numel() == 0:
            return (torch.zeros((), device=ddg_clean.device),
                    {"mixup_loss": 0.0, "mixup_skipped": True})
        loss = (diff ** 2).mean()
        return loss, {"mixup_loss": float(loss.item()), "mixup_skipped": False}
