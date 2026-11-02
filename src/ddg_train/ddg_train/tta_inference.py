"""
tta_inference
=============
Test-time augmentation (TTA) for SPURS-style stability prediction.

At inference time, we generate `n_aug` perturbed copies of each input
structure by adding small Gaussian noise to the backbone Cα coordinates,
then average the predictions. This is a cheap form of conformational
ensembling that does not require AF2-RAVE / Str2Str.

Empirical effect (expected, not measured):
    +0.005 to +0.025 Spearman on S669-style OOD benchmarks
    No change in clean accuracy
    Acts as a regulariser on prediction variance — useful if downstream
    decisions are based on top-K mutant ranking.

Two perturbation kernels are supported:
    - "gaussian": isotropic Gaussian noise on Cα positions, sigma controlled
    - "rotamer":  Cα-fixed, only side-chain orientation noise (NOOP for the
                  SPURS encoder, which only consumes backbone, so reduces to
                  zero-augmentation; kept for API completeness).

Usage in evaluate.py:
    from tta_inference import tta_predict
    preds = tta_predict(model, batch, n_aug=8, sigma=0.1)
"""

from __future__ import annotations

from typing import Callable

import torch


def perturb_backbone(
    coords: torch.Tensor,
    sigma: float = 0.1,
    keep_first: bool = True,
) -> torch.Tensor:
    """
    Add isotropic Gaussian noise to backbone coordinates.

    Args:
        coords:     (B, L, 4, 3) backbone atom coordinates (N, Cα, C, O)
                    in the SPURS convention.
        sigma:      noise stddev in angstroms. Typical: 0.05 to 0.2.
        keep_first: leave the first residue fixed so the global frame
                    doesn't drift.

    Returns:
        Perturbed coordinates, same shape.
    """
    noise = torch.randn_like(coords) * sigma
    if keep_first:
        noise[:, 0] = 0.0
    return coords + noise


@torch.no_grad()
def tta_predict(
    forward_fn: Callable,
    batch: dict,
    n_aug: int = 8,
    sigma: float = 0.1,
    coord_key: str = "X",
) -> torch.Tensor:
    """
    Run `n_aug` perturbed forward passes and return the mean prediction.
    If 'use_move_b' is True, bypasses random TTA and applies exact delta 
    coordinates for each mutation.
    """
    # --- THE MOVE B FUSION HOOK ---
    if batch.get("use_move_b", False) and "delta_coords" in batch:
        coords_clean = batch[coord_key]  # Original WT backbone, shape (1, L, 4, 3)
        deltas = batch["delta_coords"]   # Precomputed deltas, shape (K, L, 4, 3)
        K = deltas.shape[0]
        
        preds = []
        # Loop through each mutation to avoid an Out-Of-Memory (OOM) crash
        for k in range(K):
            # 1. Apply the specific mutant delta to the backbone
            batch[coord_key] = coords_clean + deltas[k:k+1].to(coords_clean.device)
            
            # 2. Run the full forward pass on this new mutant geometry
            out = forward_fn(batch).view(-1) 
            
            # 3. We ONLY want the k-th prediction from this specific mutant's pass
            preds.append(out[k])
            
        batch[coord_key] = coords_clean  # Restore original WT coordinates
        return torch.stack(preds, dim=0)
    # ------------------------------

    # Standard TTA / Baseline Inference
    if n_aug <= 1:
        return forward_fn(batch)

    coords_clean = batch[coord_key]
    preds = []
    for _ in range(n_aug):
        batch[coord_key] = perturb_backbone(coords_clean, sigma=sigma)
        preds.append(forward_fn(batch))
    batch[coord_key] = coords_clean  # restore

    return torch.stack(preds, dim=0).mean(dim=0)
