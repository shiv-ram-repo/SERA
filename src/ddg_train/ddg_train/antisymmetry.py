"""
antisymmetry
============
Architectural and loss-level enforcement of the thermodynamic anti-symmetry
property:

    ΔΔG(WT → MUT) = - ΔΔG(MUT → WT)

For SPURS-style models the forward output is already structured as
`score[mut] - score[wt]` over a 21-way head, so anti-symmetry IS exactly
satisfied **at a single position** (gather is anti-symmetric in wt/mut).

What is NOT satisfied is the cross-context case: for an inverse mutation
B→A you would normally need to fold the mutant structure and re-extract
features. Inverse mutations are typically scored on the WT structure with
swapped wt/mut indices, which violates anti-symmetry because the encoder
features still come from the WT context, not the mutant context.

This module provides two enforcement mechanisms:

1. **siamese_swap_loss**: at training time, for each mini-batch, run a
   second forward pass with wt/mut indices swapped and add a loss term
   `(pred_forward + pred_swap)^2`. This regularises the encoder to produce
   features whose ΔΔG output is anti-symmetric under index swap, without
   requiring mutant structures.

2. **transitivity_loss**: for a triplet (A→B), (B→C), (A→C), enforces
   ΔΔG(A→C) = ΔΔG(A→B) + ΔΔG(B→C). Sampled stochastically within batch.
   Optional — set weight to 0 to disable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def siamese_swap_loss(
    model_forward_fn,
    batch: dict,
    pred_forward: torch.Tensor,
) -> torch.Tensor:
    """
    Compute (pred_forward + pred_swap)^2 averaged over the batch.

    Args:
        model_forward_fn: callable that takes a batch dict and returns ΔΔG
                          predictions of shape (K,) — same signature as
                          your existing SPURS forward.
        batch:            the batch dict; must contain 'append_tensors' which
                          encodes wt/mut one-hot stacked as [..., :21, ..., 21:42]
                          OR explicit 'wt_onehot' / 'mut_onehot' tensors.
        pred_forward:     ΔΔG predictions from the *forward* direction
                          (already computed; we don't want to redo it).

    Returns:
        A scalar loss equal to mean((pred_forward + pred_swap)^2).
    """
    swapped = _make_swapped_batch(batch)
    pred_swap = model_forward_fn(swapped)

    # Anti-symmetry: pred_forward + pred_swap should be 0.
    return ((pred_forward + pred_swap) ** 2).mean()


def _make_swapped_batch(batch: dict) -> dict:
    """
    Return a shallow copy of `batch` with wt and mut one-hots swapped in
    `append_tensors`.

    SPURS encodes per-mutation features as a 42-dim vector where the first
    21 entries are the wt one-hot and the last 21 are the mut one-hot
    (or vice versa). We swap the two halves.

    If the user wires the model differently (e.g. separate wt_onehot /
    mut_onehot tensors), the caller should swap those instead before
    invoking siamese_swap_loss.
    """
    swapped = dict(batch)  # shallow copy
    if "append_tensors" in batch and batch["append_tensors"] is not None:
        at = batch["append_tensors"]            # (K, 42)
        assert at.size(-1) == 42, f"expected 42-dim append, got {at.size(-1)}"
        at_swapped = torch.cat([at[..., 21:], at[..., :21]], dim=-1)
        swapped["append_tensors"] = at_swapped
    if "mut_tokens" in batch and "tokens" in batch:
        # Swap WT and MUT tokens too if both are present.
        swapped["tokens"], swapped["mut_tokens"] = batch["mut_tokens"], batch["tokens"]
    return swapped


def transitivity_loss(
    pred_ab: torch.Tensor,
    pred_bc: torch.Tensor,
    pred_ac: torch.Tensor,
) -> torch.Tensor:
    """
    L_trans = mean((pred_ac - pred_ab - pred_bc)^2)

    Caller is responsible for constructing the three forward passes
    (typically by sampling random intermediate amino acids per position).
    """
    return ((pred_ac - pred_ab - pred_bc) ** 2).mean()
