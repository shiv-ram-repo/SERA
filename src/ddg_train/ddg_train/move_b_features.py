"""
move_b_features
===============
Use the Phase-1 precomputed Move B coordinate perturbations as INPUT
FEATURES (not as coordinate-space perturbations).

Background:
    Phase 1 cached, for every (WT_protein, mutation) pair, a coordinate
    perturbation δ ∈ R^{L×4×3} obtained by maximising

        log P_ProteinMPNN(MUT_seq | WT_coords + δ_masked)

    under an 8 Å local mask around the mutation site. The cache lives at
    /backup/new_work/runs/move_b_cache/delta_cache.pt and is keyed by
    (WT_name, mut_pos, mut_aa).

    In an earlier experiment we tried applying δ to the input coordinates
    at inference time. That broke every benchmark that uses the WT
    structure (because it pushed the WT toward the mutant structural
    state), but it INCREASED Spearman on Ssym-inverse by +0.069 (where
    the input structure IS the mutant). That signal says δ encodes real
    information about mutation-induced structural change — we just used
    it the wrong way.

This module uses δ instead as a derived per-mutation input feature: the
magnitude of the predicted structural relaxation. Intuitively, mutations
that require large local rearrangement are more likely to be destabilising,
while structurally innocuous mutations are more likely to be neutral.

Per mutation we extract a 3-dim feature:
    [ ||δ at mutation site||,
      mean ||δ_i|| inside the local sphere,
      max  ||δ_i|| inside the local sphere ]

This is appended to the SPURS muted_id_representation, growing the MLP
input from 1408 to 1411. The MLP head's forward signature is unchanged
otherwise, so the rest of the SPURS architecture and training loop work
without modification.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ====================================================================== #
# Augmentor — attaches per-mutation δ features to a batch
# ====================================================================== #
class MoveBFeatureAugmentor:
    """
    Reads the Phase 1 delta cache and, for each batch, produces a
    (K, n_feat) tensor of per-mutation features derived from the cached
    δ coordinate-perturbations.

    The cache is keyed by (WT_name, mut_pos_0indexed, mut_aa_idx) → (L, 4, 3)
    float32 tensor. The augmentor matches batch mutations to cache entries
    using the WT_name (with `.pdb` stripped and `|`→`:` normalised, matching
    SPURS's internal convention).

    Output added to the batch:
        batch["delta_features"]      : (K, n_feat) FloatTensor
        batch["delta_feat_valid"]    : (K,) BoolTensor — True if cache
                                       entry was found for this mutation,
                                       False if missing (features = 0)
    """

    N_FEAT = 3      # [norm_at_mut, mean_local_norm, max_local_norm]

    def __init__(self, cache_path: str, local_radius: float = 8.0):
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Move B cache not found: {cache_path}")
        log.info(f"Loading Move B delta cache from {cache_path}")
        self.cache: Dict[Tuple[str, int, int], torch.Tensor] = \
            torch.load(cache_path, map_location="cpu")
        log.info(f"  Move B cache loaded: {len(self.cache):,} entries")
        self.local_radius = float(local_radius)

    def _normalise_wt_name(self, raw: str) -> str:
        """Match MultimodalDDG's normalisation: strip .pdb, replace | with :."""
        return raw.split(".pdb")[0].replace("|", ":")

    def __call__(self, batch: dict) -> dict:
        """Modify batch in place AND return it."""
        # Resolve WT_name
        name = batch.get("name", None)
        if isinstance(name, (list, tuple)) and len(name) > 0:
            wt_raw = str(name[0])
        elif isinstance(name, str):
            wt_raw = name
        else:
            raise KeyError("Move B augmentor needs batch['name']")
        wt_name = self._normalise_wt_name(wt_raw)

        # mut_ids and mut_aas
        mut_ids = batch.get("mut_ids", [])
        if isinstance(mut_ids, torch.Tensor):
            mut_ids = mut_ids.cpu().tolist()
        K = len(mut_ids)

        append_tensors = batch.get("append_tensors")
        if append_tensors is None:
            raise KeyError("Move B augmentor needs batch['append_tensors']")
        mut_aas = append_tensors[..., 21:42].argmax(dim=-1).cpu().tolist()

        # CA coords for local-mask calculation (we need this to know which
        # residues are inside the sphere). Use the WT coords from the batch.
        X = batch.get("X")
        if X is None:
            raise KeyError("Move B augmentor needs batch['X']")
        # X is (1, L, 4, 3); CA atom is index 1
        CA = X[0, :, 1, :]  # (L, 3)
        L = CA.shape[0]

        features = torch.zeros((K, self.N_FEAT), dtype=torch.float32)
        valid    = torch.zeros((K,), dtype=torch.bool)

        for k in range(K):
            pos = int(mut_ids[k])
            aa  = int(mut_aas[k])
            d = self.cache.get((wt_name, pos, aa), None)
            if d is None:
                continue
            # d is (L_cache, 4, 3). If L_cache differs from L, align by
            # truncation; this should be rare but be robust.
            if d.shape[0] != L:
                use_L = min(L, d.shape[0])
                d_aligned = torch.zeros((L, 4, 3), dtype=torch.float32)
                d_aligned[:use_L] = d[:use_L]
                d = d_aligned

            # Per-residue δ norms
            per_res_norm = d.pow(2).sum(dim=(-1, -2)).sqrt()  # (L,)

            # Identify residues in local sphere (CA-CA distance < radius)
            if 0 <= pos < L:
                dist = (CA - CA[pos]).pow(2).sum(-1).sqrt().cpu()  # (L,)
                in_sphere = dist < self.local_radius
                if in_sphere.sum() > 0:
                    local_norms = per_res_norm[in_sphere]
                    feat_mean = local_norms.mean().item()
                    feat_max  = local_norms.max().item()
                else:
                    feat_mean = per_res_norm.mean().item()
                    feat_max  = per_res_norm.max().item()
                feat_at_mut = per_res_norm[pos].item()
            else:
                feat_at_mut = 0.0
                feat_mean   = per_res_norm.mean().item()
                feat_max    = per_res_norm.max().item()

            features[k, 0] = feat_at_mut
            features[k, 1] = feat_mean
            features[k, 2] = feat_max
            valid[k]       = True

        batch["delta_features"]   = features
        batch["delta_feat_valid"] = valid
        return batch


# ====================================================================== #
# Move B head: standard MLP that also reads delta_features
# ====================================================================== #
@dataclass
class MoveBHeadConfig:
    input_dim:        int  = 1408
    delta_feat_dim:   int  = 3
    hidden_dim:       List[int] = field(default_factory=lambda: [512, 512, 512])
    n_aa:             int  = 21
    dropout:          float = 0.1


class MoveBHead(nn.Module):
    """
    Drop-in replacement for the SPURS MLP head.

    Reads:
        batch["muted_id_representation"]  : (1, K, input_dim)
        batch["delta_features"]           : (K, delta_feat_dim)   [optional]
        batch["delta_feat_valid"]         : (K,)                   [optional]

    If delta_features is present, concatenates to the per-mutation representation
    and runs the MLP on (input_dim + delta_feat_dim). If absent (e.g. during
    baseline eval), uses zeros for the delta channel — this lets the same
    checkpoint be evaluated WITH or WITHOUT Move B features for comparison.
    """

    def __init__(self, cfg: MoveBHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_dim     = int(cfg.input_dim) + int(cfg.delta_feat_dim)
        hidden_dim = [int(h) for h in cfg.hidden_dim]
        n_aa       = int(cfg.n_aa)
        dropout    = float(cfg.dropout)

        # Layers laid out the same way as the MultimodalDDG MLP so existing
        # state-dict introspection (e.g. `head.fcs[0].in_features`) works.
        self.fcs = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim[0], bias=True)]
            + [nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=True)
               for i in range(len(hidden_dim) - 1)]
            + [nn.Linear(hidden_dim[-1], n_aa, bias=True)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(len(hidden_dim))]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: dict, return_embed: bool = False) -> torch.Tensor:
        x = batch.get("muted_id_representation",
                      batch.get("mpnn_outputs", None))
        if x is None:
            raise KeyError("MoveBHead expects 'muted_id_representation' in batch")

        # x shape: (1, K, input_dim).
        # Build the delta channel: (K, delta_feat_dim).
        K = x.shape[1]
        delta = batch.get("delta_features", None)
        if delta is None:
            # Allow inference without Move B (zeros pad)
            delta = torch.zeros((K, self.cfg.delta_feat_dim),
                                dtype=x.dtype, device=x.device)
        else:
            delta = delta.to(x.device).to(x.dtype)
            if delta.dim() == 2 and delta.shape[0] == K:
                pass   # OK
            elif delta.dim() == 2 and delta.shape[0] == 1 and delta.shape[1] == K * self.cfg.delta_feat_dim:
                # Defensive: in case it got flattened
                delta = delta.view(K, self.cfg.delta_feat_dim)
            else:
                # Shape mismatch: fall back to zeros rather than crash
                delta = torch.zeros((K, self.cfg.delta_feat_dim),
                                    dtype=x.dtype, device=x.device)

        # Reshape delta to (1, K, delta_feat_dim) and concat with x.
        delta_3d = delta.unsqueeze(0)
        x_aug = torch.cat([x, delta_3d], dim=-1)        # (1, K, input + delta_dim)

        # Standard MLP forward
        h = x_aug
        for i in range(len(self.fcs) - 1):
            h = self.fcs[i](h)
            h = self.dropouts[i](h)
            h = F.gelu(h)
        if return_embed:
            return h
        return self.fcs[-1](h)
