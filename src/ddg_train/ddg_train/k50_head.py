"""
k50_head
========
Multi-task K50 prediction head for SPURS-style stability prediction.

The standard SPURS MLP head outputs a single 21-dim score per mutation
position; ΔΔG is then computed as `score[mut_aa] - score[wt_aa]`.

This head replaces that with TWO parallel 21-dim heads, one per protease
assay (trypsin / chymotrypsin), each trained to predict log10_K50 for the
folded protein. At inference, ΔΔG is derived analytically using the
relation:

    ΔG_folding(assay) = -RT·ln10 · (log10_K50_unfolded - log10_K50_folded)
    ΔΔG(assay)        = ΔG_folding(MUT) - ΔG_folding(WT)
                      = -RT·ln10 · [
                            (log10_K50_unf_mut - log10_K50_mut_pred)
                          - (log10_K50_unf_wt  - log10_K50_wt_pred)
                        ]
    ΔΔG_final         = mean( ΔΔG(trypsin), ΔΔG(chymotrypsin) )

(RT·ln10 at T=298K ≈ 1.36 kcal/mol per log10 unit.)

Why this is interesting:
    * 4× supervision per mutation: WT K50_t, MUT K50_t, WT K50_c, MUT K50_c.
    * Two proteases probe different parts of the structure (trypsin cuts at
      K/R; chymotrypsin at F/Y/W/L), so the dual prediction forces the
      encoder to learn position-specific local stability, not just a global
      ΔG scalar.
    * In our K50-verification earlier, the empirically-reconstructed ΔΔG
      from the two K50s correlated with ddG_ML at Pearson = 0.987 with RMS
      = 0.282 kcal/mol — so we have a tight, near-lossless training target.

Compat: forward() returns (B, K, 21) — same shape and meaning as the
standard SPURS MLP head, so the parent SPURS model's gather arithmetic
works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# RT × ln(10) at 298 K — kcal/mol per log10 unit. Tsuboyama et al. (Methods).
RT_LN10_298 = 1.36


@dataclass
class K50HeadConfig:
    input_dim:    int = 1408
    hidden_dim:   List[int] = field(default_factory=lambda: [512, 512, 512])
    n_aa:         int   = 21
    dropout:      float = 0.1


class K50Head(nn.Module):
    """
    Two parallel 3-hidden-layer MLPs predicting log10_K50_t and log10_K50_c
    per AA. Drop-in replacement for the SPURS MLP head.
    """

    def __init__(self, cfg: K50HeadConfig):
        super().__init__()
        self.cfg = cfg
        in_dim     = int(cfg.input_dim)
        hidden_dim = [int(h) for h in cfg.hidden_dim]
        n_aa       = int(cfg.n_aa)
        dropout    = float(cfg.dropout)

        # Shared trunk identical to MultimodalDDG MLP layout
        layers = []
        prev = in_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev, h, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Two output heads (one per protease assay)
        self.head_t = nn.Linear(hidden_dim[-1], n_aa, bias=True)
        self.head_c = nn.Linear(hidden_dim[-1], n_aa, bias=True)

        # Compat shim: expose `.fcs[0]` so external code reading
        # `model.mlp.fcs[0].in_features` still gets the right number.
        # (This module is NOT used in forward — it's a tag for compatibility.)
        self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dim[0])])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def _trunk(self, batch: dict) -> torch.Tensor:
        x = batch.get("muted_id_representation",
                      batch.get("mpnn_outputs", None))
        if x is None:
            raise KeyError("K50Head expects 'muted_id_representation' or "
                           "'mpnn_outputs' in the batch.")
        return self.trunk(x)

    def forward_k50_t(self, batch: dict) -> torch.Tensor:
        """Per-AA predicted log10_K50_t. Shape (B, K, 21)."""
        return self.head_t(self._trunk(batch))

    def forward_k50_c(self, batch: dict) -> torch.Tensor:
        """Per-AA predicted log10_K50_c. Shape (B, K, 21)."""
        return self.head_c(self._trunk(batch))

    def forward_all(self, batch: dict):
        """
        Single-trunk-pass version: returns (score, h_t, h_c).

        Used during multi-task training so we don't re-run the trunk three
        times per forward. `score` is the SPURS-compatible (B, K, 21) tensor;
        `h_t` and `h_c` are the per-assay log10_K50 predictions.

        Sign convention: SPURS stores batch["ddG"] = -float(ddG_ML), so the
        model's `score[mut] - score[wt]` must equal -ddG_ML (negative for
        destabilising mutations). A destabilising mutation has lower K50
        (h_mut < h_wt), so:

            score[mut] - score[wt] = +RT_LN10 * 0.5 * ((h_t_mut + h_c_mut)
                                                    -(h_t_wt + h_c_wt))
                                   = +RT_LN10 * 0.5 * (negative)
                                   = negative,

        which matches SPURS's -ddG_ML convention.
        """
        feat = self._trunk(batch)
        h_t  = self.head_t(feat)
        h_c  = self.head_c(feat)
        score = RT_LN10_298 * 0.5 * (h_t + h_c)
        return score, h_t, h_c

    def forward(self, batch: dict, return_embed: bool = False) -> torch.Tensor:
        """
        SPURS-compatible default forward. Returns a (B, K, 21) tensor
        such that the parent SPURS `score[mut] - score[wt]` gather gives
        predicted ΔΔG in SPURS's internal sign convention (negative for
        destabilising, matching batch["ddG"] = -ddG_ML).

        score[aa] = +RT·ln10 × (h_t[aa] + h_c[aa]) / 2 for each AA.

        For pure-K50 multi-task training, the loss instead uses
        `forward_k50_t` and `forward_k50_c` directly (see K50MultiTaskLoss).
        """
        feat = self._trunk(batch)
        h_t  = self.head_t(feat)
        h_c  = self.head_c(feat)
        if return_embed:
            return h_t
        return RT_LN10_298 * 0.5 * (h_t + h_c)


# ====================================================================== #
# Multi-task K50 loss
# ====================================================================== #
class K50MultiTaskLoss(nn.Module):
    """
    Trains the K50 head against four per-mutation Huber regression targets:
    WT-protein log10_K50_t, MUT-protein log10_K50_t, and the same for
    chymotrypsin (_c).

    Predictions are gathered from the head's 21-dim output at the WT-AA
    and MUT-AA slots respectively. Targets come from the batch.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def _gather(self, h: torch.Tensor, wt_onehot: torch.Tensor,
                mut_onehot: torch.Tensor):
        """
        h          : (B, K, 21)  per-AA predictions
        wt_onehot  : (K, 21)     one-hot for wild-type residue
        mut_onehot : (K, 21)     one-hot for mutant residue

        Returns (pred_wt, pred_mut), each shape (K,).
        """
        h = h.squeeze(0)                      # (K, 21) — MultimodalDDG uses B=1 per WT
        pred_wt  = (h * wt_onehot).sum(-1)
        pred_mut = (h * mut_onehot).sum(-1)
        return pred_wt, pred_mut

    def forward(
        self,
        h_t: torch.Tensor,                # (B, K, 21) from forward_k50_t
        h_c: torch.Tensor,                # (B, K, 21) from forward_k50_c
        wt_onehot: torch.Tensor,          # (K, 21)
        mut_onehot: torch.Tensor,         # (K, 21)
        target_t_wt: torch.Tensor,        # (K,)
        target_t_mut: torch.Tensor,       # (K,)
        target_c_wt: torch.Tensor,        # (K,)
        target_c_mut: torch.Tensor,       # (K,)
    ) -> Tuple[torch.Tensor, dict]:
        pred_t_wt, pred_t_mut = self._gather(h_t, wt_onehot, mut_onehot)
        pred_c_wt, pred_c_mut = self._gather(h_c, wt_onehot, mut_onehot)

        def _huber(p, t):
            valid = torch.isfinite(t) & torch.isfinite(p)
            if valid.sum() == 0:
                return torch.zeros((), device=p.device)
            return F.smooth_l1_loss(p[valid], t[valid], beta=self.beta)

        l_t_wt  = _huber(pred_t_wt,  target_t_wt)
        l_t_mut = _huber(pred_t_mut, target_t_mut)
        l_c_wt  = _huber(pred_c_wt,  target_c_wt)
        l_c_mut = _huber(pred_c_mut, target_c_mut)
        total = (l_t_wt + l_t_mut + l_c_wt + l_c_mut) / 4.0

        return total, {
            "l_t_wt":  float(l_t_wt.item()),
            "l_t_mut": float(l_t_mut.item()),
            "l_c_wt":  float(l_c_wt.item()),
            "l_c_mut": float(l_c_mut.item()),
        }
