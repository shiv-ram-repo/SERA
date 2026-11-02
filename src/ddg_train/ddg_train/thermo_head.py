"""
thermo_head
===========
Per-residue thermodynamic decomposition head for SPURS-style stability
prediction.

Replaces the standard MLP head that outputs a single 21-dim stability
score per residue with two parallel heads that output 21-dim *enthalpy*
and *entropy* components per residue. ΔΔG and ΔTm are then derived
from these via standard linearised Gibbs-Helmholtz relations.

Physical model (linearised around physiological T):

    ΔG(T) = ΔH - T · ΔS                                          (1)

For a single point mutation at position i:

    ΔΔG(T) = (h[mut, i] - h[wt, i]) - T · (s[mut, i] - s[wt, i])  (2)

For ΔTm we use the van't Hoff approximation:

    ΔTm ≈ - ΔΔG / ΔS_global                                       (3)

where ΔS_global is a learnable scalar parameterising the protein's
overall entropy of unfolding (positive, dimensionless after scaling).
Equation (3) follows from setting ΔG(Tm) = 0 in Eq. (1) and
differentiating with respect to mutation under small-perturbation
assumption.

T is fixed at 298 K (training temperature). T enters the ΔΔG formula
as a scalar — we normalise it to 1.0 internally so the entropy head's
output is on the same numeric scale as the enthalpy head.

This head produces ΔΔG predictions with the same shape and semantics as
the standard SPURS MLP head, so it slots into the existing forward pass
without changing anything downstream. The ΔTm output is an additional
per-batch scalar that the training loop can choose to supervise.

Anti-symmetry: holds by construction because (2) is anti-symmetric under
wt ↔ mut swap. This is a stronger guarantee than the siamese-swap loss
in antisymmetry.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ThermoHeadConfig:
    """Config for the thermodynamic head."""
    input_dim:        int   = 2560 + 42       # MultimodalDDG-style input
    hidden_dim:       List[int] = field(default_factory=lambda: [512, 512, 512])
    n_aa:             int   = 21              # 20 AAs + pad/unknown
    dropout:          float = 0.1
    T_train:          float = 1.0             # normalised training temperature
    delta_s_init:     float = 1.0             # initial protein ΔS_global
    delta_s_min:      float = 0.1             # floor so we don't divide by ~0


class ThermoHead(nn.Module):
    """
    Two parallel 3-hidden-layer MLPs that output per-AA enthalpy and
    entropy contributions. ΔΔG and ΔTm are derived analytically from
    these outputs via Gibbs-Helmholtz.

    Drop-in replacement for the standard SPURS MLP head — forward()
    returns a tensor of shape (B, K, n_aa) so the parent SPURS model
    can compute ΔΔG by `out[mut_idx] - out[wt_idx]` exactly as before.

    Additionally exposes:
        - forward_components(...) -> (h, s)  per-AA enthalpy / entropy
        - predict_dtm(ddg_pred)   -> per-mutation ΔTm
    """

    def __init__(self, cfg: ThermoHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim     = int(cfg.input_dim)
        hidden_dim = [int(h) for h in cfg.hidden_dim]
        n_aa       = int(cfg.n_aa)
        dropout    = float(cfg.dropout)

        # Shared trunk — same architecture as MultimodalDDG MLP. Two output heads
        # (h and s) branch off the trunk's final hidden state.
        layers = []
        prev = in_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev, h, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Two output heads.
        self.head_h = nn.Linear(hidden_dim[-1], n_aa, bias=True)
        self.head_s = nn.Linear(hidden_dim[-1], n_aa, bias=True)

        # Protein-global ΔS_unfolding (learnable, positive).
        # Stored as log-parameter for positivity.
        self.log_delta_s = nn.Parameter(
            torch.tensor(float(cfg.delta_s_init), dtype=torch.float32).log()
        )

        # Fixed normalised training temperature (used inside the ΔΔG
        # formula). 1.0 means h and s outputs are on the same scale.
        self.register_buffer(
            "T_train",
            torch.tensor(float(cfg.T_train), dtype=torch.float32),
            persistent=False,
        )

        self._initialise_weights()

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ────────────────────────────────────────────────────────────── #
    # Forward variants
    # ────────────────────────────────────────────────────────────── #
    def forward_components(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (h, s) per-AA components.

        h, s shape : (B, K, n_aa) — same as SPURS MLP output.
        """
        x = batch.get("muted_id_representation", batch.get("mpnn_outputs", None))
        if x is None:
            raise KeyError("ThermoHead expects 'muted_id_representation' or "
                           "'mpnn_outputs' in the batch.")
        feat = self.trunk(x)
        h    = self.head_h(feat)
        s    = self.head_s(feat)
        return h, s

    def forward(self, batch: dict, return_embed: bool = False) -> torch.Tensor:
        """
        Return the SPURS-style 21-dim "stability score" per AA:

            score[aa] = h[aa] - T_train * s[aa]

        The parent SPURS model then computes ΔΔG = score[mut] - score[wt],
        which by construction equals:

            (h[mut] - h[wt]) - T_train * (s[mut] - s[wt])

        i.e. Eq. (2) above.
        """
        h, s = self.forward_components(batch)
        if return_embed:
            # Used by MultimodalDDG embedding-extraction code paths.
            return h
        score = h - self.T_train * s
        return score

    # ────────────────────────────────────────────────────────────── #
    # ΔTm derivation
    # ────────────────────────────────────────────────────────────── #
    @property
    def delta_s_global(self) -> torch.Tensor:
        """Protein-global ΔS_unfolding, constrained ≥ delta_s_min."""
        return self.log_delta_s.exp().clamp(min=self.cfg.delta_s_min)

    def predict_dtm(self, ddg_pred: torch.Tensor) -> torch.Tensor:
        """
        Van't Hoff-style approximation:    ΔTm ≈ - ΔΔG / ΔS_global

        Convention: positive ΔΔG = destabilising, negative ΔTm.
        Sign convention matches S571 / S4346 labels.

        Args:
            ddg_pred: (K,) predicted ΔΔG values.

        Returns:
            (K,) predicted ΔTm values in the same units as ΔS_global^-1.
        """
        return -ddg_pred / self.delta_s_global
