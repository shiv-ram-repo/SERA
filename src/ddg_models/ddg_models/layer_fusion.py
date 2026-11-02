"""
layer_fusion
============
Uses the ESM-2 layers that the current readout discards.

The backbone runs 33 transformer layers, injects structure through an adapter
at one of them, and then the readout consumes only `representations[-1]`. The
layer-wise probing literature is consistent that this is the wrong default: for
protein language models the last layer is rarely the best source of embeddings,
intermediate layers usually carry more task-relevant signal, and — the point
that matters here — the best layer is dataset- and task-dependent rather than
fixed. If the optimal depth is not knowable in advance, it should not be
hard-coded.

Two fusion modes, both operating on the stack of selected layers
(B, T, n_layers, E) and collapsing it to (B, T, E) so that everything
downstream of the encoder is unchanged:

ScalarMix
    ELMo-style learned convex combination over layers with a global scale,
    gamma * sum_l softmax(w)_l * LN_l(h_l). One weight per layer; the model
    learns where in the stack the signal for stability lives.

LayerMoE
    Depth routing. A per-residue router scores the layers and the top-k are
    mixed with renormalised softmax weights. Different residues can draw on
    different depths — a buried hydrophobic position may be best described by
    mid-stack packing features while a surface position may need late-stack
    semantics. This is mixture-of-experts over depth rather than over width,
    which is the natural form here because the layer-selection literature says
    the useful depth is conditional, and because it adds almost no parameters:
    the measured capacity pathology on this task means any fix must be close to
    free.

Both add on the order of 10^4 parameters, against the 1.26M dense head.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarMix(nn.Module):
    """Learned convex combination over layers (ELMo-style)."""

    def __init__(self, n_layers: int, dim: int, layer_norm: bool = True,
                 init: str = "final", init_mass: float = 0.9) -> None:
        """init="final" starts with almost all mass on the last layer, so the
        model begins equivalent to the fixed-final-layer baseline and must earn
        any deviation. Starting uniform instead begins in a degraded state --
        an average over layers that are mostly structure-blind -- which early
        stopping may never recover from, confounding the experiment."""
        super().__init__()
        w0 = torch.zeros(n_layers)
        if init == "final" and n_layers > 1:
            m = min(max(init_mass, 1.0 / n_layers + 1e-6), 0.999)
            w0[-1] = math.log(m * (n_layers - 1) / (1.0 - m))
        self.weights = nn.Parameter(w0)
        self.residual = (init == "residual")
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.norms = (
            nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
            if layer_norm else None
        )

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        # stack: (B, T, n_layers, E); the last entry is the final layer.
        final = stack[:, :, -1]
        if self.residual:
            # Identity at initialisation: alpha starts at zero, so the readout
            # receives exactly the final-layer representation the fixed-layer
            # baseline uses, untouched by LayerNorm or rescaling. Anything the
            # other layers contribute has to be earned additively.
            if self.norms is not None:
                mixed = torch.stack(
                    [self.norms[i](stack[:, :, i]) for i in range(stack.shape[2])], dim=2)
            else:
                mixed = stack
            w = F.softmax(self.weights, dim=0).view(1, 1, -1, 1)
            return final + self.alpha * (mixed * w).sum(dim=2)

        if self.norms is not None:
            stack = torch.stack(
                [self.norms[i](stack[:, :, i]) for i in range(stack.shape[2])], dim=2
            )
        w = F.softmax(self.weights, dim=0).view(1, 1, -1, 1)
        return self.gamma * (stack * w).sum(dim=2)

    @torch.no_grad()
    def layer_weights(self) -> torch.Tensor:
        return F.softmax(self.weights, dim=0).detach().cpu()


class LayerMoE(nn.Module):
    """Per-residue routing over encoder depth.

    The router reads the final-layer representation (always present, and the
    cheapest summary of the residue) and scores every candidate layer. The
    top-k layers are combined with renormalised weights. Auxiliary load
    balancing keeps the router from collapsing onto a single depth, which would
    silently reduce this to the fixed-layer baseline.
    """

    def __init__(self, n_layers: int, dim: int, top_k: int = 2,
                 layer_norm: bool = True, lb_weight: float = 1e-2,
                 residual: bool = True) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.top_k = max(1, min(top_k, n_layers))
        self.router = nn.Linear(dim, n_layers, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.gamma = nn.Parameter(torch.ones(1))
        self.norms = (
            nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
            if layer_norm else None
        )
        self.lb_weight = float(lb_weight)
        self.residual = residual
        self.alpha = nn.Parameter(torch.zeros(1))
        self._aux = None
        self.last_layer_usage = None

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        B, T, nL, E = stack.shape
        if self.norms is not None:
            stack = torch.stack([self.norms[i](stack[:, :, i]) for i in range(nL)], dim=2)

        logits = self.router(stack[:, :, -1])                 # (B, T, nL)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)
        gates = torch.zeros_like(probs).scatter(-1, topi, topv)   # (B, T, nL)

        out = (stack * gates.unsqueeze(-1)).sum(dim=2)         # (B, T, E)
        if self.residual:
            out = stack[:, :, -1] + self.alpha * out

        flat = probs.reshape(-1, nL)
        mask = torch.zeros_like(flat).scatter(
            1, topi.reshape(-1, self.top_k), 1.0)
        f = mask.mean(0)
        P = flat.mean(0)
        self._aux = nL * torch.sum(f * P) * self.lb_weight
        self.last_layer_usage = f.detach()
        return out if self.residual else self.gamma * out

    def pop_aux(self):
        a, self._aux = self._aux, None
        return a


def build_layer_fusion(mode: str, n_layers: int, dim: int, top_k: int = 2,
                       init: str = "final"):
    if mode == "scalar_mix":
        return ScalarMix(n_layers, dim, init=init)
    if mode == "layer_moe":
        return LayerMoE(n_layers, dim, top_k=top_k, residual=(init == "residual"))
    raise ValueError(f"unknown layer_fusion mode: {mode}")


def default_layer_indices(num_layers: int = 33, n_select: int = 8) -> List[int]:
    """Evenly spaced layers spanning mid-stack to the top, final layer included.

    Mid-stack is included deliberately: probing studies repeatedly find the most
    informative representations around mid-depth, where task-relevant signal is
    retained but over-specialisation has not yet set in.
    """
    lo = max(1, num_layers // 3)
    idx = sorted(set(int(round(x)) for x in
                     torch.linspace(lo, num_layers, n_select).tolist()))
    return idx
