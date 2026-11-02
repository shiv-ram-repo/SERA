"""
moe_head
========
Sparse Mixture-of-Experts readout for ΔΔG prediction.

Motivation
----------
The dense head applies one fixed MLP to every mutation, in every protein, in
every structural environment. Stratified analysis of the dense model shows the
error profile is highly heterogeneous: accuracy on large-effect and
destabilizing mutations differs markedly from the neutral/stabilizing regime,
yet all of them are served by the same parameters. An MoE readout lets
specialised experts cover different regions of that heterogeneous input space
while keeping per-mutation compute close to the dense head (only `top_k` of
`num_experts` run per mutation).

Interface
---------
Drop-in replacement for `MLP`: constructed from an `MLPConfig`, called as
`head(batch)` where `batch["muted_id_representation"]` has shape (B, K, D),
returning (B, K, 21). This means the parent model can simply assign the MoE
module to `self.mlp` and nothing else in the forward path changes.

Routing
-------
* Router input is the per-mutation representation, optionally concatenated with
  cheap structural-environment descriptors (CA neighbour counts at 8/10/12 Å
  around the mutated position, plus relative sequence position). Those
  descriptors are computed under `no_grad` and detached — they condition
  routing only.
* Noisy top-k gating (Shazeer et al., 2017) during training for exploration;
  deterministic top-k at eval.
* Gate weights are a softmax restricted to the selected top-k experts,
  renormalised so they sum to 1.
* Sparse dispatch: each expert only sees the mutations routed to it.

Auxiliary losses
----------------
* Load-balancing loss (Switch Transformer): E * Σ_i f_i · P_i, minimised at
  uniform assignment.
* Router z-loss (ST-MoE): mean(logsumexp(logits)²), keeps router logits from
  drifting to large magnitudes.

Both are accumulated across the (possibly several) forward passes performed in
one optimiser step — main pass, Siamese swapped pass, OOD-margin perturbed pass
— and retrieved once per step via `pop_aux()`. Accumulation happens only in
training mode, so evaluation never retains graph state.

Shared expert
-------------
Optionally an always-on expert is added to the routed combination
(DeepSeek-MoE style). This guarantees a dense path through the head, so the
model degrades gracefully to something close to the dense baseline if routing
is uninformative.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .mlp import MLPConfig


def _cfg_get(cfg, key, default=None):
    """Read a key from a dict / dataclass / DictConfig with a default."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class _Expert(nn.Module):
    """One expert: the same stack the dense head uses (GELU + dropout MLP)."""

    def __init__(self, input_dim: int, hidden_dim: List[int],
                 output_dim: int, dropout: float) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dim)
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)]
            + [nn.Linear(dims[-1], output_dim, bias=True)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(len(dims) - 1)]
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fcs) - 1):
            x = self.fcs[i](x)
            x = self.dropouts[i](x)
            x = F.gelu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcs[-1](self.embed(x))



class _LowRankExpertStack(nn.Module):
    """A shared MLP trunk with per-expert low-rank weight deltas.

    Motivated by the capacity pathology measured on this task: a dense readout
    larger than roughly 1.3M parameters degrades cross-dataset generalisation,
    so full-width experts pay a capacity penalty that cancels the benefit of
    specialisation. Here every expert shares one trunk and differs only by a
    rank-r update per layer,

        W_e = W + (scale / r) * B_e A_e,

    costing r*(d_in + d_out) parameters per expert per layer instead of
    d_in*d_out. With r = 8 and sixteen experts the whole head stays close to the
    size of the dense baseline while still allowing per-expert specialisation.
    """

    def __init__(self, input_dim: int, hidden_dim: List[int], output_dim: int,
                 dropout: float, n_experts: int, rank: int = 8,
                 scale: float = 1.0) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dim) + [output_dim]
        self.n_layers = len(dims) - 1
        self.rank, self.scale, self.n_experts = rank, scale, n_experts

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(self.n_layers)]
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers - 1)])
        # Low-rank deltas: A maps in->r, B maps r->out, per expert per layer.
        self.A = nn.ParameterList([
            nn.Parameter(torch.zeros(n_experts, dims[i], rank)) for i in range(self.n_layers)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.zeros(n_experts, rank, dims[i + 1])) for i in range(self.n_layers)
        ])
        for m in self.fcs:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for a in self.A:                       # B stays zero so experts start identical
            nn.init.normal_(a, std=0.02)

    def _layer(self, x, i, e, embed_only=False):
        y = self.fcs[i](x)
        y = y + (self.scale / self.rank) * ((x @ self.A[i][e]) @ self.B[i][e])
        return y

    def forward_expert(self, x: torch.Tensor, e: int, return_embed: bool = False):
        for i in range(self.n_layers - 1):
            x = self._layer(x, i, e)
            x = self.dropouts[i](x)
            x = F.gelu(x)
        if return_embed:
            return x
        return self._layer(x, self.n_layers - 1, e)

class MoEReadout(BaseModel):
    """Sparse MoE replacement for the dense ΔΔG readout head."""

    _default_cfg = MLPConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        c = self.cfg

        input_dim = int(_cfg_get(c, "input_dim"))
        output_dim = int(_cfg_get(c, "output_dim", 21))
        dropout = float(_cfg_get(c, "dropout", 0.1))

        hidden = _cfg_get(c, "moe_expert_hidden", None)
        if hidden is None:
            hidden = _cfg_get(c, "hidden_dim", [512, 512, 512])
        if hasattr(hidden, "__iter__") and not isinstance(hidden, str):
            hidden = [int(h) for h in list(hidden)]
        else:
            hidden = [int(hidden)] * 3

        self.num_experts = int(_cfg_get(c, "moe_num_experts", 8))
        self.top_k = max(1, min(int(_cfg_get(c, "moe_top_k", 2)), self.num_experts))
        self.noise_eps = float(_cfg_get(c, "moe_noise_eps", 1.0))
        self.use_shared = bool(_cfg_get(c, "moe_shared_expert", True))
        self.router_struct = bool(_cfg_get(c, "moe_router_struct", True))
        self.lb_weight = float(_cfg_get(c, "moe_lb_weight", 1.0))
        self.z_weight = float(_cfg_get(c, "moe_z_weight", 1e-3))
        # route_mode: "learned" | "chemistry" | "hybrid"
        #   chemistry = fixed partition by wild-type residue class (control)
        #   hybrid    = learned router biased by the chemistry prior
        self.route_mode = str(_cfg_get(c, "moe_route_mode", "learned"))
        # balance_mode: "aux" (Switch loss) | "loss_free" (DeepSeek expert bias)
        self.balance_mode = str(_cfg_get(c, "moe_balance_mode", "aux"))
        self.bias_gamma = float(_cfg_get(c, "moe_bias_gamma", 1e-3))
        self.n_shared = int(_cfg_get(c, "moe_shared_count", 1)) if self.use_shared else 0
        self.chem_prior_scale = float(_cfg_get(c, "moe_chem_prior_scale", 2.0))
        # expert_mode: "full" (independent MLP per expert) | "lowrank"
        # (shared trunk + per-expert low-rank deltas)
        self.expert_mode = str(_cfg_get(c, "moe_expert_mode", "full"))
        self.lora_rank = int(_cfg_get(c, "moe_lora_rank", 8))
        self.lora_scale = float(_cfg_get(c, "moe_lora_scale", 1.0))

        if self.expert_mode == "lowrank":
            self.lowrank = _LowRankExpertStack(
                input_dim, hidden, output_dim, dropout,
                self.num_experts, self.lora_rank, self.lora_scale)
            self.experts = nn.ModuleList()
        else:
            self.lowrank = None
            self.experts = nn.ModuleList(
                [_Expert(input_dim, hidden, output_dim, dropout)
                 for _ in range(self.num_experts)]
            )
        self.shared_experts = nn.ModuleList(
            [_Expert(input_dim, hidden, output_dim, dropout) for _ in range(self.n_shared)]
        )
        # backward-compatible alias for checkpoints / diagnostics
        self.shared_expert = self.shared_experts[0] if self.n_shared > 0 else None

        # Router. `n_struct` extra descriptor dims when structural routing is on.
        self.n_struct = 4 if self.router_struct else 0
        router_in = input_dim + self.n_struct
        self.router = nn.Linear(router_in, self.num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        # Separate head producing the per-expert noise scale (Shazeer noisy top-k).
        self.router_noise = nn.Linear(router_in, self.num_experts, bias=False)
        nn.init.zeros_(self.router_noise.weight)

        # Loss-free load balancing (Wang et al. 2024): a per-expert bias added to
        # the routing score for SELECTION only, nudged after each step by observed
        # utilisation. Carries no gradient, so it balances load without the
        # auxiliary-loss gradients that can degrade quality.
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))

        # Chemistry partition of the 21-letter alphabet into residue classes.
        # ACDEFGHIKLMNPQRSTVWYX
        groups = {"A": 0, "V": 0, "L": 1, "I": 1, "M": 1, "F": 2, "W": 2, "Y": 2,
                  "S": 3, "T": 3, "N": 3, "Q": 3, "K": 4, "R": 4, "H": 4,
                  "D": 5, "E": 5, "G": 6, "P": 6, "C": 7, "X": 7}
        alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        table = [groups[a] % self.num_experts for a in alphabet]
        self.register_buffer("chem_table", torch.tensor(table, dtype=torch.long))

        # Auxiliary-loss accumulators (training only).
        self._aux_sum: Optional[torch.Tensor] = None
        self._aux_n: int = 0
        # Diagnostics from the most recent forward (detached, for logging).
        self.last_expert_fraction: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Compatibility shim
    # ------------------------------------------------------------------ #
    @property
    def fcs(self):
        if self.lowrank is not None:
            return self.lowrank.fcs
        """Expose an `fcs`-like view so code that introspects the dense head
        (e.g. `head.fcs[0].in_features` for head-type detection in evaluate.py)
        keeps working. Returns the first expert's layer stack: same input and
        output dimensions as the dense head it replaces."""
        return self.experts[0].fcs

    # ------------------------------------------------------------------ #
    # Structural-environment descriptors for the router
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _structural_features(self, batch, n_tokens: int, device, dtype):
        """CA neighbour counts at 8/10/12 Å around each mutated position plus
        relative sequence position. Returns (n_tokens, 4); zeros on any
        mismatch or missing data so routing degrades to representation-only."""
        zeros = torch.zeros(n_tokens, 4, device=device, dtype=dtype)
        X = batch.get("X", None)
        mut_ids = batch.get("mut_ids", None)
        if X is None or mut_ids is None:
            return zeros
        try:
            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X)
            if X.dim() == 4:          # (B, L, 4, 3)
                X = X[0]
            if X.dim() != 3 or X.shape[1] < 2:
                return zeros
            ca = X[:, 1, :].to(device=device, dtype=torch.float32)   # (L, 3)
            L = ca.shape[0]

            if not isinstance(mut_ids, torch.Tensor):
                mut_ids = torch.as_tensor(mut_ids)
            idx = mut_ids.to(device).long().reshape(-1)
            if idx.numel() != n_tokens:
                # e.g. the return_logist path enumerates L*20 pseudo-mutations
                return zeros
            idx = idx.clamp(0, L - 1)

            good = torch.isfinite(ca).all(dim=1)
            ca = torch.where(good.unsqueeze(-1), ca, torch.zeros_like(ca))
            d = torch.cdist(ca[idx], ca)                              # (K, L)
            d = torch.where(good.unsqueeze(0), d, torch.full_like(d, 1e6))

            feats = []
            for r in (8.0, 10.0, 12.0):
                cnt = (d < r).sum(dim=-1).to(torch.float32) - 1.0
                feats.append(torch.log1p(cnt.clamp(min=0.0)) / 4.0)
            feats.append(idx.to(torch.float32) / max(L - 1, 1))
            out = torch.stack(feats, dim=-1)
            if not torch.isfinite(out).all():
                return zeros
            return out.to(dtype=dtype)
        except Exception:
            return zeros

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _chem_expert(self, batch, n_tokens: int, device):
        """Expert index implied by the wild-type residue class, or None."""
        at = batch.get("append_tensors", None)
        if at is None:
            return None
        try:
            a = at if isinstance(at, torch.Tensor) else torch.as_tensor(at)
            a = a.reshape(-1, a.shape[-1])
            if a.shape[0] != n_tokens or a.shape[-1] < 21:
                return None
            wt = a[:, :21].argmax(dim=-1).to(device)
            return self.chem_table.to(device)[wt.clamp(0, 20)]
        except Exception:
            return None

    def _route(self, flat_x: torch.Tensor, struct: torch.Tensor, chem=None):
        router_in = flat_x if self.n_struct == 0 else torch.cat([flat_x, struct], dim=-1)
        logits = self.router(router_in)                               # (N, E)

        # Fixed chemistry partition: the control that asks whether the LEARNED
        # router earns its place over hard-coded residue-class routing.
        if self.route_mode == "chemistry" and chem is not None:
            logits = torch.full_like(logits, -1e4).scatter(
                1, chem.view(-1, 1), 0.0)
        elif self.route_mode == "hybrid" and chem is not None:
            prior = torch.zeros_like(logits).scatter(1, chem.view(-1, 1), 1.0)
            logits = logits + self.chem_prior_scale * prior

        if self.training and self.noise_eps > 0 and self.route_mode != "chemistry":
            noise_scale = F.softplus(self.router_noise(router_in)) + 1e-6
            logits = logits + torch.randn_like(logits) * noise_scale * self.noise_eps

        probs = F.softmax(logits, dim=-1)                             # (N, E)

        # Selection score may carry the loss-free balancing bias; the GATE
        # weights always come from the unbiased probabilities.
        score = probs if self.balance_mode != "loss_free" else probs + self.expert_bias
        _, topk_idx = torch.topk(score, self.top_k, dim=-1)           # (N, k)
        topk_val = probs.gather(1, topk_idx)
        topk_val = topk_val / (topk_val.sum(dim=-1, keepdim=True) + 1e-9)

        gates = torch.zeros_like(probs).scatter(1, topk_idx, topk_val)

        if self.training and self.balance_mode == "loss_free":
            self._update_bias(topk_idx)
        return gates, probs, logits, topk_idx

    @torch.no_grad()
    def _update_bias(self, topk_idx: torch.Tensor) -> None:
        """Nudge per-expert bias toward uniform utilisation (no gradient)."""
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).float()
        target = counts.mean()
        self.expert_bias += self.bias_gamma * torch.sign(target - counts)

    def _aux_losses(self, probs: torch.Tensor, logits: torch.Tensor,
                    topk_idx: torch.Tensor) -> torch.Tensor:
        N, E = probs.shape
        mask = torch.zeros_like(probs).scatter(1, topk_idx, 1.0)      # (N, E)
        f = mask.sum(dim=0) / (N * self.top_k + 1e-9)                 # assignment fraction
        P = probs.mean(dim=0)                                         # mean router prob
        lb = E * torch.sum(f * P)
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        self.last_expert_fraction = f.detach()
        return self.lb_weight * lb + self.z_weight * z

    # ------------------------------------------------------------------ #
    # Auxiliary-loss accumulation across the forward passes in one step
    # ------------------------------------------------------------------ #
    def _accumulate_aux(self, aux: torch.Tensor) -> None:
        if not self.training:
            return
        self._aux_sum = aux if self._aux_sum is None else self._aux_sum + aux
        self._aux_n += 1

    def pop_aux(self) -> Optional[torch.Tensor]:
        """Mean auxiliary loss over the forward passes since the last call.
        Returns None if nothing was accumulated. Resets the accumulator."""
        if self._aux_sum is None or self._aux_n == 0:
            self._aux_sum, self._aux_n = None, 0
            return None
        aux = self._aux_sum / self._aux_n
        self._aux_sum, self._aux_n = None, 0
        return aux

    # ------------------------------------------------------------------ #
    def forward(self, batch, return_embed: bool = False):
        x = batch.get("muted_id_representation", batch.get("mpnn_outputs", None))
        if x is None:
            raise KeyError("MoEReadout: batch has no 'muted_id_representation'")

        squeeze_back = False
        if x.dim() == 2:                      # (K, D) — treat as B=1
            x = x.unsqueeze(0)
            squeeze_back = True
        B, K, D = x.shape
        flat_x = x.reshape(B * K, D)
        N = flat_x.shape[0]

        struct = (
            self._structural_features(batch, N, flat_x.device, flat_x.dtype)
            if self.router_struct else
            torch.zeros(N, 0, device=flat_x.device, dtype=flat_x.dtype)
        )

        chem = (self._chem_expert(batch, N, flat_x.device)
                if self.route_mode in ("chemistry", "hybrid") else None)
        gates, probs, logits, topk_idx = self._route(flat_x, struct, chem)
        if self.balance_mode == "aux":
            self._accumulate_aux(self._aux_losses(probs, logits, topk_idx))
        else:
            # still record utilisation for diagnostics
            with torch.no_grad():
                mask = torch.zeros_like(probs).scatter(1, topk_idx, 1.0)
                self.last_expert_fraction = (
                    mask.sum(0) / (N * self.top_k + 1e-9)).detach()

        _ref = self.lowrank.fcs if self.lowrank is not None else self.experts[0].fcs
        out_dim = (_ref[-2].out_features if return_embed else _ref[-1].out_features)
        y = torch.zeros(N, out_dim, device=flat_x.device, dtype=flat_x.dtype)

        # Sparse dispatch: each expert processes only its assigned mutations.
        for e in range(self.num_experts):
            sel = torch.nonzero(gates[:, e], as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                continue
            xe = flat_x.index_select(0, sel)
            if self.lowrank is not None:
                ye = self.lowrank.forward_expert(xe, e, return_embed=return_embed)
            else:
                expert = self.experts[e]
                ye = expert.embed(xe) if return_embed else expert(xe)
            y.index_add_(0, sel, ye * gates[sel, e].unsqueeze(-1))

        for sh in self.shared_experts:
            y = y + (sh.embed(flat_x) if return_embed else sh(flat_x))

        y = y.reshape(B, K, out_dim)
        return y.squeeze(0) if squeeze_back else y
