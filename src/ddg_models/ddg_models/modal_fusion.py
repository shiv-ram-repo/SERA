"""
modal_fusion
============
Multiplicative fusion of the sequence and structure channels.

The existing pipeline combines ESM-2 features h_seq (1280-d) with ProteinMPNN
features h_str (128-d) in two ways, both additive: a cross-attention residual
(h_seq <- h_seq + attn) and then a concatenation [h_seq ; h_str] handed to the
readout. An MLP over a concatenation can only represent the interaction between
the two channels by learning it implicitly through depth, from 212 training
proteins.

That is the wrong inductive bias for this quantity. The stability cost of a
substitution is not the sum of a sequence term and a structure term; it is
conditional. Leucine to alanine is catastrophic in a packed core and negligible
on the surface. The same sequence-level signal must mean different things in
different structural environments, which is a multiplicative relationship.

Two mechanisms, usable together:

FiLM modulation
    The structural channel predicts a per-dimension scale and shift that
    modulate the sequence channel:

        h_seq' = (1 + gamma(h_str)) * h_seq + beta(h_str)

    The output projection is zero-initialised, so gamma = beta = 0 at the start
    and the module is exactly the identity: training begins from the current
    architecture and any modulation has to be earned. This lets structure
    re-weight individual sequence features rather than merely sit beside them.

Low-rank bilinear interaction
    An explicit product term,

        z = (U h_seq) * (V h_str),  z in R^rank,

    appended to the representation. This is factorised bilinear pooling: the
    full interaction tensor would be 1280 x 128 x d, which is intractable, while
    the rank-r factorisation costs rank*(1280 + 128) parameters and still
    delivers genuine pairwise products to the head instead of making it infer
    them.

Capacity here is deliberately not minimised. The useful question is whether the
head receives interaction terms at all, not how few parameters produce them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalFusion(nn.Module):
    """Fuse sequence and structure channels with explicit interaction terms.

    mode:
        "concat"         current behaviour, [h_seq ; h_str]
        "film"           FiLM-modulated sequence, then concat
        "bilinear"       concat plus a low-rank product term
        "film_bilinear"  both
    """

    def __init__(self, seq_dim: int = 1280, struct_dim: int = 128,
                 mode: str = "film_bilinear", rank: int = 256,
                 hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.mode = mode
        self.seq_dim, self.struct_dim, self.rank = seq_dim, struct_dim, rank
        self.use_film = "film" in mode
        self.use_bilinear = "bilinear" in mode

        if self.use_film:
            self.film = nn.Sequential(
                nn.Linear(struct_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2 * seq_dim),
            )
            # Zero-init the output layer: identity at initialisation.
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

        if self.use_bilinear:
            self.U = nn.Linear(seq_dim, rank, bias=False)
            self.V = nn.Linear(struct_dim, rank, bias=False)
            self.bil_norm = nn.LayerNorm(rank)
            nn.init.normal_(self.U.weight, std=0.02)
            nn.init.normal_(self.V.weight, std=0.02)

    def out_dim(self) -> int:
        d = self.seq_dim + self.struct_dim
        if self.use_bilinear:
            d += self.rank
        return d

    def forward(self, h_seq: torch.Tensor, h_str: torch.Tensor) -> torch.Tensor:
        # h_seq: (B, T, seq_dim)   h_str: (B, T, struct_dim)
        if self.use_film:
            gb = self.film(h_str)
            gamma, beta = gb.chunk(2, dim=-1)
            h_seq = (1.0 + gamma) * h_seq + beta

        parts = [h_seq, h_str]
        if self.use_bilinear:
            z = self.U(h_seq) * self.V(h_str)          # (B, T, rank)
            # Signed square root then normalisation: standard stabilisation for
            # bilinear features, whose magnitudes are heavy-tailed.
            z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)
            parts.append(self.bil_norm(z))

        return torch.cat(parts, dim=-1)
