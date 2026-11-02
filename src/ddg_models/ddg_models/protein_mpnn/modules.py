"""
ProteinMPNN building blocks.

Faithful port of ProteinMPNN's modules from:
    https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py
also used in:
    https://github.com/Kuhlman-Lab/ThermoMPNN

Contents:
    - gather_edges / gather_nodes / gather_nodes_t / cat_neighbors_nodes
    - PositionWiseFeedForward
    - PositionalEncodings
    - EncLayer / DecLayer
    - ProteinFeatures (backbone-atom RBF + positional + chain encoding)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Gather / cat utilities
# ──────────────────────────────────────────────────────────────────────────────

def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """[B, N, N, C] features at [B, N, K] neighbour indices → [B, N, K, C]."""
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """[B, N, C] at [B, N, K] → [B, N, K, C]."""
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    out = torch.gather(nodes, 1, neighbors_flat)
    return out.view(list(neighbor_idx.shape)[:3] + [-1])


def gather_nodes_t(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """[B, N, C] at [B, K] → [B, K, C]."""
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """Concatenate node features with neighbour edge features."""
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int):
        super().__init__()
        self.W_in  = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act   = nn.GELU()

    def forward(self, h_V):
        return self.W_out(self.act(self.W_in(h_V)))


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        super().__init__()
        self.num_embeddings       = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d = (
            torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
            * mask
            + (1 - mask) * (2 * self.max_relative_feature + 1)
        )
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        return self.linear(d_onehot.float())


# ──────────────────────────────────────────────────────────────────────────────
# Encoder / decoder layers
# ──────────────────────────────────────────────────────────────────────────────

class EncLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1,
                 num_heads=None, scale: float = 30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in     = num_in
        self.scale      = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(num_hidden)
        self.norm2    = nn.LayerNorm(num_hidden)
        self.norm3    = nn.LayerNorm(num_hidden)

        self.W1  = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2  = nn.Linear(num_hidden,          num_hidden, bias=True)
        self.W3  = nn.Linear(num_hidden,          num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden,          num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden,          num_hidden, bias=True)

        self.act   = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV       = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV       = torch.cat([h_V_expand, h_EV], -1)
        h_message  = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh   = torch.sum(h_message, -2) / self.scale
        h_V  = self.norm1(h_V + self.dropout1(dh))

        dh  = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        # Update edge features
        h_EV       = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV       = torch.cat([h_V_expand, h_EV], -1)
        h_message  = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E        = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1,
                 num_heads=None, scale: float = 30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in     = num_in
        self.scale      = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(num_hidden)
        self.norm2    = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden,          num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden,          num_hidden, bias=True)

        self.act   = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV       = torch.cat([h_V_expand, h_E], -1)
        h_message  = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh  = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh  = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V
        return h_V


# ──────────────────────────────────────────────────────────────────────────────
# Featuriser — RBF distances between all backbone atom pairs + positional embed
# ──────────────────────────────────────────────────────────────────────────────

class ProteinFeatures(nn.Module):
    """
    Builds per-edge features:
        - 25 pairwise RBF expansions between backbone atoms (N, CA, C, O, Cb)
        - Positional encoding of sequence offset (modulated by chain identity)

    Concatenated and projected to `edge_features` dim with a linear + LayerNorm.
    """

    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
        num_chain_embeddings: int = 16,
    ):
        super().__init__()
        self.edge_features              = edge_features
        self.node_features              = node_features
        self.top_k                      = top_k
        self.augment_eps                = augment_eps
        self.num_rbf                    = num_rbf
        self.num_positional_embeddings  = num_positional_embeddings

        self.embeddings    = PositionalEncodings(num_positional_embeddings)
        edge_in            = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges     = nn.LayerNorm(edge_features)

    def _dist(self, X: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
        mask_2D  = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX       = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D        = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu     = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu     = D_mu.view([1, 1, 1, -1])
        D_sigma  = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))

    def _get_rbf(self, A: torch.Tensor, B: torch.Tensor, E_idx: torch.Tensor):
        D_A_B           = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        return self._rbf(D_A_B_neighbors)

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Compute Cβ via canonical reconstruction
        b  = X[:, :, 1, :] - X[:, :, 0, :]
        c  = X[:, :, 2, :] - X[:, :, 1, :]
        a  = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N  = X[:, :, 0, :]
        C  = X[:, :, 2, :]
        O  = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        # 25 RBF expansions over all pairwise backbone-atom distances
        RBF_all = [
            self._rbf(D_neighbors),               # Ca-Ca
            self._get_rbf(N,  N,  E_idx),
            self._get_rbf(C,  C,  E_idx),
            self._get_rbf(O,  O,  E_idx),
            self._get_rbf(Cb, Cb, E_idx),
            self._get_rbf(Ca, N,  E_idx),
            self._get_rbf(Ca, C,  E_idx),
            self._get_rbf(Ca, O,  E_idx),
            self._get_rbf(Ca, Cb, E_idx),
            self._get_rbf(N,  C,  E_idx),
            self._get_rbf(N,  O,  E_idx),
            self._get_rbf(N,  Cb, E_idx),
            self._get_rbf(Cb, C,  E_idx),
            self._get_rbf(Cb, O,  E_idx),
            self._get_rbf(O,  C,  E_idx),
            self._get_rbf(N,  Ca, E_idx),
            self._get_rbf(C,  Ca, E_idx),
            self._get_rbf(O,  Ca, E_idx),
            self._get_rbf(Cb, Ca, E_idx),
            self._get_rbf(C,  N,  E_idx),
            self._get_rbf(O,  N,  E_idx),
            self._get_rbf(Cb, N,  E_idx),
            self._get_rbf(C,  Cb, E_idx),
            self._get_rbf(O,  Cb, E_idx),
            self._get_rbf(C,  O,  E_idx),
        ]
        RBF_all = torch.cat(RBF_all, dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains    = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains    = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)

        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx
