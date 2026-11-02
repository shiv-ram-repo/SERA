"""
ProteinMPNN — the structural encoder used in SPURS.

Faithful port of the published ProteinMPNN model. Key behaviour:
    - Encoder: 3 EncLayers operating on backbone-atom-derived edges
    - Decoder: 3 DecLayers with masked attention. SPURS sets
      `use_input_decoding_order=False` and overrides the order mask to
      all-ones, so every position can attend to every other (one-shot
      decoding instead of autoregressive).

Returns:
    (list_of_hidden_layers, h_S, log_probs)
    where list_of_hidden_layers is reversed (last layer first), matching
    SPURS's expectation in `SPURS.forward_encoder`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    cat_neighbors_nodes,
    DecLayer,
    EncLayer,
    gather_nodes,
    ProteinFeatures,
)


class ProteinMPNN(nn.Module):
    """
    Args
    ----
    num_letters : int
        Vocabulary size (21 for canonical AAs + X).
    node_features : int
        Initial node feature dim (unused in the no-CA-only path).
    edge_features : int
        Edge feature dim coming out of `ProteinFeatures`.
    hidden_dim : int
        Hidden dim throughout the model (128 in `v_48_020`).
    num_encoder_layers : int
    num_decoder_layers : int
    vocab : int
        Width of the sequence embedding `W_s` (21 in `v_48_020`).
    k_neighbors : int
        Top-k nearest-Cα neighbours per residue (48 in `v_48_020`).
    augment_eps : float
        Coord noise during training (0.0 by default in SPURS).
    dropout : float
    ca_only : bool
        SPURS uses False (full backbone). CA-only path is not implemented
        here to keep the file focused on the ProteinMPNN code-path SPURS
        actually uses; raise an error if requested.
    """

    def __init__(
        self,
        num_letters: int,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.1,
        ca_only: bool = False,
    ):
        super().__init__()
        if ca_only:
            raise NotImplementedError(
                "ca_only is not supported in this standalone reimplementation "
                "(MultimodalDDG always uses full backbone)."
            )

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim    = hidden_dim

        self.features = ProteinFeatures(
            edge_features=node_features,
            node_features=edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
        )
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab,      hidden_dim)

        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: Optional[torch.Tensor] = None,
        use_input_decoding_order: bool = False,
        decoding_order: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        device = X.device

        # ── Encoder ──
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V      = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E      = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # ── Build decoder inputs ──
        h_S          = self.W_s(S)
        h_ES         = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask    # include missing regions

        if not use_input_decoding_order:
            # MultimodalDDG path: left-to-right with all residues visible
            mask_size = E_idx.shape[1]
            order_mask_backward = torch.ones(
                X.size(0), mask_size, mask_size, device=device
            )
        else:
            mask_size = E_idx.shape[1]
            batch_size = E_idx.shape[0]
            diagonal_matrix = (
                torch.eye(mask_size, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            )
            order_mask_backward = 1.0 - diagonal_matrix

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D     = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw     = mask_1D * mask_attend
        mask_fw     = mask_1D * (1.0 - mask_attend)

        all_hidden       = []
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V   = layer(h_V, h_ESV, mask)
            all_hidden.append(h_V)

        logits    = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return list(reversed(all_hidden)), h_S, log_probs
