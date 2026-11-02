"""
TransformerLayerWithStructuralAdapter
======================================
A standard ESM2 transformer layer plus a structural-adapter sub-module that
injects ProteinMPNN-derived features via cross-attention.

Adapter design (from SPURS / ByProt):
    out = layer_main_output
          + StructuralAdapterAttn(query=out, key=encoder_feats, value=encoder_feats)
          + StructuralAdapterFFN(...)

The adapter uses a bottleneck FFN (hidden = embed_dim // 2) which is what SPURS
calls out as "important".
"""

from __future__ import annotations

import torch
import torch.nn as nn

import esm
from esm.modules import (
    ESM1bLayerNorm,
    ESM1LayerNorm,
    FeedForwardNetwork,
    NormalizedResidualBlock,
    gelu,
)
from esm.multihead_attention import MultiheadAttention


class TransformerLayerWithStructuralAdapter(nn.Module):
    """
    Replaces a standard `esm.modules.TransformerLayer` at adapter positions.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        encoder_embed_dim: int,
        add_bias_kv: bool = True,
        use_esm1b_layer_norm: bool = False,
        use_rotary_embeddings: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim             = embed_dim
        self.ffn_embed_dim         = ffn_embed_dim
        self.attention_heads       = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.encoder_embed_dim     = encoder_embed_dim
        self.dropout               = dropout
        self.use_adapter           = True

        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim,     self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = BertLayerNorm(self.embed_dim)

        # Structural adapter: cross-attn over the encoder features + bottleneck FFN
        self.structural_adapter_attn = NormalizedResidualBlock(
            layer=MultiheadAttention(
                self.embed_dim,
                self.attention_heads,
                kdim=self.encoder_embed_dim,
                vdim=self.encoder_embed_dim,
                add_bias_kv=add_bias_kv,
                add_zero_attn=False,
                use_rotary_embeddings=True,
            ),
            embedding_dim=self.embed_dim,
            dropout=self.dropout,
        )
        self.structural_adapter_ffn = NormalizedResidualBlock(
            layer=FeedForwardNetwork(
                self.embed_dim,
                self.embed_dim // 2,        # bottleneck — keep as is
                activation_dropout=self.dropout,
            ),
            embedding_dim=self.embed_dim,
            dropout=self.dropout,
        )

    def forward(
        self,
        x,
        encoder_out,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_head_weights: bool = False,
    ):
        residual = x
        x        = self.self_attn_layer_norm(x)
        x, attn  = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x        = self.final_layer_norm(x)
        x        = gelu(self.fc1(x))
        x        = self.fc2(x)
        x        = residual + x

        if self.use_adapter:
            x = x + self._forward_adapter(
                x, encoder_out,
                attn_mask=self_attn_mask,
                attn_padding_mask=self_attn_padding_mask,
            )
        else:
            assert encoder_out is None
        return x, attn

    def _forward_adapter(self, x, encoder_out, attn_mask, attn_padding_mask):
        encoder_feats = encoder_out["feats"].transpose(0, 1)

        x = self.structural_adapter_attn(
            x,
            key=encoder_feats,
            value=encoder_feats,
            key_padding_mask=attn_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]
        x = self.structural_adapter_ffn(x)
        return x
