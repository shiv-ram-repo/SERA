"""
ESM2WithStructuralAdapter
==========================
ESM2 wrapper that replaces selected transformer layers with adapter-augmented
variants. Pretrained ESM2 weights are loaded via `esm.pretrained.<name>` and
all non-adapter parameters are frozen.

The structural adapter receives `encoder_out['feats']` from outside — these
are the ProteinMPNN-derived per-residue structural features in
`(B, T, encoder_embed_dim)` layout.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Union

import torch
import torch.nn as nn

import esm
from esm.modules import (
    ContactPredictionHead,
    ESM1bLayerNorm,
    RobertaLMHead,
    TransformerLayer,
)

try:
    from omegaconf import OmegaConf
    _HAS_OMEGACONF = True
except ImportError:
    OmegaConf = None
    _HAS_OMEGACONF = False

from .adapter_layer import TransformerLayerWithStructuralAdapter


def _compose_cfg(**kwds):
    if _HAS_OMEGACONF:
        return OmegaConf.create(kwds)
    return dict(kwds)


def _merge_cfg(default_cfg, override_cfg):
    if _HAS_OMEGACONF:
        return OmegaConf.merge(default_cfg, override_cfg)
    out = dict(default_cfg)
    out.update(override_cfg)
    return out


def _cfg_get(cfg, key, default=None):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


class ESM2WithStructuralAdapter(nn.Module):
    """
    Construct via the classmethod:

        model = ESM2WithStructuralAdapter.from_pretrained(args=cfg, name='esm2_t33_650M_UR50D')

    where `cfg` is the SPURS-style config holding at minimum:
        cfg.adapter_layer_indices  (list of ints; negatives allowed, e.g. [-1])
        cfg.encoder.d_model        (encoder feature dim, matches ProteinMPNN combined output)
        cfg.dropout                (adapter dropout)
    """

    @classmethod
    def from_pretrained(
        cls,
        args,
        override_args: Any = None,
        name: str = "esm2_t33_650M_UR50D",
    ):
        pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)

        pretrained_args = _compose_cfg(
            num_layers       = pretrained_model.num_layers,
            embed_dim        = pretrained_model.embed_dim,
            attention_heads  = pretrained_model.attention_heads,
            token_dropout    = pretrained_model.token_dropout,
        )
        args = _merge_cfg(pretrained_args, args)

        # Normalise adapter_layer_indices (-1 → last layer)
        n_layers = _cfg_get(args, "num_layers")
        raw_idx  = _cfg_get(args, "adapter_layer_indices")
        if hasattr(raw_idx, "__iter__"):
            raw_idx = list(raw_idx)
        normalised = [(n_layers + i) % n_layers for i in raw_idx]
        if _HAS_OMEGACONF:
            args.adapter_layer_indices = normalised
        else:
            args["adapter_layer_indices"] = normalised

        model  = cls(args, deepcopy(alphabet))
        result = model.load_state_dict(pretrained_model.state_dict(), strict=False)
        # Optional logging (caller can inspect; we don't depend on a logger)
        del pretrained_model

        # Freeze all non-adapter parameters
        for pname, param in model.named_parameters():
            if "adapter" not in pname:
                param.requires_grad = False
        return model

    def __init__(
        self,
        args,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
    ):
        super().__init__()
        self.args             = args
        self.num_layers       = _cfg_get(args, "num_layers")
        self.embed_dim        = _cfg_get(args, "embed_dim")
        self.attention_heads  = _cfg_get(args, "attention_heads")

        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet      = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx   = alphabet.padding_idx
        self.mask_idx      = alphabet.mask_idx
        self.cls_idx       = alphabet.cls_idx
        self.eos_idx       = alphabet.eos_idx
        self.prepend_bos   = alphabet.prepend_bos
        self.append_eos    = alphabet.append_eos
        self.token_dropout = _cfg_get(args, "token_dropout")
        self.use_adapter   = True

        self._init_submodules()

        if not self.use_adapter:
            for param in self.parameters():
                param.requires_grad = False

    def _init_submodules(self):
        self.embed_scale  = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )
        self.embed_tokens.eval()

        self.layers = nn.ModuleList(
            [self._init_layer(i) for i in range(self.num_layers)]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.contact_head.eval()
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)
        self.emb_layer_norm_after.eval()
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )
        self.lm_head.eval()

    def _init_layer(self, layer_idx: int):
        adapter_idx = _cfg_get(self.args, "adapter_layer_indices")
        encoder_cfg = _cfg_get(self.args, "encoder")
        dropout     = _cfg_get(self.args, "dropout")
        encoder_d   = _cfg_get(encoder_cfg, "d_model")

        if layer_idx in adapter_idx:
            return TransformerLayerWithStructuralAdapter(
                self.embed_dim,
                4 * self.embed_dim,
                self.attention_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
                encoder_embed_dim=encoder_d,
                dropout=dropout,
            )
        return TransformerLayer(
            self.embed_dim,
            4 * self.embed_dim,
            self.attention_heads,
            add_bias_kv=False,
            use_esm1b_layer_norm=True,
            use_rotary_embeddings=True,
        )

    def _forward_layers(
        self,
        x,
        encoder_out,
        padding_mask,
        repr_layers,
        hidden_representations,
        need_head_weights: bool,
        attn_weights,
    ):
        adapter_idx = _cfg_get(self.args, "adapter_layer_indices")
        layer_idx   = -1
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in adapter_idx:
                x, attn = layer(
                    x, encoder_out,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=need_head_weights,
                )
            else:
                x, attn = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=need_head_weights,
                )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0))
        return x, hidden_representations, attn_weights, layer_idx

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_out: dict,
        repr_layers: List[int] = (),
        need_head_weights: bool = False,
        return_contacts: bool = False,
    ):
        if return_contacts:
            need_head_weights = True
        assert tokens.ndim == 2

        padding_mask = tokens.eq(self.padding_idx)
        x            = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train    = 0.15 * 0.8
            src_lengths         = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x                   = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers            = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        attn_weights = [] if need_head_weights else None
        x = x.transpose(0, 1)   # (B, T, E) → (T, B, E)

        if not padding_mask.any():
            padding_mask = None

        x, hidden_representations, attn_weights, layer_idx = self._forward_layers(
            x, encoder_out, padding_mask,
            repr_layers=repr_layers,
            hidden_representations=hidden_representations,
            need_head_weights=need_head_weights,
            attn_weights=attn_weights,
        )

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)   # (T, B, E) → (B, T, E)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        hidden_representations[-1] = x

        x = self.lm_head(x)
        result = {"logits": x, "representations": hidden_representations}

        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                am = 1 - padding_mask.type_as(attentions)
                am = am.unsqueeze(1) * am.unsqueeze(2)
                attentions = attentions * am[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                result["contacts"] = self.contact_head(tokens, attentions)
        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]
