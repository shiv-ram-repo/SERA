"""
SPURS with Cross-Attention Fusion (Option 2)
============================================
Full SPURS model: ProteinMPNN encoder + ESM2-with-structural-adapter decoder
+ Cross-Attention Fusion layer + MLP head.

Fuses sequence representations and 3D structural representations through an 
explicit cross-attention bottleneck right before the final MLP head, forcing
the network to reconcile sequence features with geometric structural constraints.

ΔΔG prediction (all-mutants-per-pass):
    1. ProteinMPNN encodes wild-type structure → structural features
    2. ESM2 + adapter combines sequence + structure → per-residue representation
    3. Cross-Attention: ESM acts as Query; ProteinMPNN acts as Key and Value
    4. Representation is concatenated with raw encoder feats
    5. MLP head produces (B, K, 21) per-AA energies for each mutated position
    6. ΔΔG = g(mut) - g(wt) via one-hot masking from `append_tensors`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base         import BaseModel
from .mlp          import MLP, MLPConfig
from .protein_mpnn import ProteinMPNN, ProteinMPNNConfig, get_protein_mpnn
from .esm_adapter  import ESM2WithStructuralAdapter


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MultimodalDDGConfig:
    encoder: ProteinMPNNConfig    = field(default_factory=ProteinMPNNConfig)
    adapter_layer_indices: List   = field(default_factory=lambda: [-1])
    separate_loss: bool           = True
    name: str                     = "esm2_t33_650M_UR50D"
    dropout: float                = 0.1
    mlp: MLPConfig                = field(default_factory=MLPConfig)

    # Optional: path to ProteinMPNN v_48_020 checkpoint
    proteinmpnn_ckpt: str         = ""


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalDDG(BaseModel):
    """Reproduction of MultimodalDDG augmented with a late Cross-Attention Bottleneck."""

    _default_cfg = MultimodalDDGConfig()

    def __init__(self, cfg: MultimodalDDGConfig) -> None:
        super().__init__(cfg)

        # ── Pull values from cfg (handle dict or omegaconf) ────────────
        encoder_cfg = self.cfg["encoder"]            if isinstance(self.cfg, dict) else self.cfg.encoder
        mlp_cfg     = self.cfg["mlp"]                if isinstance(self.cfg, dict) else self.cfg.mlp
        name        = self.cfg["name"]               if isinstance(self.cfg, dict) else self.cfg.name
        dropout     = self.cfg["dropout"]            if isinstance(self.cfg, dict) else self.cfg.dropout
        adapter_idx = (
            self.cfg["adapter_layer_indices"] if isinstance(self.cfg, dict)
            else self.cfg.adapter_layer_indices
        )
        ckpt_path   = (
            self.cfg["proteinmpnn_ckpt"] if isinstance(self.cfg, dict)
            else getattr(self.cfg, "proteinmpnn_ckpt", "")
        )

        def _g(c, k):
            return c[k] if isinstance(c, dict) else getattr(c, k)

        def _has(c, k):
            return (k in c) if isinstance(c, dict) else hasattr(c, k)

        self.tune                     = _g(encoder_cfg, "tune")
        self.use_input_decoding_order = _g(encoder_cfg, "use_input_decoding_order")

        # ── 1. Encoder: ProteinMPNN ────────────────────────────────────
        self.encoder = get_protein_mpnn(
            tune=self.tune,
            ckpt_path=ckpt_path if ckpt_path else None,
        )

        # Patch encoder cfg's d_model to match MLP input expectation
        if isinstance(encoder_cfg, dict):
            encoder_cfg["d_model"] = _g(mlp_cfg, "input_dim")
        else:
            encoder_cfg.d_model = _g(mlp_cfg, "input_dim")

        # ── 2. Decoder: ESM2 + structural adapter ──────────────────────
        from dataclasses import dataclass as _dc, field as _f

        @_dc
        class _AdapterArgs:
            adapter_layer_indices: list = _f(default_factory=list)
            dropout:               float = 0.1
            encoder:               object = None

        adapter_args = _AdapterArgs(
            adapter_layer_indices=list(adapter_idx),
            dropout=dropout,
            encoder=encoder_cfg,
        )
        self.decoder = ESM2WithStructuralAdapter.from_pretrained(
            args=adapter_args, name=name,
        )

        # ── 3. MLP head ────────────────────────────────────────────────
        flat_dim = _g(mlp_cfg, "flat_dim")
        if flat_dim < 0:
            new_input_dim = _g(mlp_cfg, "input_dim") + 1280
        else:
            new_input_dim = _g(mlp_cfg, "input_dim") + flat_dim

        # ---- sequence/structure fusion ----------------------------------
        self.fusion_mode = str(_g(mlp_cfg, "fusion_mode")) if _has(mlp_cfg, "fusion_mode") else "concat"
        self.modal_fusion = None
        if self.fusion_mode != "concat":
            from .modal_fusion import ModalFusion
            _rank = int(_g(mlp_cfg, "fusion_rank")) if _has(mlp_cfg, "fusion_rank") else 256
            _hid = int(_g(mlp_cfg, "fusion_hidden")) if _has(mlp_cfg, "fusion_hidden") else 512
            _sdim = int(_g(mlp_cfg, "input_dim"))          # structural channel width
            _edim = int(_g(mlp_cfg, "esm_dim")) if _has(mlp_cfg, "esm_dim") else 1280
            self.modal_fusion = ModalFusion(seq_dim=_edim, struct_dim=_sdim,
                                            mode=self.fusion_mode, rank=_rank,
                                            hidden=_hid)
            new_input_dim = self.modal_fusion.out_dim()

        # Permanent-feature fusion widens the readout input by 210 when enabled.
        self.use_permanent = bool(_g(mlp_cfg, "use_permanent")) \
            if _has(mlp_cfg, "use_permanent") else False
        self.perm_dim = 210 if self.use_permanent else 0
        new_input_dim = new_input_dim + self.perm_dim

        if isinstance(mlp_cfg, dict):
            mlp_cfg["input_dim"] = new_input_dim
        else:
            mlp_cfg.input_dim = new_input_dim

        self.input_dim = _g(encoder_cfg, "d_model")   # = original mlp.input_dim

        # Energy-decomposition readout. ProteinMPNN and ESM each emit a
        # per-position distribution over amino acids; both are discarded by
        # the standard readout. Zero-initialised, so the model starts as the
        # plain difference of scores.
        self.use_edr = bool(_g(mlp_cfg, "use_energy_decomposition")) \
            if _has(mlp_cfg, "use_energy_decomposition") else False
        self.edr = None
        if self.use_edr:
            from .energy_decomposition import EnergyDecompositionReadout
            def _flag(name, default=True):
                return bool(_g(mlp_cfg, name)) if _has(mlp_cfg, name) else default
            self.edr = EnergyDecompositionReadout(
                new_input_dim,
                use_struct=_flag("edr_use_struct"),
                use_evo=_flag("edr_use_evo"),
                use_fluct=_flag("edr_use_fluct", False))

        # Pair-coupled readout: generalises the rank-1 antisymmetric
        # bilinear form to rank r. Zero-initialised, so the model starts
        # identical to the difference-of-scores readout.
        self.use_pair_readout = bool(_g(mlp_cfg, "use_pair_readout")) \
            if _has(mlp_cfg, "use_pair_readout") else False
        self.pair_coupling = None
        if self.use_pair_readout:
            from .pair_readout import PairCoupling
            _r = int(_g(mlp_cfg, "pair_rank")) if _has(mlp_cfg, "pair_rank") else 4
            _ph = int(_g(mlp_cfg, "pair_hidden")) if _has(mlp_cfg, "pair_hidden") else 0
            self.pair_coupling = PairCoupling(new_input_dim, rank=_r, hidden=_ph)
        # ---- multi-layer ESM fusion -------------------------------------
        self.use_layer_fusion = bool(_g(mlp_cfg, "use_layer_fusion")) \
            if _has(mlp_cfg, "use_layer_fusion") else False
        self.layer_fusion = None
        self.layer_indices = []
        if self.use_layer_fusion:
            from .layer_fusion import build_layer_fusion, default_layer_indices
            idx = _g(mlp_cfg, "layer_indices") if _has(mlp_cfg, "layer_indices") else None
            self.layer_indices = list(idx) if idx else default_layer_indices()
            mode = _g(mlp_cfg, "layer_fusion_mode") if _has(mlp_cfg, "layer_fusion_mode") else "scalar_mix"
            topk = int(_g(mlp_cfg, "layer_fusion_topk")) if _has(mlp_cfg, "layer_fusion_topk") else 2
            esm_dim = int(_g(mlp_cfg, "esm_dim")) if _has(mlp_cfg, "esm_dim") else 1280
            init = _g(mlp_cfg, "layer_mix_init") if _has(mlp_cfg, "layer_mix_init") else "final"
            self.layer_fusion = build_layer_fusion(
                mode, len(self.layer_indices), esm_dim, top_k=topk, init=init)

        _use_moe = (mlp_cfg.get("use_moe", False) if isinstance(mlp_cfg, dict)
                    else getattr(mlp_cfg, "use_moe", False))
        if _use_moe:
            from .moe_head import MoEReadout
            self.mlp   = MoEReadout(mlp_cfg)
        else:
            self.mlp   = MLP(mlp_cfg)

        # Optional flat projection of ESM features
        if flat_dim > 0:
            self.flat_layers = nn.Linear(1280, flat_dim)
            self.dp          = nn.Dropout(dropout)
        else:
            self.flat_layers = None
            self.dp          = None

        # =====================================================================
        # OPTION 2: CROSS-ATTENTION BOTTLENECK LAYER
        # =====================================================================
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=1280,  # ESM embedding dimension (Query)
            kdim=128,        # ProteinMPNN feature dimension (Key)
            vdim=128,        # ProteinMPNN feature dimension (Value)
            num_heads=8,
            batch_first=True
        )
        # =====================================================================

        # Expose tokeniser metadata
        self.padding_idx = self.decoder.padding_idx
        self.mask_idx    = self.decoder.mask_idx
        self.cls_idx     = self.decoder.cls_idx
        self.eos_idx     = self.decoder.eos_idx

    def forward_encoder(self, batch: dict) -> torch.Tensor:
        X                  = batch["X"]
        S                  = batch["S"]
        mask               = batch["mask"]
        chain_M            = batch["chain_M"]
        residue_idx        = batch["residue_idx"]
        chain_encoding_all = batch["chain_encoding_all"]

        all_mpnn_hid, mpnn_embed, _mpnn_lp = self.encoder(
            X, S, mask, chain_M, residue_idx, chain_encoding_all,
            None, self.use_input_decoding_order,
        )
        # Bound to the module: the stash site is in another scope.
        self._mpnn_log_probs_cache = _mpnn_lp
        return torch.cat(
            [all_mpnn_hid[0], mpnn_embed, all_mpnn_hid[1], all_mpnn_hid[2]],
            dim=-1,
        )

    def forward(self, batch: dict, **kwargs):
        # 1. Run encoder (optionally with frozen weights)
        if not self.tune:
            with torch.no_grad():
                batch["feats"] = self.forward_encoder(batch)
        else:
            batch["feats"] = self.forward_encoder(batch)

        # 2. Truncate encoder features to the size the adapter expects
        batch["feats"] = batch["feats"][:, :, : self.input_dim]
        encoder_out    = {"feats": F.pad(batch["feats"], (0, 0, 1, 1))}

        # 3. Run ESM2 + adapter
        _repr_layers = tuple(self.layer_indices) if self.use_layer_fusion else ()
        decoder_out    = self.decoder(
            tokens=batch["tokens"],
            encoder_out=encoder_out,
            repr_layers=_repr_layers,
        )
        if self.use_layer_fusion and self.layer_fusion is not None:
            reps = decoder_out["representations"]
            # Fall back to the final layer for any index the encoder did not
            # return, so a bad layer list degrades instead of crashing.
            stack = torch.stack(
                [reps.get(i, reps[-1]) for i in self.layer_indices], dim=2
            )                                                  # (B, T, nL, E)
            representation = self.layer_fusion(stack)          # (B, T, E)
        else:
            representation = decoder_out["representations"][-1]   # (B, T, embed_dim)
        # Energy landscapes the readout would otherwise never see.
        if getattr(self, "use_edr", False):
            _lp = getattr(self, "_mpnn_log_probs_cache", None)
            if _lp is not None:
                batch["mpnn_log_probs"] = _lp
            if "logits" in decoder_out:
                batch["esm_logits"] = decoder_out["logits"]

        # 4. Optional flat projection of ESM features
        if self.flat_layers is not None:
            representation = self.flat_layers(representation)
            representation = self.dp(representation)
            representation = F.gelu(representation)

        # =====================================================================
        # OPTION 2: CROSS-ATTENTION FUSION
        # Intercept and route tensors before naive late-stage concatenation
        # =====================================================================
        mpnn_feats = encoder_out["feats"]
        attn_out, _ = self.cross_attn(
            query=representation, 
            key=mpnn_feats, 
            value=mpnn_feats
        )
        # Residual connection combining linguistic and forced structural context
        representation = representation + attn_out
        # =====================================================================

        # 5. Fuse the sequence and structure channels. Concatenation is additive:
        #    it places the channels side by side and leaves any interaction to
        #    the readout. ModalFusion supplies multiplicative terms instead.
        if self.modal_fusion is not None:
            representation = self.modal_fusion(representation, mpnn_feats)
        else:
            representation = torch.cat([representation, mpnn_feats], dim=-1)

        # ── Mode A: return full (L, 20) ΔΔG table for all mutations ──
        if kwargs.get("return_logist", False):
            wt              = batch["seq"]
            L               = len(wt)
            shifted_mut_ids = torch.repeat_interleave(torch.arange(1, 1 + L), 20)
            muted_id_repr   = representation[:, shifted_mut_ids.long()]
            batch["muted_id_representation"] = muted_id_repr

            pre_output = self.mlp(batch)         # (B, L*20, 21)
            ddg_out    = pre_output.squeeze(0)   # (L*20, 21)

            mt_aa = torch.arange(20).repeat(L)
            ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
            wt_aa = torch.repeat_interleave(
                torch.tensor([ALPHABET.index(s) for s in wt]), 20,
            )
            num_classes = 21
            mt_oh = F.one_hot(mt_aa, num_classes=num_classes).to(representation.device)
            wt_oh = F.one_hot(wt_aa, num_classes=num_classes).to(representation.device)

            ddg_mut = (ddg_out * mt_oh).sum(-1)
            ddg_wt  = (ddg_out * wt_oh).sum(-1)
            return (ddg_mut - ddg_wt).reshape(-1, 20)

        # ── Mode B: scalar ΔΔG per supplied mutation ──
        mut_ids = (
            batch["mut_ids"] if isinstance(batch["mut_ids"], torch.Tensor)
            else torch.tensor(batch["mut_ids"])
        )
        shifted_ids = mut_ids.to(representation.device) + 1
        muted_id_repr = representation[:, shifted_ids.long()]

        # Permanent-feature fusion: broadcast the per-protein vector across the
        # K mutations and concatenate onto the per-mutation representation.
        if batch.get("perm_features", None) is not None:
            pf = batch["perm_features"]
            if pf.dim() == 2:
                pf = pf.unsqueeze(0)
            muted_id_repr = torch.cat(
                [muted_id_repr, pf.to(muted_id_repr.device)], dim=-1)

        batch["muted_id_representation"] = muted_id_repr

        pre_output = self.mlp(batch)             # (B, K, 21)
        ddg_out    = pre_output.squeeze()        # (K, 21) when B=1

        wt_onehot  = batch["append_tensors"][:, :21]
        mut_onehot = batch["append_tensors"][:, 21:]
        ddg_mut    = (ddg_out * mut_onehot).sum(-1)
        ddg_wt     = (ddg_out * wt_onehot).sum(-1)
        ddg        = ddg_mut - ddg_wt

        # Rank-r pair coupling. The baseline above is the k = 0 column
        # (U[:,0] = score, V[:,0] = 1); these are the remaining columns.
        if self.pair_coupling is not None:
            from .pair_readout import pair_term
            _h = batch["muted_id_representation"]
            _h = _h.squeeze(0) if _h.dim() == 3 else _h
            _U, _V = self.pair_coupling(_h)
            _extra = pair_term(_U, _V, wt_onehot, mut_onehot)
            if _extra is not None:
                ddg = ddg + _extra

        if getattr(self, "edr", None) is not None:
            _hh = batch["muted_id_representation"]
            _hh = _hh.squeeze(0) if _hh.dim() == 3 else _hh
            ddg = self.edr(_hh, ddg, wt_onehot, mut_onehot,
                           batch, mut_ids)

        ddg[torch.isnan(ddg)] = 10000.0
        return ddg
