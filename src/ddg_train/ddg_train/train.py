"""
train.py
========
End-to-end training entry point with composable loss options:

  loss_type            "huber" | "bmc" | "lds_huber"     ΔΔG regression
  use_siamese          bool                              + |f_fwd + f_rev|^2
  use_k50_head         bool                              K50 multi-task head
  use_move_b_features  bool                              Move-B input features
  use_bcas             bool                              + BCAS (bias + variance)
  use_ood_margin       bool                              + input-noise consistency

BCAS and siamese are mutually exclusive (both target the forward-reverse
sum; BCAS subsumes siamese with the addition of an explicit bias term).
If both flags are set, BCAS takes precedence and a warning is logged.

K50 and Move-B heads are head replacements; BCAS / OOD-margin are loss-only
additions and compose with either head choice.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddg_data import Alphabet, Featurizer, MegaScaleDataset
from ddg_models import MultimodalDDG, MultimodalDDGConfig, L1Loss

from .evaluate           import evaluate
from .metrics            import compute_metrics
from .imbalanced_losses  import BMCLoss, LDSWeightedHuber, compute_lds_weights
from .k50_head           import K50Head, K50HeadConfig, K50MultiTaskLoss
from .k50_batch_augment  import K50BatchAugmentor
from .move_b_features    import (
    MoveBHead, MoveBHeadConfig, MoveBFeatureAugmentor,
)
from .custom_losses      import BCASLoss, OODMarginLoss
from run_manifest import Manifest
log = logging.getLogger(__name__)


DEFAULT_MEGASCALE_CSV = (
    "/backup/new_work/data_root/data/dataset/megascale/"
    "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
)
DEFAULT_MOVE_B_CACHE = "/backup/new_work/runs/move_b_cache/delta_cache.pt"


@dataclass
class TrainConfig:
    data_root:        str = "/data/spurs_root"
    megascale_csv:    str = DEFAULT_MEGASCALE_CSV
    output_dir:       str = "runs/spurs_repro_v1"
    ckpt_filename:    str = "best.pt"
    proteinmpnn_ckpt: str = ""

    max_epochs:       int   = 200
    early_stop_pat:   int   = 30
    lr:               float = 1e-4
    weight_decay:     float = 1e-2
    val_every:        int   = 1
    grad_accum:       int   = 1
    seed:             int   = 42

    esm_name:         str = "esm2_t33_650M_UR50D"
    adapter_layer:    int = -3
    tune_mpnn:        bool = True

    loss_type:        str   = "huber"
    bmc_init_sigma:   float = 1.0
    lds_bin_size:     float = 0.1
    lds_sigma:        float = 2.0

    use_siamese:      bool  = False
    siamese_weight:   float = 0.5

    use_k50_head:     bool  = False
    k50_loss_weight:  float = 1.0

    use_move_b_features: bool  = False
    move_b_cache_path:   str   = DEFAULT_MOVE_B_CACHE
    move_b_local_radius: float = 8.0

    # NEW: BCAS loss (bias-corrected anti-symmetric)
    use_bcas:         bool  = False
    bcas_alpha:       float = 1.0   # weight on the squared batch-mean (bias)
    bcas_beta:        float = 0.5   # weight on the per-mutation variance term

    # NEW: OOD-margin loss (input-noise consistency)
    use_ood_margin:   bool  = False
    ood_margin_sigma:    float = 0.1
    ood_margin_weight:   float = 0.5
    ood_margin_samples:  int   = 1
    use_energy_decomposition: bool = False
    edr_use_struct:     bool  = True
    edr_use_evo:        bool  = True
    edr_use_fluct:      bool  = False
    use_pair_readout:   bool  = False
    pair_rank:          int   = 4
    pair_hidden:        int   = 0
    # --- Mixture-of-Experts readout -------------------------------
    use_moe:            bool  = False
    moe_num_experts:    int   = 8
    moe_top_k:          int   = 2
    moe_aux_weight:     float = 0.01    # weight on (load-balance + z) aux loss
    moe_shared_expert:  bool  = True
    moe_router_struct:  bool  = True
    moe_noise_eps:      float = 1.0
    moe_route_mode:     str   = "learned"
    moe_balance_mode:   str   = "aux"
    moe_bias_gamma:     float = 1e-3
    moe_shared_count:   int   = 1
    moe_chem_prior_scale: float = 2.0
    moe_expert_mode:    str   = "full"
    moe_lora_rank:      int   = 8
    moe_lora_scale:     float = 1.0
    use_layer_fusion:   bool  = False
    layer_fusion_mode:  str   = "scalar_mix"
    layer_fusion_topk:  int   = 2
    layer_mix_init:     str   = "final"
    fusion_mode:        str   = "concat"
    fusion_rank:        int   = 256
    fusion_hidden:      int   = 512
    # Readout width. Capacity is a free design choice here, not a constraint.
    mlp_hidden:         List[int] = field(default_factory=lambda: [512, 512, 512])
    # Structural adapters at several depths. Empty keeps the single
    # adapter at `adapter_layer`.
    adapter_layer_indices: List[int] = field(default_factory=list)
    layer_indices:      List[int] = field(default_factory=list)
    moe_expert_hidden:  List[int] = field(
        default_factory=lambda: [512, 512, 512]
    )   # narrow the experts to parameter-match the dense head
    # NOVEL: optional custom margin module (variance-scaled / learnable /
    # shift-aligned / feature-mixup). If set, it replaces the default
    # isotropic OODMarginLoss. Must expose forward(model, batch, ddg_clean, valid).
    ood_margin_module:   object = None
    # NOVEL: Sharpness-Aware Minimization (flat-minima optimization)
    use_sam:          bool  = False
    sam_rho:          float = 0.05


def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def build_dataloaders(cfg: TrainConfig):
    alphabet   = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    train_ds = MegaScaleDataset(data_root=cfg.data_root, split="train")
    val_ds   = MegaScaleDataset(data_root=cfg.data_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=featurizer)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=featurizer)
    return train_loader, val_loader


def build_model(cfg: TrainConfig) -> MultimodalDDG:
    spurs_cfg = MultimodalDDGConfig()
    spurs_cfg.adapter_layer_indices = (list(cfg.adapter_layer_indices)
                                       if getattr(cfg, "adapter_layer_indices", None)
                                       else [cfg.adapter_layer])
    spurs_cfg.name                  = cfg.esm_name
    spurs_cfg.proteinmpnn_ckpt      = cfg.proteinmpnn_ckpt
    spurs_cfg.encoder.tune          = cfg.tune_mpnn
    spurs_cfg.mlp.input_dim         = 128
    spurs_cfg.mlp.hidden_dim        = list(getattr(cfg, "mlp_hidden", None) or [512, 512, 512])
    spurs_cfg.mlp.output_dim        = 21
    spurs_cfg.mlp.dropout           = 0.1
    if getattr(cfg, "use_energy_decomposition", False):
        spurs_cfg.mlp.use_energy_decomposition = True
        spurs_cfg.mlp.edr_use_struct = cfg.edr_use_struct
        spurs_cfg.mlp.edr_use_evo    = cfg.edr_use_evo
        spurs_cfg.mlp.edr_use_fluct  = cfg.edr_use_fluct
    if getattr(cfg, "use_pair_readout", False):
        spurs_cfg.mlp.use_pair_readout = True
        spurs_cfg.mlp.pair_rank        = cfg.pair_rank
        spurs_cfg.mlp.pair_hidden      = cfg.pair_hidden
    spurs_cfg.mlp.fusion_mode       = getattr(cfg, "fusion_mode", "concat")
    spurs_cfg.mlp.fusion_rank       = getattr(cfg, "fusion_rank", 256)
    spurs_cfg.mlp.fusion_hidden     = getattr(cfg, "fusion_hidden", 512)
    if getattr(cfg, "use_layer_fusion", False):
        spurs_cfg.mlp.use_layer_fusion  = True
        spurs_cfg.mlp.layer_fusion_mode = cfg.layer_fusion_mode
        spurs_cfg.mlp.layer_fusion_topk = cfg.layer_fusion_topk
        spurs_cfg.mlp.layer_indices     = list(cfg.layer_indices)
        spurs_cfg.mlp.layer_mix_init    = cfg.layer_mix_init
    if getattr(cfg, "use_moe", False):
        spurs_cfg.mlp.use_moe            = True
        spurs_cfg.mlp.moe_num_experts    = cfg.moe_num_experts
        spurs_cfg.mlp.moe_top_k          = cfg.moe_top_k
        spurs_cfg.mlp.moe_shared_expert  = cfg.moe_shared_expert
        spurs_cfg.mlp.moe_router_struct  = cfg.moe_router_struct
        spurs_cfg.mlp.moe_noise_eps      = cfg.moe_noise_eps
        spurs_cfg.mlp.moe_expert_hidden  = list(cfg.moe_expert_hidden)
        spurs_cfg.mlp.moe_route_mode     = cfg.moe_route_mode
        spurs_cfg.mlp.moe_balance_mode   = cfg.moe_balance_mode
        spurs_cfg.mlp.moe_bias_gamma     = cfg.moe_bias_gamma
        spurs_cfg.mlp.moe_shared_count   = cfg.moe_shared_count
        spurs_cfg.mlp.moe_chem_prior_scale = cfg.moe_chem_prior_scale
        spurs_cfg.mlp.moe_expert_mode    = cfg.moe_expert_mode
        spurs_cfg.mlp.moe_lora_rank      = cfg.moe_lora_rank
        spurs_cfg.mlp.moe_lora_scale     = cfg.moe_lora_scale

    model = MultimodalDDG(spurs_cfg)

    if cfg.use_k50_head and cfg.use_move_b_features:
        log.warning("Both use_k50_head and use_move_b_features set; using K50.")

    if cfg.use_k50_head:
        head_input_dim = model.mlp.fcs[0].in_features
        log.info(f"Replacing MultimodalDDG MLP head with K50Head (input_dim={head_input_dim})")
        model.mlp = K50Head(K50HeadConfig(
            input_dim=head_input_dim, hidden_dim=[512, 512, 512],
            n_aa=21, dropout=0.1,
        ))
    elif cfg.use_move_b_features:
        head_input_dim = model.mlp.fcs[0].in_features
        log.info(f"Replacing MultimodalDDG MLP head with MoveBHead "
                 f"(input_dim={head_input_dim} + {MoveBFeatureAugmentor.N_FEAT} δ-features)")
        model.mlp = MoveBHead(MoveBHeadConfig(
            input_dim=head_input_dim,
            delta_feat_dim=MoveBFeatureAugmentor.N_FEAT,
            hidden_dim=[512, 512, 512], n_aa=21, dropout=0.1,
        ))
    return model


def _build_ddg_loss(cfg: TrainConfig, train_loader):
    if cfg.loss_type == "huber":
        base = L1Loss()
        def task_fn(pred, target):
            loss, _ = base(pred, target); return loss
        return task_fn, None, None
    if cfg.loss_type == "bmc":
        bmc = BMCLoss(init_noise_sigma=cfg.bmc_init_sigma)
        def task_fn(pred, target):
            return bmc(pred, target)
        return task_fn, bmc, None
    if cfg.loss_type == "lds_huber":
        log.info("LDS: scanning training labels...")
        all_y = []
        for batch in train_loader:
            y = batch["ddG"].view(-1).cpu().numpy()
            y = y[np.isfinite(y) & (np.abs(y) < 9000)]
            all_y.append(y)
        all_y = np.concatenate(all_y) if all_y else np.array([0.0])
        lds_info = compute_lds_weights(all_y, bin_size=cfg.lds_bin_size,
                                        sigma=cfg.lds_sigma)
        inv_table = 1.0 / lds_info["smoothed_density"]
        occ = lds_info["smoothed_density"] > lds_info["smoothed_density"].min()
        if occ.sum() > 0:
            inv_table = inv_table / inv_table[occ].mean()
        lds_info["_inv_table"] = inv_table.astype("float32")
        lds_module = LDSWeightedHuber(beta=1.0)
        def task_fn(pred, target):
            y_np = target.detach().cpu().numpy()
            bin_idx = np.clip(np.digitize(y_np, lds_info["bin_edges"]) - 1,
                              0, len(lds_info["_inv_table"]) - 1)
            w = torch.from_numpy(lds_info["_inv_table"][bin_idx]).to(target.device)
            return lds_module(pred, target, weights=w)
        return task_fn, None, lds_info
    raise ValueError(f"unknown loss_type: {cfg.loss_type}")


def _make_swapped_batch(batch: dict) -> dict:
    """WT/MUT swap for the anti-symmetry constraint."""
    swapped = dict(batch)
    if "append_tensors" in batch and batch["append_tensors"] is not None:
        at = batch["append_tensors"]
        if at.size(-1) == 42:
            swapped["append_tensors"] = torch.cat([at[..., 21:], at[..., :21]], dim=-1)
    if "ddG" in batch and batch["ddG"] is not None:
        swapped["ddG"] = -batch["ddG"]
    return swapped


def train_one_epoch(
    model, ddg_task_fn, k50_loss_fn, bcas_fn, ood_margin_fn,
    k50_augmentor, move_b_augmentor,
    loader, optimizer, device, epoch,
    *,
    grad_accum=1,
    use_siamese=False, siamese_weight=0.5,
    use_k50_head=False, k50_loss_weight=1.0,
    use_move_b=False,
    use_bcas=False, use_ood_margin=False,
    ood_margin_weight=0.5,
    use_sam=False,
    use_moe=False, moe_aux_weight=0.01,
):
    model.train()
    optimizer.zero_grad()
    running = dict(total=0.0, ddg=0.0, anti=0.0, k50=0.0,
                   bcas=0.0, ood_margin=0.0)
    n = 0

    loop = tqdm(loader, desc=f"Train Ep{epoch:03d}", leave=False)
    for step, batch in enumerate(loop):
        if use_k50_head and k50_augmentor is not None:
            batch = k50_augmentor(batch)
        if use_move_b and move_b_augmentor is not None:
            batch = move_b_augmentor(batch)
        batch = _move_batch(batch, device)

        # Forward pass (populates batch["muted_id_representation"])
        ddg_pred = model(batch).view(-1)
        ddg_true = batch["ddG"].view(-1)
        valid = (ddg_pred.abs() < 9000) & torch.isfinite(ddg_true)
        if valid.sum() == 0:
            continue
        ddg_loss = ddg_task_fn(ddg_pred[valid], ddg_true[valid])

        # K50 multi-task auxiliary
        k50_loss = torch.zeros((), device=device)
        if (use_k50_head and isinstance(model.mlp, K50Head)
                and "k50_t_wt" in batch and k50_loss_weight > 0.0):
            _, h_t, h_c = model.mlp.forward_all(batch)
            wt_oh  = batch["append_tensors"][..., :21].float()
            mut_oh = batch["append_tensors"][..., 21:42].float()
            k50_loss, _ = k50_loss_fn(h_t, h_c, wt_oh, mut_oh,
                target_t_wt=batch["k50_t_wt"], target_t_mut=batch["k50_t_mut"],
                target_c_wt=batch["k50_c_wt"], target_c_mut=batch["k50_c_mut"],
            )

        # Anti-symmetry: siamese OR BCAS (mutually exclusive)
        anti_loss = torch.zeros((), device=device)
        bcas_loss_val = torch.zeros((), device=device)

        if use_bcas:
            # Run reverse pass and apply BCAS
            swapped = _make_swapped_batch(batch)
            ddg_pred_swap = model(swapped).view(-1)
            valid_sw = valid & (ddg_pred_swap.abs() < 9000)
            bcas_loss_val, _ = bcas_fn(ddg_pred, ddg_pred_swap, valid_sw)
        elif use_siamese:
            swapped = _make_swapped_batch(batch)
            ddg_pred_swap = model(swapped).view(-1)
            valid_sw = valid & (ddg_pred_swap.abs() < 9000)
            if valid_sw.sum() > 0:
                anti_loss = ((ddg_pred[valid_sw] + ddg_pred_swap[valid_sw]) ** 2).mean()

        # OOD-margin: input-noise consistency regularization
        ood_margin_loss_val = torch.zeros((), device=device)
        if use_ood_margin and ood_margin_fn is not None:
            ood_margin_loss_val, _ = ood_margin_fn(model, batch, ddg_pred, valid)

        # Compose total
        total_loss = (
            ddg_loss
            + (siamese_weight  * anti_loss          if use_siamese     else 0.0)
            + (1.0             * bcas_loss_val      if use_bcas        else 0.0)
            + (ood_margin_weight * ood_margin_loss_val if use_ood_margin else 0.0)
            + (k50_loss_weight * k50_loss           if use_k50_head    else 0.0)
        )

        # Mixture-of-Experts auxiliary loss (load balancing + router z-loss),
        # averaged over every forward pass made in this step (main, Siamese
        # swapped, OOD-margin perturbed).
        # Depth-routing auxiliary loss (layer_moe), independent of the head MoE.
        _lf = getattr(model, "layer_fusion", None)
        if _lf is not None and hasattr(_lf, "pop_aux"):
            _lf_aux = _lf.pop_aux()
            if _lf_aux is not None:
                total_loss = total_loss + _lf_aux

        if use_moe:
            _head = getattr(model, "mlp", None)
            _aux = _head.pop_aux() if hasattr(_head, "pop_aux") else None
            if _aux is not None:
                total_loss = total_loss + moe_aux_weight * _aux
                running["moe_aux"] = running.get("moe_aux", 0.0) + float(_aux.item())

        if use_sam:
            # SAM: two forward/backward passes. First backward already done via
            # total_loss below; we recompute the full loss at perturbed weights.
            (total_loss / grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.first_step(zero_grad=True)

            # Recompute the SAME total loss at the perturbed weights w+e(w)
            ddg_pred2 = model(batch).view(-1)
            valid2 = (ddg_pred2.abs() < 9000) & torch.isfinite(ddg_true)
            ddg_loss2 = ddg_task_fn(ddg_pred2[valid2], ddg_true[valid2])
            anti_loss2 = torch.zeros((), device=device)
            bcas_loss2 = torch.zeros((), device=device)
            if use_bcas:
                sw2 = _make_swapped_batch(batch)
                dps2 = model(sw2).view(-1)
                v2 = valid2 & (dps2.abs() < 9000)
                bcas_loss2, _ = bcas_fn(ddg_pred2, dps2, v2)
            elif use_siamese:
                sw2 = _make_swapped_batch(batch)
                dps2 = model(sw2).view(-1)
                v2 = valid2 & (dps2.abs() < 9000)
                if v2.sum() > 0:
                    anti_loss2 = ((ddg_pred2[v2] + dps2[v2]) ** 2).mean()
            ood2 = torch.zeros((), device=device)
            if use_ood_margin and ood_margin_fn is not None:
                ood2, _ = ood_margin_fn(model, batch, ddg_pred2, valid2)
            total_loss2 = (
                ddg_loss2
                + (siamese_weight    * anti_loss2 if use_siamese   else 0.0)
                + (1.0               * bcas_loss2 if use_bcas      else 0.0)
                + (ood_margin_weight * ood2       if use_ood_margin else 0.0)
            )
            (total_loss2 / grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.second_step(zero_grad=True)
        else:
            (total_loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        running["total"]     += float(total_loss.item())
        running["ddg"]       += float(ddg_loss.item())
        running["anti"]      += float(anti_loss.item())
        running["bcas"]      += float(bcas_loss_val.item())
        running["ood_margin"]+= float(ood_margin_loss_val.item())
        running["k50"]       += float(k50_loss.item())
        n += 1
        loop.set_postfix(loss=f"{total_loss.item():.4f}")

    n = max(n, 1)
    return {k: v / n for k, v in running.items()}


def train(cfg: Optional[TrainConfig] = None):
    if cfg is None:
        cfg = TrainConfig()

    # Mutual exclusion: BCAS subsumes siamese
    if cfg.use_bcas and cfg.use_siamese:
        log.warning("Both use_bcas and use_siamese set; disabling use_siamese "
                    "(BCAS contains the siamese variance term).")
        cfg.use_siamese = False

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Config: {asdict(cfg)}")

    train_loader, val_loader = build_dataloaders(cfg)
    log.info(f"Train: {len(train_loader)}  Val: {len(val_loader)}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    ddg_task_fn, loss_module, _ = _build_ddg_loss(cfg, train_loader)
    if loss_module is not None:
        loss_module = loss_module.to(device)

    k50_loss_fn = K50MultiTaskLoss(beta=1.0).to(device) if cfg.use_k50_head else None
    bcas_fn = BCASLoss(alpha=cfg.bcas_alpha, beta=cfg.bcas_beta).to(device) \
        if cfg.use_bcas else None
    if cfg.use_ood_margin:
        if getattr(cfg, "ood_margin_module", None) is not None:
            ood_margin_fn = cfg.ood_margin_module.to(device)
        else:
            ood_margin_fn = OODMarginLoss(
                sigma=cfg.ood_margin_sigma, weight=cfg.ood_margin_weight,
                n_samples=cfg.ood_margin_samples,
            ).to(device)
    else:
        ood_margin_fn = None

    k50_augmentor = None
    if cfg.use_k50_head:
        k50_augmentor = K50BatchAugmentor(cfg.megascale_csv)
    move_b_augmentor = None
    if cfg.use_move_b_features:
        move_b_augmentor = MoveBFeatureAugmentor(
            cfg.move_b_cache_path, local_radius=cfg.move_b_local_radius)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if loss_module is not None:
        trainable_params += [p for p in loss_module.parameters() if p.requires_grad]
    if getattr(cfg, "use_sam", False):
        from .sam import SAM
        optimizer = SAM(trainable_params, base_optimizer=AdamW, rho=cfg.sam_rho,
                        lr=cfg.lr, weight_decay=cfg.weight_decay)
        log.info(f"Using SAM optimizer (rho={cfg.sam_rho})")
    else:
        optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    log.info(f"Loss type           : {cfg.loss_type}")
    log.info(f"Siamese anti-sym    : {cfg.use_siamese} (w={cfg.siamese_weight})")
    log.info(f"BCAS                : {cfg.use_bcas}"
             + (f" (α={cfg.bcas_alpha}, β={cfg.bcas_beta})" if cfg.use_bcas else ""))
    log.info(f"OOD-margin          : {cfg.use_ood_margin}"
             + (f" (σ={cfg.ood_margin_sigma}, w={cfg.ood_margin_weight})" if cfg.use_ood_margin else ""))
    log.info(f"K50 multi-task      : {cfg.use_k50_head}")
    log.info(f"Move-B features     : {cfg.use_move_b_features}")

    best_sp = -1.0; best_epoch = 0; no_improve = 0
    logs = []
    ckpt_path = os.path.join(cfg.output_dir, cfg.ckpt_filename)
    log_path  = os.path.join(cfg.output_dir, "training_log.csv")
    _manifest = Manifest.begin(cfg.output_dir, cfg,
                               repo_root=os.environ.get("DDG_REPO", "."),
                               data_root=cfg.data_root)    
    for epoch in range(1, cfg.max_epochs + 1):
        tr = train_one_epoch(
            model, ddg_task_fn, k50_loss_fn, bcas_fn, ood_margin_fn,
            k50_augmentor, move_b_augmentor,
            train_loader, optimizer, device, epoch,
            grad_accum=cfg.grad_accum,
            use_siamese=cfg.use_siamese, siamese_weight=cfg.siamese_weight,
            use_k50_head=cfg.use_k50_head, k50_loss_weight=cfg.k50_loss_weight,
            use_move_b=cfg.use_move_b_features,
            use_bcas=cfg.use_bcas, use_ood_margin=cfg.use_ood_margin,
            ood_margin_weight=cfg.ood_margin_weight,
            use_sam=getattr(cfg, 'use_sam', False),
            use_moe=getattr(cfg, 'use_moe', False),
            moe_aux_weight=getattr(cfg, 'moe_aux_weight', 0.01),
        )

        if epoch % cfg.val_every == 0:
            preds, targets = evaluate(model, val_loader, device, name="val")
            val_m = compute_metrics(preds, targets)
        else:
            val_m = {"spearman": float("nan"), "pearson": float("nan"),
                     "rmse": float("nan"), "mae": float("nan"),
                     "r2": float("nan"), "n": 0}

        flag = ""
        if not np.isnan(val_m["spearman"]) and val_m["spearman"] > best_sp:
            best_sp = val_m["spearman"]; best_epoch = epoch; no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = "  ← BEST"
        else:
            no_improve += 1

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in val_m.items()}}
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_path, index=False)

        parts = [f"ddg={tr['ddg']:.4f}"]
        if cfg.use_siamese: parts.append(f"anti={tr['anti']:.4f}")
        if cfg.use_bcas:    parts.append(f"bcas={tr['bcas']:.4f}")
        if cfg.use_ood_margin: parts.append(f"oodm={tr['ood_margin']:.4f}")
        if cfg.use_k50_head: parts.append(f"k50={tr['k50']:.4f}")
        log.info(
            f"Ep {epoch:03d} | {' '.join(parts)} | "
            f"val ρ={val_m['spearman']:.4f} r={val_m['pearson']:.4f} "
            f"rmse={val_m['rmse']:.4f}{flag}"
        )
        if no_improve >= cfg.early_stop_pat:
            log.info(f"Early stopping at epoch {epoch}")
            break

        log.info(f"Best checkpoint  : {ckpt_path}")
    _manifest.finish(epochs=epoch, best_val=best_sp, best_epoch=best_epoch,
                     ok=True)
    return ckpt_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    train()
