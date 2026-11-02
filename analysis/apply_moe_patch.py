

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLP = ROOT / "src/ddg_models/ddg_models/mlp.py"
MMD = ROOT / "src/ddg_models/ddg_models/multimodal_ddg.py"
TRN = ROOT / "src/ddg_train/ddg_train/train.py"
EVL = ROOT / "src/ddg_train/ddg_train/evaluate.py"

changed, skipped, failed = [], [], []


# evaluate.py is distributed as a complete file rather than patched: chaining
# independent edits there left the signature and the body out of step, which
# produced undefined-name errors only at run time.
SKIP_FILES = {"evaluate.py"}


def edit(path: Path, anchor: str, addition: str, marker: str, after: bool = True):
    """Insert `addition` next to `anchor` unless `marker` is already present."""
    if path.name in SKIP_FILES:
        skipped.append(f"{path.name}: shipped complete, not patched")
        return
    text = path.read_text()
    if marker in text:
        skipped.append(f"{path.name}: already patched ({marker!r})")
        return
    if anchor not in text:
        failed.append(f"{path.name}: anchor not found -> {anchor.strip()[:60]!r}")
        return
    if text.count(anchor) != 1:
        failed.append(f"{path.name}: anchor not unique -> {anchor.strip()[:60]!r}")
        return
    new = (anchor + addition) if after else (addition + anchor)
    path.write_text(text.replace(anchor, new))
    changed.append(f"{path.name}: inserted near {anchor.strip()[:48]!r}")


def replace(path: Path, old: str, new: str, marker: str):
    if path.name in SKIP_FILES:
        skipped.append(f"{path.name}: shipped complete, not patched")
        return
    text = path.read_text()
    if marker in text:
        skipped.append(f"{path.name}: already patched ({marker!r})")
        return
    if text.count(old) != 1:
        failed.append(f"{path.name}: replace anchor missing/not unique -> {old.strip()[:60]!r}")
        return
    path.write_text(text.replace(old, new))
    changed.append(f"{path.name}: replaced {old.strip()[:48]!r}")


# ---------------------------------------------------------------- mlp.py
edit(
    MLP,
    anchor="    flat_dim:   int   = -1              # if > 0, ESM features are pre-flattened\n"
           "                                        # to this dim before concat with encoder feats\n",
    addition=(
        "\n"
        "    # --- Mixture-of-Experts readout -------------------------------\n"
        "    use_moe:            bool  = False   # swap the dense head for MoEReadout\n"
        "    moe_num_experts:    int   = 8\n"
        "    moe_top_k:          int   = 2\n"
        "    moe_shared_expert:  bool  = True    # always-on dense path\n"
        "    moe_router_struct:  bool  = True    # add structural descriptors to router\n"
        "    moe_noise_eps:      float = 1.0     # noisy top-k exploration (train only)\n"
        "    moe_lb_weight:      float = 1.0     # load-balancing loss weight\n"
        "    moe_z_weight:       float = 1e-3    # router z-loss weight\n"
        "    moe_expert_hidden:  List[int] = field(\n"
        "        default_factory=lambda: [512, 512, 512]\n"
        "    )\n"
    ),
    marker="use_moe:",
)

# ------------------------------------------------------ multimodal_ddg.py
replace(
    MMD,
    old="        self.mlp       = MLP(mlp_cfg)",
    new=(
        "        _use_moe = (mlp_cfg.get(\"use_moe\", False) if isinstance(mlp_cfg, dict)\n"
        "                    else getattr(mlp_cfg, \"use_moe\", False))\n"
        "        if _use_moe:\n"
        "            from .moe_head import MoEReadout\n"
        "            self.mlp   = MoEReadout(mlp_cfg)\n"
        "        else:\n"
        "            self.mlp   = MLP(mlp_cfg)"
    ),
    marker="MoEReadout(mlp_cfg)",
)

# -------------------------------------------------------------- train.py
edit(
    TRN,
    anchor="    ood_margin_samples:  int   = 1\n",
    addition=(
        "    # --- Mixture-of-Experts readout -------------------------------\n"
        "    use_moe:            bool  = False\n"
        "    moe_num_experts:    int   = 8\n"
        "    moe_top_k:          int   = 2\n"
        "    moe_aux_weight:     float = 0.01    # weight on (load-balance + z) aux loss\n"
        "    moe_shared_expert:  bool  = True\n"
        "    moe_router_struct:  bool  = True\n"
        "    moe_noise_eps:      float = 1.0\n"
    ),
    marker="use_moe:            bool  = False",
)

edit(
    TRN,
    anchor="    spurs_cfg.mlp.dropout           = 0.1\n",
    addition=(
        "    if getattr(cfg, \"use_moe\", False):\n"
        "        spurs_cfg.mlp.use_moe            = True\n"
        "        spurs_cfg.mlp.moe_num_experts    = cfg.moe_num_experts\n"
        "        spurs_cfg.mlp.moe_top_k          = cfg.moe_top_k\n"
        "        spurs_cfg.mlp.moe_shared_expert  = cfg.moe_shared_expert\n"
        "        spurs_cfg.mlp.moe_router_struct  = cfg.moe_router_struct\n"
        "        spurs_cfg.mlp.moe_noise_eps      = cfg.moe_noise_eps\n"
    ),
    marker="spurs_cfg.mlp.use_moe",
)

replace(
    TRN,
    old="    ood_margin_weight=0.5,\n    use_sam=False,\n):",
    new="    ood_margin_weight=0.5,\n    use_sam=False,\n    use_moe=False, moe_aux_weight=0.01,\n):",
    marker="use_moe=False, moe_aux_weight=0.01,",
)

edit(
    TRN,
    anchor="            + (k50_loss_weight * k50_loss           if use_k50_head    else 0.0)\n        )\n",
    addition=(
        "\n"
        "        # Mixture-of-Experts auxiliary loss (load balancing + router z-loss),\n"
        "        # averaged over every forward pass made in this step (main, Siamese\n"
        "        # swapped, OOD-margin perturbed).\n"
        "        if use_moe:\n"
        "            _head = getattr(model, \"mlp\", None)\n"
        "            _aux = _head.pop_aux() if hasattr(_head, \"pop_aux\") else None\n"
        "            if _aux is not None:\n"
        "                total_loss = total_loss + moe_aux_weight * _aux\n"
        "                running[\"moe_aux\"] = running.get(\"moe_aux\", 0.0) + float(_aux.item())\n"
    ),
    marker="pop_aux()",
)

edit(
    TRN,
    anchor="            use_sam=getattr(cfg, 'use_sam', False),\n",
    addition=(
        "            use_moe=getattr(cfg, 'use_moe', False),\n"
        "            moe_aux_weight=getattr(cfg, 'moe_aux_weight', 0.01),\n"
    ),
    marker="use_moe=getattr(cfg, 'use_moe', False),",
)

# ----------------------------------------------------------- evaluate.py
edit(
    EVL,
    anchor="    move_b_local_radius: float = 8.0,\n",
    addition=(
        "    use_moe: bool = False,\n"
        "    moe_num_experts: int = 8,\n"
        "    moe_top_k: int = 2,\n"
        "    moe_shared_expert: bool = True,\n"
        "    moe_router_struct: bool = True,\n"
    ),
    marker="use_moe: bool = False,",
)

edit(
    EVL,
    anchor="    cfg.mlp.dropout           = 0.1\n",
    addition=(
        "    if use_moe:\n"
        "        cfg.mlp.use_moe           = True\n"
        "        cfg.mlp.moe_num_experts   = moe_num_experts\n"
        "        cfg.mlp.moe_top_k         = moe_top_k\n"
        "        cfg.mlp.moe_shared_expert = moe_shared_expert\n"
        "        cfg.mlp.moe_router_struct = moe_router_struct\n"
        "        cfg.mlp.moe_noise_eps     = 0.0   # deterministic routing at eval\n"
    ),
    marker="cfg.mlp.use_moe",
)


# ---- expert width (parameter-matched control) --------------------------
edit(
    TRN,
    anchor="    moe_noise_eps:      float = 1.0\n",
    addition=(
        "    moe_expert_hidden:  List[int] = field(\n"
        "        default_factory=lambda: [512, 512, 512]\n"
        "    )   # narrow the experts to parameter-match the dense head\n"
    ),
    marker="moe_expert_hidden:  List[int]",
)

edit(
    TRN,
    anchor="        spurs_cfg.mlp.moe_noise_eps      = cfg.moe_noise_eps\n",
    addition="        spurs_cfg.mlp.moe_expert_hidden  = list(cfg.moe_expert_hidden)\n",
    marker="spurs_cfg.mlp.moe_expert_hidden",
)

edit(
    EVL,
    anchor="    moe_router_struct: bool = True,\n",
    addition="    moe_expert_hidden: Optional[list] = None,\n",
    marker="moe_expert_hidden: Optional[list]",
)

edit(
    EVL,
    anchor="        cfg.mlp.moe_router_struct = moe_router_struct\n",
    addition=(
        "        if moe_expert_hidden is not None:\n"
        "            cfg.mlp.moe_expert_hidden = list(moe_expert_hidden)\n"
    ),
    marker="cfg.mlp.moe_expert_hidden",
)


# ---- evaluate.py: keep the head-swap logic away from the MoE head ------
replace(
    EVL,
    old=(
        "    head_type = _detect_head_type(state)\n"
        "    print(f\"Detected head type: {head_type}\")\n"
        "    head_input_dim = model.mlp.fcs[0].in_features\n"
    ),
    new=(
        "    head_type = _detect_head_type(state)\n"
        "    if use_moe:\n"
        "        # The MoE head is constructed from cfg above; never swap it for\n"
        "        # the k50 / move_b heads, and read its input dim defensively.\n"
        "        head_type = \"moe\"\n"
        "    print(f\"Detected head type: {head_type}\")\n"
        "    _fcs = getattr(model.mlp, \"fcs\", None)\n"
        "    head_input_dim = (_fcs[0].in_features if _fcs is not None\n"
        "                      else int(cfg.mlp.input_dim))\n"
    ),
    marker='head_type = "moe"',
)


# ---- routing / balancing modes (literature-informed variants) ----------
edit(
    MLP,
    anchor="    moe_z_weight:       float = 1e-3    # router z-loss weight\n",
    addition=(
        "    moe_route_mode:     str   = \"learned\"   # learned | chemistry | hybrid\n"
        "    moe_balance_mode:   str   = \"aux\"       # aux | loss_free\n"
        "    moe_bias_gamma:     float = 1e-3         # loss-free bias step\n"
        "    moe_shared_count:   int   = 1            # number of always-on experts\n"
        "    moe_chem_prior_scale: float = 2.0        # hybrid chemistry prior strength\n"
    ),
    marker="moe_route_mode:",
)

edit(
    TRN,
    anchor="    moe_noise_eps:      float = 1.0\n",
    addition=(
        "    moe_route_mode:     str   = \"learned\"\n"
        "    moe_balance_mode:   str   = \"aux\"\n"
        "    moe_bias_gamma:     float = 1e-3\n"
        "    moe_shared_count:   int   = 1\n"
        "    moe_chem_prior_scale: float = 2.0\n"
    ),
    marker="moe_route_mode:     str",
)

edit(
    TRN,
    anchor="        spurs_cfg.mlp.moe_expert_hidden  = list(cfg.moe_expert_hidden)\n",
    addition=(
        "        spurs_cfg.mlp.moe_route_mode     = cfg.moe_route_mode\n"
        "        spurs_cfg.mlp.moe_balance_mode   = cfg.moe_balance_mode\n"
        "        spurs_cfg.mlp.moe_bias_gamma     = cfg.moe_bias_gamma\n"
        "        spurs_cfg.mlp.moe_shared_count   = cfg.moe_shared_count\n"
        "        spurs_cfg.mlp.moe_chem_prior_scale = cfg.moe_chem_prior_scale\n"
    ),
    marker="spurs_cfg.mlp.moe_route_mode",
)

edit(
    EVL,
    anchor="    moe_expert_hidden: Optional[list] = None,\n",
    addition=(
        "    moe_route_mode: str = \"learned\",\n"
        "    moe_balance_mode: str = \"aux\",\n"
        "    moe_shared_count: int = 1,\n"
    ),
    marker="moe_route_mode: str =",
)

edit(
    EVL,
    anchor="            cfg.mlp.moe_expert_hidden = list(moe_expert_hidden)\n",
    addition=(
        "        cfg.mlp.moe_route_mode    = moe_route_mode\n"
        "        cfg.mlp.moe_balance_mode  = moe_balance_mode\n"
        "        cfg.mlp.moe_shared_count  = moe_shared_count\n"
    ),
    marker="cfg.mlp.moe_route_mode",
)


# ---- low-rank experts (shared trunk + per-expert deltas) ---------------
edit(
    MLP,
    anchor="    moe_chem_prior_scale: float = 2.0        # hybrid chemistry prior strength\n",
    addition=(
        "    moe_expert_mode:    str   = \"full\"     # full | lowrank\n"
        "    moe_lora_rank:      int   = 8\n"
        "    moe_lora_scale:     float = 1.0\n"
    ),
    marker="moe_expert_mode:",
)

edit(
    TRN,
    anchor="    moe_chem_prior_scale: float = 2.0\n",
    addition=(
        "    moe_expert_mode:    str   = \"full\"\n"
        "    moe_lora_rank:      int   = 8\n"
        "    moe_lora_scale:     float = 1.0\n"
    ),
    marker="moe_expert_mode:    str",
)

edit(
    TRN,
    anchor="        spurs_cfg.mlp.moe_chem_prior_scale = cfg.moe_chem_prior_scale\n",
    addition=(
        "        spurs_cfg.mlp.moe_expert_mode    = cfg.moe_expert_mode\n"
        "        spurs_cfg.mlp.moe_lora_rank      = cfg.moe_lora_rank\n"
        "        spurs_cfg.mlp.moe_lora_scale     = cfg.moe_lora_scale\n"
    ),
    marker="spurs_cfg.mlp.moe_expert_mode",
)

edit(
    EVL,
    anchor="    moe_shared_count: int = 1,\n",
    addition=(
        "    moe_expert_mode: str = \"full\",\n"
        "    moe_lora_rank: int = 8,\n"
    ),
    marker="moe_expert_mode: str =",
)

edit(
    EVL,
    anchor="        cfg.mlp.moe_shared_count  = moe_shared_count\n",
    addition=(
        "        cfg.mlp.moe_expert_mode   = moe_expert_mode\n"
        "        cfg.mlp.moe_lora_rank     = moe_lora_rank\n"
    ),
    marker="cfg.mlp.moe_expert_mode",
)


# ---- multi-layer ESM fusion -------------------------------------------
edit(
    MLP,
    anchor="    moe_lora_scale:     float = 1.0\n",
    addition=(
        "    # Multi-layer encoder fusion. The readout otherwise sees only the\n"
        "    # final ESM layer; probing studies find intermediate layers usually\n"
        "    # carry more task-relevant signal and that the best depth is\n"
        "    # dataset-dependent.\n"
        "    use_layer_fusion:   bool  = False\n"
        "    layer_fusion_mode:  str   = \"scalar_mix\"   # scalar_mix | layer_moe\n"
        "    layer_fusion_topk:  int   = 2\n"
        "    esm_dim:            int   = 1280\n"
        "    layer_indices:      List[int] = field(default_factory=list)\n"
    ),
    marker="use_layer_fusion:",
)

edit(
    TRN,
    anchor="    moe_lora_scale:     float = 1.0\n",
    addition=(
        "    use_layer_fusion:   bool  = False\n"
        "    layer_fusion_mode:  str   = \"scalar_mix\"\n"
        "    layer_fusion_topk:  int   = 2\n"
        "    layer_indices:      List[int] = field(default_factory=list)\n"
    ),
    marker="use_layer_fusion:   bool",
)

edit(
    TRN,
    anchor="    spurs_cfg.mlp.dropout           = 0.1\n",
    addition=(
        "    if getattr(cfg, \"use_layer_fusion\", False):\n"
        "        spurs_cfg.mlp.use_layer_fusion  = True\n"
        "        spurs_cfg.mlp.layer_fusion_mode = cfg.layer_fusion_mode\n"
        "        spurs_cfg.mlp.layer_fusion_topk = cfg.layer_fusion_topk\n"
        "        spurs_cfg.mlp.layer_indices     = list(cfg.layer_indices)\n"
    ),
    marker="spurs_cfg.mlp.use_layer_fusion",
)

edit(
    EVL,
    anchor="    moe_lora_rank: int = 8,\n",
    addition=(
        "    use_layer_fusion: bool = False,\n"
        "    layer_fusion_mode: str = \"scalar_mix\",\n"
        "    layer_fusion_topk: int = 2,\n"
        "    layer_indices: Optional[list] = None,\n"
    ),
    marker="use_layer_fusion: bool = False,",
)

edit(
    EVL,
    anchor="    cfg.mlp.dropout           = 0.1\n",
    addition=(
        "    if use_layer_fusion:\n"
        "        cfg.mlp.use_layer_fusion  = True\n"
        "        cfg.mlp.layer_fusion_mode = layer_fusion_mode\n"
        "        cfg.mlp.layer_fusion_topk = layer_fusion_topk\n"
        "        cfg.mlp.layer_indices     = list(layer_indices or [])\n"
    ),
    marker="cfg.mlp.use_layer_fusion",
)


# ---- layer-mix init + multi-depth structural injection ----------------
edit(
    MLP,
    anchor="    esm_dim:            int   = 1280\n",
    addition="    layer_mix_init:     str   = \"final\"      # final | uniform\n",
    marker="layer_mix_init:",
)

edit(
    TRN,
    anchor="    layer_fusion_topk:  int   = 2\n",
    addition=(
        "    layer_mix_init:     str   = \"final\"\n"
        "    # Structural adapters at several depths. Empty keeps the single\n"
        "    # adapter at `adapter_layer`.\n"
        "    adapter_layer_indices: List[int] = field(default_factory=list)\n"
    ),
    marker="layer_mix_init:     str",
)

replace(
    TRN,
    old="    spurs_cfg.adapter_layer_indices = [cfg.adapter_layer]",
    new=("    spurs_cfg.adapter_layer_indices = (list(cfg.adapter_layer_indices)\n"
         "                                       if getattr(cfg, \"adapter_layer_indices\", None)\n"
         "                                       else [cfg.adapter_layer])"),
    marker="list(cfg.adapter_layer_indices)",
)

edit(
    TRN,
    anchor="        spurs_cfg.mlp.layer_indices     = list(cfg.layer_indices)\n",
    addition="        spurs_cfg.mlp.layer_mix_init    = cfg.layer_mix_init\n",
    marker="spurs_cfg.mlp.layer_mix_init",
)

edit(
    EVL,
    anchor="    layer_indices: Optional[list] = None,\n",
    addition=(
        "    layer_mix_init: str = \"final\",\n"
        "    adapter_layer_indices: Optional[list] = None,\n"
    ),
    marker="layer_mix_init: str =",
)

replace(
    EVL,
    old="    cfg.adapter_layer_indices = [adapter_layer]",
    new=("    cfg.adapter_layer_indices = (list(adapter_layer_indices)\n"
         "                                 if adapter_layer_indices else [adapter_layer])"),
    marker="list(adapter_layer_indices)",
)

edit(
    EVL,
    anchor="        cfg.mlp.layer_indices     = list(layer_indices or [])\n",
    addition="        cfg.mlp.layer_mix_init    = layer_mix_init\n",
    marker="cfg.mlp.layer_mix_init",
)


# ---- multiplicative sequence/structure fusion + head width -------------
edit(
    MLP,
    anchor="    layer_mix_init:     str   = \"final\"      # final | uniform\n",
    addition=(
        "    # Sequence/structure fusion. Concatenation is additive; FiLM and the\n"
        "    # low-rank bilinear term supply multiplicative interaction.\n"
        "    fusion_mode:        str   = \"concat\"   # concat | film | bilinear | film_bilinear\n"
        "    fusion_rank:        int   = 256\n"
        "    fusion_hidden:      int   = 512\n"
    ),
    marker="fusion_mode:",
)

edit(
    TRN,
    anchor="    layer_mix_init:     str   = \"final\"\n",
    addition=(
        "    fusion_mode:        str   = \"concat\"\n"
        "    fusion_rank:        int   = 256\n"
        "    fusion_hidden:      int   = 512\n"
        "    # Readout width. Capacity is a free design choice here, not a constraint.\n"
        "    mlp_hidden:         List[int] = field(default_factory=lambda: [512, 512, 512])\n"
    ),
    marker="fusion_mode:        str",
)

replace(
    TRN,
    old="    spurs_cfg.mlp.hidden_dim        = [512, 512, 512]",
    new="    spurs_cfg.mlp.hidden_dim        = list(getattr(cfg, \"mlp_hidden\", None) or [512, 512, 512])",
    marker='getattr(cfg, "mlp_hidden"',
)

edit(
    TRN,
    anchor="    spurs_cfg.mlp.dropout           = 0.1\n",
    addition=(
        "    spurs_cfg.mlp.fusion_mode       = getattr(cfg, \"fusion_mode\", \"concat\")\n"
        "    spurs_cfg.mlp.fusion_rank       = getattr(cfg, \"fusion_rank\", 256)\n"
        "    spurs_cfg.mlp.fusion_hidden     = getattr(cfg, \"fusion_hidden\", 512)\n"
    ),
    marker="spurs_cfg.mlp.fusion_mode",
)

edit(
    EVL,
    anchor="    layer_mix_init: str = \"final\",\n",
    addition=(
        "    fusion_mode: str = \"concat\",\n"
        "    fusion_rank: int = 256,\n"
        "    fusion_hidden: int = 512,\n"
        "    mlp_hidden: Optional[list] = None,\n"
    ),
    marker="fusion_mode: str =",
)

replace(
    EVL,
    old="    cfg.mlp.hidden_dim        = [512, 512, 512]",
    new="    cfg.mlp.hidden_dim        = list(mlp_hidden or [512, 512, 512])",
    marker="list(mlp_hidden or",
)

edit(
    EVL,
    anchor="    cfg.mlp.dropout           = 0.1\n",
    addition=(
        "    cfg.mlp.fusion_mode       = fusion_mode\n"
        "    cfg.mlp.fusion_rank       = fusion_rank\n"
        "    cfg.mlp.fusion_hidden     = fusion_hidden\n"
    ),
    marker="cfg.mlp.fusion_mode",
)


for line in changed:
    print("  PATCHED  " + line)
for line in skipped:
    print("  skip     " + line)
for line in failed:
    print("  FAILED   " + line)
if failed:
    print(f"\n{len(failed)} edit(s) could not be applied. The files above differ "
          "from what this patch expects; report these lines rather than hand-editing.")

# Syntax check everything we touched.
import ast
for f in (MLP, MMD, TRN, EVL):
    ast.parse(f.read_text())
print("\nAll patched files parse cleanly.")
print("Next: python analysis/moe_smoke_test.py")
