"""
moe_routing_analysis.py
=======================
Mechanistic evidence for the MoE result: are the experts actually specialising,
or has routing collapsed onto one or two while the shared expert does the work?

A benchmark-score improvement alone does not make an architecture paper. This
produces the "why it works" analysis:

  1. Per-expert assignment fractions and normalised routing entropy
     (1.0 = perfectly uniform, ~0 = collapsed onto a single expert).
  2. Effective number of experts used (exp of the entropy).
  3. What distinguishes the mutations routed to each expert: burial
     (CA neighbour count at 8 A), |ddG| magnitude, sequence position, and the
     most common wild-type and mutant residue identities.
  4. Association strength between expert assignment and each covariate
     (Cramer's V from a contingency table), which says whether routing is
     organised by structural environment, by effect magnitude, by residue
     chemistry, or by nothing in particular.

Routing is captured by wrapping the head's `_route` method at runtime, so the
repository is not modified.

Run:
    cd /backup/constraint-aware-ddg
    export DDG_DATA_ROOT=/backup/new_work/data_root
    export MPNN_CKPT=/backup/new_work/checkpoints/v_48_020.pt
    python analysis/moe_routing_analysis.py \
        --ckpt /backup/new_work/runs/moe/E_moe_s42/best.pt \
        --split s669
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ddg_data import Alphabet, Featurizer
from ddg_data.datasets.combined import MegaScaleTestDatasets
from ddg_models import MultimodalDDG, MultimodalDDGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AA = "ACDEFGHIKLMNPQRSTVWYX"


def _move(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def build_model(ckpt_path, args, device):
    cfg = MultimodalDDGConfig()
    cfg.adapter_layer_indices = [args.adapter_layer]
    cfg.proteinmpnn_ckpt = args.mpnn_ckpt
    cfg.encoder.tune = True
    cfg.mlp.input_dim = 128
    cfg.mlp.hidden_dim = [512, 512, 512]
    cfg.mlp.output_dim = 21
    cfg.mlp.dropout = 0.1
    cfg.mlp.use_moe = True
    cfg.mlp.moe_num_experts = args.num_experts
    cfg.mlp.moe_top_k = args.top_k
    cfg.mlp.moe_shared_expert = not args.no_shared
    cfg.mlp.moe_router_struct = not args.no_struct
    cfg.mlp.moe_noise_eps = 0.0
    if args.expert_hidden:
        cfg.mlp.moe_expert_hidden = [int(x) for x in args.expert_hidden.split(",")]

    model = MultimodalDDG(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    routed = [k for k in state if "experts" in k]
    if not routed:
        raise SystemExit("Checkpoint contains no expert weights — is this an MoE run?")
    log.info(f"loaded checkpoint ({len(routed)} expert tensors, "
             f"{len(missing)} missing / {len(unexpected)} unexpected keys)")
    model.eval()
    return model


def cramers_v(expert, covariate_bins, n_experts):
    """Association between expert assignment and a binned covariate."""
    cats = sorted(set(covariate_bins))
    table = np.zeros((n_experts, len(cats)))
    for e, c in zip(expert, covariate_bins):
        table[e, cats.index(c)] += 1
    n = table.sum()
    if n == 0:
        return float("nan")
    row, col = table.sum(1, keepdims=True), table.sum(0, keepdims=True)
    exp = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table - exp) ** 2 / np.where(exp > 0, exp, np.nan))
    k = min(table.shape)
    return float(np.sqrt(chi2 / (n * (k - 1)))) if k > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="s669")
    ap.add_argument("--num_experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--no_shared", action="store_true")
    ap.add_argument("--no_struct", action="store_true")
    ap.add_argument("--expert_hidden", default="")
    ap.add_argument("--adapter_layer", type=int, default=-3)
    ap.add_argument("--data_root", default=os.environ.get("DDG_DATA_ROOT", "/backup/new_work/data_root"))
    ap.add_argument("--mpnn_ckpt", default=os.environ.get("MPNN_CKPT", "/backup/new_work/checkpoints/v_48_020.pt"))
    ap.add_argument("--out", default=os.environ.get("RESULTS_DIR", "/backup/new_work/results") + "/moe_routing")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args.ckpt, args, device)
    head = model.mlp
    if not hasattr(head, "_route"):
        raise SystemExit("model.mlp is not an MoEReadout")

    # Capture routing decisions without modifying the repository.
    captured = {}
    orig_route = head._route

    def wrapped(flat_x, struct):
        gates, probs, logits, topk_idx = orig_route(flat_x, struct)
        captured["topk"] = topk_idx.detach().cpu().numpy()
        captured["gates"] = gates.detach().cpu().numpy()
        return gates, probs, logits, topk_idx

    head._route = wrapped

    alphabet = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    coll = MegaScaleTestDatasets(data_root=args.data_root)
    ds = next((d for n, d in coll.iter_named() if n == args.split), None)
    if ds is None:
        raise SystemExit(f"split {args.split} not found")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=featurizer)

    rows = []
    with torch.no_grad():
        for batch in loader:
            try:
                batch = _move(batch, device)
                _ = model(batch)
                topk = captured.get("topk")
                gates = captured.get("gates")
                if topk is None:
                    continue
                K = topk.shape[0]

                tgt = batch.get("ddG")
                tgt = (tgt.detach().cpu().numpy().ravel() if tgt is not None
                       else np.full(K, np.nan))

                # burial + relative position from the head's own descriptors
                flat = batch["muted_id_representation"].reshape(-1, batch["muted_id_representation"].shape[-1])
                sfeat = head._structural_features(batch, flat.shape[0], flat.device, flat.dtype)
                sfeat = sfeat.detach().cpu().numpy()

                at = batch.get("append_tensors")
                if at is not None:
                    a = at.detach().cpu().numpy().reshape(-1, at.shape[-1])
                    wt_aa = a[:, :21].argmax(-1)
                    mut_aa = a[:, 21:42].argmax(-1)
                else:
                    wt_aa = mut_aa = np.full(K, -1)

                n = min(K, len(tgt), sfeat.shape[0], len(wt_aa))
                for i in range(n):
                    rows.append(dict(
                        expert=int(topk[i, 0]),
                        gate=float(gates[i, topk[i, 0]]),
                        ddg=float(tgt[i]),
                        burial=float(sfeat[i, 0]),
                        relpos=float(sfeat[i, 3]),
                        wt=AA[wt_aa[i]] if 0 <= wt_aa[i] < 21 else "?",
                        mut=AA[mut_aa[i]] if 0 <= mut_aa[i] < 21 else "?",
                    ))
            except Exception as e:                      # keep going on a bad protein
                log.warning(f"skipped a protein: {e}")

    if not rows:
        raise SystemExit("no routing records collected")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"routing_{args.split}.csv", index=False)
    E = args.num_experts

    # 1-2. utilisation and entropy
    frac = df.expert.value_counts(normalize=True).reindex(range(E), fill_value=0.0).sort_index()
    p = frac.values
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(E)
    log.info(f"\n=== Expert utilisation on {args.split} (top-1 assignment, n={len(df)}) ===")
    for e in range(E):
        bar = "#" * int(round(p[e] * 60))
        log.info(f"  expert {e:2d}: {p[e]*100:5.1f}%  {bar}")
    log.info(f"  normalised routing entropy : {ent:.3f}  (1.0 = uniform)")
    log.info(f"  effective experts used     : {np.exp(-(p[p>0]*np.log(p[p>0])).sum()):.2f} / {E}")
    if ent < 0.5:
        log.info("  >> routing is substantially collapsed; the gain is unlikely to "
                 "come from specialisation")
    elif ent > 0.85:
        log.info("  >> routing is close to uniform; experts are all in use")

    # 3. what each expert sees
    log.info(f"\n=== Expert profiles ===")
    prof = []
    for e in range(E):
        sub = df[df.expert == e]
        if len(sub) == 0:
            continue
        top_sub = Counter(f"{w}->{m}" for w, m in zip(sub.wt, sub.mut)).most_common(3)
        prof.append(dict(expert=e, n=len(sub),
                         mean_ddg=sub.ddg.mean(), mean_abs_ddg=sub.ddg.abs().mean(),
                         mean_burial=sub.burial.mean(), mean_relpos=sub.relpos.mean(),
                         mean_gate=sub.gate.mean(),
                         top_substitutions="; ".join(f"{k}({v})" for k, v in top_sub)))
    pdf = pd.DataFrame(prof)
    pdf.to_csv(out / f"expert_profiles_{args.split}.csv", index=False)
    log.info("\n" + pdf.round(3).to_string(index=False))

    # 4. association strength
    log.info(f"\n=== Association between expert assignment and covariates ===")
    assoc = {}
    for name, series in [("burial", df.burial), ("|ddg|", df.ddg.abs()),
                         ("relpos", df.relpos)]:
        try:
            bins = pd.qcut(series, 4, labels=False, duplicates="drop").astype(int).tolist()
        except Exception:
            continue
        assoc[name] = cramers_v(df.expert.tolist(), bins, E)
    assoc["wt_residue"] = cramers_v(df.expert.tolist(), df.wt.tolist(), E)
    assoc["mut_residue"] = cramers_v(df.expert.tolist(), df.mut.tolist(), E)
    for k, v in sorted(assoc.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0)):
        log.info(f"  Cramer's V(expert, {k:<12s}) = {v:.3f}")
    log.info("  >> V below ~0.1 means routing is essentially unrelated to that "
             "covariate; 0.2+ indicates real organisation.")

    pd.DataFrame([assoc]).to_csv(out / f"routing_association_{args.split}.csv", index=False)
    log.info(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
