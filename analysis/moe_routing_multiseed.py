"""
moe_routing_multiseed.py
========================
Routing analysis aggregated over seeds, with a permutation null.

Two things the single-checkpoint version could not give you:

  * seed-level variability of expert utilisation, routing entropy and the
    expert/covariate association, so the specialisation claim is not resting
    on one checkpoint;
  * a permutation null for Cramer's V. With 21 residue categories and a few
    hundred mutations, V is upward-biased, so a raw 0.49 cannot be reported as
    evidence on its own. Expert labels are shuffled many times to obtain the
    null distribution, and the reported statistic is V together with the null
    mean and a z-score.

Run (after run_moe_full.py):
    cd /backup/constraint-aware-ddg
    python analysis/moe_routing_multiseed.py --arm experts16_top2 --split s669
    python analysis/moe_routing_multiseed.py --arm experts8_top2  --split s669
"""

from __future__ import annotations

import argparse
import logging
import os
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

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "moe"
AA = "ACDEFGHIKLMNPQRSTVWYX"
SEEDS = [42, 43, 44]


def _move(b, dev):
    return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}


def cramers_v(expert, cov, n_experts):
    cats = sorted(set(cov))
    tab = np.zeros((n_experts, len(cats)))
    for e, c in zip(expert, cov):
        tab[e, cats.index(c)] += 1
    n = tab.sum()
    if n == 0 or min(tab.shape) < 2:
        return float("nan")
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((tab - exp) ** 2 / np.where(exp > 0, exp, np.nan))
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def v_with_null(expert, cov, n_experts, n_perm=500, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = cramers_v(expert, cov, n_experts)
    e = np.asarray(expert)
    null = np.array([cramers_v(rng.permutation(e).tolist(), cov, n_experts)
                     for _ in range(n_perm)])
    mu, sd = float(np.nanmean(null)), float(np.nanstd(null))
    z = (obs - mu) / sd if sd > 0 else float("nan")
    return obs, mu, z


def build(ckpt, args, device, n_experts):
    cfg = MultimodalDDGConfig()
    cfg.adapter_layer_indices = [-3]
    cfg.proteinmpnn_ckpt = args.mpnn_ckpt
    cfg.encoder.tune = True
    cfg.mlp.input_dim, cfg.mlp.hidden_dim = 128, [512, 512, 512]
    cfg.mlp.output_dim, cfg.mlp.dropout = 21, 0.1
    cfg.mlp.use_moe = True
    cfg.mlp.moe_num_experts = n_experts
    cfg.mlp.moe_top_k = args.top_k
    cfg.mlp.moe_shared_expert = True
    cfg.mlp.moe_router_struct = True
    cfg.mlp.moe_noise_eps = 0.0
    if args.expert_hidden:
        cfg.mlp.moe_expert_hidden = [int(x) for x in args.expert_hidden.split(",")]
    m = MultimodalDDG(cfg).to(device)
    st = torch.load(ckpt, map_location=device)
    if "state_dict" in st:
        st = {k.replace("model.", "", 1): v for k, v in st["state_dict"].items()}
    m.load_state_dict(st, strict=False)
    m.eval()
    return m


def collect_one(ckpt, args, n_experts, device):
    model = build(ckpt, args, device, n_experts)
    head = model.mlp
    cap, orig = {}, head._route

    def wrapped(flat_x, struct, chem=None):
        try:
            g, p, l, idx = orig(flat_x, struct, chem)
        except TypeError:                    # older signature
            g, p, l, idx = orig(flat_x, struct)
        cap["topk"] = idx.detach().cpu().numpy()
        return g, p, l, idx

    head._route = wrapped

    featurizer = Featurizer(Alphabet(name="esm"))
    coll = MegaScaleTestDatasets(data_root=args.data_root)
    ds = next((d for n, d in coll.iter_named() if n == args.split), None)
    if ds is None:
        raise SystemExit(f"split {args.split} not found")

    rows = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=2, collate_fn=featurizer):
            try:
                batch = _move(batch, device)
                _ = model(batch)
                topk = cap.get("topk")
                if topk is None:
                    continue
                K = topk.shape[0]
                tgt = batch.get("ddG")
                tgt = tgt.detach().cpu().numpy().ravel() if tgt is not None else np.full(K, np.nan)
                rep = batch["muted_id_representation"]
                flat = rep.reshape(-1, rep.shape[-1])
                sf = head._structural_features(batch, flat.shape[0], flat.device,
                                               flat.dtype).detach().cpu().numpy()
                at = batch.get("append_tensors")
                a = at.detach().cpu().numpy().reshape(-1, at.shape[-1]) if at is not None else None
                wt = a[:, :21].argmax(-1) if a is not None else np.full(K, -1)
                mu = a[:, 21:42].argmax(-1) if a is not None else np.full(K, -1)
                n = min(K, len(tgt), sf.shape[0], len(wt))
                for i in range(n):
                    rows.append(dict(expert=int(topk[i, 0]), ddg=float(tgt[i]),
                                     burial=float(sf[i, 0]), relpos=float(sf[i, 3]),
                                     wt=AA[wt[i]] if 0 <= wt[i] < 21 else "?",
                                     mut=AA[mu[i]] if 0 <= mu[i] < 21 else "?"))
            except Exception as e:
                log.warning(f"skipped protein: {e}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="experts16_top2")
    ap.add_argument("--split", default="s669")
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--expert_hidden", default="")
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--data_root", default=os.environ.get("DDG_DATA_ROOT", "/backup/new_work/data_root"))
    ap.add_argument("--mpnn_ckpt", default=os.environ.get("MPNN_CKPT", "/backup/new_work/checkpoints/v_48_020.pt"))
    args = ap.parse_args()

    reg_path = RESULTS / "registry.csv"
    if not reg_path.is_file():
        raise SystemExit("results/moe/registry.csv not found — run scripts/run_moe_full.py first")
    reg = pd.read_csv(reg_path)
    sub = reg[reg.arm == args.arm]
    if sub.empty:
        raise SystemExit(f"arm {args.arm} not in registry")
    n_experts = int(sub.iloc[0].n_experts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = RESULTS / "routing"
    out.mkdir(parents=True, exist_ok=True)

    per_seed, all_df = [], []
    for _, r in sub.iterrows():
        ckpt = Path(r.eval_dir).parent / "best.pt"
        if not ckpt.is_file():
            log.warning(f"missing checkpoint {ckpt}")
            continue
        df = collect_one(str(ckpt), args, n_experts, device)
        if df.empty:
            continue
        df["seed"] = int(r.seed)
        all_df.append(df)

        p = df.expert.value_counts(normalize=True).reindex(range(n_experts), fill_value=0).values
        ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(n_experts)
        rec = dict(seed=int(r.seed), n=len(df), entropy=ent,
                   eff_experts=float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())))
        rng = np.random.default_rng(int(r.seed))
        for name, cov in [("wt_residue", df.wt.tolist()),
                          ("mut_residue", df.mut.tolist())]:
            v, mu, z = v_with_null(df.expert.tolist(), cov, n_experts, args.n_perm, rng)
            rec[f"V_{name}"], rec[f"Vnull_{name}"], rec[f"z_{name}"] = v, mu, z
        for name, series in [("burial", df.burial), ("abs_ddg", df.ddg.abs()),
                             ("relpos", df.relpos)]:
            try:
                bins = pd.qcut(series, 4, labels=False, duplicates="drop").astype(int).tolist()
            except Exception:
                continue
            v, mu, z = v_with_null(df.expert.tolist(), bins, n_experts, args.n_perm, rng)
            rec[f"V_{name}"], rec[f"Vnull_{name}"], rec[f"z_{name}"] = v, mu, z
        per_seed.append(rec)

    if not per_seed:
        raise SystemExit("no routing data collected")

    ps = pd.DataFrame(per_seed)
    ps.to_csv(out / f"routing_per_seed_{args.arm}_{args.split}.csv", index=False)
    pd.concat(all_df).to_csv(out / f"routing_records_{args.arm}_{args.split}.csv", index=False)

    log.info(f"\n=== Routing across seeds — {args.arm} on {args.split} ===")
    log.info("\n" + ps.round(3).to_string(index=False))

    summ = {"arm": args.arm, "split": args.split, "n_experts": n_experts,
            "entropy_mean": ps.entropy.mean(), "entropy_sd": ps.entropy.std(),
            "eff_experts_mean": ps.eff_experts.mean()}
    log.info(f"\nrouting entropy      : {ps.entropy.mean():.3f} +/- {ps.entropy.std():.3f}")
    log.info(f"effective experts    : {ps.eff_experts.mean():.2f} / {n_experts}")
    log.info("\nassociation (V, null mean, z) averaged over seeds:")
    for c in [c[2:] for c in ps.columns if c.startswith("V_")]:
        v, nu, z = ps[f"V_{c}"].mean(), ps[f"Vnull_{c}"].mean(), ps[f"z_{c}"].mean()
        summ[f"V_{c}"], summ[f"Vnull_{c}"], summ[f"z_{c}"] = v, nu, z
        log.info(f"  {c:12s} V={v:.3f}  null={nu:.3f}  z={z:6.1f}")
    log.info("\nz is how many null standard deviations the observed association "
             "sits above chance; the null mean is the bias floor for that many "
             "categories, so report V against it rather than against zero.")

    pd.DataFrame([summ]).to_csv(out / f"routing_summary_{args.arm}_{args.split}.csv", index=False)
    log.info(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
