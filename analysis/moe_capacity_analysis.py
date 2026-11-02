

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "moe"
D_IN, D_OUT = 1408, 21
REFERENCE = "dense_512"
SPLITS = ["s669", "s461", "s8754", "s2648"]


def params(n_experts, hidden, shared_count, shared, mode="full", rank=8, active=None):
    h = hidden if isinstance(hidden, (list, tuple)) else [512, 512, 512]
    dims = [D_IN] + list(h) + [D_OUT]
    trunk = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    n_sh = shared_count if shared else 0
    router = (D_IN + 4) * n_experts * 2
    if mode == "lowrank":
        per_e = sum(rank * (dims[i] + dims[i + 1]) for i in range(len(dims) - 1))
        total = trunk + per_e * n_experts + router
        act = trunk + per_e * (active or 1)
    else:
        total = trunk * (n_experts + n_sh) + router
        act = trunk * ((active or 1) + n_sh)
    return total, act


def _get(row, key, default):
    """Registry rows written by older runs lack the newer columns, and pandas
    fills those with NaN. Fall back to the default in both cases."""
    try:
        v = row[key]
    except Exception:
        return default
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def load_preds(eval_dirs, split):
    ps, t = [], None
    for d in eval_dirs:
        p, q = Path(d) / f"preds_{split}.npy", Path(d) / f"targets_{split}.npy"
        if p.is_file() and q.is_file():
            ps.append(np.load(p)); t = np.load(q)
    return (np.mean(np.stack(ps, 0), 0), t) if ps else (None, None)


def boot(a, b, t, n=10000, rng=None):
    rng = rng or np.random.default_rng(0)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
    a, b, t = a[m], b[m], t[m]
    obs = spearmanr(b, t)[0] - spearmanr(a, t)[0]
    d = np.empty(n)
    for i in range(n):
        k = rng.integers(0, len(t), len(t))
        d[i] = spearmanr(b[k], t[k])[0] - spearmanr(a[k], t[k])[0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi, float(np.mean(d <= 0)), int(len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="experts16_top2", help="routed arm for the stratified analysis")
    args = ap.parse_args()

    reg = pd.read_csv(RESULTS / "registry.csv")
    long = pd.read_csv(RESULTS / "all_metrics_long.csv")
    out = RESULTS / "capacity"
    out.mkdir(parents=True, exist_ok=True)

    piv = (long.groupby(["arm", "benchmark"])["spearman"].mean()
               .unstack("benchmark"))
    meta = reg.drop_duplicates("arm").set_index("arm")

    # ---- 1. active-capacity scaling -----------------------------------
    rows = []
    for arm in piv.index:
        if arm not in meta.index:
            continue
        m = meta.loc[arm]
        try:
            hid = ast.literal_eval(str(m.expert_hidden))
        except Exception:
            hid = [512, 512, 512]
        mode = str(_get(m, "expert_mode", "full"))
        rank = int(_get(m, "lora_rank", 8))
        k = min(int(m.top_k), int(m.n_experts))
        tot, act = params(int(m.n_experts), hid, int(_get(m, "shared_count", 1)),
                          bool(_get(m, "shared", True)), mode, rank, active=k)
        rows.append(dict(arm=arm, routed=int(m.n_experts) > 1, mode=mode,
                         n_experts=int(m.n_experts), top_k=k,
                         total_M=tot / 1e6, active_M=act / 1e6,
                         s669=piv.loc[arm].get("s669"),
                         s461=piv.loc[arm].get("s461"),
                         megascale=piv.loc[arm].get("megascale_test")))
    cap = pd.DataFrame(rows).sort_values("active_M")
    cap.to_csv(out / "active_capacity.csv", index=False)
    log.info("\n=== Performance vs ACTIVE head parameters ===\n" +
             cap.round(4).to_string(index=False))

    ref = cap[cap.arm == REFERENCE]
    if not ref.empty:
        r = ref.iloc[0]
        log.info(f"\nReference {REFERENCE}: active {r.active_M:.2f}M, "
                 f"S669 {r.s669:.4f}, S461 {r.s461:.4f}")
        near = cap[(cap.active_M <= r.active_M * 1.4) & (cap.routed)]
        if not near.empty:
            log.info("Routed arms at comparable ACTIVE capacity:\n" +
                     near[["arm", "active_M", "s669", "s461"]].round(4).to_string(index=False))
            log.info("If these do not exceed the reference, sparsity is acting as a "
                     "capacity limiter rather than specialisation adding function.")

    # ---- 2. corrected bootstrap vs dense_512 ---------------------------
    def dirs(arm):
        return reg[reg.arm == arm].eval_dir.tolist()

    if REFERENCE not in set(reg.arm):
        log.warning(f"{REFERENCE} not in registry; skipping bootstrap")
    else:
        brows = []
        for split in SPLITS:
            base, t = load_preds(dirs(REFERENCE), split)
            if base is None:
                continue
            for arm in sorted(set(reg.arm)):
                if arm == REFERENCE:
                    continue
                p, _ = load_preds(dirs(arm), split)
                if p is None or len(p) != len(base):
                    continue
                o, lo, hi, pv, n = boot(base, p, t)
                brows.append(dict(split=split, arm=arm, delta=o, ci_low=lo,
                                  ci_high=hi, p=pv, n=n,
                                  beats_reference=bool(lo > 0)))
                log.info(f"{split:8s} {arm:18s} Δρ={o:+.4f} "
                         f"CI[{lo:+.4f},{hi:+.4f}] p={pv:.4f}"
                         + ("  *" if lo > 0 else ""))
        if brows:
            bdf = pd.DataFrame(brows)
            bdf.to_csv(out / "bootstrap_vs_dense512.csv", index=False)
            win = bdf[bdf.beats_reference]
            log.info(f"\nArms beating {REFERENCE} with CI excluding zero: "
                     f"{len(win)} of {len(bdf)} comparisons")
            if not win.empty:
                log.info("\n" + win[["split", "arm", "delta", "ci_low"]].round(4).to_string(index=False))

    # ---- 3. stratified win/loss ----------------------------------------
    base, t = load_preds(dirs(REFERENCE), "s669")
    cand, _ = load_preds(dirs(args.arm), "s669")
    if base is not None and cand is not None and len(base) == len(cand):
        m = np.isfinite(base) & np.isfinite(cand) & np.isfinite(t)
        b, c, y = base[m], cand[m], t[m]
        e_base, e_cand = (b - y) ** 2, (c - y) ** 2
        srows = []
        bands = [(-np.inf, -0.5, "stabilizing"), (-0.5, 0.5, "neutral"),
                 (0.5, 1.5, "destab. mild"), (1.5, np.inf, "destab. strong")]
        for lo_, hi_, name in bands:
            sel = (y >= lo_) & (y < hi_)
            if sel.sum() < 10:
                continue
            srows.append(dict(stratum=name, n=int(sel.sum()),
                              mse_dense=float(e_base[sel].mean()),
                              mse_routed=float(e_cand[sel].mean()),
                              delta_mse=float(e_cand[sel].mean() - e_base[sel].mean())))
        sdf = pd.DataFrame(srows)
        sdf.to_csv(out / f"stratified_{args.arm}_vs_{REFERENCE}.csv", index=False)
        log.info(f"\n=== Where {args.arm} wins and loses against {REFERENCE} (S669 MSE) ===\n"
                 + sdf.round(4).to_string(index=False))
        log.info("Negative delta_mse means the routed head is better in that band.")

    log.info(f"\nSaved under {out}")


if __name__ == "__main__":
    main()
