

from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "moe"
TABLES = RESULTS / "tables"

BENCHMARKS = ["s669", "s461", "s571", "ssym_direct", "ssym_inverse",
              "fireport_hf", "s783", "s8754", "s2648", "s4346", "megascale_test"]
LABEL = {"s669": "S669", "s461": "S461", "s571": r"S571 ($\Delta T_m$)",
         "ssym_direct": "Ssym (dir.)", "ssym_inverse": "Ssym (inv.)",
         "fireport_hf": "FireProt-HF", "s783": "S783", "s8754": "S8754",
         "s2648": "S2648", "s4346": "S4346", "megascale_test": "Megascale-test"}

D_IN, D_OUT = 1408, 21


def head_params(n_experts, hidden, shared_count, shared, n_struct=4):
    h = hidden if isinstance(hidden, (list, tuple)) else [512, 512, 512]
    one = D_IN * h[0] + h[0]
    for i in range(len(h) - 1):
        one += h[i] * h[i + 1] + h[i + 1]
    one += h[-1] * D_OUT + D_OUT
    n_shared = shared_count if shared else 0
    router = (D_IN + n_struct) * n_experts * 2
    return one * (n_experts + n_shared) + router


def main():
    reg_path = RESULTS / "registry.csv"
    if not reg_path.is_file():
        raise SystemExit("results/moe/registry.csv not found — run scripts/run_moe_full.py")
    reg = pd.read_csv(reg_path)
    TABLES.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in reg.iterrows():
        f = Path(r.eval_dir) / "test_metrics.csv"
        if not f.is_file():
            log.warning(f"missing {f}")
            continue
        df = pd.read_csv(f)
        for _, m in df.iterrows():
            rows.append(dict(arm=r.arm, seed=r.seed, benchmark=m["dataset"],
                             spearman=m.get("spearman"), pearson=m.get("pearson"),
                             rmse=m.get("rmse"), mae=m.get("mae"), n=m.get("n")))
    if not rows:
        raise SystemExit("no metrics found")
    long = pd.DataFrame(rows)
    long.to_csv(RESULTS / "all_metrics_long.csv", index=False)
    log.info(f"{len(long)} metric rows from {long.arm.nunique()} arms, "
             f"{long.seed.nunique()} seeds")

    agg = (long.groupby(["arm", "benchmark"])["spearman"]
               .agg(mean="mean", sd="std", k="count").reset_index())
    agg.to_csv(RESULTS / "summary_mean_sd.csv", index=False)

    piv = agg.pivot(index="arm", columns="benchmark", values="mean")
    piv = piv.reindex(columns=[b for b in BENCHMARKS if b in piv.columns])
    log.info("\n=== Mean Spearman over seeds ===\n" + piv.round(4).to_string())

    # ---- scaling table: parameters vs performance -------------------------
    meta = reg.drop_duplicates(subset=["arm"]).set_index("arm")
    srows = []
    for arm in piv.index:
        if arm not in meta.index:
            continue
        m = meta.loc[arm]
        try:
            hidden = ast.literal_eval(str(m.expert_hidden))
        except Exception:
            hidden = [512, 512, 512]
        p = head_params(int(m.n_experts), hidden,
                        int(m.get("shared_count", 1)), bool(m.get("shared", True)))
        routed = int(m.n_experts) > 1
        active = head_params(min(int(m.top_k), int(m.n_experts)), hidden,
                             int(m.get("shared_count", 1)), bool(m.get("shared", True)))
        srows.append(dict(arm=arm, routed=routed, n_experts=int(m.n_experts),
                          top_k=int(m.top_k), head_params_M=p / 1e6,
                          active_params_M=active / 1e6,
                          s669=piv.loc[arm].get("s669"), s461=piv.loc[arm].get("s461"),
                          megascale=piv.loc[arm].get("megascale_test")))
    scal = pd.DataFrame(srows).sort_values(["routed", "head_params_M"])
    scal.to_csv(RESULTS / "scaling_table.csv", index=False)
    log.info("\n=== Parameters vs performance ===\n" + scal.round(4).to_string(index=False))

    sd = agg.set_index(["arm", "benchmark"])["sd"]

    def cell(arm, b):
        if arm not in piv.index or b not in piv.columns or pd.isna(piv.loc[arm, b]):
            return "--"
        s = sd.get((arm, b), np.nan)
        return f"{piv.loc[arm, b]:.3f}" + (f"$\\pm${s:.3f}" if pd.notna(s) else "")

    def write_table(name, arms, caption, label, benches=("s669", "s461", "ssym_direct", "megascale_test")):
        arms = [a for a in arms if a in piv.index]
        if not arms:
            return
        lines = [r"\begin{table}[t]\centering", f"\\caption{{{caption}}}",
                 f"\\label{{{label}}}",
                 r"\begin{tabular}{l" + "r" * (len(benches) + 1) + "}", r"\toprule",
                 "Readout & params (M) & " + " & ".join(LABEL.get(b, b) for b in benches) + r" \\",
                 r"\midrule"]
        for a in arms:
            pm = scal.loc[scal.arm == a, "head_params_M"]
            pstr = f"{pm.iloc[0]:.1f}" if len(pm) else "--"
            lines.append(f"{a.replace('_',' ')} & {pstr} & " +
                         " & ".join(cell(a, b) for b in benches) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TABLES / name).write_text("\n".join(lines))
        log.info(f"wrote {TABLES/name}")

    write_table("table_scaling.tex",
                ["dense_512", "dense_1024", "dense_1536", "dense_2048",
                 "experts4_top2", "experts8_top2", "experts16_top2", "experts32_top2"],
                "Dense versus routed readout capacity (mean$\\pm$s.d. over three seeds). "
                "Widening the dense head degrades cross-dataset performance monotonically, "
                "while the same budget allocated to routed experts improves it.",
                "tab:scaling")
    write_table("table_routing.tex",
                ["experts8_top2", "route_chemistry", "route_hybrid", "balance_lossfree"],
                "Routing and load-balancing variants. A fixed chemistry partition "
                "underperforms the learned router, so routing is organised along "
                "residue chemistry without being reducible to it.",
                "tab:routing")
    write_table("table_ablation.tex",
                ["experts8_top2", "experts8_top1", "experts8_top4", "no_shared",
                 "no_struct_route", "aux0", "aux0p05", "experts8_narrow",
                 "experts4_narrow", "shared2_fine"],
                "Component ablation of the mixture-of-experts readout "
                "(mean$\\pm$s.d. over three seeds).",
                "tab:moe_ablation")

    log.info(f"\nAll results under {RESULTS}")


if __name__ == "__main__":
    main()
