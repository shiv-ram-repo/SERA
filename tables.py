"""
tables.py
=========
Every LaTeX table in the paper, generated from the results files rather than
transcribed from a terminal. Transcription is how a number that was corrected
in the analysis survives in the manuscript.

Each table is written twice: `tables/<name>.tex` for \\input{} into the
manuscript, and `tables/<name>.csv` for checking by eye and for the supplement.

    python tables.py --list
    python tables.py
    python tables.py --only main,contrasts
    python tables.py --booktabs false     # if the journal forbids booktabs

TABLES
------
  main        arm x benchmark, mean +/- sd over seeds. The appendix table.
  contrasts   every contrast: pooled delta, 95% CI, sign count, p. The table
              the claims are read from.
  perbench    per-benchmark protein-level intervals for the surviving
              contrasts, since the pooled claim is weaker than it looks and a
              reader is entitled to the per-benchmark numbers alongside.
  experts     expert specialisation: n, enriched residues, median ddG, burial.
  adapter     the adapter-placement ablation, stratified by k.
  provenance  seeds, epochs and source hash per arm, so the run table in the
              supplement is generated and not remembered.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("tables")

RESULTS = Path("results")
TABDIR = Path("tables")
BOOKTABS = True

BENCH_ORDER = ["s669", "s461", "s571", "s783", "s2648", "s4346", "s8754",
               "ssym_direct", "ssym_inverse", "fireport_hf", "megascale_test"]
BENCH_TEX = {"s669": "S669", "s461": "S461", "s571": "S571", "s783": "S783",
             "s2648": "S2648", "s4346": "S4346", "s8754": "S8754",
             "ssym_direct": "Ssym$_\\mathrm{dir}$",
             "ssym_inverse": "Ssym$_\\mathrm{inv}$",
             "fireport_hf": "FireProt", "megascale_test": "Megascale"}

ARM_TEX = {
    "dense_512": "Dense, width 512 (baseline)",
    "dense_1024": "Dense, width 1024",
    "dense_1536": "Dense, width 1536",
    "dense_2048": "Dense, width 2048",
    "experts8_top1": "8 experts, top-1",
    "experts8_top2": "8 experts, top-2",
    "experts8_top4": "8 experts, top-4",
    "experts4_top2": "4 experts, top-2",
    "experts16_top2": "16 experts, top-2",
    "no_struct_route": "Router without structure",
    "route_chemistry": "Router: chemistry prior",
    "route_hybrid": "Router: hybrid prior",
    "no_shared": "No shared expert",
    "aux0": "No load-balancing loss",
    "balance_lossfree": "Loss-free balancing",
    "lowrank8_r8": "Low-rank experts ($r=8$)",
    "layermix_resid": "Scalar mix over depth",
    "layermoe_resid": "Routed over depth",
    "layermoe_resid_full": "Routed over depth, full stack",
    "moe8_layermoe_resid": "Routed depth $+$ routed experts",
    "fusion_film": "FiLM fusion",
    "fusion_bilinear": "Bilinear fusion",
    "fusion_both": "FiLM $+$ bilinear",
    "pair_rank2": "Pair coupling, rank 2",
    "pair_rank4": "Pair coupling, rank 4",
}

PRODUCER = {
    "all_metrics_long.csv": "python collect.py",
    "summary_mean_sd.csv": "python collect.py",
    "run_health.csv": "python collect.py",
    "bootstrap.csv": "python bootstrap.py",
    "bootstrap_pooled.csv": "python bootstrap.py",
    "expert_specialisation.csv": "python expert_routing.py",
    "routing.csv": "python expert_routing.py",
    "routing_summary.json": "python expert_routing.py",
    "adapter_metrics_long.csv": "python run_adapter_ablation.py --check",
    "registry_adapter.csv": "python run_adapter_ablation.py",
}


def need(*names: str) -> bool:
    missing = [n for n in names if not (RESULTS / n).is_file()]
    for m in missing:
        log.info(f"    missing {RESULTS / m}  ({PRODUCER.get(m, '?')})")
    return not missing


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#")):
        if a == "\\" and ("\\mathrm" in s or "$" in s):
            continue
        s = s.replace(a, b)
    return s


def arm_label(a: str) -> str:
    return ARM_TEX.get(a, esc(a))


def write(df: pd.DataFrame, name: str, caption: str, label: str,
          align: str | None = None, note: str = "", star: bool = False):
    """One table, as .tex and as .csv. The .tex is a float with a caption and
    a label, ready to \\input{}; the .csv is for checking it."""
    TABDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABDIR / f"{name}.csv", index=False)

    ncol = len(df.columns)
    align = align or ("l" + "r" * (ncol - 1))
    top, mid, bot = ((r"\toprule", r"\midrule", r"\bottomrule") if BOOKTABS
                     else (r"\hline", r"\hline", r"\hline"))
    env = "table*" if star else "table"
    # A table with eleven benchmark columns overflows the text block in every
    # two-column style. Shrinking to fit is better than a silent overfull box
    # that only shows up as text running into the margin at proof stage.
    fit = star or len(df.columns) > 8
    lines = [f"\\begin{{{env}}}[t]", r"\centering", r"\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if fit:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{align}}}", top,
              " & ".join(str(c) for c in df.columns) + r" \\", mid]
    for _, r in df.iterrows():
        cells = ["" if (isinstance(v, float) and not np.isfinite(v))
                 or v is None or (isinstance(v, str) and v == "nan")
                 else str(v) for v in r]
        lines.append(" & ".join(cells) + r" \\")
    lines += [bot, r"\end{tabular}"]
    if fit:
        lines.append("}")
    if note:
        lines.append(r"\par\vspace{2pt}{\footnotesize " + note + "}")
    lines.append(f"\\end{{{env}}}")
    (TABDIR / f"{name}.tex").write_text("\n".join(lines) + "\n")
    log.info(f"    wrote {TABDIR}/{name}.tex and .csv  "
             f"({len(df)} rows x {ncol} cols)")
    return True


# --------------------------------------------------------------------------- #
def tab_main() -> bool:
    """Every arm on every benchmark, mean +/- sd over seeds. The sd is blank,
    never zero, where only one seed ran: a single run must not be able to
    look like a converged estimate."""
    if not need("all_metrics_long.csv"):
        return False
    L = pd.read_csv(RESULTS / "all_metrics_long.csv")
    g = L.groupby(["arm", "benchmark"])["spearman"].agg(["mean", "std",
                                                         "count"])
    benches = [b for b in BENCH_ORDER if b in set(L.benchmark)]
    arms = [a for a in ARM_TEX if a in set(L.arm)] + \
           [a for a in sorted(set(L.arm)) if a not in ARM_TEX]
    rows = []
    for a in arms:
        row = {"Readout": arm_label(a)}
        for b in benches:
            if (a, b) not in g.index:
                row[BENCH_TEX.get(b, b)] = ""
                continue
            m, s, n = g.loc[(a, b)]
            row[BENCH_TEX.get(b, b)] = (f"{m:.3f}" if n < 2 or not np.isfinite(s)
                                        else f"{m:.3f}\\,{{\\scriptsize$\\pm$"
                                             f"{s:.3f}}}")
        row["$n$"] = int(g.loc[a]["count"].max()) if a in g.index.get_level_values(0) else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    nmin = int(L.groupby(["arm", "benchmark"]).size().min())
    return write(
        df, "tab_main",
        "Spearman correlation on every benchmark, mean over seeds with the "
        "seed standard deviation. $n$ is the number of seeds.",
        "tab:main", star=True,
        note=("The standard deviation here is over seeds only. It is not the "
              "uncertainty on a comparison between two arms: see "
              "Table~\\ref{tab:contrasts}, where proteins are resampled as "
              f"well. Minimum seeds per cell: {nmin}."))


def tab_contrasts() -> bool:
    """The table the paper's claims are read from. Both intervals are shown
    side by side, because the gap between them is itself a result."""
    if not need("bootstrap_pooled.csv"):
        return False
    P = pd.read_csv(RESULTS / "bootstrap_pooled.csv").drop_duplicates(
        subset=["contrast", "arm", "control"], keep="last")
    P = P.sort_values("mean_delta")
    rows = []
    for _, r in P.iterrows():
        sig = bool(r.lo > 0 or r.hi < 0)
        d = f"{r.mean_delta:+.4f}"
        if sig:
            d = f"\\textbf{{{d}}}"
        rows.append({
            "Contrast": esc(r.contrast),
            "Arm": arm_label(r.arm),
            "Control": arm_label(r.control),
            "$\\Delta\\rho$": d,
            "95\\% CI": f"[{r.lo:+.4f}, {r.hi:+.4f}]",
            "Same sign": f"{int(r.same_sign)}/{int(r.n_benchmarks)}",
            "Sign $p$": f"{r.sign_p:.4f}",
        })
    return write(
        pd.DataFrame(rows), "tab_contrasts",
        "Every architectural contrast, pooled across benchmarks. Each "
        "benchmark contributes one observation; the interval is a $t$ "
        "interval over benchmarks and the $p$-value is a two-sided sign test. "
        "Bold marks an interval excluding zero.",
        "tab:contrasts", star=True,
        note=("Intervals come from resampling proteins and seeds jointly "
              "within each benchmark, then pooling. Resampling seeds alone "
              "gives intervals several times narrower and declares effects "
              "significant that are not; that comparison is "
              "Figure~\\ref{fig:errorbars}."))


def tab_perbench() -> bool:
    """Per-benchmark protein-level intervals for the contrasts that survived
    pooling. A pooled claim without the per-benchmark numbers invites the
    reader to assume every benchmark agreed strongly, when the claim is only
    that they agreed in direction."""
    if not need("bootstrap.csv", "bootstrap_pooled.csv"):
        return False
    B = pd.read_csv(RESULTS / "bootstrap.csv")
    P = pd.read_csv(RESULTS / "bootstrap_pooled.csv").drop_duplicates(
        subset=["contrast", "arm", "control"], keep="last")
    keep = P[(P.lo > 0) | (P.hi < 0)]
    if keep.empty:
        log.info("    no pooled contrast excludes zero; table skipped")
        return False
    if "delta" not in B.columns:
        log.info("    bootstrap.csv holds no intervals (all pairs misaligned)")
        return False
    rows = []
    for _, k in keep.iterrows():
        sub = B[(B.arm == k.arm) & (B.control == k.control)]
        for b in BENCH_ORDER:
            s = sub[sub.benchmark == b]
            if s.empty:
                continue
            r = s.iloc[-1]
            if not np.isfinite(getattr(r, "delta", np.nan)):
                cell = esc(str(getattr(r, "note", "not computed"))[:46])
                rows.append({"Contrast": esc(k.contrast),
                             "Benchmark": BENCH_TEX.get(b, b),
                             "Proteins": "", "$\\Delta\\rho$": "",
                             "95\\% CI": cell})
                continue
            rows.append({
                "Contrast": esc(k.contrast),
                "Benchmark": BENCH_TEX.get(b, b),
                "Proteins": int(r.n_proteins) if np.isfinite(
                    getattr(r, "n_proteins", np.nan)) else "",
                "$\\Delta\\rho$": f"{r.delta:+.4f}",
                "95\\% CI": f"[{r.lo:+.4f}, {r.hi:+.4f}]"
                            + ("$^{*}$" if bool(r.excludes_zero) else ""),
            })
    if not rows:
        log.info("    no per-benchmark intervals for the surviving contrasts")
        return False
    return write(
        pd.DataFrame(rows), "tab_perbench",
        "Per-benchmark intervals for the contrasts that survive pooling, "
        "resampling proteins and seeds within each benchmark. $^{*}$ marks an "
        "interval excluding zero.",
        "tab:perbench",
        note=("A benchmark is listed without an interval where the two arms "
              "were not evaluated on identical rows, or where it holds too "
              "few proteins to resample; the reason is given in place of the "
              "interval. These benchmarks are excluded from the pooled "
              "estimate as well."))


def tab_experts() -> bool:
    """What each expert took. The claim is that the partition is chemical and
    tracks burial, so the table carries both, plus the n it rests on."""
    if not need("expert_specialisation.csv"):
        return False
    E = pd.read_csv(RESULTS / "expert_specialisation.csv")
    R = (pd.read_csv(RESULTS / "routing.csv")
         if (RESULTS / "routing.csv").is_file() else None)
    bcol = None
    if R is not None:
        bcol = next((c for c in ("burial10", "burial8", "burial12")
                     if c in R.columns), None)
    rows = []
    for _, r in E.iterrows():
        row = {"Expert": int(r.expert), "$n$": int(r.n),
               "Enriched wild-type": esc(r.top_wt),
               "Enriched mutant": esc(r.top_mut),
               "Median $\\Delta\\Delta G$": f"{r.median_ddg:+.2f}"}
        if R is not None and bcol:
            sub = R[R.expert == int(r.expert)]
            if len(sub):
                row["Median burial"] = f"{float(sub[bcol].median()):.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    note = ("Residues are listed in order of enrichment over the marginal "
            "frequency. Experts carrying fewer than 30 mutations are omitted.")
    fj = RESULTS / "routing_summary.json"
    if fj.is_file():
        j = json.loads(fj.read_text())
        note += (f" Mutual information between expert assignment and residue "
                 f"class is {float(j['mi_bits']):.3f} bits against a "
                 f"permutation null of {float(j['null_mean']):.4f} "
                 f"($z={float(j['z']):.0f}$), of "
                 f"{float(j.get('class_entropy', float('nan'))):.2f} bits "
                 f"available.")
    return write(df, "tab_experts",
                 "What each expert specialises in.", "tab:experts", note=note)


def tab_adapter() -> bool:
    """The adapter-placement ablation, stratified by the number of adapters.
    k is a column rather than a footnote because comparing across it is a
    capacity comparison, not a placement one, and the table has to make that
    impossible to miss."""
    if not need("adapter_metrics_long.csv", "registry_adapter.csv"):
        return False
    A = pd.read_csv(RESULTS / "adapter_metrics_long.csv")
    G = pd.read_csv(RESULTS / "registry_adapter.csv").drop_duplicates(
        subset=["arm"], keep="last").set_index("arm")
    g = A.groupby(["arm", "benchmark"])["spearman"].agg(["mean", "std",
                                                         "count"])
    benches = [b for b in BENCH_ORDER if b in set(A.benchmark)]
    arms = sorted(set(A.arm),
                  key=lambda a: (int(G.loc[a, "k_adapters"]) if a in G.index
                                 else 99,
                                 json.loads(G.loc[a, "adapter_layers"])[0]
                                 if a in G.index else 0))
    rows = []
    for a in arms:
        if a not in G.index:
            continue
        lays = json.loads(G.loc[a, "adapter_layers"])
        k = int(G.loc[a, "k_adapters"])
        shown = (", ".join(str(x) for x in lays) if k <= 6
                 else f"all {k}")
        row = {"$k$": k, "ESM-2 layers": shown}
        for b in benches:
            if (a, b) not in g.index:
                row[BENCH_TEX.get(b, b)] = ""
                continue
            m, s, n = g.loc[(a, b)]
            row[BENCH_TEX.get(b, b)] = (f"{m:.3f}" if n < 2 or
                                        not np.isfinite(s)
                                        else f"{m:.3f}\\,{{\\scriptsize$\\pm$"
                                             f"{s:.3f}}}")
        row["$n$"] = int(g.loc[a]["count"].max())
        rows.append(row)
    if not rows:
        return False
    return write(
        pd.DataFrame(rows), "tab_adapter",
        "Where the structural adapter sits inside ESM-2. Arms with the same "
        "$k$ have identical trainable-parameter counts, so differences within "
        "a $k$ are placement effects.",
        "tab:adapter", star=True,
        note=("Each additional adapter adds roughly 5.7M trainable "
              "parameters, so rows with different $k$ are not comparable as "
              "placement evidence and are reported only as a capacity "
              "reference. $k=1$, layer 30 is the inherited SPURS default."))


def tab_provenance() -> bool:
    """Seeds, epochs and the source hash every run was trained under. The
    supplement table that lets a reader establish the runs are comparable
    rather than take it on trust."""
    if not need("run_health.csv"):
        return False
    H = pd.read_csv(RESULTS / "run_health.csv")
    rows = []
    for arm, g in H.groupby("arm"):
        ep = pd.to_numeric(g.get("epochs"), errors="coerce").dropna()
        wm = pd.to_numeric(g.get("wall_min"), errors="coerce").dropna()
        ch = sorted({str(x)[:8] for x in g.get("code_hash", pd.Series(dtype=str))
                     .dropna()})
        rows.append({
            "Readout": arm_label(str(arm)),
            "Seeds": int(len(g)),
            "Usable": int(g.ok.sum()) if "ok" in g.columns else "",
            "Epochs (median)": f"{ep.median():.0f}" if len(ep) else "",
            "Minutes (median)": f"{wm.median():.0f}" if len(wm) else "",
            "Source hash": ", ".join(ch) if ch else "",
        })
    df = pd.DataFrame(rows).sort_values("Readout")
    nhash = len({h for r in rows for h in str(r["Source hash"]).split(", ") if h})
    return write(
        df, "tab_provenance",
        "Every run: seeds trained, seeds that passed the checkpoint and "
        "manifest checks, training length, and the hash of the source tree "
        "each was trained under.",
        "tab:provenance",
        note=(f"All runs share {'a single' if nhash == 1 else nhash} source "
              f"hash" + ("." if nhash == 1 else
                         "; arms under different hashes are not comparable "
                         "and are excluded from every contrast.") +
              " A run missing its manifest, or whose checkpoint does not "
              "contain the tensors its arm name implies, is excluded rather "
              "than averaged in."))


BUILDERS = {
    "main": (tab_main, "arm x benchmark, mean +/- sd"),
    "contrasts": (tab_contrasts, "pooled contrasts with CIs (the claims)"),
    "perbench": (tab_perbench, "per-benchmark CIs for survivors"),
    "experts": (tab_experts, "expert specialisation"),
    "adapter": (tab_adapter, "adapter placement, stratified by k"),
    "provenance": (tab_provenance, "SUPP: seeds, epochs, source hashes"),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--booktabs", default="true",
                    choices=["true", "false"])
    ap.add_argument("--traceback", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, (_f, why) in BUILDERS.items():
            log.info(f"  {k:12} {why}")
        return 0

    global RESULTS, TABDIR, BOOKTABS
    RESULTS, TABDIR = Path(a.results), Path(a.out)
    BOOKTABS = (a.booktabs == "true")

    want = [x.strip() for x in a.only.split(",") if x.strip()] or list(BUILDERS)
    unknown = [w for w in want if w not in BUILDERS]
    if unknown:
        log.error(f"unknown tables: {unknown}; --list to see them all")
        return 2

    made, skipped = 0, []
    for k in want:
        fn, why = BUILDERS[k]
        log.info(f"  {k}: {why}")
        try:
            ok = bool(fn())
        except Exception as exc:                                  # noqa: BLE001
            log.error(f"    FAILED: {type(exc).__name__}: {exc}")
            if a.traceback:
                import traceback
                traceback.print_exc()
            ok = False
        made += int(ok)
        if not ok:
            skipped.append(k)

    log.info("")
    log.info(f"  {made} tables written to {TABDIR}/")
    if skipped:
        log.info(f"  not built: {', '.join(skipped)}")
    if made:
        log.info("")
        log.info("  in the manuscript preamble:  \\usepackage{booktabs}"
                 if BOOKTABS else "")
        log.info(f"  then:  \\input{{{TABDIR}/tab_contrasts.tex}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
