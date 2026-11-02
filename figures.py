#!/usr/bin/env python3
"""
figures.py
==========
Every manuscript figure, built from the files this study actually wrote.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("figures")

# Okabe-Ito, assigned in fixed order, never cycled.
C = dict(grey="#4D4D4D", orange="#E69F00", sky="#56B4E9", green="#009E73",
         yellow="#F0E442", blue="#0072B2", red="#D55E00", pink="#CC79A7")
INK, MUTED, GRID = "#1A1A1A", "#8A8A8A", "#EAEAEA"
SINGLE, DOUBLE = 3.5, 7.2

BENCH_ORDER = ["s669", "s461", "s2648", "s783", "s8754", "ssym_direct",
               "ssym_inverse", "fireport_hf", "megascale_test", "s571", "s4346"]
BENCH_LABEL = {"s669": "S669", "s461": "S461", "s2648": "S2648", "s783": "S783",
               "s8754": "S8754", "ssym_direct": "Ssym (dir)",
               "ssym_inverse": "Ssym (inv)", "fireport_hf": "FireProt",
               "megascale_test": "Megascale", "s571": r"S571 ($\Delta T_m$)",
               "s4346": r"S4346 ($\Delta T_m$)"}
TM = {"s571", "s4346"}

REPO = Path(".")
OUT = Path("figures")
SAVED: list[str] = []


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.edgecolor": INK, "axes.linewidth": 0.5,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.4,
        "legend.frameon": False, 
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def find(*patterns, require_cols=None, require_val=None):
    for pat in patterns:
        for p in sorted(REPO.glob(pat)):
            if not p.is_file(): continue
            if require_cols or require_val:
                rows = read(p)
                if not rows: continue
                low = {c.lower() for c in rows[0]}
                if require_cols and not {c.lower() for c in require_cols} <= low: continue
                if require_val and not any(require_val(r) for r in rows): continue
            return p
    return None

def read(p: Path):
    if p is None or not p.is_file(): return None
    try:
        with open(p, newline="") as fh: return list(csv.DictReader(fh))
    except Exception: return None

def jread(p: Path):
    if p is None or not p.is_file(): return None
    try: return json.loads(p.read_text())
    except Exception: return None

def f(x, d=float("nan")):
    try: return float(x)
    except (TypeError, ValueError): return d

def get(r, *names, default=None):
    for n in names:
        if n in r and r[n] not in ("", None): return r[n]
    low = {k.lower(): v for k, v in r.items()}
    for n in names:
        if low.get(n.lower()) not in ("", None): return low[n.lower()]
    return default

def miss(what, where, producer):
    log.info(f"    skipped: no {what} (looked for {where}; written by {producer})")

def save(fig, name, fmt, dpi):
    OUT.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in (["pdf", "png"] if fmt == "both" else [fmt]):
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        outs.append(f"{name}.{ext}")
    plt.close(fig)
    SAVED.append(name)
    log.info(f"    wrote {', '.join(outs)}")

def zeroline(ax, horizontal=False):
    (ax.axhline if horizontal else ax.axvline)(0, color=MUTED, lw=0.5, zorder=1)

def grid_on(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis)
    ax.grid(axis="y" if axis == "x" else "x", visible=False)

def panel(ax, letter):
    fn = getattr(ax, "text2D", ax.text)
    fn(-0.02, 1.05, f"({letter})", transform=ax.transAxes, fontsize=8,
       fontweight="bold", va="bottom", ha="left", color=INK)

def pooled_rows(name="bootstrap_pooled.csv"):
    p = find(f"**/{name}", require_cols=["arm", "control", "mean_delta"])
    rows = read(p) or []
    seen = {}
    for r in rows:
        if f(get(r, "mean_delta")) == f(get(r, "mean_delta")):
            seen[(get(r, "contrast", default=""), r["arm"], r["control"])] = r
    return list(seen.values()), p

def long_table():
    tab = defaultdict(lambda: defaultdict(list))
    found = []
    for pat in ("**/all_metrics_long.csv", "**/adapter_metrics_long.csv"):
        for p in sorted(REPO.glob(pat)):
            rows = read(p)
            if not rows: continue
            found.append(p.name)
            for r in rows:
                a = get(r, "arm", "name")
                b = get(r, "benchmark", "split")
                v = f(get(r, "spearman", "rho"))
                if a and b and v == v:
                    tab[a][b].append(v)
    return tab, found

def shorten(s, n=34):
    s = (s or "").strip()
    repl = {"sparse vs dense, matched": "routing vs matched dense",
            "sparse vs the smaller dense": "routing vs baseline",
            "capacity: width": "readout width",
            "router: chemistry prior": "fixed chemistry router",
            "router: hybrid prior": "hybrid chemistry prior",
            "router: drop structural input": "router without structure",
            "depth: scalar mix": "depth, shared weights",
            "depth: routed": "depth, routed per mutation",
            "fusion: FiLM": "fusion, FiLM",
            "fusion: bilinear": "fusion, bilinear",
            "pair coupling rank 2": "pair coupling, $r=2$"}
    for k, v in repl.items():
        if s.lower().startswith(k.lower()): return v
    return s if len(s) <= n else s[: n - 1] + "…"


# ------------------------------------------------------------------ fig 1
def fig_capacity(fmt, dpi):
    rows, src = pooled_rows()
    cap_p = find("**/active_capacity.csv", "**/scaling_table.csv", require_cols=["arm"])
    cap = read(cap_p)
    if not rows and not cap:
        miss("bootstrap_pooled.csv or active_capacity.csv", "results/**", "python bootstrap.py")
        return

    fig = plt.figure(figsize=(DOUBLE, max(2.6, 0.26 * len(rows) + 1.3)), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])

    if rows:
        rows.sort(key=lambda r: f(get(r, "mean_delta")))
        ax = fig.add_subplot(gs[0, 0])
        for i, r in enumerate(rows):
            m, lo, hi = (f(get(r, "mean_delta")), f(get(r, "lo")), f(get(r, "hi")))
            sig = lo > 0 or hi < 0
            col = (C["blue"] if m > 0 else C["red"]) if sig else MUTED
            ax.plot([lo, hi], [i, i], color=col, lw=1.2, solid_capstyle="round", zorder=3)
            ax.plot([m], [i], "o", color=col, ms=3.5, mec="white", mew=0.5, zorder=4)
        zeroline(ax)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([shorten(get(r, "contrast", "arm")) for r in rows], fontsize=6)
        ax.set_ylim(-0.75, len(rows) - 0.25)
        ax.set_xlabel(r"$\Delta$ Spearman vs matched control" "\n(pooled, nominal 95% $t$ interval)")
        grid_on(ax, "x")
        x0, x1 = ax.get_xlim()
        span = x1 - x0
        ax.set_xlim(x0, x1 + 0.26 * span)
        for i, r in enumerate(rows):
            lo, hi = f(get(r, "lo")), f(get(r, "hi"))
            ax.text(x1 + 0.03 * span, i,
                    f"{int(f(get(r,'same_sign',default=0)))}/"
                    f"{int(f(get(r,'n_benchmarks',default=0)))}",
                    va="center", ha="left", fontsize=5.5, color=INK,
                    fontweight="bold" if (lo > 0 or hi < 0) else "normal")
        ax.text(x1 + 0.03 * span, len(rows) - 0.2, "same sign", fontsize=5.5, color=MUTED, ha="left", va="bottom")
        ax.legend(handles=[
            Line2D([], [], color=C["blue"], marker="o", ms=3, lw=1.2, label="Improves (excludes 0)"),
            Line2D([], [], color=C["red"], marker="o", ms=3, lw=1.2, label="Degrades (excludes 0)"),
            Line2D([], [], color=MUTED, marker="o", ms=3, lw=1.2, label="Includes 0")],
            loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=1)
        panel(ax, "a")

    if cap:
        ax = fig.add_subplot(gs[0, 1])
        kact = "active_M" if "active_M" in cap[0] else "active_params_M"
        ktot = "total_M" if "total_M" in cap[0] else "head_params_M"
        dense = sorted((f(r[kact]), f(get(r, "s669")), r["arm"]) for r in cap if str(get(r, "routed", default="")).lower() == "false")
        routed = [(f(r[kact]), f(get(r, "s669")), r["arm"]) for r in cap if str(get(r, "routed", default="")).lower() == "true"]
        if dense:
            ax.plot([x for x, _, _ in dense], [y for _, y, _ in dense], "-o", color=C["red"], ms=3, zorder=4, label="Dense head (width 512–2048)")
            for x, y, a in dense:
                ax.annotate(a.replace("dense_", ""), (x, y), fontsize=5.5, color=C["red"], textcoords="offset points", xytext=(0, 5), ha="center")
        if routed:
            ax.scatter([x for x, _, _ in routed], [y for _, y, _ in routed], marker="^", s=15, color=C["blue"], zorder=3, label="Sparse routed heads")
            prim = [t for t in routed if t[2] == "experts8_top2"]
            if prim:
                x, y, _ = prim[0]
                ax.annotate("8 experts,\ntop-2", (x, y), fontsize=5.5, color=C["blue"], textcoords="offset points", xytext=(6, -2), va="center")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.minorticks_off()
        ax.margins(x=0.22, y=0.18)
        ax.set_xlabel("Readout parameters evaluated\nper mutation (M)")
        ax.set_ylabel(r"Spearman $\rho$, S669")
        grid_on(ax, "y")
        ax.legend(loc="lower left")
        panel(ax, "b")

    save(fig, "fig1_capacity", fmt, dpi)


# ------------------------------------------------------------------ fig 2
def fig_placement(fmt, dpi):
    tab, found = long_table()
    ad, _ = pooled_rows("bootstrap_adapter_pooled.csv")
    if not tab and not ad:
        miss("all_metrics_long.csv or bootstrap_adapter_pooled.csv", "results/**", "python collect.py / bootstrap.py")
        return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), layout="constrained", gridspec_kw={"width_ratios": [1.4, 1]})

    ctrl = "dense_512"
    arms = [(("layermix_resid", "dense_layermix"), "Shared weights", C["sky"]),
            (("layermoe_resid", "dense_layermoe"), "Routed per mutation", C["red"]),
            (("moe8_layermoe_resid",), "Routed depth + routed experts", C["orange"])]
    have = []
    for names, lab, col in arms:
        hit = next((n for n in names if n in tab), None)
        if hit:
            have.append((hit, lab, col))
    ax = axes[0]
    if have and ctrl in tab:
        benches = [b for b in BENCH_ORDER if b in tab[ctrl]]
        x = range(len(benches))
        w = 0.8 / len(have)
        for k, (arm, lab, col) in enumerate(have):
            ds, es = [], []
            for b in benches:
                if b in tab[arm] and b in tab[ctrl]:
                    ds.append(st.mean(tab[arm][b]) - st.mean(tab[ctrl][b]))
                    s1 = st.stdev(tab[arm][b]) if len(tab[arm][b]) > 1 else 0.0
                    n1 = max(len(tab[arm][b]), 1)
                    es.append(s1 / math.sqrt(n1))
                else:
                    ds.append(float("nan"))
                    es.append(float("nan"))
            pos = [i - 0.4 + w * (k + 0.5) for i in x]
            ax.bar(pos, ds, width=w * 0.85, color=col, label=lab, zorder=2)
            ax.errorbar(pos, ds, yerr=es, fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.0, zorder=3)
        zeroline(ax, horizontal=True)
        ax.set_xticks(list(x))
        ax.set_xticklabels([BENCH_LABEL.get(b, b) for b in benches], rotation=40, ha="right", fontsize=6)
        ax.set_ylabel(r"$\Delta$ Spearman vs dense baseline")
        grid_on(ax, "y")
        ax.legend(ncol=1, loc="lower left")
        panel(ax, "a")
    else:
        ax.axis("off")

    ax = axes[1]
    pts = []
    for r in ad:
        m = str(get(r, "arm", default="")).replace("adapt_L", "")
        if m.isdigit():
            pts.append((int(m), f(get(r, "mean_delta")), f(get(r, "lo")), f(get(r, "hi"))))
    pts.sort()
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo = [p[1] - p[2] for p in pts]
        hi = [p[3] - p[1] for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=C["blue"], ecolor=MUTED, elinewidth=0.6, capsize=1.5, ms=3.5, zorder=3)
        zeroline(ax, horizontal=True)
        ax.axvline(30, color=MUTED, lw=0.6, ls="--", zorder=1)
        ax.annotate("Inherited\nplacement", (30, 0.95), xycoords=("data", "axes fraction"), xytext=(-4, 0), textcoords="offset points", ha="right", va="top", fontsize=5.5, color=MUTED)
        ax.set_xlabel("ESM-2 layer carrying structural adapter")
        ax.set_ylabel(r"$\Delta$ Spearman vs $\ell=30$")
        ax.set_xticks([0, 4, 8, 16, 24, 28, 32])
        ax.margins(y=0.20)
        grid_on(ax, "y")
        panel(ax, "b")
    else:
        ax.axis("off")

    save(fig, "fig2_placement", fmt, dpi)


# ------------------------------------------------------------------ fig 3
def fig_router(fmt, dpi):
    part = read(find("**/tab_partition_by_seed.csv", require_cols=["mi", "null", "z"]))
    roles = read(find("**/tab_expert_roles.csv", require_cols=["expert", "burial", "ddg"]))
    if not part and not roles:
        miss("tab_partition_by_seed.csv or tab_expert_roles.csv", "paper/section6/**", "python section6.py")
        return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), layout="constrained")

    ax = axes[0]
    if part:
        x = range(len(part))
        mi = [f(get(r, "mi")) for r in part]
        nul = [f(get(r, "null")) for r in part]
        sdv = [f(get(r, "null_sd")) for r in part]
        ax.bar(x, mi, width=0.6, color=C["blue"], label="Observed", zorder=2)
        ax.bar(x, nul, width=0.6, color=C["red"], zorder=3, label="Residue-level permutation null")
        for i, (n, s) in enumerate(zip(nul, sdv)):
            ax.errorbar(i, n, yerr=3 * s, ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=4)
        for i, (m, r) in enumerate(zip(mi, part)):
            ax.annotate(f"$z$={f(get(r,'z')):.0f}", (i, m), fontsize=6, ha="center", textcoords="offset points", xytext=(0, 3))
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"Seed {get(r,'seed')}" for r in part], fontsize=6)
        ax.set_ylabel("MI with wild-type residue class (bits)")
        ax.set_ylim(0, max(mi) * 1.30)
        grid_on(ax, "y")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1)
        panel(ax, "a")
    else:
        ax.axis("off")

    ax = axes[1]
    if roles:
        ref = sorted({get(r, "seed") for r in roles})[0]
        sub = [r for r in roles if get(r, "seed") == ref]
        dmax = max(f(get(r, "ddg")) for r in sub)
        ns = [f(get(r, "n", default=1)) for r in sub]
        ordered = sorted(sub, key=lambda q: (f(get(q, "burial")), f(get(q, "ddg"))))
        for rank, r in enumerate(ordered):
            bu, dg = f(get(r, "burial")), f(get(r, "ddg"))
            n = f(get(r, "n", default=1))
            hot = dg == dmax
            ax.scatter([bu], [dg], s=10 + 40 * n / max(ns), color=C["blue"] if hot else C["sky"], edgecolors="white", linewidths=0.5, zorder=3)
            up = 1 if rank % 2 == 0 else -1
            dy = up * (5 + 4 * n / max(ns))
            ax.annotate(f"E{get(r, 'expert', default='?')}", (bu, dg), fontsize=6, ha="center", va="bottom" if up > 0 else "top", textcoords="offset points", xytext=(0, dy), fontweight="bold" if hot else "normal", color=INK if hot else MUTED)
        hotr = next(q for q in ordered if f(get(q, "ddg")) == dmax)
        ax.annotate(f"E{get(hotr,'expert',default='?')}: {get(hotr,'top_wt',default='')}", (0.97, 0.06), xycoords="axes fraction", fontsize=5.5, color=INK, ha="right", va="bottom")
        ax.set_xlabel("Mean burial of the routed positions")
        ax.set_ylabel(r"Median predicted $\Delta\Delta G$ (kcal mol$^{-1}$)")
        ax.margins(x=0.18, y=0.22)
        grid_on(ax, "both")
        ax.annotate("Marker area $\\propto$ mutations routed", (0.02, 0.96), xycoords="axes fraction", fontsize=5.5, color=MUTED, va="top")
        panel(ax, "b")
    else:
        ax.axis("off")

    save(fig, "fig3_router", fmt, dpi)


# ------------------------------------------------------------------ fig 4
def fig_controls(fmt, dpi):
    ann = read(find("**/tab_annotation_tests.csv", require_cols=["annotation", "z"]))
    dh = jread(find("**/dhps_case.json"))
    gate = read(find("**/tab_gate_vs_random.csv", require_cols=["coverage", "delta"]))
    seed = read(find("**/tab_seed_matching.csv", require_cols=["ratio"]))
    if not any((ann, dh, gate, seed)):
        miss("annotation/dhps/gate tables", "paper/**", "python annotate_structure.py, dhps_case.py, section6.py")
        return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), layout="constrained")

    items = []
    for r in ann or []:
        items.append((get(r, "annotation"), f(get(r, "z")), f(get(r, "cond_z"))))
    if dh:
        items.append(("DHPS structure block", dh["mi_block"]["z"], dh["burial_control"]["z"]))
        items.append(("DHPS SB3 site", dh["mi_sb3"]["z"], float("nan")))
    ax = axes[0]
    if items:
        y = range(len(items))
        h = 0.35
        ax.barh([i + h / 2 for i in y], [v[1] for v in items], height=h, color=C["sky"], label="Marginal", zorder=2)
        ax.barh([i - h / 2 for i in y], [0 if v[2] != v[2] else v[2] for v in items], height=h, color=C["pink"], label="After matched control", zorder=2)
        ax.axvline(2, color=C["red"], ls="--", lw=0.8, zorder=3)
        ax.annotate("$z=2$", (2, -0.80), fontsize=6, color=C["red"], textcoords="offset points", xytext=(3, 0))
        zeroline(ax)
        ax.set_yticks(list(y))
        ax.set_yticklabels([v[0] for v in items], fontsize=6)
        ax.set_ylim(-1.0, len(items) - 0.4)
        ax.set_xlabel("Standardised excess $z$ over the matched null")
        grid_on(ax, "x")
        ax.legend(loc="lower right")
        panel(ax, "a")
    else:
        ax.axis("off")

    ax = axes[1]
    if gate:
        g = sorted(gate, key=lambda r: f(get(r, "coverage")))
        cov = [f(get(r, "coverage")) for r in g]
        dl = [f(get(r, "delta")) for r in g]
        lo = [f(get(r, "delta")) - f(get(r, "lo")) for r in g]
        hi = [f(get(r, "hi")) - f(get(r, "delta")) for r in g]
        ax.errorbar(cov, dl, yerr=[lo, hi], fmt="o", color=C["blue"], ecolor=MUTED, elinewidth=0.6, capsize=1.5, ms=3.5, zorder=3)
        zeroline(ax, horizontal=True)
        ax.set_xlabel("Coverage retained")
        ax.set_ylabel(r"$\Delta$ Spearman," "\n" r"gate ranking $-$ random floor")
        ax.margins(x=0.12, y=0.28)
        grid_on(ax, "y")
        if seed:
            ratios = [f(get(r, "ratio")) for r in seed]
            ax.annotate(f"Cross-seed agreement {min(ratios):.2f}–{max(ratios):.2f}\n(1.0 = random pairing of experts)", (0.02, 0.99), xycoords="axes fraction", fontsize=6, color=MUTED, va="top", ha="left")
        panel(ax, "b")
    else:
        ax.axis("off")

    save(fig, "fig4_controls", fmt, dpi)


# ------------------------------------------------------------------ fig S1
def fig_consistency(fmt, dpi):
    pooled, _ = pooled_rows()
    per = read(find("**/bootstrap.csv", require_cols=["arm", "control", "benchmark", "delta"]))
    if not pooled or not per:
        miss("bootstrap.csv with per-benchmark rows", "results/**", "python bootstrap.py")
        return
    def rows_for(r):
        return [q for q in per if q["arm"] == r["arm"] and q["control"] == r["control"] and f(get(q, "delta")) == f(get(q, "delta"))]

    keep = [r for r in pooled if (f(get(r, "lo")) > 0 or f(get(r, "hi")) < 0) and len(rows_for(r)) >= 2]
    if not keep:
        return
    ncol = len(keep)
    fig, axes = plt.subplots(1, ncol, figsize=(min(DOUBLE, 2.4 * ncol), 2.9), squeeze=False, sharey=True, layout="constrained")
    axes = axes[0]
    union = [b for b in BENCH_ORDER if any(b in {q["benchmark"] for q in rows_for(r)} for r in keep)]
    for j, (ax, r) in enumerate(zip(axes, keep)):
        sub = rows_for(r)
        order = union
        yy = list(range(len(order)))[::-1]
        col = C["blue"] if f(get(r, "mean_delta")) > 0 else C["red"]
        for y_, b in zip(yy, order):
            q = next((x for x in sub if x["benchmark"] == b), None)
            if q is None: continue
            d, lo, hi = f(get(q, "delta")), f(get(q, "lo")), f(get(q, "hi"))
            ax.plot([lo, hi], [y_, y_], color=col, lw=1.2, solid_capstyle="round", zorder=3)
            ax.plot([d], [y_], "o", color=col, ms=3, mec="white", mew=0.5, zorder=4)
        zeroline(ax)
        ax.axvline(f(get(r, "mean_delta")), color=INK, lw=0.6, ls="--", zorder=2)
        ax.set_yticks(yy)
        if j == 0:
            ax.set_yticklabels([BENCH_LABEL.get(b, b) for b in order], fontsize=6)
        ax.set_xlabel(r"$\Delta$ Spearman")
        ax.set_title(f"{shorten(get(r,'contrast','arm'), 26)}\nPooled {f(get(r,'mean_delta')):+.4f}", fontsize=7)
        grid_on(ax, "x")
    save(fig, "figS1_consistency", fmt, dpi)


# ------------------------------------------------------------------ fig 5
def fig_dhps(fmt, dpi):
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        miss("Biopython", "pip install biopython", "pip")
        return
    import warnings
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    pdb = find("**/dhps_*.pdb", "**/1ad1*.pdb")
    rt = find("**/dhps/routing.csv", require_cols=["resnum", "expert"])
    if pdb is None or rt is None:
        miss("the DHPS structure or routing.csv", "**/dhps/**", "python dhps_case.py")
        return
    res = {}
    for r in read(rt) or []:
        n = int(f(get(r, "resnum")))
        res[n] = dict(expert=int(f(get(r, "expert"))), burial=f(get(r, "burial10", "burial")), block=get(r, "block", default=""))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = PDBParser(QUIET=True).get_structure("d", str(pdb))
    ca = {r.id[1]: r["CA"].get_coord() for r in next(iter(st[0])) if "CA" in r}
    nums = sorted(n for n in ca if n in res)
    if len(nums) < 20: return
    xyz = np.array([ca[n] for n in nums])
    experts = np.array([res[n]["expert"] for n in nums])
    burial = np.array([res[n]["burial"] for n in nums])

    live = sorted(set(experts.tolist()))
    mean_bur = {e: float(burial[experts == e].mean()) for e in live}
    core = max(mean_bur, key=mean_bur.get)

    resist = {}
    for r in read(find("**/dhps/tab_resistance.csv", require_cols=["mutation"])) or []:
        m = str(get(r, "mutation", default=""))
        d = "".join(c for c in m if c.isdigit())
        if d: resist[int(d)] = m

    fig = plt.figure(figsize=(DOUBLE, 4.5), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

    def ribbon(ax, colours, widths, title):
        segs = np.stack([xyz[:-1], xyz[1:]], axis=1)
        ax.add_collection3d(Line3DCollection(segs, colors=colours[:-1], linewidths=widths[:-1]))
        for n in resist:
            if n in ca:
                p = ca[n]
                ax.scatter([p[0]], [p[1]], [p[2]], s=15, facecolor="white", edgecolor=INK, linewidths=0.6, zorder=6)
        c = xyz.mean(0)
        rad = float(np.abs(xyz - c).max()) * 0.62
        ax.set_xlim(c[0] - rad, c[0] + rad)
        ax.set_ylim(c[1] - rad, c[1] + rad)
        ax.set_zlim(c[2] - rad, c[2] + rad)
        ax.set_axis_off()
        ax.view_init(elev=16, azim=-62)
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
        ax.set_title(title, fontsize=7)

    # Reordered mapped palette so panel A and the legend map exactly identical colors by burial depth.
    ehues = [C["blue"], C["sky"], C["green"], C["pink"], C["orange"], C["yellow"], C["red"], C["grey"]]
    by_burial = sorted(live, key=lambda e: -mean_bur[e])
    emap = {e: ehues[i % len(ehues)] for i, e in enumerate(by_burial)}

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    
    # Ensures only the core expert gets colored in panel A and the rest remain a neutral grey
    cols = [emap[core] if e == core else "#DCDCDC" for e in experts]
    wid = [2.0 if e == core else 1.0 for e in experts]
    ribbon(ax, cols, wid, f"Expert E{core} (most buried) against the rest")
    panel(ax, "a")

    lo, hi = float(burial.min()), float(burial.max())
    norm = (burial - lo) / max(hi - lo, 1e-9)
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    ribbon(ax2, plt.get_cmap("Blues")(0.12 + 0.86 * norm), [1.2 + 1.0 * v for v in norm], "Burial")
    panel(ax2, "b")
    cax = ax2.inset_axes([0.86, 0.10, 0.035, 0.42])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap="Blues", norm=matplotlib.colors.Normalize(lo, hi)), cax=cax)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.outline.set_linewidth(0.4)

    ax = fig.add_subplot(gs[1, :])
    for n in nums:
        ax.add_patch(plt.Rectangle((n - 0.5, 0.60), 1.0, 0.28, color=emap[res[n]["expert"]], lw=0))
    ax.plot(nums, 0.03 + 0.48 * norm, color=INK, lw=0.6, zorder=3)
    prev = None
    for n in nums:
        b = res[n]["block"]
        if b != prev:
            ax.axvline(n, color=MUTED, lw=0.4, ls="--", zorder=1)
            prev = b
    for k, n in enumerate(sorted(resist)):
        y = 1.00 if k % 2 == 0 else 1.11
        ax.plot([n, n], [0.90, y - 0.02], color=INK, lw=0.5)
        ax.annotate(f"{resist[n]} (E{res[n]['expert']})", (n, y), fontsize=5.5, ha="center", va="bottom", color=INK)
    ax.set_xlim(min(nums) - 1, max(nums) + 1)
    ax.set_ylim(0, 1.24)
    ax.set_yticks([0.27, 0.74])
    ax.set_yticklabels(["Burial", "Expert"], fontsize=6)
    ax.set_xlabel("Residue number (DHPS, PDB 1AD1 chain A)")
    ax.tick_params(axis="y", length=0)
    for sp in ("left", "right", "top"): ax.spines[sp].set_visible(False)
    ax.grid(False)
    ax.legend(handles=[Line2D([], [], color=emap[e], lw=2.5, label=f"E{e}") for e in by_burial],
              loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=8, handlelength=1.0, columnspacing=1.0)
    panel(ax, "c")
    save(fig, "fig5_dhps", fmt, dpi)

# --------------------------------------------------------------------------
BUILDERS = {
    "capacity":    (fig_capacity, "fig1: pooled contrasts + active parameters"),
    "placement":   (fig_placement, "fig2: encoder depth + adapter placement"),
    "router":      (fig_router, "fig3: MI per seed + expert burial map"),
    "controls":    (fig_controls, "fig4: matched controls + what does not hold"),
    "dhps":        (fig_dhps, "fig5: DHPS chain coloured by expert and by burial"),
    "consistency": (fig_consistency, "figS1: per-benchmark deltas"),
}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--format", default="both", choices=["pdf", "png", "both"])
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--traceback", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, (_fn, why) in BUILDERS.items(): log.info(f"  {k:12} {why}")
        return 0

    global REPO, OUT
    REPO = Path(a.repo).resolve()
    OUT = Path(a.out).resolve() if a.out else REPO / "paper" / "figures"
    style()
    tab, found = long_table()

    want = [x.strip() for x in a.only.split(",") if x.strip()] or list(BUILDERS)
    skipped = []
    for k in want:
        fn, why = BUILDERS[k]
        before = len(SAVED)
        try:
            fn(a.format, a.dpi)
        except Exception as exc:
            log.error(f"    FAILED: {type(exc).__name__}: {exc}")
        if len(SAVED) == before:
            skipped.append(k)
    return 0

if __name__ == "__main__":
    sys.exit(main())
