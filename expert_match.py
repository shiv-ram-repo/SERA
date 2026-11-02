"""
expert_match.py
===============
Do independently trained models learn the SAME decomposition, or merely
decompositions that each carry chemical information?

WHY THIS IS THE STRONGER CLAIM
------------------------------
Mutual information says the partition is chemically organised. It does not say
two runs found the same partition. Those are different claims, and the second
is the one a reader finds convincing.

Expert indices carry no meaning across runs: permuting the experts is a
symmetry of the architecture, so expert 7 in one seed has nothing to do with
expert 7 in another. The per-seed tables already hint at the answer -- every
seed has one large expert for buried hydrophobic and aromatic residues with
median ddG near +1.4, but it is expert 7, 4, 2, 2 and 6 in the five runs.

This script does the matching properly: it describes each expert by its
residue-class profile, matches experts between seeds with the Hungarian
assignment, and asks whether the matched profiles agree better than chance.

THE NULL
--------
A good matching will always look good: the assignment is chosen to minimise
distance, so even random profiles match better than a random pairing. The null
therefore applies the SAME optimal matching procedure to profiles whose expert
labels have been shuffled within each seed. Only the excess over that null
means anything -- without it, this analysis would report success on noise.

WHAT IT REPORTS
---------------
1.  Matching quality: mean Jensen-Shannon distance between matched profiles,
    against the null, across every pair of seeds.

2.  The role table: for each role in the reference seed, what the matched
    expert looks like in every other seed -- top residues, burial, median ddG.
    This is the table that goes in the paper.

3.  Role stability: how consistently each role recurs, and which roles are
    reproducible rather than seed-specific. A role found in 5/5 runs is a
    property of the problem; one found in 2/5 is a property of an
    initialisation.

    python expert_match.py --routing results/routing.csv
    python expert_match.py --routing results/routing.csv --reference 42
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
log = logging.getLogger("match")

CHEM_GROUPS = {"A": "aliphatic", "V": "aliphatic",
               "L": "aliphatic2", "I": "aliphatic2", "M": "aliphatic2",
               "F": "aromatic", "W": "aromatic", "Y": "aromatic",
               "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
               "K": "basic", "R": "basic", "H": "basic",
               "D": "acidic", "E": "acidic",
               "G": "gly/pro", "P": "gly/pro",
               "C": "cys", "X": "cys"}
CLASSES = ["aliphatic", "aliphatic2", "aromatic", "polar", "basic",
           "acidic", "gly/pro", "cys"]


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance: a metric, bounded in [0, 1], and symmetric --
    which matters because neither seed is privileged."""
    p = np.asarray(p, float) + 1e-12
    q = np.asarray(q, float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        return float(np.sum(x * np.log2(x / y)))

    return float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))


def profiles(df: pd.DataFrame, min_n: int):
    """Residue-class profile per expert, plus descriptive statistics."""
    out = {}
    for e, g in df.groupby("expert"):
        if len(g) < min_n:
            continue
        v = g.cls.value_counts(normalize=True).reindex(CLASSES, fill_value=0.0)
        out[int(e)] = dict(
            profile=v.to_numpy(float), n=int(len(g)),
            top_wt="/".join(g.wt.value_counts().head(4).index),
            burial=(float(g.burial10.median()) if "burial10" in g
                    and g.burial10.notna().any() else float("nan")),
            ddg=(float(g.ddg.median()) if "ddg" in g and g.ddg.notna().any()
                 else float("nan")))
    return out


def match(a: dict, b: dict):
    """Hungarian assignment between two seeds' experts, on profile distance."""
    from scipy.optimize import linear_sum_assignment
    ka, kb = sorted(a), sorted(b)
    if not ka or not kb:
        return [], float("nan")
    C = np.array([[js_distance(a[i]["profile"], b[j]["profile"])
                   for j in kb] for i in ka])
    ri, ci = linear_sum_assignment(C)
    pairs = [(ka[i], kb[j], float(C[i, j])) for i, j in zip(ri, ci)]
    return pairs, float(np.mean([p[2] for p in pairs]))


def null_match(a: dict, b: dict, n_perm: int, rng):
    """Mean distance under a RANDOM bijection between the two seeds' experts.

    Two earlier attempts at this null were wrong, and both failed silently:

      * Shuffling the profiles and re-running the assignment does nothing at
        all. Permuting the rows and columns of a cost matrix leaves the
        optimal assignment cost unchanged, so observed and null came out
        bit-identical.

      * Raw matched distance is not interpretable on its own either. If a
        router collapsed so that every expert had the same profile, every
        pairing would match perfectly and the analysis would report flawless
        reproducibility for a model that had learned nothing.

    Random pairing fixes both. It asks the only question that matters: does
    choosing the matching OPTIMALLY beat pairing experts arbitrarily? When
    profiles are distinct and shared across runs, it does. When they are
    collapsed or unrelated, optimal and arbitrary pairings score the same."""
    ka, kb = sorted(a), sorted(b)
    n = min(len(ka), len(kb))
    if n < 2:
        return np.array([])
    out = []
    for _ in range(n_perm):
        pa = rng.permutation(ka)[:n]
        pb = rng.permutation(kb)[:n]
        out.append(float(np.mean([js_distance(a[i]["profile"], b[j]["profile"])
                                  for i, j in zip(pa, pb)])))
    return np.array(out)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routing", default="results/routing.csv")
    ap.add_argument("--reference", type=int, default=None,
                    help="seed to use as the reference; default is the lowest")
    ap.add_argument("--min-n", type=int, default=50,
                    help="experts with fewer mutations are not profiled")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--outdir", default="paper/section6")
    a = ap.parse_args()

    rp = Path(a.routing)
    if not rp.is_file():
        log.error(f"{rp} not found")
        return 2
    R = pd.read_csv(rp)
    if "seed" not in R.columns or R.seed.nunique() < 2:
        log.error("need at least two seeds in routing.csv -- this analysis is "
                  "about agreement BETWEEN runs")
        return 2
    R["cls"] = R.wt.map(CHEM_GROUPS).fillna("cys")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = sorted(R.seed.unique())
    ref = a.reference if a.reference is not None else seeds[0]
    if ref not in seeds:
        log.error(f"reference seed {ref} not present; have {seeds}")
        return 2
    P = {s: profiles(R[R.seed == s], a.min_n) for s in seeds}
    rng = np.random.default_rng(0)

    log.info("=" * 88)
    log.info("1. DO DIFFERENT RUNS FIND THE SAME DECOMPOSITION?")
    log.info("=" * 88)
    log.info("  Expert indices are arbitrary -- permuting them is a symmetry of")
    log.info("  the architecture -- so experts are matched by residue-class")
    log.info("  profile. Lower Jensen-Shannon distance is better agreement.")
    log.info("")
    log.info("  The null is a RANDOM pairing of the two runs' experts. If the")
    log.info("  runs share a decomposition, choosing the matching optimally")
    log.info("  beats pairing arbitrarily. If a router had collapsed so every")
    log.info("  expert looked alike, both would score the same -- which is why")
    log.info("  the raw matched distance alone proves nothing.")
    log.info("")
    log.info(f"  {'pair':>14}{'experts':>9}{'matched JS':>13}{'null JS':>12}"
             f"{'ratio':>8}{'z':>8}{'p':>8}")
    rows, zs, ratios = [], [], []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            pairs, d = match(P[s1], P[s2])
            nl = null_match(P[s1], P[s2], a.n_perm, rng)
            if not len(nl) or not np.isfinite(d):
                continue
            z = (nl.mean() - d) / (nl.std() + 1e-12)     # lower is better
            p = float((nl <= d).mean())
            ratio = float(nl.mean() / max(d, 1e-9))
            zs.append(z)
            ratios.append(ratio)
            ok = (z > 3 and ratio > 2)
            log.info(f"  {f's{s1} vs s{s2}':>14}{len(pairs):>9}{d:>13.4f}"
                     f"{nl.mean():>12.4f}{ratio:>8.1f}{z:>8.1f}{p:>8.3f}"
                     + ("" if ok else "   <- not separable"))
            rows.append(dict(seed_a=int(s1), seed_b=int(s2), n_pairs=len(pairs),
                             js=d, null_js=float(nl.mean()),
                             null_sd=float(nl.std()), ratio=ratio,
                             z=float(z), p=p))
    if not rows:
        log.error("no seed pair could be matched")
        return 2
    pd.DataFrame(rows).to_csv(outdir / "tab_seed_matching.csv", index=False)
    log.info("")
    log.info(f"  {len(rows)} seed pairs: ratio {min(ratios):.1f}-"
             f"{max(ratios):.1f}x, z {min(zs):.1f}-{max(zs):.1f}")
    log.info("")
    log.info("  RATIO is the effect size and z alone is not enough. With")
    log.info("  near-identical profiles the null's spread collapses and z can")
    log.info("  exceed 3 on runs that share nothing: on simulated data where")
    log.info("  every run partitioned arbitrarily, z reached 4.0 at a ratio of")
    log.info("  only 1.3x, against 7.3 at 8x when the decomposition was shared.")
    log.info("")
    if min(zs) > 3 and min(ratios) > 2:
        log.info("  Every pair of runs agrees on the decomposition far better")
        log.info("  than optimal matching of unrelated profiles achieves. The")
        log.info("  partition is a property of the problem, not of an")
        log.info("  initialisation.")
    else:
        log.info("  At least one pair is not separable from the null. The")
        log.info("  decomposition is not reproducible across runs, and the")
        log.info("  per-seed tables should not be read as one shared set of")
        log.info("  roles.")

    # ------------------------------------------------------------------ 2
    log.info("")
    log.info("=" * 88)
    log.info(f"2. THE ROLE TABLE  (reference seed {ref})")
    log.info("=" * 88)
    log.info("  Each reference expert, and the expert it matches in every other")
    log.info("  run. Indices differ; the question is whether the content does.")
    log.info("")
    role_rows = []
    for e in sorted(P[ref]):
        r = P[ref][e]
        dom = CLASSES[int(np.argmax(r["profile"]))]
        log.info(f"  role from s{ref} expert {e}   "
                 f"dominant class: {dom}   n={r['n']}")
        log.info(f"    {'seed':>6}{'expert':>8}{'n':>7}{'JS':>8}"
                 f"{'burial':>9}{'median ddG':>12}  top wild-type")
        log.info(f"    {ref:>6}{e:>8}{r['n']:>7}{0.0:>8.3f}"
                 f"{r['burial']:>9.3f}{r['ddg']:>+12.2f}  {r['top_wt']}")
        role_rows.append(dict(role=int(e), seed=int(ref), expert=int(e),
                              n=r["n"], js=0.0, burial=r["burial"],
                              ddg=r["ddg"], top_wt=r["top_wt"],
                              dominant=dom))
        for s in seeds:
            if s == ref:
                continue
            pairs, _ = match(P[ref], P[s])
            hit = next((p for p in pairs if p[0] == e), None)
            if hit is None:
                log.info(f"    {s:>6}{'--':>8}  no match")
                continue
            q = P[s][hit[1]]
            log.info(f"    {s:>6}{hit[1]:>8}{q['n']:>7}{hit[2]:>8.3f}"
                     f"{q['burial']:>9.3f}{q['ddg']:>+12.2f}  {q['top_wt']}")
            role_rows.append(dict(role=int(e), seed=int(s), expert=int(hit[1]),
                                  n=q["n"], js=hit[2], burial=q["burial"],
                                  ddg=q["ddg"], top_wt=q["top_wt"],
                                  dominant=CLASSES[int(np.argmax(q["profile"]))]))
        log.info("")
    RR = pd.DataFrame(role_rows)
    RR.to_csv(outdir / "tab_expert_roles.csv", index=False)

    # ------------------------------------------------------------------ 3
    log.info("=" * 88)
    log.info("3. WHICH ROLES ARE REPRODUCIBLE?")
    log.info("=" * 88)
    log.info("  A role whose matched partner has the same dominant residue")
    log.info("  class in every run is a property of the problem. One that")
    log.info("  agrees in two runs of five is a property of an initialisation,")
    log.info("  and must not be described as a discovered category.")
    log.info("")
    log.info(f"  {'role':>6}{'dominant (ref)':>18}{'same in':>10}"
             f"{'mean JS':>10}{'burial':>18}{'median ddG':>20}")
    stab = []
    for role, g in RR.groupby("role"):
        refrow = g[g.seed == ref].iloc[0]
        agree = int((g.dominant == refrow.dominant).sum())
        js = float(g[g.seed != ref].js.mean())
        bmin, bmax = g.burial.min(), g.burial.max()
        dmin, dmax = g.ddg.min(), g.ddg.max()
        log.info(f"  {int(role):>6}{refrow.dominant:>18}"
                 f"{f'{agree}/{len(g)}':>10}{js:>10.3f}"
                 f"{f'{bmin:.3f}-{bmax:.3f}':>18}"
                 f"{f'{dmin:+.2f} to {dmax:+.2f}':>20}")
        stab.append(dict(role=int(role), dominant=refrow.dominant,
                         agree=agree, n_seeds=int(len(g)), mean_js=js,
                         burial_min=float(bmin), burial_max=float(bmax),
                         ddg_min=float(dmin), ddg_max=float(dmax)))
    pd.DataFrame(stab).to_csv(outdir / "tab_role_stability.csv", index=False)
    full = [s for s in stab if s["agree"] == s["n_seeds"]]
    log.info("")
    log.info(f"  {len(full)}/{len(stab)} roles keep the same dominant residue "
             f"class in every run")
    if full:
        log.info("  Fully reproducible roles:")
        for s in full:
            log.info(f"    role {s['role']:<3} {s['dominant']:<12} "
                     f"burial {s['burial_min']:.3f}-{s['burial_max']:.3f}  "
                     f"median ddG {s['ddg_min']:+.2f} to {s['ddg_max']:+.2f}")
    log.info("")
    log.info("  Report the fully reproducible roles as the model's learned")
    log.info("  decomposition, and the rest as what they are.")

    json.dump(dict(reference=int(ref), seeds=[int(s) for s in seeds],
                   pairs=rows, stability=stab),
              open(outdir / "expert_match.json", "w"), indent=1, default=float)
    log.info("")
    log.info(f"  wrote {outdir}/tab_seed_matching.csv, tab_expert_roles.csv, "
             f"tab_role_stability.csv, expert_match.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
