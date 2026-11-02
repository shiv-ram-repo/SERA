"""
shift_vs_length.py
==================
Asks whether the low-dimensional representation shift we measured IS the
assay-and-size gap, or something independent of it.

BACKGROUND
----------
The shift diagnostic established that the displacement between the training
split and each benchmark, in the 1408-dimensional fused representation, is
effectively rank 4: four directions carry a median 72% of it (65-84% across
benchmarks; 78% at k=16, 85% at k=64). It did not establish what those
directions encode.

Two possibilities, with opposite consequences:

  (a) They encode protein size and measurement modality -- the things the
      training assay structurally could not vary (Tsuboyama 2023: 40-72 aa,
      ~5 kcal/mol dynamic range, cooperative folding required). Then the
      directions carry SIGNAL about a regime the model never saw, projecting
      them out would delete information rather than nuisance, and the fix has
      to happen in the supervision.

  (b) They encode something incidental. Then a readout-level intervention is
      worth trying.

WHAT IT MEASURES
----------------
1.  R^2 of protein length regressed on the shift coordinates, WITHIN each
    benchmark. Within one benchmark the assay is constant, so any length
    signal along the shift direction is about size alone. A high R^2 means the
    direction is partly a length axis.

2.  The same for |ddG|, as a check that the direction is not simply encoding
    the label.

3.  The same-assay control: megascale_test against megascale_train. Same
    proteolysis assay, same 40-72 aa regime, held-out proteins. Its
    displacement along the BENCHMARK shift directions is the amount of shift
    attributable to ordinary held-out generalization rather than to the change
    of assay. If the control is small and the benchmarks are large, the gap is
    between assays.

4.  Pairwise subspace overlap between benchmarks. If all benchmarks shift
    along nearly the same four directions, there is one shared axis to name;
    if each shifts its own way, there is no single mechanism.

5.  Protein-disjoint estimability: the basis computed from one half of the
    benchmark proteins against the basis from the other half. This is the
    number that decides whether such a basis could ever be estimated without
    touching a test set -- without it, no method built on this is leakage-free.

    python shift_vs_length.py
    python shift_vs_length.py --k 4 --features /backup/new_work/v2/results/encoder_features
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("shift")

REFERENCE = "megascale_train"
CONTROL = "megascale_test"


def load_features(d: Path) -> dict[str, np.ndarray]:
    out = {}
    for p in sorted(d.glob("*.npy")):
        name = p.stem
        for suf in ("_features", "_feats", "_encoder"):
            if name.endswith(suf):
                name = name[: -len(suf)]
        try:
            X = np.load(p)
        except Exception as exc:                                  # noqa: BLE001
            log.warning(f"  {p.name}: {exc}")
            continue
        if X.ndim != 2:
            continue
        out[name] = X.astype(np.float64)
    return out


def shift_basis(Xr: np.ndarray, Xb: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvectors of the covariance difference (same construction as
    the shift diagnostic, so the numbers are comparable)."""
    D = np.cov(Xr, rowvar=False) - np.cov(Xb, rowvar=False)
    D = 0.5 * (D + D.T)
    w, V = np.linalg.eigh(D)
    return V[:, np.argsort(-np.abs(w))[:k]]


def r2_multi(y: np.ndarray, Z: np.ndarray) -> float:
    """R^2 of y regressed on the columns of Z, with an intercept."""
    ok = np.isfinite(y) & np.isfinite(Z).all(axis=1)
    if ok.sum() < 20:
        return float("nan")
    y, Z = y[ok], Z[ok]
    A = np.column_stack([np.ones(len(Z)), Z])
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    resid = y - A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - (resid ** 2).sum() / ss_tot)


def subspace_overlap(U: np.ndarray, V: np.ndarray) -> float:
    """Mean squared canonical correlation between two orthonormal bases:
    1.0 = identical subspace, 0.0 = orthogonal. Equals
    ||U^T V||_F^2 / k for orthonormal U, V of the same width."""
    k = min(U.shape[1], V.shape[1])
    M = U[:, :k].T @ V[:, :k]
    return float((M ** 2).sum() / k)


def subsample(X, n, rng):
    return X[rng.choice(len(X), n, replace=False)] if len(X) > n else X


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features",
                    default="/backup/new_work/v2/results/encoder_features")
    ap.add_argument("--data-root",
                    default=os.environ.get("DDG_DATA_ROOT",
                                           "/backup/new_work/data_root"))
    ap.add_argument("--meta-dir", default=None,
                    help="default: {data-root}/metadata")
    ap.add_argument("--run-dir", action="append", default=[],
                    help="directory holding targets_*.npy; repeatable")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cap", type=int, default=4000,
                    help="max rows used per set when estimating covariances")
    ap.add_argument("--out", default="shift_vs_length.json")
    a = ap.parse_args()

    fd = Path(a.features)
    if not fd.is_dir():
        log.error(f"{fd} not found")
        return 2
    meta_dir = Path(a.meta_dir) if a.meta_dir else Path(a.data_root) / "metadata"

    F = load_features(fd)
    if REFERENCE not in F:
        log.error(f"no '{REFERENCE}' among {sorted(F)}")
        return 2
    Xr_full = F[REFERENCE]
    D = Xr_full.shape[1]
    rng = np.random.default_rng(0)
    Xr = subsample(Xr_full, a.cap, rng)

    log.info(f"{len(F)} feature sets, dim {D}, k = {a.k}")
    for n, X in sorted(F.items()):
        log.info(f"  {n:18} {X.shape}")

    report = {"dim": D, "k": a.k, "benchmarks": {}}
    bases: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- 1,2,3
    log.info("")
    log.info("=" * 78)
    log.info("1-2. WHAT THE SHIFT DIRECTIONS ENCODE, WITHIN EACH BENCHMARK")
    log.info("=" * 78)
    log.info("  R^2 of length (and |ddG|) regressed on the shift coordinates.")
    log.info("  Within one benchmark the assay is fixed, so length signal here "
             "is size alone.")
    log.info("")
    log.info(f"  {'benchmark':16} {'n':>6} {'|shift|':>8} "
             f"{'R2 length':>10} {'R2 |ddG|':>9}")

    from_assay_gap = []
    for name, Xb_full in sorted(F.items()):
        if name == REFERENCE:
            continue
        Xb = subsample(Xb_full, a.cap, rng)
        try:
            V = shift_basis(Xr, Xb, a.k)
        except np.linalg.LinAlgError:
            log.warning(f"  {name}: eigendecomposition failed")
            continue
        bases[name] = V
        dmu = Xb.mean(0) - Xr.mean(0)
        snorm = float(np.linalg.norm(dmu))

        # coordinates of every mutation of this benchmark along the basis
        Z = Xb_full @ V                                   # (N, k)

        r2_len, meta = float("nan"), None
        # extract_features.py writes {name}_meta.csv beside the features;
        # make_length_metadata.py writes it under {data_root}/metadata.
        for mp in (fd / f"{name}_meta.csv", meta_dir / f"{name}_meta.csv"):
            if not mp.is_file():
                continue
            try:
                m = pd.read_csv(mp)
            except Exception as exc:                              # noqa: BLE001
                log.warning(f"  {name}: {mp} unreadable: {exc}")
                continue
            if len(m) != len(Xb_full):
                log.warning(f"  {name}: {mp.name} has {len(m)} rows but "
                            f"features have {len(Xb_full)} -- NOT aligned, "
                            f"skipped")
                continue
            meta = m
            if "length" in meta.columns:
                r2_len = r2_multi(
                    pd.to_numeric(meta["length"], errors="coerce").to_numpy(), Z)
            break

        r2_ddg = float("nan")
        # Preferred source: the `target` column extract_features.py writes
        # beside the features, which is aligned by construction.
        if meta is not None and "target" in meta.columns \
                and len(meta) == len(Xb_full):
            r2_ddg = r2_multi(
                np.abs(pd.to_numeric(meta["target"], errors="coerce").to_numpy()), Z)
        if not np.isfinite(r2_ddg):
            for rd in [Path(r) for r in a.run_dir]:
                for cand in (rd / "eval_noTTA" / f"targets_{name}.npy",
                             rd / f"targets_{name}.npy"):
                    if cand.is_file():
                        t = np.load(cand).astype(np.float64).reshape(-1)
                        if len(t) == len(Xb_full):
                            r2_ddg = r2_multi(np.abs(t), Z)
                        break
                if np.isfinite(r2_ddg):
                    break

        report["benchmarks"][name] = {
            "shift_norm": snorm, "r2_length": r2_len, "r2_absddg": r2_ddg}
        if name != CONTROL and np.isfinite(r2_len):
            from_assay_gap.append(r2_len)
        log.info(f"  {name:16} {len(Xb_full):>6} {snorm:>8.3f} "
                 f"{r2_len:>10.3f} {r2_ddg:>9.3f}"
                 + ("   <- CONTROL" if name == CONTROL else ""))

    # ---------------------------------------------------------------- 3
    log.info("")
    log.info("=" * 78)
    log.info("3. SAME-ASSAY CONTROL")
    log.info("=" * 78)
    if CONTROL in report["benchmarks"]:
        c = report["benchmarks"][CONTROL]["shift_norm"]
        others = [v["shift_norm"] for n, v in report["benchmarks"].items()
                  if n != CONTROL]
        if others:
            med = float(np.median(others))
            log.info(f"  {CONTROL:16} |shift| {c:.3f}   (same assay, same "
                     f"40-72 aa regime, held-out proteins)")
            log.info(f"  benchmarks       |shift| {med:.3f}   (median)")
            log.info(f"  ratio {med / max(c, 1e-9):.2f}x")
            report["control_shift"] = c
            report["benchmark_shift_median"] = med
            report["shift_ratio"] = med / max(c, 1e-9)
            if med > 2 * c:
                log.info("")
                log.info("  Ordinary held-out generalization moves the "
                         "representation much less than")
                log.info("  the benchmarks do. The bulk of the shift is "
                         "between assays, not between")
                log.info("  proteins -- which is outside anything the readout "
                         "can reach.")
    else:
        log.info(f"  no '{CONTROL}' features -- extract them to get this "
                 f"control, it is the")
        log.info("  single most informative number in this script")

    # ---------------------------------------------------------------- 4
    log.info("")
    log.info("=" * 78)
    log.info("4. IS IT ONE SHARED AXIS?")
    log.info("=" * 78)
    names = [n for n in sorted(bases) if n != CONTROL]
    if len(names) > 1:
        ov = {}
        log.info("  pairwise subspace overlap (1 = same directions, 0 = "
                 "orthogonal)")
        for i, n1 in enumerate(names):
            cells = []
            for n2 in names:
                o = subspace_overlap(bases[n1], bases[n2])
                ov[f"{n1}|{n2}"] = o
                cells.append(f"{o:.2f}")
            log.info(f"  {n1:16} " + " ".join(f"{c:>5}" for c in cells))
        log.info("  " + " " * 16 + " " + " ".join(f"{n[:5]:>5}" for n in names))
        off = [ov[f"{n1}|{n2}"] for i, n1 in enumerate(names)
               for n2 in names[i + 1:]]
        report["overlap_offdiag_median"] = float(np.median(off)) if off else None
        if off:
            log.info("")
            log.info(f"  median off-diagonal overlap {np.median(off):.2f}")
            if np.median(off) > 0.5:
                log.info("  The benchmarks shift along largely the SAME few "
                         "directions. One mechanism,")
                log.info("  one thing to name.")
            else:
                log.info("  The benchmarks shift along DIFFERENT directions. "
                         "There is no single axis,")
                log.info("  so a one-subspace fix cannot address all of them.")

    # ---------------------------------------------------------------- 5
    log.info("")
    log.info("=" * 78)
    log.info("5. COULD THIS BASIS BE ESTIMATED WITHOUT TOUCHING A TEST SET?")
    log.info("=" * 78)
    log.info("  Split each benchmark's rows in half and compare the two bases.")
    log.info("  Low overlap means the basis is not stably estimable at this "
             "sample size,")
    log.info("  and nothing built on it would be leakage-free.")
    log.info("")
    split_ov, mean_cos = {}, {}
    for name, Xb_full in sorted(F.items()):
        if name == REFERENCE or len(Xb_full) < 200:
            continue
        idx = rng.permutation(len(Xb_full))
        h = len(idx) // 2
        try:
            Va = shift_basis(Xr, subsample(Xb_full[idx[:h]], a.cap, rng), a.k)
            Vb = shift_basis(Xr, subsample(Xb_full[idx[h:]], a.cap, rng), a.k)
        except np.linalg.LinAlgError:
            continue
        o = subspace_overlap(Va, Vb)
        # The 72% figure was about the MEAN displacement, so measure that
        # direction's stability too: a covariance basis can be unstable while
        # the mean shift direction is perfectly reproducible, and the two
        # support different methods.
        da = Xb_full[idx[:h]].mean(0) - Xr.mean(0)
        db = Xb_full[idx[h:]].mean(0) - Xr.mean(0)
        cos = float(da @ db / max(np.linalg.norm(da) * np.linalg.norm(db), 1e-12))
        split_ov[name] = o
        mean_cos[name] = cos
        log.info(f"  {name:16} cov-basis overlap {o:.3f}   "
                 f"mean-direction cosine {cos:.3f}")
    report["split_half_overlap"] = split_ov
    report["split_half_mean_cosine"] = mean_cos
    if split_ov:
        m = float(np.median(list(split_ov.values())))
        mc = float(np.median(list(mean_cos.values()))) if mean_cos else float("nan")
        report["split_half_overlap_median"] = m
        report["split_half_mean_cosine_median"] = mc
        log.info("")
        log.info(f"  median cov-basis overlap {m:.3f}, "
                 f"median mean-direction cosine {mc:.3f}")
        if m < 0.5 and mc > 0.8:
            log.info("  The covariance basis is unstable but the MEAN shift "
                     "direction reproduces. So the")
            log.info("  shift is a displacement, not a change of shape: a "
                     "subspace-projection method is")
            log.info("  not supported, but a mean-offset correction would be.")
        elif m < 0.5:
            log.info("  Neither the basis nor the mean direction is stable at "
                     "this sample size. Nothing")
            log.info("  built on this subspace would be leakage-free.")

    # ---------------------------------------------------------------- verdict
    log.info("")
    log.info("=" * 78)
    log.info("WHAT THIS DECIDES")
    log.info("=" * 78)
    if from_assay_gap:
        m = float(np.median(from_assay_gap))
        report["r2_length_median"] = m
        log.info(f"  protein length explains a median R^2 of {m:.3f} of "
                 f"position along the shift")
        log.info(f"  directions, within benchmarks where the assay is fixed.")
        if m > 0.2:
            log.info("  The directions are substantially a size axis. Size is "
                     "what the training assay")
            log.info("  structurally could not vary, so this is a supervision "
                     "problem and the readout")
            log.info("  cannot reach it.")
        else:
            log.info("  Length does not explain much of these directions. The "
                     "size story is not the")
            log.info("  whole explanation -- look at the assay control and the "
                     "label distribution from")
            log.info("  assay_gap.py before concluding.")
    else:
        log.info("  no aligned length metadata -- run "
                 "analysis/make_length_metadata.py, then re-run this")

    Path(a.out).write_text(json.dumps(report, indent=1, default=float))
    log.info("")
    log.info(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
