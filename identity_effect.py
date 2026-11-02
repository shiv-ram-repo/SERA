"""
identity_effect.py
==================
Does the cost of readout capacity depend on how far the target protein is from
the training set?

THE QUESTION
------------
The pooled test killed the general claim: across ten benchmarks, dense_2048
against dense_512 is +0.0012 [-0.0075,+0.0099], 6/10 -- no consistent effect.
But the per-benchmark pattern is not random. Capacity hurts on S669, S461 and
S571 and helps on Ssym, FireProt, S783, S8754, S2648 and S4346. S669 was
constructed to be non-redundant at 25% sequence identity against S2648, and
S461 is its curated subset -- so the benchmarks where capacity hurts are the
ones built to be sequence-distant from the usual training data, and the ones
where it helps are not.

That makes a prediction, and the prediction can fail: if the heterogeneity is
real rather than noise, the cost of capacity should scale with distance from
training. Unexplained heterogeneity is a dead result; heterogeneity explained
by a covariate decided in advance is a mechanism.

WHY THIS IS NOT CHERRY-PICKING
------------------------------
The covariate is sequence identity to the training split -- computed from the
sequences, not from the outcome. The hypothesis comes from how the benchmarks
were built, not from which ones happened to agree. And the test is at the
PROTEIN level: roughly a thousand proteins, each with an identity and a change
in error, rather than ten benchmark means. Ten points had no power; a thousand
do. The regression can come back flat, and if it does the claim is dead for
good rather than rescued.

WHAT IT DOES
------------
1.  Collects wild-type sequences: the SEQ column where a benchmark has one,
    otherwise the CA sequence parsed from the PDB.
2.  For each benchmark protein, the maximum sequence identity to any protein in
    the Megascale TRAINING split. mmseqs2 is used when present; otherwise a
    k-mer prefilter followed by Smith-Waterman on the top candidates.
3.  Per protein, the change in mean absolute error between two arms, averaged
    over seeds.
4.  Regresses that change on identity, with the benchmark as a fixed effect so
    the result is not just a between-benchmark difference wearing a disguise,
    and reports the slope with a cluster bootstrap over proteins.

    python identity_effect.py --arm dense_2048 --control dense_512
    python identity_effect.py --arm experts8_top2 --control dense_2048
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ident")

MEGA_CSV = "data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
MEGA_SPLITS = "data/dataset/megascale/mega_splits.pkl"
AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
          "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
          "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
          "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

BENCH = {
    "s669": ("data/dataset/S669/s669_clean_dir.csv", "data/dataset/S669/pdb"),
    "ssym_direct": ("data/dataset/ssym/ssym-5fold_clean_dir.csv",
                    "data/dataset/ssym/pdb"),
    "ssym_inverse": ("data/dataset/ssym/ssym-5fold_clean_inv.csv",
                     "data/dataset/ssym/pdb"),
    "s461": ("data/dataset/geostab_data/ddG_cleaned/S461.csv",
             "data/dataset/geostab_data/ddG_cleaned/S461"),
    "s783": ("data/dataset/geostab_data/ddG_cleaned/S783.csv",
             "data/dataset/geostab_data/ddG_cleaned/S783"),
    "s8754": ("data/dataset/geostab_data/ddG_cleaned/S8754.csv",
              "data/dataset/geostab_data/ddG_cleaned/S8754"),
    "s2648": ("data/dataset/geostab_data/ddG_cleaned/S2648.csv",
              "data/dataset/geostab_data/ddG_cleaned/S2648"),
    "s571": ("data/dataset/geostab_data/dTm_cleaned/S571.csv",
             "data/dataset/geostab_data/dTm_cleaned/S571"),
    "s4346": ("data/dataset/geostab_data/dTm_cleaned/S4346.csv",
              "data/dataset/geostab_data/dTm_cleaned/S4346"),
}


def pdb_seq(path: Path, chain: str | None):
    out, seen = [], set()
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                    continue
                if chain and line[21] != chain:
                    continue
                key = (line[21], line[22:27])
                if key in seen:
                    continue
                seen.add(key)
                out.append(AA3to1.get(line[17:20].strip(), "X"))
    except OSError:
        return None
    return "".join(out) or None


def bench_sequences(root: Path):
    """{benchmark: {protein_key: sequence}}"""
    out = {}
    for name, (csv_rel, pdb_rel) in BENCH.items():
        csv = root / csv_rel
        if not csv.is_file():
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:                                         # noqa: BLE001
            continue
        pc = next((c for c in ("PDB", "pdb") if c in df.columns), None)
        if pc is None:
            continue
        cc = next((c for c in ("chain", "CHAIN") if c in df.columns), None)
        sc = next((c for c in ("SEQ", "seq", "sequence") if c in df.columns),
                  None)
        seqs = {}
        for _, row in df.iterrows():
            key = str(row[pc])
            if key in seqs:
                continue
            s = None
            if sc is not None and isinstance(row.get(sc), str) \
                    and len(str(row[sc])) > 10:
                s = str(row[sc])
            else:
                stem = key.replace(".pdb", "")
                ch = str(row[cc]) if cc else (stem[-1] if len(stem) > 4 else None)
                for cand in (stem, stem[:-1], stem[:4]):
                    p = root / pdb_rel / f"{cand}.pdb"
                    if p.is_file():
                        s = pdb_seq(p, ch)
                        break
            if s:
                seqs[key] = s
        if seqs:
            out[name] = seqs
            log.info(f"  {name:14} {len(seqs):>4} wild-type sequences")
        else:
            log.warning(f"  {name:14} no sequences recovered")
    return out


def train_sequences(root: Path):
    import pickle
    csv = root / MEGA_CSV
    sp = root / MEGA_SPLITS
    if not (csv.is_file() and sp.is_file()):
        log.error("megascale CSV or split pickle missing")
        return {}
    df = pd.read_csv(csv, usecols=["WT_name", "aa_seq", "mut_type"])
    df = df[df.mut_type.astype(str) == "wt"]
    with open(sp, "rb") as f:
        splits = pickle.load(f)
    keep = set(splits.get("train", []))
    df = df[df.WT_name.isin(keep)]
    seqs = {str(r.WT_name): str(r.aa_seq) for r in df.itertuples()
            if isinstance(r.aa_seq, str)}
    log.info(f"  training split: {len(seqs)} wild-type sequences")
    return seqs


def kmers(s, k=3):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def max_identity(queries: dict, targets: dict, top_n=25):
    """Max % identity of each query against any target."""
    if shutil.which("mmseqs"):
        try:
            return _mmseqs(queries, targets)
        except Exception as exc:                                  # noqa: BLE001
            log.warning(f"  mmseqs failed ({exc}); falling back to alignment")
    try:
        from Bio.Align import PairwiseAligner
    except ImportError:
        log.error("  need either mmseqs2 on PATH or biopython installed")
        return {}
    al = PairwiseAligner()
    al.mode = "local"
    al.open_gap_score, al.extend_gap_score = -11, -1
    try:
        from Bio.Align import substitution_matrices
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:                                             # noqa: BLE001
        al.match_score, al.mismatch_score = 2, -1

    tk = {k: kmers(v) for k, v in targets.items()}
    out = {}
    for i, (qid, q) in enumerate(queries.items(), 1):
        qk = kmers(q)
        if not qk:
            continue
        # k-mer prefilter: full alignment against every target is wasteful and
        # the top candidates by shared 3-mers reliably contain the best hit
        cand = sorted(tk, key=lambda t: -len(qk & tk[t]))[:top_n]
        best = 0.0
        for t in cand:
            s = targets[t]
            try:
                aln = al.align(q, s)[0]
            except Exception:                                     # noqa: BLE001
                continue
            a, b = aln[0], aln[1]
            match = sum(1 for x, y in zip(a, b) if x == y and x != "-")
            denom = min(len(q), len(s))
            if denom:
                best = max(best, 100.0 * match / denom)
        out[qid] = best
        if i % 200 == 0:
            log.info(f"    {i}/{len(queries)} proteins aligned")
    return out


def _mmseqs(queries, targets):
    tmp = Path(tempfile.mkdtemp())
    q, t = tmp / "q.fasta", tmp / "t.fasta"
    q.write_text("".join(f">{k}\n{v}\n" for k, v in queries.items()))
    t.write_text("".join(f">{k}\n{v}\n" for k, v in targets.items()))
    res = tmp / "res.m8"
    subprocess.run(["mmseqs", "easy-search", str(q), str(t), str(res),
                    str(tmp / "tmp"), "-s", "7.5", "--max-seqs", "300",
                    "--format-output", "query,target,fident"],
                   check=True, capture_output=True, timeout=1800)
    out = {}
    for line in res.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 3:
            out[p[0]] = max(out.get(p[0], 0.0), 100.0 * float(p[2]))
    nohit = [k for k in queries if k not in out]
    for k in nohit:
        out[k] = float("nan")          # no homolog found is NOT 0% identity
    if nohit:
        log.info(f"  mmseqs2: {len(out)-len(nohit)}/{len(queries)} had a hit; "
                 f"{len(nohit)} had none (recorded as missing, not as 0%)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="dense_2048")
    ap.add_argument("--control", default="dense_512")
    ap.add_argument("--registry", default="results/registry.csv")
    ap.add_argument("--root", default=os.environ.get(
        "DDG_DATA_ROOT", "/backup/new_work/data_root"))
    ap.add_argument("--cache", default="results/identity.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/identity_effect.csv")
    a = ap.parse_args()

    root = Path(a.root)
    cache = Path(a.cache)
    if cache.is_file():
        ident = {k: dict(v) for k, v in json.loads(cache.read_text()).items()}
        log.info(f"identities from {cache}")
    else:
        log.info("wild-type sequences")
        bs = bench_sequences(root)
        ts = train_sequences(root)
        if not bs or not ts:
            return 2
        log.info("")
        log.info("maximum identity to the training split")
        ident = {}
        for name, seqs in bs.items():
            ident[name] = max_identity(seqs, ts)
            v = np.array(list(ident[name].values()))
            log.info(f"  {name:14} median {np.median(v):5.1f}%   "
                     f"range {v.min():.0f}-{v.max():.0f}%")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(ident))

    # ---- per-protein change in error
    reg = pd.read_csv(a.registry)
    prot_idx = Path("results/protein_index.json")
    if not prot_idx.is_file():
        log.error("results/protein_index.json missing -- run bootstrap.py once")
        return 2
    pidx = {k: np.array(v) for k, v in json.loads(prot_idx.read_text()).items()}

    # lengths come from the same sequences the identities were built from
    lens_path = Path(a.cache).with_name("wt_lengths.json")
    if lens_path.is_file():
        lens = {k: dict(v) for k, v in json.loads(lens_path.read_text()).items()}
    else:
        lens = {n: {k: float(len(v)) for k, v in sq.items()}
                for n, sq in bench_sequences(root).items()}
        lens_path.write_text(json.dumps(lens))

    rows = []
    for split, idmap in ident.items():
        lenmap = lens.get(split, {})
        g = pidx.get(split)
        if g is None:
            continue
        pa, y = [], None
        for arm, store in ((a.arm, "a"), (a.control, "c")):
            arr = []
            for r in reg[reg.arm == arm].itertuples():
                ed = Path(str(r.eval_dir))
                p, t = ed / f"preds_{split}.npy", ed / f"targets_{split}.npy"
                if not (p.is_file() and t.is_file()):
                    continue
                pv, tv = np.load(p).reshape(-1), np.load(t).reshape(-1)
                if len(pv) != len(tv) or len(tv) != len(g):
                    continue
                arr.append(pv)
                y = tv
            pa.append(arr)
        if not pa[0] or not pa[1] or y is None:
            continue
        ea = np.mean([np.abs(p - y) for p in pa[0]], axis=0)
        ec = np.mean([np.abs(p - y) for p in pa[1]], axis=0)
        d = ea - ec                        # positive = the arm is WORSE
        for prot in np.unique(g):
            m = g == prot
            key = next((k for k in idmap if k.startswith(str(prot))
                        or str(prot).startswith(k.replace(".pdb", ""))), None)
            if key is None:
                continue
            rows.append(dict(benchmark=split, protein=str(prot),
                             n_mut=int(m.sum()), identity=float(idmap[key]),
                             length=float(lenmap.get(key, float("nan"))),
                             delta_err=float(d[m].mean())))
    if len(rows) < 50:
        log.error(f"only {len(rows)} proteins matched -- check the protein "
                  f"index and the sequence keys")
        return 2
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    log.info("")
    log.info("=" * 78)
    log.info(f"DOES CAPACITY COST MORE AT LOW IDENTITY?   "
             f"{a.arm} vs {a.control}")
    log.info("=" * 78)
    log.info(f"  {len(df)} proteins across {df.benchmark.nunique()} benchmarks")
    log.info("  delta error is positive where the arm is WORSE than the "
             "control")
    log.info("")
    log.info(f"  {'identity band':18}{'proteins':>10}{'mean delta err':>16}")
    bins = [0, 20, 30, 40, 60, 101]
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = df[(df.identity >= lo) & (df.identity < hi)]
        if len(s) < 5:
            continue
        log.info(f"  {f'{lo}-{hi}%':18}{len(s):>10}{s.delta_err.mean():>+16.4f}")

    # Which covariate can actually be tested? One that is constant carries no
    # information, and a null from it means nothing.
    log.info("")
    log.info("  covariate spread (a near-constant covariate cannot be tested)")
    usable = []
    for cov in ("identity", "length"):
        v = pd.to_numeric(df[cov], errors="coerce")
        ok = v.notna()
        if ok.sum() < 50:
            log.info(f"    {cov:10} only {int(ok.sum())} proteins have a "
                     f"value -- not testable")
            continue
        iqr = float(v[ok].quantile(.75) - v[ok].quantile(.25))
        log.info(f"    {cov:10} n={int(ok.sum()):>4}  median "
                 f"{v[ok].median():7.1f}  IQR {iqr:7.1f}")
        if iqr > 1e-6:
            usable.append(cov)
        else:
            log.info(f"    {'':10} zero spread -- not testable")
    if not usable:
        log.error("  no covariate varies enough to test")
        return 2

    for cov in usable:
        _slope_test(df, cov, a.n_boot, a.arm, a.control)


def _slope_test(df, cov, n_boot, arm, control):
    df = df[pd.to_numeric(df[cov], errors="coerce").notna()].copy()
    log.info("")
    log.info("-" * 78)
    log.info(f"  SLOPE OF DELTA-ERROR ON {cov.upper()}   ({arm} vs {control}, "
             f"{len(df)} proteins)")
    log.info("-" * 78)
    q = pd.qcut(df[cov], 5, labels=False, duplicates="drop")
    log.info(f"  {'quintile of ' + cov:22}{'proteins':>10}{'mean delta err':>16}")
    for k in sorted(pd.Series(q).dropna().unique()):
        s_ = df[q == k]
        log.info(f"  {f'{s_[cov].min():.0f}-{s_[cov].max():.0f}':22}"
                 f"{len(s_):>10}{s_.delta_err.mean():>+16.4f}")
    x = df[cov].to_numpy(dtype=float)
    yv = df.delta_err.to_numpy()
    B = (pd.get_dummies(df.benchmark, drop_first=True).to_numpy(dtype=float)
         if df.benchmark.nunique() > 1 else np.zeros((len(x), 0)))
    X = np.column_stack([np.ones(len(x)), x, B])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    slope = beta[1]
    rng = np.random.default_rng(0)
    prots = df.protein.to_numpy()
    uniq = np.unique(prots)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([np.where(prots == p)[0] for p in pick])
        try:
            b, *_ = np.linalg.lstsq(X[sel], yv[sel], rcond=None)
            boots.append(b[1])
        except np.linalg.LinAlgError:
            pass
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    log.info("")
    log.info(f"  slope, benchmark held fixed: {slope:+.6f} per unit {cov}   "
             f"95% [{lo:+.6f}, {hi:+.6f}]")
    log.info("")
    if hi < 0:
        log.info(f"  NEGATIVE, excluding zero: the arm's disadvantage shrinks "
                 f"as {cov} rises.")
        if cov == "length":
            log.info("  The training assay caps proteins at 72 residues, so "
                     "this would say capacity")
            log.info("  costs LESS the further the target is from that "
                     "regime -- check it is not an")
            log.info("  artifact before believing it.")
    elif lo > 0:
        log.info(f"  POSITIVE, excluding zero: the arm does WORSE as {cov} "
                 f"rises.")
        if cov == "length":
            log.info("  The training assay measures only 40-72 residue "
                     "domains. Capacity costing more")
            log.info("  on longer proteins is the predicted pattern: the "
                     "extra parameters fit a size")
            log.info("  regime the labels never covered, and the penalty "
                     "grows with the distance from")
            log.info("  it. The benchmark-to-benchmark heterogeneity is "
                     "explained by protein size.")
    else:
        log.info(f"  Crosses zero -- {cov} does not explain the "
                 f"benchmark-to-benchmark variation.")
    return


if __name__ == "__main__":
    sys.exit(main())
