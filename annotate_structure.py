"""
annotate_structure.py
=====================
Does the expert partition track structural annotations that are external,
standard, and not of our own making -- after burial is held fixed?

WHY THIS AND NOT MORE CASE STUDIES
----------------------------------
The DHPS case study tested expert assignment against the structure blocks from
Sanyal, Shivram et al. 2025. Two problems with leaning on it: the blocks are
our own construct, which a reviewer will call circular, and 33 residues in the
critical block is 33 residues however many substitutions sit on them. Another
single protein does not fix either.

This does the systematic version instead, on the hundreds of benchmark
proteins already routed, against two annotations nobody can call ours:

  SECONDARY STRUCTURE (DSSP).  Helix, strand, loop. Computed here from the
  backbone by DSSP's own hydrogen-bond energy criterion, so no mkdssp binary
  is needed. Crucially it is NOT a burial proxy: helices and strands each
  contain buried and exposed positions, so it can survive the burial control
  that the DHPS blocks did not.

  LIGAND CONTACT.  Residues with any atom within a cutoff of a non-water
  heteroatom. Functionally defined rather than structurally, so it is
  orthogonal again.

THE TEST
--------
MI(expert; annotation), then MI(expert; annotation | burial quartile), both
against a null that permutes the ANNOTATION ACROSS RESIDUES.

That null is the whole point. An annotation is a property of a position: every
substitution at one residue carries the same value. Permuting across mutations
treats 19 substitutions as 19 observations when they are one, and inflates z by
roughly the square root of that factor -- on simulated data with NO true
association it returned z = +22 where the residue-level null returned -1.7.
The DHPS block result fell from z = 169 to z = 3.4 on exactly this correction,
and the SB3 result vanished.

    python annotate_structure.py --routing results/routing.csv
    python annotate_structure.py --routing results/routing.csv \\
        --pdb-dirs "$DDG_DATA_ROOT/data/dataset/S669/pdb,..." --ligand-cutoff 4.5
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
log = logging.getLogger("annot")

Q1Q2 = 0.42 * 0.20 * 332.0        # DSSP's electrostatic prefactor, kcal/mol
HB_CUTOFF = -0.5                  # DSSP's hydrogen-bond energy threshold
WATERS = {"HOH", "DOD", "WAT", "H2O"}


# --------------------------------------------------------------------------- #
# Backbone parsing and DSSP
# --------------------------------------------------------------------------- #
def read_backbone(path: Path, chain: str | None = None):
    """Ordered residues with N, CA, C, O, plus heteroatom coordinates."""
    res: dict = {}
    order: list = []
    het: list = []
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", "ignore").rstrip()
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                break
            if rec.startswith("HETATM"):
                comp = line[17:20].strip()
                if comp not in WATERS:
                    try:
                        het.append([float(line[i:i + 8]) for i in (30, 38, 46)])
                    except ValueError:
                        pass
                if comp != "MSE":
                    continue
                line = line.replace("HETATM", "ATOM  ")
            elif not rec.startswith("ATOM"):
                continue
            ch = line[21:22]
            if chain is not None and ch != chain:
                continue
            alt = line[16:17]
            if alt not in (" ", "A"):
                continue
            key = (ch, line[22:27].strip())
            atom = line[12:16].strip()
            if key not in res:
                res[key] = {}
                order.append(key)
            if atom in ("N", "CA", "C", "O"):
                try:
                    res[key][atom] = np.array(
                        [float(line[i:i + 8]) for i in (30, 38, 46)])
                except ValueError:
                    pass
            if atom:
                res[key].setdefault("_all", []).append(line)
    keep = [k for k in order if {"N", "CA", "C", "O"} <= set(res[k])]
    return keep, res, (np.array(het) if het else np.zeros((0, 3)))


def dssp_secondary(keep, res):
    """Helix / strand / loop per residue, by DSSP's own criterion.

    The amide hydrogen is placed on N along the previous residue's C->O
    direction, the electrostatic H-bond energy is evaluated, and n-turns and
    bridges are read off the resulting bond map. This reproduces DSSP's
    definition closely enough for a categorical H/E/L label, and avoids
    depending on an mkdssp binary being installed.
    """
    n = len(keep)
    N = np.array([res[k]["N"] for k in keep])
    CA = np.array([res[k]["CA"] for k in keep])
    C = np.array([res[k]["C"] for k in keep])
    O = np.array([res[k]["O"] for k in keep])

    H = N.copy()
    for i in range(1, n):
        d = C[i - 1] - O[i - 1]
        nd = np.linalg.norm(d)
        if nd > 1e-6:
            H[i] = N[i] + d / nd
    H[0] = N[0]

    # chain breaks: a CA-CA step far from 3.8 A is not a peptide bond
    broken = np.zeros(n, bool)
    for i in range(1, n):
        if np.linalg.norm(CA[i] - CA[i - 1]) > 4.5:
            broken[i] = True

    def energy(i, j):
        """H-bond from residue i's N-H to residue j's C=O."""
        if i == 0 or broken[i]:
            return 0.0
        rON = np.linalg.norm(O[j] - N[i])
        rCH = np.linalg.norm(C[j] - H[i])
        rOH = np.linalg.norm(O[j] - H[i])
        rCN = np.linalg.norm(C[j] - N[i])
        if min(rON, rCH, rOH, rCN) < 0.5:
            return 0.0
        return Q1Q2 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN)

    close = np.linalg.norm(CA[:, None, :] - CA[None, :, :], axis=-1) < 9.0
    hb = np.zeros((n, n), bool)
    for i in range(n):
        for j in np.flatnonzero(close[i]):
            if abs(i - j) < 2:
                continue
            if energy(i, int(j)) < HB_CUTOFF:
                hb[i, j] = True

    ss = np.array(["L"] * n, dtype="<U1")
    # n-turns: two consecutive turns of the same size make a helix
    for nn in (4, 3, 5):
        turn = np.zeros(n, bool)
        for i in range(n - nn):
            if hb[i + nn, i]:
                turn[i] = True
        for i in range(n - nn - 1):
            if turn[i] and turn[i + 1]:
                ss[i + 1:i + nn + 1] = "H"
    # bridges: the parallel and antiparallel patterns
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if abs(i - j) < 3:
                continue
            anti = ((hb[i, j] and hb[j, i]) or
                    (hb[j + 1, i - 1] and hb[i + 1, j - 1]))
            para = ((hb[i, j - 1] and hb[j + 1, i]) or
                    (hb[j, i - 1] and hb[i + 1, j]))
            if anti or para:
                if ss[i] == "L":
                    ss[i] = "E"
                if ss[j] == "L":
                    ss[j] = "E"
    return ss


def ligand_contacts(keep, res, het, cutoff: float):
    if het.shape[0] == 0:
        return np.zeros(len(keep), bool)
    out = np.zeros(len(keep), bool)
    for idx, k in enumerate(keep):
        xyz = []
        for line in res[k].get("_all", []):
            try:
                xyz.append([float(line[i:i + 8]) for i in (30, 38, 46)])
            except ValueError:
                continue
        if not xyz:
            continue
        d = np.linalg.norm(np.array(xyz)[:, None, :] - het[None, :, :], axis=-1)
        out[idx] = bool((d < cutoff).any())
    return out


# --------------------------------------------------------------------------- #
def mutual_information(a, b) -> float:
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    a = a - a.min()
    b = b - b.min()
    j = np.zeros((a.max() + 1, b.max() + 1), float)
    np.add.at(j, (a, b), 1.0)
    j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = j * np.log2(j / (pa * pb))
    return float(np.nansum(t))


def permute_by_residue(labels, residue_key, rng):
    """Shuffle a RESIDUE-level annotation across residues, keeping each
    residue's substitutions together. The annotation is a property of the
    position, so the residue is the unit and permuting across mutations would
    treat 19 substitutions as 19 independent observations."""
    uniq, inv = np.unique(residue_key, return_inverse=True)
    first = np.zeros(len(uniq), dtype=labels.dtype)
    for i in range(len(uniq)):
        v = labels[inv == i]
        if len(v):
            first[i] = v[0]
    return first[rng.permutation(len(uniq))][inv]


def test_annotation(expert, annot, residue_key, burial, n_perm, rng, label):
    """Marginal and burial-conditional MI, each against a residue-level null."""
    mi = mutual_information(expert, annot)
    null = np.array([mutual_information(
        expert, permute_by_residue(annot, residue_key, rng))
        for _ in range(n_perm)])
    z = (mi - null.mean()) / (null.std() + 1e-12)
    p = float((null >= mi).mean())
    log.info(f"  {label:26} MI {mi:.4f}  null {null.mean():.4f}"
             f"+/-{null.std():.4f}  z {z:6.1f}  p {p:.3f}")
    row = dict(annotation=label, mi=mi, null=float(null.mean()),
               null_sd=float(null.std()), z=float(z), p=p)

    if burial is None:
        return row
    strat = pd.qcut(pd.Series(burial), 4, labels=False,
                    duplicates="drop").fillna(-1).astype(int).to_numpy()

    def cond(an):
        tot, ntot = 0.0, 0
        for g in np.unique(strat):
            sel = strat == g
            if sel.sum() < 50:
                continue
            tot += sel.sum() * mutual_information(expert[sel], an[sel])
            ntot += sel.sum()
        return tot / max(ntot, 1)

    cmi = cond(annot)
    cnull = []
    for _ in range(max(n_perm // 4, 100)):
        ap = annot.copy()
        for g in np.unique(strat):
            sel = np.where(strat == g)[0]
            if len(sel) > 1:
                ap[sel] = permute_by_residue(annot[sel], residue_key[sel], rng)
        cnull.append(cond(ap))
    cnull = np.array(cnull)
    cz = (cmi - cnull.mean()) / (cnull.std() + 1e-12)
    cp = float((cnull >= cmi).mean())
    log.info(f"  {'  | burial quartile':26} MI {cmi:.4f}  "
             f"null {cnull.mean():.4f}+/-{cnull.std():.4f}  z {cz:6.1f}  "
             f"p {cp:.3f}"
             + ("   SURVIVES" if (cp < 0.01 and cz > 3) else "   not separable"))
    row.update(cond_mi=cmi, cond_null=float(cnull.mean()),
               cond_z=float(cz), cond_p=cp,
               survives=bool(cp < 0.01 and cz > 3))
    return row


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routing", default="results/routing.csv")
    ap.add_argument("--data-root", default=os.environ.get(
        "DDG_DATA_ROOT", "/backup/new_work/data_root"))
    ap.add_argument("--pdb-dirs", default="",
                    help="comma-separated; default searches data-root")
    ap.add_argument("--ligand-cutoff", type=float, default=4.5)
    ap.add_argument("--n-perm", type=int, default=400)
    ap.add_argument("--seed-only", type=int, default=None,
                    help="use one seed's rows (default: the lowest)")
    ap.add_argument("--outdir", default="paper/structure")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rp = Path(a.routing)
    if not rp.is_file():
        log.error(f"{rp} not found -- run expert_routing.py first")
        return 2
    R = pd.read_csv(rp)
    for need in ("expert", "protein", "pos"):
        if need not in R.columns:
            log.error(f"routing.csv has no '{need}' column. Re-run "
                      f"expert_routing.py with the updated version -- the "
                      f"sequence position is required to join structural "
                      f"annotations.")
            return 2
    if "seed" in R.columns and R.seed.nunique() > 1:
        s = a.seed_only if a.seed_only is not None else sorted(R.seed.unique())[0]
        R = R[R.seed == s]
        log.info(f"  using seed {s} ({len(R)} rows)")
    R = R[R.pos >= 0]

    dirs = ([Path(d.strip()) for d in a.pdb_dirs.split(",") if d.strip()]
            or sorted((Path(a.data_root) / "data/dataset").glob("*/pdb")))
    log.info(f"  searching {len(dirs)} PDB directories")
    index: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.glob("*.pdb"):
            index.setdefault(f.stem.lower(), f)
    log.info(f"  {len(index)} structures indexed")
    if not index:
        log.error("  no PDB files found -- pass --pdb-dirs explicitly")
        return 2

    recs, miss = [], set()
    for prot, g in R.groupby("protein"):
        stem = str(prot).lower()
        f = index.get(stem) or index.get(stem[:4]) or index.get(stem.split("_")[0])
        if f is None:
            miss.add(str(prot))
            continue
        try:
            keep, res, het = read_backbone(f)
            if len(keep) < 20:
                miss.add(str(prot))
                continue
            ss = dssp_secondary(keep, res)
            lig = ligand_contacts(keep, res, het, a.ligand_cutoff)
        except Exception as exc:                                  # noqa: BLE001
            log.warning(f"  {prot}: {type(exc).__name__}: {exc}")
            miss.add(str(prot))
            continue
        for r in g.itertuples():
            i = int(r.pos)
            if not (0 <= i < len(keep)):
                continue
            rec = dict(protein=str(prot), pos=i, expert=int(r.expert),
                       ss=str(ss[i]), ligand=bool(lig[i]),
                       residue_key=f"{prot}:{i}")
            if "burial10" in R.columns and np.isfinite(getattr(r, "burial10",
                                                               np.nan)):
                rec["burial10"] = float(r.burial10)
            recs.append(rec)

    if len(recs) < 500:
        log.error(f"only {len(recs)} mutations annotated -- too few. "
                  f"{len(miss)} proteins had no matching structure.")
        return 2
    D = pd.DataFrame(recs)
    D.to_csv(outdir / "annotated_routing.csv", index=False)
    log.info(f"  annotated {len(D)} mutations across "
             f"{D.protein.nunique()} proteins, "
             f"{D.residue_key.nunique()} residues")
    if miss:
        log.info(f"  {len(miss)} proteins had no matching structure "
                 f"(e.g. {sorted(miss)[:4]})")
    log.info(f"  secondary structure: "
             + ", ".join(f"{k} {100*v:.0f}%" for k, v in
                         D.ss.value_counts(normalize=True).items()))
    log.info(f"  ligand-contacting: {100*D.ligand.mean():.1f}%")

    rng = np.random.default_rng(0)
    e = D.expert.to_numpy()
    rk = D.residue_key.to_numpy()
    burial = (D.burial10.to_numpy() if "burial10" in D.columns
              and D.burial10.notna().all() else None)
    if burial is None:
        log.warning("  no burial column -- the conditional test is skipped, "
                    "and a marginal association cannot be distinguished from "
                    "burial")

    log.info("")
    log.info("=" * 86)
    log.info("DOES THE EXPERT PARTITION TRACK EXTERNAL STRUCTURAL ANNOTATION?")
    log.info("=" * 86)
    log.info("  Nulls permute the annotation ACROSS RESIDUES. Permuting across")
    log.info("  mutations would treat 19 substitutions at one position as 19")
    log.info("  observations: on simulated data with no true association that")
    log.info("  returns z = +22 where the residue-level null returns -1.7.")
    log.info("")
    rows = [test_annotation(e, pd.Categorical(D.ss).codes, rk, burial,
                            a.n_perm, rng, "secondary structure")]
    if D.ligand.any() and not D.ligand.all():
        rows.append(test_annotation(e, D.ligand.to_numpy().astype(int), rk,
                                    burial, a.n_perm, rng, "ligand contact"))
    else:
        log.info("  ligand contact: no variation (all or none) -- skipped")

    log.info("")
    log.info(f"  {'expert':>7}{'n':>8}{'helix %':>10}{'strand %':>10}"
             f"{'loop %':>9}{'ligand %':>10}")
    for ex, gg in D.groupby("expert"):
        if len(gg) < 50:
            continue
        v = gg.ss.value_counts(normalize=True)
        log.info(f"  {int(ex):>7}{len(gg):>8}{100*v.get('H',0):>10.1f}"
                 f"{100*v.get('E',0):>10.1f}{100*v.get('L',0):>9.1f}"
                 f"{100*gg.ligand.mean():>10.1f}")
    (D.groupby("expert")
       .agg(n=("ss", "size"), helix=("ss", lambda s: (s == "H").mean()),
            strand=("ss", lambda s: (s == "E").mean()),
            loop=("ss", lambda s: (s == "L").mean()),
            ligand=("ligand", "mean"))
       .to_csv(outdir / "tab_expert_by_structure.csv"))

    pd.DataFrame(rows).to_csv(outdir / "tab_annotation_tests.csv", index=False)
    json.dump(rows, open(outdir / "structure_tests.json", "w"), indent=1,
              default=float)
    log.info("")
    log.info("  A SURVIVES line means the partition carries information about")
    log.info("  that annotation beyond burial -- a third axis, and the one")
    log.info("  claim the DHPS blocks could not support.")
    log.info(f"  wrote {outdir}/annotated_routing.csv, "
             f"tab_annotation_tests.csv, tab_expert_by_structure.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
