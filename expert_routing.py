"""
expert_routing.py
=================
What the experts specialise in, and whether that specialisation is biochemical.

WHY THE EXPERT ROUTER AND NOT THE LAYER ROUTER
----------------------------------------------
The depth arms lose 0.019 Spearman on S669 and cannot be read anyway: swapping
all eight mixed layers for a different eight changes the predictions by less
than a thousandth, so layer identity carries no attributable information at the
magnitude the residual gate chooses. The expert router has neither problem. It
is accuracy-neutral (experts8_top2 ties dense_512 on S669 and S461 and beats it
on eight of eleven benchmarks), and the gates act directly on the output rather
than through a 3% residual term.

It also comes with its own control. moe_head.py already partitions the
21-letter alphabet into residue classes -- aliphatic, aromatic, polar, basic,
acidic, glycine/proline, cysteine -- for the `chemistry` routing mode. So the
question "do learned experts recover biochemistry?" can be asked against a
table that is already in the model, with no new labels and no new runs.

WHAT IT MEASURES
----------------
1.  Utilisation. What fraction of mutations each expert receives. A collapsed
    router -- one expert taking nearly everything -- has nothing to interpret,
    and is the first thing to rule out.

2.  Specialisation. For each expert, the wild-type and mutant residues that
    route to it, as enrichment over the background frequency. This is the
    table a reader actually looks at: "expert 3 fires on aliphatic-to-polar
    substitutions."

3.  Agreement with chemistry. Mutual information between the learned expert
    assignment and the residue-class partition, against a permutation null.
    This is the number that decides whether "the experts are biochemical" is a
    claim or a story: if the learned router carries no more information about
    residue class than a shuffled assignment, it does not.

4.  Burial and effect size, when a metadata file is present, so an expert can
    be described by structural environment as well as by chemistry.

    python expert_routing.py --run runs/experts8_top2_s42
    python expert_routing.py --run runs/experts8_top2_s42 --splits s669,s461
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("experts")

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
# The partition moe_head.py uses for its `chemistry` routing mode.
CHEM_GROUPS = {"A": "aliphatic", "V": "aliphatic",
               "L": "aliphatic2", "I": "aliphatic2", "M": "aliphatic2",
               "F": "aromatic", "W": "aromatic", "Y": "aromatic",
               "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
               "K": "basic", "R": "basic", "H": "basic",
               "D": "acidic", "E": "acidic",
               "G": "gly/pro", "P": "gly/pro",
               "C": "cys", "X": "cys"}


class GateCapture:
    """Forward pre-hook on model.mlp. MoEReadout computes its gates inside
    forward(), so they are recaptured here by re-running the router on the same
    representation the head receives -- deterministic at eval, since
    moe_noise_eps is zeroed by the evaluation path."""

    def __init__(self, head):
        self.head = head
        self.gates, self.struct = [], []

    def __call__(self, module, args):
        import torch
        batch = args[0] if args else None
        if not isinstance(batch, dict):
            return None
        x = batch.get("muted_id_representation")
        if x is None:
            return None
        with torch.no_grad():
            t = x.detach()
            if t.dim() == 2:
                t = t.unsqueeze(0)
            B, K, D = t.shape
            flat = t.reshape(B * K, D)
            n_struct = getattr(self.head, "n_struct", 0)
            struct = (self.head._structural_features(batch, B * K, flat.device,
                                                     flat.dtype)
                      if n_struct else
                      torch.zeros(B * K, 0, device=flat.device,
                                  dtype=flat.dtype))
            chem = (self.head._chem_expert(batch, B * K, flat.device)
                    if getattr(self.head, "route_mode", "learned")
                    in ("chemistry", "hybrid") else None)
            gates, probs, logits, topk = self.head._route(flat, struct, chem)
            self.gates.append(gates.float().cpu().numpy())
            # The router's own burial descriptors: log1p CA neighbour counts at
            # 8/10/12 A plus relative sequence position. Captured here because
            # they are already computed and are the obvious rival explanation
            # for any apparent chemical specialisation.
            self.struct.append(struct.float().cpu().numpy()
                               if struct.numel() else
                               np.zeros((flat.shape[0], 0), dtype=np.float32))
        return None

    def take(self):
        g = (np.concatenate(self.gates, axis=0) if self.gates
             else np.zeros((0, 0), dtype=np.float32))
        st = (np.concatenate(self.struct, axis=0) if self.struct
              else np.zeros((0, 0), dtype=np.float32))
        self.gates, self.struct = [], []
        return g, st


def enrichment(counts: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """log2 of observed over expected, with a half-count floor."""
    obs = counts / max(counts.sum(), 1)
    exp = bg / max(bg.sum(), 1)
    return np.log2((obs + 1e-9) / (exp + 1e-9))


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """MI in bits between two integer labellings."""
    ca = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy().astype(float)
    if ca.size == 0:
        return float("nan")
    p = ca / ca.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p * np.log2(p / (pa @ pb))
    return float(np.nansum(t))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a trained MoE run directory")
    ap.add_argument("--data-root", default=os.environ.get(
        "DDG_DATA_ROOT", "/backup/new_work/data_root"))
    ap.add_argument("--mpnn-ckpt", default=os.environ.get(
        "MPNN_CKPT", "/backup/new_work/checkpoints/v_48_020.pt"))
    ap.add_argument("--splits", default="s669,s461,s2648,ssym_direct",
                    help="benchmarks to route; megascale_test is large")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed label for the output rows; inferred from the "
                         "run directory name (…_s42) when not given. "
                         "section6.py tests whether the partition holds "
                         "ACROSS seeds, so every row has to carry one.")
    a = ap.parse_args()

    run = Path(a.run)
    outdir = Path(a.out) if a.out else run / "routing"
    outdir.mkdir(parents=True, exist_ok=True)

    # Every row carries its seed so routing.csv files from different runs can
    # be concatenated and analysed across seeds. A partition that holds on one
    # seed and not the others is not a finding.
    if a.seed is not None:
        seed = int(a.seed)
    else:
        m = re.search(r"_s(\d+)$", run.name)
        seed = int(m.group(1)) if m else -1
        if seed < 0:
            log.warning(f"  cannot read a seed from '{run.name}'; rows will "
                        f"carry seed=-1. Pass --seed to set it.")
    log.info(f"  seed label for these rows: {seed}")

    man = run / "run_manifest.json"
    if not man.is_file():
        log.error(f"{man} missing -- cannot tell how this run was configured")
        return 2
    cfg = json.loads(man.read_text())["config"]
    if not cfg.get("use_moe") or int(cfg.get("moe_num_experts", 1)) < 2:
        log.error(f"{run.name} is not a routed arm "
                  f"(use_moe={cfg.get('use_moe')}, "
                  f"experts={cfg.get('moe_num_experts')})")
        return 2
    n_exp = int(cfg["moe_num_experts"])
    log.info(f"{run.name}: {n_exp} experts, top-{cfg.get('moe_top_k')}, "
             f"route_mode={cfg.get('moe_route_mode')}, "
             f"router_struct={cfg.get('moe_router_struct')}")

    import torch
    from torch.utils.data import DataLoader
    from ddg_data import Alphabet, Featurizer
    from ddg_data.datasets.combined import MegaScaleTestDatasets
    from ddg_models import MultimodalDDG, MultimodalDDGConfig
    from ddg_train.evaluate import _move_batch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mc = MultimodalDDGConfig()
    mc.adapter_layer_indices = [int(cfg.get("adapter_layer", -3))]
    mc.name = cfg.get("esm_name", "esm2_t33_650M_UR50D")
    mc.proteinmpnn_ckpt = a.mpnn_ckpt
    mc.encoder.tune = bool(cfg.get("tune_mpnn", True))
    mc.mlp.input_dim = 128
    mc.mlp.hidden_dim = [512, 512, 512]
    mc.mlp.output_dim = 21
    mc.mlp.dropout = 0.1
    mc.mlp.use_moe = True
    for k in ("moe_num_experts", "moe_top_k", "moe_shared_expert",
              "moe_router_struct", "moe_route_mode", "moe_balance_mode",
              "moe_shared_count", "moe_expert_mode", "moe_lora_rank",
              "moe_expert_hidden"):
        if k in cfg and cfg[k] is not None:
            setattr(mc.mlp, k, cfg[k])
    mc.mlp.moe_noise_eps = 0.0          # deterministic routing

    model = MultimodalDDG(mc).to(device)
    state = torch.load(run / "best.pt", map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info(f"  checkpoint: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()

    featurizer = Featurizer(Alphabet(name="esm"))
    collection = MegaScaleTestDatasets(data_root=a.data_root)
    wanted = {s.strip() for s in a.splits.split(",") if s.strip()}

    rows = []
    cap = GateCapture(model.mlp)
    handle = model.mlp.register_forward_pre_hook(cap)
    try:
        for name, ds in collection.iter_named():
            if name not in wanted:
                continue
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                                collate_fn=featurizer)
            n = 0
            with torch.no_grad():
                for batch in loader:
                    at = batch.get("append_tensors")
                    try:
                        b = _move_batch(batch, device)
                        model(b)
                    except Exception as exc:                      # noqa: BLE001
                        cap.take()
                        continue
                    g, st = cap.take()
                    if g.size == 0 or at is None:
                        continue
                    # The wild-type name, for the per-protein case study. The
                    # featurizer puts it on the batch as a plain string, but a
                    # list or tensor shows up on some paths, so all three are
                    # handled rather than assumed.
                    pname = batch.get("name", batch.get("wt_name", "?"))
                    if isinstance(pname, (list, tuple)):
                        pname = pname[0] if pname else "?"
                    if torch.is_tensor(pname):
                        pname = str(pname.tolist())
                    pname = str(pname)
                    # Sequence index of each mutated position, so structural
                    # annotations (DSSP, ligand contacts) can be joined later.
                    mut_list = batch.get("mut_ids")
                    if mut_list is not None:
                        mut_list = (mut_list.cpu().tolist()
                                    if hasattr(mut_list, "cpu")
                                    else list(mut_list))
                        if len(mut_list) != g.shape[0]:
                            mut_list = None
                    at = at if torch.is_tensor(at) else torch.as_tensor(at)
                    at = at.reshape(-1, at.shape[-1]).cpu().numpy()
                    if at.shape[0] != g.shape[0] or at.shape[1] < 42:
                        continue
                    wt = at[:, :21].argmax(1)
                    mt = at[:, 21:42].argmax(1)
                    top = g.argmax(1)
                    ddg = batch.get("ddG")
                    ddg = (ddg.reshape(-1).cpu().numpy()
                           if torch.is_tensor(ddg) else
                           np.full(len(top), np.nan))
                    if len(ddg) != len(top):
                        ddg = np.full(len(top), np.nan)
                    has_st = st.shape[0] == g.shape[0] and st.shape[1] >= 3
                    for i in range(len(top)):
                        r = dict(split=name, seed=seed, protein=pname,
                                 pos=int(mut_list[i]) if mut_list is not None
                                 else -1,
                                 expert=int(top[i]),
                                 gate=float(g[i, top[i]]),
                                 wt=ALPHABET[int(wt[i])],
                                 mut=ALPHABET[int(mt[i])],
                                 ddg=float(ddg[i]))
                        if has_st:
                            r["burial8"] = float(st[i, 0])
                            r["burial10"] = float(st[i, 1])
                            r["burial12"] = float(st[i, 2])
                        rows.append(r)
                    n += len(top)
            log.info(f"  {name}: {n} mutations routed")
            if n and "?" in set(r_["protein"] for r_ in rows[-n:]):
                log.warning(f"  {name}: some rows have no protein name -- the "
                            f"per-protein case study will be incomplete")
    finally:
        handle.remove()

    if not rows:
        log.error("nothing routed -- check the splits and the checkpoint")
        return 2
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "routing.csv", index=False)

    # ---------------------------------------------------------------- 1
    log.info("")
    log.info("=" * 78)
    log.info("1. UTILISATION")
    log.info("=" * 78)
    use = df.expert.value_counts(normalize=True).sort_index()
    for e in range(n_exp):
        f = float(use.get(e, 0.0))
        bar = "#" * int(round(f * 60))
        log.info(f"  expert {e:<2} {100*f:5.1f}%  {bar}")
    top1 = float(use.max()) if len(use) else 1.0
    live = int((use > 0.01).sum())
    log.info(f"  {live}/{n_exp} experts take more than 1% of mutations; "
             f"the busiest takes {100*top1:.1f}%")
    if live < 3:
        log.warning("  the router has collapsed -- there is nothing to "
                    "interpret here")

    # ---------------------------------------------------------------- 2
    log.info("")
    log.info("=" * 78)
    log.info("2. WHAT EACH EXPERT FIRES ON  (log2 enrichment over background)")
    log.info("=" * 78)
    bg_wt = df.wt.value_counts().reindex(list(ALPHABET), fill_value=0).to_numpy()
    bg_mt = df.mut.value_counts().reindex(list(ALPHABET), fill_value=0).to_numpy()
    spec = []
    for e in range(n_exp):
        sub = df[df.expert == e]
        if len(sub) < 30:
            continue
        ew = enrichment(sub.wt.value_counts().reindex(
            list(ALPHABET), fill_value=0).to_numpy(), bg_wt)
        em = enrichment(sub.mut.value_counts().reindex(
            list(ALPHABET), fill_value=0).to_numpy(), bg_mt)
        tw = [ALPHABET[i] for i in np.argsort(-ew)[:4]]
        tm = [ALPHABET[i] for i in np.argsort(-em)[:4]]
        med = sub.ddg.median()
        log.info(f"  expert {e:<2} n={len(sub):<6} "
                 f"wt→ {'/'.join(tw):<9} mut→ {'/'.join(tm):<9} "
                 f"median ddG {med:+.2f}")
        spec.append(dict(expert=e, n=len(sub), top_wt="/".join(tw),
                         top_mut="/".join(tm), median_ddg=med))
    pd.DataFrame(spec).to_csv(outdir / "expert_specialisation.csv", index=False)

    # ---------------------------------------------------------------- 3
    log.info("")
    log.info("=" * 78)
    log.info("3. DOES THE LEARNED ROUTER RECOVER BIOCHEMISTRY?")
    log.info("=" * 78)
    cls = df.wt.map(CHEM_GROUPS).fillna("cys")
    codes = pd.Categorical(cls).codes
    mi = mutual_information(df.expert.to_numpy(), codes)
    rng = np.random.default_rng(0)
    null = np.array([mutual_information(
        rng.permutation(df.expert.to_numpy()), codes) for _ in range(a.n_perm)])
    z = (mi - null.mean()) / (null.std() + 1e-12)
    p = float((null >= mi).mean())
    h_cls = -np.sum([(c / len(codes)) * np.log2(c / len(codes))
                     for c in np.bincount(codes) if c > 0])
    log.info(f"  MI(expert; wild-type residue class) = {mi:.4f} bits")
    log.info(f"  permutation null  {null.mean():.4f} +/- {null.std():.4f} bits"
             f"   z = {z:.1f}   p = {p:.3f}")
    log.info(f"  class entropy {h_cls:.3f} bits, so the router explains "
             f"{100*mi/max(h_cls,1e-9):.1f}% of it")
    log.info("")
    if p < 0.01 and mi / max(h_cls, 1e-9) > 0.05:
        log.info("  The learned router carries real information about residue "
                 "class. The experts")
        log.info("  are biochemically organised, and the per-expert table "
                 "above is readable.")
    elif p < 0.01:
        log.info("  Significant but small. The router knows something about "
                 "chemistry, but chemistry")
        log.info("  is not what mainly determines routing -- say that, rather "
                 "than overclaiming.")
    else:
        log.info("  No more than chance. The experts do NOT partition by "
                 "residue class, and any")
        log.info("  biochemical reading of the table above would be "
                 "storytelling.")

    # ---------------------------------------------------------------- 4
    log.info("")
    log.info("=" * 78)
    log.info("4. WHICH DRIVER?  chemistry vs burial vs effect size")
    log.info("=" * 78)
    log.info("  ESM encodes residue identity, so a router reading it knows "
             "chemistry for free.")
    log.info("  The question is whether chemistry survives once burial and "
             "effect size are held")
    log.info("  fixed -- and whether each driver carries information the "
             "others do not.")
    log.info("")

    def strata(x, k=4):
        """Quartile bins, as integer labels; NaNs go to their own bin."""
        v = pd.to_numeric(pd.Series(x), errors="coerce")
        out = pd.qcut(v, k, labels=False, duplicates="drop")
        return pd.Series(out).fillna(-1).astype(int).to_numpy()

    e = df.expert.to_numpy()
    drivers = {"residue class": codes}
    if "burial10" in df.columns and df.burial10.notna().any():
        drivers["burial (CA within 10A)"] = strata(df.burial10)
    if df.ddg.notna().any():
        drivers["effect size |ddG|"] = strata(df.ddg.abs())

    marg = {}
    for nm, lab in drivers.items():
        m = mutual_information(e, lab)
        nl = np.array([mutual_information(rng.permutation(e), lab)
                       for _ in range(a.n_perm)])
        h = -np.sum([(c / len(lab)) * np.log2(c / len(lab))
                     for c in np.bincount(lab - lab.min()) if c > 0])
        marg[nm] = (m, float(nl.mean()), float(nl.std()), h)
        log.info(f"  MI(expert; {nm:24}) = {m:.4f} bits "
                 f"(null {nl.mean():.4f}, {100*m/max(h,1e-9):4.1f}% of its "
                 f"entropy)")

    if len(drivers) > 1:
        log.info("")
        log.info("  conditional -- does each driver survive holding the "
                 "others fixed?")
        log.info("  Compared against a WITHIN-STRATUM permutation null. That "
                 "null is essential:")
        log.info("  slicing a few thousand mutations into strata leaves only a "
                 "handful per cell,")
        log.info("  and mutual-information estimates are badly biased upward "
                 "there -- on pure")
        log.info("  noise, conditioning inflates MI more than twentyfold. Only "
                 "the excess over")
        log.info("  this null means anything.")
        log.info("")

        def conditional_mi(expert, target, others):
            joint = others[0].astype(np.int64)
            for o in others[1:]:
                joint = joint * (o.max() + 2) + o
            tot, n_tot = 0.0, 0
            for g in np.unique(joint):
                sel = joint == g
                if sel.sum() < 50:
                    continue
                tot += sel.sum() * mutual_information(expert[sel], target[sel])
                n_tot += sel.sum()
            return tot / max(n_tot, 1), joint

        names = list(drivers)
        for nm in names:
            others = [drivers[o] for o in names if o != nm]
            if not others:
                continue
            cmi, joint = conditional_mi(e, drivers[nm], others)
            # permute experts WITHIN each stratum: destroys the association
            # while preserving sample sizes, so the bias is identical
            nulls = []
            for _ in range(max(a.n_perm // 4, 50)):
                ep = e.copy()
                for g in np.unique(joint):
                    sel = np.where(joint == g)[0]
                    if len(sel) > 1:
                        ep[sel] = rng.permutation(ep[sel])
                nulls.append(conditional_mi(ep, drivers[nm], others)[0])
            nulls = np.array(nulls)
            z = (cmi - nulls.mean()) / (nulls.std() + 1e-12)
            pv = float((nulls >= cmi).mean())
            if pv < 0.01 and z > 3:
                verdict = "SURVIVES -- a separable axis"
            elif pv < 0.05:
                verdict = "weak but above the null"
            else:
                verdict = "NOT separable -- a proxy for the others"
            log.info(f"    {nm:26} MI {marg[nm][0]:.4f} | cond {cmi:.4f} "
                     f"vs null {nulls.mean():.4f}  z={z:5.1f} p={pv:.3f}  "
                     f"{verdict}")

    # per-expert burial, which is what a wet-lab reader asks for
    if "burial10" in df.columns and df.burial10.notna().any():
        log.info("")
        log.info(f"  {'expert':8} {'n':>6} {'burial (median)':>16} "
                 f"{'median ddG':>11}")
        bmed = df.burial10.median()
        for ex in range(n_exp):
            sub = df[df.expert == ex]
            if len(sub) < 30:
                continue
            b = sub.burial10.median()
            tag = "buried" if b > bmed else "exposed"
            log.info(f"  {ex:<8} {len(sub):>6} {b:>16.3f}  {tag:<8} "
                     f"{sub.ddg.median():>+10.2f}")

    json.dump(dict(mi_bits=mi, null_mean=float(null.mean()),
                   null_sd=float(null.std()), z=float(z), p=p,
                   class_entropy=h_cls, n_experts=n_exp,
                   live_experts=live, busiest_share=top1),
              open(outdir / "routing_summary.json", "w"), indent=1)
    log.info("")
    log.info(f"  wrote {outdir}/routing.csv, expert_specialisation.csv, "
             f"routing_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
