#!/usr/bin/env python3
"""
run_fill_missing.py
===================
Re-trains ONLY the runs of the MoE readout grid that are missing or that fail
the architecture check, into a FRESH directory tree. Nothing already on disk is
written to, renamed or deleted.

WHY THIS EXISTS, AND WHY run_moe_full.py CANNOT DO IT
-----------------------------------------------------
`run_moe_full.run_arm` begins with `find_existing(arm, seed)`, which searches
the legacy roots for `<arm>_s<seed>` and `abl_<arm>_s<seed>` and REUSES the
first completed directory it finds. The seed-42 expert runs that fail the check
are exactly those directories: they predate the shared expert, so their
checkpoints carry no `mlp.shared_experts.*`. Re-running run_moe_full.py would
find them again and change nothing. This script disables that lookup for the
arms it is filling, so each target is trained fresh.

WHAT IT FILLS (audited live from disk, not hard-coded)
------------------------------------------------------
Every (arm, seed) in run_moe_full.ARMS x SEEDS whose checkpoint is absent or
fails moe_status.requirements(). On the registry shipped with the project that
is nine runs:

    experts8_top2      seeds 42, 43, 44   <- all three invalid; the primary arm
    experts4_top2      seed 42
    experts8_top4      seed 42
    experts8_narrow    seed 42
    no_struct_route    seed 42
    aux0               seed 42
    aux0p05            seed 42

NOT filled, deliberately:
  * fusion_film / fusion_bilinear / fusion_both / fusion_both_wide -- these 12
    runs trained plain concatenation because modal_fusion.py was not installed
    when they ran. Their replacement is the modal-fusion sweep, which already
    exists in results/moe/modal_fusion_sweep.csv with three seeds per arm.
    Re-running them here would duplicate that sweep under worse names.
  * the adapter-placement ablation -- it has its own script,
    run_adapter_ablation.py, which writes to runs_adapter/ and carries its own
    preflight and per-arm layer verification.

WHERE IT WRITES
---------------
    runs/fill_<stamp>/<arm>_s<seed>/           checkpoints + eval_noTTA/
    results/fill_<stamp>/registry_fill.csv     one row per run, with verdict
    results/fill_<stamp>/fill_manifest.json    what was attempted and why

A run whose checkpoint fails verification after training is left in place but
its directory is suffixed `__FAILED_VERIFY`, so a later collect step cannot
average it into a table by accident.

USAGE
-----
    python run_fill_missing.py --dry-run          # print the plan, train nothing
    python run_fill_missing.py                    # fill everything missing
    python run_fill_missing.py --arms experts8_top2
    python run_fill_missing.py --seeds 42
    python run_fill_missing.py --stamp 20261020   # resume an earlier fill

Re-running is safe: a target that already trained and verified inside the fill
root is skipped, and a half-finished one resumes from its own best.pt.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fill")

HERE = Path(__file__).resolve().parent
# This file is expected to sit next to run_moe_full.py (scripts/) or at the
# repo root; try both so it works either way.
CANDIDATE_REPOS = [HERE.parent, HERE]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def locate() -> tuple[Path, object, object]:
    """Find the repo root and import run_moe_full and moe_status from it."""
    for repo in CANDIDATE_REPOS:
        rmf_path = repo / "scripts" / "run_moe_full.py"
        if not rmf_path.is_file():
            rmf_path = repo / "run_moe_full.py"
        status_path = repo / "moe_status.py"
        if rmf_path.is_file() and status_path.is_file():
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            if str(repo / "scripts") not in sys.path:
                sys.path.insert(0, str(repo / "scripts"))
            status = _load("moe_status_mod", status_path)
            rmf = _load("run_moe_full_mod", rmf_path)
            return repo, rmf, status
    raise SystemExit(
        "Could not find run_moe_full.py and moe_status.py. Put this script in "
        "the repo root or in scripts/, next to run_moe_full.py."
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def read_keys_fast(repo: Path, ckpt: Path):
    """Tensor names without loading tensors; falls back to torch.load."""
    try:
        ck = _load("ckpt_keys_mod", repo / "ckpt_keys.py")
        keys, _shapes, _how = ck.read_keys(str(ckpt))
        return list(keys)
    except Exception:
        try:
            import torch
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            return list(sd.keys())
        except Exception as exc:                                  # noqa: BLE001
            raise RuntimeError(f"checkpoint unreadable: {exc}") from exc


def verdict(repo: Path, status_mod, arm: str, ckpt: Path) -> tuple[bool, list[str]]:
    """True when the checkpoint carries exactly the tensors the arm implies."""
    req, forbid, n_exp = status_mod.requirements(arm)
    try:
        keys = read_keys_fast(repo, ckpt)
    except RuntimeError as exc:
        return False, [str(exc)]
    problems = []
    for p in req:
        if not any(k.startswith(p) for k in keys):
            problems.append(f"MISSING {p}*")
    for p in forbid:
        if any(k.startswith(p) for k in keys):
            problems.append(f"UNEXPECTED {p}*")
    if n_exp is not None:
        import re
        idx = {int(m.group(1)) for k in keys
               for m in [re.match(r"mlp\.experts\.(\d+)\.", k)] if m}
        if idx and max(idx) + 1 != n_exp:
            problems.append(f"expert count {max(idx) + 1}, name implies {n_exp}")
    return (not problems), problems


def audit(repo: Path, rmf, status_mod, arms: list[str], seeds: list[int]):
    """Classify every (arm, seed) already on disk: ok | invalid | absent."""
    roots = [repo / "runs"] + [Path(p) for p in getattr(rmf, "LEGACY_ROOTS", [])]
    roots = [r for r in dict.fromkeys(roots) if r.exists()]
    subs = ["moe_full", "moe", "."]
    rows = []
    for arm in arms:
        names = [arm] + list(getattr(status_mod, "LEGACY_NAMES", {}).get(arm, []))
        for seed in seeds:
            found, state = status_mod.find(roots, subs, names, seed)
            if found is None or state is None:
                rows.append(dict(arm=arm, seed=seed, state="absent",
                                 path="", problems=""))
                continue
            ckpt = found / "best.pt"
            if not ckpt.is_file():
                rows.append(dict(arm=arm, seed=seed, state="absent",
                                 path=str(found), problems="no best.pt"))
                continue
            ok, problems = verdict(repo, status_mod, arm, ckpt)
            rows.append(dict(arm=arm, seed=seed,
                             state="ok" if ok else "invalid",
                             path=str(found), problems="; ".join(problems)))
    return rows


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", default=date.today().strftime("%Y%m%d"),
                    help="names runs/fill_<stamp>/ ; reuse to resume")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arm names")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit-only", action="store_true",
                    help="write the audit table and stop")
    args = ap.parse_args()

    repo, rmf, status_mod = locate()
    log.info(f"repo            {repo}")

    all_arms = list(getattr(rmf, "ARMS", {}))
    seeds = args.seeds or list(getattr(rmf, "SEEDS", [42, 43, 44]))
    arms = args.arms or all_arms
    unknown = [a for a in arms if a not in all_arms]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}\nknown: {sorted(all_arms)}")

    out_root = repo / "runs" / f"fill_{args.stamp}"
    res_root = repo / "results" / f"fill_{args.stamp}"

    # --- guard: never write ONTO or INSIDE an existing run/results tree -----
    # `runs/` and `results/` are plain containers: a NEW subdirectory under
    # them is the intended destination. Only the populated trees inside them
    # are off limits. (run_moe_full.LEGACY_ROOTS lists repo/"runs" itself, so
    # it must not be used for the containment test.)
    containers = {(repo / "runs").resolve(), (repo / "results").resolve()}
    no_write_onto = [repo / "runs", repo / "results",
                     repo / "runs" / "moe_full", repo / "runs" / "moe",
                     repo / "results" / "moe"]
    no_write_onto += [Path(p) for p in getattr(rmf, "LEGACY_ROOTS", [])]
    no_write_inside = [p for p in no_write_onto
                       if p.resolve() not in containers]

    dest = out_root.resolve()
    for p in no_write_onto:
        try:
            if dest == p.resolve():
                raise SystemExit(f"refusing to write onto an existing tree: {p}")
        except OSError:
            continue
    for p in no_write_inside:
        try:
            if p.resolve() in dest.parents:
                raise SystemExit(f"refusing to write inside an existing tree: {p}")
        except OSError:
            continue
    # Writing into a directory this script did not create would be a surprise.
    if out_root.exists() and any(out_root.iterdir()) \
            and not (res_root / "fill_manifest.json").is_file() \
            and not any(out_root.glob("*/fill_status.json")):
        raise SystemExit(
            f"{out_root} already exists and was not created by this script.\n"
            f"Pass a different --stamp, or move that directory aside.")
    log.info(f"fill root       {out_root}")
    log.info(f"results         {res_root}")

    # --- audit --------------------------------------------------------------
    log.info("auditing what is already on disk ...")
    rows = audit(repo, rmf, status_mod, arms, seeds)
    n_ok = sum(r["state"] == "ok" for r in rows)
    targets = [r for r in rows if r["state"] != "ok"]
    log.info(f"{n_ok} of {len(rows)} runs already valid; {len(targets)} to fill")
    for r in targets:
        log.info(f"   FILL {r['arm']:20s} s{r['seed']}  ({r['state']}"
                 + (f": {r['problems']}" if r["problems"] else "") + ")")

    if not args.dry_run:
        res_root.mkdir(parents=True, exist_ok=True)
        with open(res_root / "audit_before.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["arm", "seed", "state", "path", "problems"])
            w.writeheader()
            w.writerows(rows)
        log.info(f"wrote {res_root / 'audit_before.csv'}")

    if args.audit_only or args.dry_run or not targets:
        if not targets:
            log.info("nothing to do -- every requested run is valid")
        return 0

    # --- force fresh training into the fill root ---------------------------
    # This is the whole point: run_arm would otherwise reuse the very
    # directories that failed the check.
    out_root.mkdir(parents=True, exist_ok=True)
    rmf.RUNS = out_root
    rmf.find_existing = lambda arm, seed: None

    done = []
    for r in targets:
        arm, seed = r["arm"], int(r["seed"])
        tier_cfg = rmf.ARMS[arm]
        cfg = tier_cfg[1] if isinstance(tier_cfg, tuple) else tier_cfg
        dest = out_root / f"{arm}_s{seed}"
        if (dest / "best.pt").is_file():
            ok, problems = verdict(repo, status_mod, arm, dest / "best.pt")
            if ok and (dest / "eval_noTTA" / "test_metrics.csv").is_file():
                log.info(f"SKIP  {arm} s{seed} -- already filled and verified")
                done.append(dict(arm=arm, seed=seed, path=str(dest),
                                 verified=True, problems=""))
                continue
            log.info(f"RESUME {arm} s{seed} -- best.pt present, re-evaluating")

        log.info("=" * 74)
        log.info(f"TRAIN {arm} seed={seed}  ->  {dest}")
        try:
            path = Path(rmf.run_arm(arm, cfg, seed))
        except Exception as exc:                                  # noqa: BLE001
            log.error(f"  training failed: {type(exc).__name__}: {exc}")
            done.append(dict(arm=arm, seed=seed, path=str(dest),
                             verified=False, problems=f"train failed: {exc}"))
            continue

        ok, problems = verdict(repo, status_mod, arm, path / "best.pt")
        if not ok:
            bad = path.with_name(path.name + "__FAILED_VERIFY")
            try:
                path.rename(bad)
                path = bad
            except OSError:
                pass
            log.error(f"  VERIFY FAILED: {'; '.join(problems)}")
            log.error(f"  quarantined at {path}")
        else:
            log.info(f"  verified OK  {path}")
        (path / "fill_status.json").write_text(json.dumps(
            dict(arm=arm, seed=seed, verified=ok, problems=problems,
                 config={k: v for k, v in cfg.items()}), indent=2))
        done.append(dict(arm=arm, seed=seed, path=str(path),
                         verified=ok, problems="; ".join(problems)))

    # --- registry -----------------------------------------------------------
    reg = res_root / "registry_fill.csv"
    with open(reg, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed", "path", "verified", "problems"])
        w.writeheader()
        w.writerows(done)
    (res_root / "fill_manifest.json").write_text(json.dumps(
        dict(stamp=args.stamp, repo=str(repo), out_root=str(out_root),
             seeds=seeds, requested_arms=arms,
             attempted=len(targets),
             verified=sum(d["verified"] for d in done)), indent=2))

    good = sum(d["verified"] for d in done)
    log.info("=" * 74)
    log.info(f"filled {good} of {len(targets)} runs; registry at {reg}")
    if good < len(targets):
        log.warning("some runs did not verify -- they are suffixed "
                    "__FAILED_VERIFY and must not be collected")
    log.info("next: re-run your collect step with this fill root added, then "
             "regenerate the bootstrap tables.")
    return 0 if good == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
