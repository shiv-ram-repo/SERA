"""
Metrics
=======
Standard ΔΔG-prediction metrics. Filters out sentinel (10000) and non-finite
values before computing.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Returns: spearman, pearson, rmse, mae, r2, n
    """
    p = np.asarray(preds).flatten()
    t = np.asarray(targets).flatten()
    valid = np.isfinite(p) & np.isfinite(t) & (np.abs(p) < 9000)
    n = int(valid.sum())
    if n < 5:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "rmse": float("nan"), "mae": float("nan"),
                "r2": float("nan"), "n": n}

    p, t = p[valid], t[valid]
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    return {
        "spearman": float(spearmanr(p, t)[0]),
        "pearson":  float(pearsonr(p, t)[0]),
        "rmse":     float(np.sqrt(np.mean((p - t) ** 2))),
        "mae":      float(np.mean(np.abs(p - t))),
        "r2":       float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "n":        n,
    }
