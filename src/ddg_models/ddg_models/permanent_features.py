"""
permanent_features.py
=====================
Integrates quantum-permanent structural features into the MultimodalDDG
predictor as an additional per-protein modality.

Design (chosen to be minimal and correct):
  - Permanent features are a PER-PROTEIN 210-dim vector (they depend only
    on the wild-type structure, not on the specific mutation).
  - They are precomputed ONCE per protein and cached to disk, so training
    does not pay the permanent cost every step.
  - At forward time they are broadcast across the K mutations of a protein
    and concatenated onto muted_id_representation before the MLP head.

This file provides:
  PermanentFeatureCache   -- fit the extractor on TRAIN structures, transform
                             any protein, cache by protein name.
  inject_permanent(batch, cache, dim) -- add batch["perm_features"] (K,210).

The model change (in multimodal_ddg.py) reads batch["perm_features"] and
concatenates it; see integrate_permanent_patch.txt.
"""

from __future__ import annotations
import logging, os, pickle
from pathlib import Path
import numpy as np
import torch

log = logging.getLogger(__name__)

PERM_DIM = 210


class PermanentFeatureCache:
    """Fit-on-train, transform-any permanent feature provider with disk cache."""

    def __init__(self, cache_path: str, n_patterns: int = PERM_DIM,
                 signature_dim: int = 16, n_eigs: int = 12):
        self.cache_path = Path(cache_path)
        self.n_patterns = n_patterns
        self.signature_dim = signature_dim
        self.n_eigs = n_eigs
        self._extractor = None
        self._cache: dict[str, np.ndarray] = {}
        if self.cache_path.is_file():
            with open(self.cache_path, "rb") as f:
                blob = pickle.load(f)
            self._cache = blob["cache"]
            self._extractor = blob["extractor"]
            log.info(f"Loaded permanent cache: {len(self._cache)} proteins from {self.cache_path}")

    def fit(self, train_coords_by_name: dict[str, np.ndarray]):
        """Fit the permanent extractor on training structures (once)."""
        from quantum_permanent.protein import StructureEncoder, StructurePermanentExtractor
        enc = StructureEncoder(signature_dim=self.signature_dim,
                               n_eigs=self.n_eigs, mode="soft")
        self._extractor = StructurePermanentExtractor(
            enc, n_modes=10, n_photons=4, n_patterns=self.n_patterns,
            backend="ryser", data_encoder_method="pca", scale=0.4,
            rng=np.random.default_rng(0),
        )
        names = list(train_coords_by_name.keys())
        coords = [train_coords_by_name[n] for n in names]
        self._extractor.fit(coords)
        feats = self._extractor.transform(coords)
        for n, f in zip(names, feats):
            self._cache[n] = f.astype(np.float32)
        log.info(f"Fitted permanent extractor on {len(names)} train proteins")
        self._save()

    def transform_one(self, name: str, coords: np.ndarray) -> np.ndarray:
        """Return the 210-dim permanent feature for one protein, caching it."""
        if name in self._cache:
            return self._cache[name]
        if self._extractor is None:
            raise RuntimeError("PermanentFeatureCache.fit must be called before transform")
        f = self._extractor.transform([coords])[0].astype(np.float32)
        self._cache[name] = f
        return f

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"cache": self._cache, "extractor": self._extractor}, f)
        log.info(f"Saved permanent cache ({len(self._cache)} proteins) to {self.cache_path}")


def _ca_from_batch(batch) -> np.ndarray | None:
    """Extract CA coords (L,3) from a model batch's X tensor (1,L,4,3)."""
    X = batch.get("X", None)
    if X is None:
        return None
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    if X.ndim == 4:
        X = X[0]
    if X.ndim == 3 and X.shape[1] >= 2:
        ca = X[:, 1, :]
    elif X.ndim == 2 and X.shape[-1] == 3:
        ca = X
    else:
        return None
    good = np.isfinite(ca).all(axis=1) & (np.abs(ca).sum(axis=1) > 0)
    ca = ca[good]
    return ca if len(ca) >= 12 else None


def inject_permanent(batch, cache: PermanentFeatureCache, device, dim: int = PERM_DIM):
    """Attach batch['perm_features'] of shape (K, dim), broadcast across the
    K mutations of this protein. Falls back to zeros if unavailable."""
    name = None
    for key in ("wt_name", "name", "pdb_path", "dataset"):
        if key in batch and isinstance(batch[key], str):
            name = batch[key]
            break
    ddg = batch.get("ddG", None)
    K = (ddg.shape[0] if hasattr(ddg, "shape") else 1)

    feat = None
    if name is not None and name in cache._cache:
        feat = cache._cache[name]
    else:
        ca = _ca_from_batch(batch)
        if ca is not None and cache._extractor is not None:
            try:
                feat = cache.transform_one(name or f"anon_{id(batch)}", ca)
            except Exception as e:
                log.warning(f"permanent transform failed: {e}")
    if feat is None:
        feat = np.zeros(dim, dtype=np.float32)

    t = torch.from_numpy(np.asarray(feat, dtype=np.float32)).to(device)
    batch["perm_features"] = t.unsqueeze(0).expand(K, -1).contiguous()
    return batch
