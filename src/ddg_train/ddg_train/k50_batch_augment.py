"""
k50_batch_augment
=================
Attaches K50 fields to a SPURS batch on the fly.
"""

from __future__ import annotations
import logging
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

class K50BatchAugmentor:
    K50_COLS = ["log10_K50_t", "log10_K50_c", "log10_K50unfolded_t", "log10_K50unfolded_c"]

    def __init__(self, csv_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"K50 CSV not found: {csv_path}")
        log.info(f"K50 augmentor: reading CSV at {csv_path}")
        
        df_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        seq_col = "aa_seq" if "aa_seq" in df_cols else ("WT_sequence" if "WT_sequence" in df_cols else None)
        
        cols_to_load = ["WT_name", "mut_type"] + self.K50_COLS
        if seq_col:
            cols_to_load.append(seq_col)

        df = pd.read_csv(csv_path, usecols=cols_to_load, low_memory=False)

        self._table: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._wt_table: Dict[str, Dict[str, float]] = {}
        self._seq_to_name: Dict[str, str] = {}

        for row in df.itertuples(index=False):
            wt_name = str(row.WT_name)
            if wt_name.endswith('.pdb'):
                wt_name = wt_name[:-4]  # Force strip .pdb suffix
            
            mut_type = str(row.mut_type)
            entry = {c: float(getattr(row, c)) if _isnum(getattr(row, c)) else float("nan") for c in self.K50_COLS}
            
            self._table[(wt_name, mut_type)] = entry
            if mut_type == "wt":
                self._wt_table[wt_name] = entry
                if seq_col and hasattr(row, seq_col):
                    self._seq_to_name[str(getattr(row, seq_col))] = wt_name

        log.info(f"K50 augmentor: indexed {len(self._table):,} (wt_name, mut_type) rows, {len(self._wt_table):,} WT rows")

    def __call__(self, batch: dict) -> dict:
        name = batch.get("name", None)
        if isinstance(name, (list, tuple)) and len(name) > 0:
            wt_name = str(name[0])
        elif isinstance(name, str):
            wt_name = name
        else:
            wt_name = "?"

        # Sequence Fallback: If name is suspicious, lookup by AA sequence
        if (len(wt_name) <= 1 or wt_name not in self._wt_table) and "aa_seq" in batch:
            batch_seq = batch["aa_seq"]
            batch_seq = str(batch_seq[0]) if isinstance(batch_seq, (list, tuple)) else str(batch_seq)
            if batch_seq in self._seq_to_name:
                wt_name = self._seq_to_name[batch_seq]

        wt_record = self._wt_table.get(wt_name, None)
        if wt_record is None:
            wt_record = {c: float("nan") for c in self.K50_COLS}

        mut_ids = batch.get("mut_ids", [])
        if hasattr(mut_ids, "cpu"):
            mut_ids = mut_ids.cpu().tolist()
        K = len(mut_ids)

        mut_type_list = batch.get("mut_type", None)
        if mut_type_list is None:
            mut_type_list = self._reconstruct_mut_types(batch, K)

        k50_t_wt   = torch.full((K,), float("nan"))
        k50_t_mut  = torch.full((K,), float("nan"))
        k50_c_wt   = torch.full((K,), float("nan"))
        k50_c_mut  = torch.full((K,), float("nan"))
        k50_unf_t_wt  = torch.full((K,), float("nan"))
        k50_unf_t_mut = torch.full((K,), float("nan"))
        k50_unf_c_wt  = torch.full((K,), float("nan"))
        k50_unf_c_mut = torch.full((K,), float("nan"))

        for k in range(K):
            mt = mut_type_list[k] if k < len(mut_type_list) else None            
            mut_record = self._table.get((wt_name, mt), None) if mt else None
            if mut_record is None:
                continue
            k50_t_wt[k]      = wt_record["log10_K50_t"]
            k50_c_wt[k]      = wt_record["log10_K50_c"]
            k50_unf_t_wt[k]  = wt_record["log10_K50unfolded_t"]
            k50_unf_c_wt[k]  = wt_record["log10_K50unfolded_c"]
            k50_t_mut[k]     = mut_record["log10_K50_t"]
            k50_c_mut[k]     = mut_record["log10_K50_c"]
            k50_unf_t_mut[k] = mut_record["log10_K50unfolded_t"]
            k50_unf_c_mut[k] = mut_record["log10_K50unfolded_c"]

        batch["k50_t_wt"] = k50_t_wt
        batch["k50_t_mut"] = k50_t_mut
        batch["k50_c_wt"] = k50_c_wt
        batch["k50_c_mut"] = k50_c_mut
        batch["k50_unf_t_wt"] = k50_unf_t_wt
        batch["k50_unf_t_mut"] = k50_unf_t_mut
        batch["k50_unf_c_wt"] = k50_unf_c_wt
        batch["k50_unf_c_mut"] = k50_unf_c_mut
        return batch

    def _reconstruct_mut_types(self, batch, K) -> list:
        AA_LETTERS = list("ACDEFGHIKLMNPQRSTVWYX")
        mut_ids = batch.get("mut_ids", [])
        if hasattr(mut_ids, "cpu"):
            mut_ids = mut_ids.cpu().tolist()
        append_tensors = batch.get("append_tensors")
        if append_tensors is None or K == 0:
            return [None] * K
        if hasattr(append_tensors, "cpu"):
            wt_idx_per_k  = append_tensors[..., :21].argmax(dim=-1).cpu().tolist()
            mut_idx_per_k = append_tensors[..., 21:42].argmax(dim=-1).cpu().tolist()
        else:
            return [None] * K
        out = []
        for k in range(K):
            try:
                pos1 = int(mut_ids[k]) + 1
                wt_l = AA_LETTERS[int(wt_idx_per_k[k])]
                mu_l = AA_LETTERS[int(mut_idx_per_k[k])]
                out.append(f"{wt_l}{pos1}{mu_l}")
            except (IndexError, ValueError):
                out.append(None)
        return out

def _isnum(v):
    try:
        f = float(v)
        return np.isfinite(f)
    except (ValueError, TypeError):
        return False
