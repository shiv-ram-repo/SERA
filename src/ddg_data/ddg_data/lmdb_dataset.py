"""
lmdb_dataset
============
Lightweight LMDB-backed PyTorch Dataset.

Only required if you use LMDB caches; megascale/fireport/etc. do NOT need this.
Kept here for parity with SPURS's interface.
"""

from __future__ import annotations

import pickle
from typing import Any, Tuple

from torch.utils.data import Dataset

try:
    import lmdb
    _HAS_LMDB = True
except ImportError:
    lmdb = None
    _HAS_LMDB = False


class LMDBDataset(Dataset):
    """
    Load an LMDB file into a PyTorch Dataset.

    Parameters
    ----------
    path : str
        Path to the LMDB database directory.
    to_dict : bool
        If True, eagerly load all key/value pairs into a dict.
    to_list : bool
        If True, also build a list of values for sequential access.
    """

    def __init__(self, path: str, to_dict: bool = True, to_list: bool = True):
        super().__init__()
        if not _HAS_LMDB:
            raise ImportError("lmdb is required for LMDBDataset. Install with: pip install lmdb")

        self.lmdb_env, self.lmdb_dict, self.lmdb_list = self._load_lmdb(
            path, to_dict=to_dict, to_list=to_list,
        )

    @staticmethod
    def _load_lmdb(lmdb_path: str, to_dict: bool, to_list: bool) -> Tuple[Any, dict, list]:
        env = lmdb.open(lmdb_path, readonly=True)
        if to_dict:
            with env.begin() as txn:
                d = {key.decode(): pickle.loads(value) for key, value in txn.cursor()}
        else:
            d = {}
        if to_list:
            if to_dict:
                lst = list(d.values())
            else:
                with env.begin() as txn:
                    lst = [pickle.loads(value) for _, value in txn.cursor()]
        else:
            lst = []
        return env, d, lst

    def get_value(self, key: str) -> Any:
        if self.lmdb_dict:
            return self.lmdb_dict[key]
        with self.lmdb_env.begin() as txn:
            return pickle.loads(txn.get(key.encode()))

    def __len__(self) -> int:
        return len(self.lmdb_list)

    def __getitem__(self, idx: int) -> Any:
        return self.lmdb_list[idx]
