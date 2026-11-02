"""
ddgBenchDataset
===============
Standard ΔΔG benchmarks: Ssym (direct + inverse) and S669.

Required files (relative to `data_root` or supplied directly):
    Ssym :
        data/dataset/ssym/pdb/<id>.pdb
        data/dataset/ssym/ssym-5fold_clean_dir.csv
        data/dataset/ssym/ssym-5fold_clean_inv.csv
    S669 :
        data/dataset/S669/pdb/<id>.pdb
        data/dataset/S669/s669_clean_dir.csv

CSV schema (matches SPURS / ThermoMPNN format):
    PDB : e.g. '1aar_A'      (4-letter PDB id + chain letter)
    MUT : e.g. 'T13A'
    DDG : float, ΔΔG in kcal/mol  (sign convention from source; we negate
                                   to keep the SPURS convention)
    SEQ : str (full chain sequence, optional)
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from math import isnan

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..pdb_parser import alt_parse_PDB
from ..featurizer import get_pdb

log = logging.getLogger(__name__)

ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"


class ddgBenchDataset(Dataset):
    """
    Args
    ----
    pdb_dir : str
        Directory of PDB files for this benchmark.
    csv_fname : str
        Path to the mutations CSV.
    dataset_name : str
        Tag stored in `protein['dataset']`. Set to 'S669', 'ssym_dir', or 'ssym_inv'.
    """

    def __init__(self, pdb_dir: str, csv_fname: str, dataset_name: str):
        self.pdb_dir      = pdb_dir
        self.dataset_name = dataset_name

        df = pd.read_csv(csv_fname)
        self.df = df
        self.wt_names = df.PDB.unique()

        self.wt_seqs:  dict = {}
        self.mut_rows: dict = {}
        for wt_name in self.wt_names:
            wt_name_query    = wt_name
            wt_name_stripped = wt_name[:-1]   # drop chain letter
            self.mut_rows[wt_name_stripped] = (
                df.query("PDB == @wt_name_query").reset_index(drop=True)
            )
            if "S669" not in self.pdb_dir:
                self.wt_seqs[wt_name_stripped] = self.mut_rows[wt_name_stripped].SEQ[0]

        self.structure_path = pdb_dir
        # cache: name -> chain -> parsed pdb list
        self.json_dataset   = defaultdict(lambda: defaultdict(lambda: -1))

    def __len__(self) -> int:
        return len(self.wt_names)

    def __getitem__(self, index: int):
        wt_name_full = self.wt_names[index]
        chain        = [wt_name_full[-1]]
        wt_name      = wt_name_full.split(".pdb")[0][:-1]
        mut_data     = self.mut_rows[wt_name]

        if isinstance(self.json_dataset[wt_name][chain[0]], int):
            pdb = alt_parse_PDB(os.path.join(self.pdb_dir, wt_name + ".pdb"), chain)
            self.json_dataset[wt_name][chain[0]] = pdb
        pdb       = self.json_dataset[wt_name][chain[0]]
        resn_list = pdb[0]["resn_list"]

        protein = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)

        for _, row in mut_data.iterrows():
            mut_info = row.MUT
            wt_aa, mut_aa = mut_info[0], mut_info[-1]
            try:
                pos     = mut_info[1:-1]
                pdb_idx = resn_list.index(pos)
            except ValueError:
                continue

            try:
                assert pdb[0]["seq"][pdb_idx] == wt_aa
            except AssertionError:
                # Handle gaps in parsed sequence
                if "S669" in self.pdb_dir:
                    gaps = [g for g in pdb[0]["seq"] if g == "-"]
                else:
                    gaps = [g for g in pdb[0]["seq"][: pdb_idx + 10] if g == "-"]
                pdb_idx += (len(gaps) if gaps else 1)
                if pdb_idx is None or pdb[0]["seq"][pdb_idx] != wt_aa:
                    continue

            if row.DDG is None or isnan(row.DDG):
                ddG = None
            else:
                ddG = torch.tensor([row.DDG * -1.0], dtype=torch.float32)

            wt_onehot = torch.zeros(21); wt_onehot[ALPHABET_21.index(wt_aa)]  = 1
            mt_onehot = torch.zeros(21); mt_onehot[ALPHABET_21.index(mut_aa)] = 1
            append_tensor = torch.cat([wt_onehot, mt_onehot]).float()

            protein["mut_ids"].append(pdb_idx)
            protein["ddG"].append(ddG)
            protein["append_tensors"].append(append_tensor)

        protein["ddG"]            = torch.stack(protein["ddG"])
        protein["append_tensors"] = torch.stack(protein["append_tensors"])
        protein["pdb_path"]       = self.structure_path
        protein["dataset"]        = self.dataset_name
        return protein

    @staticmethod
    def collate_fn(batch):
        return batch[0]
