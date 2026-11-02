"""
DomainomeDataset
================
Human Domainome dataset (Beltran et al. 2025).

Required:
    pdb_dir: directory of <domain_ID>.pdb files
    csv_fname: CSV with columns
        domain_ID, variant_ID (e.g. 'XXX_T13A'), pdb_pos (int),
        aPCA_normalized_fitness (used as a fitness proxy for ΔΔG-like ranking)
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..pdb_parser import alt_parse_PDB
from ..featurizer import get_pdb

log = logging.getLogger(__name__)

ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"


class DomainomeDataset(Dataset):
    def __init__(
        self,
        pdb_dir: str,
        csv_fname: str,
        dataset_name: str,
        stage: str = "full",
        mut_seq: bool = False,
        train_size: float = 1.0,
    ):
        self.pdb_dir      = pdb_dir
        self.dataset_name = dataset_name

        df = pd.read_csv(csv_fname).dropna(subset=["aPCA_fitness"])
        self.df = df

        self.wt_names = df.domain_ID.unique()
        self.wt_names = [x for x in self.wt_names if str(x) != "nan"]

        self.mut_rows: dict = {}
        for wt_name in self.wt_names:
            self.mut_rows[wt_name] = (
                df.query("domain_ID == @wt_name").reset_index(drop=True)
            )
        self.json_dataset = defaultdict(lambda: defaultdict(lambda: -1))

    def __len__(self) -> int:
        return len(self.wt_names)

    def __getitem__(self, index: int):
        return self._get_wt_item(index)

    def _get_wt_item(self, index: int):
        wt_name  = self.wt_names[index]
        chain    = "A"
        mut_data = self.mut_rows[wt_name]

        if isinstance(self.json_dataset[wt_name][chain[0]], int):
            pdb = alt_parse_PDB(os.path.join(self.pdb_dir, wt_name + ".pdb"), chain)
            self.json_dataset[wt_name][chain[0]] = pdb
        pdb       = self.json_dataset[wt_name][chain[0]]

        protein = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)

        for _, row in mut_data.iterrows():
            mut_info = row.variant_ID.split("_")[-1]
            wt_aa, mut_aa = mut_info[0], mut_info[-1]
            pdb_idx = row.pdb_pos
            assert pdb[0]["seq"][pdb_idx] == wt_aa

            ddG = torch.tensor([row.aPCA_normalized_fitness * -1.0], dtype=torch.float32)

            wt_onehot = torch.zeros(21); wt_onehot[ALPHABET_21.index(wt_aa)]  = 1
            mt_onehot = torch.zeros(21); mt_onehot[ALPHABET_21.index(mut_aa)] = 1
            append_tensor = torch.cat([wt_onehot, mt_onehot]).float()

            protein["mut_ids"].append(pdb_idx)
            protein["ddG"].append(ddG)
            protein["append_tensors"].append(append_tensor)

        if len(protein["ddG"]) == 0:
            protein["mut_ids"]        = [1]
            protein["ddG"]            = [torch.tensor([0.0])]
            protein["append_tensors"] = [torch.zeros(42).float()]

        protein["mut_ids"]        = torch.LongTensor(protein["mut_ids"])
        protein["ddG"]            = torch.stack(protein["ddG"])
        protein["append_tensors"] = torch.stack(protein["append_tensors"])
        protein["dataset"]        = self.dataset_name + wt_name
        return protein

    @staticmethod
    def collate_fn(batch):
        return batch[0]
