"""
ddgGeoDataset
=============
Geometric-aware ΔΔG / ΔTm datasets from GeoStab (Xu et al.).

Supports:
    S461, S783, S8754, S2648 (ΔΔG)
    S571, S4346              (ΔTm — uses `DTM` column instead of `DDG`)

CSV schema:
    PDB   : e.g. '1aar'
    chain : e.g. 'A'
    MUT   : e.g. 'T13A'
    DDG   : ΔΔG (kcal/mol)         — for ΔΔG datasets
    DTM   : ΔTm                    — for ΔTm datasets
    SEQ   : full sequence (only required for Ssym-like datasets)
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from math import isnan

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..pdb_parser import alt_parse_PDB
from ..featurizer import get_pdb

log = logging.getLogger(__name__)

ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"


class ddgGeoDataset(Dataset):
    """
    Args
    ----
    pdb_dir : str
        Directory of PDB files.
    csv_fname : str
        Mutations CSV. Must have at minimum: PDB, chain, MUT, DDG or DTM.
    dataset_name : str
        Tag stored in `protein['dataset']`.
    stage : str
        'full', 'train', or 'test'. Controls batching for `mut_seq` mode.
    mut_seq : bool
        If True, also build mutant sequences per row (slower).
    """

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
        self.mut_seq      = mut_seq

        df = pd.read_csv(csv_fname)
        df.PDB = df.PDB + df.chain
        self.df = df

        self.wt_names = df.PDB.unique()
        self.wt_names = [x for x in self.wt_names if str(x) != "nan"]

        self.mut_rows: dict = {}
        self.wt_seqs:  dict = {}
        for wt_name in self.wt_names:
            wt_name_query = wt_name
            self.mut_rows[wt_name] = (
                df.query("PDB == @wt_name_query").reset_index(drop=True)
            )
            if "ssym" in self.pdb_dir:
                self.wt_seqs[wt_name] = self.mut_rows[wt_name].SEQ[0]

        len_arr      = [len(self.mut_rows[w]) for w in self.wt_names]
        total_muts   = sum(len_arr)
        log.info(f"{dataset_name}: {len(self.wt_names)} proteins, {total_muts} mutations")
        self.json_dataset = defaultdict(lambda: defaultdict(lambda: -1))

        self.fake_bs = 32 if stage == "train" else 10000
        if self.mut_seq:
            self.proteins  = [self._get_wt_item(i) for i in tqdm(range(len(self.wt_names)))]
            mut_numbers    = [
                math.ceil(len(self.mut_rows[self.wt_names[i][:4]]) / self.fake_bs)
                for i in range(len(self.wt_names))
            ]
            self.index_list  = []
            self.start_index = []
            for cur, num in enumerate(mut_numbers):
                self.index_list  += [cur] * num
                self.start_index += [i * self.fake_bs for i in range(num)]
            self.dataset_len = sum(mut_numbers)
            self.mut_numbers = mut_numbers
        self.protein_index_list = [np.arange(i) for i in len_arr]

    def __len__(self) -> int:
        return len(self.wt_names) if not self.mut_seq else self.dataset_len

    def __getitem__(self, index: int):
        return self._get_wt_item(index)

    def _get_wt_item(self, index: int):
        wt_name_full = self.wt_names[index]
        chain        = [wt_name_full[-1]]
        wt_name      = wt_name_full.split(".pdb")[0]
        mut_data     = self.mut_rows[wt_name]
        wt_name      = wt_name[:-1]

        if isinstance(self.json_dataset[wt_name][chain[0]], int):
            pdb = alt_parse_PDB(os.path.join(self.pdb_dir, wt_name + ".pdb"), chain)
            self.json_dataset[wt_name][chain[0]] = pdb
        pdb       = self.json_dataset[wt_name][chain[0]]
        resn_list = pdb[0]["resn_list"]

        protein = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)
        if self.mut_seq:
            protein["S"] = [protein["S"]]

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
                if "S669" in self.pdb_dir:
                    gaps = [g for g in pdb[0]["seq"] if g == "-"]
                else:
                    gaps = [g for g in pdb[0]["seq"][: pdb_idx + 10] if g == "-"]
                pdb_idx += (len(gaps) if gaps else 1)
                if pdb_idx is None or pdb[0]["seq"][pdb_idx] != wt_aa:
                    continue

            if self.mut_seq:
                pdb_seq_old      = pdb[0]["seq"]
                pdb[0]["seq"]    = pdb[0]["seq"][:pdb_idx] + mut_aa + pdb[0]["seq"][pdb_idx + 1 :]
                protein["mut_seq"].append(pdb[0]["seq"])
                mut_protein      = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)
                protein["S"].append(mut_protein["S"])
                pdb[0]["seq"]    = pdb_seq_old

            # ΔTm uses 'DTM' column, ΔΔG uses 'DDG'
            if "DTM" in row:
                ddG = torch.tensor([row.DTM * -1.0], dtype=torch.float32)
            else:
                ddG = None if row.DDG is None or isnan(row.DDG) else torch.tensor(
                    [row.DDG * -1.0], dtype=torch.float32
                )

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

        if self.mut_seq:
            n = len(protein["S"])
            protein["S"]                   = torch.cat(protein["S"], dim=0).clone()
            protein["X"]                   = protein["X"].expand(n, -1, -1, -1).clone()
            protein["mask"]                = protein["mask"].expand(n, -1).clone()
            protein["chain_M"]             = protein["chain_M"].expand(n, -1).clone()
            protein["chain_M_chain_M_pos"] = protein["chain_M_chain_M_pos"].expand(n, -1).clone()
            protein["residue_idx"]         = protein["residue_idx"].expand(n, -1).clone()
            protein["chain_encoding_all"]  = protein["chain_encoding_all"].expand(n, -1).clone()
            protein["randn_1"]             = protein["randn_1"].expand(n, -1).clone()

        protein["ddG"]            = torch.stack(protein["ddG"])
        protein["append_tensors"] = torch.stack(protein["append_tensors"])
        protein["dataset"]        = self.dataset_name
        return protein

    @staticmethod
    def collate_fn(batch):
        return batch[0]
