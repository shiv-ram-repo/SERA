"""
MegaScaleDataset
================
Tsuboyama et al. 2023 megascale ΔΔG dataset.

Required files (relative to a `data_root` you pass in):
    data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv
    data/dataset/megascale/mmseq_mut_search_0.25.m8
    data/dataset/megascale/mega_splits.pkl
    data/dataset/megascale/AlphaFold_model_PDBs/<wt_name>.pdb
    data/dataset/megascale/parsed_structure.json   (built on first run)

Faithful reproduction of SPURS's `MegaScaleDataset` class — no spurs imports.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..pdb_parser import parse_pdb_directory_to_json
from ..featurizer import get_pdb

log = logging.getLogger(__name__)

ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"


class MegaScaleDataset(Dataset):
    """
    Args
    ----
    data_root : str
        Root directory whose layout matches SPURS's `data/` folder.
    reduce : str or float
        '' for no reduction;
        'prot' to keep only ~58 proteins (train split only);
        float ∈ (0, 1] to subsample mutations per protein.
    split : str
        One of 'train', 'val', 'test', 'all', 'train_s669',
        or any 'cv_{train,val,test}_{0..4}' split present in the splits pickle.
    """

    def __init__(
        self,
        data_root: str,
        reduce: str = "",
        split: str = "train",
        single_mut: bool = False,
        mut_seq: bool = False,
        std_ratio: float = 0.75,
        loss_ratio: float = 1.0,
        train_ratio: float = 0.05,
    ):
        self.data_root  = data_root
        self.split      = split
        self.single_mut = single_mut
        self.mut_seq    = mut_seq
        self.std_ratio  = std_ratio
        self.loss_ratio = loss_ratio
        self.train_ratio = train_ratio

        # Load main CSV
        csv_path = os.path.join(
            data_root, "data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
        )
        df = pd.read_csv(csv_path, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"])

        # Drop unreliable rows + complex mutation strings (insertions/deletions/multi)
        df = df.loc[df.ddG_ML != "-", :].reset_index(drop=True)
        df = df.loc[
            ~df.mut_type.str.contains("ins")
            & ~df.mut_type.str.contains("del")
            & ~df.mut_type.str.contains(":"),
            :,
        ].reset_index(drop=True)
        self.df = df

        # Apply MMseqs2 sequence-similarity filter (for non-test splits only)
        if self.split != "test":
            mmseq_path = os.path.join(
                data_root, "data/dataset/megascale/mmseq_mut_search_0.25.m8"
            )
            drop_idx = []
            with open(mmseq_path, "r") as f:
                for line in f:
                    drop_idx.append(int(line.split("\t")[1]))
            previous = len(df)
            df = df.loc[~df.index.isin(drop_idx), :].reset_index(drop=True)
            log.info(f"MMseqs2 filter removed {previous - len(df)} rows")

        # Load split pickle
        splits_path = os.path.join(data_root, "data/dataset/megascale/mega_splits.pkl")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)

        self.split_wt_names = {}
        if self.split == "all":
            self.split_wt_names[self.split] = np.concatenate(
                [splits["train"], splits["val"], splits["test"]]
            )
        elif reduce == "prot" and split == "train":
            self.split_wt_names[self.split] = np.random.choice(splits["train"], 58)
        else:
            self.split_wt_names[self.split] = splits[self.split]
        self.wt_names = self.split_wt_names[self.split]

        # Build per-wt mutation rows + wt sequence
        self.wt_seqs:  dict = {}
        self.mut_rows: dict = {}
        removed = []
        for wt_name in tqdm(self.wt_names, desc="Indexing wt entries"):
            wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
            self.mut_rows[wt_name] = (
                df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
            )
            if isinstance(reduce, float) and self.split == "train":
                self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(
                    frac=float(reduce), replace=False
                )
            if len(wt_rows) == 0:
                removed.append(wt_name)
            else:
                self.wt_seqs[wt_name] = wt_rows.aa_seq[0]
        self.wt_names = list(set(self.wt_names) - set(removed))
        log.info(f"Removed {len(removed)} wt names without WT row")

        # Build / load parsed structure cache
        self.structure_path = os.path.join(data_root, "data/dataset/megascale/AlphaFold_model_PDBs/")
        json_path = os.path.join(
            data_root, "data/dataset/megascale/parsed_structure.json"
        )
        parse_pdb_directory_to_json(self.structure_path, json_path)
        log.info("Loading parsed structures JSON")
        with open(json_path, "r") as f:
            self.json_dataset = json.load(f)

    def __len__(self) -> int:
        return len(self.wt_names)

    def __getitem__(self, index: int):
        return self._get_wt_item(index)

    def _get_wt_item(self, index: int):
        wt_name  = self.wt_names[index]
        wt_seq   = self.wt_seqs[wt_name]
        mut_data = self.mut_rows[wt_name]

        # Megascale file names sometimes contain "|" — strip and reformat
        wt_name_clean = wt_name.split(".pdb")[0].replace("|", ":")

        pdb     = self.json_dataset[wt_name_clean]
        protein = get_pdb(pdb, wt_seq, wt_name_clean)

        if self.mut_seq:
            protein["S"] = [protein["S"]]

        for i in range(len(mut_data)):
            row = mut_data.iloc[i]
            if self.mut_seq:
                pdb["seq"] = row.aa_seq
                mut_protein = get_pdb(pdb, row.aa_seq, wt_name_clean)
                protein["S"].append(mut_protein["S"])

            if ("ins" in row.mut_type or "del" in row.mut_type or ":" in row.mut_type):
                return None
            assert len(row.aa_seq) == len(wt_seq)

            wt_aa  = row.mut_type[0]
            mut_aa = row.mut_type[-1]
            mut_id = int(row.mut_type[1:-1]) - 1
            assert wt_seq[mut_id]      == wt_aa
            assert row.aa_seq[mut_id]  == mut_aa
            if row.ddG_ML == "-":
                return None
            ddG = -torch.tensor([float(row.ddG_ML)], dtype=torch.float32)

            wt_onehot  = torch.zeros(21); wt_onehot[ALPHABET_21.index(wt_aa)]  = 1
            mt_onehot  = torch.zeros(21); mt_onehot[ALPHABET_21.index(mut_aa)] = 1
            append_tensor = torch.cat([wt_onehot, mt_onehot]).float()

            protein["mut_ids"].append(mut_id)
            protein["ddG"].append(ddG)
            protein["append_tensors"].append(append_tensor)
            protein["mut_seq"].append(row.aa_seq)

        protein["ddG"]            = torch.stack(protein["ddG"]).to(protein["X"].device, non_blocking=True)
        protein["append_tensors"] = torch.stack(protein["append_tensors"])
        protein["dataset"]        = "megascale"
        protein["pdb_path"]       = self.structure_path

        if self.mut_seq:
            protein["S"]                   = torch.cat(protein["S"], dim=0).clone()
            n = len(protein["S"])
            protein["X"]                   = protein["X"].expand(n, -1, -1, -1).clone()
            protein["mask"]                = protein["mask"].expand(n, -1).clone()
            protein["chain_M"]             = protein["chain_M"].expand(n, -1).clone()
            protein["chain_M_chain_M_pos"] = protein["chain_M_chain_M_pos"].expand(n, -1).clone()
            protein["residue_idx"]         = protein["residue_idx"].expand(n, -1).clone()
            protein["chain_encoding_all"]  = protein["chain_encoding_all"].expand(n, -1).clone()
            protein["randn_1"]             = protein["randn_1"].expand(n, -1).clone()

        protein["std_ratio"]  = self.std_ratio
        protein["loss_ratio"] = self.loss_ratio
        return protein

    @staticmethod
    def collate_fn(batch):
        """Identity collate — return the single-protein dict directly."""
        return batch[0]
