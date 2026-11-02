"""
FireProtDataset
===============
FireProtDB stability dataset (Stourac et al.).

Required files (relative to `data_root`):
    data/dataset/fireprot/fireprot_upload/csvs/4_fireprotDB_bestpH.csv
    data/dataset/fireprot/fireprot_upload/csvs/fireprot_splits.pkl
    data/dataset/fireprot/fireprot_upload/pdbs/<wt_name>.pdb
    data/dataset/fireprot/fireprot_upload/parsed_structure.json   (auto-built)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from math import isnan

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..pdb_parser import parse_pdb_directory_to_json
from ..featurizer import get_pdb

log = logging.getLogger(__name__)

ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"


class FireProtDataset(Dataset):
    def __init__(self, data_root: str, split: str = "homologue-free"):
        self.data_root = data_root
        self.split     = split

        csv_path = os.path.join(
            data_root, "data/dataset/fireprot/fireprot_upload/csvs/4_fireprotDB_bestpH.csv"
        )
        df = pd.read_csv(csv_path).dropna(subset=["ddG"])
        df = df.where(pd.notnull(df), None)

        seq_key = "pdb_sequence"
        self.seq_to_data = {
            wt_seq: df.query(f"{seq_key} == @wt_seq").reset_index(drop=True)
            for wt_seq in df[seq_key].unique()
        }
        self.df = df

        splits_path = os.path.join(
            data_root, "data/dataset/fireprot/fireprot_upload/csvs/fireprot_splits.pkl"
        )
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)

        if split == "all":
            all_names = [n for sub in splits.values() for n in sub]
            self.wt_names = all_names
        else:
            self.wt_names = splits[split]

        self.wt_seqs:  dict = {}
        self.mut_rows: dict = {}
        for wt_name in self.wt_names:
            self.mut_rows[wt_name] = (
                df.query("pdb_id_corrected == @wt_name").reset_index(drop=True)
            )
            self.wt_seqs[wt_name] = self.mut_rows[wt_name].pdb_sequence[0]

        self.structure_path = os.path.join(
            data_root, "data/dataset/fireprot/fireprot_upload/pdbs/"
        )
        json_path = os.path.join(
            data_root, "data/dataset/fireprot/fireprot_upload/parsed_structure.json"
        )
        parse_pdb_directory_to_json(self.structure_path, json_path)
        with open(json_path, "r") as f:
            self.json_dataset = json.load(f)

    def __len__(self) -> int:
        return len(self.wt_names)

    def __getitem__(self, index: int):
        wt_name = self.wt_names[index]
        seq     = self.wt_seqs[wt_name]
        data    = self.seq_to_data[seq]

        pdb     = self.json_dataset[data.pdb_id_corrected[0]]
        protein = get_pdb(pdb, seq, wt_name, check_assert=False)

        for _, row in data.iterrows():
            pdb_idx = row.pdb_position
            assert pdb["seq"][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]
            pdb["seq"] = pdb["seq"].replace("-", "X")

            wt_aa  = row.wild_type
            mut_aa = row.mutation
            ddG = None if row.ddG is None or isnan(row.ddG) else torch.tensor(
                [row.ddG], dtype=torch.float32
            )

            wt_onehot = torch.zeros(21); wt_onehot[ALPHABET_21.index(wt_aa)]  = 1
            mt_onehot = torch.zeros(21); mt_onehot[ALPHABET_21.index(mut_aa)] = 1
            append_tensor = torch.cat([wt_onehot, mt_onehot]).float()

            protein["mut_ids"].append(pdb_idx)
            protein["ddG"].append(ddG)
            protein["append_tensors"].append(append_tensor)

        protein["ddG"]            = torch.stack(protein["ddG"])
        protein["append_tensors"] = torch.stack(protein["append_tensors"])
        protein["pdb_path"]       = self.structure_path
        protein["dataset"]        = "fHF"
        return protein

    @staticmethod
    def collate_fn(batch):
        return batch[0]
