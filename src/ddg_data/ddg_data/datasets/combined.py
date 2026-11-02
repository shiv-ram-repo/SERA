"""
MegaScaleTestDatasets
=====================
Convenience class that aggregates the 11 standard ΔΔG / ΔTm test benchmarks
into one dataset:

    Megascale test, FireProt-HF, Ssym direct, Ssym inverse, S669,
    S461, S783, S8754, S2648, S571, S4346

Mirrors SPURS's class of the same name. Constructed datasets are exposed as
attributes so you can also iterate them individually.

Expected layout under `data_root`:

    data/dataset/
        megascale/...
        fireprot/...
        ssym/
            pdb/
            ssym-5fold_clean_dir.csv
            ssym-5fold_clean_inv.csv
        S669/
            pdb/
            s669_clean_dir.csv
        geostab_data/
            ddG_cleaned/
                S461/, S461.csv
                S783/, S783.csv
                S8754/, S8754.csv
                S2648/, S2648.csv
            dTm_cleaned/
                S571/, S571.csv
                S4346/, S4346.csv
"""

from __future__ import annotations

import os
from torch.utils.data import Dataset

from .megascale import MegaScaleDataset
from .fireprot  import FireProtDataset
from .ddgbench  import ddgBenchDataset
from .ddggeo    import ddgGeoDataset


class MegaScaleTestDatasets(Dataset):
    def __init__(self, data_root: str):
        self.data_root = data_root

        # Megascale held-out split
        self.megascale = MegaScaleDataset(
            data_root=data_root, reduce="", split="test",
        )
        # FireProt homologue-free
        self.fireport  = FireProtDataset(
            data_root=data_root, split="homologue-free",
        )
        # Ssym direct + inverse
        self.ssym_dir  = ddgBenchDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/ssym/pdb"),
            csv_fname = os.path.join(data_root, "data/dataset/ssym/ssym-5fold_clean_dir.csv"),
            dataset_name="ssym_dir",
        )
        self.ssym_inv  = ddgBenchDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/ssym/pdb"),
            csv_fname = os.path.join(data_root, "data/dataset/ssym/ssym-5fold_clean_inv.csv"),
            dataset_name="ssym_inv",
        )
        # S669
        self.s669      = ddgBenchDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/S669/pdb"),
            csv_fname = os.path.join(data_root, "data/dataset/S669/s669_clean_dir.csv"),
            dataset_name="S669",
        )
        # GeoStab ΔΔG datasets
        self.S461 = ddgGeoDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/geostab_data/ddG_cleaned/S461"),
            csv_fname = os.path.join(data_root, "data/dataset/geostab_data/ddG_cleaned/S461.csv"),
            dataset_name="S461",
        )
        self.S783 = ddgGeoDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/geostab_data/ddG_cleaned/S783"),
            csv_fname = os.path.join(data_root, "data/dataset/geostab_data/ddG_cleaned/S783.csv"),
            dataset_name="S783",
        )
        self.S8754 = ddgGeoDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/geostab_data/ddG_cleaned/S8754"),
            csv_fname = os.path.join(data_root, "data/dataset/geostab_data/ddG_cleaned/S8754.csv"),
            dataset_name="S8754",
        )
        self.S2648 = ddgGeoDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/geostab_data/ddG_cleaned/S2648"),
            csv_fname = os.path.join(data_root, "data/dataset/geostab_data/ddG_cleaned/S2648.csv"),
            dataset_name="S2648",
            stage="test",
        )
        # GeoStab ΔTm datasets
        self.S571  = ddgGeoDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/geostab_data/dTm_cleaned/S571"),
            csv_fname = os.path.join(data_root, "data/dataset/geostab_data/dTm_cleaned/S571.csv"),
            dataset_name="S571",
        )
        self.S4346 = ddgGeoDataset(
            pdb_dir   = os.path.join(data_root, "data/dataset/geostab_data/dTm_cleaned/S4346"),
            csv_fname = os.path.join(data_root, "data/dataset/geostab_data/dTm_cleaned/S4346.csv"),
            dataset_name="S4346",
        )

        # Cache lengths and offsets for indexing
        self._datasets = [
            self.megascale, self.fireport,
            self.ssym_dir,  self.ssym_inv, self.s669,
            self.S461, self.S783, self.S8754, self.S2648,
            self.S571, self.S4346,
        ]
        self._lengths  = [len(d) for d in self._datasets]
        self._offsets  = [0]
        for n in self._lengths:
            self._offsets.append(self._offsets[-1] + n)

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index: int):
        for i, dset in enumerate(self._datasets):
            lo, hi = self._offsets[i], self._offsets[i + 1]
            if lo <= index < hi:
                return dset[index - lo]
        raise IndexError(f"Index {index} out of range")

    def iter_named(self):
        """Yield (name, dataset) for every component dataset."""
        names = [
            "megascale_test", "fireport_hf",
            "ssym_direct", "ssym_inverse", "s669",
            "s461", "s783", "s8754", "s2648",
            "s571", "s4346",
        ]
        for name, dset in zip(names, self._datasets):
            yield name, dset
