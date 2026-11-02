"""
Dataset classes for protein stability benchmarks.

Each class returns a dict per protein containing:
    - X, S, mask, chain_M, chain_M_chain_M_pos, residue_idx,
      chain_encoding_all, randn_1   (ProteinMPNN tensors)
    - seq, coords, name, chain_ids
    - mut_ids        : list of mutated positions (0-indexed in seq)
    - ddG            : (N, 1) tensor of ground-truth ΔΔG values
    - append_tensors : (N, 42) one-hot WT and MUT amino acid concatenated
    - dataset        : str, dataset name tag
"""

from .megascale import MegaScaleDataset
from .fireprot  import FireProtDataset
from .ddgbench  import ddgBenchDataset
from .ddggeo    import ddgGeoDataset
from .domainome import DomainomeDataset
from .combined  import MegaScaleTestDatasets

__all__ = [
    "MegaScaleDataset",
    "FireProtDataset",
    "ddgBenchDataset",
    "ddgGeoDataset",
    "DomainomeDataset",
    "MegaScaleTestDatasets",
]
