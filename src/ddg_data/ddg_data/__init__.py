"""
ddg_data
=============
Standalone data loading and featurisation for protein stability prediction.

Faithful reimplementation of SPURS's data pipeline with no dependency on the
`spurs` package. All you need is PyTorch, NumPy, pandas, BioPython, ESM,
and atom3d/lmdb (only required for LMDB-backed datasets).

Public API
----------
PDB parsing:
    parse_PDB_biounits, parse_single_PDB, parse_pdb_dir
    alt_parse_PDB_biounits, alt_parse_PDB     (resn_list-aware variant)
    parse_pdb_directory_to_json               (build a JSON cache from a folder of PDBs)
    fermi_transform, inverse_fermi_transform

Featurisation:
    tied_featurize       (ProteinMPNN-style tensor packing)
    get_pdb              (single-protein dict builder)
    Alphabet             (ESM-1b alphabet wrapper)
    CoordBatchConverter  (ESM coord batch converter)
    Featurizer           (top-level featurizer for batches)

Datasets:
    MegaScaleDataset
    FireProtDataset
    ddgBenchDataset           (Ssym direct/inverse, S669)
    ddgGeoDataset             (S461/S783/S8754/S2648/S571/S4346)
    DomainomeDataset
    MegaScaleTestDatasets     (concat of all 11 benchmarks)

Constants:
    ALPHABET, ALPHABET_21
"""

from .pdb_parser import (
    parse_PDB_biounits,
    parse_single_PDB,
    parse_pdb_dir,
    alt_parse_PDB_biounits,
    alt_parse_PDB,
    parse_pdb_directory_to_json,
    fermi_transform,
    inverse_fermi_transform,
)

from .featurizer import (
    tied_featurize,
    get_pdb,
    Alphabet,
    CoordBatchConverter,
    Featurizer,
)

from .lmdb_dataset import LMDBDataset

from .datasets.megascale import MegaScaleDataset
from .datasets.fireprot  import FireProtDataset
from .datasets.ddgbench  import ddgBenchDataset
from .datasets.ddggeo    import ddgGeoDataset
from .datasets.domainome import DomainomeDataset
from .datasets.combined  import MegaScaleTestDatasets

ALPHABET    = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_21 = "ACDEFGHIKLMNPQRSTVWYX"

__all__ = [
    "parse_PDB_biounits", "parse_single_PDB", "parse_pdb_dir",
    "alt_parse_PDB_biounits", "alt_parse_PDB",
    "parse_pdb_directory_to_json",
    "fermi_transform", "inverse_fermi_transform",
    "tied_featurize", "get_pdb",
    "Alphabet", "CoordBatchConverter", "Featurizer",
    "LMDBDataset",
    "MegaScaleDataset", "FireProtDataset",
    "ddgBenchDataset", "ddgGeoDataset", "DomainomeDataset",
    "MegaScaleTestDatasets",
    "ALPHABET", "ALPHABET_21",
]
