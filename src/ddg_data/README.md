# ddg_data

Standalone PyTorch data-loading and feature-extraction library for protein
stability prediction. Faithful reimplementation of the SPURS data pipeline
([luo-group/SPURS](https://github.com/luo-group/SPURS)) with **no dependency
on the `spurs` package**.

## What you get

- Identical PDB parsing as ProteinMPNN / SPURS (`parse_PDB_biounits`,
  `alt_parse_PDB`)
- Identical `tied_featurize` / `get_pdb` for building ProteinMPNN tensors
- ESM-aware `Alphabet`, `CoordBatchConverter`, and `Featurizer`
- Drop-in dataset classes for all 11 SPURS benchmarks:
  - `MegaScaleDataset` (Tsuboyama 2023)
  - `FireProtDataset`
  - `ddgBenchDataset` (Ssym direct, Ssym inverse, S669)
  - `ddgGeoDataset` (S461, S783, S8754, S2648, S571, S4346)
  - `DomainomeDataset` (Human Domainome)
  - `MegaScaleTestDatasets` (concat of all 11)

## Installation

```bash
pip install torch numpy pandas tqdm biopython fair-esm
pip install lmdb atom3d   # only if you use LMDB-backed datasets

# then either copy the ddg_data folder into your project, or install:
cd ddg_data
pip install -e .
```

## Directory layout expected for data

The library expects SPURS's standard layout under a `data_root` you pass in:

```
data_root/
└── data/
    └── dataset/
        ├── megascale/
        │   ├── Tsuboyama2023_Dataset2_Dataset3_20230416.csv
        │   ├── mmseq_mut_search_0.25.m8
        │   ├── mega_splits.pkl
        │   └── AlphaFold_model_PDBs/<wt_name>.pdb
        ├── fireprot/
        │   └── fireprot_upload/
        │       ├── csvs/
        │       │   ├── 4_fireprotDB_bestpH.csv
        │       │   └── fireprot_splits.pkl
        │       └── pdbs/<wt_name>.pdb
        ├── ssym/
        │   ├── pdb/
        │   ├── ssym-5fold_clean_dir.csv
        │   └── ssym-5fold_clean_inv.csv
        ├── S669/
        │   ├── pdb/
        │   └── s669_clean_dir.csv
        └── geostab_data/
            ├── ddG_cleaned/
            │   ├── S461/, S461.csv
            │   ├── S783/, S783.csv
            │   ├── S8754/, S8754.csv
            │   └── S2648/, S2648.csv
            └── dTm_cleaned/
                ├── S571/, S571.csv
                └── S4346/, S4346.csv
```

A `parsed_structure.json` cache is generated automatically on first run inside
each PDB directory's parent.

## Quick start

### Loading megascale and evaluating on all benchmarks

```python
from torch.utils.data import DataLoader
from ddg_data import (
    Alphabet, Featurizer,
    MegaScaleDataset, MegaScaleTestDatasets,
)

DATA_ROOT = "/path/to/spurs/data/root"   # contains data/dataset/...

# Featurizer for batch processing
alphabet   = Alphabet(name="esm")
featurizer = Featurizer(alphabet)

# Training data
train_ds = MegaScaleDataset(data_root=DATA_ROOT, split="train")
val_ds   = MegaScaleDataset(data_root=DATA_ROOT, split="val")

train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=4,
    collate_fn=featurizer,
)

# All 11 test benchmarks at once
test_collection = MegaScaleTestDatasets(data_root=DATA_ROOT)
for name, dataset in test_collection.iter_named():
    loader = DataLoader(dataset, batch_size=1, collate_fn=featurizer)
    print(f"{name}: {len(dataset)} proteins")
```

### What a batch contains

Each batch is a dict for ONE wild-type protein with many bundled mutations:

```python
batch = next(iter(train_loader))

# ProteinMPNN structural tensors:
batch["X"]                    # (1, L, 4, 3)  backbone coords [N, CA, C, O]
batch["S"]                    # (1, L)        sequence indices
batch["mask"]                 # (1, L)        1 for valid positions
batch["chain_M"]              # (1, L)        chain mask
batch["chain_M_chain_M_pos"]  # (1, L)        chain mask × fixed position mask
batch["residue_idx"]          # (1, L)        residue index
batch["chain_encoding_all"]   # (1, L)        chain encoding

# ESM tokens (added by Featurizer):
batch["tokens"]               # (1, L+2)      ESM tokenised sequence

# Mutations bundled together:
batch["mut_ids"]              # list of ints,  K mutated positions
batch["ddG"]                  # (K, 1) tensor of ground-truth ΔΔG
batch["append_tensors"]       # (K, 42)        [wt_onehot_21 | mut_onehot_21]

# Metadata:
batch["seq"]                  # str, wild-type sequence
batch["coords"]               # dict of {N,CA,C,O -> (L, 3) tensors}
batch["name"]                 # str, PDB stem
batch["chain_ids"]            # str, chain letter
batch["dataset"]              # str, dataset tag
```

### One-shot ΔΔG prediction (SPURS-style)

Once you have the model output `phi` of shape `(L, 20)`, the predicted ΔΔG
for mutation `wt -> mut` at position `i` is:

```python
ddg = phi[i, mut_idx] - phi[i, wt_idx]
```

The `append_tensors` field gives you `(wt_onehot, mut_onehot)` so you can
compute all batched ΔΔGs in a single masked sum, exactly as SPURS does:

```python
# energies: (K, 20) — per-mutation 20-AA energy vector
# append_tensors: (K, 42)
wt_oh  = batch["append_tensors"][:, :20]    # drop the 21st (X) column
mut_oh = batch["append_tensors"][:, 21:41]
ddg    = (energies * mut_oh).sum(-1) - (energies * wt_oh).sum(-1)
```

## Module map

```
ddg_data/
├── __init__.py            # public API
├── pdb_parser.py          # parse_PDB_biounits, alt_parse_PDB,
│                          # parse_pdb_directory_to_json, fermi transforms
├── featurizer.py          # tied_featurize, get_pdb,
│                          # Alphabet, CoordBatchConverter, Featurizer
├── lmdb_dataset.py        # LMDBDataset (optional)
└── datasets/
    ├── __init__.py
    ├── megascale.py       # Tsuboyama megascale + train/val/test splits
    ├── fireprot.py        # FireProtDB
    ├── ddgbench.py        # Ssym direct/inverse, S669
    ├── ddggeo.py          # S461/S783/S8754/S2648/S571/S4346
    ├── domainome.py       # Human Domainome
    └── combined.py        # MegaScaleTestDatasets (all 11)
```

## Notes

- **No `spurs` imports** anywhere. The only external dependencies are
  `torch`, `numpy`, `pandas`, `tqdm`, `biopython` (for `Bio.pairwise2` if you
  add alignment code later), and `fair-esm`. `lmdb` and `atom3d` are only
  needed if you use `LMDBDataset`.
- Parsed PDB caches are written automatically to `parsed_structure.json` files
  inside the PDB directories, just like SPURS.
- The dataset classes return one wild-type protein dict per `__getitem__` call,
  with all mutations bundled — this matches SPURS's batch-size-1 training
  convention. Use `Featurizer` as your `collate_fn`.
