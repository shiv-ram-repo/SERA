# SERA

**Sparse Expert Readout Architecture for Protein Stability Prediction**

SERA is a sparse mixture-of-experts (MoE) readout architecture for predicting protein stability changes resulting from amino acid substitutions.

The framework uses wild-type protein sequence and backbone-derived structural representations to model mutation effects through a shared site-level representation and conditional expert computation.

## Key Features

- **Sparse mixture-of-experts readout:** Uses eight specialized experts with top-2 routing and an always-active shared expert.
- **Wild-type-conditioned routing:** Mutations at the same wild-type residue position share the same routing decision, independent of the substituted amino acid.
- **Antisymmetric prediction:** Predicts mutation effects using a score-difference formulation:

  \[
  \widehat{\Delta\Delta G}_{a\rightarrow b}
  = s(b)-s(a)
  \]

  This ensures exact antisymmetry when reversing a substitution under the same wild-type context.
- **Structural reasoning:** Incorporates structural information into the residue-level representation and expert routing.
- **Controlled evaluation:** Includes comparisons involving dense readouts, alternative fusion mechanisms, expert configurations, routing strategies and structural adapter placement.

## Repository Structure

```text
SERA/
│
├── src/
│   ├── ddg_models/
│   │   └── ddg_models/
│   │       ├── moe_head.py
│   │       ├── multimodal_ddg.py
│   │       ├── layer_fusion.py
│   │       ├── pair_readout.py
│   │       ├── energy_decomposition.py
│   │       ├── esm_adapter/
│   │       └── protein_mpnn/
│   │
│   ├── ddg_data/
│   │   └── ddg_data/
│   │       ├── featurizer.py
│   │       ├── pdb_parser.py
│   │       ├── datasets/
│   │       └── lmdb_dataset.py
│   │
│   └── ddg_train/
│       └── ddg_train/
│           ├── train.py
│           ├── evaluate.py
│           ├── metrics.py
│           ├── antisymmetry.py
│           ├── custom_losses.py
│           └── tta_inference.py
│
├── analysis/
│   ├── moe_routing_analysis.py
│   ├── moe_routing_multiseed.py
│   ├── moe_capacity_analysis.py
│   └── collect_moe_results.py
│
├── annotate_structure.py
├── expert_match.py
├── expert_routing.py
├── identity_effect.py
├── figures.py
├── tables.py
├── run_fill_missing.py
├── shift_vs_length.py
│
├── runs/
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

SERA is organized into three Python packages: model components, data processing and training/evaluation.

Clone the repository:

```bash
git clone git@github.com:shiv-ram-repo/SERA.git
cd SERA
```

Install the packages in editable mode:

```bash
pip install -e ./src/ddg_models
pip install -e ./src/ddg_data
pip install -e ./src/ddg_train
```

The package setup files specify dependencies including PyTorch, NumPy, pandas, SciPy, scikit-learn, Biopython, fair-esm and OmegaConf.

For workflows requiring LMDB support, install the optional data dependencies:

```bash
pip install -e "./src/ddg_data[lmdb]"
```

## Main Components

| Component | Description |
|---|---|
| `ddg_models` | Protein representation, structural processing and stability prediction architectures |
| `ddg_data` | Protein structure parsing, residue featurization and dataset loading |
| `ddg_train` | Model training, evaluation, loss functions and prediction utilities |
| `analysis` | Expert routing, capacity, specialization and statistical analyses |

## Data and Checkpoints

Benchmark datasets, pretrained model weights, large model checkpoints and generated prediction arrays are not distributed through this repository.

Users should obtain the relevant datasets and pretrained weights from their respective sources and configure the data paths for their local environment.

Large experiment artifacts are intentionally excluded from normal Git version control.

## Reproducibility

Experiment-specific logs, manifests and evaluation metrics are maintained alongside the corresponding local experiment outputs.

The training and evaluation implementations are provided in `src/ddg_train/ddg_train/`.

Exact reproduction requires the corresponding dataset versions, pretrained weights, configurations and random seeds.

## Citation

Citation information will be added when the publication details are finalized.

## License

A software license has not yet been assigned to this repository. Please do not assume that the code is available for unrestricted reuse.
