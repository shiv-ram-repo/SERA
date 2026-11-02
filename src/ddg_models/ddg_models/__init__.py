"""
ddg_models
===============
Standalone neural network models for protein stability prediction.

Faithful reimplementation of SPURS's model code with no dependency on the
`spurs` package. Pairs with `ddg_data` for end-to-end training.

Public API
----------
Base classes:
    BaseModel             — minimal nn.Module with cfg merging

Components:
    ProteinMPNN           — ProteinMPNN model
    ProteinMPNNConfig     — dataclass config
    get_protein_mpnn      — instantiate + optionally load v_48_020 checkpoint

    ESM2WithStructuralAdapter — ESM2 backbone with structural adapter layers
    AdapterLayer              — single transformer layer + adapter cross-attention

    MLP                   — final stability prediction head
    MLPConfig             — dataclass config

Full model:
    SPURS                 — Full SPURS model (encoder + decoder + MLP)
    SPURSConfig           — dataclass config

Losses:
    L1Loss, MSELoss       — thin wrappers returning (loss, log_dict)
"""

from .base   import BaseModel
from .mlp    import MLP, MLPConfig
from .loss   import L1Loss, MSELoss

from .protein_mpnn import (
    ProteinMPNN, ProteinMPNNConfig, get_protein_mpnn,
)

from .esm_adapter import (
    ESM2WithStructuralAdapter,
)

from .multimodal_ddg  import MultimodalDDG, MultimodalDDGConfig

__all__ = [
    "BaseModel",
    "MLP", "MLPConfig",
    "L1Loss", "MSELoss",
    "ProteinMPNN", "ProteinMPNNConfig", "get_protein_mpnn",
    "ESM2WithStructuralAdapter",
    "MultimodalDDG", "MultimodalDDGConfig",
]
