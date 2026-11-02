"""
get_protein_mpnn
=================
Construct a ProteinMPNN instance with the same hyperparameters as SPURS
(`v_48_020` settings: hidden=128, k=48, 3 enc/dec layers) and optionally load
the pretrained checkpoint.

By default, this function does NOT load weights (matching SPURS's current
codebase, which has the load_state_dict call commented out). To load the
official ProteinMPNN v_48_020 checkpoint, pass `ckpt_path=...` explicitly:

    model = get_protein_mpnn(
        tune=True,
        ckpt_path="vanilla_model_weights/v_48_020.pt",
    )

The checkpoint is available from:
    https://github.com/dauparas/ProteinMPNN/blob/main/vanilla_model_weights/v_48_020.pt
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .model import ProteinMPNN


def get_protein_mpnn(
    version: str           = "v_48_020.pt",
    tune: bool             = False,
    ckpt_path: Optional[str] = None,
) -> ProteinMPNN:
    """
    Build ProteinMPNN with SPURS's settings.

    Parameters
    ----------
    version : str
        Kept for API parity with SPURS. Only `v_48_020.pt` is supported.
    tune : bool
        If True, leaves the model in train mode and allows gradients.
        If False, sets eval mode and disables gradients (frozen encoder).
    ckpt_path : str or None
        If provided, load this checkpoint after construction. The expected
        format is `{'model_state_dict': state_dict}` matching the official
        ProteinMPNN checkpoint layout.
    """
    hidden_dim = 128
    num_layers = 3

    model = ProteinMPNN(
        ca_only             = False,
        num_letters         = 21,
        node_features       = hidden_dim,
        edge_features       = hidden_dim,
        hidden_dim          = hidden_dim,
        num_encoder_layers  = num_layers,
        num_decoder_layers  = num_layers,
        k_neighbors         = 48,
        augment_eps         = 0.0,
    )

    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ProteinMPNN checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    if tune:
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model
