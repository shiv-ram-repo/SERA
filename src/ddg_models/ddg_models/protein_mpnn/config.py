"""
ProteinMPNN configuration.

Default values match SPURS's `ProteinMPNNConfig`. The actual ProteinMPNN
checkpoint `v_48_020.pt` uses k_neighbors=48 and 3 enc/dec layers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProteinMPNNConfig:
    d_model:      int   = 128        # model dim (overridden by MultimodalDDG to MLPConfig.input_dim)
    d_node_feats: int   = 128
    d_edge_feats: int   = 128
    k_neighbors:  int   = 48
    augment_eps:  float = 0.0
    n_enc_layers: int   = 3
    dropout:      float = 0.1

    # Trainability / decoding behaviour
    tune: bool                     = False
    use_input_decoding_order: bool = False

    # Decoder-only options (kept for compatibility, unused by MultimodalDDG in current code)
    n_vocab:               int   = 22
    n_dec_layers:          int   = 3
    random_decoding_order: bool  = True
    nar:                   bool  = True
    crf:                   bool  = False
    use_esm_alphabet:      bool  = False
