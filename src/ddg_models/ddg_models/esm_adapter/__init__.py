"""ESM2 backbone with structural adapter layers."""

from .adapter_layer import TransformerLayerWithStructuralAdapter
from .esm2_adapter  import ESM2WithStructuralAdapter

__all__ = [
    "TransformerLayerWithStructuralAdapter",
    "ESM2WithStructuralAdapter",
]
