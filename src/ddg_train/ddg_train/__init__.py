"""
ddg_train
==============
Training and evaluation entry points for the standalone SPURS reproduction.

This package only imports from `ddg_data` and `ddg_models` —
no `spurs` package required.
"""

from .train     import train, TrainConfig
from .evaluate  import evaluate, evaluate_all_benchmarks

__all__ = [
    "train", "TrainConfig",
    "evaluate", "evaluate_all_benchmarks",
]
