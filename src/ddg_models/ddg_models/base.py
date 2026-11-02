"""
base
====
Minimal `BaseModel` class mirroring SPURS's. Just wraps `nn.Module` with a
small config-merging helper so that subclasses can use a class-level
`_default_cfg` and have user overrides applied on construction.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

try:
    from omegaconf import OmegaConf
    _HAS_OMEGACONF = True
except ImportError:
    OmegaConf = None
    _HAS_OMEGACONF = False


class BaseModel(nn.Module):
    """
    Base class for all stability models.

    Subclasses should define a class-level `_default_cfg` (a dataclass or
    plain dict). On `__init__`, the user-supplied `cfg` is merged on top of
    `_default_cfg` and stored as `self.cfg`. Merge prefers `omegaconf` if
    available (which gracefully handles dataclasses and DictConfig), and falls
    back to dict.update otherwise.
    """

    _default_cfg: Any = {}

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self._update_cfg(cfg)

    def _update_cfg(self, cfg: Any) -> None:
        if _HAS_OMEGACONF:
            self.cfg = OmegaConf.merge(
                OmegaConf.structured(self._default_cfg),
                cfg if isinstance(cfg, (dict,)) else OmegaConf.structured(cfg),
            )
        else:
            # Plain-Python fallback. Coerce dataclasses to dicts.
            from dataclasses import asdict, is_dataclass
            base = asdict(self._default_cfg) if is_dataclass(self._default_cfg) else dict(self._default_cfg)
            override = asdict(cfg) if is_dataclass(cfg) else dict(cfg)

            def _deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and isinstance(d.get(k), dict):
                        d[k] = _deep_update(d[k], v)
                    else:
                        d[k] = v
                return d
            self.cfg = _deep_update(base, override)
