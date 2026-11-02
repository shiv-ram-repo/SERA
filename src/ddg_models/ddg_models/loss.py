"""
loss
====
Loss wrappers matching SPURS's `L1Loss` and `MSELoss` interface.

Each loss returns `(loss_avg, log_dict)` where `log_dict` contains tensors
useful for logging (predictions, targets, loss sum). This shape matches
SPURS's `spurs/modules/l1loss.py` and `mseloss.py` interfaces so trainers
can swap freely.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class L1Loss(nn.L1Loss):
    def forward(self, pred_value: torch.Tensor, y: torch.Tensor):
        loss_avg = super().forward(pred_value, y)
        return loss_avg, {
            "loss_sum":   loss_avg.detach(),
            "pred_value": pred_value.detach(),
            "y":          y.detach(),
        }


class MSELoss(nn.MSELoss):
    def forward(self, pred_value: torch.Tensor, y: torch.Tensor):
        loss_avg = super().forward(pred_value, y)
        return loss_avg, {
            "loss_sum":   loss_avg.detach(),
            "pred_value": pred_value.detach(),
            "y":          y.detach(),
        }
