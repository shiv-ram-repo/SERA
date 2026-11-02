"""
sam.py
======
Sharpness-Aware Minimization (Foret et al., ICLR 2021) as an
optimization-level intervention for OOD generalization.

SAM seeks parameters in flat loss regions by, at each step:
  1. computing the gradient at the current weights,
  2. taking an "ascent" step of radius rho toward the local worst-case,
  3. computing the gradient there,
  4. applying that gradient with the base optimizer from the ORIGINAL
     weights.

This directly serves the paper's thesis that OOD robustness is governed
by the optimization landscape rather than architecture: flat minima are
known to generalize better under distribution shift. No architecture
change; it only changes how the existing loss is optimized.

Usage in the training loop (replaces optimizer.step()):

    optimizer = SAM(model.parameters(), base_optimizer=AdamW,
                    rho=0.05, lr=1e-4, weight_decay=1e-2)

    # inside the step:
    loss = compute_loss(batch)          # first forward/backward
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss2 = compute_loss(batch)         # second forward/backward at w+e(w)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

Note SAM doubles the forward/backward cost per step (two gradient
evaluations). Report this in the computational-cost table.
"""

from __future__ import annotations
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05,
                 adaptive: bool = False, **kwargs):
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)                       # ascend to w + e(w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]   # restore original weights
        self.base_optimizer.step()                # apply base optimizer update
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                w = (torch.abs(p) if group["adaptive"] else 1.0)
                norms.append((w * p.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):
        raise RuntimeError("SAM requires explicit first_step/second_step; "
                           "do not call .step() directly.")

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
