"""
moe_smoke_test.py
=================
Fast, self-contained check of the MoE readout before committing GPU time to a
3-seed run. Builds `MoEReadout` directly on synthetic tensors shaped exactly
like the real ones (B=1, K mutations, D=1408 fused representation) and verifies:

  1. output shape is (B, K, 21), matching the dense head
  2. sparse dispatch actually routes to `top_k` experts per mutation
  3. the load-balance + z auxiliary loss is finite and pops exactly once
  4. gradients reach the router, the routed experts and the shared expert
  5. eval mode is deterministic (noisy top-k disabled) and accumulates no aux
  6. the structural-descriptor router degrades to zeros when K does not match
     `mut_ids` (the return_logist path) instead of crashing

Run from the repo root:
    python analysis/moe_smoke_test.py
"""

from __future__ import annotations

import torch

from ddg_models.mlp import MLPConfig
from ddg_models.moe_head import MoEReadout

D, K, L = 1408, 37, 120
NUM_EXPERTS, TOP_K = 8, 2


def make_batch(k=K, with_struct=True):
    batch = {"muted_id_representation": torch.randn(1, k, D)}
    if with_struct:
        X = torch.randn(1, L, 4, 3) * 10.0
        batch["X"] = X
        batch["mut_ids"] = torch.randint(0, L, (k,))
    return batch


def build(**over):
    cfg = MLPConfig(input_dim=D, hidden_dim=[512, 512, 512], output_dim=21, dropout=0.1)
    cfg.use_moe = True
    cfg.moe_num_experts = over.get("num_experts", NUM_EXPERTS)
    cfg.moe_top_k = over.get("top_k", TOP_K)
    cfg.moe_shared_expert = over.get("shared", True)
    cfg.moe_router_struct = over.get("struct", True)
    cfg.moe_noise_eps = over.get("noise", 1.0)
    cfg.moe_expert_hidden = [512, 512, 512]
    return MoEReadout(cfg)


def check(label, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        raise SystemExit(f"smoke test failed at: {label}")


def main():
    torch.manual_seed(0)
    print("=== MoE readout smoke test ===")

    head = build()
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  head parameters: {n_params/1e6:.2f} M "
          f"({NUM_EXPERTS} experts, top-{TOP_K}, shared expert on)")

    # 1. shape
    head.train()
    batch = make_batch()
    out = head(batch)
    check(f"output shape {tuple(out.shape)} == (1, {K}, 21)", tuple(out.shape) == (1, K, 21))
    check("output finite", bool(torch.isfinite(out).all()))

    # 2. sparse routing
    flat = batch["muted_id_representation"].reshape(-1, D)
    struct = head._structural_features(batch, flat.shape[0], flat.device, flat.dtype)
    gates, probs, logits, topk_idx = head._route(flat, struct)
    nonzero_per_token = (gates > 0).sum(dim=-1)
    check(f"each mutation routed to exactly top_k={TOP_K} experts",
          bool((nonzero_per_token == TOP_K).all()))
    check("gate weights sum to 1 per mutation",
          bool(torch.allclose(gates.sum(-1), torch.ones(flat.shape[0]), atol=1e-5)))
    used = int((gates.sum(0) > 0).sum())
    print(f"       experts receiving at least one mutation: {used}/{NUM_EXPERTS}")

    # 3. aux loss
    aux = head.pop_aux()
    check("aux loss returned after forward", aux is not None)
    check("aux loss finite", bool(torch.isfinite(aux)))
    print(f"       aux = {float(aux):.4f}  (load-balance floor is 1.0 at perfect balance)")
    check("aux accumulator reset after pop", head.pop_aux() is None)

    # aux averaging across several forwards in one step (main + siamese + margin)
    head.train()
    for _ in range(3):
        head(make_batch())
    aux3 = head.pop_aux()
    check("aux accumulates across multiple forwards and pops once",
          aux3 is not None and head.pop_aux() is None)

    # 4. gradients
    head.zero_grad()
    out = head(make_batch())
    aux = head.pop_aux()
    (out.square().mean() + 0.01 * aux).backward()
    check("router receives gradient",
          head.router.weight.grad is not None and head.router.weight.grad.abs().sum() > 0)
    check("shared expert receives gradient",
          head.shared_expert.fcs[0].weight.grad is not None
          and head.shared_expert.fcs[0].weight.grad.abs().sum() > 0)
    n_expert_grads = sum(
        1 for e in head.experts
        if e.fcs[0].weight.grad is not None and e.fcs[0].weight.grad.abs().sum() > 0
    )
    check(f"routed experts receive gradient ({n_expert_grads}/{NUM_EXPERTS} active)",
          n_expert_grads > 0)

    # 5. eval determinism + no aux accumulation
    head.eval()
    b = make_batch()
    with torch.no_grad():
        o1, o2 = head(b), head(b)
    check("eval deterministic (noisy top-k off, dropout off)",
          bool(torch.allclose(o1, o2)))
    check("no aux accumulated in eval mode", head.pop_aux() is None)

    # 6. structural fallback when K does not match mut_ids
    head.train()
    bad = make_batch(k=K)
    bad["mut_ids"] = torch.randint(0, L, (K + 5,))     # deliberate mismatch
    out_bad = head(bad)
    head.pop_aux()
    check("K/mut_ids mismatch falls back to zero descriptors instead of crashing",
          tuple(out_bad.shape) == (1, K, 21))

    # variants
    for kw, label in [
        (dict(shared=False), "shared expert off"),
        (dict(struct=False), "structural router off"),
        (dict(top_k=1), "top-1 routing"),
        (dict(num_experts=4), "4 experts"),
    ]:
        h = build(**kw)
        h.train()
        o = h(make_batch())
        a = h.pop_aux()
        check(f"variant: {label}", tuple(o.shape) == (1, K, 21) and torch.isfinite(o).all()
              and a is not None and torch.isfinite(a))

    print("\nAll checks passed. Safe to run scripts/run_moe_experiment.py")


if __name__ == "__main__":
    main()
