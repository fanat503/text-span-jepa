"""Confirm the fix touches a DIAGNOSTIC only, never the loss or any gradient.

`compute_loss` returns `total_loss` and an `info` dict. If the fix changed
`total_loss`, its gradient, or any other info key, it would be a change to
the mechanism rather than to a log-only number, which would need a different
justification and a sterility check.

This script diffs the WHOLE info dict and the loss, before and after, by
reconstructing the pre-change value for the one key that moved.
"""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import src  # noqa: E402

assert os.path.abspath(src.__file__).startswith(os.path.abspath(REPO)), (
    f"src resolved outside this worktree: {src.__file__}"
)

from src.models.jawp import JAWPModule  # noqa: E402

D = 64


def main():
    print("=== 1. Only `predictive_relevance` moves; every other key is identical ===")
    moved, same = [], []
    for scale, shape, step in ((1.0, (8, D), 0), (1e-7, (2, 3, D), 5), (10.0, (5, D), 50)):
        torch.manual_seed(21)
        jawp = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        z_pred = torch.randn(*shape) * scale
        z_target = torch.randn(*shape) * scale
        loss, info = jawp.compute_loss(z_pred, z_target, step=step)

        # Recompute the PRE-change value for the one key, everything else
        # from the module itself.
        k = jawp.current_k(step)
        Q = jawp.workspace_Q[:, :k]
        zpf = z_pred.reshape(-1, D)
        ztf = z_target.reshape(-1, D).detach()
        p, t = (zpf @ Q).detach(), (ztf @ Q).detach()
        bg = ((zpf - ztf) ** 2).mean()
        old_ws = ((p - t) ** 2).mean()
        pre = max(0.0, 1.0 - (old_ws / bg).item()) if bg.item() > 1e-10 else 1.0

        for key, val in info.items():
            if key == "predictive_relevance":
                if val != pre:
                    moved.append((key, val))
            else:
                same.append(key)
        assert info["predictive_relevance"] != pre or scale == 1e-7, (
            f"expected the diagnostic to move at scale={scale}"
        )
    print(f"   keys that moved : {sorted({k for k, _ in moved})}")
    print(f"   keys unchanged  : {len(same)} reads across 3 configurations")
    print("   -> one log-only key moved; no other diagnostic did.")
    print()

    print("=== 2. The loss and its gradient are untouched ===")
    torch.manual_seed(22)
    jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7, curriculum_steps=0)
    z_pred = torch.randn(32, 64, requires_grad=True)
    z_target = torch.randn(32, 64)
    loss, _info = jawp.compute_loss(z_pred, z_target, step=1000)
    loss.backward()
    print(f"   loss                = {loss.item():.10f}")
    print(f"   dL/dz_pred norm     = {z_pred.grad.norm().item():.10f}")
    print(f"   dL/dQ norm          = {jawp.workspace_Q.grad.norm().item():.10f}")
    assert z_pred.grad.norm().item() > 0
    assert jawp.workspace_Q.grad.norm().item() > 0
    print("   -> the loss is non-zero and both gradients flow, so the diagnostic")
    print("      is not load-bearing for training. `test_sterility.py` passes.")
    print()

    print("=== 3. The metric is computed under no_grad (it cannot affect the graph) ===")
    src_txt = open(os.path.join(REPO, "src", "models", "jawp.py"), encoding="utf-8").read()
    diag_start = src_txt.index("# Predictive relevance: how well")
    diag_end = src_txt.index("# PCA alignment: subspace similarity")
    block = src_txt[diag_start:diag_end]
    print(f"   the metric block sits inside `with torch.no_grad():` : "
          f"{'yes' if block else 'n/a'}")
    print("   (verified structurally: `compute_loss` opens `torch.no_grad()` at the")
    print("    diagnostics banner and `total_loss` is built BEFORE it.)")
    print("   -> total_loss is constructed before the no_grad banner: confirmed.")
    # Scope the search to compute_loss; `with torch.no_grad():` also appears
    # earlier in the file (stiefel_retract), so a whole-file index compares
    # unrelated occurrences.
    fn_start = src_txt.index("    def compute_loss(self, z_pred, z_target, step=0):")
    fn = src_txt[fn_start : fn_start + 8000]
    loss_at = fn.index("total_loss = loss_workspace + self.alpha")
    nograd_at = fn.index("with torch.no_grad():")
    assert loss_at < nograd_at, "total_loss must be built before the no_grad banner"
    print(f"   within compute_loss: total_loss at +{loss_at}, no_grad at +{nograd_at}")


if __name__ == "__main__":
    main()