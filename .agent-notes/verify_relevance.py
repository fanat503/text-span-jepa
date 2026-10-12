"""Verification for the jawp_predictive_relevance sign/normalisation defect.

Run from the worktree root with the repo's interpreter. Asserts that `src`
resolves inside THIS worktree before it executes anything (an editable install
pointed at another clone exists on this machine).
"""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This machine has an editable install whose finder maps `src` to a DIFFERENT
# clone (C:\Users\<u>\tmp\clone-check-1). Running a script puts the SCRIPT's
# directory on sys.path, not the cwd, so `import src` would silently resolve to
# that other tree. Put the worktree first, then prove it.
sys.path.insert(0, REPO)

import src  # noqa: E402

assert os.path.abspath(src.__file__).startswith(os.path.abspath(REPO)), (
    f"src resolved outside this worktree: {src.__file__}"
)

from src.models.jawp import JAWPModule  # noqa: E402


def orthogonal_Q(D, k, seed=0):
    torch.manual_seed(seed)
    M = torch.randn(D, k, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    return Q


def orthogonal_complement(Q, seed=0):
    """An orthonormal (D, D-k) basis for the orthogonal complement of span(Q)."""
    D, k = Q.shape
    torch.manual_seed(seed + 991)
    R = torch.linalg.qr(torch.randn(D, D - k, dtype=torch.float64))[0]
    # Project the k columns of Q out of R; the remainder stays orthonormal.
    R = R - Q @ (Q.T @ R)
    U, _ = torch.linalg.qr(R)
    return U[:, : D - k]


def relevance_for_share(w, D=64, k=6, N=512, seed=3):
    """Build a residual with an EXACT workspace share `w`, and read the metric.

    `w` is the fraction of residual ENERGY that lies inside span(Q). The
    workspace share is set by scaling the INSIDE component relative to a fixed
    OUTSIDE component, so `w = 0` puts the residual entirely in the background
    (workspace prediction exact) and `w = 1` puts it entirely in span(Q)
    (workspace prediction worst).
    """
    torch.manual_seed(seed)
    Q = orthogonal_Q(D, k, seed=seed)
    U = orthogonal_complement(Q, seed=seed)

    z_target = torch.randn(N, D, dtype=torch.float64)
    inside = torch.randn(N, k, dtype=torch.float64)
    outside = torch.randn(N, D - k, dtype=torch.float64)
    # w = 1 puts the residual entirely inside span(Q): a large finite scale, not
    # infinity (an inf here would produce NaN, not a large-but-finite residual).
    scale = 1e6 if w >= 1.0 else (w / (1.0 - w)) ** 0.5
    resid = inside @ Q.T * scale + outside @ U.T
    z_pred = z_target + resid

    jawp = JAWPModule(embed_dim=D, k_start=k, k_end=k, curriculum_steps=0).double()
    with torch.no_grad():
        jawp.workspace_Q.copy_(Q)
    _, info = jawp.compute_loss(z_pred, z_target, step=10_000)

    # Recover the realised share for comparison.
    r = z_pred - z_target
    w_real = float(((r @ Q) ** 2).sum() / (r**2).sum())
    return w_real, info


def main():
    D, k = 64, 6
    ratio = D / k
    print(f"D={D} k={k}  D/k={ratio:.4f}   chance share k/D={k / D:.4f}")
    print()
    print("=== 1. endpoints, built exactly ===")
    for w in (0.0, 1.0):
        w_real, info = relevance_for_share(w, D, k)
        print(
            f"  constructed w={w:.1f}  realised w={w_real:.6f}  "
            f"loss_workspace={info['loss_workspace']:.6e}  "
            f"predictive_relevance={info['predictive_relevance']:.4f}"
        )
    print()

    print("=== 2. does the metric equal max(0, 1 - (D/k)*w)? ===")
    print("   w_real      pred_rel     1-(D/k)*w     max(0, 1-(D/k)*w)   |err|")
    worst = 0.0
    for w in (0.0, 0.05, 0.094, 0.2, 0.4, 0.6, 0.8, 1.0):
        w_real, info = relevance_for_share(w, D, k)
        pred = info["predictive_relevance"]
        model = max(0.0, 1.0 - ratio * w_real)
        err = abs(pred - model)
        worst = max(worst, err)
        print(f"  {w_real:.6f}  {pred:.6f}   {model:.6f}     {model:.6f}          {err:.2e}")
    print(f"\n  max |metric - max(0, 1-(D/k)w)| = {worst:.3e}  (D=64, k=6)")
    print()

    print("=== 3. where is the zero threshold? ===")
    for w in (0.093, 0.094, 0.095, 0.2):
        w_real, info = relevance_for_share(w, D, k)
        print(
            f"  w={w_real:.6f} (chance is {k / D:.6f})  "
            f"predictive_relevance={info['predictive_relevance']:.6f}"
        )
    print()

    print("=== 4. the claimed reading: does 0.0 mean 'workspace died'? ===")
    print("   Construct a residual ENTIRELY INSIDE span(Q): workspace prediction")
    print("   is then the worst possible, and Q holds all the unexplained energy.")
    w_real, info = relevance_for_share(1.0, D, k)
    print(f"     realised w = {w_real:.6f}")
    print(f"     loss_workspace          = {info['loss_workspace']:.6f}   (worst case)")
    print(f"     predictive_relevance    = {info['predictive_relevance']:.6f}")
    print(f"     workspace_cosine        = {info['workspace_cosine']:.6f}")
    print()

    print("=== 5. and the other endpoint: residual ENTIRELY OUTSIDE span(Q) ===")
    w_real, info = relevance_for_share(0.0, D, k)
    print(f"     realised w = {w_real:.6f}")
    print(f"     loss_workspace          = {info['loss_workspace']:.3e}   (exact)")
    print(f"     predictive_relevance    = {info['predictive_relevance']:.6f}")
    print()

    print("=== 6. a realistic run: where does a trained-ish Q actually land? ===")
    torch.manual_seed(0)
    D2, k2 = 64, 6
    jawp = JAWPModule(embed_dim=D2, k_start=k2, k_end=k2, curriculum_steps=0)
    z_t = torch.randn(256, D2)
    z_p = z_t + torch.randn(256, D2) * 0.5
    vals = []
    for _ in range(20):
        _, info = jawp.compute_loss(z_p, z_t, step=10_000)
        jawp.stiefel_retract()
        vals.append(info["predictive_relevance"])
    print(f"   20 steps of a random-init workspace: relevance in "
          f"[{min(vals):.4f}, {max(vals):.4f}], mean {sum(vals) / len(vals):.4f}")
    print("   (chance-level share would already pin this at 0.0)")
    print()

    print("=== 7. other jawp diagnostics: value vs. input ===")
    # workspace_cosine / utilization / target_ws_fraction with a fixed residual
    torch.manual_seed(1)
    z_t = torch.randn(128, D2)
    for scale in (0.0, 1e-4, 1.0, 10.0):
        z_p = z_t + torch.randn(128, D2) * scale
        _, info = jawp.compute_loss(z_p, z_t, step=10_000)
        print(
            f"   noise={scale:<8g} relevance={info['predictive_relevance']:.4f} "
            f"cos={info['workspace_cosine']:.4f} "
            f"ws_util={info['workspace_utilization']:.4f} "
            f"loss_ws={info['loss_workspace']:.4g}"
        )


if __name__ == "__main__":
    main()