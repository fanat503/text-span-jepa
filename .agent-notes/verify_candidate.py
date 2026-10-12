"""What should the metric be, and is the D/k factor the whole story?

The two errors being divided:

  ws_pred_error = ((Q^T r) ** 2).mean()   over an (N, k) tensor  -> divides by N*k
  bg_pred_error = (r ** 2).mean()         over an (N, D) tensor  -> divides by N*D

Their ratio is therefore (D/k) times the true energy share

    w = ||Q^T r||^2_F / ||r||^2_F   in [0, 1]   (Q orthonormal)

so `1 - ratio` is `1 - (D/k) w`, not `1 - w`. Two consequences:

  (1) SCALE. The zero threshold lands at w = k/D (chance), not at w = 1.
  (2) The quantity's NAME claims it is a "relevance"/"quality" score in [0,1],
      and the consumers read it that way, but `1 - (D/k)w` is not in [0,1] —
      it is a chance-corrected excess. Any w above chance reads 0.

This script measures BOTH candidate metrics on the same constructed residuals
so the choice between them is made on numbers, not on the docstring.
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


def complement(Q, seed=0):
    D, k = Q.shape
    g = torch.Generator().manual_seed(seed + 991)
    R = torch.linalg.qr(torch.randn(D, D - k, generator=g, dtype=torch.float64))[0]
    R = R - Q @ (Q.T @ R)
    return torch.linalg.qr(R)[0][:, : D - k]


def build(w, D, k, N=512, seed=3):
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(D, k, generator=g, dtype=torch.float64))[0]
    U = complement(Q, seed)
    z_target = torch.randn(N, D, generator=g, dtype=torch.float64)
    inside = torch.randn(N, k, generator=g, dtype=torch.float64)
    outside = torch.randn(N, D - k, generator=g, dtype=torch.float64)
    # Normalise both parts to unit energy so the realised share is exactly w.
    inside = inside / inside.norm() * (w**0.5)
    outside = outside / outside.norm() * ((1.0 - w) ** 0.5)
    r = inside @ Q.T + outside @ U.T
    return Q, z_target, z_target + r


def main():
    D, k = 64, 6
    print(f"D={D} k={D and k}  D/k={D / k:.4f}  chance share k/D={k / D:.4f}\n")
    print("  w        current: max(0,1-(D/k)w)    candidate: max(0,1-w)")
    print("  " + "-" * 58)
    for w in (0.0, 0.05, 0.094, 0.2, 0.5, 0.8, 1.0):
        _Q, zt, zp = build(w, D, k)
        r = zp - zt
        w_real = float(((r @ _Q) ** 2).sum() / (r**2).sum())
        cur = max(0.0, 1.0 - (D / k) * w_real)
        cand = max(0.0, 1.0 - w_real)
        print(f"  {w_real:.4f}   {cur:22.4f}    {cand:24.4f}")
    print()
    print("  -> The candidate 1-w spans the full [0,1] and reaches 0 only at the")
    print("     TRUE worst case (all residual inside the workspace). The current")
    print("     form is already 0 by w=0.094, i.e. at chance level.")
    print()

    print("=== Is 1-w a monotone 'how well is the workspace predicted' score? ===")
    print("  Constructing residuals with a KNOWN workspace residual, everything")
    print("  else fixed. A relevance score should FALL as the workspace gets")
    print("  harder to predict, and be 0 only in the limit.")
    for w in (0.0, 0.2, 0.5, 0.8, 0.95, 1.0):
        _Q, zt, zp = build(w, D, k)
        r = zp - zt
        w_real = float(((r @ _Q) ** 2).sum() / (r**2).sum())
        # Per-coordinate workspace error, for orientation only.
        ws_err = float((((zp @ _Q) - (zt @ _Q)) ** 2).mean())
        print(
            f"    w={w_real:.4f}  loss_workspace(per-coord)={ws_err:.6f}  "
            f"1-w={1 - w_real:.4f}"
        )
    print("  -> 1-w falls monotonically as the workspace residual grows, and")
    print("     hits 0 exactly when the workspace holds ALL the error. That is")
    print("     the direction every consumer already assumes.")
    print()

    print("=== Does 1-w stay in [0,1] for an ORTHONORMAL Q only? ===")
    print("  With Q orthonormal, ||Q^T r||^2 <= ||r||^2 by Cauchy-Schwarz, so")
    print("  1-w is in [0,1] with no clamp needed. Check numerically:")
    g = torch.Generator().manual_seed(11)
    Q = torch.linalg.qr(torch.randn(64, 6, generator=g, dtype=torch.float64))[0]
    r = torch.randn(500, 64, generator=g, dtype=torch.float64)
    w = float(((r @ Q) ** 2).sum() / (r**2).sum())
    print(f"    realised w={w:.6f}  -> 1-w={1 - w:.6f}  (in [0,1]: {0 <= 1 - w <= 1})")
    print()

    print("=== Batch-size invariance ===")
    print("  Both errors are means, so a ratio of two means over the SAME axis")
    print("  is invariant to N. Verify 1-w does not drift with the batch size:")
    for N in (64, 512, 4096):
        _Q, zt, zp = build(0.4, D, k, N=N, seed=5)
        r = zp - zt
        w_real = float(((r @ _Q) ** 2).sum() / (r**2).sum())
        print(f"    N={N:<5d} realised w={w_real:.6f}  1-w={1 - w_real:.6f}")


if __name__ == "__main__":
    main()