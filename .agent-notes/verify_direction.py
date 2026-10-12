"""Is the SIGN inverted, or only the NORMALISATION?

The card asserts the sign is inverted: "reads 1.0 when workspace prediction
is exact and 0.0 when it is worst ... The sign is inverted relative to every
consumer."

That claim is testable, because the consumers all state a direction and one
of them is pinned by a test. This script lays the two candidate readings side
by side against each consumer statement.

  Candidate "quality" reading (what the consumers assume, and what the
  current sign expresses): relevance FALLS as the workspace gets harder to
  predict, so a collapsed workspace reads LOW and a perfect one reads HIGH.

  Candidate "share" reading (what the card's framing implies): relevance is
  the share of residual energy the workspace HOLDS, so it RISES as the
  workspace absorbs more error, and a collapsed workspace reads HIGH.
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


def build(w, D, k, N=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(D, k, generator=g, dtype=torch.float64))[0]
    R = torch.randn(D, D - k, generator=g, dtype=torch.float64)
    R = R - Q @ (Q.T @ R)
    U = torch.linalg.qr(R)[0][:, : D - k]
    z_target = torch.randn(N, D, generator=g, dtype=torch.float64)
    inside = torch.randn(N, k, generator=g, dtype=torch.float64) / (N * k) ** 0.5
    outside = torch.randn(N, D - k, generator=g, dtype=torch.float64) / (N * (D - k)) ** 0.5
    r = inside @ Q.T * (w**0.5) + outside @ U.T * ((1.0 - w) ** 0.5)
    return Q, z_target, z_target + r


def main():
    D, k = 64, 6
    print("What each consumer REQUIRES of the number:\n")
    print("  proofs/wsd.md:30   'collapses to 0.0000 - the workspace stops being")
    print("                     predictable'")
    print("  proofs/wsd.md:222  'collapses to zero - the workspace stops being")
    print("                     predictable at all'")
    print("  proofs/jawp.md:264 'a collapse of jawk_predictive_relevance to 0'")
    print("  src/models/wsd.py:219 'jawp_predictive_relevance collapses to 0'")
    print("  src/models/jawp.py:78 'collapses to 0 and the workspace stops being")
    print("                     predictable at all'")
    print("  tests/test_host_reads.py:524  ZERO residual -> relevance == 1.0")
    print()
    print("  Every one of the five prose sites reads LOW as BAD.")
    print("  The pinned test reads a PERFECT prediction as HIGH (1.0).")
    print("  So the required direction is unambiguous: LOW == workspace is dead.")
    print()
    print("Now: which reading produces that direction?\n")

    print("  w        loss_workspace   1-w  'quality'   w  'share'")
    print("  " + "-" * 60)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        Q, zt, zp = build(w, D, k)
        r = zp - zt
        w_real = float(((r @ Q) ** 2).sum() / (r**2).sum())
        loss_ws = float((((zp @ Q) - (zt @ Q)) ** 2).mean())
        print(
            f"  {w_real:.4f}   {loss_ws:14.6f}   {1 - w_real:7.4f}              "
            f"{w_real:.4f}"
        )
    print()
    print("  'quality' (1-w) FALLS as the workspace gets worse  -> LOW == dead.")
    print("  'share'   (w)   RISES as the workspace gets worse -> HIGH == dead.")
    print()
    print("  The CURRENT code computes max(0, 1 - (D/k)w), which FALLS in w.")
    print("  So the current SIGN already matches every consumer, and matches the")
    print("  pinned test at test_host_reads.py:524.")
    print()
    print("  => The sign is NOT inverted. Only the SCALE is wrong: the D/k factor")
    print("     moves the zero threshold from w=1 to w=k/D (chance), which is why")
    print("     'collapses to 0.0000' is cited as evidence of collapse when it is")
    print("     only evidence that the workspace exceeds a chance-level residual share.")
    print()
    print("  Consequence for the fix: correcting the normalisation to 1-w KEEPS")
    print("  the direction all five prose sites and the one test already assume.")
    print("  Flipping the sign instead would contradict all six.")


if __name__ == "__main__":
    main()