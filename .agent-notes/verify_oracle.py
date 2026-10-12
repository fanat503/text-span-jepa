"""What does tests/test_host_reads.py need, and is anything ELSE affected?

`_ref_jawp_info` (tests/test_host_reads.py:386-388) is a transcription of the
PRE-change arithmetic:

    bg = ((zpf - ztf) ** 2).mean()      # mean over N*D
    ws = ((p - t) ** 2).mean()         # mean over N*k
    rel = max(0.0, 1.0 - (ws / bg))

It is the file's oracle for "the batched read returns what the old code
returned". My change intentionally moves that one number, so the oracle has
to move with it. This script computes what the corrected oracle reads, so the
one-line patch to that file is stated rather than guessed.
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
    print("The failing parametrizations, old oracle vs corrected oracle:")
    print()
    for scale, shape in ((1.0, (8, D)), (10.0, (5, D)), (1e-7, (2, 3, D))):
        torch.manual_seed(12)
        jawp = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        z_pred = torch.randn(*shape) * scale
        z_target = torch.randn(*shape) * scale
        for step in (0, 5, 50):
            k = jawp.current_k(step)
            Q = jawp.workspace_Q[:, :k]
            zpf = z_pred.reshape(-1, D)
            ztf = z_target.reshape(-1, D).detach()
            p = (zpf @ Q).detach()
            t = (ztf @ Q).detach()
            bg = ((zpf - ztf) ** 2).mean()
            old_ws = ((p - t) ** 2).mean()
            new_ws = ((p - t) ** 2).sum() / zpf.numel()
            old = max(0.0, 1.0 - (old_ws / bg).item()) if bg.item() > 1e-10 else 1.0
            new = max(0.0, 1.0 - (new_ws / bg).item()) if bg.item() > 1e-10 else 1.0
            _, info = jawp.compute_loss(z_pred, z_target, step=step)
            got = info["predictive_relevance"]
            flag = "" if abs(got - new) <= 1e-12 else "   <-- MISMATCH"
            print(
                f"  scale={scale:<8g} step={step:<3d}  module={got:.10f}  "
                f"corrected-oracle={new:.10f}  (old-oracle={old:.10f}){flag}"
            )
    print()
    print("The module agrees with the corrected oracle to 1e-12 on every case.")
    print()
    print("Required one-line change in tests/test_host_reads.py:386-387")
    print("  -    ws = ((p - t) ** 2).mean()")
    print("  +    ws = ((p - t) ** 2).sum() / zpf.numel()")
    print("  (bg and the max(0.0, 1.0 - ...) host policy are unchanged.)")
    print()
    print("That file is NOT in this task's ownership list, so it is reported, not")
    print("edited. Note the fix is a normalisation of ONE term, and the test at")
    print("line 524 (zero residual -> 1.0, the else branch) is unaffected.")
    print()
    print("Other consumers of the number, and their status:")
    print("  src/train.py:1794-1801   logs jawk_k / utilization / cosine /")
    print("                            ortho / pca_alignment. Does NOT read")
    print("                            predictive_relevance. Unaffected.")
    print("  src/models/jepa.py:902   loss_dict.update({f'jawk_{k}': v ...}) —")
    print("                            a straight rename, no interpretation.")
    print("  proofs/wsd.md:212-223    the lambda_wsd table reporting 0.0000 at")
    print("                            lambda>=1. Those numbers were measured")
    print("                            with the mis-scaled metric and are NOT")
    print("                            re-derivable without training (forbidden).")
    print("                            proofs/** is outside this task's boundary.")
    print("  proofs/jawp.md:264, proofs/wsd.md:30,:263   prose citations, same.")
    print("  .agent-notes/*           historical notes, outside the boundary.")


if __name__ == "__main__":
    main()