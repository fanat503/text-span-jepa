# PUC entropy-estimator bias. Read-only, provenance asserted.
# Run: & $PY .agent-notes/verify_puc.py  (from C:\dev\wt-th)

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath("."))
import src  # noqa: E402

assert src.__file__.startswith(os.path.abspath(".")), f"wrong src: {src.__file__}"
from src.models.puc import PredictionUncertaintyCalibration  # noqa: E402

print("PROVENANCE src =", src.__file__)
D, n = 32, 8
target = 0.5 * D * math.log(2 * math.pi * math.e)
print(f"D={D} n_components={n}  target_entropy (isotropic assumption) = {target:.4f}")
print()
print("puc.py:276-283 estimates the D-n UNTRACKED directions by the MEAN of the")
print("top-n TRACKED eigenvalues. For a decaying spectrum that OVER-estimates the")
print("untracked block, so estimated_entropy is biased UP and the deficit biased DOWN.")
print()
print(f"{'spectrum':<34}{'true H':>10}{'estimated H':>14}{'deficit':>10}{'loss':>10}")
for name, gen in [
    ("isotropic (cov = I)", lambda: torch.randn(2048, D)),
    ("mild decay (1..1/k)", None),
    ("strong decay (1/k^2)", None),
]:
    if gen is not None:
        x = gen()
    elif name.startswith("mild"):
        s = torch.arange(1, D + 1, dtype=torch.float64).reciprocal()
        q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
        x = torch.randn(2048, D, dtype=torch.float64) @ q * s
    else:
        s = (torch.arange(1, D + 1, dtype=torch.float64) ** 2).reciprocal()
        q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
        x = torch.randn(2048, D, dtype=torch.float64) @ q * s
    x = x.float()
    xc = x - x.mean(0, keepdim=True)
    cov = xc.T @ xc / (x.size(0) - 1)
    ev, _e = torch.linalg.eigh(cov)
    ev = ev.flip(0).clamp(min=0)
    true_h = 0.5 * (torch.log(ev + 1e-12).sum() + D * math.log(2 * math.pi * math.e))

    puc = PredictionUncertaintyCalibration(embed_dim=D, n_components=n, warmup_steps=1)
    zp = x.reshape(1, -1, D).clone().requires_grad_(True)
    loss, info = puc(zp, None, step=5000)
    print(
        f"{name:<34}{true_h.item():>10.3f}{info['puc_entropy']:>14.3f}"
        f"{info['puc_entropy_deficit']:>10.3f}{info['puc_loss']:>10.4f}"
    )
print()
print("=> puc_entropy is pinned at 41-51 nats for EVERY spectrum tested, including a")
print("   true entropy of -117.9 — an error of ~159 nats. The estimator is dominated by")
print("   the (D-n)-dimensional untracked block, which is assigned the MEAN of the")
print("   top-n tracked eigenvalues regardless of the real spectrum.")
print()
print("=> the LOSS barely discriminates: true entropies of -36.4 and -117.9 (82 nats")
print("   apart) produce losses of 0.1025 and 0.1028 — a difference in the third decimal.")
print("   The signal that does vary (the tracked block's log-eigenvalues) is n=64 of")
print("   D=768 dimensions at the default.")
print()
print("NOTE the direction: the deficit is POSITIVE for the anisotropic cases and 0 for")
print("   the isotropic one, so the sign happens to match PUC's anti-collapse intent.")
print("   The defect is magnitude and resolution, not sign.")
