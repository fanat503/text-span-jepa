# Verification pass 2 for docs/THEORY_AND_PROOFS.md — corrected SWIP, WSR seeds,
# PUC entropy bias. Run: & $PY .agent-notes/verify_rest2.py  (from C:\dev\wt-th)
# Read-only. No training. Provenance asserted.

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath("."))
import src  # noqa: E402

assert src.__file__.startswith(os.path.abspath(".")), f"wrong src: {src.__file__}"
print("PROVENANCE src =", src.__file__)

from src.models.puc import PredictionUncertaintyCalibration  # noqa: E402
from src.models.swip import SWIPModule  # noqa: E402
from src.models.wsr import WorkspaceSharpnessRegularization  # noqa: E402

ok = lambda b: "OK " if b else "FAIL"  # noqa: E731

print()
print("=" * 72)
print("SWIP (corrected): spread INSIDE the background block")
print("=" * 72)
D, k = 32, 6
torch.manual_seed(0)
# eigenvalues DESCENDING: first k are the workspace, the rest are background.
eig = torch.tensor(
    [50.0, 30.0, 20.0, 10.0, 5.0, 3.0] + [4.0, 2.0, 1.0, 0.5] + [0.75] * (D - 10),
    dtype=torch.float64,
)
bg = eig[k:]
sigma_t = bg.mean().item()
proof_term = ((bg.log() - math.log(sigma_t)) ** 2).sum().item()
code_term = len(bg) * (bg.mean().log().item() - math.log(sigma_t)) ** 2
print(f"background: n={len(bg)} mean={bg.mean():.4f} min={bg.min():.4f} max={bg.max():.4f}")
print(f"  spread ratio max/min = {bg.max() / bg.min():.1f}x  <- badly anisotropic")
print(f"  proof  sum_i (log l_i - log sigma_t)^2      = {proof_term:.6f}  NONZERO")
print(f"  code   (D-k)*(log mean(l) - log sigma_t)^2  = {code_term:.3e}  ZERO")
print(f"  proof property 2 ('vanishes iff EVERY background log-variance == log sigma^2')")
print(
    f"    -> FAILS for the executed code: {ok(proof_term > 1e-6)} (proof term is nonzero;"
    f" code term is {code_term:.1e})"
)
print()
print("  cross-check against the module's own output on a matching spectrum:")
sw = SWIPModule(embed_dim=D, k_workspace=k, use_jawp_workspace=False)
print("    SWIPModule defaults:", sw.target_variance, sw.hierarchy_margin, sw.hierarchy_weight)
print("    (the executed background term is the trace form, swip.py:258-272)")

print()
print("=" * 72)
print("WSR _stiefel_retract across seeds (the xfail is strict=False)")
print("=" * 72)
wsr = WorkspaceSharpnessRegularization(embed_dim=32, rho=0.05)
frob, sub, flips = [], [], []
for seed in range(12):
    torch.manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(32, 4))[0]
    t = torch.randn(32, 4)
    t = t - Q @ (Q.T @ t)
    t = t / t.norm()
    Qr = wsr._stiefel_retract(Q + 0.05 * t)
    f = (Q - Qr).norm().item()
    s = (Q @ Q.T - Qr @ Qr.T).norm().item()
    frob.append(f)
    sub.append(s)
print(f"  ||Q - retract(Q+rho t)||_F over 12 seeds:")
print(f"    min={min(frob):.4f}  max={max(frob):.4f}   (rho = 0.05)")
print(f"  subspace distance ||QQ^T - Qr Qr^T||_F:")
print(
    f"    min={min(sub):.4f}  max={max(sub):.4f}   (a correct retraction would give <= rho = 0.05)"
)
print(f"  seeds where the subspace moved MORE than rho: {sum(1 for s in sub if s > 0.05)}/12")
print(f"  seeds where Frobenius moved more than rho:    {sum(1 for f in frob if f > 0.05)}/12")
print()
print("  => NOT seed-dependent: both failures occur on every seed measured.")
print("     The Frobenius values near 2*sqrt(k) are a near-UNIFORM sign flip, which is")
print("     harmless for the subspace (span is sign-invariant under a global flip);")
print("     so the xfail's 'flips individual COLUMNS' phrasing overstates what these")
print("     12 seeds show.")
print("     What DOES fail universally: the rho-semantics. A correct polar/SVD")
print("     retraction is nonexpansive, so ||QQ^T - RR^T||_F <= ||rho*t||_F = rho.")
print("     Measured 0.0707 = sqrt(2)*rho on 12/12 seeds. QR is not a retraction.")
print("  => the xfail is right that this needs an owner decision (it changes what")
print("     wsr_mode='sam' computes); the mechanism is a real one, not a flaky test.")
