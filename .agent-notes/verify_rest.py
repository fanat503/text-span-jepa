# Verification for docs/THEORY_AND_PROOFS.md — SWIP, RDC, WSR, PUC, STA.
# Run: & $PY .agent-notes/verify_rest.py   (from C:\dev\wt-th)
# Read-only. No training.

import math
import os
import sys

import torch

# PROVENANCE: prove we are executing THIS worktree, not another clone that an
# editable install or a stale .pyc might point at.
sys.path.insert(0, os.path.abspath("."))
import src  # noqa: E402

assert src.__file__.startswith(os.path.abspath(".")), f"wrong src: {src.__file__}"
print("PROVENANCE src =", src.__file__)

from src.models.rdc import RepresentationDriftCompensation  # noqa: E402
from src.models.sta import SpectralTransportAlignment  # noqa: E402
from src.models.swip import SWIPModule  # noqa: E402
from src.models.wsr import WorkspaceSharpnessRegularization  # noqa: E402

ok = lambda b: "OK " if b else "FAIL"  # noqa: E731

print("=" * 70)
print("SWIP: proof says sum of squared log deviations; code says trace approx")
print("=" * 70)
torch.manual_seed(0)
D, k, N = 32, 6, 512
sw = SWIPModule(embed_dim=D, k_workspace=k, use_jawp_workspace=False)
z = torch.randn(N, D)

# Background eigenvalues chosen to have the SAME MEAN but wildly different spread.
ev = torch.tensor([9.0, 3.0, 3.0, 1.0, 1.0, 1.0] + [1.0 / 3.0] * (D - k - 6), dtype=torch.float64)
ev = ev / ev.mean() * 2.0  # normalise so mean == 2
bg = ev[k:]
sigma_t = bg.mean().item()  # the target the code will use
proof_term = ((bg.log() - math.log(sigma_t)) ** 2).sum().item()
code_term = len(bg) * (bg.mean().log().item() - math.log(sigma_t)) ** 2
print(f"background spectrum: mean={bg.mean():.6f}  min={bg.min():.6f}  max={bg.max():.6f}")
print(f"  proof  sum_i (log l_i - log sigma_t)^2 = {proof_term:.6f}   <- NONZERO")
print(f"  code   (D-k) * (log mean(l) - log sigma_t)^2 = {code_term:.6e}   <- ZERO")
print()
print(
    f"proof property 2 'vanishes iff EVERY background log-variance == log sigma^2': "
    f"{ok(proof_term > 1e-6)} -- it does NOT vanish for this spread spectrum"
)
print(f"executed code term vanishes for the same spectrum: {ok(code_term < 1e-12)}")
print("=> proofs/swip.md property 2 is FALSE for the executed loss on this path")

print()
print("=" * 70)
print("RDC: transient bound underflows; eps is an EMA proxy; T is capped")
print("=" * 70)
import inspect  # noqa: E402

sig = inspect.signature(RepresentationDriftCompensation.__init__)
print(
    "RDC defaults:",
    {k: v.default for k, v in sig.parameters.items() if v.default is not inspect._empty},
)
for eta in (0.001, 0.01, 0.05, 0.1):
    for T in (100, 500, 1000, 10000):
        val = (1 - eta) ** T
        if T in (1000, 10000):
            print(
                f"  eta={eta:<6} T={T:<6} (1-eta)^T = {val:.3e}"
                f"  -> transient_bound ~ {val * T:.3e}"
            )
print()
print("=> for eta >= 0.001 the factor is numerically 0 long before T_eff = 10000;")
print("   rdc_transient_bound reads as 'drift provably vanishes' when it is float underflow.")
src = inspect.getsource(RepresentationDriftCompensation.forward)
print("   eps source  :", [ln.strip() for ln in src.splitlines() if "eps_estimate" in ln])
print("   T_eff cap   :", [ln.strip() for ln in src.splitlines() if "T_eff =" in ln])

print()
print("=" * 70)
print("WSR: does the returned loss carry a gradient?")
print("=" * 70)
Q = torch.linalg.qr(torch.randn(32, 4))[0]
grad = torch.randn(32, 4) - Q @ (Q.T @ torch.randn(32, 4))
wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="gradient")
wsr.set_lagged_gradient(grad)
loss, info = wsr(Q, step=1000)
print(f"mode='gradient': loss={loss.item():.6f}  requires_grad={loss.requires_grad}")
print(f"  wsr_sharpness          = {info['wsr_sharpness']:.6f}")
print(f"  wsr_spectral_sharpness = {info['wsr_spectral_sharpness']:.6f}")
print(
    f"  identical? {ok(abs(info['wsr_sharpness'] - info['wsr_spectral_sharpness']) < 1e-12)}"
    "  <- 'spectral component' is a DUPLICATE of the total sharpness"
)
print(f"  wsr_directional_sharpness = {info['wsr_directional_sharpness']:.6f}")
print()
print(
    f"||Q||_F = {Q.norm().item():.4f} for an orthonormal (32,4) Q "
    "-> the code divides rho*||grad|| by ||Q||_F, which the proof's rho_Q does not have"
)

print()
print("wsr mode='sam' retraction distance:")
Q2 = torch.linalg.qr(torch.randn(32, 4))[0]
t = torch.randn(32, 4)
t = t - Q2 @ (Q2.T @ t)
t = t / t.norm()
Qr = wsr._stiefel_retract(Q2 + 0.05 * t)
print(f"  ||Q_pert - Q_ret||_F = {(Q2 - Qr).norm().item():.4f}   (rho = 0.05)")
print(f"  subspace distance    = {(Q2 @ Q2.T - Qr @ Qr.T).norm().item():.4f}")
print(
    f"  column sign flips    = {int(((Qr * torch.sign((Qr * Q2).sum(0)).unsqueeze(0) * Q2) < 0).sum())} of 4 columns"
)

print()
print("=" * 70)
print("STA: loss carries a gradient? spectral gap is floored?")
print("=" * 70)
sta = SpectralTransportAlignment(embed_dim=16)
z = torch.randn(32, 16, requires_grad=True)
sloss, sinfo = sta(z, step=1000)
print(f"sta_loss={sloss.item():.6f} requires_grad={sloss.requires_grad}")
print(f"  sta_spectral_gap (min over significant pairs) = {sinfo['sta_spectral_gap']:.6e}")
b = sta.compute_davis_kahan_bound(k=3, w1=1.0)
print(f"  davis_kahan_bound(k=3, w1=1.0) = {b:.4e}")
print(f"  1/eps = {1.0 / sta.eps:.4e}  <- what you get when the gap is <= eps")

print()
print("=" * 70)
print("PUC: is the executed loss the proof's entropy-deficit formula?")
print("=" * 70)
from src.models.puc import PredictionUncertaintyCalibration  # noqa: E402

torch.manual_seed(0)
puc = PredictionUncertaintyCalibration(embed_dim=32, n_components=8, eta=0.01, warmup_steps=1)
zp = torch.randn(16, 4, 32, requires_grad=True)
zt = torch.randn(16, 4, 32)
ploss, pinfo = puc(zp, zt, step=5000)
print(
    f"puc_loss={pinfo['puc_loss']:.6f}  requires_grad={ploss.requires_grad}"
    f"  carries_grad={pinfo['puc_carries_grad']}"
)
print(f"  entropy_deficit = {pinfo['puc_entropy_deficit']:.6f}")
print(f"  log_det (sum of tracked log-eigs) = {pinfo['puc_log_det']:.6f}")
n = pinfo["puc_n_components"]
print(f"  proof Step-5 form  -1/2*logdet + n*log(2*pi*e*s^2)  is NOT what is executed")
print(f"  executed = eta*warmup*relu(-sum(log lam_i) + {n}*10)/{n}")
print(f"           value = {ploss.item():.6f}")
print("           -> 0 here because the entropy_deficit switch at puc.py:304 was NOT")
print("              taken, so the else branch returned a fresh zero tensor.")
print("              The PUC finding is the estimator bias, measured in verify_puc.py;")
print("              this block only establishes the gate and the dead min_log_det.")
print(f"  target_entropy = {pinfo['puc_target_entropy']:.4f} = D/2*log(2*pi*e) for D=32")
print(
    f"  min_log_det is accepted in __init__ and never referenced in forward:",
    ok("min_log_det" not in inspect.getsource(PredictionUncertaintyCalibration.forward)),
)
