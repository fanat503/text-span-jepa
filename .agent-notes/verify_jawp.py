# Verification for docs/THEORY_AND_PROOFS.md - JAWP predictive_relevance.
# Run: & $PY .agent-notes/verify_jawp.py   (from C:\dev\wt-th)
# Read-only. No training.

import os
import sys

import torch

sys.path.insert(0, os.path.abspath("."))
import src  # noqa: E402

assert src.__file__.startswith(os.path.abspath(".")), f"wrong src: {src.__file__}"
print("PROVENANCE src =", src.__file__)

from src.models.jawp import JAWPModule  # noqa: E402

print("=== 1. predictive_relevance: is the ratio normalised by k and D consistently? ===")
print("jawp.py:463  ws_pred_error  = ((pred_ws_d - target_ws_d)**2).mean()   -> mean over N*k")
print("jawp.py:462  bg_pred_error  = ((z_pred_flat - z_target_flat)**2).mean() -> mean over N*D")
print("jawp.py:503  predictive_relevance = max(0, 1 - ws/bg)")
print()
print("=> pred_rel_div = (D/k) * (workspace share of residual energy), NOT the share itself.")
print()

D, k, N = 64, 6, 256
torch.manual_seed(0)
Q = torch.linalg.qr(torch.randn(D, k))[0]

# Construct residual r with a controlled fraction w of its energy in span(Q).
P = Q @ Q.T
for w in [0.0, 0.02, 0.05, 0.094, 0.15, 0.3, 0.6, 1.0]:
    r_ws = torch.randn(N, k)
    r_bg = torch.randn(N, D - k)
    r = r_ws @ Q.T + r_bg @ (torch.linalg.qr(torch.randn(D, D - k))[0]).T
    # rescale so exactly a fraction w of total energy lies in span(Q)
    e_ws = (r @ Q).pow(2).sum()
    e_tot = r.pow(2).sum()
    scale_ws = (w / (1 - w)) * (e_tot / max(e_ws, 1e-12)) if w < 1.0 else 1e3
    r = (r @ Q) * scale_ws @ Q.T + (r - (r @ Q) @ Q.T)
    e_ws = (r @ Q).pow(2).sum()
    e_tot = r.pow(2).sum()
    share = (e_ws / e_tot).item()

    z_target = torch.zeros(N, D)
    z_pred = r  # residual r = z_pred - z_target

    jawp = JAWPModule(embed_dim=D, k_start=k, k_end=k, alpha=0.1)
    with torch.no_grad():
        jawp.workspace_Q.copy_(Q)
    _loss, info = jawp.compute_loss(z_pred, z_target, step=10**9)

    pred = info["predictive_relevance"]
    expect = max(0.0, 1.0 - (D / k) * share)
    print(
        f"workspace energy share w={share:.4f}  "
        f"predictive_relevance={pred:.4f}  "
        f"predicted max(0,1-(D/k)w)={expect:.4f}  match={abs(pred - expect) < 1e-4}"
    )

print()
print("=== 2. Direction of the metric ===")
print("w = 0.00  -> residual ORTHOGONAL to span(Q) -> Q^T r = 0 -> workspace prediction is")
print("             EXACT, and the workspace has nothing to predict -> metric reads 1.0 (max)")
print("w >= k/D  -> metric reads 0.0, including w = 1.0 where workspace prediction is WORST")
print()
print("chance level k/D =", k / D)

print()
print("=== 3. The two endpoints, constructed exactly ===")
print("(section 1's sweep is the general case; these are w=1 and w=0 exactly)")
print("NOTE: 'workspace prediction quality' = Q^T(z_pred - z_target) being small.")
print("      w=0 -> the residual is ORTHOGONAL to span(Q) -> Q^T r = 0 -> workspace")
print("             prediction is EXACT (and vacuous).")
print("      w=1 -> the residual is INSIDE span(Q)       -> Q^T r = r -> workspace")
print("             prediction is as WRONG as it can be.")
jw = JAWPModule(embed_dim=D, k_start=k, k_end=k, alpha=0.1)
# U must be the orthogonal COMPLEMENT of this exact Q, not an independent draw.
_compl = torch.eye(D) - Q @ Q.T
U = torch.linalg.qr(_compl)[0][:, : D - k]
assert (U.T @ Q).abs().max() < 1e-5, "U is not orthogonal to Q"
assert torch.allclose(U.T @ U, torch.eye(D - k), atol=1e-5), "U is not orthonormal"
print(f"checked: U (D, D-k) is exactly orthogonal to Q; max |U^T Q| = {(U.T @ Q).abs().max():.2e}")
with torch.no_grad():
    jw.workspace_Q.copy_(Q)

# w = 0: residual entirely OUTSIDE span(Q). Workspace prediction is exact.
r_bg = torch.randn(N, D - k) @ U.T
_loss, info = jw.compute_loss(r_bg, torch.zeros(N, D), step=10**9)
print()
print("w = 0.0  residual entirely OUTSIDE span(Q):")
print(f"   loss_workspace       = {info['loss_workspace']:.3e}   (~zero => prediction EXACT)")
print(f"   predictive_relevance = {info['predictive_relevance']:.4f}   <- reads MAXIMAL")

# w = 1: residual entirely INSIDE span(Q). Workspace prediction is worst possible.
r_ws = torch.randn(N, k) @ Q.T
_loss, info = jw.compute_loss(r_ws, torch.zeros(N, D), step=10**9)
print()
print("w = 1.0  residual entirely INSIDE span(Q):")
print(f"   loss_workspace       = {info['loss_workspace']:.3e}   (large => prediction WORST)")
print(f"   predictive_relevance = {info['predictive_relevance']:.4f}   <- reads ZERO")

print()
print("=> the metric reads 1.0 exactly when workspace prediction is EXACT, and 0.0")
print("   exactly when workspace prediction is WORST. It is inverted.")
