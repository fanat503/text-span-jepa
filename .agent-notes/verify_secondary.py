"""Verification for the secondary findings the card asked about:

  1. wsd_steady_state_error == wsd_running_drift exactly (a tautology).
  2. wsd_lambda_estimate == 1/sync_interval, not lambda_wsd.
  3. jawp.detect_workspace_dimension(min_gap_ratio=...) — is the parameter
     ever read?
  4. compute_background_complexity's ratio — is it on a single axis?

REPORT ONLY. Nothing here is fixed by this task.
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
from src.models.wsd import WorkspaceSyncDrift  # noqa: E402


def main():
    print("=== 1. wsd_steady_state_error vs wsd_running_drift (D=16, k=3, T=7) ===")
    torch.manual_seed(0)
    wsd = WorkspaceSyncDrift(embed_dim=16, k=3, sync_interval=7)
    wsd.train()
    D, k = 16, 3
    Q = torch.linalg.qr(torch.randn(D, k))[0]
    worst = 0.0
    for step in range(12):
        h = torch.randn(8, D)
        _loss, info = wsd.compute_drift(Q, h_target=h, step=step)
        worst = max(worst, abs(info["wsd_steady_state_error"] - info["wsd_running_drift"]))
    print(f"   max |steady_state_error - running_drift| over 12 steps = {worst:.3e}")
    print("   -> identical: the 'constructive bound' is the running-drift EMA again")
    print()

    print("=== 2. wsd_lambda_estimate vs 1/sync_interval and vs lambda_wsd ===")
    for interval in (7, 50, 100):
        wsd = WorkspaceSyncDrift(embed_dim=16, k=3, sync_interval=interval)
        wsd.train()
        Q = torch.linalg.qr(torch.randn(D, k))[0]
        _loss, info = wsd.compute_drift(Q, h_target=torch.randn(8, D), step=0)
        lam = info["wsd_lambda_estimate"]
        print(
            f"   sync_interval={interval:<4d}  wsd_lambda_estimate={lam:.6f}  "
            f"1/sync_interval={1.0 / interval:.6f}  "
            f"defaults lambda_wsd=0.01 -> {'matches' if abs(lam - 0.01) < 1e-12 else 'DIVERGES'}"
        )
    print("   -> the reported lambda is 1/T, a function of the sync interval only.")
    print("      It coincides with lambda_wsd=0.01 at the shipped default and")
    print("      diverges on any config that moves either value.")
    print()

    print("=== 3. is min_gap_ratio ever read? ===")
    src_txt = open(
        os.path.join(REPO, "src", "models", "jawp.py"), encoding="utf-8"
    ).read()
    print(f"   occurrences of 'min_gap_ratio' in jawp.py: {src_txt.count('min_gap_ratio')}")
    print("   (1 = the parameter in the signature only; 2+ would mean it is used)")
    torch.manual_seed(0)
    jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
    z_t, z_p = torch.randn(200, 32), torch.randn(200, 32)
    a, _ = jawp.detect_workspace_dimension(z_p, z_t, min_gap_ratio=1.01)
    b, _ = jawp.detect_workspace_dimension(z_p, z_t, min_gap_ratio=1000.0)
    c, _ = jawp.detect_workspace_dimension(z_p, z_t)
    print(f"   min_gap_ratio=1.01 -> k*={a};  min_gap_ratio=1000 -> k*={b};  default -> k*={c}")
    print(f"   all three equal: {a == b == c}  -> the parameter changes nothing")
    print()

    print("=== 4. bg_complexity_ratio axis check (D=64, k=6) ===")
    torch.manual_seed(2)
    D2, k2 = 64, 6
    # float32 throughout: compute_background_complexity casts its inputs with
    # .float() (jawp.py:834-835), so a float64 module would mismatch.
    Q2 = torch.linalg.qr(torch.randn(D2, k2))[0]
    torch.manual_seed(4)
    R = torch.linalg.qr(torch.randn(D2, D2 - k2))[0]
    R = R - Q2 @ (Q2.T @ R)
    U = torch.linalg.qr(R)[0][:, : D2 - k2]
    z_t2 = torch.randn(256, D2)
    for w in (0.1, 0.3, 0.6, 0.9):
        inside = torch.randn(256, k2)
        outside = torch.randn(256, D2 - k2)
        s = (w / (1 - w)) ** 0.5
        resid = (inside @ Q2.T) * s + outside @ U.T
        z_p2 = z_t2 + resid
        jawp = JAWPModule(embed_dim=D2, k_start=k2, k_end=k2, curriculum_steps=0)
        with torch.no_grad():
            jawp.workspace_Q.copy_(Q2)
        got, info = jawp.compute_background_complexity(z_p2, z_t2)
        # The stated quantity: background residual / workspace residual, on ONE axis.
        true_ratio = float(info["bg_residual"] / info["ws_residual"]) if info["ws_residual"] else float(
            "nan"
        )
        # What the docstring's words mean in energy terms: (1-w)/w.
        energy_ratio = (1 - w) / w
        print(
            f"   realised w={w:<4g}  bg_complexity={got:10.5f}  "
            f"bg_res/ws_res(same axis)={true_ratio:10.5f}  "
            f"reported/same-axis={got / true_ratio:.5f}  k/D={k2 / D2:.5f}"
        )
    print("   -> compute_background_complexity mixes an N*k mean with an N*D mean,")
    print("      the SAME axis bug as predictive_relevance (jawp.py:840 vs :846).")
    print("      A constant factor, so ORDERING is preserved; only the scale moves.")


if __name__ == "__main__":
    main()