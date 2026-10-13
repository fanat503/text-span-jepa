"""Tautology checks — every pair asserted here is checked by MEASUREMENT."""

from __future__ import annotations

from mh import guard, record, save  # noqa: E402

guard()

import math

import torch  # noqa: E402

from src.models.jawp import JAWPModule  # noqa: E402
from src.models.spc import SpectralPredictiveCoding  # noqa: E402

B, T, D, K = 4, 32, 64, 6
N = B * T


def rnd(n, d, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(n, d, generator=g, dtype=torch.float64)


print("\n=== 1. bg_fraction == 1 - wip_score ? ===")
d1 = []
for seed in range(8):
    m = JAWPModule(embed_dim=D, k_start=K, k_end=K, curriculum_steps=10, alpha=0.1)
    m.workspace_Q.data = m.workspace_Q.data.float()
    torch.manual_seed(seed)
    with torch.no_grad():
        m.workspace_Q.copy_(
            torch.linalg.qr(rnd(D, D, seed=400 + seed).float())[0][:, :K].contiguous()
        )
    _s, inf = m.workspace_information_preservation(rnd(300, D, seed=500 + seed).float(),
                                                   rnd(300, D, seed=600 + seed).float())
    d1.append(abs(inf["bg_fraction"] - (1.0 - inf["wip_score"])))
    print(f"  seed={seed}: wip={inf['wip_score']:.12f} bg={inf['bg_fraction']:.12f}"
          f" |bg-(1-wip)|={d1[-1]:.3e}")
record("tautology", "|bg_fraction - (1 - wip_score)|", d1, "8 random (Q, z) pairs")

print("\n=== 2. rank_utilization == effective_rank / k ? ===")
d2 = []
for r in [1, 2, 3, 5, 6]:
    Qe = torch.linalg.qr(rnd(D, D, seed=700 + r))[0][:, :K].contiguous().float()
    g = torch.Generator().manual_seed(800 + r)
    Mx = torch.randn(r, r, generator=g)
    X = torch.randn(r, 4000, generator=g)
    zp = (Qe[:, :r] @ Mx @ X).T
    m = JAWPModule(embed_dim=D, k_start=K, k_end=K, curriculum_steps=10, alpha=0.1)
    m.workspace_Q.data = m.workspace_Q.data.float()
    with torch.no_grad():
        m.workspace_Q.copy_(Qe)
    ri = m.compute_predictive_rank(zp)
    d2.append(abs(ri["rank_utilization"] - min(ri["effective_rank"] / K, 1.0)))
    print(f"  rank {r}/{K}: eff={ri['effective_rank']:.10f} util={ri['rank_utilization']:.10f}"
          f" eff/k={ri['effective_rank'] / K:.10f} |diff|={d2[-1]:.3e}")
record("tautology", "|rank_utilization - min(effective_rank/k, 1)|", d2, "true rank 1..6")

print("\n=== 3. spc_band_losses == the per-band residual list ? ===")
d3 = []
for nb in [1, 2, 4, 8, 16]:
    sp = SpectralPredictiveCoding(embed_dim=D, n_bands=nb, init="dct")
    z1 = rnd(B * T, D, seed=900 + nb).float()
    z2 = (z1 + 0.1 * rnd(B * T, D, seed=910 + nb)).float()
    _l, info = sp(z1, z2)
    br, bl = info["spc_band_residuals"], info["spc_band_losses"]
    d3.extend([abs(a - b) for a, b in zip(br, bl)])
    print(f"  n_bands={nb:3d}: band_residuals={br} band_losses={bl}")
record("tautology", "|spc_band_residual - spc_band_loss| per band", d3,
       "n_bands 1..16")

print("\n=== 4. spc_total_loss == sum(weights * band_losses) / ? ===")
d4 = []
for nb in [2, 4, 8]:
    sp = SpectralPredictiveCoding(embed_dim=D, n_bands=nb, init="dct")
    z1 = rnd(B * T, D, seed=920 + nb).float()
    z2 = (z1 + 0.1 * rnd(B * T, D, seed=930 + nb)).float()
    _l, info = sp(z1, z2)
    w = info["spc_band_weights"]
    rec = sum(wi * li for wi, li in zip(w, info["spc_band_losses"])) / sum(w)
    d4.append(abs(rec - info["spc_total_loss"]))
    print(f"  n_bands={nb:3d}: spc_total_loss={info['spc_total_loss']:.10f}"
          f"  sum(w*l)/sum(w)={rec:.10f}  |diff|={d4[-1]:.3e}")
record("tautology", "|spc_total_loss - weighted mean of band_losses|", d4, "n_bands 2/4/8")

save(r".agent-notes\res_taut.json")