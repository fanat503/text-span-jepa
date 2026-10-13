"""Strictness pass — resolve the grades that were still judgement calls."""

from __future__ import annotations

from mh import guard, ortho_Q, record, save  # noqa: E402

guard()

import torch  # noqa: E402

from src.models.puc import PredictionUncertaintyCalibration  # noqa: E402
from src.models.spc import SpectralPredictiveCoding  # noqa: E402
from src.models.swip import SWIPModule  # noqa: E402

B, T, D, K = 4, 32, 64, 6


def rnd(*shape, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(*shape, generator=g)


print("\n=== A. spc_ortho_error under a genuine off-diagonal in freq_basis ===")
v = []
for eps in [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
    sp = SpectralPredictiveCoding(embed_dim=D, n_bands=8, init="dct")
    with torch.no_grad():
        sp.freq_basis.add_(eps * rnd(D, D, seed=1000))
    z1 = rnd(B * T, D, seed=1001)
    z2 = z1 + 0.1 * rnd(B * T, D, seed=1002)
    _l, info = sp(z1, z2)
    v.append(info["spc_ortho_error"])
    print(f"  eps={eps:8g}: spc_ortho_error={info['spc_ortho_error']:.6e}")
record("strict", "spc_ortho_error vs injected off-diagonal", v, "eps 0..0.1")

print("\n=== B. spc_uniform_loss == mean(spc_band_residuals) ? ===")
d = []
for nb in [1, 2, 4, 8, 16]:
    sp = SpectralPredictiveCoding(embed_dim=D, n_bands=nb, init="dct")
    z1 = rnd(B * T, D, seed=1010 + nb)
    z2 = z1 + 0.1 * rnd(B * T, D, seed=1020 + nb)
    _l, info = sp(z1, z2)
    m = sum(info["spc_band_residuals"]) / len(info["spc_band_residuals"])
    d.append(abs(info["spc_uniform_loss"] - m))
    print(f"  n_bands={nb:3d}: uniform_loss={info['spc_uniform_loss']:.10f}"
          f"  mean(band_residuals)={m:.10f}  |diff|={d[-1]:.3e}")
record("strict", "|spc_uniform_loss - mean(spc_band_residuals)|", d, "n_bands 1..16")

print("\n=== C. spc_band_predictability over steps ===")
v = []
sp = SpectralPredictiveCoding(embed_dim=D, n_bands=4, init="dct")
for stp in range(12):
    z1 = rnd(B * T, D, seed=1100 + stp)
    z2 = z1 + 0.05 * rnd(B * T, D, seed=1200 + stp)
    _l, info = sp(z1, z2)
    v.append(round(sum(info["spc_band_predictability"]), 8))
    print(f"  step={stp:3d}: sum(band_predictability)={v[-1]:.8f}  {info['spc_band_predictability']}")
record("strict", "sum(spc_band_predictability) over 12 steps", v, "four bands")

print("\n=== D. swip condition_number == anisotropy_ratio ? ===")
d = []
for at, ab in [(1.0, 1.0), (10.0, 0.1), (1.0, 10.0), (0.1, 0.1), (100.0, 1.0)]:
    QF = ortho_Q(D, D, seed=5).float()
    z = (
        at * (rnd(B, T, K, seed=1300) @ QF[:, :K].T)
        + ab * (rnd(B, T, D - K, seed=1301) @ QF[:, K:].T)
    )
    s = SWIPModule(embed_dim=D, k_workspace=K, target_variance=1.0)
    fd = s.compute_full_diagnostics(z)
    _l, info = s(z)
    d.append(abs(fd["condition_number"] - fd["anisotropy_ratio"]))
    print(
        f"  ({at},{ab}): full.condition_number={fd['condition_number']:.6f}"
        f"  full.anisotropy_ratio={fd['anisotropy_ratio']:.6f}"
        f"  |diff|={d[-1]:.3e}   forward.anisotropy_ratio={info['anisotropy_ratio']:.6f}"
    )
record("strict", "|swip condition_number - anisotropy_ratio| (full diagnostics)", d, "five spectra")

print("\n=== E. puc_overconfidence == deficit / target ? ===")
QF = ortho_Q(D, D, seed=5).double()
d = []
for decay in [0.0, 1.0, 2.0, 4.0, 8.0]:
    scale = torch.tensor([0.5 ** (i * decay / (D - 1)) for i in range(D)], dtype=torch.float64)
    z = rnd(B, T, D, seed=1400).double() * scale
    p = PredictionUncertaintyCalibration(
        embed_dim=D, eta=0.01, ema_beta=0.999, warmup_steps=500
    ).double()
    _l, i2 = p(z, step=2000)
    expect = i2["puc_entropy_deficit"] / (i2["puc_target_entropy"] + 1e-8)
    d.append(abs(i2["puc_overconfidence"] - expect))
    print(
        f"  decay={decay:4g}: overconfidence={i2['puc_overconfidence']:.10f}"
        f"  deficit/target={expect:.10f}  |diff|={d[-1]:.3e}"
    )
record("strict", "|puc_overconfidence - deficit/target|", d, "decay 0..8")

print("\n=== F. swip bg_variance_fraction == 1 - ws_variance_fraction ? ===")
d = []
for at, ab in [(1.0, 1.0), (10.0, 1.0), (10.0, 0.1), (1.0, 10.0), (5.0, 0.2)]:
    QF = ortho_Q(D, D, seed=6).float()
    z = (
        at * (rnd(B, T, K, seed=1500) @ QF[:, :K].T)
        + ab * (rnd(B, T, D - K, seed=1501) @ QF[:, K:].T)
    )
    s = SWIPModule(embed_dim=D, k_workspace=K, target_variance=1.0)
    fd = s.compute_full_diagnostics(z)
    d.append(abs(fd["bg_variance_fraction"] - (1.0 - fd["ws_variance_fraction"])))
    print(
        f"  ({at},{ab}): bg_frac={fd['bg_variance_fraction']:.8f}"
        f"  1-ws_frac={1 - fd['ws_variance_fraction']:.8f}  |diff|={d[-1]:.3e}"
    )
record("strict", "|swip bg_variance_fraction - (1 - ws_variance_fraction)|", d, "five spectra")

save(r".agent-notes\res_strict.json")