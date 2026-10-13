"""SWIP / SPC / PCR / CGN metric sweep — metrics inventory driver 4 of 5."""

from __future__ import annotations

import warnings

from mh import guard, ortho_Q, record, save  # noqa: E402

guard()

import torch  # noqa: E402

from src.models.cgn import ContextualGatingNetwork  # noqa: E402
from src.models.pcr import PredictiveCascadeRefinement  # noqa: E402
from src.models.spc import SpectralPredictiveCoding  # noqa: E402
from src.models.swip import SWIPModule  # noqa: E402

B, T, D, K = 4, 32, 64, 6
N = B * T
QF = ortho_Q(D, D, seed=1)


def rnd(shape, seed=None, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(*shape, generator=g, dtype=dtype)


def spec_z(amp_top, amp_bot, seed=700):
    """A representation whose spectrum is `amp_top` in the first K principal
    directions and `amp_bot` everywhere else."""
    a = rnd((B, T, K), seed=seed, dtype=torch.float64) @ QF[:, :K].T
    b = rnd((B, T, D - K), seed=seed + 1, dtype=torch.float64) @ QF[:, K:].T
    return amp_top * a + amp_bot * b


# ══════════════════════════════════════════════════════════════════════════
print("\n=== SWIP ===")
rows = []
for at, ab, name in [(1.0, 1.0, "isotropic"), (10.0, 1.0, "mild peak"),
                     (10.0, 0.1, "strong peak"), (1.0, 10.0, "inverted"),
                     (50.0, 0.01, "very sharp")]:
    z = spec_z(at, ab, seed=710)
    s = SWIPModule(embed_dim=D, k_workspace=K, target_variance=1.0)
    s.freq_basis = None if hasattr(s, "freq_basis") else None
    loss, info = s(z.to(torch.float32))
    rows.append(info)
    print(
        f"  {name:12s}: anisotropy={info['anisotropy_ratio']:.6f}"
        f" bg_uniformity={info['bg_uniformity']:.6f}"
        f" ws_var_frac={info['ws_variance_fraction']:.6f}"
        f" spectral_gap={info['spectral_gap']:.6f} k_ws={info['k_workspace']} bg_dim={info['bg_dim']}"
    )
for key in ["anisotropy_ratio", "bg_uniformity", "ws_variance_fraction", "spectral_gap",
            "k_workspace", "bg_dim"]:
    record("swip", key, [r[key] for r in rows],
           "representation spectrum: iso / peak / sharp peak / inverted / very sharp")

sw = []
for at, ab, name in [(1.0, 1.0, "iso"), (10.0, 0.1, "sharp"), (1.0, 10.0, "inv")]:
    z = spec_z(at, ab, seed=720).to(torch.float32)
    s = SWIPModule(embed_dim=D, k_workspace=K, target_variance=1.0)
    for stp in range(1, 6):
        loss, info = s(z)
    sw.append((info, float(loss)))
    print(
        f"  {name:6s}: loss={float(loss):.6f}"
        f" anisotropy={info['anisotropy_ratio']:.6f} bg_uniformity={info['bg_uniformity']:.6f}"
        f" ws_var_frac={info['ws_variance_fraction']:.6f}"
    )
record("swip", "swip_loss (background term)", [x[1] for x in sw], "same three spectra")

# Isotropy vs anisotropy at a MATCHED mean: the trace-form property.
print("\n  SWIP background term: matched mean, different spread")
bg = []
for spread in [1.0, 2.0, 4.0, 8.0]:
    a = rnd((B, T, K), seed=730, dtype=torch.float64) * spread
    b = rnd((B, T, D - K), seed=731, dtype=torch.float64) / spread
    z = (a @ QF[:, :K].T + b @ QF[:, K:].T).to(torch.float32)
    s = SWIPModule(embed_dim=D, k_workspace=K, target_variance=1.0)
    loss, info = s(z)
    c = z.reshape(-1, D).float()
    c = c - c.mean(0)
    ev = torch.linalg.eigvalsh((c.T @ c) / (c.shape[0] - 1))
    bge = ev[: K]  # ascending: the bottom K are the 'background' under this init
    proof_form = float(
        ((torch.log(bge.clamp(min=1e-12)) - torch.log(torch.tensor(1.0))) ** 2).sum()
    )
    bg.append((spread, float(loss), proof_form, info["bg_uniformity"]))
    print(
        f"    spread={spread:4g}x: swip_loss={float(loss):.6e}"
        f"  proof's sum-of-squares form={proof_form:.6f}  bg_uniformity={info['bg_uniformity']:.4f}"
    )
record("swip", "swip_loss at MATCHED mean, spread 1..8x", [x[1] for x in bg],
       "background spectra with equal mean but 1x..8x anisotropy")
record("swip", "the proof's own sum-of-squares on the same spectra", [x[2] for x in bg], "same")
record("swip", "bg_uniformity at matched mean", [x[3] for x in bg], "same")

fd = []
for at, ab in [(1.0, 1.0), (10.0, 0.1), (1.0, 10.0)]:
    z = spec_z(at, ab, seed=740).to(torch.float32)
    s = SWIPModule(embed_dim=D, k_workspace=K, target_variance=1.0)
    d = s.compute_full_diagnostics(z)
    fd.append(d)
    print(f"  full diagnostics at ({at},{ab}): {sorted(d)}")
for key in sorted(fd[0]):
    record("swip", f"compute_full_diagnostics: {key}", [d.get(key) for d in fd],
           "three spectra")

# ══════════════════════════════════════════════════════════════════════════
print("\n=== SPC ===")
srows = []
for nb in [1, 2, 4, 8, 16]:
    sp = SpectralPredictiveCoding(embed_dim=D, n_bands=nb, init="dct")
    z1 = spec_z(10.0, 1.0, seed=750).to(torch.float32)
    z2 = (z1 + 0.1 * rnd((B, T, D), seed=751)).to(torch.float32)
    loss, info = sp(z1, z2)
    srows.append(info)
    mse = float(((z1 - z2) ** 2).mean())
    print(
        f"  n_bands={nb:3d}: spc_total_loss={info['spc_total_loss']:.6f}"
        f" uniform={info['spc_uniform_loss']:.6f} MSE={mse:.6f}"
        f" SPC/MSE={info['spc_total_loss'] / mse:.4f} SPC/(B*MSE)={info['spc_total_loss'] / (nb * mse):.4f}"
        f" weight_entropy={info['spc_weight_entropy']:.4f}"
        f" n_sig={info['spc_n_significant_bands']}"
        f" tilt={info['spc_spectral_tilt']:.6f} ortho_err={info['spc_ortho_error']:.3e}"
    )
for key in ["spc_total_loss", "spc_uniform_loss", "spc_weight_entropy",
            "spc_n_significant_bands", "spc_spectral_tilt", "spc_ortho_error"]:
    record("spc", key, [r[key] for r in srows], "n_bands 1/2/4/8/16")

_z1 = spec_z(10.0, 1.0, seed=750).to(torch.float32)
_z2 = (_z1 + 0.1 * rnd((B, T, D), seed=751)).to(torch.float32)
_mse = float(((_z1 - _z2) ** 2).mean())
ratios = [r["spc_total_loss"] / _mse for r in srows]
print(f"  SPC/MSE at uniform weights: {ratios}  (n_bands was 1,2,4,8,16; MSE={_mse:.6f})")
record("spc", "SPC / MSE at uniform weights", ratios, "the Parseval scaling claim")
record("spc", "spc_band_weights", [r["spc_band_weights"] for r in srows], "n_bands sweep")
record("spc", "spc_band_predictability", [r["spc_band_predictability"] for r in srows], "same")
record("spc", "spc_band_residuals", [r["spc_band_residuals"] for r in srows], "same")
record("spc", "spc_band_losses", [r["spc_band_losses"] for r in srows], "same")

ba = []
for at, ab in [(1.0, 1.0), (10.0, 1.0), (10.0, 0.1)]:
    z1 = spec_z(at, ab, seed=760).to(torch.float32)
    z2 = (z1 + 0.05 * rnd((B, T, D), seed=761)).to(torch.float32)
    sp = SpectralPredictiveCoding(embed_dim=D, n_bands=8, init="dct")
    for stp in range(3):
        loss, info = sp(z1, z2)
    ba.append(info)
    print(f"  band analysis at ({at},{ab}): {sorted(info)}")
for key in ["spc_weight_entropy", "spc_spectral_tilt", "spc_n_significant_bands"]:
    record("spc", key + " (spectrum sweep)", [r[key] for r in ba], "three spectra")

# ══════════════════════════════════════════════════════════════════════════
print("\n=== PCR ===")
prows = []
for past in [0.0, 0.05, 0.2, 0.5, 1.0]:
    p = PredictiveCascadeRefinement(embed_dim=D, n_levels=3)
    p.warmup_steps = 10
    z1 = spec_z(10.0, 1.0, seed=770).to(torch.float32)
    z2 = (z1 + past * rnd((B, T, D), seed=771)).to(torch.float32)
    out = z1.clone()
    for stp in range(0, 30):
        out, info = p(out, z2, step=stp)
    prows.append(info)
    print(
        f"  |pred-target|={past:4g}: improvement={info['pcr_improvement']:.6e}"
        f" init_res={info['pcr_initial_residual']:.4f} final_res={info['pcr_final_residual']:.4f}"
        f" refinement_norm={info['pcr_total_refinement_norm']:.6f}"
        f" ortho={info['pcr_ortho_score']:.6f} n_levels={info['pcr_n_levels']}"
        f" warmup={info['pcr_warmup_factor']}"
    )
for key in ["pcr_improvement", "pcr_total_refinement_norm", "pcr_initial_residual",
            "pcr_final_residual", "pcr_ortho_score", "pcr_n_levels", "pcr_warmup_factor"]:
    record("pcr", key, [r[key] for r in prows], "prediction error 0..1")

for lv in range(3):
    for suffix in ["correction_norm", "gate", "subspace_fraction", "dim"]:
        vals = [r[f"pcr_level_{lv}_{suffix}"] for r in prows]
        record("pcr", f"pcr_level_{lv}_{suffix}", vals, "prediction error 0..1")

ortho = []
for scale in [1.0, 1.0001, 1.001, 1.01, 1.1, 2.0]:
    p = PredictiveCascadeRefinement(embed_dim=D, n_levels=3)
    p.warmup_steps = 10
    with torch.no_grad():
        p.workspace_Q.mul_(scale)
    z1 = spec_z(10.0, 1.0, seed=780).to(torch.float32)
    z2 = (z1 + 0.2 * rnd((B, T, D), seed=781)).to(torch.float32)
    out = z1.clone()
    for stp in range(20):
        out, info = p(out, z2, step=stp)
    off = float((p.workspace_Q.T @ p.workspace_Q - torch.eye(p.workspace_Q.shape[1])).abs().mean())
    ortho.append(info["pcr_ortho_score"])
    print(f"  Q scaled x{scale:g}: pcr_ortho_score={info['pcr_ortho_score']:.6f} (mean|offdiag|={off:.3e})")
record("pcr", "pcr_ortho_score vs actual off-diagonal mass", ortho, "Q scaled 1x..2x")

cb = []
for past in [0.0, 0.1, 0.5, 1.0]:
    p = PredictiveCascadeRefinement(embed_dim=D, n_levels=3)
    p.warmup_steps = 10
    z1 = spec_z(10.0, 1.0, seed=790).to(torch.float32)
    z2 = (z1 + past * rnd((B, T, D), seed=791)).to(torch.float32)
    b, bi = p.compute_cascade_capacity_bound(z1, z2)
    cb.append((past, b, bi))
    print(f"  |pred-target|={past:4g}: cascade_capacity_bound={b:.6g}  keys={sorted(bi)}")
record("pcr", "compute_cascade_capacity_bound", [x[1] for x in cb], "prediction error 0..1")

# ══════════════════════════════════════════════════════════════════════════
print("\n=== CGN ===")
crows = []
for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]:
    c = ContextualGatingNetwork(embed_dim=D, n_groups=4, tau_start=1.0, tau_end=0.1,
                                anneal_steps=1000)
    z = spec_z(10.0, 1.0, seed=800).to(torch.float32)
    m = torch.zeros(B, T, dtype=torch.bool)
    m[:, : int(frac * T)] = True
    out, info = c(z, m, step=5000)
    crows.append(info)
    print(
        f"  masked={frac:4g}: tau={info['cgn_tau']:.6f} gate_diff={info['cgn_gate_diff']:.6f}"
        f" sparsity={info['cgn_sparsity']:.6f} entropy={info['cgn_entropy']:.6f}"
        f" mean_vis={info['cgn_mean_gate_visible']:.6f}"
        f" mean_masked={info['cgn_mean_gate_masked']:.6f}"
        f" routing_gap={info['cgn_routing_gap']:.6f}"
    )
for key in ["cgn_tau", "cgn_gate_diff", "cgn_sparsity", "cgn_entropy",
            "cgn_mean_gate_visible", "cgn_mean_gate_masked", "cgn_routing_gap"]:
    record("cgn", key, [r[key] for r in crows], "masked fraction 0..0.9")

tau = []
c = ContextualGatingNetwork(embed_dim=D, n_groups=4, tau_start=1.0, tau_end=0.1,
                            anneal_steps=1000)
z = spec_z(10.0, 1.0, seed=810).to(torch.float32)
m = torch.zeros(B, T, dtype=torch.bool)
m[:, :8] = True
for stp in [0, 100, 500, 999, 1000, 5000]:
    out, info = c(z, m, step=stp)
    tau.append(info["cgn_tau"])
    print(f"    step={stp:5d}: cgn_tau={info['cgn_tau']:.6f}")
record("cgn", "cgn_tau vs step", tau, "anneal_steps=1000, tau 1.0 -> 0.1")

dec = []
for lg in [0.0, 1.0, 5.0, 20.0, -20.0]:
    c = ContextualGatingNetwork(embed_dim=D, n_groups=4)
    with torch.no_grad():
        c.gate_logits_visible.copy_(torch.full_like(c.gate_logits_visible, lg))
    v = c.compute_orthogonality_score()
    dec.append(v)
    print(f"    logit={lg:6g}: orthogonality_score (decisiveness)={v:.6f}")
record("cgn", "compute_orthogonality_score (decisiveness)", dec, "gate logit -20..20")

re_ = []
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    c = ContextualGatingNetwork(embed_dim=D, n_groups=4)
    z = spec_z(10.0, 1.0, seed=820).to(torch.float32)
    m = torch.zeros(B, T, dtype=torch.bool)
    m[:, : int(frac * T)] = True
    r = c.compute_routing_efficiency(z, m)
    re_.append(r)
    print(f"  masked={frac:4g}: routing_efficiency={r} ")
for key in ["routing_efficiency", "context_preservation", "prediction_focus"]:
    record("cgn", f"compute_routing_efficiency: {key}", [r.get(key) for r in re_],
           "masked fraction 0..1")

save(r".agent-notes\res_route.json")