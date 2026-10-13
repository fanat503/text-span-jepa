"""WSD / STA / RDC / WSR metric sweep вЂ” metrics inventory driver 2 of 5."""

from __future__ import annotations

import math

from mh import guard, ortho_Q, record, save  # noqa: E402

guard()

from torch import float64  # noqa: E402

import torch  # noqa: E402

from src.models.rdc import RepresentationDriftCompensation  # noqa: E402
from src.models.sta import SpectralTransportAlignment  # noqa: E402
from src.models.wsd import WorkspaceSyncDrift  # noqa: E402
from src.models.wsr import WorkspaceSharpnessRegularization  # noqa: E402

B, T, D, K = 4, 32, 64, 6
N = B * T


def r32(n, d, seed=None, dtype=torch.float64):
    """float64 by default: every module in this driver is cast to float64."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    if isinstance(n, tuple):  # r32((B, T, D), seed=...)
        shape, n = n, d
        return torch.randn(*shape, generator=g, dtype=dtype)
    return torch.randn(n, d, generator=g, dtype=dtype)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
print("\n=== WSD ===")
QF = ortho_Q(D, D, seed=1)


def qrot(ang, seed=1, k=K):
    """Orthonormal Q whose span is rotated `ang` away from QF[:, :k]."""
    Q1 = QF[:, :k].contiguous()
    v = QF[:, k]
    cols = [math.cos(ang) * Q1[:, 0] + math.sin(ang) * v] + [Q1[:, i] for i in range(1, k)]
    return torch.linalg.qr(torch.stack(cols, dim=1))[0][:, :k].contiguous()


ANGS = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5]
drift, drift_sq, run_drift, ovl, sal, mpa, mpa_mean, kk = ([] for _ in range(8))
sse, nu, conv, lam = [], [], [], []
# The target workspace is set once, from a target whose top-k PCA is exactly
# span(QF[:, :k]); the JAWP workspace is then rotated away from it by `a`.
# (If h_target is random the two subspaces are unrelated and every diagnostic
# saturates вЂ” which is why the first attempt of this sweep read flat.)
HT = 1.0 * (r32(N, K, seed=89) @ QF[:, :K].T.double())
for i, a in enumerate(ANGS):
    w = WorkspaceSyncDrift(embed_dim=D, k=K, sync_interval=5)
    w = w.to(torch.float64)
    _l, _i = w.compute_drift(QF[:, :K].contiguous(), HT, step=0)  # resync
    for step in range(1, 5):
        _l, info = w.compute_drift(qrot(a), None, step=step)
    drift.append(info["wsd_drift"])
    drift_sq.append(info["wsd_drift_sq"])
    run_drift.append(info["wsd_running_drift"])
    ovl.append(info["wsd_overlap"])
    sal.append(info["wsd_spectral_alignment"])
    mpa.append(info["wsd_max_principal_angle"])
    mpa_mean.append(info["wsd_mean_principal_angle"])
    kk.append(info["wsd_k"])
    sse.append(info["wsd_steady_state_error"])
    nu.append(info["wsd_nu_max_estimate"])
    conv.append(info["wsd_convergence_time"])
    lam.append(info["wsd_lambda_estimate"])
    print(
        f"  rot={a:4.2f}: drift={info['wsd_drift']:.6f} overlap={info['wsd_overlap']:.4f}"
        f" maxang={info['wsd_max_principal_angle']:.6f} sse={info['wsd_steady_state_error']:.6f}"
    )
record("wsd", "wsd_drift", drift, "Q rotated 0..1.5 rad from the target workspace")
record("wsd", "wsd_drift_sq", drift_sq, "same")
record("wsd", "wsd_running_drift", run_drift, "same")
record("wsd", "wsd_overlap", ovl, "same")
record("wsd", "wsd_spectral_alignment", sal, "same")
record("wsd", "wsd_max_principal_angle", mpa, "same")
record("wsd", "wsd_mean_principal_angle", mpa_mean, "same")
record("wsd", "wsd_k", kk, "same вЂ” constructor echo")
record("wsd", "wsd_steady_state_error", sse, "same")
record("wsd", "wsd_nu_max_estimate", nu, "same")
record("wsd", "wsd_convergence_time", conv, "same вЂ” 1/sync_interval")
record("wsd", "wsd_lambda_estimate", lam, "same вЂ” 1/sync_interval")

# THE TAUTOLOGY: steady_state_error == running_drift, bit for bit.
diffs = []
for interval in [1, 3, 7, 100, 1000]:
    w = WorkspaceSyncDrift(embed_dim=D, k=K, sync_interval=interval).to(torch.float64)
    _l, _i = w.compute_drift(QF[:, :K].contiguous(), HT, step=0)
    for step in range(1, 13):
        _l, info = w.compute_drift(qrot(0.3 + 0.1 * step), None, step=step)
        diffs.append(abs(info["wsd_steady_state_error"] - info["wsd_running_drift"]))
    print(
        f"  sync_interval={interval:5d}: sse={info['wsd_steady_state_error']:.10g} "
        f"running_drift={info['wsd_running_drift']:.10g} "
        f"|diff|={abs(info['wsd_steady_state_error'] - info['wsd_running_drift']):.3e}"
    )
record("wsd", "|wsd_steady_state_error - wsd_running_drift|", diffs,
       "60 (interval, step) pairs, D=64 k=6")

# lambda_estimate vs the real lambda_wsd: sweep the config they can disagree on.
lam = []
for interval in [1, 2, 5, 10, 100, 500]:
    w = WorkspaceSyncDrift(embed_dim=D, k=K, sync_interval=interval).to(torch.float64)
    _l, _i = w.compute_drift(QF[:, :K].contiguous(), HT, step=0)
    for step in range(1, 5):
        _l, info = w.compute_drift(qrot(0.4), None, step=step)
    lam.append(info["wsd_lambda_estimate"])
    print(
        f"  sync_interval={interval:4d} -> wsd_lambda_estimate={info['wsd_lambda_estimate']:.6f}"
        f"   (= 1/{interval} = {1 / interval:.6f});  true lambda_wsd default = 0.01"
    )
record("wsd", "wsd_lambda_estimate", lam, "sync_interval 1..500; true lambda_wsd is config, fixed at 0.01")

# D-WSD-3: a non-orthonormal Q reads as perfectly synchronised.
clamp = []
for scale in [1.0, 1.5, 2.0, 3.0, 5.0]:
    w = WorkspaceSyncDrift(embed_dim=D, k=K, sync_interval=5).to(torch.float64)
    _l, _i = w.compute_drift(QF[:, :K].contiguous(), HT, step=0)
    Qb = qrot(0.9) * scale
    for step in range(1, 5):
        _l, info = w.compute_drift(Qb, None, step=step)
    clamp.append(info["wsd_drift"])
    print(
        f"  Q scaled x{scale}: wsd_drift={info['wsd_drift']:.6f}"
        f"  wsd_spectral_alignment={info['wsd_spectral_alignment']:.6f}"
        f"  wsd_overlap={info['wsd_overlap']:.4f}"
    )
record("wsd", "wsd_drift (non-orthonormal Q)", clamp,
       "genuinely misaligned subspace, ||Q|| scaled 1..5")

# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
print("\n=== STA ===")
STA_L, STA_W1, STA_WF, STA_WARM, STA_GAP = [], [], [], [], []
STA_RW1, STA_RGAP, STA_MAX, STA_MIN, STA_COND = [], [], [], [], []
sta = SpectralTransportAlignment(embed_dim=D, eta=0.01, ema_beta=0.999).to(torch.float64)
prev = r32(N, D, seed=100)
for step in range(1, 9):
    cur = r32(N, D, seed=200 + step) * (1 + 0.5 * step)
    loss, info = sta(cur, step=step)
    STA_L.append(info["sta_loss"])
    STA_W1.append(info["sta_w1"])
    STA_WF.append(info["sta_warmup_factor"])
    STA_WARM.append(info["sta_warmup"])
    STA_GAP.append(info["sta_spectral_gap"])
    STA_RW1.append(info["sta_running_w1"])
    STA_RGAP.append(info["sta_running_spectral_gap"])
    STA_MAX.append(info["sta_max_eigenvalue"])
    STA_MIN.append(info["sta_min_eigenvalue"])
    STA_COND.append(info["sta_condition_number"])
record("sta", "sta_loss", STA_L, "spectrum rescaled 1x..4.5x over 8 steps")
record("sta", "sta_w1", STA_W1, "same")
record("sta", "sta_warmup_factor", STA_WF, "same")
record("sta", "sta_warmup", STA_WARM, "same вЂ” boolean flag")
record("sta", "sta_spectral_gap", STA_GAP, "same")
record("sta", "sta_running_w1", STA_RW1, "same")
record("sta", "sta_running_spectral_gap", STA_RGAP, "same")
record("sta", "sta_max_eigenvalue", STA_MAX, "same")
record("sta", "sta_min_eigenvalue", STA_MIN, "same")
record("sta", "sta_condition_number", STA_COND, "same")
print(f"  sta_loss.requires_grad = {loss.requires_grad}")

# STA spectral gap: construct a spectrum WITH a clear gap vs one without.
gapv = []
for shape in ["flat", "clear_gap_at_6", "decaying", "one_huge"]:
    if shape == "flat":
        z = r32(N, D, seed=301)
    elif shape == "clear_gap_at_6":
        z = 0.05 * (r32(N, K, seed=302) @ QF[:, :K].T.double()) + 1.0 * (r32(N, D - K, seed=303) @ QF[:, K:].T.double())
    elif shape == "decaying":
        z = r32(N, D, seed=304) * torch.arange(D, 0, -1, dtype=torch.float64).sqrt()
    else:
        z = 1e4 * (r32(N, 1, seed=305) @ QF[:, 0:1].T.double()) + r32(N, D, seed=306)
    s2 = SpectralTransportAlignment(embed_dim=D, eta=0.01, ema_beta=0.999).to(torch.float64)
    _l, i2 = s2(z, step=1)
    _l, i2 = s2(z, step=2)
    gapv.append(i2["sta_spectral_gap"])
    print(
        f"  {shape:16s}: sta_spectral_gap={i2['sta_spectral_gap']:.6f}"
        f"  w1={i2['sta_w1']:.3e}  max_eig={i2['sta_max_eigenvalue']:.4g}"
    )
record("sta", "sta_spectral_gap (constructed spectra)", gapv,
       "flat / gap-at-6 / decaying / one-huge")

# Davis-Kahan bound: does it respond to the gap it names?
dk = []
dk_gap = []
for kk2, gapval in [(2, 0.01), (6, 0.01), (6, 0.5), (6, 2.0), (6, 10.0)]:
    z = 0.05 * (r32(N, K, seed=310) @ QF[:, :K].T.double()) + 1.0 * (r32(N, D - K, seed=311) @ QF[:, K:].T.double())
    s4 = SpectralTransportAlignment(embed_dim=D, eta=0.01, ema_beta=0.999).to(torch.float64)
    _l, i4 = s4(z, step=1)
    b = s4.compute_davis_kahan_bound(k=kk2)
    dk.append(b)
    dk_gap.append(i4["sta_spectral_gap"])
    print(f"  k={kk2}: davis_kahan_bound={b:.6g}  reported sta_spectral_gap={i4['sta_spectral_gap']:.6f}")
record("sta", "compute_davis_kahan_bound", dk, "k and a constructed gap at the k/k+1 boundary")
record("sta", "sta_spectral_gap at that same call", dk_gap, "the delta the bound divides by")
dk2 = []
for amp in [0.01, 0.1, 0.5, 1.0, 5.0, 50.0]:
    z = amp * (r32(N, K, seed=320) @ QF[:, :K].T) + 1.0 * (r32(N, D - K, seed=321) @ QF[:, K:].T)
    s6 = SpectralTransportAlignment(embed_dim=D, eta=0.01, ema_beta=0.999).to(torch.float64)
    for stp in [1, 2, 3]:
        _l, i6 = s6(z, step=stp)
    b = s6.compute_davis_kahan_bound(k=K)
    dk2.append((b, i6["sta_running_w1"], i6["sta_spectral_gap"]))
    print(
        f"  signal amp={amp:6g}: davis_kahan_bound={b:.6g}  running_w1={dk2[-1][1]:.6g}"
        f"  reported gap={dk2[-1][2]:.6g}  bound/w1={b / max(dk2[-1][1], 1e-30):.4g}"
    )
record("sta", "compute_davis_kahan_bound (spectrum separation)", [x[0] for x in dk2],
       "signal amplitude 0.01..50 in a k=6 signal block")
record("sta", "sta_running_w1 at the same call", [x[1] for x in dk2], "same")
record("sta", "sta_spectral_gap at the same call", [x[2] for x in dk2], "same")
s5 = None
dsb = []
for amp in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
    s5 = SpectralTransportAlignment(embed_dim=D, eta=0.01, ema_beta=0.999).to(torch.float64)
    z = amp * (r32(N, K, seed=350) @ QF[:, :K].T) + r32(N, D, seed=351)
    for stp in range(1, 12):
        _l, _i = s5(z * (1 + 0.1 * stp), step=stp)
    v = s5.compute_downstream_stability_bound(probe_norm=1.0, total_variance=1.0, k=K)
    dsb.append(v)
    print(f"  signal amp={amp:6g}: downstream_stability_bound={v:.6g}  running_w1={s5.running_w1.item():.6g}")
record("sta", "compute_downstream_stability_bound", dsb, "driven by module spectrum state, 0..100")
ws_warm = []
for stp in [0, 1, 500, 999, 1000, 1001, 2000]:
    s7 = SpectralTransportAlignment(embed_dim=D, eta=0.01, ema_beta=0.999).to(float64)
    _l, i7 = s7(r32(N, D, seed=330 + stp), step=stp)
    ws_warm.append((i7["sta_warmup_factor"], i7["sta_warmup"]))
record("sta", "sta_warmup_factor / sta_warmup vs step", ws_warm,
       "step 0..2000, warmup_steps default 1000")

# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
print("\n=== RDC ===")
RK, Qr = ortho_Q(D, D, seed=2), None
Qw = ortho_Q(D, K, seed=3)


def rdc_case(frac_in, step, eta=0.01):
    r = RepresentationDriftCompensation(
        embed_dim=D, eta=eta, ema_beta=0.999, k_workspace=K
    ).to(torch.float64)
    r.warmup_steps = 500
    base = torch.randn(B, T, D, generator=torch.Generator().manual_seed(400), dtype=torch.float64)
    drift_t = torch.randn(B, T, D, generator=torch.Generator().manual_seed(401), dtype=torch.float64)
    drift_t = (1 - frac_in) * (drift_t - (drift_t @ Qw) @ Qw.T) + frac_in * (drift_t @ Qw) @ Qw.T
    for s in range(1, max(step, 1) + 1):
        _l, info = r(
            base, z_previous=base + drift_t * (s + 1) / 10.0, workspace_Q=Qw, step=s
        )
    return info


# The warmup early-return emits a DIFFERENT, shorter key set.
_r0 = RepresentationDriftCompensation(embed_dim=D, eta=0.01, ema_beta=0.999, k_workspace=K)
_r0 = _r0.to(torch.float64)
_l, i0 = _r0(
    torch.zeros(B, T, D, dtype=torch.float64),
    z_previous=torch.zeros(B, T, D, dtype=torch.float64),
    workspace_Q=Qw,
    step=0,
)
print(f"  rdc info keys at step=0 (warmup early-return): {sorted(i0)}")
_l, i1 = _r0(
    torch.zeros(B, T, D, dtype=torch.float64),
    z_previous=torch.zeros(B, T, D, dtype=torch.float64),
    workspace_Q=Qw,
    step=10,
)
print(f"  rdc info keys at step=10: {len(i1)} keys")
print(f"  keys present at step=10 but ABSENT at step=0: {sorted(set(i1) - set(i0))}")


for step in [0, 50, 500, 1000, 5000, 10000, 50000]:
    info = rdc_case(0.0, max(step, 1))
    print(
        f"  step={step:6d}: rdc_transient_bound={info['rdc_transient_bound']:.6e}"
        f"  rdc_stationary_bound={info['rdc_stationary_bound']:.6e}"
        f"  rdc_theoretical_bound={info['rdc_theoretical_bound']:.6e}"
        f"  ortho_drift={info['rdc_ortho_drift_norm']:.6f}"
        f"  loss={info['rdc_loss']:.6e}"
    )

tb = [rdc_case(0.0, s)["rdc_transient_bound"] for s in [10, 100, 500, 1000, 2000, 5000, 10000, 50000]]
sb = [rdc_case(0.0, s)["rdc_stationary_bound"] for s in [10, 100, 500, 1000, 2000, 5000, 10000, 50000]]
th = [rdc_case(0.0, s)["rdc_theoretical_bound"] for s in [10, 100, 500, 1000, 2000, 5000, 10000, 50000]]
record("rdc", "rdc_transient_bound", tb, "step 10..50000, eta=0.01, k=6")
record("rdc", "rdc_stationary_bound", sb, "same")
record("rdc", "rdc_theoretical_bound", th, "same")

sw = []
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    sw.append(
        [
            rdc_case(frac, 2000)[kk]
            for kk in [
                "rdc_loss",
                "rdc_ortho_drift_norm",
                "rdc_workspace_drift_norm",
                "rdc_total_drift_norm",
                "rdc_drift_ratio",
            ]
        ]
    )
for i, name in enumerate(
    [
        "rdc_loss",
        "rdc_ortho_drift_norm",
        "rdc_workspace_drift_norm",
        "rdc_total_drift_norm",
        "rdc_drift_ratio",
    ]
):
    record("rdc", name, [row[i] for row in sw], "drift placed 0..100% INSIDE span(Q)")
row = rdc_case(0.0, 2000)
record("rdc", "rdc_warmup_factor", [rdc_case(0.0, s)["rdc_warmup_factor"] for s in [0, 100, 499, 500, 600]],
       "step 0..600 with warmup_steps=500")
record("rdc", "rdc_k_workspace", [row["rdc_k_workspace"]] * 5, "constructor echo")
record("rdc", "rdc_eta", [row["rdc_eta"]] * 5, "constructor echo")

# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
print("\n=== WSR ===")


def wsr_case(gs, step=2000, mode="gradient", seed=0):
    w = WorkspaceSharpnessRegularization(
        embed_dim=D, rho=0.05, eta=0.01, ema_beta=0.999, mode=mode
    ).to(torch.float64)
    w.warmup_steps = 500
    Qb = ortho_Q(D, K, seed=4)
    w.set_lagged_gradient(gs * r32(D, K, seed=seed + 50))
    return w, Qb


SC, SS, SPD, SDIR, SG, SR = [], [], [], [], [], []
for gs in [0.0, 1e-8, 1e-4, 1e-2, 1.0, 100.0]:
    w, Qb = wsr_case(gs, seed=1)
    for s in range(3):
        loss, info = w(Qb, step=s)
    SC.append(info["wsr_loss"])
    SS.append(info["wsr_sharpness"])
    SPD.append(info["wsr_spectral_sharpness"])
    SDIR.append(info["wsr_directional_sharpness"])
    SG.append(info["wsr_grad_norm"])
    print(
        f"  |grad|={gs:8g}: wsr_sharpness={info['wsr_sharpness']:.8g}"
        f"  wsr_spectral_sharpness={info['wsr_spectral_sharpness']:.8g}"
        f"  wsr_directional={info['wsr_directional_sharpness']:.8g}"
        f"  loss={info['wsr_loss']:.8g}  requires_grad={loss.requires_grad}"
    )
record("wsr", "wsr_loss", SC, "lagged gradient magnitude 0..100")
record("wsr", "wsr_sharpness", SS, "same")
record("wsr", "wsr_spectral_sharpness", SPD, "same")
record("wsr", "wsr_directional_sharpness", SDIR, "same")
record("wsr", "wsr_grad_norm", SG, "same")
dup = [abs(a - b) for a, b in zip(SS, SPD)]
record("wsr", "|wsr_sharpness - wsr_spectral_sharpness|", dup, "same")

w, Qb = wsr_case(1e-2, seed=1)
_l, info = w(Qb, step=2000)
record("wsr", "wsr_rho", [info["wsr_rho"]] * 5, "constructor echo")
record("wsr", "wsr_warmup_factor", [
    wsr_case(1e-2, seed=1)[0](Qb, step=s)[1]["wsr_warmup_factor"] for s in [0, 100, 499, 500, 900]
], "step 0..900, warmup_steps=500")
record("wsr", "wsr_warmup", [info["wsr_warmup"]] * 5, "boolean flag")
record("wsr", "wsr_grad_source", ["lagged"] * 5, "provenance string, set by the caller")
record("wsr", "wsr_gradient_substituted", [info["wsr_gradient_substituted"]] * 5,
       "boolean, False whenever a lagged gradient was supplied")

# No gradient supplied at all -> the documented sharpness is not computed.
w2 = WorkspaceSharpnessRegularization(
    embed_dim=D, rho=0.05, eta=0.01, ema_beta=0.999, mode="gradient"
).to(torch.float64)
w2.warmup_steps = 500
Qz = ortho_Q(D, K, seed=4)
_l, i5 = w2(Qz, step=2000)
print(f"  no gradient supplied: source={i5['wsr_grad_source']} substituted={i5['wsr_gradient_substituted']}")
record("wsr", "wsr_sharpness (no gradient available)", [i5["wsr_sharpness"], i5["wsr_loss"]],
       "exactly-orthonormal Q with no dL/dQ available")

# sam mode
def sam_case(gs, seed=0):
    w = WorkspaceSharpnessRegularization(
        embed_dim=D, rho=0.05, eta=0.01, ema_beta=0.999, mode="sam"
    ).to(torch.float64)
    w.warmup_steps = 500
    Qb = ortho_Q(D, K, seed=4)

    def loss_fn(Q, zp, zt):
        # A genuine, Q-dependent objective so the perturbation has an effect.
        return ((zp - zt) @ Q).pow(2).mean()

    w.set_lagged_gradient(gs * r32(D, K, seed=seed + 60))
    return w, Qb, loss_fn


zp_s = r32(N, D, seed=500)
zt_s = r32(N, D, seed=501)
SAML, SAMLOSS = [], []
for gs in [0.0, 1e-6, 1e-3, 1.0]:
    w, Qb, lf = sam_case(gs, seed=2)
    loss, info = w(Qb, step=2000, loss_fn=lf, z_pred=zp_s, z_target=zt_s)
    SAML.append(info["wsr_sharpness"])
    SAMLOSS.append((info["wsr_loss_current"], info["wsr_loss_perturbed"]))
    print(
        f"  sam |grad|={gs:8g}: sharpness={info['wsr_sharpness']:.8g}"
        f"  L(Q)={info['wsr_loss_current']:.8g} L(Q+dt)={info['wsr_loss_perturbed']:.8g}"
        f"  spectral={info['wsr_spectral_sharpness']:.8g}"
    )
record("wsr", "wsr_sharpness (sam mode)", SAML, "lagged gradient magnitude 0..1 on a real loss_fn")
record("wsr", "wsr_loss_current (sam mode)", [x[0] for x in SAMLOSS], "same")
record("wsr", "wsr_loss_perturbed (sam mode)", [x[1] for x in SAMLOSS], "same")

save(r".agent-notes\res_stab.json")
