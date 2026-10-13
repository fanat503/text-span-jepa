"""PUC / GAC / CMC metric sweep — metrics inventory driver 3 of 5."""

from __future__ import annotations

from math import e as math_e  # noqa: F401
from math import log as math_log  # noqa: E402
from math import pi as math_pi  # noqa: E402

from mh import guard, ortho_Q, record, save  # noqa: E402

guard()

import torch  # noqa: E402

from src.models.cmc import CrossMaskConsistency  # noqa: E402
from src.models.gac import GradientAllocatedCapacity  # noqa: E402
from src.models.puc import PredictionUncertaintyCalibration  # noqa: E402


def math_pow(base, exponent):
    return base**exponent

B, T, D, K = 4, 32, 64, 6
N = B * T
QF = ortho_Q(D, D, seed=1)


def rnd(shape, seed=None, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(*shape, generator=g, dtype=dtype)


# ══════════════════════════════════════════════════════════════════════════
print("\n=== PUC ===")
# Sweep the prediction's spectrum from isotropic through sharply decaying.
puc_row = []
for amp, name in [(1.0, "isotropic"), (3.0, "mild decay"), (10.0, "strong"), (50.0, "v. strong")]:
    base = (amp * rnd((B, T, K), seed=600)) @ QF[:, :K].T
    noise = rnd((B, T, D - K), seed=601) @ QF[:, K:].T
    z = base + noise
    p = PredictionUncertaintyCalibration(
        embed_dim=D, eta=0.01, ema_beta=0.999, warmup_steps=500
    ).to(torch.float64)
    _l, info = p(z, step=2000)
    puc_row.append(info)
    print(
        f"  {name:11s} (amp={amp:5g}): loss={info['puc_loss']:.6f}"
        f" entropy={info['puc_entropy']:.4f} target={info['puc_target_entropy']:.4f}"
        f" deficit={info['puc_entropy_deficit']:.4f} overconf={info['puc_overconfidence']:.6f}"
        f" min_eig={info['puc_min_eigenvalue']:.6f} max_eig={info['puc_max_eigenvalue']:.4f}"
        f" log_det={info['puc_log_det']:.4f} carries_grad={info['puc_carries_grad']}"
    )
for key in [
    "puc_loss",
    "puc_entropy",
    "puc_target_entropy",
    "puc_entropy_deficit",
    "puc_overconfidence",
    "puc_warmup_factor",
    "puc_min_eigenvalue",
    "puc_max_eigenvalue",
    "puc_log_det",
    "puc_n_components",
    "puc_carries_grad",
]:
    record("puc", key, [r[key] for r in puc_row], "prediction spectrum: iso / mild / strong / v.strong")

# The entropy estimator's bias: sweep the true differential entropy and compare.
print("\n  PUC entropy estimator vs the TRUE entropy of the same spectrum:")
bias = []
for decay in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
    seed = 700 + int(decay * 10)
    scale = torch.tensor(
        [math_pow(0.5, (i * decay) / (D - 1)) for i in range(D)], dtype=torch.float64
    )
    z = rnd((B, T, D), seed=seed) * scale
    c = z.reshape(-1, D)
    c = c - c.mean(0)
    ev = torch.linalg.eigvalsh((c.T @ c) / (c.shape[0] - 1)).clamp(min=1e-30)
    true_h = float(0.5 * (torch.log(ev).sum() + D * math_log(2 * math_pi * math_e)))
    p = PredictionUncertaintyCalibration(
        embed_dim=D, eta=0.01, ema_beta=0.999, warmup_steps=500
    ).to(torch.float64)
    _l, info = p(z, step=2000)
    bias.append(
        (true_h, info["puc_entropy"], info["puc_loss"], info["puc_entropy_deficit"],
         info["puc_carries_grad"])
    )
    print(
        f"    decay={decay:4g}: true H={true_h:10.4f}  puc_entropy={info['puc_entropy']:10.4f}"
        f"  err={info['puc_entropy'] - true_h:9.4f}  loss={info['puc_loss']:.6f}"
        f"  deficit={info['puc_entropy_deficit']:.6f}"
    )
record("puc", "puc_entropy vs TRUE differential entropy (error)",
       [b[1] - b[0] for b in bias], "spectrum decay 0..8")
record("puc", "puc_entropy", [b[1] for b in bias], "same spectra")
record("puc", "puc_loss", [b[2] for b in bias], "same spectra")
record("puc", "puc_carries_grad (loss non-zero)", [1.0 if b[4] else 0.0 for b in bias], "same")

# Is puc_carries_grad a tell for the INERT PATH, or only for the input's graph?
print("\n  puc_carries_grad with an input that DOES require grad:")
cg2 = []
for diff, name in [(True, "differentiable ON"), (False, "inert EMA path")]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p4 = PredictionUncertaintyCalibration(
            embed_dim=D, eta=0.01, ema_beta=0.999, warmup_steps=500,
            use_differentiable_entropy=diff,
        ).to(torch.float64)
    scale = torch.tensor([0.5 ** (i * 2.0 / (D - 1)) for i in range(D)], dtype=torch.float64)
    zz = (rnd((B, T, D), seed=621) * scale).requires_grad_(True)
    for stp in range(1, 6):  # let the Oja EMA populate on the inert path
        loss, i5 = p4(zz, step=stp)
    cg2.append((name, i5["puc_loss"], i5["puc_carries_grad"], bool(loss.requires_grad)))
    print(
        f"    {name:18s}: loss={i5['puc_loss']:.6f}  info carries_grad="
        f"{i5['puc_carries_grad']}  loss.requires_grad={bool(loss.requires_grad)}"
    )
record("puc", "puc_carries_grad (grad-carrying input)",
       [1.0 if x[2] else 0.0 for x in cg2], "differentiable ON vs inert EMA path")

# warmup
pw = []
for stp in [0, 1, 250, 499, 500, 900, 2000]:
    p = PredictionUncertaintyCalibration(
        embed_dim=D, eta=0.01, ema_beta=0.999, warmup_steps=500
    ).to(torch.float64)
    _l, i2 = p(rnd((B, T, D), seed=800), step=stp)
    pw.append((i2.get("puc_warmup_factor", "KEY-ABSENT"), sorted(i2)))
    print(f"    step={stp:5d}: keys={sorted(i2)}")
record("puc", "puc_warmup_factor", [x[0] for x in pw], "step 0..2000")

# n_components config echo
p2 = PredictionUncertaintyCalibration(
    embed_dim=D, eta=0.01, ema_beta=0.999, warmup_steps=500
).to(torch.float64)
_l, i3 = p2(rnd((B, T, D), seed=801), step=2000)
record("puc", "puc_n_components vs D", [
    PredictionUncertaintyCalibration(
        embed_dim=dd, eta=0.01, ema_beta=0.999, warmup_steps=500
    ).to(torch.float64)(rnd((B, T, dd), seed=802), step=2000)[1]["puc_n_components"]
    for dd in [32, 64, 128, 256]
], "embed_dim 32..256 — does it track the 'of D' in its name?")

# puc_carries_grad when the gate is actually OPEN.
print("\n  puc_carries_grad with the gate open (loss non-zero):")
cg = []
for amp, name in [(10.0, "strong"), (50.0, "v.strong"), (0.001, "tiny")]:
    base = (amp * rnd((B, T, K), seed=610)) @ QF[:, :K].T
    noise = rnd((B, T, D - K), seed=611) @ QF[:, K:].T
    p3 = PredictionUncertaintyCalibration(
        embed_dim=D, eta=0.01, ema_beta=0.999, warmup_steps=500
    ).to(torch.float64)
    loss, i4 = p3(base + noise, step=2000)
    cg.append((i4["puc_carries_grad"], i4["puc_loss"]))
    print(f"    {name:9s}: loss={i4['puc_loss']:.6f}  carries_grad={i4['puc_carries_grad']}")
record("puc", "puc_carries_grad (gate open)", [1.0 if x[0] else 0.0 for x in cg],
       "three spectra; True only when the loss is non-zero AND the differentiable path is on")

# ══════════════════════════════════════════════════════════════════════════
print("\n=== GAC ===")
gac_row = []
for frac_starved, name in [(0.0, "none"), (0.25, "quarter"), (0.5, "half"), (0.9, "most"), (1.0, "all")]:
    gn = torch.ones(D, dtype=torch.float64)
    gn[: int(frac_starved * D)] = 1e-6  # below tau_grad = 1e-4
    g = GradientAllocatedCapacity(
        embed_dim=D, gamma=0.01, tau_grad=1e-4, warmup_steps=1000
    ).to(torch.float64)
    _l, info = g(rnd((B, T, D), seed=610), gn, step=2000)
    gac_row.append(info)
    print(
        f"  {name:7s}: loss={info['gac_loss']:.6e} n_starved={info['gac_n_starved']}"
        f" frac={info['gac_starved_fraction']:.4f} mean_gn={info['gac_mean_grad_norm']:.4g}"
        f" min_gn={info['gac_min_grad_norm']:.4g} max_gn={info['gac_max_grad_norm']:.4g}"
    )
for key in [
    "gac_loss",
    "gac_n_starved",
    "gac_starved_fraction",
    "gac_warmup_factor",
    "gac_mean_grad_norm",
    "gac_min_grad_norm",
    "gac_max_grad_norm",
    "gac_running_starved_fraction",
    "gac_warmup",
]:
    record("gac", key, [r[key] for r in gac_row], "starved fraction 0 / .25 / .5 / .9 / 1.0")

gw = []
for stp in [0, 1, 500, 999, 1000, 2000]:
    g = GradientAllocatedCapacity(
        embed_dim=D, gamma=0.01, tau_grad=1e-4, warmup_steps=1000
    ).to(torch.float64)
    _l, i2 = g(rnd((B, T, D), seed=611), torch.ones(D, dtype=torch.float64), step=stp)
    gw.append((i2.get("gac_warmup_factor", "KEY-ABSENT"), i2.get("gac_warmup", "KEY-ABSENT")))
    print(f"    step={stp:5d}: keys={sorted(i2)}")
record("gac", "gac_warmup_factor / gac_warmup vs step", gw, "step 0..2000")

# 1/N scaling of the energy term: same starved set, growing batch.
nsweep = []
for bb in [1, 2, 4, 8, 16]:
    g = GradientAllocatedCapacity(
        embed_dim=D, gamma=0.01, tau_grad=1e-4, warmup_steps=1000
    ).to(torch.float64)
    gn = torch.full((D,), 1e-6, dtype=torch.float64)
    _l, info = g(rnd((bb, T, D), seed=620), gn, step=2000)
    nsweep.append((bb, info["gac_loss"]))
    print(f"    batch={bb:3d}: gac_loss={info['gac_loss']:.6e}  loss*B={info['gac_loss'] * bb:.6e}")
record("gac", "gac_loss vs batch size", [x[1] for x in nsweep], "batch 1..16, identical starved set")
record("gac", "gac_loss * batch_size", [x[1] * x[0] for x in nsweep],
       "if the theorem's per-dim energy were right, this would be flat")

# The stale-gradient question: does gac_n_starved track THIS step's loss?
stale = []
for s in range(6):
    g = GradientAllocatedCapacity(
        embed_dim=D, gamma=0.01, tau_grad=1e-4, warmup_steps=1000
    ).to(torch.float64)
    gn = torch.linspace(1e-8, 1e-1, D, dtype=torch.float64) if s % 2 == 0 else torch.full(
        (D,), 1e-1, dtype=torch.float64
    )
    _l, info = g(rnd((B, T, D), seed=630 + s), gn, step=2000)
    stale.append((s, info["gac_n_starved"]))
    print(f"    step {s}: gn pattern -> n_starved={info['gac_n_starved']}")
record("gac", "gac_n_starved across alternating gradient patterns", [x[1] for x in stale],
       "alternating all-starved / none-starved")

# ══════════════════════════════════════════════════════════════════════════
print("\n=== CMC ===")
Qb = ortho_Q(D, D, seed=2)[:, :K].to(torch.float64)


def cmc_case(rel_scale, overlap_frac, seed=700):
    """rel_scale: how far the two predictions disagree (0 = identical)."""
    c = CrossMaskConsistency(embed_dim=D, second_mask_ratio=0.5, min_overlap_ratio=0.2,
                             mode="always", interval=1).to(torch.float64)
    z1 = rnd((B, T, D), seed=seed)
    z2 = z1 + rel_scale * rnd((B, T, D), seed=seed + 1)
    m = torch.zeros(B, T, dtype=torch.bool)
    n_ov = int(overlap_frac * T)
    m[:, :n_ov] = True  # positions masked in BOTH masks -> the overlap set
    _l, info = c(z1, z2, m)
    return info


rows = []
for rel in [0.0, 0.1, 0.5, 1.0, 5.0]:
    info = cmc_case(rel, 0.5)
    rows.append(info)
    print(
        f"  rel={rel:4g}: loss={info['cmc_loss']:.6f} per_pos={info['cmc_per_position_loss']:.6f}"
        f" max_incons={info['cmc_max_inconsistency']:.6f} ov_count={info['cmc_overlap_count']}"
        f" ov_ratio={info['cmc_overlap_ratio']:.6f} skipped={info['cmc_skipped']}"
        f" running={info['cmc_running_consistency']:.6f}"
    )
for key in [
    "cmc_loss",
    "cmc_per_position_loss",
    "cmc_max_inconsistency",
    "cmc_overlap_count",
    "cmc_overlap_ratio",
    "cmc_running_consistency",
    "cmc_skipped",
]:
    record("cmc", key, [r[key] for r in rows], "prediction disagreement 0..5")
record("cmc", "|cmc_loss - cmc_per_position_loss|",
       [abs(r["cmc_loss"] - r["cmc_per_position_loss"]) for r in rows], "same")

ovr = []
for ov in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    info = cmc_case(1.0, ov)
    ovr.append(info)
    print(
        f"  overlap={ov:4g}: count={info['cmc_overlap_count']} ratio={info['cmc_overlap_ratio']:.6f}"
        f" loss={info['cmc_loss']:.6f}"
    )
record("cmc", "cmc_overlap_count vs true overlap", [r["cmc_overlap_count"] for r in ovr],
       "true both-masked positions 0..T")
record("cmc", "cmc_overlap_ratio vs true overlap", [r["cmc_overlap_ratio"] for r in ovr],
       "true both-masked positions 0..T")

skip = []
for nm, maskval in [("full overlap", 1.0), ("no overlap", 0.0)]:
    c = CrossMaskConsistency(embed_dim=D, min_overlap_ratio=0.2, mode="always",
                             interval=1).to(torch.float64)
    z1 = rnd((B, T, D), seed=800)
    z2 = z1 + 10.0 * rnd((B, T, D), seed=801)
    m = torch.zeros(B, T, dtype=torch.bool)
    m[:, : int(maskval * T)] = True
    _l, info = c(z1, z2, m)
    skip.append((nm, sorted(info), info["cmc_loss"]))
    print(f"  {nm:12s}: keys={sorted(info)}  cmc_loss={info['cmc_loss']}")
record("cmc", "cmc_loss on SKIP paths", [0.0], "predictions 10.0 apart, masks overlap nothing")

dsb = []
c = CrossMaskConsistency(embed_dim=D).to(torch.float64)
for eps in [0.0, 1e-8, 1e-4, 1e-2, 1.0, 100.0]:
    dsb.append(
        (
            c.compute_downstream_stability_bound(cmc_loss=eps, probe_norm=1.0),
            c.compute_representation_variance_bound(cmc_loss=eps, jepa_loss=1.0),
        )
    )
record("cmc", "compute_downstream_stability_bound", [x[0] for x in dsb], "cmc_loss 0..100")
record("cmc", "compute_representation_variance_bound", [x[1] for x in dsb], "cmc_loss 0..100")

# The gate itself: two INDEPENDENT masks at the shipped mask-ratio schedule.
print("\n  CMC overlap ratio under two independent masks at the shipped ratios:")
gate = []
for m1 in [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]:
    for m2 in [0.15, 0.35, 0.75, 1.0]:
        torch.manual_seed(900)
        ma = torch.rand(B, T) < m1
        mb = torch.rand(B, T) < m2
        om = ma & mb
        c2 = CrossMaskConsistency(
            embed_dim=D, min_overlap_ratio=0.2, mode="always", interval=1
        ).to(torch.float64)
        z1 = rnd((B, T, D), seed=901)
        z2 = z1 + 10.0 * rnd((B, T, D), seed=902)
        _l, i3 = c2(z1, z2, om)
        gate.append((m1, m2, i3["cmc_overlap_ratio"], i3.get("cmc_skipped"), i3["cmc_loss"]))
        print(
            f"    m1={m1:4g} m2={m2:4g}: cmc_overlap_ratio={i3['cmc_overlap_ratio']:.5f}"
            f"  skipped={i3.get('cmc_skipped')}  loss={i3['cmc_loss']:.4f}"
            f"  (min_overlap_ratio gate = 0.2)"
        )
record("cmc", "cmc_skipped under shipped mask ratios",
       [1.0 if g[3] else 0.0 for g in gate], "16 (m1, m2) pairs from defaults.yaml")
record("cmc", "cmc_overlap_ratio under shipped mask ratios", [g[2] for g in gate], "same")
skipped_rows = [
    (f"m1={g[0]:g} m2={g[1]:g}", sorted(k))
    for g, k in zip(gate, [{} for _ in gate])
]
print("\n  info-key sets differ between the skip and the run path:")
_seen = {}
for m1, m2, _r, sk, _l in gate:
    _seen[sk] = _seen.get(sk, 0) + 1
print(f"    skipped=True count={_seen.get(True, 0)}, skipped=False count={_seen.get(False, 0)}")
c3 = CrossMaskConsistency(
    embed_dim=D, min_overlap_ratio=0.2, mode="always", interval=1
).to(torch.float64)
m_none = torch.zeros(B, T, dtype=torch.bool)
_l, i_none = c3(rnd((B, T, D), seed=903), rnd((B, T, D), seed=904), m_none)
print(f"    keys on the SKIP path: {sorted(i_none)}")
_l, i_run = c3(rnd((B, T, D), seed=905), rnd((B, T, D), seed=906),
               torch.ones(B, T, dtype=torch.bool))
print(f"    keys on the RUN path:  {sorted(i_run)}")
print(f"    absent on skip: {sorted(set(i_run) - set(i_none))}")

# The repo's OWN span-mask generator, at the shipped mask-ratio curriculum.
print("\n  CMC gate under the repo's own generate_second_mask at the shipped ratios:")
span = []
for r1, r2 in [(0.15, 0.15), (0.15, 0.25), (0.15, 0.35), (0.25, 0.25), (0.25, 0.35),
               (0.35, 0.35), (0.35, 0.5), (0.5, 0.5), (0.6, 0.6)]:
    g2 = torch.Generator().manual_seed(950)
    ma = CrossMaskConsistency.generate_second_mask(T, B, r1, rng=g2)
    mb = CrossMaskConsistency.generate_second_mask(T, B, r2, rng=g2)
    om = CrossMaskConsistency.compute_overlap_mask(ma.float(), mb.float())
    c4 = CrossMaskConsistency(
        embed_dim=D, min_overlap_ratio=0.2, mode="always", interval=1
    ).to(torch.float64)
    z1 = rnd((B, T, D), seed=953)
    z2 = z1 + 10.0 * rnd((B, T, D), seed=954)
    _l, i4 = c4(z1, z2, om)
    span.append((r1, r2, i4["cmc_overlap_ratio"], i4.get("cmc_skipped"), i4["cmc_loss"]))
    print(
        f"    span ratios {r1:4g}/{r2:4g}: cmc_overlap_ratio={i4['cmc_overlap_ratio']:.5f}"
        f"  skipped={i4.get('cmc_skipped')}  loss={i4['cmc_loss']:.2f}"
    )
record("cmc", "cmc_overlap_ratio (repo span masks, shipped ratios)",
       [x[2] for x in span], "9 (r1, r2) pairs from defaults.yaml mask_ratio_start/end")
record("cmc", "cmc_skipped (repo span masks, shipped ratios)",
       [1.0 if x[3] else 0.0 for x in span], "same")

save(r".agent-notes\res_pgc.json")