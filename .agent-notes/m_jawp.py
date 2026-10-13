"""JAWP metric sweep — metrics inventory driver 1 of 5.

Method: each metric is driven by inputs constructed so the quantity its NAME
refers to genuinely varies, and the input the metric actually READS is the one
being varied. A metric whose range is zero across such a sweep is DEGENERATE.
No grade below is decided by reading code; only by the sweep.
"""

from __future__ import annotations

import math

from mh import guard, ortho_Q, record, save  # noqa: E402

guard()

import torch  # noqa: E402

from src.models.jawp import JAWPModule  # noqa: E402

B, T, D, K = 4, 32, 64, 6
N = B * T
M = "jawp"


def fresh(k=K, k_start=None, alpha=0.1, seed=0, k_end=None, dt=torch.float64):
    m = JAWPModule(
        embed_dim=D, k_start=k_start or k, k_end=k_end or k, curriculum_steps=10, alpha=alpha
    )
    m.workspace_Q.data = m.workspace_Q.data.to(dt)
    torch.manual_seed(seed)
    return m


def install(m, Q):
    with torch.no_grad():
        m.workspace_Q[:, : Q.shape[1]].copy_(Q)
    return m


def loss_of(m, z, zt, step=0):
    m.zero_grad(set_to_none=True)
    loss, info = m.compute_loss(z, zt, step=step)
    return loss.item(), info


def rand(n, d, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def rand32(n, d, seed=None):
    """float32 variant — required by the public diagnostics, which call .float()
    on their inputs before multiplying by Q."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(n, d, generator=g, dtype=torch.float32)


def install32(m, Q):
    """Install a float32 Q — for the public diagnostics, which call .float() on
    their inputs before multiplying by Q."""
    m.workspace_Q.data = m.workspace_Q.data.float()
    with torch.no_grad():
        m.workspace_Q[:, : Q.shape[1]].copy_(Q.float())
    return m


print("\n=== A1. Q-dependent diagnostics, Q installed, target varied ===")
QFULL = ortho_Q(D, D, seed=3)
QIN = QFULL[:, :K]  # (D,K)
QOUT = QFULL[:, K:]  # (D,D-K)

rel, util, tws, cos, pca, ws_loss, focus = ([] for _ in range(7))
zt_s = rand(N, D, seed=41)
m = fresh()
install(m, QIN)
for w in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    r = math.sqrt(w) * (rand(N, K, seed=int(w * 100)) @ QIN.T) + math.sqrt(
        1 - w
    ) * (rand(N, D - K, seed=900 + int(w * 100)) @ QOUT.T)
    r = r / r.norm() * 5.0
    zp = zt_s + r
    _l, info = loss_of(m, zp, zt_s)
    rel.append(info["predictive_relevance"])
    util.append(info["workspace_utilization"])
    tws.append(info["target_ws_fraction"])
    cos.append(info["workspace_cosine"])
    pca.append(info["pca_alignment"])
    ws_loss.append(info["loss_workspace"])
    focus.append(info["loss_predictor_focus"])
print(f"  residual share w = {[0.0,0.1,0.25,0.5,0.75,0.9,1.0]}")
record(M, "predictive_relevance", rel, "w = share of residual energy inside span(Q)")
record(M, "workspace_utilization", util, "same")
record(M, "target_ws_fraction", tws, "same (zt fixed, so expect flat)")
record(M, "workspace_cosine", cos, "same")
record(M, "pca_alignment", pca, "same (zt fixed, so expect flat)")
record(M, "loss_workspace", ws_loss, "same")
record(M, "loss_predictor_focus", focus, "same")

print("\n=== A2. target-side diagnostics, z_target varied, Q installed ===")
tws, pca = [], []
for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    t = math.sqrt(frac) * (rand(N, K, seed=700 + int(frac * 100)) @ QIN.T) + math.sqrt(
        1 - frac
    ) * (rand(N, D - K, seed=800 + int(frac * 100)) @ QOUT.T)
    m2 = fresh()
    install(m2, QIN)
    zp = rand(N, D, seed=5)
    _l, info = loss_of(m2, zp, t)
    tws.append(info["target_ws_fraction"])
    pca.append(info["pca_alignment"])
record(M, "target_ws_fraction", tws, "share of z_target energy inside span(Q) 0..1")
record(M, "pca_alignment", pca, "same")

print("\n=== A3. pca_alignment vs how well Q spans z_target's top-k PCA ===")
pa = []
for seed in range(8):
    mm = fresh(seed=seed)
    _l, info = loss_of(mm, rand(N, D, seed=1000 + seed), rand(N, D, seed=2000 + seed))
    pa.append(info["pca_alignment"])
record(M, "pca_alignment", pa, "8 random inits of Q against random targets")
# Constructed: Q = top-k PCA of the target -> alignment should be 1.0
tt = rand(2000, D, seed=31)
c = tt - tt.mean(0)
_ev, evec = torch.linalg.eigh((c.T @ c) / (c.shape[0] - 1))
TopK = evec.flip(1)[:, :K]
BotK = evec[:, :K]
cons = []
for nm, Qc in [("spans target top-k PCA", TopK), ("spans target BOTTOM-k PCA", BotK)]:
    mm = fresh()
    install(mm, Qc)
    _l, info = loss_of(mm, rand(2000, D, seed=77), tt)
    cons.append(info["pca_alignment"])
    print(f"  pca_alignment, Q {nm}: {info['pca_alignment']:.6f}")
record(M, "pca_alignment (constructed)", cons, "Q = top-k PCA vs Q = bottom-k PCA of target")

print("\n=== A4. ortho_score vs genuine off-diagonal structure in Q ===")
orth = []
for eps in [0.0, 1e-12, 1e-6, 1e-3, 0.1, 0.5]:
    mm = fresh()
    Qb = QIN + eps * rand(D, K, seed=3)
    install(mm, Qb)
    _l, info = loss_of(mm, rand(N, D, seed=5), rand(N, D, seed=6))
    orth.append(info["ortho_score"])
record(M, "ortho_score", orth, "off-diagonal mass injected into Q, eps 0..0.5")
scale = []
for s in [1.0, 2.0, 3.0]:
    mm = fresh()
    with torch.no_grad():
        mm.workspace_Q.mul_(s)
    _l, info = loss_of(mm, rand(N, D, seed=5), rand(N, D, seed=6))
    scale.append(info["ortho_score"])
record(M, "ortho_score (uniform rescale)", scale, "||Q|| scaled 1x/2x/3x — off-diag stays 0")

print("\n=== A5. prediction-error scale ===")
lw, lf, rel2, cos2 = [], [], [], []
m3 = fresh()
install(m3, QIN)
zt0 = rand(N, D, seed=21)
for scale in [0.0, 1e-8, 1e-3, 1e-1, 1.0, 10.0, 100.0]:
    zp = zt0 + scale * rand(N, D, seed=31)
    _l, info = loss_of(m3, zp, zt0)
    lw.append(info["loss_workspace"])
    lf.append(info["loss_predictor_focus"])
    rel2.append(info["predictive_relevance"])
    cos2.append(info["workspace_cosine"])
record(M, "loss_workspace", lw, "residual scale 0..100")
record(M, "loss_predictor_focus", lf, "same")
record(M, "predictive_relevance", rel2, "same (random residual direction, fixed k/D)")
record(M, "workspace_cosine", cos2, "same")

print("\n=== A6. workspace_cosine driven to anti-correlation ===")
cosv = []
zt1 = rand(N, D, seed=11)
for s in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    zp = zt1 + s * rand(N, D, seed=12)
    m4 = fresh()
    install(m4, QIN)
    _l, info = loss_of(m4, zp, zt1)
    cosv.append(info["workspace_cosine"])
zp_anti = -zt1
m5 = fresh()
install(m5, QIN)
_l, info = loss_of(m5, zp_anti, zt1)
cosv.append(info["workspace_cosine"])
record(M, "workspace_cosine", cosv, "z_pred = z_target + s*noise, plus exactly anti-correlated")

print("\n=== A7. k curriculum ===")
mc = JAWPModule(embed_dim=D, k_start=1, k_end=24, curriculum_steps=100)
mc.workspace_Q.data = mc.workspace_Q.data.double()
ks = []
for step in [0, 10, 50, 99, 100, 500, 5000]:
    _l, info = loss_of(mc, rand(N, D, seed=1), rand(N, D, seed=2), step=step)
    ks.append(info["k"])
record(M, "k (jawp)", ks, "curriculum step 0..5000 with k_start=1 k_end=24")
mfix = fresh()
ks2 = []
for step in range(4):
    _l, info = loss_of(mfix, rand(N, D, seed=1), rand(N, D, seed=2), step=step)
    ks2.append(info["k"])
record(M, "k (jawp) fixed-width config", ks2, "k_start == k_end == 6, step 0..3")

print("\n=== B. workspace_information_preservation (WIP, numbered capability #2) ===")
tt2 = rand32(2000, D, seed=61)
c2 = tt2 - tt2.mean(0)
_ev2, evec2 = torch.linalg.eigh((c2.T @ c2) / (tt2.shape[0] - 1))
TopK2, BotK2 = evec2.flip(1)[:, :K], evec2[:, :K]
names, ws_, bg_, mn_, mx_ = [], [], [], [], []
for nm, Qc in [
    ("Q spans top-k PCA of target", TopK2),
    ("Q spans bottom-k PCA of target", BotK2),
    ("Q random orthonormal", ortho_Q(D, K, seed=9)),
]:
    mm = fresh()
    install32(mm, Qc)
    _s, inf = mm.workspace_information_preservation(rand32(2000, D, seed=62), tt2)
    names.append(nm)
    ws_.append(inf["wip_score"])
    bg_.append(inf["bg_fraction"])
    mn_.append(inf["min_preservation"])
    mx_.append(inf["max_preservation"])
    print(f"  {nm:34s} wip={inf['wip_score']:.6f} bg={inf['bg_fraction']:.6f}")
record(M, "wip_score", ws_, " | ".join(names))
record(M, "bg_fraction", bg_, " | ".join(names))
record(M, "min_preservation", mn_, " | ".join(names))
record(M, "max_preservation", mx_, " | ".join(names))
wrand = []
for seed in range(6):
    mm = fresh(dt=torch.float32)
    install32(mm, ortho_Q(D, K, seed=300 + seed))
    _s, inf = mm.workspace_information_preservation(rand32(500, D, seed=300 + seed), tt2)
    wrand.append(inf["wip_score"])
record(M, "wip_score (6 independent random Q)", wrand, "Q drawn from 6 different Haar subspaces")
# Does the WIP "information preservation" respond to a genuine exogenous feature
# being removed from span(Q)? `features` is documented as (..., D) SAMPLES, so
# pass (N, D) samples whose energy is `keep` inside span(Q).
wbare = []
for keep in [1.0, 0.9, 0.7, 0.5, 0.2, 0.0]:
    mm = fresh()
    install32(mm, TopK2)
    feats = keep * (rand32(500, K, seed=400) @ TopK2.T) + (1 - keep) * (
        rand32(500, K, seed=401) @ BotK2.T
    )
    _s, inf = mm.workspace_information_preservation(rand32(2000, D, seed=62), tt2, features=feats)
    wbare.append(inf["wip_score"])
print(f"  wip_score vs share of feature energy inside span(Q) 1.0..0.0: {wbare}")
record(M, "wip_score (explicit features)", wbare, "(N,D) feature samples, keep 1.0..0.0 inside span(Q)")

print("\n=== C. detect_workspace_dimension (numbered capability #3) ===")
kstar, kgap, kmp, mgr, mplb, sig2 = [], [], [], [], [], []
for ktrue in [1, 2, 3, 6, 12, 24]:
    basis = ortho_Q(D, D, seed=ktrue)[:, :ktrue]  # (D, ktrue)
    # residual must itself carry the rank-k structure: z_pred and z_target must
    # DISAGREE on the signal, or the signal cancels and the residual is isotropic.
    sig = rand(4000, ktrue, seed=2000 + ktrue) @ basis.T
    zt = sig + 0.1 * rand(4000, D, seed=3000 + ktrue)
    zp = 0.2 * sig + rand(4000, D, seed=4000 + ktrue)
    mm = fresh()
    ks_, gi = mm.detect_workspace_dimension(zp, zt)
    print(
        f"  k_true={ktrue:2d} -> k*={ks_:2d}  k_gap={gi['k_gap']:2d}  k_mp={gi['k_mp']:2d}"
        f"  max_gap_ratio={gi['max_gap_ratio']:.4f}  mp_lower={gi['mp_lower_bound']:.5f}"
    )
    kstar.append(gi["k_star"])
    kgap.append(gi["k_gap"])
    kmp.append(gi["k_mp"])
    mgr.append(gi["max_gap_ratio"])
    mplb.append(gi["mp_lower_bound"])
    sig2.append(gi["sigma2_est"])
record(M, "k_star", kstar, "constructed residual of true rank 1,2,3,6,12,24")
record(M, "k_gap", kgap, "same")
record(M, "k_mp", kmp, "same")
record(M, "max_gap_ratio", mgr, "same")
record(M, "mp_lower_bound", mplb, "same")
record(M, "sigma2_est", sig2, "same")

basis = ortho_Q(D, D, seed=6)[:, :6]
sig = rand(4000, 6, seed=2006) @ basis.T
zt = sig + 0.1 * rand(4000, D, seed=3006)
zp = 0.2 * sig + rand(4000, D, seed=4006)
mm = fresh()
mmr = [mm.detect_workspace_dimension(zp, zt, min_gap_ratio=v)[0] for v in [1.01, 1.1, 2.0, 10.0, 1000.0, 1e9]]
mmr.append(mm.detect_workspace_dimension(zp, zt)[0])
print(f"  k* at min_gap_ratio 1.01/1.1/2/10/1e3/1e9/default: {mmr}")
record(M, "k_star vs min_gap_ratio param", mmr, "parameter swept 1.01 .. 1e9")

# FAVOURABLE construction for the MP branch: a residual whose spectrum is exactly
# k low-variance directions and D-k high-variance ones, i.e. the MP picture the
# docstring describes. This is the best case for k_mp; if it does not move here,
# the branch is structurally dead rather than merely hard.
kfav, kfav_gap, kfav_mp = [], [], []
for ktrue in [1, 2, 3, 6, 12, 24]:
    for amp in [0.1, 0.5]:
        Qb = ortho_Q(D, D, seed=100 + ktrue)
        s = amp * rand(4000, ktrue, seed=6000 + ktrue) @ Qb[:, :ktrue].T
        n = rand(4000, D - ktrue, seed=7000 + ktrue) @ Qb[:, ktrue:].T
        resid = s + n
        mm = fresh()
        ks2, gi = mm.detect_workspace_dimension(resid, torch.zeros(4000, D, dtype=torch.float64))
        print(
            f"  MP-favourable k_true={ktrue:2d} amp={amp}: k*={ks2:2d} "
            f"(k_gap={gi['k_gap']:2d}, k_mp={gi['k_mp']:2d}, mp_lower={gi['mp_lower_bound']:.4f})"
        )
        kfav.append(ks2)
        kfav_gap.append(gi["k_gap"])
        kfav_mp.append(gi["k_mp"])
record(M, "k_star (MP-favourable spectrum)", kfav, "residual with exactly k low-variance directions")
record(M, "k_gap (MP-favourable)", kfav_gap, "same")
record(M, "k_mp (MP-favourable)", kfav_mp, "same")

# How small must the signal be before k_mp fires at all?
amps = []
for amp in [1.0, 0.7, 0.5, 0.4, 0.3, 0.29, 0.25, 0.1]:
    Qb = ortho_Q(D, D, seed=11)
    s = amp * rand(4000, 6, seed=8000) @ Qb[:, :6].T
    n = rand(4000, D - 6, seed=8001) @ Qb[:, 6:].T
    mm = fresh()
    ks3, gi = mm.detect_workspace_dimension(
        s + n, torch.zeros(4000, D, dtype=torch.float64)
    )
    amps.append(gi["k_mp"])
    print(f"  signal/noise amp={amp:4.2f}: k_mp={gi['k_mp']:2d} k_gap={gi['k_gap']:2d} k*={ks3}")
record(M, "k_mp vs signal/noise amplitude", amps, "true k*=6, amp 1.0..0.1")

print("\n=== D. compute_background_complexity ===")
ws_r, bg_r, bcr = [], [], []
for frac in [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]:
    base = rand32(400, D, seed=51)
    QIN32 = QIN.float()
    a = base - (base @ QIN32) @ QIN32.T
    b = rand32(400, K, seed=52)
    zpr = base + math.sqrt(frac) * (b @ QIN32.T) + math.sqrt(1 - frac) * a
    mm = fresh()
    install32(mm, QIN)
    _c, inf = mm.compute_background_complexity(zpr, base)
    ws_r.append(inf["ws_residual"])
    bg_r.append(inf["bg_residual"])
    bcr.append(inf["bg_complexity_ratio"])
    print(f"  frac={frac}: ws_res={inf['ws_residual']:.6f} bg_res={inf['bg_residual']:.6f} ratio={inf['bg_complexity_ratio']:.4f}")
record(M, "ws_residual", ws_r, "residual placed 0..100% inside span(Q)")
record(M, "bg_residual", bg_r, "same")
record(M, "bg_complexity_ratio", bcr, "same")

print("\n=== E. compute_predictive_rank / predictive_rank_loss (capability #5) ===")
eff, ut, msv, cn, rl = [], [], [], [], []
for r in [1, 2, 3, 5, 6]:
    Qe = ortho_Q(D, K, seed=17)
    g = torch.Generator().manual_seed(5000 + r)
    Mx = torch.randn(r, r, generator=g)
    X = torch.randn(r, 4000, generator=g)
    zp = (Qe.float()[:, :r] @ Mx @ X).T  # (4000, D), workspace rank exactly r
    mm = fresh()
    install32(mm, Qe)
    ri = mm.compute_predictive_rank(zp)
    eff.append(ri["effective_rank"])
    ut.append(ri["rank_utilization"])
    msv.append(ri["min_singular"])
    cn.append(ri["condition_number"])
    rl.append(mm.predictive_rank_loss(zp).item())
    print(
        f"  workspace rank {r}/{K}: eff_rank={ri['effective_rank']:.4f}"
        f" util={ri['rank_utilization']:.4f} cond={ri['condition_number']:.3e}"
    )
record(M, "effective_rank", eff, "workspace prediction of true rank 1,2,3,5,6 (k=6)")
record(M, "rank_utilization", ut, "same")
record(M, "min_singular", msv, "same")
record(M, "condition_number", cn, "same")
record(M, "predictive_rank_loss", rl, "same")
rls = []
for s in [1e-3, 1e-1, 1.0, 10.0, 100.0]:
    mm = fresh()
    install32(mm, ortho_Q(D, K, seed=17))
    rls.append(mm.predictive_rank_loss(s * rand32(4000, D, seed=77)).item())
record(M, "predictive_rank_loss (scale sweep)", rls, "z_pred scaled 1e-3..100")

print("\n=== F. grassmann_retract gauge_norm (capability #4) ===")
gn = []
for gs in [0.0, 1e-12, 1e-6, 1e-3, 1.0, 100.0]:
    mm = fresh()
    install(mm, ortho_Q(D, K, seed=19))
    mm.workspace_Q.grad = gs * rand(D, K, seed=90)
    gn.append(mm.grassmann_retract())
record(M, "gauge_norm", gn, "gradient magnitude 0..100 on workspace_Q")

print("\n=== G. principal_angles / subspace_distance ===")
pa, sd = [], []
QF = ortho_Q(D, D, seed=23)
for ang in [0.0, 1e-8, 0.05, 0.2, 0.5, 1.0, 1.5707963]:
    # Rotate ONE column of the workspace OUT of its own span. A basis rotation
    # Q1 @ R would leave the subspace unchanged and read 0 for every angle.
    Q1 = QF[:, :K].contiguous()
    v = QF[:, K]  # orthogonal to every column of Q1
    cols = [math.cos(ang) * Q1[:, 0] + math.sin(ang) * v] + [Q1[:, i] for i in range(1, K)]
    Q2 = torch.linalg.qr(torch.stack(cols, dim=1))[0][:, :K]
    mm = fresh()
    install(mm, Q1)
    angles, _c = mm.principal_angles(Q2)
    pa.append(max(angles))
    sd.append(mm.subspace_distance(Q2))
    print(f"  rotation {ang:.6f} rad -> max principal angle {max(angles):.6f}, d {mm.subspace_distance(Q2):.6f}")
record(M, "max principal angle", pa, "one column rotated out of span(Q1), 0..pi/2")
record(M, "subspace_distance", sd, "same")

save(r".agent-notes\res_jawp.json")