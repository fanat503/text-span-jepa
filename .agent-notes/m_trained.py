"""Trained-parameter addendum — metrics inventory driver 5 of 5.

PCR's heads and CGN's gate logits are zero-initialised, so every diagnostic
they feed reads a constant at step 0 no matter what the data does. That is not
evidence of degeneracy — it is evidence of an untrained parameter. This driver
takes a few Adam steps on THOSE PARAMETERS ONLY, against a synthetic objective,
to establish whether the diagnostic can move once the parameter it reads has
moved. It is not a training run of the model and touches nothing outside
`src/models/<mech>.py`'s own module object.
"""

from __future__ import annotations

from mh import guard, ortho_Q, record, save  # noqa: E402

guard()

import torch  # noqa: E402

from src.models.cgn import ContextualGatingNetwork  # noqa: E402
from src.models.pcr import PredictiveCascadeRefinement  # noqa: E402
from src.models.spc import SpectralPredictiveCoding  # noqa: E402

B, T, D, K = 4, 32, 64, 6
QF = ortho_Q(D, D, seed=1).float()


def rnd(shape, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    if isinstance(shape, tuple):
        return torch.randn(*shape, generator=g)
    return torch.randn(shape, shape, generator=g)


# ══════════════════════════════════════════════════════════════════════════
print("\n=== PCR: pcr_ortho_score vs GENUINE off-diagonal structure ===")
orth = []
for eps in [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.2]:
    p = PredictiveCascadeRefinement(embed_dim=D, n_levels=3)
    p.warmup_steps = 10
    with torch.no_grad():
        p.workspace_Q.add_(eps * rnd((D, p.workspace_Q.shape[1]), seed=901))
    z1 = rnd((B, T, D), seed=902)
    z2 = z1 + 0.2 * rnd((B, T, D), seed=903)
    out = z1.clone()
    for stp in range(15):
        out, info = p(out, z2, step=stp)
    Q = p.workspace_Q
    gram = Q.T @ Q
    off = gram.clone()
    off.fill_diagonal_(0)
    m = float(off.abs().mean())
    orth.append(info["pcr_ortho_score"])
    print(f"  eps={eps:8g}: mean|offdiag(Q^TQ)|={m:.6e}  pcr_ortho_score={info['pcr_ortho_score']:.6f}")
record("pcr", "pcr_ortho_score vs genuine off-diagonal mass", orth, "eps 0..0.2 injected into Q")

print("\n=== PCR: do the refinement diagnostics move once the heads are trained? ===")
imp, ref, cn, gt, init_r, fin_r = [], [], [], [], [], []
for steps in [0, 10, 30, 100, 300]:
    torch.manual_seed(950)
    p = PredictiveCascadeRefinement(embed_dim=D, n_levels=3)
    p.warmup_steps = 10
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    z1 = rnd((B, T, D), seed=904)
    z2 = z1 + 0.3 * rnd((B, T, D), seed=905)
    for stp in range(max(steps, 1)):
        out, _i = p(z1, z2, step=stp)
        loss = ((out - z2) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        p.stiefel_retract()
    out, info = p(z1, z2, step=1000)
    imp.append(info["pcr_improvement"])
    ref.append(info["pcr_total_refinement_norm"])
    cn.append(info["pcr_level_0_correction_norm"])
    gt.append(info["pcr_level_0_gate"])
    init_r.append(info["pcr_initial_residual"])
    fin_r.append(info["pcr_final_residual"])
    print(
        f"  {steps:4d} Adam steps: improvement={info['pcr_improvement']:.6f}"
        f" refinement_norm={info['pcr_total_refinement_norm']:.6f}"
        f" level0_correction={info['pcr_level_0_correction_norm']:.6f}"
        f" level0_gate={info['pcr_level_0_gate']:.6f}"
        f" init_res={info['pcr_initial_residual']:.4f} final_res={info['pcr_final_residual']:.4f}"
    )
record("pcr", "pcr_improvement (after training the heads)", imp, "0/10/30/100/300 Adam steps")
record("pcr", "pcr_total_refinement_norm (trained)", ref, "same")
record("pcr", "pcr_level_0_correction_norm (trained)", cn, "same")
record("pcr", "pcr_level_0_gate (trained)", gt, "same")
record("pcr", "pcr_initial_residual", init_r, "same — data-driven, never trained")
record("pcr", "pcr_final_residual", fin_r, "same")

for lv in [1, 2]:
    for suffix in ["correction_norm", "gate", "subspace_fraction"]:
        vals = []
        for steps in [0, 10, 30, 100, 300]:
            torch.manual_seed(960)
            p = PredictiveCascadeRefinement(embed_dim=D, n_levels=3)
            p.warmup_steps = 10
            opt = torch.optim.Adam(p.parameters(), lr=1e-3)
            z1 = rnd((B, T, D), seed=906)
            z2 = z1 + 0.3 * rnd((B, T, D), seed=907)
            for stp in range(max(steps, 1)):
                out, _i = p(z1, z2, step=stp)
                loss = ((out - z2) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                p.stiefel_retract()
            out, info = p(z1, z2, step=1000)
            vals.append(info[f"pcr_level_{lv}_{suffix}"])
        record("pcr", f"pcr_level_{lv}_{suffix} (trained)", vals, "0/10/30/100/300 Adam steps")

# ══════════════════════════════════════════════════════════════════════════
print("\n=== SPC: do the weight diagnostics move once adapt_weights runs? ===")
we, st, ns = [], [], []
for steps in [0, 1, 5, 20, 100, 500]:
    torch.manual_seed(970)
    sp = SpectralPredictiveCoding(embed_dim=D, n_bands=8, init="dct")
    z1 = rnd((B, T, D), seed=908)
    z2 = z1 + 0.05 * rnd((B, T, D), seed=909)
    for stp in range(max(steps, 1)):
        loss, info = sp(z1, z2)
    we.append(info["spc_weight_entropy"])
    st.append(info["spc_spectral_tilt"])
    ns.append(info["spc_n_significant_bands"])
    print(
        f"  {steps:4d} steps: weight_entropy={info['spc_weight_entropy']:.6f}"
        f" (log 8 = {torch.tensor(8.0).log().item():.6f})"
        f" tilt={info['spc_spectral_tilt']:.6f} n_significant={info['spc_n_significant_bands']}"
        f" total_loss={info['spc_total_loss']:.6f} uniform_loss={info['spc_uniform_loss']:.6f}"
    )
record("spc", "spc_weight_entropy (vs log n_bands)", we, "0/1/5/20/100/500 steps")
record("spc", "spc_spectral_tilt", st, "same")
record("spc", "spc_n_significant_bands", ns, "same")

print("\n=== SPC: the same, WITH the adaptation train.py actually calls ===")
we2, st2, ns2 = [], [], []
for steps in [0, 1, 5, 20, 100, 500]:
    torch.manual_seed(970)
    sp = SpectralPredictiveCoding(embed_dim=D, n_bands=8, init="dct")
    z1 = rnd((B, T, D), seed=908)
    z2 = z1 + 0.05 * rnd((B, T, D), seed=909)
    for stp in range(max(steps, 1)):
        loss, info = sp(z1, z2)
        sp.adapt_weights_to_predictability()  # src/train.py:1742
    we2.append(info["spc_weight_entropy"])
    st2.append(info["spc_spectral_tilt"])
    ns2.append(info["spc_n_significant_bands"])
    print(
        f"  {steps:4d} adapt calls: weight_entropy={info['spc_weight_entropy']:.6f}"
        f" tilt={info['spc_spectral_tilt']:.6f} n_significant={info['spc_n_significant_bands']}"
    )
record("spc", "spc_weight_entropy (with adapt_weights)", we2, "0/1/5/20/100/500 adapt calls")
record("spc", "spc_spectral_tilt (with adapt_weights)", st2, "same")
record("spc", "spc_n_significant_bands (with adapt_weights)", ns2, "same")

# ══════════════════════════════════════════════════════════════════════════
print("\n=== CGN: orthogonality_score (decisiveness) vs the gate logit ===")
dec = []
for lg in [-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0]:
    c = ContextualGatingNetwork(embed_dim=D, n_groups=4)
    with torch.no_grad():
        c.gate_logits_visible.zero_()
        c.gate_logits_visible[:, 1] = lg  # only the ON logit
    v = c.compute_orthogonality_score()
    dec.append(v)
    print(f"  ON-logit={lg:6g}: decisiveness (named 'orthogonality_score')={v:.6f}")
record("cgn", "compute_orthogonality_score vs ON-logit", dec, "-20..20")

print("\n=== CGN: do the gate diagnostics move once the logits are trained? ===")
keys = ["cgn_gate_diff", "cgn_sparsity", "cgn_entropy", "cgn_mean_gate_visible",
        "cgn_mean_gate_masked", "cgn_routing_gap"]
cols = {k: [] for k in keys}
cols["routing_efficiency"] = []
cols["context_preservation"] = []
cols["prediction_focus"] = []
cols["orthogonality_score"] = []
for steps in [0, 10, 50, 200, 800]:
    torch.manual_seed(980)
    c = ContextualGatingNetwork(embed_dim=D, n_groups=4, tau_start=1.0, tau_end=0.1,
                                anneal_steps=1000)
    opt = torch.optim.Adam(c.parameters(), lr=1e-2)
    z = rnd((B, T, D), seed=910)
    m = torch.zeros(B, T, dtype=torch.bool)
    m[:, :16] = True
    for stp in range(max(steps, 1)):
        out, _i = c(z, m, step=stp)
        loss = out.pow(2).mean()  # anything; we only need the logits to move
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    out, info = c(z, m, step=1000)
    r = c.compute_routing_efficiency(z, m)
    for k in keys:
        cols[k].append(info[k])
    cols["routing_efficiency"].append(r["routing_efficiency"])
    cols["context_preservation"].append(r["context_preservation"])
    cols["prediction_focus"].append(r["prediction_focus"])
    cols["orthogonality_score"].append(c.compute_orthogonality_score())
    print(
        f"  {steps:4d} Adam steps: gate_diff={info['cgn_gate_diff']:.6f}"
        f" sparsity={info['cgn_sparsity']:.6f} entropy={info['cgn_entropy']:.6f}"
        f" routing_gap={info['cgn_routing_gap']:.6f}"
        f" eff={r['routing_efficiency']:.6f} ctx={r['context_preservation']:.6f}"
        f" focus={r['prediction_focus']:.6f} orth={c.compute_orthogonality_score():.6f}"
    )
for k, v in cols.items():
    record("cgn", k + " (after training the logits)", v, "0/10/50/200/800 Adam steps")

save(r".agent-notes\res_trained.json")