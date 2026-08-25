# SPC: Spectral Predictive Coding — Implemented Specification

> Status: RECONCILED with `src/models/spc.py` (audit R14, 2026-08-24).
> The previous version of this document described a SIMPLEX-constrained,
> Gumbel-Softmax-weighted, NORM-SQUARED band loss. That object is not what
> the code computes. This document states the implemented mechanism, the
> properties that provably hold, and which v1 claims are void.

## Implemented mechanism

Basis `F ∈ O(D)`: DCT-II initialization, maintained on O(D) by SVD
retraction after every optimizer step (`train.py` Stiefel block); the
`spc_ortho_error` diagnostic reports `max|FᵀF − I|` every forward.
(R11 found the retraction's exception-fallback called a non-existent
`torch.linalg.q3r`; fixed to `qr`.)

Band weights:

```
w = softmax(θ) · B            # θ = log_band_weights, learned by backprop
w = clamp(w, min=min_weight)  # default min_weight = 0.1
w = w · B / sum(w)            # renormalize: sum(w) == B exactly
```

Loss (`z_pred`, `z_target` flattened to (N, D)):

```
L_SPC = Σ_b w_b · mean_{n,d}( ‖z_pred^(b) − z_target^(b)‖² )
      = (1/(N·d)) · Σ_b w_b · ‖Δc_b‖²          # equal band width d = D/B
```

Adaptation: `adapt_weights_to_predictability()` moves `θ` toward
`w* ∝ running_residual_var · running_predictability` with learning rate
`weight_lr = 0.01`. Wired into the training loop every 100 steps as of
R14 — previously this method existed but was never called (same failure
class as CMC/GAC wiring).

## Properties that HOLD for this implementation

1. **Exact weight normalization.** After clamp+renormalize, `Σ w_b = B`
   and `w_b ≥ min_weight` for every band — no band can be starved.
2. **Operational orthonormality of F.** Enforced by per-step retraction;
   `ortho_err` is exported every forward so drift would be visible.
3. **Parseval consistency up to a constant.** With `F ∈ O(D)` and equal
   band widths, `L_SPC = (1/(N·d)) · Σ_b w_b ‖Δc_b‖²`, i.e. the weighted
   band decomposition differs from the plain MSE only by the weight vector
   and the constant `1/(N·d)` — uniform weights reproduce standard MSE/`B`.
4. **Deterministic adaptation path.** Weight updates are a fixed
   exponential interpolation toward `var × predictability` under a local
   seed-free rule (no hidden RNG).

## VOID claims from the v1 document

1. ~~`w ∈ Δ^{B-1}` simplex constraint~~ — weights sum to **B**, not 1, and
   carry an explicit floor; they live on a clamped-renormalized affine image
   of the simplex, not the simplex itself.
2. ~~Gumbel-Softmax weighting `exp((α+g)/τ)`~~ — no Gumbel noise exists in
   the implementation; weights are deterministic functions of learnable
   logits.
3. ~~Unscaled norm-squared statement `L = Σ w_b‖Δc_b‖²`~~ — the code uses
   per-band MEANS; equivalent up to the constant `1/(N·d)` only because all
   bands have equal width.
4. ~~"Guaranteed orthonormality" unconditionally~~ — orthonormality holds
   operationally (per-step retraction) and its fallback path was broken
   until R14 (`q3r` typo).
5. ~~Theorem (Information-Proportional Capacity Allocation) as a theorem~~ —
   the `w* ∝ σ²·R²` derivation is a heuristic narrative: it mixes two
   conflicting objectives (the module header's own corollary claims gradient
   descent moves weight in the OPPOSITE direction of the stated optimum),
   and the adaptation that realizes it was never invoked. Kept as motivated
   heuristic; the wired soft-update is its operational form.

## What would restore v1-style statements

1. True simplex parameterization (sparsemax/Gumbel-Softmax) and dropping
   `min_weight`, or restating bounds on the clamped-renormalized set.
2. Fixing the band-width convention (equal widths today) before asserting
   unscaled Parseval-style identities.
3. A proof for the soft-update rule actually wired into training
   (contraction toward `w*` under the EMA statistics).
