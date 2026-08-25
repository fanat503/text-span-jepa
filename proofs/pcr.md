# PCR: Predictive Cascade Refinement — Implemented Specification

> Status: RECONCILED with `src/models/pcr.py` (audit R13, 2026-08-24).
> The previous version of this document proved an UNGATED LINEAR cascade with
> per-level mutual-information guarantees. That object is NOT implemented.
> This document describes what the code actually computes, what properties
> hold provably, and which v1 claims are void.

## Implemented mechanism

With `z_0 = Pred(z_ctx)` (base predictor output), `t = z_tgt` (detached
target, available ONLY at training time), level projections `P_l` given by
disjoint column blocks of a shared orthonormal matrix
`Q ∈ R^{D × Σd_l}` (`Q` maintained on O(D) by SVD retraction after each
optimizer step — see `train.py` Stiefel retractions):

```
r_{l-1} = t − z_{l-1}
c_l     = MLP_l( r_{l-1} @ P_l ) @ P_l^T        # MLP_l: d_l → 2·max(d) → d_l,
                                                # final layer zero-initialized
g_l     = sigmoid(α_l) · min(1, max(0, (step − W)/W))   # α_l learned scalar,
                                                        # W = warmup_steps = 1000
z_l     = z_{l-1} + g_l · c_l
```

Default `d_l` geometrically decreasing from D/4; `Σ d_l ≤ D` asserted;
`level_offsets` partition Q's columns so `P_l^T P_m = δ_{lm} I`.

## Properties that HOLD for this implementation

1. **Identity at init / during warmup.** Zero-initialized MLP heads give
   `c_l = 0`, and `g_l = 0` for `step < W`; hence
   `z_refined ≡ z_pred` exactly (verified: `tests/test_pcr.py`).
2. **Orthogonal correction subspaces.** Because `Q` is orthonormal (retracted)
   and blocks are disjoint, `span(c_l) ⊆ span(P_l)` with
   `P_l^T P_m = 0` for `l ≠ m`: level corrections never interfere
   linearly.
3. **Bounded gating.** `g_l ∈ [0, 1]`, so each level contributes a
   convex-style bounded correction; the module cannot amplify beyond its
   MLP output magnitude.
4. **Operational orthonormality monitoring.** `pcr_ortho_score`
   (`1 − mean|offdiag(QᵀQ)|`) is computed every forward.

## VOID claims from the v1 document

The following statements are retracted — they held for the v1 paper-object,
not for the implemented gated MLP cascade:

1. ~~Per-level strict residual contraction
   `‖r_l‖² = ‖r_{l-1}‖² − ‖P_l^T r_{l-1}‖²`.~~ The correction is
   `MLP(P_l^T r)`, not `P_l^T r`; no contraction identity follows, and the
   MLP may increase the residual along other directions.
2. ~~Mutual-information lower bound
   `I(z_ctx; z_L) ≥ I(z_ctx; z_0) + Σ_l I(r_{l-1}; P_l^T r_{l-1})`.~~ The
   DPI chain required the correction to be a deterministic projection OF THE
   RESIDUAL; a learned MLP conditioned on the residual breaks the Markov
   structure used in Step 3.
3. ~~"PCR can recover ALL lost information" corollary.~~ Depends on 1–2.
4. ~~Inference-time framing ("drop-in wrapper around any predictor").~~ The
   refinement consumes `z_target`: it is a TRAINING-TIME objective shaper
   for the predictor (gradient path through the shared trunk), not an
   inference component.

## What was gained by the orthogonal-block design (still true)

- Disjoint subspaces prevent two levels from fighting over the same
  coordinates (linear interference is structurally excluded).
- The shared orthonormal `Q` gives a single knob for capacity control
  (`Σ d_l ≤ D`) with a cheap operational check (`pcr_ortho_score`).

## What would be required to restore v1-style guarantees

1. Replace each `RefinementBlock` with the exact linear map `P_l^T (·)`
   (or constrain the MLP to be contractive along `span(P_l)`).
2. Remove the sigmoid gate or account for it explicitly in the bound.
3. Re-derive the information inequality for the gated composition — the
   current architecture admits no such statement without additional
   assumptions on the MLP class.
