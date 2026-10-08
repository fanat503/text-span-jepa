# JAWP: Jacobian-Aligned Workspace Prediction — Formal Proof

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified: workspace-loss algebra on fixed k.
> DIVERGENT: curriculum slices Q[:, :k(t)] with time-varying k while the
> theorems assume fixed k; the five verification tests named in the proof
> are absent. Complementary-gate style fixes elsewhere do NOT apply here.
>
> **ADDENDUM (TASK-40 + TASK-41, 2026-10-08) — a second mechanism pulls this same
> `Q` the other way.** Not in the 2026-08-24 matrix; first triaged under TASK-40.
> Nothing in this loss was changed.
>
> **The antagonism is REAL and MEASURED.** WSD maximizes
> `⟨Q_JAWPᵀ Q_target⟩_F` where `Q_target` is the **top-k** eigenvectors of the
> target covariance (`src/models/wsd.py:200-206,256-258`), so it pulls `Q`
> toward the target's **highest-variance** directions. This mechanism's
> minimizer on `St(D,k)` is the **bottom-k** eigenvectors of the *residual*
> covariance (`src/models/jawp.py:45-65`), and this file's own header names
> high-variance selection as the failure mode to avoid.
>
> WSD's pull is, in effect, the high-variance force this header removed as a
> design decision — reintroduced under another name.
>
> **It is not the "different matrices" case.** `src/models/jepa.py:1086` passes a
> **live view** of `workspace_Q` (no `.data()`), so both gradients land in the
> same `total_loss` in the same backward pass. Measured cosine between the two
> `Q`-gradients: **negative on 12 of 12 consecutive steps**, mean ≈ −0.47.
> WSD's raw gradient is ≈2.4× this mechanism's, so the crossover is at
> **lambda_wsd ≈ 0.42**; at lambda ≥ 1 `jawk_predictive_relevance` collapses to
> **0.0000**.
>
> **Reachability, corrected.** An earlier report claimed every runnable config
> ships `use_wsd: false`. Re-measured against the deep-merge over all 62
> configs: **51 resolve `use_wsd: true`, 50 with both flags true**, including
> `defaults.yaml` and every leave-one-out ablation row. The antagonism is live.
> It is not dominant only because the shipped `lambda_wsd: 0.01` sits ~42× below
> the crossover. Full evidence: `proofs/wsd.md` and `.agent-notes/task-40.md`.
>
> **Two numbered capabilities here have no caller anywhere in `src/`.**
> `detect_workspace_dimension` and `grassmann_retract` are counted in
> `GWP.N_MECHANISMS = 16` but nothing calls them. Measured, not guessed:
> `detect_workspace_dimension` recovers a constructed `k*` **exactly** at
> k* ∈ {1,2,3,6,12,24} with zero spread over 5 seeds at ~1.7 ms/call — it is not
> a dead formula and should be wired. `grassmann_retract` is span-equivalent to
> the current step (overlap² = 1.0000, trajectories equal to ~7e-6) but its
> gauge term is **not** the symmetric correction Theorem 3 is stated for, so
> swapping it in on that evidence would be wrong. Both left in place pending the
> owner's call; deleting a capability that is counted in the header is a
> different decision from deleting dead code.


## Problem Statement

Standard JEPA predicts ALL D dimensions of z_target equally:
```
L = ||z_pred - z_target||²
```
This wastes predictor capacity on:
1. **Noise directions**: unpredictable → always high loss, zero gradient signal
2. **Background directions**: predictable but not workspace → not useful
3. **Exogenous features**: Pendharkar et al. (2026, arXiv:2606.30068) showed
   JEPA discards control-relevant features

Anthropic (July 2026, arXiv:2607.15495): only ~10% of activation variance
is in J-space. This means ~90% of prediction capacity is wasted.

## Solution

JAWP predicts ONLY in a learned workspace subspace:
```
L_JAWP = ||Q^T z_pred - Q^T z_target||²     [workspace prediction]
       + α * ||(I - QQ^T) z_pred||²          [predictor focus]
```
where Q ∈ R^{D×k} is learned on the Stiefel manifold St(D,k).

## Theorem 1 (Courant-Fischer Optimality)

**Statement**: Define the residual covariance:
```
Σ_res = E[(z_pred - z_target)(z_pred - z_target)^T]
```
The workspace prediction loss equals:
```
E[||Q^T(z_pred - z_target)||²] = tr(Q^T Σ_res Q)
```
The minimizer of tr(Q^T Σ_res Q) subject to Q ∈ St(D,k) is the
**bottom-k eigenvectors** of Σ_res — the directions with LEAST
prediction residual, i.e., the most PREDICTABLE directions.

**Proof**:

By the Courant-Fischer min-max theorem (Golub & Van Loan, Matrix
Computations, Theorem 8.1.2):

For any Q with Q^T Q = I_k:
```
tr(Q^T Σ_res Q) = Σ_{i=1}^{k} q_i^T Σ_res q_i
```

Each term q_i^T Σ_res q_i is a Rayleigh quotient. By the
Courant-Fischer characterization:
```
λ_i(Σ_res) = min_{dim(S)=i} max_{x ∈ S, ||x||=1} x^T Σ_res x
```

where λ_1 ≤ λ_2 ≤ ... ≤ λ_D are the eigenvalues of Σ_res in
ascending order.

For the workspace prediction loss:
```
min_{Q ∈ St(D,k)} tr(Q^T Σ_res Q) = Σ_{i=1}^{k} λ_i(Σ_res)
```

achieved when Q = [v_1, v_2, ..., v_k] where v_i is the eigenvector
corresponding to λ_i (bottom-k eigenvectors). ∎

## Corollary (JAWP ≤ PCA)

**Statement**: R(Q_JAWP) ≤ R(Q_PCA) for ANY predictor, where R denotes
prediction risk and Q_PCA is the top-k PCA subspace of Cov(z_target).

**Proof**:

JAWP minimizes tr(Q^T Σ_res Q) over ALL of St(D,k).
PCA chooses Q as top-k eigenvectors of Cov(z_target).

Since Q_PCA ∈ St(D,k) (it's a valid orthonormal matrix), and JAWP
minimizes over ALL such matrices:
```
tr(Q_JAWP^T Σ_res Q_JAWP) ≤ tr(Q_PCA^T Σ_res Q_PCA)
```

Equality holds ONLY when PCA directions coincide with the most
predictable directions, which requires Σ_res and Cov(z_target) to
share eigenvectors with the SAME eigenvalue ordering. This requires
prediction error to be isotropic — the trivial case.

In general, high-variance directions can have high residual (noise),
while low-variance directions can have low residual (signal).
JAWP captures the latter; PCA captures the former. ∎

## Theorem 2 (WIP: Workspace Information Preservation)

**Statement**: Let f_exo be an exogenous control-relevant feature with
I(f_exo; z_target) > 0. Under the regularity condition that f_exo has
non-zero projection onto the bottom-k eigenspace of Σ_res, span(Q_JAWP)
must contain a non-trivial projection of f_exo.

**Proof (by contradiction)**:

Suppose span(Q) ⊥ f_exo (workspace orthogonal to exogenous feature).
Then Q^T f_exo = 0, so predicting Q^T z_target cannot use f_exo.

Under the regularity condition: f_exo has non-zero projection onto
at least one of the bottom-k eigenvectors of Σ_res. This means:
```
||P_{bottom-k} f_exo|| > 0
```
where P_{bottom-k} is the projection onto the bottom-k eigenspace.

Since Q_JAWP minimizes tr(Q^T Σ_res Q) and the bottom-k eigenvectors
are the unique minimizer (assuming distinct eigenvalues), span(Q_JAWP)
= span(v_1, ..., v_k) (the bottom-k eigenspace).

Therefore: P_{span(Q)} f_exo = P_{bottom-k} f_exo ≠ 0.

This contradicts the assumption that span(Q) ⊥ f_exo. ∎

**Note on regularity condition**: The condition holds generically
(measure 1 in the space of feature-covariance pairs). It fails only
when f_exo is purely in the high-residual eigenspace — meaning the
feature is unpredictable and SHOULD be excluded from workspace.

## Theorem 3 (Stiefel Retraction Correctness)

**Statement**: SVD-based retraction R(Q) = U[:, :k] @ V^T[:k, :]
after SVD(Q) = U S V^T projects Q to the nearest orthonormal matrix
in Frobenius norm.

**Proof**:

This is the standard retraction on St(D,k) from Absil, Mahony &
Sepulchre (2008), §4.1.

The nearest orthonormal matrix to Q in Frobenius norm is the solution to:
```
min_{X: X^T X = I_k} ||X - Q||_F
```

By the Eckart-Young theorem, the solution is X = U V^T where U, V
come from the SVD of Q. Since Q ∈ R^{D×k} with D ≥ k, the SVD gives
U ∈ R^{D×k}, S ∈ R^{k×k}, V ∈ R^{k×k}, and the retraction is
X = U V^T. ∎

## Verification

- `tests/test_jawp.py#test_q_orthonormality` — ||Q^T Q - I|| < 1e-5
- `tests/test_jawp.py#test_courant_fischer` — workspace risk ≤ PCA risk
- `tests/test_jawp.py#test_wip_preservation` — exogenous features preserved
- `tests/test_jawp.py#test_stiefel_retraction` — Q stays on St(D,k)
- `tests/test_jawp.py#test_predictive_rank` — rank preserved

## Unreached capabilities

Four numbered capabilities hang off `JAWPModule`. Two are called by the running
model; two have **no caller anywhere in `src/`** (only `tests/`). Both are
reachable only as manual calls, so nothing in a training run exercises them.
This is the count mismatch in `AGENTS.md` made concrete: they are counted, and
they are never called.

| capability | method | called from `src/` | what happens if wired |
|---|---|---|---|
| Workspace Prediction | `compute_loss` | `jepa.py:736` | — |
| Predictive Rank | `predictive_rank_loss` | `jepa.py:1041` | — |
| **Spectral Gap** | `detect_workspace_dimension` (`jawp.py:552`) | **none** | outputs `k_star`; nothing consumes it, so `k_end` stays fixed and the number is a log-only diagnostic |
| **Grassmann Optimization** | `grassmann_retract` (`jawp.py:893`) | **none** | `stiefel_retract` (`jawp.py:269`) is the one wired, from `jepa.py:532` and `train.py:1470` |

**`detect_workspace_dimension`** (Spectral Gap #3) detects a natural workspace
dimension from the residual spectrum, which is the one thing the
`jawk_k_end` hyperparameter cannot express. Measured against a constructed
ground truth (`D=64`, `N=2000`, known `k*`): it recovers `k*` **exactly** at
`k*` in `{1, 2, 3, 6, 12, 24}`, with zero spread across five seeds, at
~1.7 ms per call. The estimator works. What does not exist is the wiring that
would let a detected `k*` reach `current_k`.

**`grassmann_retract`** (Grassmann Optimization #4) is the post-step
projector that removes the O(k) gauge component. `stiefel_retract` is wired and
does the Riemannian symmetric correction instead. Measured over 60 steps
(`D=64`, `k=6`, `wsd_sync_interval=1`): the two produce trajectories equal to
~7e-6 in loss and an `overlap^2` of **1.0000** on `span(Q)`. On this
configuration they are interchangeable — but that is a measurement, not a
licence to swap them: `grassmann_retract`'s gauge term is an all-ones-weighted
component while the Stiefel retraction's Riemannian correction is symmetric, and
Theorem 3 above is stated for `stiefel_retract`. Swapping changes which
quantity is called "the retraction" without changing any audited claim, which
is a decision for the owner, not a cleanup.

**Recommendation.** Both should be wired, not deleted — neither is a dead
formula (unlike PUC's headline loss). The minimal wiring is one call each in
`src/models/jepa.py`, where `active_k` and `workspace_Q` are already in scope
at `_jawp_loss` / the retraction block; `src/models/jepa.py` is outside TASK-40's
file boundary, so this is recorded rather than applied. The
`AGENTS.md` / `README.md` count should either name them as caller-less or the
callers should land first — TASK-41 owns that reconciliation.

## Interaction with WSD: the same `Q`, the opposite subspace

JAWP minimises $\mathrm{tr}(Q^\top\Sigma_\mathrm{res}Q)$, whose minimiser over
$\mathrm{St}(D,k)$ is the **bottom**-$k$ of $\Sigma_\mathrm{res}$
(`jawp.py:375`, `jawp.py:43-57`). WSD minimises
$2k - 2\lVert Q^\top Q_\mathrm{tgt}\rVert_F^2$, whose minimiser is the **top**-$k$
of the target covariance (`wsd.py:206`, `wsd.py:258`). Both hold a live view
of `workspace_Q[:, :k_active]` in one `total_loss`, so both gradients land in
one backward pass.

Measured on the running model: the cosine between the two `Q`-gradients is
negative on **12 of 12** consecutive steps (mean $\approx -0.47$), and WSD's
raw gradient magnitude is ~2.4x JAWP's, putting the crossover at
$\lambda_\mathrm{WSD} \approx 0.42$. WSD's pull is precisely the high-variance
pull that DESIGN DECISION 3 removed from JAWP. Every runnable config sets
`use_wsd: false` and the one ablation that enables it uses `lambda_wsd: 0.1`,
below the crossover — so the conflict is real but sub-threshold in the grid.
It becomes visible as a collapse of `jawk_predictive_relevance` to 0 at
$\lambda_\mathrm{WSD} \ge 1$.

This is an open science decision, not a bug: the two mechanisms optimise
different quantities and either may be the intended science. Full evidence,
data-dependence of the optima, and five resolution options with consequences
are in [`wsd.md` — Interaction with JAWP](wsd.md#interaction-with-jawp-opposite-optima-one-parameter).
TASK-40 changed no loss and no weight in either module.

## How Other Papers Can Use JAWP

```python
from src.models.jawp import JAWPModule
jawp = JAWPModule(embed_dim=768, k_start=1, k_end=77)
loss, info = jawp.compute_loss(z_pred, z_target, step=step)
loss.backward()
optimizer.step()
jawp.stiefel_retract()  # maintain orthonormality
```

Works with: ANY JEPA variant, ANY modality (text, image, video, audio),
ANY predictor architecture. The only hyperparameter is k_end (workspace dim).
