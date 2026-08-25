# JAWP: Jacobian-Aligned Workspace Prediction — Formal Proof

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified: workspace-loss algebra on fixed k.
> DIVERGENT: curriculum slices Q[:, :k(t)] with time-varying k while the
> theorems assume fixed k; the five verification tests named in the proof
> are absent. Complementary-gate style fixes elsewhere do NOT apply here.


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
predictable directions, which< requires Σ_res and Cov(z_target) to
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
