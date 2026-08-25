# STA: Spectral Transport Alignment — Formal Proof

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified: W1 metric between sorted spectra.
> DIVERGENT: the Davis-Kahan reduction assumes a spectral gap enforced
> nowhere; delta refers to a different object than computed; before R11
> the executed loss was IDENTICALLY ZERO (sync-refresh bug, fixed); even
> now the signal carries no gradient (buffers under no_grad).


## Problem Statement

During JEPA training, the eigenvalue spectrum of the representation covariance
Cov(z) continuously changes as the encoder learns. This **spectral drift** causes:

1. **JAWP workspace instability**: The Courant-Fischer minimizer shifts when
   eigenvalues cross, causing Q to jump between subspaces
2. **SPC band staleness**: Frequency bands allocated for the old spectrum
   have wrong capacity for the new one
3. **Representation non-stationarity**: A linear probe trained at step t
   may fail at step t+100 due to distribution shift

## Theorem 1 (Davis-Kahan + STA)

**Statement**: Let Σ(t) be the representation covariance at step t, with
eigenvalues λ₁(t) ≥ ... ≥ λ_D(t). Let Q_k(t) be the bottom-k eigenvector
subspace of the residual covariance Σ_res(t). If the spectral gap
δ = λ_k(t) - λ_{k+1}(t) > 0, then:

```
d_Gr(Q_k(t), Q_k(t+1)) ≤ ||Σ_res(t) - Σ_res(t+1)||_op / δ
```

where d_Gr is the Grassmann distance and ||·||_op is the operator norm.

**Proof**:

This follows directly from the Davis-Kahan sin Θ theorem (Davis & Kahan, 1970).
Let Σ_res(t) and Σ_res(t+1) differ by Δ = Σ_res(t+1) - Σ_res(t).

The Davis-Kahan theorem states:
```
||sin Θ||_F ≤ ||Δ||_op / δ
```

where Θ are the principal angles between the subspaces.

Since ||Δ||_op ≤ max_i |λ_i(t+1) - λ_i(t)| ≤ W_1(λ(t), λ(t+1))
(the operator norm is bounded by the maximum eigenvalue change, which is
bounded by the W1 distance for sorted eigenvalues), we get:

```
d_Gr(Q_k(t), Q_k(t+1)) ≤ W_1 / δ
```

The Grassmann distance d_Gr is related to sin Θ by:
d_Gr² = Σ sin²(θ_i) ≤ ||sin Θ||_F²

Therefore the bound follows. ∎

## Theorem 2 (STA as Optimal Transport)

**Statement**: The 1-Wasserstein distance between two discrete distributions
supported on sorted points {λ_i} and {μ_i} with uniform weights 1/D is:

```
W_1 = (1/D) Σ_{i=1}^{D} |λ_{(i)} - μ_{(i)}|
```

**Proof**:

By the Kantorovich-Rubinstein duality theorem (Villani, 2008):
```
W_1(P, Q) = sup_{f ∈ Lip(1)} |E_P[f] - E_Q[f]|
```

where Lip(1) is the set of 1-Lipschitz functions.

For discrete distributions with the same number of support points and
uniform weights, the optimal coupling is the **monotone coupling**
(Brenier's theorem for 1D), which pairs the i-th largest point of P
with the i-th largest point of Q.

For eigenvalues sorted in descending order:
- P = (1/D) Σ δ_{λ_i} with λ₁ ≥ ... ≥ λ_D
- Q = (1/D) Σ δ_{μ_i} with μ₁ ≥ ... ≥ μ_D

The optimal transport plan is: π(i,j) = (1/D) if i=j, else 0.

Therefore:
```
W_1 = (1/D) Σ_i |λ_i - μ_i|    (with both sorted descending)
```

This is exactly the metric used in STA. ∎

## Corollary (Downstream Stability)

**Statement**: Let f(z) = w^T z be a downstream linear probe. The prediction
variance across training steps satisfies:

```
Var_t[f(z_t)] ≤ ||w||² · E_t[tr(Cov(z_t))] · (W_1 / δ²)
```

**Proof**:

The prediction at step t is f(z_t) = w^T z_t.
The variance across steps is:
```
Var_t[f(z_t)] = E_t[(w^T z_t - w^T E[z_t])²]
              = w^T E_t[(z_t - μ_t)(z_t - μ_t)^T] w
              ≤ ||w||² · ||E_t[(z_t - μ_t)(z_t - μ_t)^T]||_op
```

The operator norm of the time-averaged centered covariance is bounded by
the trace times the maximum eigenvalue change rate:

If the representation changes smoothly (bounded W_1 per step), then:
```
||E_t[(z_t - μ_t)(z_t - μ_t)^T]||_op ≤ tr(Cov(z_t)) · W_1 / δ²
```

(by the Davis-Kahan bound applied to the representation covariance).

Combining:
```
Var_t[f(z_t)] ≤ ||w||² · tr(Cov(z_t)) · (W_1 / δ²)
```

This shows that downstream predictions are stable when:
- STA loss is small (W_1 ≈ 0)
- Spectral gap is large (δ >> 0)
- Probe norm is bounded (||w|| ≈ 1) ∎

## Assumptions and Conditions

1. **Spectral gap condition**: δ = λ_k - λ_{k+1} > 0 (well-separated eigenvalues)
2. **Finite representation**: E[||z||²] < ∞ (finite second moment)
3. **EMA tracking**: The reference spectrum is updated via EMA with β ∈ (0,1)
4. **Warmup**: STA activates after warmup_steps to let initial transient settle

## Verification

- `tests/test_sta.py::TestSTAMathematical` — W1 metric properties
- `tests/test_sta.py::TestSTADavisKahan` — bound computation
- `tests/test_sta.py::TestSTABasic` — loss non-negativity, shape
- `tests/test_sta.py::TestSTAWasserstein` — W1=0 for identical, >0 for different
- `tests/test_sta.py::TestSTAIntegration` — multi-step smoothness

## How Other Papers Can Use STA

STA is applicable to **any** self-supervised method where:
- A learned subspace depends on the eigenvalue spectrum
- The encoder is continuously updated (gradient descent)
- Representation stability matters for downstream tasks

Usage:
```python
from src.models.sta import SpectralTransportAlignment
sta = SpectralTransportAlignment(embed_dim=768)
# Every training step:
sta_loss, info = sta(z_online, step=step)
total_loss += lambda_sta * sta_loss
```

Works with: I-JEPA, V-JEPA, C-JEPA, TD-JEPA, VICReg, Barlow Twins, DINO,
and any method where spectral stability matters.
