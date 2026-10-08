# WSD: Workspace-Target Synchronization Drift

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified: chordal distance loss and Grassmann machinery.
> DIVERGENT: load-bearing assumptions (exact orthonormality of target_Q,
> lambda-to-penalty coupling) are not enforced; the 'constructive'
> Davis-Kahan bound is circular; adaptive tau and per-step resync are
> unimplemented. Silent eig-failure now warns (fixed R15).
>
> **Cross-mechanism, not in the 2026-08-24 audit (added 2026-10-08, found by
> TASK-40).** The `Q_target` this theorem penalizes distance to is the
> **top-k** eigenvectors of an EMA of the *target covariance*
> (`src/models/wsd.py:200-206`), and the loss maximizes `‖Q_JAWPᵀ Q_target‖²_F`
> (`src/models/wsd.py:256-258`) — so WSD pulls `Q` toward the target's
> **highest-variance** directions. JAWP's minimizer on `St(D,k)` is the
> **bottom-k** eigenvectors of the *residual* covariance
> (`src/models/jawp.py:45-65`), and JAWP's header names high-variance
> selection as the failure mode to avoid. Same parameter, opposite criteria,
> and the reference model has both on (`defaults.yaml`: `use_wsd: true`,
> `lambda_wsd: 0.01`, `use_jawp: true`). This is a science decision, not a
> bug: the two act on different matrices (`Cov(z_target)` vs `Σ_res`) and on
> different clocks (`wsd_sync_interval: 100` vs every step), so they may be
> reconcilable — but nothing in the repo says so. Recorded in
> `proofs/IMPLEMENTATION_STATUS.md` pattern 5; the resolution is the owner's.


## Statement

**Theorem (Drift Bound).**
Let $Q_\mathrm{online}(t)$ be the JAWP workspace of the online encoder at time $t$, and $Q_\mathrm{target}(t)$ be the top-$k$ PCA subspace of the target encoder. The Grassmann distance $d_\mathrm{Gr}(Q_\mathrm{online}, Q_\mathrm{target})$ satisfies:

$$\Delta_\mathrm{WSD}(t) \leq \Delta(0) \cdot e^{-\lambda t} + \frac{\nu_\max}{\lambda}$$

where:
- $\lambda > 0$: synchronization rate (controlled by WSD penalty strength)
- $\nu_\max$: maximum drift rate of the target encoder
- $\Delta(0)$: initial drift

### Steady-State Error
$$\Delta_\mathrm{WSD}(\infty) = \frac{\nu_\max}{\lambda}$$

The workspace can never perfectly track the target — there is always a lag proportional to the target's drift rate.

## Proof

### Step 1: Drift Dynamics
The online workspace evolves under two forces:
1. **JAWP optimization**: drives $Q$ toward the online encoder's workspace
2. **EMA target update**: shifts the target encoder at rate $\nu(t)$

The drift dynamics:
$$\frac{d}{dt} \Delta(t) = -\lambda \Delta(t) + \nu(t)$$

where $\Delta(t) = d_\mathrm{Gr}(Q_\mathrm{online}(t), Q_\mathrm{target}(t))$.

### Step 2: Solution of the ODE
This is a first-order linear ODE. Solution:

$$\Delta(t) = \Delta(0) e^{-\lambda t} + \int_0^t e^{-\lambda(t-s)} \nu(s) \, ds$$

### Step 3: Upper Bound
Since $\nu(s) \leq \nu_\max$:

$$\Delta(t) \leq \Delta(0) e^{-\lambda t} + \nu_\max \int_0^t e^{-\lambda(t-s)} \, ds = \Delta(0) e^{-\lambda t} + \frac{\nu_\max}{\lambda}(1 - e^{-\lambda t})$$

For $t \to \infty$: $\Delta(t) \leq \frac{\nu_\max}{\lambda}$.

### Step 4: WSD Loss
We penalize the current drift:
$$\mathcal{L}_\mathrm{WSD} = d_\mathrm{Gr}(Q_\mathrm{online}, Q_\mathrm{target})^2$$

This increases the effective $\lambda$, reducing the steady-state error.

## Constructive Bound via STA
The drift bound is non-constructive because $\nu_\max$ is unknown. STA (mechanism #13) provides a **constructive** bound via Davis-Kahan:

$$d_\mathrm{Gr}(Q_\mathrm{online}, Q_\mathrm{target}) \leq \frac{\|\Sigma_\mathrm{online} - \Sigma_\mathrm{target}\|_2}{\delta}$$

where $\delta$ is the spectral gap (observable from eigenvalues).

## Connection to EMA Scheduling
Standard I-JEPA adjusts $\tau$ to control target update speed, but doesn't detect when $Q$ becomes stale. WSD provides a **direct measurement** of workspace-target misalignment, enabling adaptive $\tau$ scheduling.

## Practical Computation
1. Compute $Q_\mathrm{online}$ from JAWP (already available)
2. Compute $Q_\mathrm{target}$ via top-$k$ SVD of target encoder output covariance
3. Compute Grassmann distance: $d_\mathrm{Gr} = \|\sin\Theta\|_F$ where $\Theta$ are principal angles
4. Update running drift statistics (EMA)
5. Add $\lambda_\mathrm{WSD} \cdot d_\mathrm{Gr}^2$ to total loss
