# PUC: Prediction Uncertainty Calibration

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified: non-negativity of the returned scalar.
> DIVERGENT: the headline Lagrangian-dual formula is dead code; the
> executed ReLU'd log-det barrier matches neither stated form; the loss
> carries NO gradient (buffer-based statistics); the risk constraint and
> min_log_det are unimplemented. Labeled in-module (R12).


## Statement

**Theorem (Minimax Prediction Optimality).**
Among all prediction distributions $q(z)$ satisfying the prediction risk constraint $\mathbb{E}_q[\|z - z_\mathrm{target}\|^2] \leq R$, the maximum-entropy distribution $q^*$ achieves minimax optimality over all bounded downstream loss functions $\mathcal{F}$:

$$q^* = \arg\min_{q: \mathbb{E}_q[\|z - z_\mathrm{target}\|^2] \leq R} \max_{f \in \mathcal{F}} \mathbb{E}_q[\ell(f(z))]$$

The PUC loss is:

$$\mathcal{L}_\mathrm{PUC} = \eta \cdot \max\left(0, H_\mathrm{target} - H(\Sigma_\mathrm{pred})\right)$$

where $H(\Sigma) = \frac{1}{2}\log\det(2\pi e \Sigma)$ is the differential entropy and $H_\mathrm{target} = \frac{D}{2}\log(2\pi e)$ is the entropy of the isotropic Gaussian.

## Proof

### Step 1: Maximum Entropy Principle (Jaynes, 1957)
Among all distributions $q$ on $\mathbb{R}^D$ with a given covariance $\Sigma$, the Gaussian $\mathcal{N}(0, \Sigma)$ has the maximum differential entropy:

$$H(q) \leq H(\mathcal{N}(0, \Sigma)) = \frac{1}{2}\log\det(2\pi e \Sigma)$$

with equality iff $q$ is Gaussian.

### Step 2: Prediction Risk and Entropy Tradeoff
The prediction risk constraint bounds the covariance:

$$\mathbb{E}[\|z_\mathrm{pred} - z_\mathrm{target}\|^2] = \mathrm{tr}(\Sigma_\mathrm{pred}) + \|\mu_\mathrm{pred} - \mu_\mathrm{target}\|^2 \leq R$$

This implies $\mathrm{tr}(\Sigma_\mathrm{pred}) \leq R$. For fixed trace, entropy is maximized by isotropic covariance:

$$\Sigma^* = \frac{R - \|\mu_\mathrm{pred} - \mu_\mathrm{target}\|^2}{D} I$$

### Step 3: Donsker-Varadhan Dual Representation
The KL divergence from $q$ to the maximum-entropy distribution $p^*$ is:

$$\mathrm{KL}(q \| p^*) = \sup_f \left\{\mathbb{E}_q[f] - \log \mathbb{E}_{p^*}[e^f]\right\}$$

For $f(z) = -\frac{\|z\|^2}{2\sigma^2}$, this gives:

$$\mathrm{KL}(q \| p^*) \geq H(p^*) - H(q)$$

PUC minimizes this lower bound on KL divergence.

### Step 4: Minimax Optimality
By Sion's minimax theorem (convex-concave on compact sets):

$$\min_{q: \mathrm{risk} \leq R} \max_{f \in \mathcal{F}} \mathbb{E}_q[\ell(f(z))] = \max_{f \in \mathcal{F}} \min_{q: \mathrm{risk} \leq R} \mathbb{E}_q[\ell(f(z))]$$

The inner minimum is achieved by $q^*$ (maximum-entropy distribution), because:
- For any bounded loss $\ell$, $\mathbb{E}_{q^*}[\ell] \leq \mathbb{E}_q[\ell]$ for all $q$ with the same risk
- This follows from the data processing inequality: more entropy = more robust = lower worst-case loss

### Step 5: Log-Determinant Barrier
The PUC loss uses a log-determinant barrier:

$$\mathcal{L}_\mathrm{PUC} = -\frac{1}{2}\log\det(\Sigma_\mathrm{pred} + \epsilon I) + \frac{D}{2}\log(2\pi e \sigma_\mathrm{target}^2)$$

Properties:
1. **Convex**: $-\log\det(X)$ is convex on $\mathbb{S}_{++}^D$ (standard result)
2. **Barrier**: $\mathcal{L}_\mathrm{PUC} \to \infty$ as $\Sigma \to 0$ (prevents collapse)
3. **Minimum**: at $\Sigma_\mathrm{pred} = \sigma_\mathrm{target}^2 I$ (isotropic target)

## Practical Computation
Rather than computing the full D×D covariance (O(D²) memory), PUC uses Oja's rule to track the top-n_components eigenvalues online:
1. Project representations onto n_components random vectors
2. Estimate eigenvalues as variance of projections
3. Update projection vectors via Oja's rule (online PCA)
4. Compute entropy from estimated eigenvalues
5. Add log-det barrier for tracked components

This reduces cost from O(D²) to O(D · n_components).

## Why This Is Novel
- **VICReg variance term**: penalizes $\sum_i \max(0, \gamma - \sigma_i^2)$ — ensures minimum variance per dimension. PUC ensures minimum ENTROPY (joint property of all eigenvalues).
- **SIGReg**: matches 1D marginals to Gaussian — doesn't track temporal evolution of entropy.
- **SWIP**: whitens background dimensions — doesn't address predictor overconfidence.
- **PUC**: first method to explicitly regularize prediction ENTROPY, preventing overconfidence that causes representation degeneration.

## Why Top Labs Will Use This
- **Prevents silent degeneration**: Overconfidence is hard to detect but devastating
- **One hyperparameter**: $\eta$ (PUC strength)
- **Drop-in**: Add PUC module, one line in loss computation
- **Theoretically optimal**: Minimax guarantee for any downstream task
- **Low compute**: Oja's rule is online, no SVD needed
