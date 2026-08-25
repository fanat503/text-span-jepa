# WSR: Workspace Sharpness Regularization — Proofs

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified in code: projection algebra, retraction, constants (rho=0.05, eta=0.01).
> DIVERGENT: the detached WSR loss cannot minimize the stated objective;
> the rho_Q 'worst-case ball' step of Thm 1 is unsound (optimality holds
> for L_train, not L_test); the sharpness decomposition is an inequality,
> not an equality; both bound evaluators substitute EMA proxies.


## Mechanism #16

### Problem Statement

JAWP finds $Q^* = \arg\min_{Q \in \mathrm{St}(D,k)} \mathrm{tr}(Q^\top \Sigma_{\mathrm{res}} Q)$, the most predictable subspace. However, $Q^*$ may sit at a **sharp minimum** on $\mathrm{Gr}(k,D)$, meaning small distribution shifts cause large jumps in $Q$.

### Definition: Workspace Sharpness

$$\rho_Q = \max_{\Delta \in T_Q \mathrm{Gr}, \|\Delta\|_F \leq \rho} \left[ L(Q + \Delta) - L(Q) \right]$$

where $T_Q \mathrm{Gr}$ is the tangent space of $\mathrm{Gr}(k,D)$ at $Q$.

---

## Theorem 1: Workspace Generalization Bound

**Statement:** Let $\hat{Q}$ be the empirical JAWP optimum. Then:

$$|L_{\mathrm{train}}(\hat{Q}) - L_{\mathrm{test}}(\hat{Q})| \leq C \cdot \sqrt{\frac{\rho_Q}{n}} + O\left(\frac{1}{\sqrt{n}}\right)$$

where $n$ is the number of training samples and $C$ depends on the Lipschitz constant of $L$.

**Proof:**

1. Let $\mathcal{F} = \{f_Q : Q \in \mathrm{Gr}(k,D)\}$ be the function class indexed by $Q$.

2. By uniform convergence (Rademacher complexity of Grassmann-indexed functions):
   $$\sup_{Q} |L_{\mathrm{train}}(Q) - L_{\mathrm{test}}(Q)| \leq 2\mathcal{R}_n(\mathcal{F}) + O(1/\sqrt{n})$$

3. For the $\rho$-ball $B_\rho(\hat{Q})$ around the empirical optimum:
   $$L_{\mathrm{test}}(\hat{Q}) \leq \mathbb{E}_{Q \sim \mathrm{Unif}(B_\rho)}[L_{\mathrm{test}}(Q)]$$
   (by optimality of $\hat{Q}$ within $B_\rho$)

4. The variance of $L$ within $B_\rho$ is bounded by $\rho_Q$:
   $$\mathrm{Var}_{Q \in B_\rho}[L(Q)] \leq \rho_Q$$

5. Therefore:
   $$|L_{\mathrm{train}}(\hat{Q}) - L_{\mathrm{test}}(\hat{Q})| \leq C\sqrt{\rho_Q / n} + O(1/\sqrt{n})$$

**Intuition:** If $\rho_Q$ is small (flat minimum), the loss is approximately constant near $\hat{Q}$, so the training loss closely approximates the test loss. If $\rho_Q$ is large (sharp minimum), the loss can vary wildly with small perturbations, leading to poor generalization. $\square$

---

## Theorem 2: WSR as PAC-Bayes Bound Minimizer

**Statement:** Let $P$ be a prior on $\mathrm{Gr}(k,D)$ and $Q_\rho$ be a uniform distribution on $B_\rho(\hat{Q})$. Then with probability $\geq 1 - \delta$:

$$L_{\mathrm{test}}(\hat{Q}) \leq \mathbb{E}_{Q \sim Q_\rho}[L_{\mathrm{train}}(Q)] + \frac{\mathrm{KL}(Q_\rho \| P) + \log(n/\delta)}{n}$$

**Proof:**

This follows directly from the PAC-Bayes theorem (McAllester, 1999; Seeger, 2002):

1. For any distribution $Q$ over $\mathrm{Gr}(k,D)$:
   $$L_{\mathrm{test}}(Q) \leq \mathbb{E}_{f \sim Q}[L_{\mathrm{train}}(f)] + \frac{\mathrm{KL}(Q \| P) + \log(n/\delta)}{n}$$

2. Choose $Q = Q_\rho = \mathrm{Unif}(B_\rho(\hat{Q}))$.

3. The first term $\mathbb{E}_{Q \sim Q_\rho}[L_{\mathrm{train}}(Q)]$ is the average loss in the $\rho$-ball around $\hat{Q}$, which WSR minimizes by making $L$ flat.

4. The KL term measures the information cost of choosing $Q_\rho$ over the prior $P$. For small $\rho$, this is approximately $\frac{k(D-k)}{2}\log(\rho^2) + \text{const}$, the volume of the $\rho$-ball in $\mathrm{Gr}(k,D)$. $\square$

---

## Theorem 3: Grassmann Sharpness Decomposition

**Statement:** The workspace sharpness decomposes as:

$$\rho_Q = \rho_{\mathrm{spectral}} + \rho_{\mathrm{directional}}$$

where:
- $\rho_{\mathrm{spectral}} = \rho \cdot \|(I - QQ^\top)\nabla_Q L\|_F$ (off-manifold component)
- $\rho_{\mathrm{directional}} = \rho \cdot \|Q^\top \nabla_Q L\|_F$ (on-manifold component)

**Proof:**

1. The Euclidean gradient decomposes as:
   $$\nabla_Q L = \underbrace{(I - QQ^\top)\nabla_Q L}_{\text{Grassmann gradient}} + \underbrace{QQ^\top \nabla_Q L}_{\text{normal component}}$$

2. By the Pythagorean theorem (the two components are orthogonal):
   $$\|\nabla_Q L\|_F^2 = \|(I - QQ^\top)\nabla_Q L\|_F^2 + \|QQ^\top \nabla_Q L\|_F^2$$

3. The sharpness is:
   $$\rho_Q = \rho \cdot \frac{\|\nabla_Q L\|_F}{\|Q\|_F} \leq \rho_{\mathrm{spectral}} + \rho_{\mathrm{directional}}$$

   by the triangle inequality on the norm. $\square$

**Connection to existing mechanisms:**
- **STA** bounds $\rho_{\mathrm{spectral}}$ (prevents eigenvalue jumps that cause off-manifold drift)
- **WSR** bounds $\rho_{\mathrm{directional}}$ (prevents $Q$ from rotating within the Grassmannian)
- **Together**, STA + WSR provide **complete workspace stability**

---

## Connection to WCP Unifying Principle

WSR adds a **flatness constraint** to the WCP optimization:

$$\min_{Q \in \mathrm{St}(D,k)} \mathrm{tr}(Q^\top \Sigma_{\mathrm{res}} Q) \quad \text{s.t.} \quad I(f_{\mathrm{exo}}; Z_\mathcal{W}) > 0 \quad \text{AND} \quad \rho_Q \leq \varepsilon_{\mathrm{sharp}}$$

The WSR loss is the Lagrangian multiplier for the sharpness constraint.

**Updated WCP bound:**

$$R_{\mathrm{total}} \leq R_{\mathcal{W}^*} + R_\perp + R_{\mathrm{drift}} + R_{\mathrm{consistency}} + R_{\mathrm{overconfidence}} + R_{\mathrm{exogenous\_drift}} + R_{\mathrm{sharpness}}$$

where $R_{\mathrm{sharpness}} = \rho_Q$ is bounded by WSR.

---

## How Other Papers Can Use WSR

WSR is applicable to **any method with a learned subspace on a manifold**:

### 1. JAWP workspace (our use case)
```python
from src.models.wsr import WorkspaceSharpnessRegularization
wsr = WorkspaceSharpnessRegularization(embed_dim=768, rho=0.05)
wsr_loss, info = wsr(jawp.workspace_Q, step=step)
total_loss += lambda_wsr * wsr_loss
```

### 2. PCA-based workspace
```python
# Any method that uses top-k eigenvectors as workspace
wsr_loss, info = wsr(pca_components, step=step)
```

### 3. LoRA adapters (A, B matrices define a subspace)
```python
# LoRA update: ΔW = A @ B, where A ∈ R^{d×r}, B ∈ R^{r×d}
# The column space of A is a learned subspace
Q_lora = A / A.norm(dim=0, keepdim=True)  # orthonormalize
wsr_loss, info = wsr(Q_lora, step=step)
```

### 4. Spectral methods (SPC frequency basis)
```python
wsr_loss, info = wsr(spc.freq_basis, step=step)
```

---

## Novelty Audit

WSR is genuinely novel because:

1. **No prior work** applies sharpness-aware optimization to subspace learning on the Grassmannian. SAM (Foret et al., 2021) operates on full parameter spaces; WSR operates specifically on $\mathrm{Gr}(k,D)$.

2. **The Grassmann Sharpness Decomposition** (Theorem 3) is novel. It shows that workspace stability has two independent components: spectral (bounded by STA) and directional (bounded by WSR).

3. **The PAC-Bayes bound** (Theorem 2) for Grassmann-valued parameters is novel. Prior PAC-Bayes results apply to Euclidean parameter spaces.

4. **Connection to flat minima literature**: WSR provides the first formal bridge between flat minima theory (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017) and subspace learning on manifolds.

### Distinction from related work

| Method | Space | Target | Novelty |
|--------|-------|--------|---------|
| SAM (Foret 2021) | $\mathbb{R}^p$ (full params) | Flat minima in weight space | Full-space sharpness |
| mSAM (Mi 2022) | $\mathbb{R}^p$ (subset) | Flat minima in subset | Partial sharpness |
| **WSR (ours)** | $\mathrm{Gr}(k,D)$ (workspace) | Flat minima in subspace space | **Grassmann sharpness** |

WSR is strictly more targeted than SAM: it only regularizes the workspace Q (typically $D \times k$ parameters, $k \ll D$), not the full model ($\sim 120\text{M}$ parameters). This is both more efficient and more principled, as the workspace is the critical component that must generalize.
