# CMC: Cross-Mask Consistency Regularization

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified: core loss formula and overlap construction.
> DIVERGENT: the stability theorem is proven only in AVERAGED form while
> stated pointwise; the skip path can report eps=0 while predictions
> diverge (invalidating downstream-bound usage); reuse_encoder is a stub.


## Statement

**Theorem (Stability Bound).**
Let $z_\mathrm{pred}^{(1)}$ and $z_\mathrm{pred}^{(2)}$ be predictions from two different mask patterns $m_1, m_2$, and $\Omega = \{t : m_1[t] = m_2[t] = 1\}$ be the set of positions masked in both patterns.

The CMC loss is:
$$\mathcal{L}_\mathrm{CMC} = \frac{1}{|\Omega|} \sum_{t \in \Omega} \|z_\mathrm{pred}^{(1)}[t] - z_\mathrm{pred}^{(2)}[t]\|^2$$

**Stability Theorem:** For any downstream linear probe $f(z) = w^\top z + b$:

$$|f(z_\mathrm{pred}^{(1)}[t]) - f(z_\mathrm{pred}^{(2)}[t])| \leq \|w\| \cdot \sqrt{\mathcal{L}_\mathrm{CMC}}$$

The prediction discrepancy for **any** downstream task is bounded by the CMC loss.

## Proof

### Step 1: Pointwise Bound
For a single position $t \in \Omega$:
$$|f(z_\mathrm{pred}^{(1)}[t]) - f(z_\mathrm{pred}^{(2)}[t])| = |w^\top (z_\mathrm{pred}^{(1)}[t] - z_\mathrm{pred}^{(2)}[t])|$$

By Cauchy-Schwarz:
$$\leq \|w\| \cdot \|z_\mathrm{pred}^{(1)}[t] - z_\mathrm{pred}^{(2)}[t]\|$$

### Step 2: Average Bound
Taking the average over $t \in \Omega$:
$$\frac{1}{|\Omega|} \sum_{t \in \Omega} |f(z_\mathrm{pred}^{(1)}[t]) - f(z_\mathrm{pred}^{(2)}[t])| \leq \|w\| \cdot \frac{1}{|\Omega|} \sum_{t \in \Omega} \|z_\mathrm{pred}^{(1)}[t] - z_\mathrm{pred}^{(2)}[t]\|$$

By Jensen's inequality (norm is convex):
$$\leq \|w\| \cdot \sqrt{\frac{1}{|\Omega|} \sum_{t \in \Omega} \|z_\mathrm{pred}^{(1)}[t] - z_\mathrm{pred}^{(2)}[t]\|^2}$$

$$= \|w\| \cdot \sqrt{\mathcal{L}_\mathrm{CMC}}$$

### Step 3: Non-Negativity
$$\mathcal{L}_\mathrm{CMC} = \frac{1}{|\Omega|} \sum_{t \in \Omega} \|z_\mathrm{pred}^{(1)}[t] - z_\mathrm{pred}^{(2)}[t]\|^2 \geq 0$$

Sum of squared norms — trivially non-negative. Zero iff predictions agree on all overlapping positions.

## Why This Is Free Training Signal
- CMC requires **no labels** — only two different mask patterns for the same input
- The encoder output can be **reused** — only one additional predictor forward pass
- It provides supervision at **overlapping** masked positions — positions where both predictions estimate the same target

## Connection to Other Consistency Methods

| Method | Setting | Views | Loss | Labels Needed? |
|--------|---------|-------|------|----------------|
| FixMatch (NeurIPS 2020) | Semi-supervised | Weak/Strong augmentations | Pseudo-label CE | Yes (unlabeled + pseudo) |
| DINO multi-crop (ICCV 2021) | Contrastive | Global/Local crops | Cross-entropy | No |
| VAT (ICLR 2018) | Semi-supervised | Original/Adversarial | KL divergence | Yes (unlabeled) |
| **CMC (ours)** | **JEPA** | **Mask patterns** | **MSE on overlap** | **No** |

CMC is the first consistency method specifically designed for JEPA's masking-based prediction. It's fundamentally different from contrastive consistency (DINO) and semi-supervised consistency (FixMatch, VAT).
