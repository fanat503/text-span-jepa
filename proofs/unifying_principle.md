# Unifying Principle: Workspace-Conditioned Prediction

## Central Claim

All **16 numbered GWP capabilities** — the twelve mechanism *modules* plus the
four *methods of `JAWPModule`* (WIP, Spectral Gap, Grassmann Optimization,
Predictive Rank), per the counting convention in [`README.md`](README.md) — are
instances of a single optimization principle. (This file previously said "All 15
mechanisms" in two places while the table immediately below listed 16 rows; the
table was right and the sentence was not. Corrected 2026-10-08, TASK-41.)

**Workspace-Conditioned Prediction (WCP):**
> Predict $Y$ from $X$ using a representation that maximizes $I(Z_\mathcal{W}; Y)$ while preserving $I(Z_\mathcal{W}; f_{\mathrm{exo}})$, where $\mathcal{W} = \operatorname{span}(Q)$ is a learned subspace optimized on $\mathrm{St}(D,k)$.

## Derivation of Each Mechanism

### Core Principle
$$\min_{Q \in \mathrm{St}(D,k)} \underbrace{\operatorname{tr}(Q^\top \Sigma_{\mathrm{res}} Q)}_{\text{JAWP}} \quad \text{s.t.} \quad \underbrace{I(f_{\mathrm{exo}}; Z_\mathcal{W}) > 0}_{\text{WIP}}$$

| Mechanism | WCP Instance | What It Optimizes |
|-----------|-------------|-------------------|
| **JAWP** | Core objective | $\min \operatorname{tr}(Q^\top \Sigma_{\mathrm{res}} Q)$ on $\mathrm{St}(D,k)$ |
| **WIP** | Constraint | $I(f_{\mathrm{exo}}; Z_\mathcal{W}) > 0$ |
| **Spectral Gap** | Dimension selection | $k^* = \arg\max_k [\lambda_k - \lambda_{k+1}]$ (MP law) |
| **Grassmann** | Geometry | Optimize on $\mathrm{Gr}(k,D)$ instead of $\mathrm{St}(D,k)$ |
| **Predictive Rank** | Regularization | $\log\det(Q^\top \Sigma_\mathcal{W} Q) > 0$ (barrier) |
| **CGN** | Routing | Split prediction into workspace/background pathways |
| **SWIP** | Background shaping | Whitening $\Sigma_\perp$ while preserving $\Sigma_\mathcal{W}$ eigenvalue hierarchy |
| **PCR** | Refinement | Recover information lost through $k$-dimensional bottleneck |
| **SPC** | Spectral allocation | Weight prediction loss by frequency-band information content |
| **WSD** | Stability | Monitor $d_\mathrm{Gr}(Q_\mathrm{online}, Q_\mathrm{target})$ over training |
| **CMC** | Consistency | Ensure predictions agree across different masks |
| **GAC** | Exploration | Ensure background dimensions receive gradient |
| **STA** | Transport | Align spectral distributions via $W_1$ metric |
| **PUC** | Uncertainty | Prevent predictor overconfidence via entropy constraint |
| **RDC** | Drift control | Prevent orthogonal drift that discards exogenous features |
| **WSR** | Flatness | Ensure workspace sits at flat minimum on Gr(k,D) |

## The WCP Theorem

**Theorem (Workspace-Conditioned Prediction Bound).**
Let $\mathcal{W}^* = \operatorname{span}(Q^*)$ where $Q^*$ solves the WCP optimization. Then the total downstream risk satisfies:

$$R_{\mathrm{total}} \leq R_{\mathcal{W}^*} + R_\perp + R_{\mathrm{drift}} + R_{\mathrm{consistency}} + R_{\mathrm{overconfidence}} + R_{\mathrm{exogenous\_drift}}$$

where:
- $R_{\mathcal{W}^*}$: prediction risk in workspace (minimized by JAWP + SPC)
- $R_\perp$: background risk (controlled by SWIP + GAC)
- $R_{\mathrm{drift}}$: workspace-target drift (bounded by WSD + STA)
- $R_{\mathrm{consistency}}$: cross-mask inconsistency (bounded by CMC)
- $R_{\mathrm{overconfidence}}$: predictor entropy deficit (bounded by PUC)
- $R_{\mathrm{exogenous\_drift}}$: orthogonal drift discarding exogenous info (bounded by RDC)

### Proof Sketch

1. **Decomposition**: By the orthogonality of $\mathcal{W}$ and $\mathcal{W}^\perp$:
   $$\|Z - \hat{Z}\|^2 = \|Z_\mathcal{W} - \hat{Z}_\mathcal{W}\|^2 + \|Z_\perp - \hat{Z}_\perp\|^2$$

2. **Workspace risk**: $\|Z_\mathcal{W} - \hat{Z}_\mathcal{W}\|^2 = \operatorname{tr}(Q^\top \Sigma_{\mathrm{res}} Q)$ — minimized by JAWP (Courant-Fischer). SPC further reduces this by weighting high-information bands.

3. **Background risk**: Controlled by SWIP (prevents collapse) and GAC (prevents starvation). Without these, background can degenerate.

4. **Drift term**: As the target encoder evolves, $Q$ may become stale. WSD bounds $d_\mathrm{Gr}(Q, Q_\mathrm{target})$, and STA provides constructive bounds via Davis-Kahan.

5. **Consistency term**: CMC ensures predictions are stable across mask patterns, reducing variance of downstream estimates.

## Why This Unification Matters

### For Paper
- **Single claim**: "We introduce Workspace-Conditioned Prediction, a principled framework for text representation learning."
- **Each mechanism is an instance**: Not 16 ad-hoc tricks, but 16 components of ONE optimization problem. (Was "13"; 13 was the *proof-document* count, not a mechanism count — corrected 2026-10-08, TASK-41.)
- **Ablation hierarchy**: Removing mechanisms = relaxing constraints, with clear mathematical interpretation.

### For Reviewers
- **"Too many mechanisms"**: They're not independent — they're all derived from one principle.
- **"No unifying theory"**: WCP is the unifying theory.
- **"Ablation needed"**: Each ablation corresponds to dropping a term in the WCP bound.

### For Top Labs
- **Adoption path**: Start with JAWP (core), add mechanisms incrementally.
- **Each mechanism is optional**: The framework works with any subset.
- **The bound tells you what you're missing**: Each dropped term increases a specific risk component.

### PUC (Mechanism #14) — Prediction Uncertainty

PUC addresses the **entropy constraint** of WCP:

$$H(Z_\mathrm{pred}) \geq H_\mathrm{target}$$

Without this constraint, the predictor can degenerate to a delta function (zero entropy), providing no gradient signal to the encoder. PUC enforces this via log-determinant barrier:

$$\mathcal{L}_\mathrm{PUC} = \eta \cdot \max(0, H_\mathrm{target} - H(\Sigma_\mathrm{pred}))$$

By Donsker-Varadhan duality, this is the tightest convex relaxation of the KL divergence to the maximum-entropy distribution.

**WCP bound contribution**: PUC adds an entropy term to the total risk:

$$R_\mathrm{total} \leq R_{\mathcal{W}^*} + R_\perp + R_\mathrm{drift} + R_\mathrm{consistency} + R_\mathrm{overconfidence}$$

where $R_\mathrm{overconfidence} = \max(0, H_\mathrm{target} - H(\Sigma_\mathrm{pred}))$ is bounded by PUC.

### Mechanism #15: RDC — Representation Drift Compensation

**WCP constraint**: ||Δz_⊥||² ≤ ε_max

RDC adds a drift constraint to the WCP optimization:

$$\min_{Q \in \mathrm{St}(D,k)} \mathrm{tr}(Q^\top \Sigma_{\mathrm{res}} Q) \quad \text{s.t.} \quad I(f_{\mathrm{exo}}; Z_\mathcal{W}) > 0 \quad \text{AND} \quad ||\Delta z_\perp||^2 \leq \varepsilon_\max$$

The RDC loss $L_\mathrm{RDC} = \eta \cdot ||\Delta z_\perp||^2$ is the Lagrangian multiplier for the drift constraint.

**Drift Compensation Bound**: $||z_{\perp,T} - z_{\perp,0}|| \leq \varepsilon(1-\eta_\mathrm{rdc})^T \cdot T/\sqrt{k}$

**WCP bound contribution**: RDC adds a drift compensation term:

$$R_\mathrm{total} \leq R_{\mathcal{W}^*} + R_\perp + R_\mathrm{drift} + R_\mathrm{consistency} + R_\mathrm{overconfidence} + R_\mathrm{exogenous\_drift}$$

where $R_\mathrm{exogenous\_drift} = \eta_\mathrm{rdc} \cdot ||\Delta z_\perp||^2$ is the penalty for drift that could discard exogenous features.

**Full bound.** Ten risk terms, covering **eleven of the twelve modules** —
WSR appears nowhere in it, and the four non-module capabilities (WIP, Spectral
Gap, Grassmann Optimization, Predictive Rank) are not risk terms at all. (This
line previously read "with all 15 mechanisms", which was true of neither the 12
nor the 16. Corrected 2026-10-08, TASK-41; the missing WSR term is left missing
because inventing it would be designing a theorem, which is a human decision.)

$$R_\mathrm{total} \leq \underbrace{R_{\mathcal{W}^*}}_{\text{JAWP}} + \underbrace{R_\perp}_{\text{SWIP}} + \underbrace{R_\mathrm{drift}}_{\text{WSD+STA}} + \underbrace{R_\mathrm{consistency}}_{\text{CMC}} + \underbrace{R_\mathrm{overconfidence}}_{\text{PUC}} + \underbrace{R_\mathrm{exogenous\_drift}}_{\text{RDC}} + \underbrace{R_\mathrm{bottleneck}}_{\text{PCR}} + \underbrace{R_\mathrm{spectral}}_{\text{SPC}} + \underbrace{R_\mathrm{routing}}_{\text{CGN}} + \underbrace{R_\mathrm{exploration}}_{\text{GAC}}$$
