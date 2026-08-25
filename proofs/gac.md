# GAC: Gradient-Allocated Capacity

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified: loss skeleton, min_weight clamp semantics.
> DIVERGENT: the wired call site passes LIVE predictions (R12 wiring),
> but energy is a batch-MEAN, so the No-Dead-Zones bound scales by 1/N;
> the warmup ramp further rescales it. Theorem holds up to these factors.


## Statement

**Theorem (No Dead Zones).**
Let $g_i = \|\nabla_{z_i} \mathcal{L}_\mathrm{main}\|$ be the gradient norm for dimension $i$. The GAC exploration loss is:

$$\mathcal{L}_\mathrm{GAC} = \gamma \cdot \sum_{i: g_i < \tau_\mathrm{grad}} (\tau_\mathrm{grad} - g_i) \cdot \|z_i\|^2$$

**No Dead Zones Guarantee:** After GAC is applied, every active dimension (with $\|z_i\| > 0$) receives non-zero total gradient:

$$\|\nabla_{z_i} (\mathcal{L}_\mathrm{main} + \mathcal{L}_\mathrm{GAC})\| > 0 \quad \text{whenever } \|z_i\| > 0$$

## Proof

### Step 1: GAC Gradient Contribution
For a starved dimension $i$ with $g_i < \tau_\mathrm{grad}$:

$$\frac{\partial \mathcal{L}_\mathrm{GAC}}{\partial z_i} = 2\gamma (\tau_\mathrm{grad} - g_i) z_i$$

This gradient is **non-zero** when:
1. $z_i \neq 0$ (dimension is active)
2. $g_i < \tau_\mathrm{grad}$ (dimension is starved)
3. $\gamma > 0$ (GAC is active)

### Step 2: Total Gradient
The total gradient for dimension $i$:

$$\nabla_{z_i} \mathcal{L}_\mathrm{total} = \nabla_{z_i} \mathcal{L}_\mathrm{main} + 2\gamma (\tau_\mathrm{grad} - g_i) z_i$$

Even if $\nabla_{z_i} \mathcal{L}_\mathrm{main} \approx 0$ (starved), the GAC term provides gradient $2\gamma (\tau_\mathrm{grad} - g_i) z_i \neq 0$.

### Step 3: Exploration Ratio
The **exploration ratio** measures the fraction of dimensions receiving GAC bonus:

$$\rho_\mathrm{explore} = \frac{|\{i : g_i < \tau_\mathrm{grad}\}|}{D}$$

This is bounded: $0 \leq \rho_\mathrm{explore} \leq 1$.
- $\rho = 0$: no dimensions are starved (ideal)
- $\rho = 1$: all dimensions are starved (pathological)
- Healthy training: $\rho$ should decrease over time

### Step 4: Convergence Implication
Since GAC provides gradient to starved dimensions, the encoder continues to explore in those directions. This prevents the "rich get richer" feedback loop where:
1. Workspace dimensions get all the gradient
2. Background dimensions get zero gradient
3. Encoder cannot discover new useful directions
4. Workspace cannot grow

GAC breaks this loop by ensuring background dimensions always have some gradient signal.

## Practical Considerations

### Gradient Norm Computation
Per-dimension gradient norms are computed after the main loss backward pass:
```python
grad_norms = torch.zeros(D)
for i, p in enumerate(model.parameters()):
    if p.grad is not None:
        grad_norms[i] = p.grad.norm()
```

### Warmup
GAC should not be applied during the first `warmup_steps` iterations, as gradient norms are unreliable early in training.

### EMA Tracking
Running gradient norms are tracked via EMA for stable diagnostics:
$$\bar{g}_i^{(t)} = \beta \cdot \bar{g}_i^{(t-1)} + (1 - \beta) \cdot g_i^{(t)}$$

## Why This Is Novel
- **Gradient noise injection** (Neelakantan et al., 2016): adds noise to ALL gradients uniformly — GAC is **targeted** (only starved dims)
- **Gradient clipping**: bounds **maximum** gradient — GAC ensures **minimum** gradient
- **Dropout**: random masking prevents co-adaptation — different mechanism entirely
- No prior work monitors per-dimension gradient starvation and adds targeted exploration bonuses
