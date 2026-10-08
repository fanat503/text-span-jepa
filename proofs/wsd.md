# WSD: Workspace-Target Synchronization Drift

> **IMPLEMENTATION STATUS (audited 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`.
> Verified: chordal distance loss and Grassmann machinery.
> DIVERGENT: load-bearing assumptions (exact orthonormality of target_Q,
> lambda-to-penalty coupling) are not enforced; the 'constructive'
> Davis-Kahan bound is circular; adaptive tau and per-step resync are
> unimplemented. Silent eig-failure now warns (fixed R15).
>
> **ADDENDUM (TASK-40 + TASK-41, 2026-10-08) — WSD and JAWP pull the SAME
> parameter in OPPOSITE directions.** Not in the 2026-08-24 matrix; first triaged
> under TASK-40. Nothing in either loss was changed.
>
> **The mechanism.** `Q_target` here is the **top-k** eigenvectors of an EMA of
> the *target* covariance (`src/models/wsd.py:200-206`), and the loss maximizes
> `⟨Q_JAWPᵀ Q_target⟩_F` (`src/models/wsd.py:256-258`) — so WSD pulls `Q` toward
> the target's **highest-variance** directions. JAWP's minimizer on `St(D,k)` is
> the **bottom-k** eigenvectors of the *residual* covariance
> (`src/models/jawp.py:45-65`), and JAWP's header names high-variance selection as
> the failure mode to avoid.
>
> **Not a false alarm, and the "different matrices / different clocks" escape does
> not apply.** `src/models/jepa.py:1086` hands WSD a **live view** of
> `jawp.workspace_Q` (no `.data()`), and both terms sit in the same `total_loss`
> in the same backward pass. Measured in situ at D=64, k=6: the cosine between
> the two `Q`-gradients is **negative on 12 of 12 consecutive steps** (mean
> ≈ −0.47), and WSD's raw `Q`-gradient is ≈2.4× JAWP's, giving a measured
> crossover at **lambda_wsd ≈ 0.42**. At lambda ≥ 1
> `jawk_predictive_relevance` collapses to **0.0000** — the workspace stops
> being predictable.
>
> **Reachability — corrected against the code, and it matters.** TASK-40 first
> reported that every runnable config ships `use_wsd: false`. **That was wrong.**
> Re-measuring the deep-merge over all 62 configs: `defaults.yaml` sets
> `use_wsd: true`, `use_jawp: true`, `lambda_wsd: 0.01`, and **51 of 62 configs
> resolve `use_wsd: true`, 50 with both flags true** — including every
> leave-one-out ablation row, which is the table the paper will run. The
> antagonism is therefore **live**, not dormant.
>
> **Why nothing has gone wrong yet.** At the shipped `lambda_wsd: 0.01` it sits
> roughly **42× below** the measured crossover, so it is present but not
> dominant. The crossover figure is scale- and schedule-specific (D=64, k=6, 60
> steps, random tokens) — an order of magnitude, not a portable constant.
>
> **Traps recorded.** The tempting one-line fix (detach `Q` so the two stops
> share a parameter) makes this Drift Bound false *by construction* while
> `tests/test_sterility.py` still passes, because `test_wsd_component_nonzero`
> only asserts the logged number is > 0. It is **not recommended**. Separately,
> `src/models/mechanisms.py:475` passes `.data`, so `MechanismBundle`'s WSD loss
> currently carries **no gradient at all** (verified `requires_grad = False`) —
> reachable in `TextSpanJEPA`, dead in the bundle path.
>
> Two of the most consequential changes available on this seam are **invisible
> to the entire required verify set**: flipping WSD to bottom-k, and removing
> that `.data`, each leave all 105 verify-set tests passing. The five options
> with their consequences are in `.agent-notes/task-40.md`. **The resolution is
> the owner's science call.**

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

---

## Interaction with JAWP: opposite optima, one parameter

> **Status: OPEN — owner science call.** TASK-40 established the facts and
> measured the magnitude. It deliberately changed no loss, no weight and no
> `lambda_wsd`. Section added 2026-10-08.

### The coupling is real, not a documentation artefact

`src/models/jepa.py:1086` hands WSD a **live view** of the JAWP parameter:

```python
Q_ws = self.jawp.workspace_Q[:, :k_active]   # no .data()
return self.wsd.compute_drift(Q_ws, h_target=h_target.detach(), step=current_step)
```

`h_target` is detached, but `Q_ws` is not, and `jepa.py:862` adds
`config.lambda_wsd * loss_wsd` to `total_loss`. So `d L_WSD / d Q` is a live
gradient on the same tensor, same columns (`[:, :k_active]`), in the same
optimizer step as `d L_JAWP / d Q`. This is **not** a false alarm of the
"different quantities at different times" kind — both terms are in one
`total_loss.backward()`.

### The two gradients

| | loss w.r.t. $Q$ | $\partial L / \partial Q$ | optimum |
|---|---|---|---|
| JAWP | $\|Q^\top(z_\mathrm{pred}-z_\mathrm{target})\|^2 = \mathrm{tr}(Q^\top\Sigma_\mathrm{res}Q)$ | $2\Sigma_\mathrm{res}Q$ | **bottom**-$k$ of $\Sigma_\mathrm{res}$ |
| WSD | $2k - 2\lVert Q^\top Q_\mathrm{tgt}\rVert_F^2$ (`wsd.py:258`) | $-4\lambda\,P_\mathrm{tgt}Q$ | **top**-$k$ of $\mathrm{Cov}(h_\mathrm{target})$ |

$\Sigma_\mathrm{res}$ and $\mathrm{Cov}(h_\mathrm{target})$ are different
matrices, so the two optima need not be distinct subspaces — but nothing in
either proof states a relation between them, and `proofs/jawp.md` explicitly
names high-variance selection as the failure mode to avoid (see its header
"DESIGN DECISION 3", where a $\beta\lVert(I-QQ^\top)z_\mathrm{target}\rVert^2$
term was **removed** for pulling $Q$ toward high-variance directions). WSD's
$\partial L/\partial Q = -4\lambda P_\mathrm{tgt}Q$ is that same pull, in the
same units, on the same parameter.

### The optima are data-dependent, and both extremes occur

Measured as $\text{overlap}^2(\text{bottom-}k(\Sigma_\mathrm{res}),\ \text{top-}k(\mathrm{Cov}_\mathrm{tgt}))$, $D=8$, $k=2$:

| constructed $\Sigma_\mathrm{res}$ | overlap$^2$ | reading |
|---|---|---|
| $\mathrm{Cov}_\mathrm{tgt} + 4I$ (isotropic predictor error) | **0.0000** | exact opposition |
| reversed spectrum (variance $\Rightarrow$ unpredictability) | 1.0000 | exact agreement |
| predictable low-variance directions | **0.0000** | exact opposition |
| $\Sigma_\mathrm{res}=I$ (no structure) | 1.0000 | degenerate — any subspace minimises |

Isotropic predictor error is the case the *proofs* implicitly assume and it is
the case that gives **exactly zero** overlap. The two mechanisms agree only in
the regimes the theory argues against.

### In situ, on the real model

`TextSpanJEPA`, `D=64`, $k=6$, `jawk_init=random`, curriculum off, one
forward/backward per step, isolating $\partial L_\mathrm{WSD}/\partial Q$ by
differencing the `workspace_Q.grad` with WSD on against off at the same seed:

```
cos( dL_JAWP/dQ , dL_WSD/dQ )   and   ||dL_WSD/dQ|| / ||dL_JAWP/dQ||
  step=0   cos=-0.5221   ratio=3.1187
  step=1   cos=-0.4999   ratio=2.5273
  step=2   cos=-0.3038   ratio=2.9813
  step=3   cos=-0.5722   ratio=2.2670
  step=4   cos=-0.4715   ratio=2.2245
  step=5   cos=-0.4835   ratio=2.3973
  step=6   cos=-0.3843   ratio=2.5417
  step=7   cos=-0.3588   ratio=2.2291
  step=8   cos=-0.5933   ratio=1.8862
  step=9   cos=-0.5337   ratio=2.1856
  step=10  cos=-0.4744   ratio=2.0272
  step=11  cos=-0.5056   ratio=2.4764
  -> 12/12 steps NEGATIVE;  median ratio 2.3973  =>  crossover lambda_wsd ~ 0.417
```

**Every measured step is opposed**, and WSD's raw gradient on $Q$ is ~2.4x
JAWP's. The **crossover is $\lambda_\mathrm{WSD} \approx 0.42$**: above that,
WSD dominates the $Q$ update.

### Effect of `lambda_wsd` on where the workspace actually lands

60-step runs, `wsd_sync_interval=1`, everything else equal. `ov_top` is
$\text{overlap}^2(\mathrm{span}(Q),\ \text{top-}k\ \mathrm{EMA\ Cov}(h_\mathrm{target}))$;
chance is $k/D = 0.094$.

| `lambda_wsd` | `ov_top` | `jawk_pca_alignment` | `jawk_predictive_relevance` | total loss |
|---|---|---|---|---|
| 0.0 | 0.0494 | 0.0633 | 0.3251 | 1.151 |
| 0.001 | 0.0497 | 0.0634 | 0.3249 | 1.162 |
| 0.01 | 0.0525 | 0.0637 | 0.3237 | 1.266 |
| 0.1 | 0.0913 | 0.0703 | 0.2991 | 2.267 |
| 1.0 | 0.3705 | 0.1837 | **0.0000** | 9.095 |
| 10.0 | 0.3866 | 0.2005 | **0.0000** | 75.227 |

`span(Q)` is measurably dragged toward the top-variance target subspace, and at
$\lambda \ge 1$ JAWP's own predictive-relevance diagnostic collapses to zero —
the workspace stops being predictable at all. The conflict is not academic.

### Which weights are actually shipped

- `defaults.yaml:241` `lambda_wsd: 0.01`, `defaults.yaml:237` `use_wsd: true`.
- **Every** runnable config sets `use_wsd: false`: `config/scaling/*.yaml`,
  `config/wikitext/*`, `config/tinystories/*`, `config/kaggle/*`,
  `config/ablations/{none,sigreg_only,no_wsd}.yaml`.
- The only config that enables it is `config/ablations/wsd_on.yaml`
  (`use_wsd: true`, `lambda_wsd: 0.1`) — 4x below the measured crossover.

So the antagonism is **live but sub-threshold in the grid**, and **latent in
every headline config**. That is the accurate severity, and it is why this was
never caught by an ablation sweep: no shipped arm sits near $\lambda \approx
0.42$.

### Second, independent defect on the same seam

`MechanismBundle.forward` (`mechanisms.py:475`) passes
`self.jawp.workspace_Q.data[:, :k_active]` — **detached**. Verified:

```
MechanismBundle info['wsd_loss'].requires_grad = False
MechanismBundle info['jawp_loss'].requires_grad = True
```

`MechanismBundle` is constructed nowhere in `src/` outside `mechanisms.py`, so
`TextSpanJEPA` is the live path and `mechanisms.py` is a divergent copy of the
same wiring. Anyone using the documented bundle API gets a WSD loss that
cannot train anything, with no warning. `mechanisms.py` is outside TASK-40's
file boundary; recorded here and in `.agent-notes/task-40.md` for the owner.

### Options, with consequences

None of these was applied. Listed so the owner can choose.

1. **Keep both as they are.** Zero code change. The grid never crosses
   $\lambda \approx 0.42$, so nothing observable breaks today. Cost: the trap
   stays armed — a sweep to $\lambda \in \{1, 10\}$ silently replaces JAWP's
   workspace with the target's top-variance subspace, and
   `jawk_predictive_relevance` reads 0.0 rather than raising.

2. **Make WSD agree by construction** — take $Q_\mathrm{tgt}$ from the
   *bottom*-$k$ of the target covariance, or from $\Sigma_\mathrm{res}$
   directly. Then both mechanisms minimise the same trace and the opposition
   vanishes. Cost: WSD stops being an *independent* synchronisation monitor —
   it becomes a second copy of JAWP's objective, and the drift it reports is
   no longer "has the target encoder moved?" but "does Q minimise residual?".
   The Davis-Kahan/STA section above and the top-$k$ definition in "Practical
   Computation" would both need rewriting. This is the option that makes the
   drift bound vacuous.

3. **Detach Q in WSD and keep it a pure monitor.** One line in `jepa.py:1086`.
   Cost: the Drift Bound theorem becomes false **by construction** — the
   penalty is precisely what supplies $\lambda$ in
   $d\Delta/dt = \nu - \lambda\Delta$; with no gradient there is no restoring
   force and $\Delta$ is unbounded. Worse, `tests/test_sterility.py` would
   still pass: `test_wsd_component_nonzero` only asserts the logged number is
   `> 0`, and a gradient-free term still logs a number. This option silently
   disarms the sterility gate.

4. **Give WSD its own projection** and leave `workspace_Q` to JAWP. Clean
   separation, both mechanisms keep their meaning. Cost: a new tensor to
   checkpoint (and `train.py:277-289` reconstructs the `workspace_Q`-family
   buffers by explicit name, so a new one needs a matching entry), a new
   consistency loss to invent, and two notions of "the workspace" — while
   SWIP, RDC and WSR all read `jawp.workspace_Q`. Largest blast radius.

5. **Guard it, change nothing.** Record the measured crossover here, cap
   `lambda_wsd` in `defaults.yaml` below it with a comment citing this
   section, and add a regression test that fails when a config's
   `lambda_wsd` crosses the measured threshold. Cost: a new test file plus a
   `defaults.yaml`/`config/**` edit, both outside TASK-40's boundary. Makes
   the trap explicit without pre-empting the science call.

**Not recommended by TASK-40, for the record:** option 3. It looks like the
cheap fix and it is the only one that turns a measured conflict into an
undetectable no-op.



