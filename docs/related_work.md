# Related work

Scope: this document maps what the code in `src/models/` actually computes onto the
established literature, and marks each item **NOVEL**, **RECOMBINATION** or
**UNCLEAR**. It is written to audit standards, because the repo's own history is
one of plausible-looking numbers that did not survive contact with the code (25
audit cards). Verdicts below cite the executed line where the distinction matters.

**Headline result: not one of the twelve mechanism modules is defensible as novel
in isolation.** Eight are recombinations whose parts are all published and
individually unremarkable; three are recombinations of a published technique with
a real defect or a dead gradient path attached; one (Spectral Gap) is not reachable
from training at all. The defensible contribution, if any, is the *coupling* — a
single learned subspace `Q` on `St(D,k)`, maintained by retraction, that the
prediction loss is restricted to and that eight other regularizers read as their
shared coordinate system. Whether that coupling is a contribution or an
incremental synthesis of reviewed techniques is exactly the kind of call that
belongs to the owner, not to a related-work section. See
[What is left](#what-is-left).

---

## 1. The real counts

The repo advertises 16 mechanisms in three places that disagree with the code.

| Quantity | Value | Source |
|---|---|---|
| Numbered capabilities in the GWP header | 16 | `mechanisms.py:37-56` |
| `GWP.N_MECHANISMS` | 16 | `mechanisms.py:682` |
| `GWP.N_GROUPS` | 3 | `mechanisms.py:683` |
| Modules in `ALL_MECHANISMS` | **12** | `mechanisms.py:100-113` |
| Modules by group | core 1, routing 4, stability 7 | `MechanismBundle.GROUPS` |
| Proof documents | 13 (12 modules + `wip.md`) | `proofs/` |
| Audited matrix rows | 12 | `proofs/IMPLEMENTATION_STATUS.md` |
| Modules whose loss reaches the total loss | see §3 | `jepa.py:851-869` |

`N_MECHANISMS = 16` counts **numbered capabilities**, of which four are methods of
`JAWPModule` rather than modules: WIP (`workspace_information_preservation`),
Spectral Gap (`detect_workspace_dimension`), Grassmann Optimization
(`grassmann_retract` / `principal_angles` / `subspace_distance`) and Predictive
Rank (`predictive_rank_loss`). `ALL_MECHANISMS` is the twelve that are trained,
toggled and ablated. Both numbers are defensible; quoting either unqualified is
not. `proofs/README.md` states this convention and `README.md` restates it.

Two of the four non-module capabilities are **not reachable from a training run**:

- `detect_workspace_dimension` has no caller in `src/` (only `tests/test_v025_integration.py`).
  The workspace width actually used is `JAWPModule.current_k`, a cosine curriculum
  from `k_start` to `k_end`. So the Marchenko-Pastur "automatic k*" is a
  measurement tool, not part of the method.
- `grassmann_retract`, `principal_angles`, `subspace_distance` and
  `save_workspace_snapshot` have no caller in `src/`. The trained path calls
  `stiefel_retract` (`train.py:1470-1484`).

`predictive_rank_loss` *is* a real trained term (`lambda_predictive_rank`,
default `0.0`, `jepa.py:858`) but has no `ALL_MECHANISMS` entry, so it is invisible
to `active_mechanisms()` and `GWP.summary()`, which print `Core: ['jawp']`.

---

## 2. The mechanism table

| # | Mechanism | What the code computes (executed form) | Closest prior work | Verdict |
|---|---|---|---|---|
| 1 | **JAWP** | `‖Qᵀz_pred − Qᵀz_target‖² + α‖(I−QQᵀ)z_pred‖²`, `Q ∈ St(D,k)`, SVD retraction | Reduced-rank regression; retraction optimization (Absil, Edelman & Lewen 2008) | RECOMBINATION |
| 2 | WIP | Fraction of feature energy inside `span(Q)`; features default to top-k PCA of `z_target` | Principal angles / canonical correlations; CCA | RECOMBINATION — and circular as a "guarantee" |
| 3 | Spectral Gap | Rank from largest eigenvalue ratio ∩ Marchenko-Pastur lower bound | Marchenko-Pastur law; Gavish-Donoho optimal hard threshold | RECOMBINATION — dead in training |
| 4 | Grassmann Optimization | `grassmann_retract` (horizontal projection + SVD), principal angles, chordal distance | Edelman, Arias & Smith 1998; Björck-Golub principal angles | RECOMBINATION — textbook, unused |
| 5 | Predictive Rank | `−log det(Qᵀ Cov(z_pred) Q + εI)`; effective rank via entropy of singular values | Log-determinant barrier (Vogelstein); DPP diversity; Vershynin effective rank | RECOMBINATION |
| 6 | **CGN** | Gumbel-Softmax Bernoulli per group, mask-interpolated, masked pathway = `1 − visible` | Gumbel-Softmax (Jang et al. 2017); Concrete/L0 (Maddox et al. 2017); MoE gating | RECOMBINATION |
| 7 | **SWIP** | Background log-eigenvalue matching to `σ²` + workspace ordering ReLU hinge | ZCA/PCA whitening (Huang et al. 2019); Barlow Twins; VICReg variance/covariance | RECOMBINATION |
| 8 | PCR | Gated MLP cascade on learned orthogonal residual subspaces | Iterative refinement (I-JEPA, DDNM); Deep Equilibrium Models; multigrid | RECOMBINATION |
| 9 | SPC | Floored-softmax-weighted sum of per-band residual variances, DCT-initialised learned basis, EMA band adaptation | DFT-domain loss weighting; uncertainty weighting (Kendall et al. 2018); GradNorm | RECOMBINATION |
| 10 | WSD | Squared chordal Grassmann distance between `Q` and top-k eigenvectors of an EMA target covariance | EMA target networks (BYOL, I-JEPA); online PCA / Oja subspace tracking | RECOMBINATION — with an unresolved contradiction |
| 11 | CMC | `‖z_pred¹[t] − z_pred²[t]‖²` at positions masked in both masks, stop-grad on the primary | Consistency regularization (Mean Teacher, FixMatch/UDA); BYOL two-view prediction | RECOMBINATION |
| 12 | GAC | `γ·relu(τ_grad − ‖g_i‖)·mean(z_i²)` on starved coordinates | Gradient-magnitude exploration bonuses (RL/ES); dormant-unit reactivation | RECOMBINATION |
| 13 | STA | `η·W₁(sorted λ_current, sorted λ_EMA)` on covariance spectra | Spectral stability penalties; continuum-learning spectral anchoring | RECOMBINATION — with a design risk |
| 14 | PUC | Oja power iteration + ReLU'd `−log det` barrier, gated on an entropy deficit | VICReg variance term; Barlow Twins; log-det diversity | RECOMBINATION — **dead gradient path as shipped** |
| 15 | RDC | `η·mean‖(I−QQᵀ)(z_t − z_{t−1})‖²` | Continual-learning drift penalties; representation-stability regularization | RECOMBINATION |
| 16 | WSR | `η·ρ·‖(I−QQᵀ)G‖_F/‖Q‖_F` (gradient mode), or a retracted SAM perturbation (sam mode) | SAM (Foret et al. 2021); manifold-SAM variants | RECOMBINATION — and see §5 |

Bolded rows (JAWP, SWIP) are the two the framework's framing rests on; see
[What is left](#what-is-left).

---

## 3. Per-mechanism notes

### 1. JAWP — reduced-rank regression on a learned manifold

The executed loss (`jawp.py:372-386`) is a projection-restricted MSE plus a
predictor-focus penalty with `Q` detached in the second term. The Courant-Fischer
argument in the module header (`jawp.py:45-65`) is the standard characterization
of the minimizer of `tr(QᵀΣ_res Q)` on `St(D,k)`: the bottom-*k* eigenvectors of
the residual covariance. **That theorem is textbook (Courant–Fischer / Ky Fan) and
cannot be claimed as this repo's contribution.** The gradient path is what is
actually new-ish: `Q` receives gradients from both sides of the MSE, which is what
makes the minimizer statement reachable in practice, and the curriculum on `k`
grows the workspace. None of that is published as a JEPA mechanism that I can
verify, and none of it is a new mathematical object. **RECOMBINATION** —
reduced-rank regression (Bock; Hastie–Tibshirani–Friedman) + retraction-based
manifold optimization, specialized to a latent prediction loss.

The theorem is also stated for fixed `k`, while `current_k(step)` varies `k`
across training. That mismatch is recorded in the audit matrix and is not
resolved here.

### 2. WIP — a diagnostic that measures the thing it claims to guarantee

`workspace_information_preservation` computes
`mean_i ‖Qᵀf_i‖²/‖f_i‖²` (`jawp.py:745-753`). When called without `features`, `f`
is **the top-k PCA directions of `z_target`** (`jawp.py:728-737`) — a variance-
ordered basis of the target, used as a stand-in for "exogenous features". A
score that projects a variance basis onto a subspace and reports what fraction
survives is not a guarantee of information preservation; it is a subspace
overlap number whose inputs were chosen by the same variance criterion the module
elsewhere argues against. No `use_wip` key exists in any config; the only caller
is `MechanismBundle.compute_capacity_bound`, a composite diagnostic whose own
source comment calls it `COMPOSITE PROXY` (`mechanisms.py:538`). **RECOMBINATION**
— principal-angle/subspace-similarity machinery, presented as a theorem.

### 3. Spectral Gap — standard random-matrix rank selection, not in the method

`detect_workspace_dimension` combines a largest-relative-gap heuristic with a
Marchenko-Pastur lower bound (`jawp.py:606-640`). Marchenko-Pastur eigenvalue
detection and eigenvalue-gap rank heuristics are long-established (Gavish &
Donoho's optimal hard threshold; Minka). Two honest observations: `c_est` is
hardcoded to `0.5` with the comment "conservative estimate", so the MP bound is
computed at a guessed aspect ratio; and the method has no caller in `src/`.
**RECOMBINATION**, and describing it as "NO MANUAL TUNING NEEDED" in a methods
section would be unsupportable while `k` is set by a curriculum.

### 4. Grassmann Optimization — correct Grassmann machinery, unused

`grassmann_retract` projects the gradient to `(I − QQᵀ)G` and then applies the
same SVD retraction as the Stiefel path (`jawp.py:920-951`). `principal_angles`
via `svdvals(Q₁ᵀQ₂)` and chordal distance `√Σ sin²θ` (`jawp.py:1005-1018`) are
standard. Everything here follows Edelman, Arias & Smith (1998) and Absil,
Edelman & Lewen (2008). Nothing calls it during training. **RECOMBINATION**,
and any related-work claim that the *method* optimizes on `Gr(k,D)` is
contradicted by `train.py`, which retracts on `St(D,k)`.

### 5. Predictive Rank — log-det barrier

`predictive_rank_loss` returns `−Σ log(λ_i + ε)` over the workspace covariance
(`jawp.py:1196-1206`), which maximization-under-minimization keeps the workspace
full-rank. This is the standard determinant barrier; the effective-rank readout is
Vershynin's entropy definition applied to `svdvals` of a covariance rather than of
a matrix. **RECOMBINATION.** It is a genuine trained term but ships with
`lambda_predictive_rank: 0.0`.

### 6. CGN — Gumbel-Softmax gating, with the orthogonality term deleted

Gumbel-Softmax (Jang, Gu & Dan 2017) with cosine temperature annealing, and a
mask-conditioned Bernoulli per group. The distinguishing choice is that the masked
pathway is the **complement** of the visible one by construction:
`gate_on_masked = 1.0 - gate_on_visible` (`cgn.py:263-264`), so
`g_visible + g_masked = 1` holds identically rather than being penalized. The
header's proof (`cgn.py:48-83`) is the chain rule of mutual information plus
non-negativity of conditional mutual information — correct and unremarkable.

Two honest notes. The config weight `lambda_cgn_ortho: 0.01` (`defaults.yaml:209`)
is wired to a function that returns a zero tensor (`jepa.py:1043-1048`), because
complementary gating makes the legacy orthogonality penalty structurally
unnecessary. And `compute_orthogonality_score` was redefined as
`2·mean|P_ON − 0.5|` (`cgn.py:336-346`) because the original quantity is trivially
1.0 under complementarity. **RECOMBINATION** — Gumbel/Concrete gating applied to
JEPA predictor routing.

### 7. SWIP — selective whitening, the closest thing to a differentiated idea

The intent is a two-regime spectral treatment: leave the workspace spectrum alone,
push the background spectrum to `σ²`. Both paths are implemented
(`swip.py:199-319`): with a JAWP `Q` the background term is a *trace-based Jensen
approximation*, `(D−k)·(log(bg_trace/(D−k)) − log σ²)²`, and without `Q` it is the
per-eigenvalue sum. The workspace-ordering hinge (proof term 2) was missing and
was added in audit round 15.

Nearest prior art: whitening transforms for batch normalization (Huang, Hsieh &
Krieger 2019 / Ma et al.), and in representation learning, shaping the covariance
spectrum toward a target is what Barlow Twins (cross-correlation → identity) and
VICReg (variance + covariance terms) already do globally. The *selective* split
— applying the constraint only outside a learned subspace — is the part with no
direct cite I can point to, and it is genuinely a different constraint from
anything in the three papers above, which have no notion of a protected subspace.

One claim in the header must not survive: `swip.py:81-85` argues scale
invariance of the log-eigenvalue loss. The module's own docstring retracts it
(`swip.py:154-156`: "scale-INVARIANCE DOES NOT HOLD as implemented"). **RECOMBINATION**
with one differentiating constraint choice; the scale-invariance and
"ONLY method that…" (`swip.py:110`) claims should be cut.

### 8. PCR — iterative refinement on orthogonal subspaces

`pcr.py:359-376` runs `n_levels` gated MLP corrections, each projecting the
residual onto a learned orthogonal block. Two observations. First, the predictor
already performs `num_refine_steps` iterative passes on the *same* subspace
(`predictor.py:57-60`), so PCR is a second refinement stack with a subspace
twist. Second, the header's Cayley-retraction description (`pcr.py:105-108`) does
not match the code: `stiefel_retract` is SVD, not Cayley (`pcr.py:294-309`).
The audit matrix already records that the proof analyzes a linear Stiefel cascade
while the code is a gated MLP cascade. **RECOMBINATION** — iterative refinement /
deep-equilibrium machinery with learned orthogonal structure.

### 9. SPC — frequency-weighted loss with learned band weights

DCT-II-initialised learned orthogonal basis, `n_bands` equal blocks, simplex
weights via `softmax(log_w)·B` floored at `min_weight` (`spc.py:289-301`), plus an
EMA adaptation step toward `residual_var × predictability` called from
`train.py:1490`. Weighting an L2 loss by spectral bands is classical signal
practice; weighting losses by learned per-component confidences is a reviewed
family (uncertainty weighting, Kendall et al. 2018; GradNorm, Chen et al. 2018).
The learned-basis variant adds a genuine data-adaptive choice, but the executed
object is a floored-softmax-weighted sum of band variances, not the
information-proportional object the theorem describes — the audit matrix records
that divergence. **RECOMBINATION.**

### 10. WSD — and a contradiction worth surfacing

WSD penalizes the squared chordal distance between `Q` and `Q_target`, where
`Q_target` is the **top-k eigenvectors of an EMA of the target covariance**
(`wsd.py:196-215`). The loss itself is
`2k − 2‖Q₁ᵀQ₂‖²_F`, a standard chordal/Grassmann distance.

The contradiction: JAWP's own stated criterion selects the **bottom-k**
eigenvectors of the *residual* covariance — the most predictable directions — and
its header explicitly argues that selecting high-variance directions is the failure
mode to avoid (`jawp.py:100-111`: the removed `β` term "pushes Q toward
high-VARIANCE directions, CONFLICTING with workspace prediction loss"). WSD pulls
`Q` toward the target's **highest-variance** directions. The two mechanisms are
pulling the same parameter toward different optima, and nothing in the repo
reconciles them. The audit matrix notes WSD's assumptions are unenforced and its
Davis-Kahan reduction is circular; the variance-versus-residual conflict is a
separate, unrecorded issue. **RECOMBINATION**, flagged **UNCLEAR — needs the
owner's judgement** on whether WSD should track a residual-covariance subspace,
in which case it duplicates part of WSD/STA's job, or a variance subspace, in which
case it fights JAWP.

### 11. CMC — consistency regularization over two masks

Mean squared difference of predictions at positions masked in both masks, with
stop-gradient on the primary (`cmc.py:460-476`), computed on an interval
cadence with a second forward pass. The construction is textbook consistency
regularization (Mean Teacher, UDA, FixMatch) transplanted onto per-position latent
predictions, with the overlap restriction as the JEPA-specific detail. The audit
matrix records that the stability theorem is proven in averaged form, not
pointwise, and that `mode="reuse_encoder"` is a stub. **RECOMBINATION.**

### 12. GAC — gradient-starvation exploration bonus

`gac.py:200`: `γ·relu(τ − ‖g_i‖)·mean_i(z_i²)`, applied to coordinates whose
per-dimension gradient norm falls below `τ`. The per-dimension gradient
exploration bonus has clear relatives in reinforcement-learning exploration and in
dormant-unit reactivation work. Note the loss is added after `backward()` in
`train.py:1410-1413` on a *live* `span_preds` slot, so it does carry a gradient —
this one is not dead. But the theorem's No-Dead-Zones bound is stated for a form
the batch-mean, EMA-smoothed implementation does not compute; the module's own
`compute_gradient_bound` is labelled `PROXY ESTIMATE` (`gac.py:225-228`).
**RECOMBINATION.**

### 13. STA — and a design risk

`W₁` between the sorted current-batch spectrum and an EMA reference, scaled by `η`
(`sta.py:297-316`). The Kantorovich-Rubinstein argument for sorted discrete
supports is correct and standard. The spectral-stability regularizer as a genre is
established (continual-learning anchoring, spectral clipping).

The risk, stated plainly: this term penalizes *any* change in the eigenvalue
spectrum relative to its own recent history, including the change that learning
causes. With `lambda_sta: 0.01` and `eta: 0.01` the magnitude is small, but the
direction of the objective is toward a frozen representation. A reviewer will ask
whether this helps or just slows learning; the honest answer from the code is that
the repo does not know, and neither can be asserted without an ablation. **RECOMBINITION**,
with the caveat that the term's sign convention may be doing the opposite of what
the header claims.

### 14. PUC — the executed loss cannot train

The module docstring already flags this (`puc.py:26-29`): the executed loss is a
ReLU'd log-det barrier over Oja-tracked eigenvalues, gated on an entropy deficit,
not the Lagrangian-dual object in the theorem.

I verified the consequence end to end. With the shipped defaults,
`use_differentiable_entropy=False` (`puc.py:79`) — and that flag is **not exposed**
through `TextSpanJEPAConfig`, so `jepa.py` never passes it — the eigenvalue path is
`F.softplus(self.running_eigenvalues - 5.0)`, which reads an EMA buffer with no
autograd connection to `z_pred`. The returned tensor therefore has
`requires_grad=False`. Measured:

```
use_differentiable_entropy=False  loss 0.1401  requires_grad False
use_differentiable_entropy=True   loss 0.1037  requires_grad True   z.grad norm 1.4e-05
```

So as wired, `PUC` contributes a constant to `total_loss` and cannot influence
training. Every regularization is gated in `TextSpanJEPA.validate()` and every
config key carries a non-zero weight, so the wiring looks live; the gradient is
what is missing. This should be stated in a paper rather than discovered by a
reviewer. **RECOMBINATION, dead gradient path as shipped.**

### 15. RDC — drift penalty decomposed by the workspace

`η·mean‖(I−QQᵀ)(z_t − z_{t−1})‖²` (`rdc.py:179-192`). Penalizing representation
drift, and specifically slowing the components outside a protected subspace, is
established continual-learning practice (representation-stability penalties, drift
correction). The workspace decomposition is the twist; the theorem's transient
bound is internally inconsistent per the audit matrix, and the reported bound is a
`PROXY ESTIMATE` on EMA-smoothed inputs (`rdc.py:204-205`). **RECOMBINATION.**

### 16. WSR — SAM transplanted, with the SAM path broken

Gradient mode computes `η·ρ·‖(I−QQᵀ)G‖_F/‖Q‖_F` (`wsr.py:394-434`). This is a
first-order sharpness proxy, which is the standard SAM approximation (Foret et
al. 2021). Transplanting SAM to a manifold-valued parameter is a natural
extension and is described as such in manifold-optimization work.

Two honesty items. First, minimizing `‖grad_Gr L‖` is not the same objective as
minimizing sharpness: it asks for `Q` to be a stationary point of the workspace
loss, not for a flat neighbourhood. The gradient-mode term is a pull toward
stationarity wearing SAM's name. Second, `mode="sam"` does not work — see §5.

**RECOMBINATION** (SAM) whose executed default is a different objective from its
documented one.

---

## 4. Framework-level positioning

| Prior work | What this repo borrows | What is genuinely different |
|---|---|---|
| **I-JEPA** (Assran et al., ICCV 2023) | EMA target encoder, target-only latent loss, no negatives, no pixel reconstruction, context/target block structure, iterative predictor (`predictor.py:14-34`), span masking adapted from I-JEPA multiblock + SpanBERT (`masks/span.py:3-4`) | The prediction is restricted to a learned subspace (§3.1). Also: this repo has a **token decoder** with `cross_entropy` at `lambda_decoder: 0.1` (`jepa.py:762-768`), so unlike I-JEPA it does reconstruct tokens. Any "purely latent, never predicts pixels/tokens" claim is false here. |
| **VICReg** (Bardes, Ponce & LeCun, ICLR 2022) | Variance and covariance terms implemented essentially verbatim (`collapse.py:108-154`), target centering (`collapse.py:157-176`) | That collapse is treated as a *solved* problem (`lambda_variance 0.1`, `lambda_covariance 0.04`) and the twelve mechanisms are spent elsewhere. VICReg's framing — an explicit stated regularizer against collapse — is the house style, not a contrast. |
| **Barlow Twins** (Zbontar et al., ICML 2021) | Redundancy reduction by shaping a spectrum/cross-correlation toward a target | SWIP shapes only the background half of the spectrum (§3.7). Not the same constraint. |
| **BYOL** (Grill et al., NeurIPS 2020) | EMA target + predictor + stop-gradient, no negatives, no batch norm | Nothing structural. The repo's non-collapse strategy is VICReg + SIGReg + target centering, not BYOL's implicit bias. |
| **SimSiam** (Chen & He, CVPR 2021) | Stop-gradient on the primary prediction in CMC (`cmc.py:460-463`) | Nothing structural. |
| **SigLIP** (Zhai et al., 2023) | **Nothing.** There is no sigmoid pairwise loss, no logit scale, no negative set anywhere in `src/models/` or `src/train.py`. | This repo is on the no-negatives side entirely. There is no sigmoid-loss variant to compare against, so no claim of that kind can be made in either direction. |
| **LeJEPA / SIGReg** (Balestriero & LeCun) | `SIGReg` implemented (`sigreg.py:33-121`); `WeakSIGReg`; `VISReg` | Shipped **off**: `lambda_sigreg: 0.0` in `defaults.yaml`. `VISReg` is an undocumented 2026 construction with no cited source — do not cite it. |
| **data2vec 2.0** (Baevski et al.) | Target centering + layer-norm on the target (`jepa.py:671-673`), tied token regression head (`decoder.py:1-6`) | The decoder is the one place this repo predicts in token space. |
| **SAM** (Foret et al., ICLR 2021) | WSR's sharpness notion (§3.16) | Manifold-valued parameter instead of a flat parameter vector. |
| **Absil, Edelman & Lewen (2008)**; **Edelman, Arias & Smith (1998)** | SVD retraction, tangent projection, principal angles, chordal Grassmann distance — used correctly in JAWP/PCR/SPC | Nothing. These are cited correctly in the source. See §5 for the one place a retraction is *not* correct. |

---

## 5. Retractions: one correct, one wrong, and three sign-ambiguous

The retraction question is not cosmetic. A retraction must satisfy
`R_Q(0) = Q` and approximate the identity locally; a sign-ambiguous QR does not.

**Correct.** `JAWPModule.stiefel_retract` computes `U, S, Vᵀ = svd(Q)` and sets
`Q ← U[:, :k] @ Vᵀ[:k, :]` (`jawp.py:306-309`). That is the polar decomposition, the
nearest orthonormal matrix in Frobenius norm, and it is the retraction Absil,
Edelman & Lewen describe. The tangent projection
`G ← G − Q sym(QᵀG)` (`jawp.py:300-304`) is the standard Stiefel correction, and
`train.py:1433` now calls `project_tangent_gradient()` *before* the optimizer step
so it is not a dead post-step write.

**Wrong, and flagged in-tree.** `WorkspaceSharpnessRegularization._stiefel_retract`
(`wsr.py:242-247`) takes the QR factor and then applies "canonical" sign
normalization read from `diag(Q_retracted[:k, :])`. For a `(D, k)` matrix with
`D > k` that leading block is **not triangular**, so the signs are arbitrary and
the procedure flips individual columns instead of nudging the subspace by `ρ`. The
module's own docstring records the measurement — retracted `Q` sits ~4.0 away
from `Q` in Frobenius norm where `ρ = 0.05` should put it ~0.05 — and
`tests/test_training_state_guards.py` pins it as `xfail`
(`test_retraction_preserves_column_orientation`). A retraction with wrong sign
selection is not a retraction; the defect is real, measured, and open pending an
owner decision. **`wsr_mode: sam` therefore cannot be presented as a working
SAM-on-the-Grassmann contribution.** In the same run `mode="sam"` is also
degenerate for a second reason: `L_perturbed` and `L_current` are both computed
under `torch.no_grad()` (`wsr.py:500-528`), so the returned scalar carries no
gradient regardless. Measured: `wsr sam-mode requires_grad: False, sharpness
0.0`. The shipped default is `wsr_mode: gradient`.

**Sign-ambiguous, low severity.** The QR fallbacks in `jawp.py:312`, `pcr.py:307`
and `spc.py:284` call `torch.linalg.qr(Q)` and keep `Q` without normalizing `R`'s
diagonal. LAPACK's sign convention is arbitrary, so a column can flip. These are
`except` branches reached only when the SVD fails, and a sign flip still lands on
`St(D,k)`, so the manifold constraint holds; but these are not valid retractions
and should not be described as such. The `init` paths that build a starting
`Q` from a random Gaussian (`jawp.py:208`, `pcr.py:289`, `spc.py:266`) are fine —
initialization has no continuity requirement.

---

## 6. Claims this document refuses to make

Collected so a later draft cannot reintroduce them by accident.

1. **No mechanism is novel.** Not JAWP, not SWIP, not CGN, not any of the twelve.
   Every one is a recombination of published techniques. This is a statement about
   prior-art coverage, not a verdict on whether the work is publishable.
2. **The Courant-Fischer optimality result is textbook** and cannot be claimed.
   The repo's contribution is the gradient plumbing and the curriculum, not the
   theorem.
3. **The WIP theorem is not a guarantee.** Its "exogenous features" default input
   is the target's own top-k PCA basis, so the score is subspace overlap with a
   variance-chosen reference, not evidence of information preservation.
4. **"Automatic k* / no manual tuning" is not supported.** `k` comes from a cosine
   curriculum; `detect_workspace_dimension` has no caller, and it hardcodes
   `c_est = 0.5`.
5. **The method does not optimize on the Grassmannian.** `train.py` retracts on
   `St(D,k)`; every `Gr(k,D)` method is uncalled.
6. **SWIP is not scale-invariant.** Its own docstring retracts the claim.
7. **"The only method that preserves the workspace hierarchy while whitening the
   background"** (`swip.py:110`) is an unsupported universal claim over a
   literature I have not exhaustively surveyed. Soften to "to our knowledge" or
   cut.
8. **`wsr_mode: sam` cannot be presented as a working mechanism.** The retraction
   sign selection is wrong (measured, `xfail`-pinned) and the loss carries no
   gradient.
9. **PUC cannot be presented as an active regularizer.** With the shipped
   defaults its loss has `requires_grad=False`; it is a constant in the total.
10. **Gradient-mode WSR is not sharpness minimization.** It minimizes the
    gradient norm, which is a stationarity objective.
11. **`lambda_cgn_ortho` is a dead weight**, wired to a function that returns zero.
12. **No sigmoid-loss (SigLIP-style) baseline exists** in the repo, so no claim of
    that kind is available in either direction.
13. **`VISReg`** (`sigreg.py:164-222`) is labelled "(2026)" with no citation. Do
    not cite it.
14. **No ablation results are claimed here.** `config/ablations/` ships the
    leave-one-out grid, but this document has read the configs, not the runs.
    Nothing in this file should be read as evidence that any mechanism helps.
15. **No comparison numbers.** This is a positioning document. It asserts what the
    code is made of, not what it achieves.

---

## What is left

Stated without advocacy, because the judgement is the owner's.

Nothing in the twelve survives as a novel *mechanism*. Two candidates survive as
plausible *contributions*, and both are claims about combination rather than
about new mathematics:

- **The learned workspace as the prediction domain.** Restricting a JEPA
  prediction loss to a learned `k`-dimensional subspace on `St(D,k)`, maintained
  by an SVD retraction, with the workspace width annealed during training. Every
  component is published — reduced-rank regression, retraction optimization,
  subspace-restricted regression. What I cannot find in the literature is the
  combination *inside a latent predictor*, and that is a narrow claim.
- **Selective spectral shaping.** Applying a covariance-shaping constraint only
  outside a learned protected subspace, leaving the subspace's spectrum free
  (`swip.py`). VICReg and Barlow Twins shape the whole spectrum; I know of no
  prior work that carves out a protected subspace. Also narrow, and weakened by
  the branch that matters most: with a JAWP `Q` the background term is a
  trace-based Jensen approximation, not the per-eigenvalue matching the header
  describes.

The eight other regularizers that read `Q` as a shared coordinate system (WSD,
CMC, GAC, STA, PUC, RDC, WSR, and Predictive Rank) are the frame around those two.
Several are diagnostic rather than causal, one is dead, and one fights the
mechanism it corrects. Whether that frame reads as "a principled manifold
framework" or as "twelve plausible-looking numbers" is precisely the question the
25 previous audit cards were opened to answer — and for the mechanisms above, the
answers are as recorded: **RECOMBINATION**, with PUC additionally inert and WSR's
SAM path additionally broken.
