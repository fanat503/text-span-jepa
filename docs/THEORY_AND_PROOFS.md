# Theory and Proofs: What the Code Actually Satisfies

**This document is an audit, not an endorsement.** It exists because the repo's
own rule — repeated in every proof document's banner —
is that **`proofs/` holds design documents with audited gaps, not certified
descriptions of the running code**. "Never 'fix code to match the proof' without
an explicit human decision. The proof may be aspirational; the code may be the
deliberate choice. Escalate instead."

So this document does three things per mechanism, and nothing else:

1. **(a) the claim** — what `proofs/<mech>.md` actually asserts;
2. **(b) the assumptions** — which of the claim's hypotheses `src/models/<mech>.py`
   satisfies, marked HOLDS or FAILS with a `file:line` citation;
3. **(c) the consequence** — for a theorem whose hypotheses do not hold, exactly
   what a reader mis-believes when they trust a number the module prints.

**A theorem whose hypotheses do not hold in the implementation is a DIVERGENCE.
Naming one is the deliverable. Repairing one is not.** Every divergence below is
a statement about the *current* code at the commit named in the appendix, and
each is a decision for the owner, not a ticket for an agent.

## How to read this

- **HOLDS** — I read the code, or ran a check, and the hypothesis is satisfied.
- **FAILS** — I read the code, or ran a check, and the hypothesis is not satisfied.
- **UNVERIFIABLE** — I could not settle it. These are listed, not hidden.
- Citations are `path:line`. No source is quoted in full.

Per the repo convention: **12 modules** are audited here
(`MechanismBundle.ALL_MECHANISMS`, `src/models/mechanisms.py:100`), and the
GWP header numbers **16 numbered capabilities** — the twelve modules plus four
methods of `JAWPModule`. Quote 12 when counting modules and 16 when counting
numbered capabilities, never either unqualified. The full convention is in
`proofs/README.md`.

### What this document deliberately does not do

- It does **not** re-derive the Verified / Divergent / Gaps counts in
  `proofs/IMPLEMENTATION_STATUS.md`. Those are a dated audit artifact
  (2026-08-24) and re-counting claims is a re-audit, not a documentation edit.
  Where I disagree with a recorded **headline**, I say so explicitly and give my
  evidence; I do not invent replacement numbers.
- It does **not** propose patches. Several findings below have obvious one-line
  fixes. Those are deliberately withheld, because whether the *code* or the
  *proof* is the intended science is not a question this document can answer.
  Where a fix would be actively harmful, that is said too.

---

## 1. JAWP — Jacobian-Aligned Workspace Prediction

`proofs/jawp.md` · `src/models/jawp.py` · capability #1 (module) + #2–#5 (JAWP methods)

### (a) The claim

`L_JAWP = ‖Qᵀz_pred − Qᵀz_target‖² + α‖(I−QQᵀ)z_pred‖²` with `Q ∈ St(D,k)`.
**Theorem 1** (Courant-Fischer): the first term equals `tr(QᵀΣ_res Q)` and its
minimizer over `St(D,k)` is the **bottom-k** eigenvectors of `Σ_res`.
**Corollary**: `R(Q_JAWP) ≤ R(Q_PCA)` for any predictor.
**Theorem 2** (WIP): exogenous features survive in `span(Q)` under a regularity
condition. **Theorem 3**: SVD retraction is the nearest orthonormal matrix.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | `Q ∈ St(D,k)` — `QᵀQ = I_k` exactly | **HOLDS** — retraction at `jawp.py:331-332`; `project_tangent_gradient` at `:241-259` |
| A2 | `k` is **fixed** across the theorem's argument | **FAILS** — `current_k` varies with step (`jawp.py:340-346`); `defaults.yaml:219` sets a 10 000-step curriculum; the loss slices `Q[:, :k(t)]` at `jawp.py:387` |
| A3 | The α-term contributes to learning `Q` | **FAILS** — `Q` is detached in both factors at `jawp.py:402-404` (documented as DESIGN DECISION 4 at `jawp.py:127-131`), so `∂/∂Q` of that term is identically zero |
| A4 | The whole displayed loss is the object Theorem 1 minimises | **HOLDS for the first term** — `F.mse_loss` at `jawp.py:398` is `tr(QᵀΣ_res Q)/N` up to a positive constant |
| A5 | `Q` actually attains the `St(D,k)` minimiser | **UNVERIFIABLE** — the corollary compares *minima*, not iterates; nothing measures the gap |
| A6 | Theorem 3's retraction formula | **HOLDS** — `U[:, :k_total] @ Vh[:k_total, :]` at `jawp.py:332`, with a QR fallback at `:335` |

### (c) Divergences

**D-JAWP-1 — `predictive_relevance` is normalised by two different axes and its
meaning is inverted.** *(new; recorded nowhere in the repo)*

`ws_pred_error` is a mean over `N·k` coordinates (`jawp.py:463`);
`bg_pred_error` is a mean over `N·D` (`jawp.py:462`); the metric is
`max(0, 1 − ws/bg)` (`jawp.py:503`). Their ratio is therefore

```
pred_rel_div = (D/k) · w ,   w = workspace share of residual energy ∈ [0,1]
```

I verified `predictive_relevance == max(0, 1 − (D/k)·w)` to within 1e-4 at eight
values of `w` (D=64, k=6), and constructed both endpoints exactly, with `U`
verified orthogonal to this precise `Q`:

| constructed residual | `loss_workspace` | `predictive_relevance` |
|---|---|---|
| `w = 0` — residual **entirely outside** `span(Q)` | `1.841e-11` — prediction **exact** | **1.0000** (maximum) |
| `w = 1` — residual **entirely inside** `span(Q)` | `1.037e+00` — prediction **worst** | **0.0000** |

So the metric reads **maximal when workspace prediction is exact** (because the
residual is orthogonal to the workspace and there is nothing left to predict
there), and **zero when workspace prediction is worst**. It is inverted, and
the threshold is low: `w ≥ k/D` already reads `0.0`, which at D=64, k=6 is
`w ≥ 0.094` — chance level. A zero-residual model takes the `else` branch at
`jawp.py:504-505` and also reads 1.0 (pinned by `tests/test_host_reads.py:524`).

**Consequence for a reader:** `jawk_predictive_relevance → 0.0000` is cited at
`proofs/wsd.md:30`, `proofs/jawp.md:264`, `proofs/wsd.md:263` and `jawp.py:78` as
evidence that "the workspace stops being predictable". **That reading is not
supported by this metric.** A reading of 0 is the *normal* value for any
workspace holding at least a chance-level share of the residual — and `w = 1`,
where workspace prediction is *worst possible*, reads 0.0 as well. The metric
peaks at 1.0 precisely when `Qᵀ(z_pred − z_target) ≈ 0`, i.e. when the workspace
has nothing to predict. I am not claiming the underlying WSD/JAWP antagonism is
absent — I have not disproved it — only that this diagnostic cannot distinguish
"collapsed workspace" from "perfect workspace", and so cannot be its symptom. The
gradient-cosine measurements at `proofs/wsd.md:186-199` are independent of this
diagnostic and are not affected.

**D-JAWP-2 — fixed-k theorems over a time-varying `k`.** Known and recorded. The
consequence is that Theorem 1's minimiser statement does not describe any single
subspace the loss actually optimises: the first `k(t)` columns are being shaped
by a sequence of different problems while the remaining columns receive no
gradient at all and are held orthonormal only by the retraction.

**D-JAWP-3 — the α-term is inert with respect to `Q`.** The proof's Solution
display shows one loss with two `Q`-dependent terms. In the code the second term
has `∂/∂Q ≡ 0`, and its `.mean()` at `jawp.py:406` is over `N·D` while the first
term's is over `N·k` (`jawp.py:398`), so `jawk_alpha`'s effective weight carries
a `D/k` factor the proof does not mention.

**D-JAWP-4 — the α-term's divergence from the proof is a documented *design*
decision, not an oversight.** `jawp.py:127-131` states it: a live `Q` in that
term pushes toward high-variance directions. This is recorded here so it is not
re-raised as a defect.

### What holds up

Theorem 3 in full. Theorem 1's algebra (`jawp.py:396-398` is `tr(QᵀΣ_res Q)/N`)
on a fixed `k`. The bottom-k characterisation, and the code's own refusal to add
the `β` high-variance term (`jawp.py:123-125`).

### Could not determine

Whether the iterate `Q` is close to the `St(D,k)` minimiser (A5). Nothing in the
repo measures `tr(QᵀΣ_res Q) − min_{St(D,k)} tr(QᵀΣ_res Q)`, and reproducing it
needs the residual covariance, which no mechanism exposes.

---

## 2. WSD — Workspace-Target Synchronization Drift

`proofs/wsd.md` · `src/models/wsd.py`

### (a) The claim

**Theorem (Drift Bound).** `Δ_WSD(t) ≤ Δ(0)·e^{−λt} + ν_max/λ`, with steady-state
error `ν_max/λ`. `L_WSD = d_Gr(Q_online, Q_target)²` supplies the restoring force.
A second, "constructive" bound via STA/Davis–Kahan, and a claim that the drift
enables adaptive `τ` scheduling.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | `Q_target(t)` is the top-k PCA subspace of the target encoder, moving at rate `ν(t)` | **HOLDS only on resync steps** — `Q_target` is the top-k of an **EMA** covariance (`wsd.py:189-194`), recomputed only when `step % sync_interval == 0` (`wsd.py:261`); it is piecewise constant for 99 of every 100 steps at the default |
| A2 | `λ` is the WSD penalty strength `lambda_wsd` | **FAILS in the module's own reporting** — `wsd_lambda_estimate` is hard-coded to `1/sync_interval` (`wsd.py:322`), not read from config |
| A3 | `ν_max` is the max rate of change of the target workspace | **FAILS** — `nu_max_est` is `running_drift / sync_interval` (`wsd.py:320`), i.e. the *misalignment* divided by a time, not a rate of change |
| A4 | `d_Gr` is the sine Grassmann distance `‖sinΘ‖_F` | **FAILS** — proof §Practical Computation step 3 defines it that way (`wsd.md:120`); the code computes the **chordal** form `2k − 2‖Q₁ᵀQ₂‖²_F` (`wsd.py:274`, acknowledged at `wsd.py:53-54`) |
| A5 | `Q_JAWP` and `Q_target` are orthonormal | **UNVERIFIABLE at the call site**, but see D-WSD-3 |
| A6 | The constructive Davis–Kahan bound exists | **FAILS — not implemented at all**; no term resembling `‖Σ_online − Σ_target‖₂/δ` appears in `wsd.py` or in `jepa.py`'s WSD path |
| A7 | Adaptive `τ` scheduling is enabled | **FAILS — not implemented**; `ema_beta` is written once at `wsd.py:156` and never modified |

### (c) Divergences

**D-WSD-1 — `wsd_steady_state_error` is an algebraic identity, not a bound.**

The proof presents `ν_max/λ` as the steady-state error (`wsd.md:73`). The code
computes (`wsd.py:318-326`):

```
nu_max_est = running_drift / T          (T = sync_interval)
lambda_wsd = 1 / T
steady_state_error = nu_max_est / lambda_wsd = running_drift
```

**The `T` cancels.** I verified `wsd_steady_state_error == wsd_running_drift` to
0.0 difference across 12 steps (D=16, k=3, `sync_interval=7`). The "constructive"
steady-state error is a second name for the running-drift EMA.

**Consequence for a reader:** `wsd_steady_state_error` carries no information
that `wsd_running_drift` does not, and it does not bound anything. The two keys
should never be plotted as independent diagnostics.

**D-WSD-2 — the reported `λ` is not the loss weight, and the coincidence hides
it.** `defaults.yaml:268` is `lambda_wsd: 0.01` and `defaults.yaml:266` is
`wsd_sync_interval: 100`, so `1/sync_interval = 0.01` — the reported λ *matches*
the true loss weight at the default. Move `wsd_sync_interval` or `lambda_wsd`
and they diverge silently. `config/ablations/wsd_on.yaml` uses
`lambda_wsd: 0.1`; with the default interval the module would still report
`wsd_lambda_estimate = 0.01`, a 10× error in the one number named after λ.

**D-WSD-3 — the `clamp` turns a broken workspace into a perfect one.** Both
`wsd.py:274` and `wsd.py:281` clamp the chordal expression at 0. For two
orthonormal `(D,k)` matrices Cauchy–Schwarz caps `‖Q₁ᵀQ₂‖²_F` at `k`, so the
expression is already non-negative and the clamp is inert *in the intended
regime*. It is not inert outside it: I passed a `Q` scaled by 3 and the module
reported `wsd_drift_sq = 0.000000`, while the same subspace orthonormalised
reported `3.354395`. A workspace whose norms have blown up reads as **perfectly
synchronised**, with zero loss and zero gradient, and `wsd_overlap` /
`wsd_spectral_alignment` computed at `wsd.py:291-294` do not catch it either.

**Consequence for a reader:** `wsd_drift = 0` is ambiguous between "aligned" and
"`Q` is not a projection matrix". `wsd.py`'s own docstring says `Q_workspace` is
"(D, k) **orthonormal** workspace projection"; nothing at the call site asserts it.

**D-WSD-4 — the drift is measured against a stale, EMA-smoothed target.** This is
the repo's documented pattern 2 ("bounds are evaluated on EMA-smoothed proxies
instead of theorem inputs"), and it is the deepest one here: the theorem's `ν(t)`
is the continuous drift rate of `Q_target`, but `Q_target` is piecewise constant
between resyncs, so the quantity the bound reasons about has a staircase shape
while the bound reasons as if it were smooth.

**D-WSD-5 — two forms of `d_Gr` are in circulation.** `wsd.md:120` says
`‖sinΘ‖_F`; `wsd.py:53-54` says the chordal distance and explains why. The
theorem statement at `wsd.md:63` uses `d_Gr` without saying which. `‖sinΘ‖_F`
and `√(2k − 2‖Q₁ᵀQ₂‖²_F)` differ, so a reader who computes one and checks
against the other will not agree.

**D-WSD-6 — the proof document contradicts itself on reachability, and so does
`jawp.md`.** The addendum at `wsd.md:33-39` corrects an earlier claim and
reports 51 of 62 configs resolving `use_wsd: true`. The body section
`wsd.md:226-237` still says **"Every runnable config sets `use_wsd: false`"** and
names one enabling config. `proofs/jawp.md:261-262` repeats the same stale claim
while `proofs/jawp.md:35` contradicts it. **Re-measuring** the deep-merge over
every yaml under `config/` (68 files) I find **56 resolve `use_wsd: true` AND
`use_jawp: true`**, including `config/ablations/all_core.yaml` and all 39
leave-one-out rows. My file set differs from the 62 the addendum used, so the
counts are not directly comparable — but the direction is unambiguous and the
stale body text is wrong.

**Consequence for a reader:** a reader who skims to the body concludes the
WSD/JAWP antagonism is dormant. It is not.

### What holds up

The chordal-distance algebra itself, and the live-view wiring at
`jepa.py:1088-1093` — `h_target` is detached, `Q_ws` is not, so `dL/dQ` really
does reach `workspace_Q` in the same backward pass as JAWP. The measured
gradient opposition at `proofs/wsd.md:186-199` is a gradient-cosine measurement
and does not depend on any of the diagnostics above.

**D-WSD-7 — two sections of the proof document have no implementation at all.**
The "Constructive Bound via STA" section (`wsd.md:107-112`) states
`d_Gr ≤ ‖Σ_online − Σ_target‖₂/δ`, and the "Connection to EMA Scheduling" section
(`wsd.md:114-115`) claims the drift "enabl[es] adaptive `τ` scheduling". Neither
is implemented: there is no `Σ_online` anywhere in `wsd.py` or in WSD's path in
`jepa.py`, and `ema_beta` is assigned once at `wsd.py:156` and never written
again. `wsd_steady_state_error` (D-WSD-1) does not substitute for the first —
it is the running-drift EMA, not a spectral-gap ratio.

### Could not determine

The measured crossover `λ_wsd ≈ 0.42` is scale- and schedule-specific
(`D=64, k=6, random tokens`, per the document's own caveat) and I did not
re-measure it. Whether the antagonism is *harmful* is a science question, not a
code question, and this document does not answer it.

---

## 3. SWIP — Selective Whitening with Information Preservation

`proofs/swip.md` · `src/models/swip.py`

### (a) The claim

`L_SWIP = Σ_{i>k}(log σᵢ − log σ_target)² + w_h·Σ_{i<k}ReLU(λ_{i+1} − λᵢ + δ)`.
Property 2: "background isotropy at optimum — the term vanishes **iff every**
background log-variance equals log σ²".

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | Term 2 (workspace hierarchy) is implemented | **HOLDS** — `swip.py:245-252`, applied on both code paths as the document states |
| A2 | The background term is a **sum over individual eigenvalues** | **FAILS** — the code uses the trace/Jensen form `(D−k)·(log bg_mean − log σ_target)²` (`swip.py:258-272`) |
| A3 | Property 2 ("vanishes iff every background variance matches") | **FAILS** — see D-SWIP-1 |
| A4 | Scale-invariance | **RETRACTED in the document itself** (`swip.md:31-35`); correctly so |
| A5 | `F ∈ O(D)` maintained by per-step retraction | **HOLDS operationally**; it is a training-loop guarantee, and `ortho_err` is exported |

### (c) Divergences

**D-SWIP-1 — property 2 is false for the executed loss.** The sum-of-squares and
the trace form have different zero sets. I constructed a background spectrum
spanning 0.5 to 4.0 (8× anisotropy) whose *mean* equals σ_target:

```
proof form  Σᵢ (log λᵢ − log σ_t)²          = 4.078781     (nonzero)
executed    (D−k)·(log mean(λ) − log σ_t)²   = 0.000e+00   (zero)
```

The executed background term is **identically zero for any background spectrum
whose mean matches the target**, however anisotropic. The proof's property 2
cannot be read as a description of the shipped code.

**Consequence for a reader:** "SWIP has whitened the background" means "the
background's *mean* eigenvalue hit σ_target". It says nothing about isotropy, and
a run can report a zero background penalty while the background is strongly
anisotropic.

**D-SWIP-2 — the document contradicts itself about which form is implemented.**
§Statement (`swip.md:12`) presents the sum as "the implemented loss"; §Notes
(`swip.md:42-44`) says the JAWP path uses "a trace-based Jensen approximation".
The §Properties list is written for the sum and is silently wrong for the trace
form.

### What holds up

Term 2 is genuinely present and matches the document's formula. Non-negativity is
structural (a sum of squares and ReLUs). The retraction claim, and the honesty of
the retraction of the v1 scale-invariance claim — that retraction is the correct
call and is recorded in place rather than quietly dropped.

### Could not determine

Whether the two code paths (JAWP-projected spectrum vs top-k PCA spectrum) agree
on the background term's scale. Both route through `swip.py:258-272`, but I did
not trace `k_eff` slicing on both.

---

## 4. PCR — Predictive Cascade Refinement

`proofs/pcr.md` · `src/models/pcr.py`

### (a) The claim

A gated MLP cascade over disjoint column blocks `P_l` of a shared orthonormal
`Q ∈ R^{D×Σd_l}`. Four properties hold; four v1 theorems are **retracted in place**.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | Identity at init / during warmup | **HOLDS** — zero-initialised heads plus `warmup_factor = 0` for `step < 1000` (`pcr.py:348-351`, gate applied at `:372-373`) |
| A2 | Gate formula `sigmoid(α_l)·min(1, max(0, (step−W)/W))` | **HOLDS** — `pcr.py:348-351, 372` |
| A3 | `g_l ∈ [0,1]` | **HOLDS** — product of a sigmoid and a factor in `[0,1]` |
| A4 | `P_lᵀP_m = δ_lm I`, i.e. `Q` is on `O(D)` | **HOLDS only if the training loop's per-step SVD retraction runs.** It is an external guarantee; the module does not enforce it |
| A5 | The retracted v1 theorems | **Correctly retracted in the document** (`pcr.md:44-62`) |

### (c) Divergences

**D-PCR-1 — orthogonality is an unverified external precondition.** Every property
that depends on disjoint blocks rests on `Q` being retracted onto `O(D)` after
each optimizer step. Nothing in `pcr.py` asserts it. `pcr_ortho_score`
(`1 − mean|offdiag(QᵀQ)|`) is a *log-only* diagnostic.

**Consequence for a reader:** a checkpoint resumed without the retraction loop, or
any path where the retraction is skipped, silently loses the orthogonality that
property 2 rests on, and no number in the info dict reports it as a failure —
`pcr_ortho_score` just drifts toward 0.

**D-PCR-2 — this document is already doing what the rest of `proofs/` is not.**
It is worth naming as the reference pattern: the unsupported theorems are
retracted *in place*, with the reason, rather than left in the file as a latent
misrepresentation. No claim in this section disputes it.

### Could not determine

Whether `pcr_ortho_score` is ever asserted anywhere in `tests/`. I did not read
the full test file.

---

## 5. CGN — Contextual Gating Network

`proofs/cgn.md` · `src/models/cgn.py`

### (a) The claim

**Theorem (Information Routing)** and **Corollary (Partition of Unity)**: because
`g_v + g_m = 1`, "the total information is preserved — no information is lost by
gating, only re-routed."

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | `g_v + g_m = 1` on the gate that is actually applied to `Z` | **FAILS** — see D-CGN-1 |
| A2 | The two gated components are approximately conditionally independent given `(Z,Y)`, so the routing loss ≈ 0 | **FAILS — the code makes them exactly dependent.** `g_m = 1 − g_v` by construction (`cgn.py:264`) |
| A3 | Temperature annealing `τ(t) = τ_end + (τ_start − τ_end)(1 − t/T)` | **HOLDS** — `current_tau` is consumed at `cgn.py:251`; the schedule is implemented |
| A4 | "zero-cost: one linear layer + softmax per group" | **FAILS as stated** — the module adds `context_proj(z)` per position (`cgn.py:268`), a data-dependent term the proof never mentions |

### (c) Divergences

**D-CGN-1 — partition of unity holds for an intermediate, then is destroyed
twice.** The complementary pair is formed correctly at `cgn.py:263-264`
(`gate_on_masked = 1.0 - gate_on_visible`), interpolated per position at
`:278-280`. Then:

- `gate_values = gate_base * (0.5 + 0.5 * context_scores)` (`cgn.py:284`)
  multiplies by a **data-dependent** per-position, per-group factor. At a
  visible position the applied gate is `g_v·m_t`; at a masked one, `g_m·m_t`.
  Their sum is `m_t`, not 1.
- `gate_values.clamp(min_gate, 1.0)` (`cgn.py:287`, `min_gate = 0.01`) is a
  further non-affine break.

The gate multiplied into `z` at `cgn.py:294` does **not** satisfy `g_v + g_m = 1`
at any position.

**Consequence for a reader:** the Corollary's headline — "no information is lost
by gating, only re-routed" — does not describe the executed module. Any
information-preservation argument resting on it is unsupported.

**D-CGN-2 — the code's own header states the opposite of the proof's premise.**
`proofs/cgn.md:10` requires the gated components to be "approximately
independent given (Z,Y)" for the routing loss to vanish. The module makes them
**exact mirrors** — `P_ON(masked) = 1 − P_ON(visible)` — which the code says
outright at `cgn.py:340-343` ("the two pathways are exact mirrors, so the legacy
'orthogonality' is trivially satisfied"). Perfectly dependent gates maximise the
redundancy term, not minimise it. `compute_orthogonality_score` was redefined to
report *decisiveness* rather than orthogonality for exactly this reason.

**Consequence for a reader:** any CGN diagnostic named "orthogonality" is a
decisiveness score in `[0,1]`, and the proof's Step 2 reduction does not apply.

**D-CGN-3 — the proof's own Corollary overreaches its Theorem.** `cgn.md:8`
states the theorem *with* the conditional-entropy term on the right-hand side;
`cgn.md:14-15` then drops it and asserts preservation. The dropped term is
precisely what D-CGN-1 and D-CGN-2 populate.

### What holds up

The annealing schedule and the bounded-gate construction. `cgn.md` is one of the
documents with **no test at all** pinning any of its claims (per
`proofs/README.md:138`, "partition of unity — ❌ no test references
partition/unity"), which is consistent with D-CGN-1 surviving since the document
was written.

---

## 6. SPC — Spectral Predictive Coding

`proofs/spc.md` · `src/models/spc.py`

### (a) The claim

`L_SPC = Σ_b w_b·mean_{n,d}(‖z_pred^(b) − z_target^(b)‖²) = (1/(N·d))·Σ_b w_b‖Δc_b‖²`.
Property 1: `Σw_b = B` and `w_b ≥ min_weight`. Property 3: "Parseval consistency
up to a constant … uniform weights reproduce standard MSE/B".

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | `F ∈ O(D)` | **HOLDS operationally** via per-step retraction; `ortho_err` exported |
| A2 | `Σ w_b = B` exactly after clamp + renormalise | **HOLDS** — the renormalisation at `spc.py` is explicit in the document and in code |
| A3 | Equal band widths `d = D/B` | **HOLDS** — `n_bands` divides `D` in the constructor |
| A4 | **"uniform weights reproduce standard MSE/`B`"** | **FAILS — inverted by `B²`.** See D-SPC-1 |

### (c) Divergences

**D-SPC-1 — property 3's scaling is wrong by a factor of `B²`, and the document
contradicts the test it cites.**

Working it through: with `F` orthogonal and equal band widths,
`Σ_b‖Δc_b‖²_F = ‖Δz‖²_F`, so

```
L_SPC = (1/(N·d))·‖Δz‖²_F = (1/(N·d))·(N·d·B·MSE) = B · MSE
```

Measured, at uniform logits:

| `n_bands` | SPC | SPC/MSE | SPC/(B·MSE) |
|---|---|---|---|
| 1 | 2.009986 | 1.0000 | 1.0000 |
| 2 | 4.019972 | 2.0000 | 1.0000 |
| 4 | 8.039945 | 4.0000 | 1.0000 |

The code gives **`B × MSE`**. The document says `MSE/B`. `proofs/README.md:137`
records that `tests/test_spc.py::TestSPCTheorem::test_spc_subsumes_uniform_mse`
asserts `0.8 < SPC/(n_bands·MSE) < 1.2` — which agrees with **my measurement**,
not with the document's sentence.

**Consequence for a reader:** the property as *written* is false. Anyone reading
`spc.md` for the Parseval identity will get the constant backwards. Note this is
a documentation error, not a code defect — the code and its test agree with each
other.

**D-SPC-2 — the v1 simplex constraint and Gumbel weighting are retracted in
place** (`spc.md:52-71`), correctly, and `adapt_weights_to_predictability` is
genuinely wired (`spc.md:32-36`). This document is also close to the PCR
reference pattern.

### Could not determine

Whether the `w* ∝ var × predictability` adaptation (`spc.md:32-36`) actually
contracts toward `w*` — the document itself lists this as open work at
`spc.md:79-80`, and I did not test it.

---

## 7. STA — Spectral Transport Alignment

`proofs/sta.md` · `src/models/sta.py`

### (a) The claim

**Theorem 1** (Davis–Kahan): `d_Gr(Q_k(t), Q_k(t+1)) ≤ ‖Σ_res(t) − Σ_res(t+1)‖_op/δ`,
with `δ = λ_k − λ_{k+1} > 0`, reduced to `W_1/δ`.
**Theorem 2**: `W_1 = (1/D)Σ|λ_i − μ_i|` for sorted spectra.
**Corollary**: a downstream-variance bound in `W_1/δ²`.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | Spectral gap `δ > 0` at the `k`/`k+1` boundary | **FAILS — not enforced.** `delta = max(delta, self.eps)` (`sta.py:368`) |
| A2 | `δ` comes from the **residual** covariance `Σ_res` | **FAILS** — `sta.py:365-367` reads `self.current_eigenvalues`, the spectrum of `Cov(z)` |
| A3 | `δ` is at the **bottom**-k boundary (JAWP's workspace is bottom-k) | **FAILS** — `sort(descending=True)` at `sta.py:365` puts the boundary at the **top**-k |
| A4 | Reference spectrum tracked by EMA | **HOLDS** — `sta.py:231, 294-295` |
| A5 | Warmup | **HOLDS** — `sta.py:266-274` |
| A6 | STA's loss carries gradient | **FAILS — it cannot.** See D-STA-1 |

### (c) Divergences

**D-STA-1 — STA is a pure monitor; it cannot train anything.** `eig_cur` is
computed from `z.detach()` (`sta.py:286`) and `ref_eigenvalues` is a buffer. I
measured `sta_loss.requires_grad == False`. Correctly labelled "PROXY ESTIMATE"
at `sta.py:337-339`, but the *loss* carries no proxy caveat.

**Consequence for a reader:** `lambda_sta` scales a constant. An ablation on
`lambda_sta` measures nothing about STA.

**D-STA-2 — `δ` is wrong in three independent ways.** Wrong matrix (Cov(z) not
`Σ_res`), wrong end of the spectrum (descending sort, top-k boundary, while
JAWP's workspace is bottom-k), and floored at `eps` instead of reporting
absence. With `w1 = 1.0` and a real gap at k=3 I measured
`compute_davis_kahan_bound = 10.83`; with `δ ≤ eps` the same call returns
`1/eps = 1e8`. A smoothly decaying spectrum — the normal case — produces the
latter, printed as an "upper bound".

**Consequence for a reader:** `sta_davis_kahan_bound` is not a bound on anything.
It is `W_1` divided by a floored number. A large value is the *absence* of a
spectral gap, not evidence of workspace instability — the inverse of the
reading.

**D-STA-3 — `sta_spectral_gap` and the `δ` inside the bound are different
quantities.** `forward` reports the **minimum** gap over all significant
consecutive pairs (`sta.py:302-313`); `compute_davis_kahan_bound` uses the single
gap at the `k`/`k+1` boundary (`sta.py:367`). Neither matches the theorem's `δ`.
Plotting the first against the second will not be consistent.

**D-STA-4 — the proof's own reduction chain is invalid.** `sta.md:49-51`:

> `‖Δ‖_op ≤ max_i|λ_i(t+1) − λ_i(t)| ≤ W_1(λ(t), λ(t+1))`

The second inequality is backwards. `W_1 = (1/D)Σ_i|λ_i − μ_i|` is an **average**,
so `max_i|λ_i − μ_i| ≥ W_1`, always. The chain substitutes a *smaller* quantity
for `‖Δ‖_op` and thereby makes the bound tighter — which does not follow. This
is an error in the proof, independent of the code, and it would need a `√D` or
max-side correction even if the code were right.

**D-STA-5 — a Frobenius-vs-operator slip.** `sta.md:44` writes
`‖sinΘ‖_F ≤ ‖Δ‖_op/δ`. The standard sin-Θ theorem bounds the **operator** norm;
the Frobenius form carries a `√k` factor that the proof does not carry. The
module's comments already call this a "contested Davis–Kahan reduction"
(`sta.py:337-339, 374-375`), which is the right posture.

### What holds up

Theorem 2 (`W_1` between sorted spectra with uniform weights) — the derivation is
correct and the computation at `sta.py:299-300` matches it. The R11 fix: the
current spectrum is now recomputed every step while the reference follows the
interval cadence (`sta.py:276-295`), so `W_1` is no longer identically zero. I
measured `sta_loss = 0.0` on a single-step call only because the reference was
initialised from the same tensor on that step, which is the documented
`update_interval` behaviour, not the old bug.

### Could not determine

Whether `update_interval` is configured so that the reference and current
spectra diverge in practice on a real run. I did not read `defaults.yaml` for it.

---

## 8. CMC — Cross-Mask Consistency

`proofs/cmc.md` · `src/models/cmc.py`

### (a) The claim

`L_CMC = (1/|Ω|)Σ_{t∈Ω}‖z_pred⁽¹⁾[t] − z_pred⁽²⁾[t]‖²`, and a **Stability Theorem**
stating that for *any* probe `f`, `|f(z⁽¹⁾[t]) − f(z⁽²⁾[t])| ≤ ‖w‖√(L_CMC)`.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | The loss is the stated `1/|Ω|`-normalised sum over the overlap set | **HOLDS exactly** — `cmc.py:471-476` |
| A2 | Non-negativity | **HOLDS** — structural |
| A3 | The stability theorem is **pointwise** in `t` | **FAILS** — the proof derives it only in **averaged** form. See D-CMC-1 |
| A4 | `reuse_encoder` actually reuses the encoder output | **FAILS — it is a stub.** `cmc.py:311-312` returns `True` with the comment "cheap enough to always compute", i.e. it behaves as `mode="always"` |
| A5 | `cmc_overlap_ratio` is `|Ω| / |masked|` | **FAILS** — see D-CMC-3 |

### (c) Divergences

**D-CMC-1 — the Stability Theorem is misstated: averaged proof, pointwise
claim.** The proof's own Step 1 (`cmc.md:29-32`) obtains the pointwise bound with
`‖z⁽¹⁾[t] − z⁽²⁾[t]‖`. Its Step 2 (`cmc.md:36-41`) then *averages* over `t` to
replace that with `√L_CMC`. The Statement (`cmc.md:19-21`) nonetheless asserts
the averaged form pointwise, with **no quantifier over `t`**. As stated, the
theorem claims a bound on *every* position's discrepancy in terms of the
*average* — which is false: a single position can be arbitrarily far above the
mean.

The fix is not to average harder; the module already computes the worst-case
quantity (`cmc_max_inconsistency`, `cmc.py:481-486`), and the theorem wants
`max_t` where it has `mean_t`.

**Consequence for a reader:** "CMC loss = ε ⇒ every position's probe
discrepancy is ≤ ‖w‖√ε" is not what the module supports. The module supports
the average, plus a separately-reported maximum.

**D-CMC-2 — the skip path reports zero loss while predictions may diverge.**
Two exits return `cmc_loss: 0.0` with `cmc_skipped: True` — no overlap
(`cmc.py:429-439`) and insufficient overlap (`cmc.py:446-456`). Both are correct
in isolation, but `cmc_loss = 0` is ambiguous between "the two masks agree
perfectly" and "CMC did not run this step".

**Consequence for a reader:** any metric that averages `cmc_loss` over steps —
the natural thing, and the only thing most logging does — silently counts skipped
steps as perfect agreement. `cmc_skipped` is the discriminator and nothing
aggregates it.

**D-CMC-3 — `cmc_overlap_ratio` does not measure what `Ω` measures.**
`cmc.py:442` computes `overlap_count.float() / max(B, 1)` and **discards the
result** (a dead statement). The ratio actually returned at `cmc.py:444`
divides the overlap count by `primary_masked`, which is derived at
`cmc.py:443` from *whether predictions are non-zero* — `(z_pred_primary.abs().sum(-1) > 0).float()`,
carrying the module's own comment `# approximate` — not from the primary mask.

**Consequence for a reader:** `cmc_overlap_ratio` is not `|Ω|/|masked|` as
`cmc.md:14` defines `Ω`. It compares a count of both-masked positions against a
count of nonzero-prediction positions, which is a different thing.

### What holds up

The loss formula, exactly and including the `1/|Ω|` normalisation and the
overlap-only masking. Non-negativity. `tests/test_cmc.py` pins the triangle
inequality (`proofs/README.md:140`), which is a real property test — one of only
six documents with any.

---

## 9. GAC — Gradient-Allocated Capacity

`proofs/gac.md` · `src/models/gac.py`

### (a) The claim

`L_GAC = γ·Σ_{i: g_i < τ_grad}(τ_grad − g_i)·‖z_i‖²`, giving a **No Dead Zones
Guarantee**: every active dimension receives non-zero total gradient whenever
`‖z_i‖ > 0`.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | The summation is over starved dimensions | **HOLDS** — `starved_mask` at `gac.py:190`, applied at `:200` |
| A2 | `‖z_i‖²` is the per-dimension energy | **FAILS as a per-dimension quantity** — `dim_energy = (z_flat**2).mean(dim=0)` (`gac.py:197`) is a **batch mean**, so the theorem's `‖z_i‖²` is scaled by `1/N` |
| A3 | `g_i` is the gradient norm *of this batch's* `L_main` at `z_i` | **FAILS** — `gn = grad_norms.detach()` (`gac.py:173`); the norms arrive pre-computed and are detached, so the starvation *decision* is stale and non-differentiable |
| A4 | The No-Dead-Zones guarantee transfers to the trained model | **FAILS** — because of A2 and A3. The guarantee is about the *gradient*, which does flow (`dim_energy` is live), but the *threshold test* it is conditioned on is not part of the graph |
| A5 | `0 ≤ ρ_explore ≤ 1` | **UNVERIFIABLE in this form** — `compute_exploration_ratio` (`gac.py:255-267`) returns `max(γ·τ/g)` over active dims, which is not `ρ_explore` as `gac.md:57` defines it (a starved fraction) |

### (c) Divergences

**D-GAC-1 — the energy term is a batch mean, so the bound scales by `1/N`.**
Known and recorded (`proofs/IMPLEMENTATION_STATUS.md`, 2026-10-08 correction).
`gac.md:29` writes `‖z_i‖²`; `gac.py:197` computes `(z_flat**2).mean(dim=0)`.

**Consequence for a reader:** the No-Dead-Zones guarantee's constant is not the
one in the theorem. The *qualitative* claim ("a starved dimension with `‖z_i‖ > 0`
receives a non-zero contribution") survives the `1/N` factor, because the
gradient direction `z_i` is unchanged — only the magnitude is. The *quantitative*
reading does not.

**D-GAC-2 — the warmup ramp rescales it again.** `loss = γ · warmup_factor · (...)`
(`gac.py:200`). So the executed bonus is the theorem's term times two factors the
theorem does not have: `warmup_factor` and `1/N`. The document's banner concedes
the first; the `1/N` and the fact that they compound is the part that matters for
anyone comparing a printed `gac_loss` against a predicted magnitude.

**D-GAC-3 — the starvation test uses a detached, stale `g_i`.** `gac.py:173`
detaches. The loss still carries gradient through `dim_energy`, so `z_pred.grad`
is non-zero (verified by the 2026-10-08 audit: `6.75e-07` after one backward).
But *which* dimensions count as starved is decided by a tensor outside the graph,
computed by the caller before this call.

**Consequence for a reader:** `gac_n_starved` / `gac_starved_fraction` describe
the gradient of the **previous** step's main loss, not the one being
differentiated. The guarantee is about the current step.

### What holds up

The loss skeleton and the `min_weight`-style clamp semantics. `proofs/README.md:141`
records `tests/test_gac.py::TestGACCore::test_starved_count` as partial coverage:
the count and fraction are pinned, the `0 ≤ ρ ≤ 1` bound is **not** asserted —
consistent with A5 above.

### Could not determine

Whether `τ_grad = 1e-4` (`gac.py:127`) is calibrated against the gradient scale
actually seen in training, which depends on `lambda_*` for the other eleven
mechanisms. I did not run a training smoke test (out of scope for that review), and a
single sample would not have settled it anyway.

---

## 10. PUC — Prediction Uncertainty Calibration

`proofs/puc.md` · `src/models/puc.py`

### (a) The claim

`L_PUC = η·max(0, H_target − H(Σ_pred))` (Statement) — and, separately, a
log-det barrier `−½logdet(Σ+εI) + D/2·log(2πe σ_target²)` (Step 5). **Theorem
(Minimax Prediction Optimality)** via Jaynes + Sion + Donsker–Varadhan.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | The executed loss carries an autograd edge to `z_pred` | **HOLDS** (the 2026-10-08 fix, verified — see below) |
| A2 | The executed loss is the stated entropy-deficit formula | **FAILS** — see D-PUC-1 |
| A3 | The risk constraint `E‖z−z_target‖² ≤ R` is formed | **FAILS — never formed.** `z_target` is accepted at `puc.py:223` and never read in the body |
| A4 | `min_log_det` floors the log-det | **FAILS — accepted at `puc.py:99,108` and never referenced in `forward`.** Verified by source inspection |
| A5 | `H(Σ_pred)` estimates the entropy of `z_pred` | **FAILS** — see D-PUC-2 |
| A6 | `H_target = D/2·log(2πe)` is the right target | **HOLDS by construction** (`puc.py:125`), but see D-PUC-2: it assumes isotropy |
| A7 | The Sion minimax step (proof Step 4) is established | **FAILS — the proof does not establish it.** See D-PUC-3 |

### (c) Divergences

**D-PUC-1 — there are three formulas and the code implements a fourth.** The
proof states one form (`:34`), derives a different one (`:79`), and the code
executes `η·warmup·relu(−Σᵢlog λᵢ + 10·n_components)/n_components`
(`puc.py:296-301`), gated on `entropy_deficit > 0` (`puc.py:304-307`). The
entropy deficit is computed (`:289`) and then used **only as a switch**.

The constant `10·n_components` at `puc.py:299` has **no derivation anywhere** —
not in the proof, not in a comment. It sets the ReLU's threshold: the loss is
active whenever `Σ log λᵢ < 10n`, i.e. whenever the geometric mean of the
tracked eigenvalues is below `e^10 ≈ 22026`.

**Consequence for a reader:** `puc_loss` is not an entropy deficit, is not the
`η` in the proof, and its scale is set by an undocumented constant.

**D-PUC-2 — `puc_entropy` is not the entropy of `z_pred`, and the bias is large.**
The estimator sums the tracked block exactly and then assigns the `D−n`
**untracked** directions the **mean of the top-`n` tracked eigenvalues**
(`puc.py:276-281`). For a decaying spectrum that over-states the untracked block,
and since `D−n = 24` of `D = 32`, it dominates. Measured:

| spectrum | true `H` | `puc_entropy` | deficit | loss |
|---|---|---|---|---|
| isotropic | 45.414 | 51.166 | 0.000 | 0.0000 |
| mild decay | −36.363 | 41.712 | 3.694 | 0.1025 |
| strong decay | −117.926 | 41.204 | 4.202 | 0.1028 |

**`puc_entropy` reads 41–51 for every spectrum tested, including one whose true
entropy is −117.9 — an error of ~159 nats.** And the *loss* barely discriminates:
true entropies of −36.4 and −117.9 differ by 82 nats and produce losses of
`0.1025` and `0.1028`.

**Consequence for a reader:** `puc_entropy`, `puc_entropy_deficit` and
`puc_overconfidence` are dominated by a block-level modelling assumption, not by
the data. The one signal that discriminates (the tracked block's log-eigenvalues)
is `n = 64` of `D = 768` dimensions at the default (`puc.py:104`).

**D-PUC-3 — the proof's Step 4 asserts what it needed to prove.** `puc.md:72-74`:

> "The inner minimum is achieved by `q*` … For any bounded loss `ℓ`,
> `E_{q*}[ℓ] ≤ E_q[ℓ]` for all `q` with the same risk. This follows from the data
> processing inequality: more entropy = more robust = lower worst-case loss."

Maximum entropy does **not** minimise an arbitrary bounded loss; that is a
separate theorem with its own hypotheses, and it is not what the data processing
inequality says. The minimax-optimality claim is therefore not established by
this argument. This is a proof-internal gap, not a code divergence, and it would
survive any amount of code auditing.

**D-PUC-4 — the "no gradient" headline is half-retired, correctly.** The 2026-10-08 fix
flipped `use_differentiable_entropy` to `True` (`puc.py:100`). I verified the
fixed path genuinely has the edge: `puc.py:263` is `torch.linalg.eigvalsh(cov_b)`
on this batch's covariance, `cov_b` is built from a **non-detached** `z_flat`
(`:261-262`), and `log_det_tracked = log_eigenvalues.sum()` (`:293`) feeds
`loss_tensor` (`:296-301`). **So: yes, the fixed path has the autograd edge.**

Two caveats belong with that answer:
- `tracked_entropy` — which carries the edge — is immediately `.item()`'d at
  `puc.py:283` on its way into `estimated_entropy`. So the *entropy* diagnostics
  are correctly detached; only `log_det_tracked` retains the graph. That is the
  right design, but it means **the quantity that gates the loss (`:304`) is
  itself detached**, so the gate is a discontinuous, non-differentiable function
  of the batch.
- The inert path still exists, still warns at construction (`puc.py:110-119`),
  and is **not reachable from any config** — there is deliberately no
  `puc_use_differentiable_entropy` key. `puc_carries_grad` (`puc.py:340`) is the
  runtime tell. All of this is consistent with the 2026-10-08 correction.

**D-PUC-5 — I disagree with the matrix's phrase "EMA-smoothed proxies" for
PUC.** That phrase is accurate for WSR/RDC/WSD. For PUC's *default* path the
eigenvalues come from `eigvalsh` of this batch's covariance (`puc.py:263`), not
from an EMA. The EMA (`running_eigenvalues`, `:190-192`) is only read on the
inert branch. The matrix's PUC row does not make this claim, so this is a note
against a pattern, not a matrix correction — but the headline that matters for
PUC is (a): the executed loss matches neither stated form.

### What holds up

Non-negativity of the returned scalar, which is structural (`relu` at
`puc.py:299`, and the `else` branch at `:307` returns exactly 0). The corrected
gradient story in D-PUC-4 — verified, not assumed.

---

## 11. RDC — Representation Drift Compensation

`proofs/rdc.md` · `src/models/rdc.py`

### (a) The claim

`L_RDC = η·E‖Δz_⊥‖²`. **Drift Compensation Bound:**
`‖z_T − z*_T‖_⊥ ≤ ε·(1−η_rdc)^T·T/√k`, with `ε = max_t‖δ_⊥,t‖`.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | `Δz_⊥ = (I − QQᵀ)Δz` is an orthogonal projection | **FAILS unless `Q ∈ St(D,k)`**, which the module does not enforce |
| A2 | `η_rdc` attenuates the per-step orthogonal drift by exactly `(1−η_rdc)` | **FAILS — an empirical assumption about the optimiser**, with no basis in gradient descent. See D-RDC-2 |
| A3 | `ε = max_t‖δ_⊥,t‖` | **FAILS** — `eps_estimate = self.running_ortho_drift_norm.item()` (`rdc.py:203`), an EMA |
| A4 | `T` is the step at which the bound is evaluated | **FAILS** — `T_eff = min(step, 10000)` (`rdc.py:206`), an arbitrary cap |

### (c) Divergences

**D-RDC-1 — `rdc_transient_bound` underflows to zero and reports that as a
result.** The factor is `(1−η)^T_eff` with `η = 0.01` at the default
(`rdc.py:__init__`, verified) and `T_eff = step`:

| `η` | `(1−η)^1000` | `(1−η)^10000` |
|---|---|---|
| 0.001 | 3.677e−01 | 4.517e−05 |
| 0.01 | 4.317e−05 | 2.249e−44 |
| 0.05 | 5.292e−23 | 1.722e−223 |
| 0.1 | 1.748e−46 | **0.000e+00** |

**Consequence for a reader:** past roughly step 500 at the default `η = 0.01`,
`rdc_transient_bound` is a float-underflow artefact reading `~0`. It looks like a
proof that orthogonal drift provably vanishes. It is a number that has fallen off
the bottom of float64. The code's own comment at `rdc.py:205` says
"arbitrary `T_eff` cap — diagnostics, not certified bounds", which is the right
posture — but the *number* carries no such caveat, and
`rdc_theoretical_bound = min(transient, stationary)` (`:221`) inherits it.

**D-RDC-2 — the proof's own argument does not reach its stated result, and the
document says so mid-derivation.** `rdc.md:70-89` contains a visible
self-correction ("Hmm, this measures how RDC CHANGES the trajectory, not how much
drift remains. Let me restate the bound correctly."). Two problems remain after
the restatement:

- Step 3 introduces `(1−η_rdc)^t` as a *compounding* factor with the parenthetical
  "(Here the `(1−η_rdc)^t` accounts for compounding: each step's drift is further
  reduced because RDC also acts on drift from previous steps accumulated in
  `z_⊥`)". Step 2's sum has `(1−η)^1`, not `(1−η)^t`. The exponent is asserted,
  not derived.
- Step 5 derives `≤ ε(1−η)·T/√k`, then the **Result line at `rdc.md:148`
  reverts to `ε(1−η)^T·T/√k`** — the form that was abandoned. The final stated
  bound is not the one the steps produce.

Step 4's `‖δ_⊥,t‖ ≤ ε/√k` is also unexplained: `ε` is *defined* at `rdc.md:50` as
`max_t‖δ_⊥,t‖`, so dividing by `√k` weakens the bound by a factor the text does
not justify.

**D-RDC-3 — `ε` is an EMA proxy and `T` is capped.** Per A3/A4. This is the repo's
documented pattern 2, and it is the deepest instance here because the bound has
*three* degrees of freedom and two of them are proxies.

### What holds up

The projector algebra itself (`:170-179` in `rdc.py`): given an orthonormal `Q`,
`drift_flat − Q(Qᵀdrift_flat)` is the orthogonal component, and
`drift_ortho.pow(2).sum(dim=-1).mean()` is `E‖Δz_⊥‖²` as stated. The stationary
bound's *form* `ε(1−η)/(η√k)` is the correct evaluation of the geometric series
in Step 5. Non-negativity is structural.

### Could not determine

Whether `Q`'s orthonormality is actually violated during a real run. The
retraction runs per step (`jepa.py:532`), and I did not instrument a forward pass
to measure `‖QᵀQ − I‖` at the moment RDC reads `Q`.

---

## 12. WSR — Workspace Sharpness Regularization

`proofs/wsr.md` · `src/models/wsr.py`

### (a) The claim

`ρ_Q = max_{‖Δ‖_F ≤ ρ, Δ ∈ T_Q Gr}[L(Q+Δ) − L(Q)]`.
**Theorem 1**: `|L_train(Q̂) − L_test(Q̂)| ≤ C√(ρ_Q/n) + O(1/√n)`.
**Theorem 2**: PAC-Bayes. **Theorem 3**: `ρ_Q = ρ_spectral + ρ_directional`.

### (b) Assumptions

| # | Assumption | Verdict |
|---|---|---|
| A1 | The loss minimises `ρ_Q` | **FAILS — the loss carries no gradient in either mode.** See D-WSR-1 |
| A2 | `ρ_Q` is a max over a `ρ`-ball | **FAILS in `mode='gradient'`** — it is `ρ·‖∇_Gr L‖/‖Q‖_F` (`wsr.py:424`), a first-order directional scale |
| A3 | `ρ_spectral = ρ‖(I−QQᵀ)∇_Q L‖_F` | **FAILS** — never computed. See D-WSR-2 |
| A4 | `ρ_spectral + ρ_directional = ρ_Q` (equality) | **FAILS — the theorem is stated as an equality and its own proof derives an inequality** (`wsr.md:95`: `≤ ρ_spectral + ρ_directional` by the triangle inequality) |
| A5 | The perturbation retraction returns `Q` to within `ρ` | **FAILS** — see D-WSR-3 |
| A6 | Theorem 1's optimality step | **FAILS — the step is unsound.** See D-WSR-4 |

### (c) Divergences

**D-WSR-1 — WSR cannot minimise anything. The returned loss is a constant.**

Both modes produce a tensor with no autograd history:

- `mode='gradient'`: `_resolve_euclidean_gradient` returns
  `self._lagged_gradient.detach().clone()` (`wsr.py:300, 326`). Everything
  downstream — `sharpness` at `:424`, `loss` at `:434` — is arithmetic on
  detached floats.
- `mode='sam'`: `L_current` and `L_perturbed` are `.item()` calls under
  `torch.no_grad()` (`wsr.py:501, 525-528`); `loss_tensor = Q.new_tensor(...)`
  at `:534` has no graph.

I measured `mode='gradient'` → `loss = 0.002904`, `requires_grad = False`.

This is the matrix's headline ("detached loss cannot minimize the claimed
objective") and it is correct. **The consequence is the whole of Theorem 2**:
WSR is presented as "the PAC-Bayes bound minimizer" (`wsr.md:55`), and it does
not minimise the KL-compression term, or anything else.

**D-WSR-2 — `wsr_spectral_sharpness` is a duplicate of `wsr_sharpness`.**
`wsr.py:424` and `wsr.py:428` are the same expression, character for character.
I measured them identical (`0.290367`, difference `0.0`). Meanwhile
Theorem 3's actual spectral component, `ρ‖(I−QQᵀ)∇_Q L‖_F` (`wsr.md:83`), is
**never computed anywhere in the file**.

**Consequence for a reader:** Theorem 3's two-component decomposition is not
reported. `wsr_spectral_sharpness` is the headline value under a second name, and
`wsr_directional_sharpness` (`wsr.py:431`) is reported *alongside* it rather than
as a part of it. Any "spectral vs directional" comparison in a log is comparing
the total against one of its parts.

**FIXED (PR #23) — the decomposition is now reported, and the theorem was
restated in quadrature.** Two distinct defects were hiding in D-WSR-2:

1. *The missing quantity.* The theorem's total `ρ_Q = ρ·‖∇_Q L‖_F` was never
   computed by any code path — so the decomposition had no observable left side.
   The module now reports it as **`wsr_ambient_sharpness`** in both modes (and
   in the warmup sentinel dict). Measured with a lagged gradient whose in-span
   and off-span parts are both nonzero: `ambient = 0.394467`,
   `spectral = 0.370940`, `directional = 0.134193`, and
   `‖ambient² − (spectral² + directional²)‖/ambient² = 7.3e-10` in float64 —
   the identity holds, and `ambient ≠ spectral` so the total is not a fourth
   name for an existing part.
2. *A false identity.* The header theorem claimed the LINEAR sum
   `ρ_Q = ρ_spectral + ρ_directional`. That is false in general: the two
   components live in orthogonal complements of R^D (`(I−QQᵀ)∇` vs `QQᵀ∇`),
   so they add in **quadrature** — `ρ_Q² = ρ_spectral² + ρ_directional²`. The
   linear claim only ever held numerically when one part was ~0. The header now
   states quadrature, and `tests/test_wsr.py::TestWSRBasic` pins both the
   triangle inequality and the Pythagoras identity over 10 random (Q, ∇) pairs.

The headline `wsr_sharpness` still equals the spectral component in gradient
mode — that is now a *stated* geometric consequence (`grad_Gr == (I−QQᵀ)∇`,
see `_grassmann_gradient`), not an aliasing accident, and the WSR loss is
unchanged: `η·warmup·ρ_spectral/√k`, the off-manifold component, which is the
one that moves the subspace. The header comment previously paired WSR with
ρ_directional; that pairing contradicted the module's own loss and has been
corrected there as part of this fix.

MUTATION VERDICT for the two strengthened/added tests in `TestWSRBasic`:
reverting the wsr.py change makes both fail immediately with
`KeyError('wsr_ambient_sharpness')` — the ambient total is the only key the fix
adds, and it is the quantity every assertion now names.

**D-WSR-3 — the `mode='sam'` retraction does not return within `ρ`.** This is
the open defect marked by the `strict=False` xfail at
`tests/test_training_state_guards.py:508-533`. I measured it over 12 seeds
(D=32, k=4, ρ=0.05):

```
‖Q − retract(Q + ρt)‖_F        : 3.4639 … 3.9997   (all 12 seeds)
‖QQᵀ − Q_r Q_rᵀ‖_F            : 0.0707              (all 12 seeds)
seeds where the subspace moved more than ρ : 12/12
```

The diagnosis in the xfail is mechanistically right — the canonical-QR sign fix
reads `diag(Q_retracted[:k, :])` (`wsr.py:244`), a non-triangular block, where it
should read `diag(R)`. But **I should correct the severity description**: on these
12 seeds the sign vector is essentially uniform (a global `±`), which is
*harmless for the subspace* — the subspace distance stays at `0.0707` — while the
Frobenius distance is near the maximum `2√k = 4.0`. The "flips individual
COLUMNS … in a near-arbitrary direction" phrasing describes a failure mode these
seeds do not exhibit.

What *does* fail on all 12 seeds is the ρ-semantics: a correct polar/SVD
retraction is nonexpansive in Frobenius, so `‖QQᵀ − R Rᵀ‖_F ≤ ‖ρt‖_F = ρ`. The
measured `0.0707 = √2·ρ` exceeds that on every seed. **QR is not a retraction.**

**Consequence for a reader:** in `mode='sam'`, `wsr_sharpness` is
`L(Q_pert) − L(Q)` where `Q_pert` was displaced by up to `√2·ρ`, not `ρ`. The
sharpness is not the sharpness the theorem defines.

**D-WSR-4 — Theorem 1's step 3 is unsound.** `wsr.md:42-43`:

> `L_test(Q̂) ≤ E_{Q ~ Unif(B_ρ)}[L_test(Q)]` (by optimality of `Q̂` within `B_ρ`)

`Q̂` is the minimiser of `L_train`, not of `L_test`. Optimality of `Q̂` within
`B_ρ` holds for `L_train` and transfers to nothing. The bound as stated does not
follow. Recorded in the matrix headline; restated here because it is the step a
reader would most likely accept without checking.

**D-WSR-5 — an undocumented `√(Dk)` divisor.** `sharpness` divides by `‖Q‖_F`
(`wsr.py:424`). For an orthonormal `(D,k)` matrix that is `√(Dk) ≈ 19.6` at
D=64, k=6. `ρ_Q` in `wsr.md:20` has no such divisor.

**Consequence for a reader:** `wsr_sharpness` is not comparable across `D`, `k`,
or configs — an ablation over the workspace dimension will show the diagnostic
shrinking for arithmetic reasons.

**D-WSR-6 — Theorem 3's statement contradicts its own proof.** The Statement
(`wsr.md:80`) says `ρ_Q = ρ_spectral + ρ_directional`, an **equality**. The
proof's Step 3 (`wsr.md:95`) derives `ρ_Q ≤ ρ_spectral + ρ_directional`, an
inequality, "by the triangle inequality on the norm". Even with the correct
decomposition implemented (which it is not — D-WSR-2), the two reported
components would not be certified to sum to the headline.

### What holds up

The Grassmann tangent projection. `grad_grassmann = G − Q(QᵀG)` (`wsr.py:216-217`)
does land in `T_Q Gr` — the `(I−QQᵀ)G` remainder is the Grassmann tangent
direction, dimension `k(D−k)`, not the Stiefel `QΩ`. That is the right projection
for the manifold `ρ_Q` is defined over. The constants `ρ = 0.05`, `η = 0.01` match
the document. `wsr_grad_source` / `wsr_gradient_substituted` (`:461-462`) are
exemplary provenance reporting: they say plainly when the printed number is not
the documented one.

---

# Dry divergence list

Scan table. `MECH-DIV-n` is the local tag; the middle column is the one-line form.
**Type**: `H` = a hypothesis of a stated theorem fails in the code;
`P` = an error internal to the proof document; `S` = a number's stated meaning
does not match what the module computes.

Four further findings are tagged in the body but deliberately **not** listed
here, because they are corrections rather than divergences: **D-JAWP-4** (the
α-term's `Q`-detachment is a documented design decision, not an oversight),
**D-PCR-2** and **D-SPC-2** (both documents correctly retracted their
unsupported claims in place), and **D-PUC-5** (a note against the "EMA proxies"
pattern, which does not describe PUC's default path).

| ID | Mech | Type | Divergence | Where |
|---|---|---|---|---|
| **D-JAWP-1** | JAWP | **S** | `predictive_relevance` mixes an `N·k` mean with an `N·D` mean; reads **1.0** when workspace prediction is *exact* and **0.0** when it is *worst*. Inverted | `jawp.py:462-503` |
| D-JAWP-2 | JAWP | H | Theorems assume fixed `k`; the loss slices `Q[:, :k(t)]` under a 10 000-step curriculum | `jawp.py:340-346, 387` |
| D-JAWP-3 | JAWP | H | The α-term's `Q` is detached, so it contributes nothing to learning `Q`; and `.mean()` over `N·D` vs `N·k` | `jawp.py:398-406` |
| **D-WSD-1** | WSD | **S** | `wsd_steady_state_error ≡ wsd_running_drift` exactly — `ν/λ` with `ν=rd/T`, `λ=1/T`. A tautology, not a bound | `wsd.py:318-326` |
| **D-WSD-2** | WSD | **S** | `wsd_lambda_estimate` is `1/sync_interval`, not `lambda_wsd`. They coincide at the default and diverge 10× on the WSD ablation | `wsd.py:322` |
| D-WSD-3 | WSD | H | `clamp(min=0)` reports a 3×-scaled, non-orthonormal `Q` as **perfectly synchronised** (drift 0) | `wsd.py:274, 281` |
| D-WSD-4 | WSD | H | `Q_target` is top-k of an EMA, piecewise-constant between resyncs; the theorem's `ν(t)` is a continuous drift rate | `wsd.py:189-206, 261` |
| D-WSD-5 | WSD | P | Proof defines `d_Gr = ‖sinΘ‖_F`; the code computes the **chordal** form. Two live definitions | `wsd.md:120` vs `wsd.py:274` |
| D-WSD-6 | WSD | P | `wsd.md:226-237` and `jawp.md:261-262` still claim every config ships `use_wsd: false`; **56 of 68** configs resolve both flags true | `wsd.md`, `jawp.md` |
| D-WSD-7 | WSD | H | The constructive Davis–Kahan bound and the adaptive-`τ` claim are **not implemented at all** | `wsd.md:107-115` |
| **D-SWIP-1** | SWIP | **S** | Property 2 is false for the executed trace-form background term: it vanishes for any spectrum whose **mean** matches σ_target (verified 4.079 vs 0.0) | `swip.py:258-272` |
| D-SWIP-2 | SWIP | P | §Statement presents the sum as "the implemented loss"; §Notes says the code uses a trace approximation. The §Properties list is wrong for the code | `swip.md:12` vs `:42-44` |
| D-PCR-1 | PCR | H | Orthogonality of `Q` (which every "property that holds" depends on) is an unverified external precondition; `pcr_ortho_score` is log-only | `pcr.py`, retraction in `train.py` |
| **D-CGN-1** | CGN | **S** | `g_v + g_m = 1` holds for `gate_base`, then is destroyed by a data-dependent modulation and a clamp. The applied gate does **not** satisfy it | `cgn.py:278-294` |
| **D-CGN-2** | CGN | **S** | The proof needs conditionally **independent** gates; the code makes them exact mirrors (`g_m = 1 − g_v`), maximising redundancy | `cgn.py:264, 340-343` |
| D-CGN-3 | CGN | P | The Corollary drops the conditional-entropy term that its own Theorem carries — and that D-CGN-1/2 populate | `cgn.md:8` vs `:14-15` |
| **D-SPC-1** | SPC | **P** | "uniform weights reproduce standard MSE/**B**" — inverted by `B²`. Measured `SPC = B·MSE` for B ∈ {1,2,4}; the test the doc cites agrees with the code, not the doc | `spc.md:46-47` |
| **D-STA-1** | STA | **S** | `sta_loss.requires_grad == False` (measured). STA cannot train | `sta.py:286, 316` |
| **D-STA-2** | STA | **S** | `δ` comes from `Cov(z)` not `Σ_res`, at the **top**-k boundary not bottom-k, and is floored at `eps` → `w1/1e-8` when the gap is absent | `sta.py:365-370` |
| **D-STA-3** | STA | S | `sta_spectral_gap` (min over all significant pairs) ≠ the `δ` used inside the bound (single k/k+1 pair) | `sta.py:302-313` vs `:367` |
| **D-STA-4** | STA | **P** | `max_i|Δλ_i| ≤ W_1` is backwards — `W_1` is a mean. The proof's own reduction chain is invalid | `sta.md:49-51` |
| D-STA-5 | STA | P | `‖sinΘ‖_F ≤ ‖Δ‖_op/δ` omits the `√k` factor of the sin-Θ theorem | `sta.md:44` |
| **D-CMC-1** | CMC | **P** | Stability Theorem is stated **pointwise** but proved only **averaged**. As stated, false | `cmc.md:19-21` vs `:29-41` |
| D-CMC-2 | CMC | S | Skipped steps report `cmc_loss = 0.0`, indistinguishable from perfect agreement | `cmc.py:429-439, 446-456` |
| D-CMC-3 | CMC | S | `cmc_overlap_ratio` divides by a nonzero-*prediction* proxy, not by the primary mask; the true ratio at `:442` is computed and discarded | `cmc.py:442-444` |
| D-GAC-1 | GAC | S | Energy is a batch **mean**; the theorem's `‖z_i‖²` is scaled by `1/N` | `gac.py:197` |
| D-GAC-2 | GAC | S | The executed bonus carries `1/N` **and** `warmup_factor`; the two compound | `gac.py:200` |
| D-GAC-3 | GAC | H | The starvation test uses a **detached, stale** `g_i`, so `gac_n_starved` describes the previous step's main loss | `gac.py:173, 190` |
| **D-PUC-1** | PUC | **S** | Executed loss is `η·warmup·relu(−Σlog λᵢ + 10n)/n` — a fourth form, gated on a detached deficit, with an **underived** constant `10n` | `puc.py:296-307` |
| **D-PUC-2** | PUC | **S** | `puc_entropy` reads 41–51 nats for every spectrum tested, including a true entropy of **−117.9**. The untracked block is assigned the top-n mean | `puc.py:276-283` |
| D-PUC-3 | PUC | P | Step 4 asserts `E_{q*}[ℓ] ≤ E_q[ℓ]` "by the data processing inequality" — not what DPI says. Minimax-optimality is not established by this argument | `puc.md:72-74` |
| D-PUC-4 | PUC | H | The risk constraint is never formed; `z_target` is accepted and never read. `min_log_det` is accepted and never used | `puc.py:99, 108, 223` |
| **D-RDC-1** | RDC | **S** | `rdc_transient_bound` underflows: `(1−0.01)^T` is `4e-44` at T=10 000, `0.0` at η=0.1. Reads as "drift provably vanishes" | `rdc.py:206-207` |
| D-RDC-2 | RDC | P | The `(1−η)^t` compounding factor is asserted, not derived; the **Result line reverts to a bound the steps abandoned** | `rdc.md:97-101, 148` |
| D-RDC-3 | RDC | H/S | `ε` is an EMA proxy, `T` is capped at an arbitrary 10 000, and `η_rdc` is assumed to attenuate drift by exactly `(1−η)` per step | `rdc.py:203, 206` |
| **D-WSR-1** | WSR | **S** | `wsr_loss` carries **no gradient** in either mode (measured `requires_grad = False`). Theorem 2's "bound minimizer" minimises nothing | `wsr.py:300-434, 500-534` |
| **D-WSR-2** | WSR | **S** | `wsr_spectral_sharpness` is character-identical to `wsr_sharpness`. Theorem 3's actual spectral term is never computed — **fixed in PR #23** (ambient total added, theorem in quadrature) | `wsr.py:424` vs `:428` |
| **D-WSR-3** | WSR | H | `mode='sam'` retraction displaces the subspace by `√2·ρ`, exceeding `ρ` on 12/12 seeds; QR is not a retraction. **Open, escalated, not fixed** | `wsr.py:242-247` |
| D-WSR-4 | WSR | P | Theorem 1 step 3 uses optimality of `Q̂` for `L_test`; `Q̂` minimises `L_train` | `wsr.md:42-43` |
| D-WSR-5 | WSR | S | `sharpness` divides by `√(Dk)`; `ρ_Q` has no such divisor. Not comparable across `k` or `D` | `wsr.py:424` |
| D-WSR-6 | WSR | P | Theorem 3 is **stated as an equality** and its own proof derives an inequality (triangle) | `wsr.md:80` vs `:95` |

## Cross-cutting patterns this audit confirms, and one it adds

The four patterns recorded at `proofs/IMPLEMENTATION_STATUS.md:107-116` all
reproduce, with named instances:

1. **Loss formulas diverge from their stated form** — PUC (D-PUC-1), SPC
   (D-SPC-1, in the *document*, not the code), SWIP (D-SWIP-1), STA (D-STA-1).
2. **Bounds computed on EMA-smoothed proxies** — WSD (D-WSD-1, D-WSD-4), RDC
   (D-RDC-3), STA (D-STA-2).
3. **Proof assumptions not enforced** — orthonormality at the WSR call site
   (D-WSR-3), exact simplexes at SPC (retracted), fixed `k` at JAWP (D-JAWP-2).
4. **Theorems about objects never implemented** — CGN's complementary gates
   (D-CGN-1, D-CGN-2); PCR's cascade form is now correctly retracted in place.

The pattern this audit adds:

5. **A diagnostic's stated meaning is not what it computes, and the
   disagreement is invisible in the number.** Seven instances: D-JAWP-1,
   D-WSD-1, D-WSD-2, D-WSR-2, D-PUC-2, D-RDC-1, D-STA-2. In five of the seven the
   printed value is internally *consistent* — it is just a different quantity
   under the same name, or a number that has underflowed to a reassuring zero.
   A reviewer reading a training log has no way to tell, and the
   `proofs/README.md` verification table does not cover any of them.

Two of these (D-JAWP-1 and D-SPC-1) also contradict the tests the documents cite,
in opposite directions: SPC's *test* is right and its *document* is wrong; JAWP's
diagnostic is wrong and nothing tests it. That asymmetry is why
`proofs/README.md:123-146` is the authority and this document is the supplement.

## What is NOT wrong

Recorded because a document that lists only defects misleads as much as one that
lists only successes:

- **JAWP Theorem 3** (SVD retraction) is implemented exactly as stated
  (`jawp.py:332`), and its test exists — under a different name than the document
  claims.
- **JAWP Theorem 1's algebra** — `jawp.py:396-398` really is
  `tr(QᵀΣ_res Q)/N`.
- **CMC's loss formula** — `cmc.py:471-476` is the stated `1/|Ω|`-normalised sum,
  exactly.
- **STA Theorem 2** — the `W_1` derivation is correct and the implementation
  matches it.
- **SWIP term 2** — genuinely implemented (R15), and the v1 scale-invariance
  claim was retracted *in place*, correctly.
- **WSR's Grassmann tangent projection** — the right manifold.
- **The R11 sync fix** — STA's `W_1` is no longer identically zero.
- **PUC's gradient** — the 2026-10-08 fix genuinely works; the autograd edge is
  there (D-PUC-4).
- **`proofs/pcr.md` and `proofs/spc.md`** are the reference pattern: they retract
  their unsupported claims in place, with reasons, rather than leaving them to be
  discovered. More documents should be written this way.

---

## Appendix: how to reproduce, and what was not done

**Commit audited:** `main` at `6832d35`.

Every claim marked *measured* above was produced by a one-shot verification
script (not kept in the repository), each asserting `src.__file__` resolves to
the worktree root before executing anything:

| Script | Establishes |
|---|---|
| `verify_jawp.py` | D-JAWP-1 — `predictive_relevance == max(0, 1 − (D/k)·w)` at 8 values of `w` |
| `verify_wsd.py` | D-WSD-1 (tautology), D-WSD-2, D-WSD-3, and the config reachability count |
| `verify_rest.py` | SWIP proof-vs-code terms, RDC underflow table, WSR gradient + duplicate, STA gradient, PUC |
| `verify_rest2.py` | corrected SWIP spectrum (D-SWIP-1), WSR retraction across 12 seeds (D-WSR-3) |
| `verify_puc.py` | D-PUC-2 — entropy-estimator bias table |
| `verify_spc.py` | D-SPC-1 — `SPC = B·MSE` |

The provenance assertion matters: an editable install pointed at a different clone
was present on this machine, and a stale `__pycache__` produced traceback paths
naming that other tree. Every script asserts the resolved path before running.

**Not done, deliberately:**

- **No training.** Out of scope for that review, and it would not have settled most of the
  open questions anyway — these are code-vs-document disagreements, not
  empirical ones. Nothing in this document is a claim about how any run behaves.
- **No edits to `src/`, `proofs/`, or any existing file.** By design.
  Several findings have obvious one-line fixes and none are applied: whether the
  code or the proof is the intended science is the owner's call.
- **No re-derivation of the matrix counts**, per the review's scope.
- **No full test suite run.** The scripts import and call individual modules.
  A green suite would not have detected D-JAWP-1, D-SWIP-1 or D-WSR-2 — three of
  the findings here are precisely about numbers nothing asserts.

**Open, escalated, still the owner's decision:** D-WSR-3 (`_stiefel_retract`'s
sign convention; `strict=False` xfail at
`tests/test_training_state_guards.py:508`), and the WSD/JAWP opposition
(D-WSD-6 and the 2026-10-08 addenda), whose five resolution options are in
`proofs/wsd.md:255-300`. Neither is a defect this document recommends fixing;
both are decisions.