# TASK-40 — WSD / JAWP opposite optima on one parameter

Branch `agent/task-40` @ `ac7d034`, worktree `C:\dev\wt-40`, base `agent/wave-7`.
Files touched: `src/models/wsd.py`, `src/models/jawp.py`, `proofs/wsd.md`,
`proofs/jawp.md`. Nothing else. `mechanisms.py`, `train.py`, `config/**`,
`jepa.py`, `tests/**` untouched.

## VERDICT: it is a GENUINE CONFLICT, not a false alarm — and it is
## sub-threshold at every shipped weight.

The "two mechanisms, two different quantities, so no conflict" escape does not
apply here, because both losses are in the **same `total_loss`** and hit the
**same tensor** in the **same backward pass**. It is not academic either: past
a measured threshold it destroys JAWP's objective. But at the weights that
actually ship, nothing observable is wrong today.

### The coupling (code, not inference)

`src/models/jepa.py:1086`:

```python
Q_ws = self.jawp.workspace_Q[:, :k_active]   # LIVE view — no .data()
return self.wsd.compute_drift(Q_ws, h_target=h_target.detach(), step=current_step)
```

`h_target` is detached; `Q_ws` is **not**. `jepa.py:862` adds
`config.lambda_wsd * loss_wsd` to `total_loss`. So `dL_WSD/dQ` is a live
gradient on `jawp.workspace_Q[:, :k_active]` — the identical columns JAWP
optimises. Verified empirically: `workspace_Q.grad` differs between WSD-on and
WSD-off runs at the same seed.

### The two gradients

| | loss w.r.t. Q | ∂L/∂Q | optimum |
|---|---|---|---|
| JAWP `jawp.py:375` | `‖Qᵀ(z_pred−z_target)‖² = tr(QᵀΣ_res Q)` | `2Σ_res Q` | **bottom**-k of Σ_res |
| WSD `wsd.py:258` | `2k − 2‖QᵀQ_tgt‖²` | `−4λ·P_tgt·Q` | **top**-k of Cov(h_target) (`wsd.py:206`) |

Σ_res and Cov(h_target) are different matrices, so the optima need not be
distinct — but **nothing in either proof states a relation between them**, and
`jawp.py:100-102` (DESIGN DECISION 3) *removed* a
`β‖(I−QQᵀ)z_target‖²` term for pushing Q toward high-variance directions.
WSD's `∂L/∂Q` is that same pull, same units, same parameter. The author had
already identified this exact force and taken it out of JAWP; WSD puts it back
under a different name.

### Evidence 1 — the optima are data-dependent, and the theory's own regime is the worst case

`overlap²(bottom-k(Σ_res), top-k(Cov_tgt))`, D=8, k=2:

| constructed Σ_res | overlap² |
|---|---|
| `Cov_tgt + 4I` — isotropic predictor error | **0.0000** |
| reversed spectrum (variance ⇒ unpredictability) | 1.0000 |
| predictable low-variance directions | **0.0000** |
| `Σ_res = I` (degenerate) | 1.0000 |

Isotropic error is the regime the proofs implicitly assume — and it gives
**exactly zero** overlap. The mechanisms agree only in the regimes the theory
argues against.

### Evidence 2 — in situ on the running model

`TextSpanJEPA`, D=64, k=6, `jawk_init=random`, curriculum off. `∂L_WSD/∂Q`
isolated by differencing `workspace_Q.grad` WSD-on vs WSD-off at the same seed:

```
step  cos    ratio ||dWSD/dQ||/||dJAWP/dQ||
  0   -0.5221   3.1187      6   -0.4715   2.2245
  1   -0.4999   2.5273      7   -0.3588   2.2291
  2   -0.3038   2.9813      8   -0.5933   1.8862
  3   -0.5722   2.2670      9   -0.5337   2.1856
  4   -0.4835   2.3973     10   -0.4744   2.0272
  5   -0.3843   2.5417     11   -0.5056   2.4764
=> 12/12 steps NEGATIVE.  median ratio 2.3973  =>  crossover lambda_wsd ~ 0.417
```

**Every measured step is opposed**, and WSD's raw Q-gradient is ~2.4× JAWP's.
Crossover **λ_wsd ≈ 0.42**.

### Evidence 3 — where the workspace actually lands

60-step runs, `wsd_sync_interval=1`, everything else equal. `ov_top` =
`overlap²(span(Q), top-k EMA Cov(h_target))`; chance = k/D = 0.094.

| λ_wsd | ov_top | `jawk_pca_alignment` | `jawk_predictive_relevance` | total loss |
|---|---|---|---|---|
| 0.0 | 0.0494 | 0.0633 | 0.3251 | 1.151 |
| 0.001 | 0.0497 | 0.0634 | 0.3249 | 1.162 |
| **0.01** | 0.0525 | 0.0637 | 0.3237 | 1.266 |
| **0.1** | 0.0913 | 0.0703 | 0.2991 | 2.267 |
| 1.0 | 0.3705 | 0.1837 | **0.0000** | 9.095 |
| 10.0 | 0.3866 | 0.2005 | **0.0000** | 75.227 |

At λ ≥ 1, WSD drags span(Q) ~4× above chance onto the target's top-variance
subspace and JAWP's own predictive-relevance diagnostic collapses to zero — the
workspace stops being predictable at all.

### Severity: live but sub-threshold, and latent in every headline config

- `defaults.yaml:237,241` — `use_wsd: true`, `lambda_wsd: 0.01` (≈2.4% of the Q-gradient).
- **Every** runnable config sets `use_wsd: false`: `config/scaling/*.yaml`
  (all 4), `config/wikitext/*`, `config/tinystories/*`, `config/kaggle/*`,
  `config/ablations/{none,sigreg_only,no_wsd}.yaml`.
- The only arm that enables it is `config/ablations/wsd_on.yaml`
  (`lambda_wsd: 0.1`) — **4× below** the crossover.

So the trap is armed but no shipped arm is near it, which is exactly why no
ablation sweep ever surfaced this.

### Second, independent defect on the same seam

`MechanismBundle.forward` (`mechanisms.py:475`) passes
`self.jawp.workspace_Q.data[...]` — **detached**. Verified:

```
MechanismBundle info['wsd_loss'].requires_grad = False
MechanismBundle info['jawp_loss'].requires_grad = True
```

`MechanismBundle` is constructed nowhere in `src/` outside `mechanisms.py`, so
`TextSpanJEPA` is the live path and `mechanisms.py` is a divergent copy of the
same wiring. Anyone using the documented bundle API gets a WSD loss that cannot
train anything, with no warning. **`mechanisms.py` is outside my boundary** —
reported, not touched.

## OPTIONS (recorded in `proofs/wsd.md`, none applied)

1. **Keep both.** Zero change. Grid is below crossover so nothing breaks
   today. Cost: the trap stays armed — a sweep to λ∈{1,10} silently replaces
   JAWP's workspace with the target's top-variance subspace, and
   `jawk_predictive_relevance` reads 0.0 rather than raising.
2. **Make them agree by construction** (take Q_tgt from bottom-k, or from
   Σ_res). Opposition vanishes. Cost: WSD stops being an *independent*
   synchronisation monitor and becomes a second copy of JAWP's objective; the
   Davis-Kahan/STA section and the "top-k" definition both need rewriting, and
   the Drift Bound becomes vacuous.
3. **Detach Q; keep WSD a pure monitor.** One line. **NOT RECOMMENDED.** The
   penalty is precisely what supplies λ in `dΔ/dt = ν − λΔ`; without a gradient
   Δ is unbounded, so the Drift Bound is false *by construction*. Worse,
   `test_sterility.py` still passes — `test_wsd_component_nonzero` only asserts
   the logged number `> 0`, and a gradient-free term still logs a number. This
   is the one option that turns a measured conflict into an **undetectable
   no-op**.
4. **Give WSD its own projection.** Clean separation, both keep their meaning.
   Cost: a new checkpointed tensor (`train.py:277-289` rebuilds the
   `workspace_Q` family by explicit name), a new consistency loss, and two
   notions of "the workspace" while SWIP/RDC/WSR all read `jawp.workspace_Q`.
   Largest blast radius.
5. **Guard it, change nothing.** Record the crossover (done), cap `lambda_wsd`
   in `defaults.yaml` with a comment citing the section, and add a regression
   test that fails when a config crosses the threshold. Cost: a test file plus
   a `defaults.yaml`/`config/**` edit — both outside my boundary. Makes the
   trap explicit without pre-empting the science.

**My recommendation: 5 now, and let the owner choose between 2 and 4 when they
decide what WSD is for.** I did not apply any of them.

## THE TWO DEAD CAPABILITIES — what I did, and what I could not

`detect_workspace_dimension` (Spectral Gap #3) and `grassmann_retract`
(Grassmann Optimization #4) have **no caller in `src/`** — confirmed by grep
over `src/`, `tests/`, `baselines/`, `tools/`, `scripts/`. The wired
counterparts are `stiefel_retract` (`jawp.py:269`, called from
`mechanisms.py:532` and `train.py:1470`) and `predictive_rank_loss`
(`jepa.py:1041`).

I **did not delete them and did not half-wire them.** Both actions are blocked
by my file boundary, and I would rather report that than fake it:

- Deleting either breaks `tests/test_jawp.py` (grassmann) and
  `tests/test_v025_integration.py` (spectral gap). `tests/**` is not mine.
- Wiring either *usefully* requires an argument reachable from config, and
  `jepa.py:430-439` passes a fixed kwargs set to `JAWPModule`, so a new
  constructor parameter is unreachable without editing `jepa.py` — forbidden.
- A log-only wire is possible inside `jawp.py` (a new `info` key propagates to
  `loss_dict` automatically via `jepa.py:895-896`) but it adds an
  unconfigurable O(D³) eigendecomposition to every forward and creates a new
  key nobody asked for. That is speculative work.

What I did instead, inside my boundary: **measured what each is worth**, so the
owner's choice is informed rather than guessed, and **marked both honestly**
in `jawp.py` and `proofs/jawp.md` so the count of 16 stops overstating them.

**`detect_workspace_dimension` — WIRE it. It is not a dead formula.** Against
a constructed ground truth (D=64, N=2000, known k*) it recovers k* **exactly**
at k* ∈ {1,2,3,6,12,24}, with **zero spread across 5 seeds**, at ~1.7 ms per
call. Caveat found: the `min(k_gap, k_mp)` combination only worked because
`k_mp` came out at 25-26 (a loose upper bound); `c_est = 0.5` is hardcoded and
`sigma2_est` is the median of the top half of the residual spectrum, so the MP
branch is unprincipled and would silently truncate a correct detection if it
ever under-estimated. The gap branch carries the method.

**`grassmann_retract` — WIRE it, but do not swap it in on this evidence.**
Measured over 60 steps (D=64, k=6): the two produce loss trajectories equal to
~7e-6 and `overlap²(span(Q))` = **1.0000** — interchangeable in practice at
this scale. But its gauge term `Q_active @ (Q_activeᵀ @ grad)` is not the
symmetric Riemannian correction `stiefel_retract` applies, and `jawp.md`
Theorem 3 is stated for `stiefel_retract`. Swapping changes which quantity the
proof calls "the retraction" without changing any measured behaviour — an
owner decision, not a cleanup. Also note the measurement's scope: one scale,
60 steps; the disagreement is between a symmetric and a non-symmetric tangent
projection, and I would not generalise it.

## WHAT I CHANGED

Code: **comments only.** No loss, no weight, no default, no signature changed.
- `wsd.py` (+23) — at `resync_target_workspace`, records that "top-k" is
  load-bearing and opposes JAWP, with the measured cosine and crossover, the
  mutation result that proves the direction is unpinned, and a pointer to the
  proof section.
- `jawp.py` (+62) — a banner after the Courant-Fischer corollary naming the
  antagonism and that it is an open science call; `⚠ NO CALLER IN src/` blocks
  on both dead capabilities with their measurements and the wire-don't-delete
  recommendation.
- `proofs/wsd.md` (+188) — the full section: coupling, gradients, the
  data-dependence table, the 12-step cosine table, the λ_wsd sweep, which
  weights ship, the `mechanisms.py` detached-loss divergence, five options.
- `proofs/jawp.md` (+82) — reciprocal summary, an "Unreached capabilities"
  table with both measurements, and a fix for the stray `which<` typo at the
  Corollary (JAWP ≤ PCA).

I did **not** touch `proofs/IMPLEMENTATION_STATUS.md`: it is a shared matrix
and a forbidden collision hotspot, and my change moves no mechanism between
Verified / Divergent / Gaps (no behaviour changed). The matrix is stale in the
sense that this conflict was never a row — TASK-41 owns that reconciliation,
and `proofs/wsd.md`'s addendum banner points at it.

## VERIFY (pasted)

Baseline, before any edit:

```
$ & $PY tools\rt.py tests/test_wsd.py tests/test_jawp.py tests/test_sterility.py
collected 105 items
tests\test_wsd.py ................                                       [ 15%]
tests\test_jawp.py .....................................................   [ 65%]
....                                                                     [ 69%]
tests\test_sterility.py ................................                 [100%]
======================= 105 passed, 1 warning in 10.99s =======================
```

After the change, final run:

```
$ & $PY tools\rt.py tests/test_wsd.py tests/test_jawp.py tests/test_sterility.py
======================= 105 passed, 1 warning in 11.32s =======================
```

Lint and format:

```
$ & $PY -m ruff check src/models/wsd.py src/models/jawp.py
All checks passed!
$ & $PY -m black --check src/models/wsd.py src/models/jawp.py
2 files would be left unchanged.
```

## MUTATION VERDICT

| # | mutation | caught by | verdict |
|---|---|---|---|
| M1 | `wsd.py:206` `eigenvectors[:, -k:]` → `eigenvectors[:, :k]` — i.e. "just make the two mechanisms agree" | **NOTHING** — 105 pass | **the antagonism direction is completely unpinned.** This is why it went untriaged for the life of the mechanism, and why the fix cannot be a test-only guard. |
| M2 | `mechanisms.py:475` `.data` → live view — fixes the detached-loss divergence | **NOTHING** — 105 pass | the `.data`/`jepa.py` divergence is unpinned |
| M3 | delete `grassmann_retract` body | 5 tests in `tests/test_jawp.py::TestGrassmannOptimization` fail | test-pinned, but **no `src/` caller** |
| M4 | delete `detect_workspace_dimension` body | **NOTHING** in my 3-file verify set — 105 pass. `tests/test_v025_integration.py::test_spectral_gap_detection` and `::test_spectral_gap_returns_fallback_on_small_data` fail (2 failed, 43 passed, verified with `--slow`) | test-pinned only *outside* the required verify set; **no `src/` caller** |

All four mutations reverted; `git status --short` empty, `git diff HEAD` empty,
`git log -1` = `ac7d034`.

**Verdict: mutation testing says the conflict is real AND that no test in this
repo defends it in either direction.** M1 is the important one — the single
most consequential scientific change available on this seam (silently making
WSD agree with JAWP) is invisible to the entire required verify set. M3/M4 are
the concrete form of the count mismatch: two numbered capabilities held alive
by tests, reached by nothing in `src/`. No skip, no xfail, no `--no-verify`;
the one `--slow` use was rt.py's own refusal guard, on a file outside the
required set, to complete M4.

## РИСКИ

1. **I did not resolve the conflict, by design.** The owner must choose between
   options 2, 4 and 5. Nothing I changed makes any of them easier to get wrong
   — the crossover number is recorded in both proofs and both module headers.
2. **Option 3 is a trap.** Detaching Q in WSD looks like the one-line fix and
   is the only one that silently disarms `test_sterility.py`'s WSD gate. It is
   flagged as not-recommended in `proofs/wsd.md`; if someone picks it anyway,
   the suite will not object.
3. **My crossover (0.42) is scale- and schedule-specific** — D=64, k=6, 60
   steps, AdamW lr=1e-3, uniform random tokens, CPU, no real corpus. The
   *existence* of the opposition is a code fact; the *number* will move with
   embed_dim, k, the optimizer and the data. Treat 0.42 as "same order of
   magnitude as the shipped 0.01/0.1", not as a portable constant.
4. **The `min(k_gap, k_mp)` fragility in `detect_workspace_dimension`** is
   recorded but not fixed. If the owner wires that capability, fix `c_est`
   and the MP combination first, or it will truncate correct detections on
   data unlike my synthetic control.
5. **`mechanisms.py`'s detached WSD loss is still broken** and still out of my
   boundary. Any consumer of `MechanismBundle` gets a WSD loss that cannot
   train. It is now written down in `proofs/wsd.md` and here, but nobody has
   been assigned the one-line fix.
6. **No test was added.** A regression test for the crossover (option 5) is
   the natural follow-up and needs a home in `tests/` plus a `lambda_wsd`
   ceiling in `defaults.yaml` — both outside this card.
7. **`k_mp` sensitivity note:** in my control `k_mp` came out 25-26 while
   `k_gap` was correct. The method works because `min()` picks the right one;
   that is luck of the construction, not a property. Do not read my recovery
   result as validating the MP branch.
