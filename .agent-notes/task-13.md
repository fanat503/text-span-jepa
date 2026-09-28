# TASK-13 — feature-interference control is disjoint and selection is seeded

- **status**: done
- **files**:
  - `src/interp/feature_composition.py` (only file owned; +86/-16)
  - `tests/test_feature_composition.py` (new, 253 lines, 12 tests)
  - nothing else touched. `src/interp/__init__.py` was NOT edited (owned by another
    agent on this tick) — see `не_сделано`.

---

## verify

### BEFORE (pristine `HEAD:src/interp/feature_composition.py` = the audited code)

```
$ git -C C:\dev\wt-13 cat-file blob HEAD:src/interp/feature_composition.py > src/interp/feature_composition.py
$ $PY C:\Users\Илья\AppData\Local\Temp\opencode\t13_measure.py
input_dim=32 latent_dim=256 k=16 n_features=50 n_top=100  (N = dataset size)
      N    mean_interference   n_treated  n_control
    100         9.161293e-10           -          -
   1000         2.888911e-02           -          -

determinism: 4 identical calls, N=1000, n_top=100
  call 0: 2.808498e-02
  call 1: 2.906169e-02
  call 2: 2.894283e-02
  call 3: 2.842134e-02
  spread: 3.48% of the minimum

N-comparability: same structure, disjoint uniform subsets of one 4000-row pool
  N=  200: mean_interference=1.473627e-02
  N= 1000: mean_interference=2.775857e-02
  N= 4000: mean_interference=4.722625e-02
```

`9.16e-10` at `N == n_top == 100` is float round-off of `x̄ − x̄` — the audit's
`1.4e-10` reproduced in kind. The 3.48 % call-to-call spread reproduces the
audit's non-determinism (1.840e-2 / 1.886e-2 / 1.940e-2 / 1.831e-2).

### AFTER

```
$ $PY C:\Users\Илья\AppData\Local\Temp\opencode\t13_measure.py
input_dim=32 latent_dim=256 k=16 n_features=50 n_top=100  (N = dataset size)
      N    mean_interference   n_treated  n_control
    100         4.103390e-02          50         50
   1000         3.156435e-02         100        900

determinism: 4 identical calls, N=1000, n_top=100
  call 0: 3.156435e-02
  call 1: 3.156435e-02
  call 2: 3.156435e-02
  call 3: 3.156435e-02
  spread: 0.00% of the minimum

N-comparability: same structure, disjoint uniform subsets of one 4000-row pool
  N=  200: mean_interference=2.962554e-02
  N= 1000: mean_interference=3.130135e-02
  N= 4000: mean_interference=4.724188e-02
```

### side by side

| quantity | before | after |
|---|---|---|
| `mean_interference`, **N = 100** (= `n_top`) | `9.16e-10` (zero) | **`4.10e-02`**, `n_treated=50 n_control=50` |
| `mean_interference`, **N = 1000** | `2.89e-02` | **`3.16e-02`**, `n_treated=100 n_control=900` |
| determinism, 4 identical calls @ N=1000 | 3.48 % spread (`2.81e-2…2.91e-2`) | **0.00 %** spread (all four `3.156435e-02`) |
| same pool, N = 200 / 1000 / 4000 | `1.47e-2 / 2.78e-2 / 4.72e-2` (1.9× between N=200 and N=1000) | **`2.96e-2 / 3.13e-2 / 4.72e-2`** (1.06×) |

### tests

```
$ $PY tools\rt.py tests\test_interp.py tests\test_feature_composition.py
rt.py: ... -m pytest -q --no-header -p no:cacheprovider tests\test_interp.py tests\test_feature_composition.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 106 items

tests\test_interp.py ................................................... [ 48%]
...........................................                             [ 88%]
tests\test_feature_composition.py ............                           [100%]

============================= 106 passed in 6.82s =============================
```

`tests/test_interp.py` alone (the file the card names as the gate) was **94 passed
in 9.18 s** before my change and **94 passed in 7.88–9.18 s** after — unchanged
and still fast. The new file is **12 tests in ~2.8 s**; nothing exceeds the budget.

### lint / format

```
$ $PY -m ruff check . --output-format concise
All checks passed!
ruff exit: 0

$ $PY -m black --check .
All done! ✨ 🍰 ✨
101 files would be left unchanged.
```

---

## disjointness — how it is guaranteed and asserted

**Guarantee (implementation).** One function is the single source of the split,
`FeatureInterferenceScore.treated_control_split(act_vals, n_top)`:

```python
N = act_vals.numel()
n_treated = min(int(n_top), N // 2)      # the cap: a control always exists
treated_idx = act_vals.topk(n_treated).indices
control_mask = torch.ones(N, dtype=torch.bool, device=act_vals.device)
control_mask[treated_idx] = False        # the control is the complement, by construction
control_idx = control_mask.nonzero(as_tuple=True)[0]
```

The `N // 2` cap is what makes `N == n_top` well-posed: without it the treatment
would be all `N` rows and there would be no sample left to act as a control.
`compute()` consumes it (`z_control = z[control_idx]`) and reports
`n_treated` / `n_control` on every return path, including the empty and
`except` branches.

**Assertion (tests).** Four independent angles, all in
`tests/test_feature_composition.py`:

1. `TestTreatedControlSplit::test_split_is_disjoint_and_covers_the_dataset` —
   set intersection is empty, union is all rows, sizes add up.
2. `test_treatment_is_capped_at_half_the_dataset` — the `N == n_top` case: 50/50
   split, disjoint, and the treatment really is the strongest half.
3. `TestControlDisjointFromTreatment::test_compute_reports_a_non_overlapping_split_for_every_feature`
   — runs the *public* API with `return_indices=True` and asserts, for **every**
   tested feature, that `treated_idx ∩ control_idx == ∅` and
   `treated_idx ∪ control_idx == range(N)`.
4. `test_score_is_computed_from_the_reported_split` — recomputes `min`/`max`/`mean`
   from the returned index sets and demands bit-identical agreement. This is the
   one that closes the loop: a clean report with a dirty computation fails here.

---

## determinism — the measurement

- `feature_idx = torch.randperm(M, generator=gen)[:n_test]` with
  `gen = torch.Generator().manual_seed(seed)`. A **private** generator; the
  process-global stream is never consumed.
- `seed=0` is a new keyword on `compute` (backwards compatible).
- **Four identical calls at N=1000, `n_top=100`:** before
  `2.808498e-02 / 2.906169e-02 / 2.894283e-02 / 2.842134e-02` (3.48 % spread);
  after **all four `3.156435e-02`**, 0.00 % spread.
- `TestDeterminism::test_global_rng_is_not_consumed` asserts the private-generator
  property directly: it churns the global stream (`torch.manual_seed(12345);
  torch.randn(500)`) between two calls and requires bit-identical output.

**`src/utils/seed.py` — decision, stated as required.** I did **not** import it,
and I recommend against it. `seed_everything` exists to seed the *process-global*
streams (`random`, `numpy`, `torch.manual_seed`) — which is precisely the thing
this defect is about not touching. The established convention in `src/interp/` is
a local generator: `causal_scrubbing.py:121`, `causal_intervention.py:312/615`,
`information_theory.py:169`, `disentanglement.py:222`, and even
`feature_composition.py:152` in the same file all do
`torch.Generator().manual_seed(seed)` inline. Following that convention keeps
`compute` a pure function of its arguments and keeps audit finding **I16**
("`src/utils/seed.py` is never imported by anything in `src/interp/`") from being
silently *worsened* by making a library module reseed the caller's process.

---

## diff-stat

```
$ git -C C:\dev\wt-13 diff --stat campaign/integration...HEAD
 src/interp/feature_composition.py | 102 ++++++++++++---
 tests/test_feature_composition.py | 253 ++++++++++++++++++++++++++++++++++++++
 2 files changed, 339 insertions(+), 16 deletions(-)
```

Commit `1658032`. Not pushed. Working tree clean.

---

## mutation-verdict

**The contamination is detected. Three mutations were applied to the fixed
implementation, run, and reverted. Two of the three mutants are caught by two
tests each.**

### Mutation A — the control is allowed to overlap the treatment
`z_control = z[control_idx]` → `z_control = z` (the original defect, verbatim):

```
FAILED tests/test_feature_composition.py::TestControlDisjointFromTreatment::test_score_is_computed_from_the_reported_split
FAILED tests/test_feature_composition.py::TestComparableAcrossDatasetSize::test_score_does_not_scale_with_n_top_over_n
======================== 2 failed, 104 passed in 9.84s ========================
E   AssertionError: score moved by 0.438x between N=240 and N=2400 on identical structure; n_top/N attenuation is
still present
```

A variant was also run that keeps the *reported* split clean and only corrupts
the *use* of it, to prove assertion #4 is doing the work — same two failures.

### Mutation C — the `N // 2` cap removed
`n_treated = min(int(n_top), N // 2)` → `min(int(n_top), N)`, which restores the
original `N == n_top` degeneracy:

```
FAILED tests/test_feature_composition.py::TestTreatedControlSplit::test_treatment_is_capped_at_half_the_dataset
FAILED tests/test_feature_composition.py::TestTreatedControlSplit::test_tiny_dataset_degrades_without_crashing
FAILED tests/test_feature_composition.py::TestNotTriviallyZeroAtNTopEqualsNTreated::test_score_is_not_float_round_off_when_n_top_equals_n
======================== 3 failed, 9 passed in 2.99s =========================
```

### Mutation B — selection reverted to the global RNG
`torch.randperm(M, generator=gen)` → `torch.randperm(M)`:

```
FAILED tests/test_feature_composition.py::TestDeterminism::test_same_seed_gives_identical_output
FAILED tests/test_feature_composition.py::TestDeterminism::test_global_rng_is_not_consumed
======================== 2 failed, 10 passed in 2.85s ========================
```

### Threshold justification for the comparability band
The assertion is `0.65 < score(N=240)/score(N=2400) < 1.25` on one fixed pool
with block structure. Measured over **six feature subsamples × three
`n_features` settings**, old and new implementations side by side:

```
n_features=20
  NEW ratios: ['0.768', '0.974', '0.790', '0.902', '0.795', '0.893']
  OLD ratios: ['0.468', '0.593', '0.481', '0.549', '0.484', '0.544']
  NEW min=0.768   OLD max=0.593
n_features=25
  NEW ratios: ['0.719', '0.883', '0.739', '0.917', '0.877', '0.922']
  OLD ratios: ['0.438', '0.537', '0.450', '0.558', '0.534', '0.561']
  NEW min=0.719   OLD max=0.561
n_features=50
  NEW ratios: ['0.900', '0.910', '0.784', '0.957', '0.895', '0.969']
  OLD ratios: ['0.548', '0.554', '0.477', '0.582', '0.545', '0.590']
  NEW min=0.784   OLD max=0.590
```

New spans `[0.719, 0.974]`, old spans `[0.438, 0.593]`. `0.65` sits in the
`0.593 → 0.719` gap, so it is not a fitted constant. A mild downward bias
survives at small N and is **not** a bug: with `n_top` fixed, the top-100 rows
of a 2400-row sample are genuinely more extreme than the top-100 of a 240-row
one. That is a property of top-*k* selection, not of the contamination; it is
documented in the test docstring.

**Honest caveat.** `test_score_is_computed_from_the_reported_split` is the test
that makes the fix *unenforceless-proof*; without it, Mutation A is caught by
exactly one test, and a future edit that preserved the reported split while
computing over the full dataset would have slipped through on the
disjointness assertions alone. I added it precisely because the first mutation
run failed only one test.

---

## reported-not-fixed — `systematic_composition_test`

All four findings below are in the class I do own, but they are outside the
card's two defects, so per instructions they are **reported, not fixed**.

**R1 — `best_cos` is a max over `cand_b` scored on the selection data.**
`src/interp/feature_composition.py:141-166`. `cand_a` and `cand_b` are chosen by
`_find_discriminating_feature` on `a_reps`/`b_reps` vs `neither_reps`; the score
`cos_real = cos(z_a[cand_a[0]] ⊕ z_b[b_idx], z_ab)` is then maximised over
`b_idx ∈ cand_b` and reported as `results.append(best_cos)`. There is no held-out
split anywhere in this function. `n_feature_search` is 10 by default, so up to
10 of 10 candidates are tried and the maximum of 10 noisy numbers on the same data
is an order statistic, not an estimate. The placebo calibration partly compensates
(same search run on shuffled `ab_reps`), but `mean_composition_score`,
`max_score` and `min_score` themselves remain best-of-10 winners. The code
already says so in a comment at L179-181 ("the raw score is selection-biased
upward by best-of-k search") and reports `mean_advantage` next to it — the fix is
a decision about which of those keys downstream consumers may read, and
`compare_models` (L238-241) reads precisely the biased one
(`jepa_result["mean_composition_score"]`) to set `jepa_more_compositional`.

**R2 — the placebo is computed only for the winning `b_idx`.**
`L157, L163-165`. `best_placebo` is assigned *inside* the `if cos_real > best_cos`
branch, so it is the placebo of the argmax, not the placebo of a search over the
same candidate set. `mean_advantage` therefore compares a max-over-10 against a
single draw, and `mean_placebo` understates the placebo distribution's spread. The
matched procedure would record the placebo at every `b_idx` and take its max too.

**R3 — `cand_a[0]` is fixed, so only `B` is searched.**
`L158-160`. The loop varies `b_idx` over `cand_b` but always uses
`cand_a[0]`; the `n_feature_search`-th best A-feature is never tried, despite
`_find_discriminating_feature` returning a ranked list for A on the same terms as
for B. The asymmetry is undocumented in the docstring, which says "find the SAE
features that best separate A vs not-A, and B vs not-B" (plural, both searched).
Either the docstring or the loop is wrong; which one is a human decision.

**R4 — `z_a`, `z_b`, `z_ab` are SAE encodings of mean-pooled vectors.**
`L144-146`: `self.sae.encode(a_reps.mean(dim=0, keepdim=True))[0]`. Each is a
*single* code vector for the centroid of a property set, not a per-example code.
Consequences: (a) the TopK activation inside `SparseAutoencoder.encode` selects the
top 12 coordinates of a single mean vector, so the "sparse feature" being
manipulated is largely an artefact of which coordinates win a TopK over one
centroid; (b) `feature_arithmetic_test` (L82-84) does the same mean-pooling and
the two paths are not consistent with each other about what a code is; (c) any
per-example structure is destroyed before the arithmetic, so the test cannot
distinguish "features compose" from "centroids compose"; (d) the same
mean-pooling appears again at L154 for the placebo, which at least keeps the
placebo comparable.

None of the four is fixed here. Each needs a decision about what the module
claims to measure before it can be corrected.

---

## не_сделано / риски

**не_сделано**

1. `src/interp/__init__.py` was **not** updated — it is on another agent's
   ownership list for this tick. `FeatureInterferenceScore` is already exported
   there (L56, L155), and `treated_control_split` is reachable as
   `FeatureInterferenceScore.treated_control_split`, so nothing is unreachable;
   but if that owner wants it in the flat namespace, they must add it.
2. `TASKS.md` card 13 was not marked `done` and
   `docs/plans/2026-09-27-wave1-audit-findings-interp.md` finding I15 was not
   struck through. Both files are shared campaign state, not mine to touch
   unilaterally — the coordinator owns the bookkeeping.
3. `FeatureCompositionScore` / `systematic_composition_test` is untouched
   (R1-R4 above).
4. Pre-existing `except Exception:` in `compute` still returns
   `mean_interference = inf` on any internal error. Left alone deliberately, but
   note it can mask a future regression: my tests guard against it by asserting
   `math.isfinite(...)` and `n_features_tested > 0`, not by touching the handler.

**риски**

1. **The control is a matched complement, not a random draw.** `control` is
   the low-activation tail of feature *fi*. If *fi* and *fj* are positively
   correlated, the tail has low *fj*, which *inflates* the measured interference
   relative to a random held-out control. I chose this because the card says
   "control from samples EXCLUDED from the treatment set" and because a matched
   control removes *fi*'s own activation rank as a confound (`fi` is excluded
   from `other_idx` anyway). The alternative — sample the control at random from
   the non-treated rows — would give a smaller, unconfounded number. **The two
   are not comparable to each other and any historical number in this repo
   predates both.** This is a second-order change to the estimand and belongs to
   whoever owns the hypothesis test, not to a defect fix.
2. **The `N // 2` cap silently changes `n_treated` for small datasets.** At
   `N = 100, n_top = 100` the treatment is 50 rows, not 100. Any caller that
   assumed `n_treated == n_top` will be wrong; the new `n_treated` key exists so
   this is visible, but no caller reads it yet. Grep found exactly one caller in
   the tree (`tests/test_interp.py:538`, `N=50 n_top=10`, unaffected — 10 ≤ 25).
3. **The comparability band is measured on synthetic block structure.** A real
   JEPA representation set may have heavier tails, and the residual small-N
   bias (top-*k* extremeness, documented above) could exceed the ±25 % upper
   bound. If that happens the test is telling you something true about the
   estimator, and the fix is a different estimand — not a wider tolerance.
4. **`torch.randperm(M, generator=gen).to(z.device)` builds the permutation on
   the CPU then copies.** Correct on GPU, but a `device`-native generator would
   be marginally cleaner. Left as-is because it matches the module's existing
   pattern (`:153`) and keeps the private-generator property obvious.
5. The measurement script lives outside the repo
   (`%LOCALAPPDATA%\Temp\opencode\t13_*.py`), so the before/after numbers in this
   report are reproducible but not checked in. If the coordinator wants them as a
   regression artefact, they should be promoted into
   `tests/test_feature_composition.py` as a slow-marked test — which I did not do
   because `rt.py` has a hard 90 s cumulative budget and this card is not mine.
