# TASK-15 — the Benjamini-Hochberg adjusted p and the BH significance set now agree

- **status**: done
- **files**:
  - `src/interp/statistical_tests.py` (only source file owned; +51/−22)
  - `tests/test_statistical_tests.py` (NEW, 349 lines, 19 tests)
  - nothing else touched. No file outside `src/interp/statistical_tests.py` was
    edited — `layer_analysis.py`, `feature_composition.py`, `ground_truth.py`,
    `run_comparison.py` untouched.
- **branch**: `agent/task-15`, commit `39acf9b`. Not pushed. Working tree clean.

---

## the defect, reproduced verbatim

`benjamini_hochberg` computed **two outputs by two different rules**:

| output | rule | source (pre-fix) |
|---|---|---|
| `corrected` | naive `p * n / rank`, capped at 1 | L286-289 |
| `significant` | step-up: largest `k` with `p_(k) <= k/n * alpha` | L279-283 |

The card's repro, measured on the pristine worktree file
(`sys.path.insert(0, r"C:\dev\wt-15")` first, so `src` is the worktree and not
the stale `pip install -e .` clone — the hazard this tick):

```
src loaded from: C:\dev\wt-15\src\interp\statistical_tests.py

card repro p=[0.03,0.049]
  raw        [0.03, 0.049]
  corrected  [0.06, 0.049]          <-- the naive formula
  reference  [0.049, 0.049]         <-- the true BH adjusted p
  sign idx   [0, 1]
  corrected<=alpha vs in significant -> [(0, 0.06, 0)]     <-- DISAGREE
  corrected == reference BH?   False
```

Index 0 is reported as a discovery while its own adjusted p reads `0.06 > 0.05`.
`MetricComparisonReport.generate` published exactly that pair
(`p_value_bh = 0.06`, `significant_bh = True`) — verified end to end, see the
test named `test_the_card_p_values_survive_the_report_unchanged`.

The same probe exposed two more consequences of the same omission, both
measured:

- **non-monotone adjusted p.** `p = [0.551, 0.708]` (n=2) gave
  `corrected = [1.0, 0.708]` — `q_(1) > q_(2)`, which is impossible for an
  adjusted p-value. Five of five all-null p-values `0.6` gave
  `corrected = [1.0, 1.0, 1.0, 0.75, 0.6]` instead of `[0.6]*5`.
- **ties split arbitrarily.** `p = [0.02, 0.02, 0.02]` gave
  `corrected = [0.06, 0.03, 0.02]`. Three identical hypotheses, three different
  adjusted p-values, decided by stable-sort order.

---

## the fix

The BH adjusted p-value is the **running minimum taken from the top rank**:

```
q_(i) = min over j >= i of  min( p_(j) * n / j, 1 )
```

and Benjamini & Hochberg (1995, eq. 2.4) prove `{ i : q_i <= alpha }` is
*exactly* the step-up rejection set. So the two outputs are not two opinions,
they are one decision written twice — which is why deriving them separately was
the bug.

`src/interp/statistical_tests.py:262-343` now:

1. sorts ascending (stable, so input order breaks ties),
2. builds **one** array `scaled[j] = min(p_(j) * n / j, 1.0)`,
3. takes the running minimum from the top of `scaled`,
4. scatters back into the caller's order,
5. reads `significant = sorted(i for i : corrected[i] <= alpha)`.

Step 2 is deliberate, beyond the mathematical point: both outputs now read the
*same float*, so they cannot drift apart through floating-point reassociation
either. The old code compared `p <= rank/n*alpha` in one place and computed
`p*n/rank` in another; those two expressions are not bitwise equal in general,
so even a correct pair of rules could have produced a boundary disagreement.

Also in scope of the same function: p-values are coerced with `float(x)` so
numpy scalars and 0-d tensors cannot leak into the arithmetic, and the
empty-input return now carries `n_significant` / `n_total` like every other
return path (it previously omitted both keys).

### after, same probe

```
card repro p=[0.03,0.049]
  corrected  [0.049, 0.049]
  reference  [0.049, 0.049]
  corrected<=alpha vs in significant -> AGREE
  corrected == reference BH?   True

p=[0.01,0.03,0.04,0.50]  corrected [0.04, 0.053333, 0.053333, 0.5]   == reference, AGREE
all null p=0.6 x5         corrected [0.6, 0.6, 0.6, 0.6, 0.6]         == reference, AGREE
one real p=0.001 x5       corrected [0.005, 0.6, 0.6, 0.6, 0.6]       == reference, AGREE
monotone p=0.01..0.05 x5  corrected [0.05, 0.05, 0.05, 0.05, 0.05]     == reference, AGREE
non-monotone p x6         corrected == reference, AGREE
```

Six of six cases now equal the independent reference and agree with
`significant`. Only the last two changed value; the other four were already
correct, which is why the bug looked harmless in most cases.

---

## tests

`tests/test_statistical_tests.py`, 19 tests in three classes. Two independent
textbook transcriptions sit at the top of the file (`_reference_adjusted`,
`_reference_step_up`) written from the definition, plus `_naive_p_times_n_over_rank`
— the defective formula, kept so the tests can show the difference rather than
assert a constant.

| class | tests | what it pins |
|---|---|---|
| `TestBenjaminiHochbergAgreement` | 7 | the agreement identity, the reference match, the card repro, ties |
| `TestBenjaminiHochbergProperties` | 9 | monotonicity, `q >= p`, range, BH vs Bonferroni, prefix structure, input-order fidelity, `p == 0`, empty input |
| `TestBenjaminiHochbergOnRealPValues` | 3 | the module's own permutation p-values; the report surface |

**The load-bearing test** is
`TestBenjaminiHochbergAgreement::test_significant_set_is_exactly_corrected_at_or_below_alpha`.
It asserts `significant[i] == (corrected[i] <= alpha)` for every `i` over
7 sizes x 3 alphas x 25 draws = **525 p-value sets / 6,900 index checks**,
fixed seed `20260929`, all in-process arithmetic (no I/O, ~0.3 s).

Supporting tests, and why each is a *property* rather than a magic number:

- `test_adjusted_matches_the_reference_running_minimum` — 120 random sets,
  equality against the transcription. Fails loudly and points at the formula.
- `test_adjusted_is_monotone_in_the_sorted_order` — `q_(1) <= ... <= q_(n)`.
  This is a theorem about adjusted p-values, not a fitted band; the naive
  formula violates it (`[1.0, 0.708]`, above).
- `test_ties_receive_identical_adjusted_values` — three identical hypotheses
  must get one adjusted value. The naive formula gives three.
- `test_discovery_set_is_a_prefix_of_the_sorted_order` — the step-up structure.
- `test_results_are_returned_in_the_input_order` — reindexing the input
  reindexes the output by the same permutation. Catches a `corrected` list
  returned in sorted order.
- `test_p_values_may_arrive_as_zero` — a real permutation test can emit `p == 0`;
  it must not become negative or wrap.

**The end-to-end test** is
`test_the_card_p_values_survive_the_report_unchanged`. Real data cannot reach the
card's regime and I measured why before stubbing anything: `PairedPermutationTest`
floors its p at `1/(n_permutations+1)`, and every metric with a real effect hits
that floor together, so the two smallest p-values are never inside the window
`p_(1)*n/1 > alpha >= p_(2)*n/2`. Nine seeds x six metrics were checked
(`t15_probe2.py`); all nine produced two p-values of exactly `0.005` and four
nulls above `0.1` — the naive formula was never violated, so a real-data-only
end-to-end test would have been a **non-discriminating** test and I would not
have known. So the permutation test alone is stubbed with the card's
`[0.03, 0.049]`; the correction and the whole `MetricComparisonReport` body run
unstubbed. This test was red pre-fix (`0.06 != 0.049`).

A second end-to-end test, `test_a_report_never_shows_p_value_above_alpha_next_to_significant`,
runs the report on **real** data (6 metrics, 2 with genuine effects) and asserts
the same invariant on whatever p-values come out. It was green pre-fix — it is
an invariant guard on the public surface, not a discriminator. Reported as such.

No skip, no xfail, no `pytest.importorskip` used to dodge a failure: the
`importorskip("numpy")` / `("torch")` calls at the top of the real-p-value tests
are the same defensive shape already used elsewhere in the suite for optional
deps; both are hard requirements of this module and both are installed, so
neither branch is taken. Nothing in the file weakens an assertion.

---

## verify

### AFTER

```
$ $PY tools\rt.py tests\test_statistical_tests.py tests\test_interp.py
rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests\test_statistical_tests.py tests\test_interp.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 113 items

tests\test_statistical_tests.py ...................                      [ 16%]
tests\test_interp.py ................................................... [ 61%]
...........................................                              [100%]

============================= 113 passed in 4.66s =============================
```

`tests/test_interp.py` is the card's gate and is unchanged: 94 passed before my
change, 94 passed after, ~4.5 s total for both files. The 19 new tests cost
0.60 s of that.

### BEFORE (red, on the pristine file)

```
collected 18 items
tests\test_statistical_tests.py FFF.F.F...F.......                       [100%]
...
E   AssertionError: n=21 alpha=0.1 p=[0.0669..., 0.8568..., ...]: index 10 corrected=0.12721065640767332 alpha=0.1 sig=[10, 14]
E   assert (10 in {10, 14}) == (0.12721065640767332 <= 0.1)
E   AssertionError: assert 0.06 == 0.049
======================== 6 failed, 12 passed in 0.60s =========================
```

### downstream

```
$ $PY tools\rt.py tests\test_run_comparison.py
collected 16 items
tests\test_run_comparison.py ................                            [100%]
============================= 16 passed in 2.62s =============================
```

`benjamini_hochberg` has exactly one production caller
(`MetricComparisonReport.generate`, L520) and that report is never called
outside its own module, so the blast radius is nil. Verified by grep, not
assumed.

### lint / format

```
$ $PY -m ruff check src/interp/statistical_tests.py tests/test_statistical_tests.py --output-format concise
All checks passed!
ruff exit: 0

$ $PY -m black --check src/interp/statistical_tests.py tests/test_statistical_tests.py
All done! ✨ 🍰 ✨
2 files would be left unchanged.
black exit: 0
```

(`tests/test_statistical_tests.py` needed one `black` reformat pass before the
final run; `src/interp/statistical_tests.py` was already conformant.)

---

## diff-stat

```
$ git -C C:\dev\wt-15 diff --stat agent/wave-2...HEAD
 src/interp/statistical_tests.py |  73 ++++++---
 tests/test_statistical_tests.py | 349 ++++++++++++++++++++++++++++++++++++++++
 2 files changed, 400 insertions(+), 22 deletions(-)
```

---

## mutation-verdict

Three mutations, each reverted, each run through `rt.py`. The working tree is
byte-identical to `HEAD` after the last one (`git diff` empty, confirmed below).

### BEFORE any mutation — the original code, the historical defect

```
======================== 6 failed, 12 passed in 0.60s =========================
FAILED ...TestBenjaminiHochbergAgreement::test_significant_set_is_exactly_corrected_at_or_below_alpha
FAILED ...TestBenjaminiHochbergAgreement::test_agreement_holds_at_exactly_the_card_repro
FAILED ...TestBenjaminiHochbergAgreement::test_adjusted_matches_the_reference_running_minimum
FAILED ...TestBenjaminiHochbergAgreement::test_ties_receive_identical_adjusted_values
FAILED ...TestBenjaminiHochbergProperties::test_adjusted_is_monotone_in_the_sorted_order
FAILED ...TestBenjaminiHochbergProperties::test_single_test_is_unchanged
```

**The card's load-bearing test is in this list** —
`test_significant_set_is_exactly_corrected_at_or_below_alpha` is red on the
original code, exactly as the card predicted. (`test_single_test_is_unchanged`
is also here because I had asserted a wrong expectation of my own, `p = 0.031`
with `n = 1` *is* significant under BH; I corrected the test, not the code.)

### MUTATION A — restore the naive `p*n/rank`

The running minimum deleted, verbatim:

```python
adjusted_sorted = list(scaled)  # MUTATION: running minimum removed
```

```
======================== 7 failed, 106 passed in 4.62s ========================
FAILED tests/test_statistical_tests.py::TestBenjaminiHochbergAgreement::test_agreement_holds_at_exactly_the_card_repro
FAILED tests/test_statistical_tests.py::TestBenjaminiHochbergAgreement::test_adjusted_matches_the_reference_running_minimum
FAILED tests/test_statistical_tests.py::TestBenjaminiHochbergAgreement::test_significant_matches_the_reference_step_up
FAILED tests/test_statistical_tests.py::TestBenjaminiHochbergAgreement::test_ties_receive_identical_adjusted_values
FAILED tests/test_statistical_tests.py::TestBenjaminiHochbergProperties::test_adjusted_is_monotone_in_the_sorted_order
FAILED tests/test_statistical_tests.py::TestBenjaminiHochbergProperties::test_discovery_set_is_a_prefix_of_the_sorted_order
FAILED tests/test_statistical_tests.py::TestBenjaminiHochbergOnRealPValues::test_the_card_p_values_survive_the_report_unchanged
```

Seven red, including the card's scenario end to end. **The card's load-bearing
test is NOT in this list**, and that is a real result, not a coverage gap: after
the fix `significant` is derived from `corrected`, so deleting the running
minimum makes both outputs wrong *together* and the agreement identity still
holds. The agreement test's job is to fire on the **two-rule split** (the
historical defect), and on Mutation B below the two-rule split is provably
impossible. The formula is what needs the other seven tests, and it has them.

### MUTATION B — re-split the two rules, with a *correct* step-up

`significant` decided by the step-up rule again while `corrected` keeps the
running minimum — the original architecture, one correct rule:

```
============================= 113 passed in 4.52s =============================
```

**Zero red. This mutation is behaviour-preserving, and that is the point.**
BH 1995 eq. 2.4 says the step-up set and `{ i : q_i <= alpha }` are the same
set, so re-introducing the split with the right rule changes nothing — the
disagreement the card found came *only* from the wrong formula for `corrected`,
not from having two rules. Reporting this as a clean result for the fix, and
also as the honest reason the agreement test can no longer be provoked by the
old architecture: there is no longer an old architecture to provoke it with.

### restore

```
$ git -C C:\dev\wt-15 diff --stat
--- diff vs HEAD (must be empty) ---
--- restored, running ---
============================= 113 passed in 4.50s =============================
```

`git diff` empty, working tree clean, branch `agent/task-15` at `39acf9b`,
nothing pushed.

---

## REPORTED, NOT FIXED — same file, out of card scope

The card says: *"do not bundle ... Those are separate cards; note them in the
report."* Both confirmed by reading the code, neither touched.

### (a) `PairedPermutationTest` — unpaired Cohen's d on paired data

`src/interp/statistical_tests.py:225-227`:

```python
# Cohen's d
pooled_std = math.sqrt((values_a.var() + values_b.var()) / 2)
cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0.0
```

That is the **independent-groups** pooled SD, `sqrt((s_a² + s_b²)/2)`. The
paired effect size is `mean(a - b) / sd(a - b)`. The test exists for the
*correlated* case — `MetricComparisonReport.generate` feeds it per-sample metric
values for the same inputs under two models, L489-493 — and that is exactly
where the two estimators part company.

Quantitatively: if `corr(a, b) = ρ` and both have variance `σ²`, then
`sd(a - b) = σ·sqrt(2(1-ρ))` while the code's denominator is `σ`. So the
reported `d` is `d_paired / sqrt(2(1-ρ))` — **understated** by that factor.
At the repo's usual ρ for two models scored on the same inputs (high), the
reported effect size collapses toward 0. `ρ = 0.9` → the true `d` is divided by
`0.447`, i.e. a true `d = 2.0` is published as `d = 0.89`, and the
`effect_size_label` at L508-510 then reads `"medium"` where it should read
`"large"`. `ρ = 0.99` → factor `0.141`; a large effect becomes `"small"`.

This is not cosmetic: `effect_size_label` is a published claim, and
`_summary["n_large_effect"]` (L534) is counted from it. `EffectSize.cohens_d`
(L322-338) is the unpaired formula too, but that one is *correct for its
signature* — it takes two groups and is not claimed to be paired — so only
L226-227 is wrong. `tests/test_interp.py:674-682` only asserts
`abs(effect_size) < 1.0` under H0, which the wrong formula satisfies, so
nothing currently pins it.

### (b) `BayesianComparison` — a bootstrap wearing a posterior's name

`src/interp/statistical_tests.py:369-424`. The class docstring is
`P(JEPA > baseline | data)`, the method is
`probability_a_greater_b`, the return keys are `credible_lower` /
`credible_upper`, and the module header advertises *"Bayesian posterior
probability of JEPA > baseline"* and *"there is a 97% probability that JEPA is
better than baseline"* (L372-374). The body is
`np.random.RandomState(seed)` + a resampling loop (L406-412). There is no
prior, no likelihood and no posterior; the percentile of a bootstrap
distribution is a **confidence** interval, and `prob` is a bootstrap
proportion, not a probability of a parameter.

Two further problems in the same block, both read directly:

- **It discards the pairing.** L409-410 resamples `a` and `b`
  *independently* (`idx_a`, `idx_b` drawn separately), so the bootstrap
  variance of the difference is inflated by the between-condition variance that
  the pairing was there to remove. For `corr(a, b) = ρ` the true SD of the
  paired difference is `σ·sqrt(2(1-ρ))`; this estimates `σ·sqrt(2)`. The
  reported `credible_lower/upper` is therefore too wide by `1/sqrt(1-ρ)`, and
  the bootstrap is *anti-conservative for significance* (too wide) but
  *conservative for the probability claim* — it pushes `prob` toward 0.5, so
  `"decision": "A > B"` at L423 is harder to reach than the data warrants.
  `PairedPermutationTest`, three classes up, gets the pairing right; the two
  disagree about whether the observations are paired.
- **`prob_b_greater_a = 1 - prob` (L418) is not the probability that
  `B > A`.** It is its complement, so the two entries can never both exceed
  0.5 and their sum is exactly 1 by construction. If a reviewer reads
  `prob_a_greater_b = 0.97` as *"3 % chance the baseline is better"*, the
  number next to it says `prob_b_greater_a = 0.03` for a reason that has
  nothing to do with the data.
- `credible_lower/upper` are hard-coded to the 2.5/97.5 percentiles with no
  parameter for the level, and the `na < 2 or nb < 2` guard at L403-404
  returns `{"prob_a_greater": 0.5, ...}` — key name `prob_a_greater`, not
  `prob_a_greater_b`, so a caller reading the documented key gets a
  `KeyError` instead of the 0.5. Small, but it is a wrong-key failure on a
  degenerate input.

Neither is fixed here. (a) is a change to a published estimand and (b) is a
claim about what the module *is* — bootstrap or posterior — which is a human
decision about the paper's language, not a bug fix. They belong to whichever
card owns the module's statistical identity.

### one cosmetic residual inside my own change's blast radius

`MetricComparisonReport.generate:530` sets
`results[name]["significant_bonferroni"] = bonf[i] < alpha` — strict `<`, while
the BH side uses `<=`. On a continuous p-value the two are indistinguishable;
they can differ only when a permutation p lands exactly on `alpha`, which
needs `alpha · (n_permutations + 1)` to be an integer. I left it alone (outside
the card, no measurable effect) and flag it so nobody reads the asymmetry as
intentional.

---

## не_сделано / риски

**не_сделано**

1. `PairedPermutationTest`'s Cohen's d and `BayesianComparison` — reported
   above, per the card's explicit instruction, not bundled.
2. `TASKS.md` card 15 was **not** marked `done`, and
   `docs/plans/2026-09-27-wave1-audit-findings-interp.md` I10 was **not**
   struck through. Both are shared campaign state and other agents are writing
   them concurrently; the coordinator owns the bookkeeping.
3. `proofs/IMPLEMENTATION_STATUS.md` untouched — this is `src/interp/`, not a
   GWP mechanism, so the matrix does not cover it and there is no row to
   update.
4. `src/interp/__init__.py` untouched — `MultipleComparisonCorrection` was
   already exported (L116, L176) and the signature of
   `benjamini_hochberg(p_values, alpha)` is unchanged, so no caller needed
   updating. Grepped for every use of `benjamini_hochberg`,
   `MultipleComparisonCorrection`, `significant_bh` and `p_value_bh`; the only
   production consumer is L520.
5. The `significant_bonferroni` `<` / `<=` asymmetry above — flagged, not fixed.

**риски**

1. **Behaviour change for anyone who consumed `corrected` as the naive
   quantity.** `corrected` values can only go **down** (the running minimum is
   a minimum), and `significant` can only stay the same or grow. Any historic
   number in this repo computed as `p * n / rank` and labelled "BH corrected"
   is too large and is not comparable to the new output. Grep found no such
   consumer outside this module, but nothing checks stored results.
2. **`corrected` is still not a posterior.** This card makes the two BH
   outputs agree with each other; it does not make the adjusted p-value a
   statement about a probability being true. Combined with (b) above, a reader
   of a `MetricComparisonReport` still has no Bayesian content anywhere in the
   module. Fixing (b) is a separate card and I would not want the two merged.
3. **The card's headline is not reachable from real data.** Measured over 9
   seeds x 6 metrics, `PairedPermutationTest`'s `1/(n_permutations+1)` floor
   means the disagreement regime essentially never arises from this module's own
   pipeline today. That is a statement about the *permutation test*, not about
   the correction: `MetricComparisonReport` is a general reporting function and
   its `benjamini_hochberg` is the only place correction exists in the tree
   (audit finding I10), so any other p-value source routed through it inherits
   the regime. The end-to-end test therefore stubs the permutation p-values,
   and that stub is a permanent part of the test, not scaffolding I forgot.
4. **The agreement identity is now structural.** Because `significant` is
   derived from `corrected`, the load-bearing test can no longer fail from a
   disagreement — only from a wrong `corrected`. It is still the right
   assertion to keep (it is the spec, and it is what fired on the original
   code), but a future maintainer should not expect it to catch a *rule* change.
   The seven tests that Mutation A turned red are the ones that guard the
   formula; Mutation A is the mutation to re-run when touching the running
   minimum.
5. **Mutation B is behaviour-preserving, so it is not in the test suite.** It
   cannot be: there is no assertion that distinguishes a correct step-up
   recomputation from the derived set, because there is no observable
   difference. Anyone tempted to add one should read BH 1995 eq. 2.4 first.
6. The measurement scripts live outside the repo
   (`%LOCALAPPDATA%\Temp\opencode\t15_probe.py`, `t15_probe2.py`), so the
   before/after tables above are reproducible but not checked in. The
   discriminating parts of them are promoted into the test file; the seed sweep
   behind risk 3 is not, and I would rather say so than leave it implied.
7. `rt.py` has a 90 s cumulative budget and the box is shared, so I ran
   `tests/test_interp.py` + my file together (4.5 s of 90 s) and never the
   whole suite. `tests/test_run_comparison.py` was run separately (2.6 s) as
   downstream insurance. No other file in the repo imports
   `statistical_tests` (grepped), so the remaining suite is unaffected by
   construction — but that is a grep argument, not a full-suite run.
