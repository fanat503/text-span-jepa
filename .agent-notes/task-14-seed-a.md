# TASK-14 — seed A — `src/interp/ground_truth.py` thresholds calibrated on measured nulls

**status: done.** Branch `agent/task-14a`, worktree `C:\dev\wt-14a`, commit `1d987cd`.
Nothing pushed. `C:\dev\text-span-jepa` untouched.

---

## 1. Strategy

Minimal, as assigned. Kept the class/method structure, every function signature and
every existing return key. Changed only the threshold values and the failure/exception
handling, and added a calibration block that records where each number came from.
Added new return keys (additive, no caller breaks); added one new test file.

`git diff --stat`:

```
 src/interp/ground_truth.py                   | 183 ++++++++-
 tests/test_interp_ground_truth_thresholds.py | 562 ++++++++++++++++++++++++++++
 2 files changed, 728 insertions(+), 17 deletions(-)
```

---

## 2. The null suite (the measurement everything else is read off)

A "null" here is a **structure-free representation matrix of the same shape the
validators actually see** — `N=300, D=64` — carrying none of the class / position /
depth signal the generator plants. Six families, 260 draws, every matrix drawn from an
explicit `torch.Generator` so the whole table is reproducible.

| null family | what it is | draws |
|---|---|---|
| `isotropic` | iid Gaussian — the audit's example null | 60 |
| `same_scale` | iid Gaussian rescaled to the generator's own per-dim std: same nuisance scale profile, no semantics | 60 |
| `het_scale.5` | iid Gaussian × per-column lognormal(0, 0.5): nuisance per-dim scale spread, which real encoders have | 60 |
| `low_rank8` | rank-8 signal + 0.2 noise | 40 |
| `dead16` | iid Gaussian with 16 zero dims | 20 |
| `rank1` | rank-1 signal (full collapse) | 20 |

**STRUCTURED** reference (50 `SyntheticStructuredModel.generate` seeds, `snr=5`):

```
effective_dimension   min 37    median 38    max 39     mean 38.36
anisotropy            min 0.957547   median 0.961584   max 0.964939   mean 0.961449
```

**NULL SUITE** (260 draws):

| family | `effective_dimension` | `anisotropy` |
|---|---|---|
| isotropic | 62 .. 63 | 0.578705 .. 0.648987 |
| same_scale | 58 .. 59 | 0.884535 .. **0.911024** |
| het_scale.5 | 55 .. 59 | 0.865978 .. **0.970509** |
| low_rank8 | **8** (exact) | **0.987390** .. 0.990536 |
| dead16 | **47** (exact) | 1.0 (exact) |
| rank1 | 1 (exact) | 1.0 (exact) |

**Verdict of the OLD windows `(10, 60)` and `(0, 0.99)` on that suite:**

```
isotropic        old_pass=  0/ 60    new_pass=  0/ 60
same_scale       old_pass= 60/ 60    new_pass=  0/ 60
het_scale.5      old_pass= 60/ 60    new_pass=  0/ 60
low_rank8        old_pass=  0/ 40    new_pass=  0/ 40
dead16           old_pass=  0/ 20    new_pass=  0/ 20
rank1            old_pass=  0/ 20    new_pass=  0/ 20
```

**120 of 260 structure-free matrices were declared a valid structured representation.**

---

## 3. The measured null behind EACH threshold

### 3.1 `validate_polysemanticity` — `frac_monosemantic > 0` → `mean_psi > 0.0`

Measurement (PSI configured exactly as `validate_polysemanticity` configures it:
`n_clusters_range=(2,3)`, `n_top_activations=30`, `n_dimensions_sample=8`):

| dataset | `frac_monosemantic` | `mean_psi` |
|---|---|---|
| structured `SyntheticStructuredModel(300, 64)` | **1.0** (15/15) | **0.0905771255** (deterministic) |
| iid Gaussian 300×64 (the null) | **1.0** (40/40) | **0.0 exactly, 40/40** |
| broken PSI — every dim raises | **1.0** (by construction) | **0.0** |

`frac_monosemantic` is 1.0 on all three, so the old predicate was satisfied by the
structured data, by the null **and** by a completely broken index: it had zero
discriminative power, which is exactly the card's headline defect. `mean_psi` separates
all three, so the threshold is the null's measured maximum, `0.0`.

**Threshold: `mean_psi > 0.0`.** Structured margin 0.0906.

Also added the exception handling the old code lacked — `pipeline_valid` now requires
`math.isfinite(mean_psi) and n_dims_scored > 0`, because `PolysemanticityIndex.compute`'s
outer handler returns `mean_psi = inf` (which `> 0` would have *accepted*) and an empty
`per_dim_psi` list means nothing was scored at all.

### 3.2 `validate_geometry` — `10 < eff_dim < 60` → `8.0 < eff_dim < 47.0`

- Lower edge **8.0** = the largest effective dimension any null produces *below* the
  structured value. `low_rank8` gives exactly 8 on 40/40 draws; `rank1` gives 1.
- Upper edge **47.0** = the smallest effective dimension any null produces *above* the
  structured value. `dead16` gives exactly 47 on 20/20 draws; `het_scale.5` starts at 55,
  `same_scale` at 58, `isotropic` at 62.

Both null bounds are exact integers, so the strict inequalities reject them with no
floating-point slack to argue about. **Structured margin: 29 above the lower edge, 8
below the upper one** (structured range 37..39).

### 3.3 `validate_geometry` — `0 < aniso < 0.99` → `0.93 < aniso < 0.98`

- Lower edge **0.93**, placed between the largest anisotropy among the nulls that stay
  below the structured value (`same_scale`, 0.911024 over 60 draws; `isotropic`,
  0.648987) and the structured minimum (0.957547).
- Upper edge **0.98**, placed between the smallest anisotropy among the nulls above the
  structured value (`low_rank8`, 0.987390 over 40 draws; `dead16` and `rank1` are
  exactly 1.0) and the structured maximum (0.964939).

**Structured margin: 0.0275 above the lower edge, 0.0151 below the upper one.**

### 3.4 `validate_disentanglement` — `informativeness > 0.1` → `> 0.167770056`

The card says this one is "already improved". It is improved, but the literal `0.1` still
sits *inside* the new estimator's null, so I calibrated it too (it is a threshold in this
file and the card's requirement is per-threshold).

Pooled null over **192** independent N=200, D=64, K=3 draws (three
`DCIMetrics.measure_noise_floor(..., n_trials=64)` batches, seeds 0 / 99 / 7):

```
mean = 0.000522662    sd = 0.041811848    observed max = 0.101496144
```

**Threshold = mean + 4 sd = 0.167770056.** 4 sd is a two-sided normal tail of ≈6e-5, so
a factor-free run passes by chance about once in 17 000. The old `0.1` is below the
observed null max — one draw of 192 reached 0.1015 and passed.
**Structured measurement: 0.9915197211**, so the margin is enormous.

### 3.5 `full_validation` — `n_valid >= n_total - 1` → `n_valid == n_total`

The derivation is a coverage argument, measured rather than asserted:

- The suite has **4 checks** and the audit found **4 distinct blind spots**. A
  tolerate-one rule is therefore a licence for exactly one whole class of blindness to
  go unnoticed.
- The blind checks are the ones that *cannot* fail, so the one tolerated failure is
  always a real one, never a harmless one.
- Measured on the code as it stood: the probing check already fails
  (`class_min_depth = 2` while `class_max_acc = 1.0`, i.e. a linear probe scores 100% but
  the curve reports depth 2), and the old rule still reported
  `pipeline_reliable: True`.
- The test pins the property directly: inject one known failure into an otherwise
  healthy run and the summary must flip. Under the old literal it does not flip.

Added `n_failed` and `failed_tests` to `_summary` so a failure names itself.

---

## 4. Verify (full paste)

```
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
& $PY tools\rt.py tests/test_interp.py tests/test_interp_ground_truth_thresholds.py -q
```

```
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_interp.py tests/test_interp_ground_truth_thresholds.py -q
rt.py: threads=1  total_budget=90s  slow_ok=False
........................................................................ [ 61%]
.............................................                            [100%]
117 passed in 21.45s
```

Neighbouring interp suites, unchanged:

```
$PY tools\rt.py tests/test_info_and_disentangle.py tests/test_probes_split.py -q
..................................................................       [100%]
66 passed in 9.56s
```

Lint / format:

```
$PY -m ruff check src/interp/ground_truth.py tests/test_interp_ground_truth_thresholds.py
All checks passed!

$PY -m black --check src/interp/ground_truth.py tests/test_interp_ground_truth_thresholds.py
2 files would be left unchanged.
```

Import-identity guard: every probe and every measurement was run through `tools/rt.py`
(pins `cwd` to the worktree), and the first probe asserted
`src.__file__ = C:\dev\wt-14a\src\__init__.py` before any number was recorded, so none of
the above came from the stale `pip install -e` clone.

---

## 5. Mutation verdict

Nine mutations, each one applied alone and then reverted. **All nine go red.**

```
=== M0 baseline (unmutated, final file) ===
.......................                                                  [100%]
23 passed in 17.04s

============================================================
 MUTATION: M1  frac_monosemantic > 0   [old polysemanticity threshold]
============================================================
FAILED ...::TestPolysemanticityThreshold::test_broken_psi_fails_validation
FAILED ...::TestPolysemanticityThreshold::test_broken_psi_whose_result_is_infinite_fails_validation
FAILED ...::TestPolysemanticityThreshold::test_psi_that_scored_nothing_fails_validation
FAILED ...::TestFullValidation::test_broken_psi_inside_the_full_suite_is_caught
4 failed, 19 passed in 18.25s

============================================================
 MUTATION: M2  10 < eff_dim < 60        [old geometry window]
============================================================
    assert result["reasonable_eff_dim"] is False, result
E   AssertionError: {'effective_dim': 58.0, 'anisotropy': 0.9104525372385979, 'eff_dim_window': (8.0, 47.0), 'aniso_window': (0.93, 0.98), ...}
E   assert True is False
FAILED ...::TestGeometryThresholdsRejectTheNull::test_the_validator_rejects_a_null_it_is_handed
1 failed, 22 passed in 17.40s

============================================================
 MUTATION: M3  0 < aniso < 0.99         [old geometry window]
============================================================
    assert result["reasonable_anisotropy"] is False, result
E   AssertionError: {'effective_dim': 58.0, 'anisotropy': 0.9104525372385979, 'eff_dim_window': (8.0, 47.0), 'aniso_window': (0.93, 0.98), ...}
E   assert True is False
FAILED ...::TestGeometryThresholdsRejectTheNull::test_the_validator_rejects_a_null_it_is_handed
1 failed, 22 passed in 18.60s

============================================================
 MUTATION: M4  n_valid >= n_total - 1   [old full_validation rule]
============================================================
E   assert True is False
FAILED ...::TestFullValidation::test_summary_uses_unanimity_not_a_tolerated_failure
FAILED ...::TestFullValidation::test_one_injected_failure_makes_the_suite_unreliable
FAILED ...::TestFullValidation::test_broken_psi_inside_the_full_suite_is_caught
3 failed, 20 passed in 17.83s

============================================================
 MUTATION: M5  informativeness > 0.1    [old DCI threshold]
============================================================
    assert out["pipeline_valid"] is False, out
E   AssertionError: {'disentanglement': 0.212508425116539, 'completeness': 0.07596594095230103, 'informativeness': 0.10149614435511438, 'informativeness_threshold': 0.167770054, ...}
E   assert True is False
FAILED ...::TestInformativenessThreshold::test_the_validator_rejects_an_informativeness_the_null_actually_produced
1 failed, 22 passed in 17.86s

============================================================
 MUTATION: M6  drop the failure guards on the PSI result
============================================================
    assert result["psi_usable"] is False
E   assert True is False
FAILED ...::TestPolysemanticityThreshold::test_broken_psi_whose_result_is_infinite_fails_validation
FAILED ...::TestPolysemanticityThreshold::test_psi_that_scored_nothing_fails_validation
2 failed, 21 passed in 18.01s

============================================================
 MUTATION: M7  widen the eff window to (-inf, inf)
============================================================
E    +  where False = <built-in function isfinite>(-inf)
FAILED ...::TestGeometryThresholdsRejectTheNull::test_collapsed_data_is_rejected_from_both_ends
FAILED ...::test_thresholds_are_finite_and_ordered
2 failed, 21 passed in 17.67s

============================================================
 MUTATION: M8  widen the aniso window to (-inf, inf)
============================================================
E    +  and   False = <built-in function isfinite>(inf)
FAILED ...::TestGeometryThresholdsRejectTheNull::test_collapsed_data_is_rejected_from_both_ends
FAILED ...::test_thresholds_are_finite_and_ordered
2 failed, 21 passed in 17.04s

============================================================
 M9  the WHOLE original module (git HEAD) vs the new tests
============================================================
19 failed, 4 passed in 12.98s

=== restored; confirming green ===
.......................                                                  [100%]
23 passed in 16.96s
```

**One round of this battery did not go red, and I fixed it before reporting.** The first
version of the test file asserted the module-level *constants* against the null suite, so
M2, M3 and M5 all survived: mutating the predicate while leaving the constant in place
was invisible. I added three tests that drive the module's own predicates —
`validate_geometry` fed the geometry of a real `same_scale` null via a patched
`RepresentationGeometry.compute_all` (with `reasonable_eff_dim` and
`reasonable_anisotropy` asserted **separately**, so loosening one cannot hide behind the
other), and `validate_disentanglement` fed the worst of 64 real null draws. After that,
every mutation is red. M9 also had to be fixed: the original first version of the test
module imported the new constants directly and died with an `ImportError` during
collection, which proves nothing, so the imports now go through `getattr` and M9 produces
a real 19-failure run.

Claimed as load-bearing: `test_broken_psi_fails_validation`,
`test_broken_psi_whose_result_is_infinite_fails_validation`,
`test_psi_that_scored_nothing_fails_validation`,
`test_broken_psi_inside_the_full_suite_is_caught`,
`test_the_validator_rejects_a_null_it_is_handed`,
`test_summary_uses_unanimity_not_a_tolerated_failure`,
`test_one_injected_failure_makes_the_suite_unreliable`,
`test_the_validator_rejects_an_informativeness_the_null_actually_produced`,
`test_collapsed_data_is_rejected_from_both_ends`,
`test_thresholds_are_finite_and_ordered`. Each went red under at least one mutation.

The remaining 13 tests are guard-rail coverage that no single mutation isolates
(`test_synthetic`-style shape checks, the "old windows admitted the null" regression
assertion, NaN-handling, summary bookkeeping). I am **not** claiming them as
load-bearing; a sibling worker in wave 3 declined to claim an unmutated test and that is
the standard here.

---

## 6. не_сделано / риски

**Not done / deliberately out of scope**

- `validate_probing_complexity` untouched. It is not one of the four named thresholds.
  But it **fails today, deterministically**: `class_min_depth = 2` while
  `class_max_acc = 1.0` — a depth-1 linear probe scores 100% and the curve still reports
  depth 2. That is a bug in `probing_complexity.py`, not in this file, and it is not
  mine to fix. The visible consequence of my change: `full_validation` now reports
  `pipeline_reliable: False`, where before it reported `True`. That is the correct
  answer to a suite that has a real failure, but downstream callers reading
  `pipeline_reliable` will see a behaviour change. `scripts/run_experiment.sh:76` is
  such a caller.
- `SyntheticStructuredModel.generate` still calls `torch.manual_seed` on the
  process-global RNG (TASKS.md:456, a different card). I relied on it for determinism
  and did not touch it.
- I did not restructure the API. `frac_monosemantic` is still reported, it is just no
  longer used to decide.
- I did not verify the whole suite (688 tests). `rt.py` refuses it by design. I ran
  `test_interp.py` (the card's verify command) plus the two adjacent interp files.

**Risks**

1. **The card's example null is the one the old window happens to catch.** The card says
   "a random Gaussian 300×64 matrix lands inside both windows". Measured: an iid Gaussian
   gives `effective_dimension` **62..63**, which is *outside* `10 < eff_dim < 60`. So the
   old effective-dimension window did reject that particular null. The card's *conclusion*
   still holds and is worse than stated — two other structure-free nulls
   (`same_scale`, `het_scale.5`) sit inside **both** old windows and were admitted
   60/60. I report this because the numbers in the module header contradict the card's
   worked example, and the header is what a reviewer will check.
2. **`anisotropy` cannot exclude `het_scale.5` on its own.** That null reaches 0.970509,
   which crosses the structured range [0.957547, 0.964939]. The effective-dimension
   window is what rejects it (55..59 > 47). The conjunction is sound; the individual
   anisotropy window is not, and that is stated in the module.
3. **The anisotropy upper margin is 0.0151** (0.98 vs a structured max of 0.964939). A
   different torch/BLAS build, a different `svdvals` backend, or a different
   `SyntheticStructuredModel` RNG stream could move the structured anisotropy by more
   than that. If this test starts failing on another machine, re-measure the null block —
   do not widen the window by hand.
4. **`het_scale` is a judgement call about what counts as a null.** I included it because
   real encoder representations do have per-dimension scale spread, so it is the null a
   reviewer should care about. A reader who disagrees can delete it and the remaining
   five families still pin both windows from the same measurements.
5. **`mean_psi > 0.0` is a weak discriminator in the other direction.** Measured against
   a *label-permuted* structured matrix (representation structure intact, class labels
   destroyed), `mean_psi` runs 0.0 .. 0.2217 with 9 of 40 draws at exactly 0.0 — so that
   null can score either side. I did not treat it as a null because PSI clusters in
   representation space, not label space, so the representation structure is genuinely
   still there. The consequence is stated plainly: this check now catches *PSI collapsing
   to zero* and *PSI returning nothing*, and it does **not** catch *PSI keyed on the
   wrong labels*. Catching that would need a class-vs-noise-dims comparison, which is a
   restructure, not a threshold value.
6. **`pipeline_reliable` is now a stricter contract.** Anyone treating it as "mostly
   fine" loses that. That is the point of the card, but it is a caller-visible change.
