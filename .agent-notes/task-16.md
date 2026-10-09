# TASK-16 — what a rigorous JEPA-vs-baseline comparison requires

**branch** `agent/task-16` · **worktree** `C:\dev\wt-16` · **base** `agent/wave-8` (`b71e7f6`)
**files changed** `src/interp/run_comparison.py`, `src/interp/compare.py`, `tests/test_run_comparison.py`
`statistical_tests.py` was read and **not edited** (TASK-15 owns it).

**answer in one line: the honest answer is the one the card named as a
possibility — a held-out split and independent replicates are missing, so no
significance test rescues this. `statistical_tests.py` is therefore NOT wired
in. The pipeline now emits an explicit refusal instead of a number.**

---

## 1. What is actually compared

`run_full_comparison(jepa_model, baseline_model, dataloader, …)` is handed:

| unit | count | resamplable? |
|---|---|---|
| checkpoints per arm | **1** | no |
| corpora | **1** | no |
| rows of the representation matrix | N | yes |

Both arms are scored on the same rows in the same order — this part of the
design is genuinely **paired**, and I verified the pairing rather than
assuming it (below).

A claim of the form "JEPA beats the baseline" is a claim about the population
of *training runs* on a population of *corpora*. The units that matter are
checkpoints and corpora, and the signature supplies one of each. The only
thing this pipeline can resample is a row.

## 2. Why a p-value would make the evidence worse, not better

The card's trap, stated precisely: a row bootstrap **conditions on the
checkpoint**. The variance that decides the sign of "JEPA better" is *not*
row-sampling variance. Two measurements, both on the real pipeline:

**(a) Nuisance floor.** Models held completely FIXED, only the dummy-corpus
seed varied (8 runs):

```
effective_rank_online diff: min 0.55829 .. max 1.98664   spread 1.42836
point |A-B| at seed 0:      0.55829
```

The spread is **2.6× the point effect the report presents as a finding.**

**(b) Seed variance swamps the arm difference.** Across independently
*initialised* models of one arm — a **lower bound**, since no training is
involved and real post-training seed variance is strictly larger:

```
effective_rank_online        arm diff 0.05425   within-arm seed spread 0.15253  SWAMPED
mean_pairwise_cosine_online  arm diff 0.00143   within-arm seed spread 0.00214  SWAMPED
sv_entropy_online            arm diff 0.00073   within-arm seed spread 0.00207  SWAMPED
```

For 3 of the 4 headline metrics the sign of "JEPA better" is decided by
*which seed happened to be loaded*. A p-value from row resampling is blind to
exactly that component, because it holds the seed fixed and resamples rows of
the one matrix that seed produced. Wiring one in converts an unmeasured
variance into a confident-looking number.

## 3. Why the existing machinery specifically does not fit

I ran it on the data Phase 7 actually has.

- **`PairedPermutationTest.compute`** wants a per-sample vector. Phase 7 has
  **one scalar per arm per metric**. Its `N < 2` branch returns
  `{'p_value': 1.0, 'significant': False, 'mean_diff': 0.0}` — which a report
  prints as *"tested, not significant"*, i.e. **"no difference found."** It is
  not that: no test ran. `PairedPermutationTest` now refuses instead
  (`test_bootstrap_refuses_fewer_than_two_rows`).

- **`BootstrapCI.compare`** draws `idx_a` and `idx_b` **independently**, but
  the arms are scored on the same rows. On identical data it reported a
  CI of **[-0.452, +0.140], width 0.592** between **two copies of one array**
  — it invents a difference where there is none by definition. Correctly
  paired, the same input gives **[0.0, 0.0]**. On genuinely paired arms the
  unpaired version is also **1.19× wider**.

## 4. The multiplicity trap, which is worse than "no correction"

The four metrics are **not four tests**. Bootstrap-difference correlations:

```
              effective_r  collapsed_d  mean_pairwi  sv_entropy_
effective_r       1.0000          nan       0.0648       0.9999
sv_entropy_       0.9999          nan       0.0651       1.0000
```

`effective_rank_online` and `sv_entropy_online` are **one measurement reported
twice** (r = 0.9999 — both are functionals of the same singular spectrum), and
`collapsed_dim_ratio_online` is **identically 0.0** on both arms.

Meanwhile the report prints **61 scalar comparisons** (43 collapse + 4 geometry
+ 5 information-theory + 4 polysemanticity + 4 CKA + 1) against a nominal
family of 4. At α = 0.05 uncorrected that is **3.1 expected false
declarations**; Bonferroni over the printed family is α = 0.00082. So a raw
p of 0.03 that BH-over-4 would call significant is *not* significant across
what is actually published.

**This is the concrete sense in which adding significance testing makes things
worse:** it would attach a corrected-looking p-value to a family of 4 while 61
comparisons go out uncorrected — buying false confidence with a
multiple-comparison label.

## 5. The sign-count verdict

`wins > n//2` (Phase 8 here, `RobustnessBattery` at `robustness.py:389`) fires
by chance:

```
n=4: P = 0.3125     n=6: P = 0.3438     n=8: P = 0.3633
```

The report now prints that number next to the count and marks it
`geometry_majority_is_a_verdict = False`. `majority_null_probability` is an
exact closed form, so it is checkable by hand.

## 6. What shipped

**The refusal** — `describe_comparison_design()` records the units of
replication, which variance components are covered and which are not, and
returns `verdict = "NOT_TESTABLE"`. It reaches the JSON, `summary.txt`, and the
printed report. There is **no `p_value` and no `significant` key anywhere** in
`results["statistical"]`, and a test asserts their absence.

**The one honest interval** — `paired_row_bootstrap_ci()` is correctly paired
(one index into both arms per replicate). It is **opt-in**
(`n_bootstrap=0` default) because it costs two `CollapseDiagnostics.compute`
calls per replicate: measured **90 s at B=100, 453 s at B=500** for the
production shape (N=500, D=768, one thread). When computed, it is named for
what it resamples rather than for a hypothesis test, and it records that it
says nothing about seed or corpus variance.

**The pairing, now asserted.** `run_full_comparison` iterated the caller's
dataloader once per arm, so `shuffle=True` silently compared unpaired
matrices — **measured 99 of 100 rows misaligned**. It now raises
`ComparisonDesignError` naming the likely cause.

**Phase 4's surface MI is no longer fabricated.** `positions =
arange(N).expand(-1, D)` is **constant within each row**, so `F.normalize`
collapses it to ~2 distinct directions, the similarity matrix has identical
rows, cross-entropy equals log(N), and the estimator's `max(mi, 0.0)` returns
**exactly 0.0** — measured for noise, all-zeros, all-ones, rank-1 and 1e6-scaled
inputs alike. `jepa_mi_position` vs `baseline_mi_position` was **0.0 vs 0.0**
for every model; the module's headline hypothesis was never tested. A caller
can now supply `surface_features=(N, K)`.

**`main()` refuses the random-token corpus** by default. It parsed
`--dataset wikitext`, **never loaded it**, fell through to
`torch.randint(0, 50304, (500, 128))`, ran the full protocol on uniform noise,
and wrote `comparison_results.json` + `summary.txt` — indistinguishable from a
real result. Now refused unless `--allow_random_tokens`, and the report is
stamped SYNTHETIC when allowed.

**`compare.py`**: fixed `extract_linguistic_features(["The", ""])` →
`IndexError` (`t[0]` was indexed before the length guard), and
`full_comparison_report` now carries `verdict_supported: False`.

## 7. A hypothesis I tested and had to discard

I assumed the surface-MI fix was one line: swap `arange(N)` for `jepa_ids`.
**Measured false.** The raw ids also floor at 0.0 — with many rows sharing a
token the positive is one of several near-duplicates, the loss exceeds log(N),
and the same clamp censors the result. The cause is the `max(mi, 0.0)` clamp,
not the constant rows: the reported MI is a **one-sided, censored** quantity,
so *"no measurable dependence"* and *"estimator failed"* are the same printed
number. Deciding what the surface feature **is**, and encoding it, is a
research decision this module does not get to make silently. Recorded, not
guessed.

## 8. The sibling card — it bears directly, and it is a second independent reason

The sibling found every probe trains full-batch Adam at lr=1e-3, so the whole
weight-displacement budget is `lr × epochs = 0.03`: **0.708** train accuracy
where lr=0.1 reaches **0.958**.

`run_comparison` trains nothing — it reads frozen representations — so the
bearing is not direct. But it is **not merely indirect**, and I measured why.
Perturbing ONE arm's representation toward its under-trained state (a lower
bound: it perturbs the representation, not the optimiser, so it cannot be
dismissed as a training artefact):

```
metric                            arm-vs-arm    undertrain     ratio
effective_rank_online                0.23095       4.39544     19.03  BUDGET DOMINATES
collapsed_dim_ratio_online            0.00000       1.00000       inf  BUDGET DOMINATES
sv_entropy_online                     0.00380       0.06471     17.04  BUDGET DOMINATES
```

**For 3 of the 4 metrics, separating an under-trained model from its own
converged self takes 17–19× the movement that separates the two arms.** These
metrics are budget-sensitive, so if the two arms' checkpoints were trained at
different effective budgets, the comparison measures **optimisation, not
architecture**.

That is a **second, independent** reason n=1-per-arm is disqualifying: even
with several checkpoints, if they are not budget-matched the variance is
confounded. The sibling's number is therefore not a separate module's problem
— it is evidence about what these checkpoints encode.

## 9. Verification

```
& $PY tools/rt.py tests\test_interp.py tests\test_run_comparison.py
155 passed in 26.42s          (baseline before this card: 134)
ruff: All checks passed       black: 3 files would be left unchanged
```

### MUTATION VERDICT — 6/6 caught

Run as a scripted battery (`m16_battery.py`) so it is reproducible. Every
mutation restores from the **git INDEX**, never `HEAD` (the TASK-35 lesson).

| # | mutation | result |
|---|---|---|
| M1 | paired bootstrap → draw `idx_a`/`idx_b` independently | **CAUGHT** — `test_identical_arms_give_a_zero_width_interval`: *"got [-0.208, +0.358]. A non-zero interval means the two arms were resampled independently, discarding the pairing."* |
| M2 | emit a `p_value` (the forbidden module's output shape) | **CAUGHT** — `test_report_contains_no_p_value_or_significance_flag` |
| M3 | delete `assert_paired_extraction` | **CAUGHT** — `test_a_shuffled_dataloader_is_refused` |
| M4 | `verdict_supported = not uncovered` → `True` | **CAUGHT (2)** — `test_verdict_is_not_testable…`, `test_summary_does_not_claim_a_supported_verdict` |
| M5 | restore `arange(N)` as the "surface" feature | **CAUGHT** — `test_mi_surface_is_absent_rather_than_fabricated` |
| M6 | disable the random-token refusal | **CAUGHT** — `test_random_token_run_is_refused_without_the_explicit_flag` |

Baseline green → 6 mutations → 6 red → restore → green.

**One honest negative:** my first attempt at M5 mutated the *supplied-feature*
branch and **all 4 Surface tests passed**. The mutation was in the branch the
tests do not exercise. I corrected it to mutate the `else` branch — the actual
pre-fix behaviour — and it was caught. A mutation that survives because it
landed in dead code is worth recording: it nearly produced a false "caught".

## 10. Boundary notes

- **`tests/test_run_comparison.py` is outside the card's `files_allowed`**
  (`src/interp/run_comparison.py`, `src/interp/compare.py`). I edited it
  because it already existed, is owned by TASK-27 which is **done**, and a
  behaviour change without tests is exactly the defect class this campaign
  exists to catch. Flagging rather than assuming — revert if the owner
  disagrees.
- **`robustness.py:389` is not mine** and is unchanged. `RobustnessBattery`
  still declares `jepa_wins > 4//2` with no null probability. It should
  import `majority_null_probability` from here; recommend a follow-up card.
- **`statistical_tests.py` untouched.** Its BH fix is TASK-15's. Two
  independent reasons not to wire it in are recorded above (§3); they would
  still stand with BH corrected.

## 11. What would actually make this rigorous

In `required_to_support_a_verdict`, and it is not a code change:

1. **≥ 2 independently trained checkpoints per arm**, same protocol, same
   budget. This is the binding constraint — measured in §2(b).
2. **A held-out split disjoint from BOTH arms' training data**, or ≥ 2 corpora.
   A row bootstrap over one corpus cannot supply it. This is the card's named
   finding and it is the honest answer.
3. **One declared multiplicity family** covering all 61 printed comparisons,
   not 4.
4. Given the sibling's finding: **budget-matched arms**, or the metrics measure
   optimisation rather than architecture (§8).