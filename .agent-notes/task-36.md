# task-36 — CI-failing `TestComparableAcrossDatasetSize::test_score_does_not_scale_with_n_top_over_n`

**status: DONE — closed as (a), test-only fix, committed on `agent/task-36` as `6c32216`, not pushed.**
Worktree `C:\dev\wt-36` (branch `agent/task-36`, base `agent/wave-4` = `9adc4e2`).
`C:\dev\text-span-jepa` untouched.

---

## 1. The question that decides the fix: wide or tight?

Measured first, on the **shipped** seed scheme (`sae=d, pool=1000+d, order=2000+d,
feature seed=d`), 240 decorrelated draws. Probes ran from inside the worktree root and
every one asserts `src.__file__` is under `C:\dev\wt-36` before measuring.

**Per-draw ratio `score(N=240)/score(N=2400)`, 240 draws:**

| `n_top` | fixed min..max | fixed mean | old min..max | old mean | `A(n_top)` |
|---|---|---|---|---|---|
| 25 | 0.611 .. 1.304 | 0.966 | 0.553 .. 1.180 | 0.874 | 0.9053 |
| 50 | 0.590 .. 1.384 | 1.011 | 0.477 .. 1.119 | 0.818 | 0.8085 |
| 100 | **0.623 .. 1.311** | 0.940 | 0.379 .. 0.798 | 0.572 | 0.6087 |
| 200 | 0.501 .. 1.912 | 0.900 | 0.078 .. 0.366 | 0.139 | 0.1818 |

**Answer: WIDE, and Linux's 0.609 was below every draw I measured.**
At `n_top=100`, **0 of 240** draws fall below 0.609 (measured min 0.623). This box's
0.719 is a typical draw. So the `0.65 < ratio < 1.25` band was not a tight property the
Windows runs straddled — it was a knife edge balanced on one platform's single draw, and
the distribution is ~2x wider than the band on each side.

The bias is real and systematic (medians 0.94–1.01, well below 1 at `n_top=100` and
`n_top=200`), but the per-draw *noise* around it is what broke the test.

## 2. Two measurements that changed the fix, not the number

**(i) The old defect has a closed form.** With the control set equal to the whole
dataset the score is *exactly* `(N - n_top)/N` times the correct one. Verified to
**1.3e-07** (float32) across 60 draws:
`old_ratio(n) = A(n) · fixed_ratio(n)`, `A(n) = ((Ns-n)/Ns)/((Nl-n)/Nl)` =
0.905 / 0.809 / 0.609 / 0.182 for `n_top` = 25 / 50 / 100 / 200. It is a deterministic
multiplicative attenuation, i.e. **one-directional** — so a one-sided bound is the right
shape, not a two-sided band.

**(ii) No single ratio band at a single `n_top` can separate the two implementations.**
At `n_top <= 100` the fixed and old ranges overlap heavily. The removed
`0.65 < ratio < 1.25` band let the **attenuated implementation pass in 8 of 60 draws**.
That assertion was never a regression test. What *does* separate them is the dependence
on `n_top`, where `A(n)` swings **4.98x** while the fixed ratio moves **1.12x**.

So the fix is a **better statistic**, not a better number. Widening the band would have
made it *worse* — the wider it got, the more attenuated draws it admitted.

**Dead end, reported because it looked promising:** exact row replication
(`base.repeat(m,1)`) should have given a noise-free invariance proof. It does not:
`topk` tie-breaking over exactly-duplicated rows is not index-stable (the returned
`treated_idx` set differed in **6 of 24** draws) and the score moved by up to **0.38**
relative. Measured, not assumed — the idea was dropped.

## 3. Verdict: (a), not (b)

**The residual bias is a real property of the estimand; the test was wrong.**
`FeatureInterferenceScore` is **not** defective:

* After the fix the score is a function of the treatment and control **row sets alone**.
  `N` enters only through the documented `min(n_top, N // 2)` cap. No `N`-carried
  multiplicative factor survives — that is the `(1 - n_top/N)` claim, and it is closed
  form, not empirical.
* The residual is an **order-statistics/estimand** effect: `n_top` counts *rows*, so at
  `n_top=100` the treatment is 42% of a 240-row dataset but 4% of a 2400-row one. It
  spans more of the block structure at small N, so the mean shift is genuinely smaller
  there. That is the correct value of the question being asked on that dataset, not an
  artifact.
* Removing it would require `n_top` to be a **fraction** of `N` — a different parameter
  contract, contradicting the class docstring and four other assertions in the same file
  (`n_treated == 40` at N=120, `== 50` at N=100, `== 20` at N=60), and `n_top` is used by
  callers outside my file ownership.

**No behaviour change to `src/interp/feature_composition.py`.** Docs-only: the class
docstring said the fix made the score "comparable across datasets" unqualified, which
overstates it (mean 0.940, per-draw 0.62..1.31). It now carries the measured numbers and
points at the test.

## 4. Assertion I replaced, deliberately

`assert 0.65 < ratio < 1.25` on a **single deterministic draw** at `n_top=100`.
It encoded a wrong expectation: it asserted exact invariance of an estimand that is not
invariant, from one draw of a statistic with a 0.69x-wide range — and it had **no teeth**
(8/60 attenuated draws pass it). It is the only assertion in the file I touched, and it is
replaced, not relaxed.

Every other existing assertion is **kept verbatim**, including the two inside the test I
rewrote: `small["n_treated"] == large["n_treated"] == 100` and `mean_interference > 0`
(now checked inside `_mean_ratios` for all 16 draws at `n_top=100`).

## 5. The replacement, and the measurement behind each number

Statistics are **means over `DRAWS = 16` decorrelated draws**, at `n_top` in
(25, 50, 100, 200):

| assertion | threshold | fixed, 15 blocks | old, 15 blocks | margin (fixed / old) |
|---|---|---|---|---|
| per-`n_top` mean ratio > floor | 0.40 | worst **0.8235**, 15/15 pass | best **0.1528**, 0/15 | **2.06x** above / **2.62x** below |
| `max_n mean_n / min_n mean_n` < cap | 2.5 | worst **1.2330**, 15/15 pass | best **5.6414**, 0/15 | **2.03x** below / **2.26x** above |
| `0.75 < mean R(100) < 1.35` | (0.75,1.35) | 0.8825..0.9967, 15/15 pass | 0.5372..0.6067, 0/15 | **1.18x** above / **1.24x** below |

Justifications, not "widened until green":
* **cap 2.5** = geometric midpoint of the two measured ranges, `sqrt(1.2330 · 5.6414) = 2.64`.
  Chosen in an empty gap with ~2x on both sides. *A per-draw cap of 2.25 was tried first
  and rejected:* on this seed scheme the fixed code reached **2.12** — only 1.06x headroom.
  That measurement is why the statistic is an aggregate.
* **floor 0.40** — the defect is an attenuation, so only the lower edge needs to carry
  weight. Old max there is 0.153.
* **band 0.75..1.35** — per-draw ratio has sd 0.146, so `DRAWS=16` gives sem 0.037; 0.75
  sits ~5 standard errors below the measured mean 0.940 and 0.13 below the worst of 15
  blocks. Safety was bought with **draws**, not with a wider edge.
* Thresholds were picked on one 240-draw block set and re-checked on a **held-out,
  disjoint** 200-draw seed block (`pool=300000+d`, `order=500000+d`) before adoption.

## 6. Verify (full paste)

```
$ & $PY tools\rt.py tests/test_feature_composition.py
rt.py: ... -m pytest -q --no-header -p no:cacheprovider tests/test_feature_composition.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 13 items

tests\test_feature_composition.py .............                          [100%]

============================= 13 passed in 9.39s =============================
```
```
$ & $PY -m ruff check tests/test_feature_composition.py src/interp/feature_composition.py
All checks passed!

$ & $PY -m black --check tests/test_feature_composition.py src/interp/feature_composition.py
All done! ✨ 🍰 ✨
2 files would be left unchanged.
```
Consumers and the gate's shared 90s budget (my file grew 1s -> 9.4s, so this had to be checked):
```
$ & $PY tools\rt.py tests/test_interp.py
collected 94 items ... 94 passed in 7.49s

$ & $PY tools\rt.py tests/test_cmc_resume.py tests/test_run_comparison.py \
      tests/test_determinism.py tests/test_feature_composition.py \
      tests/test_ablation_module.py
collected 78 items ... 78 passed in 29.47s
```
No training, no whole-suite run, 1 thread, everything via `tools/rt.py`.

## 7. Mutation verdict against the OLD implementation

`FeatureInterferenceScore.compute` monkeypatched back to the pre-fix defect
(`z_baseline = z`, control set = the whole dataset), shipped tests run **unmodified**:

```
-- BASELINE (unmutated)
  PASS  (all 13)

-- MUTATED: compute = OLD (control == whole dataset)
  FAIL  TestControlDisjointFromTreatment::test_compute_reports_a_non_overlapping_split_for_every_feature
  FAIL  TestControlDisjointFromTreatment::test_control_is_not_the_full_dataset
  FAIL  TestNotTriviallyZeroAtNTopEqualsNTreated::test_score_is_not_float_round_off_when_n_top_equals_n
  FAIL  TestComparableAcrossDatasetSize::test_mean_score_is_comparable_across_a_tenfold_dataset_growth
         -> mean score moved 0.607x between N=240 and N=2400 over 16 draws
  FAIL  TestComparableAcrossDatasetSize::test_score_does_not_scale_with_n_top_over_n
         -> n_top=200: score is 0.143x its N=2400 value on average over 16 draws.
            The closed-form 0.182x (1 - n_top/N) attenuation is exactly what this looks like.

verdict: 5/13 tests FAIL under the old implementation
shipped TestComparableAcrossDatasetSize tests, killed under the mutation?
  YES  ...::test_mean_score_is_comparable_across_a_tenfold_dataset_growth
  YES  ...::test_score_does_not_scale_with_n_top_over_n
```
Block-simulation form of the same verdict: **fixed 15/15, attenuated 0/15** on every one
of the three thresholds. The assertion I removed **would have passed** the old implementation.

Honest detail: `test_score_is_computed_from_the_reported_split` **survives** this mutation.
That is correct — my mutation keeps the reported indices and the arithmetic mutually
consistent, and that test only pins honesty of reporting against arithmetic. It does *not*
catch a self-consistent contamination; the two new assertions do. Pre-existing gap, not
introduced here.

## 8. A bug the mutation harness caught in my own work

My first version memoised `_mean_ratios` on the class. The mutation verdict then showed
**both new tests PASSING against the mutated implementation** — the cache handed back
unmutated values from the baseline run in the same process. Removed, with a comment
recording why, so the next person does not re-add it. Re-verified after removal.

## 9. diff-stat

```
 src/interp/feature_composition.py |  13 +++
 tests/test_feature_composition.py | 173 ++++++++++++++++++++++++++++++++------
 2 files changed, 159 insertions(+), 27 deletions(-)
```
`src/interp/feature_composition.py` is **+13/-0 — docstring only**. Nothing else in
`src/interp/**` touched. No skip, no xfail, no `--no-verify`.

## 10. Will this pass on Linux CI?

**Probably yes, and materially safer than what it replaces — but I cannot prove it, and
one number carries the residual risk.** Stated plainly rather than claimed as certain:

* The two robust assertions (floor 0.40, cap 2.5) have **~2x margin on both sides** across
  15 blocks and 440 draws, three different seed schemes. The defect is a *deterministic*
  multiplier, so these cannot be luck.
* `test_mean_score_is_comparable_across_a_tenfold_dataset_growth` is the thin one: worst
  measured block 0.8825 vs the 0.75 edge = **1.18x**. Extrapolating the measured sd
  (0.146 per draw, sem 0.037 at `DRAWS=16`), that edge is ~5 sem out — roughly a 1e-7
  chance of a false failure. The old test, by the same arithmetic on its own spread, failed
  about **0.4%** of draws (0 of 240 below 0.609 is a ~1-in-240 tail event on this box).
  So this is roughly a 1000x improvement, not an elimination.
* **Why the platform difference existed at all, honestly:** the test's data are built from
  seeded `torch` generators, which are deterministic per torch build but *not* guaranteed
  identical across the Windows and Linux wheels or across Python versions. So a change of
  wheel alone can move the draw anywhere in the measured 0.62..1.31 range. The new test is
  robust to that **because it averages 16 draws and asserts ~2x-wide gaps**, not because
  the data are now reproducible.
* Cannot rule out: a CI-only draw outside the whole measured range. The per-draw
  distribution does have a right tail (max 1.91 at `n_top=200`), but no assertion here
  bounds the right tail.

## 11. не_сделано / риски

* **The residual bias is still there** (mean `R(100)` = 0.940, not 1.0). Deliberately, with
  the reasoning in §3. If a human wants it gone, that is an API decision — `n_top` as a
  fraction — not a bug fix, and it would touch callers outside my ownership. **Flagging,
  not deciding.**
* `src/interp/feature_composition.py` behaviour **unchanged** — if the campaign expected a
  source fix here, this task concluded it was not one.
* **I did not run the whole suite** (rt.py refuses it by design, and training is forbidden).
  Coverage beyond this file: `test_interp.py` (94 passed) and the gate's 5-file stage
  (78 passed). A full-suite run remains CI's job.
* **Linux unverified.** No Linux available; §10 is reasoning from measurements, not a run.
* Test file grew 1s -> 9.4s (`DRAWS=16`, four `n_top` values, two dataset sizes). Verified
  inside the gate's shared 90s budget (29.5s for 5 files), but it is now the second-largest
  file in that stage; further growth here would need `DRAWS` revisited.
* `_probe_*.py` scratch under `.agent-notes/` is gitignored per the repo's own rule; I also
  deleted two `.out` files that were **not** covered by that pattern so they could not be
  committed. Measurements quoted above are reproducible from those probes.
* `proofs/IMPLEMENTATION_STATUS.md` **not** updated: `src/interp/**` is not a GWP mechanism,
  the matrix has no row for it, and it is not in my file list.