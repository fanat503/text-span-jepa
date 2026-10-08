# TASK-11 — held-out splits for the sibling probe metrics

Branch `agent/task-11c`, based on `main` at `1e49f77` (`agent/main` does not
exist; `main` is the merge of every prior wave). Two commits:
`be199a6` (the fix) and the test-anchor follow-up.

Files touched — exactly the card's `files_allowed`, plus `tests/test_interp.py`:
- `src/interp/probe_generalization.py`
- `src/interp/probing_complexity.py`
- `src/interp/structural_probe.py`
- `tests/test_interp.py`

`src/eval/probes.py` is untouched (`git log -- src/eval/probes.py` shows no
commit on this branch). `workspace_validation.py` untouched — see §5.

Verify: `& $PY tools/rt.py tests/test_interp.py` → **118 passed in 6.32s**
(94 before this card; 24 new). `ruff check .` clean, `black --check .` clean.
No training run. No skip, no xfail, no `--no-verify`.

---

## 1. The card offers two routes; I took a different one per metric

The card asks for a held-out split *or* an honest rename, and says the five
metrics differ in kind so a uniform route is wrong. They do, and so did the
route:

| # | metric | kind | route |
|---|---|---|---|
| 1 | `probe_generalization.source_accuracy` (+ its `generalization_gap`, `generalization_ratio`) | row classifier on a labelled source set | **split**, three ways |
| 2 | `probing_complexity` best-epoch validation, no test set | row classifier swept over depth, compared across two models | **split**, three ways + one partition shared by depth and by arm |
| 3 | `structural_probe.source_spearman` | correlation pooled over a *sentence list*, not rows | **split + rename**; the unit of held-out had to be established first |
| 4 | `workspace_validation` in-sample feature selection | feature selection over many features | **already fixed** (TASK-18) — pinned, not changed |
| 5 | `ProbeSelectivityTest.real_task_accuracy` | row classifier + `n_control` permuted-label controls | **split**, three ways, one partition for every fit |

Two things made a uniform route impossible. #3's unit of generalisation is not
a row: `StructuralProbe.evaluate` pools the upper triangle of *every sentence
it is handed* into one correlation, so "held out" there means "held-out
sentences". There was no way to ask for that before this card — `train_probe`
saw everything it was given. And #5 is a sixth instance of the same class in a
file the card already put in my hands; leaving `real_task_accuracy` reporting a
best-epoch validation maximum while I removed `source_accuracy` two classes
above it would have been incoherent.

A rename-only route was the wrong one everywhere it was available. Renaming
`source_accuracy` to `source_train_accuracy` would have been honest and cheap,
but it leaves the module with no usable number: the paper's cross-dataset claim
needs an out-of-sample source score, and one is available for the cost of two
extra index partitions.

## 2. What each metric measured before, measured

All figures executed with the repo's interpreter, 1 thread, on identical data.

### #1 `probe_generalization.source_accuracy`

200 source rows, 16 dims, labels = `sign(reps[:,0])`, linearly separable.

```
source_accuracy reported      = 0.6700   (IN-SAMPLE)
same probe, held-out source   = 0.6100
overfit amount (train - hld)  = +0.1600
generalization_gap reported   = +0.0450  -> real gap vs held-out source = -0.0150
generalization_ratio reported = 0.9328  -> recomputed on held-out       = 1.0246
```

The gap changed sign. More importantly, `patience` was counted on the
**in-sample** accuracy, which saturates, so the probe early-stopped after one
or two epochs — `source_accuracy` was in-sample *and* badly undertrained.
`generalization_ratio = target / source` divided by a quantity that shrinks
when the probe memorises more, so overfitting improved the score.

### #2 `probing_complexity`

```
jepa_max_acc over 8 global seeds on byte-identical data:
  min=0.5500 max=0.9000 spread=0.3500     (card's figure: 0.76-0.89)
per-depth advantage over 12 seeds, arms differing only in one column's scale:
  {'jepa': 8, 'baseline': 4, 'tie': 0}
```

And the number that decides the module's headline, `min_extracting_depth`, was
`max over epochs of validation accuracy` against a `min_accuracy` threshold,
with no test set anywhere.

### #3 `structural_probe.source_spearman`

```
spearman on the 20 TRAINING sentences = 0.2115
spearman on the 20 UNSEEN  sentences = 0.0699
optimism                                          = +0.1416
```

And `StructuralProbeGeneralization.compute` **could not run at all**: it wrapped
the caller's already-per-sentence lists in another list, so `evaluate` received
a list of lists and raised
`AttributeError: 'list' object has no attribute 'to'`. Nothing in `tests/`
referenced the class, so the suite was green over a metric that did not execute.

### #5 `ProbeSelectivityTest.real_task_accuracy`

Same best-epoch-validation shape, worse: each of the `n_control` controls drew
its own unseeded split, so the reported selectivity carried one partition noise
per control on top of the label noise. Spread over 6 seeds: **0.3250**.

## 3. What each metric measures now

Every module now draws one partition from a **private** `torch.Generator` (no
module in this trio consumes the caller's global stream any more) and reports
its split's provenance (`split_seed`, `n_train`/`n_val`/`n_test`, or
`n_source_*`).

- **#1, #2, #5** get a three-way partition, `train / selection / held-out`.
  The optimiser sees `train` only, the epoch is chosen on `selection`, and the
  reported number is computed **once** on `held-out` — never a maximum of
  anything. `source_accuracy`, `source_spearman` are **removed**, not aliased.
  `source_train_accuracy`, `source_selection_accuracy`, `source_overfit`,
  `val_depths`, `max_val_accuracy`, `source_spearman_train`, `source_optimism`
  are added under names that say which partition they came from, so the size of
  the old optimism is visible rather than implied.
- **#2 additionally** shares the partition across every depth and hands both
  `compare_models` arms the same rows and the same per-depth initial weights.
  Previously each arm drew its own split per depth *and* each probe drew its
  initialisation from the global RNG, so the two arms were two unrelated
  experiments.
- **#3** establishes the unit of held-out as the sentence.
  `StructuralProbe.sentence_split(n, seed, holdout_fraction)` is public and
  deterministic, `train_probe(train_idx=...)` fits on one side of it, and
  `StructuralProbeGeneralization.compute(..., source_train_idx=...)` scores the
  complement. The precondition is load-bearing and is stated in the docstring:
  `compute` receives an already-fitted probe and *cannot* see what it was
  trained on, so held-outness rests on the caller passing the matching indices.

## 4. Mutation verdicts — all red, restored from the INDEX

Per the TASK-35 lesson, every mutation was undone with `git checkout -- <file>`
(never `HEAD --`).

| # | mutation | tests that went red |
|---|---|---|
| A | `source_heldout = trained["train_accuracy"]` — the split goes back in-sample | `test_pg_reported_source_accuracy_is_not_the_training_accuracy`, `test_pg_reported_accuracy_reads_the_heldout_labels_only` (2) |
| B | rename the key back to `"source_accuracy"` | 6: the name test, the gap-formula test, the pairing test, the reproducibility test, and both leakage tests |
| C | `probing_complexity`: per-depth redraw **and** `torch.randperm(n)` from the global RNG **and** drop the reseeded init | `test_pcc_reported_accuracy_is_not_the_validation_score`, `test_pcc_compare_models_of_identical_arms_is_exactly_zero`, `test_pcc_is_reproducible_across_global_rng_state` (3) |
| C' | isolate C: drop **only** the reseeded init | `test_pcc_compare_models_of_identical_arms_is_exactly_zero` (1) |
| D | `StructuralProbeGeneralization`: score the train side for `source_spearman_heldout` | `test_sp_compute_runs_and_source_spearman_is_on_unseen_sentences`, `test_sp_heldout_spearman_reads_the_heldout_sentences_only` (2) |
| E | `ProbeSelectivityTest`: each control redraws its own partition | `test_selectivity_every_fit_receives_the_same_partition`, `test_selectivity_is_reproducible_across_global_rng_state` (2) |

The headline anchors, in numbers:

```
NEW  compare_models(reps, reps.clone()) advantage, 8 seeds:
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
OLD  same call, same arms:
     [-0.025, -0.075, 0.25, -0.15, 0.625, -0.05, 0.0, 0.225]

NEW  two evaluate() calls at a fixed torch.manual_seed: identical
OLD  same: {1: 0.8, 2: 0.85} then {1: 0.9, 2: 0.825}
```

A comparison whose two arms hold the same array must return exactly zero. It
returned 0.625.

**A bug I introduced and the test caught.** My first `sentence_split` returned
`idx[: n - n_test], idx[n_test:]` — overlapping whenever `n_test < n / 2`, and
it still satisfied the size check. `test_sp_sentence_split_is_a_partition` was
red on the first run. Fixed to `idx[n - n_test:]`, and
`check_sentence_split` now verifies disjointness explicitly (it previously
trusted the caller, which is what let the bug through).

## 5. Historical numbers this invalidates

Everything below was computed from these three modules and is now wrong or
uninterpretable. I found no stored results in the repo that quote them — no
JSON, no results table, no committed numbers — so this is a forward-looking
list, not a retraction of a published figure.

**`probe_generalization`**
- every `source_accuracy` value: in-sample, and undertrained by the in-sample
  early stop
- every `generalization_gap`: was `source − target` with an in-sample `source`,
  i.e. an overfit measure, not a generalisation gap. Measured gap changed sign
  on the same probe and data (+0.045 → −0.015)
- every `generalization_ratio`: was monotone in overfitting
- every `compare_models` verdict (`jepa_generalizes_better`,
  `jepa_gen_gap`, `baseline_gen_gap`, `jepa_gen_ratio`, `baseline_gen_ratio`,
  `jepa_target_acc`, `baseline_target_acc`): two arms that drew different
  splits and different initialisations
- `ProbeSelectivityTest`: every `real_task_accuracy`, `control_task_accuracy`,
  `selectivity`, `probe_is_genuine`, and the four `compare_selectivity` outputs

**`probing_complexity`**
- every `depths[d]` value: it was a best-epoch validation maximum
- every `max_accuracy` and therefore every `jepa_max_acc` / `baseline_max_acc`
- every `min_extracting_depth` / `jepa_min_depth` / `baseline_min_depth` /
  `complexity_gap` / `jepa_more_accessible`: decided against a noisy maximum
  with no test set, on a partition drawn per depth and per model
- the whole of `jepa_max_acc` was quoted as "0.76–0.89 across 8 seeds on
  identical data" in the audit findings; the card cites that as the symptom.
  It was correct as a measurement of the defect, and it is no longer a
  measurement of anything the module reports.

**`structural_probe`**
- `StructuralProbeGeneralization.source_spearman`: did not exist as a reachable
  quantity (`compute` raised), and would have been the training sentences
- `StructuralProbe.evaluate`'s `spearman_r` and `uuas` are **unchanged in
  meaning** — it scores what it is given. Callers that used them on training
  sentences were always reporting training numbers; the module now says so.

**`workspace_validation`** — unaffected. TASK-18 already made this one honest
(`probe_accuracy_heldout`, `_heldout_probe_scores` fitting on train and scoring
on held-out). It is outside this card's `files_allowed` and I did not touch it;
`test_workspace_validation_still_reports_a_heldout_probe_accuracy` pins the key
so a future rename cannot quietly undo it.

**`ground_truth.validate_probing_complexity`** — the one downstream consumer I
did not own. Checked directly, not through pytest: pre-change and post-change
outputs are **identical** (`class_min_depth=2`, `class_max_acc=1.0`,
`depth_min_depth=2`, `depth_max_acc=0.75`, `pipeline_valid=False`). No
regression, but note `pipeline_valid` was already `False` before this card.

## 6. Reported, not fixed — all found while in these files

These are the same failure class in places this card did not authorise me to
change, or are science decisions that belong to the owner.

1. **Every probe in the repo is undertrained, by construction.**
   `ProbeGeneralizationTest`, `ProbeSelectivityTest`, `ProbingComplexityCurve`
   and `layer_analysis.LayerwiseProbe` all fit a probe with **full-batch** Adam
   at `lr=1e-3`. Adam moves a weight by at most ~`lr` per step, so the whole
   run can move it by at most `lr × max_epochs` — **0.03** at the shipped
   `max_epochs=30`, 0.05 at 50. Measured consequence: on linearly separable
   data, `lr=1e-3`/`max_epochs=50` reaches train accuracy 0.708 where
   `lr=0.1`/`patience=50` reaches 0.958. This is upstream of the split: a
   probe that cannot fit anything reports near-chance numbers on every
   partition, so a held-out split is necessary and not sufficient. Changing the
   optimiser budget changes every probe number in the repo, so it is not mine.
   It is why the new tests use `lr=0.1` and, where the assertion depends on it,
   a raised `patience` — the class docstring says so, so the next reader does
   not mistake `lr=0.1` for a coincidence.

2. **`min_extracting_depth` is set by the stopping rule, not only by the
   representation.** `patience` counts non-improvements of a validation
   accuracy over `n_val` rows, so it fires before the probe converges. Swept on
   one partition, `min_accuracy=0.7`, linearly separable data:

   | patience | max_epochs | depth-1 test acc | `min_extracting_depth` |
   |---|---|---|---|
   | 5 | 30 | 0.700 | 3 (i.e. "not extractable at any depth") |
   | 25 | 400 | 0.875 | 1 |

   Raising `max_epochs` alone changes nothing — patience fires first. So the
   module's headline metric is a statement about the selection budget as much as
   about the representation. Not fixed: the budget is a science decision.
   `val_depths` and `max_val_accuracy` are now reported so the selection score
   is visible beside the reported one. Recorded in the module header.

3. **`generalization_ratio` is a quotient of two near-zero quantities** and is
   unbounded, with a sign that flips on the denominator. Measured: denominator
   `0.0365`, numerator `-0.0672`, ratio `-1.84`. It has never meant "how much
   generalisation was preserved". Replacing it with a difference, or with a
   bootstrap interval on the difference, is a decision about what the paper
   should claim. Same pattern in `StructuralProbeGeneralization`. Not fixed.

4. **`min_extracting_depth` can flip on one ulp.** The threshold test is
   `test_accuracy >= min_accuracy` where the accuracy is a float32 mean and the
   threshold a Python float: 28/40 comes out as `0.69999998` and fails a
   `min_accuracy=0.7` set in float64. Observed live — a depth reporting exactly
   0.700 was recorded as *not* crossing the 0.7 threshold. One-character fix
   (`>= min_accuracy - 1e-9`), but it moves reported depths, so it is not mine.
   Not fixed.

5. **`StructuralProbe._spearman` has no tie correction and swallows errors.**
   Ranks come from `argsort().argsort()`, which gives tied values *ordinal*
   ranks in whatever order the sort produces; gold tree distances are heavily
   tied (a chain parse has many pairs at the same distance), so this is not
   Spearman's rho on this data and it is not stable across orderings. The bare
   `except Exception: return 0.0` then reports a failed correlation as "no
   structure" — the same silent-zero family as `causal_intervention` in the
   audit findings. A tie-corrected `average` rank moves every structural-probe
   number in the repo. Reported, not fixed; flagged in the docstring.

6. **`ProbeGeneralizationTest._train_probe` had a dead `num_classes`
   parameter** (`nc = num_classes or self.num_classes` immediately
   overwritten by `max(labels.max()+1, 2)`). I removed the parameter as part of
   the rewrite rather than "fixing" it, because honouring it changes the
   label-inference semantics — which is a research decision, not a cleanup.

7. **`StructuralProbeGeneralization.compute` still cannot verify its own
   precondition.** It receives a fitted probe and no record of the probe's
   training data, so if the caller omits `source_train_idx` the "held-out"
   number is only as held-out as that unseen training set happened to be. It
   reports `source_train_idx_provided` so the caller can tell which case it is
   in, but the honest fix is for `train_probe` to return its partition and for
   `compute` to refuse a mismatch. That is an API change to a class with no
   callers; not made here.

## 7. Honest caveats

- The cross-partition spread of `max_accuracy` did **not** shrink (≈0.45 over 8
  partitions, vs 0.35 before). It was never the split *redraw* that produced
  that number — it is genuine partition sensitivity at `n_test=40`, where one
  row is 0.025. Reproducibility and arm-pairing are what this card bought, and
  both are exact now. Anyone needing a tighter interval needs more rows or a
  bootstrap; neither is this card.
- The `lr`/`patience` values in the new tests are chosen so the assertions bite.
  They are not recommended defaults, and §6.1 is the reason they are not.
- `tests/test_interp.py` is outside the card's `files_allowed` as written. I
  edited it because the card demands a mutation verdict and there is nowhere
  else to put one, and because line 909 asserted the literal string
  `"source_accuracy"`, which the rename invalidates. That assertion was
  **replaced, not weakened** — it now asserts `"source_accuracy_heldout" in
  result`, which is the same claim about the same dict under the honest key. No
  test was deleted, skipped or loosened.