# TASK-18 — `src/interp/workspace_validation.py`

status: **done**. Branch `agent/task-18b`, worktree `C:\dev\wt-18`, commit `c0ca2d0`.
Files touched: `src/interp/workspace_validation.py` (owned),
`tests/test_workspace_validation_honesty.py` (new).

## Card mismatch — read this first

The card text quoted in my prompt was **TASK-17's body** (`ablation.py`,
status already `done`, its own fix and test file present and green). The
authoritative card in `TASKS.md` — `### TASK-18`, `files_allowed:
src/interp/workspace_validation.py`, and the `requirement:` line I was
given verbatim ("a verdict computed from an untrained model must be
refused") — is the `workspace_validation.py` card. I followed the
authoritative card. `ablation.py` was not touched.

## The requirement, implemented

`validate_workspace_claim` no longer produces a verdict from an untrained
model. It raises `UntrainedSAEError` in both directions:

- `sae=None` used to build a random `TopKSAE`, log a warning, and return
  `workspace_claim_valid`. It now raises, and the message names the two
  honest alternatives (pass a trained SAE, or set `n_sae_train_steps>0`).
- an SAE passed in with zero optimizer steps is refused too.

The one opt-in that survives, `allow_untrained_sae=True`, returns a
refusal record with **no number in it**: `verdict_status=
"refused_untrained_sae"`, `workspace_claim_valid=None`, and no
`subspace_similarity` / `principal_angles` / `mean_angle` /
`similarity_ci_lower` / `claim_margin_above_placebo` keys at all. A test
asserts the absence of each of those by name, so "attach a caveat" cannot
regress into "attach a number".

`TopKSAE` now carries a persistent `n_train_steps` buffer. Only real
optimizer steps increment it (`train_topk_sae`, and `mark_trained()` for an
externally trained SAE). A forward pass does not, and the count survives a
`state_dict` round trip, so a checkpoint cannot inherit "trained".

## No-ops found, and why each is now impossible

**W1 — untrained model measured and reported.** Random decoder → decisive
looking verdict (the card's measured `subspace_similarity=0.447`,
`placebo=0.399`, `workspace_claim_valid=False`). Now raises; only
optimizer steps or an explicit declaration mark an SAE trained.

**W2 — "high probe accuracy" was in-sample.** `identify_workspace_features`
promised features predictive of downstream tasks and scored them by a
point-biserial correlation computed and evaluated on all N samples: a probe
scored on its own fitting data. Now a one-threshold probe is fitted on the
train split (threshold = train mean, orientation = train correlation) and
scored by **balanced accuracy on held-out rows**. `info` reports
`probe_accuracy_heldout` next to `probe_accuracy_chance`, plus
`workspace_features_identified`, `n_tasks_used`, `n_train`, `n_holdout`.
A caller can pass its own split via `train_mask` to keep the held-out claim
auditable. Non-binary tasks are skipped but **counted** in `n_tasks_used`, so
"no usable probe" is visible rather than silently scoring every feature 0.

**W3 — the wrong object for the angles, twice.**
`svdvals(Qᵀ P)` is the principal-angle cosines only when `k == r`. At k=4,
r=51 the old mean divides the same total by 51 instead of 4, and the
reported angle list is longer than the number of angles that exist. Now:
SVD of the decoder rows → orthonormal basis `Y` of `span(P)` truncated at
the numerical rank → eigenvalues of the k×k `Qᵀ Y Yᵀ Q` → `arccos(sqrt(λ))`.
`subspace_similarity` is in [0,1] by construction and is 1 exactly when
`span(Q) ⊆ span(P)`.

A defect I introduced and then found: my first version formed the Gram from
the **raw** decoder rows. `G` then scales with ‖P‖², eigenvalues reached 64
at D=64, and the `clamp(0,1)` I had written was turning an unbounded number
into a clean 1.0 — a fabricated verdict, exactly the class I was removing.
It is now built from an orthonormal basis, and the code raises if an
eigenvalue ever exceeds 1 by more than round-off rather than clamping it.
`test_similarity_does_not_depend_on_decoder_row_scale` (scales 1e-3…1e4) and
`test_matches_the_gram_matrix_definition` are the guards.

A second defect in the same line: I initially used `torch.linalg.qr` for
that basis. With a rank-deficient leading block — duplicated decoder rows,
the normal case when the workspace has m ≫ k features — LAPACK sets tau=0
and skips the reflector, so the later `Q` columns are an arbitrary
orthonormal completion *outside* `span(P)`. Measured on a 3-D span at D=16:
similarity 0.67 where the truth is 1.0, order-dependent (0.9999 with rows in
sorted order, 0.667 after a shuffle). The SVD's left singular vectors span
the column space by construction, which is why it is used instead.
`test_duplicate_decoder_rows_do_not_hide_the_span` pins it over three
different row orders.

**W4 — the promised CI did not exist.** The header promised "Bootstrap CI for
the similarity"; the bootstrap ran on `ws_util`, and the module's headline
number had no interval. `bootstrap_similarity_ci` now resamples the N
representations with their labels, re-runs feature identification on the
resample and recomputes the similarity, so the interval covers both which
samples were used and which features the probes picked. `decide_workspace_claim`
gates on the CI's **lower** end, not the point estimate.

**W5 — cleanup skipped, poisoning the next step.** `identify_workspace_features`
called `sae.eval()` and never restored the caller's mode, so an analysis call
could leave an SAE that was about to be trained in eval mode (and
`nn.Module` defaults to train mode, so it also silently changed the other
direction). Now wrapped in try/finally, restored on the error path too.

**W6 — degenerate inputs returned plausible-looking numbers.** An empty
workspace feature set returned `0.0` similarity / `90.0°` / `dim 0`, which
is indistinguishable from a measurement of an orthogonal workspace; it now
raises. A non-orthonormal `Q` gave undefined "angles" and could return a
similarity above 1; `Q` is now checked and rejected with a message naming
`torch.linalg.qr`. A non-finite decoder was reduced to a zero singular value
by a bare `except Exception: singular_values = [0.0]`; there is no exception
handler left in the module. Non-finite `Q`/`representations`, SAE/
representation dim mismatch, label count mismatch, out-of-range or
out-of-range-shaped indices all raise.

**W6b — a single placebo draw, compared point-to-point.** The
shuffled-basis control was one random basis. Now `n_placebo` draws (default
8, device-aware private generators) and the verdict must beat the **worst**
of them; mean and max are both reported.

**W7 — a zero-width interval that looks like a tight measurement.**
`bootstrap_ci` on fewer than 2 samples returned `(m, m, m)`. Now raises;
also rejects non-finite samples, `n_bootstrap < 1`, and confidence outside
(0,1). Non-finite values are moved to CPU once and used (the old
`values.cpu().numpy()` result was discarded, and a CPU index tensor was
used to index `values` regardless of its device).

**Bonus: no global-RNG consumption.** The split, the placebo draws and the
bootstraps all use private `torch.Generator`s, so running this pipeline no
longer perturbs the caller's RNG (and so cannot perturb training).

## verify — FULL paste

```
$ & $PY tools\rt.py tests/test_workspace_validation_honesty.py tests/test_interp.py tests/test_v025_integration.py --slow
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_workspace_validation_honesty.py tests/test_interp.py tests/test_v025_integration.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 195 items

tests\test_workspace_validation_honesty.py ............................. [ 14%]
...........................                                              [ 28%]
tests\test_interp.py ................................................... [ 54%]
...........................................                              [ 76%]
tests\test_v025_integration.py ......................................... [ 97%]
....                                                                     [100%]

============================= 195 passed in 6.15s =============================
```

Also green, in separate budgeted runs:

```
$ & $PY tools\rt.py tests/test_ablation_module.py tests/test_statistical_tests.py tests/test_layer_analysis.py tests/test_causal_intervention.py tests/test_probes_split.py tests/test_feature_composition.py tests/test_interp_ground_truth_thresholds.py
collected 178 items ... 178 passed in 25.07s

$ & $PY tools\rt.py tests/test_determinism.py tests/test_info_and_disentangle.py tests/test_index_and_cka.py tests/test_run_comparison.py
collected 132 items ... 132 passed in 11.14s
```

Lint and format (ruff then black, per repo convention):

```
$ & $PY -m ruff check src/interp/workspace_validation.py tests/test_workspace_validation_honesty.py
All checks passed!
$ & $PY -m black --check src/interp/workspace_validation.py tests/test_workspace_validation_honesty.py
All done! ✨ 🍰 ✨
2 files would be left unchanged.
```

## diff-stat

```
 src/interp/workspace_validation.py         | 900 ++++++++++++++++++++++++-----
 tests/test_workspace_validation_honesty.py | 793 +++++++++++++++++++++++++
 2 files changed, 1538 insertions(+), 155 deletions(-)
```

## Mutation verdict, per instance

A driver restored each lie one at a time against the committed test file and
recorded which test(s) went red. Every line below is a test that failed.

| # | no-op restored | red test(s) |
|---|---|---|
| W1a | `sae=None` builds a random `TopKSAE` and continues | `TestUntrainedSAERefusal::test_no_sae_is_refused_not_replaced_by_a_random_one` |
| W1b | `elif not sae.is_trained:` → `elif False:` | `test_untrained_sae_passed_in_is_refused`, `test_explicit_opt_in_returns_a_refusal_carrying_no_number` |
| W2 | identification fitted **and** scored on all N rows | `test_reports_heldout_accuracy_next_to_chance`, `test_in_sample_correlation_is_not_reported_as_accuracy`, `test_a_supplied_split_is_honoured_exactly`, `test_split_too_small_to_hold_out_raises`, `test_unusable_task_count_is_reported` (5) |
| W3 | `eigvalsh(gram)` → `svdvals(M).pow(2)` (the original cross-matrix object) | `TestSimilarityMath::test_matches_the_gram_matrix_definition`, `test_narrower_sae_workspace_reports_only_min_k_r_defined_angles` |
| W3b | orthonormalise with QR instead of SVD (my first bug) | `test_duplicate_decoder_rows_do_not_hide_the_span` |
| W3c | form the Gram from raw rows, `U = P` | 8 tests: `test_identical_subspaces_score_one`, `test_wide_sae_workspace_does_not_dilute_the_similarity`, `test_matches_the_gram_matrix_definition`, `test_similarity_does_not_depend_on_decoder_row_scale`, `test_unrelated_subspaces_do_not_score_one_however_long_the_decoder`, `test_duplicate_decoder_rows_do_not_hide_the_span`, `test_similarity_is_bounded_by_construction`, `test_narrower_sae_workspace_reports_only_min_k_r_defined_angles` |
| W4 | bootstrap the similarity CI on `ws_util` instead | `test_similarity_has_a_bootstrap_ci_from_its_own_promises`, `test_ci_is_computed_on_the_similarity_not_on_ws_utilisation` |
| W4b | verdict from the point estimate only | `TestVerdict::test_pipeline_refuses_a_high_point_estimate_with_a_low_ci` |
| W5 | `sae.eval()` with no restore | `test_eval_mode_is_restored_after_identification`, `test_mode_is_restored_when_identification_raises` |
| W6a | empty workspace set returns `0.0` / `90°` | `test_empty_workspace_feature_set_raises` |
| W6b | orthonormality check disabled | `test_non_orthonormal_q_raises` |
| W6c | non-finite decoder swallowed to zeros (`except Exception`) | `test_non_finite_decoder_raises_not_zero` |
| W6d | placebo max → placebo mean | `TestVerdict::test_the_control_is_the_worst_draw_not_the_average` |
| W7 | `bootstrap_ci` returns `(m, m, m)` for n < 2 | `test_bootstrap_ci_on_one_sample_raises` |
| W8 | constructor marks the SAE trained | 7 tests in `TestUntrainedSAERefusal`, incl. `test_a_forward_pass_does_not_make_an_sae_trained`, `test_n_train_steps_survives_a_state_dict_round_trip` |

Two tests are **not** load-bearing and I do not claim them as such:
`test_training_an_sae_does_not_disturb_the_global_rng` (its
`torch.allclose(expected, after)` line asserts a property of the
`torch.Generator`, not of the code) — but
`test_pipeline_does_not_consume_the_global_rng` does pass only because the
pipeline uses private generators, and W2's private-generator path is what
`train_topk_sae` relies on. `test_import` in `test_v025_integration.py` was
already a `pass` before this card and I left it alone.

## не_сделано / риски

- **Not done:** no SAE is trained by default anywhere in the repo; I did not
  search for or wire up callers of `validate_workspace_claim` because the
  grep found none outside `workspace_validation.py` itself and
  `tests/test_v025_integration.py` (which only imports `TopKSAE`,
  `bootstrap_ci` and `compute_workspace_similarity`, all still compatible —
  verified above). `src/interp/__init__.py` does not export anything from
  this module and I did not change it (forbidden: other `src/interp/**`).
- **API break, deliberately.** `validate_workspace_claim` now raises where
  it used to return, `n_bootstrap` default dropped 1000→200 (the similarity
  CI re-runs identification per resample; 1000 would be ~5x the cost for no
  useful extra precision), `n_sae_features`/`sae_k` now only affect the
  train-here path, and `compute_workspace_similarity` returns two new keys.
  Any external caller needs a look; I found none in-tree.
- **Risk: the similarity is now stricter.** Old code at k=4, r=51 could
  report a similarity for a 51-dimensional SAE workspace; the new one is
  bounded by `min(k, r)` and cannot be diluted by width. Any historical
  number computed with the old formula is not comparable — the fix should be
  read as invalidating old `subspace_similarity` values, not reinterpreting
  them.
- **Risk: the CI is expensive.** `bootstrap_similarity_ci` runs full
  identification per resample (encode once, outside the loop). At the
  shipping defaults (`n_sae_features=8192`, `n_bootstrap=200`) this is the
  most expensive thing in the module by an order of magnitude. Tests use
  tiny dims. I did not profile it at realistic sizes; a caller running this
  in a sweep should know.
- **Not done:** `identify_workspace_features` assumes **binary** probe labels
  (point-biserial, as the original did). A non-binary task is skipped and
  counted, not fitted with a multiclass probe. Same limitation as before,
  now visible in `n_tasks_used` instead of invisible.
- **Not done:** `proofs/IMPLEMENTATION_STATUS.md` untouched — this module is
  `src/interp/**`, not `src/models/`, and the card lists no matrix row for
  it. If the matrix does cover interpretability modules somewhere I did not
  look, that row may now be stale.
- I never trained anything and never ran the full suite; all runs went
  through `tools/rt.py` with a named file.