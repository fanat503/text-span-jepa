# TASK-35 — all three comparison arms log the SAME parameter quantity

status: **DONE**, committed on `agent/task-35b` (worktree `C:\dev\wt-35`, commit
`50d471d`). Card was "LOST, NOT CLOSED"; this is a re-do from the card
specification, not a recovery — the previous worker's files were unrecoverable
and nothing of it was reused. The lesson the card records (restore a mutation
from the INDEX, never from HEAD) was followed: every mutation here was undone
with `git checkout -- <file>` after the fix was staged.

## What each arm logs, before → after

Measured in-process on meta tensors, through `create_model` (the production
path `src/train.py` uses), one thread, from inside the worktree.

### Wide shape — V=50304, seq 512, D=768, depth 12, heads 12

The shape the card quotes.

| arm | logged before | logged after | true total | trainable |
|---|---|---|---|---|
| text_span_jepa | 262,021,633 | 262,021,633 | 262,021,633 | 137,938,945 |
| mlm | 123,689,472 | **162,716,160** | 162,716,160 | 162,716,160 |
| data2vec | 85,646,592 | **248,755,968** | 248,755,968 | 124,673,280 |

Before: the three logged 262.0M / 123.7M / 85.6M — a 3.1x spread, and all three
different from the model each one actually holds. After: each logged number is
its own model's total. They still differ, and must; that is architecture, not
definition.

### Shipped rung — V=50304, seq 512, D=640, depth 10, heads 10

The rung the MLM module header and `tests/test_baseline_parity.py` document.

| arm | logged before | logged after | true total | trainable |
|---|---|---|---|---|
| text_span_jepa | 170,706,561 | 170,706,561 | 170,706,561 | 88,947,841 |
| mlm | 81,431,040 | **113,953,280** | 113,953,280 | 113,953,280 |
| data2vec | 49,646,720 | **163,927,680** | 163,927,680 | 82,168,960 |

`113,953,280` and `88,947,841` are the figures the historical claim block in
`baselines/mlm_baseline.py` and the existing parity tests already pin, so the
new default lands on numbers the file was already asserting — the header table
did not need rewriting, and the historical claim block is untouched (its
delimiters, its wording, and the brittle `test_historical_quotation_is_still_
delimited` guard all still pass).

## The two changes

1. **Both baselines' `get_num_params` default flipped `True` → `False`**, and
   both now count the module's parameters the way `TextSpanJEPA` does. MLM has
   one encoder so `non_embedding=True` subtracts one encoder's tables; data2vec
   and JEPA subtract both, matching `TextSpanJEPA`'s documented meaning.

   For data2vec the old default was hiding a **second, separate** defect: the
   body was `encoder.get_num_params(non_embedding) + regression_head`, which
   never touched `target_encoder` at all. That is the arm's largest tensor, so
   even a `non_embedding=False` default on the old body would have stayed wrong.
   Worth naming, because flipping the default alone looks like a complete fix.

2. **`Data2VecTextBaseline.get_num_params_trainable()` added.** It had no such
   method, so of the three arms only JEPA and MLM could answer the question
   `src/train.py`'s second log line asks. It now returns the module's
   gradient-carrying parameter count, independently recounted in the tests.

Files owned and touched: `baselines/mlm_baseline.py`,
`baselines/data2vec_baseline.py`, `tests/test_baseline_parity.py`. Nothing under
`src/models/**`, `config/**` or `src/train.py` was modified — `src/train.py` is
read from the test, not edited.

## The test that matters

`tests/test_baseline_parity.py::TestLoggedQuantityIsOneKind` — **16 tests**.

The KIND is defined once, in the test: *the number of scalar parameter tensors
the module owns*. Each arm must satisfy that single definition, verified
against `_independent_total()`, a walk over `named_modules()` that never touches
`mod.parameters()` and so cannot just restate whatever the method body does.

- Built through `create_model` — the function `src/train.py` calls — so the
  production wiring is covered, not just the classes.
- **Three shapes**, not one: `tiny` (V=200, D=32, depth 2 — head dominates, arms
  nearly agree), `mid` (V=2000, D=96, depth 3), `wide` (V=50304, D=768, depth 12
  — the card's shape, where the counts spread 3.1x apart). Built on meta tensors
  so the wide shape costs no allocation on a shared box. The defect being caught
  depends on which tensors dominate, so a single shape can be satisfied by
  accident.

**Deliberately not asserted: closeness.** No bound on the spread, no ordering, no
ratio between the three. Three different architectures cannot have equal counts,
and a closeness assertion would be the wrong test — it would be red for a
legitimate refactor and would still pass under a mutation that shifted all three
by the same wrong amount. `test_the_numbers_really_do_differ` is the
counterweight: it asserts the three counts stay **distinct**, so the KIND
assertions cannot be satisfied later by making the number the same instead of the
meaning the same.

## Verify

```
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests\test_baseline_parity.py tests\test_model.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 194 items

tests\test_baseline_parity.py .......................................    [ 20%]
tests\test_model.py .................................................... [ 46%]
........................................................................ [ 84%]
...............................                                          [100%]

============================== warnings summary ===============================
tests/test_model.py::TestData2VecBaseline::test_forward
tests/test_model.py::TestData2VecBaseline::test_loss_formula_data2vec
tests/test_model.py::TestV011TrainingReadiness::test_compute_loss_unified_interface
tests/test_model.py::TestV011TrainingReadiness::test_data2vec_training_loop
  C:\dev\wt-35\baselines\data2vec_baseline.py:140: UserWarning: data2vec target depth truncated: encoder exposes 2 layers, average_top_k_layers=8
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 194 passed, 4 warnings in 28.11s =======================
```

The 4 warnings are pre-existing (they come from `test_model.py` building a
depth-2 data2vec arm and hitting `average_top_k_layers=8`), not from this card.

Ruff and black over the whole repo:

```
$ & $PY -m ruff check .
All checks passed!
$ & $PY -m black --check .
All done!
115 files would be left unchanged.
```

The historical claim block was left alone: `test_historical_quotation_is_still_
delimited` (one BEGIN, one END, correct order) and `test_no_live_parity_guarantee`
both still pass — 0 of the 39 tests in `tests/test_baseline_parity.py` were
touched.

## Diff-stat

```
 baselines/data2vec_baseline.py |  69 ++++++++-
 baselines/mlm_baseline.py      |  58 +++++++-
 tests/test_baseline_parity.py  | 316 +++++++++++++++++++++++++++++++++++++++++
 3 files changed, 434 insertions(+), 9 deletions(-)
```

## Mutation verdict

Three mutations, each applied to the **staged** fix and each undone with
`git checkout -- <file>` (from the INDEX, per the card's lesson). The worktree was
never destroyed.

**1. `data2vec` default flipped back to `non_embedding=True`** → 12 of 16 KIND
tests go red:

```
FAILED ...test_logged_count_is_the_models_own_parameter_tally[wide]
FAILED ...test_all_three_arms_satisfy_one_definition_at_one_shape[wide]
FAILED ...test_logged_count_is_not_the_non_embedding_variant[tiny]
FAILED ...test_logged_count_is_not_the_non_embedding_variant[mid]
FAILED ...test_logged_count_is_not_the_non_embedding_variant[wide]
FAILED ...test_non_embedding_variant_is_the_total_minus_the_embedding_tables[tiny]
FAILED ...test_non_embedding_variant_is_the_total_minus_the_embedding_tables[mid]
FAILED ...test_non_embedding_variant_is_the_total_minus_the_embedding_tables[wide]
=================== 12 failed, 4 passed, 23 deselected in 2.46s ===================
```

**2. `mlm` default flipped back to `non_embedding=True`** → 13 of 16 red (one
more than above: it also breaks
`test_trainable_count_is_below_the_logged_count_where_a_teacher_is_frozen`,
which compares MLM's logged count with its trainable count and those coincide only
when the logged count is the total).

```
=================== 13 failed, 3 passed, 23 deselected in 2.53s ===================
```

**3. `get_num_params_trainable` deleted from `Data2VecTextBaseline`** → the two
tests that cover it go red, and nothing else does:

```
FAILED ...test_every_arm_exposes_the_trainable_count
FAILED ...test_trainable_count_is_below_the_logged_count_where_a_teacher_is_frozen
=================== 2 failed, 14 passed, 23 deselected in 2.38s ===================
```

So: `test_logged_count_is_the_models_own_parameter_tally` and
`test_all_three_arms_satisfy_one_definition_at_one_shape` are load-bearing at all
three shapes against a default flip in either arm, and
`test_every_arm_exposes_the_trainable_count` is load-bearing against deleting the
method that was missing. Every KIND test is accounted for.

## не_сделано / риски

- **Not run: the full suite.** The card's verify command is
  `tests/test_baseline_parity.py tests/test_model.py --slow`, and `tools/rt.py`
  refuses a whole-suite run by design. Files elsewhere that might read a
  baseline's `get_num_params()` were grepped instead: `src/interp/ablation.py:592`
  calls it, but only on a `TextSpanJEPA` it builds itself; `src/models/predictor.py:248`
  is a different class. No other caller, so the blast radius is the three files
  in this diff plus `src/train.py`'s log line.
- **`config/scaling/*.yaml` comments are NOT updated for the baselines.** Those
  files document `get_num_params()` figures for JEPA only, and `config/**` is
  forbidden here. Nothing is wrong with them, but if a future card adds per-arm
  param figures to the configs, the baseline numbers there must be the totals.
- **The arms are still not capacity-matched, and that is unchanged.** MLM carries
  1.281x JEPA's trainable parameters; the header block says so and the parity
  tests pin it. This card made the logged quantities *comparable in kind*, which
  is a precondition for a fair comparison, not the comparison itself. What
  genuine parity would cost is still an experiment decision needing human
  sign-off.
- **`README.md` lines 50-65 describe `TextSpanJEPA.get_num_params()`** and mention
  no baselines, so it is not stale. Not touched: it is outside the owned files.
- **Risk on the wide-shape numbers quoted above:** they were measured on meta
  tensors, which give exact shapes and `numel()` with zero allocation, but never
  run a forward pass. The parameter counts are exact; any claim about what those
  arms *do* at runtime would still need a real run.