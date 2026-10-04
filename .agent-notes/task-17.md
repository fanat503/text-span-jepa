# TASK-17 — ablations can no longer be labelled while doing nothing

- **status**: done
- **files changed**:
  - `src/interp/ablation.py` (+251 / −44)
  - `tests/test_ablation_module.py` (new, 473 lines, 22 tests)
  - branch `agent/task-17`, commit `ebb61b6`, worktree `C:\dev\wt-17`. No push.

All four defects share one root cause and one fix shape: **the module never
checked that the ablation happened.**

---

## verify

### 1. RED — new tests against the *unmodified* `src/interp/ablation.py`

```
$ & $PY tools\rt.py tests\test_ablation_module.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 22 items

tests\test_ablation_module.py FFFFFFFF..F.FFFFFFF..F                     [100%]

================================== FAILURES ===================================
_ TestMissingTargetIsLoud.test_missing_info_key_raises_instead_of_doing_nothing _
tests\test_ablation_module.py:188: in test_missing_info_key_raises_instead_of_doing_nothing
    with pytest.raises(RuntimeError) as excinfo:
E   Failed: DID NOT RAISE RuntimeError
___ TestMissingTargetIsLoud.test_missing_weight_on_the_model_config_raises ____
tests\test_ablation_module.py:196: in test_missing_weight_on_the_model_config_raises
    with pytest.raises(RuntimeError) as excinfo:
E   Failed: DID NOT RAISE RuntimeError
__ TestMissingTargetIsLoud.test_flag_with_no_mechanism_raises[use_predictor] __
tests\test_ablation_module.py:206: in test_flag_with_no_mechanism_raises[use_predictor]
    with pytest.raises(RuntimeError) as excinfo:
E   Failed: DID NOT RAISE RuntimeError
__ TestMissingTargetIsLoud.test_flag_with_no_mechanism_raises[use_target_centering] __
tests\test_ablation_module.py:206: in test_flag_with_no_mechanism_raises[use_target_centering]
    with pytest.raises(RuntimeError) as excinfo:
E   Failed: DID NOT RAISE RuntimeError
_______ TestMissingTargetIsLoud.test_error_type_is_public_and_specific _______
tests\test_ablation_module.py:211: in test_error_type_is_public_and_specific
    assert issubclass(abl_mod.AblationTargetMissingError, RuntimeError)
E   AttributeError: module 'src.interp.ablation' has no attribute 'AblationTargetMissingError'
_ TestMissingTargetIsLoud.test_allow_missing_targets_is_the_explicit_escape_hatch _
tests\test_ablation_module.py:225: in test_allow_missing_targets_is_the_explicit_escape_hatch
    with pytest.warns(RuntimeWarning, match="loss_future"):
E   Failed: DID NOT WARN. No warnings of type (<class 'RuntimeWarning'>,) were emitted.
_ TestMissingTargetIsLoud.test_every_standard_ablation_either_applies_or_refuses _
tests\test_ablation_module.py:253: in test_every_standard_ablation_either_applies_or_refuses
    assert info["ablation_applied"] == expected, name
E   KeyError: 'ablation_applied'
_ TestRefinementStateIsRestored.test_num_refine_steps_restored_when_the_forward_raises _
tests\test_ablation_module.py:271: in test_num_refine_steps_restored_when_the_forward_raises
    assert model.predictor.num_refine_steps == 3
E   assert 0 == 3
E    +  where 0 = _StubPredictor().num_refine_steps
_______ TestCrashesAreVisible.test_run_all_marks_the_crashed_cell_failed _______
tests\test_ablation_module.py:321: in test_run_all_marks_the_crashed_cell_failed
    assert cell["status"] == "failed"
E   KeyError: 'status'
____ TestCrashesAreVisible.test_the_aggregate_says_it_was_incomplete ______
tests\test_ablation_module.py:350: in test_the_aggregate_says_it_was_incomplete
    assert results.is_complete is False
E   AttributeError: 'dict' object has no attribute 'is_complete'
_ TestNoEmaArmIsReal.test_ema_entry_point_is_intercepted_by_the_wrapper ____
tests\test_ablation_module.py:402: in test_ema_entry_point_is_intercepted_by_the_wrapper
    ablated.update_target_encoder(tau)
E   AttributeError: 'AblatedModel' object has no attribute 'update_target_encoder'
_ TestNoEmaArmIsReal.test_no_ema_arm_keeps_target_online_across_training ____
tests\test_ablation_module.py:418: in test_no_ema_arm_keeps_target_online_across_training
    assert _target_gap(model) == 0.0
E   assert 0.9900000095367432 == 0.0
____ TestKnownGaps.test_the_full_cell_is_marked_not_trained ____
tests\test_ablation_module.py:475: in test_the_full_cell_is_marked_not_trained
    assert results["full"]["trained"] is False
E   KeyError: 'trained'
=============================== short test summary info ============================
FAILED ...::TestMissingTargetIsLoud::test_missing_info_key_raises_instead_of_doing_nothing
FAILED ...::TestMissingTargetIsLoud::test_missing_weight_on_the_model_config_raises
FAILED ...::TestMissingTargetIsLoud::test_flag_with_no_mechanism_raises[use_predictor]
FAILED ...::TestMissingTargetIsLoud::test_flag_with_no_mechanism_raises[use_target_centering]
FAILED ...::TestMissingTargetIsLoud::test_error_type_is_public_and_specific
FAILED ...::TestMissingTargetIsLoud::test_allow_missing_targets_is_the_explicit_escape_hatch
FAILED ...::TestMissingTargetIsLoud::test_every_standard_ablation_either_applies_or_refuses
FAILED ...::TestRefinementStateIsRestored::test_num_refine_steps_restored_when_the_forward_raises
FAILED ...::TestCrashesAreVisible::test_run_all_marks_the_crashed_cell_failed
FAILED ...::TestCrashesAreVisible::test_the_other_cells_survive_a_crash
FAILED ...::TestCrashesAreVisible::test_the_aggregate_says_it_was_incomplete
FAILED ...::TestCrashesAreVisible::test_a_clean_run_is_complete
FAILED ...::TestCrashesAreVisible::test_run_scaling_marks_the_crashed_cell_failed
FAILED ...::TestNoEmaArmIsReal::test_ema_entry_point_is_intercepted_by_the_wrapper
FAILED ...::TestNoEmaArmIsReal::test_no_ema_arm_keeps_target_online_across_training
FAILED ...::TestNoEmaArmIsReal::test_forward_re_syncs_the_target_for_the_ablated_arm
FAILED ...::TestKnownGaps::test_the_full_cell_is_marked_not_trained
=================== 17 failed, 5 passed, 1 warning in 6.89s ===================
```

The 5 that pass pre-fix are deliberate controls: success-path restore,
`a_failed_cell_has_no_loss_to_plot` (true even under the old shape), and the
two determinism tests + `a_run_does_not_mutate_the_base_model`.

### 2. GREEN — after the fix

```
$ & $PY tools\rt.py tests\test_ablation_module.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 22 items

tests\test_ablation_module.py ......................                     [100%]

============================= 22 passed in 7.29s =============================
```

```
$ & $PY tools\rt.py tests\test_interp.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 94 items

tests\test_interp.py ................................................... [ 54%]
...........................................                              [100%]

============================= 94 passed in 9.74s =============================
ELAPSED_TEST_INTERP: 12s
```

Baseline before my change was `94 passed in 8.96s`; it is still 94 tests,
~10 s — well inside the ~15 s budget.

```
$ & $PY tools\rt.py tests\test_ablation_module.py tests\test_interp.py
collected 116 items
tests\test_ablation_module.py ......................                     [ 18%]
tests\test_interp.py ................................................... [ 62%]
...........................................                              [100%]
============================= 116 passed in 7.56s =============================
```

The pre-existing ablation/EMA tests in the slow file also still pass:

```
$ & $PY tools\rt.py tests\test_model.py --slow -k "ablation or ema or target_encoder"
collected 151 items / 137 deselected / 14 selected
tests\test_model.py ..............                                       [100%]
===================== 14 passed, 137 deselected in 3.64s =====================
```

Plus the neighbouring interp suites (they import `src.interp.__init__`):
`test_index_and_cka.py` + `test_info_and_disentangle.py` → 101 passed;
`test_mechanism_wiring.py` + `test_sterility.py` → 38 passed.

### 3. lint / format

```
$ & $PY -m ruff check . --output-format concise
All checks passed!
RUFF_EXIT=0
$ & $PY -m black --check .
All done! ✨ 🍰 ✨
101 files would be left unchanged.
BLACK_EXIT=0
```

---

## no-ema arm — what I did, and whether a config rename is needed

**I made the arm real. No rename is needed; no config filename references it.**

`ABLATION_CONFIGS["no_ema"]` / `describe() == "no EMA target"` now means
"the target encoder IS the online encoder, at every step":

1. `AblatedModel.update_target_encoder(tau)` is a new method. The training
   loop's EMA entry point is `model.update_target_encoder(tau)`
   (`src/train.py:679`, via `do_ema_update`), and it is called on whatever
   object the ablation handed it. The wrapper now intercepts it: when the
   arm is active it re-syncs instead of blending; otherwise it delegates to
   the base model. Before this the wrapper had no such method, so
   `do_ema_update(ablated, ...)` would have raised `AttributeError` — the
   arm could not have worked through the real training path even if the
   caller had remembered to call `skip_ema_update()`.
2. `forward()` re-syncs the target before every forward, so the arm holds
   even for a `train_fn` that never calls the EMA entry point (the two
   mechanisms are independent on purpose; both are mutation-tested).

`ablate_ema()` and `skip_ema_update()` are unchanged and still work for the
`scripts/run_experiment.sh` demo, which calls them by hand.

Cost: one parameter copy per forward, only for this arm.

---

## diff-stat

```
$ git diff --stat campaign/integration...HEAD
 src/interp/ablation.py        | 295 ++++++++++++++++++++++----
 tests/test_ablation_module.py | 473 ++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 724 insertions(+), 44 deletions(-)
```

---

## mutation-verdict — one line per defect, all four enforced

Each fix was reverted individually and the suite re-run; every one has a
detector.

1. **D1 silent no-op** — mutant: `raise AblationTargetMissingError` → `pass`.
   Detected by 4 tests in `TestMissingTargetIsLoud`:
   `test_missing_info_key_raises_instead_of_doing_nothing`,
   `test_missing_weight_on_the_model_config_raises`,
   `test_flag_with_no_mechanism_raises[use_predictor]`,
   `test_flag_with_no_mechanism_raises[use_target_centering]`
   → `4 failed, 3 passed`.
2. **D2 no try/finally** — mutant: restore moved back after the call.
   Detected by `test_num_refine_steps_restored_when_the_forward_raises`
   (`assert 0 == 3`) and `test_a_crash_does_not_corrupt_a_later_forward`
   (`assert [0, 0] == [0, 3]`, i.e. a crash leaks into a later, unrelated
   arm) → `2 failed, 1 passed`.
3. **D3 swallowed exceptions** — mutant: failed cell reverted to
   `{"ablation": ..., "error": str(e)}`.
   Detected by `test_run_all_marks_the_crashed_cell_failed`,
   `test_run_scaling_marks_the_crashed_cell_failed`,
   `test_the_aggregate_says_it_was_incomplete` → `3 failed, 3 passed`.
   (I hardened `AblationResults.failures` to also treat any cell carrying
   an `error` key as failed, so even the old shape cannot read as complete.)
4. **D4 no-EMA arm** — two mutants, one per half of the fix.
   Removing the `update_target_encoder` re-sync →
   `test_ema_entry_point_is_intercepted_by_the_wrapper` (`1.0 != 0.0`) and
   `test_no_ema_arm_keeps_target_online_across_training`
   (`0.0100135 != 0.0`, a real `TextSpanJEPA`, 2 steps) → `2 failed`.
   Removing the forward-time re-sync →
   `test_forward_re_syncs_the_target_for_the_ablated_arm`
   (`0.99 != 0.0`) → `1 failed`.

No defect is left without a detector. Determinism is covered by
`test_two_identical_runs_agree` (two `run_single` runs of a seeded
`train_fn` produce bit-identical `loss_history`) and
`test_a_run_does_not_mutate_the_base_model`.

---

## reported-not-fixed

1. **`full` reports a loss of zero.** `AblationStudy.run_all` special-cases
   `"full"` (pre-fix L318-325; now the `results[name] = {...}` block around
   L513-524) and never trains it: the cell is
   `{"ablation": "full", "description": "full model", "final_loss": 0,
   "loss_history": []}`. Its own comment says "just extract
   representations, no training needed", but no representations are in the
   dict — so in every comparison table the full model shows `final_loss == 0`
   and `run_scaling_ablations` (which *does* train `full`) disagrees with it.
   Left as instructed. I added two **additive** markers so the lie is at
   least labelled: `status="not_trained"` and `trained=False`
   (pinned by `TestKnownGaps::test_the_full_cell_is_marked_not_trained`).
   `final_loss: 0` itself is unchanged. A real fix is a human decision:
   either train `full` like every other arm, or drop the key and let the
   table show "not measured".
2. **`AblationStudy`'s docstring promises work `run_single` does not do.**
   The class docstring (pre-fix L252-255) says "1. Train the ablated model
   for N steps 2. Extract representations 3. Compute all metrics
   4. Compare to full model". `run_single` does (1) only: it returns
   `{"ablation", "description", "final_loss", "loss_history"}` (pre-fix
   L295-300) — no representations, no metrics, no CKA. Those live in a
   separate, uncalled method, `compare_representations`, which expects a
   `full_model_reps` tensor that `run_all` never produces. So step 4 of the
   documented workflow is unreachable from `run_all`/`run_scaling_ablations`.
   Docstring left untouched per the card.

---

## Extra finding inside D1 (fixed, flagging loudly)

`AblationConfig` has two more flags that **no code path has ever acted on**:
`use_predictor` and `use_target_centering`. `no_predictor` is a headline
ablation in this module's own header (L16) and in
`run_scaling_ablations`' default list, yet `AblatedModel.forward` never
touched either flag — the "no predictor" run was a full model wearing the
label. This is the same root cause as D1, so I refused it the same way:
`UNIMPLEMENTED_ABLATIONS = ("use_predictor", "use_target_centering")`
makes `forward()` raise when one of them is False. Consequence to be aware
of at integration:

* `ABLATION_CONFIGS["no_predictor"]`, `["no_centering"]` and
  `["predictor_only"]` (which also sets `use_target_centering=False`) now
  produce **visible failed cells** in `run_all` / `run_scaling_ablations`
  instead of full-model rows labelled as ablations. That is the intended
  outcome, but it means those three arms are unusable until someone
  implements them; removing the names from `UNIMPLEMENTED_ABLATIONS` is the
  one-line revert.
* No config file references them (checked `config/**`); `ABLATION_CONFIGS`
  keys are unchanged, so no yaml rename.

A third no-op I deliberately did **not** make fatal: a term whose weight is
0 (`lambda_decoder: 0` with `use_decoder=False`) *is* refused, because
subtracting zero removes nothing — that is the same lie. It only trips for
user configs that already weight a term at zero.

---

## не_сделано

- `final_loss: 0` for `full` and the `AblationStudy` docstring gap are
  reported, not fixed (per card).
- `use_predictor` / `use_target_centering` are refused, **not implemented**.
  Implementing a real "no predictor" arm means bypassing
  `TextSpanJEPA.compute_loss_with_targets`, which is `src/models/jepa.py` —
  not my file, and a much larger change than this card.
- `src/interp/__init__.py` still exports only the four original names, so
  `AblationTargetMissingError`, `AblationResults`, `LOSS_TERM_ABLATIONS`
  and `UNIMPLEMENTED_ABLATIONS` must be imported from
  `src.interp.ablation` directly. That file belongs to another card.
- No full-suite run (CPU rules; `tools/rt.py` refuses it anyway) and no
  training run of any kind.

## риски

- **Behaviour change for callers of `run_all` / `run_scaling_ablations` with
  the default ablation list**: `no_predictor`, `no_centering` and
  `predictor_only` now fail loudly instead of silently passing. Anyone
  reading `results` with `if "error" in r` still works; anyone who assumed
  every key was trainable will now hit `status == "failed"`.
- **Ablated cells changed shape**: they now also carry `status`, `trained`,
  `error_type` and `traceback` (the traceback can be long if a study is
  serialised to JSON). The `full` cell carries `status`/`trained`. Success
  cells gain `status`/`trained`. No key was removed, so
  `scripts/run_experiment.sh` (which does `if 'error' in r: ... else:
  r['final_loss']`) still works unchanged.
- **Performance**: `AblatedModel.update_target_encoder` adds one Python
  attribute lookup per training step for non-ablated arms; the forward-time
  re-sync costs one parameter copy per forward, only when `no_ema` is
  active. `AblationResults` is a `dict` subclass with three properties —
  `failures` rebuilds a dict, so it is O(n) per call and is not cached.
  Do not call `.failures` in a hot loop.
- `warnings.warn(..., stacklevel=3)` in `_resolve_loss_terms` assumes the
  call depth (forward → _resolve_loss_terms → warn). If someone refactors
  the call chain the warning will point one frame off; the exception path
  (the one that matters) is unaffected.
- A single-threaded CPU budget was respected throughout: the largest run
  was 22 tests in 7.3 s, and `tests/test_interp.py` is unchanged at ~10 s.
