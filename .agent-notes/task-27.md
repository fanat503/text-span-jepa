# TASK-27 — tests for `src/interp/run_comparison.py` (wave-1 TASK-07 shipped without them)

status: **done**

branch: `agent/task-27`, commit `48a00cb` (from `agent/wave-1` @ `b91b273`), nothing pushed.

## files

| file | change |
|---|---|
| `tests/test_run_comparison.py` | **new**, 438 lines, 16 tests |
| `src/interp/run_comparison.py` | **not modified in the final commit** — it was reverted temporarily for the mutation runs and restored with `git checkout HEAD --`. `git diff HEAD` is empty for it. |

No other file touched. No test in the existing suite was edited, weakened or deleted.

## what the file pins

The five must-pins, mapped to tests:

| must-pin | test | what it does |
|---|---|---|
| (1) real `save_checkpoint` -> `load_model` round trip, max-abs-difference exactly 0.0 | `TestJepaRoundTrip::test_save_load_round_trip_is_bit_exact` | payload written by the real `src.train.save_checkpoint`, read by the real `src.interp.run_comparison.load_model`; asserts key-set equality then `max \|saved - loaded\| == 0.0` across all 126 tensors |
| (2) `jawp` / `target_centering` / `sigreg` present after the trip | `TestJepaRoundTrip::test_tensors_outside_the_four_legacy_submodules_survive`, `::test_load_covers_more_than_the_legacy_submodules` | five named tensors (`jawp.workspace_Q`, `jawp.active_k`, `target_centering.center`, `sigreg.sketch_directions`, `sigreg.t_points`) asserted bitwise back; the count gate is a ratio of the fixture's own keys, so a mechanism that grows a tensor later is covered without editing the file |
| (3) `regression_head.0.weight` restored bitwise | `TestBaselineBranches::test_data2vec_regression_head_is_restored_bitwise` | also asserts the restored head is **not** equal to a freshly initialised one, so a random head fails it |
| (4) legacy-shaped checkpoint -> clear format error | `TestLegacyCheckpointRefused` (6 tests) | `CheckpointLoadError`, not `KeyError`; message must name both the wanted key (`model`) and what was found (`encoder`, `model_name`); all three branches refuse; non-dict payload refused |
| (5) `strict=True` genuinely strict | `TestStrictIsStrict` (4 tests) | a deleted tensor, an extra tensor and a mis-shaped tensor each raise `RuntimeError` naming the key, with `Missing key` / `Unexpected key` asserted |

Plus `TestBaselineBranches::test_data2vec_round_trip_is_bit_exact` and
`::test_mlm_branch_restores_its_head` (whole-state 0.0 for the two baselines) and
`TestStrictIsStrict::test_loaded_model_is_in_eval_mode`.

**Prohibition honoured.** I did **not** re-emit the legacy per-module keys from the
writer, and I did not add a compatibility branch to the reader. The file only asserts
that a checkpoint *without* `model` is refused.

Two design points worth the reviewer's attention:

* **Toy width.** `load_model` hardcodes production widths — the JEPA default config is
  262,021,633 parameters and ~1 GB of `torch.save`, and the baselines are built at
  `embed_dim=768, depth=12`. That is unaffordable on this box. The `toy_width` fixture
  therefore monkeypatches **only the three constructors** that `load_model` reaches
  through its function-local imports (`TextSpanJEPAConfig`, `MLMBaseline`,
  `Data2VecTextBaseline`) down to `embed_dim=32, depth=1`. The format check, the `state`
  dict, `load_state_dict(state, strict=True)` and the branch structure are the real ones.
  Every property asserted is width-independent: completeness of the restore, not tensor
  count.
* **Scramble.** Before saving, every tensor is overwritten with `ramp + 1000*(index+1)`,
  so a tensor that fails to be restored is distinguishable from an init value *and*
  cannot alias a neighbouring key's value. This is what makes mutation variant C (below)
  detectable at all.

## verify

```
PS C:\dev\wt-27> $PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
PS C:\dev\wt-27> & $PY tools\rt.py tests/test_run_comparison.py -v
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_run_comparison.py -v
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collecting ... collected 16 items

tests/test_run_comparison.py::TestJepaRoundTrip::test_save_load_round_trip_is_bit_exact PASSED [  6%]
tests/test_run_comparison.py::TestJepaRoundTrip::test_tensors_outside_the_four_legacy_submodules_survive PASSED [ 12%]
tests/test_run_comparison.py::TestJepaRoundTrip::test_load_covers_more_than_the_legacy_submodules PASSED [ 18%]
tests/test_run_comparison.py::TestBaselineBranches::test_data2vec_regression_head_is_restored_bitwise PASSED [ 25%]
tests/test_run_comparison.py::TestBaselineBranches::test_data2vec_round_trip_is_bit_exact PASSED [ 31%]
tests/test_run_comparison.py::TestBaselineBranches::test_mlm_branch_restores_its_head PASSED [ 37%]
tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_legacy_shaped_checkpoint_raises_a_named_format_error PASSED [ 43%]
tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[jepa] PASSED [ 50%]
tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[mlm] PASSED [ 56%]
tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[data2vec] PASSED [ 62%]
tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_non_dict_payload_is_refused PASSED [ 68%]
tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_unknown_model_type_still_raises_value_error PASSED [ 75%]
tests/test_run_comparison.py::TestStrictIsStrict::test_a_missing_tensor_is_rejected_by_name PASSED [ 81%]
tests/test_run_comparison.py::TestStrictIsStrict::test_an_unexpected_tensor_is_rejected_by_name PASSED [ 87%]
tests/test_run_comparison.py::TestStrictIsStrict::test_shape_mismatch_is_rejected PASSED [ 93%]
tests/test_run_comparison.py::TestStrictIsStrict::test_loaded_model_is_in_eval_mode PASSED [100%]

============================= 16 passed in 4.19s =============================
```

Also `& $PY -m ruff check tests/test_run_comparison.py` -> `All checks passed!` and
`& $PY -m black --check tests/test_run_comparison.py` -> `1 file would be left unchanged.`
Only `tools/rt.py` was used to run tests. No training, no whole-suite run, threads=1.

### the 0.0, measured directly

A throwaway probe (deleted; not committed) imported the test helpers and printed the
measured delta rather than trusting the assertion:

```
JEPA     : 126 tensors (59 reachable by the 4 legacy submodule reads, 67 not)  max|saved - loaded| = 0.0
data2vec : 34 tensors  max|saved - loaded| = 0.0
           regression_head.0.weight bitwise equal: True
```

Note the **59 of 126** figure: the four pre-fix submodule reads reached 47% of the
checkpoint on this fixture. The card said "85 of 90" from TASK-07's own (different,
smaller) fixture; on a fixture with all 16 mechanisms on the gap is larger, and it is the
same defect.

## mutation-verdict

The card asked for one mutation; I ran two, because the first one cannot distinguish the
property the file is really about. **Both runs are pasted in full below.**

### variant A — the literal revert (`git show 626e9bf^:src/interp/run_comparison.py`)

The four per-module `.get()` reads, exactly as the card specified.

```
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_run_comparison.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 16 items

tests\test_run_comparison.py FFFFF.FFFFF.FFFF                            [100%]

=================================== FAILURES ===================================
__________ TestJepaRoundTrip.test_save_load_round_trip_is_bit_exact ___________
tests\test_run_comparison.py:185: in test_save_load_round_trip_is_bit_exact
    loaded = load_model(path, "jepa", "cpu")
src\interp\run_comparison.py:37: in load_model
    model.encoder.load_state_dict(ckpt.get("encoder", {}))
C:\Users\Илья\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py:2638: in load_state_dict
    raise RuntimeError(
E   RuntimeError: Error(s) in loading state_dict for TextSpanJEPLEncoder:
E   	Missing key(s) in state_dict: "pos_embedding", "token_embedding.weight", "blocks.0.norm1.weight", ... "norm.bias".
__ TestJepaRoundTrip.test_tensors_outside_the_four_legacy_submodules_survive __
... (same RuntimeError from run_comparison.py:37)
___ TestJepaRoundTrip.test_load_covers_more_than_the_legacy_submodules ___
... (same RuntimeError from run_comparison.py:37)
_ TestBaselineBranches.test_data2vec_regression_head_is_restored_bitwise ____
src\interp\run_comparison.py:64: in load_model
    model.encoder.load_state_dict(ckpt.get("encoder", {}))
E   RuntimeError: Error(s) in loading state_dict for TextSpanJEPLEncoder:
E   	Missing key(s) in state_dict: ...
____ TestBaselineBranches.test_data2vec_round_trip_is_bit_exact ______________
... (same RuntimeError from run_comparison.py:64)
_ TestLegacyCheckpointRefused.test_legacy_shaped_checkpoint_raises_a_named_format_error _
tests\test_run_comparison.py:317: in test_legacy_shaped_checkpoint_raises_a_named_format_error
    with pytest.raises(CheckpointLoadError) as excinfo:
E   Failed: DID NOT RAISE CheckpointLoadError
_ TestLegacyCheckpointRefused.test_every_branch_refuses_a_legacy_checkpoint[jepa] _
E   Failed: DID NOT RAISE CheckpointLoadError
_ TestLegacyCheckpointRefused.test_every_branch_refuses_a_legacy_checkpoint[mlm] _
src\interp\run_comparison.py:52: in load_model
    model.load_state_dict(ckpt.get("model", {}))
E   RuntimeError: Error(s) in loading state_dict for MLMBaseline:
E   	Missing key(s) in state_dict: "encoder.pos_embedding", ..., "mlm_head.weight", "decoder.weight".
_ TestLegacyCheckpointRefused.test_every_branch_refuses_a_legacy_checkpoint[data2vec] _
src\interp\run_comparison.py:64: in load_model
E   RuntimeError: Error(s) in loading state_dict for TextSpanJEPLEncoder:
E   	size mismatch for blocks.0.mlp.fc1.weight: copying a param with shape torch.Size([64, 32]) from checkpoint, the shape in current model is torch.Size([128, 32]).
_ TestLegacyCheckpointRefused.test_non_dict_payload_is_refused ______________
tests\test_run_comparison.py:348: in test_non_dict_payload_is_refused
    load_model(path, "jepa", "cpu")
src\interp\run_comparison.py:37: in load_model
E   AttributeError: 'list' object has no attribute 'get'
_ TestStrictIsStrict.test_a_missing_tensor_is_rejected_by_name ______________
E   AssertionError: the error does not name the missing tensor 'target_centering.center': 'Error(s) in loading state_dict for TextSpanJEPLEncoder:\n\tMissing key(s) in state_dict: "pos_embedding", ... '
_ TestStrictIsStrict.test_an_unexpected_tensor_is_rejected_by_name ___________
E   AssertionError: the error does not name the unexpected tensor: 'Error(s) in loading state_dict for TextSpanJEPLEncoder: ... '
______ TestStrictIsStrict.test_shape_mismatch_is_rejected __________________
E   AssertionError: the error does not name the mis-shaped tensor: 'Error(s) in loading state_dict for TextSpanJEPLEncoder: ... '
_____ TestStrictIsStrict.test_loaded_model_is_in_eval_mode ___________________
... (same RuntimeError from run_comparison.py:37)
=========================== short test summary info ===========================
FAILED tests/test_run_comparison.py::TestJepaRoundTrip::test_save_load_round_trip_is_bit_exact
FAILED tests/test_run_comparison.py::TestJepaRoundTrip::test_tensors_outside_the_four_legacy_submodules_survive
FAILED tests/test_run_comparison.py::TestJepaRoundTrip::test_load_covers_more_than_the_legacy_submodules
FAILED tests/test_run_comparison.py::TestBaselineBranches::test_data2vec_regression_head_is_restored_bitwise
FAILED tests/test_run_comparison.py::TestBaselineBranches::test_data2vec_round_trip_is_bit_exact
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_legacy_shaped_checkpoint_raises_a_named_format_error
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[jepa]
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[mlm]
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[data2vec]
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_non_dict_payload_is_refused
FAILED tests/test_run_comparison.py::TestStrictIsStrict::test_a_missing_tensor_is_rejected_by_name
FAILED tests/test_run_comparison.py::TestStrictIsStrict::test_an_unexpected_tensor_is_rejected_by_name
FAILED tests/test_run_comparison.py::TestStrictIsStrict::test_shape_mismatch_is_rejected
FAILED tests/test_run_comparison.py::TestStrictIsStrict::test_loaded_model_is_in_eval_mode
======================== 14 failed, 2 passed in 5.96s =========================
```

**14 red, 2 green.** The 2 survivors are `test_mlm_branch_restores_its_head` (the old
mlm branch already read `ckpt["model"]` whole) and
`test_unknown_model_type_still_raises_value_error` (that `ValueError` predates the fix).
Every one of the five must-pins goes red.

### variant C — hand-listed subset + `strict=False` (why this one matters)

Variant A raises on the *first* submodule, so it proves "the format check exists" but it
cannot distinguish "the mechanism tensors came back" from "the load never ran". The
silent-partial-restore mode — which is what the fix commit's own docstring names as the
real defect — needs the load to *succeed*. Variant C restores only keys under
`encoder.`/`target_encoder.`/`predictor.`/`decoder.` with `strict=False`:

```
collected 16 items
tests\test_run_comparison.py FFFFF.FFFFF.FFF.                            [100%]

=================================== FAILURES ===================================
__________ TestJepaRoundTrip.test_save_load_round_trip_is_bit_exact ___________
tests\test_run_comparison.py:185: in test_save_load_round_trip_is_bit_exact
    max_abs_diff = _max_abs_diff(saved, loaded.state_dict())
tests\test_run_comparison.py:159: in _max_abs_diff
    assert torch.equal(before, after), f"{name}: integer tensor differs after the round trip"
E   AssertionError: jawp.active_k: integer tensor differs after the round trip
E   assert False
E    +    where False = <built-in method equal of type object at 0x00007EFDB4CB80>(tensor(64000), tensor(1))
__ TestJepaRoundTrip.test_tensors_outside_the_four_legacy_submodules_survive __
tests\test_run_comparison.py:223: in test_tensors_outside_the_four_legacy_submodules_survive
    assert torch.equal(saved[name], loaded[name]), (
E   AssertionError: jawp.workspace_Q came back different from what was saved (max |delta| = 63095.0)
E   assert False
E    +    where False = <built-in method equal of type object at 0x00007FFDA6FC20>(tensor([[63000., 63001., 63002.], ...]), tensor([[1., 0., 0.],\n        [0., 1., 0.], ...]))
___ TestJepaRoundTrip.test_load_covers_more_than_the_legacy_submodules ___
tests\test_run_comparison.py:249: in test_load_covers_more_than_the_legacy_submodules
    max_abs_diff = _max_abs_diff(saved, loaded)
E   AssertionError: max |saved - loaded| = 125000.0 over 126 tensors; integer tensors also
differ: ['jawp.active_k (saved 64000, got 1)', 'cgn.total_steps (saved 66000, got 0)',
'pcr.level_offsets (saved [70000, 70001, 70002], got [0, 8, 16])', 'spc.adapt_step (saved 90000,
got 0)', 'wsd.is_initialized (saved True, got False)', 'wsd.step_count (saved 94000, got -1)',
'cmc.total_cmc_steps (saved 98000, got 0)', 'gac.total_gac_steps (saved 101000, got 0)',
'sta.is_initialized (saved True, got False)', 'sta.step_count (saved 108000, got 0)',
'puc.total_steps (saved 111000, got 0)', 'rdc.total_steps (saved 119000, got 0)',
'wsr.total_steps (saved 126000, got 0)']
_ TestBaselineBranches.test_data2vec_regression_head_is_restored_bitwise ____
tests\test_run_comparison.py:277: in test_data2vec_regression_head_is_restored_bitwise
    assert torch.equal(saved[name], restored), (
E   AssertionError: regression_head.0.weight was not restored: max |delta| = 34022.88671875
E   assert False
____ TestBaselineBranches.test_data2vec_round_trip_is_bit_exact ______________
tests\test_run_comparison.py:290: in test_data2vec_round_trip_is_bit_exact
    assert _max_abs_diff(saved, loaded.state_dict()) == 0.0
E   AssertionError: assert 34030.93359375 == 0.0
_ TestLegacyCheckpointRefused.test_legacy_shaped_checkpoint_raises_a_named_format_error _
E   Failed: DID NOT RAISE CheckpointLoadError
_ TestLegacyCheckpointRefused.test_every_branch_refuses_a_legacy_checkpoint[jepa] _
E   Failed: DID NOT RAISE CheckpointLoadError
_ TestLegacyCheckpointRefused.test_every_branch_refuses_a_legacy_checkpoint[mlm] _
E   Failed: DID NOT RAISE CheckpointLoadError
_ TestLegacyCheckpointRefused.test_every_branch_refuses_a_legacy_checkpoint[data2vec] _
E   Failed: DID NOT RAISE CheckpointLoadError
________ TestLegacyCheckpointRefused.test_non_dict_payload_is_refused _________
src\interp\run_comparison.py:35: in load_model
    model.load_state_dict(_hand_listed(ckpt), strict=False)
E   AttributeError: 'list' object has no attribute 'get'
_ TestStrictIsStrict.test_a_missing_tensor_is_rejected_by_name ______________
E   Failed: DID NOT RAISE RuntimeError
_ TestStrictIsStrict.test_an_unexpected_tensor_is_rejected_by_name ___________
E   Failed: DID NOT RAISE RuntimeError
______ TestStrictIsStrict.test_shape_mismatch_is_rejected __________________
E   Failed: DID NOT RAISE RuntimeError
=========================== short test summary info ===========================
FAILED tests/test_run_comparison.py::TestJepaRoundTrip::test_save_load_round_trip_is_bit_exact
FAILED tests/test_run_comparison.py::TestJepaRoundTrip::test_tensors_outside_the_four_legacy_submodules_survive
FAILED tests/test_run_comparison.py::TestJepaRoundTrip::test_load_covers_more_than_the_legacy_submodules
FAILED tests/test_run_comparison.py::TestBaselineBranches::test_data2vec_regression_head_is_restored_bitwise
FAILED tests/test_run_comparison.py::TestBaselineBranches::test_data2vec_round_trip_is_bit_exact
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_legacy_shaped_checkpoint_raises_a_named_format_error
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[jepa]
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[mlm]
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_every_branch_refuses_a_legacy_checkpoint[data2vec]
FAILED tests/test_run_comparison.py::TestLegacyCheckpointRefused::test_non_dict_payload_is_refused
FAILED tests/test_run_comparison.py::TestStrictIsStrict::test_a_missing_tensor_is_rejected_by_name
FAILED tests/test_run_comparison.py::TestStrictIsStrict::test_an_unexpected_tensor_is_rejected_by_name
FAILED tests/test_run_comparison.py::TestStrictIsStrict::test_shape_mismatch_is_rejected
======================== 13 failed, 3 passed in 4.53s =========================
```

**13 red, 3 green.** The load *succeeds* under variant C and the tests still fail on the
values, which is the whole point:

* `jawp.workspace_Q` came back as `[[1,0,0],[0,1,0],[0,0,1],[0,...]]` — its `identity`
  init — instead of what was saved. A Grassmann workspace at identity is exactly the
  state the mechanism starts in, so a comparison run would have scored a
  never-trained workspace.
* 13 integer/flag tensors reverted: `wsd.is_initialized` back to `False`,
  `sta.is_initialized` back to `False`, every `*_total_steps` back to `0`,
  `wsd.step_count` back to `-1`. That is the exact `sta.is_initialized == True` while
  `sta.ref_cov` sits at `0.01 * I` pathology `tests/test_checkpoint_fidelity.py` was
  written to catch — reintroduced through the interp loader, which nothing tested.
* `regression_head.0.weight` was never touched: max |delta| 34022.9 against a saved ramp,
  restored value still the random init. Every data2vec-vs-JEPA comparison would have been
  scored through a randomly initialised head, with no error anywhere.
* the 3 strictness tests fail with `DID NOT RAISE` — `strict=False` swallows the missing
  tensor, the unexpected tensor and the shape mismatch that must be loud.

`test_loaded_model_is_in_eval_mode` and `test_unknown_model_type_...` stay green under C
(as they should — C restores completely for the mlm branch and eval/device handling is
untouched); `test_mlm_branch_restores_its_head` stays green because the old mlm branch
already read the whole dict.

### restored

`git checkout HEAD -- src/interp/run_comparison.py` after each run; the green run above is
on the restored file and `git status` is clean apart from this notes file.

## diff-stat

```
 tests/test_run_comparison.py | 438 +++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 438 insertions(+)
```

Commit `48a00cb` on `agent/task-27`.

## не_сделано / риски

* **Nothing beyond the card.** I did not add a test for `run_full_comparison`,
  `extract_representations`, `extract_layer_representations` or
  `generate_text_summary`. They are not in scope (the card is about `load_model`) and
  `run_full_comparison` needs a real trained pair to mean anything.
* **The toy-width fixture is the honest cost of this box.** A reviewer may reasonably
  object that monkeypatching the constructors means the tests never exercise the real
  `TextSpanJEPAConfig()` wiring. Mitigating facts: the properties asserted are
  width-independent, and no test asserts a tensor count as a magic number. But if CI has
  budget, one test at the *real* default width would be the stronger gate and I did not
  write it — 262M params and ~1 GB of `torch.save` is not a shared-CPU test.
* **`sigreg.sketch_directions` is random at construction**, so the SIGReg sketch is
  correctly detected as unrestored under variant C; good. But note that a *saved* SIGReg
  sketch is deterministic given the config, so in production a partial restore there is
  not detectable from the value alone — only from the fact that the module was not
  covered. My test detects it because of the scramble, which is a test artefact. Worth
  knowing: in the wild, this particular tensor would silently revert and look plausible.
* I did not touch `proofs/IMPLEMENTATION_STATUS.md`. No mechanism state moved (this is a
  test-only card and no loss formula changed), so there is no matrix row to update.
* I did not run the full suite — `tools/rt.py` refuses it and the box is in use. The only
  claim I can make is that this file is green and self-contained; whether my 16 tests
  interact badly with another worker's file is unverified.
