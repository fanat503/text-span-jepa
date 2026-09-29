# TASK-08 — `target_centering.center` was still mutated under `eval()`

- **status:** done (fix accepted, corrected, extended with tests)
- **files:** `src/models/jepa.py` (only file I own), **new**
  `tests/test_target_centering_state.py`. Touched nothing else. No training run.

## verdict on the pre-existing (dead-agent) diff: **CORRECT in substance, INCOMPLETE as landed**

The worktree arrived with an uncommitted, untested diff to `src/models/jepa.py`
made `TextSpanJEPA` inherit `TrainingStateGuard` and replaced
`h_target = self.target_centering(h_target)` with a `_mutate_state`-routed
`update_center` call plus an explicit subtraction. I verified it rather than
trusting it. It is the right fix, but it would have landed `ruff check .` red,
and one clause of its comment had become false. Both fixed (below).

### Is "arithmetically identical in training mode" true? Yes — bit-exactly.

`TargetCentering.forward` is `update_center(x); return x - self.center`
(`src/models/collapse.py:93-95`), and `update_center` is
`center = m*center + (1-m)*x.mean(dim=(0,1), keepdim=True)` under `@torch.no_grad()`.
The diff calls the same two expressions, in the same order, with the same
tensors; `_mutate_state` forwards verbatim when `self.training` is True. So it
must be identical — and it is, measured rather than argued: a temporary harness
(the file is deleted; it is not in the commit) ran a deterministic 6-step
training loop — loss, loss bits, `target_centering.center` bytes, concatenated
gradient bytes, gradient L2, and the `target_center_norm` diagnostic — on the
pre-fix tree and on this tree, and the two JSON fingerprints hash the same:

```
prefix : C270CDD67BD32F5AB1B8760FE2EBD5A7895F48AD6BC4B1B86CD54AB4FD8C62F6
final  : C270CDD67BD32F5AB1B8760FE2EBD5A7895F48AD6BC4B1B86CD54AB4FD8C62F6
identical: True
```

Every step's gradient hash is inside that comparison, so **no gradient path
changed**, and the per-step loss hashes pin the **layer_norm ordering** as
unchanged (a reorder changes the first hash).

### Two defects in the diff as it landed, both fixed

1. **`ruff check .` was red.** Once `TextSpanJEPA` stopped inheriting `nn.Module`
   directly, `from torch import nn` in `jepa.py` had no remaining use (the only
   `nn.` references left are inside comments) — `F401`. The dead agent left the
   repo failing the project's own lint gate. Removed the import; verified no
   module imports `nn` from `jepa` (`git grep "jepa.nn\|jepa import nn"` → no
   hits).
2. **The comment described a call that no longer exists.** It said
   "`TargetCentering.forward` ... and it is called here". After the change the
   model no longer calls `forward` at all. Reworded to describe the post-change
   reality (the model reaches `TargetCentering` only here, and routes the write
   itself). Substance kept, 12 lines → 10.

### Why this shape and not another

`src/models/collapse.py` is **not** in `files_allowed`, so making
`TargetCentering` inherit `TrainingStateGuard` (what every other mechanism does)
is not available to me. `jepa.py` is the only caller
(`git grep target_centering` — one call site), and
`src/models/_state_guard.py`'s rule is explicit: route every state write through
`self._mutate_state` so `grep -n "_mutate_state" src/models/*.py` lists all of
them. Hence: guard on the write, keep the use. An inline `if self.training:`
would satisfy the behaviour but break the greppable rule; guarding inside
`collapse.py` would be the right end state but is out of bounds. This is the
correct move *within the card's boundary*, and the boundary should be widened by
a follow-up (see не_сделано).

## what the fix does

```python
self._mutate_state(self.target_centering.update_center, h_target)
h_target = h_target - self.target_centering.center
h_target = F.layer_norm(h_target, (h_target.size(-1),))   # unchanged
```

Under `eval()` the EMA write is skipped and the subtraction still happens with
the frozen training-time center — the guard is on the **write**, not on the
**use**, which is why eval loss stays the same *quantity* it was.

## verify

### 1. card verify — `tools/rt.py tests/test_training_state_guards.py`

```
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 59 items

tests\test_training_state_guards.py .................................... [ 61%]
.................x.....                                                  [100%]

======================== 58 passed, 1 xfailed in 0.52s ========================
```

The one `x` is TASK-09's pre-existing `xfail` on `wsr._stiefel_retract` — not
mine, not touched.

### 2. new regression file — `tools/rt.py tests/test_target_centering_state.py`

```
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 10 items

tests/test_target_centering_state.py::TestEvalDoesNotMoveTheTargetCenter::test_eval_leaves_the_center_bit_identical PASSED [ 10%]
tests/test_target_centering_state.py::TestEvalDoesNotMoveTheTargetCenter::test_repeated_eval_is_a_pure_function PASSED [ 20%]
tests/test_target_centering_state.py::TestEvalDoesNotMoveTheTargetCenter::test_no_buffer_moves_under_eval PASSED [ 30%]
tests/test_target_centering_state.py::TestEvalDoesNotMoveTheTargetCenter::test_validation_pass_does_not_perturb_the_training_trajectory PASSED [ 40%]
tests/test_target_centering_state.py::TestTrainingStillMovesTheTargetCenter::test_training_updates_the_center PASSED [ 50%]
tests/test_target_centering_state.py::TestTrainingStillMovesTheTargetCenter::test_training_keeps_the_data2vec_ema_arithmetic PASSED [ 60%]
tests/test_target_centering_state.py::TestTrainingStillMovesTheTargetCenter::test_layer_norm_still_comes_after_the_centering PASSED [ 70%]
tests/test_target_centering_state.py::TestTrainingStillMovesTheTargetCenter::test_eval_centers_with_the_frozen_center PASSED [ 80%]
tests/test_target_centering_state.py::TestTheWriteIsRoutedThroughTheSharedGuard::test_model_inherits_the_shared_guard PASSED [ 90%]
tests/test_target_centering_state.py::TestTheWriteIsRoutedThroughTheSharedGuard::test_jepa_routes_state_writes_through_mutate_state PASSED [100%]

============================= 10 passed in 0.50s =============================
```

No skip, no xfail, no `--no-verify`; `strict=True`-style property assertions,
no pinned magic numbers. Ten tests because the card's own failure had two faces:
the buffer must freeze (bug side) **and** training must keep moving (over-
correction side, which is how a guard silently freezes training instead).

### 3. measured before/after — the centre is now the *only* thing that moved, and now moves not at all

Pre-fix, one `eval()+no_grad()` loss call moved the centre by up to **0.0608**
(max-abs) and made two identical eval calls disagree
(1.8167136907577515 vs 1.8165409564971924) — see the mutation run below. Card's
figure was 0.071; same order, my fixture is smaller.

Post-fix, on the **default** config (`use_jawp=True`) an `eval()+no_grad()` loss
call now moves **no buffer at all** — measured over the model's 4 registered
buffers:

```json
{ "n_buffers": 4, "moved": [] }
```

### 4. lint / format — clean

```
> ruff check src/models/jepa.py tests/test_target_centering_state.py
All checks passed!

> black --check src/models/jepa.py tests/test_target_centering_state.py
All done! ✨ 🍰 ✨
2 files would be left unchanged.
```

## diff-stat

```
 src/models/jepa.py                   |  25 ++-
 tests/test_target_centering_state.py | 311 +++++++++++++++++++++++++++++++++++
 2 files changed, 332 insertions(+), 4 deletions(-)
```

Commit on `agent/task-08`, nothing pushed.

## mutation-verdict

Reverted `src/models/jepa.py` to HEAD (`git checkout -- src/models/jepa.py`,
i.e. the exact pre-fix tree, both hunks: no inheritance, unguarded
`self.target_centering(h_target)`) and re-ran my file.

**7 of 10 go red. The 3 that stay green are exactly the training-mode ones —
which is the point: the mutation must not disturb them.**

```
tests\test_target_centering_state.py FFFF...FFF                          [100%]

_ TestEvalDoesNotMoveTheTargetCenter.test_eval_leaves_the_center_bit_identical _
E   AssertionError: eval() moved target_centering.center (max delta 0.0607622)
_ TestEvalDoesNotMoveTheTargetCenter.test_repeated_eval_is_a_pure_function _
E   AssertionError: two identical eval() calls disagreed: 1.8167136907577515 vs 1.8165409564971924
_ TestEvalDoesNotMoveTheTargetCenter.test_no_buffer_moves_under_eval _
E   AssertionError: assert not ['target_centering.center']
_ TestEvalDoesNotMoveTheTargetCenter.test_validation_pass_does_not_perturb_the_training_trajectory _
E   AssertionError: losses diverged: [1.1617900133132935, 1.1622059345245361, 1.162619709968567] vs [1.1622059345245361, 1.162619709968567, 1.1630195379257202]
_ TestTrainingStillMovesTheTargetCenter.test_eval_centers_with_the_frozen_center _
E   AssertionError: eval() did not center with the frozen center
_ TestTheWriteIsRoutedThroughTheSharedGuard.test_model_inherits_the_shared_guard _
E   AssertionError: assert False
E    +  where False =   isinstance(TextSpanJEPA(...), <class 'src.models._state_guard.TrainingStateGuard'>)
_ TestTheWriteIsRoutedThroughTheSharedGuard.test_jepa_routes_state_writes_through_mutate_state _
E   AssertionError: jepa.py writes state without the shared guard

=========================== 7 failed, 3 passed in 0.61s ===========================
```

Green under the mutation (correctly — the training path is untouched by it):
`test_training_updates_the_center`, `test_training_keeps_the_data2vec_ema_arithmetic`,
`test_layer_norm_still_comes_after_the_centering`.

Restored with `git apply` of the saved patch, re-ran both files: the card file
58 passed / 1 xfailed, mine 10 passed (§1, §2 above).

## не_сделано

- **`tests/test_model.py --slow` (the card's second verify command) was NOT
  run.** My orchestrator restricted me to
  `tools\rt.py tests\test_training_state_guards.py` because the box's owner is
  gaming, and `rt.py` lists `tests/test_model.py` under `SLOW_FILES` for
  exactly the reason. This is a deliberate, disclosed gap in the card's verify.
  The three tests in that file that touch this code path, and my reading of
  each (**reasoned, not run**):
  - `test_centering` (`tests/test_model.py:284`) builds a bare
    `TargetCentering` and calls it directly. `collapse.py` is untouched → passes.
  - `test_checkpoint_saves_centering_state` (`:1746`) calls
    `compute_loss_with_targets` in default train mode, then round-trips the
    centre through `save_checkpoint`/`load_checkpoint`. The centre still updates
    in train mode (asserted bit-exactly by my own test) → passes.
  - `:627` asserts the diagnostics dict contains `"target_center_norm"`; the key
    is still computed, from the now-frozen centre under `eval()` → passes.
  Whoever runs the gate should still run it. It costs tens of seconds.
- **`TargetCentering` itself is still unguarded.** `collapse.py` is outside this
  card. The correct end state is that `TargetCentering(TrainingStateGuard)`
  guards its own `update_center`, which would also protect any future caller.
  Today the guarantee lives in one caller in one file, protected only by a grep
  test (`test_jepa_routes_state_writes_through_mutate_state`) and by the fact
  that there is exactly one call site. **Recommend a follow-up card with
  `src/models/collapse.py` allowed.**
- **No trainer-side change.** `_snapshot_training_buffers` in `src/train.py` is
  left in place. That is right — it is the backstop, not the fix — but note the
  two now overlap: the fix makes the snapshot redundant *for this buffer*.
  I did not touch `src/train.py` to say so.
- **Not run:** `ruff check .` / `black --check .` over the whole repo (I ran them
  on my two files only — the other 100 files are not mine and may be mid-edit by
  other agents); no training of any kind; no other test file.

## риски

- **`jawp.active_k` is still filled under `eval()`** (`jawp.py:326`, inside
  `compute_loss`, which the model calls on every loss pass). That is a
  letter-of-the-invariant violation of the same class, and it is the one thing
  that makes "the last of 24 buffers" not quite true as stated. It measured
  **0 delta** post-fix (see §3) because `_validate` passes the *same*
  `current_step` the next training step will use, so `fill_(current_k(step))`
  is idempotent, and because it feeds only the workspace width, not an EMA. It
  is therefore not a trajectory hazard today — but it is unguarded, unmasked at
  source, and it would become one if a caller ever validated at a different
  step. `jawp.py` is not mine; **suggest a follow-up card.**
- **`_prev_target_h` is a plain attribute, not a buffer**, and
  `compute_loss_with_targets` overwrites it unconditionally at the end of every
  pass (`jepa.py:914`), eval included. It feeds
  `CollapseDiagnostics.compute(prev_target_h=...)` only — no loss, no gradient —
  so it cannot change trained weights, and my trajectory test (losses +
  all buffers) confirms that. It *can* change a logged drift diagnostic to depend
  on validation history, and `_snapshot_training_buffers` does not restore it.
  Out of scope for this card (the card is about the 24 *buffers*); flagged
  because "eval() is a pure function of its inputs" is not literally true of the
  model object today.
- **Behaviour change on the eval path, intended.** Anyone who called the model
  under `eval()` and relied on the target encoder tracking the running mean now
  gets a frozen mean. That is the documented purpose of the buffer (data2vec
  trains the teacher EMA under training) and matches every other guarded
  mechanism, but it is a real semantic change for inference/eval code paths
  that never trained (freshly-constructed models have `center = 0`, which was
  already true before — `update_center` on the first call with m=0.9 leaves it
  at 0.1·mean, so a one-off eval used to nudge it and now does not).
- **Bit-exactness proofs cover a tiny config** (dim 32, depth 2, seq 16, no
  dropout) and one deterministic seed. That is the strongest cheap check; it is
  not a proof for `large_300m`, though the code path taken is identical.
- The identity claim rests on `TargetCentering.forward` being exactly
  `update_center(x); x - self.center`. If someone later changes `forward` to
  apply layer_norm internally or update on a different statistic, the two copies
  will silently diverge — there is no test that the two agree, because there is
  no second copy to compare against once `forward` is unused. A cheap follow-up:
  delete `TargetCentering.forward` once no caller remains (see не_сделано).
