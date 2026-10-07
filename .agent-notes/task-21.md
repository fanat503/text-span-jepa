# TASK-21 — halve peak activation memory under gradient accumulation

**status: done, and the card's goal is NOT met. It is not reachable inside
`files_allowed`.** Branch `agent/task-21b`, worktree `C:\dev\wt-21`, base
`main` @ `11bcbd0`. Commit `5197554`. Not pushed.

**The card's premise is wrong, and I can prove it.** `self._gac_z` and
`self._cmc_pass["slots"]` are *not* what pins a micro-batch's activations, so
releasing them cannot halve anything. Measured attribution, 4-micro-batch
gradient accumulation, B=4 T=64 D=128, `lambda_gac=lambda_cmc=0.01` (repo
defaults):

| | peak live-graph |
|---|---|
| A. today's `train.py` shape | **110.21 MB** |
| B. A + `model._gac_z`/`_cmc_pass` also dropped at step end | **110.21 MB** |
| C. no cross-step references at all | **74.62 MB** |
| | |
| **attributable to `model._gac_z` + `model._cmc_pass`** | **+0.00 MB** |
| **attributable to `src/train.py`'s locals** | **+35.59 MB** |

`src/train.py` snapshots both attributes into `_gac_primary` (`:1321`),
`_cmc_primary` (`:1326`) and `_cmc_secondary` (`:1364`), and rebinds them only
*after* the next pass has already built its graph. Those locals are the last
reference; the model attributes never are. Clearing them is a no-op — I shipped
it and measured exactly zero.

---

## 1. Peak memory before / after, and the measurement method

### Method (and a trap I fell into first)

**Do not measure this with `torch.autograd.graph.saved_tensors_hooks`.** My
first four probes used it and reported ~54 MB retained *per micro-batch*,
growing without bound. That was the instrument, not the model. Pure-torch
control, 400 iterations of `y = relu(x@w) @ w.T .sum()` with
`saved_tensors_hooks` installed:

```
  retain_graph=False: rss   188.2 ->   188.2MB (+   0.2MB)   HOOK METER says      0.0MB
  retain_graph=True : rss   188.2 ->  1790.4MB (+1602.1MB)   HOOK METER says    800.0MB
```

and the same 400 iterations with **no hooks at all**:

```
  retain_graph=False: rss 178.7 -> 188.1MB (+ 9.4MB over 400 iters)
  retain_graph=True : rss 188.1 -> 188.1MB (+ 0.0MB over 400 iters)
```

`retain_graph=True` leaks nothing on its own (+0.0 MB), and the hooks
manufactured 1.6 GB. Every "retained MB" number from probes 1–4 was void;
I discarded them and re-measured.

**What I use instead.** A census of live autograd-graph tensors, which is
undistorted because it only *reads*:

```python
gc.collect()                      # required: reference cycles defer collection,
                                  # and without it the numbers are uncollected
                                  # garbage rather than retention
n = b = 0
for o in gc.get_objects():
    if type(o) is torch.Tensor and o.grad_fn is not None:
        n += 1; b += o.numel() * o.element_size()
```

Sampled at three points per micro-batch (after the primary forward, after the
CMC secondary forward, at end of step) and maximised. The harness
(`.agent-notes/_probe_task21s.py`) mirrors `src/train.py:1311-1420` line for
line, **including the named `scaled_loss` local at `:1382`** — an earlier
version of my harness inlined it and under-reported the "today" arm by 16.3 MB,
which is itself the lesson: peak here is set by which references the *caller*
keeps, so a harness that drops one is measuring a different program. All
numbers below reproduced bit-for-bit across 3 consecutive runs.

### Before / after

`after` for my change is `+0.00 MB` on the main path, by the attribution above.
The two rows that show real movement are the empty-batch path (mine) and the
`train.py` patch (not mine):

| config | today | after my change | after the `train.py` patch |
|---|---|---|---|
| `lambda_gac=0.01, lambda_cmc=0.01` (defaults) | 110.21 MB | 110.21 MB (**+0.00**) | 74.62 MB (−35.59, −32%) |
| `lambda_cmc=0` (no CMC) | 71.50 MB | 71.50 MB (**+0.00**) | **35.90 MB (−35.60, −50%)** |
| `lambda_gac=0` (no GAC) | 55.22 MB | 55.22 MB (+0.00) | 55.22 MB (+0.00) |
| both 0 (floor: one micro-batch) | 35.94 MB | — | 35.90 MB |

The `lambda_cmc=0` row is the clean 2× → 1× the card asked for, and it lands
exactly on the floor. It is a **one-line `train.py` change**, not a `jepa.py` one.

### What my change actually buys: the empty-batch path

`compute_loss_with_targets` returns at `if masked_input_ids.size(0) == 0:`
*before* the stash is rewritten, so base code pinned the entire previous pass
across the boundary. Measured, with the model attributes as the only reference:

```
BASE :  after EMPTY batch, stashes are: gac='live-graph-tensor' cmc='dict'
        after empty batch, live-graph census: (124, 19.31MB)
FIXED:  after EMPTY batch, stashes are: gac=None cmc=None
        after empty batch, live-graph census: (0, 0.00MB)
```

Narrow (it needs a zero-row batch — reachable under `DistributedSampler` with
uneven ranks) but real, and it is the only retention `jepa.py` owns.

---

## 2. Gradient correctness — which of the two is used in a backward

**Both. Neither may be detached.** The card's warning was well aimed: a
`.detach()` here saves memory and silently deletes a gradient, raising nothing.

### `_gac_z` — consumed by a real backward

`src/models/gac.py:162,197,200` builds the bonus as
`gamma * warmup * (relu(tau - gn) * (z_flat**2).mean(0) * starved).sum()`. The
`z_flat**2` is **not** detached, so `loss_gac.requires_grad` is True and
`src/train.py:1412` runs `loss_gac.backward()`, which re-enters the predictor and
the online encoder through these slots. That is also why `:1393-1395` passes
`retain_graph=gac_wiring`. The exploration gradient is not optional — it is the
mechanism (`proofs/gac.md`, "No Gradient Dead Zones").

Proven by asserting **gradients move**, not shapes: snapshot every
`p.grad`, run only `loss_gac.backward()`, and require that parameters changed.
Counterfactual asserted too: `model.gac(z_ref.detach(), gn, step)` yields
`requires_grad=False`.

```
MUTATION 2: self._gac_z = span_preds.detach()   ->  5 failed, 5 passed
  FAILED TestStashesStayLive::test_gac_z_is_graph_attached_after_the_forward
  FAILED TestGacExplorationGradient::test_exploration_backward_reaches_the_encoder
  FAILED TestGacExplorationGradient::test_detaching_the_stash_silently_deletes_that_gradient
  FAILED TestEarlyRelease::test_empty_batch_does_not_pin_the_previous_pass
  FAILED TestEarlyRelease::test_release_does_not_cost_the_current_pass_its_stash
```

### `_cmc_pass["slots"]` — consumed by a real backward

`compute_cmc_between_passes` (which *is* in my file, `jepa.py:978-979`)
deliberately takes the primary detached (`slots_det`) and the secondary **live**:
`z2 = _scatter(secondary_pass, live=True)`. `z2` enters `total_loss` at
`train.py:1370`, so the gradient reaches the encoder through the second mask.
Detaching the secondary's slots deletes that path.

```
MUTATION 3: "slots": span_preds.detach()        ->  4 failed, 6 passed
  FAILED TestStashesStayLive::test_cmc_slots_are_graph_attached_after_the_forward
  FAILED TestCmcSecondaryGradient::test_secondary_path_moves_encoder_gradients
  FAILED TestCmcSecondaryGradient::test_detaching_secondary_slots_silently_deletes_that_gradient
  FAILED TestEarlyRelease::test_release_does_not_cost_the_current_pass_its_stash
```

My first version of `test_secondary_path_moves_encoder_gradients` backpropped
`total1 + lambda_cmc * loss_cmc` together, so encoder gradients stayed non-zero
even with the secondary detached and the test **survived mutation 3**. Fixed by
backpropping the CMC term alone; re-run against the same mutant, it went red.

---

## 3. The out-of-scope fix, for whoever owns `src/train.py`

`src/train.py` is forbidden for this card. The patch is at the **end** of the
loop body, after the GAC exploration backward at `:1412-1418` and before the next
iteration's forward at `:1311`:

```python
# train.py, after the `if gac_wiring:` block
_gac_primary = _cmc_primary = _cmc_secondary = None
```

Measured effect (same harness): `lambda_cmc=0` 71.50 → **35.90 MB, −50%, exactly
on the floor**; defaults 110.21 → 74.62 MB (−32%).

I also tested a second candidate — demoting the primary dict's live twin right
after `:1326`, since the bridge only ever reads `primary["slots_det"]`:

```python
if _cmc_primary is not None:
    _cmc_primary = {**_cmc_primary, "slots": _cmc_primary["slots_det"]}
```

**It buys −0.04 MB — nothing.** So the primary's live `slots` entry is not the
cost; the cost is that the primary *graph* must stay alive while the secondary
forward builds, which is structural to CMC's two-pass design and not fixable by
demoting one key.

---

## 4. Verify (full paste)

Card's verify command:

```
$ & $PY tools\rt.py tests\test_activation_release.py tests\test_mechanism_wiring.py tests\test_gac.py tests\test_cmc.py --slow
rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests\test_activation_release.py tests\test_mechanism_wiring.py tests\test_gac.py tests\test_cmc.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 67 items

tests\test_activation_release.py ..........                              [ 14%]
tests\test_mechanism_wiring.py ......                                    [ 23%]
tests\test_gac.py ..................                                     [ 50%]
tests\test_cmc.py .................................                      [100%]

============================== warnings summary ===============================
tests\test_mechanism_wiring.py::TestGACHook::test_stashed_slots_receive_grads
tests\test_mechanism_wiring.py::TestLiveWorkspaceViews::test_regularizers_give_workspace_grads
tests\test_mechanism_wiring.py::TestLiveWorkspaceViews::test_retraction_projects_grad_inplace_and_keeps_orthogonal
  C:\dev\wt-21\src\models\wsr.py:285: UserWarning: WSR mode='gradient': no dL/dQ is available (Q.is_leaf=False, ...)  [pre-existing]

======================= 67 passed, 3 warnings in 0.99s ========================
```

My prompt's verify command, run **before** any edit as a baseline and again
after — identical, 187 passed both times:

```
$ & $PY tools\rt.py tests\test_model.py --slow tests\test_sterility.py
============================= test session starts =============================
collected 187 items

tests\test_model.py ....................................................  [ 27%]
........................................................................  [ 66%]
...............................                                          [ 82%]
tests\test_sterility.py ................................                 [100%]

====================== 187 passed, 4 warnings in 24.74s =======================   (baseline, before edit)
====================== 187 passed, 4 warnings in 22.89s =======================   (after edit)
```

Downstream consumers of the stash, run separately:

```
$ & $PY tools\rt.py tests\test_activation_release.py tests\test_baseline_parity.py tests\test_target_centering_state.py tests\test_seed.py
collected 80 items
tests\test_activation_release.py ..........                              [ 12%]
tests\test_baseline_parity.py .......................                    [ 41%]
tests\test_target_centering_state.py ..........                          [ 53%]
tests\test_seed.py .....................................                 [100%]
============================ 80 passed in 25.97s =============================

$ & $PY tools\rt.py tests\test_cgn.py tests\test_swip.py tests\test_jawp.py tests\test_pcr.py
collected 138 items
============================ 138 passed in 7.28s ==============================

$ & $PY tools\rt.py tests\test_v025_integration.py --slow tests\test_training_state_guards.py
collected 104 items
======================= 103 passed, 1 xfailed in 3.50s ========================
```

The 1 `xfail` is pre-existing and untouched by this diff (it is in
`test_training_state_guards.py`, which this change cannot reach; an introduced
regression would surface as `xpass`). `test_sterility.py` passes.

`ruff check` and `black --check` on both files: clean.

---

## 5. Diff-stat

```
 src/models/jepa.py               |  35 +++-
 tests/test_activation_release.py | 352 +++++++++++++++++++++++++++++++++++++++
 2 files changed, 385 insertions(+), 2 deletions(-)
```

`jepa.py` is `+33 / -2`: the two `= None` assignments moved from mid-forward to
the top, plus 31 lines of comment. No behaviour outside `compute_loss_with_targets`
touched. No skip, no `xfail`, no `--no-verify`, no test deleted or weakened. The
`.agent-notes/_probe_*.py` scratch files are covered by `.gitignore` (`:73`) and
are not in the commit.

---

## 6. Mutation verdict

Four mutations, each reverted. Three were the card's own suggestions; one found
a hole in my own test.

| # | mutation | result |
|---|---|---|
| 1 | revert to base (stashes reset mid-forward; empty batch pins the previous pass) | **RED** ×2 — `test_empty_batch_does_not_pin_the_previous_pass` (`assert ... ._gac_z is None`, got a `grad_fn=<IndexPutBackward0>` tensor) and `test_release_happens_before_the_new_graph_is_built` |
| 2 | `self._gac_z = span_preds.detach()` | **RED** ×5 — the detach the card warned about is caught by five tests |
| 3 | `"slots": span_preds.detach()` | **RED** ×3 on first run — and one test **survived**: `test_secondary_path_moves_encoder_gradients`, because it backpropped the primary loss together with the CMC term, leaving encoder gradients non-zero. Strengthened to backprop the CMC term alone; re-run against the same mutant → **RED** ×4 |
| 4 | inline `scaled_loss = total / grad_accum` in the measurement harness instead of naming it | not a source mutation, but the important one: it made the "today" arm read **93.89 MB instead of 110.21 MB**. The harness has to mirror `train.py`'s locals or it measures a different program. Corrected before any number was reported. |

**Load-bearing claims, each with the mutation that killed it:** the early
release (1), `_gac_z` must stay live (2), the secondary CMC slots must stay live
(3, after fixing the test that mutation 3 exposed). I make no claim without one.

---

## 7. не_сделано / риски

**не_сделано**

- **Peak activation memory is NOT halved.** +0.00 MB on the main accumulation
  path, measured. The card's premise — that `_gac_z` and `_cmc_pass["slots"]`
  hold the previous micro-batch's graphs — is false; `src/train.py`'s locals
  hold them. §3 has the patch and its measured value. `src/train.py` is
  forbidden for this card, so I did not apply it.
- **No change to `src/models/gac.py` or `src/models/cmc.py`**, both forbidden.
  Note `gac.py:162`'s docstring claims `z_pred` is "detached from graph" and it
  is **not** — `z_flat = z_pred.reshape(-1, D)` then `(z_flat**2).mean(dim=0)`.
  The code is right (the exploration gradient is the mechanism); the docstring
  is wrong and contradicts it. Reported, not fixed.
- I did not run the full suite (`tools/rt.py` refuses it by design) and did not
  train. The files above cover `jepa.py`'s own consumers: grep for
  `compute_loss_with_targets` returns 9 test files, 8 of which I ran plus
  `test_layer_analysis.py`, which does not call it.

**риски**

- **My change's benefit is narrow and its cost is a comment that could be
  misread.** The comment block in `jepa.py` is 31 lines on a file eight other
  workers are editing. I wrote the measured scope into it explicitly
  ("MEASURED SCOPE, read before assuming this halves peak memory: it does not")
  because a future reader who skips it would otherwise assume it does. If the
  maintainer would rather not carry a comment on a no-op line, the honest
  alternative is to drop the whole commit and keep only the test file — the
  tests are the part that has value regardless.
- **The census metric counts Python-visible tensors, not saved buffers.** It is
  undistorted (it never installs a hook) and it reproduces bit-for-bit, but a
  tensor held only inside a C++ `SavedVariable` is not gc-tracked the same way,
  so the absolute MB is a lower bound on the true figure. The *deltas* are what
  the conclusions rest on, and the deltas are consistent across three
  independent harnesses.
- **The census needs `gc.collect()` before it is read.** Reference cycles in the
  autograd graph defer collection; without the collect, "retention" and
  "uncollected garbage" are indistinguishable, and I reported the wrong number
  twice before catching it.
- **`weakref.ref(tensor)` is used in one test** to assert collectability. That is
  a CPython/PyTensor lifetime detail, not a language guarantee. It passed on
  torch 2.13.0+cpu; it is the most fragile assertion in the new file.
- **`test_training_state_guards.py` has 1 pre-existing xfail.** If a future
  change makes it `xpass`, that is a real finding, not noise — I did not
  investigate it.