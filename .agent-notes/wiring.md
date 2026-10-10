# Wiring the SSL baselines into the training entry point

Branch `agent/wire-baselines`, worktree `C:\dev\wt-wire`, base `agent/ssl-baselines`
(`f07c8aa`). One commit: `dafcded`. Not pushed.

Files touched — both owned, nothing else:

| File | Δ |
|---|---|
| `src/train.py` | +262 / −20 |
| `tests/test_ssl_baselines.py` | +357 / −10 (`TestTrainerContract` only, plus one module-level helper and two imports) |

**`src/models/mechanisms.py` was NOT touched.** Reasoning below.

---

## 1. The dispatch rule

The rule is now **declared, not inferred**.

```
LOSS_PROTOCOLS : canonical arm name -> one of three NAMED protocols
                     "text_span_jepa" -> jepa_targets      (compute_loss_with_targets(a,b,mask,**) -> 3-tuple)
                     "data2vec"        -> data2vec_forward  (model(a,b,mask)                        -> (loss, info))
                     "mlm"             -> compute_loss
                     "byol"            -> compute_loss
                     "barlow"          -> compute_loss
                     "vicreg"          -> compute_loss
                     "simsiam"         -> compute_loss

create_model(...)  ->  model.loss_protocol = LOSS_PROTOCOLS[model_name]   # stamp, table lookup
compute_loss(...)  ->  protocol = model.loss_protocol  (fallback: _infer_loss_protocol)
                       dispatch on `protocol`, ONE call site per protocol
```

`compute_loss` has exactly three branches, each one a named protocol rather than
a property of the model object. An arm added to `create_model` without a
`LOSS_PROTOCOLS` entry raises `KeyError` at construction — at step 0, not in a
results table six weeks later.

### Why the old chain was the trap

The old chain was, in order:

```
hasattr(model, "compute_loss_with_targets")                        -> JEPA
hasattr(model, "forward") and hasattr(model, "regression_head")    -> data2vec
hasattr(model, "compute_loss")                                     -> MLM
```

Both earlier branches are **positive tests for an incidental attribute name**, so
the branch an arm landed in was a property of what it happened to be *called*.
`regression_head` is precisely the obvious name for a regression-style head.

**A correction to the framing in my brief**, because it changes how the trap reads:
the four new arms would *not* have been silently routed to data2vec's loss. None
of them owns an attribute called `regression_head`, so they would have fallen
through to the `compute_loss` branch and been routed **correctly**. I verified
this: the sibling's own `TestTrainerContract::test_arm_is_dispatched_to_the_compute_loss_branch`
asserts exactly that absence, and it passes today.

So the real defect is subtler and worth stating plainly: the four arms were safe
by **luck, not by design**. The guarantee was "works as long as nobody names a
head `regression_head`", and that is a guarantee about the *absence* of a name,
which no reviewer can verify by reading the arm and no test in the file was
checking. I did not want to ship a card whose whole justification was a
defect that had not actually fired, so the fix is the declared protocol above,
and `test_a_decoy_regression_head_no_longer_reroutes_the_arm` turns the
hypothetical into a test that fails loudly today.

### The compatibility path, and why the signature did not change

`compute_loss`'s signature is **unchanged**, deliberately:

- `tests/test_grad_scaler.py::test_validate_forwards_current_step` and
  `tests/test_checkpoint_fidelity.py::test_validation_model_without_buffers_still_runs`
  both `monkeypatch` / reassign `src.train.compute_loss` with a stub of exactly
  `(model, masked, original, mask, current_step=0, total_steps=1)`. Neither file
  is mine. Had `main()` and `_validate()` gained a `model_name=` keyword, both
  would die with `TypeError: unexpected keyword argument`.

So the declared protocol travels on the **model** rather than on the call. That
is what let the fix land without touching a test file this card does not own.

`_infer_loss_protocol` preserves the old chain verbatim, narrowed to returning
one of the three constants. It has two callers: models a caller or test built by
hand (`tests/test_model.py` constructs `TextSpanJEPA`, `MLMBaseline` and
`Data2VecTextBaseline` directly and hands them to `compute_loss` — **that test
is the reason this fallback exists and still passes**), and any future arm wired
in without going through `create_model`. Nothing the trainer builds is ever
routed by it, so the trap it encodes cannot reach a training run.

---

## 2. The second silent routing I found, which the brief did not name

`do_ema_update` handled `text_span_jepa` (with `tau`) and `data2vec` only. Wiring
BYOL into `create_model` alone would have produced an arm that **trains with its
EMA teacher frozen at random initialisation** — the run descends, logs a falling
loss, writes checkpoints, and is not BYOL. Nothing anywhere would have failed.

Two statements had to change together, and changing only one is the failure mode:

- `do_ema_update` → `elif model_name in SELF_EMA_ARMS`
- `main()`'s per-step `elif model_name == "data2vec"` → `elif model_name in SELF_EMA_ARMS`

The second is guarded by `test_the_loop_reaches_ema_through_that_set`, because a
future edit could plausibly fix the function and forget the call site.

Also in this family, and equally silent:

- **`get_param_groups`**: without a branch, all four fall to the `else` catch-all
  `[{"params": list(model.parameters())}]`, which (a) puts BYOL's **frozen**
  `target_encoder`/`target_projector` into the optimizer — a parameter that can
  never receive a gradient — and (b) applies `weight_decay=0.04` to *everything*
  including LayerNorm scales and biases, against the convention every other arm
  in this repo follows. A baseline trained that way is not a faithful baseline,
  and the arms still converge, so nothing looks wrong.
- **`num_updates`**: a plain `int`, absent from `state_dict()`, so a resume
  rewinds it to 0. Now checkpointed and restored for BYOL alongside data2vec.
  For BYOL this is currently **cosmetic** (its momentum is constant, so the
  counter feeds nothing) — I included it because `save_checkpoint`'s stated
  contract is that a resume be indistinguishable from an uninterrupted run, and
  this key is the one place that claim would have been false for the new arms.

---

## 3. The regression check

`src/train.py` is a hotspot, so the net is "nothing existing moved". Every file
that imports `src.train` was run, before and after.

**Baseline, before any edit:**

```
rt.py tests\test_ssl_baselines.py tests\test_config_system.py --slow
638 passed, 21 skipped in 61.42s
```

**After, same command:**

```
669 passed, 21 skipped in 71.91s
```

669 − 638 = **+31**, all of it new (`TestTrainerContract` 21 → 52 cases). Suite
collection re-derived by stashing the two files, because AGENTS.md is right that
a count nobody re-measured is worse than none:

```
2058 tests collected   (baseline, stashed)
2089 tests collected   (with this branch)   -> exactly the +31
```

`AGENTS.md`'s written `1960` is stale against this base — noted, not "fixed",
since that file is not mine.

Coupled files, all green after the change:

```
tests\test_model.py --slow                                             155 passed
tests\test_baseline_parity.py tests\test_grad_scaler.py
tests\test_train_device.py tests\test_checkpoint_fidelity.py           77 passed
tests\test_run_comparison.py tests\test_v025_integration.py --slow     82 passed
tests\test_training_e2e.py --slow                                       3 passed
```

`rt.py` refused one batch for budget and killed it at 90s rather than saturating
the box (it is shared); `test_training_e2e.py` was re-run alone. No skips, no
xfails, no `--no-verify` added, no training run.

`ruff check` and `black --check` clean on both files.

---

## 4. Mutation verdict — three mutations, each red, all restored

**A. Corrupt the routing table** (`LOSS_PROTOCOLS["byol"] = LOSS_DATA2VEC_FORWARD`):

```
FAILED ...test_create_model_declares_the_protocol_instead_of_inferring_it[byol]
  AssertionError: assert 'data2vec_forward' == 'compute_loss'
FAILED ...test_a_decoy_regression_head_no_longer_reroutes_the_arm[byol]
  src\train.py:761: in compute_loss
      loss, info = model(masked_input_ids, original_input_ids, mask_positions)
  TypeError: BYOLBaseline.forward() takes 3 positional arguments but 4 were given
2 failed, 50 passed
```

That `TypeError` is the trap itself: BYOL handed data2vec's call signature. Loud,
and at step 0.

**B. Revert the loop's EMA branch** to `elif model_name == "data2vec":`

```
FAILED ...test_the_loop_reaches_ema_through_that_set
  AssertionError: main()'s EMA branch no longer tests SELF_EMA_ARMS; an arm
  added to that set would silently train without its teacher being advanced
1 failed, 51 passed
```

**C. Disable only the `get_param_groups` branch** (arms fall to the catch-all):

```
FAILED ...test_the_optimizer_holds_exactly_the_trainable_parameters[byol]
FAILED ...test_the_encoder_keeps_the_repo_weight_decay_split[barlow]
FAILED ...test_the_encoder_keeps_the_repo_weight_decay_split[byol]
FAILED ...test_the_encoder_keeps_the_repo_weight_decay_split[simsiam]
FAILED ...test_the_encoder_keeps_the_repo_weight_decay_split[vicreg]
5 failed, 47 passed
```

BYOL alone fails the optimizer test — the frozen teacher is BYOL-specific, and
the other three have no teacher, so this assertion is narrow on purpose. All
four fail the weight-decay split.

First attempt at C disabled both `create_model` and `get_param_groups` (shared
`SSL_BASELINE_ARMS` guard) and went 22 red; I redid it narrowed so the verdict
isolates the optimizer branch rather than "the arm vanished".

All three restored and re-verified green before commit.

---

## 5. `MechanismBundle`: the sibling's claim, and whether to touch it

**The claim is true for `src/`, with one correction of scope.**

- `MechanismBundle` appears in `src/` **only inside `src/models/mechanisms.py`**
  (class def, docstring, `from_config`, `GWP(MechanismBundle)`, `__all__`).
- `src/models/jepa.py` does not import it — the three `mechanisms` hits are
  comments. `TextSpanJEPA` wires its twelve inline.
- **But it is not dead code**: `scripts/results_apparatus.py` constructs it at
  five sites (`results_apparatus.py:505` is `MechanismBundle.from_config(...)`),
  and `tests/` constructs it too. So the accurate statement is "not on the
  *training* path", not "unreachable". `README.md:156` and
  `proofs/wsd.md:249` already phrase it this way.

**My change does not touch it, and should not.** These four arms are *external
baselines* — comparison arms, not GWP mechanisms. Adding them to
`ALL_MECHANISMS` would take the count 12 → 16 and turn two pinned tests red
(`test_all_mechanisms_length_is_stable`, `test_gwp_import`) while landing the
wrong thing in the concept: `LOSS_PROTOCOLS` and `SSL_BASELINE_ARMS` are the
registry for these arms, and they live in `src/train.py`, which is the file
that owns model construction.

The divergence risk between the bundle and `jepa.py` is real and pre-existing
(`proofs/IMPLEMENTATION_STATUS.md`, V8) and is **not** something this card
should absorb — it is a separate, larger card about the training path.

---

## 6. Риски / risks

1. **No config file exists for these arms.** `meta.model_name: byol` in a new
   `config/wikitext/*.yaml` is all that is needed; nothing shipped today.
   `config/` and `defaults.yaml` are not mine, so none was written. **Follow-up
   for whoever owns them.**

2. **No method hyperparameter is settable from config.** `target_momentum`,
   `lambda_offdiag`, `sim_weight`/`var_weight`/`cov_weight`/`gamma` all take
   their published-paper constructor defaults, so an arm built from
   `defaults.yaml` alone trains the method as published. This is a **deliberate
   constraint, not an oversight**: each such key needs `defaults.yaml` +
   `extra_known` in `src/train.py` + `_TRAINER_EXTRA_KNOWN` in
   `tests/test_config_system.py`, and `test_trainer_extra_known_matches_the_trainer`
   fails if only the first moves. Exempting a key no shipped config uses is a
   permanent hole in the typo detector, bought for a knob that already has a
   correct default.
   `test_create_model_reads_only_keys_defaults_yaml_declares` reads the branch's
   `model_cfg.get(...)` calls **out of `create_model`'s source**, so adding one
   turns that test red and names the two files that must move with it.

3. **`BatchNorm1d` train-mode batch-of-one is a live limit**, not a port bug —
   these four heads are BatchNorm-based, as the reference implementations are.
   A `batch_size: 1` config trains fine; it just cannot have `train()` called on
   it at that size. Already documented by
   `TestShapesAndEdgeCases::test_train_mode_batch_of_one_is_a_batchnorm_limit_not_a_port_bug`,
   but a config reviewer will hit it.

4. **Validation with `max_batches=1` in train-shaped batches is fine** — `_validate`
   calls `model.eval()`, so BatchNorm uses running stats. Verified by the
   `create_model` → `compute_loss` → `save_checkpoint` → `load_checkpoint` smoke
   over all seven arms.

5. **`_normalize_model_name` changed shape** from an if-chain to a table-driven
   loop. Same order semantics (`text_span_jepa` is tested before `jepa`), and
   `tests/test_model.py::TestGWPFramk`'s three normalisation tests plus
   `tests/test_config_system.py` (571 cases) pass unchanged. The pinned
   correspondence `_ARM_PREFIXES` ⇄ `LOSS_PROTOCOLS` is asserted in **both**
   directions by
   `TestTrainerContract::test_the_config_name_normalises_to_a_declared_arm`,
   because a one-way check passes when only one table has drifted.

6. **The `create_model` error message changed** from a literal list to
   `sorted(LOSS_PROTOCOLS)`, so it now names all seven arms. No test pins that
   string (grepped); it is a user-facing improvement and the reason it is
   derived rather than hand-maintained.

7. **Untested by construction: the full `main()` loop for the four arms.** I ran
   `create_model` → `compute_loss` → `get_param_groups` → `do_ema_update` →
   `save_checkpoint` → `load_checkpoint` for all seven arms in a scratch script,
   and every existing e2e test still passes, but no test trains one of these
   arms through `main()` on real data — that needs the wiki corpus, and **no
   training on this machine**. The residual risk is concentrated in exactly the
   place this card changed: the loop's EMA branch, which is guarded by a source
   assertion rather than by an executed run. **Recommend the first real run of
   `byol` confirm the teacher is moving** (log `ema_momentum` / compare teacher
   and student weights at step N) before the number is believed.

8. **Test budget.** The 31 new cases are all construction-and-dispatch, no
   training, so `TestTrainerContract` runs in **0.83s**. No pressure on
   `rt.py`'s 90s budget. The file total moved 61s → 72s, which is measurement
   noise plus 31 cheap tests, not new CPU load.