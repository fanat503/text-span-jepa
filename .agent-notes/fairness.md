# Fairness defects in the baseline comparison — per-defect findings

Branch `agent/fairness`, worktree `C:\dev\wt-fair`, off `agent/wave-10` (`80b1916`).
Scope: three measured defects that make the BYOL / Barlow Twins / VICReg / SimSiam
comparison invalid. No training was run. Every claim below is either a pasted
measurement or a mutation verdict.

**Summary of decisions**

| # | Defect | Verdict | Change |
|---|--------|---------|--------|
| 1 | views are `(masked, clean)`, not two augmentations | Code is *honest but the naming is a lie*. Not repaired; **declared at the point of use** and pinned by tests | docstrings + config headers + 4 new tests |
| 2 | `data.mask_ratio` inert | **Dead knob. Deleted** from `defaults.yaml` and all 10 configs, behaviour-preserving | key removal + 3 new tests + 1 vacuity guard |
| 3 | drop-path asymmetry | **Code defect. Fixed** — every arm now gets the configured `drop_path_rate` | `create_model` + 6 baseline constructors + 3 new tests |

Two findings beyond the three defects are recorded at the end; one of them is a
test that was passing *vacuously* and would have stayed that way silently.

---

## Defect 1 — the two "views" are `(span-masked, clean)`

### What the code actually does

`src/train.py::compute_loss` dispatches
`model.compute_loss(masked_input_ids, original_input_ids, mask_positions)`.
Probed through the real `SpanMaskCollator` and the real dispatch, capturing the
tensors the arm is actually called with:

```
original == raw batch     : True
masked  == original       : False
n masked tokens           : 8 of 64
view_a is the masked input: True
view_b is the clean input: True
view_a == view_b          : False
```

So `view_a` is the span-masked input at the `0.15 -> 0.35` curriculum and
`view_b` is the untouched input. The pair is **nested**, not two independent
augmentation draws. All four papers define their objective over two *augmented*
views (SimSiam's are two random resized crops; BYOL/Barlow/VICReg use the
augmentation pipeline as a load-bearing part of the method).

### Decision: declare it at the point of use. Do not half-wire it.

This is a real divergence from the published methods and I am not going to
pretend otherwise. But wiring a second augmentation is a **science change, not
a bug fix**, and doing it halfway would be worse than not doing it:

1. **It needs a text augmentation policy per paper.** There is no published
   text equivalent of "RandomResizedCrop + colour jitter + solarisation". Any
   choice is this repo's invention and would itself need justifying.
2. **The augmentation strength is load-bearing for the collapse results.**
   Every threshold in `tests/test_ssl_baselines.py` was measured under a
   specific augmentation strength — and *differently per arm*: Barlow Twins is
   tested at 0.3 and BYOL at 0.8, because the two collapse under different
   conditions. Changing the view construction invalidates all of them.
3. **It needs a second independent draw per step** in `src/masks/span.py` plus
   a new trainer path, and `span.py` is shared with JEPA, MLM and data2vec.
4. **It would change published numbers**, which is an experiment decision for
   the owner, not one an agent should make silently.

Per the brief — *do not leave the four baselines silently misdefined* — the fix
is that **a method which is not the published method says so where it is used**:

- Each of the four `compute_loss` docstrings now names `view_a` as the trainer's
  SPAN-MASKED input and `view_b` as the CLEAN input, and carries a
  `NOT THE PUBLISHED VIEW CONSTRUCTION` section stating the divergence, naming
  the paper, and pointing at this note. This is the *point of use*: the parameter
  list of the function that consumes the views, not only a module header a
  refactor could drift away from.
- Each of the four `config/ablations/*.yaml` headers previously said
  "Recorded rather than papered over". That wording claims the problem was
  handled. It now says plainly: **this arm is not published BYOL/Barlow/VICReg/
  SimSiam**, with the same reasoning and the guard name.

**What a real fix would take** (for the owner, deliberately not attempted here):
add a per-arm augmentation policy object; draw two independent augmented views
per step in the collator; decide whether the mask curriculum applies to view A,
view B, or both; re-measure all collapse thresholds and the parameter-matching
band; and re-run every arm. That is a paper section, not a patch.

### Tests

`TestTheTwoViewsAreNotThePublishedAugmentationPair` — four tests, all four arms:

- `test_the_arm_declares_its_view_construction` — the docstring must name masked
  *and* clean. Scoped to the `view_a`/`view_b` lines: an earlier draft searched
  the whole docstring and passed BYOL/SimSiam on the word "masked" appearing in
  an unrelated sentence about `mask_positions`. A test satisfiable by a word in
  the wrong place is not a test; the tightened version fails all four arms
  against the pre-fix code.
- `test_the_masked_view_really_is_the_masked_one` — runs the real collator and
  the real `src.train.compute_loss`, spies on what the arm receives, and asserts
  `view_a == masked` and `view_b == clean`. The declaration is **checked against
  the trainer, not trusted**. If a future edit really does wire two
  augmentations, this goes red and forces the declarations to be updated in the
  same commit.
- `test_the_pair_is_not_two_independent_draws` — asserts the nesting property
  that makes the naming question unavoidable.

### Mutation verdict

Reverting the four docstrings to `two views` (the pre-fix text):

```
FAILED ...test_the_arm_declares_its_view_construction[barlow]
FAILED ...test_the_arm_declares_its_view_construction[byol]
FAILED ...test_the_arm_declares_its_view_construction[simsiam]
FAILED ...test_the_arm_declares_its_view_construction[vicreg]
4 failed, 8 passed
```

The two behaviour tests pass both before and after — **that is the point**: they
are characterisation tests of a fact that does not change, and they exist so the
documentation cannot drift away from the code in either direction.

---

## Defect 2 — `data.mask_ratio` is inert in every shipped config

### What the code actually does

`SpanMaskCollator.__init__` takes a flat `mask_ratio` **and** a
`mask_ratio_start`/`mask_ratio_end` ramp. `current_mask_ratio` returns the
**ramp** whenever `curriculum_steps > 0`, falling back to the flat value only
when there is no curriculum. `src/train.py:1442` sets `curriculum_steps = 10000`
whenever both ramp endpoints are present — and `defaults.yaml` declares both.

```
defaults data.mask_ratio        : 0.35
defaults model.mask_ratio_start : 0.15
defaults model.mask_ratio_end   : 0.35
step 0     current_mask_ratio : 0.15
step 4000  current_mask_ratio : 0.23002
no-ramp (curriculum_steps=0)  : 0.35
```

So `data.mask_ratio: 0.15`, set by **ten** shipped configs
(`config/wikitext/{mlm,data2vec}_wikitext_{base,large,small,xsmall}.yaml`,
`data2vec_wikitext_train.yaml`, `config/kaggle/{mlm,data2vec}_kaggle.yaml`), did
nothing at all. Every one of those runs used the 0.15 -> 0.35 ramp.

### Decision: **delete it**, do not wire it

- **Wiring it** would change the mask schedule of every arm in the repo,
  including runs whose results may already be quoted. That is a science decision
  about what those experiments *were*, and it is not mine to make silently.
- **Deleting it is behaviour-preserving.** Proven, not assumed — the full
  10 000-step curriculum compared with and without the key:

  ```
  identical over the whole 10000-step run: True
  step 0    : 0.15 0.15
  step 5000 : 0.25 0.25
  step 10000: 0.35 0.35
  differing steps: 0
  ```

- Deleting also **strengthens the existing typo detector**: `data.mask_ratio` is
  no longer a path in `defaults.yaml`, so any future config that sets it now
  gets the "possible typo" warning instead of being silently accepted.

### Why the tests that pinned it should change rather than be weakened

`data.mask_ratio` was in `LADDER_CONSTANTS`, pinned as "constant along the
capacity ladder". **That pin was asserting a property of a number no run used**,
so it was never protection — it read as coverage and was not. The mask schedule
is still pinned along the ladder, by the two keys that actually drive it,
`model.mask_ratio_start` and `model.mask_ratio_end`, both still in the tuple.
Coverage is not lost; the misleading half is removed. Removing it from the
tuple is the *correct* edit, and weakening anything else would not be.

### The vacuous test this uncovered (important)

With the key deleted, `test_constant_along_the_ladder[data.mask_ratio]` still
went **green** — because `_get` returns the `KeyError` *class* when a path is
missing, and every rung then has the identical `repr`, so the assertion
"all four rungs agree" is trivially true for a key that exists nowhere:

```
per-rung values: {'xsmall_30m.yaml': "<class 'KeyError'>", ...}
distinct reprs: 1
=> test_constant_along_the_ladder PASSES VACUOUSLY: True
```

A guard was added: `TestScalingLadder::test_every_pinned_constant_actually_exists`,
which asserts every `LADDER_CONSTANTS` entry **resolves**. Without it, the
deletion would have left a permanently-vacuous green test asserting the scaling
ladder is controlled.

### Scope note

`src/train.py` still passes `data_cfg.get("mask_ratio", 0.35)` to the collator,
so **hand-built config dicts that supply no ramp still work**. Two out-of-scope
fixtures do exactly that (`tests/test_training_e2e.py:54`,
`tests/test_checkpoint_fidelity.py:753`, both `mask_ratio: 0.3`) — verified live:

```
hand-built fixture (no ramp) -> current_mask_ratio: 0.3
shipped defaults (ramp)     -> step0: 0.15
```

They are not deep-merged over `defaults.yaml`, so they are unaffected; they now
log one "possible typo" warning per run, which is *correct* — the key is no
longer part of the config schema. No test in either file asserts on warnings
(checked: zero `caplog`/`assert.*warn` matches). I did not modify those files;
they are not mine and their behaviour is unchanged.

### Tests

`TestOnlyOneMaskRatioKnobIsLive` — `test_data_mask_ratio_is_not_declared_in_defaults`,
`test_no_config_sets_the_inert_key`, and `test_the_ramp_is_what_actually_drives_the_ratio`
(measures the collator's schedule from the shipped defaults, so it goes red if
the key ever becomes live again).

### Mutation verdict

Re-adding the key to `defaults.yaml`:

```
FAILED ...test_data_mask_ratio_is_not_declared_in_defaults
AssertionError: defaults.yaml declares data.mask_ratio, but src/train.py always passes a
mask_ratio_start/end ramp, so SpanMaskCollator.current_mask_ratio returns the ramp and the
value is never read...
```

Re-adding `mask_ratio: 0.15` to one config (key still absent from defaults):

```
FAILED ...TestKeyPaths::test_every_key_exists_at_its_exact_path[config/wikitext/mlm_wikitext_small.yaml]
FAILED ...TestOnlyOneMaskRatioKnobIsLive::test_no_config_sets_the_inert_key
```

Re-pinning the deleted key in `LADDER_CONSTANTS` (the vacuity guard):

```
FAILED ...TestScalingLadder::test_every_pinned_constant_actually_exists
AssertionError: LADDER_CONSTANTS pins ['data.mask_ratio'], which no scaling rung declares...
test_constant_along_the_ladder PASSES VACUOUSLY for these
```

All mutations reverted; `grep` confirms zero `mask_ratio: 0.15` remain.

---

## Defect 3 — drop-path asymmetry: JEPA at 0.1, baselines at none

### What the code actually does

Measured at `depth=2, drop_path_rate=0.1` through `create_model`:

```
text_span_jepa   ['Identity', 'DropPath']
mlm              ['Identity', 'Identity']
data2vec         ['Identity', 'Identity']
byol             ['Identity', 'Identity']
barlow           ['Identity', 'Identity']
vicreg           ['Identity', 'Identity']
simsiam          ['Identity', 'Identity']
```

`create_model` forwarded `drop_rate` to all six baseline branches and
`drop_path_rate` to none of them. At the reference config the JEPA column ran
stochastic depth and every baseline column ran `nn.Identity`.

**There were two layers, and both are now fixed.** `create_model` never passed
the argument; and *even if it had*, all six baseline classes end their
constructor with `**kwargs` and never forward it, so
`BYOLBaseline(drop_path_rate=0.1)` constructs successfully, accepts the argument,
and trains with no drop-path:

```
byol with drop_path_rate kwarg: ['Identity', 'Identity']
```

That second layer is why the tests read `block.drop_path.drop_prob` off the
built modules rather than checking that the constructor accepted the argument.

### Decision: **pass it to every arm**

Neither reading of "the convention" survives contact with the code:

- *Pass it* — every arm builds the same `TextSpanJEPAEncoder` trunk, and
  `drop_path_rate` is a property of that trunk, not of any method. The JEPA
  branch honoured it and the baseline branches did not, which is an omission in
  a factory branch, not a methodological position. `drop_rate` was read by all
  six; `drop_path_rate` by one.
- *State that baselines do not use it* — nothing supports this. No paper here
  specifies stochastic depth either way, and there is no mechanism-specific
  argument for it. Declaring it would mean inventing a convention that no code
  ever expressed, to excuse an asymmetry that is plainly an oversight.

A baseline row is read as "this method at this budget". If one column
regularises and another does not, the row measures the method **and** the
regulariser. Note the direction of the error is not knowable in advance —
stochastic depth can help or hurt at a fixed budget — so this is not a
correction that can be waved through as "obviously helps the baselines".

`defaults.yaml` already declares `drop_path_rate: 0.1` and already pins it along
the scaling ladder ("drop-path is a regulariser; letting it track model size
makes a capacity curve uninterpretable"), so the config author clearly intended
it to be a property of every run.

### Changes

`create_model` now passes `drop_path_rate=model_cfg.get("drop_path_rate", 0.0)`
to the `mlm`, `data2vec` and `SSL_BASELINE_ARMS` branches; all six baseline
constructors accept it and forward it to `TextSpanJEPAEncoder`. The branch
comment is updated to record that `drop_path_rate` is deliberately read while
*method* hyperparameters are deliberately not.

### Tests

`TestRegularizationParity` — spans all seven arms, not just the four new ones,
because a property holding for four of six baselines is not a property of the table:

- `test_the_jepa_column_is_not_the_only_one_with_stochastic_depth` — the defect
  in one assertion, plus a guard that the JEPA column really does have drop-path
  (if that ever becomes false the defect has *moved*, and the message says so).
- `test_every_arm_honours_a_configured_drop_path` — non-zero reaches the encoder
  **and** zero does not, which is the pair that pins the argument as load-bearing
  in both directions and catches the `**kwargs` swallow from either side.
- `test_the_rate_the_config_declares_is_the_rate_every_arm_gets` — reads the
  number out of `defaults.yaml` and requires every arm to reproduce it, so
  arm-specific special cases cannot satisfy it.

### Mutation verdict

Layer 1 — remove `drop_path_rate` from the SSL branch of `create_model`:

```
FAILED test_the_jepa_column_is_not_the_only_one_with_stochastic_depth
FAILED test_every_arm_honours_a_configured_drop_path[byol|barlow|vicreg|simsiam]
FAILED test_the_rate_the_config_declares_is_the_rate_every_arm_gets
AssertionError: byol resolves to drop-path [0.0, 0.0] at the declared
drop_path_rate=0.1, against the JEPA column's [0.0, 0.10000000149011612]
```

Layer 2 — keep `create_model` passing it but make `BYOLBaseline` swallow it
again (the `**kwargs` trap):

```
FAILED test_the_jepa_column_is_not_the_only_one_with_stochastic_depth
FAILED test_every_arm_honours_a_configured_drop_path[byol]
FAILED test_the_rate_the_config_declares_is_the_rate_every_arm_gets
```

Note the second mutation fails **only the BYOL cases** while barlow/vicreg/
simsiam stay green — isolating exactly the layer it targets. Both mutations
reverted.

---

## Risks / things the owner should decide

1. **No numbers were re-run.** These arms have never been trained; there are no
   results to invalidate. But the *first* comparison run of these four arms will
   now be the first run at parity on drop-path **and** the first run labelled as
   not-the-published-method. Both must appear in the results write-up.
2. **The arm names still say `byol`/`barlow`/`vicreg`/`simsiam`.** I kept them:
   renaming to e.g. `byol_masked` is a schema change touching `LOSS_PROTOCOLS`,
   `_ARM_PREFIXES`, six configs and several tests, and it hides rather than
   solves the divergence. The divergence is now stated in three places per arm
   (class docstring, config header, this note). If the owner wants honest names
   in the results table, that is a separate, explicit decision.
3. **The `data.mask_ratio` deletion touches ten configs**, including
   `data2vec_wikitext_train.yaml`. Behaviour is provably identical, but if any
   of those runs is quoted elsewhere, the *config text* changed even though the
   *run* did not.
4. **Two out-of-scope fixtures now log a warning** per run
   (`test_training_e2e.py`, `test_checkpoint_fidelity.py`). Behaviour unchanged
   and no test asserts on it; I did not modify files outside my scope.
5. **The vacuity guard generalises.** Any future key deleted from `defaults.yaml`
   while still listed in `LADDER_CONSTANTS` will now fail loudly instead of
   turning that pin green forever.

## Verification

```
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"

& $PY -m ruff check C:\dev\wt-fair
All checks passed!

& $PY -m black --check C:\dev\wt-fair
125 files would be left unchanged.

& $PY tools\rt.py --slow tests\test_config_system.py tests\test_ssl_baselines.py
collected 762 items
tests\test_config_system.py ............................................ [  5%]
...
tests\test_ssl_baselines.py ............................................ [ 90%]
..................                                                   [100%]
====================== 729 passed, 33 skipped in 60.62s (0:01:00) ======================
```

Baseline before this branch: **707 passed, 33 skipped**. Now **729 passed, 33
skipped** — +22 tests, 0 failures, 0 skips added, nothing weakened.

Test counts for the whole suite are **not** quoted here: per `AGENTS.md` a local
full-suite run saturates the owner's 6-core box and training is off-limits, so
the suite total must come from CI. The two files above are the only ones I ran,
and both were run only through `tools/rt.py`.

## Files touched (all inside `C:\dev\wt-fair`, none shared with other agents)

- `src/train.py` — `create_model` drop-path forwarding (3 branches) + comment
- `baselines/{byol,barlow,vicreg,simsiam}_baseline.py` — `drop_path_rate` param
  + view-construction declarations
- `baselines/{mlm,data2vec}_baseline.py` — `drop_path_rate` param
- `defaults.yaml` — `data.mask_ratio` deleted, with the reason
- `config/ablations/{byol,barlow,vicreg,simsiam}.yaml` — "not the published
  method" headers
- `config/ablations/{mlm,data2vec}.yaml` — dead-key comments corrected
- `config/wikitext/*.yaml`, `config/kaggle/*.yaml` — 10 x `mask_ratio: 0.15`
  removed
- `tests/test_ssl_baselines.py` — `TestRegularizationParity` (3),
  `TestTheTwoViewsAreNotThePublishedAugmentationPair` (4)
- `tests/test_config_system.py` — `TestOnlyOneMaskRatioKnobIsLive` (3),
  `TestScalingLadder::test_every_pinned_constant_actually_exists` (1),
  `LADDER_CONSTANTS` de-pinned with the reason recorded