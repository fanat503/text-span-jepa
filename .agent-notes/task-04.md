# TASK-04 — the ema_tau_end fallback no longer freezes the target encoder

- **status**: done
- **files**: `src/train.py` (1 literal + comment), `tests/test_config_system.py`
  (skip marker removed, no assertion touched)
- **commit**: `94a9f34` on `agent/task-04`

## choice: `default 0.9999`, not `raise`

I took the second of the two offered options. Three reasons, in order of weight:

1. **`raise` would not have satisfied the acceptance criterion as written.**
   `test_trainer_ema_fallback_is_not_the_frozen_value` reaches its
   `assert` only via `re.search(r'model_cfg\.get\("ema_tau_end",\s*([0-9.]+)\)',
   _TRAIN_SRC)`. Deleting the `.get()` to raise instead makes the regex miss,
   and the test then hits its *other* branch —
   `pytest.skip("src/train.py no longer supplies an ema_tau_end fallback")`.
   The test would still be a **skip**, i.e. the exact thing this card exists to
   delete. So `raise` was not actually on the table here.

2. **Consistency is the actual fix.** `0.9999` is already the value in four
   places; the trainer was the lone outlier:
   - `defaults.yaml:101` → `model.ema_tau_end: 0.9999`
   - `src/models/jepa.py:72` → `kwargs.get("ema_tau_end", 0.9999)`
   - `src/utils/schedulers.py:75` → `def __init__(..., tau_end=0.9999, ...)`
   - `src/train.py:836` → **was `1.0`**

   `create_model` (train.py:470) builds `TextSpanJEPAConfig(**model_cfg)` and
   calls `config.validate()`; on a missing key the dataclass default `0.9999`
   applies and validation passes. So pre-fix, a `--no_defaults` run produced a
   model whose config said `0.9999` and a schedule that said `1.0`, and the
   schedule won — it is what gets passed to `update_target_encoder`. The
   mismatch is the bug; aligning the literal removes it at the source rather
   than papering over the symptom.

3. **`raise` would have added a *new* divergence.** It makes the trainer reject
   a key that `TextSpanJEPAConfig` and `EMATauSchedule` both accept, i.e. a
   fresh inconsistent pair in place of the old one — trading one
   two-sources-disagree bug for another.

Explicit `ema_tau_end >= 1.0` was already fatal before my change, via
`config.validate()` at jepa.py:354, and `create_model` runs first
(train.py:1132) — before `_build_optimization` at :1176. So the fallback is
only reachable via a *missing* key, and no new guard is needed. I did not add
one: the card says do not change other behaviour, and a literal
`0.999 < 1.0 <= tau_end` check would be exactly that.

## verify

### 1. The previously-skipped test now RUNS

Before the fix, `-rs` explicitly reported the skip:

```
$ & $PY tools\rt.py tests/test_config_system.py --slow -k "ema" -rs
SKIPPED [1] tests\test_config_system.py:973: KNOWN src/train.py DEFECT (not a config
defect): the ema_tau_end fallback is 1.0, which freezes the target encoder on any
--no_defaults run. Every shipped config supplies the key, so no config here is
affected. Fix: src/train.py line ~836, default 0.9999.
=============== 112 passed, 21 skipped, 445 deselected in 4.03s ===============
```

After — `PASSED`, and the skip is gone from the summary entirely:

```
$ & $PY tools\rt.py tests/test_config_system.py --slow -k "trainer_ema_fallback" -v -rs
tests/test_config_system.py::TestNoDeadKeys::test_trainer_ema_fallback_is_not_the_frozen_value PASSED [100%]
====================== 1 passed, 577 deselected in 2.78s ======================
```

Full file — skip count drops 22 → 21, pass count rises 556 → 557, i.e. exactly
one skip converted to a pass:

```
$ & $PY tools\rt.py tests/test_config_system.py --slow -rs
SKIPPED [1] tests\test_config_system.py:399: JAWP is the root: WSD is constructed only when JAWP is on
SKIPPED [10] tests\test_config_system.py:702: data2vec / MLM baselines own their EMA
SKIPPED [10] tests\test_config_system.py:715: data2vec / MLM baselines own their EMA
======================= 557 passed, 21 skipped in 9.53s =======================
```

### 2. `tests/test_checkpoint_fidelity.py`

```
$ & $PY tools\rt.py tests/test_checkpoint_fidelity.py
tests\test_checkpoint_fidelity.py ....................                   [100%]
======================= 20 passed, 3 warnings in 12.64s =======================
```

Also ran the EMA subset of `tests/test_model.py`, since that file reads the same
key: `10 passed, 141 deselected in 2.36s`.

### 3. Throwaway probe — target actually moves, on BOTH paths

Untracked script, run then deleted (never staged; `git status` clean apart from
the two committed files). It builds a real `TextSpanJEPA` (28 encoder/target
param pairs), sets `p_k := 0` and `p_q := 1`, calls the real
`TextSpanJEPA.update_target_encoder(tau)` with the tau produced by
`_build_optimization` for each path. With those sentinels the expected result is
exactly `tau*0 + (1-tau)*1 = 1 - tau`, so the check is exact, not approximate.

```
=== defaults-merged (defaults.yaml model:) ===
  model.ema_tau_start in cfg : 0.996
  model.ema_tau_end   in cfg : 0.9999
  EMATauSchedule.tau_start   : 0.996
  EMATauSchedule.tau_end     : 0.9999
  tau after 2 step(s)   : 0.9999
  1 - tau (expected delta)   : 9.999999999998899e-05
  target param BEFORE        : 0.0
  target param AFTER         : 9.999999747378752e-05
  expected AFTER             : 9.999999999998899e-05
  |AFTER - expected|         : 2.526e-12
  all 28 target params moved off 0.0: True

=== --no_defaults (key absent) ===
  model.ema_tau_start in cfg : <absent>
  model.ema_tau_end   in cfg : <absent>
  EMATauSchedule.tau_start   : 0.996
  EMATauSchedule.tau_end     : 0.9999
  tau after 2 step(s)   : 0.9999
  1 - tau (expected delta)   : 9.999999999998899e-05
  target param BEFORE        : 0.0
  target param AFTER         : 9.999999747378752e-05
  expected AFTER             : 9.999999999998899e-05
  |AFTER - expected|         : 2.526e-12
  all 28 target params moved off 0.0: True

=== both paths agree ===
  defaults-merged (defaults.yaml model:)   -> (tau_start, tau_end, tau, target_after) = (0.996, 0.9999, 0.9999, 9.999999747378752e-05)
  --no_defaults (key absent)               -> (tau_start, tau_end, tau, target_after) = (0.996, 0.9999, 0.9999, 9.999999747378752e-05)
  IDENTICAL: (0.996, 0.9999, 0.9999, 9.999999747378752e-05)

OK: target encoder moves on both paths, and the paths agree.
```

The `9.999999747378752e-05` vs `9.999999999998899e-05` gap is float32 storage
(9.999999747378752e-05 is exactly the nearest float32 to 1e-4); the 2.5e-12
residual is that representation, not a schedule error.

### 4. The same probe with `1.0` restored — this is the money shot

Mutating the literal back to `1.0` and re-running the *unmodified* probe
reproduces the defect and shows the two paths **diverge** on the same model:

```
  target param BEFORE        : 0.0
  target param AFTER         : 9.999999747378752e-05     <- defaults-merged: still moves
...
=== --no_defaults (key absent) ===
  EMATauSchedule.tau_end     : 1.0
  tau after 2 step(s)   : 1.0
  1 - tau (expected delta)   : 0.0
  target param BEFORE        : 0.0
  target param AFTER         : 0.0
  expected AFTER             : 0.0
  all 28 target params moved off 0.0: False
    assert after != before, "TARGET ENCODER FROZEN"
AssertionError: TARGET ENCODER FROZEN
```

All 28 target parameters stay at their initial value for the entire run.

### 5. lint / format

```
$ & $PY -m ruff check . --output-format concise
All checks passed!
exit 0
$ & $PY -m black --check .
All done! ✨ 🍰 ✨
100 files would be left unchanged.
exit 0
```

## skip removed

`tests/test_config_system.py::TestNoDeadKeys::test_trainer_ema_fallback_is_not_the_frozen_value`

Proof it now runs rather than skips — `PASSED` with `-v`, and the
`SKIPPED [1] ...:973: KNOWN src/train.py DEFECT` line is absent from `-rs`:

```
tests/test_config_system.py::TestNoDeadKeys::test_trainer_ema_fallback_is_not_the_frozen_value PASSED [100%]
====================== 1 passed, 577 deselected in 2.78s ======================
```

What I removed: only the

```python
if float(m.group(1)) >= 1.0:
    pytest.skip(f"KNOWN src/train.py DEFECT ... Fix: src/train.py line ~836, default 0.9999.")
```

block. The surviving `assert float(m.group(1)) < 1.0` is byte-identical to
before, and the unrelated `if m is None: pytest.skip(...)` branch is untouched.
I also rewrote the docstring, which asserted the false thing "Reported as a skip
rather than a failure". No `assert` was added, removed, relaxed, or reordered.

## diff-stat

```
$ git diff --stat campaign/integration...HEAD
 src/train.py                | 16 ++++++++++++++--
 tests/test_config_system.py | 30 +++++++++++-------------------
 2 files changed, 25 insertions(+), 21 deletions(-)
```

Behaviour change in `src/train.py` is exactly one token: `1.0` → `0.9999` on the
`tau_end=` line (now line 851 after the comment). The other 14 added lines are
a comment.

## mutation-verdict

**Detector: `test_trainer_ema_fallback_is_not_the_frozen_value`.** Restoring
`1.0` makes it fail — as a failure, not a skip, which is the entire point:

```
$ & $PY tools\rt.py tests/test_config_system.py --slow -k "trainer_ema_fallback" -rs
tests\test_config_system.py F                                            [100%]
______ TestNoDeadKeys.test_trainer_ema_fallback_is_not_the_frozen_value _______
tests\test_config_system.py:972: in test_trainer_ema_fallback_is_not_the_frozen_value
    assert float(m.group(1)) < 1.0
E   assert 1.0 < 1.0
E    +  where 1.0 = float('1.0')
E    +    where '1.0' = <built-in method group of re.Match object ...>(1)
E    +      where <built-in method group of re.Match object ...> = <re.Match; span=(34656, 34689), match='model_cfg.get("ema_tau_end", 1.0)'>.group
====================== 1 failed, 577 deselected in 3.25s ======================
```

**Is it the right detector? Yes, with one honest caveat.**

Right: it pins the *cause* at its exact site. The freeze is not a diffuse
numerical accident, it is one literal at one call site, and this test reads that
literal. It is also the only guard that survives someone "fixing" the freeze by
deleting the schedule (the `m is None` skip would fire) or by having the trainer
compute tau some other way. Crucially, it is a **fail-closed** test where every
other EMA test in the tree is **fail-open**: `test_ema_tau_end_is_below_one` and
`test_every_config_supplies_the_ema_endpoint_...` both pass on a trainer whose
fallback is 1.0, because they inspect `defaults.yaml` and the 18 configs, which
are all correct. That is precisely how the defect survived — the config-side
tests were green and green was read as safe. This is the first test in the tree
that goes red on it.

Caveat, stated plainly: it is a **text/regex** test, not a behavioural one. It
inspects source text, so it would not catch a freeze introduced by editing
`EMATauSchedule.step()` or `update_target_encoder` instead. It also would not
fire on a *non-frozen but still wrong* fallback, e.g. `0.99` — it only asks
"is it below 1.0", not "does it equal the other three declarations". The real
behavioural coverage is the throwaway probe, which I did not commit because I
do not own the test files for `src/train.py`... except I do own
`tests/test_config_system.py`, so **this is the obvious follow-up and I left it
out only because the card scoped my test edit to "remove the skip".** Recommend a
follow-up card adding a behavioural test that calls `_build_optimization` with
`model_cfg={}` and asserts the two paths produce the same schedule — that closes
the `< 1.0 but still wrong` gap this detector leaves open. Flagging rather than
silently expanding scope.

## sibling EMA parameters at the same call site — audited, all clean

| key | trainer fallback | other declarations | verdict |
|---|---|---|---|
| `ema_tau_start` | `0.996` | `defaults.yaml:100` = 0.996, `jepa.py:71` = 0.996, `schedulers.py:75` = 0.996 | consistent, 4-way |
| `ema_tau_end` | `1.0` → **`0.9999`** | `defaults.yaml:101`, `jepa.py:72`, `schedulers.py:75` all 0.9999 | **was the sole outlier, now 4-way** |
| `ema_decay` (train.py:511) | `0.999` | `defaults.yaml:267` = 0.999 | consistent |
| `ema_end_decay` (:512) | `0.9999` | `defaults.yaml:268` = 0.9999 | consistent |
| `ema_anneal_end_step` (:513) | `100000` | `defaults.yaml:269` = 100000 | consistent |

No other sibling EMA parameter has a silently-wrong default.

Two **out-of-scope observations** for the integrator, not changed by me:

1. `tests/test_model.py:762` still constructs `EMATauSchedule(tau_start=0.996,
   tau_end=1.0, total_steps=200)` in its own test body. It is a test-local
   literal, not the trainer fallback, so it does not freeze anything real — but
   it now teaches the wrong constant to whoever reads it next, and it is the
   reason a grep for `tau_end=1.0` will still hit this repo. Owned by another
   card; I did not touch it.
2. `docs/decisions.md:156` records this fallback as "Open" and will need a
   line. Not my file.

## не_сделано / риски

- **Approaches used: 1 of 2.** I did not implement the `raise` variant, because
  it cannot satisfy the acceptance criterion (it converts the skip into a
  *different* skip — reasoning in full above). Limit not exceeded.
- **I did not add a behavioural test.** Deliberate: my test-file authority is
  "remove the skip", and the regex detector's real gap (`< 1.0` but still
  inconsistent) is documented above for a follow-up card.
- **I did not add a runtime guard** rejecting `tau_end >= 1.0` inside
  `_build_optimization`. `config.validate()` already covers the explicit case
  and runs earlier, and adding one would change other behaviour of a shared
  file. Also the risk: it would need to fire *after* `create_model`, so it is
  unreachable in the only case that matters (a missing key).
- **Risk — shared file.** `src/train.py` is a declared collision hotspot and
  another card is editing it this tick. My hunk is one contiguous comment+literal
  block inside `_build_optimization` (:832–852). A concurrent edit to the same
  function, or a whole-file reformat by another card, will need a manual merge.
  I staged by explicit path only; no `git add -A`, no force, no rebase, no push.
- **Risk — comment length.** 14 lines of comment on a one-token fix. I trimmed
  once already; the remainder is the "why `1.0` is fatal and not a placeholder"
  argument, which is the thing that was missing when this bug shipped. It
  matches the file's existing habit of documenting non-obvious behaviour
  (see the GradScaler note at :806 and the `_dataloader_worker_init` docstring).
  Easy to cut to one line if the integrator prefers.
