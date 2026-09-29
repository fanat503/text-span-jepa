# TASK-29 — config typo detector compares paths, not bare leaf names

**status:** done
**branch:** `agent/task-29` (worktree `C:\dev\wt-29`, off `agent/wave-1` @ `b91b273`)
**commit:** `bccd57b`
**approaches used:** 1 (no escalation needed)

---

## 1. Correction to the card's premise — read this first

The card and the brief both say the three controls in
`TestTrainerTypoDetectorGap` are **"RED today … they must go green."**
They were **GREEN today**, and that is the whole design of the control.

The original tests asserted the *bug*:

```python
def test_misplaced_key_is_invisible_to_the_trainer(self):
    misplaced = {"model": {"batch_size": 64}}
    assert not _trainer_would_warn(misplaced), (
        "src/train.py now detects misplaced keys by path; retire this control ..."
    )
```

and the class docstring said so outright:

> *"They are negative controls: they assert the gap still exists, so a future
> fix to `src/train.py` turns this file red and prompts the test to be
> rewritten rather than left stale."*

Baseline evidence, before any edit of mine:

```
$ & $PY tools\rt.py tests/test_config_system.py --slow -k "TrainerTypoDetectorGap or KeyPaths"
collected 578 items / 512 deselected / 66 selected
tests\test_config_system.py ............................................ [ 66%]
......................                                                   [100%]
===================== 66 passed, 512 deselected in 1.90s =====================
```

So the controls were green *because* they encoded the defect. "They must go
green" is satisfied in the sense the file itself defines: the fix turned
them red, and the rewrite they demanded ("retire this control") is what
made them green again **by pinning the fixed behaviour instead of the bug**.
I did not delete them, and I did not weaken them — every assertion was
strengthened from `not warned` to `warned == {exact path}`.

---

## 2. Files changed

| file | change |
|---|---|
| `src/train.py` | `_warn_unknown_config_keys` only (lines 951–1035). No other function touched. |
| `tests/test_config_system.py` | `TestTrainerTypoDetectorGap` rewritten; `_trainer_would_warn` now calls the real trainer; `_TRAINER_EXTRA_KNOWN` path-scoped; two stale docstrings corrected. |

`config/**` and `defaults.yaml` untouched. `pyproject.toml` untouched
(see §6 — that is why no pytest marker was introduced).

### The fix

```python
known_paths = _paths(known)          # was: {p.split(".")[-1] for p in _leaves(known)}
...
elif (
    p not in known_paths             # was: k not in known_leaf_names
    and p not in extra_known
    and k not in metadata_keys
    and not p.startswith(metadata_prefixes)
):
```

`extra_known` is now path-scoped, because a bare name there would re-open
the identical hole. The prefixes were read out of the code, not guessed:

* `model.{average_top_k_layers,loss_beta,loss_scale,ema_decay,ema_end_decay,ema_anneal_end_step,head_layers}`
  — `src/train.py:508-514`, all read from `model_cfg.get(...)`.
* `meta.dataset` — `defaults.yaml:50`.
* `data.allow_missing_validation` — `src/train.py:737`, `data_cfg.get(...)`.

`_meta.*` and `description` exemptions are unchanged and still work.

### The test rewrite

`_trainer_would_warn` no longer **mirrors** the trainer. It now calls
`src.train._warn_unknown_config_keys` and reads the paths back out of the
emitted warnings via `caplog`. This matters: a mirror pins the copy, and the
three cases this class exists to catch are exactly the ones where copy and
function disagree about what a "key" is. The mirror is what let the control
stay green while the defect was real. A `ContextVar` carries the `caplog`
handle; the fixture is a class-scoped `autouse`, so no pytest marker had to
be registered (see §6).

Three tests added, all of which the brief explicitly asked for:

* `test_a_correctly_pathed_key_is_not_warned_about` — the **false-positive
  guard**. Correctly-pathed keys in `model:` / `data:` / `optimization:` must
  stay silent. A stricter detector is not a better one.
* `test_meta_and_documented_exemptions_stay_exempt` — `_meta.*`,
  `description`, and `data.allow_missing_validation` stay silent; **and**
  `model.allow_missing_validation` is now *caught*, proving the exemptions are
  path-scoped and cannot be borrowed from the wrong subtree.
* `test_trainer_extra_known_matches_the_trainer` — reads the trainer's
  `extra_known` out of `src/train.py` with a regex and diffs it against this
  file's copy, and asserts every entry contains a `.`. The two lists can no
  longer drift, and the list cannot silently go back to bare names.

`_DEFAULTS_LEAF_NAMES` was deleted with a comment saying why: it existed only
to mirror the leaf-NAME comparison and would invite the mirror back.

---

## 3. Verify — FULL paste

```
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
```

**Card's command — `& $PY tools\rt.py tests/test_config_system.py --slow`:**

```
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_config_system.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 581 items

tests\test_config_system.py ............................................ [  7%]
........................................................................ [ 19%]
...........................................................s............ [ 32%]
........................................................................ [ 44%]
........................................................................ [ 57%]
........................................................................ [ 69%]
........ss..........ssssssss..........................................ss [ 81%]
..........ssssssss...................................................... [ 94%]
.................................                                        [100%]

======================= 560 passed, 21 skipped in 7.22s =======================
```

**No skips and no xfails were added.** The 21 skips are pre-existing and
untouched: `data2vec`/MLM arms of the EMA tests, and
`TestNoDeadKeys::test_ema_schedule_is_inert_in_the_trainer`. Test count went
578 → 581 (net +3: the three added guards; no test was removed or renamed
away).

**The class on its own:**

```
$ & $PY tools\rt.py tests/test_config_system.py --slow -k "TrainerTypoDetectorGap"
collected 581 items / 575 deselected / 6 selected
tests\test_config_system.py ......                                       [100%]
====================== 6 passed, 575 deselected in 0.65s ======================
```

**Adjacent files that exercise `src/train.py` / config wiring** (my change to
`src/train.py` is confined to a log-only function, but I checked rather than
assumed):

```
$ & $PY tools\rt.py tests/test_mechanism_wiring.py tests/test_train_device.py tests/test_grad_scaler.py
rt.py: threads=1  total_budget=90s  slow_ok=False
collected 24 items

tests/test_mechanism_wiring.py ......                                    [ 25%]
tests/test_train_device.py ...............                               [ 87%]
tests/test_grad_scaler.py ...                                            [100%]

============================== warnings summary ===============================
tests/test_mechanism_wiring.py::TestGACHook::test_stashed_slots_receive_grads
tests/test_mechanism_wiring.py::TestLiveWorkspaceViews::test_regularizers_give_workspace_grads
tests/test_mechanism_wiring.py::TestLiveWorkspaceViews::test_retraction_projects_grad_inplace_and_keeps_orthogonal
  C:\dev\wt-29\src\models\wsr.py:285: UserWarning: WSR mode='gradient': no dL/dQ is
  available ... Call wsr.set_lagged_gradient(workspace_Q.grad) after
  optimizer.step() and before zero_grad(). ...

======================== 24 passed, 3 warnings in 2.76s ========================
```

Those 3 WSR warnings are pre-existing (the WSR "orthonormality proxy"
divergence AGENTS.md documents) and are not mine.

**Lint / format:**

```
$ & $PY -m ruff check src/train.py tests/test_config_system.py
All checks passed!

$ & $PY -m black --check src/train.py tests/test_config_system.py
All done! ✨ 🍰 ✨
2 files would be left unchanged.
```

**No training was run.** `tools/rt.py` refuses `src.train` anyway.

---

## 4. Mutation verdict — revert to leaf-name matching

Three edits to `src/train.py` restored the old semantics:
`known_paths = {p.split(".")[-1] for p in _paths(known)}`, `extra_known`
back to 9 bare names, and the guard back to `k not in known_paths`.

**Run A — mutated (leaf-name matching):**

```
$ & $PY tools\rt.py tests/test_config_system.py --slow -k "TrainerTypoDetectorGap"
collected 581 items / 575 deselected / 6 selected
tests\test_config_system.py FF..FF                                       [100%]

================================== FAILURES ===================================
__________ TestTrainerTypoDetectorGap.test_misplaced_key_is_detected __________
tests\test_config_system.py:572: in test_misplaced_key_is_detected
    assert _trainer_would_warn(misplaced) == {"model.batch_size"}, (
E   AssertionError: the trainer cannot see a real key sitting in the wrong subtree.
E   `model.batch_size` is read by nobody, so the run trains at the defaults.yaml
E   micro-batch whatever the config says.
E   assert set() == {'model.batch_size'}

________ TestTrainerTypoDetectorGap.test_misspelled_section_is_detected _______
tests\test_config_system.py:581: in test_misspelled_section_is_detected
    assert _trainer_would_warn({"modle": {"embed_dim": 8}}) == {"modle.embed_dim"}
E   AssertionError: assert set() == {'modle.embed_dim'}

_ TestTrainerTypoDetectorGap.test_meta_and_documented_exemptions_stay_exempt __
tests\test_config_system.py:615: in test_meta_and_documented_exemptions_stay_exempt
    assert _trainer_would_warn({"model": {"allow_missing_validation": True}}) == {
E   AssertionError: assert set() == {'model.allow...g_validation'}

___ TestTrainerTypoDetectorGap.test_trainer_extra_known_matches_the_trainer ___
tests\test_config_system.py:630: in test_trainer_extra_known_matches_the_trainer
    assert all("." in k for k in found), (
E   AssertionError: src/train.py's extra_known went back to bare leaf names:
E   ['allow_missing_validation', 'average_top_k_layers', 'dataset',
E    'ema_anneal_end_step', 'ema_decay', 'ema_end_decay', 'head_layers',
E    'loss_beta', 'loss_scale']. A name matches in every section, which is the
E   hole this class pins shut.
E   assert False
=========================== short test summary info ===========================
FAILED tests/test_config_system.py::TestTrainerTypoDetectorGap::test_misplaced_key_is_detected
FAILED tests/test_config_system.py::TestTrainerTypoDetectorGap::test_misspelled_section_is_detected
FAILED tests/test_config_system.py::TestTrainerTypoDetectorGap::test_meta_and_documented_exemptions_stay_exempt
FAILED tests/test_config_system.py::TestTrainerTypoDetectorGap::test_trainer_extra_known_matches_the_trainer
================= 4 failed, 2 passed, 575 deselected in 0.59s =================
```

**Run B — restored (path matching), the same probe script:**

I also ran a read-only probe (`%TEMP%\opencode\probe29.py`, outside the repo,
not committed) that calls the trainer directly and prints the verdict per
case, so the three named non-detections are demonstrated rather than inferred.

```
$ & $PY "%TEMP%\opencode\probe29.py"        # MUTATED
ACCEPTED!  model.batch_size  (real key, wrong subtree)             -> warns NOTHING (silent)
ACCEPTED!  modle section    (misspelled section)                   -> warns NOTHING (silent)
ACCEPTED!  optimisation     (misspelled section)                   -> warns NOTHING (silent)
WARNING:root:Unknown config key 'model.lamda_swip' is not a path in defaults.yaml — ...
DETECTED   model.lamda_swip (nested typo, was already caught)      -> warns ['model.lamda_swip']

$ & $PY "%TEMP%\opencode\probe29.py"        # RESTORED
WARNING:root:Unknown config key 'model.batch_size' is not a path in defaults.yaml — ...
DETECTED   model.batch_size  (real key, wrong subtree)             -> warns ['model.batch_size']
WARNING:root:Unknown config key 'modle.embed_dim' is not a path in defaults.yaml — ...
DETECTED   modle section    (misspelled section)                   -> warns ['modle.embed_dim']
WARNING:root:Unknown config key 'optimisation.lr' is not a path in defaults.yaml — ...
DETECTED   optimisation     (misspelled section)                   -> warns ['optimisation.lr']
WARNING:root:Unknown config key 'model.lamda_swip' is not a path in defaults.yaml — ...
DETECTED   model.lamda_swip (nested typo, was already caught)      -> warns ['model.lamda_swip']
```

### The three cases that were silently accepted, named

1. **`model.batch_size`** — a real key name, but it lives at
   `data.batch_size`. Under `model:` nothing reads it, so the run trains at
   the defaults.yaml micro-batch regardless of the config.
2. **`modle: {embed_dim: 8}`** — a misspelled top-level section. Every leaf
   under it is a brand-new namespace that no reader ever touches.
3. **`optimisation: {lr: 0.01}`** — same, for the schedule.

**Honest qualification on "all three go red":** four of the six tests in the
class go red, covering all three named non-detections — but the *third
original control* (`test_nested_typo_inside_a_known_section_is_still_caught`,
`model.lamda_swip`) stays **green** under the mutation, and it must. The card
itself says a genuine nested typo *was* already caught; that test exists to
prove the stricter detector did not lose the one case the old one got right.
Reporting it as red would be false. The mutation is caught 4/6, not 3/3.

---

## 5. Diff-stat

```
 src/train.py                |  67 ++++++++++-----
 tests/test_config_system.py | 193 ++++++++++++++++++++++++++++++++------------
 2 files changed, 187 insertions(+), 73 deletions(-)
```

`src/train.py` is 4000+ lines and a repo-wide collision hotspot; the diff
touches one function, `951–1035`, and nothing else.

---

## 6. не_сделано / риски

**Not done, deliberately**

* **A misspelled section with no leaves is still silent.** `{"modle": {}}`,
  or `{"modle": {"_meta": {...}}}`, emits no warning, because `_walk` only
  tests leaves. Fixing it means validating *interior* nodes against
  `_DEFAULTS_NODES`, which widens the diff into a behaviour the card did not
  ask for and risks false positives on legitimate non-`defaults.yaml`
  sections. Recorded, not done. If you want it, it is a three-line addition
  plus a control.
* **`extra_known` was not pruned.** Eight of its nine entries now duplicate
  paths that already exist in `defaults.yaml` (e.g. `model.loss_beta`), so
  they are redundant. I left them, because removing them changes what the
  trainer accepts and `defaults.yaml` is outside my file budget. If
  `defaults.yaml` ever drops `model.ema_schedule`-style keys, the list is
  the safety net. Worth a separate decision.
* **The detector still only warns.** A warning is a log line, not a stop. A
  config with a misplaced key now says so loudly and then trains the wrong
  thing anyway. `TestKeyPaths` is the real gate; this is the early warning.
  I adjusted that test's failure message, which claimed the trainer "accepts"
  such keys — a claim that became false with this commit.
* **Whole suite not run.** `tools/rt.py` refuses it and the box is shared. I
  ran the card's file plus the three adjacent files that touch `src/train.py`
  and config wiring. `src/train.py`'s change is confined to a function that
  only emits log records, and `grep` across the repo found no other consumer
  of the old message text or of `_warn_unknown_config_keys`.
* **No pytest marker was used.** `pyproject.toml` is not in
  `files_allowed`, so registering one would have been out of bounds; an
  unregistered marker emits `PytestUnknownMarkWarning`. Used a class-scoped
  `autouse` fixture instead, which needs no registration.

**Risks**

* `_trainer_would_warn` now parses the warning text
  (`r"Unknown config key '([^']+)'"`). If someone rewords the message the
  helper silently returns `set()` and the class fails — loudly, not quietly,
  so it is a safe coupling, but it is a coupling. Mitigation: the message
  prefix is now also asserted by `test_trainer_extra_known_matches_the_trainer`
  in spirit; a reworded message is a deliberate edit someone will make.
* `caplog` with `logger=""` temporarily lowers the **root** logger level for
  the duration of one call. `src/train.py` uses `logging.getLogger()` (the
  root) and installs a `basicConfig` handler at import. pytest restores log
  state around every test, so nothing leaks between tests; this was
  considered before choosing it over a private logger.
* The stricter detector is *newly able to warn about configs that are
  committed today* if any shipped config has a path-clean-but-still-odd
  key. `TestKeyPaths` proves none does (578 leaf-path checks over 57
  configs, all green), and `test_a_correctly_pathed_key_is_not_warned_about`
  pins the negative direction. Residual risk: low, and it is a *warning*,
  not a failure — a false positive here cannot stop a run.
* A user who today relies on a key being accepted in the wrong subtree (an
  undeclared but working override) will start seeing a warning. That is the
  point of the card, but it is a user-visible change and is not silent.
