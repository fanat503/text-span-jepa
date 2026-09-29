# TASK-28 — pin the determinism fixture

- **status**: done
- **files**: `tests/test_determinism.py` (new, 15 tests). Nothing else created,
  modified, or deleted. `tests/conftest.py` is byte-identical to
  `agent/wave-1` — verified by SHA-256 before and after all three mutations
  (`1FCBD59D5F32A57529A7B85166625563E0F23CC5A90FE69B929E0A65115D4796`, matches the
  backup taken before the first mutation, `git diff HEAD` empty).
  Branch `agent/task-28` off `agent/wave-1` @ `b91b273`, worktree `C:\dev\wt-28`.
  No push, no upstream.

## verify

```
PS> & $PY tools\rt.py tests/test_determinism.py tests/test_sta.py
rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_determinism.py tests/test_sta.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 42 items

tests\test_determinism.py ...............                                [ 35%]
tests\test_sta.py ...........................                            [100%]

============================ 42 passed in 5.63s =============================
```

Repo-wide lint and format, as the gate runs them:

```
PS> & $PY -m ruff check . --output-format concise
All checks passed!
PS> & $PY -m black --check .
All done! 107 files would be left unchanged.
```

Stability / robustness, all through `rt.py`, all 42 green:

```
=== run 1 (repeat) ===                    42 passed in 7.97s
=== run 2 (repeat) ===                    42 passed in 8.32s
=== reversed file order (sta first) ===   42 passed in 9.07s
=== PYTEST_BASE_SEED=424242 ===           42 passed in 7.53s
=== + test_gac.py ===                     60 passed in 5.68s
=== + test_cmc.py + test_probes_split.py ===  95 passed in 11.50s
```

The last two matter because `tests/test_causal_intervention.py` has its own
autouse `seed_everything(0)` and `tests/test_probes_split.py` holds the suite's
only module-scoped fixture — both are places where this file could have collided.

## mutation-verdict

Three mutations, all reverted. **`tests/test_sta.py` (27 tests) stayed 100 % green
under every one of them** — that is the finding, not an aside.

### A (the card's mutation) — constant seed instead of a node-id-derived one

```diff
-    seed = seed_for_node(request.node.nodeid)
+    seed = DEFAULT_BASE_SEED  # MUTATION-28: a constant seed, not derived from the node id
```

```
tests\test_determinism.py .F.FFF.........                                [ 35%]
tests\test_sta.py ...........................                            [100%]
FAILED tests/test_determinism.py::TestFixtureIsLive::test_every_global_rng_sits_exactly_at_the_seed_for_this_node_id
FAILED tests/test_determinism.py::TestPerTestStream::test_observer_alpha_sees_the_stream_of_its_own_node_id
FAILED tests/test_determinism.py::TestPerTestStream::test_observer_beta_sees_a_different_stream_from_alpha
FAILED tests/test_determinism.py::TestPerTestStream::test_the_observed_streams_are_those_two_node_ids_predict
======================== 4 failed, 38 passed in 9.35s ========================
```

All four are the live-state assertions, and the diffs show the smoking gun: alpha
and beta both entered the test body at `('bf5f38c4b30…6e428f2e63b0', …)` — the
*same* stream — while their node ids predict `20c0f77830c…` and `fa513aaffc6…`.
That is the "reproducible but order-dependent" failure the conftest docstring
warns about, caught by name.

### B — builtin `hash()` instead of `blake2b`

```diff
-    digest = hashlib.blake2b(node_id.encode("utf-8"), digest_size=8).digest()
-    return (base + int.from_bytes(digest, "big")) % _SEED_MODULUS
+    digest = hash(node_id) & 0xFFFFFFFF  # MUTATION-28: builtin hash, not blake2b
+    return (base + int.from_bytes(digest.to_bytes(4, "big"), "big")) % _SEED_MODULUS
```

```
tests\test_determinism.py ........FFF....                                [ 35%]
tests\test_sta.py ...........................                            [100%]
FAILED tests/test_determinism.py::TestSeedDerivation::test_the_seed_is_recomputable_from_the_node_id_alone
FAILED tests/test_determinism.py::TestSeedDerivation::test_two_runs_in_separate_processes_agree
FAILED tests/test_determinism.py::TestSeedDerivation::test_the_seed_does_not_depend_on_python_hash_randomisation
======================== 3 failed, 39 passed in 9.63s ========================
```

The decisive one, with the seed it actually produced in each interpreter:

```
E   AssertionError: the seed changed with PYTHONHASHSEED. seed_for_node is using
E   the builtin hash() of the node id, which CPython randomises per process; every
E   test now gets a new stream on every run and no test would fail.
E   assert 1764128725 == 530568056
```

This is the mutation the verifier called "the most valuable property this file has
and nothing else defends it". Two real interpreters, `PYTHONHASHSEED=1` and
`=987654`, disagree on the seed by 3.5 billion — and **27 mechanism tests stay
green**. A whole-suite reproducibility regression, invisible to everything except
this file.

### C — delete `tests/conftest.py` outright

```
collected 27 items / 1 error
=================================== ERRORS ====================================
_________________ ERROR collecting tests/test_determinism.py __________________
ImportError while importing test module 'C:\dev\wt-28\tests\test_determinism.py'.
tests\test_determinism.py:63: in <module>
    import conftest as determinism
E   ModuleNotFoundError: No module named 'conftest'
========================== 1 error in 2.63s ===========================
```

Hard red, as required. Note the cost honestly: pytest **interrupts the entire
session** on a collection error, so `test_sta.py` did not run. A defect in this
file therefore takes the whole run down. That is the intended severity for
"someone deleted the determinism guard", and it is also the reason a typo in this
file is a red CI rather than a grey one. There is no gentler assertion of a
file's absence; the alternative — a lazy import plus 14 `NameError`s — is
strictly less diagnosable.

## diff-stat

```
$ git diff --stat HEAD
 tests/test_determinism.py | 396 ++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 396 insertions(+)
```

15 tests, 4 classes: `TestFixtureIsLive` (3), `TestPerTestStream` (5),
`TestSeedDerivation` (6), `TestDeletionIsDetected` (1).

Card's must-pin list, mapped to tests:

| must pin | test |
|---|---|
| autouse → two tests, different streams | `TestPerTestStream::test_observer_beta_sees_a_different_stream_from_alpha` (plus `TestFixtureIsLive::test_the_determinism_fixture_is_autouse`) |
| seed stable across runs ⇒ `hash()` must not be used | `TestSeedDerivation::test_the_seed_does_not_depend_on_python_hash_randomisation` |
| global RNG state at test start is a function of node id | `TestFixtureIsLive::test_every_global_rng_sits_exactly_at_the_seed_for_this_node_id` |
| two runs in separate processes agree | `TestSeedDerivation::test_two_runs_in_separate_processes_agree` |

## The honest caveat, recorded not hidden

Carried in the module docstring of `tests/test_determinism.py` (lines 19–35) so it
travels with the file rather than living only in this report:

> The suite is **robustly non-flaky** — no test in it has ever failed because of
> unseeded randomness. It is also **blind to its own determinism**. Every
> assertion in it is a property-style bound (loss non-negative, shape correct,
> top-1 in [0,1], Gram matrix close) and those are insensitive to *which* random
> draw they saw. The insensitivity that makes the suite reliable is the same
> insensitivity that would let the seeding be deleted, weakened to a constant, or
> switched to the builtin `hash()`, with a green run every time.

Mutation C is the direct evidence for the second half of that: the verifier's
measurement was 168 green tests with the file renamed away, and I reproduced a
hard red instead. Mutation A and B are the evidence that the guard is not merely
present but *load-bearing*.

## не_сделано

* **Did not add this file to `.agent-notes/gate.sh`.** It is not in
  `files_allowed`, and it is a shared file. Right now `tests/test_determinism.py`
  is therefore **not run by the campaign gate** — it only runs under bare `pytest`
  in GitHub Actions, which is the required CI check. *Coordinator action, one
  line:* add it to the "mechanism regression" stage at `gate.sh:75-79`, or a new
  stage. The verifier's tick-1 report already flagged the same omission for
  `test_interp.py`, `test_feature_composition.py` and `test_ablation_module.py`,
  so this is one of five files in the same hole.
* **Did not cover module-scoped fixtures.** The determinism fixture is
  function-scoped, so it cannot seed `future_clean` in `tests/test_probes_split.py`.
  That file asserts `evaluate()` leaves the global RNG untouched, so it is safe
  today. Confirming it here would need a change to `tests/conftest.py`, which the
  card forbids.
* **Did not verify against the 4 `SLOW_FILES`.** `tools/rt.py` refuses
  `test_model.py`, `test_config_system.py`, `test_training_e2e.py` and
  `test_v025_integration.py` without `--slow`, and the card's verify command does
  not use it. TASK-01 flagged the same gap: those four are the most likely home of
  a seed-sensitive assertion, and my file does not cover them. What I *can* say:
  my mutations A and B changed nothing in `test_sta.py`, which is evidence that
  the blind spot is general rather than specific to the files I ran.
* **Did not exercise `pytest -n auto` / `xdist`.** Never installed or run here.
  `seed_for_node` is a pure function of the node id, so it is worker-independent
  by construction, but that is reasoning, not a measurement, and I will not
  claim otherwise.
* **Did not test the fixture's error path** (the `except` branch that restores the
  thread count and re-raises). Seeding three primitives with an in-range integer
  does not raise, so reaching it needs a monkeypatched `torch.manual_seed`. Not
  worth the coupling; recorded as untested rather than asserted.
* **Never ran training.** Every invocation went through `rt.py` with an explicit
  test path. Longest single run 11.5 s against a 90 s cumulative budget.

## риски

1. **`-k` on a single test in this file can fail on purpose.**
   `test_observer_beta_…` needs `test_observer_alpha_…` to have run first (pytest
   runs a module in definition order). Selecting it alone fails with a message that
   says exactly that, rather than silently comparing nothing:

   ```
   E   AssertionError: test_observer_alpha must run for this comparison. Select the
   E   file, not a single test: pytest runs a module's tests in definition order.
   ```

   I chose a loud failure over a `skip` (skips are forbidden here) and over a
   silent no-op (a silent no-op is precisely the decoration this file exists to
   remove). Cost: someone debugging with `-k` hits a red test that is not a bug.

2. **`test_every_global_rng_sits_exactly_at_the_seed_for_this_node_id` asserts on
   live global state**, so it will go red if any *other* autouse fixture anywhere
   in the suite starts seeding after this one. That is a true positive, not a
   flake, but it couples this file to fixture ordering. The verifier flagged the
   same trade and I think it is the right one: proving autouse behaviour without
   touching pytest internals requires reading the live state.

3. **The autouse assert uses the fixture's *name*.** `request.fixturenames` is
   public API and stable, but renaming `deterministic_rng` turns two tests red. I
   judged that correct — "the suite has an autouse determinism fixture" is the
   property, and a rename is a deliberate act that should be noticed. The check
   deliberately avoids `pytest.fixture`'s internals, which differ between pytest
   7 (`_pytestfixturefunction`) and pytest 9 (a `FixtureFunctionDefinition`); this
   box runs pytest 9.1.1 and a version-coupled test would rot.

4. **Cost.** 15 tests, ~5–9 s wall, of which ~7.4 s is the two child interpreters
   (torch import dominates). A module-scoped fixture pays that once. Under
   `pytest -n auto` the children run per worker, so wall time does not divide.
   Cheap for a suite that takes ~110 s.

5. **`test_the_seed_is_recomputable_from_the_node_id_alone` pins the derivation
   formula** (`base + blake2b(node_id) mod 2**32`), not just its stability. This
   is the one place I deliberately chose a tight assertion over a property. The
   reason: swapping blake2b for another *stable* digest would keep every other
   test green while silently re-seeding all ~1500 tests in the suite — a real
   hazard this campaign has already been bitten by (TASK-01 risk 2, a rename
   re-seeds a whole file). The failure message tells the next author to update the
   docstring in the same commit. If the coordinator prefers a looser assert, this
   is the one to relax; the `PYTHONHASHSEED` test already covers the failure the
   card actually cares about.

6. **`_OBSERVED` is module-level global state.** It is written by two tests and
   read by two, entirely within this module, and each reader re-derives its
   expectation from the node id rather than trusting the writer. It cannot leak
   into other files. But it does mean the file is not safe to run twice in one
   session (`pytest tests/test_determinism.py tests/test_determinism.py`); pytest
   refuses that on duplicate paths anyway.
