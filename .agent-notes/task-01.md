# TASK-01 — deterministic conftest fixture

- **status**: done
- **files**: `tests/conftest.py` (new, 130 lines) — the ONLY file created or modified.
  Branch `agent/task-01` off `campaign/integration`, worktree `C:\dev\wt-01`.
  The throwaway probe (`tests/test_zz_probe_tmp.py`, `probe_log.txt`) was **deleted**;
  `git status --short` is clean apart from the committed conftest.

## verify

### 1. ruff + black on the new file

```
PS> & $PY -m ruff check tests/conftest.py
All checks passed!
PS> & $PY -m black --check tests/conftest.py
All done! ✨ 🍰 ✨
1 file would be left unchanged.
```

### 2. The three named files pass

```
PS> & $PY tools\rt.py tests/test_sta.py tests/test_cgn.py tests/test_interp.py
rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_sta.py tests/test_cgn.py tests/test_interp.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 147 items

tests\test_sta.py ...........................                            [ 18%]
tests\test_cgn.py ..........................                             [ 36%]
tests\test_interp.py ................................................... [ 70%]
...........................................                              [100%]

============================= 147 passed in 8.02s =============================
```

### 3. Order independence — two orderings, both green, same result

Ordering A:
```
PS> & $PY tools\rt.py tests/test_sta.py tests/test_cgn.py tests/test_interp.py
collected 147 items
tests\test_sta.py ...........................                            [ 18%]
tests\test_cgn.py ..........................                             [ 36%]
tests\test_interp.py ................................................... [ 70%]
...........................................                              [100%]
============================= 147 passed in 8.02s =============================
AFTER-A sta,cgn,interp  wall=12.8s
```

Ordering B:
```
PS> & $PY tools\rt.py tests/test_interp.py tests/test_cgn.py tests/test_sta.py
collected 147 items
tests\test_interp.py ................................................... [ 34%]
...........................................                              [ 63%]
tests\test_cgn.py ..........................                             [ 81%]
tests\test_sta.py ...........................                            [100%]
============================= 147 passed in 8.12s =============================
AFTER-B interp,cgn,sta  wall=13.9s
```

Same file twice (ordering B repeated):
```
PS> & $PY tools\rt.py tests/test_interp.py tests/test_cgn.py tests/test_sta.py
tests\test_sta.py ...........................                            [100%]
============================= 147 passed in 8.12s =============================
AFTER-B interp,cgn,sta  wall=13.9s
--- second, identical invocation ---
tests\test_cgn.py ..........................                             [ 81%]
tests\test_sta.py ...........................                            [100%]
============================= 147 passed in 8.75s =============================
AFTER-B2 interp,cgn,sta (repeat)  wall=14.3s
```

147/147 in all three, identical. (Wall seconds drift because the box is shared and
someone is gaming; the pytest-reported seconds are the fair number.)

### 4. Proof the fixture is autouse and really seeds — distinct stream per test

Throwaway probe (deleted, not committed). It records the *global* RNG state at the
start of two tests that never ask for any fixture:

```
PS> & $PY tools\rt.py tests/test_zz_probe_tmp.py -s
collected 5 items

tests\test_zz_probe_tmp.py
PROBE-ALPHA {'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_alpha', 'torch_rng': 'e85e1504fa80', 'random': '0.4230211247345338', 'numpy': '0.8060229573478523', 'threads': 1}
PROBE-BETA  {'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_beta',  'torch_rng': '4a0c7a1223c7', 'random': '0.4451034254440219', 'numpy': '0.32431792233857093', 'threads': 1}
.....

============================== 5 passed in 0.10s =============================
```

* `test_probe_alpha` / `test_probe_beta` request **no** fixture and still arrive at a
  seeded, single-threaded state → `autouse=True` is real.
* `torch_rng` differs between the two tests (`e85e1504fa80` vs `4a0c7a1223c7`) → the
  seed is a function of the node id, not a constant.
* `threads: 1` → the thread pin is applied without being requested.
* A hard-assertion test in the probe recomputed `seed_for_node(nodeid)` and proved the
  live `random` / `numpy` / `torch` streams sit exactly at that seed; a separate test
  proved `seed_for_node` is distinct for `::test_one`, `::test_two`,
  `::test_one[3-7]`, stable across calls, and that `PYTEST_BASE_SEED=999` shifts the
  base while `PYTEST_BASE_SEED=not-a-number` aborts collection with a usage error.

### 5. The fixture removes a real order-dependence (probe alone vs probe after 94 other tests)

WITH the fixture — byte-identical streams regardless of what ran first:
```
PS> & $PY tools\rt.py tests/test_zz_probe_tmp.py
============================== 5 passed in 0.07s =============================
PS> & $PY tools\rt.py tests/test_interp.py tests/test_zz_probe_tmp.py
============================= 99 passed in 7.16s =============================
---- probe_log.txt AFTER FIXTURE (probe-alone then probe-after-interp) ----
{'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_alpha', 'torch_rng': 'e85e1504fa80', 'random': '0.4230211247345338', 'numpy': '0.8060229573478523', 'threads': 1}
{'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_beta',  'torch_rng': '4a0c7a1223c7', 'random': '0.4451034254440219', 'numpy': '0.32431792233857093', 'threads': 1}
{'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_alpha', 'torch_rng': 'e85e1504fa80', 'random': '0.4230211247345338', 'numpy': '0.8060229573478523', 'threads': 1}
{'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_beta',  'torch_rng': '4a0c7a1223c7', 'random': '0.4451034254440219', 'numpy': '0.32431792233857093', 'threads': 1}
```

### 6. Before / after wall time, interleaved (conftest physically moved in and out, verified each way)

```
round 1  WITHOUT conftest : ============================= 147 passed in 8.58s =============================  wall=11.4s
round 1  WITH    conftest : ============================= 147 passed in 7.43s =============================  wall=12.4s
round 2  WITHOUT conftest : ============================= 147 passed in 9.75s =============================  wall=11.9s
round 2  WITH    conftest : ============================= 147 passed in 6.98s =============================  wall=11.3s
```

Caveat, stated honestly: my first attempt at this comparison used `Rename-Item` with a
relative destination, which resolved to the wrong directory and **silently failed** —
so that round of numbers was invalid (all four runs actually had the conftest). I
re-ran it with absolute `Move-Item` paths plus a `Test-Path` abort guard in each
direction; only the block above counts. The first, untainted measurements at the very
start of the card, before `tests/conftest.py` existed, were `147 passed in 8.05s` and
`147 passed in 9.45s` — consistent with the "WITHOUT" numbers above. **No slowdown;
if anything marginally faster, which is within noise.**

### 7. Blast radius — every other non-slow test file, with the conftest in place

The suite has 29 test files; `tools/rt.py` refuses 4 of them without `--slow`
(`test_model`, `test_training_e2e`, `test_v025_integration`, `test_config_system`).
I ran all 25 permitted files:

```
PS> & $PY tools\rt.py tests/test_probes_split.py tests/test_causal_intervention.py tests/test_rdc.py tests/test_spc.py
============================= 117 passed in 8.85s =============================

PS> & $PY tools\rt.py tests/test_jawp.py tests/test_pcr.py tests/test_wsd.py tests/test_wsr.py tests/test_cmc.py tests/test_gac.py
====================== 186 passed, 7 warnings in 10.32s =======================

PS> & $PY tools\rt.py tests/test_index_and_cka.py tests/test_info_and_disentangle.py tests/test_sterility.py tests/test_swip.py tests/test_sigreg_jspace.py tests/test_mechanism_wiring.py tests/test_grad_scaler.py tests/test_torchio.py tests/test_train_device.py tests/test_distributed_helpers.py tests/test_training_state_guards.py
================= 294 passed, 1 xfailed, 3 warnings in 14.91s ==================

PS> & $PY tools\rt.py tests/test_checkpoint_fidelity.py
======================= 20 passed, 3 warnings in 10.05s =======================
```

**764 passed, 0 failed.** The 1 `xfail` is pre-existing
(`tests/test_training_state_guards.py:489`), as are the WSR `UserWarning`s about
`mode='gradient'` degeneracy — both are documented in `proofs/IMPLEMENTATION_STATUS.md`
and are untouched by this card. I did not skip, xfail, weaken or delete anything.

Interaction with existing fixtures, checked deliberately:
* `tests/test_causal_intervention.py:125` has its own autouse `seed_everything(0)`.
  Conftest-level autouse fixtures are instantiated first, so that file's own seed
  still wins inside it. Its 38 tests pass.
* `tests/test_probes_split.py:286` is the suite's only `scope="module"` fixture.
  It is *not* covered by a function-scoped autouse fixture. It is safe today because
  it passes an explicit `seed=`, and lines 270-275 / 391-396 of that file **assert**
  that `evaluate()` leaves the caller's global RNG untouched. Its 20 tests pass.
  Residual hole documented in the conftest docstring.

### diff-stat

```
$ git diff --stat campaign/integration...HEAD
 tests/conftest.py | 130 ++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 130 insertions(+)
```

## mutation-verdict

**Stated plainly, and it is the weaker of the two claims:**

> Removing the seed call from the fixture does **not** change any existing test.
> All 147 tests still pass, in both orderings:

```
--- MUTATION ON: ordering A (sta,cgn,interp) ---
============================= 147 passed in 7.63s =============================
--- MUTATION ON: ordering B (interp,cgn,sta) ---
tests\test_sta.py ...........................                            [100%]
============================= 147 passed in 7.49s =============================
```

So: **no existing assertion in the suite currently depends on the fixture.** By the
literal reading of the card's own criterion — "if you cannot demonstrate a test that
depends on the fixture, SAY SO" — I am saying so. As of this commit the fixture is
*decoration with respect to the pass/fail signal*; it changes zero outcomes today.

What it is **not** is decoration with respect to the property it exists to buy. I
measured that property directly. With the seed call removed, the same probe run alone
and run after `test_interp.py` gives:

```
==== probe_log.txt WITH SEED CALL REMOVED (probe-alone then probe-after-interp) ====
{'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_alpha', 'torch_rng': '1a72804e5241', 'random': '0.4230211247345338', 'numpy': '0.5428009695676712', 'threads': 1}
{'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_beta',  'torch_rng': '1a72804e5241', 'random': '0.4451034254440219', 'numpy': '0.6767905627642139', 'threads': 1}
{'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_alpha', 'torch_rng': 'ede4b43d0200', 'random': '0.4230211247345338', 'numpy': '0.736437196690155', 'threads': 1}
{'nodeid': 'tests/test_zz_probe_tmp.py::test_probe_beta',  'torch_rng': 'ede4b43d0200', 'random': '0.4451034254440219', 'numpy': '0.5625572844394293', 'threads': 1}
```

versus the WITH-fixture table in section 5. Two things flip:

1. **Per-test distinctness is lost.** Alpha and beta now show the *same* `torch_rng`
   (`1a72804e5241`, then `ede4b43d0200`) — they run from one shared stream. This is
   the exact "constant seed" failure the card warned about: reproducible, still
   order-dependent.
2. **The stream depends on what ran before.** Alone → `1a72804e5241`; after 94
   `test_interp.py` tests → `ede4b43d0200`. `numpy` moves too
   (`0.5428…` → `0.7364…`).

**So the precise verdict: removing the seed call changes the *property* "every test
starts from a stream determined solely by its own node id" — it provably reverts to
"the stream depends on execution order", because nothing in the fixture re-seeds
`torch`/`numpy` between tests. It changes *no test outcome*, because the suite's
assertions are all property-style bounds (loss non-negative, shapes, top-1 in
[0,1], Gram-matrix closeness) that are insensitive to which random draw they saw.**

That insensitivity is the actual finding, and it cuts both ways: the suite is
robustly non-flaky today, but it also means the suite cannot *detect* a
regression in determinism. Nothing would fail if someone deleted this file.

Corollary worth flagging to the campaign: the 4 `SLOW_FILES` I was not permitted to
run (`test_model.py` is 95 KB, `test_config_system.py` 48 KB) are exactly where a
seed-sensitive assertion is most likely to live, and CI is the only place they get
run. **The conftest's effect on those four is unverified by me.**

One honest oddity from the data: `python random`'s values (`0.4230211247345338` for
alpha) were *identical* with and without seeding. Nothing in `src/interp/` draws from
the `random` module — it uses `torch` and `numpy`. So of the three seeds, `torch` and
`numpy` are the load-bearing ones for the interp path.

## не_сделано

* **Did not enable `torch.use_deterministic_algorithms(True)`**, per instruction. I
  did not audit which specific ops in this repo lack a deterministic kernel, so I am
  **not** filing the "here are the offending ops" follow-up the card invited — I would
  be guessing, and a wrong list is worse than none. The concrete follow-up is:
  bisect by running with `torch.use_deterministic_algorithms(True)` plus
  `warn_only=True` in a scratch checkout and collecting the `torch.utils.deterministic`
  warnings; only then decide. Candidates I noticed by grep, **unverified**:
  `torch.linalg.svd` / `svdvals` in the Grassmann workspace code, and any
  `index_put_(accumulate=True)` / `scatter_add_` in the routing mechanisms.
* **Did not cover module-scoped fixtures.** A function-scoped autouse fixture cannot
  seed them, and the suite has exactly one (`future_clean` in
  `tests/test_probes_split.py`), which is currently RNG-safe by its own assertions. A
  module-scoped autouse conftest fixture would close the hole, but adding it for one
  already-safe call site is speculative work, and I could not confirm without testing
  that it would be ordered *before* `future_clean` rather than after. Documented in
  the conftest docstring instead.
* **Did not touch `src/utils/seed.py`**, although `AGENTS.md` says "Tests must be
  deterministic. Use `src/utils/seed.py`." I bypassed `seed_everything` on purpose and
  said why in the file docstring: its `deterministic=False` default sets
  `cudnn.benchmark = True`, an explicit opt-out of deterministic cuDNN algorithm
  selection, and `deterministic=True` sets cuDNN flags that are dead weight on this
  CPU-only box. Its `os.environ["PYTHONHASHSEED"]` write is also a no-op — the
  interpreter has already hashed its own strings by then. Wiring a box-dependent
  determinism switch into a fixture whose job is determinism is the wrong dependency.
  If the project prefers strict `AGENTS.md` conformance, the follow-up is to split
  `seed_everything` into a pure `seed_prngs()` and keep the cuDNN toggling in
  `src/train.py` — but that is a change to a file I do not own.
* **Did not run the 4 slow files** (`--slow` refused by policy) and **never ran
  training**; every run went through `tools/rt.py` with a test path, each invocation
  well under the 90 s budget (max observed 20 s).
* **Did not commit the probe** or `probe_log.txt`; both deleted, `git status` clean.

## риски

1. **Renaming or re-parametrising a test changes its seed.** Node-id-derived seeds
   mean a test that currently passes with seed X may fail with seed Y after a rename
   or a new `@pytest.mark.parametrize` case. That is the intended trade (it is what
   buys order independence) but it will surface as "I only renamed it" flakes. The
   `PYTEST_BASE_SEED` env var is the mitigation: re-run with a different base to
   tell a structural failure from a data-dependent one.
2. **Renaming a test file changes the seeds of all its tests**, since the node id
   contains the path. During a refactor-heavy campaign, expect flaky-looking
   failures from pure renames. Worth telling the other agents.
3. **The fixture cannot detect its own removal.** See the mutation verdict — nothing
   in the suite would fail if this file were deleted. A `tests/test_determinism.py`
   asserting that two differently-named tests see different RNG states, and that
   running a file after another file leaves the stream unchanged, would close that.
   I did not write it because I do not own any other file under `tests/`, and
   `tests/conftest.py` is not the right home for a test. **Recommend the campaign
   assign that to someone owning `tests/test_determinism.py`.**
4. **A test that legitimately wants a specific seed must reseed inside its body.**
   Several already do (`test_probes_split.py:270,391`, and
   `test_causal_intervention.py`'s autouse `seed_everything(0)`), and they pass. But
   the autouse fixture will now override a module-level `torch.manual_seed(...)` call
   at import time in any *new* test file. Whoever writes those needs to know.
5. **`torch.set_num_threads(1)` per test has a small but non-zero cost** and is
   called twice per test. Measured as noise (section 6), but on a file with hundreds
   of trivial tests it is a real per-test call into torch. If someone later wants to
   drop it, dropping the *restore* is safe; dropping the *set* silently re-enables
   6 threads on any machine where `OMP_NUM_THREADS` is unset (i.e. plain `pytest`,
   which is what CI runs) and that is a CPU-safety regression, not a speed one.
6. **Unverified against the 4 slow files** and against `pytest -n auto` /
   `xdist`, which I never exercised. With `xdist` each worker re-imports conftest and
   `seed_for_node` stays correct (it is a pure function of the node id, not of the
   worker), so it should be fine, but I did not run it and will not claim it.
