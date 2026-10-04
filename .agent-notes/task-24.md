# TASK-24 — thread count and `deterministic`: made reachable

**status: done.** Branch `agent/task-24`, worktree `C:\dev\wt-24`, base `agent/wave-2` @ `d34a345`.
One commit: `ff8cab2`. Nothing pushed. `src/train.py` untouched. No training run.

---

## 1. What the card asked, and what I concluded

> goal: make thread count and `deterministic` reachable, or state that they are not.

I did **both halves**, and the split is not a dodge:

| route | before | after |
|---|---|---|
| Python API | `seed_everything(seed, deterministic=False)` — always called with one arg, so `deterministic` was dead | `seed_everything(seed, deterministic=None, num_threads=None)`; the flag now **does something on CPU** |
| environment | did not exist | `TEXT_SPAN_JEPA_DETERMINISTIC`, `TEXT_SPAN_JEPA_NUM_THREADS` — **work today with no call-site change**, because `src/train.py:1148` calls `seed_everything(seed)` and the seeder now resolves the env itself |
| config key | did not exist, and nothing read one | **still does not exist, and that is a decision, not an omission** — see §5 |

The headline finding stands and I reproduced it independently: **on CPU the thread count is the axis that moves the numbers, and the old `deterministic` flag never touched it.**

---

## 2. The thread-count measurement

Card required an ad-hoc 2-thread-vs-1-thread measurement, so I did one, and I
guarded the stale-clone hazard the brief warned about.

### 2.1 The hazard was real, and the guard is in the assertion path

```
$ & $PY -c "import src; print(src.__file__)"
SRC= C:\dev\wt-24\src\__init__.py          # cwd is sys.path[0], so cwd wins

# ...but the installed package maps elsewhere:
$ Select-String __editable___text_span_jepa_1_0_0rc15_finder.py -Pattern MAPPING
MAPPING: dict[str, str] = {'src': 'C:\\Users\\...\\tmp\\clone-check-1\\src'}
```

`src` is served by a setuptools `meta_path` finder pointing at
`C:\Users\<user>\tmp\clone-check-1\src`. It wins whenever the running script's
own directory has no `src` — i.e. for any `python somescript.py` outside the
repo root. My harness therefore forces `sys.path.insert(0, worktree)` **and**
asserts `src.__file__` is under the worktree before emitting any number, and
every subprocess in `tests/test_seed.py` does the same
(`test_both_children_imported_the_tree_under_test`).

### 2.2 Measurement — 1 thread vs 2 threads, 10 steps

Real `TextSpanJEPA` (xsmall arm: `embed_dim=384`, `depth=6`, all 11 GWP
mechanisms off, exactly what `config/scaling/xsmall_30m.yaml` describes), one
fixed batch from `SpanMaskCollator`, `AdamW`, `update_target_encoder(0.99)`,
`seed=42`, `deterministic=False`. Only the thread count differs.

```
=== t1.json (threads=1) vs t2.json (threads=2) ===
src_file A: C:\dev\wt-24\src\__init__.py
src_file B: C:\dev\wt-24\src\__init__.py
  step  0: A=1.9095903635025024  B=1.9095903635025024  |d|=0.000e+00
  step  1: A=1.7885462045669556  B=1.7885462045669556  |d|=0.000e+00
  step  2: A=1.7147507667541504  B=1.7147507667541504  |d|=0.000e+00
  step  3: A=1.6500935554504395  B=1.65009343624115     |d|=1.192e-07
  step  4: A=1.5731045007705688  B=1.5731043815612793   |d|=1.192e-07
  step  5: A=1.4803268909454346  B=1.4803268909454346   |d|=0.000e+00
  step  6: A=1.3733474016189575  B=1.373347520828247    |d|=1.192e-07
  step  7: A=1.2617430686950684  B=1.2617430686950684   |d|=0.000e+00
  step  8: A=1.1644177436828613  B=1.1644177436828613   |d|=0.000e+00
  step  9: A=1.1106089353561401  B=1.1106091737747192   |d|=2.384e-07
  final qkv sha256: A=9ac782b88e283fbe...  B=de2f7c871b386300...  bitwise_equal=False
  first step whose loss differs: 3
```

This reproduces the card's R7 finding exactly (steps 0-2 identical, step 3
onward differs, `d` in the 1.19e-07 … 3.58e-07 band, `qkv` not bitwise equal) at
1-vs-2 rather than 1-vs-8.

**Control — same thread count twice, in two independent processes:**

```
=== t1.json (threads=1) vs t1b.json (threads=1) ===
  ... all ten steps |d|=0.000e+00
  final qkv sha256: A=9ac782b88e283fbe...  B=9ac782b88e283fbe...  bitwise_equal=True
```

So the runs are reproducible; the thread count is the variable.

**Where the divergence comes from — measured, not assumed.** It is *not* the
GEMMs. At these sizes `torch` gives bitwise-identical results for
`mm(512)`, `mm(1024)`, `mm(2048)` and a 3-D contraction at 1/2/3/4 threads. It
is the **reductions**:

```
sum_65536      1v2_differ=False  all4_equal=True
sum_1048576    1v2_differ=False  all4_equal=False
sum_4194304    1v2_differ=True   all4_equal=False
sum_16777216   1v2_differ=True   all4_equal=False
mm_512/1024/2048, mm_chain        all4_equal=True
```

Float addition is not associative, and `TensorIterator` splits a large `sum`
across the intra-op pool. That is exactly the shape of the model path: `sum`,
`mean` and `layer_norm` over `(B·T, D)` activations.

### 2.3 `deterministic=True` did **nothing**, measured

Same harness, `--deterministic` on the **pre-fix** code:

```
=== d1.json vs d2.json (deterministic=True, 1 vs 2 threads) ===
  step  3: A=1.6500935554504395  B=1.65009343624115     |d|=1.192e-07
  step  9: A=1.1106089353561401  B=1.1106091737747192   |d|=2.384e-07
  final qkv sha256: A=9ac782b8...  B=de2f7c87...  bitwise_equal=False
```

Byte-for-byte identical to the `deterministic=False` pair. Because the old body
set only `cudnn.deterministic` / `cudnn.benchmark`, which are CUDA convolution
switches and are never read on this CPU-only host.

### 2.4 After the fix, the same two launches agree bitwise

Launched with `OMP_NUM_THREADS=2`, `deterministic=True`:

```
threads=1 deterministic=True torch.get_num_threads()=1 src=C:\dev\wt-24\src\__init__.py qkv_sha=9ac782b88e283fbe
threads=2 deterministic=True torch.get_num_threads()=1 src=C:\dev\wt-24\src\__init__.py qkv_sha=9ac782b88e283fbe

=== n_d1.json (threads=1) vs n_d2.json (threads=2) ===
  ... all ten steps |d|=0.000e+00
  final qkv sha256: A=9ac782b88e283fbe...  B=9ac782b88e283fbe...  bitwise_equal=True
  first step whose loss differs: None
```

`torch.get_num_threads()` reports **1 in both**, i.e. the pin overrode the
launcher's `OMP_NUM_THREADS=2`, and the sha matches the earlier plain 1-thread
run — so asking for determinism removed the freedom without changing the
arithmetic. `use_deterministic_algorithms(warn_only=True)` selected no
different kernel on this path.

`tests/test_seed.py::TestEndToEndThreadCount` runs the same experiment on a
smaller model and prints the measured pair:

```
[test_seed.py] non-deterministic pair: 1 thread sha=3a8fe1985ba9240b 2 threads sha=bce4e394fb38e177 differ=True
```

---

## 3. The `deterministic` finding

`seed_everything(seed, deterministic=False)` was called with the default from
everywhere in the repo (`src/train.py:1148`, plus `tests/test_ablation_module.py`,
`test_causal_intervention.py`, `test_checkpoint_fidelity.py`,
`test_distributed_helpers.py`, `test_sterility.py`, `test_training_e2e.py`,
`test_model.py`). It then did:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

— a CUDA convolution-algorithm switch, and on this box not one line of it is
ever read. `deterministic=True` was a **guarantee of nothing**, and no config
key or CLI flag could ask for it (`src/train.py`'s parser has exactly
`--fname`, `--output_dir`, `--no_defaults`).

What it does now, all four axes, on both branches so it cannot be left half
applied:

| axis | `deterministic=True` | `deterministic=False` |
|---|---|---|
| intra-op threads | pinned to 1 (or to an explicit `num_threads`) | untouched |
| `torch.use_deterministic_algorithms` | `True`, `warn_only=True` | `False` |
| `cudnn.deterministic` / `benchmark` | `True` / `False` | `False` / `True` |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` via `setdefault` | untouched |

`warn_only=True` is **load-bearing, not lazy**: strict mode raises `RuntimeError`
for any op with no deterministic kernel, and this repo is full of them
(`torch.linalg.svd`, `index_put_`, `scatter`, the Grassmann workspace ops — as
`tests/conftest.py` already documents). Strict mode would make
`deterministic=True` crash on the first batch, so nobody would ever request it
and it could never be tested. There is a test whose only job is to keep
`warn_only` on.

`CUBLAS_WORKSPACE_CONFIG` is set with `setdefault` — cuBLAS reads it at handle
creation, so a launcher that set it first owns the value, and we never clobber
it. `OMP_NUM_THREADS`/`MKL_NUM_THREADS` are deliberately **not** set from
inside the process: they are read at process start, so writing them after torch
imported would advertise a setting that is not in effect. `torch.set_num_threads`
is the control I verified.

**Reachability.** `TEXT_SPAN_JEPA_DETERMINISTIC=1 python -m src.train --fname …`
works today. The default stays **off**: threading determinism on by default would
tax every run by the core count, and a reproducibility feature nobody pays for
is a feature nobody turns on. `resolve_num_threads()` returns `None` (= leave
torch alone) when nothing asks.

**A bad value is an error, not a default.** `TEXT_SPAN_JEPA_DETERMINISTIC=treu`
and `TEXT_SPAN_JEPA_NUM_THREADS=0` raise `ValueError`. Swallowing them is how a
reproducibility knob becomes decoration: the one run that was supposed to be
reproducible is the one nobody noticed.

---

## 4. `reproducibility_state()`

A checkpoint carries weights, not a thread count, so after the fact nobody can
tell what a run was pinned to. `reproducibility_state()` reports what is
**actually in force**, not what was requested — including
`interop_pinned`, because `torch.set_num_interop_threads` can only be called
once per process and a second call raises, so the honest answer is "could not".

---

## 5. The config path: stated CLOSED, not faked — `src/train.py` change I recommend and did **not** make

**I did not add `meta.deterministic` / `meta.num_threads` to `defaults.yaml`,**
even though the card lists the file as mine. Reasoning, written into
`defaults.yaml` at `meta:` and cross-referenced from the file header:

Declaring a key that `src/train.py` never reads makes `_warn_unknown_config_keys`
stop warning about it. Today a user who writes `meta.deterministic: true` gets a
startup *"possible typo"* line and knows it did nothing. After declaring it they
get **silence and no determinism** — the silent-no-op failure mode, which is the
thing this repo treats as worse than a missing key (TASK-17's whole card is about
an ablation that is a silent no-op while the run is still labelled as that
ablation). A dead key is not a cheaper defect than a typo warning; it is the more
expensive one.

### The recommended call-site change (needs `src/train.py`'s owner)

At **`src/train.py:1148`**, replace

```python
    seed_everything(seed)
```

with

```python
    seed_everything(
        seed,
        deterministic=args.get("meta", {}).get("deterministic"),
        num_threads=args.get("meta", {}).get("num_threads"),
    )
```

and, **in the same commit**, declare in `defaults.yaml`:

```yaml
meta:
  deterministic: false      # false | true; null/absent == off
  num_threads: null          # null == leave torch's own count alone
```

`null` costs nothing by default, so the change is inert until a run opts in.

And the third line, which the card's scope makes impossible for me: log
`reproducibility_state()` next to the seed. `src/train.py:1148` is immediately
followed by `_warn_unknown_config_keys(args)`; a `logger.info("Reproducibility:
%s", reproducibility_state())` there is the difference between "this run is
reproducible" and "this run is reproducible and here is what it pinned".

**Tripwire.** `tests/test_seed.py::TestConfigPathIsClosed` goes **red** the
moment either half is done without the other, and its message says exactly what
to do. So this cannot rot: see the second mutation in §8.

---

## 6. Published numbers — the card's caution, answered concretely

### 6.1 Which numbers exist in the repo

I enumerated rather than guessed. There is **no trained checkpoint, no
`train_log.csv`, no metrics artefact anywhere in the tree** (`git ls-files` for
`*.pt/*.pth/*.tar/*.csv/*.json/*.log/*.npz` returns only
`.agent-notes/baseline-lint.txt`, `protection.json`, `requirements.txt`;
`Get-ChildItem -Recurse` for checkpoints and `train_log.csv` returns nothing;
`Test-Path output` is `False`). So:

| where | numbers | thread provenance |
|---|---|---|
| `README.md` | **no metrics at all** — only a mechanism count and a "~100M params" comment | n/a |
| `proofs/HYPOTHESES.md` | `?2%`, `params^0.35 + 0.1`, `?1%` — **predictions, not measurements** (HYPOTHESES.md:23,130,151,160) | n/a |
| `proofs/IMPLEMENTATION_STATUS.md`, `proofs/README.md` | claim counts | n/a |
| `config/scaling/*.yaml` | parameter counts (62,509,249 / 170,706,561 / 262,021,633 / 537,990,145) | **thread-independent** — integer `p.numel()` sums; my re-merge of all 62 configs confirmed they are unaffected |
| `docs/plans/…wave1-audit-findings.md` (Part 2) | `r1=1.2265701293945312`, `r2=…`; resume deltas `1.305e-01`/`2.260e-01` | **line 201 states "6 threads"** — these are 6-thread numbers and only reproduce at 6 threads |
| `docs/plans/…wave1-audit-findings-perf.md` | `1.60s → 4.88s`, FLOP ratios 0.58/0.56/0.52 | **line 85 states "4 threads"**; timings are inherently count-dependent |
| `docs/plans/…wave1-audit-findings-interp.md` | `0.677 / 0.632 / 0.759`, `0.489`, `0.884 vs 0.907`, 357 s, 97.5 min | thread count not recorded; thread-induced error is ~1e-7 against effects of 0.10–0.24 |
| `docs/decisions.md`, `plan.md` | 1.498x / 1.281x parameter ratios | integer counts, thread-independent |

**So the only thread-sensitive published numbers are the audit documents'
own measurements, and the two that matter both state their thread count
explicitly** (6 threads, 4 threads). That is the honest summary: they are
reproducible *at the count they name*, and the audit authors were careful to
name it. What was missing is any way to *require* that count, which is what this
card supplies.

**Does the 1e-7 perturbation invalidate any of them? No.** The interp numbers
are quoted to 3 decimals against noise bands of 0.10–0.24 — six orders of
magnitude above a thread-induced perturbation. The perf numbers are timings and
were taken at a stated 4 threads. The parameter counts are integers. The
reproducibility audit's `r1`/`r2` are the one place where the perturbation
would matter, and they are *about* reproducibility, measured at a stated 6
threads, so they are self-consistent.

### 6.2 CI (2-4 cores) vs local (6 cores): real, and now localised

**The card's framing is right, and I localised it.** `.github/workflows/ci.yaml`
runs `python -m pytest tests/ -v --tb=short` with no thread pinning at all; the
campaign runner (`tools/rt.py`) sets `OMP_NUM_THREADS=1` and friends. Inside a
**test body** both are 1 thread, because `tests/conftest.py`'s autouse fixture
(TASK-01) calls `torch.set_num_threads(1)`. But **module-scoped fixtures are
set up before the function-scoped autouse fixture**, so they see the
environment default. Measured with a throwaway probe (run, then deleted):

```
### via tools/rt.py (OMP_NUM_THREADS=1 in the env)
SCRATCH: {'module_fixture': 1, 'test_body': 1}

### bare pytest, no OMP env -- what .github/workflows/ci.yaml actually does
SCRATCH: {'module_fixture': 2, 'test_body': 1}
```

So the CI-vs-local thread difference survives in exactly the module-scoped
fixtures. There are three (plus mine, which sets the count explicitly per child
and is unaffected):

1. `tests/test_baseline_parity.py:68 arms` — builds a ~1.1 GB model pair.
   Parameter init is elementwise (`kaiming_uniform_`/`normal_`, no reduction),
   and the tests compare `p.numel()` counts. **Not numerically at risk.** (It is
   a memory/time risk on a 2-core runner, which is a different card.)
2. `tests/test_probes_split.py:286 future_clean` — trains a 300-step probe, the
   only module-scoped fixture that trains anything. It runs on a 32×6
   `_FakeModel` probe and every assertion is **structural**
   (`0.0 <= top1 <= 1.0`, `top5 >= top1`, key-presence), not exact. **Cannot go
   red from a thread difference.**
3. `tests/test_determinism.py:136 two_processes` — subprocess hash-seed
   independence. Not thread-sensitive.

**Conclusion: CI and local results are not bitwise comparable inside those three
fixtures, and that cannot change any assertion in the current suite.** It could
become a problem the moment someone adds an exact-value assertion on top of a
module-scoped fixture — which is a `tests/conftest.py` fix (pin threads at
session scope), owned by TASK-01, **not mine**. Reported, not touched.

**The bigger CI-vs-local reason is not threads at all:** CI is `ubuntu-latest`
on Python 3.11 with the PyPI CPU wheel; local is Windows on Python 3.10 with a
different BLAS. Bitwise equality across that boundary was never on offer, and
nothing in this card changes it. Pinning the thread count closes the axis this
card owns; it does not make two platforms agree.

---

## 7. `defaults.yaml` and the 62-config blast radius

The brief flagged `defaults.yaml` as a collision hotspot (a sibling measured 18
of 62 configs previously unrunnable). What I added: **comments only, zero new
keys.** Verified independently of the test suite:

```
defaults.yaml parses OK. meta keys: ['dataset', 'load_checkpoint', 'model_name', 'read_checkpoint', 'seed', 'use_bfloat16']
62 configs merged over the new defaults.yaml, 0 problems
```

`meta` has exactly the six keys it had before, so `config/scaling/*.yaml` gain
nothing from the merge and cannot be destabilised. `tests/test_config_system.py`
is unchanged and green, including `TestDeltaPurity` (the exemption budget is
still exactly `12 + 5 = 17` leaves) and `TestScalingLadder`.

The `config/scaling/*.yaml` changes are **17 comment lines each, zero keys**, so
`TestDeltaPurity` cannot see them. They state the reproducibility status of the
capacity ladder next to the parameter counts, because that is where a reader of
a paper appendix looks.

---

## 8. Verify — full paste

### The card's command

```
$ $PY tools\rt.py tests/test_torchio.py tests/test_config_system.py --slow
rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_torchio.py tests/test_config_system.py
rt.py: threads=1  total_budget=300s  slow_ok=True
============================= test session starts =============================
collected 591 items

tests\test_torchio.py ..........                                         [  1%]
tests\test_config_system.py ............................................ [  9%]
........................................................................ [ 21%]
...................................s............                         [ 33%]
........................................................................ [ 45%]
........................................................................ [ 57%]
........................................................................ [ 70%]
........ss..........ssssssss..........................................ss [ 82%]
..........ssssssss...................................................... [ 94%]
.................................                                        [100%]

======================= 570 passed, 21 skipped in 6.20s =======================
```

**Identical to the pre-change baseline I took first** (`570 passed, 21 skipped
in 6.66s`, 591 collected). No test changed status; no skip added or removed.

### The new file

```
$ $PY tools\rt.py tests/test_seed.py
rt.py: threads=1  total_budget=300s  slow_ok=False
============================= test session starts =============================
collected 37 items

tests\test_seed.py .....................................                 [100%]

============================= 37 passed in 16.25s =============================
```

### Every other file that calls `seed_everything`

```
$ $PY tools\rt.py tests/test_determinism.py tests/test_cmc_resume.py tests/test_sterility.py
collected 59 items
tests\test_determinism.py ...............                                [ 25%]
tests\test_cmc_resume.py ............                                    [ 45%]
tests\test_sterility.py ................................                 [100%]
============================= 59 passed in 17.15s ==============================

$ $PY tools\rt.py tests/test_ablation_module.py tests/test_distributed_helpers.py tests/test_causal_intervention.py
collected 100 items
tests\test_ablation_module.py ......................                     [ 22%]
tests\test_distributed_helpers.py ................................       [ 54%]
tests\test_causal_intervention.py ...................................... [ 92%]
........                                                                 [100%]
============================= 100 passed in 4.06s ==============================

$ $PY tools\rt.py tests/test_checkpoint_fidelity.py tests/test_training_e2e.py --slow
================== 23 passed, 6 warnings in 68.95s (0:01:08) ===================

$ $PY tools\rt.py tests/test_model.py -k "seed_everything or seed or thread" --slow
collected 151 items / 149 deselected / 2 selected
tests\test_model.py ..                                                   [100%]
====================== 2 passed, 149 deselected in 2.11s =======================
```

### Lint and format (both were clean before; both are mine to keep clean)

```
$ $PY -m ruff check .
All checks passed!

$ $PY -m black --check .
All done!
110 files would be left unchanged.
```

`ruff --unsafe-fixes` did not silently rewrite semantics: it converted
`typing.Optional[X]`/`Dict` to `X | None`/`dict` (legal here because
`from __future__ import annotations` is at the top, exactly as `AGENTS.md`
prescribes), added `check=False` to the `subprocess.run`, and used
`.values()`. `test_model.py`, `test_config_system.py` and `test_determinism.py`
were **not** modified.

---

## 9. Mutation verdict

Two mutations, both reverted, both pasted.

### Mutation 1 — revert the deterministic body to cudnn-flags-only

```python
    # MUTATION (TASK-24): reverted to the pre-card body. cudnn flags only --
    # which are no-ops on a CPU-only host -- and no thread pin, no ATen
    # algorithm switch, no cuBLAS workspace.
    torch.backends.cudnn.deterministic = want_deterministic
    torch.backends.cudnn.benchmark = not want_deterministic
    return seed
```

**7 tests go red:**

```
$ $PY tools\rt.py tests/test_seed.py
collected 37 items
tests\test_seed.py FFFF...F............F.............F..                 [100%]

______ TestDeterministicIsNotANoOp.test_it_pins_the_intra_op_thread_pool ______
    assert torch.get_num_threads() == 1, (
E   AssertionError: seed_everything(deterministic=True) left torch on 4 threads. Two runs that differ only in their thread count differ from optimizer step 3 onward, so a 'deterministic' run that is still multi-threaded is not one.
E   assert 4 == 1
______ TestDeterministicIsNotANoOp.test_it_enables_deterministic_algorithms _____
    assert torch.are_deterministic_algorithms_enabled() is True
E   assert False is True
______ TestDeterministicIsNotANoOp.test_warn_only_is_kept_on ____________
E   AssertionError: warn_only was turned off. On a repo with non-deterministic ops this makes deterministic=True raise RuntimeError instead of running.
E   assert False is True
______ TestDeterministicIsNotANoOp.test_it_sets_the_cublas_workspace_config _____
E   AssertionError: CUBLAS_WORKSPACE_CONFIG is unset, so cuBLAS may split a GEMM differently at a different thread count
E   assert None
_ TestEnvironmentRoute.test_deterministic_env_reaches_the_existing_trainer_call_site _
    assert torch.get_num_threads() == 1
E   assert 4 == 1
_ TestEnvironmentRoute.test_deterministic_honours_an_explicit_thread_count ____
    assert torch.get_num_threads() == 2
E   assert 1 == 2
_ TestEndToEndThreadCount.test_determinism_makes_two_thread_counts_agree_bitwise _
E   AssertionError: determinism did not pin the pool: the children launched with 1 and 2 threads ran on 1 and 2 threads
E   assert (1 == 1 and 2 == 1)

======================== 7 failed, 30 passed in 19.59s =========================
```

Note the last one: the **end-to-end subprocess test** goes red too, so the
mutation is caught on the real model and not only on the flag. Restored →
`37 passed in 17.00s`.

### Mutation 2 — declare the two keys in `defaults.yaml` without wiring `src/train.py`

Added `deterministic: false` / `num_threads: null` under `meta:`:

```
$ $PY tools\rt.py tests/test_seed.py tests/test_config_system.py --slow -k "seed or ConfigPathIsClosed or EveryConfigRuns or KeyPaths or NoDeadKeys or DeltaPurity"
collected 618 items / 319 deselected / 299 selected
...
________ TestConfigPathIsClosed.test_they_are_not_declared_in_defaults ________
    assert key not in meta, (
E   AssertionError: defaults.yaml now declares meta.deterministic, but src/train.py does not read it. The trainer's own startup warning has just been silenced for a key that does nothing, which turns a loud 'this did not work' into a silent no-op. Wire src/train.py in the SAME commit, or remove the key -- see the meta: block of defaults.yaml.
E   assert 'deterministic' not in {'seed': 42, 'dataset': 'wikitext103', 'use_bfloat16': True, 'load_checkpoint': False, ...}
=================== 1 failed, 298 passed, 319 deselected in 17.81s ===================
```

The other 298 config tests stayed green — which is the point: **no existing test
would have caught a dead key added to `defaults.yaml`.** The tripwire is the
only thing standing between a future editor and that silent no-op. Restored →
`37 passed in 17.77s`, `ruff`/`black` clean, `git status` back to the intended
7 files.

---

## 10. diff-stat

```
$ git show --stat --oneline HEAD
 config/scaling/base_140m.yaml  |  17 ++
 config/scaling/large_300m.yaml |  17 ++
 config/scaling/small_100m.yaml |  17 ++
 config/scaling/xsmall_30m.yaml |  17 ++
 defaults.yaml                  |  58 +++++
 src/utils/seed.py              | 274 +++++++++++++++++++++++++++++++++++-
 tests/test_seed.py             | 547 +++++++++++++++++++++++++++++++++++++
 7 files changed, 935 insertions(+), 12 deletions(-)
```

Commit `ff8cab2` on `agent/task-24`. Not pushed. `src/train.py` and `.github/`
untouched.

---

## 11. не_сделано / риски

**не сделано (deliberately, with reasons)**

1. **`src/train.py:1148` call-site change — reported, not made.** The card and
   my brief both forbid it. Exact patch in §5. Until it lands, the config path
   is closed and only the env route works.
2. **`meta.deterministic` / `meta.num_threads` not added to `defaults.yaml`.**
   §5. Declaring an unread key is worse than not having it: it silences the
   typo warning and produces a silent no-op. The tripwire enforces the choice.
3. **No training run of any kind.** Never invoked `src.train`. The only
   forward/backward work is the 10-step 1-vs-2-thread measurement the card
   explicitly required, on a shape far below `xsmall_30m` (embed 384, depth 6,
   batch 8, seq 128) and never as a validation step.
4. **No `.github/**` edit.** CI is live. I read `ci.yaml` to localise the
   2-4-core finding and changed nothing.
5. **Did not touch `tests/conftest.py`.** The session-scope thread pin that
   would close the module-scoped-fixture hole is TASK-01's file.
6. **The 1-vs-2 non-deterministic difference is measured, not asserted, in the
   test file.** Whether a given host's BLAS splits a reduction by thread count
   is a property of the host, not of this repo; asserting it would make CI
   flaky on a different kernel. The direction that *is* asserted — determinism
   makes both launches agree — holds on any host, because both runs are pinned
   to one thread.

**risks**

1. **A pre-existing test could depend on `cudnn.benchmark` staying `True`.**
   Checked: nothing in `tests/` reads `cudnn`, and no caller passes
   `deterministic=True` today, so the default path is bit-identical to before.
   The only behaviour change for existing callers is
   `torch.use_deterministic_algorithms(False, warn_only=True)` being called
   explicitly on every `seed_everything` call — that *resets* the flag to off,
   which is the safe direction and is what the symmetry test pins.
2. **`worker_init_fn` does not pin threads in DataLoader workers.** With
   `data.num_workers: 2` and N intra-op threads, workers inherit N and can
   oversubscribe. That is thread *oversubscription* (a throughput bug), not
   thread-count non-determinism (workers draw from their own seeded
   `np.random`/`random`, and the `RandomSampler` runs in the main process), so
   it is out of scope here — but it is a real follow-up and I flag it rather
   than leave it implicit. `tools/rt.py` sidesteps it with
   `OMP_NUM_THREADS=1` in the child env.
3. **Pinning to `N > 1` is not portable.** `deterministic=True, num_threads=4`
   makes a run repeatable *at 4 threads on that CPU*; two machines with 4
   threads can still differ because the BLAS behind ATen may pick a different
   kernel. The docstring says this plainly rather than implying the flag buys
   cross-machine bitwise equality, which on CPU it cannot.
4. **A future editor may add the config keys without the trainer change.** That
   is the top risk of this card's design, and §9's mutation 2 is the mitigation:
   it goes red and the message names the fix. If someone also edits
   `test_the_trainer_still_passes_neither` to match, the guard is gone — the
   card is honest that a test can be removed.
5. **`tests/test_seed.py` costs ~16 s** (4 subprocesses, ~3 s each of
   interpreter + torch import), paid by CI on every push. Module-scoped
   fixture, so it is four children rather than eight. If that becomes
   unacceptable the honest move is to shrink the model, not to delete the
   end-to-end test — it is the only thing in the file that catches the defect
   on the real model rather than on a flag.
6. **I could not verify the Linux path.** Every number in this report is
   Windows / `torch 2.13.0+cpu`. The fix is platform-independent (it pins the
   pool, and the pinned pool gives the same computation everywhere), but the
   *magnitude* of the 1-vs-2 divergence on `ubuntu-latest` with the PyPI wheel
   is unmeasured. `warn_only=True` in particular means CI may emit
   `UserWarning`s from ATen during any future deterministic run; that is the
   intended behaviour, not a failure, and `pyproject.toml` does not set
   `filterwarnings = error`, so it will not turn CI red.
