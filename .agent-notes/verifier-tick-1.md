# TICK 1 VERDICT

Verifier: independent verifier, tick 1. Branch judged: `agent/wave-1` @ `c356369`.
Base: `campaign/integration` @ `d92d8a7`. All 7 task branches share that base.

```
- gate in tree: GREEN
    policy 61 passed; config syntax OK; ruff clean; black 106 files unchanged
    test_config_system.py        557 passed, 21 skipped
    test_torchio.py              10 passed
    test_checkpoint_fidelity.py  20 passed
    test_training_state_guards.py 58 passed, 1 xfailed
    mechanism x14                373 passed
    interp contracts x4          167 passed
    GATE: GREEN (exit 0)

- gate in fresh clone: GREEN   (C:/Users/Илья/tmp/clone-check-1, on agent/wave-1)
    identical stage-by-stage counts: 557/21, 10, 20, 58+1x, 373, 167 -> GATE: GREEN (exit 0)

- boundary violations: NONE
    Every task branch touched only its own files + its own .agent-notes report.
    No worker touched .github/, TASKS.md, config/**, defaults.yaml, or another
    task's src/interp/** file. Only src/train.py was modified by TASK-04 alone.
    No push: no task branch has an upstream; origin/main is unmoved at 2268639.

- new skips/xfails: NONE
    Wave diff over tests/ contains zero +lines matching skip/xfail/pytest.skip.
    Tree xfail count is exactly 1, pre-existing (tests/test_training_state_guards.py:489).
    pytest.skip( call sites in test_config_system.py: 6 -> 5 (TASK-04 removed one).
```

The gate is green. That is the weakest fact in this report. Everything below is
about what the green does not cover.

---

## REGRESSIONS INTRODUCED

### SEVERITY: HIGH — TASK-02 broke resume fidelity on a default-on code path

**The worker's claim is TRUE. I reproduced it, and it is worse than they framed it.**

The claim:

- `src/models/cmc.py:148` — `_mask_rng_default: torch.Generator | None = None` is a
  **module global**, not a tensor, not a registered buffer.
- `src/models/cmc.py:172-177` — it is built once from `torch.initial_seed()` and
  then advanced for the life of the process. Its position is not in `state_dict`.
- `src/models/cmc.py:222` — the no-args path yields that global.
- `src/train.py:95-118` — `_capture_rng_state()` stores `python_random`, `numpy`,
  `torch` (and `cuda`). There is no CMC entry, and there could not be: it is a free
  function and the generator is not reachable from it.
- `src/train.py:1311-1316` — the **production call site** passes neither `rng=` nor
  `seed=`, so production takes exactly the uncheckpointed path.

**Why nothing caught it.** `tests/test_checkpoint_fidelity.py::_model_config()`
(lines 86-105) enables jawp/cgn/pcr/spc/sta/wsd/puc/rdc and **does not set
`use_cmc`**. `src/models/jepa.py:144` defaults `use_cmc` to `False`. So the decisive
resume gate has never once executed the code TASK-02 changed. Separately, that
fixture runs its "fresh process" branch **in the same interpreter**
(`test_resume_reproduces_continuous_run`, lines 188-221), which would have hidden a
module-global position bug even if CMC were on. Two independent reasons for the same
blind spot.

**Measurement.** I transcribed the CMC branch of `src/train.py:1292-1351` verbatim
into a scratch harness (it lives in `main()`, not `compute_loss`, so calling
`compute_loss` directly does not reach it), turned CMC on, and split the run across
two OS processes. Second-mask fingerprints, md5 of the mask tensor:

| step | continuous (1 process) | resumed (NEW process) |
|------|------------------------|----------------------|
| 3 | `b1d1b2e66e18` | `7bdddb25a9ef` |
| 4 | `337a828ade04` | `89984429db79` |
| 5 | `a24f7d67abd5` | `ba655f7a60cc` |

The resumed run draws the masks that were already consumed at **steps 0, 1, 2**,
before the cut. The stream rewinds to position 0. Not "part-way" — to the beginning.

Same harness against the pre-fix `cmc.py` (restored from `d92d8a7`):

| step | continuous | resumed |
|------|-----------|---------|
| 3 | `b3614933efcc` | `b3614933efcc` |
| 4 | `b7cb23769ea0` | `b7cb23769ea0` |
| 5 | `ac73a24e4200` | `ac73a24e4200` |

Bit-exact, all 17 significant digits of the loss included. **The worker's claim that
resume was exact before the fix is confirmed, not just asserted.**

**How bad, honestly.** On the toy fixture the first post-resume loss differs by
**1.2e-6 relative** (1.4576359987258911 vs 1.4576377868652344). That is float32
round-off, three orders of magnitude below the 5.8% this campaign exists to kill.
I am reporting that number because it is what I measured, but it is **not** the
severity, and anyone quoting it will mislead you. The toy model's consistency term
is ~1.5e-5 because a 32-dim model after 3 steps is trivially self-consistent — that
is what CMC is *for*. The severity is in the mechanism, not the toy magnitude:

- `defaults.yaml:187-192`: `use_cmc: true`, `cmc_mode: interval`, `cmc_interval: 10`,
  `lambda_cmc: 0.01`. **`config/ablations/cmc_on.yaml:28` uses `lambda_cmc: 0.1`.**
  CMC is on by default. This is not an opt-in path.
- With `interval: 10` the stream advances once every 10 steps. A run resumed at
  step 5000 restarts the mask schedule at its position 0 and stays permanently
  offset from the data order for the rest of training. That is an unbounded,
  monotone misalignment of a regularizer against its input, not a 1e-6 blip.
- The early-training regime, where a JEPA model is *least* self-consistent and the
  consistency term is *largest*, is exactly the regime where the per-step divergence
  is biggest.

**Minimal fix — and it does not need `src/train.py`.** Make the default stream a
module *buffer* so it round-trips through `state_dict` on the existing path. Add to
`CrossMaskConsistency.__init__`:

```python
self.register_buffer("_mask_rng_state", torch.empty(0, dtype=torch.uint8))
```

and in `_default_mask_rng`, seed a `torch.Generator` from `self._mask_rng_state`
when it is non-empty, then write `gen.get_state()` back into the buffer after the
draw. `save_checkpoint` already stores `model.state_dict()` wholesale
(`src/train.py:235`), so the stream is captured with no change to `train.py` and no
change to the checkpoint format. Owner: whoever holds `src/models/cmc.py` next tick.
**Do not let this ship without the resume test in the patch below** — otherwise the
same regression returns silently.

The alternative (add `"cmc_mask"` to `_capture_rng_state`) needs `src/train.py` to
take the model, which means touching a collision-hotspot file for the second
consecutive tick. Prefer the buffer.

### SEVERITY: MEDIUM — TASK-07 made `load_model("jepa")` unusable for every real checkpoint

Not in the worker's report, and I believe not noticed.

`src/interp/run_comparison.py:70` builds `config = TextSpanJEPAConfig()` with **no
arguments**. Per `src/models/jepa.py:44-59` that is vocab 50304 / seq 512 /
embed_dim 768 / depth 12 / heads 12 / predictor 384x6. `_full_model_state` (line 29)
now demands a full `state_dict`, and line 97 loads it with `strict=True`.

No shipped config matches those defaults. `config/scaling/xsmall_30m.yaml` is
384/6, `small_100m.yaml` 640/10, `large_300m.yaml` 1024/16, `base_140m.yaml`
768/12 (and that one still differs elsewhere). And `save_checkpoint`
(`src/train.py:233-242`) does **not** store the model config, so `load_model` has no
way to reconstruct the right architecture even if it wanted to.

So: pre-fix, `load_model` silently loaded a wrong partial model. Post-fix, it raises
for every checkpoint this repo can actually produce. The second is *more honest* and
still a broken tool. The real fix is to persist the model config in the checkpoint
and rebuild from it — that is a `train.py` + `run_comparison.py` change and needs a
decision, not a drive-by. Flagging it as unowned and outstanding.

---

## FIXES WITH NO DETECTOR — acceptable?

All three workers' self-reports are **accurate**. I verified each rather than
taking them.

### TASK-02 `src/models/cmc.py` — admission TRUE, detector warranted

I restored the pre-fix file and ran the tests that could plausibly notice:

- `tests/test_cmc.py` + `test_mechanism_wiring.py` + `test_sterility.py`:
  **71 passed**.
- `tests/test_checkpoint_fidelity.py`: **20 passed**.

91 tests, zero red, with the entire fix reverted. The admission is honest and the
gap is total. A test is clearly warranted.

Add to **`tests/test_cmc.py`** (a file the gate already runs, so this becomes
gate-enforced for free):

```python
class TestMaskRngIsolation:
    """CMC's mask draws must not touch the process-global torch RNG.

    `src/models/cmc.py` used to call `torch.randint(..., generator=None)`, so every
    span length and offset came out of the stream that DropPath, the CGN Gumbel and
    every other consumer shared. These pin the isolation the private generator
    provides, on all three paths (`rng=`, `seed=`, and the default).

    `_mask_rng_default` is module state, so it is reset before every test here: a
    leaked stream would make this file itself order-dependent.
    """

    @pytest.fixture(autouse=True)
    def _reset_private_stream(self):
        from src.models import cmc as cmc_mod

        cmc_mod._mask_rng_default = None
        yield
        cmc_mod._mask_rng_default = None

    @staticmethod
    def _global_state():
        return torch.get_rng_state().clone()

    def test_default_path_does_not_advance_the_global_stream(self):
        from src.models.cmc import CrossMaskConsistency

        before = self._global_state()
        mask = CrossMaskConsistency.generate_second_mask(16, 2, 0.3)
        assert mask.shape == (2, 16)
        assert torch.equal(self._global_state(), before), (
            "the default mask path consumed the process-global torch RNG; every "
            "other consumer's stream is now coupled to CMC"
        )

    def test_seed_path_is_reproducible_and_leaves_the_global_stream_alone(self):
        from src.models.cmc import CrossMaskConsistency

        before = self._global_state()
        a = CrossMaskConsistency.generate_second_mask(16, 2, 0.3, seed=1234)
        mid = self._global_state()
        b = CrossMaskConsistency.generate_second_mask(16, 2, 0.3, seed=1234)
        c = CrossMaskConsistency.generate_second_mask(16, 2, 0.3, seed=5678)
        assert torch.equal(a, b), "the same seed produced two different masks"
        assert not torch.equal(a, c), "different seeds produced the same mask"
        assert torch.equal(mid, before), "the seed= path consumed the global stream"
        assert torch.equal(self._global_state(), before)

    def test_rng_path_advances_only_the_callers_generator(self):
        from src.models.cmc import CrossMaskConsistency

        before = self._global_state()
        caller = torch.Generator().manual_seed(99)
        a = CrossMaskConsistency.generate_second_mask(16, 2, 0.3, rng=caller)
        b = CrossMaskConsistency.generate_second_mask(16, 2, 0.3, rng=caller)
        assert not torch.equal(a, b), (
            "a caller-owned generator must advance between calls, otherwise the "
            "second mask is not a second mask"
        )
        assert torch.equal(self._global_state(), before)

    def test_rng_and_seed_together_raise(self):
        from src.models.cmc import CrossMaskConsistency

        with pytest.raises(ValueError, match="not both"):
            CrossMaskConsistency.generate_second_mask(
                16, 2, 0.3, rng=torch.Generator(), seed=1
            )
```

And add to **`tests/test_checkpoint_fidelity.py`**, which is the only place in the
repo that can catch the regression above. The existing
`test_resume_reproduces_continuous_run` must be parameterised over `use_cmc` — but
because that fixture's "fresh process" is the same interpreter, a module-global
position bug still hides. The test has to run the two halves in subprocesses, or
`seed_for`-style re-entry must be simulated explicitly. Cheapest honest version that
catches *this* class of bug without spawning interpreters: assert the resume
contract on the mask stream directly.

```python
    def test_cmc_mask_stream_survives_the_checkpoint_round_trip(self, tmp_path):
        """CMC's second mask must be checkpointed, or every resume is a new run.

        `src/train.py:1311` calls `generate_second_mask` with no `rng=` and no
        `seed=`, i.e. the default process-private stream. That stream is a module
        global (`src/models/cmc.py:148`), so it is absent from `state_dict` and
        `_capture_rng_state` cannot reach it. Measured consequence: a run resumed
        in a new process re-drew the masks consumed *before* the cut, and the
        first post-resume loss moved by 1.2e-6 on the toy fixture and by an
        unbounded amount as `lambda_cmc` grows.

        This test turns CMC on -- the shipped `_model_config()` never does, and
        `use_cmc` defaults to False, which is why the regression shipped green.
        """
        from src.models import cmc as cmc_mod

        cmc_mod._mask_rng_default = None
        try:
            state = _build(seed=0)
            model = state[0]
            assert model.cmc is not None, "fixture does not exercise CMC"

            first = model.cmc.generate_second_mask(SEQ, 2, 0.3)
            second = model.cmc.generate_second_mask(SEQ, 2, 0.3)
            assert not torch.equal(first, second), "the private stream is not advancing"

            path = str(tmp_path / "ckpt.pth.tar")
            save_checkpoint(
                path, model, state[1], state[6], epoch=0, global_step=1,
                model_name="text_span_jepa",
            )
            payload = safe_torch_load(path, map_location="cpu")
            held = [k for k in payload["model"] if "mask_rng" in k]
            assert held, (
                "CMC's mask RNG is not in the checkpoint payload. Every resume "
                "restarts the mask schedule at position 0, permanently offset from "
                "the data order. Fix: register the generator state as a buffer on "
                "CrossMaskConsistency so state_dict carries it."
            )
        finally:
            cmc_mod._mask_rng_default = None
```

**Note for the coordinator:** that second test asserts the *fix*, not the current
behaviour, so it will be red until the buffer lands. Land them in one commit, or the
wave goes red on its own detector. That is the correct order.

### TASK-07 `src/interp/run_comparison.py` — admission TRUE, detector warranted

`Select-String tests\*.py -Pattern "run_comparison"` → **0 matches**. The only
references anywhere are prose in `.agent-notes/`. The module is not in
`src/interp/__init__.py`. It is reachable only via
`scripts/run_experiment.sh`. A test is warranted; it is cheap because the
interesting paths are refusals that need no model construction.

New file `tests/test_run_comparison.py`:

```python
# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""`run_comparison.load_model` must not half-restore a checkpoint.

The pre-fix implementation read `ckpt.get("encoder", {})` and friends with
`strict=False`, a format `src.train.save_checkpoint` has not written for some
time. Every real key missed, so the model came back with its encoder at random
init and a "comparison" was run between an untrained network and a baseline.
`_full_model_state` refuses that shape by name instead. These pin the refusal.
"""

import pytest
import torch

from src.interp.run_comparison import _full_model_state, load_model
from src.train import CheckpointLoadError


def test_refuses_the_legacy_partial_format(tmp_path):
    """A checkpoint with named submodules and no `model` key is refused."""
    path = tmp_path / "legacy.pth.tar"
    torch.save({"encoder": {"w": torch.ones(2)}, "predictor": {}}, str(path))
    with pytest.raises(CheckpointLoadError, match="full-state_dict"):
        load_model(str(path), "jepa")


def test_refuses_a_payload_that_is_not_a_dict(tmp_path):
    path = tmp_path / "list.pth.tar"
    torch.save([1, 2, 3], str(path))
    with pytest.raises(CheckpointLoadError):
        load_model(str(path), "jepa")


def test_refusal_names_the_keys_it_found(tmp_path):
    """The error must say what WAS there, or the next author repeats the bug."""
    path = tmp_path / "legacy.pth.tar"
    torch.save({"encoder": {}, "target_encoder": {}}, str(path))
    with pytest.raises(CheckpointLoadError) as exc:
        load_model(str(path), "jepa")
    assert "encoder" in str(exc.value)


def test_unknown_model_type_raises_before_touching_disk():
    with pytest.raises(ValueError, match="Unknown model type"):
        load_model("does-not-exist.pth.tar", "gpt2")


def test_full_model_state_passes_through_a_well_formed_payload():
    state = {"encoder.blocks.0.w": torch.ones(2)}
    assert _full_model_state({"model": state}, "p") is state
```

The last one is deliberately trivial; it is the only test in the file that can
exist without instantiating a 140M-parameter model, which is the deeper problem
described in REGRESSIONS above.

### TASK-01 `tests/conftest.py` — my call: legitimate guard, but it rots today. Pin it.

I renamed the file away and ran 168 tests across `test_cmc`, `test_cgn`,
`test_probes_split`, `test_index_and_cka`, `test_feature_composition`,
`test_ablation_module`. **168 passed.** The admission is exact.

**My position: keep it, and pin it.** The argument for deletion is weak on the
facts: 12 of 21 test files have no seeding call, and `src/interp/` draws from the
global RNG at 28 sites without importing `src/utils/seed.py`. Deleting a guard
because no test *currently* observes it is how a suite becomes order-dependent
without anyone noticing — that is the same class of error as this whole campaign.
The argument against is equally real: an unfalsifiable fixture is one refactor away
from being decoration, and the fixture's own docstring already anticipates the
subtle failure (builtin `hash()` instead of `blake2b`, which would silently destroy
reproducibility while every test still passed). That property is worth more than
the seeding itself and nothing defends it.

So: keep, and add `tests/test_conftest_determinism.py`. Every test below fails if
the file is deleted, if the fixture stops being autouse, or if the seed derivation
changes. The autouse proof is the important one — it is the only assertion in the
file that would notice the fixture being turned into a normal opt-in fixture.

```python
# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Detectors for `tests/conftest.py`.

`tests/conftest.py` installs the suite's only determinism enforcement: an autouse
fixture that seeds `random`, `numpy` and `torch` from a per-node-id hash and pins
torch to one thread. Without it, 12 of the 21 test files contain no seeding call at
all, and `src/interp/` draws from the global RNG at 28 sites.

On its own the fixture is unfalsifiable. Deleting `tests/conftest.py` leaves 168
tests green across the mechanism, interp and ablation files. These are the tests
that make it falsifiable.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

import conftest as determinism


def test_the_fixture_module_exists():
    """Deleting tests/conftest.py must fail the suite, not pass it."""
    assert callable(determinism.seed_for_node)
    assert determinism.BASE_SEED_ENV == "PYTEST_BASE_SEED"


def test_seed_for_node_is_stable_across_processes():
    """`hashlib.blake2b`, not the builtin `hash()`.

    The builtin randomises string hashing per interpreter process, so a
    `hash(nodeid)` seed would differ on every run and every test would still pass
    while the suite quietly stopped being reproducible. That is the single most
    valuable property this file has and nothing else defends it, so it is checked
    in a real second process with a deliberately different PYTHONHASHSEED.
    """
    code = (
        "import sys; sys.path.insert(0, r'tests');"
        "import conftest; print(conftest.seed_for_node('tests/test_x.py::test_a'))"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
    )
    assert int(out.stdout.strip()) == determinism.seed_for_node("tests/test_x.py::test_a")


def test_distinct_node_ids_get_distinct_seeds():
    ids = [
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two",
        "tests/test_b.py::test_one[k1]",
        "tests/test_b.py::test_one[k2]",
    ]
    seeds = {determinism.seed_for_node(i) for i in ids}
    assert len(seeds) == len(ids), "two node ids collided onto one seed"


def test_the_base_seed_shifts_every_node_uniformly():
    """Bisecting a stochastic flake is the documented use of the env var."""
    node = "tests/test_a.py::test_one"
    base = determinism.seed_for_node(node, base=0)
    shifted = determinism.seed_for_node(node, base=1)
    assert base != shifted
    assert determinism.seed_for_node(node, base=determinism._SEED_MODULUS) == base


def test_a_non_integer_base_seed_is_a_usage_error(monkeypatch):
    monkeypatch.setenv(determinism.BASE_SEED_ENV, "not-a-number")
    with pytest.raises(pytest.UsageError):
        determinism._read_base_seed()


def test_the_autouse_fixture_actually_seeded_this_node(request):
    """The end-to-end proof that the fixture runs without being asked.

    This test never requests `deterministic_rng`. The fixture's last action before
    the test body is `torch.manual_seed(seed_for_node(<this node id>))`, so the live
    global state must equal a fresh generator seeded identically. It does not if
    the fixture is deleted, if it stops being autouse, or if anything else seeds
    the global RNG in between -- all three of which are silent failures otherwise.
    """
    expected = torch.Generator()
    expected.manual_seed(determinism.seed_for_node(request.node.nodeid))
    assert torch.equal(torch.get_rng_state(), expected.get_state()), (
        "the autouse determinism fixture did not seed the global torch RNG with "
        "seed_for_node() for this node id"
    )
    np.random.seed(determinism.seed_for_node(request.node.nodeid))
    probe = np.random.rand()
    np.random.seed(determinism.seed_for_node(request.node.nodeid))
    assert probe == np.random.rand()
```

Caveat the coordinator should know: `test_the_autouse_fixture_actually_seeded_this_node`
asserts on live global state, so it will fail if any *other* autouse fixture in the
suite starts seeding after this one. That is a true positive, not a flake, but it
makes the test coupled to fixture ordering. It is the price of proving autouse
behaviour without pytest private API.

---

## MUTATION SPOT-CHECK

- **TASK-13: DETECTED.** Mutant replaced the disjoint control with the full
  dataset (`control_idx = torch.arange(N)`), i.e. the original overlap bug.
  `tests/test_feature_composition.py` went red on **6 assertions across multiple
  tests**, including the exact property assertions:
  `a sample is both treated and control` (`{1,5} & {0,1,2,3,4,5}`),
  `control must exclude the treated samples`, and the `n_treated`/`n_control`
  size checks. Reverted.

- **TASK-17: DETECTED.** Mutant removed the `try`/`finally` around
  `compute_loss_with_targets` in `AblatedModel.forward`
  (`src/interp/ablation.py:340-352`), leaving the restore on the non-raising path
  only. `tests/test_ablation_module.py` → **2 failed, 20 passed**:
  `TestRefinementStateIsRestored::test_num_refine_steps_restored_when_the_forward_raises`
  and `::test_a_crash_does_not_corrupt_a_later_forward`. Reverted.

Both workers' mutation claims hold up. These are the two tasks that shipped tests,
and the tests are real.

- **worktree state after: clean.** `git diff --stat HEAD` empty. `git status
  --short` shows only `?? .agent-notes/raid/control-scout.md`, which I did not
  create and did not touch — it appeared during my session from another agent.
  All scratch files are under `C:\Users\Илья\AppData\Local\Temp\opencode\`
  (`probe2.py`, `cmc_resume_probe.py`, `constify.py`, `cmc_prefix.py`,
  `cmc_postfix.bak`, `ck1/`, `ck2/`, `gate-tree.txt`, `gate-clone.txt`) and are
  left there deliberately for the coordinator to inspect. Nothing was left in the
  repo.

---

## RULES COMPLIANCE

- **No skip / xfail / `--no-verify` added.** The wave diff over `tests/` has zero
  added lines matching `pytest.mark.skip`, `pytest.mark.xfail`, `pytest.skip`, or
  `pytest.xfail`.
- **xfail count is still exactly 1**, at `tests/test_training_state_guards.py:489`,
  pre-existing, `strict=False`, on a real WSR defect. Its reason text names the
  defect precisely (`_stiefel_retract` takes column signs from `diag((Q R)[:k, :])`
  instead of `diag(R)`) and states it needs a human decision. That is exactly the
  escalation discipline `AGENTS.md` asks for. **It should not block the wave.**
  Minor note: `strict=False` means a future fix XPASSes silently. Acceptable here.
- **TASK-04 did remove a real skip**, and removed the right one. The skip it deleted
  was the honest-but-paper-over kind: a test named
  `test_trainer_ema_fallback_is_not_the_frozen_value` that reported a `src/train.py`
  defect and skipped rather than failing. It is now a real assertion, and the fix is
  real (`src/train.py:836`, `1.0` → `0.9999`, which is what
  `EMATauSchedule.step()` returns, and `1 - 1.0 == 0` made
  `update_target_encoder` a no-op). Skip call sites 6 → 5. This is the best-executed
  task in the wave.
  **Residual weakness:** the test now begins with
  `if m is None: pytest.skip(...)` and otherwise regexes the *source text* of
  `train.py`. It is a self-disabling test — delete the fallback and it skips again —
  and it pins a literal, not a behaviour. A behaviour test (build the scheduler
  through `_build_optimization` on a config that omits the key, step it, assert
  `tau < 1.0`) would be strictly better and cannot be disabled. Not a violation; a
  note for whoever owns `train.py` next.
- **TASK-17's `status="failed"`: honest, but it breaks 3 of 12 arms, not 2.**

  Nothing existing broke. `tests/test_config_system.py` is green (557/21) and
  `tests/test_interp.py` is green (94 passed) — `test_interp.py:992-995` exercises
  `AblationConfig("no_pred", use_predictor=False)` and still passes, because it only
  checks registry membership, not that the arm can run.

  And `status="failed"` is the *honest* choice: the cell carries no `final_loss`,
  `AblationResults.failures` is non-empty, `is_complete` is False, and `summary()`
  prints `INCOMPLETE`. Nothing can be plotted from it as a result. That is the
  opposite of a silent failure. **But the audit undercounts the blast radius.**
  `src/interp/ablation.py:150-174` has **three** arms that now fail permanently, not
  two:
  - `no_predictor` (line 152) — `use_predictor=False`
  - `no_centering` (line 164) — `use_target_centering=False`
  - `predictor_only` (line 166-173) — `use_target_centering=False`, an *on*-arm
    style config, not a leave-one-out. The task brief did not mention it.

  So `AblationStudy.run_all()` with no arguments now returns a study that is
  permanently 25% incomplete, and the reason string a user sees is
  `use_target_centering: no ablation mechanism is implemented for it`. That is
  truthful and actionable, which is the point. The *resolution* — implement the two
  flags, or drop the three arms from `ABLATION_CONFIGS` — is unowned, touches
  mechanism behaviour, and per `AGENTS.md` must not be decided unilaterally. It
  needs a card.
- **Gate coverage gap worth closing.** The wave modified
  `src/interp/ablation.py` and `src/interp/feature_composition.py`, and the gate's
  interpretability stage runs only `test_probes_split`, `test_index_and_cka`,
  `test_info_and_disentangle`, `test_causal_intervention`. **`tests/test_interp.py`
  and the two new files are not in the gate.** Had `test_interp.py` broken, this gate
  would have stayed green. Add both new test files to
  `.agent-notes/gate.sh:79-81`.
- **The gate itself was edited after the wave merged** (`f973ccd`, not a worker).
  I checked the diff: it only *adds* two stages (policy, config syntax). No test
  stage was removed or weakened. Fine.
- **One process note that will bite someone:** `bash` on this host resolves to
  `C:\Windows\system32\bash.exe` → WSL, and under WSL the gate's interpreter pin
  `[ ! -x "$PY" ]` fails and it silently falls back to `/usr/bin/python3`, which has
  no pytest, ruff or black. The gate then reports **GATE: RED** for entirely
  environmental reasons. Run it as
  `"C:\Program Files\Git\bin\bash.exe" .agent-notes/gate.sh`.
  Both of my green results used Git-Bash. Worth a line in the gate header.

---

## WHAT IS STILL MISSING

- **The 1 xfail** — `tests/test_training_state_guards.py:489`, the WSR retraction
  column-sign defect. Pre-existing, precisely documented, awaiting a human decision
  by the repo's own rule. **Does not block the wave.**

- **`tests/test_interp.py` is close to inert as a metric gate.** Measured, not
  estimated. I wrote a pytest plugin (outside the repo) that wraps metric entry
  points in `feature_composition`, `polysemanticity` (×3), `disentanglement`,
  `information_theory` (×3) and `causal_intervention`, runs the real function, and
  then overwrites **every float leaf in the result with a constant in range** —
  preserving keys, shapes and types, destroying all values.

  - `tests/test_interp.py` (94 test functions, 183 assert statements):
    **1 test goes red. 93 survive** — `TestInterpretabilityIndex::test_compare`, and
    only because it compares two constified calls against each other.
  - The gate's own interp files (167 tests), same mutant: **8 go red**
    (`test_index_and_cka` ×3, `test_causal_intervention` ×5).

  So the answer to the question as asked: of 94 test functions, **93 would pass with
  the metrics replaced by constants** — roughly 99%. The assertions are dominated by
  shape checks (`assert recons.shape == (4, 32)`), key-presence checks
  (`assert "uuas" in result`), finiteness checks (`assert torch.isfinite(loss)`) and
  range checks (`assert 0 <= result["disentanglement"] <= 1`). All four survive a
  metric that returns a constant. The file is a smoke test and behaves like one.

  The encouraging half: the dedicated contract files the gate *does* run are
  genuinely load-bearing, 8 red for the same mutant. The coverage is in the right
  place; `test_interp.py` is just mislabelled by living next to it. Either promote
  its value assertions or move it out of the interp naming.

---

## WHAT I COULD NOT VERIFY

- **Accumulated divergence of the CMC resume regression over a long run.** My
  harness runs 3+3 steps on a 32-dim model. It proves the mask stream rewinds
  exactly, and it proves the pre-fix path was bit-exact. It cannot tell you what a
  50k-step run's final loss does, and I am not going to guess. I was told never to
  run training, and a defensible number here needs a real run. **Treat the severity
  as "mechanism proven, magnitude unmeasured at scale."**
- **The exact pre-wave collected skip/pass split in `test_config_system.py`.** I
  proved the source-level change (6 → 5 `pytest.skip(` call sites) and that the
  removed skip was conditional on the literal being `>= 1.0`, which it was, so
  556/22 → 557/21 is the expected move. I did not run the suite at
  `campaign/integration` to confirm the collected numbers. The reasoning is sound;
  the measurement is not mine.
- **`tools/rt.py` budget accounting across my whole session.** I ran roughly 20
  invocations, all single-file or short lists, all through `rt.py`, all one thread.
  I did not total the wall clock. No full-suite run, no training, no `torchrun`.
- **Whether the 12 other tests named in the new-file patches behave as I predict.**
  I wrote the patches as text for the coordinator and did not apply them, so the
  `test_cmc_mask_stream_survives_the_checkpoint_round_trip` test is known-red by
  design (it asserts the fix) and the rest are unexecuted. They need a real run.
- **CI.** The authoritative `pytest` check runs on GitHub Actions, off-machine. I
  saw no CI result. My green gates cover ~1247 of the repo's tests; the full suite
  and the four `--slow` files were not run here.
- **I did not re-verify TASK-03's counting claims** (the 12-modules / 16-capabilities
  convention). It is a documentation-only change with a 10-line test rename; I
  confirmed the diff touches no logic and nothing skips, which is what rules
  compliance required, but I did not audit whether the new prose is *true*.
