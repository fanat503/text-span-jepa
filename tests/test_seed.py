# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Contracts for `src/utils/seed.py`: the thread pin and the determinism switch.

Why this file exists
--------------------
The card behind it (`TASK-24`) found two facts:

* `src/train.py` calls `seed_everything(seed)` with no way to pass anything
  else, and `seed_everything` took `deterministic` only as a Python argument
  that nothing ever set. `cudnn.deterministic` / `cudnn.benchmark` are no-ops on
  a CPU-only host, so the flag that named full reproducibility delivered none
  of it here, and no config key or CLI flag could ask for it.
* Nothing pinned the intra-op thread count, which is the axis that actually
  moves the numbers. Measured on this host: 1 vs 2 threads, same seed, same
  data, 10 steps -- steps 0-2 bitwise equal, steps 3-9 differing by
  1.19e-07 to 2.38e-07, final `qkv` weight not bitwise equal.

`seed_everything(deterministic=True)` now pins the thread pool, turns on
`torch.use_deterministic_algorithms`, sets the cudnn flags and sets
`CUBLAS_WORKSPACE_CONFIG`; and both knobs are reachable from the environment,
which works with the trainer's existing call site unchanged.

What is pinned here, and what is deliberately not
-------------------------------------------------
`torch.use_deterministic_algorithms(True)` **without** `warn_only` raises
`RuntimeError` for any op that has no deterministic kernel, and this repo is
full of them (`torch.linalg.svd`, `index_put_`, `scatter`, the Grassmann
workspace ops). The `warn_only=True` is therefore load-bearing rather than
convenient, and there is a test that says so: a strict-mode request would crash
on the first batch, determinism would never be requested, and the feature would
be untestable in practice.

The config path is pinned as CLOSED. `meta.deterministic` and
`meta.num_threads` are not declared in `defaults.yaml` and not read by
`src/train.py`, and that is a decision: declaring an unread key stops the
trainer's own "possible typo" startup warning, so a run asking for determinism
would get silence and no determinism. There is a tripwire that goes red the
moment someone wires the trainer, telling them to declare the keys in the same
commit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.utils.seed import (
    CUBLAS_WORKSPACE_CONFIG_ENV,
    DETERMINISTIC_ENV,
    NUM_THREADS_ENV,
    configure_threads,
    reproducibility_state,
    resolve_deterministic,
    resolve_num_threads,
    seed_everything,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = REPO / "defaults.yaml"
TRAIN_PATH = REPO / "src" / "train.py"
UNREACHABLE_KEYS = ("meta.deterministic", "meta.num_threads")


@pytest.fixture(autouse=True)
def restore_process_state():
    """Put back everything `seed_everything` changes process-wide.

    `tests/conftest.py` restores the thread count but nothing restores
    `torch.use_deterministic_algorithms`, `cudnn.benchmark`, or
    `CUBLAS_WORKSPACE_CONFIG`. Without this fixture a test that asks for
    determinism silently makes every later test in the session deterministic
    too, and the suite stops being the thing it claims to be.
    """
    before = reproducibility_state()
    had_cublas = CUBLAS_WORKSPACE_CONFIG_ENV in os.environ
    cublas_before = os.environ.get(CUBLAS_WORKSPACE_CONFIG_ENV)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            before["deterministic_algorithms"],
            warn_only=before["warn_only_algorithms"],
        )
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        if had_cublas:
            os.environ[CUBLAS_WORKSPACE_CONFIG_ENV] = cublas_before
        else:
            os.environ.pop(CUBLAS_WORKSPACE_CONFIG_ENV, None)
        torch.set_num_threads(before["num_threads"])


@pytest.fixture(autouse=True)
def clean_determinism_env(monkeypatch):
    """Neither knob may leak in from the developer's shell.

    Without this, a box that exports ``TEXT_SPAN_JEPA_DETERMINISTIC=1`` -- which
    is the documented way to run a reproducible job -- would make every
    assertion about the *default* pass or fail depending on who is running the
    suite.
    """
    for name in (DETERMINISTIC_ENV, NUM_THREADS_ENV):
        monkeypatch.delenv(name, raising=False)


# ══════════════════════════════════════════════════════════════════════
#  1. `deterministic=True` actually does something on a CPU-only host
# ══════════════════════════════════════════════════════════════════════


class TestDeterministicIsNotANoOp:
    def test_it_pins_the_intra_op_thread_pool(self):
        """The one that matters here, and the one the old body omitted.

        Float addition is not associative, so a parallel reduction sums in a
        different order at N=1 than at N=8. `cudnn.deterministic` says nothing
        about that: it is a CUDA convolution-algorithm switch and is never read
        on this host.
        """
        torch.set_num_threads(4)
        assert torch.get_num_threads() != 1, "precondition: torch is not already pinned"

        seed_everything(42, deterministic=True)

        assert torch.get_num_threads() == 1, (
            "seed_everything(deterministic=True) left torch on "
            f"{torch.get_num_threads()} threads. Two runs that differ only in "
            "their thread count differ from optimizer step 3 onward, so a "
            "'deterministic' run that is still multi-threaded is not one."
        )

    def test_it_enables_deterministic_algorithms(self):
        """Not just cudnn: the ATen op-level switch, which is the CPU half."""
        seed_everything(42, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled() is True

    def test_warn_only_is_kept_on(self):
        """`warn_only=True` is load-bearing, not a convenience.

        Strict mode raises `RuntimeError` for any op with no deterministic
        kernel, and this repo calls `torch.linalg.svd`, `index_put_` and
        `scatter`. Strict mode would turn a reproducibility request into a
        crash on the first batch, so determinism would never be requested.
        """
        seed_everything(42, deterministic=True)
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True, (
            "warn_only was turned off. On a repo with non-deterministic ops this "
            "makes deterministic=True raise RuntimeError instead of running."
        )

    def test_it_sets_the_cublas_workspace_config(self):
        """cuBLAS splits some GEMMs across threads, so its order is the count."""
        monkey = os.environ.pop(CUBLAS_WORKSPACE_CONFIG_ENV, None)
        try:
            seed_everything(42, deterministic=True)
            assert os.environ.get(CUBLAS_WORKSPACE_CONFIG_ENV), (
                "CUBLAS_WORKSPACE_CONFIG is unset, so cuBLAS may split a GEMM "
                "differently at a different thread count"
            )
        finally:
            if monkey is not None:
                os.environ[CUBLAS_WORKSPACE_CONFIG_ENV] = monkey

    def test_it_never_overwrites_a_cublas_workspace_config_a_launcher_chose(self, monkeypatch):
        """A launcher that got there first owns the value; we only fill a gap."""
        monkeypatch.setenv(CUBLAS_WORKSPACE_CONFIG_ENV, ":16:8")
        seed_everything(42, deterministic=True)
        assert os.environ[CUBLAS_WORKSPACE_CONFIG_ENV] == ":16:8"


class TestTheSwitchIsSymmetric:
    """A function that can be turned on but not off leaks into the next caller.

    `tests/conftest.py` restores the thread count after every test and nothing
    restored the rest, so a single `deterministic=True` would have silently
    made the remainder of the session deterministic. Both directions are
    assigned on every call for that reason.
    """

    def test_false_restores_the_fast_path(self):
        seed_everything(42, deterministic=True)
        seed_everything(42, deterministic=False)
        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False

    def test_the_default_does_not_ask_for_determinism(self):
        """Threading it on by default would tax every run by the core count.

        A reproducibility feature nobody pays for is a feature nobody turns on,
        so the default stays off and the cost stays opt-in.
        """
        seed_everything(42)
        assert torch.are_deterministic_algorithms_enabled() is False
        assert resolve_deterministic() is False
        assert resolve_num_threads() is None


# ══════════════════════════════════════════════════════════════════════
#  2. Reachability without a `src/train.py` edit
# ══════════════════════════════════════════════════════════════════════


class TestEnvironmentRoute:
    """The trainer's call site is `seed_everything(seed)`, so this is the only
    way a run can request determinism today without a forbidden edit."""

    def test_deterministic_env_reaches_the_existing_trainer_call_site(self, monkeypatch):
        """`seed_everything(seed)` -- one positional arg, exactly as train.py
        calls it. This is the whole reachability claim."""
        monkeypatch.setenv(DETERMINISTIC_ENV, "1")
        torch.set_num_threads(4)
        seed_everything(42)
        assert torch.get_num_threads() == 1
        assert torch.are_deterministic_algorithms_enabled() is True

    @pytest.mark.parametrize("spelling", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_spellings(self, monkeypatch, spelling):
        monkeypatch.setenv(DETERMINISTIC_ENV, spelling)
        assert resolve_deterministic() is True, spelling

    @pytest.mark.parametrize("spelling", ["0", "false", "No", "off"])
    def test_falsy_spellings(self, monkeypatch, spelling):
        monkeypatch.setenv(DETERMINISTIC_ENV, spelling)
        assert resolve_deterministic() is False, spelling

    def test_num_threads_env_pins_the_pool_on_its_own(self, monkeypatch):
        """Repeatable at a fixed N is still a real contract, and it is the one
        a throughput run can afford."""
        monkeypatch.setenv(NUM_THREADS_ENV, "1")
        seed_everything(42)
        assert torch.get_num_threads() == 1
        # ...without implying determinism was asked for.
        assert torch.are_deterministic_algorithms_enabled() is False

    def test_an_explicit_argument_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv(DETERMINISTIC_ENV, "1")
        assert resolve_deterministic(deterministic=False) is False
        assert resolve_deterministic(deterministic=True) is True
        monkeypatch.delenv(DETERMINISTIC_ENV)
        assert resolve_deterministic(deterministic=True) is True

    def test_an_explicit_num_threads_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv(NUM_THREADS_ENV, "1")
        assert resolve_num_threads(num_threads=2) == 2

    def test_deterministic_honours_an_explicit_thread_count(self):
        """The pin is not unconditional: a 4-thread deterministic run is a
        legitimate thing to want, and silently forcing 1 would make the
        argument a lie."""
        seed_everything(42, deterministic=True, num_threads=2)
        assert torch.get_num_threads() == 2
        assert torch.are_deterministic_algorithms_enabled() is True

    @pytest.mark.parametrize("bad", ["0", "-1", "many", "1.5", ""])
    def test_an_unusable_num_threads_env_is_an_error_not_a_default(self, monkeypatch, bad):
        """A typo on the reproducibility run must not silently mean 'off'.

        Swallowing it is how a reproducibility knob becomes decoration: the one
        run that was supposed to be reproducible is the one nobody noticed.
        """
        monkeypatch.setenv(NUM_THREADS_ENV, bad)
        if not bad.strip():
            # Empty means "unset" for this variable, by design.
            assert resolve_num_threads() is None
            return
        with pytest.raises(ValueError):
            resolve_num_threads()

    def test_an_unusable_deterministic_env_is_an_error_not_a_default(self, monkeypatch):
        monkeypatch.setenv(DETERMINISTIC_ENV, "treu")
        with pytest.raises(ValueError):
            resolve_deterministic()


# ══════════════════════════════════════════════════════════════════════
#  3. The state a run can report about itself
# ══════════════════════════════════════════════════════════════════════


class TestReportedState:
    def test_it_reports_the_count_in_force_not_the_count_requested(self):
        """`reproducibility_state()` exists to be logged next to the seed, and
        a checkpoint carries weights, not a thread count. Reporting the
        requested value would be reporting a wish."""
        torch.set_num_threads(1)
        seed_everything(42, deterministic=True, num_threads=1)
        assert reproducibility_state()["num_threads"] == torch.get_num_threads()
        assert reproducibility_state()["num_threads"] == 1

    def test_configure_threads_reports_what_it_actually_applied(self):
        applied = configure_threads(1)
        assert applied["num_threads"] == 1
        assert applied["intraop_before"] >= 1
        # The inter-op pool can only be created once, so a second call cannot
        # be honest about it; the key has to say so rather than imply success.
        assert isinstance(applied["interop_pinned"], bool)
        again = configure_threads(1)
        assert again["num_threads"] == 1

    def test_configure_threads_none_touches_nothing(self):
        before = torch.get_num_threads()
        applied = configure_threads(None)
        assert applied["num_threads"] is None
        assert torch.get_num_threads() == before


# ══════════════════════════════════════════════════════════════════════
#  4. The config path is CLOSED, and the tripwire that says so
# ══════════════════════════════════════════════════════════════════════


def _defaults_text() -> str:
    return DEFAULTS_PATH.read_text(encoding="utf-8")


def _train_text() -> str:
    return TRAIN_PATH.read_text(encoding="utf-8")


class TestConfigPathIsClosed:
    """`or state that they are not` -- stated, and pinned so it cannot rot.

    `meta.deterministic` / `meta.num_threads` are not declared in
    `defaults.yaml` and not read by `src/train.py`. Declaring an unread key
    stops `_warn_unknown_config_keys` from warning about it, so a run asking
    for determinism would get silence and no determinism: a silent no-op, which
    this repo treats as worse than the typo warning it would otherwise get.
    """

    def test_they_are_not_declared_in_defaults(self):
        import yaml

        defaults = yaml.safe_load(_defaults_text())
        meta = defaults.get("meta", {})
        for key in ("deterministic", "num_threads"):
            assert key not in meta, (
                f"defaults.yaml now declares meta.{key}, but src/train.py does not "
                "read it. The trainer's own startup warning has just been silenced "
                "for a key that does nothing, which turns a loud 'this did not "
                "work' into a silent no-op. Wire src/train.py in the SAME commit, "
                "or remove the key -- see the meta: block of defaults.yaml."
            )

    def test_the_trainer_still_passes_neither(self):
        src = _train_text()
        assert "seed_everything(seed)" in src, (
            "src/train.py no longer contains the bare `seed_everything(seed)` call. "
            "If the trainer now forwards meta.deterministic / meta.num_threads, "
            "declare both in defaults.yaml and update "
            "TestConfigPathIsClosed::test_they_are_not_declared_in_defaults -- "
            "they are reachable from config now and this file is lying."
        )

    def test_the_environment_names_are_documented_where_a_reader_looks(self):
        """defaults.yaml is where someone goes to find the knobs. If the names
        there and the names in code drift, the documented route is a no-op."""
        text = _defaults_text()
        assert DETERMINISTIC_ENV in text, (
            f"defaults.yaml never mentions {DETERMINISTIC_ENV}, so the only "
            "working route to determinism is undocumented"
        )
        assert NUM_THREADS_ENV in text, f"defaults.yaml never mentions {NUM_THREADS_ENV}"

    def test_the_measured_numbers_in_defaults_are_the_measured_numbers(self):
        """The doc block quotes |d| = 1.19e-07 ... 2.38e-07. If the mechanism
        or the range changes, the quoted evidence has to change with it."""
        text = _defaults_text()
        assert re.search(r"1\.19e-07\s*\.\.\.\s*2\.38e-07", text), (
            "defaults.yaml's reproducibility block no longer quotes the measured "
            "divergence range, so the claim there has lost its evidence"
        )


# ══════════════════════════════════════════════════════════════════════
#  5. End to end, in a subprocess, on the repo's own model
# ══════════════════════════════════════════════════════════════════════


_SUBPROCESS = r"""
import hashlib, json, os, sys
sys.path.insert(0, REPO)                      # the worktree, not the stale clone
import src
assert WORKTREE in os.path.realpath(src.__file__), src.__file__

import torch
from src.masks.span import SpanMaskCollator
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.utils.seed import seed_everything

# The ONLY difference between the two runs: how many threads the environment
# offered. seed_everything is called exactly as src/train.py calls it.
seed_everything(SEED)

cfg = TextSpanJEPAConfig(
    vocab_size=256, max_seq_len=32, embed_dim=64, encoder_depth=1, num_heads=2,
    predictor_embed_dim=32, predictor_depth=1, future_offsets=[1],
    num_refine_steps=1, drop_path_rate=0.0,
)
cfg.validate()
model = TextSpanJEPA(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

gen = torch.Generator().manual_seed(SEED)
batch = [{"input_ids": torch.randint(1, 256, (32,), generator=gen)} for _ in range(2)]
collated = SpanMaskCollator(mask_ratio=0.35, span_length_range=(3, 10), mask_token_id=0)(batch)

for step in range(STEPS):
    total, _l, _d = model.compute_loss_with_targets(
        collated["masked_input_ids"], collated["original_input_ids"],
        collated["mask_positions"], current_step=step, total_steps=STEPS,
    )
    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()

blob = b"".join(
    p.detach().numpy().tobytes() for p in model.state_dict().values()
    if p.is_floating_point()
)
print(json.dumps({
    "threads": torch.get_num_threads(),
    "sha": hashlib.sha256(blob).hexdigest(),
    "src": os.path.realpath(src.__file__),
}))
"""


def _run_child(threads: int, deterministic: bool, seed: int = 7, steps: int = 4) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    if deterministic:
        env[DETERMINISTIC_ENV] = "1"
    else:
        env.pop(DETERMINISTIC_ENV, None)
    env.pop(NUM_THREADS_ENV, None)
    script = (
        f"REPO = {str(REPO)!r}\n"
        f"WORKTREE = {str(REPO)!r}\n"
        f"SEED = {seed}\nSTEPS = {steps}\n" + _SUBPROCESS
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    line = [x for x in proc.stdout.splitlines() if x.strip().startswith("{")][-1]
    return json.loads(line)


@pytest.fixture(scope="module")
def child_runs():
    """Four child processes, run once each: {deterministic: {threads: result}}.

    Module-scoped because each child pays ~3s of interpreter and torch import,
    and re-running them per test would quadruple the suite's cost for no extra
    information. Every assertion below reads the same four results.
    """
    return {
        det: {th: _run_child(threads=th, deterministic=det) for th in (1, 2)}
        for det in (True, False)
    }


class TestEndToEndThreadCount:
    """The card's own experiment, on this repo's model, in a subprocess.

    A subprocess because the thread count is a process-wide pool: measuring it
    in-process would be measuring whatever the previous test left behind. It
    also pins the fix the only way that counts -- `src` is imported from the
    tree under test, and the child refuses to run if it did not get it.

    Two of the three directions are asserted, and the split is deliberate:

    * WITH determinism, two offered thread counts must agree bitwise. Asserted:
      both runs are pinned to one thread and so perform the same computation
      in the same order on any host. This is the direction the fix owns, and it
      is the one that goes red if the pin is reverted on a host where the
      thread count matters.
    * The child really did HONOUR the thread count it was offered. Asserted:
      without it the whole 1-vs-2 framing is a claim about two runs that
      actually ran on one thread, which is exactly the kind of assumption this
      repo's audit notes keep finding.
    * WITHOUT determinism, two offered thread counts may or may not differ.
      That is a property of the host's BLAS kernel selection, not of this repo,
      so it is measured and printed rather than asserted. The measurement for
      this host is in `.agent-notes/task-24.md`.
    """

    def test_determinism_makes_two_thread_counts_agree_bitwise(self, child_runs):
        one = child_runs[True][1]
        two = child_runs[True][2]

        assert one["threads"] == 1 and two["threads"] == 1, (
            f"determinism did not pin the pool: the children launched with 1 and 2 "
            f"threads ran on {one['threads']} and {two['threads']} threads"
        )
        assert one["sha"] == two["sha"], (
            "two runs of identical code, identical seed and identical data "
            "differed across the whole state dict once the thread count was "
            "allowed to differ. This is the defect this card is about."
        )

    def test_the_experiment_actually_varied_the_thread_count(self, child_runs):
        """Premise check. If the child ignored OMP_NUM_THREADS, the pair above
        would be two identical runs and its agreement would prove nothing."""
        assert child_runs[False][1]["threads"] == 1
        assert child_runs[False][2]["threads"] == 2, (
            "a child launched with OMP_NUM_THREADS=2 ran on "
            f"{child_runs[False][2]['threads']} threads, so the 1-vs-2 comparison "
            "is not testing what it claims to test on this host"
        )
        print(
            "\n  [task-24] non-deterministic pair: "
            f"1 thread sha={child_runs[False][1]['sha'][:16]} "
            f"2 threads sha={child_runs[False][2]['sha'][:16]} "
            f"differ={child_runs[False][1]['sha'] != child_runs[False][2]['sha']}"
        )

    def test_both_children_imported_the_tree_under_test(self, child_runs):
        """The stale-clone hazard, asserted rather than assumed.

        `pip install -e .` maps `src` to a checkout that is not this one via a
        setuptools meta_path finder, and that finder wins whenever the running
        script's own directory has no `src`. On a card about reproducibility,
        measuring the wrong checkout is the worst available failure, so the
        check sits in the assertion path rather than in a comment.
        """
        for det, per_threads in child_runs.items():
            for out in per_threads.values():
                assert Path(out["src"]).resolve() == (REPO / "src" / "__init__.py"), (
                    f"a child (deterministic={det}) imported src from {out['src']}, "
                    f"not from {REPO}. Every number this file produces would be "
                    "from another checkout."
                )
