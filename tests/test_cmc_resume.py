# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Resume safety of the CMC second-mask stream.

TASK-02 gave `src/models/cmc.py` a private generator so the second mask would
stop consuming the global torch RNG. It made that generator a module global, and
a module global's POSITION is not a tensor, so it never reached `state_dict` and
`src/train.py::_capture_rng_state` could not save it. `defaults.yaml` sets
`use_cmc: true`, so the broken path was the default path.

The gate was green through all of it, for two independent reasons worth
remembering when writing the next test:

  1. `tests/test_checkpoint_fidelity.py`'s `_model_config()` never sets
     `use_cmc`, so the CMC branch never executed.
  2. Its "fresh process" branch runs in the SAME interpreter, which would hide a
     module-global bug even if that branch did execute.

So the tests below are written to FAIL on that design, not merely to describe
it. The decisive one is `test_resume_reproduces_the_mask_sequence`, which
genuinely uses two interpreters.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import yaml

from src.models.cmc import CrossMaskConsistency

REPO = Path(__file__).resolve().parent.parent

# A child program that draws the mask for a sequence of steps and prints one
# hash per line, so the parent can compare sequences across processes.
_CHILD = textwrap.dedent(
    """
    import hashlib
    import sys

    import torch

    from src.models.cmc import CrossMaskConsistency

    torch.manual_seed(1337)
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    out = []
    for step in range(lo, hi, 10):
        m = CrossMaskConsistency.generate_second_mask(
            seq_len=48, batch_size=3, mask_ratio=0.35, step=step
        )
        out.append(hashlib.sha256(m.cpu().numpy().tobytes()).hexdigest()[:16])
    print(chr(10).join(out))
    """
)


def _draw(step: int) -> str:
    mask = CrossMaskConsistency.generate_second_mask(
        seq_len=48, batch_size=3, mask_ratio=0.35, step=step
    )
    return hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()[:16]


def _run_child(lo: int, hi: int) -> list[str]:
    """Draw masks for steps in [lo, hi) in a genuinely separate interpreter."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(lo), str(hi)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO),
    )
    return proc.stdout.split()


def _call_args(source: str, marker: str) -> str:
    """Return the argument text of the first call to `marker`.

    Balanced-paren scan rather than ``index(")")``: the CMC call contains
    ``mask_ratio=mask_positions.float().mean().item()``, so the first closing
    paren sits in the middle of an argument and a naive split truncates.
    """
    start = source.index(marker) + len(marker)
    depth = 1
    out: list[str] = []
    for ch in source[start:]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out)


class TestCMCMaskIsAPureFunctionOfStep:
    def test_same_step_gives_the_same_mask_however_many_draws_preceded_it(self):
        """The property the stateful design violated.

        On the old design the mask for a given position depended on how many
        masks had already been drawn, so replaying step 20 after a fresh start
        produced something different. With a step-indexed derivation there is no
        position to depend on.
        """
        first = _draw(20)
        for s in range(7):
            _draw(99 + s)
        assert _draw(20) == first, (
            "the mask for a given step is not reproducible, so a resumed run "
            "diverges. This is the regression TASK-02 introduced."
        )

    def test_consecutive_steps_give_different_masks(self):
        """The counterexample to the previous test: pure must not mean constant."""
        assert _draw(0) != _draw(10)

    def test_the_default_path_does_not_touch_the_global_torch_stream(self):
        """The original TASK-02 goal, which must survive the rewrite."""
        torch.manual_seed(4242)
        before = torch.random.get_rng_state()
        _draw(0)
        assert torch.equal(before, torch.random.get_rng_state())

    def test_an_explicit_generator_still_takes_priority(self):
        """The caller-owned path must remain untouched by the rewrite."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(11)
        CrossMaskConsistency.generate_second_mask(
            seq_len=32, batch_size=2, mask_ratio=0.35, rng=gen
        )
        after = gen.get_state()
        CrossMaskConsistency.generate_second_mask(
            seq_len=32, batch_size=2, mask_ratio=0.35, rng=gen
        )
        assert not torch.equal(after, gen.get_state()), (
            "a caller-owned generator must advance; the rewrite must not have "
            "rerouted it through the step derivation"
        )

    def test_an_explicit_seed_is_honoured(self):
        a = CrossMaskConsistency.generate_second_mask(
            seq_len=32, batch_size=2, mask_ratio=0.35, seed=5
        )
        b = CrossMaskConsistency.generate_second_mask(
            seq_len=32, batch_size=2, mask_ratio=0.35, seed=5
        )
        c = CrossMaskConsistency.generate_second_mask(
            seq_len=32, batch_size=2, mask_ratio=0.35, seed=6
        )
        assert torch.equal(a, b)
        assert not torch.equal(a, c)


class TestResumeReproducesTheMaskSequence:
    def test_resume_reproduces_the_mask_sequence(self):
        """The decisive test: a fresh interpreter is what a resume actually is.

        Draw steps 0..40 in one interpreter, then draw 0..20 and 20..40 in two
        SEPARATE ones, and require the halves to match the whole. On the
        module-global design the second interpreter starts its generator at
        position 0 and replays steps 0 and 10, so the halves disagree.
        """
        whole = _run_child(0, 40)
        first_half = _run_child(0, 20)
        second_half = _run_child(20, 40)

        assert len(whole) == 4, whole
        assert len(first_half) == 2, first_half
        assert len(second_half) == 2, second_half
        assert whole == first_half + second_half, (
            "a fresh interpreter replayed the mask sequence instead of "
            "continuing it, so a resumed run trains on different masks than the "
            "uninterrupted one"
        )

    def test_the_training_path_actually_passes_the_step(self):
        """Guards the guarantee structurally.

        The convenience path (no step) is process state and is NOT resume-exact.
        It is fine for tests and inspection, so nothing forces the training loop
        onto the correct path except this assertion. If someone drops the
        `step=` argument, the test above still passes, because it calls the
        function directly; this is what catches it.
        """
        src = (REPO / "src" / "train.py").read_text(encoding="utf-8")
        assert "generate_second_mask(" in src, "the CMC call site moved"
        args = _call_args(src, "generate_second_mask(")
        assert "step=global_step" in args, (
            "the production CMC call no longer passes step=, so training sits on "
            "the non-resume-exact convenience path"
        )


class TestWhyThisWasHighSeverity:
    def test_use_cmc_is_on_by_default(self):
        """Why the regression was HIGH and not LOW.

        If CMC were off by default the broken path would be a rarely-taken branch
        and the severity a footnote. It is on, so every default training run
        executed it.
        """
        defaults = yaml.safe_load((REPO / "defaults.yaml").read_text(encoding="utf-8"))
        assert defaults["model"]["use_cmc"] is True

    def test_cmc_interval_makes_the_draw_rare_enough_to_hide(self):
        """The default draws once per 10 steps, so a short test can miss it."""
        defaults = yaml.safe_load((REPO / "defaults.yaml").read_text(encoding="utf-8"))
        assert defaults["model"]["cmc_interval"] > 1

    @pytest.mark.parametrize("interval", [1, 10])
    def test_a_draw_never_needs_the_checkpoint_to_be_resumable(self, interval):
        """Property: the draw depends only on its inputs, never on hidden state.

        Two draws with identical inputs are identical even when the module was
        used in between. That is what makes this path independent of the
        checkpoint format, which is the point of the rewrite.
        """
        first = _draw(interval)
        for s in range(5):
            _draw(1000 + s)
        assert _draw(interval) == first


class TestNoStatefulSingletonRemains:
    def test_the_module_global_generator_is_gone(self):
        """Structural guard against the exact regression returning.

        A future refactor that reintroduces a module-level generator would pass
        every behavioural test above only if the step path were also kept, so
        this asserts the mechanism itself is absent.
        """
        src = (REPO / "src" / "models" / "cmc.py").read_text(encoding="utf-8")
        assert "_mask_rng_default" not in src, (
            "a module-global generator is back in src/models/cmc.py; its position "
            "is not checkpointable and a resume will replay the mask sequence"
        )
        assert "def _default_mask_rng" not in src
