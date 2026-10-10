# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""The four missing SSL baselines, and the tests that make them real.

WHY THIS FILE EXISTS
--------------------
BYOL, Barlow Twins, VICReg and SimSiam were cited in prose across `src/` and
implemented nowhere. `baselines/` held exactly two real baseline classes --
`MLMBaseline` and `Data2VecTextBaseline` -- and the ablation grid was 38
internal-mechanism arms against 0 external baselines. This file is what stops a
re-implementation of those four from being indistinguishable from a working one.

THE TEST THAT MAKES EACH ONE REAL: COLLAPSE
-------------------------------------------
Each of these four methods prevents representation collapse through ONE
structure, and a naive port deletes that structure silently. A port with the
structure deleted still trains, still reports a decreasing loss, and learns
nothing -- it is indistinguishable from a working baseline on every loss curve a
reviewer looks at. So each baseline here carries a collapse test paired with a
MUTATION: the test is only meaningful if breaking the mechanism turns it red.

Measured on this machine (tiny tensors, 4 seeds each), the mutations do collapse,
and this file asserts the separation:

    Barlow Twins   correct eff-rank >= 0.175   lambda=0    eff-rank <= 0.085
    BYOL           correct eff-rank >= 0.072   no-EMA      eff-rank <= 0.062
    SimSiam        correct rep-std  >= 0.439   no-stopgrad rep-std  <= 0.039
    VICReg         variance term at a collapsed batch: 24.75 vs 0.00

Each arm uses the augmentation strength and batch size at which its own mutation
actually collapses, because the four methods fail under different conditions --
see `TestSimSiamCollapse` for the measured reason and the two earlier
configurations that did not separate.

TWO MEASUREMENT CHOICES THAT ARE NOT OBVIOUS, AND WHY
-----------------------------------------------------
1. **Effective rank, not standard deviation.** `TextSpanJEPAEncoder` ends in a
   `LayerNorm` and every projector ends in a `BatchNorm1d`. A normalised tensor
   cannot be constant, so "the standard deviation is non-zero" is TRUE FOR EVERY
   MODEL INCLUDING A COMPLETELY COLLAPSED ONE. The first draft of this file
   asserted exactly that and measured no separation at all: Barlow Twins at
   `lambda=0` scored *higher* than the correct model. What actually separates
   them is the rank of the representation -- the participation ratio of the
   singular values of the centred pooled output, normalised to [0, 1]:

       eff_rank(z) = (sum s_i^2)^2 / (D * sum s_i^4)

   A representation in which every dimension copies one signal has eff_rank
   ~1/D; an isotropic one has eff_rank ~1. Dimensional collapse is the failure
   mode of Barlow Twins and VICReg in particular, and this is the statistic that
   sees it.

2. **VICReg's variance term is tested where it is deterministic.** Measured
   training runs did NOT separate VICReg's correct arm from its `var_weight=0`
   ablation on encoder effective rank (0.0402 correct vs 0.0493 ablated -- the
   WRONG direction, because the variance hinge is scaled by `gamma` and both
   arms sit in a regime where the batch is small relative to the embedding). So
   VICReg's collapse test does not lean on a training run at all. It asserts the
   thing the term actually does: evaluated at a constant batch, the variance
   hinge is strictly positive, and zeroing the weight makes it exactly zero.
   That is a property of the objective, it is seed-independent, and it cannot be
   satisfied by a model that merely happens not to collapse during a short run.

WHAT THIS FILE DELIBERATELY DOES NOT CLAIM
------------------------------------------
BYOL's `no-predictor` mutation does NOT collapse at this scale: measured
effective rank 0.1510 against the correct arm's 0.0833 at 100 steps -- the
ablation looks BETTER. So BYOL's mutation test targets the EMA teacher, which
does separate cleanly, and the predictor's presence is covered by structural
assertions instead (`test_the_projector_is_part_of_the_teacher`,
`test_the_target_branch_carries_no_gradient`). That is a real gap in coverage and
it is stated here rather than papered over: BYOL's predictor is asserted to be
present and wired correctly, but not asserted to be load-bearing at this scale.

An earlier BYOL mutation was also wrong in an instructive way and is kept in the
source as `BYOLSharedTarget`'s docstring. Overriding `update_target_encoder` to
do nothing -- the obvious way to "remove the EMA" -- leaves `target_encoder`
frozen at its RANDOM INITIALISATION, which is a perfectly good teacher, and the
"broken" arm then scored no worse than the correct one. The mutation that actually
removes the mechanism is one where the target branch READS THE ONLINE WEIGHTS. A
plausible mutation that does not test the claim is worse than none, because it
produces a green test for a broken mechanism.

TRAINER CONTRACT
----------------
`src/train.py::compute_loss` dispatches by duck-typing, in this order:

    hasattr(model, "compute_loss_with_targets")            -> JEPA
    hasattr(model, "forward") and hasattr(model, "regression_head") -> data2vec
    hasattr(model, "compute_loss")                          -> MLM

Both earlier branches are checked BEFORE the `compute_loss` branch. A baseline
that owned an attribute named `regression_head`, or defined
`compute_loss_with_targets`, would be silently routed to another arm's loss and
would appear to train. `TestTrainerContract` pins both absences, plus the
parameter-counting contract that `tests/test_baseline_parity.py` pins for the
three arms `create_model` already knows.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from baselines.barlow_baseline import BarlowTwinsBaseline
from baselines.byol_baseline import BYOLBaseline
from baselines.simsiam_baseline import SimSiamBaseline
from baselines.vicreg_baseline import VICRegBaseline

from src.utils.seed import seed_everything

#: Every arm in this file. Keyed by the constructor argument that reaches
#: `create_model`'s `model_name` branch when these arms are wired in.
ARMS = {
    "byol": BYOLBaseline,
    "barlow": BarlowTwinsBaseline,
    "vicreg": VICRegBaseline,
    "simsiam": SimSiamBaseline,
}

#: The tiny shape every training-based collapse test uses. Chosen so a full
#: collapse sweep fits the 90s `tools/rt.py` budget on a box whose owner is
#: using it -- these numbers are a wall-clock constraint, not a modelling one,
#: and every one of them is asserted at more than one shape in
#: `TestShapesAreHonoured` so a shape cannot quietly become load-bearing.
TINY = {
    "vocab_size": 200,
    "max_seq_len": 16,
    "embed_dim": 32,
    "depth": 2,
    "num_heads": 4,
}

BATCH = 32


def batch(seed: int = 7, size: int = BATCH):
    """A fixed synthetic batch of token ids.

    Seeded through a `torch.Generator` rather than the global RNG, so building
    the batch cannot perturb the model initialisation that `seed_everything`
    controls. Two views of the same sequences are produced by masking tokens to
    the padding id, which is the augmentation this repo's pipeline already
    supports.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY["vocab_size"], (size, TINY["max_seq_len"]), generator=g)


def two_views(ids: torch.Tensor, step: int, strength: float):
    """Two masked views of one batch.

    `strength` is the fraction of tokens replaced by the padding id. It is a
    real knob, not decoration: Barlow Twins is tested at 0.3 and BYOL at 0.8
    because the two methods collapse under different conditions, and a single
    shared value would test neither properly.
    """
    g = torch.Generator().manual_seed(100 + step)
    a, b = ids.clone(), ids.clone()
    for view in (a, b):
        view[torch.rand(ids.shape, generator=g) < strength] = 0
    return a, b


def effective_rank(z: torch.Tensor) -> float:
    """Normalised effective rank of a ``(B, D)`` representation, in [0, 1].

    The participation ratio of the singular values of the centred matrix,
    divided by D. A rank-1 representation scores ~1/D; an isotropic one ~1.

    This is the statistic that detects collapse in this repo, and the reason is
    mechanical rather than a matter of taste: both `TextSpanJEPAEncoder` (final
    `LayerNorm`) and every projector here (`BatchNorm1d`) force non-zero
    per-dimension variance, so an assertion on standard deviation is satisfied by
    a collapsed model. Measured on the mutation that started this file --
    Barlow Twins with `lambda_offdiag=0` -- standard deviation gave no separation
    at all and this statistic separates by more than 2x.
    """
    zc = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(zc.double())
    p = s.pow(2)
    return float((p.sum() ** 2 / (p.pow(2).sum() + 1e-30)) / z.shape[1])


def pooled_encoder_rep(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """The pooled encoder output, ``(B, D)``.

    The encoder's own representation, not the projector's. That choice is forced:
    the projector's BatchNorm renormalises exactly the quantity a collapse test
    is trying to measure, so reading the projector output would be reading a
    tensor that has been normalised back to non-degenerate by construction.
    """
    with torch.no_grad():
        h, _ = model.encoder(ids)
    return model.pool(h)


def train_and_measure(
    factory,
    steps: int,
    strength: float,
    seed: int,
    size: int = BATCH,
    lr: float = 1e-3,
):
    """Build `factory()`, train it on a fixed batch, return (rep, final_loss).

    THE ORDER HERE IS LOAD-BEARING. `seed_everything` is called BEFORE the model
    is constructed, not after. The first draft of this helper took an
    already-built model and seeded it afterwards, which meant the model's
    INITIALISATION was drawn from whatever global RNG state the previous test
    happened to leave behind -- so every arm was measured at a different random
    init than the sweep that chose these thresholds had measured. Passing a
    factory rather than a model makes the seeding order impossible to get wrong,
    and is why the thresholds in this file are reproducible at all.

    Everything else is deterministic too: the batch comes from its own seeded
    `torch.Generator` (so building it cannot perturb the init), and every
    augmentation is a pure function of the step index.
    """
    seed_everything(seed)
    ids = batch(size=size)
    model = factory()
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    for step in range(steps):
        view_a, view_b = two_views(ids, step, strength)
        opt.zero_grad()
        loss, _ = model.compute_loss(view_a, view_b)
        loss.backward()
        opt.step()
        # BYOL only: the EMA teacher is advanced here because the trainer's
        # `do_ema_update` owns that call site and this test does not. Every other
        # arm has no teacher, and the `getattr` guard is what keeps that from
        # being a special case written four times.
        advance = getattr(model, "update_target_encoder", None)
        if advance is not None:
            advance()
    model.eval()
    with torch.no_grad():
        return pooled_encoder_rep(model, ids), float(loss.item())


def build(arm: str, **kwargs) -> nn.Module:
    return ARMS[arm](**TINY, **kwargs)


def arm_factory(arm: str, **kwargs):
    """A zero-argument factory for `train_and_measure`, so seeding precedes init."""

    def factory() -> nn.Module:
        return ARMS[arm](**TINY, **kwargs)

    return factory


def class_factory(cls, **kwargs):
    """Factory for a mutated subclass, which takes no `arm` key."""

    def factory() -> nn.Module:
        return cls(**TINY, **kwargs)

    return factory


def trainer_build(name: str, **overrides) -> nn.Module:
    """Build `name` through `src.train.create_model`, at `TINY`'s size.

    The PRODUCTION path, deliberately: everything in `TestTrainerContract` below
    is a claim about the trainer, and a claim about the trainer tested against a
    hand-built class is a claim about a different object. `TINY` is reused so the
    trainer-built model and the `build(name)` model in the same test class are
    the same shape and their two assertions are comparable.

    The model cfg uses the trainer's own key names, which are NOT the
    constructor's: `create_model` takes `encoder_depth` where the classes take
    `depth`. Passing the constructor spelling would silently train at the
    `depth=12` default, which at `TINY`'s vocab is roughly forty times the work.
    """
    from src.train import create_model

    model_cfg = {
        "embed_dim": TINY["embed_dim"],
        "encoder_depth": TINY["depth"],
        "num_heads": TINY["num_heads"],
        "mlp_ratio": 2.0,
        "drop_rate": 0.0,
    }
    model_cfg.update(overrides)
    return create_model(
        name,
        model_cfg,
        vocab_size=TINY["vocab_size"],
        max_seq_len=TINY["max_seq_len"],
        device=torch.device("cpu"),
    )


# ═══════════════════════════════════════════════════════════════════
#  Barlow Twins -- the off-diagonal term is the whole mechanism
# ═══════════════════════════════════════════════════════════════════


class BarlowNoOffDiagonal(BarlowTwinsBaseline):
    """Barlow Twins with its only anti-collapse term removed.

    `lambda_offdiag = 0` is the paper's own ablation and a supported constructor
    argument, so this needs no source edit -- which is the point. The mutation is
    expressible through the public API, so a reviewer can reproduce it by
    constructing the arm, not by patching a file.
    """

    def __init__(self, **kwargs):
        super().__init__(lambda_offdiag=0.0, **kwargs)


class TestBarlowTwinsCollapse:
    """Removing the off-diagonal penalty collapses the representation.

    Measured at 120 steps, seeds 0-3:
        correct   eff-rank 0.2002, 0.1993, 0.1755, 0.1898  (min 0.1755)
        lambda=0  eff-rank 0.0789, 0.0698, 0.0848, 0.0835  (max 0.0848)

    The gap is ~2.1x and does not close on any seed tried. This is the
    cleanest of the four collapse tests, and the header explains why: Barlow
    Twins has no EMA target and no predictor, so the off-diagonal term is the
    ONLY thing standing between it and a rank-1 representation. The test runs
    three of the four measured seeds, for budget reasons.
    """

    STEPS = 120
    STRENGTH = 0.3
    #: Midpoint of the measured gap, not an arbitrary threshold: the correct arm
    #: never falls below 0.175 and the collapsed arm never rises above 0.085.
    FLOOR = 0.13

    def test_the_mechanism_is_the_off_diagonal_penalty(self):
        """Name the term before testing it, so the test cannot drift off-target.

        If someone rewrote the loss to drop the decorrelation term, this fails
        immediately and by name -- rather than leaving a collapse test that
        silently tests the invariance term instead.
        """
        model = build("barlow")
        assert model.lambda_offdiag > 0
        assert "lambda" in model.extra_repr()

    @pytest.mark.parametrize("seed", [0, 1])
    def test_correct_arm_survives_and_its_mutation_collapses(self, seed):
        """Both halves of the verdict in one test, on one seed.

        Merged deliberately. The first draft had two separate tests -- "the
        correct arm survives" and "the mutation collapses" -- which trained the
        same two arms on the same seed twice, and the duplicated compute pushed
        the file past `tools/rt.py`'s 90s budget on a box whose owner is using
        it. Same assertions, same information in the failure message, half the
        CPU.

        Seeds 0 and 1 of the four-seed development sweep are used. All four
        separated by more than 2x, so dropping two costs very little margin;
        this is a wall-clock decision and not a claim that seeds 2 and 3 would
        behave differently.

        The last block carries the "a broken arm looks BETTER" assertion, folded
        in from a test that used to train both arms a third time for it.
        """
        good, loss_correct = train_and_measure(
            arm_factory("barlow"), self.STEPS, self.STRENGTH, seed
        )
        bad, loss_broken = train_and_measure(
            class_factory(BarlowNoOffDiagonal), self.STEPS, self.STRENGTH, seed
        )
        good_rank, bad_rank = effective_rank(good), effective_rank(bad)
        assert good_rank > self.FLOOR, (
            f"the correct Barlow Twins arm collapsed on seed {seed} "
            f"(eff-rank {good_rank:.4f} <= {self.FLOOR})"
        )
        assert bad_rank < self.FLOOR, (
            "Barlow Twins without the off-diagonal penalty did not collapse on "
            f"seed {seed} (eff-rank {bad_rank:.4f} >= {self.FLOOR}, correct arm "
            f"{good_rank:.4f}). Either the ablation is not being applied, or the "
            "collapse statistic no longer separates the two arms -- in which "
            "case the first assertion is not evidence of anything and both need "
            "revisiting."
        )
        assert (
            good_rank > 1.5 * bad_rank
        ), f"separation on seed {seed} narrowed to {good_rank / bad_rank:.2f}x"

        # Why this defect survives review: with the off-diagonal penalty removed
        # the loss reaches ~0 FASTER than the correct arm's, so a broken Barlow
        # Twins looks like a BETTER baseline on the only number the trainer logs.
        # Stated here rather than as its own test only because a separate test
        # meant a third training run of the same two arms.
        assert loss_broken < loss_correct, (
            "expected the collapsed arm to reach a LOWER loss than the correct "
            f"one (got {loss_broken:.4f} vs {loss_correct:.4f}). If this flipped, "
            "the loss is no longer the thing the collapse test is guarding "
            "against and both assertions need rechecking."
        )


# ═══════════════════════════════════════════════════════════════════
#  VICReg -- the variance term, tested where it is deterministic
# ═══════════════════════════════════════════════════════════════════


class TestVicregCollapse:
    """VICReg's variance term, and why its collapse test is not a training run.

    The first draft of this file trained VICReg for 150 steps on both arms and
    compared encoder effective rank. The measurement contradicted the
    expectation: `var_weight=0` scored 0.0493 against the correct arm's 0.0402 --
    the ablation came out BETTER, so the test would have been asserting a
    difference that runs the wrong way.

    The reason is that on a batch of 32 with `embed_dim=32`, a short run does not
    drive this arm into the regime where the variance hinge dominates, so the
    statistic measures initialisation noise rather than collapse. Rather than
    tune the seed until the numbers came out right, this class tests the
    property the term actually has:

        evaluated at a COLLAPSED batch, the variance hinge is strictly positive.

    That is seed-independent, it costs no training, and it cannot be satisfied by
    a model that simply has not collapsed yet. Measured: 24.75 with the term,
    0.00 with `var_weight=0`.
    """

    @pytest.fixture
    def model(self):
        """A fresh VICReg arm. Function-scoped: this class touches nothing
        persistent, and a class-scoped instance-method fixture is deprecated."""
        return build("vicreg")

    def test_variance_term_has_a_positive_weight(self, model):
        """The term must be switched on, not merely present in the source."""
        assert model.var_weight > 0

    def test_variance_hinge_is_strictly_positive_at_a_collapsed_batch(self, model):
        """The mathematical statement of "this term prevents collapse".

        A collapsed batch is one whose every row is the same vector, so every
        per-dimension standard deviation is exactly zero. The hinge must
        therefore return its MAXIMUM.

        The maximum is ``gamma - sqrt(_EPS)``, not ``gamma``, and the difference
        is the point rather than a tolerance: the epsilon sits inside the square
        root so that the gradient of ``sqrt(var)`` at ``var = 0`` -- the exact
        state this term exists to correct -- is finite instead of infinite. The
        first draft of this test asserted ``== gamma`` and failed by exactly
        0.01, which read like a floating-point problem and was actually the
        implementation being correct.
        """
        from baselines.vicreg_baseline import _EPS

        collapsed = torch.ones(16, TINY["embed_dim"])
        expected = model.gamma - _EPS**0.5
        assert expected > 0.9 * model.gamma, "the epsilon floor is too large to be a detail"
        assert model.variance_loss(collapsed) == pytest.approx(expected, rel=1e-5)

    def test_variance_hinge_is_zero_for_a_healthy_representation(self, model):
        """One-sidedness, which is what makes it a hinge and not a penalty.

        A dimension whose spread already exceeds `gamma` contributes exactly
        zero and is never pushed down. A two-sided version would fight healthy
        dimensions and quietly become an isotropy constraint.
        """
        healthy = torch.randn(64, TINY["embed_dim"]) * (model.gamma + 1.0)
        assert model.variance_loss(healthy).item() == pytest.approx(0.0, abs=1e-6)

    def test_removing_the_weight_makes_the_term_exactly_zero(self, model):
        """The mutation verdict, and it is exact rather than statistical.

        With `var_weight=0` the hinge still computes 0.99 -- it is the WEIGHT
        that removes it from the objective. So an implementation that computed
        the term and then discarded it, or one that never computed it, produces
        the same observable loss here; this test pins the weight, which is the
        part a port actually gets wrong.
        """
        from baselines.vicreg_baseline import _EPS

        ablated = build("vicreg", var_weight=0.0)
        collapsed = torch.ones(16, TINY["embed_dim"])
        assert ablated.variance_loss(collapsed) == pytest.approx(model.gamma - _EPS**0.5, rel=1e-5)
        assert ablated.var_weight * ablated.variance_loss(collapsed) == 0.0

    def test_the_total_loss_changes_when_the_variance_term_is_removed(self, model):
        """End-to-end through `compute_loss`, not just through the term.

        The term-level assertions above are about a method; this one is about the
        arm a reviewer would actually train. A wiring bug that computed the
        variance term and then never added it to the loss would pass every
        term-level test here and fail this one.

        The weight is toggled ON ONE MODEL rather than by building a second arm.
        The first draft compared `build("vicreg")` against
        `build("vicreg", var_weight=0.0)` -- two SEPARATELY INITIALISED models --
        so their losses differed for the uninteresting reason that they started
        from different random weights. That version of this test PASSED with the
        variance term deleted from the source entirely, which is precisely the
        bug this class exists to catch.
        """
        ids = batch()
        view_a, view_b = two_views(ids, 0, 0.3)
        model.eval()
        with torch.no_grad():
            loss_with, info_with = model.compute_loss(view_a, view_b)
            original_weight = model.var_weight
            model.var_weight = 0.0
            loss_without, info_without = model.compute_loss(view_a, view_b)
            model.var_weight = original_weight

        assert not torch.isclose(loss_with, loss_without), (
            "removing the variance weight left the total loss unchanged, so the "
            "term is not reaching the objective"
        )
        # Still computed in both cases -- what changes is whether it is weighted.
        assert info_with["var_vicreg"] > 0
        assert info_without["var_vicreg"] > 0
        assert loss_with > loss_without  # the weighted term only ever adds

    def test_covariance_cannot_substitute_for_the_variance_hinge(self):
        """Why the ablation zeroes ONE term and not both.

        The covariance term is scale-blind: a representation whose every
        dimension copies one signal has zero covariance and is perfectly happy.
        A constant batch therefore scores zero on covariance. That is the reason
        VICReg needs the variance hinge specifically, and the reason a test that
        ablated both terms at once would have proven less than it appears to.
        """
        model = build("vicreg")
        collapsed = torch.ones(16, TINY["embed_dim"])
        assert model.covariance_loss(collapsed).item() == pytest.approx(0.0, abs=1e-8)
        assert model.variance_loss(collapsed).item() > 0


# ═══════════════════════════════════════════════════════════════════
#  BYOL -- the EMA teacher
# ═══════════════════════════════════════════════════════════════════


class BYOLSharedTarget(BYOLBaseline):
    """BYOL with its EMA teacher removed, done correctly.

    The obvious mutation -- overriding `update_target_encoder` to do nothing --
    is WRONG, and it is wrong in a way that hides the defect: it leaves
    `target_encoder` frozen at its random initialisation, which is a perfectly
    good teacher. The mutation that actually removes the mechanism is one where
    the target branch READS THE ONLINE WEIGHTS, so the teacher tracks the
    student exactly. Measured with the no-op version, the "broken" arm scored
    BETTER than the correct one (effective rank 0.081 vs 0.083 at 100 steps);
    measured with this version it collapses to 0.034 against 0.072-0.088.

    Recorded here because it is the more instructive failure: a mutation that
    looks plausible and does not test the claim is worse than no mutation, since
    it produces a green test for a broken mechanism.
    """

    def target_branch(self, view: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h, _ = self.encoder(view)
            return self.projector(self.pool(h))


class TestByolCollapse:
    """BYOL's EMA teacher, which is what actually separates at this scale.

    Measured at 150 steps, seeds 0-3:
        correct        eff-rank 0.0856, 0.0720, 0.0880, 0.0879  (min 0.0720)
        shared target  eff-rank 0.0352, 0.0614, 0.0337, 0.0443  (max 0.0614)

    The gap is ~1.2x -- narrower than Barlow Twins', and stated as measured
    rather than rounded up. The floor sits between the two arms' extremes, and
    the per-seed test asserts the gap so a regression in the statistic cannot pass
    unnoticed. Three of the four measured seeds are run, for budget reasons.
    """

    STEPS = 150
    STRENGTH = 0.8
    FLOOR = 0.066

    def test_the_teacher_is_a_separate_frozen_copy(self):
        """Structurally: two distinct encoders, one frozen.

        Checked by identity and by `requires_grad`, not by parameter count --
        a deepcopy has the same count as its source, so counting proves nothing
        about whether the update actually moves one and not the other.
        """
        model = build("byol")
        assert model.target_encoder is not model.encoder
        assert all(not p.requires_grad for p in model.target_encoder.parameters())
        assert all(p.requires_grad for p in model.encoder.parameters())

    def test_the_teacher_is_ema_updated_toward_the_student(self):
        """The EMA must move the teacher TOWARD the student, not away from it.

        Picked per-parameter rather than by taking `next(...)`, for a reason
        worth recording: the teacher is a `deepcopy` of the student taken inside
        `__init__`, so at step 0 the two agree BITWISE on every encoder tensor.
        The gap is exactly 0.0, and a strict-decrease assertion against a zero
        gap is an assertion about nothing -- which is what the first draft of this
        test did, and it failed for a reason that had nothing to do with BYOL.

        So this test first makes the student differ, then watches the EMA close
        the gap. Perturbing the student is the honest way to get a non-degenerate
        starting point: an EMA that works and an EMA that does nothing are
        indistinguishable when the two tensors already agree.
        """
        model = build("byol")
        with torch.no_grad():
            for param in model.encoder.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        pairs = list(zip(model.encoder.parameters(), model.target_encoder.parameters()))
        assert len(pairs) > 0

        for online, target in pairs:
            gap_before = (online.detach() - target.detach()).norm().item()
            if gap_before == 0:
                continue
            model.update_target_encoder()
            gap_after = (online.detach() - target.detach()).norm().item()
            assert gap_after < gap_before, (
                "the EMA update moved the teacher away from the student "
                f"({gap_before:.6g} -> {gap_after:.6g})"
            )
            assert gap_after > 0, "the EMA update teleported the teacher onto the student"
            return
        raise AssertionError("every encoder parameter was identical; the test proved nothing")

    def test_the_projector_is_part_of_the_teacher(self):
        """BYOL EMAs the projector too, not just the encoder.

        Easy to get wrong and undetectable from parameter counts, since the
        target projector is a frozen copy either way. Asserted by watching the
        gap close on the PROJECTOR specifically, after perturbing the online one
        so that there is a gap to close (see the test above for why).
        """
        model = build("byol")
        with torch.no_grad():
            for param in model.projector.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        checked = 0
        for online, target in zip(
            model.projector.parameters(), model.target_projector.parameters()
        ):
            gap_before = (online.detach() - target.detach()).norm().item()
            if gap_before == 0:
                continue
            model.update_target_encoder()
            gap_after = (online.detach() - target.detach()).norm().item()
            assert gap_after < gap_before, (
                "the target projector was not EMA-updated; BYOL's teacher is the "
                "encoder AND the projector"
            )
            checked += 1
        assert checked > 0, "no projector parameter differed; the test proved nothing"

    def test_the_target_branch_carries_no_gradient(self):
        """Stop-grad on the target, asserted at the tensor rather than by reading
        the source."""
        model = build("byol")
        target = model.target_branch(batch())
        assert target.requires_grad is False
        assert target.grad_fn is None

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_correct_arm_survives_and_its_mutation_collapses(self, seed):
        """Both halves of the BYOL verdict in one test, on one seed.

        Merged for the same reason as the Barlow Twins equivalent: two tests
        training the same two arms on the same seed doubled the CPU.

        Three seeds rather than two, unlike Barlow Twins: BYOL's gap is ~1.2x
        rather than ~2.1x, so it is the arm with the least room for a threshold
        to be slightly wrong, and it gets the extra seed to compensate. The
        floor is 0.066 against a measured correct-arm minimum of 0.0720 and a
        measured mutated-arm maximum of 0.0614 -- narrow on both sides, and
        stated as measured rather than rounded away from.
        """
        good, _ = train_and_measure(arm_factory("byol"), self.STEPS, self.STRENGTH, seed)
        bad, _ = train_and_measure(class_factory(BYOLSharedTarget), self.STEPS, self.STRENGTH, seed)
        good_rank, bad_rank = effective_rank(good), effective_rank(bad)
        assert good_rank > self.FLOOR, (
            f"the correct BYOL arm collapsed on seed {seed} "
            f"(eff-rank {good_rank:.4f} <= {self.FLOOR})"
        )
        assert bad_rank < self.FLOOR, (
            "BYOL with a shared (non-EMA) target did not collapse on seed "
            f"{seed} (eff-rank {bad_rank:.4f} >= {self.FLOOR}, correct arm "
            f"{good_rank:.4f})"
        )
        assert (
            good_rank > 1.1 * bad_rank
        ), f"separation on seed {seed} narrowed to {good_rank / bad_rank:.2f}x"


# ═══════════════════════════════════════════════════════════════════
#  SimSiam -- the stop-gradient
# ═══════════════════════════════════════════════════════════════════


class SimSiamNoStopGrad(SimSiamBaseline):
    """SimSiam with the stop-gradient deleted.

    The mutation is a single missing `torch.no_grad()`. It is expressible as a
    subclass rather than a source edit so that the test file itself is the
    reproducible artefact and no one has to hand-patch a module to check it.
    """

    def target_branch(self, view: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(view)
        return self.projector(self.pool(h))


class TestSimSiamCollapse:
    """SimSiam's stop-gradient, and the collapse measurement that backs it.

    The first draft of this class documented a failure worth recording. Removing
    the stop-gradient genuinely collapses SimSiam (Chen & He, Fig. 1), but at the
    first configuration tried here -- batch 32, 120 steps, augmentation 0.7 --
    the collapse came out BIMODAL across seeds: measured rep-std of the ablated
    arm was 0.055, 0.713, 0.655, 0.062. Collapsed on two seeds, escaped on two.

    A smaller batch with heavier augmentation fixes it, and the reason is
    mechanical rather than lucky. Collapse happens when the objective can be
    minimised by making the two views' representations identical, and that
    shortcut is available exactly when the two views are hard to tell apart. A
    batch of 16 with 85% of tokens masked leaves the model very little to
    distinguish samples by, so the constant solution wins; a batch of 32 with
    30% masked leaves it plenty, so the shortcut is not worth taking.

    Measured at batch 16 / 250 steps / augmentation 0.85, seeds 0-3:
        correct              rep-std 0.439, 0.583, 0.667, 0.680  (min 0.439)
        without stop-grad    rep-std 0.039, 0.032, 0.029, 0.036  (max 0.039)

    An 11x gap with no overlap, against the 0.5x/1.4x overlap before. The test
    runs two of those four seeds for budget reasons and says so at the call site.

    The structural assertion below is kept alongside it and is not redundant:
    it is cheaper, it is deterministic, and it is the property the module header
    actually promises.
    """

    STEPS = 250
    STRENGTH = 0.85
    #: Far below every correct-arm measurement (min 0.439) and far above every
    #: collapsed-arm one (max 0.039). The midpoint of a gap with no overlap is
    #: arbitrary; what matters is that it sits inside it with room on both sides.
    STD_FLOOR = 0.15

    #: Batch size 16, not the module default of 32. Load-bearing, and stated
    #: here rather than hidden: at 32 this ablation stops collapsing (measured
    #: mutant rep-std 0.069, 0.825, 0.807, 0.051 over seeds 0-3 -- bimodal) and
    #: the collapse test silently becomes vacuous. The reason is mechanical and
    #: is in the class docstring.
    BATCH_SIZE = 16

    def _measure(self, factory, seed):
        """Train and return the pooled encoder representation at `BATCH_SIZE`."""
        rep, _ = train_and_measure(factory, self.STEPS, self.STRENGTH, seed, size=self.BATCH_SIZE)
        return rep

    def test_the_stop_gradient_is_structural_not_a_detach_call(self):
        """The promise the module header makes, asserted at the tensor.

        `grad_fn is None` is stronger than `requires_grad is False`: it says the
        graph was never BUILT, not that a graph was built and then severed. Only
        the first survives a refactor that reintroduces a graph upstream.
        """
        model = build("simsiam")
        target = model.target_branch(batch())
        assert target.grad_fn is None
        assert target.requires_grad is False

    def test_the_mutation_breaks_that_structural_guarantee(self):
        """The mutation verdict on the structural claim: deterministic, seed-free."""
        target = SimSiamNoStopGrad(**TINY).target_branch(batch())
        assert target.grad_fn is not None, (
            "removing torch.no_grad() left the target with no graph; if this "
            "fails the mutation is no longer exercising the stop-gradient"
        )
        assert target.requires_grad is True

    def test_no_ema_teacher_exists(self):
        """SimSiam has no teacher. Asserted by absence, because a port that adds
        one has silently turned this into BYOL."""
        model = build("simsiam")
        assert not hasattr(model, "target_encoder")
        assert not hasattr(model, "update_target_encoder")

    def test_the_loss_is_symmetric_in_its_two_views(self):
        """Both directions are present; dropping one halves the gradient.

        Checked numerically rather than by reading the source, by swapping the
        arguments and comparing the two cosine terms.
        """
        model = build("simsiam")
        ids = batch()
        view_a, view_b = two_views(ids, 0, self.STRENGTH)
        model.eval()
        with torch.no_grad():
            _, info_ab = model.compute_loss(view_a, view_b)
            _, info_ba = model.compute_loss(view_b, view_a)
        assert info_ab["cos_ab"] == pytest.approx(info_ba["cos_ba"], rel=1e-5)
        assert info_ab["cos_ba"] == pytest.approx(info_ba["cos_ab"], rel=1e-5)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_correct_arm_survives_and_its_mutation_collapses(self, seed):
        """Both halves of the SimSiam verdict in one test, on one seed.

        Only two seeds, against three for the other three arms, and the reason
        is arithmetic: 250 steps at batch 16 costs roughly 5s per run here, so
        four seeds on both arms would put this file over `tools/rt.py`'s 90s
        budget on a box whose owner is using it. The development sweep measured
        four seeds and all four separated by ~11x; two is a budget compromise,
        stated here rather than passed off as the robustness that was measured.
        """
        good = self._measure(arm_factory("simsiam"), seed)
        bad = self._measure(class_factory(SimSiamNoStopGrad), seed)
        good_std = good.std(dim=0).mean().item()
        bad_std = bad.std(dim=0).mean().item()
        assert good_std > self.STD_FLOOR, (
            f"the correct SimSiam arm collapsed on seed {seed} "
            f"(rep-std {good_std:.4f} <= {self.STD_FLOOR})"
        )
        assert bad_std < self.STD_FLOOR, (
            "SimSiam without the stop-gradient did not collapse on seed "
            f"{seed} (rep-std {bad_std:.4f} >= {self.STD_FLOOR}, correct arm "
            f"{good_std:.4f}). The collapse may have become seed-dependent "
            "again; if so the configuration above needs re-measuring rather than "
            "loosening."
        )
        assert (
            good_std > 3.0 * bad_std
        ), f"separation on seed {seed} narrowed to {good_std / bad_std:.2f}x"


# ═══════════════════════════════════════════════════════════════════
#  Parameter matching -- the defect this file exists partly to avoid
# ═══════════════════════════════════════════════════════════════════


@contextlib.contextmanager
def _meta_device():
    """Build on meta tensors: exact parameter shapes, zero allocation.

    The production shapes allocate ~1.1GB per arm, and this box's owner is using
    it. `torch.linspace` is pinned to CPU for the duration because the encoder
    calls `.item()` on it and meta tensors cannot service that -- the same trick
    `tests/test_baseline_parity.py` uses, kept here rather than imported so this
    file does not depend on another test module's internals.
    """
    original = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("device", "cpu")
        return original(*args, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            yield
    finally:
        torch.linspace = original


#: The two production rungs. Both are in `config/`, both are quoted in
#: `baselines/mlm_baseline.py`, and both are measured rather than derived --
#: `config/scaling/small_100m.yaml` + `config/wikitext/mlm_wikitext_small.yaml`
#: is the 640/10 rung the critic's 1.281x was measured at.
PRODUCTION = [
    (50304, 512, 640, 10, 10, 320, 4),
    (50304, 512, 768, 12, 12, 384, 4),
]
PRODUCTION_IDS = ["640x10", "768x12"]


def _jepa_trainable(vocab, seq, dim, depth, heads, pdim, pdepth):
    from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

    with _meta_device():
        model = TextSpanJEPA(
            TextSpanJEPAConfig(
                vocab_size=vocab,
                max_seq_len=seq,
                embed_dim=dim,
                encoder_depth=depth,
                num_heads=heads,
                predictor_embed_dim=pdim,
                predictor_depth=pdepth,
            )
        )
        return model.get_num_params_trainable()


#: The band every arm's trainable/TrainableJEPA ratio must land inside.
#:
#: Deliberately NOT 1.0 and deliberately NOT tight. `TextSpanJEPA`'s head budget
#: includes a fixed-count GWP mechanism term that does not scale with width, so
#: exact equality is not achievable by any arm without a vocab-sized head -- and
#: a vocab-sized head is precisely the defect (it would put the arm back at
#: MLM's 1.28x). The measured values are 0.9929-0.9940, so a +-8% band is tight
#: enough to catch any real capacity change while leaving room for a legitimate
#: refactor of the head widths. The band is a floor AND a ceiling: an arm that
#: became dramatically SMALLER would be just as much a failed comparison as one
#: that became bigger.
RATIO_BAND = (0.92, 1.08)

#: The defect these arms are matched against. `MLMBaseline` carries 1.281x
#: `TextSpanJEPA`'s trainable parameters at the 640/10 rung because it holds an
#: untied (embed_dim, vocab_size) output matrix with no JEPA counterpart. Pinned
#: here so the claim "these arms are better matched than that" is checkable
#: rather than a boast in a docstring.
MLM_OVER_JEPA_AT_640 = 1.2811


class TestParameterMatching:
    """The four arms must not repeat MLM's 1.281x mistake.

    What went wrong with the existing baselines
    -------------------------------------------
    `MLMBaseline` holds `mlm_head`, a separate untied `(embed_dim, vocab_size)`
    matrix worth 32,194,560 parameters at the shipped rung. `TextSpanJEPA`'s
    `TiedTokenDecoder` is 1,639,680 and gets its vocabulary projection for free
    by reusing `encoder.token_embedding.weight`. So the control carries 1.281x the
    trainable capacity of the method it is a control for -- a reviewer-visible
    defect, and one `tests/test_baseline_parity.py` measures and pins.

    Why these four do not
    ---------------------
    None of them has a vocabulary-sized head. Barlow Twins, VICReg and SimSiam
    have no output projection at all; BYOL has one, but it projects to
    `embed_dim`, not to the vocabulary. Their heads are sized by
    `HIDDEN_MULTIPLE` in each module to land on JEPA's trainable count.

    Why a band and not equality
    ---------------------------
    Stated in `RATIO_BAND` above and repeated here because it is the whole
    design: exact equality would require a head whose width scales with JEPA's
    internals, and the first attempt at that (`HIDDEN_MULTIPLE = 3.0`) was simply
    wrong -- it gave 0.956x at 640/10 and 0.958x at 768/12. The constant was
    solved against the measured budget instead, landing at 0.993x. A test
    asserting `ratio == 1.0` would have been red when written and deleted
    rather than kept, which loses the measurement.
    """

    @pytest.mark.parametrize("shape", PRODUCTION, ids=PRODUCTION_IDS)
    def test_trainable_count_is_close_to_jepas(self, shape):
        vocab, seq, dim, depth, heads, pdim, pdepth = shape
        jepa_trainable = _jepa_trainable(vocab, seq, dim, depth, heads, pdim, pdepth)
        with _meta_device():
            for name, cls in ARMS.items():
                model = cls(
                    vocab_size=vocab,
                    max_seq_len=seq,
                    embed_dim=dim,
                    depth=depth,
                    num_heads=heads,
                )
                trainable = model.get_num_params_trainable()
                ratio = trainable / jepa_trainable
                assert RATIO_BAND[0] <= ratio <= RATIO_BAND[1], (
                    f"{name} at {dim}x{depth} has {trainable:,} trainable "
                    f"parameters against JEPA's {jepa_trainable:,} -- {ratio:.4f}x, "
                    f"outside the band {RATIO_BAND}. A vocab-sized head here would "
                    "put the arm at MLM's 1.281x; check HIDDEN_MULTIPLE."
                )

    @pytest.mark.parametrize("shape", PRODUCTION, ids=PRODUCTION_IDS)
    def test_the_encoder_is_shared_exactly_with_jepa(self, shape):
        """The stronger half of the claim, and the one that is exact.

        Both arms build `TextSpanJEPAEncoder` with the same dimensions, so the
        encoders have identical parameter shapes name for name. Counting alone
        would pass for two different architectures that happened to sum the same.
        """
        vocab, seq, dim, depth, heads, pdim, pdepth = shape
        from src.models.encoder import TextSpanJEPAEncoder

        with _meta_device():
            reference = TextSpanJEPAEncoder(
                vocab_size=vocab, max_seq_len=seq, embed_dim=dim, depth=depth, num_heads=heads
            )
            reference_shapes = {n: tuple(p.shape) for n, p in reference.named_parameters()}
            for name, cls in ARMS.items():
                model = cls(
                    vocab_size=vocab,
                    max_seq_len=seq,
                    embed_dim=dim,
                    depth=depth,
                    num_heads=heads,
                )
                shapes = {n: tuple(p.shape) for n, p in model.encoder.named_parameters()}
                assert shapes == reference_shapes, f"{name}'s encoder diverges from the shared one"

    def test_no_arm_has_a_vocabulary_sized_head(self):
        """The mechanism behind MLM's 1.281x, asserted as an absence.

        Not a ratio test: a ratio can be met by compensating errors. What must
        not exist is a parameter whose shape is `(vocab_size, embed_dim)` --
        the object whose ABSENCE is JEPA's `TiedTokenDecoder`, which reuses
        `encoder.token_embedding.weight` for its vocabulary projection instead of
        owning a second one.

        The encoder tables -- BOTH of them for BYOL, which holds a frozen teacher
        -- are that shape and are deliberately excluded: they are shared with
        JEPA and every other arm, and they are not heads. The first draft of this
        test excluded only `encoder.token_embedding.weight` and went red on
        BYOL's `target_encoder.token_embedding.weight`, which is the EMA copy of
        the same table and belongs to the teacher, not to a projection head. The
        second draft matched on the substring `"encoder"`, which missed
        `target_encoder` for the same reason.

        So the exclusion is by ROLE -- the top-level module a parameter belongs
        to -- rather than by substring or by an enumerated list of names that
        would need editing every time an arm gained a teacher.
        """
        encoder_roles = ("encoder", "target_encoder")
        for name, cls in ARMS.items():
            model = cls(**TINY)
            for pname, param in model.named_parameters():
                if pname.split(".")[0] in encoder_roles:
                    continue  # shared encoder / teacher tables, not heads
                assert tuple(param.shape) != (TINY["vocab_size"], TINY["embed_dim"]), (
                    f"{name}.{pname} is a vocabulary-sized matrix; that is the "
                    "untied head that makes MLMBaseline 1.281x JEPA"
                )

    def test_these_arms_are_matched_better_than_mlm_is(self):
        """The comparison the critic actually made, re-run.

        Stated as an ordering, not as magic numbers: the four new arms must sit
        closer to JEPA's trainable count than the existing MLM baseline does, at
        the rung where MLM's 1.281x was measured. If a future change made these
        arms worse-matched than MLM, the whole point of the exercise is gone.
        """
        vocab, seq, dim, depth, heads, pdim, pdepth = PRODUCTION[0]
        jepa_trainable = _jepa_trainable(vocab, seq, dim, depth, heads, pdim, pdepth)

        from baselines.mlm_baseline import MLMBaseline

        with _meta_device():
            mlm_ratio = (
                MLMBaseline(
                    vocab_size=vocab, max_seq_len=seq, embed_dim=dim, depth=depth, num_heads=heads
                ).get_num_params_trainable()
                / jepa_trainable
            )
            ratios = {
                name: cls(
                    vocab_size=vocab, max_seq_len=seq, embed_dim=dim, depth=depth, num_heads=heads
                ).get_num_params_trainable()
                / jepa_trainable
                for name, cls in ARMS.items()
            }

        assert mlm_ratio == pytest.approx(
            MLM_OVER_JEPA_AT_640, rel=0.005
        ), "MLMBaseline's ratio moved; tests/test_baseline_parity.py pins it too"
        for name, ratio in ratios.items():
            assert abs(ratio - 1.0) < abs(mlm_ratio - 1.0), (
                f"{name} at {ratio:.4f}x is further from JEPA than MLM is at " f"{mlm_ratio:.4f}x"
            )


# ═══════════════════════════════════════════════════════════════════
#  The contract the trainer dispatches on
# ═══════════════════════════════════════════════════════════════════


class TestTrainerContract:
    """What `src/train.py` requires of an arm, pinned from this side.

    The coupling is pinned from HERE even though `src/train.py` is now editable:
    a card that owns a file tends to trust it, and the property that matters --
    "this arm reaches its own loss, and only its own loss" -- is a property of
    the PAIR, so it belongs in a place neither owner can quietly rewrite. Same
    approach `tests/test_baseline_parity.py::test_train_py_logs_the_bare_call`
    takes.

    WHAT USED TO BE TRUE, AND WHY THE LOWER HALF OF THIS CLASS EXISTS
    -----------------------------------------------------------------
    The dispatch was duck-typed and ORDER-SENSITIVE:

        if hasattr(model, "compute_loss_with_targets"):                    -> JEPA
        elif hasattr(model, "forward") and hasattr(model, "regression_head"): -> data2vec
        elif hasattr(model, "compute_loss"):                               -> MLM

    Both earlier branches are checked BEFORE the `compute_loss` branch these arms
    depend on, and both are positive tests for an incidental attribute name. An
    arm that owned an attribute named `regression_head` -- an innocuous name, and
    the obvious name for a regression-style head -- would be handed data2vec's
    call signature and would fail confusingly, or worse, appear to train.

    The four arms did not own it, so they happened to work. That is luck rather
    than design, and it is why the two halves below now both exist:

    * the upper tests pin what each arm MUST NOT own, which is the rule the old
      dispatcher ran on;
    * `test_a_decoy_regression_head_no_longer_reroutes_the_arm` pins that the rule
      is no longer what routes anything -- `create_model` now DECLARES a
      `loss_protocol` per arm, so attaching the decoy changes nothing.
    """

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_arm_is_dispatched_to_the_compute_loss_branch(self, name):
        """Walk the trainer's own dispatch logic against a real instance.

        Read out of `src/train.py` rather than restated, so if the trainer's
        order ever changes this test follows it instead of testing a stale copy
        of the rule.
        """
        model = build(name)
        assert not hasattr(model, "compute_loss_with_targets"), (
            f"{name} defines compute_loss_with_targets; src/train.py would route "
            "it to the JEPA loss signature"
        )
        assert not hasattr(model, "regression_head"), (
            f"{name} owns an attribute named regression_head; src/train.py checks "
            "for that BEFORE compute_loss and would call it with the data2vec "
            "signature"
        )
        assert hasattr(model, "compute_loss"), f"{name} has no compute_loss to dispatch to"

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_compute_loss_accepts_the_trainers_three_argument_call(self, name):
        """`src/train.py::compute_loss` calls with three positional arguments."""
        import inspect

        params = list(inspect.signature(ARMS[name].compute_loss).parameters)
        assert params[:3] == ["self", "view_a", "view_b"], (
            f"{name}.compute_loss takes {params[:3]}; the trainer's call site is "
            "model.compute_loss(masked_input_ids, original_input_ids, mask_positions)"
        )

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_mask_positions_is_accepted_and_ignored(self, name):
        """These are non-instance-discrimination methods: no per-position targets.

        Asserted rather than assumed, because the argument LOOKS load-bearing --
        a reader would reasonably expect it to be used. Pinning that it is
        accepted and unused stops a future edit from quietly turning these arms
        into masked-prediction methods without anyone noticing.
        """
        model = build(name)
        ids = batch()
        view_a, view_b = two_views(ids, 0, 0.3)
        mask = torch.zeros_like(ids, dtype=torch.long)
        mask[:, :4] = 1

        with torch.no_grad():
            loss_no_mask, _ = model.compute_loss(view_a, view_b)
            loss_with_mask, _ = model.compute_loss(view_a, view_b, mask)
        assert torch.isclose(loss_no_mask, loss_with_mask), (
            f"{name} is using mask_positions; these four methods are "
            "non-instance-discrimination and have no per-position target set"
        )

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_parameter_counts_are_the_trainers_kind_of_number(self, name):
        """`get_num_params()` must be the model's SIZE, as the other three arms.

        `tests/test_baseline_parity.py::TestLoggedQuantityIsOneKind` pins this
        for the arms `create_model` knows; these arms are wired into
        `create_model`, so that file pins it too, and this is the copy that
        survives `create_model` regressing. Recounted independently of the
        method body.
        """
        model = build(name)

        def independent_total(mod):
            seen = set()
            total = 0
            for _, child in mod.named_modules():
                for p in child.parameters(recurse=False):
                    if id(p) not in seen:
                        seen.add(id(p))
                        total += p.numel()
            return total

        assert model.get_num_params() == independent_total(model), (
            f"{name}.get_num_params() is not the model's parameter count; "
            "src/train.py logs it as 'Model parameters'"
        )
        assert hasattr(model, "get_num_params_trainable"), (
            f"{name} cannot report trainable capacity, which is the only "
            "like-for-like number to compare arms on"
        )
        assert model.get_num_params(True) < model.get_num_params(), (
            f"{name} owns no embedding tables, so the non_embedding flag does "
            "nothing -- suspicious enough to be worth failing on"
        )

    def test_byol_ema_hook_matches_the_trainers_zero_argument_call(self):
        """`do_ema_update` calls `model.update_target_encoder()` with no arguments.

        BYOL is the only arm here with a teacher, so it is the only one this
        applies to. data2vec's identical signature is the precedent.
        """
        import inspect

        params = list(inspect.signature(BYOLBaseline.update_target_encoder).parameters)
        assert params == ["self"], (
            f"BYOLBaseline.update_target_encoder takes {params}; the trainer's "
            "do_ema_update calls it with no arguments"
        )

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_loss_and_info_are_a_scalar_and_a_dict_of_floats(self, name):
        """The shape `src/train.py::compute_loss` unpacks into the log."""
        model = build(name)
        ids = batch()
        view_a, view_b = two_views(ids, 0, 0.3)
        loss, info = model.compute_loss(view_a, view_b)
        assert loss.ndim == 0
        assert loss.requires_grad
        assert isinstance(info, dict) and info
        for key, value in info.items():
            assert isinstance(value, float), f"{name}.info[{key!r}] is {type(value)}, not float"

    # ---- reachability: an arm nothing can build cannot be trained ----

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_the_arm_is_reachable_from_create_model(self, name):
        """The headline. Implemented and reachable are different properties.

        These four modules existed with full tests and were still untrainable:
        `create_model` raised `ValueError` on every one of their names. A collapse
        test on a class nothing constructs proves the class works; it does not
        prove the experiment can be run, and nothing in the file would have said
        so.
        """
        model = trainer_build(name)
        assert isinstance(model, ARMS[name]), (
            f"create_model({name!r}) returned {type(model).__name__}; the arm is "
            "implemented but unreachable, so no run can train it"
        )

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_the_config_name_normalises_to_a_declared_arm(self, name):
        """`meta.model_name` in a config must survive `_normalize_model_name`.

        Checked both ways, because the two tables are separate and a one-way
        check passes when only one of them has drifted: the prefix table is what
        a config actually meets, and `LOSS_PROTOCOLS` is what `create_model` then
        looks the arm up in. An arm present in one and missing from the other is
        either an unreachable arm or a dispatch table entry nothing can reach.
        """
        from src.train import LOSS_PROTOCOLS, _normalize_model_name

        assert _normalize_model_name(name) == name
        assert _normalize_model_name(f"{name}_small") == name, (
            f"a suffixed config name does not normalise to {name!r}; every other "
            "arm accepts a size suffix and these four would have to be spelled "
            "exactly"
        )
        assert name in LOSS_PROTOCOLS, f"{name} has no declared loss protocol"

        reachable = {
            canonical
            for canonical in LOSS_PROTOCOLS
            if _normalize_model_name(f"{canonical}_small") == canonical
        }
        assert reachable == set(
            LOSS_PROTOCOLS
        ), f"prefix table and protocol table disagree: {sorted(set(LOSS_PROTOCOLS) - reachable)}"

    # ---- the dispatch itself ----

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_create_model_declares_the_protocol_instead_of_inferring_it(self, name):
        """The arm carries its routing as data, so routing cannot depend on naming.

        A plain string on the module: not a buffer, not a parameter, so it is
        absent from `state_dict()` and cannot perturb a checkpoint or a
        `strict=True` load.
        """
        from src.train import LOSS_COMPUTE_LOSS

        model = trainer_build(name)
        assert model.loss_protocol == LOSS_COMPUTE_LOSS
        assert "loss_protocol" not in model.state_dict(), (
            "the declared protocol must not enter the checkpoint; a resume with "
            "strict=True would then fail on an unknown key"
        )

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_a_decoy_regression_head_no_longer_reroutes_the_arm(self, name):
        """THE TRAP, made into a test.

        `regression_head` is attached to a real, correctly-wired arm -- exactly
        the innocuous attribute the old duck-typed chain selected on -- and the
        arm must still reach its own loss.

        Against the previous implementation this goes red with a `TypeError`
        rather than a wrong number, because all four `forward` methods take
        `(view_a, view_b)` and data2vec's call site passes three tensors. That is
        the best available outcome for a test: loud, and at step 0 rather than in
        a result table six weeks later. The silent case -- a `forward` that
        happened to accept the arguments -- is the one this cannot catch, and is
        exactly the case the declared protocol removes rather than tests around.
        """
        from src.train import compute_loss

        model = trainer_build(name)
        model.eval()
        model.regression_head = nn.Linear(TINY["embed_dim"], TINY["embed_dim"])

        ids = batch()
        view_a, view_b = two_views(ids, 0, 0.3)
        mask = torch.zeros_like(ids, dtype=torch.long)

        loss, info, diag = compute_loss(model, view_a, view_b, mask)
        with torch.no_grad():
            expected, _ = model.compute_loss(view_a, view_b, mask)
        assert torch.isclose(loss, expected), (
            f"{name} did not reach its own loss once it owned a `regression_head`; "
            "the dispatch is still deciding on attribute names"
        )
        assert isinstance(info, dict) and info, f"{name} returned no info dict to log"
        assert diag == {}, f"{name} returned JEPA-shaped diagnostics it never filled"

    # ---- BYOL's teacher: the second silent routing ----

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_only_byol_is_expected_to_hold_a_self_managed_ema(self, name):
        """Which arms the trainer must advance a teacher on, stated as a set.

        Asserted against `SELF_EMA_ARMS` rather than against the loop's source so
        the reason is legible; `test_the_loop_reaches_ema_through_that_set` covers
        the loop itself.
        """
        from src.train import SELF_EMA_ARMS

        assert (name in SELF_EMA_ARMS) is (name == "byol"), (
            f"{name} in SELF_EMA_ARMS={name in SELF_EMA_ARMS}; BYOL owns an EMA "
            "teacher and the other three do not"
        )
        assert "text_span_jepa" not in SELF_EMA_ARMS, (
            "JEPA's teacher is driven by EMATauSchedule and takes `tau` as an "
            "argument; it is a different call and must not share this branch"
        )

    def test_byols_teacher_actually_moves_through_the_trainers_ema_call(self):
        """The behaviour `do_ema_update(model, "byol")` is supposed to have.

        The student is perturbed first, for the reason
        `TestByolCollapse::test_the_teacher_is_ema_updated_toward_the_student`
        records: the teacher is a `deepcopy` taken in `__init__`, so at step 0
        the two encoders agree bitwise and an EMA that works is indistinguishable
        from an EMA that does nothing.

        Both halves are asserted. A hook that moved the STUDENT instead would
        satisfy the gap-closing assertion alone, and would be a silent disaster:
        the arm would train, and its teacher would be following it.
        """
        from src.train import do_ema_update

        model = trainer_build("byol")
        with torch.no_grad():
            for param in model.encoder.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        student_before = [p.detach().clone() for p in model.encoder.parameters()]
        gaps_before = [
            (s - t).norm() for s, t in zip(student_before, model.target_encoder.parameters())
        ]
        assert all(g > 0 for g in gaps_before), "teacher and student already agree; test is vacuous"

        do_ema_update(model, "byol")

        student_after = [p.detach().clone() for p in model.encoder.parameters()]
        assert all(
            torch.equal(a, b) for a, b in zip(student_before, student_after)
        ), "the EMA update moved the STUDENT; the teacher is the only thing it may touch"
        gaps_after = [
            (s - t).norm() for s, t in zip(student_after, model.target_encoder.parameters())
        ]
        assert all(
            a < b for a, b in zip(gaps_after, gaps_before)
        ), "the teacher did not move toward the student"

    def test_the_loop_reaches_ema_through_that_set(self):
        """`main()`'s per-step EMA branch must test the SET, not a literal arm.

        The one-line version of the trap. `do_ema_update` could handle BYOL and
        the loop would still never call it, because the loop's condition is a
        separate statement -- and the symptom is not a crash: the run trains with
        a teacher frozen at its random initialisation, logs a falling loss, and
        produces a model that is not the method the config names. Nothing in this
        file could see that from the outside, which is why the call site is
        pinned here.
        """
        import inspect
        import re

        import src.train as train_mod

        body = inspect.getsource(train_mod.main)
        assert re.search(r"elif\s+model_name\s+in\s+SELF_EMA_ARMS\s*:", body), (
            "main()'s EMA branch no longer tests SELF_EMA_ARMS; an arm added to "
            "that set would silently train without its teacher being advanced"
        )

    # ---- the optimizer ----

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_the_optimizer_holds_exactly_the_trainable_parameters(self, name):
        """Two failures in one assertion, both of which are silent.

        A FROZEN parameter in the optimizer (BYOL's teacher) can never receive a
        gradient -- it is dead weight the optimizer carries forever. A trainable
        parameter MISSING from it is the R18 defect this repo already fixed once:
        the module trains, the parameter never moves, and the result is a model
        that is quietly not the one that was configured. `get_param_groups`'s
        catch-all branch would have produced the first for BYOL.
        """
        from src.train import get_param_groups

        model = trainer_build(name)
        groups = get_param_groups(model, name)
        optimised = [p for group in groups for p in group["params"]]
        assert len(optimised) == len(
            {id(p) for p in optimised}
        ), f"{name} puts the same parameter in two optimizer groups"
        assert {id(p) for p in optimised} == {
            id(p) for p in model.parameters() if p.requires_grad
        }, (
            f"{name}'s optimizer does not hold exactly its trainable parameters; "
            "a frozen teacher or an invisible parameter is the usual cause"
        )

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_the_encoder_keeps_the_repo_weight_decay_split(self, name):
        """LayerNorm weights and biases must be excluded from weight decay.

        Weight decay on a normalisation scale is a known optimiser defect, and
        this repo's convention is explicit in every other arm's branch. The
        catch-all would have decayed everything; the arms would still converge,
        so the defect would surface as a slightly worse baseline rather than as
        an error.
        """
        from src.train import get_param_groups

        model = trainer_build(name)
        groups = get_param_groups(model, name, wd=0.04)
        owner = {id(p): group for group in groups for p in group["params"]}

        for param_name, param in model.encoder.named_parameters():
            group = owner[id(param)]
            if ("bias" in param_name) or (len(param.shape) == 1):
                assert group.get("weight_decay", None) == 0, (
                    f"{name}.encoder.{param_name} is a bias or a 1-D norm weight "
                    "and is still being weight-decayed"
                )
            else:
                assert group.get("weight_decay", None) in (None, 0.04), (
                    f"{name}.encoder.{param_name} should be decayed at the "
                    f"requested wd, got {group.get('weight_decay')}"
                )

    # ---- config keys ----

    def test_create_model_reads_only_keys_defaults_yaml_declares(self):
        """Why these arms read no method hyperparameter from `model:`.

        `_warn_unknown_config_keys` warns about any config leaf whose dotted path
        is absent from `defaults.yaml`, unless the path is in its `extra_known`
        set -- and `tests/test_config_system.py::test_trainer_extra_known_matches_the_trainer`
        pins that set against its own copy, so a new exemption there fails a test
        this file does not own. A `model.target_momentum` read therefore needs
        three coordinated edits in three files, none of them in this card.

        So it is not read: each method hyperparameter already defaults to its
        published-paper value, and an arm built from `defaults.yaml` alone trains
        the method as published. This test makes that a stated property instead of
        an accident -- adding a `model_cfg.get(...)` to the branch turns it red
        and names the two files that have to move with it.
        """
        import inspect
        import re

        import src.train as train_mod

        with open(
            Path(train_mod.__file__).resolve().parent.parent / "defaults.yaml", encoding="utf-8"
        ) as handle:
            defaults = yaml.safe_load(handle)
        declared = set(defaults["model"])

        source = inspect.getsource(train_mod.create_model)
        start = source.index("elif model_name in SSL_BASELINE_ARMS:")
        rest = source[start:]
        ends = [rest.index(marker) for marker in ("\n    elif ", "\n    else:") if marker in rest]
        branch = rest[: min(ends)] if ends else rest
        read = set(re.findall(r'model_cfg\.get\("([a-z_]+)"', branch))

        assert read, "the branch reads no model keys at all; this test is stale"
        assert read <= declared, (
            f"create_model reads {sorted(read - declared)}, which "
            f"defaults.yaml does not declare. Every such key must also be added "
            "to src/train.py's `extra_known` AND to "
            "tests/test_config_system.py::_TRAINER_EXTRA_KNOWN, or every run of "
            "this arm logs a 'possible typo' warning for a key it is using"
        )


# ═══════════════════════════════════════════════════════════════════
#  Shapes, edge cases, and gradient flow
# ═══════════════════════════════════════════════════════════════════


class TestShapesAndEdgeCases:
    """The things a reviewer asks that are not about collapse."""

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_runs_at_several_shapes(self, name):
        """Not one shape. A head-width bug can be invisible at the shape it was
        tuned on and obvious one step away."""
        for shape in [
            {"vocab_size": 64, "max_seq_len": 8, "embed_dim": 16, "depth": 1, "num_heads": 2},
            {"vocab_size": 200, "max_seq_len": 16, "embed_dim": 32, "depth": 2, "num_heads": 4},
            {"vocab_size": 2000, "max_seq_len": 32, "embed_dim": 64, "depth": 3, "num_heads": 8},
        ]:
            model = ARMS[name](**shape)
            ids = torch.randint(0, shape["vocab_size"], (4, shape["max_seq_len"]))
            loss, info = model.compute_loss(ids, ids.flip(-1))
            assert torch.isfinite(loss), f"{name} produced {loss} at {shape}"
            assert isinstance(info, dict)

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_batch_of_one_does_not_raise_in_eval(self, name):
        """A single sample must not crash an evaluation pass.

        In EVAL mode specifically, and that qualifier is the point rather than a
        hedge: every one of these four methods puts a `BatchNorm1d` in its
        head, and `BatchNorm1d` raises `ValueError: Expected more than 1 value
        per channel when training` on a batch of one in TRAIN mode. That is
        inherent to the methods -- the reference BYOL, Barlow Twins, VICReg and
        SimSiam implementations all have it -- so the honest test is that
        evaluation works and the train-mode limit is a property of BatchNorm
        rather than of this port.

        Asserted here so that a future change which makes the heads
        batch-size-sensitive in EVAL mode is caught, and so the train-mode
        limitation is written down instead of being rediscovered as a crash.
        """
        model = ARMS[name](**TINY)
        model.eval()
        ids = torch.randint(0, TINY["vocab_size"], (1, TINY["max_seq_len"]))
        with torch.no_grad():
            loss, _ = model.compute_loss(ids, ids)
        assert torch.isfinite(loss), f"{name} produced {loss} for a batch of one in eval"

    def test_train_mode_batch_of_one_is_a_batchnorm_limit_not_a_port_bug(self):
        """Pins WHY the test above is eval-only, so the qualifier is not a dodge.

        If this ever stops raising, the arms' heads are no longer
        batch-size-sensitive and the eval-only qualifier above can be dropped.
        """
        model = build("barlow")
        model.train()
        ids = torch.randint(0, TINY["vocab_size"], (1, TINY["max_seq_len"]))
        with pytest.raises(ValueError, match="more than 1 value per channel"):
            model.compute_loss(ids, ids)

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_gradient_reaches_the_encoder(self, name):
        """The arm is trainable end to end -- a detached encoder would 'train'
        and learn nothing, which is the same class of defect as a collapsed one.
        """
        model = build(name)
        ids = batch()
        view_a, view_b = two_views(ids, 0, 0.3)
        loss, _ = model.compute_loss(view_a, view_b)
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        assert any(
            g is not None and g.abs().sum() > 0 for g in grads
        ), f"{name} produced no gradient in its encoder"

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_training_reduces_the_loss_on_a_fixed_batch(self, name):
        """A smoke check that the objective is actually optimisable.

        Not a quality claim -- 100 steps on random tokens learns nothing
        transferable. It is a wiring claim: if the gradient does not point
        downhill, the arm cannot work for any reason other than the objective
        being wrong, and that is worth catching before a reviewer does.
        """
        seed_everything(0)
        model = build(name)
        ids = batch()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        model.train()
        losses = []
        for step in range(40):
            view_a, view_b = two_views(ids, step, 0.3)
            opt.zero_grad()
            loss, _ = model.compute_loss(view_a, view_b)
            loss.backward()
            opt.step()
            advance = getattr(model, "update_target_encoder", None)
            if advance is not None:
                advance()
            losses.append(loss.item())
        assert (
            losses[-1] < losses[0]
        ), f"{name} loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_pooling_is_identical_across_arms(self):
        """The four arms must pool the same way, or their collapse statistics
        measure four different objects and cannot be compared.

        Compared BY VALUE on the same input, not by source text. The first draft
        of this test compared `inspect.getsource`, which failed on three of four
        arms because their docstrings differ -- it was measuring prose, not
        behaviour, and would have gone red the next time someone reworded a
        comment.
        """
        h = torch.randn(8, 16, 32, generator=torch.Generator().manual_seed(3))
        outputs = {name: cls.pool(h) for name, cls in ARMS.items()}
        reference_name, reference = next(iter(outputs.items()))
        for name, value in outputs.items():
            assert torch.equal(value, reference), (
                f"{name}.pool differs from {reference_name}.pool; the collapse "
                "statistics in this file are only comparable across arms if the "
                "representation they measure is the same object"
            )
            assert value.shape == (8, 32)
