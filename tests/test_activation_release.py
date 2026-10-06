# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Cross-micro-batch retention of `jepa._gac_z` and `jepa._cmc_pass["slots"]`.

TASK-21. The card that motivated these tests claimed both attributes pin a
previous micro-batch's autograd graphs and asked for the graphs to be dropped
so peak activation memory halves. Two things were established by measurement
and are pinned here:

1. Neither attribute may be detached. Both are consumed by a real backward --
   `_gac_z` by GAC's exploration bonus, `_cmc_pass["slots"]` by the CMC
   secondary path -- so `.detach()` would silently delete a gradient rather
   than save memory. `TestGacExplorationGradient` and `TestCmcSecondaryGradient`
   assert the gradient itself (that encoder gradients MOVE), not a shape, and
   each carries the detaching counterfactual that shows what would be lost.

2. Dropping the attributes early does NOT halve peak memory on the main
   accumulation path, because `src/train.py` snapshots them into locals that
   outlive the pass (`_gac_primary`/`_cmc_primary`/`_cmc_secondary`,
   rebinding only after the next forward). `TestEarlyRelease` pins the release
   that *is* in `jepa.py`'s control: the empty-batch early return, which used
   to pin the whole previous pass.

Measured on this box (B=4, T=64, D=128, 4-micro-batch accumulation, census of
live graph tensors):
    train.py shape, attributes kept .................. 110.25MB peak
    train.py shape, attributes also dropped ......... 110.25MB peak  (+0.00)
    no cross-step references .........................  74.62MB peak
    empty batch with attributes kept ................  19.31MB pinned
    empty batch with early release ..................   0.00MB pinned
"""

import gc
import weakref

import torch

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig


def _tiny_config() -> TextSpanJEPAConfig:
    return TextSpanJEPAConfig(
        vocab_size=64,
        max_seq_len=16,
        embed_dim=32,
        encoder_depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        predictor_embed_dim=16,
        predictor_depth=1,
        future_offsets=[1],
        num_refine_steps=1,
        use_jawp=True,
        jawk_k_start=2,
        jawk_k_end=4,
        jawk_curriculum_steps=0,
        use_cmc=True,
        cmc_interval=1,
        lambda_cmc=0.05,
        use_gac=True,
        gac_gamma=0.01,
        # A threshold larger than any attainable gradient marks every dimension
        # starved, so the exploration bonus cannot be accidentally zero and the
        # test fails loudly rather than passing on an all-zero gradient.
        gac_tau_grad=1e6,
        gac_warmup_steps=0,
        lambda_gac=0.02,
        future_warmup_steps=0,
    )


def _masks(T: int = 16):
    m1 = torch.zeros(2, T, dtype=torch.long)
    m2 = torch.zeros(2, T, dtype=torch.long)
    m1[:, 2:8] = 1
    m2[:, 5:11] = 1
    return m1, m2


def _grad_snapshot(model):
    return {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in model.named_parameters()
    }


def _moved_since(before, model):
    """Names of parameters whose gradient CHANGED (not merely non-zero)."""
    moved = []
    for n, p in model.named_parameters():
        old = before.get(n)
        if p.grad is None:
            continue
        if old is None or not torch.equal(old, p.grad):
            moved.append(n)
    return moved


def _live_graph_tensors():
    gc.collect()
    out = []
    for o in gc.get_objects():
        try:
            if type(o) is torch.Tensor and o.grad_fn is not None:
                out.append(o)
        except Exception:
            pass
    return out


class TestStashesStayLive:
    """The values must remain attached to the graph inside a pass."""

    def test_gac_z_is_graph_attached_after_the_forward(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        assert model._gac_z is not None
        assert model._gac_z.grad_fn is not None, (
            "_gac_z must stay attached: GAC's exploration backward re-enters the "
            "predictor and encoder through it"
        )
        assert model._gac_z.requires_grad

    def test_cmc_slots_are_graph_attached_after_the_forward(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        assert model._cmc_pass is not None
        slots = model._cmc_pass["slots"]
        assert slots.grad_fn is not None, (
            "the secondary pass's slots must stay attached: compute_cmc_between_"
            "passes scatters them live so the gradient reaches the encoder"
        )
        # The detached twin is a value-only mirror and must stay value-equal.
        assert torch.equal(model._cmc_pass["slots_det"], slots.detach())


class TestGacExplorationGradient:
    """Proves `_gac_z` is used in a backward -- by asserting gradient MOVES."""

    def test_exploration_backward_reaches_the_encoder(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        total, _, _ = model.compute_loss_with_targets(ids, ids, m1, 10, 100)

        z_ref = model._gac_z
        assert z_ref is not None
        total.backward(retain_graph=True)
        assert z_ref.grad is not None, "retain_grad() must populate the stash"

        before = _grad_snapshot(model)
        D = z_ref.size(-1)
        g_norms = z_ref.grad.detach().reshape(-1, D).norm(dim=0)
        loss_gac, _ = model.gac(z_ref, g_norms, step=10)
        assert loss_gac.requires_grad, (
            "GAC's bonus is built from z_pred**2 without detaching, so the "
            "exploration term must be differentiable"
        )
        loss_gac.backward()
        moved = _moved_since(before, model)
        assert moved, (
            "loss_gac.backward() must move encoder/predictor gradients; it can "
            "only do so by traversing the graph held by _gac_z"
        )

    def test_detaching_the_stash_silently_deletes_that_gradient(self):
        """The counterfactual. A `.detach()` 'memory fix' on `_gac_z` turns the
        exploration bonus into a constant: no graph, no gradient, no error."""
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        total, _, _ = model.compute_loss_with_targets(ids, ids, m1, 10, 100)

        z_ref = model._gac_z
        total.backward(retain_graph=True)
        D = z_ref.size(-1)
        g_norms = z_ref.grad.detach().reshape(-1, D).norm(dim=0)

        detached_loss, _ = model.gac(z_ref.detach(), g_norms, step=10)
        assert not detached_loss.requires_grad, (
            "expected a detached z_pred to yield a non-differentiable bonus -- "
            "that IS the silent gradient break this test exists to prevent"
        )

        before = _grad_snapshot(model)
        live_loss, _ = model.gac(z_ref, g_norms, step=10)
        live_loss.backward()
        assert _moved_since(before, model), "sanity: the live path does move gradients"


class TestCmcSecondaryGradient:
    """Proves the secondary pass's `slots` are used in a backward."""

    def _run_bridge(self, detach_secondary):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, m2 = _masks()

        total1, _, _ = model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        primary = model._cmc_pass
        model.compute_loss_with_targets(ids, ids, m2, 10, 100)
        secondary = model._cmc_pass
        if detach_secondary:
            secondary = dict(secondary)
            secondary["slots"] = secondary["slots"].detach()

        loss_cmc, info = model.compute_cmc_between_passes(primary, secondary)
        return model, total1, loss_cmc, info

    def test_secondary_path_moves_encoder_gradients(self):
        model, _, loss_cmc, info = self._run_bridge(detach_secondary=False)
        assert torch.isfinite(loss_cmc).all()
        assert not info.get("cmc_skipped"), "masks must overlap for this test to mean anything"
        assert loss_cmc.requires_grad, "the bridge must stay attached to the secondary pass's slots"
        # Backprop the CMC term ALONE. Backpropagating the primary pass's loss as
        # well would keep encoder gradients non-zero even with the secondary path
        # detached, which is exactly the hole this test has to close.
        before = _grad_snapshot(model)
        loss_cmc.backward()
        moved = _moved_since(before, model)
        assert any(
            n.startswith("encoder.") for n in moved
        ), "the CMC secondary pass alone must backprop into the online encoder"

    def test_detaching_secondary_slots_silently_deletes_that_gradient(self):
        """The counterfactual, again: detaching costs a gradient and raises nothing."""
        model, _, loss_cmc_det, info = self._run_bridge(detach_secondary=True)
        assert torch.isfinite(loss_cmc_det).all()
        assert not loss_cmc_det.requires_grad, (
            "expected the detached secondary slots to drop out of the graph -- "
            "that IS the silent gradient break this test exists to prevent"
        )

        model2, _, loss_cmc_live, _ = self._run_bridge(detach_secondary=False)
        assert loss_cmc_live.requires_grad, "sanity: the live bridge is differentiable"


class TestEarlyRelease:
    """The release `jepa.py` actually controls: the empty-batch early return."""

    def test_empty_batch_does_not_pin_the_previous_pass(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()

        total, _, _ = model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        stashed = model._gac_z
        total.backward(retain_graph=True)
        assert stashed.grad_fn is not None
        del total

        # The stash is the model's only reference at this point.
        ref = weakref.ref(stashed)
        del stashed

        empty = ids[:0]
        model.compute_loss_with_targets(empty, empty, m1[:0], 11, 100)

        assert model._gac_z is None, "the empty-batch early return must drop _gac_z"
        assert model._cmc_pass is None, "the empty-batch early return must drop _cmc_pass"
        gc.collect()
        assert ref() is None, (
            "the previous pass's slot tensor must be collectable after the "
            "empty-batch forward; if it is still alive the graph is pinned"
        )

    def test_release_does_not_cost_the_current_pass_its_stash(self):
        """Releasing at the top must not take the CURRENT pass's stash with it."""
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()

        empty = ids[:0]
        model.compute_loss_with_targets(empty, empty, m1[:0], 11, 100)
        model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        assert model._gac_z is not None and model._gac_z.grad_fn is not None
        assert model._cmc_pass is not None
        assert model._cmc_pass["slots"].grad_fn is not None

    def test_release_happens_before_the_new_graph_is_built(self):
        """Ordering: the old stash must be gone by the time `encoder` runs.

        Sampled with a forward pre-hook, so this fails if the reset is moved back
        behind the first allocation of the new graph.
        """
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, m2 = _masks()

        model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        assert model._gac_z is not None

        seen = {}
        handle = model.encoder.register_forward_pre_hook(
            lambda mod, args: seen.update(gac=model._gac_z, cmc=model._cmc_pass)
        )
        try:
            model.compute_loss_with_targets(ids, ids, m2, 11, 100)
        finally:
            handle.remove()
        assert (
            seen["gac"] is None
        ), "the previous pass's _gac_z must be released before self.encoder(...)"
        assert (
            seen["cmc"] is None
        ), "the previous pass's _cmc_pass must be released before self.encoder(...)"
        assert model._gac_z is not None, "and the new pass must still stash its own"


class TestAttribution:
    """Pins the measured attribution so the card is not re-run blind.

    These assert the *ownership* of the retention, which is the thing the card
    got wrong. They are cheap and deterministic.
    """

    def test_dropping_the_stashes_is_not_what_halves_peak_memory(self):
        """With no other reference, dropping the stashes frees the graph.

        This is the property the early release relies on. It is also why the
        card's remedy could not work in `train.py`: there, the locals pin the
        graph first, so the attributes are never the last reference.
        """
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()

        total, _, _ = model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        total.backward()
        del total
        baseline = len(_live_graph_tensors())
        assert baseline > 0, "the stashes should still be holding the pass's graph"

        model._gac_z = None
        model._cmc_pass = None
        assert len(_live_graph_tensors()) == 0, (
            "once the model drops its own references, nothing may keep the graph "
            "alive -- if something else does, the fix has to live there instead"
        )
