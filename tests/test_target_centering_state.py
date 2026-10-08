# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""`target_centering.center` is training state, not a loss input.

`src/train.py::_validate` runs the whole loss under ``model.eval()`` +
``torch.no_grad()``.  `no_grad` blocks *gradient* writes and nothing else, and
`TargetCentering.forward` used to fold the validation split's mean into the
EMA statistic that the next *training* step subtracts.  So the trained weights
depended on whether a validation split was loaded: same seed, different model.
The trainer-side snapshot-restore masks this; the fix is at source.

`TargetCentering` lives in `src/models/collapse.py` and is a plain
`nn.Module`, so it cannot guard its own write.  The model reaches it in
exactly one place (`TextSpanJEPA.compute_loss_with_targets`), so the guard is
routed there through `TrainingStateGuard._mutate_state`.

The contract pinned here:

  * eval() + no_grad() leaves the center bit-identical (and, on a
    mechanism-free model, moves no buffer at all)
  * train() still moves it, with the data2vec EMA arithmetic intact and
    layer_norm still applied *after* the subtraction
  * eval() still centers, with the frozen center -- the guard is on the
    write, not on the use
  * a validation pass does not perturb a training trajectory
  * the write is reachable by grepping the single name `_mutate_state`
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 20260824
SEQ = 16
VOCAB = 100


def _model(*, use_jawp: bool = True, seed: int = SEED):
    """A tiny TextSpanJEPA with every stochastic path switched off.

    `drop_rate` / `attn_drop_rate` / `drop_path_rate` are zero so that a
    forward is a pure function of its weights and inputs -- required by the
    bit-exactness assertions below.

    Args:
        use_jawp: keep the default mechanism bundle, or strip it down to a
            model whose only mutable buffer is the target center.
        seed: torch seed applied before construction.

    Returns:
        The constructed model.

    """
    from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

    torch.manual_seed(seed)
    cfg = TextSpanJEPAConfig(
        vocab_size=VOCAB,
        max_seq_len=SEQ,
        embed_dim=32,
        encoder_depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        predictor_embed_dim=16,
        predictor_depth=2,
        future_offsets=(1,),
        num_refine_steps=1,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_jawp=use_jawp,
    )
    return TextSpanJEPA(cfg)


def _batch():
    """A (ids, ids, mask_positions) triple, fixed for every call."""
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, VOCAB, (2, SEQ), generator=g)
    mask = torch.zeros(2, SEQ, dtype=torch.long)
    mask[:, 3:6] = 1
    return ids, ids.clone(), mask


def _step(model):
    """Run one training-style loss call, returning the scalar loss."""
    ids, orig, mask = _batch()
    loss, _loss_dict, _diag = model.compute_loss_with_targets(ids, orig, mask, current_step=1)
    return loss


def _validate_call(model):
    """Run one `src/train.py::_validate`-shaped call: eval() + no_grad()."""
    ids, orig, mask = _batch()
    model.eval()
    with torch.no_grad():
        loss, _ld, _d = model.compute_loss_with_targets(ids, orig, mask, current_step=1)
    return loss


def _buffers(model):
    return {n: b.detach().clone() for n, b in model.named_buffers()}


def _moved(before, after):
    return sorted(n for n, b in before.items() if not torch.equal(b, after[n]))


def _spy(model):
    """Capture the raw target embedding and the tensor the predictor saw.

    Returns:
        (spy_install, recorded) -- call ``spy_install()`` to remove the spies.

    """
    recorded = {}
    orig_encoder = model.target_encoder.forward
    orig_predictor = model.predictor.forward

    def enc_spy(*args, **kwargs):
        out = orig_encoder(*args, **kwargs)
        recorded["raw_target"] = out[0].detach().clone()
        return out

    def pred_spy(h_online, mask_positions, token_embeds, target_h):
        recorded["predictor_target"] = target_h.detach().clone()
        return orig_predictor(h_online, mask_positions, token_embeds, target_h)

    model.target_encoder.forward = enc_spy
    model.predictor.forward = pred_spy

    def restore():
        model.target_encoder.forward = orig_encoder
        model.predictor.forward = orig_predictor

    return restore, recorded


class TestEvalDoesNotMoveTheTargetCenter:
    def test_eval_leaves_the_center_bit_identical(self):
        model = _model()
        model.train()
        _step(model)  # give the EMA something to hold
        before = model.target_centering.center.detach().clone()
        assert before.norm() > 0, "fixture did not populate the center"

        model.eval()
        with torch.no_grad():
            _step(model)
        assert torch.equal(model.target_centering.center, before), (
            "eval() moved target_centering.center (max delta "
            f"{(model.target_centering.center - before).abs().max().item():.6g})"
        )

    def test_repeated_eval_is_a_pure_function(self):
        """Two identical eval() calls: same loss, same center.

        Before the fix the first eval call folded the batch mean into the EMA,
        so the second call saw a different (and wrong) subtraction target.
        """
        model = _model()
        model.train()
        _step(model)

        model.eval()
        with torch.no_grad():
            first = _step(model)
            center_after_first = model.target_centering.center.detach().clone()
            second = _step(model)

        assert torch.equal(
            first, second
        ), f"two identical eval() calls disagreed: {first.item()} vs {second.item()}"
        assert torch.equal(model.target_centering.center, center_after_first)

    def test_no_buffer_moves_under_eval(self):
        """Mechanism-free model: `target_centering.center` was the last
        buffer `no_grad()` let through, so the whole set must now be frozen."""
        model = _model(use_jawp=False)
        model.train()
        _step(model)
        before = _buffers(model)
        assert len(before) > 0

        for _ in range(3):
            _validate_call(model)
        assert not _moved(before, _buffers(model))

    def test_validation_pass_does_not_perturb_the_training_trajectory(self):
        """The headline consequence, at model level.

        Two runs from the same seed -- one interrupted by a validation call --
        must produce the same weights, the same center, and bit-identical
        losses for every subsequent training step.
        """
        clean = _model(use_jawp=False)
        dirty = _model(use_jawp=False)
        assert _moved(_buffers(clean), _buffers(dirty)) == []

        clean_losses, dirty_losses = [], []
        for step in range(3):
            clean.train()
            clean_losses.append(_step(clean).item())
            if step == 0:
                _validate_call(dirty)  # the interruption
            dirty.train()
            dirty_losses.append(_step(dirty).item())

        assert clean_losses == dirty_losses, f"losses diverged: {clean_losses} vs {dirty_losses}"
        moved = _moved(_buffers(clean), _buffers(dirty))
        assert not moved, f"a validation call changed {len(moved)} buffers: {moved}"


class TestTrainingStillMovesTheTargetCenter:
    def test_training_updates_the_center(self):
        model = _model()
        model.train()
        _step(model)
        c1 = model.target_centering.center.detach().clone()
        _step(model)
        c2 = model.target_centering.center.detach().clone()
        assert not torch.equal(c1, c2), "guard over-corrected: training froze the center"

    def test_training_keeps_the_data2vec_ema_arithmetic(self):
        """Bit-exact: center <- m*center + (1-m)*mean(h_target).

        This is what makes the source-level guard an identity in training
        mode: the same expression, the same order, on the same tensor.
        """
        model = _model()
        model.train()
        _step(model)  # non-zero starting point

        restore, recorded = _spy(model)
        try:
            center_before = model.target_centering.center.detach().clone()
            m = model.target_centering.momentum
            _step(model)
        finally:
            restore()

        expected = m * center_before + (1 - m) * recorded["raw_target"].mean(
            dim=(0, 1), keepdim=True
        )
        assert torch.equal(model.target_centering.center, expected), (
            "the EMA update changed: max delta "
            f"{(model.target_centering.center - expected).abs().max().item():.6g}"
        )

    def test_layer_norm_still_comes_after_the_centering(self):
        """Ordering pin: the predictor must see layer_norm(h - center).

        Guards the two ways the fix could silently change what is predicted:
        centering after layer_norm, or subtracting the pre-update center.
        """
        model = _model()
        model.train()
        _step(model)

        restore, recorded = _spy(model)
        try:
            _step(model)
        finally:
            restore()

        raw = recorded["raw_target"]
        center = model.target_centering.center.detach().clone()
        expected = F.layer_norm(raw - center, (raw.size(-1),))
        assert torch.equal(recorded["predictor_target"], expected)

    def test_eval_centers_with_the_frozen_center(self):
        """The guard is on the *write*, not on the *use*.

        A fix that skipped the subtraction under eval would pass a
        frozen-buffer test while making the eval loss a different quantity.
        """
        model = _model()
        model.train()
        _step(model)

        model.eval()
        center_before = model.target_centering.center.detach().clone()
        restore, recorded = _spy(model)
        try:
            with torch.no_grad():
                _step(model)
        finally:
            restore()

        raw = recorded["raw_target"]
        expected = F.layer_norm(raw - center_before, (raw.size(-1),))
        assert torch.equal(
            recorded["predictor_target"], expected
        ), "eval() did not center with the frozen center"
        assert torch.equal(model.target_centering.center, center_before)


class TestTheWriteIsRoutedThroughTheSharedGuard:
    def test_model_inherits_the_shared_guard(self):
        from src.models._state_guard import TrainingStateGuard

        model = _model()
        assert isinstance(model, TrainingStateGuard)

    def test_jepa_routes_state_writes_through_mutate_state(self):
        """One grep has to list every state write; the model must be in it."""
        src = (Path(__file__).resolve().parent.parent / "src" / "models" / "jepa.py").read_text()
        assert "_mutate_state(" in src, "jepa.py writes state without the shared guard"
