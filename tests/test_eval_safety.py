# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Eval-safety and diagnostics-gating gates (improve-2 / bugs B9, B19).

Before these fixes a validation pass mutated live training state:
- TargetCentering absorbed validation batches into the running center;
- WSD resynced its target workspace on validation data (step=0 satisfies
  the sync-interval check) and rewound step_count;
- JAWP reset active_k to k_start on every validation pass, and the epoch
  checkpoint saved right after validation carried that polluted state.

The SVD/CKA diagnostics pack (18 full SVDs at bs=64/T=512) also ran on
every forward — including the CMC second pass and validation, which both
discarded it (95-99% of step wall-time, B19).
"""

import math

import pytest
import torch

from src.models.collapse import CollapseDiagnostics, TargetCentering
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig


def test_target_centering_not_updated_in_eval():
    """B9a: validation batches must not move the running center."""
    centering = TargetCentering(dim=8)
    x_train = torch.randn(2, 4, 8)
    x_val = torch.randn(2, 4, 8) + 10.0  # far away — pollution would be obvious

    centering.train()
    centering(x_train)
    center_after_train = centering.center.clone()

    centering.eval()
    centering(x_val)
    assert torch.equal(
        centering.center, center_after_train
    ), "eval forward must not update the running center"


def test_target_centering_updated_in_train():
    """Sanity: training forwards still update the running center."""
    centering = TargetCentering(dim=8, momentum=0.5)
    x = torch.randn(2, 4, 8)
    centering.train()
    before = centering.center.clone()
    centering(x)
    assert not torch.equal(centering.center, before)


def test_diagnostics_subsample_bounds_rows():
    """B19: row subsample bounds O(N²)/O(N·D²) metrics deterministically."""
    diag = CollapseDiagnostics(diag_max_rows=64)
    online = torch.randn(16, 64, 8)  # N = 1024 rows > 64
    target = torch.randn(16, 64, 8)

    m1 = diag.compute(online, target)
    m2 = diag.compute(online, target)

    assert m1["cka_linear"] == m2["cka_linear"]  # deterministic (no RNG)
    for key in ("cka_linear", "cka_rbf", "effective_rank_online", "uniformity_online"):
        assert key in m1
        value = m1[key]
        assert not math.isnan(value) and abs(value) < 1e6  # finite, no blowup


def test_diag_gating_skips_pack_and_keeps_state():
    """B19: want_diag=False returns an empty diag_dict and does not touch
    _prev_target_h (the CMC second pass used to overwrite it)."""
    config = TextSpanJEPAConfig(
        embed_dim=32,
        num_heads=4,
        encoder_depth=1,
        predictor_depth=1,
        vocab_size=32,
        max_seq_len=8,
    )
    model = TextSpanJEPA(config)
    ids = torch.randint(0, 32, (2, 8))
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 2] = True

    _, _, diag_on = model.compute_loss_with_targets(ids, ids, mask, want_diag=True)
    assert diag_on, "want_diag=True must produce diagnostics"
    prev_after_diag = model._prev_target_h.clone()

    _, _, diag_off = model.compute_loss_with_targets(ids, ids, mask, want_diag=False)
    assert diag_off == {}
    assert torch.equal(
        model._prev_target_h, prev_after_diag
    ), "non-diag forward must not advance _prev_target_h"


@pytest.mark.parametrize("use_jawp", [True, False])
def test_full_model_eval_pass_leaves_state_untouched(use_jawp):
    """B9 end-to-end: an eval pass must not move centering/step state."""
    kwargs = {
        "embed_dim": 32,
        "num_heads": 4,
        "encoder_depth": 1,
        "predictor_depth": 1,
        "vocab_size": 32,
        "max_seq_len": 8,
    }
    if use_jawp:
        kwargs.update(use_jawp=True, jawp_k_start=1, jawp_k_end=4, jawp_curriculum_steps=8)
    model = TextSpanJEPA(TextSpanJEPAConfig(**kwargs))
    ids = torch.randint(0, 32, (2, 8))
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 3] = True

    model.train()
    model.compute_loss_with_targets(ids, ids, mask, current_step=8, total_steps=16)
    center_after_train = model.target_centering.center.clone()
    k_after_train = model.jawp.active_k.clone() if use_jawp else None

    model.eval()
    with torch.no_grad():
        for _ in range(3):  # several eval passes, step=0 each time
            model.compute_loss_with_targets(ids, ids, mask, current_step=0, total_steps=16)

    assert torch.equal(model.target_centering.center, center_after_train)
    if use_jawp:
        assert torch.equal(
            model.jawp.active_k, k_after_train
        ), "eval must keep the training curriculum position (B9)"


def test_wsd_no_resync_in_eval():
    """B9b: eval passes must not resync the target workspace nor rewind
    step_count (step=0 satisfied the sync-interval check)."""
    from src.models.wsd import WorkspaceSyncDrift

    wsd = WorkspaceSyncDrift(embed_dim=8, k=2, sync_interval=1)
    q = torch.randn(8, 2)
    h_train = torch.randn(2, 8)

    wsd.train()
    wsd.compute_drift(q, h_target=h_train, step=1)  # interval hit -> resync+init
    cov_after_train = wsd.target_cov.clone()
    step_after_train = wsd.step_count.clone()
    assert wsd.is_initialized.item() is True or bool(wsd.is_initialized)

    h_val = torch.randn(2, 8) + 10.0
    wsd.eval()
    for step in (0, 0, 0):  # step=0 used to satisfy the interval check
        wsd.compute_drift(q, h_target=h_val, step=step)

    assert torch.equal(wsd.target_cov, cov_after_train), "eval must not resync"
    assert torch.equal(wsd.step_count, step_after_train), "eval must not rewind step_count"
