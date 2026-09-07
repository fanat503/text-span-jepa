# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""GradScaler hardening gates (bugs B2 + B1 from the train-loop audit).

B2: bfloat16 (the only AMP mode in this repo) requires no loss scaling, so
GradScaler must never be enabled — an enabled scaler silently multiplies
every gradient by ~65536.

B1: clip_grad_norm_ in train.py runs BEFORE any unscale step; with a disabled
(pass-through) scaler the clip sees UNSCALED gradients. With an enabled one
the effective clip was 1.0/65536 ≈ every step clipped to ~1.5e-5 norm.

Discrimination note: on CPU-only hosts `GradScaler("cuda", enabled=True)`
does NOT raise — torch silently disables it with a UserWarning. Therefore
`assert scaler.is_enabled() is False` passes on the OLD code too; the
discriminator is a constructor spy capturing the `enabled` kwarg.
"""

import types

import torch

from src.train import _build_optimization


def _build(meta_use_bf16=True):
    model = torch.nn.Linear(4, 2)
    args = {
        "optimization": {"epochs": 1, "warmup": 1},
        "meta": {"use_bfloat16": meta_use_bf16},
    }
    # model_name="other" -> get_param_groups fallback branch, ema_scheduler=None.
    out = _build_optimization(args, model, "other", {}, types.SimpleNamespace(type="cuda"), ipe=10)
    return out[4]  # scaler


def test_grad_scaler_is_never_enabled(monkeypatch):
    """B2 regression: bf16 needs no loss scaling; GradScaler must be disabled."""
    captured, real = {}, torch.amp.GradScaler

    def spy(device_type, **kwargs):
        captured.update(kwargs)
        return real(device_type, enabled=False)

    monkeypatch.setattr(torch.amp, "GradScaler", spy)
    _build()
    assert captured.get("enabled") is not True, (
        "GradScaler must never be enabled: bf16 needs no loss scaling "
        "(B2 is the root cause of the B1 effective-clip bug)"
    )


def test_scaler_is_strict_passthrough():
    """B1 invariant: train.py's clip sees UNSCALED gradients."""
    scaler = _build()
    t = torch.tensor(2.0)
    assert scaler.is_enabled() is False
    assert scaler.get_scale() == 1.0
    assert scaler.scale(t) is t  # identity -> backward graph is never scaled


def test_validate_forwards_current_step(monkeypatch):
    """B10: validation must see the schedule position (else future-weight = 0)."""
    from src.train import _validate

    captured = {}

    def fake_compute_loss(
        model, masked, original, mask, current_step=0, total_steps=1, want_diag=True
    ):
        captured.update(step=current_step, total=total_steps, want_diag=want_diag)
        return torch.tensor(0.5), {}, {}

    monkeypatch.setattr("src.train.compute_loss", fake_compute_loss)
    batch = {"input_ids": torch.randint(0, 10, (1, 8))}
    collated = {
        "masked_input_ids": batch["input_ids"].clone(),
        "original_input_ids": batch["input_ids"],
        "mask_positions": torch.zeros(1, 8, dtype=torch.bool),
    }
    _validate(
        types.SimpleNamespace(eval=lambda: None, train=lambda: None),
        [batch],
        lambda b: collated,
        torch.device("cpu"),
        "text_span_jepa",
        max_batches=1,
        current_step=7,
        total_steps=100,
    )
    assert captured == {"step": 7, "total": 100, "want_diag": False}
