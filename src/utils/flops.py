# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""FLOP accounting for scaling analysis.

Two estimators live here, and the difference between them matters.

`estimate_transformer_flops` is the textbook Kaplan et al. (2020) approximation
``C ~= 6 * N * L * B``. It is kept because it is the number a reader expects,
but it is only correct when ``N`` is "the parameters that live inside
matmuls" and the model is a single transformer stack. It cannot see attention,
a target encoder, a predictor that runs more than once, or a decoder head. Fed
the parameters of a Text-Span JEPA step it is wrong by ~2x, and the error grows
with sequence length -- see the test in `tests/test_model.py`.

`estimate_jepa_step_flops` is the structural estimator: it takes the model
dimensions and counts the matmuls the step actually executes. It was written
against `torch.utils.flop_counter.FlopCounterMode` and reproduces it to within
0.04% on the real model at base_140m dims (768/12, predictor 384/6, vocab 4096,
B=4, mask_ratio 0.35), with the per-step diagnostics switched off:

    T     FlopCounterMode      estimate_jepa_step_flops    est/actual
    128   5.530159e+11         5.529057e+11                0.99980
    256   1.111817e+12         1.111460e+12                0.99968
    512   2.229419e+12         2.228600e+12                0.99963

Four things the ``6 * N * L * B`` form cannot represent, all of them in the
signature below:

1. the **target encoder**, a second full forward of the encoder under
   `torch.no_grad()` -- no backward, so 1x not 3x, but 15% of the step;
2. the **predictor**, which runs `num_refine_steps` passes over the whole
   sequence plus one more per live future offset over ``L - d`` tokens, at its
   own width and depth;
3. the **decoder** head, applied only at masked positions, and the VICReg
   covariance term;
4. the **per-step diagnostics** (`CollapseDiagnostics.compute` and
   `JSpaceMetrics.compute`, both called unconditionally), a further 1.5-2% of
   the step at these dims -- not modelled here, see the function docstring.

Why `attention_flops` is reported but NOT folded into `total_flops`
-------------------------------------------------------------------
`FlopCounterMode` on torch 2.13.0+cpu does **not** count attention at all: both
attention paths in this repo lower to a fused kernel that the counter's
formula table has no entry for, and it reports exactly 0. Measured directly:

    F.scaled_dot_product_attention(q, q, q)      -> 0.0 FLOPs
    nn.MultiheadAttention(...)(z, z, z)          -> in_proj + out_proj only

So the reference number this module is pinned against contains no attention
term. Folding attention into `total_flops` would move the estimate *away* from
that reference (0.98 -> 0.87 at T=512), not toward it. Reporting the two
separately is the honest arrangement: `total_flops` is the matmul work a
profiler on this build will report, `attention_flops` is the O(L^2) work it
will not. A compute budget needs the sum; a comparison against torch's counter
needs the first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "estimate_jepa_step_flops",
    "estimate_training_flops",
    "estimate_transformer_flops",
    "model_size_category",
]

# A forward pass costs 1x; backward costs 2x on top, for a 3x total. This is
# exact for every Linear, and verified against FlopCounterMode on every block
# in this repo (encoder, predictor, decoder: ratio 3.00000).
_FWD_BWD_MULTIPLIER = 3.0


def _stack_linear_flops(n_tokens: int, dim: int, mlp_ratio: float, n_layers: int) -> float:
    """Matmul FLOPs of the Linear layers in `n_layers` pre-norm transformer blocks.

    Per token per block, counting a multiply-accumulate as 2 FLOPs:
        qkv     dim -> 3*dim          6*dim^2
        proj    dim -> dim            2*dim^2
        mlp     dim -> r*dim -> dim  16*dim^2   (r = 4)
                                          -------
                                          2 * dim^2 * (4 + 2*r)

    Attention is excluded; see the module docstring.
    """
    return 2.0 * n_tokens * dim * dim * (4.0 + 2.0 * mlp_ratio) * n_layers


def _stack_attention_flops(batch: int, seq_len: int, dim: int, n_layers: int) -> float:
    """O(seq_len^2) QK^T + AV FLOPs of `n_layers` blocks, one pass.

    2 * batch * heads * seq^2 * head_dim for each of QK^T and AV, and
    heads * head_dim == dim, so 4 * batch * seq^2 * dim per layer.
    """
    return 4.0 * batch * seq_len * seq_len * dim * n_layers


def _live_offsets(future_offsets: Iterable[int], seq_len: int) -> list[int]:
    """Future offsets that actually run: `predictor.forward_future_prediction`
    skips any offset at or beyond the sequence length."""
    return [d for d in future_offsets if d < seq_len]


def estimate_transformer_flops(
    num_params: int,
    seq_len: int,
    batch_size: int = 1,
    forward_only: bool = False,
    *,
    embed_dim: int | None = None,
    num_layers: int | None = None,
    mlp_ratio: float = 4.0,
) -> dict[str, Any]:
    """Estimate FLOPs for a transformer forward pass, with or without backward.

    Two methods, selected by whether the stack geometry is supplied:

    ``structural``
        `embed_dim` and `num_layers` are given, so the matmuls of the stack are
        counted exactly: ``2 * B * L * D^2 * (4 + 2*mlp_ratio) * num_layers``
        forward, three times that for forward+backward. `num_params` is unused.

    ``kaplan_6nd``
        Otherwise, the Kaplan et al. (2020) approximation ``6 * N * L * B``
        (or ``2 * N`` forward-only). This is an approximation over a parameter
        count, not a measurement: it cannot see attention (O(L^2)), embeddings,
        a target encoder, or a module that runs more than once per step. Use it
        for order-of-magnitude scaling plots, never as a step cost. The returned
        ``method`` key says which branch produced the number.

    Args:
        num_params: total parameter count, used only by the Kaplan branch.
        seq_len: sequence length.
        batch_size: batch size.
        forward_only: if True, count the forward pass only.
        embed_dim: model width, required for the structural branch.
        num_layers: block count, required for the structural branch.
        mlp_ratio: MLP expansion factor.

    Returns:
        dict of FLOPs estimates, including ``method``.
    """
    if embed_dim is not None and num_layers is not None:
        per_token = 2.0 * embed_dim * embed_dim * (4.0 + 2.0 * mlp_ratio) * num_layers
        if not forward_only:
            per_token *= _FWD_BWD_MULTIPLIER
        total_flops = per_token * seq_len * batch_size
        method = "structural"
    else:
        per_token = 2.0 * num_params if forward_only else 6.0 * num_params
        total_flops = per_token * seq_len * batch_size
        method = "kaplan_6nd"

    return {
        "total_flops": float(total_flops),
        "flops_per_token": float(per_token),
        "flops_per_sample": float(per_token * seq_len),
        "gflops": total_flops / 1e9,
        "tflops": total_flops / 1e12,
        "method": method,
    }


def estimate_jepa_step_flops(
    *,
    embed_dim: int,
    encoder_depth: int,
    predictor_embed_dim: int,
    predictor_depth: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    mlp_ratio: float = 4.0,
    predictor_mlp_ratio: float = 4.0,
    num_refine_steps: int = 0,
    future_offsets: Sequence[int] = (),
    mask_ratio: float = 0.0,
    forward_only: bool = False,
) -> dict[str, float]:
    """Structural FLOP estimate for one `TextSpanJEPA.compute_loss_with_targets` step.

    Counts the four matmul groups the step actually runs, which is what the
    ``6 * N * L * B`` form of `estimate_transformer_flops` cannot represent:

    1. the encoder, forward + backward;
    2. the **target encoder**, a second full forward under `torch.no_grad()`
       -- it has no backward, so it is 1x, not 3x, and it is not free either;
    3. the **predictor**, which runs `num_refine_steps` passes over the whole
       sequence plus one further pass per live future offset over ``L - d``
       tokens, with its own width and depth;
    4. the **decoder** head, applied only at masked positions, and the VICReg
       covariance term ``2 * B * L * D^2``.

    Deliberately NOT counted: the per-step diagnostics
    (`CollapseDiagnostics.compute`, `JSpaceMetrics.compute`), which
    `compute_loss_with_targets` also calls unconditionally. They are a measured
    1.5-2% of the step at base_140m dims, but the count of D-by-D Gram matrices
    they issue varies with the number of active metrics (13.6x, 15.3x, 18.2x
    `2*B*T*D^2` at T=128/256/512), so it is not a fixed coefficient that can be
    stated without re-measuring. They are instrumentation, not model compute;
    a scaling plot that wants the wall-clock-inclusive number should add its own
    measured diagnostics term. This function reproduces the FlopCounterMode
    reading with diagnostics disabled to within 0.04%.

    Args:
        embed_dim: encoder width.
        encoder_depth: encoder block count.
        predictor_embed_dim: predictor width.
        predictor_depth: predictor block count.
        seq_len: sequence length.
        batch_size: batch size.
        vocab_size: vocabulary, for the tied decoder output projection.
        mlp_ratio: encoder MLP expansion.
        predictor_mlp_ratio: predictor MLP expansion.
        num_refine_steps: iterative refinement passes over the full sequence.
        future_offsets: future-prediction offsets; those >= seq_len never run.
        mask_ratio: fraction of positions masked, driving the decoder head.
        forward_only: if True, count forward passes only.

    Returns:
        dict with a per-component breakdown. ``total_flops`` excludes both
        attention (see the module docstring) and the diagnostics term above;
        ``attention_flops`` reports the former separately.
    """
    offsets = _live_offsets(future_offsets, seq_len)

    encoder = _stack_linear_flops(batch_size * seq_len, embed_dim, mlp_ratio, encoder_depth)
    target_encoder = encoder  # same stack, forward only -- `jepa.py` runs it under no_grad

    # `_iterative_refine` runs num_refine_steps full-length passes; each future
    # offset adds one pass over (seq_len - offset) tokens.
    block_tokens = num_refine_steps * seq_len + sum(seq_len - d for d in offsets)
    # `predictor_embed` and `predictor_proj` each run once over the whole
    # sequence (span pass) and once per future offset.
    proj_tokens = seq_len + sum(seq_len - d for d in offsets)
    predictor = (
        _stack_linear_flops(
            batch_size * block_tokens, predictor_embed_dim, predictor_mlp_ratio, predictor_depth
        )
        + 4.0 * batch_size * embed_dim * predictor_embed_dim * proj_tokens
    )

    n_masked = round(batch_size * seq_len * mask_ratio)
    decoder = 2.0 * n_masked * (4.0 * embed_dim * embed_dim + embed_dim * vocab_size)
    covariance = 2.0 * batch_size * seq_len * embed_dim * embed_dim

    trainable = encoder + predictor + decoder + covariance
    total_flops = trainable + target_encoder
    if not forward_only:
        total_flops = trainable * _FWD_BWD_MULTIPLIER + target_encoder

    attention_tokens = [seq_len] * num_refine_steps + [seq_len - d for d in offsets]
    attention = _stack_attention_flops(batch_size, seq_len, embed_dim, encoder_depth) + 4.0 * (
        batch_size * predictor_embed_dim * predictor_depth
    ) * sum(t * t for t in attention_tokens)
    attention_flops = attention if forward_only else attention * _FWD_BWD_MULTIPLIER

    return {
        "total_flops": float(total_flops),
        "attention_flops": float(attention_flops),
        "encoder_flops": float(encoder * (1.0 if forward_only else _FWD_BWD_MULTIPLIER)),
        "target_encoder_flops": float(target_encoder),
        "predictor_flops": float(predictor * (1.0 if forward_only else _FWD_BWD_MULTIPLIER)),
        "decoder_flops": float(decoder * (1.0 if forward_only else _FWD_BWD_MULTIPLIER)),
        "covariance_flops": float(covariance * (1.0 if forward_only else _FWD_BWD_MULTIPLIER)),
        "gflops": total_flops / 1e9,
        "tflops": total_flops / 1e12,
    }


def estimate_training_flops(flops_per_step: float, num_steps: int) -> dict[str, float]:
    """Total FLOPs for a run of `num_steps` identical steps.

    Takes the cost of one step rather than a parameter count, so that the
    caller cannot accidentally reintroduce the Kaplan `6ND` guess at this
    level as well. Pass `estimate_jepa_step_flops(...)["total_flops"]`.

    Args:
        flops_per_step: FLOPs of one training step.
        num_steps: total training steps.

    Returns:
        dict of total FLOPs for the run.
    """
    total = float(flops_per_step) * num_steps
    return {
        "total_flops": total,
        "gflops": total / 1e9,
        "tflops": total / 1e12,
        "pflops": total / 1e15,
        "per_step_gflops": float(flops_per_step) / 1e9,
        "num_steps": num_steps,
    }


def model_size_category(num_params: int) -> str:
    """Categorize model by size for scaling analysis."""
    if num_params < 10e6:
        return "tiny"
    elif num_params < 100e6:
        return "small"
    elif num_params < 500e6:
        return "base"
    elif num_params < 2e9:
        return "large"
    else:
        return "xl"
