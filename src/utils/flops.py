# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# FLOPs computation for scaling analysis
# Based on Kaplan et al. (2020) scaling laws + PaLM compute formulas


def estimate_transformer_flops(
    num_params: int,
    seq_len: int,
    batch_size: int = 1,
    forward_only: bool = False,
) -> dict[str, float]:
    """Estimate FLOPs for a transformer forward+backward pass.

    From Kaplan et al. (2020): C ≈ 6N for forward+backward
    (where N = number of parameters), times sequence length and batch size.

    More precise: C = 2 * P * L * B (forward) + 4 * P * L * B (backward)
    where P = params, L = seq_len, B = batch_size

    But for transformer with attention:
    C_forward ≈ 2 * N * L * B + 2 * n_layers * L^2 * d * B
    (first term: linear layers, second term: attention QK^T)

    Args:
        num_params: total number of model parameters
        seq_len: sequence length
        batch_size: batch size
        forward_only: if True, only count forward pass

    Returns:
        dict with FLOPs estimates

    """
    # Kaplan estimate: 6N for forward+backward per token
    flops_per_token = 6 * num_params
    if forward_only:
        flops_per_token = 2 * num_params

    total_flops = flops_per_token * seq_len * batch_size

    return {
        "total_flops": float(total_flops),
        "flops_per_token": float(flops_per_token),
        "flops_per_sample": float(flops_per_token * seq_len),
        "gflops": total_flops / 1e9,
        "tflops": total_flops / 1e12,
    }


def estimate_training_flops(
    num_params: int,
    seq_len: int,
    batch_size: int,
    num_steps: int,
) -> dict[str, float]:
    """Estimate total FLOPs for a training run.

    Args:
        num_params: number of trainable parameters
        seq_len: sequence length
        batch_size: batch size
        num_steps: total training steps

    Returns:
        dict with total FLOPs for the training run

    """
    per_step = estimate_transformer_flops(num_params, seq_len, batch_size)
    total = per_step["total_flops"] * num_steps

    return {
        "total_flops": total,
        "gflops": total / 1e9,
        "tflops": total / 1e12,
        "pflops": total / 1e15,
        "per_step_gflops": per_step["gflops"],
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
