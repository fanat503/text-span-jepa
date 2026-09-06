# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Causal intervention methods for mechanistic interpretability
# From Redwood Research causal scrubbing, Meng et al. (2022) activation patching,
# and Turner et al. (2023) activation steering

import torch
import torch.nn.functional as F


def direction_ablation(representations, direction):
    """Ablate a specific direction from representations.

    Projects out the component along `direction`, zeroing its influence.
    Used to test: "is this direction necessary for the model's behavior?"

    Args:
        representations: (B, T, D) or (B, D)
        direction: (D,) unit vector to ablate
    Returns:
        ablated: same shape as representations, with direction removed

    """
    if representations.dim() == 3:
        B, T, D = representations.shape
        flat = representations.reshape(B * T, D)
    else:
        flat = representations

    # Project out: x' = x - (x · d) * d
    direction = F.normalize(direction, dim=0)
    proj = (flat @ direction.unsqueeze(1)) * direction.unsqueeze(0)
    ablated = flat - proj

    if representations.dim() == 3:
        return ablated.reshape(B, T, D)
    return ablated


def feature_steering(representations, direction, scale=1.0):
    """Steer representations along a specific direction.

    Adds scaled direction to all representations.
    Used to test: "does adding this direction cause predictable behavior change?"

    Args:
        representations: (B, T, D) or (B, D)
        direction: (D,) direction to steer along
        scale: scaling factor (positive or negative)
    Returns:
        steered: same shape as representations, steered along direction

    """
    direction = F.normalize(direction, dim=0)
    steered = representations + scale * direction
    return steered


def activation_patching(source_reps, target_reps, patch_mask):
    """Patch specific positions from source into target representations.

    From Meng et al. (2022) causal tracing: replace activations at
    specific positions to test causal influence.

    Args:
        source_reps: (B, T, D) source representations (corrupted/counterfactual)
        target_reps: (B, T, D) target representations (clean)
        patch_mask: (B, T) boolean mask — True = patch from source
    Returns:
        patched: (B, T, D) mixed representations

    """
    mask = patch_mask.unsqueeze(-1).float()  # (B, T, 1)
    return target_reps * (1 - mask) + source_reps * mask


@torch.no_grad()
def intervention_predictability_score(
    model,
    input_ids,
    direction,
    probe_fn,
    scales=(-2, -1, 0, 1, 2),
    layer_idx=-1,
    device="cpu",
):
    """Measure how predictable the effect of an intervention is.

    Key metric for mechanistic interpretability: if steering along
    a JEPA feature causes a MONOTONIC change in some probe output,
    that feature has clean causal structure.

    Predictability score = |Spearman correlation between steering scale
    and probe output|. High = predictable, low = noisy.

    Args:
        model: Text-Span JEPA model (or any model with .encoder)
        input_ids: (B, T) input token IDs
        direction: (D,) direction to steer along
        probe_fn: callable(representations) -> scalar per sample
        scales: list of steering scales to test
        layer_idx: which layer's representations to steer (-1 = last)
        device: compute device

    Returns:
        dict with 'predictability', 'monotonicity', 'probe_values'

    """
    model.eval()
    input_ids = input_ids.to(device)
    direction = direction.to(device)

    probe_values = []
    for scale in scales:
        h, _ = model.encoder(input_ids)
        # Pool: mean over sequence
        h_pooled = h.mean(dim=1)  # (B, D)
        # Steer
        h_steered = feature_steering(h_pooled, direction, scale=scale)
        # Probe
        probe_out = probe_fn(h_steered)
        if isinstance(probe_out, torch.Tensor):
            probe_out = probe_out.mean().item()
        probe_values.append(probe_out)

    # Compute predictability: |Spearman r(scale, probe_value)|
    scales_t = torch.tensor(scales, dtype=torch.float32)
    probes_t = torch.tensor(probe_values, dtype=torch.float32)

    if probes_t.std() == 0 or scales_t.std() == 0:
        return {
            "predictability": 0.0,
            "monotonicity": 0.0,
            "probe_values": probe_values,
        }

    # Spearman rank correlation
    rs = scales_t.argsort().argsort().float()
    rp = probes_t.argsort().argsort().float()
    rs = rs - rs.mean()
    rp = rp - rp.mean()
    denom = rs.norm() * rp.norm()
    spearman = (rs @ rp / denom).item() if denom > 0 else 0.0

    # Monotonicity: fraction of consecutive pairs that go in the same direction
    diffs = probes_t[1:] - probes_t[:-1]
    if diffs.abs().sum() == 0:
        mono = 0.0
    else:
        # Consistent sign = monotonic
        signs = (diffs > 0).float()
        mono = max(signs.mean().item(), (1 - signs.mean()).item())

    return {
        "predictability": abs(spearman),
        "monotonicity": mono,
        "probe_values": probe_values,
    }


class CausalIntervention:
    """Collection of causal intervention experiments for comparing JEPA vs MLM."""

    def __init__(self, jepa_model, baseline_model, device="cpu"):
        self.jepa = jepa_model
        self.baseline = baseline_model
        self.device = device

    @torch.no_grad()
    def ablation_comparison(self, input_ids, directions, probe_fn):
        """Compare how ablation affects JEPA vs baseline.

        For each direction: ablate, run probe, measure effect.
        JEPA hypothesis: ablation of JEPA features has more predictable
        (larger, more consistent) effects than ablation of MLM features.

        Args:
            input_ids: (B, T) token IDs
            directions: dict of {name: (D,) tensor}
            probe_fn: callable(pooled_repr) -> scalar

        Returns:
            dict with per-direction results for both models

        """
        results = {}
        input_ids = input_ids.to(self.device)

        # Get baseline (no ablation) probe values
        h_jepa, _ = self.jepa.encoder(input_ids)
        h_base, _ = self.baseline.encoder(input_ids)
        jepa_baseline = probe_fn(h_jepa.mean(dim=1))
        base_baseline = probe_fn(h_base.mean(dim=1))

        for name, direction in directions.items():
            direction = direction.to(self.device)

            # JEPA ablation
            h_jepa_abl = direction_ablation(h_jepa.mean(dim=1), direction)
            jepa_probed = probe_fn(h_jepa_abl)
            jepa_effect = (
                (jepa_probed - jepa_baseline).abs().mean().item()
                if isinstance(jepa_probed, torch.Tensor)
                else abs(jepa_probed - jepa_baseline)
            )

            # Baseline ablation
            h_base_abl = direction_ablation(h_base.mean(dim=1), direction)
            base_probed = probe_fn(h_base_abl)
            base_effect = (
                (base_probed - base_baseline).abs().mean().item()
                if isinstance(base_probed, torch.Tensor)
                else abs(base_probed - base_baseline)
            )

            results[name] = {
                "jepa_ablation_effect": jepa_effect,
                "baseline_ablation_effect": base_effect,
                "jepa_more_predictable": jepa_effect > base_effect,
            }

        return results

    @torch.no_grad()
    def steering_comparison(self, input_ids, direction, probe_fn, scales=(-2, -1, 0, 1, 2)):
        """Compare steering predictability between JEPA and baseline.

        Returns:
            dict with predictability scores for both models

        """
        direction = direction.to(self.device)

        jepa_result = intervention_predictability_score(
            self.jepa,
            input_ids,
            direction,
            probe_fn,
            scales,
            device=self.device,
        )

        baseline_result = intervention_predictability_score(
            self.baseline,
            input_ids,
            direction,
            probe_fn,
            scales,
            device=self.device,
        )

        return {
            "jepa_predictability": jepa_result["predictability"],
            "jepa_monotonicity": jepa_result["monotonicity"],
            "baseline_predictability": baseline_result["predictability"],
            "baseline_monotonicity": baseline_result["monotonicity"],
            "jepa_more_predictable": jepa_result["predictability"]
            > baseline_result["predictability"],
        }
