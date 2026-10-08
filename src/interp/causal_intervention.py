# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Causal intervention methods for mechanistic interpretability
# From Redwood Research causal scrubbing, Meng et al. (2022) activation patching,
# and Turner et al. (2023) activation steering
#
# CAUSALITY CONTRACT
# -----------------
# Every function in this module that claims a causal effect performs the edit
# *inside the forward pass*: a `register_forward_hook` on the encoder block at
# `layer_idx` rewrites that block's output, so every later block and the final
# norm consume the modified activation. Editing the pooled output vector after
# the forward pass is NOT a causal intervention — it measures linear algebra on
# a feature vector, not what the model would do.
#
# Every reported verdict is a test, not a magnitude comparison:
#   - steering/ablation effects are compared against a random-direction null
#   - the scale -> probe Spearman is tested by permutation, p-value reported
#   - intervention scales are expressed in units of the model's own activation
#     norm, so a model with larger representations is not "more causal"
#   - if the hook cannot be installed or does not fire, we RAISE

import contextlib
import itertools
import math

import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════
# Pure tensor ops (kept as-is: these are the primitives, not the experiments)
# ═══════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════
# Real in-forward-pass interventions
# ═══════════════════════════════════════════════════════════════════


def _as_output_tensor(output):
    """Coerce a module's forward output to the activation tensor we edit."""
    if isinstance(output, torch.Tensor):
        return output
    if (
        isinstance(output, (tuple, list))
        and len(output) == 1
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]
    raise TypeError(
        "causal intervention expects the hooked module to return a tensor "
        f"(or a 1-tuple containing one), got {type(output)}",
    )


class _InterventionHandle:
    """Forward hook that rewrites one encoder block's output.

    `fired` counts how many times the block actually ran under the hook. Callers
    MUST check it: a hook that never fired means no intervention happened, and
    every number derived from it would be a null result.
    """

    def __init__(self, block, direction, mode, scale):
        if mode not in ("ablate", "steer"):
            raise ValueError(f"mode must be 'ablate' or 'steer', got {mode!r}")
        self.block = block
        self.direction = direction
        self.mode = mode
        self.scale = float(scale)
        self.fired = 0

    def apply(self, x):
        d = self.direction.to(device=x.device, dtype=x.dtype)
        if self.mode == "ablate":
            return direction_ablation(x, d)
        return x + self.scale * F.normalize(d, dim=0)

    def __call__(self, module, inputs, output):
        self.fired += 1
        return self.apply(_as_output_tensor(output))


def _resolve_block(model, layer_idx):
    """Locate the encoder block to intervene on.

    Raises instead of falling back: a silently skipped hook is exactly the bug
    this module used to have (`layer_idx` was accepted and ignored).
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise TypeError(f"model {type(model).__name__} has no `.encoder` to intervene on")
    blocks = getattr(encoder, "blocks", None)
    if blocks is None or not hasattr(blocks, "__len__") or len(blocks) == 0:
        raise TypeError(
            f"{type(encoder).__name__} exposes no non-empty `.blocks`; a real "
            "intervention cannot be installed on this model",
        )
    n = len(blocks)
    idx = layer_idx if layer_idx >= 0 else n + layer_idx
    if not 0 <= idx < n:
        raise IndexError(f"layer_idx={layer_idx} out of range for an encoder with {n} blocks")
    return blocks[idx], idx


@contextlib.contextmanager
def activation_intervention(model, direction, mode="ablate", layer_idx=-1, scale=1.0):
    """Install a real forward hook on encoder block `layer_idx`.

    While the context is active, the block's *output* is rewritten, so every
    downstream block and the final norm see the modified activation. This is a
    causal intervention; editing the returned representation afterwards is not.

    Args:
        model: module exposing `.encoder.blocks`
        direction: (D,) direction in activation space
        mode: "ablate" (project the direction out) or "steer" (add scale*d)
        layer_idx: block index; negative indexes count from the end (-1 = last)
        scale: steering magnitude in activation units (ignored when ablating)

    Yields:
        _InterventionHandle — check `.fired` to confirm the hook ran.

    """
    block, _idx = _resolve_block(model, layer_idx)
    handle = _InterventionHandle(block, direction, mode, scale)
    hook = block.register_forward_hook(handle)
    try:
        yield handle
    finally:
        hook.remove()


@torch.no_grad()
def _clean_block_output(model, input_ids, block):
    """Clean (unintervened) output of `block`, used as the scale reference."""
    captured = []

    def _record(_module, _inputs, output):
        captured.append(_as_output_tensor(output).detach())

    hook = block.register_forward_hook(_record)
    try:
        model.encoder(input_ids)
    finally:
        hook.remove()
    if not captured:
        raise RuntimeError(
            "encoder block never ran; cannot establish an activation-scale reference"
        )
    return captured[0]


@torch.no_grad()
def _intervened_pooled(model, input_ids, direction, mode, layer_idx, scale):
    """Run the model with the intervention installed and return pooled reps.

    Raises unless the hook fired exactly once — an intervention that did not
    execute is a null result, and reporting it as a measurement is the bug.
    """
    with activation_intervention(
        model, direction, mode=mode, layer_idx=layer_idx, scale=scale
    ) as handle:
        h, _ = model.encoder(input_ids)
    if handle.fired != 1:
        raise RuntimeError(
            f"intervention hook fired {handle.fired} times (expected 1): the "
            "model was not actually intervened on, so any effect measured here is zero by default",
        )
    return h, handle.fired


# ═══════════════════════════════════════════════════════════════════
# Statistics: a predictability claim needs a null
# ═══════════════════════════════════════════════════════════════════

MONOTONE_NULL = (
    "H0: the probe response is independent of the intervention scale "
    "(no monotone association between scale and probe output)."
)
ABLATION_NULL = (
    "H0: ablating this direction moves the probe no more than ablating a random "
    "direction of the same unit length."
)


def _spearman(x, y):
    """Spearman rank correlation of two 1-D float tensors (ties: ordinal ranks)."""
    if x.numel() < 2:
        return 0.0
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if denom <= 0:
        return 0.0
    return (rx @ ry / denom).item()


def _null_draw_count(n_points, n_permutations):
    """How many null samples the permutation test actually used."""
    if n_points <= 7:
        return math.factorial(n_points)
    return int(n_permutations)


def permutation_spearman_test(scales, values, n_permutations=1000, generator=None):
    """Two-sided permutation test of |Spearman(scales, values)|.

    At n=5 points, |r| > 0.9 happens by chance roughly 10% of the time, so a
    bare |r| is not evidence. Here the null is the distribution of |r| over all
    permutations of the observed probe values (enumerated exactly for n <= 7,
    sampled beyond that). Note the resulting power limit: at n=5 only 2 of the
    120 permutations are perfectly monotone, so even a flawless signal can only
    reach p = 2/120 = 0.0167.

    Returns:
        (r_observed, p_value) with p_value in (0, 1].

    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    s = torch.as_tensor(list(scales), dtype=torch.float32)
    v = torch.as_tensor(list(values), dtype=torch.float32)
    if s.numel() != v.numel():
        raise ValueError("scales and values must have the same length")
    if s.numel() < 2 or s.std() == 0 or v.std() == 0:
        return 0.0, 1.0  # no signal to test

    r_obs = _spearman(s, v)
    target = abs(r_obs)

    n = int(v.numel())
    if n <= 7:  # exact null: 7! = 5040 permutations
        count = 0
        total = 0
        for perm in itertools.permutations(range(n)):
            total += 1
            if abs(_spearman(s, v[list(perm)])) >= target - 1e-12:
                count += 1
        return r_obs, count / total

    if generator is None:
        generator = torch.Generator().manual_seed(0)
    count = 0
    for _ in range(n_permutations):
        perm = torch.randperm(n, generator=generator)
        if abs(_spearman(s, v[perm])) >= target - 1e-12:
            count += 1
    return r_obs, (count + 1) / (n_permutations + 1)


def _binomial_two_sided(k, n):
    """Exact two-sided binomial tail under p=0.5 (no scipy dependency).

    Two-sided p = 2 * P(X <= min(k, n-k)) with p=0.5, capped at 1.
    """
    if n == 0:
        return 1.0
    m = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def monotonicity_excess(values):
    """Agreement of consecutive differences with the majority sign, above chance.

    The previous implementation returned max(frac>0, 1-frac>0), which has a floor
    of 0.5 for *any* non-constant signal — a random walk scored 0.5-0.75 and was
    reported as a "predictability" number with no null. Here the chance level is
    subtracted and the result rescaled, so 0.0 means "no better than chance" and
    1.0 means "perfectly monotone".

    Returns:
        (excess in [0, 1], exact two-sided binomial p-value under p=0.5)

    """
    v = torch.as_tensor(list(values), dtype=torch.float32)
    if v.numel() < 3:
        return 0.0, 1.0
    diffs = v[1:] - v[:-1]
    n = int(diffs.numel())
    if float(diffs.abs().sum()) == 0:
        return 0.0, 1.0
    n_pos = int((diffs > 0).sum())
    maj = max(n_pos, n - n_pos)
    return (maj - n / 2.0) / (n / 2.0), _binomial_two_sided(maj, n)


def _probe_scalar(probe_fn, pooled):
    """Apply `probe_fn` and reduce to a float (probes may return per-sample values)."""
    out = probe_fn(pooled)
    if isinstance(out, torch.Tensor):
        return float(out.mean())
    return float(out)


def _match_scale(pooled):
    """Project pooled representations onto the unit sphere.

    Two models that differ only in representation *norm* produce identical
    probe inputs after this, so an effect-size comparison between them is a
    comparison of causal structure rather than of embedding scale.
    """
    return F.normalize(pooled, dim=-1)


def _effect(control, probed):
    """Absolute probe movement between the clean control and the intervened run."""
    if isinstance(control, torch.Tensor) or isinstance(probed, torch.Tensor):
        c = control if isinstance(control, torch.Tensor) else torch.as_tensor(float(control))
        p = probed if isinstance(probed, torch.Tensor) else torch.as_tensor(float(probed))
        return float((p - c).abs().mean())
    return abs(probed - control)


def _null_p_value(observed, null_values):
    """Empirical p-value of `observed` against a null sample (plus-one corrected)."""
    null = list(null_values)
    if not null:
        return 1.0
    count = sum(1 for v in null if abs(v) >= abs(observed) - 1e-12)
    return (count + 1) / (len(null) + 1)


# ═══════════════════════════════════════════════════════════════════
# Predictability of an intervention
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def intervention_predictability_score(
    model,
    input_ids,
    direction,
    probe_fn,
    scales=(-2, -1, 0, 1, 2),
    layer_idx=-1,
    device="cpu",
    seed=0,
    n_permutations=1000,
    alpha=0.05,
):
    """Measure how predictable the effect of a real intervention is.

    Steering is applied by a forward hook on encoder block `layer_idx`, so the
    probe reads representations produced by the *intervened* model rather than
    a hand-edited vector. `scales` are in units of the model's own clean
    activation norm at that block, so the same scale means the same
    perturbation magnitude for any model — a model with larger embeddings is
    not thereby "more predictable".

    Predictability = |Spearman(scale, probe output)|, reported together with a
    permutation p-value against H0 = no monotone association. At 5 points a bare
    |r| is not evidence, so the p-value is the load-bearing number and
    `significant` is what a caller should branch on.

    Args:
        model: Text-Span JEPA model (or any model with `.encoder.blocks`)
        input_ids: (B, T) input token IDs
        direction: (D,) direction to steer along
        probe_fn: callable(pooled_repr) -> scalar or per-sample tensor
        scales: steering scales, in units of the clean activation norm
        layer_idx: block to intervene on (-1 = last block)
        device: compute device
        seed: seed for the permutation null
        n_permutations: permutation draws (enumerated exactly when len(scales) <= 7)
        alpha: significance threshold for `significant`

    Returns:
        dict with 'predictability', 'monotonicity', 'probe_values', 'p_value',
        'significant', 'monotonicity_p_value', 'null_hypothesis', 'n_hook_calls',
        'n_null_draws', 'reference_norm', 'degenerate'

    """
    model.eval()
    input_ids = input_ids.to(device)
    direction = direction.to(device)
    block, _idx = _resolve_block(model, layer_idx)

    clean = _clean_block_output(model, input_ids, block)
    reference_norm = float(clean.norm(dim=-1).mean())
    if not reference_norm > 0:
        raise ValueError("clean activation norm is zero; cannot scale an intervention")

    probe_values = []
    n_hook_calls = 0
    for scale in scales:
        h, fired = _intervened_pooled(
            model,
            input_ids,
            direction,
            mode="steer",
            layer_idx=layer_idx,
            scale=scale * reference_norm,
        )
        n_hook_calls += fired
        probe_values.append(_probe_scalar(probe_fn, h.mean(dim=1)))

    scales_t = torch.tensor(list(scales), dtype=torch.float32)
    probes_t = torch.tensor(probe_values, dtype=torch.float32)
    n_null_draws = _null_draw_count(len(probe_values), n_permutations)

    if len(probe_values) < 2 or probes_t.std() == 0 or scales_t.std() == 0:
        # No signal: report a null result explicitly instead of |r| on noise.
        return {
            "predictability": 0.0,
            "monotonicity": 0.0,
            "monotonicity_p_value": 1.0,
            "monotonicity_significant": False,
            "p_value": 1.0,
            "significant": False,
            "probe_values": probe_values,
            "n_hook_calls": n_hook_calls,
            "n_permutations": int(n_permutations),
            "n_null_draws": n_null_draws,
            "reference_norm": reference_norm,
            "null_hypothesis": MONOTONE_NULL,
            "degenerate": True,
        }

    spearman, p_value = permutation_spearman_test(
        scales, probe_values, n_permutations, torch.Generator().manual_seed(seed)
    )
    mono, mono_p = monotonicity_excess(probe_values)

    return {
        "predictability": abs(spearman),
        "monotonicity": mono,
        "monotonicity_p_value": mono_p,
        "monotonicity_significant": mono_p < alpha,
        "p_value": p_value,
        "significant": p_value < alpha,
        "spearman": spearman,
        "probe_values": probe_values,
        "n_hook_calls": n_hook_calls,
        "n_permutations": int(n_permutations),
        "n_null_draws": n_null_draws,
        "reference_norm": reference_norm,
        "null_hypothesis": MONOTONE_NULL,
        "degenerate": False,
    }


class CausalIntervention:
    """Collection of causal intervention experiments for comparing JEPA vs MLM."""

    def __init__(self, jepa_model, baseline_model, device="cpu"):
        self.jepa = jepa_model
        self.baseline = baseline_model
        self.device = device

    @torch.no_grad()
    def _ablation_stats(self, model, input_ids, directions, probe_fn, layer_idx, null_dirs):
        """Clean vs ablated probe values, plus a random-direction null.

        The probe is applied to L2-normalised pooled representations, so the
        effect size is invariant to the model's representation norm. The null is
        the same measurement made with random unit directions, which is the
        matched control the previous "compare magnitudes" version lacked.
        """
        model.eval()
        block, _idx = _resolve_block(model, layer_idx)
        h_clean, _ = model.encoder(input_ids)
        control = probe_fn(_match_scale(h_clean.mean(dim=1)))

        per_direction = {}
        for name, direction in directions.items():
            h_abl, fired = _intervened_pooled(
                model, input_ids, direction, mode="ablate", layer_idx=layer_idx, scale=0.0
            )
            probed = probe_fn(_match_scale(h_abl.mean(dim=1)))
            per_direction[name] = {
                "effect": _effect(control, probed),
                "n_hook_calls": fired,
            }

        null_effects = []
        n_null_calls = 0
        for direction in null_dirs:
            h_null, fired = _intervened_pooled(
                model, input_ids, direction, mode="ablate", layer_idx=layer_idx, scale=0.0
            )
            probed = probe_fn(_match_scale(h_null.mean(dim=1)))
            null_effects.append(_effect(control, probed))
            n_null_calls += fired

        n_hook_calls = sum(d["n_hook_calls"] for d in per_direction.values()) + n_null_calls
        for name, entry in per_direction.items():
            effect = entry["effect"]
            null_mean = sum(null_effects) / len(null_effects) if null_effects else 0.0
            var = (
                sum((v - null_mean) ** 2 for v in null_effects) / (len(null_effects) - 1)
                if len(null_effects) > 1
                else 0.0
            )
            entry["effect_z"] = (effect - null_mean) / (var**0.5 + 1e-12)
            entry["effect_p"] = _null_p_value(effect, null_effects)
            entry["n_null_calls"] = n_null_calls
        return per_direction, n_hook_calls

    @torch.no_grad()
    def ablation_comparison(
        self,
        input_ids,
        directions,
        probe_fn,
        layer_idx=-1,
        n_random_directions=32,
        seed=0,
    ):
        """Compare how ablating a direction affects JEPA vs baseline.

        Each ablation is a forward hook at `layer_idx`, so both probe values come
        from intervened forward passes. Effects are measured on
        representation-scale-matched (L2-normalised) pooled outputs, and the
        verdict is a comparison of how unusual each effect is against a
        random-direction null — NOT `jepa_effect > baseline_effect`, which was
        decided by representation norm.

        NOTE: because the probe input is L2-normalised, a probe that is a
        function of the *norm* of the representation (`h.norm(dim=-1)`) becomes
        constant and every effect is exactly 0. Use a directional probe.

        Args:
            input_ids: (B, T) input token IDs
            directions: dict of {name: (D,) tensor}
            probe_fn: callable(pooled_repr) -> scalar or per-sample tensor
            layer_idx: encoder block to ablate at (-1 = last)
            n_random_directions: null sample size (shared by both models)
            seed: seed for the null directions (drawn once, used for both models)

        Returns:
            dict with per-direction results for both models

        """
        if n_random_directions < 1:
            raise ValueError(f"n_random_directions must be >= 1, got {n_random_directions}")
        input_ids = input_ids.to(self.device)
        directions = {k: v.to(self.device) for k, v in directions.items()}

        # Establish the representation width from a real forward pass.
        self.jepa.eval()
        h_probe, _ = self.jepa.encoder(input_ids)
        dim = h_probe.size(-1)

        gen = torch.Generator().manual_seed(seed)
        null_dirs = [
            F.normalize(torch.randn(dim, generator=gen), dim=0).to(self.device)
            for _ in range(n_random_directions)
        ]

        jepa_stats, jepa_calls = self._ablation_stats(
            self.jepa, input_ids, directions, probe_fn, layer_idx, null_dirs
        )
        base_stats, base_calls = self._ablation_stats(
            self.baseline, input_ids, directions, probe_fn, layer_idx, null_dirs
        )

        results = {}
        for name in directions:
            j, b = jepa_stats[name], base_stats[name]
            results[name] = {
                "jepa_ablation_effect": j["effect"],
                "baseline_ablation_effect": b["effect"],
                "jepa_effect_z": j["effect_z"],
                "baseline_effect_z": b["effect_z"],
                "jepa_effect_p": j["effect_p"],
                "baseline_effect_p": b["effect_p"],
                # A tie (equal p, which is what a pure scale difference produces)
                # is not evidence for either model.
                "jepa_more_predictable": bool(j["effect_p"] < b["effect_p"]),
                "n_hook_calls": (
                    j["n_hook_calls"] + j["n_null_calls"] + b["n_hook_calls"] + b["n_null_calls"]
                ),
                "n_random_directions": n_random_directions,
                "null_hypothesis": ABLATION_NULL,
            }
        return results

    @torch.no_grad()
    def steering_comparison(
        self,
        input_ids,
        direction,
        probe_fn,
        scales=(-2, -1, 0, 1, 2),
        layer_idx=-1,
        seed=0,
        n_permutations=1000,
        alpha=0.05,
    ):
        """Compare steering predictability between JEPA and baseline.

        Both sides use the same hooked intervention, with scales in units of
        each model's own activation norm. The verdict requires the JEPA effect to
        clear the permutation null *and* to be more extreme than the baseline's
        — a raw |r| comparison at n=5 points is a coin flip.

        Returns:
            dict with predictability scores and p-values for both models

        """
        kwargs = {
            "scales": scales,
            "layer_idx": layer_idx,
            "device": self.device,
            "seed": seed,
            "n_permutations": n_permutations,
            "alpha": alpha,
        }
        jepa_result = intervention_predictability_score(
            self.jepa, input_ids, direction, probe_fn, **kwargs
        )
        baseline_result = intervention_predictability_score(
            self.baseline, input_ids, direction, probe_fn, **kwargs
        )

        return {
            "jepa_predictability": jepa_result["predictability"],
            "jepa_monotonicity": jepa_result["monotonicity"],
            "jepa_p_value": jepa_result["p_value"],
            "jepa_significant": jepa_result["significant"],
            "jepa_n_hook_calls": jepa_result["n_hook_calls"],
            "baseline_predictability": baseline_result["predictability"],
            "baseline_monotonicity": baseline_result["monotonicity"],
            "baseline_p_value": baseline_result["p_value"],
            "baseline_significant": baseline_result["significant"],
            "baseline_n_hook_calls": baseline_result["n_hook_calls"],
            "jepa_more_predictable": bool(
                jepa_result["significant"] and jepa_result["p_value"] < baseline_result["p_value"]
            ),
            "null_hypothesis": MONOTONE_NULL,
        }
