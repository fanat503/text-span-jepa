# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Causal Scrubbing: rigorous test for mechanistic interpretability hypotheses
# From Redwood Research (Chan et al., 2022)
# Unified in Geiger et al. (2024) "Causal Abstraction"
#
# Gold standard for testing: "Does this feature causally implement
# this behavior?" Instead of ablation (zeros), resample from the
# correct distribution. If performance survives scrubbing of "irrelevant"
# inputs, the hypothesis is validated.
#
# Key idea: if we hypothesize that feature F encodes property P,
# then scrubbing F (replacing with a random input from the same
# distribution of P) should:
# - Preserve performance if F is NOT part of the mechanism for P
# - Destroy performance if F IS part of the mechanism for P
#
# This is MORE rigorous than ablation because:
# - Ablation (zeroing) confounds "F is important" with "F has nonzero mean"
# - Scrubbing controls for the statistical effect of F's distribution
#
# HONESTY CONTRACT
# ----------------
# Every failure mode here previously reported the HYPOTHESIS-SUPPORTING value:
#   - a `behavior_fn` that did not accept `h_override` raised TypeError inside
#     `except: return 0.0`, so the scrubbed behaviour was 0.0, the change ratio
#     was 1.0, and `hypothesis_valid` came out True. A broken intervention was
#     reported as a validated hypothesis.
#   - resampling shuffled the BATCH axis, which at B=1 (the normal inference
#     case) is `randperm(1) == [0]`: a literal no-op, so behaviour was trivially
#     "preserved".
#   - a failed SVD returned relevant=ones / irrelevant=zeros, i.e. nothing was
#     scrubbed, so behaviour was trivially preserved.
# Now: a broken callback, an unresamplable input, an empty scrub set and a failed
# decomposition all RAISE. The only way to get a favourable verdict is to
# actually perform the intervention and have the behaviour survive it.

import inspect

import torch

from .causal_intervention import intervention_predictability_score


# ═══════════════════════════════════════════════════════════════════
# Callback contract
# ═══════════════════════════════════════════════════════════════════


def _accepts_h_override(behavior_fn):
    """Whether `behavior_fn` can receive scrubbed representations.

    Raises rather than guessing: a callback that cannot be handed the scrubbed
    representation would make every trial measure the clean model instead.
    """
    if not callable(behavior_fn):
        raise TypeError(f"behavior_fn must be callable, got {type(behavior_fn).__name__}")
    try:
        sig = inspect.signature(behavior_fn)
    except (TypeError, ValueError) as exc:  # un-introspectable callable
        raise TypeError(
            f"cannot inspect behavior_fn {behavior_fn!r}; it must declare an "
            "`h_override` parameter so scrubbed representations can reach it"
        ) from exc
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if name == "h_override" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _require_h_override(behavior_fn):
    if not _accepts_h_override(behavior_fn):
        raise TypeError(
            "behavior_fn must accept `h_override`, i.e. it must be callable as "
            "behavior_fn(model, input_ids, h_override=None) where h_override=None "
            "means the clean forward pass and a (B, T, D) tensor means 'use these "
            "scrubbed representations instead'. Got a signature without it "
            f"({inspect.signature(behavior_fn)}). Refusing to run: without the "
            "scrubbed representation every trial would measure the clean model and "
            "the hypothesis would be validated by construction.",
        )


# ═══════════════════════════════════════════════════════════════════
# Resampling
# ═══════════════════════════════════════════════════════════════════


def resample_positions(h, generator=None, seed=None):
    """Resample every position from another position of the same sequence.

    Redwood-style resampling permutes the positions being scrubbed. The previous
    implementation shuffled the BATCH axis, which is a literal no-op at B=1 --
    the normal inference case -- because `randperm(1) == [0]`.

    Args:
        h: (B, T, D) representations to resample
        generator: torch.Generator for the permutation draw (seed is ignored if given)
        seed: alternative to `generator`

    Returns:
        (B, T, D) resampled tensor. The permutation is guaranteed non-identity,
        so for T >= 2 the result always moves every token.

    """
    if h.dim() != 3:
        raise ValueError(f"resample_positions expects (B, T, D), got shape {tuple(h.shape)}")
    B, T, D = h.shape
    if T < 2:
        raise ValueError(
            "cannot resample along the token axis with T=1: there is no other "
            "position to draw from, so the scrub would be a no-op",
        )
    if generator is None:
        generator = torch.Generator().manual_seed(0 if seed is None else seed)
    idx = torch.argsort(torch.rand(B, T, generator=generator), dim=1).to(h.device)
    # Guarantee a real permutation: swap the first two positions of any row that
    # came out as the identity, so no trial can silently scrub nothing.
    is_identity = (idx == torch.arange(T, device=idx.device).unsqueeze(0)).all(dim=1)
    if bool(is_identity.any()):
        swapped = idx.clone()
        swapped[is_identity, 0] = idx[is_identity, 1]
        swapped[is_identity, 1] = idx[is_identity, 0]
        idx = swapped
    return torch.gather(h, 1, idx.unsqueeze(-1).expand(B, T, D))


# ═══════════════════════════════════════════════════════════════════
# Scrubber
# ═══════════════════════════════════════════════════════════════════


class CausalScrubber:
    """Causal scrubbing for testing interpretability hypotheses.

    Given a hypothesis about which features matter for a behavior,
    scrub away features the hypothesis says are irrelevant and
    measure whether the behavior is preserved.

    If behavior is preserved → hypothesis is correct (those features
    really are irrelevant).
    If behavior is destroyed → hypothesis is wrong (those features
    matter more than you thought).
    """

    def __init__(self, model, device="cpu"):
        """
        Args:
            model: Text-Span JEPA model (or any model with .encoder)
            device: compute device

        """
        self.model = model
        self.device = device

    @torch.no_grad()
    def scrub_and_evaluate(
        self,
        input_ids,
        hypothesis_fn,
        behavior_fn,
        n_resamples=10,
        seed=0,
    ):
        """Scrub irrelevant features and test if behavior is preserved.

        Args:
            input_ids: (B, T) input token IDs (clean distribution)
            hypothesis_fn: callable(representations) -> (relevant_mask, irrelevant_mask)
                relevant_mask: (B, T, D) boolean, True = keep this feature
                irrelevant_mask: (B, T, D) boolean, True = scrub this feature
            behavior_fn: callable(model, input_ids, h_override=None) -> scalar.
                Called with h_override=None for the baseline (clean forward pass)
                and with the (B, T, D) scrubbed tensor for every trial. MUST use
                `h_override` when it is not None — that is how the intervention
                reaches the model.
            n_resamples: number of resampling trials
            seed: seed for the resample permutations (default 0 → reproducible)

        Returns:
            dict with scrubbed behavior metrics. `hypothesis_valid` is True when
            the behavior SURVIVED scrubbing, i.e. the scrubbed features really
            were irrelevant.

        """
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
        _require_h_override(behavior_fn)
        self.model.eval()
        input_ids = input_ids.to(self.device)

        # Baseline behavior (clean). Same callback, h_override=None → clean pass.
        baseline_behavior = self._call_behavior(behavior_fn, input_ids, None)

        h, _ = self.model.encoder(input_ids)
        relevant_mask, irrelevant_mask = hypothesis_fn(h)
        relevant_mask, irrelevant_mask = _validate_masks(h, relevant_mask, irrelevant_mask)

        generator = torch.Generator().manual_seed(seed)
        scrubbed_behaviors = []
        for _ in range(n_resamples):
            h_resampled = resample_positions(h, generator=generator)
            h_scrubbed = torch.where(
                irrelevant_mask,
                h_resampled.to(h.dtype),
                h,
            )
            if torch.equal(h_scrubbed, h):
                raise RuntimeError(
                    "the scrub produced a representation identical to the clean one; "
                    "nothing was resampled, so behaviour is trivially preserved and "
                    "the hypothesis cannot be evaluated",
                )
            scrubbed_behaviors.append(self._call_behavior(behavior_fn, input_ids, h_scrubbed))

        mean_scrubbed = sum(scrubbed_behaviors) / len(scrubbed_behaviors)
        relative_change = abs(mean_scrubbed - baseline_behavior) / max(
            abs(baseline_behavior),
            1e-10,
        )
        # Preservation = how much of the behaviour survived. It is 1 - relative
        # change, so 1.0 means "scrubbing changed nothing".
        preservation = 1.0 - relative_change
        # Higher preservation = hypothesis correct (scrubbed features are irrelevant)
        # Lower preservation = hypothesis wrong (scrubbed features matter)

        return {
            "baseline_behavior": baseline_behavior,
            "scrubbed_behavior_mean": mean_scrubbed,
            "scrubbed_behavior_std": _std(scrubbed_behaviors),
            "scrubbed_behaviors": scrubbed_behaviors,
            "relative_behavior_change": relative_change,
            "behavior_preservation_ratio": preservation,
            "hypothesis_valid": preservation > 0.8,
            "n_resamples": n_resamples,
            "n_scrubbed_features": int(irrelevant_mask.sum().item()),
        }

    def _call_behavior(self, behavior_fn, input_ids, h_override):
        """Single call site for `behavior_fn`. Never swallows an exception."""
        return float(behavior_fn(self.model, input_ids, h_override=h_override))

    def _behavior_from_representations(self, h_scrubbed, behavior_fn, input_ids):
        """Evaluate behavior using scrubbed representations.

        For probing tasks: pool representations and pass through probe.
        For generation tasks: not applicable (encoder-only).
        """
        return self._call_behavior(behavior_fn, input_ids, h_scrubbed)

    @torch.no_grad()
    def compare_scrubbing(
        self,
        jepa_model,
        baseline_model,
        input_ids,
        hypothesis_fn,
        behavior_fn,
        n_resamples=10,
        seed=0,
    ):
        """Compare causal scrubbing results between JEPA and baseline.

        THE KEY COMPARISON: if JEPA's behavior is more preserved under
        scrubbing of irrelevant features, it means JEPA's features have
        cleaner causal structure (hypothesis correctly identifies what
        matters).

        Args:
            jepa_model: JEPA model
            baseline_model: baseline model (MLM/data2vec)
            input_ids: (B, T) input token IDs
            hypothesis_fn: hypothesis about feature relevance
            behavior_fn: behavior to test
            n_resamples: number of resampling trials
            seed: seed for the resample permutations (same seed for both models)

        Returns:
            dict with comparison results

        """
        jepa_scrubber = CausalScrubber(jepa_model, self.device)
        baseline_scrubber = CausalScrubber(baseline_model, self.device)

        jepa_result = jepa_scrubber.scrub_and_evaluate(
            input_ids,
            hypothesis_fn,
            behavior_fn,
            n_resamples,
            seed,
        )
        baseline_result = baseline_scrubber.scrub_and_evaluate(
            input_ids,
            hypothesis_fn,
            behavior_fn,
            n_resamples,
            seed,
        )

        return {
            "jepa_preservation": jepa_result["behavior_preservation_ratio"],
            "baseline_preservation": baseline_result["behavior_preservation_ratio"],
            "jepa_hypothesis_valid": jepa_result["hypothesis_valid"],
            "baseline_hypothesis_valid": baseline_result["hypothesis_valid"],
            # Cleaner causal structure == behaviour survives scrubbing BETTER.
            "jepa_cleaner_causal_structure": bool(
                jepa_result["behavior_preservation_ratio"]
                > baseline_result["behavior_preservation_ratio"]
            ),
        }


def _validate_masks(h, relevant_mask, irrelevant_mask):
    """Reject hypotheses whose masks cannot define a scrub."""
    if relevant_mask.shape != h.shape or irrelevant_mask.shape != h.shape:
        raise ValueError(
            f"hypothesis masks must match the representation shape {tuple(h.shape)}, got "
            f"{tuple(relevant_mask.shape)} and {tuple(irrelevant_mask.shape)}"
        )
    relevant_mask = relevant_mask.bool()
    irrelevant_mask = irrelevant_mask.bool()
    if bool((relevant_mask & irrelevant_mask).any()):
        raise ValueError("relevant_mask and irrelevant_mask overlap; a feature cannot be both")
    uncovered = ~(relevant_mask | irrelevant_mask)
    if bool(uncovered.any()):
        raise ValueError(
            "hypothesis masks do not cover the representation space "
            f"({int(uncovered.sum().item())} features are neither relevant nor irrelevant)"
        )
    if not bool(irrelevant_mask.any()):
        raise ValueError(
            "hypothesis marks no feature as irrelevant: the scrub would be a no-op, "
            "so behaviour would be trivially preserved and the hypothesis would be "
            "validated without an intervention"
        )
    return relevant_mask, irrelevant_mask


class FeatureHypothesis:
    """Builders for common interpretability hypotheses.

    A hypothesis specifies which features are RELEVANT and which are
    IRRELEVANT for a given behavior. The scrubber then tests this.
    """

    @staticmethod
    def svd_directions_hypothesis(representations, n_relevant_dims=50):
        """Hypothesis: top-SVD directions are relevant, rest is irrelevant.

        Tests whether the principal components of representations
        carry the causal structure (vs noise in minor components).

        Raises (never falls back to a hypothesis that scrubs nothing):
        a decomposition that fails leaves us with no hypothesis to test, and
        silently returning relevant=ones/irrelevant=zeros would report "nothing
        was scrubbed, so behaviour is preserved" as a validated hypothesis.
        """
        if representations.dim() != 3:
            raise ValueError(
                f"svd_directions_hypothesis expects (B, T, D), got shape {tuple(representations.shape)}"
            )
        B, T, D = representations.shape
        flat = representations.reshape(B * T, D)
        max_dims = min(B * T, D)
        if not 1 <= n_relevant_dims <= max_dims:
            raise ValueError(
                f"n_relevant_dims must be in [1, {max_dims}] for a ({B}, {T}, {D}) input, "
                f"got {n_relevant_dims}"
            )

        # SVD — a non-finite input or a failed LAPACK convergence propagates.
        _U, _S, Vh = torch.linalg.svd(flat.to(torch.float32), full_matrices=False)
        V = Vh.T  # Right singular vectors

        # Top-n_relevant_dims components are "relevant"
        relevant_dirs = V[:, :n_relevant_dims]  # (D, k)

        # Project: relevant component
        proj_relevant = flat @ relevant_dirs @ relevant_dirs.T
        proj_irrelevant = flat - proj_relevant

        # For simplicity: mask based on projection magnitude
        proj_r = proj_relevant.reshape(B, T, D)
        proj_i = proj_irrelevant.reshape(B, T, D)

        relevant_mask = proj_r.abs() > proj_i.abs()
        irrelevant_mask = ~relevant_mask

        return relevant_mask, irrelevant_mask

    @staticmethod
    def position_based_hypothesis(representations, keep_fraction=0.5):
        """Hypothesis: early sequence positions are relevant, late are irrelevant.

        Tests whether the causal structure is concentrated in the
        early tokens (which carry more context in autoregressive
        models, but JEPA should use the full sequence).
        """
        B, T, D = representations.shape
        cutoff = int(T * keep_fraction)

        relevant_mask = torch.zeros(B, T, D, dtype=torch.bool)
        relevant_mask[:, :cutoff, :] = True

        irrelevant_mask = ~relevant_mask

        return relevant_mask, irrelevant_mask

    @staticmethod
    def random_hypothesis(representations, relevant_fraction=0.5, seed=0):
        """Null hypothesis: random feature subset is relevant.

        Should FAIL scrubbing (behavior destroyed) if features
        have genuine causal structure. Used as control.
        """
        B, T, D = representations.shape
        generator = torch.Generator().manual_seed(seed)
        relevant_mask = torch.rand(B, T, D, generator=generator) < relevant_fraction
        irrelevant_mask = ~relevant_mask
        return relevant_mask, irrelevant_mask


class InterventionPredictabilityScorer:
    """Score how predictable interventions are on JEPA vs baseline.

    Complements causal scrubbing: scrubbing tests IF features matter,
    predictability tests HOW CLEANLY they matter.

    If steering along JEPA feature A causes a monotonic change in
    probe output, but steering along MLM feature A causes noisy
    changes → JEPA has cleaner causal structure.

    This used to be an independent copy of the same bug — it ran the model
    once and added `scale * direction` to the pooled output vector, so no
    layer ever saw the modification. It now delegates to
    `intervention_predictability_score`, which installs a real forward hook,
    and inherits its permutation p-value.
    """

    @staticmethod
    @torch.no_grad()
    def compute_predictability(
        model,
        input_ids,
        direction,
        probe_fn,
        scales=(-3, -2, -1, 0, 1, 2, 3),
        device="cpu",
        layer_idx=-1,
        seed=0,
        n_permutations=1000,
        alpha=0.05,
    ):
        """Compute predictability score for a direction.

        Args:
            model: encoder model
            input_ids: (B, T) input token IDs
            direction: (D,) steering direction
            probe_fn: callable(pooled_repr) -> scalar
            scales: steering scales, in units of the clean activation norm
            device: compute device
            layer_idx: encoder block to steer at (-1 = last)
            seed: seed for the permutation null
            n_permutations: permutation draws
            alpha: significance threshold

        Returns:
            dict with predictability metrics, including a permutation p-value

        """
        return intervention_predictability_score(
            model,
            input_ids,
            direction,
            probe_fn,
            scales=scales,
            layer_idx=layer_idx,
            device=device,
            seed=seed,
            n_permutations=n_permutations,
            alpha=alpha,
        )


def _std(values):
    """Standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var**0.5
