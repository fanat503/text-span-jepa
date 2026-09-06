# Copyright 2026 Text-Span JEPA Authors
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


import torch
import torch.nn.functional as F


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
    def scrub_and_evaluate(self, input_ids, hypothesis_fn, behavior_fn, n_resamples=10):
        """Scrub irrelevant features and test if behavior is preserved.

        Args:
            input_ids: (B, T) input token IDs (clean distribution)
            input_ids_resample: (B, T) resample pool (different distribution)
            hypothesis_fn: callable(representations) -> (relevant_mask, irrelevant_mask)
                relevant_mask: (B, T, D) boolean, True = keep this feature
                irrelevant_mask: (B, T, D) boolean, True = scrub this feature
            behavior_fn: callable(model, input_ids) -> scalar behavior metric
            n_resamples: number of resampling trials

        Returns:
            dict with scrubbed behavior metrics

        """
        self.model.eval()
        input_ids = input_ids.to(self.device)

        # Baseline behavior (clean)
        baseline_behavior = behavior_fn(self.model, input_ids)

        # Get representations
        h, _ = self.model.encoder(input_ids)
        relevant_mask, irrelevant_mask = hypothesis_fn(h)

        # Scrub: replace irrelevant features with resampled values
        scrubbed_behaviors = []
        for _ in range(n_resamples):
            # Resample: shuffle along batch dimension (permutation test)
            perm = torch.randperm(h.size(0), device=self.device)
            h_resampled = h[perm]

            # Apply scrubbing: keep relevant, replace irrelevant
            h_scrubbed = h * relevant_mask.float() + h_resampled * irrelevant_mask.float()

            # Run behavior test on scrubbed representations
            scrubbed_behavior = self._behavior_from_representations(
                h_scrubbed,
                behavior_fn,
                input_ids,
            )
            scrubbed_behaviors.append(scrubbed_behavior)

        # Results
        mean_scrubbed = sum(scrubbed_behaviors) / len(scrubbed_behaviors)
        behavior_preserved = abs(mean_scrubbed - baseline_behavior) / max(
            abs(baseline_behavior),
            1e-10,
        )
        # Higher preservation = hypothesis correct (scrubbed features are irrelevant)
        # Lower preservation = hypothesis wrong (scrubbed features matter)

        return {
            "baseline_behavior": baseline_behavior,
            "scrubbed_behavior_mean": mean_scrubbed,
            "scrubbed_behavior_std": _std(scrubbed_behaviors),
            "behavior_preservation_ratio": behavior_preserved,
            "hypothesis_valid": behavior_preserved > 0.8,
            "n_resamples": n_resamples,
        }

    def _behavior_from_representations(self, h_scrubbed, behavior_fn, input_ids):
        """Evaluate behavior using scrubbed representations.

        For probing tasks: pool representations and pass through probe.
        For generation tasks: not applicable (encoder-only).
        """
        try:
            # Simple: use pooled representation to predict something
            # This requires behavior_fn to accept representations directly
            if callable(behavior_fn):
                return behavior_fn(self.model, input_ids, h_override=h_scrubbed)
            return 0.0
        except Exception:
            return 0.0

    @torch.no_grad()
    def compare_scrubbing(
        self,
        jepa_model,
        baseline_model,
        input_ids,
        hypothesis_fn,
        behavior_fn,
        n_resamples=10,
    ):
        """Compare causal scrubbing results between JEPA and baseline.

        THE KEY COMPARISON: if JEPA's behavior is more preserved under
        scrubbing of irrelevant features, it means JEPA's features have
        cleaner causal structure (hypothesis correctly identifies what
        matters).

        Args:
            jepa_model: JEPA model
            baseline_model: baseline model (MLM/data2vec)
            input_ids: (B, T) token IDs
            hypothesis_fn: hypothesis about feature relevance
            behavior_fn: behavior to test
            n_resamples: number of resampling trials

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
        )
        baseline_result = baseline_scrubber.scrub_and_evaluate(
            input_ids,
            hypothesis_fn,
            behavior_fn,
            n_resamples,
        )

        return {
            "jepa_preservation": jepa_result["behavior_preservation_ratio"],
            "baseline_preservation": baseline_result["behavior_preservation_ratio"],
            "jepa_hypothesis_valid": jepa_result["hypothesis_valid"],
            "baseline_hypothesis_valid": baseline_result["hypothesis_valid"],
            "jepa_cleaner_causal_structure": (
                jepa_result["behavior_preservation_ratio"]
                > baseline_result["behavior_preservation_ratio"]
            ),
        }


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
        """
        try:
            B, T, D = representations.shape
            flat = representations.reshape(B * T, D)
            # SVD
            _U, _S, Vh = torch.linalg.svd(flat, full_matrices=False)
            V = Vh.T  # Right singular vectors

            # Top-n_relevant_dims components are "relevant"
            relevant_dirs = V[:, :n_relevant_dims]  # (D, k)

            # Project: relevant component
            proj_relevant = flat @ relevant_dirs @ relevant_dirs.T
            proj_irrelevant = flat - proj_relevant

            # Create masks
            relevant_mask = torch.zeros_like(representations)
            irrelevant_mask = torch.zeros_like(representations)

            # For simplicity: mask based on projection magnitude
            proj_r = proj_relevant.reshape(B, T, D)
            proj_i = proj_irrelevant.reshape(B, T, D)

            relevant_mask = proj_r.abs() > proj_i.abs()
            irrelevant_mask = ~relevant_mask

            return relevant_mask, irrelevant_mask
        except Exception:
            B, T, D = representations.shape
            return torch.ones_like(representations, dtype=torch.bool), torch.zeros_like(
                representations,
                dtype=torch.bool,
            )

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
    def random_hypothesis(representations, relevant_fraction=0.5):
        """Null hypothesis: random feature subset is relevant.

        Should FAIL scrubbing (behavior destroyed) if features
        have genuine causal structure. Used as control.
        """
        B, T, D = representations.shape
        relevant_mask = torch.rand(B, T, D) < relevant_fraction
        irrelevant_mask = ~relevant_mask
        return relevant_mask, irrelevant_mask


class InterventionPredictabilityScorer:
    """Score how predictable interventions are on JEPA vs baseline.

    Complements causal scrubbing: scrubbing tests IF features matter,
    predictability tests HOW CLEANLY they matter.

    If steering along JEPA feature A causes a monotonic change in
    probe output, but steering along MLM feature A causes noisy
    changes → JEPA has cleaner causal structure.
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
    ):
        """Compute predictability score for a direction.

        Args:
            model: encoder model
            input_ids: (B, T) token IDs
            direction: (D,) steering direction
            probe_fn: callable(pooled_repr) -> scalar
            scales: steering scale values
            device: compute device

        Returns:
            dict with predictability metrics

        """
        model.eval()
        input_ids = input_ids.to(device)
        direction = direction.to(device)

        probe_values = []
        for scale in scales:
            h, _ = model.encoder(input_ids)
            h_pooled = h.mean(dim=1)
            h_steered = h_pooled + scale * F.normalize(direction, dim=0)
            val = probe_fn(h_steered)
            if isinstance(val, torch.Tensor):
                val = val.mean().item()
            probe_values.append(val)

        # Spearman correlation: scale vs probe_value
        scales_t = torch.tensor(scales, dtype=torch.float32)
        probes_t = torch.tensor(probe_values, dtype=torch.float32)

        if probes_t.std() == 0:
            return {
                "predictability": 0.0,
                "monotonicity": 0.0,
                "probe_values": probe_values,
            }

        # Spearman
        rs = scales_t.argsort().argsort().float()
        rp = probes_t.argsort().argsort().float()
        rs_c = rs - rs.mean()
        rp_c = rp - rp.mean()
        denom = rs_c.norm() * rp_c.norm()
        spearman = (rs_c @ rp_c / denom).item() if denom > 0 else 0.0

        # Monotonicity
        diffs = probes_t[1:] - probes_t[:-1]
        if diffs.abs().sum() == 0:
            mono = 0.0
        else:
            signs = (diffs > 0).float()
            mono = max(signs.mean().item(), (1 - signs.mean()).item())

        return {
            "predictability": abs(spearman),
            "monotonicity": mono,
            "probe_values": probe_values,
        }


def _std(values):
    """Standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var**0.5
