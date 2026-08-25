# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Feature Composition Score: are JEPA features more compositional?
#
# If JEPA features are more compositional (as the hypothesis predicts),
# then SAE feature arithmetic should work:
#   feature_A (e.g., "plural") + feature_B (e.g., "noun")
#   ≈ representation of "plural noun"
#
# This directly tests the "compositional features" part of the hypothesis.
# Inspired by word2vec analogies (Mikolov et al., 2013) applied to
# SAE features (Bricken et al., 2023).


import torch
import torch.nn.functional as F


class FeatureCompositionScore:
    """Score how compositional SAE features are.

    Method:
    1. Train SAE on model representations
    2. Find feature pairs (A, B) that correspond to known linguistic properties
    3. Test whether A + B (in feature space) maps to the combined property
       (in representation space)
    4. Score = correlation between "combined feature" and "combined property"

    If JEPA features score higher → JEPA is more compositional → evidence
    for the inductive bias hypothesis.
    """

    def __init__(self, sae_model, encoder_model, device="cpu"):
        """
        Args:
            sae_model: trained SparseAutoencoder
            encoder_model: encoder model (for generating representations)
            device: compute device
        """
        self.sae = sae_model
        self.encoder = encoder_model
        self.device = device

    @torch.no_grad()
    def feature_arithmetic_test(
        self, input_ids_a, input_ids_b, input_ids_ab, sae_feature_a_idx, sae_feature_b_idx
    ):
        """Test feature arithmetic: A + B ≈ AB.

        Args:
            input_ids_a: (1, T) tokens exhibiting property A (e.g., singular noun)
            input_ids_b: (1, T) tokens exhibiting property B (e.g., plural verb)
            input_ids_ab: (1, T) tokens exhibiting both A and B (e.g., plural noun)
            sae_feature_a_idx: index of SAE feature that activates for A
            sae_feature_b_idx: index of SAE feature that activates for B

        Returns:
            dict with arithmetic test results
        """
        self.sae.eval()
        self.encoder.eval()

        # Get representations
        h_a, _ = self.encoder(input_ids_a.to(self.device))
        h_b, _ = self.encoder(input_ids_b.to(self.device))
        h_ab, _ = self.encoder(input_ids_ab.to(self.device))

        # Pool
        h_a_pooled = h_a.mean(dim=1)
        h_b_pooled = h_b.mean(dim=1)
        h_ab_pooled = h_ab.mean(dim=1)

        # Encode to SAE feature space
        z_a, _, _ = self.sae.encode(h_a_pooled)
        z_b, _, _ = self.sae.encode(h_b_pooled)
        z_ab, _, _ = self.sae.encode(h_ab_pooled)

        # Feature arithmetic: z_a + z_b should approximate z_ab
        z_combined = z_a + z_b

        # Cosine similarity between combined and actual AB
        cos_sim = F.cosine_similarity(z_combined.unsqueeze(0), z_ab.unsqueeze(0)).item()

        # Also test: decode z_combined back to representation space
        h_combined = self.sae.decode(z_combined)
        cos_sim_repr = F.cosine_similarity(h_combined.unsqueeze(0), h_ab_pooled.unsqueeze(0)).item()

        # Baseline: just z_a (or just z_b) — should be lower
        cos_a_only = F.cosine_similarity(z_a.unsqueeze(0), z_ab.unsqueeze(0)).item()
        cos_b_only = F.cosine_similarity(z_b.unsqueeze(0), z_ab.unsqueeze(0)).item()

        # Composition advantage: how much better is A+B than just A or just B?
        best_baseline = max(cos_a_only, cos_b_only)
        composition_advantage = cos_sim - best_baseline

        return {
            "cosine_feature_space": cos_sim,
            "cosine_repr_space": cos_sim_repr,
            "cosine_a_only": cos_a_only,
            "cosine_b_only": cos_b_only,
            "composition_advantage": composition_advantage,
            "composition_works": composition_advantage > 0,
        }

    @torch.no_grad()
    def systematic_composition_test(self, test_pairs, n_feature_search=10):
        """Systematic test of feature composition across many property pairs.

        For each pair of linguistic properties (A, B), find the SAE features
        that best separate A vs not-A, and B vs not-B, then test A + B ≈ AB.

        Args:
            test_pairs: list of dicts with:
                'a_reps': (N, D) representations with property A
                'b_reps': (N, D) representations with property B
                'ab_reps': (N, D) representations with both A and B
                'neither_reps': (N, D) representations with neither
            n_feature_search: how many top SAE features to consider per property

        Returns:
            dict with aggregate composition score
        """
        results = []
        placebos = []
        for pair_idx, pair in enumerate(test_pairs):
            a_reps = pair["a_reps"].to(self.device)
            b_reps = pair["b_reps"].to(self.device)
            ab_reps = pair["ab_reps"].to(self.device)
            neither_reps = pair["neither_reps"].to(self.device)

            # Find candidate SAE features for A and B (honors n_feature_search)
            cand_a = self._find_discriminating_feature(a_reps, neither_reps, n_feature_search)
            cand_b = self._find_discriminating_feature(b_reps, neither_reps, n_feature_search)

            z_a = self.sae.encode(a_reps.mean(dim=0, keepdim=True))[0]
            z_b = self.sae.encode(b_reps.mean(dim=0, keepdim=True))[0]
            z_ab = self.sae.encode(ab_reps.mean(dim=0, keepdim=True))[0]

            # Placebo control (fleet R3): permuting AB rows destroys sample
            # identity while keeping the marginal SAE code distribution; the
            # same selection run on shuffled AB calibrates the tautology of
            # comparing against data that literally contains A and B.
            gen = torch.Generator().manual_seed(1234 + pair_idx)
            perm = torch.randperm(ab_reps.size(0), generator=gen).to(ab_reps.device)
            z_ab_placebo = self.sae.encode(ab_reps[perm].mean(dim=0, keepdim=True))[0]

            best_cos = -2.0
            best_placebo = 0.0
            for b_idx in cand_b:
                z_combined = torch.zeros_like(z_a)
                z_combined[:, cand_a[0]] = z_a[:, cand_a[0]]
                z_combined[:, b_idx] = z_b[:, b_idx]
                cos_real = F.cosine_similarity(z_combined, z_ab).item()
                if cos_real > best_cos:
                    best_cos = cos_real
                    best_placebo = F.cosine_similarity(z_combined, z_ab_placebo).item()
            results.append(best_cos)
            placebos.append(best_placebo)

        if not results:
            return {"mean_composition_score": 0.0, "n_pairs": 0}

        n = len(results)
        return {
            "mean_composition_score": sum(results) / n,
            "n_pairs": n,
            "max_score": max(results),
            "min_score": min(results),
            "fraction_positive": sum(1 for r in results if r > 0) / n,
            # Placebo-calibrated reporting: the raw score is selection-biased
            # upward by best-of-k search; advantage over matched placebo is
            # the interpretable quantity.
            "mean_placebo": sum(placebos) / n,
            "mean_advantage": sum(r - p for r, p in zip(results, placebos)) / n,
            "fraction_above_placebo": sum(1 for r, p in zip(results, placebos) if r > p) / n,
            "n_search_used": int(n_feature_search),
        }

    def _find_discriminating_feature(self, pos_reps, neg_reps, n_search=10):
        """Find the SAE feature that best discriminates positive from negative.

        Returns:
            int: feature index with highest discrimination score
        """
        z_pos, _, _ = self.sae.encode(pos_reps)
        z_neg, _, _ = self.sae.encode(neg_reps)

        # Mean activation difference per feature
        pos_mean = z_pos.mean(dim=0)
        neg_mean = z_neg.mean(dim=0)
        diff = (pos_mean - neg_mean).abs()

        k = min(n_search, diff.numel())
        # Return ALL candidates: the caller runs its selection over this list
        # (top-1-only silently ignored n_feature_search — fleet R3 finding).
        return diff.topk(k).indices.tolist()

    @staticmethod
    def compare_models(
        jepa_sae, baseline_sae, jepa_encoder, baseline_encoder, test_pairs, device="cpu"
    ):
        """Compare composition scores between JEPA and baseline.

        Args:
            jepa_sae: SAE trained on JEPA representations
            baseline_sae: SAE trained on baseline representations
            jepa_encoder: JEPA encoder
            baseline_encoder: baseline encoder
            test_pairs: list of test pair dicts
            device: compute device

        Returns:
            dict with comparison
        """
        jepa_scorer = FeatureCompositionScore(jepa_sae, jepa_encoder, device)
        baseline_scorer = FeatureCompositionScore(baseline_sae, baseline_encoder, device)

        jepa_result = jepa_scorer.systematic_composition_test(test_pairs)
        baseline_result = baseline_scorer.systematic_composition_test(test_pairs)

        return {
            "jepa_composition_score": jepa_result["mean_composition_score"],
            "baseline_composition_score": baseline_result["mean_composition_score"],
            "jepa_more_compositional": (
                jepa_result["mean_composition_score"] > baseline_result["mean_composition_score"]
            ),
            "jepa_details": jepa_result,
            "baseline_details": baseline_result,
        }


class FeatureInterferenceScore:
    """Measure interference between SAE features.

    If features are truly independent (disentangled), activating
    feature A should NOT affect feature B's activations.

    Interference = |mean(z_B | activate A) - mean(z_B | baseline)|
    Low interference = features are independent = more interpretable

    JEPA hypothesis: JEPA features have LOWER interference than MLM.
    """

    @staticmethod
    @torch.no_grad()
    def compute(sae, representations, n_features=50, n_top=100):
        """Compute feature interference score.

        Args:
            sae: trained SparseAutoencoder
            representations: (N, D)
            n_features: number of features to test
            n_top: number of top-activating samples per feature

        Returns:
            dict with interference metrics
        """
        try:
            z, _, _ = sae.encode(representations)
            N, M = z.shape  # M = latent dim

            n_test = min(n_features, M)
            feature_idx = torch.randperm(M)[:n_test]

            interference_scores = []

            for fi in feature_idx:
                # Find top-activating samples for feature fi
                act_vals = z[:, fi]
                _, top_idx = act_vals.topk(min(n_top, N))

                # How much do other features change when fi is active?
                z_active = z[top_idx]
                z_baseline = z  # Full dataset

                # Mean activation of other features when fi is active vs baseline
                other_idx = [j for j in range(M) if j != fi]
                if not other_idx:
                    continue
                other_idx_t = torch.tensor(other_idx)

                active_other_mean = z_active[:, other_idx_t].mean(dim=0)
                baseline_other_mean = z_baseline[:, other_idx_t].mean(dim=0)

                # Interference: how much other features shift
                interference = (active_other_mean - baseline_other_mean).abs().mean().item()
                interference_scores.append(interference)

            if not interference_scores:
                return {"mean_interference": 0.0, "max_interference": 0.0}

            return {
                "mean_interference": sum(interference_scores) / len(interference_scores),
                "max_interference": max(interference_scores),
                "min_interference": min(interference_scores),
                "n_features_tested": len(interference_scores),
            }
        except Exception:
            return {
                "mean_interference": float("inf"),
                "max_interference": float("inf"),
                "min_interference": 0.0,
                "n_features_tested": 0,
            }
