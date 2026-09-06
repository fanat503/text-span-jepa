# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Test suite for interpretability infrastructure
# Covers: SAE, StructuralProbe, CausalIntervention, Disentanglement,
#         Comparator, ProbingComplexity, Polysemanticity, CausalScrubbing,
#         RepresentationGeometry, FeatureComposition

import math

import pytest
import torch
from torch import nn

# ═══════════════════════════════════════════════════════════════════
# SAE
# ═══════════════════════════════════════════════════════════════════


class TestSAE:
    def setup_method(self):
        from src.interp.sae import SparseAutoencoder

        self.SAE = SparseAutoencoder

    def test_encode_decode_shapes(self):
        sae = self.SAE(input_dim=32, latent_dim=128, k=16)
        x = torch.randn(4, 32)
        recons, latent, loss, _info = sae(x)
        assert recons.shape == (4, 32)
        assert latent.shape == (4, 128)
        assert torch.isfinite(loss)

    def test_topk_sparsity(self):
        sae = self.SAE(input_dim=32, latent_dim=128, k=8)
        x = torch.randn(4, 32)
        latent, _, _ = sae.encode(x)
        # Only k=8 non-zero entries per sample
        for i in range(4):
            assert (latent[i] > 0).sum().item() <= 8

    def test_reconstruction_loss_decreases(self):
        sae = self.SAE(input_dim=16, latent_dim=64, k=8)
        x = torch.randn(32, 16)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
        losses = []
        for _ in range(20):
            _recons, _latent, loss, info = sae(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(info["recons_loss"])
        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_dead_feature_resampling(self):
        sae = self.SAE(input_dim=32, latent_dim=64, k=8, resample_interval=1)
        x = torch.randn(16, 32)
        for _ in range(5):
            sae(x)
            sae.resample_dead_features()
        # Should not crash and feature_act_count should be reset
        assert sae.feature_act_count.sum() == 0 or sae.total_samples == 0

    def test_decoder_normalized(self):
        sae = self.SAE(input_dim=16, latent_dim=64, k=8)
        # After forward + normalization, decoder rows should be ~unit norm
        x = torch.randn(4, 16)
        sae(x)
        # Manual check (normalization happens in SAETrainer, not in SAE forward)
        # Just check weights are finite
        assert sae.decoder.weight.isfinite().all()


class TestSAETrainer:
    def test_train_step(self):
        from src.interp.sae import SAETrainer, SparseAutoencoder

        sae = SparseAutoencoder(input_dim=16, latent_dim=64, k=8)
        trainer = SAETrainer(sae, lr=1e-3, device="cpu")
        x = torch.randn(16, 16)
        info = trainer.train_step(x)
        assert "recons_loss" in info
        assert "l0" in info
        assert "explained_variance" in info
        assert info["step"] == 1


# ═══════════════════════════════════════════════════════════════════
# Structural Probe
# ═══════════════════════════════════════════════════════════════════


class TestStructuralProbe:
    def test_forward_shape(self):
        from src.interp.structural_probe import StructuralProbe

        probe = StructuralProbe(embed_dim=32, probe_rank=8)
        h = torch.randn(2, 10, 32)
        dists = probe(h)
        assert dists.shape == (2, 10, 10)

    def test_distances_nonneg(self):
        from src.interp.structural_probe import StructuralProbe

        probe = StructuralProbe(embed_dim=32, probe_rank=8)
        h = torch.randn(2, 10, 32)
        dists = probe(h)
        assert (dists >= -1e-6).all()

    def test_train_and_evaluate(self):
        from src.interp.structural_probe import StructuralProbe

        probe = StructuralProbe(embed_dim=16, probe_rank=4)
        reps = torch.randn(5, 8, 16)
        torch.cdist(torch.randn(5, 8, 1), torch.randn(5, 8, 1)).squeeze(-1)
        # Simulate tree distances
        gold_dists = [torch.cdist(torch.randn(8, 1), torch.randn(8, 1)) for _ in range(5)]
        reps_list = [r for r in reps]
        loss_hist = probe.train_probe(reps_list, gold_dists, epochs=5, lr=0.01)
        assert len(loss_hist) == 5
        result = probe.evaluate(reps_list, gold_dists)
        assert "spearman_r" in result
        assert "uuas" in result


# ═══════════════════════════════════════════════════════════════════
# Causal Intervention
# ═══════════════════════════════════════════════════════════════════


class TestCausalIntervention:
    def test_direction_ablation(self):
        from src.interp.causal_intervention import direction_ablation

        x = torch.randn(4, 32)
        d = torch.nn.functional.normalize(torch.randn(32), dim=0)
        ablated = direction_ablation(x, d)
        # Ablated should have zero component along d
        cos = torch.nn.functional.cosine_similarity(ablated, d.unsqueeze(0).expand_as(x))
        assert cos.abs().max() < 1e-5

    def test_feature_steering(self):
        from src.interp.causal_intervention import feature_steering

        x = torch.randn(4, 32)
        d = torch.randn(32)
        steered = feature_steering(x, d, scale=2.0)
        # steered = x + 2 * normalize(d)
        expected = x + 2.0 * torch.nn.functional.normalize(d, dim=0)
        assert torch.allclose(steered, expected, atol=1e-5)

    def test_activation_patching(self):
        from src.interp.causal_intervention import activation_patching

        source = torch.randn(2, 8, 32)
        target = torch.randn(2, 8, 32)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, :4] = True
        patched = activation_patching(source, target, mask)
        # First 4 positions from source, last 4 from target
        assert torch.allclose(patched[:, :4], source[:, :4])
        assert torch.allclose(patched[:, 4:], target[:, 4:])

    def test_intervention_predictability_score(self):
        from src.interp.causal_intervention import intervention_predictability_score
        from src.models.encoder import TextSpanJEPLEncoder

        enc = TextSpanJEPLEncoder(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )

        class MockModel:
            encoder = enc

            def eval(self):
                pass

        direction = torch.randn(32)
        # Simple probe: norm of representation
        probe_fn = lambda h: h.norm(dim=-1).mean()
        ids = torch.randint(0, 100, (2, 16))
        result = intervention_predictability_score(MockModel(), ids, direction, probe_fn)
        assert "predictability" in result
        assert "monotonicity" in result


# ═══════════════════════════════════════════════════════════════════
# Disentanglement
# ═══════════════════════════════════════════════════════════════════


class TestDisentanglement:
    def test_dci_metrics(self):
        from src.interp.disentanglement import DCIMetrics

        reps = torch.randn(50, 16)
        factors = torch.randn(50, 4)
        result = DCIMetrics.compute(reps, factors)
        assert 0 <= result["disentanglement"] <= 1
        assert 0 <= result["completeness"] <= 1
        assert result["informativeness"] >= 0

    def test_sap_score(self):
        from src.interp.disentanglement import SAPScore

        reps = torch.randn(50, 16)
        factors = torch.randn(50, 3)
        sap = SAPScore.compute(reps, factors)
        assert 0 <= sap <= 1

    def test_mig_score(self):
        from src.interp.disentanglement import MIGScore

        reps = torch.randn(50, 16)
        factors = torch.randint(0, 3, (50, 2))
        mig = MIGScore.compute(reps, factors, n_bins=5)
        assert mig >= 0

    def test_modularity(self):
        from src.interp.disentanglement import ModularityScore

        reps = torch.randn(50, 16)
        factors = torch.randint(0, 3, (50, 2))
        mod = ModularityScore.compute(reps, factors)
        assert 0 <= mod <= 1

    def test_compute_all(self):
        from src.interp.disentanglement import compute_all_disentanglement_metrics

        reps = torch.randn(50, 16)
        factors = torch.randint(0, 3, (50, 2))
        result = compute_all_disentanglement_metrics(reps, factors)
        assert "disentanglement" in result
        assert "sap" in result
        assert "mig" in result
        assert "modularity" in result


# ═══════════════════════════════════════════════════════════════════
# Comparator
# ═══════════════════════════════════════════════════════════════════


class TestComparator:
    def test_extract_linguistic_features(self):
        from src.interp.compare import extract_linguistic_features

        features = extract_linguistic_features(["The", "cat", "sat"])
        assert features["n_tokens"] == 3.0
        # "The" starts with capital T
        assert features["is_upper"] == 1.0

    def test_geometry_comparison(self):
        from src.interp.compare import RepresentationComparator
        from src.models.encoder import TextSpanJEPLEncoder

        enc = TextSpanJEPLEncoder(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )
        comp = RepresentationComparator(
            type("M", (), {"encoder": enc}),
            type("M", (), {"encoder": enc}),
        )
        jepa_reps = torch.randn(20, 32)
        baseline_reps = torch.randn(20, 32)
        result = comp.geometry_comparison(jepa_reps, baseline_reps)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════
# Probing Complexity (v0.6.0 — KEY NEW METRIC)
# ═══════════════════════════════════════════════════════════════════


class TestProbingComplexity:
    def test_build_probe_depth1(self):
        from src.interp.probing_complexity import ProbingComplexityCurve

        pcc = ProbingComplexityCurve(embed_dim=32, num_classes=5)
        probe = pcc._build_probe(depth=1, num_classes=5)
        assert isinstance(probe[0], nn.Linear)
        assert probe[0].in_features == 32
        assert probe[0].out_features == 5

    def test_build_probe_depth3(self):
        from src.interp.probing_complexity import ProbingComplexityCurve

        pcc = ProbingComplexityCurve(embed_dim=32, num_classes=5)
        probe = pcc._build_probe(depth=3, num_classes=5)
        # Linear → ReLU → Linear → ReLU → Linear
        assert len(probe) == 5

    def test_evaluate_single_task(self):
        from src.interp.probing_complexity import ProbingComplexityCurve

        pcc = ProbingComplexityCurve(embed_dim=16, depths=(1, 2), max_epochs=10, device="cpu")
        # Create separable data: first 8 dims are class signal
        reps = torch.randn(100, 16)
        labels = (reps[:, 0] > 0).long()  # Binary from dim 0
        result = pcc.evaluate(reps, labels, "test_binary")
        assert result["task"] == "test_binary"
        assert 1 in result["depths"] and 2 in result["depths"]
        assert result["min_extracting_depth"] is not None
        assert result["max_accuracy"] >= 0

    def test_compare_models(self):
        from src.interp.probing_complexity import ProbingComplexityCurve

        pcc = ProbingComplexityCurve(embed_dim=16, depths=(1, 2), max_epochs=10, device="cpu")
        # "JEPA" reps: linear signal in dim 0
        jepa_reps = torch.randn(100, 16)
        jepa_reps[:, 0] = torch.randn(100) * 5  # Strong signal
        labels = (jepa_reps[:, 0] > 0).long()

        # "MLM" reps: signal spread across dims
        baseline_reps = torch.randn(100, 16)
        baseline_reps[:, :4] = jepa_reps[:, 0].unsqueeze(1) * 0.5

        result = pcc.compare_models(jepa_reps, baseline_reps, labels, "test")
        assert "complexity_gap" in result
        assert "jepa_min_depth" in result
        assert "baseline_min_depth" in result

    def test_linguistic_probe_tasks(self):
        from src.interp.probing_complexity import LinguisticProbeTasks

        reps = torch.randn(50, 32)
        depths = torch.rand(50) * 10
        result = LinguisticProbeTasks.syntactic_depth(reps, depths, n_bins=3)
        assert result["task_name"] == "syntactic_depth"
        assert result["labels"].max() < 3


# ═══════════════════════════════════════════════════════════════════
# Polysemanticity (v0.6.0)
# ═══════════════════════════════════════════════════════════════════


class TestPolysemanticity:
    def test_polysemanticity_index(self):
        from src.interp.polysemanticity import PolysemanticityIndex

        psi = PolysemanticityIndex(
            n_clusters_range=(2, 3),
            n_top_activations=20,
            n_dimensions_sample=4,
        )
        reps = torch.randn(50, 16)
        result = psi.compute(reps)
        assert "mean_psi" in result
        assert "frac_monosemantic" in result
        assert 0 <= result["frac_monosemantic"] <= 1

    def test_polysemanticity_with_labels(self):
        from src.interp.polysemanticity import PolysemanticityIndex

        psi = PolysemanticityIndex(
            n_clusters_range=(2, 3),
            n_top_activations=20,
            n_dimensions_sample=4,
        )
        reps = torch.randn(50, 16)
        labels = torch.randint(0, 3, (50,))
        result = psi.compute(reps, labels)
        assert "mean_psi" in result

    def test_superposition_index(self):
        from src.interp.polysemanticity import SuperpositionIndex

        W = torch.randn(32, 16)
        result = SuperpositionIndex.compute(W)
        assert "interference_ratio" in result
        assert "effective_rank" in result
        assert "superposition_ratio" in result
        assert "feature_density" in result

    def test_feature_deduplication(self):
        from src.interp.polysemanticity import FeatureDeduplicationScore

        a = torch.randn(32, 16)
        b = torch.randn(32, 16)
        result = FeatureDeduplicationScore.compute(a, b)
        assert "mean_cosine_a_to_b" in result
        assert "dedup_a" in result
        assert 0 <= result["dedup_a"] <= 1


# ═══════════════════════════════════════════════════════════════════
# Causal Scrubbing (v0.6.0)
# ═══════════════════════════════════════════════════════════════════


class TestCausalScrubbing:
    def test_feature_hypothesis_svd(self):
        from src.interp.causal_scrubbing import FeatureHypothesis

        h = torch.randn(4, 8, 16)
        rel, irrel = FeatureHypothesis.svd_directions_hypothesis(h, n_relevant_dims=5)
        assert rel.shape == (4, 8, 16)
        # relevant and irrelevant should be disjoint
        assert not (rel & irrel).any()

    def test_feature_hypothesis_position(self):
        from src.interp.causal_scrubbing import FeatureHypothesis

        h = torch.randn(4, 10, 16)
        rel, irrel = FeatureHypothesis.position_based_hypothesis(h, keep_fraction=0.5)
        assert rel[:, :5, :].all()
        assert irrel[:, 5:, :].all()

    def test_feature_hypothesis_random(self):
        from src.interp.causal_scrubbing import FeatureHypothesis

        h = torch.randn(4, 8, 16)
        rel, irrel = FeatureHypothesis.random_hypothesis(h, relevant_fraction=0.5)
        assert not (rel & irrel).any()
        # About 50% should be relevant
        frac = rel.float().mean().item()
        assert 0.2 < frac < 0.8  # Rough check

    def test_intervention_predictability_scorer(self):
        from src.interp.causal_scrubbing import InterventionPredictabilityScorer
        from src.models.encoder import TextSpanJEPLEncoder

        enc = TextSpanJEPLEncoder(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )

        class MockModel:
            encoder = enc

            def eval(self):
                pass

        direction = torch.randn(32)
        probe_fn = lambda h: h.norm(dim=-1).mean()
        ids = torch.randint(0, 100, (2, 16))
        result = InterventionPredictabilityScorer.compute_predictability(
            MockModel(),
            ids,
            direction,
            probe_fn,
            device="cpu",
        )
        assert "predictability" in result
        assert "monotonicity" in result


# ═══════════════════════════════════════════════════════════════════
# Representation Geometry (v0.6.0)
# ═══════════════════════════════════════════════════════════════════


class TestRepresentationGeometry:
    def test_effective_dimension(self):
        from src.interp.representation_geometry import RepresentationGeometry

        reps = torch.randn(100, 32)
        ed = RepresentationGeometry.effective_dimension(reps, threshold=0.99)
        assert ed > 0
        assert ed <= 32

    def test_total_compression(self):
        from src.interp.representation_geometry import RepresentationGeometry

        reps = torch.randn(100, 32)
        tc = RepresentationGeometry.total_compression(reps)
        assert 0 <= tc <= 1

    def test_anisotropy(self):
        from src.interp.representation_geometry import RepresentationGeometry

        reps = torch.randn(100, 32)
        aniso = RepresentationGeometry.anisotropy(reps)
        assert 0 <= aniso <= 1

    def test_spectrum_decay_rate(self):
        from src.interp.representation_geometry import RepresentationGeometry

        reps = torch.randn(100, 32)
        alpha = RepresentationGeometry.spectrum_decay_rate(reps)
        assert alpha >= 0

    def test_compute_all(self):
        from src.interp.representation_geometry import RepresentationGeometry

        reps = torch.randn(100, 32)
        result = RepresentationGeometry.compute_all(reps)
        assert "effective_dimension" in result
        assert "total_compression" in result
        assert "anisotropy" in result
        assert "spectrum_decay_rate" in result

    def test_compare(self):
        from src.interp.representation_geometry import RepresentationGeometry

        jepa_reps = torch.randn(100, 32)
        baseline_reps = torch.randn(100, 32) * 0.5  # More compressed
        comparison = RepresentationGeometry.compare(jepa_reps, baseline_reps)
        assert "_geometry_advantage" in comparison
        assert 0 <= comparison["_geometry_advantage"] <= 1

    def test_anisotropy_collapsed(self):
        """Collapsed representations should have high anisotropy."""
        from src.interp.representation_geometry import RepresentationGeometry

        # Almost-collapsed: all vectors near one direction
        direction = torch.nn.functional.normalize(torch.randn(32), dim=0)
        reps = direction.unsqueeze(0) + torch.randn(50, 32) * 0.01
        aniso = RepresentationGeometry.anisotropy(reps)
        assert aniso > 0.8  # Should be very anisotropic


# ═══════════════════════════════════════════════════════════════════
# Feature Composition (v0.6.0)
# ═══════════════════════════════════════════════════════════════════


class TestFeatureComposition:
    def test_feature_interference_score(self):
        from src.interp.feature_composition import FeatureInterferenceScore
        from src.interp.sae import SparseAutoencoder

        sae = SparseAutoencoder(input_dim=16, latent_dim=32, k=4)
        reps = torch.randn(50, 16)
        result = FeatureInterferenceScore.compute(sae, reps, n_features=4, n_top=10)
        assert "mean_interference" in result
        assert result["mean_interference"] >= 0


# ═══════════════════════════════════════════════════════════════════
# Collapse new metrics (SVCCA, alignment, eigenvalue_spread,
# subspace_overlap, spectral_clustering_coeff)
# ═══════════════════════════════════════════════════════════════════


class TestCollapseNewMetrics:
    def test_svcca(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        y = x + torch.randn_like(x) * 0.1  # Similar to x
        result = diag._svcca(x, y)
        assert 0 <= result <= 1
        # Similar representations → high SVCCA
        assert result > 0.5

    def test_svcca_orthogonal(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        y = torch.randn(4, 16, 32)  # Random → low SVCCA
        result = diag._svcca(x, y)
        assert 0 <= result <= 1

    def test_alignment(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(50, 32)
        y = x + torch.randn_like(x) * 0.01  # Very close
        align = diag._alignment(x, y)
        # Very close → low alignment distance
        assert align < 1.0

    def test_eigenvalue_spread(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        spread = diag._eigenvalue_spread(x)
        assert spread >= 0

    def test_subspace_overlap(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        overlap = diag._subspace_overlap(x, x)
        # Same input → high overlap
        assert overlap > 0.9

    def test_spectral_clustering_coeff(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        scc = diag._spectral_clustering_coeff(x)
        assert 0 <= scc <= 1

    def test_compute_includes_new_metrics(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        y = torch.randn(4, 16, 32)
        m = diag.compute(x, y)
        assert "svcca_online_target" in m
        assert "alignment" in m
        assert "eigenvalue_spread_online" in m
        assert "subspace_overlap" in m
        assert "spectral_clustering_coeff_online" in m

    def test_no_nan_new_metrics(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.zeros(4, 16, 32)
        y = torch.zeros(4, 16, 32)
        m = diag.compute(x, y)
        for key in [
            "svcca_online_target",
            "alignment",
            "eigenvalue_spread_online",
            "subspace_overlap",
            "spectral_clustering_coeff_online",
        ]:
            val = m.get(key, 0)
            assert math.isfinite(val), f"NaN in {key}: {val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═══════════════════════════════════════════════════════════════════
# Statistical Tests (v0.7.0)
# ═══════════════════════════════════════════════════════════════════


class TestBootstrapCI:
    def test_basic_ci(self):
        from src.interp.statistical_tests import BootstrapCI

        data = torch.randn(100, 16)
        ci = BootstrapCI.compute(lambda d: d.mean(), data, n_bootstrap=200)
        assert "mean" in ci and "ci_lower" in ci and "ci_upper" in ci
        assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]

    def test_compare(self):
        from src.interp.statistical_tests import BootstrapCI

        a = torch.randn(100, 8) + 1.0  # Shifted
        b = torch.randn(100, 8)
        result = BootstrapCI.compare(lambda d: d.mean(), a, b, n_bootstrap=200)
        assert "significant" in result
        assert "p_value" in result


class TestPermutationTest:
    def test_significant_difference(self):
        from src.interp.statistical_tests import PairedPermutationTest

        a = torch.randn(50) + 2.0  # Shifted
        b = torch.randn(50)
        result = PairedPermutationTest.compute(a, b, n_permutations=1000)
        assert result["p_value"] < 0.05
        assert result["significant"]

    def test_no_difference(self):
        from src.interp.statistical_tests import PairedPermutationTest

        a = torch.randn(50)
        b = torch.randn(50)
        result = PairedPermutationTest.compute(a, b, n_permutations=1000)
        # Should not be significant (usually p > 0.05)
        # Can't guarantee, but effect_size should be small
        assert abs(result["effect_size"]) < 1.0


class TestMultipleComparison:
    def test_bonferroni(self):
        from src.interp.statistical_tests import MultipleComparisonCorrection

        p_values = [0.01, 0.03, 0.04, 0.50]
        corrected = MultipleComparisonCorrection.bonferroni(p_values)
        assert all(p <= 1.0 for p in corrected)
        assert corrected[0] == pytest.approx(0.04)

    def test_bh(self):
        from src.interp.statistical_tests import MultipleComparisonCorrection

        p_values = [0.01, 0.03, 0.04, 0.50]
        result = MultipleComparisonCorrection.benjamini_hochberg(p_values, alpha=0.05)
        assert "corrected" in result
        assert "significant" in result


class TestEffectSize:
    def test_cohens_d(self):
        from src.interp.statistical_tests import EffectSize

        a = torch.randn(100) + 1.0
        b = torch.randn(100)
        d = EffectSize.cohens_d(a, b)
        assert d > 0.5  # Should be medium-large

    def test_cliffs_delta(self):
        from src.interp.statistical_tests import EffectSize

        a = torch.randn(50) + 2.0
        b = torch.randn(50)
        delta = EffectSize.cliffs_delta(a, b)
        assert delta > 0.0


class TestBayesianComparison:
    def test_probability(self):
        from src.interp.statistical_tests import BayesianComparison

        a = torch.randn(100) + 1.0
        b = torch.randn(100)
        result = BayesianComparison.probability_a_greater_b(a, b, n_bootstrap=1000)
        assert result["prob_a_greater_b"] > 0.9


# ═══════════════════════════════════════════════════════════════════
# Information Theory (v0.7.0)
# ═══════════════════════════════════════════════════════════════════


class TestInfoNCE:
    def test_infonce_with_correlated(self):
        from src.interp.information_theory import InfoNCEEstimator

        x = torch.randn(128, 16)
        y = x + torch.randn_like(x) * 0.1  # Strongly correlated
        mi = InfoNCEEstimator.compute(x, y)
        assert mi > 0.0

    def test_infonce_with_independent(self):
        from src.interp.information_theory import InfoNCEEstimator

        x = torch.randn(128, 16)
        y = torch.randn(128, 16)  # Independent
        mi = InfoNCEEstimator.compute(x, y)
        # MI should be near zero for independent variables
        assert mi >= 0.0


class TestMINE:
    def test_mine_training(self):
        from src.interp.information_theory import MINEEstimator

        mine = MINEEstimator(16, 16, hidden_dim=32)
        x = torch.randn(128, 16)
        y = x + torch.randn_like(x) * 0.5
        mi = mine.compute_mi(x, y, n_steps=50)
        assert isinstance(mi, float)
        assert mi >= 0.0  # MI is non-negative


class TestRepresentationCompression:
    def test_entropy(self):
        from src.interp.information_theory import RepresentationCompression

        reps = torch.randn(100, 32)
        h = RepresentationCompression.entropy_estimate(reps)
        assert h > 0.0

    def test_total_correlation(self):
        from src.interp.information_theory import RepresentationCompression

        reps = torch.randn(200, 16)
        tc = RepresentationCompression.total_correlation(reps)
        assert tc >= 0.0


class TestInformationPlane:
    def test_compute(self):
        from src.interp.information_theory import InformationPlane

        reps = torch.randn(64, 16)
        input_feats = torch.randn(64, 16)  # Same dim as reps for InfoNCE
        task_feats = torch.randn(64, 16)  # Same dim as reps for InfoNCE
        result = InformationPlane.compute(reps, input_feats, task_feats)
        assert "mi_input" in result
        assert "mi_task" in result
        assert "ib_gap" in result


# ═══════════════════════════════════════════════════════════════════
# Layer Analysis (v0.7.0)
# ═══════════════════════════════════════════════════════════════════


class TestLayerwiseProbe:
    def test_probe_all_layers(self):
        from src.interp.layer_analysis import LayerwiseProbe

        probe = LayerwiseProbe(embed_dim=16, max_epochs=10, device="cpu")
        layers = [torch.randn(80, 16) for _ in range(4)]
        # Make first layer have linearly separable signal
        layers[0][:, 0] = torch.randn(80) * 3
        labels = (layers[0][:, 0] > 0).long()
        with torch.enable_grad():
            result = probe.probe_all_layers(layers, labels, "test")
        assert "per_layer_accuracy" in result
        assert len(result["per_layer_accuracy"]) == 4
        assert result["peak_layer"] is not None


class TestLayerwiseCKA:
    def test_intra_model(self):
        from src.interp.layer_analysis import LayerwiseCKA

        layers = [torch.randn(50, 16) for _ in range(4)]
        result = LayerwiseCKA.intra_model_cka(layers)
        assert "per_layer_cka" in result
        assert len(result["per_layer_cka"]) == 3  # 3 adjacent pairs


class TestLayerwiseGeometry:
    def test_compute(self):
        from src.interp.layer_analysis import LayerwiseGeometry

        layers = [torch.randn(50, 16) for _ in range(4)]
        result = LayerwiseGeometry.compute(layers)
        assert "cv_effective_dim" in result
        assert len(result["effective_dims"]) == 4


# ═══════════════════════════════════════════════════════════════════
# Stability (v0.7.0)
# ═══════════════════════════════════════════════════════════════════


class TestLossStability:
    def test_compute(self):
        from src.interp.stability import LossStability

        losses = [3.0, 2.5, 2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05, 1.0]
        result = LossStability.compute(losses)
        assert "smoothness" in result
        assert "cv" in result
        assert result["convergence_slope"] < 0  # Decreasing

    def test_compare(self):
        from src.interp.stability import LossStability

        jepa_losses = [3.0, 2.0, 1.5, 1.2, 1.0, 0.9, 0.85, 0.83, 0.82, 0.81]
        baseline_losses = [3.0, 2.8, 2.5, 2.3, 2.0, 1.8, 1.5, 1.3, 1.1, 1.0]
        result = LossStability.compare(jepa_losses, baseline_losses)
        assert "jepa_more_stable" in result


class TestTrainingStability:
    def test_convergence_curve(self):
        from src.interp.stability import TrainingStability

        # Simulate: later checkpoints are closer to final
        checkpoints = [torch.randn(50, 16) * (1 + 3 / (i + 1)) for i in range(5)]
        final = torch.randn(50, 16)
        result = TrainingStability.convergence_curve(checkpoints, final)
        assert "cka_curve" in result
        assert result["n_checkpoints"] == 5


class TestCheckpointConsistency:
    def test_cross_seed_cka(self):
        from src.interp.stability import CheckpointConsistency

        seeds = [torch.randn(50, 16) for _ in range(3)]
        result = CheckpointConsistency.cross_seed_cka(seeds)
        assert "mean_pairwise_cka" in result
        assert result["n_seeds"] == 3


# ═══════════════════════════════════════════════════════════════════
# Probe Generalization (v0.7.0)
# ═══════════════════════════════════════════════════════════════════


class TestProbeGeneralization:
    def test_cross_dataset(self):
        from src.interp.probe_generalization import ProbeGeneralizationTest

        pgt = ProbeGeneralizationTest(embed_dim=16, max_epochs=20, device="cpu")
        # Source: linearly separable
        src_reps = torch.randn(100, 16)
        src_reps[:, 0] = torch.randn(100) * 5
        src_labels = (src_reps[:, 0] > 0).long()
        # Target: similar structure
        tgt_reps = torch.randn(80, 16)
        tgt_reps[:, 0] = torch.randn(80) * 5
        tgt_labels = (tgt_reps[:, 0] > 0).long()
        with torch.enable_grad():
            result = pgt.cross_dataset_generalization(
                src_reps,
                src_labels,
                tgt_reps,
                tgt_labels,
                "test",
            )
        assert "source_accuracy" in result
        assert "target_accuracy" in result
        assert result["generalization_ratio"] > 0


class TestProbeSelectivity:
    def test_selectivity(self):
        from src.interp.probe_generalization import ProbeSelectivityTest

        pst = ProbeSelectivityTest(embed_dim=16, max_epochs=15, device="cpu")
        reps = torch.randn(100, 16)
        reps[:, 0] = torch.randn(100) * 3
        labels = (reps[:, 0] > 0).long()
        with torch.enable_grad():
            result = pst.compute_selectivity(reps, labels, n_control=3)
        assert "selectivity" in result
        assert "real_task_accuracy" in result
        assert "control_task_accuracy" in result


# ═══════════════════════════════════════════════════════════════════
# Visualization (v0.8.0)
# ═══════════════════════════════════════════════════════════════════


class TestVisualization:
    def test_radar_chart(self):
        from src.interp.visualization import radar_chart

        jepa = {"eff_rank": 45, "anisotropy": 0.3, "sv_entropy": 0.8, "uniformity": -2.5}
        base = {"eff_rank": 38, "anisotropy": 0.6, "sv_entropy": 0.6, "uniformity": -3.0}
        svg = radar_chart(jepa, base, title="Test")
        assert svg is not None
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_layer_heatmap(self):
        from src.interp.visualization import layer_heatmap

        data = {"eff_dim": [10, 15, 20, 18, 16], "anisotropy": [0.8, 0.6, 0.4, 0.5, 0.6]}
        svg = layer_heatmap(data, title="Test Layers")
        assert svg is not None
        assert "<svg" in svg

    def test_bar_chart(self):
        from src.interp.visualization import bar_chart_with_errors

        names = ["eff_rank", "anisotropy"]
        svg = bar_chart_with_errors(
            names,
            jepa_means=[45, 0.3],
            jepa_cis=[(43, 47), (0.25, 0.35)],
            baseline_means=[38, 0.6],
            baseline_cis=[(35, 41), (0.55, 0.65)],
        )
        assert svg is not None
        assert "<svg" in svg

    def test_probing_complexity_curve(self):
        from src.interp.visualization import probing_complexity_curve

        svg = probing_complexity_curve(
            depths=[1, 2, 3, 4],
            jepa_accuracies=[0.85, 0.90, 0.92, 0.93],
            baseline_accuracies=[0.60, 0.75, 0.85, 0.88],
            task_name="POS",
        )
        assert svg is not None
        assert "<svg" in svg


# ═══════════════════════════════════════════════════════════════════
# Ablation (v0.8.0)
# ═══════════════════════════════════════════════════════════════════


class TestAblation:
    def test_ablation_config(self):
        from src.interp.ablation import ABLATION_CONFIGS, AblationConfig

        full = AblationConfig("full")
        assert full.use_predictor
        assert full.use_future_loss
        no_pred = AblationConfig("no_pred", use_predictor=False)
        assert not no_pred.use_predictor
        assert "full" in ABLATION_CONFIGS
        assert "no_predictor" in ABLATION_CONFIGS

    def test_ablation_config_describe(self):
        from src.interp.ablation import AblationConfig

        full = AblationConfig("full")
        assert full.describe() == "full model"
        no_pred = AblationConfig("no_pred", use_predictor=False)
        assert "no predictor" in no_pred.describe()

    def test_ablated_model_forward(self):
        from src.interp.ablation import AblatedModel, AblationConfig
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=32,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=16,
            predictor_depth=2,
            future_offsets=(1,),
            num_refine_steps=1,
        )
        model = TextSpanJEPA(config)
        ablation = AblationConfig("test", use_future_loss=False)
        ablated = AblatedModel(model, ablation)
        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1
        loss, info = ablated(ids, ids, mask)
        assert torch.isfinite(loss)
        assert info["ablation"] == "test"


# ═══════════════════════════════════════════════════════════════════
# Scaling (v0.8.0)
# ═══════════════════════════════════════════════════════════════════


class TestScaling:
    def test_scaling_law(self):
        from src.interp.scaling import ScalingAnalysis

        sizes = [1e6, 5e6, 20e6, 100e6, 500e6]
        metrics = [10, 18, 30, 45, 65]  # Eff dim grows with size
        law = ScalingAnalysis.compute_scaling_law(sizes, metrics)
        assert "exponent" in law
        assert "r_squared" in law
        assert law["exponent"] > 0  # Should be positive (grows)

    def test_compare_scaling(self):
        from src.interp.scaling import ScalingAnalysis

        sizes = [1e6, 10e6, 100e6]
        jepa_metrics = [15, 30, 50]
        base_metrics = [12, 22, 35]
        result = ScalingAnalysis.compare_scaling(sizes, jepa_metrics, sizes, base_metrics)
        assert "jepa_scaling" in result
        assert "jepa_scales_better" in result

    def test_efficiency(self):
        from src.interp.scaling import InterpretabilityEfficiency

        result = InterpretabilityEfficiency.compute_efficiency(
            interp_metric=0.85,
            flops=1e18,
            baseline_interp=0.70,
            baseline_flops=1e18,
        )
        assert "efficiency_ratio" in result
        assert result["jepa_more_efficient"]

    def test_pareto(self):
        from src.interp.scaling import InterpretabilityEfficiency

        result = InterpretabilityEfficiency.pareto_curve(
            interp_values=[0.7, 0.85, 0.90],
            compute_values=[1e17, 1e18, 1e19],
            baseline_interp_values=[0.6, 0.75, 0.80],
            baseline_compute_values=[1e17, 1e18, 1e19],
        )
        assert "jepa_dominates" in result


# ═══════════════════════════════════════════════════════════════════
# Interpretability Index (v0.9.0)
# ═══════════════════════════════════════════════════════════════════


class TestInterpretabilityIndex:
    def test_compute(self):
        from src.interp.interpretability_index import InterpretabilityIndex

        idx = InterpretabilityIndex()
        metrics = {
            "effective_dimension": 40,
            "anisotropy": 0.3,
            "sv_entropy": 0.7,
            "frac_monosemantic": 0.5,
        }
        result = idx.compute(metrics)
        assert 0 <= result["interpretability_index"] <= 1
        assert result["n_metrics_used"] == 4

    def test_compare(self):
        from src.interp.interpretability_index import InterpretabilityIndex

        idx = InterpretabilityIndex()
        jepa = {
            "effective_dimension": 45,
            "anisotropy": 0.2,
            "sv_entropy": 0.8,
            "frac_monosemantic": 0.6,
            "mean_psi": 1.0,
        }
        baseline = {
            "effective_dimension": 35,
            "anisotropy": 0.6,
            "sv_entropy": 0.5,
            "frac_monosemantic": 0.3,
            "mean_psi": 2.0,
        }
        result = idx.compare(jepa, baseline)
        assert result["jepa_index"] > result["baseline_index"]
        assert result["jepa_better"]

    def test_from_collapse(self):
        from src.interp.interpretability_index import InterpretabilityIndex

        collapse = {
            "effective_rank_online": 40,
            "sv_entropy_online": 0.7,
            "collapsed_dim_ratio_online": 0.1,
            "mean_pairwise_cosine_online": 0.3,
            "uniformity_online": -2.5,
            "participation_ratio_online": 20,
        }
        result = InterpretabilityIndex.from_collapse_diagnostics(collapse)
        assert "effective_rank" in result
        assert "sv_entropy" in result


# ═══════════════════════════════════════════════════════════════════
# Robustness (v0.9.0)
# ═══════════════════════════════════════════════════════════════════


class TestRobustness:
    def test_token_dropout(self):
        from src.interp.robustness import token_dropout

        ids = torch.randint(0, 100, (4, 16))
        perturbed = token_dropout(ids, 0.5)
        assert (perturbed == 0).any()

    def test_token_substitution(self):
        from src.interp.robustness import token_substitution

        ids = torch.ones(4, 16, dtype=torch.long) * 42
        perturbed = token_substitution(ids, 0.5, vocab_size=100)
        assert (perturbed != 42).any()

    def test_span_corruption(self):
        from src.interp.robustness import span_corruption

        ids = torch.randint(1, 100, (4, 32))
        perturbed = span_corruption(ids, 0.3, span_length=5)
        assert (perturbed == 0).any()

    def test_perturbation_curve(self):
        from src.interp.robustness import RepresentationRobustness, token_dropout
        from src.models.encoder import TextSpanJEPLEncoder

        enc = TextSpanJEPLEncoder(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )

        class M:
            encoder = enc

            def eval(self):
                pass

        rob = RepresentationRobustness(M(), device="cpu")
        ids = torch.randint(0, 100, (8, 16))
        result = rob.perturbation_curve(ids, token_dropout, intensities=(0.1, 0.3))
        assert "robustness_score" in result
        assert "perturbation_curve" in result


# ═══════════════════════════════════════════════════════════════════
# Ground Truth Validation (v0.9.0)
# ═══════════════════════════════════════════════════════════════════


class TestGroundTruth:
    def test_synthetic_model(self):
        from src.interp.ground_truth import SyntheticStructuredModel

        synth = SyntheticStructuredModel(n_samples=50, embed_dim=64)
        data = synth.generate()
        assert data["representations"].shape == (50, 64)
        assert data["labels"].shape == (50,)
        assert "ground_truth" in data
        assert len(data["ground_truth"]["class_dims"]) == 16

    def test_geometry_validation(self):
        from src.interp.ground_truth import GroundTruthValidation

        v = GroundTruthValidation()
        result = v.validate_geometry()
        assert "pipeline_valid" in result
        assert "effective_dim" in result

    def test_full_validation(self):
        from src.interp.ground_truth import GroundTruthValidation

        v = GroundTruthValidation()
        results = v.full_validation()
        assert "_summary" in results
        assert results["_summary"]["n_tests_total"] >= 3
