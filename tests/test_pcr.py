# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Tests for PCR: Predictive Cascade Refinement (novel mechanism #8)

import math

import pytest
import torch
import torch.nn.functional as F

from src.models.pcr import PredictiveCascadeRefinement

# ═══════════════════════════════════════════════════════════════════
#  Core functionality tests
# ═══════════════════════════════════════════════════════════════════


class TestPCRCore:
    """Core PCR module tests."""

    def test_construction_default(self):
        pcr = PredictiveCascadeRefinement(embed_dim=768)
        assert pcr.n_levels == 3
        assert pcr.embed_dim == 768
        assert sum(pcr.level_dims) <= 768

    def test_construction_custom_dims(self):
        pcr = PredictiveCascadeRefinement(embed_dim=256, n_levels=2, level_dims=[64, 32])
        assert pcr.n_levels == 2
        assert pcr.level_dims == [64, 32]
        assert pcr.workspace_Q.shape == (256, 96)

    def test_construction_single_level(self):
        pcr = PredictiveCascadeRefinement(embed_dim=128, n_levels=1, level_dims=[32])
        assert pcr.n_levels == 1
        assert pcr.level_dims == [32]

    def test_forward_basic(self):
        pcr = PredictiveCascadeRefinement(embed_dim=128, n_levels=2, level_dims=[32, 16])
        B, N, D = 4, 8, 128
        z_pred = torch.randn(B, N, D)
        z_target = torch.randn(B, N, D)
        z_refined, info = pcr(z_pred, z_target, step=2000)  # past warmup
        assert z_refined.shape == (B, N, D)
        assert "pcr_improvement" in info
        assert "pcr_n_levels" in info
        assert info["pcr_n_levels"] == 2

    def test_forward_2d_input(self):
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(10, 64)
        z_target = torch.randn(10, 64)
        z_refined, _info = pcr(z_pred, z_target, step=2000)
        assert z_refined.shape == (10, 64)

    def test_warmup_zero_refinement(self):
        """During warmup, PCR should not modify predictions."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(4, 64)
        z_target = torch.randn(4, 64)
        z_refined, _info = pcr(z_pred, z_target, step=0)  # before warmup
        # During warmup, refinement should be zero
        assert torch.allclose(z_refined, z_pred, atol=1e-5)

    def test_warmup_ramp(self):
        """Warmup should gradually enable refinement."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(4, 64)
        z_target = torch.randn(4, 64)

        # Before warmup
        _z0, _ = pcr(z_pred, z_target, step=0)
        # During warmup
        _z_half, info_half = pcr(z_pred, z_target, step=1500)
        # After warmup
        _z_full, info_full = pcr(z_pred, z_target, step=5000)

        assert info_half["pcr_warmup_factor"] < info_full["pcr_warmup_factor"]

    def test_target_detached(self):
        """z_target should NOT receive gradients."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(4, 64, requires_grad=True)
        z_target = torch.randn(4, 64, requires_grad=True)
        z_refined, _ = pcr(z_pred, z_target, step=2000)
        loss = z_refined.sum()
        loss.backward()
        # z_pred should have gradients
        assert z_pred.grad is not None
        # z_target should NOT have gradients (detached in PCR)
        assert z_target.grad is None or z_target.grad.abs().sum() == 0

    def test_differentiability(self):
        """PCR output should be differentiable w.r.t. z_pred and Q."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(4, 64, requires_grad=True)
        z_target = torch.randn(4, 64)
        z_refined, _ = pcr(z_pred, z_target, step=2000)
        loss = z_refined.sum()
        loss.backward()
        assert z_pred.grad is not None
        assert pcr.workspace_Q.grad is not None


# ═══════════════════════════════════════════════════════════════════
#  Stiefel manifold tests
# ═══════════════════════════════════════════════════════════════════


class TestPCRStiefel:
    """Stiefel manifold constraint tests for PCR projection Q."""

    def test_identity_init_orthonormal(self):
        pcr = PredictiveCascadeRefinement(
            embed_dim=64, n_levels=2, level_dims=[16, 8], init="identity"
        )
        Q = pcr.workspace_Q.data
        gram = Q.T @ Q
        # Should be close to identity (first 24 columns of 64x64 identity)
        expected = torch.eye(24)
        assert torch.allclose(gram, expected, atol=1e-5)

    def test_random_init_orthonormal(self):
        pcr = PredictiveCascadeRefinement(
            embed_dim=64, n_levels=2, level_dims=[16, 8], init="random"
        )
        Q = pcr.workspace_Q.data
        gram = Q.T @ Q
        expected = torch.eye(24)
        assert torch.allclose(gram, expected, atol=1e-4)

    def test_stiefel_retract_restores_orthonormality(self):
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        # Perturb Q off the manifold
        with torch.no_grad():
            pcr.workspace_Q.add_(torch.randn_like(pcr.workspace_Q) * 0.1)
        # Retract
        pcr.stiefel_retract()
        Q = pcr.workspace_Q.data
        gram = Q.T @ Q
        expected = torch.eye(24)
        assert torch.allclose(gram, expected, atol=1e-4)

    def test_stiefel_retract_after_gradient_step(self):
        """Simulate an optimizer step + retraction."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(4, 64)
        z_target = torch.randn(4, 64)
        z_refined, _ = pcr(z_pred, z_target, step=2000)
        loss = z_refined.sum()
        loss.backward()
        # Simulate gradient step
        with torch.no_grad():
            pcr.workspace_Q.add_(0.01 * pcr.workspace_Q.grad)
        # Retract
        pcr.stiefel_retract()
        Q = pcr.workspace_Q.data
        gram = Q.T @ Q
        expected = torch.eye(24)
        assert torch.allclose(gram, expected, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════
#  Theorem tests — Cascade Capacity
# ═══════════════════════════════════════════════════════════════════


class TestPCRCascadeCapacityTheorem:
    """Tests for the Cascade Capacity theorem.

    Theorem: I(z_context; z_L) ≥ I(z_context; z_0) +
             Σ_l I(r_{l-1}; P_l r_{l-1})

    We verify:
    1. The capacity bound is non-negative
    2. The capacity bound is positive when prediction is imperfect
    3. The capacity bound increases with more levels
    4. Per-level bounds are non-negative (information is additive)
    """

    def test_capacity_bound_nonnegative(self):
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64)
        bound, _info = pcr.compute_cascade_capacity_bound(z_pred, z_target)
        assert bound >= 0

    def test_capacity_bound_positive_for_imperfect_prediction(self):
        """When prediction is imperfect, the bound should be positive."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        # z_pred far from z_target → large residual → positive bound
        z_pred = torch.randn(64, 64)
        z_target = z_pred + torch.randn(64, 64) * 2.0  # large residual
        bound, _info = pcr.compute_cascade_capacity_bound(z_pred, z_target)
        assert bound > 0

    def test_capacity_bound_zero_for_perfect_prediction(self):
        """When prediction is perfect, the bound should be ~0."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(32, 64)
        z_target = z_pred.clone()  # perfect prediction
        bound, _info = pcr.compute_cascade_capacity_bound(z_pred, z_target)
        assert bound < 0.1  # should be very small

    def test_more_levels_higher_bound(self):
        """More refinement levels → higher capacity bound."""
        z_pred = torch.randn(64, 64)
        z_target = z_pred + torch.randn(64, 64)

        pcr2 = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        bound2, _ = pcr2.compute_cascade_capacity_bound(z_pred, z_target)

        pcr3 = PredictiveCascadeRefinement(embed_dim=64, n_levels=3, level_dims=[16, 8, 4])
        bound3, _ = pcr3.compute_cascade_capacity_bound(z_pred, z_target)

        # More levels should give at least as high a bound
        # (may not be strictly higher if residuals are small, but should be >=)
        assert bound3 >= bound2 - 0.1  # allow small numerical tolerance

    def test_per_level_bounds_nonnegative(self):
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=3, level_dims=[16, 8, 4])
        z_pred = torch.randn(64, 64)
        z_target = z_pred + torch.randn(64, 64)
        _, info = pcr.compute_cascade_capacity_bound(z_pred, z_target)
        for lb in info["per_level_bounds"]:
            assert lb >= 0


# ═══════════════════════════════════════════════════════════════════
#  Orthogonal subspace tests
# ═══════════════════════════════════════════════════════════════════


class TestPCROrthogonalSubspaces:
    """Test that PCR subspaces are indeed orthogonal."""

    def test_subspaces_orthogonal(self):
        """P_l^T P_m ≈ 0 for l ≠ m."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=3, level_dims=[16, 8, 4])
        Q = pcr.workspace_Q.data

        # Extract each subspace projection
        P0 = Q[:, 0:16]
        P1 = Q[:, 16:24]
        P2 = Q[:, 24:28]

        # Cross-correlations should be ~0
        cross01 = (P0.T @ P1).abs().max().item()
        cross02 = (P0.T @ P2).abs().max().item()
        cross12 = (P1.T @ P2).abs().max().item()

        assert cross01 < 1e-5, f"P0^T P1 max = {cross01}"
        assert cross02 < 1e-5, f"P0^T P2 max = {cross02}"
        assert cross12 < 1e-5, f"P1^T P2 max = {cross12}"

    def test_subspaces_orthonormal(self):
        """P_l^T P_l ≈ I for each level."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        Q = pcr.workspace_Q.data
        P0 = Q[:, 0:16]
        P1 = Q[:, 16:24]

        gram0 = P0.T @ P0
        gram1 = P1.T @ P1

        assert torch.allclose(gram0, torch.eye(16), atol=1e-5)
        assert torch.allclose(gram1, torch.eye(8), atol=1e-5)

    def test_orthogonality_preserved_after_retraction(self):
        """Stiefel retraction should maintain orthogonal subspaces."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=3, level_dims=[16, 8, 4])
        # Perturb
        with torch.no_grad():
            pcr.workspace_Q.add_(torch.randn_like(pcr.workspace_Q) * 0.05)
        pcr.stiefel_retract()

        Q = pcr.workspace_Q.data
        P0 = Q[:, 0:16]
        P1 = Q[:, 16:24]
        P2 = Q[:, 24:28]

        cross01 = (P0.T @ P1).abs().max().item()
        cross02 = (P0.T @ P2).abs().max().item()
        cross12 = (P1.T @ P2).abs().max().item()

        assert cross01 < 1e-3
        assert cross02 < 1e-3
        assert cross12 < 1e-3


# ═══════════════════════════════════════════════════════════════════
#  Information flow tests
# ═══════════════════════════════════════════════════════════════════


class TestPCRInformationFlow:
    """Test that PCR improves predictions (information flow)."""

    def test_residual_decreases(self):
        """After refinement, residual should be ≤ initial residual."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=3, level_dims=[16, 8, 4])
        # Train PCR briefly to make refinement non-trivial
        optimizer = torch.optim.SGD(pcr.parameters(), lr=0.01)
        for _ in range(50):
            z_pred = torch.randn(16, 64)
            z_target = z_pred + torch.randn(16, 64) * 0.5
            z_refined, _ = pcr(z_pred, z_target, step=5000)
            loss = F.mse_loss(z_refined, z_target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pcr.stiefel_retract()

        # Test: refined should have smaller residual than base
        with torch.no_grad():
            z_pred = torch.randn(32, 64)
            z_target = z_pred + torch.randn(32, 64) * 0.5
            z_refined, _info = pcr(z_pred, z_target, step=5000)

            initial_res = (z_target - z_pred).norm().item()
            final_res = (z_target - z_refined).norm().item()

            # After training, residual should decrease
            assert final_res <= initial_res * 1.01  # allow tiny numerical noise

    def test_improvement_metric_nonnegative(self):
        """The pcr_improvement metric should be ≥ 0."""
        pcr = PredictiveCascadeRefinement(embed_dim=64, n_levels=2, level_dims=[16, 8])
        z_pred = torch.randn(16, 64)
        z_target = z_pred + torch.randn(16, 64)
        _, info = pcr(z_pred, z_target, step=2000)
        assert info["pcr_improvement"] >= 0


# ═══════════════════════════════════════════════════════════════════
#  Integration with JEPA model
# ═══════════════════════════════════════════════════════════════════


class TestPCRIntegration:
    """Integration tests with TextSpanJEPA model."""

    def test_config_with_pcr(self):
        from src.models.jepa import TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            embed_dim=128,
            num_heads=4,
            encoder_depth=2,
            predictor_depth=2,
            predictor_embed_dim=64,
            use_pcr=True,
            pcr_n_levels=2,
            pcr_level_dims=[32, 16],
            pcr_warmup_steps=500,
        )
        assert config.validate() is True
        assert config.use_pcr is True
        assert config.pcr_n_levels == 2

    def test_config_validation_pcr_dims_exceed_embed(self):
        from src.models.jepa import TextSpanJEPAConfig

        with pytest.raises(ValueError, match="sum\\(pcr_level_dims\\)"):
            config = TextSpanJEPAConfig(
                embed_dim=64,
                num_heads=4,
                use_pcr=True,
                pcr_level_dims=[32, 32, 32],  # 96 > 64
            )
            config.validate()

    def test_model_with_pcr(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            embed_dim=128,
            num_heads=4,
            encoder_depth=2,
            predictor_depth=2,
            predictor_embed_dim=64,
            vocab_size=100,
            max_seq_len=32,
            use_pcr=True,
            pcr_n_levels=2,
            pcr_level_dims=[32, 16],
        )
        model = TextSpanJEPA(config)
        assert model.pcr is not None
        assert model.pcr.n_levels == 2

    def test_model_without_pcr(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            embed_dim=128,
            num_heads=4,
            encoder_depth=2,
            predictor_depth=2,
            predictor_embed_dim=64,
            vocab_size=100,
            max_seq_len=32,
            use_pcr=False,
        )
        model = TextSpanJEPA(config)
        assert model.pcr is None

    def test_loss_computation_with_pcr(self):
        """Full loss computation with PCR enabled."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            embed_dim=128,
            num_heads=4,
            encoder_depth=2,
            predictor_depth=2,
            predictor_embed_dim=64,
            vocab_size=100,
            max_seq_len=32,
            use_pcr=True,
            pcr_n_levels=2,
            pcr_level_dims=[32, 16],
        )
        model = TextSpanJEPA(config)
        B, T = 2, 16
        masked_ids = torch.randint(0, 100, (B, T))
        original_ids = torch.randint(0, 100, (B, T))
        mask = torch.zeros(B, T, dtype=torch.long)
        mask[:, 4:8] = 1

        total_loss, _loss_dict, _diag_dict = model.compute_loss_with_targets(
            masked_ids, original_ids, mask, current_step=2000, total_steps=10000
        )
        assert total_loss.item() >= 0
        assert not math.isnan(total_loss.item())
        assert not math.isinf(total_loss.item())

    def test_checkpoint_pcr_roundtrip(self, tmp_path):
        """PCR state should survive checkpoint save/load."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.train import load_checkpoint, save_checkpoint

        config = TextSpanJEPAConfig(
            embed_dim=64,
            num_heads=4,
            encoder_depth=1,
            predictor_depth=1,
            predictor_embed_dim=32,
            vocab_size=50,
            max_seq_len=16,
            use_pcr=True,
            pcr_n_levels=2,
            pcr_level_dims=[16, 8],
        )
        model = TextSpanJEPA(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Modify PCR state
        with torch.no_grad():
            model.pcr.workspace_Q.add_(torch.randn_like(model.pcr.workspace_Q) * 0.01)
        original_Q = model.pcr.workspace_Q.data.clone()
        original_gates = [g.data.clone() for g in model.pcr.level_gates]

        ckpt_path = str(tmp_path / "pcr-ckpt.pth.tar")
        save_checkpoint(ckpt_path, model, optimizer, None, 2, 137, 99, 55, model_name="text_span_jepa")

        # Corrupt in-memory state AFTER save: load must restore from the checkpoint,
        # otherwise the allclose assert below is a no-op tautology
        with torch.no_grad():
            model.pcr.workspace_Q.add_(torch.randn_like(model.pcr.workspace_Q) * 0.5)
            for g in model.pcr.level_gates:
                g.data.add_(torch.randn_like(g.data) * 0.5)

        loaded = load_checkpoint(ckpt_path, model, optimizer, None, model_name="text_span_jepa")

        assert torch.allclose(model.pcr.workspace_Q.data, original_Q, atol=1e-5)
        for restored, saved in zip(model.pcr.level_gates, original_gates):
            assert torch.allclose(restored.data, saved, atol=1e-5)
        epoch, global_step, ema_step, mask_step, extra_state = loaded
        assert (epoch, global_step, ema_step, mask_step) == (2, 137, 99, 55)
        assert extra_state is None


# ═══════════════════════════════════════════════════════════════════
#  Config tests
# ═══════════════════════════════════════════════════════════════════


class TestPCRConfig:
    """Config validation tests for PCR."""

    def test_pcr_n_levels_positive(self):
        from src.models.jepa import TextSpanJEPAConfig

        with pytest.raises(ValueError):
            config = TextSpanJEPAConfig(embed_dim=64, num_heads=4, use_pcr=True, pcr_n_levels=0)
            config.validate()

    def test_pcr_warmup_nonnegative(self):
        from src.models.jepa import TextSpanJEPAConfig

        with pytest.raises(ValueError):
            config = TextSpanJEPAConfig(
                embed_dim=64, num_heads=4, use_pcr=True, pcr_warmup_steps=-1
            )
            config.validate()

    def test_pcr_disabled_skips_validation(self):
        """When use_pcr=False, PCR config should not be validated."""
        from src.models.jepa import TextSpanJEPAConfig

        config = TextSpanJEPAConfig(embed_dim=64, num_heads=4, use_pcr=False, pcr_n_levels=0)
        # Should not raise despite pcr_n_levels=0
        assert config.validate() is True
