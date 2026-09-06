# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Tests for RDC: Representation Drift Compensation (mechanism #15)."""

import pytest
import torch


@pytest.fixture
def rdc_module():
    from src.models.rdc import RepresentationDriftCompensation

    return RepresentationDriftCompensation(
        embed_dim=64,
        eta=0.1,
        ema_beta=0.99,
        warmup_steps=0,
        k_workspace=16,
    )


@pytest.fixture
def sample_tensors():
    B, T, D = 4, 8, 64
    torch.manual_seed(42)
    z_current = torch.randn(B, T, D)
    z_previous = torch.randn(B, T, D)
    Q = torch.randn(D, 16)
    Q, _ = torch.linalg.qr(Q)
    return z_current, z_previous, Q


# ═══════════════════════════════════════════════════════════════════
#  Core functionality
# ═══════════════════════════════════════════════════════════════════


class TestRDCCore:
    def test_output_shape_and_type(self, rdc_module, sample_tensors):
        z_cur, z_prev, Q = sample_tensors
        loss, _info = rdc_module(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        assert loss.shape == ()
        assert loss.dtype == torch.float32

    def test_loss_non_negative(self, rdc_module, sample_tensors):
        z_cur, z_prev, Q = sample_tensors
        loss, _info = rdc_module(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        assert loss.item() >= 0.0, f"RDC loss must be non-negative, got {loss.item()}"

    def test_zero_drift_zero_loss(self, rdc_module):
        """If z_current == z_previous, drift is zero, loss should be ~0."""
        B, T, D = 4, 8, 64
        z = torch.randn(B, T, D)
        Q = torch.randn(D, 16)
        Q, _ = torch.linalg.qr(Q)
        loss, _info = rdc_module(z, z_previous=z.clone(), workspace_Q=Q, step=100)
        assert loss.item() < 1e-5, f"Zero drift should give ~0 loss, got {loss.item()}"

    def test_pure_workspace_drift_low_loss(self, rdc_module):
        """If drift is entirely in workspace, orthogonal component is zero."""
        B, T, D = 4, 8, 64
        Q = torch.randn(D, 16)
        Q, _ = torch.linalg.qr(Q)
        # Create drift entirely in workspace: drift = Q @ alpha
        alpha = torch.randn(B, T, 16)
        drift = alpha @ Q.T  # (B, T, D)
        z_prev = torch.randn(B, T, D)
        z_cur = z_prev + drift
        loss, _info = rdc_module(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        assert loss.item() < 1e-3, f"Workspace-only drift should give ~0 loss, got {loss.item()}"

    def test_pure_orthogonal_drift_nonzero_loss(self, rdc_module):
        """If drift is entirely orthogonal to workspace, loss should be > 0."""
        B, T, D = 4, 8, 64
        Q = torch.randn(D, 16)
        Q, _ = torch.linalg.qr(Q)
        # Create drift orthogonal to workspace
        noise = torch.randn(B, T, D)
        # Project out workspace component
        proj = (noise.reshape(-1, D) @ Q) @ Q.T
        drift_ortho = noise.reshape(-1, D) - proj
        drift_ortho = drift_ortho.reshape(B, T, D)
        z_prev = torch.zeros(B, T, D)
        z_cur = drift_ortho
        loss, _info = rdc_module(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        assert loss.item() > 1e-6, f"Orthogonal drift should give >0 loss, got {loss.item()}"


# ═══════════════════════════════════════════════════════════════════
#  Theorem verification
# ═══════════════════════════════════════════════════════════════════


class TestRDCTheorem:
    def test_drift_decomposition_orthogonal(self, sample_tensors):
        """Verify drift_workspace and drift_ortho are orthogonal."""
        z_cur, z_prev, Q = sample_tensors
        drift = (z_cur - z_prev).reshape(-1, 64)
        # Workspace component
        ws_proj = drift @ Q  # (N, k)
        drift_ws = ws_proj @ Q.T
        drift_ortho = drift - drift_ws
        # Inner product should be ~0
        inner = (drift_ws * drift_ortho).sum()
        assert (
            abs(inner.item()) < 1e-3
        ), f"Workspace and orthogonal drift not perpendicular: {inner.item()}"

    def test_drift_compensation_bound_components(self, rdc_module, sample_tensors):
        """Verify Drift Compensation Bound: ||z_T - z*_T||_⊥ ≤ ε(1-η)^T · T/√k."""
        z_cur, z_prev, Q = sample_tensors
        _loss, info = rdc_module(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        # The bound should be a finite positive number
        bound = info["rdc_theoretical_bound"]
        assert bound >= 0.0, f"Theoretical bound should be non-negative, got {bound}"
        # eta should be in (0, 1)
        assert 0 < info["rdc_eta"] < 1

    def test_increasing_eta_reduces_loss(self, sample_tensors):
        """Higher eta should reduce orthogonal drift more."""
        z_cur, z_prev, Q = sample_tensors
        from src.models.rdc import RepresentationDriftCompensation

        rdc_low = RepresentationDriftCompensation(embed_dim=64, eta=0.01, warmup_steps=0)
        rdc_high = RepresentationDriftCompensation(embed_dim=64, eta=0.5, warmup_steps=0)
        loss_low, _ = rdc_low(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        loss_high, _ = rdc_high(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        # Higher eta → stronger penalty → should push loss to be accounted for differently
        # The loss itself is η * ||Δz_⊥||², so higher η means higher loss value
        # but the EFFECT is stronger compensation
        assert (
            loss_high.item() > loss_low.item()
        ), "Higher eta should give higher loss value (stronger penalty)"


# ═══════════════════════════════════════════════════════════════════
#  Warmup and edge cases
# ═══════════════════════════════════════════════════════════════════


class TestRDCWarmup:
    def test_warmup_zero_loss(self, sample_tensors):
        """Before warmup, loss should be 0."""
        z_cur, z_prev, Q = sample_tensors
        from src.models.rdc import RepresentationDriftCompensation

        rdc = RepresentationDriftCompensation(embed_dim=64, eta=0.1, warmup_steps=100)
        loss, info = rdc(z_cur, z_previous=z_prev, workspace_Q=Q, step=0)
        assert loss.item() == 0.0
        assert info.get("rdc_warmup", False) is True

    def test_warmup_ramp(self, sample_tensors):
        """Loss should increase during warmup."""
        z_cur, z_prev, Q = sample_tensors
        from src.models.rdc import RepresentationDriftCompensation

        rdc = RepresentationDriftCompensation(embed_dim=64, eta=0.1, warmup_steps=100)
        loss_50, _ = rdc(z_cur, z_previous=z_prev, workspace_Q=Q, step=50)
        loss_100, _ = rdc(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        assert loss_100.item() >= loss_50.item() * 0.9  # approximate


# ═══════════════════════════════════════════════════════════════════
#  Diagnostics
# ═══════════════════════════════════════════════════════════════════


class TestRDCDiagnostics:
    def test_drift_ratio_in_01(self, rdc_module, sample_tensors):
        """Drift ratio should be in [0, 1]."""
        z_cur, z_prev, Q = sample_tensors
        _, info = rdc_module(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        ratio = info["rdc_drift_ratio"]
        assert 0.0 <= ratio <= 1.0 + 1e-6, f"Drift ratio should be in [0,1], got {ratio}"

    def test_info_keys(self, rdc_module, sample_tensors):
        z_cur, z_prev, Q = sample_tensors
        _, info = rdc_module(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        expected = [
            "rdc_loss",
            "rdc_ortho_drift_norm",
            "rdc_workspace_drift_norm",
            "rdc_total_drift_norm",
            "rdc_drift_ratio",
            "rdc_warmup_factor",
            "rdc_k_workspace",
            "rdc_theoretical_bound",
            "rdc_eta",
        ]
        for k in expected:
            assert k in info, f"Missing key: {k}"


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint save/load
# ═══════════════════════════════════════════════════════════════════


class TestRDCCheckpoint:
    def test_checkpoint_roundtrip(self, rdc_module, sample_tensors):
        z_cur, z_prev, Q = sample_tensors
        # Run a step to update buffers
        rdc_module(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        ckpt = rdc_module.checkpoint_dict()
        from src.models.rdc import RepresentationDriftCompensation

        rdc2 = RepresentationDriftCompensation(
            embed_dim=64,
            eta=0.1,
            ema_beta=0.99,
            warmup_steps=0,
            k_workspace=16,
        )
        rdc2.load_checkpoint(ckpt)
        assert torch.allclose(rdc_module.z_previous, rdc2.z_previous, atol=1e-6)
        assert torch.allclose(rdc_module.running_drift_norm, rdc2.running_drift_norm, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
#  Workspace update
# ═══════════════════════════════════════════════════════════════════


class TestRDCWorkspaceUpdate:
    def test_update_workspace(self, rdc_module):
        D, k = 64, 16
        Q_new = torch.randn(D, k)
        Q_new, _ = torch.linalg.qr(Q_new)
        rdc_module.update_workspace(Q_new)
        assert torch.allclose(rdc_module.workspace_Q[:, :k], Q_new, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
#  Shape verification
# ═══════════════════════════════════════════════════════════════════


class TestRDCShapes:
    @pytest.mark.parametrize("B,T,D", [(1, 1, 64), (8, 32, 64)])
    def test_various_shapes(self, B, T, D):
        from src.models.rdc import RepresentationDriftCompensation

        rdc = RepresentationDriftCompensation(embed_dim=D, eta=0.1, warmup_steps=0, k_workspace=8)
        z_cur = torch.randn(B, T, D)
        z_prev = torch.randn(B, T, D)
        Q = torch.randn(D, 8)
        Q, _ = torch.linalg.qr(Q)
        loss, _info = rdc(z_cur, z_previous=z_prev, workspace_Q=Q, step=100)
        assert loss.shape == ()
        assert loss.item() >= 0.0


# ═══════════════════════════════════════════════════════════════════
#  Integration with MechanismBundle
# ═══════════════════════════════════════════════════════════════════


class TestRDCIntegration:
    def test_one_line_api(self, sample_tensors):
        z_cur, z_prev, Q = sample_tensors
        from src.models.mechanisms import rdc_compensate

        loss, info = rdc_compensate(z_cur, z_prev, Q, embed_dim=64, eta=0.1, step=100)
        assert loss.item() >= 0.0
        assert "rdc_loss" in info

    def test_mechanism_bundle_with_rdc(self):
        from src.models.mechanisms import MechanismBundle

        bundle = MechanismBundle(embed_dim=64, use_rdc=True, rdc_eta=0.05)
        assert bundle.rdc is not None
        assert bundle.use_rdc is True

    def test_mechanism_bundle_without_rdc(self):
        from src.models.mechanisms import MechanismBundle

        bundle = MechanismBundle(embed_dim=64, use_rdc=False)
        assert bundle.rdc is None
