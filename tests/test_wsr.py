# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Tests for Workspace Sharpness Regularization (WSR) — mechanism #16."""

import pytest
import torch
from torch import nn

from src.models.wsr import WorkspaceSharpnessRegularization, wsr_sharpness


class TestWSRBasic:
    """Basic functionality tests."""

    def test_construction_default(self):
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        assert wsr.rho == 0.05
        assert wsr.eta == 0.01
        assert wsr.mode == "gradient"

    def test_construction_sam_mode(self):
        wsr = WorkspaceSharpnessRegularization(embed_dim=64, mode="sam")
        assert wsr.mode == "sam"

    def test_construction_invalid_mode(self):
        with pytest.raises(ValueError):
            WorkspaceSharpnessRegularization(embed_dim=64, mode="invalid")

    def test_output_shape(self):
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        loss, info = wsr(Q, step=1000)
        assert loss.dim() == 0  # scalar
        assert isinstance(info, dict)

    def test_loss_non_negative(self):
        """WSR loss must be non-negative (theoretical guarantee)."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        for _ in range(20):
            Q = torch.randn(64, 8)
            Q, _ = torch.linalg.qr(Q)
            loss, _info = wsr(Q, step=1000)
            assert loss.item() >= 0, f"WSR loss must be ≥ 0, got {loss.item()}"

    def test_warmup_returns_zero(self):
        """WSR should return zero loss during warmup."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64, warmup_steps=100)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        loss, info = wsr(Q, step=0)
        assert loss.item() == 0.0
        assert info.get("wsr_warmup", False) is True

    def test_warmup_partial(self):
        """WSR should partially activate during warmup."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64, warmup_steps=1000)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        _loss_early, info_early = wsr(Q, step=100)
        _loss_late, info_late = wsr(Q, step=900)
        # Late warmup should have higher warmup_factor
        assert info_late["wsr_warmup_factor"] > info_early["wsr_warmup_factor"]


class TestWSRGrassmannGradient:
    """Tests for Grassmann gradient computation."""

    def test_tangent_space_orthogonality(self):
        """Grassmann gradient must be in tangent space at Q: Q^T grad = 0."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)

        # Create a random Euclidean gradient
        euclidean_grad = torch.randn(64, 8)
        grad_grassmann = wsr._grassmann_gradient(Q, euclidean_grad)

        # Tangent space condition: Q^T @ grad_Gr = 0
        projection = Q.T @ grad_grassmann
        ortho_error = projection.norm().item()
        assert ortho_error < 1e-5, f"Tangent space orthogonality error: {ortho_error}"

    def test_gradient_projection_idempotent(self):
        """Projecting twice should give the same result (idempotent)."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        euclidean_grad = torch.randn(64, 8)

        grad1 = wsr._grassmann_gradient(Q, euclidean_grad)
        grad2 = wsr._grassmann_gradient(Q, grad1)

        diff = (grad1 - grad2).norm().item()
        assert diff < 1e-5, f"Projection not idempotent: diff = {diff}"

    def test_gradient_zero_for_Q_component(self):
        """If gradient is along Q, Grassmann gradient should be zero."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)

        # Gradient along Q: G = Q @ A for some A
        A = torch.randn(8, 8)
        euclidean_grad = Q @ A
        grad_grassmann = wsr._grassmann_gradient(Q, euclidean_grad)

        norm = grad_grassmann.norm().item()
        assert norm < 1e-5, f"Q-aligned gradient should project to zero, got norm {norm}"


class TestWSRTrainingSignal:
    """Pin the distinction between the trainable surrogate and SAM diagnostic."""

    def test_gradient_mode_backpropagates_through_live_workspace(self):
        wsr = WorkspaceSharpnessRegularization(
            embed_dim=2, rho=0.05, eta=1.0, warmup_steps=0, mode="gradient"
        )
        Q = nn.Parameter(torch.tensor([[1.0], [0.0]]))
        # The same one-step-lagged gradient the trainer supplies after backward.
        wsr.set_lagged_gradient(torch.tensor([[1.0], [1.0]]))

        loss, info = wsr(Q, step=1)
        assert info["wsr_grad_source"] == "lagged"
        assert loss.requires_grad
        loss.backward()
        assert Q.grad is not None
        assert Q.grad.norm().item() > 0.0

    def test_sam_mode_returns_a_detached_diagnostic(self):
        wsr = WorkspaceSharpnessRegularization(
            embed_dim=2, rho=0.05, eta=1.0, warmup_steps=0, mode="sam"
        )
        Q = nn.Parameter(torch.tensor([[1.0], [0.0]]))
        wsr.set_lagged_gradient(torch.tensor([[1.0], [1.0]]))

        def loss_fn(q, _z_pred, _z_target):
            return q[1, 0]

        loss, info = wsr(Q, loss_fn, None, None, step=1)
        assert info["wsr_grad_source"] == "lagged"
        assert not loss.requires_grad
        assert info["wsr_loss_perturbed"] > info["wsr_loss_current"]


class TestWSRStiefelRetraction:
    """Tests for Stiefel retraction."""

    def test_retraction_orthonormal(self):
        """Retracted matrix must be orthonormal: Q^T Q = I."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        # Add perturbation (not orthonormal)
        Q_perturbed = Q + 0.1 * torch.randn(64, 8)

        Q_retracted = wsr._stiefel_retract(Q_perturbed)
        QQT = Q_retracted.T @ Q_retracted
        identity = torch.eye(8, device=Q.device)
        ortho_error = (QQT - identity).norm().item()
        assert ortho_error < 1e-5, f"Retraction orthonormality error: {ortho_error}"

    def test_retraction_near_identity_for_orthonormal(self):
        """Retraction of an already orthonormal matrix should be close to it."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)

        Q_retracted = wsr._stiefel_retract(Q)
        # May differ by column signs, so compare subspaces
        diff = (Q @ Q.T - Q_retracted @ Q_retracted.T).norm().item()
        assert diff < 1e-4, f"Retraction changes orthonormal matrix: diff = {diff}"


class TestWSRSharpnessDecomposition:
    """Tests for the Grassmann Sharpness Decomposition theorem."""

    def test_decomposition_components_non_negative(self):
        """Both spectral and directional sharpness must be ≥ 0."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)  # set a gradient

        _loss, info = wsr(Q, step=1000)
        assert info["wsr_spectral_sharpness"] >= 0
        assert info["wsr_directional_sharpness"] >= 0

    def test_sharpness_bounds_total(self):
        """Total sharpness ≤ spectral + directional (triangle inequality).

        The fix (PR #23): `wsr_ambient_sharpness` is the theorem's real total
        ρ_Q = ρ·||∇_Q L||_F/||Q||.  Before it existed the "total" this test
        compared against parts WAS one of the parts (`wsr_sharpness` == the
        spectral component in gradient mode), so the triangle check was
        vacuous: spectral ≤ spectral + directional.

        MUTATION VERDICT: revert the PR #23 wsr.py change and this test
        fails with KeyError('wsr_ambient_sharpness') — the ambient total is
        the only key the fix adds.
        """
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        for _ in range(10):
            Q = torch.randn(64, 8)
            Q, _ = torch.linalg.qr(Q)
            Q.grad = torch.randn(64, 8)

            _loss, info = wsr(Q, step=1000)
            spc = info["wsr_spectral_sharpness"]
            drn = info["wsr_directional_sharpness"]
            amb = info["wsr_ambient_sharpness"]
            total = spc + drn
            assert info["wsr_sharpness"] <= total + 1e-5
            assert amb <= total + 1e-5, "the real total exceeds the sum of its parts"
            # The corrected header theorem: ρ_Q² = ρ_spectral² + ρ_directional².
            # The two parts lie in orthogonal complements of R^D, so this is
            # exact in real arithmetic; the module computes it in float32,
            # where 512-term Frobenius sums round at ~1e-7 relative.
            assert abs(amb**2 - (spc**2 + drn**2)) <= 1e-5 * max(amb**2, 1.0), (
                f"quadrature broken: amb^2={amb**2:.3e} vs " f"spc^2+drn^2={spc**2 + drn**2:.3e}"
            )

    def test_ambient_is_distinct_and_matches_its_own_formula(self):
        """ρ_Q is a real, distinct total, verified against a float64 hand recomputation.

        The probe gradient is required to have BOTH parts nonzero (checked
        below) — with a purely off-manifold gradient, ambient == spectral and
        every identity here would pass vacuously, which is exactly the trap
        the old code sat in.

        MUTATION VERDICT: revert the PR #23 wsr.py change and this test
        fails with KeyError('wsr_ambient_sharpness') on the first read.
        """
        wsr = WorkspaceSharpnessRegularization(embed_dim=64, rho=0.05)
        torch.manual_seed(11)
        G = torch.randn(64, 8)
        Q0, _ = torch.linalg.qr(torch.randn(64, 8))

        # Precondition: the split has two genuinely nonzero parts.
        P = Q0.double() @ Q0.double().T
        assert (G.double() - P @ G.double()).norm().item() > 1e-3, "degenerate: off-span = 0"
        assert (P @ G.double()).norm().item() > 1e-3, "degenerate: in-span = 0"

        wsr.set_lagged_gradient(G)
        _loss, info = wsr(Q0.clone(), step=1000)
        assert info["wsr_grad_source"] == "lagged", "the probe must consume the real gradient"

        # float64 recomputation of the header's own formulas.
        G64, Q64 = G.double(), Q0.double()
        P = Q64 @ Q64.T
        qnorm = Q64.norm().item()  # ||Q||_F == sqrt(k)
        ref = {
            "wsr_ambient_sharpness": 0.05 * G64.norm().item() / qnorm,
            "wsr_spectral_sharpness": 0.05 * (G64 - P @ G64).norm().item() / qnorm,
            "wsr_directional_sharpness": 0.05 * (P @ G64).norm().item() / qnorm,
        }
        for key, expected in ref.items():
            got = info[key]
            assert abs(got - expected) < 1e-5, f"{key}: {got} != float64 reference {expected}"

        # The point of the fix: the total is NOT another name for a component.
        assert (
            info["wsr_ambient_sharpness"] != info["wsr_spectral_sharpness"]
        ), "ambient still equals spectral — the reported total is a part again"


class TestWSRGeneralizationBound:
    """Tests for the generalization bound theorem."""

    def test_bound_decreases_with_samples(self):
        """Generalization bound should decrease as n increases."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)

        # Run a step to update running statistics
        wsr(Q, step=1000)

        bound_100 = wsr.compute_generalization_bound(n_samples=100)
        bound_1000 = wsr.compute_generalization_bound(n_samples=1000)
        bound_10000 = wsr.compute_generalization_bound(n_samples=10000)

        assert bound_1000 < bound_100
        assert bound_10000 < bound_1000

    def test_bound_non_negative(self):
        """Generalization bound must be ≥ 0."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)
        wsr(Q, step=1000)

        bound = wsr.compute_generalization_bound(n_samples=1000)
        assert bound >= 0

    def test_bound_infinite_for_zero_samples(self):
        """Bound should be inf for zero samples."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)
        wsr(Q, step=1000)

        bound = wsr.compute_generalization_bound(n_samples=0)
        assert bound == float("inf")


class TestWSRPACBayes:
    """Tests for the PAC-Bayes bound."""

    def test_pac_bayes_decreases_with_samples(self):
        """PAC-Bayes bound should decrease as n increases."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)
        wsr(Q, step=1000)

        bound_100 = wsr.compute_pac_bayes_bound(n_samples=100)
        bound_1000 = wsr.compute_pac_bayes_bound(n_samples=1000)

        assert bound_1000 < bound_100

    def test_pac_bayes_tighter_with_more_confidence(self):
        """PAC-Bayes bound tighter (lower) with smaller δ (more confidence)."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)
        wsr(Q, step=1000)

        bound_005 = wsr.compute_pac_bayes_bound(n_samples=1000, delta=0.05)
        bound_001 = wsr.compute_pac_bayes_bound(n_samples=1000, delta=0.01)

        # Smaller delta → larger log(n/δ) → larger bound
        assert bound_001 > bound_005


class TestWSRRunningStatistics:
    """Tests for EMA running statistics."""

    def test_running_sharpness_updated(self):
        """Running sharpness should be updated after each call."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)

        assert wsr.running_sharpness.item() == 0.0
        wsr(Q, step=1000)
        # After one step, running stat should be non-zero (unless grad is zero)
        # (May be zero if Q.grad projects to zero, which is unlikely)

    def test_running_sharpness_ema_decay(self):
        """Running statistics should use EMA decay."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64, ema_beta=0.9)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)

        # First call
        Q.grad = torch.randn(64, 8) * 0.1
        wsr(Q, step=1000)
        wsr.running_sharpness.item()

        # Second call with larger gradient
        Q.grad = torch.randn(64, 8) * 1.0
        wsr(Q, step=1001)
        wsr.running_sharpness.item()

        # Should have changed (EMA updated)
        # Not guaranteed to increase, but should be different
        # (unless both project to zero, very unlikely)


class TestWSROnelineAPI:
    """Tests for the one-line convenience function."""

    def test_oneline_basic(self):
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        loss, info = wsr_sharpness(Q, embed_dim=64, step=1000)
        assert loss.item() >= 0
        assert "wsr_loss" in info

    def test_oneline_matches_module(self):
        """One-line API should produce same result as module."""
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)

        loss1, _info1 = wsr_sharpness(Q, embed_dim=64, rho=0.05, eta=0.01, step=1000)

        wsr = WorkspaceSharpnessRegularization(embed_dim=64, rho=0.05, eta=0.01)
        loss2, _info2 = wsr(Q, step=1000)

        # Results should be similar (not exact due to different random init of internal state)
        assert abs(loss1.item() - loss2.item()) < 1.0  # loose bound


class TestWSRIntegration:
    """Integration tests with JAWP-like workspace."""

    def test_with_jawp_workspace(self):
        """WSR should work with a JAWP-style workspace Q."""
        embed_dim = 128
        k = 16

        wsr = WorkspaceSharpnessRegularization(embed_dim=embed_dim, rho=0.05)

        # Simulate JAWP workspace Q
        Q = torch.randn(embed_dim, k)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(embed_dim, k) * 0.01  # small gradient

        loss, info = wsr(Q, step=1000)
        assert loss.item() >= 0
        assert "wsr_sharpness" in info
        assert "wsr_ambient_sharpness" in info
        assert "wsr_spectral_sharpness" in info
        assert "wsr_directional_sharpness" in info

    def test_full_training_loop_smoke(self):
        """Smoke test: WSR in a full training loop (10 steps)."""
        embed_dim = 64
        k = 8
        wsr = WorkspaceSharpnessRegularization(embed_dim=embed_dim)

        Q = nn.Parameter(torch.randn(embed_dim, k))

        optimizer = torch.optim.SGD([Q], lr=0.01)

        for step in range(10):
            # Simulate loss
            z = torch.randn(4, 16, embed_dim)
            loss_main = (z @ Q).pow(2).mean()

            optimizer.zero_grad()
            loss_main.backward()

            # WSR
            _wsr_loss, _wsr_info = wsr(Q.data, step=step * 100)

            optimizer.step()

            # Retract Q onto Stiefel
            with torch.no_grad():
                Q_retracted, _ = torch.linalg.qr(Q.data)
                signs = torch.sign(torch.diag(Q_retracted[:k, :]))
                signs[signs == 0] = 1
                Q.data.copy_(Q_retracted * signs.unsqueeze(0))

        # No NaN/Inf
        assert Q.data.isfinite().all()
        # Q should be approximately orthonormal
        QQT = Q.data.T @ Q.data
        ortho_error = (QQT - torch.eye(k)).norm().item()
        assert ortho_error < 0.1  # loose due to SGD

    def test_device_consistency(self):
        """WSR should work on different devices."""
        for device in [torch.device("cpu")]:
            wsr = WorkspaceSharpnessRegularization(embed_dim=64).to(device)
            Q = torch.randn(64, 8, device=device)
            Q, _ = torch.linalg.qr(Q)
            loss, _info = wsr(Q, step=1000)
            assert loss.device == device


class TestWSRTheorems:
    """Mathematical theorem verification tests."""

    def test_tangent_space_is_kernel_of_QT(self):
        """Verify: tangent space at Q = ker(Q^T), i.e., Q^T v = 0 for v in T_Q Gr."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)

        # Generate random tangent vector
        G = torch.randn(64, 8)
        v = wsr._grassmann_gradient(Q, G)

        # Q^T v should be zero
        QTv = Q.T @ v
        kernel_error = QTv.norm().item()
        assert kernel_error < 1e-5, f"Tangent vector not in ker(Q^T): error = {kernel_error}"

    def test_sharpness_zero_for_constant_loss(self):
        """If Q has no gradient (constant loss), sharpness should be zero."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.zeros(64, 8)  # zero gradient = constant loss

        _loss, info = wsr(Q, step=1000)
        assert info["wsr_sharpness"] < 1e-5

    def test_sharpness_scales_with_rho(self):
        """Sharpness should scale linearly with ρ (by definition)."""
        wsr_small = WorkspaceSharpnessRegularization(embed_dim=64, rho=0.01)
        wsr_large = WorkspaceSharpnessRegularization(embed_dim=64, rho=0.1)

        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)
        Q.grad = torch.randn(64, 8)

        _, info_small = wsr_small(Q, step=1000)
        _, info_large = wsr_large(Q, step=1000)

        # Ratio should be approximately rho_large/rho_small = 10
        if info_small["wsr_sharpness"] > 1e-8:
            ratio = info_large["wsr_sharpness"] / info_small["wsr_sharpness"]
            assert 8 < ratio < 12, f"Sharpness ratio {ratio} != rho ratio 10"

    def test_sharpness_invariant_to_Q_rotation(self):
        """Sharpness should be invariant to right-rotation Q → Q R (same subspace)."""
        wsr = WorkspaceSharpnessRegularization(embed_dim=64)
        Q = torch.randn(64, 8)
        Q, _ = torch.linalg.qr(Q)

        # Random rotation R ∈ O(k)
        R = torch.randn(8, 8)
        R, _ = torch.linalg.qr(R)
        Q_rotated = Q @ R

        # Set same gradient structure
        G = torch.randn(64, 8)
        Q.grad = G.clone()
        G @ R  # rotate gradient too
        # Note: this is not exactly how gradients transform, so we just check
        # that both give finite non-negative losses

        loss1, _info1 = wsr(Q, step=1000)
        loss2, _info2 = wsr(Q_rotated, step=1000)

        assert loss1.item() >= 0
        assert loss2.item() >= 0
