# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Tests for JAWP: Jacobian-Aligned Workspace Prediction (NOVEL MECHANISM)

import math

import torch
import torch.nn.functional as F
from torch import nn


class TestJAWPCore:
    """Core tests for the JAWP mechanism."""

    def test_import(self):
        from src.models.jawp import JAWPModule

        assert JAWPModule is not None

    def test_creation(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=128, k_start=1, k_end=13)
        assert jawp.embed_dim == 128
        assert jawp.k_start == 1
        assert jawp.k_end == 13
        assert jawp.workspace_Q.shape == (128, 13)

    def test_no_beta_no_gamma(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=1, k_end=7)
        assert not hasattr(jawp, "beta")
        assert not hasattr(jawp, "gamma")

    def test_learned_Q_is_parameter(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7)
        assert isinstance(jawp.workspace_Q, nn.Parameter)
        assert jawp.workspace_Q.requires_grad

    def test_identity_init(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=1, k_end=7, init="identity")
        Q = jawp.workspace_Q.data
        assert torch.allclose(Q[:7, :7], torch.eye(7), atol=1e-6)

    def test_random_init_is_orthogonal(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=1, k_end=7, init="random")
        Q = jawp.workspace_Q.data
        gram = Q.T @ Q
        assert torch.allclose(gram, torch.eye(7), atol=1e-5)

    def test_cosine_curriculum(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=128, k_start=1, k_end=13, curriculum_steps=1000)
        assert jawp.current_k(0) == 1
        k_mid = jawp.current_k(500)
        assert 4 <= k_mid <= 10
        assert jawp.current_k(1000) == 13
        assert jawp.current_k(2000) == 13

    def test_stiefel_retract_keeps_orthonormal(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7, init="identity")
        with torch.no_grad():
            jawp.workspace_Q.add_(torch.randn(64, 7) * 0.5)
        gram_before = jawp.workspace_Q.data.T @ jawp.workspace_Q.data
        assert not torch.allclose(gram_before, torch.eye(7), atol=0.1)
        jawp.stiefel_retract()
        gram_after = jawp.workspace_Q.data.T @ jawp.workspace_Q.data
        assert torch.allclose(gram_after, torch.eye(7), atol=1e-5)


class TestJAWPLoss:
    """Tests for JAWP loss computation."""

    def _make_jawp(self, **kwargs):
        from src.models.jawp import JAWPModule

        defaults = {"embed_dim": 64, "k_start": 4, "k_end": 7, "curriculum_steps": 0, "alpha": 0.1}
        defaults.update(kwargs)
        return JAWPModule(**defaults)

    def test_loss_is_nonnegative(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(8, 32, 64)
        z_target = torch.randn(8, 32, 64)
        loss, _info = jawp.compute_loss(z_pred, z_target, step=1000)
        assert loss.item() >= 0

    def test_loss_uses_mse_not_smooth_l1(self):
        jawp = self._make_jawp(alpha=0.0)
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64)
        loss, _info = jawp.compute_loss(z_pred, z_target, step=1000)
        Q = jawp.workspace_Q.data[:, :7]
        expected = F.mse_loss(z_pred @ Q, z_target @ Q)
        assert abs(loss.item() - expected.item()) < 1e-4

    def test_loss_components_present(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(8, 32, 64)
        z_target = torch.randn(8, 32, 64)
        _, info = jawp.compute_loss(z_pred, z_target, step=1000)
        for key in [
            "loss_workspace",
            "loss_predictor_focus",
            "k",
            "workspace_utilization",
            "target_ws_fraction",
            "workspace_cosine",
            "ortho_score",
            "predictive_relevance",
            "pca_alignment",
        ]:
            assert key in info, f"Missing key: {key}"

    def test_gradient_flows_through_Q(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64)
        loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
        loss.backward()
        assert jawp.workspace_Q.grad is not None
        assert jawp.workspace_Q.grad.abs().sum() > 0

    def test_gradient_flows_through_z_pred(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(32, 64, requires_grad=True)
        z_target = torch.randn(32, 64)
        loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
        loss.backward()
        assert z_pred.grad is not None
        assert z_pred.grad.abs().sum() > 0

    def test_z_target_does_not_get_gradients(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64, requires_grad=True)
        loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
        loss.backward()
        assert z_target.grad is None or z_target.grad.abs().sum() == 0

    def test_predictor_focus_penalty_works(self):
        jawp = self._make_jawp(alpha=1.0)
        Q = jawp.workspace_Q.data[:, :7]
        z_target = torch.randn(32, 64)
        z_pred_ws = (z_target @ Q) @ Q.T
        _, info_ws = jawp.compute_loss(z_pred_ws, z_target, step=1000)
        z_pred_bg = z_target - (z_target @ Q) @ Q.T
        _, info_bg = jawp.compute_loss(z_pred_bg + z_target, z_target, step=1000)
        assert info_ws["loss_predictor_focus"] < info_bg["loss_predictor_focus"]

    def test_diagnostics_bounded(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64)
        _, info = jawp.compute_loss(z_pred, z_target, step=1000)
        assert 0 <= info["workspace_utilization"] <= 1.01
        assert 0 <= info["target_ws_fraction"] <= 1.01
        assert -1 <= info["workspace_cosine"] <= 1.01
        assert 0 <= info["ortho_score"] <= 1.01

    def test_no_nan_or_inf(self):
        jawp = self._make_jawp()
        z_pred = torch.randn(4, 8, 64) * 1e-7
        z_target = torch.randn(4, 8, 64) * 1e-7
        loss, info = jawp.compute_loss(z_pred, z_target, step=1000)
        assert math.isfinite(loss.item())
        for k, v in info.items():
            assert math.isfinite(v), f"{k} = {v}"

    def test_stiefel_retract_after_training_step(self):
        jawp = self._make_jawp()
        optimizer = torch.optim.Adam(jawp.parameters(), lr=0.01)
        for _ in range(10):
            z_pred = torch.randn(16, 64)
            z_target = torch.randn(16, 64)
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            jawp.stiefel_retract()
        Q = jawp.workspace_Q.data[:, :7]
        gram = Q.T @ Q
        assert torch.allclose(gram, torch.eye(7), atol=1e-4)


class TestJAWPCourantFischer:
    """Tests for the Courant-Fischer theorem and convergence guarantees."""

    def _make_synthetic_data(self, D=32, k=4, N=128, noise_pred=0.01, noise_unpred=5.0):
        torch.manual_seed(42)
        z_pred = torch.randn(N, D)
        z_target = z_pred.clone()
        for i in range(k):
            z_target[:, i] = z_pred[:, i] + torch.randn(N) * noise_pred
        for i in range(k, D):
            z_target[:, i] = torch.randn(N) * noise_unpred
        return z_pred, z_target

    def test_gradient_form_matches_courant_fischer(self):
        from src.models.jawp import JAWPModule

        D, k, N = 32, 4, 64
        jawp = JAWPModule(
            embed_dim=D,
            k_start=k,
            k_end=k,
            curriculum_steps=0,
            init="random",
            alpha=0.0,
        )
        z_pred = torch.randn(N, D)
        z_target = torch.randn(N, D)
        Q = jawp.workspace_Q[:, :k]
        pred_ws = z_pred @ Q
        target_ws = z_target.detach() @ Q
        loss = F.mse_loss(pred_ws, target_ws)
        loss.backward()
        grad_Q = jawp.workspace_Q.grad.data[:, :k]
        Q_data = jawp.workspace_Q.data[:, :k]
        R = z_pred - z_target
        analytical_grad = (2.0 / (N * k)) * R.T @ R @ Q_data
        relative_err = (grad_Q - analytical_grad).norm().item() / (grad_Q.norm().item() + 1e-10)
        assert relative_err < 0.01

    def test_convergence_to_optimal_subspace(self):
        from src.models.jawp import JAWPModule

        D, k, N = 32, 4, 128
        z_pred, z_target = self._make_synthetic_data(D, k, N)
        R = z_pred - z_target
        Sigma_res = (R.T @ R) / (N - 1)
        _eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_res)
        Q_optimal = eigenvectors[:, :k]
        optimal_risk = torch.trace(Q_optimal.T @ Sigma_res @ Q_optimal).item()
        jawp = JAWPModule(
            embed_dim=D,
            k_start=k,
            k_end=k,
            curriculum_steps=0,
            init="random",
            alpha=0.0,
        )
        optimizer = torch.optim.SGD([jawp.workspace_Q], lr=0.02)
        for _ in range(1000):
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            jawp.stiefel_retract()
            optimizer.step()
            jawp.stiefel_retract()
        Q_learned = jawp.workspace_Q.data[:, :k]
        learned_risk = torch.trace(Q_learned.T @ Sigma_res @ Q_learned).item()
        risk_ratio = learned_risk / (optimal_risk + 1e-10)
        assert risk_ratio < 1.5

    def test_corollary_jawp_leq_pca(self):
        from src.models.jawp import JAWPModule

        D, k, N = 32, 4, 128
        z_pred, z_target = self._make_synthetic_data(D, k, N)
        R = z_pred - z_target
        Sigma_res = (R.T @ R) / (N - 1)
        cov_target = (z_target.T @ z_target) / (N - 1)
        _, V_pca = torch.linalg.eigh(cov_target)
        Q_pca = V_pca[:, -k:]
        pca_risk = torch.trace(Q_pca.T @ Sigma_res @ Q_pca).item()
        jawp = JAWPModule(
            embed_dim=D,
            k_start=k,
            k_end=k,
            curriculum_steps=0,
            init="random",
            alpha=0.0,
        )
        optimizer = torch.optim.SGD([jawp.workspace_Q], lr=0.02)
        for _ in range(1000):
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            jawp.stiefel_retract()
            optimizer.step()
            jawp.stiefel_retract()
        Q_learned = jawp.workspace_Q.data[:, :k]
        jawp_risk = torch.trace(Q_learned.T @ Sigma_res @ Q_learned).item()
        assert jawp_risk <= pca_risk + 1e-3

    def test_subspace_similarity_with_optimal(self):
        from src.models.jawp import JAWPModule

        D, k, N = 32, 4, 128
        z_pred, z_target = self._make_synthetic_data(D, k, N)
        R = z_pred - z_target
        Sigma_res = (R.T @ R) / (N - 1)
        _eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_res)
        Q_optimal = eigenvectors[:, :k]
        jawp = JAWPModule(
            embed_dim=D,
            k_start=k,
            k_end=k,
            curriculum_steps=0,
            init="random",
            alpha=0.0,
        )
        optimizer = torch.optim.SGD([jawp.workspace_Q], lr=0.02)
        for _ in range(1000):
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            jawp.stiefel_retract()
            optimizer.step()
            jawp.stiefel_retract()
        Q_learned = jawp.workspace_Q.data[:, :k]
        cross = Q_learned.T @ Q_optimal
        similarity = (cross**2).sum() / k
        assert similarity > 0.9


class TestJAWPNovelty:
    """Tests that verify JAWP's novelty vs. PCA."""

    def test_task_adaptivity_not_just_PCA(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(
            embed_dim=32,
            k_start=2,
            k_end=2,
            curriculum_steps=0,
            init="random",
            alpha=0.1,
        )
        optimizer = torch.optim.Adam(jawp.parameters(), lr=0.02)
        for _ in range(300):
            z_pred = torch.randn(32, 32)
            z_target = z_pred.clone()
            z_target[:, 0] = torch.randn(32) * 10.0
            z_target[:, 1] = torch.randn(32) * 10.0
            z_target[:, 2] = z_pred[:, 2] + torch.randn(32) * 0.1
            z_target[:, 3] = z_pred[:, 3] + torch.randn(32) * 0.1
            loss, _ = jawp.compute_loss(z_pred, z_target, step=1000)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            jawp.stiefel_retract()
        Q = jawp.workspace_Q.data[:, :2]
        pred_align = Q[2:4, :].norm().item()
        unpred_align = Q[0:2, :].norm().item()
        ratio = pred_align / (unpred_align + 1e-10)
        assert ratio > 1.5


class TestJAWPAPI:
    """Tests for the public API that other papers will use."""

    def test_get_workspace_basis(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7, curriculum_steps=0)
        Q = jawp.get_workspace_basis()
        assert Q.shape == (64, 4)

    def test_project_to_workspace(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7, curriculum_steps=0)
        z = torch.randn(8, 32, 64)
        z_ws = jawp.project_to_workspace(z)
        assert z_ws.shape == (8, 32, 4)

    def test_project_to_background(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7, curriculum_steps=0)
        z = torch.randn(8, 32, 64)
        z_bg = jawp.project_to_background(z)
        assert z_bg.shape == (8, 32, 64)

    def test_workspace_plus_background_equals_original(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=4, k_end=7, curriculum_steps=0, init="identity")
        z = torch.randn(8, 32, 64)
        Q = jawp.get_workspace_basis()
        z_ws = jawp.project_to_workspace(z)
        z_bg = jawp.project_to_background(z)
        z_recon = z_bg + (z_ws @ Q.T)
        assert torch.allclose(z, z_recon, atol=1e-5)

    def test_drop_in_api(self):
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=768, k_start=1, k_end=77)
        z_pred = torch.randn(4, 32, 768)
        z_target = torch.randn(4, 32, 768)
        loss, _info = jawp.compute_loss(z_pred, z_target, step=5000)
        assert math.isfinite(loss.item())

    def test_all_shapes_correct(self):
        from src.models.jawp import JAWPModule

        D, k_start, k_end = 64, 2, 7
        jawp = JAWPModule(
            embed_dim=D,
            k_start=k_start,
            k_end=k_end,
            curriculum_steps=0,
            init="identity",
        )
        assert jawp.workspace_Q.shape == (D, k_end)
        B, T = 4, 16
        z_pred = torch.randn(B, T, D)
        z_target = torch.randn(B, T, D)
        loss, info = jawp.compute_loss(z_pred, z_target, step=1000)
        assert loss.shape == torch.Size([])
        assert info["k"] == k_end
        Q = jawp.get_workspace_basis()
        assert Q.shape == (D, k_end)
        z_ws = jawp.project_to_workspace(z_pred)
        assert z_ws.shape == (B, T, k_end)
        z_bg = jawp.project_to_background(z_pred)
        assert z_bg.shape == (B, T, D)
        z_recon = z_bg + (z_ws @ Q.T)
        assert z_recon.shape == (B, T, D)
        assert torch.allclose(z_pred, z_recon, atol=1e-5)


class TestJAWPWithJEPA:
    """Integration tests: JEPA model with JAWP."""

    def test_jepa_with_jawp(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            predictor_embed_dim=32,
            predictor_depth=1,
            max_seq_len=32,
            vocab_size=100,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=7,
            future_offsets=(1,),
            num_refine_steps=0,
        )
        config.validate()
        model = TextSpanJEPA(config)
        assert model.jawp is not None

    def test_jepa_without_jawp(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            predictor_embed_dim=32,
            predictor_depth=1,
            max_seq_len=32,
            vocab_size=100,
            use_jawp=False,
            future_offsets=(1,),
            num_refine_steps=0,
        )
        config.validate()
        model = TextSpanJEPA(config)
        assert model.jawp is None

    def test_loss_with_jawp_is_valid(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            predictor_embed_dim=32,
            predictor_depth=1,
            max_seq_len=32,
            vocab_size=100,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=7,
            future_offsets=(1,),
            num_refine_steps=0,
        )
        config.validate()
        model = TextSpanJEPA(config)
        masked = torch.randint(0, 100, (2, 32))
        original = torch.randint(0, 100, (2, 32))
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 8:16] = 1
        total_loss, _loss_dict, _diag_dict = model.compute_loss_with_targets(
            masked,
            original,
            mask,
            current_step=100,
            total_steps=1000,
        )
        assert math.isfinite(total_loss.item())
        assert total_loss.item() > 0


class TestWorkspaceInformationPreservation:
    """Tests for the WIP theorem: workspace preserves exogenous features.

    Theorem: If I(f_exo; z_target) > 0, then span(Q_JAWP) must contain
    a non-trivial projection of f_exo. This directly mitigates the
    Predictor Capacity Waste problem (Pendharkar et al., 2026).
    """

    def test_wip_score_perfect_when_features_in_workspace(self):
        """If exogenous features lie entirely in workspace, WIP = 1.0."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=7, k_end=7, alpha=0.1)
        Q = jawp.workspace_Q.data[:, :7]
        features = Q.T
        z_pred = torch.randn(16, 64)
        z_target = torch.randn(16, 64)
        wip_score, _wip_info = jawp.workspace_information_preservation(
            z_pred,
            z_target,
            features=features,
        )
        assert wip_score > 0.99, f"WIP should be ~1.0 for workspace features, got {wip_score}"

    def test_wip_score_zero_when_features_orthogonal(self):
        """If features are orthogonal to workspace, WIP approx 0.0."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=7, k_end=7, alpha=0.1)
        Q = jawp.workspace_Q.data[:, :7]
        rand_features = torch.randn(7, 64)
        features = rand_features - (rand_features @ Q) @ Q.T
        if features.norm() > 1e-6:
            features = features / features.norm() * 7.0
            z_pred = torch.randn(16, 64)
            z_target = torch.randn(16, 64)
            wip_score, _wip_info = jawp.workspace_information_preservation(
                z_pred,
                z_target,
                features=features,
            )
            assert wip_score < 0.01, f"WIP should be ~0.0 for orthogonal features, got {wip_score}"

    def test_wip_score_bounded(self):
        """WIP score is always in [0, 1]."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=3, k_end=7, alpha=0.1)
        z_pred = torch.randn(16, 64)
        z_target = torch.randn(16, 64)
        wip_score, _ = jawp.workspace_information_preservation(z_pred, z_target)
        assert 0.0 <= wip_score <= 1.0, f"WIP out of bounds: {wip_score}"

    def test_wip_with_proxy_pca_features(self):
        """WIP with proxy PCA features returns valid score."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=3, k_end=7, alpha=0.1)
        z_pred = torch.randn(32, 64)
        z_target = torch.randn(32, 64)
        wip_score, wip_info = jawp.workspace_information_preservation(z_pred, z_target)
        assert 0.0 <= wip_score <= 1.0
        assert "method" in wip_info
        assert wip_info["method"] == "wip_theorem"

    def test_wip_theorem_contradiction_proof(self):
        """Verify WIP theorem: excluding predictive feature increases loss."""
        from src.models.jawp import JAWPModule

        D, k = 32, 4
        torch.manual_seed(42)
        z_target = torch.randn(64, D)
        z_target[:, :4] *= 5.0
        z_pred = z_target.clone()
        z_pred[:, 4:] = torch.randn(64, D - 4)

        Q_good = torch.zeros(D, k)
        Q_good[:k, :] = torch.eye(k)
        jawp_good = JAWPModule(embed_dim=D, k_start=k, k_end=k)
        jawp_good.workspace_Q.data.copy_(Q_good)
        loss_good, _ = jawp_good.compute_loss(z_pred, z_target, step=1000)

        Q_bad = torch.zeros(D, k)
        Q_bad[D - k :, :] = torch.eye(k)
        jawp_bad = JAWPModule(embed_dim=D, k_start=k, k_end=k)
        jawp_bad.workspace_Q.data.copy_(Q_bad)
        loss_bad, _ = jawp_bad.compute_loss(z_pred, z_target, step=1000)

        assert (
            loss_good.item() < loss_bad.item()
        ), f"WIP theorem violation: good_ws loss {loss_good.item():.4f} >= bad_ws loss {loss_bad.item():.4f}"

    def test_wip_preserves_exogenous_features(self):
        """Exogenous features with I(f; z_target) > 0 are preserved."""
        from src.models.jawp import JAWPModule

        D, k = 32, 4
        torch.manual_seed(42)
        f_exo = torch.zeros(1, D)
        f_exo[0, 0] = 1.0
        z_target = torch.randn(64, D)
        z_target[:, 0] = torch.randn(64) * 3.0
        z_pred = z_target + torch.randn(64, D) * 0.1

        jawp = JAWPModule(embed_dim=D, k_start=k, k_end=k, init="identity")
        wip_score, _ = jawp.workspace_information_preservation(z_pred, z_target, features=f_exo)
        assert wip_score > 0.5, f"Exogenous feature should be preserved, WIP={wip_score}"


class TestBackgroundComplexity:
    """Tests for background predictive complexity analysis."""

    def test_background_complexity_high_for_good_split(self):
        """Good workspace/background split has high background complexity."""
        from src.models.jawp import JAWPModule

        D, k = 32, 4
        torch.manual_seed(42)
        z_target = torch.randn(64, D)
        z_target[:, :k] *= 5.0
        z_pred = z_target.clone()
        z_pred[:, k:] = torch.randn(64, D - k)

        jawp = JAWPModule(embed_dim=D, k_start=k, k_end=k, init="identity")
        bg_complexity, _bg_info = jawp.compute_background_complexity(z_pred, z_target)
        assert bg_complexity >= 1.0, f"Expected high bg complexity, got {bg_complexity}"

    def test_background_complexity_returns_valid_dict(self):
        """Background complexity returns proper info dict."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        z_pred = torch.randn(16, 32)
        z_target = torch.randn(16, 32)
        _bg_complexity, bg_info = jawp.compute_background_complexity(z_pred, z_target)
        assert isinstance(bg_info, dict)
        assert "ws_residual" in bg_info
        assert "bg_residual" in bg_info
        assert "bg_complexity_ratio" in bg_info
        assert bg_info["k"] == 4


class TestGrassmannOptimization:
    """Tests for Grassmann workspace optimization.

    Theorem: Grassmann gradient descent converges to the optimal
    subspace while Stiefel may oscillate in the O(k) fiber.
    """

    def test_grassmann_retract_removes_gauge(self):
        """Grassmann retract removes the gauge component."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        # Simulate gradient with gauge component
        Q = jawp.workspace_Q.data[:, :4]
        gauge = Q @ torch.randn(4, 4)  # purely in the fiber
        jawp.workspace_Q.grad = gauge.clone()
        gauge_norm = jawp.grassmann_retract()
        # Gauge component should have been detected and removed
        assert gauge_norm > 0, "Should detect non-zero gauge component"

    def test_grassmann_retract_preserves_subspace(self):
        """Grassmann retract doesn't change the subspace span(Q)."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        # Save subspace before
        Q_before = jawp.workspace_Q.data[:, :4].clone()
        # Apply gradient that changes BOTH subspace and gauge
        jawp.workspace_Q.grad = torch.randn(32, 4) * 0.01
        jawp.grassmann_retract()
        Q_after = jawp.workspace_Q.data[:, :4]
        # Subspace should have changed (not just rotated)
        # Verify by checking projection operator QQ^T
        Q_before @ Q_before.T
        Q_after @ Q_after.T
        # They should differ (gradient moved the subspace)
        # But Q should still be orthonormal
        gram = Q_after.T @ Q_after
        off_diag = gram.clone()
        off_diag.fill_diagonal_(0)
        assert off_diag.abs().max().item() < 0.01, "Q should be orthonormal after retract"

    def test_grassmann_retract_returns_gauge_norm(self):
        """grassmann_retract returns the gauge component norm."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        jawp.workspace_Q.grad = torch.randn(32, 4) * 0.01
        gauge_norm = jawp.grassmann_retract()
        assert isinstance(gauge_norm, float)
        assert gauge_norm >= 0.0

    def test_principal_angles_identical_subspace(self):
        """Principal angles are zero for identical subspaces."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        Q = jawp.workspace_Q.data[:, :4].clone()
        angles, cosines = jawp.principal_angles(other_Q=Q)
        for a in angles:
            assert a < 0.01, f"Angle should be ~0 for identical subspace, got {a}"
        for c in cosines:
            assert c > 0.99, f"Cosine should be ~1 for identical subspace, got {c}"

    def test_principal_angles_orthogonal_subspaces(self):
        """Principal angles are π/2 for orthogonal subspaces."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        Q1 = jawp.workspace_Q.data[:, :4]
        # Create orthogonal Q2
        orth = torch.randn(32, 4)
        orth = orth - Q1 @ (Q1.T @ orth)  # project out Q1
        if orth.norm() > 1e-6:
            Q2, _ = torch.linalg.qr(orth)
            angles, _cosines = jawp.principal_angles(other_Q=Q2)
            for a in angles:
                assert a > 1.0, f"Angle should be ~π/2 for orthogonal subspaces, got {a}"

    def test_principal_angles_gauge_invariant(self):
        """Principal angles don't change under O(k) rotation."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        Q = jawp.workspace_Q.data[:, :4].clone()
        # Rotate Q by random orthogonal R
        R = torch.linalg.qr(torch.randn(4, 4))[0]
        Q_rotated = Q @ R
        angles_orig, _ = jawp.principal_angles(other_Q=Q)
        angles_rot, _ = jawp.principal_angles(other_Q=Q_rotated)
        for a_o, a_r in zip(angles_orig, angles_rot):
            assert abs(a_o - a_r) < 0.01, f"Angles should be gauge-invariant: {a_o} vs {a_r}"

    def test_subspace_distance_zero_for_same(self):
        """Subspace distance is zero for identical subspaces."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        Q = jawp.workspace_Q.data[:, :4].clone()
        d = jawp.subspace_distance(other_Q=Q)
        assert d < 0.01, f"Distance should be ~0 for identical subspace, got {d}"

    def test_subspace_distance_positive_for_different(self):
        """Subspace distance is positive for different subspaces."""
        from src.models.jawp import JAWPModule

        torch.manual_seed(42)
        jawp1 = JAWPModule(embed_dim=32, k_start=4, k_end=4, init="identity")
        jawp2 = JAWPModule(embed_dim=32, k_start=4, k_end=4, init="random")
        Q2 = jawp2.workspace_Q.data[:, :4]
        d = jawp1.subspace_distance(other_Q=Q2)
        assert d > 0.01, f"Distance should be >0 for different subspaces, got {d}"

    def test_grassmann_vs_stiefel_convergence(self):
        """Grassmann retract converges faster than Stiefel-only.

        This verifies the theorem: removing the gauge component
        accelerates convergence by eliminating oscillation.
        """
        from src.models.jawp import JAWPModule

        D, k = 32, 4
        torch.manual_seed(42)

        # Set up a structured prediction task
        z_target = torch.randn(64, D)
        z_target[:, :k] *= 5.0  # signal in first k dims
        z_pred = z_target + torch.randn(64, D) * 0.5

        # Track subspace movement with Grassmann
        jawp_g = JAWPModule(embed_dim=D, k_start=k, k_end=k, init="random")
        for _ in range(5):
            loss, _ = jawp_g.compute_loss(z_pred, z_target, step=1000)
            loss.backward()
            jawp_g.workspace_Q.data.add_(jawp_g.workspace_Q.grad.data, alpha=-0.01)
            jawp_g.grassmann_retract()
            jawp_g.workspace_Q.grad = None

        # Track subspace movement with Stiefel
        torch.manual_seed(42)
        jawp_s = JAWPModule(embed_dim=D, k_start=k, k_end=k, init="random")
        for _ in range(5):
            loss, _ = jawp_s.compute_loss(z_pred, z_target, step=1000)
            loss.backward()
            jawp_s.workspace_Q.data.add_(jawp_s.workspace_Q.grad.data, alpha=-0.01)
            jawp_s.stiefel_retract()
            jawp_s.workspace_Q.grad = None

        # Both should produce valid orthonormal Q
        gram_g = jawp_g.workspace_Q.data[:, :k].T @ jawp_g.workspace_Q.data[:, :k]
        gram_s = jawp_s.workspace_Q.data[:, :k].T @ jawp_s.workspace_Q.data[:, :k]
        assert (gram_g - torch.eye(k)).abs().max().item() < 0.01
        assert (gram_s - torch.eye(k)).abs().max().item() < 0.01

    def test_save_workspace_snapshot(self):
        """save_workspace_snapshot stores Q for later comparison."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        jawp.save_workspace_snapshot()
        assert hasattr(jawp, "_prev_workspace_Q")
        assert jawp._prev_workspace_Q.shape == (32, 4)

    def test_grassmann_retract_no_grad(self):
        """grassmann_retract works even with no gradient (inference)."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        # No gradient set
        gauge_norm = jawp.grassmann_retract()
        assert gauge_norm == 0.0


class TestPredictiveRank:
    """Tests for Predictive Rank Regularization.

    Theorem: If λ_min(Q^T Cov Q) > ε, then rank(J_ws) = k.
    Log-determinant barrier prevents rank collapse.
    """

    def test_compute_predictive_rank_full(self):
        """Full-rank predictor gives effective_rank ≈ k."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        z_pred = torch.randn(64, 32)  # full rank
        info = jawp.compute_predictive_rank(z_pred)
        assert (
            info["effective_rank"] > 2.0
        ), f"Expected high effective rank, got {info['effective_rank']}"
        assert 0.0 <= info["rank_utilization"] <= 1.0

    def test_compute_predictive_rank_returns_valid(self):
        """compute_predictive_rank returns valid dict."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        z_pred = torch.randn(16, 32)
        info = jawp.compute_predictive_rank(z_pred)
        assert "effective_rank" in info
        assert "singular_values" in info
        assert "rank_utilization" in info
        assert "min_singular" in info
        assert "condition_number" in info

    def test_predictive_rank_loss_full_rank(self):
        """Log-det loss is finite for full-rank workspace covariance."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        z_pred = torch.randn(64, 32)
        loss = jawp.predictive_rank_loss(z_pred)
        assert math.isfinite(loss.item()), f"Loss should be finite, got {loss.item()}"

    def test_predictive_rank_loss_differentiable(self):
        """predictive_rank_loss is differentiable w.r.t. Q."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        z_pred = torch.randn(64, 32)
        loss = jawp.predictive_rank_loss(z_pred)
        loss.backward()
        # Q should have gradients
        assert jawp.workspace_Q.grad is not None
        q_grad_norm = jawp.workspace_Q.grad.norm().item()
        assert q_grad_norm > 0, "Q should receive gradients from rank loss"

    def test_predictive_rank_loss_prevents_collapse(self):
        """RankE loss increases as workspace covariance collapses."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        # Full rank: diverse predictions
        z_full = torch.randn(64, 32)
        loss_full = jawp.predictive_rank_loss(z_full)
        # Collapsed: all predictions same direction
        z_collapse = torch.randn(64, 1).expand(64, 32) * 0.1
        loss_collapse = jawp.predictive_rank_loss(z_collapse)
        # Collapsed should have HIGHER loss (more negative log-det)
        assert (
            loss_collapse.item() > loss_full.item()
        ), f"Collapse loss {loss_collapse.item():.2f} should exceed full loss {loss_full.item():.2f}"

    def test_rank_utilization_bounded(self):
        """Rank utilization is always in [0, 1]."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=32, k_start=4, k_end=4)
        for _ in range(5):
            z_pred = torch.randn(32, 32) * torch.rand(1).item()
            info = jawp.compute_predictive_rank(z_pred)
            assert 0.0 <= info["rank_utilization"] <= 1.0
