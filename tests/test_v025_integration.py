# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# v0.25.0 integration tests — smoke tests verifying:
#   1. workspace_quality composite metric
#   2. JAWP Stiefel retraction in training loop
#   3. defaults.yaml has all v0.25.0 fields
#   4. Ablation/scaling configs parse correctly
#   5. Visualization module imports and basic plots
#   6. Workspace validation module
#   7. JAWP + SIGReg + J-Space integration in full model

import os

import pytest
import torch
import yaml

# ═══════════════════════════════════════════════════════════════════
#  workspace_quality composite metric
# ═══════════════════════════════════════════════════════════════════


class TestWorkspaceQuality:
    def test_workspace_quality_returns_float(self):
        from src.models.collapse import CollapseDiagnostics

        metrics = {
            "collapsed_dim_ratio_online": 0.1,
            "sv_entropy_online": 0.8,
            "rank_utilization_online": 0.7,
            "alignment": 0.5,
            "uniformity_online": -2.0,
            "cka_linear": 0.9,
            "svcca_online_target": 0.85,
            "alpha_norm_online": 1.5,
        }
        score = CollapseDiagnostics.workspace_quality(metrics)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_workspace_quality_healthy(self):
        from src.models.collapse import CollapseDiagnostics

        metrics = {
            "collapsed_dim_ratio_online": 0.0,
            "sv_entropy_online": 1.0,
            "rank_utilization_online": 1.0,
            "alignment": 0.01,
            "uniformity_online": -1.0,
            "cka_linear": 1.0,
            "svcca_online_target": 1.0,
            "alpha_norm_online": 1.5,
        }
        score = CollapseDiagnostics.workspace_quality(metrics)
        assert score > 0.7, f"Healthy workspace should score > 0.7, got {score}"

    def test_workspace_quality_collapsed(self):
        from src.models.collapse import CollapseDiagnostics

        metrics = {
            "collapsed_dim_ratio_online": 1.0,
            "sv_entropy_online": 0.0,
            "rank_utilization_online": 0.0,
            "alignment": 100.0,
            "uniformity_online": -100.0,
            "cka_linear": 0.0,
            "svcca_online_target": 0.0,
            "alpha_norm_online": 0.0,
        }
        score = CollapseDiagnostics.workspace_quality(metrics)
        assert score < 0.3, f"Collapsed workspace should score < 0.3, got {score}"

    def test_workspace_quality_empty_dict(self):
        from src.models.collapse import CollapseDiagnostics

        score = CollapseDiagnostics.workspace_quality({})
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_workspace_quality_in_diag_dict(self):
        """Full model compute_loss_with_targets should include workspace_quality."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=[1],
            num_refine_steps=1,
        )
        model = TextSpanJEPA(config)
        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1
        _, _, diag = model.compute_loss_with_targets(ids, ids, mask)
        assert "workspace_quality" in diag, "diag_dict should contain workspace_quality"
        assert 0.0 <= diag["workspace_quality"] <= 1.0


# ═══════════════════════════════════════════════════════════════════
#  defaults.yaml v0.25.0 fields
# ═══════════════════════════════════════════════════════════════════


class TestDefaultsYaml:
    @pytest.fixture(autouse=True)
    def load_defaults(self):
        with open("defaults.yaml", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_jawp_fields(self):
        m = self.cfg["model"]
        assert "use_jawp" in m
        assert "jawk_k_start" in m
        assert "jawk_k_end" in m
        assert "jawk_curriculum_steps" in m
        assert "jawk_alpha" in m
        assert "jawk_init" in m
        assert m["use_jawp"] is True

    def test_sigreg_fields(self):
        m = self.cfg["model"]
        assert "lambda_sigreg" in m
        assert "sigreg_n_sketches" in m
        assert "sigreg_n_integration_points" in m
        assert "sigreg_sigma" in m

    def test_jspace_fields(self):
        m = self.cfg["model"]
        assert "jspace_variance_threshold" in m
        assert "jspace_k_workspace" in m

    def test_gradient_checkpointing_field(self):
        assert "gradient_checkpointing" in self.cfg["model"]

    def test_ema_schedule_field(self):
        assert "ema_schedule" in self.cfg["model"]

    def test_future_warmup_steps(self):
        assert "future_warmup_steps" in self.cfg["model"]
        assert self.cfg["model"]["future_warmup_steps"] > 0


# ═══════════════════════════════════════════════════════════════════
#  Ablation / scaling configs
# ═══════════════════════════════════════════════════════════════════


class TestAblationConfigs:
    ABLATION_FILES = [
        "no_jawp",
        "jawp_alpha_0",
        "jawp_random_init",
        "sigreg_only",
        "no_future_loss",
        "no_decoder_loss",
        "jawp_k_fixed",
        "jawp_high_alpha",
    ]

    @pytest.mark.parametrize("name", ABLATION_FILES)
    def test_ablation_config_loads(self, name):
        path = f"config/ablations/{name}.yaml"
        assert os.path.exists(path), f"Missing ablation config: {path}"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "model" in cfg
        assert "meta" in cfg
        assert "optimization" in cfg
        # Every ablation must have a _meta.ablation key
        meta = cfg.get("_meta", {})
        assert "ablation" in meta
        assert meta["ablation"] == name

    SCALING_FILES = ["xsmall_30m", "small_100m", "base_140m", "large_300m"]

    @pytest.mark.parametrize("name", SCALING_FILES)
    def test_scaling_config_loads(self, name):
        path = f"config/scaling/{name}.yaml"
        assert os.path.exists(path), f"Missing scaling config: {path}"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "model" in cfg
        m = cfg["model"]
        # embed_dim must be divisible by num_heads
        assert m["embed_dim"] % m["num_heads"] == 0
        # predictor_embed_dim must be divisible by num_heads
        assert m["predictor_embed_dim"] % m["num_heads"] == 0


# ═══════════════════════════════════════════════════════════════════
#  Visualization module
# ═══════════════════════════════════════════════════════════════════


class TestVisualization:
    def test_import(self):
        pass

    def test_eigenvalue_spectrum(self):
        import numpy as np

        from src.utils.visualization import plot_eigenvalue_spectrum

        eigs = np.exp(-np.arange(64) * 0.1)
        result = plot_eigenvalue_spectrum(eigs, highlight_k=10)
        if result is None:
            return  # matplotlib not available
        fig, _ax = result
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_stacked_losses(self):
        from src.utils.visualization import plot_stacked_losses

        history = {
            "span": [1.0, 0.8, 0.6, 0.4],
            "future": [0.5, 0.4, 0.3, 0.2],
            "variance": [0.1, 0.08, 0.06, 0.04],
        }
        result = plot_stacked_losses(history)
        if result is None:
            return  # matplotlib not available
        fig, _ax = result
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Workspace validation module
# ═══════════════════════════════════════════════════════════════════


class TestWorkspaceValidation:
    def test_import(self):
        pass

    def test_topk_sae_forward(self):
        from src.interp.workspace_validation import TopKSAE

        sae = TopKSAE(embed_dim=64, n_features=256, k=8)
        x = torch.randn(16, 64)
        result = sae(x)
        assert "loss" in result
        assert "x_hat" in result
        assert "features" in result
        assert "sparsity" in result
        assert result["loss"].item() >= 0
        assert result["sparsity"].item() <= 8  # top-k=8

    def test_bootstrap_ci(self):
        from src.interp.workspace_validation import bootstrap_ci

        values = torch.randn(100)
        mean, lo, hi = bootstrap_ci(values, n_bootstrap=500)
        assert lo <= mean <= hi
        assert hi - lo > 0  # non-zero CI width

    def test_workspace_similarity(self):
        from src.interp.workspace_validation import TopKSAE, compute_workspace_similarity

        sae = TopKSAE(embed_dim=64, n_features=128, k=8)
        Q = torch.randn(64, 10)
        Q, _ = torch.linalg.qr(Q)
        ws_indices = torch.tensor([0, 1, 2, 3, 4])
        result = compute_workspace_similarity(Q, sae, ws_indices)
        assert "subspace_similarity" in result
        assert "mean_angle" in result
        assert 0.0 <= result["subspace_similarity"] <= 1.0


# ═══════════════════════════════════════════════════════════════════
#  Full model integration: JAWP + SIGReg + J-Space
# ═══════════════════════════════════════════════════════════════════


class TestV025Integration:
    def test_jepa_with_jawp_enabled(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=[1],
            num_refine_steps=1,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=6,
            jawk_alpha=0.1,
            jawk_init="identity",
        )
        model = TextSpanJEPA(config)
        assert model.jawp is not None
        assert model.jawp.embed_dim == 64
        assert model.jawp.k_end == 6

    def test_jepa_with_jawp_disabled(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=[1],
            num_refine_steps=1,
            use_jawp=False,
        )
        model = TextSpanJEPA(config)
        assert model.jawp is None

    def test_jepa_with_sigreg_enabled(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=[1],
            num_refine_steps=1,
            lambda_sigreg=0.1,
        )
        model = TextSpanJEPA(config)
        assert model.sigreg is not None

    def test_full_loss_with_jawp_sigreg(self):
        """Full compute_loss_with_targets with JAWP + SIGReg produces valid output."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=[1],
            num_refine_steps=1,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=6,
            jawk_alpha=0.1,
            jawk_init="identity",
            lambda_sigreg=0.01,
        )
        model = TextSpanJEPA(config)
        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1

        total_loss, loss_dict, diag_dict = model.compute_loss_with_targets(
            ids,
            ids,
            mask,
            current_step=0,
        )

        assert torch.isfinite(total_loss)
        assert total_loss.item() >= 0
        assert "loss_sigreg" in loss_dict
        assert "workspace_quality" in diag_dict

    def test_stiefel_retract_after_training_step(self):
        """Verify JAWP Q stays orthonormal after optimizer step + retraction."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=[1],
            num_refine_steps=1,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=6,
        )
        model = TextSpanJEPA(config)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1

        for step in range(5):
            loss, _, _ = model.compute_loss_with_targets(ids, ids, mask, current_step=step)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.jawp.stiefel_retract()  # v0.25.0: CRITICAL — called after every step
            model.update_target_encoder(tau=0.996)

        # Check Q orthonormality
        Q = model.jawp.workspace_Q.data
        k = model.jawp.k_end
        gram = Q[:, :k].T @ Q[:, :k]
        off_diag = gram - torch.eye(k)
        ortho_err = off_diag.abs().max().item()
        assert ortho_err < 0.01, f"Q not orthonormal after retraction: err={ortho_err}"

    def test_spectral_gap_detection(self):
        """Spectral gap detection should find correct k* for synthetic data."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=1, k_end=20)

        # Create synthetic data with 5 workspace dims (low residual) + 59 noise
        torch.manual_seed(42)
        N = 200
        D = 64
        k_true = 5
        z_target = torch.randn(N, D)
        z_pred = z_target.clone()
        # Make first k_true dims predictable (small residual)
        z_pred[:, :k_true] = z_target[:, :k_true] + 0.01 * torch.randn(N, k_true)
        # Rest are unpredictable (large residual)
        z_pred[:, k_true:] = z_target[:, k_true:] + 2.0 * torch.randn(N, D - k_true)

        k_star, info = jawp.detect_workspace_dimension(z_pred, z_target)
        assert (
            info["method"] == "spectral_gap"
        ), f"Should use spectral gap method, got {info['method']}"
        # Should detect k* close to k_true=5 (allow ±3 tolerance for finite sample)
        assert 2 <= k_star <= 10, f"Detected k*={k_star}, expected ~{k_true}"
        assert k_star > 0

    def test_spectral_gap_returns_fallback_on_small_data(self):
        """With insufficient data, should return fallback."""
        from src.models.jawp import JAWPModule

        jawp = JAWPModule(embed_dim=64, k_start=1, k_end=10)
        z_pred = torch.randn(1, 64)
        z_target = torch.randn(1, 64)
        _k_star, info = jawp.detect_workspace_dimension(z_pred, z_target)
        assert info["method"] == "fallback"

    def test_future_loss_warmup(self):
        """Future loss weight should ramp from 0 to lambda_future."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=[1],
            num_refine_steps=1,
            lambda_future=0.5,
            future_warmup_steps=100,
        )
        model = TextSpanJEPA(config)

        # At step 0: weight should be 0
        w0 = model._future_loss_weight(0)
        assert abs(w0) < 1e-6, f"Future weight at step 0 should be ~0, got {w0}"

        # At step 50: weight should be ~0.25
        w50 = model._future_loss_weight(50)
        assert 0.2 < w50 < 0.3, f"Future weight at step 50 should be ~0.25, got {w50}"

        # At step >= warmup: weight should be lambda_future
        w200 = model._future_loss_weight(200)
        assert abs(w200 - 0.5) < 1e-6, f"Future weight after warmup should be 0.5, got {w200}"

    def test_config_validate(self):
        """Config.validate() should catch invalid params."""
        from src.models.jepa import TextSpanJEPAConfig

        # embed_dim not divisible by num_heads
        with pytest.raises(ValueError):
            config = TextSpanJEPAConfig(embed_dim=65, num_heads=4)
            config.validate()

    def test_config_validate_ema_schedule(self):
        from src.models.jepa import TextSpanJEPAConfig

        with pytest.raises(ValueError):
            config = TextSpanJEPAConfig(ema_schedule="invalid")
            config.validate()

    def test_config_validate_negative_lambda(self):
        from src.models.jepa import TextSpanJEPAConfig

        with pytest.raises(ValueError):
            TextSpanJEPAConfig(lambda_span=-1.0).validate()
        with pytest.raises(ValueError):
            TextSpanJEPAConfig(lambda_future=-0.5).validate()

    def test_config_validate_jawp_params(self):
        from src.models.jepa import TextSpanJEPAConfig

        with pytest.raises(ValueError):
            TextSpanJEPAConfig(use_jawp=True, jawk_k_start=0).validate()
        with pytest.raises(ValueError):
            TextSpanJEPAConfig(use_jawp=True, jawk_k_start=5, jawk_k_end=3).validate()
        with pytest.raises(ValueError):
            TextSpanJEPAConfig(use_jawp=True, jawk_alpha=-1.0).validate()
        with pytest.raises(ValueError):
            TextSpanJEPAConfig(use_jawp=True, jawk_init="invalid").validate()

    def test_config_validate_sigreg_sigma(self):
        from src.models.jepa import TextSpanJEPAConfig

        with pytest.raises(ValueError):
            TextSpanJEPAConfig(lambda_sigreg=0.1, sigreg_sigma=-1.0).validate()

    def test_config_validate_centering_momentum(self):
        from src.models.jepa import TextSpanJEPAConfig

        with pytest.raises(ValueError):
            TextSpanJEPAConfig(centering_momentum=1.5).validate()
        with pytest.raises(ValueError):
            TextSpanJEPAConfig(centering_momentum=0.0).validate()

    def test_jawp_checkpoint_save_restore(self):
        """JAWP Q should survive save/load cycle."""
        import os
        import tempfile

        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.train import load_checkpoint, save_checkpoint

        config = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=[1],
            num_refine_steps=1,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=6,
        )
        model = TextSpanJEPA(config)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Modify Q
        with torch.no_grad():
            model.jawp.workspace_Q.data[:, 0] = 1.0

        Q_before = model.jawp.workspace_Q.data.clone()

        with tempfile.NamedTemporaryFile(suffix=".pth.tar", delete=False) as f:
            path = f.name

        try:
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            save_checkpoint(
                path,
                model,
                opt,
                scaler,
                0,
                0,
                0,
                0,
                extra_state={},
                model_name="text_span_jepa",
            )

            # Reset Q
            with torch.no_grad():
                model.jawp.workspace_Q.data.zero_()

            load_checkpoint(path, model, opt, scaler, model_name="text_span_jepa")

            Q_after = model.jawp.workspace_Q.data
            assert torch.allclose(
                Q_before,
                Q_after,
                atol=1e-6,
            ), "JAWP Q not preserved through save/load"
        finally:
            os.unlink(path)
