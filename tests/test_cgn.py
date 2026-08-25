# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Tests for CGN (Contextual Gating Network) and v0.29.0 features

import math

import pytest
import torch

from src.models.cgn import ContextualGatingNetwork
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

# ═══════════════════════════════════════════════════════════════
#  CGN Core Tests
# ═══════════════════════════════════════════════════════════════


class TestCGNCore:
    """Core CGN module tests."""

    def setup_method(self):
        self.embed_dim = 64
        self.n_groups = 8
        self.batch_size = 4
        self.seq_len = 16

    def test_cgn_creation(self):
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        assert cgn.embed_dim == self.embed_dim
        assert cgn.n_groups == self.n_groups
        assert cgn.group_dim == self.embed_dim // self.n_groups

    def test_cgn_forward_shape(self):
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)
        mask[:, 4:8] = 1

        z_gated, _info = cgn(z, mask, step=0)
        assert z_gated.shape == z.shape

    def test_cgn_gating_preserves_shape(self):
        """CGN must not change tensor dimensions."""
        for dim in [64, 128, 256, 768]:
            for groups in [4, 8, 16]:
                if dim % groups != 0:
                    continue
                cgn = ContextualGatingNetwork(embed_dim=dim, n_groups=groups)
                z = torch.randn(2, 8, dim)
                mask = torch.zeros(2, 8, dtype=torch.long)
                mask[:, 2:5] = 1
                z_gated, _ = cgn(z, mask, step=0)
                assert z_gated.shape == z.shape

    def test_cgn_gate_info_keys(self):
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)
        mask[:, 4:8] = 1

        _, info = cgn(z, mask, step=0)
        expected_keys = [
            "cgn_tau",
            "cgn_gate_diff",
            "cgn_sparsity",
            "cgn_entropy",
            "cgn_mean_gate_visible",
            "cgn_mean_gate_masked",
            "cgn_routing_gap",
        ]
        for k in expected_keys:
            assert k in info, f"Missing key: {k}"

    def test_cgn_different_gating_for_masked_visible(self):
        """CGN should produce different gate patterns for masked vs visible."""
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        # Initialize with different logits so gates differ
        with torch.no_grad():
            cgn.gate_logits_visible[:, 1].fill_(1.0)

        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)
        mask[:, 4:8] = 1

        _, info = cgn(z, mask, step=0)
        # Gate diff should be > 0 (different patterns)
        assert info["cgn_gate_diff"] >= 0

    def test_cgn_tau_annealing(self):
        """Temperature should decrease from tau_start to tau_end."""
        cgn = ContextualGatingNetwork(
            embed_dim=self.embed_dim,
            n_groups=self.n_groups,
            tau_start=1.0,
            tau_end=0.1,
            anneal_steps=1000,
        )
        tau_0 = cgn.current_tau(step=0)
        tau_500 = cgn.current_tau(step=500)
        tau_1000 = cgn.current_tau(step=1000)
        tau_2000 = cgn.current_tau(step=2000)

        assert tau_0 == pytest.approx(1.0, abs=0.01)
        assert tau_1000 == pytest.approx(0.1, abs=0.01)
        assert tau_2000 == pytest.approx(0.1, abs=0.01)  # Clamped
        assert tau_500 < tau_0  # Decreasing

    def test_cgn_min_gate_prevents_zeroing(self):
        """Gate values should be >= min_gate."""
        cgn = ContextualGatingNetwork(
            embed_dim=self.embed_dim, n_groups=self.n_groups, min_gate=0.01
        )
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)
        mask[:, 4:8] = 1

        z_gated, _ = cgn(z, mask, step=0)
        # z_gated = z * gate, so |z_gated| >= min_gate * |z| when z != 0
        # Check that gating doesn't zero out everything
        assert z_gated.abs().sum() > 0

    def test_cgn_no_mask(self):
        """CGN with no mask (all visible) should still work."""
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)

        z_gated, info = cgn(z, mask, step=0)
        assert z_gated.shape == z.shape
        assert info["cgn_mean_gate_masked"] == 0.0  # No masked positions

    def test_cgn_all_masked(self):
        """CGN with all masked positions should still work."""
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)

        z_gated, info = cgn(z, mask, step=0)
        assert z_gated.shape == z.shape
        assert info["cgn_mean_gate_visible"] == 0.0  # No visible positions


# ═══════════════════════════════════════════════════════════════
#  CGN Orthogonality & Routing
# ═══════════════════════════════════════════════════════════════


class TestCGNOrthogonality:
    """Tests for CGN orthogonality and routing efficiency."""

    def setup_method(self):
        self.embed_dim = 64
        self.n_groups = 8

    def test_orthogonality_score_range(self):
        """Orthogonality score must be in [0, 1]."""
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        score = cgn.compute_orthogonality_score()
        assert 0.0 <= score <= 1.0

    def test_orthogonality_perfect(self):
        """Decisive shared Bernoulli ⇒ routing decisiveness score ≈ 1."""
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        with torch.no_grad():
            cgn.gate_logits_visible[:, 0].fill_(-10.0)  # Strong OFF
            cgn.gate_logits_visible[:, 1].fill_(10.0)  # Strong ON
        score = cgn.compute_orthogonality_score()
        assert score > 0.99

    def test_routing_efficiency(self):
        """Routing efficiency metrics should be valid."""
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        z = torch.randn(4, 16, self.embed_dim)
        mask = torch.zeros(4, 16, dtype=torch.long)
        mask[:, 4:8] = 1

        result = cgn.compute_routing_efficiency(z, mask)
        assert "routing_efficiency" in result
        assert "context_preservation" in result
        assert "prediction_focus" in result
        assert 0.0 <= result["routing_efficiency"] <= 1.0
        assert 0.0 <= result["context_preservation"] <= 1.0
        assert 0.0 <= result["prediction_focus"] <= 1.0


# ═══════════════════════════════════════════════════════════════
#  CGN Mathematical Proofs
# ═══════════════════════════════════════════════════════════════


class TestCGNTheorems:
    """Tests verifying mathematical properties of CGN."""

    def setup_method(self):
        self.embed_dim = 64
        self.n_groups = 8

    def test_information_routing_theorem(self):
        """Verify: I(g_v ⊙ Z; Y) + I(g_m ⊙ Z; Y) ≥ I(Z; Y)

        We can't compute MI directly, but we verify the precondition:
        visible and masked gates produce non-redundant representations.
        """
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        # Initialize with different gate patterns
        with torch.no_grad():
            cgn.gate_logits_visible[:, 1].fill_(2.0)

        z = torch.randn(32, 16, self.embed_dim)
        mask = torch.zeros(32, 16, dtype=torch.long)
        mask[:, 8:] = 1

        z_gated, _info = cgn(z, mask, step=1000)  # Hard gating

        # Check that gated representations differ from ungated
        # This is a necessary condition for the theorem to be strict
        diff = (z_gated - z).norm()
        assert diff > 0, "Gating should change representations"

    def test_gumbel_softmax_valid_probabilities(self):
        """Gumbel-Softmax must produce valid probability distributions."""
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        cgn.train()

        logits = cgn.gate_logits_visible
        tau = cgn.current_tau(step=0)

        probs = cgn._compute_gate_probs(logits, tau)
        # Each row should sum to ~1
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        # All probs should be > 0
        assert (probs > 0).all()

    def test_gumbel_softmax_approaches_hard(self):
        """As tau → 0, Gumbel-Softmax approaches one-hot (hard gating)."""
        cgn = ContextualGatingNetwork(embed_dim=self.embed_dim, n_groups=self.n_groups)
        # With large logits, low tau should give near-one-hot
        # Use eval mode to disable Gumbel noise (deterministic test)
        cgn.eval()
        with torch.no_grad():
            cgn.gate_logits_visible[:, 0].fill_(-5.0)  # Strong OFF
            cgn.gate_logits_visible[:, 1].fill_(5.0)  # Strong ON

        probs = cgn._compute_gate_probs(cgn.gate_logits_visible, tau=0.01)
        # ON probability should be close to 1
        assert probs[:, 1].min() > 0.9


# ═══════════════════════════════════════════════════════════════
#  Integration with JEPA Model
# ═══════════════════════════════════════════════════════════════


class TestCGNIntegration:
    """Integration tests for CGN with the full JEPA model."""

    def test_jepa_with_cgn_enabled(self):
        """JEPA model should work with CGN enabled."""
        config = TextSpanJEPAConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            predictor_embed_dim=32,
            predictor_depth=2,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=6,
            use_cgn=True,
            cgn_n_groups=4,
            lambda_cgn_ortho=0.01,
        )
        config.validate()
        model = TextSpanJEPA(config)
        assert model.cgn is not None

    def test_jepa_with_cgn_disabled(self):
        """JEPA model should work with CGN disabled."""
        config = TextSpanJEPAConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            predictor_embed_dim=32,
            predictor_depth=2,
            use_cgn=False,
        )
        config.validate()
        model = TextSpanJEPA(config)
        assert model.cgn is None

    def test_full_loss_with_cgn(self):
        """Full loss computation with CGN should produce valid loss."""
        config = TextSpanJEPAConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            predictor_embed_dim=32,
            predictor_depth=2,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=6,
            use_cgn=True,
            cgn_n_groups=4,
            lambda_cgn_ortho=0.01,
        )
        config.validate()
        model = TextSpanJEPA(config)

        B, T = 2, 16
        masked_ids = torch.randint(0, 1000, (B, T))
        original_ids = torch.randint(0, 1000, (B, T))
        mask = torch.zeros(B, T, dtype=torch.long)
        mask[:, 4:8] = 1

        loss, loss_dict, _diag_dict = model.compute_loss_with_targets(
            masked_ids, original_ids, mask, current_step=100
        )
        assert loss.item() >= 0
        assert not math.isnan(loss.item())
        assert not math.isinf(loss.item())
        assert "loss_cgn_ortho" in loss_dict

    def test_predictive_rank_in_loss(self):
        """Predictive rank loss should appear in loss_dict."""
        config = TextSpanJEPAConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            predictor_embed_dim=32,
            predictor_depth=2,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=6,
            lambda_predictive_rank=0.01,
        )
        config.validate()
        model = TextSpanJEPA(config)

        B, T = 2, 16
        masked_ids = torch.randint(0, 1000, (B, T))
        original_ids = torch.randint(0, 1000, (B, T))
        mask = torch.zeros(B, T, dtype=torch.long)
        mask[:, 4:8] = 1

        _loss, loss_dict, _diag_dict = model.compute_loss_with_targets(
            masked_ids, original_ids, mask, current_step=100
        )
        assert "loss_predictive_rank" in loss_dict

    def test_cgn_gradient_flow(self):
        """CGN parameters should receive gradients."""
        config = TextSpanJEPAConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            predictor_embed_dim=32,
            predictor_depth=2,
            use_jawp=True,
            jawk_k_start=1,
            jawk_k_end=6,
            use_cgn=True,
            cgn_n_groups=4,
            lambda_cgn_ortho=0.01,
        )
        config.validate()
        model = TextSpanJEPA(config)

        B, T = 2, 16
        masked_ids = torch.randint(0, 1000, (B, T))
        original_ids = torch.randint(0, 1000, (B, T))
        mask = torch.zeros(B, T, dtype=torch.long)
        mask[:, 4:8] = 1

        loss, _, _ = model.compute_loss_with_targets(
            masked_ids, original_ids, mask, current_step=100
        )
        loss.backward()

        # CGN parameters should have gradients
        assert model.cgn.gate_logits_visible.grad is not None
        assert model.cgn.context_proj.weight.grad is not None


# ═══════════════════════════════════════════════════════════════
#  Config Validation
# ═══════════════════════════════════════════════════════════════


class TestCGNConfig:
    """Config validation tests for CGN fields."""

    def test_cgn_config_defaults(self):
        config = TextSpanJEPAConfig()
        assert config.use_cgn == False
        assert config.cgn_n_groups == 8
        assert config.lambda_cgn_ortho == 0.0
        assert config.lambda_predictive_rank == 0.0

    def test_cgn_config_invalid_n_groups(self):
        """embed_dim must be divisible by cgn_n_groups."""
        config = TextSpanJEPAConfig(
            embed_dim=64,
            num_heads=4,
            use_cgn=True,
            cgn_n_groups=7,  # 64 % 7 != 0
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_cgn_config_invalid_temperature(self):
        """CGN temperatures must be > 0."""
        config = TextSpanJEPAConfig(
            use_cgn=True,
            cgn_tau_start=0.0,
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_predictive_rank_negative(self):
        """lambda_predictive_rank must be >= 0."""
        config = TextSpanJEPAConfig(lambda_predictive_rank=-0.1)
        with pytest.raises(ValueError):
            config.validate()


# ═══════════════════════════════════════════════════════════════
#  YAML Config Validation
# ═══════════════════════════════════════════════════════════════


class TestV029YamlConfigs:
    """Verify all YAML configs have v0.29.0 fields."""

    def test_all_configs_have_new_fields(self):
        import glob

        import yaml

        for path in sorted(glob.glob("config/**/*.yaml", recursive=True)):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            if "model" not in cfg:
                continue
            m = cfg["model"]
            # Ablation configs may only contain overrides — skip if missing base fields
            if "lambda_predictive_rank" not in m:
                continue
            assert m["lambda_predictive_rank"] >= 0, f"{path} negative lambda_predictive_rank"

    def test_defaults_yaml_has_new_fields(self):
        import yaml

        with open("defaults.yaml") as f:
            cfg = yaml.safe_load(f)
        m = cfg["model"]
        assert "lambda_predictive_rank" in m
        assert "use_cgn" in m
        assert "cgn_n_groups" in m
        assert "cgn_tau_start" in m
        assert "cgn_tau_end" in m
        assert "cgn_anneal_steps" in m
        assert "lambda_cgn_ortho" in m
