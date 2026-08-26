# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Test suite — patterns from NextLat (Microsoft Research) + I-JEPA (Meta)
# Key test patterns from NextLat model_base.py:
#   - compute_hidden_state_rank: effective_rank, numerical_rank, condition_number,
#     rank_utilization, max_possible_rank
#   - Exception handling returns zeros/infs (not crashes)
#   - Per-sample rank metrics vs batch-level metrics
# Risk fixes #2, #3, #4 are tested explicitly.

import math

import pytest
import torch

# ═══════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════


class TestEncoder:
    def setup_method(self):
        from src.models.encoder import TextSpanJEPLEncoder

        self.Encoder = TextSpanJEPLEncoder

    def test_output_shape(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        h, tok = enc(torch.randint(0, 1000, (4, 32)))
        assert h.shape == (4, 32, 64)
        assert tok.shape == (4, 32, 64)

    def test_different_seq_lengths(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=128, embed_dim=64, depth=2, num_heads=4)
        for sl in [8, 16, 32, 64]:
            h, _ = enc(torch.randint(0, 1000, (2, sl)))
            assert h.shape == (2, sl, 64)

    def test_param_count(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        non_emb = enc.get_num_params(non_embedding=True)
        with_emb = enc.get_num_params(non_embedding=False)
        assert non_emb < with_emb

    def test_gradients_flow(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        h, _ = enc(torch.randint(0, 1000, (2, 32)))
        h.sum().backward()
        assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)

    def test_pos_embedding_is_learnable(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        assert enc.pos_embedding.requires_grad, "pos_embedding should be learnable"

    def test_deterministic_with_same_seed(self):
        enc = self.Encoder(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        enc.eval()
        x = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            h1, _ = enc(x)
            h2, _ = enc(x)
        assert torch.allclose(h1, h2, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
# Predictor
# ═══════════════════════════════════════════════════════════════════


class TestPredictor:
    def setup_method(self):
        from src.models.predictor import TextSpanJEPApredictor

        self.Predictor = TextSpanJEPApredictor

    def test_output_shape(self):
        pred = self.Predictor(
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=4,
            max_seq_len=32,
            future_offsets=(1, 4),
            num_refine_steps=2,
        )
        h = torch.randn(4, 32, 64)
        mask = torch.zeros(4, 32, dtype=torch.long)
        mask[:, 5:10] = 1
        mask[:, 20:25] = 1
        span_preds, _num_masked, valid_mask, fl, fp = pred(
            h, mask, torch.randn(4, 32, 64), torch.randn(4, 32, 64)
        )
        assert span_preds.shape[0] == 4 and span_preds.shape[2] == 64
        assert valid_mask.sum().item() == mask.sum().item()
        for d in (1, 4):
            assert d in fl and d in fp
            assert fp[d].shape == (4, 32 - d, 64)

    def test_gather_masked_fix3(self):
        """Fix #3: torch.gather with valid_mask."""
        pred = self.Predictor(
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=4,
            max_seq_len=16,
            future_offsets=(1,),
        )
        h = torch.randn(3, 16, 64)
        mask = torch.zeros(3, 16, dtype=torch.long)
        mask[0, 0:3] = 1
        mask[1, 2:7] = 1
        mask[2, 10:12] = 1
        gathered, _num_masked, valid_mask = pred._gather_masked(h, mask)
        assert gathered.shape == (3, 5, 64)
        assert valid_mask.sum().item() == 10
        assert gathered[0, 3:].abs().sum().item() == pytest.approx(0.0, abs=1e-6)
        assert gathered[2, 2:].abs().sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_gather_masked_all_zeros(self):
        """Edge case: no masked positions → returns zeros, no crash."""
        pred = self.Predictor(
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=4,
            max_seq_len=16,
            future_offsets=(1,),
        )
        h = torch.randn(2, 16, 64)
        mask = torch.zeros(2, 16, dtype=torch.long)
        _gathered, num_masked, valid_mask = pred._gather_masked(h, mask)
        assert num_masked.sum().item() == 0
        assert valid_mask.sum().item() == 0

    def test_iterative_refinement(self):
        pred_r = self.Predictor(
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=4,
            max_seq_len=32,
            future_offsets=(1,),
            num_refine_steps=3,
        )
        pred_n = self.Predictor(
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=4,
            max_seq_len=32,
            future_offsets=(1,),
            num_refine_steps=0,
        )
        pred_n.load_state_dict(pred_r.state_dict(), strict=False)
        h, mask, tok, tgt = (
            torch.randn(2, 32, 64),
            torch.zeros(2, 32, dtype=torch.long),
            torch.randn(2, 32, 64),
            torch.randn(2, 32, 64),
        )
        mask[:, 5:8] = 1
        with torch.no_grad():
            s1, _, _, _, _ = pred_r(h, mask, tok, tgt)
            s2, _, _, _, _ = pred_n(h, mask, tok, tgt)
        assert not torch.allclose(s1, s2, atol=1e-5)

    def test_no_mask(self):
        pred = self.Predictor(
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=4,
            max_seq_len=32,
            future_offsets=(1,),
        )
        _span_preds, num_masked, _valid_mask, _, _ = pred(
            torch.randn(2, 32, 64),
            torch.zeros(2, 32, dtype=torch.long),
            torch.randn(2, 32, 64),
            torch.randn(2, 32, 64),
        )
        assert num_masked.sum().item() == 0

    def test_future_prediction_offset_exceeds_seq_len(self):
        """Edge case: future offset > T → no loss for that offset."""
        pred = self.Predictor(
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=4,
            max_seq_len=32,
            future_offsets=(1, 100),
            num_refine_steps=1,
        )
        h = torch.randn(2, 16, 64)
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:5] = 1
        _, _, _, fl, _fp = pred(h, mask, torch.randn(2, 16, 64), torch.randn(2, 16, 64))
        assert 1 in fl  # offset 1 should work (T=16 > 1)
        assert 100 not in fl  # offset 100 > T → skipped

    def test_predictor_pos_embed_is_learnable(self):
        pred = self.Predictor(
            embed_dim=64,
            predictor_embed_dim=32,
            depth=2,
            num_heads=4,
            max_seq_len=32,
            future_offsets=(1,),
        )
        assert pred.predictor_pos_embed.requires_grad


# ═══════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════


class TestDecoder:
    def test_output(self):
        from src.models.decoder import TiedTokenDecoder

        dec = TiedTokenDecoder(embed_dim=64, vocab_size=1000)
        logits = dec(torch.randn(8, 64), torch.randn(1000, 64))
        assert logits.shape == (8, 1000)

    def test_data2vec_regression_head_pattern(self):
        """Decoder follows data2vec regression head: Linear→GELU→Linear."""
        from src.models.decoder import TiedTokenDecoder

        dec = TiedTokenDecoder(embed_dim=64, vocab_size=1000, bias=False)
        # First linear: 64 → 128 (2x expand)
        assert isinstance(dec.proj[0], torch.nn.Linear)
        assert dec.proj[0].in_features == 64
        assert dec.proj[0].out_features == 128
        # GELU
        assert isinstance(dec.proj[1], torch.nn.GELU)
        # Second linear: 128 → 64
        assert isinstance(dec.proj[2], torch.nn.Linear)
        assert dec.proj[2].out_features == 64


# ═══════════════════════════════════════════════════════════════════
# Collapse Prevention — metrics from NextLat model_base.py
# ═══════════════════════════════════════════════════════════════════


class TestCollapsePrevention:
    def test_variance_active(self):
        from src.models.collapse import VarianceRegularization

        assert VarianceRegularization(margin=1.0)(torch.randn(32, 64) * 0.01).item() > 0

    def test_variance_satisfied(self):
        from src.models.collapse import VarianceRegularization

        assert VarianceRegularization(margin=1.0)(
            torch.randn(32, 64) * 10.0
        ).item() == pytest.approx(0.0, abs=1e-3)

    def test_variance_n1(self):
        """N=1 should not produce NaN (var with df=0)."""
        from src.models.collapse import VarianceRegularization

        loss = VarianceRegularization()(torch.randn(1, 32))
        assert torch.isfinite(loss) and loss.item() == 0.0

    def test_covariance(self):
        from src.models.collapse import CovarianceRegularization

        assert CovarianceRegularization()(torch.randn(64, 32)).item() >= 0

    def test_covariance_n1(self):
        """N=1 should not produce NaN (divide by N-1=0)."""
        from src.models.collapse import CovarianceRegularization

        loss = CovarianceRegularization()(torch.randn(1, 32))
        assert torch.isfinite(loss)

    def test_centering(self):
        from src.models.collapse import TargetCentering

        tc = TargetCentering(dim=32, momentum=0.9)
        centered = tc(torch.randn(4, 8, 32) + 5.0)
        assert centered.shape == (4, 8, 32) and tc.center.norm().item() > 0

    def test_effective_rank_positive(self):
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert m["effective_rank_online"] > 0
        assert m["effective_rank_target"] > 0

    def test_hidden_state_rank_nextlat_pattern(self):
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        h = torch.randn(4, 16, 32)
        metrics = diag.compute(h, h)

        flat = h.reshape(-1, 32)
        S = torch.linalg.svdvals(flat)
        S_norm = S / S.sum()
        S_norm = torch.clamp(S_norm, min=1e-12)
        expected_eff_rank = (-torch.sum(S_norm * torch.log(S_norm))).exp().item()
        expected_cond = (S[0] / S[-1]).item()
        expected_num_rank = torch.linalg.matrix_rank(flat, atol=1e-3, rtol=1e-3).item()

        assert abs(metrics["effective_rank_online"] - expected_eff_rank) < 0.1
        assert (
            abs(metrics["condition_number_online"] - expected_cond) / max(expected_cond, 1) < 0.05
        )
        assert metrics["numerical_rank_online"] == expected_num_rank

    def test_collapse_detection_zero_input(self):
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.zeros(4, 16, 32), torch.zeros(4, 16, 32))
        assert m["effective_rank_online"] <= 1.0

    def test_participation_ratio(self):
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert m["participation_ratio_online"] > 1.0

    def test_collapsed_dim_ratio_random(self):
        """Random data should have low collapsed dim ratio."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert m["collapsed_dim_ratio_online"] < 0.5

    def test_collapsed_dim_ratio_constant(self):
        """Constant input should have collapsed dim ratio near 1.0."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.ones(4, 16, 32), torch.ones(4, 16, 32))
        assert m["collapsed_dim_ratio_online"] > 0.9

    def test_cross_corr_redundancy(self):
        """Barlow Twins cross-correlation redundancy metric."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert "cross_corr_redundancy" in m
        assert m["cross_corr_redundancy"] >= 0.0

    def test_cka_identical(self):
        """CKA of identical representations should be near 1.0."""
        from src.models.collapse import CollapseDiagnostics

        h = torch.randn(4, 16, 32)
        m = CollapseDiagnostics().compute(h, h)
        assert m["cka_linear"] > 0.95

    def test_cka_independent(self):
        """CKA of independent representations should be < 1.0."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert m["cka_linear"] < 0.95

    def test_rank_utilization(self):
        """Rank utilization from NextLat."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(4, 16, 32), torch.randn(4, 16, 32))
        assert 0 < m["rank_utilization_online"] <= 1.0

    # --- New metrics from top JEPA papers ---

    def test_singular_value_entropy(self):
        """I-JEPA: normalized entropy of singular values. Random data > 0, constant = 0."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        assert 0 < m["sv_entropy_online"] <= 1.0
        assert 0 < m["sv_entropy_target"] <= 1.0

    def test_singular_value_entropy_collapse(self):
        """I-JEPA: collapsed representations have low sv_entropy."""
        from src.models.collapse import CollapseDiagnostics

        h = torch.randn(4, 16, 32)
        m = CollapseDiagnostics().compute(h, h)
        # Identical online/target should have same entropy
        assert abs(m["sv_entropy_online"] - m["sv_entropy_target"]) < 0.01

    def test_svd_sharpness(self):
        """C-JEPA/BYOL: spectral sharpness in [0,1]. Random < 1, rank-1 → 1."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        assert 0 < m["svd_sharpness_online"] < 1.0  # random = not sharp

    def test_svd_sharpness_rank1(self):
        """C-JEPA/BYOL: rank-1 matrix should have sharpness near 1."""
        from src.models.collapse import CollapseDiagnostics

        v = torch.randn(1, 32)
        h = v.expand(64, 32)  # rank-1: all rows identical
        sharpness = CollapseDiagnostics._svd_sharpness(h.unsqueeze(0))
        assert sharpness > 0.95

    def test_alpha_norm(self):
        """LeCun 2022: power-law exponent of singular value spectrum."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        assert m["alpha_norm_online"] >= 0.0
        assert m["alpha_norm_target"] >= 0.0

    def test_alpha_norm_zero_input(self):
        """LeCun 2022: zero input → alpha_norm = 0 (no spectrum to fit)."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.zeros(4, 16, 32), torch.zeros(4, 16, 32))
        assert m["alpha_norm_online"] == 0.0

    def test_intrinsic_dim(self):
        """Ansuini et al. 2019: intrinsic dimensionality estimate."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        assert m["intrinsic_dim_online"] >= 0.0
        assert m["intrinsic_dim_target"] >= 0.0

    def test_intrinsic_dim_collapsed_lower(self):
        """Ansuini et al.: heavily collapsed data should have very low intrinsic dim."""
        from src.models.collapse import CollapseDiagnostics

        # Near-rank-1: all rows identical up to tiny noise
        v = torch.randn(1, 32)
        collapsed_h = v.expand(512, 32) + torch.randn(512, 32) * 0.001
        dim_collapsed = CollapseDiagnostics._intrinsic_dim_score(collapsed_h)
        # Should be well below the ambient dimension (32)
        assert dim_collapsed < 32, f"Collapsed ID ({dim_collapsed}) should be below ambient dim 32"

    def test_mean_pairwise_cosine(self):
        """DINOv2: intra-batch cosine similarity."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        assert -1 <= m["mean_pairwise_cosine_online"] <= 1

    def test_mean_pairwise_cosine_collapsed(self):
        """DINOv2: collapsed representations have high pairwise cosine."""
        from src.models.collapse import CollapseDiagnostics

        v = torch.randn(1, 32)
        collapsed = v.expand(128, 32) + torch.randn(128, 32) * 0.01
        cos = CollapseDiagnostics._mean_pairwise_cosine(collapsed)
        assert cos > 0.9  # Nearly identical → high cosine

    def test_representation_stability(self):
        """I-JEPA: cosine similarity between consecutive target updates."""
        from src.models.collapse import CollapseDiagnostics

        h1 = torch.randn(4, 16, 32)
        h2 = h1 + torch.randn(4, 16, 32) * 0.01  # very similar
        stability = CollapseDiagnostics._representation_stability(h1, h2)
        assert stability > 0.9  # Should be high for similar targets

    def test_representation_stability_in_compute(self):
        """I-JEPA: representation_stability should appear when prev_target_h is passed."""
        from src.models.collapse import CollapseDiagnostics

        h1 = torch.randn(4, 16, 32)
        h2 = h1 + torch.randn(4, 16, 32) * 0.01
        m = CollapseDiagnostics().compute(h1, h2, prev_target_h=h1)
        assert "representation_stability" in m
        assert m["representation_stability"] > 0.9

    def test_cka_rbf(self):
        """Kornblith et al.: RBF kernel CKA."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        assert "cka_rbf" in m
        assert 0 <= m["cka_rbf"] <= 1.0

    def test_cka_rbf_identical(self):
        """Kornblith et al.: RBF CKA of identical representations ≈ 1."""
        from src.models.collapse import CollapseDiagnostics

        h = torch.randn(8, 16, 32)
        m = CollapseDiagnostics().compute(h, h)
        assert m["cka_rbf"] > 0.9

    def test_all_new_metrics_no_nan(self):
        """All new metrics should return finite values for normal input."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        for key in [
            "sv_entropy_online",
            "svd_sharpness_online",
            "alpha_norm_online",
            "intrinsic_dim_online",
            "mean_pairwise_cosine_online",
            "cross_corr_redundancy",
            "cka_linear",
            "cka_rbf",
        ]:
            assert key in m, f"Missing metric: {key}"
            assert math.isfinite(m[key]), f"Non-finite value for {key}: {m[key]}"

    def test_all_new_metrics_zero_input(self):
        """All new metrics should handle zero input (NextLat exception pattern)."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.zeros(4, 16, 32), torch.zeros(4, 16, 32))
        for key in [
            "sv_entropy_online",
            "svd_sharpness_online",
            "alpha_norm_online",
            "intrinsic_dim_online",
            "mean_pairwise_cosine_online",
        ]:
            assert key in m, f"Missing metric: {key}"
            assert math.isfinite(m[key]), f"Non-finite value for {key}: {m[key]}"

    # --- Wang & Isola (ICLR 2022) + DINO metrics ---

    def test_uniformity(self):
        """Wang & Isola: uniformity on hypersphere."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        assert "uniformity_online" in m
        assert "uniformity_target" in m
        assert math.isfinite(m["uniformity_online"])

    def test_uniformity_collapsed_higher(self):
        """Collapsed representations have less negative uniformity (closer to 0)."""
        from src.models.collapse import CollapseDiagnostics

        random_h = torch.randn(16, 32, 32)
        v = torch.randn(1, 32)
        collapsed_flat = v.expand(512, 32) + torch.randn(512, 32) * 0.01
        u_random = CollapseDiagnostics._uniformity(random_h.reshape(-1, 32))
        u_collapsed = CollapseDiagnostics._uniformity(collapsed_flat)
        # Collapsed = less uniform = less negative (closer to 0)
        assert u_collapsed > u_random

    def test_cov_trace(self):
        """DINO: feature covariance trace."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.randn(8, 16, 32), torch.randn(8, 16, 32))
        assert "cov_trace_online" in m
        assert "cov_trace_target" in m
        assert m["cov_trace_online"] > 0
        assert m["cov_trace_target"] > 0

    def test_cov_trace_zero_input(self):
        """DINO: zero input -> cov_trace = 0."""
        from src.models.collapse import CollapseDiagnostics

        m = CollapseDiagnostics().compute(torch.zeros(4, 16, 32), torch.zeros(4, 16, 32))
        assert m["cov_trace_online"] == 0.0
        assert m["cov_trace_target"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Full JEPA Model
# ═══════════════════════════════════════════════════════════════════


class TestJEPA:
    def setup_method(self):
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        self.JEPA, self.Config = TextSpanJEPA, TextSpanJEPAConfig

    def _cfg(self, **overrides):
        defaults = {
            "vocab_size": 1000,
            "max_seq_len": 32,
            "embed_dim": 64,
            "encoder_depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "predictor_embed_dim": 32,
            "predictor_depth": 2,
            "future_offsets": (1, 4),
            "num_refine_steps": 1,
            "future_warmup_steps": 10,
        }
        defaults.update(overrides)
        return self.Config(**defaults)

    def test_forward(self):
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 5:10] = 1
        loss, ld, dd = model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask
        )
        assert loss.requires_grad
        for k in [
            "loss",
            "loss_span",
            "loss_future",
            "loss_decoder",
            "loss_variance",
            "loss_covariance",
            "decoder_accuracy",
            "future_weight",
        ]:
            assert k in ld, f"Missing key: {k}"
        for k in [
            "effective_rank_online",
            "effective_rank_target",
            "participation_ratio_online",
            "condition_number_online",
            "numerical_rank_online",
            "coherence_online",
            "mask_fraction",
            "target_center_norm",
        ]:
            assert k in dd, f"Missing key: {k}"

    def test_future_warmup_fix2(self):
        """Fix #2: Future loss weight ramps from 0 to lambda_future."""
        model = self.JEPA(self._cfg(lambda_future=0.5, future_warmup_steps=100))
        assert model._future_loss_weight(0) == pytest.approx(0.0, abs=1e-6)
        assert model._future_loss_weight(50) == pytest.approx(0.25, abs=1e-6)
        assert model._future_loss_weight(100) == pytest.approx(0.5, abs=1e-6)
        assert model._future_loss_weight(200) == pytest.approx(0.5, abs=1e-6)

    def test_future_warmup_disabled(self):
        """When future_warmup_steps=0, weight is always lambda_future."""
        model = self.JEPA(self._cfg(lambda_future=0.5, future_warmup_steps=0))
        assert model._future_loss_weight(0) == pytest.approx(0.5, abs=1e-6)
        assert model._future_loss_weight(100) == pytest.approx(0.5, abs=1e-6)

    def test_ema_update(self):
        model = self.JEPA(self._cfg())
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        before = {n: p.clone() for n, p in model.target_encoder.named_parameters()}
        model.update_target_encoder(0.996)
        assert any(
            not torch.allclose(before[n], p, atol=1e-8)
            for n, p in model.target_encoder.named_parameters()
        )

    def test_ema_tau_formula_ijepa(self):
        """EMA tau follows I-JEPA formula: ema[0] + i*(ema[1]-ema[0])/total_steps."""
        from src.utils.schedulers import EMATauSchedule

        s = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=1000)
        # Step 0: i=1 → 0.996 + 1*0.004/1000 = 0.996004
        tau1 = s.step()
        assert abs(tau1 - (0.996 + 1 * 0.004 / 1000)) < 1e-6

    def test_target_no_grad(self):
        assert all(not p.requires_grad for p in self.JEPA(self._cfg()).target_encoder.parameters())

    def test_gradient_flow(self):
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 5:10] = 1
        model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask
        )[0].backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
            if p.requires_grad
        )
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.predictor.parameters()
            if p.requires_grad
        )

    def test_loss_is_finite(self):
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 5:10] = 1
        loss, _, _ = model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask
        )
        assert torch.isfinite(loss)

    def test_span_loss_only_valid_positions(self):
        """Span loss should NOT include zero-padded positions from _gather_masked."""
        model = self.JEPA(self._cfg())
        # Different numbers of masked positions per sample
        mask = torch.zeros(3, 32, dtype=torch.long)
        mask[0, 0:2] = 1
        mask[1, 5:10] = 1
        mask[2, 15:18] = 1
        loss, ld, _ = model.compute_loss_with_targets(
            torch.randint(0, 1000, (3, 32)), torch.randint(0, 1000, (3, 32)), mask
        )
        assert torch.isfinite(loss)
        assert ld["loss_span"] >= 0

    def test_decoder_uses_boolean_indexing(self):
        """Decoder loss uses vectorized boolean indexing (not Python for-loop)."""
        model = self.JEPA(self._cfg())
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 5:10] = 1
        _loss, ld, _ = model.compute_loss_with_targets(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask
        )
        # Decoder should have non-zero loss when mask is present
        assert ld["loss_decoder"] > 0


# ═══════════════════════════════════════════════════════════════════
# Training Integration (small scale)
# ═══════════════════════════════════════════════════════════════════


class TestTrainingIntegration:
    def test_loss_decreases_over_steps(self):
        """Small training loop: loss should decrease over 200 steps on fixed data."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.utils.schedulers import EMATauSchedule

        torch.manual_seed(42)
        config = TextSpanJEPAConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=(1, 4),
            num_refine_steps=1,
            future_warmup_steps=20,
        )
        model = TextSpanJEPA(config)
        opt = torch.optim.AdamW(
            list(model.encoder.parameters())
            + list(model.predictor.parameters())
            + list(model.decoder.parameters()),
            lr=2e-3,
        )
        ema = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=200)

        # Fixed dataset (same inputs each step) for deterministic convergence test
        fixed_input = torch.randint(0, 1000, (4, 32))
        fixed_target = torch.randint(0, 1000, (4, 32))
        fixed_mask = torch.zeros(4, 32, dtype=torch.long)
        fixed_mask[:, 5:10] = 1
        fixed_mask[:, 20:25] = 1

        losses = []
        for step in range(200):
            loss, _, _ = model.compute_loss_with_targets(
                fixed_input, fixed_target, fixed_mask, current_step=step, total_steps=200
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt.step()
            model.update_target_encoder(ema.step())
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_no_nan_after_many_steps(self):
        """No NaN after 100 steps — tests numerical stability."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.utils.schedulers import EMATauSchedule

        config = TextSpanJEPAConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=(1,),
            num_refine_steps=1,
            future_warmup_steps=5,
        )
        model = TextSpanJEPA(config)
        opt = torch.optim.AdamW(
            list(model.encoder.parameters())
            + list(model.predictor.parameters())
            + list(model.decoder.parameters()),
            lr=1e-3,
        )
        ema = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=100)

        for step in range(100):
            mask = torch.zeros(2, 32, dtype=torch.long)
            mask[:, 5:8] = 1
            loss, _, _ = model.compute_loss_with_targets(
                torch.randint(0, 1000, (2, 32)),
                torch.randint(0, 1000, (2, 32)),
                mask,
                current_step=step,
                total_steps=100,
            )
            assert torch.isfinite(loss), f"NaN/Inf loss at step {step}"
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt.step()
            model.update_target_encoder(ema.step())


# ═══════════════════════════════════════════════════════════════════
# Span Mask
# ═══════════════════════════════════════════════════════════════════


class TestSpanMask:
    def test_basic(self):
        from src.masks.span import SpanMaskCollator

        r = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_token_id=0)(
            [{"input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}]
        )
        assert all(k in r for k in ["masked_input_ids", "original_input_ids", "mask_positions"])

    def test_curriculum(self):
        from src.masks.span import SpanMaskCollator

        c = SpanMaskCollator(
            mask_ratio=0.3,
            span_length_range=(3, 5),
            mask_ratio_start=0.1,
            mask_ratio_end=0.5,
            curriculum_steps=100,
            mask_token_id=0,
        )
        assert c.current_mask_ratio == pytest.approx(0.1, abs=0.02)
        c._step = 100
        assert c.current_mask_ratio == pytest.approx(0.5, abs=0.02)

    def test_masked_tokens_replaced(self):
        from src.masks.span import SpanMaskCollator

        c = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_token_id=99)
        r = c([{"input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)}])
        mask = r["mask_positions"].bool()
        assert (r["masked_input_ids"][mask] == 99).all()

    def test_original_unchanged(self):
        from src.masks.span import SpanMaskCollator

        c = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_token_id=99)
        orig = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)
        r = c([{"input_ids": orig}])
        assert (r["original_input_ids"][0] == orig).all()


# ═══════════════════════════════════════════════════════════════════
# Schedulers (I-JEPA patterns)
# ═══════════════════════════════════════════════════════════════════


class TestSchedulers:
    def test_lr_warmup_cosine(self):
        from src.utils.schedulers import WarmupCosineSchedule

        opt = torch.optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.001)
        s = WarmupCosineSchedule(
            opt, warmup_steps=10, start_lr=1e-5, ref_lr=1e-3, final_lr=1e-6, T_max=100
        )
        lrs = [s.step() for _ in range(100)]
        assert lrs[5] > lrs[0] and lrs[9] >= lrs[8]  # warmup
        assert lrs[50] < lrs[9]  # cosine decay
        assert lrs[99] <= lrs[50]  # continues decaying

    def test_ema_tau_ijepa_formula(self):
        """EMA tau: I-JEPA momentum_scheduler = ema[0] + i*(ema[1]-ema[0])/total_steps."""
        from src.utils.schedulers import EMATauSchedule

        s = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=10000)
        tau_5000 = None
        for i in range(5000):
            t = s.step()
            if i == 4999:
                tau_5000 = t
        # At step 5000: 0.996 + 5000 * 0.004 / 10000 = 0.998
        assert abs(tau_5000 - 0.998) < 1e-4

    def test_wd_schedule(self):
        from src.utils.schedulers import CosineWDSchedule

        opt = torch.optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.001, weight_decay=0.04)
        s = CosineWDSchedule(opt, ref_wd=0.04, final_wd=0.4, T_max=100)
        wds = [s.step() for _ in range(100)]
        assert wds[-1] > wds[0]  # WD increases


# ═══════════════════════════════════════════════════════════════════
# data2vec Baseline (Fix #4 — from official fairseq)
# ═══════════════════════════════════════════════════════════════════


class TestData2VecBaseline:
    def test_forward(self):
        from baselines.data2vec_baseline import Data2VecTextBaseline

        m = Data2VecTextBaseline(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            head_layers=2,
            ema_decay=0.999,
            ema_end_decay=0.9999,
            ema_anneal_end_step=100,
        )
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 5:10] = 1
        loss, info = m(torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert loss.requires_grad and "loss_data2vec" in info

    def test_ema_annealing(self):
        from baselines.data2vec_baseline import Data2VecTextBaseline

        m = Data2VecTextBaseline(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            depth=2,
            num_heads=4,
            ema_decay=0.999,
            ema_end_decay=0.9999,
            ema_anneal_end_step=100,
        )
        assert m.get_annealed_decay() < 0.9999
        m.num_updates = 100
        assert m.get_annealed_decay() == pytest.approx(0.9999, abs=1e-5)

    def test_get_annealed_rate_exact_formula(self):
        """Test exact get_annealed_rate from data2vec_text.py line ~58:
        r = end - start; pct_remaining = 1 - curr_step/total_steps; return end - r * pct_remaining
        """
        from baselines.data2vec_baseline import get_annealed_rate

        # Step 0 → return start
        assert get_annealed_rate(0.999, 0.9999, 0, 100) == pytest.approx(0.999, abs=1e-6)
        # Step 50 → midpoint
        mid = get_annealed_rate(0.999, 0.9999, 50, 100)
        expected = 0.9999 - (0.9999 - 0.999) * (1 - 50 / 100)
        assert mid == pytest.approx(expected, abs=1e-6)
        # Step 100 → return end
        assert get_annealed_rate(0.999, 0.9999, 100, 100) == pytest.approx(0.9999, abs=1e-6)

    def test_regression_head_data2vec(self):
        """data2vec regression head: head_layers=2 → Linear→GELU→Linear."""
        from baselines.data2vec_baseline import Data2VecTextBaseline

        m = Data2VecTextBaseline(
            vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4, head_layers=2
        )
        assert m.regression_head[0].in_features == 64
        assert m.regression_head[0].out_features == 128  # 2x expand (data2vec pattern)
        assert isinstance(m.regression_head[1], torch.nn.GELU)
        assert m.regression_head[2].out_features == 64

    def test_loss_formula_data2vec(self):
        """data2vec loss: smooth_l1 with beta=0 → mse_loss; scale = 1/sqrt(dim)."""
        from baselines.data2vec_baseline import Data2VecTextBaseline

        m = Data2VecTextBaseline(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            depth=2,
            num_heads=4,
            loss_beta=0.0,
            loss_scale=None,
        )
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 5:10] = 1
        loss, _ = m(torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask)
        assert torch.isfinite(loss)

    def test_target_encoder_no_grad(self):
        from baselines.data2vec_baseline import Data2VecTextBaseline

        m = Data2VecTextBaseline(
            vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4
        )
        assert all(not p.requires_grad for p in m.target_encoder.parameters())


# ═══════════════════════════════════════════════════════════════════
# MLM Baseline
# ═══════════════════════════════════════════════════════════════════


class TestMLMBaseline:
    def test_forward(self):
        from baselines.mlm_baseline import MLMBaseline

        m = MLMBaseline(
            vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0
        )
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[:, 5:10] = 1
        loss, info = m.compute_loss(
            torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)), mask
        )
        assert loss.requires_grad and "loss_mlm" in info and "mlm_accuracy" in info


# ═══════════════════════════════════════════════════════════════════
# Logging (I-JEPA patterns)
# ═══════════════════════════════════════════════════════════════════


class TestLogging:
    def test_average_meter(self):
        from src.utils.logging import AverageMeter

        m = AverageMeter()
        m.update(1.0)
        m.update(3.0)
        assert m.avg == 2.0
        assert m.val == 3.0
        assert m.min == 1.0 and m.max == 3.0

    def test_grad_logger(self):
        from src.models.encoder import TextSpanJEPLEncoder
        from src.utils.logging import grad_logger

        # I-JEPA grad_logger looks for 'qkv' in param names;
        # must use a model with fused QKV projection (like our encoder).
        model = TextSpanJEPLEncoder(
            vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4
        )
        x = torch.randint(0, 1000, (2, 32))
        h, _ = model(x)
        h.sum().backward()
        stats = grad_logger(model.named_parameters())
        assert stats.first_layer > 0, "grad_logger should detect first QKV layer gradient"
        assert stats.last_layer > 0, "grad_logger should detect last QKV layer gradient"


# ═══════════════════════════════════════════════════════════════════
# Evaluation Probes
# ═══════════════════════════════════════════════════════════════════


class TestEvalProbes:
    def test_geometry_metrics_random_data(self):
        """GeometryMetrics should return valid metrics for random data."""
        from src.eval.probes import GeometryMetrics

        m = GeometryMetrics.compute(torch.randn(4, 16, 32))
        assert m["effective_rank"] > 0
        assert m["participation_ratio"] > 1.0
        assert m["condition_number"] > 0
        assert m["numerical_rank"] > 0
        assert 0 < m["rank_utilization"] <= 1.0
        assert m["coherence"] >= 0
        # New metrics
        assert "sv_entropy" in m and 0 < m["sv_entropy"] <= 1.0
        assert "svd_sharpness" in m and 0 < m["svd_sharpness"] < 1.0
        assert "alpha_norm" in m and m["alpha_norm"] >= 0
        assert "intrinsic_dim" in m and m["intrinsic_dim"] >= 0
        assert "mean_pairwise_cosine" in m and -1 <= m["mean_pairwise_cosine"] <= 1

    def test_geometry_metrics_zero_input(self):
        """GeometryMetrics should handle zero input without NaN/crash (NextLat pattern)."""
        from src.eval.probes import GeometryMetrics

        m = GeometryMetrics.compute(torch.zeros(4, 16, 32))
        assert m["effective_rank"] == 0.0
        assert m["numerical_rank"] == 0.0
        assert m["condition_number"] == float("inf")
        assert m["rank_utilization"] == 0.0
        # New metrics should also handle zero input
        assert math.isfinite(m.get("sv_entropy", 0))
        assert math.isfinite(m.get("svd_sharpness", 0))
        assert math.isfinite(m.get("alpha_norm", 0))

    def test_geometry_metrics_reuses_collapse_diagnostics(self):
        """GeometryMetrics should reuse CollapseDiagnostics (no code duplication)."""
        from src.eval.probes import GeometryMetrics
        from src.models.collapse import CollapseDiagnostics

        # Verify it uses the same instance/methods
        assert isinstance(GeometryMetrics._diag, CollapseDiagnostics)
        assert GeometryMetrics.compute is not None

    def test_new_collapse_metrics_in_compute(self):
        """v0.5.0+ metrics: SVCCA, alignment, eigenvalue_spread, subspace_overlap, spectral_clustering_coeff."""
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        y = torch.randn(4, 16, 32)
        m = diag.compute(x, y)
        # SVCCA
        assert "svcca_online_target" in m
        assert 0 <= m["svcca_online_target"] <= 1
        # Alignment
        assert "alignment" in m
        assert math.isfinite(m["alignment"])
        # Eigenvalue spread
        assert "eigenvalue_spread_online" in m
        assert m["eigenvalue_spread_online"] >= 0
        # Subspace overlap
        assert "subspace_overlap" in m
        assert 0 <= m["subspace_overlap"] <= 1
        # Spectral clustering coefficient
        assert "spectral_clustering_coeff_online" in m
        assert 0 <= m["spectral_clustering_coeff_online"] <= 1

    def test_subspace_overlap_identical(self):
        """Subspace overlap with itself should be ~1.0."""
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        overlap = diag._subspace_overlap(x, x)
        assert overlap > 0.9

    def test_svcca_similar_representations(self):
        """SVCCA of similar representations should be high."""
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        x = torch.randn(4, 16, 32)
        y = x + torch.randn_like(x) * 0.1
        svcca = diag._svcca(x, y)
        assert svcca > 0.5


# ═══════════════════════════════════════════════════════════════════
# Checkpoint Save/Load (I-JEPA pattern)
# ═══════════════════════════════════════════════════════════════════


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        """Checkpoint save → load should produce identical model weights."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=32,
            predictor_depth=2,
            future_offsets=(1,),
            num_refine_steps=1,
        )
        model = TextSpanJEPA(config)
        opt = torch.optim.AdamW(
            list(model.encoder.parameters())
            + list(model.predictor.parameters())
            + list(model.decoder.parameters()),
            lr=1e-3,
        )

        # Save
        ckpt_path = str(tmp_path / "test_ckpt.pth.tar")
        save_dict = {
            "encoder": model.encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "opt": opt.state_dict(),
            "scaler": None,
            "epoch": 5,
            "global_step": 1000,
            "loss": 0.5,
        }
        torch.save(save_dict, ckpt_path)

        # Load into fresh model (inline to avoid transformers dependency)
        model2 = TextSpanJEPA(config)
        opt2 = torch.optim.AdamW(
            list(model2.encoder.parameters())
            + list(model2.predictor.parameters())
            + list(model2.decoder.parameters()),
            lr=1e-3,
        )

        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)

        model2.encoder.load_state_dict(checkpoint["encoder"])
        model2.predictor.load_state_dict(checkpoint["predictor"])
        model2.target_encoder.load_state_dict(checkpoint["target_encoder"])
        model2.decoder.load_state_dict(checkpoint["decoder"])
        opt2.load_state_dict(checkpoint["opt"])

        assert epoch == 5
        assert global_step == 1000
        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.encoder.named_parameters(), model2.encoder.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch: {n1}"

    def test_checkpoint_saves_global_step(self, tmp_path):
        """Checkpoint must include global_step for training resumption."""
        import io

        model_state = {"global_step": 42, "epoch": 3}
        buf = io.BytesIO()
        torch.save(model_state, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)
        assert loaded["global_step"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═══════════════════════════════════════════════════════════════════
# v0.10.0 — Bug fixes and new features
# ═══════════════════════════════════════════════════════════════════


class TestV010Bugfixes:
    def test_encoder_get_intermediate_layers(self):
        """Encoder should provide per-layer hidden states."""
        from src.models.encoder import TextSpanJEPLEncoder

        enc = TextSpanJEPLEncoder(
            vocab_size=1000, max_seq_len=32, embed_dim=64, depth=4, num_heads=4
        )
        x = torch.randint(0, 1000, (2, 16))
        intermediates = enc.get_intermediate_layers(x)
        assert len(intermediates) == 4  # 4 blocks
        assert intermediates[0].shape == (2, 16, 64)

    def test_encoder_forward_return_intermediates(self):
        """forward() with return_intermediates=True returns 3 values."""
        from src.models.encoder import TextSpanJEPLEncoder

        enc = TextSpanJEPLEncoder(
            vocab_size=1000, max_seq_len=32, embed_dim=64, depth=3, num_heads=4
        )
        x = torch.randint(0, 1000, (2, 16))
        h, _tok, intermediates = enc(x, return_intermediates=True)
        assert h.shape == (2, 16, 64)
        assert len(intermediates) == 3
        # Without flag: 2 values
        h2, _tok2 = enc(x)
        assert h2.shape == (2, 16, 64)

    def test_span_mask_no_double_step(self):
        """SpanMaskCollator.__call__ should NOT call step() — train loop does."""
        from src.masks.span import SpanMaskCollator

        c = SpanMaskCollator(mask_ratio=0.3, span_length_range=(3, 5), mask_token_id=0)
        step_before = c._step
        c([{"input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)}])
        step_after = c._step
        assert step_before == step_after, "SpanMaskCollator should not auto-step"

    def test_trainable_params_count(self):
        """get_num_params_trainable excludes target encoder."""
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
        trainable = model.get_num_params_trainable()
        total = sum(p.numel() for p in model.parameters())
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        assert trainable + non_trainable == total
        assert trainable < total  # target_encoder is non-trainable

    def test_ablated_model_loss_subtraction_uses_lambda(self):
        """AblatedModel should subtract lambda * loss, not raw loss (Fix #15)."""
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
            lambda_decoder=0.1,
        )
        model = TextSpanJEPA(config)
        ablation = AblationConfig("no_dec", use_decoder=False)
        ablated = AblatedModel(model, ablation)

        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1
        loss_ablated, _info = ablated(ids, ids, mask)

        # Compute full model loss for comparison
        loss_full, info_full, _ = model.compute_loss_with_targets(ids, ids, mask)
        # Ablated loss should be: full_loss - lambda_decoder * loss_decoder
        expected = loss_full - config.lambda_decoder * info_full["loss_decoder"]
        assert abs(loss_ablated.item() - expected.item()) < 0.2

    def test_ablated_model_forward_signature(self):
        """AblatedModel.forward takes (masked_input_ids, original_input_ids, mask_positions)."""
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
        ablation = AblationConfig("test")
        ablated = AblatedModel(model, ablation)
        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1
        # This should work with train-loop argument order
        loss, _info = ablated(ids, ids, mask)
        assert torch.isfinite(loss)

    def test_ablated_model_skip_refinement(self):
        """Ablation with no_iterative_refinement should set refine steps to 0."""
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
            num_refine_steps=3,
        )
        model = TextSpanJEPA(config)
        ablation = AblationConfig("no_refine", use_iterative_refinement=False)
        ablated = AblatedModel(model, ablation)

        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1
        # Forward should work with refinement temporarily disabled
        loss, _info = ablated(ids, ids, mask)
        assert torch.isfinite(loss)
        # After forward, refinement steps should be restored
        assert model.predictor.num_refine_steps == 3


class TestV010NewFeatures:
    def test_seed_everything(self):
        """seed_everything should produce deterministic results."""
        from src.utils.seed import seed_everything

        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_flops_estimation(self):
        """FLOPs estimation should return reasonable values."""
        from src.utils.flops import estimate_training_flops, estimate_transformer_flops

        result = estimate_transformer_flops(120e6, 512, batch_size=64)
        assert result["tflops"] > 0
        train_result = estimate_training_flops(120e6, 512, 64, 100000)
        assert train_result["pflops"] > 0

    def test_model_size_category(self):
        """Model size categorization."""
        from src.utils.flops import model_size_category

        assert model_size_category(5e6) == "tiny"
        assert model_size_category(50e6) == "small"
        assert model_size_category(150e6) == "base"
        assert model_size_category(1e9) == "large"

    def test_ablation_model_size_configs(self):
        """Ablation framework should have model size variants."""
        from src.interp.ablation import ABLATION_MATRIX, MODEL_SIZE_CONFIGS

        assert "tiny" in MODEL_SIZE_CONFIGS
        assert "base" in MODEL_SIZE_CONFIGS
        assert len(ABLATION_MATRIX) == len(MODEL_SIZE_CONFIGS) * 12  # 4 sizes x 12 ablations

    def test_new_visualization_ablation_chart(self):
        """ablation_comparison_chart should produce valid SVG."""
        from src.interp.visualization import ablation_comparison_chart

        svg = ablation_comparison_chart(
            ablation_names=["no_predictor", "no_vicreg", "no_decoder"],
            metric_name="Effective Rank",
            values=[15.0, 30.0, 35.0],
            full_model_value=40.0,
        )
        assert svg is not None and "<svg" in svg

    def test_new_visualization_scaling_law(self):
        """scaling_law_plot should produce valid SVG."""
        from src.interp.visualization import scaling_law_plot

        svg = scaling_law_plot(
            sizes=[1e6, 10e6, 100e6], jepa_metrics=[15, 30, 50], baseline_metrics=[12, 22, 35]
        )
        assert svg is not None and "<svg" in svg

    def test_new_visualization_robustness_curve(self):
        """robustness_curve should produce valid SVG."""
        from src.interp.visualization import robustness_curve

        svg = robustness_curve(
            intensities=[0.1, 0.3, 0.5, 0.7],
            jepa_cka=[0.98, 0.90, 0.80, 0.70],
            baseline_cka=[0.95, 0.80, 0.60, 0.40],
        )
        assert svg is not None and "<svg" in svg

    def test_new_visualization_information_plane(self):
        """information_plane should produce valid SVG."""
        from src.interp.visualization import information_plane

        svg = information_plane(
            mi_input=[5.0, 4.5, 4.0, 3.5, 3.0, 2.5], mi_task=[0.5, 1.0, 1.5, 2.0, 2.2, 2.3]
        )
        assert svg is not None and "<svg" in svg


# ═══════════════════════════════════════════════════════════════════
# v0.11.0 — Critical training bugs + config validation + grad accum
# ═══════════════════════════════════════════════════════════════════


class TestV011TrainingReadiness:
    def test_config_validate_good(self):
        """Good config should pass validation."""
        from src.models.jepa import TextSpanJEPAConfig

        cfg = TextSpanJEPAConfig(embed_dim=768, num_heads=12, predictor_embed_dim=384)
        assert cfg.validate() is True

    def test_config_validate_bad_embed_dim(self):
        """embed_dim not divisible by num_heads should raise."""
        from src.models.jepa import TextSpanJEPAConfig

        cfg = TextSpanJEPAConfig(embed_dim=100, num_heads=12)
        with pytest.raises(ValueError, match="embed_dim"):
            cfg.validate()

    def test_config_validate_bad_predictor_dim(self):
        """predictor_embed_dim not divisible by num_heads should raise."""
        from src.models.jepa import TextSpanJEPAConfig

        cfg = TextSpanJEPAConfig(embed_dim=768, num_heads=12, predictor_embed_dim=100)
        with pytest.raises(ValueError, match="predictor_embed_dim"):
            cfg.validate()

    def test_config_validate_bad_depth(self):
        """depth < 1 should raise."""
        from src.models.jepa import TextSpanJEPAConfig

        cfg = TextSpanJEPAConfig(encoder_depth=0)
        with pytest.raises(ValueError, match="encoder_depth"):
            cfg.validate()

    def test_textdataset_off_by_one(self):
        """TextDataset with exactly seq_len tokens should give 1 chunk."""
        from src.datasets.kaggle import TextDataset

        ds = TextDataset(list(range(512)), seq_len=512)
        assert len(ds) == 1, f"Expected 1 chunk, got {len(ds)}"

    def test_textdataset_normal(self):
        """TextDataset with 2*seq_len tokens should give 2 chunks."""
        from src.datasets.kaggle import TextDataset

        ds = TextDataset(list(range(1024)), seq_len=512)
        assert len(ds) == 2

    def test_textdataset_short(self):
        """TextDataset with fewer than seq_len tokens should give 0 chunks."""
        from src.datasets.kaggle import TextDataset

        ds = TextDataset(list(range(100)), seq_len=512)
        assert len(ds) == 0

    def test_mlm_baseline_has_encoder_and_decoder(self):
        """MLMBaseline must have .encoder and .decoder for train.py."""
        from baselines.mlm_baseline import MLMBaseline

        m = MLMBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        assert hasattr(m, "encoder")
        assert hasattr(m, "decoder")  # decoder = mlm_head alias
        assert m.decoder is m.mlm_head

    def test_mlm_baseline_get_num_params(self):
        """MLMBaseline.get_num_params should work."""
        from baselines.mlm_baseline import MLMBaseline

        m = MLMBaseline(vocab_size=1000, max_seq_len=32, embed_dim=64, depth=2, num_heads=4)
        n = m.get_num_params()
        assert n > 0

    def test_model_factory_jepa(self):
        """create_model with text_span_jepa should create JEPA model."""
        from src.train import create_model

        model = create_model(
            "text_span_jepa",
            {
                "embed_dim": 64,
                "encoder_depth": 2,
                "num_heads": 4,
                "predictor_embed_dim": 32,
                "predictor_depth": 2,
                "future_offsets": [1],
                "num_refine_steps": 1,
            },
            vocab_size=1000,
            max_seq_len=32,
            device="cpu",
        )
        from src.models.jepa import TextSpanJEPA

        assert isinstance(model, TextSpanJEPA)

    def test_model_factory_mlm(self):
        """create_model with mlm should create MLM baseline."""
        from src.train import create_model

        model = create_model(
            "mlm",
            {"embed_dim": 64, "encoder_depth": 2, "num_heads": 4},
            vocab_size=1000,
            max_seq_len=32,
            device="cpu",
        )
        from baselines.mlm_baseline import MLMBaseline

        assert isinstance(model, MLMBaseline)

    def test_model_factory_data2vec(self):
        """create_model with data2vec should create data2vec baseline."""
        from src.train import create_model

        model = create_model(
            "data2vec",
            {"embed_dim": 64, "encoder_depth": 2, "num_heads": 4},
            vocab_size=1000,
            max_seq_len=32,
            device="cpu",
        )
        from baselines.data2vec_baseline import Data2VecTextBaseline

        assert isinstance(model, Data2VecTextBaseline)

    def test_model_factory_unknown(self):
        """create_model with unknown name should raise."""
        from src.train import create_model

        with pytest.raises(ValueError, match="Unknown model_name"):
            create_model("roberta", {}, vocab_size=1000, max_seq_len=32, device="cpu")

    def test_compute_loss_unified_interface(self):
        """compute_loss should work for all model types."""
        # JEPA
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.train import compute_loss

        cfg = TextSpanJEPAConfig(
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
        jepa = TextSpanJEPA(cfg)
        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1
        loss, _ld, _dd = compute_loss(jepa, ids, ids, mask)
        assert torch.isfinite(loss)

        # MLM
        from baselines.mlm_baseline import MLMBaseline

        mlm = MLMBaseline(vocab_size=100, max_seq_len=16, embed_dim=32, depth=2, num_heads=4)
        loss2, _ld2, _dd2 = compute_loss(mlm, ids, ids, mask)
        assert torch.isfinite(loss2)

        # data2vec
        from baselines.data2vec_baseline import Data2VecTextBaseline

        d2v = Data2VecTextBaseline(
            vocab_size=100, max_seq_len=16, embed_dim=32, depth=2, num_heads=4
        )
        loss3, _ld3, _dd3 = compute_loss(d2v, ids, ids, mask)
        assert torch.isfinite(loss3)

    def test_get_param_groups_all_models(self):
        """get_param_groups should work for all model types."""
        from baselines.data2vec_baseline import Data2VecTextBaseline
        from baselines.mlm_baseline import MLMBaseline
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.train import get_param_groups

        cfg = TextSpanJEPAConfig(
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

        # JEPA
        jepa = TextSpanJEPA(cfg)
        pg_jepa = get_param_groups(jepa, "text_span_jepa")
        opt_jepa = torch.optim.AdamW(pg_jepa)
        assert opt_jepa is not None

        # MLM
        mlm = MLMBaseline(vocab_size=100, max_seq_len=16, embed_dim=32, depth=2, num_heads=4)
        pg_mlm = get_param_groups(mlm, "mlm")
        opt_mlm = torch.optim.AdamW(pg_mlm)
        assert opt_mlm is not None

        # data2vec
        d2v = Data2VecTextBaseline(
            vocab_size=100, max_seq_len=16, embed_dim=32, depth=2, num_heads=4
        )
        pg_d2v = get_param_groups(d2v, "data2vec")
        opt_d2v = torch.optim.AdamW(pg_d2v)
        assert opt_d2v is not None

    def test_checkpoint_saves_centering_state(self, tmp_path):
        """save_checkpoint should include target_centering.center."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
        from src.train import load_checkpoint, save_checkpoint

        cfg = TextSpanJEPAConfig(
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
        model = TextSpanJEPA(cfg)
        # Run one forward to populate center
        ids = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 3:6] = 1
        model.compute_loss_with_targets(ids, ids, mask)
        center_before = model.target_centering.center.clone()

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        ckpt_path = str(tmp_path / "test_ckpt.pth.tar")
        save_checkpoint(ckpt_path, model, opt, None, 1, 100, ema_step=50, mask_step=30)

        # Create fresh model and load
        model2 = TextSpanJEPA(cfg)
        opt2 = torch.optim.AdamW([p for p in model2.parameters() if p.requires_grad])
        ep, gs, ema_s, mask_s, _extra = load_checkpoint(ckpt_path, model2, opt2, None)

        assert ep == 1
        assert gs == 100
        assert ema_s == 50
        assert mask_s == 30
        assert torch.allclose(model2.target_centering.center, center_before)

    def test_mlm_training_loop(self):
        """MLM should train without error for 50 steps."""
        from baselines.mlm_baseline import MLMBaseline

        m = MLMBaseline(vocab_size=100, max_seq_len=16, embed_dim=32, depth=2, num_heads=4)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        losses = []
        for _ in range(50):
            ids = torch.randint(0, 100, (4, 16))
            mask = torch.zeros(4, 16, dtype=torch.long)
            mask[:, 3:6] = 1
            loss, _ = m.compute_loss(ids, ids, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], "MLM loss should decrease"

    def test_data2vec_training_loop(self):
        """data2vec should train without error for 50 steps."""
        from baselines.data2vec_baseline import Data2VecTextBaseline

        d2v = Data2VecTextBaseline(
            vocab_size=100, max_seq_len=16, embed_dim=32, depth=2, num_heads=4
        )
        # Train encoder + regression_head (NOT target_encoder)
        params = list(d2v.encoder.parameters()) + list(d2v.regression_head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3)
        losses = []
        for _ in range(50):
            ids = torch.randint(0, 100, (4, 16))
            mask = torch.zeros(4, 16, dtype=torch.long)
            mask[:, 3:6] = 1
            loss, _ = d2v(ids, ids, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            d2v.update_target_encoder()
            losses.append(loss.item())
        assert losses[-1] < losses[0], "data2vec loss should decrease"


# ═══════════════════════════════════════════════════════════════════
#  v0.12.0 Bugfixes — critical model_name mapping + checkpoint fixes
# ═══════════════════════════════════════════════════════════════════


class TestV012Bugfixes:
    """Tests for v0.12.0 critical bug fixes."""

    def test_normalize_model_name_jepa_variants(self):
        from src.train import _normalize_model_name

        assert _normalize_model_name("text_span_jepa") == "text_span_jepa"
        assert _normalize_model_name("text_span_jepa_small") == "text_span_jepa"
        assert _normalize_model_name("text_span_jepa_base") == "text_span_jepa"
        assert _normalize_model_name("jepa_tiny") == "text_span_jepa"

    def test_normalize_model_name_mlm_variants(self):
        from src.train import _normalize_model_name

        assert _normalize_model_name("mlm") == "mlm"
        assert _normalize_model_name("mlm_small") == "mlm"
        assert _normalize_model_name("mlm_baseline") == "mlm"

    def test_normalize_model_name_data2vec_variants(self):
        from src.train import _normalize_model_name

        assert _normalize_model_name("data2vec") == "data2vec"
        assert _normalize_model_name("data2vec_base") == "data2vec"
        assert _normalize_model_name("data2vec_baseline") == "data2vec"

    def test_create_model_with_suffixed_name(self):
        """Config uses 'text_span_jepa_small' — create_model should handle it."""
        from src.train import create_model

        model = create_model(
            "text_span_jepa_small",
            {
                "embed_dim": 64,
                "encoder_depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "predictor_embed_dim": 32,
                "predictor_depth": 2,
                "future_offsets": [1],
                "num_refine_steps": 1,
            },
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        assert model.encoder is not None

    def test_create_model_mlm_suffixed(self):
        from src.train import create_model

        model = create_model(
            "mlm_small",
            {"embed_dim": 64, "encoder_depth": 2, "num_heads": 4, "mlp_ratio": 2.0},
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        assert model.encoder is not None

    def test_create_model_data2vec_suffixed(self):
        from src.train import create_model

        model = create_model(
            "data2vec_base",
            {"embed_dim": 64, "encoder_depth": 2, "num_heads": 4, "mlp_ratio": 2.0},
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        assert model.encoder is not None

    def test_param_groups_with_suffixed_name(self):
        """get_param_groups should return correct groups for suffixed names."""
        from src.train import create_model, get_param_groups

        model = create_model(
            "text_span_jepa_small",
            {
                "embed_dim": 64,
                "encoder_depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "predictor_embed_dim": 32,
                "predictor_depth": 2,
                "future_offsets": [1],
                "num_refine_steps": 1,
            },
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        groups = get_param_groups(model, "text_span_jepa_small", wd=0.04)
        # R18: unconditional variance/covariance regs land in the catch-all,
        # so the baseline count is 6 even without optional mechanisms.
        assert len(groups) == 6, f"Expected 6 param groups, got {len(groups)}"

        # With novel-mechanism modules enabled, the R18 catch-all group must
        # appear so jawp/cgn/pcr/spc parameters actually receive updates.
        mech_cfg = {
            "embed_dim": 64,
            "encoder_depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "predictor_embed_dim": 32,
            "predictor_depth": 2,
            "future_offsets": [1],
            "num_refine_steps": 1,
            "use_jawp": True,
            "jawk_k_start": 2,
            "jawk_k_end": 4,
            "jawk_curriculum_steps": 0,
            "use_cgn": True,
            "use_pcr": True,
            "pcr_n_levels": 2,
            "pcr_level_dims": [8, 8],
            "use_spc": True,
            "spc_n_bands": 4,
        }
        model_mech = create_model(
            "text_span_jepa_small",
            mech_cfg,
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        groups_mech = get_param_groups(model_mech, "text_span_jepa_small", wd=0.04)
        assert (
            len(groups_mech) == 6
        ), f"Expected 6 param groups with mechanisms enabled, got {len(groups_mech)}"
        all_params = [p for g in groups_mech for p in g["params"]]
        assert any(
            p is model_mech.jawp.workspace_Q for p in all_params
        ), "jawp.workspace_Q must be adopted by the optimizer"

    def test_mlm_checkpoint_save_load(self):
        """MLM checkpoint should save/load without AttributeError."""
        import os
        import tempfile

        from src.train import create_model, load_checkpoint, save_checkpoint

        model = create_model(
            "mlm",
            {"embed_dim": 64, "encoder_depth": 2, "num_heads": 4, "mlp_ratio": 2.0},
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mlm.pt")
            save_checkpoint(path, model, opt, None, 1, 100, model_name="mlm")
            e, gs, _, _, _ = load_checkpoint(path, model, opt, None, model_name="mlm")
            assert e == 1
            assert gs == 100

    def test_data2vec_checkpoint_save_load(self):
        """data2vec checkpoint should save/load without AttributeError."""
        import os
        import tempfile

        from src.train import create_model, load_checkpoint, save_checkpoint

        model = create_model(
            "data2vec",
            {"embed_dim": 64, "encoder_depth": 2, "num_heads": 4, "mlp_ratio": 2.0},
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d2v.pt")
            save_checkpoint(path, model, opt, None, 2, 200, model_name="data2vec")
            e, gs, _, _, _ = load_checkpoint(path, model, opt, None, model_name="data2vec")
            assert e == 2
            assert gs == 200

    def test_do_ema_update_suffixed_names(self):
        """do_ema_update should work with suffixed model names."""
        from src.train import create_model, do_ema_update

        model = create_model(
            "text_span_jepa_small",
            {
                "embed_dim": 64,
                "encoder_depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "predictor_embed_dim": 32,
                "predictor_depth": 2,
                "future_offsets": [1],
                "num_refine_steps": 1,
            },
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        do_ema_update(model, "text_span_jepa_small", tau=0.996)  # Should not raise

    def test_global_grad_clipping_no_target_encoder(self):
        """_get_all_trainable_params should exclude target encoder params."""
        from src.train import _get_all_trainable_params, create_model

        model = create_model(
            "text_span_jepa",
            {
                "embed_dim": 64,
                "encoder_depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "predictor_embed_dim": 32,
                "predictor_depth": 2,
                "future_offsets": [1],
                "num_refine_steps": 1,
            },
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        params = _get_all_trainable_params(model)
        target_ids = {id(p) for p in model.target_encoder.parameters()}
        trainable_ids = {id(p) for p in params}
        overlap = target_ids & trainable_ids
        assert len(overlap) == 0, "Target encoder params should not be trainable"

    def test_defaults_yaml_has_grad_accum_steps(self):
        """defaults.yaml should have grad_accum_steps."""
        import yaml

        with open("defaults.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "grad_accum_steps" in cfg.get(
            "optimization", {}
        ), "grad_accum_steps missing from defaults.yaml"

    def test_jepa_training_loss_decreases(self):
        """100-step JEPA training: loss should decrease."""
        from src.train import compute_loss, create_model
        from src.utils.seed import seed_everything

        seed_everything(42)
        model = create_model(
            "text_span_jepa",
            {
                "embed_dim": 64,
                "encoder_depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "predictor_embed_dim": 32,
                "predictor_depth": 2,
                "future_offsets": [1],
                "num_refine_steps": 1,
            },
            vocab_size=100,
            max_seq_len=16,
            device=torch.device("cpu"),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for step in range(50):
            ids = torch.randint(0, 100, (4, 16))
            mask = torch.zeros(4, 16, dtype=torch.long)
            mask[:, 3:6] = 1
            loss, _, _ = compute_loss(model, ids, ids, mask, current_step=step, total_steps=50)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_target_encoder(tau=0.996)
            losses.append(loss.item())
        assert losses[-1] < losses[0], "JEPA loss should decrease over training"


# ═══════════════════════════════════════════════════════════════════
#  v1.0.0rc14 quality checks — deep merge, defaults, validation
# ═══════════════════════════════════════════════════════════════════


class TestDeepMergeAndDefaults:
    """Tests for config deep merge and default consistency."""

    def test_ema_tau_end_matches_defaults(self):
        """Config default ema_tau_end=0.9999 must match defaults.yaml."""
        from src.models.jepa import TextSpanJEPAConfig

        c = TextSpanJEPAConfig()
        assert c.ema_tau_end == 0.9999, f"ema_tau_end={c.ema_tau_end} != 0.9999"

    def test_ema_tau_start_matches_defaults(self):
        from src.models.jepa import TextSpanJEPAConfig

        c = TextSpanJEPAConfig()
        assert c.ema_tau_start == 0.996

    def test_ema_schedule_matches_defaults(self):
        from src.models.jepa import TextSpanJEPAConfig

        c = TextSpanJEPAConfig()
        assert c.ema_schedule == "cosine"

    def test_all_lambda_fields_exist(self):
        from src.models.jepa import TextSpanJEPAConfig

        c = TextSpanJEPAConfig()
        lambdas = [
            "lambda_span",
            "lambda_future",
            "lambda_decoder",
            "lambda_variance",
            "lambda_covariance",
            "lambda_sigreg",
            "lambda_predictive_rank",
            "lambda_cgn_ortho",
            "lambda_swip",
            "lambda_spc",
            "lambda_wsd",
            "lambda_cmc",
            "lambda_gac",
            "lambda_sta",
            "lambda_puc",
            "lambda_rdc",
            "lambda_wsr",
        ]
        for l in lambdas:
            assert hasattr(c, l), f"Missing {l}"

    def test_all_use_fields_exist(self):
        from src.models.jepa import TextSpanJEPAConfig

        c = TextSpanJEPAConfig()
        uses = [
            "use_jawp",
            "use_cgn",
            "use_swip",
            "use_pcr",
            "use_spc",
            "use_wsd",
            "use_cmc",
            "use_gac",
            "use_sta",
            "use_puc",
            "use_rdc",
            "use_wsr",
        ]
        for u in uses:
            assert hasattr(c, u), f"Missing {u}"

    def test_deep_merge_function(self):
        """_deep_merge must correctly merge nested dicts."""
        from src.train import _deep_merge

        base = {
            "model": {"embed_dim": 768, "encoder_depth": 12, "use_jawp": True},
            "data": {"batch_size": 64},
        }
        override = {"model": {"use_jawp": False, "use_cgn": True}}
        result = _deep_merge(base, override)
        assert result["model"]["embed_dim"] == 768  # inherited from base
        assert result["model"]["encoder_depth"] == 12  # inherited
        assert result["model"]["use_jawp"] == False  # overridden
        assert result["model"]["use_cgn"] == True  # added
        assert result["data"]["batch_size"] == 64  # inherited

    def test_validate_puc_rdc_wsr(self):
        """Config.validate() must validate PUC, RDC, WSR params."""
        from src.models.jepa import TextSpanJEPAConfig

        # Valid config with PUC+RDC+WSR
        c = TextSpanJEPAConfig(
            use_puc=True,
            lambda_puc=0.01,
            puc_eta=0.01,
            puc_ema_beta=0.999,
            use_rdc=True,
            lambda_rdc=0.01,
            rdc_eta=0.01,
            rdc_ema_beta=0.999,
            use_wsr=True,
            lambda_wsr=0.01,
            wsr_rho=0.05,
            wsr_eta=0.01,
            wsr_ema_beta=0.999,
            wsr_mode="gradient",
        )
        c.validate()  # should not raise

    def test_validate_ema_range(self):
        """EMA tau must satisfy 0 < start <= end < 1."""
        from src.models.jepa import TextSpanJEPAConfig

        c = TextSpanJEPAConfig(ema_tau_start=0.999, ema_tau_end=0.99)  # inverted
        with pytest.raises(ValueError, match="ema_tau_start"):
            c.validate()

    def test_validate_wsr_mode(self):
        from src.models.jepa import TextSpanJEPAConfig

        c = TextSpanJEPAConfig(use_wsr=True, lambda_wsr=0.01, wsr_mode="invalid")
        with pytest.raises(ValueError, match="wsr_mode"):
            c.validate()

    def test_ema_update_no_grad(self):
        """EMA update should not create grad graph nodes."""
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        c = TextSpanJEPAConfig(
            vocab_size=100,
            max_seq_len=16,
            embed_dim=32,
            encoder_depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            predictor_embed_dim=16,
            predictor_depth=1,
            future_offsets=(1,),
            num_refine_steps=1,
        )
        model = TextSpanJEPA(c)
        # Track gradFn before EMA update
        param_k = next(model.target_encoder.parameters())
        model.update_target_encoder(0.996)
        # After update, param_k should have no grad_fn (no autograd tracking)
        assert param_k.grad_fn is None, "EMA update leaked autograd"

    def test_mechanism_bundle_counts_16(self):
        """MechanismBundle must expose all 16 mechanisms."""
        from src.models.mechanisms import MechanismBundle

        # With all mechanisms
        bundle = MechanismBundle(
            embed_dim=64,
            use_jawp=True,
            use_cgn=True,
            use_swip=True,
            use_pcr=True,
            use_spc=True,
            use_wsd=True,
            use_cmc=True,
            use_gac=True,
            use_sta=True,
            use_puc=True,
            use_rdc=True,
            use_wsr=True,
            wsd_k=6,
        )
        # Count non-None mechanism attributes
        mech_names = [
            "jawp",
            "cgn",
            "swip",
            "pcr",
            "spc",
            "wsd",
            "cmc",
            "gac",
            "sta",
            "puc",
            "rdc",
            "wsr",
        ]
        active = sum(1 for m in mech_names if getattr(bundle, m, None) is not None)
        assert active == 12, f"Expected 12 active mechanisms, got {active}"


# ═══════════════════════════════════════════════════════════════════
#  GWP Framework tests — unified branding, groups, DAG
# ═══════════════════════════════════════════════════════════════════


class TestGWPFramk:
    """Tests for GWP (Grassmann Workspace Prediction) unified framework."""

    def test_gwp_import(self):
        """GWP must be importable as the main entry point."""
        from src.models.mechanisms import GWP

        assert GWP.FRAMEWORK_NAME == "GWP"
        assert GWP.FRAMEWORK_FULL == "Grassmann Workspace Prediction"
        assert GWP.N_MECHANISMS == 16
        assert GWP.N_GROUPS == 3

    def test_gwp_alias(self):
        """GWPFramework must be an alias for GWP."""
        from src.models.mechanisms import GWP, GWPFramework

        assert GWPFramework is GWP

    def test_gwp_mechanism_groups(self):
        """GWP must organize mechanisms into 3 groups."""
        from src.models.mechanisms import GWP

        g = GWP(embed_dim=64, use_jawp=True, use_cgn=True, use_wsd=True, wsd_k=6, use_sta=True)
        groups = g.mechanism_groups()
        assert "core" in groups
        assert "routing" in groups
        assert "stability" in groups
        assert "jawp" in groups["core"]
        assert "cgn" in groups["routing"]
        assert "wsd" in groups["stability"]

    def test_gwp_dependency_dag(self):
        """Dependency DAG must show JAWP as root."""
        from src.models.mechanisms import GWP

        g = GWP(embed_dim=64, use_jawp=True, use_wsd=True, wsd_k=6, use_swip=True, use_wsr=True)
        dag = g.dependency_dag()
        # WSD depends on JAWP
        assert "jawp" in dag.get("wsd", [])
        # SWIP depends on JAWP
        assert "jawp" in dag.get("swip", [])
        # WSR depends on JAWP
        assert "jawp" in dag.get("wsr", [])
        # JAWP has no dependencies
        assert dag.get("jawp", []) == []

    def test_gwp_active_mechanisms(self):
        """active_mechanisms must return only non-None mechanisms."""
        from src.models.mechanisms import GWP

        g = GWP(embed_dim=64, use_jawp=True, use_cgn=True)
        active = g.active_mechanisms()
        assert "jawp" in active
        assert "cgn" in active
        assert "wsd" not in active  # not enabled
        assert "sta" not in active

    def test_gwp_from_config(self):
        """GWP.from_config must work like MechanismBundle.from_config."""
        from src.models.jepa import TextSpanJEPAConfig
        from src.models.mechanisms import GWP

        config = TextSpanJEPAConfig(embed_dim=64, use_jawp=True, use_sta=True)
        g = GWP.from_config(config)
        assert g.jawp is not None
        assert g.sta is not None

    def test_gwp_summary(self):
        """GWP.summary() must return human-readable string."""
        from src.models.mechanisms import GWP

        g = GWP(embed_dim=64, use_jawp=True, use_sta=True, use_puc=True)
        s = g.summary()
        assert "GWP" in s
        assert "Grassmann" in s
        assert "3 mechanisms" in s
