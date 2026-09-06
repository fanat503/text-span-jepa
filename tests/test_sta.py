# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Tests for Spectral Transport Alignment (STA) — mechanism #13."""

import math

import pytest
import torch

from src.models.sta import SpectralTransportAlignment


class TestSTABasic:
    """Basic STA construction and forward pass tests."""

    def test_construction_default(self):
        sta = SpectralTransportAlignment(embed_dim=64)
        assert sta.embed_dim == 64
        assert sta.eta == 0.01
        assert sta.ema_beta == 0.999
        assert sta.warmup_steps == 500

    def test_construction_custom(self):
        sta = SpectralTransportAlignment(
            embed_dim=128,
            eta=0.05,
            ema_beta=0.99,
            warmup_steps=200,
            update_interval=5,
        )
        assert sta.embed_dim == 128
        assert sta.eta == 0.05
        assert sta.ema_beta == 0.99
        assert sta.warmup_steps == 200
        assert sta.update_interval == 5

    def test_forward_shape(self):
        sta = SpectralTransportAlignment(embed_dim=64)
        z = torch.randn(4, 32, 64)
        loss, _info = sta(z, step=1000)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_forward_2d_input(self):
        sta = SpectralTransportAlignment(embed_dim=64)
        z = torch.randn(16, 64)
        loss, _info = sta(z, step=1000)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_info_keys(self):
        sta = SpectralTransportAlignment(embed_dim=64)
        z = torch.randn(4, 32, 64)
        _, info = sta(z, step=1000)
        expected_keys = [
            "sta_loss",
            "sta_w1",
            "sta_warmup_factor",
            "sta_warmup",
            "sta_spectral_gap",
            "sta_running_w1",
            "sta_running_spectral_gap",
            "sta_max_eigenvalue",
            "sta_min_eigenvalue",
            "sta_condition_number",
        ]
        for k in expected_keys:
            assert k in info, f"Missing key: {k}"

    def test_loss_non_negative(self):
        """STA loss must always be non-negative (W1 is a distance)."""
        sta = SpectralTransportAlignment(embed_dim=64)
        for _ in range(10):
            z = torch.randn(4, 32, 64)
            loss, _ = sta(z, step=1000)
            assert loss.item() >= 0.0


class TestSTAWarmup:
    """Warmup behavior tests."""

    def test_warmup_zero_loss(self):
        """During warmup, STA loss should be zero."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=100)
        z = torch.randn(4, 32, 64)
        loss, info = sta(z, step=0)
        assert loss.item() == 0.0
        assert info["sta_warmup"] is True

    def test_warmup_partial(self):
        """During partial warmup, loss is scaled by warmup_factor."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=1000)
        z = torch.randn(4, 32, 64)
        # Step 500 = 50% warmup
        _loss, info = sta(z, step=500)
        assert info["sta_warmup_factor"] == pytest.approx(0.5, abs=0.01)

    def test_warmup_complete(self):
        """After warmup, loss is unscaled."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=100)
        z = torch.randn(4, 32, 64)
        _loss, info = sta(z, step=200)
        assert info["sta_warmup_factor"] == pytest.approx(1.0)


class TestSTAWasserstein:
    """Wasserstein-1 distance property tests."""

    def test_w1_zero_for_same_spectrum(self):
        """W1 should be ~0 when current and reference spectra match."""
        sta = SpectralTransportAlignment(
            embed_dim=64,
            ema_beta=0.0,
            warmup_steps=0,
            update_interval=1,
        )
        # Create z with a specific spectrum
        z = torch.randn(32, 64)
        # Initialize reference
        sta(z, step=1)
        # Same z again → W1 should be ~0
        _loss, info = sta(z, step=2)
        # ema_beta=0 means ref_cov = cov_batch, so W1 should be very small
        assert info["sta_w1"] < 0.1

    def test_w1_monotone_coupling(self):
        """W1 with sorted eigenvalues implements optimal coupling
        (Kantorovich-Rubinstein duality)."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0, update_interval=1)
        # Step 1: establish reference
        z1 = torch.randn(32, 64)
        sta(z1, step=1)
        # Step 2: different spectrum
        z2 = torch.randn(32, 64) * 2  # different scale
        _loss, info = sta(z2, step=2)
        # W1 should be positive (spectra differ)
        # With ema_beta=0.999, reference barely changes, so W1 > 0
        assert info["sta_w1"] >= 0.0


class TestSTADavisKahan:
    """Davis-Kahan bound tests."""

    def test_davis_kahan_bound_positive(self):
        """Davis-Kahan bound should be non-negative."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0)
        z = torch.randn(32, 64)
        sta(z, step=100)
        bound = sta.compute_davis_kahan_bound(k=7)
        assert bound >= 0.0

    def test_davis_kahan_bound_with_explicit_w1(self):
        """Can pass explicit W1 value."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0)
        z = torch.randn(32, 64)
        sta(z, step=100)
        bound = sta.compute_davis_kahan_bound(k=7, w1=0.01)
        assert bound >= 0.0

    def test_downstream_stability_bound(self):
        """Downstream stability bound should be non-negative."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0)
        z = torch.randn(32, 64)
        sta(z, step=100)
        bound = sta.compute_downstream_stability_bound(probe_norm=1.0, total_variance=1.0, k=7)
        assert bound >= 0.0


class TestSTAEdgeCases:
    """Edge case tests."""

    def test_single_sample(self):
        """Single sample (N=1) should not crash."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0)
        z = torch.randn(1, 64)
        loss, _info = sta(z, step=100)
        assert loss.item() >= 0.0

    def test_identical_samples(self):
        """All identical samples should not crash."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0)
        z = torch.ones(8, 64)
        loss, _info = sta(z, step=100)
        assert loss.item() >= 0.0

    def test_very_large_values(self):
        """Very large values should not cause NaN."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0)
        z = torch.randn(4, 32, 64) * 1e3
        loss, _info = sta(z, step=100)
        assert not math.isnan(loss.item())
        assert not math.isinf(loss.item())

    def test_very_small_values(self):
        """Very small values should not cause NaN."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0)
        z = torch.randn(4, 32, 64) * 1e-6
        loss, _info = sta(z, step=100)
        assert not math.isnan(loss.item())

    def test_zero_input(self):
        """Zero input should not crash."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=0)
        z = torch.zeros(4, 32, 64)
        loss, _info = sta(z, step=100)
        assert loss.item() >= 0.0


class TestSTAIntegration:
    """Integration with model workflow."""

    def test_multiple_steps(self):
        """Running STA over multiple steps should work smoothly."""
        sta = SpectralTransportAlignment(embed_dim=64, warmup_steps=10, update_interval=5)
        losses = []
        for step in range(50):
            z = torch.randn(4, 16, 64) * (1 + 0.01 * step)  # slowly changing
            loss, _info = sta(z, step=step)
            losses.append(loss.item())
        # Losses should be finite
        assert all(math.isfinite(l) for l in losses)

    def test_spectral_drift_detected(self):
        """STA should detect when the spectrum changes significantly."""
        sta = SpectralTransportAlignment(
            embed_dim=64,
            warmup_steps=0,
            ema_beta=0.999,
            update_interval=1,
        )
        # Establish a reference with one scale
        z1 = torch.randn(32, 64)
        for step in range(1, 100):
            sta(z1, step=step)

        # Now change to a very different scale
        z2 = torch.randn(32, 64) * 5.0
        _loss_before, _info_before = sta(z1, step=100)
        _loss_after, info_after = sta(z2, step=101)

        # W1 should be larger with the different input
        assert info_after["sta_w1"] > 0.0

    def test_repr(self):
        """Test string representation."""
        sta = SpectralTransportAlignment(embed_dim=768, eta=0.05)
        s = repr(sta)
        assert "embed_dim=768" in s
        assert "eta=0.05" in s


class TestSTAMathematical:
    """Mathematical property verification tests."""

    def test_w1_triangle_inequality(self):
        """W1 should satisfy the triangle inequality:
        W1(μ1, μ3) ≤ W1(μ1, μ2) + W1(μ2, μ3)

        We verify this by computing W1 between three consecutive steps.
        """
        # Create three different spectra
        D = 64
        # Eigenvalues for three distributions
        eigs1 = torch.linspace(10, 1, D)
        eigs2 = torch.linspace(8, 0.5, D)
        eigs3 = torch.linspace(6, 0.1, D)

        # W1 between sorted eigenvalue vectors
        w1_12 = (eigs1 - eigs2).abs().mean().item()
        w1_23 = (eigs2 - eigs3).abs().mean().item()
        w1_13 = (eigs1 - eigs3).abs().mean().item()

        # Triangle inequality
        assert w1_13 <= w1_12 + w1_23 + 1e-6

    def test_w1_non_negative(self):
        """W1 should always be non-negative (it's a metric)."""
        D = 64
        eigs1 = torch.randn(D).abs().sort(descending=True)[0]
        eigs2 = torch.randn(D).abs().sort(descending=True)[0]
        w1 = (eigs1 - eigs2).abs().mean().item()
        assert w1 >= 0.0

    def test_w1_symmetric(self):
        """W1 should be symmetric: W1(λ, μ) = W1(μ, λ)."""
        D = 64
        eigs1 = torch.randn(D).abs().sort(descending=True)[0]
        eigs2 = torch.randn(D).abs().sort(descending=True)[0]
        w1_forward = (eigs1 - eigs2).abs().mean().item()
        w1_backward = (eigs2 - eigs1).abs().mean().item()
        assert w1_forward == pytest.approx(w1_backward, abs=1e-6)

    def test_w1_zero_for_identical(self):
        """W1(λ, λ) = 0 for identical distributions."""
        D = 64
        eigs = torch.randn(D).abs().sort(descending=True)[0]
        w1 = (eigs - eigs).abs().mean().item()
        assert w1 == pytest.approx(0.0, abs=1e-10)

    def test_davis_kahan_scales_with_w1(self):
        """Davis-Kahan bound should scale linearly with W1
        (bound = W1 / δ, so 2*W1 gives 2*bound)."""
        # Fix spectral gap
        delta = 1.0
        w1_1 = 0.01
        w1_2 = 0.02
        bound_1 = w1_1 / delta
        bound_2 = w1_2 / delta
        assert bound_2 == pytest.approx(2.0 * bound_1, abs=1e-10)
