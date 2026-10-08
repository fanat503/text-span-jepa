# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
# Tests for SIGReg + J-Space metrics

import math

import torch


class TestSIGReg:
    def test_import(self):
        from src.models.sigreg import SIGReg

        assert SIGReg is not None

    def test_creation(self):
        from src.models.sigreg import SIGReg

        sigreg = SIGReg(embed_dim=64, n_sketches=8, n_integration_points=5)
        assert sigreg.embed_dim == 64

    def test_loss_nonnegative(self):
        from src.models.sigreg import SIGReg

        sigreg = SIGReg(embed_dim=64, n_sketches=8, n_integration_points=5)
        x = torch.randn(16, 32, 64)
        loss = sigreg(x)
        assert loss.item() >= 0
        assert math.isfinite(loss.item())

    def test_loss_finite(self):
        from src.models.sigreg import SIGReg

        sigreg = SIGReg(embed_dim=32, n_sketches=4, n_integration_points=3)
        x = torch.randn(4, 8, 32)
        loss = sigreg(x)
        assert math.isfinite(loss.item())

    def test_sketch_directions_orthogonal(self):
        from src.models.sigreg import SIGReg

        sigreg = SIGReg(embed_dim=64, n_sketches=8, n_integration_points=5)
        dirs = sigreg.sketch_directions
        # Each direction should be unit norm
        norms = dirs.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_weak_sigreg(self):
        from src.models.sigreg import WeakSIGReg

        ws = WeakSIGReg(embed_dim=32, n_sketches=4)
        x = torch.randn(8, 16, 32)
        loss = ws(x)
        assert loss.item() >= 0
        assert math.isfinite(loss.item())

    def test_visreg(self):
        from src.models.sigreg import VISReg

        vr = VISReg(embed_dim=32, n_sketches=4, n_integration_points=3)
        x = torch.randn(8, 16, 32)
        loss, info = vr(x)
        assert loss.item() >= 0
        assert math.isfinite(loss.item())
        assert "loss_variance" in info
        assert "loss_sigreg" in info

    def test_empty_batch(self):
        from src.models.sigreg import SIGReg

        sigreg = SIGReg(embed_dim=32, n_sketches=4, n_integration_points=3)
        x = torch.randn(1, 8, 32)
        loss = sigreg(x)
        assert math.isfinite(loss.item())


class TestJSpace:
    def test_import(self):
        from src.models.jspace import JSpaceMetrics

        assert JSpaceMetrics is not None

    def test_creation(self):
        from src.models.jspace import JSpaceMetrics

        js = JSpaceMetrics()
        assert js.variance_threshold == 0.10

    def test_compute_returns_metrics(self):
        from src.models.jspace import JSpaceMetrics

        js = JSpaceMetrics()
        online_h = torch.randn(4, 32, 64)
        target_h = torch.randn(4, 32, 64)
        metrics = js.compute(online_h, target_h)
        assert "jspace_quality" in metrics
        assert "jspace_dim_fraction" in metrics
        assert "jspace_spectral_gap" in metrics

    def test_metrics_bounded(self):
        from src.models.jspace import JSpaceMetrics

        js = JSpaceMetrics()
        online_h = torch.randn(4, 32, 64)
        target_h = torch.randn(4, 32, 64)
        metrics = js.compute(online_h, target_h)
        assert 0 <= metrics["jspace_quality"] <= 1.01
        assert 0 <= metrics["jspace_dim_fraction"] <= 1.01
        assert 0 <= metrics["jspace_spectral_gap"] <= 1.01

    def test_small_batch_returns_zero_metrics(self):
        from src.models.jspace import JSpaceMetrics

        js = JSpaceMetrics()
        online_h = torch.randn(1, 1, 4)
        target_h = torch.randn(1, 1, 4)
        metrics = js.compute(online_h, target_h)
        assert metrics["jspace_quality"] == 0.0

    def test_stability_across_steps(self):
        from src.models.jspace import JSpaceMetrics

        js = JSpaceMetrics()
        online_h = torch.randn(4, 32, 64)
        target_h = torch.randn(4, 32, 64)
        js.compute(online_h, target_h)
        m2 = js.compute(online_h, target_h)
        assert 0 <= m2["jspace_stability"] <= 1.01
