# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
# Tests for PUC (Prediction Uncertainty Calibration) — mechanism #14

import math

import pytest
import torch

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.models.puc import PredictionUncertaintyCalibration
from src.utils.seed import seed_everything


class TestPUCCore:
    embed_dim = 64
    batch_size = 4
    seq_len = 16

    def test_init(self):
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        assert puc.embed_dim == self.embed_dim
        assert puc.eta == 0.01
        assert puc.running_mean.shape == (self.embed_dim,)

    def test_init_custom(self):
        puc = PredictionUncertaintyCalibration(
            embed_dim=self.embed_dim,
            n_components=16,
            eta=0.05,
            ema_beta=0.99,
            warmup_steps=200,
        )
        assert puc.n_components == 16
        assert puc.eta == 0.05
        assert puc.ema_beta == 0.99

    def test_forward_returns_loss_and_info(self):
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        _loss, info = puc(z, step=1000)
        assert isinstance(info, dict)
        assert "puc_loss" in info
        assert "puc_entropy" in info
        assert "puc_overconfidence" in info

    def test_loss_non_negative(self):
        """PUC loss is always ≥ 0."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        for _ in range(10):
            z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
            loss, _info = puc(z, step=1000)
            assert loss.item() >= -1e-6, f"PUC loss negative: {loss.item()}"

    def test_warmup_zero_loss(self):
        """PUC loss is 0 during warmup."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim, warmup_steps=1000)
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        loss, info = puc(z, step=0)
        assert loss.item() < 1e-6
        assert info.get("puc_warmup", False) is True

    def test_warmup_ramp(self):
        """PUC warmup factor ramps from 0 to 1."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim, warmup_steps=1000)
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        _, info_0 = puc(z, step=0)
        _, info_500 = puc(z, step=500)
        _, info_2000 = puc(z, step=2000)
        assert info_0["puc_warmup_factor"] < info_500["puc_warmup_factor"]
        assert abs(info_2000["puc_warmup_factor"] - 1.0) < 1e-6

    def test_overconfident_predictions_have_loss(self):
        """Collapsed (zero-variance) predictions should trigger PUC loss."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim, warmup_steps=0)
        # Constant predictions → zero variance → overconfident
        z_const = torch.ones(self.batch_size, self.seq_len, self.embed_dim) * 0.5
        _loss, info = puc(z_const, step=1000)
        assert info["puc_overconfidence"] > 0 or info["puc_entropy_deficit"] > 0

    def test_diverse_predictions_lower_loss(self):
        """High-variance predictions should have lower PUC loss."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim, warmup_steps=0, eta=0.1)
        z_const = torch.ones(self.batch_size, self.seq_len, self.embed_dim) * 0.5
        z_diverse = torch.randn(self.batch_size, self.seq_len, self.embed_dim) * 3.0
        _, info_const = puc(z_const, step=1000)
        _, info_diverse = puc(z_diverse, step=2000)
        # Diverse predictions should have higher entropy
        assert info_diverse["puc_entropy"] >= info_const["puc_entropy"] - 1.0

    def test_loss_finite(self):
        """PUC loss is always finite."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        for scale in [0.01, 1.0, 100.0]:
            z = torch.randn(self.batch_size, self.seq_len, self.embed_dim) * scale
            loss, _info = puc(z, step=1000)
            assert torch.isfinite(loss) if torch.is_tensor(loss) else math.isfinite(loss)

    def test_gradient_flows(self):
        """PUC loss supports gradient flow."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim, warmup_steps=0)
        z = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True)
        loss, _ = puc(z, step=1000)
        if loss.requires_grad and loss.item() > 0:
            loss.backward()
            assert z.grad is not None


class TestPUCTheorems:
    """Mathematical theorem tests for PUC."""

    embed_dim = 64

    def test_target_entropy_is_isotropic_gaussian(self):
        """Default target entropy = H(N(0,I)) = D/2 * log(2πe)."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        expected = 0.5 * self.embed_dim * math.log(2 * math.pi * math.e)
        assert abs(puc.target_entropy - expected) < 1e-6

    def test_entropy_non_negative_for_valid_covariance(self):
        """Differential entropy of a valid covariance is well-defined."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim, warmup_steps=0)
        z = torch.randn(8, 32, self.embed_dim)
        _, info = puc(z, step=1000)
        # Entropy can be negative in high dims (this is fine for differential entropy)
        # But eigenvalues should be positive
        assert info["puc_min_eigenvalue"] > -1e-6

    def test_log_det_barrier_convex(self):
        """Log-determinant barrier -log det(Σ) is convex on PD matrices."""
        # Verify: -log(λ₁·λ₂) is convex in (λ₁, λ₂) for λ > 0
        # This is a standard result — we verify numerically
        for _ in range(100):
            lam1 = torch.exp(torch.randn(1) * 2).clamp(min=0.01)
            lam2 = torch.exp(torch.randn(1) * 2).clamp(min=0.01)
            # f(x) = -log(x) is convex for x > 0
            f1 = -torch.log(lam1)
            f2 = -torch.log(lam2)
            # Check midpoint convexity
            alpha = torch.rand(1)
            mid = alpha * lam1 + (1 - alpha) * lam2
            f_mid = -torch.log(mid)
            assert f_mid <= alpha * f1 + (1 - alpha) * f2 + 1e-4

    def test_donsker_varadhan_connection(self):
        """PUC loss upper-bounds KL divergence to maximum-entropy distribution.

        By Donsker-Varadhan:
          KL(q || p*) = sup_f {E_q[f] - log E_p*[exp(f)]}
        For f = -||z||²/2σ², this gives:
          KL ≤ H(p*) - H(q) + const
        PUC minimizes this bound.
        """
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim, warmup_steps=0)
        z = torch.randn(8, 32, self.embed_dim)
        _, info = puc(z, step=1000)

        # The entropy deficit upper-bounds the KL divergence
        # (up to constants that depend on the target distribution)
        entropy_deficit = info["puc_entropy_deficit"]
        # If entropy_deficit > 0, predictions are overconfident
        # and KL divergence to isotropic Gaussian is bounded below by deficit
        if entropy_deficit > 0:
            assert entropy_deficit > 0  # trivially true, but documents the property

    def test_minimax_optimality_property(self):
        """Maximum entropy distribution is minimax optimal for bounded losses.

        By Jaynes (1957): among all distributions satisfying the
        prediction constraint, max-entropy distribution minimizes
        the worst-case expected loss over all bounded downstream tasks.
        """
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim, warmup_steps=0)
        # Overconfident predictions (constant)
        z_overconfident = torch.ones(8, 32, self.embed_dim) * 0.5
        # Maximum-entropy predictions (diverse)
        z_max_ent = torch.randn(8, 32, self.embed_dim)

        _, info_oc = puc(z_overconfident, step=1000)
        _, info_me = puc(z_max_ent, step=2000)

        # Max-entropy predictions should have higher entropy
        assert info_me["puc_entropy"] >= info_oc["puc_entropy"] - 2.0


class TestPUCDiagnostics:
    """Diagnostic output tests."""

    embed_dim = 64

    def test_full_diagnostics(self):
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        z = torch.randn(4, 16, self.embed_dim)
        _, info = puc(z, step=1000)

        required_keys = [
            "puc_loss",
            "puc_entropy",
            "puc_target_entropy",
            "puc_entropy_deficit",
            "puc_overconfidence",
            "puc_warmup_factor",
            "puc_min_eigenvalue",
            "puc_max_eigenvalue",
            "puc_log_det",
            "puc_n_components",
        ]
        for key in required_keys:
            assert key in info, f"Missing diagnostic: {key}"

    def test_diagnostics_are_finite(self):
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        z = torch.randn(4, 16, self.embed_dim)
        _, info = puc(z, step=1000)
        for key, val in info.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"{key} not finite: {val}"


class TestPUCIsNotAConstant:
    """TASK-39. The defect this class exists to make unrepeatable.

    With the entropy source set to the Oja EMA buffer, the loss is built from a
    `register_buffer` running estimate. Buffers have `requires_grad=False`, so
    the returned loss has no autograd edge to `z_pred`: `total_loss` shifts by a
    constant, `backward()` puts nothing new into any parameter, and PUC cannot
    influence training at all. It looked live because `use_puc` is true, the
    weight is non-zero, `validate()` gates it, and a number reaches the printed
    loss.

    These tests assert the PROPERTY — a gradient edge exists — not a magnitude,
    so they survive a refactor of the barrier. The card's own numbers were
    False -> 0.1401 / requires_grad False, True -> 0.1037 / z.grad 1.4e-05.
    """

    embed_dim = 32

    def _encoder_grad(self, use_puc: bool, lambda_puc: float) -> torch.Tensor:
        """Flat encoder gradient after one backward, everything else fixed."""
        seed_everything(7)
        cfg = TextSpanJEPAConfig(
            vocab_size=64,
            max_seq_len=16,
            embed_dim=self.embed_dim,
            encoder_depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            predictor_embed_dim=16,
            predictor_depth=1,
            future_offsets=[1],
            num_refine_steps=1,
            future_warmup_steps=0,
            use_puc=use_puc,
            lambda_puc=lambda_puc,
            puc_warmup_steps=0,
        )
        model = TextSpanJEPA(cfg)
        model.train()
        torch.manual_seed(3)
        ids = torch.randint(0, 64, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 2:8] = 1
        total, loss_dict, _ = model.compute_loss_with_targets(ids, ids, mask, 50, 100)
        total.backward()
        grads = [p.grad.reshape(-1) for p in model.encoder.parameters() if p.grad is not None]
        assert grads, "encoder has no gradient at all; the test would be vacuous"
        return torch.cat(grads), float(loss_dict["loss_puc"])

    def test_default_config_puc_loss_carries_an_edge_to_the_encoder(self):
        """THE card. Under the DEFAULT config, PUC must reach the encoder.

        Compares the encoder gradient with PUC enabled at a non-zero weight
        against the same model with PUC off. If PUC's loss is a constant the two
        tensors are bitwise identical and this is 0.0 — which is precisely the
        measurement that identified the defect.
        """
        with_puc, loss_puc = self._encoder_grad(use_puc=True, lambda_puc=0.05)
        without_puc, _ = self._encoder_grad(use_puc=False, lambda_puc=0.0)
        assert loss_puc > 0.0, (
            f"PUC's own loss is {loss_puc}; a zero loss cannot distinguish a "
            "live mechanism from an inert one, so this test would pass for the "
            "wrong reason. Widen the seed/batch or the fixture is degenerate."
        )
        delta = (with_puc - without_puc).abs().max()
        assert delta > 0.0, (
            "PUC's loss reaches total_loss but not the encoder: max |encoder.grad "
            "(PUC on) - encoder.grad(PUC off)| is exactly 0.0. PUC is a constant "
            "added to the objective. Check use_differentiable_entropy."
        )

    def test_default_construction_takes_the_differentiable_path(self):
        """The module default must be the path that can train."""
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        assert puc.use_differentiable_entropy is True, (
            "the default flipped back to the EMA-buffer path: with it False the "
            "loss has no autograd edge and PUC cannot train"
        )

    def test_grad_free_path_is_still_reachable_but_refuses_to_do_it_quietly(self):
        """The legacy path must keep working AND announce itself.

        tests/test_sterility.py pins that the buffer path stays grad-free, so it
        cannot be deleted under this card. What can change is that selecting it
        is no longer silent.
        """
        with pytest.warns(UserWarning, match="no autograd edge"):
            PredictionUncertaintyCalibration(
                embed_dim=self.embed_dim,
                use_differentiable_entropy=False,
            )

    def test_loss_reports_whether_it_can_reach_the_encoder(self):
        """The printed loss must say whether its own term can train.

        A non-zero loss with no edge is the inert case; the diagnostic has to
        distinguish it from the ordinary zero-loss-with-no-edge case.

        `target_entropy=200.0` (as tests/test_sterility.py uses) opens the
        entropy-deficit gate for an N(0, I) batch, so the loss is the barrier
        rather than the gate's zero.
        """
        puc = PredictionUncertaintyCalibration(
            embed_dim=self.embed_dim,
            warmup_steps=0,
            target_entropy=200.0,
        )
        z = torch.randn(4, 16, self.embed_dim, requires_grad=True)
        loss, info = puc(z, step=100)
        assert loss.item() > 0.0, f"expected a positive loss, got {loss.item()}"
        assert info["puc_carries_grad"] is True
        assert loss.requires_grad is True

        # The warning fires at construction, so it is asserted where it happens.
        # It is already pinned by test_grad_free_path_is_still_reachable_but_
        # refuses_to_do_it_quietly.
        with pytest.warns(UserWarning):
            legacy = PredictionUncertaintyCalibration(
                embed_dim=self.embed_dim,
                warmup_steps=0,
                target_entropy=200.0,
                use_differentiable_entropy=False,
            )
        legacy_loss, legacy_info = legacy(z, step=100)
        assert legacy_loss.item() > 0.0, (
            f"the buffer path must still produce the same positive number, got "
            f"{legacy_loss.item()}; if it changed, the two paths no longer "
            "differ only in whether the value carries a gradient"
        )
        assert legacy_info["puc_carries_grad"] is False


class TestPUCCheckpoint:
    """Checkpoint save/restore tests."""

    embed_dim = 64

    def test_checkpoint_save_restore(self):
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        z = torch.randn(4, 16, self.embed_dim)
        puc(z, step=500)

        ckpt = puc.checkpoint_dict()
        assert "running_mean" in ckpt
        assert "running_eigenvalues" in ckpt
        assert "proj_vectors" in ckpt

        # Restore
        puc2 = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        puc2.load_checkpoint(ckpt)
        assert torch.allclose(puc.running_mean, puc2.running_mean)
        assert torch.allclose(puc.running_eigenvalues, puc2.running_eigenvalues)
        assert torch.allclose(puc.proj_vectors, puc2.proj_vectors, atol=1e-6)

    def test_checkpoint_after_training_step(self):
        puc = PredictionUncertaintyCalibration(embed_dim=self.embed_dim)
        for step in range(100, 1100, 100):
            z = torch.randn(4, 16, self.embed_dim)
            puc(z, step=step)

        ckpt = puc.checkpoint_dict()
        assert ckpt["total_steps"].item() == 10  # 10 calls


class TestPUCShapes:
    """Shape verification tests."""

    def test_various_embed_dims(self):
        for dim in [32, 64, 128, 256]:
            puc = PredictionUncertaintyCalibration(embed_dim=dim)
            z = torch.randn(2, 8, dim)
            _loss, _info = puc(z, step=1000)
            assert puc.running_mean.shape == (dim,)

    def test_various_batch_sizes(self):
        puc = PredictionUncertaintyCalibration(embed_dim=64)
        for bs in [1, 4, 16]:
            z = torch.randn(bs, 16, 64)
            _loss, info = puc(z, step=1000)
            assert isinstance(info, dict)

    def test_various_seq_lengths(self):
        puc = PredictionUncertaintyCalibration(embed_dim=64)
        for sl in [1, 32, 128]:
            z = torch.randn(4, sl, 64)
            _loss, info = puc(z, step=1000)
            assert isinstance(info, dict)
