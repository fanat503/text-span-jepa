# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""WSR ``mode='sam'`` must compute a real sharpness and train the workspace.

Regression tests for two defects: the model never handed ``loss_fn`` to the
module (so L(Q + delta) - L(Q) was 0 - 0), and the SAM loss was built from
``.item()`` values, a constant that carried no gradient into Q.
"""

import torch
import torch.nn.functional as F

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.models.wsr import WorkspaceSharpnessRegularization


def _workspace_mse(Q, z_pred, z_target):
    return F.mse_loss(z_pred @ Q, z_target @ Q)


def _orthonormal_Q(D=16, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, k, generator=g))
    return Q.clone().requires_grad_(True)


class TestSamModuleGradient:
    def test_sharpness_is_positive_and_comes_from_autograd(self):
        g = torch.Generator().manual_seed(1)
        z_pred, z_target = torch.randn(32, 16, generator=g), torch.randn(32, 16, generator=g)
        wsr = WorkspaceSharpnessRegularization(
            embed_dim=16, mode="sam", rho=0.1, eta=1.0, warmup_steps=0
        )
        _, info = wsr(_orthonormal_Q(), _workspace_mse, z_pred, z_target, step=1)
        assert info["wsr_grad_source"] == "autograd"
        assert info["wsr_gradient_substituted"] is False
        assert info["wsr_loss_perturbed"] > info["wsr_loss_current"]
        assert info["wsr_sharpness"] > 0.0

    def test_loss_back_propagates_into_Q(self):
        g = torch.Generator().manual_seed(2)
        z_pred, z_target = torch.randn(32, 16, generator=g), torch.randn(32, 16, generator=g)
        Q = _orthonormal_Q()
        wsr = WorkspaceSharpnessRegularization(
            embed_dim=16, mode="sam", rho=0.1, eta=1.0, warmup_steps=0
        )
        loss, _ = wsr(Q, _workspace_mse, z_pred, z_target, step=1)
        assert loss.requires_grad, "SAM loss must stay attached to the graph"
        loss.backward()
        assert Q.grad is not None and Q.grad.abs().sum() > 0

    def test_loss_matches_finite_difference_definition(self):
        g = torch.Generator().manual_seed(3)
        z_pred, z_target = torch.randn(32, 16, generator=g), torch.randn(32, 16, generator=g)
        Q = _orthonormal_Q()
        wsr = WorkspaceSharpnessRegularization(
            embed_dim=16, mode="sam", rho=0.1, eta=2.0, warmup_steps=0
        )
        loss, info = wsr(Q, _workspace_mse, z_pred, z_target, step=1)
        expected = 2.0 * max(info["wsr_loss_perturbed"] - info["wsr_loss_current"], 0.0)
        assert abs(loss.item() - expected) < 1e-6


class TestSamInsideModel:
    @staticmethod
    def _config(mode):
        return TextSpanJEPAConfig(
            vocab_size=64,
            max_seq_len=16,
            embed_dim=32,
            encoder_depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            predictor_embed_dim=16,
            predictor_depth=1,
            future_offsets=[1],
            num_refine_steps=1,
            use_jawp=True,
            jawk_k_start=2,
            jawk_k_end=4,
            jawk_curriculum_steps=0,
            use_wsr=True,
            lambda_wsr=1.0,
            wsr_rho=0.1,
            wsr_eta=1.0,
            wsr_mode=mode,
            wsr_warmup_steps=0,
            future_warmup_steps=0,
        )

    def test_model_passes_loss_fn_so_sam_is_not_a_constant_zero(self):
        torch.manual_seed(0)
        model = TextSpanJEPA(self._config("sam"))
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.long)
        mask[:, 2:8] = 1
        _, info, _ = model.compute_loss_with_targets(ids, ids, mask, 1, 100)
        assert info["wsr_wsr_grad_source"] == "autograd"
        assert info["wsr_wsr_loss_perturbed"] != 0.0
        assert info["wsr_wsr_loss_perturbed"] >= info["wsr_wsr_loss_current"]
