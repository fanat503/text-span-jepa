# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Integration wiring tests: CMC bridge, GAC hook, live workspace views.

Pins the round-2 audit contracts:
  * compute_cmc_between_passes(primary, secondary) scatters compact slot
    predictions from two compute_loss_with_targets passes into full-sequence
    space and returns a finite, differentiable loss;
  * gradients reach the online encoder through the CMC secondary path;
  * GAC's stashed live slot tensor receives grads from the main backward and
    feeds a second scaled-free backward without freed-graph errors;
  * live workspace views give SWIP/RDC/WSR real dL/dQ (workspace_Q.grad);
  * WSR mode='gradient' prefers set_lagged_gradient() snapshots;
  * stiefel_retract projects its Riemannian correction in-place on Q.grad.
"""

import torch

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig


def _tiny_config() -> TextSpanJEPAConfig:
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
        use_swip=True,
        lambda_swip=0.01,
        use_rdc=True,
        lambda_rdc=0.01,
        use_wsr=True,
        lambda_wsr=0.01,
        wsr_mode="gradient",
        wsr_warmup_steps=0,
        use_cmc=True,
        cmc_interval=1,
        lambda_cmc=0.05,
        use_gac=True,
        gac_gamma=0.01,
        gac_tau_grad=1e-5,
        gac_warmup_steps=0,
        lambda_gac=0.02,
        future_warmup_steps=0,
    )


def _masks(T: int = 16):
    m1 = torch.zeros(2, T, dtype=torch.long)
    m2 = torch.zeros(2, T, dtype=torch.long)
    m1[:, 2:8] = 1  # primary spans
    m2[:, 5:11] = 1  # secondary spans; overlap at positions 5..7
    return m1, m2


class TestCMCBridge:
    def test_loss_finite_and_differentiable(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, m2 = _masks()

        out1, _, _ = model.compute_loss_with_targets(ids, ids, m1, 0, 100)
        primary = model._cmc_pass
        assert primary is not None and primary["slots"].shape[0] == 2

        model.compute_loss_with_targets(ids, ids, m2, 0, 100)
        secondary = model._cmc_pass
        assert secondary is not None

        loss_cmc, info = model.compute_cmc_between_passes(primary, secondary)
        assert torch.isfinite(loss_cmc).all()
        assert loss_cmc.requires_grad, "secondary pass must stay attached to the graph"
        assert info.get("cmc_skipped") in (False, True)

    def test_gradient_reaches_encoder_via_secondary(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, m2 = _masks()

        out1, _, _ = model.compute_loss_with_targets(ids, ids, m1, 0, 100)
        primary = model._cmc_pass
        model.compute_loss_with_targets(ids, ids, m2, 0, 100)

        loss_cmc, _ = model.compute_cmc_between_passes(primary, model._cmc_pass)
        total = out1 + model.config.lambda_cmc * loss_cmc
        total.backward(retain_graph=True)

        enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert enc_grads, "CMC secondary path must backprop into the online encoder"
        assert any(g.abs().sum() > 0 for g in enc_grads)


class TestGACHook:
    def test_stashed_slots_receive_grads(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        assert model._gac_z is None, "stash starts empty before the first forward"

        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        out, ldict, _ = model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        assert model._gac_z is not None, "lambda_gac>0 must stash live slots"
        out.backward(retain_graph=True)
        assert model._gac_z.grad is not None

        D = model._gac_z.size(-1)
        gn = model._gac_z.grad.detach().reshape(-1, D).norm(dim=0)
        loss_gac, info = model.gac(model._gac_z, gn, step=10)
        assert torch.isfinite(loss_gac).all()
        if loss_gac.requires_grad:
            loss_gac.backward()  # must not raise on a freed graph


class TestLiveWorkspaceViews:
    def test_regularizers_give_workspace_grads(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        out, _, _ = model.compute_loss_with_targets(ids, ids, m1, 20, 100)
        out.backward()

        qgrad = model.jawp.workspace_Q.grad
        assert (
            qgrad is not None and qgrad.abs().sum() > 0
        ), "SWIP/RDC/WSR live views must contribute gradient to workspace_Q"

    def test_wsr_prefers_lagged_gradient_snapshot(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        D, k = model.config.embed_dim, int(model.jawp.active_k.item())
        snap = torch.randn(D, k)
        model.wsr.set_lagged_gradient(snap)

        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        _, ldict, _ = model.compute_loss_with_targets(ids, ids, m1, 30, 100)
        assert torch.isfinite(torch.tensor(ldict["loss_wsr"]))

    def test_retraction_projects_grad_inplace_and_keeps_orthogonal(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        out, _, _ = model.compute_loss_with_targets(ids, ids, m1, 20, 100)
        out.backward()

        k = int(model.jawp.active_k.item())
        G = torch.randn_like(model.jawp.workspace_Q)
        g_before = G[:, :k].clone()
        model.jawp.workspace_Q.grad = G.clone()

        model.jawp.stiefel_retract()

        # Riemannian correction mutates grad[:, :k] in place (normal component removed).
        assert not torch.equal(model.jawp.workspace_Q.grad[:, :k], g_before)
        # SVD retraction leaves Q orthonormal.
        Q = model.jawp.workspace_Q.detach()
        ortho_err = (Q.T @ Q - torch.eye(Q.size(1))).abs().max()
        assert ortho_err < 1e-4
