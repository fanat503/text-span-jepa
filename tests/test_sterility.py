# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Sterility gates added in audit rounds 11-18:

1. Determinism — identical seed ⇒ identical loss trajectory on CPU.
2. Mechanism toggles — every weighted mechanism must change the total loss
   (guards the "decorative mechanism" failure class found in rounds 2-3).
3. Visualization smoke — public plot_* surface renders tiny inputs.
4. PUC differentiable entropy — gradient truth for the R18 branch.
"""

import numpy as np
import pytest
import torch

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.utils.seed import seed_everything


def _base_config() -> TextSpanJEPAConfig:
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
        future_warmup_steps=0,
    )


def _batch(seed=3):
    torch.manual_seed(seed)
    ids = torch.randint(0, 64, (2, 16))
    mask = torch.zeros(2, 16, dtype=torch.long)
    mask[:, 2:8] = 1
    return ids, mask


class TestDeterminism:
    def test_same_seed_same_loss_trajectory(self):
        traj = []
        for _ in range(2):
            seed_everything(42)
            model = TextSpanJEPA(_base_config())
            model.train()
            ids, mask = _batch()
            step_losses = []
            for step in range(3):
                out, _, _ = model.compute_loss_with_targets(
                    ids, ids, mask, current_step=step, total_steps=100
                )
                step_losses.append(out.item())
            traj.append(step_losses)
        assert traj[0] == traj[1], "same seed must reproduce the trajectory"


# Pure mechanisms: positive weight must change the total loss.
PURE_MECHANISMS = [
    ("jawp", {"use_jawp": True, "jawk_k_start": 2, "jawk_k_end": 4, "jawk_curriculum_steps": 0}),
    ("swip", {"use_swip": True, "lambda_swip": 0.05}),
    ("puc", {"use_puc": True, "lambda_puc": 0.05}),
    ("rdc", {"use_rdc": True, "lambda_rdc": 0.05}),
    ("spc", {"use_spc": True, "lambda_spc": 0.05}),
    ("sigreg", {"lambda_sigreg": 0.05}),
]

# Stateful mechanisms: asserted via their loss COMPONENT because the signal
# depends on cross-call state (WSD/WSR need JAWP; STA measures DRIFT between
# differing inputs; WSR consumes the lagged gradient snapshot).
STATEFUL_KWARGS = {
    "wsd": {
        "use_wsd": True,
        "lambda_wsd": 0.05,
        "use_jawp": True,
        "jawk_k_start": 2,
        "jawk_k_end": 4,
        "jawk_curriculum_steps": 0,
        "jawk_init": "random",
    },
    "sta": {"use_sta": True, "lambda_sta": 0.05, "sta_warmup_steps": 0},
    "wsr": {
        "use_wsr": True,
        "lambda_wsr": 0.05,
        "use_jawp": True,
        "jawk_k_start": 2,
        "jawk_k_end": 4,
        "jawk_curriculum_steps": 0,
        "jawk_init": "random",
        "wsr_mode": "gradient",
        "wsr_warmup_steps": 0,
    },
}


class TestMechanismToggles:
    def _model(self, cfg_kwargs):
        seed_everything(7)
        merged = {**_base_config().__dict__, **cfg_kwargs}
        model = TextSpanJEPA(TextSpanJEPAConfig(**merged))
        model.train()
        return model

    def _total(self, cfg_kwargs):
        model = self._model(cfg_kwargs)
        ids, mask = _batch()
        total, _, _ = model.compute_loss_with_targets(ids, ids, mask, 50, 100)
        return total

    @pytest.mark.parametrize("name,kwargs", PURE_MECHANISMS, ids=[m[0] for m in PURE_MECHANISMS])
    def test_pure_mechanism_changes_total_loss(self, name, kwargs):
        assert (
            abs(self._total(kwargs) - self._total({})) > 1e-9
        ), f"mechanism {name} enabled with positive weight must move the total loss"

    def test_wsd_component_nonzero(self):
        model = self._model(STATEFUL_KWARGS["wsd"])
        ids, mask = _batch()
        _, ld, _ = model.compute_loss_with_targets(ids, ids, mask, 60, 100)
        assert ld["loss_wsd"] > 0, "drift between random Q and identity target must be > 0"

    def test_sta_drift_after_reference_init(self):
        model = self._model(STATEFUL_KWARGS["sta"])
        ids, base_mask = _batch()
        # The encoder input must actually DIFFER between calls: identical
        # inputs yield identical spectra and zero drift by definition.
        mi1 = ids.clone()
        mi1[base_mask.bool()] = 3
        alt = torch.zeros_like(base_mask)
        alt[:, ::3] = 1
        mi2 = ids.clone()
        mi2[alt.bool()] = 3
        model.compute_loss_with_targets(mi1, ids, base_mask, 50, 100)  # init ref
        _, ld, _ = model.compute_loss_with_targets(mi2, ids, alt, 51, 100)
        assert ld["loss_sta"] > 0, "spectrum changed ⇒ W1 against reference must be > 0"

    def test_wsr_consumes_lagged_gradient(self):
        model = self._model(STATEFUL_KWARGS["wsr"])
        k = int(model.jawp.active_k.item())
        model.wsr.set_lagged_gradient(torch.randn(model.config.embed_dim, k))
        ids, mask = _batch()
        _, ld, _ = model.compute_loss_with_targets(ids, ids, mask, 60, 100)
        assert ld["loss_wsr"] > 0


class TestPUCDifferentiableEntropy:
    @staticmethod
    def _run(use_diff, target_entropy):
        from src.models.puc import PredictionUncertaintyCalibration

        puc = PredictionUncertaintyCalibration(
            embed_dim=32,
            n_components=4,
            warmup_steps=0,
            target_entropy=target_entropy,
            use_differentiable_entropy=use_diff,
        )
        z = torch.randn(4, 8, 32, requires_grad=True)
        loss, info = puc(z, step=10)
        return z, loss, info

    def test_differentiable_path_carries_gradient(self):
        # Large target entropy forces a non-zero deficit ⇒ grad-carrying loss.
        z, loss, _ = self._run(use_diff=True, target_entropy=200.0)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert z.grad is not None and z.grad.abs().sum() > 0

    def test_legacy_buffer_path_is_grad_free_and_safe(self):
        # Legacy path never carries gradient; a zero-deficit step must yield
        # a clean scalar(0) WITHOUT grad_fn (backward would raise).
        z, loss, _ = self._run(use_diff=False, target_entropy=200.0)
        assert torch.isfinite(loss).all()
        assert not loss.requires_grad


class TestVizSmoke:
    """Every listed public plot renders on tiny synthetic input."""

    CASES = [
        ("plot_eigenvalue_spectrum", lambda m: (np.linspace(4.0, 0.1, 8),)),
        ("plot_cka_heatmap", lambda m: (np.eye(3) * 0.5,)),
        ("plot_svcca_curve", lambda m: (np.linspace(0.99, 0.4, 6),)),
        (
            "plot_workspace_evolution",
            lambda m: ([0, 1, 2], [0.2, 0.3, 0.4], [0.1, 0.15, 0.2], [4, 4, 4]),
        ),
        (
            "plot_collapse_timeline",
            lambda m: ([0, 1, 2], [12.0, 10.0, 8.0], [0.02, 0.01, 0.0], [0.7, 0.6, 0.5]),
        ),
        ("plot_scaling_curve", lambda m: ([1e6, 1e7], {"loss": [2.0, 1.5]})),
        ("plot_jawp_vs_pca", lambda m: ([0, 1, 2], [1.0, 0.8, 0.6], [1.2, 1.0, 0.9])),
        ("plot_gating_pattern", lambda m: (np.random.rand(4), np.random.rand(4))),
        ("plot_rank_utilization", lambda m: ([0, 1, 2], [8.0, 6.0, 5.0], 16)),
        ("plot_spectral_waterfall", lambda m: ([np.linspace(3, 0.1, 5)] * 3, [0, 1, 2])),
        ("plot_information_flow", lambda m: ({"lvl0": 0.7, "lvl1": 0.4},)),
        (
            "plot_swip_spectral_shaping",
            lambda m: (np.linspace(3, 0.5, 6), np.linspace(2, 0.8, 6), 2),
        ),
        (
            "plot_spc_band_analysis",
            lambda m: (np.random.rand(8), np.random.rand(8), np.random.rand(8)),
        ),
        ("plot_wsd_drift", lambda m: ([0, 1, 2], [0.5, 0.3, 0.1])),
        ("plot_cmc_consistency", lambda m: ([0, 1, 2], [0.4, 0.2, 0.1], [0.3, 0.3, 0.3])),
        ("plot_sta_spectral_alignment", lambda m: ([0, 1, 2], [0.6, 0.4, 0.2])),
        ("plot_gac_starved_fraction", lambda m: ([0, 1, 2], [0.5, 0.3, 0.1])),
        (
            "plot_puc_overconfidence_timeline",
            lambda m: ([0, 1, 2], [0.8, 0.5, 0.2]),
        ),
    ]

    @pytest.mark.parametrize("fn_name,args_fn", CASES, ids=[c[0] for c in CASES])
    def test_plot_renders_and_returns_figure(self, fn_name, args_fn):
        import matplotlib.pyplot as plt

        from src.utils import visualization as viz

        assert viz._ensure_mpl(), "matplotlib required for viz smoke tests"
        fn = getattr(viz, fn_name)
        out = fn(*args_fn(np))
        assert out is not None
        fig = out[0] if isinstance(out, tuple) else out
        plt.close(fig)
