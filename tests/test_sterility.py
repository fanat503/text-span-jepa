# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Sterility gates added in audit round 11:

1. Determinism — identical seed ⇒ identical loss trajectory on CPU.
2. Mechanism toggles — every weighted mechanism must change the total loss
   (guards the "decorative mechanism" failure class found in rounds 2-3).
3. Visualization smoke — the public plot_* surface renders on tiny inputs
   without exceptions and returns closed-able figures.
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


MECHANISMS = [
    # (name, config overrides, number of forwards before sampling)
    # Stateful mechanisms (EMA/target init on first call) need >=2 calls.
    ("jawp", {"use_jawp": True, "jawk_k_start": 2, "jawk_k_end": 4, "jawk_curriculum_steps": 0}, 1),
    ("swip", {"use_swip": True, "lambda_swip": 0.05}, 1),
    (
        "wsd",
        {
            "use_wsd": True,
            "lambda_wsd": 0.05,
            "use_jawp": True,
            "jawk_k_start": 2,
            "jawk_k_end": 4,
            "jawk_curriculum_steps": 0,
            "jawk_init": "random",
        },
        2,
    ),
    ("sta", {"use_sta": True, "lambda_sta": 0.05, "sta_warmup_steps": 0}, 2),
    ("puc", {"use_puc": True, "lambda_puc": 0.05}, 1),
    ("rdc", {"use_rdc": True, "lambda_rdc": 0.05}, 1),
    (
        "wsr",
        {
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
        2,
    ),
    ("sigreg", {"lambda_sigreg": 0.05}, 1),
]


class TestMechanismToggles:
    def _loss_for(self, cfg_kwargs, calls=1):
        seed_everything(7)
        model = TextSpanJEPA(TextSpanJEPAConfig(**{**_base_config().__dict__, **cfg_kwargs}))
        model.train()
        ids, mask = _batch()
        # Drift source: alternate REAL masked inputs (the encoder must see
        # different tokens), not just mask_positions.
        variants = []
        alt = torch.zeros_like(mask)
        alt[:, ::2] = 1
        for mm in (mask, alt):
            mi = ids.clone()
            mi[mm.bool()] = 3
            variants.append((mi, mm))
        total = None
        for step, (mi, mm) in enumerate(variants * ((calls + 1) // 2)):
            total, _, _ = model.compute_loss_with_targets(mi, ids, mm, 50 + step, 100)
        return total

    @pytest.mark.parametrize("name,kwargs,calls", MECHANISMS, ids=[m[0] for m in MECHANISMS])
    def test_enabled_mechanism_changes_total_loss(self, name, kwargs, calls):
        off = self._loss_for({}, calls)
        on = self._loss_for(kwargs, calls)
        assert (
            abs(on.item() - off.item()) > 1e-9
        ), f"mechanism {name} enabled with positive weight must move the total loss"


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
        (
            "plot_gating_pattern",
            lambda m: (np.random.rand(4), np.random.rand(4)),
        ),
        ("plot_rank_utilization", lambda m: ([0, 1, 2], [8.0, 6.0, 5.0], 16)),
        (
            "plot_spectral_waterfall",
            lambda m: ([np.linspace(3, 0.1, 5)] * 3, [0, 1, 2]),
        ),
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
