# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Ablation honesty: a run may not be labelled as an ablation that did nothing.

Four ways ``src/interp/ablation.py`` used to lie, all with the same root
cause — nothing ever checked that the ablation had actually happened:

  D1  ``AblatedModel.forward`` removed a term only ``if "loss_X" in info``.
      A renamed loss key, a model config with no weight for the term, or a
      flag with no mechanism behind it at all left the loss untouched while
      the run was still reported as that ablation.
  D2  ``predictor.num_refine_steps`` was mutated before the forward and
      restored after it with no ``try/finally``: one exception pinned the
      model at 0 refinement steps for the rest of the training run.
  D3  ``run_scaling_ablations`` / ``run_all`` swallowed every exception into
      ``{"error": ...}`` and kept looping, so a crashed cell was
      indistinguishable from a completed one unless the caller went looking.
  D4  the "no EMA target" arm only re-initialised the target encoder;
      ``skip_ema_update()`` had no caller, so ordinary EMA updates resumed
      on the very next training step.

The contract enforced here:

  * a missing ablation target is a hard, loud failure — never a silent no-op
  * a no-op must be asked for explicitly (``allow_missing_targets=True``)
  * mutated model state survives an exception
  * a crashed cell is visibly failed in the structure *and* the aggregate
  * the no-EMA arm stays no-EMA while training
"""

from __future__ import annotations

import copy

import pytest
import torch
from src.interp import ablation as abl_mod
from src.utils.seed import seed_everything
from torch import nn

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

# The five terms a well-behaved model always reports.  A stub that hands back
# exactly this dict is a "model that supports every loss ablation".
ALL_TERMS = {
    "loss_span": 1.0,
    "loss_future": 2.0,
    "loss_decoder": 0.5,
    "loss_variance": 0.25,
    "loss_covariance": 0.125,
}


class _StubConfig:
    """Just the weights AblatedModel reads off a model config."""

    def __init__(self, **overrides):
        for key, value in ALL_TERMS.items():
            setattr(self, "lambda_" + key[len("loss_") :], value)
        for key, value in overrides.items():
            setattr(self, key, value)


class _StubPredictor(nn.Module):
    def __init__(self, num_refine_steps=3):
        super().__init__()
        self.num_refine_steps = num_refine_steps
        self.seen_refine_steps = []


class _StubModel(nn.Module):
    """Minimal TextSpanJEPA stand-in: reports exactly the info it is given."""

    def __init__(self, info=None, config=None, raises=None, num_refine_steps=3):
        super().__init__()
        self.config = config or _StubConfig()
        self.info = dict(ALL_TERMS if info is None else info)
        self.raises = raises
        self.forward_calls = 0
        self.predictor = _StubPredictor(num_refine_steps)
        self.encoder = nn.Linear(2, 2)
        # TextSpanJEPA initialises target_encoder as a copy of the encoder.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def compute_loss_with_targets(
        self, masked_input_ids, original_input_ids, mask_positions, current_step=0, total_steps=1
    ):
        self.forward_calls += 1
        self.predictor.seen_refine_steps.append(self.predictor.num_refine_steps)
        if self.raises is not None:
            raise self.raises
        total = torch.zeros((), requires_grad=True)
        for value in self.info.values():
            total = total + float(value)
        return total, dict(self.info), {}

    def update_target_encoder(self, tau):
        with torch.no_grad():
            for p_q, p_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                p_k.data.mul_(tau).add_(p_q.data, alpha=1.0 - tau)


def _perturb_online(model, amount=1.0):
    """Move the online encoder, leaving the target behind (an optimizer step)."""
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(amount)


def _target_gap(model):
    """Max |target - online| across the encoder pair."""
    return max(
        float((p_k.detach() - p_q.detach()).abs().max())
        for p_q, p_k in zip(model.encoder.parameters(), model.target_encoder.parameters())
    )


def _inputs(batch=2, seq=8, vocab=50):
    ids = torch.randint(0, vocab, (batch, seq))
    mask = torch.zeros(batch, seq, dtype=torch.long)
    mask[:, 2:5] = 1
    return ids, ids, mask


def _tiny_jepa(**overrides):
    from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

    kwargs = {
        "vocab_size": 100,
        "max_seq_len": 16,
        "embed_dim": 32,
        "encoder_depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "predictor_embed_dim": 16,
        "predictor_depth": 1,
        "future_offsets": (1,),
        "num_refine_steps": 1,
    }
    kwargs.update(overrides)
    return TextSpanJEPA(TextSpanJEPAConfig(**kwargs))


def _real_train_fn(seed=1234, tau=0.99, do_ema=True, max_steps=2):
    """A standard training loop: forward, step, then the per-step EMA call."""

    def train_fn(ablated, n_steps):
        seed_everything(seed)
        gen = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam([p for p in ablated.parameters() if p.requires_grad], lr=1e-2)
        losses = []
        for step in range(min(n_steps, max_steps)):
            ids = torch.randint(0, 100, (2, 16), generator=gen)
            mask = torch.zeros(2, 16, dtype=torch.long)
            mask[:, 3:6] = 1
            loss, _info = ablated(ids, ids, mask, current_step=step, total_steps=n_steps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if do_ema:
                # What src.train.do_ema_update does once per step.
                ablated.update_target_encoder(tau)
            losses.append(float(loss.item()))
        return losses

    return train_fn


# ═══════════════════════════════════════════════════════════════════════════
#  D1 — a missing ablation target is a hard failure, not a silent no-op
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingTargetIsLoud:
    def test_missing_info_key_raises_instead_of_doing_nothing(self):
        """A renamed loss key must not leave the loss untouched and still 'pass'."""
        info = dict(ALL_TERMS)
        del info["loss_future"]  # e.g. the model renamed it to loss_future_lm
        model = _StubModel(info=info)
        ablated = abl_mod.AblatedModel(
            model, abl_mod.AblationConfig("no_future_loss", use_future_loss=False)
        )

        with pytest.raises(RuntimeError) as excinfo:
            ablated(*_inputs())
        assert "loss_future" in str(excinfo.value)

    def test_missing_weight_on_the_model_config_raises(self):
        model = _StubModel(config=_StubConfig(lambda_decoder=0.0))
        ablated = abl_mod.AblatedModel(model, abl_mod.AblationConfig("no_dec", use_decoder=False))

        with pytest.raises(RuntimeError) as excinfo:
            ablated(*_inputs())
        assert "lambda_decoder" in str(excinfo.value)

    @pytest.mark.parametrize("flag", ["use_predictor", "use_target_centering"])
    def test_flag_with_no_mechanism_raises(self, flag):
        """A flag nothing implements would otherwise run as a labelled no-op."""
        model = _StubModel()
        ablated = abl_mod.AblatedModel(model, abl_mod.AblationConfig(flag, **{flag: False}))

        with pytest.raises(RuntimeError) as excinfo:
            ablated(*_inputs())
        assert flag in str(excinfo.value)

    def test_error_type_is_public_and_specific(self):
        assert issubclass(abl_mod.AblationTargetMissingError, RuntimeError)

    def test_allow_missing_targets_is_the_explicit_escape_hatch(self):
        """A caller who really wants a no-op must say so, and be warned."""
        info = dict(ALL_TERMS)
        del info["loss_future"]
        model = _StubModel(info=info)
        ablated = abl_mod.AblatedModel(
            model,
            abl_mod.AblationConfig(
                "no_future_loss", use_future_loss=False, allow_missing_targets=True
            ),
        )

        with pytest.warns(RuntimeWarning, match="loss_future"):
            loss, out_info = ablated(*_inputs())

        assert torch.isfinite(loss)
        assert out_info["ablation_ineffective"]

    def test_every_standard_ablation_either_applies_or_refuses(self):
        """No shipped config may run through and quietly do nothing.

        For each standard ablation: either the forward raises, or the terms it
        removed are exactly the terms the config disabled.
        """
        # (ablation flag, key the model reports the term under)
        terms = (
            ("use_future_loss", "loss_future"),
            ("use_decoder", "loss_decoder"),
            ("use_span_loss", "loss_span"),
            ("use_variance_reg", "loss_variance"),
            ("use_covariance_reg", "loss_covariance"),
        )
        for name, config in abl_mod.ABLATION_CONFIGS.items():
            model = _StubModel()
            ablated = abl_mod.AblatedModel(model, config)
            expected = [key for flag, key in terms if not getattr(config, flag)]
            try:
                _loss, info = ablated(*_inputs())
            except RuntimeError:
                continue  # refused loudly — acceptable
            assert info["ablation_applied"] == expected, name


# ═══════════════════════════════════════════════════════════════════════════
#  D2 — mutated state survives an exception
# ═══════════════════════════════════════════════════════════════════════════


class TestRefinementStateIsRestored:
    def test_num_refine_steps_restored_when_the_forward_raises(self):
        model = _StubModel(raises=RuntimeError("boom"))
        ablated = abl_mod.AblatedModel(
            model, abl_mod.AblationConfig("no_refine", use_iterative_refinement=False)
        )

        with pytest.raises(RuntimeError, match="boom"):
            ablated(*_inputs())

        assert model.predictor.num_refine_steps == 3

    def test_num_refine_steps_restored_on_success(self):
        model = _StubModel(num_refine_steps=3)
        ablated = abl_mod.AblatedModel(
            model, abl_mod.AblationConfig("no_refine", use_iterative_refinement=False)
        )

        ablated(*_inputs())

        assert model.predictor.num_refine_steps == 3
        assert model.predictor.seen_refine_steps == [0]  # ablation did apply

    def test_a_crash_does_not_corrupt_a_later_forward(self):
        """The damage of a leaked mutation outlives the ablation that caused it."""
        model = _StubModel(raises=RuntimeError("boom"))
        ablated = abl_mod.AblatedModel(
            model, abl_mod.AblationConfig("no_refine", use_iterative_refinement=False)
        )
        with pytest.raises(RuntimeError):
            ablated(*_inputs())

        model.raises = None
        # A different arm, refinement NOT ablated: it must see the real
        # refinement depth, not the 0 the crashed forward left behind.
        later = abl_mod.AblatedModel(model, abl_mod.AblationConfig("full"))
        loss, _info = later(*_inputs())

        assert torch.isfinite(loss)
        assert model.predictor.seen_refine_steps == [0, 3]


# ═══════════════════════════════════════════════════════════════════════════
#  D3 — a crashed cell is visibly a crash
# ═══════════════════════════════════════════════════════════════════════════


def _crashing_train_fn(crash_on):
    def train_fn(ablated, n_steps):
        if ablated.config.name == crash_on:
            raise ValueError("kaboom")
        return [1.0, 0.5]

    return train_fn


class TestCrashesAreVisible:
    def test_run_all_marks_the_crashed_cell_failed(self):
        study = abl_mod.AblationStudy(_tiny_jepa(), _crashing_train_fn("no_vicreg"))
        results = study.run_all(n_steps=2, ablations=["no_future_loss", "no_vicreg"])

        cell = results["no_vicreg"]
        assert cell["status"] == "failed"
        assert cell["error_type"] == "ValueError"
        assert "kaboom" in cell["error"]

    def test_a_failed_cell_has_no_loss_to_plot(self):
        study = abl_mod.AblationStudy(_tiny_jepa(), _crashing_train_fn("no_vicreg"))
        results = study.run_all(n_steps=2, ablations=["no_vicreg"])

        assert "final_loss" not in results["no_vicreg"]
        assert "loss_history" not in results["no_vicreg"]

    def test_the_other_cells_survive_a_crash(self):
        study = abl_mod.AblationStudy(_tiny_jepa(), _crashing_train_fn("no_vicreg"))
        results = study.run_all(n_steps=2, ablations=["no_future_loss", "no_vicreg"])

        survivor = results["no_future_loss"]
        assert survivor["status"] == "ok"
        assert survivor["final_loss"] == pytest.approx(0.5)

    def test_the_aggregate_says_it_was_incomplete(self):
        study = abl_mod.AblationStudy(_tiny_jepa(), _crashing_train_fn("no_vicreg"))
        results = study.run_all(n_steps=2, ablations=["no_future_loss", "no_vicreg"])

        assert results.is_complete is False
        assert set(results.failures) == {"no_vicreg"}
        assert "INCOMPLETE" in repr(results)
        assert "no_vicreg" in results.summary()

    def test_a_clean_run_is_complete(self):
        study = abl_mod.AblationStudy(_tiny_jepa(), _real_train_fn())
        results = study.run_all(n_steps=2, ablations=["no_future_loss", "no_decoder"])

        assert results.is_complete is True
        assert results.failures == {}
        assert "INCOMPLETE" not in repr(results)

    def test_run_scaling_marks_the_crashed_cell_failed(self, monkeypatch):
        monkeypatch.setitem(
            abl_mod.MODEL_SIZE_CONFIGS,
            "micro",
            {
                "embed_dim": 32,
                "encoder_depth": 1,
                "num_heads": 4,
                "predictor_embed_dim": 16,
                "predictor_depth": 1,
                "mlp_ratio": 2.0,
            },
        )
        study = abl_mod.AblationStudy(_tiny_jepa(), _crashing_train_fn("no_vicreg"))
        results = study.run_scaling_ablations(
            n_steps=2, model_sizes=["micro"], ablations=["no_future_loss", "no_vicreg"]
        )

        cell = results["no_vicreg_micro"]
        assert cell["status"] == "failed"
        assert cell["model_size"] == "micro"
        assert "final_loss" not in cell
        assert results["no_future_loss_micro"]["status"] == "ok"
        assert results.is_complete is False
        assert set(results.failures) == {"no_vicreg_micro"}
        assert "INCOMPLETE" in results.summary()


# ═══════════════════════════════════════════════════════════════════════════
#  D4 — the "no EMA target" arm is actually no-EMA
# ═══════════════════════════════════════════════════════════════════════════


class TestNoEmaArmIsReal:
    def test_ema_entry_point_is_intercepted_by_the_wrapper(self):
        """src.train calls model.update_target_encoder(tau); the ablation hears it."""
        model = _StubModel()
        ablated = abl_mod.AblatedModel(model, abl_mod.AblationConfig("full"))
        _perturb_online(model)
        ablated.update_target_encoder(0.99)
        assert _target_gap(model) > 0.0  # EMA target legitimately lags

        model = _StubModel()
        ablated = abl_mod.AblatedModel(
            model, abl_mod.AblationConfig("no_ema", use_ema_target=False)
        )
        _perturb_online(model)
        ablated.update_target_encoder(0.99)
        assert _target_gap(model) == 0.0  # ablated: target IS the online encoder

    def test_no_ema_arm_keeps_target_online_across_training(self):
        """Two steps of a real JEPA, with the real per-step EMA entry point."""

        def trained_target_gap(ablation):
            ablated = abl_mod.AblatedModel(_tiny_jepa(), ablation)
            ablated.ablate_ema()  # what run_single does at init
            _real_train_fn()(ablated, 2)
            return _target_gap(ablated.model)

        assert trained_target_gap(abl_mod.AblationConfig("no_ema", use_ema_target=False)) == 0.0
        assert trained_target_gap(abl_mod.AblationConfig("full")) > 0.0

    def test_forward_re_syncs_the_target_for_the_ablated_arm(self):
        model = _StubModel()
        ablated = abl_mod.AblatedModel(
            model, abl_mod.AblationConfig("no_ema", use_ema_target=False)
        )
        _perturb_online(model)
        model.update_target_encoder(0.99)  # a caller-side EMA blend slipped in
        assert _target_gap(model) > 0.0

        ablated(*_inputs())

        assert _target_gap(model) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Determinism
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_two_identical_runs_agree(self):
        base = _tiny_jepa()
        study = abl_mod.AblationStudy(base, _real_train_fn(do_ema=False))
        first = study.run_single("no_future_loss", n_steps=2)
        second = study.run_single("no_future_loss", n_steps=2)

        assert first["loss_history"] == second["loss_history"]
        assert first["final_loss"] == second["final_loss"]

    def test_a_run_does_not_mutate_the_base_model(self):
        base = _tiny_jepa()
        before = [p.detach().clone() for p in base.parameters()]
        abl_mod.AblationStudy(base, _real_train_fn(do_ema=False)).run_single(
            "no_decoder", n_steps=2
        )

        for param, ref in zip(base.parameters(), before):
            assert torch.equal(param, ref)


# ═══════════════════════════════════════════════════════════════════════════
#  The reported-not-fixed items stay visible
# ═══════════════════════════════════════════════════════════════════════════


class TestKnownGaps:
    def test_the_full_cell_is_marked_not_trained(self):
        """run_all still reports final_loss 0 for 'full' — now it says so."""
        study = abl_mod.AblationStudy(_tiny_jepa(), _real_train_fn())
        results = study.run_all(n_steps=2, ablations=["full", "no_decoder"])

        assert results["full"]["trained"] is False
        assert results["full"]["status"] == "not_trained"
