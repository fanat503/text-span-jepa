# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Regression tests for the causal-claim defects in
# src/interp/causal_intervention.py and src/interp/causal_scrubbing.py.
#
# These modules previously made causal claims without performing any causal
# intervention: they ran the model once, edited the pooled output vector with
# linear algebra, and reported the result. The tests below pin the properties
# that make an intervention *real*:
#   - a forward hook fires inside the forward pass and changes the encoder output
#   - the change is transformed by downstream layers (not a post-hoc vector edit)
#   - a broken/mismatched callback RAISES instead of returning a favourable value
#   - a B=1 scrub actually scrubs something
#   - the n-point Spearman is tested against a permutation null
#   - the JEPA-vs-baseline verdict is not an artefact of representation norm
#
# All models here are tiny (embed_dim=32, depth=2, B<=2, T<=16) and CPU-only.

import pytest
import torch
from torch import nn

from src.interp.causal_intervention import (
    CausalIntervention,
    activation_intervention,
    intervention_predictability_score,
    monotonicity_excess,
    permutation_spearman_test,
)
from src.interp.causal_scrubbing import (
    CausalScrubber,
    FeatureHypothesis,
    InterventionPredictabilityScorer,
    resample_positions,
)
from src.models.encoder import TextSpanJEPLEncoder
from src.utils.seed import seed_everything

EMBED_DIM = 32
DEPTH = 2
VOCAB = 100
SEQ = 16


class MockModel(nn.Module):
    """Minimal stand-in for Text-SpanJEPA: an `.encoder` with real Blocks."""

    def __init__(self):
        super().__init__()
        self.encoder = TextSpanJEPLEncoder(
            vocab_size=VOCAB,
            max_seq_len=SEQ,
            embed_dim=EMBED_DIM,
            depth=DEPTH,
            num_heads=4,
        )


class NoBlocksModel(nn.Module):
    """An `.encoder` without `.blocks` — hooking must raise, not no-op."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM))


class _ScaledEncoder(nn.Module):
    """Wraps a real encoder and returns representations scaled by `scale`.

    This is a pure change of representation *scale*: the function computed by the
    blocks is bit-for-bit the same, only the magnitude of the returned
    representations differs (as with a missing final LayerNorm). Note that
    scaling the encoder's *weights* would NOT be a pure scale change — the
    attention softmax is not scale-equivariant.
    """

    def __init__(self, inner, scale):
        super().__init__()
        self.inner = inner
        self.blocks = inner.blocks  # so a hook can be installed normally
        self.embed_dim = inner.embed_dim
        self.scale = scale

    def forward(self, input_ids, return_intermediates=False):
        h, embeds = self.inner(input_ids)
        return h * self.scale, embeds


class RescaledModel(nn.Module):
    """A model identical to `base` except its representations are `scale`x bigger."""

    def __init__(self, base, scale=3.0):
        super().__init__()
        self.encoder = _ScaledEncoder(base.encoder, scale)


class _NeverRunsEncoder(nn.Module):
    """Exposes real blocks but never calls them — the hook cannot fire."""

    def __init__(self, inner):
        super().__init__()
        self.blocks = inner.blocks
        self.embed_dim = inner.embed_dim

    def forward(self, input_ids, return_intermediates=False):
        b, t = input_ids.shape
        return input_ids.new_zeros(b, t, self.embed_dim), None


class NeverRunsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _NeverRunsEncoder(
            TextSpanJEPLEncoder(
                vocab_size=VOCAB,
                max_seq_len=SEQ,
                embed_dim=EMBED_DIM,
                depth=DEPTH,
                num_heads=4,
            )
        )


@pytest.fixture(autouse=True)
def _deterministic():
    seed_everything(0)
    yield


@pytest.fixture
def ids():
    return torch.randint(0, VOCAB, (2, SEQ))


@pytest.fixture
def direction():
    return nn.functional.normalize(torch.randn(EMBED_DIM), dim=0)


@pytest.fixture
def probe():
    """Linear probe -> exercises the real prediction path."""

    w = nn.functional.normalize(torch.randn(EMBED_DIM), dim=0)

    def _probe(h):
        return (h @ w).mean()

    return _probe


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ═══════════════════════════════════════════════════════════════════
# 1. The intervention must be a real forward-pass intervention
# ═══════════════════════════════════════════════════════════════════


class TestRealIntervention:
    def test_hook_fires_and_changes_encoder_output(self, ids, direction):
        """Ablating at block 0 must change what the encoder returns."""
        model = MockModel().eval()
        with torch.no_grad():
            h_clean, _ = model.encoder(ids)

        with activation_intervention(model, direction, mode="ablate", layer_idx=0) as handle:
            assert handle.fired == 0
            with torch.no_grad():
                h_ablated, _ = model.encoder(ids)

        assert handle.fired == 1, "forward hook never fired -> intervention is a no-op"
        assert not torch.allclose(h_clean, h_ablated, atol=1e-6)

    def test_intervention_is_causal_not_a_posthoc_vector_edit(self, ids, direction):
        """Downstream layers must transform the edit.

        If the module merely added the direction to the pooled output vector, the
        final output would equal `h_clean.mean(1) + direction`. It must not.
        """
        model = MockModel().eval()
        with torch.no_grad():
            h_clean, _ = model.encoder(ids)

        with (
            activation_intervention(
                model, direction, mode="steer", layer_idx=0, scale=1.0
            ) as handle,
            torch.no_grad(),
        ):
            h_out, _ = model.encoder(ids)

        assert handle.fired == 1
        posthoc = h_clean.mean(dim=1) + direction
        assert not torch.allclose(h_out.mean(dim=1), posthoc, atol=1e-5)

    def test_last_layer_intervention_still_reaches_the_output(self, ids, direction):
        """layer_idx=-1 (default) must intervene, not silently be ignored."""
        model = MockModel().eval()
        with torch.no_grad():
            h_clean, _ = model.encoder(ids)
        with (
            activation_intervention(model, direction, mode="steer", layer_idx=-1, scale=5.0),
            torch.no_grad(),
        ):
            h_steered, _ = model.encoder(ids)
        assert not torch.allclose(h_clean, h_steered, atol=1e-6)

    def test_hook_is_removed_after_context(self, ids, direction):
        model = MockModel().eval()
        block = model.encoder.blocks[0]
        before = len(block._forward_hooks)
        with activation_intervention(model, direction, mode="ablate", layer_idx=0):
            assert len(block._forward_hooks) == before + 1
        assert len(block._forward_hooks) == before

    def test_hook_is_removed_even_when_forward_raises(self, direction):
        model = MockModel().eval()
        block = model.encoder.blocks[0]
        before = len(block._forward_hooks)
        with (
            pytest.raises(RuntimeError),
            activation_intervention(model, direction, mode="ablate", layer_idx=0),
        ):
            raise RuntimeError("boom")
        assert len(block._forward_hooks) == before

    def test_model_without_blocks_raises(self, ids, direction):
        """Silently skipping the hook would report a null intervention as a result."""
        with (
            pytest.raises((AttributeError, TypeError, ValueError)),
            activation_intervention(NoBlocksModel().eval(), direction, mode="ablate", layer_idx=0),
        ):
            pass

    def test_out_of_range_layer_raises(self, ids, direction):
        model = MockModel().eval()
        with (
            pytest.raises((IndexError, ValueError)),
            activation_intervention(model, direction, mode="ablate", layer_idx=99),
        ):
            pass

    def test_score_reports_hook_fires_per_scale(self, ids, direction, probe):
        model = MockModel().eval()
        scales = (-2, -1, 0, 1, 2)
        res = intervention_predictability_score(
            model,
            ids,
            direction,
            probe,
            scales=scales,
            layer_idx=0,
            n_permutations=199,
        )
        assert res["n_hook_calls"] == len(scales)

    def test_single_scale_is_reported_as_degenerate(self, ids, direction, probe):
        """One point is not a correlation; it must not yield a p-value < 1."""
        model = MockModel().eval()
        res = intervention_predictability_score(
            model, ids, direction, probe, scales=(1.0,), layer_idx=0
        )
        assert res["degenerate"] is True
        assert res["predictability"] == 0.0
        assert res["p_value"] == 1.0
        assert res["n_hook_calls"] == 1

    def test_score_probes_the_intervened_forward_pass(self, ids, direction):
        """The probe values must come from a hooked forward pass.

        This is the regression test for the original defect: the module used to
        add `scale * direction` to the pooled output vector. With a non-linear
        probe, a post-hoc edit and a real in-forward intervention give different
        numbers, so this pins the causal path through the public API.
        """
        model = MockModel().eval()

        def nonlinear_probe(h):
            return torch.tanh(h).sum(dim=-1).mean()

        scales = (-1.0, 0.5, 2.0)
        res = intervention_predictability_score(
            model,
            ids,
            direction,
            nonlinear_probe,
            scales=scales,
            layer_idx=0,
            n_permutations=99,
        )

        for scale, reported in zip(scales, res["probe_values"]):
            with (
                activation_intervention(
                    model,
                    direction,
                    mode="steer",
                    layer_idx=0,
                    scale=scale * res["reference_norm"],
                ) as handle,
                torch.no_grad(),
            ):
                h_ref, _ = model.encoder(ids)
            assert handle.fired == 1
            assert reported == pytest.approx(float(nonlinear_probe(h_ref.mean(dim=1))), rel=1e-5)

        # and the post-hoc edit it replaced gives different numbers
        with torch.no_grad():
            h_clean, _ = model.encoder(ids)
        posthoc = float(
            nonlinear_probe(h_clean.mean(dim=1) + (2.0 * res["reference_norm"]) * direction)
        )
        assert abs(res["probe_values"][-1] - posthoc) > 1e-3 * max(1.0, abs(posthoc))

    def test_score_raises_when_the_block_never_runs(self, ids, direction, probe):
        """A model whose block never executes cannot be intervened on."""
        with pytest.raises(RuntimeError):
            intervention_predictability_score(
                NeverRunsModel().eval(),
                ids,
                direction,
                probe,
                scales=(-1, 1),
                layer_idx=0,
            )

    def test_score_is_deterministic(self, ids, direction, probe):
        model = MockModel().eval()
        kwargs = {"scales": (-2, -1, 0, 1, 2), "layer_idx": 0, "n_permutations": 199, "seed": 7}
        a = intervention_predictability_score(model, ids, direction, probe, **kwargs)
        b = intervention_predictability_score(model, ids, direction, probe, **kwargs)
        assert a == b


# ═══════════════════════════════════════════════════════════════════
# 2. The permutation null
# ═══════════════════════════════════════════════════════════════════


class TestPermutationNull:
    def test_perfectly_monotone_signal_is_extreme(self):
        r, p = permutation_spearman_test(
            [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0], 199, _gen()
        )
        assert r == pytest.approx(1.0)
        # Exactly 2 of the 5! = 120 permutations are perfectly monotone (the
        # identity and its reverse), so even a flawless n=5 signal bottoms out
        # at p = 2/120. This is the power limit the old |r| number hid.
        assert p == pytest.approx(2 / 120)

    def test_unstructured_signal_is_not_significant(self):
        # same points, order scrambled -> no monotone relation
        r, p = permutation_spearman_test(
            [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 5.0, 4.0], 999, _gen()
        )
        assert abs(r) == pytest.approx(0.8)
        assert p > 0.05
        assert p <= 1.0

    def test_reversed_signal_is_equally_extreme(self):
        r, p = permutation_spearman_test(
            [0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 4.0, 3.0, 2.0, 1.0], 199, _gen()
        )
        assert r == pytest.approx(-1.0)
        assert p == pytest.approx(2 / 120)

    def test_sampled_null_for_more_than_seven_points(self):
        r, p = permutation_spearman_test(
            list(range(9)), [4.0, 0.0, 7.0, 1.0, 8.0, 2.0, 6.0, 3.0, 5.0], 99, _gen()
        )
        assert abs(r) < 0.5
        assert p > 0.05

    def test_score_reports_p_value_and_null_hypothesis(self, ids, direction):
        model = MockModel().eval()
        scales = (-2, -1, 0, 1, 2)
        res = intervention_predictability_score(
            model,
            ids,
            direction,
            lambda h: h.norm(dim=-1).mean(),
            scales=scales,
            layer_idx=0,
            n_permutations=199,
        )
        assert 0.0 < res["p_value"] <= 1.0
        assert res["n_permutations"] == 199
        # 5 points -> the null is enumerated exactly over 5! = 120 permutations
        assert res["n_null_draws"] == 120
        assert isinstance(res["null_hypothesis"], str)
        assert "no monotone" in res["null_hypothesis"]
        assert res["significant"] is (res["p_value"] < 0.05)
        # the reported p-value is the permutation null of the reported data, not
        # a placeholder: it must reproduce from probe_values independently
        _r, expected_p = permutation_spearman_test(
            scales, res["probe_values"], 199, torch.Generator().manual_seed(0)
        )
        assert res["p_value"] == pytest.approx(expected_p)
        assert res["spearman"] == pytest.approx(abs(res["predictability"]))

    def test_constant_signal_is_not_significant(self, ids, direction):
        model = MockModel().eval()
        res = intervention_predictability_score(
            model,
            ids,
            direction,
            lambda h: torch.tensor(1.0),
            scales=(-2, -1, 0, 1, 2),
            layer_idx=0,
            n_permutations=199,
        )
        assert res["degenerate"] is True
        assert res["predictability"] == 0.0
        assert res["monotonicity"] == 0.0
        assert res["p_value"] == 1.0
        assert res["significant"] is False

    def test_monotonicity_has_a_chance_floor_of_zero(self, ids, direction):
        """Old code reported max(frac>0, 1-frac>0) >= 0.5 for ANY signal.

        Monotonicity must now be reported as excess over chance, so a signal with
        no order information scores 0.0 and carries a binomial p-value.
        """
        model = MockModel().eval()
        res = intervention_predictability_score(
            model,
            ids,
            direction,
            lambda h: h.norm(dim=-1).mean(),
            scales=(-2, -1, 0, 1, 2),
            layer_idx=0,
            n_permutations=99,
        )
        assert 0.0 <= res["monotonicity"] <= 1.0
        assert 0.0 <= res["monotonicity_p_value"] <= 1.0
        assert res["monotonicity_significant"] is (res["monotonicity_p_value"] < 0.05)

    def test_alternating_signal_has_zero_monotonicity(self):
        """No order information -> 0.0 excess, and a non-significant p-value."""
        excess, p = monotonicity_excess([0.0, 1.0, 0.0, 1.0, 0.0])
        assert excess == 0.0
        assert p == pytest.approx(1.0)

    def test_perfectly_monotone_signal_has_maximum_monotonicity(self):
        excess, p = monotonicity_excess([0.0, 1.0, 2.0, 3.0, 4.0])
        assert excess == pytest.approx(1.0)
        # 4 consecutive diffs, all agreeing: the exact two-sided binomial floor
        # is 2/16 = 0.125, i.e. even a perfect n=5 signal cannot clear alpha=0.05
        assert p == pytest.approx(0.125)

    def test_constant_signal_has_zero_monotonicity(self):
        excess, p = monotonicity_excess([1.0, 1.0, 1.0, 1.0, 1.0])
        assert excess == 0.0
        assert p == 1.0

    def test_scorer_shares_the_real_intervention(self, ids, direction):
        """InterventionPredictabilityScorer duplicated the old no-op path."""
        model = MockModel().eval()
        res = InterventionPredictabilityScorer.compute_predictability(
            model,
            ids,
            direction,
            lambda h: h.norm(dim=-1).mean(),
            scales=(-2, -1, 0, 1, 2),
            layer_idx=0,
            n_permutations=99,
        )
        assert res["n_hook_calls"] == 5
        assert "p_value" in res


# ═══════════════════════════════════════════════════════════════════
# 3. The comparison must not be an artefact of representation norm
# ═══════════════════════════════════════════════════════════════════


class TestScaleMatchedComparison:
    @pytest.fixture
    def scaled_pair(self):
        # Bit-identical computation, representations 3x larger. Any "JEPA is
        # cleaner causal structure" verdict here is pure representation norm.
        base = MockModel().eval()
        return RescaledModel(base, 3.0).eval(), base

    def test_ablation_effect_is_scale_invariant(self, scaled_pair, ids, probe):
        small, big = scaled_pair
        d = nn.functional.normalize(torch.randn(EMBED_DIM), dim=0)
        res = CausalIntervention(big, small).ablation_comparison(
            ids,
            {"d": d},
            probe,
            layer_idx=0,
            n_random_directions=8,
            seed=0,
        )["d"]

        assert res["jepa_ablation_effect"] == pytest.approx(
            res["baseline_ablation_effect"], rel=1e-3
        )
        assert res["jepa_effect_z"] == pytest.approx(res["baseline_effect_z"], rel=1e-2)
        assert res["jepa_more_predictable"] is False
        assert isinstance(res["null_hypothesis"], str)

    def test_steering_verdict_is_scale_invariant(self, scaled_pair, ids, direction):
        small, big = scaled_pair
        res = CausalIntervention(big, small).steering_comparison(
            ids,
            direction,
            lambda h: h.norm(dim=-1).mean(),
            scales=(-2, -1, 0, 1, 2),
            layer_idx=0,
            n_permutations=199,
        )
        assert res["jepa_p_value"] == pytest.approx(res["baseline_p_value"], abs=1e-9)
        assert res["jepa_more_predictable"] is False
        assert res["null_hypothesis"]

    def test_ablation_comparison_intervenes_for_real(self, ids, probe):
        """The ablated probe value must come from an ablated forward pass."""
        model = MockModel().eval()
        d = nn.functional.normalize(torch.randn(EMBED_DIM), dim=0)
        ci = CausalIntervention(model, model)
        res = ci.ablation_comparison(
            ids, {"d": d}, probe, layer_idx=0, n_random_directions=8, seed=0
        )["d"]
        assert res["jepa_ablation_effect"] == pytest.approx(
            res["baseline_ablation_effect"], rel=1e-6
        ), "same model both sides must give identical effects"
        assert res["jepa_more_predictable"] is False
        # one intervened forward per direction per model, plus the null sample
        assert res["n_hook_calls"] == 2 * (1 + 8)

    def test_a_null_tie_is_not_evidence(self):
        """A tie in the discrete null must not be called a win on magnitude.

        Self-contained on purpose: the construction order (models, then ids, then
        probe weight, then direction) fixes the draws, and `n_random_directions=1`
        makes the null p-value coarse (0.5 or 1.0), which is what produces the
        tie. The old rule `jepa_effect > base_effect` called this a JEPA win
        purely on magnitude; a tie in the null is not evidence for either model.
        """
        seed_everything(0)
        a, b = MockModel().eval(), MockModel().eval()
        ids = torch.randint(0, VOCAB, (2, SEQ))
        w = nn.functional.normalize(torch.randn(EMBED_DIM), dim=0)
        d = nn.functional.normalize(torch.randn(EMBED_DIM), dim=0)

        res = CausalIntervention(a, b).ablation_comparison(
            ids,
            {"d": d},
            lambda h: (h @ w).mean(),
            layer_idx=0,
            n_random_directions=1,
            seed=0,
        )["d"]

        # a 5x magnitude gap, yet the nulls tie
        assert res["jepa_ablation_effect"] > 3 * res["baseline_ablation_effect"]
        assert res["jepa_effect_p"] == res["baseline_effect_p"]
        assert res["jepa_more_predictable"] is False


# ═══════════════════════════════════════════════════════════════════
# 4. Causal scrubbing: callbacks must not fail silently
# ═══════════════════════════════════════════════════════════════════


def _behavior_with_override(model, input_ids, h_override=None):
    """The documented contract: h_override=None means the clean forward pass.

    Reads every position, so a scrub of the late positions is observable.
    """
    with torch.no_grad():
        h = model.encoder(input_ids)[0] if h_override is None else h_override
    return float(h.sum())


def _behavior_two_arg(model, input_ids):
    with torch.no_grad():
        h, _ = model.encoder(input_ids)
    return float(h.sum())


def _position_hypothesis(h):
    """Keep the first half of the token positions relevant, scrub the rest."""
    cut = h.size(1) // 2
    relevant = torch.zeros_like(h, dtype=torch.bool)
    relevant[:, :cut, :] = True
    return relevant, ~relevant


def _all_relevant(h):
    mask = torch.ones_like(h, dtype=torch.bool)
    return mask, ~mask


class TestScrubCallbacks:
    def test_two_arg_behavior_fn_raises(self, ids):
        """Previously: TypeError swallowed -> 0.0 -> ratio 1.0 -> valid=True."""
        scrubber = CausalScrubber(MockModel().eval())
        with pytest.raises(TypeError, match="h_override"):
            scrubber.scrub_and_evaluate(ids, _position_hypothesis, _behavior_two_arg, n_resamples=2)

    def test_two_arg_behavior_fn_never_reports_a_favourable_result(self, ids):
        scrubber = CausalScrubber(MockModel().eval())
        try:
            out = scrubber.scrub_and_evaluate(
                ids, _position_hypothesis, _behavior_two_arg, n_resamples=2
            )
        except TypeError:
            return
        pytest.fail(f"must raise, got {out}")

    def test_hypothesis_fn_raising_propagates(self, ids):
        def bad_hypothesis(h):
            raise RuntimeError("hypothesis failure")

        scrubber = CausalScrubber(MockModel().eval())
        with pytest.raises(RuntimeError, match="hypothesis failure"):
            scrubber.scrub_and_evaluate(ids, bad_hypothesis, _behavior_with_override, n_resamples=2)

    def test_behavior_fn_raising_at_runtime_propagates(self, ids):
        """A well-formed callback that blows up mid-trial must not be swallowed.

        The old `except Exception: return 0.0` turned any crash into a behaviour
        of 0.0 — the "intervention failed, therefore the hypothesis is validated"
        path. The signature check cannot catch this, so the call must propagate.
        """

        def exploding_behavior(model, input_ids, h_override=None):
            if h_override is not None:
                raise RuntimeError("probe exploded")
            return 1.0

        scrubber = CausalScrubber(MockModel().eval())
        with pytest.raises(RuntimeError, match="probe exploded"):
            scrubber.scrub_and_evaluate(
                ids, _position_hypothesis, exploding_behavior, n_resamples=2
            )

    def test_nothing_to_scrub_raises(self, ids):
        """An all-relevant hypothesis scrubs nothing; reporting it as valid is
        the same defect as returning 0.0 for a failed intervention."""
        scrubber = CausalScrubber(MockModel().eval())
        with pytest.raises(ValueError, match="no-op"):
            scrubber.scrub_and_evaluate(ids, _all_relevant, _behavior_with_override, n_resamples=2)

    def test_masks_must_be_complementary(self, ids):
        def overlapping(h):
            m = torch.ones_like(h, dtype=torch.bool)
            return m, m

        scrubber = CausalScrubber(MockModel().eval())
        with pytest.raises(ValueError):
            scrubber.scrub_and_evaluate(ids, overlapping, _behavior_with_override, n_resamples=2)

    def test_scrub_end_to_end_with_valid_callback(self, ids):
        scrubber = CausalScrubber(MockModel().eval())
        res = scrubber.scrub_and_evaluate(
            ids, _position_hypothesis, _behavior_with_override, n_resamples=4
        )
        assert res["scrubbed_behavior_std"] >= 0.0
        # a real scrub of half the positions must move the behaviour
        assert res["scrubbed_behavior_mean"] != pytest.approx(res["baseline_behavior"])


# ═══════════════════════════════════════════════════════════════════
# 5. B=1 must not be a no-op
# ═══════════════════════════════════════════════════════════════════


class TestScrubResampling:
    def test_b1_resample_is_not_identity(self):
        h = torch.randn(1, 8, 16)
        out = resample_positions(h, _gen(0))
        assert out.shape == h.shape
        assert not torch.equal(out, h), "B=1 resample is a no-op (batch-axis shuffle)"

    def test_resample_only_moves_the_token_axis(self):
        h = torch.randn(1, 4, 8)
        out = resample_positions(h, _gen(1))
        # every output vector must be some *other* position of the same input
        rows = {tuple(h[0, t].tolist()) for t in range(h.size(1))}
        assert tuple(out[0, 0].tolist()) in rows

    def test_resample_is_a_permutation_per_batch_row(self):
        h = torch.randn(2, 6, 4)
        out = resample_positions(h, _gen(3))
        for b in range(h.size(0)):
            assert torch.equal(out[b].sort(dim=0).values, h[b].sort(dim=0).values)

    def test_single_token_input_raises(self):
        with pytest.raises(ValueError):
            resample_positions(torch.randn(2, 1, 8), _gen(0))

    def test_b1_scrub_changes_the_behaviour(self):
        """End-to-end: B=1 is the normal inference case, so the scrub must bite."""
        model = MockModel().eval()
        ids = torch.randint(0, VOCAB, (1, SEQ))
        scrubber = CausalScrubber(model)
        res = scrubber.scrub_and_evaluate(
            ids, _position_hypothesis, _behavior_with_override, n_resamples=4
        )
        assert res["scrubbed_behavior_mean"] != pytest.approx(res["baseline_behavior"])


# ═══════════════════════════════════════════════════════════════════
# 6. Failing decompositions must raise
# ═══════════════════════════════════════════════════════════════════


class TestHypothesisFailures:
    def test_non_finite_svd_raises(self):
        h = torch.randn(2, 4, 8)
        h[0, 0, 0] = float("nan")
        with pytest.raises(RuntimeError):
            FeatureHypothesis.svd_directions_hypothesis(h, n_relevant_dims=2)

    def test_svd_on_non_3d_raises(self):
        with pytest.raises(ValueError):
            FeatureHypothesis.svd_directions_hypothesis(torch.randn(4, 8), n_relevant_dims=2)

    def test_too_many_relevant_dims_raises(self):
        h = torch.randn(2, 4, 8)
        with pytest.raises(ValueError):
            FeatureHypothesis.svd_directions_hypothesis(h, n_relevant_dims=64)

    def test_random_hypothesis_is_seeded(self):
        h = torch.randn(2, 4, 8)
        a, _ = FeatureHypothesis.random_hypothesis(h, relevant_fraction=0.5, seed=0)
        b, _ = FeatureHypothesis.random_hypothesis(h, relevant_fraction=0.5, seed=0)
        c, _ = FeatureHypothesis.random_hypothesis(h, relevant_fraction=0.5, seed=1)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_position_hypothesis_is_seed_free_and_complementary(self):
        h = torch.randn(2, 5, 4)
        rel, irrel = FeatureHypothesis.position_based_hypothesis(h, keep_fraction=0.5)
        assert rel[:, :2, :].all() and irrel[:, 2:, :].all()
        assert not (rel & irrel).any()


# ═══════════════════════════════════════════════════════════════════
# 7. Determinism
# ═══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_scrub_and_evaluate_is_deterministic(self, ids):
        model = MockModel().eval()
        a = CausalScrubber(model).scrub_and_evaluate(
            ids, _position_hypothesis, _behavior_with_override, n_resamples=4, seed=3
        )
        b = CausalScrubber(model).scrub_and_evaluate(
            ids, _position_hypothesis, _behavior_with_override, n_resamples=4, seed=3
        )
        assert a == b

    def test_compare_scrubbing_is_deterministic(self, ids):
        import copy

        m1 = MockModel().eval()
        m2 = copy.deepcopy(m1)
        scrubber = CausalScrubber(m1)
        kw = {"n_resamples": 3, "seed": 1}
        a = scrubber.compare_scrubbing(
            m1, m2, ids, _position_hypothesis, _behavior_with_override, **kw
        )
        b = scrubber.compare_scrubbing(
            m1, m2, ids, _position_hypothesis, _behavior_with_override, **kw
        )
        assert a == b
        # identical models: no spurious "cleaner causal structure" verdict
        assert a["jepa_preservation"] == pytest.approx(a["baseline_preservation"], rel=1e-6)
        assert a["jepa_cleaner_causal_structure"] is False
