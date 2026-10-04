# Copyright 2026 Text-Span-JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Pins the train/validation split contract in ``src/interp/layer_analysis.py``.

The defect
----------
``LayerwiseProbe._train_linear_probe`` drew ``idx = torch.randperm(N)`` on
every call, from the process-global RNG. ``probe_all_layers`` calls it once
per layer, so a 12-layer accuracy profile was 12 measurements taken on 12
different train/validation partitions of the same data, and
``layer_uniformity`` -- a dispersion statistic over exactly those 12 numbers
-- inherited the union of that noise.

What is pinned here
-------------------
1.  ONE split per call, shared by every layer (the card's defect).
2.  The split is a function of an explicit ``seed``, drawn from a PRIVATE
    ``torch.Generator``. The process-global RNG is never consumed, so this
    module is not a hidden order-dependence vector for the rest of the suite.
3.  The same fix holds across the two *comparisons* the module actually
    performs: JEPA vs baseline in ``compare_layer_profiles``, and
    baseline vs leave-one-out in ``LayerRoutingAnalysis.routing_score``.
    Both are differences of two probe accuracies, so drawing the two halves
    from different partitions measures the split, not the ablation.
4.  The second unseeded per-layer draw -- the ``nn.Linear`` weight
    initialisation -- is closed too, because measurement (see
    ``.agent-notes/task-12.md``) shows it is the LARGER of the two noise
    channels: freezing the split alone widened the seed-to-seed band on
    byte-identical layers from 0.104 to 0.158. The replacement reproduces
    ``nn.Linear.reset_parameters``'s law from a private generator, and that
    equivalence is asserted bit-for-bit below rather than assumed.

Deliberately NOT pinned
-----------------------
``layer_uniformity = 1 - std/mean`` is maximised by making all layers
identical. The card reports this and explicitly makes the direction a
separate decision, so this file pins only the *formula*, and says so.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.interp import layer_analysis
from src.interp.layer_analysis import LayerRoutingAnalysis, LayerwiseProbe

# Deliberately tiny: these are 40x8 linear problems.
N, D, N_LAYERS = 40, 8, 4
SMALL = {"max_epochs": 8, "patience": 3, "device": "cpu"}


def make_data(n=N, d=D, seed=7):
    """One fixed dataset: a separable signal in dim 0 plus noise."""
    g = torch.Generator().manual_seed(seed)
    reps = torch.randn(n, d, generator=g)
    reps[:, 0] += 3.0 * (2.0 * (torch.rand(n, generator=g) < 0.5) - 1.0)
    labels = (torch.rand(n, generator=g) < 0.5).long()
    return reps, labels


def make_layers(count=N_LAYERS, n=N, d=D, seed=7):
    """``count`` byte-identical layers, so any spread in the profile is the
    module's doing and not the data's."""
    reps, labels = make_data(n, d, seed)
    return [reps.clone() for _ in range(count)], labels


class _RecordingProbe(LayerwiseProbe):
    """A ``LayerwiseProbe`` that remembers the split handed to each layer.

    This is how "one split, shared" is observed from outside. It binds to the
    ``split=`` / ``generator=`` keywords of ``_train_linear_probe``, which is
    deliberate: the contract under test *is* that a caller-supplied split
    exists and is threaded down.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.splits_seen = []
        self.generators_seen = []

    def _train_linear_probe(self, representations, labels, split=None, generator=None):
        self.splits_seen.append(split)
        self.generators_seen.append(generator)
        return super()._train_linear_probe(
            representations, labels, split=split, generator=generator
        )


def _distinct_splits(splits):
    """Count distinct non-None split payloads among the recorded calls."""
    seen = set()
    for s in splits:
        if s is None:
            continue
        train_idx, val_idx = s
        seen.add((tuple(train_idx.tolist()), tuple(val_idx.tolist())))
    return seen


# ═══════════════════════════════════════════════════════════════════
# The card's defect: one split per call, shared across layers
# ═══════════════════════════════════════════════════════════════════


class TestOneSplitPerCall:
    def test_every_layer_receives_a_split(self):
        """No layer may fall back to drawing its own."""
        layers, labels = make_layers()
        probe = _RecordingProbe(embed_dim=D, **SMALL)
        with torch.enable_grad():
            probe.probe_all_layers(layers, labels, "shared")

        assert len(probe.splits_seen) == N_LAYERS
        assert all(s is not None for s in probe.splits_seen), (
            "at least one layer was handed no split and drew its own: "
            f"{[s is None for s in probe.splits_seen]}"
        )

    def test_all_layers_receive_the_very_same_split(self):
        layers, labels = make_layers()
        probe = _RecordingProbe(embed_dim=D, **SMALL)
        with torch.enable_grad():
            probe.probe_all_layers(layers, labels, "shared")

        distinct = _distinct_splits(probe.splits_seen)
        assert len(distinct) == 1, (
            f"{len(probe.splits_seen)} layers were measured on {len(distinct)} "
            f"different train/validation partitions"
        )
        # Not merely equal in value -- the same object, so no per-layer redraw
        # can hide behind an accidental collision.
        assert len({id(s) for s in probe.splits_seen}) == 1

    def test_the_shared_split_is_the_one_that_is_reported(self):
        """``return_split=True`` must show the split the numbers came from."""
        layers, labels = make_layers()
        probe = _RecordingProbe(embed_dim=D, **SMALL)
        with torch.enable_grad():
            result = probe.probe_all_layers(layers, labels, "shared", return_split=True)

        reported = (tuple(result["train_idx"]), tuple(result["val_idx"]))
        assert _distinct_splits(probe.splits_seen) == {reported}

    def test_split_is_absent_from_the_payload_unless_requested(self):
        """The default payload stays small; index vectors are opt-in."""
        layers, labels = make_layers()
        probe = LayerwiseProbe(embed_dim=D, **SMALL)
        with torch.enable_grad():
            result = probe.probe_all_layers(layers, labels, "plain")
        assert "train_idx" not in result and "val_idx" not in result
        # ... but the provenance is always visible.
        assert result["split_seed"] == probe.seed
        assert result["n_train"] + result["n_val"] == N

    def test_compare_layer_profiles_uses_one_split_for_both_arms(self):
        """JEPA vs baseline is a difference of two accuracies, so the two
        arms must be scored on the same held-out rows."""
        layers, labels = make_layers()
        probe = _RecordingProbe(embed_dim=D, **SMALL)
        with torch.enable_grad():
            probe.compare_layer_profiles(layers, layers, labels, "cmp")

        assert len(probe.splits_seen) == 2 * N_LAYERS
        distinct = _distinct_splits(probe.splits_seen)
        assert len(distinct) == 1, (
            f"the two arms were measured on {len(distinct)} different partitions; "
            f"a cross-model difference is then a difference of splits"
        )

    def test_routing_score_uses_one_split_for_baseline_and_every_loo_arm(self, monkeypatch):
        """``routing[i, t] = baseline_acc - loo_acc``: a difference of two probe
        accuracies, one per probe, so every probe needs the same partition."""
        layers, labels = make_layers(count=3)
        seen = []

        class Spy(LayerwiseProbe):
            def _train_linear_probe(self, reps, lbl, split=None, generator=None):
                seen.append(split)
                return super()._train_linear_probe(reps, lbl, split=split, generator=generator)

        monkeypatch.setattr(layer_analysis, "LayerwiseProbe", Spy)
        with torch.enable_grad():
            LayerRoutingAnalysis.routing_score(layers, {"t": labels}, embed_dim=D)

        assert len(seen) == 1 + 3, "one baseline probe plus one per leave-one-out layer"
        assert all(s is not None for s in seen)
        assert len(_distinct_splits(seen)) == 1


# ═══════════════════════════════════════════════════════════════════
# The split is seeded, and the global RNG is untouched
# ═══════════════════════════════════════════════════════════════════


class TestSeededAndPrivate:
    def test_same_seed_gives_the_same_split(self):
        a = LayerwiseProbe.train_val_split(N, seed=3)
        b = LayerwiseProbe.train_val_split(N, seed=3)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_different_seeds_give_different_splits(self):
        """The seed must be load-bearing, not a constant in disguise."""
        seen = {tuple(LayerwiseProbe.train_val_split(N, seed=s)[0].tolist()) for s in range(5)}
        assert len(seen) > 1, f"seed does not change the split: {seen}"

    def test_the_split_is_a_permutation_of_every_row(self):
        train_idx, val_idx = LayerwiseProbe.train_val_split(N, seed=11)
        assert sorted(train_idx.tolist() + val_idx.tolist()) == list(range(N))

    def test_the_split_is_the_documented_fraction(self):
        train_idx, val_idx = LayerwiseProbe.train_val_split(N, seed=5, train_frac=0.75)
        assert train_idx.numel() == int(0.75 * N)
        assert val_idx.numel() == N - int(0.75 * N)

    def test_a_degenerate_fraction_is_refused(self):
        """`train_frac` that leaves one side empty is a config error, and an
        empty validation set would make the reported accuracy NaN."""
        for bad in (0.0, 1.0, 1.5, -0.2):
            with pytest.raises(ValueError):
                LayerwiseProbe.train_val_split(N, seed=0, train_frac=bad)

    def test_global_rng_is_not_consumed_by_the_split(self):
        torch.manual_seed(4242)
        a = LayerwiseProbe.train_val_split(N, seed=1)
        torch.manual_seed(999)
        torch.randn(500)
        b = LayerwiseProbe.train_val_split(N, seed=1)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_probe_all_layers_does_not_consume_the_global_rng(self):
        """The whole call, weights included. Before the fix, churning the
        global stream between two identical calls moved the result."""
        layers, labels = make_layers()
        probe = LayerwiseProbe(embed_dim=D, seed=2, **SMALL)
        with torch.enable_grad():
            a = probe.probe_all_layers(layers, labels, "g")
        torch.manual_seed(31337)
        _ = torch.randn(1000)
        with torch.enable_grad():
            b = probe.probe_all_layers(layers, labels, "g")
        assert a["per_layer_accuracy"] == b["per_layer_accuracy"]
        assert a["layer_uniformity"] == b["layer_uniformity"]

    def test_probe_all_layers_is_reproducible(self):
        layers, labels = make_layers()
        p1 = LayerwiseProbe(embed_dim=D, seed=6, **SMALL)
        p2 = LayerwiseProbe(embed_dim=D, seed=6, **SMALL)
        with torch.enable_grad():
            a = p1.probe_all_layers(layers, labels, "r")
            b = p2.probe_all_layers(layers, labels, "r")
        assert a["per_layer_accuracy"] == b["per_layer_accuracy"]

    def test_routing_score_is_reproducible_for_a_fixed_seed(self):
        layers, labels = make_layers(count=3)
        out = []
        for _ in range(2):
            with torch.enable_grad():
                out.append(
                    LayerRoutingAnalysis.routing_score(layers, {"t": labels}, embed_dim=D, seed=4)
                )
        assert torch.equal(out[0]["routing_matrix"], out[1]["routing_matrix"])


# ═══════════════════════════════════════════════════════════════════
# The noise band the card names: identical layers, different answers
# ═══════════════════════════════════════════════════════════════════


class TestProfileIsNotAmbientNoise:
    def test_the_seed_moves_the_profile_and_nothing_else_does(self):
        """Two seeds may differ; the global RNG must not be able to."""
        layers, labels = make_layers()
        out = []
        for s in (1, 2):
            torch.manual_seed(1234 + s)
            p = LayerwiseProbe(embed_dim=D, seed=s, **SMALL)
            with torch.enable_grad():
                out.append(p.probe_all_layers(layers, labels, "band"))
        assert (
            out[0]["per_layer_accuracy"] != out[1]["per_layer_accuracy"]
        ), "the seed is being ignored, so the split is a constant"

    def test_reruns_of_the_same_seed_collapse_to_zero_width(self):
        """The card's symptom: `layer_uniformity` moved by ~0.10 across 5
        seeds on byte-identical layers. Run-to-run at a fixed seed, the
        width must be exactly zero."""
        layers, labels = make_layers()
        widths = []
        for rep in range(4):
            torch.manual_seed(1000 + rep)
            p = LayerwiseProbe(embed_dim=D, seed=9, **SMALL)
            with torch.enable_grad():
                widths.append(p.probe_all_layers(layers, labels, "w")["layer_uniformity"])
        assert max(widths) - min(widths) == 0.0, widths


# ═══════════════════════════════════════════════════════════════════
# The weight initialisation: the second unseeded per-layer draw
# ═══════════════════════════════════════════════════════════════════


class TestSeededProbeInit:
    def test_seeded_linear_matches_nn_linear_reset_parameters(self):
        """Bit-for-bit, not 'statistically similar'.

        The private-generator replacement is only legitimate if it draws the
        same distribution in the same order. If a torch release ever changes
        ``nn.Linear.reset_parameters``, this goes red and says so, instead of
        the module silently drifting to a different initialisation law.
        """
        seed = 4242
        torch.manual_seed(seed)
        reference = nn.Linear(D, 3)

        gen = torch.Generator().manual_seed(seed)
        ours = LayerwiseProbe.seeded_linear(D, 3, generator=gen)

        assert torch.equal(reference.weight.detach(), ours.weight.detach())
        assert torch.equal(reference.bias.detach(), ours.bias.detach())
        assert ours.bias is not None

    def test_seeded_linear_supports_bias_free_construction(self):
        gen = torch.Generator().manual_seed(1)
        layer = LayerwiseProbe.seeded_linear(D, 2, generator=gen, bias=False)
        assert layer.bias is None
        assert layer.weight.shape == (2, D)

    def test_seeded_linear_never_touches_the_global_rng(self):
        torch.manual_seed(77)
        LayerwiseProbe.seeded_linear(D, 2, generator=torch.Generator().manual_seed(5))
        after_a = torch.randn(3)
        torch.manual_seed(77)
        LayerwiseProbe.seeded_linear(D, 2, generator=torch.Generator().manual_seed(5))
        after_b = torch.randn(3)
        assert torch.equal(after_a, after_b)

    def test_each_layer_gets_its_own_weight_stream(self):
        """A shared *split* with a shared *initialisation* would be a paired
        design; the module's existing estimand is one independent probe per
        layer. This pins which of the two it is, so the choice is visible."""
        layers, labels = make_layers()
        probe = _RecordingProbe(embed_dim=D, seed=3, **SMALL)
        with torch.enable_grad():
            probe.probe_all_layers(layers, labels, "w")
        assert all(g is not None for g in probe.generators_seen)
        # One generator threaded through the loop -> inits differ per layer.
        assert len({id(g) for g in probe.generators_seen}) == 1

    def test_identical_layers_still_differ_per_layer(self):
        """Documents the residual init channel that the report raises as a
        follow-up decision. Paired inits would collapse this to one value."""
        layers, labels = make_layers()
        probe = LayerwiseProbe(embed_dim=D, seed=3, **SMALL)
        with torch.enable_grad():
            acc = probe.probe_all_layers(layers, labels, "w")["per_layer_accuracy"]
        assert len(set(acc)) > 1, (
            "identical layers now score identically: the probe initialisation "
            "has been paired across layers, which is an estimand change and "
            "needs a decision (see .agent-notes/task-12.md)"
        )


# ═══════════════════════════════════════════════════════════════════
# Split hygiene
# ═══════════════════════════════════════════════════════════════════


class TestSplitHygiene:
    def test_too_few_rows_degrades_without_drawing_a_split(self):
        probe = LayerwiseProbe(embed_dim=D, seed=1, **SMALL)
        with torch.enable_grad():
            acc = probe._train_linear_probe(torch.randn(9, D), torch.zeros(9, dtype=torch.long))
        assert acc == 0.0

    def test_a_split_for_the_wrong_row_count_is_refused(self):
        """A split drawn for N=40 must not be silently applied to N=50."""
        stale = LayerwiseProbe.train_val_split(N, seed=0)
        probe = LayerwiseProbe(embed_dim=D, seed=0, **SMALL)
        reps, labels = make_data(n=50)
        with torch.enable_grad(), pytest.raises(ValueError):
            probe._train_linear_probe(reps, labels, split=stale)

    def test_an_empty_side_is_refused(self):
        """An empty validation set would report the accuracy of nothing."""
        empty = (torch.arange(N), torch.zeros(0, dtype=torch.long))
        probe = LayerwiseProbe(embed_dim=D, seed=0, **SMALL)
        reps, labels = make_data()
        with torch.enable_grad(), pytest.raises(ValueError):
            probe._train_linear_probe(reps, labels, split=empty)

    def test_a_label_vector_of_the_wrong_length_is_refused_by_routing_score(self):
        layers, labels = make_layers(count=3)
        with pytest.raises(ValueError), torch.enable_grad():
            LayerRoutingAnalysis.routing_score(layers, {"t": labels[: N // 2]}, embed_dim=D)

    def test_routing_score_on_a_single_layer_is_still_a_no_op(self):
        """Pre-existing contract: leave-one-out is undefined for one layer."""
        layers, labels = make_layers(count=1)
        with torch.enable_grad():
            out = LayerRoutingAnalysis.routing_score(layers, {"t": labels}, embed_dim=D)
        assert out["skipped"] is True


# ═══════════════════════════════════════════════════════════════════
# The formula, not the direction
# ═══════════════════════════════════════════════════════════════════


class TestUniformityFormula:
    """``layer_uniformity = 1 - std/mean``.

    The card reports that this is MAXIMISED by making all layers identical and
    explicitly makes the direction a separate decision, so the formula is
    pinned and the direction is not endorsed. If you change this metric,
    change this test deliberately.
    """

    def _uniformity(self, accs):
        """The formula the module documents, restated in the test."""
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        return 1.0 - std / mean if mean > 0 else 0.0

    def test_constant_accuracies_are_perfectly_uniform(self):
        assert self._uniformity([0.8] * 6) == pytest.approx(1.0)

    def test_the_formula_is_coefficient_of_variation_complement(self):
        assert self._uniformity([0.5, 0.5, 0.5, 0.9]) == pytest.approx(
            1.0 - ((sum((a - 0.6) ** 2 for a in [0.5, 0.5, 0.5, 0.9]) / 4) ** 0.5) / 0.6
        )

    def test_uniformity_falls_as_the_profile_spreads(self):
        """A restatement of the direction problem as an executable fact: the
        metric rewards sameness. Reported, not fixed -- see the card."""
        assert self._uniformity([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]) > self._uniformity(
            [0.8, 0.2, 0.8, 0.2, 0.8, 0.2]
        )

    def test_reported_uniformity_is_in_the_unit_interval(self):
        layers, labels = make_layers()
        probe = LayerwiseProbe(embed_dim=D, seed=0, **SMALL)
        with torch.enable_grad():
            result = probe.probe_all_layers(layers, labels, "u")
        assert 0.0 <= result["layer_uniformity"] <= 1.0
        assert result["layer_uniformity"] == max(min(result["layer_uniformity"], 1.0), 0.0)


# ═══════════════════════════════════════════════════════════════════
# The constructor keyword
# ═══════════════════════════════════════════════════════════════════


class TestSeedKeyword:
    def test_seed_is_a_keyword_and_defaults_to_zero(self):
        p = LayerwiseProbe(embed_dim=D, **SMALL)
        assert p.seed == 0
        assert LayerwiseProbe(embed_dim=D, seed=17, **SMALL).seed == 17

    def test_the_old_positional_signature_still_works(self):
        """Every existing caller passes embed_dim/lr/max_epochs/patience/device
        positionally or by keyword; none may break."""
        p = LayerwiseProbe(8, 2, 1e-3, 8, 3, "cpu")
        assert (p.embed_dim, p.num_classes, p.lr, p.max_epochs, p.patience, p.device) == (
            8,
            2,
            1e-3,
            8,
            3,
            "cpu",
        )
        assert p.seed == 0

    def test_train_frac_is_a_keyword(self):
        assert LayerwiseProbe(embed_dim=D, train_frac=0.5, **SMALL).train_frac == 0.5

    def test_seeded_linear_bound_matches_the_kaiming_identity(self):
        """Sanity on the constant, so a wrong `bound` cannot hide."""
        gen = torch.Generator().manual_seed(0)
        layer = LayerwiseProbe.seeded_linear(64, 2, generator=gen)
        assert float(layer.weight.detach().abs().max()) <= 1.0 / math.sqrt(64) + 1e-6
