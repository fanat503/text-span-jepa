# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Regression tests for the CKA primitive and the composite Interpretability Index.

Both modules produce the paper's headline numbers, and both had defects that a
plausible-looking test would not have caught:

* ``cka_metrics`` computed the **biased** V-statistic HSIC while documenting an
  unbiased one, reported a sample-count mismatch as ``0.0`` (i.e. maximum
  dissimilarity), spent O(N^3) where O(N^2) suffices, and used one shared RBF
  bandwidth for two inputs that may be on different scales.
* ``InterpretabilityIndex`` averaged over whatever metrics it happened to be
  given, so ``compare()`` subtracted two averages with different denominators,
  scored an empty metric dict at the neutral 0.5, and produced a number that
  isotropic white noise improves on.

Everything here is deterministic, CPU-only, and tiny (N <= 200, D <= 32) so the
file finishes in seconds.
"""

import math

import pytest
import torch

from src.interp.interpretability_index import (
    METRIC_DEFINITIONS,
    InterpretabilityIndex,
    geometry_metrics,
    noise_baseline_metrics,
)
from src.interp.visualization import NON_INDEX_LOWER_IS_BETTER, metric_direction, radar_chart
from src.utils.cka_metrics import (
    MIN_SAMPLES,
    _center,
    _hsic,
    cka_null_baseline,
    linear_cka,
    rbf_cka,
)

# Tiny but non-degenerate: enough samples for the unbiased estimator (N > 3),
# enough dimensions to be isotropic in.
N_SMALL, D_SMALL = 32, 8


def _independent_pair(n=N_SMALL, d=D_SMALL, seed=0):
    """Two independent Gaussian matrices — CKA should land on the null."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, d, generator=g)


# ═══════════════════════════════════════════════════════════════════
# CKA: the null floor
# ═══════════════════════════════════════════════════════════════════


class TestCKABias:
    """CKA on independent matrices must be small AND explainable.

    The closed form for two independent N x D Gaussians is ``D / (N + D + 1)``.
    Asserting against the closed form is a strictly stronger bias test than
    asserting "near 0": it pins the estimator to the value theory predicts at
    every N, so a regression in either direction fails.
    """

    @pytest.mark.parametrize("n", [32, 100, 200])
    def test_independent_matches_closed_form_null(self, n):
        d = D_SMALL
        vals = [linear_cka(*_independent_pair(n, d, seed=s)) for s in range(8)]
        measured = sum(vals) / len(vals)
        expected = cka_null_baseline(n, d)
        assert measured == pytest.approx(expected, abs=0.01), (
            f"independent CKA at N={n}, D={d} should equal the closed-form null "
            f"D/(N+D+1)={expected:.4f}, measured {measured:.4f}"
        )

    @pytest.mark.parametrize("n", [32, 100, 200])
    def test_independent_is_far_from_identical(self, n):
        val = linear_cka(*_independent_pair(n, D_SMALL, seed=n))
        assert val < 0.5, f"independent matrices scored {val:.3f} at N={n}"

    def test_null_floor_is_monotone_in_n(self):
        nulls = [cka_null_baseline(n, 768) for n in (32, 100, 400, 800)]
        assert all(
            a > b for a, b in zip(nulls, nulls[1:])
        ), f"the D/N confound must be visible in the reported null: {nulls}"
        # The regime that produced the bogus "CKA = 0.96 means similar" claim.
        assert cka_null_baseline(32, 768) > 0.9

    def test_stable_across_n_for_shared_structure(self):
        """A fixed 3-factor latent must score similarly at every N.

        This is the property the biased estimator violated: the value tracked N
        rather than the representations.
        """
        vals = []
        for n in (32, 100, 200):
            g = torch.Generator().manual_seed(7)
            z = torch.randn(n, 3, generator=g)
            x = z @ torch.randn(3, 16, generator=g) + 0.1 * torch.randn(n, 16, generator=g)
            y = z @ torch.randn(3, 16, generator=g) + 0.1 * torch.randn(n, 16, generator=g)
            vals.append(linear_cka(x, y))
        assert min(vals) > 0.5, f"shared-latent CKA collapsed with N: {vals}"
        assert max(vals) - min(vals) < 0.25, f"CKA of shared structure is N-unstable: {vals}"


class TestCKAInvariants:
    def test_identical_inputs_is_one(self):
        for n, d in ((32, 8), (100, 8), (200, 16)):
            x = torch.randn(n, d, generator=torch.Generator().manual_seed(n))
            assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_invariance(self):
        """CKA is invariant to an invertible linear map of one side."""
        d = 8
        x = torch.randn(100, d, generator=torch.Generator().manual_seed(3))
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(4)))
        assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-6)

    def test_row_permutation_invariance(self):
        n = 100
        x = torch.randn(n, 8, generator=torch.Generator().manual_seed(5))
        y = torch.randn(n, 8, generator=torch.Generator().manual_seed(6))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(7))
        assert linear_cka(x, y) == pytest.approx(linear_cka(x[perm], y[perm]), abs=1e-6)

    def test_different_feature_dims_allowed(self):
        x = torch.randn(100, 8, generator=torch.Generator().manual_seed(8))
        y = torch.randn(100, 24, generator=torch.Generator().manual_seed(9))
        assert 0.0 <= linear_cka(x, y) <= 1.0


class TestCKAShapeContract:
    """A shape mismatch used to be swallowed and reported as 0.0."""

    def test_mismatched_n_raises_not_zero(self):
        x = torch.randn(100, 8, generator=torch.Generator().manual_seed(10))
        y = torch.randn(120, 8, generator=torch.Generator().manual_seed(11))
        with pytest.raises(ValueError, match="same number of samples"):
            linear_cka(x, y)

    def test_rbf_mismatched_n_raises_not_zero(self):
        x = torch.randn(100, 8, generator=torch.Generator().manual_seed(12))
        y = torch.randn(120, 8, generator=torch.Generator().manual_seed(13))
        with pytest.raises(ValueError, match="same number of samples"):
            rbf_cka(x, y)

    def test_three_dimensional_input_raises(self):
        x = torch.randn(8, 4, 8, generator=torch.Generator().manual_seed(14))
        y = torch.randn(8, 4, 8, generator=torch.Generator().manual_seed(15))
        with pytest.raises(ValueError, match="2-D"):
            linear_cka(x, y)

    def test_too_few_samples_raises(self):
        x = torch.randn(MIN_SAMPLES - 1, 8)
        y = torch.randn(MIN_SAMPLES - 1, 8)
        with pytest.raises(ValueError, match="at least"):
            linear_cka(x, y)

    def test_zero_variance_input_is_undefined_geometry_not_a_comparison_error(self):
        """0/0 is a different failure mode from a shape mismatch.

        ``CollapseDiagnostics.compute`` calls into CKA with whatever it is
        handed and must stay total, so a constant input yields 0.0 rather than
        raising. Refusing to *report* a degenerate representation is the index's
        job, not this primitive's — see TestIndexNoiseGuard.
        """
        x = torch.randn(100, 8, generator=torch.Generator().manual_seed(16))
        y = torch.ones(100, 8)
        assert linear_cka(x, y) == 0.0
        assert linear_cka(y, y) == 0.0

    def test_collapse_is_refused_upstream_not_here(self):
        """The CKA primitive stays total; the index refuses collapsed reps."""
        res = InterpretabilityIndex().compute(
            {"collapsed_dim_ratio": 0.99, "mean_pairwise_cosine": 0.99}
        )
        assert res["interpretability_index"] is None
        assert res["degenerate"] is True


class TestRBFScaleInvariance:
    def test_invariant_to_common_global_rescaling(self):
        a, b = _independent_pair(100, 8, seed=21)
        base = rbf_cka(a, b)
        for factor in (1e-6, 1e-3, 1e3, 1e6):
            scaled = rbf_cka(a * factor, b * factor)
            assert scaled == pytest.approx(
                base, abs=1e-6
            ), f"rbf_cka changed under a common rescale by {factor}: {base} -> {scaled}"

    def test_not_invariant_to_relative_scale_mismatch(self):
        """The old single shared bandwidth turned a scale mismatch into 'similar'.

        With per-matrix medians the relative scale of x and y is normalised away,
        so this now reports the same number as the unscaled pair instead of a
        spurious 0.33.
        """
        a, b = _independent_pair(100, 8, seed=22)
        base = rbf_cka(a, b)
        mismatched = rbf_cka(a * 0.01, b * 100)
        assert mismatched == pytest.approx(base, abs=0.02), (
            f"rbf_cka is still driven by a relative scale mismatch: " f"{base} vs {mismatched}"
        )

    def test_identical_inputs_is_one(self):
        a = torch.randn(100, 8, generator=torch.Generator().manual_seed(23))
        assert rbf_cka(a, a) == pytest.approx(1.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════
# Index: denominators must match
# ═══════════════════════════════════════════════════════════════════


class TestIndexDenominators:
    def test_empty_metrics_raises_instead_of_scoring_half(self):
        with pytest.raises(ValueError, match="empty metric dict"):
            InterpretabilityIndex().compute({})

    def test_unknown_metrics_only_raises(self):
        with pytest.raises(ValueError, match="nothing to weight"):
            InterpretabilityIndex().compute({"not_a_real_metric": 1.0})

    def test_non_finite_metric_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            InterpretabilityIndex().compute({"sv_entropy": float("nan")})

    def test_missing_metrics_are_flagged(self):
        res = InterpretabilityIndex().compute({"sv_entropy": 0.7})
        assert res["unreliable"]
        assert any("of 24 known metrics" in r for r in res["unreliable_reasons"])

    def test_require_complete_raises(self):
        with pytest.raises(ValueError, match="require_complete"):
            InterpretabilityIndex().compute({"sv_entropy": 0.7}, require_complete=True)

    def test_silently_dropped_unknown_keys_are_reported(self):
        res = InterpretabilityIndex().compute({"sv_entropy": 0.7, "typo_metric": 3.0})
        assert res["unknown_metrics"] == ["typo_metric"]
        assert any("silently dropped" in r for r in res["unreliable_reasons"])


def _direction_aware(value):
    """A metric vector that is uniformly *good* or uniformly *bad*.

    Lower-is-better metrics get the low number for "good" and a high one for
    "bad" (and vice versa), so a "good" vector wins every component it
    contains. The "bad" levels stay clear of the collapse guard
    (collapsed_dim_ratio < 0.9, mean_pairwise_cosine < 0.95, and never both
    effective_rank and sv_entropy at their floor).
    """
    if value == "good":
        low, high = 0.1, 0.8
    else:
        # high must stay above COLLAPSE_ENTROPY_MAX so that bad is merely
        # worse, not degenerate — otherwise the guard refuses it and the
        # comparison has no verdict to check.
        low, high = 0.85, 0.3
    return {
        k: (low if METRIC_DEFINITIONS[k]["direction"] == "lower" else high)
        for k in METRIC_DEFINITIONS
    }


class TestIndexCompare:
    def test_differing_metric_sets_raises(self):
        idx = InterpretabilityIndex()
        jepa = {"effective_rank": 40.0, "sv_entropy": 0.7}
        base = {"effective_rank": 40.0, "sv_entropy": 0.7, "anisotropy": 0.1}
        with pytest.raises(ValueError, match="same metric set"):
            idx.compare(jepa, base)

    def test_differing_sets_report_undefined_when_not_strict(self):
        idx = InterpretabilityIndex()
        jepa = {"effective_rank": 40.0, "sv_entropy": 0.7}
        base = {"effective_rank": 40.0, "sv_entropy": 0.7, "anisotropy": 0.1}
        res = idx.compare(jepa, base, strict_metric_set=False)
        assert res["comparison_defined"] is False
        assert any("same metric set" in r for r in res["comparison_reasons"])

    def test_the_original_contradiction_is_impossible(self):
        """4 good metrics vs the same 4 plus 20 catastrophic ones.

        Previously this produced ``jepa_better=True, index_gap=+0.318`` from the
        same call that reported ``jepa_wins = 0/24``. The aggregate and the
        components now share a denominator, so they cannot disagree.
        """
        idx = InterpretabilityIndex()
        shared = {k: 0.5 for k in list(idx.weights)[:4]}
        padded = dict(shared)
        for k in list(idx.weights)[4:]:
            padded[k] = -1e6
        with pytest.raises(ValueError, match="same metric set"):
            idx.compare(shared, padded)

    def test_matched_sets_agree_between_gap_and_wins(self):
        """The aggregate and the per-component tally share a denominator now.

        The old contradiction was ``jepa_better=True, index_gap=+0.318`` from
        the same call that reported ``jepa_wins = 0/24``. With one metric set,
        winning zero components forces a non-positive weighted gap, because
        every normalized component is below its counterpart.
        """
        idx = InterpretabilityIndex()
        jepa = _direction_aware("good")
        bad = _direction_aware("bad")
        res = idx.compare(jepa, bad)
        assert res["n_metrics_compared"] == len(METRIC_DEFINITIONS)
        assert res["jepa_wins_fraction"] == 1.0
        assert res["jepa_better"] is True
        assert res["index_gap"] > 0

        flipped = idx.compare(bad, jepa)
        assert flipped["jepa_wins_fraction"] == 0.0
        assert flipped["jepa_better"] is False
        assert flipped["index_gap"] < 0

        # The invariant the old code violated: zero wins cannot coexist with a
        # positive gap once both sides share a denominator.
        assert not (flipped["jepa_wins_fraction"] == 0.0 and flipped["index_gap"] > 0)

    def test_comparison_reasons_are_reported_not_hidden(self):
        """A partial comparison is 'not fully defined', and says why."""
        idx = InterpretabilityIndex()
        keys = list(idx.weights)[:6]
        good = _direction_aware("good")
        bad = _direction_aware("bad")
        res = idx.compare(
            {k: good[k] for k in keys},
            {k: bad[k] for k in keys},
        )
        assert res["comparison_defined"] is False
        assert res["comparison_reasons"], "an undefined comparison must say why"
        assert any("of 24 known metrics" in r for r in res["comparison_reasons"])
        # The verdict is still emitted, because nothing is degenerate.
        assert res["jepa_better"] is True
        assert res["jepa_wins_n_out_of"] == f"{len(keys)}/{len(keys)}"

    def test_jepa_better_is_none_when_a_side_is_degenerate(self):
        idx = InterpretabilityIndex()
        # collapsed_dim_ratio >= 0.9 is collapse on the metric's own terms.
        jepa = {"collapsed_dim_ratio": 0.99, "sv_entropy": 0.2}
        base = {"collapsed_dim_ratio": 0.1, "sv_entropy": 0.9}
        res = idx.compare(jepa, base)
        assert res["jepa_details"]["interpretability_index"] is None
        assert res["jepa_better"] is None
        assert res["index_gap"] is None
        assert res["comparison_defined"] is False
        assert any("collapse" in r for r in res["comparison_reasons"])


# ═══════════════════════════════════════════════════════════════════
# Index: the white-noise guard
# ═══════════════════════════════════════════════════════════════════


def _structured_and_noisy(n=200, d=8, seed=11):
    """An 8-dim rep built from 2 latent factors, plus an isotropic-noise copy."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 2, generator=g)
    w = torch.randn(2, d, generator=g)
    structured = (z @ w) * 3.0
    noisy = structured + torch.randn(n, d, generator=g) * 3.0
    return structured, noisy


class TestIndexNoiseGuard:
    def test_white_noise_cannot_raise_the_index(self):
        """The headline defect: adding white noise moved 0.12 -> 0.695.

        A noise-dominated representation must either refuse the headline or
        score no higher. It must never come out ahead.
        """
        n, d = 200, 8
        structured, noisy = _structured_and_noisy(n, d)
        baseline = noise_baseline_metrics(n_samples=n, n_features=d, n_draws=8)
        idx = InterpretabilityIndex()

        clean = idx.compute(geometry_metrics(structured), noise_baseline=baseline)
        dirty = idx.compute(geometry_metrics(noisy), noise_baseline=baseline)

        if dirty["interpretability_index"] is not None:
            # Not refused: then it must not be a meaningful improvement.
            assert dirty["interpretability_index"] <= clean["interpretability_index"] + 1e-9, (
                f"white noise raised the index: {clean['interpretability_index']:.4f} -> "
                f"{dirty['interpretability_index']:.4f}"
            )
        else:
            assert dirty["unreliable_reasons"], "a refusal must carry a reason"
            assert any("isotropic noise" in r for r in dirty["unreliable_reasons"])

    def test_pure_noise_is_refused_with_a_reason(self):
        n, d = 200, 8
        baseline = noise_baseline_metrics(n_samples=n, n_features=d, n_draws=8)
        res = InterpretabilityIndex().compute(
            geometry_metrics(torch.randn(n, d, generator=torch.Generator().manual_seed(1))),
            noise_baseline=baseline,
        )
        assert res["interpretability_index"] is None
        assert res["degenerate"] is True
        assert res["interpretability_index_raw"] is not None
        reasons = " ".join(res["unreliable_reasons"])
        assert "isotropic noise" in reasons
        assert "indistinguishable" in reasons

    def test_clean_structure_is_not_refused(self):
        """The guard must not condemn a genuinely structured representation."""
        n, d = 200, 8
        structured, _ = _structured_and_noisy(n, d)
        baseline = noise_baseline_metrics(n_samples=n, n_features=d, n_draws=8)
        res = InterpretabilityIndex().compute(geometry_metrics(structured), noise_baseline=baseline)
        assert res["interpretability_index"] is not None
        assert res["degenerate"] is False
        assert 0.0 <= res["interpretability_index"] <= 1.0

    def test_noise_floor_is_disclosed_next_to_the_score(self):
        n, d = 200, 8
        baseline = noise_baseline_metrics(n_samples=n, n_features=d, n_draws=8)
        res = InterpretabilityIndex().compute(
            geometry_metrics(torch.randn(n, d, generator=torch.Generator().manual_seed(2))),
            noise_baseline=baseline,
        )
        nb = res["noise_baseline"]
        assert nb["covered_metrics"], "the noise control must cover the geometry metrics"
        assert nb["noise_index"] is not None
        assert nb["excess_over_noise"] is not None
        assert nb["saturated_weight_fraction"] >= 0.5
        assert (
            nb["excluded_metrics"] == []
        ), "a pure geometry metric set must have nothing excluded from the control"

    def test_collapse_is_detected_without_any_baseline(self):
        res = InterpretabilityIndex().compute(
            {"collapsed_dim_ratio": 0.97, "mean_pairwise_cosine": 0.99}
        )
        assert res["interpretability_index"] is None
        reasons = " ".join(res["unreliable_reasons"])
        assert "collapsed_dim_ratio" in reasons
        assert "collapse" in reasons

    def test_no_baseline_means_the_guard_is_declared_inert(self):
        res = InterpretabilityIndex().compute({"effective_rank": 40.0})
        assert res["interpretability_index"] is not None
        assert any("no white-noise baseline" in r for r in res["unreliable_reasons"])

    def test_baseline_is_deterministic_and_does_not_disturb_global_rng(self):
        torch.manual_seed(1234)
        expected_next = torch.randn(3)
        torch.manual_seed(1234)
        a = noise_baseline_metrics(n_samples=64, n_features=8, n_draws=4)
        actual_next = torch.randn(3)
        b = noise_baseline_metrics(n_samples=64, n_features=8, n_draws=4)
        assert torch.equal(expected_next, actual_next), "global RNG state was disturbed"
        assert a["metrics"] == b["metrics"]

    def test_index_reports_that_noise_outscores_structure(self):
        """The honest finding the index cannot fix on its own.

        As specified, the direction table puts white noise ABOVE a clean
        2-factor representation. The guard flags it; it cannot re-order it.
        """
        n, d = 200, 8
        structured, _ = _structured_and_noisy(n, d)
        baseline = noise_baseline_metrics(n_samples=n, n_features=d, n_draws=8)
        idx = InterpretabilityIndex()
        clean = idx.compute(geometry_metrics(structured), noise_baseline=baseline)
        noise = idx.compute(geometry_metrics(torch.randn(n, d)), noise_baseline=baseline)
        assert noise["interpretability_index_raw"] > clean["interpretability_index_raw"]
        assert clean["noise_baseline"]["excess_over_noise"] < 0


# ═══════════════════════════════════════════════════════════════════
# One source of truth for metric direction
# ═══════════════════════════════════════════════════════════════════


class TestMetricDirectionIsSingleSource:
    def test_agrees_with_metric_definitions_for_every_index_key(self):
        for name, defn in METRIC_DEFINITIONS.items():
            assert metric_direction(name) == defn["direction"], (
                f"{name}: visualization says {metric_direction(name)!r}, "
                f"METRIC_DEFINITIONS says {defn['direction']!r}"
            )

    def test_no_index_key_is_redeclared_locally(self):
        """The old local set omitted four indexed metrics; that must not recur."""
        for name in (
            "total_correlation",
            "intrinsic_dim",
            "intrinsic_dim_score",
            "probing_complexity",
        ):
            assert name not in NON_INDEX_LOWER_IS_BETTER, (
                f"{name} is an indexed metric; its direction belongs to "
                f"METRIC_DEFINITIONS, not to the visualization-only set"
            )

    def test_exact_lookup_not_substring(self):
        # "cv_effective_dim" contains "cv" but is its own indexed metric.
        assert (
            metric_direction("cv_effective_dim")
            == METRIC_DEFINITIONS["cv_effective_dim"]["direction"]
        )
        assert metric_direction("mean_pairwise_cosine") == "lower"
        assert metric_direction("totally_unknown_metric") == "higher"

    def test_radar_chart_uses_the_canonical_direction(self):
        """A lower-is-better metric must be plotted inverted."""
        metrics = {
            "effective_rank": 40.0,
            "sv_entropy": 0.7,
            "anisotropy": 0.1,
        }
        baseline = dict(metrics, anisotropy=0.9)
        svg = radar_chart(metrics, baseline, title="dir")
        assert svg is not None
        # Inverting the lower-is-better anisotropy must change the geometry.
        swapped = radar_chart(dict(metrics, anisotropy=0.9), baseline, title="dir")
        assert svg != swapped


class TestHSICCentering:
    """The estimator drops Kornblith's 1'Kh1 terms because HKH has zero sums.

    If ``_center`` does not reproduce ``H @ m @ H`` exactly, those terms stop
    being zero and the "unbiased" estimator silently becomes wrong. For a
    linear Gram matrix the grand mean is already 0, so the bug is invisible
    there and only shows up on a raw RBF kernel — hence an explicit test on a
    kernel that is *not* double-centered.
    """

    def test_center_matches_explicit_HmH(self):
        n = 40
        g = torch.Generator().manual_seed(31)
        # float64: the equivalence is exact in exact arithmetic, and float32
        # cannot resolve it to better than ~5e-7 on O(1) entries.
        m = torch.rand(n, n, generator=g, dtype=torch.float64) + 0.5
        h = torch.eye(n, dtype=torch.float64) - 1.0 / n
        assert torch.allclose(_center(m), h @ m @ h, atol=1e-12)

    def test_centered_kernel_has_zero_row_and_column_sums(self):
        n = 40
        g = torch.Generator().manual_seed(32)
        d = torch.cdist(
            torch.randn(n, 5, generator=g, dtype=torch.float64),
            torch.randn(n, 5, generator=g, dtype=torch.float64),
        )
        k = torch.exp(-0.5 * d**2)
        centered = _center(k)
        assert centered.sum(dim=1).abs().max() < 1e-10
        assert centered.sum(dim=0).abs().max() < 1e-10

    def test_rbf_cka_is_not_inflated_by_a_noncentered_kernel(self):
        """A regression guard on the bug this class documents.

        A mis-centred raw RBF kernel adds a constant to every entry, which
        inflates CKA towards 1 for unrelated inputs. Independent matrices on a
        median-heuristic bandwidth must not land near 1.
        """
        a, b = _independent_pair(100, 8, seed=33)
        val = rbf_cka(a, b)
        assert val < 0.9, (
            f"rbf_cka on independent matrices came out {val:.4f}, which is the "
            f"signature of a non-double-centered kernel, not of real similarity"
        )

    def test_unbiased_hsic_of_a_kernel_with_itself_is_nonnegative(self):
        n = 60
        g = torch.Generator().manual_seed(34)
        x = torch.randn(n, 6, generator=g)
        k = x @ x.T
        assert _hsic(k, k).item() > 0

    def test_kornblith_diagonal_terms_actually_vanish(self):
        """The estimator drops 1'Kh1 because HKH has zero row sums.

        If a future refactor breaks the centering, these terms stop being zero
        and the estimator silently changes. Assert they are negligible on a
        kernel that is NOT already double-centered.
        """
        n = 60
        g = torch.Generator().manual_seed(35)
        d = torch.cdist(
            torch.randn(n, 4, generator=g, dtype=torch.float64),
            torch.randn(n, 4, generator=g, dtype=torch.float64),
        )
        k = torch.exp(-0.5 * d**2 / 2.0**2)
        kh = _center(k)
        assert abs(kh.sum().item()) < 1e-9
        # And the full Kornblith bracket reduces to tr(Kh Lh) because of it.
        lhs = _hsic(k, k) * (n * (n - 3.0))
        rhs = (kh * kh).sum()
        assert lhs.item() == pytest.approx(rhs.item(), rel=1e-9)


class TestUnbiasedHSICIsWhatItClaims:
    """Pin the measured fact that the diagonal correction cancels in CKA.

    The original audit blamed the 0.96-on-independent-matrices result on the
    biased V-statistic. It is not: the correction is a scalar multiplying
    numerator and both denominator terms, so it cancels. Anyone who 'fixes'
    this again expecting the value to move will be disappointed; this test says
    so up front.
    """

    @staticmethod
    def _biased_ratio(K, L):
        """The original implementation's CKA: biased V-statistic numerator."""
        n = K.size(0)
        h = torch.eye(n, dtype=K.dtype) - 1.0 / n
        num = torch.trace(K @ h @ L @ h) / ((n - 1) ** 2)
        kk = torch.trace(K @ h @ K @ h) / ((n - 1) ** 2)
        ll = torch.trace(L @ h @ L @ h) / ((n - 1) ** 2)
        return (num / (kk * ll).sqrt()).item()

    def test_correction_cancels_for_linear_kernel(self):
        n, d = 100, 8
        g = torch.Generator().manual_seed(36)
        x = torch.randn(n, d, generator=g, dtype=torch.float64)
        y = torch.randn(n, d, generator=g, dtype=torch.float64)
        x = x - x.mean(0, keepdim=True)
        y = y - y.mean(0, keepdim=True)
        K, L = x @ x.T, y @ y.T
        unbiased = (_hsic(K, L) / (_hsic(K, K) * _hsic(L, L)).sqrt()).item()
        assert self._biased_ratio(K, L) == pytest.approx(unbiased, abs=1e-9)

    def test_correction_cancels_for_rbf_kernel(self):
        n = 100
        g = torch.Generator().manual_seed(37)
        a = torch.randn(n, 8, generator=g, dtype=torch.float64)
        b = torch.randn(n, 8, generator=g, dtype=torch.float64)
        sa = float(torch.pdist(a).median())
        sb = float(torch.pdist(b).median())
        K = torch.exp(-0.5 * torch.cdist(a, a) ** 2 / sa**2)
        L = torch.exp(-0.5 * torch.cdist(b, b) ** 2 / sb**2)
        unbiased = (_hsic(K, L) / (_hsic(K, K) * _hsic(L, L)).sqrt()).item()
        assert self._biased_ratio(K, L) == pytest.approx(unbiased, abs=1e-9)


def test_cka_null_baseline_rejects_degenerate_shapes():
    with pytest.raises(ValueError):
        cka_null_baseline(0, 8)
    with pytest.raises(ValueError):
        cka_null_baseline(10, -1)


def test_noise_baseline_rejects_too_few_draws():
    with pytest.raises(ValueError, match="n_draws"):
        noise_baseline_metrics(n_samples=32, n_features=8, n_draws=0)


def test_math_helpers_are_sane():
    """Guard against a vacuous suite: the closed forms are actually used."""
    assert math.isclose(cka_null_baseline(31, 8), 8 / 40, rel_tol=1e-12)
    assert len(METRIC_DEFINITIONS) == 24
    assert MIN_SAMPLES == 4
