# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Tests for seed aggregation: sample std, t intervals, pairing, and formatting.

The numbers are derived in the test from the definition (mean, ddof=1 variance,
t(0.975, n-1)) rather than copied from the implementation, so a change in the
convention fails here instead of silently moving every paper table.
"""

from __future__ import annotations

import math

import pytest

from src.utils import seed_stats as SS


class TestSummarize:
    def test_known_mean_and_sample_std(self):
        s = SS.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["n"] == 5
        assert s["mean"] == pytest.approx(3.0)
        # ddof=1: sum of squared deviations = 10, variance = 10/4.
        expected_std = math.sqrt(10.0 / 4.0)
        assert s["std"] == pytest.approx(expected_std)
        assert s["sem"] == pytest.approx(expected_std / math.sqrt(5))
        assert s["min"] == 1.0 and s["max"] == 5.0

    def test_interval_uses_student_t_not_normal(self):
        s = SS.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["ci_low"] == pytest.approx(3.0 - 2.776 * s["sem"])
        assert s["ci_high"] == pytest.approx(3.0 + 2.776 * s["sem"])
        # A normal 1.96 interval would be narrower; the test above pins the
        # difference, this guards the direction of the mistake.
        assert (s["ci_high"] - s["ci_low"]) > 2 * 1.96 * s["sem"]

    def test_two_seeds_get_a_wide_interval(self):
        s = SS.summarize([1.0, 2.0])
        assert s["std"] == pytest.approx(math.sqrt(0.5))
        assert s["sem"] == pytest.approx(0.5)
        # df=1 -> t=12.706, not 1.96.
        assert s["ci_low"] == pytest.approx(1.5 - 12.706 * 0.5)
        assert s["ci_high"] == pytest.approx(1.5 + 12.706 * 0.5)

    def test_single_seed_has_no_interval(self):
        s = SS.summarize([4.0])
        assert s["n"] == 1
        assert s["mean"] == 4.0
        assert math.isnan(s["std"])
        assert math.isnan(s["sem"])
        assert math.isnan(s["ci_low"]) and math.isnan(s["ci_high"])
        assert s["min"] == s["max"] == 4.0

    def test_empty_is_nan_not_zero(self):
        s = SS.summarize([])
        assert s["n"] == 0
        for key in ("mean", "std", "sem", "ci_low", "ci_high", "min", "max"):
            assert math.isnan(s[key]), f"{key} must be NaN for n=0"

    def test_non_finite_entries_are_dropped_and_n_reports_survivors(self):
        s = SS.summarize([1.0, float("nan"), float("inf"), 2.0, None])
        assert s["n"] == 2
        assert s["mean"] == pytest.approx(1.5)

    def test_other_confidence_levels_refuse(self):
        with pytest.raises(ValueError, match="95%"):
            SS.summarize([1.0, 2.0, 3.0], confidence=0.99)

    def test_t_table_boundary(self):
        assert SS._t_critical(30) == 2.042
        assert SS._t_critical(31) == 1.96
        with pytest.raises(ValueError):
            SS._t_critical(0)


class TestAggregateRuns:
    def test_union_of_keys_and_per_key_n(self):
        runs = [{"a": 1.0, "b": 2.0}, {"a": 3.0}, {"a": 5.0, "b": 8.0}]
        out = SS.aggregate_runs(runs)
        assert sorted(out) == ["a", "b"]
        assert out["a"]["n"] == 3 and out["a"]["mean"] == pytest.approx(3.0)
        # One run did not log b: n=2, not 3 and not an error.
        assert out["b"]["n"] == 2 and out["b"]["mean"] == pytest.approx(5.0)

    def test_explicit_keys_subset(self):
        runs = [{"a": 1.0, "b": 2.0}, {"a": 3.0}]
        out = SS.aggregate_runs(runs, keys=["b"])
        assert list(out) == ["b"]
        assert out["b"]["n"] == 1


class TestPairedDelta:
    def test_delta_is_treatment_minus_baseline(self):
        out = SS.paired_delta([0.5, 0.6, 0.7], [0.4, 0.4, 0.6])
        assert out["n"] == 3
        assert out["mean"] == pytest.approx((0.1 + 0.2 + 0.1) / 3)
        assert out["wins"] == 3 and out["losses"] == 0 and out["ties"] == 0

    def test_swapping_the_arms_flips_the_sign(self):
        out = SS.paired_delta([0.4, 0.4, 0.6], [0.5, 0.6, 0.7])
        assert out["mean"] < 0
        assert out["wins"] == 0 and out["losses"] == 3

    def test_unequal_lengths_raise_instead_of_truncating(self):
        with pytest.raises(ValueError, match="one baseline value per treatment seed"):
            SS.paired_delta([1.0, 2.0], [1.0])

    def test_non_finite_pairs_are_skipped(self):
        out = SS.paired_delta([1.0, float("nan"), 2.0], [0.0, 0.0, 1.0])
        assert out["n"] == 2
        assert out["wins"] == 2

    def test_ties_are_counted(self):
        out = SS.paired_delta([1.0, 2.0], [1.0, 1.0])
        assert out["ties"] == 1 and out["wins"] == 1


class TestFormatSummary:
    def test_two_or_more_seeds_show_interval_and_n(self):
        text = SS.format_summary(SS.summarize([1.0, 2.0, 3.0]))
        assert "±" in text and "n=3" in text and "95% CI" in text

    def test_single_seed_says_there_is_no_interval(self):
        text = SS.format_summary(SS.summarize([1.0]))
        assert "no interval" in text and "n=1" in text

    def test_empty_is_explicit(self):
        assert SS.format_summary(SS.summarize([])) == "n/a (n=0)"
