# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Tests for src/interp/statistical_tests.py — Benjamini-Hochberg FDR.
#
# The load-bearing test in this file is
#   TestBenjaminiHochbergAgreement::test_significant_set_is_exactly_corrected_at_or_below_alpha
# It asserts `significant[i] == (corrected[i] <= alpha)` for every i. That
# identity is the definition of the BH adjusted p-value (Benjamini & Hochberg
# 1995, eq. 2.4), so it is the one assertion that cannot be satisfied by a
# formula which happens to look reasonable.

import math

import pytest

# ═══════════════════════════════════════════════════════════════════
# Independent reference implementations (textbook transcriptions)
# ═══════════════════════════════════════════════════════════════════


def _reference_adjusted(p_values, alpha=0.05):
    """BH adjusted p-values, transcribed from Benjamini & Hochberg (1995).

    q_(i) = min over j >= i of  min(p_(j) * n / j, 1)

    i.e. the running minimum taken from the TOP rank downwards. Written from the
    definition, independently of the implementation under test.
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    scaled = [min(p_values[i] * n / (rank + 1), 1.0) for rank, i in enumerate(order)]
    adj = list(scaled)
    for j in range(n - 2, -1, -1):
        adj[j] = min(adj[j], adj[j + 1])
    corrected = [0.0] * n
    for rank, i in enumerate(order):
        corrected[i] = adj[rank]
    return corrected


def _reference_step_up(p_values, alpha=0.05):
    """BH step-up: reject every rank up to the largest k with p_(k) <= k*alpha/n."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    k = 0
    for rank, i in enumerate(order, 1):
        if min(p_values[i] * n / rank, 1.0) <= alpha:
            k = rank
    return sorted(order[:k])


def _naive_p_times_n_over_rank(p_values):
    """The defective formula the module used: corrected = p * n / rank, no
    running minimum. Kept here so the tests can show the difference explicitly."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    for rank, i in enumerate(order, 1):
        corrected[i] = min(p_values[i] * n / rank, 1.0)
    return corrected


def _bh(p_values, alpha=0.05):
    from src.interp.statistical_tests import MultipleComparisonCorrection

    return MultipleComparisonCorrection.benjamini_hochberg(list(p_values), alpha)


def _random_p_values(n, rng, low=1e-4, high=1.0):
    return [float(rng.uniform(low, high)) for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════
# The agreement property — the load-bearing assertion
# ═══════════════════════════════════════════════════════════════════


class TestBenjaminiHochbergAgreement:
    def test_significant_set_is_exactly_corrected_at_or_below_alpha(self):
        """`significant` and `corrected` must not disagree, for any p-values.

        This is the defect: the two were computed by different rules, so a
        report could carry `p_value_bh = 0.06` beside `significant_bh = True`.
        """
        rng = pytest.importorskip("numpy").random.RandomState(20260929)
        for n in (2, 3, 5, 8, 13, 21, 40):
            for alpha in (0.01, 0.05, 0.10):
                for _ in range(25):
                    p = _random_p_values(n, rng)
                    r = _bh(p, alpha)
                    significant = set(r["significant"])
                    for i, q in enumerate(r["corrected"]):
                        assert (i in significant) == (
                            q <= alpha
                        ), f"n={n} alpha={alpha} p={p}: index {i} corrected={q} alpha={alpha} sig={sorted(significant)}"

    def test_agreement_holds_at_exactly_the_card_repro(self):
        """p = [0.03, 0.049] is the card's repro. Guard it verbatim."""
        r = _bh([0.03, 0.049], 0.05)
        # q_(1) = min(0.03*2/1, 0.049*2/2) = min(0.06, 0.049) = 0.049
        assert r["corrected"][0] == pytest.approx(0.049)
        assert r["corrected"][1] == pytest.approx(0.049)
        assert r["significant"] == [0, 1]
        # and the naive formula would have said "not significant" for index 0
        assert _naive_p_times_n_over_rank([0.03, 0.049])[0] > 0.05

    def test_adjusted_matches_the_reference_running_minimum(self):
        rng = pytest.importorskip("numpy").random.RandomState(7)
        for n in (1, 2, 4, 9, 17, 33):
            for _ in range(20):
                p = _random_p_values(n, rng)
                r = _bh(p, 0.05)
                assert r["corrected"] == pytest.approx(_reference_adjusted(p))

    def test_significant_matches_the_reference_step_up(self):
        rng = pytest.importorskip("numpy").random.RandomState(11)
        for n in (1, 2, 4, 9, 17, 33):
            for alpha in (0.01, 0.05, 0.2):
                for _ in range(20):
                    p = _random_p_values(n, rng)
                    assert sorted(_bh(p, alpha)["significant"]) == _reference_step_up(p, alpha)

    def test_ties_receive_identical_adjusted_values(self):
        """A tie must not be split by arbitrary sort order."""
        r = _bh([0.02, 0.02, 0.02], 0.05)
        assert r["corrected"][0] == pytest.approx(r["corrected"][1])
        assert r["corrected"][1] == pytest.approx(r["corrected"][2])
        # the naive formula splits them: [0.06, 0.03, 0.02]
        assert len(set(r["corrected"])) == 1

    def test_a_strong_signal_beats_a_run_of_nulls(self):
        """The regime the running minimum exists for: one real effect among nulls."""
        p = [0.001, 0.6, 0.7, 0.8, 0.9]
        r = _bh(p, 0.05)
        assert r["significant"] == [0]
        assert r["corrected"][0] == pytest.approx(0.005)
        # every null must stay non-significant, and not be dragged below alpha
        for i in range(1, 5):
            assert r["corrected"][i] > 0.05
        assert r["n_significant"] == 1
        assert r["n_total"] == 5


# ═══════════════════════════════════════════════════════════════════
# Structural properties of the BH adjusted p-value
# ═══════════════════════════════════════════════════════════════════


class TestBenjaminiHochbergProperties:
    def test_adjusted_is_monotone_in_the_sorted_order(self):
        """q_(1) <= q_(2) <= ... <= q_(n). The naive formula violates this."""
        rng = pytest.importorskip("numpy").random.RandomState(3)
        for n in (2, 3, 5, 10, 25):
            for _ in range(20):
                p = _random_p_values(n, rng)
                r = _bh(p)
                order = sorted(range(n), key=lambda i: p[i])
                seq = [r["corrected"][i] for i in order]
                assert all(
                    seq[j] <= seq[j + 1] + 1e-12 for j in range(n - 1)
                ), f"n={n} p={p} -> {seq}"

    def test_adjusted_never_goes_below_the_raw_p_value(self):
        rng = pytest.importorskip("numpy").random.RandomState(5)
        for n in (2, 4, 11, 30):
            for _ in range(20):
                p = _random_p_values(n, rng)
                r = _bh(p)
                for i in range(n):
                    assert r["corrected"][i] >= p[i] - 1e-12

    def test_adjusted_is_a_probability(self):
        rng = pytest.importorskip("numpy").random.RandomState(9)
        for n in (1, 2, 7, 19):
            for _ in range(20):
                r = _bh(_random_p_values(n, rng))
                assert all(0.0 <= q <= 1.0 for q in r["corrected"])
                assert all(math.isfinite(q) for q in r["corrected"])

    def test_bh_is_never_more_conservative_than_bonferroni(self):
        from src.interp.statistical_tests import MultipleComparisonCorrection

        rng = pytest.importorskip("numpy").random.RandomState(13)
        for n in (2, 3, 8, 20):
            for _ in range(20):
                p = _random_p_values(n, rng)
                bh = _bh(p)["corrected"]
                bonf = MultipleComparisonCorrection.bonferroni(p)
                for i in range(n):
                    assert bh[i] <= bonf[i] + 1e-12

    def test_single_test_is_unchanged(self):
        """With one hypothesis BH is the identity (n/rank == 1/1)."""
        below = _bh([0.031])
        assert below["corrected"][0] == pytest.approx(0.031)
        assert below["significant"] == [0]

        above = _bh([0.06])
        assert above["corrected"][0] == pytest.approx(0.06)
        assert above["significant"] == []
        assert above["n_significant"] == 0 and above["n_total"] == 1

    def test_discovery_set_is_a_prefix_of_the_sorted_order(self):
        """Step-up structure: if rank k is a discovery, so is every smaller rank."""
        rng = pytest.importorskip("numpy").random.RandomState(17)
        for n in (2, 4, 11, 23):
            for alpha in (0.02, 0.05, 0.15):
                for _ in range(20):
                    p = _random_p_values(n, rng)
                    order = sorted(range(n), key=lambda i: p[i])
                    significant = sorted(_bh(p, alpha)["significant"])
                    assert significant == order[: len(significant)]

    def test_results_are_returned_in_the_input_order(self):
        """Reindexing the input must reindex the output by the same permutation."""
        rng = pytest.importorskip("numpy").random.RandomState(19)
        for _ in range(20):
            n = 6
            p = _random_p_values(n, rng)
            perm = rng.permutation(n).tolist()
            a = _bh(p, 0.05)
            b = _bh([p[i] for i in perm], 0.05)
            sig_a, sig_b = set(a["significant"]), set(b["significant"])
            for pos, src in enumerate(perm):
                assert a["corrected"][src] == pytest.approx(b["corrected"][pos])
                assert (src in sig_a) == (pos in sig_b)

    def test_p_values_may_arrive_as_zero(self):
        """An exact permutation test can emit p == 0; it must not become negative."""
        r = _bh([0.0, 0.5, 0.9], 0.05)
        assert r["corrected"][0] == pytest.approx(0.0)
        assert r["significant"] == [0]
        assert all(q >= 0.0 for q in r["corrected"])

    def test_threshold_is_reported_unchanged(self):
        r = _bh([0.01, 0.9], 0.01)
        assert r["threshold"] == 0.01
        assert _bh([0.01, 0.9], 0.05)["threshold"] == 0.05

    def test_empty_input(self):
        r = _bh([], 0.05)
        assert r["corrected"] == []
        assert r["significant"] == []


# ═══════════════════════════════════════════════════════════════════
# End-to-end: real permutation p-values, then BH
# ═══════════════════════════════════════════════════════════════════


class TestBenjaminiHochbergOnRealPValues:
    def test_permutation_p_values_under_a_null_are_handled_consistently(self):
        """p-values produced by the module's own permutation test, under H0."""
        pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")

        from src.interp.statistical_tests import PairedPermutationTest

        gen = torch.Generator().manual_seed(4242)
        p_values = []
        for _ in range(10):
            a = torch.randn(24, generator=gen)
            b = a + 0.05 * torch.randn(24, generator=gen)  # same distribution
            p_values.append(PairedPermutationTest.compute(a, b, n_permutations=199)["p_value"])
        r = _bh(p_values, 0.05)
        significant = set(r["significant"])
        for i, q in enumerate(r["corrected"]):
            assert (i in significant) == (q <= 0.05)
            assert q >= p_values[i] - 1e-12
        # every p-value here is a real permutation p-value: in [0, 1]
        assert all(0.0 <= p <= 1.0 for p in p_values)
        assert len(r["corrected"]) == 10

    def test_a_report_never_shows_p_value_above_alpha_next_to_significant(self):
        """`MetricComparisonReport` is the surface where the two were published
        side by side. Assert the invariant on the report it emits (real data)."""
        pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")

        from src.interp.statistical_tests import MetricComparisonReport

        gen = torch.Generator().manual_seed(909)
        names = ["m0", "m1", "m2", "m3", "m4", "m5"]
        jepa = {n: torch.randn(20, generator=gen) for n in names}
        base = {n: torch.randn(20, generator=gen) for n in names}
        # give two metrics a genuine effect so the correction has something to do
        jepa["m0"] = jepa["m0"] + 3.0
        jepa["m3"] = jepa["m3"] + 1.5

        report = MetricComparisonReport.generate(
            names, jepa, base, n_bootstrap=40, n_permutations=199, alpha=0.05
        )
        assert set(report) >= set(names)
        for name in names:
            row = report[name]
            assert row["significant_bh"] == (row["p_value_bh"] <= 0.05), (
                f"{name}: p_value_bh={row['p_value_bh']} " f"significant_bh={row['significant_bh']}"
            )
            assert 0.0 <= row["p_value_bh"] <= 1.0
            assert row["p_value_bh"] >= row["p_value"] - 1e-12

        summary = report["_summary"]
        assert summary["n_significant_bh"] == sum(1 for n in names if report[n]["significant_bh"])

    def test_the_card_p_values_survive_the_report_unchanged(self, monkeypatch):
        """The card's exact p-vector, driven through the public report.

        Random data cannot reach this regime: `PairedPermutationTest` floors its
        p-value at 1/(n_permutations+1), and every metric with a real effect
        hits that floor together, so the two smallest p-values are never inside
        the window where `p_(1)*n/1 > alpha >= p_(2)*n/2` (measured over 9
        seeds x 6 metrics, all identical). Only the permutation test is stubbed
        here; the correction and the whole `MetricComparisonReport` body run
        unstubbed, so this is the card's scenario end to end.
        """
        pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")

        import src.interp.statistical_tests as st

        names = ["a", "b"]
        card_p = [0.03, 0.049]  # the card's repro, in report order
        real_compute = st.PairedPermutationTest.compute
        cursor = iter(card_p)

        def stub(values_a, values_b, n_permutations=10000, seed=42):
            out = real_compute(values_a, values_b, n_permutations=n_permutations, seed=seed)
            return dict(out, p_value=next(cursor))

        monkeypatch.setattr(st.PairedPermutationTest, "compute", staticmethod(stub))

        jepa = {"a": torch.randn(16) + 1.0, "b": torch.randn(16) + 0.2}
        base = {"a": torch.randn(16), "b": torch.randn(16)}

        report = st.MetricComparisonReport.generate(
            names, jepa, base, n_bootstrap=40, n_permutations=199, alpha=0.05
        )

        for name, expected_p in zip(names, card_p):
            row = report[name]
            assert row["p_value"] == pytest.approx(expected_p)
            # the card's headline: p_value_bh must not read 0.06 beside True
            assert row["p_value_bh"] == pytest.approx(0.049)
            assert row["significant_bh"] is True
            assert row["significant_bh"] == (row["p_value_bh"] <= 0.05)
        assert report["_summary"]["n_significant_bh"] == 2
        # the naive formula would have published 0.06 for metric "a"
        assert _naive_p_times_n_over_rank(card_p)[0] > 0.05
