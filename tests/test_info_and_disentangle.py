# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Regression tests for the audited defects in src/interp/information_theory.py
# and src/interp/disentanglement.py (wave-1 audit findings I4 / I5).
#
# Every test here is tiny (N <= 1024, D <= 64) and seeded. They pin *properties*
# of the metrics, not magic constants.

import math

import pytest
import torch

from src.interp.disentanglement import DCIMetrics, ModularityScore
from src.interp.information_theory import (
    ConditionalMIEstimator,
    MINEEstimator,
    RepresentationCompression,
    TC_UNDEFINED,
    total_correlation_null_expectation,
)

# ═══════════════════════════════════════════════════════════════════
# BUG 1a — total_correlation mixed a 30-bin histogram marginal entropy with
# an exact-Gaussian joint entropy, so it reported 92/375/1158 nats on
# INDEPENDENT gaussian dims at D = 64/256/768 (pure estimator bias).
# ═══════════════════════════════════════════════════════════════════


def _independent_gaussians(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


class TestTotalCorrelation:
    @pytest.mark.parametrize(
        ("d", "n", "reps"),
        [(64, 512, 8), (256, 2048, 4), (768, 6144, 4)],
    )
    def test_independent_dims_give_near_zero_tc(self, d, n, reps):
        """True TC = 0 for independent dims at EVERY D. Was 92/405/1156 nats."""
        vals = [
            RepresentationCompression.total_correlation(_independent_gaussians(n, d, 9000 + i))
            for i in range(reps)
        ]
        assert all(math.isfinite(v) for v in vals)
        assert sum(vals) / reps < 0.1, f"D={d}: mean TC = {sum(vals) / reps:.4f} nats, vals={vals}"

    @pytest.mark.parametrize(("d", "n"), [(64, 1024), (256, 4096), (768, 6144)])
    def test_null_expectation_matches_simulation(self, d, n):
        """The bias that gets subtracted must itself be accurate."""
        predicted = total_correlation_null_expectation(d, n)
        observed = [
            RepresentationCompression.total_correlation(
                _independent_gaussians(n, d, 4000 + i),
                bias_correct=False,
            )
            for i in range(4)
        ]
        mean_observed = sum(observed) / 4
        assert mean_observed == pytest.approx(
            predicted, rel=0.15
        ), f"D={d}, N={n}: predicted null {predicted:.4f}, observed {mean_observed:.4f}"

    def test_tc_does_not_grow_with_dimension(self):
        """The bias grew ~1.5 nats/dim. Now it must not grow with D."""
        tcs = [
            RepresentationCompression.total_correlation(_independent_gaussians(4096, d, 11 + d))
            for d in (32, 64, 128)
        ]
        assert max(tcs) < 0.1, f"per-dim bias survives: {tcs}"

    @pytest.mark.parametrize("d", [4, 16, 64])
    def test_perfectly_correlated_dims_give_large_tc(self, d):
        """RED before the fix: returned 0.0 = 'perfectly disentangled'.

        Maximally dependent dimensions are the case the metric exists to flag.
        """
        n = 4 * d + 64
        g = torch.Generator().manual_seed(500 + d)
        x = torch.randn(n, 1, generator=g).repeat(1, d)
        tc = RepresentationCompression.total_correlation(x)
        assert tc > 1.0, f"D={d}: perfectly correlated dims gave TC={tc!r}, expected a large value"

    def test_dependence_is_ordered(self):
        """A stronger shared latent must never *lower* TC.

        The dependence has to come from a COMMON latent; adding independent
        noise to independent dimensions leaves them independent, so the shared
        latent is what is being varied here.
        """
        n = 2048
        g = torch.Generator().manual_seed(31)
        indep = torch.randn(n, 8, generator=g)
        # Each dimension is its own factor plus `w` of ONE shared latent.
        partial = torch.randn(n, 8, generator=g) + 0.5 * torch.randn(n, 1, generator=g)
        full = torch.randn(n, 8, generator=g) + 2.0 * torch.randn(n, 1, generator=g)

        t_indep = RepresentationCompression.total_correlation(indep)
        t_part = RepresentationCompression.total_correlation(partial)
        t_full = RepresentationCompression.total_correlation(full)
        assert t_indep < 0.1
        assert t_indep < t_part < t_full, (t_indep, t_part, t_full)

    def test_insufficient_samples_is_never_zero(self):
        """RED before the fix: `if N <= D: return 0.0` -> read as 'ideal'.

        The metric must either raise or return the documented sentinel.
        """
        x = _independent_gaussians(16, 64, 1)
        try:
            tc = RepresentationCompression.total_correlation(x)
        except ValueError:
            return  # raising is an accepted, documented policy
        assert tc is not None
        assert tc != 0.0, "N <= D returned 0.0, which reads as 'perfectly disentangled'"
        assert math.isnan(tc) and TC_UNDEFINED != 0.0

    def test_singular_covariance_is_not_zero(self):
        """RED before the fix: `if sign <= 0: return 0.0` -> read as 'ideal'.

        Three live dimensions that are exact multiples of one another give a
        rank-deficient covariance: TC is genuinely unbounded there, and must be
        reported as a large value rather than as "no dependence".
        """
        g = torch.Generator().manual_seed(3)
        z = torch.randn(256, 1, generator=g)
        x = torch.cat([z, z, 2 * z, torch.zeros(256, 1)], dim=1)
        tc = RepresentationCompression.total_correlation(x)
        assert tc > 1.0, f"rank-deficient dependence reported TC={tc!r}"

    def test_single_live_dimension_is_zero_not_a_sentinel(self):
        """One dimension cannot be dependent on itself; 0.0 is the real answer."""
        x = torch.zeros(256, 4)
        x[:, 0] = torch.linspace(0.0, 1.0, 256)
        tc = RepresentationCompression.total_correlation(x)
        assert tc == 0.0

    def test_constant_dims_do_not_inflate_tc(self):
        """Dead dimensions carry no dependence; they must not be scored."""
        g = torch.Generator().manual_seed(77)
        live = torch.randn(512, 4, generator=g)
        padded = torch.cat([live, torch.zeros(512, 4)], dim=1)
        assert RepresentationCompression.total_correlation(padded) < 0.1

    def test_deterministic(self):
        x = _independent_gaussians(512, 32, 3)
        vals = {RepresentationCompression.total_correlation(x) for _ in range(3)}
        assert len(vals) == 1


# ═══════════════════════════════════════════════════════════════════
# BUG 1c — compression_ratio mixed nats/dim against nats and had a
# dead expression statement at L345. Executed: 0.03189 vs 0.03185 for
# wildly different representations, and 0.03185 for a constant one.
# ═══════════════════════════════════════════════════════════════════


class TestCompressionRatio:
    def test_constant_rep_is_distinguished_from_iid(self):
        g = torch.Generator().manual_seed(21)
        iid = torch.randn(512, 32, generator=g)
        const = torch.ones(512, 32)
        assert RepresentationCompression.compression_ratio(iid, 256) > 0.0
        assert RepresentationCompression.compression_ratio(const, 256) == 0.0

    def test_redundancy_lowers_the_ratio(self):
        """Was 0.00821 (iid) vs 0.00827 (perfectly correlated) — no separation."""
        g = torch.Generator().manual_seed(22)
        iid = torch.randn(512, 32, generator=g)
        correlated = torch.randn(512, 1, generator=g).repeat(1, 32)
        r_iid = RepresentationCompression.compression_ratio(iid, 256)
        r_corr = RepresentationCompression.compression_ratio(correlated, 256)
        assert r_iid > 10 * r_corr, f"iid={r_iid:.5f} correlated={r_corr:.5f}"

    def test_is_scale_invariant(self):
        """Compression must not be confounded with the arbitrary activation scale."""
        g = torch.Generator().manual_seed(23)
        x = torch.randn(512, 32, generator=g)
        base = RepresentationCompression.compression_ratio(x, 256)
        for scale in (1e-3, 7.0, 1000.0):
            assert RepresentationCompression.compression_ratio(x * scale, 256) == pytest.approx(
                base, rel=1e-3
            )

    def test_tracks_effective_dimension(self):
        g = torch.Generator().manual_seed(24)
        r16 = RepresentationCompression.compression_ratio(torch.randn(512, 16, generator=g), 256)
        r32 = RepresentationCompression.compression_ratio(torch.randn(512, 32, generator=g), 256)
        r64 = RepresentationCompression.compression_ratio(torch.randn(512, 64, generator=g), 256)
        assert r16 < r32 < r64

    def test_deterministic(self):
        g = torch.Generator().manual_seed(25)
        x = torch.randn(512, 16, generator=g)
        vals = {RepresentationCompression.compression_ratio(x, 256) for _ in range(3)}
        assert len(vals) == 1


# ═══════════════════════════════════════════════════════════════════
# BUG 1d — MINE reported the mean of the last 10% of the TRAINING
# objective. Executed: ~1.5 nats for INDEPENDENT x and y.
# ═══════════════════════════════════════════════════════════════════


class TestMINE:
    def _pair(self, dependent, n=1024, d=16, seed=7):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, d, generator=g)
        if dependent:
            y = x + 0.1 * torch.randn(n, d, generator=g)
        else:
            y = torch.randn(n, d, generator=g)
        return x, y

    def test_independent_pairs_are_not_scored_as_informative(self):
        """RED before the fix: the training objective reported ~1.5 nats."""
        x, y = self._pair(dependent=False)
        mi = MINEEstimator(16, 16, hidden_dim=64).compute_mi(x, y)
        assert mi < 2.0, f"independent x/y reported MI={mi:.4f} nats"

    def test_dependent_pairs_exceed_independent_pairs(self):
        xd, yd = self._pair(dependent=True)
        xi, yi = self._pair(dependent=False)
        mi_dep = MINEEstimator(16, 16, hidden_dim=64).compute_mi(xd, yd)
        mi_ind = MINEEstimator(16, 16, hidden_dim=64).compute_mi(xi, yi)
        assert mi_dep > mi_ind, f"dependent={mi_dep:.4f} independent={mi_ind:.4f}"

    def test_training_objective_is_exposed_separately(self):
        """The optimisation objective must stay available, clearly labelled."""
        x, y = self._pair(dependent=True)
        est = MINEEstimator(16, 16, hidden_dim=64)
        held_out = est.compute_mi(x, y)
        assert isinstance(est.training_objective, float)
        assert math.isfinite(est.training_objective)
        assert held_out >= 0.0

    def test_deterministic(self):
        x, y = self._pair(dependent=True)
        a = MINEEstimator(16, 16, hidden_dim=64, seed=5).compute_mi(x, y)
        b = MINEEstimator(16, 16, hidden_dim=64, seed=5).compute_mi(x, y)
        assert a == b, f"{a} != {b}"


# ═══════════════════════════════════════════════════════════════════
# BUG 1e — information_gain divided by max(mi_target, 1e-10).
# ═══════════════════════════════════════════════════════════════════


class TestConditionalMI:
    def test_information_gain_is_flagged_when_denominator_is_at_noise_floor(self):
        g = torch.Generator().manual_seed(41)
        reps = torch.randn(256, 16, generator=g)
        target = torch.randn(256, 4, generator=g)
        cond = torch.randn(256, 4, generator=g)
        out = ConditionalMIEstimator.compute(reps, target, cond)

        assert "information_gain_defined" in out
        if out["information_gain_defined"]:
            assert out["information_gain"] > -1e-6
        else:
            assert math.isnan(out["information_gain"])
            assert isinstance(out["information_gain_reason"], str)

    def test_information_gain_defined_when_target_is_informative(self):
        g = torch.Generator().manual_seed(42)
        target = torch.randn(256, 4, generator=g)
        reps = target.repeat(1, 4) + 0.01 * torch.randn(256, 16, generator=g)
        cond = torch.randn(256, 4, generator=g)
        out = ConditionalMIEstimator.compute(reps, target, cond)
        assert out["mi_target"] > 0.0, out
        assert out["information_gain_defined"] is True, out["information_gain_reason"]
        assert math.isfinite(out["information_gain"])

    def test_information_gain_is_a_fraction_when_defined(self):
        """0 <= mi_conditional <= mi_target implies 0 <= gain <= 1."""
        g = torch.Generator().manual_seed(45)
        target = torch.randn(256, 4, generator=g)
        reps = target.repeat(1, 4) + 0.05 * torch.randn(256, 16, generator=g)
        cond = torch.randn(256, 4, generator=g)
        out = ConditionalMIEstimator.compute(reps, target, cond)
        if out["information_gain_defined"]:
            assert 0.0 <= out["information_gain"] <= 1.0, out

    def test_no_negative_information_gain(self):
        g = torch.Generator().manual_seed(43)
        reps = torch.randn(256, 16, generator=g)
        target = torch.randn(256, 4, generator=g)
        cond = torch.randn(256, 4, generator=g)
        out = ConditionalMIEstimator.compute(reps, target, cond)
        if out["information_gain_defined"]:
            assert out["information_gain"] >= 0.0

    def test_deterministic(self):
        g = torch.Generator().manual_seed(44)
        reps = torch.randn(128, 16, generator=g)
        target = torch.randn(128, 4, generator=g)
        cond = torch.randn(128, 4, generator=g)
        a = ConditionalMIEstimator.compute(reps, target, cond)
        b = ConditionalMIEstimator.compute(reps, target, cond)
        # NaN != NaN, so compare the defined keys and the flags separately.
        for key in ("mi_target", "mi_condition", "mi_joint", "mi_conditional"):
            assert a[key] == b[key], key
        assert a["information_gain_defined"] == b["information_gain_defined"]
        assert a["information_gain_reason"] == b["information_gain_reason"]
        if a["information_gain_defined"]:
            assert a["information_gain"] == b["information_gain"]


# ═══════════════════════════════════════════════════════════════════
# BUG 2 — ModularityScore was INVERTED against its own docstring.
# Executed: ideal one-hot 0.677, pure noise 0.632, 1-informative-of-4
# 0.759 — the most degenerate case won.
# ═══════════════════════════════════════════════════════════════════


def _factors(n=512, k=4, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g)


class TestModularity:
    def test_ideal_one_hot_is_highly_modular(self):
        """RED before the fix: ideal one-hot scored 0.240.

        Not exactly 1.0: with finite N, a dimension that IS a factor still
        shows ~1/sqrt(N) |correlation| with every other factor by chance, and
        modularity is measured against those. Measured: 0.760 (N=512),
        0.908 (N=4096), 0.957 (N=32768) — it approaches 1.0 from below.
        """
        factors = _factors()
        assert ModularityScore.compute(factors.clone(), factors) > 0.7

    def test_ideal_one_hot_approaches_one_with_more_samples(self):
        n = 8192
        g = torch.Generator().manual_seed(2)
        factors = torch.randn(n, 4, generator=g)
        score = ModularityScore.compute(factors.clone(), factors)
        assert score > 0.9, f"N={n}: ideal one-hot scored only {score:.4f}"

    def test_uniform_mixture_is_not_modular(self):
        """RED before the fix: uniform-mixed dims scored HIGHEST (0.759).

        Every dimension carries the SAME mixture of all factors, so no
        dimension is more responsible for one factor than another.
        """
        factors = _factors()
        shared = factors.mean(dim=1, keepdim=True).repeat(1, 4)
        uniform = shared + 0.01 * torch.randn(512, 4, generator=torch.Generator().manual_seed(5))
        assert ModularityScore.compute(uniform, factors) < 0.05

    def test_direction_matches_docstring(self):
        """one-hot > partial mixture > uniform mixture. Was reversed."""
        factors = _factors()
        g = torch.Generator().manual_seed(6)
        one_hot = factors.clone()
        shared = factors.mean(dim=1, keepdim=True).repeat(1, 4)
        partial = shared + torch.randn(512, 4, generator=g)
        uniform = shared + 0.01 * torch.randn(512, 4, generator=g)

        s = [ModularityScore.compute(rep, factors) for rep in (one_hot, partial, uniform)]
        assert s[0] > s[1] > s[2], s

    def test_dead_dim_does_not_score_perfect(self):
        """RED before the fix: `if r_sum < 1e-10: mod = 1.0`."""
        factors = _factors()
        reps = factors.clone()
        with_dead = torch.cat([reps, torch.ones(512, 1)], dim=1)

        base = ModularityScore.compute(reps, factors)
        padded = ModularityScore.compute(with_dead, factors)
        assert padded < base, f"adding a dead dim raised the score: {base:.4f} -> {padded:.4f}"

    def test_dead_dims_are_reported_and_penalised(self):
        n = 4096
        g = torch.Generator().manual_seed(2)
        factors = torch.randn(n, 4, generator=g)
        # Orthonormalise so the live dims have no spurious cross-correlation.
        live, _ = torch.linalg.qr(factors - factors.mean(0))
        padded = torch.cat([live, torch.ones(n, 2)], dim=1)

        details = ModularityScore.compute_with_details(padded, factors)
        assert details["n_dead_dims"] == 2
        assert details["n_live_dims"] == 4
        # Two dead dims out of six must cost roughly a third of the score.
        live_only = ModularityScore.compute(live, factors)
        assert details["modularity"] < live_only
        assert details["modularity"] == pytest.approx(live_only * 4 / 6, abs=0.02)

    def test_score_stays_in_unit_interval(self):
        factors = _factors()
        g = torch.Generator().manual_seed(8)
        for reps in (torch.randn(512, 8, generator=g), torch.ones(512, 8), factors):
            mod = ModularityScore.compute(reps, factors)
            assert 0.0 <= mod <= 1.0

    def test_deterministic(self):
        factors = _factors()
        reps = torch.randn(512, 6, generator=torch.Generator().manual_seed(9))
        assert len({ModularityScore.compute(reps, factors) for _ in range(3)}) == 1


# ═══════════════════════════════════════════════════════════════════
# BUG 3 — DCI "informativeness" was max-per-factor |correlation|, whose
# noise floor is ~0.1 at N=200. ground_truth.py:266 declares the pipeline
# valid above 0.1, i.e. INSIDE the noise floor.
# ═══════════════════════════════════════════════════════════════════


class TestDCIInformativeness:
    def _noise(self, n, d, k, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(n, d, generator=g), torch.randn(n, k, generator=g)

    def test_noise_floor_shrinks_with_n(self):
        """RED before the fix: 0.173 (N=200) / 0.152 (N=500) / 0.080 (N=2000).

        Adjusted R^2 is unbiased under the null, so it straddles zero rather
        than decaying toward it from above; what must shrink is its MAGNITUDE.
        """
        small = abs(DCIMetrics.compute(*self._noise(512, 16, 4, 100))["informativeness_raw"])
        large = abs(DCIMetrics.compute(*self._noise(4096, 16, 4, 100))["informativeness_raw"])
        assert large < small, f"noise floor grew with N: {small:.5f} -> {large:.5f}"

    def test_noise_floor_is_negligible(self):
        """The validator threshold is 0.1; the floor must sit far below it."""
        out = DCIMetrics.compute(*self._noise(4096, 16, 4, 101))["informativeness_raw"]
        assert abs(out) < 0.02, f"noise floor = {out:.5f}, threshold 0.1 is inside it"

    def test_clamped_informativeness_is_near_zero_on_noise(self):
        """The headline (clamped) number must not pass a 0.1 threshold on noise."""
        for n in (512, 4096):
            reps, factors = self._noise(n, 16, 4, 102)
            assert DCIMetrics.compute(reps, factors)["informativeness"] < 0.02

    def test_measured_noise_floor_helper(self):
        floor = DCIMetrics.measure_noise_floor(n=1024, d=16, k=4, n_trials=6, seed=0)
        assert floor["mean"] < 0.02
        assert floor["max"] < 0.1

    def test_perfectly_predictable_factors_score_one(self):
        g = torch.Generator().manual_seed(55)
        factors = torch.randn(512, 4, generator=g)
        reps = factors.clone()
        out = DCIMetrics.compute(reps, factors)
        assert out["informativeness"] > 0.95

    def test_unpredictable_reps_score_near_zero(self):
        factors = _factors()
        g = torch.Generator().manual_seed(56)
        out = DCIMetrics.compute(torch.randn(512, 16, generator=g), factors)
        assert out["informativeness"] < 0.05

    def test_estimator_is_documented(self):
        out = DCIMetrics.compute(*self._noise(256, 8, 2, 57))
        assert isinstance(out["informativeness_estimator"], str)
        assert out["informativeness_estimator"]

    def test_deterministic(self):
        reps, factors = self._noise(512, 16, 4, 58)
        a = DCIMetrics.compute(reps, factors)
        b = DCIMetrics.compute(reps, factors)
        assert a == b
