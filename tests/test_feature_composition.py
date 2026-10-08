# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# FeatureInterferenceScore: the control set must be disjoint from the
# treatment set, and the feature subsample must be seeded.
#
# Regression tests for the two defects measured in
# docs/plans/2026-09-27-wave1-audit-findings-interp.md (I15):
#   * `z_baseline = z` was the FULL dataset, so at N == n_top every sample was
#     both treated and control and `mean_interference` collapsed to 1.4e-10
#     (float round-off of x - x). At any other N the score was attenuated by
#     n_top/N, so it was not comparable across datasets.
#   * `feature_idx = torch.randperm(M)[:n_test]` drew from the process-global
#     RNG, so the reported number was not reproducible.
#
# These tests assert *properties* (disjointness, comparability, determinism),
# not the numeric values of the score.

import math

import torch

from src.interp.feature_composition import FeatureInterferenceScore
from src.interp.sae import SparseAutoencoder

IN_DIM = 24
LATENT = 128
K = 12
N_BLOCKS = 3


def _block_reps(n, seed=0):
    """Representations with a strong, stable block structure.

    Without structure the two-sample mean difference is pure noise and any
    tolerance below ~100% is meaningless. Three well-separated blocks give a
    real signal: samples that activate a feature strongly tend to share a
    block, and every other feature shifts with the block.
    """
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(N_BLOCKS, IN_DIM, generator=g) * 3.0
    which = torch.randint(0, N_BLOCKS, (n,), generator=g)
    return centres[which] + torch.randn(n, IN_DIM, generator=g) * 0.3


def _sae(seed=0):
    """A SparseAutoencoder with deterministic, non-degenerate weights."""
    g = torch.Generator().manual_seed(seed)
    sae = SparseAutoencoder(input_dim=IN_DIM, latent_dim=LATENT, k=K)
    for p in sae.parameters():
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return sae


def _code_scale(sae, reps):
    """Typical magnitude of a non-zero SAE code, used as the yardstick."""
    z, _, _ = sae.encode(reps)
    return z.abs().mean().item()


class TestTreatedControlSplit:
    """`treated_control_split` is the single source of the treatment/control split."""

    def test_split_is_disjoint_and_covers_the_dataset(self):
        act = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.2, 0.8])
        treated, control = FeatureInterferenceScore.treated_control_split(act, n_top=2)

        t, c = set(treated.tolist()), set(control.tolist())
        assert not (t & c), "a sample is both treated and control"
        assert t | c == set(range(6)), "the split must cover every sample"
        assert len(t) + len(c) == 6, "no sample may be dropped or double-counted"

    def test_treatment_is_the_top_activating_samples(self):
        act = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.2, 0.8])
        treated, _ = FeatureInterferenceScore.treated_control_split(act, n_top=2)
        assert treated.tolist() == [1, 5], "treatment must be the two largest activations"

    def test_treatment_is_capped_at_half_the_dataset(self):
        """n_top >= N must still leave a control: this is the N == n_top case."""
        act = torch.arange(100, dtype=torch.float32)
        treated, control = FeatureInterferenceScore.treated_control_split(act, n_top=100)

        assert treated.numel() == 50
        assert control.numel() == 50
        assert not (set(treated.tolist()) & set(control.tolist()))
        # The treatment really is the strongest half. (topk returns the
        # indices in descending activation order, so compare as sets.)
        assert set(treated.tolist()) == set(range(50, 100))

    def test_tiny_dataset_degrades_without_crashing(self):
        treated, control = FeatureInterferenceScore.treated_control_split(
            torch.tensor([1.0]), n_top=8
        )
        assert treated.numel() == 0 and control.numel() == 1


class TestControlDisjointFromTreatment:
    def test_compute_reports_a_non_overlapping_split_for_every_feature(self):
        reps = _block_reps(120)
        result = FeatureInterferenceScore.compute(
            _sae(), reps, n_features=20, n_top=40, seed=0, return_indices=True
        )

        assert math.isfinite(result["mean_interference"]), "the except branch returns inf"
        assert result["n_treated"] == 40
        assert result["n_control"] == 80
        assert len(result["treated_idx"]) == result["n_features_tested"]
        assert len(result["treated_idx"]) == len(result["control_idx"])

        for treated, control in zip(result["treated_idx"], result["control_idx"]):
            t, c = set(treated), set(control)
            assert not (
                t & c
            ), f"contamination: {len(t & c)} sample(s) are both treated and control"
            assert t | c == set(range(120)), "every sample must be in exactly one side"

    def test_control_is_not_the_full_dataset(self):
        """The defect was literally `z_baseline = z`."""
        reps = _block_reps(120)
        result = FeatureInterferenceScore.compute(_sae(), reps, n_features=8, n_top=40, seed=0)

        assert 0 < result["n_treated"] < 120
        assert 0 < result["n_control"] < 120, "control must exclude the treated samples"
        assert result["n_treated"] + result["n_control"] == 120

    def test_score_is_computed_from_the_reported_split(self):
        """Ties the reported indices to the arithmetic that produced the score.

        Reporting a clean split is not enough: a future edit could keep
        `return_indices` honest while computing the mean over the full
        dataset. Recomputing both statistics from the returned index sets and
        demanding bit-identical agreement fails if the code used anything
        other than the control set it published.
        """
        sae = _sae()
        reps = _block_reps(120)
        result = FeatureInterferenceScore.compute(
            sae, reps, n_features=10, n_top=40, seed=0, return_indices=True
        )
        z, _, _ = sae.encode(reps)
        assert result["n_features_tested"] == 10
        assert len(result["feature_idx"]) == 10

        per_feature = []
        for fi, treated, control in zip(
            result["feature_idx"], result["treated_idx"], result["control_idx"]
        ):
            t = torch.tensor(treated, dtype=torch.long)
            c = torch.tensor(control, dtype=torch.long)
            others = [j for j in range(z.shape[1]) if j != fi]
            oi = torch.tensor(others)
            diff = (z[t][:, oi].mean(dim=0) - z[c][:, oi].mean(dim=0)).abs().mean()
            per_feature.append(diff.item())

        assert min(per_feature) == result["min_interference"]
        assert max(per_feature) == result["max_interference"]
        assert sum(per_feature) / len(per_feature) == result["mean_interference"]

    def test_indices_are_not_exposed_by_default(self):
        """The index dump is opt-in so the hot path stays cheap."""
        reps = _block_reps(60)
        result = FeatureInterferenceScore.compute(_sae(), reps, n_features=4, n_top=20, seed=0)
        assert not ({"feature_idx", "treated_idx", "control_idx"} & set(result))
        assert {"mean_interference", "n_treated", "n_control"} <= set(result)


class TestNotTriviallyZeroAtNTopEqualsNTreated:
    def test_score_is_not_float_round_off_when_n_top_equals_n(self):
        """N == n_top used to return ~1e-10, i.e. exactly zero to float32.

        The yardstick is the typical code magnitude, so the assertion says
        "the measured shift is a real fraction of the signal" rather than
        pinning any particular value.
        """
        sae = _sae()
        reps = _block_reps(100)
        scale = _code_scale(sae, reps)
        assert scale > 0

        result = FeatureInterferenceScore.compute(sae, reps, n_features=20, n_top=100, seed=0)

        assert math.isfinite(result["mean_interference"])
        assert result["n_treated"] == 50 and result["n_control"] == 50
        # The contaminated implementation lands ~9 orders of magnitude below this.
        assert result["mean_interference"] > 0.01 * scale, (
            f"mean_interference={result['mean_interference']:.3e} is negligible "
            f"against a code scale of {scale:.3e}: the control is still "
            f"contaminated by the treatment"
        )


class TestComparableAcrossDatasetSize:
    """The score must not carry the old ``(1 - n_top/N)`` multiplicative factor.

    All numbers below are measured on this seed scheme over 240 draws (15 of
    which are non-overlapping blocks of 16, used as the block simulation for the
    shipped ``DRAWS``). ``n_features=25``, ``N_small=240``, ``N_large=2400``:

    ============  =====================  ==================  ==============
    ``n_top``     fixed: per-draw min..max  old: per-draw min..max  A(n_top)
    ============  =====================  ==================  ==============
        25         0.611 .. 1.304         0.553 .. 1.180       0.9053
        50         0.590 .. 1.384         0.477 .. 1.119       0.8085
       100         0.623 .. 1.311         0.379 .. 0.798       0.6087
       200         0.501 .. 1.912         0.078 .. 0.366       0.1818
    mean of ratio 0.966 / 1.011 / 0.940 / 0.900         (old: 0.874 / 0.818 /
                                                          0.572 / 0.139)

    ``A(n_top) = ((Ns-n)/Ns) / ((Nl-n)/Nl)`` is the old defect in closed form:
    with the control set equal to the whole dataset the score is *exactly*
    ``(N - n_top)/N`` times the correct one, so ``old_ratio(n) = A(n) *
    fixed_ratio(n)`` holds to 1.3e-07 in float32.

    Three measured facts shape every assertion below.

    1. **A per-draw ratio band at a single ``n_top`` cannot separate the two
       implementations**, and a single draw is not a property of the statistic.
       At ``n_top <= 100`` the ranges overlap heavily, so the previous
       ``0.65 < ratio < 1.25`` band let the *attenuated* implementation through
       in 8 of 60 draws -- it was never a regression test. The per-draw ratio
       at ``n_top=100`` ranges over 0.623..1.311 and *0 of 240* draws fall below
       the 0.609 that Linux CI produced where this box produced 0.719, so that
       band was a knife edge resting on one platform's draw.

    2. **The defect is deterministic**, so it is best seen in the *mean* over
       draws, which is far better behaved: the block simulation gives
       ``mean R(100)`` in 0.883..0.997 for the fixed code against 0.537..0.607
       for the old one, and per-``n_top`` means whose smallest value is 0.824
       against 0.153.

    3. **What separates them is the dependence on ``n_top`` itself.** ``A(n_top)``
       collapses from 0.905 to 0.182 across the sweep -- a 4.98x swing -- while
       the fixed ratio barely moves. ``max_n mean_n / min_n mean_n`` is
       1.056..1.233 across blocks for the fixed code against 5.641..6.940 for
       the old one. The 2.5 cap below sits in that gap: 2.03x above the fixed
       maximum, 2.26x below the old minimum, against a geometric midpoint of
       sqrt(1.233 * 5.641) = 2.64.

    A residual downward bias survives and is *not* an artifact: with a fixed
    absolute ``n_top`` the treatment covers 42% of a 240-row dataset but 4% of a
    2400-row one, so it spans more of the block structure at small N. Removing
    it would mean turning ``n_top`` into a fraction, which contradicts this
    module's documented contract and four other assertions here.
    """

    N_SMALL = 240
    N_LARGE = 2400
    N_FEATURES = 25
    N_TOP_SWEEP = (25, 50, 100, 200)
    # 16 draws. The per-draw ratio has sd 0.146, so the mean over DRAWS draws
    # has a standard error of 0.146/sqrt(DRAWS): 0.037 here, which puts the
    # 0.75 edge of the band below about 5 standard errors from the measured
    # mean of 0.94. Fewer draws would make the band a statement about a
    # handful of samples, which is the failure mode this class is fixing.
    DRAWS = 16

    @classmethod
    def _draw_ratios(cls, draw):
        """Ratio small/large at every n_top, for one decorrelated seed tuple."""
        sae = _sae(draw)
        pool = _block_reps(cls.N_LARGE, seed=1000 + draw)
        order = torch.randperm(cls.N_LARGE, generator=torch.Generator().manual_seed(2000 + draw))
        small = pool[order[: cls.N_SMALL]]
        large = pool[order[: cls.N_LARGE]]

        out = {}
        for n_top in cls.N_TOP_SWEEP:
            lo = FeatureInterferenceScore.compute(
                sae, small, n_features=cls.N_FEATURES, n_top=n_top, seed=draw
            )
            hi = FeatureInterferenceScore.compute(
                sae, large, n_features=cls.N_FEATURES, n_top=n_top, seed=draw
            )
            out[n_top] = (lo, hi)
        return out

    @classmethod
    def _old_attenuation(cls, n_top):
        """The exact multiplicative factor the ``z_baseline = z`` defect carried."""
        return ((cls.N_SMALL - n_top) / cls.N_SMALL) / ((cls.N_LARGE - n_top) / cls.N_LARGE)

    @classmethod
    def _mean_ratios(cls):
        """``n_top -> mean over DRAWS draws`` of (score at N_SMALL / at N_LARGE).

        Deliberately NOT memoised. Each test re-derives from the
        implementation it is asserting about; caching the inputs would let a
        stale value survive a change in ``compute`` (verified: a class-level
        cache made both tests pass against a deliberately broken
        implementation in the same process).
        """
        totals = {n: [] for n in cls.N_TOP_SWEEP}
        for draw in range(cls.DRAWS):
            for n_top, (lo, hi) in cls._draw_ratios(draw).items():
                if n_top == 100:
                    # Unchanged: at n_top=100 both ends get a 100-row treatment,
                    # so any movement is the statistic, not the split size.
                    assert lo["n_treated"] == hi["n_treated"] == n_top
                assert hi["mean_interference"] > 0
                totals[n_top].append(lo["mean_interference"] / hi["mean_interference"])
        return {n: sum(v) / len(v) for n, v in totals.items()}

    def test_score_does_not_scale_with_n_top_over_n(self):
        means = self._mean_ratios()

        # One-sided floor per n_top. The defect was an *attenuation*, so the
        # lower edge is what carries the weight: measured worst per-n_top mean
        # over 15 blocks is 0.824 (fixed) against 0.153 (old).
        for n_top, mean_ratio in means.items():
            assert mean_ratio > 0.40, (
                f"n_top={n_top}: score is {mean_ratio:.3f}x its N={self.N_LARGE} "
                f"value on average over {self.DRAWS} draws. The closed-form "
                f"{self._old_attenuation(n_top):.3f}x (1 - n_top/N) attenuation "
                f"is exactly what this looks like."
            )

        # The dependence on n_top is the discriminator. Measured across blocks:
        # 1.056..1.233 (fixed) against 5.641..6.940 (attenuated). 2.5 sits in
        # the gap with ~2x on both sides.
        spread = max(means.values()) / min(means.values())
        assert spread < 2.5, (
            f"mean score varies {spread:.2f}x across n_top={list(self.N_TOP_SWEEP)} "
            f"on identical structure; {means} vs expected near 1 each. The old "
            f"defect's factor alone spans "
            f"{self._old_attenuation(25) / self._old_attenuation(200):.2f}x over "
            f"the same sweep, which is the n_top/N attenuation returning."
        )

    def test_mean_score_is_comparable_across_a_tenfold_dataset_growth(self):
        """Positive form of the same claim, averaged so the band can be tight.

        Per-draw the ratio has sd 0.146 and cannot be pinned; its mean over
        DRAWS draws can. Measured block means at n_top=100 are 0.883..0.997
        (fixed) against 0.537..0.607 (attenuated). The 0.75 edge sits 0.13
        below the worst measured block and about 5 standard errors below the
        population mean, so this is a statement about the population, not about
        one draw.
        """
        mean_ratio = self._mean_ratios()[100]
        assert 0.75 < mean_ratio < 1.35, (
            f"mean score moved {mean_ratio:.3f}x between N={self.N_SMALL} and "
            f"N={self.N_LARGE} over {self.DRAWS} draws; a score used to compare "
            f"models across corpora must not depend on N"
        )


class TestDeterminism:
    def test_same_seed_gives_identical_output(self):
        sae, reps = _sae(), _block_reps(300)
        first = FeatureInterferenceScore.compute(sae, reps, n_features=20, n_top=60, seed=3)
        second = FeatureInterferenceScore.compute(sae, reps, n_features=20, n_top=60, seed=3)

        for key in ("mean_interference", "max_interference", "min_interference", "n_treated"):
            assert first[key] == second[key], f"{key} is not reproducible"

    def test_global_rng_is_not_consumed(self):
        """The private-generator requirement, asserted directly.

        If `compute` drew from the process-global stream, perturbing that
        stream between the two calls would change the feature subsample and
        therefore the score.
        """
        sae, reps = _sae(), _block_reps(300)
        torch.manual_seed(0)
        first = FeatureInterferenceScore.compute(sae, reps, n_features=20, n_top=60, seed=0)

        torch.manual_seed(12345)
        torch.randn(500)  # churn the global stream
        second = FeatureInterferenceScore.compute(sae, reps, n_features=20, n_top=60, seed=0)

        assert first["mean_interference"] == second["mean_interference"]
