# Copyright 2026 Text-Span JEPA Authors
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
    def test_score_does_not_scale_with_n_top_over_n(self):
        """Same pool, same structure, two dataset sizes.

        The old implementation multiplied the score by (1 - n_top/N): 0.5 at
        N=240 versus 0.95 at N=2400. A measure that is supposed to compare
        models across corpora cannot carry that factor.

        Measured over six feature subsamples with this data: the fixed
        implementation returns ratios in [0.72, 0.97] and the attenuated one
        in [0.44, 0.59], so 0.65 separates them with margin on both sides.
        A mild downward bias survives at small N -- with a fixed n_top the top
        100 rows are genuinely more extreme in a 2400-row sample than in a
        240-row one -- but the n_top/N factor itself is gone.
        """
        sae = _sae()
        pool = _block_reps(2400)
        order = torch.randperm(2400, generator=torch.Generator().manual_seed(99))

        small = FeatureInterferenceScore.compute(
            sae, pool[order[:240]], n_features=25, n_top=100, seed=0
        )
        large = FeatureInterferenceScore.compute(
            sae, pool[order[:2400]], n_features=25, n_top=100, seed=0
        )

        assert small["n_treated"] == large["n_treated"] == 100
        assert large["mean_interference"] > 0

        ratio = small["mean_interference"] / large["mean_interference"]
        assert 0.65 < ratio < 1.25, (
            f"score moved by {ratio:.3f}x between N=240 and N=2400 on identical "
            f"structure; n_top/N attenuation is still present"
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
