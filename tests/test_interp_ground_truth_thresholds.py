# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""TASK-14: the pass/fail thresholds in `ground_truth.py` must be able to fail.

What this file is for
--------------------
`src/interp/ground_truth.py` is the module whose stated job is catching
exactly the failure modes this campaign found, and every one of its four
pass/fail decisions was a bare literal that could not fail:

    validate_polysemanticity   frac_monosemantic > 0
    validate_geometry          10 < eff_dim < 60
    validate_geometry          0 < aniso < 0.99
    full_validation            n_valid >= n_total - 1

Each is replaced by a bound measured against a NULL: a structure-free
representation matrix of the same shape as the data the validator actually
sees. These tests rebuild that null suite and assert two things about it —
the thresholds reject all of it, and they still accept the structured data.
A test that only asserted the second would be satisfied by a threshold of
`-inf < x < inf`, which is the defect being fixed.

Determinism
-----------
Every matrix here comes from an explicit `torch.Generator`, and the
structured data comes from `SyntheticStructuredModel.generate(seed=...)`,
which reseeds. The measurements in the module's threshold block were taken
with exactly these seeds, so re-running this file reproduces them.
"""

from __future__ import annotations

import math

import torch

from src.interp import ground_truth as gt
from src.interp.ground_truth import GroundTruthValidation, SyntheticStructuredModel

# Read through `getattr` rather than importing directly: against a module that
# has no calibrated thresholds at all (the pre-TASK-14 version) the test file
# must still import and then fail on behaviour, instead of dying during
# collection with an ImportError that proves nothing.
_EFF_DIM_NULL_LOWER = getattr(gt, "_EFF_DIM_NULL_LOWER", None)
_EFF_DIM_NULL_UPPER = getattr(gt, "_EFF_DIM_NULL_UPPER", None)
_ANISO_NULL_LOWER = getattr(gt, "_ANISO_NULL_LOWER", None)
_ANISO_NULL_UPPER = getattr(gt, "_ANISO_NULL_UPPER", None)
_PSI_NULL_MAX_MEAN_PSI = getattr(gt, "_PSI_NULL_MAX_MEAN_PSI", None)
_DCI_NULL_THRESHOLD = getattr(gt, "_DCI_NULL_THRESHOLD", None)

N, D = 300, 64


# ═══════════════════════════════════════════════════════════════════
# The null suite: structure-free 300x64 representation matrices
# ═══════════════════════════════════════════════════════════════════


def _isotropic(n=20, seed0=7000):
    """iid Gaussian — the audit's example null, no scale profile at all."""
    return [torch.randn(N, D, generator=torch.Generator().manual_seed(seed0 + s)) for s in range(n)]


def _same_scale(n=20, seed0=1100):
    """Gaussian rescaled to the generator's own per-dimension std.

    Same nuisance scale profile as the real data, none of its semantics.
    This is the realistic null and the one the old windows waved through.
    """
    out = []
    for s in range(n):
        reps = SyntheticStructuredModel(n_samples=N, embed_dim=D).generate(seed=142 + s)[
            "representations"
        ]
        gen = torch.Generator().manual_seed(seed0 + s)
        out.append(torch.randn(N, D, generator=gen) * reps.std(dim=0, keepdim=True))
    return out


def _het_scale(n=20, seed0=300, scale_seed0=3300, sigma=0.5):
    """Gaussian with per-column lognormal scale: nuisance scale spread."""
    out = []
    for s in range(n):
        col = torch.randn(N, D, generator=torch.Generator().manual_seed(seed0 + s))
        scale = torch.exp(
            torch.randn(D, generator=torch.Generator().manual_seed(scale_seed0 + s)) * sigma
        )
        out.append(col * scale)
    return out


def _low_rank(rank=8, n=10, seed0=400, noise_seed0=4100):
    """`rank` live directions plus noise — too few effective dimensions."""
    out = []
    for s in range(n):
        a = torch.randn(N, rank, generator=torch.Generator().manual_seed(seed0 + s))
        b = torch.randn(rank, D, generator=torch.Generator().manual_seed(seed0 + 5000 + s))
        noise = torch.randn(N, D, generator=torch.Generator().manual_seed(noise_seed0 + s))
        out.append(a @ b + 0.2 * noise)
    return out


def _dead_dims(k=16, n=6, seed0=100):
    """iid Gaussian with `k` zeroed dimensions — partially collapsed."""
    out = []
    for s in range(n):
        reps = torch.randn(N, D, generator=torch.Generator().manual_seed(seed0 + s))
        reps[:, D - k :] = 0.0
        out.append(reps)
    return out


def _rank1(n=6, seed0=200, coef_seed0=2200):
    """Fully collapsed: a single direction."""
    out = []
    for s in range(n):
        v = torch.randn(N, 1, generator=torch.Generator().manual_seed(seed0 + s))
        coef = torch.randn(1, D, generator=torch.Generator().manual_seed(coef_seed0 + s))
        out.append(v @ coef)
    return out


NULL_SUITE = {
    "isotropic": _isotropic,
    "same_scale": _same_scale,
    "het_scale": _het_scale,
    "low_rank8": _low_rank,
    "dead16": _dead_dims,
    "rank1": _rank1,
}

# The literals the audit found, kept here so the tests can measure what they
# accepted rather than merely assert that the new ones are strict.
OLD_EFF_DIM_WINDOW = (10.0, 60.0)
OLD_ANISO_WINDOW = (0.0, 0.99)


def _geometry(reps):
    from src.interp.representation_geometry import RepresentationGeometry

    return RepresentationGeometry.compute_all(reps)


def _inside(value, window):
    return window[0] < value < window[1]


# ═══════════════════════════════════════════════════════════════════
# Geometry: the two windows must exclude every structure-free matrix
# ═══════════════════════════════════════════════════════════════════


class TestGeometryThresholdsRejectTheNull:
    def test_structured_data_is_inside_the_measured_windows(self):
        """Sanity: the replacement windows still accept the real data.

        Without this, a window that excluded everything would satisfy the
        rejection tests below.
        """
        eff, aniso = [], []
        for s in range(6):
            reps = SyntheticStructuredModel(n_samples=N, embed_dim=D).generate(seed=42 + s)[
                "representations"
            ]
            g = _geometry(reps)
            eff.append(g["effective_dimension"])
            aniso.append(g["anisotropy"])

        assert all(_inside(v, (_EFF_DIM_NULL_LOWER, _EFF_DIM_NULL_UPPER)) for v in eff), eff
        assert all(_inside(v, (_ANISO_NULL_LOWER, _ANISO_NULL_UPPER)) for v in aniso), aniso

    def test_every_null_family_is_outside_both_windows(self):
        offenders = []
        for name, build in NULL_SUITE.items():
            for i, reps in enumerate(build()):
                g = _geometry(reps)
                if _inside(
                    g["effective_dimension"], (_EFF_DIM_NULL_LOWER, _EFF_DIM_NULL_UPPER)
                ) and (_inside(g["anisotropy"], (_ANISO_NULL_LOWER, _ANISO_NULL_UPPER))):
                    offenders.append((name, i, g["effective_dimension"], g["anisotropy"]))
        assert offenders == [], offenders

    def test_the_old_windows_admitted_structure_free_data(self):
        """Why the old literals had to go: they are not merely loose.

        This is the regression that motivated the card. It is asserted, not
        assumed, so the numbers in the module's threshold block stay honest.
        """
        admitted = {
            name: sum(
                1
                for reps in build()
                if _inside(_geometry(reps)["effective_dimension"], OLD_EFF_DIM_WINDOW)
                and _inside(_geometry(reps)["anisotropy"], OLD_ANISO_WINDOW)
            )
            for name, build in NULL_SUITE.items()
        }
        # `same_scale` and `het_scale` are the offenders; both are admitted
        # by every single draw. If this ever stops holding, the threshold
        # block's provenance needs re-measuring, not defending.
        assert admitted["same_scale"] == 20, admitted
        assert admitted["het_scale"] == 20, admitted

    def test_validate_geometry_fails_on_a_structure_free_matrix(self):
        """End to end: the validator's own predicate must reject a null.

        The validator computes its geometry on its own synthetic data, so
        the predicate is exercised here against a null through the same
        window constants the module uses.
        """
        reps = _same_scale(n=1)[0]
        g = _geometry(reps)
        rejected = not (
            _inside(g["effective_dimension"], (_EFF_DIM_NULL_LOWER, _EFF_DIM_NULL_UPPER))
            and _inside(g["anisotropy"], (_ANISO_NULL_LOWER, _ANISO_NULL_UPPER))
        )
        assert rejected, g

    def test_collapsed_data_is_rejected_from_both_ends(self):
        """Both edges are null bounds, so collapse cannot sneak in either.

        The module docstring promises "not ~0 (collapsed)". A window with a
        plausible-looking lower edge that collapse satisfies would be as
        useless as the one it replaced.
        """
        for build in (_rank1, _dead_dims):
            for reps in build():
                g = _geometry(reps)
                assert not _inside(
                    g["effective_dimension"],
                    (_EFF_DIM_NULL_LOWER, _EFF_DIM_NULL_UPPER),
                ), g
                assert not _inside(g["anisotropy"], (_ANISO_NULL_LOWER, _ANISO_NULL_UPPER)), g

    def test_validate_geometry_reports_its_own_windows(self):
        result = GroundTruthValidation().validate_geometry()
        assert result["eff_dim_window"] == (_EFF_DIM_NULL_LOWER, _EFF_DIM_NULL_UPPER)
        assert result["aniso_window"] == (_ANISO_NULL_LOWER, _ANISO_NULL_UPPER)
        assert result["pipeline_valid"] is True
        assert result["reasonable_eff_dim"] is True
        assert result["reasonable_anisotropy"] is True

    def test_the_validator_rejects_a_null_it_is_handed(self, monkeypatch):
        """The module's own predicate, run on a real structure-free matrix.

        The tests above check the window CONSTANTS against a null suite. That
        is not enough on its own: a module could keep a calibrated constant
        next to a predicate that ignores it, and every one of those tests
        would still pass. This one drives `validate_geometry` itself, feeding
        it the geometry of a matrix that carries none of the planted signal.

        The `same_scale` null is the discriminating case: its effective
        dimension sits INSIDE the old `(10, 60)` window and its anisotropy
        sits INSIDE the old `(0, 0.99)` window, so the old literals accepted
        it on both counts. Each of the two sub-predicates is asserted
        separately, so loosening one does not hide behind the other.
        """
        from src.interp.representation_geometry import RepresentationGeometry

        null_geom = _geometry(_same_scale(n=1)[0])
        monkeypatch.setattr(
            RepresentationGeometry,
            "compute_all",
            staticmethod(lambda representations: dict(null_geom)),
        )

        result = GroundTruthValidation().validate_geometry()
        assert result["effective_dim"] == null_geom["effective_dimension"]
        assert result["anisotropy"] == null_geom["anisotropy"]
        assert result["reasonable_eff_dim"] is False, result
        assert result["reasonable_anisotropy"] is False, result
        assert result["pipeline_valid"] is False, result

    def test_the_validator_rejects_a_collapsed_matrix(self, monkeypatch):
        """The docstring's "not ~0 (collapsed)" guard, through the predicate."""
        from src.interp.representation_geometry import RepresentationGeometry

        for build in (_rank1, _dead_dims):
            null_geom = _geometry(build(n=1)[0])
            monkeypatch.setattr(
                RepresentationGeometry,
                "compute_all",
                staticmethod(lambda representations, g=null_geom: dict(g)),
            )
            result = GroundTruthValidation().validate_geometry()
            assert result["pipeline_valid"] is False, result


# ═══════════════════════════════════════════════════════════════════
# Polysemanticity: a broken PSI must fail
# ═══════════════════════════════════════════════════════════════════


def _psi(reps, labels):
    from src.interp.polysemanticity import PolysemanticityIndex

    return PolysemanticityIndex(
        n_clusters_range=(2, 3), n_top_activations=30, n_dimensions_sample=8
    ).compute(reps, labels)


def _label_psi_result(n_classes=5, n_scored=8, mean_psi=0.0):
    """The dict `PolysemanticityIndex.compute` returns when it is broken.

    `_compute_dim_psi` has `except Exception: return 0.0`, so an index that
    fails on every dimension reports the best score the metric can produce.
    """
    return {
        "mean_psi": mean_psi,
        "frac_monosemantic": 1.0,
        "per_dim_psi": [0.0] * n_scored,
        "max_psi": 0.0,
        "min_psi": 0.0,
    }


class TestPolysemanticityThreshold:
    def test_null_psi_is_at_or_below_the_threshold(self):
        """The threshold is the null's maximum, measured on this data."""
        for reps in _isotropic(n=10):
            labels = torch.randint(0, 5, (N,), generator=torch.Generator().manual_seed(11))
            mean_psi = _psi(reps, labels)["mean_psi"]
            assert mean_psi <= _PSI_NULL_MAX_MEAN_PSI, mean_psi

    def test_broken_psi_fails_validation(self, monkeypatch):
        """The card's headline defect: a completely broken PSI passed.

        A PSI that raises on every dimension is indistinguishable, by its
        own output, from a perfect one. The validator has to notice anyway.
        """
        from src.interp import polysemanticity

        monkeypatch.setattr(
            polysemanticity.PolysemanticityIndex,
            "compute",
            lambda self, representations, labels=None: _label_psi_result(),
        )
        result = GroundTruthValidation().validate_polysemanticity()
        assert result["pipeline_valid"] is False
        assert result["psi_usable"] is True  # well-formed, just all zeros
        assert result["psi_above_null"] is False

    def test_broken_psi_whose_result_is_infinite_fails_validation(self, monkeypatch):
        """The outer handler returns mean_psi = inf, which `> 0` would accept."""
        from src.interp import polysemanticity

        broken = _label_psi_result()
        broken["mean_psi"] = float("inf")
        monkeypatch.setattr(
            polysemanticity.PolysemanticityIndex,
            "compute",
            lambda self, representations, labels=None: broken,
        )
        result = GroundTruthValidation().validate_polysemanticity()
        assert result["pipeline_valid"] is False
        assert result["psi_usable"] is False

    def test_psi_that_scored_nothing_fails_validation(self, monkeypatch):
        from src.interp import polysemanticity

        empty = _label_psi_result()
        empty["per_dim_psi"] = []
        monkeypatch.setattr(
            polysemanticity.PolysemanticityIndex,
            "compute",
            lambda self, representations, labels=None: empty,
        )
        result = GroundTruthValidation().validate_polysemanticity()
        assert result["pipeline_valid"] is False
        assert result["psi_usable"] is False

    def test_structured_data_clears_the_threshold(self):
        result = GroundTruthValidation().validate_polysemanticity()
        assert result["mean_psi"] > _PSI_NULL_MAX_MEAN_PSI
        assert result["pipeline_valid"] is True
        assert result["n_dims_scored"] == 8

    def test_frac_monosemantic_alone_could_not_have_worked(self):
        """Why the statistic was swapped, asserted rather than asserted-about.

        `frac_monosemantic` is 1.0 on the structured data, 1.0 on a
        structure-free null, and 1.0 on a broken index. The old predicate
        `frac_monosemantic > 0` was satisfied by all three.
        """
        structured = GroundTruthValidation().validate_polysemanticity()
        reps = _isotropic(n=1)[0]
        labels = torch.randint(0, 5, (N,), generator=torch.Generator().manual_seed(11))
        null = _psi(reps, labels)
        broken = _label_psi_result()

        assert structured["frac_monosemantic"] == 1.0
        assert null["frac_monosemantic"] == 1.0
        assert broken["frac_monosemantic"] == 1.0
        # ...while the statistic now used separates them.
        assert structured["mean_psi"] > null["mean_psi"]
        assert structured["mean_psi"] > broken["mean_psi"]


# ═══════════════════════════════════════════════════════════════════
# Disentanglement: the informativeness bound must sit above the noise floor
# ═══════════════════════════════════════════════════════════════════


class TestInformativenessThreshold:
    def test_measured_null_stays_under_the_threshold(self):
        from src.interp.disentanglement import DCIMetrics

        floor = DCIMetrics.measure_noise_floor(200, 64, 3, n_trials=16, seed=0)
        assert floor["max"] < _DCI_NULL_THRESHOLD, floor
        # The literal that stood here. One of 192 measured null draws
        # exceeded it, which is how a structure-free matrix passed.
        assert floor["max"] > 0.1, floor

    def test_the_validator_rejects_an_informativeness_the_null_actually_produced(self, monkeypatch):
        """The module's own predicate, run on a real null measurement.

        This replicates `DCIMetrics.measure_noise_floor(200, 64, 3, 64,
        seed=0)` draw for draw and keeps the worst one. The maximum is not
        an invented value: it is the largest mean adjusted R^2 the estimator
        produced over 64 independent draws of factor-free Gaussian data,
        and it is the same 0.101496144 the threshold block records. The old
        literal was `> 0.1`, so that draw passed. The bound has to reject
        the value the null really reached, not merely a value chosen to sit
        under it.
        """
        from src.interp.disentanglement import DCIMetrics

        gen = torch.Generator().manual_seed(0)
        worst, worst_raw = None, None
        for _ in range(64):
            reps = torch.randn(200, 64, generator=gen)
            factors = torch.randn(200, 3, generator=gen)
            result = DCIMetrics.compute(reps, factors)
            if worst is None or result["informativeness"] > worst["informativeness"]:
                worst, worst_raw = dict(result), result

        assert worst["informativeness"] > 0.1, worst
        assert worst["informativeness"] <= _DCI_NULL_THRESHOLD, worst

        monkeypatch.setattr(
            DCIMetrics,
            "compute",
            staticmethod(lambda representations, factors: dict(worst)),
        )
        out = GroundTruthValidation().validate_disentanglement()
        assert out["informativeness"] == worst["informativeness"]
        assert out["pipeline_valid"] is False, out

    def test_structured_data_clears_the_threshold(self):
        result = GroundTruthValidation().validate_disentanglement()
        assert result["informativeness"] > _DCI_NULL_THRESHOLD
        assert result["pipeline_valid"] is True
        assert result["informativeness_threshold"] == _DCI_NULL_THRESHOLD

    def test_nan_informativeness_fails_rather_than_passes(self):
        """DCIMetrics' failure path returns NaN; NaN > x is False."""
        from src.interp import disentanglement

        real_compute = disentanglement.DCIMetrics.compute
        try:
            disentanglement.DCIMetrics.compute = staticmethod(
                lambda reps, factors: {
                    "disentanglement": 0.0,
                    "completeness": 0.0,
                    "informativeness": float("nan"),
                }
            )
            result = GroundTruthValidation().validate_disentanglement()
        finally:
            disentanglement.DCIMetrics.compute = real_compute
        assert result["pipeline_valid"] is False


# ═══════════════════════════════════════════════════════════════════
# full_validation: one known failure must flip the summary
# ═══════════════════════════════════════════════════════════════════


class TestFullValidation:
    def test_summary_uses_unanimity_not_a_tolerated_failure(self):
        results = GroundTruthValidation().full_validation()
        summary = results["_summary"]
        assert summary["n_tests_total"] == 4
        assert summary["all_passed"] == (summary["n_tests_passed"] == summary["n_tests_total"])
        # The old rule was `>= n_total - 1`, i.e. True whenever 3 of 4 pass.
        assert summary["pipeline_reliable"] == summary["all_passed"]

    def test_one_injected_failure_makes_the_suite_unreliable(self, monkeypatch):
        """The derivation: tolerate-one-failure licenses one broken check.

        All four checks pass, then exactly one is made to fail. The summary
        must report unreliable. Under the old rule it reported reliable,
        which is the same as saying the suite cannot notice its own blindness.
        """
        v = GroundTruthValidation()
        monkeypatch.setattr(
            v,
            "validate_geometry",
            lambda: {"pipeline_valid": False, "injected": True},
        )
        monkeypatch.setattr(
            v,
            "validate_probing_complexity",
            lambda: {"pipeline_valid": True},
        )
        monkeypatch.setattr(v, "validate_polysemanticity", lambda: {"pipeline_valid": True})
        monkeypatch.setattr(v, "validate_disentanglement", lambda: {"pipeline_valid": True})

        summary = v.full_validation()["_summary"]
        assert summary["n_tests_passed"] == 3
        assert summary["n_tests_total"] == 4
        assert summary["all_passed"] is False
        assert summary["pipeline_reliable"] is False
        assert summary["n_failed"] == 1
        assert summary["failed_tests"] == ["geometry"]

    def test_a_raising_check_counts_as_a_failure(self, monkeypatch):
        v = GroundTruthValidation()
        monkeypatch.setattr(
            v,
            "validate_polysemanticity",
            lambda: (_ for _ in ()).throw(RuntimeError("psi exploded")),
        )
        results = v.full_validation()
        assert results["polysemanticity"]["pipeline_valid"] is False
        assert "psi exploded" in results["polysemanticity"]["error"]
        assert results["_summary"]["pipeline_reliable"] is False
        assert "polysemanticity" in results["_summary"]["failed_tests"]

    def test_broken_psi_inside_the_full_suite_is_caught(self, monkeypatch):
        """The two defects together: broken PSI, old summary rule.

        Both must be fixed for this to go red, which is the point — the
        broken index alone only cost one of four checks.
        """
        from src.interp import polysemanticity

        monkeypatch.setattr(
            polysemanticity.PolysemanticityIndex,
            "compute",
            lambda self, representations, labels=None: _label_psi_result(),
        )
        v = GroundTruthValidation()
        monkeypatch.setattr(v, "validate_probing_complexity", lambda: {"pipeline_valid": True})
        monkeypatch.setattr(v, "validate_disentanglement", lambda: {"pipeline_valid": True})

        summary = v.full_validation()["_summary"]
        assert summary["failed_tests"] == ["polysemanticity"]
        assert summary["pipeline_reliable"] is False


def test_thresholds_are_finite_and_ordered():
    """No `inf < x < inf` escape hatch, and no missing calibration block."""
    for lo, hi in (
        (_EFF_DIM_NULL_LOWER, _EFF_DIM_NULL_UPPER),
        (_ANISO_NULL_LOWER, _ANISO_NULL_UPPER),
    ):
        assert lo is not None and hi is not None
        assert math.isfinite(lo) and math.isfinite(hi)
        assert lo < hi
    assert math.isfinite(_PSI_NULL_MAX_MEAN_PSI)
    assert math.isfinite(_DCI_NULL_THRESHOLD)
