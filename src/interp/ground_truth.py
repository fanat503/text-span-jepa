# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Ground truth recovery test: validate the interpretability pipeline
#
# THE PROBLEM: how do you know your metrics aren't lying?
#
# THE SOLUTION: create a model with KNOWN structure, then check
# if the pipeline recovers it. If it does → pipeline is valid.
# If not → pipeline has a bug or metric is misleading.
#
# This is the equivalent of a "unit test for the entire pipeline."
# No interpretability paper does this, and reviewers love it.
#
# Method:
# 1. Create a synthetic model where features are KNOWN
#    (e.g., first 10 dims = class, next 10 = position, rest = noise)
# 2. Run the full pipeline on this model
# 3. Verify:
#    - SAE discovers the known feature groups
#    - Probing complexity matches known structure
#    - Polysemanticity correctly identifies non-polysemantic dims
#    - CKA correctly reflects structural similarity
#
# A validity test is only worth its cost if it can FAIL. Every pass/fail
# decision below is therefore a bound read off a measured null rather than a
# plausible-looking constant; the measurements and the null suite they come
# from are in the threshold block further down this file.

import math

import torch

# ═══════════════════════════════════════════════════════════════════════
# PASS/FAIL THRESHOLDS — every one measured against a null
# ═══════════════════════════════════════════════════════════════════════
#
# Before this block the four pass/fail decisions in this module were bare
# literals: `frac_monosemantic > 0`, `10 < eff_dim < 60`, `0 < aniso < 0.99`,
# `n_valid >= n_total - 1`. None of them was a measurement, and the audit
# showed each one accepts data the module is supposed to reject.
#
# Every constant below is a bound read off a null distribution measured on
# this data, at N=300, D=64 (the shape the validators actually use). The null
# suite is a set of STRUCTURE-FREE representation matrices — matrices that
# carry none of the class / position / depth signal the generator plants. The
# windows are the gaps between the structured measurement and the nearest
# null on each side.
#
# STRUCTURED (50 `SyntheticStructuredModel.generate` seeds):
#     effective_dimension  min 37   median 38   max 39
#     anisotropy           min 0.957547  median 0.961584  max 0.964939
#
# NULL SUITE (260 draws, structure-free):
#     name          draws  effective_dimension      anisotropy
#     isotropic      60    62 .. 63                 0.578705 .. 0.648987
#     same_scale     60    58 .. 59                 0.884535 .. 0.911024
#     het_scale.5    60    55 .. 59                 0.865978 .. 0.970509
#     low_rank8      40    8                        0.987390 .. 0.990536
#     dead16         20    47                       1.0
#     rank1          20    1                        1.0
#
#     isotropic  = iid Gaussian.                    (the audit's example null)
#     same_scale = iid Gaussian rescaled to the generator's own per-dimension
#                  standard deviation: same nuisance scale profile, no
#                  semantics. This is the realistic null.
#     het_scale.5= iid Gaussian with per-column lognormal(0, 0.5) scale:
#                  a nuisance per-dimension scale spread, which real encoder
#                  representations have.
#     low_rank8  = rank-8 signal + 0.2 noise.       (too few live dimensions)
#     dead16     = iid Gaussian with 16 zero dims.  (partially collapsed)
#     rank1      = rank-1 signal.                   (fully collapsed)
#
# Verdict of the old windows `(10, 60)` and `(0, 0.99)` on that suite:
#     same_scale 60/60 PASS, het_scale.5 60/60 PASS, everything else rejected.
#     120 of 260 structure-free matrices were declared a valid structured
#     representation. (The iid Gaussian the audit names is the one null the
#     old effective-dimension window happens to catch, at 62..63.)
#
# Verdict of the windows below on the same suite: 0/260 pass.

# Effective-dimension window. Lower edge: the largest effective dimension any
# null produces BELOW the structured value (low_rank8 is exactly 8 on all 40
# draws; rank1 is 1). Upper edge: the smallest effective dimension any null
# produces ABOVE it (dead16 is exactly 47 on all 20 draws; het_scale.5 starts
# at 55, same_scale at 58, isotropic at 62). Both nulls are exact integers, so
# the strict inequalities reject them with no floating-point slack to argue
# about. Structured margin: 29 above the lower edge, 8 below the upper one.
_EFF_DIM_NULL_LOWER = 8.0
_EFF_DIM_NULL_UPPER = 47.0

# Anisotropy window. Lower edge 0.93, placed between the largest anisotropy
# measured among the nulls that stay below the structured value (same_scale,
# 0.911024 over 60 draws; isotropic, 0.648987) and the structured minimum
# (0.957547). het_scale.5 reaches 0.970509 and so CROSSES the structured
# range — anisotropy alone cannot exclude that null, and the
# effective-dimension window above is what rejects it. Upper edge 0.98,
# placed between the smallest anisotropy measured above the structured value
# (low_rank8, 0.987390 over 40 draws; dead16 and rank1 are exactly 1.0) and
# the structured maximum (0.964939).
# Structured margin: 0.0275 above the lower edge, 0.0151 below the upper one.
_ANISO_NULL_LOWER = 0.93
_ANISO_NULL_UPPER = 0.98

# PSI. `_compute_dim_psi` returns 0.0 for a dimension that raises, and the
# outer handler returns mean_psi = inf, so a completely broken index reports
# the best possible score (frac_monosemantic = 1.0). Measured:
#     iid Gaussian 300x64, 40 draws -> mean_psi = 0.0 exactly, 40/40
#     broken index (every dim raises) -> mean_psi = 0.0, frac_monosemantic = 1.0
#     the structured data           -> mean_psi = 0.0905771255 (deterministic)
# `frac_monosemantic` is 1.0 on the structured data, on the null, AND on a
# broken index — three-way degenerate, zero discriminative power, so it is
# reported but not used to decide. `mean_psi` separates all three.
_PSI_NULL_MAX_MEAN_PSI = 0.0

# DCI informativeness. The estimator is the mean adjusted R^2 of a full
# linear predictor (see `disentanglement.DCIMetrics`). Pooled null over 192
# independent N=200, D=64, K=3 draws (three `measure_noise_floor` batches of
# 64): mean 0.000522662, sd 0.041811848, observed maximum 0.101496144.
# The threshold is mean + 4 sd = 0.167770056; 4 sd is a two-sided normal
# tail of about 6e-5, so a run with no factor information passes by chance
# once in ~17000. The old literal 0.1 sits INSIDE the measured null (one
# draw of 192 reached 0.1015). Structured measurement: 0.9915197211.
_DCI_NULL_MEAN = 0.000522662
_DCI_NULL_SD = 0.041811848
_DCI_NULL_SIGMAS = 4.0
_DCI_NULL_THRESHOLD = _DCI_NULL_MEAN + _DCI_NULL_SIGMAS * _DCI_NULL_SD  # 0.167770056


class SyntheticStructuredModel:
    """A model with KNOWN feature structure for pipeline validation.

    Feature layout (D=64 default):
    - Dims 0-15: class identity (one-hot-ish, high SNR)
    - Dims 16-31: position encoding (sinusoidal)
    - Dims 32-47: syntactic depth (continuous)
    - Dims 48-63: noise (Gaussian, no structure)

    All interpretability metrics should recover this structure:
    - Class dims: low PSI, high probe selectivity
    - Position dims: structured but different from class
    - Depth dims: continuous, moderately interpretable
    - Noise dims: high PSI, low selectivity, should be ignored
    """

    def __init__(self, n_samples=500, embed_dim=64, n_classes=5, seq_len=32, snr=5.0, device="cpu"):
        self.n_samples = n_samples
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.snr = snr
        self.device = device
        # Feature group boundaries
        self.class_end = 16
        self.position_end = 32
        self.depth_end = 48
        self.noise_end = 64

        # Class encoding uses stride-3 slots inside [0, class_end): more than
        # class_end // 3 classes would silently overlap/wrap onto neighbors.
        if self.n_classes * 3 > self.class_end:
            raise ValueError(
                f"n_classes={self.n_classes} does not fit the class feature "
                f"budget: need {self.n_classes * 3} dims, have {self.class_end}",
            )

    def generate(self, seed=42):
        """Generate representations with known structure.

        Returns:
            dict with representations, labels, and ground truth info

        """
        torch.manual_seed(seed)

        N = self.n_samples
        D = self.embed_dim

        representations = torch.zeros(N, D)
        labels = torch.randint(0, self.n_classes, (N,))
        positions = torch.rand(N) * self.seq_len  # Position in sequence
        depths = torch.rand(N) * 10  # Syntactic depth [0, 10]

        # Class features: one-hot + noise
        for i in range(N):
            # One-hot-like encoding for class
            start = (labels[i].item() * 3) % self.class_end
            representations[i, start : start + 3] = self.snr
            # Add noise
            representations[i, : self.class_end] += torch.randn(self.class_end) * 0.5

        # Position features: sinusoidal
        for i in range(N):
            pos = positions[i]
            for j in range(self.position_end - self.class_end):
                idx = self.class_end + j
                freq = (j // 2 + 1) * 0.1
                if j % 2 == 0:
                    representations[i, idx] = math.sin(pos * freq) * self.snr
                else:
                    representations[i, idx] = math.cos(pos * freq) * self.snr
            representations[i, self.class_end : self.position_end] += (
                torch.randn(self.position_end - self.class_end) * 0.3
            )

        # Depth features: continuous
        for i in range(N):
            representations[i, self.position_end : self.depth_end] = (
                depths[i] / 10.0 * self.snr + torch.randn(self.depth_end - self.position_end) * 0.5
            )

        # Noise features: pure Gaussian, written PER SAMPLE (the original code
        # sat outside the sample loop and only populated the last row).
        for i in range(N):
            representations[i, self.depth_end : self.noise_end] = torch.randn(
                self.noise_end - self.depth_end,
            )

        return {
            "representations": representations,
            "labels": labels,
            "positions": positions,
            "depths": depths,
            "ground_truth": {
                "class_dims": list(range(self.class_end)),
                "position_dims": list(range(self.class_end, self.position_end)),
                "depth_dims": list(range(self.position_end, self.depth_end)),
                "noise_dims": list(range(self.depth_end, self.noise_end)),
                "snr": self.snr,
                "n_classes": self.n_classes,
            },
        }


class GroundTruthValidation:
    """Validate the interpretability pipeline on known-structure data.

    If the pipeline correctly recovers the known structure, we can
    trust its results on real models.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def validate_probing_complexity(self):
        """Test: can probing complexity detect that class info
        is linearly accessible but depth info needs nonlinear probes?

        Expected: class → depth=1 sufficient, depth → depth>1 needed
        """
        from src.interp.probing_complexity import ProbingComplexityCurve

        synth = SyntheticStructuredModel(n_samples=200, embed_dim=64, device=self.device)
        data = synth.generate()

        pcc = ProbingComplexityCurve(
            embed_dim=64,
            depths=(1, 2, 3),
            max_epochs=30,
            min_accuracy=0.6,
            device=self.device,
        )

        # Class labels: should be accessible with linear probe
        with torch.enable_grad():
            class_result = pcc.evaluate(data["representations"], data["labels"], "class")
            # Depth labels: might need deeper probe
            depth_labels = (data["depths"] * 3).long().clamp(0, 4)  # Bin into 5 classes
            depth_result = pcc.evaluate(data["representations"], depth_labels, "depth")

        class_linear_sufficient = class_result["min_extracting_depth"] <= 1
        depth_needs_nonlinear = depth_result["min_extracting_depth"] > 1

        return {
            "class_min_depth": class_result["min_extracting_depth"],
            "depth_min_depth": depth_result["min_extracting_depth"],
            "class_linear_sufficient": class_linear_sufficient,
            "depth_needs_nonlinear": depth_needs_nonlinear,
            "pipeline_valid": class_linear_sufficient,  # At minimum, class should be linear
            "class_max_acc": class_result["max_accuracy"],
            "depth_max_acc": depth_result["max_accuracy"],
        }

    def validate_polysemanticity(self):
        """Test: does PSI score the known-structure data above a
        structure-free null?

        Expected: mean PSI > 0 on the structured data, and the class dims
        are the monosemantic ones.

        The old predicate was `frac_monosemantic > 0`. Measured, that
        statistic is 1.0 on the structured data, 1.0 on an iid Gaussian of
        the same shape (40/40 draws) and 1.0 on a PSI that raises on every
        dimension — so it could not fail, which is the defect this check
        exists to catch. `mean_psi` is the discriminator: 0.0 on the null,
        0.0 on a broken index, 0.0906 on the structured data.
        """
        from src.interp.polysemanticity import PolysemanticityIndex

        synth = SyntheticStructuredModel(n_samples=300, embed_dim=64, device=self.device)
        data = synth.generate()

        psi = PolysemanticityIndex(
            n_clusters_range=(2, 3),
            n_top_activations=30,
            n_dimensions_sample=8,
        )

        result = psi.compute(data["representations"], data["labels"])

        mean_psi = float(result["mean_psi"])
        n_scored = len(result.get("per_dim_psi", []))

        # A PSI that raised returns mean_psi = inf (outer handler) or
        # scored nothing at all; either must be a failure, not a pass.
        psi_usable = math.isfinite(mean_psi) and n_scored > 0
        above_null = mean_psi > _PSI_NULL_MAX_MEAN_PSI

        return {
            "mean_psi": mean_psi,
            "frac_monosemantic": result["frac_monosemantic"],
            "n_dims_scored": n_scored,
            "psi_null_max_mean_psi": _PSI_NULL_MAX_MEAN_PSI,
            "psi_usable": psi_usable,
            "psi_above_null": above_null,
            "pipeline_valid": psi_usable and above_null,
        }

    def validate_geometry(self):
        """Test: does geometry correctly identify that the synthetic
        model has structured (non-random) representations?

        Expected: effective dimension and anisotropy both land in the gap
        between the structured data and the measured null suite. See the
        threshold block at the top of this module for the suite and the
        numbers; the short version is that the old windows (10, 60) and
        (0, 0.99) admitted 120 of 260 structure-free matrices and these
        admit none.
        """
        from src.interp.representation_geometry import RepresentationGeometry

        synth = SyntheticStructuredModel(n_samples=300, embed_dim=64, device=self.device)
        data = synth.generate()

        geom = RepresentationGeometry.compute_all(data["representations"])

        # Both edges are null bounds, not round numbers: the structured data
        # is excluded from neither, and every null is excluded from both.
        reasonable_eff_dim = _EFF_DIM_NULL_LOWER < geom["effective_dimension"] < _EFF_DIM_NULL_UPPER
        reasonable_anisotropy = _ANISO_NULL_LOWER < geom["anisotropy"] < _ANISO_NULL_UPPER

        return {
            "effective_dim": geom["effective_dimension"],
            "anisotropy": geom["anisotropy"],
            "eff_dim_window": (_EFF_DIM_NULL_LOWER, _EFF_DIM_NULL_UPPER),
            "aniso_window": (_ANISO_NULL_LOWER, _ANISO_NULL_UPPER),
            "reasonable_eff_dim": reasonable_eff_dim,
            "reasonable_anisotropy": reasonable_anisotropy,
            "pipeline_valid": reasonable_eff_dim and reasonable_anisotropy,
        }

    def validate_disentanglement(self):
        """Test: do disentanglement metrics correctly identify
        that the synthetic model has partially disentangled features?

        Expected: class dims disentangle, position dims somewhat,
        noise dims don't.
        """
        from src.interp.disentanglement import DCIMetrics

        synth = SyntheticStructuredModel(n_samples=200, embed_dim=64, device=self.device)
        data = synth.generate()

        # Use class and position as "factors"
        factors = torch.stack(
            [
                data["labels"].float(),
                data["positions"] / 32,
                data["depths"] / 10,
            ],
            dim=1,
        )

        result = DCIMetrics.compute(data["representations"], factors)

        informativeness = result["informativeness"]

        # The literal 0.1 that stood here sat inside the measured null: 1 of
        # 192 iid N=200/D=64/K=3 draws reached 0.1015, so noise could pass.
        # The bound is now mean + 4 sd of that null (see the threshold block).
        # A NaN from DCIMetrics' own failure path compares False here, which
        # is the behaviour a failure path should have.
        return {
            "disentanglement": result["disentanglement"],
            "completeness": result["completeness"],
            "informativeness": informativeness,
            "informativeness_threshold": _DCI_NULL_THRESHOLD,
            "pipeline_valid": informativeness > _DCI_NULL_THRESHOLD,
        }

    def full_validation(self):
        """Run all validation tests.

        Returns:
            dict with per-test results and overall pass/fail

        `pipeline_reliable` was `n_valid >= n_total - 1`, which is a licence
        for exactly one of the four checks to be broken. The suite has four
        checks and the audit found four distinct ways for it to be blind, so
        tolerating one failure is tolerating one whole class of blindness —
        and the blind checks are the ones that cannot fail at all, so in
        practice the tolerated failure is a real one. Measured on the code as
        it stood: the probing check already fails (class_min_depth = 2 while
        class_max_acc = 1.0) and the old rule still reported
        `pipeline_reliable: True`.

        The rule is now unanimity. Injecting one known failure into an
        otherwise healthy run must flip the summary, which is the property
        `tests/test_interp_ground_truth_thresholds.py` pins.
        """
        results = {}

        try:
            results["probing_complexity"] = self.validate_probing_complexity()
        except Exception as e:
            results["probing_complexity"] = {"pipeline_valid": False, "error": str(e)}

        try:
            results["polysemanticity"] = self.validate_polysemanticity()
        except Exception as e:
            results["polysemanticity"] = {"pipeline_valid": False, "error": str(e)}

        try:
            results["geometry"] = self.validate_geometry()
        except Exception as e:
            results["geometry"] = {"pipeline_valid": False, "error": str(e)}

        try:
            results["disentanglement"] = self.validate_disentanglement()
        except Exception as e:
            results["disentanglement"] = {"pipeline_valid": False, "error": str(e)}

        n_valid = sum(1 for v in results.values() if v.get("pipeline_valid", False))
        n_total = len(results)

        results["_summary"] = {
            "n_tests_passed": n_valid,
            "n_tests_total": n_total,
            "all_passed": n_valid == n_total,
            "pipeline_reliable": n_valid == n_total,
            "n_failed": n_total - n_valid,
            "failed_tests": sorted(
                name for name, v in results.items() if not v.get("pipeline_valid", False)
            ),
        }

        return results
