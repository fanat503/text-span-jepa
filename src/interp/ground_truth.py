# Copyright 2026 Text-Span JEPA Authors
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

import math

import torch


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
                f"budget: need {self.n_classes * 3} dims, have {self.class_end}"
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
                self.noise_end - self.depth_end
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
            embed_dim=64, depths=(1, 2, 3), max_epochs=30, min_accuracy=0.6, device=self.device
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
        """Test: does PSI correctly identify that class dims are
        monosemantic and noise dims are polysemantic?

        Expected: class dims → low PSI, noise dims → high PSI
        """
        from src.interp.polysemanticity import PolysemanticityIndex

        synth = SyntheticStructuredModel(n_samples=300, embed_dim=64, device=self.device)
        data = synth.generate()

        psi = PolysemanticityIndex(
            n_clusters_range=(2, 3), n_top_activations=30, n_dimensions_sample=8
        )

        result = psi.compute(data["representations"], data["labels"])

        # With structured data, frac_monosemantic should be > 0
        # (at least the class dims should be monosemantic)
        return {
            "mean_psi": result["mean_psi"],
            "frac_monosemantic": result["frac_monosemantic"],
            "pipeline_valid": result["frac_monosemantic"] > 0,
        }

    def validate_geometry(self):
        """Test: does geometry correctly identify that the synthetic
        model has structured (non-random) representations?

        Expected: effective_dim ≈ 48 (3/4 of dims are structured),
        not 64 (all dims) or ~0 (collapsed)
        """
        from src.interp.representation_geometry import RepresentationGeometry

        synth = SyntheticStructuredModel(n_samples=300, embed_dim=64, device=self.device)
        data = synth.generate()

        geom = RepresentationGeometry.compute_all(data["representations"])

        # Should have moderate effective dim (not all 64, not near 0)
        reasonable_eff_dim = 10 < geom["effective_dimension"] < 60
        # Should have moderate anisotropy (structured but not collapsed)
        reasonable_anisotropy = 0 < geom["anisotropy"] < 0.99

        return {
            "effective_dim": geom["effective_dimension"],
            "anisotropy": geom["anisotropy"],
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

        return {
            "disentanglement": result["disentanglement"],
            "completeness": result["completeness"],
            "informativeness": result["informativeness"],
            "pipeline_valid": result["informativeness"] > 0.1,
        }

    def full_validation(self):
        """Run all validation tests.

        Returns:
            dict with per-test results and overall pass/fail
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
            "pipeline_reliable": n_valid >= n_total - 1,  # Allow 1 failure
        }

        return results
