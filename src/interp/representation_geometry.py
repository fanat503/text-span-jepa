# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Representation geometry metrics for predicting model quality
# From Yadav (2026) "On the Relationship Between Representation Geometry
# and Generalization in Deep Neural Networks"
#
# Key finding: effective dimension (unsupervised geometric metric)
# strongly predicts accuracy (r=0.75, p < 10^-10) across 52 ImageNet models
# AND generalizes to NLP: effective dimension predicts performance for
# 8 encoder models on SST-2/MNLI (r=0.69, p=0.004).
#
# This creates a DIRECT link between JEPA's geometry advantage and
# downstream performance, which is critical for the paper's narrative.

import torch


class RepresentationGeometry:
    """Representation geometry metrics linked to generalization.

    From Yadav (2026):
    - Effective dimension predicts accuracy (r=0.75)
    - Total compression inversely predicts accuracy (r=-0.72)
    - Bidirectional causality: degrading geometry → accuracy loss (r=-0.94)

    For our paper: if JEPA has higher effective dimension AND better
    downstream performance, that's quantitative evidence linking
    geometry to generalization.
    """

    @staticmethod
    @torch.no_grad()
    def effective_dimension(representations, threshold=0.99):
        """Effective dimension: number of SVD components needed to
        explain `threshold` fraction of total variance.

        Yadav (2026): this single metric predicts accuracy better
        than model size, parameter count, or training compute.

        Args:
            representations: (N, D) or (B, T, D)
            threshold: variance explained threshold (0.99 default)

        Returns:
            float: effective dimension

        """
        try:
            if representations.dim() == 3:
                flat = representations.reshape(-1, representations.size(-1))
            else:
                flat = representations

            N, D = flat.shape
            if N < 2 or D < 2:
                return 0.0

            # SVD
            s = torch.linalg.svdvals(flat.float())
            total_var = (s**2).sum()
            if total_var == 0:
                return 0.0

            # Cumulative explained variance
            cumvar = (s**2).cumsum(0) / total_var
            # Find threshold
            eff_dim = (cumvar < threshold).sum().item() + 1
            return min(float(eff_dim), D)
        except Exception:
            return 0.0

    @staticmethod
    @torch.no_grad()
    def total_compression(representations):
        """Total compression: ratio of intrinsic to ambient dimension.

        Yadav (2026): compression inversely predicts accuracy (r=-0.72).
        Lower compression = more information preserved = better.

        Args:
            representations: (N, D) or (B, T, D)

        Returns:
            float: compression ratio in [0, 1]

        """
        try:
            if representations.dim() == 3:
                flat = representations.reshape(-1, representations.size(-1))
            else:
                flat = representations

            _N, D = flat.shape

            # Effective dimension at 0.99 variance
            eff_dim = RepresentationGeometry.effective_dimension(flat, 0.99)
            # Compression = 1 - eff_dim/D
            if D == 0:
                return 0.0
            return 1.0 - eff_dim / D
        except Exception:
            return 0.0

    @staticmethod
    @torch.no_grad()
    def anisotropy(representations):
        """Representation anisotropy: how much representations
        cluster in preferred directions.

        Anisotropy = 1 - (min singular value / max singular value).
        High anisotropy = representations stretched in a few directions.
        Low anisotropy = isotropic (spread equally in all directions).

        MLM models (BERT) are known to be highly anisotropic.
        JEPA should be LESS anisotropic (more isotropic representations).

        Args:
            representations: (N, D) or (B, T, D)

        Returns:
            float: anisotropy in [0, 1]

        """
        try:
            if representations.dim() == 3:
                flat = representations.reshape(-1, representations.size(-1))
            else:
                flat = representations

            flat = flat - flat.mean(dim=0)
            s = torch.linalg.svdvals(flat.float())

            if s[0] == 0:
                return 0.0

            # Anisotropy = 1 - s_min / s_max
            anisotropy = 1.0 - (s[-1] / s[0]).item()
            return max(min(anisotropy, 1.0), 0.0)
        except Exception:
            return 0.0

    @staticmethod
    @torch.no_grad()
    def spectrum_decay_rate(representations):
        """Power-law decay rate of singular value spectrum.

        Fits a power law s_k ~ k^(-alpha) to the singular value spectrum.
        Higher alpha = steeper decay = more anisotropic, fewer effective dims.
        Lower alpha = flatter spectrum = more isotropic, more effective dims.

        BERT (MLM) has steep decay (alpha ~ 1.0).
        JEPA should have shallower decay (alpha ~ 0.5-0.7).

        Args:
            representations: (N, D) or (B, T, D)

        Returns:
            float: power-law exponent alpha

        """
        try:
            if representations.dim() == 3:
                flat = representations.reshape(-1, representations.size(-1))
            else:
                flat = representations

            s = torch.linalg.svdvals(flat.float())
            s = s[s > 1e-10]  # Filter near-zero
            if s.numel() < 3:
                return 0.0

            # Log-log regression: log(s_k) = -alpha * log(k) + const
            k = torch.arange(1, s.numel() + 1, dtype=torch.float32, device=s.device)
            log_k = torch.log(k)
            log_s = torch.log(s)

            # Least squares: alpha = -cov(log_k, log_s) / var(log_k)
            log_k_mean = log_k.mean()
            log_s_mean = log_s.mean()
            cov = ((log_k - log_k_mean) * (log_s - log_s_mean)).sum()
            var = ((log_k - log_k_mean) ** 2).sum()

            if var < 1e-10:
                return 0.0

            alpha = -cov / var
            return max(alpha.item(), 0.0)
        except Exception:
            return 0.0

    @staticmethod
    @torch.no_grad()
    def compute_all(representations):
        """Compute all representation geometry metrics.

        Args:
            representations: (N, D) or (B, T, D)

        Returns:
            dict with all metrics

        """
        return {
            "effective_dimension": RepresentationGeometry.effective_dimension(representations),
            "total_compression": RepresentationGeometry.total_compression(representations),
            "anisotropy": RepresentationGeometry.anisotropy(representations),
            "spectrum_decay_rate": RepresentationGeometry.spectrum_decay_rate(representations),
        }

    @staticmethod
    @torch.no_grad()
    def compare(jepa_reps, baseline_reps):
        """Compare representation geometry between JEPA and baseline.

        THE KEY COMPARISON linking geometry to the paper's narrative.

        Yadav (2026) predicts:
        - Higher effective dimension → better generalization
        - Lower compression → better generalization
        - Lower anisotropy → better generalization (for text models)
        - Shallower decay → more isotropic → better

        Returns:
            dict with per-metric comparison

        """
        jepa_geom = RepresentationGeometry.compute_all(jepa_reps)
        baseline_geom = RepresentationGeometry.compute_all(baseline_reps)

        # Yadav predictions for "better"
        higher_is_better = {"effective_dimension"}
        lower_is_better = {"total_compression", "anisotropy", "spectrum_decay_rate"}

        comparison = {}
        for key in jepa_geom:
            j_val = jepa_geom[key]
            b_val = baseline_geom[key]
            if key in higher_is_better:
                jepa_better = j_val > b_val
            elif key in lower_is_better:
                jepa_better = j_val < b_val
            else:
                jepa_better = None

            comparison[key] = {
                "jepa": j_val,
                "baseline": b_val,
                "diff": j_val - b_val,
                "jepa_better": jepa_better,
            }

        # Overall geometry advantage
        n_better = sum(1 for v in comparison.values() if v["jepa_better"] is True)
        n_total = sum(1 for v in comparison.values() if v["jepa_better"] is not None)
        comparison["_geometry_advantage"] = n_better / max(n_total, 1)

        return comparison


class GeometryDegradationTest:
    """Test bidirectional causality: geometry → accuracy.

    From Yadav (2026): "degrading geometry via noise causes accuracy
    loss (r=-0.94, p < 10^-9), while improving geometry via PCA
    maintains accuracy across architectures."

    For our paper: add noise to JEPA representations, measure both
    geometry degradation AND accuracy loss. If the correlation
    between geometry degradation and accuracy loss is steeper for
    JEPA than MLM, it means JEPA's accuracy depends more on
    geometry quality (geometry is more informative).
    """

    @staticmethod
    @torch.no_grad()
    def noise_degradation_curve(
        model,
        representations,
        labels,
        probe_fn,
        noise_levels=(0.01, 0.05, 0.1, 0.2, 0.5),
        n_trials=3,
        device="cpu",
    ):
        """Add Gaussian noise to representations and measure geometry + accuracy.

        Args:
            model: encoder model
            representations: (N, D) clean representations
            labels: (N,) labels for probe
            probe_fn: callable(representations) -> accuracy
            noise_levels: list of noise standard deviations
            n_trials: number of noise trials per level
            device: compute device

        Returns:
            dict with per-noise-level geometry and accuracy metrics

        """
        results = {}
        clean_geom = RepresentationGeometry.compute_all(representations)
        clean_acc = probe_fn(representations)

        for noise_std in noise_levels:
            geom_list = []
            acc_list = []

            for _ in range(n_trials):
                noisy = representations + torch.randn_like(representations) * noise_std
                geom = RepresentationGeometry.compute_all(noisy)
                acc = probe_fn(noisy)
                geom_list.append(geom)
                acc_list.append(acc)

            # Average over trials
            avg_geom = {}
            for key in geom_list[0]:
                avg_geom[key] = sum(g[key] for g in geom_list) / len(geom_list)

            avg_acc = sum(acc_list) / len(acc_list)

            results[noise_std] = {
                "geometry": avg_geom,
                "accuracy": avg_acc,
                "geometry_degradation": {k: clean_geom[k] - avg_geom[k] for k in avg_geom},
                "accuracy_degradation": clean_acc - avg_acc,
            }

        # Correlation between geometry degradation and accuracy degradation
        geom_degs = []
        acc_degs = []
        for noise_std in noise_levels:
            if noise_std > 0:
                # Use effective_dimension degradation
                ed_deg = results[noise_std]["geometry_degradation"].get("effective_dimension", 0)
                a_deg = results[noise_std]["accuracy_degradation"]
                geom_degs.append(ed_deg)
                acc_degs.append(a_deg)

        correlation = _pearson_r(geom_degs, acc_degs) if len(geom_degs) > 1 else 0.0

        return {
            "clean_geometry": clean_geom,
            "clean_accuracy": clean_acc,
            "per_noise_level": results,
            "geometry_accuracy_correlation": correlation,
        }


def _pearson_r(x_list, y_list):
    """Pearson correlation from lists."""
    if len(x_list) < 2:
        return 0.0
    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = x_c.norm() * y_c.norm()
    if denom < 1e-10:
        return 0.0
    return (x_c @ y_c / denom).item()
