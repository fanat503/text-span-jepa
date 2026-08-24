# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Statistical testing for interpretability claims
#
# WITHOUT proper statistical testing, any claim about JEPA vs MLM
# differences will be dismissed by reviewers. Every metric comparison
# needs confidence intervals and p-values.
#
# Implements:
# - Bootstrap confidence intervals (Efron, 1979)
# - Paired permutation tests (Good, 2000)
# - Multiple comparison correction (Bonferroni, Benjamini-Hochberg)
# - Effect size (Cohen's d)
# - Bayesian posterior probability of JEPA > baseline

import math

import numpy as np
import torch


class BootstrapCI:
    """Bootstrap confidence intervals for any metric.

    Efron (1979). Resample with replacement, compute metric on each
    resample, take percentiles as CI.

    Usage:
        ci = BootstrapCI.compute(metric_fn, data, n_bootstrap=1000, alpha=0.05)
        # ci = {'mean': 0.85, 'ci_lower': 0.82, 'ci_upper': 0.88, 'std': 0.03}
    """

    @staticmethod
    def compute(metric_fn, data, n_bootstrap=1000, alpha=0.05, seed=42, return_samples=False):
        """Compute bootstrap CI.

        Args:
            metric_fn: callable(data_subset) -> float
            data: input data (any format, passed to metric_fn)
            n_bootstrap: number of bootstrap samples
            alpha: significance level (0.05 = 95% CI)
            seed: random seed
            return_samples: whether to return all bootstrap values

        Returns:
            dict with mean, CI bounds, std
        """
        rng = np.random.RandomState(seed)

        # If data is a tensor, we know how to resample
        if isinstance(data, torch.Tensor):
            N = data.size(0)
            bootstrap_values = []
            for _ in range(n_bootstrap):
                idx = torch.from_numpy(rng.randint(0, N, size=N)).long()
                sample = data[idx]
                try:
                    val = metric_fn(sample)
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    bootstrap_values.append(val)
                except Exception:
                    continue
        elif isinstance(data, (list, tuple)):
            N = len(data)
            bootstrap_values = []
            for _ in range(n_bootstrap):
                idx = rng.randint(0, N, size=N)
                sample = [data[i] for i in idx]
                try:
                    val = metric_fn(sample)
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    bootstrap_values.append(val)
                except Exception:
                    continue
        else:
            # Fallback: just compute once
            try:
                val = metric_fn(data)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                return {"mean": val, "ci_lower": val, "ci_upper": val, "std": 0.0, "n_bootstrap": 1}
            except Exception:
                return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n_bootstrap": 0}

        if not bootstrap_values:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n_bootstrap": 0}

        values = np.array(bootstrap_values)
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))

        result = {
            "mean": float(values.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "std": float(values.std()),
            "n_bootstrap": len(bootstrap_values),
        }

        if return_samples:
            result["samples"] = bootstrap_values

        return result

    @staticmethod
    def compare(metric_fn, data_a, data_b, n_bootstrap=1000, alpha=0.05, seed=42):
        """Bootstrap comparison: is metric(data_a) significantly different
        from metric(data_b)?

        Returns:
            dict with difference CI and significance
        """
        rng = np.random.RandomState(seed)

        if isinstance(data_a, torch.Tensor) and isinstance(data_b, torch.Tensor):
            Na, Nb = data_a.size(0), data_b.size(0)
            diffs = []
            for _ in range(n_bootstrap):
                idx_a = torch.from_numpy(rng.randint(0, Na, size=Na)).long()
                idx_b = torch.from_numpy(rng.randint(0, Nb, size=Nb)).long()
                try:
                    va = metric_fn(data_a[idx_a])
                    vb = metric_fn(data_b[idx_b])
                    if isinstance(va, torch.Tensor):
                        va = va.item()
                    if isinstance(vb, torch.Tensor):
                        vb = vb.item()
                    diffs.append(va - vb)
                except Exception:
                    continue
        else:
            return {"significant": False, "p_value": 1.0}

        if not diffs:
            return {"significant": False, "p_value": 1.0}

        diffs = np.array(diffs)
        lower = np.percentile(diffs, 100 * alpha / 2)
        upper = np.percentile(diffs, 100 * (1 - alpha / 2))

        # Significance: CI doesn't include 0
        significant = not (lower <= 0 <= upper)

        # p-value: fraction of bootstrap diffs with opposite sign
        if diffs.mean() > 0:
            p_value = (diffs <= 0).mean()
        else:
            p_value = (diffs >= 0).mean()
        p_value = max(p_value, 1.0 / n_bootstrap)  # Floor

        return {
            "mean_diff": float(diffs.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "significant": significant,
            "p_value": float(p_value),
            "n_bootstrap": len(diffs),
        }


class PairedPermutationTest:
    """Paired permutation test for comparing JEPA vs baseline metrics.

    More powerful than bootstrap when you have paired observations
    (same inputs, different models).

    Good (2000): under H0, the pair (x_i, y_i) is exchangeable.
    We permute the sign of x_i - y_i and recompute the difference.
    p-value = fraction of permutations with |diff| >= observed |diff|.
    """

    @staticmethod
    def compute(values_a, values_b, n_permutations=10000, seed=42):
        """Paired permutation test.

        Args:
            values_a: (N,) array of metric values for model A
            values_b: (N,) array of metric values for model B
            n_permutations: number of permutations
            seed: random seed

        Returns:
            dict with p-value and effect size
        """
        if isinstance(values_a, torch.Tensor):
            values_a = values_a.cpu().numpy()
        if isinstance(values_b, torch.Tensor):
            values_b = values_b.cpu().numpy()

        values_a = np.array(values_a, dtype=float)
        values_b = np.array(values_b, dtype=float)

        N = len(values_a)
        if N != len(values_b) or N < 2:
            return {
                "p_value": 1.0,
                "effect_size": 0.0,
                "significant": False,
                "mean_diff": 0.0,
            }

        # Observed difference
        observed_diff = values_a.mean() - values_b.mean()

        # Permutation test
        rng = np.random.RandomState(seed)
        diffs = values_a - values_b
        count = 0

        for _ in range(n_permutations):
            # Random sign flip
            signs = rng.choice([-1, 1], size=N)
            perm_diff = (diffs * signs).mean()
            if abs(perm_diff) >= abs(observed_diff):
                count += 1

        p_value = (count + 1) / (n_permutations + 1)  # +1 for the observed

        # Cohen's d
        pooled_std = math.sqrt((values_a.var() + values_b.var()) / 2)
        cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0.0

        return {
            "p_value": float(p_value),
            "effect_size": float(cohens_d),
            "significant": p_value < 0.05,
            "mean_diff": float(observed_diff),
            "n_observations": N,
            "n_permutations": n_permutations,
        }


class MultipleComparisonCorrection:
    """Correct for multiple comparisons when testing many metrics.

    When you test 50 metrics, ~2-3 will be "significant" by chance
    at p=0.05. Reviewers WILL catch this.

    Methods:
    - Bonferroni: conservative, p_i * n_tests
    - Benjamini-Hochberg (FDR): less conservative, controls false discovery rate
    """

    @staticmethod
    def bonferroni(p_values):
        """Bonferroni correction: multiply each p-value by number of tests.

        Very conservative. Use when you need strong family-wise error control.
        """
        n = len(p_values)
        corrected = [min(p * n, 1.0) for p in p_values]
        return corrected

    @staticmethod
    def benjamini_hochberg(p_values, alpha=0.05):
        """Benjamini-Hochberg FDR correction.

        Controls the expected fraction of false discoveries.
        Less conservative than Bonferroni.

        Returns:
            dict with corrected p-values and list of significant indices
        """
        n = len(p_values)
        if n == 0:
            return {"corrected": [], "significant": [], "threshold": alpha}

        # Sort p-values
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])

        # BH procedure: find largest k where p_(k) <= k/n * alpha
        threshold_idx = -1
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            if p <= rank / n * alpha:
                threshold_idx = rank - 1

        # Corrected p-values
        corrected = [0.0] * n
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            bh_p = p * n / rank
            corrected[orig_idx] = min(bh_p, 1.0)

        # Significant indices
        significant = []
        if threshold_idx >= 0:
            for i in range(threshold_idx + 1):
                significant.append(indexed[i][0])

        return {
            "corrected": corrected,
            "significant": sorted(significant),
            "threshold": alpha,
            "n_significant": len(significant),
            "n_total": n,
        }


class EffectSize:
    """Effect size measures for interpretability comparisons.

    Statistical significance is not enough: with enough data,
    tiny differences become "significant". Effect size tells
    you HOW BIG the difference is.

    Conventions (Cohen, 1988):
    - d = 0.2: small
    - d = 0.5: medium
    - d = 0.8: large

    For Oral: need LARGE effect sizes (d > 0.8) on key metrics.
    """

    @staticmethod
    def cohens_d(group_a, group_b):
        """Cohen's d: standardized mean difference."""
        if isinstance(group_a, torch.Tensor):
            group_a = group_a.cpu().numpy()
        if isinstance(group_b, torch.Tensor):
            group_b = group_b.cpu().numpy()

        a = np.array(group_a, dtype=float)
        b = np.array(group_b, dtype=float)

        if len(a) < 2 or len(b) < 2:
            return 0.0

        pooled_std = math.sqrt((a.var() + b.var()) / 2)
        if pooled_std < 1e-10:
            return 0.0
        return float((a.mean() - b.mean()) / pooled_std)

    @staticmethod
    def cliffs_delta(group_a, group_b):
        """Cliff's delta: non-parametric effect size.

        More robust than Cohen's d for non-normal distributions.
        Delta = 2 * (P(a > b) - P(a < b)) / (na * nb)

        Range: [-1, 1]. 0 = no difference, ±1 = complete separation.
        """
        if isinstance(group_a, torch.Tensor):
            group_a = group_a.cpu().numpy()
        if isinstance(group_b, torch.Tensor):
            group_b = group_b.cpu().numpy()

        a = np.array(group_a, dtype=float)
        b = np.array(group_b, dtype=float)

        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return 0.0

        # Count pairs
        more = sum(1 for ai in a for bj in b if ai > bj)
        less = sum(1 for ai in a for bj in b if ai < bj)

        delta = 2.0 * (more - less) / (na * nb)
        return float(delta)


class BayesianComparison:
    """Bayesian comparison: P(JEPA > baseline | data).

    More informative than p-values. Instead of "we reject H0 at p=0.03",
    you say "there is a 97% probability that JEPA is better than baseline."

    Uses bootstrap distribution of the difference.
    """

    @staticmethod
    def probability_a_greater_b(values_a, values_b, n_bootstrap=10000, seed=42):
        """P(A > B) based on bootstrap of means.

        Args:
            values_a: (N,) metric values for model A (JEPA)
            values_b: (N,) metric values for model B (baseline)
            n_bootstrap: number of bootstrap samples
            seed: random seed

        Returns:
            dict with probability and credible interval
        """
        rng = np.random.RandomState(seed)

        if isinstance(values_a, torch.Tensor):
            values_a = values_a.cpu().numpy()
        if isinstance(values_b, torch.Tensor):
            values_b = values_b.cpu().numpy()

        a = np.array(values_a, dtype=float)
        b = np.array(values_b, dtype=float)

        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return {"prob_a_greater": 0.5, "credible_lower": 0, "credible_upper": 0}

        boot_diffs = []
        for _ in range(n_bootstrap):
            idx_a = rng.randint(0, na, size=na)
            idx_b = rng.randint(0, nb, size=nb)
            diff = a[idx_a].mean() - b[idx_b].mean()
            boot_diffs.append(diff)

        diffs = np.array(boot_diffs)
        prob = (diffs > 0).mean()

        return {
            "prob_a_greater_b": float(prob),
            "prob_b_greater_a": float(1 - prob),
            "mean_diff": float(diffs.mean()),
            "credible_lower": float(np.percentile(diffs, 2.5)),
            "credible_upper": float(np.percentile(diffs, 97.5)),
            "n_bootstrap": n_bootstrap,
            "decision": "A > B" if prob > 0.95 else ("B > A" if prob < 0.05 else "Inconclusive"),
        }


class MetricComparisonReport:
    """Full statistical report comparing JEPA vs baseline across all metrics.

    Produces a publication-ready table with:
    - Metric name
    - JEPA value (mean ± CI)
    - Baseline value (mean ± CI)
    - Effect size (Cohen's d)
    - p-value (paired permutation)
    - P(JEPA > baseline)
    - BH-corrected significance
    """

    @staticmethod
    def generate(
        metric_names,
        jepa_values_dict,
        baseline_values_dict,
        n_bootstrap=5000,
        n_permutations=10000,
        alpha=0.05,
    ):
        """Generate full statistical comparison report.

        Args:
            metric_names: list of metric names to compare
            jepa_values_dict: {metric_name: (N,)} per-sample metric values
            baseline_values_dict: {metric_name: (N,)} per-sample metric values
            n_bootstrap: bootstrap samples
            n_permutations: permutation test samples
            alpha: significance level

        Returns:
            dict with per-metric statistics and corrected significance
        """
        results = {}
        raw_p_values = []

        for name in metric_names:
            jepa_vals = jepa_values_dict.get(name)
            base_vals = baseline_values_dict.get(name)

            if jepa_vals is None or base_vals is None:
                continue

            # Bootstrap CIs
            jepa_ci = BootstrapCI.compute(
                lambda d: d.mean() if isinstance(d, torch.Tensor) else np.mean(d),
                jepa_vals if isinstance(jepa_vals, torch.Tensor) else torch.tensor(jepa_vals),
                n_bootstrap=min(n_bootstrap, 2000),
            )
            base_ci = BootstrapCI.compute(
                lambda d: d.mean() if isinstance(d, torch.Tensor) else np.mean(d),
                base_vals if isinstance(base_vals, torch.Tensor) else torch.tensor(base_vals),
                n_bootstrap=min(n_bootstrap, 2000),
            )

            # Effect size
            d = EffectSize.cohens_d(jepa_vals, base_vals)

            # Paired permutation test
            perm = PairedPermutationTest.compute(
                jepa_vals, base_vals, n_permutations=min(n_permutations, 5000)
            )

            # Bayesian
            bayes = BayesianComparison.probability_a_greater_b(
                jepa_vals, base_vals, n_bootstrap=min(n_bootstrap, 2000)
            )

            results[name] = {
                "jepa_mean": jepa_ci["mean"],
                "jepa_ci": (jepa_ci["ci_lower"], jepa_ci["ci_upper"]),
                "baseline_mean": base_ci["mean"],
                "baseline_ci": (base_ci["ci_lower"], base_ci["ci_upper"]),
                "effect_size_d": d,
                "effect_size_label": (
                    "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")
                ),
                "p_value": perm["p_value"],
                "prob_jepa_better": bayes["prob_a_greater_b"],
                "mean_diff": jepa_ci["mean"] - base_ci["mean"],
            }

            raw_p_values.append(perm["p_value"])

        # Multiple comparison correction
        if raw_p_values:
            bh = MultipleComparisonCorrection.benjamini_hochberg(raw_p_values, alpha)
            bonf = MultipleComparisonCorrection.bonferroni(raw_p_values)

            metric_list = list(results.keys())
            for i, name in enumerate(metric_list):
                results[name]["p_value_bh"] = bh["corrected"][i]
                results[name]["p_value_bonferroni"] = bonf[i]
                results[name]["significant_bh"] = name in [
                    metric_list[j] for j in bh["significant"]
                ]
                results[name]["significant_bonferroni"] = bonf[i] < alpha

        # Summary
        n_significant_bh = sum(1 for v in results.values() if v.get("significant_bh", False))
        n_large_effect = sum(1 for v in results.values() if v.get("effect_size_label") == "large")

        results["_summary"] = {
            "n_metrics": len(results),  # dict literal evaluates before _summary is inserted
            "n_significant_bh": n_significant_bh,
            "n_large_effect": n_large_effect,
            "alpha": alpha,
            "jepa_wins": sum(
                1
                for v in results.values()
                if isinstance(v, dict) and v.get("prob_jepa_better", 0) > 0.95
            ),
        }

        return results
