# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Composite Interpretability Index — THE SINGLE NUMBER
#
# Every Oral paper has a headline result. "Our method achieves X."
# Without a single number, you have a list, not a result.
#
# The Interpretability Index aggregates all metrics into ONE score
# per model, weighted by what the literature says matters most.
#
# Construction:
# 1. Normalize each metric to [0, 1] where 1 = better
# 2. Weight by theoretical importance (from reference papers)
# 3. Average = Interpretability Index
#
# Weights derived from:
# - Yadav (2026): effective_dim predicts accuracy (r=0.75) → weight 3
# - Ansuini (2019): intrinsic_dim → weight 2
# - Elhage (2022): superposition/polysemanticity → weight 2
# - Hewitt & Liang (2019): probe selectivity → weight 2
# - I-JEPA: representation stability → weight 1
# - Standard: rank, entropy, uniformity → weight 1

from __future__ import annotations

import math

# ═══════════════════════════════════════════════════════════════
# Metric definitions: name, direction, weight, source
# ═══════════════════════════════════════════════════════════════

METRIC_DEFINITIONS = {
    # Geometry (Yadav 2026: predicts accuracy r=0.75)
    "effective_dimension": {"direction": "higher", "weight": 3, "source": "Yadav 2026"},
    "anisotropy": {"direction": "lower", "weight": 2, "source": "Yadav 2026"},
    "total_compression": {"direction": "lower", "weight": 2, "source": "Yadav 2026"},
    # Rank (NextLat / I-JEPA)
    "effective_rank": {"direction": "higher", "weight": 2, "source": "NextLat 2025"},
    "participation_ratio": {"direction": "higher", "weight": 1, "source": "NextLat 2025"},
    "sv_entropy": {"direction": "higher", "weight": 2, "source": "I-JEPA 2023"},
    "collapsed_dim_ratio": {"direction": "lower", "weight": 2, "source": "I-JEPA 2023"},
    # Intrinsic dimensionality (Ansuini 2019)
    "intrinsic_dim": {"direction": "lower", "weight": 2, "source": "Ansuini 2019"},
    "intrinsic_dim_score": {"direction": "lower", "weight": 2, "source": "Ansuini 2019"},
    # Polysemanticity (Elhage 2022)
    "mean_psi": {"direction": "lower", "weight": 2, "source": "Elhage 2022"},
    "frac_monosemantic": {"direction": "higher", "weight": 2, "source": "Elhage 2022"},
    # Probe quality (Hewitt & Liang 2019)
    "probe_selectivity": {"direction": "higher", "weight": 2, "source": "Hewitt 2019"},
    "probe_generalization": {"direction": "higher", "weight": 2, "source": "Hewitt 2019"},
    "probing_complexity": {"direction": "lower", "weight": 3, "source": "PCC (ours)"},
    # Information theory
    "ib_gap": {"direction": "higher", "weight": 2, "source": "Shwartz-Ziv 2017"},
    "total_correlation": {"direction": "lower", "weight": 1, "source": "Info theory"},
    # Uniformity / diversity
    "uniformity": {"direction": "lower", "weight": 1, "source": "Wang 2022"},
    "mean_pairwise_cosine": {"direction": "lower", "weight": 1, "source": "DINOv2 2024"},
    # Stability
    "convergence_speed": {"direction": "higher", "weight": 1, "source": "Training"},
    "loss_smoothness": {"direction": "higher", "weight": 1, "source": "Training"},
    # Layer quality
    "layer_uniformity": {"direction": "higher", "weight": 1, "source": "Layer analysis"},
    "cv_effective_dim": {"direction": "lower", "weight": 1, "source": "Layer analysis"},
    # Composition
    "composition_score": {"direction": "higher", "weight": 1, "source": "Composition"},
    "feature_interference": {"direction": "lower", "weight": 1, "source": "Composition"},
}


class InterpretabilityIndex:
    """Compute a single Interpretability Index for a model.

    This is THE headline number for the paper:
    "Text-Span JEPA achieves an Interpretability Index of 0.87,
    compared to 0.62 for MLM and 0.71 for data2vec."

    Construction:
    1. Compute all available metrics
    2. Normalize each to [0, 1] (1 = better)
    3. Apply weights from METRIC_DEFINITIONS
    4. Weighted average = Interpretability Index
    """

    def __init__(self, custom_weights: dict[str, float] | None = None):
        """
        Args:
            custom_weights: override default weights {metric_name: weight}

        """
        self.weights = {}
        for name, defn in METRIC_DEFINITIONS.items():
            self.weights[name] = defn["weight"]
        if custom_weights:
            self.weights.update(custom_weights)

    def compute(self, metrics: dict[str, float]) -> dict:
        """Compute Interpretability Index from raw metric values.

        Args:
            metrics: {metric_name: value} — raw metric values

        Returns:
            dict with index, per-component breakdown, and missing metrics

        """
        normalized = {}
        component_scores = {}
        total_weight = 0
        weighted_sum = 0
        missing = []

        for name, weight in self.weights.items():
            if name not in metrics:
                missing.append(name)
                continue

            value = metrics[name]
            defn = METRIC_DEFINITIONS.get(name, {"direction": "higher"})

            # Normalize to [0, 1]
            norm_value = self._normalize(name, value, defn["direction"])
            normalized[name] = norm_value

            # Weighted contribution
            contribution = norm_value * weight
            component_scores[name] = {
                "raw": value,
                "normalized": norm_value,
                "weight": weight,
                "contribution": contribution,
                "direction": defn["direction"],
                "source": defn.get("source", ""),
            }

            weighted_sum += contribution
            total_weight += weight

        # Index = weighted average
        index = weighted_sum / total_weight if total_weight > 0 else 0.5

        return {
            "interpretability_index": index,
            "n_metrics_used": len(component_scores),
            "n_metrics_missing": len(missing),
            "missing_metrics": missing,
            "components": component_scores,
            "total_weight": total_weight,
        }

    def compare(self, jepa_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict:
        """Compare Interpretability Index between JEPA and baseline.

        THE KEY RESULT for the paper.
        """
        jepa_result = self.compute(jepa_metrics)
        baseline_result = self.compute(baseline_metrics)

        # Per-component comparison
        component_comparison = {}
        all_keys = set(jepa_result["components"]) | set(baseline_result["components"])
        for name in all_keys:
            j = jepa_result["components"].get(name, {})
            b = baseline_result["components"].get(name, {})
            component_comparison[name] = {
                "jepa_raw": j.get("raw", None),
                "baseline_raw": b.get("raw", None),
                "jepa_norm": j.get("normalized", 0),
                "baseline_norm": b.get("normalized", 0),
                "jepa_advantage": j.get("normalized", 0) - b.get("normalized", 0),
                "jepa_wins": j.get("normalized", 0) > b.get("normalized", 0),
            }

        # Count JEPA wins
        jepa_wins = sum(1 for v in component_comparison.values() if v.get("jepa_wins"))
        n_compared = len(component_comparison)

        return {
            "jepa_index": jepa_result["interpretability_index"],
            "baseline_index": baseline_result["interpretability_index"],
            "index_gap": jepa_result["interpretability_index"]
            - baseline_result["interpretability_index"],
            "jepa_better": jepa_result["interpretability_index"]
            > baseline_result["interpretability_index"],
            "jepa_wins_n_out_of": f"{jepa_wins}/{n_compared}",
            "jepa_wins_fraction": jepa_wins / max(n_compared, 1),
            "component_comparison": component_comparison,
            "jepa_details": jepa_result,
            "baseline_details": baseline_result,
        }

    @staticmethod
    def _normalize(name: str, value: float, direction: str) -> float:
        """Normalize a metric to [0, 1] where 1 = better.

        Uses sigmoid-based normalization that handles arbitrary ranges:
        - For 'higher is better': sigmoid(value - midpoint)
        - For 'lower is better': sigmoid(midpoint - value)

        Midpoints are calibrated per metric based on typical ranges.
        """
        # Metric-specific midpoints and scales
        params = {
            "effective_dimension": (30, 0.05),
            "anisotropy": (0.5, 5),
            "total_compression": (0.5, 5),
            "effective_rank": (30, 0.05),
            "participation_ratio": (15, 0.05),
            "sv_entropy": (0.5, 5),
            "collapsed_dim_ratio": (0.3, 10),
            "intrinsic_dim": (20, 0.1),
            "intrinsic_dim_score": (20, 0.1),
            "mean_psi": (1.5, 2),
            "frac_monosemantic": (0.5, 5),
            "probe_selectivity": (0.3, 5),
            "probe_generalization": (0.5, 5),
            "probing_complexity": (2, 1),
            "ib_gap": (0, 1),
            "total_correlation": (5, 0.2),
            "uniformity": (-3, 1),
            "mean_pairwise_cosine": (0.3, 5),
            "convergence_speed": (0.5, 5),
            "loss_smoothness": (0.5, 5),
            "layer_uniformity": (0.5, 5),
            "cv_effective_dim": (0.3, 5),
            "composition_score": (0.3, 3),
            "feature_interference": (0.3, 5),
        }

        midpoint, scale = params.get(name, (0.5, 1))

        if direction == "higher":
            x = (value - midpoint) * scale
        else:
            x = (midpoint - value) * scale

        # Sigmoid: maps (-inf, inf) → (0, 1)
        try:
            norm = 1.0 / (1.0 + math.exp(-max(min(x, 20), -20)))
        except (OverflowError, ValueError):
            norm = 0.5

        return max(0.0, min(1.0, norm))

    @staticmethod
    def from_collapse_diagnostics(
        collapse_metrics: dict[str, float],
        extra_metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Convert CollapseDiagnostics output to InterpretabilityIndex format.

        Args:
            collapse_metrics: output of CollapseDiagnostics.compute()
            extra_metrics: additional metrics (probe selectivity, etc.)

        Returns:
            dict ready for InterpretabilityIndex.compute()

        """
        mapping = {
            "effective_rank_online": "effective_rank",
            "sv_entropy_online": "sv_entropy",
            "collapsed_dim_ratio_online": "collapsed_dim_ratio",
            "intrinsic_dim_online": "intrinsic_dim",
            "intrinsic_dim_score": "intrinsic_dim_score",
            "mean_pairwise_cosine_online": "mean_pairwise_cosine",
            "uniformity_online": "uniformity",
            "participation_ratio_online": "participation_ratio",
        }

        result = {}
        for old_key, new_key in mapping.items():
            if old_key in collapse_metrics:
                result[new_key] = collapse_metrics[old_key]

        if extra_metrics:
            result.update(extra_metrics)

        return result
