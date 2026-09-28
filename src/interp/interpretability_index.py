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
#
# ─────────────────────────────────────────────────────────────────────────
# AUDIT FINDINGS THAT SHAPED THIS FILE (do not undo them silently)
# ─────────────────────────────────────────────────────────────────────────
# 1. **The weights and sigmoid midpoints are NOT derived from anything.** They
#    are hand-set constants. They have not been validated against any
#    ground truth. Do not present the index as a calibrated measure.
# 2. **The index was maximisable by degradation.** Adding isotropic white noise
#    to an 8-dim structured representation RAISED the index (0.39 -> 0.64),
#    because the weight-3 `effective_dimension` and the weight-2 rank metrics
#    all saturate at the isotropic-noise value — noise fills every direction,
#    which is the maximum those metrics can take. Measured: the index of pure
#    white noise (0.6424) is HIGHER than the index of a clean 2-factor
#    representation (0.3920). The metric set cannot distinguish the two.
#    `compute()` therefore refuses to emit a headline for any representation
#    that does not beat the white-noise floor of the same shape. See
#    :func:`noise_baseline_metrics` and the `unreliable` flag.
# 3. **The denominator was whatever happened to be present.** `compute()`
#    averaged over the metrics it found, so "index" meant a different thing for
#    every call. `compare()` then computed two independent averages over
#    different denominators and reported the difference as `index_gap`, which
#    is meaningless. `compare()` now requires the same metric set on both sides.
# 4. **An empty metric dict scored 0.5** — the neutral midpoint — for having
#    measured nothing. `compute()` now raises.
# 5. Metric DIRECTION lives only in `METRIC_DEFINITIONS`. `visualization.py`
#    used to keep a second, divergent copy; it now imports from here.
#
# WHAT THIS FILE STILL CANNOT DO
# ──────────────────────────────
# It cannot make the index *accurate* — only *not-wrong-at-face-value*. The
# premise "higher effective dimension is better" is exactly the premise that
# noise maximises. Re-deriving the directions and midpoints from data is a
# human decision and is out of scope. Until then, read `unreliable_reasons`.

from __future__ import annotations

import math

import torch

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

#: Metric family that the white-noise baseline can be computed for, i.e. the
#: metrics that are functions of a representation matrix alone. Everything else
#: (probes, polysemanticity, composition, training dynamics) needs labels, a
#: trained model, or a second run, so the degeneracy guard is scoped to these.
NOISE_COMPARABLE = (
    "effective_dimension",
    "anisotropy",
    "total_compression",
    "effective_rank",
    "participation_ratio",
    "sv_entropy",
    "collapsed_dim_ratio",
    "intrinsic_dim",
    "mean_pairwise_cosine",
    "uniformity",
)

#: Collapse thresholds that make a representation degenerate on their own,
#: independent of any baseline comparison.
COLLAPSE_RATIO_MAX = 0.9
COLLAPSE_COSINE_MAX = 0.95
COLLAPSE_RANK_MAX = 1.5
COLLAPSE_ENTROPY_MAX = 0.15

#: A metric counts as **noise-saturated** when the representation's value is
#: within this relative distance of what isotropic noise achieves on it:
#: ``|value - noise| <= NOISE_SATURATION_TOL * max(|noise|, 1)``. DECLARED, not
#: derived — there is no published threshold for this, and inventing one
#: silently would be the same mistake as the sigmoid midpoints. It is exposed
#: as a module constant so a reader can see it and change it deliberately.
NOISE_SATURATION_TOL = 0.25

#: Refuse the headline when the noise-saturated metrics carry at least this
#: fraction of the measured weight. DECLARED, not derived — see above.
SATURATED_WEIGHT_MAX = 0.5


def _as_float(value, name: str) -> float:
    """Coerce a metric to float, accepting 0-dim tensors, or explain why not."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"metric {name!r} is not numeric: {value!r}")


def geometry_metrics(representations) -> dict:
    """Compute the noise-comparable geometry metrics for a representation.

    Thin, dependency-light wrapper over the repo's own implementations
    (``src.models.collapse.CollapseDiagnostics`` and
    ``src.interp.representation_geometry.RepresentationGeometry``) so that a
    candidate representation and its white-noise control are always measured
    with the same code.

    Args:
        representations: (N, D) or (B, T, D) tensor.

    Returns:
        {metric_name: float} keyed by METRIC_DEFINITIONS names.

    """
    from src.interp.representation_geometry import RepresentationGeometry
    from src.models.collapse import CollapseDiagnostics

    diag = CollapseDiagnostics()
    m = diag.compute(representations, representations)
    out = InterpretabilityIndex.from_collapse_diagnostics(m)
    out["effective_dimension"] = RepresentationGeometry.effective_dimension(representations)
    out["total_compression"] = RepresentationGeometry.total_compression(representations)
    out["anisotropy"] = RepresentationGeometry.anisotropy(representations)
    return out


def noise_baseline_metrics(
    n_samples: int = 200,
    n_features: int = 64,
    n_draws: int = 8,
    seed: int = 0,
) -> dict:
    """White-noise control for the index: what pure isotropic noise scores.

    Draws ``n_draws`` independent N x D standard-normal matrices, measures the
    same :func:`geometry_metrics` on each, and reports the mean and spread. The
    resulting index is the **floor** a real representation has to beat.

    Measured on an 8-dim / 200-sample control: white noise scores **0.6424**
    while a clean 2-factor representation scores **0.3920**. The index as
    specified prefers noise, which is why :meth:`InterpretabilityIndex.compute`
    will not quote a headline for a representation at or below this floor.

    Args:
        n_samples: N, the number of samples. CollapseDiagnostics subsamples
            internally above 256, so values above that add cost, not fidelity.
        n_features: D, the ambient representation dimension.
        n_draws: how many noise draws to average. >= 3 is needed for a spread.
        seed: base seed. Global torch RNG state is saved and restored, so this
            is deterministic and does not perturb the caller's stream.

    Returns:
        dict with ``metrics``, ``metrics_std``, ``index``, ``index_std``,
        ``n_samples``, ``n_features``, ``n_draws``, ``seed``.

    """
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    if n_samples < 2 or n_features < 2:
        raise ValueError(
            f"noise baseline needs n_samples, n_features >= 2, got {n_samples}, {n_features}"
        )

    idx = InterpretabilityIndex()
    draws = []
    state = torch.random.get_rng_state()
    try:
        for i in range(n_draws):
            torch.manual_seed(int(seed) + 7919 * i)
            draws.append(geometry_metrics(torch.randn(int(n_samples), int(n_features))))
    finally:
        torch.random.set_rng_state(state)

    keys = sorted(draws[0])
    metrics = {k: float(sum(d[k] for d in draws) / n_draws) for k in keys}
    metrics_std = {}
    for k in keys:
        vals = torch.tensor([d[k] for d in draws], dtype=torch.float64)
        metrics_std[k] = float(vals.std().item()) if n_draws > 1 else 0.0

    # The noise index must be averaged over the same denominator the candidate
    # will use, otherwise we would be committing the very bug this file exists
    # to fix. Averaging per-metric first, then scoring once, is the only way to
    # do that without an extra scoring pass per draw.
    result = {
        "metrics": metrics,
        "metrics_std": metrics_std,
        "index": idx.compute(metrics)["interpretability_index"],
        "index_std": (
            float(
                torch.tensor(
                    [idx.compute(d)["interpretability_index"] for d in draws], dtype=torch.float64
                )
                .std()
                .item()
            )
            if n_draws > 1
            else 0.0
        ),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_draws": n_draws,
        "seed": int(seed),
    }
    return result


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

    **Read ``unreliable_reasons`` before quoting the number.** A degenerate
    representation (collapsed, or no better than white noise) yields
    ``interpretability_index = None`` and must not be given a headline value.
    See the module docstring for the measurements behind that.
    """

    def __init__(self, custom_weights: dict | None = None):
        """
        Args:
            custom_weights: override default weights {metric_name: weight}

        """
        self.weights = {}
        for name, defn in METRIC_DEFINITIONS.items():
            self.weights[name] = defn["weight"]
        if custom_weights:
            self.weights.update(custom_weights)

    # ─────────────────────────────────────────────────────────────
    # Core
    # ─────────────────────────────────────────────────────────────

    def compute(
        self,
        metrics: dict,
        dim: int | None = None,
        n_samples: int | None = None,
        noise_baseline: dict | None = None,
        require_complete: bool = False,
        noise_tol_mult: float = 3.0,
    ) -> dict:
        """Compute Interpretability Index from raw metric values.

        Args:
            metrics: {metric_name: value} — raw metric values. Must be non-empty
                and finite; a total failure to measure is not a neutral score.
            dim: ambient representation dimension D. If given (with
                ``n_samples``, or with ``noise_baseline``) the white-noise floor
                is computed automatically and the degeneracy guard runs.
            n_samples: number of samples N behind ``metrics``. Defaults to the
                metric count when omitted, which makes the floor indicative
                rather than exact; pass the real N.
            noise_baseline: precomputed :func:`noise_baseline_metrics` result.
                Takes precedence over ``dim``/``n_samples``.
            require_complete: raise if any known metric is missing. Off by
                default so that partial dashboards keep working, but the
                production/paper path should turn it on — see finding (3) in
                the module docstring.
            noise_tol_mult: how many noise-draw standard deviations of slack
                the white-noise comparison gets before it is called a tie.

        Returns:
            dict with ``interpretability_index`` (float, or **None** when the
            representation is degenerate and no headline may be quoted),
            ``interpretability_index_raw`` (always the weighted average, for
            diagnostics), ``reliable``, ``unreliable_reasons``,
            ``noise_baseline``, per-component breakdown, and missing metrics.

        Raises:
            ValueError: on an empty metric dict, a non-finite metric, an
                unknown metric under ``require_complete``, or non-positive N/D.

        """
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics must be a dict, got {type(metrics).__name__}")
        if not metrics:
            raise ValueError(
                "InterpretabilityIndex.compute() got an empty metric dict. "
                "Measuring nothing is not a score of 0.5; compute the metrics "
                "first, or use InterpretabilityIndex.from_collapse_diagnostics()."
            )

        index, component_scores, missing, total_weight = self._weighted_average(metrics)
        unknown = sorted(k for k in metrics if k not in self.weights)
        if require_complete and missing:
            raise ValueError(
                f"require_complete=True but {len(missing)} of {len(self.weights)} "
                f"known metrics are missing: {missing}"
            )

        result = {
            "interpretability_index": index,
            "interpretability_index_raw": index,
            "reliable": True,
            "unreliable": False,
            "unreliable_reasons": [],
            "degenerate": False,
            "n_metrics_used": len(component_scores),
            "n_metrics_missing": len(missing),
            "missing_metrics": missing,
            "unknown_metrics": unknown,
            "components": component_scores,
            "total_weight": total_weight,
            "noise_baseline": None,
        }

        reasons = result["unreliable_reasons"]

        # ── advisory: the denominator is not the designed one ──
        if missing:
            reasons.append(
                f"index averaged over {len(component_scores)} of {len(self.weights)} known "
                f"metrics; a partial average is not comparable to a complete one, and it is "
                f"maximisable by reporting only the metrics that flatter you. Missing: "
                f"{missing}. Pass require_complete=True to make this an error."
            )
        if unknown:
            reasons.append(
                f"supplied metrics not in METRIC_DEFINITIONS were silently dropped and did "
                f"not affect the index: {unknown}"
            )

        # ── refusal 1: representation is collapsed on its own terms ──
        collapse = self._collapse_reasons(metrics)
        if collapse:
            result["degenerate"] = True
            reasons.extend(collapse)

        # ── refusal 2: no better than white noise of the same shape ──
        baseline = noise_baseline
        if baseline is None and dim is not None:
            baseline = noise_baseline_metrics(
                n_samples=int(n_samples if n_samples is not None else max(len(metrics), 8)),
                n_features=int(dim),
            )
        if baseline is None:
            reasons.append(
                "no white-noise baseline was supplied (pass dim= and n_samples=, or "
                "noise_baseline=), so collapse-by-noise could not be excluded. "
                "This index IS maximised by isotropic white noise without that check."
            )
        else:
            nb = self._compare_to_noise(metrics, index, baseline, noise_tol_mult)
            result["noise_baseline"] = nb
            if nb["saturated_weight_fraction"] >= SATURATED_WEIGHT_MAX:
                result["degenerate"] = True
                reasons.append(nb["reason"])
            if (
                nb["at_or_below_noise_floor"]
                and nb["saturated_weight_fraction"] < SATURATED_WEIGHT_MAX
            ):
                reasons.append(nb["reason"])
            if not nb["covered_metrics"]:
                reasons.append(
                    "white-noise comparison covered none of the supplied metrics, so the "
                    "degeneracy guard was vacuous for this call."
                )
            elif nb["excluded_metrics"]:
                reasons.append(
                    f"the white-noise control could not be built for {nb['excluded_metrics']}, "
                    f"so the guard covers {len(nb['covered_metrics'])} of "
                    f"{len(nb['covered_metrics']) + len(nb['excluded_metrics'])} measured "
                    f"metrics. Probes, polysemanticity and training dynamics need a model or a "
                    f"second run, so they are unguarded."
                )

        if reasons:
            result["reliable"] = not result["degenerate"]
            result["unreliable"] = True
            if result["degenerate"]:
                # Positive evidence of degeneration: refuse the headline outright
                # rather than publish a number that noise improves on.
                result["interpretability_index"] = None

        return result

    def compare(
        self,
        jepa_metrics: dict,
        baseline_metrics: dict,
        dim: int | None = None,
        n_samples: int | None = None,
        noise_baseline: dict | None = None,
        require_complete: bool = False,
        strict_metric_set: bool = True,
        noise_tol_mult: float = 3.0,
    ) -> dict:
        """Compare Interpretability Index between JEPA and baseline.

        THE KEY RESULT for the paper.

        Like-for-like only. The two indices are weighted averages, and a
        weighted average over a different set of metrics is a different
        quantity: comparing them measures the metric sets, not the models. So
        both sides must carry the **same** metric keys (and, by extension, the
        same weight denominator), or this raises.

        A per-component ``jepa_wins`` count and a headline ``jepa_better``
        boolean were previously allowed to disagree, because the aggregate was
        taken over different denominators than the components. They cannot
        disagree now.

        Args:
            jepa_metrics: {metric_name: value} for the JEPA model.
            baseline_metrics: {metric_name: value} for the baseline. Must have
                exactly the same keys as ``jepa_metrics``.
            dim: ambient representation dimension, forwarded to ``compute``.
            n_samples: sample count N, forwarded to ``compute``.
            noise_baseline: precomputed :func:`noise_baseline_metrics` result.
            require_complete: forwarded to ``compute`` (both sides).
            strict_metric_set: raise on differing key sets. Set False only to
                get a diagnostic report; ``comparison_defined`` will be False.
            noise_tol_mult: forwarded to ``compute``.

        Returns:
            dict with ``jepa_index``, ``baseline_index``, ``index_gap``,
            ``jepa_better`` (None if either side is degenerate),
            ``jepa_wins_n_out_of``, ``comparison_defined``,
            ``comparison_reasons`` and the per-component breakdown.

        Raises:
            ValueError: on differing metric key sets (when strict), or on
                anything ``compute`` raises.

        """
        if not isinstance(jepa_metrics, dict) or not isinstance(baseline_metrics, dict):
            raise TypeError("compare() expects two dicts of metrics")

        set_reasons = []
        jepa_keys = set(jepa_metrics)
        base_keys = set(baseline_metrics)
        if jepa_keys != base_keys:
            only_jepa = sorted(jepa_keys - base_keys)
            only_base = sorted(base_keys - jepa_keys)
            message = (
                "compare() requires the same metric set on both sides. A weighted average "
                "over different denominators is not a comparison: the resulting index_gap "
                "measures the metric sets, not the models. "
                f"Only in jepa: {only_jepa}; only in baseline: {only_base}."
            )
            if strict_metric_set:
                raise ValueError(message)
            set_reasons.append(message)

        jepa_result = self.compute(
            jepa_metrics,
            dim=dim,
            n_samples=n_samples,
            noise_baseline=noise_baseline,
            require_complete=require_complete,
            noise_tol_mult=noise_tol_mult,
        )
        baseline_result = self.compute(
            baseline_metrics,
            dim=dim,
            n_samples=n_samples,
            noise_baseline=noise_baseline,
            require_complete=require_complete,
            noise_tol_mult=noise_tol_mult,
        )

        # Per-component comparison
        component_comparison = {}
        all_keys = sorted(set(jepa_result["components"]) | set(baseline_result["components"]))
        for name in all_keys:
            j = jepa_result["components"].get(name, {})
            b = baseline_result["components"].get(name, {})
            jn = j.get("normalized")
            bn = b.get("normalized")
            component_comparison[name] = {
                "jepa_raw": j.get("raw", None),
                "baseline_raw": b.get("raw", None),
                "jepa_norm": jn,
                "baseline_norm": bn,
                "jepa_advantage": None if jn is None or bn is None else jn - bn,
                "jepa_wins": None if jn is None or bn is None else jn > bn,
                "comparable": jn is not None and bn is not None,
            }

        comparable = {k: v for k, v in component_comparison.items() if v["comparable"]}
        jepa_wins = sum(1 for v in comparable.values() if v["jepa_wins"])

        jepa_index = jepa_result["interpretability_index"]
        baseline_index = baseline_result["interpretability_index"]
        degenerate = jepa_index is None or baseline_index is None

        comparison_reasons = list(set_reasons)
        for label, res in (("jepa", jepa_result), ("baseline", baseline_result)):
            for r in res["unreliable_reasons"]:
                comparison_reasons.append(f"[{label}] {r}")
        if degenerate:
            comparison_reasons.append(
                "at least one side is degenerate, so no headline comparison is defined; "
                "jepa_better is None rather than a guess."
            )

        return {
            "jepa_index": jepa_index,
            "baseline_index": baseline_index,
            "index_gap": (
                None
                if degenerate
                else jepa_result["interpretability_index_raw"]
                - baseline_result["interpretability_index_raw"]
            ),
            "jepa_better": None if degenerate else jepa_index > baseline_index,
            "jepa_wins_n_out_of": f"{jepa_wins}/{len(comparable)}",
            "jepa_wins_fraction": jepa_wins / len(comparable) if comparable else 0.0,
            "n_metrics_compared": len(comparable),
            "comparison_defined": not comparison_reasons,
            "comparison_reasons": comparison_reasons,
            "component_comparison": component_comparison,
            "jepa_details": jepa_result,
            "baseline_details": baseline_result,
        }

    # ─────────────────────────────────────────────────────────────
    # Guards
    # ─────────────────────────────────────────────────────────────

    def _weighted_average(self, metrics: dict):
        """Normalize, weight and average. The only place the index is formed.

        Returns (index, component_scores, missing, total_weight). Split out of
        ``compute`` so that the white-noise floor is built by exactly the same
        arithmetic as the candidate, with no nested guard evaluation.
        """
        component_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        missing = []
        non_finite = []

        for name, value in metrics.items():
            if name not in self.weights:
                continue
            value = _as_float(value, name)
            if not math.isfinite(value):
                non_finite.append(name)

        if non_finite:
            raise ValueError(
                "non-finite metric values cannot be scored: "
                + ", ".join(f"{n}={metrics[n]!r}" for n in sorted(non_finite))
            )

        for name, weight in self.weights.items():
            if name not in metrics:
                missing.append(name)
                continue

            value = _as_float(metrics[name], name)
            defn = METRIC_DEFINITIONS.get(name, {"direction": "higher"})

            # Normalize to [0, 1]
            norm_value = self._normalize(name, value, defn["direction"])

            # Weighted contribution
            component_scores[name] = {
                "raw": value,
                "normalized": norm_value,
                "weight": weight,
                "contribution": norm_value * weight,
                "direction": defn["direction"],
                "source": defn.get("source", ""),
            }
            weighted_sum += norm_value * weight
            total_weight += weight

        if total_weight <= 0:
            raise ValueError(
                "none of the supplied metrics are known to METRIC_DEFINITIONS, so "
                f"there is nothing to weight. Unknown keys: "
                f"{sorted(k for k in metrics if k not in self.weights)}"
            )

        return weighted_sum / total_weight, component_scores, missing, total_weight

    @staticmethod
    def _collapse_reasons(metrics: dict) -> list:
        """Degeneracy detectable from the metric values alone, no baseline needed."""
        reasons = []
        cdr = metrics.get("collapsed_dim_ratio")
        if cdr is not None and cdr >= COLLAPSE_RATIO_MAX:
            reasons.append(
                f"collapsed_dim_ratio={cdr:.4g} >= {COLLAPSE_RATIO_MAX}: at least "
                f"{cdr:.1%} of the representation's directions carry no variance, so the "
                f"index is scoring a lower-rank object than it claims to."
            )
        mpc = metrics.get("mean_pairwise_cosine")
        if mpc is not None and mpc >= COLLAPSE_COSINE_MAX:
            reasons.append(
                f"mean_pairwise_cosine={mpc:.4g} >= {COLLAPSE_COSINE_MAX}: every sample "
                f"points the same way. This is collapse, not interpretability."
            )
        er = metrics.get("effective_rank")
        se = metrics.get("sv_entropy")
        if (
            er is not None
            and se is not None
            and er <= COLLAPSE_RANK_MAX
            and se <= COLLAPSE_ENTROPY_MAX
        ):
            reasons.append(
                f"effective_rank={er:.4g} and sv_entropy={se:.4g} are both at their floor: "
                f"the representation is effectively one-dimensional."
            )
        return reasons

    def _compare_to_noise(
        self, metrics: dict, index: float, baseline: dict, noise_tol_mult: float
    ) -> dict:
        """Compare a score against the white-noise floor on a matched denominator.

        The floor is recomputed over exactly the metrics the caller supplied, so
        the two sides share a denominator. Metrics the baseline cannot supply
        (probes, polysemanticity, training dynamics) are excluded from the
        comparison and reported, rather than silently dropped.

        Two distinct facts are established here, and they are kept apart on
        purpose:

        * **Saturation** (refusal). A metric is *noise-saturated* when the
          representation's value is within ``NOISE_SATURATION_TOL`` of what
          isotropic noise achieves on it. This is decisive, because the
          direction table says noise already scores at or near the best
          achievable value: the weight-3 ``effective_dimension`` saturates at
          0.82 for eff_dim >= 60 and hits 1.0 at 768, and 768 is exactly the
          value pure noise produces. A saturated metric is a noise detector,
          not an interpretability measure. When the saturated metrics carry at
          least ``SATURATED_WEIGHT_MAX`` of the weight, the index is not
          measuring interpretability and the headline is refused.
        * **Excess** (advisory). ``index - noise_index``. Frequently negative
          for a perfectly healthy representation, which is the real finding
          about the index: as specified, it prefers noise. That is reported
          loudly but is not by itself grounds for refusing, because the metrics
          do still separate the two representations even though the index does
          not order them correctly.
        """
        base_metrics = baseline.get("metrics") or {}
        covered = sorted(k for k in metrics if k in base_metrics and k in self.weights)
        excluded = sorted(k for k in metrics if k in self.weights and k not in base_metrics)

        if not covered:
            return {
                "at_or_below_noise_floor": False,
                "saturated_weight_fraction": 0.0,
                "saturated_metrics": [],
                "covered_metrics": [],
                "excluded_metrics": excluded,
                "reason": "",
                "noise_index": None,
                "noise_index_std": float(baseline.get("index_std", 0.0) or 0.0),
                "excess_over_noise": None,
            }

        saturated = []
        detail = {}
        saturated_weight = 0.0
        covered_weight = 0.0
        for name in covered:
            weight = self.weights[name]
            covered_weight += weight
            noise_value = float(base_metrics[name])
            # Reference magnitude: the noise value itself, floored at 1 so that
            # metrics living on a bounded scale (sv_entropy, anisotropy) are
            # compared on their own scale rather than on a value near zero.
            ref = max(abs(noise_value), 1.0)
            delta = abs(float(metrics[name]) - noise_value)
            is_sat = delta <= NOISE_SATURATION_TOL * ref
            detail[name] = {
                "value": float(metrics[name]),
                "noise_value": noise_value,
                "abs_delta": delta,
                "tolerance": NOISE_SATURATION_TOL * ref,
                "noise_saturated": is_sat,
                "weight": weight,
            }
            if is_sat:
                saturated.append(name)
                saturated_weight += weight

        fraction = saturated_weight / covered_weight if covered_weight else 0.0
        matched_baseline = {k: base_metrics[k] for k in covered}
        noise_index_raw, _, _, _ = self._weighted_average(matched_baseline)
        noise_std = float(baseline.get("index_std", 0.0) or 0.0)
        threshold = noise_index_raw + noise_tol_mult * noise_std
        excess = index - noise_index_raw
        at_floor = index <= threshold
        saturated_enough = fraction >= SATURATED_WEIGHT_MAX

        reason = ""
        if saturated_enough:
            reason = (
                f"{saturated_weight:.0f} of {covered_weight:.0f} weight "
                f"({fraction:.0%} >= {SATURATED_WEIGHT_MAX:.0%}) sits on metrics whose values "
                f"are indistinguishable from what isotropic noise produces on a matrix of the "
                f"same shape (N={baseline.get('n_samples')}, D={baseline.get('n_features')}), "
                f"measured with the same code on the same {len(covered)} metrics. Saturated: "
                f"{saturated}. Per the direction table noise already scores at the best "
                f"achievable value for these metrics, so they are detecting noise, not "
                f"interpretability, and this index is maximised by degradation. No headline "
                f"may be quoted. Metrics not covered by the noise control: {excluded}."
            )
        elif at_floor:
            reason = (
                f"index {index:.4f} is at or below the white-noise floor {noise_index_raw:.4f} "
                f"(+{noise_tol_mult:g} draws of spread = {threshold:.4f}) on the same "
                f"{len(covered)} metrics. The index as specified orders noise above a clean "
                f"structured representation; treat the ordering, not the level, as the defect."
            )

        return {
            "at_or_below_noise_floor": at_floor,
            "saturated_weight_fraction": fraction,
            "saturated_metrics": saturated,
            "per_metric": detail,
            "covered_metrics": covered,
            "excluded_metrics": excluded,
            "reason": reason,
            "noise_index": noise_index_raw,
            "noise_index_std": noise_std,
            "threshold": threshold,
            "excess_over_noise": excess,
        }

    # ─────────────────────────────────────────────────────────────
    # Normalization
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(name: str, value: float, direction: str) -> float:
        """Normalize a metric to [0, 1] where 1 = better.

        Uses sigmoid-based normalization that handles arbitrary ranges:
        - For 'higher is better': sigmoid(value - midpoint)
        - For 'lower is better': sigmoid(midpoint - value)

        Midpoints and scales are hand-set constants. They are NOT derived from
        any reference, they have NOT been validated, and they are known to be
        miscalibrated: the weight-3 `effective_dimension` midpoint of 30
        saturates for any effective dimension at or above 60 and reaches 1.0 at
        768, which is precisely the isotropic-noise value. Treat the normalized
        components as an ordering heuristic, not as measurements. Re-deriving
        them from data is a human decision — see the module docstring.
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

    # ─────────────────────────────────────────────────────────────
    # Adapters
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def from_collapse_diagnostics(
        collapse_metrics: dict,
        extra_metrics: dict | None = None,
    ) -> dict:
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
