# Copyright 2026 Text-Span-JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Disentanglement metrics for comparing JEPA vs baseline representations
# DCI: Eastwood & Williams (2018) "A Framework for the Quantitative Evaluation
#       of Disentangled Representations"
# MIG: Chen et al. (2018) "Isolating Sources of Disentanglement in VAEs"
# SAP: Kumar et al. (2018) "Variational Inference for Monte Carlo Objectives"
# Modularity: Ridgeway & Mozer (2018) "Learning Deep Disentangled Embeddings"

from __future__ import annotations

import math
import warnings

import torch


class DCIMetrics:
    """Disentanglement, Completeness, Informativeness (DCI).

    Eastwood & Williams (2018):
    - Disentanglement: each dimension captures at most one generative factor
    - Completeness: each generative factor is captured by at most one dimension
    - Informativeness: how well the representation predicts the factors

    Requires: representations + ground-truth factor labels

    Returned keys:
        disentanglement            in [0, 1], higher is better
        completeness               in [0, 1], higher is better
        informativeness            in [0, 1] (NaN if N <= D + 1), the full-predictor
                                  adjusted R^2, clamped
        informativeness_raw        unclamped mean adjusted R^2; may be slightly
                                  negative under the null, which makes the noise
                                  floor visible rather than hidden by clamping
        informativeness_estimator  name of the estimator, for provenance

    Callers that threshold on informativeness should calibrate against
    ``DCIMetrics.measure_noise_floor``, not against a fixed constant: the
    noise floor is a property of N, D and K, not a constant.
    """

    @staticmethod
    def compute(representations, factors):
        """Compute DCI metrics.

        Args:
            representations: (N, D) representation vectors
            factors: (N, K) ground-truth factor values (one-hot or continuous)

        Returns:
            dict with 'disentanglement', 'completeness', 'informativeness'

        """
        try:
            _N, D = representations.shape
            K = factors.shape[1] if factors.dim() > 1 else 1
            if factors.dim() == 1:
                factors = factors.unsqueeze(1)

            # Compute importance matrix R[i,j] = mutual information / correlation
            # between dimension i and factor j
            # Simplified: use absolute correlation as importance
            R = torch.zeros(D, K)
            for i in range(D):
                for j in range(K):
                    r = _pearson_correlation(representations[:, i], factors[:, j])
                    R[i, j] = abs(r)

            # Normalize each row to sum to 1 (probability distribution per dimension)
            row_sums = R.sum(dim=1, keepdim=True)
            P = R / (row_sums + 1e-10)

            # Disentanglement: per-dimension entropy of P
            disent_per_dim = torch.zeros(D)
            for i in range(D):
                if row_sums[i] > 1e-10:
                    p = P[i]
                    p = p[p > 1e-10]
                    disent_per_dim[i] = 1.0 + (p * torch.log(p + 1e-10)).sum() / math.log(K + 1e-10)
                else:
                    disent_per_dim[i] = 0.0

            # Weight by importance
            weights = row_sums.squeeze()
            weights = weights / (weights.sum() + 1e-10)
            disentanglement = (disent_per_dim * weights).sum().item()

            # Completeness: per-factor entropy of column-normalized R
            col_sums = R.sum(dim=0, keepdim=True)
            Q = R / (col_sums + 1e-10)
            compl_per_factor = torch.zeros(K)
            for j in range(K):
                if col_sums[0, j] > 1e-10:
                    q = Q[:, j]
                    q = q[q > 1e-10]
                    compl_per_factor[j] = 1.0 + (q * torch.log(q + 1e-10)).sum() / math.log(
                        D + 1e-10,
                    )
                else:
                    compl_per_factor[j] = 0.0

            weights_c = col_sums.squeeze()
            weights_c = weights_c / (weights_c.sum() + 1e-10)
            completeness = (compl_per_factor * weights_c).sum().item()

            # Informativeness: how well the FULL representation predicts each
            # factor. See _informativeness_adjusted_r2 for what this measures.
            informativeness, informativeness_raw = DCIMetrics._informativeness_adjusted_r2(
                representations,
                factors,
            )

            return {
                "disentanglement": max(min(disentanglement, 1.0), 0.0),
                "completeness": max(min(completeness, 1.0), 0.0),
                "informativeness": (
                    float("nan")
                    if math.isnan(informativeness)
                    else max(min(informativeness, 1.0), 0.0)
                ),
                "informativeness_raw": informativeness_raw,
                "informativeness_estimator": "mean_adjusted_r2_full_linear_predictor",
            }
        except Exception:
            return {
                "disentanglement": 0.0,
                "completiveness": 0.0,
                "informativeness": float("nan"),
                "informativeness_raw": float("nan"),
                "informativeness_estimator": "mean_adjusted_r2_full_linear_predictor",
                "error": "DCIMetrics.compute raised; see traceback",
            }

    @staticmethod
    def _informativeness_adjusted_r2(representations, factors):
        """Mean adjusted R^2 of predicting each factor from ALL dimensions.

        Eastwood & Williams define informativeness as how well a *full
        predictor* — using every representation dimension — recovers each
        generative factor, so D = 0 means no dimension carries factor
        information and D = 1 means it is perfectly recoverable.

        This implementation fits, for each factor j, an ordinary-least-squares
        predictor over the full D-dimensional representation plus an intercept,
        and reports the mean **adjusted** R^2 across factors.

        AUDIT NOTE (wave-1 finding I5). The previous implementation was
        ``R.max(dim=0).values.mean()`` — the maximum per-factor correlation,
        not a full-predictor R^2. Its noise floor is the maximum of D noisy
        correlations, which is large and decays only like 1/sqrt(N):
        0.173 (N=200), 0.152 (N=500), 0.080 (N=2000) on pure noise.

        Adjusted R^2 is used rather than raw R^2 because raw R^2 has an
        upward null bias of about D/N even for a perfect linear predictor;
        adjusted R^2 is unbiased under the null for any N > D + 1. Measured
        noise floor with this estimator: 0.0004 (N=2048, D=16) and 0.0002
        (N=8192, D=64) — two to three orders of magnitude below the raw
        max-correlation floor, and far below any usable threshold.

        Unlike the previous max-correlation proxy, a non-linear predictor can
        only score higher, so a representation that a tree model decodes well
        but a linear model does not is under-reported here. That is a known
        limitation of a linear surrogate, documented rather than hidden.

        Returns:
            (float, float): (clamped to [0, 1], raw mean adjusted R^2, which
                            may be slightly negative under the null)

        """
        N, D = representations.shape
        if N <= D + 1:
            # Not enough residual degrees of freedom for an adjusted R^2.
            return float("nan"), float("nan")

        reps = representations.float()
        design = torch.cat([torch.ones(N, 1, dtype=reps.dtype), reps], dim=1)
        n_predictors = design.shape[1] - 1

        scores = []
        for j in range(factors.shape[1]):
            y = factors[:, j].float()
            centered = y - y.mean()
            ss_tot = float((centered * centered).sum())
            if ss_tot < 1e-20:
                # Constant factor carries no information to predict.
                continue
            try:
                solution = torch.linalg.lstsq(design, y.unsqueeze(1), driver="gelsd").solution
            except Exception:
                continue
            residual = y - design @ solution.squeeze(1)
            ss_res = float((residual * residual).sum())
            r2 = 1.0 - ss_res / ss_tot
            scores.append(1.0 - (1.0 - r2) * (N - 1) / (N - n_predictors - 1))

        if not scores:
            return float("nan"), float("nan")
        raw = sum(scores) / len(scores)
        return raw, raw

    @staticmethod
    def measure_noise_floor(n, d, k, n_trials=8, seed=0):
        """Measure the informativeness noise floor on independent random data.

        Callers should compare any informativeness threshold against this
        number rather than against a hard-coded constant: the floor depends on
        N, D and K.

        Args:
            n: number of samples
            d: number of representation dimensions
            k: number of factors
            n_trials: independent trials to average over
            seed: RNG seed

        Returns:
            dict with 'mean', 'std', 'max' and 'min' raw adjusted R^2

        """
        gen = torch.Generator().manual_seed(seed)
        values = []
        for _ in range(n_trials):
            reps = torch.randn(n, d, generator=gen)
            factors = torch.randn(n, k, generator=gen)
            _clamped, raw = DCIMetrics._informativeness_adjusted_r2(reps, factors)
            if not math.isnan(raw):
                values.append(raw)
        if not values:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "max": float("nan"),
                "min": float("nan"),
            }
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        return {
            "mean": mean,
            "std": math.sqrt(var),
            "max": max(values),
            "min": min(values),
        }


class SAPScore:
    """Separate Attribute Predictability (SAP).

    Kumar et al. (2018): measures if each attribute can be predicted
    from single dimensions. Higher = more disentangled.

    Note (audit R9): for single-dimension linear predictability, R² of the
    OLS fit equals squared Pearson correlation, and the sign is already
    discarded below — so |rho| here is numerically identical to the paper's
    regression-R² variant. A non-linear predictability option (binned MI /
    trees) remains future work; do not "fix" this to R² expecting different
    numbers.
    """

    @staticmethod
    def compute(representations, factors):
        """Compute SAP score.

        Args:
            representations: (N, D)
            factors: (N, K) ground-truth factors

        Returns:
            float: SAP score

        """
        try:
            _N, D = representations.shape
            K = factors.shape[1] if factors.dim() > 1 else 1
            if factors.dim() == 1:
                factors = factors.unsqueeze(1)

            score_matrix = torch.zeros(D, K)
            for i in range(D):
                for j in range(K):
                    score_matrix[i, j] = abs(
                        _pearson_correlation(representations[:, i], factors[:, j]),
                    )

            # For each factor, take top-2 most predictive dimensions
            sap = 0.0
            for j in range(K):
                top2 = score_matrix[:, j].topk(min(2, D)).values
                gap = top2[0] - (top2[1] if len(top2) > 1 else torch.tensor(0.0))
                sap += gap.item()

            return max(min(sap / K, 1.0), 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def compute_nonlinear(representations, factors, n_bins=20):
        """Non-linear SAP variant (audit R15 backlog): discretized-MI
        predictability replaces |Pearson|, capturing monotone-nonlinear
        dependencies the linear variant misses. Same top-2-gap aggregation;
        note MI is unbounded above, so values are NOT comparable to the
        linear SAP score — use it for ranking, not absolute levels.
        """
        try:
            _N, D = representations.shape
            K = factors.shape[1] if factors.dim() > 1 else 1
            if factors.dim() == 1:
                factors = factors.unsqueeze(1)

            score_matrix = torch.zeros(D, K)
            for i in range(D):
                d_i = _discretize(representations[:, i], n_bins)
                for j in range(K):
                    f_j = _discretize(factors[:, j], n_bins)
                    score_matrix[i, j] = _mutual_information(d_i, f_j, n_bins)

            sap = 0.0
            for j in range(K):
                top2 = score_matrix[:, j].topk(min(2, D)).values
                gap = top2[0] - (top2[1] if len(top2) > 1 else torch.tensor(0.0))
                sap += gap.item()
            return max(min(sap / K, 1.0), 0.0)
        except Exception:
            return 0.0


class MIGScore:
    """Mutual Information Gap (MIG).

    Chen et al. (2018): for each generative factor, computes the gap
    between the top-2 most informative dimensions.
    Higher MIG = more disentangled.
    """

    @staticmethod
    def compute(representations, factors, n_bins=20):
        """Compute MIG score.

        Args:
            representations: (N, D)
            factors: (N, K) ground-truth factors
            n_bins: number of bins for discretization

        Returns:
            float: MIG score

        """
        try:
            _N, D = representations.shape
            K = factors.shape[1] if factors.dim() > 1 else 1
            if factors.dim() == 1:
                factors = factors.unsqueeze(1)

            mig = 0.0
            n_valid = 0  # factors with non-zero entropy actually contribute
            for j in range(K):
                # Compute mutual information between each dim and factor j
                mi_values = torch.zeros(D)
                f_j = factors[:, j]
                # Discretize factor
                f_j_disc = _discretize(f_j, n_bins)
                h_factor = _entropy(f_j_disc, n_bins)

                if h_factor == 0:
                    continue

                n_valid += 1

                for i in range(D):
                    d_i = _discretize(representations[:, i], n_bins)
                    mi_values[i] = _mutual_information(d_i, f_j_disc, n_bins)

                # Gap between top-2 MI values, normalized by factor entropy
                top2 = mi_values.topk(min(2, D)).values
                gap = top2[0] - (top2[1] if len(top2) > 1 else torch.tensor(0.0))
                mig += gap.item() / (h_factor + 1e-10)

            # Normalize by CONTRIBUTING factors: zero-entropy factors were
            # skipped above, so dividing by K deflated the score silently.
            return max(min(mig / max(n_valid, 1), 1.0), 0.0)
        except Exception:
            return 0.0


class ModularityScore:
    """Modularity metric from Ridgeway & Mozer (2018).

    Measures whether each dimension depends on at most one factor.
    Different from DCI disentanglement: uses deviation from perfect
    one-hot importance allocation.

    Direction: HIGHER is more modular. The score is the mean over dimensions
    of

        mod_i = (sum_j p_ij^2 - 1/K) / (1 - 1/K),   p_i = R_i / sum_j R_ij

    which is 1.0 when dimension i puts all its importance on a single factor
    (the "deviation from perfect one-hot" the docstring describes) and 0.0 when
    it spreads importance uniformly across all K factors.

    AUDIT NOTE (wave-1 finding I5). The previous implementation computed
    ``1 - (sum p^2 - 1/K) / (1 - 1/K)``, which inverts that range: ideal
    one-hot scored 0 and maximally-mixed scored 1. Measured before the fix:
    ideal one-hot 0.240, partial mixture 0.256, uniform 0.349, polysemantic
    0.911 — the most degenerate case won. It also assigned modularity 1.0 to
    any dimension correlated with nothing.

    Dead (constant, or uncorrelated-with-every-factor) dimensions score 0.0,
    so a representation padded with dead dimensions is penalised rather than
    rewarded. Use ``compute_with_details`` to see the live/dead split.
    """

    @staticmethod
    def compute(representations, factors):
        """Compute modularity score.

        Args:
            representations: (N, D)
            factors: (N, K) ground-truth factors

        Returns:
            float: modularity score in [0, 1]

        """
        try:
            _N, D = representations.shape
            K = factors.shape[1] if factors.dim() > 1 else 1
            if factors.dim() == 1:
                factors = factors.unsqueeze(1)

            return ModularityScore.compute_with_details(representations, factors)["modularity"]
        except Exception:
            return 0.0

    @staticmethod
    def compute_with_details(representations, factors):
        """Compute modularity plus the live/dead dimension split.

        Args:
            representations: (N, D)
            factors: (N, K) ground-truth factors

        Returns:
            dict with 'modularity', 'n_dead_dims', 'n_live_dims'

        """
        try:
            _N, D = representations.shape
            K = factors.shape[1] if factors.dim() > 1 else 1
            if factors.dim() == 1:
                factors = factors.unsqueeze(1)

            # Importance: |correlation| between each dim and factor
            R = torch.zeros(D, K)
            for i in range(D):
                for j in range(K):
                    R[i, j] = abs(_pearson_correlation(representations[:, i], factors[:, j]))

            # Per-dimension modularity
            mod_per_dim = torch.zeros(D)
            n_dead = 0
            for i in range(D):
                r = R[i]
                r_sum = r.sum()
                if r_sum < 1e-10:
                    # A dimension correlated with NO factor has no importance
                    # distribution to be peaked, so its modularity is 0 — not
                    # 1. Scoring it perfect meant the most degenerate
                    # representation won the metric.
                    mod_per_dim[i] = 0.0
                    n_dead += 1
                    continue
                # Peakedness of the normalised importance distribution:
                #   1.0 when one factor carries all the importance (one-hot)
                #   0.0 when all K factors are equally important (polysemantic)
                # mod = (sum(p^2) - 1/K) / (1 - 1/K)
                p = r / r_sum
                p_sq_sum = (p**2).sum()
                mod = (p_sq_sum - 1.0 / K) / (1.0 - 1.0 / K)
                mod_per_dim[i] = max(min(mod, 1.0), 0.0)

            return {
                "modularity": mod_per_dim.mean().item(),
                "n_dead_dims": n_dead,
                "n_live_dims": D - n_dead,
            }
        except Exception:
            return {"modularity": 0.0, "n_dead_dims": 0, "n_live_dims": 0}


def compute_all_disentanglement_metrics(representations, factors):
    """Compute all disentanglement metrics at once.

    Args:
        representations: (N, D) representation vectors
        factors: (N, K) ground-truth factor labels

    Returns:
        dict with all metrics

    """
    results = {}
    results.update(DCIMetrics.compute(representations, factors))
    results["sap"] = SAPScore.compute(representations, factors)
    results["mig"] = MIGScore.compute(representations, factors)
    results["modularity"] = ModularityScore.compute(representations, factors)
    return results


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════


def _pearson_correlation(x, y):
    """Pearson correlation coefficient between two vectors."""
    x = x.float()
    y = y.float()
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = x_centered.norm() * y_centered.norm()
    if denom < 1e-10:
        return torch.tensor(0.0)
    return (x_centered @ y_centered) / denom


def _discretize(x, n_bins):
    """Discretize continuous values into bins."""
    try:
        bins = torch.linspace(x.min(), x.max() + 1e-8, n_bins + 1)
        return torch.bucketize(x.contiguous(), bins[1:-1].contiguous())  # bin indices
    except Exception as e:
        warnings.warn(f"_discretize failed ({e}); returning all-zero bins")
        return torch.zeros_like(x, dtype=torch.long)


def _entropy(x, n_classes):
    """Shannon entropy of a discrete distribution."""
    try:
        counts = torch.bincount(x.long().clamp(0, n_classes - 1), minlength=n_classes).float()
        probs = counts / (counts.sum() + 1e-10)
        probs = probs[probs > 1e-10]
        return -(probs * torch.log(probs)).sum().item()
    except Exception:
        return 0.0


def _mutual_information(x, y, n_classes):
    """Mutual information between two discrete variables."""
    try:
        x = x.long().clamp(0, n_classes - 1)
        y = y.long().clamp(0, n_classes - 1)
        N = x.numel()
        # Joint distribution
        joint = torch.zeros(n_classes, n_classes)
        for i in range(N):
            joint[x[i], y[i]] += 1
        joint = joint / (N + 1e-10)

        # Marginals
        px = joint.sum(dim=1)
        py = joint.sum(dim=0)

        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for i in range(n_classes):
            for j in range(n_classes):
                if joint[i, j] > 1e-10 and px[i] > 1e-10 and py[j] > 1e-10:
                    mi += joint[i, j] * math.log(joint[i, j] / (px[i] * py[j]))
        return max(mi, 0.0)
    except Exception:
        return 0.0
