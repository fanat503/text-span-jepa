# Copyright 2026 Text-Span-JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Disentanglement metrics for comparing JEPA vs baseline representations
# DCI: Eastwood & Williams (2018) "A Framework for the Quantitative Evaluation
#       of Disentangled Representations"
# MIG: Chen et al. (2018) "Isolating Sources of Disentanglement in VAEs"
# SAP: Kumar et al. (2018) "Variational Inference for Monte Carlo Objectives"
# Modularity: Ridgeway & Mozer (2018) "Learning Deep Disentangled Embeddings"

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
                        D + 1e-10
                    )
                else:
                    compl_per_factor[j] = 0.0

            weights_c = col_sums.squeeze()
            weights_c = weights_c / (weights_c.sum() + 1e-10)
            completeness = (compl_per_factor * weights_c).sum().item()

            # Informativeness: mean R-squared across all (dimension, factor) pairs
            informativeness = R.max(dim=0).values.mean().item()

            return {
                "disentanglement": max(min(disentanglement, 1.0), 0.0),
                "completeness": max(min(completeness, 1.0), 0.0),
                "informativeness": max(min(informativeness, 1.0), 0.0),
            }
        except Exception:
            return {"disentanglement": 0.0, "completeness": 0.0, "informativeness": 0.0}


class SAPScore:
    """Separate Attribute Predictability (SAP).

    Kumar et al. (2018): measures if each attribute can be predicted
    from single dimensions. Higher = more disentangled.
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
                        _pearson_correlation(representations[:, i], factors[:, j])
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

            # Importance: |correlation| between each dim and factor
            R = torch.zeros(D, K)
            for i in range(D):
                for j in range(K):
                    R[i, j] = abs(_pearson_correlation(representations[:, i], factors[:, j]))

            # Per-dimension modularity
            mod_per_dim = torch.zeros(D)
            for i in range(D):
                r = R[i]
                r_sum = r.sum()
                if r_sum < 1e-10:
                    mod_per_dim[i] = 1.0
                    continue
                # Deviation from perfect one-hot (one factor = 1, rest = 0)
                # mod = 1 - (sum(r^2) / (sum(r))^2 - 1/K) / (1 - 1/K)
                p = r / r_sum
                p_sq_sum = (p**2).sum()
                mod = 1.0 - (p_sq_sum - 1.0 / K) / (1.0 - 1.0 / K + 1e-10)
                mod_per_dim[i] = max(min(mod, 1.0), 0.0)

            return mod_per_dim.mean().item()
        except Exception:
            return 0.0


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
