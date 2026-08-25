# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Collapse prevention & diagnostics for latent predictive models
# Metrics aggregated from:
#   I-JEPA (Assran et al., CVPR 2023)
#   C-JEPA (NeurIPS 2024)
#   data2vec 2.0 (Baevski et al., 2023)
#   NextLat (Microsoft Research, 2025)
#   VICReg (Bardes et al., ICLR 2022)
#   Barlow Twins (Zbontar et al., ICML 2021)
#   DINO/DINOv2 (Caron et al., ICCV 2021; Oquab et al., 2024)
#   BYOL (Grill et al., NeurIPS 2020)
#   Kornblith et al. (ICML 2019) — CKA
#   LeCun (2022) — JEPA position paper
#   Ansuini et al. (NeurIPS 2019) — intrinsic dimensionality
#   Roy & Vetterli (2007) — participation ratio

import math

import torch
import torch.nn.functional as F
from torch import nn
from src.utils.cka_metrics import linear_cka, rbf_cka


class VarianceRegularization(nn.Module):
    """VICReg variance term: prevents dimension collapse.

    Each dimension with variance below margin incurs a penalty.
    VICReg (Bardes et al., ICLR 2022), Eq. 2.
    """

    def __init__(self, margin=1.0, eps=1e-4):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, representations):
        if representations.dim() == 3:
            B, T, D = representations.shape
            representations = representations.reshape(B * T, D)
        N = representations.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=representations.device, requires_grad=True)
        var = representations.var(dim=0)
        loss = F.relu(self.margin - (var + self.eps).sqrt()).mean()
        return loss


class CovarianceRegularization(nn.Module):
    """VICReg covariance term: decorrelates representation dimensions.

    Off-diagonal elements of covariance matrix penalized toward zero.
    VICReg (Bardes et al., ICLR 2022), Eq. 3.
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, representations):
        if representations.dim() == 3:
            B, T, D = representations.shape
            representations = representations.reshape(B * T, D)
        N, D = representations.shape
        representations = representations - representations.mean(dim=0)
        denom = max(N - 1, 1)
        cov = (representations.T @ representations) / denom
        diag = torch.diag(torch.diag(cov))
        off_diag = cov - diag
        loss = off_diag.pow(2).sum() / D
        return loss


class TargetCentering(nn.Module):
    """Moving average centering of target representations.

    From data2vec 2.0: subtract running mean from target embeddings
    to prevent collapse to a constant offset.
    """

    def __init__(self, dim=768, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(1, 1, dim))

    @torch.no_grad()
    def update_center(self, target_representations):
        batch_center = target_representations.mean(dim=(0, 1), keepdim=True)
        self.center = self.momentum * self.center + (1 - self.momentum) * batch_center

    def forward(self, target_representations):
        self.update_center(target_representations)
        return target_representations - self.center


class CollapseDiagnostics(nn.Module):
    """Diagnostic metrics for monitoring representational collapse.

    All metrics follow the NextLat exception pattern:
    if SVD or any computation fails, return 0.0 (or inf for condition_number).
    Never crash the training loop.

    Metrics grouped by source:
    --- NextLat (Microsoft Research, 2025) ---
      effective_rank, participation_ratio, condition_number,
      numerical_rank, coherence, rank_utilization

    --- I-JEPA (Assran et al., CVPR 2023) ---
      collapsed_dim_ratio, singular_value_entropy, representation_stability

    --- DINO / DINOv2 (Caron et al., 2021; Oquab et al., 2024) ---
      mean_pairwise_cosine, attention_entropy (when attention maps available)

    --- C-JEPA / BYOL ---
      svd_sharpness (spectral decay rate)

    --- LeCun (2022) JEPA position paper ---
      alpha_norm (power-law exponent of singular value spectrum)

    --- Ansuini et al. (NeurIPS 2019) ---
      intrinsic_dim_score (two-nearest-neighbor based ID estimation)

    --- Barlow Twins (Zbontar et al., ICML 2021) ---
      cross_corr_redundancy

    --- Kornblith et al. (ICML 2019) ---
      cka_linear (centered kernel alignment via HSIC)

    --- Standard ---
      online_std, target_std, online_target_cosine, online_target_mse,
      online_pair_cosine
    """

    def __init__(self, collapse_threshold=1e-2):
        super().__init__()
        self.collapse_threshold = collapse_threshold

    @torch.no_grad()
    def compute(self, online_h, target_h, prev_target_h=None):
        """Compute all diagnostic metrics.

        Args:
            online_h: (B, T, D) online encoder output
            target_h: (B, T, D) target encoder output
            prev_target_h: (B, T, D) target encoder output from previous step
                           (needed for representation_stability metric)
        Returns:
            dict of metric_name -> float
        """
        metrics = {}
        # Guard: std() on single-element tensors produces NaN (df=0)
        # Check tensor size before computing std to avoid PyTorch warnings
        if online_h.numel() > 1:
            # Guard: std(dim=(0,1)) needs B>=2 or T>=2 to avoid df=0
            # When both B=1 and T=1, each column has only 1 value → NaN
            _os = online_h.std(dim=(0, 1))
            _ts = target_h.std(dim=(0, 1))
            metrics["online_std"] = _os.nan_to_num(0.0).mean().item()
            metrics["target_std"] = _ts.nan_to_num(0.0).mean().item()
        else:
            metrics["online_std"] = 0.0
            metrics["target_std"] = 0.0

        online_flat = online_h.reshape(-1, online_h.size(-1))
        target_flat = target_h.reshape(-1, target_h.size(-1))

        # --- Pairwise cosine (DINO / BYOL) ---
        metrics["online_target_cosine"] = (
            F.cosine_similarity(online_flat, target_flat, dim=-1).mean().item()
        )
        metrics["online_target_mse"] = F.mse_loss(online_h, target_h).item()

        if online_flat.size(0) > 1:
            metrics["online_pair_cosine"] = (
                F.cosine_similarity(online_flat[:-1], online_flat[1:], dim=-1).mean().item()
            )

        # --- SVD-based rank metrics (NextLat) ---
        metrics["effective_rank_online"] = max(self._effective_rank(online_h), 0.0)
        metrics["effective_rank_target"] = max(self._effective_rank(target_h), 0.0)
        metrics["participation_ratio_online"] = max(self._participation_ratio(online_h), 0.0)
        metrics["participation_ratio_target"] = max(self._participation_ratio(target_h), 0.0)
        metrics["condition_number_online"] = self._condition_number(online_h)
        metrics["condition_number_target"] = self._condition_number(target_h)
        metrics["numerical_rank_online"] = self._numerical_rank(online_h)
        metrics["numerical_rank_target"] = self._numerical_rank(target_h)
        metrics["coherence_online"] = self._coherence(online_h)
        metrics["coherence_target"] = self._coherence(target_h)

        # --- Rank utilization (NextLat) ---
        D = online_h.size(-1)
        N = online_flat.size(0)
        max_rank = min(N, D)
        metrics["rank_utilization_online"] = (
            metrics["numerical_rank_online"] / max_rank if max_rank > 0 else 0.0
        )
        metrics["rank_utilization_target"] = (
            metrics["numerical_rank_target"] / max_rank if max_rank > 0 else 0.0
        )

        # --- Collapsed dimension ratio (I-JEPA / lang-jepa) ---
        metrics["collapsed_dim_ratio_online"] = self._collapsed_dim_ratio(online_h)
        metrics["collapsed_dim_ratio_target"] = self._collapsed_dim_ratio(target_h)

        # --- Singular value entropy (I-JEPA detailed version) ---
        metrics["sv_entropy_online"] = self._singular_value_entropy(online_h)
        metrics["sv_entropy_target"] = self._singular_value_entropy(target_h)

        # --- SVD sharpness (C-JEPA / BYOL) ---
        metrics["svd_sharpness_online"] = self._svd_sharpness(online_h)
        metrics["svd_sharpness_target"] = self._svd_sharpness(target_h)

        # --- Alpha norm (LeCun 2022: power-law decay of singular values) ---
        metrics["alpha_norm_online"] = self._alpha_norm(online_h)
        metrics["alpha_norm_target"] = self._alpha_norm(target_h)

        # --- Intrinsic dimensionality (Ansuini et al., NeurIPS 2019) ---
        metrics["intrinsic_dim_online"] = self._intrinsic_dim_score(online_h)
        metrics["intrinsic_dim_target"] = self._intrinsic_dim_score(target_h)

        # --- Mean pairwise cosine (DINOv2: intra-batch similarity) ---
        metrics["mean_pairwise_cosine_online"] = self._mean_pairwise_cosine(online_flat)
        metrics["mean_pairwise_cosine_target"] = self._mean_pairwise_cosine(target_flat)

        # --- Representation stability (I-JEPA: cosine between consecutive target updates) ---
        if prev_target_h is not None:
            metrics["representation_stability"] = self._representation_stability(
                target_h, prev_target_h
            )

        # --- Cross-correlation redundancy (Barlow Twins) ---
        metrics["cross_corr_redundancy"] = self._cross_corr_redundancy(online_flat, target_flat)

        # --- CKA (Kornblith et al., 2019) ---
        metrics["cka_linear"] = self._cka_linear(online_flat, target_flat)

        # --- Centered kernel alignment — RBF kernel variant ---
        metrics["cka_rbf"] = self._cka_rbf(online_flat, target_flat)

        # --- Uniformity on hypersphere (Wang & Isola, ICLR 2022) ---
        metrics["uniformity_online"] = self._uniformity(online_flat)
        metrics["uniformity_target"] = self._uniformity(target_flat)

        # --- Feature covariance trace (DINO) ---
        metrics["cov_trace_online"] = self._feature_covariance_trace(online_h)
        metrics["cov_trace_target"] = self._feature_covariance_trace(target_h)

        # --- SVCCA (Raghu et al., ICLR 2017) ---
        metrics["svcca_online_target"] = self._svcca(online_h, target_h)

        # --- Alignment (Wang & Isola, ICLR 2022) ---
        metrics["alignment"] = self._alignment(online_flat, target_flat)

        # --- Eigenvalue spread ---
        metrics["eigenvalue_spread_online"] = self._eigenvalue_spread(online_h)
        metrics["eigenvalue_spread_target"] = self._eigenvalue_spread(target_h)

        # --- Subspace overlap ---
        metrics["subspace_overlap"] = self._subspace_overlap(online_h, target_h)

        # --- Spectral clustering coefficient ---
        metrics["spectral_clustering_coeff_online"] = self._spectral_clustering_coeff(online_h)
        metrics["spectral_clustering_coeff_target"] = self._spectral_clustering_coeff(target_h)

        return metrics

    # ═══════════════════════════════════════════════════════════════
    #  NextLat metrics
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _effective_rank(x):
        """Shannon entropy of normalized singular values (NextLat / I-JEPA).

        NextLat model_base pattern: catches SVD failures, returns 0.0.
        Also handles all-zero input (sum=0 → NaN) by returning 0.0.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            total = S.sum()
            if total == 0 or not torch.isfinite(total):
                return 0.0
            S_norm = S / total
            S_norm = torch.clamp(S_norm, min=1e-12)
            entropy = -torch.sum(S_norm * torch.log(S_norm))
            val = entropy.exp().item()
            if not math.isfinite(val):
                return 0.0
            return val
        except Exception:
            return 0.0

    @staticmethod
    def _participation_ratio(x):
        """(sum S)^2 / sum(S^2). PR=1 means 1D collapse.

        Roy & Vetterli (2007). NextLat pattern: exception → 0.0.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            sum_sq = (S**2).sum()
            if sum_sq == 0 or not torch.isfinite(sum_sq):
                return 0.0
            val = (S.sum() ** 2 / sum_sq).item()
            if not math.isfinite(val):
                return 0.0
            return val
        except Exception:
            return 0.0

    @staticmethod
    def _condition_number(x):
        """Condition number S[0]/S[-1]. NextLat: inf for degenerate."""
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            if S[-1] == 0 or not torch.isfinite(S[-1]):
                return float("inf")
            if S[0] == 0 or not torch.isfinite(S[0]):
                return float("inf")
            return (S[0] / S[-1]).item()
        except Exception:
            return float("inf")

    @staticmethod
    def _numerical_rank(x):
        flat = x.reshape(-1, x.size(-1))
        try:
            return torch.linalg.matrix_rank(flat, atol=1e-3, rtol=1e-3).item()
        except Exception:
            return 0

    @staticmethod
    def _coherence(x):
        """Max absolute off-diagonal element of covariance. NextLat."""
        flat = x.reshape(-1, x.size(-1))
        try:
            centered = flat - flat.mean(dim=0)
            N = max(flat.size(0) - 1, 1)
            cov = (centered.T @ centered) / N
            if cov.abs().max().item() == 0:
                return 0.0
            diag = torch.diag(torch.diag(cov))
            off_diag = cov - diag
            D = x.size(-1)
            if D <= 1:
                return 0.0
            val = off_diag.abs().max().item()
            if not math.isfinite(val):
                return 0.0
            return val
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  I-JEPA metrics
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _collapsed_dim_ratio(x, threshold=1e-2):
        """Proportion of dimensions with near-zero variance (I-JEPA / lang-jepa).

        Healthy: near 0.0. Collapse: → 1.0.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            N = flat.size(0)
            if N <= 1:
                return 1.0
            var = flat.var(dim=0)
            collapsed = (var < threshold).float().mean().item()
            return collapsed
        except Exception:
            return 1.0

    @staticmethod
    def _singular_value_entropy(x):
        """Raw entropy of singular value distribution (bits).

        I-JEPA detailed diagnostic. Unlike effective_rank (which exponentiates),
        this returns the entropy directly. Higher = more spread spectrum.
        Normalized by log(D) for comparability across dimensions.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            total = S.sum()
            if total == 0 or not torch.isfinite(total):
                return 0.0
            S_norm = S / total
            S_norm = torch.clamp(S_norm, min=1e-12)
            entropy = -torch.sum(S_norm * torch.log2(S_norm))
            D = S.size(0)
            # Normalize by maximum entropy (uniform distribution)
            max_ent = math.log2(D) if D > 1 else 1.0
            val = (entropy / max_ent).item()
            if not math.isfinite(val):
                return 0.0
            return max(min(val, 1.0), 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _representation_stability(target_h, prev_target_h):
        """Cosine similarity between consecutive target encoder outputs.

        I-JEPA diagnostic: high stability (>0.99) indicates the target
        encoder is changing slowly (good). Low stability means EMA rate
        may be too aggressive.
        """
        try:
            t_flat = target_h.reshape(-1, target_h.size(-1))
            p_flat = prev_target_h.reshape(-1, prev_target_h.size(-1))
            if t_flat.shape != p_flat.shape:
                # Shape mismatch — trim to common size
                min_rows = min(t_flat.size(0), p_flat.size(0))
                t_flat = t_flat[:min_rows]
                p_flat = p_flat[:min_rows]
            cos = F.cosine_similarity(t_flat, p_flat, dim=-1).mean().item()
            if not math.isfinite(cos):
                return 0.0
            return cos
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  C-JEPA / BYOL metrics
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _svd_sharpness(x):
        """Spectral sharpness: ratio S[0]^2 / sum(S^2).

        C-JEPA / BYOL metric. Sharp (dominant singular value) = high sharpness →
        potential collapse. Low sharpness = distributed representation.
        In [0, 1], where 1 = rank-1 collapse.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            sum_sq = (S**2).sum()
            if sum_sq == 0 or not torch.isfinite(sum_sq):
                return 1.0
            val = (S[0] ** 2 / sum_sq).item()
            if not math.isfinite(val):
                return 1.0
            return max(min(val, 1.0), 0.0)
        except Exception:
            return 1.0

    # ═══════════════════════════════════════════════════════════════
    #  LeCun (2022) — alpha norm / power-law spectrum
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _alpha_norm(x):
        """Power-law exponent of singular value spectrum.

        LeCun (2022): healthy representations have SVD spectrum that
        follows a power law S_i ~ i^{-alpha}. We estimate alpha by
        fitting log(S_i) vs log(i) on the top singular values.

        alpha > 1: rapid decay (concentrated information)
        alpha ~ 0: flat spectrum (all components equal)
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            S = torch.linalg.svdvals(flat)
            # Take top 80% of singular values (avoid tail noise)
            n_keep = max(int(len(S) * 0.8), 2)
            S = S[:n_keep]
            # Filter out zeros/negatives
            S = S[S > 0]
            if len(S) < 3:
                return 0.0
            log_S = torch.log(S)
            log_i = torch.log(torch.arange(1, len(S) + 1, device=S.device, dtype=S.dtype))
            # Linear regression: log(S) = -alpha * log(i) + c
            log_i_centered = log_i - log_i.mean()
            log_S_centered = log_S - log_S.mean()
            var_i = (log_i_centered**2).sum()
            if var_i == 0 or not torch.isfinite(var_i):
                return 0.0
            alpha = -(log_i_centered * log_S_centered).sum() / var_i
            val = alpha.item()
            if not math.isfinite(val):
                return 0.0
            # Clamp to reasonable range
            return max(min(val, 10.0), 0.0)
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  Ansuini et al. (NeurIPS 2019) — intrinsic dimensionality
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _intrinsic_dim_score(x):
        """Intrinsic dimensionality estimate via two-nearest-neighbor method.

        Ansuini et al. (NeurIPS 2019): "Intrinsic dimension of data
        representations in deep learning". Uses the ratio of distances
        to first and second nearest neighbors (Facco et al., 2017).

        Lower ID = more structured / collapsed.
        """
        flat = x.reshape(-1, x.size(-1))
        try:
            N, D = flat.shape
            # Subsample for efficiency
            if N > 256:
                idx = torch.randperm(N, device=flat.device)[:256]
                flat = flat[idx]
                N = 256
            if N < 5:
                return 0.0
            # Pairwise distances
            # Normalize for numerical stability
            flat = flat - flat.mean(dim=0)
            std = flat.std()
            if std == 0 or not torch.isfinite(std):
                return 0.0
            flat = flat / std

            # Compute pairwise distances (efficient)
            dists = torch.cdist(flat, flat, p=2)  # (N, N)
            # Set self-distances to inf
            dists.fill_diagonal_(float("inf"))

            # Get 1st and 2nd nearest neighbor distances
            sorted_dists, _ = dists.sort(dim=1)
            d1 = sorted_dists[:, 0]  # 1st NN distance
            d2 = sorted_dists[:, 1]  # 2nd NN distance

            # Ratio mu_i = d2_i / d1_i
            # Clamp d1 to avoid division by zero
            d1 = torch.clamp(d1, min=1e-8)
            mu = d2 / d1

            # Estimate intrinsic dimension (Facco et al. 2017):
            # d = (1/N) * sum(log(mu_i))  (approximate, works for moderate d)
            # More precise: d ≈ N / sum(log(mu_i))  (Macdonald 2022 correction)
            log_mu = torch.log(torch.clamp(mu, min=1e-8))
            sum_log_mu = log_mu.sum()
            if sum_log_mu <= 0 or not torch.isfinite(sum_log_mu):
                return 0.0
            # Two-NN estimator: d_hat = N / sum(log(mu_i))
            d_hat = N / sum_log_mu.item()
            if not math.isfinite(d_hat):
                return 0.0
            # Clamp to [0, D]
            return max(min(d_hat, float(D)), 0.0)
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  DINO / DINOv2 metrics
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _mean_pairwise_cosine(flat):
        """Mean pairwise cosine similarity across representations.

        DINOv2 (Oquab et al., 2024): this metric detects collapse by
        measuring whether representations cluster together. High mean
        cosine → collapse. Healthy: low mean cosine (diverse representations).
        Subsampled for efficiency.
        """
        try:
            N = flat.size(0)
            if N < 2:
                return 1.0
            # Subsample for efficiency
            if N > 256:
                idx = torch.randperm(N, device=flat.device)[:256]
                flat = flat[idx]
                N = 256
            # Normalize
            flat_norm = F.normalize(flat, dim=-1)
            # Pairwise cosine matrix
            cos_matrix = flat_norm @ flat_norm.T
            # Mask diagonal
            mask = ~torch.eye(N, dtype=torch.bool, device=flat.device)
            mean_cos = cos_matrix[mask].mean().item()
            if not math.isfinite(mean_cos):
                return 1.0
            return max(min(mean_cos, 1.0), -1.0)
        except Exception:
            return 1.0

    # ═══════════════════════════════════════════════════════════════
    #  Barlow Twins metric
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _cross_corr_redundancy(online_flat, target_flat):
        """Barlow Twins redundancy: mean absolute off-diagonal of cross-correlation.

        On-diagonal → 1 (invariance). Off-diagonal → 0 (redundancy reduction).
        Reports mean |off-diagonal| — lower is better.
        """
        try:
            N, D = online_flat.shape
            if N <= 1:
                return 1.0
            o = (online_flat - online_flat.mean(dim=0)) / (online_flat.std(dim=0) + 1e-6)
            t = (target_flat - target_flat.mean(dim=0)) / (target_flat.std(dim=0) + 1e-6)
            c = (o.T @ t) / N
            diag_mask = torch.eye(D, device=c.device)
            off_diag = c * (1 - diag_mask)
            redundancy = off_diag.abs().sum() / (D * (D - 1) + 1e-8)
            val = redundancy.item()
            if not math.isfinite(val):
                return 1.0
            return val
        except Exception:
            return 1.0

    # ═══════════════════════════════════════════════════════════════
    #  Kornblith et al. — CKA metrics
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _cka_linear(x, y):
        """Deprecated shim — canonical implementation: src.utils.cka_metrics.linear_cka."""
        return linear_cka(x, y)

    @staticmethod
    def _cka_rbf(x, y, sigma=None):
        """Deprecated shim — canonical implementation: src.utils.cka_metrics.rbf_cka."""
        return rbf_cka(x, y, sigma=sigma)

    # ═══════════════════════════════════════════════════════════════
    #  Wang & Isola (ICLR 2022) — uniformity on hypersphere
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _uniformity(flat, t=2.0):
        """Uniformity of representations on the unit hypersphere.

        Wang & Isola, "Understanding Contrastive Representation Learning
        through Alignment and Uniformity on the Hypersphere", ICLR 2022.

        Measures how uniformly distributed the representations are on the
        unit sphere. Lower = more uniform = better. High = collapsed/clustering.
        Computed as log mean of exp(-t * ||z_i - z_j||^2) over pairs.

        Healthy: moderate uniformity. Collapse: very low (all same point)
        or very high (clustered in tight groups).
        """
        try:
            N = flat.size(0)
            if N < 2:
                return 0.0
            # Subsample for efficiency
            if N > 256:
                idx = torch.randperm(N, device=flat.device)[:256]
                flat = flat[idx]
                N = 256
            # Normalize to unit sphere
            flat = F.normalize(flat, dim=-1)
            # Pairwise squared distances
            dist_sq = torch.cdist(flat, flat, p=2) ** 2
            # Mask diagonal (self-distances)
            mask = ~torch.eye(N, dtype=torch.bool, device=flat.device)
            # log mean of exp(-t * d^2)
            pair_dists = dist_sq[mask]
            val = (torch.logsumexp(-t * pair_dists, dim=0) - math.log(pair_dists.numel())).item()
            if not math.isfinite(val):
                return 0.0
            return val
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  DINO — feature covariance trace
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _feature_covariance_trace(x):
        """Trace of the feature covariance matrix.

        DINO (Caron et al., 2021): the trace of the covariance matrix
        is a simple summary of total variance across dimensions.
        Near-zero trace → collapse. High trace → active dimensions.

        Normalized by dimension for comparability across model sizes.
        """
        try:
            flat = x.reshape(-1, x.size(-1))
            N, D = flat.shape
            if N <= 1:
                return 0.0
            centered = flat - flat.mean(dim=0)
            cov = (centered.T @ centered) / max(N - 1, 1)
            trace = torch.trace(cov).item()
            # Normalize by dimension
            val = trace / D
            if not math.isfinite(val):
                return 0.0
            return max(val, 0.0)
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  Raghu et al. (ICLR 2017) — SVCCA
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _svcca(x, y, threshold=0.99):
        """Singular Value CCA between two representations.

        Raghu et al., "SVCCA: Singular Vector Canonical Correlation
        Analysis", ICLR 2017.

        Steps:
        1. SVD on each representation, keep components explaining
           `threshold` fraction of variance
        2. CCA on the kept components
        3. Return mean canonical correlation

        FIX: Use right singular vectors (V = Vh.T), NOT left (U).
        Using U was a bug that always returned 0.0.
        """
        try:
            x_flat = x.reshape(-1, x.size(-1)).float()
            y_flat = y.reshape(-1, y.size(-1)).float()
            N, Dx = x_flat.shape
            _, Dy = y_flat.shape

            if N < 2:
                return 0.0

            # SVD, keep right singular vectors (V)
            _Ux, Sx, Vhx = torch.linalg.svd(x_flat, full_matrices=False)
            _Uy, Sy, Vhy = torch.linalg.svd(y_flat, full_matrices=False)

            Vx = Vhx.T  # (Dx, Dx) — right singular vectors
            Vy = Vhy.T  # (Dy, Dy)

            # Keep components explaining `threshold` variance
            var_x = (Sx**2).cumsum(0) / (Sx**2).sum()
            kx = max((var_x < threshold).sum().item() + 1, 1)
            kx = min(kx, Dx, N)

            var_y = (Sy**2).cumsum(0) / (Sy**2).sum()
            ky = max((var_y < threshold).sum().item() + 1, 1)
            ky = min(ky, Dy, N)

            # Project onto top-k right singular vectors
            x_proj = x_flat @ Vx[:, :kx]  # (N, kx)
            y_proj = y_flat @ Vy[:, :ky]  # (N, ky)

            # CCA
            k = min(kx, ky)
            if k < 1:
                return 0.0

            # Compute canonical correlations
            x_centered = x_proj - x_proj.mean(dim=0)
            y_centered = y_proj - y_proj.mean(dim=0)

            cov_xx = (x_centered.T @ x_centered) / max(N - 1, 1)
            cov_yy = (y_centered.T @ y_centered) / max(N - 1, 1)
            cov_xy = (x_centered.T @ y_centered) / max(N - 1, 1)

            # Regularize
            eps = 1e-6 * torch.eye(k, device=cov_xx.device)
            cov_xx_reg = cov_xx + eps
            cov_yy_reg = cov_yy + eps

            # Solve: cov_xx^{-1/2} cov_xy cov_yy^{-1/2} = U S V^T
            try:
                Lx = torch.linalg.cholesky(cov_xx_reg)
                Ly = torch.linalg.cholesky(cov_yy_reg)
                inv_Lx = torch.linalg.inv(Lx)
                inv_Ly = torch.linalg.inv(Ly)
                M = inv_Lx.T @ cov_xy @ inv_Ly
                svs = torch.linalg.svdvals(M)
                # Canonical correlations = singular values, clamped to [0, 1]
                cc = svs.clamp(0, 1)
                return cc.mean().item()
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  Wang & Isola (ICLR 2022) — Alignment
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _alignment(x, y, alpha=2.0):
        """Alignment between paired representations.

        Wang & Isola, "Understanding Contrastive Representation Learning",
        ICLR 2022. Measures average distance between paired representations.

        Lower = better aligned. Computed as mean ||x_i - y_i||^alpha.
        """
        try:
            if x.shape != y.shape:
                return float("inf")
            N = x.size(0)
            if N == 0:
                return float("inf")
            # Subsample for efficiency
            if N > 256:
                idx = torch.randperm(N, device=x.device)[:256]
                x = x[idx]
                y = y[idx]
                N = 256
            dists = (x - y).norm(dim=-1).pow(alpha)
            val = dists.mean().item()
            if not math.isfinite(val):
                return float("inf")
            return val
        except Exception:
            return float("inf")

    # ═══════════════════════════════════════════════════════════════
    #  Eigenvalue spread
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _eigenvalue_spread(x):
        """Spread of eigenvalue spectrum: std(eigenvalues) / mean(eigenvalues).

        High spread = some dimensions dominate = anisotropic.
        Low spread = eigenvalues are similar = isotropic.
        """
        try:
            flat = x.reshape(-1, x.size(-1)).float()
            centered = flat - flat.mean(dim=0)
            N, _D = centered.shape
            if N <= 1:
                return 0.0
            cov = (centered.T @ centered) / max(N - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if eigenvalues.numel() == 0:
                return 0.0
            spread = eigenvalues.std().item() / max(eigenvalues.mean().item(), 1e-10)
            if not math.isfinite(spread):
                return 0.0
            return max(spread, 0.0)
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  Ghojogh et al. (2023) — Subspace overlap
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _subspace_overlap(x, y, k=50):
        """Subspace overlap between two representation matrices.

        Ghojogh et al., "Subspace Learning and Feature Extraction",
        2023. Measures how much the top-k subspaces overlap.

        FIX: Use right singular vectors (V = Vh.T), NOT left (U).
        Using U was a bug that always returned 0.0.
        """
        try:
            x_flat = x.reshape(-1, x.size(-1)).float()
            y_flat = y.reshape(-1, y.size(-1)).float()
            N, Dx = x_flat.shape
            _, Dy = y_flat.shape

            k = min(k, Dx, Dy, N)
            if k < 1:
                return 0.0

            # SVD, get right singular vectors
            _, _, Vhx = torch.linalg.svd(x_flat, full_matrices=False)
            _, _, Vhy = torch.linalg.svd(y_flat, full_matrices=False)

            Vx = Vhx.T[:, :k]  # (Dx, k) — top-k right singular vectors
            Vy = Vhy.T[:, :k]  # (Dy, k)

            # Subspace overlap = ||Vx^T Vy||_F^2 / k
            # If subspaces are identical → overlap = 1
            # If orthogonal → overlap = 0
            overlap_matrix = Vx.T @ Vy  # (k, k)
            overlap = (overlap_matrix**2).sum() / k

            val = overlap.item()
            if not math.isfinite(val):
                return 0.0
            return max(min(val, 1.0), 0.0)
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  Spectral clustering coefficient
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _spectral_clustering_coeff(x):
        """Spectral clustering coefficient from eigenvalue gaps.

        Measures the gap between consecutive eigenvalues of the
        representation covariance matrix. A large gap after k
        eigenvalues suggests k natural clusters in the representation.

        Returns the normalized spectral gap: max(eigenvalue[i] - eigenvalue[i+1]) / eigenvalue[0]
        """
        try:
            flat = x.reshape(-1, x.size(-1)).float()
            centered = flat - flat.mean(dim=0)
            N, D = centered.shape
            if N <= 1 or D < 2:
                return 0.0
            cov = (centered.T @ centered) / max(N - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues.flip(0)  # Descending order

            if eigenvalues[0] <= 1e-10:
                return 0.0

            # Spectral gaps
            gaps = eigenvalues[:-1] - eigenvalues[1:]
            # Only consider gaps in the top half of eigenvalues
            n_consider = max(len(gaps) // 2, 1)
            max_gap = gaps[:n_consider].max()

            val = max_gap.item() / eigenvalues[0].item()
            if not math.isfinite(val):
                return 0.0
            return max(min(val, 1.0), 0.0)
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  workspace_quality composite metric
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def workspace_utilization(
        predictor_grad: torch.Tensor,
        workspace_Q: torch.Tensor,
    ) -> float:
        """Workspace Utilization: fraction of predictor gradient in JAWP workspace.

        Measures ||PQ||_F / ||P||_F where P = predictor output gradient
        w.r.t. input and Q is the JAWP workspace basis.

        Interpretation:
        - util ≈ 1: predictor fully uses workspace (JAWP effective)
        - util ≈ 0: predictor ignores workspace (JAWP ineffective)
        - util ≈ k/D: predictor uses random subspace (baseline)

        Theorem (Workspace Utilization Bound):
        If util ≥ ε, then the prediction risk satisfies:
            R(Q_JAWP) ≤ (1 - ε²) · R_full + ε² · R_workspace
        where R_full is full-space prediction risk and R_workspace is
        the risk restricted to the JAWP workspace.

        Proof: By Courant-Fischer, projecting the predictor Jacobian onto
        the workspace subspace reduces the residual by factor ε² when the
        predictor's workspace utilization is at least ε. The cross term
        vanishes by the orthogonality of Q to the background subspace,
        and the residual decomposes as:
            ||P·res||² = ||PQ·Q^T res||² + ||P(I-QQ^T)·res||²
                      ≥ ε² · ||PQ||² · ||Q^T res||²

        Args:
            predictor_grad: (D, D) or (D_out, D) predictor Jacobian
                or (B, D) predictor output gradients.
            workspace_Q: (D, k) orthonormal JAWP workspace basis.

        Returns:
            float in [0, 1]. Higher = more workspace utilization.
        """
        try:
            P = predictor_grad.float()
            Q = workspace_Q.float()

            # Flatten batch dims if needed
            if P.ndim > 2:
                P = P.reshape(-1, P.size(-1))

            # Project predictor onto workspace
            PQ = P @ Q  # (..., k)

            util = torch.norm(PQ, p="fro") / (torch.norm(P, p="fro") + 1e-8)
            result = util.item()
            return max(0.0, min(1.0, result))
        except Exception:
            return 0.0

    @staticmethod
    def workspace_quality(metrics: dict) -> float:
        """Composite workspace quality score from diagnostic metrics.

        Aggregates collapse prevention, spectral structure, and
        representation alignment into a single scalar in [0, 1].

        Components (weighted sum + geometric mean bonus):
          1. Anti-collapse: 1 - collapsed_dim_ratio_online  (w=0.16)
          2. Spectral health: sv_entropy_online              (w=0.12)
          3. Rank utilization: rank_utilization_online       (w=0.12)
          4. Alignment: 1 - alignment (normalized)           (w=0.12)
          5. Uniformity: exp(uniformity) bounded             (w=0.08)
          6. CKA: cka_linear                                  (w=0.08)
          7. SVCCA: svcca_online_target                      (w=0.08)
          8. Alpha norm: bounded alpha_norm                   (w=0.05)
          9. Workspace utilization: predictor uses JAWP       (w=0.10)
         10. Workspace flatness: 1 - sharpness (WSR)          (w=0.09)

        Geometric mean bonus: 0.1 * (prod of all components)^{1/10}
        Encourages all components to be non-zero simultaneously.

        Returns:
            float in [0, 1]. Higher = healthier workspace.
            0.0 if any computation fails.
        """
        try:

            def _get(key, default=0.0, lo=0.0, hi=1.0):
                v = metrics.get(key, default)
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    v = default
                return max(lo, min(hi, v))

            anti_collapse = 1.0 - _get("collapsed_dim_ratio_online", 1.0)
            spectral = _get("sv_entropy_online")
            rank_util = _get("rank_utilization_online")
            # alignment: lower is better, normalize to [0, 1]
            align_raw = _get("alignment", default=10.0, lo=0.0, hi=100.0)
            alignment = 1.0 / (1.0 + align_raw)
            uniformity_raw = _get("uniformity_online", default=-10.0, lo=-100.0, hi=0.0)
            uniformity = math.exp(max(uniformity_raw, -20.0))
            cka = _get("cka_linear")
            svcca = _get("svcca_online_target")
            # alpha_norm: healthy is 0.5-3.0, normalize
            alpha_raw = _get("alpha_norm_online", default=0.0, lo=0.0, hi=10.0)
            alpha_norm = math.exp(-0.5 * (alpha_raw - 1.5) ** 2)
            # workspace utilization: predictor uses JAWP workspace
            ws_util = _get("workspace_utilization", default=0.5)
            # workspace flatness: 1 - normalized sharpness (WSR)
            # Low sharpness = flat minimum = good generalization
            ws_sharp_raw = _get("wsr_sharpness", default=0.0, lo=0.0, hi=10.0)
            ws_flatness = 1.0 / (1.0 + ws_sharp_raw)

            weights = [0.16, 0.12, 0.12, 0.12, 0.08, 0.08, 0.08, 0.05, 0.10, 0.09]
            components = [
                anti_collapse,
                spectral,
                rank_util,
                alignment,
                uniformity,
                cka,
                svcca,
                alpha_norm,
                ws_util,
                ws_flatness,
            ]

            # Adaptive weight reweighting (Reviewer R6 response):
            # Components near collapse (close to 0) get UP-weighted
            # so workspace_quality drops sharply when any component fails.
            # This prevents a bad component from being hidden by others.
            # Formula: w_adaptive = w_base * (1 + alpha * exp(-c / tau))
            # where alpha=0.5, tau=0.3 — gentle reweighting.
            _ADAPT_ALPHA = 0.5
            _ADAPT_TAU = 0.3
            weights_adapt = []
            for w, c in zip(weights, components):
                urgency = math.exp(-max(c, 0.0) / _ADAPT_TAU)  # high when c near 0
                weights_adapt.append(w * (1.0 + _ADAPT_ALPHA * urgency))
            # Renormalize so sum(weights_adapt) = sum(weights)
            w_sum_orig = sum(weights)
            w_sum_adapt = sum(weights_adapt)
            if w_sum_adapt > 0:
                weights_adapt = [w * w_sum_orig / w_sum_adapt for w in weights_adapt]

            weighted_sum = sum(w * c for w, c in zip(weights_adapt, components))

            # Geometric mean bonus
            product = 1.0
            for c in components:
                product *= max(c, 1e-10)
            geo_mean = product ** (1.0 / len(components))
            bonus = 0.1 * geo_mean

            score = weighted_sum + bonus
            score = max(0.0, min(1.0, score))
            if not math.isfinite(score):
                return 0.0
            return score
        except Exception:
            return 0.0
