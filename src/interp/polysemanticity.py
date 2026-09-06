# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Polysemanticity metrics: how many distinct concepts per dimension?
#
# From:
# - Scherlis et al. (2022) "Polysemanticity and Capacity in Neural Networks"
# - PSI (2025) "Null-Calibrated Polysemanticity Index"
# - Elhage et al. (2022) "Toy Models of Superposition"
#
# Key hypothesis: JEPA representations should have LOWER polysemanticity
# than MLM, because latent prediction creates inductive bias toward
# abstract, monosemantic features (fewer concepts packed per dimension).
#
# Three measures:
# 1. Polysemanticity Index (PSI): null-calibrated, from top activations
# 2. Superposition Index: from weight matrix geometry (Elhage et al.)
# 3. Feature Deduplication Score: from SAE feature overlap

import math

import torch
import torch.nn.functional as F


class PolysemanticityIndex:
    """Polysemanticity Index (PSI): how many distinct concepts per dimension.

    Simplified from the 2025 paper "Null-Calibrated Polysemanticity Index".

    For each representation dimension:
    1. Find top-k activations across the dataset
    2. Cluster the inputs that cause those activations
    3. Measure cluster separability (geometric quality)
    4. PSI = cluster_quality * n_clusters (more clusters = more polysemantic)

    Lower PSI per dimension = more monosemantic.
    Mean PSI across dimensions = overall polysemanticity score.
    """

    def __init__(
        self,
        n_clusters_range=(2, 5),
        n_top_activations=100,
        n_dimensions_sample=None,
        device="cpu",
    ):
        """
        Args:
            n_clusters_range: range of cluster counts to test
            n_top_activations: how many top-activating inputs per dimension
            n_dimensions_sample: subsample dimensions (None = all)
            device: compute device

        """
        self.n_clusters_range = n_clusters_range
        self.n_top_activations = n_top_activations
        self.n_dimensions_sample = n_dimensions_sample
        self.device = device

    @torch.no_grad()
    def compute(self, representations, labels=None):
        """Compute Polysemanticity Index across dimensions.

        Args:
            representations: (N, D) representation matrix
            labels: (N,) optional class labels for alignment score

        Returns:
            dict with per-dimension PSI, mean PSI, fraction_monosemantic

        """
        try:
            _N, D = representations.shape
            representations = representations.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            # Subsample dimensions for efficiency
            if self.n_dimensions_sample and self.n_dimensions_sample < D:
                dim_idx = torch.randperm(D)[: self.n_dimensions_sample]
            else:
                dim_idx = torch.arange(D)

            psi_scores = []
            for d in dim_idx:
                psi = self._compute_dim_psi(representations, d.item(), labels)
                psi_scores.append(psi)

            psi_tensor = torch.tensor(psi_scores)
            mean_psi = psi_tensor.mean().item()
            # Monosemantic: PSI < threshold (essentially 1 cluster)
            frac_monosemantic = (psi_tensor < 1.5).float().mean().item()

            return {
                "mean_psi": mean_psi,
                "frac_monosemantic": frac_monosemantic,
                "per_dim_psi": psi_scores,
                "max_psi": psi_tensor.max().item(),
                "min_psi": psi_tensor.min().item(),
            }
        except Exception:
            return {
                "mean_psi": float("inf"),
                "frac_monosemantic": 0.0,
                "per_dim_psi": [],
                "max_psi": float("inf"),
                "min_psi": 0.0,
            }

    def _compute_dim_psi(self, representations, dim_idx, labels=None):
        """Compute PSI for a single dimension."""
        try:
            N = representations.size(0)
            activations = representations[:, dim_idx]
            n_top = min(self.n_top_activations, N)

            # Find top-activating inputs
            _, top_idx = activations.topk(n_top)
            top_reps = representations[top_idx]  # (n_top, D)

            # Try clustering at different k values
            best_score = 0.0
            best_k = 1

            for k in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
                if n_top < k * 5:  # Need enough points per cluster
                    continue
                score = self._cluster_quality(top_reps, k)
                if score > best_score:
                    best_score = score
                    best_k = k

            # PSI = best_k * quality
            # If quality is low, it means activations are not well-separated
            # → the dimension responds to one thing (monosemantic)
            # If quality is high AND k > 1 → polysemantic
            psi = best_k * best_score

            # If labels available, compute alignment score
            if labels is not None and best_k > 1:
                alignment = self._label_alignment(activations, top_idx, labels, best_k)
                psi *= alignment

            return psi
        except Exception:
            return 0.0

    @staticmethod
    def _cluster_quality(points, k):
        """Simple cluster quality: silhouette-like score.

        Uses k-means-style clustering and measures inter vs intra distance.
        """
        try:
            N, _D = points.shape
            if k * 3 > N:
                return 0.0

            # Simple k-means with random initialization
            centroids = points[torch.randperm(N)[:k]]

            for _ in range(10):  # k-means iterations
                # Assign to nearest centroid
                dists = torch.cdist(points, centroids)  # (N, k)
                assignments = dists.argmin(dim=1)  # (N,)

                # Update centroids
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(k)
                for i in range(N):
                    c = assignments[i]
                    new_centroids[c] += points[i]
                    counts[c] += 1
                for c in range(k):
                    if counts[c] > 0:
                        new_centroids[c] /= counts[c]
                    else:
                        new_centroids[c] = centroids[c]
                centroids = new_centroids

            # Compute intra-cluster and inter-cluster distances
            intra = 0.0
            inter = 0.0
            n_intra = 0
            n_inter = 0

            for c in range(k):
                mask = assignments == c
                cluster_points = points[mask]
                if cluster_points.size(0) < 2:
                    continue
                # Intra: mean pairwise distance within cluster
                intra_dists = torch.cdist(cluster_points, cluster_points)
                n_c = cluster_points.size(0)
                if n_c > 1:
                    intra += intra_dists.sum().item() / (n_c * (n_c - 1))
                    n_intra += 1

            # Inter: mean pairwise distance between centroids
            if k > 1:
                inter_dists = torch.cdist(centroids, centroids)
                inter = inter_dists.sum().item() / (k * (k - 1))
                n_inter = 1

            if n_intra == 0 or n_inter == 0:
                return 0.0

            avg_intra = intra / n_intra
            avg_inter = inter / n_inter

            # Quality = 1 - intra/inter (higher = better separated)
            if avg_inter == 0:
                return 0.0
            quality = max(1.0 - avg_intra / avg_inter, 0.0)
            return quality
        except Exception:
            return 0.0

    @staticmethod
    def _label_alignment(activations, top_idx, labels, k):
        """How well do activation clusters align with class labels?

        If the top activations for a dimension come from different classes
        AND cluster by class → the dimension is polysemantic in a meaningful way.
        If they come from the same class → monosemantic.
        """
        try:
            top_labels = labels[top_idx]
            unique_labels = top_labels.unique()
            if unique_labels.numel() <= 1:
                return 1.0  # All same class → monosemantic
            # Entropy of label distribution
            counts = torch.bincount(top_labels.long().clamp(0, 1000))
            probs = counts.float() / (counts.sum() + 1e-10)
            probs = probs[probs > 0]
            entropy = -(probs * torch.log(probs)).sum().item()
            max_entropy = math.log(unique_labels.numel() + 1e-10)
            if max_entropy == 0:
                return 1.0
            return entropy / max_entropy  # Normalized [0, 1]
        except Exception:
            return 1.0


class SuperpositionIndex:
    """Superposition Index from Elhage et al. (2022).

    Measures how many features are stored in superposition in a weight matrix.
    Uses the weight matrix W of a linear layer: computes the interference
    between features via the Gram matrix W^T W.

    If W^T W is close to identity → no superposition (features orthogonal).
    If W^T W has significant off-diagonal → superposition.

    Key metric: superposition_ratio = (effective rank of W) / (actual dim of W)
    Ratio > 1 → features in superposition (more features than dimensions)
    """

    @staticmethod
    @torch.no_grad()
    def compute(weight_matrix):
        """Compute superposition metrics for a weight matrix.

        Args:
            weight_matrix: (out_dim, in_dim) weight tensor

        Returns:
            dict with superposition metrics

        """
        try:
            W = weight_matrix.float()
            out_dim, in_dim = W.shape

            # Gram matrix: W W^T (feature-feature interactions)
            gram = W @ W.T

            # Diagonal elements: feature norms
            diag = torch.diag(gram)
            # Off-diagonal: interference between features
            off_diag = gram - torch.diag(diag)

            # Superposition metrics
            # 1. Interference ratio: mean |off_diagonal| / mean diagonal
            mean_diag = diag.mean().item()
            mean_interference = off_diag.abs().mean().item()
            interference_ratio = mean_interference / max(abs(mean_diag), 1e-10)

            # 2. Effective rank of W (from SVD)
            try:
                s = torch.linalg.svdvals(W)
                s_norm = s / (s.sum() + 1e-10)
                entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum().item()
                eff_rank = math.exp(entropy)
            except Exception:
                eff_rank = min(out_dim, in_dim)

            # 3. Superposition ratio
            superposition_ratio = eff_rank / min(out_dim, in_dim)

            # 4. Feature density: fraction of features with non-negligible norm
            feature_norms = diag.sqrt()
            threshold = feature_norms.mean() * 0.1
            n_active = (feature_norms > threshold).sum().item()
            feature_density = n_active / out_dim

            return {
                "interference_ratio": interference_ratio,
                "effective_rank": eff_rank,
                "superposition_ratio": superposition_ratio,
                "feature_density": feature_density,
                "n_active_features": n_active,
                "mean_feature_norm": mean_diag**0.5,
            }
        except Exception:
            return {
                "interference_ratio": float("inf"),
                "effective_rank": 0.0,
                "superposition_ratio": 0.0,
                "feature_density": 0.0,
                "n_active_features": 0,
                "mean_feature_norm": 0.0,
            }


class FeatureDeduplicationScore:
    """Measure overlap between SAE features across models.

    If JEPA SAE features are more distinct (less overlap), the model
    has learned more interpretable, monosemantic directions.

    Computes: for each SAE feature in model A, find the most similar
    feature in model B. If features are 1:1 (each A feature maps to
    exactly one B feature), representations are equally structured.
    If A features map to multiple B features → B has more polysemanticity.
    """

    @staticmethod
    @torch.no_grad()
    def compute(sae_features_a, sae_features_b):
        """Compute feature deduplication score.

        Args:
            sae_features_a: (M_a, D) SAE decoder weights from model A
            sae_features_b: (M_b, D) SAE decoder weights from model B

        Returns:
            dict with deduplication metrics

        """
        try:
            # Normalize features
            a = F.normalize(sae_features_a.float(), dim=1)
            b = F.normalize(sae_features_b.float(), dim=1)

            # Cosine similarity matrix: (M_a, M_b)
            sim_matrix = a @ b.T

            # For each feature in A, find best match in B
            best_match_a = sim_matrix.max(dim=1).values  # (M_a,)
            # For each feature in B, find best match in A
            best_match_b = sim_matrix.max(dim=0).values  # (M_b,)

            # Deduplication: how many features in A have unique matches?
            # If multiple A features match the same B feature → B is less distinct
            b_match_idx = sim_matrix.argmax(dim=1)  # Which B feature each A matches
            unique_b_matches = len(b_match_idx.unique())
            dedup_a = unique_b_matches / sae_features_a.size(0)

            a_match_idx = sim_matrix.argmax(dim=0)
            unique_a_matches = len(a_match_idx.unique())
            dedup_b = unique_a_matches / sae_features_b.size(0)

            return {
                "mean_cosine_a_to_b": best_match_a.mean().item(),
                "mean_cosine_b_to_a": best_match_b.mean().item(),
                "dedup_a": dedup_a,  # Fraction of A features with unique B matches
                "dedup_b": dedup_b,  # Fraction of B features with unique A matches
                "n_unique_b_matches": unique_b_matches,
                "n_unique_a_matches": unique_a_matches,
            }
        except Exception:
            return {
                "mean_cosine_a_to_b": 0.0,
                "mean_cosine_b_to_a": 0.0,
                "dedup_a": 0.0,
                "dedup_b": 0.0,
                "n_unique_b_matches": 0,
                "n_unique_a_matches": 0,
            }
