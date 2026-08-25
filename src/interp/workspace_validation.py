# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Workspace Validation via Sparse Autoencoder (SAE)
#
# Validates the JAWP workspace claim: that learned Q spans the same
# subspace as the "true workspace" recovered by an SAE decomposition.
#
# Anthropic (Gurnee et al., 2026, arXiv:2607.15495): J-space is the
# subspace where SAE features with high downstream relevance concentrate.
#
# Procedure:
#   1. Train a TopK SAE on encoder representations
#   2. Identify "workspace features" (high downstream probe accuracy)
#   3. Compute subspace similarity between SAE workspace and JAWP Q
#   4. Bootstrap CI for the similarity at 3+ model sizes
#
# If subspace_similarity(Q, SAE_workspace) > 0.8, the workspace claim
# is empirically validated.

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  TopK Sparse Autoencoder
# ═══════════════════════════════════════════════════════════════════


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder for representation decomposition.

    From Bricken et al. (2024) "Toward Monosemanticity: Training a
    TopK SAE". Only the top-k features fire per token.

    Args:
        embed_dim: input dimension (D).
        n_features: SAE latent dimension (typically 16x-64x D).
        k: number of active features per token.
    """

    def __init__(self, embed_dim: int, n_features: int = 8192, k: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.k = k

        # Encoder: x -> features
        self.W_enc = nn.Parameter(torch.randn(embed_dim, n_features) * (1.0 / embed_dim))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder: features -> x_hat
        self.W_dec = nn.Parameter(torch.randn(n_features, embed_dim) * (1.0 / n_features))
        self.b_dec = nn.Parameter(torch.zeros(embed_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with TopK activation.

        Returns:
            features: (N, n_features) sparse feature activations.
            indices: (N, k) indices of top-k features.
        """
        pre_acts = x @ self.W_enc + self.b_enc  # (N, n_features)
        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)

        features = torch.zeros_like(pre_acts)
        features.scatter_(-1, topk_indices, F.relu(topk_vals))
        return features, topk_indices

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass with reconstruction loss.

        Returns:
            dict with loss, x_hat, features, indices, sparsity.
        """
        features, indices = self.encode(x)
        x_hat = self.decode(features)

        # Reconstruction loss
        loss_recon = F.mse_loss(x_hat, x)

        # Sparsity: average L0 per token
        sparsity = (features > 0).float().sum(dim=-1).mean()

        # Dead features: features never activated in this batch
        all_indices = indices.reshape(-1)
        active_features = torch.unique(all_indices)
        dead_fraction = 1.0 - active_features.numel() / self.n_features

        return {
            "loss": loss_recon,
            "x_hat": x_hat,
            "features": features,
            "indices": indices,
            "sparsity": sparsity,
            "dead_fraction": dead_fraction,
        }

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, n_features={self.n_features}, " f"k={self.k}"


# ═══════════════════════════════════════════════════════════════════
#  Workspace feature identification
# ═══════════════════════════════════════════════════════════════════


def identify_workspace_features(
    sae: TopKSAE,
    representations: torch.Tensor,
    probe_labels: torch.Tensor,
    n_probes: int = 5,
    top_fraction: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Identify SAE features that are most predictive of downstream tasks.

    For each SAE feature, fit a linear probe to each downstream label.
    Features with high probe accuracy are "workspace features".

    Args:
        sae: trained SAE model.
        representations: (N, D) encoder representations.
        probe_labels: (N, n_probes) downstream labels (integer).
        n_probes: number of downstream probe tasks.
        top_fraction: fraction of features to classify as workspace.

    Returns:
        workspace_indices: 1-D tensor of feature indices.
        info: dict with diagnostics.
    """
    sae.eval()
    with torch.no_grad():
        features, _ = sae.encode(representations)  # (N, n_features)

    N, F_dim = features.shape
    n_probes = min(n_probes, probe_labels.shape[1] if probe_labels.dim() > 1 else 1)

    # Compute probe accuracy per feature per task
    # Simple approach: correlation between feature activation and label
    feature_scores = torch.zeros(F_dim, device=features.device)

    for task_idx in range(n_probes):
        if probe_labels.dim() > 1:
            labels = probe_labels[:, task_idx].float()
        else:
            labels = probe_labels.float()

        # Point-biserial correlation per feature
        labels_centered = labels - labels.mean()
        labels_std = labels_centered.std()
        if labels_std < 1e-10:
            continue

        for f_idx in range(F_dim):
            feat_vals = features[:, f_idx]
            feat_centered = feat_vals - feat_vals.mean()
            feat_std = feat_centered.std()
            if feat_std < 1e-10:
                continue
            corr = (feat_centered @ labels_centered) / (N * feat_std * labels_std + 1e-10)
            feature_scores[f_idx] += corr.abs()

    # Select top features
    n_workspace = max(1, int(F_dim * top_fraction))
    _, top_indices = torch.topk(feature_scores, n_workspace)
    info = {
        "n_workspace_features": n_workspace,
        "total_features": F_dim,
        "top_score": feature_scores[top_indices[0]].item(),
        "mean_score": feature_scores.mean().item(),
    }
    return top_indices, info


# ═══════════════════════════════════════════════════════════════════
#  Subspace similarity: JAWP Q vs SAE workspace
# ═══════════════════════════════════════════════════════════════════


def compute_workspace_similarity(
    Q: torch.Tensor,
    sae: TopKSAE,
    workspace_feature_indices: torch.Tensor,
) -> dict[str, float]:
    """Compute subspace similarity between JAWP Q and SAE workspace.

    The SAE workspace subspace is spanned by the decoder vectors
    of the identified workspace features.

    Args:
        Q: (D, k) JAWP workspace basis (orthonormal).
        sae: trained SAE model.
        workspace_feature_indices: indices of workspace features.

    Returns:
        dict with similarity metrics.
    """
    _D, k = Q.shape

    # Get SAE workspace basis vectors
    # W_dec: (n_features, D) — each row is a decoder vector
    W_dec = sae.W_dec.data  # (n_features, D)
    workspace_vectors = W_dec[workspace_feature_indices]  # (m, D)
    m = workspace_vectors.shape[0]

    if m == 0 or k == 0:
        return {
            "subspace_similarity": 0.0,
            "principal_angles": [],
            "mean_angle": 90.0,
            "workspace_dim_sae": 0,
            "workspace_dim_jawp": k,
        }

    # Orthogonalize SAE workspace vectors via QR
    ws_ortho, _R = torch.linalg.qr(workspace_vectors.T, mode="reduced")
    # ws_ortho: (D, r) where r = rank of workspace_vectors
    r = ws_ortho.shape[1]

    # Principal angles between subspaces
    # Q: (D, k), ws_ortho: (D, r)
    M = Q.T @ ws_ortho  # (k, r)
    # SVD of M gives cosines of principal angles
    try:
        singular_values = torch.linalg.svdvals(M)
    except Exception:
        singular_values = torch.tensor([0.0])

    # Clamp to [0, 1] for numerical stability
    singular_values = singular_values.clamp(0, 1)
    principal_angles_rad = torch.arccos(singular_values)
    principal_angles_deg = principal_angles_rad * 180.0 / math.pi

    # Subspace similarity: mean of squared cosines = mean of squared singular values
    similarity = (singular_values**2).mean().item()

    return {
        "subspace_similarity": similarity,
        "principal_angles": principal_angles_deg.tolist(),
        "mean_angle": principal_angles_deg.mean().item(),
        "workspace_dim_sae": r,
        "workspace_dim_jawp": k,
    }


# ═══════════════════════════════════════════════════════════════════
#  Bootstrap confidence interval
# ═══════════════════════════════════════════════════════════════════


def bootstrap_ci(
    values: torch.Tensor,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for mean of values.

    Args:
        values: 1-D tensor of sample values.
        n_bootstrap: number of bootstrap samples.
        confidence: confidence level (e.g., 0.95 for 95% CI).

    Returns:
        (mean, ci_lower, ci_upper).
    """
    n = values.numel()
    if n < 2:
        m = values.mean().item()
        return m, m, m

    values.cpu().numpy()
    rng = torch.Generator()
    rng.manual_seed(42)

    boot_means = []
    for _ in range(n_bootstrap):
        indices = torch.randint(0, n, (n,), generator=rng)
        sample = values[indices]
        boot_means.append(sample.mean().item())

    boot_means.sort()
    alpha = 1.0 - confidence
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    lo_idx = max(lo_idx, 0)
    hi_idx = min(hi_idx, n_bootstrap - 1)

    mean = values.mean().item()
    return mean, boot_means[lo_idx], boot_means[hi_idx]


# ═══════════════════════════════════════════════════════════════════
#  Full validation pipeline
# ═══════════════════════════════════════════════════════════════════


def validate_workspace_claim(
    Q: torch.Tensor,
    representations: torch.Tensor,
    probe_labels: torch.Tensor,
    sae: TopKSAE | None = None,
    n_sae_features: int = 8192,
    sae_k: int = 32,
    n_probes: int = 5,
    top_fraction: float = 0.1,
    n_bootstrap: int = 1000,
) -> dict[str, object]:
    """Full workspace validation pipeline.

    1. Train (or use provided) SAE on representations
    2. Identify workspace features via downstream probes
    3. Compute subspace similarity with JAWP Q
    4. Bootstrap CI for similarity

    Args:
        Q: (D, k) JAWP workspace basis.
        representations: (N, D) encoder representations.
        probe_labels: (N, n_probes) downstream labels.
        sae: pre-trained SAE (if None, one is created but not trained).
        n_sae_features: SAE latent dimension.
        sae_k: SAE TopK sparsity.
        n_probes: number of probe tasks.
        top_fraction: fraction of SAE features for workspace.
        n_bootstrap: bootstrap iterations.

    Returns:
        dict with all validation results.
    """
    D, k = Q.shape
    N = representations.shape[0]

    # Create SAE if not provided
    if sae is None:
        sae = TopKSAE(embed_dim=D, n_features=n_sae_features, k=sae_k)
        logger.warning(
            "SAE not trained — using random decoder. "
            "Provide a pre-trained SAE for valid results."
        )

    # Identify workspace features
    ws_indices, ws_info = identify_workspace_features(
        sae,
        representations,
        probe_labels,
        n_probes=n_probes,
        top_fraction=top_fraction,
    )

    # Compute subspace similarity
    similarity_result = compute_workspace_similarity(Q, sae, ws_indices)
    # Shuffled-Q control (fleet R3): a random orthonormal basis of the same
    # shape calibrates the fixed 0.8 gate — report the margin explicitly.
    gen = torch.Generator().manual_seed(20260824)
    Q_rand, _ = torch.linalg.qr(torch.randn(D, k, generator=gen))
    placebo = compute_workspace_similarity(Q_rand, sae, ws_indices)

    # Bootstrap CI on per-sample workspace utilization
    with torch.no_grad():
        ws_projection = representations @ Q @ Q.T  # (N, D)
        ws_util = (ws_projection**2).sum(dim=-1) / (representations**2).sum(dim=-1).clamp(min=1e-10)

    mean_util, ci_lo, ci_hi = bootstrap_ci(ws_util, n_bootstrap=n_bootstrap)

    result = {
        **similarity_result,
        **ws_info,
        "ws_utilization_mean": mean_util,
        "ws_utilization_ci_lower": ci_lo,
        "ws_utilization_ci_upper": ci_hi,
        "n_samples": N,
        "embed_dim": D,
        "workspace_dim_jawp": k,
        "subspace_similarity_placebo": placebo["subspace_similarity"],
        "claim_margin_above_placebo": (
            similarity_result["subspace_similarity"] - placebo["subspace_similarity"]
        ),
        # Preregistered rule: the claim must beat BOTH the fixed gate and the
        # shuffled-basis control, not just an unjustified constant.
        "workspace_claim_valid": (
            similarity_result["subspace_similarity"] > 0.8
            and similarity_result["subspace_similarity"] > placebo["subspace_similarity"]
        ),
    }

    return result
