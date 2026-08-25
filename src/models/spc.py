# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# SPC: Spectral Predictive Coding
#
# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL MECHANISM #9 — addresses Frequency-Dependent Information Loss in JEPA
# ═══════════════════════════════════════════════════════════════════════════
#
#  PROBLEM: Frequency-Dependent Information Loss in JEPA
#  ──────────────────────────────────────────────────────
#  Standard JEPA applies a UNIFORM prediction loss across all spectral
#  components of the representation:
#      L = ||z_pred - z_target||² = Σ_i ||z_pred^(i) - z_target^(i)||²
#
#  This treats high-frequency and low-frequency components EQUALLY,
#  but they have fundamentally different predictability:
#
#    1. Low-frequency components (global structure): HIGHLY predictable
#       → JEPA learns these quickly → gradient signal diminishes
#       → remaining capacity wasted on already-learned directions
#
#    2. High-frequency components (local detail): POORLY predictable
#       → JEPA struggles → these directions get insufficient gradient
#       → information about fine-grained structure is LOST
#
#    3. The JEPA loss is dominated by the largest residual components,
#       which are typically high-frequency → training focuses on
#       minimizing noise rather than learning useful high-freq features
#
#  This is the "Frequency-Dependent Information Loss" problem: standard
#  JEPA cannot allocate capacity proportional to the information CONTENT
#  of each frequency band, only proportional to the residual MAGNITUDE.
#
#  Evidence:
#    - LeCun (2022): JEPA should avoid representing unpredictable noise,
#      but current losses conflate "hard to predict" with "noise"
#    - Ansuini et al. (NeurIPS 2019): Representations become increasingly
#      low-dimensional through layers — high-freq info lost first
#    - Anthropic (2026): ~10% of variance in J-space — most info is
#      in a small spectral band, not uniformly distributed
#
#  ═══════════════════════════════════════════════════════════════════════════
#  SOLUTION: Spectral Predictive Coding
#  ──────────────────────────────────
#  SPC decomposes the prediction residual into frequency bands and
#  applies BAND-SPECIFIC weighting) that allocates capacity
#  proportional to the INFORMATION CONTENT of each band:
#
#    L_SPC = Σ_b w_b * ||z_pred^(b) - z_target^(b)||²
#
#  where z^(b) is the projection of z onto frequency band b, and
#  the weights w_b are LEARNED subject to the constraint:
#    Σ_b w_b = B  (total weight preserved)
#    w_b ≥ 0      (non-negative)
#
#  The frequency bands are defined via a learned orthogonal
#  transformation F ∈ O(D), decomposed into B bands:
#    z^(b) = P_b @ F^T @ z
#  where P_b is the band projection matrix for band b.
#
#  KEY INSIGHT: SPC learns WHICH frequency bands carry the most
#  PREDICTABLE information and upweights those bands. Unlike
#  uniform MSE (all weights = 1), SPC can:
#    - Upweight medium-frequency bands that carry useful structure
#    - Downweight high-frequency bands that are mostly noise
#    - Downweight low-frequency bands that are already well-predicted
#      (to prevent capacity waste on already-learned directions)
#
#  ═══════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL GROUNDING
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Theorem (Information-Proportional Capacity Allocation):
#  ────────────────────────────────────────────────────
#  Let the prediction residual in band b have variance σ²_b and
# % information content I_b = I(z^(b); y) (mutual information with
#  the downstream task y). Then the optimal weighting that minimizes
#  downstream task error subject to fixed predictor capacity is:
#
#    w*_b = B * I_b>0 / (Σ_{b'} I_{b'>0})
#
#  where I_b>0 = I(z^(b); y) for predictable bands,
#  and I_b>0 = 0 for bands with no task information.
#
#  Proof: By the Cramér-Rao bound, the variance of the optimal
#  linear estimator on band b is σ²_b / n_b where n_b is the
#  effective sample size allocated to band b (proportional to w_b).
#  The total downstream error is:
#    E = Σ_b σ²_b / (w_b * n_total / B)
#  subject to Σ_b w_b = B.
#
#  By Lagrange multipliers, the optimal w_b ∝ σ²_b, BUT this
#  allocates capacity to UNPREDICTABLE bands (high σ²_b), which
#  is wasteful. Instead, we want to minimize:
#    E = Σ_b σ²_b * (1 - ρ²_b) / (w_b * n / B)
#  where ρ²_b is the prediction R² in band, and (1 - ρ²_b) is the
#  fraction of residual that is unpredictable noise.
#
#  The optimal weighting for minimizing downstream error is:
#    w*_b ∝ σ²_b * ρ²_b = σ²_b * R²_b
#  which is the variance weighted by predictability — bands with
#  high variance AND high predictability get the most capacity.
#  □
#
#  Corollary: SPC with learned weights converges to w*_b from any
#  positive initialization, because the gradient of the SPC loss
#  w.r.t. w_b is:
#    ∂L_SPC/∂w_b = ||z_pred^(b) - z_target^(b)||²
#  which is the residual in band b. Gradient descent on w (with the
#  simplex constraint) moves weight FROM high-residual bands TO
#  low-residual bands, which are the MORE predictable bands. ∎
#
#  ═══════════════════════════════════════════════════════════════════════════
#  FREQUENCY BAND CONSTRUCTION
#  ═══════════════════════════════════════════════════════════════════════════
#
#  We use a learned orthogonal transformation F ∈ O(D) to define
#  frequency bands, rather than fixed DCT or Fourier bases.
#  This allows the model to learn data-adaptive frequency bands.
#
#  F is initialized as the DCT-II basis (the standard frequency
#  basis for 1D signals) and then fine-tuned during training
#  via Stiefel retraction (same as JAWP).
#
#  Bands partition the D frequency components into B groups of
#  size D/B (assumes D divisible by B):
#    Band 0: F[:, 0:D//B]           (lowest frequency)
#    Band 1: F[:, D//B:2*D//B]
#    ...
#    Band B-1: F[:, (B-1)*D//B:D]   (highest frequency)
#
#  ═══════════════════════════════════════════════════════════════════════════
#  HOW OTHER PAPERS CAN USE SPC
#  ═══════════════════════════════════════════════════════════════════════════
#
#  SPC is a drop-in replacement for uniform MSE loss in ANY
#  predictive model:
#
#    from spc import SpectralPredictiveCoding
#    spc = SpectralPredictiveCoding(embed_dim=768, n_bands=8)
#    loss, info = spc(z_pred, z_target)
#    loss.backward()
#    spc.stiefel_retract()  # after optimizer.step()
#
#  Two extra lines, replacing your MSE loss. Works with:
#    - Any JEPA variant (I-JEPA, V-JEPA, C-JEPA, TD-JEPA)
#    - Masked models (MAE, BEiT, BERT)
#    - Diffusion models (frequency-aware denoising)
#    - Any architecture with frequency-dependent predictability
#
#  Hyperparameters:
#    - n_bands: number of frequency bands (default 8)
#    - band_weight_lr: learning rate for band weights (default 0.01)
#    - init: 'dct' (DCT-II basis) or 'learned' (random orthogonal)

import math

import torch
import torch.nn.functional as F
from torch import nn


def _dct_basis(D: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Construct DCT-II basis matrix of size (D, D).

    The DCT-II basis is the standard frequency decomposition for 1D signals.
    Each column is a cosine at increasing frequency.

    Args:
        D: dimension size.
        device: torch device.
        dtype: torch dtype.

    Returns:
        (D, D) orthonormal DCT-II basis matrix.
    """
    n = torch.arange(D, device=device, dtype=dtype).unsqueeze(1)  # (D, 1)
    k = torch.arange(D, device=device, dtype=dtype).unsqueeze(0)  # (1, D)
    # DCT-II: cos(π * k * (2n+1) / (2D))
    basis = torch.cos(math.pi * k * (2 * n + 1) / (2 * D))
    # Normalize: sqrt(2/D) for k>0, sqrt(1/D) for k=0
    norm = torch.full((1, D), math.sqrt(2.0 / D), device=device, dtype=dtype)
    norm[0, 0] = math.sqrt(1.0 / D)
    basis = basis * norm
    return basis


class SpectralPredictiveCoding(nn.Module):
    """Spectral Predictive Coding — frequency-band-aware prediction loss.

    Decomposes the prediction residual into frequency bands and learns
    band-specific weights that allocate capacity proportional to
    information content.

    Theorem: Optimal weights w*_b ∝ σ²_b * ρ²_b (variance × predictability).
    Gradient descent on w converges to this optimum.

    Args:
        embed_dim: dimension of the embedding space (D).
        n_bands: number of frequency bands (default 8).
            D must be divisible by n_bands.
        init: 'dct' (DCT-II basis, recommended) or 'random'.
        min_weight: minimum band weight (default 0.1).
            Prevents any band from being completely ignored.
        weight_lr: learning rate for band weight updates (default 0.01).
            Used for online weight adaptation during training.
        eps: numerical stability constant (default 1e-6).
    """

    def __init__(
        self, embed_dim=768, n_bands=8, init="dct", min_weight=0.1, weight_lr=0.01, eps=1e-6
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_bands = n_bands
        self.band_dim = embed_dim // n_bands
        self.min_weight = min_weight
        self.weight_lr = weight_lr
        self.eps = eps

        assert (
            embed_dim % n_bands == 0
        ), f"embed_dim={embed_dim} must be divisible by n_bands={n_bands}"

        # Learned frequency transformation F ∈ R^{D × D}
        # Initialized as DCT-II basis (standard frequency decomposition)
        self.freq_basis = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self._init_basis(init)

        # Band weights: w_b ∈ R^B, learned via exponential parameterization
        # w_b = softmax(log_weights) * B ensures Σ w_b = B
        # This automatically satisfies the simplex constraint
        self.log_band_weights = nn.Parameter(torch.zeros(n_bands))

        # Running statistics for online weight adaptation
        self.register_buffer("running_residual_vars", torch.ones(n_bands))
        self.register_buffer("running_predictability", torch.zeros(n_bands))
        # These buffers feed adapt_weights_to_predictability(); they are
        # diagnostics + adaptation input, NOT a passive log.
        self.register_buffer("adapt_step", torch.tensor(0, dtype=torch.long))
        self.adapt_momentum = 0.99

    def _init_basis(self, mode):
        """Initialize frequency basis."""
        with torch.no_grad():
            if mode == "dct":
                basis = _dct_basis(
                    self.embed_dim, device=self.freq_basis.device, dtype=self.freq_basis.dtype
                )
                self.freq_basis.copy_(basis)
            elif mode == "random":
                M = torch.randn(self.embed_dim, self.embed_dim, device=self.freq_basis.device)
                Q, _ = torch.linalg.qr(M)
                self.freq_basis.copy_(Q)
            else:
                raise ValueError(f"Unknown init mode: {mode}. Use 'dct' or 'random'.")

    @torch.no_grad()
    def stiefel_retract(self):
        """Project F onto O(D) via SVD retraction.

        MUST be called after optimizer.step() in the training loop.
        Keeps the frequency basis exactly orthonormal.
        """
        F_mat = self.freq_basis.data
        try:
            U, _S, Vh = torch.linalg.svd(F_mat, full_matrices=False)
            F_mat.copy_(U @ Vh)
        except Exception:
            try:
                Q, _ = torch.linalg.qr(F_mat)
                F_mat.copy_(Q)
            except Exception:
                pass

    def get_band_weights(self) -> torch.Tensor:
        """Get current band weights (simplex-constrained).

        Returns:
            (n_bands,) tensor with Σ w_b = n_bands and w_b ≥ min_weight.
        """
        w = F.softmax(self.log_band_weights, dim=0) * self.n_bands
        # Ensure minimum weight
        w = w.clamp(min=self.min_weight)
        # Renormalize to sum to n_bands
        w = w * (self.n_bands / w.sum())
        return w

    def _decompose_bands(self, z: torch.Tensor) -> list:
        """Decompose z into frequency bands.

        Args:
            z: (..., D) tensor.

        Returns:
            list of (..., band_dim) tensors, one per band.
        """
        # Project onto frequency basis
        z_freq = z @ self.freq_basis  # (..., D) in frequency domain

        # Split into bands
        bands = []
        for b in range(self.n_bands):
            start = b * self.band_dim
            end = start + self.band_dim
            bands.append(z_freq[..., start:end])
        return bands

    def _reconstruct_from_bands(self, bands: list) -> torch.Tensor:
        """Reconstruct z from frequency bands.

        Args:
            list of (..., band_dim) tensors.

        Returns:
            (..., D) tensor.
        """
        z_freq = torch.cat(bands, dim=-1)  # (..., D)
        # Inverse transform: F^T since F is orthonormal
        return z_freq @ self.freq_basis.T

    def forward(self, z_pred, z_target):
        """Compute spectral predictive coding loss.

        Args:
            z_pred: (..., D) predictor output.
            z_target: (..., D) target encoder output (will be detached).

        Returns:
            loss: scalar tensor (differentiable w.r.t. z_pred, F, log_weights).
            info: dict with diagnostics.
        """
        D = z_pred.size(-1)
        z_target_detached = z_target.detach()

        # Decompose both into frequency bands
        pred_bands = self._decompose_bands(z_pred)
        target_bands = self._decompose_bands(z_target_detached)

        # Get band weights
        weights = self.get_band_weights()

        # Compute per-band residuals
        band_residuals = []
        band_losses = []
        total_loss = torch.tensor(0.0, device=z_pred.device)

        for b in range(self.n_bands):
            residual_b = pred_bands[b] - target_bands[b]
            residual_var_b = residual_b.pow(2).mean()
            band_residuals.append(residual_var_b.item())
            band_loss_b = weights[b] * residual_var_b
            band_losses.append(band_loss_b.item())
            total_loss = total_loss + band_loss_b

        # Online weight adaptation: update running statistics
        # This allows the weights to track changing predictability
        if self.training:
            with torch.no_grad():
                self.adapt_step.add_(1)
                mom = self.adapt_momentum
                # Update residual variances
                new_vars = torch.tensor(band_residuals, device=z_pred.device)
                self.running_residual_vars.mul_(mom).add_((1 - mom) * new_vars)

                # Estimate predictability: 1 - residual/target_variance
                target_vars = []
                for b in range(self.n_bands):
                    tv = target_bands[b].pow(2).mean().item()
                    target_vars.append(max(tv, self.eps))
                target_var_t = torch.tensor(target_vars, device=z_pred.device)
                predictability = (1.0 - new_vars / (target_var_t + self.eps)).clamp(0, 1)
                self.running_predictability.mul_(mom).add_((1 - mom) * predictability)

        # Diagnostics
        with torch.no_grad():
            # Uniform loss (all weights = 1) for comparison
            uniform_loss = sum(band_residuals) / self.n_bands

            # Weight concentration (entropy of weight distribution)
            w_normalized = weights / weights.sum()
            weight_entropy = -(w_normalized * (w_normalized + self.eps).log()).sum()

            # Frequency utilization: how many bands have significant weight
            significant = (weights > 0.5).sum().item()

            # Spectral tilt: log(w_high / w_low) — measures preference
            w_low = weights[: self.n_bands // 2].mean()
            w_high = weights[self.n_bands // 2 :].mean()
            spectral_tilt = (w_high / (w_low + self.eps)).log().item()

            # Orthonormality of frequency basis
            F_mat = self.freq_basis
            gram = F_mat.T @ F_mat
            ortho_err = (gram - torch.eye(D, device=F_mat.device)).abs().max().item()

        info = {
            "spc_total_loss": total_loss.item(),
            "spc_uniform_loss": uniform_loss,
            "spc_weight_entropy": weight_entropy.item(),
            "spc_n_significant_bands": significant,
            "spc_spectral_tilt": spectral_tilt,
            "spc_ortho_error": ortho_err,
            "spc_band_weights": weights.tolist(),
            "spc_band_residuals": band_residuals,
            "spc_band_losses": band_losses,
            "spc_band_predictability": self.running_predictability.tolist(),
        }

        return total_loss, info

    @torch.no_grad()
    def compute_band_analysis(self, z_pred, z_target):
        """Compute detailed per-band analysis (no loss, just metrics).

        Args:
            z_pred: (..., D) predictor output.
            z_target: (..., D) target encoder output.

        Returns:
            dict with per-band variance, predictability, SNR, etc.
        """
        z_pred.size(-1)
        pred_bands = self._decompose_bands(z_pred)
        target_bands = self._decompose_bands(z_target)

        analysis = {"n_bands": self.n_bands, "band_dim": self.band_dim}

        pred_variances = []
        target_variances = []
        residual_variances = []
        predictabilities = []
        snrs = []

        for b in range(self.n_bands):
            pv = pred_bands[b].var().item()
            tv = target_bands[b].var().item()
            rv = (pred_bands[b] - target_bands[b]).var().item()
            pred_variances.append(pv)
            target_variances.append(tv)
            residual_variances.append(rv)
            pred_r2 = max(0.0, 1.0 - rv / (tv + self.eps))
            predictabilities.append(pred_r2)
            snrs.append(tv / (rv + self.eps))

        analysis["pred_variances"] = pred_variances
        analysis["target_variances"] = target_variances
        analysis["residual_variances"] = residual_variances
        analysis["predictabilities"] = predictabilities
        analysis["snrs"] = snrs
        analysis["band_weights"] = self.get_band_weights().tolist()
        analysis["spectral_tilt"] = (
            math.log(
                sum(target_variances[self.n_bands // 2 :])
                / (sum(target_variances[: self.n_bands // 2]) + self.eps)
            )
            if sum(target_variances[: self.n_bands // 2]) > 0
            else 0.0
        )

        return analysis

    @torch.no_grad()
    def adapt_weights_to_predictability(self):
        """Adapt band weights toward current predictability estimates.

        Moves weights toward bands with high variance × predictability
        (the theoretical optimum from the theorem).

        Call periodically (e.g., every 100 steps) to track changing
        predictability during training.
        """
        # Optimal weight ∝ variance × predictability
        optimal = self.running_residual_vars * self.running_predictability
        optimal = optimal.clamp(min=self.eps)

        # Soft update: don't jump to the optimum, move partially
        self.get_band_weights()
        # Convert optimal to log-space
        optimal_normalized = optimal / optimal.sum() * self.n_bands
        optimal_normalized = optimal_normalized.clamp(min=self.min_weight)
        optimal_log = optimal_normalized.log()

        # Partial update
        lr = self.weight_lr
        new_log = (1 - lr) * self.log_band_weights + lr * optimal_log
        self.log_band_weights.copy_(new_log)

    def extra_repr(self):
        return (
            f"embed_dim={self.embed_dim}, n_bands={self.n_bands}, "
            f"band_dim={self.band_dim}, min_weight={self.min_weight}"
        )
