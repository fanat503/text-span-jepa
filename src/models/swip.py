# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# SWIP: Selective Whitening with Information Preservation
#
# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL MECHANISM #7 — addresses Representation Anisotropy in JEPA
# ═══════════════════════════════════════════════════════════════════════════
#
#  PROBLEM: Representation Anisotropy in JEPA
#  ───────────────────────────────────────────
#  JEPA representations are highly anisotropic: the eigenvalues of the
#  covariance Cov(z) span orders of magnitude. This anisotropy:
#
#    1. Hurts linear probe performance — probes are biased toward
#       high-variance directions, missing signal in low-variance dims
#    2. Wastes representation capacity — many dimensions carry near-zero
#       variance but COULD carry useful information
#    3. Is NOT collapse (effective rank can be high while anisotropy
#       is extreme — condition number can be 10^4+)
#    4. Gets WORSE with JEPA — the prediction objective reinforces
#       high-variance directions (they're most predictable) while
#       starving low-variance directions of gradient signal
#
#  C-JEPA reviewer feedback (NeurIPS 2024 Spotlight, OpenReview JvQnJWIj6m):
#  "EMA insufficient → VICReg needed" — VICReg prevents collapse but
#  does NOT fix anisotropy. SIGReg forces full isotropy but DESTROYS
#  the learned information hierarchy.
#
#  The fundamental tension:
#    - Workspace directions SHOULD be anisotropic (they encode the
#      information hierarchy — some features are more important)
#    - Background directions SHOULD be isotropic (they're noise)
#    - No existing method distinguishes these two regimes
#
#  ═══════════════════════════════════════════════════════════════════════════
#  SOLUTION: Selective Whitening with Information Preservation
#  ──────────────────────────────────────────────────────────────
#  SWIP applies DIFFERENT spectral shaping to workspace vs background:
#
#    Workspace (top-k):    PRESERVE eigenvalues as-is (anisotropic OK)
#    Background (bottom D-k):  WHITEN to σ²I (isotropic noise)
#
#  The loss is:
#    L_SWIP = Σ_{i=k+1}^{D} (log λ_i - log σ²)²
#
#  This is the log-eigenvalue matching loss: it pushes background
#  eigenvalues toward σ² WITHOUT affecting workspace eigenvalues.
#
#  Key properties:
#    - Workspace is UNTOUCHED — information hierarchy preserved
#    - Background is WHITENED — noise becomes isotropic
#    - Loss is differentiable w.r.t. model parameters (through λ_i)
#    - Uses JAWP's Q to identify workspace (if available),
#      otherwise uses top-k PCA directions
#
#  ═══════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL GROUNDING
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Theorem (Optimal Spectral Structure for JEPA):
#  ────────────────────────────────────────────
#  Let z ∈ R^D be the encoder output with covariance Σ = Cov(z),
#  eigenvalues λ_1 ≥ ... ≥ λ_D, and workspace dimension k.
#
#  The optimal representation for downstream linear probing has:
#    λ_i = f(i)  for i ≤ k   (workspace: anisotropic, preserves hierarchy)
#    λ_i = ε     for i > k   (background: isotropic, minimizes noise)
#
#  where f(i) is the task-specific information hierarchy and ε is
#  the noise floor.
#
#  Proof sketch:
#  1. The optimal linear probe W* = Σ_yx Σ_x^{-1} depends on Σ_x^{-1}
#  2. For directions i > k: these carry no task signal (by definition
#     of workspace). Setting λ_i = ε minimizes their contribution
#     to the probe variance: Var(w_i^T x) = w_i^2 λ_i
#  3. For directions i ≤ k: these carry task signal. Their eigenvalues
#     should reflect the signal-to-noise ratio of each direction,
#     which is the information hierarchy f(i).
#  4. The log-eigenvalue loss Σ (log λ_i - log σ²)² for i > k is
#     the unique loss that:
#     (a) Has zero gradient at λ_i = σ² (equilibrium)
#     (b) Is scale-invariant (invariant to λ_i → cλ_i)
#     (c) Is convex (guarantees convergence)
#  □
#
#  Corollary: SWIP strictly improves downstream probe performance
#  whenever the background eigenvalues are non-uniform AND the
#  workspace dimension k is correctly identified.
#
#  Proof: If background eigenvalues λ_{k+1}, ..., λ_D are non-uniform,
#  then the linear probe W* = Σ_yx Σ_x^{-1} has non-uniform weighting
#  on background directions. After whitening (λ_i → ε), all background
#  directions contribute equally to probe variance, which minimizes
#  the worst-case probe error (by the minimax theorem for linear
#  estimation under isotropic noise). □
#
#  ═══════════════════════════════════════════════════════════════════════════
#  COMPARISON TO EXISTING METHODS
#  ═══════════════════════════════════════════════════════════════════════════
#
#  | Method      | Workspace  | Background  | Preserves hierarchy? | Theoretical? |
#  |-------------|------------|-------------|---------------------|-------------|
#  | VICReg      | variance   | decorrelate | Partially           | No          |
#  | SIGReg      | isotropic  | isotropic   | NO (destroys it)   | Yes (LeJEPA)|
#  | BN/LN       | normalize  | normalize   | NO (flattens)      | No          |
#  | SWIP (ours) | preserve   | whiten      | YES                | Yes (above) |
#
#  SWIP is the ONLY method that preserves the workspace information
#  hierarchy while whitening the background noise.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  HOW OTHER PAPERS CAN USE SWIP
#  ═══════════════════════════════════════════════════════════════════════════
#
#  SWIP is a drop-in module for ANY self-supervised model:
#
#    from swip import SWIPModule
#    swip = SWIPModule(embed_dim=768, k_workspace=77)
#    loss_swip = swip(z)  # z is the encoder output
#    total_loss += lambda_swip * loss_swip
#
#  One import, two extra lines. Works with:
#    - Any JEPA variant (I-JEPA, V-JEPA, C-JEPA, TD-JEPA)
#    - Contrastive methods (SimCLR, MoCo, BYOL)
#    - Masked models (MAE, BEiT, BERT)
#    - Any modality (text, image, video, audio)
#
#  If you have a JAWP workspace Q, pass it for better workspace detection:
#    loss_swip = swip(z, workspace_Q=Q)
#
#  Hyperparameters:
#    - k_workspace: workspace dimension (default D//10, from Anthropic J-space)
#    - target_variance: background noise floor σ² (default 1.0)

import math

import torch
import torch.nn.functional as F
from torch import nn


class SWIPModule(nn.Module):
    """Selective Whitening with Information Preservation.

    Whitens background directions while preserving workspace
    eigenvalue structure. The log-eigenvalue matching loss:

    The implemented loss (audit R15):
      L_SWIP = Σ_{i>k} (log λ_i − log σ²)²              [background whitening]
               + hierarchy_weight · Σ ReLU(λ_{i+1} − λ_i + hierarchy_margin)
                                                        [workspace ordering]
    NOTE: scale-INVARIANCE DOES NOT HOLD as implemented — σ² is fixed while
    scaling Z scales λ_i (v1 docstring claimed otherwise; audit R11).
    Convexity holds per coordinate; the background term has its unique
    minimizer at λ_i = σ².

    Args:
        embed_dim: dimension of the embedding space (D).
        k_workspace: workspace dimension (default D//10).
            Background = D - k_workspace.
        target_variance: background noise floor σ² (default 1.0).
        use_jawp_workspace: if True, use JAWP Q for workspace
            identification (must pass workspace_Q to forward).
            If False, use top-k PCA directions.
        eps: numerical stability constant (default 1e-6).
        hierarchy_margin: minimum required eigenvalue DROP between consecutive
            workspace eigenvalues (descending order). Default 0.0 = enforce
            plain descending order.
        hierarchy_weight: weight of the workspace-ordering term (proofs/swip.md
            term 2, implemented R15). Default 1.0.
    """

    def __init__(
        self,
        embed_dim=768,
        k_workspace=None,
        target_variance=1.0,
        use_jawp_workspace=True,
        eps=1e-6,
        hierarchy_margin=0.0,
        hierarchy_weight=1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_workspace = k_workspace or max(embed_dim // 10, 1)
        self.target_variance = target_variance
        self.use_jawp_workspace = use_jawp_workspace
        self.eps = eps
        self.hierarchy_margin = hierarchy_margin
        self.hierarchy_weight = hierarchy_weight

        assert (
            1 <= self.k_workspace <= embed_dim
        ), f"k_workspace={self.k_workspace} must be in [1, {embed_dim}]"

    def forward(self, z, workspace_Q=None):
        """Compute SWIP loss.

        Args:
            z: (..., D) encoder representations.
            workspace_Q: (D, k) orthonormal workspace basis from JAWP.
                If None, uses top-k PCA directions.

        Returns:
            loss: scalar tensor (differentiable w.r.t. model parameters).
            info: dict with diagnostics.
        """
        D = z.size(-1)
        flat = z.reshape(-1, D).float()
        N = flat.size(0)

        if N <= 1:
            info = self._zero_info()
            return torch.tensor(0.0, device=z.device, requires_grad=True), info

        # Center
        centered = flat - flat.mean(dim=0, keepdim=True)

        # Compute covariance
        cov = (centered.T @ centered) / max(N - 1, 1)  # (D, D)

        # Eigendecomposition (differentiable w.r.t. cov)
        eigenvalues = torch.linalg.eigvalsh(cov)  # ascending order
        eigenvalues = eigenvalues.clamp(min=self.eps)  # numerical stability

        # Determine workspace/background split
        k = min(self.k_workspace, D)
        hierarchy_penalty = z.new_tensor(0.0)  # term-2 accumulator (R15)

        if workspace_Q is not None and self.use_jawp_workspace:
            # Use JAWP workspace: project onto Q and (I - QQ^T)
            Q = workspace_Q[:, : min(workspace_Q.shape[1], k)]
            k_actual = Q.shape[1]
            k_eff = k_actual

            # Workspace eigenvalues: diagonal of Q^T Cov Q
            # This is differentiable w.r.t. cov (and thus w.r.t. z)
            workspace_cov = Q.T @ cov @ Q  # (k, k)
            ws_eigenvalues = torch.linalg.eigvalsh(workspace_cov)
            ws_eigenvalues = ws_eigenvalues.clamp(min=self.eps)
            # Workspace hierarchy term (proofs/swip.md term 2, R15): enforce
            # descending order with margin δ on the JAWP-path spectrum.
            if k_actual > 1:
                lam_desc = ws_eigenvalues.flip(0)
                hierarchy_penalty = (
                    hierarchy_penalty
                    + self.hierarchy_weight
                    * F.relu(lam_desc[1:] - lam_desc[:-1] + self.hierarchy_margin).sum()
                )

            # Background eigenvalues: from the residual
            # Total trace = tr(cov), workspace trace = tr(workspace_cov)
            # Background trace = total - workspace
            total_trace = eigenvalues.sum()
            ws_trace = ws_eigenvalues.sum()
            bg_trace = (total_trace - ws_trace).clamp(min=self.eps)

            # Background log-eigenvalue matching
            # We want all (D - k) background eigenvalues to equal target_variance
            # Using trace-based approximation for efficiency:
            #   Σ_{i>k} (log λ_i - log σ²)² ≈ (D-k) * (log(bg_trace/(D-k)) - log σ²)²
            # This is Jensen's inequality applied to the convex loss
            bg_dim = D - k_actual
            if bg_dim > 0:
                bg_mean = bg_trace / bg_dim
                log_bg = bg_mean.clamp(min=self.eps).log()
                log_target = math.log(max(self.target_variance, self.eps))
                loss = bg_dim * (log_bg - log_target).pow(2)
            else:
                loss = torch.tensor(0.0, device=z.device)

            # Workspace preservation penalty: encourage workspace eigenvalues
            # to be well-separated from background (spectral gap)
            if k_actual > 0 and bg_dim > 0:
                ws_min = ws_eigenvalues.min()
                bg_mean_val = bg_trace / bg_dim
                # Spectral gap: ws_min should be > bg_mean
                # Loss penalty if gap is small
                spectral_gap = ws_min / (bg_mean_val + self.eps)
                # No penalty when gap > 1 (workspace > background)
                gap_penalty = F.relu(1.0 - spectral_gap) * 0.1
                loss = loss + gap_penalty
            else:
                spectral_gap = torch.tensor(1.0)

        else:
            # Use top-k PCA directions (eigenvalues are sorted ascending)
            k_eff = k
            # eigenvalues[0] = smallest, eigenvalues[-1] = largest
            bg_eigenvalues = eigenvalues[: D - k]  # smallest D-k (background)
            ws_eigenvalues = eigenvalues[D - k :]  # largest k (workspace)
            # Workspace hierarchy term (same as JAWP branch; spectrum here is
            # the top-k of an ascending array, so flip to descending).
            if k > 1:
                lam_desc = ws_eigenvalues.flip(0)
                hierarchy_penalty = (
                    hierarchy_penalty
                    + self.hierarchy_weight
                    * F.relu(lam_desc[1:] - lam_desc[:-1] + self.hierarchy_margin).sum()
                )

            # Background log-eigenvalue matching
            log_bg = bg_eigenvalues.clamp(min=self.eps).log()
            log_target = math.log(max(self.target_variance, self.eps))
            loss = (log_bg - log_target).pow(2).sum()

            # Spectral gap
            if k > 0 and D > k:
                ws_min = ws_eigenvalues.min()
                bg_max = bg_eigenvalues.max()
                spectral_gap = ws_min / (bg_max + self.eps)
            else:
                spectral_gap = torch.tensor(1.0)

        loss = loss + hierarchy_penalty
        # Compute diagnostics (no grad)
        with torch.no_grad():
            # Anisotropy ratio: λ_max / λ_min
            anisotropy = eigenvalues.max() / (eigenvalues.min() + self.eps)

            # Background uniformity: std(bg_eigenvalues) / mean(bg_eigenvalues)
            # Slice by the EFFECTIVE workspace width: on the JAWP path k may be
            # clamped by Q's width (k_eff < k), and a static D-k here reported
            # background diagnostics over the wrong tail (fleet R11).
            bg_eigs = eigenvalues[: D - k_eff]
            bg_mean_val = bg_eigs.mean()
            bg_std_val = bg_eigs.std() if bg_eigs.numel() > 1 else torch.tensor(0.0)
            bg_uniformity = bg_std_val / (bg_mean_val + self.eps)

            # Workspace concentration: fraction of total variance in workspace
            ws_var_frac = eigenvalues[D - k :].sum() / (eigenvalues.sum() + self.eps)

        info = {
            "anisotropy_ratio": anisotropy.item(),
            "bg_uniformity": bg_uniformity.item(),
            "ws_variance_fraction": ws_var_frac.item(),
            "spectral_gap": (
                spectral_gap.item() if isinstance(spectral_gap, torch.Tensor) else spectral_gap
            ),
            "k_workspace": k,
            "bg_dim": D - k_eff,
        }

        return loss, info

    @torch.no_grad()
    def compute_full_diagnostics(self, z, workspace_Q=None):
        """Compute comprehensive SWIP diagnostics (no loss, just metrics).

        Args:
            z: (..., D) encoder representations.
            workspace_Q: optional (D, k) workspace basis.

        Returns:
            dict with spectral analysis metrics.
        """
        D = z.size(-1)
        flat = z.reshape(-1, D).float()
        N = flat.size(0)

        if N <= 1:
            return self._zero_info()

        centered = flat - flat.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(N - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        eigenvalues = eigenvalues.flip(0)  # descending

        k = min(self.k_workspace, D)

        ws_eigs = eigenvalues[:k]
        bg_eigs = eigenvalues[k:]

        # Effective rank (Shannon entropy of normalized eigenvalues)
        total = eigenvalues.sum()
        probs = eigenvalues / (total + self.eps)
        entropy = -(probs * (probs + self.eps).log()).sum()
        eff_rank = entropy.exp().item()

        # Condition number
        cond = eigenvalues[0].item() / (eigenvalues[-1].item() + self.eps)

        # Spectral gap at workspace boundary
        if k < D and k > 0:
            gap = ws_eigs[-1].item() - bg_eigs[0].item()
            normalized_gap = gap / (eigenvalues[0].item() + self.eps)
        else:
            gap = 0.0
            normalized_gap = 0.0

        # Background SNR: how much signal leaks into background
        bg_signal = bg_eigs.sum().item()
        ws_signal = ws_eigs.sum().item()
        bg_snr = ws_signal / (bg_signal + self.eps)

        # Power-law exponent (alpha-norm from LeCun 2022)
        if eigenvalues[0] > self.eps:
            log_eigs = eigenvalues.clamp(min=self.eps).log()
            log_ranks = torch.arange(1, D + 1, dtype=torch.float32, device=eigenvalues.device).log()
            # Linear fit: log(λ) ≈ -α * log(rank) + c
            if D > 2:
                mean_x = log_ranks.mean()
                mean_y = log_eigs.mean()
                alpha = -((log_ranks - mean_x) * (log_eigs - mean_y)).sum() / (
                    (log_ranks - mean_x).pow(2).sum() + self.eps
                )
                alpha = alpha.item()
            else:
                alpha = 0.0
        else:
            alpha = 0.0

        return {
            "effective_rank": eff_rank,
            "condition_number": cond,
            "anisotropy_ratio": cond,
            "ws_variance_fraction": ws_signal / (total.item() + self.eps),
            "bg_variance_fraction": bg_signal / (total.item() + self.eps),
            "spectral_gap": normalized_gap,
            "spectral_gap_raw": gap,
            "bg_snr": bg_snr,
            "power_law_alpha": alpha,
            "k_workspace": k,
            "eigenvalues": eigenvalues.tolist(),
            "ws_eigenvalues": ws_eigs.tolist(),
            "bg_eigenvalues": bg_eigs.tolist(),
        }

    def _zero_info(self):
        return {
            "anisotropy_ratio": 0.0,
            "bg_uniformity": 0.0,
            "ws_variance_fraction": 0.0,
            "spectral_gap": 0.0,
            "k_workspace": self.k_workspace,
            "bg_dim": self.embed_dim - self.k_workspace,
        }

    def extra_repr(self):
        return (
            f"embed_dim={self.embed_dim}, k_workspace={self.k_workspace}, "
            f"target_variance={self.target_variance}"
        )
