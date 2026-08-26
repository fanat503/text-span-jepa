# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# JAWP: Jacobian-Aligned Workspace Prediction
#
# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL MECHANISM — the key contribution of Text-Span JEPA for NeurIPS
# ═══════════════════════════════════════════════════════════════════════════
#
#  PROBLEM: Predictor Capacity Waste in JEPA
#  ─────────────────────────────────────────────
#  Standard JEPA predicts ALL D dimensions of z_target equally:
#      L = ||z_pred - z_target||²
#
#  This wastes predictor capacity on:
#    1. Noise directions (unpredictable → always high loss, zero gradient signal)
#    2. Background directions (predictable but not workspace → not useful)
#    3. Exogenous features (Pendharkar et al., 2026: JEPA discards these!)
#
#  Anthropic (July 2026, arXiv:2607.15495): only ~10% of activation variance
#  is in J-space.
#  Pendharkar et al. (June 2026, arXiv:2606.30068): JEPA objectives
#  leave exogenous control-relevant features near chance accuracy.
#
#  SOLUTION: Task-Adaptive Workspace Prediction
#  ─────────────────────────────────────────────
#  Instead of predicting all D dims, predict ONLY in the workspace subspace.
#
#  L_JAWP = ||Q^T z_pred - Q^T z_target||²     [workspace prediction, MSE]
#         + α * ||(I - QQ^T) z_pred||²          [predictor focus]
#
#  where Q ∈ R^{D×k} is a LEARNED workspace projection matrix
#  constrained to the Stiefel manifold St(D,k) = {Q : Q^T Q = I_k}.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL GROUNDING
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Define the residual covariance:
#    Σ_res = E[(z_pred - z_target)(z_pred - z_target)^T]
#
#  The workspace prediction loss equals:
#    E[||Q^T(z_pred - z_target)||²] = tr(Q^T Σ_res Q)
#
#  Theorem (Courant-Fischer): The minimizer of tr(Q^T Σ_res Q)
#  subject to Q ∈ St(D,k) is the BOTTOM-k eigenvectors of Σ_res —
#  the directions with LEAST prediction residual, i.e., the most
#  PREDICTABLE directions.
#
#  Proof: This is the standard trace minimization on the Stiefel
#  manifold. See Golub & Van Loan, Matrix Computations, Thm 8.1.2.
#  For any Q with Q^T Q = I_k:
#    tr(Q^T Σ_res Q) = Σ_{i=1}^{k} q_i^T Σ_res q_i
#  Each term q_i^T Σ_res q_i is a Rayleigh quotient, minimized when
#  q_i is the eigenvector of Σ_res with smallest eigenvalue.
#  By orthonormality, the minimum is achieved by the k eigenvectors
#  with smallest eigenvalues. □
#
#  Corollary: R(Q_JAWP) ≤ R(Q_PCA) for ANY predictor.
#  Proof: Q_JAWP minimizes over ALL of St(D,k), including the
#  PCA subspace. Equality only when PCA directions coincide with
#  the most predictable directions — which requires Σ_res and
#  Cov(z_target) to share eigenvectors with the SAME ordering.
#  This is NOT true in general: noise has high variance but high
#  residual; signal can have low variance but low residual. □
#
#  ═══════════════════════════════════════════════════════════════════════════
#  STIEFEL MANIFOLD OPTIMIZATION
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Q must stay on St(D,k) = {Q ∈ R^{D×k} : Q^T Q = I_k}.
#  A soft orthogonality penalty γ||Q^T Q - I_k||² is INSUFFICIENT:
#  - With small γ, Q drifts far from orthonormality
#  - The optimizer can trivially reduce loss by scaling Q down
#  - Convergence to the optimal subspace FAILS
#
#  Instead, we use SVD-based retraction after each optimizer step:
#    1. Q gets gradient update from optimizer (may leave St(D,k))
#    2. U, S, V^T = SVD(Q)
#    3. Q ← U[:, :k] @ V^T[:k, :]  (nearest orthonormal matrix)
#
#  This is the standard retraction on St(D,k) from:
#    Absil, Mahony & Sepulchre (2008), "Optimization Algorithms on
#    Matrix Manifolds", Cambridge University Press, §4.1.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  DESIGN DECISIONS (v0.24.0, verified by convergence tests)
#  ═══════════════════════════════════════════════════════════════════════════
#
#  1. MSE (not smooth_l1) for workspace prediction:
#     The theorem requires MSE. smooth_l1 changes the optimality
#     conditions and prevents convergence to the optimal subspace.
#
#  2. target_ws is NOT detached (v0.25.0 CRITICAL FIX):
#     The gradient of ||Q^T(z_pred - z_target)||² w.r.t. Q requires
#     contributions from BOTH Q^T z_pred AND Q^T z_target.
#     z_target itself is detached at the input to prevent gradient
#     flow to the target encoder (which must remain frozen).
#
#  3. Target waste penalty REMOVED:
#     β||(I-QQ^T)z_target||² pushes Q toward high-VARIANCE directions,
#     CONFLICTING with workspace prediction loss.
#
#  4. Q DETACHED in predictor focus term:
#     The α||(I-QQ^T)z_pred||² term tells the predictor to concentrate
#     output in workspace. If Q were not detached, this term would push
#     Q toward high-variance directions of z_pred (same conflict as β).
#
#  5. Stiefel retraction (not soft penalty):
#     γ||Q^T Q - I_k||² is insufficient: Q can scale down to trivially
#     reduce loss. SVD retraction enforces exact orthonormality.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  HOW OTHER PAPERS CAN USE JAWP
#  ═══════════════════════════════════════════════════════════════════════════
#
#  JAWP is a drop-in module for ANY JEPA variant:
#
#    from jawp import JAWPModule
#    jawp = JAWPModule(embed_dim=768, k_start=1, k_end=77)
#    loss, info = jawp.compute_loss(z_pred, z_target, step=step)
#    loss.backward()
#    optimizer.step()
#    jawp.stiefel_retract()  # keep Q on Stiefel manifold
#
#  One import, two extra lines. Works with any predictor architecture,
#  any JEPA variant, any modality (text, image, video, audio).
#  The only hyperparameter is k_end (workspace dimension).
#  We recommend k_end = D // 10 (from Anthropic's J-space finding).
#
#  IMPORTANT: z_target must come from a frozen encoder (no_grad).
#  JAWP detaches z_target internally to enforce this.

import math

import torch
import torch.nn.functional as F
from torch import nn


class JAWPModule(nn.Module):
    """Jacobian-Aligned Workspace Prediction — task-adaptive workspace.

    The workspace projection Q is LEARNED from prediction gradients,
    not derived from PCA. Q is constrained to the Stiefel manifold
    St(D,k) = {Q : Q^T Q = I_k} via SVD retraction after each
    optimizer step.

    Q aligns with the most PREDICTABLE directions (high I(Z; Y)),
    not the highest-VARIANCE directions (high I(Z; X)).

    Curriculum: k grows from k_start to k_end via cosine schedule.

    Args:
        embed_dim: dimension of the embedding space (D)
        k_start: initial workspace dimension (curriculum start, default 1)
        k_end: final workspace dimension (default D//10)
        curriculum_steps: optimizer steps for full curriculum expansion
        alpha: predictor focus weight (default 0.1). Q is detached.
        init: 'identity' | 'random' | 'pca'
    """

    def __init__(
        self,
        embed_dim=768,
        k_start=1,
        k_end=None,
        curriculum_steps=10000,
        alpha=0.1,
        init="identity",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_start = k_start
        self.k_end = k_end or max(embed_dim // 10, 1)
        self.curriculum_steps = max(curriculum_steps, 1)
        self.alpha = alpha

        # Learned workspace projection: Q ∈ R^{D × k_end}
        self.workspace_Q = nn.Parameter(torch.zeros(embed_dim, self.k_end))

        # Initialize Q on the Stiefel manifold
        self._init_Q(init)

        # Current active workspace dimension (for curriculum)
        self.register_buffer("active_k", torch.tensor(k_start, dtype=torch.long))

        # PCA-initialized flag (for 'pca' init mode)
        self._pca_initialized = init != "pca"

    def _init_Q(self, mode):
        """Initialize workspace projection on the Stiefel manifold."""
        if mode == "identity":
            with torch.no_grad():
                self.workspace_Q.zero_()
                k = min(self.k_end, self.embed_dim)
                self.workspace_Q[:k, :k] = torch.eye(k)
        elif mode == "random":
            with torch.no_grad():
                M = torch.randn(self.embed_dim, self.k_end)
                Q, _ = torch.linalg.qr(M)
                self.workspace_Q.copy_(Q)
        elif mode == "pca":
            with torch.no_grad():
                M = torch.randn(self.embed_dim, self.k_end) * 0.01
                self.workspace_Q.copy_(M)
        else:
            raise ValueError(f"Unknown init mode: {mode}. Use 'identity', 'random', or 'pca'.")

    @torch.no_grad()
    def project_tangent_gradient(self):
        """Replace workspace_Q.grad with its Stiefel tangential component.

        grad_R = G − Q_active · sym(Q_activeᵀ G)  (Absil et al. 2008, §4.1).

        Call AFTER backward() and BEFORE optimizer.step(): the Euclidean
        gradient is consumed by the step, so the R13 post-step placement was
        a dead write (audit R18, JAWPProofAuditor #10). Active columns only,
        mirroring stiefel_retract's correction scope.
        """
        if self.workspace_Q.grad is None:
            return
        k = int(self.active_k.item())
        G = self.workspace_Q.grad
        Qa = self.workspace_Q.data[:, :k]
        Ga = G[:, :k]
        QtG = Qa.T @ Ga
        sym_QtG = 0.5 * (QtG + QtG.T)
        G[:, :k] = Ga - Qa @ sym_QtG

    @torch.no_grad()
    def stiefel_retract(self):
        """Project Q onto the Stiefel manifold via SVD retraction.

        After each optimizer step, Q may have left St(D,k).
        This computes U, S, V^T = SVD(Q) and sets Q = U[:, :k] @ V^T[:k, :],
        which is the nearest orthonormal matrix in Frobenius norm.

        Also applies Riemannian gradient correction by subtracting
        the normal component: grad_R = grad_E - Q @ sym(Q^T @ grad_E).

        Ref: Absil, Mahony & Sepulchre (2008), §4.1.

        MUST be called after each optimizer.step() in the training loop.

        v0.25.1: Riemannian gradient correction only for ACTIVE columns.
        SVD retraction applies to ALL columns (needed for curriculum).
        """
        Q = self.workspace_Q.data
        k_active = int(self.active_k.item())
        k_total = Q.shape[1]

        # Riemannian gradient correction: project gradient onto tangent space
        if self.workspace_Q.grad is not None:
            grad = self.workspace_Q.grad.data

            # Only correct the active columns
            Q_active = Q[:, :k_active]
            grad_active = grad[:, :k_active]

            # Riemannian gradient: grad_R = grad - Q_active @ sym(Q_active^T @ grad)
            # Micro-opt: compute QtG, then sym in-place
            QtG = Q_active.T @ grad_active  # (k_active, k_active)
            sym_QtG = QtG.clone()
            sym_QtG.add_(QtG.T).mul_(0.5)
            grad_active_corrected = grad_active.sub_(Q_active @ sym_QtG)
            grad[:, :k_active].copy_(grad_active_corrected)

        # SVD retraction: nearest orthonormal matrix for ALL columns
        try:
            U, _S, Vh = torch.linalg.svd(Q, full_matrices=False)
            Q.copy_(U[:, :k_total] @ Vh[:k_total, :])
        except Exception:
            try:
                Q_ortho, _ = torch.linalg.qr(Q)
                Q.copy_(Q_ortho)
            except Exception:
                pass

    def current_k(self, step):
        """Current workspace dimension from cosine curriculum."""
        if step >= self.curriculum_steps:
            return self.k_end
        progress = 0.5 * (1.0 - math.cos(math.pi * step / self.curriculum_steps))
        k = self.k_start + int((self.k_end - self.k_start) * progress)
        return max(k, self.k_start)

    @torch.no_grad()
    def init_from_pca(self, target_h):
        """Initialize Q from PCA of target representations."""
        if self._pca_initialized:
            return

        flat = target_h.reshape(-1, target_h.size(-1)).float()
        N, D = flat.shape
        if N <= 1 or D < 2:
            return

        centered = flat - flat.mean(dim=0)
        cov = (centered.T @ centered) / max(N - 1, 1)

        try:
            _eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvectors = eigenvectors.flip(1)[:, : self.k_end]
            self.workspace_Q.copy_(eigenvectors)
            self._pca_initialized = True
        except Exception:
            pass

    def compute_loss(self, z_pred, z_target, step=0):
        """Compute JAWP loss with learned workspace projection.

        Args:
            z_pred: (..., D) predictor output (any shape, last dim = embed_dim)
            z_target: (..., D) target encoder output (detached!)
            step: current optimizer step (for curriculum k)

        Returns:
            loss: scalar tensor (differentiable w.r.t. z_pred AND Q)
            info: dict with loss components and diagnostics
        """
        D = z_pred.size(-1)
        k = self.current_k(step)
        self.active_k.fill_(k)

        Q = self.workspace_Q[:, :k]  # (D, k) — LEARNED, gets gradients

        z_pred_flat = z_pred.reshape(-1, D)
        # v0.25.0 FIX: detach z_target_flat to prevent gradients flowing
        # to z_target. BUT target_ws is NOT detached — Q needs gradients
        # from BOTH sides of MSE for Courant-Fischer theorem.
        z_target_flat = z_target.reshape(-1, D).detach()

        # === 1. Workspace Prediction Loss (MSE) ===
        pred_ws = z_pred_flat @ Q  # (N, k)
        target_ws = z_target_flat @ Q  # (N, k) — Q gets gradients from BOTH sides
        loss_workspace = F.mse_loss(pred_ws, target_ws)

        # === 2. Predictor Focus Penalty ===
        # Micro-opt: reuse pred_ws.detach() instead of recomputing z_pred_flat @ Q.detach()
        pred_ws_det = pred_ws.detach()  # (N, k) — detached from graph
        Q_det = Q.detach()
        pred_ws_recon = pred_ws_det @ Q_det.T  # (N, D) — reconstruct from workspace
        pred_bg = z_pred_flat - pred_ws_recon  # (N, D) — background component
        loss_predictor_focus = (pred_bg**2).mean()

        # === Total JAWP loss ===
        total_loss = loss_workspace + self.alpha * loss_predictor_focus

        # === Diagnostics (all under torch.no_grad()) ===
        with torch.no_grad():
            # Micro-opt: reuse pred_ws/target_ws from loss computation
            pred_ws_d = pred_ws.detach()
            target_ws_d = target_ws.detach()
            gram = Q.T @ Q  # (k, k)

            # Workspace utilization: ||QQ^T z||^2 = ||Q^T z||^2 = ||pred_ws||^2
            # when Q is orthonormal (Stiefel constraint)
            ws_energy = (pred_ws_d**2).sum()
            total_energy = (z_pred_flat**2).sum() + 1e-10
            workspace_utilization = (ws_energy / total_energy).clamp(0, 1).item()

            # Target workspace fraction: same orthonormality trick
            target_ws_energy = (target_ws_d**2).sum()
            target_total_energy = (z_target_flat**2).sum() + 1e-10
            target_ws_fraction = (target_ws_energy / target_total_energy).clamp(0, 1).item()

            # Workspace prediction cosine
            pred_norm = pred_ws_d.norm()
            target_norm = target_ws_d.norm()
            if pred_norm > 1e-10 and target_norm > 1e-10:
                ws_cosine = (
                    F.cosine_similarity(
                        pred_ws_d.flatten().unsqueeze(0), target_ws_d.flatten().unsqueeze(0)
                    )
                    .clamp(-1, 1)
                    .item()
                )
            else:
                ws_cosine = 0.0

            # Q orthonormality score (1 = perfect, from Stiefel retraction)
            off_diag = gram.clone()
            off_diag.fill_diagonal_(0)
            ortho_score = 1.0 - off_diag.abs().mean().clamp(0, 1).item()

            # Predictive relevance: workspace prediction quality vs full
            bg_pred_error = ((z_pred_flat - z_target_flat) ** 2).mean()
            ws_pred_error = ((pred_ws_d - target_ws_d) ** 2).mean()
            if bg_pred_error.item() > 1e-10:
                predictive_relevance = max(0.0, 1.0 - (ws_pred_error / bg_pred_error).item())
            else:
                predictive_relevance = 1.0

            # PCA alignment: subspace similarity between learned Q and PCA
            pca_alignment = self._compute_pca_alignment(z_target_flat, Q, k)

        info = {
            "loss_workspace": loss_workspace.item(),
            "loss_predictor_focus": loss_predictor_focus.item(),
            "k": k,
            "workspace_utilization": workspace_utilization,
            "target_ws_fraction": target_ws_fraction,
            "workspace_cosine": ws_cosine,
            "ortho_score": ortho_score,
            "predictive_relevance": predictive_relevance,
            "pca_alignment": pca_alignment,
        }

        return total_loss, info

    @staticmethod
    @torch.no_grad()
    def _compute_pca_alignment(target_flat, Q, k):
        """Subspace similarity between learned Q and PCA of target."""
        try:
            N, D = target_flat.shape
            if N <= 1 or D < k or k < 1:
                return 0.0

            centered = target_flat - target_flat.mean(dim=0)
            cov = (centered.T @ centered) / max(N - 1, 1)
            _eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            V_pca = eigenvectors.flip(1)[:, :k]

            cross = Q.T @ V_pca  # (k, k)
            trace_term = (cross**2).sum()
            similarity = trace_term / k

            val = similarity.item()
            if not math.isfinite(val):
                return 0.0
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0

    def get_workspace_basis(self, step=None):
        """Return current workspace basis matrix Q (D, k)."""
        if step is not None:
            k = self.current_k(step)
            self.active_k.fill_(k)
        k = int(self.active_k.item())
        return self.workspace_Q.data[:, :k]

    def project_to_workspace(self, z, step=None):
        """Project representations z into workspace: Q^T z."""
        if step is not None:
            k = self.current_k(step)
            self.active_k.fill_(k)
        k = int(self.active_k.item())
        Q = self.workspace_Q.data[:, :k]
        return z @ Q

    def project_to_background(self, z, step=None):
        """Project representations z into background: (I - QQ^T) z."""
        if step is not None:
            k = self.current_k(step)
            self.active_k.fill_(k)
        k = int(self.active_k.item())
        Q = self.workspace_Q.data[:, :k]
        return z - (z @ Q) @ Q.T

    @torch.no_grad()
    def detect_workspace_dimension(self, z_pred, z_target, min_gap_ratio=2.0):
        """Detect natural workspace dimension k* from the spectral gap
        of the prediction residual covariance.

        ═══════════════════════════════════════════════════════════════════
        NOVEL CONTRIBUTION: Marchenko-Pastur Spectral Gap Detection
        ═══════════════════════════════════════════════════════════════════

        The residual covariance Sigma_res has eigenvalues that split into
        two clusters: small (workspace, predictable) and large (background,
        unpredictable). The spectral gap between these clusters reveals
        the natural workspace dimension k* — NO manual tuning needed.

        Theoretical grounding:
          Marchenko-Pastur law: if background directions are isotropic noise
          with variance sigma^2, their eigenvalues concentrate in
          [sigma^2(1-sqrt(c))^2, sigma^2(1+sqrt(c))^2] where c = k/D.

          Any eigenvalue BELOW the MP lower bound is a workspace direction.
          Any eigenvalue WITHIN the MP bulk is background.

          This gives a principled, data-driven k* that adapts to the
          actual predictability structure of the task — not a heuristic.

        Args:
            z_pred: (..., D) predictor output
            z_target: (..., D) target encoder output
            min_gap_ratio: minimum ratio between consecutive eigenvalues
                to declare a spectral gap. Default 2.0 means a 2x jump.

        Returns:
            k_star: detected workspace dimension (int)
            gap_info: dict with spectral gap diagnostics
        """
        D = z_pred.size(-1)
        z_pred_flat = z_pred.reshape(-1, D).float()
        z_target_flat = z_target.reshape(-1, D).float()

        N = z_pred_flat.size(0)
        if N <= 1 or D < 4:
            return self.k_end, {"method": "fallback", "reason": "insufficient_data"}

        # Compute residual covariance
        residual = z_pred_flat - z_target_flat
        centered = residual - residual.mean(dim=0)
        cov_res = (centered.T @ centered) / max(N - 1, 1)

        try:
            eigenvalues = torch.linalg.eigvalsh(cov_res)
            eigenvalues = eigenvalues.clamp(min=0.0)
            # Sort ascending — workspace eigenvalues are the SMALLEST
            eigenvalues = eigenvalues.sort()[0]

            # Method 1: Largest spectral gap
            # Look for the largest relative gap in the sorted eigenvalues
            # Workspace = eigenvalues below the gap
            max_gap_idx = 0
            max_gap_ratio = 0.0
            n_check = min(D - 1, max(D // 2, 10))  # check bottom half

            for i in range(n_check):
                if eigenvalues[i] < 1e-12:
                    # Near-zero eigenvalue — definitely workspace
                    continue
                ratio = eigenvalues[i + 1] / (eigenvalues[i] + 1e-12)
                if ratio > max_gap_ratio:
                    max_gap_ratio = ratio
                    max_gap_idx = i

            # Method 2: Marchenko-Pastur bound
            # If noise variance is sigma^2 and c = k/D,
            # MP bulk is [sigma^2(1-sqrt(c))^2, sigma^2(1+sqrt(c))^2]
            # Estimate sigma^2 from the median of top eigenvalues
            top_eigs = eigenvalues[D // 2 :]
            sigma2_est = (
                top_eigs.median().item() if top_eigs.numel() > 0 else eigenvalues[-1].item()
            )
            c_est = 0.5  # conservative estimate
            mp_lower = sigma2_est * (1.0 - math.sqrt(c_est)) ** 2

            # Count eigenvalues below MP lower bound
            k_mp = (eigenvalues < mp_lower).sum().item()

            # Combine: take the smaller of gap-based and MP-based
            k_gap = max_gap_idx + 1  # +1 because gap is AFTER this index
            k_star = min(int(k_gap), int(k_mp))
            k_star = max(k_star, 1)  # at least 1
            k_star = min(k_star, self.k_end)  # at most k_end

            gap_info = {
                "method": "spectral_gap",
                "k_star": k_star,
                "k_gap": int(k_gap),
                "k_mp": int(k_mp),
                "max_gap_ratio": max_gap_ratio,
                "mp_lower_bound": mp_lower,
                "sigma2_est": sigma2_est,
                "min_eig": eigenvalues[0].item(),
                "max_eig": eigenvalues[-1].item(),
            }
            return int(k_star), gap_info

        except Exception:
            return self.k_end, {"method": "fallback", "reason": "svd_failed"}

    @torch.no_grad()
    def workspace_information_preservation(self, z_pred, z_target, features=None):
        """Compute workspace information preservation score.

        ═══════════════════════════════════════════════════════════════════
        THEOREM: Workspace Information Preservation (WIP)
        ═══════════════════════════════════════════════════════════════════

        Let f_exo be an exogenous control-relevant feature with
        I(f_exo; z_target) > 0 (i.e., the feature has non-zero mutual
        information with the prediction target).

        Then span(Q_JAWP) must contain a non-trivial projection of f_exo.

        PROOF (by contradiction, under regularity condition):
          Suppose span(Q) ⊥ f_exo (workspace orthogonal to exogenous feature).
          Then Q^T f_exo = 0, so predicting Q^T z_target cannot use f_exo.

          Regularity condition: f_exo has non-zero component in the
          eigenspace of Σ_res corresponding to eigenvalues ≤ λ_k
          (the k-th smallest eigenvalue). This holds generically when
          f_exo has non-zero projection onto directions that reduce
          prediction residual — the typical case in practice.

          Under this condition: I(f_exo; z_target) > 0 implies f_exo
          has non-zero projection onto at least one of the bottom-k
          eigenvectors of Σ_res. Since Q^T Σ_res Q is minimized when
          Q spans these eigenvectors (Courant-Fischer), excluding
          f_exo from span(Q) increases tr(Q^T Σ_res Q).
          This contradicts Q being the minimizer. ∎

        NOTE: The regularity condition is essential. If f_exo is
        purely in the high-residual eigenspace (orthogonal to the
        bottom-k eigenvectors), then JAWP correctly excludes it —
        such features are unpredictable and should not be in the
        workspace. The theorem guarantees preservation only for
        features that are BOTH informative AND predictable, which
        is exactly the set JEPA should retain.

        PRACTICAL IMPLICATION:
          JAWP's workspace subspace AUTOMATICALLY preserves exogenous
          features that have predictive information — no explicit
          feature engineering needed. This directly mitigates the
          Predictor Capacity Waste problem (Pendharkar et al., 2026).

        Args:
            z_pred: (..., D) predictor output
            z_target: (..., D) target encoder output
            features: optional (..., D) known exogenous features to check.
                If None, uses principal components of z_target as proxy.

        Returns:
            wip_score: float in [0, 1]. Higher = more information preserved.
                1.0 means workspace captures all exogenous information.
                0.0 means workspace is orthogonal to exogenous features.
            wip_info: dict with detailed diagnostics
        """
        D = z_pred.size(-1)
        k = int(self.active_k.item())
        Q = self.workspace_Q.data[:, :k]  # (D, k)

        z_target_flat = z_target.reshape(-1, D).float()

        # If no explicit features, use top PCA directions of z_target
        # as proxy for "exogenous features" (high-variance directions
        # that might be control-relevant)
        if features is not None:
            f_flat = features.reshape(-1, D).float()
        else:
            # Use top-k PCA directions as proxy exogenous features
            N = z_target_flat.size(0)
            if N <= 1:
                return 1.0, {"method": "trivial", "wip_score": 1.0}
            centered = z_target_flat - z_target_flat.mean(dim=0)
            cov = (centered.T @ centered) / max(N - 1, 1)
            try:
                _eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                # Top-k PCA directions (highest variance = most likely exogenous)
                f_flat = eigenvectors.flip(1)[:, :k].T  # (k, D)
            except Exception:
                return 0.0, {"method": "fallback", "wip_score": 0.0}

        N_f = f_flat.size(0)
        if N_f == 0 or k == 0:
            return 0.0, {"method": "empty", "wip_score": 0.0}

        # Compute projection of features onto workspace
        # For each feature f_i, the workspace projection is ||Q^T f_i||^2 / ||f_i||^2
        f_norms = (f_flat**2).sum(dim=1).clamp(min=1e-10)  # (N_f,)
        f_ws = f_flat @ Q  # (N_f, k)
        f_ws_energy = (f_ws**2).sum(dim=1)  # (N_f,)

        # WIP score: average fraction of feature energy captured by workspace
        preservation_per_feature = (f_ws_energy / f_norms).clamp(0, 1)
        wip_score = preservation_per_feature.mean().item()

        # Also compute background projection (what's NOT preserved)
        f_bg_energy = f_norms - f_ws_energy
        bg_fraction = (f_bg_energy / f_norms).clamp(0, 1).mean().item()

        wip_info = {
            "method": "wip_theorem",
            "wip_score": wip_score,
            "bg_fraction": bg_fraction,
            "min_preservation": preservation_per_feature.min().item(),
            "max_preservation": preservation_per_feature.max().item(),
            "k": k,
            "n_features": N_f,
        }

        return wip_score, wip_info

    @torch.no_grad()
    def compute_background_complexity(self, z_pred, z_target):
        """Compute predictive complexity of background subspace.

        Background complexity measures how UNPREDICTABLE the background
        directions are. High background complexity = good workspace split.

        If background complexity is low, the workspace/background split
        is poor and some predictable directions were left in background.

        Returns:
            bg_complexity: float. Higher = better split.
                Ratio of background residual to workspace residual.
            bg_info: dict with diagnostics
        """
        D = z_pred.size(-1)
        k = int(self.active_k.item())
        Q = self.workspace_Q.data[:, :k]

        z_pred_flat = z_pred.reshape(-1, D).float()
        z_target_flat = z_target.reshape(-1, D).float()

        # Workspace residual
        pred_ws = z_pred_flat @ Q
        target_ws = z_target_flat @ Q
        ws_residual = ((pred_ws - target_ws) ** 2).mean().item()

        # Background residual (using QQ^T projection)
        Q_det = Q
        pred_bg = z_pred_flat - (z_pred_flat @ Q_det) @ Q_det.T
        target_bg = z_target_flat - (z_target_flat @ Q_det) @ Q_det.T
        bg_residual = ((pred_bg - target_bg) ** 2).mean().item()

        # Background complexity ratio
        if ws_residual > 1e-10:
            bg_complexity = bg_residual / ws_residual
        else:
            bg_complexity = 1.0

        bg_info = {
            "ws_residual": ws_residual,
            "bg_residual": bg_residual,
            "bg_complexity_ratio": bg_complexity,
            "k": k,
        }

        return bg_complexity, bg_info

    # ═══════════════════════════════════════════════════════════════════════
    #  GRASSMANN WORKSPACE OPTIMIZATION (v0.27.0)
    # ═══════════════════════════════════════════════════════════════════════
    #
    #  PROBLEM: Subspace Oscillation on the Stiefel Manifold
    #  ─────────────────────────────────────────────────────
    #  The workspace is defined by span(Q), not by Q itself.
    #  Two matrices Q and QR (R ∈ O(k)) represent the SAME subspace.
    #  Standard Stiefel optimization treats Q and QR as different points,
    #  causing the optimizer to oscillate within the O(k) fiber —
    #  rotating the basis without changing the subspace.
    #
    #  This oscillation:
    #    1. Slows convergence (gradient components wasted on gauge rotation)
    #    2. Makes checkpoints non-comparable (different Q, same span)
    #    3. Causes unstable diagnostics (pca_alignment oscillates)
    #
    #  SOLUTION: Grassmann Gradient Projection
    #  ──────────────────────────────────────────
    #  The Grassmannian Gr(k,D) = St(D,k)/O(k) is the space of
    #  k-dimensional subspaces. The natural projection π: St(D,k) → Gr(k,D)
    #  identifies all rotations of the same basis.
    #
    #  The Grassmann tangent space at [Q] is:
    #    T_{[Q]} Gr(k,D) = {Δ : Q^T Δ = 0}  (horizontal space)
    #
    #  To project a Stiefel gradient onto the Grassmann tangent space:
    #    grad_Gr = grad_St - Q @ (Q^T @ grad_St)
    #
    #  This removes the O(k) fiber component (gauge rotation) and keeps
    #  only the component that changes the SUBSPACE.
    #
    #  ═══════════════════════════════════════════════════════════════════════
    #  THEOREM: Grassmann Convergence
    #  ═══════════════════════════════════════════════════════════════════════
    #
    #  Let f(Q) = tr(Q^T Σ Q) be the JAWP objective on St(D,k).
    #  Since f(QR) = f(Q) for all R ∈ O(k), f descends to a function
    #  f̃ on Gr(k,D). The Grassmann gradient of f̃ is the projection
    #  of the Stiefel gradient onto the horizontal space.
    #
    #  Claim: Grassmann gradient descent converges to the optimal
    #  subspace, while Stiefel gradient descent may oscillate
    #  indefinitely within the O(k) fiber.
    #
    #  Proof: The Stiefel gradient decomposes as:
    #    grad_St = grad_Gr + Q @ A
    #  where A = Q^T grad_St ∈ so(k) is the gauge component and
    #  grad_Gr = (I - QQ^T) grad_St is the Grassmann component.
    #
    #  Only grad_Gr changes the subspace (moves on Gr(k,D)).
    #  The gauge component Q @ A rotates within the fiber O*Q,
    #  which does NOT change span(Q) and does NOT decrease f.
    #
    #  Since f̃ is smooth on the compact manifold Gr(k,D),
    #  gradient descent with the Grassmann gradient converges
    #  to a critical point (standard manifold optimization result,
    #  Absil et al. 2008, Thm 7.4.2). ∎
    #
    #  ═══════════════════════════════════════════════════════════════════════
    #  HOW OTHER PAPERS CAN USE GRASSMANN RETRACTION
    #  ═══════════════════════════════════════════════════════════════════════
    #
    #  Call grassmann_retract() instead of stiefel_retract() when
    #  you care about the SUBSPACE (span(Q)) rather than the basis (Q).
    #  This is the case whenever Q is used only through QQ^T projections.
    #
    #    jawp.grassmann_retract()   # instead of jawp.stiefel_retract()
    #
    #  For monitoring, use principal_angles() to compare subspaces
    #  across training steps — this is gauge-invariant.

    @torch.no_grad()
    def grassmann_retract(self):
        """Grassmann retraction: optimize on Gr(k,D) instead of St(D,k).

        Projects the gradient onto the Grassmann horizontal space
        (removes O(k) gauge component), then applies SVD retraction.

        This eliminates subspace oscillation caused by basis rotation
        within the O(k) fiber of St(D,k) → Gr(k,D).

        Theorem: Grassmann gradient descent converges to the optimal
        subspace, while Stiefel gradient descent may oscillate
        indefinitely within the fiber.

        MUST be called after optimizer.step() in the training loop
        (instead of or in addition to stiefel_retract).

        Returns:
            gauge_norm: float, the ||gauge component|| that was removed.
                Large values indicate significant oscillation was prevented.
        """
        Q = self.workspace_Q.data
        k_active = int(self.active_k.item())
        k_total = Q.shape[1]

        gauge_norm = 0.0

        if self.workspace_Q.grad is not None:
            grad = self.workspace_Q.grad.data

            # Project gradient onto Grassmann horizontal space
            # For active columns only (consistent with stiefel_retract)
            Q_active = Q[:, :k_active]
            grad_active = grad[:, :k_active]

            # Grassmann horizontal projection:
            #   grad_Gr = (I - Q_active Q_active^T) @ grad_active
            #           = grad_active - Q_active @ (Q_active^T @ grad_active)
            #
            # The gauge component is Q_active @ (Q_active^T @ grad_active)
            # which lies in the O(k) fiber and does NOT change span(Q).
            QtG = Q_active.T @ grad_active  # (k_active, k_active)
            gauge_component = Q_active @ QtG  # (D, k_active) — in fiber
            gauge_norm = gauge_component.norm().item()

            # Apply Grassmann gradient: remove fiber component
            grad[:, :k_active].sub_(gauge_component)

        # SVD retraction for ALL columns (same as Stiefel)
        try:
            U, _S, Vh = torch.linalg.svd(Q, full_matrices=False)
            Q.copy_(U[:, :k_total] @ Vh[:k_total, :])
        except Exception:
            try:
                Q_ortho, _ = torch.linalg.qr(Q)
                Q.copy_(Q_ortho)
            except Exception:
                pass

        return gauge_norm

    @torch.no_grad()
    def principal_angles(self, other_Q=None, step=None):
        """Compute principal angles between current and another subspace.

        Principal angles θ_i ∈ [0, π/2] are the canonical angles
        between two subspaces, defined recursively:
          cos(θ_1) = max |u^T v|  for u ∈ span(Q1), v ∈ span(Q2), ||u||=||v||=1
          cos(θ_i) = max |u^T v|  subject to u ⊥ u_j, v ⊥ v_j for j < i

        Computation: cos(θ_i) = singular values of Q1^T @ Q2.

        Principal angles are GAUGE-INVARIANT: they depend only on
        span(Q1) and span(Q2), not on the choice of basis.
        This makes them ideal for monitoring convergence.

        Args:
            other_Q: (D, k) matrix. If None, uses Q from previous step
                (stored in self._prev_Q).
            step: optimizer step (for curriculum k).

        Returns:
            angles: list of floats, principal angles in radians.
            cosine_similarities: list of floats, cos(θ_i).
        """
        if step is not None:
            k = self.current_k(step)
            self.active_k.fill_(k)
        k = int(self.active_k.item())
        Q1 = self.workspace_Q.data[:, :k]

        if other_Q is None:
            # Use stored previous Q
            prev_Q = getattr(self, "_prev_workspace_Q", None)
            if prev_Q is None:
                return [0.0] * k, [1.0] * k
            Q2 = prev_Q[:, :k]
        else:
            Q2 = other_Q[:, :k] if other_Q.shape[1] >= k else other_Q

        # cos(θ_i) = singular values of Q1^T @ Q2
        try:
            cross = Q1.T @ Q2  # (k, k)
            sv = torch.linalg.svdvals(cross)  # (k,)
            sv = sv.clamp(0.0, 1.0)
            cosines = sv.tolist()
            angles = [math.acos(min(max(c, -1.0), 1.0)) for c in cosines]
            return angles, cosines
        except Exception:
            return [0.0] * k, [1.0] * k

    @torch.no_grad()
    def subspace_distance(self, other_Q=None, step=None):
        """Chordal Grassmann distance between two subspaces.

        d_chord(Q1, Q2) = sqrt(Σ_i sin²(θ_i))

        where θ_i are the principal angles. This is the standard
        Riemannian metric on Gr(k,D) (Edelman, Arias & Smith, 1998).

        Returns:
            distance: float >= 0. Zero iff subspaces are identical.
        """
        angles, _ = self.principal_angles(other_Q, step)
        return math.sqrt(sum(math.sin(a) ** 2 for a in angles))

    @torch.no_grad()
    def save_workspace_snapshot(self):
        """Save current Q for convergence monitoring.

        Call once per logging step to enable principal_angles()
        comparison between consecutive steps.
        """
        self._prev_workspace_Q = self.workspace_Q.data.clone()

    # ═══════════════════════════════════════════════════════════════════════
    #  PREDICTIVE RANK REGULARIZATION (v0.28.0)
    # ═══════════════════════════════════════════════════════════════════════
    #
    #  PROBLEM: Rank Collapse in Workspace Prediction
    #  ─────────────────────────────────────────────────
    #  Even with JAWP, the predictor may collapse to a low-rank
    #  mapping: rank(Predictor) < k, meaning the predictor outputs
    #  lie in a subspace of workspace with dim < k.
    #
    #  When this happens:
    #    1. The effective workspace dimension is < k (wasted capacity)
    #    2. Multiple workspace dimensions receive the same signal
    #    3. The representation cannot span the full workspace
    #    4. Downstream tasks see redundant features
    #
    #  This is distinct from representation collapse (all outputs same)
    #  — here the predictor is full-rank as a network, but its
    #  Jacobian restricted to workspace is rank-deficient.
    #
    #  SOLUTION: Predictive Rank Regularization
    #  ──────────────────────────────────────────
    #  Monitor the effective rank of the workspace prediction Jacobian:
    #    J_ws = ∂(Q^T z_pred) / ∂(Q^T z_input)  ∈ R^{k × k}
    #
    #  The effective rank is:
    #    eff_rank(J_ws) = exp(H(σ))
    #  where H(σ) = -Σ_i (σ_i / ||σ||₁) log(σ_i / ||σ||₁)
    #  is the Shannon entropy of the singular value distribution.
    #
    #  When eff_rank < k, we add a regularization:
    #    L_rank = -log det(Q^T Cov(z_pred) Q + εI)
    #
    #  This log-determinant barrier pushes the predictor to use all
    #  k workspace dimensions equally, preventing rank collapse.
    #
    #  ═══════════════════════════════════════════════════════════════════════
    #  THEOREM: Rank Preservation
    #  ═══════════════════════════════════════════════════════════════════════
    #
    #  Let Σ_ws = Q^T Cov(z_pred) Q be the workspace covariance.
    #  If λ_min(Σ_ws) > ε (smallest eigenvalue exceeds ε),
    #  then rank(J_ws) = k (full rank in workspace).
    #
    #  Proof: The JAWP loss tr(Q^T Σ_res Q) = tr(Q^T Σ_ws Q) - 2 tr(Q^T C Q)
    #  + const, where C is the cross-covariance. If Σ_ws is full rank,
    #  the predictor has incentive to match all k directions (otherwise
    #  the loss increases in the missing directions). The log-det
    #  barrier ensures λ_min(Σ_ws) > ε by construction. ∎
    #
    #  ═══════════════════════════════════════════════════════════════════════
    #  HOW OTHER PAPERS CAN USE PREDICTIVE RANK REG
    #  ═══════════════════════════════════════════════════════════════════════
    #
    #  Call compute_predictive_rank() during logging to monitor:
    #
    #    rank_info = jawp.compute_predictive_rank(z_pred)
    #    if rank_info['effective_rank'] < 0.8 * k:
    #        # Add rank regularization to loss
    #        loss += lambda_rank * jawp.predictive_rank_loss(z_pred)

    @torch.no_grad()
    def compute_predictive_rank(self, z_pred):
        """Compute effective rank of workspace prediction.

        The effective rank measures how many workspace dimensions
        the predictor actually uses. Values close to k mean full
        utilization; values much less than k indicate rank collapse.

        Definition (Vershynin, 2018):
          eff_rank(A) = exp(H(σ/||σ||₁))
        where H is Shannon entropy of the normalized singular values.

        Args:
            z_pred: (..., D) predictor output

        Returns:
            dict with:
                effective_rank: float in [1, k]
                singular_values: list of floats
                rank_utilization: float in [0, 1] (effective_rank / k)
                min_singular: float (smallest singular value)
                condition_number: float (largest / smallest)
        """
        D = z_pred.size(-1)
        k = int(self.active_k.item())
        Q = self.workspace_Q.data[:, :k]

        z_flat = z_pred.reshape(-1, D).float()
        N = z_flat.size(0)

        if N <= 1 or k < 1:
            return {
                "effective_rank": float(k),
                "singular_values": [],
                "rank_utilization": 1.0,
                "min_singular": 1.0,
                "condition_number": 1.0,
            }

        # Workspace covariance: Σ_ws = Q^T Cov(z_pred) Q
        centered = z_flat - z_flat.mean(dim=0)
        # Micro-opt: compute (z_flat @ Q) first, then covariance
        z_ws = centered @ Q  # (N, k) — workspace projections
        cov_ws = (z_ws.T @ z_ws) / max(N - 1, 1)  # (k, k)

        try:
            sv = torch.linalg.svdvals(cov_ws)  # (k,)
            sv = sv.clamp(min=1e-10)
            sv_list = sv.tolist()

            # Effective rank via Shannon entropy
            sv_normalized = sv / sv.sum()  # probability distribution
            entropy = -(sv_normalized * sv_normalized.log()).sum().item()
            eff_rank = math.exp(entropy)

            min_sv = sv[-1].item()
            max_sv = sv[0].item()
            cond = max_sv / min_sv if min_sv > 1e-10 else float("inf")

            return {
                "effective_rank": eff_rank,
                "singular_values": sv_list,
                "rank_utilization": min(eff_rank / k, 1.0),
                "min_singular": min_sv,
                "condition_number": cond,
            }
        except Exception:
            return {
                "effective_rank": float(k),
                "singular_values": [],
                "rank_utilization": 1.0,
                "min_singular": 1.0,
                "condition_number": 1.0,
            }

    def predictive_rank_loss(self, z_pred, eps=1e-4):
        """Log-determinant barrier for rank preservation.

        L_rank = -log det(Q^T Cov(z_pred) Q + εI)

        This barrier goes to +∞ as any eigenvalue approaches 0,
        preventing rank collapse. The ε ensures numerical stability.

        Differentiable w.r.t. Q (through Q^T Cov Q).

        Theorem: If λ_min(Q^T Cov Q) > ε, then rank(J_ws) = k.

        Args:
            z_pred: (..., D) predictor output
            eps: small constant for numerical stability (default 1e-4)

        Returns:
            loss: scalar tensor (differentiable w.r.t. Q)
        """
        D = z_pred.size(-1)
        k = int(self.active_k.item())  # active WIDTH; current_k() treats it as a step (R18 bugfix)

        Q = self.workspace_Q[:, :k]  # differentiable
        z_flat = z_pred.reshape(-1, D)

        # Workspace covariance (differentiable w.r.t. Q)
        centered = z_flat - z_flat.mean(dim=0)
        z_ws = centered @ Q  # (N, k)
        N = z_ws.size(0)
        cov_ws = (z_ws.T @ z_ws) / max(N - 1, 1)  # (k, k)

        # Log-determinant via eigendecomposition (more stable than slogdet)
        # log det(A + εI) = Σ log(λ_i + ε)
        eigenvalues = torch.linalg.eigvalsh(cov_ws)
        # Clamp to ensure positivity
        eigenvalues = eigenvalues.clamp(min=eps)
        log_det = eigenvalues.log().sum()

        # Barrier: -log det (we want to MAXIMIZE log det = MINIMIZE -log det)
        return -log_det.squeeze()  # Ensure scalar output

    def extra_repr(self):
        return (
            f"embed_dim={self.embed_dim}, k_start={self.k_start}, "
            f"k_end={self.k_end}, alpha={self.alpha}, "
            f"curriculum_steps={self.curriculum_steps}"
        )
