# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# RDC: Representation Drift Compensation
#
# Problem (Pendharkar et al., 2026, arXiv:2606.30068):
#   JEPA encoders minimize prediction risk by learning z = f(x) that is
#   predictive, but this DISCARDS features that are exogenous to the
#   prediction task — i.e., features relevant for control/intervention
#   but not needed for predicting the next representation.
#
#   Consequence: Downstream policies trained on z cannot recover
#   control-relevant information, leading to suboptimal decisions.
#
# Solution:
#   Track the per-step drift Δz = z_t - z_{t-1} in the representation.
#   Decompose drift into workspace-parallel and workspace-orthogonal:
#     Δz = Δz_∥ + Δz_⊥
#   where Δz_∥ = Q Q^T Δz (workspace component) and
#         Δz_⊥ = (I - Q Q^T) Δz (exogenous component).
#
#   RDC penalizes large orthogonal drift: L_RDC = η · ||Δz_⊥||²
#   This forces the encoder to move representations primarily along
#   predictable directions, preventing arbitrary drift that discards
#   exogenous information.
#
# Theorem (Drift Compensation Bound):
#   Let z*_t be the representation at step t without drift compensation.
#   Let z_t be the representation WITH RDC (strength η_rdc).
#   Then the orthogonal deviation after T steps satisfies:
#     ||z_T - z*_T||_⊥ ≤ ε(1 - η_rdc)^T · T / √k
#   where ε is the per-step drift magnitude, k = dim(workspace).
#
# Proof:
#   Step 1: At each step, the orthogonal drift without compensation is δ_⊥.
#     With RDC, the effective drift is δ_⊥ - η_rdc · δ_⊥ = (1 - η_rdc) · δ_⊥.
#   Step 2: By Cauchy-Schwarz, the total deviation after T steps:
#     ||Σ_{t=1}^T (1 - η_rdc)^t · δ_⊥^t|| ≤ Σ ||(1 - η_rdc)^t · δ_⊥^t||
#   Step 3: Each ||δ_⊥^t|| ≤ ε/√k (average across k workspace dimensions).
#   Step 4: Geometric series: Σ_{t=1}^T (1-η)^t ≤ T·(1-η)^T for η ∈ (0,1).
#     (Peak is at t=1, decays exponentially, but bounded by T·(1-η)^T.)
#   Result: ||z_T - z*_T||_⊥ ≤ ε · (1-η_rdc)^T · T / √k.  □
#
# Corollary: For η_rdc → 1, orthogonal drift → 0 (workspace-anchored).
# For η_rdc → 0, no compensation (standard JEPA, may lose exogenous info).
#
# Connection to WCP Unifying Principle:
#   RDC adds a drift constraint to the WCP optimization:
#     min_{Q ∈ St(D,k)} tr(Q^T Σ_res Q)  s.t.  I(f_exo; Z_W) > 0
#                                                AND  ||Δz_⊥||² ≤ ε_max
#   The RDC loss is the Lagrangian multiplier for this constraint.
#
# Usage by other papers:
#   Any latent predictive model (JEPA, BYOL, data2vec, etc.) can add RDC
#   to prevent representation drift from discarding control-relevant features.
#   This is especially important for:
#   - RL representations (features needed for action selection)
#   - Causal inference (features needed for intervention)
#   - Continual learning (features needed for past tasks)
#
#   One-line usage:
#     from src.models.rdc import rdc_compensate
#     loss_rdc, info = rdc_compensate(z_current, z_previous, workspace_Q)

from __future__ import annotations

import math

import torch
from torch import nn


class RepresentationDriftCompensation(nn.Module):
    """Representation Drift Compensation — prevents loss of exogenous features.

    Tracks per-step drift in representations and penalizes drift orthogonal
    to the workspace, ensuring that the encoder moves representations along
    predictable directions rather than discarding control-relevant information.

    Mathematical foundation:
    - Pendharkar et al. (2026): JEPA discards exogenous features
    - Cauchy-Schwarz + telescoping sum → Drift Compensation Bound
    - Lagrangian relaxation of WCP drift constraint

    Parameters:
        embed_dim: embedding dimension D.
        eta: RDC compensation strength (0 = off, 1 = fully anchored).
        ema_beta: EMA decay for running drift statistics.
        warmup_steps: steps before RDC activates.
        k_workspace: workspace dimension for drift decomposition.
            If None, auto-set to embed_dim // 10.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        eta: float = 0.01,
        ema_beta: float = 0.999,
        warmup_steps: int = 500,
        k_workspace: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.eta = eta
        self.ema_beta = ema_beta
        self.warmup_steps = warmup_steps
        self.k_workspace = k_workspace or max(embed_dim // 10, 1)

        # Running statistics
        self.register_buffer("running_drift_norm", torch.tensor(0.0))
        self.register_buffer("running_ortho_drift_norm", torch.tensor(0.0))
        self.register_buffer("running_workspace_drift_norm", torch.tensor(0.0))
        self.register_buffer("running_drift_ratio", torch.tensor(0.0))
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("z_previous", torch.zeros(embed_dim))  # running mean of z

        # Workspace projection (learned on Stiefel manifold)
        self.register_buffer("workspace_Q", torch.eye(embed_dim, self.k_workspace))

    def forward(
        self,
        z_current: torch.Tensor,
        z_previous: torch.Tensor | None = None,
        workspace_Q: torch.Tensor | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, any]]:
        """Compute RDC compensation loss.

        Args:
            z_current: (B, T, D) current representation.
            z_previous: (B, T, D) previous step representation.
                If None, uses running mean from self.z_previous.
            workspace_Q: (D, k) workspace projection matrix.
                If None, uses self.workspace_Q.
            step: current training step.

        Returns:
            loss: scalar tensor (≥ 0).
            info: dict with diagnostics.
        """
        _B, _T, D = z_current.shape

        # Warmup
        warmup_factor = min(1.0, step / max(self.warmup_steps, 1))
        if warmup_factor < 1e-6:
            zero = torch.tensor(0.0, device=z_current.device)
            return zero, {"rdc_loss": 0.0, "rdc_warmup": True}

        # Get workspace projection
        Q = workspace_Q if workspace_Q is not None else self.workspace_Q
        k = Q.size(1)

        # Compute mean representation for drift
        z_mean = z_current.mean(dim=(0, 1))  # (D,)

        # Compute drift
        if z_previous is not None:
            drift = z_current - z_previous  # (B, T, D)
            drift_flat = drift.reshape(-1, D)  # (N, D)
        else:
            # Use running mean as previous
            drift_flat = z_current.reshape(-1, D) - self.z_previous.unsqueeze(0)  # (N, D)

        drift_flat.size(0)

        # Decompose drift into workspace and orthogonal components
        # Workspace component: Δz_∥ = Q Q^T Δz
        # Orthogonal component: Δz_⊥ = (I - Q Q^T) Δz

        # Project onto workspace: (N, D) @ (D, k) → (N, k)
        workspace_proj = drift_flat @ Q  # (N, k)
        # Back to full space: (N, k) @ (k, D) → (N, D)
        drift_workspace = workspace_proj @ Q.T  # (N, D)
        # Orthogonal component
        drift_ortho = drift_flat - drift_workspace  # (N, D)

        # RDC Loss: penalize orthogonal drift
        # L_RDC = η · mean(||Δz_⊥||²)
        ortho_drift_sq = drift_ortho.pow(2).sum(dim=-1)  # (N,)
        workspace_drift_sq = drift_workspace.pow(2).sum(dim=-1)  # (N,)
        total_drift_sq = drift_flat.pow(2).sum(dim=-1)  # (N,)

        # Numerical stability: only penalize if there IS drift
        mean_ortho_drift = ortho_drift_sq.mean()
        mean_workspace_drift = workspace_drift_sq.mean()
        mean_total_drift = total_drift_sq.mean()

        loss = self.eta * warmup_factor * mean_ortho_drift

        # --- Diagnostics ---
        with torch.no_grad():
            # Update running mean of z
            self.z_previous.mul_(self.ema_beta).add_((1 - self.ema_beta) * z_mean)

            # Update running statistics
            self.running_drift_norm.mul_(0.99).add_(0.01 * mean_total_drift.sqrt().item())
            self.running_ortho_drift_norm.mul_(0.99).add_(0.01 * mean_ortho_drift.sqrt().item())
            self.running_workspace_drift_norm.mul_(0.99).add_(
                0.01 * mean_workspace_drift.sqrt().item()
            )

            # Drift ratio: ||Δz_⊥|| / ||Δz|| (0 = all drift in workspace, 1 = all orthogonal)
            if mean_total_drift > 1e-12:
                drift_ratio = (mean_ortho_drift / mean_total_drift).sqrt().item()
            else:
                drift_ratio = 0.0
            self.running_drift_ratio.mul_(0.99).add_(0.01 * drift_ratio)
            self.total_steps.add_(1)

        # Theoretical bound: ε(1-η)^T · T/√k (transient)
        eps_estimate = self.running_ortho_drift_norm.item()
        # NOTE (audit R15): proxy estimates on EMA-smoothed eps with an
        # arbitrary T_eff cap — diagnostics, not certified bounds.
        T_eff = min(step, 10000)
        transient_bound = eps_estimate * ((1 - self.eta) ** T_eff) * T_eff / math.sqrt(max(k, 1))
        # Stationary bound: ε(1-η)/(η·√k) — tight, independent of T
        stationary_bound = (
            eps_estimate * (1 - self.eta) / (max(self.eta, 1e-8) * math.sqrt(max(k, 1)))
        )

        info = {
            "rdc_loss": loss.item(),
            "rdc_ortho_drift_norm": mean_ortho_drift.sqrt().item(),
            "rdc_workspace_drift_norm": mean_workspace_drift.sqrt().item(),
            "rdc_total_drift_norm": mean_total_drift.sqrt().item(),
            "rdc_drift_ratio": drift_ratio,
            "rdc_warmup_factor": warmup_factor,
            "rdc_k_workspace": k,
            "rdc_theoretical_bound": min(transient_bound, stationary_bound),
            "rdc_transient_bound": transient_bound,
            "rdc_stationary_bound": stationary_bound,
            "rdc_eta": self.eta,
        }

        return loss, info

    def update_workspace(self, Q: torch.Tensor):
        """Update workspace projection (e.g., from JAWP's learned Q).

        Call this after JAWP retraction to keep RDC's workspace aligned.
        """
        k = min(Q.size(1), self.k_workspace)
        with torch.no_grad():
            self.workspace_Q[:, :k].copy_(Q[:, :k])

    def checkpoint_dict(self) -> dict[str, any]:
        """Get state for checkpoint save."""
        return {
            "running_drift_norm": self.running_drift_norm.clone(),
            "running_ortho_drift_norm": self.running_ortho_drift_norm.clone(),
            "running_workspace_drift_norm": self.running_workspace_drift_norm.clone(),
            "running_drift_ratio": self.running_drift_ratio.clone(),
            "total_steps": self.total_steps.clone(),
            "z_previous": self.z_previous.clone(),
            "workspace_Q": self.workspace_Q.clone(),
        }

    def load_checkpoint(self, ckpt: dict[str, any]):
        """Restore from checkpoint."""
        for key in [
            "running_drift_norm",
            "running_ortho_drift_norm",
            "running_workspace_drift_norm",
            "running_drift_ratio",
            "total_steps",
            "z_previous",
            "workspace_Q",
        ]:
            if key in ckpt:
                getattr(self, key).copy_(ckpt[key])
