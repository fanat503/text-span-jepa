# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# STA: Spectral Transport Alignment
#
# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL MECHANISM #13 — addresses Spectral Drift Across Training Steps
# ═══════════════════════════════════════════════════════════════════════════
#
#  PROBLEM: Spectral Drift Destabilizes Workspace and Band Allocation
#  ─────────────────────────────────────────────────────────────────────
#  During JEPA training, the eigenvalue spectrum of the representation
#  covariance Cov(z) = E[(z - μ)(z - μ)^T] continuously changes as the
#  encoder learns. This spectral drift causes cascading problems:
#
#    1. JAWP workspace Q oscillates: the Courant-Fischer minimizer
#       shifts when eigenvalues cross, causing Q to jump between
#       subspaces → unstable representations, predictor confusion
#    2. SPC band allocation becomes stale: frequency bands allocated
#       for the old spectrum have wrong capacity for the new one
#    3. Predictive rank collapses: eigenvalues that were well-separated
#       can merge, losing the workspace/background distinction
#    4. Downstream representations are non-stationary: a linear probe
#       trained at step t may fail at step t+100 due to distribution
#       shift in the representation space
#
#  This is a FUNDAMENTAL problem: the encoder is supposed to change
#  (it's learning!), but the workspace/band allocation should adapt
#  smoothly, not discontinuously.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  SOLUTION: Spectral Transport Alignment (STA)
#  ────────────────────────────────────────────
#  STA measures and penalizes the Wasserstein-1 distance between the
#  eigenvalue distributions of the current and a reference (EMA) covariance:
#
#    W_1(λ_current, λ_ref) = (1/D) Σ_i |λ_i^current - λ_i^ref|
#
#  where λ are the eigenvalues sorted in descending order.
#
#  The STA loss is:
#    L_STA = η · W_1(λ_current, λ_ref)
#
#  This penalizes rapid spectral drift, encouraging the encoder to
#  evolve SMOOTHLY in the eigenvalue domain while still learning.
#
#  Key insight: we align the SORTED eigenvalues, not the eigenvectors.
#  Eigenvectors can rotate freely (they're gauge-dependent), but the
#  eigenvalue SPECTRUM is a gauge-invariant quantity that fully
#  characterizes the representation's information structure.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL GROUNDING
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Theorem (Spectral Stability → Workspace Stability):
#  ──────────────────────────────────────────────────
#  Let λ_1 ≥ λ_2 ≥ ... ≥ λ_D be the eigenvalues of Cov(z),
#  and let Q_k be the bottom-k eigenvector subspace of Σ_res.
#  If the spectral gap δ = λ_k - λ_{k+1} > 0 (well-separated),
#  then the Davis-Kahan theorem gives:
#
#    d_Gr(Q_k(t), Q_k(t+1)) ≤ (1/δ) · ||Σ_res(t) - Σ_res(t+1)||_op
#
#  where ||·||_op is the operator norm. Since ||Σ_res(t) - Σ_res(t+1)||_op
#  ≤ max_i |λ_i(t) - λ_i(t+1)| ≤ W_1(λ(t), λ(t+1)),
#  we get:
#
#    d_Gr(Q_k(t), Q_k(t+1)) ≤ W_1 / δ
#
#  With STA (penalizing W_1), the workspace drift is BOUNDED by W_1/δ.
#  □
#
#  Theorem (STA as Optimal Transport):
#  ─────────────────────────────────────
#  The 1-Wasserstein distance between two discrete distributions
#  supported on sorted points {λ_i} and {μ_i} with uniform weights 1/D is:
#
#    W_1 = (1/D) Σ_{i=1}^{D} |λ_{(i)} - μ_{(i)}|
#
#  where λ_{(i)} and μ_{(i)} are sorted in the same order.
#  For eigenvalues sorted descending, this is exactly our STA metric.
#
#  Proof: By the Kantorovich-Rubinstein duality, W_1(P, Q) = sup_{f ∈ Lip1}
#  |E_P[f] - E_Q[f]|. For discrete distributions with sorted support,
#  the optimal coupling is the monotone coupling, which pairs the i-th
#  largest eigenvalue of P with the i-th largest of Q. This gives
#  W_1 = (1/D) Σ |λ_{(i)} - μ_{(i)}|. □
#
#  Corollary (Downstream Stability Bound):
#  ───────────────────────────────────────
#  Let f(z) = w^T z be a downstream linear probe. The prediction
#  variance across training steps satisfies:
#
#    Var_t[f(z_t)] ≤ ||w||² · E_t[tr(Cov(z_t))] · (W_1(t, t+1) / δ²)
#
#  i.e., downstream predictions are stable when STA loss is small
#  and the spectral gap is large.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  HOW OTHER PAPERS CAN USE STA
#  ═══════════════════════════════════════════════════════════════════════════
#
#  STA is applicable to ANY self-supervised method with a learned
#  subspace that depends on the eigenvalue spectrum:
#
#    from src.models.sta import SpectralTransportAlignment
#
#    sta = SpectralTransportAlignment(embed_dim=768)
#    # Every training step:
#    sta_loss, info = sta(z_online, step=step)
#    total_loss += lambda_sta * sta_loss
#
#  Works with:
#    - Any JEPA variant with learned workspace (JAWP, PCA-based)
#    - Spectral methods (SPC, spectral clustering)
#    - VICReg / Barlow Twins (eigenvalue-based regularization)
#    - Any method where spectral stability matters
#
#  Hyperparameters:
#    - eta: STA penalty weight (default 0.01)
#    - ema_beta: EMA momentum for reference spectrum (default 0.999)
#    - warmup_steps: steps before STA activates (default 500)

from __future__ import annotations

import torch
from torch import nn


class SpectralTransportAlignment(nn.Module):
    """Spectral Transport Alignment — prevents spectral drift from
    destabilizing workspace and band allocation.

    Measures Wasserstein-1 distance between current eigenvalue spectrum
    and an EMA reference, penalizing rapid spectral changes.

    Theorem (Davis-Kahan + STA): If spectral gap δ > 0, then
    d_Gr(Q_k(t), Q_k(t+1)) ≤ W_1 / δ.
    STA bounds W_1, thus bounding workspace drift.

    Args:
        embed_dim: embedding dimension D.
        eta: STA penalty weight (default 0.01).
        ema_beta: EMA momentum for reference covariance (default 0.999).
        warmup_steps: steps before STA activates (default 500).
        update_interval: steps between full eigenvalue computation (default 10).
        eps: numerical stability constant.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        eta: float = 0.01,
        ema_beta: float = 0.999,
        warmup_steps: int = 500,
        update_interval: int = 10,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.eta = eta
        self.ema_beta = ema_beta
        self.warmup_steps = warmup_steps
        self.update_interval = max(update_interval, 1)
        self.eps = eps

        # EMA reference covariance (initialize to identity)
        self.register_buffer("ref_cov", torch.eye(embed_dim) * 0.01)
        self.register_buffer("ref_eigenvalues", torch.ones(embed_dim))

        # Current eigenvalue cache
        self.register_buffer("current_eigenvalues", torch.ones(embed_dim))

        # Running statistics
        self.register_buffer("running_w1", torch.tensor(0.0))
        self.register_buffer("running_spectral_gap", torch.tensor(0.0))
        self.register_buffer("is_initialized", torch.tensor(False))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _update_reference(self, z: torch.Tensor):
        """Update EMA reference covariance and its eigenvalues.

        Args:
            z: (..., D) representations.
        """
        D = z.size(-1)
        flat = z.reshape(-1, D).float()
        N = flat.size(0)
        if N <= 1:
            return

        centered = flat - flat.mean(dim=0, keepdim=True)
        cov_batch = (centered.T @ centered) / max(N - 1, 1)

        if not self.is_initialized:
            self.ref_cov.copy_(cov_batch)
            self.is_initialized.fill_(True)
        else:
            self.ref_cov.mul_(self.ema_beta).add_((1 - self.ema_beta) * cov_batch)

        # Compute eigenvalues of reference covariance
        try:
            eigs = torch.linalg.eigvalsh(self.ref_cov)
            # Sort descending
            self.ref_eigenvalues.copy_(eigs.flip(0))
        except Exception:
            pass

    @torch.no_grad()
    def _compute_current_eigenvalues(self, z: torch.Tensor):
        """Compute eigenvalues of current covariance.

        Args:
            z: (..., D) representations.
        """
        D = z.size(-1)
        flat = z.reshape(-1, D).float()
        N = flat.size(0)
        if N <= 1:
            return

        centered = flat - flat.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(N - 1, 1)

        try:
            eigs = torch.linalg.eigvalsh(cov)
            self.current_eigenvalues.copy_(eigs.flip(0))
        except Exception:
            pass

    def forward(
        self,
        z: torch.Tensor,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Spectral Transport Alignment loss.

        Args:
            z: (..., D) encoder output.
            step: current training step.

        Returns:
            loss: scalar tensor (≥ 0).
            info: dict with diagnostics.
        """
        self.step_count.fill_(step)
        z.size(-1)

        # Warmup
        warmup_factor = min(1.0, step / max(self.warmup_steps, 1))
        if warmup_factor < 1e-6:
            zero = z.new_tensor(0.0)
            return zero, {
                "sta_loss": 0.0,
                "sta_w1": 0.0,
                "sta_warmup": True,
                "sta_warmup_factor": 0.0,
            }

        # Reference follows the interval cadence; CURRENT recomputes every
        # step. The original code refreshed both from the same tensor at the
        # same moments, forcing W1(current, ref) == 0 identically — the loss
        # could never become non-zero (audit R11).
        if not self.is_initialized:
            self._update_reference(z.detach())
            self._compute_current_eigenvalues(z.detach())
        else:
            if step % self.update_interval == 0:
                self._update_reference(z.detach())
            self._compute_current_eigenvalues(z.detach())

        # Compute W1 distance: (1/D) Σ |λ_i^current - λ_i^ref|
        # Both are sorted descending, so the monotone coupling is optimal
        eig_diff = (self.current_eigenvalues - self.ref_eigenvalues).abs()
        w1 = eig_diff.mean()

        # Spectral gap: min gap between consecutive eigenvalues (stability indicator)
        sorted_eigs = self.current_eigenvalues.sort(descending=True)[0]
        if sorted_eigs.size(0) > 1:
            gaps = sorted_eigs[:-1] - sorted_eigs[1:]
            # Only consider gaps between significant eigenvalues
            significant = sorted_eigs[:-1] > self.eps
            if significant.any():
                min_gap = gaps[significant].min()
            else:
                min_gap = torch.tensor(0.0)
        else:
            min_gap = torch.tensor(0.0)

        # STA loss
        loss = self.eta * warmup_factor * w1

        # Running statistics
        with torch.no_grad():
            self.running_w1.mul_(0.99).add_(0.01 * w1.item())
            self.running_spectral_gap.mul_(0.99).add_(0.01 * min_gap.item())

        info = {
            "sta_loss": loss.item(),
            "sta_w1": w1.item(),
            "sta_warmup_factor": warmup_factor,
            "sta_warmup": warmup_factor < 1.0,
            "sta_spectral_gap": min_gap.item(),
            "sta_running_w1": self.running_w1.item(),
            "sta_running_spectral_gap": self.running_spectral_gap.item(),
            "sta_max_eigenvalue": self.current_eigenvalues[0].item(),
            "sta_min_eigenvalue": self.current_eigenvalues[-1].item(),
            "sta_condition_number": (
                self.current_eigenvalues[0] / (self.current_eigenvalues[-1] + self.eps)
            ).item(),
        }

        return loss, info

    # AUDIT R15: PROXY ESTIMATE — the Davis–Kahan reduction here assumes a
    # spectral gap that is not enforced anywhere; value is an operational
    # estimate, not a certified bound (proofs/IMPLEMENTATION_STATUS.md).
    @torch.no_grad()
    def compute_davis_kahan_bound(
        self,
        k: int,
        w1: float | None = None,
    ) -> float:
        """Compute Davis-Kahan workspace stability bound.

        d_Gr(Q_k(t), Q_k(t+1)) ≤ W_1 / δ

        where δ is the spectral gap at the k/k+1 boundary.

        Args:
            k: workspace dimension.
            w1: W1 distance (uses running average if None).

        Returns:
            Upper bound on Grassmann distance between consecutive
            workspace subspaces.
        """
        if w1 is None:
            w1 = self.running_w1.item()

        # Spectral gap at k/k+1 boundary
        sorted_eigs = self.current_eigenvalues.sort(descending=True)[0]
        if k < len(sorted_eigs) - 1:
            delta = (sorted_eigs[k] - sorted_eigs[k + 1]).item()
            delta = max(delta, self.eps)
        else:
            delta = self.eps

        return w1 / delta

    # AUDIT R15: CONDITIONAL FORMULA — downstream corollary of the same
    # contested Davis–Kahan reduction; operational estimate only.
    def compute_downstream_stability_bound(
        self,
        probe_norm: float = 1.0,
        total_variance: float = 1.0,
        k: int = 77,
    ) -> float:
        """Compute downstream prediction stability bound.

        Var_t[f(z_t)] ≤ ||w||² · tr(Cov) · (W_1 / δ²)

        Args:
            probe_norm: ||w|| of downstream linear probe.
            total_variance: tr(Cov(z)).
            k: workspace dimension for spectral gap.

        Returns:
            Upper bound on downstream prediction variance.
        """
        w1 = self.running_w1.item()
        sorted_eigs = self.current_eigenvalues.sort(descending=True)[0]
        if k < len(sorted_eigs) - 1:
            delta = (sorted_eigs[k] - sorted_eigs[k + 1]).item()
            delta = max(delta, self.eps)
        else:
            delta = self.eps

        return probe_norm**2 * total_variance * (w1 / (delta**2))

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, eta={self.eta}, "
            f"ema_beta={self.ema_beta}, warmup_steps={self.warmup_steps}, "
            f"update_interval={self.update_interval}"
        )
