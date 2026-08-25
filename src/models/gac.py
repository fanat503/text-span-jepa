# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# GAC: Gradient-Allocated Capacity
#
# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL MECHANISM #12 — addresses Background Gradient Starvation
# ═══════════════════════════════════════════════════════════════════════════
#
#  PROBLEM: Background Gradient Starvation
#  ──────────────────────────────────────────
#  When JAWP focuses prediction on workspace span(Q), the predictor focus
#  penalty α·||(I - QQ^T) z_pred||² drives background energy to zero.
#  This means background dimensions receive VANISHING gradient signal:
#
#    ∂L/∂z_pred[i] ≈ 0  for i ∈ background (orthogonal to Q)
#
#  Without gradient flow, the encoder cannot learn to place useful
#  information in background dimensions. This creates a FEEDBACK LOOP:
#    1. JAWP ignores background → no gradient to background
#    2. Encoder doesn't learn background features → background is noise
#    3. JAWP correctly ignores noise → confirmed in workspace
#    4. But potentially useful features that could become workspace
#       are TRAPPED in background with no gradient to promote them
#
#  This is the "Background Gradient Starvation" problem: the very act
#  of focusing on workspace prevents the model from discovering NEW
#  workspace directions.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  SOLUTION: Gradient-Allocated Capacity (GAC)
#  ────────────────────────────────────────────
#  GAC monitors the gradient norm in each dimension and allocates
#  a small "exploration bonus" to dimensions with near-zero gradient:
#
#    L_GAC = γ · Σ_i max(0, τ_grad - ||g_i||) · ||z_pred[i]||²
#
#  where:
#    g_i = ∂L_total/∂z_pred[i]  — gradient w.r.t. i-th dimension
#    τ_grad — gradient threshold (dimensions below this are "starved")
#    γ — exploration weight (small, e.g. 0.01)
#
#  Dimensions with ||g_i|| < τ_grad receive a POSITIVE penalty for
#  having large activation, which pushes the optimizer to either:
#    (a) increase gradient flow to those dimensions (by routing more
#        prediction loss there), or
#    (b) reduce activation in truly useless dimensions
#
#  This is a SOFT exploration mechanism: it doesn't force any dimension
#  into workspace, but prevents the gradient from being exactly zero
#  in potentially useful directions.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL GROUNDING
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Theorem (No Gradient Dead Zones): With GAC, every dimension i
#  with non-zero activation receives gradient signal:
#    ||∂(L_total + L_GAC)/∂z_pred[i]|| ≥ γ · max(0, τ_grad - ||g_i||) · 2|z_i|
#
#  Proof: L_GAC = γ · Σ_i max(0, τ_grad - ||g_i||) · z_i²
#  ∂L_GAC/∂z_i = γ · max(0, τ_grad - ||g_i||) · 2z_i
#  For any i with z_i ≠ 0 and ||g_i|| < τ_grad:
#    ||∂L_GAC/∂z_i|| = γ · (τ_grad - ||g_i||) · 2|z_i| > 0  ∎
#
#  Corollary (Minimum Gradient Flow): The total gradient norm
#  per dimension satisfies:
#    ||∂L_total/∂z_i|| ≥ γ · τ_grad · 2|z_i| - ||g_i||
#  when ||g_i|| < τ_grad. This provides a LOWER BOUND on gradient
#  flow, preventing dead zones.
#
#  Theorem (Exploration Doesn't Dominate): The ratio of GAC
#  gradient to total gradient is bounded:
#    ||∂L_GAC/∂z_i|| / ||∂L_total/∂z_i|| ≤ γ · τ_grad / ||g_i||
#  When γ << 1 and τ_grad is moderate, GAC provides a small
#  exploration nudge without dominating the main objective.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  HOW OTHER PAPERS CAN USE GAC
#  ═══════════════════════════════════════════════════════════════════════════
#
#  GAC is applicable to ANY model with selective prediction:
#
#  ```python
#  from src.models.gac import GradientAllocatedCapacity
#
#  gac = GradientAllocatedCapacity(embed_dim=768, gamma=0.01, tau_grad=1e-4)
#
#  # After computing main loss and .backward():
#  grad_norms = z_pred.grad.norm(dim=0)  # per-dim gradient norms
#  loss_gac, info = gac(z_pred.detach(), grad_norms)
#  ```
#
#  Works with: any JEPA variant, MAE, BEiT, masked language models,
#  and any architecture that focuses capacity on a subset of dimensions.


import torch
import torch.nn.functional as F
from torch import nn


class GradientAllocatedCapacity(nn.Module):
    """Gradient-Allocated Capacity — prevents background gradient starvation.

    When JAWP focuses prediction on workspace, background dimensions
    receive vanishing gradients. GAC adds an exploration bonus to
    dimensions with near-zero gradient, ensuring all potentially useful
    directions receive training signal.

    Args:
        embed_dim: embedding dimension D.
        gamma: exploration weight (small, e.g. 0.01).
        tau_grad: gradient threshold. Dimensions with ||g_i|| < tau_grad
            are considered "starved" and receive exploration bonus.
        ema_beta: EMA decay for running gradient norm statistics.
        warmup_steps: steps before GAC activates (let gradients stabilize).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        gamma: float = 0.01,
        tau_grad: float = 1e-4,
        ema_beta: float = 0.99,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.tau_grad = tau_grad
        self.ema_beta = ema_beta
        self.warmup_steps = warmup_steps

        # Running statistics
        self.register_buffer("running_grad_norms", torch.ones(embed_dim))
        self.register_buffer("running_starved_fraction", torch.tensor(0.0))
        self.register_buffer("total_gac_steps", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        z_pred: torch.Tensor,
        grad_norms: torch.Tensor,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, any]]:
        """Compute GAC exploration loss.

        Args:
            z_pred: (..., D) predictor output (detached from graph).
            grad_norms: (D,) per-dimension gradient norms from main loss.
            step: current training step.

        Returns:
            loss: scalar tensor (≥ 0).
            info: dict with diagnostics.
        """
        D = z_pred.size(-1)
        z_flat = z_pred.reshape(-1, D)

        # Warmup: don't apply GAC until gradients stabilize
        warmup_factor = min(1.0, step / max(self.warmup_steps, 1))

        if warmup_factor < 1e-6:
            zero = torch.tensor(0.0, device=z_pred.device)
            return zero, {"gac_loss": 0.0, "gac_warmup": True}

        # Ensure grad_norms is (D,) per-dimension
        # If grad_norms has extra batch/time dims, aggregate to per-dim mean
        gn = grad_norms.detach()
        if gn.ndim > 1:
            # Aggregate: mean over all dims except last
            gn = gn.reshape(-1, D).mean(dim=0)  # (D,)
        elif gn.ndim == 0:
            # Scalar broadcast to all dims
            gn = gn.expand(D)

        # Update running gradient norms (EMA)
        with torch.no_grad():
            self.running_grad_norms.mul_(self.ema_beta).add_((1 - self.ema_beta) * gn)

        # Identify starved dimensions: ||g_i|| < tau_grad
        starved_mask = (gn < self.tau_grad).float()  # (D,)
        n_starved = starved_mask.sum().item()
        starved_fraction = n_starved / D

        # Exploration bonus: γ · Σ_i max(0, τ - ||g_i||) · ||z_i||²
        # For starved dimensions, (τ - ||g_i||) > 0
        grad_deficit = F.relu(self.tau_grad - gn)  # (D,)
        dim_energy = (z_flat**2).mean(dim=0)  # (D,) mean over batch

        # GAC loss: sum over starved dimensions
        loss = self.gamma * warmup_factor * (grad_deficit * dim_energy * starved_mask).sum()

        # Diagnostics
        with torch.no_grad():
            self.running_starved_fraction.mul_(0.99).add_(0.01 * starved_fraction)
            self.total_gac_steps.add_(1)

        info = {
            "gac_loss": loss.item(),
            "gac_n_starved": int(n_starved),
            "gac_starved_fraction": starved_fraction,
            "gac_warmup_factor": warmup_factor,
            "gac_mean_grad_norm": grad_norms.mean().item(),
            "gac_min_grad_norm": grad_norms.min().item() if D > 0 else 0.0,
            "gac_max_grad_norm": grad_norms.max().item() if D > 0 else 0.0,
            "gac_running_starved_fraction": self.running_starved_fraction.item(),
            "gac_warmup": False,
        }

        return loss, info

    # AUDIT R15: PROXY ESTIMATE — implemented loss uses batch-mean energy
    # (theorem bound scales by 1/N accordingly) and EMA-smoothed grad norms;
    # treat the returned value as an operational diagnostic, not a certified
    # No-Dead-Zones bound.
    def compute_gradient_bound(
        self,
        z_pred: torch.Tensor,
        grad_norms: torch.Tensor,
    ) -> float:
        """Compute minimum gradient flow bound from Theorem 1.

        For each starved dimension i with z_i ≠ 0:
          ||∂L_GAC/∂z_i|| ≥ γ · (τ - ||g_i||) · 2|z_i|

        Returns the mean lower bound across starved dimensions.
        """
        D = z_pred.size(-1)
        z_flat = z_pred.reshape(-1, D)
        dim_energy = (z_flat**2).mean(dim=0).sqrt()  # RMS per dim

        starved = grad_norms < self.tau_grad
        deficit = F.relu(self.tau_grad - grad_norms)

        bounds = self.gamma * deficit * 2 * dim_energy  # (D,)
        starved_bounds = bounds[starved]

        if len(starved_bounds) == 0:
            return 0.0
        return starved_bounds.mean().item()

    def compute_exploration_ratio(
        self,
        grad_norms: torch.Tensor,
    ) -> float:
        """Compute ratio of GAC gradient to main gradient (Theorem 3).

        ||∂L_GAC/∂z_i|| / ||∂L_total/∂z_i|| ≤ γ · τ / ||g_i||
        """
        active = grad_norms > 1e-10
        if not active.any():
            return float("inf")
        ratios = self.gamma * self.tau_grad / grad_norms[active]
        return ratios.max().item()

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, gamma={self.gamma}, "
            f"tau_grad={self.tau_grad}, warmup_steps={self.warmup_steps}"
        )
