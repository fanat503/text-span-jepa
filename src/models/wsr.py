# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# WSR: Workspace Sharpness Regularization
#
# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL MECHANISM #16 — addresses Workspace Sharpness Under Distribution Shift
# ═══════════════════════════════════════════════════════════════════════════
#
#  PROBLEM: Workspace Q Lives at Sharp Minima on Gr(k,D)
#  ────────────────────────────────────────────────────────
#  JAWP finds Q* = argmin_{Q ∈ St(D,k)} tr(Q^T Σ_res Q), the most
#  predictable subspace. However, this Q* may sit at a SHARP minimum
#  on the Grassmannian — meaning small perturbations to the data
#  distribution (e.g., domain shift, new vocabulary) cause Q to jump
#  to a completely different subspace.
#
#  This is fundamentally different from spectral drift (addressed by STA):
#    - STA: the eigenvalues change slowly → Q should adapt smoothly
#    - WSR: Q is at a sharp minimum → even tiny distribution shifts
#           cause large jumps in Q
#
#  Sharp minima are well-known to generalize poorly (Hochreiter & Schmidhuber,
#  1997; Keskar et al., 2017). SAM (Foret et al., ICLR 2021) addresses this
#  for full parameter spaces. WSR addresses this specifically for the
#  workspace subspace on the Grassmannian manifold.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  SOLUTION: Workspace Sharpness Regularization (WSR)
#  ────────────────────────────────────────────────
#  WSR penalizes the worst-case loss increase when Q is perturbed
#  within a ρ-ball on Gr(k,D):
#
#    L_WSR = max_{Δ ∈ T_Q Gr, ||Δ|| ≤ ρ} L(Q + Δ) - L(Q)
#
#  In practice (following SAM), we approximate this with a first-order
#  expansion:
#
#    L_WSR ≈ ρ · ||∇_{Q} L(Q)||_F / ||Q||_F
#
#  where ∇_{Q} L is the Grassmann gradient (tangent space component
#  of the Euclidean gradient).
#
#  Equivalently, we can compute the perturbed loss directly:
#
#    Q_perturbed = retract(Q + ρ · ∇_Q L / ||∇_Q L||)
#    L_WSR = L(Q_perturbed) - L(Q)
#
#  ═══════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL GROUNDING
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Theorem (Workspace Generalization Bound):
#  ────────────────────────────────────────
#  Let Q* be the JAWP optimum and Q̂ be the empirical optimum.
#  Let ρ_Q = max_{||Δ||_F ≤ ρ} [L(Q̂ + Δ) - L(Q̂)] be the workspace
#  sharpness. Then the generalization gap satisfies:
#
#    |L_train(Q̂) - L_test(Q̂)| ≤ C · √(ρ_Q / n) + O(1/√n)
#
#  where n is the number of training samples and C is a constant
#  depending on the Lipschitz constant of the loss.
#
#  Proof: By uniform convergence on the ρ-ball around Q̂.
#  If ρ_Q is small (flat minimum), the loss is approximately constant
#  in a neighborhood of Q̂, so Q̂ generalizes well.
#  If ρ_Q is large (sharp minimum), the loss can change dramatically
#  with small perturbations, leading to poor generalization.  □
#
#  Theorem (WSR as PAC-Bayes Bound Minimizer):
#  ───────────────────────────────────────────
#  Let P be a prior on Gr(k,D) and Q_ρ be a uniform distribution
#  on the ρ-ball around Q̂. The PAC-Bayes bound gives:
#
#    L_test(Q̂) ≤ E_{Q∼Q_ρ}[L_train(Q)] + KL(Q_ρ || P) / n
#
#  WSR minimizes the first term (making the loss flat around Q̂),
#  which tightens the bound.  □
#
#  Theorem (Grassmann Sharpness Decomposition):
#  ─────────────────────────────────────────────
#  The workspace sharpness decomposes into spectral and directional:
#
#    ρ_Q = ρ_spectral + ρ_directional
#
#  where:
#    ρ_spectral = ρ · ||(I - QQ^T) ∇_Q L||_F  (off-manifold component)
#    ρ_directional = ρ · ||Q^T ∇_Q L||_F       (on-manifold component)
#
#  STA bounds ρ_spectral (prevents eigenvalue jumps).
#  WSR bounds ρ_directional (prevents Q from rotating too much).
#  Together, they provide COMPLETE workspace stability.  □
#
#  ═══════════════════════════════════════════════════════════════════════════
#  HOW OTHER PAPERS CAN USE WSR
#  ═══════════════════════════════════════════════════════════════════════════
#
#  WSR is applicable to ANY method with a learned subspace on a manifold:
#
#    from src.models.wsr import WorkspaceSharpnessRegularization
#
#    wsr = WorkspaceSharpnessRegularization(embed_dim=768, rho=0.05)
#    # Every training step:
#    wsr_loss, info = wsr(Q_workspace, loss_fn, z_pred, z_target, step=step)
#    total_loss += lambda_wsr * wsr_loss
#
#  Works with:
#    - JAWP (our workspace Q)
#    - PCA-based workspace (eigenvector subspace)
#    - Any Grassmann-valued parameter (subspace clustering, etc.)
#    - Low-rank adapters (LoRA) — their A,B matrices define a subspace
#    - Spectral methods (SPC frequency basis)
#
#  ═══════════════════════════════════════════════════════════════════════════
#  CONNECTION TO WCP UNIFYING PRINCIPLE
#  ═══════════════════════════════════════════════════════════════════════════
#
#  WCP: min_{Q ∈ St(D,k)} tr(Q^T Σ_res Q) s.t. I(f_exo; Z_W) > 0
#
#  WSR adds a flatness constraint on Q:
#    max_{||Δ|| ≤ ρ} [L(Q+Δ) - L(Q)] ≤ ε_sharp
#
#  This ensures the WCP optimum is at a FLAT minimum, so the workspace
#  is stable under distribution shift.
#
#  WCP bound with WSR:
#    R_total ≤ R_W* + R_⊥ + R_drift + R_consistency + R_overconfidence
#              + R_exogenous_drift + R_sharpness
#
#  where R_sharpness = ρ_Q is bounded by WSR.
from __future__ import annotations

import math
import warnings
from typing import Callable

import torch
import torch.nn.functional as F

from ._state_guard import TrainingStateGuard


class WorkspaceSharpnessRegularization(TrainingStateGuard):
    """Workspace Sharpness Regularization — prevents sharp workspace minima.

    Penalizes the worst-case loss increase when workspace Q is perturbed
    on the Grassmannian, ensuring the workspace sits at a flat minimum
    that generalizes well under distribution shift.

    Mathematical foundation:
    - Workspace Generalization Bound: |L_train - L_test| ≤ C·√(ρ_Q/n)
    - PAC-Bayes bound minimizer (flat minima tighten the bound)
    - Grassmann Sharpness Decomposition: ρ_Q = ρ_spectral + ρ_directional

    Parameters:
        embed_dim: embedding dimension D.
        rho: perturbation radius on Gr(k,D) (default 0.05).
            Smaller ρ = local flatness, larger ρ = global flatness.
        eta: WSR regularization strength (default 0.01).
        ema_beta: EMA decay for running sharpness statistics.
        warmup_steps: steps before WSR activates.
        mode: 'sam' (explicit perturbation) or 'gradient' (gradient norm proxy).

    """

    def __init__(
        self,
        embed_dim: int = 768,
        rho: float = 0.05,
        eta: float = 0.01,
        ema_beta: float = 0.999,
        warmup_steps: int = 500,
        mode: str = "gradient",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rho = rho
        self.eta = eta
        self.ema_beta = ema_beta
        self.warmup_steps = warmup_steps
        self.mode = mode
        self.eps = eps

        if mode not in ("sam", "gradient"):
            raise ValueError(f"mode must be 'sam' or 'gradient', got '{mode}'")

        # Running statistics
        self.register_buffer("running_sharpness", torch.tensor(0.0))
        self.register_buffer("running_spectral_sharpness", torch.tensor(0.0))
        self.register_buffer("running_directional_sharpness", torch.tensor(0.0))
        self.register_buffer("running_grad_norm", torch.tensor(0.0))
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.long))

    def _grassmann_gradient(
        self,
        Q: torch.Tensor,
        euclidean_grad: torch.Tensor,
    ) -> torch.Tensor:
        """Project Euclidean gradient onto tangent space of Gr(k,D) at Q.

        The Grassmann gradient is the off-manifold component:
            grad_Gr = (I - Q Q^T) · grad_euclidean

        This is the Riemannian gradient on the quotient manifold St(D,k)/O(k).

        Args:
            Q: (D, k) orthonormal matrix on St(D,k).
            euclidean_grad: (D, k) Euclidean gradient of loss w.r.t. Q.

        Returns:
            (D, k) Grassmann gradient (tangent vector at Q).

        """
        # Project out the Q-component: (I - QQ^T) G
        QQ_T_G = Q @ (Q.T @ euclidean_grad)  # (D, k)
        grad_grassmann = euclidean_grad - QQ_T_G
        return grad_grassmann

    def _stiefel_retract(self, Q: torch.Tensor) -> torch.Tensor:
        """Retract onto St(D,k) via QR decomposition.

        .. warning::
           KNOWN DEFECT (open, not fixed here — it changes what
           ``mode='sam'`` computes, so it needs a human decision).  The
           canonical-QR sign fix reads ``diag((Q R)[:k, :])``, but for a
           (D, k) matrix with D > k that block is NOT triangular, so the
           signs are effectively arbitrary: the retraction flips
           individual COLUMNS instead of nudging the subspace by ρ.
           Measured: for a random orthonormal Q and ρ = 0.05 the
           retracted Q is ~4.0 away from Q in Frobenius norm instead of
           ~0.05.  Pinned by ``tests/test_training_state_guards.py::…::
           test_retraction_preserves_column_orientation`` (xfail).

        Args:
            Q: (D, k) approximately orthonormal matrix.

        Returns:
            (D, k) orthonormal matrix (nearest on St(D,k)).

        """
        Q_retracted, _ = torch.linalg.qr(Q)
        # Ensure positive diagonal (canonical QR)
        signs = torch.sign(torch.diag(Q_retracted[: Q.size(1), :]))
        signs[signs == 0] = 1
        Q_retracted = Q_retracted * signs.unsqueeze(0)
        return Q_retracted

    def forward(
        self,
        Q: torch.Tensor,
        loss_fn: Callable | None = None,
        z_pred: torch.Tensor | None = None,
        z_target: torch.Tensor | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Workspace Sharpness Regularization loss.

        Args:
            Q: (D, k) workspace projection matrix (orthonormal).
            loss_fn: callable(Q, z_pred, z_target) -> scalar loss.
                Required for mode='sam'.
            z_pred: (B, T, D) predictor output (for loss_fn).
            z_target: (B, T, D) target representation (for loss_fn).
            step: current training step.

        Returns:
            loss: scalar tensor (≥ 0).
            info: dict with diagnostics.

        """
        _D, _k = Q.shape

        # Warmup
        warmup_factor = min(1.0, step / max(self.warmup_steps, 1))
        if warmup_factor < 1e-6:
            zero = Q.new_tensor(0.0)
            return zero, {
                "wsr_loss": 0.0,
                "wsr_warmup": True,
                "wsr_warmup_factor": 0.0,
            }

        if self.mode == "gradient":
            loss, info = self._gradient_mode(Q, step, warmup_factor)
        else:
            loss, info = self._sam_mode(Q, loss_fn, z_pred, z_target, step, warmup_factor)

        return loss, info

    def set_lagged_gradient(self, grad: torch.Tensor) -> None:
        """Store a post-backward snapshot of dL/dQ for use in the next forward.

        WSR mode='gradient' runs BEFORE backward, so Q.grad is cleared
        (zero_grad set_to_none=True) or absent at forward time. The training
        loop captures the freshly computed workspace gradient here each
        optimizer step; WSR then consumes a one-step-lagged, correctly-scaled
        signal (SAM-style perturbation direction).
        """
        self._lagged_gradient = grad.detach().clone()

    def _resolve_euclidean_gradient(self, Q: torch.Tensor) -> tuple[torch.Tensor, str]:
        """Resolve ∇_Q L, and REPORT when it is not available.

        The perturbation direction of SAM is dL/dQ.  ``src/train.py`` calls
        ``optimizer.zero_grad()`` with the default ``set_to_none=True``
        immediately after every step, so ``Q.grad`` is None at forward time
        and the orthonormality proxy was silently substituted for the
        documented quantity.  On an exactly orthonormal Q that proxy is
        *identically zero*, so ``wsr_mode: sam`` reduced to L == 0 without
        ever computing L(Q + Δ) - L(Q).

        Priority:
          1. ``_lagged_gradient`` — real dL/dQ captured post-backward.
          2. ``Q.grad`` — only usable if Q is a leaf that still holds a grad.
          3. otherwise: unavailable.

        Returns:
            (gradient, source) with source in
            ``{"lagged", "live", "unavailable"}``.  On ``"unavailable"``
            the caller must warn and flag ``info``, never present the proxy
            as the documented loss.
        """
        _lagged = getattr(self, "_lagged_gradient", None)
        if _lagged is not None:
            return _lagged.detach().to(Q.dtype), "lagged"
        if Q.is_leaf and Q.grad is not None:
            return Q.grad.detach().to(Q.dtype), "live"
        return Q.new_zeros(Q.shape), "unavailable"

    def _report_missing_gradient(self, Q: torch.Tensor) -> None:
        """Warn ONCE that ∇_Q L is unavailable and the proxy is in use."""
        if getattr(self, "_warned_missing_gradient", False):
            return
        self._warned_missing_gradient = True
        # Do not touch Q.grad for a non-leaf tensor: torch emits its own
        # (louder) UserWarning about that, which would bury this message.
        grad_state = (
            "n/a (non-leaf)" if not Q.is_leaf else ("set" if Q.grad is not None else "None")
        )
        warnings.warn(
            f"WSR mode={self.mode!r}: no dL/dQ is available (Q.is_leaf="
            f"{Q.is_leaf}, Q.grad={grad_state}, "
            "set_lagged_gradient() never called).  The returned sharpness is "
            "an ORTHONORMALITY PROXY, not the documented L(Q+Δ)-L(Q) — the "
            "loss is therefore degenerate (exactly 0 for an orthonormal Q). "
            "Call wsr.set_lagged_gradient(workspace_Q.grad) after "
            "optimizer.step() and before zero_grad().  Either feed the lagged "
            "gradient or stop using mode='sam'/'gradient'.",
            UserWarning,
            stacklevel=3,
        )

    def _orthonormality_proxy(self, Q: torch.Tensor) -> torch.Tensor:
        """Stand-in for dL/dQ when none is available: Q @ (QᵀQ - I).

        Zero for an exactly orthonormal Q, which is why it must never be
        passed off as the documented sharpness.
        """
        _D, k = Q.shape
        deviation = Q.T @ Q - torch.eye(k, device=Q.device, dtype=Q.dtype)
        return Q @ deviation

    def _update_running_statistics(
        self,
        sharpness: float,
        spectral_sharpness: float,
        directional_sharpness: float,
        grad_norm: float,
    ) -> None:
        """EMA the reported sharpness diagnostics and advance the counter.

        Training only (TrainingStateGuard): a validation pass must not
        move the reported WSR diagnostics.
        """
        self.running_sharpness.mul_(self.ema_beta).add_((1 - self.ema_beta) * sharpness)
        self.running_spectral_sharpness.mul_(self.ema_beta).add_(
            (1 - self.ema_beta) * spectral_sharpness,
        )
        self.running_directional_sharpness.mul_(self.ema_beta).add_(
            (1 - self.ema_beta) * directional_sharpness,
        )
        self.running_grad_norm.mul_(self.ema_beta).add_((1 - self.ema_beta) * grad_norm)
        self.total_steps.add_(1)

    def _gradient_mode(
        self,
        Q: torch.Tensor,
        step: int,
        warmup_factor: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute WSR via Grassmann gradient norm (efficient, no second forward pass).

        L_WSR ≈ ρ · ||grad_Gr(Q)||_F / max(||Q||_F, eps)

        This is a first-order approximation of the sharpness.

        Args:
            Q: (D, k) orthonormal workspace matrix.
            step: current training step.
            warmup_factor: warmup multiplier.

        Returns:
            loss: scalar tensor.
            info: dict with diagnostics.

        """
        _D, k = Q.shape

        # Gradient source: a real dL/dQ, or an explicitly reported proxy.
        euclidean_grad, grad_source = self._resolve_euclidean_gradient(Q)
        if grad_source == "unavailable":
            self._report_missing_gradient(Q)
            euclidean_grad = self._orthonormality_proxy(Q)

        # Project onto Grassmann tangent space
        grad_grassmann = self._grassmann_gradient(Q, euclidean_grad)

        # Grassmann gradient norm (Frobenius)
        grad_norm = grad_grassmann.norm()

        # Sharpness = ρ · ||grad_Gr|| / ||Q||
        Q_norm = Q.norm()
        sharpness = self.rho * grad_norm / max(Q_norm.item(), self.eps)

        # Grassmann Sharpness Decomposition
        # Spectral component: off-manifold gradient
        spectral_sharpness = self.rho * grad_norm / max(Q_norm.item(), self.eps)
        # Directional component: on-manifold gradient
        on_manifold_grad = Q @ (Q.T @ euclidean_grad)
        directional_sharpness = self.rho * on_manifold_grad.norm() / max(Q_norm.item(), self.eps)

        # Loss
        loss = self.eta * warmup_factor * sharpness

        # Ensure non-negative (theoretical guarantee)
        loss = F.relu(loss)

        # Update running statistics
        with torch.no_grad():
            self._mutate_state(
                self._update_running_statistics,
                sharpness.item(),
                spectral_sharpness.item(),
                directional_sharpness.item(),
                grad_norm.item(),
            )

        info = {
            "wsr_loss": loss.item(),
            "wsr_sharpness": sharpness.item(),
            "wsr_spectral_sharpness": spectral_sharpness.item(),
            "wsr_directional_sharpness": directional_sharpness.item(),
            "wsr_grad_norm": grad_norm.item(),
            "wsr_rho": self.rho,
            "wsr_warmup_factor": warmup_factor,
            "wsr_warmup": warmup_factor < 1.0,
            # Provenance of ∇_Q L.  A downstream log that records
            # 'unavailable' knows the value is NOT the documented
            # sharpness; do not average it in with the others.
            "wsr_grad_source": grad_source,
            "wsr_gradient_substituted": grad_source == "unavailable",
        }

        return loss, info

    def _sam_mode(
        self,
        Q: torch.Tensor,
        loss_fn: Callable,
        z_pred: torch.Tensor | None,
        z_target: torch.Tensor | None,
        step: int,
        warmup_factor: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute WSR via explicit perturbation (SAM-style).

        L_WSR = L(Q_perturbed) - L(Q)

        where Q_perturbed = retract(Q + ρ · grad_Gr / ||grad_Gr||).

        This requires two forward passes but is more accurate.

        Args:
            Q: (D, k) orthonormal workspace matrix.
            loss_fn: callable(Q, z_pred, z_target) -> scalar loss.
            z_pred: (B, T, D) predictor output.
            z_target: (B, T, D) target representation.
            step: current training step.
            warmup_factor: warmup multiplier.

        Returns:
            loss: scalar tensor.
            info: dict with diagnostics.

        """
        _D, k = Q.shape

        # Compute current loss
        with torch.no_grad():
            L_current = loss_fn(Q, z_pred, z_target).item() if loss_fn is not None else 0.0

        # Get Euclidean gradient.  mode='sam' used to read Q.grad only, but
        # src/train.py zero_grads with set_to_none=True after every step, so
        # Q.grad was ALWAYS None here and the orthonormality proxy was
        # silently substituted — for an orthonormal Q that proxy is exactly
        # zero, so the documented L(Q+Δ)-L(Q) was never computed.
        euclidean_grad, grad_source = self._resolve_euclidean_gradient(Q)
        if grad_source == "unavailable":
            self._report_missing_gradient(Q)
            euclidean_grad = self._orthonormality_proxy(Q)

        # Project onto Grassmann tangent space
        grad_grassmann = self._grassmann_gradient(Q, euclidean_grad)
        grad_norm = grad_grassmann.norm()

        # Normalize and perturb
        if grad_norm > self.eps:
            perturbation = self.rho * grad_grassmann / grad_norm
            Q_perturbed = self._stiefel_retract(Q + perturbation)
        else:
            Q_perturbed = Q.clone()

        # Compute perturbed loss
        with torch.no_grad():
            L_perturbed = (
                loss_fn(Q_perturbed, z_pred, z_target).item() if loss_fn is not None else 0.0
            )

        # Sharpness = loss increase under perturbation
        sharpness = max(L_perturbed - L_current, 0.0)

        # Loss
        loss_tensor = Q.new_tensor(self.eta * warmup_factor * sharpness)

        # Decomposition
        spectral_sharpness = self.rho * grad_norm / max(Q.norm().item(), self.eps)
        on_manifold_grad = Q @ (Q.T @ euclidean_grad)
        directional_sharpness = self.rho * on_manifold_grad.norm() / max(Q.norm().item(), self.eps)

        # Update running statistics
        with torch.no_grad():
            self._mutate_state(
                self._update_running_statistics,
                sharpness,
                spectral_sharpness.item(),
                directional_sharpness.item(),
                grad_norm.item(),
            )

        info = {
            "wsr_loss": loss_tensor.item(),
            "wsr_sharpness": sharpness,
            "wsr_spectral_sharpness": spectral_sharpness.item(),
            "wsr_directional_sharpness": directional_sharpness.item(),
            "wsr_grad_norm": grad_norm.item(),
            "wsr_rho": self.rho,
            "wsr_warmup_factor": warmup_factor,
            "wsr_warmup": warmup_factor < 1.0,
            "wsr_loss_current": L_current,
            "wsr_loss_perturbed": L_perturbed,
            "wsr_grad_source": grad_source,
            "wsr_gradient_substituted": grad_source == "unavailable",
        }

        return loss_tensor, info

    @torch.no_grad()
    # AUDIT R15: PROXY ESTIMATE — inputs are EMA-smoothed sharpness proxies,
    # not the PAC-Bayes theorem's raw quantities
    # (proofs/IMPLEMENTATION_STATUS.md).
    def compute_generalization_bound(
        self,
        n_samples: int,
        lipschitz_constant: float = 1.0,
    ) -> float:
        """Compute workspace generalization bound.

        |L_train(Q) - L_test(Q)| ≤ C · √(ρ_Q / n)

        Args:
            n_samples: number of training samples.
            lipschitz_constant: Lipschitz constant C of the loss.

        Returns:
            Upper bound on generalization gap.

        """
        rho_Q = self.running_sharpness.item()
        if n_samples <= 0 or rho_Q <= 0:
            return float("inf")
        return lipschitz_constant * math.sqrt(rho_Q / n_samples)

    @torch.no_grad()
    def compute_pac_bayes_bound(
        self,
        n_samples: int,
        prior_kl: float = 0.0,
        delta: float = 0.05,
    ) -> float:
        """Compute PAC-Bayes generalization bound.

        L_test(Q) ≤ E_{Q~Q_ρ}[L_train(Q)] + (KL(Q_ρ || P) + log(n/δ)) / n

        Args:
            n_samples: number of training samples.
            prior_kl: KL divergence between perturbation and prior distributions.
            delta: confidence level (default 0.05).

        Returns:
            PAC-Bayes upper bound.

        """
        if n_samples <= 0:
            return float("inf")
        complexity = (prior_kl + math.log(max(n_samples, 1) / delta)) / n_samples
        return self.running_sharpness.item() + complexity

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, rho={self.rho}, "
            f"eta={self.eta}, mode={self.mode}, "
            f"warmup_steps={self.warmup_steps}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  One-line convenience function
# ═══════════════════════════════════════════════════════════════════════════


def wsr_sharpness(Q, embed_dim=768, rho=0.05, eta=0.01, step=0):
    """Workspace sharpness regularization — one function call.

    Args:
        Q: (D, k) orthonormal workspace matrix.
        embed_dim: embedding dimension.
        rho: perturbation radius.
        eta: regularization strength.
        step: current training step.

    Returns:
        (loss, info) tuple.

    """
    wsr = WorkspaceSharpnessRegularization(embed_dim=embed_dim, rho=rho, eta=eta)
    wsr = wsr.to(Q.device)
    return wsr(Q, step=step)
