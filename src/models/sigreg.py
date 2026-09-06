# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# SIGReg: Sketched Isotropic Gaussian Regularization
# From LeJEPA (Balestriero & LeCun, 2025), arXiv:2511.08544
#
# Theoretical foundation:
#   LeJEPA proves that the isotropic Gaussian N(0,I) is the UNIQUE optimal
#   embedding distribution for minimizing worst-case prediction risk on
#   downstream tasks (both linear and nonlinear probes).
#
#   SIGReg enforces this by:
#   1. Projecting embeddings onto M random 1D directions (Cramer-Wold principle)
#   2. For each projection, matching to standard Gaussian via characteristic function
#   3. Using Epps-Pulley normality test statistic
#
# Key properties:
#   - O(N) time and memory complexity (linear in batch size)
#   - Provably bounded loss, gradients, and curvature (Theorem 4, LeJEPA)
#   - Prevents collapse BY CONSTRUCTION (not heuristic like stop-gradient/EMA)
#   - Single hyperparameter lambda_sigreg (vs 3 for VICReg)
#
# References:
#   Balestriero & LeCun (2025) "LeJEPA" arXiv:2511.08544
#   Epps & Pulley (1983) — normality test statistic
#   Cramer & Wold (1936) — Cramer-Wold device

import torch
import torch.nn.functional as F
from torch import nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA, 2025).

    Projects embeddings onto random 1D directions and matches each
    univariate projection to a standard Gaussian using the Epps-Pulley
    characteristic function test.

    Args:
        embed_dim: dimension of the embedding space
        n_sketches: number of random projection directions (M in paper).
        n_integration_points: number of integration points for
            characteristic function evaluation (L in paper).
        sigma: target standard deviation. Default 1.0.

    """

    def __init__(self, embed_dim=768, n_sketches=64, n_integration_points=17, sigma=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_sketches = n_sketches
        self.n_integration_points = n_integration_points
        self.sigma = sigma

        self.register_buffer(
            "sketch_directions",
            self._generate_sketch_directions(embed_dim, n_sketches),
        )

        t_max = 3.0 / max(sigma, 1e-6)
        t = torch.linspace(0, t_max, n_integration_points)
        self.register_buffer("t_points", t)

    @staticmethod
    def _generate_sketch_directions(dim, n_sketches):
        if n_sketches >= dim:
            directions = torch.randn(n_sketches, dim)
        else:
            random_matrix = torch.randn(dim, n_sketches)
            Q, _ = torch.linalg.qr(random_matrix)
            directions = Q.T

        norms = directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        directions = directions / norms
        return directions

    def forward(self, embeddings):
        if embeddings.dim() == 3:
            B, T, D = embeddings.shape
            flat = embeddings.reshape(B * T, D)
        else:
            flat = embeddings
        N, D = flat.shape

        if N <= 1 or D < 2:
            return torch.tensor(0.0, device=flat.device, requires_grad=True)

        flat = flat - flat.mean(dim=0, keepdim=True)
        projections = flat @ self.sketch_directions.T  # (N, M)
        loss = self._epps_pulley_loss(projections)
        return loss

    def _epps_pulley_loss(self, projections):
        _N, _M = projections.shape
        L = self.n_integration_points
        sigma = self.sigma

        t = self.t_points
        phi_gauss = torch.exp(-(sigma**2) * t**2 / 2.0)

        total_loss = torch.tensor(0.0, device=projections.device)

        chunk_size = max(1, min(L, 8))
        for t_start in range(0, L, chunk_size):
            t_end = min(t_start + chunk_size, L)
            t_chunk = t[t_start:t_end]

            angles = t_chunk[:, None, None] * projections[None, :, :]
            cos_mean = torch.cos(angles).mean(dim=1)
            sin_mean = torch.sin(angles).mean(dim=1)
            phi_emp = (cos_mean**2 + sin_mean**2).sqrt()

            phi_target = phi_gauss[t_start:t_end, None]
            diff_sq = (phi_emp - phi_target) ** 2
            chunk_loss = diff_sq.mean()

            total_loss = total_loss + chunk_loss * (t_end - t_start)

        total_loss = total_loss / L
        return total_loss

    def extra_repr(self):
        return (
            f"embed_dim={self.embed_dim}, n_sketches={self.n_sketches}, "
            f"n_integration_points={self.n_integration_points}, "
            f"sigma={self.sigma}"
        )


class WeakSIGReg(nn.Module):
    """Weak-SIGReg: covariance-only variant of SIGReg.

    Matches the variance of each 1D projection to sigma^2.
    """

    def __init__(self, embed_dim=768, n_sketches=64, sigma=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_sketches = n_sketches
        self.sigma = sigma
        self.register_buffer(
            "sketch_directions",
            SIGReg._generate_sketch_directions(embed_dim, n_sketches),
        )

    def forward(self, embeddings):
        if embeddings.dim() == 3:
            flat = embeddings.reshape(-1, embeddings.size(-1))
        else:
            flat = embeddings
        N, _D = flat.shape

        if N <= 1:
            return torch.tensor(0.0, device=flat.device, requires_grad=True)

        flat = flat - flat.mean(dim=0, keepdim=True)
        projections = flat @ self.sketch_directions.T
        var_per_sketch = projections.var(dim=0)
        loss = ((var_per_sketch - self.sigma**2) ** 2).mean()
        return loss


class VISReg(nn.Module):
    """VISReg: Variance-Invariance-Sketching Regularization (2026).

    Combines VICReg's flexibility with SIGReg's distributional shape enforcement.
    """

    def __init__(
        self,
        embed_dim=768,
        variance_margin=1.0,
        n_sketches=64,
        n_integration_points=17,
        sigma=1.0,
        lambda_variance=1.0,
        lambda_covariance=0.0,
        lambda_sigreg=1.0,
    ):
        super().__init__()
        self.variance_margin = variance_margin
        self.lambda_variance = lambda_variance
        self.lambda_covariance = lambda_covariance
        self.lambda_sigreg = lambda_sigreg

        self.variance_reg = VarianceRegularization(margin=variance_margin)
        self.covariance_reg = CovarianceRegularization()
        self.sigreg = SIGReg(
            embed_dim=embed_dim,
            n_sketches=n_sketches,
            n_integration_points=n_integration_points,
            sigma=sigma,
        )

    def forward(self, embeddings_online, embeddings_target=None):
        loss_variance = self.variance_reg(embeddings_online)
        loss_covariance = self.covariance_reg(embeddings_online)
        loss_sigreg = self.sigreg(embeddings_online)

        loss_invariance = torch.tensor(0.0, device=embeddings_online.device)
        if embeddings_target is not None:
            loss_invariance = F.mse_loss(
                embeddings_online.mean(dim=(0, 1)),
                embeddings_target.detach().mean(dim=(0, 1)),
            )

        total_loss = (
            self.lambda_variance * loss_variance
            + self.lambda_covariance * loss_covariance
            + self.lambda_sigreg * loss_sigreg
            + loss_invariance
        )

        loss_dict = {
            "loss_variance": loss_variance.item(),
            "loss_covariance": loss_covariance.item(),
            "loss_sigreg": loss_sigreg.item(),
            "loss_invariance": loss_invariance.item(),
        }

        return total_loss, loss_dict


from .collapse import CovarianceRegularization, VarianceRegularization
