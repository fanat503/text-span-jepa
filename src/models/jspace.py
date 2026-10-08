# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# J-Space Metrics: Workspace formation grounded in Anthropic's Jacobian Lens
#
# Reference: Gurnee et al. (2026) arXiv:2607.15495
# Key findings: LLMs have a privileged subspace (J-space) where
# verbalizable representations reside — ~10% of activation variance,
# ~25 simultaneous concepts, 5 Global Workspace Theory properties.

import math

import torch
from torch import nn


class JSpaceMetrics(nn.Module):
    """J-Space workspace metrics grounded in Anthropic's Jacobian Lens paper."""

    def __init__(self, variance_threshold=0.10, k_workspace=25):
        super().__init__()
        self.variance_threshold = variance_threshold
        self.k_workspace = k_workspace
        self._prev_jspace_vectors = None

    @torch.no_grad()
    def compute(self, online_h, target_h, predictor_h=None, prev_online_h=None):
        metrics = {}
        try:
            flat_online = online_h.reshape(-1, online_h.size(-1)).float()
            flat_target = target_h.reshape(-1, target_h.size(-1)).float()
            N, D = flat_online.shape
            if N <= 1 or D < 4:
                return self._zero_metrics()

            centered = flat_online - flat_online.mean(dim=0)
            cov = (centered.T @ centered) / max(N - 1, 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.flip(0)
            eigenvectors = eigenvectors.flip(1)
            eigenvalues = eigenvalues.clamp(min=0.0)
            total_var = eigenvalues.sum()
            if total_var < 1e-10:
                return self._zero_metrics()

            cumvar = torch.cumsum(eigenvalues, dim=0) / total_var
            jspace_dim = (cumvar < self.variance_threshold).sum().item() + 1
            jspace_dim = min(jspace_dim, D)
            metrics["jspace_dim_fraction"] = jspace_dim / D

            jspace_eigs = eigenvalues[:jspace_dim]
            if jspace_eigs.sum() > 1e-10:
                jspace_probs = jspace_eigs / jspace_eigs.sum()
                entropy = -(jspace_probs * (jspace_probs + 1e-10).log()).sum()
                capacity = entropy.exp().item()
            else:
                capacity = 1.0
            metrics["jspace_capacity"] = min(capacity, float(D))

            jspace_var = eigenvalues[:jspace_dim].sum()
            non_jspace_var = eigenvalues[jspace_dim:].sum()
            n_non_jspace = max(D - jspace_dim, 1)
            n_jspace = max(jspace_dim, 1)
            concentration = (jspace_var / n_jspace) / (non_jspace_var / n_non_jspace + 1e-10)
            metrics["jspace_concentration"] = min(concentration.item(), 100.0) / 100.0

            if jspace_dim < D and jspace_dim > 0:
                gap = eigenvalues[jspace_dim - 1] - eigenvalues[jspace_dim]
                normalized_gap = gap / (eigenvalues[0] + 1e-10)
                metrics["jspace_spectral_gap"] = max(normalized_gap.item(), 0.0)
            else:
                metrics["jspace_spectral_gap"] = 0.0

            V_jspace = eigenvectors[:, :jspace_dim]
            V_bg = eigenvectors[:, jspace_dim:]
            cross = V_jspace.T @ V_bg
            ortho_score = 1.0 - cross.abs().mean().item()
            metrics["jspace_orthogonality"] = max(ortho_score, 0.0)

            if predictor_h is not None:
                flat_pred = predictor_h.reshape(-1, predictor_h.size(-1)).float()
                proj = flat_pred @ V_jspace @ V_jspace.T
                pred_norm = flat_pred.norm()
                if pred_norm > 1e-10:
                    broadcast = proj.norm() / pred_norm
                    metrics["jspace_broadcast_score"] = min(broadcast.item(), 1.0)
                else:
                    metrics["jspace_broadcast_score"] = 0.0
            else:
                metrics["jspace_broadcast_score"] = 0.0

            if (
                self._prev_jspace_vectors is not None
                and self._prev_jspace_vectors.shape == V_jspace.shape
            ):
                similarity = V_jspace.T @ self._prev_jspace_vectors
                stability = (similarity**2).mean().sqrt().item()
                metrics["jspace_stability"] = min(max(stability, 0.0), 1.0)
            else:
                metrics["jspace_stability"] = 0.0
            self._prev_jspace_vectors = V_jspace.clone()

            metrics["jspace_quality"] = self._compute_jspace_quality(
                spectral_gap=metrics["jspace_spectral_gap"],
                orthogonality=metrics["jspace_orthogonality"],
                concentration=metrics["jspace_concentration"],
                broadcast=metrics["jspace_broadcast_score"],
                stability=metrics["jspace_stability"],
            )

            target_centered = flat_target - flat_target.mean(dim=0)
            cov_target = (target_centered.T @ target_centered) / max(N - 1, 1)
            eigvals_t, eigvecs_t = torch.linalg.eigh(cov_target)
            eigvals_t = eigvals_t.flip(0).clamp(min=0.0)
            eigvecs_t = eigvecs_t.flip(1)
            total_var_t = eigvals_t.sum()
            if total_var_t > 1e-10:
                cumvar_t = torch.cumsum(eigvals_t, dim=0) / total_var_t
                jspace_dim_t = (cumvar_t < self.variance_threshold).sum().item() + 1
                V_jspace_t = eigvecs_t[:, : min(jspace_dim_t, jspace_dim)]
                k_common = min(V_jspace.size(1), V_jspace_t.size(1))
                if k_common > 0:
                    sim = V_jspace[:, :k_common].T @ V_jspace_t[:, :k_common]
                    separation = (sim**2).mean().sqrt().item()
                    metrics["jspace_separation"] = min(max(separation, 0.0), 1.0)
                else:
                    metrics["jspace_separation"] = 0.0
            else:
                metrics["jspace_separation"] = 0.0

            for k, v in metrics.items():
                if not math.isfinite(v):
                    metrics[k] = 0.0
        except Exception:
            return self._zero_metrics()
        return metrics

    @staticmethod
    def _compute_jspace_quality(
        spectral_gap,
        orthogonality,
        concentration,
        broadcast,
        stability,
        w_gap=0.25,
        w_ortho=0.2,
        w_conc=0.2,
        w_broadcast=0.15,
        w_stability=0.2,
    ):
        try:
            gap = max(0.0, min(1.0, spectral_gap))
            ortho = max(0.0, min(1.0, orthogonality))
            conc = max(0.0, min(1.0, concentration))
            bcast = max(0.0, min(1.0, broadcast))
            stab = max(0.0, min(1.0, stability))
            quality = (
                w_gap * gap
                + w_ortho * ortho
                + w_conc * conc
                + w_broadcast * bcast
                + w_stability * stab
            )
            geometric_mean = (gap * ortho * conc * bcast * stab) ** 0.2
            bonus = 0.1 * geometric_mean
            quality = quality + bonus
            quality = max(0.0, min(1.0, quality))
            if not math.isfinite(quality):
                return 0.0
            return quality
        except Exception:
            return 0.0

    def _zero_metrics(self):
        return {
            "jspace_dim_fraction": 0.0,
            "jspace_capacity": 0.0,
            "jspace_concentration": 0.0,
            "jspace_spectral_gap": 0.0,
            "jspace_broadcast_score": 0.0,
            "jspace_orthogonality": 0.0,
            "jspace_stability": 0.0,
            "jspace_quality": 0.0,
            "jspace_separation": 0.0,
        }

    def extra_repr(self):
        return f"variance_threshold={self.variance_threshold}, " f"k_workspace={self.k_workspace}"
