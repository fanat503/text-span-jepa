# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Canonical CKA implementations shared across the repo.

Single source of truth for Linear and RBF-kernel Centered Kernel Alignment
(Kornblith et al., ICML 2019). Previously duplicated as private statics on
CollapseDiagnostics and reached into by six interpreter modules — both stacks
had begun to drift (see audit reports v3/v10).
"""

import math

import torch


def _hsic(x, y):
    """Hilbert-Schmidt Independence Criterion (unbiased estimator)."""
    N = x.size(0)
    if N <= 3:
        return torch.tensor(0.0, device=x.device)
    K = x @ x.T
    L = y @ y.T
    H = torch.eye(N, device=x.device) - 1.0 / N
    KH = K @ H
    LH = L @ H
    return torch.trace(KH @ LH) / ((N - 1) ** 2)


def _rbf_kernel(x, sigma):
    """RBF (Gaussian) kernel matrix."""
    dists = torch.cdist(x, x, p=2)
    return torch.exp(-0.5 * dists**2 / (sigma**2))


def linear_cka(x, y):
    """Linear CKA between two representation matrices.

    Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019. Measures similarity of representation geometry
    independent of orthogonal transformations.
    Returns value in [0, 1]; 1 = identical geometry.
    """
    try:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        hsic_xy = _hsic(x, y)
        hsic_xx = _hsic(x, x)
        hsic_yy = _hsic(y, y)
        denom = (hsic_xx * hsic_yy).sqrt() + 1e-10
        val = (hsic_xy / denom).item()
        if not math.isfinite(val):
            return 0.0
        return max(min(val, 1.0), 0.0)
    except Exception:
        return 0.0


def rbf_cka(x, y, sigma=None):
    """RBF-kernel CKA between two representations.

    Captures nonlinear similarity. More sensitive than linear CKA for
    detecting representation differences.
    Returns value in [0, 1]; 1 = identical geometry.
    """
    try:
        N = x.size(0)
        if N <= 3:
            return 0.0
        if sigma is None:
            # Median heuristic for bandwidth
            dists = torch.pdist(x)
            if dists.numel() > 0:
                sigma = dists.median().item()
            else:
                sigma = 1.0
            sigma = max(sigma, 1e-8)

        K = _rbf_kernel(x, sigma)
        L = _rbf_kernel(y, sigma)

        H = torch.eye(N, device=x.device) - 1.0 / N
        KH = K @ H
        LH = L @ H

        hsic_kl = torch.trace(KH @ LH) / ((N - 1) ** 2)
        hsic_kk = torch.trace(KH @ KH) / ((N - 1) ** 2)
        hsic_ll = torch.trace(LH @ LH) / ((N - 1) ** 2)

        denom = (hsic_kk * hsic_ll).sqrt() + 1e-10
        val = (hsic_kl / denom).item()
        if not math.isfinite(val):
            return 0.0
        return max(min(val, 1.0), 0.0)
    except Exception:
        return 0.0
