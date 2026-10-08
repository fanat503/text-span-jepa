# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Canonical CKA implementations shared across the repo.

Single source of truth for Linear and RBF-kernel Centered Kernel Alignment
(Kornblith et al., ICML 2019). Previously duplicated as private statics on
CollapseDiagnostics and reached into by six interpreter modules — both stacks
had begun to drift (see audit reports v3/v10).

WHAT THIS MODULE GUARANTEES
---------------------------
1. **Unbiased HSIC.** Kornblith et al. 2019, eq. 4. The previous code
   documented "unbiased estimator" but computed the biased V-statistic
   ``trace(KH@LH)/(N-1)**2`` and omitted the diagonal correction. The estimator
   here is now genuinely the unbiased one.

   **Measured consequence: none, for the CKA value.** The diagonal correction
   multiplies the numerator and *both* denominator terms by the same scalar
   ``(N-1)**2 / (N*(N-3))``, which cancels in the ratio. Verified to 4 decimals
   on both kernel types: linear 0.1953 vs 0.1953, RBF 0.1196 vs 0.1196. So the
   "CKA = 0.96 on independent matrices" claim in the original audit report was
   **not** caused by the biased estimator, and switching estimators does not
   fix it. See "KNOWN LIMITATION" below, which is the actual cause. What this
   change does buy is item 2 (the same algebra is O(N^2) instead of O(N^3)),
   item 3, item 4, and docstrings that match the code.
2. **O(N^2 D), not O(N^3).** ``torch.trace(KH @ LH)`` on N x N matrices costs
   O(N^3). The identical quantity is ``sum(Kh * Lh)`` — elementwise, O(N^2).
   Measured on this machine (N=4000, D=8): 3.50 s -> 0.10 s.
3. **Shape mismatch raises.** ``except Exception: return 0.0`` reported a
   sample-count mismatch as *maximum* dissimilarity, which is a maximum lie:
   ``ablation.py`` hits it whenever an ablation yields a different N.
4. **Per-matrix bandwidth.** ``rbf_cka`` used one ``sigma`` (the median
   distance of ``x``) for both kernels, so two inputs on incomparable scales
   produced a number, not a similarity.

KNOWN LIMITATION (read before quoting any CKA number)
----------------------------------------------------
CKA has a large **null floor that depends on N and D**. For two *independent*
N x D Gaussian matrices the expectation is closed form (verified to 3 decimals
against this implementation):

    E[CKA_independent] = D / (N + D + 1)

So "CKA = 0.96" between two 768-dim representations scored on 32 samples is
**not** evidence of shared structure — it is the null. Use
:func:`cka_null_baseline` and report ``cka - null``, or report
``cka / null``, never the raw value. See ``STILL UNSOUND`` in the audit.
"""

from __future__ import annotations

import math

import torch

# N below this makes Kornblith's unbiased estimator undefined (the N-3 factor).
MIN_SAMPLES = 4


def _center(m: torch.Tensor) -> torch.Tensor:
    """Return ``H @ m @ H`` for the centering matrix ``H = I - 11'/N``.

    Uses row/column means instead of materialising ``H`` (which would be an
    extra N x N tensor) and instead of ``m @ H`` (which would be O(N^3)).
    Mathematically identical to ``H @ m @ H``.

    Ordering matters and is load-bearing: the result must have BOTH zero row
    sums and zero column sums, because Kornblith's estimator drops the
    ``1'Kh1`` terms on exactly that basis. Subtracting the column means first
    changes the row means, so the second subtraction has to use the *original*
    row means minus the grand mean — not the row means of the already-shifted
    matrix. Adding the grand mean back at the end (the obvious reading of
    ``m - col - row + grand``) silently leaves every row summing to the grand
    mean, which is invisible for a linear Gram matrix (already double-centered,
    grand mean 0) and badly wrong for a raw RBF kernel.
    """
    grand = m.mean()
    row_means = m.mean(dim=1, keepdim=True) - grand
    m = m - m.mean(dim=0, keepdim=True)
    return m - row_means


def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased Hilbert-Schmidt Independence Criterion.

    Kornblith et al., ICML 2019, eq. 4:

        HSIC_u(K, L) = 1/(N(N-3)) * [ tr(Kh Lh)
                                       + 1'Kh1 * 1'Lh1
                                       - (2/N) * 1'KhLh1 ]

    where ``Kh = H K H``. Every ``1'Kh1``-type term is identically zero for a
    double-centered kernel (``H 1 = 0``), so the whole bracket collapses to
    ``tr(Kh Lh)`` and the estimator is exactly

        sum(Kh * Lh) / (N * (N - 3))

    — which is simultaneously the unbiased estimator *and* the O(N^2) one. The
    previous implementation used the biased V-statistic
    ``trace(KH @ LH)/(N-1)**2`` and paid O(N^3) for it.

    Args:
        K, L: (N, N) kernel matrices for the same N samples.

    Returns:
        Scalar tensor; may be negative (the unbiased estimator is not
        guaranteed non-negative).

    """
    N = K.size(0)
    if N <= 3:
        raise ValueError(f"unbiased HSIC needs N > 3, got N={N}")
    return (_center(K) * _center(L)).sum() / (N * (N - 3.0))


def _validate_pair(x, y, label: str) -> int:
    """Shared shape contract for both CKA variants.

    Raises:
        ValueError: if either input is not 2-D, or if the two inputs disagree
            on the number of samples. Silently coercing this to 0.0 previously
            turned a caller bug ("maximum dissimilarity") into a result.

    """
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise ValueError(f"{label} expects torch.Tensor inputs, got {type(x).__name__}")
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(
            f"{label} expects 2-D (N, D) inputs, got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    if x.size(0) != y.size(0):
        raise ValueError(
            f"{label} requires the same number of samples on both sides, "
            f"got N={x.size(0)} and N={y.size(0)}. CKA is undefined for "
            f"mismatched sample counts — resample or truncate to a common N "
            f"before calling."
        )
    return int(x.size(0))


def _median_bandwidth(x: torch.Tensor) -> float:
    """Median-heuristic bandwidth for one matrix (never shared across x and y)."""
    dists = torch.pdist(x)
    if dists.numel() == 0:
        return 1.0
    return max(float(dists.median().item()), 1e-8)


def _rbf_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF (Gaussian) kernel matrix with the given bandwidth."""
    dists = torch.cdist(x, x, p=2)
    return torch.exp(-0.5 * dists**2 / (sigma**2))


def _normalize_cka(K: torch.Tensor, L: torch.Tensor) -> float:
    """Unbiased-HSIC CKA of two kernel matrices, clamped to [0, 1].

    The unbiased estimator's expectation is 0 for independent kernels and it
    can come out slightly negative; the clamp keeps the [0, 1] contract that
    every caller in this repo plots against. Values below 0 are therefore
    reported as 0 — read :func:`cka_null_baseline` before concluding that a
    low CKA means "dissimilar".

    A **zero-variance** input has no representation geometry at all, so the
    ratio is 0/0. That returns 0.0 rather than raising, because
    ``CollapseDiagnostics.compute`` calls into here on whatever it is handed
    and must stay total on degenerate input (``tests/test_interp.py::
    TestCollapseNewMetrics::test_no_nan_new_metrics`` feeds it all-zeros). This
    is a *different* failure mode from a shape mismatch, which raises: here the
    geometry is undefined, not the comparison. Detecting degenerate
    representations as such is :mod:`src.interp.interpretability_index`'s job,
    and it refuses to quote a headline for them.
    """
    denom = (_hsic(K, K) * _hsic(L, L)).sqrt()
    if not torch.isfinite(denom) or denom.abs() < 1e-12:
        return 0.0
    val = (_hsic(K, L) / denom).item()
    if not math.isfinite(val):
        return 0.0
    return max(min(val, 1.0), 0.0)


def cka_null_baseline(n_samples: int, n_features: int) -> float:
    """Expected CKA between two **independent** N x D Gaussian matrices.

    Closed form: ``D / (N + D + 1)``.

    This is the value CKA reports when the two representations share *nothing*
    at all. It is the single most important number for interpreting a CKA
    result in this repo, and it is large whenever D is comparable to or larger
    than N — e.g. 0.96 at D=768, N=32.

    Args:
        n_samples: N, the number of samples scored.
        n_features: D, the representation dimension of the *taller* input.

    Returns:
        float in (0, 1).

    """
    n_samples = int(n_samples)
    n_features = int(n_features)
    if n_samples <= 0 or n_features <= 0:
        raise ValueError(
            f"n_samples and n_features must be positive, got {n_samples}, {n_features}"
        )
    return n_features / (n_samples + n_features + 1.0)


def linear_cka(x, y):
    """Linear CKA between two representation matrices.

    Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019. Measures similarity of representation geometry
    independent of orthogonal transformations.

    Uses the **unbiased** HSIC of Kornblith et al. 2019 eq. 4 (see :func:`_hsic`).
    For a linear kernel the Gram matrix is already double-centered, so
    ``Kh = K`` and the estimator differs from the biased V-statistic only by the
    scalar ``(N-1)**2 / (N*(N-3))``, which **cancels in this ratio**. The
    reported value is therefore unchanged by the de-biasing; what changed is
    the cost (O(N^3) -> O(N^2)) and the failure behaviour (raises instead of
    returning 0.0).

    IMPORTANT: compare against :func:`cka_null_baseline` before reading this
    as a similarity. Independent N x D matrices already score
    ``D / (N + D + 1)``.

    Args:
        x, y: 2-D (N, D_x) and (N, D_y) tensors with the same N. D may differ.

    Returns:
        float in [0, 1]; 1 = identical geometry.

    Raises:
        ValueError: on shape mismatch, on N <= 3, or on a degenerate (constant)
            input. Never returns 0.0 to signal a problem.

    """
    N = _validate_pair(x, y, "linear_cka")
    if N < MIN_SAMPLES:
        raise ValueError(
            f"linear_cka needs at least {MIN_SAMPLES} samples for the unbiased "
            f"estimator, got N={N}"
        )
    x = x.double() - x.double().mean(dim=0, keepdim=True)
    y = y.double() - y.double().mean(dim=0, keepdim=True)
    return _normalize_cka(x @ x.T, y @ y.T)


def rbf_cka(x, y, sigma=None):
    """RBF-kernel CKA between two representations.

    Captures nonlinear similarity. More sensitive than linear CKA for
    detecting representation differences. Uses the **unbiased** HSIC
    (Kornblith et al. 2019 eq. 4). As with the linear variant, the diagonal
    correction is a scalar that cancels in the CKA ratio, so it does not change
    the reported value; the real differences are the bandwidth, the shape
    contract, and the cost.

    Bandwidth: by default K and L each get their **own** median-heuristic
    bandwidth, computed from their own pairwise distances. This makes the
    result invariant to a common global rescaling of both inputs, and stops it
    from being dominated by whichever input happens to be on the larger scale.
    The previous version used the median distance of ``x`` for both kernels, so
    ``rbf_cka(x*0.01, y*100)`` returned 0.33 — a number driven entirely by the
    scale mismatch, not by the representations.

    IMPORTANT: compare against :func:`cka_null_baseline` before reading this as
    a similarity.

    Args:
        x, y: 2-D (N, D_x) and (N, D_y) tensors with the same N.
        sigma: optional explicit bandwidth for BOTH kernels. Only meaningful
            when x and y are known to be on the same scale; leave as None (the
            default) to get the scale-robust per-matrix bandwidth.

    Returns:
        float in [0, 1]; 1 = identical geometry.

    Raises:
        ValueError: on shape mismatch, on N <= 3, or on a degenerate input.

    """
    N = _validate_pair(x, y, "rbf_cka")
    if N < MIN_SAMPLES:
        raise ValueError(
            f"rbf_cka needs at least {MIN_SAMPLES} samples for the unbiased "
            f"estimator, got N={N}"
        )
    x = x.double()
    y = y.double()
    if sigma is None:
        sigma_x = _median_bandwidth(x)
        sigma_y = _median_bandwidth(y)
    else:
        sigma_x = sigma_y = max(float(sigma), 1e-8)
    return _normalize_cka(_rbf_kernel(x, sigma_x), _rbf_kernel(y, sigma_y))
