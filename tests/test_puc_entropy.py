
"""PUC's reported entropy must be the entropy of the eigenvalues it used.

`puc_entropy` is printed in nats and compared against `target_entropy`, so a value
that is not the entropy of anything is worse than no value. It previously passed
the eigenvalues through `softplus`, which is a monotone squash - the number still
moved, it just did not mean H. This pins the arithmetic against an independent
computation of the module's own documented formula.
"""

from __future__ import annotations

import math

import torch

from src.models.puc import PredictionUncertaintyCalibration

LOG_2PIE = math.log(2 * math.pi * math.e)


def _expected(z_flat, n_components):
    """The module's documented formula, computed independently."""
    D = z_flat.shape[1]
    cen = z_flat - z_flat.mean(dim=0, keepdim=True)
    cov = cen.T @ cen / max(z_flat.size(0) - 1, 1)
    lam = torch.linalg.eigvalsh(cov).flip(0)[:n_components].clamp_min(1e-6)
    tracked = 0.5 * (lam.log().sum().item() + n_components * LOG_2PIE)
    untracked = 0.5 * (D - n_components) * (LOG_2PIE + math.log(lam.mean().item() + 1e-8))
    return tracked + untracked


class TestEntropyIsTheEntropyOfTheEigenvalues:
    def test_reported_entropy_matches_its_own_formula(self):
        p = PredictionUncertaintyCalibration(embed_dim=24, n_components=4)
        torch.manual_seed(0)
        z = torch.randn(4, 16, 24)
        _loss, info = p(z, step=1)
        z_flat = z.reshape(-1, 24)
        expected = _expected(z_flat, p.n_components)
        assert abs(info["puc_entropy"] - expected) < 1e-6, (
            f"puc_entropy={info['puc_entropy']:.6f} but H(the eigenvalues)={expected:.6f}"
        )

    def test_no_softplus_is_silently_reintroduced(self):
        """The regression this file exists for.

        A softplus is monotone, so the value still moves and a naive test on
        range or sign still passes. Pin the actual transform.
        """
        import inspect

        import src.models.puc as mod

        src_text = inspect.getsource(mod)
        for bad in ("softplus(eigs_top", "softplus(self.running_eigenvalues"):
            assert bad not in src_text, (
                f"{bad!r} is back: the eigenvalues are being squashed before the "
                "entropy is computed, which makes puc_entropy not an entropy"
            )

    def test_entropy_is_finite_for_a_well_conditioned_batch(self):
        p = PredictionUncertaintyCalibration(embed_dim=16, n_components=4)
        torch.manual_seed(3)
        z = torch.randn(8, 8, 16) * 3.0
        _loss, info = p(z, step=1)
        assert math.isfinite(info["puc_entropy"]), "entropy is NaN or inf"
        assert info["puc_entropy_deficit"] >= 0.0, "deficit must not be negative"
