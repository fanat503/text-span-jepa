# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# CollapseDiagnostics: the two properties that make its cost safe to reduce.
#
# 1.  The 19 decompositions per step are now 5, and the deduplication is
#     bit-identical. These tests pin the *count* (a budget) and the *values*
#     (an equivalence), because a dedup that quietly changes a number is worse
#     than the 13 extra SVDs it removed.
#
# 2.  The four subsampling metrics no longer draw from the training device's
#     global RNG. Before this, `torch.randperm(..., device=flat.device)`
#     consumed the same generator DropPath does, so merely *running* a
#     diagnostic steered the model -- and gating the diagnostics behind
#     `log_freq` would have silently moved the training trajectory while
#     looking like a pure optimisation. These tests pin the RNG contract,
#     which is the precondition for that gate.
#
# The 1-2 game: no skips, no xfails. Every assertion here is a property that
# must hold, not a constant that happens to hold today.

import math

import pytest
import torch

from src.models import collapse as collapse_mod
from src.models.collapse import (
    _SUBSAMPLE_LIMIT,
    CollapseDiagnostics,
    _derive_subsample_rng,
    _next_subsample_rng,
)
from src.utils.seed import seed_everything

# Rows chosen so that (a) the matrices are well conditioned, (b) N > the
# 256-row subsample cap so the RNG path is genuinely exercised. Tensors stay
# tiny: this file must fit inside a budgeted run on a shared 6-core box.
B, T, D = 4, 96, 24  # N = 384 rows > _SUBSAMPLE_LIMIT
SMALL_B, SMALL_T, SMALL_D = 2, 16, 12  # N = 32 rows <= cap, nothing subsamples


def _pair(seed=0, b=B, t=T, d=D):
    seed_everything(seed)
    return torch.randn(b, t, d), torch.randn(b, t, d)


# ═══════════════════════════════════════════════════════════════════
#  1. The decomposition budget
# ═══════════════════════════════════════════════════════════════════


class TestDecompositionBudget:
    """A per-step diagnostic that decomposes the activation matrix 19 times is
    the defect this card is about. Counting the calls is the regression guard;
    asserting on a time would be flaky on a shared box."""

    @staticmethod
    def _count(names):
        counts = {}

        class _Counter:
            def __enter__(self):
                self.saved = {}
                for n in names:
                    orig = getattr(torch.linalg, n)
                    self.saved[n] = orig

                    def make(nm, o):
                        def counted(*a, **kw):
                            counts[nm] = counts.get(nm, 0) + 1
                            return o(*a, **kw)

                        return counted

                    setattr(torch.linalg, n, make(n, orig))
                return self

            def __exit__(self, *exc):
                for n, orig in self.saved.items():
                    setattr(torch.linalg, n, orig)
                return False

        return _Counter(), counts

    def test_one_svd_per_distinct_matrix(self):
        """Every metric of one `compute` must share decompositions.

        Two distinct activation matrices are handed in, so at most two
        `svdvals` and two full `svd` calls can be legitimate. `svdvals` also
        fires once on the small (k, k) CCA matrix inside `_svcca`, hence 3.
        """
        online, target = _pair()
        counter, counts = self._count(("svdvals", "svd", "matrix_rank"))
        with counter:
            CollapseDiagnostics().compute(online, target)

        assert counts.get("svdvals", 0) <= 3, (
            f"{counts.get('svdvals')} svdvals calls for two activation matrices; "
            "the svdvals-based metrics are not sharing a decomposition"
        )
        assert counts.get("svd", 0) <= 2, (
            f"{counts.get('svd')} full svd calls; `_svcca` and `_subspace_overlap` "
            "must share one decomposition each"
        )
        assert counts.get("matrix_rank", 0) == 0, (
            "matrix_rank is an SVD in disguise; `_numerical_rank` must count from "
            "the spectrum it was handed"
        )

    def test_eigvalsh_is_not_repeated_per_metric(self):
        """`_eigenvalue_spread` and `_spectral_clustering_coeff` build the same
        covariance and want the same eigenvalues."""
        online, target = _pair()
        counter, counts = self._count(("eigvalsh",))
        with counter:
            CollapseDiagnostics().compute(online, target)
        assert (
            counts.get("eigvalsh", 0) <= 2
        ), f"{counts.get('eigvalsh')} eigvalsh calls for two covariance matrices"

    def test_budget_holds_when_a_metric_is_computed_alone(self):
        """The saving must come from sharing, not from dropping a metric: a
        lone call still decomposes exactly once and still returns a number."""
        x = torch.randn(64, 8)
        counter, counts = self._count(("svdvals",))
        with counter:
            value = CollapseDiagnostics._effective_rank(x)
        assert counts.get("svdvals") == 1
        assert value > 0.0


# ═══════════════════════════════════════════════════════════════════
#  2. The deduplication is value-preserving
# ═══════════════════════════════════════════════════════════════════


class TestDedupIsValuePreserving:
    """Sharing a decomposition is only safe if it shares the *answer*. These
    compare the shared path against the standalone path, helper by helper, on
    the same input -- the shape of a mistake that shows up only on some inputs
    is why the degenerate shapes are in the list."""

    @staticmethod
    def _shapes():
        g = torch.Generator().manual_seed(20260824)
        base = torch.randn(64, 12, generator=g)
        out = {
            "well_conditioned": base,
            "rank1": torch.randn(1, 12, generator=g).expand(64, 12).contiguous(),
            "rank3": torch.randn(64, 3, generator=g) @ torch.randn(3, 12, generator=g),
            "zeros": torch.zeros(64, 12),
            "ones": torch.ones(64, 12),
            "tiny": base * 1e-9,
            "huge": base * 1e8,
            "one_dead_column": torch.cat([base[:, :6], torch.zeros(64, 6)], dim=1),
            "wide": torch.randn(12, 64, generator=g),
        }
        return out

    def test_svd_based_helpers_match_shared_and_standalone(self):
        names = (
            "_effective_rank",
            "_participation_ratio",
            "_condition_number",
            "_numerical_rank",
            "_singular_value_entropy",
            "_svd_sharpness",
            "_alpha_norm",
        )
        for label, x in self._shapes().items():
            S = torch.linalg.svdvals(x.reshape(-1, x.size(-1)))
            for name in names:
                fn = getattr(CollapseDiagnostics, name)
                shared = fn(x, svals=S)
                alone = fn(x)
                assert shared == alone, f"{name} on {label}: shared={shared} alone={alone}"

    def test_eigval_helpers_match_shared_and_standalone(self):
        for label, x in self._shapes().items():
            flat = x.reshape(-1, x.size(-1)).float()
            if flat.size(0) <= 1:
                continue
            centered = flat - flat.mean(dim=0)
            cov = (centered.T @ centered) / max(flat.size(0) - 1, 1)
            ev = torch.linalg.eigvalsh(cov)
            for name in ("_eigenvalue_spread", "_spectral_clustering_coeff"):
                fn = getattr(CollapseDiagnostics, name)
                assert fn(x, eigvals=ev) == fn(x), f"{name} on {label}"

    def test_svcca_and_subspace_overlap_share_one_decomposition(self):
        x, y = torch.randn(64, 12), torch.randn(64, 10)
        Sx = torch.linalg.svd(x.float(), full_matrices=False)
        Sy = torch.linalg.svd(y.float(), full_matrices=False)
        assert CollapseDiagnostics._svcca(x, y, svd_x=Sx, svd_y=Sy) == (
            CollapseDiagnostics._svcca(x, y)
        )
        assert CollapseDiagnostics._subspace_overlap(x, y, svd_x=Sx, svd_y=Sy) == (
            CollapseDiagnostics._subspace_overlap(x, y)
        )

    def test_compute_is_bit_identical_when_nothing_subsamples(self):
        """Below the subsample cap no row selection happens at all, so the whole
        44-key dict must come out bit-identical to a run whose decompositions
        were computed separately. This is the equivalence the dedup rests on."""
        online, target = _pair(b=SMALL_B, t=SMALL_T, d=SMALL_D)
        m = CollapseDiagnostics().compute(online, target, prev_target_h=target.clone())
        assert len(m) == 44, f"metric count changed: {len(m)} keys, expected 44"

        # Recompute every shared quantity the way the helpers would on their own
        # and require exact equality against what `compute` reported.
        on = online.reshape(-1, SMALL_D)
        tg = target.reshape(-1, SMALL_D)
        S_on, S_tg = torch.linalg.svdvals(on), torch.linalg.svdvals(tg)
        assert m["effective_rank_online"] == CollapseDiagnostics._effective_rank(online)
        assert m["effective_rank_online"] == CollapseDiagnostics._effective_rank(online, svals=S_on)
        assert m["numerical_rank_online"] == CollapseDiagnostics._numerical_rank(online)
        assert m["numerical_rank_online"] == CollapseDiagnostics._numerical_rank(online, svals=S_on)
        assert m["alpha_norm_target"] == CollapseDiagnostics._alpha_norm(target, svals=S_tg)
        assert m["svd_sharpness_online"] == CollapseDiagnostics._svd_sharpness(online, svals=S_on)
        assert m["svcca_online_target"] == CollapseDiagnostics._svcca(online, target)
        assert m["subspace_overlap"] == CollapseDiagnostics._subspace_overlap(online, target)
        assert m["eigenvalue_spread_online"] == CollapseDiagnostics._eigenvalue_spread(online)
        assert m["spectral_clustering_coeff_target"] == (
            CollapseDiagnostics._spectral_clustering_coeff(target)
        )


class TestNumericalRankMatchesTorch:
    """`_numerical_rank` used to call `torch.linalg.matrix_rank`, which is an SVD
    in disguise and was two of the nineteen per-step decompositions. It now
    counts from a spectrum it was handed, so it has to reproduce torch's own
    tolerance rule -- including on the inputs where a plausible reimplementation
    disagrees with torch. `rtol` scales the LARGEST singular value; a rule
    written against S[-1] matches on random matrices and diverges on
    column-rescaled ones, which is exactly what the last case here is for."""

    ATOL = 1e-3
    RTOL = 1e-3

    def _cases(self):
        g = torch.Generator().manual_seed(11)
        scaled = torch.randn(64, 8, generator=g) * torch.tensor(
            [1.0, 1e-3, 1e-5, 1.0, 1e-3, 1e-5, 1.0, 1e-3]
        )
        return {
            "well_conditioned": torch.randn(64, 16, generator=g),
            "zeros": torch.zeros(32, 8),
            "ones": torch.ones(32, 8),
            "rank1": torch.randn(64, 8, generator=g) @ torch.randn(8, 1, generator=g),
            "rank2": torch.randn(64, 16, generator=g) @ torch.randn(16, 2, generator=g),
            "tiny": torch.randn(64, 16, generator=g) * 1e-6,
            "huge": torch.randn(64, 16, generator=g) * 1e6,
            "dead_tail": torch.cat([torch.randn(32, 4, generator=g), torch.zeros(32, 4)], dim=1),
            "half_dead": torch.cat(
                [torch.randn(32, 4, generator=g), torch.randn(32, 4, generator=g) * 1e-7],
                dim=1,
            ),
            "wide": torch.randn(16, 64, generator=g),
            "column_rescaled": scaled,
        }

    def test_spectrum_count_equals_torch_matrix_rank(self):
        for label, A in self._cases().items():
            S = torch.linalg.svdvals(A)
            torch_rank = torch.linalg.matrix_rank(A, atol=self.ATOL, rtol=self.RTOL).item()
            ours = CollapseDiagnostics._numerical_rank(A, svals=S)
            assert ours == torch_rank, (
                f"numerical_rank on {label}: ours={ours} torch={torch_rank} "
                f"(S[0]={S[0].item():.4e}, S[-1]={S[-1].item():.4e})"
            )

    def test_tolerance_scales_with_the_largest_singular_value(self):
        """Stated as its own assertion so that, if torch ever changes the rule,
        the failure names the rule rather than a number.

        The spectrum is chosen so the two candidate rules disagree: 5.0 is above
        both tolerances, 0.05 is above the S[-1]-scaled one and below the
        S[0]-scaled one.
        """
        S = torch.tensor([100.0, 5.0, 0.05])
        assert float((S > max(self.ATOL, self.RTOL * S[0])).sum()) == 2
        assert float((S > max(self.ATOL, self.RTOL * S[-1])).sum()) == 3

    def test_default_path_still_delegates_to_torch(self):
        """No spectrum supplied -> `torch.linalg.matrix_rank`, unchanged."""
        A = torch.randn(32, 8)
        assert (
            CollapseDiagnostics._numerical_rank(A)
            == torch.linalg.matrix_rank(A, atol=1e-3, rtol=1e-3).item()
        )


# ═══════════════════════════════════════════════════════════════════
#  3. The RNG contract -- the precondition for any frequency gate
# ═══════════════════════════════════════════════════════════════════


class TestDiagnosticsDoNotTouchTheGlobalRNG:
    """The reason this card's obvious fix is not automatically safe.

    A diagnostic is a read-only observer. If it draws from the training device's
    global generator, then the number of draws it makes becomes part of the
    model's randomness: change the batch size, or run the diagnostics on 9 of
    every 10 steps, and the training trajectory moves. That is why gating
    `compute()` behind `itr % log_freq == 0` is only a pure optimisation once
    these hold.
    """

    def test_compute_leaves_the_global_cpu_state_untouched(self):
        online, target = _pair()
        seed_everything(4242)
        before = torch.get_rng_state().clone()
        CollapseDiagnostics().compute(online, target)
        assert torch.equal(torch.get_rng_state(), before)

    def test_the_draw_after_compute_is_the_draw_without_it(self):
        """Stronger and easier to read than comparing state objects: the next
        random number a consumer sees must not depend on whether a diagnostic
        ran."""
        online, target = _pair()
        seed_everything(4242)
        without = torch.rand(8).clone()
        seed_everything(4242)
        CollapseDiagnostics().compute(online, target)
        with_diag = torch.rand(8).clone()
        assert torch.equal(without, with_diag)

    @pytest.mark.parametrize(
        "name,args",
        [
            ("_intrinsic_dim_score", lambda x: (x,)),
            ("_mean_pairwise_cosine", lambda x: (x.reshape(-1, x.size(-1)),)),
            ("_uniformity", lambda x: (x.reshape(-1, x.size(-1)),)),
            ("_alignment", lambda x: (x.reshape(-1, x.size(-1)), x.reshape(-1, x.size(-1)))),
        ],
    )
    def test_each_subsampling_helper_leaves_the_global_state_untouched(self, name, args):
        x = torch.randn(_SUBSAMPLE_LIMIT + 40, 8)
        fn = getattr(CollapseDiagnostics, name)
        seed_everything(99)
        before = torch.get_rng_state().clone()
        fn(*args(x))
        assert torch.equal(torch.get_rng_state(), before), f"{name} consumed the global RNG"

    def test_repeated_calls_still_vary(self):
        """Decoupling from the global stream must not freeze the subsample: a
        fixed row set every step would make the four metrics see one slice of
        the batch and quietly stop being estimators."""
        x = torch.randn(_SUBSAMPLE_LIMIT * 3, 8)
        vals = [CollapseDiagnostics._mean_pairwise_cosine(x) for _ in range(4)]
        assert len(set(vals)) > 1, "subsample is not varying between calls"

    def test_generator_is_a_pure_function_of_the_draw_index(self):
        a = torch.randperm(1000, generator=_derive_subsample_rng(3))
        b = torch.randperm(1000, generator=_derive_subsample_rng(3))
        c = torch.randperm(1000, generator=_derive_subsample_rng(4))
        assert torch.equal(a, b), "the same draw index must give the same permutation"
        assert not torch.equal(a, c), "different draw indices must give different permutations"

    def test_seeded_run_is_reproducible(self):
        """A full `seed_everything` still reproduces the diagnostic dict, so the
        decoupling did not trade training determinism for diagnostics.

        The process-level draw counter is rewound before each call, because it
        is deliberately not checkpointed -- a run's reproducibility is
        "same seed AND same call sequence", which is what this asserts.
        """
        saved = collapse_mod._subsample_draw_counter
        try:
            collapse_mod._subsample_draw_counter = 0
            online, target = _pair(seed=7)
            first = CollapseDiagnostics().compute(online, target)

            collapse_mod._subsample_draw_counter = 0
            online2, target2 = _pair(seed=7)
            second = CollapseDiagnostics().compute(online2, target2)
        finally:
            collapse_mod._subsample_draw_counter = saved
        assert first == second

    def test_advance_helper_returns_a_fresh_generator_each_call(self):
        g1 = _next_subsample_rng()
        g2 = _next_subsample_rng()
        assert g1 is not g2
        # Both must actually draw, and differently.
        assert not torch.equal(torch.randperm(64, generator=g1), torch.randperm(64, generator=g2))

    def test_subsample_index_is_none_below_the_cap(self):
        assert (
            CollapseDiagnostics._subsample_index(10, _SUBSAMPLE_LIMIT, torch.device("cpu")) is None
        )
        idx = CollapseDiagnostics._subsample_index(1000, _SUBSAMPLE_LIMIT, torch.device("cpu"))
        assert idx is not None
        assert idx.numel() == _SUBSAMPLE_LIMIT
        assert idx.unique().numel() == _SUBSAMPLE_LIMIT, "rows must be distinct"

    def test_alignment_subsamples_pairs_together(self):
        """x and y are indexed with the same index, so a pair stays a pair. If
        they drifted apart, `alignment` would be measuring the distance between
        unrelated rows and would still look plausible.

        One column, so the row distance under a constant offset is exactly 1.0
        and the expected value is readable rather than derived.
        """
        x = torch.randn(_SUBSAMPLE_LIMIT + 20, 1)
        y = x + 1.0  # preserved only if x[idx] and y[idx] share idx
        assert CollapseDiagnostics._alignment(x, y) == pytest.approx(1.0, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════
#  4. The failure contract still holds
# ═══════════════════════════════════════════════════════════════════


class TestFailureContract:
    """`_SharedSpectral` swallows a decomposition failure and hands back
    `None`, which every metric already understood as "compute your own". The
    module advertises that it never crashes the training loop; deduplication
    must not quietly turn a LAPACK error into a `KeyError`."""

    def test_compute_survives_a_decomposition_that_raises(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("simulated LAPACK failure")

        for name in ("svdvals", "svd", "eigvalsh", "matrix_rank"):
            monkeypatch.setattr(torch.linalg, name, boom)
        online, target = _pair(b=SMALL_B, t=SMALL_T, d=SMALL_D)
        m = CollapseDiagnostics().compute(online, target, prev_target_h=target.clone())

        # A full dict of sentinels, not an exception and not a short dict.
        assert len(m) == 44
        for key in (
            "effective_rank_online",
            "participation_ratio_target",
            "collapsed_dim_ratio_online",
            "sv_entropy_online",
            "svcca_online_target",
            "subspace_overlap",
            "eigenvalue_spread_online",
        ):
            assert key in m, f"{key} vanished when the decompositions failed"
        assert m["effective_rank_online"] == 0.0
        assert math.isinf(m["condition_number_online"])

    def test_one_side_failing_does_not_break_the_other(self, monkeypatch):
        """`online` is (B, T, D) and `target` is (B, T, D), so a decomposition
        that fails only for the second one must still leave the first one's
        metrics real. Pins that the cache is keyed per tensor, not globally."""
        online, target = _pair(b=SMALL_B, t=SMALL_T, d=SMALL_D)
        target_bad = target.clone()
        target_bad[0, 0, 0] = float("nan")
        m = CollapseDiagnostics().compute(online, target_bad)
        assert math.isfinite(m["effective_rank_online"])
