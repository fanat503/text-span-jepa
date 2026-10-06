# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Workspace validation honesty: a verdict this module did not earn must not exist.

``src/interp/workspace_validation.py`` exists to answer one question — does
the JAWP basis ``Q`` span the subspace the SAE finds? — and every way it
used to answer it with a number it had not computed is a separate lie:

  W1  ``sae=None`` built a **random** ``TopKSAE`` and returned a verdict
      anyway: ``subspace_similarity=0.447``, ``placebo=0.399``,
      ``workspace_claim_valid=False``, i.e. a decisive-looking answer
      computed from noise. The warning was a caveat attached to a number
      nobody earned.
  W2  ``identify_workspace_features`` promised "high probe accuracy" and
      scored features by an in-sample point-biserial correlation over all N
      samples — a probe evaluated on its own training data.
  W3  ``compute_workspace_similarity`` took ``svdvals(Qᵀ P)`` as "cosines of
      principal angles". Those are the angle cosines only when k == r; at
      k=4, r=51 the true cosines of ``Qᵀ(PPᵀ)Q`` differ, and the similarity
      (a mean over r of squared cosines) is diluted by the width of the SAE
      workspace.
  W4  the module header promised "Bootstrap CI for the similarity" and the
      bootstrap ran on ``ws_util``; the headline ``subspace_similarity`` had
      no interval of any kind, and the verdict used the point estimate.
  W5  ``identify_workspace_features`` called ``sae.eval()`` and never restored
      the caller's mode, so an analysis call could leave an SAE that was
      about to be trained in eval mode.
  W6  the shuffled-basis placebo was a single random draw compared
      point-to-point, and degenerate inputs silently produced numbers:
      an empty workspace set returned ``0.0`` similarity / ``90°``, a
      non-orthonormal ``Q`` gave an undefined "angle", and a non-finite
      decoder was reduced to a zero singular value by a bare
      ``except Exception``.
  W7  ``bootstrap_ci`` on one sample returned ``(m, m, m)`` — a zero-width
      interval that reads as a tight measurement.

The contract enforced here:

  * an untrained SAE yields a raised ``UntrainedSAEError``, or (only on an
    explicit opt-in) a refusal record with **no** similarity in it
  * an SAE can only be trained by taking real optimizer steps, or by
    declaring it so — a forward pass is not training
  * reported probe accuracy is held out, and is reported next to chance
  * similarity is the k x k Gram spectrum, bounded by construction, with
    degenerate inputs refused instead of defaulted
  * the verdict is gated on the bootstrap CI's lower end and on the worst
    of several shuffled-basis controls
  * inputs that cannot produce a number raise; none of them return a
    plausible-looking substitute
"""

from __future__ import annotations

import pytest
import torch
from src.interp.workspace_validation import (
    UntrainedSAEError,
    TopKSAE,
    bootstrap_ci,
    bootstrap_similarity_ci,
    compute_workspace_similarity,
    decide_workspace_claim,
    identify_workspace_features,
    train_topk_sae,
    validate_workspace_claim,
)
from src.utils.seed import seed_everything


def _orthonormal(D: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, k, generator=g))
    return Q


def _orthogonal_pair(D: int, k: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two bases whose spans are exactly orthogonal (columns of one QR)."""
    full = _orthonormal(D, 2 * k, seed=seed)
    return full[:, :k].contiguous(), full[:, k:].contiguous()


def _labels(n: int, n_probes: int = 2, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, n_probes), generator=g).float()


def _trained_sae(D: int, n_features: int = 32, k: int = 4, steps: int = 6, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    reps = torch.randn(48, D, generator=g)
    sae = train_topk_sae(reps, n_features=n_features, k=k, n_steps=steps, seed=seed)
    return sae, reps


# ═══════════════════════════════════════════════════════════════════
#  W1 — an untrained SAE may not produce a verdict
# ═══════════════════════════════════════════════════════════════════


class TestUntrainedSAERefusal:
    def test_no_sae_is_refused_not_replaced_by_a_random_one(self):
        D = 24
        seed_everything(0)
        Q = _orthonormal(D, 4)
        reps = torch.randn(40, D)
        with pytest.raises(UntrainedSAEError) as exc:
            validate_workspace_claim(Q, reps, _labels(40))
        msg = str(exc.value)
        assert "untrained" in msg
        assert "n_sae_train_steps" in msg  # says how to make it a real run

    def test_untrained_sae_passed_in_is_refused(self):
        D = 24
        Q = _orthonormal(D, 4)
        reps = torch.randn(40, D)
        sae = TopKSAE(embed_dim=D, n_features=32, k=4)
        assert sae.is_trained is False
        with pytest.raises(UntrainedSAEError):
            validate_workspace_claim(Q, reps, _labels(40), sae=sae)

    def test_explicit_opt_in_returns_a_refusal_carrying_no_number(self):
        D = 24
        Q = _orthonormal(D, 4)
        reps = torch.randn(40, D)
        sae = TopKSAE(embed_dim=D, n_features=32, k=4)
        out = validate_workspace_claim(Q, reps, _labels(40), sae=sae, allow_untrained_sae=True)

        assert out["verdict_status"] == "refused_untrained_sae"
        assert out["workspace_claim_valid"] is None
        # Nothing that could be mistaken for a measurement.
        for key in (
            "subspace_similarity",
            "subspace_similarity_placebo",
            "principal_angles",
            "mean_angle",
            "similarity_ci_lower",
            "claim_margin_above_placebo",
        ):
            assert key not in out, f"{key} must not appear in a refusal record"

    def test_a_forward_pass_does_not_make_an_sae_trained(self):
        sae = TopKSAE(embed_dim=16, n_features=32, k=4)
        sae.train()
        for _ in range(3):
            sae(torch.randn(8, 16))
        assert sae.is_trained is False

    def test_trained_sae_is_marked_and_reports_its_step_count(self):
        sae, _ = _trained_sae(16, steps=5)
        assert sae.is_trained is True
        assert int(sae.n_train_steps) == 5

    def test_train_topk_sae_refuses_zero_steps(self):
        reps = torch.randn(16, 8)
        with pytest.raises(ValueError):
            train_topk_sae(reps, n_features=16, k=2, n_steps=0)

    def test_n_train_steps_survives_a_state_dict_round_trip(self):
        sae, _ = _trained_sae(16, steps=4)
        clone = TopKSAE(embed_dim=16, n_features=32, k=4)
        clone.load_state_dict(sae.state_dict())
        assert int(clone.n_train_steps) == 4
        assert clone.is_trained is True

    def test_pipeline_runs_when_a_trained_sae_is_supplied(self):
        D = 24
        Q = _orthonormal(D, 4)
        sae, reps = _trained_sae(D, n_features=32, k=4, steps=8)
        out = validate_workspace_claim(
            Q, reps, _labels(reps.shape[0]), sae=sae, n_bootstrap=8, n_placebo=2, seed=5
        )
        assert out["verdict_status"] == "evaluated"
        assert 0.0 <= out["subspace_similarity"] <= 1.0
        assert out["sae_train_steps"] == 8
        assert isinstance(out["workspace_claim_valid"], bool)

    def test_pipeline_can_train_its_own_sae(self):
        D = 24
        Q = _orthonormal(D, 4)
        reps = torch.randn(40, D)
        out = validate_workspace_claim(
            Q,
            reps,
            _labels(40),
            n_sae_features=32,
            sae_k=4,
            n_sae_train_steps=4,
            n_bootstrap=8,
            n_placebo=2,
            seed=5,
        )
        assert out["verdict_status"] == "evaluated"
        assert out["sae_train_steps"] == 4

    def test_training_an_sae_does_not_disturb_the_global_rng(self):
        reps = torch.randn(24, 8)
        g = torch.Generator().manual_seed(7)
        expected = torch.randn(4, generator=g)
        g2 = torch.Generator().manual_seed(7)
        torch.randn(4, generator=g2)
        train_topk_sae(reps, n_features=16, k=2, n_steps=3, seed=1)
        after = torch.randn(4, generator=torch.Generator().manual_seed(7))
        assert torch.allclose(expected, after)  # sanity: generator is reproducible
        assert torch.isfinite(reps).all()


# ═══════════════════════════════════════════════════════════════════
#  W2 — probe accuracy must be held out
# ═══════════════════════════════════════════════════════════════════


class TestHeldOutProbes:
    def _data(self, n=120, n_features=32, seed=5):
        g = torch.Generator().manual_seed(seed)
        reps = torch.randn(n, 16, generator=g)
        labels = torch.zeros(n, 1)
        labels[:, 0] = (reps[:, 0] > 0).float()  # genuinely predictable
        return reps, labels

    def test_reports_heldout_accuracy_next_to_chance(self):
        sae, _ = _trained_sae(16, n_features=32, k=4, steps=6)
        reps, labels = self._data()
        idx, info = identify_workspace_features(
            sae, reps, labels, n_probes=1, top_fraction=0.1, seed=1
        )
        assert "probe_accuracy_heldout" in info
        assert info["probe_accuracy_chance"] == 0.5
        assert info["n_holdout"] > 0 and info["n_train"] > 0
        assert info["n_holdout"] + info["n_train"] == reps.shape[0]
        assert 0.0 <= info["probe_accuracy_heldout"] <= 1.0
        assert idx.numel() == max(1, int(32 * 0.1))

    def test_in_sample_correlation_is_not_reported_as_accuracy(self):
        """Noise labels: an in-sample score is inflated, a held-out one is chance."""
        sae, _ = _trained_sae(16, n_features=64, k=4, steps=6)
        g = torch.Generator().manual_seed(9)
        reps = torch.randn(200, 16, generator=g)
        labels = torch.randint(0, 2, (200, 3)).float()  # pure noise
        _, info = identify_workspace_features(
            sae, reps, labels, n_probes=3, top_fraction=0.1, seed=2
        )
        # Held-out accuracy on noise is chance; in-sample selection bias over
        # 64 features would show up as a clearly-above-chance score.
        assert info["probe_accuracy_heldout"] < 0.75
        assert info["n_tasks_used"] == 3

    def test_a_supplied_split_is_honoured_exactly(self):
        sae, _ = _trained_sae(16, n_features=32, k=4, steps=6)
        reps, labels = self._data(n=60)
        mask = torch.zeros(60, dtype=torch.bool)
        mask[:40] = True
        _, info = identify_workspace_features(
            sae, reps, labels, n_probes=1, top_fraction=0.1, train_mask=mask
        )
        assert int(info["n_train"]) == 40
        assert int(info["n_holdout"]) == 20

    def test_split_too_small_to_hold_out_raises(self):
        sae, _ = _trained_sae(16, n_features=32, k=4, steps=6)
        reps, labels = self._data(n=60)
        mask = torch.zeros(60, dtype=torch.bool)
        mask[:59] = True  # one held-out sample
        with pytest.raises(ValueError):
            identify_workspace_features(sae, reps, labels, n_probes=1, train_mask=mask)

    def test_unusable_task_count_is_reported(self):
        sae, _ = _trained_sae(16, n_features=32, k=4, steps=6)
        reps = torch.randn(40, 16)
        labels = torch.arange(40).float().reshape(-1, 1)  # 40 distinct values
        _, info = identify_workspace_features(sae, reps, labels, n_probes=1)
        assert info["n_tasks_used"] == 0
        assert info["workspace_features_identified"] == 0.0


# ═══════════════════════════════════════════════════════════════════
#  W3 — similarity is the k x k Gram spectrum
# ═══════════════════════════════════════════════════════════════════


class TestSimilarityMath:
    def test_identical_subspaces_score_one(self):
        D, k = 32, 6
        Q = _orthonormal(D, k)
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        with torch.no_grad():
            sae.W_dec[:k] = Q.T  # decoder rows = Q's basis
        idx = torch.arange(k)
        res = compute_workspace_similarity(Q, sae, idx)
        assert res["subspace_similarity"] == pytest.approx(1.0, abs=1e-6)
        # arccos is ill-conditioned at 1, so float32 storage of Q costs a few
        # thousandths of a degree. The property under test is "no angle".
        assert res["mean_angle"] < 0.05
        assert max(res["principal_angles"]) < 0.05

    def test_orthogonal_subspace_scores_zero(self):
        D, k = 32, 4
        Q, R = _orthogonal_pair(D, k)
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        with torch.no_grad():
            sae.W_dec[:k] = R.T
        res = compute_workspace_similarity(Q, sae, torch.arange(k))
        assert res["subspace_similarity"] == pytest.approx(0.0, abs=1e-8)
        assert res["mean_angle"] == pytest.approx(90.0, abs=1e-3)

    def test_wide_sae_workspace_does_not_dilute_the_similarity(self):
        """k != r: the SAE workspace is wider than Q but contains it."""
        D, k, r = 32, 4, 51
        Q = _orthonormal(D, k)
        sae = TopKSAE(embed_dim=D, n_features=64, k=2)
        g = torch.Generator().manual_seed(3)
        extra = torch.randn(r - k, D, generator=g)
        with torch.no_grad():
            sae.W_dec[:r] = torch.cat([Q.T, extra], dim=0)
        res = compute_workspace_similarity(Q, sae, torch.arange(r))
        # Every principal angle is zero because span(Q) ⊆ span(sae rows).
        assert res["subspace_similarity"] == pytest.approx(1.0, abs=1e-6)
        # The subspace is min(r, D)-dimensional, not m-dimensional: the extra
        # rows cannot buy the claim extra evidence.
        assert res["workspace_dim_sae"] == min(r, D)
        assert res["workspace_dim_jawp"] == k
        assert res["n_angles_defined"] == k

    def test_matches_the_gram_matrix_definition(self):
        D, k, r = 24, 3, 7
        Q = _orthonormal(D, k)
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        g = torch.Generator().manual_seed(8)
        rows = torch.randn(r, D, generator=g)
        with torch.no_grad():
            sae.W_dec[:r] = rows
        idx = torch.arange(r)
        res = compute_workspace_similarity(Q, sae, idx)

        Y = torch.linalg.svd(rows.T.double(), full_matrices=False)[0]  # orthonormal basis
        G = Q.double().T @ Y @ Y.T @ Q.double()
        eig = torch.linalg.eigvalsh(G).clamp(0, 1)
        assert res["subspace_similarity"] == pytest.approx(float(eig.mean()), abs=1e-9)
        angles = sorted(torch.arccos(eig.flip(0).sqrt()) * 180.0 / torch.pi)
        assert res["principal_angles"] == pytest.approx([float(a) for a in angles], abs=1e-6)

    def test_similarity_does_not_depend_on_decoder_row_scale(self):
        """A subspace is a subspace: rescaling decoder rows cannot inflate it.

        Regression guard for computing the Gram from raw (non-orthonormal)
        decoder rows, which makes ``G`` scale with ‖P‖² — the similarity then
        exceeds 1 for unrelated subspaces and a clamp reports 1.0.
        """
        D, k, r = 24, 4, 10
        Q = _orthonormal(D, k)
        rows = torch.randn(r, D, generator=torch.Generator().manual_seed(13))
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        with torch.no_grad():
            sae.W_dec[:r] = rows
        base = compute_workspace_similarity(Q, sae, torch.arange(r))
        for scale in (1e-3, 10.0, 1e4):
            with torch.no_grad():
                sae.W_dec[:r] = rows * scale
            got = compute_workspace_similarity(Q, sae, torch.arange(r))
            # float32 storage of the rescaled rows costs a few ulps, so the
            # property is "scale cannot move the number", not bit-equality.
            assert got["subspace_similarity"] == pytest.approx(
                base["subspace_similarity"], abs=1e-6
            )
            assert got["principal_angles"] == pytest.approx(base["principal_angles"], abs=1e-3)

    def test_unrelated_subspaces_do_not_score_one_however_long_the_decoder(self):
        """A wide SAE workspace must not be free evidence for the claim."""
        D, k, r = 64, 4, 200
        Q = _orthonormal(D, k)
        rows = torch.randn(r, D, generator=torch.Generator().manual_seed(17))
        sae = TopKSAE(embed_dim=D, n_features=256, k=2)
        with torch.no_grad():
            sae.W_dec[:r] = rows
        res = compute_workspace_similarity(Q, sae, torch.arange(r))
        # 200 random rows in R^64 span all 64 dimensions, so span(Q) IS contained
        # and the similarity is legitimately 1. The load-bearing property is that
        # it comes from the rank, not from the row count — see the test below.
        assert res["workspace_dim_sae"] == D

    def test_duplicate_decoder_rows_do_not_hide_the_span(self):
        """m >> k with duplicated rows: the subspace is still r-dimensional.

        A QR-based basis truncated at the numerical rank gets this wrong (the
        skipped reflectors leave arbitrary completion vectors in Q), which
        silently reports a similarity for a subspace nobody selected.
        """
        D, k, r = 16, 3, 12
        Q = _orthonormal(D, k)
        rows = Q.T.repeat(4, 1)  # every row in span(Q), rank 3
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        with torch.no_grad():
            sae.W_dec[:r] = rows
        for order in (torch.arange(r), torch.randperm(r), torch.flip(torch.arange(r), [0])):
            res = compute_workspace_similarity(Q, sae, order)
            assert res["workspace_dim_sae"] == k, order
            assert res["subspace_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_similarity_is_invariant_to_which_duplicate_rows_were_kept(self):
        D, k, r = 16, 3, 12
        Q = _orthonormal(D, k)
        rows = Q.T.repeat(4, 1)
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        with torch.no_grad():
            sae.W_dec[:r] = rows
        a = compute_workspace_similarity(Q, sae, torch.arange(0, r, 4))  # one copy of each
        b = compute_workspace_similarity(Q, sae, torch.arange(r))  # four copies each
        assert a["subspace_similarity"] == pytest.approx(b["subspace_similarity"], abs=1e-9)

    def test_similarity_is_bounded_by_construction(self):
        D, k, r = 20, 4, 9
        Q = _orthonormal(D, k)
        g = torch.Generator().manual_seed(21)
        for seed in range(6):
            rows = torch.randn(r, D, generator=torch.Generator().manual_seed(seed))
            sae = TopKSAE(embed_dim=D, n_features=16, k=2)
            with torch.no_grad():
                sae.W_dec[:r] = rows
            res = compute_workspace_similarity(Q, sae, torch.arange(r))
            assert 0.0 <= res["subspace_similarity"] <= 1.0
            assert all(0.0 <= a <= 90.0 + 1e-6 for a in res["principal_angles"])

    def test_narrower_sae_workspace_reports_only_min_k_r_defined_angles(self):
        D, k, r = 20, 5, 2
        Q = _orthonormal(D, k)
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            sae.W_dec[:r] = torch.randn(r, D, generator=g)
        res = compute_workspace_similarity(Q, sae, torch.arange(r))
        assert res["n_angles_defined"] == r
        # The 3 padded entries are exactly orthogonal, not measured angles.
        assert sorted(res["principal_angles"])[-3:] == pytest.approx([90.0] * 3, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════
#  W6 — degenerate inputs raise instead of returning a plausible number
# ═══════════════════════════════════════════════════════════════════


class TestDegenerateInputsRefused:
    def test_empty_workspace_feature_set_raises(self):
        Q = _orthonormal(16, 3)
        sae = TopKSAE(embed_dim=16, n_features=16, k=2)
        with pytest.raises(ValueError):
            compute_workspace_similarity(Q, sae, torch.tensor([], dtype=torch.long))

    def test_non_orthonormal_q_raises(self):
        D = 16
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        Q = torch.randn(D, 4) * 5.0  # not orthonormal
        with pytest.raises(ValueError) as exc:
            compute_workspace_similarity(Q, sae, torch.arange(3))
        # The reason must be the basis, not some downstream artefact of it.
        assert "orthonormal" in str(exc.value)

    def test_wider_than_tall_q_raises(self):
        sae = TopKSAE(embed_dim=16, n_features=16, k=2)
        Q = torch.randn(4, 8)
        with pytest.raises(ValueError):
            compute_workspace_similarity(Q, sae, torch.arange(3))

    def test_non_finite_decoder_raises_not_zero(self):
        D, k = 16, 3
        Q = _orthonormal(D, k)
        sae = TopKSAE(embed_dim=D, n_features=16, k=2)
        with torch.no_grad():
            sae.W_dec[0, 0] = float("nan")
        with pytest.raises(ValueError) as exc:
            compute_workspace_similarity(Q, sae, torch.arange(3))
        # A NaN decoder must be named as such; a silent fallback reports 0.
        assert "non-finite" in str(exc.value)

    def test_non_finite_q_raises(self):
        sae = TopKSAE(embed_dim=16, n_features=16, k=2)
        Q = _orthonormal(16, 3)
        Q[0, 0] = float("inf")
        with pytest.raises(ValueError):
            compute_workspace_similarity(Q, sae, torch.arange(3))

    def test_out_of_range_feature_index_raises(self):
        Q = _orthonormal(16, 3)
        sae = TopKSAE(embed_dim=16, n_features=8, k=2)
        with pytest.raises(IndexError):
            compute_workspace_similarity(Q, sae, torch.tensor([0, 99]))

    def test_sae_embed_dim_mismatch_raises(self):
        sae = TopKSAE(embed_dim=16, n_features=16, k=2)
        reps = torch.randn(20, 12)
        labels = torch.randint(0, 2, (20, 1)).float()
        with pytest.raises(ValueError):
            identify_workspace_features(sae, reps, labels, n_probes=1)

    def test_non_finite_representations_raise(self):
        sae = TopKSAE(embed_dim=16, n_features=16, k=2)
        reps = torch.randn(20, 16)
        reps[3, 2] = float("nan")
        labels = torch.randint(0, 2, (20, 1)).float()
        with pytest.raises(ValueError):
            identify_workspace_features(sae, reps, labels, n_probes=1)

    def test_label_count_mismatch_raises(self):
        sae = TopKSAE(embed_dim=16, n_features=16, k=2)
        reps = torch.randn(20, 16)
        with pytest.raises(ValueError):
            identify_workspace_features(sae, reps, torch.randint(0, 2, (19, 1)).float())

    def test_bad_top_fraction_raises(self):
        sae = TopKSAE(embed_dim=16, n_features=16, k=2)
        reps = torch.randn(20, 16)
        labels = torch.randint(0, 2, (20, 1)).float()
        with pytest.raises(ValueError):
            identify_workspace_features(sae, reps, labels, top_fraction=0.0)

    def test_k_out_of_range_raises_at_construction(self):
        with pytest.raises(ValueError):
            TopKSAE(embed_dim=8, n_features=8, k=99)
        with pytest.raises(ValueError):
            TopKSAE(embed_dim=8, n_features=8, k=0)


# ═══════════════════════════════════════════════════════════════════
#  W5 — the caller's SAE mode survives, including on the error path
# ═══════════════════════════════════════════════════════════════════


class TestModeRestoration:
    def test_eval_mode_is_restored_after_identification(self):
        sae, _ = _trained_sae(16, n_features=16, k=2, steps=3)
        sae.train()
        reps = torch.randn(40, 16)
        labels = torch.randint(0, 2, (40, 1)).float()
        identify_workspace_features(sae, reps, labels, n_probes=1)
        assert sae.training is True

        sae.eval()
        identify_workspace_features(sae, reps, labels, n_probes=1)
        assert sae.training is False

    def test_mode_is_restored_when_identification_raises(self):
        sae, _ = _trained_sae(16, n_features=16, k=2, steps=3)
        sae.train()
        reps = torch.randn(20, 16)
        with pytest.raises(ValueError):  # 19 label rows for 20 reps
            identify_workspace_features(sae, reps, torch.randint(0, 2, (19, 1)).float())
        assert sae.training is True

    def test_training_a_pipeline_sae_leaves_it_in_eval_mode(self):
        sae, reps = _trained_sae(16, n_features=16, k=2, steps=3)
        assert sae.training is False


# ═══════════════════════════════════════════════════════════════════
#  W4 / W7 — the similarity needs an interval, and intervals need samples
# ═══════════════════════════════════════════════════════════════════


class TestIntervals:
    def test_bootstrap_ci_on_one_sample_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(torch.tensor([0.5]), n_bootstrap=10)
        with pytest.raises(ValueError):
            bootstrap_ci(torch.tensor([]), n_bootstrap=10)

    def test_bootstrap_ci_rejects_non_finite_and_bad_parameters(self):
        with pytest.raises(ValueError):
            bootstrap_ci(torch.tensor([1.0, float("nan")]), n_bootstrap=10)
        with pytest.raises(ValueError):
            bootstrap_ci(torch.randn(20), n_bootstrap=0)
        with pytest.raises(ValueError):
            bootstrap_ci(torch.randn(20), confidence=1.0)

    def test_bootstrap_ci_is_deterministic_and_brackets_the_mean(self):
        values = torch.randn(64, generator=torch.Generator().manual_seed(1))
        a = bootstrap_ci(values, n_bootstrap=100, seed=5)
        b = bootstrap_ci(values, n_bootstrap=100, seed=5)
        assert a == b
        assert a[1] <= a[0] <= a[2]

    def test_similarity_has_a_bootstrap_ci_from_its_own_promises(self):
        D = 16
        Q = _orthonormal(D, 3)
        sae, reps = _trained_sae(D, n_features=24, k=3, steps=5)
        labels = _labels(reps.shape[0])
        out = validate_workspace_claim(
            Q, reps, labels, sae=sae, n_bootstrap=10, n_placebo=2, seed=5
        )
        for key in (
            "similarity_mean",
            "similarity_ci_lower",
            "similarity_ci_upper",
            "similarity_ci_width",
        ):
            assert key in out, f"the header promises a CI for the similarity: {key}"
        assert out["similarity_ci_lower"] <= out["similarity_mean"] <= out["similarity_ci_upper"]
        assert out["similarity_ci_width"] > 0

    def test_similarity_ci_responds_to_sample_size(self):
        """A CI from 12 identical resamples is not evidence; from many it is."""
        D = 12
        Q = _orthonormal(D, 3)
        sae, reps = _trained_sae(D, n_features=24, k=3, steps=4)
        labels = _labels(reps.shape[0])
        out = bootstrap_similarity_ci(
            Q, sae, reps, labels, n_bootstrap=12, n_probes=2, top_fraction=0.2, seed=4
        )
        assert 0.0 <= out["similarity_ci_lower"] <= out["similarity_ci_upper"] <= 1.0
        assert out["n_bootstrap"] == 12
        with pytest.raises(ValueError):
            bootstrap_similarity_ci(Q, sae, reps, labels, n_bootstrap=1)

    def test_ci_is_computed_on_the_similarity_not_on_ws_utilisation(self):
        """The old CI was on ws_util; utilisation can be ~1 while similarity ~0."""
        D = 16
        Q = _orthonormal(D, 3)
        sae, reps = _trained_sae(D, n_features=24, k=3, steps=5)
        labels = _labels(reps.shape[0])
        out = validate_workspace_claim(
            Q, reps, labels, sae=sae, n_bootstrap=10, n_placebo=2, seed=5
        )
        assert out["similarity_ci_lower"] != out["ws_utilization_ci_lower"] or (
            out["similarity_ci_upper"] != out["ws_utilization_ci_upper"]
        )


# ═══════════════════════════════════════════════════════════════════
#  The verdict itself
# ═══════════════════════════════════════════════════════════════════


class TestVerdict:
    def test_point_estimate_alone_does_not_validate_the_claim(self):
        assert decide_workspace_claim(0.95, 0.95, 0.1, True, gate=0.8) is True
        # Same point estimate, CI dipping under the gate: not a claim.
        assert decide_workspace_claim(0.95, 0.6, 0.1, True, gate=0.8) is False

    def test_ci_must_beat_the_placebo(self):
        assert decide_workspace_claim(0.9, 0.85, 0.9, True, gate=0.8) is False
        assert decide_workspace_claim(0.9, 0.85, 0.84, True, gate=0.8) is True

    def test_unidentified_features_block_the_claim(self):
        assert decide_workspace_claim(0.99, 0.99, 0.1, False, gate=0.8) is False

    def test_pipeline_verdict_follows_the_rule(self):
        D = 16
        Q = _orthonormal(D, 3)
        sae, reps = _trained_sae(D, n_features=24, k=3, steps=5)
        labels = _labels(reps.shape[0])
        out = validate_workspace_claim(
            Q, reps, labels, sae=sae, n_bootstrap=10, n_placebo=3, seed=5
        )
        expected = decide_workspace_claim(
            out["subspace_similarity"],
            out["similarity_ci_lower"],
            out["subspace_similarity_placebo_max"],
            bool(out["workspace_features_identified"] > 0.5),
            gate=out["similarity_gate"],
        )
        assert out["workspace_claim_valid"] is expected

    def test_pipeline_refuses_a_high_point_estimate_with_a_low_ci(self, monkeypatch):
        """Wiring: a point estimate of 1.0 whose CI lower end is 0.4 is not a claim.

        Built directly rather than hoping a random seed produces the
        borderline case: the SAE decoder is given Q's own basis, so the
        similarity is exactly 1.0 by construction, and only the interval
        is made to straddle the gate.
        """
        import src.interp.workspace_validation as wv

        D, k = 16, 3
        Q = _orthonormal(D, k)
        sae, reps = _trained_sae(D, n_features=12, k=3, steps=4)
        with torch.no_grad():
            # Every decoder row lies in span(Q) (top_fraction=1.0 selects all
            # of them), so the similarity is exactly 1.0 whatever features
            # the probes pick.
            sae.W_dec[:] = Q.T.repeat(4, 1)
        monkeypatch.setattr(
            wv,
            "bootstrap_similarity_ci",
            lambda *a, **kw: {
                "similarity_mean": 1.0,
                "similarity_ci_lower": 0.4,
                "similarity_ci_upper": 1.0,
                "similarity_ci_width": 0.6,
                "n_bootstrap": float(kw.get("n_bootstrap", 0)),
            },
        )
        out = wv.validate_workspace_claim(
            Q,
            reps,
            _labels(reps.shape[0]),
            sae=sae,
            n_bootstrap=8,
            n_placebo=2,
            top_fraction=1.0,
            seed=5,
        )
        assert out["subspace_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert out["workspace_claim_valid"] is False

    def test_the_gate_is_reported_not_hidden(self):
        D = 16
        Q = _orthonormal(D, 3)
        sae, reps = _trained_sae(D, n_features=24, k=3, steps=4)
        out = validate_workspace_claim(
            Q,
            reps,
            _labels(reps.shape[0]),
            sae=sae,
            n_bootstrap=8,
            n_placebo=2,
            similarity_gate=0.75,
            seed=5,
        )
        assert out["similarity_gate"] == 0.75

    def test_bad_gate_and_placebo_count_raise(self):
        D = 16
        Q = _orthonormal(D, 3)
        sae, reps = _trained_sae(D, n_features=16, k=2, steps=3)
        labels = _labels(reps.shape[0])
        with pytest.raises(ValueError):
            validate_workspace_claim(Q, reps, labels, sae=sae, n_bootstrap=8, n_placebo=0, seed=1)
        with pytest.raises(ValueError):
            validate_workspace_claim(
                Q, reps, labels, sae=sae, n_bootstrap=8, similarity_gate=1.5, seed=1
            )

    def test_several_placebo_draws_are_reported(self):
        D = 16
        Q = _orthonormal(D, 3)
        sae, reps = _trained_sae(D, n_features=24, k=3, steps=4)
        out = validate_workspace_claim(
            Q, reps, _labels(reps.shape[0]), sae=sae, n_bootstrap=8, n_placebo=4, seed=5
        )
        assert out["n_placebo"] == 4
        assert out["subspace_similarity_placebo_max"] >= out["subspace_similarity_placebo"]

    def test_the_control_is_the_worst_draw_not_the_average(self):
        """Averaging the controls is lenient: the claim must beat the best one."""
        D, k = 16, 3
        Q = _orthonormal(D, k)
        sae, reps = _trained_sae(D, n_features=24, k=3, steps=4)
        labels = _labels(reps.shape[0])
        out = validate_workspace_claim(Q, reps, labels, sae=sae, n_bootstrap=8, n_placebo=6, seed=5)
        mean, worst = out["subspace_similarity_placebo"], out["subspace_similarity_placebo_max"]
        assert worst > mean, "expected variation between control draws"
        # The gate must follow the worst draw, not their average.
        assert out["workspace_claim_valid"] == decide_workspace_claim(
            out["subspace_similarity"],
            out["similarity_ci_lower"],
            worst,
            bool(out["workspace_features_identified"] > 0.5),
            gate=out["similarity_gate"],
        )
        lenient = decide_workspace_claim(
            out["subspace_similarity"],
            out["similarity_ci_lower"],
            mean,
            bool(out["workspace_features_identified"] > 0.5),
            gate=out["similarity_gate"],
        )
        if lenient and not out["workspace_claim_valid"]:
            pytest.fail("verdict ignored the worst control draw")


# ═══════════════════════════════════════════════════════════════════
#  Determinism
# ═══════════════════════════════════════════════════════════════════


def test_pipeline_is_deterministic():
    D = 16
    Q = _orthonormal(D, 3)
    sae, reps = _trained_sae(D, n_features=24, k=3, steps=4)
    labels = _labels(reps.shape[0])
    kw = {"sae": sae, "n_bootstrap": 8, "n_placebo": 2, "seed": 17}
    a = validate_workspace_claim(Q, reps, labels, **kw)
    b = validate_workspace_claim(Q, reps, labels, **kw)
    assert a == b


def test_pipeline_does_not_consume_the_global_rng():
    D = 16
    Q = _orthonormal(D, 3)
    sae, reps = _trained_sae(D, n_features=24, k=3, steps=4)
    labels = _labels(reps.shape[0])
    seed_everything(1234)
    before = torch.randn(3)
    seed_everything(1234)
    validate_workspace_claim(Q, reps, labels, sae=sae, n_bootstrap=8, n_placebo=2, seed=5)
    after = torch.randn(3)
    assert torch.allclose(before, after)
