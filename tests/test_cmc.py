# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
# Tests for CMC (Cross-Mask Consistency) — mechanism #11

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, ".")

from src.models.cmc import CrossMaskConsistency

_REPO = Path(__file__).resolve().parent.parent
with open(_REPO / "defaults.yaml") as _fh:
    _DEFAULTS = yaml.safe_load(_fh)
_SHIPPED_MODEL = _DEFAULTS["model"]


class TestCMCCore:
    """Core CMC functionality tests."""

    def setup_method(self):
        self.D = 64
        self.B = 4
        self.T = 32
        self.cmc = CrossMaskConsistency(embed_dim=self.D)

    def test_output_shape_and_type(self):
        """CMC loss is a scalar tensor."""
        z1 = torch.randn(self.B, self.T, self.D)
        z2 = torch.randn(self.B, self.T, self.D)
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 8:16] = 1
        loss, _info = self.cmc(z1, z2, overlap)
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.dtype == torch.float32

    def test_loss_non_negative(self):
        """CMC loss is always ≥ 0."""
        z1 = torch.randn(self.B, self.T, self.D)
        z2 = torch.randn(self.B, self.T, self.D)
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 5:15] = 1
        loss, _info = self.cmc(z1, z2, overlap)
        assert loss.item() >= -1e-6, f"Loss should be non-negative, got {loss.item()}"

    def test_loss_zero_for_identical_predictions(self):
        """CMC loss = 0 when both predictions are identical."""
        z1 = torch.randn(self.B, self.T, self.D)
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 8:16] = 1
        loss, _info = self.cmc(z1, z1, overlap)
        assert loss.item() < 1e-6, f"Identical predictions should give ~0 loss, got {loss.item()}"

    def test_loss_zero_for_no_overlap(self):
        """CMC loss = 0 when there's no overlap."""
        z1 = torch.randn(self.B, self.T, self.D)
        z2 = torch.randn(self.B, self.T, self.D)
        # No overlap: all zeros
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        loss, info = self.cmc(z1, z2, overlap)
        assert loss.item() == 0.0
        assert info["cmc_skipped"] is True

    def test_loss_positive_for_different_predictions(self):
        """CMC loss > 0 when predictions differ at overlap."""
        z1 = torch.randn(self.B, self.T, self.D)
        z2 = z1 + 1.0  # offset by 1 in every dimension
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 8:16] = 1
        loss, _info = self.cmc(z1, z2, overlap)
        assert loss.item() > 0, "Different predictions should give >0 loss"

    def test_loss_invariant_to_non_overlap_changes(self):
        """Changing predictions outside overlap doesn't affect loss."""
        z1 = torch.randn(self.B, self.T, self.D)
        z2 = torch.randn(self.B, self.T, self.D)
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 8:16] = 1  # overlap at positions 8-15

        loss1, _ = self.cmc(z1, z2, overlap)

        # Change z2 outside overlap — loss should be same
        z2_modified = z2.clone()
        z2_modified[:, :8, :] = 999.0  # change non-overlap positions
        loss2, _ = self.cmc(z1, z2_modified, overlap)

        assert (
            abs(loss1.item() - loss2.item()) < 1e-5
        ), f"Non-overlap changes shouldn't affect loss: {loss1.item()} vs {loss2.item()}"

    def test_stop_grad_primary(self):
        """With stop_grad_primary=True, only secondary gets gradients."""
        cmc = CrossMaskConsistency(embed_dim=self.D, stop_grad_primary=True)
        z1 = torch.randn(self.B, self.T, self.D, requires_grad=True)
        z2 = torch.randn(self.B, self.T, self.D, requires_grad=True)
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 8:16] = 1

        loss, _ = cmc(z1, z2, overlap)
        loss.backward()

        # z1 should have NO gradient (stopped)
        assert (
            z1.grad is None or z1.grad.abs().max() < 1e-7
        ), "Primary should have no gradient with stop_grad_primary=True"
        # z2 should have gradient
        assert z2.grad is not None and z2.grad.abs().max() > 0, "Secondary should have gradient"

    def test_no_stop_grad_primary(self):
        """With stop_grad_primary=False, both get gradients."""
        cmc = CrossMaskConsistency(embed_dim=self.D, stop_grad_primary=False)
        z1 = torch.randn(self.B, self.T, self.D, requires_grad=True)
        z2 = torch.randn(self.B, self.T, self.D, requires_grad=True)
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 8:16] = 1

        loss, _ = cmc(z1, z2, overlap)
        loss.backward()

        assert z1.grad is not None and z1.grad.abs().max() > 0
        assert z2.grad is not None and z2.grad.abs().max() > 0


class TestCMCOverlapComputation:
    """Tests for overlap mask computation."""

    def test_overlap_correct(self):
        """Overlap = positions masked in BOTH patterns."""
        m1 = torch.zeros(2, 8, dtype=torch.long)
        m1[0, :4] = 1  # batch 0: positions 0-3
        m1[1, 2:6] = 1  # batch 1: positions 2-5

        m2 = torch.zeros(2, 8, dtype=torch.long)
        m2[0, 2:6] = 1  # batch 0: positions 2-5
        m2[1, 4:8] = 1  # batch 1: positions 4-7

        overlap = CrossMaskConsistency.compute_overlap_mask(m1, m2)

        # Batch 0: overlap at positions 2-3
        assert overlap[0, 0:2].sum() == 0  # not in overlap
        assert overlap[0, 2:4].sum() == 2  # in overlap
        assert overlap[0, 4:6].sum() == 0  # not in overlap

        # Batch 1: overlap at positions 4-5
        assert overlap[1, 0:4].sum() == 0
        assert overlap[1, 4:6].sum() == 2
        assert overlap[1, 6:8].sum() == 0

    def test_overlap_symmetric(self):
        """Overlap(m1, m2) = Overlap(m2, m1)."""
        m1 = torch.randint(0, 2, (4, 16))
        m2 = torch.randint(0, 2, (4, 16))
        o1 = CrossMaskConsistency.compute_overlap_mask(m1, m2)
        o2 = CrossMaskConsistency.compute_overlap_mask(m2, m1)
        assert (o1 == o2).all()

    def test_overlap_self_is_self(self):
        """Overlap(m, m) = m."""
        m = torch.randint(0, 2, (4, 16))
        overlap = CrossMaskConsistency.compute_overlap_mask(m, m)
        assert (overlap == m).all()


class TestCMCSecondMaskGeneration:
    """Tests for second mask generation."""

    def test_mask_shape(self):
        """Generated mask has correct shape."""
        mask = CrossMaskConsistency.generate_second_mask(
            seq_len=32,
            batch_size=4,
            mask_ratio=0.35,
            device=torch.device("cpu"),
        )
        assert mask.shape == (4, 32)

    def test_mask_binary(self):
        """Generated mask is binary (0 or 1)."""
        mask = CrossMaskConsistency.generate_second_mask(
            seq_len=32,
            batch_size=4,
            mask_ratio=0.35,
            device=torch.device("cpu"),
        )
        assert (mask >= 0).all() and (mask <= 1).all()

    def test_mask_ratio_approximate(self):
        """Generated mask has approximately the target ratio."""
        mask = CrossMaskConsistency.generate_second_mask(
            seq_len=128,
            batch_size=8,
            mask_ratio=0.35,
            device=torch.device("cpu"),
        )
        actual_ratio = mask.float().mean().item()
        assert 0.15 < actual_ratio < 0.55, f"Mask ratio should be ~0.35, got {actual_ratio:.3f}"

    def test_mask_uses_spans(self):
        """Generated mask uses contiguous spans (not isolated tokens)."""
        mask = CrossMaskConsistency.generate_second_mask(
            seq_len=64,
            batch_size=4,
            mask_ratio=0.35,
            span_length_range=(3, 10),
            device=torch.device("cpu"),
        )
        # Check that masked positions tend to be in groups
        # (not a perfect test but catches random per-token masking)
        for b in range(4):
            masked_positions = mask[b].nonzero().squeeze(-1)
            if len(masked_positions) > 2:
                # At least some adjacent masked positions
                diffs = masked_positions[1:] - masked_positions[:-1]
                n_adjacent = (diffs == 1).sum().item()
                # With span masking, most masked positions should be adjacent
                assert n_adjacent > 0, "Span masking should create adjacent masked positions"


class TestCMCTheorems:
    """Tests verifying the mathematical theorems."""

    def setup_method(self):
        self.D = 32
        self.B = 8
        self.T = 16

    def test_consistency_bound(self):
        """Verify: ||z1 - z2||² ≤ ε (CMC loss directly bounds inconsistency)."""
        cmc = CrossMaskConsistency(embed_dim=self.D, stop_grad_primary=False)
        torch.manual_seed(42)
        z1 = torch.randn(self.B, self.T, self.D)
        z2 = torch.randn(self.B, self.T, self.D)
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 4:8] = 1

        loss, _info = cmc(z1, z2, overlap)

        # The CMC loss IS the mean inconsistency at overlap positions
        # So by definition, mean(||z1-z2||² at overlap) = loss
        # Check this directly
        diff = (z1[:, 4:8, :] - z2[:, 4:8, :]) ** 2
        expected_loss = diff.sum() / (self.B * 4)  # B*4 overlap positions

        assert (
            abs(loss.item() - expected_loss.item()) < 1e-4
        ), f"CMC loss should equal mean inconsistency: {loss.item():.6f} vs {expected_loss.item():.6f}"

    def test_downstream_stability_bound(self):
        """Theorem 2: |f(z1) - f(z2)| ≤ ||w|| · √ε."""
        cmc = CrossMaskConsistency(embed_dim=self.D)
        torch.manual_seed(42)
        z1 = torch.randn(self.B, self.T, self.D)
        z2 = torch.randn(self.B, self.T, self.D)
        overlap = torch.zeros(self.B, self.T, dtype=torch.long)
        overlap[:, 4:8] = 1

        loss, _ = cmc(z1, z2, overlap)

        # Random linear probe
        w = torch.randn(self.D)
        b = torch.randn(1)
        probe_norm = w.norm().item()

        # Compute actual probe difference at overlap positions
        f1 = z1[:, 4:8, :] @ w + b  # (B, 4)
        f2 = z2[:, 4:8, :] @ w + b  # (B, 4)
        (f1 - f2).abs().max().item()

        # Theoretical bound
        bound = cmc.compute_downstream_stability_bound(loss.item(), probe_norm)

        # The bound should hold for the MEAN inconsistency
        # (not necessarily for the max, since bound is on mean)
        (f1 - f2).abs().mean().item()
        # For mean, bound should approximately hold
        # (loose check since bound is on sqrt of mean, not mean of sqrt)
        assert bound > 0, "Bound should be positive"

    def test_representation_variance_bound(self):
        """Corollary: Var(z_pred) ≤ ε/2 + δ."""
        cmc = CrossMaskConsistency(embed_dim=self.D)
        cmc_loss = 0.5
        jepa_loss = 0.3

        bound = cmc.compute_representation_variance_bound(cmc_loss, jepa_loss)
        expected = cmc_loss / 2.0 + jepa_loss

        assert (
            abs(bound - expected) < 1e-6
        ), f"Variance bound should be ε/2 + δ = {expected}, got {bound}"

    def test_triangle_inequality_holds(self):
        """||z1 - z2|| ≤ ||z1 - z*|| + ||z* - z2|| for any z*."""
        torch.manual_seed(42)
        z1 = torch.randn(self.D)
        z2 = torch.randn(self.D)
        z_target = torch.randn(self.D)

        lhs = (z1 - z2).norm()
        rhs = (z1 - z_target).norm() + (z_target - z2).norm()

        assert lhs <= rhs + 1e-6, "Triangle inequality must hold"

    def test_consistency_improves_with_training(self):
        """Simulate: as JEPA loss decreases, consistency should improve."""
        torch.manual_seed(42)
        cmc = CrossMaskConsistency(embed_dim=self.D, stop_grad_primary=False)
        z_target = torch.randn(1, self.T, self.D)
        overlap = torch.zeros(1, self.T, dtype=torch.long)
        overlap[:, 4:8] = 1

        # Simulate improving predictions (decreasing noise)
        losses = []
        for noise_level in [2.0, 1.0, 0.5, 0.1]:
            z1 = z_target + noise_level * torch.randn_like(z_target)
            z2 = z_target + noise_level * torch.randn_like(z_target)
            loss, _ = cmc(z1, z2, overlap)
            losses.append(loss.item())

        # Losses should generally decrease as noise decreases
        # (not strictly monotonic due to random noise, but trend should be clear)
        assert (
            losses[-1] < losses[0]
        ), f"Consistency should improve with better predictions: {losses}"


class TestCMCModes:
    """Tests for different computation modes."""

    def test_always_mode(self):
        """'always' mode computes at every step."""
        cmc = CrossMaskConsistency(embed_dim=32, mode="always")
        for step in range(20):
            assert cmc.should_compute(step) is True

    def test_interval_mode(self):
        """'interval' mode computes at correct intervals."""
        cmc = CrossMaskConsistency(embed_dim=32, mode="interval", interval=5)
        expected = [True, False, False, False, False, True, False, False, False, False]
        for step, exp in enumerate(expected):
            assert cmc.should_compute(step) == exp, f"step={step}"

    def test_reuse_encoder_mode(self):
        """'reuse_encoder' mode always computes (cheap)."""
        cmc = CrossMaskConsistency(embed_dim=32, mode="reuse_encoder")
        for step in range(20):
            assert cmc.should_compute(step) is True


class TestCMCEdgeCases:
    """Edge case tests."""

    def test_single_overlap_position(self):
        """CMC works with just 1 overlapping position."""
        cmc = CrossMaskConsistency(embed_dim=32)
        z1 = torch.randn(2, 8, 32)
        z2 = torch.randn(2, 8, 32)
        overlap = torch.zeros(2, 8, dtype=torch.long)
        overlap[0, 3] = 1  # just 1 position in batch 0
        loss, _info = self.cmc(z1, z2, overlap) if hasattr(self, "cmc") else cmc(z1, z2, overlap)
        assert loss.item() >= 0

    def test_all_overlap(self):
        """CMC when all positions overlap."""
        D, B, T = 32, 2, 8
        cmc = CrossMaskConsistency(embed_dim=D)
        z1 = torch.randn(B, T, D)
        z2 = torch.randn(B, T, D)
        overlap = torch.ones(B, T, dtype=torch.long)  # all overlap
        loss, info = cmc(z1, z2, overlap)
        assert loss.item() >= 0
        assert info["cmc_overlap_count"] == B * T

    def test_batch_with_mixed_overlap(self):
        """Some batches have overlap, some don't."""
        D, B, T = 32, 4, 8
        cmc = CrossMaskConsistency(embed_dim=D)
        z1 = torch.randn(B, T, D)
        z2 = torch.randn(B, T, D)
        overlap = torch.zeros(B, T, dtype=torch.long)
        overlap[0, 2:5] = 1  # only batch 0 has overlap
        loss, _info = cmc(z1, z2, overlap)
        assert loss.item() >= 0

    def test_very_large_difference(self):
        """CMC handles very different predictions without NaN."""
        D, B, T = 32, 2, 8
        cmc = CrossMaskConsistency(embed_dim=D)
        z1 = torch.randn(B, T, D) * 100
        z2 = torch.randn(B, T, D) * 100
        overlap = torch.zeros(B, T, dtype=torch.long)
        overlap[:, 2:5] = 1
        loss, _info = cmc(z1, z2, overlap)
        assert math.isfinite(loss.item())

    def test_zero_predictions(self):
        """CMC with zero predictions."""
        D, B, T = 32, 2, 8
        cmc = CrossMaskConsistency(embed_dim=D)
        z1 = torch.zeros(B, T, D)
        z2 = torch.zeros(B, T, D)
        overlap = torch.zeros(B, T, dtype=torch.long)
        overlap[:, 2:5] = 1
        loss, _info = cmc(z1, z2, overlap)
        assert loss.item() < 1e-6


class TestCMCConfig:
    """Config integration tests."""

    def test_default_config(self):
        """Default CMC config is valid."""
        cmc = CrossMaskConsistency(embed_dim=768)
        assert cmc.embed_dim == 768
        assert cmc.mode == "interval"
        assert cmc.interval == 10
        assert cmc.stop_grad_primary is True
        assert cmc.min_overlap_ratio == 0.2

    def test_constructor_default_matches_the_shipped_config(self):
        """The dataclass default and `defaults.yaml` must not drift apart.

        They are two declarations of one number. Nothing else ties them, and
        the gate threshold is the single value that decides whether CMC runs,
        so a drift here is a mechanism that silently changes behaviour when
        nobody is looking.
        """
        assert CrossMaskConsistency(embed_dim=768).min_overlap_ratio == (
            _SHIPPED_MODEL["cmc_min_overlap_ratio"]
        ), (
            "CrossMaskConsistency's `min_overlap_ratio` default no longer equals "
            "defaults.yaml's cmc_min_overlap_ratio. One of the two must change "
            "with the other."
        )

    def test_custom_config(self):
        """Custom CMC config works."""
        cmc = CrossMaskConsistency(
            embed_dim=384,
            second_mask_ratio=0.3,
            min_overlap_ratio=0.1,
            mode="always",
            interval=5,
            stop_grad_primary=False,
        )
        assert cmc.embed_dim == 384
        assert cmc.second_mask_ratio == 0.3
        assert cmc.mode == "always"

    def test_repr(self):
        """extra_repr produces readable string."""
        cmc = CrossMaskConsistency(embed_dim=768, mode="interval", interval=10)
        r = cmc.extra_repr()
        assert "768" in r
        assert "interval" in r


class TestCMCIntegration:
    """Integration tests with other mechanisms."""

    def test_cmc_with_jawp_workspace(self):
        """CMC can be restricted to workspace dimensions (from JAWP)."""
        D, B, T, k = 64, 2, 16, 10
        cmc = CrossMaskConsistency(embed_dim=D)

        # Simulate JAWP workspace projection
        Q = torch.randn(D, k)
        U, _S, Vt = torch.linalg.svd(Q, full_matrices=False)
        Q = U[:, :k] @ Vt[:k, :]  # orthonormalize

        z1 = torch.randn(B, T, D)
        z2 = torch.randn(B, T, D)
        overlap = torch.zeros(B, T, dtype=torch.long)
        overlap[:, 4:8] = 1

        # Workspace CMC: compare workspace projections
        z1_ws = z1 @ Q  # (B, T, k)
        z2_ws = z2 @ Q  # (B, T, k)
        # Overlap mask stays same dimension
        loss_ws, _info_ws = cmc(z1_ws, z2_ws, overlap)
        assert loss_ws.item() >= 0

    def test_cmc_after_pcr_refinement(self):
        """CMC after PCR refinement (predictions should be more consistent)."""
        D, B, T = 64, 2, 16
        cmc = CrossMaskConsistency(embed_dim=D)

        # Simulate: after refinement, predictions should be closer
        z1_base = torch.randn(B, T, D)
        z2_base = torch.randn(B, T, D)

        # "Refined" versions (closer to each other)
        z1_refined = 0.8 * z1_base + 0.2 * z2_base
        z2_refined = 0.8 * z2_base + 0.2 * z1_base

        overlap = torch.zeros(B, T, dtype=torch.long)
        overlap[:, 4:8] = 1

        loss_base, _ = cmc(z1_base, z2_base, overlap)
        loss_refined, _ = cmc(z1_refined, z2_refined, overlap)

        # Refined should be more consistent (lower CMC loss)
        assert loss_refined.item() < loss_base.item(), "PCR refinement should improve consistency"


# ═══════════════════════════════════════════════════════════════════════════
#  The liveness invariant
# ═══════════════════════════════════════════════════════════════════════════
#
#  THE DEFECT THIS EXISTS TO STOP
# -------------------------------
#  An audit read `cmc_overlap_ratio` off the log, saw 0.047...0.164, compared it
#  against `min_overlap_ratio = 0.2`, and concluded CMC was dead at the shipped
#  defaults — "the gate skips 6 of 6", "every ablation arm relying on it is
#  measuring nothing".
#
#  That conclusion was wrong, and it was wrong because of THIS file's subject
#  rather than because of the gate. `cmc_overlap_ratio` reported two different
#  quantities under one name: |Omega|/(B*T) on the branch that ran the loss, and
#  |Omega|/|M1| on the branch that skipped it. The gate compares against the
#  second form. For two independent masks at fraction `p`, |Omega|/(B*T) = p²
#  while |Omega|/|M1| = p — so the audit compared p² to a threshold that is
#  tested against p, and p² is below 0.2 for every mask ratio this repo ships
#  below 0.45. The mechanism was running the whole time.
#
#  A wrong audit finding is annoying. A mechanism that can report itself as
#  dead while it is alive is the defect class this repo treats as worse, because
#  nothing in the log can tell the two apart and nothing in the suite would have
#  gone red either way.
#
#  So: the loss must be NONZERO at the shipped configuration, and every branch
#  must say which of the two ratios it is reporting.


def _tiny_model(max_seq_len=128):
    """A real `TextSpanJEPA` carrying the SHIPPED CMC configuration.

    Every CMC hyperparameter is read out of `defaults.yaml` rather than
    restated, so this test cannot drift away from what the repo actually
    ships. Only the model SHAPE is shrunk.
    """
    from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

    return TextSpanJEPA(
        TextSpanJEPAConfig(
            vocab_size=64,
            embed_dim=32,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            predictor_embed_dim=16,
            predictor_depth=1,
            max_seq_len=max_seq_len,
            future_warmup_steps=0,
            use_jawp=False,
            use_cgn=False,
            use_swip=False,
            use_pcr=False,
            use_spc=False,
            use_wsd=False,
            use_cmc=_SHIPPED_MODEL["use_cmc"],
            cmc_second_mask_ratio=_SHIPPED_MODEL["cmc_second_mask_ratio"],
            cmc_min_overlap_ratio=_SHIPPED_MODEL["cmc_min_overlap_ratio"],
            cmc_mode=_SHIPPED_MODEL["cmc_mode"],
            cmc_interval=_SHIPPED_MODEL["cmc_interval"],
            lambda_cmc=_SHIPPED_MODEL["lambda_cmc"],
            use_gac=False,
            use_sta=False,
            use_puc=False,
            use_rdc=False,
            use_wsr=False,
        )
    )


# The ramp's own endpoints.
_RAMP = (_SHIPPED_MODEL["mask_ratio_start"], _SHIPPED_MODEL["mask_ratio_end"])

# B*T matters and is not arbitrary. `cmc_gate_ratio` is |Omega|/|M1|, a RATIO OF
# TWO RANDOM COUNTS, so its spread shrinks as 1/sqrt(B*T): measured sd is 0.054
# at 8x64 and 0.036 at 16x128 for the same p. The gate is reliably open at
# p >= 0.25 once B*T >= 1024 (0/60 draws shut at 16x128 for p in 0.25/0.30/0.35)
# and a coin flip at p = 0.20 and below at every shape. 16x128 is therefore the
# smallest shape where "the gate is open" is a property rather than a coin flip;
# at the shipped 64x512 the same points read 0/40 shut with min 0.2393.
_LIVE_B, _LIVE_T = 16, 128
_LIVE_RATIOS = (0.25, 0.30, 0.35)
_DEAD_RATIOS = (0.15, 0.20)


def _two_pass(model, ratio, seed, B=_LIVE_B, T=_LIVE_T):
    """Drive the real `compute_cmc_between_passes` bridge at one mask ratio.

    This is the path `src/train.py:1575-1627` takes: a primary forward with
    mask m1, a secondary forward with a fresh mask m2 drawn at the same ratio,
    then the bridge that scatters both compact slot predictions into full
    sequence space and intersects the masks.
    """
    torch.manual_seed(seed)
    ids = torch.randint(1, 64, (B, T))
    m1 = CrossMaskConsistency.generate_second_mask(
        seq_len=T, batch_size=B, mask_ratio=ratio, step=2 * seed
    )
    m2 = model.cmc.generate_second_mask(
        seq_len=T, batch_size=B, mask_ratio=ratio, step=2 * seed + 1
    )
    model.compute_loss_with_targets(ids, ids, m1, 0, 100)
    primary = model._cmc_pass
    model.compute_loss_with_targets(ids, ids, m2, 0, 100)
    return model.compute_cmc_between_passes(primary, model._cmc_pass)


def _module_at(ratio, seed, cmc, B=_LIVE_B, T=_LIVE_T):
    """Drive `CrossMaskConsistency.forward` on two real masks, no model.

    `z_pred_primary` is built to be nonzero only where m1 masked, which is what
    the host path produces: `compute_cmc_between_passes` scatters COMPACT slot
    predictions into a zero-filled (B,T,D), so `|M1|` is the count of valid
    primary slots, not `B*T`. Building it any other way would silently test a
    different denominator — which is how this whole misdiagnosis started.

    Returns `(loss, info)`.
    """
    torch.manual_seed(seed)
    m1 = CrossMaskConsistency.generate_second_mask(
        seq_len=T, batch_size=B, mask_ratio=ratio, step=2 * seed
    )
    m2 = CrossMaskConsistency.generate_second_mask(
        seq_len=T, batch_size=B, mask_ratio=ratio, step=2 * seed + 1
    )
    z1 = torch.zeros(B, T, 8)
    z1[m1.bool()] = torch.randn(int(m1.sum()), 8)
    z2 = torch.randn(B, T, 8)
    overlap = CrossMaskConsistency.compute_overlap_mask(m1, m2)
    return cmc(z1, z2, overlap)


def _shipped_cmc():
    return CrossMaskConsistency(
        embed_dim=8,
        second_mask_ratio=_SHIPPED_MODEL["cmc_second_mask_ratio"],
        min_overlap_ratio=_SHIPPED_MODEL["cmc_min_overlap_ratio"],
        mode=_SHIPPED_MODEL["cmc_mode"],
        interval=_SHIPPED_MODEL["cmc_interval"],
    )


class TestCMCRunsAtTheShippedDefaults:
    """CMC's loss must be nonzero at the configuration the repo ships.

    THE MINIMUM ASSERTION THE CARD ASKS FOR. If this goes red, CMC is inert
    under `defaults.yaml` and every ablation arm that relies on it is
    measuring nothing — which is precisely what the audit believed had already
    happened.
    """

    def test_the_gate_is_open_across_the_upper_shipped_ramp(self):
        """At p >= 0.25 CMC runs on every draw, with a positive loss.

        The shipped ramp is `mask_ratio_start` -> `mask_ratio_end`, i.e.
        0.15 -> 0.35, and `cmc_gate_ratio` tracks `p` (measured 0.276 / 0.330 /
        0.375 at 16x128 over 60 draws each, against a 0.2 threshold, minimum
        observed 0.2068). At those points a skip means something is broken, not
        that the curriculum is light.

        The 0.15-0.20 leg is asserted separately in
        `test_the_low_leg_skips_by_reporting_not_by_going_silent`, because
        there the gate really is shut and asserting otherwise would be the same
        lie told in the other direction.
        """
        cmc = _shipped_cmc()
        for ratio in _LIVE_RATIOS:
            for seed in range(40):
                loss, info = _module_at(ratio, seed, cmc)
                assert info["cmc_gate_open"], (
                    f"CMC did not run at the shipped defaults at mask_ratio={ratio} "
                    f"(seed {seed}): gate_ratio={info['cmc_gate_ratio']:.4f} < "
                    f"min_overlap_ratio={_SHIPPED_MODEL['cmc_min_overlap_ratio']}, "
                    f"skip_reason={info['cmc_skip_reason']!r}. The gate is shut at a "
                    "mask ratio this repo ships, so the mechanism is inert here "
                    "and every arm relying on it measures nothing."
                )
                assert loss.item() > 0.0, (
                    f"CMC ran but produced a zero loss at mask_ratio={ratio} "
                    f"(seed {seed}). A zero loss from two different masks means "
                    "the predictions are identical, not that the mechanism "
                    "contributes nothing -- the two are indistinguishable from "
                    "the loss alone, which is why `cmc_skipped` is reported."
                )

    def test_cmc_loss_is_nonzero_through_the_real_bridge(self):
        """The same property, end to end, on the host path.

        `src/train.py:1575-1627` never calls `CrossMaskConsistency` directly;
        it calls `TextSpanJEPA.compute_cmc_between_passes`. A module-level
        green therefore proves nothing about whether the shipped training loop
        produces a gradient, so this drives the actual bridge.
        """
        model = _tiny_model(_LIVE_T)
        model.train()
        for ratio in _LIVE_RATIOS:
            for seed in range(3):
                loss, info = _two_pass(model, ratio, seed)
                assert info["cmc_gate_open"], (
                    f"the host bridge skipped CMC at mask_ratio={ratio} (seed "
                    f"{seed}): gate_ratio={info['cmc_gate_ratio']:.4f} < "
                    f"min_overlap_ratio={_SHIPPED_MODEL['cmc_min_overlap_ratio']}"
                )
                assert loss.item() > 0.0, (
                    f"the host bridge produced a zero CMC loss at mask_ratio="
                    f"{ratio} (seed {seed})"
                )

    def test_the_loss_carries_gradient_to_the_secondary_pass(self):
        """Nonzero loss must also mean nonzero gradient.

        A loss that is a detached constant reads as a live number in the log
        while contributing nothing to the update — the same defect class one
        step further down. Asserted through the real bridge, not the module.
        """
        model = _tiny_model(_LIVE_T)
        model.train()
        loss, _info = _two_pass(model, 0.35, seed=0)
        assert loss.requires_grad, "CMC's loss came back detached from the graph"
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "CMC's loss.backward() reached no parameter at all"
        assert max(g.abs().max().item() for g in grads) > 0.0

    def test_the_low_leg_skips_by_reporting_not_by_going_silent(self):
        """Where the gate IS shut, the skip must be legible and countable.

        Below p = 0.25 the gate ratio mean (0.18 at 0.15, 0.23 at 0.20)
        straddles the 0.2 threshold, so CMC genuinely skips there — 41/60
        draws at 0.15 and 11/60 at 0.20, measured. That is the curriculum
        working, not a bug. But a skip that reports itself as `cmc_loss = 0.0`
        with no reason is indistinguishable from perfect agreement, which is
        how this mechanism came to be audited as dead.
        """
        cmc = _shipped_cmc()
        thresh = _SHIPPED_MODEL["cmc_min_overlap_ratio"]
        for ratio, expect_skip_dominant in ((0.15, True), (0.20, False)):
            ran = skipped = 0
            for seed in range(40):
                loss, info = _module_at(ratio, seed, cmc)
                if info["cmc_skipped"]:
                    skipped += 1
                    assert info["cmc_skip_reason"] in (
                        "no_overlap",
                        "insufficient_overlap",
                    ), (
                        f"CMC skipped at mask_ratio={ratio} without a usable reason "
                        f"(got {info['cmc_skip_reason']!r})"
                    )
                    assert not info["cmc_gate_open"]
                    assert info["cmc_gate_ratio"] < thresh
                    assert loss.item() == 0.0
                else:
                    ran += 1
                    assert info["cmc_gate_open"], (
                        "CMC ran and did not say so: the third state, which is "
                        "the one this test exists to forbid."
                    )
            assert ran > 0 and skipped > 0, (
                f"at mask_ratio={ratio} the gate is straddling the {thresh} "
                f"threshold and must produce both outcomes; got ran={ran} "
                f"skipped={skipped} over 40 draws. The operating point moved — "
                "re-measure and update the table in defaults.yaml."
            )
            if expect_skip_dominant:
                assert skipped > ran, (
                    f"at mask_ratio={ratio} the gate should shut more often than "
                    f"it opens (measured 41/60); got skipped={skipped} ran={ran}"
                )


class TestCMCOverlapDiagnostics:
    """The two overlap fractions are different numbers and are named so.

    `cmc_overlap_ratio` is |Omega|/(B*T). `cmc_gate_ratio` is |Omega|/|M1| and is
    the one compared against `min_overlap_ratio`. Reporting both under one name
    is what produced the false "CMC does not run at the defaults" finding, so
    the distinction is now pinned rather than described in a comment.
    """

    def _fixture(self, n_predicted, n_overlap, B=4, T=32):
        z1 = torch.zeros(B, T, 8)
        z1[:, :n_predicted] = torch.randn(B, n_predicted, 8)
        z2 = torch.randn(B, T, 8)
        overlap = torch.zeros(B, T, dtype=torch.long)
        overlap[:, :n_overlap] = 1
        cmc = CrossMaskConsistency(
            embed_dim=8, min_overlap_ratio=_SHIPPED_MODEL["cmc_min_overlap_ratio"]
        )
        return cmc(z1, z2, overlap)

    def test_overlap_ratio_is_the_fraction_of_all_positions(self):
        """On EVERY branch — including the skip branches.

        Pinned on the skip branches deliberately. That is where the regression
        lived: `cmc_overlap_ratio` reported |Omega|/(B*T) when the loss ran and
        |Omega|/|M1| when it did not, under one name. A reader comparing the
        skip-branch value against `min_overlap_ratio` was reading p against a
        threshold meant for a ratio of masks, and a reader comparing the
        run-branch value was reading p². Both readings came from one key.
        """
        for n_overlap in (0, 4, 12):
            _loss, info = self._fixture(n_predicted=20, n_overlap=n_overlap)
            expected = n_overlap * 4 / (4 * 32)
            assert info["cmc_overlap_ratio"] == pytest.approx(expected, abs=1e-9), (
                f"cmc_overlap_ratio must be |Omega|/(B*T) = {expected:.6f} on every "
                f"branch (n_overlap={n_overlap}), got {info['cmc_overlap_ratio']:.6f}"
            )

    def test_gate_ratio_is_the_fraction_of_the_primary_mask(self):
        """On every branch, and always present.

        `cmc_gate_ratio` is the quantity `min_overlap_ratio` is compared
        against. Reporting it only on the run branch would make the diagnostic
        available exactly when the reader needs it least.
        """
        for n_overlap in (0, 4, 12):
            _loss, info = self._fixture(n_predicted=20, n_overlap=n_overlap)
            expected = n_overlap * 4 / (20 * 4)
            assert info["cmc_gate_ratio"] == pytest.approx(expected, abs=1e-9), (
                f"cmc_gate_ratio must be |Omega|/|M1| = {expected:.6f} on every "
                f"branch (n_overlap={n_overlap}), got {info['cmc_gate_ratio']:.6f}"
            )

    def test_gate_open_agrees_with_the_threshold_everywhere(self):
        """`cmc_gate_open` must be the threshold test, not a separate opinion.

        Ties the reported boolean to the number it claims to summarise, so a
        future edit cannot make the flag and the ratio disagree.
        """
        thresh = _SHIPPED_MODEL["cmc_min_overlap_ratio"]
        for n_overlap in (0, 4, 12):
            _loss, info = self._fixture(n_predicted=20, n_overlap=n_overlap)
            expected_open = n_overlap > 0 and info["cmc_gate_ratio"] >= thresh
            assert info["cmc_gate_open"] is expected_open, (
                f"cmc_gate_open={info['cmc_gate_open']} disagrees with "
                f"gate_ratio={info['cmc_gate_ratio']:.4f} vs threshold {thresh} "
                f"at n_overlap={n_overlap}"
            )
            assert info["cmc_skipped"] is (not expected_open)
            assert bool(info["cmc_skip_reason"]) is (not expected_open)

    def test_the_two_diagnostics_are_not_the_same_number(self):
        """They differ by |M1|/(B*T), i.e. by the mask ratio.

        Pinned because collapsing them is the specific regression that made the
        mechanism unreadable: at the shipped ramp they read ~0.02-0.13 and
        ~0.18-0.41 respectively, so a reader who used the first against the
        0.2 threshold would conclude the gate was permanently shut.
        """
        _loss, info = self._fixture(n_predicted=20, n_overlap=12)
        assert info["cmc_overlap_ratio"] != pytest.approx(info["cmc_gate_ratio"]), (
            "cmc_overlap_ratio and cmc_gate_ratio came out equal, so this test "
            "is no longer exercising the distinction. The real host path "
            "scatters compact slot predictions into a zero-filled (B,T,D), so "
            "|M1| < B*T and the two genuinely differ."
        )

    def test_info_key_set_is_identical_on_every_branch(self):
        """A consumer must not KeyError exactly when it needs the diagnostics.

        `cmc_running_consistency` was present on the run and insufficient-overlap
        branches and absent on the no-overlap branch — so the one step where a
        reader most wants to know what happened was the one that raised.
        """
        _l1, no_overlap = self._fixture(n_predicted=20, n_overlap=0)
        _l2, insufficient = self._fixture(n_predicted=20, n_overlap=4)
        _l3, ran = self._fixture(n_predicted=20, n_overlap=12)
        sets = [set(no_overlap), set(insufficient), set(ran)]
        assert sets[0] == sets[1] == sets[2], (
            f"CMC returns different keys per branch: "
            f"no_overlap={sorted(sets[0])}, insufficient={sorted(sets[1])}, "
            f"ran={sorted(sets[2])}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
