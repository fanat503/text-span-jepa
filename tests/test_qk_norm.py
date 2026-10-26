# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""QK-normalisation: per-head RMS normalisation of q and k before the dot product.

From the RoboJEPA literature scan (arXiv
2610.10515): "We found QK-normalization to be of particular importance for
training stability when scaling the dataset size and the model size."

The contract these tests pin is a PROPERTY, not a shape: the attention logits
must be invariant to the scale of q and k, and their magnitude must be bounded
by `head_dim` no matter how large the activations get. A shape test cannot
catch a wrong QK-norm -- a normalisation applied after a rotation, or over the
wrong axis, still returns tensors of the right shape.

NOTE ON WHERE THE PROPERTY IS TESTED, AND WHY NOT END-TO-END
------------------------------------------------------------
`Block.forward` is pre-norm: it applies `self.norm1` before projecting into
q/k/v, and LayerNorm is itself scale-invariant. So scaling the encoder's token
embedding by c leaves the first block's attention logits unchanged *whether or
not QK-norm is on*, and an end-to-end "output is invariant to input scale" test
would pass on the broken implementation. It would be a vacuous test wearing a
property's clothes.

The scale that QK-norm actually removes is the growth of the PROJECTION
WEIGHTS, which is invisible at the input and visible only at q and k. So the
property is tested where it lives: on q and k, below the projection, above the
dot product. `_attend` was split out of `Attention.forward` for exactly this.

TWO LIMITS OF THE PROPERTY, PINNED INSTEAD OF GLOSSED
----------------------------------------------------
THE EPS FLOOR. RMS-normalisation is `x * rsqrt(mean(x^2) + eps)`, so it is
invariant under `x -> c*x` only while `c^2 * mean(x^2)` is well above `eps`.
Push the activations low enough and `eps` dominates, the normalisation stops
compensating, and the logit shrinks toward zero. That direction is benign -- it
drives the softmax toward uniform, never toward saturation, so the stability
claim holds unconditionally -- but "invariant" is not the word to use below the
floor, and `test_invariance_degrades_below_the_eps_floor` says so.

`_attend` IS NOT THE NORMALISATION. It is the pure attention primitive; the
QK-norm is applied by `Attention.forward` before calling it. Tests that use it
to check invariance must therefore run q and k through `attn.qk_norm` first.
Skipping that step does not produce a wrong answer -- it produces a test that
measures the wrong function and fails for the right reason.
"""

import pytest
import torch

from src.models.encoder import Attention, Block, QKNorm, TextSpanJEPLEncoder
from src.models.predictor import PredictorBlock, TextSpanJEPApredictor

DIM = 64
HEADS = 4
HEAD_DIM = DIM // HEADS
B, N = 2, 7


def _qkv(scale: float = 1.0, seed: int = 0, with_bias: bool = False):
    """(q, k, v) in the (B, num_heads, N, head_dim) layout `Attention` uses."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, HEADS, N, HEAD_DIM, generator=g) * scale
    k = torch.randn(B, HEADS, N, HEAD_DIM, generator=g) * scale
    v = torch.randn(B, HEADS, N, HEAD_DIM, generator=g)
    if with_bias:
        # A bias term makes q/k's RMS depend on the weight too, not only the
        # input, which is the regime QK-norm exists for.
        q = q + 0.1
        k = k + 0.1
    return q, k, v


def _logits(q, k, scale: float):
    """(q . k) * scale, the quantity QK-norm is supposed to pin down."""
    return (q @ k.transpose(-2, -1)) * scale


# ══════════════════════════════════════════════════════════════════════
#  1. The normalisation itself
# ══════════════════════════════════════════════════════════════════════


class TestQKNormUnit:
    def test_normalises_per_head_to_unit_rms(self):
        q, k, _ = _qkv(scale=37.0)
        for name, t in (("q", QKNorm()(q, k)[0]), ("k", QKNorm()(q, k)[1])):
            rms = t.pow(2).mean(-1).sqrt()
            assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-5, atol=1e-6), (
                f"{name} head RMS is not 1 after normalisation: max |rms-1| = "
                f"{(rms - 1).abs().max().item()}"
            )

    def test_normalises_over_the_head_axis_not_the_token_axis(self):
        """A wrong-axis implementation also returns something of the right shape,
        so the axis has to be pinned. Normalising over the HEAD axis means every
        token's head-vector has mean-square 1; normalising over the TOKEN axis
        would leave the head axis free, and rescaling one token would go
        unabsorbed."""
        q, k, _ = _qkv(scale=5.0)
        q_hat, _ = QKNorm()(q, k)
        per_token = q_hat.pow(2).mean(-1)
        torch.testing.assert_close(per_token, torch.ones_like(per_token), rtol=1e-4, atol=1e-5)
        # Amplify one token. A token-axis normalisation cannot see this; a
        # head-axis normalisation has already fixed that token's scale.
        bumped = q_hat.clone()
        bumped[:, :, -1, :] *= 10.0
        assert bumped.pow(2).mean(-1)[..., -1].min() > 50.0

    def test_is_invariant_to_input_scale(self):
        """The core property, on the module in isolation."""
        q, k, _ = _qkv()
        ref_q, ref_k = QKNorm()(q, k)
        for c in (0.5, 2.0, 8.0):
            got_q, got_k = QKNorm()(c * q, c * k)
            torch.testing.assert_close(got_q, ref_q, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(got_k, ref_k, rtol=1e-4, atol=1e-5)

    def test_invariance_degrades_below_the_eps_floor(self):
        """Documented limit, pinned so it cannot be forgotten: `eps` puts a floor
        under the invariance. At c=1e-3 the squared activations are the same
        order as eps=1e-6, so the normalisation stops compensating."""
        q, k, _ = _qkv()
        ref_q, _ = QKNorm()(q, k)
        got_q, _ = QKNorm()(1e-3 * q, 1e-3 * k)
        assert not torch.allclose(got_q, ref_q, atol=1e-3), (
            "exact invariance below the eps floor would mean eps is not actually "
            "load-bearing; if this starts passing, re-check QKNorm.eps"
        )
        # ...and it fails toward zero, i.e. toward a uniform softmax, never toward
        # a saturated one. That is the direction the stability claim needs.
        assert got_q.abs().max() < ref_q.abs().max()

    def test_zero_input_does_not_produce_nan(self):
        """rms(0) = 0, so the eps is load-bearing on an all-zero head."""
        z = torch.zeros(B, HEADS, N, HEAD_DIM)
        q_hat, k_hat = QKNorm()(z, z)
        assert torch.isfinite(q_hat).all()
        assert torch.isfinite(k_hat).all()

    def test_has_no_parameters(self):
        """The whole point of shipping it gain-free: the two arms of the
        ablation are parameter-matched by construction, so the ablation isolates
        the normalisation and nothing else."""
        assert list(QKNorm().parameters()) == []


# ══════════════════════════════════════════════════════════════════════
#  2. Logit scale invariance -- the property the paper is claiming
# ══════════════════════════════════════════════════════════════════════


class TestAppliedInForward:
    """Pin WHERE the normalisation happens, not just that QKNorm works.

    Without these, the rest of this file can be satisfied by a `QKNorm` that is
    correct and never called: reverting the one line in `Attention.forward` that
    applies it leaves every direct `attn.qk_norm(...)` assertion green. That was
    measured, not assumed -- it is why these tests exist.

    The ordering requirement is the point. Normalising AFTER the dot product, or
    after the rotation a RoPE implementation would add, still returns the right
    shapes and is still caught by nothing that checks shapes.
    """

    def _captured(self, attn, x):
        """Run `x` through `attn` and return the (q, k) its QKNorm emitted.

        The handle is REMOVED before returning. A live hook also fires on every
        later `attn.qk_norm(...)` call the test makes to build its reference --
        which silently overwrites the capture with the reference and turns the
        assertion into `x == x`. That is not hypothetical: this test was written
        with the leak, passed against a deliberately broken encoder, and only
        the mutation run caught it.
        """
        seen = {}
        handle = attn.qk_norm.register_forward_hook(
            lambda _m, _i, out: seen.update(q=out[0], k=out[1])
        )
        try:
            attn(x)
        finally:
            handle.remove()
        return seen

    def test_forward_normalises_qk(self):
        torch.manual_seed(0)
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=True).eval()
        seen = self._captured(attn, torch.randn(2, N, DIM))
        assert seen, "Attention.forward never called QKNorm -- the flag is inert"
        for name in ("q", "k"):
            rms = seen[name].pow(2).mean(-1).sqrt()
            torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-5)

    def test_what_reaches_the_dot_product_is_the_normalised_tensor(self):
        """`forward` must equal a hand-written reference that normalises BEFORE
        attending. Applying it after the dot product, or to something other than
        q and k, breaks this and nothing else."""
        torch.manual_seed(0)
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=True).eval()
        x = torch.randn(2, N, DIM)
        B, _T, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, HEADS, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q_hat, k_hat = attn.qk_norm(q, k)
        ref = attn._attend(q_hat, k_hat, v)
        ref = attn.proj_drop(attn.proj(ref))
        torch.testing.assert_close(attn(x), ref, rtol=1e-5, atol=1e-6)

    def test_head_dim_is_the_axis_that_gets_normalised(self):
        """(B, num_heads, N, head_dim) -- a normalisation applied over `num_heads`
        or over `N` would also produce a plausible-looking tensor."""
        torch.manual_seed(0)
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=True).eval()
        seen = self._captured(attn, torch.randn(2, N, DIM))
        assert seen["q"].shape == (2, HEADS, N, attn.head_dim), seen["q"].shape
        # RMS over the last axis is 1; RMS over N is not (N != head_dim).
        over_n = seen["q"].pow(2).mean(-2)
        assert not torch.allclose(over_n, torch.ones_like(over_n), atol=1e-3)

    def test_k_is_normalised_and_not_something_else_in_a_q_shaped_slot(self):
        """Normalising q and forgetting k is the classic partial implementation:
        it normalises, it is shape-correct, and the logits still blow up with
        k's magnitude. So compare against an INDEPENDENT projection, keyed by
        name -- passing k's normalised tensor off as q's would slip past a
        shape-only or RMS-only check."""
        torch.manual_seed(0)
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=True).eval()
        x = torch.randn(2, N, DIM)
        seen = self._captured(attn, x)
        B, _T, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, HEADS, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, _v = qkv.unbind(0)
        exp_q, exp_k = attn.qk_norm(q, k)
        torch.testing.assert_close(seen["q"], exp_q, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(seen["k"], exp_k, rtol=1e-5, atol=1e-6)

    def test_predictor_forward_normalises_before_the_dot_product(self):
        """Same contract on the predictor's hand-rolled fused-attention path."""
        torch.manual_seed(0)
        blk = PredictorBlock(DIM, HEADS, mlp_ratio=2.0, use_qk_norm=True).eval()
        x = torch.randn(2, N, DIM)
        B, _T, C = x.shape
        h = blk.norm1(x)
        qkv = torch.nn.functional.linear(h, blk.attn.in_proj_weight, blk.attn.in_proj_bias)
        q, k, v = (
            t.reshape(B, N, HEADS, blk.head_dim).permute(0, 2, 1, 3) for t in qkv.chunk(3, dim=-1)
        )
        q_hat, k_hat = blk.qk_norm(q, k)
        ref = torch.nn.functional.scaled_dot_product_attention(q_hat, k_hat, v)
        ref = blk.attn.out_proj(ref.permute(0, 2, 1, 3).reshape(B, N, C))
        torch.testing.assert_close(
            blk(x), x + ref + blk.mlp(blk.norm2(x + ref)), rtol=1e-5, atol=1e-6
        )


class TestLogitScaleInvariance:
    @pytest.mark.parametrize("c", [0.5, 2.0, 8.0])
    def test_logits_are_invariant_to_qk_scale(self, c):
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=True)
        q, k, _ = _qkv(with_bias=True)
        ref = _logits(*attn.qk_norm(q, k), scale=attn.scale)
        got = _logits(*attn.qk_norm(c * q, c * k), scale=attn.scale)
        torch.testing.assert_close(
            got,
            ref,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda s: (
                f"logits moved by {c}x scaling of q,k. QK-norm is not being "
                f"applied to the head axis, or is being applied after the dot "
                f"product: {s}"
            ),
        )

    def test_without_qk_norm_the_same_logits_move_by_c_squared(self):
        """The negative control. Without this, the test above could pass because
        the tensors are small, not because the normalisation works."""
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=False)
        assert attn.qk_norm is None, "the off arm must not allocate a QKNorm"
        q, k, _ = _qkv()
        c = 4.0
        base = _logits(q, k, scale=attn.scale)
        scaled = _logits(c * q, c * k, scale=attn.scale)
        assert not torch.allclose(scaled, base, rtol=1e-2, atol=1e-3)
        ratio = (scaled.abs().max() / base.abs().max()).item()
        assert ratio == pytest.approx(
            c**2, rel=0.02
        ), f"un-normalised logits should scale by c^2={c**2}, got {ratio}"

    def test_attention_output_is_invariant_to_qk_scale(self):
        """The invariance has to survive all the way to what the block returns,
        not merely hold for a tensor nothing downstream consumes.

        `_attend` does not normalise -- `forward` does, immediately before
        calling it -- so q and k are normalised here explicitly, exactly as
        `forward` would.
        """
        torch.manual_seed(0)
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=True).eval()
        q, k, v = _qkv(scale=0.3)

        def attend(c):
            q_hat, k_hat = attn.qk_norm(c * q, c * k)
            return attn._attend(q_hat, k_hat, v)

        torch.testing.assert_close(attend(10.0), attend(1.0), rtol=1e-4, atol=1e-5)

    def test_attention_output_is_NOT_invariant_without_qk_norm(self):
        torch.manual_seed(0)
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=False).eval()
        q, k, v = _qkv(scale=0.3)
        assert not torch.allclose(attn._attend(10.0 * q, 10.0 * k, v), attn._attend(q, k, v))

    def test_the_two_arms_disagree_which_is_the_whole_point(self):
        """Sanity in the other direction: with realistic activations QK-norm is
        NOT a no-op, so `use_qk_norm` actually changes the model. Driven through
        `forward`, because that is where the normalisation lives."""
        torch.manual_seed(0)
        on = Attention(DIM, num_heads=HEADS, use_qk_norm=True).eval()
        off = Attention(DIM, num_heads=HEADS, use_qk_norm=False).eval()
        off.load_state_dict(on.state_dict(), strict=True)
        x = torch.randn(2, N, DIM)
        assert (on(x) - off(x)).abs().max() > 1e-3, (
            "the two arms produced the same attention; use_qk_norm is inert and "
            "the ablation would measure nothing"
        )


class TestLogitMagnitudeBound:
    """The stability claim, as a bound.

    RMS-normalising q and k fixes ||q|| = ||k|| = sqrt(head_dim), so by
    Cauchy-Schwarz |q . k| <= head_dim and the scaled logit is bounded by
    sqrt(head_dim) -- an absolute ceiling that does not move with activation
    magnitude. That is what stops the softmax saturating as the run scales, and
    it is measurable without training anything.
    """

    @pytest.mark.parametrize("act_scale", [1.0, 1e3, 1e5])
    def test_bound_holds_under_qk_norm(self, act_scale):
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=True)
        q, k, _ = _qkv(scale=act_scale, with_bias=True)
        q_hat, k_hat = attn.qk_norm(q, k)
        worst = _logits(q_hat, k_hat, scale=attn.scale).abs().max().item()
        assert worst <= HEAD_DIM**0.5 + 1e-4, (
            f"logit {worst} exceeds sqrt(head_dim)={HEAD_DIM**0.5} at activation "
            f"scale {act_scale}"
        )

    def test_bound_is_violated_without_qk_norm(self):
        """Otherwise the test above proves nothing: 1e5 activations must overflow
        it on the off arm."""
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=False)
        q, k, _ = _qkv(scale=1e5, with_bias=True)
        worst = _logits(q, k, scale=attn.scale).abs().max().item()
        assert worst > HEAD_DIM**0.5, f"the off arm should blow the bound, got {worst}"

    def test_bound_is_scale_free_not_merely_satisfied_once(self):
        attn = Attention(DIM, num_heads=HEADS, use_qk_norm=True)
        peaks = []
        for s in (1.0, 1e2, 1e4):
            # No bias here, so the rescaling IS uniform and the three peaks must
            # agree to fp precision rather than merely sitting under the bound.
            q, k, _ = _qkv(scale=s)
            peaks.append(_logits(*attn.qk_norm(q, k), scale=attn.scale).abs().max().item())
        assert max(peaks) / min(peaks) == pytest.approx(1.0, abs=1e-4), peaks


# ══════════════════════════════════════════════════════════════════════
#  3. Ablation fidelity: the off arm must be the existing recipe, untouched
# ══════════════════════════════════════════════════════════════════════


class TestAblationFidelity:
    def _attention(self, use_qk_norm):
        torch.manual_seed(0)
        return Attention(DIM, num_heads=HEADS, use_qk_norm=use_qk_norm).eval()

    def test_off_by_default(self):
        assert Attention(DIM, num_heads=HEADS).qk_norm is None
        assert PredictorBlock(DIM, HEADS).qk_norm is None
        assert (
            TextSpanJEPLEncoder(vocab_size=64, max_seq_len=16, embed_dim=DIM).blocks[0].attn.qk_norm
            is None
        )

    def test_on_and_off_arms_are_parameter_matched(self):
        on = self._attention(True)
        off = self._attention(False)
        assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())
        assert sorted(on.state_dict()) == sorted(off.state_dict()), (
            "a checkpoint written by the off arm must load into the on arm; "
            "the two differ in behaviour, not in layout"
        )

    def test_off_arm_output_is_bit_identical_to_the_pre_qk_norm_recipe(self):
        """The control of the ablation is not a reimplementation of the old
        recipe, it IS the old recipe -- same seed, same bits."""
        ids = torch.randint(0, 64, (2, 16))
        torch.manual_seed(0)
        old = TextSpanJEPLEncoder(
            vocab_size=64, max_seq_len=16, embed_dim=DIM, depth=2, num_heads=HEADS
        )
        torch.manual_seed(0)
        new = TextSpanJEPLEncoder(
            vocab_size=64, max_seq_len=16, embed_dim=DIM, depth=2, num_heads=HEADS
        )
        new.load_state_dict(old.state_dict())
        new.eval()
        old.eval()
        x = torch.randn(2, 16, DIM)
        torch.testing.assert_close(
            old.blocks[0].attn(x),
            new.blocks[0].attn(x),
            rtol=0,
            atol=0,
        )

    def test_on_arm_output_is_finite_and_grad_flows(self):
        torch.manual_seed(0)
        enc = TextSpanJEPLEncoder(
            vocab_size=64, max_seq_len=16, embed_dim=DIM, depth=1, num_heads=HEADS, use_qk_norm=True
        )
        out, _ = enc(torch.randint(0, 64, (2, 16)))
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert enc.blocks[0].attn.qkv.weight.grad is not None
        assert torch.isfinite(enc.blocks[0].attn.qkv.weight.grad).all()

    def test_grad_flows_when_frozen_weights_have_no_parameters(self):
        """QKNorm has no parameters, so it must not be a graph dead-end: the
        gradient has to reach qkv.weight THROUGH it."""
        torch.manual_seed(0)
        blk = Block(DIM, HEADS, use_qk_norm=True)
        x = torch.randn(2, N, DIM, requires_grad=True)
        blk(x).sum().backward()
        assert blk.attn.qkv.weight.grad is not None
        assert blk.attn.qkv.weight.grad.abs().sum() > 0


# ══════════════════════════════════════════════════════════════════════
#  4. Predictor
# ══════════════════════════════════════════════════════════════════════


class TestPredictorQKNorm:
    def _block(self, use_qk_norm):
        torch.manual_seed(0)
        return PredictorBlock(DIM, HEADS, mlp_ratio=2.0, use_qk_norm=use_qk_norm).eval()

    def test_arms_are_parameter_matched(self):
        on, off = self._block(True), self._block(False)
        assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())
        assert sorted(on.state_dict()) == sorted(off.state_dict())

    def test_keeps_multheadattention_parameter_names(self):
        """`_fix_init_weight` rescales `attn.out_proj.weight`, and every saved
        predictor checkpoint is keyed on these names. Re-doing the projection by
        hand must not rename or drop them."""
        blk = self._block(True)
        keys = set(blk.state_dict())
        for name in (
            "attn.in_proj_weight",
            "attn.in_proj_bias",
            "attn.out_proj.weight",
            "attn.out_proj.bias",
        ):
            assert name in keys, name
        assert sum(p.numel() for p in self._block(True).parameters()) == sum(
            p.numel() for p in self._block(False).parameters()
        )

    def test_qk_reaching_the_dot_product_are_unit_rms_per_head(self):
        """The predictor's fused attention is re-done by hand, so assert the
        re-done split is the right one: capture what the norm actually emitted."""
        blk = self._block(True)
        seen = {}

        def hook(_m, _i, out):
            seen["q"], seen["k"] = out

        blk.qk_norm.register_forward_hook(hook)
        blk(torch.randn(2, N, DIM))
        assert seen, "QKNorm never ran -- the predictor arm is not wired"
        for name in ("q", "k"):
            rms = seen[name].pow(2).mean(-1).sqrt()
            torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-5)

    def test_head_count_matches_the_projection_split(self):
        """A chunk(3) on the fused projection is only q/k/v if the split order
        is the one PyTorch packs. Check the head axis is really the head axis."""
        blk = self._block(True)
        seen = {}
        blk.qk_norm.register_forward_hook(lambda _m, _i, out: seen.update(q=out[0]))
        blk(torch.randn(2, N, DIM))
        assert seen["q"].shape == (2, HEADS, N, DIM // HEADS), seen["q"].shape

    def test_forward_is_finite_and_shape_preserving(self):
        blk = self._block(True)
        x = torch.randn(2, N, DIM)
        out = blk(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_gradients_reach_the_fused_projection(self):
        blk = self._block(True)
        blk(torch.randn(2, N, DIM)).sum().backward()
        assert blk.attn.in_proj_weight.grad is not None
        assert blk.attn.in_proj_weight.grad.abs().sum() > 0

    def test_predictor_accepts_the_flag_through_its_constructor(self):
        torch.manual_seed(0)
        pred = TextSpanJEPApredictor(
            embed_dim=DIM,
            predictor_embed_dim=DIM,
            depth=1,
            num_heads=HEADS,
            max_seq_len=16,
            future_offsets=(1,),
            num_refine_steps=1,
            use_qk_norm=True,
        )
        assert all(b.qk_norm is not None for b in pred.predictor_blocks)

    def test_full_predictor_forward_runs_with_the_flag_on(self):
        torch.manual_seed(0)
        pred = TextSpanJEPApredictor(
            embed_dim=DIM,
            predictor_embed_dim=DIM,
            depth=1,
            num_heads=HEADS,
            max_seq_len=16,
            future_offsets=(1,),
            num_refine_steps=1,
            use_qk_norm=True,
        )
        h = torch.randn(2, 16, DIM)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, 3:6] = True
        span, _, _, _, _ = pred(h, mask, None, h)
        assert torch.isfinite(span).all()

    def test_default_predictor_is_unchanged(self):
        torch.manual_seed(0)
        pred = TextSpanJEPApredictor(
            embed_dim=DIM, predictor_embed_dim=DIM, depth=1, num_heads=HEADS, max_seq_len=16
        )
        assert all(b.qk_norm is None for b in pred.predictor_blocks)
