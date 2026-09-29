# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Does the MLM baseline actually have the capacity the module claims?

Background
----------
`baselines/mlm_baseline.py` used to state a guarantee in its header: "identical
model capacity (encoder params match exactly), identical compute (same FLOPs per
forward pass), only the training objective differs". Both quantitative halves are
false, and the class docstring repeated it. The claim has been replaced in the
module by the measured figures; this file is what stops those figures from going
stale, and what stops the false guarantee from creeping back in.

Measured at the shipped 640/10 rung (`config/scaling/small_100m.yaml`,
`config/wikitext/mlm_wikitext_small.yaml`): V=50304, max_seq_len=512.

    one encoder, either arm                       81,758,720
    MLMBaseline.mlm_head, untied                  32,194,560
    MLMBaseline trainable == total               113,953,280
    TextSpanJEPA total                           170,706,561
    TextSpanJEPA trainable                        88,947,841

Why the test asserts a RATIO and not equality
---------------------------------------------
Parity is not a property this repo currently has, so a test demanding it would
be red on arrival. What can be pinned honestly is the *asymmetry*: its direction
(MLM larger on trainable capacity), its magnitude, and the mechanism that
produces it. If someone later ties the head, widens the predictor, or adopts a
published budget, these tests go red and the header has to be rewritten with
them -- which is the point. A test that asserted `ratio == 1.0` would have been
red when written and deleted rather than kept, which loses the measurement.

The tolerance below is deliberately TIGHT (0.5%) around the measured ratios. It
is not slack for a real refactor to hide in; it is float-comparison headroom, so
that an architecture change of any size registers as a failure.
"""

from __future__ import annotations

import pytest
import torch

# The rung the module header quotes, and the one
# config/wikitext/mlm_wikitext_small.yaml + config/scaling/small_100m.yaml share.
# NB the key names: the configs say `encoder_depth`, but `MLMBaseline.__init__`
# takes `depth` and swallows the rest in `**kwargs`. Passing `encoder_depth=10`
# here silently built a depth-12 encoder, which is how this file's first draft
# measured 1.392x instead of 1.281x. `src/train.py:476-484` maps the names
# explicitly, so the production path is correct.
SHIPPED = {"vocab_size": 50304, "max_seq_len": 512, "embed_dim": 640, "num_heads": 10}
SHIPPED_DEPTH = 10
# predictor_embed_dim / predictor_depth as declared in both matching configs.
SHIPPED_PREDICTOR = {"predictor_embed_dim": 320, "predictor_depth": 4}

#: Relative tolerance on the documented capacity ratios. 0.5% of 1.281 is
#: ~6.4M parameters -- far below any change a reviewer would care about, and far
#: above the float noise in an exact integer sum.
RATIO_TOL = 0.005

#: Documented at the shipped rung. Asserted as ratios, not as raw counts, so the
#: test pins the *relationship* the header claims rather than a magic integer.
#: (a future vocabulary change legitimately moves both counts; it must not be
#: able to move the RATIO without this going red.)
MLM_OVER_JEPA_TRAINABLE = 113_953_280 / 88_947_841  # 1.2811
JEPA_OVER_MLM_TOTAL = 170_706_561 / 113_953_280  # 1.4980


@pytest.fixture(scope="module")
def arms():
    """Build both arms once at the shipped rung. ~3.5s, ~1.1GB of parameters."""
    from baselines.mlm_baseline import MLMBaseline
    from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

    mlm = MLMBaseline(depth=SHIPPED_DEPTH, **SHIPPED)
    jepa = TextSpanJEPA(
        TextSpanJEPAConfig(encoder_depth=SHIPPED_DEPTH, **SHIPPED, **SHIPPED_PREDICTOR)
    )
    return mlm, jepa


def _n(mod):
    return sum(p.numel() for p in mod.parameters())


def _n_trainable(mod):
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# The one half of the old claim that is true: the encoder is shared
# ═══════════════════════════════════════════════════════════════════


class TestEncoderIsShared:
    def test_encoder_parameter_counts_are_identical(self, arms):
        """The parenthetical "encoder params match exactly" holds. Exactly."""
        mlm, jepa = arms
        assert _n(mlm.encoder) == _n(jepa.encoder)
        assert _n(mlm.encoder) == 81_758_720

    def test_encoder_shapes_match_elementwise(self, arms):
        """Not just the same count -- the same parameter shapes, name for name."""
        mlm, jepa = arms
        mlm_shapes = {n: tuple(p.shape) for n, p in mlm.encoder.named_parameters()}
        jepa_shapes = {n: tuple(p.shape) for n, p in jepa.encoder.named_parameters()}
        assert mlm_shapes == jepa_shapes


# ═══════════════════════════════════════════════════════════════════
# The half that is false: the arms are NOT capacity-matched
# ═══════════════════════════════════════════════════════════════════


class TestTrainableCapacityAsymmetry:
    def test_baseline_is_the_larger_trainable_model(self, arms):
        """Direction of the asymmetry, which is the reviewer-relevant part.

        The old header claimed parity. The truth is the control carries MORE
        trainable capacity than the method it is a control for.
        """
        mlm, jepa = arms
        assert _n_trainable(mlm) > _n_trainable(jepa)

    def test_documented_trainable_ratio_holds(self, arms):
        """Pin the 1.281x the module header quotes."""
        mlm, jepa = arms
        ratio = _n_trainable(mlm) / _n_trainable(jepa)
        assert ratio == pytest.approx(MLM_OVER_JEPA_TRAINABLE, rel=RATIO_TOL)

    def test_documented_total_ratio_holds(self, arms):
        """Pin the 1.498x the module header quotes, and its cause.

        JEPA's higher total is a frozen `target_encoder` deepcopy, not extra
        trainable capacity -- so the two ratios point in opposite directions and
        quoting only one of them would be misleading.
        """
        mlm, jepa = arms
        assert _n(jepa) / _n(mlm) == pytest.approx(JEPA_OVER_MLM_TOTAL, rel=RATIO_TOL)

    def test_every_baseline_parameter_is_trainable(self, arms):
        """The baseline has no frozen copy, so trainable == total for it."""
        mlm, _ = arms
        assert _n_trainable(mlm) == _n(mlm)
        assert mlm.get_num_params_trainable() == _n(mlm)

    def test_jepa_frozen_copy_carries_no_gradient(self, arms):
        """The 1.498x is not a capacity advantage -- this is why.

        Guards the reading the control-scout ruled out: a frozen EMA copy has no
        gradient path, so counting it as capacity would invert the finding.
        """
        _, jepa = arms
        assert all(not p.requires_grad for p in jepa.target_encoder.parameters())
        assert _n(jepa.target_encoder) == _n(jepa.encoder)

    def test_untied_head_is_the_entire_excess(self, arms):
        """The mechanism: the excess IS the head, to the parameter.

        The two encoders are the same class at the same dims, so they cancel and
        the whole trainable gap reduces to head - (JEPA's trainable non-encoder
        parameters). JEPA also spends 5,508,481 on a predictor and 1,639,680 on
        a tied decoder, which is why the gap is 25,005,439 rather than the head's
        full 32,194,560.
        """
        mlm, jepa = arms
        head = _n(mlm.mlm_head)
        jepa_non_encoder = _n_trainable(jepa) - _n(jepa.encoder)
        assert _n_trainable(mlm) - _n_trainable(jepa) == head - jepa_non_encoder
        assert head - jepa_non_encoder == 25_005_439

    def test_head_is_a_separate_untied_matrix(self, arms):
        """What JEPA does not have at all: a full (V, D) matrix of its own.

        `mlm_head` is a distinct Parameter from `encoder.token_embedding.weight`,
        not an alias of it. `MLMBaseline.decoder is MLMBaseline.mlm_head` is an
        attribute alias and must not be mistaken for weight tying.
        """
        mlm, _ = arms
        embed = mlm.encoder.token_embedding.weight
        assert mlm.mlm_head.weight is not embed
        assert mlm.mlm_head.weight.data_ptr() != embed.data_ptr()
        assert mlm.mlm_head.weight.shape == embed.shape
        assert mlm.decoder is mlm.mlm_head  # attribute alias, not a weight tie

    def test_jepa_decoder_has_no_vocab_matrix(self, arms):
        """JEPA's head is 1,639,680 and reuses the token embedding."""
        _, jepa = arms
        assert _n(jepa.decoder) == 1_639_680
        assert _n(jepa.decoder) < _n(jepa.encoder) // 40
        # No (V, D) parameter anywhere in the decoder -- the vocab projection is
        # `F.linear(x, token_embedding_weight)`, so it owns no vocab matrix.
        assert not any(
            tuple(p.shape) == (jepa.config.vocab_size, jepa.config.embed_dim)
            for p in jepa.decoder.parameters()
        )

    def test_asymmetry_is_a_property_of_the_architecture(self):
        """The identity holds at other shapes too, not just the shipped rung.

        Cheap shapes only. If this failed at a small shape the shipped-rung
        number would be an artefact of one config rather than a structural fact.
        """
        from baselines.mlm_baseline import MLMBaseline
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        for vocab, seq, dim, depth, heads, pdim, pdepth in [
            (1000, 32, 64, 2, 4, 32, 2),
            (50304, 512, 768, 12, 12, 384, 4),
        ]:
            mlm = MLMBaseline(
                vocab_size=vocab, max_seq_len=seq, embed_dim=dim, depth=depth, num_heads=heads
            )
            jepa = TextSpanJEPA(
                TextSpanJEPAConfig(
                    vocab_size=vocab,
                    max_seq_len=seq,
                    embed_dim=dim,
                    encoder_depth=depth,
                    num_heads=heads,
                    predictor_embed_dim=pdim,
                    predictor_depth=pdepth,
                )
            )
            assert _n(mlm.encoder) == _n(jepa.encoder)
            excess = _n_trainable(mlm) - _n_trainable(jepa)
            assert excess == _n(mlm.mlm_head) - (_n_trainable(jepa) - _n(jepa.encoder))


# ═══════════════════════════════════════════════════════════════════
# The computation half: not matched, and the asymmetry is measurable
# ═══════════════════════════════════════════════════════════════════


class TestComputeAsymmetry:
    def test_baseline_materialises_dense_logits_before_gather(self):
        """`compute_loss` projects all B*T positions, then gathers.

        This is the mechanical reason the two arms differ in activation memory,
        and it is a property of where the gather sits, not of the objective.
        Built tiny on purpose: the shipped rung is asserted by shape arithmetic in
        the next test, never allocated.
        """
        from baselines.mlm_baseline import MLMBaseline

        tiny = MLMBaseline(vocab_size=200, max_seq_len=8, embed_dim=16, depth=1, num_heads=4)
        ids = torch.randint(0, 200, (2, 8))
        mask = torch.zeros(2, 8, dtype=torch.long)
        mask[:, 2:4] = 1

        # forward() is the dense projection: (B, T, V), all 16 positions, of which
        # compute_loss then keeps 4.
        assert tiny(ids).shape == (2, 8, 200)
        assert int(mask.sum()) == 4
        loss, _ = tiny.compute_loss(ids, torch.randint(0, 200, (2, 8)), mask)
        assert loss.requires_grad

    def test_dense_logit_tensor_dwarfs_the_masked_one(self):
        """The 6.14 GiB vs 2.15 GiB figure the module header quotes.

        Arithmetic on shapes, not an allocation -- a real (64, 512, 50304) fp32
        tensor is 6.14 GiB and must not be built on a shared box.
        """
        b, seq, vocab = 64, 512, 50304
        for mask_ratio, expected_gib in [(0.15, 0.91), (0.35, 2.15)]:
            masked = b * int(seq * mask_ratio) * vocab * 4
            assert masked / 2**30 == pytest.approx(expected_gib, abs=0.01)
        dense = b * seq * vocab * 4
        assert dense / 2**30 == pytest.approx(6.14, abs=0.01)
        # MLM's own 0.15 budget is the harsher comparison, not the 0.35 one.
        assert dense / (b * int(seq * 0.15) * vocab * 4) == pytest.approx(6.737, rel=0.001)

    def test_gather_position_is_a_property_of_the_source(self):
        """The gather sits AFTER the head here and BEFORE it in the other arms.

        Pinned by source inspection because moving it is a behaviour change to a
        training path and this card deliberately does not make it; the test
        exists so that whoever does will see it break and update the header.
        """
        import inspect

        from baselines.data2vec_baseline import Data2VecTextBaseline
        from baselines.mlm_baseline import MLMBaseline

        mlm_src = inspect.getsource(MLMBaseline.forward)
        assert "mask_positions" not in mlm_src  # no mask available to gather on
        assert "self.mlm_head(h)" in mlm_src  # dense (B, T, V) projection

        d2v_src = inspect.getsource(Data2VecTextBaseline.forward)
        gather = d2v_src.index("h_online[masked_indices]")
        assert gather < d2v_src.index("self.regression_head(")  # gathers first

    def test_jepa_decodes_masked_rows_only(self):
        """JEPA's decode is masked-row-only, which is the other half of the 2.86x.

        Read from source rather than executed: a real forward at the shipped
        shape would allocate gigabytes.
        """
        import inspect

        from src.models import jepa as jepa_mod

        src = inspect.getsource(jepa_mod.TextSpanJEPA.compute_loss_with_targets)
        gather = src.index("h_online[mask_positions.bool()]")
        decode = src.index("self.decoder(h_at_masked")
        assert gather < decode
        assert "self.decoder(h_online" not in src  # never decodes the dense tensor


# ═══════════════════════════════════════════════════════════════════
# The claim itself must not come back
# ═══════════════════════════════════════════════════════════════════


#: Markers delimiting the verbatim quotation of the old guarantee inside the
#: module docstring. A test that forbids a live guarantee has to be able to tell
#: a quotation from an assertion, or it forbids the historical record too.
HIST_BEGIN = "--- BEGIN HISTORICAL CLAIM"
HIST_END = "--- END HISTORICAL CLAIM"


def _live_text() -> str:
    """Every word this module says, minus the delimited historical quotation.

    Includes the file header comments, the module docstring and the class
    docstring, since the false guarantee appeared in all three.
    """
    import baselines.mlm_baseline as mod

    with open(mod.__file__, encoding="utf-8") as fh:
        text = fh.read() + "\n" + (mod.__doc__ or "") + "\n" + (mod.MLMBaseline.__doc__ or "")
    while HIST_BEGIN in text and HIST_END in text:
        head, rest = text.split(HIST_BEGIN, 1)
        _, text = rest.split(HIST_END, 1)
        text = head + text
    assert HIST_BEGIN not in text, "unterminated historical block"
    return text.lower()


class TestFalseGuaranteeIsGone:
    @pytest.mark.parametrize(
        "phrase",
        [
            "identical model capacity",
            "identical compute",
            "same flops per forward pass",
            "only the training objective differs",
            "fair comparison guarantee",
        ],
    )
    def test_no_live_parity_guarantee(self, phrase):
        """The old guarantee may be quoted as history, never asserted.

        Scoped to live text by `HIST_BEGIN`/`HIST_END`, so restoring the
        guarantee as a live claim fails here even though the quotation of it
        stays in the docstring.
        """
        assert phrase not in _live_text(), (
            f"baselines/mlm_baseline.py asserts {phrase!r} again. The arms are not "
            f"capacity- or compute-matched; see the module header for the measured "
            f"figures."
        )

    def test_historical_quotation_is_still_delimited(self):
        """Guard the guard: the markers must exist and be balanced.

        Without this, someone could 'fix' a red no-live-guarantee test by
        widening the historical block to cover the whole file.
        """
        import baselines.mlm_baseline as mod

        with open(mod.__file__, encoding="utf-8") as fh:
            text = fh.read()
        assert text.count(HIST_BEGIN) == 1
        assert text.count(HIST_END) == 1
        assert text.index(HIST_BEGIN) < text.index(HIST_END)

    def test_class_docstring_carries_no_guarantee(self):
        """The claim was repeated in the class docstring; that copy is gone too."""
        from baselines.mlm_baseline import MLMBaseline

        doc = (MLMBaseline.__doc__ or "").lower()
        for phrase in ("identical model capacity", "identical compute", "fair comparison:"):
            assert phrase not in doc

    def test_module_states_the_measured_numbers(self):
        """The replacement claim is a measurement, so it carries the numbers."""
        import baselines.mlm_baseline as mod

        with open(mod.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for figure in ("81,758,720", "32,194,560", "113,953,280", "88,947,841", "1.281"):
            assert figure in text, f"header lost the measured figure {figure}"
