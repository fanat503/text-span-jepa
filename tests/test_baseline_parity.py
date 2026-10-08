# Copyright 2026 Slyatski Ilya
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

import contextlib
import re

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


# ═══════════════════════════════════════════════════════════════════
# All three arms must log the same KIND of parameter count
# ═══════════════════════════════════════════════════════════════════


@contextlib.contextmanager
def _meta_device():
    """Build on meta tensors: exact parameter shapes, zero allocation.

    Same trick as `tests/test_config_system.py::_meta_device`. The encoder calls
    `torch.linspace(...).item()`, which meta tensors cannot service, so
    `linspace` is pinned to CPU for the duration. Meta is what makes the second
    and third shapes below affordable at all -- the wide one is a 0.5GB
    allocation per arm otherwise, on a shared box.
    """
    original = torch.linspace

    def cpu_linspace(*a, **k):
        k = dict(k)
        k.setdefault("device", "cpu")
        return original(*a, **k)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            yield
    finally:
        torch.linspace = original


#: The three arms `src/train.py` knows how to build. This is the whole comparison
#: surface: `create_model` returns one of exactly these, into one log directory.
ARMS = ("text_span_jepa", "mlm", "data2vec")

#: (vocab_size, max_seq_len, embed_dim, encoder_depth, num_heads). Three shapes,
#: deliberately unlike each other, because one shape can be an accident:
#:
#:   tiny  -- vocab dominated by the head, depth 2, so mechanisms are a rounding
#:            error and the arms nearly agree;
#:   mid   -- the shipped mini config's proportions with a small vocab, so the
#:            MLM head stops dominating;
#:   wide  -- V=50304 / D=768 / depth 12, the shape where the three logged
#:            numbers were 262,021,633 / 123,689,472 / 85,646,592, i.e. 3.1x
#:            apart while all three were 2x-3x wrong about the same quantity.
SHAPES = [
    (200, 16, 32, 2, 4),
    (2000, 64, 96, 3, 6),
    (50304, 512, 768, 12, 12),
]
SHAPE_IDS = ["tiny", "mid", "wide"]


def _model_cfg(dim, depth, heads):
    """A model config that is valid for all three arms.

    Keys the baselines ignore are absorbed by their `**kwargs`, so one dict
    builds any arm. `create_model` is the production path: it is what
    `src/train.py` calls, so going through it tests the wiring and not just the
    classes.
    """
    return {
        "embed_dim": dim,
        "encoder_depth": depth,
        "num_heads": heads,
        "mlp_ratio": 2.0,
        "predictor_embed_dim": dim // 2,
        "predictor_depth": 2,
        "future_offsets": (1, 2),
        "num_refine_steps": 1,
        "jspace_k_workspace": 2,
    }


def _build_arm(arm, vocab, seq, dim, depth, heads):
    """One arm at one shape, on meta tensors."""
    from src.train import create_model

    with _meta_device():
        return create_model(arm, _model_cfg(dim, depth, heads), vocab, seq, device="meta")


def _build_all(vocab, seq, dim, depth, heads):
    return {arm: _build_arm(arm, vocab, seq, dim, depth, heads) for arm in ARMS}


def _independent_total(mod) -> int:
    """Recount the module's parameters without `mod.parameters()`.

    An independent walk, so an assertion built on it can catch a
    `get_num_params()` that reflects whatever its author happened to walk
    rather than the model. `nn.Module.parameters()` deduplicates shared tensors;
    so does this, via `id()`.
    """
    seen: set = set()
    total = 0
    for _, child in mod.named_modules():
        for p in child.parameters(recurse=False):
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def _independent_trainable(mod) -> int:
    return _independent_total(mod) - sum(
        p.numel() for m in mod.modules() for p in m.parameters(recurse=False) if not p.requires_grad
    )


class TestLoggedQuantityIsOneKind:
    """The three arms must log the same KIND of number: the model's size.

    What broke
    ----------
    `src/train.py` logs one line, `Model parameters (get_num_params())`, for
    whichever arm it built. `TextSpanJEPA.get_num_params()` defaults to the
    whole model. `MLMBaseline` and `Data2VecTextBaseline` defaulted to
    `non_embedding=True`, each returning its encoder-minus-embeddings plus its
    own head -- and data2vec's version also dropped the target encoder, the
    single largest tensor in that arm. Three arms in one comparison directory
    therefore logged three different quantities, 3.1x apart at the wide shape.

    Before that, all three were wrong in the same direction, so a comparison was
    at least consistently wrong. Inconsistently wrong is the harder state to
    catch and the easier one to publish, which is why this is a guard and not a
    comment.

    What KIND means here
    -------------------
    The quantity is *the number of scalar parameter tensors the module owns* --
    what is allocated, optimised around and written to the checkpoint. That is
    one definition, written down once, here. Every arm must satisfy it.

    What this class deliberately does NOT assert
    --------------------------------------------
    It asserts nothing about the three numbers being close, or ordered, or
    within any factor of each other. They are three different architectures; a
    closeness assertion would be the wrong test, it would be red for a legitimate
    refactor, and it would still pass under a mutation that shifted all three by
    the same wrong amount. `test_the_numbers_really_do_differ` below is what
    keeps this honest in the other direction: if a future change made the three
    counts coincide, that is a change of architecture, and this class should
    fail loudly rather than pass silently.

    At three shapes, not one
    -----------------------
    A single shape can be satisfied by accident: the defect it exists to catch
    depends on which tensors dominate, and the wide shape is the one where the
    three counts spread out. `tiny` has the head dominating, `mid` sits between.
    One of the three is a coincidence; a property that holds at all three is a
    property of the definitions.
    """

    # ---- the KIND itself, per arm, at every shape ----

    @pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
    def test_logged_count_is_the_models_own_parameter_tally(self, shape):
        """Each arm's bare `get_num_params()` is the sum over its parameters.

        Recounted independently of the method body. For data2vec this is also
        the test that the frozen target encoder is inside the number: it is the
        arm's largest tensor, and dropping it is exactly what the old default
        did.
        """
        for arm, mod in _build_all(*shape).items():
            reported = mod.get_num_params()
            assert reported == _independent_total(mod), (
                f"{arm}.get_num_params() returned {reported}, but the module holds "
                f"{_independent_total(mod)} parameters. train.py logs this number "
                f"as 'Model parameters', so it must be the whole model."
            )

    @pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
    def test_all_three_arms_satisfy_one_definition_at_one_shape(self, shape):
        """The headline case, stated as the card states it.

        One shape, all three arms, one definition. Printed as a table on failure
        because the failure mode is exactly a reader comparing three numbers and
        seeing three different meanings.
        """
        arms = _build_all(*shape)
        kinds = {arm: (mod.get_num_params(), _independent_total(mod)) for arm, mod in arms.items()}
        mismatched = {a: (r, t) for a, (r, t) in kinds.items() if r != t}
        assert not mismatched, (
            "these arms log a quantity that is not the model's parameter count "
            f"(logged, actual): {mismatched}. Three different architectures may "
            "legitimately have different counts; they may not report different "
            "KINDINGS of count."
        )
        assert len(ARMS) == 3

    # ---- what KIND is not, so the flag cannot quietly redefine the default ----

    @pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
    def test_logged_count_is_not_the_non_embedding_variant(self, shape):
        """The bare call must not be the `non_embedding=True` subtraction.

        Every arm owns at least one embedding table here, so this separates the
        two conventions. It is the assertion that fails if a default flips back
        to `True` -- the shape of the original defect -- while leaving a module
        that correctly reports the total untouched.
        """
        for arm, mod in _build_all(*shape).items():
            assert mod.get_num_params() > mod.get_num_params(non_embedding=True), (
                f"{arm}.get_num_params() equals its non_embedding count, so the "
                "default is reporting the convention, not the model"
            )

    @pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
    def test_non_embedding_variant_is_the_total_minus_the_embedding_tables(self, shape):
        """The flag stays available and keeps its documented meaning.

        Both encoders' tables for the arms that hold a frozen copy, one for MLM
        (it has no second encoder). Mirrors
        `TestParamCountReporting::test_non_embedding_variant_drops_both_embedding_tables`
        for JEPA, which this file cannot see.
        """
        for arm, mod in _build_all(*shape).items():
            # One encoder for the MLM arm, two for the arms that keep a frozen
            # copy. Counted rather than assumed, so adding or removing a teacher
            # does not silently move the expected subtraction.
            encoders = [mod.encoder]
            target = getattr(mod, "target_encoder", None)
            if target is not None:
                encoders.append(target)
            per_enc = sum(
                enc.token_embedding.weight.numel() + enc.pos_embedding.numel() for enc in encoders
            )
            assert per_enc > 0
            total = mod.get_num_params()
            assert mod.get_num_params(non_embedding=True) == total - per_enc, (
                f"{arm}: non_embedding=True must drop the token and position "
                f"embeddings of all {len(encoders)} encoder(s) it holds"
            )

    # ---- the API the trainer's second log line depends on ----

    def test_every_arm_exposes_the_trainable_count(self):
        """`Data2VecTextBaseline` had no `get_num_params_trainable()` at all.

        So of the three arms, one reported trainable capacity and two did not:
        the startup log's second line was not available for every arm it prints
        beside. All three now answer the same question the same way, which is
        what a comparison needs -- different architectures, same definition.
        """
        shape = SHAPES[0]
        for arm, mod in _build_all(*shape).items():
            assert hasattr(mod, "get_num_params_trainable"), (
                f"{arm} has no get_num_params_trainable(); train.py prints the "
                "trainable line for every arm, so every arm must be able to "
                "report it"
            )
            assert mod.get_num_params_trainable() == _independent_trainable(mod), (
                f"{arm}.get_num_params_trainable() is not the parameter count of "
                "the module's gradient-carrying parameters"
            )

    def test_trainable_count_is_below_the_logged_count_where_a_teacher_is_frozen(self):
        """The two lines mean different things, on purpose.

        JEPA and data2vec both hold a frozen copy, so trainable < total there.
        MLM has no frozen tensor, so the two coincide for it -- and that
        coincidence is exactly why the counts must not be compared blind.
        """
        shape = SHAPES[1]
        arms = _build_all(*shape)
        for arm in ("text_span_jepa", "data2vec"):
            mod = arms[arm]
            assert mod.get_num_params_trainable() < mod.get_num_params()
        assert arms["mlm"].get_num_params_trainable() == arms["mlm"].get_num_params()

    # ---- the coupling that actually put the number in the log ----

    def test_train_py_logs_the_bare_call(self):
        """Why the SIGNATURE default is what this card is about.

        `src/train.py` is not editable here, so the coupling is pinned from this
        side: it calls `model.get_num_params()` with no arguments. That is what
        makes the default the logged quantity, and it is why "just flip the
        default" is a real fix rather than a cosmetic one.
        """
        from src.train import __file__ as train_path

        with open(train_path, encoding="utf-8") as fh:
            src = fh.read()
        bare = re.search(r"=\s*model\.get_num_params\(\s*\)", src)
        assert bare, (
            "src/train.py no longer calls model.get_num_params() with no "
            "arguments; if it now passes a flag, the logged quantity and the "
            "KIND pinned here are different numbers"
        )

    # ---- the counterweight: KIND is not closeness ----

    def test_the_numbers_really_do_differ(self):
        """Guard the guard.

        If a future refactor made all three counts coincide, "same KIND" would
        pass without saying anything. They must remain distinct -- three
        architectures do not have equal parameter counts -- and this is the test
        that says so, so the KIND assertions cannot be satisfied by making the
        number the same instead of the meaning.
        """
        shape = SHAPES[2]
        logged = {arm: mod.get_num_params() for arm, mod in _build_all(*shape).items()}
        assert len(set(logged.values())) == 3, (
            f"all three arms reported the same count {logged}; either the arms "
            "became the same architecture or the count stopped measuring them"
        )
        # And they are allowed to be far apart: no upper bound on the spread is
        # asserted, because none is defensible.
