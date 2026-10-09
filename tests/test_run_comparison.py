# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""`src/interp/run_comparison.load_model` must restore the WHOLE model.

Why this file exists
--------------------
`src/interp/run_comparison.py` is the one-command JEPA-vs-baseline interpretability
pipeline. Its `load_model` is the only place in `src/interp/` that reads a
training checkpoint, and for the whole of wave 1 **no test touched it**:
`git grep run_comparison tests/` returned nothing, so restoring the pre-fix
`load_model` left the suite green.

The pre-fix reader was four per-submodule `.get()` reads:

    model.encoder.load_state_dict(ckpt.get("encoder", {}))
    model.target_encoder.load_state_dict(ckpt.get("target_encoder", {}))
    model.predictor.load_state_dict(ckpt.get("predictor", {}))
    model.decoder.load_state_dict(ckpt.get("decoder", {}))

`src.train.save_checkpoint` writes `state["model"] = model.state_dict()` and
nothing per-submodule, so all four reads resolved to `{}` and the loader was
reading a format that had not been written since before the fidelity campaign.
Two things were structurally invisible to it:

* every mechanism tensor. On the fixture below (all 16 mechanisms on, toy
  width) the four submodules reach 59 of 126 state-dict keys; the 67 they miss
  are every mechanism tensor — `jawp.workspace_Q` (a Stiefel parameter),
  `target_centering.center` and `sigreg.sketch_directions` (loss inputs),
  `wsd.target_cov`, `sta.ref_cov`, the twelve `*_total_steps` counters.
  Restored with `strict=False` they would have sat at their initial values — a
  comparison run would have scored representations from a half-randomised model,
  and one that claimed `sta.is_initialized == True` while `sta.ref_cov` sat at
  `0.01 * I`;
* `regression_head` for the data2vec branch. There was **no** regression-head
  read anywhere in the file, so every data2vec-vs-JEPA comparison was scored
  against a randomly initialised head.

Those are the two silent-drops this file closes, plus the three loudness
properties that make a silent drop impossible in the first place: a legacy-shaped
checkpoint is refused by name, `strict=True` is real, and the format check is
the same one every branch shares.

Architecture width
------------------
`load_model` hardcodes production widths (JEPA default config = 262M
parameters, ~1 GB of `torch.save`; the two baselines are built at
`embed_dim=768, depth=12`). That is unaffordable on a shared CPU, so each test
monkeypatches only the *constructor* that `load_model` reaches through a
function-local import (`TextSpanJEPAConfig`, `MLMBaseline`,
`Data2VecTextBaseline`) down to a toy width. Nothing else about `load_model` is
substituted: the same `_full_model_state` check, the same `state` dict, the
same `load_state_dict(state, strict=True)`, the same three branches. Every
property asserted below is width-independent — completeness of the restore,
not tensor count.
"""

import pytest
import torch

from baselines.data2vec_baseline import Data2VecTextBaseline
from baselines.mlm_baseline import MLMBaseline
from src.interp.run_comparison import load_model
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.train import CheckpointLoadError, save_checkpoint
from src.utils.torchio import safe_torch_load

VOCAB = 64
SEQ = 16
EMBED = 32

#: The four submodules the pre-fix reader looked for. A key outside this set is
#: a key that reader could not have restored.
LEGACY_MODULE_PREFIXES = ("encoder.", "target_encoder.", "predictor.", "decoder.")


def _toy_jepa_config():
    """Every mechanism on, at toy width, so no module is `None`."""
    return TextSpanJEPAConfig(
        vocab_size=VOCAB,
        max_seq_len=SEQ,
        embed_dim=EMBED,
        encoder_depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        predictor_embed_dim=16,
        predictor_depth=1,
        future_offsets=(1,),
        num_refine_steps=1,
        use_jawp=True,
        use_cgn=True,
        use_pcr=True,
        use_spc=True,
        use_sta=True,
        use_wsd=True,
        use_puc=True,
        use_rdc=True,
        use_cmc=True,
        use_gac=True,
        use_wsr=True,
        use_swip=True,
    )


def _toy_mlm(**_ignored):
    """`load_model` passes production kwargs; the width is not what is under test."""
    return MLMBaseline(vocab_size=VOCAB, max_seq_len=SEQ, embed_dim=EMBED, depth=1, num_heads=2)


def _toy_data2vec(**_ignored):
    return Data2VecTextBaseline(
        vocab_size=VOCAB, max_seq_len=SEQ, embed_dim=EMBED, depth=1, num_heads=2
    )


@pytest.fixture
def toy_width(monkeypatch):
    """Shrink only what `load_model` constructs. See the module docstring."""
    import baselines.data2vec_baseline as d2v_mod
    import baselines.mlm_baseline as mlm_mod
    import src.models.jepa as jepa_mod

    monkeypatch.setattr(jepa_mod, "TextSpanJEPAConfig", _toy_jepa_config)
    monkeypatch.setattr(mlm_mod, "MLMBaseline", _toy_mlm)
    monkeypatch.setattr(d2v_mod, "Data2VecTextBaseline", _toy_data2vec)


def _scramble(model):
    """Overwrite every tensor with a value no fresh init would produce.

    A restore that misses a tensor is then detectable by inspection rather than
    by a coincidence: every tensor carries its own index, so a shifted or
    dropped key cannot alias another key's value either. Returns the mapping
    that was written.
    """
    with torch.no_grad():
        state = model.state_dict()
        for i, (_name, tensor) in enumerate(state.items()):
            ramp = torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
            tensor.copy_(ramp + 1000.0 * (i + 1))
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _max_abs_diff(saved, loaded):
    """Largest ``|saved - loaded|`` over the whole state dict. 0.0 iff bitwise equal.

    Key sets must match exactly first: a missing key is the failure this file
    is about, and averaging it away would hide it.
    """
    assert set(saved) == set(loaded), (
        f"key sets differ after the round trip: "
        f"missing={sorted(set(saved) - set(loaded))}, "
        f"unexpected={sorted(set(loaded) - set(saved))}"
    )
    worst = 0.0
    integer_mismatches = []
    for name, before in saved.items():
        after = loaded[name]
        assert before.shape == after.shape, f"{name}: shape {after.shape} != {before.shape}"
        if before.is_floating_point():
            worst = max(worst, float((before.float() - after.float()).abs().max()))
        elif not torch.equal(before, after):
            integer_mismatches.append(f"{name} (saved {before.tolist()}, got {after.tolist()})")
    # Report every discrepancy, not the first one: a partial restore drops
    # several tensors and the operator needs the list.
    assert worst == 0.0 and not integer_mismatches, (
        f"max |saved - loaded| = {worst!r} over {len(saved)} tensors; "
        f"integer tensors also differ: {integer_mismatches or 'none'}"
    )
    return worst


def _save(tmp_path, name, model, model_name):
    path = str(tmp_path / name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(path, model, optimizer, None, epoch=1, global_step=7, model_name=model_name)
    return path


class TestJepaRoundTrip:
    def test_save_load_round_trip_is_bit_exact(self, tmp_path, toy_width):
        """A real `save_checkpoint` -> `load_model` round trip, max |delta| == 0.0.

        Not `load_state_dict` against a dict the test built: the payload is
        written by `src.train.save_checkpoint`, the one the trainer writes, and
        read back by `run_comparison.load_model`, the one the comparison
        pipeline reads.
        """
        source = TextSpanJEPA(_toy_jepa_config())
        saved = _scramble(source)
        path = _save(tmp_path, "jepa.pth.tar", source, "text_span_jepa")

        loaded = load_model(path, "jepa", "cpu")

        max_abs_diff = _max_abs_diff(saved, loaded.state_dict())
        assert max_abs_diff == 0.0, (
            f"round trip is not bit-exact: max |saved - loaded| = {max_abs_diff!r} "
            f"over {len(saved)} tensors"
        )

    def test_tensors_outside_the_four_legacy_submodules_survive(self, tmp_path, toy_width):
        """The mechanism tensors the per-module `.get()` reads could not reach.

        Each name here is a `load_model`-invisible tensor on the pre-fix reader:
        `jawp.workspace_Q` is a Stiefel parameter, `target_centering.center`
        and `sigreg.sketch_directions` are loss inputs, `sigreg.t_points` fixes
        the SIGReg integration grid.
        """
        wanted = (
            "jawp.workspace_Q",
            "jawp.active_k",
            "target_centering.center",
            "sigreg.sketch_directions",
            "sigreg.t_points",
        )
        source = TextSpanJEPA(_toy_jepa_config())
        saved = _scramble(source)
        for name in wanted:
            assert name in saved, f"fixture does not build {name}"
            assert not name.startswith(LEGACY_MODULE_PREFIXES), (
                f"{name} is inside a legacy submodule prefix, so it no longer "
                f"pins anything about the dropped mechanism tensors"
            )
        path = _save(tmp_path, "jepa.pth.tar", source, "text_span_jepa")

        loaded = load_model(path, "jepa", "cpu").state_dict()

        for name in wanted:
            assert name in loaded, f"{name} absent after the round trip"
            assert torch.equal(saved[name], loaded[name]), (
                f"{name} came back different from what was saved "
                f"(max |delta| = {float((saved[name].float() - loaded[name].float()).abs().max())})"
            )

    def test_load_covers_more_than_the_legacy_submodules(self, tmp_path, toy_width):
        """Restored key count vs. the count the old reader reached.

        Pinned as a *ratio of the fixture's own keys*, not a magic number: the
        point is that the restore is total, so a mechanism that grows a new
        tensor later is covered without editing this file.
        """
        source = TextSpanJEPA(_toy_jepa_config())
        saved = _scramble(source)
        path = _save(tmp_path, "jepa.pth.tar", source, "text_span_jepa")

        loaded = load_model(path, "jepa", "cpu").state_dict()

        legacy_reachable = {n for n in saved if n.startswith(LEGACY_MODULE_PREFIXES)}
        assert set(loaded) == set(saved)
        assert len(loaded) > len(legacy_reachable), (
            f"restore covers {len(loaded)} keys but the legacy reader reached "
            f"{len(legacy_reachable)}; the gap is the silent drop"
        )
        # Key coverage alone is not the property: a partial restore presents the
        # right key set and the wrong values. Check the tensors too.
        max_abs_diff = _max_abs_diff(saved, loaded)
        assert max_abs_diff == 0.0, (
            f"{len(loaded)} keys restored but max |saved - loaded| = {max_abs_diff!r}; "
            f"the tensors outside {LEGACY_MODULE_PREFIXES} kept their initial values"
        )


class TestBaselineBranches:
    def test_data2vec_regression_head_is_restored_bitwise(self, tmp_path, toy_width):
        """`regression_head.0.weight` — the tensor no read in the file ever named.

        A data2vec comparison scores predictions through this head, so a
        randomly initialised one silently compares against noise.
        """
        name = "regression_head.0.weight"
        source = _toy_data2vec()
        assert name in source.state_dict(), "fixture does not build a regression head"
        fresh = source.state_dict()[name].detach().clone()
        saved = _scramble(source)
        assert not torch.equal(saved[name], fresh), (
            "scramble left the regression head at its initial value, so a "
            "missing restore would be indistinguishable from a successful one"
        )
        path = _save(tmp_path, "data2vec.pth.tar", source, "data2vec")

        loaded = load_model(path, "data2vec", "cpu")

        restored = loaded.state_dict()[name]
        assert torch.equal(saved[name], restored), (
            f"{name} was not restored: max |delta| = "
            f"{float((saved[name] - restored).abs().max())}"
        )
        assert not torch.equal(restored, fresh), "regression head is still at its initial value"

    def test_data2vec_round_trip_is_bit_exact(self, tmp_path, toy_width):
        source = _toy_data2vec()
        saved = _scramble(source)
        path = _save(tmp_path, "data2vec.pth.tar", source, "data2vec")

        loaded = load_model(path, "data2vec", "cpu")

        assert _max_abs_diff(saved, loaded.state_dict()) == 0.0

    def test_mlm_branch_restores_its_head(self, tmp_path, toy_width):
        source = _toy_mlm()
        saved = _scramble(source)
        path = _save(tmp_path, "mlm.pth.tar", source, "mlm")

        loaded = load_model(path, "mlm", "cpu")

        assert _max_abs_diff(saved, loaded.state_dict()) == 0.0


class TestLegacyCheckpointRefused:
    """A pre-`state_dict` checkpoint is refused by name, never half-read."""

    def _legacy_payload(self, model):
        """The shape the pre-fix reader wanted: named submodules, no `"model"`."""
        return {
            "model_name": "text_span_jepa",
            "encoder": model.encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "decoder": model.decoder.state_dict(),
            "target_centering_center": model.target_centering.center.detach().clone(),
            "jawp_workspace_Q": model.jawp.workspace_Q.detach().clone(),
            "epoch": 1,
            "global_step": 7,
        }

    def test_legacy_shaped_checkpoint_raises_a_named_format_error(self, tmp_path, toy_width):
        model = TextSpanJEPA(_toy_jepa_config())
        path = str(tmp_path / "legacy.pth.tar")
        torch.save(self._legacy_payload(model), path)

        with pytest.raises(CheckpointLoadError) as excinfo:
            load_model(path, "jepa", "cpu")

        message = str(excinfo.value)
        assert not isinstance(
            excinfo.value, KeyError
        ), "a bare KeyError is not a usable error message for a checkpoint format change"
        assert "model" in message, f"the message does not name the key it wanted: {message!r}"
        assert "encoder" in message, (
            f"the message does not show what it found, so the operator cannot "
            f"tell a legacy checkpoint from a truncated one: {message!r}"
        )
        assert (
            "model_name" in message
        ), f"the message does not list the keys actually present: {message!r}"

    @pytest.mark.parametrize("model_type", ["jepa", "mlm", "data2vec"])
    def test_every_branch_refuses_a_legacy_checkpoint(self, tmp_path, toy_width, model_type):
        """One format check, shared. A branch that skipped it would half-read."""
        model = TextSpanJEPA(_toy_jepa_config())
        path = str(tmp_path / f"legacy_{model_type}.pth.tar")
        torch.save(self._legacy_payload(model), path)

        with pytest.raises(CheckpointLoadError):
            load_model(path, model_type, "cpu")

    def test_non_dict_payload_is_refused(self, tmp_path, toy_width):
        path = str(tmp_path / "not_a_state.pth.tar")
        torch.save([1, 2, 3], path)

        with pytest.raises(CheckpointLoadError):
            load_model(path, "jepa", "cpu")

    def test_unknown_model_type_still_raises_value_error(self, tmp_path, toy_width):
        model = TextSpanJEPA(_toy_jepa_config())
        path = _save(tmp_path, "jepa.pth.tar", model, "text_span_jepa")

        with pytest.raises(ValueError, match="Unknown model type"):
            load_model(path, "bert", "cpu")


class TestStrictIsStrict:
    """`strict=True` must be a real check, not decoration on the call."""

    def _payload(self, tmp_path):
        source = TextSpanJEPA(_toy_jepa_config())
        _scramble(source)
        path = _save(tmp_path, "jepa.pth.tar", source, "text_span_jepa")
        return safe_torch_load(path, map_location="cpu")

    def test_a_missing_tensor_is_rejected_by_name(self, tmp_path, toy_width):
        payload = self._payload(tmp_path)
        missing = "target_centering.center"
        assert not missing.startswith(LEGACY_MODULE_PREFIXES), (
            f"{missing} is readable by the legacy per-module loads, so dropping it "
            f"would not prove strictness"
        )
        del payload["model"][missing]
        path = str(tmp_path / "missing.pth.tar")
        torch.save(payload, path)

        with pytest.raises(RuntimeError) as excinfo:
            load_model(path, "jepa", "cpu")

        message = str(excinfo.value)
        assert (
            missing in message
        ), f"the error does not name the missing tensor {missing!r}: {message!r}"
        assert "Missing key" in message, f"unexpected strictness failure: {message!r}"

    def test_an_unexpected_tensor_is_rejected_by_name(self, tmp_path, toy_width):
        payload = self._payload(tmp_path)
        payload["model"]["not_a_real_tensor"] = torch.zeros(3)
        path = str(tmp_path / "extra.pth.tar")
        torch.save(payload, path)

        with pytest.raises(RuntimeError) as excinfo:
            load_model(path, "jepa", "cpu")

        message = str(excinfo.value)
        assert (
            "not_a_real_tensor" in message
        ), f"the error does not name the unexpected tensor: {message!r}"
        assert "Unexpected key" in message, f"unexpected strictness failure: {message!r}"

    def test_shape_mismatch_is_rejected(self, tmp_path, toy_width):
        payload = self._payload(tmp_path)
        name = "sigreg.t_points"
        payload["model"][name] = torch.zeros(5)
        path = str(tmp_path / "shape.pth.tar")
        torch.save(payload, path)

        with pytest.raises(RuntimeError) as excinfo:
            load_model(path, "jepa", "cpu")

        assert name in str(
            excinfo.value
        ), f"the error does not name the mis-shaped tensor: {str(excinfo.value)!r}"

    def test_loaded_model_is_in_eval_mode(self, tmp_path, toy_width):
        """The comparison pipeline scores, so it must not be reading dropout state."""
        source = TextSpanJEPA(_toy_jepa_config())
        _scramble(source)
        path = _save(tmp_path, "jepa.pth.tar", source, "text_span_jepa")

        loaded = load_model(path, "jepa", "cpu")

        assert not loaded.training
        assert next(loaded.parameters()).device.type == "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# What the comparison can and cannot conclude.
#
# `run_full_comparison` is handed ONE checkpoint per arm and ONE dataloader.
# A claim of the form "JEPA beats the baseline" is a claim about the
# population of training runs on a population of corpora, so the units that
# matter are checkpoints and corpora -- and the signature supplies exactly
# one of each. The only thing it can resample is the ROW of a representation
# matrix.
#
# These tests pin the consequence, which is that the pipeline REFUSES rather
# than emits a number. They are written so the pre-fix code fails them:
#
#   * `results["statistical"]` held a subtraction under the key "statistical"
#     with no p-value anywhere. The tests assert there is still no p-value,
#     and that the refusal is explicit.
#   * the pairing of the two arms was an accident of the caller passing
#     shuffle=False. A shuffled loader now raises.
#   * Phase 4's "MI with position (surface)" was `arange(N)`, an arithmetic
#     identity returning 0.0 for every representation.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def comparison_arms(monkeypatch, toy_width):
    """Two toy-width models, built only once per test."""
    import baselines.mlm_baseline as mlm_mod
    import src.models.jepa as jepa_mod

    torch.manual_seed(0)
    jepa = TextSpanJEPA(_toy_jepa_config()).eval()
    base = mlm_mod.MLMBaseline(
        vocab_size=VOCAB, max_seq_len=SEQ, embed_dim=EMBED, depth=1, num_heads=2
    ).eval()
    return jepa, base


def _loader(shuffle, n_rows=32, seed=0):
    from torch.utils.data import DataLoader, TensorDataset

    from src.interp import rng

    gen = rng.generator_for(seed, "test_run_comparison.loader")
    ids = torch.randint(0, VOCAB, (n_rows, SEQ), generator=gen)
    return DataLoader(TensorDataset(ids), batch_size=8, shuffle=shuffle)


def _run(tmp_path, arms, **kwargs):
    from src.interp.run_comparison import run_full_comparison

    jepa, base = arms
    return run_full_comparison(jepa, base, _loader(kwargs.pop("shuffle", False)), str(tmp_path), "cpu", 4, **kwargs)


class TestNoPValueTheDesignCannotSupport:
    """The central refusal. No significance claim, because n_seed == 1."""

    def test_report_contains_no_p_value_or_significance_flag(self, tmp_path, comparison_arms):
        """The whole point: wiring in `statistical_tests.py` would make this red.

        A p-value here conditions on the checkpoint, while the report's
        verdicts are about training runs. Emitting one would be false
        confidence, so its absence is the property under test.
        """
        results = _run(tmp_path, comparison_arms)

        stat = results["statistical"]
        assert stat, "the comparison block is empty; the run produced nothing to compare"
        for metric, entry in stat.items():
            assert "jepa" in entry and "baseline" in entry, f"{metric} lost its point values"
            assert "p_value" not in entry, f"{metric} reports a p-value the design cannot support"
            assert "significant" not in entry, (
                f"{metric} reports a significance flag. With one checkpoint per arm "
                f"the p-value conditions on the seed and certifies nothing about it."
            )

    def test_verdict_is_not_testable_and_says_what_is_missing(self, tmp_path, comparison_arms):
        """The report must state its own limits, not merely omit a number."""
        design = _run(tmp_path, comparison_arms)["comparison_design"]

        assert design["verdict"] == "NOT_TESTABLE", (
            f"the report believes it can support a verdict: {design['verdict']!r}"
        )
        units = design["units_of_replication"]
        assert units["checkpoints_per_arm"] == 1
        assert units["corpora"] == 1

        uncovered = " ".join(design["variance_not_covered"])
        assert "seed" in uncovered, f"the missing seed variance is not recorded: {uncovered!r}"
        assert "corpus" in uncovered, f"the missing corpus variance is not recorded: {uncovered!r}"
        assert design["required_to_support_a_verdict"], (
            "a refusal that does not say what would fix it is not actionable"
        )

    def test_summary_does_not_claim_a_supported_verdict(self, tmp_path, comparison_arms):
        summary = _run(tmp_path, comparison_arms)["summary"]

        assert summary["verdict_supported"] is False, (
            "the summary reports a supported verdict while the design block "
            "refuses one; a reader sees only the summary"
        )

    def test_the_four_metrics_are_not_four_tests(self, tmp_path, comparison_arms):
        """`effective_rank_online` and `sv_entropy_online` are one measurement.

        Measured bootstrap-difference correlation between the two is r=0.9999
        -- both are functionals of the same singular spectrum -- and
        `collapsed_dim_ratio_online` is identically 0.0 on both arms. The
        multiplicity block must therefore record that the nominal family is
        smaller than the four entries, rather than letting three numbers from
        one spectrum read as three findings.
        """
        design = _run(tmp_path, comparison_arms)["comparison_design"]
        multiplicity = design["multiplicity"]

        assert multiplicity["metrics_in_the_statistical_block"] == 4
        assert multiplicity["comparisons_published_in_this_report"] > 4, (
            "the report publishes more scalar comparisons than the statistical "
            "block holds, so no correction over the smaller family covers the "
            f"larger one: {multiplicity}"
        )
        assert "note" in multiplicity

    def test_the_sign_count_carries_its_null_probability(self, tmp_path, comparison_arms):
        """`wins > n//2` fires 31% of the time on nothing.

        Phase 8 and `RobustnessBattery` both treat a majority of "JEPA
        better" booleans as an advantage. Under the null that the arms are
        indistinguishable, 3-of-4 has probability 0.3125, so the count is
        reported next to that number and explicitly not called a verdict.
        """
        summary = _run(tmp_path, comparison_arms)["summary"]

        assert summary["geometry_majority_is_a_verdict"] is False
        assert 0.0 < summary["geometry_majority_null_probability"] <= 1.0

    @pytest.mark.parametrize(
        "n_wins,n_total,expected",
        [
            (3, 4, 0.3125),  # 5/16 -- the RobustnessBattery case
            (4, 6, 0.34375),  # 22/64
            (5, 8, 0.36328125),  # 93/256
            (0, 0, 1.0),
        ],
    )
    def test_majority_null_probability_matches_the_exact_binomial(self, n_wins, n_total, expected):
        """Closed form, so the number in the report is checkable by hand."""
        from src.interp.run_comparison import majority_null_probability

        assert majority_null_probability(n_wins, n_total) == pytest.approx(expected)


class TestPairingIsAssertedNotAssumed:
    """The two arms must be scored on the same sequences in the same order."""

    def test_a_shuffled_dataloader_is_refused(self, tmp_path, comparison_arms):
        """`run_full_comparison` iterates the caller's loader once per arm.

        With `shuffle=True` each arm gets a different ordering, so the
        comparison is unpaired while still being reported as a comparison --
        measured 47 of 48 rows misaligned. This raises instead.
        """
        from src.interp.run_comparison import ComparisonDesignError

        with pytest.raises(ComparisonDesignError) as excinfo:
            _run(tmp_path, comparison_arms, shuffle=True)

        message = str(excinfo.value)
        assert "shuffle" in message, (
            f"the error does not name the likely cause, so the operator "
            f"cannot act on it: {message!r}"
        )

    def test_a_non_shuffling_dataloader_is_accepted(self, tmp_path, comparison_arms):
        design = _run(tmp_path, comparison_arms, shuffle=False)["comparison_design"]
        assert design["arms_paired"] is True

    def test_mismatched_shapes_are_refused_with_both_shapes_named(self, comparison_arms):
        from src.interp.run_comparison import ComparisonDesignError, assert_paired_extraction

        with pytest.raises(ComparisonDesignError) as excinfo:
            assert_paired_extraction(torch.zeros(4, 8), torch.zeros(3, 8))

        message = str(excinfo.value)
        assert "(4, 8)" in message and "(3, 8)" in message, (
            f"the error does not show both shapes: {message!r}"
        )


class TestPairedBootstrapIsActuallyPaired:
    """The one interval this design can honestly produce."""

    def test_identical_arms_give_a_zero_width_interval(self):
        """The sharpest available property, and it discriminates.

        Two copies of the same matrix differ by exactly zero on every
        replicate, so a PAIRED resampling returns `ci_lower == ci_upper == 0`.
        An unpaired one draws two independent row multisets, compares
        different subsets of the same array, and returns a WIDE interval on
        this degenerate input -- measured [-0.452, +0.140], i.e. it invents
        a difference between two copies of one array. That is what
        `statistical_tests.BootstrapCI.compare` does, and it is why the
        forbidden module is not wired in.
        """
        from src.interp.run_comparison import paired_row_bootstrap_ci

        gen = torch.Generator().manual_seed(0)
        reps = torch.randn(40, 8, generator=gen) * (1 + torch.arange(8).float() / 8)

        result = paired_row_bootstrap_ci(reps, reps, "effective_rank_online", n_bootstrap=8, seed=3)

        assert result["ci_lower"] == 0.0 and result["ci_upper"] == 0.0, (
            f"resampling the same matrix through both arms must give exactly "
            f"zero difference, got [{result['ci_lower']}, {result['ci_upper']}]. "
            f"A non-zero interval means the two arms were resampled "
            f"independently, discarding the pairing."
        )
        assert result["mean_diff"] == 0.0

    def test_a_real_difference_is_still_detected(self):
        """The degenerate case above must not pass because nothing ever moves."""
        from src.interp.run_comparison import paired_row_bootstrap_ci

        gen = torch.Generator().manual_seed(0)
        scale = 1 + torch.arange(8).float() / 8
        a = torch.randn(40, 8, generator=gen) * scale
        b = torch.randn(40, 8, generator=gen) * scale * 3.0

        result = paired_row_bootstrap_ci(a, b, "effective_rank_online", n_bootstrap=8, seed=3)

        assert result["ci_upper"] > result["ci_lower"], (
            "the interval is degenerate on inputs that DO differ, so the "
            "zero-width case above proves nothing"
        )
        assert result["mean_diff"] != 0.0

    def test_bootstrap_refuses_fewer_than_two_rows(self):
        """N < 2 is a refusal, not a null result.

        `statistical_tests.PairedPermutationTest.compute` returns
        `p_value: 1.0, significant: False` for N < 2, which a report prints
        as "tested, not significant" -- i.e. "no difference found". It is
        not that; no test ran.
        """
        from src.interp.run_comparison import ComparisonDesignError, paired_row_bootstrap_ci

        with pytest.raises(ComparisonDesignError) as excinfo:
            paired_row_bootstrap_ci(torch.randn(1, 8), torch.randn(1, 8), "effective_rank_online")

        assert "at least 2 rows" in str(excinfo.value)

    def test_the_interval_is_opt_in_and_says_what_it_covers(self, tmp_path, comparison_arms):
        """It costs two `CollapseDiagnostics.compute` calls per replicate.

        Measured 90 s at B=100 and 453 s at B=500 for the production shape
        (N=500, D=768, one thread), so it is off by default rather than
        silently multiplying the pipeline's runtime. When it IS computed, the
        record must say what question it answers.
        """
        without = _run(tmp_path, comparison_arms)["statistical"]["effective_rank_online"]
        assert "paired_row_bootstrap" not in without, "the interval ran at the default cost"

        with_it = _run(tmp_path, comparison_arms, n_bootstrap=4, seed=0)
        entry = with_it["statistical"]["effective_rank_online"]
        assert entry["paired_row_bootstrap"]["n_resamples"] == 4
        assert "paired" in entry["paired_row_bootstrap"]["method"]

        covered = with_it["comparison_design"]["paired_row_bootstrap"]["covers"]
        assert "seed" in covered and "corpus" in covered, (
            f"the interval does not state what it cannot speak to: {covered!r}"
        )


class TestSurfaceFeatureIsNotTheRowIndex:
    """Phase 4's headline hypothesis was being tested against nothing."""

    def test_mi_surface_is_absent_rather_than_fabricated(self, tmp_path, comparison_arms):
        """`arange(N)` is an arithmetic identity, not a weak surface feature.

        Every row of `positions = arange(N).expand(-1, D)` is CONSTANT, so
        `F.normalize` maps all of them to one direction, the similarity
        matrix has identical rows, cross-entropy equals log(N), and the
        estimator's `max(mi, 0.0)` returns exactly 0.0 -- measured for noise,
        all-zeros, all-ones, rank-1 and 1e6-scaled representations alike. So
        `jepa_mi_position` and `baseline_mi_position` were 0.0 vs 0.0 for
        every model, and "JEPA has lower MI with surface features" was never
        tested.
        """
        it = _run(tmp_path, comparison_arms)["information_theory"]

        assert it["jepa_mi_surface"] is None, (
            f"a surface MI was reported with no surface feature supplied: "
            f"{it['jepa_mi_surface']!r}"
        )
        assert it["baseline_mi_surface"] is None
        assert "NOT COMPUTED" in it["surface_mi_note"], (
            f"the note does not say the hypothesis went untested: {it['surface_mi_note']!r}"
        )

    def test_the_untested_hypothesis_is_recorded_in_the_design_block(self, tmp_path, comparison_arms):
        design = _run(tmp_path, comparison_arms)["comparison_design"]
        assert any(
            "surface" in item for item in design["variance_not_covered"]
        ), f"the missing surface feature is not in the refusal: {design['variance_not_covered']!r}"

    def test_a_supplied_surface_feature_is_computed(self, tmp_path, comparison_arms):
        """The fix is one argument, not a redesign: supply the feature."""
        torch.manual_seed(0)
        feature = torch.randint(0, 8, (32, 4)).float()

        it = _run(tmp_path, comparison_arms, surface_features=feature)["information_theory"]

        assert it["jepa_mi_surface"] is not None
        assert "clamp" in it["surface_mi_note"], (
            "the note must say that a 0.0 means 'at the estimator's floor', "
            "not 'no dependence' -- the two are the same printed number"
        )

    def test_mismatched_surface_feature_length_is_refused(self, tmp_path, comparison_arms):
        with pytest.raises(ValueError) as excinfo:
            _run(tmp_path, comparison_arms, surface_features=torch.zeros(7, 4))
        assert "one row per scored sequence" in str(excinfo.value)


class TestRefuseNonsenseCorpusByDefault:
    """`main()` built random token IDs and wrote a report that looked real."""

    def test_random_token_run_is_refused_without_the_explicit_flag(self, tmp_path, monkeypatch):
        """The entry point must not silently produce a plausible artefact.

        It parsed `--dataset wikitext`, never loaded it, fell through to
        `torch.randint(0, 50304, (500, 128))`, ran the full protocol on
        uniform noise, and wrote `comparison_results.json` and `summary.txt`
        -- files indistinguishable from a real result.
        """
        import src.interp.run_comparison as rc

        monkeypatch.setattr(
            rc.argparse.ArgumentParser, "parse_args", lambda self, _a=None: self.parse_known_args()[0]
        )
        monkeypatch.setattr(rc, "load_model", lambda *a, **k: torch.zeros(1))
        monkeypatch.setattr("sys.argv", ["run_comparison", "--jepa_ckpt", "a", "--baseline_ckpt", "b"])

        with pytest.raises(SystemExit) as excinfo:
            rc.main()

        message = str(excinfo.value)
        assert "REFUSED" in message
        assert "--allow_random_tokens" in message, (
            f"the refusal does not name the flag that permits the run: {message!r}"
        )
        assert not (tmp_path / "comparison_results.json").exists(), (
            "a report was written despite the refusal"
        )
