# Copyright 2026 Text-Span JEPA Authors
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
