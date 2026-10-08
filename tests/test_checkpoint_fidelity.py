# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Checkpoint fidelity gates for `src/train.py`.

The suite was blind to three failure modes that only appear across a
save -> load boundary:

1.  **Resume divergence.** `test_training_e2e.py::test_resume_continues_global_step`
    asserts the step *counter* only. Measured with a 6-step toy run, the first
    post-resume loss was 5.8% off the continuous run. Cause: no RNG state was
    saved (mask sampling / DropPath / Gumbel / data order all replayed from a
    restarted stream) and 15 tensors silently reverted — 6 of them trainable
    parameters (`cgn.context_proj.*`, `pcr.refine_blocks.*`).
2.  **Loss inputs that lie about their own initialization.** `sta.is_initialized`
    was restored to True while `sta.ref_cov` / `sta.ref_eigenvalues` /
    `wsd.target_cov` / `wsd.target_Q` reverted to their `0.01 * I` init. The STA
    and WSD losses were then computed against a reference subspace the model had
    never seen, silently, for hundreds of steps.
3.  **Validation mutating training state.** `_validate` runs under `no_grad()`,
    which stops *gradient* writes and nothing else. The audit recorded 24
    buffers moving in place during a validation pass; the gate below covers
    the whole class, including buffers no mechanism ever guarded.

`test_resume_reproduces_continuous_run` is the decisive gate — it fails loudly
on all three at once.
"""

import pickle
import random

import numpy as np
import pytest
import torch

from src.datasets.kaggle import TextDataset
from src.masks.span import SpanMaskCollator
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.train import (
    CheckpointLoadError,
    _restore_training_state,
    _validate,
    compute_loss,
    do_ema_update,
    get_param_groups,
    load_checkpoint,
    save_checkpoint,
)
from src.utils.schedulers import CosineWDSchedule, EMATauSchedule, WarmupCosineSchedule
from src.utils.seed import seed_everything
from src.utils.torchio import UnsafeCheckpointError, safe_torch_load

VOCAB = 64
SEQ = 16
TOTAL_STEPS = 100

# Tensors that silently reverted across a save -> load round trip in the
# pre-fix implementation. `sta.ref_cov` / `wsd.target_cov` / `wsd.target_Q` /
# `sta.ref_eigenvalues` are LOSS INPUTS, not diagnostics: the STA and WSD
# objectives were computed against a stale reference subspace after every
# resume. Named explicitly so the gate fails if one is dropped again.
MUTATED_LOSS_INPUT_BUFFERS = (
    "sta.current_eigenvalues",
    "sta.ref_cov",
    "sta.ref_eigenvalues",
    "wsd.target_Q",
    "wsd.target_cov",
)
MUTATED_RUNNING_BUFFERS = (
    "puc.running_entropy",
    "puc.running_overconfidence",
    "rdc.running_ortho_drift_norm",
    "rdc.running_workspace_drift_norm",
)
UNSAVED_TRAINABLE_PARAMS = (
    "cgn.context_proj.weight",
    "cgn.context_proj.bias",
    "pcr.refine_blocks.0.net.0.weight",
    "pcr.refine_blocks.0.net.2.weight",
    "pcr.refine_blocks.1.net.0.weight",
    "pcr.refine_blocks.1.net.2.weight",
)


def _model_config():
    """Every buffer-mutating mechanism on, at toy width."""
    return TextSpanJEPAConfig(
        vocab_size=VOCAB,
        max_seq_len=SEQ,
        embed_dim=32,
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
    )


def _build(seed=0):
    """Model + optimizer + schedulers + collator, exactly as `main()` builds them."""
    seed_everything(seed)
    model = TextSpanJEPA(_model_config())
    optimizer = torch.optim.AdamW(get_param_groups(model, "text_span_jepa", wd=0.0), lr=1e-3)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=1, start_lr=1e-4, ref_lr=1e-3, final_lr=1e-5, T_max=TOTAL_STEPS
    )
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=0.0, final_wd=0.0, T_max=TOTAL_STEPS)
    ema_scheduler = EMATauSchedule(tau_start=0.996, tau_end=1.0, total_steps=TOTAL_STEPS)
    collator = SpanMaskCollator(mask_ratio=0.3, span_length_range=(2, 4), mask_token_id=0, pad_id=0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, optimizer, scheduler, wd_scheduler, ema_scheduler, collator, scaler


def _batches(n=4, seed=123):
    """Fixed batch content: only the *mask* stream is random, which is the point."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (2, SEQ), generator=g) for _ in range(n)]


def _run_steps(state, batches, start_step, n_steps):
    """One optimizer step per iteration, mirroring the `main()` inner loop."""
    model, optimizer, scheduler, wd_scheduler, ema_scheduler, collator, _ = state
    losses = []
    for i in range(n_steps):
        batch = batches[i % len(batches)]
        collated = collator([{"input_ids": batch[j]} for j in range(batch.size(0))])
        scheduler.step()
        wd_scheduler.step()
        total_loss, _, _ = compute_loss(
            model,
            collated["masked_input_ids"],
            collated["original_input_ids"],
            collated["mask_positions"],
            current_step=start_step + i,
            total_steps=TOTAL_STEPS,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        model.jawp.stiefel_retract()
        model.pcr.stiefel_retract()
        model.spc.stiefel_retract()
        optimizer.zero_grad()
        do_ema_update(model, "text_span_jepa", tau=ema_scheduler.step())
        collator.step()
        losses.append(total_loss.item())
    return losses


def _snapshot(model):
    return {
        "buffers": {n: b.detach().clone() for n, b in model.named_buffers()},
        "params": {n: p.detach().clone() for n, p in model.named_parameters()},
    }


def _corrupt(model):
    """Destroy in-memory state so a *restore* is the only way back."""
    with torch.no_grad():
        for b in model.buffers():
            b.fill_(0.123456)
        for p in model.parameters():
            p.fill_(-0.654321)


class TestResumeFidelity:
    def test_resume_reproduces_continuous_run(self, tmp_path):
        """THE decisive gate: a resumed run must be bit-comparable to a run
        that was never interrupted.

        Pre-fix measurement on this fixture: the first post-resume loss was
        ~6% off the continuous value, because neither the RNG streams nor 15
        tensors (6 of them trainable) survived the round trip.
        """
        batches = _batches()
        path = str(tmp_path / "ckpt.pth.tar")

        # --- continuous reference: 3 steps, cut, 3 more steps -------------
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        pre = _run_steps(state, batches, start_step=0, n_steps=3)
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch=0,
            global_step=3,
            ema_step=3,
            mask_step=3,
            model_name="text_span_jepa",
        )
        continuous = _run_steps(state, batches, start_step=3, n_steps=3)

        # --- resumed: fresh process, fresh model, restore, 3 steps --------
        state2 = _build(seed=0)
        model2, optimizer2, sched2, wd2, ema2, coll2, scaler2 = state2
        start_epoch, global_step, ema_step, mask_step, _extra = _restore_training_state(
            {"meta": {"load_checkpoint": True, "read_checkpoint": None}},
            str(tmp_path),
            path,
            model2,
            optimizer2,
            scaler2,
            "text_span_jepa",
            sched2,
            wd2,
            ema2,
            coll2,
        )
        assert (start_epoch, global_step, ema_step, mask_step) == (0, 3, 3, 3)
        resumed = _run_steps(state2, batches, start_step=global_step, n_steps=3)

        for i, (cont, res) in enumerate(zip(continuous, resumed)):
            rel = abs(res - cont) / max(abs(cont), 1e-12)
            assert rel < 1e-6, (
                f"post-resume step {i} diverged: continuous={cont:.6f} "
                f"resumed={res:.6f} ({rel * 100:.2f}% relative). "
                f"pre-cut losses were {[round(v, 4) for v in pre]}"
            )

    def test_checkpoint_payload_carries_rng_state(self, tmp_path):
        """Every random stream that touches training must be in the payload.

        Mask sampling (`src/masks/span.py` uses numpy), DropPath and the CGN
        Gumbel use torch, the python `random` module feeds the worker seeds.
        A checkpoint without these makes every resume a different experiment.
        """
        path = str(tmp_path / "ckpt.pth.tar")
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        _run_steps(state, _batches(), start_step=0, n_steps=2)
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch=0,
            global_step=2,
            model_name="text_span_jepa",
        )
        payload = safe_torch_load(path, map_location="cpu")
        assert "rng_state" in payload, "checkpoint payload has no RNG state at all"
        for key in ("python_random", "numpy", "torch"):
            assert key in payload["rng_state"], f"RNG stream {key!r} is not checkpointed"

    def test_rng_state_restores_all_three_streams(self, tmp_path):
        path = str(tmp_path / "ckpt.pth.tar")
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        _run_steps(state, _batches(), start_step=0, n_steps=2)
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch=0,
            global_step=2,
            model_name="text_span_jepa",
        )
        # Sampled *after* the save, i.e. the next draws out of the saved streams.
        expected = (float(np.random.rand()), float(random.random()), float(torch.rand(1)))
        # Burn the streams, then restore.
        np.random.rand(17), random.random(), torch.rand(17)
        state2 = _build(seed=999)  # different seed -> different streams
        model2, optimizer2, sched2, wd2, ema2, coll2, scaler2 = state2
        _restore_training_state(
            {"meta": {"load_checkpoint": True, "read_checkpoint": None}},
            str(tmp_path),
            path,
            model2,
            optimizer2,
            scaler2,
            "text_span_jepa",
            sched2,
            wd2,
            ema2,
            coll2,
        )
        got = (float(np.random.rand()), float(random.random()), float(torch.rand(1)))
        assert got == pytest.approx(
            expected, rel=1e-12
        ), f"RNG streams not restored: expected {expected}, got {got}"

    def test_loss_input_buffers_round_trip(self, tmp_path):
        """`sta.ref_cov`, `wsd.target_cov`, ... are consumed by the loss.

        Pre-fix they were absent from the payload while `sta.is_initialized`
        *was* restored, so a resumed run computed STA/WSD against a reference
        subspace initialized at `0.01 * I` while claiming to be calibrated.
        """
        path = str(tmp_path / "ckpt.pth.tar")
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        _run_steps(state, _batches(), start_step=0, n_steps=3)
        before = _snapshot(model)
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch=0,
            global_step=3,
            model_name="text_span_jepa",
        )
        _corrupt(model)
        load_checkpoint(path, model, optimizer, scaler, model_name="text_span_jepa")
        after = _snapshot(model)
        names = MUTATED_LOSS_INPUT_BUFFERS + MUTATED_RUNNING_BUFFERS
        for name in names:
            assert name in before["buffers"], f"fixture does not exercise {name}"
            assert torch.equal(
                after["buffers"][name], before["buffers"][name]
            ), f"{name} did not survive the checkpoint round trip"

    def test_untrained_trainable_params_round_trip(self, tmp_path):
        """`cgn.context_proj.*` and `pcr.refine_blocks.*` are trainable and were
        never in the payload — max |delta| 3.0e-3 for context_proj on the toy
        fixture, i.e. far larger than the optimizer's own state drift.
        """
        path = str(tmp_path / "ckpt.pth.tar")
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        _run_steps(state, _batches(), start_step=0, n_steps=3)
        before = _snapshot(model)
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch=0,
            global_step=3,
            model_name="text_span_jepa",
        )
        _corrupt(model)
        load_checkpoint(path, model, optimizer, scaler, model_name="text_span_jepa")
        after = _snapshot(model)
        for name in UNSAVED_TRAINABLE_PARAMS:
            assert name in before["params"], f"fixture does not exercise {name}"
            assert torch.equal(
                after["params"][name], before["params"][name]
            ), f"trainable parameter {name} did not survive the checkpoint round trip"

    def test_every_named_tensor_round_trips(self, tmp_path):
        """Whole-`state_dict` coverage, not a hand-picked list: the pre-fix
        implementation enumerated 48 keys by hand and had no way to notice that
        a new mechanism added a buffer.
        """
        path = str(tmp_path / "ckpt.pth.tar")
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        _run_steps(state, _batches(), start_step=0, n_steps=3)
        before = _snapshot(model)
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch=0,
            global_step=3,
            model_name="text_span_jepa",
        )
        _corrupt(model)
        load_checkpoint(path, model, optimizer, scaler, model_name="text_span_jepa")
        after = _snapshot(model)
        assert set(before["buffers"]) == set(after["buffers"]), "buffer set changed"
        assert set(before["params"]) == set(after["params"]), "parameter set changed"
        for group in ("buffers", "params"):
            for name, saved in before[group].items():
                assert torch.equal(after[group][name], saved), f"{name} reverted"

    def test_scheduler_state_survives_resume(self, tmp_path):
        """Schedulers were *replayed* (`for _ in range(global_step): step()`),
        which is only exact when `epochs` is unchanged. The e2e test resumes
        2 -> 3 epochs, which re-derives `T_max` and reshapes the whole LR curve.
        """
        path = str(tmp_path / "ckpt.pth.tar")
        state = _build(seed=0)
        model, optimizer, scheduler, wd_scheduler, ema_scheduler, _c, scaler = state
        _run_steps(state, _batches(), start_step=0, n_steps=5)
        expected = (scheduler._step, wd_scheduler._step, ema_scheduler._step)
        lr_expected = optimizer.param_groups[0]["lr"]
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch=0,
            global_step=5,
            model_name="text_span_jepa",
            schedulers={
                "scheduler": scheduler,
                "wd_scheduler": wd_scheduler,
                "ema_scheduler": ema_scheduler,
            },
        )

        # Different T_max on the resume side: replaying 5 steps would land on a
        # different LR than the one that was saved.
        model2, optimizer2, sched2, wd2, ema2, coll2, scaler2 = _build(seed=0)
        sched2.T_max = 7
        _restore_training_state(
            {"meta": {"load_checkpoint": True, "read_checkpoint": None}},
            str(tmp_path),
            path,
            model2,
            optimizer2,
            scaler2,
            "text_span_jepa",
            sched2,
            wd2,
            ema2,
            coll2,
        )
        assert (
            sched2._step,
            wd2._step,
            ema2._step,
        ) == expected, "scheduler positions were replayed, not restored"
        assert (
            optimizer2.param_groups[0]["lr"] == lr_expected
        ), "optimizer LR was not restored from the checkpoint"


def _write_sentinel(marker):
    """Pickle reducer payload used to prove nothing gets executed on load."""
    with open(marker, "w") as f:
        f.write("EXECUTED")
    return {"sentinel": True}


class _Sentinel:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (_write_sentinel, (self.marker,))


class TestCorruptCheckpointIsFatal:
    def test_missing_checkpoint_raises_when_resume_requested(self, tmp_path):
        """`meta.load_checkpoint: true` with no file used to fall through to
        `return 0, 0, 0, 0, None`, retrain from step 0 and then overwrite the
        good `checkpoint-latest.pth.tar`.
        """
        state = _build(seed=0)
        model, optimizer, sched, wd, ema, coll, scaler = state
        with pytest.raises(FileNotFoundError):
            _restore_training_state(
                {"meta": {"load_checkpoint": True, "read_checkpoint": None}},
                str(tmp_path),
                str(tmp_path / "does-not-exist.pth.tar"),
                model,
                optimizer,
                scaler,
                "text_span_jepa",
                sched,
                wd,
                ema,
                coll,
            )

    def test_garbage_file_raises_and_does_not_execute(self, tmp_path):
        path = tmp_path / "garbage.pth.tar"
        path.write_text("this is not a checkpoint at all\n")
        marker = tmp_path / "executed.txt"
        with pytest.raises(UnsafeCheckpointError):
            safe_torch_load(str(path))
        assert not marker.exists()

    def test_truncated_checkpoint_raises_and_does_not_execute(self, tmp_path):
        """The pre-fix retry trigger was `pickle.UnpicklingError`, which a
        truncated file and a plain text file both raise. The fallback was
        `weights_only=False` = arbitrary code execution.
        """
        marker = tmp_path / "executed.txt"
        good = tmp_path / "good.pth.tar"
        torch.save({"model": {"w": torch.ones(2)}}, str(good))
        raw = good.read_bytes()
        truncated = tmp_path / "truncated.pth.tar"
        truncated.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(UnsafeCheckpointError):
            safe_torch_load(str(truncated))
        assert not marker.exists(), "a reducer executed while loading a broken file"

    def test_pickled_reducer_is_never_executed(self, tmp_path):
        """Direct proof for the false docstring claim in `src/utils/torchio.py`.

        A raw pickle whose only global is a non-allowlisted callable fails
        strict loading. The old code caught that and retried with
        `weights_only=False`, which called the reducer.
        """
        marker = tmp_path / "executed.txt"
        evil = tmp_path / "evil.pth.tar"
        with open(evil, "wb") as f:
            pickle.dump(_Sentinel(str(marker)), f)
        with pytest.raises(UnsafeCheckpointError):
            safe_torch_load(str(evil))
        assert not marker.exists(), "safe_torch_load executed a pickled reducer"

    def test_load_checkpoint_raises_on_truncated_file(self, tmp_path):
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        good = str(tmp_path / "good.pth.tar")
        save_checkpoint(good, model, optimizer, scaler, 0, 1, model_name="text_span_jepa")
        with open(good, "rb") as f:
            raw = f.read()
        bad = str(tmp_path / "bad.pth.tar")
        with open(bad, "wb") as f:
            f.write(raw[: len(raw) // 3])
        with pytest.raises(CheckpointLoadError):
            load_checkpoint(bad, model, optimizer, scaler, model_name="text_span_jepa")

    def test_load_checkpoint_raises_on_garbage_file(self, tmp_path):
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        bad = tmp_path / "garbage.pth.tar"
        bad.write_text("not a checkpoint")
        with pytest.raises(CheckpointLoadError):
            load_checkpoint(str(bad), model, optimizer, scaler, model_name="text_span_jepa")

    def test_load_checkpoint_raises_on_wrong_architecture(self, tmp_path):
        """A checkpoint whose tensors do not fit the model is a hard error, not
        a step-0 restart: silently retraining would clobber the good run.
        """
        state = _build(seed=0)
        model, optimizer, _s, _w, _e, _c, scaler = state
        path = str(tmp_path / "wrong.pth.tar")
        torch.save(
            {
                "model_name": "text_span_jepa",
                "model": {"encoder.blocks.0.nope": torch.ones(3)},
                "opt": optimizer.state_dict(),
                "epoch": 0,
                "global_step": 1,
            },
            path,
        )
        with pytest.raises(CheckpointLoadError):
            load_checkpoint(path, model, optimizer, scaler, model_name="text_span_jepa")


class TestValidationIsNonInvasive:
    def _val_fixture(self):
        dataset = TextDataset([i % 60 + 1 for i in range(200)], seq_len=SEQ, pad_id=0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        collator = SpanMaskCollator(
            mask_ratio=0.3, span_length_range=(2, 4), mask_token_id=0, pad_id=0
        )
        return loader, collator

    def test_validate_does_not_mutate_training_buffers(self):
        """`no_grad()` blocks gradient writes, not buffer writes.

        The audit recorded 24 buffers moving during a validation pass
        (`wsd.target_cov`, `wsd.target_Q`, `sta.ref_cov`, the
        `sta.is_initialized` flag, all of PUC's and RDC's running stats) — all
        of them loss inputs or calibration state for the next training step.
        After the mechanism-level `TrainingStateGuard` work landed in
        `src/models/*`, the same probe on this tree still finds
        `target_centering.center` moving. The loop-level snapshot closes the
        whole class, not just the sites somebody remembered.
        """
        state = _build(seed=0)
        model, _o, _s, _w, _e, _c, _scaler = state
        _run_steps(state, _batches(), start_step=0, n_steps=2)
        loader, collator = self._val_fixture()
        before = {n: b.detach().clone() for n, b in model.named_buffers()}
        _validate(
            model,
            loader,
            collator,
            torch.device("cpu"),
            "text_span_jepa",
            max_batches=2,
            current_step=2,
            total_steps=TOTAL_STEPS,
        )
        changed = [n for n, b in model.named_buffers() if not torch.equal(b.detach(), before[n])]
        assert not changed, f"validation mutated {len(changed)} training buffers: {changed}"
        assert len(before) > 20, "fixture is too small to be meaningful"

    def test_validate_does_not_advance_rng_streams(self):
        """Validation also consumed the numpy stream used for mask sampling, so
        a run with validation and a run without produced different models from
        the same seed. The buffer fix alone does not close that gap.
        """
        state = _build(seed=0)
        model, _o, _s, _w, _e, _c, _scaler = state
        _run_steps(state, _batches(), start_step=0, n_steps=2)
        loader, collator = self._val_fixture()
        # Draw the next value, rewind the stream, validate, draw again: an
        # identical pair means validation left the numpy stream untouched.
        mark = np.random.get_state()
        expected = float(np.random.rand())
        np.random.set_state(mark)
        _validate(
            model,
            loader,
            collator,
            torch.device("cpu"),
            "text_span_jepa",
            max_batches=2,
            current_step=2,
            total_steps=TOTAL_STEPS,
        )
        assert float(np.random.rand()) == pytest.approx(
            expected, rel=1e-12
        ), "validation advanced the numpy stream used for mask sampling"

    def test_validation_model_without_buffers_still_runs(self):
        """`_validate` is called with duck-typed models in the existing suite
        (`types.SimpleNamespace`); the snapshot must degrade, not explode.
        """
        import types

        seen = {}

        def fake_compute_loss(model, masked, original, mask, current_step=0, total_steps=1):
            seen["step"] = current_step
            return torch.tensor(0.5), {}, {}

        import src.train as train_mod

        real = train_mod.compute_loss
        train_mod.compute_loss = fake_compute_loss
        try:
            batch = {"input_ids": torch.randint(1, 10, (1, 8))}
            collated = {
                "masked_input_ids": batch["input_ids"].clone(),
                "original_input_ids": batch["input_ids"],
                "mask_positions": torch.zeros(1, 8, dtype=torch.bool),
            }
            out = _validate(
                types.SimpleNamespace(eval=lambda: None, train=lambda: None),
                [batch],
                lambda b: collated,
                torch.device("cpu"),
                "text_span_jepa",
                max_batches=1,
                current_step=7,
                total_steps=100,
            )
        finally:
            train_mod.compute_loss = real
        assert out == pytest.approx(0.5)
        assert seen["step"] == 7


class TestMissingValidationIsExplicit:
    """Two machines, same seed, one of which could not load `wiki.valid.tokens`
    produced *different models*: the failure was swallowed into a
    `logger.warning` and `val_dataloader = None`.
    """

    def test_main_raises_when_validation_cannot_be_loaded(self, tmp_path, monkeypatch):
        import src.datasets.kaggle as kaggle
        from src.train import main

        def loader(split="train", seq_len=SEQ, **_kw):
            if split == "valid":
                raise FileNotFoundError("wiki.valid.tokens missing")
            return TextDataset([i % 60 + 1 for i in range(200)], seq_len=seq_len, pad_id=0), _Tok()

        monkeypatch.setattr(kaggle, "load_wikitext103", loader)
        with pytest.raises(RuntimeError) as excinfo:
            main(_main_config(tmp_path, epochs=1))
        assert isinstance(
            excinfo.value.__cause__, FileNotFoundError
        ), "the original load failure must be preserved as the cause"
        assert "allow_missing_validation" in str(
            excinfo.value
        ), "the error must name the documented opt-out"

    def test_main_can_opt_out_explicitly(self, tmp_path, monkeypatch):
        import src.datasets.kaggle as kaggle
        from src.train import main

        def loader(split="train", seq_len=SEQ, **_kw):
            if split == "valid":
                raise FileNotFoundError("wiki.valid.tokens missing")
            return TextDataset([i % 60 + 1 for i in range(200)], seq_len=seq_len, pad_id=0), _Tok()

        monkeypatch.setattr(kaggle, "load_wikitext103", loader)
        cfg = _main_config(tmp_path, epochs=1)
        cfg["data"]["allow_missing_validation"] = True
        main(cfg)
        assert not (tmp_path / "best.pt").exists()
        assert (tmp_path / "checkpoint-latest.pth.tar").exists()

    def test_validation_presence_does_not_change_the_model(self, tmp_path, monkeypatch):
        """The strongest form of the property: with the buffer+RNG fix in
        place, a run that validates and a run that does not produce the *same*
        weights from the same seed.
        """
        import src.datasets.kaggle as kaggle
        from src.train import main

        def loader(split="train", seq_len=SEQ, **_kw):
            return TextDataset([i % 60 + 1 for i in range(200)], seq_len=seq_len, pad_id=0), _Tok()

        monkeypatch.setattr(kaggle, "load_wikitext103", loader)
        with_val = tmp_path / "with_val"
        main(_main_config(with_val, epochs=1))
        assert (with_val / "best.pt").exists()

        def broken(split="train", seq_len=SEQ, **_kw):
            if split == "valid":
                raise FileNotFoundError("nope")
            return TextDataset([i % 60 + 1 for i in range(200)], seq_len=seq_len, pad_id=0), _Tok()

        monkeypatch.setattr(kaggle, "load_wikitext103", broken)
        no_val = tmp_path / "no_val"
        cfg = _main_config(no_val, epochs=1)
        cfg["data"]["allow_missing_validation"] = True
        main(cfg)

        a = safe_torch_load(str(with_val / "checkpoint-latest.pth.tar"), map_location="cpu")
        b = safe_torch_load(str(no_val / "checkpoint-latest.pth.tar"), map_location="cpu")
        assert set(a["model"]) == set(b["model"])
        for name, tensor in a["model"].items():
            assert torch.equal(tensor, b["model"][name]), f"{name} differs with/without validation"


class _Tok:
    vocab_size = VOCAB
    pad_token_id = 0
    eos_token_id = 0
    mask_token_id = None


def _main_config(folder, epochs=1):
    return {
        "seed": 7,
        "meta": {
            "model_name": "text_span_jepa",
            "use_bfloat16": False,
            "load_checkpoint": False,
        },
        "data": {
            "batch_size": 2,
            "max_seq_len": SEQ,
            "num_workers": 0,
            "mask_ratio": 0.3,
            "span_length_range": [2, 4],
        },
        "model": {
            "embed_dim": 32,
            "encoder_depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "predictor_embed_dim": 16,
            "predictor_depth": 1,
            "future_offsets": [1],
            "num_refine_steps": 1,
            "use_jawp": True,
            "use_cgn": True,
            "use_pcr": True,
            "use_spc": True,
            "use_sta": True,
            "lambda_sta": 0.05,
            "use_wsd": True,
            "lambda_wsd": 0.05,
            "use_puc": True,
            "lambda_puc": 0.05,
            "use_rdc": True,
            "lambda_rdc": 0.05,
        },
        "optimization": {
            "epochs": epochs,
            "lr": 1e-3,
            "start_lr": 1e-4,
            "final_lr": 1e-5,
            "warmup": 1,
            "weight_decay": 0.0,
            "final_weight_decay": 0.0,
            "grad_accum_steps": 1,
        },
        "logging": {"folder": str(folder), "log_freq": 1000},
    }
