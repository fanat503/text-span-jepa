# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
# Main training loop — supports JEPA, MLM, and data2vec baselines
# Training loop patterns from I-JEPA (Assran et al., CVPR 2023):
#   - momentum_scheduler generator (I-JEPA train.py line ~152)
#   - param_groups with WD_exclude (I-JEPA helper.py init_opt)
#   - loss_fn: smooth_l1_loss (I-JEPA train.py loss_fn)
#   - target: layer_norm(h, (h.size(-1),))  (I-JEPA train.py forward_target)
#   - AMP with GradScaler (I-JEPA train.py train_step)
#   - checkpoint saving/loading pattern (I-JEPA train.py save_checkpoint)
#   - AverageMeter, CSVLogger (I-JEPA src/utils/logging.py)

import functools
import importlib.util
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

from src.utils.logging import AverageMeter, CSVLogger
from src.utils.seed import seed_everything, worker_init_fn
from src.utils.torchio import safe_torch_load

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ═══════════════════════════════════════════════════════════════════
#  Deep merge for defaults + experiment config
# ═══════════════════════════════════════════════════════════════════


def _deep_merge(base, override):
    """Recursively merge override dict into base dict.

    Lists and scalars from override replace base values.
    Nested dicts are merged recursively.
    This enables ablation configs that only specify mechanism flags
    to inherit all other settings from defaults.yaml.
    """
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ═══════════════════════════════════════════════════════════════════
#  Arm registry — the one place that says what an arm IS
# ═══════════════════════════════════════════════════════════════════

# The three call signatures `compute_loss` has to choose between. They are named
# rather than inferred, and that is the whole point of this section.
#
# WHAT WENT WRONG BEFORE
# ----------------------
# The dispatch asked, in order, "does the model have `compute_loss_with_targets`?"
# then "does it have `forward` AND an attribute named `regression_head`?", and only
# then fell through to `compute_loss`. Both earlier branches are POSITIVE tests for
# an incidental attribute name, so the branch an arm landed in was a property of
# what it happened to be called rather than of what it is. `regression_head` is in
# particular the obvious name for a regression-style head, so an arm that adopted
# it would have been handed data2vec's call signature -- and, because
# `MLMBaseline` is the arm that reaches the `compute_loss` branch, the four SSL
# baselines would all have been routed through whatever the FIRST matching arm
# was. They happened not to own the name, so nothing broke. That is luck, and the
# next arm to grow a head called `regression_head` inherits a silent mis-route.
#
# The protocol is now DECLARED per arm, in one table, by `create_model`. There is
# exactly one call site per protocol regardless of how many arms share it.

LOSS_JEPA_TARGETS = "jepa_targets"  # compute_loss_with_targets(a, b, mask, **) -> 3-tuple
LOSS_DATA2VEC_FORWARD = "data2vec_forward"  # model(a, b, mask) -> (loss, info)
LOSS_COMPUTE_LOSS = "compute_loss"  # model.compute_loss(a, b, mask) -> (loss, info)

#: Canonical arm name (what `_normalize_model_name` returns) -> declared protocol.
#:
#: Keys must be exactly the set `_normalize_model_name` can return for an arm
#: `create_model` builds. `tests/test_ssl_baselines.py::TestTrainerContract` pins
#: that correspondence in both directions, so an arm added to `create_model`
#: without an entry here is a failing test rather than a lucky duck-type.
LOSS_PROTOCOLS = {
    "text_span_jepa": LOSS_JEPA_TARGETS,
    "data2vec": LOSS_DATA2VEC_FORWARD,
    "mlm": LOSS_COMPUTE_LOSS,
    "byol": LOSS_COMPUTE_LOSS,
    "barlow": LOSS_COMPUTE_LOSS,
    "vicreg": LOSS_COMPUTE_LOSS,
    "simsiam": LOSS_COMPUTE_LOSS,
}

#: Arms that own an EMA teacher the trainer advances through a zero-argument
#: `model.update_target_encoder()`, and that therefore need no `tau`.
#:
#: `text_span_jepa` is deliberately absent: its teacher is driven by the
#: `EMATauSchedule` and takes `tau` as an argument, so it is a different call.
SELF_EMA_ARMS = frozenset({"data2vec", "byol"})

#: Canonical prefix -> canonical arm name, used by `_normalize_model_name`. Held
#: as a table rather than an if-chain so the prefixes and `LOSS_PROTOCOLS` can be
#: checked against each other by a test instead of by reading.
_ARM_PREFIXES = {
    "text_span_jepa": "text_span_jepa",
    "jepa": "text_span_jepa",
    "mlm": "mlm",
    "data2vec": "data2vec",
    "byol": "byol",
    "barlow": "barlow",
    "vicreg": "vicreg",
    "simsiam": "simsiam",
}


def _normalize_model_name(raw_name):
    """Normalize model_name from config to canonical form.

    Configs may use suffixed names like 'text_span_jepa_small',
    'mlm_small', 'data2vec_base'. We strip the suffix to get
    the canonical name that create_model() understands.

    Canonical names: the keys of `LOSS_PROTOCOLS` -- text_span_jepa, mlm,
    data2vec, byol, barlow, vicreg, simsiam.

    A name that matches no prefix is returned as-is and fails in `create_model`
    with the list of what IS supported.
    """
    name = raw_name.strip().lower()
    for prefix, canonical in _ARM_PREFIXES.items():
        if name.startswith(prefix):
            return canonical
    return name  # Return as-is, will fail in create_model with clear error


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint save/load — I-JEPA pattern, all model types
# ═══════════════════════════════════════════════════════════════════


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded into the given model.

    Deliberately fatal. The previous implementation wrapped the whole load in
    `except Exception` and returned `(0, 0, 0, 0, None)`, which turned a corrupt
    or wrong-architecture checkpoint into "train from step 0" — and the run then
    overwrote the good `checkpoint-latest.pth.tar` with a step-0 model. A
    checkpoint the loader cannot understand is a stop-the-line event.
    """


def _capture_rng_state():
    """Snapshot every random stream that training consumes.

    Without this, a resume is a different experiment: mask sampling
    (`src/masks/span.py` draws from `np.random`), DropPath, the CGN Gumbel,
    CMC's second mask and the data-shuffle stream all restart from scratch.
    Measured on the toy fixture before the fix: the first post-resume loss was
    29.6% away from the uninterrupted run.

    The numpy state is stored as plain Python ints rather than its native
    5-tuple, because the native form contains an `ndarray` whose
    `numpy._core.multiarray._reconstruct` GLOBAL is *not* in torch's
    `weights_only` allowlist — storing it raw makes the whole checkpoint
    unloadable under the strict loader.
    """
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "python_random": random.getstate(),
        "numpy": (name, [int(k) for k in keys], int(pos), int(has_gauss), float(cached)),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    """Inverse of `_capture_rng_state`.

    A missing key is a no-op rather than an error: a checkpoint written before
    this fix simply has no `rng_state` entry and is still readable.
    """
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy" in state:
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state(
            (name, np.array(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached)),
        )
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _scheduler_state(scheduler):
    """Portable snapshot of a schedule's position.

    The three schedules in `src/utils/schedulers.py` predate `state_dict()` —
    they are plain classes holding a `_step` counter — so this falls back to the
    instance attributes that are not the optimizer reference. A
    `torch.optim.lr_scheduler` object takes the `state_dict()` branch.
    """
    if scheduler is None:
        return None
    if hasattr(scheduler, "state_dict"):
        return scheduler.state_dict()
    return {k: v for k, v in vars(scheduler).items() if k != "optimizer"}


def _load_scheduler_state(scheduler, state):
    """Inverse of `_scheduler_state`. Returns True when the state was applied."""
    if scheduler is None or not state:
        return False
    if hasattr(scheduler, "load_state_dict") and hasattr(scheduler, "state_dict"):
        scheduler.load_state_dict(state)
        return True
    applied = False
    for key, value in state.items():
        if hasattr(scheduler, key):
            setattr(scheduler, key, value)
            applied = True
    return applied


def _mechanism_extras(model):
    """Collect `checkpoint_dict()` payloads from mechanisms that provide one.

    `src/models/rdc.py` and `src/models/puc.py` already implement complete
    `checkpoint_dict()` / `load_checkpoint()` pairs covering buffers that the
    hand-rolled key list in this file never mentioned
    (`rdc.running_ortho_drift_norm`, `rdc.running_workspace_drift_norm`,
    `puc.running_entropy`, `puc.running_overconfidence`). All of them are also
    registered buffers, so `model.state_dict()` already carries them; storing
    the helper output as well means a future field added to a mechanism helper
    is checkpointed without touching this file.
    """
    extras = {}
    for name in ("rdc", "puc"):
        mech = getattr(model, name, None)
        if mech is not None and hasattr(mech, "checkpoint_dict"):
            extras[name] = mech.checkpoint_dict()
    return extras


def _restore_mechanism_extras(model, extras):
    for name, payload in (extras or {}).items():
        mech = getattr(model, name, None)
        if mech is not None and hasattr(mech, "load_checkpoint"):
            mech.load_checkpoint(payload)


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    ema_step=0,
    mask_step=0,
    extra_state=None,
    model_name="text_span_jepa",
    schedulers=None,
):
    """Save complete training state for resumption — all model types.

    Handles JEPA (encoder + predictor + target_encoder + decoder),
    MLM (encoder + mlm_head), and data2vec (encoder + target_encoder + regression_head).

    Fidelity contract — a resume must be indistinguishable from an uninterrupted
    run, so the payload carries:

    * `model.state_dict()`: the *whole* module, not a hand-enumerated list of
      48 keys. The enumeration silently dropped 26 tensors on the toy fixture
      (14 of them trainable parameters) because nobody remembered to update it
      when a mechanism grew a tensor.
    * `optimizer.state_dict()`: already includes `param_groups`, so the live LR
      and weight decay survive.
    * `rng_state`: python `random`, numpy and torch streams (plus CUDA).
    * `schedulers`: LR / weight-decay / EMA-tau positions, so a resume need not
      replay `global_step` scheduler calls — which was only exact when `epochs`
      was unchanged, and is O(global_step) wasted work besides.
    * `mechanism_extras`: `rdc.checkpoint_dict()` / `puc.checkpoint_dict()`.
    """
    model_name = _normalize_model_name(model_name)
    state = {
        "model_name": model_name,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "ema_step": ema_step,
        "mask_step": mask_step,
        "rng_state": _capture_rng_state(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    if model_name == "text_span_jepa":
        state["mechanism_extras"] = _mechanism_extras(model)
    elif model_name in SELF_EMA_ARMS and hasattr(model, "num_updates"):
        # A self-managed teacher counts its own updates. `num_updates` is a plain
        # int, not a tensor, so it is NOT in `model.state_dict()` and a resume
        # would silently rewind it to 0. For BYOL this is currently cosmetic
        # (its momentum is constant, so the counter feeds nothing); for data2vec
        # it is not, which is why the key already existed.
        state["num_updates"] = model.num_updates

    if schedulers:
        saved = {name: _scheduler_state(s) for name, s in schedulers.items()}
        state["schedulers"] = {k: v for k, v in saved.items() if v is not None}

    if extra_state is not None:
        state["extra"] = extra_state
    torch.save(state, path)


def _load_legacy_jepa_state(model, checkpoint):
    """Restore a pre-`state_dict` checkpoint (the old 48-key hand-rolled format).

    Kept so a checkpoint written by an earlier commit of this repo can still be
    read. It is lossy by construction — every tensor it never named is gone,
    including `sta.ref_cov`, `wsd.target_cov` and `puc.running_entropy` — which
    is exactly why new writes use `model.state_dict()`.
    """
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.predictor.load_state_dict(checkpoint["predictor"])
    model.target_encoder.load_state_dict(checkpoint["target_encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    if "target_centering_center" in checkpoint and hasattr(model, "target_centering"):
        model.target_centering.center.copy_(checkpoint["target_centering_center"])

    # (checkpoint key, dotted attribute path, needs .data indirection)
    tensor_paths = [
        ("jawp_workspace_Q", "jawp.workspace_Q", True),
        ("jawp_active_k", "jawp.active_k", False),
        ("cgn_gate_logits_visible", "cgn.gate_logits_visible", True),
        ("cgn_total_steps", "cgn.total_steps", False),
        ("pcr_workspace_Q", "pcr.workspace_Q", True),
        ("spc_freq_basis", "spc.freq_basis", True),
        ("spc_log_band_weights", "spc.log_band_weights", True),
        ("spc_running_residual_vars", "spc.running_residual_vars", False),
        ("spc_running_predictability", "spc.running_predictability", False),
        ("wsd_running_drift", "wsd.running_drift", False),
        ("wsd_is_initialized", "wsd.is_initialized", False),
        ("wsd_step_count", "wsd.step_count", False),
        ("cmc_running_consistency", "cmc.running_consistency", False),
        ("cmc_running_overlap_ratio", "cmc.running_overlap_ratio", False),
        ("cmc_total_cmc_steps", "cmc.total_cmc_steps", False),
        ("gac_running_grad_norms", "gac.running_grad_norms", False),
        ("gac_running_starved_fraction", "gac.running_starved_fraction", False),
        ("gac_total_gac_steps", "gac.total_gac_steps", False),
        ("sta_running_w1", "sta.running_w1", False),
        ("sta_running_spectral_gap", "sta.running_spectral_gap", False),
        ("sta_is_initialized", "sta.is_initialized", False),
        ("sta_step_count", "sta.step_count", False),
        ("puc_running_mean", "puc.running_mean", False),
        ("puc_running_eigenvalues", "puc.running_eigenvalues", False),
        ("puc_proj_vectors", "puc.proj_vectors", False),
        ("puc_total_steps", "puc.total_steps", False),
        ("rdc_z_previous", "rdc.z_previous", False),
        ("rdc_workspace_Q", "rdc.workspace_Q", False),
        ("rdc_running_drift_norm", "rdc.running_drift_norm", False),
        ("rdc_running_drift_ratio", "rdc.running_drift_ratio", False),
        ("rdc_total_steps", "rdc.total_steps", False),
        ("wsr_running_sharpness", "wsr.running_sharpness", False),
        ("wsr_running_spectral_sharpness", "wsr.running_spectral_sharpness", False),
        (
            "wsr_running_directional_sharpness",
            "wsr.running_directional_sharpness",
            False,
        ),
        ("wsr_running_grad_norm", "wsr.running_grad_norm", False),
        ("wsr_total_steps", "wsr.total_steps", False),
    ]
    for key, path, is_param in tensor_paths:
        if key not in checkpoint:
            continue
        target = model
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part, None)
            if target is None:
                break
        if target is None:
            continue
        leaf = getattr(target, parts[-1], None)
        if leaf is None:
            continue
        (leaf.data if is_param else leaf).copy_(checkpoint[key])

    if "pcr_level_gates" in checkpoint and getattr(model, "pcr", None) is not None:
        for i, gate in enumerate(checkpoint["pcr_level_gates"]):
            if i < len(model.pcr.level_gates):
                model.pcr.level_gates[i].data.copy_(gate)


def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    model_name="text_span_jepa",
    schedulers=None,
    report=None,
):
    """Load checkpoint — I-JEPA helper.py load_checkpoint pattern.

    Handles all model types. Returns (epoch, global_step, ema_step, mask_step, extra_state)

    Raises `CheckpointLoadError` (or the underlying `FileNotFoundError`) instead
    of returning zeros. Silently "recovering" from a failed load is what let a
    corrupt checkpoint destroy the previous good one.
    """
    model_name = _normalize_model_name(model_name)
    try:
        checkpoint = safe_torch_load(path, map_location=torch.device("cpu"))
    except FileNotFoundError:
        raise  # the caller's resume path names the file it expected
    except Exception as e:
        # Includes UnsafeCheckpointError: a file the strict loader refuses is a
        # stop-the-line event, not a reason to retrain from step 0.
        raise CheckpointLoadError(f"Could not read checkpoint {path}: {e}") from e

    if not isinstance(checkpoint, dict):
        raise CheckpointLoadError(
            f"Checkpoint {path} holds a {type(checkpoint).__name__}, not a " f"training-state dict",
        )

    missing = [k for k in ("epoch", "global_step", "opt") if k not in checkpoint]
    if missing:
        raise CheckpointLoadError(
            f"Checkpoint {path} is missing required keys {missing}; "
            f"it holds {sorted(checkpoint)[:8]}...",
        )

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    ema_step = checkpoint.get("ema_step", 0)
    mask_step = checkpoint.get("mask_step", 0)

    ckpt_model_name = _normalize_model_name(checkpoint.get("model_name", model_name))
    if ckpt_model_name != model_name:
        raise CheckpointLoadError(
            f"Checkpoint {path} holds a {ckpt_model_name!r} model but the config "
            f"asks for {model_name!r}. Refusing to load: training from step 0 would "
            f"overwrite the existing run.",
        )

    try:
        if "model" in checkpoint:
            # Canonical path: the complete module state_dict, strict so that a
            # shape or architecture mismatch is loud rather than partial.
            model.load_state_dict(checkpoint["model"], strict=True)
            if ckpt_model_name == "text_span_jepa":
                _restore_mechanism_extras(model, checkpoint.get("mechanism_extras"))
            elif (
                ckpt_model_name in SELF_EMA_ARMS
                and hasattr(model, "num_updates")
                and "num_updates" in checkpoint
            ):
                # Inverse of `save_checkpoint`. A checkpoint written before the key
                # existed is NOT an error: `num_updates` simply stays at 0, which is
                # the behaviour these arms have always had.
                model.num_updates = checkpoint["num_updates"]
        elif ckpt_model_name == "text_span_jepa":
            logger.warning(
                f"Checkpoint {path} predates full-state_dict writes; restoring the "
                f"legacy 48-key format. Tensors it never named (sta.ref_cov, "
                f"wsd.target_cov, puc.running_entropy, ...) are lost — do not resume "
                f"an old run for real.",
            )
            _load_legacy_jepa_state(model, checkpoint)
        elif ckpt_model_name == "mlm":
            model.encoder.load_state_dict(checkpoint["encoder"])
            model.mlm_head.load_state_dict(checkpoint["mlm_head"])
        elif ckpt_model_name == "data2vec":
            model.encoder.load_state_dict(checkpoint["encoder"])
            model.target_encoder.load_state_dict(checkpoint["target_encoder"])
            model.regression_head.load_state_dict(checkpoint["regression_head"])
            if "num_updates" in checkpoint and hasattr(model, "num_updates"):
                model.num_updates = checkpoint["num_updates"]

        optimizer.load_state_dict(checkpoint["opt"])

        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        if schedulers:
            saved_schedulers = checkpoint.get("schedulers")
            applied = False
            for name, scheduler in schedulers.items():
                applied |= _load_scheduler_state(scheduler, (saved_schedulers or {}).get(name))
            if report is not None:
                report["schedulers_restored"] = applied
            if not applied:
                logger.warning(
                    "Checkpoint has no scheduler state; schedule positions must be "
                    "replayed, which is only exact if `epochs` is unchanged.",
                )

        # Last, deliberately: the RNG streams are what make the *next* step
        # identical, so nothing after this point may consume randomness.
        _restore_rng_state(checkpoint.get("rng_state"))
    except CheckpointLoadError:
        raise
    except Exception as e:
        raise CheckpointLoadError(
            f"Could not restore {model_name} from checkpoint {path}: " f"{type(e).__name__}: {e}",
        ) from e

    extra_state = checkpoint.get("extra", None)
    logger.info(
        f"Loaded checkpoint: epoch={epoch}, step={global_step}, "
        f"ema_step={ema_step}, mask_step={mask_step}",
    )
    return epoch, global_step, ema_step, mask_step, extra_state


# ═══════════════════════════════════════════════════════════════════
#  The four SSL arms — BYOL, Barlow Twins, VICReg, SimSiam
# ═══════════════════════════════════════════════════════════════════


def _load_byol():
    from baselines.byol_baseline import BYOLBaseline

    return BYOLBaseline


def _load_barlow():
    from baselines.barlow_baseline import BarlowTwinsBaseline

    return BarlowTwinsBaseline


def _load_vicreg():
    from baselines.vicreg_baseline import VICRegBaseline

    return VICRegBaseline


def _load_simsiam():
    from baselines.simsiam_baseline import SimSiamBaseline

    return SimSiamBaseline


#: Canonical arm name -> zero-argument class loader.
#:
#: Resolved through callables rather than imported at module scope for the same
#: reason every other `create_model` branch imports inside its own `elif`: a bare
#: `import src.train` must not drag in four baseline modules and their encoder.
SSL_BASELINE_ARMS = {
    "byol": _load_byol,
    "barlow": _load_barlow,
    "vicreg": _load_vicreg,
    "simsiam": _load_simsiam,
}

#: Which head submodules each SSL arm puts in the optimizer, in order.
#:
#: BYOL is the only one with a teacher; `target_encoder` and `target_projector`
#: are deliberately absent, and that is not an oversight -- they are frozen
#: (`requires_grad = False`), so a parameter in the optimizer that can never
#: receive a gradient is a parameter the optimizer is silently carrying.
#: `Data2VecTextBaseline` is handled the same way, which is the precedent.
_SSL_HEAD_ATTRS = {
    "byol": ("projector", "predictor"),
    "barlow": ("projector",),
    "vicreg": ("projector", "predictor"),
    "simsiam": ("projector", "predictor"),
}


# ═══════════════════════════════════════════════════════════════════
#  Model creation factory
# ═══════════════════════════════════════════════════════════════════


def create_model(model_name, model_cfg, vocab_size, max_seq_len, device):
    """Create model by type — supports jepa, mlm, data2vec and four SSL arms.

    model_name is automatically normalized from config values like
    'text_span_jepa_small' -> 'text_span_jepa'.
    """
    model_name = _normalize_model_name(model_name)

    if model_name == "text_span_jepa":
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        full_cfg = {**model_cfg, "vocab_size": vocab_size, "max_seq_len": max_seq_len}
        config = TextSpanJEPAConfig(**full_cfg)
        config.validate()  # Catch dimension errors early
        model = TextSpanJEPA(config).to(device)
    elif model_name == "mlm":
        from baselines.mlm_baseline import MLMBaseline

        model = MLMBaseline(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=model_cfg.get("embed_dim", 768),
            depth=model_cfg.get("encoder_depth", 12),
            num_heads=model_cfg.get("num_heads", 12),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            drop_rate=model_cfg.get("drop_rate", 0.1),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        ).to(device)
        # Add a .config attribute for compatibility
        model.config = type(
            "Cfg",
            (),
            {
                "lambda_decoder": 0.1,
                "lambda_variance": 0.1,
                "lambda_covariance": 0.04,
                "lambda_span": 1.0,
                "lambda_future": 0.5,
            },
        )()
    elif model_name == "data2vec":
        from baselines.data2vec_baseline import Data2VecTextBaseline

        model = Data2VecTextBaseline(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=model_cfg.get("embed_dim", 768),
            depth=model_cfg.get("encoder_depth", 12),
            num_heads=model_cfg.get("num_heads", 12),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            drop_rate=model_cfg.get("drop_rate", 0.0),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
            average_top_k_layers=model_cfg.get("average_top_k_layers", 8),
            loss_beta=model_cfg.get("loss_beta", 0.0),
            loss_scale=model_cfg.get("loss_scale", None),
            ema_decay=model_cfg.get("ema_decay", 0.999),
            ema_end_decay=model_cfg.get("ema_end_decay", 0.9999),
            ema_anneal_end_step=model_cfg.get("ema_anneal_end_step", 100000),
            head_layers=model_cfg.get("head_layers", 2),
        ).to(device)
        model.config = type(
            "Cfg",
            (),
            {
                "lambda_decoder": 0.1,
                "lambda_variance": 0.1,
                "lambda_covariance": 0.04,
                "lambda_span": 1.0,
                "lambda_future": 0.5,
            },
        )()
    elif model_name in SSL_BASELINE_ARMS:
        # One branch for all four, because they are constructed from the SAME
        # arguments: the shared `TextSpanJEPAEncoder` at the config's shape plus
        # nothing else. Their head widths come from each module's
        # `HIDDEN_MULTIPLE * embed_dim`, which is what lands them on JEPA's
        # trainable count, so no width needs to be passed in for the parameter
        # matching to hold.
        #
        # No METHOD hyperparameter is read here on purpose. Every method
        # hyperparameter these arms take (`target_momentum`, `lambda_offdiag`,
        # `sim_weight` / `var_weight` / `cov_weight` / `gamma`) already has its
        # published-paper value as the constructor default, so an arm built from
        # `defaults.yaml` alone trains the method as published. Making them
        # settable is a `defaults.yaml` change -- which also means new entries in
        # `_warn_unknown_config_keys`' `extra_known` and in
        # `tests/test_config_system.py::_TRAINER_EXTRA_KNOWN`, which is kept in
        # step with the trainer by
        # `test_trainer_extra_known_matches_the_trainer`. That is deliberately
        # NOT done here: exempting a key that no shipped config uses is a
        # permanent hole in the typo detector, bought for a knob that has a
        # correct default.
        #
        # `drop_path_rate` is a different category and IS read. It is not part
        # of any of the four methods -- it is a property of the shared
        # `TextSpanJEPAEncoder` trunk, which every arm in this repo builds. This
        # branch used to omit it while the JEPA branch honoured it, so at the
        # reference config the JEPA column ran stochastic depth and all six
        # baseline columns ran `nn.Identity`: the row then measured the method
        # AND a regulariser, which is not a comparison of methods. Omitting it
        # here is exactly the asymmetry of the `drop_rate` argument four lines
        # up, which IS read. Pinned by
        # `tests/test_ssl_baselines.py::TestRegularizationParity`, which reads
        # the built modules rather than the constructor signature because these
        # classes swallow unknown keywords in `**kwargs`.
        model = SSL_BASELINE_ARMS[model_name]()(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=model_cfg.get("embed_dim", 768),
            depth=model_cfg.get("encoder_depth", 12),
            num_heads=model_cfg.get("num_heads", 12),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            drop_rate=model_cfg.get("drop_rate", 0.0),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        ).to(device)
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. " f"Supported: {', '.join(sorted(LOSS_PROTOCOLS))}",
        )

    # Stamp the DECLARED loss protocol so `compute_loss` dispatches on what this
    # arm IS rather than on which attribute names it happens to own. A table
    # lookup, so an arm added to the branches above without a `LOSS_PROTOCOLS`
    # entry raises here instead of being routed by accident at step 0.
    model.loss_protocol = LOSS_PROTOCOLS[model_name]
    return model


def _infer_loss_protocol(model):
    """Best-effort protocol for a model `create_model` did not build.

    This is the OLD duck-typed chain, kept verbatim and narrowed to returning one
    of the three protocol constants. It exists for exactly two callers: models
    constructed directly by a caller or a test (`tests/test_model.py` builds a
    `TextSpanJEPA`, an `MLMBaseline` and a `Data2VecTextBaseline` by hand and hands
    them to `compute_loss`), and any future arm wired in without going through
    `create_model`.

    It is a COMPATIBILITY path, not the dispatch: nothing `create_model` builds is
    ever routed by it, so the trap it encodes -- an arm silently picked up by the
    `regression_head` test -- cannot reach a training run. Returns None when no
    protocol applies, which `compute_loss` turns into the same ValueError as
    before.
    """
    if hasattr(model, "compute_loss_with_targets"):
        return LOSS_JEPA_TARGETS
    if hasattr(model, "forward") and hasattr(model, "regression_head"):
        return LOSS_DATA2VEC_FORWARD
    if hasattr(model, "compute_loss"):
        return LOSS_COMPUTE_LOSS
    return None


def compute_loss(
    model,
    masked_input_ids,
    original_input_ids,
    mask_positions,
    current_step=0,
    total_steps=1,
):
    """Compute loss for any model type — unified interface.

    Always returns (total_loss, loss_dict, diag_dict) for consistency.

    Dispatches on the DECLARED protocol (`model.loss_protocol`, stamped by
    `create_model`), not on duck-typing. See the arm registry at the top of this
    file for what the old order-based chain cost. The signature is unchanged
    because two tests replace this function outright with a five-argument stub
    (`tests/test_grad_scaler.py`, `tests/test_checkpoint_fidelity.py`) and the
    production call sites therefore cannot grow a `model_name=` keyword.
    """
    protocol = getattr(model, "loss_protocol", None)
    if protocol is None:
        protocol = _infer_loss_protocol(model)

    if protocol == LOSS_JEPA_TARGETS:
        # JEPA model — returns (loss, loss_dict, diag_dict)
        return model.compute_loss_with_targets(
            masked_input_ids,
            original_input_ids,
            mask_positions,
            current_step=current_step,
            total_steps=total_steps,
        )
    if protocol == LOSS_DATA2VEC_FORWARD:
        # data2vec — returns (loss, info_dict)
        loss, info = model(masked_input_ids, original_input_ids, mask_positions)
        return loss, info, {}
    if protocol == LOSS_COMPUTE_LOSS:
        # MLM and the four SSL arms — returns (loss, info_dict)
        loss, info = model.compute_loss(masked_input_ids, original_input_ids, mask_positions)
        return loss, info, {}
    raise ValueError(f"Model {type(model).__name__} has no supported loss method")


def get_param_groups(model, model_name, wd=0.04):
    """Build optimizer param groups with WD_exclude for bias/norm.

    Audit R18 headline fix: the text_span_jepa branch previously covered ONLY
    encoder/predictor/decoder — every novel-mechanism parameter (jawp
    workspace_Q, cgn gates, pcr cascade, spc basis, ...) was invisible to the
    optimizer and therefore frozen for the entire history of the repo.
    A final catch-all group now adopts any remaining trainable parameter,
    weight-decay-excluded to preserve manifold semantics (retraction owns
    the geometry of Q-like parameters).
    """
    if model_name.startswith(("text_span_jepa", "jepa")):
        groups = [
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ],
            },
            {
                "params": [
                    p
                    for n, p in model.predictor.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ],
            },
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ],
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {
                "params": [
                    p
                    for n, p in model.predictor.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ],
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {"params": list(model.decoder.parameters()), "weight_decay": wd},
        ]

        covered = {id(p) for g in groups for p in g["params"]}
        mechanism_params = [
            p for n, p in model.named_parameters() if id(p) not in covered and p.requires_grad
        ]
        if mechanism_params:
            groups.append({"params": mechanism_params, "WD_exclude": True, "weight_decay": 0})
        return groups
    elif model_name == "mlm":
        # MLM: all encoder params + mlm_head
        return [
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ],
            },
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ],
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {"params": list(model.mlm_head.parameters()), "weight_decay": wd},
        ]
    elif model_name == "data2vec":
        # data2vec: encoder + regression_head (target encoder is EMA)
        return [
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ],
            },
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ],
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {"params": list(model.regression_head.parameters()), "weight_decay": wd},
        ]
    elif model_name in SSL_BASELINE_ARMS:
        # Encoder with the repo's WD_exclude split, then each head at full WD.
        # The split is the same one every other arm gets: weight decay on
        # LayerNorm weights and biases is a known optimiser defect, and letting
        # these four arms fall through to the `else` catch-all below would apply
        # `weight_decay=0.04` to EVERY parameter -- a baseline trained that way is
        # not a faithful baseline, and the defect would be invisible because the
        # arms still converge.
        #
        # BYOL's frozen teacher is excluded by `_SSL_HEAD_ATTRS` naming only the
        # online heads; see that table.
        groups = [
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ],
            },
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ],
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]
        for attr in _SSL_HEAD_ATTRS[model_name]:
            groups.append({"params": list(getattr(model, attr).parameters()), "weight_decay": wd})
        return groups
    else:
        return [{"params": list(model.parameters())}]


def do_ema_update(model, model_name, tau=None):
    """Perform EMA update of target encoder — works for all model types.

    model_name is automatically normalized.
    For JEPA: uses scheduled tau from EMATauSchedule.
    For data2vec and BYOL: uses the model's own zero-argument update, which
    reads its own annealed momentum internally.
    """
    model_name = _normalize_model_name(model_name)

    if model_name == "text_span_jepa":
        if tau is not None:
            model.update_target_encoder(tau)
    elif model_name in SELF_EMA_ARMS:
        model.update_target_encoder()
    # MLM has no EMA target — no-op


def _get_all_trainable_params(model):
    """Get all trainable parameters as a single list for global grad clipping."""
    return [p for p in model.parameters() if p.requires_grad]


# ═══════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════


def _dataloader_worker_init(worker_id, base_seed=42):
    """Picklable DataLoader `worker_init_fn`.

    Must live at module scope. The previous
    `lambda wid: worker_init_fn(wid, seed)` is a closure, and the host start
    method here is `spawn`, which pickles `worker_init_fn` to send it to the
    worker: `PicklingError: Can't pickle local object`. With the shipped
    `data.num_workers: 2`, `python -m src.train` therefore could not start its
    own dataloader workers on Windows. The suite never caught it because
    `tests/test_training_e2e.py` sets `num_workers: 0`.

    Also seeds `torch`, which `src.utils.seed.worker_init_fn` does not: every
    worker inherits the parent's torch RNG seed, so worker 0 and worker 1 draw
    the *same* random stream. That matters as soon as any tensor-side
    randomness (DropPath, Gumbel) happens inside a worker.

    Note `persistent_workers=True` in `make_dataloader` means this runs once
    per worker *lifetime*, not once per epoch.
    """
    worker_seed = int(base_seed) + int(worker_id)
    worker_init_fn(worker_id, base_seed)
    torch.manual_seed(worker_seed)


def _worker_init_for(base_seed):
    """Bind the seed once; the returned partial is picklable, a lambda is not."""
    return functools.partial(_dataloader_worker_init, base_seed=int(base_seed))


def _build_data_pipeline(args, seed):
    """Load dataset(s) and construct train/validation loaders.

    Returns: (dataloader, val_dataloader, tokenizer, mask_token_id, seq_len).

    A missing validation set is a hard error unless
    `data.allow_missing_validation: true` is set explicitly. It used to be
    swallowed into a `logger.warning` with `val_dataloader = None`, which meant
    two machines with the same seed produced *different models* — one of them
    silently, and the difference was only visible in `best_val_loss`.
    """
    data_cfg = args.get("data", {})
    seq_len = data_cfg.get("max_seq_len", 512)
    allow_missing_validation = bool(data_cfg.get("allow_missing_validation", False))

    logger.info("Loading dataset...")
    from src.datasets.kaggle import get_mask_token_id, load_wikitext103, make_dataloader

    dataset, tokenizer = load_wikitext103(
        tokenizer_name=data_cfg.get("tokenizer", "gpt2"),
        seq_len=seq_len,
        split="train",
        data_dir=data_cfg.get("root_path", "/kaggle/input/wikitext-103"),
    )
    mask_token_id = get_mask_token_id(tokenizer)
    init_fn = _worker_init_for(seed)

    # Validation set
    try:
        val_dataset, _ = load_wikitext103(
            tokenizer_name=data_cfg.get("tokenizer", "gpt2"),
            seq_len=seq_len,
            split="valid",
            data_dir=data_cfg.get("root_path", "/kaggle/input/wikitext-103"),
        )
        val_dataloader = make_dataloader(
            val_dataset,
            batch_size=data_cfg.get("batch_size", 64),
            num_workers=data_cfg.get("num_workers", 2),
            shuffle=False,
            worker_init_fn=init_fn,
        )
    except Exception as e:
        if not allow_missing_validation:
            raise RuntimeError(
                f"Could not build the validation set ({type(e).__name__}: {e}). "
                f"Training without validation changes the result — a different "
                f"number of validation passes means a different model from the "
                f"same seed — so it is refused. Fix the data path, or set "
                f"data.allow_missing_validation: true in the config to accept it "
                f"deliberately.",
            ) from e
        logger.error(
            f"Validation unavailable ({type(e).__name__}: {e}) and "
            f"data.allow_missing_validation is true — training WITHOUT validation. "
            f"best.pt will not be written and these results are not comparable to "
            f"runs that do validate.",
        )
        val_dataloader = None

    dataloader = make_dataloader(
        dataset,
        batch_size=data_cfg.get("batch_size", 64),
        num_workers=data_cfg.get("num_workers", 2),
        worker_init_fn=init_fn,
    )
    return dataloader, val_dataloader, tokenizer, mask_token_id, seq_len


def _build_optimization(args, model, model_name, model_cfg, device, ipe):
    """Build optimizer, AMP scaler and LR/WD/EMA schedules.

    Returns: (num_epochs, grad_accum_steps, optimizer, use_bfloat16, scaler,
              total_steps, scheduler, wd_scheduler, ema_scheduler).
    """
    opt_cfg = args.get("optimization", {})
    grad_accum_steps = opt_cfg.get("grad_accum_steps", 1)  # For OOM on small GPUs
    param_groups = get_param_groups(model, model_name, wd=opt_cfg.get("weight_decay", 0.04))
    optimizer = torch.optim.AdamW(param_groups)

    # AMP: respect device availability
    use_bfloat16 = args.get("meta", {}).get("use_bfloat16", True) and device.type == "cuda"
    # bf16 (the only AMP mode in this repo) needs no loss scaling; fp16 is unused.
    # A disabled GradScaler is a pass-through: scale()=identity, get_scale()=1.0,
    # step()=optimizer.step(), state_dict()={} — so clip_grad_norm_ below sees
    # UNSCALED gradients (fixes the effective-clip ~1e-5 bug) and checkpoint
    # plumbing is unchanged in both directions.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    num_epochs = opt_cfg.get("epochs", 50)
    total_steps = int(opt_cfg.get("ipe_scale", 1.0) * num_epochs * ipe)

    from src.utils.schedulers import CosineWDSchedule, EMATauSchedule, WarmupCosineSchedule

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(opt_cfg.get("warmup", 5) * ipe),
        start_lr=opt_cfg.get("start_lr", 1e-4),
        ref_lr=opt_cfg.get("lr", 1e-3),
        final_lr=opt_cfg.get("final_lr", 1e-5),
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=opt_cfg.get("weight_decay", 0.04),
        final_wd=opt_cfg.get("final_weight_decay", 0.4),
        T_max=total_steps,
    )
    # EMA schedule — only for JEPA models.
    #
    # Both fallbacks MUST equal the values defaults.yaml declares and the ones
    # TextSpanJEPAConfig / EMATauSchedule already default to. A key missing on a
    # `--no_defaults` run has to land on the same schedule as the merged run, or
    # the two paths silently train different models.
    #
    # tau_end was `1.0`, which is not a placeholder. EMATauSchedule.step()
    # returns exactly tau_end, so at tau == 1.0 update_target_encoder becomes
    # mul_(1.0).add_(q, alpha=0.0) — a no-op. The target encoder never moves, and
    # since the target IS the regression target, the run regresses onto a fixed
    # random init. Guarded by tests/test_config_system.py::
    # TestNoDeadKeys::test_trainer_ema_fallback_is_not_the_frozen_value.
    if model_name == "text_span_jepa":
        ema_scheduler = EMATauSchedule(
            tau_start=model_cfg.get("ema_tau_start", 0.996),
            tau_end=model_cfg.get("ema_tau_end", 0.9999),
            total_steps=total_steps,
        )
    else:
        # data2vec handles its own EMA internally
        ema_scheduler = None
    return (
        num_epochs,
        grad_accum_steps,
        optimizer,
        use_bfloat16,
        scaler,
        total_steps,
        scheduler,
        wd_scheduler,
        ema_scheduler,
    )


def _restore_training_state(
    args,
    log_dir,
    latest_path,
    model,
    optimizer,
    scaler,
    model_name,
    scheduler,
    wd_scheduler,
    ema_scheduler,
    mask_collator,
):
    """Resume-from-checkpoint: restore model, optimizer, schedulers and streams.

    Returns: (start_epoch, global_step, ema_step, mask_step, best_val_loss).

    `meta.load_checkpoint: true` with a missing file is a `FileNotFoundError`.
    The old code had no `else` for that case, so it fell through to
    `global_step = 0`, trained a fresh model and overwrote the good
    `checkpoint-latest.pth.tar` with a step-0 checkpoint.
    """
    start_epoch = 0
    global_step = 0
    ema_step = 0
    mask_step = 0
    best_val_loss = float("inf")

    r_file = args.get("meta", {}).get("read_checkpoint", None)
    load_model = args.get("meta", {}).get("load_checkpoint", False)

    if not load_model:
        return start_epoch, global_step, ema_step, mask_step, best_val_loss

    load_path = os.path.join(log_dir, r_file) if r_file else latest_path
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"meta.load_checkpoint is true but {load_path!r} does not exist. "
            f"Refusing to start from step 0: this run would overwrite "
            f"{latest_path!r} and destroy the previous one. Set "
            f"meta.load_checkpoint: false to start a fresh run in a different "
            f"logging.folder, or point meta.read_checkpoint at the file you mean.",
        )

    report = {}
    start_epoch, global_step, ema_step, mask_step, extra = load_checkpoint(
        load_path,
        model,
        optimizer,
        scaler,
        model_name=model_name,
        schedulers={
            "scheduler": scheduler,
            "wd_scheduler": wd_scheduler,
            "ema_scheduler": ema_scheduler,
        },
        report=report,
    )
    if extra and "best_val_loss" in extra:
        best_val_loss = extra["best_val_loss"]

    if not report.get("schedulers_restored", False):
        # Legacy checkpoint: replay. Exact only when `epochs` is unchanged on
        # resume, because `T_max` is re-derived from `epochs * ipe`.
        logger.warning(
            f"{load_path} carries no scheduler state; replaying {global_step} "
            f"steps. The schedule you get back matches the original run only if "
            f"optimization.epochs is unchanged.",
        )
        for _ in range(global_step):
            scheduler.step()
            wd_scheduler.step()
            if ema_scheduler is not None:
                ema_scheduler.step()

    # Mask curriculum: `SpanMaskCollator.step()` only increments `_step`, so set
    # the position directly instead of replaying O(mask_step) calls.
    if mask_step and hasattr(mask_collator, "_step"):
        mask_collator._step = int(mask_step)

    logger.info(f"Resumed: epoch={start_epoch}, step={global_step}")
    return start_epoch, global_step, ema_step, mask_step, best_val_loss


def _warn_unknown_config_keys(args):
    """Warn about config leaf keys absent from defaults.yaml (likely typos).

    Compares the full dotted PATH, not the bare leaf name. Leaf-name matching
    was blind to two whole classes of misconfiguration that then trained
    something other than what the config said:

    * a real key in the WRONG subtree — `model.batch_size` is a key name that
      exists, but it lives at `data.batch_size`, so it is inert in `model:`;
    * a wholly misspelled SECTION — `modle:` / `optimisation:` are new
      top-level namespaces, so every leaf under them is unknown and none of
      them reached a reader.

    Both are the same failure as `lamda_swip`: the value is never read. The
    nested typo was caught only by accident, because the leaf name happened to
    be absent from defaults.yaml entirely.

    Sections are matched as paths too, so `description` (prose at any level)
    and `_meta.*` (provenance subtrees) stay exempt, and the
    `data.allow_missing_validation` code-level opt-in keeps working.

    Pinned by tests/test_config_system.py::TestTrainerTypoDetectorGap.
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        defaults_path = os.path.join(base, "defaults.yaml")
        if not os.path.exists(defaults_path):
            defaults_path = os.path.join(base, "..", "defaults.yaml")
        with open(defaults_path) as f:
            known = yaml.safe_load(f)
    except Exception:
        return

    def _paths(d, prefix=""):
        """Every dotted path in a nested mapping: interior sections and leaves."""
        out = set()
        if isinstance(d, dict):
            for k, v in d.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                out.add(p)
                out |= _paths(v, p)
        return out

    known_paths = _paths(known)
    # Full paths, not names, for exactly the reason `known_paths` is paths --
    # a bare name here would re-open the wrong-subtree hole this function
    # exists to close. Keys invisible to a textual defaults.yaml diff:
    #   - consumed dynamically by baselines via model_cfg.get(...)
    #   - CLI-only overrides / descriptive provenance
    #   - read by src/train.py itself rather than by a model builder
    #     (`data.allow_missing_validation` gates a deliberate
    #      train-without-validate decision; it is a code-level opt-in, not a
    #      model hyperparameter, and defaults.yaml is owned elsewhere)
    extra_known = {
        "model.average_top_k_layers",
        "model.loss_beta",
        "model.loss_scale",
        "model.ema_decay",
        "model.ema_end_decay",
        "model.ema_anneal_end_step",
        "model.head_layers",
        "meta.dataset",
        "data.allow_missing_validation",
    }
    metadata_keys = {"_meta", "description"}  # declarative namespaces
    metadata_prefixes = ("_meta.",)  # _meta.* provenance subtrees

    def _walk(d, prefix=""):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                _walk(v, p)
            elif (
                p not in known_paths
                and p not in extra_known
                and k not in metadata_keys
                and not p.startswith(metadata_prefixes)
            ):
                logger.warning(
                    f"Unknown config key '{p}' is not a path in defaults.yaml — "
                    "possible typo, or a real key in the wrong section"
                )

    _walk(args)


# ═══════════════════════════════════════════════════════════════════
#  Device resolution — TPU / DDP-GPU / CUDA / CPU
# ═══════════════════════════════════════════════════════════════════

XLA_INSTALL_COMMAND = "pip install pytorch_xla[tpu]"


def xla_missing_error(pjrt_device=None):
    """Actionable error for `PJRT_DEVICE` set without an importable `pytorch_xla`.

    A function rather than a module constant because the message embeds the
    offending value, and a constant would freeze `os.environ` at import time.
    """
    if pjrt_device is None:
        pjrt_device = os.environ.get("PJRT_DEVICE", "")
    return RuntimeError(
        f"PJRT_DEVICE is set (={pjrt_device!r}) but pytorch_xla is not importable. "
        f"Install it on the TPU host with `{XLA_INSTALL_COMMAND}`, or unset "
        f"PJRT_DEVICE to fall back to CUDA/CPU. Refusing to continue: without this "
        f"check the run dies later with an AttributeError from inside XLA, long "
        f"after the dataset has been loaded."
    )


def xla_is_available():
    """True when `pytorch_xla` can be imported AND a PJRT device is configured.

    Importability alone is not enough: a CPU-side box with `pytorch_xla`
    installed for testing must not be hijacked. Requiring `PJRT_DEVICE` as well
    keeps the check pure and hardware-free.
    """
    if not os.environ.get("PJRT_DEVICE"):
        return False
    return importlib.util.find_spec("pytorch_xla") is not None


def resolve_device(env=None, xla_available=None, cuda_available=None):
    """Pick the training device from the environment. Pure: no hardware touched.

    Priority, highest first:

    1. **TPU** — `PJRT_DEVICE` set (or `pytorch_xla` importable *and* a PJRT
       device configured). Returns `torch.device("xla")`.
    2. **DDP GPU** — `LOCAL_RANK` present, which `torchrun` sets per process.
       Returns `cuda:{LOCAL_RANK}`. Without this every rank resolved to
       `cuda:0` and the ranks fought over one device.
    3. **CUDA** — `cuda.is_available()`.
    4. **CPU**.

    Raises `RuntimeError` (never `AttributeError` from deep inside a training
    step) when `PJRT_DEVICE` is set but `pytorch_xla` is missing.

    Args:
        env: environment mapping; defaults to `os.environ`. Injectable so the
            whole branch structure is unit-testable with no GPU.
        xla_available: callable returning a bool; defaults to `xla_is_available`.
        cuda_available: callable returning a bool; defaults to
            `torch.cuda.is_available`.

    Note on step 2: `LOCAL_RANK` is trusted rather than gated on
    `cuda_available`. Under `torchrun --nproc_per_node=N` on a CPU-only node
    (gloo backend) the honest device is CPU, so pass
    `cuda_available=lambda: True` semantics aside — the caller in `src/train.py`
    is the only production entry point and it logs a warning when CUDA is
    absent. Fail-fast at tensor allocation, where the error names the real
    problem, beats guessing here.
    """
    env = os.environ if env is None else env
    if xla_available is None:
        xla_available = xla_is_available
    if cuda_available is None:
        cuda_available = torch.cuda.is_available

    if env.get("PJRT_DEVICE"):
        if not xla_available():
            raise xla_missing_error(env.get("PJRT_DEVICE"))
        return torch.device("xla")
    if xla_available():
        return torch.device("xla")

    if env.get("LOCAL_RANK") is not None:
        try:
            local_rank = int(env["LOCAL_RANK"])
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"LOCAL_RANK={env['LOCAL_RANK']!r} is not an integer. It is set by "
                f"torchrun; unset it to run single-process."
            ) from e
        if not cuda_available():
            logger.warning(
                f"LOCAL_RANK={local_rank} is set but CUDA is not available. Every "
                f"rank will be placed on cuda:{local_rank} and fail at the first "
                f"tensor allocation. For CPU DDP, unset LOCAL_RANK or set it only "
                f"together with a working CPU backend.",
            )
        return torch.device("cuda", local_rank)

    if cuda_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def main(args):
    # ---- Config ----
    meta_seed = args.get("meta", {}).get("seed")
    top_seed = args.get("seed")
    # Explicit None chain (no falsy-or): seed=0 must be honored; defaults.yaml
    # documents top-level `seed`, code historically read only meta.seed.
    seed = meta_seed if meta_seed is not None else (top_seed if top_seed is not None else 42)
    seed_everything(seed)
    _warn_unknown_config_keys(args)

    device = resolve_device()
    logger.info(f"Using device: {device}")

    # Normalize model_name once at the start — all downstream functions use it
    raw_model_name = args.get("meta", {}).get("model_name", "text_span_jepa")
    model_name = _normalize_model_name(raw_model_name)
    logger.info(f"Model type: {raw_model_name} -> {model_name}")

    # ---- Data ----
    data_cfg = args.get("data", {})
    (
        dataloader,
        val_dataloader,
        tokenizer,
        mask_token_id,
        seq_len,
    ) = _build_data_pipeline(args, seed)

    # ---- Model ----
    model_cfg = args.get("model", {})
    model = create_model(model_name, model_cfg, tokenizer.vocab_size, seq_len, device)

    if hasattr(model, "get_num_params"):
        num_params = model.get_num_params()
        logger.info(f"Model parameters (get_num_params()): {num_params:,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # Mechanism wiring status (round-2 audit): CMC and GAC losses are
    # wired into the optimization below; WSR mode='gradient' consumes
    # the one-step-lagged workspace gradient captured post-backward.

    # ---- Mask Collator ----
    # Pass mask curriculum params so mask ratio ramps up during training
    from src.masks.span import SpanMaskCollator

    mask_ratio_start = model_cfg.get("mask_ratio_start", None)
    mask_ratio_end = model_cfg.get("mask_ratio_end", None)
    curriculum_steps = None
    if mask_ratio_start is not None and mask_ratio_end is not None:
        curriculum_steps = 10000  # Default: 10K steps for curriculum

    mask_collator = SpanMaskCollator(
        mask_ratio=data_cfg.get("mask_ratio", 0.35),
        span_length_range=tuple(data_cfg.get("span_length_range", [3, 10])),
        mask_token_id=mask_token_id,
        mask_ratio_start=mask_ratio_start,
        mask_ratio_end=mask_ratio_end,
        curriculum_steps=curriculum_steps or 0,
        # GPT-2 id 0 is a live "!" token; pad-aware masking needs the real id.
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )

    # ---- Optimizer + Schedulers ----
    (
        num_epochs,
        grad_accum_steps,
        optimizer,
        use_bfloat16,
        scaler,
        total_steps,
        scheduler,
        wd_scheduler,
        ema_scheduler,
    ) = _build_optimization(args, model, model_name, model_cfg, device, len(dataloader))

    # ---- Logging ----
    log_cfg = args.get("logging", {})
    log_dir = log_cfg.get("folder", "output/")
    os.makedirs(log_dir, exist_ok=True)
    log_freq = log_cfg.get("log_freq", 10)

    # Dump config
    dump_path = os.path.join(log_dir, "params-text-span-jepa.yaml")
    with open(dump_path, "w") as f:
        yaml.dump(args, f)

    # CSV loss logger — I-JEPA pattern
    csv_path = os.path.join(log_dir, "train_log.csv")
    csv_logger = CSVLogger(
        csv_path,
        ("%f", "loss"),
        ("%f", "lr"),
        ("%f", "wd"),
        ("%f", "loss_span"),
        ("%f", "loss_future"),
        ("%f", "loss_decoder"),
        ("%f", "loss_variance"),
        ("%f", "loss_covariance"),
        ("%f", "effective_rank"),
        ("%f", "collapsed_dim_ratio"),
        ("%f", "mask_fraction"),
        ("%f", "decoder_accuracy"),
    )

    # Single name for the schedules so `save_checkpoint` and `_restore_training_state`
    # cannot disagree about which objects carry schedule state.
    _schedulers = {
        "scheduler": scheduler,
        "wd_scheduler": wd_scheduler,
        "ema_scheduler": ema_scheduler,
    }

    # ---- Resume from checkpoint ----
    latest_path = os.path.join(log_dir, "checkpoint-latest.pth.tar")
    (
        start_epoch,
        global_step,
        ema_step,
        mask_step,
        best_val_loss,
    ) = _restore_training_state(
        args,
        log_dir,
        latest_path,
        model,
        optimizer,
        scaler,
        model_name,
        scheduler,
        wd_scheduler,
        ema_scheduler,
        mask_collator,
    )

    # ---- Training Loop ----
    logger.info(
        f"Starting training: {num_epochs} epochs, {total_steps} total steps, "
        f"model={model_name}, grad_accum={grad_accum_steps}",
    )
    logger.info(
        f"Mask curriculum: start={mask_ratio_start}, end={mask_ratio_end}, "
        f"curriculum_steps={curriculum_steps}",
    )

    for epoch in range(start_epoch, num_epochs):
        loss_meter = AverageMeter()
        model.train()
        epoch_start = time.time()

        for itr, batch in enumerate(dataloader):
            # Collate with masking
            collated = mask_collator(
                [{"input_ids": batch["input_ids"][i]} for i in range(batch["input_ids"].size(0))],
            )
            masked_input_ids = collated["masked_input_ids"].to(device)
            original_input_ids = collated["original_input_ids"].to(device)
            mask_positions = collated["mask_positions"].to(device)

            # LR + WD step
            new_lr = scheduler.step()
            new_wd = wd_scheduler.step()

            # Forward + backward
            autocast_device = device.type if device.type == "cuda" else "cpu"
            with torch.amp.autocast(
                autocast_device,
                enabled=use_bfloat16,
                dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            ):
                total_loss, loss_dict, diag_dict = compute_loss(
                    model,
                    masked_input_ids,
                    original_input_ids,
                    mask_positions,
                    current_step=global_step,
                    total_steps=total_steps,
                )
                # Snapshot the PRIMARY pass slot tensor for GAC before a CMC
                # second forward re-stashes it (restored below the branch).
                _gac_primary = getattr(model, "_gac_z", None)

                # CMC: Cross-Mask Consistency — optional second forward pass
                # When enabled, compute consistency loss between predictions
                # from the current mask and a second different mask.
                _cmc_primary = getattr(model, "_cmc_pass", None)
                if (
                    model_name == "text_span_jepa"
                    and hasattr(model, "cmc")
                    and model.cmc is not None
                    and model.cmc.should_compute(global_step)
                    # Skip the wasted second forward unless a CMC loss weight is
                    # actually configured (consistency term itself remains unwired).
                    and getattr(model.config, "lambda_cmc", 0.0) > 0
                ):
                    with torch.no_grad():
                        # Generate second mask for same input
                        second_mask = model.cmc.generate_second_mask(
                            seq_len=mask_positions.size(1),
                            batch_size=mask_positions.size(0),
                            mask_ratio=mask_positions.float().mean().item(),
                            device=mask_positions.device,
                            step=global_step,
                        )
                        overlap = model.cmc.compute_overlap_mask(mask_positions, second_mask)
                    # Second forward pass (detached — only provides gradient
                    # to z_pred_secondary, not to encoder weights)
                    with torch.amp.autocast(
                        autocast_device,
                        enabled=use_bfloat16,
                        dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                    ):
                        _, _loss_dict_2, _ = compute_loss(
                            model,
                            masked_input_ids,
                            original_input_ids,
                            second_mask,
                            current_step=global_step,
                            total_steps=total_steps,
                        )
                    # Wire the consistency term: bridge compact slot predictions
                    # from both passes into full-sequence space and add
                    # lambda_cmc * L_CMC to the total loss.
                    _cmc_secondary = getattr(model, "_cmc_pass", None)
                    if _cmc_primary is not None and _cmc_secondary is not None:
                        loss_cmc_extra, cmc_info = model.compute_cmc_between_passes(
                            _cmc_primary,
                            _cmc_secondary,
                        )
                        total_loss = total_loss + model.config.lambda_cmc * loss_cmc_extra
                        loss_dict["loss_cmc"] = float(loss_cmc_extra.item())
                        loss_dict["cmc_skipped"] = cmc_info.get("cmc_skipped", False)
                    with torch.no_grad():
                        loss_dict["cmc_overlap_count"] = overlap.sum().item()
                        loss_dict["cmc_overlap_ratio"] = overlap.float().mean().item()
                # Restore GAC's target to the primary pass: the bridge consumes
                # DETACHED primary slots, so their gradients carry main-loss
                # signal only — tau_grad semantics stay clean.
                model._gac_z = _gac_primary

                # Scale loss for gradient accumulation
                scaled_loss = total_loss / grad_accum_steps

            gac_wiring = (
                model_name == "text_span_jepa"
                and getattr(model, "gac", None) is not None
                and getattr(model.config, "lambda_gac", 0.0) > 0
                and getattr(model, "_gac_z", None) is not None
            )
            # retain_graph: the GAC exploration backward traverses the same graph
            # as the main loss (live slot predictions); without GAC it frees.
            if use_bfloat16:
                scaler.scale(scaled_loss).backward(retain_graph=gac_wiring)
            else:
                scaled_loss.backward(retain_graph=gac_wiring)

            if gac_wiring:
                z_ref = getattr(model, "_gac_z", None)
                if z_ref is not None and z_ref.grad is not None:
                    # Rescale accumulated micro-batch grads back to single-batch
                    # calibration so tau_grad keeps its documented meaning.
                    k_micro = (itr % grad_accum_steps) + 1
                    scale_now = scaler.get_scale() if use_bfloat16 else 1.0
                    g_norms = (
                        (z_ref.grad.detach() / (scale_now * k_micro))
                        .reshape(-1, z_ref.size(-1))
                        .norm(dim=0)
                    )
                    if torch.isfinite(g_norms).all():
                        loss_gac, gac_info = model.gac(z_ref, g_norms, step=global_step)
                        if loss_gac.requires_grad:
                            scaler.scale(loss_gac / grad_accum_steps).backward()
                        loss_dict["loss_gac"] = float(loss_gac.item())
                        for _k2, _v2 in gac_info.items():
                            loss_dict[f"gac_{_k2}"] = _v2
                    else:
                        # Overflow micro-batch leaves inf/NaN scaled grads that
                        # would poison the window via found_inf — skip.
                        pass
                    z_ref.grad = None  # avoid feedback into next micro-batch read

            # Only update weights every grad_accum_steps
            if (itr + 1) % grad_accum_steps == 0:
                # Riemannian tangent projection BEFORE the step consumes the
                # gradient. Previously the correction ran AFTER optimizer.step()
                # and was destroyed by zero_grad — a documented-but-dead write
                # (audit R18, JAWPProofAuditor finding #10).
                if (
                    model_name == "text_span_jepa"
                    and getattr(model, "jawp", None) is not None
                    and model.jawp.workspace_Q.grad is not None
                ):
                    model.jawp.project_tangent_gradient()

                # Global gradient clipping (I-JEPA pattern: single clip_grad_norm)
                all_trainable = _get_all_trainable_params(model)
                if all_trainable:
                    torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)

                if use_bfloat16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # Capture one-step-lagged workspace gradient BEFORE zero_grad
                # wipes it (set_to_none=True default) — feeds WSR mode='gradient'.
                if (
                    model_name == "text_span_jepa"
                    and getattr(model, "wsr", None) is not None
                    and getattr(model.wsr, "mode", "") == "gradient"
                    and model.jawp is not None
                ):
                    _qgrad = model.jawp.workspace_Q.grad
                    if _qgrad is not None:
                        k_active_cap = int(model.jawp.active_k.item())
                        # Gradients are never loss-scaled (scaler is disabled: bf16
                        # needs no loss scaling) — only the accumulation factor
                        # remains. Dividing by get_scale() too would silence
                        # mode='gradient' if an fp16 path is ever added.
                        model.wsr.set_lagged_gradient(_qgrad[:, :k_active_cap] / grad_accum_steps)

                # JAWP Stiefel manifold retraction — MUST run after optimizer.step()
                # and BEFORE zero_grad: its Riemannian correction reads Q.grad
                # (previously ordered after zero_grad, which silenced it — audit).
                if (
                    model_name == "text_span_jepa"
                    and hasattr(model, "jawp")
                    and model.jawp is not None
                ):
                    model.jawp.stiefel_retract()
                # PCR Stiefel retraction — keeps cascade projection Q orthonormal
                if (
                    model_name == "text_span_jepa"
                    and hasattr(model, "pcr")
                    and model.pcr is not None
                ):
                    model.pcr.stiefel_retract()
                # SPC Stiefel retraction — keeps frequency basis orthonormal
                if (
                    model_name == "text_span_jepa"
                    and hasattr(model, "spc")
                    and model.spc is not None
                ):
                    model.spc.stiefel_retract()
                    # Information-proportional weight adaptation (proofs/spc.md):
                    # nudge band weights toward variance x predictability every
                    # 100 steps. Without this call the adaptation method was
                    # dead code and weights moved only via backprop (audit R11).
                    if global_step > 0 and global_step % 100 == 0:
                        model.spc.adapt_weights_to_predictability()

                optimizer.zero_grad()

            # EMA update
            if ema_scheduler is not None:
                tau = ema_scheduler.step()
                do_ema_update(model, model_name, tau)
                ema_step += 1
            elif model_name in SELF_EMA_ARMS:
                # Arms whose teacher is annealed internally. BYOL is here for a
                # reason worth writing down: without this branch it trains with a
                # teacher FROZEN AT RANDOM INITIALISATION. `do_ema_update` would
                # handle it, but nothing would call it -- the arm still descends,
                # still logs a falling loss, and is not BYOL. The failure is a
                # model that only shows up as a bad number much later.
                do_ema_update(model, model_name)

            mask_collator.step()
            mask_step += 1
            global_step += 1

            loss_val = total_loss.item()  # Unscaled for logging
            loss_meter.update(loss_val)

            # Logging
            if itr % log_freq == 0 or np.isnan(loss_val) or np.isinf(loss_val):
                mem = torch.cuda.max_memory_allocated() / 1024.0**2 if device.type == "cuda" else 0
                logger.info(
                    f"[{epoch+1}, {itr:5d}] loss={loss_meter.avg:.3f} "
                    f"lr={new_lr:.2e} wd={new_wd:.2e} mem={mem:.0f}MB",
                )
                # Log individual loss components
                logger.info(
                    f"[{epoch+1}, {itr:5d}] losses: "
                    f'span={loss_dict.get("loss_span", 0):.4f} '
                    f'future={loss_dict.get("loss_future", 0):.4f} '
                    f'decoder={loss_dict.get("loss_decoder", 0):.4f} '
                    f'var={loss_dict.get("loss_variance", 0):.4f} '
                    f'cov={loss_dict.get("loss_covariance", 0):.4f} '
                    f'dec_acc={loss_dict.get("decoder_accuracy", 0):.3f}',
                )
                if diag_dict:
                    logger.info(
                        f"[{epoch+1}, {itr:5d}] diag: "
                        f'eff_rank={diag_dict.get("effective_rank_online",0):.1f} '
                        f'collapsed={diag_dict.get("collapsed_dim_ratio_online",0):.3f} '
                        f'mask_frac={diag_dict.get("mask_fraction",0):.2f} '
                        f'target_center_norm={diag_dict.get("target_center_norm",0):.2f} '
                        f'ws_quality={diag_dict.get("workspace_quality",0):.3f}',
                    )
                    # JAWP-specific diagnostics
                    if "jawk_k" in loss_dict:
                        logger.info(
                            f"[{epoch+1}, {itr:5d}] jawp: "
                            f'k={loss_dict.get("jawk_k",0)} '
                            f'ws_util={loss_dict.get("jawk_workspace_utilization",0):.3f} '
                            f'ws_cos={loss_dict.get("jawk_workspace_cosine",0):.3f} '
                            f'ortho={loss_dict.get("jawk_ortho_score",0):.3f} '
                            f'pca_align={loss_dict.get("jawk_pca_alignment",0):.3f}',
                        )
                    # CGN-specific diagnostics
                    if "cgn_tau" in loss_dict:
                        logger.info(
                            f"[{epoch+1}, {itr:5d}] cgn: "
                            f'tau={loss_dict.get("cgn_tau",0):.3f} '
                            f'gate_diff={loss_dict.get("cgn_gate_diff",0):.3f} '
                            f'routing_gap={loss_dict.get("cgn_routing_gap",0):.3f} '
                            f'sparsity={loss_dict.get("cgn_sparsity",0):.3f}',
                        )

                # CSV logging
                csv_logger.log(
                    loss_val,
                    new_lr,
                    new_wd,
                    loss_dict.get("loss_span", 0),
                    loss_dict.get("loss_future", 0),
                    loss_dict.get("loss_decoder", 0),
                    loss_dict.get("loss_variance", 0),
                    loss_dict.get("loss_covariance", 0),
                    diag_dict.get("effective_rank_online", 0),
                    diag_dict.get("collapsed_dim_ratio_online", 0),
                    diag_dict.get("mask_fraction", 0),
                    loss_dict.get("decoder_accuracy", 0),
                )

            if np.isnan(loss_val):
                # Save emergency checkpoint before crashing
                logger.error("NaN loss detected! Saving emergency checkpoint...")
                save_checkpoint(
                    os.path.join(log_dir, "checkpoint-nan.pth.tar"),
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    ema_step,
                    mask_step,
                    extra_state={"best_val_loss": best_val_loss},
                    model_name=model_name,
                    schedulers=_schedulers,
                )
                raise RuntimeError(f"Loss is NaN at epoch {epoch+1}, step {global_step}")

        # ---- End of epoch ----
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1} avg loss: {loss_meter.avg:.4f} " f"time: {epoch_time:.0f}s")

        # ---- Validation ----
        val_loss = None
        if val_dataloader is not None:
            val_loss = _validate(
                model,
                val_dataloader,
                mask_collator,
                device,
                model_name,
                max_batches=50,
                current_step=global_step,
                total_steps=total_steps,
            )
            logger.info(f"  Validation loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(log_dir, "best.pt")
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    global_step,
                    ema_step,
                    mask_step,
                    extra_state={"best_val_loss": best_val_loss},
                    model_name=model_name,
                    schedulers=_schedulers,
                )
                logger.info(f"  New best model! val_loss={best_val_loss:.4f}")

        # ---- Checkpoint ----
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scaler,
            epoch + 1,
            global_step,
            ema_step,
            mask_step,
            extra_state={"best_val_loss": best_val_loss},
            model_name=model_name,
            schedulers=_schedulers,
        )
        epoch_path = os.path.join(log_dir, f"checkpoint-ep{epoch+1}.pth.tar")
        save_checkpoint(
            epoch_path,
            model,
            optimizer,
            scaler,
            epoch + 1,
            global_step,
            ema_step,
            mask_step,
            extra_state={"best_val_loss": best_val_loss},
            model_name=model_name,
            schedulers=_schedulers,
        )
        logger.info(f"Saved checkpoint: {epoch_path}")

        # Optional retention (audit R4 backlog): keep only the newest K epoch
        # checkpoints. Default null keeps every file (previous behavior).
        keep_k = log_cfg.get("keep_last_epoch_ckpts", None)
        if keep_k is not None:
            import re

            epoch_ckpts = []
            for fname in os.listdir(log_dir):
                m = re.fullmatch(r"checkpoint-ep(\d+)\.pth\.tar", fname)
                if m:
                    epoch_ckpts.append((int(m.group(1)), fname))
            epoch_ckpts.sort()
            k_int = int(keep_k)
            stale = epoch_ckpts[: len(epoch_ckpts) - k_int] if k_int > 0 else epoch_ckpts
            for _, stale_name in stale:
                try:
                    os.remove(os.path.join(log_dir, stale_name))
                except OSError:
                    logger.warning(f"Could not prune old checkpoint: {stale_name}")

    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")


def _snapshot_training_buffers(model):
    """Clone every registered buffer. Returns None for duck-typed models.

    Buffer *values* are what a mechanism's running statistics and reference
    subspaces live in, and `no_grad()` does not protect them.

    Only in-place mutation is covered. A module that *replaces* a buffer
    attribute during validation would leave a new tensor in place; every
    mechanism in `src/models/` mutates in place, and `_restore_training_buffers`
    warns about anything it cannot find, so the failure mode is a log line
    rather than a silent divergence.
    """
    named_buffers = getattr(model, "named_buffers", None)
    if named_buffers is None:
        return None
    return {name: buf.detach().clone() for name, buf in named_buffers()}


def _restore_training_buffers(model, snapshot):
    """Inverse of `_snapshot_training_buffers`; warns about anything it cannot place."""
    if not snapshot:
        return
    named_buffers = dict(model.named_buffers())
    missing = []
    with torch.no_grad():
        for name, saved in snapshot.items():
            buf = named_buffers.get(name)
            if buf is None or buf.shape != saved.shape:
                missing.append(name)
                continue
            buf.copy_(saved)
    if missing:
        logger.warning(
            f"Could not restore {len(missing)} training buffers after validation: "
            f"{missing}. They were created or reshaped during the pass.",
        )


def _validate(
    model,
    val_dataloader,
    mask_collator,
    device,
    model_name,
    max_batches=50,
    current_step=0,
    total_steps=1,
):
    """Run validation and return average loss.

    Validation must not touch training state. `torch.no_grad()` blocks
    *gradient* writes and nothing else: one pass moves reference subspaces and
    running statistics that the next training step reads as loss inputs. The
    audit recorded 24 such buffers (`sta.ref_cov`, `wsd.target_cov`,
    `wsd.target_Q`, the `sta.is_initialized` flag, all of PUC's and RDC's
    running stats); after the mechanism-level `TrainingStateGuard` work landed
    in `src/models/*` the same measurement on this repo's current tree reports
    one remaining offender, `target_centering.center`. Snapshotting at the loop
    level closes the class of bug rather than each site: it covers the
    mechanism-free model, the un-guardable buffers, and any mechanism added
    later.

    The numpy stream that mask sampling draws from is snapshotted too.
    Validation consumed it, so a run that validated and a run that did not
    produced different models from the same seed.
    """
    buffer_snapshot = _snapshot_training_buffers(model)
    rng_snapshot = _capture_rng_state()
    was_training = getattr(model, "training", False)
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_batches:
                break
            collated = mask_collator(
                [{"input_ids": batch["input_ids"][j]} for j in range(batch["input_ids"].size(0))],
            )
            masked = collated["masked_input_ids"].to(device)
            original = collated["original_input_ids"].to(device)
            mask = collated["mask_positions"].to(device)

            total_loss, _, _ = compute_loss(
                model,
                masked,
                original,
                mask,
                current_step=current_step,
                total_steps=total_steps,
            )
            val_losses.append(total_loss.item())
    _restore_training_buffers(model, buffer_snapshot)
    _restore_rng_state(rng_snapshot)
    if hasattr(model, "training"):
        model.train(was_training)
    else:
        model.train()
    # float() cast: np.mean returns np.float64, which is not weights_only-allowlisted
    # and would poison every checkpoint's best_val_loss (forces the legacy-pickle path).
    return float(np.mean(val_losses)) if val_losses else float("inf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname",
        type=str,
        default="config/wikitext/textspanjepa_wikitext_small.yaml",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--no_defaults",
        action="store_true",
        help="Skip merging defaults.yaml (use config as-is)",
    )
    args = parser.parse_args()

    # ── Deep-merge with defaults.yaml ──────────────────────────────
    # Ablation configs only override mechanism flags; all other fields
    # come from defaults.yaml. Without this merge, ablation configs
    # are broken (missing embed_dim, encoder_depth, etc.).
    # I-JEPA / C-JEPA pattern: base config + experiment overrides.
    with open(args.fname) as f:
        config = yaml.safe_load(f)

    if not args.no_defaults:
        # Find defaults.yaml (same directory as train.py, or repo root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        defaults_path = os.path.join(script_dir, "defaults.yaml")
        if not os.path.exists(defaults_path):
            # Try repo root
            defaults_path = os.path.join(script_dir, "..", "defaults.yaml")
        if os.path.exists(defaults_path):
            with open(defaults_path) as f:
                defaults = yaml.safe_load(f)
            config = _deep_merge(defaults, config)

    if args.output_dir is not None:
        config.setdefault("logging", {})["folder"] = args.output_dir
    main(config)
