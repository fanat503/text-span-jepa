# Copyright 2026 Text-Span JEPA Authors
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

import logging
import os
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
#  Model name normalization — handles config suffixes
# ═══════════════════════════════════════════════════════════════════


def _normalize_model_name(raw_name):
    """Normalize model_name from config to canonical form.

    Configs may use suffixed names like 'text_span_jepa_small',
    'mlm_small', 'data2vec_base'. We strip the suffix to get
    the canonical name that create_model() understands.

    Canonical names: text_span_jepa, mlm, data2vec
    """
    name = raw_name.strip().lower()
    if name.startswith(("text_span_jepa", "jepa")):
        return "text_span_jepa"
    if name.startswith("mlm"):
        return "mlm"
    if name.startswith("data2vec"):
        return "data2vec"
    return name  # Return as-is, will fail in create_model with clear error


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint save/load — I-JEPA pattern, all model types
# ═══════════════════════════════════════════════════════════════════


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
):
    """Save complete training state for resumption — all model types.

    Handles JEPA (encoder + predictor + target_encoder + decoder),
    MLM (encoder + mlm_head), and data2vec (encoder + target_encoder + regression_head).
    """
    model_name = _normalize_model_name(model_name)
    state = {
        "model_name": model_name,
        "opt": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "ema_step": ema_step,
        "mask_step": mask_step,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    if model_name == "text_span_jepa":
        state["encoder"] = model.encoder.state_dict()
        state["predictor"] = model.predictor.state_dict()
        state["target_encoder"] = model.target_encoder.state_dict()
        state["decoder"] = model.decoder.state_dict()
        if hasattr(model, "target_centering"):
            state["target_centering_center"] = model.target_centering.center.clone()
        # JAWP workspace Q — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "jawp") and model.jawp is not None:
            state["jawp_workspace_Q"] = model.jawp.workspace_Q.data.clone()
            state["jawp_active_k"] = model.jawp.active_k.clone()
        # CGN gate logits — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "cgn") and model.cgn is not None:
            state["cgn_gate_logits_visible"] = model.cgn.gate_logits_visible.data.clone()
            state["cgn_gate_logits_masked"] = model.cgn.gate_logits_masked.data.clone()
            state["cgn_total_steps"] = model.cgn.total_steps.clone()
        # PCR projection Q — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "pcr") and model.pcr is not None:
            state["pcr_workspace_Q"] = model.pcr.workspace_Q.data.clone()
            state["pcr_level_gates"] = [g.data.clone() for g in model.pcr.level_gates]
        # SPC frequency basis and band weights — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "spc") and model.spc is not None:
            state["spc_freq_basis"] = model.spc.freq_basis.data.clone()
            state["spc_log_band_weights"] = model.spc.log_band_weights.data.clone()
            state["spc_running_residual_vars"] = model.spc.running_residual_vars.clone()
            state["spc_running_predictability"] = model.spc.running_predictability.clone()
        # WSD running statistics — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "wsd") and model.wsd is not None:
            state["wsd_running_drift"] = model.wsd.running_drift.clone()
            state["wsd_target_ema"] = model.wsd.target_ema.clone()
            state["wsd_total_syncs"] = model.wsd.total_syncs.clone()
        # CMC running statistics — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "cmc") and model.cmc is not None:
            state["cmc_running_consistency"] = model.cmc.running_consistency.clone()
            state["cmc_running_overlap_ratio"] = model.cmc.running_overlap_ratio.clone()
            state["cmc_total_cmc_steps"] = model.cmc.total_cmc_steps.clone()
        # GAC running statistics — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "gac") and model.gac is not None:
            state["gac_running_grad_norms"] = model.gac.running_grad_norms.clone()
            state["gac_running_starved_fraction"] = model.gac.running_starved_fraction.clone()
            state["gac_total_gac_steps"] = model.gac.total_gac_steps.clone()
        # STA running statistics — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "sta") and model.sta is not None:
            state["sta_running_spectrum"] = model.sta.running_spectrum.clone()
            state["sta_running_transport_cost"] = model.sta.running_transport_cost.clone()
            state["sta_total_sta_steps"] = model.sta.total_steps.clone()
        # PUC running statistics — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "puc") and model.puc is not None:
            state["puc_running_mean"] = model.puc.running_mean.clone()
            state["puc_running_eigenvalues"] = model.puc.running_eigenvalues.clone()
            state["puc_proj_vectors"] = model.puc.proj_vectors.clone()
            state["puc_total_steps"] = model.puc.total_steps.clone()
        # RDC running statistics — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "rdc") and model.rdc is not None:
            state["rdc_z_previous"] = model.rdc.z_previous.clone()
            state["rdc_workspace_Q"] = model.rdc.workspace_Q.clone()
            state["rdc_running_drift_norm"] = model.rdc.running_drift_norm.clone()
            state["rdc_running_drift_ratio"] = model.rdc.running_drift_ratio.clone()
            state["rdc_total_steps"] = model.rdc.total_steps.clone()
        # WSR running statistics — must be saved for resumption
        if model_name == "text_span_jepa" and hasattr(model, "wsr") and model.wsr is not None:
            state["wsr_running_sharpness"] = model.wsr.running_sharpness.clone()
            state["wsr_running_spectral_sharpness"] = model.wsr.running_spectral_sharpness.clone()
            state["wsr_running_directional_sharpness"] = (
                model.wsr.running_directional_sharpness.clone()
            )
            state["wsr_running_grad_norm"] = model.wsr.running_grad_norm.clone()
            state["wsr_total_steps"] = model.wsr.total_steps.clone()
    elif model_name == "mlm":
        state["encoder"] = model.encoder.state_dict()
        state["mlm_head"] = model.mlm_head.state_dict()
    elif model_name == "data2vec":
        state["encoder"] = model.encoder.state_dict()
        state["target_encoder"] = model.target_encoder.state_dict()
        state["regression_head"] = model.regression_head.state_dict()
        if hasattr(model, "num_updates"):
            state["num_updates"] = model.num_updates

    if extra_state is not None:
        state["extra"] = extra_state
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scaler, model_name="text_span_jepa"):
    """Load checkpoint — I-JEPA helper.py load_checkpoint pattern.

    Handles all model types. Returns (epoch, global_step, ema_step, mask_step, extra_state)
    """
    model_name = _normalize_model_name(model_name)
    try:
        checkpoint = safe_torch_load(path, map_location=torch.device("cpu"))

        epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        ema_step = checkpoint.get("ema_step", 0)
        mask_step = checkpoint.get("mask_step", 0)

        # Determine model type from checkpoint if available
        ckpt_model_name = checkpoint.get("model_name", model_name)
        ckpt_model_name = _normalize_model_name(ckpt_model_name)

        if ckpt_model_name == "text_span_jepa":
            model.encoder.load_state_dict(checkpoint["encoder"])
            model.predictor.load_state_dict(checkpoint["predictor"])
            model.target_encoder.load_state_dict(checkpoint["target_encoder"])
            model.decoder.load_state_dict(checkpoint["decoder"])
            if "target_centering_center" in checkpoint and hasattr(model, "target_centering"):
                model.target_centering.center.copy_(checkpoint["target_centering_center"])
            # JAWP workspace Q restoration
            if (
                "jawp_workspace_Q" in checkpoint
                and hasattr(model, "jawp")
                and model.jawp is not None
            ):
                model.jawp.workspace_Q.data.copy_(checkpoint["jawp_workspace_Q"])
            if "jawp_active_k" in checkpoint and hasattr(model, "jawp") and model.jawp is not None:
                model.jawp.active_k.copy_(checkpoint["jawp_active_k"])
            # CGN gate logits restoration
            if (
                "cgn_gate_logits_visible" in checkpoint
                and hasattr(model, "cgn")
                and model.cgn is not None
            ):
                model.cgn.gate_logits_visible.data.copy_(checkpoint["cgn_gate_logits_visible"])
            if "cgn_total_steps" in checkpoint and hasattr(model, "cgn") and model.cgn is not None:
                model.cgn.total_steps.copy_(checkpoint["cgn_total_steps"])
            # PCR projection Q restoration
            if "pcr_workspace_Q" in checkpoint and hasattr(model, "pcr") and model.pcr is not None:
                model.pcr.workspace_Q.data.copy_(checkpoint["pcr_workspace_Q"])
            if "pcr_level_gates" in checkpoint and hasattr(model, "pcr") and model.pcr is not None:
                for i, g in enumerate(checkpoint["pcr_level_gates"]):
                    if i < len(model.pcr.level_gates):
                        model.pcr.level_gates[i].data.copy_(g)
            # SPC frequency basis and band weights restoration
            if "spc_freq_basis" in checkpoint and hasattr(model, "spc") and model.spc is not None:
                model.spc.freq_basis.data.copy_(checkpoint["spc_freq_basis"])
            if (
                "spc_log_band_weights" in checkpoint
                and hasattr(model, "spc")
                and model.spc is not None
            ):
                model.spc.log_band_weights.data.copy_(checkpoint["spc_log_band_weights"])
            if (
                "spc_running_residual_vars" in checkpoint
                and hasattr(model, "spc")
                and model.spc is not None
            ):
                model.spc.running_residual_vars.copy_(checkpoint["spc_running_residual_vars"])
            if (
                "spc_running_predictability" in checkpoint
                and hasattr(model, "spc")
                and model.spc is not None
            ):
                model.spc.running_predictability.copy_(checkpoint["spc_running_predictability"])
            # WSD running statistics restoration
            if (
                "wsd_running_drift" in checkpoint
                and hasattr(model, "wsd")
                and model.wsd is not None
            ):
                model.wsd.running_drift.copy_(checkpoint["wsd_running_drift"])
            if "wsd_target_ema" in checkpoint and hasattr(model, "wsd") and model.wsd is not None:
                model.wsd.target_ema.copy_(checkpoint["wsd_target_ema"])
            if "wsd_total_syncs" in checkpoint and hasattr(model, "wsd") and model.wsd is not None:
                model.wsd.total_syncs.copy_(checkpoint["wsd_total_syncs"])
            # CMC running statistics restoration
            if (
                "cmc_running_consistency" in checkpoint
                and hasattr(model, "cmc")
                and model.cmc is not None
            ):
                model.cmc.running_consistency.copy_(checkpoint["cmc_running_consistency"])
            if (
                "cmc_running_overlap_ratio" in checkpoint
                and hasattr(model, "cmc")
                and model.cmc is not None
            ):
                model.cmc.running_overlap_ratio.copy_(checkpoint["cmc_running_overlap_ratio"])
            if (
                "cmc_total_cmc_steps" in checkpoint
                and hasattr(model, "cmc")
                and model.cmc is not None
            ):
                model.cmc.total_cmc_steps.copy_(checkpoint["cmc_total_cmc_steps"])
            # GAC running statistics restoration
            if (
                "gac_running_grad_norms" in checkpoint
                and hasattr(model, "gac")
                and model.gac is not None
            ):
                model.gac.running_grad_norms.copy_(checkpoint["gac_running_grad_norms"])
            if (
                "gac_running_starved_fraction" in checkpoint
                and hasattr(model, "gac")
                and model.gac is not None
            ):
                model.gac.running_starved_fraction.copy_(checkpoint["gac_running_starved_fraction"])
            if (
                "gac_total_gac_steps" in checkpoint
                and hasattr(model, "gac")
                and model.gac is not None
            ):
                model.gac.total_gac_steps.copy_(checkpoint["gac_total_gac_steps"])
            # STA running statistics restoration
            if (
                "sta_running_spectrum" in checkpoint
                and hasattr(model, "sta")
                and model.sta is not None
            ):
                model.sta.running_spectrum.copy_(checkpoint["sta_running_spectrum"])
            if (
                "sta_running_transport_cost" in checkpoint
                and hasattr(model, "sta")
                and model.sta is not None
            ):
                model.sta.running_transport_cost.copy_(checkpoint["sta_running_transport_cost"])
            if (
                "sta_total_sta_steps" in checkpoint
                and hasattr(model, "sta")
                and model.sta is not None
            ):
                model.sta.total_steps.copy_(checkpoint["sta_total_sta_steps"])
            # PUC running statistics restoration
            if "puc_running_mean" in checkpoint and hasattr(model, "puc") and model.puc is not None:
                model.puc.running_mean.copy_(checkpoint["puc_running_mean"])
            if (
                "puc_running_eigenvalues" in checkpoint
                and hasattr(model, "puc")
                and model.puc is not None
            ):
                model.puc.running_eigenvalues.copy_(checkpoint["puc_running_eigenvalues"])
            if "puc_proj_vectors" in checkpoint and hasattr(model, "puc") and model.puc is not None:
                model.puc.proj_vectors.copy_(checkpoint["puc_proj_vectors"])
            if "puc_total_steps" in checkpoint and hasattr(model, "puc") and model.puc is not None:
                model.puc.total_steps.copy_(checkpoint["puc_total_steps"])
            # RDC running statistics restoration
            if "rdc_z_previous" in checkpoint and hasattr(model, "rdc") and model.rdc is not None:
                model.rdc.z_previous.copy_(checkpoint["rdc_z_previous"])
            if "rdc_workspace_Q" in checkpoint and hasattr(model, "rdc") and model.rdc is not None:
                model.rdc.workspace_Q.copy_(checkpoint["rdc_workspace_Q"])
            if (
                "rdc_running_drift_norm" in checkpoint
                and hasattr(model, "rdc")
                and model.rdc is not None
            ):
                model.rdc.running_drift_norm.copy_(checkpoint["rdc_running_drift_norm"])
            if (
                "rdc_running_drift_ratio" in checkpoint
                and hasattr(model, "rdc")
                and model.rdc is not None
            ):
                model.rdc.running_drift_ratio.copy_(checkpoint["rdc_running_drift_ratio"])
            if "rdc_total_steps" in checkpoint and hasattr(model, "rdc") and model.rdc is not None:
                model.rdc.total_steps.copy_(checkpoint["rdc_total_steps"])
            # WSR restoration
            if (
                "wsr_running_sharpness" in checkpoint
                and hasattr(model, "wsr")
                and model.wsr is not None
            ):
                model.wsr.running_sharpness.copy_(checkpoint["wsr_running_sharpness"])
            if (
                "wsr_running_spectral_sharpness" in checkpoint
                and hasattr(model, "wsr")
                and model.wsr is not None
            ):
                model.wsr.running_spectral_sharpness.copy_(
                    checkpoint["wsr_running_spectral_sharpness"]
                )
            if (
                "wsr_running_directional_sharpness" in checkpoint
                and hasattr(model, "wsr")
                and model.wsr is not None
            ):
                model.wsr.running_directional_sharpness.copy_(
                    checkpoint["wsr_running_directional_sharpness"]
                )
            if (
                "wsr_running_grad_norm" in checkpoint
                and hasattr(model, "wsr")
                and model.wsr is not None
            ):
                model.wsr.running_grad_norm.copy_(checkpoint["wsr_running_grad_norm"])
            if "wsr_total_steps" in checkpoint and hasattr(model, "wsr") and model.wsr is not None:
                model.wsr.total_steps.copy_(checkpoint["wsr_total_steps"])
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

        extra_state = checkpoint.get("extra", None)
        logger.info(
            f"Loaded checkpoint: epoch={epoch}, step={global_step}, "
            f"ema_step={ema_step}, mask_step={mask_step}"
        )
        return epoch, global_step, ema_step, mask_step, extra_state

    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return 0, 0, 0, 0, None


# ═══════════════════════════════════════════════════════════════════
#  Model creation factory
# ═══════════════════════════════════════════════════════════════════


def create_model(model_name, model_cfg, vocab_size, max_seq_len, device):
    """Create model by type — supports jepa, mlm, data2vec.

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
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. " f"Supported: text_span_jepa, mlm, data2vec"
        )
    return model


def compute_loss(
    model, masked_input_ids, original_input_ids, mask_positions, current_step=0, total_steps=1
):
    """Compute loss for any model type — unified interface.

    Always returns (total_loss, loss_dict, diag_dict) for consistency.
    """
    if hasattr(model, "compute_loss_with_targets"):
        # JEPA model — returns (loss, loss_dict, diag_dict)
        return model.compute_loss_with_targets(
            masked_input_ids,
            original_input_ids,
            mask_positions,
            current_step=current_step,
            total_steps=total_steps,
        )
    elif hasattr(model, "forward") and hasattr(model, "regression_head"):
        # data2vec — returns (loss, info_dict)
        loss, info = model(masked_input_ids, original_input_ids, mask_positions)
        return loss, info, {}
    elif hasattr(model, "compute_loss"):
        # MLM — returns (loss, info_dict)
        loss, info = model.compute_loss(masked_input_ids, original_input_ids, mask_positions)
        return loss, info, {}
    else:
        raise ValueError(f"Model {type(model).__name__} has no supported loss method")


def get_param_groups(model, model_name, wd=0.04):
    """Build optimizer param groups with WD_exclude for bias/norm.

    model_name is automatically normalized.
    """
    model_name = _normalize_model_name(model_name)

    if model_name == "text_span_jepa":
        return [
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ]
            },
            {
                "params": [
                    p
                    for n, p in model.predictor.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ]
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
    elif model_name == "mlm":
        # MLM: all encoder params + mlm_head
        return [
            {
                "params": [
                    p
                    for n, p in model.encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ]
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
                ]
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
    else:
        return [{"params": list(model.parameters())}]


def do_ema_update(model, model_name, tau=None):
    """Perform EMA update of target encoder — works for all model types.

    model_name is automatically normalized.
    For JEPA: uses scheduled tau from EMATauSchedule.
    For data2vec: uses model's internal get_annealed_decay().
    """
    model_name = _normalize_model_name(model_name)

    if model_name == "text_span_jepa":
        if tau is not None:
            model.update_target_encoder(tau)
    elif model_name == "data2vec":
        model.update_target_encoder()
    # MLM has no EMA target — no-op


def _get_all_trainable_params(model):
    """Get all trainable parameters as a single list for global grad clipping."""
    return [p for p in model.parameters() if p.requires_grad]


# ═══════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════


def _build_data_pipeline(args, seed):
    """Load dataset(s) and construct train/validation loaders.

    Returns: (dataloader, val_dataloader, tokenizer, mask_token_id, seq_len).
    """
    data_cfg = args.get("data", {})
    seq_len = data_cfg.get("max_seq_len", 512)

    logger.info("Loading dataset...")
    from src.datasets.kaggle import get_mask_token_id, load_wikitext103, make_dataloader

    dataset, tokenizer = load_wikitext103(
        tokenizer_name=data_cfg.get("tokenizer", "gpt2"),
        seq_len=seq_len,
        split="train",
        data_dir=data_cfg.get("root_path", "/kaggle/input/wikitext-103"),
    )
    mask_token_id = get_mask_token_id(tokenizer)

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
            worker_init_fn=lambda wid: worker_init_fn(wid, seed),
        )
    except Exception as e:
        logger.warning(
            f"Validation unavailable ({type(e).__name__}: {e}) — training without validation"
        )
        val_dataloader = None

    dataloader = make_dataloader(
        dataset,
        batch_size=data_cfg.get("batch_size", 64),
        num_workers=data_cfg.get("num_workers", 2),
        worker_init_fn=lambda wid: worker_init_fn(wid, seed),
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_bfloat16)

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
    # EMA schedule — only for JEPA models
    if model_name == "text_span_jepa":
        ema_scheduler = EMATauSchedule(
            tau_start=model_cfg.get("ema_tau_start", 0.996),
            tau_end=model_cfg.get("ema_tau_end", 1.0),
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
    """Resume-from-checkpoint: replay schedulers/curriculum to the saved step.

    Returns: (start_epoch, global_step, ema_step, mask_step, best_val_loss).
    """
    start_epoch = 0
    global_step = 0
    ema_step = 0
    mask_step = 0
    best_val_loss = float("inf")

    r_file = args.get("meta", {}).get("read_checkpoint", None)
    load_model = args.get("meta", {}).get("load_checkpoint", False)

    if load_model:
        load_path = os.path.join(log_dir, r_file) if r_file else latest_path
        if os.path.exists(load_path):
            start_epoch, global_step, ema_step, mask_step, extra = load_checkpoint(
                load_path, model, optimizer, scaler, model_name=model_name
            )
            if extra and "best_val_loss" in extra:
                best_val_loss = extra["best_val_loss"]
            # Advance schedulers to correct step
            for _ in range(global_step):
                scheduler.step()
                wd_scheduler.step()
                if ema_scheduler is not None:
                    ema_scheduler.step()
            # Advance mask curriculum
            for _ in range(mask_step):
                mask_collator.step()
            logger.info(f"Resumed: epoch={start_epoch}, step={global_step}")
    return start_epoch, global_step, ema_step, mask_step, best_val_loss


def _warn_unknown_config_keys(args):
    """Warn about config leaf keys absent from defaults.yaml (likely typos)."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        defaults_path = os.path.join(base, "defaults.yaml")
        if not os.path.exists(defaults_path):
            defaults_path = os.path.join(base, "..", "defaults.yaml")
        with open(defaults_path, "r") as f:
            known = yaml.safe_load(f)
    except Exception:
        return

    def _leaves(d, prefix=""):
        out = set()
        if isinstance(d, dict):
            for k, v in d.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                out.add(p)
                out |= _leaves(v, p)
        return out

    known_leaf_names = {p.split(".")[-1] for p in _leaves(known)}
    # Keys invisible to a textual defaults.yaml diff:
    #   - consumed dynamically by baselines via model_cfg.get(...)
    #   - CLI-only overrides / descriptive provenance
    extra_known = {
        "average_top_k_layers",
        "loss_beta",
        "loss_scale",
        "ema_decay",
        "ema_end_decay",
        "ema_anneal_end_step",
        "head_layers",
        "dataset",
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
                k not in known_leaf_names
                and k not in metadata_keys
                and k not in extra_known
                and not p.startswith(metadata_prefixes)
            ):
                logger.warning(f"Unknown config key '{p}' is not in defaults.yaml — possible typo")

    _walk(args)


def main(args):
    # ---- Config ----
    meta_seed = args.get("meta", {}).get("seed")
    top_seed = args.get("seed")
    # Explicit None chain (no falsy-or): seed=0 must be honored; defaults.yaml
    # documents top-level `seed`, code historically read only meta.seed.
    seed = meta_seed if meta_seed is not None else (top_seed if top_seed is not None else 42)
    seed_everything(seed)
    _warn_unknown_config_keys(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        logger.info(f"Model parameters (non-embedding): {num_params:,}")
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
        f"model={model_name}, grad_accum={grad_accum_steps}"
    )
    logger.info(
        f"Mask curriculum: start={mask_ratio_start}, end={mask_ratio_end}, "
        f"curriculum_steps={curriculum_steps}"
    )

    for epoch in range(start_epoch, num_epochs):
        loss_meter = AverageMeter()
        model.train()
        epoch_start = time.time()

        for itr, batch in enumerate(dataloader):
            # Collate with masking
            collated = mask_collator(
                [{"input_ids": batch["input_ids"][i]} for i in range(batch["input_ids"].size(0))]
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
                            _cmc_primary, _cmc_secondary
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
                        # scaler.step() has ALREADY unscaled .grad in-place here;
                        # only the accumulation factor remains. Dividing by
                        # get_scale() too would silence mode='gradient' on AMP.
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
            elif model_name == "data2vec":
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
                    f"lr={new_lr:.2e} wd={new_wd:.2e} mem={mem:.0f}MB"
                )
                # Log individual loss components
                logger.info(
                    f"[{epoch+1}, {itr:5d}] losses: "
                    f'span={loss_dict.get("loss_span", 0):.4f} '
                    f'future={loss_dict.get("loss_future", 0):.4f} '
                    f'decoder={loss_dict.get("loss_decoder", 0):.4f} '
                    f'var={loss_dict.get("loss_variance", 0):.4f} '
                    f'cov={loss_dict.get("loss_covariance", 0):.4f} '
                    f'dec_acc={loss_dict.get("decoder_accuracy", 0):.3f}'
                )
                if diag_dict:
                    logger.info(
                        f"[{epoch+1}, {itr:5d}] diag: "
                        f'eff_rank={diag_dict.get("effective_rank_online",0):.1f} '
                        f'collapsed={diag_dict.get("collapsed_dim_ratio_online",0):.3f} '
                        f'mask_frac={diag_dict.get("mask_fraction",0):.2f} '
                        f'target_center_norm={diag_dict.get("target_center_norm",0):.2f} '
                        f'ws_quality={diag_dict.get("workspace_quality",0):.3f}'
                    )
                    # JAWP-specific diagnostics
                    if "jawk_k" in loss_dict:
                        logger.info(
                            f"[{epoch+1}, {itr:5d}] jawp: "
                            f'k={loss_dict.get("jawk_k",0)} '
                            f'ws_util={loss_dict.get("jawk_workspace_utilization",0):.3f} '
                            f'ws_cos={loss_dict.get("jawk_workspace_cosine",0):.3f} '
                            f'ortho={loss_dict.get("jawk_ortho_score",0):.3f} '
                            f'pca_align={loss_dict.get("jawk_pca_alignment",0):.3f}'
                        )
                    # CGN-specific diagnostics
                    if "cgn_tau" in loss_dict:
                        logger.info(
                            f"[{epoch+1}, {itr:5d}] cgn: "
                            f'tau={loss_dict.get("cgn_tau",0):.3f} '
                            f'gate_diff={loss_dict.get("cgn_gate_diff",0):.3f} '
                            f'routing_gap={loss_dict.get("cgn_routing_gap",0):.3f} '
                            f'sparsity={loss_dict.get("cgn_sparsity",0):.3f}'
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
                )
                raise RuntimeError(f"Loss is NaN at epoch {epoch+1}, step {global_step}")

        # ---- End of epoch ----
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1} avg loss: {loss_meter.avg:.4f} " f"time: {epoch_time:.0f}s")

        # ---- Validation ----
        val_loss = None
        if val_dataloader is not None:
            val_loss = _validate(
                model, val_dataloader, mask_collator, device, model_name, max_batches=50
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


def _validate(model, val_dataloader, mask_collator, device, model_name, max_batches=50):
    """Run validation and return average loss."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_batches:
                break
            collated = mask_collator(
                [{"input_ids": batch["input_ids"][j]} for j in range(batch["input_ids"].size(0))]
            )
            masked = collated["masked_input_ids"].to(device)
            original = collated["original_input_ids"].to(device)
            mask = collated["mask_positions"].to(device)

            total_loss, _, _ = compute_loss(model, masked, original, mask)
            val_losses.append(total_loss.item())
    model.train()
    return np.mean(val_losses) if val_losses else float("inf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname", type=str, default="config/wikitext/textspanjepa_wikitext_small.yaml"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--no_defaults", action="store_true", help="Skip merging defaults.yaml (use config as-is)"
    )
    args = parser.parse_args()

    # ── Deep-merge with defaults.yaml ──────────────────────────────
    # Ablation configs only override mechanism flags; all other fields
    # come from defaults.yaml. Without this merge, ablation configs
    # are broken (missing embed_dim, encoder_depth, etc.).
    # I-JEPA / C-JEPA pattern: base config + experiment overrides.
    with open(args.fname, "r") as f:
        config = yaml.safe_load(f)

    if not args.no_defaults:
        # Find defaults.yaml (same directory as train.py, or repo root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        defaults_path = os.path.join(script_dir, "defaults.yaml")
        if not os.path.exists(defaults_path):
            # Try repo root
            defaults_path = os.path.join(script_dir, "..", "defaults.yaml")
        if os.path.exists(defaults_path):
            with open(defaults_path, "r") as f:
                defaults = yaml.safe_load(f)
            config = _deep_merge(defaults, config)

    if args.output_dir is not None:
        config.setdefault("logging", {})["folder"] = args.output_dir
    main(config)
