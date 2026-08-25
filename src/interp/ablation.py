# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Ablation framework: what happens when you remove each JEPA component?
#
# Reviewers WILL ask: "Is the predictor necessary? What about future loss?
# What about VICReg? What about the iterative refinement?"
#
# This module provides a systematic way to:
# 1. Disable individual components
# 2. Train ablated models at ANY model size
# 3. Compare metrics across ablations
# 4. Determine which components are necessary for which properties
#
# Key ablations:
# - No predictor (just encoder + decoder)
# - No future loss (span prediction only)
# - No VICReg (no variance/covariance regularization)
# - No iterative refinement (single-pass predictor)
# - No EMA target (same encoder for both paths)
# - No decoder (pure latent prediction, no reconstruction)

import copy

import torch
from src.utils.cka_metrics import linear_cka
from torch import nn


class AblationConfig:
    """Configuration for an ablation study.

    Each boolean flag controls whether a component is active.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        # Components that can be ablated
        self.use_predictor = kwargs.get("use_predictor", True)
        self.use_future_loss = kwargs.get("use_future_loss", True)
        self.use_variance_reg = kwargs.get("use_variance_reg", True)
        self.use_covariance_reg = kwargs.get("use_covariance_reg", True)
        self.use_iterative_refinement = kwargs.get("use_iterative_refinement", True)
        self.use_ema_target = kwargs.get("use_ema_target", True)
        self.use_decoder = kwargs.get("use_decoder", True)
        self.use_target_centering = kwargs.get("use_target_centering", True)
        self.use_span_loss = kwargs.get("use_span_loss", True)

    def describe(self):
        """Human-readable description of what's ablated."""
        parts = []
        if not self.use_predictor:
            parts.append("no predictor")
        if not self.use_future_loss:
            parts.append("no future loss")
        if not self.use_variance_reg:
            parts.append("no VICReg variance")
        if not self.use_covariance_reg:
            parts.append("no VICReg covariance")
        if not self.use_iterative_refinement:
            parts.append("no iterative refinement")
        if not self.use_ema_target:
            parts.append("no EMA target")
        if not self.use_decoder:
            parts.append("no decoder")
        if not self.use_target_centering:
            parts.append("no target centering")
        if not self.use_span_loss:
            parts.append("no span loss")
        if not parts:
            return "full model"
        return " + ".join(parts)


# Standard ablation configs
ABLATION_CONFIGS = {
    "full": AblationConfig("full"),
    "no_predictor": AblationConfig(
        "no_predictor", use_predictor=False, use_iterative_refinement=False
    ),
    "no_future_loss": AblationConfig("no_future_loss", use_future_loss=False),
    "no_vicreg": AblationConfig("no_vicreg", use_variance_reg=False, use_covariance_reg=False),
    "no_variance_only": AblationConfig("no_variance_only", use_variance_reg=False),
    "no_covariance_only": AblationConfig("no_covariance_only", use_covariance_reg=False),
    "no_refinement": AblationConfig("no_refinement", use_iterative_refinement=False),
    "no_ema": AblationConfig("no_ema", use_ema_target=False),
    "no_decoder": AblationConfig("no_decoder", use_decoder=False),
    "no_centering": AblationConfig("no_centering", use_target_centering=False),
    "no_span_loss": AblationConfig("no_span_loss", use_span_loss=False),
    "predictor_only": AblationConfig(
        "predictor_only",
        use_future_loss=False,
        use_decoder=False,
        use_variance_reg=False,
        use_covariance_reg=False,
        use_target_centering=False,
    ),
}

# ═══════════════════════════════════════════════════════════════
# Model size variants for ablation
# ═══════════════════════════════════════════════════════════════

MODEL_SIZE_CONFIGS = {
    "tiny": {
        "embed_dim": 256,
        "encoder_depth": 4,
        "num_heads": 4,
        "predictor_embed_dim": 128,
        "predictor_depth": 2,
        "mlp_ratio": 4.0,
    },
    "small": {
        "embed_dim": 512,
        "encoder_depth": 8,
        "num_heads": 8,
        "predictor_embed_dim": 256,
        "predictor_depth": 4,
        "mlp_ratio": 4.0,
    },
    "base": {
        "embed_dim": 768,
        "encoder_depth": 12,
        "num_heads": 12,
        "predictor_embed_dim": 384,
        "predictor_depth": 6,
        "mlp_ratio": 4.0,
    },
    "large": {
        "embed_dim": 1024,
        "encoder_depth": 16,
        "num_heads": 16,
        "predictor_embed_dim": 512,
        "predictor_depth": 8,
        "mlp_ratio": 4.0,
    },
}

# Full ablation matrix: each ablation x each model size
ABLATION_MATRIX = {}
for abl_name, abl_config in ABLATION_CONFIGS.items():
    for size_name, size_config in MODEL_SIZE_CONFIGS.items():
        key = f"{abl_name}_{size_name}"
        ABLATION_MATRIX[key] = {
            "ablation": abl_config,
            "model_size": size_name,
            "model_config": size_config,
        }


class AblatedModel(nn.Module):
    """Wrapper that ablates specific components from the JEPA model.

    Instead of modifying the model architecture, we:
    1. Modify the loss computation (zero out ablated components)
    2. Control iterative refinement via config override
    3. Control EMA updates via skip mechanism
    """

    def __init__(self, base_model, ablation_config: AblationConfig):
        super().__init__()
        self.model = base_model
        self.config = ablation_config

    def forward(
        self, masked_input_ids, original_input_ids, mask_positions, current_step=0, total_steps=1
    ):
        """Forward pass with ablation.

        Args aligned with TextSpanJEPA.compute_loss_with_targets convention:
        (masked_input_ids, original_input_ids, mask_positions)

        Returns modified loss where ablated components have zero contribution.
        """
        # Override iterative refinement for this forward pass
        original_refine_steps = self.model.predictor.num_refine_steps
        if not self.config.use_iterative_refinement:
            self.model.predictor.num_refine_steps = 0

        result = self.model.compute_loss_with_targets(
            masked_input_ids,
            original_input_ids,
            mask_positions,
            current_step=current_step,
            total_steps=total_steps,
        )

        # Restore refinement steps
        self.model.predictor.num_refine_steps = original_refine_steps

        # compute_loss_with_targets returns (loss, loss_dict, diag_dict)
        if len(result) == 3:
            loss, info, diag = result
        elif len(result) == 2:
            loss, info = result
        else:
            loss, info, _diag = result[0], {}, {}

        # Zero out ablated loss components.
        # FIX: subtract the weighted contribution (lambda * loss), not just the raw loss.
        # The total_loss already includes lambda * loss_component,
        # so we subtract the same amount to neutralize it.
        cfg = self.model.config
        if not self.config.use_future_loss and "loss_future" in info:
            loss = loss - cfg.lambda_future * info["loss_future"]

        if not self.config.use_decoder and "loss_decoder" in info:
            loss = loss - cfg.lambda_decoder * info["loss_decoder"]

        if not self.config.use_span_loss and "loss_span" in info:
            loss = loss - cfg.lambda_span * info["loss_span"]

        if not self.config.use_variance_reg and "loss_variance" in info:
            loss = loss - cfg.lambda_variance * info["loss_variance"]

        if not self.config.use_covariance_reg and "loss_covariance" in info:
            loss = loss - cfg.lambda_covariance * info["loss_covariance"]

        # Record ablation info
        info["ablation"] = self.config.name
        info["ablation_desc"] = self.config.describe()

        return loss, info

    @torch.no_grad()
    def ablate_ema(self):
        """If EMA is ablated, copy online weights to target (no EMA)."""
        if not self.config.use_ema_target:
            for p_q, p_k in zip(
                self.model.encoder.parameters(), self.model.target_encoder.parameters()
            ):
                p_k.data.copy_(p_q.data)

    @torch.no_grad()
    def skip_ema_update(self):
        """Return True if EMA update should be skipped (ablated)."""
        return not self.config.use_ema_target


class AblationStudy:
    """Run a systematic ablation study.

    For each ablation config:
    1. Train the ablated model for N steps
    2. Extract representations
    3. Compute all metrics
    4. Compare to full model
    """

    def __init__(self, base_model, train_fn, device="cpu"):
        """
        Args:
            base_model: TextSpanJEPA model
            train_fn: callable(model, n_steps) -> loss_history
            device: compute device
        """
        self.base_model = base_model
        self.train_fn = train_fn
        self.device = device

    def run_single(self, ablation_name: str, n_steps=1000):
        """Run a single ablation.

        Args:
            ablation_name: key from ABLATION_CONFIGS
            n_steps: number of training steps

        Returns:
            dict with training results
        """
        config = ABLATION_CONFIGS.get(ablation_name)
        if config is None:
            raise ValueError(f"Unknown ablation: {ablation_name}")

        # Create ablated model
        model = copy.deepcopy(self.base_model)
        ablated = AblatedModel(model, config)

        # If EMA ablated, copy weights initially
        ablated.ablate_ema()

        # Train
        loss_history = self.train_fn(ablated, n_steps)

        return {
            "ablation": ablation_name,
            "description": config.describe(),
            "final_loss": loss_history[-1] if loss_history else float("inf"),
            "loss_history": loss_history,
        }

    def run_all(self, n_steps=1000, ablations=None):
        """Run all standard ablations.

        Args:
            n_steps: training steps per ablation
            ablations: list of ablation names (None = all)

        Returns:
            dict of {ablation_name: results}
        """
        if ablations is None:
            ablations = list(ABLATION_CONFIGS.keys())

        results = {}
        for name in ablations:
            if name == "full":
                # Full model — just extract representations, no training needed
                results[name] = {
                    "ablation": "full",
                    "description": "full model",
                    "final_loss": 0,
                    "loss_history": [],
                }
                continue

            print(f"Running ablation: {name}...")
            try:
                result = self.run_single(name, n_steps)
                results[name] = result
            except Exception as e:
                results[name] = {
                    "ablation": name,
                    "error": str(e),
                }

        return results

    def run_scaling_ablations(self, n_steps=500, model_sizes=None, ablations=None):
        """Run ablations at multiple model sizes.

        THE KEY EXPERIMENT for Oral: does JEPA's advantage scale?
        Run each ablation at tiny/small/base to see if the component's
        importance changes with model size.

        Args:
            n_steps: training steps per ablation
            model_sizes: list of size names from MODEL_SIZE_CONFIGS
            ablations: list of ablation names

        Returns:
            dict of {(ablation, size): results}
        """
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        if model_sizes is None:
            model_sizes = ["tiny", "small", "base"]
        if ablations is None:
            ablations = [
                "full",
                "no_predictor",
                "no_future_loss",
                "no_vicreg",
                "no_refinement",
                "no_decoder",
            ]

        results = {}
        for size_name in model_sizes:
            size_config = MODEL_SIZE_CONFIGS[size_name]
            print(f"\n=== Model size: {size_name} ===")

            # Create model at this size
            cfg = TextSpanJEPAConfig(
                vocab_size=1000,
                max_seq_len=64,
                **size_config,
                future_offsets=(1, 4),
                num_refine_steps=3,
            )
            model = TextSpanJEPA(cfg).to(self.device)
            n_params = model.get_num_params()
            print(f"  Parameters: {n_params:,}")

            for abl_name in ablations:
                key = f"{abl_name}_{size_name}"
                abl_config = ABLATION_CONFIGS.get(abl_name)
                if abl_config is None:
                    continue

                print(f"  Ablation: {abl_name}...", end=" ")
                try:
                    model_copy = copy.deepcopy(model)
                    ablated = AblatedModel(model_copy, abl_config)
                    ablated.ablate_ema()
                    loss_history = self.train_fn(ablated, n_steps)

                    results[key] = {
                        "ablation": abl_name,
                        "model_size": size_name,
                        "n_params": n_params,
                        "description": abl_config.describe(),
                        "final_loss": loss_history[-1] if loss_history else float("inf"),
                        "loss_history": loss_history,
                    }
                    print(f"final loss: {results[key]['final_loss']:.4f}")
                except Exception as e:
                    results[key] = {
                        "ablation": abl_name,
                        "model_size": size_name,
                        "error": str(e),
                    }
                    print(f"ERROR: {e}")

        return results

    @torch.no_grad()
    def compare_representations(self, ablated_reps_dict, full_model_reps):
        """Compare representations of each ablation to the full model.

        Args:
            ablated_reps_dict: {ablation_name: (N, D) representations}
            full_model_reps: (N, D) full model representations

        Returns:
            dict with per-ablation CKA and geometry comparison
        """
        from src.interp.representation_geometry import RepresentationGeometry
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        full_geom = RepresentationGeometry.compute_all(full_model_reps)

        comparisons = {}
        for name, reps in ablated_reps_dict.items():
            cka = linear_cka(reps, full_model_reps)
            geom = RepresentationGeometry.compute_all(reps)

            comparisons[name] = {
                "cka_to_full": cka,
                "geometry": geom,
                "geometry_diff": {k: geom.get(k, 0) - full_geom.get(k, 0) for k in full_geom},
            }

        return comparisons
