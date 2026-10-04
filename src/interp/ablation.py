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
#
# Honesty contract (added 2026-09-28, TASK-17): a run may only be reported
# under the name of an ablation it actually performed.
#   * a term the model did not report (renamed key), a term with no weight in
#     the model config, or a flag with no mechanism behind it is a HARD
#     FAILURE (AblationTargetMissingError) — not a silent no-op. Pass
#     AblationConfig(..., allow_missing_targets=True) to opt out deliberately.
#   * a cell that crashed is returned with status="failed", the error type and
#     traceback, and NO final_loss key, so a missing number cannot be plotted
#     as a result; the aggregate (AblationResults) says the study is incomplete.
#   * "no EMA target" is enforced for the whole run, not just at init: the
#     wrapper intercepts the per-step update_target_encoder() the training loop
#     calls, and re-syncs the target before every forward.

import copy
import logging
import traceback
import warnings

import torch
from src.utils.cka_metrics import linear_cka
from torch import nn

_LOGGER = logging.getLogger(__name__)


class AblationTargetMissingError(RuntimeError):
    """An ablation was requested for a term the model cannot supply.

    Raised instead of letting a run that ablated nothing be reported under
    the name of the ablation it failed to perform.
    """


class AblationResults(dict):
    """``{cell_key: result}`` that knows whether the study finished.

    Still a plain ``dict`` for every existing caller. What it adds is that a
    crashed cell cannot be mistaken for a completed one: it carries
    ``status == "failed"``, the error type and traceback, and no
    ``final_loss``/``loss_history`` key at all.
    """

    @property
    def failures(self):
        """Cells that crashed.

        A cell counts as failed when it says so, or when it carries an error
        (the older shape). An untagged cell is never quietly counted as a
        result.
        """
        return {k: v for k, v in self.items() if v.get("status") == "failed" or "error" in v}

    @property
    def completed(self):
        """Cells that trained to a result."""
        return {k: v for k, v in self.items() if v.get("status") == "ok"}

    @property
    def is_complete(self):
        """True only when no requested cell crashed."""
        return not self.failures

    def summary(self):
        n_ok = len(self.completed)
        head = f"{n_ok}/{len(self)} ablations completed"
        if not self.failures:
            return head
        listed = ", ".join(
            f"{key} ({cell.get('error_type', 'error')})"
            for key, cell in sorted(self.failures.items())
        )
        return f"{head} — INCOMPLETE, failed: {listed}"

    def __repr__(self):
        return f"AblationResults({self.summary()})"


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
        # Escape hatch: when True, a term the model cannot supply is warned
        # about and skipped instead of raising. Default False — an ablation
        # that did nothing must never be reported as one.
        self.allow_missing_targets = kwargs.get("allow_missing_targets", False)

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
        "no_predictor",
        use_predictor=False,
        use_iterative_refinement=False,
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


# Every loss term an AblationConfig flag can remove, as
# (flag, key the model reports the term under, weight on the model config).
# The forward pass removes a term by subtracting weight * value, so if any of
# the three is missing there is nothing to remove and the run is refused
# rather than reported as an ablation.
LOSS_TERM_ABLATIONS = (
    ("use_future_loss", "loss_future", "lambda_future"),
    ("use_decoder", "loss_decoder", "lambda_decoder"),
    ("use_span_loss", "loss_span", "lambda_span"),
    ("use_variance_reg", "loss_variance", "lambda_variance"),
    ("use_covariance_reg", "loss_covariance", "lambda_covariance"),
)

# Flags enforced by AblatedModel itself rather than by subtracting a term.
STATE_FLAG_ABLATIONS = ("use_iterative_refinement", "use_ema_target")

# Flags AblationConfig accepts and describe() reports, but that nothing acts
# on. Setting one False yields a run identical to the full model wearing an
# ablation's label, so forward() refuses it.
UNIMPLEMENTED_ABLATIONS = ("use_predictor", "use_target_centering")


class AblatedModel(nn.Module):
    """Wrapper that ablates specific components from the JEPA model.

    Instead of modifying the model architecture, we:
    1. Modify the loss computation (zero out ablated components)
    2. Control iterative refinement via config override
    3. Control EMA updates via skip mechanism

    Nothing here silently does nothing: an ablation that cannot be performed
    raises AblationTargetMissingError rather than returning an unablated
    loss under an ablation's name.
    """

    def __init__(self, base_model, ablation_config: AblationConfig):
        super().__init__()
        self.model = base_model
        self.config = ablation_config

    def _resolve_loss_terms(self, info, model_config):
        """Work out which loss terms this ablation can actually remove.

        Returns ``(terms, problems)`` where terms is ``[(info_key, weight)]``.
        problems is empty when every requested ablation was honoured; when it
        is not, the caller refuses (or, with allow_missing_targets, warns and
        names them in the returned info dict).
        """
        terms = []
        problems = []
        for flag, key, weight_attr in LOSS_TERM_ABLATIONS:
            if getattr(self.config, flag):
                continue
            if key not in info:
                problems.append(f"{flag}: the model reported no {key!r} term")
                continue
            weight = getattr(model_config, weight_attr, None)
            if weight is None:
                problems.append(f"{flag}: the model config has no {weight_attr!r} weight")
            elif weight == 0:
                problems.append(
                    f"{flag}: {weight_attr} is 0, so {key!r} contributes nothing to remove"
                )
            else:
                terms.append((key, weight))
        for flag in UNIMPLEMENTED_ABLATIONS:
            if not getattr(self.config, flag):
                problems.append(f"{flag}: no ablation mechanism is implemented for it")
        if problems:
            message = self._no_op_message(problems)
            if not self.config.allow_missing_targets:
                raise AblationTargetMissingError(message)
            warnings.warn(
                message + "\nRunning anyway: allow_missing_targets=True.",
                RuntimeWarning,
                stacklevel=3,
            )
        return terms, problems

    def _no_op_message(self, problems):
        listed = "\n".join(f"  - {p}" for p in problems)
        return (
            f"ablation {self.config.name!r} ({self.config.describe()}) cannot be performed:\n"
            f"{listed}\n"
            "Refusing to report this run as an ablation that did nothing. "
            "Pass AblationConfig(..., allow_missing_targets=True) if a no-op is what you want."
        )

    def forward(
        self,
        masked_input_ids,
        original_input_ids,
        mask_positions,
        current_step=0,
        total_steps=1,
    ):
        """Forward pass with ablation.

        Args aligned with TextSpanJEPA.compute_loss_with_targets convention:
        (masked_input_ids, original_input_ids, mask_positions)

        Returns modified loss where ablated components have zero contribution.
        """
        # Keep target == online for the whole run when EMA is ablated. Doing it
        # once at init (ablate_ema) is not enough: the first per-step
        # update_target_encoder() would start the target lagging again.
        self.ablate_ema()

        # Override iterative refinement for this forward pass
        original_refine_steps = self.model.predictor.num_refine_steps
        if not self.config.use_iterative_refinement:
            self.model.predictor.num_refine_steps = 0

        try:
            result = self.model.compute_loss_with_targets(
                masked_input_ids,
                original_input_ids,
                mask_positions,
                current_step=current_step,
                total_steps=total_steps,
            )
        finally:
            # Restore even when the forward raised. Without this, one exception
            # leaves the model pinned at num_refine_steps=0 for the rest of the
            # training run — a silent corruption of every later step.
            self.model.predictor.num_refine_steps = original_refine_steps

        # compute_loss_with_targets returns (loss, loss_dict, diag_dict)
        if len(result) == 3:
            loss, info, diag = result
        elif len(result) == 2:
            loss, info = result
        else:
            loss, info, _diag = result[0], {}, {}

        # Zero out ablated loss components.
        # The total_loss already includes lambda * loss_component, so we
        # subtract the same amount to neutralize it. A term that cannot be
        # removed raises instead of being skipped.
        terms, problems = self._resolve_loss_terms(info, self.model.config)
        for key, weight in terms:
            loss = loss - weight * info[key]

        # Record ablation info
        info["ablation"] = self.config.name
        info["ablation_desc"] = self.config.describe()
        info["ablation_applied"] = [key for key, _weight in terms]
        info["ema_target_skipped"] = self.skip_ema_update()
        if problems:
            info["ablation_ineffective"] = problems

        return loss, info

    @torch.no_grad()
    def ablate_ema(self):
        """If EMA is ablated, copy online weights to target (no EMA)."""
        if not self.config.use_ema_target:
            for p_q, p_k in zip(
                self.model.encoder.parameters(),
                self.model.target_encoder.parameters(),
            ):
                p_k.data.copy_(p_q.data)

    @torch.no_grad()
    def skip_ema_update(self):
        """Return True if EMA update should be skipped (ablated)."""
        return not self.config.use_ema_target

    def update_target_encoder(self, tau):
        """Per-step EMA entry point, intercepted by the ablation.

        The training loop calls ``model.update_target_encoder(tau)`` once per
        step (src.train.do_ema_update). Routing it through the wrapper is what
        makes the "no EMA target" arm real for the whole run rather than a
        one-off re-initialisation: when the arm is active the target is copied
        from the online encoder instead of being blended.
        """
        if self.skip_ema_update():
            self.ablate_ema()
            return
        self.model.update_target_encoder(tau)


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
            "status": "ok",
            "trained": True,
            "final_loss": loss_history[-1] if loss_history else float("inf"),
            "loss_history": loss_history,
        }

    @staticmethod
    def _failed_cell(ablation_name, model_size, n_params, exc):
        """A crashed cell, shaped so it cannot be read as a completed one.

        Deliberately carries no final_loss/loss_history key: a comparison table
        must raise rather than quietly plot a number that was never measured.
        """
        _LOGGER.error(
            "ablation %s%s FAILED: %s: %s",
            ablation_name,
            f" at size {model_size}" if model_size else "",
            type(exc).__name__,
            exc,
        )
        cell = {
            "ablation": ablation_name,
            "status": "failed",
            "trained": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        if model_size is not None:
            cell["model_size"] = model_size
        if n_params is not None:
            cell["n_params"] = n_params
        return cell

    @staticmethod
    def _report_completeness(results):
        """A crashed cell must be visible in the log, not only in the dict."""
        if results.is_complete:
            print(results.summary())
            return
        _LOGGER.error("Ablation study %s", results.summary())
        print(f"!! {results.summary()}")

    def run_all(self, n_steps=1000, ablations=None):
        """Run all standard ablations.

        Args:
            n_steps: training steps per ablation
            ablations: list of ablation names (None = all)

        Returns:
            AblationResults ({ablation_name: results}). Check ``is_complete``
            / ``failures``: a crashed ablation is present with
            ``status == "failed"``, the error type, and no ``final_loss``.

        """
        if ablations is None:
            ablations = list(ABLATION_CONFIGS.keys())

        results = AblationResults()
        for name in ablations:
            if name == "full":
                # Full model — just extract representations, no training needed.
                # REPORTED, NOT FIXED (TASK-17): final_loss is 0 and
                # loss_history is empty, so "full" plots as a lossless model.
                results[name] = {
                    "ablation": "full",
                    "description": "full model",
                    "status": "not_trained",
                    "trained": False,
                    "final_loss": 0,
                    "loss_history": [],
                }
                continue

            print(f"Running ablation: {name}...")
            try:
                result = self.run_single(name, n_steps)
                results[name] = result
            except Exception as e:
                results[name] = self._failed_cell(name, None, None, e)
                print(f"  !! FAILED ({type(e).__name__}): {e}")

        self._report_completeness(results)
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
            AblationResults ({(ablation, size): results}). Check
            ``is_complete`` / ``failures``: a crashed cell is present with
            ``status == "failed"``, the error type, and no ``final_loss``.

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

        results = AblationResults()
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
                        "status": "ok",
                        "trained": True,
                        "final_loss": loss_history[-1] if loss_history else float("inf"),
                        "loss_history": loss_history,
                    }
                    print(f"final loss: {results[key]['final_loss']:.4f}\n")
                except Exception as e:
                    results[key] = self._failed_cell(abl_name, size_name, n_params, e)
                    print(f"  !! FAILED ({type(e).__name__}): {e}\n")

        self._report_completeness(results)
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
