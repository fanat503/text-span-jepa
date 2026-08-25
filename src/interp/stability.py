# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Representation stability analysis across training
#
# Key question: how quickly do representations stabilize during training?
# JEPA should converge FASTER and be MORE STABLE than MLM because:
# 1. Latent prediction has a smoother loss landscape than token reconstruction
# 2. EMA target provides a stable learning signal
# 3. No reconstruction noise from token-level predictions
#
# Metrics:
# - CKA between consecutive checkpoints (convergence speed)
# - Representational similarity across training (stability)
# - Loss curve smoothness
# - Early stopping advantage: how early can you stop and still get good representations?


import torch
from src.utils.cka_metrics import linear_cka


class TrainingStability:
    """Analyze how stable representations are during training.

    From training checkpoints, measure:
    1. Convergence speed: how many steps until CKA(current, final) > 0.95?
    2. Stability: CKA between consecutive checkpoints
    3. Early stopping: minimum training for acceptable downstream performance
    """

    @staticmethod
    @torch.no_grad()
    def convergence_curve(checkpoint_representations, final_representations=None):
        """CKA between each checkpoint and the final model.

        Args:
            checkpoint_representations: list of (N, D) tensors from different steps
            final_representations: (N, D) final model (if None, use last checkpoint)

        Returns:
            dict with convergence metrics
        """
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()

        if final_representations is None:
            final_representations = checkpoint_representations[-1]

        N = (
            final_representations.size(0)
            if final_representations.dim() == 2
            else final_representations.size(0)
        )
        cka_values = []

        for ckpt_reps in checkpoint_representations:
            flat_ckpt = ckpt_reps.reshape(-1, ckpt_reps.size(-1))[:N]
            flat_final = final_representations.reshape(-1, final_representations.size(-1))[:N]
            n = min(flat_ckpt.size(0), flat_final.size(0))
            cka = linear_cka(flat_ckpt[:n], flat_final[:n])
            cka_values.append(cka)

        # Convergence step: first checkpoint with CKA > 0.95
        convergence_step = None
        for i, cka in enumerate(cka_values):
            if cka > 0.95:
                convergence_step = i
                break

        # Stability: mean CKA between consecutive checkpoints
        consecutive_cka = []
        for i in range(len(cka_values) - 1):
            consecutive_cka.append(cka_values[i + 1] - cka_values[i])

        return {
            "cka_curve": cka_values,
            "convergence_step": convergence_step,
            "final_cka": cka_values[-1] if cka_values else 0.0,
            "consecutive_cka_changes": consecutive_cka,
            "mean_consecutive_change": (
                sum(consecutive_cka) / len(consecutive_cka) if consecutive_cka else 0.0
            ),
            "n_checkpoints": len(checkpoint_representations),
        }

    @staticmethod
    def compare_convergence(jepa_checkpoints, baseline_checkpoints):
        """Compare convergence between JEPA and baseline.

        Args:
            jepa_checkpoints: list of (N, D) tensors
            baseline_checkpoints: list of (N, D) tensors

        Returns:
            dict with comparison
        """
        jepa_conv = TrainingStability.convergence_curve(jepa_checkpoints)
        baseline_conv = TrainingStability.convergence_curve(baseline_checkpoints)

        return {
            "jepa_convergence_step": jepa_conv["convergence_step"],
            "baseline_convergence_step": baseline_conv["convergence_step"],
            "jepa_converges_faster": (
                jepa_conv["convergence_step"] is not None
                and baseline_conv["convergence_step"] is not None
                and jepa_conv["convergence_step"] < baseline_conv["convergence_step"]
            ),
            "jepa_final_cka": jepa_conv["final_cka"],
            "baseline_final_cka": baseline_conv["final_cka"],
        }


class LossStability:
    """Analyze loss curve stability during training.

    JEPA should have smoother loss curves (less variance)
    because the EMA target provides a stable learning signal.
    """

    @staticmethod
    def compute(loss_values, window_size=10):
        """Compute loss stability metrics.

        Args:
            loss_values: list of float loss values per step
            window_size: window for moving average

        Returns:
            dict with stability metrics
        """
        if len(loss_values) < window_size:
            return {
                "mean": sum(loss_values) / len(loss_values) if loss_values else 0,
                "std": 0.0,
                "cv": 0.0,
                "smoothness": 0.0,
            }

        losses = torch.tensor(loss_values, dtype=torch.float32)

        # Moving average
        kernel = torch.ones(window_size) / window_size
        ma = torch.conv1d(
            losses.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=window_size // 2,
        ).squeeze()[
            : len(losses)
        ]  # Truncate to original length

        # Coefficient of variation of the loss (lower = more stable)
        mean_loss = losses.mean().item()
        std_loss = losses.std().item()
        cv = std_loss / mean_loss if mean_loss > 0 else float("inf")

        # Smoothness: 1 - mean |loss[i] - MA[i]| / mean_loss
        residuals = (losses[: ma.size(0)] - ma).abs()
        smoothness = 1.0 - residuals.mean().item() / mean_loss if mean_loss > 0 else 0.0

        # Convergence: slope of last 20% of loss curve
        n_last = max(len(loss_values) // 5, 2)
        last_losses = losses[-n_last:]
        if n_last >= 2:
            steps = torch.arange(n_last, dtype=torch.float32)
            slope = ((last_losses * steps).mean() - last_losses.mean() * steps.mean()) / max(
                (steps**2).mean() - steps.mean() ** 2, 1e-10
            )
            convergence_slope = slope.item()
        else:
            convergence_slope = 0.0

        return {
            "mean": mean_loss,
            "std": std_loss,
            "cv": cv,
            "smoothness": max(smoothness, 0.0),
            "convergence_slope": convergence_slope,
            "min": losses.min().item(),
            "max": losses.max().item(),
        }

    @staticmethod
    def compare(jepa_losses, baseline_losses):
        """Compare loss stability between JEPA and baseline."""
        jepa_stab = LossStability.compute(jepa_losses)
        baseline_stab = LossStability.compute(baseline_losses)

        return {
            "jepa_smoothness": jepa_stab["smoothness"],
            "baseline_smoothness": baseline_stab["smoothness"],
            "jepa_more_stable": jepa_stab["smoothness"] > baseline_stab["smoothness"],
            "jepa_cv": jepa_stab["cv"],
            "baseline_cv": baseline_stab["cv"],
            "jepa_lower_variance": jepa_stab["cv"] < baseline_stab["cv"],
        }


class EarlyStoppingAdvantage:
    """Measure how early you can stop training and still get good representations.

    JEPA should have an advantage: earlier checkpoints already
    encode useful structure because the predictor forces early
    layers to be informative.

    Metric: at what training fraction does CKA(checkpoint, final) > 0.9?
    Earlier = more efficient = better for low-compute scenarios.
    """

    @staticmethod
    @torch.no_grad()
    def compute(
        checkpoint_representations, checkpoint_fractions, threshold=0.9, probe_fn=None, labels=None
    ):
        """Find the earliest checkpoint with acceptable quality.

        Args:
            checkpoint_representations: list of (N, D) tensors
            checkpoint_fractions: list of float, training fraction [0, 1]
            threshold: CKA threshold for "acceptable"
            probe_fn: optional probe function for task-based early stopping
            labels: optional labels for probe-based evaluation

        Returns:
            dict with early stopping fraction
        """
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()

        final = checkpoint_representations[-1]
        cka_values = []

        for reps, frac in zip(checkpoint_representations, checkpoint_fractions):
            flat_ckpt = reps.reshape(-1, reps.size(-1))
            flat_final = final.reshape(-1, final.size(-1))
            n = min(flat_ckpt.size(0), flat_final.size(0))
            cka = linear_cka(flat_ckpt[:n], flat_final[:n])
            cka_values.append((frac, cka))

        # Find earliest fraction where CKA > threshold
        early_stop_fraction = 1.0
        for frac, cka in cka_values:
            if cka > threshold:
                early_stop_fraction = frac
                break

        return {
            "early_stop_fraction": early_stop_fraction,
            "cka_threshold": threshold,
            "cka_curve": [(f, c) for f, c in cka_values],
            "saving_fraction": 1.0 - early_stop_fraction,  # How much training you can skip
        }

    @staticmethod
    def compare(
        jepa_checkpoints, baseline_checkpoints, jepa_fractions, baseline_fractions, threshold=0.9
    ):
        """Compare early stopping advantage between JEPA and baseline."""
        jepa_early = EarlyStoppingAdvantage.compute(jepa_checkpoints, jepa_fractions, threshold)
        baseline_early = EarlyStoppingAdvantage.compute(
            baseline_checkpoints, baseline_fractions, threshold
        )

        return {
            "jepa_early_stop": jepa_early["early_stop_fraction"],
            "baseline_early_stop": baseline_early["early_stop_fraction"],
            "jepa_converges_earlier": jepa_early["early_stop_fraction"]
            < baseline_early["early_stop_fraction"],
            "compute_saving": baseline_early["early_stop_fraction"]
            - jepa_early["early_stop_fraction"],
        }


class CheckpointConsistency:
    """How consistent are representations across random seeds?

    If JEPA representations are consistent across seeds but MLM
    representations vary → JEPA has a more robust learning signal.
    """

    @staticmethod
    @torch.no_grad()
    def cross_seed_cka(seed_representations):
        """CKA between representations trained with different seeds.

        Args:
            seed_representations: list of (N, D) tensors, one per seed

        Returns:
            dict with pairwise CKA
        """
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()

        n_seeds = len(seed_representations)
        pairwise_cka = []

        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                flat_a = seed_representations[i].reshape(-1, seed_representations[i].size(-1))
                flat_b = seed_representations[j].reshape(-1, seed_representations[j].size(-1))
                n = min(flat_a.size(0), flat_b.size(0))
                cka = linear_cka(flat_a[:n], flat_b[:n])
                pairwise_cka.append(cka)

        return {
            "mean_pairwise_cka": sum(pairwise_cka) / len(pairwise_cka) if pairwise_cka else 0.0,
            "min_pairwise_cka": min(pairwise_cka) if pairwise_cka else 0.0,
            "std_pairwise_cka": (
                (
                    sum((c - sum(pairwise_cka) / len(pairwise_cka)) ** 2 for c in pairwise_cka)
                    / max(len(pairwise_cka), 1)
                )
                ** 0.5
                if pairwise_cka
                else 0.0
            ),
            "n_seeds": n_seeds,
            "n_pairs": len(pairwise_cka),
        }

    @staticmethod
    def compare(jepa_seed_reps, baseline_seed_reps):
        """Compare cross-seed consistency between JEPA and baseline."""
        jepa_cons = CheckpointConsistency.cross_seed_cka(jepa_seed_reps)
        baseline_cons = CheckpointConsistency.cross_seed_cka(baseline_seed_reps)

        return {
            "jepa_consistency": jepa_cons["mean_pairwise_cka"],
            "baseline_consistency": baseline_cons["mean_pairwise_cka"],
            "jepa_more_consistent": jepa_cons["mean_pairwise_cka"]
            > baseline_cons["mean_pairwise_cka"],
        }
