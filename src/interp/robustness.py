# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Robustness analysis: how do representations respond to input perturbations?
#
# KEY PREDICTION: JEPA should be MORE ROBUST than MLM because:
# 1. JEPA predicts in latent space, not token space
# 2. Token-level noise doesn't affect latent predictions as much
# 3. MLM must reconstruct exact tokens → sensitive to any input change
#
# This is a clean, testable prediction that directly supports the
# central hypothesis. If confirmed, it's strong evidence for Oral.
#
# Types of perturbations:
# - Token dropout: randomly zero out tokens
# - Token substitution: replace tokens with random
# - Token permutation: shuffle token order
# - Noise injection: add Gaussian noise to embeddings
# - Span corruption: corrupt spans (JEPA-specific: trained for this)


import inspect

import torch

from src.interp import rng
from src.utils.cka_metrics import linear_cka


def perturbation_fn_name(perturbation_fn):
    """A stable stream-label suffix for a perturbation callable.

    ``__qualname__`` rather than ``repr``: two different lambdas can share a
    ``repr``-ish form, while two *different* perturbations cannot share a
    qualname, and the same perturbation must map to the same stream across runs
    (a bound method's qualname does not include the instance address).
    """
    return getattr(perturbation_fn, "__qualname__", None) or type(perturbation_fn).__qualname__


class RepresentationRobustness:
    """Measure how robust representations are to input perturbations.

    For each perturbation type and intensity:
    1. Perturb the input
    2. Get representations from perturbed input
    3. Compute CKA(repr_clean, repr_perturbed)
    4. Compute metric degradation

    JEPA should maintain HIGHER CKA under perturbation (more robust).
    """

    def __init__(self, model, device="cpu"):
        """
        Args:
            model: model with .encoder attribute
            device: compute device

        """
        self.model = model
        self.device = device

    @torch.no_grad()
    def get_representations(self, input_ids):
        """Get pooled representations from model."""
        if hasattr(self.model, "encoder"):
            h, _ = self.model.encoder(input_ids.to(self.device))
        else:
            h = self.model(input_ids.to(self.device))
        return h.mean(dim=1).cpu()

    @staticmethod
    def _accepts_generator(perturbation_fn):
        """Whether `perturbation_fn` takes a ``generator`` keyword.

        The four perturbation functions in this module do; a caller-supplied
        one need not, and passing the keyword unconditionally would raise
        ``TypeError`` on a perfectly good 2-argument callable. Probed with
        ``inspect`` rather than by trial call so a ``TypeError`` raised *inside*
        a perturbation is not mistaken for a signature mismatch.
        """
        try:
            params = inspect.signature(perturbation_fn).parameters
        except (TypeError, ValueError):
            return False
        if "generator" in params:
            return True
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    @torch.no_grad()
    def perturbation_curve(
        self,
        input_ids,
        perturbation_fn,
        intensities=(0.1, 0.2, 0.3, 0.5, 0.7),
        n_trials=3,
        seed=None,
    ):
        """Compute robustness curve for one perturbation type.

        Args:
            input_ids: (B, T) clean input IDs
            perturbation_fn: callable(input_ids, intensity) -> perturbed_ids,
                optionally accepting a ``generator`` keyword
            intensities: list of perturbation intensities
            n_trials: number of random trials per intensity
            seed: base seed for the perturbation draws. Each trial gets its own
                generator derived from ``(seed, intensity, trial)``, so the whole
                curve is reproducible and each trial is an independent draw.
                Omitted, the stream is derived from the run seed instead. Either
                way the process-global RNG is never touched.

        Returns:
            dict with per-intensity CKA and metric degradation

        """
        from src.interp.representation_geometry import RepresentationGeometry
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        clean_reps = self.get_representations(input_ids)
        clean_geom = RepresentationGeometry.compute_all(clean_reps)
        thread_generator = self._accepts_generator(perturbation_fn)

        results = {}
        for i_int, intensity in enumerate(intensities):
            cka_values = []
            eff_dim_drops = []
            anisotropy_changes = []

            for trial in range(n_trials):
                # Keyed by (intensity, trial) rather than a running counter, so
                # the curve is a pure function of `seed`: dropping a trial or
                # reordering intensities does not relabel the others.
                stream = i_int * 1_000_003 + trial
                trial_gen = (
                    rng.derive_generator(
                        f"robustness.curve.{perturbation_fn_name(perturbation_fn)}", stream
                    )
                    if seed is None
                    else torch.Generator(device="cpu").manual_seed(
                        (int(seed) * 1_000_003 + stream) % (2**31 - 1)
                    )
                )
                perturbed_ids = (
                    perturbation_fn(input_ids, intensity, generator=trial_gen)
                    if thread_generator
                    else perturbation_fn(input_ids, intensity)
                )
                perturbed_reps = self.get_representations(perturbed_ids)

                # CKA between clean and perturbed
                flat_clean = clean_reps.reshape(-1, clean_reps.size(-1))
                flat_pert = perturbed_reps.reshape(-1, perturbed_reps.size(-1))
                n = min(flat_clean.size(0), flat_pert.size(0))
                cka = linear_cka(flat_clean[:n], flat_pert[:n])
                cka_values.append(cka)

                # Geometry degradation
                pert_geom = RepresentationGeometry.compute_all(perturbed_reps)
                eff_dim_drops.append(
                    clean_geom["effective_dimension"] - pert_geom["effective_dimension"],
                )
                anisotropy_changes.append(pert_geom["anisotropy"] - clean_geom["anisotropy"])

            results[intensity] = {
                "cka_mean": sum(cka_values) / len(cka_values),
                "cka_min": min(cka_values),
                "eff_dim_drop_mean": sum(eff_dim_drops) / len(eff_dim_drops),
                "anisotropy_change_mean": sum(anisotropy_changes) / len(anisotropy_changes),
                "n_trials": n_trials,
            }

        # Robustness score: area under the CKA curve
        if results:
            intensities_sorted = sorted(results.keys())
            cka_curve = [results[i]["cka_mean"] for i in intensities_sorted]
            # AUC (trapezoidal)
            auc = 0
            for i in range(len(cka_curve) - 1):
                dx = intensities_sorted[i + 1] - intensities_sorted[i]
                auc += (cka_curve[i] + cka_curve[i + 1]) / 2 * dx
            # Normalize by max possible AUC
            max_auc = (max(intensities_sorted) - min(intensities_sorted)) * 1.0
            robustness_score = auc / max_auc if max_auc > 0 else 0.0
        else:
            robustness_score = 0.0

        return {
            "perturbation_curve": results,
            "robustness_score": robustness_score,  # Higher = more robust
            "cka_at_max_intensity": results.get(max(intensities) if intensities else 0, {}).get(
                "cka_mean",
                0,
            ),
        }

    @staticmethod
    def compare(
        jepa_model,
        baseline_model,
        input_ids,
        perturbation_fn,
        intensities=(0.1, 0.3, 0.5),
        device="cpu",
        n_trials=3,
        seed=None,
    ):
        """Compare robustness between JEPA and baseline.

        THE KEY COMPARISON: if JEPA's CKA curve stays higher,
        JEPA is more robust.

        Args:
            seed: forwarded to **both** arms, so the two models are scored on the
                *same* perturbed inputs. That makes this a paired comparison;
                letting each arm draw its own perturbations would add between-arm
                variance to a difference whose whole point is the within-pair
                one.
        """
        jepa_rob = RepresentationRobustness(jepa_model, device)
        base_rob = RepresentationRobustness(baseline_model, device)

        jepa_result = jepa_rob.perturbation_curve(
            input_ids, perturbation_fn, intensities, n_trials, seed=seed
        )
        base_result = base_rob.perturbation_curve(
            input_ids, perturbation_fn, intensities, n_trials, seed=seed
        )

        return {
            "jepa_robustness_score": jepa_result["robustness_score"],
            "baseline_robustness_score": base_result["robustness_score"],
            "jepa_more_robust": jepa_result["robustness_score"] > base_result["robustness_score"],
            "jepa_curve": jepa_result["perturbation_curve"],
            "baseline_curve": base_result["perturbation_curve"],
        }


# ═══════════════════════════════════════════════════════════════
# Standard perturbation functions
# ═══════════════════════════════════════════════════════════════


def token_dropout(input_ids, intensity, pad_id=0, vocab_size=50304, generator=None):
    """Randomly replace tokens with PAD (zero out).

    Args:
        generator: optional caller-owned CPU ``torch.Generator``. Omitted, a
            private generator is taken from the run seed, so the perturbation is
            reproducible for a run and the process-global stream is untouched.
    """
    gen = generator if generator is not None else rng.counter_generator("robustness.token_dropout")
    mask = torch.rand_like(input_ids.float(), generator=gen) < intensity
    return input_ids * (~mask).long() + pad_id * mask.long()


def token_substitution(input_ids, intensity, vocab_size=50304, generator=None):
    """Randomly replace tokens with random tokens.

    ``generator`` has the same contract as in :func:`token_dropout`.
    """
    gen = (
        generator
        if generator is not None
        else rng.counter_generator("robustness.token_substitution")
    )
    mask = torch.rand_like(input_ids.float(), generator=gen) < intensity
    random_tokens = torch.randint_like(input_ids, 0, vocab_size, generator=gen)
    return input_ids * (~mask).long() + random_tokens * mask.long()


def token_permutation(input_ids, intensity, generator=None):
    """Randomly shuffle token positions (fraction = intensity).

    ``generator`` has the same contract as in :func:`token_dropout`. One is
    threaded through both permutations of a row so the row's shuffle and the
    shuffle of the positions it moves cannot come from two streams.
    """
    gen = (
        generator
        if generator is not None
        else rng.counter_generator("robustness.token_permutation")
    )
    B, T = input_ids.shape
    result = input_ids.clone()
    for b in range(B):
        n_shuffle = int(T * intensity)
        if n_shuffle < 2:
            continue
        idx = torch.randperm(T, generator=gen)[:n_shuffle]
        shuffled_idx = idx[torch.randperm(n_shuffle, generator=gen)]
        result[b, idx] = input_ids[b, shuffled_idx]
    return result


def embedding_noise(input_ids, intensity, model=None, embed_dim=768):
    """Add Gaussian noise to token embeddings before encoding.

    This requires hooking into the model's embedding layer.
    For simplicity, returns a modified input that simulates the effect.
    """
    # This is a placeholder — actual implementation requires
    # modifying the forward pass. For now, return input_ids unchanged
    # and note that this needs a custom forward hook.
    return input_ids  # TODO: implement with forward hook


def span_corruption(input_ids, intensity, span_length=5, mask_id=0, generator=None):
    """Corrupt spans of tokens (like training, but as perturbation).

    This is JEPA-specific: JEPA is TRAINED on span masking,
    so it should be MORE robust to this perturbation than MLM.

    ``generator`` has the same contract as in :func:`token_dropout`.
    """
    gen = (
        generator if generator is not None else rng.counter_generator("robustness.span_corruption")
    )
    B, T = input_ids.shape
    result = input_ids.clone()
    n_spans = int(T * intensity / span_length)  # 0 stays 0: low-intensity end of curves

    for b in range(B):
        for _ in range(n_spans):
            start = torch.randint(0, max(T - span_length, 1), (1,), generator=gen).item()
            result[b, start : start + span_length] = mask_id

    return result


# ═══════════════════════════════════════════════════════════════
# Full robustness battery
# ═══════════════════════════════════════════════════════════════


class RobustnessBattery:
    """Run all perturbation types and compare JEPA vs baseline."""

    PERTURBATIONS = {
        "token_dropout": token_dropout,
        "token_substitution": token_substitution,
        "token_permutation": token_permutation,
        "span_corruption": span_corruption,
    }

    @staticmethod
    def run(
        jepa_model,
        baseline_model,
        input_ids,
        intensities=(0.1, 0.3, 0.5),
        device="cpu",
        seed=None,
    ):
        """Run full robustness battery.

        Args:
            seed: forwarded to every perturbation, so the battery is
                reproducible and each perturbation type is scored on the same
                draws every run.

        Returns:
            dict with per-perturbation comparison

        """
        results = {}
        for name, perturb_fn in RobustnessBattery.PERTURBATIONS.items():
            result = RepresentationRobustness.compare(
                jepa_model,
                baseline_model,
                input_ids,
                perturb_fn,
                intensities,
                device,
                seed=seed,
            )
            results[name] = {
                "jepa_score": result["jepa_robustness_score"],
                "baseline_score": result["baseline_robustness_score"],
                "jepa_more_robust": result["jepa_more_robust"],
            }

        # Summary: JEPA wins on how many perturbation types?
        jepa_wins = sum(1 for r in results.values() if r["jepa_more_robust"])
        results["_summary"] = {
            "jepa_wins": jepa_wins,
            "n_perturbations": len(RobustnessBattery.PERTURBATIONS),
            "jepa_robustness_advantage": jepa_wins > len(RobustnessBattery.PERTURBATIONS) // 2,
        }

        return results
