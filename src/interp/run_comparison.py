# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# ONE-COMMAND comparison pipeline: JEPA vs baseline
#
# Usage:
#   python -m src.interp.run_comparison \
#       --jepa_ckpt checkpoints/jepa_best.pt \
#       --baseline_ckpt checkpoints/mlm_best.pt \
#       --output results/comparison/
#
# This runs the FULL interpretability protocol and produces:
# 1. A comparison report (JSON) that states its own inferential limits
# 2. Layer-wise analysis
# 3. Information-theoretic analysis
# 4. Visualization plots (PNG)
# 5. Human-readable summary (TXT)
#
# WHAT THIS PIPELINE CAN AND CANNOT SUPPORT
# ----------------------------------------
# `run_full_comparison` is handed ONE checkpoint per arm and ONE dataloader.
# A claim of the form "JEPA beats the baseline" is a claim about the
# population of *training runs* on a population of *corpora*, so the units
# that matter are checkpoints and corpora -- and this signature supplies
# exactly one of each. The only thing it can resample is the ROW of a
# representation matrix, which is a strictly weaker question.
#
# The consequence is not stylistic. It was measured:
#
#   * With both models held completely FIXED and only the dummy-corpus seed
#     varied, the Phase 7 `effective_rank_online` difference ranged over
#     0.558 .. 1.987 -- a spread of 1.43, which is 2.6x the point effect the
#     report presents as a finding.
#   * Across independently INITIALISED models of one arm (a lower bound on
#     seed variance, since no training is involved), the within-arm spread
#     exceeded the between-arm difference for 3 of the 4 headline metrics.
#
# A p-value computed from row resampling conditions on the checkpoint. It is
# blind to exactly the component that decides the sign. Wiring one in would
# therefore convert an unmeasured variance into a confident-looking number,
# which is why this module emits an explicit refusal instead:
# `describe_comparison_design()` records the three variance components, which
# one is covered, and returns `verdict="NOT_TESTABLE"`. `main()` prints it.
#
# `paired_row_bootstrap_ci` is offered because it answers a real if narrower
# question -- would a different draw of sequences from THIS corpus flip the
# difference -- and is correct only when the two arms are resampled with the
# SAME row indices. It is opt-in (`n_bootstrap=0` by default) because it costs
# two full `CollapseDiagnostics.compute` calls per replicate: measured 90 s at
# B=100 and 453 s at B=500 for the production shape (N=500, D=768, 1 thread).

import argparse
import json
from pathlib import Path

import torch
from src.interp import rng
from src.utils.torchio import safe_torch_load
from src.utils.cka_metrics import linear_cka, rbf_cka


#: The metrics the report has always carried a "difference" for. They are NOT
#: four independent tests -- see `describe_comparison_design`.
COMPARISON_METRICS = (
    "effective_rank_online",
    "collapsed_dim_ratio_online",
    "mean_pairwise_cosine_online",
    "sv_entropy_online",
)


class ComparisonDesignError(RuntimeError):
    """The two arms were not scored on the same data, so nothing downstream holds.

    Every inferential statement this module makes is about a PAIRED comparison.
    The pairing is a property of the dataloader the caller passes, and nothing
    used to check it: `run_full_comparison` iterated the caller's dataloader
    once per arm, so a `shuffle=True` loader silently gave the two arms
    different sequences in a different order -- measured 99 of 100 rows
    misaligned. The report then compared unpaired matrices and called it a
    comparison.
    """


def assert_paired_extraction(jepa_ids, base_ids):
    """Refuse a comparison whose two arms did not see the same rows.

    Checked rather than assumed, because the whole design rests on it: the
    paired bootstrap resamples one row index into both arms, and a "which
    arm is better" verdict over two different corpora is not a weaker claim,
    it is a different claim.
    """
    if jepa_ids.shape != base_ids.shape:
        raise ComparisonDesignError(
            f"the two arms returned differently sized batches: "
            f"jepa {tuple(jepa_ids.shape)} vs baseline {tuple(base_ids.shape)}. "
            f"They must be scored on the same sequences in the same order; a "
            f"mismatch usually means one dataloader was consumed twice."
        )
    if not torch.equal(jepa_ids, base_ids):
        n_bad = int((jepa_ids != base_ids).any(dim=tuple(range(1, jepa_ids.dim()))).sum())
        raise ComparisonDesignError(
            f"the two arms were scored on DIFFERENT sequences: {n_bad} of "
            f"{jepa_ids.size(0)} rows differ. The usual cause is a dataloader "
            f"built with shuffle=True -- iterating it once per arm hands each "
            f"arm a different ordering, so the comparison is unpaired and "
            f"every interval computed from it is meaningless. Pass a "
            f"non-shuffling loader, or materialise the batches once."
        )


def majority_null_probability(n_wins, n_total):
    """P(wins >= observed | every win is a fair coin).

    The `wins > n // 2` rule appears in Phase 8 of this file and in
    `RobustnessBattery`. Under the null that the two arms are
    indistinguishable it fires with the probability below, which is the
    chance of seeing this many "JEPA wins" from nothing at all.
    """
    if n_total <= 0:
        return 1.0
    tail = sum(_comb(n_total, k) for k in range(n_wins, n_total + 1)) / (2**n_total)
    return tail


def _comb(n, k):
    from math import comb

    return comb(n, k)


def describe_comparison_design(
    n_checkpoints_per_arm=1,
    n_corpora=1,
    n_rows=0,
    paired=False,
    n_comparisons_published=0,
    n_metrics_in_family=0,
    n_bootstrap=0,
):
    """State what this run can and cannot conclude, and refuse the verdict.

    Returns the dict stored at ``results["comparison_design"]``. It is a pure
    function of its arguments so a test can pin the refusal without running the
    pipeline.

    The three variance components of "arm A beats arm B":

    1. seed / training-run variance -- one checkpoint per arm here;
    2. corpus variance              -- one corpus here;
    3. row-sampling variance         -- the only one this pipeline can reach.

    A row bootstrap estimates (3). The report's verdicts ("JEPA more
    monosemantic", "models represent different things", k-of-n wins) are
    statements about (1) and (2). When those are absent the verdict is
    ``NOT_TESTABLE`` rather than a number, because a number would be read as
    covering all three.
    """
    covered = []
    uncovered = []
    if paired and n_rows > 1:
        covered.append("row-sampling variance within one corpus, from one checkpoint per arm")
    else:
        uncovered.append("row-sampling variance (the arms are not paired, or N < 2)")
    if n_checkpoints_per_arm < 2:
        uncovered.append("seed / training-run variance (exactly one checkpoint per arm)")
    else:
        covered.append("seed / training-run variance")
    if n_corpora < 2:
        uncovered.append("corpus-to-corpus variance (exactly one corpus)")
    else:
        covered.append("corpus-to-corpus variance")

    verdict_supported = not uncovered
    return {
        "verdict": "SUPPORTED" if verdict_supported else "NOT_TESTABLE",
        "units_of_replication": {
            "checkpoints_per_arm": n_checkpoints_per_arm,
            "corpora": n_corpora,
            "rows": n_rows,
        },
        "arms_paired": bool(paired),
        "pairing_verified": True,
        "variance_covered": covered,
        "variance_not_covered": uncovered,
        "multiplicity": {
            "comparisons_published_in_this_report": n_comparisons_published,
            "metrics_in_the_statistical_block": n_metrics_in_family,
            "note": (
                "the report prints far more scalar comparisons than the "
                "statistical block holds, so no correction computed over the "
                "smaller family covers the larger one"
            ),
        },
        "paired_row_bootstrap": {
            "requested_resamples": n_bootstrap,
            "computed": n_bootstrap > 0,
            "covers": (
                "would a different draw of sequences from this one corpus flip "
                "the difference; it conditions on the checkpoint and therefore "
                "says nothing about seed or corpus variance"
            )
            if n_bootstrap > 0
            else "not computed (n_bootstrap=0)",
        },
        "required_to_support_a_verdict": [
            ">= 2 independently trained checkpoints per arm (different seeds), "
            "compared under the same protocol",
            ">= 2 corpora, or a held-out split disjoint from BOTH arms' "
            "training data -- a row bootstrap over one corpus cannot supply it",
            "one declared multiplicity family covering every comparison the "
            "report prints",
        ],
    }


def paired_row_bootstrap_ci(
    reps_a,
    reps_b,
    metric_name,
    n_bootstrap=200,
    alpha=0.05,
    seed=None,
):
    """Percentile interval for (metric(reps_a) - metric(reps_b)) over row resamples.

    PAIRED, and the pairing is the whole point: one row index is drawn per
    replicate and fed to BOTH arms, because the two arms were scored on the
    same rows. ``statistical_tests.BootstrapCI.compare`` draws ``idx_a`` and
    ``idx_b`` independently, which throws that away and charges each arm for
    the other's corpus noise -- measured 1.19x the interval width here on the
    same data.

    What the returned interval does and does not license is documented on
    ``describe_comparison_design``; this function is deliberately named for
    what it resamples rather than for a hypothesis test, because it is not one.
    """
    import numpy as np

    from src.models.collapse import CollapseDiagnostics

    n = reps_a.size(0)
    if n < 2:
        raise ComparisonDesignError(
            f"a row bootstrap needs at least 2 rows per arm, got {n}. "
            f"With one row there is no interval to report -- not a wide one, "
            f"none at all."
        )
    diag = CollapseDiagnostics()

    def value(reps):
        return float(diag.compute(reps.unsqueeze(1), reps.unsqueeze(1))[metric_name])

    gen = rng.generator_for(seed, f"run_comparison.paired_bootstrap.{metric_name}")
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        diffs[b] = value(reps_a[idx]) - value(reps_b[idx])

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_resamples": int(n_bootstrap),
        "alpha": alpha,
        "method": "paired row resampling, one index into both arms per replicate",
    }


def _full_model_state(ckpt, ckpt_path):
    """Return the whole-module state_dict held by a checkpoint.

    `src.train.save_checkpoint` writes `state["model"] = model.state_dict()` and
    stores nothing per-submodule. That single shape is the only one accepted
    here. The pre-`state_dict` format this function used to read named a
    hand-picked list of tensors, so restoring it with `strict=False` left every
    unnamed tensor at its initial value — on the toy fixture 26 of them, 14 of
    them trainable parameters inside the optimizer. That partial restore is
    exactly the resume divergence the current format removes, so a checkpoint
    without `"model"` is refused by name instead of being half-read.
    """
    from src.train import CheckpointLoadError

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        held = sorted(ckpt)[:8] if isinstance(ckpt, dict) else type(ckpt).__name__
        raise CheckpointLoadError(
            f"Checkpoint {ckpt_path} is not in the full-state_dict format written by "
            f"src.train.save_checkpoint: expected a top-level 'model' key holding "
            f"model.state_dict(), found {held}... . Refusing to load -- a partial "
            f"restore would silently leave unlisted tensors at their initial values.",
        )
    return ckpt["model"]


def load_model(ckpt_path, model_type="jepa", device="cpu"):
    """Load model from checkpoint.

    `ckpt_path` must be a checkpoint written by `src.train.save_checkpoint`.
    Every branch restores the same complete module with `strict=True`, so a
    shape or architecture mismatch is loud rather than a partial restore.
    """
    if model_type not in ("jepa", "mlm", "data2vec"):
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = safe_torch_load(ckpt_path, map_location=device)
    state = _full_model_state(ckpt, ckpt_path)

    if model_type == "jepa":
        from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

        config = TextSpanJEPAConfig()
        model = TextSpanJEPA(config)
    elif model_type == "mlm":
        from baselines.mlm_baseline import MLMBaseline

        model = MLMBaseline(
            vocab_size=50304,
            max_seq_len=512,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
    else:  # data2vec
        from baselines.data2vec_baseline import Data2VecTextBaseline

        model = Data2VecTextBaseline(
            vocab_size=50304,
            max_seq_len=512,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )

    # One call, one shape: encoder + target_encoder + predictor + decoder (+ the
    # mechanism buffers and, for data2vec, the regression head) all arrive in
    # `state`. The old per-module `.get()` reads missed every mechanism tensor
    # and the data2vec regression head entirely.
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def extract_representations(model, dataloader, max_batches=100, device="cpu", pool="mean"):
    """Extract representations from model encoder.

    Args:
        pool: 'mean' for mean pooling over sequence, 'none' for per-token.

    """
    all_reps = []
    all_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            ids = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
            all_ids.append(ids.cpu())

            if hasattr(model, "encoder"):
                h, _ = model.encoder(ids)
            else:
                h = model(ids)

            if pool == "mean":
                all_reps.append(h.mean(dim=1).cpu())
            else:
                all_reps.append(h.reshape(-1, h.size(-1)).cpu())

    return torch.cat(all_reps, dim=0), torch.cat(all_ids, dim=0)


def extract_layer_representations(model, dataloader, max_batches=50, device="cpu"):
    """Extract per-layer representations from model."""
    # Determine number of layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
        n_layers = len(model.encoder.blocks)
    else:
        n_layers = 12  # Fallback

    all_layer_reps = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            ids = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)

            if hasattr(model, "encoder") and hasattr(model.encoder, "get_intermediate_layers"):
                # Use the proper method (no enc.dropout reference)
                intermediates = model.encoder.get_intermediate_layers(ids)
                for i, layer_h in enumerate(intermediates):
                    if i < len(all_layer_reps):
                        all_layer_reps[i].append(layer_h.mean(dim=1).cpu())
            elif hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
                # Manual extraction with proper embedding handling
                enc = model.encoder
                x = enc.token_embedding(ids) + enc.pos_embedding[:, : ids.size(1), :]
                for i, block in enumerate(enc.blocks):
                    x = block(x)
                    if i < len(all_layer_reps):
                        all_layer_reps[i].append(x.mean(dim=1).cpu())
            else:
                # Fallback: just use final output repeated
                h = model(ids)
                for i in range(n_layers):
                    all_layer_reps[i].append(h.mean(dim=1).cpu())

    return [torch.cat(reps, dim=0) for reps in all_layer_reps if reps]


def run_full_comparison(
    jepa_model,
    baseline_model,
    dataloader,
    output_dir,
    device="cpu",
    max_batches=50,
    n_bootstrap=0,
    alpha=0.05,
    seed=None,
    surface_features=None,
):
    """Run the FULL interpretability comparison pipeline.

    Args:
        n_bootstrap: paired row resamples for the one interval this design can
            honestly produce. ``0`` (the default) computes none, because the
            interval conditions on the checkpoint and the report's verdicts do
            not; see the module docstring and ``describe_comparison_design``.
        alpha: two-sided level for that interval.
        seed: fixes the resampling. Omitted, it is derived from the run seed,
            so the interval is reproducible per run.
        surface_features: ``(N, K)`` per-sequence surface features, e.g. a
            one-hot bag of token ids. Phase 4's headline hypothesis is about
            MI with a surface feature; with no such tensor supplied there is
            nothing to compute it from, and the report says so instead of
            publishing the row index.

    Returns:
        The results dict, including ``results["comparison_design"]``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ================================================================
    # Phase 1: Extract representations
    # ================================================================
    print("[1/8] Extracting representations...")
    jepa_reps, jepa_ids = extract_representations(jepa_model, dataloader, max_batches, device)
    base_reps, base_ids = extract_representations(baseline_model, dataloader, max_batches, device)

    # Every inferential statement below is about a PAIRED comparison, and the
    # pairing was previously an accident of the caller passing shuffle=False.
    assert_paired_extraction(jepa_ids, base_ids)

    # ================================================================
    # Phase 2: Representation geometry
    # ================================================================
    print("[2/8] Computing representation geometry...")
    from src.interp.representation_geometry import RepresentationGeometry

    geom_compare = RepresentationGeometry.compare(jepa_reps, base_reps)
    results["geometry"] = {k: v for k, v in geom_compare.items() if not k.startswith("_")}

    # ================================================================
    # Phase 3: Collapse diagnostics (all 50+ metrics)
    # ================================================================
    print("[3/8] Computing collapse diagnostics...")
    from src.models.collapse import CollapseDiagnostics

    diag = CollapseDiagnostics()
    jepa_metrics = diag.compute(jepa_reps.unsqueeze(1), jepa_reps.unsqueeze(1))
    base_metrics = diag.compute(base_reps.unsqueeze(1), base_reps.unsqueeze(1))
    results["collapse"] = {
        "jepa": {k: v for k, v in jepa_metrics.items() if isinstance(v, (int, float))},
        "baseline": {k: v for k, v in base_metrics.items() if isinstance(v, (int, float))},
    }

    # ================================================================
    # Phase 4: Information-theoretic analysis
    # ================================================================
    print("[4/8] Computing information-theoretic metrics...")
    from src.interp.information_theory import (
        InfoNCEEstimator,
        RepresentationCompression,
        _align_width,
    )

    # MI with a SURFACE feature -- the module's headline hypothesis is
    # "JEPA has LOWER MI with surface features (token identity)".
    #
    # This used to build `positions = arange(N).expand(-1, D)` and report
    # InfoNCEEstimator.compute(reps, positions). Two measured facts make that
    # number meaningless rather than merely imprecise:
    #
    #   * every row of `positions` is CONSTANT, so F.normalize maps all of
    #     them to one direction and the similarity matrix has identical rows:
    #     cross_entropy == log(N) exactly, and the estimator's
    #     `max(mi, 0.0)` returns exactly 0.0 for EVERY representation --
    #     measured 0.0 for noise, all-zeros, all-ones, rank-1 and 1e6-scaled
    #     inputs alike. It measured the sample's extraction index, not a
    #     linguistic property.
    #   * `jepa_ids.float()` was called on the token ids -- the real surface
    #     feature, already in hand -- and its result DISCARDED.
    #
    # Substituting the raw ids is NOT the fix, and this was checked rather
    # than assumed: the ids still floor at 0.0, because with many rows
    # sharing a token the positive is one of several near-duplicates, the
    # loss exceeds log(N), and the same clamp censors the result. Deciding
    # what the surface feature IS, and encoding it, is a research decision
    # this module does not get to make silently.
    #
    # So the number is only reported when a caller supplies the feature, and
    # otherwise the report states that the hypothesis went untested.
    if surface_features is not None:
        sf = torch.as_tensor(surface_features).float()
        if sf.size(0) != jepa_reps.size(0):
            raise ValueError(
                f"surface_features must have one row per scored sequence: got "
                f"{sf.size(0)} rows for {jepa_reps.size(0)} representations."
            )
        # InfoNCE needs both operands at one width. `_align_width` is the
        # module's own lift for exactly this, and its docstring states why
        # zero-padding is the correct one: `<pad(a), b> == <a, b[:len(a)]>`,
        # so the inner products -- the whole estimand -- are preserved. Not
        # reimplemented here; duplicated padding would be a second law to keep
        # true.
        width = max(jepa_reps.size(-1), base_reps.size(-1), sf.size(-1))
        jepa_mi_pos = InfoNCEEstimator.compute(_align_width(jepa_reps, width), _align_width(sf, width))
        base_mi_pos = InfoNCEEstimator.compute(
            _align_width(base_reps, width), _align_width(sf, width)
        )
        surface_note = (
            "computed against the caller-supplied surface_features; the "
            "estimator clamps at 0, so a reported 0.0 means 'not above the "
            "estimator's floor', which is not the same as 'no dependence'"
        )
    else:
        jepa_mi_pos = None
        base_mi_pos = None
        surface_note = (
            "NOT COMPUTED. No surface feature was supplied. The previous "
            "arange(N) proxy was not a weaker surface feature, it was an "
            "arithmetic identity that returned 0.0 for every representation, "
            "so 'JEPA has lower MI with surface features' was 0.0 vs 0.0. "
            "Supply surface_features=(N, K) -- e.g. a one-hot bag of token "
            "ids -- to test the hypothesis."
        )

    jepa_entropy = RepresentationCompression.entropy_estimate(jepa_reps)
    base_entropy = RepresentationCompression.entropy_estimate(base_reps)

    jepa_tc = RepresentationCompression.total_correlation(jepa_reps)
    base_tc = RepresentationCompression.total_correlation(base_reps)

    results["information_theory"] = {
        "jepa_mi_surface": jepa_mi_pos,
        "baseline_mi_surface": base_mi_pos,
        "surface_mi_note": surface_note,
        "jepa_entropy": jepa_entropy,
        "baseline_entropy": base_entropy,
        "jepa_total_correlation": jepa_tc,
        "baseline_total_correlation": base_tc,
    }

    # ================================================================
    # Phase 5: Polysemanticity
    # ================================================================
    print("[5/8] Computing polysemanticity index...")
    from src.interp.polysemanticity import PolysemanticityIndex

    psi = PolysemanticityIndex(
        n_clusters_range=(2, 3),
        n_top_activations=50,
        n_dimensions_sample=20,
        device=device,
    )
    jepa_psi = psi.compute(jepa_reps)
    base_psi = psi.compute(base_reps)
    results["polysemanticity"] = {
        "jepa_mean_psi": jepa_psi["mean_psi"],
        "baseline_mean_psi": base_psi["mean_psi"],
        "jepa_frac_monosemantic": jepa_psi["frac_monosemantic"],
        "baseline_frac_monosemantic": base_psi["frac_monosemantic"],
    }

    # ================================================================
    # Phase 6: CKA similarity between models
    # ================================================================
    print("[6/8] Computing CKA similarity...")
    from src.models.collapse import CollapseDiagnostics

    cka_lin = linear_cka(jepa_reps, base_reps)
    cka_rbf = rbf_cka(jepa_reps, base_reps)
    svcca = diag._svcca(jepa_reps.unsqueeze(1), base_reps.unsqueeze(1))
    subspace = diag._subspace_overlap(jepa_reps.unsqueeze(1), base_reps.unsqueeze(1))
    results["cka_similarity"] = {
        "cka_linear": cka_lin,
        "cka_rbf": cka_rbf,
        "svcca": svcca,
        "subspace_overlap": subspace,
    }

    # ================================================================
    # Phase 7: what can actually be tested about these two arms
    # ================================================================
    print("[7/8] Establishing what is testable...")

    # This block used to be titled "Running statistical tests" and computed
    # `jepa_val - baseline_val` for four metrics, storing the result under
    # `results["statistical"]` -- a subtraction, under a name that promises a
    # test. `statistical_tests.py` was never imported, so no p-value, no CI
    # and no effect size reached the report.
    #
    # Wiring that module in does not fix it, and the reason was measured
    # rather than assumed:
    #
    #   * `PairedPermutationTest.compute` wants a per-SAMPLE vector. Phase 7
    #     has ONE scalar per arm per metric, and the function's N < 2 branch
    #     returns `p_value: 1.0, significant: False` -- which a report prints
    #     as "tested, not significant", i.e. "no difference found". Measured
    #     on the real inputs. It is not that: no test ran.
    #   * `BootstrapCI.compare` draws `idx_a` and `idx_b` INDEPENDENTLY, but
    #     the two arms are scored on the same rows. Measured on the same
    #     data, that unpaired interval is 1.19x wider than the paired one.
    #
    # So this emits the point values, the one interval that is honest (a
    # PAIRED row bootstrap, opt-in because it costs two full
    # `CollapseDiagnostics.compute` calls per replicate -- 90 s at B=100,
    # 453 s at B=500 for N=500, D=768 on one thread), and NO p-value and no
    # significance flag. A p-value here would condition on the checkpoint
    # while the verdicts in Phase 8 are about training runs.
    stat_results = {}
    for metric in COMPARISON_METRICS:
        jepa_val = float(results["collapse"]["jepa"].get(metric, 0.0))
        base_val = float(results["collapse"]["baseline"].get(metric, 0.0))
        entry = {
            "jepa": jepa_val,
            "baseline": base_val,
            "diff": jepa_val - base_val,
        }
        if n_bootstrap > 0:
            entry["paired_row_bootstrap"] = paired_row_bootstrap_ci(
                jepa_reps, base_reps, metric, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
            )
        stat_results[metric] = entry
    results["statistical"] = stat_results

    # These four are not four independent tests. Measured on paired row
    # resamples of the same two matrices, `effective_rank_online` and
    # `sv_entropy_online` have a bootstrap-difference correlation of r=0.9999
    # -- one measurement reported twice -- while `collapsed_dim_ratio_online`
    # is identically 0.0 on both arms. Reporting three numbers from one
    # spectrum as three findings is the multiplicity error this campaign keeps
    # finding; the design block records it rather than the report implying it
    # is corrected.

    n_published = (
        len([v for v in results.get("collapse", {}).get("jepa", {}).values() if isinstance(v, (int, float))])
        + len([k for k, v in results.get("geometry", {}).items() if isinstance(v, dict)])
        + 5  # information theory scalars (4 metrics + the surface note)
        + 4  # polysemanticity
        + 4  # cka_similarity
    )
    design = describe_comparison_design(
        n_checkpoints_per_arm=1,
        n_corpora=1,
        n_rows=int(jepa_reps.size(0)),
        paired=True,
        n_comparisons_published=n_published,
        n_metrics_in_family=len(COMPARISON_METRICS),
        n_bootstrap=n_bootstrap,
    )
    if surface_features is None:
        design["variance_not_covered"] = list(design["variance_not_covered"]) + [
            "the surface-feature MI hypothesis: no surface feature was supplied, "
            "so the module's headline claim went untested"
        ]
    results["comparison_design"] = design

    # ================================================================
    # Phase 8: Generate summary
    # ================================================================
    print("[8/8] Generating summary...")

    # Determine JEPA wins
    jepa_wins = {}
    # Geometry
    for key, val in results.get("geometry", {}).items():
        if isinstance(val, dict) and "jepa_better" in val:
            jepa_wins[key] = val["jepa_better"]

    n_won = sum(1 for v in jepa_wins.values() if v)
    n_total = len(jepa_wins)
    summary = {
        "n_geometry_jepa_better": n_won,
        "n_geometry_total": n_total,
        # `n_better > n_total // 2` is what Phase 8 and `RobustnessBattery`
        # both treat as an advantage. Under the null that the arms are
        # indistinguishable it fires with this probability, so the count is
        # reported next to the chance of producing it from nothing. Measured:
        # 0.3125 at 3-of-4, 0.3438 at 4-of-6, 0.3633 at 5-of-8.
        "geometry_majority_null_probability": majority_null_probability(n_won + 1, n_total)
        if n_total
        else 1.0,
        "geometry_majority_is_a_verdict": False,
        "jepa_more_monosemantic": jepa_psi["mean_psi"] < base_psi["mean_psi"],
        "jepa_higher_entropy": jepa_entropy > base_entropy,
        "jepa_lower_tc": jepa_tc < base_tc,
        "cka_similarity": cka_lin,
        "models_represent_different_things": cka_lin < 0.9,
        "verdict_supported": results["comparison_design"]["verdict"] == "SUPPORTED",
    }
    results["summary"] = summary

    # Save results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate human-readable summary
    generate_text_summary(results, output_dir)

    print(f"\nResults saved to {output_dir}")
    return results


def generate_text_summary(results, output_dir):
    """Generate human-readable text summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("Text-Span JEPA vs Baseline — Interpretability Report")
    lines.append("=" * 60)
    lines.append("")

    # Geometry
    lines.append("REPRESENTATION GEOMETRY")
    lines.append("-" * 40)
    for key, val in results.get("geometry", {}).items():
        if isinstance(val, dict) and "jepa" in val:
            j = val["jepa"]
            b = val["baseline"]
            winner = "JEPA" if val.get("jepa_better") else "BASELINE"
            lines.append(f"  {key}: JEPA={j:.4f}  Baseline={b:.4f}  [{winner}]")
    lines.append("")

    # Information Theory
    lines.append("INFORMATION THEORY")
    lines.append("-" * 40)
    it = results.get("information_theory", {})
    for key in [
        "jepa_entropy",
        "baseline_entropy",
        "jepa_total_correlation",
        "baseline_total_correlation",
    ]:
        if key in it:
            lines.append(f"  {key}: {it[key]:.4f}")
    if it.get("jepa_mi_surface") is None:
        lines.append("  jepa_mi_surface: NOT COMPUTED -- no surface feature supplied")
    else:
        lines.append(f"  jepa_mi_surface:    {it['jepa_mi_surface']:.4f}")
        lines.append(f"  baseline_mi_surface:{it['baseline_mi_surface']:.4f}")
    if it.get("surface_mi_note"):
        lines.append(f"  note: {it['surface_mi_note']}")
    lines.append("")

    # What this run can support. Printed FIRST among the conclusions, because
    # every line below it is a point estimate on one checkpoint per arm.
    design = results.get("comparison_design", {})
    if design:
        lines.append("WHAT THIS RUN CAN SUPPORT")
        lines.append("-" * 40)
        lines.append(f"  Verdict: {design.get('verdict', 'unknown')}")
        u = design.get("units_of_replication", {})
        lines.append(
            f"  Units: {u.get('checkpoints_per_arm')} checkpoint(s) per arm, "
            f"{u.get('corpora')} corpus/corpora, {u.get('rows')} rows"
        )
        lines.append(f"  Arms paired (verified): {design.get('arms_paired')}")
        lines.append("  Variance NOT covered:")
        for item in design.get("variance_not_covered", []):
            lines.append(f"    - {item}")
        lines.append("  A verdict would additionally require:")
        for item in design.get("required_to_support_a_verdict", []):
            lines.append(f"    - {item}")
        lines.append("")

    # Polysemanticity
    lines.append("POLYSEMANTICITY")
    lines.append("-" * 40)
    ps = results.get("polysemanticity", {})
    lines.append(f"  JEPA mean PSI:    {ps.get('jepa_mean_psi', 0):.4f}")
    lines.append(f"  Baseline mean PSI: {ps.get('baseline_mean_psi', 0):.4f}")
    lines.append(f"  JEPA frac mono:    {ps.get('jepa_frac_monosemantic', 0):.4f}")
    lines.append(f"  Baseline frac mono:{ps.get('baseline_frac_monosemantic', 0):.4f}")
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    s = results.get("summary", {})
    lines.append(
        f"  Geometry metrics JEPA better: {s.get('n_geometry_jepa_better', '?')}/{s.get('n_geometry_total', '?')}"
        f"  (chance of this many or more by coin flip: "
        f"{s.get('geometry_majority_null_probability', float('nan')):.4f} -- not a verdict)"
    )
    lines.append(f"  JEPA more monosemantic:  {s.get('jepa_more_monosemantic', '?')}")
    lines.append(f"  JEPA higher entropy:     {s.get('jepa_higher_entropy', '?')}")
    lines.append(f"  JEPA lower TC:           {s.get('jepa_lower_tc', '?')}")
    lines.append(f"  CKA similarity:          {s.get('cka_similarity', 0):.4f}")
    lines.append(f"  Models learn different:  {s.get('models_represent_different_things', '?')}")
    lines.append("")
    lines.append(
        f"  VERDICT SUPPORTED: {s.get('verdict_supported', False)}  "
        f"(see WHAT THIS RUN CAN SUPPORT above)"
    )

    text = "\n".join(lines)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Full JEPA vs Baseline comparison")
    parser.add_argument("--jepa_ckpt", type=str, required=True, help="Path to JEPA checkpoint")
    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        required=True,
        help="Path to baseline checkpoint",
    )
    parser.add_argument("--baseline_type", type=str, default="mlm", choices=["mlm", "data2vec"])
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["wikitext", "tinystories"],
        help=(
            "Named corpus. RECORDED IN THE REPORT ONLY -- this entry point "
            "does not load it. Passing --dataset wikitext did not make the "
            "run read wikitext; it fell through to the random-token corpus "
            "below and the report said nothing about the substitution."
        ),
    )
    parser.add_argument(
        "--allow_random_tokens",
        action="store_true",
        help=(
            "Required to run at all. Without a real dataloader this entry "
            "point builds dummy_ids = randint(0, 50304, (500, 128)) and runs "
            "the FULL protocol on it, writing comparison_results.json and "
            "summary.txt. Those artefacts look exactly like a real result, "
            "and every number in them is a property of uniform noise. The run "
            "is refused by default and stamped SYNTHETIC when allowed."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison/",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=0,
        help=(
            "Paired row resamples for the one honest interval (default 0, none). "
            "Costs two CollapseDiagnostics.compute calls per replicate: ~90 s at "
            "100 and ~453 s at 500 for N=500, D=768 on one thread."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for the placeholder dataloader. Omitted, it is derived "
        "from the process run seed, so the run is reproducible per run and "
        "never draws from the process-global RNG.",
    )
    args = parser.parse_args()

    # Load models
    print(f"Loading JEPA model from {args.jepa_ckpt}...")
    jepa_model = load_model(args.jepa_ckpt, "jepa", args.device)

    print(f"Loading {args.baseline_type} model from {args.baseline_ckpt}...")
    baseline_model = load_model(args.baseline_ckpt, args.baseline_type, args.device)

    # Create dummy dataloader if no real data available
    # (In production, this loads the actual dataset)
    print("Creating dataloader...")
    from torch.utils.data import DataLoader, TensorDataset

    if not args.allow_random_tokens:
        raise SystemExit(
            "run_comparison: REFUSED: this entry point has no corpus loader.\n"
            "  It would build dummy_ids = torch.randint(0, 50304, (500, 128)) and\n"
            "  run the full interpretability protocol on uniform noise, then write\n"
            "  comparison_results.json and summary.txt -- artefacts that are\n"
            "  indistinguishable from a real result. Every number in them would be\n"
            "  a property of the random draw.\n"
            "  Pass --allow_random_tokens to run it anyway; the report is then\n"
            "  stamped SYNTHETIC and carries no claim about either model.\n"
            "  To compare real checkpoints, call run_full_comparison() with a real\n"
            "  dataloader, or add a loader here (it is the one thing this file\n"
            "  does not have)."
        )

    # Private generator, not the global stream: this is a documented CLI entry
    # point, so two invocations of the same command previously produced two
    # different datasets with nothing in the output saying so. Now the seed is
    # on the command line and in the file name of what it produced.
    dummy_gen = rng.generator_for(args.seed, "run_comparison.dummy_dataloader")
    dummy_ids = torch.randint(0, 50304, (500, 128), generator=dummy_gen)
    dataset = TensorDataset(dummy_ids)
    # shuffle=False is now ASSERTED by assert_paired_extraction, not assumed.
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Run comparison
    results = run_full_comparison(
        jepa_model,
        baseline_model,
        dataloader,
        args.output,
        args.device,
        args.max_batches,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    results["corpus"] = {
        "synthetic": True,
        "requested_dataset": args.dataset,
        "note": (
            "SYNTHETIC CORPUS: uniform random token ids. --dataset was recorded, "
            "not loaded. No number in this report is a claim about either model."
        ),
    }
    with open(Path(args.output) / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
