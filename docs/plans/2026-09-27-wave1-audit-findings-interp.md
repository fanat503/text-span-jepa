# Part 3 — Interpretability layer audit (reproduced with experiments)

All numeric claims were executed against the repo's interpreter. No repo files
were modified.

**Verdict up front:** the headline number of this repo is a training-set score,
and the module built to catch that class of bug cannot fail.

---

## I1 — the headline linear-probe number has no held-out split  · critical · VERIFIED

`src/eval/probes.py`:
- `LinearProbe.evaluate` trains the classifier on `dataset` (L31-42), then
  computes `accuracy` by iterating **the same `dataset`** (L47-54).
- `FutureTokenProbe.evaluate` accumulates `total_correct` from logits computed
  **before** `opt.step()` on each training batch (L95-101), over 5000 steps of
  the same loader.

There is no split anywhere. `future_probe_d{d}` is a running *training*
accuracy. Every "JEPA linear-probe accuracy" derived from this is meaningless —
and this is the repo's most-quoted number.

Five more of the same class: `probe_generalization.source_accuracy`,
`probing_complexity` best-epoch validation, `structural_probe.source_spearman`,
and `workspace_validation`'s in-sample feature selection.

## I2 — the causal modules do not intervene on any computation  · critical · VERIFIED

`causal_intervention.py`: `intervention_predictability_score` (L115-125),
`ablation_comparison` (L200-215), `steering_comparison` (L226-251) run the
model **once**, then ablate/steer the *pooled output vector* and hand it to
`probe_fn`. No layer downstream ever sees the modified activation. `layer_idx`
is accepted (L85) and ignored. These are linear-algebra ops on a feature
vector, not causal interventions.

`causal_scrubbing.py`: `scrub_and_evaluate` calls `behavior_fn(model, input_ids)`
for the baseline (L72) but `behavior_fn(model, input_ids, h_override=…)` for
every scrubbed trial (L124), inside `try/except: return 0.0` (L126-127).

Executed: with the natural `behavior_fn` signature every scrubbed call raises
`TypeError`, is swallowed, returns `0.0`, so
`behavior_preservation_ratio = |0 − 0.7|/0.7 = 1.0` → **`hypothesis_valid:
True` unconditionally.** The failure of the intervention is reported as
validation of the hypothesis.

Same module, second cause: `h_resampled = h[torch.randperm(h.size(0))]`
(L82-83) shuffles the **batch** axis only. At `B=1` — the normal case —
`randperm(1) = [0]`, so the scrub is a literal no-op. Verified.

Third: `svd_directions_hypothesis`'s `except` returns `relevant=ones,
irrelevant=zeros` (L226-231). Failed SVD → nothing scrubbed → behaviour
trivially preserved → `hypothesis_valid=True`.

## I3 — the composite index is maximized by adding white noise  · critical · VERIFIED

`interpretability_index.py`:
- `compute()` averages over only the metrics present (L113-140); `compare()`
  calls it independently per side (L156-157). **The two indices have different
  denominators.** Executed: model A with 4 metrics vs model B with the same 4
  plus 20 catastrophic ones → `jepa_better=True, index_gap=+0.318` while the
  same call reports `jepa_wins = 0/24`.
- Hand-picked sigmoid midpoints/scales (L203-228). Measured: adding isotropic
  white noise to an 8-dim signal rep moves the index **0.121 → 0.695** using only
  the repo's own metrics.
- `index = 0.5` for an empty metric dict (L140) — a total failure to measure
  anything is reported as the neutral midpoint.

The docstring presents it as a weighted average of theoretically-grounded
quantities. It is a heuristic, and it rewards degradation.

## I4 — total_correlation reports estimator bias, and is inverted  · critical · VERIFIED

`information_theory.total_correlation` mixes estimators: `marginal_entropy` is
a **30-bin histogram** mean-per-dim (L307), `joint_entropy` is an **exact
Gaussian log-det** (L315-318).

Executed on *independent* Gaussian dims, where true TC = 0 by definition:

| D | reported TC | bias/dim |
|---|---|---|
| 64 | 92.2 nats | 1.44 |
| 256 | 375 nats | 1.46 |
| 768 | 1158 nats | 1.51 |

The metric measures D and the bin count, not dependence.

And `if N <= D: return 0.0` (L303), `if sign <= 0: return 0.0` (L316-317).
Executed: perfectly-correlated dims (maximal dependence) and "too few samples"
**both return 0.0 = "perfectly disentangled"**, and the index rewards lower TC.
The metric is inverted on exactly the inputs it should flag.

`compression_ratio` (L327-350) is worse: numerator is mean-per-dim entropy,
denominator `0.5·original_dim·(1+log 2π)`, and L345 is a no-op expression
statement. Executed: **0.03189 / 0.03185** for wildly different reps, 0.0 for a
constant one. A constant ≈3.2e-2 carrying no signal.

## I5 — DCI "informativeness" has a noise floor above the validator threshold  · critical · VERIFIED

`disentanglement.DCIMetrics` informativeness = `R.max(dim=0).values.mean()`
(L93) — a max correlation, not Eastwood's informativeness.

Executed on **pure noise** reps + factors:

| N | informativeness |
|---|---|
| 200 | 0.173 |
| 500 | 0.152 |
| 2000 | 0.080 |

`ground_truth.py:266` declares the pipeline valid when
`informativeness > 0.1`. **The validator's threshold sits inside the measured
noise floor** — it passes on noise.

Same module, `ModularityScore` (L287) is **inverted** against its own docstring:
ideal one-hot dims → `mod = 0`, maximally-mixed dims → `mod = 1`. Executed:
ideal one-hot scores 0.677, pure noise 0.632, and 1-informative-dim-of-4 scores
**0.759** — the most degenerate case wins. And L280-282 scores a
correlated-with-nothing dim as *perfectly modular*.

## I6 — ground_truth.py cannot fail  · critical · VERIFIED

- `validate_polysemanticity`: `pipeline_valid = frac_monosemantic > 0` (L208).
  Since `psi = 0.0` on every exception, **a completely broken PSI passes**.
- `validate_geometry` uses `10 < eff_dim < 60` (L226) and `0 < aniso < 0.99`
  (L228) — a random Gaussian 300×64 matrix lands inside both windows.
- `full_validation`: `pipeline_reliable = n_valid >= n_total - 1` (L305).
  With 4 tests, 3 passing — including 3 that cannot fail — is "reliable".

The module whose stated purpose is catching exactly these failures is blind to
them.

## I7 — polysemanticity: noise scores perfectly monosemantic  · critical · VERIFIED

`_compute_dim_psi` `except: return 0.0` (L146-147); the outer `except` returns
`mean_psi = inf` (L102-109). Any error silently becomes the
hypothesis-favouring value. Two opposite silent fallbacks in one function.

Executed: **pure isotropic Gaussian noise → `mean_psi = 0.0`,
`frac_monosemantic = 1.0`** (the best possible score), while a representation
with 2 genuinely well-separated clusters scores 0.489. Adding white noise
maximizes `frac_monosemantic`, which is in the index with weight 2.

The header (L16) and class docstring (L28) call PSI "**null-calibrated**". No
null calibration exists anywhere in the module.

## I8 — CKA is a function of N, and O(N³)  · high · VERIFIED

`cka_metrics._hsic` docstring claims "**unbiased** estimator" (L17); the code is
the **biased V-statistic** (L26), missing Kornblith's diagonal correction.

Executed on **independent** random matrices:

| N | `linear_cka` |
|---|---|
| 32 | 0.96 |
| 100 | 0.89 |
| 800 | 0.49 |

The value tracks N, not the representations.

Cost: `torch.trace(KH @ LH)` on N×N is O(N³) where O(N²D) suffices. Measured
N=4000, D=8 → **4.69 s**. At a realistic activation count (B=32, T=512 →
N=16384) that extrapolates to **≈2.4 h per CKA**, plus 1 GB per N×N buffer.

`except Exception: return 0.0` (L54-55, L94-95). Executed: a shape mismatch
(`linear_cka(N=100, N=120)`) → **0.0**, i.e. *maximum* dissimilarity. And
`ablation.compare_representations` (L439) hits exactly that whenever an ablation
yields a different sample count.

## I9 — CKA consumers are effectively dead at scale  · high · VERIFIED

- `layer_analysis.inter_model_cka` = L_j × L_b = **144 CKA calls** on full
  flattened `(B·T, D)` → 10¹³ flops at 4096 tokens.
- `stability.convergence_curve` (L52-62): `N = size(0)`, so for 3-D input only
  the first `B` **tokens** are compared. The whole curve is computed on ~64
  tokens. And both ternary branches of L52-56 are identical.
- `disentanglement.MIGScore` at D=768, K=5, N=2000 → **357 s**; at real
  `B·T = 32768` → **97.5 min**. `SAPScore.compute_nonlinear` D=768 → 277 s.
- `structural_probe` L48 builds a `(B,T,T,R)` intermediate → 537 MB at
  B=8,T=512,R=64.
- `information_theory.py:139` `x_norm @ y_norm.T` → 1 GB at N=16000.

## I10 — the statistical-rigor module is never called  · high · VERIFIED

`run_comparison.py:261-279` "Phase 7 — Running statistical tests" computes raw
differences and stores them under `results["statistical"]`.
**`statistical_tests.py` is never imported.** The one-command pipeline emits a
JSON report labelled "statistical" with no p-value, no CI, no effect size.

Worse, where it *is* used, it is wrong:
- `benjamini_hochberg` `corrected` uses naive `p·n/rank` (L288) while
  `significant` uses the step-up rule (L280-284). Executed with
  `p=[0.03, 0.049]`: `corrected=[0.06, 0.049]`, `significant=[0,1]` — a report
  claiming `p_value_bh=0.06` (not significant) next to `significant_bh=True`.
- `PairedPermutationTest` uses the **unpaired** pooled SD for Cohen's d (L226)
  where the paired value is `mean_diff / sd(diffs)` — systematically understating
  |d| in exactly the correlated case the test exists for.
- `BayesianComparison` is a **bootstrap** (L406-414) documented as a posterior:
  "credible interval", "97% probability JEPA is better". It also resamples `a`
  and `b` independently, discarding the pairing.
- Correction exists only inside `MetricComparisonReport.generate` (L518-530),
  which is **never called** outside its own module. `RobustnessBattery` declares
  `jepa_robustness_advantage = wins > 4//2` (L276) — ≥3 of 4 binary wins happens
  by chance **31%** of the time.

## I11 — the documented pipeline runs on random token IDs  · high · VERIFIED

`run_comparison.main()` builds `dummy_ids = torch.randint(0, 50304, (500,128))`
(L421) and runs the full protocol on it, writing `comparison_results.json` and
`summary.txt`.

And L194-203: `positions = arange(N).unsqueeze(1).expand(-1, D)` then
`InfoNCEEstimator.compute(jepa_reps, positions)`. "MI with position (surface)"
is **MI with the sample's index in extraction order**; the actual surface feature
(token identity) is computed nowhere. The module's headline hypothesis
(L13, "JEPA has LOWER MI with surface features") is never tested.

## I12 — ablation.py has silent no-ops  · high · REPORTED

- `AblatedModel.forward` ablates by subtracting `cfg.lambda_X * info["loss_X"]`
  guarded only by `if "loss_X" in info` (L211-224). A renamed loss key → the
  ablation silently does nothing and is still labelled as that ablation.
- `self.model.predictor.num_refine_steps` is mutated and restored around the
  forward (L183-196) with **no `try/finally`** → an exception leaves the model
  permanently at `num_refine_steps=0`.
- `ablate_ema()` is called once at init (L290, L397);
  `skip_ema_update()` (L242-245) is **never called anywhere** (grep-verified).
  The arm labelled "no EMA target" is not one.
- `run_all` special-cases `"full"` to **not train** and reports `final_loss: 0`
  (L318-325) — in every table the full model shows loss 0.
- `run_scaling_ablations` swallows every exception into `{"error": str(e)}`
  (L409-415) and keeps looping.

## I13 — `compare.py` passes the same tensor as both arguments  · high · VERIFIED

`diag.compute(reps.unsqueeze(1), reps.unsqueeze(1))` (compare.py L90-97;
run_comparison.py L180-181) — **x versus x**. Executed:
`svcca_online_target=0.9946, alignment=0.0, subspace_overlap=1.0`. These
tautologies are written into `comparison_results.json` under "collapse: all 50+
metrics". The `.unsqueeze(1)` also collapses the sequence axis.

`extract_linguistic_features` L200-205 computes "token length" as
`input_ids/max_id` and comments "short tokens tend to have lower IDs in BPE" —
**token ID magnitude is frequency rank, not length.** This "ground truth factor"
feeds the disentanglement metrics.

L244: `sum(1 for t in tokens if t[0].isupper() if len(t) > 0)` evaluates `t[0]`
**before** the length guard. Executed: `extract_linguistic_features(["The", ""])`
→ `IndexError`.

## I14 — workspace_validation gates on a random decoder  · high · VERIFIED

`validate_workspace_claim` with `sae=None` builds a **random untrained** `TopKSAE`
(L352-357) and still returns `subspace_similarity` and `workspace_claim_valid`.
Executed end-to-end: the untrained path returns `subspace_similarity=0.447`,
`placebo=0.399`, `workspace_claim_valid=False` — a verdict computed from a
random decoder, still emitted, still in the result dict.

`identify_workspace_features` docstring promises a fitted linear probe with
"high probe accuracy" (L133-135); the code computes an **in-sample
point-biserial correlation** over all N samples (L164-177). No probe, no
accuracy, no split — and the top 10% of F=8192 features is upward-biased by
multiple comparisons.

`compute_workspace_similarity` (L239-248) uses `svdvals(Q.T @ ws_ortho)` as
"cosines of principal angles". The correct object is the k×k Gram
`Qᵀ(PPᵀ)Q`; the singular values of the k×r cross-matrix are the angle cosines
only when k == r. Executed k=4, r=51: code gives mean_angle 71.8°, true
cosines give 79.7°.

The module header and docstring both promise "4. **Bootstrap CI for the
similarity**" (L16, L331). The bootstrap is on `ws_util`, and the emitted keys
are `ws_utilization_ci_*` (L387-388). **`subspace_similarity` — the number the
module exists to produce, gated at 0.8 — has no confidence interval of any
kind.**

## I15 — split resampling makes cross-model comparisons noise  · high · VERIFIED

`probing_complexity._train_probe` redraws `idx = torch.randperm(N)` (L119) **for
every depth and for every model** (L188-193, L222-223), and returns `best_acc` =
max over epochs of validation accuracy (L152-165) on a split with no test set.

Executed: `jepa_max_acc` for **identical data** ranges **0.76–0.89 across 8
seeds** — a 13-point swing from the split alone, against a `min_accuracy=0.7`
threshold.

`layer_analysis._train_linear_probe` redraws an unseeded split **for every
layer** (L61). Executed on **byte-identical** layers, `layer_uniformity` ranges
**0.882–0.983** — a 0.10 noise band, the same magnitude as the between-condition
effect it is meant to detect. And `layer_uniformity = 1 − std/mean` (L123) is
maximized by making all layers identical, which is untestable: 6 byte-identical
layers score 0.884 vs 6 independent random layers 0.907.

`feature_composition` L290-300: `z_baseline = z` is the **full dataset including
the top-activating samples**. Executed at N=100 (= n_top) → `mean_interference
= 1.4e-10`, **exactly zero**; every sample is both treated and control.

## I16 — the interp layer is entirely unseeded  · medium · VERIFIED

`src/utils/seed.py` is **never imported by anything in `src/interp/`**
(grep-verified). 28 unseeded draw sites across 15 modules, including inside a
training loop (`sae.py:150,162,167`) and inside a documented entry point
(`run_comparison.py:421`).

Worst offender: `ground_truth.py:74` calls `torch.manual_seed(seed)` on the
**process-global** RNG. Calling any `GroundTruthValidation` method silently
reseeds every unseeded site above — order-dependent across the whole suite.

## I17 — `polysemanticity.superposition_ratio` name and docstring are inverted  · medium · VERIFIED

Docstring says the Gram is `WᵀW` (L251, L276); the code computes `W @ W.T` (L277)
— output-unit correlations, not input-feature interactions.
`superposition_ratio = eff_rank/min(d,e)` (L300) is bounded by 1, so the
docstring's "**Ratio > 1 → superposition**" is unreachable. A **higher** value
means *less* superposition, so any reading of the name is backwards.
Executed: W=ones (rank-1, fully collapsed) → 0.031; W=gauss → 0.924.

## I18 — `test_interp.py` pins almost nothing  · medium · VERIFIED

638 of 1222 lines are smoke tests on `torch.randn` asserting `0 <= x <= 1` and
`"key" in result`.

- `test_dci_metrics` (L197-205) runs the DCI suite on **pure noise** and asserts
  only range — the suite is pinned to *firing* on noise, not to being right.
- `test_validate_geometry` (L1208) and `test_full_validation` (L1216) never
  assert `pipeline_valid` or `all_passed`.
- `test_ablated_model_forward` (L1005-1029) asserts only `isfinite(loss)`; it
  does not verify the ablated term was actually removed.
- `test_geometry_comparison` (L257-275) passes the **same** encoder as both JEPA
  and baseline and asserts `isinstance(result, dict)`.

Zero coverage (grep-verified, no import anywhere in `tests/`): `CausalScrubber`
(the unconditional-`hypothesis_valid` bug), `validate_workspace_claim`,
`identify_workspace_features`, `EarlyStoppingAdvantage`,
`MetricComparisonReport`, `ConditionalMIEstimator`, `AblationStudy`,
`FeatureCompositionScore`, `GeometryDegradationTest`, `RobustnessBattery`, and
`run_comparison` entirely.

The one meaningful property assertion in the file — `test_anisotropy_collapsed`
(L515-523) — is correct.

## What is genuinely correct

`direction_ablation` / `feature_steering` / `activation_patching` are correct
linear algebra (`activation_patching` appears only in tests).
`superposition_ratio` / `effective_rank` from an SVD are correct given the name
mismatch. `MIGScore`'s divide-by-contributing-factors fix is correct and
documented. `SAPScore`'s note on |ρ| ≡ OLS R² is correct. `Bonferroni` is
correct. The paired sign-flip permutation test itself is correct.
`SparseAutoencoder` + `SAETrainer` (TopK, dead-feature resampling, checkpointed
cadence) are sound. `visualization.py` is pure SVG rendering with no statistics
of its own.

---

## The pattern, stated plainly

The failures are not typos. They are one repeated structural mistake:

1. **A score computed on the data it was fit on.** I1, I4, I5, I7, I15.
2. **An exception handler that returns the hypothesis-favouring value.**
   I2 (three times), I6, I7. When a probe fails, it reports success.
3. **A metric with a trivial maximizer that is degradation.** I3, I5, I7, I8.
4. **A validator whose threshold sits below the noise floor of what it
   validates.** I5, I6, I8.
5. **A number whose name promises a guarantee the code does not implement.**
   I8 ("unbiased"), I11 ("MI with surface"), I12, I14 ("bootstrap CI"),
   I17 ("null-calibrated"), I10 ("Bayesian").

Every one of these is invisible to a passing test suite, because the tests
assert *shape* rather than *meaning* (I18).

The one module designed to catch this class — `ground_truth.py` — cannot fail
(I6). That is the most consequential finding in the whole audit, because it
means the repo's own safety net is inoperative precisely where it is needed.

---

## Still open

1. **Parameter counts** — never measured. The script failed on
   `TextSpanJEPAConfig.__dataclass_fields__` (not a dataclass). Claim that
   scaling-config names are off by 1.7-2.1× remains unverified.
2. **Performance/scaling audit** — the auditor was killed mid-probe when its
   measurement took 27% of the machine and 4.7 GB RAM. What it produced before
   dying is not in this document. The CKA cost analysis in I9 and I8 came from
   the interpretability auditor instead.
