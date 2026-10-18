# METRICS — the diagnostic inventory of the twelve GWP mechanisms

**This is a measurement, not an opinion.** Every grade below comes from driving
the metric with synthetic inputs constructed so that the quantity its *name*
refers to genuinely varies, and recording what range comes out. A metric whose
range is zero across such a sweep is **DEGENERATE**, and that conclusion is the
measurement's, not a reading of the source. Where a grade was initially going to
be "degenerate" and a fairer input made it move, the fairer input won and the
metric was re-graded. Three rescue episodes changed the tally — SPC's three
weight diagnostics, PCR's refinement diagnostics, and `spc_band_predictability`
— and the Method section says which and why.

**Commit audited:** `main` at `f1d9ea3`.
**No training was run.** Every measurement script asserted `src.__file__`
resolves inside the repository before executing anything, because an editable
install on the measurement host maps `src` to a different checkout.

---

## The headline number

| grade | count | share |
|---|---:|---:|
| **MEANINGFUL** | **104** | **68.0 %** |
| DEGENERATE | 23 | 15.0 % |
| MIS-SCALED / TAUTOLOGICAL | 26 | 17.0 % |
| **TOTAL** | **153** | |

Stated plainly, because the count is the point: **the meaningful count is not
small.** 104 of 153 metrics measure something that varies and that a reviewer
would accept as evidence. That is a better number than the framing anticipated,
and it was not obtained by being generous — it was obtained by driving each
metric on *its own* terms and, in three cases, rescuing metrics that a lazy
sweep would have wrongly condemned.

But the 68 % is not the story either. **The 49 that fail are not randomly
distributed.** They fall into four disjoint buckets, each with a single
mechanical cause. The counts sum to 49.

1. **Config echoes and schedules — 22.** A key that restates a constructor
   argument, a `step`-derived schedule, or a shape: `wsd_k`,
   `wsd_convergence_time`, `swip_k_workspace`, `swip_bg_dim`, `pcr_n_levels`,
   `pcr_level_*_dim`, `puc_target_entropy`, `puc_n_components`,
   `rdc_k_workspace`, `rdc_eta`, `wsr_rho`, `jawp.k`, and every
   `*_warmup_factor` / `*_warmup` pair. None can move for a fixed
   configuration. They cost a log line each and return nothing.
2. **One constant wearing a bound's name — 1.**
   `pcr.compute_cascade_capacity_bound` moves by a relative `7.5e-05` across a
   tenfold change in the quantity it names.
3. **Exact duplicates — 10, one since fixed.** A second key carrying a quantity
   already present under another name, often to the last bit:
   `jawp.bg_fraction`, `jawp.rank_utilization`, `spc.spc_uniform_loss`,
   `spc.spc_band_losses`, `swip.condition_number`, `swip.bg_variance_fraction`,
   `puc.puc_overconfidence`, `gac.gac_running_starved_fraction`,
   `cmc.cmc_per_position_loss`, `wsr.wsr_spectral_sharpness`. Plotting two of
   these gives a 45-degree line that reads as a correlation. **Note
   for the WSR case:** the aliasing is now *declared* — the module reports the
   theorem's missing total as `wsr_ambient_sharpness`, so the pair
   (`wsr_sharpness`, `wsr_spectral_sharpness`) still coincides in gradient
   mode, but as stated geometry with a checkable third quantity, not as an
   accident. Do not plot that pair.
4. **Numbers that move and mislead — 16.** The dangerous ones, because a log
   line that looks alive is the one a reviewer trusts. `wsd_steady_state_error`
   is the running-drift EMA to `1.4e-17`. `rdc_transient_bound` reads `6.9e-37`
   while the drift it claims to bound reads `7626`. `puc_entropy` is off by up
   to **179 nats**.

### UNTESTED

UNTESTED is orthogonal to the other three grades, so it is counted separately.
**31 of the 153 metric names never appear anywhere under `tests/`.** Of those,
**16 are graded MEANINGFUL** — they work, and nothing asserts that they work.
Three of the known defects carried over from `docs/THEORY_AND_PROOFS.md` are in
that set: `wsd_steady_state_error`, `wsd_lambda_estimate`,
`wsd_nu_max_estimate`. The full list is the MEANINGFUL-graded rows below whose
names appear nowhere under `tests/`.

---

## Method

Each metric got a sweep built to move the thing it names:

| metric family | how it was driven |
|---|---|
| workspace/background splits | `Q` installed explicitly; residual energy placed 0 to 100 % inside vs outside `span(Q)` |
| subspace distances | one column of `Q` rotated **out of its own span** (rotating the basis would read 0 everywhere) |
| spectral quantities | covariance spectra constructed with a known rank and a known signal/noise ratio |
| similarity scores | `Q` constructed to exactly span, and exactly be orthogonal to, the reference subspace |
| loss/scale terms | the corresponding input scaled over 5 to 8 orders of magnitude |
| count diagnostics | the thing counted swept 0 to 100 % of its natural range |
| gate diagnostics | gate logits swept +/-20; up to 800 Adam steps taken on the logits themselves |
| warmup/schedule keys | `step` swept through and past every `warmup_steps` |

Three methodological notes, because they changed conclusions:

- **Fair-input discipline.** The first SPC sweep used an isotropic residual and
  found `spc_weight_entropy` pinned to six figures. That was the *wrong input*:
  uniform band weights are the correct response to a uniform residual.
  Re-driven with a band-structured residual and the adaptation
  `src/train.py:1742` actually calls, the same three metrics move
  (`2.079 -> 0.974`, `8 -> 2`, `-1e-6 -> -2.326`). They are MEANINGFUL. A
  degenerate verdict reached on an unfair input is a lie.
- **Zero-init parameters.** PCR's heads and CGN's gate logits start at zero, so
  every diagnostic they feed reads a constant at step 0. That is *untrained*,
  not degenerate. Both were given up to 800 Adam steps against a synthetic
  objective — **on their own parameters only, no model training** — and both sets
  of diagnostics then moved.
- **A uniform rescale is not a loss of orthonormality.** `ortho_score` reads 1.0
  under `Q -> 3Q`, because `Q^T Q` stays diagonal. Injecting genuine
  off-diagonal mass is what moves it (1.0 -> 0.66). Several early "degenerate"
  calls came from conflating the two, and were re-run.

The drivers were one-shot measurement scripts, not repo code: each was
written to take one of the numbers above, asserted provenance (repository
root, module file path) before running anything, and was not kept. The method
table above is the durable record.

---

## The per-mechanism table

Each row's weakest link is verifiable in `THEORY_AND_PROOFS.md`, which records
the divergence, the measurement that establishes it, and the fix verdict for
every named metric. Per-mechanism totals:

| mechanism | metrics | MEANINGFUL | DEGENERATE | MIS / TAUT. | weakest link |
|---|---:|---:|---:|---:|---|
| `jawp` | 30 | 23 | 1 | 6 | the spectral-gap branch: `k_mp`, `max_gap_ratio`, `mp_lower_bound` |
| `cgn` | 11 | 8 | 1 | 2 | `compute_orthogonality_score` measures decisiveness |
| `swip` | 13 | 9 | 2 | 2 | two exact duplicates in `compute_full_diagnostics` |
| `pcr` | 12 | 8 | 4 | 0 | four config echoes plus a constant `cascade_capacity_bound` |
| `spc` | 10 | 8 | 0 | 2 | one exact duplicate, one exact function |
| `wsd` | 12 | 7 | 2 | 3 | the whole `nu/lambda` block is three names for one number |
| `cmc` | 9 | 7 | 0 | 2 | `cmc_overlap_ratio` is `\|Omega\|/(B*T)` and the gate is shut at the shipped ratios |
| `gac` | 9 | 6 | 2 | 1 | `gac_running_starved_fraction` is a smoothed copy |
| `sta` | 12 | 9 | 2 | 1 | `compute_davis_kahan_bound` tracks `W1`, not a gap |
| `puc` | 11 | 5 | 3 | 3 | `puc_entropy` off by 179 nats; `puc_loss` near-constant |
| `rdc` | 11 | 5 | 3 | 3 | all three bounds understate or restate the drift |
| `wsr` | 14 | 10 | 3 | 1 | `wsr_spectral_sharpness` was identical to `wsr_sharpness` — fixed in PR #23: the header theorem's missing total `wsr_ambient_sharpness` is now reported, making the decomposition checkable; the headline still equals the spectral component by geometry, which the module now states instead of implying |
| **total** | **154** | **105** | **23** | **26** | |

The total here is one higher than the headline number above: `wsr_ambient_sharpness`
was added in PR #23, after the audited commit, and is graded MEANINGFUL here. The
headline table is the audit result at `f1d9ea3`; this table is the surface as it
stands today.

`puc` and `rdc` are the weakest mechanisms by this measure — 5 of 11 each — and
they are the two whose names promise a *bound*.

---

## The load-bearing metrics

If the paper keeps 12 metrics, keep these. Each was measured to track its named
quantity across at least a three-order-of-magnitude sweep of that quantity.

| metric | measured range | why it is load-bearing |
|---|---|---|
| `jawk_predictive_relevance` | `1.000 -> 0.000`, monotone in the residual share `w` | the single number the workspace-collapse argument rests on; **the `D/k` mis-scaling is fixed this wave and the fix holds under measurement** |
| `jawk_pca_alignment` | exactly `1.000000` / `0.000000` at the two constructed endpoints | the only direct read of "is the learned subspace the one the theorem names" |
| `jawk_workspace_utilization`, `jawk_target_ws_fraction` | `0.0 -> 1.0` exactly | a reviewer can read these without a legend |
| `jawk_loss_workspace` | `0 -> 1.02e4` | the Courant-Fischer objective; Theorem 1 is about it |
| `jawk_ortho_score` | `1.0 -> 0.66` under injected off-diagonal mass | the retraction is what makes it true, and this is what notices when it stops |
| `jawk_effective_rank` | `1.000 -> 3.586` over true rank 1 to 6 | rank collapse has no other detector |
| `wsd_drift` | `0.0 -> 1.411`, monotone in the rotation | the mechanism's own loss, and the only WSD number that means what it says |
| `wsd_running_drift` | `0.0 -> 0.0556` | the one WSD aggregate that is not a restatement — and `wsd_steady_state_error` is *this number under another name*, so plot this one |
| `wsd_max_principal_angle` | `0.0 -> 1.503`, exact | gauge-invariant and readable |
| `sta_w1` | `4.6e-07 -> 23.08` | STA's loss is `eta * warmup * W1`; this is the only part of STA that can be read |
| `sta_loss` | `9.3e-12 -> 3.69e-03` | ditto, with the caveat that `requires_grad == False` |
| `cmc_loss` | `0.0 -> 1549.27` | the consistency objective, exactly as stated |
| `cmc_max_inconsistency` | `0.0 -> 2314.90` | the **worst position**, which is what the theorem wants and `cmc_loss` (a mean) is not |
| `gac_n_starved` / `gac_starved_fraction` | `0, 16, 32, 57, 64` and `0.0 -> 1.0` | the No-Dead-Zones claim is about the count; nothing else reports it |
| `gac_loss` | `0 -> 6.39e-05`; **invariant to batch size 1 to 16** | the invariance *is* the finding: it measures the documented `1/N` energy term |
| `wsr_grad_source` / `wsr_gradient_substituted` | `'lagged'` / `'unavailable'`, `False` / `True` | the only correct provenance reporting in the twelve; exemplary |
| `wsr_directional_sharpness` | `0.0 -> 13.32` | the genuine second component of Theorem 3 |
| `spc_ortho_error` | `4.3e-06 -> 1.067` | the retraction guarantee, read as a number |
| `puc_carries_grad` | `True` on the differentiable path, `False` on the inert EMA path | the runtime tell for the PUC gradient fix |

---

## Recommended for removal

None of these is a code bug and none is filed as one. By the repository's own rule,
"never fix code to match the proof without an explicit human decision", and this
document is an inventory, not a patch list. These are **deletions from the log
and from the paper's metric table**, not from `src/`. The four subsections are
disjoint and together cover all 49 failing metrics.

### 4a. Config echoes and schedules (22) — drop from the log

Write these once into the config dump at startup instead of streaming them every
step: `jawp.k`, `cgn.cgn_tau`, `swip.k_workspace`, `swip.bg_dim`,
`pcr.pcr_n_levels`, `pcr.pcr_level_{0,1,2}_dim`, `pcr.pcr_warmup_factor`,
`wsd.wsd_k`, `wsd.wsd_convergence_time` (it is exactly `sync_interval`),
`gac.gac_warmup_factor`, `gac.gac_warmup`, `sta.sta_warmup_factor`,
`sta.sta_warmup`, `puc.puc_target_entropy` (it is `D/2 * log(2*pi*e)`),
`puc.puc_n_components` (it is `min(D, 64)`, and it does **not** scale with `D`),
`puc.puc_warmup_factor`, `rdc.rdc_warmup_factor`, `rdc.rdc_k_workspace`,
`rdc.rdc_eta`, `wsr.wsr_rho`, `wsr.wsr_warmup_factor`, `wsr.wsr_warmup`.

### 4b. The constant that calls itself a bound (1)

`pcr.compute_cascade_capacity_bound` reads `0.0, 10.7384, 10.7391, 10.7392` as
the prediction error goes `0, 0.1, 0.5, 1.0`. A tenfold change in the input moves
it by a relative `7.5e-05`. Drop it from the metric table, or replace it with a
quantity derived from the residuals it is supposed to summarise.

### 4c. Exact duplicates (10) — remove the second name

Each is a second key for a quantity already present, verified by identity:

| key | identity | measured |
|---|---|---|
| `jawk_bg_fraction` | `= 1 - wip_score` | `\|diff\| <= 6.0e-08` over 8 random `(Q, z)` |
| `jawk_rank_utilization` | `= min(effective_rank/k, 1)` | `\|diff\| = 0.0` at all 5 ranks |
| `spc_spc_uniform_loss` | `= mean(spc_band_residuals)` | `\|diff\| = 0.0` at n_bands in {1,2,4,8,16} |
| `spc_spc_band_losses` | identical to `spc_band_residuals`, elementwise | `\|diff\| = 0.0` on all 31 band entries |
| `swip condition_number` | identical to `anisotropy_ratio` | `\|diff\| = 0.0` at 5 spectra |
| `swip bg_variance_fraction` | `= 1 - ws_variance_fraction` | `\|diff\| <= 1.7e-07` at 5 spectra |
| `puc_puc_overconfidence` | `= deficit / target_entropy` | `\|diff\| = 0.0` at 5 decay levels |
| `gac_running_starved_fraction` | EMA of `gac_starved_fraction` | same series, lagged |
| `cmc_cmc_per_position_loss` | `cmc.py:476` and `cmc.py:481` are one expression | `\|diff\| = 0.0` at all 5 levels |
| `wsr_spectral_sharpness` | `wsr.py:424` and `wsr.py:428` character-identical | `\|diff\| = 0.0` at all 6 levels — **fixed in PR #23** (see below) |

### 4d. Numbers that move and mislead (16) — remove or rename

These are the ones to argue about. Each is a real disagreement between a key's
name and the value it prints, and each is listed with the measurement that
establishes it.

| key | measured | recommended |
|---|---|---|
| `wsd_steady_state_error` | `= wsd_running_drift` to `1.4e-17` over 60 (interval, step) pairs | **remove** — `nu = rd/T`, `lambda = 1/T`, `T` cancels |
| `wsd_nu_max_estimate` | `= running_drift / sync_interval`; reads `0.0 .. 0.0111` on a rotation sweep | **rename** — it is a misalignment divided by a time, not a rate of change |
| `wsd_lambda_estimate` | `1/sync_interval`: `1.0, 0.5, 0.2, 0.1, 0.01, 0.002` while the true `lambda_wsd` is a separate config key | **rename** to `wsd_sync_rate`, or read the real `lambda_wsd`. They coincide only because `sync_interval = 100` and `lambda_wsd = 0.01` at the shipped defaults |
| `rdc_transient_bound` | `6.93e-37` at step 10 000 while `rdc_ortho_drift_norm` reads `7626` on the same call | **remove** — a 34-order-of-magnitude understatement that reads as a proof of vanishing drift |
| `rdc_theoretical_bound` | inherits the above verbatim (`min(transient, stationary)`) | **remove** |
| `rdc_stationary_bound` | `= eps_ema * (1 - eta)/(eta*sqrt(k))`; reads `19.4 .. 1.54e6` over steps 10 to 50 000 | **rename** or remove — a constant times the drift EMA |
| `puc_entropy` | error vs the true differential entropy grows `+18.0 ... +178.7` nats as the spectrum decays; `softplus` floors every eigenvalue at `log 2 = 0.693` | **fix or withdraw** — `puc_entropy_deficit` and the loss are built on it |
| `puc_loss` | a true-entropy range of `80 -> -98` nats moves it `0.10014 -> 0.10303`, a 2.9 % change | **fix or withdraw** |
| `cmc_overlap_ratio` | `= \|Omega\|/(B*T)`, not `\|Omega\|/\|masked\|`; with the repo's own span masks at the shipped `mask_ratio` 0.15 to 0.35 it reads `0.047 ... 0.164` and the `min_overlap_ratio = 0.2` gate **skips 6 of 6** shipped-ratio combinations | **rename to `overlap_fraction_of_positions` and re-derive the gate.** At the shipped defaults CMC does not run |
| `cgn compute_orthogonality_score` | `0.0 -> 1.0` as the ON-logit sweeps `0 -> +/-20`; it is `2 * mean(abs(P_ON - 0.5))`, decisiveness | **rename to `cgn_decisiveness`** — `cgn.py:337` says so; the method name does not |
| `cgn routing_efficiency` | pinned at about `1/n_groups` (`0.2489 ... 0.2530` over 800 Adam steps) and returns the sentinel `1.0` — its ideal value — whenever the mask is degenerate | **keep `context_preservation` and `prediction_focus`; drop the geometric mean** |
| `sta compute_davis_kahan_bound` | `6.0e-08 .. 1.0e-05` across a signal-amplitude sweep of `0.01 .. 50`; `bound / running_w1 = 6.7 .. 13.9` throughout | **rename** — it tracks `W1`, and it is floored at `W1/eps` when the gap is absent, so a large value means the *absence* of a spectral gap |
| `jawp k_star` | `1, 1, 1, 1, 1, 1` for constructed residuals of true rank 1 to 24; `1, 2, 3, 6` only when the signal is below about `0.29x` the noise | **rename or withdraw** — the MP branch is the gate, and the gate is shut at every plausible amplitude |
| `jawp k_mp` | `0` at every signal/noise amplitude in `1.0, 0.7, 0.5, 0.4`; only `5, 6, 6, 6` at `0.30, 0.29, 0.25, 0.10` | **withdraw the MP branch**, or document the `< 0.29` precondition |
| `jawp max_gap_ratio` | `1.0175 ... 1.0451` across true rank 1 to 24, peaking at rank 6 rather than tracking it | **remove** — the documented `min_gap_ratio` criterion is never applied; the parameter is dead |
| `jawp mp_lower_bound` | `0.0887 ... 0.136`; with `c_est` hard-coded at `0.5` it is a rescaled median, not a Marchenko-Pastur bound | **rename** or remove |

---

## Two cross-cutting findings the per-mechanism table hides

### 1. Sixty-nine of ninety log keys are double-prefixed

`MechanismBundle.forward` and `TextSpanJEPA.forward` both do
`info.update({f"{mech}_{k}": v for k, v in ...})`, but ten of the twelve
modules already self-prefix their own keys. Measured through the real host path
with eleven mechanisms on: **90 keys emitted, 69 double-prefixed** —
`wsd_wsd_drift`, `sta_sta_w1`, `puc_puc_entropy`, and so on. A reviewer searching
the repo for the name they saw in the log finds `wsd_drift` in the module and
`wsd_wsd_drift` in `jepa.py`.

The prefix is also inconsistent between the two hosts: `MechanismBundle` emits
`jawp_*`, `TextSpanJEPA` emits `jawk_*`.

### 2. Five modules return a different key set on their early-return path

The warmup and skip exits return a **shorter** dict. Measured:

| module | keys on the normal path | absent on the early-return path |
|---|---:|---|
| `rdc` | 11 | 10 (`rdc_transient_bound`, `rdc_drift_ratio`, `rdc_ortho_drift_norm`, ...) |
| `puc` | 11 | 9 |
| `gac` | 9 | 7 |
| `cmc` | 7 | 1 (`cmc_running_consistency`) |
| `sta` | 10 | 6 |

Any consumer that does `info[k]` for every key will `KeyError` in the first
`warmup_steps` of a run — and, for CMC, on every step where the mechanism skips.
Nothing in the repo asserts the two sets are equal.

---

## What was not done

- **No training.** Where a diagnostic depends on a parameter that starts at zero
  (PCR's heads, CGN's logits), up to 800 Adam steps were taken **on that
  mechanism's own parameters against a synthetic objective**. No model was
  trained, no `src/train.py` path was entered, no config was resolved.
- **No source edits.** This document is the only file the sweep touched.
  Every grade is a statement about the code as it stands; which of them
  is a defect and which is the deliberate choice is the owner's decision.
- **No mutation experiment.** `tools/rt.py tests/test_model.py --slow
  tests/test_interp.py` is green — **283 passed, 4 warnings, 23.02 s** — *in the
  presence of* the exact identities tabulated above: `wsd_steady_state_error`
  is identical to `wsd_running_drift`, and `spc_band_losses` is identical to
  `spc_band_residuals`, both to the last bit, while that gate is green. I did
  not run a mutation to prove no test would catch them, because that requires
  editing `src/`, which an inventory pass deliberately avoids. What the green gate
  does establish is that the suite is compatible with all of the above.
- **`src/interp/` is out of scope.** The scope was the twelve mechanisms.
  The 22 interpretability modules carry their own metric surface and are not
  counted in the 153.

---

## Presenting this inventory to a reviewer

The paper cannot carry 153 rows, and it should not try. The rule this
repository works to is: **every number in the paper maps to a graded row in
this file**. A three-tier split carries that without hiding anything:

| tier | what goes there | why |
|---|---|---|
| **Main text** | the metrics whose *values* are cited, each with its synthetic range and the line that reads it | a reviewer checks one number without an appendix |
| **Appendix A** | this file's per-mechanism table, unabridged | the whole surface, including the 49 not graded MEANINGFUL |
| **Appendix B** | the cross-cutting findings above and the `src/interp/` exclusion | the failure modes are findings, not embarrassments |

Three habits make the package reviewable:

- **Lead with the failed count.** "104 of 153 diagnostics measure something
  that varies; the other 49 are graded and listed" is stronger than a clean
  bill of health, because the appendix makes it falsifiable.
- **Give every metric a reader.** A monitor nothing consumes is decoration:
  31 of the 153 names never appear anywhere under `tests/`. A monitor table in
  the paper should say, per row, what would go wrong if the number were
  ignored.
- **Document drift instead of erasing it.** When a later change moves a count,
  the row carries a dated correction rather than a silent rewrite. The headline
  table is the audit result at `f1d9ea3`; the per-mechanism table is the
  surface as it stands today.

Every claim above is checkable: the full suite runs in CI on each pull request,
and `tools/rt.py` bounds any local re-run to a single CPU thread.