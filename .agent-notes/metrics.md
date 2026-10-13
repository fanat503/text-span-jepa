# Metrics inventory — agent/metrics

**Branch:** `agent/metrics`, based on `main` at `f1d9ea3`.
**Worktree:** `C:\dev\wt-metrics`.
**Files I own and touched:** `docs/METRICS.md` (new) and `.agent-notes/*` (new).
No `src/`, no `tests/`, no `proofs/`, no `defaults.yaml`, no config.

---

## Counts

**153 metrics across the twelve modules in `MechanismBundle.ALL_MECHANISMS`.**

| grade | count | share |
|---|---:|---:|
| MEANINGFUL | 104 | 68.0 % |
| DEGENERATE | 23 | 15.0 % |
| MIS-SCALED / TAUTOLOGICAL | 26 | 17.0 % |
| **TOTAL** | **153** | |

UNTESTED is orthogonal and counted separately: **31 of the 153 names never
appear anywhere under `tests/`, and 16 of those are graded MEANINGFUL.**

**The meaningful count is not small, and I am saying so plainly.** 104 of 153
measure something that varies and that a reviewer would accept as evidence. The
card's framing anticipated a much worse number; the measurement did not support
it. What *is* bad is the shape of the remaining 49: 22 config echoes, 1 constant
bound, 10 exact duplicates, and 16 numbers that move and lie.

Per-mechanism (metrics / MEANINGFUL / DEGENERATE / MIS):

```
jawp 30/23/1/6   cgn 11/8/1/2   swip 13/9/2/2   pcr 12/8/4/0
spc  10/8/0/2    wsd 12/7/2/3   cmc  9/7/0/2    gac  9/6/2/1
sta  12/9/2/1    puc 11/5/3/3   rdc 11/5/3/3    wsr 13/9/3/1
```

Counts are **derived** by `.agent-notes/m_inventory.py`, not typed. Its stdout is
`.agent-notes/inventory.txt`.

---

## The per-mechanism table

Full grade per metric is in `docs/METRICS.md` and `.agent-notes/inventory.txt`.
Summary of where each mechanism's failures are:

- **jawp** — 6 MIS: `bg_fraction` (= 1 − wip_score), `rank_utilization`
  (= eff/k), and the whole spectral-gap branch `k_star` / `k_mp` /
  `max_gap_ratio` / `mp_lower_bound`. 1 DEGENERATE: `k` is a schedule.
- **cgn** — 2 MIS (`routing_efficiency`, `compute_orthogonality_score`),
  1 DEGENERATE (`cgn_tau` is a pure function of `step`).
- **swip** — 2 MIS, both exact duplicates inside `compute_full_diagnostics`
  (`condition_number` ≡ `anisotropy_ratio`, `bg_variance_fraction` ≡
  1 − `ws_variance_fraction`). 2 DEGENERATE config echoes.
- **pcr** — 4 DEGENERATE: three config echoes plus
  `compute_cascade_capacity_bound` (10.7384 → 10.7392 across a 10× input change).
- **spc** — 2 MIS, both exact (`spc_uniform_loss` ≡ mean of `band_residuals`;
  `spc_band_losses` ≡ `spc_band_residuals`). 0 DEGENERATE — I was wrong about
  three of these at first and re-measured; see "rescued" below.
- **wsd** — the `nu/lambda` block: `wsd_steady_state_error` ≡
  `wsd_running_drift` to 1.4e-17, `wsd_lambda_estimate` = 1/`sync_interval`,
  `wsd_nu_max_estimate` = `running_drift`/`sync_interval`, plus
  `wsd_convergence_time` ≡ `sync_interval`. 7 of 12 WSD metrics are MEANINGFUL;
  they are the drift/overlap/angle family and they work.
- **cmc** — 2 MIS: `cmc_per_position_loss` ≡ `cmc_loss` (`cmc.py:476` and `:481`
  are the same expression) and `cmc_overlap_ratio`, which is `|Ω|/(B·T)` rather
  than `|Ω|/|masked|`.
- **gac** — 2 DEGENERATE (warmup pair), 1 MIS (`gac_running_starved_fraction` is
  a smoothed copy of `gac_starved_fraction`).
- **sta** — 2 DEGENERATE (warmup pair), 1 MIS (`compute_davis_kahan_bound`
  tracks `W1`, and is floored at `W1/eps` when the gap is absent).
- **puc** — 5 MEANINGFUL out of 11, the weakest with rdc. `puc_entropy` is off
  by **+18.0 to +178.7 nats** against the true differential entropy, because
  `softplus` floors every eigenvalue at `log 2`. `puc_loss` moves 2.9 % across a
  true-entropy range of 80 to −98 nats.
- **rdc** — 5 MEANINGFUL out of 11. `rdc_transient_bound` reads **6.93e-37** at
  step 10 000 while `rdc_ortho_drift_norm` reads **7626** on the same call.
- **wsr** — 3 DEGENERATE (config echoes + warmup pair), 1 MIS
  (`wsr_spectral_sharpness` is character-identical to `wsr_sharpness`).

---

## Load-bearing for the paper

If the paper keeps 12 numbers, keep these. Each was measured to track its named
quantity across ≥3 orders of magnitude of that quantity:

1. `jawk_predictive_relevance` — `1.000 → 0.000`, monotone in the residual
   share `w`. **The `D/k` mis-scaling is fixed this wave and I re-verified the
   fix by measurement.** This is the number the workspace-collapse argument
   rests on.
2. `jawk_pca_alignment` — exactly `1.000000` when `Q` spans the target's top-k
   PCA and exactly `0.000000` when it spans the bottom-k.
3. `jawk_workspace_utilization` / `jawk_target_ws_fraction` — `0.0 → 1.0`
   exactly.
4. `jawk_loss_workspace` — `0 → 1.02e4`.
5. `jawk_ortho_score` — `1.0 → 0.66` under injected off-diagonal mass.
6. `jawk_effective_rank` — `1.000 → 3.586` over true rank 1…6.
7. `wsd_drift` — `0.0 → 1.411`, monotone in the rotation.
8. `wsd_running_drift` — the one WSD aggregate that is not a restatement, and
   therefore the one to plot in place of `wsd_steady_state_error`.
9. `wsd_max_principal_angle` — `0.0 → 1.503`, exact, gauge-invariant.
10. `sta_w1` and `sta_loss` — STA's loss is `η·warmup·W1`; these are the only
    parts of STA that can be read (`requires_grad == False`).
11. `cmc_loss` and `cmc_max_inconsistency` — the latter is the **worst position**,
    which is what the theorem wants and the mean is not.
12. `gac_n_starved` / `gac_starved_fraction` — the No-Dead-Zones claim is about
    the count; nothing else reports it. `gac_loss` is also worth keeping
    precisely because it is **invariant to batch size 1…16**, which is the
    measured form of the documented `1/N` energy term.

Also keep, as the correct parts: `wsr_grad_source` / `wsr_gradient_substituted`
(the only honest provenance reporting in the twelve), `wsr_directional_sharpness`
(the genuine second component of Theorem 3), `spc_ortho_error` (`4.3e-06 →
1.067`), and `puc_carries_grad` (`True` on the differentiable path, `False` on
the inert EMA path — a valid discriminator).

---

## Recommended for removal entirely

**22 config echoes / schedules** (drop from the log, write once into the config
dump): `jawp.k`, `cgn.cgn_tau`, `swip.k_workspace`, `swip.bg_dim`,
`pcr.pcr_n_levels`, `pcr.pcr_level_{0,1,2}_dim`, `pcr.pcr_warmup_factor`,
`wsd.wsd_k`, `wsd.wsd_convergence_time`, `gac.gac_warmup_factor`,
`gac.gac_warmup`, `sta.sta_warmup_factor`, `sta.sta_warmup`,
`puc.puc_target_entropy`, `puc.puc_n_components`, `puc.puc_warmup_factor`,
`rdc.rdc_warmup_factor`, `rdc.rdc_k_workspace`, `rdc.rdc_eta`, `wsr.wsr_rho`,
`wsr.wsr_warmup_factor`, `wsr.wsr_warmup`.

**1 constant wearing a bound's name**: `pcr.compute_cascade_capacity_bound`.

**10 exact duplicates** (remove the second name — each verified by identity):
`jawk_bg_fraction`, `jawk_rank_utilization`, `spc_spc_uniform_loss`,
`spc_spc_band_losses`, `swip.condition_number`, `swip.bg_variance_fraction`,
`puc_puc_overconfidence`, `gac_running_starved_fraction`,
`cmc_cmc_per_position_loss`, `wsr_spectral_sharpness`.

**16 that move and mislead** (remove or rename): `wsd_steady_state_error`,
`wsd_nu_max_estimate`, `wsd_lambda_estimate`, `rdc_transient_bound`,
`rdc_theoretical_bound`, `rdc_stationary_bound`, `puc_entropy`, `puc_loss`,
`cmc_overlap_ratio`, `cgn.routing_efficiency`, `cgn.compute_orthogonality_score`,
`sta.compute_davis_kahan_bound`, `jawp.k_star`, `jawp.k_mp`,
`jawp.max_gap_ratio`, `jawp.mp_lower_bound`.

---

## New findings not in `docs/THEORY_AND_PROOFS.md`

The theory audit listed five instances to start from. All five reproduced. These
are additional, found by the same method:

1. **`cmc_overlap_ratio` is `|Ω|/(B·T)`, not `|Ω|/|masked|`, and at the shipped
   mask ratios the gate is shut.** Measured with the repo's own
   `generate_second_mask` at `mask_ratio_start: 0.15` → `mask_ratio_end: 0.35`
   (`defaults.yaml:198-199`): the ratio reads `0.047, 0.094, 0.141, 0.125,
   0.141, 0.164` for the six shipped-ratio combinations and
   `min_overlap_ratio: 0.2` **skips 6 of 6**. It only starts running at ratio
   ≥ 0.5. CMC does not run at the shipped defaults.
2. **`spc_band_losses` is elementwise identical to `spc_band_residuals`**
   (`|diff| = 0.0` on all 31 entries across n_bands ∈ {1,2,4,8,16}).
3. **`spc_uniform_loss` is exactly `mean(spc_band_residuals)`** (`|diff| = 0.0`).
4. **`swip.condition_number` is exactly `swip.anisotropy_ratio`** (`|diff| =
   0.0` at 5 spectra) and **`bg_variance_fraction` is `1 − ws_variance_fraction`**
   (`|diff| ≤ 1.7e-07`).
5. **`puc_overconfidence` is exactly `deficit / target_entropy`** (`|diff| =
   0.0` at 5 decay levels) — a fixed rescale of another reported key.
6. **`jawp.bg_fraction` is exactly `1 − wip_score`** (`|diff| ≤ 6.0e-08` over 8
   random `(Q, z)`) and **`rank_utilization` is exactly `min(effective_rank/k, 1)`**
   (`|diff| = 0.0`).
7. **`cmc_loss` and `cmc_per_position_loss` are the same expression** —
   `cmc.py:476` and `cmc.py:481`.
8. **`pcr.compute_cascade_capacity_bound` is a constant** — `10.7384 →
   10.7392` across a tenfold change in prediction error.
9. **69 of the 90 keys the host emits are double-prefixed.** Ten of the twelve
   modules self-prefix their info keys and both hosts prepend the mechanism name
   again: `wsd_wsd_drift`, `sta_sta_w1`, `puc_puc_entropy`. Measured through the
   real `MechanismBundle.forward`. The prefix is also inconsistent between hosts
   (`jawp_*` vs `jawk_*`).
10. **Five modules return a shorter key set on their early-return path** —
    `rdc` (10 of 11 keys absent), `puc` (9), `gac` (7), `sta` (6), `cmc` (1).
    A consumer doing `info[k]` per key will `KeyError` in the first
    `warmup_steps` of a run, and on every CMC skip.
11. **`puc_entropy`'s error has a specific, fixable cause**: `softplus(λ)` floors
    every eigenvalue at `log 2 = 0.6931`, so `puc_min_eigenvalue` never reports
    anything below `0.693` however small it is. The entropy error grows with the
    spectrum decay the metric is supposed to measure (`+18.0 → +178.7` nats).
12. **`wsd_drift` reports 0 for a non-orthonormal `Q`.** Scaling a genuinely
    misaligned subspace by 1.5× drives `wsd_drift` to `0.000000` and
    `wsd_overlap` to `1.0000`. Confirms and sharpens D-WSD-3.

---

## Corrections to my own earlier verdicts

Recorded because a wrong verdict is worse than no verdict:

- **SPC's `spc_weight_entropy`, `spc_n_significant_bands` and `spc_spectral_tilt`
  are MEANINGFUL, not degenerate.** My first sweep used an isotropic residual
  and found all three pinned to six figures. That was the wrong input: uniform
  band weights are the correct response to a uniform residual. With a
  band-structured residual and the `adapt_weights_to_predictability()` call
  `src/train.py:1742` actually makes, they move: entropy `2.079 → 0.974`,
  significant bands `8 → 2`, tilt `−1e-6 → −2.326`.
- **PCR's `pcr_improvement`, `pcr_total_refinement_norm` and the per-level
  `correction_norm` / `gate` are MEANINGFUL, not degenerate.** Their heads are
  zero-initialised, so they read 0 before training — that is *untrained*, not
  degenerate. After 300 Adam steps on the module's own parameters:
  improvement `0 → 0.4987`, refinement norm `0 → 39.58`, gate `0.500 → 0.570`.
- **`spc_band_predictability` is genuinely per-band.** My first sweep gave four
  near-identical values; that was an isotropic-residual artefact. With
  band-structured residuals it reads `7e-05, 0.00927, ..., 0.00990`.

---

## Provenance and verification

Every script asserts `os.path.abspath(src.__file__)` starts with
`C:\dev\wt-metrics` before doing anything, and prints the resolved path. This
matters: an editable install on this box maps `src` to a different clone.

**Required gate:**
```
$PY tools\rt.py tests\test_model.py --slow tests\test_interp.py
-> 283 passed, 4 warnings in 22.50s
```
**Mutation verdict:** that gate is green *while* `wsd_steady_state_error` is
bit-identical to `wsd_running_drift` and `spc_band_losses` is bit-identical to
`spc_band_residuals`. I did **not** run a mutation experiment, because that needs
edits to `src/`, which is outside my file boundary. What is established is that
the suite is compatible with every defect listed here.

`ruff check .` — no Python files under `docs/`, clean.

No training was run. No config was resolved. No `src/`, `tests/`, or `proofs/`
file was modified.

---

## Commit

`docs(METRICS): a measured inventory of the twelve mechanisms' 153 diagnostics`
— adds `docs/METRICS.md` and the `.agent-notes/` measurement harness. 104 of 153
metrics MEANINGFUL; 22 config echoes, 10 exact duplicates, 1 constant bound and
16 numbers that move and mislead, each with the range that establishes it.