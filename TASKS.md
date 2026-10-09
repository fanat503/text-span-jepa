<!-- swarm header — update every tick -->
X: 12 | TICK: 7 | cap: - | 429: 0 | volna: 7

# TASKS

Campaign against `text-span-jepa`. Every card below comes from a finding in
`docs/plans/2026-09-27-wave1-audit-findings*.md` that is **still open**. Cards
whose defect has already been fixed by the setup waves are not listed.

`mode: warmup` — 2-3 simplest cards, closed by TICK 1. They exist to prove the
whole pipeline end to end before any hard thinking happens. Their role is the
Euler problem: do the easy one first so the machinery is known good.
`mode: arena` — anywhere ≥2 reasonable approaches exist (performance, redesign,
anything >~40 lines). Runs as a tournament of Q=3.

Global rules that apply to every card and are not repeated in each one:
`.github/`, `.agent-notes/`, `TASKS.md` are forbidden. No skip, no xfail, no
`--no-verify`, no weakening or commenting out a test. If a test looks wrong,
investigate and raise it — do not touch it silently. Never run training. Run
tests only through `tools/rt.py` (1 thread, cumulative budget, no whole-suite).

---

### TASK-01 | group: G-COVER | status: done | mode: warmup
- goal: add `tests/conftest.py` with an autouse fixture that seeds every RNG and
  pins `torch.set_num_threads(1)`, so the suite cannot become order-dependent.
- why: 12 of 21 test files contain zero seeding calls and there is no conftest.
  `AGENTS.md` asserts tests are deterministic and nothing enforces it.
- files_allowed: `tests/conftest.py`
- files_forbidden: everything in `tests/` except the new conftest; `src/**`
- verify: `python tools/rt.py tests/test_sta.py tests/test_cgn.py`
  then confirm the fixture is autouse by running two files in one process and in
  reverse order and getting identical results.
- report: `.agent-notes/task-01.md`
- mutation-verdict: remove the seed call from the fixture → a test that reads a
  global RNG must become order-dependent; state which one and why.

### TASK-02 | group: G-FIX | status: done | mode: warmup
- goal: make `src/models/cmc.py:251-252` use the `rng` parameter it is handed
  instead of falling back to the global torch RNG.
- why: `src/train.py:1073` passes `rng=None`, so the dedicated generator
  parameter is dead code and the CMC mask stream is not the one the signature
  promises.
- files_allowed: `src/models/cmc.py`
- files_forbidden: `src/train.py` (owned elsewhere; report the call-site change
  needed instead of making it)
- verify: `python tools/rt.py tests/test_cmc.py`
- report: `.agent-notes/task-02.md`
- mutation-verdict: pass the global RNG again → a test asserting the generator is
  consumed must fail, because a supplied generator's state must advance and the
  global one's must not.

### TASK-03 | group: G-HYGIENE | status: done | mode: warmup
- goal: make `README.md`, `proofs/README.md` and
  `proofs/IMPLEMENTATION_STATUS.md` state the same mechanism count as the code.
- why: `mechanisms.py` header and `README.md` say 16, `ALL_MECHANISMS` has 12,
  `IMPLEMENTATION_STATUS.md` says 13 in its header with 12 rows and no WIP row,
  `proofs/README.md` says 13 and tabulates 11, and
  `tests/test_model.py:2262` is *named* `test_mechanism_bundle_counts_16` while
  asserting 12. A paper that cites any of these is wrong in its first paragraph.
- Decide and state the convention once, then make every site agree. Note that
  "16" comes from numbering five items under "Mechanism 1-5" (JAWP, WIP,
  Spectral Gap, Grassmann, Predictive Rank) while four of those five are
  *methods* on `JAWPModule`, and Predictive Rank has no `ALL_MECHANISMS` entry.
- files_allowed: `README.md`, `proofs/README.md`,
  `proofs/IMPLEMENTATION_STATUS.md`, `tests/test_model.py` (rename only, and
  only if the name is what lies)
- files_forbidden: `src/**`, `config/**`
- verify: `bash .agent-notes/gate.sh`
- report: `.agent-notes/task-03.md`
- decision: record the chosen convention in `docs/decisions.md`.

---

### TASK-04 | group: G-FIX | status: done | mode: solo
- goal: fix `src/train.py:836`, where `ema_tau_end` falls back to `1.0`, so a
  `--no_defaults` run no longer freezes the target encoder.
- why: `EMATauSchedule.step()` returns exactly `tau_end`, so the fallback makes
  `update_target_encoder` a no-op. The fallback is invisible whenever the defaults
  merge runs, which is why it survived.
- files_allowed: `src/train.py`
- files_forbidden: `config/**`, `defaults.yaml`
- verify: `python tools/rt.py tests/test_config_system.py --slow` and
  `python tools/rt.py tests/test_checkpoint_fidelity.py`
- report: `.agent-notes/task-04.md`
- note: `tests/test_config_system.py::TestNoDeadKeys::test_trainer_ema_fallback_is_not_the_frozen_value`
  is currently a documented skip. It must pass and the skip must be removed.
  Do not delete the test.

### TASK-05 | group: G-FIX | status: dup | mode: none
- DUPLICATE of TASK-29. Same goal, same files (`src/train.py` +
  `tests/test_config_system.py`), same fix. Closed with TASK-29, which is done.
- third duplicate found while selecting the wave-4 set, which means the card
  board had been written from findings without ever being diffed against itself.
  Card-writing now diffs against the board before it adds.
- goal: make the `src/train.py:849,874` config typo detector compare **paths**,
  not bare leaf names.
- why: it builds `{p.split(".")[-1] for p in _leaves(known)}` and tests the key,
  so `model.batch_size` (real key, wrong subtree) and an entirely misspelled
  section (`modle:`, `optimisation:`) are accepted silently while looking
  configured. `tests/test_config_system.py::TestTrainerTypoDetectorGap` holds
  three negative controls that are red today and must go green.
- files_allowed: `src/train.py`, `tests/test_config_system.py` (to retire the
  three negative controls once they pass)
- files_forbidden: `config/**`, `defaults.yaml`
- verify: `python tools/rt.py tests/test_config_system.py --slow`
- report: `.agent-notes/task-05.md`
- mutation-verdict: revert to leaf-name matching → the three negative controls
  go red again, naming the three cases that were silently accepted.

---

### TASK-06 | group: G-FIX | status: todo | mode: arena
- goal: wire `src/utils/distributed.py` into the training entry point so a
  `torchrun` launch is a real distributed run.
- why: the module exists and is import-inert, but `main()` still creates no
  process group, performs no all-reduce, seeds identically on every rank,
  injects no `DistributedSampler`, and gates no writes on rank 0. Running
  `scripts/wikitext/train_ddp.sh` today produces N independent replicas that
  write the same checkpoint filenames non-atomically. Any checkpoint that script
  has ever produced is unusable.
- known blocker: `src/datasets/kaggle.py:125` `make_dataloader` has no `sampler`
  parameter, so the sampler cannot be injected without changing it.
- files_allowed: `src/train.py`, `src/datasets/kaggle.py`, `src/utils/distributed.py`
- files_forbidden: `config/**`, `scripts/**`
- verify: `python tools/rt.py tests/test_distributed_helpers.py` plus a
  gloo-CPU single-process group in a fixture. Do NOT run a real multi-rank job.
- report: `.agent-notes/task-06.md`
- seeds: A minimal (wire only what exists) | B structural (make the workspace
  buffers consensus-bearing so `broadcast_buffers` is honest) | C bold (make
  them `nn.Parameter` so all-reduce covers them, and document the state-dict
  change).
- caution: `find_unused_parameters` cannot be a constant. `pcr.level_gates` and
  `spc.freq_basis` are in-graph only on steps where their branch fires, and
  `src/train.py:1127-1146` runs a second separate `backward()` for GAC.

### TASK-07 | group: G-FIX | status: done | mode: solo
- goal: repair `run_comparison.load_model()`, which reads per-module checkpoint
  keys that the new writer no longer emits.
- why: `save_checkpoint` now writes `state["model"] = model.state_dict()`. Four
  `load_state_dict` calls still look for `ckpt.get("encoder")` etc. and will
  raise on missing keys. The `mlm` branch already looked for `ckpt["model"]`,
  which the old writer never emitted, so it was broken before and is now correct.
- fix: replace each of the four `load_state_dict` pairs with
  `model.load_state_dict(ckpt["model"], strict=True)`.
- files_allowed: `src/interp/run_comparison.py`
- files_forbidden: `src/train.py`, `src/utils/torchio.py`
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-07.md`
- do not: re-emit the legacy keys as well. A new checkpoint read by old code
  would restore only the hand-listed tensors and silently reproduce the 29%
  resume divergence this campaign removed.

### TASK-08 | group: G-FIX | status: done | mode: solo
- goal: stop `target_centering.center` from being mutated under `eval()`.
- why: it was the last of the 24 buffers that `_validate` mutated, and the
  trainer-side snapshot-restore masks it rather than fixing it. Measured
  pre-fix delta 0.071. The other 23 are fixed at source; this one is not.
- files_allowed: `src/models/jepa.py`
- files_forbidden: `src/train.py`, `src/models/_state_guard.py`
- verify: `python tools/rt.py tests/test_training_state_guards.py` and
  `tests/test_model.py --slow`
- report: `.agent-notes/task-08.md`
- note: `jepa.py` is a hotspot. Confirm no other agent holds it this tick.

### TASK-09 | group: G-FIX | status: todo | mode: solo
- goal: fix `src/models/wsr.py` `_stiefel_retract` column-sign selection.
- why: it takes signs from `diag((Q R)[:k, :])`, which is not triangular for a
  `(D, k)` matrix when `D > k`, so the retraction flips individual columns.
  Measured: a random orthonormal Q with `rho=0.05` lands 4.0 away in Frobenius
  norm instead of ~0.05.
- blocked on: TASK-10. The sign fix and the definition of what `sam` should
  compute are one decision, not two.
- files_allowed: `src/models/wsr.py`
- files_forbidden: `src/models/_state_guard.py`, `proofs/**`
- verify: `python tools/rt.py tests/test_wsr.py tests/test_training_state_guards.py`
- report: `.agent-notes/task-09.md`
- gate: the xfail named in `docs/decisions.md` D-010 must pass and the xfail
  decorator must be gone. Leaving it is not acceptance.

### TASK-10 | group: G-FIX | status: todo | mode: arena
- goal: decide what `wsr_mode=sam` is supposed to compute, and make the code and
  its docstring agree.
- why: it reads `Q.grad` during forward, but `src/train.py:1226` calls
  `zero_grad()` with the default `set_to_none=True`, so `Q.grad` is always
  `None` and the orthonormality proxy is always substituted. On an orthonormal Q
  that proxy is exactly 0, so the loss is identically zero — measured
  `wsr_grad_norm 1.2e-13`, `loss_perturbed == loss_current == 2.1903`. The
  documented loss is never the one computed. The code now warns, which is
  honest, but a mode that always degenerates is not a mode.
- options: (a) feed the lagged gradient properly and fix the retraction signs
  (TASK-09 does the second half); (b) redefine the mode to match what the code
  actually does and rename it; (c) remove the mode and raise if anyone selects it.
- files_allowed: `proofs/wsr.md`, `src/models/wsr.py` (docstring only for the
  redefinition option)
- files_forbidden: `src/train.py`, `config/**`
- verify: `python tools/rt.py tests/test_wsr.py`
- report: `.agent-notes/task-10.md`
- decision: this is a research decision. Record the chosen option and the
  rejected ones in `docs/decisions.md`, and say which needs the owner.

---

### TASK-11 | group: G-COVER | status: done | mode: arena
- goal: give the five sibling probe metrics a held-out split, or rename them to
  say they do not have one.
- why: after `src/eval/probes.py` was fixed, five more modules still report
  training-set scores under names that do not say so:
  `probe_generalization.source_accuracy` (and therefore its
  `generalization_gap`, which is an overfit measure, and
  `generalization_ratio`, which is monotone in the overfit amount);
  `probing_complexity` best-epoch validation with no test set;
  `structural_probe.source_spearman`; and `workspace_validation`'s in-sample
  feature selection.
- files_allowed: `src/interp/probe_generalization.py`,
  `src/interp/probing_complexity.py`, `src/interp/structural_probe.py`
- files_forbidden: `src/eval/probes.py` (already correct, do not touch)
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-11.md`
- seeds: A split everything consistently | B split only where a held-out set is
  meaningful and rename the rest | C split, and add a leakage test per module
  (relabel the test split, assert the number moves).
- caution: `probing_complexity._train_probe` redraws `randperm(N)` for **every
  depth and every model**, so a JEPA-vs-baseline comparison is two different
  random draws. Measured: 0.76-0.89 across 8 seeds on identical data, against a
  `min_accuracy=0.7` threshold. Splitting without fixing the shared split does
  not fix this.

### TASK-12 | group: G-COVER | status: done | mode: solo
- goal: fix the split redraw in `src/interp/layer_analysis.py`.
- also covers: the former TASK-33, which duplicated this card.
- why: `_train_linear_probe` draws a fresh unseeded `torch.randperm(N)` for
  **every layer** and reports max-over-epochs validation accuracy, so the
  12-layer accuracy profile is 12 measurements on 12 different splits.
  Measured on byte-identical layers: `layer_uniformity` ranges 0.882-0.983
  across 5 seeds — a 0.10 noise band, the same magnitude as the between-condition
  effect it exists to detect.
- files_allowed: `src/interp/layer_analysis.py`
- files_forbidden: other `src/interp/**`
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-12.md`
- also: `layer_uniformity = 1 - std/mean` is maximized by making all layers
  identical, which the module cannot distinguish from a real result — 6
  byte-identical layers score 0.884 vs 6 independent random layers 0.907.
  Report it; fixing the direction is a separate decision.

### TASK-13 | group: G-COVER | status: done | mode: solo
- goal: stop `feature_composition.FeatureInterferenceScore` using the treated
  samples as its own control.
- why: `z_baseline = z` is the whole dataset including the top-activating
  samples. Measured at `N=100` (equal to `n_top`): `mean_interference = 1.4e-10`,
  exactly zero, because every sample is both treated and control. The score is
  attenuated by `n_top/N`, so it is not comparable across datasets. The feature
  subsample at `feature_idx = torch.randperm(M)[:n_test]` is also unseeded, so
  the reported number is not reproducible.
- files_allowed: `src/interp/feature_composition.py`
- files_forbidden: other `src/interp/**`
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-13.md`

### TASK-14 | group: G-COVER | status: done | mode: arena
- goal: make `src/interp/ground_truth.py` able to fail.
- why: it is the module whose stated job is catching exactly the failures this
  campaign found, and it cannot catch any of them.
  `validate_polysemanticity`: `pipeline_valid = frac_monosemantic > 0`, and PSI
  returns 0.0 on every exception, so a completely broken PSI passes.
  `validate_geometry`: `10 < eff_dim < 60` and `0 < aniso < 0.99` — a random
  Gaussian 300x64 matrix lands inside both windows.
  `full_validation`: `pipeline_reliable = n_valid >= n_total - 1`, so 3 of 4
  passing is "reliable".
  The `informativeness > 0.1` threshold also sat inside the measured noise floor;
  the new estimator's floor is much lower, so that one is already improved.
- files_allowed: `src/interp/ground_truth.py`
- files_forbidden: `src/interp/disentanglement.py`,
  `src/interp/information_theory.py`
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-14.md`
- requirement: every threshold must be **derived from a measured null on this
  data**, not chosen to look plausible. State the measurement for each.

### TASK-15 | group: G-COVER | status: done | mode: solo
- also covers: the former TASK-32, which duplicated this card.
- goal: fix the Benjamini-Hochberg implementation in
  `src/interp/statistical_tests.py`.
- why: `corrected` uses the naive `p*n/rank` (L288) while `significant` uses the
  step-up rule (L280-284). They disagree, so a report can carry
  `p_value_bh = 0.06` (not significant) beside `significant_bh = True`.
  Reproduced with `p = [0.03, 0.049]`: `corrected = [0.06, 0.049]`,
  `significant = [0, 1]`. The true BH adjusted p is the running minimum from the
  top rank, which the code omits.
- files_allowed: `src/interp/statistical_tests.py`
- files_forbidden: other `src/interp/**`
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-15.md`
- do not bundle: `PairedPermutationTest` uses unpaired pooled SD for Cohen's d
  where the paired value is `mean_diff / sd(diffs)`, and `BayesianComparison` is a
  bootstrap documented as a posterior with a "credible interval". Those are
  separate cards; note them in the report.

### TASK-16 | group: G-COVER | status: todo | mode: arena
- goal: decide whether `statistical_tests.py` should be wired into
  `run_comparison.py`, and what a rigorous comparison requires.
- why: `run_comparison.py:261-279` has a "Phase 7 — Running statistical tests"
  that computes raw differences and stores them under
  `results["statistical"]`. `statistical_tests.py` is **never imported**. The
  one-command pipeline emits a JSON report labelled statistical with no p-value,
  no CI and no effect size. Meanwhile `MetricComparisonReport.generate`, the only
  place correction is applied, is never called outside its own module, and
  `RobustnessBattery` declares `jepa_robustness_advantage = wins > 4//2` — 3 of
  4 binary wins happens by chance 31% of the time.
- files_allowed: `src/interp/run_comparison.py`, `src/interp/compare.py`
- files_forbidden: `src/interp/statistical_tests.py` (TASK-15 owns it)
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-16.md`
- seeds: A wire the existing machinery and report what it cannot support
  honestly | B redesign the comparison protocol so every verdict carries an
  effect size and a corrected p-value | C do not automate the verdict; emit the
  numbers and make a human sign off.
- also: `run_comparison.main()` builds `dummy_ids = torch.randint(0, 50304,
  (500, 128))` and runs the full protocol on it, and `positions =
  arange(N).unsqueeze(1).expand(-1, D)` makes "MI with position (surface)" an MI
  with the sample's extraction index. The surface feature is computed nowhere,
  so the module's headline hypothesis is never tested.

### TASK-17 | group: G-COVER | status: done | mode: solo
- goal: fix the silent no-ops and the missing cleanup in `src/interp/ablation.py`.
- why, all measured or read directly:
  - `AblatedModel.forward` subtracts `cfg.lambda_X * info["loss_X"]` guarded only
    by `if "loss_X" in info` (L211-224), so a renamed loss key makes the ablation
    a silent no-op while the run is still labelled as that ablation.
  - `self.model.predictor.num_refine_steps` is mutated and restored around the
    forward (L183-196) with **no `try/finally`**, so an exception leaves the
    model permanently at `num_refine_steps=0` for the rest of the run.
  - `run_scaling_ablations` swallows every exception into
    `{"ablation": ..., "error": str(e)}` (L409-415) and keeps looping, so a
    crashed ablation is indistinguishable from a successful one.
  - `ablate_ema()` is called once at init (L290, L397) and `skip_ema_update()`
    (L242-245) is called nowhere in the repo, so the arm labelled "no EMA target"
    is not one.
- files_allowed: `src/interp/ablation.py`
- files_forbidden: other `src/interp/**`
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-17.md`

### TASK-18 | group: G-COVER | status: done | mode: arena
- goal: fix `src/interp/workspace_validation.py`.
- why: `validate_workspace_claim` with `sae=None` builds a **random untrained**
  `TopKSAE` (L352-357) and still returns a verdict. Measured end to end:
  `subspace_similarity=0.447`, `placebo=0.399`, `workspace_claim_valid=False` —
  a verdict computed from a random decoder.
  `compute_workspace_similarity` uses `svdvals(Q.T @ ws_ortho)` as "cosines of
  principal angles" (L239-248); the correct object is the k x k Gram
  `Qᵀ(PPᵀ)Q`, and the cross-matrix singular values are the angle cosines only
  when k == r. Measured at k=4, r=51: code gives mean_angle 71.8°, the true
  cosines give 79.7°.
  `identify_workspace_features` promises a fitted probe with "high probe
  accuracy" (L133-135) and computes an in-sample point-biserial correlation over
  all N samples.
  The module header promises a "Bootstrap CI for the similarity" (L16, L331); the
  bootstrap is on `ws_util` and the number the module exists to produce,
  `subspace_similarity`, has no CI of any kind.
- files_allowed: `src/interp/workspace_validation.py`
- files_forbidden: other `src/interp/**`
- verify: `python tools/rt.py tests/test_interp.py tests/test_v025_integration.py --slow`
- report: `.agent-notes/task-18.md`
- requirement: a verdict computed from an untrained model must be refused, not
  returned with a caveat.

---

### TASK-19 | group: G-PERF | status: done | mode: arena
- goal: stop `CollapseDiagnostics.compute` from running every training step.
- why: measured **67% of the entire forward pass**. It performs 13
  `torch.linalg.svdvals` plus 4 full `torch.linalg.svd` on the full `(B·T, D)`
  activation matrix, unconditionally, at `jepa.py:872`. Forward went 1.60s ->
  4.88s at xsmall dims, B=8, T=256, fp32, 4 threads. It computes 56 metrics and
  `train.py:1260-1267` prints **4**, only every `log_freq` steps.
  `jspace_metrics.compute` at `jepa.py:880` repeats the pattern for 1 printed
  metric.
- files_allowed: `src/models/collapse.py`, `src/models/jspace.py`
- files_forbidden: `src/train.py` (own the config knob in the report instead),
  `config/**`, `defaults.yaml`
- verify: `python tools/rt.py tests/test_interp.py` plus a before/after timing
  in your report, median of 3 runs, recorded in `docs/campaign/perf.md`.
- report: `.agent-notes/task-19.md`
- seeds: A gate the whole thing behind `itr % log_freq == 0` | B gate the SVDs
  and keep the cheap closed-form metrics always on | C subsample `B·T` before
  the covariance and the SVDs.
- **caution, and the reason this is not a safe win on its own:**
  `collapse.py:518,663` call `torch.randperm(..., device=flat.device)`, which
  consumes the training-device RNG and perturbs DropPath. Those two sites need a
  private `torch.Generator` **first**, or gating the diagnostics changes results
  while looking like a pure optimisation. Do that in the same card and say so.

### TASK-20 | group: G-PERF | status: done | mode: arena
- goal: cut the device syncs in the per-step path.
- why: measured by patching `torch.Tensor.item` and tracing the caller —
  **92 syncs per forward** with the shipping config, **201** with all twelve
  mechanisms on. Each is a hard device sync.
  The cheapest large win is redundant: one forward with all mechanisms on
  performs **6 slices of `jawp.workspace_Q[:, :k_active]` and 4
  `active_k.item()` syncs** (`jepa.py:1008, 1038, 1059, 1066`), plus a 5th at
  `train.py:1189`. `spc.py:366,368,384` adds 24 syncs from per-band `.item()`.
- files_allowed: `src/models/jawp.py`, `src/models/spc.py`, `src/models/pcr.py`
- files_forbidden: `src/train.py`, `src/models/jepa.py`, `src/models/mechanisms.py`
- verify: `python tools/rt.py` on the three mechanism suites, and a sync count
  before/after in your report using the same `torch.Tensor.item` patch method.
- report: `.agent-notes/task-20.md`
- requirement: bit-identical. Hoisting 4 syncs and 6 slices to 1 and 1 is
  bit-identical; batching `spc`'s per-band `.item()` into one `.tolist()` is too.
  If your change moves a number, it is not this card.
- note: `mechanisms.py:461,475,499` slices `.data[:, :k]`, which silently
  detaches the Stiefel-projection gradient. That is a correctness issue, not a
  perf one, and `mechanisms.py` is not yours — report it.

### TASK-21 | group: G-PERF | status: done | mode: arena
- goal: halve peak activation memory under gradient accumulation.
- why: after `backward()` returns, `self._gac_z` (`jepa.py:650`) and
  `self._cmc_pass["slots"]` (`jepa.py:767`) still hold **live autograd graphs**
  (`grad_fn=True`). Both survive `optimizer.zero_grad()` at `train.py:1226`. With
  `grad_accum_steps` of 2/4/8 — which the scaling configs use — the previous
  micro-batch's graph is alive while the next is built, so peak activation
  memory is roughly 2x and scales with accumulation depth.
  The `info` dicts are clean: 125 `loss_dict` keys and 56 `diag_dict` keys were
  checked and contain zero tensors with `grad_fn`.
- files_allowed: `src/models/jepa.py`
- files_forbidden: `src/train.py`, `src/models/gac.py`, `src/models/cmc.py`
- verify: `python tools/rt.py tests/test_mechanism_wiring.py tests/test_gac.py tests/test_cmc.py --slow`
  plus a `tracemalloc` or `torch.cuda`-free peak measurement in your report.
- report: `.agent-notes/task-21.md`
- note: `jepa.py` is a hotspot and TASK-08 also wants it. They cannot run in the
  same tick. Order matters — see the tick plan.

### TASK-22 | group: G-PERF | status: done | mode: solo
- goal: make `tools`-free `src/utils/flops.py` either correct or gone.
- why: measured against `torch.utils.flop_counter.FlopCounterMode` at base_140m
  dims, vocab 4096, B=4: real fwd+bwd is 5.419e11 at T=128, 1.117e12 at T=256,
  2.440e12 at T=512. `flops.py` estimates 3.147e11, 6.295e11, 1.259e12 —
  ratios 0.58, 0.56, 0.52. The error **grows** with T because `6·N·L·B` has no
  attention term and no term for the repeated predictor passes, and the function
  has no `n_layers`/`embed_dim` in its signature, so it structurally cannot be
  right. It is imported by nothing but `src/utils/__init__.py` and one test.
- files_allowed: `src/utils/flops.py`, `tests/test_model.py` (the one test that
  imports it)
- files_forbidden: `src/models/**`, `config/**`
- verify: `python tools/rt.py tests/test_model.py --slow`
- report: `.agent-notes/task-22.md`
- decision: if you fix it, add the attention term, the target-encoder forward
  and the `(refine + 1 + len(offsets))` predictor multiplier, and pin it against
  `FlopCounterMode`. If you conclude it is not worth carrying, delete it and its
  test, and say so — but a test may not be deleted to make a linter pass, so if
  you delete this one, the report must justify why the test was testing a wrong
  expectation rather than a wrong implementation.

### TASK-23 | group: G-HYGIENE | status: todo | mode: solo
- goal: seed the 28 unseeded RNG draw sites in `src/interp/`, or make the
  unseeded ones take an explicit generator.
- why: `src/utils/seed.py` is **never imported by anything in `src/interp/`**.
  Draw sites include `sae.py:150,162,167` inside a training loop and
  `run_comparison.py:421` inside a documented entry point. Worst offender:
  `ground_truth.py:74` calls `torch.manual_seed(seed)` on the **process-global**
  generator, so calling any `GroundTruthValidation` method silently reseeds
  every other unseeded site — order-dependent across the whole suite.
- files_allowed: `src/interp/*.py`
- files_forbidden: `src/interp/ground_truth.py` (TASK-14 owns it),
  `tests/**`, `src/models/**`
- verify: `python tools/rt.py tests/test_interp.py`
- report: `.agent-notes/task-23.md`
- caution: hardcoded seeds that are not derived from the run seed — 42, and
  `20260824` — are reproducible but an ablation cannot vary them. Decide per site
  whether it should be a parameter. Record the decision list.

### TASK-24 | group: G-HYGIENE | status: done | mode: solo
- goal: make thread count and `deterministic` reachable, or state that they are
  not.
- why: measured 1 thread vs 8, same seed and same data, 10 steps: steps 0-2 are
  identical and **step 3 onwards differs** (`d = 1.19e-07` … `3.58e-07`), and the
  final `qkv` weight is not bitwise equal. So a run on a 4-core laptop is not
  reproducible on a 16-core server, and nothing in the repo or the environment
  pins it. Separately, `seed_everything(seed)` is always called with
  `deterministic=False` -> `cudnn.benchmark = True`, and no config or CLI key can
  request determinism.
- files_allowed: `src/utils/seed.py`, `defaults.yaml`, `config/scaling/*.yaml`
- files_forbidden: `src/train.py` (report the call-site change needed),
  `.github/`
- verify: `python tools/rt.py tests/test_torchio.py tests/test_config_system.py --slow`
- report: `.agent-notes/task-24.md`
- caution: the full-suite run in CI is on a 2-4 core runner while local runs are
  on 6, so CI and local results are already not bitwise comparable. Say whether
  that matters for any published number.

### TASK-25 | group: G-HYGIENE | status: done | mode: solo
- goal: make the parameter count the repo reports match the model it builds.
- why: `get_num_params()` at `jepa.py:1069` subtracts token and position
  embeddings from the encoder and **omits the target encoder entirely**, so the
  number `train.py:917-918` logs as "Model parameters" is ~2.5x smaller than the
  checkpoint it saves. Measured totals: `xsmall_30m` 62.5M (claims 30M),
  `small_100m` 170.7M (100M), `base_140m` 262.0M (140M), `large_300m` 538.0M
  (300M). The filenames track trainable parameters and are within 1-15%; the
  total is 1.8-2.1x the claim. `large_300m` needs 538M weights plus 1.79GB of
  AdamW fp32 state. A scaling paper reading the trainer's own startup log prints
  98.9M for a 262.0M model.
- files_allowed: `src/models/jepa.py` (the function only), `README.md`
- files_forbidden: `src/train.py`, `config/**`
- verify: `python tools/rt.py tests/test_model.py --slow`
- report: `.agent-notes/task-25.md`
- note: `jepa.py` is a hotspot and TASK-08 and TASK-21 also want it. Do not run
  those in the same tick.
- decision: report the true total, report trainable separately, or rename the
  configs. Record the choice and the rejected alternatives.

### TASK-26 | group: G-RAID | status: done | mode: solo
- goal: read-only re-scan for defects the five audits missed, in the three areas
  they covered least.
- why: five audits covered config, interpretability, reproducibility, integration
  and performance. They did **not** deeply cover `baselines/`, `scripts/`, or the
  `config/scaling/devices/` family added during setup. `scripts/` is 8 bash files
  and none of them run on the Windows dev host (`#!/bin/bash`, `set -euo
  pipefail`, `$(cd "$(dirname "$0")/.." && pwd)`, `CUDA_VISIBLE_DEVICES=0 python`).
  `src/datasets/kaggle.py:50,92` default to `/kaggle/input/...` absolute POSIX
  paths, and the fallback when `root_path` is absent is a **network** call to
  `load_dataset("wikitext", ...)`.
- files_allowed: **none. This card is read-only.**
- files_forbidden: everything
- verify: findings written to `.agent-notes/raid/seed-<n>.md`, one line each, in
  the seed format: finding | evidence `file:line` | benefit 1-5 | proposed card.
- report: `.agent-notes/raid/`
- note: every finding with benefit ≥4 goes to a control-scout whose job is to
  prove it is NOT a defect. Kills are recorded in `docs/decisions.md` with the
  reason, not silently dropped.

---

## Wave 2 cards (added after tick 1)

### TASK-27 | group: G-COVER | status: done | mode: solo
- goal: write the tests TASK-07 shipped without. `git grep` finds zero test
  references to `run_comparison`, and restoring the old file leaves the suite
  green.
- files_allowed: `tests/test_run_comparison.py` (new), `src/interp/run_comparison.py`
- verify: `& $PY tools/rt.py tests/test_run_comparison.py`
- must pin: (1) a real `save_checkpoint` -> `load_model` round trip with
  max-abs-difference exactly 0.0; (2) the mechanism tensors (jawp,
  target_centering, sigreg) present after the trip, because the old four reads
  covered 85 of 90 tensors and dropped the rest silently; (3)
  `regression_head.0.weight` restored bitwise, which is what a randomly
  initialised head would have failed; (4) a legacy-shaped checkpoint raises a
  clear format error rather than a bare KeyError or a silent partial restore;
  (5) `strict=True` is genuinely strict.
- mutation-verdict: revert `run_comparison` to the per-module `.get()` reads and
  record which tests go red.
- report: `.agent-notes/task-27.md`

### TASK-28 | group: G-COVER | status: done | mode: solo
- goal: pin the determinism fixture TASK-01 shipped without.
- why: the worker reported honestly that no existing test fails if
  `tests/conftest.py` is deleted, and the verifier reproduced it (168 tests green
  with the file renamed away). The guard is currently decoration.
- files_allowed: `tests/test_determinism.py` (new)
- files_forbidden: `tests/conftest.py` itself, all `src/**`
- verify: `& $PY tools/rt.py tests/test_determinism.py tests/test_sta.py`
- must pin: the fixture is autouse, so two tests observe different per-test
  streams; the seed derives from the node id and is therefore STABLE across
  runs, which is why `hash()` must not be used, since builtin string hashing is
  randomised per process; the global RNG state at the start of a test is a
  function of its node id.
- record the honest caveat: the suite is robustly non-flaky but cannot detect its
  own determinism regression, so this file is what closes that.
- report: `.agent-notes/task-28.md`

### TASK-29 | group: G-FIX | status: done | mode: solo
- goal: make the config typo detector compare paths, not bare leaf names.
- why: `src/train.py:849,874` builds `{p.split(".")[-1] for p in _leaves(known)}`
  and tests the key, so a key in the WRONG subtree, or an entirely misspelled
  section, is accepted silently. Verified non-detections: `model.batch_size`,
  `{"modle": ...}`, `{"optimisation": ...}`. A genuine nested typo
  (`model.lamda_swip`) IS caught.
- files_allowed: `src/train.py`, `tests/test_config_system.py` (retire the three
  negative controls once they pass)
- files_forbidden: `config/**`, `defaults.yaml`
- verify: `& $PY tools/rt.py tests/test_config_system.py --slow`
- note: `TestTrainerTypoDetectorGap` holds three negative controls that are red
  today. They must go green. Do not delete them: a red control is the
  specification.
- mutation-verdict: revert to leaf-name matching and confirm all three go red
  again, naming the three cases that were silently accepted.
- report: `.agent-notes/task-29.md`

### TASK-30 | group: G-FIX | status: done | mode: solo
- goal: make the baseline actually a control.
- why: `baselines/mlm_baseline.py:5-7` claims "identical model capacity /
  identical compute". The control-scout reproduced every number and found the
  claim false in the direction that FAVOURS the baseline. Corrected figures:
  JEPA total is 1.498x the baseline's, but the baseline's TRAINABLE count is
  1.281x JEPA's, and the entire excess is a separate untied `mlm_head` of
  32,194,560 parameters that JEPA does not have at all (JEPA's
  `TiedTokenDecoder` is 1,639,680 and reuses the token embedding). The baseline
  is also given 3.89x more prediction targets. A paper whose control is not a
  control is rejected at review, not at rebuttal.
- files_allowed: `baselines/mlm_baseline.py`, `tests/test_baseline_parity.py` (new)
- files_forbidden: `config/**`, `src/models/**`, other `src/interp/**`
- verify: `& $PY tools/rt.py tests/test_baseline_parity.py tests/test_model.py --slow`
- must pin: trainable-parameter parity within a stated tolerance, and a
  computation-parity statement that is either true or removed. If parity is
  genuinely unreachable without changing the science, say so and name what would
  have to change. Do not paper over it with a docstring.
- report: `.agent-notes/task-30.md`

### TASK-31 | group: G-HYGIENE | status: done | mode: solo
- goal: give each run its own `logging.folder`.
- why: the control-scout merged all 62 configs over `defaults.yaml` and found 60
  distinct folders, with exactly ONE collision in the entire repo: the three
  `config/kaggle/*.yaml` arms share one. That kills the obvious refutation that
  Kaggle sessions are ephemeral, since it is a deviation from the repo's own
  convention. Concurrent arms overwrite each other's `train_log.csv`, `best.pt`
  and `checkpoint-latest.pth.tar`.
- files_allowed: `config/kaggle/*.yaml`, `tests/test_config_system.py` (add a
  uniqueness guard only)
- verify: `& $PY tools/rt.py tests/test_config_system.py --slow`
- must pin: every shipped config resolves to a distinct `logging.folder`, with
  the three kaggle arms named.
- report: `.agent-notes/task-31.md`

### TASK-32 | group: G-COVER | status: dup | mode: none
- DUPLICATE of TASK-15. Same file, same defect, same fix. Closed with TASK-15.
- kept rather than deleted because the duplication is the useful record:
  the wave-2 block was written from RAID seeds without diffing against
  the cards already on the board, and two seeds independently re-stated
  two existing cards.
- lesson: a partial-line edit to a ledger orphans the rest of the card.
  This entry's first attempt replaced the header and the first goal line
  only, and TASK-32's body ended up filed under TASK-33 - which then read
  as a third card in the same file with a different goal. Edit whole cards.
### TASK-33 | group: G-COVER | status: dup | mode: none
- DUPLICATE of TASK-12. Same file, same defect, same fix. Closed with TASK-12.
- the goal it restated, kept so the duplication stays auditable: stop
  `layer_analysis` redrawing its split per layer.
### TASK-34 | group: G-FIX | status: todo | mode: arena
- goal: finish or refute the structural rival to TASK-14 seed A.
- why: TASK-14 was run as a tournament and only seed A completed. Seed B
  (the module measures its own null at call time, calibration travels with the
  verdict, nulls cached at module scope) produced a coherent 824-line diff and
  then its session died; three attempts to hand that diff to a fresh agent were
  interrupted before any of them ran a command. Seed C was never dispatched.
  So the structural approach is UNPROVEN, not refuted.
- why it still matters: a measured-null CONSTANT is what seed A shipped, and a
  corrected constant silently rots when the data changes, where a computed
  threshold cannot. Seed B's own notes put the cold cost of full_validation at
  roughly 4x with module-scope caching, and bound the per-test false-positive
  level at 1/(2*n_null+1). Those are claims to verify, not conclusions.
- evidence: `.agent-notes/task-14-seed-b-abandoned.diff` (verbatim, 97KB),
  branch `agent/task-14b`.
- files_allowed: `src/interp/ground_truth.py`
- files_forbidden: `src/interp/disentanglement.py`,
  `src/interp/information_theory.py`, all other `src/interp/**`
- verify: `& $PY tools/rt.py tests/test_interp.py`
- must decide: is the per-call cost acceptable on a 6-core box that is in use,
  measured cold AND warm? "Structurally better but 4x slower" is not better
  unless the caching holds.
- report: `.agent-notes/task-34.md`

### TASK-35 | group: G-FIX | status: done | mode: solo
- LOST, NOT CLOSED. The orchestrator's own error destroyed the work; read this
  before re-dispatching it.
- what happened: the worker's session was interrupted after it had staged a
  working fix (two baselines defaulting to the total, plus a 14-test
  `TestLoggedQuantityIsOneKind` guard). To undo an in-flight mutation test the
  orchestrator ran `git checkout HEAD -- <files>`, which restores from the
  BRANCH POINT and not from the index - so it discarded the staged fix along
  with the mutation. `git checkout -- <file>` restores from the index and is
  the correct command for undoing a mutation. Recovery via `git fsck` found only
  the two 244-byte mutation fragments, not the 20KB files, so the work is gone.
- lesson, the general one: restore a mutation from the INDEX, never from HEAD.
  A mutation you cannot undo is cheap; a worktree you destroy while trying to
  undo it is not.
- the fix as specified was sound, and was verified green before the loss:
  37 passed, ruff clean, black clean, the KIND guard present with 14 tests.
- goal: make all three comparison arms log the SAME parameter quantity.
- why: TASK-25 made `get_num_params()` report the TOTAL for JEPA. The two
  baselines kept their old default, so the three arms in the one directory the
  repo builds so those runs are comparable now log three different quantities,
  3.1x apart: jepa 262,021,633 / mlm 123,689,472 / data2vec 85,646,592. The
  worker that caused this reported it rather than calling it safe, which is the
  only reason this card exists.
- why this is worse than the old state: before TASK-25 all three were wrong in
  the same direction, so a comparison was at least consistently wrong. Now they
  are inconsistently wrong, which is harder to catch and easier to publish.
- files_allowed: `baselines/mlm_baseline.py`, `baselines/data2vec_baseline.py`
- files_forbidden: `src/models/**`, `config/**`, `src/train.py`
- verify: `& $PY tools/rt.py tests/test_baseline_parity.py tests/test_model.py --slow`
- must pin: a test that builds all three arms at one shape and asserts the
  logged quantity is the same KIND for all three, not merely that the numbers
  are close. Different architectures cannot have equal counts; equal KIND is
  what a comparison needs.
- also: `Data2VecTextBaseline` has no `get_num_params_trainable()` at all, so
  today only JEPA emits the like-for-like `Trainable parameters:` line.
- report: `.agent-notes/task-35.md`

---

### TASK-36 | group: G-HYGIENE | status: done | mode: solo
- goal: merge PR #11, which the GitHub API was refusing. DONE - the GraphQL
  failure was transient. PR #11 merged 2026-10-07T17:16:53Z as 1e49f77 with CI
  green at 1910 passed. Recorded rather than deleted because a transient API
  error that blocks a merge is worth a future tick knowing to simply retry.
- state: the work is SAFE and this card is only about the final click.
  `agent/wave-6` is pushed, PR #11 is OPEN, `mergeable: MERGEABLE`,
  `mergeStateStatus: CLEAN`, CI green: **1910 passed, 21 skipped, 1 xfailed,
  0 failed in 65.66s**. The local branch matches the remote head OID 9732de2
  exactly, so nothing is unpushed and nothing is lost.
- the blocker: `gh pr merge 11 --squash` fails with a GraphQL error, and REST
  `PUT /repos/.../pulls/11/merge` fails with `unexpected end of JSON input`.
  Three attempts, three invocations, same API-side failure. Not a
  branch-protection or conflict problem: the PR reports itself MERGEABLE/CLEAN.
- next steps in order: (1) retry, it may be a transient outage; (2) merge from
  https://github.com/fanat503/text-span-jepa/pull/11 with "Squash and merge";
  (3) if both fail, check whether a merge queue or auto-merge requirement was
  added after PR #10, since `gh pr merge` behaves differently then.
- do NOT merge locally and force-push main. main is protected and that would
  destroy reviewed history for no gain.
- verify: `git fetch && git log --oneline -1 origin/main` shows a new squash
  commit, and the 1910-test run is attached to it.
- report: `.agent-notes/task-36.md`

### TASK-37 | group: G-COVER | status: done | mode: solo
- goal: build the results apparatus, so the paper has a results section and not
  only a methods section.
- why: the whole campaign has been about correctness. CI is green at 1910 tests
  and 25 cards closed real defects, but the repo still has **no results table,
  no training curve and no ablation figure**, because running training is
  forbidden on this machine and the GPU belongs to the owner.
- what to build: everything computable WITHOUT training, so the only thing left
  for the owner is the run itself. A script that emits the results table in the
  form the paper needs, wired to the metrics the repo already computes, plus a
  stated estimate of what each run costs in wall-clock on their GTX 1650 so they
  can decide what to run first.
- the hard constraint: it must NOT fabricate, extrapolate or interpolate a
  number. A table with cells visibly marked "not yet run" is useful. A table
  with plausible-looking numbers is precisely the failure this campaign exists
  to prevent, and the owner would not be able to tell the difference at a glance.
- files_allowed: `scripts/**`, `docs/**`
- files_forbidden: `src/**`, `config/**`, `tests/**`
- verify: `& $PY tools/rt.py tests/test_config_system.py --slow`, plus running
  the script in dry-run and pasting the output
- report: `.agent-notes/task-37.md`

### TASK-38 | group: G-COVER | status: done | mode: solo
- goal: position every mechanism against the literature. Delivered as
  `docs/related_work.md`.
- verdict: **nothing is novel in isolation.** All twelve mechanisms are
  recombinations of published technique. Two survive as narrow COMBINATION
  claims: JAWP (JEPA prediction restricted to a learned k-dim subspace under SVD
  retraction) and SWIP (spectral shaping applied only OUTSIDE a protected
  subspace — VICReg and Barlow Twins shape the whole spectrum). Neither is a new
  mathematical object, and the Courant-Fischer step is textbook Ky Fan.
- fifteen refused claims are listed explicitly so a later draft cannot
  reintroduce them by accident.

### TASK-39 | group: G-FIX | status: done | mode: solo
- goal: **PUC contributes a constant to `total_loss` under the shipped defaults.**
- why this is the most serious defect the campaign has found, and it survived
  25 cards: with `use_differentiable_entropy=False` — a flag never exposed
  through `TextSpanJEPAConfig` — the PUC loss reads an EMA buffer with no
  autograd edge to `z_pred`, so it is a constant added to the objective.
  Measured by the worker that found it:
    use_differentiable_entropy=False   loss 0.1401  requires_grad False
    use_differentiable_entropy=True    loss 0.1037  requires_grad True   z.grad 1.4e-05
  It looks live: the config key is non-zero and `validate()` gates it. That is
  the worst shape a defect can have — a mechanism that appears in the ablation
  grid, passes a config check, contributes a number to a printed loss, and
  cannot influence training.
- files_allowed: `src/models/puc.py`, `defaults.yaml`
- files_forbidden: `src/models/mechanisms.py`, `src/train.py`, `proofs/**`
  (per AGENTS.md the proof is a design doc; record the divergence there, do not
  edit code to match it without a human decision)
- verify: `& $PY tools/rt.py tests/test_puc.py tests/test_sterility.py`
- must pin: a test that asserts the PUC loss carries an autograd edge to the
  encoder under the DEFAULT config. If the default is genuinely meant to be off,
  then `default: 0` in defaults.yaml and the mechanism must not appear in any
  shipped ablation arm — say which, and why.
- also: `use_differentiable_entropy` is unreachable from
  `TextSpanJEPAConfig`. Either expose it or delete it; an unreachable knob that
  changes the objective is a trap.
- report: `.agent-notes/task-39.md`

### TASK-40 | group: G-FIX | status: done | mode: arena
- goal: reconcile WSD and JAWP, which pull the SAME parameter in OPPOSITE
  directions with nothing tying them together.
- why: WSD pulls `Q` toward the **top-k** eigenvectors of the target covariance
  (highest variance); JAWP selects the **bottom-k** of the residual, and JAWP's
  own header names high-variance selection as the failure mode to avoid. Same
  `Q`, opposite optima. Found while writing the related-work section, and it is
  NOT in the audit matrix, so it has never been triaged.
- this is a science decision, not a bug: both may be correct if they act on
  different quantities at different times. Establish from the code which it is,
  and if it is a genuine conflict, the resolution is the owner's.
- files_allowed: `src/models/wsd.py`, `src/models/jawp.py`, `proofs/wsd.md`,
  `proofs/jawp.md`
- files_forbidden: `src/models/mechanisms.py`, `src/train.py`, `config/**`
- verify: `& $PY tools/rt.py tests/test_wsd.py tests/test_jawp.py tests/test_sterility.py`
- must also: two of JAWP's four numbered capabilities,
  `detect_workspace_dimension` and `grassmann_retract`, have NO caller in `src/`
  at all. Either wire them or delete them — a capability that appears in the
  header's count of 16 but is never called is the count mismatch in AGENTS.md
  made concrete.
- report: `.agent-notes/task-40.md`

### TASK-41 | group: G-COVER | status: done | mode: arena
- goal: the stale-documentation sweep. Everything below is already KNOWN to be
  wrong and is recorded nowhere as a task.
- why: 25 cards changed behaviour and the prose did not follow. A reviewer reads
  the prose. Known-stale, verified by this campaign:
  - `AGENTS.md` says "686 tests pass, ~108s". It is 1910 in ~66s on CI.
  - `AGENTS.md` says "16 mechanisms" against 12 modules and N_MECHANISMS=16.
  - the wave-1 audit plan still says the parameter count is 2.7x. It is 1.87x.
  - `layer_analysis` published numbers are invalidated (TASK-12).
  - `probe_generalization` `source_accuracy` / `generalization_gap` /
    `generalization_ratio` / `compare_models` outputs are invalidated (TASK-11);
    `probing_complexity` `depths` / `max_accuracy` / `min_extracting_depth` /
    `complexity_gap` likewise; `source_spearman` never executed at all.
  - `subspace_similarity` values are not comparable to new ones (TASK-18).
  - seven `CollapseDiagnostics` values changed by design (TASK-19).
  - `gac.py:162`'s docstring claims `z_pred` is "detached from graph". It is not.
    The code is right and the docstring is wrong.
  - `mlm_baseline.py` claimed "identical model capacity" (TASK-30, now removed).
- method: grep the docs for every NUMBER and every module name, then check each
  against the code. A number in prose that no longer matches the code is a
  defect of the same class as a wrong number in a metric, and this campaign has
  spent 25 cards establishing that distinction.
- do NOT rewrite history: dated records stay dated, but must be MARKED as
  superseded and say by which card. A dated audit that silently disagrees with
  the code is worse than one that says "as of 2026-08-24, wrong, superseded by
  TASK-30".
- files_allowed: `README.md`, `AGENTS.md`, `proofs/**`, `docs/**`,
  `config/ablations/README.md`
- files_forbidden: `src/**`, `config/**` except that `config/ablations/README.md`
  is allowed, `tests/**`
- verify: `& $PY tools/rt.py tests/test_config_system.py --slow`, plus for each
  correction the command that PROVES the new text is right
- report: `.agent-notes/task-41.md`
