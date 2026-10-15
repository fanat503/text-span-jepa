# Text-Span JEPA

> **Submission status: there are no results in this repository.** Not one training
> run has been performed, so not one results cell is filled. Every result column in
> [`docs/results/TABLES.md`](docs/results/TABLES.md) and
> [`docs/results/COST.md`](docs/results/COST.md) renders `NOT-RUN` or
> `NOT-APPLICABLE` — by measurement, not by omission. Read
> [Status](#status-no-results-are-included) before you read anything else, and read
> [Known divergences](#known-divergences-proof-code) before you trust a number a
> module prints.

## TL;DR

**Text-Span JEPA** is a self-supervised JEPA for text: contiguous **spans** are
masked out of the input, an encoder embeds what is left, and a predictor is trained
to match **latent** targets produced by a frozen EMA copy of the encoder running on
the *unmasked* sequence — nothing reconstructs tokens. Around that core sits
**GWP (Grassmann Workspace Prediction)**, **12 mechanism modules** in three groups
that shape a learned workspace `Q ∈ St(D, k)` and constrain the statistics of the
representation around it.

## Dataflow

```
 (1) input ids (B, T)
          |
          v
     SpanMaskCollator ................................. src/masks/span.py
          |  contiguous spans -> <mask>; `mask_positions` (B, T) kept
          |  `original_input_ids` kept intact alongside
          |
          +-----------------------------------+-------------------------------+
          |                                                                   |
          v                                                                   v
 (2) encoder  (TRAINABLE)                                  (2') target_encoder (FROZEN)
     h_online  (B, T, D)                                      h_target  (B, T, D)
          |                                                   no_grad; EMA-updated after
          |                                                   every optimizer step,
          v                                                   tau 0.996 -> 0.9999 cosine;
 (3) CGN contextual gate  [routing]                           centered + LayerNorm
          |
          v
 (4) WORKSPACE  Q in St(D, k)  -- learned, D x k, on the Stiefel manifold
     constructed by JAWPModule; k follows a curriculum (k_end = D // 10)
     *a parameter, not a tensor flowing between (2) and (5): the mechanism bundle
     shapes it and 7 of the 12 mechanisms read it back*
          |
          v
 (5) predictor  (TRAINABLE)   <-- visible context + masked-slot queries
     span   : (B, S, D) predictions for the S masked slots
     future : (B, F, D) predictions at offsets [1, 4, 16]
          |
          v
     PCR cascade refinement  [routing]  -- refines span predictions in
          |                                disjoint column blocks of Q
          |                                    |
          |                                    |
 (6) LATENT TARGETS                      LATENT TARGETS
     h_target gathered at the              h_target at the future
     S masked slots                        offsets
          |                                 |
          +----------------------+----------+
                                 |
                                 v
 (7) GWP -- 12 MECHANISM MODULES, one shared workspace Q
     core       jawp
     routing    cgn   swip   pcr   spc
     stability  wsd   cmc    gac   sta   puc   rdc   wsr
                                 |
                                 v
 (8) total loss = sum over every weighted term
     lambda_span * L_span          + lambda_future  * L_future
   + lambda_decoder * L_decoder    + lambda_variance* L_variance
   + lambda_covariance * L_cov     + lambda_sigreg  * L_sigreg
   + sum over the 12 mechanisms of lambda_<mech> * L_<mech>
                                 |
                                 v
     optimizer.step()  ->  retract()
     SVD retraction returns Q to St(D, k), i.e. Q^T Q = I_k exactly
```

Stages (2)–(8) are one function: `TextSpanJEPA.compute_loss_with_targets` in
[`src/models/jepa.py`](src/models/jepa.py). The trainer's gradient-accumulation
loop sits on top of it and is the only caller.

## Reproducing every figure

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
& $PY scripts/results_apparatus.py curves --out-dir docs/results/figures
```

That is the only figure-producing command in the repo. It runs today, on CPU,
with no training — and it produces **zero figures**, because it draws from
`train_log.csv` and no `train_log.csv` exists. It writes no PNG at all and prints
`NOT-RUN` once per config. That is the intended behaviour, not a failure: a figure
with empty axes and a flat zero line reads as a result.

**So the honest answer to "one command to reproduce every figure" is: no such
command exists yet, and every figure in this paper requires a training run that
has not happened.** I am not printing a command shaped like one. What does exist
is the apparatus around the run, which is a pure function of *(configs on disk,
run artifacts on disk)* and therefore cannot go stale:

```powershell
& $PY scripts/results_apparatus.py howto   # the owner's run order, one command per run
& $PY scripts/results_apparatus.py table --out docs/results/TABLES.md
& $PY scripts/results_apparatus.py cost  --out docs/results/COST.md
& $PY scripts/results_apparatus.py verify # the anti-fabrication self-check; run in CI
```

`verify` is the guard that keeps this README honest. It asserts that no unmeasured
cell can render as a digit, a dash, `N/A` or `?`, then tries twelve ways to defeat
that guard and fails if any of them succeeds — because a check never observed to
fail is indistinguishable from a check that cannot fail.

## Cite

```bibtex
@misc{textspanjepa2026,
  title  = {Text-Span JEPA: Latent Predictive Learning for Language Representations},
  author = {Slyatski Ilya},
  year   = {2026},
  note   = {Preprint. No training-run results are included; see the repository README.}
}
```

---

## Status: no results are included

This repository ships **68 configs and zero completed training runs.** Of the 68:
46 are ablation arms; 52 are JEPA arms and 16 are baseline arms spanning 6
baseline objectives (MLM, data2vec, VICReg, BYOL, SimSiam, Barlow Twins).

Why: the development machine is CPU-only (`torch 2.13.0+cpu`, Python 3.10, no
CUDA), it was in continuous use by other work, and training runs belong to the
repository owner, not to the agents working in the tree. Training was forbidden
during development, so nothing here has ever been stepped. This is stated at the
top of the file because a README that reads as though results exist is the single
most damaging thing this repository could contain: a reviewer who discovers it
later will not trust anything else in the file.

**What *is* measured, with nothing run:**

| quantity | how | where |
|---|---|---|
| parameter counts for all 68 configs | `sum(p.numel())` over the model `src.train.create_model` builds from the deep-merged config; built on meta tensors where that works, CPU where `.item()` forces it | [`docs/results/param_counts.json`](docs/results/param_counts.json) |
| the scaling ladder (below) | same | table in this file |
| config resolution, mechanism reachability, the `12`-vs-`16` convention | the test suite | `tests/` |

The parameter-count cache holds 62 of the 68 rows — the 6 baseline arms post-date
it. The cache is content-addressed (`sha256(defaults.yaml + the config)`), so a
`table` run measures the missing six on demand and editing a config invalidates
that row and nothing else.

**What is *not* measured, and cannot be without a run:** every training metric,
every training curve, every validation loss, every wall-clock estimate, and every
ablation comparison. `docs/results/README.md` also records that no run this repo
can currently produce emits a downstream metric at all: `src/eval/probes.py` has a
working `LinearProbe`, but nothing in `src/train.py` calls it, so there is no
representation-quality column to fill even after training. That is a `src/train.py`
change and belongs to that file's owner.

## Known divergences: proof ↔ code

[`proofs/`](proofs/) holds **design documents with audited gaps**, not certified
descriptions of the running code. That is this repository's own standing rule, and
it is why the audit matrix exists at all. As of the 2026-08-24 audit, **all twelve
modules have divergences** — none is Verified-only.

The full per-mechanism audit, with `file:line` citations and the measurement behind
each claim, is **[`docs/THEORY_AND_PROOFS.md`](docs/THEORY_AND_PROOFS.md)**. It is
an audit, not an endorsement: it names a divergence, it does not repair one,
because whether the *code* or the *proof* is the intended science is the owner's
decision and not an agent's. Three findings a reviewer is most likely to hit
unaided:

1. **`jawp_predictive_relevance` is mis-scaled, and inverted.**
   `ws_pred_error` is a mean over `N·k` coordinates and `bg_pred_error` a mean over
   `N·D` (`src/models/jawp.py:462-463`), so their ratio carries a `D/k` factor
   before the `max(0, 1 − ratio)` at `:503`. Measured on this tree at `D=64, k=6`:
   the ratio equals `(D/k)·w` exactly, where `w` is the workspace's share of the
   residual energy. At `w = 1.0` — residual **entirely inside** `span(Q)`,
   workspace prediction as bad as it can be — the metric reads `0.0000`. At a
   zero residual it takes the `else` branch and reads `1.0000`. The diagnostic
   cannot distinguish "the workspace collapsed" from "the workspace has nothing
   left to predict". Two proof documents (`proofs/jawp.md`, `proofs/wsd.md` — three
   citation sites between them) and the `jawp.py:70-88` header all cite
   `predictive_relevance → 0` as evidence that the workspace stopped being
   predictable; that reading is not supported by this number. Audit tag
   **D-JAWP-1**, open.

2. **WSD and JAWP pull the same parameter toward opposite optima.**
   JAWP minimises `tr(Qᵀ Σ_res Q)`, whose minimiser on `St(D,k)` is the **bottom-k**
   eigenvectors of the residual covariance — and `jawp.py:45-65` names
   high-variance selection as the failure mode to avoid. WSD maximises
   `‖Q_JAWPᵀ Q_target‖²_F` over the **top-k** eigenvectors of an EMA of
   `Cov(h_target)` (`src/models/wsd.py:200-206`), and `jepa.py` hands it a *live*
   view of the same `workspace_Q`, so both gradients land in one backward pass.
   Measured in `jawp.py:70-88`: the gradient cosine is negative on 12/12
   consecutive steps, with crossover at `lambda_wsd ≈ 0.42`. This is an **open
   science decision, not a defect** — the two optimise different quantities at
   different cadences — and the five resolution options live in
   `proofs/wsd.md`. It is recorded here because the proof documents still contain
   text asserting the antagonism is dormant, which is false: 56 of the 68 configs
   resolve both `use_wsd: true` and `use_jawp: true`.

3. **PUC's documented objective is not the one that runs.**
   `proofs/puc.md` states one form, derives a second, and the code executes a third
   — `η·warmup·relu(−Σᵢ log λᵢ + 10·n_components)/n_components`
   (`src/models/puc.py:296-301`), gated on an entropy deficit that is computed
   only to be used as a switch, with a `10·n` threshold constant that has no
   derivation anywhere. The minimax-optimality theorem is therefore not what the
   module implements, and the risk constraint `E‖z − z_target‖² ≤ R` is never
   formed: `z_target` is accepted at `:223` and never read. On the *gradient* half,
   the 2026-08-24 audit's "carries no autograd edge" finding was **retired** — the
   `use_differentiable_entropy` default flipped to `True` so the loss reads this
   batch's covariance — but there is deliberately no config key to select the
   inert path, so `puc_carries_grad` is the only runtime tell for which branch ran.
   Audit tags **D-PUC-1** through **D-PUC-4**.

A fourth pattern worth knowing about before reading any diagnostic: **a number's
stated meaning is not what the module computes, and the disagreement is invisible
in the number itself.** Three of the worst instances were **removed from the code**
rather than documented: `wsd_steady_state_error` was algebraically identical to
`wsd_running_drift` (the `T` cancels in `ν/λ` with `ν = rd/T`, `λ = 1/T`),
`wsr_spectral_sharpness` was byte-identical to `wsr_sharpness`, and
`rdc_transient_bound` underflowed to `0.0` past ~step 500 while the drift it
claimed to bound still read `3.687`. Worse than reading zero, `rdc_theoretical_bound`
was `min(transient, stationary)` — so `min` returned the underflow and the reported
bound became `0.0` on a parameter measuring thousands. Two additive contributions
were being reduced with `min`; it is now their sum, and `rdc_theoretic_bound`
tracks the drift (`0.129 → 0.348` across steps 1 → 10000). `THEORY_AND_PROOFS.md`
lists the full set, each with the script that establishes it.

**Counting convention.** Quote **12** when counting modules and **16** when
counting numbered capabilities, and never either unqualified. Twelve is
`MechanismBundle.ALL_MECHANISMS` — what is constructed, toggled and ablated.
Sixteen is the GWP header's numbering, and the four items beyond the twelve are
*methods of `JAWPModule`* (WIP, Spectral Gap, Grassmann Optimization, Predictive
Rank), not modules. The convention, its justification and the 16→12 mapping table
are in [`proofs/README.md`](proofs/README.md), recorded as decision D-013 in
[`docs/decisions.md`](docs/decisions.md), and pinned by two tests that name the
sites to update if either number moves.

## The 12 GWP modules

Three groups. Each addresses a specific failure mode of standard JEPA.

**Core — workspace construction**

- `jawp` — Jacobian-aligned workspace prediction (Courant–Fischer optimality)

**Routing — information flow**

- `cgn` — contextual gating network (partition of unity)
- `swip` — selective whitening with information preservation
- `pcr` — predictive cascade refinement (cascade capacity)
- `spc` — spectral predictive coding (information-proportional allocation)

**Stability — workspace integrity**

- `wsd` — workspace-target synchronisation drift (drift bound)
- `cmc` — cross-mask consistency (Cauchy–Schwarz stability)
- `gac` — gradient-allocated capacity (no dead zones)
- `sta` — spectral transport alignment (Davis–Kahan + Wasserstein-1)
- `puc` — prediction uncertainty calibration (minimax)
- `rdc` — representation drift compensation (drift bound)
- `wsr` — workspace sharpness regularisation (generalisation)

The four additional **numbered capabilities** are methods of
[`src/models/jawp.py`](src/models/jawp.py), not modules of their own:

- `#2` WIP — `workspace_information_preservation`. No `use_wip` key exists anywhere;
  reachable only through `MechanismBundle.compute_capacity_bound`, a composite
  diagnostic. It is the only one of the four with its own proof
  ([`proofs/wip.md`](proofs/wip.md)), and that proof is marked **UNAUDITED** in the
  matrix — treat every claim in it as unverified.
- `#3` Spectral Gap — `detect_workspace_dimension`, selects the active rank.
- `#4` Grassmann Optimisation — `grassmann_retract` / `principal_angles` /
  `subspace_distance`. Holds the manifold constraint and reports diagnostics.
- `#5` Predictive Rank — `predictive_rank_loss`. **It is** a trained loss term
  (`lambda_predictive_rank`, default `0.0`) but has no `ALL_MECHANISMS` entry, so
  it is invisible to `active_mechanisms()` and `GWP.summary()`, which report
  `Core: ['jawp']`. Known visibility gap.

Proof-verification status: of the thirteen proof documents, six have a property
test that pins what the document claims, six have none, and `wip.md` is unaudited.
The per-claim table is in [`proofs/README.md`](proofs/README.md).

## Setup

```powershell
pip install -r requirements.txt
pip install -e ".[dev,eval]"   # dev: pytest/ruff/black · eval: scikit-learn/matplotlib
```

Python ≥ 3.9, PyTorch ≥ 2.3. On this machine `python` on `PATH` is a broken shim —
use the working interpreter explicitly:

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
& $PY -c "import src, sklearn, scipy"
```

## Training

```powershell
# JEPA on WikiText-103 — see "Parameter counts" for what the 640/10 rung really costs
& $PY -m src.train --fname config/scaling/small_100m.yaml

# an ablation arm: each toggles one mechanism against defaults.yaml
& $PY -m src.train --fname config/ablations/swip_on.yaml

# baseline objectives — SAME ENCODER CLASS, NOT capacity-matched.
# at the 640/10 rung: encoder 81,758,720 in all three arms (matched exactly);
# MLM total 113,953,280 vs JEPA total 170,706,561 / trainable 88,947,841.
# do not read the three arms' parameter counts as a capacity comparison.
& $PY -m src.train --fname config/wikitext/mlm_wikitext_small.yaml
& $PY -m src.train --fname config/wikitext/data2vec_wikitext_train.yaml
```

No run has been executed. The order is `scripts/results_apparatus.py howto`, which
emits one command per run and prints `NOT-APPLICABLE` for the configs that are not
runnable as shipped (`config/scaling/devices/*` needs N GPUs to mean anything;
`config/tinystories/*` cannot run because `src/train.py` hard-codes
`load_wikitext103`).

Configs live in `config/`, and every one of them is **deep-merged over
[`defaults.yaml`](defaults.yaml)**, so an ablation yaml contains only its delta:

- `scaling/` — `xsmall_30m`, `small_100m`, `base_140m`, `large_300m`, plus `devices/`
- `wikitext/`, `tinystories/`, `kaggle/` — dataset and variant
- `ablations/` — the ablation grid (46 arms)

CLI: `--fname`, `--output_dir`, `--no_defaults`. Resume is `meta.load_checkpoint:
true`, which loads `<logging.folder>/checkpoint-latest.pth.tar`.

## Parameter counts

Three numbers exist per model and they are not interchangeable:

- **total** — every parameter tensor, including the frozen `target_encoder` (an
  exact `deepcopy` of the encoder, so roughly half the model) and both embedding
  tables. This is the model's size, what occupies memory, and what lands in the
  checkpoint. `TextSpanJEPA.get_num_params()` returns it.
- **trainable** — what receives gradients, i.e. total minus the frozen target
  encoder. `TextSpanJEPA.get_num_params_trainable()` returns it, and the trainer
  logs it on the line after the total.
- **non-embedding** — total minus the token and position embedding tables of
  *both* encoders. `get_num_params(non_embedding=True)`. A published convention,
  but **not** the model's size; do not quote it as one.

The scaling filenames track **trainable** parameters, not total. Measured with
each config exactly as it resolves (GPT-2 vocab 50304, `max_seq_len` 512):

| config | filename claims | total | trainable | total / claim |
|---|---|---|---|---|
| `xsmall_30m` (d384 L6 h6) | 30 M | 62,509,249 | 32,348,353 | 2.08× |
| `small_100m` (d640 L10 h10) | 100 M | 170,706,561 | 88,947,841 | 1.71× |
| `base_140m` (d768 L12 h12) | 140 M | 262,021,633 | 137,938,945 | 1.87× |
| `large_300m` (d1024 L16 h16) | 300 M | 537,990,145 | 284,412,929 | 1.79× |
| `ablations/all_core.yaml` (d768 L12 h12) | — | 263,135,220 | 139,052,532 | — |

So a rung's total is 1.7–2.1× its name, and **budget memory against `total`**:
`large_300m` is 538 M parameters ≈ 2.00 GiB of fp32 weights plus **2.12 GiB** of
AdamW m/v state. The optimizer is built from `requires_grad` parameters only, so
the moment buffers cover the 284.4 M *trainable* parameters, two fp32 states each.

## Operational notes

- `logging.keep_last_epoch_ckpts: <K>` prunes older `checkpoint-ep{N}` files every
  epoch (default: keep everything).
- Checkpoint loading tries `weights_only=True` first and falls back with a warning
  for legacy pickled files.
- The trainer warns about config keys absent from `defaults.yaml` — this catches
  real typos like `lamda_swip`, and the warning is never to be silenced. `_meta.*`
  subtrees are exempt.
- `retract()` must be called after `optimizer.step()`; it is what holds `Q` on
  `St(D, k)`, and several modules' properties are conditional on it. Do not remove
  it.
- Validation loss is written to stdout and into `best.pt`'s `extra.best_val_loss` —
  it is **not** in any CSV, so the results apparatus marks that column `NOT-RUN`
  with the extraction command rather than back-filling it from the training loss,
  which is a different quantity.

## Repository layout

| path | what |
|---|---|
| `src/models/` | encoder, predictor, decoder, and one module per GWP mechanism; `mechanisms.py` wires the bundle |
| `src/masks/span.py` | the span-mask collator and its curriculum |
| `src/interp/` | the interpretability modules (probe generalisation, representation geometry, workspace validation, …) |
| `src/eval/probes.py` | `LinearProbe` — **not wired into training**, so no run emits a downstream metric |
| `baselines/` | the six baseline objectives |
| `config/` | 68 shipped configs, deep-merged over `defaults.yaml` |
| `proofs/` | 13 design documents + the audited `IMPLEMENTATION_STATUS.md` matrix |
| `docs/THEORY_AND_PROOFS.md` | the per-mechanism proof-vs-code audit |
| `docs/results/` | the results apparatus; generated by one script, never hand-edited |
| `scripts/results_apparatus.py` | builds every table and figure from what is on disk, and cannot emit an unmeasured number |

## Tests

```powershell
& $PY tools\rt.py tests\test_config_system.py --slow   # one file, via the runner
& $PY -m pytest -q                                     # full suite (prefer CI)
& $PY -m ruff check .
& $PY -m black --check .
```

Tests are deterministic (`src/utils/seed.py`), CPU-only, and never touch the
network. One file per mechanism. `tests/test_sterility.py` guards the sterility
property — if you make a mechanism stronger, that is the file that must keep
passing. Full-suite counts are taken from CI, not from a local wall clock; see
decisions D-001/D-002 in [`docs/decisions.md`](docs/decisions.md).

## License

Apache 2.0. Single author: Slyatski Ilya.