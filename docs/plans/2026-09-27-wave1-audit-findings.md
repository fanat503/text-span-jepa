# Wave-1 audit findings — verified

Date: 2026-09-27
Baseline: **686 tests pass** in ~108s, torch 2.13.0+cpu, Python 3.10, no CUDA.

Status legend: `VERIFIED` = I ran it and saw the output. `REPORTED` = an auditor
claimed it with file:line evidence but I have not re-run it yet.

---

## V1 — 18 of 57 configs cannot run at all  · severity: critical · VERIFIED

`TextSpanJEPAConfig.validate()` requires `ema_tau_end < 1.0`. Eighteen shipped
configs set `ema_tau_end: 1.0` and therefore raise on construction:

```
ValueError: ema_tau_start=0.996 must be in (0, ema_tau_end) and
            ema_tau_end=1.0 must be in (ema_tau_start, 1.0)
```

Blocked, 18 files:

- `config/scaling/xsmall_30m.yaml`, `small_100m.yaml`, `base_140m.yaml`,
  `large_300m.yaml` — **the entire scaling ladder**
- `config/wikitext/textspanjepa_wikitext_{small,base}.yaml`
- `config/kaggle/textspanjepa_kaggle.yaml`
- `config/tinystories/textspanjepa_tinystories.yaml`
- 10 ablations: `no_jawp`, `cgn_on`, `sigreg_only`, `no_decoder_loss`,
  `no_future_loss`, `predictive_rank_on`, `jawp_alpha_0`, `jawp_high_alpha`,
  `jawp_k_fixed`, `jawp_random_init`

Consequence: the four scaling configs named in the README cannot be run to
reproduce the scaling claim, and the `verify-gate` skill's own smoke-train
command is currently broken. 39 of 57 configs are fine.

Open question for a human: is `ema_tau_end: 1.0` a deliberate "freeze the
target encoder" choice that `validate()` wrongly rejects, or is `1.0` a typo for
something like `0.9999`? Do **not** guess — the two fixes have different
scientific meanings.

---

## V2 — `wsd_on.yaml` is a dead ablation  · severity: critical · VERIFIED

`config/ablations/wsd_on.yaml` sets `use_wsd: true` **and** `use_jawp: false`.
`jepa.py:483` guards construction on `if config.use_wsd and config.use_jawp:`,
so `self.wsd` is `None` and the WSD loss returns zero. The run trains **with
WSD off while being reported as the WSD-on arm**.

Verified active sets:

| config | `use_wsd` | `use_jawp` | WSD |
|---|---|---|---|
| `wsd_on.yaml` | true | **false** | **DEAD** |
| `all_core.yaml` | true | true | ALIVE |

---

## V3 — the leave-one-out ablation grid does not exist  · severity: high · VERIFIED

`defaults.yaml` has `use_jawp: true` and every other `use_*` false. Nine of the
twelve `no_<mech>.yaml` files additionally set `use_jawp: false`, so after the
deep merge their active mechanism set is **empty** — identical to `none.yaml`:

```
no_cmn  n=0 []      no_spc  n=0 []      no_wsd  n=0 []
no_gac  n=0 []      no_sta  n=0 []      no_cgn  n=1 ['jawp']
no_jawp n=0 []      no_puc  n=0 []      no_wsr  n=1 ['jawp']
no_pcr  n=0 []      no_rdc  n=0 []
no_swip n=0 []
```

So `no_spc`, `no_sta`, `no_cgn`-and-others are **two-variable deltas**, not
one-mechanism ablations, and any conclusion drawn from them is confounded.
There is also **no `n=11` config anywhere** — the true leave-one-out row of the
ablation table is entirely missing.

This is the highest-value finding for the paper: the ablation table cannot be
built from the current config set.

---

## V4 — the ablation table spans two different models  · severity: critical · REPORTED

Two families, not one grid:

- 29 files → base-768 (depth 12, 50 epochs, bs 64, ga 1)
- 10 files → small-640 (depth 10, 30 epochs, bs 128, ga 4) — `cgn_on`,
  `jawp_*`, `no_jawp`, `no_decoder_loss`, `no_future_loss`,
  `predictive_rank_on`, `sigreg_only`

Those 10 restate `embed_dim / encoder_depth / epochs / batch_size /
grad_accum_steps` in the ablation file itself. Any cross-mechanism comparison
drawn from the table compares different architectures. Needs re-verification,
but it is consistent with V5.

---

## V5 — the scaling ladder confounds size with everything else  · severity: critical · REPORTED

| config | depth | width | heads | bs | ga | eff bs | epochs | lr | refine | drop_path |
|---|---|---|---|---|---|---|---|---|---|---|
| xsmall_30m | 6 | 384 | 6 | 256 | 2 | **512** | 30 | 1e-3 | 2 | 0.05 |
| small_100m | 10 | 640 | 10 | 128 | 4 | **512** | 30 | 1e-3 | 3 | 0.10 |
| base_140m | 12 | 768 | 12 | 64 | 4 | **256** | 50 | 1e-3 | 3 | 0.10 |
| large_300m | 16 | 1024 | 16 | 32 | 8 | **256** | 50 | 5e-4 | 3 | 0.15 |

Depth/width/heads vary monotonically (good), but effective batch size, epochs,
learning rate, `num_refine_steps`, and `drop_path` all move with model size. A
30M-vs-300M result cannot be attributed to scale.

---

## V6 — the typo detector matches leaf names, not paths  · severity: high · REPORTED

`src/train.py:849,874` builds a set of bare leaf key names
(`{p.split(".")[-1] for p in _leaves(known)}`) and tests membership against
`k`, so a key in the wrong subtree — or an entire misspelled section — is
accepted silently. Reported non-detections: `model.batch_size`,
`optimisation.lr`, `modle.embed_dim`. A genuine nested typo
(`model.lamda_swip`) *is* caught.

The warning currently gives false assurance, which matters because the README
advertises it as a typo guard.

---

## V7 — "16 mechanisms" is a prose count; the code has 12  · severity: medium · REPORTED

| location | claim | actual |
|---|---|---|
| `mechanisms.py` header, `GWP` docstring, `README.md:80` | 16 | prose only |
| `MechanismBundle.ALL_MECHANISMS` | — | **12** |
| `GWP.N_MECHANISMS` | 16 | a dead literal, read by no code |
| `GWP.N_GROUPS` | 3 | 3, correct |
| `proofs/IMPLEMENTATION_STATUS.md:4` header | 13 | 12 table rows |
| `proofs/README.md:3,7` | 13 | 11 rows (RDC, WSR missing) |
| `tests/test_model.py:2262` | name says 16 | asserts `== 12` |

"16" is derived by counting five items under "Mechanism 1-5" (JAWP, WIP,
Spectral Gap, Grassmann, Predictive Rank), but **four of those five are methods
on `JAWPModule`, not modules** — and Predictive Rank has no `ALL_MECHANISMS`
entry, so it is invisible to `active_mechanisms()`, `dependency_dag()`, and
`summary()`. `proofs/wip.md` exists with no matrix row.

Verified clean: `__all__` is 15/15 defined (`import *` is safe); `GROUPS` and
`ALL_MECHANISMS` have zero orphans either direction; `retract()` covers all
three manifold-bearing parameters.

---

## V8 — `MechanismBundle` is not on the production path  · severity: high · REPORTED

`jepa.py:425-563` re-implements all 12 mechanism constructions inline and adds
`sta_warmup_steps` / `sta_update_interval`, which the bundle does not accept.
`MechanismBundle.from_config` is called only from `tests/test_model.py` and
`tests/test_rdc.py` — never from `train.py` or `jepa.py`.

Consequences:
- Two hand-synced code paths; every fix must land twice.
- The advertised 3-line GWP recipe differs behaviourally from the trainer:
  `forward()` passes `self.jawp.workspace_Q.data[...]` to SWIP/WSD/WSR
  (detached), while `jepa.py` passes the live tensor. In the bundle path those
  three mechanisms contribute **zero gradient** to `workspace_Q`.
- `from_config` honours no `lambda_*` weight, so bundle losses are unweighted
  where the trainer gates every one of them.
- `GAC` and `RDC` have no bundle accessor at all, and
  `proofs/rdc.md:196` claims a `bundle.forward()` RDC path that does not exist.
  `proofs/rdc.md:189` gives an import path (`from src.models.rdc import
  rdc_compensate`) that raises `ImportError` — the function lives in
  `mechanisms.py`.

---

## V9 — silent resume can overwrite a good checkpoint  · severity: medium · REPORTED

`load_checkpoint: true` with a missing file falls through with no warning
(`train.py:805` has no `else`), and `load_checkpoint()` swallows every
exception, returning zeros with only a `logger.warning` (`train.py:419-421`).
Either path trains from step 0 and then overwrites
`checkpoint-latest.pth.tar`. A corrupt or unreadable checkpoint is therefore
destroyed rather than preserved.

---

## V10 — dead config keys  · severity: medium · REPORTED

- `defaults.yaml:180` `optimization.ema: [0.996, 0.9999]` — never read; the
  trainer uses `model.ema_tau_start/end`. Editing it changes nothing.
- `defaults.yaml:194` `logging.write_tag` — set by 39 ablations + 4 scaling
  configs, read by nothing.
- `defaults.yaml:6` declares top-level `seed:` but 28 configs use `meta.seed`,
  which survives only because `train.py:886` reads it explicitly. The typo
  detector cannot see it.

---

# Part 2 — Reproducibility audit (reproduced with experiments)

Everything below was measured by running code, not inferred from reading.
Baseline: torch 2.13.0+cpu, Python 3.10.5, no CUDA, 6 threads, spawn start method.

## R1 — resumed runs diverge from continuous runs  · severity: critical · VERIFIED

**Good news first:** a *fresh* run with the same seed is **bitwise**
reproducible, with and without dropout/drop-path:

```
EXP1 no-dropout  r1=1.2265701293945312  r2=1.2265701293945312  bitwise_equal=True
EXP2 dropout-on  r1=1.1670680046081543  r2=1.1670680046081543  bitwise_equal=True
```

**But resume breaks it.** Same seed, same data, 6 optimizer steps, cut after
step 3, save, re-seed, rebuild, load, continue:

```
continuous : 1.6646  1.6917  1.3765  2.2459  2.3339  1.1452
resumed    :                            2.1155  2.1080  1.2218
  step 3: delta=1.305e-01   identical=False
  step 4: delta=2.260e-01   identical=False
  step 5: delta=7.668e-02   identical=False
```

5.8% relative loss divergence on the **first** post-resume step.
`tests/test_training_e2e.py::test_resume_continues_global_step` only asserts the
step *counter* continues, so the suite is blind to this.

## R2 — RNG state is never checkpointed  · severity: critical · VERIFIED

`src/train.py:99-190` saves no RNG state; `:193-421` restores none. Verified:
`RNG-STATE keys in checkpoint: []`.

Consequence: mask sampling (`masks/span.py:60-61`), DropPath
(`encoder.py:27`), CGN Gumbel (`cgn.py:218`), CMC masks (`cmc.py:251-252`),
and data shuffle all replay from a *restarted* stream. This is the direct cause
of R1.

Note `cmc.py:251` accepts an `rng` parameter that `train.py:1073` passes as
`None` — the dedicated generator is dead code.

## R3 — 15 tensors silently revert, 6 of them trainable  · severity: critical · VERIFIED

Full `named_buffers()` / `named_parameters()` diff across a save→load round trip:

```
BUFFERS that differ:
  puc.running_entropy              puc.running_overconfidence
  rdc.running_ortho_drift_norm     rdc.running_workspace_drift_norm
  sta.current_eigenvalues          sta.ref_cov
  sta.ref_eigenvalues              wsd.target_Q
  wsd.target_cov
PARAMETERS that differ:
  cgn.context_proj.weight / .bias            max|d| = 3.0e-03
  pcr.refine_blocks.{0,1}.net.0.weight/bias  max|d| = 1.1e-05
opt state mismatches: 0  (58 params)
```

The optimizer round-trips perfectly; the **model** does not. CGN gate modulation
and PCR cascade refinement revert to random init **while the rest of the model
resumes**.

Worst of these: `wsd.target_cov`, `wsd.target_Q`, `sta.ref_cov`,
`sta.ref_eigenvalues` are **loss inputs**, yet `sta.is_initialized` *is* restored
to `True`. After resume `sta.ref_cov` restarts near `0.01·I` while claiming to
be initialized, so the EMA decays from near-zero and STA/WSD losses are wrong
for hundreds of steps. No warning.

**The fix already exists and is dead code:** `src/models/rdc.py:249` and
`src/models/puc.py:268` both implement a complete
`checkpoint_dict()` / `load_checkpoint()` pair. `src/train.py` uses neither and
hand-rolls an incomplete key list. Checkpoint top level is 48 keys.

## R4 — validation pollutes training state  · severity: high · VERIFIED

`_validate` (`src/train.py:1408-1444`) runs `compute_loss` under `model.eval()` +
`no_grad`, yet **24 buffers are still mutated**. Measured deltas:
`sta.ref_cov` Δ1.56, `wsd.target_cov` Δ1.59, `wsd.target_Q` Δ1.31,
`sta.is_initialized` flipped, `target_centering.center` Δ0.071, plus all
PUC/RDC running stats.

Only SPC (`spc.py:373`) and CGN (`cgn.py:216,246`) guard on `self.training`.
Missing in `wsd.py:236-240`, `sta.py:252,270-276`, `rdc.py:194-211`,
`puc.py:151-188`, `gac.py:181-200`.

This compounds R2: the mask RNG that validation consumed shifts the training
stream. And `train.py:700-704` swallows a validation-load failure into a
`logger.warning`, setting `val_dataloader=None` — so two machines with the
same seed produce **different models**.

## R5 — dataloader workers cannot start on Windows  · severity: high · VERIFIED

`src/train.py:698,710` passes `worker_init_fn=lambda wid: worker_init_fn(wid, seed)`.
Verified: `pickle.dumps(lambda)` → `PicklingError`. Host start method is
`spawn`.

`python -m src.train` with the shipped default `data.num_workers: 2` — and
`xsmall_30m.yaml`'s `4` — **cannot start workers on this machine**. The suite is
green only because the single e2e test uses `num_workers: 0`
(`test_training_e2e.py:53`).

## R6 — the unpickler security claim is false  · severity: high · VERIFIED

`src/utils/torchio.py:29-35` docstring: *"non-pickle garbage propagates
untouched: an attacker can no longer craft an exception that flips the strict
unpickler off."*

**False.** A plain text file and a truncated file both raise
`pickle.UnpicklingError` — which is exactly the retry trigger. The fallback is
`weights_only=False`, i.e. arbitrary code execution. Verified with a benign
reducer: `fallback EXECUTED the pickled reducer: True`.

The `warnings.warn` does fire, but `UserWarning` is silenced by `-W ignore` or
any `warnings.filterwarnings` entry.

## R7 — thread count changes results  · severity: high · VERIFIED

1 vs 8 threads, same seed, same data, 10 steps: steps 0-2 identical, **step 3
onwards differs** (`d=1.19e-07` … `3.58e-07`); final `qkv` weight not bitwise
equal.

Errors are tiny per step, but this is a chaotic system and they compound. A run
on a 4-core laptop is not bitwise reproducible on a 16-core server. Nothing in
the repo or environment pins thread count — no `torch.set_num_threads`, no
`OMP_NUM_THREADS`, no `torch.use_deterministic_algorithms`.

## R8 — DDP is entirely non-functional  · severity: medium · VERIFIED

`scripts/wikitext/train_ddp.sh` runs `torchrun --nproc_per_node=N -m src.train`,
but `src/` contains **zero** DDP machinery — no `DistributedSampler`,
`init_process_group`, `DistributedDataParallel`, or even the string `ddp`.

N ranks therefore run identical independent loops with the same seed and the
same `logging.folder`, all writing the **same checkpoint filenames**. That
corrupts checkpoints. The multi-GPU path cannot work.

## R9 — `wsr_mode: sam` is a silent no-op  · severity: medium · VERIFIED

`wsr.py:412-417` reads `Q.grad` during forward, but `train.py:1226` calls
`optimizer.zero_grad()` (default `set_to_none=True`) after every step, so
`Q.grad is None` and the orthonormality proxy is always substituted. The
documented loss is never the one computed.

## R10 — plain tensor attributes bypass the state dict  · severity: medium · VERIFIED

Assigned in `forward`/training, not `register_buffer`, so they miss `.to(device)`
and are absent from `state_dict` (verified `False` for `model._prev_target_h`):

| location | attribute | consequence |
|---|---|---|
| `src/models/jepa.py:895` | `_prev_target_h` | `(B,T,D)` held between steps, wrong device after `.to()` |
| `src/models/wsr.py:287` | `_lagged_gradient` | on resume WSR falls back to the proxy for one step |
| `src/models/jspace.py:101` | `_prev_jspace_vectors` | `jspace_stability` silently resets on resume |
| `src/models/jepa.py:566,567,642,650` | `_gac_z`, `_cmc_pass` | hold **live autograd graphs** between forwards |

## R11 — 22 of 22 `src/interp/` modules are effectively unseeded  · severity: medium · VERIFIED

`src/interp/` has exactly **one** seeding call (`ground_truth.py:74`) — and that
one calls `torch.manual_seed` on the **global** generator, destroying the
caller's stream.

Unseeded draw sites across 15 modules, including inside a *training* loop
(`sae.py:150,162,167`) and inside a documented entry point
(`run_comparison.py:421`, `torch.randint(0, 50304, (500,128))`).

Also hardcoded seeds not derived from the run seed, so an ablation cannot vary
them: `workspace_validation.py:291,372`, `feature_composition.py:152`,
`statistical_tests.py:50,118,212,392`, `src/utils/seed.py:42`.

## R12 — no `conftest.py` enforces determinism in tests  · severity: medium · VERIFIED

12 of 21 test files contain **zero** seeding calls. No autouse fixture exists.
The suite is green today, but any future exact-value assertion becomes
order-dependent. `AGENTS.md` asserts "tests must be deterministic" with nothing
enforcing it.

## R13 — one more unrunnable config  · severity: medium · VERIFIED

`config/wikitext/data2vec_wikitext_small.yaml`: `predictor_embed_dim=384` is not
divisible by `num_heads=10`. On top of the 18 `ema_tau_end: 1.0` configs from
Part 1.

## Scheduler replay caveat  · severity: medium · REPORTED

LR / WD / EMA-tau scheduler *state* is not checkpointed; it is replayed by
`for _ in range(global_step): scheduler.step()` (`train.py:740-753`). Exact only
if `epochs` is unchanged on resume — and the e2e test resumes 2→3 epochs, which
re-derives `T_max` and reshapes the entire LR/WD/EMA trajectory. Also
O(global_step) wasted work per resume.

## Verified clean (reproducibility)

- Fresh-run same-seed determinism is **bitwise**, dropout included.
- AdamW optimizer state round-trips exactly — 58/58 params, 0 mismatches.
- `keep_last_epoch_ckpts` can never delete `checkpoint-latest.pth.tar`
  (`re.fullmatch(r"checkpoint-ep(\d+)\.pth\.tar")`); confirmed end-to-end by
  `test_two_epochs_train_save_prune`.
- The CPU AMP path is an explicit no-op, not a silent skip: `GradScaler` is
  hard-disabled at `train.py:733` and `use_bfloat16` requires
  `device.type == "cuda"`.
- `num_workers` does not change batch order — the `RandomSampler` runs in the
  main process, so R11's worker-seeding gap is latent, not active.
- **No test requires network or a GPU.** The only `cuda` hits in `tests/` are a
  docstring and a `SimpleNamespace` fake.
- No `set_default_dtype` / `set_default_device` / `set_grad_enabled` anywhere.

## Top three fixes, in order

1. **Checkpoint completeness.** Persist and restore RNG state plus the 15
   missing tensors — or, better, replace the hand-rolled key list with
   `model.state_dict()` + `scheduler.state_dict()` and let
   `rdc.checkpoint_dict()` / `puc.checkpoint_dict()` (already written) be used.
2. **`load_checkpoint` must raise**, not return `(0,0,0,0,None)`. A corrupt or
   wrong-arch checkpoint must not be able to overwrite the good one.
3. **Guard mechanism buffer mutation on `self.training`** in
   `_wsd_loss` / `_sta_loss` / `_rdc_loss` / `_puc_loss` / `_gac_loss` so
   validation stops writing training state.

---

## Deferred — not yet measured

Parameter counts were **not** verified. The measurement script failed on
`TextSpanJEPAConfig.__dataclass_fields__` (it is not a dataclass), and the work
was paused to avoid loading the CPU. The claim that `xsmall_30m` is 62.5M total
/ 12.8M non-embedding — i.e. scaling-config names off by 1.7-2.1× — remains
**unverified**.

The interpretability and performance/scaling audits are still outstanding; the
performance auditor was killed mid-probe when its FLOPs/parameter measurement
was consuming ~27% of the machine and 4.7 GB RAM.

