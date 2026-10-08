# Part 4 — Performance & scaling audit (reproduced with measurements)

Repo left unmodified. FLOPs via `torch.utils.flop_counter.FlopCounterMode`;
parameter counts by direct instantiation matching `src/train.py:create_model`.

---

> ## ⚠ PARTIALLY SUPERSEDED — read this before quoting anything below
>
> **A dated snapshot of 2026-09-27, kept as written.** Findings are not
> rewritten after fixes land. Reconciliation added 2026-10-08 by TASK-41.
>
> ### The number this file is most often quoted for, and what replaced it
>
> **P1's `2.7×` is wrong as a description of the current tree, and the card's
> replacement figure (`1.87×`) is right.** Two different quantities were confused:
>
> - The line that said `get_num_params()` "is **2.7× smaller than the checkpoint
>   it saves" described a `get_num_params()` that subtracted the embeddings *and
>   omitted the target encoder*. **That function no longer exists in that form.**
>   It now returns `sum(p.numel() for p in self.parameters())` — the whole model,
>   including the frozen `target_encoder` — which is exactly what the trainer
>   logs, and `docs/results/param_counts.json` measures that way for all 62
>   configs. So the "2.7× under-report" is **FIXED**, and the column of
>   `get_num_params()` values in the P1 table (12.82 M / 56.38 M / 98.85 M /
>   232.27 M) is **void**: the correct values are the `total` column beside them.
> - The **surviving** ratio is `total ÷ filename claim`, which is **1.87×** for
>   `base_140m` and 1.71–2.08× across the ladder — `2.08× / 1.71× / 1.87× /
>   1.79×` for `xsmall_30m / small_100m / base_140m / large_300m`. That is the
>   number the wave-1 findings doc's "Deferred — not yet measured" section was
>   still hedging about, and it is what `README.md` now states. The *filenames*
>   are not wrong: they track **trainable** parameters, within −11%…+8%.
> - A `2.7×` that is **still correct** and must not be "fixed": the width range
>   across the ladder is 384→1024 = 2.67×, which is what
>   `config/scaling/large_300m.yaml:10` refers to when it says holding `lr` fixed
>   across a 2.7× width range. Different quantity, same digits.
>
> ### Per-finding status
>
> | # | status as of 2026-10-08 |
> |---|---|
> | P0 | **FIXED** (`ema_tau_end: 0.9999` declared once in `defaults.yaml`; decision D-006). The `ema_schedule: cosine` half is **still dead code** and is recorded as the open item in D-006 — `defaults.yaml:155` says so in place. |
> | P1 | **SUPERSEDED as above.** Two sub-claims survive: `get_num_params()`'s docstring records the old under-report, and the component split is exactly still right — re-measured 2026-10-08 on `config/scaling/base_140m.yaml`: encoder 124,082,688 + `target_encoder` 124,082,688 + predictor 11,437,057 + decoder 2,360,832 = **262,021,633 total**, leaving 58,368 for the mechanisms and everything else. The "only `use_jawp` enabled" note was right about the four scaling rungs and **is now also true of the other 58 configs** — `defaults.yaml` enables all twelve, so a plain `python -m src.train` trains the full model, and the four rungs opt out explicitly. |
> | P2 | **FIXED — this file's `0.52–0.58` est/actual column is void.** `src/utils/flops.py` was rewritten as a *structural* estimator (TASK-22): its docstring now carries its own `FlopCounterMode` comparison and reports **est/actual 0.99963–0.99980** across T = 128/256/512 at `base_140m` dims, and names the four things the old `6·N·L·B` form could not represent (target encoder, predictor passes, decoder head, per-step diagnostics). The per-step diagnostics are still called unconditionally and still not modelled in the estimate — that residual is documented in the function's own docstring. **"dead code" is also no longer accurate**: it has a measured reproduction target and a documented gap. |
> | P3 | **PARTIALLY FIXED.** `CollapseDiagnostics.compute` is still called on every step from `src/models/jepa.py:920`, but the **19 SVD-bearing calls are now 5** (`_SharedSpectral`, TASK-19). Measured before→after by that card: `compute` 2.1641 s → 1.6562 s, **51.2% → 45.0% of the forward** (at 1 thread; this file's 67% was measured at 4). The gate was analysed and **deliberately not added** — see `.agent-notes/task-19.md` §3 for the three correctness traps. Also: **seven metric values changed by design** in that same change (`intrinsic_dim_{online,target}`, `mean_pairwise_cosine_{online,target}`, `uniformity_{online,target}`, `alignment`) because their 256-row subsample moved to a private generator; any number published for those seven is void, the other 37 are bitwise unchanged. |
> | P4 | **LARGELY FIXED.** The GAC/CMC retained-graph problem is gone and several `.item()` sites were batched; the 92/201 sync counts were measured before any of that and **should not be quoted**. The `_gather_masked` and `SpanMaskCollator` items are still worth re-measuring before repeating the 130 ms / 89 ms figures. |
> | P5 | **FIXED.** `_gac_z` and `_cmc_pass` are released after `backward()`. `gradient_checkpointing` is still `false` by default, and the predictor still has no checkpointing path. |
> | P6 | **STILL TRUE**, now with a written spec: `docs/plans/2026-09-28-distributed-training-spec.md`. The five numbered problems are unchanged. The `find_unused_parameters` note is still correct and now has a name — `gac_wiring` in `src/train.py`. |
> | P7 | **SUPERSEDED — the ladder table is no longer the ladder.** All four rungs now inherit `epochs 50`, `batch_size 64`, `grad_accum_steps 8` (effective **512** for all four), `lr 1e-3`, `num_refine_steps 3`, `drop_path_rate 0.1`, `max_seq_len 512` from `defaults.yaml`, and each declares only `embed_dim / encoder_depth / num_heads / predictor_embed_dim / predictor_depth`. So this finding's "**two points per group**, not four" is **fixed**: it is now one group of four, differing only in width and depth. The curriculum-length confound is also gone (`jawk_curriculum_steps` and `future_warmup_steps` are inherited). **Attribution:** this landed in the squash commit `11bcbd0` (PR #10, "21 cards, 82 commits"); there is no per-card note for it, so this row cites the commit rather than inventing a card number. |
> | "Safe wins" list | Items 1 (partly — the dedup and the private RNG landed; the *gate* did not), 4, 7 and 8 landed. Items 3, 5 and 6 were not taken. The `drop_path`/`lr`/curriculum items from P7 were taken instead. Re-verify each before repeating it. |
> | "Trade-offs" list | Still a decision list, still nobody's decision. `flops.py`'s status is stale — see P2. |
>
> Note also that the doc's own closing premise — that a local full-suite run is
> not affordable — is why `AGENTS.md` takes its test count from CI rather than
> from a wall clock on this machine (decisions D-001/D-002).

---

## P0 — `ema_tau_end: 1.0` is a *semantic* bug, not just a validation error

Part 1 established that `ema_tau_end: 1.0` makes 18 configs raise. This audit
found the deeper problem: even if `validate()` were relaxed, the value is wrong.

`EMATauSchedule.step()` (`src/utils/schedulers.py:85`) returns exactly `tau_end`.
At `tau = 1.0`, `update_target_encoder` becomes
`mul_(1.0).add_(q, alpha=0.0)` — **the target encoder freezes**. In a
self-distillation method the target encoder *is* the training signal; freezing
it is not a rounding difference.

So the correct fix is a semantic decision, not a validation relaxation. Either
the intent was a real EMA endpoint (≈0.9999) or the target encoder was meant to
freeze and the value should be opt-in with loud documentation.

Separately, **`ema_schedule: cosine` is dead code**:
`EMATauSchedule.step_cosine()` exists (`schedulers.py:87`) and is never called;
`train.py:1230` calls the linear I-JEPA form.

## P1 — parameter counts (this closes the open item from Parts 1-3)

Three mutually inconsistent numbers exist per model. `N` excludes the frozen
target encoder; `total` includes it.

| Config | name claims | **total** | **trainable** | `get_num_params()` (what train.py logs) | trainable vs claim |
|---|---|---|---|---|---|
| `xsmall_30m` (d384 L6 h6) | 30 M | **62.51 M** | **32.35 M** | 12.82 M | +7.8 % |
| `small_100m` (d640 L10 h10) | 100 M | **170.71 M** | **88.95 M** | 56.38 M | −11.1 % |
| `base_140m` (d768 L12 h12) | 140 M | **262.02 M** | **137.94 M** | 98.85 M | −1.5 % |
| `large_300m` (d1024 L16 h16) | 300 M | **537.99 M** | **284.41 M** | 232.27 M | −5.2 % |

So the earlier "names off by 1.7-2.1×" claim is **half right**: the names track
*trainable* parameters and are within 1-11%. But `total` — what occupies
memory, what lands in the checkpoint, what the AdamW state is sized against
(2× for m/v) — is **1.8-2.1× the claimed figure**. `large_300m` actually needs
538 M weights + 1.79 GB of AdamW fp32 state.

Component split (base_140m, total): encoder 124.08 M + target_encoder 124.08 M
(an exact `deepcopy`) + predictor 11.44 M + decoder 2.36 M.

**`get_num_params()` (`jepa.py:1069`) subtracts token+pos embeddings and omits
the target encoder**, so the number `src/train.py:917-918` logs as "Model
parameters" is **2.7× smaller than the checkpoint it saves**. Any compute or
memory estimate in a scaling paper built on the logged number is wrong.

All 12 mechanisms enabled add only +0.4-0.5 % of parameters. Note the shipping
configs enable **only `use_jawp`** — the other 11 flags are `false` in
`defaults.yaml` and in all 4 scaling yamls.

## P2 — `flops.py` under-estimates ~2× and the error grows with T

`FlopCounterMode` around `compute_loss_with_targets` and `.backward()`,
base_140m dims, vocab 4096, B=4:

| T | real fwd+bwd | `flops.py` estimate | est/actual |
|---|---|---|---|
| 128 | 5.419e11 | 3.147e11 | **0.58** |
| 256 | 1.117e12 | 6.295e11 | **0.56** |
| 512 | 2.440e12 | 1.259e12 | **0.52** |

Not a constant factor — it degrades with T because `6·N·L·B` has no attention
term (`2·n_layers·L²·d·B`) and no term for the repeated predictor passes.
`flops.py` has no `n_layers`/`embed_dim` in its signature, so it structurally
cannot be correct. It is imported by nothing outside `src/utils/__init__.py`
and one test: **dead code that would mislead the moment a scaling script adopts
it.**

Forward split at B=4/T=512: encoder 39.6 %, **target encoder 39.6 %** (no_grad,
same cost), predictor 20.8 %.

## P3 — diagnostics are 67 % of the forward pass  · high · VERIFIED

`CollapseDiagnostics.compute` (`collapse.py:141`) is called unconditionally at
`jepa.py:872` and performs **13 `torch.linalg.svdvals` + 4 full
`torch.linalg.svd`** on the full `(B·T, D)` activation matrix, **every step**.

Measured: **67 % of the entire forward pass** (1.60 s → 4.88 s, xsmall dims,
B=8/T=256, fp32/4 threads). It computes 56 metrics; `train.py:1260-1267` prints
**4**, and only every `log_freq` steps.

`jspace_metrics.compute` (`jepa.py:880`) repeats the pattern for 1 printed
metric.

## P4 — 92-201 device syncs per forward  · high · VERIFIED

Measured by patching `torch.Tensor.item` and tracing the caller frame.

- **92 `.item()` calls per forward with the shipping config** (67 in `src/models/`)
- **201 per forward with all 12 mechanisms on**

Each is a hard device sync; on GPU they serialize the step.

Redundant slicing quantified: with all 12 on, one forward performs **6 slices of
`jawp.workspace_Q[:, :k_active]` and 4 `active_k.item()` syncs**
(`jepa.py:1008,1038,1059,1066`). `spc.py` alone adds 24 syncs from per-band
`.item()`. `train.py:1189` is a 5th sync of the same tensor.

Python loops where a vectorized op exists:
- `predictor._gather_masked` (`predictor.py:183-187`) — B iterations despite a
  docstring claiming "no boolean indexing". **130 ms/step at B=256.**
- `jepa._slot_indices` (`jepa.py:925-927`) — same pattern, rebuilt each step.
- `SpanMaskCollator` (`span.py:57-96`) — O(B·T/3) Python/numpy loop **in the main
  process** (`train.py:1028` calls it after the DataLoader, so `num_workers: 4`
  does nothing for masking), plus **double collation**: 4 full B×T copies where
  1 suffices. **89 ms/step at B=256.**

## P5 — activation memory ≈ 2× under grad accumulation  · high · VERIFIED

The `info` dicts are clean: measured `loss_dict` (125 keys) and `diag_dict`
(56 keys) contain **0 tensors with `.grad_fn`**.

The leak is on the module. After `backward()` returns:
- `self._gac_z` (`jepa.py:650`) — live autograd graph, `grad_fn=True`
- `self._cmc_pass["slots"]` (`jepa.py:767`) — live autograd graph, annotated
  `# live — secondary pass keeps its graph`

Both survive past `optimizer.zero_grad()` (`train.py:1226`). With
`grad_accum_steps` 2/4/8 (the scaling configs use 2/4/4/8) the previous
micro-batch's graph is still alive while the next is built → **peak activation
memory ≈ 2×, scaling with accumulation depth.**

Also: `gradient_checkpointing` is `false` in `defaults.yaml:32` **and
hard-coded `false` in all 20+ configs**, and the flag reaches only the online
encoder (`jepa.py:380` → `encoder.py:248`). The predictor — 6.3 full-sequence
passes — has no checkpointing path at all.

## P6 — DDP cannot work, and destroys checkpoints  · high · VERIFIED

`train_ddp.sh:10` runs `torchrun --nproc_per_node=4`. `torch.distributed` is
**never imported anywhere in the repo**. Five independent fatal problems:

1. **No process group, no all-reduce.** Each rank is a fully independent replica.
   Effective batch is 4× B with gradients never summed.
2. **All ranks land on `cuda:0`.** `train.py:894` is
   `torch.device("cuda" if cuda.is_available() else "cpu")` — no `local_rank`.
   4 full replicas on GPU 0 → immediate OOM at any shipping batch size.
3. **Checkpoint corruption.** `logging.folder` is identical across ranks, so all
   4 write `params-*.yaml`, `train_log.csv`, `best.pt`,
   `checkpoint-latest.pth.tar` non-atomically (`torch.save`, `train.py:190`).
   **Any checkpoint this script ever produced is unusable.**
4. **Every rank sees the identical batch order and masks.** `seed_everything(42)`
   is unconditional; `make_dataloader` has no `sampler` argument, so a
   `DistributedSampler` cannot be injected without a code change.
5. **Workspace subspaces split across parameter/buffer.** `jawp.workspace_Q` is
   a Parameter (safe under all-reduce), but `rdc.workspace_Q` and
   `wsd.target_Q` are **buffers** mutated in place. With
   `broadcast_buffers=True` the behaviour is "rank 0's value, lagged one step",
   never an average — silent, not a crash.

For whoever adds real DDP: `find_unused_parameters` **cannot be a single
constant**, because the mechanism branches are step-dependent (`pcr.level_gates`
and `spc.freq_basis` are in-graph only on steps where their branch fires, and
`train.py:1127-1146` runs a second separate `backward()` for GAC). `True` is the
only safe setting.

## P7 — the scaling ladder has 2 points per group, not 4  · high · REPORTED

| Config | d | enc L | h | refine | T | B | accum | eff. batch | epochs | lr |
|---|---|---|---|---|---|---|---|---|---|---|
| xsmall_30m | 384 | 6 | 6 | 2 | 512 | 256 | 2 | **512** | 30 | 1e-3 |
| small_100m | 640 | 10 | 10 | 3 | 512 | 128 | 4 | **512** | 30 | 1e-3 |
| base_140m | 768 | 12 | 12 | 3 | 512 | 64 | 4 | **256** | 50 | 1e-3 |
| large_300m | 1024 | 16 | 16 | 3 | 512 | 32 | 8 | **256** | 50 | 5e-4 |

Good: width/depth differ monotonically; `max_seq_len` is held constant at 512.

Bad: `B × accum × T` = 262144 / 262144 / 131072 / 131072 — the ladder splits into
**two tiers**. A two-point-per-group curve cannot separate model size from
effective batch.

Four further confounds beyond batch: `lr` 1e-3→5e-4 (and `start_lr` halves) at
the top; `epochs` 30 vs 50 (2.5× more updates, 1.67× more data) for
base/large; `num_refine_steps` 2 for xsmall vs 3 elsewhere (so xsmall is not a
scaled-down version); and the **mechanism curriculum length itself scales with
size** (`jawk_curriculum_steps` 5000/10000/10000/15000,
`future_warmup_steps` 3000/5000/5000/8000) — confounding "bigger model learns
the workspace faster" with "bigger model was given a longer curriculum".
Plus `drop_path` 0.05/0.1/0.1/0.15.

## Safe wins — bit-identical results

1. **Gate `diagnostics.compute` and `jspace_metrics.compute` behind
   `itr % log_freq == 0`.** Removes 17 SVDs/step and ~50 syncs/step on 9 of 10
   steps. *Caveat:* `collapse.py:518,663` call `torch.randperm(..., device=...)`,
   consuming the training-device RNG and perturbing DropPath. Give those two
   sites a private `torch.Generator` **first**; then this is a true safe win.
2. **Hoist the 4-7 `active_k.item()` reads into one**, slice
   `workspace_Q[:, :k]` once, share the view. 4 syncs + 6 slices → 1 + 1.
3. **Vectorize `_gather_masked` and `_slot_indices`** into one `nonzero` on the
   (B,T) matrix plus a scatter, preserving `max_num_masked`/`-1` semantics.
   130 ms/step at B=256.
4. **Feed `SpanMaskCollator` the already-collated (B,T) tensor.** Removes the
   `train.py:1029` list-comp, the second `pad_sequence`, and the per-sample
   `.clone().detach()`. Masking algorithm untouched.
5. **Add `gradient_checkpointing` to `TextSpanJPAPredictor`.** The predictor has
   no drop_path, so re-materialization is numerically exact.
6. **Fix or delete `flops.py`** — it is dead code that would mislead.
7. **Batch the `spc.py` per-band `.item()`** (`:366,368,384`) into single
   `.tolist()` calls. 24 syncs → 3, identical values.
8. **Detach `_gac_z` / `_cmc_pass` after `backward()`** to halve peak activation
   memory. This changes the RNG/graph retention, so verify it is numerics-neutral
   before shipping.

## Trade-offs — change results, do not bundle with the safe wins

- Reducing `num_refine_steps` (2/3 → 1) or restricting `_iterative_refine` to
  masked slots. ~10 % of forward FLOPs; changes `span_preds` and every loss.
- Subsampling B·T before the covariance regularizer or the 17 SVDs.
- Any real fix to `train_ddp.sh` — necessary, but the cost is zero in practice
  because nothing it produced is usable.
- Consuming the 52 `wsd/sta/puc/rdc/spc/wsr` info-dict syncs only at log time
  (same generator caveat as safe win 1).
- Wiring up `ema_schedule: cosine` — changes every target-encoder trajectory.
- Passing `sta_warmup_steps` / `sta_update_interval` to
  `SpectralTransportAlignment` in `MechanismBundle.__init__`
  (`mechanisms.py:322-326` drops both though `from_config:406-407` forwards
  them). Free, because the bundle path is dead code.
