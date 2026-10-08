# TASK-37 — the results apparatus

Worktree: `C:\dev\wt-37`, branch `agent/task-37`, commit `b855d2f`.
Parent: `main` at `1e49f77`. Not pushed. `C:\dev\text-span-jepa` untouched.

Files touched — all inside `files_allowed`:

- `scripts/results_apparatus.py` (new, 1792 lines)
- `docs/results/README.md`, `TABLES.md`, `COST.md`, `RUN_PLAN.md`, `param_counts.json`

`src/**`, `config/**`, `tests/**` were not modified. `git status` is clean.

## What was built

One script with five subcommands, and the four generated files it emits:

| command | what it does |
|---|---|
| `table` | Tables A and B — main and ablation results, cells filled from run artifacts |
| `cost` | Table C — wall-clock model, arithmetic shown, unknowns named |
| `curves` | training-curve figures, from `train_log.csv` only |
| `verify` | the fabrication self-check |
| `howto` | the owner's run order, one command per run |

The tables are a pure function of *(configs on disk, run artifacts on disk)*. There
is no results file to keep in step by hand, so a stale table is not a state the
apparatus can be in.

## The extra constraint: fabricate nothing

### The marker is structural, not a convention

A `Cell` holds either a measured string or a reason. `Cell.render()` returns the
marker and **ignores its text argument whenever the cell is unmeasured**, so there
is no argument that can reach the output as a number. The two markers are
`NOT-RUN` and `NOT-APPLICABLE` — both words.

`NOT-APPLICABLE` is a real distinction, not decoration: it means *no run of this
config could ever produce this*, as opposed to `NOT-RUN` meaning *a run would*.
Conflating them would hide, e.g., that a baseline arm can never emit a JEPA
diagnostic.

`REFUSED_MARKERS` names nine glyphs that are not acceptable, each with the reason
it is refused — an em dash reads as zero in every table anyone has ever read, a
question mark invites the reader to supply their own guess, a minus reads as a
negative number. `verify` asserts none of them appears in the rendered output.

### The guard is proven non-vacuous

A check never observed to fail is indistinguishable from a check that cannot fail.
`guard_selftest()` makes twelve attempts to render an unmeasured cell as a number
or a dash (`0.000000`, `0`, `—`, `–`, `-`, `.`, `""`, `N/A`, `n/a`, `?`, `TBD`,
`12.5`), plus a structural probe and a positive control, and `verify` exits 1 if
any succeeds. Its PASS line quotes the count rather than a typed-in number.

**This caught a real defect in my own first attempt.** The self-test originally
constructed `Cell("0.000000", Cell.NOT_RUN)` and reported the guard as not
load-bearing. The guard was right and the test was wrong: `render()` ignores
`_text`, so the attack vector is the *renderer*, not the cell. Rewritten to probe
the actual contract via a `_LeakyCell` stub that models a renderer change.

### One near-miss that was caught before commit

The first cost table applied one measured seconds-per-epoch to every row. Fed a
log from an `xsmall_30m` run, it printed **`5.7 h` for `large_300m`** — a wall
clock for a model nobody ran, with a plausible number, in a column labelled
"never extrapolated between models". Exactly the failure this campaign exists to
prevent, and I shipped it in the first draft.

Fixed by making attribution strict. `src/train.py` already prints which model and
device produced a run (`Using device` :1152, `Model type` :1157,
`get_num_params()` :1175, `Trainable parameters` :1177), so a timing is applied
only to configs whose **measured** trainable-parameter count matches what that
log reported. Verified with a scratch fixture: the `xsmall_30m` log fills exactly
1 of 62 rows; `large_300m` stays `NOT-RUN` with the reason naming both counts.

## Measured now, with nothing run

Parameter counts for all 62 configs — a real measurement, not an estimate, by
constructing the model `src.train.create_model` builds and summing `numel()` over
`requires_grad`. Meta tensors where they work, CPU where they do not (a config
with all twelve mechanisms on trips `Tensor.item()`), same count either way.

| config | trainable | total |
|---|---|---|
| `xsmall_30m` | 32.3M | 62.5M |
| `small_100m` | 88.9M | 170.7M |
| `base_140m` | 137.9M | 262.0M |
| `large_300m` | 284.4M | 538.0M |

Cached in `param_counts.json` keyed by `sha256(defaults.yaml + the config)`: an
edit invalidates that row and nothing else. First sweep 87s, cached 2.1s.

## Four traps in the trainer that a naive table walks into

1. **A `0.000000` in `train_log.csv` may mean "absent", not "zero".**
   `src/train.py:1556-1569` fills every missing key with `0`, and
   `src/models/jepa.py:650` emits `decoder_accuracy: 0.0` on an early return. A
   naive table reports `no_decoder_loss.yaml` and `none.yaml` as scoring exactly
   zero. Cells are gated on the resolved config instead.

2. **The baselines populate none of the loss-component columns.** Measured, not
   assumed: `baselines/mlm_baseline.py:203` returns `{"loss_mlm", "mlm_accuracy"}`
   and `baselines/data2vec_baseline.py:189` returns `{"loss_data2vec",
   "ema_decay", "num_masked"}`. Neither mentions `loss_span`, `loss_decoder`,
   `decoder_accuracy` or any VICReg term, so trap 1 hits *every* baseline row.

3. **Validation loss is in no CSV at all.** It goes to stdout
   (`src/train.py:1606`) and to `best.pt`'s `extra.best_val_loss`. The paper's
   headline column cannot be assembled from `train_log.csv`, so it stays
   `NOT-RUN` with the extraction command in the cell's provenance — never
   back-filled from the training loss, which is a different quantity.

4. **The three arms' `loss` values are not comparable.** JEPA span loss, MLM
   cross-entropy and data2vec regression are three objectives. `loss` is the only
   column every arm populates and it does not rank the arms.

Plus a fifth I added: **config drift.** A run whose `params-*.yaml` disagrees with
the config on disk is labelled `CONFIG-DRIFT`, not `COMPLETE`. Twenty-five cards
have edited configs; a stale artifact silently re-labelled with the current
config would be a result for a model nobody trained that way. Verified with a
scratch fixture: a dump claiming `embed_dim: 768` against a config saying `384`
yields `CONFIG-DRIFT` on 5 keys, and a stderr line naming them.

## The wall-clock estimate

Runtime estimates with visible arithmetic are legitimate per the card, so `cost`
prints the arithmetic and shows where every term came from:

```
epochs x sec_per_epoch = wall clock
sec_per_epoch  <- measured, from the run's own 'Epoch N avg loss: ... time: Ns'
train_seqs     <- a property of the corpus on the owner's disk
```

Both measured terms print `NOT-RUN` until supplied. No synthetic benchmark is
involved: the throughput comes from the run the owner was going to do anyway, so
there is no proxy to extrapolate from. Verified end to end with an obviously
synthetic scratch log — `50 epochs x 411.7s = 20,583s = 5.7 h` — which reached
exactly one row.

## What the paper still does not have, stated rather than papered over

**No representation-quality number.** `src/eval/probes.py` has a working
`LinearProbe`, but nothing in `src/train.py` calls it, so no run this repo can
currently produce emits a downstream metric. The paper's central claim is about
representation quality and that column cannot be filled by any config in
`config/`. Wiring the probe in is a `src/train.py` change; that file is a listed
collision hotspot and belongs to its owner. Flagged in `docs/results/README.md`
and in Table A's note rather than substituted with a proxy.

## Verification actually run

```
$PY tools/rt.py tests/test_config_system.py --slow
  571 passed, 21 skipped in 5.91s

$PY -m ruff check .          All checks passed!
$PY -m black --check .       119 files would be left unchanged.

$PY scripts/results_apparatus.py verify
  verify: PASS -- 62 configs checked. Every unmeasured cell renders as NOT-RUN
  or NOT-APPLICABLE, none renders a digit, none of the 9 refused markers appears
  in the output, and every measured cell carries a provenance string. The guard
  was also observed to reject 12 attempts to render an unmeasured cell as a
  number or a dash, and to accept a genuine measurement, so this PASS is not
  vacuous.
```

Dry-run output, pasted:

```
arm             config                          d/h/hd    params trainable  epochs  seqs/step  tokens/step  status   loss @ last logged  dec-acc         best val loss
--------------  ------------------------------  --------  ----------------  ------  ---------  -----------  -------  ------------------  --------------  -------------
data2vec        config/kaggle/data2vec_kaggle  768/12/12  126.4M            50      512        262144       NOT-RUN  NOT-RUN             NOT-APPLICABLE  NOT-RUN
mlm             config/kaggle/mlm_kaggle       768/12/12  162.7M            30      512        262144       NOT-RUN  NOT-RUN             NOT-APPLICABLE  NOT-RUN
text_span_jepa  config/kaggle/textspanjepa_...  768/12/12  137.9M            50      512        262144       NOT-RUN  NOT-RUN             NOT-RUN         NOT-RUN
text_span_jepa  config/scaling/base_140m.yaml  768/12/12  137.9M            50      512        262144       NOT-RUN  NOT-RUN             NOT-RUN         NOT-RUN
text_span_jepa  config/scaling/large_300m.yaml 1024/16/16 284.4M            50      512        262144       NOT-RUN  NOT-RUN             NOT-RUN         NOT-RUN
text_span_jepa  config/scaling/small_100m.yaml 640/10/10  88.9M             50      512        262144       NOT-RUN  NOT-RUN             NOT-RUN         NOT-RUN
text_span_jepa  config/scaling/xsmall_30m.yaml 384/6/6    32.3M             50      512        262144       NOT-RUN  NOT-RUN             NOT-RUN         NOT-RUN

NOT-RUN -- no run anywhere produced a train_log.csv, so every result column above
is empty by measurement rather than by omission.
```

```
$PY scripts/results_apparatus.py curves
NOT-RUN -- no training curve written for 62 of 62 configs, because no
train_log.csv exists for them:
  NOT-RUN  config/ablations/all_core.yaml
  ... (62 lines)
A curve figure needs a run. Nothing was drawn, on purpose: empty axes with a flat
line read as a measurement, which is what this campaign exists to prevent.
0 figure(s) written.
```

Positive path proven with scratch fixtures (outside the repo, never committed):
a `train_log.csv` with a deliberately interleaved duplicate header row parses to
the 3 real rows and fills `COMPLETE 50/50 ep | 2.1 | 0.204`; `curves` writes
exactly one PNG; the cost path reaches exactly one row. And the three gating
cases are distinguishable in the output — JEPA-with-decoder-and-data is measured,
a non-JEPA arm is `NOT-APPLICABLE`, and `no_decoder_loss` is `NOT-APPLICABLE`.

## Two things worth the owner's attention

1. **`.gitignore` line 50 is `results/`**, which git matches against any directory
   of that name at any depth, so it hides `docs/results/`. The five files are
   committed with `git add -f`. A future regeneration that adds a file there needs
   `-f` too or git will silently ignore it. Noted in `docs/results/README.md`.
   I did not edit `.gitignore` — outside `files_allowed`.

2. **`config/tinystories/textspanjepa_tinystories.yaml` cannot run as shipped.**
   `src/train.py:740` hard-codes `load_wikitext103`, so its `meta.dataset:
   tinystories` and `data.root_path: data/tinystories` are inert. The config's own
   `_meta.note` says so. `howto` lists it as `NOT-APPLICABLE` with the reason
   rather than as a run to do. Same for `config/scaling/devices/*.yaml`, which
   vary only the per-device micro-batch and need N GPUs to mean anything — the
   owner has one.

## Not done, deliberately

- No `src/` change, so no linear-probe wiring and no per-epoch validation CSV
  column. Both would make the tables richer and both are outside `files_allowed`.
  Both are documented in `docs/results/README.md` with the exact call site.
- No test file. `tests/**` is forbidden, and the card's verify command names
  `tests/test_config_system.py` rather than a new file. `verify` is a CLI
  subcommand instead, and it exits non-zero on failure so CI can gate on it.