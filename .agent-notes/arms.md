# External-baseline ablation arms — BYOL, Barlow, VICReg, SimSiam, MLM, data2vec

Branch `agent/baseline-arms`, commit `3a39ee1`, base `agent/wave-10` (`03eb344`).
Worktree `C:\dev\wt-arms`. **Not pushed.** `git status` shows only the seven owned
files plus this note.

Owned: `config/ablations/{byol,barlow,vicreg,simsiam,mlm,data2vec}.yaml` and
`tests/test_config_system.py` (a completeness guard only — no existing test was
touched). `defaults.yaml`, every `src/` file and every pre-existing ablation
config are unmodified.

---

## 1. The arms

| file | `meta.model_name` | canonical | class `create_model` builds | protocol | trainable |
|---|---|---|---|---|---|
| `config/ablations/byol.yaml` | `byol` | `byol` | `BYOLBaseline` | `compute_loss` | 0.993x |
| `config/ablations/barlow.yaml` | `barlow` | `barlow` | `BarlowTwinsBaseline` | `compute_loss` | 0.993x |
| `config/ablations/vicreg.yaml` | `vicreg` | `vicreg` | `VICRegBaseline` | `compute_loss` | 0.993x |
| `config/ablations/simsiam.yaml` | `simsiam` | `simsiam` | `SimSiamBaseline` | `compute_loss` | 0.993x |
| `config/ablations/mlm.yaml` | `mlm` | `mlm` | `MLMBaseline` | `compute_loss` | ~1.28x |
| `config/ablations/data2vec.yaml` | `data2vec` | `data2vec` | `Data2VecTextBaseline` | `data2vec_forward` | JEPA-like |

**Naming rule the guard encodes:** `baselines/<arm>_baseline.py` →
`config/ablations/<arm>.yaml` → `meta.model_name: <arm>`. One rule, three
places, no table to keep in sync.

Note `barlow`: the *arm* name is `barlow` and the *class* is
`BarlowTwinsBaseline`. Spelling the config `barlow_twins` would match no prefix
in `_ARM_PREFIXES`, be returned unchanged and raise in `create_model`.

**Shape:** each arm declares no dimensions, so all six inherit the reference
ViT-Base shape (768 / 12 / 12, mlp_ratio 4.0, drop_rate 0.0) — the shape the four
SSL arms were parameter-matched at. Each arm also declares **no** `use_*` /
`lambda_*` key: `create_model` never builds a `TextSpanJEPAConfig` for these
names, so no `MechanismBundle` exists and the GWP keys deep-merged in from
`defaults.yaml` are inert. Same reasoning `config/kaggle/mlm_kaggle.yaml`
records. Shape keys `predictor_*`, `jawk_*` etc. are inert for the same reason.

The only data delta is in `mlm.yaml` and `data2vec.yaml`:
`data.span_length_range: [1, 1]` — the BERT-style single-token corruption every
other MLM/data2vec config in this repo uses (`config/wikitext/mlm_*.yaml`,
`config/wikitext/data2vec_*.yaml`, `config/kaggle/*_kaggle.yaml`). It is live:
`src/train.py:1431` passes `span_length_range` to `SpanMaskCollator`
unconditionally.

---

## 2. How `model_name` resolves (the silent-JEPA trap)

`src/train.py:1391` reads `args["meta"]["model_name"]` → `_normalize_model_name`
(lowercase + match the `_ARM_PREFIXES` table, first match wins) → `create_model`
dispatches on the canonical name and stamps `LOSS_PROTOCOLS[canonical]` onto the
model as `loss_protocol`.

Two wrong outcomes, and only one of them is loud:

1. **No prefix match** → the name is returned as-is → `create_model` raises
   `ValueError` at startup. Loud. Fine.
2. **Prefix matches `text_span_jepa`** → a `TextSpanJEPA` is built, it trains, the
   loss falls, and the ablation row is a duplicate of the JEPA column. **Silent.**

So the guard does not check the name — it **builds the model** from the arm's own
merged config and asserts `isinstance(model, TheClass)` and
`not isinstance(model, TextSpanJEPA)`. Measured over the whole directory:

```
arm                      meta.model_name      -> canonical     class built          protocol
------------------------------------------------------------------------------------------------
all_core.yaml            text_span_jepa       text_span_jepa   TextSpanJEPA         jepa_targets
barlow.yaml              barlow               barlow           BarlowTwinsBaseline  compute_loss
byol.yaml                byol                 byol             BYOLBaseline         compute_loss
data2vec.yaml            data2vec             data2vec         Data2VecTextBaseline data2vec_forward
mlm.yaml                 mlm                  mlm              MLMBaseline          compute_loss
simsiam.yaml             simsiam              simsiam          SimSiamBaseline      compute_loss
vicreg.yaml              vicreg               vicreg           VICRegBaseline       compute_loss
... (the other 39 arms)  text_span_jepa       text_span_jepa   TextSpanJEPA         jepa_targets

46 arms, 1 distinct budget(s):
  epochs=50  max_seq_len=512  micro-batch=64  grad_accum=8
  -> effective batch 512 sequences = 262,144 tokens/step   [46 arms]
output folders distinct: True
```

(The 39 JEPA rows build for real here too, at a shrunk shape; that dump was a
one-off script in the temp dir, not added to the repo.)

---

## 3. The budget table

**Every arm in `config/ablations/` — all 46 — gets one budget:**

| | value | source |
|---|---|---|
| epochs | 50 | `defaults.yaml: optimization.epochs` |
| micro-batch | 64 | `defaults.yaml: data.batch_size` |
| grad accumulation | 8 | `defaults.yaml: optimization.grad_accum_steps` |
| max_seq_len | 512 | `defaults.yaml: data.max_seq_len` |
| **effective batch** | **512 sequences** | 64 × 8 |
| **tokens / optimizer step** | **262,144** | 512 × 512 |
| lr / warmup / wd | 1e-3 / 10 / 0.04→0.4 | `defaults.yaml` (untouched) |

Each of the six arms declares `optimization: {}` — a no-op in `_deep_merge` —
which states "this arm inherits the reference schedule" instead of leaving the
reader to infer it, and makes the budget paragraph in each header true of the
*merged* config rather than of an intention.

**Deliberate decision:** the baselines do **not** inherit the wikitext column's
per-method schedule. `config/wikitext/mlm_wikitext_*.yaml` runs MLM at
`epochs: 30, lr 5e-4, wd 0.01`, and `config/kaggle/mlm_kaggle.yaml` repeats it —
that is the *capacity* column (one method per column, each at its own tuned
schedule). This is the *ablation* table. If the MLM arm here ran 30 epochs
against 50-epoch JEPA rows, the row would measure the budget. `lr` is likewise
**not** pinned by the guard: a per-method learning rate is a legitimate
experimental variable, and `test_config_system.py` has no business choosing an
experiment's schedule. Budget is what is pinned.

---

## 4. The guard

`tests/test_config_system.py` gains one class,
`TestBaselineArmsAreCompleteAndComparable` — 14 tests, **no skip, no xfail**.

| test | what it holds |
|---|---|
| `test_every_baseline_module_has_an_arm` | every `baselines/*_baseline.py` has `config/ablations/<arm>.yaml` |
| `test_each_module_defines_exactly_one_baseline_class` [×6] | one `*Baseline` class per module, so a second class cannot hide behind the first's arm |
| `test_the_arm_builds_that_class_and_not_a_jepa` [×6] | the arm's own merged config builds that class, is not a `TextSpanJEPA`, and has the declared `loss_protocol` |
| `test_every_arm_in_this_directory_gets_the_same_training_budget` | all 46 yamls resolve to one `(epochs, max_seq_len, batch_size, grad_accum_steps)` |

Discovery is **filesystem-driven, not table-driven**: `BASELINE_MODULES` is
`glob("*_baseline.py")` and the arm name is derived from the module name. A
baseline added tomorrow is covered from the moment it lands, with no edit to
this file. Classes are found by `inspect.getmembers` filtered on
`*Baseline` + `nn.Module` + source file, so a re-export does not masquerade as a
second class.

The budget guard compares the **merged** config, not the file: an arm that
declares no `optimization:` block inherits the reference, and that is the
property being asserted. Its failure message orders groups smallest-first, so
the one arm that deviates leads.

---

## 5. Verify paste

```
$ & $PY tools\rt.py tests\test_config_system.py --slow
rt.py: threads=1  total_budget=90s  slow_ok=True
collected 642 items

tests\test_config_system.py ............................................ [  6%]
........................................................................ [ 18%]
.......................................................................s [ 29%]
........................................................................ [ 40%]
........................................................................ [ 51%]
.................................................................ss...s. [ 62%]
.......s.....................s.....s..ss..........ssssssss...ss...s..... [ 74%]
...s.....................s.....s..ss..........ssssssss.................. [ 85%]
........................................................................ [ 96%]
......................                                                   [100%]

====================== 609 passed, 33 skipped in 11.00s =======================
```

Baseline before this change, same command, same box: **571 passed, 21 skipped
in 11.57s**. Delta **+38 passed, +12 skipped** = 6 configs × 4 passing
parametrized tests + 14 guard tests, and 6 configs × 2 `pytest.skip`s.

The +12 skips are `TestEMATargetStillMoves`' two `if not _is_jepa(rel):
pytest.skip("data2vec / MLM baselines own their EMA")` branches, hit once per
new non-JEPA arm. **The skip is correct** — `ema_tau_*` is inert for a
non-JEPA arm — but its message is now stale: it names only data2vec and MLM
while BYOL (which genuinely owns an EMA, via `SELF_EMA_ARMS`) and the other
three skip on it too. Editing that message was left to `src/`'s and the
existing tests' owner rather than done here.

```
$ & $PY -m ruff check tests\test_config_system.py
All checks passed!

$ & $PY -m black --check tests\test_config_system.py
All done! 1 file would be left unchanged.
```

No training was run. `src.train` was never invoked as an entry point; the only
model construction is inside the guard, at vocab 64 / seq 16 / dim 32 / depth 1.

---

## 6. Mutation verdicts — three, each restored byte-identical

### M1 — remove one arm (the mandated one)

```
$ Remove-Item config\ablations\byol.yaml
$ & $PY -m pytest tests\test_config_system.py -k "TestBaselineArmsAreCompleteAndComparable"

tests\test_config_system.py F.......F.....                               [100%]

_ TestBaselineArmsAreCompleteAndComparable.test_every_baseline_module_has_an_arm _
E   AssertionError: 1 of 6 baselines have no ablation arm: ['config/ablations/byol.yaml'].
    An implemented baseline with no config cannot be compared, and nothing else in the tree
    notices that -- this test is the only thing standing between the next baseline and the
    state this directory was in. Write the arm as config/ablations/<module name minus
    '_baseline'>.yaml.

_ TestBaselineArmsAreCompleteAndComparable.test_the_arm_builds_that_class_and_not_a_jepa[byol_baseline.py] _
E   AssertionError: baselines/byol_baseline.py has no arm at config/ablations/byol.yaml

================ 2 failed, 12 passed, 622 deselected in 0.63s =================
restored hash=D2445A7545631405C3ABED31993EA7A0CA00AA394890A9EF4A03A1DD8CC2A5C4
```

The first version of the second test raised a bare `FileNotFoundError` from the
yaml loader; an `is_file()` assertion was added so a missing arm fails with the
same sentence as the completeness test. Both were then re-verified red.

### M2 — the silent JEPA

```
$ (barlow.yaml)  model_name: barlow  ->  model_name: text_span_jepa_barlow

_ ...test_the_arm_builds_that_class_and_not_a_jepa[barlow_baseline.py] _
E   AssertionError: config/ablations/barlow.yaml: meta.model_name='text_span_jepa_barlow'
    normalises to 'text_span_jepa', not 'barlow'. Either it matches no prefix in
    `_ARM_PREFIXES` (and `create_model` raises), or it normalises to some other arm
    -- including `text_span_jepa`, which would train a JEPA and leave this arm looking
    like it ran.
E   assert 'text_span_jepa' == 'barlow'

================ 1 failed, 13 passed, 628 deselected in 0.78s =================
restored hash=7DC8652CB481017A9E1FE102AB83471D8EB7397E66A6248001520AF21F917D63
```

### M3 — the budget

```
$ (vicreg.yaml)  optimization: {}  ->  optimization:\n  epochs: 30

_ ...test_every_arm_in_this_directory_gets_the_same_training_budget _
E   AssertionError: 2 training budgets across 46 arms of config/ablations/ on
    ['optimization.epochs', 'data.max_seq_len', 'data.batch_size', 'optimization.grad_accum_steps']:
    {'optimization.epochs': 30, ...} <- 1 arm(s), smallest first: ['config/ablations/vicreg.yaml'] |
    {'optimization.epochs': 50, ...} <- 45 arm(s), smallest first: ['config/ablations/all_core.yaml',
    'config/ablations/barlow.yaml', 'config/ablations/byol.yaml']. A comparison whose arms see
    different token budgets measures the budget, not the method. Declare an arm's schedule only
    with a reason.

================ 1 failed, 641 deselected in 0.77s =================
restored hash=40F06435A0CBD2788EA9F6609E84D29E7C454F72B78AB236869356438FA32CED
```

---

## 7. Риски / risks

**1. The two "views" of BYOL/Barlow/VICReg/SimSiam are not the papers'
augmentation.** `src/train.py::compute_loss` hands
`model.compute_loss(masked_input_ids, original_input_ids, mask_positions)`, so
view A is this repo's span-masked input at the reference mask curriculum
(0.15 → 0.35, spans of 3–10) and view B is the clean input. These are
non-instance-discrimination methods that want two *independent* views. They get
one masked and one clean. **This is recorded in all four headers, not fixed** —
it lives in `src/train.py` and the data pipeline, which this arm does not own.
It is the single biggest threat to the validity of the four SSL rows and belongs
on the experiment's list of decisions.

**2. `data.mask_ratio` is inert in every shipped config.** `defaults.yaml`
always declares `model.mask_ratio_start` *and* `model.mask_ratio_end`, and
`src/train.py:1426-1435` hands both to `SpanMaskCollator` as a ramp that
overrides `data.mask_ratio` whenever both are present
(`src/masks/span.py:35`). So the `data.mask_ratio: 0.15` in
`config/wikitext/mlm_*.yaml` and `config/wikitext/data2vec_*.yaml` and
`config/kaggle/*_kaggle.yaml` does nothing. For the same reason these two arms
declare `span_length_range` but **not** `data.mask_ratio`: declaring a key that
looks load-bearing and is not is worse than leaving it out. **Not fixed here** —
the override is in `src/train.py` and `defaults.yaml`. Whoever owns those should
know that three shipped configs carry a dead key.

**3. Drop-path asymmetry between JEPA and the baselines.** The reference trains
JEPA with `model.drop_path_rate: 0.1`; `create_model` does not pass
`drop_path_rate` to any baseline, so all six run with no drop-path at all. The
arms declare no `drop_rate` (inheriting the reference 0.0) rather than copying
the `0.1` from the wikitext baseline columns — matching the JEPA rows is what
this table is for, and a partial compensation would introduce a config difference
instead of removing one. The asymmetry is real and is the baselines' favour;
flagged, not fixed.

**4. MLM is ~1.28x JEPA's trainable parameters**, because of its untied
`(embed_dim, vocab_size)` head. Measured by
`tests/test_baseline_parity.py::TestTrainableCapacityAsymmetry`. The header says
so; the arm does not hide it. A reviewer comparing `mlm` against `none` is
comparing 1.28x capacity against 1.0x. `data2vec` is the closest capacity
match among the six.

**5. lr 1e-3 is JEPA's, not the baselines'.** Inherited deliberately (see §3), but
it is a real risk that BYOL/SimSiam/Barlow underperform their published recipes
for optimisation reasons rather than method reasons. If the SSL rows come back
poor, the first thing to check is the learning rate, not the method — and
changing it breaks the budget guard by design, which is the correct friction.

**6. Two different budget conventions now coexist in the repo.** This table is
one budget for 46 arms; `config/wikitext/*` has per-column tuned schedules
(MLM at 30 epochs / lr 5e-4). Both are defensible, but a paper that quotes a
number from this table and a number from the wikitext column is quoting two
protocols. Worth one line in the paper's experimental-setup section.

**7. `config/ablations/README.md` is now stale** — it says "all **40** files" in
this directory and its "other arms" table does not list these six. It is not in
this task's file ownership, so it was left alone; it needs one edit before the
table is quoted anywhere. Same for `config/wikitext/`: the capacity column has
no SSL rungs, only MLM and data2vec.

**8. Skip count grew by 12** (21 → 33) and the reason is benign but worth
stating plainly rather than letting a reader diff the numbers and wonder: the
six new non-JEPA configs each hit the two pre-existing
`TestEMATargetStillMoves` non-JEPA branches. No test was weakened and no skip was
added by this change.