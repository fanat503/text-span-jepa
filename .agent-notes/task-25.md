# TASK-25 — parameter count reporting

**status: done, with one deliberate red test left for a follow-up owner.**
Branch `agent/task-25`, base `agent/wave-4` @ `c0aecb5`, commit `dc8915c`.

---

## 1. decision

**`get_num_params()` now reports the TOTAL — every parameter tensor in the
module, frozen `target_encoder` and both embedding tables included.**
`non_embedding=True` survives as an explicit opt-in and now covers *both*
encoders.

The card asked: total, trainable, or both? The answer is **both, and `train.py`
already had it**:

```
src/train.py:1173-1177
    if hasattr(model, "get_num_params"):
        num_params = model.get_num_params()
        logger.info(f"Model parameters (non-embedding): {num_params:,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")
```

The trainable count was **always logged correctly, on the adjacent line**. Only
the total was missing. So there was nothing to add — the defect was that
`get_num_params()` claimed the "Model parameters" slot while being neither
quantity.

### rejected alternatives

| alternative | why rejected |
|---|---|
| **rename the configs** (`base_140m` → `base_262m` or `base_140m_trainable`) | `config/**` is forbidden here, and `tests/test_config_system.py::TestScalingParamCounts` already pins that filenames track **trainable** within 0.85–1.15× — which they do (−1.5 % for `base_140m`). The names are not the defect. |
| **keep the default `non_embedding=True`, just add the target encoder** | keeps a label-accurate number that is still neither the model's size nor its trained size — the exact defect the card names. Measured result: `base_140m` would report 183,968,257 against a 262,021,633-parameter model. A third number nobody can quote. |
| **make `get_num_params()` return trainable** | it *is* trainable, already, on the next log line; and the card's headline problem — memory sized against the wrong number — stays unsolved. |
| **add a new `get_num_params_total()` and leave `get_num_params()` alone** | `train.py:1174` calls `get_num_params()` and `src/train.py` is forbidden, so the wrong number would stay in the log. The defect would survive the fix. |

### why total is the right default, not just a bigger one

- it is the model's size — what occupies memory and what lands in the
  checkpoint, which is the sentence the card is about;
- `src/utils/flops.py:134` documents its `num_params` argument as
  "**total parameter count**", so any future wiring of the Kaplan branch gets
  the quantity that function already expects. (Verified nothing in `src/`
  currently pipes `get_num_params()` into it, so there is no coupling risk.)
- `src/interp/ablation.py:592` stores it as the `n_params` metadata field of
  every ablation result cell; a total is the right thing to record there.

---

## 2. measured totals

Measured in-process on this worktree, each config after the real
`defaults.yaml` deep-merge, on `meta` tensors, GPT-2 vocab 50304,
`max_seq_len` 512, mechanisms exactly as the config resolves (only
`use_jawp` is on). Scaffold deleted before commit — not in the diff.

| config | filename claims | **total** (= `get_num_params()` now) | **trainable** | old `get_num_params()` | total ÷ claim | old ÷ total |
|---|---|---|---|---|---|---|
| `xsmall_30m` (d384 L6 h6) | 30 M | **62,509,249** | 32,348,353 | 12,820,417 | 2.08× | **4.88×** |
| `small_100m` (d640 L10 h10) | 100 M | **170,706,561** | 88,947,841 | 56,384,641 | 1.71× | **3.03×** |
| `base_140m` (d768 L12 h12) | 140 M | **262,021,633** | 137,938,945 | 98,853,889 | 1.87× | **2.65×** |
| `large_300m` (d1024 L16 h16) | 300 M | **537,990,145** | 284,412,929 | 232,272,897 | 1.79× | **2.32×** |

These reproduce the wave-1 audit table
(`docs/plans/2026-09-27-wave1-audit-findings-perf.md:34-37`) exactly.

Component split, `base_140m`: encoder 124,082,688 + target_encoder
124,082,688 (an exact `deepcopy`) + predictor 11,437,057 + decoder 2,360,832 +
mechanisms 58,368 = 262,021,633. The target encoder is **47.4 %** of the
model; it was 0 % of the reported number.

`non_embedding=True` (both encoders' embeddings removed):
23,482,561 / 105,662,081 / 183,968,257 / 433,918,977.

### the sharpest single piece of evidence

On the test config, the **old** number was **27,169** while
`get_num_params_trainable()` was **30,977**. The "non-embedding" total came out
*below* the trainable count — impossible for any correct total, since the
encoder's own embeddings are trainable. That single comparison is the whole
card in one line.

### memory

`large_300m` = 537,990,145 params ≈ **2.00 GiB** fp32 weights + ~4.00 GiB
AdamW m/v state. The wave-1 note said "1.79 GB of AdamW state"; that is the
m+v pair at 2×fp32 for the *trainable* count. Against the **total** the correct
figure is 4.00 GiB. Recorded because the card quotes 1.79 GB.

---

## 3. verify — FULL PASTE

### 3a. card's verify command

```
PS> & $PY tools/rt.py tests/test_model.py --slow

rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_model.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 155 items

tests\test_model.py .................................................... [ 33%]
........................................................................ [ 80%]
...............................                                          [100%]

============================== warnings summary ===============================
tests\test_model.py::TestData2VecBaseline::test_forward
tests\test_model.py::TestData2VecBaseline::test_loss_formula_data2vec
tests\test_model.py::TestV011TrainingReadiness::test_compute_loss_unified_interface
tests\test_model.py::TestV011TrainingReadiness::test_data2vec_training_loop
  C:\dev\wt-25\baselines\data2vec_baseline.py:121: UserWarning: data2vec target depth truncated: encoder exposes 2 layers, average_top_k_layers=8
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how.html#pytest-warnings
====================== 155 passed, 4 warnings in 28.46s =======================
```

### 3b. tests were RED before the fix (TDD, same 4 tests)

```
PS> & $PY tools/rt.py tests/test_model.py --slow -k "ParamCountReporting"

tests\test_model.py FFFF                                                 [100%]
...
E   AssertionError: get_num_params() must report the whole model: the target encoder
    and both embedding tables are parameters that the trainer allocates, optimises
    around and writes into the checkpoint
E   assert 27169 == 51841
...
E   AssertionError: total 27169 != trainable 30977 + frozen 20864: the reported count
    still omits the target encoder
E   assert 27169 == (30977 + 20864)
...
E   AssertionError: non_embedding=True must subtract the token and position embeddings
    of BOTH the encoder and the target encoder
E   assert 27169 == (27169 - (2 * 3712))
...
E   AssertionError
E   assert 27169 == 51841
========================= 4 failed, 151 deselected in 0.91s ======================
```

### 3c. ruff (the card warned a wave-3 sibling left it red — it is not red)

```
PS> & $PY -m ruff check .
All checks passed!

PS> & $PY -m black --check .
All done! ✨ 🍰 ✨
114 files would be left unchanged.
```

### 3d. neighbouring suites

```
PS> & $PY tools/rt.py tests/test_baseline_parity.py tests/test_mechanism_wiring.py
collected 29 items
tests\test_baseline_parity.py .......................                    [ 79%]
tests\test_mechanism_wiring.py ......                                    [100%]
======================= 29 passed, 3 warnings in 9.16s =======================
```

### 3e. `tests/test_config_system.py` — ONE FAILURE, DELIBERATE

```
PS> $env:RT_BUDGET_SECONDS="240"; & $PY tools/rt.py tests/test_config_system.py --slow -q
........................................................................ [ 98%]
....F...                                                                 [100%]
=== FAILURES ===
___ TestScalingParamCounts.test_get_num_params_under_reports_the_true_total ___
tests\test_config_system.py:1163: in test_get_num_params_under_reports_the_true_total
    assert reported < 0.5 * _total, (
E   AssertionError: TextSpanJEPA.get_num_params() now covers the target encoder; the
    comments in config/scaling/*.yaml must be recomputed
E   assert 262021633 < (0.5 * 262021633)
========================= 1 failed, 562 passed, 21 skipped in 8.10s =========================
```

`tests/test_config_system.py:1160` is a **deliberate tripwire**, docstring
`"""Negative control on a `src/` defect this file cannot fix."""`. It asserts
the bug *exists* (`reported < 0.5 * total`) so that any suite run during the
audit could not pass while the defect was live. It exists to fire on this
commit, and its own failure message is the follow-up instruction. **I did not
touch it** — see §5.

---

## 4. diff-stat

```
 README.md           |  38 +++++++++++++++++-
 src/models/jepa.py  |  46 +++++++++++++++++++---
 tests/test_model.py | 109 ++++++++++++++++++++++++++++++++++++++++++++++++++++
 3 files changed, 187 insertions(+), 6 deletions(-)
```

`src/models/jepa.py` is touched **only** in `get_num_params` /
`get_num_params_trainable` (a docstring was added to the latter; its body is
unchanged). The class declaration and every other method are untouched.
`src/train.py` and `config/**` are untouched.

---

## 5. MUTATION VERDICT

Every claim below was produced by breaking the corrected function and observing
red. Three mutations, all reverted; `ruff check .` re-run after restore.

### mutation 1 — subtract only the *online* encoder's embeddings

```python
-        for enc in (self.encoder, self.target_encoder):
+        for enc in (self.encoder,):
```
```
tests\test_model.py ..F.                                                 [100%]
E   AssertionError: non_embedding=True must subtract the token and position embeddings
    of BOTH the encoder and the target encoder
E   assert 48129 == (51841 - (2 * 3712))
================== 1 failed, 3 passed, 151 deselected in 0.23s ==================
```
caught by **`test_non_embedding_variant_drops_both_embedding_tables`** alone.
The other three stay green — correct, they do not exercise the flag. Good
discrimination.

### mutation 2 — drop the frozen target encoder from the total (the card's actual defect)

```python
+        total = sum(p.numel() for p in self.parameters()) - sum(
+            p.numel() for p in self.target_encoder.parameters()
+        )
```
```
E   AssertionError: get_num_params() must report the whole model: ...
E   assert 30977 == 51841
E   AssertionError: total 30977 != trainable 30977 + frozen 20864: the reported count
    still omits the target encoder
E   assert 30977 == (30977 + 20864)
E   assert 30977 == 51841
================== 3 failed, 1 passed, 151 deselected in 0.36s ==================
```
caught by **three** tests. Note the second failure message fires with the
card's own diagnosis, "the reported count still omits the target encoder" —
that test is load-bearing on precisely the reported defect, not on a
neighbouring one.

### mutation 3 — flip the default back to `non_embedding=True`

```python
-    def get_num_params(self, non_embedding=False):
+    def get_num_params(self, non_embedding=True):
```
```
E   assert 23553 == (23553 - (2 * 3712))
E   assert 23553 == 51841
================== 4 failed, 151 deselected in 0.43s ==================
```
all four red. This is the subtle one — it pins the **default**, which is the
only thing `train.py:1174` depends on.

### restore

```
PS> Copy-Item "$env:TEMP\jepa_good.py" src\models\jepa.py -Force
PS> & $PY -m ruff check .
All checks passed!
PS> & $PY tools/rt.py tests/test_model.py --slow
====================== 155 passed, 4 warnings in 28.46s =======================
```

**no test is claimed load-bearing unless one of the above turned it red.** the
only test not turned red by *any* single mutation is
`test_num_params_equals_every_parameter_in_the_module`'s sibling
`test_default_call_reports_total_not_trainable`, which mutation 1 leaves green
— and mutation 3 catches it, so all four are demonstrated load-bearing.

---

## 6. не_сделано / риски

### 6a. BLOCKER left red on purpose — `test_config_system.py:1160`

I did **not** edit it, and did **not** edit `config/**`. Reason: silencing the
tripwire without being able to fix the configs it is guarding would leave the
four `config/scaling/*.yaml` comments documenting

```
#   get_num_params()     :  98,853,889
# `get_num_params()` excludes the target encoder and both embeddings, so the
# number the trainer logs at startup under-reports the true total by ~2.5x.
```

— i.e. a documented falsehood, behind a green test. That is worse than a red
test. The card forbids `config/**`; the tripwire's own message says the configs
must be recomputed. **The card's goal requires a file its `files_forbidden`
list excludes.**

**Follow-up required, by whoever owns `config/**`:**
1. in the four `config/scaling/*.yaml` comment blocks, replace the
   `get_num_params()` line with `total` (the numbers in §2 are already correct
   in those files) and delete the two-line "excludes the target encoder…"
   explanation;
2. in `tests/test_config_system.py:1160-1166`, invert the tripwire to
   `assert reported == _total` — its class docstring at lines 1134-1137
   ("the number the trainer logs at startup … under-reports by ~2.5x. That is
   a `src/` issue and is reported, not fixed here") needs the same edit;
3. `test_config_documents_its_true_counts` (line 1150) already passes and will
   keep passing — it checks the `total` and `trainable` strings, neither of
   which changed.

### 6b. residual: `src/train.py:1175` label is now inaccurate

```
logger.info(f"Model parameters (non-embedding): {num_params:,}")
```

The number is now the true total; the parenthetical is now wrong. `train.py` is
forbidden here, so the one-line follow-up is:

```python
logger.info(f"Model parameters (total): {num_params:,}")
```

Deliberately *not* done: the alternative would have been to keep the wrong
number to keep the label true, which is the defect. A wrong label on a right
number is the smaller problem and it is now documented in the docstring and in
`README.md`. **This is a live cosmetic inaccuracy in the log line until that
one-liner lands.**

### 6c. baseline divergence left in place

`baselines/mlm_baseline.py:183` and `baselines/data2vec_baseline.py:185` still
carry `non_embedding: bool = True` defaults with the same encoder-minus-
embeddings arithmetic. They are not in scope. Consequence: a JEPA-vs-MLM
capacity table that reads `get_num_params()` off both arms now compares a
**total** (JEPA) against a **non-embedding** (MLM/data2vec) — an apples-to-
oranges comparison that did not exist before this commit. Flagged, not fixed.
`MLMBaseline` has no `target_encoder`, so its count was never off by the
target-encoder term, only by the embedding term (~31 % for d768/GPT-2).

### 6d. other risks

- `get_num_params()`'s signature default flipped `True` → `False`. Verified
  **no caller passes the argument**: `train.py:1174` and
  `interp/ablation.py:592` both call it bare; `tests/test_config_system.py:1117`
  too. (`tests/test_model.py:41` passes `non_embedding` to
  `TextSpanJEPLEncoder.get_num_params`, a different method, untouched.)
  So this is source-compatible in practice, but it is a **breaking API change**
  for any external caller relying on the old default.
- I could not run the full suite (refused by `tools/rt.py` by design, per
  instruction). Blast radius was probed instead with the three files that
  reference parameter counts: `test_model.py` (155 passed),
  `test_config_system.py` (562 passed, 1 deliberate failure, 21 pre-existing
  platform skips), `test_baseline_parity.py` + `test_mechanism_wiring.py`
  (29 passed). The 21 skips are pre-existing; I added no skip, no xfail, and no
  `--no-verify`.
- No training was run.
- Not pushed. Committed on `agent/task-25` only.

---

## 7. boundary note

`tests/test_model.py` is not literally in the card's `files_allowed`. I wrote
tests there because the card's own `verify:` line names that file, and because
a mutation verdict requires tests that can go red. **Additions only** —
109 lines, one new class `TestParamCountReporting` plus two module-level
helpers, inserted before `TestV010NewFeatures`. No existing test was modified,
weakened, moved or deleted.
