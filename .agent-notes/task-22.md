# TASK-22 — `src/utils/flops.py`: correct or gone

**status: DONE — route taken: FIX (not delete).**
Worktree `C:\dev\wt-22`, branch `agent/task-22` off `agent/wave-2`. Nothing pushed.

---

## 0. The hazard, dealt with first

The sibling agent's warning was real, and I confirmed the mechanism before
trusting any number.

`pip install -e .` installed an **editable finder**:

```
site-packages\__editable__.text_span_jepa-1.0.0rc15.pth
  -> __editable___text_span_jepa_1_0_0rc15_finder.install()
MAPPING: dict[str, str] = {'src': 'C:\Users\<user>\tmp\clone-check-1\src'}
```

`sys.meta_path` order is `[DistutilsMetaFinder, BuiltinImporter, FrozenImporter,
PathFinder, _EditableFinder, _EditableFinder]`. The editable finder is
**appended**, so it sits *after* `PathFinder`. Consequence: it only wins when
the running script's own directory does not contain `src`. Measured both ways:

```
=== script INSIDE wt-22 ===        src -> C:\dev\wt-22\src\utils\flops.py      (correct)
=== script from TEMP dir ===       src -> C:\Users\<u>\tmp\clone-check-1\src   (STALE)
```

`tools/rt.py` is safe because pytest puts rootdir first. **Ad-hoc
`python myscript.py` is not safe unless the script lives in the worktree.** Every
probe I wrote does `sys.path.insert(0, REPO)` *and* asserts
`os.path.realpath(src.__file__).startswith(REPO)`, so a stale import would have
raised rather than silently produced wrong ratios. `src/utils/__init__.py`
(unchanged) does the same thing more loosely — it relies on cwd.

---

## 1. Which route, and why

The card offers two routes and the honest answer is **fix**, for three
reasons:

1. **The test was not testing a wrong expectation.** `test_flops_estimation`
   asserted `result["tflops"] > 0` and `train_result["pflops"] > 0`. Those are
   true of any function that multiplies its inputs; they do not encode a
   correct cost. The *implementation* was wrong, not the expectation. Deleting
   the test to make a linter pass is exactly what the card forbids, and it
   would have hidden the defect rather than fixed it.
2. **The function is not the problem; its signature is.** The defect is
   structural (`6·N·L·B` cannot express depth, width, a target encoder, or
   repeated predictor passes). That is fixable in the function I own.
3. **A FLOPs estimator is a real deliverable for a scaling study.** The repo
   has a `config/scaling/` ladder; deleting the only compute-accounting helper
   leaves that with nothing.

I could not touch `src/models/**`, so I did **not** change the model to match
the estimate. I changed the estimator to match the model, and pinned it against
an independent measurement.

---

## 2. What the old formula actually got wrong (measured, not assumed)

Card's premise confirmed on my own measurements, base_140m dims, vocab 4096,
B=4, mask_ratio 0.35. The card's `3.147e11 / 6.295e11 / 1.259e12` figures are
reproduced exactly by `6·N·L·B` with **N = trainable params (102,451,201)** —
that is the input the card used, and it is confirmed here.

`FlopCounterMode` counts **only `mm`/`addmm`/`bmm`/`baddbmm`**. Verified
directly:

```
F.scaled_dot_product_attention(q, q, q)  ->  0.0 FLOPs
nn.MultiheadAttention(...)(z, z, z)       ->  in_proj + out_proj only
torch.linalg.svd / svdvals / eigh / qr   ->  0.0 FLOPs
cross_entropy / log_softmax / argmax     ->  0.0 FLOPs
```

**This is the finding that shaped the design.** The card says "add the
attention term ... and pin it against `FlopCounterMode`." Those two instructions
conflict on this build: the reference contains no attention term, so adding one
moves the estimate *away* from the reference. I resolved it by reporting
attention separately and documenting why it is not folded in.

Each term, isolated and measured:

| component | analytic | measured | ratio |
|---|---|---|---|
| encoder fwd+bwd | `3 · 24·B·T·D²·L` | 5.218385e11 | **3.00000** |
| target encoder (no_grad fwd) | `24·B·T·D²·L` | 1.739462e11 | 1.00000 |
| predictor, refine branch | `3 · 24·B·R·T·Dp²·Lp` | — | **delta 0.000e+00** |
| predictor, future branch | `3 · 24·B·Σ(T−d)·Dp²·Lp` | — | **delta 0.000e+00** |
| decoder | `3 · 2·M·(4D² + D·V)` | 1.175873e10 | **3.00000** |
| VICReg covariance | `3 · 2·B·T·D²` | 3.623879e09 | **3.00000** |

So the backward multiplier is exactly 3, and the predictor multiplier is
`num_refine_steps + len(live_offsets)` — matching the card's
`(refine + 1 + len(offsets))` except the `+1` is wrong: `_iterative_refine`
runs `num_refine_steps` passes and there is no additional full pass. Measured
by hook: `block passes = 36 = depth 6 × (refine 3 + offsets 3)`. I used the
measured count.

**The last 1.5-2% residual** was the per-step diagnostics
(`CollapseDiagnostics.compute`, `JSpaceMetrics.compute`), called unconditionally
at `jepa.py:872`/`:880`. I attributed it by dispatch-tracing every matmul to its
caller frame. It is 13.6×/15.3×/18.2× `2·B·T·D²` at T=128/256/512 — **not a
constant**, so I did not fit a magic coefficient; the docstring says so and
leaves it to the caller.

---

## 3. Files changed (only the two I own)

```
 src/utils/flops.py  | 292 +++++++++++++++++++++++++++++++++++++++++++++-----
 tests/test_model.py | 139 ++++++++++++++++++++++++++++++++++++-
 2 files changed, 388 insertions(+), 43 deletions(-)
```

`src/utils/__init__.py` still re-exports the three old names — I did not add
`estimate_jepa_step_flops` to it because that file is outside my boundary. See
§7.

---

## 4. VERIFY — full paste, with measured ratios

### 4a. The ad-hoc FlopCounterMode probe (`.agent-notes/_probe_verify.py`)

```
src guard OK -> C:\dev\wt-22\src\__init__.py
torch 2.13.0+cpu | threads 1

=== OLD estimator (Kaplan 6*N*L*B) vs NEW vs FlopCounterMode ===
    base_140m dims (768/12, predictor 384/6), vocab 4096, B=4, mask 0.35

    T    real fwd+bwd         OLD 6ND  OLD/real |         NEW est  NEW/real  OLD/NEW
  128    5.612274e+11    3.147301e+11   0.56079 |    5.529057e+11   0.98517   0.5692
  256    1.130262e+12    6.294602e+11   0.55692 |    1.111460e+12   0.98336   0.5663
  512    2.273488e+12    1.258920e+12   0.55374 |    2.228600e+12   0.98026   0.5649

(OLD/real degrades 0.58 -> 0.52 with T; NEW/real is flat. The NEW number
 still counts the per-step diagnostics, which the OLD one ignores too.)

=== NEW vs FlopCounterMode with the per-step diagnostics disabled ===
    (CollapseDiagnostics.compute + JSpaceMetrics.compute, called
     unconditionally by compute_loss_with_targets)

    T  real (no diag)         NEW est  est/real       diag cost
  128    5.530159e+11    5.529057e+11   0.99980    8.211533e+09
  256    1.111817e+12    1.111460e+12   0.99968    1.844533e+10
  512    2.229419e+12    2.228600e+12   0.99963    4.406932e+10

=== component breakdown @ T=512 ===
             total_flops  2.228600e+12
           encoder_flops  1.043677e+12
    target_encoder_flops  3.478924e+11
         predictor_flops  8.061007e+11
           decoder_flops  2.368261e+10
        covariance_flops  7.247757e+09
         attention_flops  2.875623e+11

=== FlopCounterMode counts NO attention on this build ===
  F.scaled_dot_product_attention -> 0.0 FLOPs
  => attention_flops is reported separately, never folded into total_flops.
```

**Error against the reference, old vs new:**

| T | old `6·N·L·B` / real | new / real | new / real (no diagnostics) |
|---|---|---|---|
| 128 | 0.5608 | 0.9852 | 0.99980 |
| 256 | 0.5569 | 0.9834 | 0.99968 |
| 512 | 0.5537 | 0.9803 | 0.99963 |

The old estimate is ~1.8× low and **degrades with T** (0.5608 → 0.5537); the new
one is flat and, with instrumentation excluded, within 0.04%.

### 4b. The card's mandated test run

```
> & $PY tools\rt.py tests/test_model.py --slow
rt.py: ...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_model.py
rt.py: threads=1  total_budget=900s  slow_ok=True
============================= test session starts =============================
collected 151 items

tests\test_model.py .................................................... [ 34%]
........................................................................ [ 82%]
...........................                                              [100%]

============================== warnings summary ===============================
tests\test_model.py::TestData2VecBaseline::test_forward
tests\test_model.py::TestData2VecBaseline::test_loss_formula_data2vec
tests\test_model.py::TestV011TrainingReadiness::test_compute_loss_unified_interface
tests\test_model.py::TestV011TrainingReadiness::test_data2vec_training_loop
  baselines\data2vec_baseline.py:121: UserWarning: data2vec target depth truncated: encoder exposes 2 layers, average_top_k_layers=8

================== 151 passed, 4 warnings in 26.28s ===================
```

**151 passed.** Lint and format:

```
> & $PY -m ruff check src/utils/flops.py tests/test_model.py
All checks passed!

> & $PY -m black --check src/utils/flops.py tests/test_model.py
All done! 2 files would be left unchanged.
```

---

## 5. MUTATION VERDICT

Three perturbations of the pinned quantities; all three turned the test red, and
all three were restored. The test pins a **ratio against an independent
measurement**, not a constant, so a mutation has to move the physics.

### Mutation 1 — drop the target-encoder forward

```diff
-    target_encoder = encoder  # same stack, forward only
+    target_encoder = 0.0  # MUTATION
```

```
__________________ TestV010NewFeatures.test_flops_estimation __________________
tests\test_model.py:1540: in test_flops_estimation
    assert estimate["target_encoder_flops"] > 0, "target encoder is 1x forward"
E   AssertionError: target encoder is 1x forward
E   assert 0.0 > 0
=========================== short test summary info ===========================
FAILED tests/test_model.py::TestV010NewFeatures::test_flops_estimation - Asse...
1 failed, 150 deselected in 1.98s
```

### Mutation 2 — drop the predictor's refine passes

```diff
-    block_tokens = num_refine_steps * seq_len + sum(seq_len - d for d in offsets)
+    block_tokens = seq_len + sum(seq_len - d for d in offsets)  # MUTATION
```

```
__________________ TestV010NewFeatures.test_flops_estimation __________________
tests\test_model.py:1547: in test_flops_estimation
    assert 0.99 <= ratio <= 1.01, f"seq_len={seq_len}: est/measured = {ratio:.5f}"
E   AssertionError: seq_len=16: est/measured = 0.91975
E   assert 0.99 <= 0.9197476595032957
=========================== short test summary info ===========================
FAILED tests/test_model.py::TestV010NewFeatures::test_flops_estimation - Asse...
1 failed, 150 deselected in 2.01s
```

### Mutation 3 — make the attention term linear in T instead of quadratic

This is the *original* defect's shape, reintroduced.

```diff
-    return 4.0 * batch * seq_len * seq_len * dim * n_layers
+    return 4.0 * batch * seq_len * dim * n_layers  # MUTATION
```

First attempt **failed to catch it** — the predictor's quadratic term masked
the encoder's. I fixed the test (below), then:

```
__________________ TestV010NewFeatures.test_flops_estimation __________________
tests\test_model.py:1586: in test_flops_estimation
    assert encoder_attention_at(64) == pytest.approx(4 * encoder_attention_at(32))
E   assert 98304.0 == 196608.0 - 0.196608
E     Obtained: 98304.0
E     Expected: 196608.0 - 0.196608
=========================== short test summary info ===========================
FAILED tests/test_model.py::TestV010NewFeatures::test_flops_estimation - Asse...
1 failed, 150 deselected in 2.36s
```

**Honest note:** the first version of the attention assertion was too weak and
mutation 3 survived it. I did not loosen anything — I added a second assertion
that pins the encoder's attention term on its own, where the `T²` law is exact.
Along the way two of my own expectations were wrong and the test caught *me*:
the combined attention ratio is not 4 (the future-offset passes scale as
`(T−d)²`), and it is 4.086 rather than just under 4. Both are now stated with
the reason, not a magic bound.

---

## 6. What the test now pins

`tests/test_model.py::TestV010NewFeatures::test_flops_estimation` — the same
test, strengthened (not deleted, not skipped, not xfailed):

- `est/measured ∈ [0.99, 1.01]` against `FlopCounterMode`, at T = 16/32/64;
- each of the four matmul groups is individually positive;
- the error **does not grow with T** (`ratios[-1] <= ratios[0] + 0.005`) —
  the exact property the old formula violated;
- the encoder's attention term is exactly `T²`;
- `estimate_training_flops` composes a per-step cost, not a parameter count.

Diagnostics are stubbed inside the test, with a comment naming the two modules
and saying why. That is scoping the measurement to the thing under test, not
weakening an assertion — the ratio bound still fails if the estimator drifts.

---

## 7. не_сделано / риски

1. **`src/utils/__init__.py` not updated.** `estimate_jepa_step_flops` is not
   re-exported from `src.utils`. Callers must import from `src.utils.flops`
   directly. That file is outside my card boundary; a one-line follow-up.
2. **`estimate_training_flops` signature changed** from
   `(num_params, seq_len, batch_size, num_steps)` to
   `(flops_per_step, num_steps)`. This removes the old `6·N·L·B` guess from a
   second location. Only the test called it, and I own that test — but it is a
   breaking change for any out-of-tree caller.
3. **The diagnostics term is not modelled.** ~1.5-2% of the step, and the
   multiplier is not constant across T (13.6×/15.3×/18.2× `2·B·T·D²`), so a
   fitted constant would rot. A caller wanting wall-clock-inclusive cost must
   add their own measured term. Documented in the docstring.
4. **The attention caveat is build-specific.** On CUDA, or on a torch build
   that registers a formula for the fused attention kernel, `FlopCounterMode`
   *would* count attention and the `total_flops`/`attention_flops` split would
   need revisiting. The docstring says the counter scores SDPA at 0 **on
   torch 2.13.0+cpu** rather than claiming it universally.
5. **The tolerance `[0.99, 1.01]` was measured, not derived.** It holds with
   ~0.02% headroom at the three T values tested. A future torch that counts an
   extra op would trip it — correctly, since that would be a real change, but
   the failure would need re-derivation.
6. **The small-config test dims are not the shipping dims.** The test uses
   64/3 with predictor 32/2 to stay inside the CPU budget; the base_140m
   numbers in this report come from the probe, not from the test.
7. **Scratch probes** (`.agent-notes/_probe_*.py`, 33 files) were used for the
   measurement work. `.agent-notes` is git-tracked in this repo, so they are
   left on disk deliberately as the evidence behind the report. Delete them if
   the board prefers a clean tree.
