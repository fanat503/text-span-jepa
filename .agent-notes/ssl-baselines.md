# SSL baselines — BYOL, Barlow Twins, VICReg, SimSiam

Branch `agent/ssl-baselines`, commit `77c20d5`, base `agent/wave-10` (`3b26ab4`).
Worktree `C:\dev\wt-ssl`. Not pushed. Never touched `src/train.py`,
`baselines/__init__.py`, the two existing baselines, or any `src/` file —
`git status` shows only five new files.

---

## 1. What was built

| File | Lines | Class |
|---|---|---|
| `baselines/byol_baseline.py` | 387 | `BYOLBaseline` |
| `baselines/barlow_baseline.py` | 304 | `BarlowTwinsBaseline` |
| `baselines/vicreg_baseline.py` | 348 | `VICRegBaseline` |
| `baselines/simsiam_baseline.py` | 277 | `SimSiamBaseline` |
| `tests/test_ssl_baselines.py` | 1304 | 67 tests |

All four match the real `create_model` contract, read from source rather than
guessed: ctor kwargs `(vocab_size, max_seq_len, embed_dim, depth, num_heads,
mlp_ratio, drop_rate)` + method kwargs, `get_num_params(non_embedding=False)`,
`get_num_params_trainable()`, `compute_loss(...) -> (loss, info)`.

---

## 2. Parameter matching — the requested ratio

`MLMBaseline` carries **1.281x** JEPA's trainable parameters because it holds an
untied `(embed_dim, vocab_size)` matrix. These four carry **0.993x**.

```
D=640 depth=10   TextSpanJEPA trainable    88,947,841   1.000x
                 BYOLBaseline trainable    88,312,320   0.9929x
                 BarlowTwinsBaseline       88,312,320   0.9929x
                 SimSiamBaseline           88,312,320   0.9929x
                 VICRegBaseline            88,322,560   0.9930x
                 MLMBaseline trainable    113,953,280   1.2811x   <- the defect

D=768 depth=12   TextSpanJEPA trainable   134,390,017   1.000x
                 BYOL / Barlow / SimSiam  133,519,872   0.9935x
                 VICRegBaseline           133,532,160   0.9936x
                 MLMBaseline trainable    162,716,160   1.2108x
```

**Achieved ratio: 0.9929x – 0.9936x, against MLM's 1.2811x.**

How the constant was derived — this mattered and the first guess was wrong.
None of the four has a vocabulary-sized head, so `HIDDEN_MULTIPLE` was *solved*
against JEPA's measured non-encoder trainable budget (7,189,121 at D=640):

- BYOL / VICReg / SimSiam: 2 MLPs of width `4*D`, cost `16*D²`
- Barlow Twins: 1 MLP of width `8*D`, cost `16*D²` — same budget, one head

My first guess of `3.0` gave **0.956x** at 640/10 and 0.958x at 768/12. It was
wrong and the header now carries the solved numbers, not the guess.

The ratio is asserted into a **band (0.92–1.08), not to equality**, and the
reason is in the code: JEPA's head budget includes a fixed-count GWP mechanism
term that does not scale with width, so exact equality is unreachable without a
vocab-sized head — which is the defect itself. A test asserting `ratio == 1.0`
would have been red on arrival and deleted rather than kept.

Also asserted as an **absence**, which a ratio cannot catch (a ratio can be met
by compensating errors): no parameter outside an `encoder`/`target_encoder` has
shape `(vocab_size, embed_dim)`.

---

## 3. Collapse tests — the core of the task

Each method prevents collapse through exactly one structure, and a port that
deletes it still trains and still reports a falling loss. Each test is paired
with a mutation that removes the mechanism.

| Arm | Mechanism | Correct | Mutated |
|---|---|---|---|
| Barlow Twins | `lambda_offdiag` (off-diagonal decorrelation) | eff-rank ≥ **0.175** | ≤ **0.085** |
| BYOL | EMA teacher (encoder + projector) | eff-rank ≥ **0.072** | ≤ **0.062** |
| SimSiam | `torch.no_grad()` stop-gradient | rep-std ≥ **0.439** | ≤ **0.039** |
| VICReg | variance hinge `var_weight` | hinge at collapsed batch = **24.75** | **0.00** |

### The measurement choice that made them work

My first collapse statistic was **standard deviation, and it did not work.**
Both `TextSpanJEPAEncoder` (final `LayerNorm`) and every projector
(`BatchNorm1d`) force non-zero per-dimension variance, so a std assertion is
satisfied by a *fully collapsed* model. Measured with std:

```
Barlow Twins   correct 0.3956   lambda=0  0.4934   <- ablation looks BETTER
VICReg         correct 0.3338   var=0     0.3603   <- ablation looks BETTER
```

Zero separation, wrong direction. Replaced with **normalised effective rank**
(participation ratio of the centred matrix's singular values), which detects
*dimensional* collapse — every dimension a copy of one signal. That gives 0.175
vs 0.085 for Barlow Twins.

### VICReg is tested where it is deterministic

Measured training runs did **not** separate VICReg's arms on encoder effective
rank (0.0402 correct vs 0.0493 ablated — wrong direction again, because at
batch 32 / dim 32 a short run never reaches the regime where the variance hinge
dominates). Rather than tune the seed until the numbers looked right, VICReg's
test asserts the property the term actually has:

> evaluated at a collapsed batch, the variance hinge returns its maximum.

Seed-independent, costs no training, and cannot be satisfied by a model that
merely hasn't collapsed yet.

---

## 4. Mutation verdicts

Each mutation applied to the **source**, run, then restored. Restores verified
byte-identical (`Compare-Object`: 0 differing lines for all four modules).

```
SIMSIAM   remove torch.no_grad() from target_branch
          -> 3 failed, 12 passed
             test_the_stop_gradient_is_structural_not_a_detach_call
             test_correct_arm_survives_and_its_mutation_collapses[0]
                 AssertionError: the correct SimSiam arm collapsed on seed 0 (rep-std 0.0388 <= 0.15)
             test_correct_arm_survives_and_its_mutation_collapses[1]
                 AssertionError: the correct SimSiam arm collapsed on seed 1 (rep-std 0.0321 <= 0.15)

BARLOW    DEFAULT_LAMBDA = 5e-3 -> 0.0
          -> 3 failed, 9 passed
             test_the_mechanism_is_the_off_diagonal_penalty
             test_correct_arm_survives_and_its_mutation_collapses[0]
             test_correct_arm_survives_and_its_mutation_collapses[1]
                 AssertionError: the correct Barlow Twins arm collapsed on seed 1 (eff-rank 0.0698 <= 0.13)

VICREG    DEFAULT_VAR_WEIGHT = 25.0 -> 0.0   (the classic port bug)
          -> 2 failed, 13 passed
             test_variance_term_has_a_positive_weight
                 AssertionError: assert 0.0 > 0
             test_the_total_loss_changes_when_the_variance_term_is_removed
                 AssertionError: removing the variance weight left the total loss unchanged

BYOL      target_branch reads self.encoder/self.projector (EMA teacher removed)
          -> 3 failed, 14 passed
             test_correct_arm_survives_and_its_mutation_collapses[0,1,2]
                 AssertionError: the correct BYOL arm collapsed on seed 2 (eff-rank 0.0337 <= 0.066)
```

---

## 5. Verify paste

```
$ & $PY tools\rt.py tests\test_ssl_baselines.py
rt.py: threads=1  total_budget=90s  slow_ok=False
collected 67 items

tests\test_ssl_baselines.py ............................................ [ 65%]
.......................                                                  [100%]

============================= 67 passed in 53.92s =============================

$ & $PY -m ruff check baselines/ tests/test_ssl_baselines.py
All checks passed!            (exit 0)

$ & $PY -m black --check <the five files>
5 files would be left unchanged   (exit 0)

$ & $PY tools\rt.py tests\test_baseline_parity.py      # pre-existing, unaffected
39 passed in 10.17s
```

Suite count: `pytest tests --collect-only -q` → **2058** on this tree
(1991 before this change; +67).

---

## 6. Defects found — four, and two were in my own tests

Reported because "tests passed" would be misleading without them.

1. **Real bug in `VICRegBaseline`.** `variance_loss` used `z.var(dim=0, unbiased=True)`
   unconditionally. The unbiased estimator is undefined at `B=1` and returned
   **NaN**, so the arm emitted a NaN loss on a one-sample batch. Fixed: the
   unbiased estimator is selected only when `B > 1`. This was caught by
   `test_batch_of_one_does_not_raise_in_eval[vicreg]`, not by inspection.

2. **Vacuous test of my own.** `test_the_total_loss_changes_when_the_variance_term_is_removed`
   compared `build("vicreg")` against `build("vicreg", var_weight=0.0)` — two
   *separately initialised* models — so their losses differed for the
   uninteresting reason that they started from different weights. **It passed
   with the variance term deleted from the source entirely.** Fixed to toggle
   the weight on one model; it then correctly went red under the mutation.

3. **Degenerate assertion of my own.** BYOL's EMA test read the first encoder
   parameter, whose gap to the teacher is exactly **0.0** before the first
   update (the teacher is a `deepcopy`), and asserted a strict decrease against
   it. Now perturbs the student first, so there is a gap for the EMA to close.

4. **Wrong harness assumption.** `train_and_measure` originally took an
   already-built model and seeded *afterwards*, so model initialisation came
   from whatever RNG state the previous test left behind — every arm was
   measured at a different init than the sweep that chose the thresholds had
   measured. Now takes a factory and seeds before construction.

---

## 7. Coverage gap — stated, not papered over

**BYOL's predictor is not asserted to be load-bearing at this scale.**
Removing it (`no-predictor`) does **not** collapse BYOL here — measured
eff-rank 0.1510 vs the correct arm's 0.0833 at 100 steps; the ablation looks
*better*. So BYOL's mutation test targets the EMA teacher, which does separate
cleanly, and the predictor is covered by structural assertions only
(`test_the_projector_is_part_of_the_teacher`, `test_the_target_branch_carries_no_gradient`).

Grill et al. show the predictor matters in the full ImageNet setting; that is
not reproducible at this scale on a shared 6-core box. Recorded in the module
docstring rather than implied away.

Two related notes on mutation design, both kept in the source because they are
the more instructive failures:

- **The obvious BYOL mutation is wrong.** Overriding `update_target_encoder` to
  do nothing leaves `target_encoder` frozen at *random initialisation* — a
  perfectly good teacher — and the "broken" arm then scores no worse than the
  correct one. The mutation that actually removes the mechanism is one where the
  target branch **reads the online weights**. A plausible mutation that does not
  test the claim is worse than none, because it produces a green test for a
  broken mechanism.
- **The four methods collapse under different conditions.** SimSiam's collapse
  was bimodal across seeds at batch 32 / augmentation 0.7 (mutant rep-std 0.055,
  0.713, 0.655, 0.062). It becomes deterministic at batch 16 / augmentation 0.85
  (0.039, 0.032, 0.029, 0.036). The reason is mechanical: collapse is available
  exactly when the two views are hard to distinguish. Each arm therefore uses
  the augmentation and batch at which *its own* mutation collapses — Barlow
  0.3, BYOL 0.8, SimSiam 0.85 @ batch 16.

---

## 8. Not wired into `create_model` — the one thing left undone

`src/train.py` is not owned by this change, so these four arms are importable
and constructible but **not buildable by name** today. That is a wiring gap, not
a code gap, and it is stated in every module header.

`TestTrainerContract` pins the contract they must satisfy when someone wires
them. The trap it guards is real and silent:

```python
if hasattr(model, "compute_loss_with_targets"):                     # JEPA
elif hasattr(model, "forward") and hasattr(model, "regression_head"):  # data2vec
elif hasattr(model, "compute_loss"):                                # MLM
```

Both earlier branches are tested **before** the `compute_loss` branch these
arms rely on. An arm that owned an attribute named `regression_head` — an
innocuous name, and the obvious name for a regression-style head — would be
routed to the data2vec call signature. The tests assert both absences.

To wire: add four `elif model_name == ...` branches to `create_model`, four to
`_normalize_model_name`, one to `do_ema_update` (BYOL only, zero-arg, matching
data2vec's existing signature), and export from `baselines/__init__.py`. Roughly
60 lines in files this change does not own.

---

## 9. Constraints honoured

- No training. `src.train` never invoked; only `tools/rt.py` on the one test file.
- 67 tests in **54s**, inside the 90s budget, `threads=1` (rt.py enforced).
- Tiny tensors: the largest model built for real work is `embed_dim=32, depth=2`.
  Production shapes are built on **meta tensors** (the `_meta_device` trick from
  `tests/test_baseline_parity.py`) — exact parameter counts, ~0 bytes allocated.
- Deterministic: `seed_everything` from `src/utils/seed.py`, fixed generators for
  data and augmentation, seeding before model construction.
- **No skip, no xfail, no `--no-verify`.**
- Lint and format clean at exit 0.
- Only the five owned files changed; four other workers' files untouched.
