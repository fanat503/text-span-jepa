# TASK-12 — one seeded split per `layer_analysis` probe, not one per layer

- **status**: done
- **branch**: `agent/task-12`, commit `bdbbf5e`. Not pushed. Working tree clean.
- **worktree**: `C:\dev\wt-12` (created from `agent/wave-2` @ `d34a345`).
  `C:\dev\text-span-jepa` was never edited.

- **files** (only these two; nothing else in `src/interp/**` touched —
  `ground_truth.py` and `statistical_tests.py` were not opened for edit):
  - `src/interp/layer_analysis.py` (owned) +222/-21
  - `tests/test_layer_analysis.py` (new, 485 lines, 35 tests)

---

## verify

### BEFORE — the card's claim, reproduced in this worktree

`N=80, D=16, L=12`, all 12 layers **byte-identical**, one fixed separable
dataset. "Varying only the global seed" is the only knob the shipped code has,
because `LayerwiseProbe` has no `seed` parameter at all
(`has a seed parameter on LayerwiseProbe: False`).

```
$ $PY t12_measure.py
importing src from: C:\dev\wt-12\src\__init__.py
N=80 D=16 L=12  (all 12 layers byte-identical)

layer_uniformity on BYTE-IDENTICAL layers, varying only the global seed:
  global_seed=0  layer_uniformity=0.7220  peak_layer=3   n_distinct_accuracies=6
  global_seed=1  layer_uniformity=0.6972  peak_layer=11  n_distinct_accuracies=6
  global_seed=2  layer_uniformity=0.7784  peak_layer=4   n_distinct_accuracies=5
  global_seed=3  layer_uniformity=0.7776  peak_layer=5   n_distinct_accuracies=7
  global_seed=4  layer_uniformity=0.8014  peak_layer=3   n_distinct_accuracies=6
  --> band = 0.697 .. 0.801  (width 0.104)

per_layer_accuracy on identical input (12 identical layers):
   [0.625, 0.5, 0.375, 0.875, 0.625, 0.312, 0.688, 0.5, 0.5, 0.5, 0.5, 0.375]
  distinct values: 6 of 12

global-RNG consumption: churn the global stream between two calls
  before churn: 0.725189
  after  churn: 0.792130
  identical: False

how many distinct split draws happen for 12 layers:
  torch.randperm calls inside probe_all_layers: 12
  distinct permutations among them:          12

compare_layer_profiles: distinct split draws across BOTH arms (12 + 12 layers):
  torch.randperm calls: 24  distinct: 24

routing_score distinct split draws (1 baseline + 3 LOO, 1 task):
  torch.randperm calls: 4  distinct: 4

layer_uniformity direction (report only):
  6 byte-identical layers          layer_uniformity = 0.664 .. 0.856
  6 independent random layers      layer_uniformity = 0.808 .. 0.909

has a seed parameter on LayerwiseProbe: False
```

The audit's `0.882–0.983` band reproduced in kind as **`0.697–0.801`, width
0.104** on identical data. Twelve identical layers produced twelve different
splits, and `compare_layer_profiles` drew **24 distinct** permutations for a
comparison that is a *difference* of two accuracies.

### The measurement that set the scope

The card names the split. Before writing the fix I checked whether the split
is actually the dominant channel, by freezing one at a time:

```
$ $PY t12_decompose.py
importing src from: C:\dev\wt-12\src\__init__.py
as shipped (split redraw + unseeded init)        0.6972 .. 0.8014  width=0.1042
split frozen, init still redrawn (global)        0.6722 .. 0.8299  width=0.1577
split frozen AND probe init frozen               1.0000 .. 1.0000  width=0.0000
```

**Freezing the split alone does not close the band** — it widens it, because a
single frozen draw is just one more arbitrary partition, while the second
unseeded per-layer draw (`nn.Linear`'s weight init, drawn from the global RNG
because `nn.Linear` takes no `generator`) keeps moving. Only closing both
gives width 0.

So the fix closes both. That is beyond the card's literal text and I am
flagging it as a deliberate scope decision, not quietly widening the card: the
card's own symptom is a *seed-to-seed noise band*, and a fix that shares the
split but leaves the init ambient does not remove it.

### AFTER — same script, same data, same global seeds

```
$ $PY t12_measure.py
importing src from: C:\dev\wt-12\src\__init__.py
N=80 D=16 L=12  (all 12 layers byte-identical)

layer_uniformity on BYTE-IDENTICAL layers, varying only the global seed:
  global_seed=0  layer_uniformity=0.6928  peak_layer=0  n_distinct_accuracies=7
  global_seed=1  layer_uniformity=0.6928  peak_layer=0  n_distinct_accuracies=7
  global_seed=2  layer_uniformity=0.6928  peak_layer=0  n_distinct_accuracies=7
  global_seed=3  layer_uniformity=0.6928  peak_layer=0  n_distinct_accuracies=7
  global_seed=4  layer_uniformity=0.6928  peak_layer=0  n_distinct_accuracies=7
  --> band = 0.693 .. 0.693  (width 0.000)

global-RNG consumption: churn the global stream between two calls
  before churn: 0.692754
  after  churn: 0.692754
  identical: True

how many distinct split draws happen for 12 layers:
  torch.randperm calls inside probe_all_layers: 1
  distinct permutations among them:          1

compare_layer_profiles: distinct split draws across BOTH arms (12 + 12 layers):
  torch.randperm calls: 2  distinct: 1

routing_score distinct split draws (1 baseline + 3 LOO, 1 task):
  torch.randperm calls: 1  distinct: 1

layer_uniformity direction (report only):
  6 byte-identical layers          layer_uniformity = 0.592 .. 0.592
  6 independent random layers      layer_uniformity = 0.813 .. 0.813

has a seed parameter on LayerwiseProbe: True
```

### side by side

| quantity | before | after |
|---|---|---|
| seed-to-seed band on **byte-identical** layers | 0.697–0.801 (**0.104**) | **0.693** exactly (**0.000**) |
| distinct `randperm` permutations, 12 layers | **12** | **1** |
| distinct permutations, `compare_layer_profiles` (24 probes) | **24** | **1** (2 calls, same split) |
| distinct permutations, `routing_score` (4 probes) | **4** | **1** |
| global-RNG churn between two identical calls | moves the result | **no effect** |

Note the direction rows: before, both the identical and the random set were
*ranges* that overlapped, so the direction problem below was unmeasurable
through the noise. Now each is a single number and the gap is clean.

### tests

```
$ $PY tools\rt.py tests\test_layer_analysis.py
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests\test_layer_analysis.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 35 items

tests\test_layer_analysis.py ...................................         [100%]

============================= 35 passed in 1.97s =============================
```

**The card's gate**, `tests/test_interp.py`, plus the new file:

```
$ $PY tools\rt.py tests\test_interp.py tests\test_layer_analysis.py
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests\test_interp.py tests\test_layer_analysis.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 129 items

tests\test_interp.py ................................................... [ 39%]
...........................................                              [ 72%]
tests\test_layer_analysis.py ...................................         [100%]

============================= 129 passed in 4.63s =============================
```

The other interp-touching suites, to confirm no collateral:

```
$ $PY tools\rt.py tests\test_index_and_cka.py tests\test_causal_intervention.py tests\test_feature_composition.py tests\test_run_comparison.py tests\test_layer_analysis.py
rt.py: threads=1  total_budget=90s  slow_ok=False
collected 164 items
tests\test_index_and_cka.py ............................................ [ 26%]
...........                                                              [ 33%]
tests\test_causal_intervention.py ...................................... [ 56%]
........                                                                 [ 61%]
tests\test_feature_composition.py ............                           [ 68%]
tests\test_run_comparison.py ................                            [ 78%]
tests\test_layer_analysis.py ...................................         [100%]
============================= 164 passed in 5.23s =============================
```

### lint / format

```
$ $PY -m ruff check . --output-format concise
All checks passed!
ruff exit: 0

$ $PY -m black --check .
All done! ✨ 🍰 ✨
110 files would be left unchanged.
```

No test was weakened, deleted, commented out, skipped or xfailed. The suite
grew from 94 to 129 in this file pair. No training was run.

### an environment hazard worth recording

`pip install -e .` on this box maps the `src` package to
`C:\Users\<user>\tmp\clone-check-1\src`, **not** to any worktree
(`__editable___text_span_jepa_1_0_0rc15_finder.py`, `MAPPING = {'src':
'C:\\Users\\…\\tmp\\clone-check-1\\src'}`). A standalone
`python <script>.py` run therefore silently measures a stale clone unless the
script prepends the worktree to `sys.path`. I hit this on my first
measurement run and every pasted number above is from a corrected script that
prints `importing src from: C:\dev\wt-12\src\__init__.py` on its first line.

`tools/rt.py` is **not** affected: it runs pytest with `cwd=REPO`, and
`tests/conftest.py` puts the rootdir first. I verified this explicitly with a
throwaway test asserting `r"C:\dev\wt-12" in layer_analysis.__file__` — passed.
So the test evidence in this report is sound; only ad-hoc scripts are at risk.
**TASK-23 and TASK-24 owners should know this.**

---

## diff-stat

```
$ git diff --stat agent/wave-2...HEAD
 src/interp/layer_analysis.py | 243 ++++++++++++++++++++--
 tests/test_layer_analysis.py | 485 +++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 707 insertions(+), 21 deletions(-)
```

---

## mutation-verdict

**The defect is detected, and the pre-existing suite is blind to it.** Two
mutations were applied to the fixed source and reverted; the test file was
never touched. An earlier attempt at Mutation A produced a *spurious* red
(a `NameError` because the mutation left the new provenance keys behind); it was
discarded and re-run faithfully against the original `probe_all_layers` return
dict, and only that second run is reported.

### Mutation A — the card's mutation: the per-layer unseeded split, restored verbatim

```python
n_train = int(0.8 * N)
idx = torch.randperm(N)                                   # global RNG, per layer
train_idx, val_idx = idx[:n_train], idx[n_train:]
```
plus `probe_all_layers` and `routing_score` reverted to not draw or pass a
split.

```
$ $PY tools\rt.py tests\test_layer_analysis.py
collected 35 items
tests\test_layer_analysis.py .....................F........................... [100%]
...
FAILED tests/test_layer_analysis.py::TestOneSplitPerCall::test_every_layer_receives_a_split
FAILED tests/test_layer_analysis.py::TestOneSplitPerCall::test_all_layers_receive_the_very_same_split
FAILED tests/test_layer_analysis.py::TestOneSplitPerCall::test_the_shared_split_is_the_one_that_is_reported
FAILED tests/test_layer_analysis.py::TestOneSplitPerCall::test_split_is_absent_from_the_payload_unless_requested
FAILED tests/test_layer_analysis.py::TestOneSplitPerCall::test_compare_layer_profiles_uses_one_split_for_both_arms
FAILED tests/test_layer_analysis.py::TestOneSplitPerCall::test_routing_score_uses_one_split_for_baseline_and_every_loo_arm
FAILED tests/test_layer_analysis.py::TestSeededAndPrivate::test_probe_all_layers_does_not_consume_the_global_rng
FAILED tests/test_layer_analysis.py::TestSeededAndPrivate::test_probe_all_layers_is_reproducible
FAILED tests/test_layer_analysis.py::TestSeededAndPrivate::test_routing_score_is_reproducible_for_a_fixed_seed
FAILED tests/test_layer_analysis.py::TestProfileIsNotAmbientNoise::test_reruns_of_the_same_seed_collapse_to_zero_width
FAILED tests/test_layer_analysis.py::TestSplitHygiene::test_a_split_for_the_wrong_row_count_is_refused
FAILED tests/test_layer_analysis.py::TestSplitHygiene::test_an_empty_side_is_refused
======================== 12 failed, 23 passed in 2.06s ========================
```

The two `TestSplitHygiene` reds are the loudest: `DID NOT RAISE ValueError`.
The original code accepts a split drawn for a different dataset and an empty
validation side without complaint, because it never looks at the split it is
given — it does not have one.

**And the card's own gate, run with Mutation A still applied:**

```
$ $PY tools\rt.py tests\test_interp.py
collected 94 items
tests\test_interp.py ................................................... [ 54%]
...........................................                              [100%]
============================= 94 passed in 4.25s =============================
```

**94 passed with the defect fully restored.** `tests/test_interp.py:802
test_probe_all_layers` asserts `"per_layer_accuracy" in result` and
`result["peak_layer"] is not None`, and nothing else. This is the real
justification for the new file, and it is the same finding as audit I18
("`test_interp.py` pins almost nothing").

**Honest negative.** One test that *looks* load-bearing is not:
`test_the_seed_moves_the_profile_and_nothing_else_does` stayed **green**
under Mutation A. With the seed ignored, the two "different seed" calls still
differed — from ambient RNG, not from the seed. It only has power in
combination with `test_probe_all_layers_is_reproducible`. I left it (the
property "the seed must matter" is worth asserting) but it is not
load-bearing on its own and I am not claiming it is.

### Mutation B — the second unseeded per-layer draw: stock `nn.Linear`

`self.seeded_linear(...)` → `nn.Linear(self.embed_dim, num_classes)`, i.e. the
weight init back on the global RNG while the split stays shared and seeded.
This is the mutation the card does not ask for, and it is the one the
decomposition above says matters most.

```
$ $PY tools\rt.py tests\test_layer_analysis.py
FAILED tests/test_layer_analysis.py::TestSeededAndPrivate::test_probe_all_layers_does_not_consume_the_global_rng
FAILED tests/test_layer_analysis.py::TestSeededAndPrivate::test_probe_all_layers_is_reproducible
FAILED tests/test_layer_analysis.py::TestSeededAndPrivate::test_routing_score_is_reproducible_for_a_fixed_seed
FAILED tests/test_layer_analysis.py::TestProfileIsNotAmbientNoise::test_reruns_of_the_same_seed_collapse_to_zero_width
======================== 4 failed, 31 passed in 1.95s =========================
```

with the band visible in the message:

```
E   AssertionError: [0.8888888888888888, 0.7788916806429733, 0.8981146583783013, 0.8232233047033631]
E   assert (0.8981146583783013 - 0.7788916806429733) == 0.0
```

That is the 0.12 band the card describes, reproduced with the split perfectly
shared — which is the evidence that closing only the split would have left the
card's symptom in place.

`test_seeded_linear_matches_nn_linear_reset_parameters` and
`test_seeded_linear_never_touches_the_global_rng` stayed green under B, which
is correct: B removes the *call site*, not the helper, and those two test the
helper directly. The four behavioural tests are what carry the mutation.

### Restored

```
$ $PY -m ruff check . --output-format concise
All checks passed!
$ $PY -m black --check .
All done! ✨ 🍰 ✨
110 files would be left unchanged.

$ $PY tools\rt.py tests\test_interp.py tests\test_layer_analysis.py
collected 129 items
tests\test_interp.py ................................................... [ 39%]
...........................................                              [ 72%]
tests\test_layer_analysis.py ...................................         [100%]
============================= 129 passed in 4.63s =============================
```

---

## the fix, and the three decisions inside it

**1. One split, drawn once, from a private generator.**
`LayerwiseProbe.train_val_split(N, seed, train_frac)` is a public staticmethod
returning `(train_idx, val_idx)` from `torch.Generator().manual_seed(seed)`.
`probe_all_layers` draws it once and threads it into every
`_train_linear_probe(reps, labels, split=split, generator=generator)`.

**2. `src/utils/seed.py` deliberately NOT imported — the card is right and
TASK-13's reasoning generalises.** `seed_everything` seeds `torch.manual_seed`
on the *process-global* stream, which is precisely the thing this defect is
about not touching; importing it here would make a library module reseed the
caller's process. The established `src/interp/` convention is a local
generator: `causal_scrubbing.py:121`, `causal_intervention.py:312/615`,
`information_theory.py:169`, `disentanglement.py:222`,
`feature_composition.py:327`. The card also says `ground_truth.py:74` does
exactly the global reseed and is why it is forbidden here — confirmed by grep.
Following the convention also keeps audit finding **I16** from being
*worsened* by making a library module reach for a global seed.

**3. `seeded_linear` reimplements `nn.Linear.reset_parameters` from the
private generator, and that equivalence is asserted bit-for-bit.** `nn.Linear`
takes no `generator`, so the alternative was `torch.random.fork_rng` (in-repo
precedent at `interpretability_index.py:202`) — but that reseeds the global
stream, which is the thing the card forbids. The vendored law is torch's own:
`kaiming_uniform_(a=sqrt(5))` is exactly `uniform(-1/sqrt(fan_in),
1/sqrt(fan_in))`, plus `uniform_` for the bias, in that order, and the draws
are bit-identical:

```
torch w [-0.1564660668373108, 0.24684089422225952, 0.12306633591651917, 0.10329106450080872]
vend  w [-0.1564660668373108, 0.24684089422225952, 0.12306633591651917, 0.10329106450080872]
weight bit-equal: True
bias   bit-equal: True
```

`TestSeededProbeInit::test_seeded_linear_matches_nn_linear_reset_parameters`
pins that equality rather than trusting it, so a torch release that changes the
init law turns the suite red instead of silently drifting the module to a
different initialisation distribution.

`WEIGHT_STREAM_OFFSET = 1` keeps the split stream and the weight stream
independent under one `seed` — the split is a property of the data, the weights
are not. The same generator is threaded through the layer loop, so each layer
still trains its own independently initialised probe: that is the module's
pre-existing estimand and I did not change it (see R2).

---

## reported, NOT fixed

**R1 — `layer_uniformity = 1 - std/mean` rewards degeneracy. The card's
point, re-measured here, and now measurable.** With the noise gone the defect
is exact rather than buried: 6 byte-identical layers score **0.592**, 6
independent random layers score **0.813**. A model whose layers are literally
copies of each other scores *worse* on "uniformity" than one whose layers are
independent noise, and the module cannot tell the two apart. The reported
`jepa_more_uniform = jepa_uniformity > baseline_uniformity` in
`compare_layer_profiles` therefore rewards *heterogeneity*, which is the
opposite of the claim in the class docstring ("JEPA layers have MORE UNIFORM
probe accuracy"). I pinned the **formula** and labelled the direction in the
source, and added `test_uniformity_falls_as_the_profile_spreads` so the
direction problem is an executable fact rather than a comment nobody reads. I
did not change the metric: that is a research decision, and the card says so.

**R2 — paired probe inits across layers is the next noise channel, and I left
it open deliberately.** With the split shared and the weights seeded but
*independent per layer*, byte-identical layers still return 7 distinct
accuracies out of 12. Sharing one initialisation across layers (common random
numbers on the init) would collapse that to 1, and would make
"identical layers → identical profile" a testable sanity property. I did not do
it: it changes what the numbers mean rather than making them reproducible, so
it belongs with R1 as one decision, not inside a defect fix. It is pinned from
the other side by `test_identical_layers_still_differ_per_layer`, whose docstring
says so, so whoever flips it gets a deliberate red.

**R3 — `best_acc` is a max over epochs of the only held-out set.** `_train_linear_probe`
returns the best validation accuracy over up to `max_epochs` epochs and there
is no test set, so the number is optimistically biased by construction and
biased by an amount that depends on the split and the learning rate. The card
names this in passing but the fix it asks for is the split; a test set is
TASK-11's territory and `src/interp/layer_analysis.py` is not in that card's
file list. Reported.

**R4 — `self.num_classes` is dead.** The constructor takes
`num_classes=2` and `_train_linear_probe:200` immediately overwrites it with
`max(int(labels.max()) + 1, 2)`. A caller passing `num_classes=5` silently gets
a 2-class probe. Untouched — it is not a split defect, and honouring the
parameter would change behaviour for existing callers.

**R5 — the split is a plain permutation, not stratified.** One shared draw now
determines all 12 measurements, so an unlucky draw (a rare class entirely in
the held-out 20 %) is a common failure for the whole profile instead of 12
independent ones. That is a real trade: it removes noise and adds a shared
failure mode. Stratifying by label would remove it but changes the estimator,
so it is a decision, not a fix. The `n_train`/`n_val`/`split_seed` keys and
`return_split=True` are there so a report can show which draw it used.

---

## не_сделано

1. **`TASKS.md` not marked `done`**, and audit finding I15
   (`docs/plans/2026-09-27-wave1-audit-findings-interp.md:288`) not struck
   through. Shared campaign state; the coordinator owns the bookkeeping. Note
   the card text also says `tests/test_layer_analysis.py` is new — TASK-33
   (the duplicate) already claimed that path, so the two cards agree and there
   is no file collision, but whoever merges should fold TASK-33 into TASK-12
   rather than run it.
2. **`src/interp/__init__.py` not updated** (owned elsewhere this tick). The
   new public names `LayerwiseProbe.train_val_split`,
   `LayerwiseProbe.seeded_linear` and the module constants
   `WEIGHT_STREAM_OFFSET` / `DEFAULT_TRAIN_FRAC` are reachable through the
   class; nothing is unreachable. If the flat namespace is wanted, the owner
   must add it.
3. **R1–R5 above are reported, not fixed**, by instruction and by judgement.
4. **No `run_comparison.py` / `ablation.py` integration.** `git grep` finds
   `LayerwiseProbe` and `probe_all_layers` used nowhere outside this module and
   `tests/test_interp.py`, so `compare_layer_profiles` and `routing_score` have
   **no production caller** — the numbers this card de-noised are not yet on a
   published path. If TASK-16 wires this module into the one-command pipeline,
   `probe_all_layers(..., return_split=True)` is what a report should call.
5. The measurement scripts live outside the repo
   (`%LOCALAPPDATA%\Temp\opencode\t12_*.py`), so the before/after numbers are
   reproducible but not checked in. I did not promote them into a slow-marked
   test: `rt.py` has a hard 90 s cumulative budget and the box is shared.

## риски

1. **Every previously published `layer_uniformity` / routing number from this
   module is unreproducible and now known to be wrong.** They were computed on
   12–24 different partitions. Anyone holding a layer profile from before this
   commit should recompute it; the numbers will not match, and that is the
   point.
2. **`routing_score` now raises `ValueError` where it used to raise
   `IndexError`** when a task's label vector length disagrees with the layers.
   Clearer, but it is a behaviour change: any caller catching `IndexError`
   around it now misses the error. No caller in the tree does
   (`git grep routing_score` → one test of mine, one call site inside the
   module).
3. **Splitting on `layer_representations[0].size(0)` assumes every layer has
   the same row count.** That was already true (the old code indexed every
   layer with one permutation) but it is now load-bearing earlier, and
   `_check_split` will raise a `ValueError` naming the mismatch rather than
   silently scoring fewer rows. `_check_split` is O(1) deliberately — it
   validates sizes and bounds, not duplicate rows, because a full permutation
   check is `O(N log N)` per layer and would dominate a 12-layer profile over
   100k rows. Duplicated rows cannot come out of `train_val_split`; a caller
   that hand-builds a split owns it, and the docstring says so.
4. **The `seeded_linear` reimplementation is a maintenance edge.** It is
   bit-identical to `nn.Linear.reset_parameters` at torch 2.13.0+cpu and pinned
   by test, so a future torch change surfaces as a red test rather than silent
   drift — but it is one more place to re-pin after a torch upgrade.
5. **Adding `split_seed`, `n_train`, `n_val` to the returned dict is a payload
   change.** No consumer reads them yet and no test asserts an exact key set,
   but a strict schema check added later would need to know about them. The
   index vectors themselves are behind `return_split=True` and are *not* in the
   default payload, so a JSON report does not grow by `2N` integers.
6. **`default seed=0` is a fixed constant, not derived from a run seed.** The
   card says "seeded from the run seed", and there is no run seed in
   `src/interp/` — nothing in the package has one, and `src/utils/seed.py` is
   never imported here (see decision 2). So the seed is a *parameter* that a
   caller must plumb, and today no caller exists (R-not-done #4). If
   `run_comparison.py` is wired up (TASK-16), the run's seed must be threaded
   in or every report will be `seed=0`. This is the same decision TASK-23 is
   making across the package, and it should be made once, not twice.
