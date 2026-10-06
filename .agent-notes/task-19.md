# TASK-19 — stop `CollapseDiagnostics.compute` from costing half the forward pass

**status: done (partially — the part I could not wire is specified, not made)**
branch `agent/task-19b`, worktree `C:\dev\wt-19`. Never pushed.

---

## 1. What I actually changed, and why it is not the frequency gate

The card asks for a gate. I could not place one: the gate lives at a call site
(`src/models/jepa.py:889,897`) and the frequency lives in `src/train.py:1221`,
and both files are outside my boundary (`jepa.py` is the documented collision
hotspot; `train.py` was explicitly withheld).

So I shipped the half of the win that **is** reachable from
`src/models/collapse.py`, and it is the half that needs no knob and no
staleness:

**The 19 decompositions per step were 19 decompositions of the same two
matrices.** Every SVD-bearing metric decomposed the activation matrix itself.
Measured by wrapping `torch.linalg.*` (probe below), one `compute()` call at
B=8/T=256/D=384 did:

| call | before | after | why it was duplicated |
|---|---|---|---|
| `svdvals` | **13** | 3 | 7 metrics × 2 tensors = 12, +1 on the small (k,k) CCA matrix |
| `svd` | **4** | 2 | `_svcca` and `_subspace_overlap` each decomposed both tensors |
| `matrix_rank` | **2** | 0 | an SVD in disguise, on a spectrum we already had |
| `eigvalsh` | **4** | 2 | `_eigenvalue_spread` and `_spectral_clustering_coeff` built the *identical* covariance |
| **SVD-bearing total** | **19** | **5** | |

`_SharedSpectral` (new, module-private) computes each distinct decomposition
once per `compute()` and hands it to the metrics through new keyword arguments
(`svals=`, `svd_x=`/`svd_y=`, `eigvals=`). Every helper still computes its own
decomposition when not given one, so the direct callers
(`src/interp/run_comparison.py:278-279` reaches into `_svcca`/`_subspace_overlap`;
the tests call the helpers individually) are unaffected.

**This is bit-identical.** Not "close", not "within tolerance" — for every one of
the 44 metrics except the seven that the RNG fix deliberately changes (§4), the
value is bitwise equal to the old code, verified against a snapshot of the
pre-change file. The `_numerical_rank` rewrite reproduces
`torch.linalg.matrix_rank`'s own tolerance rule (`rtol` scales the **largest**
singular value) and is pinned against torch on 11 shapes including the ones
where the plausible-but-wrong rule (scaling by `S[-1]`) silently disagrees.

### The second, mandatory half: the RNG

The card's caution is correct and slightly understated — it is **four** sites,
not two. `torch.randperm(N, device=flat.device)` at what were `collapse.py:518,
581, 663, 815` consumed the **training device's global generator**, the same one
DropPath draws from. So `compute()` was steering the model it was observing, and
gating it would have silently moved the training trajectory while looking like a
pure optimisation. All four now draw from a private `torch.Generator`, using the
pattern `src/models/cmc.py:153-199` already established (`_MASK_RNG_SALT`,
`_derive_mask_rng`), extracted to `_subsample_index`.

### Measured share, before → after

`collapse.py` + `jspace.py` + `tests/test_collapse_dedup.py`. Median of 3,
1 thread, xsmall dims (`embed_dim=384`, `encoder_depth=6`), B=8, T=256, fp32,
`model.eval()`, `torch.no_grad()`:

| | before | after | Δ |
|---|---|---|---|
| `CollapseDiagnostics.compute` | 2.1641 s | **1.6562 s** | **−23.5 %** |
| `JSpaceMetrics.compute` | 0.0351 s | 0.0335 s | −4.6 % |
| both | 2.2457 s | **1.7131 s** | −23.7 % |
| forward (`compute_loss_with_targets`) | 4.3833 s | **3.8086 s** | **−13.1 %** |
| **diagnostics share of forward** | **51.2 %** | **45.0 %** | −6.2 pts |
| forward, diagnostics stubbed out | 2.1592 s | 2.1224 s | (−1.7 %, i.e. noise — the control) |

The stubbed-forward control is unchanged, which is what says the 13 % is the
dedup and not the box drifting. The card's 67 % was measured at 4 threads; at 1
thread I measure 51 %. I am reporting my own number, not the card's.

**What the recommended gate would add on top** (simulated in the probe, not
wired — §3): mean forward **2.2901 s/step at `log_freq=10`, −40.4 %**, i.e. the
diagnostics amortise to 0.156 s/step instead of 1.740 s/step.

### `src/models/jspace.py` — unchanged, deliberately

No edit. Measured: 2 `torch.linalg.eigh` on (384,384), 0.0335 s, **1.9 %** of
the diagnostics' cost and 0.9 % of the forward. There is no redundant work in it
to remove — each `eigh` is on a distinct matrix — and it draws no randomness, so
the RNG defect does not exist there. It does have one structural inefficiency I
could not reach (§6, R2): it recomputes the covariance `eigvalsh` that
`collapse.py` already computed for the same two tensors. Fixing that needs a
shared object across two module instances, which means changing `jepa.py`.

---

## 2. Consumer table — who reads what, and at what cadence

**There is no validation consumer. There is no consumer outside the training
loop.** `CollapseDiagnostics.compute` has exactly one production caller:
`jepa.py:889`. Its 44-key dict, plus 3 keys added at `jepa.py:894,895,905` and
`workspace_quality` at `jepa.py:910`, become `diag_dict`, and `diag_dict` is
read at **six places, all inside one `if` block**:

| # | reader | file:line | keys | actual cadence it needs |
|---|---|---|---|---|
| 1 | `logger.info` "diag:" line | `train.py:1529-1533` | `effective_rank_online`, `collapsed_dim_ratio_online`, `mask_fraction`, `target_center_norm`, `workspace_quality` | every `log_freq` steps (10) |
| 2 | `csv_logger.log` | `train.py:1565-1567` | `effective_rank_online`, `collapsed_dim_ratio_online`, `mask_fraction` | every `log_freq` steps |
| 3-6 | — | — | — | none |

Both readers sit inside `if itr % log_freq == 0 or np.isnan(loss_val) or
np.isinf(loss_val):` (`train.py:1510`). So **44 metrics are computed every step
to feed 5 printed values and 3 CSV columns, every 10th step.** 37 keys are
computed for nobody at all: no reader in `src/`, no reader in `tests/`, no
plotted consumer. `src/utils/visualization.py:1664`
(`plot_workspace_quality_components`) and `:194` (`plot_svcca_curve`) take the
keys as *arguments* from the caller — they read nothing that flows from this call
site.

**Separate consumers of the same class, off the hot path.** 8 modules construct
their **own** `CollapseDiagnostics()` and call `compute()` directly:
`src/interp/ablation.py:641`, `src/interp/compare.py:88,139`,
`src/interp/stability.py:47,239,312`, `src/interp/layer_analysis.py:360,394`,
`src/interp/run_comparison.py:208`, `src/interp/robustness.py:80`,
`src/eval/probes.py:399`, `src/interp/interpretability_index.py:171`. All are
offline analysis on a fixed set, all want all 44 keys, all are unaffected by a
frequency gate — **and all are why the gate must not live inside `compute()`**
with a non-default default. Their `compute()` keeps every key today.

### The stale-value trap the card was right to warn about

Gating naively is a correctness bug in three places, all outside my files:

1. **`workspace_quality` would become a number built from nothing.** It is a
   weighted blend of 10 `diag_dict` entries (`collapse.py:1034-1052`), each with a
   `default`. Empty dict in → it still returns a plausible `0.x` out of pure
   Python arithmetic. `train.py:1533` prints it. **It must be inside the gate.**
2. **The CSV has no staleness channel.** `train.py:1565-1566` writes
   `diag_dict.get("effective_rank_online", 0)`. On a gated step that writes
   `0.0` into a column whose header promises a rank. The block is entered on
   **NaN/inf steps even when `itr % log_freq != 0`** (`train.py:1510`), so this
   fires on exactly the emergency step a human reads first. **Carry the last
   computed value forward, or add an age column.** This is the one that would
   have shipped unnoticed.
3. **`representation_stability` silently changes meaning.** It compares against
   `self._prev_target_h`, which `jepa.py:912` refreshes *unconditionally*, outside
   the gate — so it would go from a 1-step difference to a `log_freq`-step one
   while its docstring still says "consecutive". Same for `jspace_stability`,
   which compares against `_prev_jspace_vectors` refreshed *inside*
   `jspace.py:101` — gate it and it spans `log_freq` steps.

---

## 3. The exact change I RECOMMEND but did NOT make

### 3a. The config key — one line, `defaults.yaml`, `model:` block

```yaml
  diagnostics_log_freq: 0   # 0 = every step (today's behaviour); N = every Nth
```

Named and placed to match the mechanism knobs that already work this way and are
already read in `jepa.py:139-163` (`wsd_sync_interval`, `cmc_mode`/`cmc_interval`,
`sta_update_interval`) and gated in `jepa.py` by a `should_compute(step)` method
(`cmc.py:297-313`). `0` is the default so that **landing the key alone changes
no run's numbers** and no one has to edit the 62 configs that deep-merge onto
`defaults.yaml`.

### 3b. `src/models/jepa.py` — read it, and gate three lines

Next to the other interval reads (`jepa.py:139`, after `self.sta_update_interval`
at `:163`):

```python
        self.diagnostics_log_freq = kwargs.get("diagnostics_log_freq", 0)
        self.diagnostics_always = self.diagnostics_log_freq <= 1
```

`compute_loss_with_targets` **already receives `current_step`** (`train.py:1316`
passes `current_step=global_step`), so `compute_loss` (`train.py:546-554`) needs
**no change** — the frequency reaches the model without a new plumbing argument.
Then wrap `jepa.py:889-902` in:

```python
        want_diag = self.diagnostics_always or (
            self.diagnostics_log_freq > 0 and current_step % self.diagnostics_log_freq == 0
        )
        if want_diag:
            diag_dict = self.diagnostics.compute(
                h_online.detach(), h_target.detach(), prev_target_h=self._prev_target_h,
            )
            jspace_dict = self.jspace_metrics.compute(h_online.detach(), h_target.detach())
            diag_dict.update(jspace_dict)
            diag_dict["workspace_quality"] = CollapseDiagnostics.workspace_quality(diag_dict)
```

**Leave `jepa.py:894-895` and `:905` outside the gate.** `target_center_norm` and
`mask_fraction` are printed at `train.py:1531-1532` and are one reduction each —
seed B's "keep the cheap closed-form metrics always on". They do not depend on any
decomposition.

### 3c. `src/train.py` — the two lines that make it honest

1. **`train.py:1565-1566`** — do not write a default into a column that has a
   header. Carry the last computed value, and label its age:

```python
_last_diag = {}          # next to csv_logger, train.py:1230
...
_last_diag.update(diag_dict)
csv_logger.log(..., _last_diag.get("effective_rank_online", float("nan")),
                  _last_diag.get("collapsed_dim_ratio_online", float("nan")), ...)
```

A blank cell beats a `0.0` that reads as "the model has rank zero".
2. **`train.py:1760`** — `total_loss, _, _ = compute_loss(...)`. **Validation
   computes all 44 metrics for every validation batch and throws them away.**
   At the default `max_batches` that is the single largest remaining waste of
   this exact kind. Recommend `compute_loss(..., compute_diagnostics=False)` and a
   `jepa.py` branch that skips the block. Free 40 % with no frequency knob at all.

I did not make 3a, 3b or 3c. `config/**`, `defaults.yaml`, `src/train.py` and
`src/models/jepa.py` are all outside `files_allowed`.

### 3d. If you would rather have the knob in my file

A 4-line `should_compute(self, step)` on `CollapseDiagnostics`, lifted verbatim
from `cmc.py:297-313`, would be the natural home. **I did not add it** because
nothing would call it: an unread method is dead code, and this repo's own stated
position (`src/utils/seed.py:44-52`) is that faking an unread knob is worse than
a missing one because it converts a loud "this did nothing" into a silent no-op.
It goes in with 3b, not before.

---

## 4. Does the cheap gate the card hinted at already exist? No.

> *"check whether `jspace.py` has a deterministic/eval-mode path, because a
> diagnostic that only matters for validation should not run during training
> steps at all."*

I checked, and it does not exist:

```
grep -n "training|eval|deterministic" src/models/jspace.py   -> No matches found
grep -n "training|eval|deterministic" src/models/collapse.py -> No matches found
```

Neither module reads `self.training`, and neither has a deterministic or eval
branch. There is no gate to use, in these files or anywhere else in `src/models/`
— the only step-frequency gates in the repo are `cmc.should_compute`,
`wsd.sync_interval` (`wsd.py:238`) and `sta.update_interval` (`sta.py:294`).

Stronger than the hint, and it changes the recommendation: these diagnostics do
not matter for validation either. **Nothing outside the training loop reads
them at all** (§2). Validation computes them and discards them (`train.py:1760`).
So the framing "should not run during *training* steps" understates it — the
correct statement is that `train.py` computes 44 metrics it reads 5 of, on the
steps it reads them, and that the frequency gate is the fix. The eval-mode idea
would have been the better fix only if a validation reader existed; there is none.

---

## 5. VERIFY — full paste

### 5a. New test file — 24 tests, 1.8 s

```
rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests\test_collapse_dedup.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 24 items

tests\test_collapse_dedup.py ........................                    [100%]

============================= 24 passed in 1.79s ==============================
```

### 5b. Card's verify line: `tests/test_interp.py`

```
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 94 items

tests\test_interp.py ...................................................  [ 54%]
...........................................                              [100%]

============================= 94 passed in 4.75s =============================
```

### 5c. Card's CPU line: `--slow test_model.py test_sterility.py`

```
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 187 items

tests\test_model.py ....................................................  [ 27%]
........................................................................ [ 66%]
...............................                                          [ 82%]
tests\test_sterility.py ................................                 [100%]

============================== warnings summary ===============================
tests/test_model.py::TestData2VecBaseline::test_forward
tests/test_model.py::TestData2VecBaseline::test_loss_formula_data2vec
tests/test_model.py::TestV011TrainingReadiness::test_compute_loss_unified_interface
tests/test_model.py::TestV011TrainingReadiness::test_data2vec_training_loop
  baselines\data2vec_baseline.py:121: UserWarning: data2vec target depth truncated: encoder exposes 2 layers, average_top_k_layers=8
    warnings.warn(

====================== 187 passed, 4 warnings in 19.89s =======================
```

### 5d. Everything else that touches the diagnostics or builds a model — 439 passed

```
collected 239 items
tests\test_interp.py ................................................... [ 21%]
tests\test_collapse_dedup.py ........................                    [ 49%]
tests\test_index_and_cka.py ............................................ [ 67%]
tests\test_sigreg_jspace.py ..............                               [ 78%]
tests\test_determinism.py ...............                                 [ 84%]
tests\test_seed.py .....................................                 [100%]
============================= 239 passed in 25.66s ==============================
```
```
collected 45 items      (--slow tests/test_v025_integration.py)
tests\test_v025_integration.py .........................................  [ 91%]
....                                                                     [100%]
============================= 45 passed in 2.72s ==============================
```
```
collected 151 items
tests\test_mechanism_wiring.py ......                                    [  3%]
tests\test_checkpoint_fidelity.py ....................                   [ 17%]
tests\test_target_centering_state.py ..........                          [ 23%]
tests\test_jawp.py .....................................................  [ 58%]
tests\test_cgn.py ..........................                             [ 78%]
tests\test_pcr.py ................................                       [100%]
============================ 151 passed, 6 warnings in 12.31s ========================
```

### 5e. Lint / format

```
$PY -m ruff check src\models\collapse.py src\models\jspace.py tests\test_collapse_dedup.py
All checks passed!

$PY -m black --check  (same three files)
3 files would be left unchanged.
```

### 5f. Probe — inventory, timing, equivalence (`python .agent-notes/_probe_task19.py`)

```
src guard OK -> C:\dev\wt-19
torch 2.13.0+cpu | threads 1

=== 1. LAPACK calls per compute(), N=B*T=2048, D=384 ===
  CollapseDiagnostics.compute
      svdvals      x3
      svd          x2
      eigvalsh     x2
      cholesky     x2
      inv          x2
      -> SVD-bearing calls (svdvals+svd+matrix_rank): 5
         decomposed svd on (2048, 384)
         decomposed svdvals on (374, 374)
         decomposed svdvals on (2048, 384)
      -> 44 metrics returned
  JSpaceMetrics.compute
      eigh         x2
      -> 9 metrics returned

=== 2. wall clock, median of 3, 1 thread ===
  CollapseDiagnostics.compute    1.6792s   raw ['1.6769', '1.6988', '1.6792']
  JSpaceMetrics.compute          0.0334s   raw ['0.0334', '0.0342', '0.0334']
  both                           1.7397s   raw ['1.7448', '1.7397', '1.7379']
  forward (xsmall dims)          3.8412s   raw ['3.8729', '3.8412', '3.8177']
  -> diagnostics share of forward = 45.3%
  forward, diagnostics stubbed   2.1345s   raw ['2.1153', '2.1565', '2.1345']
  -> with diagnostics gated off, forward is 1.80x faster

=== 2b. recommended gate (itr % log_freq == 0), simulated ===
  mean forward over 10 steps, 1 computing: 2.2901s/step
  -> 40.4% off the mean forward at log_freq=10
  -> diagnostics amortised to 0.1556s/step (from 1.7397s/step ungated)

=== 3. deduplicated vs original, all metric values ===
  keys only in original: []
  keys only in new     : []
  compared 44 keys, 7 differ
  differing keys: ['alignment', 'intrinsic_dim_online', 'intrinsic_dim_target',
                   'mean_pairwise_cosine_online', 'mean_pairwise_cosine_target',
                   'uniformity_online', 'uniformity_target']
  UNEXPECTED differences (not a subsampling metric): none
  max |diff| = 8.801e+00 on 'intrinsic_dim_online'
  N=128 rows (<= _SUBSAMPLE_LIMIT, nothing subsamples): 0 differ []
  global CPU RNG state unchanged by compute(): True
  the draw AFTER compute() is identical to the draw without it: True
  jspace: 9 keys, max |diff| = 0.000e+00 on ''
```

Two lines there are the whole correctness argument:
* `N=128 rows ...: 0 differ` — below the subsample cap nothing is resampled, so
  the 37 non-RNG metrics come out **bitwise** identical to the old code.
* `the draw AFTER compute() is identical to the draw without it: True` — the
  observable consequence of the RNG fix: the next random number a consumer sees
  does not depend on whether a diagnostic ran.

---

## 6. MUTATION VERDICT — both properties forced back, tests went red, restored

Two separate mutations, because this card made two separable claims.

### M1 — force every metric to decompose for itself (`_SharedSpectral._once` → `return None`)

```
tests\test_collapse_dedup.py FF......................                    [100%]

__________ TestDecompositionBudget.test_one_svd_per_distinct_matrix ___________
tests\test_collapse_dedup.py:98: in test_one_svd_per_distinct_matrix
    assert counts.get("svdvals", 0) <= 3, (
E   AssertionError: 13 svdvals calls for two activation matrices; the svdvals-based metrics are not sharing a decomposition
E   assert 13 <= 3
E    +  where 13 = <built-in method get of dict object ...>('svdvals', 0)
E    +    +  where ... = {'svdvals': 13, 'matrix_rank': 2, 'svd': 4}.get
__________ TestDecompositionBudget.test_eigvalsh_is_not_repeated_per_metric ___________
tests\test_collapse_dedup.py:118: in test_eigvalsh_is_not_repeated_per_metric
E   AssertionError: 4 eigvalsh calls for two covariance matrices
E   assert 4 <= 2
======================== 2 failed, 22 passed in 1.85s =========================
```

The mutant reports **13 `svdvals`, 4 `svd`, 2 `matrix_rank`, 4 `eigvalsh`** —
the pre-card numbers, 19 SVD-bearing calls. The budget test is red and names the
count. Restored → 24 passed.

### M2 — force the subsample back onto the training device's global RNG

```
tests\test_collapse_dedup.py ..........FFFFFF........                    [100%]

FAILED ...test_compute_leaves_the_global_cpu_state_untouched
E   assert False
E    +  where False = <built-in method equal of type object ...>(tensor([146,  16,   0,  ...]), tensor([146,  16,   0,  ...]))
FAILED ...test_the_draw_after_compute_is_the_draw_without_it
E   assert False
E    +  where False = ...(tensor([0.1871, 0.9937, ...]), tensor([0.9444, 0.2083, ...]))
FAILED ...test_each_subsampling_helper_leaves_the_global_state_untouched[_intrinsic_dim_score-<lambda>]
E   AssertionError: _intrinsic_dim_score consumed the global RNG
FAILED ...test_each_subsampling_helper_leaves_the_global_state_untouched[_mean_pairwise_cosine-<lambda>]
E   AssertionError: _mean_pairwise_cosine consumed the global RNG
FAILED ...test_each_subsampling_helper_leaves_the_global_state_untouched[_uniformity-<lambda>]
E   AssertionError: _uniformity consumed the global RNG
FAILED ...test_each_subsampling_helper_leaves_the_global_state_untouched[_alignment-<lambda>]
E   AssertionError: _alignment consumed the global RNG
======================== 6 failed, 18 passed in 1.80s =========================
```

Six red, and the parameterised one names **each of the four offending helpers**,
not just "the RNG moved". Restored → 24 passed. Both mutations are gone:
`grep -rn MUTATION .` → no matches.

No skip, no xfail, no `--no-verify`, no test weakened or deleted anywhere in
this branch.

---

## 7. diff-stat

```
 src/models/collapse.py       | 417 ++++++++++++++++++++++++++++++-------
 tests/test_collapse_dedup.py | 454 +++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 797 insertions(+), 74 deletions(-)
```

`git diff --cached --numstat`: `343  74  src/models/collapse.py` and
`454  0  tests/test_collapse_dedup.py`. `src/models/jspace.py`: 0 changes.

`git diff --stat` on tracked files, plus one new file. `.agent-notes/_probe_*.py`
(the two probes and the two baseline snapshots) are already covered by
`.gitignore:73`.

---

## 8. не_сделано / риски

**не сделано (deliberately, with the change specified in §3)**

1. **The frequency gate itself.** No key in `defaults.yaml`, no gate in
   `jepa.py`, no change in `train.py`. §3 has the exact key, the exact lines,
   and the three things that must move with it or it becomes a silent lie.
2. **Validation still pays for 44 metrics it discards** (`train.py:1760` →
   `total_loss, _, _`). This is the largest unclaimed win in this class and it
   is entirely in `train.py`/`jepa.py`.
3. **`jspace.py` cross-module duplicate.** It recomputes the covariance
   `eigvalsh` that `collapse.py` already did for the same two tensors. Closing it
   means one shared object across two module instances, i.e. a `jepa.py` change.
   Worth ~2 % of the forward. Left for whoever holds that file.
4. **`should_compute` on `CollapseDiagnostics`.** Not added — nothing would call
   it. See §3d for why that is the right call in this repo.
5. **`docs/campaign/perf.md` does not exist** (`docs/` contains only `plans/`),
   and creating it is outside `files_allowed`. §1 and §5f carry the numbers; the
   integrator can paste them.

**риски**

1. **Seven metric values change, by design.** `intrinsic_dim_{online,target}`,
   `mean_pairwise_cosine_{online,target}`, `uniformity_{online,target}`,
   `alignment` — a different random 256-row sample. Same estimator, same
   distribution, different draw. `alignment` and `uniformity_*` feed
   `workspace_quality`, so that composite moves too. **Any previously published
   number for those seven is void**; the other 37 are bitwise unchanged.
   `test_compute_is_bit_identical_when_nothing_subsamples` pins that split.
2. **The private subsample RNG is not resume-exact.** `_subsample_draw_counter`
   is process state, so a resume rewinds it. Same trade `cmc.py:185-199`
   documents, and it is the right one here — these are diagnostics, not
   checkpoint state. But it is a real difference from `seed_everything`'s promise
   and it is why `test_seeded_run_is_reproducible` has to rewind the counter
   explicitly.
3. **Validation now advances the private counter.** `validate()` restores the
   global RNG (`train.py:1745,1770`) but cannot restore a Python int, so the
   training-step subsample indices now depend on how many validation batches ran.
   Before my change validation's effect was fully unwound. It is deterministic
   and affects only these seven values, but it is a new coupling. One-line fix
   for whoever owns `train.py`: snapshot/restore
   `src.models.collapse._subsample_draw_counter` in `validate` alongside
   `_snapshot_training_buffers`, which is the same pattern already there at
   `train.py:1744,1769`.
4. **The dedup relies on LAPACK determinism** for a fixed shape, dtype and
   thread count. On a different BLAS/thread count both old and new code move
   together, so this is no worse than before — but "bit-identical" is a claim
   about *this* box, and `src/utils/seed.py:62-64` already documents that CPU
   reproducibility is per-thread-count.
5. **Unmeasured on CUDA.** The host is CPU-only. The dedup is device-agnostic
   (fewer LAPACK calls, same ops) and should help more on GPU where the SVD
   launch cost is worse — but I did not measure it and will not claim it.
6. **Peak memory is unchanged.** The dedup frees the intermediate `U`/`Vh`/`S`
   tensors sooner but computes them anyway; only the repeated copies are gone. No
   claim either way.