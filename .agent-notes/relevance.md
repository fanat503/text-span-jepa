# `jawp_predictive_relevance` — decision, evidence, and what I did

Branch `agent/relevance`, worktree `C:\dev\wt-rel`, base `main` @ `e955968`,
commit **`2a1a398`**. Files changed: `src/models/jawp.py`, `src/models/wsd.py`,
`tests/test_jawp.py`. Nothing pushed.

---

## 1. The decision: **(a)**, with a correction to the card's premise

**I chose (a): the metric's arithmetic was wrong.** But the card's stated reason —
"the sign is inverted relative to every consumer" — **is not correct, and acting on
it would have been the wrong fix.** I verified this before changing anything.

The card's evidence is that the metric reads 1.0 when workspace prediction is
exact and 0.0 when it is worst. That is true. But it is also exactly what
**every consumer already expects**. I enumerated them:

| site | text | direction it requires |
|---|---|---|
| `proofs/wsd.md:30` | "collapses to **0.0000** — the workspace stops being predictable" | low = dead |
| `proofs/wsd.md:222` | "collapses to zero — the workspace stops being predictable at all" | low = dead |
| `proofs/wsd.md:263` | "reads 0.0 rather than raising" | low = dead |
| `proofs/jawp.md:264` | "a collapse of `jawk_predictive_relevance` to 0" | low = dead |
| `src/models/jawp.py:78` | "collapses to 0 and the workspace stops being predictable" | low = dead |
| `src/models/wsd.py:219` | "jawp_predictive_relevance collapses to 0" | low = dead |
| `tests/test_host_reads.py:524` | zero residual → **1.0** | perfect = high |

All seven want **low = dead**, and `max(0, 1 - …)` already falls as the workspace
gets worse. Option (b) — "the metric is right, the consumers are wrong" — therefore
has **no supporting evidence anywhere in the repo**, and the card's own escape
hatch applies: *"If you find that some consumers already expect the current sign,
THAT is the evidence — follow it."* They do. So (b) is out.

**What is actually broken is the SCALE, not the sign.**

```
ws_pred_error = ((Q^T r)**2).mean()   over (N, k)   -> divides by N*k
bg_pred_error = (r**2).mean()         over (N, D)   -> divides by N*D
ratio = (D/k) * w          where w = ||Q^T r||_F^2 / ||r||_F^2 in [0,1]
```

So the reported number was `max(0, 1 - (D/k)*w)`. Same direction, wrong scale.
Consequence: the zero threshold sits at `w = k/D` (chance level, **0.094** at
D=64, k=6) instead of at `w = 1`. **Any** workspace holding even a chance-level
share of the residual reads exactly `0.0`.

Measured, exact endpoints (`.agent-notes/verify_relevance.py`, D=64, k=6):

| constructed residual | realised w | `loss_workspace` | old metric | fixed metric |
|---|---|---|---|---|
| residual **entirely outside** `span(Q)` | 0.000000 | 6.831e-29 (exact) | 1.0000 | **1.0000** |
| w = 0.05 | 0.050000 | — | 0.4667 | 0.9500 |
| w = chance = 0.0938 | 0.093750 | — | **0.0000** | 0.9063 |
| w = 0.5 | 0.500000 | — | **0.0000** | 0.5000 |
| residual **entirely inside** `span(Q)` | 1.000000 | 9.963e+11 (worst) | 0.0000 | **0.0000** |

And the identity, at eight values of `w`:

```
max |metric - max(0, 1 - (D/k)·w)| = 1.110e-16      (D=64, k=6)
```

**This is why the cited collapse is uninformative, not inverted.** A reading of
0.0 was reachable at chance — it was already 0 before the workspace could die.
On a random-init workspace over 20 steps the old metric read `0.0000` on **20/20**
steps. That is the actual defect: an unfalsifiable diagnostic, not a reversed one.

---

## 2. Consumers reconciled

Reconciled **in my files**:

- **`src/models/jawp.py:75-95`** (module header, was line 78) — removed the
  "Above it, jawp_predictive_relevance collapses to 0 and the workspace stops
  being predictable" claim. Replaced with the reason it was never evidence, plus
  an explicit note that the antagonism numbers have **not** been re-measured.
- **`src/models/wsd.py:219`** (was in the `resync_target_workspace` comment) —
  same claim, same correction, same "not re-measured" caveat.
- **`src/models/jawp.py:470-508`** — the metric's own comment now states the
  definition, the axis requirement, the old bug, and why the old threshold was
  chance rather than the true worst case.

**Not reconciled — outside my file boundary, reported instead:**

- **`proofs/wsd.md:30, 212-223, 263`**, **`proofs/jawp.md:264`** — the `lambda_wsd`
  table showing `0.0000` at `lambda_wsd >= 1` was measured with the mis-scaled
  metric. **Those numbers are not re-derivable here**: re-measuring needs a
  training run, which the card forbids, and `proofs/**` is outside my ownership.
  They should be re-measured or struck by whoever owns that file. This is the one
  place where a number in the repo is now known-wrong and I could not fix it.
- **`docs/THEORY_AND_PROOFS.md` D-JAWP-1** — the finding is correct and now
  actionable; its "the meaning is inverted" framing is the part that is wrong.

---

## 3. The fix

```python
bg_pred_error = ((z_pred_flat - z_target_flat) ** 2).mean()
ws_pred_energy = ((pred_ws_d - target_ws_d) ** 2).sum()
# `z_pred_flat.numel()` is N*D; `bg_pred_error.numel()` would be 1,
# because a `.mean()` reduces to a 0-dim scalar.
ws_pred_error = ws_pred_energy / z_pred_flat.numel()
pred_rel_div = ws_pred_error / bg_pred_error
```

Both terms now sit on the same N·D axis, so the ratio is a dimensionless share and
the score spans the full [0, 1]. For orthonormal Q, Cauchy–Schwarz bounds `w <= 1`,
so `max(0.0, …)` is now only a guard for the non-orthonormal case.

The name `predictive_relevance` is **kept**. Option (b)'s "rename so it cannot be
misread again" is unnecessary: the name already means what the corrected metric
measures, and it is the *scale* that let it be misread.

**Scope — diagnostic only.** Verified, not assumed (`.agent-notes/verify_scope.py`):
`predictive_relevance` is the only info key that moved (24 other reads identical
across 3 configurations); `total_loss`, `‖dL/dz_pred‖` and `‖dL/dQ‖` are unchanged;
the metric sits inside the existing `torch.no_grad()` block, built after
`total_loss`. `tests/test_sterility.py` passes.

---

## 4. Verify paste

Required set — **169 passed, 1 xfailed**:

```
$ & $PY tools\rt.py tests/test_jawp.py tests/test_wsr.py tests/test_wsd.py tests/test_training_state_guards.py
rt.py: threads=1  total_budget=90s  slow_ok=False
tests\test_jawp.py ......................................... [100%]
======================= 169 passed, 1 xfailed, 7 warnings in 10.40s =======================
```

The `xfail` is the pre-existing `mode='sam'` retraction xfail in
`test_training_state_guards.py`, present in the baseline too. Baseline before my
change was 161 passed; the delta is my 8 new tests.

Lint/format: `ruff check src/ tests/` → **All checks passed**;
`black --check` on the three files → **3 files would be left unchanged**.

### Red before green

Written first, run against unfixed code — 3 of 8 red, **for the normalisation,
not the sign**:

```
FAILED test_chance_level_share_is_not_scored_as_zero
  a chance-level share (0.0938) must not read as a dead workspace, got 0.0
FAILED test_score_is_the_workspace_share_not_a_dimension_ratio
  must not depend on D/k; it read {'D=64,k=6': 0.0, 'D=64,k=32': 0.5, 'D=128,k=4': 0.0}
FAILED test_score_is_invariant_to_the_overall_error_scale
```

The other 5 passed pre-fix — which is itself the evidence that the sign was already
right and only the scale was wrong.

### MUTATION VERDICT

"Green" is not evidence. Two mutations, both reverted and re-verified:

**A — reintroduce the inversion** (`ws_pred_energy / pred_ws_d.numel()`, restoring
the N·k mean). **2 tests red:**

```
FAILED test_chance_level_share_is_not_scored_as_zero        (got 0.0)
FAILED test_score_is_the_workspace_share_not_a_dimension_ratio
```

**B — flip the sign** (`max(0.0, pred_rel_div_v)` instead of `max(0.0, 1.0 - …)`).
**5 tests red**, including both direction endpoints:

```
FAILED test_direction_is_high_when_the_workspace_prediction_is_exact
FAILED test_direction_is_zero_when_the_workspace_holds_all_the_error
FAILED test_score_falls_as_the_workspace_residual_grows
FAILED test_chance_level_share_is_not_scored_as_zero
FAILED test_score_tracks_the_reported_workspace_loss
  score must track the loss: 0.05 !> 0.95
```

Both mutations restored; `jawp.py` byte-identical to the fixed version
(`read(...) == read(...)` → `True`) and the required set re-run green.

---

## 5. Same-family and secondary findings — **reported, not fixed**

### 5a. Same defect, different symptom: `compute_background_complexity`

`jawp.py:840` and `:846` have **the same axis bug**. `ws_residual` is a mean over
(N, k) and `bg_residual` over (N, D), so the reported ratio carries a `k/D` factor
against its docstring's "Ratio of background residual to workspace residual".
Measured: `reported / same-axis = 1.00000` at every share I tested, i.e. the
reported number is **already** the ratio of the two stored residuals — the axis
mismatch is *between the two stored quantities and the docstring's claim*, not
inside the reported ratio. **Lower severity than the metric fixed here**: it is a
ratio, so a constant factor preserves ordering and only mis-scales the magnitude.
Reported per the card's instruction not to silently fix the family.

### 5b. Dead parameter: `min_gap_ratio`

`detect_workspace_dimension(z_pred, z_target, min_gap_ratio=2.0)` — the parameter
appears **twice** in `jawp.py`, both in the signature and the docstring, and is
**never read in the body**. Measured: `min_gap_ratio=1.01` → `k*=1`;
`min_gap_ratio=1000` → `k*=1`; default → `k*=1`. A caller cannot distinguish it
from a constant. Exactly the card's "a parameter you cannot distinguish from a
constant" — a different bug class from the inverted scale, so reported, not fixed.

### 5c. WSD tautologies — confirmed, not fixed

Both verified (`src/models/wsd.py:318-330`), **inside my file list but left alone
deliberately**, because each fix is an owner science call:

- **`wsd_steady_state_error ≡ wsd_running_drift`** exactly.
  `nu_max_est = running_drift / T`, `lambda_wsd = 1/T`, so `nu/lambda = running_drift`
  and **T cancels**. Measured: `max |steady_state_error - running_drift| = 2.776e-17`
  over 12 steps (D=16, k=3, T=7). The "constructive bound" is a second name for the
  drift EMA and bounds nothing.
- **`wsd_lambda_estimate` is `1/sync_interval`, not `lambda_wsd`.** Measured:
  T=7 → 0.142857; T=50 → 0.020000; T=100 → 0.010000. It matches the shipped
  `lambda_wsd: 0.01` **only** at the default interval, so the agreement is a
  coincidence that silently breaks on any config that moves either value — e.g.
  `config/ablations/wsd_on.yaml` at `lambda_wsd: 0.1`.

I did not "fix" these to look plausible. Making `wsd_lambda_estimate` honest means
reading the real `lambda_wsd` from config, which is a wiring change to a mechanism
whose gradient path is itself contested (D-WSR-3 / the `mechanisms.py:475` `.data`
detachment). Escalate, don't improvise.

---

## 6. Риски / risks

1. **A known-wrong number survives in `proofs/wsd.md:212-223`** — the `lambda_wsd`
   table's `0.0000` entries were produced by the mis-scaled metric and I could not
   re-measure them (no training; `proofs/**` out of bounds). **This is the highest
   residual risk**: the table looks like measured evidence and no longer is. It
   should be re-run or struck before that table reaches a paper.
2. **`tests/test_host_reads.py` now fails 2 tests — outside my boundary.**
   `_ref_jawp_info` (`:386-387`) is a transcription of the pre-change arithmetic
   used as an oracle for "the batched read returns what the old code returned". My
   change intentionally moves that one number, so the oracle must move with it.
   **Exact patch, verified to make the module agree to 1e-12** (`.agent-notes/verify_oracle.py`):
   ```python
   -  ws = ((p - t) ** 2).mean()
   +  ws = ((p - t) ** 2).sum() / zpf.numel()
   ```
   `bg` and the `max(0.0, 1.0 - …)` host policy are unchanged; `:524` (zero residual
   → 1.0) is unaffected and still passes. I did **not** edit it — it is not in my
   ownership list, and weakening an oracle to go green is exactly what I must not do.
   **Current state: 2 failing tests on this branch until its owner applies that one line.**
3. **The corrected metric changes no log threshold anyone has agreed on.** The value
   moves a lot (0.257 → 0.988 on one `test_host_reads` case), so any plot or table
   built on the old numbers needs re-baselining, not rescaling.
4. **`1 - w` still cannot distinguish "workspace perfect" from "workspace empty".**
   Both read 1.0 when the residual is orthogonal to `span(Q)`. That is inherent to a
   residual-share metric, not something this fix introduces, and it is why
   `workspace_utilization` / `target_ws_fraction` remain worth reading alongside it.
5. **Provenance hazard on this machine:** the editable install maps `src` to
   `C:\Users\<u>\mp\clone-check-1`, so a script run from `.agent-notes/` resolves to
   a *different clone* than pytest run from the repo root. All verification scripts
   `sys.path.insert(0, REPO)` and assert `src.__file__` starts at the worktree. A
   verification run that skipped that would have measured the wrong tree.
6. **Not fixed, in the same file, by design:** `min_gap_ratio` dead (§5b) and the
   `compute_background_complexity` axis (§5a). Both are recorded above rather than
   folded into this commit, since neither is the reported defect.