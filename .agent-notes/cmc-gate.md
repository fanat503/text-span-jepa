# CMC overlap gate — report

**Worktree** `C:\dev\wt-cmc` · **branch** `agent/cmc-gate` · **base** `7f46e63` (`main`)
**Provenance asserted before every run:** `src` → `C:\dev\wt-cmc\src\__init__.py`
**No training.** `src.train` never entered. All tests via `tools/rt.py`.

Files touched (exactly the four I own):
`src/models/cmc.py`, `defaults.yaml`, `tests/test_cmc.py`, `tests/test_config_system.py`.

---

## 1. Verdict: **(c), with the premise corrected. Not (a), not (b).**

The card's premise is **false**, and so are the first two options. The
arithmetic:

> **The gate is not `|Ω|/(B·T)`.** `src/models/cmc.py:444` compares
> `min_overlap_ratio` against `|Ω| / |{t : ‖z_pred_primary[t]‖ > 0}|` —
> the overlap as a fraction of the **primary mask** `|M1|`. The key the audit
> read, `cmc_overlap_ratio`, is `|Ω|/(B·T)` — the overlap as a fraction of
> **all positions**. One name, two denominators, depending on the branch.

For two independent masks at fraction `p`, positions iid:

| quantity | value | at the shipped ramp (0.15 → 0.35) |
|---|---|---|
| `E\|Ω\|` | `B·T·p²` | — |
| `\|M1\|` | `B·T·p` | — |
| **gate ratio** `\|Ω\|/\|M1\|` | **`p`** | **0.18 → 0.38** |
| logged ratio `\|Ω\|/(B·T)` | `p²` | 0.02 → 0.13 |

The threshold `0.2` is a test on the **second mask's ratio**, not on `p²`.
It is reachable: the ramp tops out at `mask_ratio_end = 0.35 > 0.2`, measured
`0.369` against `0.2`.

### Measured — real `SpanMaskCollator` primary, real `generate_second_mask` secondary
B=16, T=128, 25 draws, `span_length_range=[3,10]`:

| p | \|M1\|/BT | \|Ω\|/BT | gate = \|Ω\|/\|M1\| | shut |
|---|---|---|---|---|
| 0.15 | 0.1484 | 0.0267 | 0.1796 | 18/25 |
| 0.20 | 0.1953 | 0.0445 | 0.2279 | 3/25 |
| 0.25 | 0.2500 | 0.0688 | 0.2750 | 0/25 |
| 0.30 | 0.2969 | 0.0934 | 0.3146 | 0/25 |
| 0.35 | 0.3438 | 0.1267 | 0.3687 | 0/25 |

With 25 % padding: gate 0.1842 / 0.2369 / 0.2987 / 0.3264 / 0.3698,
shut 18, 6, 0, 0, 0 out of 25.

The gate ratio tracks `p` to within ~5 %, exactly as the arithmetic says. The
auditor's `0.047 … 0.164` is the `|Ω|/(B·T)` column, reproduced here as
0.0267 … 0.1267.

### Through the real bridge — `compute_cmc_between_passes`, real `defaults.yaml`
B=8, T=64, 8 draws:

| p | ran | mean `cmc_loss` |
|---|---|---|
| 0.15 | 2/8 | 6.22e-05 |
| 0.20 | 7/8 | 8.56e-05 |
| 0.25 | 7/8 | 8.81e-05 |
| 0.30 | 7/8 | 9.59e-05 |
| 0.35 | 8/8 | 9.55e-05 |

**CMC runs at the shipped defaults with a nonzero loss for most of the
curriculum.** "Skips 6 of 6" and "CMC does not run at the defaults" are both
false.

### The counterfactual that *would* be structurally shut
If the gate compared `|Ω|/(B·T)` against 0.2 — the auditor's reading — it needs
`p ≈ 0.45`:

| p | \|Ω\|/(B·T) | shut |
|---|---|---|
| 0.15 | 0.0245 | 20/20 |
| 0.25 | 0.0658 | 20/20 |
| 0.35 | 0.1278 | 20/20 |
| 0.45 | 0.2086 | 0/20 |

So (a) really is unreachable-by-tuning *as a reading of the wrong quantity* —
which is precisely the point: **that reading is not the gate.**

### Why not (a)
Lowering `min_overlap_ratio` buys only the 0.15 → 0.20 leg and weakens the
only guard there is. What the threshold protects is the opposite end: a
`second_mask_ratio` set far below the primary's, where `|Ω|` collapses to a
few positions and the `1/|Ω|` mean becomes a one-sample estimate. At
`|M1| ≈ 11.5k` (shipped `batch_size 64 × max_seq_len 512 × 0.35`), 0.2 already
demands `|Ω| ≳ 2.3k` — a generous floor, not a strict one.

### Why not (b)
The 0.15 → 0.35 ramp is the one deliberate mask knob, documented at
`defaults.yaml:123-133` and pinned by
`TestOnlyOneMaskRatioKnobIsLive`. Moving it changes the mask schedule of every
other mechanism and every published arm. That is a science decision, not this
card's.

### What (c) gets right, and what it gets wrong
Right remedy: **make it visible, not make it run.**
Wrong premise: CMC's value is *not* confined to configurations nobody ships.

The honest statement: **CMC is live above `p ≈ 0.25` — roughly the second half
of the curriculum — and shut below `p ≈ 0.20`, with a coin-flip band between.**
Measured at 16×128 over 60 draws per point: 41/60 skipped at 0.15, 11/60 at
0.20, 0/60 at 0.25/0.30/0.35. That is a real fact about the shipped
configuration that was previously recorded backwards.

---

## 2. The consequential defect (the one actually fixed)

**`cmc_overlap_ratio` reported a different quantity on each branch**, so the one
key that could tell a reader whether CMC was running was unusable — and it is
what produced the false finding. Measured, 4×32, predictions nonzero at 20/32,
threshold 0.2:

| branch | `cmc_skipped` | `cmc_overlap_ratio` | \|Ω\|/(B·T) | \|Ω\|/\|prim\| | keys |
|---|---|---|---|---|---|
| no-overlap | True | 0.0000 | 0.0000 | 0.0000 | **6** |
| insufficient | True | 0.1500 | 0.0938 | 0.1500 | **7** |
| ran | False | 0.3750 | 0.3750 | 0.6000 | **7** |

Three branches, **three key sets**, and one key meaning two things. The two
forms differ by `|M1|/(B·T) ≈ p` — a factor of 2–3 here. Any reader comparing
either against 0.2 draws a wrong conclusion. `src/train.py:1627` additionally
overwrites `cmc_overlap_ratio` with the `|Ω|/(B·T)` form while
`jepa.py:912` writes the module's own value under `cmc_cmc_overlap_ratio`, so
the log carries both forms adjacent under near-identical names.

This is `docs/THEORY_AND_PROOFS.md` **D-CMC-3** (already filed, correct) escalated
in `docs/METRICS.md` into a false mechanism-level claim. It is the same defect
class the card names — a mechanism that can no-op invisibly — but the no-op was
in the *reporting*, not in the *loss*.

## 3. End state

**`src/models/cmc.py`**
- `forward` restructured to **one return**, so the key set cannot drift.
- `cmc_overlap_ratio` = `|Ω|/(B·T)` on **every** branch (one meaning; also what
  `train.py:1627` writes, so module and host now agree).
- **`cmc_gate_ratio`** added — the quantity actually compared to
  `min_overlap_ratio` — present on **every** branch.
- **`cmc_gate_open`** (bool) and **`cmc_skip_reason`** (`""` when it ran) added,
  on every branch. A skip is now legible and a silent no-op is impossible.
- Removed two computed-and-discarded statements (`:427`, `:442`) — the exact
  lines D-CMC-3 names as "the true ratio is computed and discarded".
- `float(loss.detach())` fixes a `requires_grad`→scalar UserWarning.
- **No behavioural change** to the loss or the gate's firing rule. Keys: 7 → 10.

**`defaults.yaml`** — measurement table and the arithmetic next to
`cmc_min_overlap_ratio`, stating that it is compared against ≈`p` not `p²`,
which ramp leg is live, and that the two diagnostics must not be merged again.
**No value changed.**

**`tests/test_cmc.py`** (+13) — `TestCMCRunsAtTheShippedDefaults`:
- `test_the_gate_is_open_across_the_upper_shipped_ramp` — 40 draws × p ∈
  {0.25, 0.30, 0.35}: CMC runs on **every** draw and `loss > 0`. *The card's
  minimum.*
- `test_cmc_loss_is_nonzero_through_the_real_bridge` — end to end through
  `compute_cmc_between_passes`, the path `train.py` actually takes.
- `test_the_loss_carries_gradient_to_the_secondary_pass` — nonzero loss ⇒
  nonzero gradient. A detached constant reads live and trains nothing.
- `test_the_low_leg_skips_by_reporting_not_by_going_silent` — at p ∈ {0.15,
  0.20} both outcomes must occur, every skip must carry a reason, and no
  third state ("ran but didn't say so") may exist.
- `test_constructor_default_matches_the_shipped_config` — the dataclass
  default and `defaults.yaml` are two declarations of one number.

`TestCMCOverlapDiagnostics` (+5): both ratios pinned **on all three branches**
(including the skip branches, where the regression lived), the two proven
distinct, `cmc_gate_open` proven to be the threshold test, and the key set
proven equal across branches.

**`tests/test_config_system.py`** (+9) —
`TestCMCOverlapGateIsReachableAtTheShippedRamp`:
- `test_the_shipped_threshold_is_reached_inside_the_shipped_ramp`
- `test_the_gate_opens_across_the_upper_half_of_the_ramp`
- `test_the_square_of_the_mask_ratio_cannot_reach_the_threshold` — pins the
  auditor's arithmetic as a **fact**, so it cannot be re-derived as a reading
- `test_the_two_ratios_are_distinct_shipped_keys`

## 4. Verify paste

```
$PY -c "import sys;sys.path.insert(0,'.');import src;print('TREE:',src.__file__)"
TREE: C:\dev\wt-cmc\.\src\__init__.py

$PY tools\rt.py tests\test_cmc.py tests\test_config_system.py --slow tests\test_sterility.py
collected 723 items
tests\test_cmc.py ...........................................            [  5%]
tests\test_config_system.py ............................................ [ 12%]
...
tests\test_sterility.py ................................                 [100%]
================= 690 passed, 33 skipped, 1 warning in 28.95s =================

$PY tools\rt.py tests\test_mechanism_wiring.py tests\test_activation_release.py tests\test_model.py --slow
====================== 171 passed, 7 warnings in 18.27s =======================

$PY -m ruff check .        -> All checks passed!
$PY -m black --check .     -> 128 files would be left unchanged.
```

Count re-derived, not inherited: **2232 → 2246 collected (+14, none removed).**
Those 33 skips are pre-existing (`test_config_system.py`'s own
`pytest.mark.skipif` set), unchanged by this work.

## 5. Mutation verdict

| # | mutation | result |
|---|---|---|
| **A** | gate compares `\|Ω\|/(B·T)` — **the auditor's reading** | **6 red**: `test_the_gate_is_open_across_the_upper_shipped_ramp`, `test_cmc_loss_is_nonzero_through_the_real_bridge`, `test_the_loss_carries_gradient_to_the_secondary_pass`, `test_the_low_leg_skips_...`, `test_gate_ratio_is_the_fraction_of_the_primary_mask`, `test_the_two_diagnostics_are_not_the_same_number` |
| **B** | `cmc_min_overlap_ratio: 0.2 → 0.9` — **option (a), tune the threshold** | **7 red**, spanning both files: the 5 above minus `_not_the_same_number`, plus `test_constructor_default_matches_the_shipped_config`, `test_the_shipped_threshold_is_reached_inside_the_shipped_ramp`, `test_the_gate_opens_across_the_upper_half_of_the_ramp` |
| **C** | skip branch reports `cmc_gate_open=True` — **silent no-op** | **2 red**: `test_the_low_leg_skips_by_reporting_not_by_going_silent`, `test_gate_open_agrees_with_the_threshold_everywhere` |

Restored and re-greened after each: **690 passed, 33 skipped, 0 failed**;
`Select-String "MUTATION"` over all four files returns nothing.
A pristine copy was kept at `.agent-notes/cmc.py.pristine` and
`.agent-notes/defaults.yaml.pristine` and used for the restores.

**Every green claim above carries a mutation.** Mutation A is the one that
matters: it is the auditor's exact reading, and the suite now refuses it.

## 6. Риски / risks

1. **I did not change the gate or the ramp, so CMC stays shut for the first
   ~quarter of the curriculum** (measured: 41/60 draws skipped at
   `mask_ratio_start = 0.15`). That is a real, now-documented property, not a
   bug — but it is a science question whether a mechanism that engages at
   `p ≈ 0.25` is what the paper wants, and **that call is the owner's**.
   Per `AGENTS.md` I did not make it silently. If the answer is "CMC must
   engage from step 0", the two honest moves are raising
   `mask_ratio_start` (a curriculum change affecting all 12 mechanisms) or
   lowering `cmc_min_overlap_ratio` below 0.15 (which admits the degenerate
   tiny-`second_mask_ratio` cases the gate exists to reject). Both are
   escalations, not edits.
2. **`docs/METRICS.md` is now wrong and I do not own it.** Its
   `cmc_overlap_ratio` row claims "the gate skips 6 of 6 … CMC does not run at
   the defaults", and its per-mechanism table carries the claim as the `cmc`
   weakest link. That is a false statement about the mechanism and should be
   corrected by whoever owns the file. `docs/THEORY_AND_PROOFS.md`'s
   **D-CMC-3** is accurate and needs no change. **Escalated, not touched.**
3. **`proofs/IMPLEMENTATION_STATUS.md` row for CMC** (4 / 6 / 2) counts
   claims, and my change moves no claim between Verified / Divergent / Gaps —
   the loss formula, the overlap construction and the non-negativity are
   untouched, and the file is not mine to edit. Flagged so the owner can
   confirm the row still reads correctly. No count was invented.
4. **`cmc_per_position_loss` is an exact duplicate of `cmc_loss`** (METRICS.md
   §4c, verified `|diff| = 0.0`). Left in place: removing a logged key is the
   metrics owner's call and is orthogonal to the gate.
5. **The gate's denominator is a nonzero-*prediction* proxy, not the primary
   mask** (D-CMC-3, still true). On the host path that happens to equal the
   valid-slot count, so the arithmetic above is right; on the module's public
   API a caller passing dense predictions gets `|Ω|/(B·T)` semantics and a
   shut gate. Fixing that properly means taking the primary mask as an
   argument — a signature change to `cmc.py` and `jepa.py`, outside this
   card's boundary and a bigger decision than the evidence supports. The
   diagnostic now reports both quantities, so the distinction is visible
   rather than silent.
6. **New log keys**: `cmc_gate_ratio`, `cmc_gate_open`, `cmc_skip_reason`.
   `MechanismBundle.forward` / `jepa.py:912` will double-prefix them
   (`cmc_cmc_gate_ratio`, …), consistent with the existing 69-of-90
   double-prefix finding, which is not mine to fix. `train.py:1624` reads
   `cmc_info.get("cmc_skipped")` — still present and still a bool.
7. **Test-shape sensitivity.** `cmc_gate_ratio` is a ratio of two random
   counts, so its spread shrinks as `1/sqrt(B·T)` (measured sd 0.054 at 8×64
   vs 0.036 at 16×128). The liveness tests use 16×128, the smallest shape where
   "the gate is open" is a property rather than a coin flip (0/60 shut); the
   shipped 64×512 reads 0/40 with minimum 0.2393. A reader re-running these
   at a much smaller shape will see intermittent skips that are the shape's
   doing, not the mechanism's. This is documented at the test.
8. **Scripts in `.agent-notes/`** (`m_cmc.py`, `m_defaults.py`, `m_gate.py`,
   `m_spread.py`, `m_shape.py`, `m_lowleg.py`, `cmc-measurements.md`) are the
   measurement record. Every one asserts its own `src.__file__` provenance.
   Untracked and not committed — flag if the owner wants them in-tree.

## 7. Not done, deliberately

- No training. No `src/train.py` path entered.
- No file outside the four I own. `train.py`, `jepa.py`, `mechanisms.py`,
  `proofs/`, and `docs/` were read and left alone.
- No skip, no xfail, no `--no-verify`, no assertion weakened to reach green.
- The threshold and the ramp are unchanged — both are science decisions, and
  changing either on the strength of a metrics audit would be the same class
  of error as the audit itself.