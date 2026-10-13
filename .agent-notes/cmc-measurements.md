# CMC gate — measured facts (worktree C:\dev\wt-cmc, base 7f46e63)

Provenance asserted in every script: `src` resolves to `C:\dev\wt-cmc\src\__init__.py`.
No training. No `src/` edits at measurement time.

## 1. The gate is NOT `|Omega|/(B*T)`

`src/models/cmc.py:444`

    primary_masked = (z_pred_primary.abs().sum(dim=-1) > 0).float()   # :443
    overlap_ratio  = overlap_count.float() / max(primary_masked.sum(), 1)

The denominator is the count of positions carrying a NONZERO PREDICTION, not
`B*T`. The auditor read the LOGGED key (`:489`, `mean_b |Om_b| / T` = `|Om|/(B*T)`)
and compared *that* against `min_overlap_ratio`. Two different quantities.

## 2. First principles

Two masks, each masking fraction `p`, positions iid:

    P[t in both]   = p^2
    E|Om|          = B*T*p^2          <- what the LOGGED key reports
    |M1|           = B*T*p           <- what the GATE divides by
    gate ratio     = p               <- NOT p^2
    logged ratio   = p^2

The threshold `0.2` is therefore a test on the SECOND MASK'S RATIO. It is
reachable: the shipped ramp tops out at `mask_ratio_end = 0.35 > 0.2`.

## 3. Measured — real `SpanMaskCollator` primary, real `generate_second_mask` secondary
   B=16 T=128, 25 trials, `data.span_length_range=[3,10]` (`m_gate.py`)

    p      |M1|/BT   |Om|/BT   gate=|Om|/|M1|   shut
    0.15   0.1484    0.0267    0.1796           18/25
    0.20   0.1953    0.0445    0.2279            3/25
    0.25   0.2500    0.0688    0.2750            0/25
    0.30   0.2969    0.0934    0.3146            0/25
    0.35   0.3438    0.1267    0.3687            0/25

with 25 % padding: gate 0.1842 / 0.2369 / 0.2987 / 0.3264 / 0.3698,
shut 18/25, 6/25, 0/25, 0/25, 0/25.

Gate ratio tracks `p` to within ~5 %, exactly as the arithmetic says. The
auditor's `0.047 .. 0.164` is the `|Om|/(B*T)` column, reproduced here as
0.0267 .. 0.1267.

## 4. Measured — through the REAL bridge (`compute_cmc_between_passes`, real defaults)
   B=8 T=64, 8 trials (`m_gate.py`)

    p      ran/total   mean cmc_loss(ran)
    0.15   2/8         6.2202e-05
    0.20   7/8         8.55952e-05
    0.25   7/8         8.80515e-05
    0.30   7/8         9.59391e-05
    0.35   8/8         9.54934e-05

CMC RUNS at the shipped defaults, with a non-zero loss, for 4 of the 5 ramp
points. The card's premise ("skips 6 of 6", "CMC does not run at the
defaults") is false.

## 5. THE REAL DEFECT — one key, two meanings

Measured with 4x32, predictions nonzero at 20/32 positions, threshold 0.2:

    branch         skipped  cmc_overlap_ratio  |Om|/(B*T)  |Om|/|prim|  keys
    no-overlap     True     0.0000             0.0000      0.0000       6
    insufficient   True     0.1500             0.0938      0.1500       7
    ran            False    0.3750             0.3750      0.6000       7

Same key: gate form on the skip branch, fraction-of-positions on the run
branch. They differ by `|M1|/(B*T) ~ p`, i.e. by a factor of 2-3 here. Three
branches, three key sets. A reader cannot tell which branch produced the
number, and `src/train.py:1627` independently overwrites `cmc_overlap_ratio`
with the `|Om|/(B*T)` form while `jepa.py:912` writes the module's own value
under `cmc_cmc_overlap_ratio` — so the log carries BOTH forms adjacent to
each other under near-identical names.

This is why a live mechanism was audited as dead: the key that could have
told the reader CMC was running was reporting the wrong quantity.

## 6. Verdict on (a)/(b)/(c)

(a) lower the threshold — NO. 0.2 is already met from `p >= 0.20` upward and
    is cleared with margin at the ramp endpoint (measured 0.369 vs 0.2).
    Lowering buys only the 0.15->0.20 leg (~1/4 of `jawk_curriculum_steps`
    = 10000, so ~2500 steps) and weakens the only guard the gate provides.

(b) the shipped mask ratio is wrong — NO. The 0.15->0.35 ramp is the one
    deliberate knob, documented at `defaults.yaml:123-133` and pinned by
    `TestOnlyOneMaskRatioKnobIsLive`. Moving it changes the mask schedule of
    every other mechanism and every published arm.

(c) make it VISIBLE, not make it run — RIGHT REMEDY, WRONG PREMISE. CMC's
    value is NOT confined to configurations nobody ships; it is live at the
    shipped defaults. What is missing is any way to SEE that.

So: keep 0.2, keep the ramp, and make the liveness legible and un-hideable.

## 7. Scripts

`.agent-notes/m_cmc.py`     — arithmetic, mask-only sweep, bridge sweep, dense/sparse
`.agent-notes/m_defaults.py` — resolves the real defaults.yaml, drives the bridge
`.agent-notes/m_gate.py`     — real collator + real bridge + key semantics