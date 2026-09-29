# TASK-30 — make the baseline actually a control

**status: done.** Branch `agent/task-30`, commit `34e7fd0`, off `agent/wave-1` @ `b91b273`.
Worktree `C:\dev\wt-30`. Nothing pushed. No test skipped, no xfail, no `--no-verify`,
no assertion weakened or deleted. No training run.

## Files

| file | status |
|---|---|
| `baselines/mlm_baseline.py` | modified — the false guarantee removed, replaced by measured figures; `get_num_params_trainable()` added |
| `tests/test_baseline_parity.py` | new, 23 tests |

Nothing outside `files_allowed` was touched. `config/**` and `src/models/**` untouched
(confirmed by the diff-stat below). No forbidden path appears.

## The claim was false in the direction that favours the control

I re-measured every figure myself rather than copying them from the seed note or the
control-scout, including checking the seed's known digit transposition. All figures
reproduce exactly. Measured at the shipped 640/10 rung (V=50304, max_seq_len=512):

```
one encoder (either arm)            81,758,720
MLM mlm_head, untied                 32,194,560
MLM total                           113,953,280
MLM trainable                       113,953,280
JEPA encoder                        81,758,720
JEPA target_encoder (frozen)        81,758,720
JEPA predictor                       5,508,481
JEPA decoder (own params)            1,639,680
JEPA total                          170,706,561
JEPA trainable                       88,947,841

JEPA TOTAL / MLM TOTAL              1.4980x
MLM TRAINABLE / JEPA TRAINABLE      1.2811x
MLM excess trainable                25,005,439
frozen deepcopy == encoder exactly  True
```

I confirmed the corrected figure **81,758,720** (not the seed's transposed 81,958,720).
It reconciles as `170,706,561 - 88,947,841`, i.e. exactly one encoder deepcopy, and it
matches the repo's own documented total in `config/scaling/small_100m.yaml:20-21`.

**The excess is a structural identity, not a coincidence at one shape.** The two
encoders are the same class at the same dims, so they cancel and
`excess == mlm_head - (JEPA trainable that is not its encoder)`. I verified this holds
exactly at five shapes, including one where the sign flips (V=257/D=128 gives
**-155,521**, i.e. JEPA larger). It is now a test over two shapes.

**"Excess is a frozen EMA copy" is backwards, and the test says so.** The
`target_encoder` has `requires_grad=False` on every parameter (asserted), so it carries
no gradient path and is not a capacity advantage. The untied head is. Both readings are
now pinned, so a future reader cannot re-derive the inverted one.

**Compute is not matched, and cannot be.** `compute_loss` projects all B*T positions
before gathering; `jepa.py:716-718` decodes masked rows only. At the 64x512 micro-batch
fp32: dense (B,T,50304) = **6.14 GiB**; masked at 0.35 = **2.15 GiB** (2.860x); masked at
MLM's own 0.15 = **0.91 GiB** (6.737x). `data2vec_baseline.py` gathers *before* its
head, so data2vec is unaffected — pinned too.

Note the honest direction of the compute story, which the card's framing does not
emphasise: the dense-logit penalty disadvantages **the baseline**. The capacity
asymmetry favours it. The two asymmetries point in opposite directions, which is why
the header reports both ratios rather than the flattering one.

## What I did about it, and why not a docstring edit

The card is explicit that a docstring edit alone is not a fix. So the claim is
**removed**, and the replacement is a *measurement* — the header now carries the
numbers, the mechanism, and the cost of parity — and `tests/test_baseline_parity.py`
pins every number in it. If the architecture changes, a test goes red and the header
must be rewritten with it. That is the difference between a docstring and a pinned
claim.

**Genuine parity is not reachable without changing the science, so I did not fake it.**
The header names exactly what would have to change, and each option changes a trained
result, so it is a human decision (AGENTS.md: "never fix code to match the proof without
an explicit human decision" — same principle):
1. Tie `mlm_head` to the token embedding. Measured consequence: this arm drops to
   81,758,720 trainable = **0.9192x** JEPA. The asymmetry **flips sign**, it does not
   vanish, and it changes what the control *is*.
2. Widen JEPA's predictor by ~25M parameters to absorb the head.
3. Train both arms under a published parameter budget and report the budget.

I verified 0.9192 by computation rather than asserting it in prose.

One thing I deliberately did **not** do: move the gather above the head in
`compute_loss`. It would remove the 2.86x activation penalty, but it is a behaviour
change to a training path and would not equalise FLOPs anyway. The control-scout
recommended the same split, and the test pins the current gather position in all three
arms so whoever does it sees it break.

## Verify — full paste

```
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
& $PY tools\rt.py tests/test_baseline_parity.py tests/test_model.py --slow
```

```
rt.py: C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_baseline_parity.py tests/test_model.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 174 items

tests/test_baseline_parity.py ...............................          [100%]
tests/test_model.py ..........................                             [100%]

============================== warnings summary ===============================
tests/test_model.py::TestData2VecBaseline::test_forward
tests/test_model.py::TestData2VecBaseline::test_loss_formula_data2vec
tests/test_model.py::TestV011TrainingReadiness::test_compute_loss_unified_interface
tests/test_model.py::TestV011TrainingReadiness::test_data2vec_training_loop
  C:\dev\wt-30\baselines\data2vec_baseline.py:121: UserWarning: data2vec target depth truncated: encoder exposes 2 layers, average_top_k_layers=8
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 174 passed, 4 warnings in 37.80s =======================
```

**174 passed** = 151 pre-existing (`tests/test_model.py`, unchanged, measured green
before I started) + 23 new. The 4 warnings are pre-existing and come from
`data2vec_baseline.py`, untouched by me. Budget 90s, used 37.8s.

Lint and format, both clean:
```
$ & $PY -m ruff check baselines/mlm_baseline.py tests/test_baseline_parity.py
All checks passed!
$ & $PY -m black --check baselines/mlm_baseline.py tests/test_baseline_parity.py
2 files would be left unchanged.
```

## Diff-stat

```
 baselines/mlm_baseline.py     |  93 +++++++++-
 tests/test_baseline_parity.py | 392 ++++++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 477 insertions(+), 8 deletions(-)
```

## Mutation verdict

Loosening the pins and showing red, three ways. Both sides pasted.

**1. Loosen `RATIO_TOL` 0.005 → 0.40** — stayed green, and that is the correct result,
not a missed mutation. The pinned *value* was still true, so widening the comparison
band around a true value cannot fail. This is evidence the tolerance is not doing the
work; the value is. Restored.

```
tests\test_baseline_parity.py .......................                    [100%]
23 passed in 8.71s
```

**2. Corrupt the pinned ratio by one digit** (`88_947_841` → `98_947_841`) — **RED**.
This is the doc-drift the tolerance exists to catch, and it catches a single digit:

```
tests\test_baseline_parity.py ...F...................                    [100%]
____ TestTrainableCapacityAsymmetry.test_documented_trainable_ratio_holds ____
tests\test_baseline_parity.py:128: in test_documented_trainable_ratio_holds
    assert ratio == pytest.approx(MLM_OVER_JEPA_TRAINABLE, rel=RATIO_TOL)
E   assert 1.2811247436573532 == 1.151649989007845 ± 0.00575825
E     comparison failed
E     Obtained: 1.2811247436573532
E     Expected: 1.151649989007845 ± 0.00575825
========================= 1 failed, 22 passed in 9.05s =========================
```

**3. Restore the guarantee as a live claim in the header** — **RED, 3 tests**. This is
the exact regression the card is about, and the guard names it:

```
E   AssertionError: baselines/mlm_baseline.py asserts 'fair comparison guarantee' again.
E   The arms are not capacity- or compute-matched; see the module header for the
E   measured figures.
E   assert 'fair comparison guarantee' not in '# copyright... have)\n    '
E     'fair comparison guarantee' is contained here:
E       -------
E       # fair comparison guarantee: identical model capacity, identical compute.
E     ?           ++++++++++++++++++++++++++++++++++++++++++++
=========================== 3 failed, 20 passed in 9.01s ===========================
FAILED ...test_no_live_parity_guarantee[identical model capacity]
FAILED ...test_no_live_parity_guarantee[identical compute]
FAILED ...test_no_live_parity_guarantee[fair comparison guarantee]
```

The guard can tell a quotation from an assertion: the old wording survives in the
module docstring inside `--- BEGIN/END HISTORICAL CLAIM ---` markers, and
`test_historical_quotation_is_still_delimited` fails if someone widens that block to
cover the file rather than fixing the claim. A separate mutation-test I did not need to
run confirmed the marker logic is load-bearing: the first draft of this guard was a
bare substring scan, and it went red on my own historical quotation — which is why it
is marker-scoped rather than a scan.

All three restored; `git status` clean, 174 passed after restore.

## не_сделано

- **Parity itself is not achieved.** Deliberate. Every route to it changes a trained
  result, so it is a human experiment decision, not a code fix. The header names the
  three options with measured consequences; I did not choose one.
- **Gather moved above the head** — not done. Behaviour change to a training path, and
  it does not make FLOPs equal. Pinned instead.
- **The 3.89x prediction-target asymmetry** (50ep x 0.35 vs 30ep x 0.15) — I confirmed
  the arithmetic but did **not** touch it. The control-scout killed it as a card: it is
  uniform across all 8 baseline configs, already disclosed in every one of their
  headers, and explicitly routed to a human as an experiment decision. It is also not
  capacity, and not in this card's scope.
- **`data2vec_baseline.py`** — same "same encoder" family, but it makes no parity claim
  (grep for `identical model capacity|identical compute|fair comparison` hits only
  `mlm_baseline.py`), and it is not in `files_allowed`. Its gather-before-head is
  asserted as the contrast case but not edited.
- **`config/wikitext/mlm_wikitext_small.yaml` header** — the card forbade `config/**`.
  The control-scout suggested documenting the asymmetry there too. Not done; the
  numbers live in the module and its test, and the config header's existing
  token-budget disclosure is a separate, already-closed matter.
- **Full suite / training** — not run. The wrapper refuses whole-suite runs and I did
  not attempt to work around it. Only the two assigned files were verified, plus lint
  and format on those two.

## риски

- **Cost.** The `arms` fixture builds both models at the full shipped rung
  (~113M + ~171M params, ~1.1 GB RAM, ~3.5s). That is why `tests/test_model.py` is
  `--slow`-gated and this file is not; together 37.8s against a 90s cumulative budget.
  The other 21 tests are cheap. On a much smaller box the fixture is the thing to
  shrink — but shrinking it below the shipped rung would stop pinning the numbers the
  header quotes, which is the whole point.
- **Tolerance is 0.5% on ratios, and ratios not counts.** Deliberate: a legitimate
  future vocabulary change moves both counts and should not need a test edit, but must
  not be able to move the *relationship* unnoticed. The raw absolute counts in the
  header are checked separately by `test_module_states_the_measured_numbers`, which
  asserts the figures are present as text — that is the weaker half of the pair and it
  would not catch a count that changed in both code and prose.
- **`test_gather_position_is_a_property_of_the_source` is source inspection**, not
  behaviour. It pins *where the gather sits* by reading `inspect.getsource`. If someone
  refactors `forward` to compute the head through a helper, this goes red on a
  `ValueError: substring not found` rather than a clean assertion. I hit exactly that
  during development (it first said `self.head(`, and data2vec's attribute is
  `regression_head`). It is brittle by construction and deliberately so — the point is
  to make that refactor a deliberate act — but a future reader should not mistake the
  failure for a real regression.
- **Two of my own bugs, both caught, both worth recording.** My first fixture passed
  `encoder_depth=10` to `MLMBaseline`, which takes `depth` and swallows the rest in
  `**kwargs` — it silently built a depth-12 encoder and measured **1.392x** instead of
  1.281x. `src/train.py:476-484` maps the names explicitly, so the production path is
  correct; only the test was wrong. I left a comment at the fixture so the trap is
  recorded. Second, I initially mis-stated the excess identity (forgetting the
  encoders cancel) and it read `False` at every shape; the corrected form is in the
  test. Both are the reason the final numbers are the ones I measured rather than the
  ones I first believed.
- **Unverified claim I did not make.** I did not check whether any trained checkpoint
  or published result already depends on the current untied head. Option 1 above would
  invalidate any such result; that is a further reason it needs a human, not me.
