# TASK-41 — the stale-documentation sweep

**branch** `agent/task-41` · **base** `agent/wave-7` @ `9d5b0f9` · **commit** `546e2dc`
**worktree** `C:\dev\wt-41` · nothing pushed, nothing in `C:\dev\text-span-jepa` touched.

Files owned and touched: 26 (25 modified, 1 new). No file outside
`README.md`, `AGENTS.md`, `proofs/**`, `docs/**`, `config/ablations/README.md`
was modified. `src/**` and `tests/**` were read only.

## How I worked

The card's method, in the order it actually paid off:

1. `git log --oneline` over the wave commits to learn what changed.
2. For every **number** and every **module/metric name** in the docs, go to the
   code. Config-derived numbers come from `docs/results/param_counts.json` and
   `scripts/results_apparatus.py verify`; model structure from a direct
   instantiation; names and line numbers from `Select-String`.
3. **Extract every `file.py:NNN` citation in the current-authority docs and print
   the line it points at.** This was not in the card and turned out to be the
   single cheapest defect class in the repo.
4. Anything I could not verify, leave and say so.

## Verification, per correction

Required gate, run last, on the committed tree:

```
$ $PY tools\rt.py tests/test_config_system.py --slow
571 passed, 21 skipped in 8.08s          (592 collected)

$ $PY -m ruff check .          All checks passed!
$ $PY -m black --check .       119 files would be left unchanged.
$ $PY scripts/results_apparatus.py verify
verify: PASS -- 62 configs checked. ... none of the 9 refused markers appears ...
```

Every claim below has its own evidence.

### Test count — `AGENTS.md`

Was "686 tests pass, ~108s".

```
$ $PY -m pytest tests --collect-only -q
1960 tests collected in 6.58s                     # this tree, commit 9d5b0f9
$ git rev-list --count 9732de2..HEAD
16                                              # CI's commit is 16 behind
$ git merge-base --is-ancestor 9732de2 HEAD ; echo $?
0                                              # it IS an ancestor, so it is a floor
```

`1910 passed, 21 skipped, 1 xfailed, 0 failed in 65.66s` is CI's own figure for
`9732de2`, quoted from `TASKS.md:705`. I did not run the full suite (the card
forbids it), so the text says **collected**, not passed, and tells the reader to
re-derive rather than trust the number.

I also removed the `~110s` from the Commands block and added a rule not to quote
a local wall clock, because D-001/D-002 already decided that number comes from
CI and the box belongs to the owner.

### 12 vs 16 — `AGENTS.md`

Was "Mismatched counts are a real, currently-open defect in this repo."

```
$ $PY -c "import src.models.mechanisms as M; print(len(M.MechanismBundle.ALL_MECHANISMS), M.GWP.N_MECHANISMS, M.GWP.N_GROUPS)"
12 16 3
$ Select-String defaults.yaml '^\s*use_[a-z_]+:'      -> 12 keys, all true
$ Get-ChildItem proofs -Filter *.md                     -> cgn cmc gac jawp pcr puc rdc spc sta swip wip wsd wsr = 13
$ (Get-ChildItem config/ablations -Filter 'no_*.yaml').Count  -> 14, of which 12 are no_<mech>.yaml
```

The mismatch was **already resolved by convention** in `proofs/README.md` and
decision D-013. I followed that convention, did not invent a new one, and
replaced the "open defect" framing with the two tests that actually pin the
numbers:

- `test_all_mechanisms_length_is_stable` asserts `== 12`; I read its failure
  message and corrected my first draft, which wrongly said it names this file —
  it names the ablation grid, the `mechanisms.py` header, `N_MECHANISMS` and the
  matrix. (Its *docstring* is what points at AGENTS.md.)
- `test_gwp_import` asserts `N_MECHANISMS == 16` and `N_GROUPS == 3`.

### Memory budget — `README.md`

Was "538M parameters ≈ 2.0 GiB of fp32 weights plus **~4.1 GiB** of AdamW m/v state."

```
$ $PY -c "tr=284412929; tot=537990145; G=1024**3; print(tot*4/G, 2*tr*4/G)"
2.003907853782177   2.119260251522064
```

and the premise behind it — that the optimizer covers the total, not the
trainable set — is false:

```
$ Select-String src/train.py 'p for n, p in model.named_parameters\(\) if id\(p\) not in covered and p.requires_grad'
src/train.py:617
$ Select-String src/train.py 'torch.optim.AdamW'
src/train.py:802:  optimizer = torch.optim.AdamW(param_groups)      # no amsgrad -> 2 fp32 states
```

2.12 GiB, not 4.1 GiB — wrong by 1.93×. **This number was not on the card's
list.**

### Baseline capacity — `README.md`

Was "baseline objectives share the same encoder/capacity." TASK-30 removed that
claim in `baselines/mlm_baseline.py` and the README kept it.

```
$ Select-String baselines/mlm_baseline.py 'capacity|identical'
mlm_baseline.py:22: # So the encoder is identical, and the arms are NOT capacity-matched
mlm_baseline.py:78: NOT capacity-matched and NOT compute-matched to the JEPA arm.
$ $PY -c "import json; d=json.load(open('docs/results/param_counts.json'))['configs']; print(d['config/wikitext/mlm_wikitext_small.yaml'], d['config/wikitext/textspanjepa_wikitext_small.yaml'])"
{'total': 113953280, 'trainable': 113953280} {'total': 170706561, 'trainable': 88947841}
```

Encoder 81,758,720 matched exactly in all three arms (measured in
`mlm_baseline.py:16`); the arms are not capacity matched. README now says so,
with the numbers, so a reader cannot re-derive the false claim.

### `pytorch 2.0+` — `README.md`

```
$ Select-String pyproject.toml 'torch>='
pyproject.toml:15:    "torch>=2.3.0",
```

### "CGN and STA have been reconciled" — `README.md`

Contradicted by the matrix rows for both:

```
$ Select-String proofs/IMPLEMENTATION_STATUS.md '^\| (CGN|STA) \|'
| CGN | 4 | 5 | 5 | complementary-gates identity not what the module computes |
| STA | 3 | 6 | 3 | W1(current,ref) was identically zero ... DK reduction invalid |
$ Get-ChildItem proofs -Filter *.md | Select-String 'RECONCILED|Implemented Specification'
pcr.md, spc.md, swip.md
```

Replaced with the accurate list and "every one of the twelve modules has
divergences". **Not on the card's list.**

### Ablation shape — `config/ablations/README.md`

Was "the other **29** shipped configs that train 768/12/12".

```
$ $PY -c "...resolve all 40 ablations over defaults.yaml, count (embed_dim, encoder_depth)..."
resolved ablation shapes (embed_dim, depth): {(768, 12): 40}
```

40 files, all 768/12/12. 29 counted neither the current 40 nor the pre-fix 30.
The other claims in that file **checked out and were left alone**: budget 17
(`PURITY_BUDGET = len(ALL_MECHANISMS) + len(LADDER_VARYING)` = 12 + 5), the
eight `meta: {}` / `optimization: {}` files (8 and 8, verified), the five shape
keys in `base_140m.yaml` (5, verified), `jawk_k_start = jawk_k_end = 76 =
768//10` (verified), and "only SWIP has such a test today"
(`tests/test_swip.py` is the only file asserting `missing lambda_swip`).

### 15 / 13 / 10 — the proofs' internal counts

```
$ Select-String proofs/unifying_principle.md '^#\|'
16 rows (JAWP WIP SpectralGap Grassmann PredictiveRank CGN SWIP PCR SPC WSD CMC GAC STA PUC RDC WSR)
  vs the file's own sentence "All 15 mechanisms"      -> 15 is wrong either way
  vs "Not 13 ad-hoc tricks, but 13 components"       -> 13 was the proof-doc count
$ Select-String proofs/HYPOTHESES.md '^## '           -> H1..H12 = 12
  vs proofs/README.md "10 pre-experimental hypotheses"
```

Corrected to the convention (16 numbered capabilities) with the reason stated
inline. The "Full bound with all 15 mechanisms" line was worse still: the bound
names **ten** risk terms covering **eleven** of the twelve modules — **WSR is
absent**. I said so and left the term missing, because inventing it is designing
a theorem, which is a human decision.

### "All theorems are computationally verified in the test suite" — `proofs/README.md`

This contradicted the matrix's own JAWP row ("cited verification tests absent")
and `jawp.md`'s own banner. Replaced with a table of what is actually pinned:

```
$ Select-String tests/test_jawp.py 'def (test_q_orthonormality|test_courant_fischer|test_wip_preservation|test_stiefel_retraction|test_predictive_rank)\b'
(all five MISSING)                      <- the names jawp.md:156-160 cites
$ Select-String tests/test_jawp.py 'def test_stiefel_retract_keeps_orthonormal'
test_jawp.py:68    (asserts torch.allclose(gram, eye, atol=1e-5))  <- the property IS tested, under another name
$ Select-String tests/test_spc.py 'Parseval'      -> test_spc.py:218, tolerance is `0.8 < ratio < 1.2`
$ Select-String tests/test_cgn.py 'partition|unity'  -> nothing
```

So: 6 of 13 documents have a property test, 6 have none, `wip.md` is UNAUDITED.
**And the old "SPC Parseval relative error < 1e-4" described a precision the
suite never pinned** — the test's band is 20%. The `1e-4` figures that *are*
pinned are the DCT-orthonormality checks, a different property.

### `file.py:NNN` citations — not on the card's list

A line number in prose is a number in prose. I extracted every
`file.py:NNN` from the current-authority docs and printed the target line.
**All six `jawp.py` references in `proofs/README.md` had drifted by 84–85
lines**, plus `jepa.py:810` (now 858), `defaults.yaml:140` (now 198),
`mechanisms.py:544` (now 540), `jawp.py:574` in the matrix's WIP row (now 659),
and the six in `docs/decisions.md` D-013:

```
$ Select-String src/models/jawp.py '^\s*def (workspace_information_preservation|detect_workspace_dimension|grassmann_retract|principal_angles|subspace_distance|predictive_rank_loss)'
jawp.py:552  detect_workspace_dimension        (was cited as 467)
jawp.py:659  workspace_information_preservation(was 574)
jawp.py:893  grassmann_retract                 (was 808)
jawp.py:955  principal_angles                  (was 870)
jawp.py:1005 subspace_distance                 (was 921)
jawp.py:1166 predictive_rank_loss              (was 1082)
```

All re-derived. `docs/related_work.md` — written against recent code — checked
out almost entirely clean across ~90 citations. The four stale citations that
**remain** are deliberate: they are old line numbers quoted inside preserved
history, each next to a note saying they had drifted.

## Recorded, not applied — `src/**` is forbidden to this card

### `gac.py:162` — the card said 162; it is at **152**

The card's line number had itself drifted, which is the same defect class.

```
$ Select-String src/models/gac.py 'detached from graph'
src/models/gac.py:152:            z_pred: (..., D) predictor output (detached from graph).
```

The code is right and the docstring is wrong. Evidence:

```
$ Select-String src/models/jepa.py '_gac_z = span_preds'      -> jepa.py:698  self._gac_z = span_preds
$ Select-String src/train.py 'z_ref = getattr\(model, "_gac_z"'  -> train.py:1398
$ Select-String src/train.py 'model.gac\(z_ref'                 -> train.py:1410
$ Select-String src/train.py 'if loss_gac.requires_grad:'       -> train.py:1411
$ Select-String src/train.py 'scaled\(loss_gac / grad_accum_steps\).backward' -> train.py:1412
```

Executed, module called both ways:

```
z_pred.requires_grad        : True   grad_fn: False
gac loss.requires_grad      : True
max |z.grad| after backward : 6.752795229658659e-07
detached-input loss.requires_grad: False      <-- what the docstring describes,
                                                 and what train.py:1411 skips
```

**The exact replacement wording is recorded verbatim** in
`proofs/IMPLEMENTATION_STATUS.md` and referenced from `proofs/gac.md`. Not
applied, because the file is `src/**`.

This also means the matrix's **GAC headline** ("No-Dead-Zones theorem does not
match **detached-input** batch-mean implementation") repeated the same false
premise. `proofs/gac.md`'s banner already said "the wired call site passes LIVE
predictions", so the matrix row was the odd one out. Struck and corrected: the
real divergence is the batch-MEAN energy and the EMA-smoothed grad norms.

### PUC — the matrix row and `puc.md` banner

```
$ Select-String src/models/puc.py 'use_differentiable_entropy: bool'
puc.py:100:        use_differentiable_entropy: bool = True,          <- TASK-39 flipped this
$ Select-String src/models/puc.py 'eigvalsh'                          -> puc.py:263, inside `if self.use_differentiable_entropy:`
$ Select-String src/models/jepa.py 'PredictionUncertaintyCalibration\(' -> jepa.py:537, and the kwarg list ends at :542 without the flag
$ Select-String defaults.yaml 'puc_use_differentiable'                 -> nothing; the absence is explained at defaults.yaml:272-285
```

So "the loss carries NO gradient" described the **old default**, not the shipped
path. Both corrected with a dated marker; **the Verified/Divergent/Gaps counts
are unchanged** (see "refused" below). Recorded that the inert path still exists,
is still tested, and is unreachable from any config.

### TASK-40's WSD/JAWP conflict — was not in the matrix

Verified the mechanism from the code rather than taking the card's word:

```
$ Select-String src/models/wsd.py 'eigenvectors\[:, -self.k :\]'   -> wsd.py:206   TOP-k of target_cov
$ Select-String src/models/wsd.py 'drift_loss = '                 -> wsd.py:258, maximises ||Q_JAWP^T Q_target||_F^2
$ Select-String src/models/jawp.py 'BOTTOM-k eigenvectors'        -> jawp.py:46
$ Select-String src/models/jawp.py 'high-VARIANCE directions, CONFLICTING'  -> jawp.py:107
```

One correction to the card's framing, found by checking rather than assuming: I
first wrote that the conflict is "not where the shipped configs put it". **Wrong
— 50 of the 62 configs resolve with both `use_wsd` and `use_jawp` true**, including
`defaults.yaml` (λ_wsd 0.01), `all_core.yaml` and every leave-one-out row. Only
the four capacity rungs and four wikitext/tinystories/kaggle JEPA configs set
`use_wsd: false`. The conflict is live in the ablation table, which is the
experiment the paper will run.

Recorded as matrix pattern 5, in `proofs/wsd.md`, `proofs/jawp.md`, and
`docs/related_work.md` §3.10 (where TASK-37 had already found it from the
prior-art direction — so the two were meeting at a fact nobody had put in the
matrix). **Not resolved** — it is a science decision.

## Found beyond the known list

Beyond the card's eleven known-stale items:

1. `README.md`'s AdamW state: 4.1 GiB → **2.12 GiB** (1.93× wrong).
2. `README.md` "baseline objectives share the same encoder/capacity" — TASK-30's
   removed claim, still asserted in the README.
3. `README.md` "pytorch 2.0+" vs `torch>=2.3.0`.
4. `README.md` "CGN and STA have been reconciled" — contradicted by both rows.
5. `config/ablations/README.md` "the other 29 shipped configs" → 39 of 40.
6. `proofs/unifying_principle.md` "All 15 mechanisms" ×2 and "13 components" —
   wrong against its own 16-row table; and the "Full bound" names ten terms
   covering eleven of twelve modules, omitting **WSR**.
7. `proofs/README.md` "10 pre-experimental hypotheses" → 12.
8. `proofs/README.md` "All theorems are computationally verified" — the sentence
   the matrix's own JAWP row contradicts; and the SPC `< 1e-4` tolerance the
   suite never pinned.
9. **The `file.py:NNN` failure class**, 15 drifted citations across three files.
10. **`docs/decisions.md` D-006's open item is closed**: it said `train.py:836`
    read `ema_tau_end` with a `1.0` fallback, so `--no_defaults` froze the
    target encoder. The code is now `train.py:848` with a **`0.9999`** fallback.
11. **The scaling ladder's confounds are gone** — V5's and P7's central premise
    no longer holds. All four rungs now inherit effective batch **512**, lr 1e-3,
    epochs 50, `drop_path_rate` 0.1 and one shared curriculum. "Two points per
    group" is now one group of four.
12. **`flops.py` is not dead and not 2× off** — TASK-22 rewrote it; it now
    reports est/actual **0.99963–0.99980** against `FlopCounterMode`.
13. **PUC was named in `docs/plans/2026-09-05-improve-3` on 2026-09-05 as "B4
    WSD↔JAWP конфликт"** — two weeks *before* TASK-40, and it had still reached
    neither the matrix nor related_work. Recorded, because it changes what
    "never triaged" means.
14. **`src/interp/` still draws from the global RNG at three sites**
    (`ground_truth.py:174`, `information_theory.py:102`,
    `interpretability_index.py:223`) — R11's headline "exactly one seeding call"
    is now **three**, though almost everything else moved to private generators.
15. **I18 is moot**: `tests/test_interp.py` is now **1735 lines**, not the 1222
    I18 counted, so its "638 of 1222 lines are smoke tests" ratio is void; there
    are now **45** test files (I18 counted 21) and a `tests/conftest.py`.
16. **Six wave-1 findings were closed outright** (V6 typo detector now matches
    dotted paths, `TestTrainerTypoDetectorGap`; V9 `load_checkpoint` now raises
    `CheckpointLoadError`; R2/R3/R5/R12; R6 is stronger than asked — there is now
    **no automatic `weights_only=False` fallback at all**).

## Refused, and why

1. **I did not re-derive the matrix's Verified/Divergent/Gaps counts.** Counting
   claims is an adversarial re-audit, not a documentation edit, and I did not run
   one. Inventing new counts would be a guess wearing a measurement's clothes —
   strictly worse than a stale number, because nobody would ever look again. I
   changed only the **headline issue** column, annotated each change with a date
   and a card, and added a header paragraph saying the counts are the
   2026-08-24 audit's and must not be moved without a re-audit. Same reasoning
   for AGENTS.md's new rule that a headline is prose but a count is not.
2. **I did not edit `proofs/HYPOTHESES.md`'s content.** A pre-registered
   hypothesis edited after the fact is not a pre-registration. I added a status
   header only — that none has been run, that **H2/H10/H12 cannot be evaluated as
   written** because no config emits a downstream metric, and that H2's "vs
   `no_cgn` (implicit in default)" is misnamed now that `defaults.yaml` enables
   all twelve (D-005).
3. **I did not touch `README.md`'s first line** ("This repo was a bit edited by
   LLM"). Draft plan improve-5 proposed deleting it. It is the author's sentence,
   not a number, and deleting a person's self-description is not a documentation
   sweep's call.
4. **I did not fix the open defects I found.** I4, I9, I11, I12, I13, V8's
   bundle-path half, D-010's `wsr_mode: sam` xfail, P6's DDP: all are marked OPEN
   against the code with what I verified, and left to their owners. `src/**` is
   forbidden and most are code defects anyway.
5. **I did not touch `plan.md`, `TASKS.md`, `.agent-notes/**` (other cards') or
   `baselines/**`** — outside my file grant. Note for the integrator:
   `plan.md:33,37` and `.agent-notes/SKILL.md:21,34` and
   `.agent-notes/mechanism-fixer.md:57,67,84` still say "686", and
   `TASKS.md:810` carries the same stale count as the card did. `TASKS.md` is a
   live board, so that one is probably intentional; the others are not.
6. **I did not regenerate `docs/results/*`.** It is generated, and `verify`
   passes — 62 configs, guard intact. Nothing to do.

## One judgement call worth flagging

I read "tests only via `tools/rt.py tests/test_config_system.py --slow`" as
*don't run the full suite* (the box belongs to the owner; D-001/D-002 say so).
I used `pytest --collect-only -q` to count, and direct model instantiation for
parameter numbers. Neither executes a test or trains. If a strict reading was
intended, the test-count sentence is the only thing affected and it is labelled
"collected" rather than "passed" so it cannot be over-read.