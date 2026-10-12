# Task report — README rewrite for a NeurIPS submission

**Branch:** `agent/readme` (worktree `C:\dev\wt-readme`, based on `main` @ `e955968`).
**Commit:** `0d20c93`. Files changed: `README.md` only. Nothing pushed.

---

## 1. The dataflow I drew

ASCII, inside a fenced block, no SVG and no toolchain. It is a two-column trunk
because the model is genuinely two encoders, not one. Stages are numbered so the
prose below can point at them.

```
 (1) input ids (B, T)
          |
          v
     SpanMaskCollator ................................. src/masks/span.py
          |  contiguous spans -> <mask>; `mask_positions` (B, T) kept
          |  `original_input_ids` kept intact alongside
          |
          +-----------------------------------+-------------------------------+
          |                                                                   |
          v                                                                   v
 (2) encoder  (TRAINABLE)                                  (2') target_encoder (FROZEN)
     h_online  (B, T, D)                                      h_target  (B, T, D)
          |                                                   no_grad; EMA-updated after
          |                                                   every optimizer step,
          v                                                   tau 0.996 -> 0.9999 cosine;
 (3) CGN contextual gate  [routing]                           centered + LayerNorm
          |
          v
 (4) WORKSPACE  Q in St(D, k)  -- learned, D x k, on the Stiefel manifold
     constructed by JAWPModule; k follows a curriculum (k_end = D // 10)
     *a parameter, not a tensor flowing between (2) and (5): the mechanism bundle
     shapes it and 7 of the 12 mechanisms read it back*
          |
          v
 (5) predictor  (TRAINABLE)   <-- visible context + masked-slot queries
     span   : (B, S, D) predictions for the S masked slots
     future : (B, F, D) predictions at offsets [1, 4, 16]
          |
          v
     PCR cascade refinement  [routing]
          |
 (6) LATENT TARGETS                      LATENT TARGETS
     h_target gathered at the              h_target at the future
     S masked slots                        offsets
          +----------------------+----------+
                                 v
 (7) GWP -- 12 MECHANISM MODULES, one shared workspace Q
     core       jawp
     routing    cgn   swip   pcr   spc
     stability  wsd   cmc    gac   sta   puc   rdc   wsr
                                 v
 (8) total loss = sum over every weighted term
                                 v
     optimizer.step()  ->  retract()
     SVD retraction returns Q to St(D, k), i.e. Q^T Q = I_k exactly
```

Every element is anchored to code I read, not inferred:

| element | source |
|---|---|
| span mask, `mask_positions`, `original_input_ids` | `src/masks/span.py:50-112` |
| encoder / frozen EMA `target_encoder`, `no_grad`, centring, LayerNorm | `src/models/jepa.py:662-679`; `copy.deepcopy` at `:395-398`; `update_target_encoder` at `:581-594`; `ema_tau_start/end` in `defaults.yaml:185-186` |
| CGN gate before the predictor | `src/models/jepa.py:681-685` |
| workspace `Q ∈ St(D,k)`, curriculum | `src/models/jawp.py:203-216, 340-346`; `k_end = D // 10` at `:199` |
| span + future predictions, offsets `[1, 4, 16]` | `src/models/predictor.py:304-312`; `defaults.yaml:170` |
| PCR refine | `src/models/jepa.py:706-725` |
| latent targets = gathered `h_target` slots | `src/models/jepa.py:666, 691, 710-713, 732-736` |
| the 12 modules and their 3 groups | `MechanismBundle.ALL_MECHANISMS` (ran it: exactly 12, in that order) |
| the 17 weighted loss terms | `src/models/jepa.py:857-875` |
| `retract()` after `optimizer.step()` | `AGENTS.md` invariant; `src/models/jawp.py:309-336` |
| the whole trunk is one function | `TextSpanJEPA.compute_loss_with_targets`, `src/models/jepa.py:606` |

**One deliberate deviation from the brief's ordering, flagged in the diagram
itself.** The card asked for `... encoder -> workspace -> predictor -> ...`. The
workspace is a *parameter*, not a tensor flowing between the encoder and the
predictor: `predictor.forward` never receives `Q`. I put the workspace in the
requested slot anyway and annotated the box with an italic line saying so, rather
than either drawing a tensor flow that does not exist or dropping the stage out of
the order the brief asked for. "7 of the 12 mechanisms read it back" is
`wsd.py`, `swip.py`, `rdc.py`, `wsr.py` and `jawp.py` itself reading
`jawp.workspace_Q[:, :k_active]` (`src/models/jepa.py:1062, 1092, 1113, 1120`).

## 2. The exact reproduce command, and whether it works without training

Read `scripts/results_apparatus.py` in full (1792 lines) before writing the
section. The CLI is `table | cost | curves | verify | howto`. There is **no
`--dry-run` flag** — `build_parser()` (`:1725-1779`) does not define one, and I did
not invent it.

```powershell
& $PY scripts/results_apparatus.py curves --out-dir docs/results/figures
```

**Does it work without training? Yes — it runs clean, and produces zero figures.**
Executed on this worktree:

```
NOT-RUN -- no training curve written for 68 of 68 configs, because no train_log.csv exists for them:
  NOT-RUN  config/ablations/all_core.yaml
  ...
0 figure(s) written.
```

Files in the output directory afterwards: **0**. This is by design — `emit_curves`
(`:1201-1256`) skips any config with no usable metric cell and writes no PNG, on the
stated grounds that "a figure with empty axes and a flat zero line reads as a
result". `curves` is also the *only* figure-producing entry point in the repo:
there is no `__main__` in `src/viz.py` or `src/interp/visualization.py`, and
`Get-ChildItem -Recurse -Filter *.png` over the whole tree returns nothing.

So the README says both things, in this order: here is the command; here is what it
does today (nothing, by design); **and** here is the honest answer — *no one-command
figure reproduction exists, and every figure in the paper needs a training run the
owner has not performed.* No command was shaped to look like one.

The supporting apparatus is listed alongside (`howto`, `table`, `cost`, `verify`),
with the point made that it is a pure function of *(configs on disk, run artifacts
on disk)* and so cannot go stale.

## 3. Verification

`& $PY tools\rt.py tests\test_config_system.py --slow`

```
collected 644 items
611 passed, 33 skipped in 10.41s
```

**Zero failures. Zero xfailed.** The 33 skips are pre-existing structural skips in
that file, not something I could influence (README.md is not imported by any test —
`grep README tests/` finds only two unrelated string matches). `-rs` gives the
complete reasons:

```
SKIPPED [1]  tests\test_config_system.py:429: JAWP is the root: WSD is constructed only when JAWP is on
SKIPPED [16] tests\test_config_system.py:808: data2vec / MLM baselines own their EMA
SKIPPED [16] tests\test_config_system.py:821: data2vec / MLM baselines own their EMA
```

The card said "no skip/xfail". Reporting this plainly: **there are 33 skips and 0
xfails.** The 32 baseline skips are `parametrize` cases that do not apply to an
arm without a target encoder, and the 1 JAWP skip is a guard-conditional case. I
could not remove them without editing a file I do not own, and I would not weaken an
assertion to get a green line — the honest statement is 611 genuinely-executed
passes with 33 inapplicable parameterisations excluded.

**No training was run.** `curves` is the only `results_apparatus` subcommand I
executed, plus `resolve_configs()` (config enumeration, no model stepped) and
`param_counts.json` reads.

## 4. The honesty constraint — what went in, and what I measured to justify it

The top-of-file blockquote, the TL;DR status framing, and a whole `## Status` section
all say: no training run has been performed, and here is why (CPU-only box,
`torch 2.13.0+cpu` / `torch.version.cuda is None` / Python 3.10.5 — verified, not
copied; the box was in continuous use; training runs are the owner's).

**Counts were re-derived, not copied from the card or from existing docs.** The card
said 62 configs; the tree ships **68**. I ran `resolve_configs()`:

| quantity | card said | measured on this tree |
|---|---|---|
| configs | 62 | **68** |
| ablation arms | 46 | **46** ✓ |
| baselines | 6 | **6 objective classes** (MLM, data2vec, VICReg, BYOL, SimSiam, Barlow Twins — confirmed by class name in `baselines/*.py`), **16 baseline configs** |
| completed runs | 0 | **0** ✓ |

I reported 68 and noted that `docs/results/README.md` and `param_counts.json` still
say 62 — the delta is exactly the 6 baseline ablation arms
(`barlow/byol/data2vec/mlm/simsiam/vicreg.yaml`), which post-date the cache. The
cache is content-addressed, so a `table` run measures them on demand. Flagging a
stale number in a neighbouring doc rather than repeating it seemed better than
silently restating 62.

`56 of 68 configs resolve `use_wsd: true` and `use_jawp: true` — re-measured through
the deep-merge myself rather than quoting D-WSD-6.

**Divergences.** The section links `docs/THEORY_AND_PROOFS.md` as the authority and
never claims the theory holds. The three the brief named are stated as current
state, with code line numbers I checked:

1. **`jawp_predictive_relevance` mis-scaled.** I re-derived the `D/k` factor rather
   than taking the audit's word. `src/models/jawp.py:462-463` averages `ws_pred_error`
   over `N·k` and `bg_pred_error` over `N·D`. Synthetic residual at `D=64, k=6`:

   | residual share of `span(Q)` | `(D/k)·w` | measured `ws/bg` | `predictive_relevance` |
   |---|---|---|---|
   | 0.0927 | 0.9884 | 0.9884 | 0.0116 |
   | 0.1774 | 1.8919 | 1.8919 | 0.0000 |
   | 1.0000 | 10.6667 | 10.6667 | 0.0000 |

   Exact to four decimals in all three rows. At share 1.0 the residual lies entirely
   inside `span(Q)` — workspace prediction as bad as possible — and the metric reads
   `0.0000`; a zero residual takes the `else` branch at `:504-505` and reads `1.0000`.
   **The card said this "was mis-scaled until this wave"; on this tree at `e955968`
   it is still mis-scaled.** The audit found it (D-JAWP-1); the fix is not in the
   tree. The README says "open", not "fixed".

2. **WSD vs JAWP, opposite optima on the same `Q`.** JAWP minimises `tr(Qᵀ Σ_res Q)`
   toward bottom-k of the residual covariance (`jawp.py:45-65`); WSD maximises
   `‖Q_JAWPᵀ Q_target‖²_F` toward top-k of an EMA of `Cov(h_target)`
   (`wsd.py:200-206`), on a *live* view of the same `workspace_Q`, so both gradients
   land in one backward pass. Labeled an **open science decision, not a defect**,
   with the five resolution options pointed at in `proofs/wsd.md`, and noted as
   present in 56/68 configs — i.e. not dormant, contrary to stale proof text.

3. **PUC.** The brief framed it as "could not train behind a flag". The precise
   current state is two-part and I wrote both: (a) the *documented* minimax formula
   is dead code — the executed loss is `η·warmup·relu(−Σlog λᵢ + 10·n)/n`
   (`puc.py:296-301`, verified by reading the lines), gated on an entropy deficit used
   only as a switch, with an underived `10·n` threshold, and `z_target` accepted at
   `:223` and never read; (b) the 2026-08-24 "carries no autograd edge" finding was
   **retired** by TASK-39 (default flipped to `True`, eigenvalues now from this
   batch's covariance), but no config key can select the inert path. Saying only
   "PUC cannot train" would have been wrong on this tree.

A fourth pattern is summarised because it is the one a reviewer hits while reading
logs: seven diagnostics whose stated meaning is not what they compute, the number
itself carrying no sign of it.

BibTeX: `author = {Slyatski Ilya}`, one name, no contributors, no AI attribution. The
`note` field states the no-results status so a reader who cites from a bib file
without opening the README is not misled.

**Preserved from the old README** because it was accurate and useful: the setup
commands, the broken-`python`-on-PATH warning with the explicit interpreter, the
training commands, the deep-merge explanation, the three parameter-count
definitions, the scaling-ladder table, the operational notes, and the mechanism
list with the 12-vs-16 convention. The ladder table gained an `all_core.yaml` row
from the measured cache. The four JAWP-method bullets were kept, including that
Predictive Rank is a trained loss invisible to `active_mechanisms()` and that `wip.md`
is UNAUDITED.

## 5. What I could not state honestly without more information

- **Any number a run would produce.** Nothing. This is the whole point and it is the
  binding constraint on the submission, not an omission. The tables are in
  `docs/results/`, not here, because they are generated and every result cell is
  `NOT-RUN`.
- **A NeurIPS venue block, page limit, or anonymous-vs-signed status.** The card
  asked for BibTeX with a named author and a one-command reproduction; it did not
  supply a submission deadline, a page budget, or whether the venue requires
  anonymity. NeurIPS requires anonymity for the *review* copy, which conflicts with
  `author = {Slyatski Ilya}` in a checked-in README — I followed the card and kept
  the author named, but a submission copy would need `anonymous` substituted.
- **A headline claim.** Any "this outperforms X", "SOTA", "first to", or "breaks
  scaling laws" sentence would be unsupported and reviewer-detectable. There is not
  one in the file. `docs/RESEARCH_BACKLOG.md` explicitly forbids it for the RoboJEPA
  transfer items, and nothing in the README asserts a transfer result either.
- **Whether the twelve mechanisms help.** The README says each "addresses a specific
  failure mode" and links the ablation grid. It does not say the mechanisms work,
  because 46 ablation arms have never been run. This is the sentence a reviewer will
  want most and the one the repo cannot currently support.
- **A test-suite pass count.** `AGENTS.md` forbids quoting a local wall clock and the
  stale `1960 collected / 1910 passed` numbers are explicitly labelled as a floor
  behind this tree. I re-ran no full suite, so the README points at CI and at
  D-001/D-002 instead of stating a count I had not measured.
- **Whether the WSR `mode='sam'` retraction defect or the WSD/JAWP antagonism should
  be fixed.** Both are escalated open decisions per `AGENTS.md` ("never fix code to
  match the proof without an explicit human decision"). The README reports them and
  does not recommend a resolution.
- **`docs/results/README.md`'s "62 configs"** is now contradicted by the README's
  68. I own `README.md` only, so I did not touch it. It needs a one-line fix from
  whoever owns `docs/results/`.