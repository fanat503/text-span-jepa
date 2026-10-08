# Implementation Status of Proofs (audited 2026-08-24)

Automated adversarial comparison of every proof document against its
implementation. **Twelve capabilities have been audited** — the twelve modules
in `MechanismBundle.ALL_MECHANISMS`
([`src/models/mechanisms.py:100`](../src/models/mechanisms.py)). Of the other
four numbered capabilities, only WIP (#2) has a proof document; it is listed
below for completeness, marked **UNAUDITED**, with no claim counts recorded for
it. Spectral Gap (#3), Grassmann Optimization (#4) and Predictive Rank (#5) are
methods of `JAWPModule` with no proof documents of their own, so they have no
rows here — their claims are argued inside `jawp.md`. Treat proofs/ as DESIGN
documents with audited gaps, not as certified descriptions of the running
code.

The 12-vs-16 counting convention these rows sit in — twelve modules, sixteen
numbered capabilities, four of them JAWP methods — is stated in full in
[`README.md`](README.md). There are thirteen proof documents: one per audited
module, plus `wip.md`.

**The Verified / Divergent / Gaps numbers are the 2026-08-24 audit's, unchanged
since.** They have deliberately not been re-derived by later cards, because
re-counting claims is an adversarial re-audit, not a documentation edit. What
later cards *did* change is the **headline issue** column and the cross-mechanism
list; those carry a dated correction marker and name the card that moved them.
A headline that silently disagrees with the code is the defect this matrix
exists to prevent, so update it in the same commit as the code — and never
invent new counts.

| Mechanism | Verified | Divergent | Gaps | Headline issue |
|---|---|---|---|---|
| WSR | 3 | 7 | 5 | detached loss cannot minimize the claimed objective; bound uses EMA proxies |
| WSD | 6 | 6 | 3 | orthonormality/lambda-coupling assumptions not enforced; Davis-Kahan bound circular. **Also: pulls the same `Q` toward the opposite optimum from JAWP — see cross-mechanism pattern 5 (added 2026-10-08, TASK-40/TASK-41).** |
| SWIP | 4 | 6 | 5 | theorem term 2 missing; scale-invariance false; dead conditional fixed in R11 |
| STA | 3 | 6 | 3 | W1(current,ref) was identically zero (sync refresh bug) — fixed in R11; DK reduction invalid |
| SPC | 3 | 5 | 5 | proved simplex/norm-squared object differs from implemented floored-softmax MSE |
| PCR | 2 | 7 | 3 | proof analyzes linear Stiefel cascade; code is gated MLP cascade |
| PUC | 2 | 8 | 5 | headline formula is dead code; executed loss matches neither stated form. **Correction 2026-10-08 (TASK-39): the "no gradient" part of this row is no longer true of the shipped default — see the note under the table.** |
| RDC | 2 | 6 | 4 | transient bound internally inconsistent in proof, falsely justified in header |
| GAC | 3 | 6 | 4 | ~~No-Dead-Zones theorem does not match detached-input batch-mean implementation~~ **Correction 2026-10-08 (TASK-41): "detached-input" was wrong and is struck.** The wired call site passes LIVE predictions; the remaining divergence is the batch-MEAN energy (and the EMA-smoothed grad norms), so the No-Dead-Zones bound scales by 1/N and the warmup ramp rescales it again. |
| CGN | 4 | 5 | 5 | complementary-gates identity not what the module computes |
| CMC | 4 | 6 | 2 | stability theorem misstated (averaged vs pointwise); reuse_encoder path is a stub |
| JAWP | 1 | 6 | 4 | time-varying curriculum k vs fixed-k theorems; cited verification tests absent |
| WIP | **UNAUDITED** | **UNAUDITED** | **UNAUDITED** | **UNAUDITED** — no adversarial pass was run for this capability, so this row records no claim counts. WIP is the only proof document with no row in this matrix. [`wip.md`](wip.md) lacks the inline audit banner, though so do `cgn.md`/`pcr.md`/`spc.md`, which *are* audited above — the absence of the banner is not what marks a capability unaudited. `JAWPModule.workspace_information_preservation` (`src/models/jawp.py:659`; the `574` originally written here had drifted) also has no `use_wip` key, so no config arm exercises it. Treat every claim in `wip.md` as unverified. |

### PUC — the "no gradient" half of the row, superseded 2026-10-08 (TASK-39)

The 2026-08-24 audit recorded two separate PUC problems: (a) the headline
Lagrangian-dual formula is dead code, and (b) the executed loss carries no
autograd edge, because it read the Oja **EMA buffer** rather than this batch's
covariance. **(a) still stands.** **(b) is fixed for the shipped default and
was wrong as a description of what a run does.**

TASK-39 changed `PredictionUncertaintyCalibration.__init__`'s
`use_differentiable_entropy` default from `False` to `True`
(`src/models/puc.py:100`). With it `True`, the loss's eigenvalues come from
`torch.linalg.eigvalsh` of **this batch's** covariance, computed with autograd
(`src/models/puc.py:261-264`), so the returned scalar carries an edge to
`z_pred`. `src/models/jepa.py:537-542` does not pass the flag, so the training
path takes the grad-carrying branch. Measured on the default config: the max
`|encoder.grad(PUC on) − encoder.grad(PUC off)|` was exactly `0.0` before the
default flipped and `2.2e-08` after (`.agent-notes/task-39.md`).

Two things about this correction, because they are the reason the headline is
annotated rather than simply rewritten:

- **The flag is still unreachable from config.** There is deliberately no
  `puc_use_differentiable_entropy` key (`defaults.yaml:272-285`), so no shipped
  arm can select the inert path. Exposing it needs `jepa.py` and
  `mechanisms.py`, outside that card's ownership. The dead-gradient behaviour
  still exists and is still tested; it is just not the default.
- **The counts were not re-derived.** PUC is still `2 / 8 / 5`. The divergence
  that matters for a paper is (a): the executed loss matches neither stated
  form, and `puc_carries_grad` in the info dict is the runtime tell for which
  branch ran.

### GAC — "detached-input" was never true of the wired path, superseded 2026-10-08 (TASK-41)

`proofs/gac.md`'s banner already said so ("the wired call site passes LIVE
predictions (R12 wiring)"), which made the matrix row the odd one out. Verified
against the code:

- `src/train.py:1398` reads `model._gac_z`; `src/models/jepa.py:698` sets
  `_gac_z = span_preds`, the live slot predictions.
- `src/train.py:1410` calls `model.gac(z_ref, g_norms, step=...)` and
  `src/train.py:1411-1412` backward-s `loss_gac` **into that graph**, guarded on
  `if loss_gac.requires_grad`.
- Executed: with a grad-carrying `z_pred`, `loss.requires_grad` is `True` and
  `z_pred.grad` reaches `6.75e-07` after one `backward()`. With a detached
  `z_pred` the loss has `requires_grad=False` — which is exactly the branch
  `train.py:1411` skips, and what `src/models/gac.py:152`'s docstring claims.

`src/**` was not this card's to edit, so the docstring correction is recorded
rather than applied:

> **`src/models/gac.py:152`** — `z_pred: (..., D) predictor output (detached from graph).`
> should read: *`z_pred: (..., D) predictor output, LIVE — not detached.
> `src/train.py:1410` passes `model._gac_z`, the live slot predictions, and
> backward-s `loss_gac` into that graph at `src/train.py:1412`. A detached
> `z_pred` yields a loss with `requires_grad=False`, which `train.py:1411` then
> skips — so the docstring describes a path the trainer never takes.*

The batch-mean / EMA-proxy divergence the row is really about is unchanged; only
the false premise is struck.

## Cross-mechanism patterns

1. Loss formulas diverge from their stated form after refactors
   (PUC dead formula, SPC loss type). **SWIP's missing term 2 was in this
   list on 2026-08-24 and is now implemented (R11/R15) with the
   scale-invariance claim retracted, so it no longer belongs here.**
2. Bounds are evaluated on EMA-smoothed proxies instead of theorem inputs
   (WSR, RDC, WSD).
3. Assumptions used by proofs (orthonormality, exact simplexes, fixed k)
   are not enforced by the corresponding modules.
4. Several theorems prove properties of objects that were never implemented
   (PCR cascade form, CGN complementary gates).
5. **Added 2026-10-08 (found by TASK-40, recorded here by TASK-41). Two
   mechanisms pull the same parameter toward opposite optima, and no audit
   pass had recorded it.** WSD's loss maximizes
   `‖Q_JAWPᵀ Q_target‖²_F` over the **top-k** eigenvectors of an EMA of the
   target covariance (`src/models/wsd.py:200-206,258`) — i.e. it pulls `Q`
   toward the target's **highest-variance** directions. JAWP's objective
   minimizes `tr(Qᵀ Σ_res Q)`, whose minimizer on `St(D,k)` is the
   **bottom-k** eigenvectors of the *residual* covariance
   (`src/models/jawp.py:45-65`), and JAWP's own header names high-variance
   selection as the failure mode to avoid. Same `Q`, opposite criteria.
   **This is a science decision, not a defect:** both may be right if they act
   on different quantities (`Cov(z_target)` vs `Σ_res`) at different times
   (`WSD` resyncs its `Q_target` every `wsd_sync_interval` steps — 100 in
   `defaults.yaml` — while JAWP's objective is optimized every step).
   **The interaction is live in the reference model, not hypothetical:**
   `defaults.yaml` has `use_wsd: true` and `lambda_wsd: 0.01`, and 50 of the
   62 shipped configs resolve with both `use_wsd` and `use_jawp` true,
   including `config/ablations/all_core.yaml` and every leave-one-out row. The
   four *capacity* rungs (`xsmall_30m` … `large_300m`) and the four
   wikitext/tinystories/kaggle JEPA configs set `use_wsd: false`, so the
   scaling ladder is not where this shows up — the ablation table is.
   Recorded, not resolved; the resolution is the owner's.
   `docs/related_work.md` §3.10 has the same finding from the prior-art
   direction.

Recommended use: cite mechanisms by their IMPLEMENTED behavior; treat the
WCP unifying bound as a design sketch until per-theorem rewrites land.
