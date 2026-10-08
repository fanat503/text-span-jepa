# TASK-38 — related-work section

Deliverable: `docs/related_work.md` (new, ~340 lines). No `src/`, `config/`, or
`tests/` file was touched. Branch `agent/task-38`, committed, not pushed.

## The real counts (as instructed: do not trust the header)

| Quantity | Value | Where |
|---|---|---|
| Numbered capabilities in the GWP header | 16 | `mechanisms.py:37-56` |
| `GWP.N_MECHANISMS` | 16 | `mechanisms.py:682` |
| `GWP.N_GROUPS` | 3 | `mechanisms.py:683` |
| Modules in `ALL_MECHANISMS` | **12** | `mechanisms.py:100-113` |
| Group sizes | core 1 / routing 4 / stability 7 | `MechanismBundle.GROUPS` |
| Proof docs | 13 (12 modules + `wip.md`) | `proofs/` |
| Audited matrix rows | 12 | `proofs/IMPLEMENTATION_STATUS.md` |

Verified by running the code, not by reading it:

```
ALL_MECHANISMS: 12 ['jawp','cgn','swip','pcr','spc','wsd','cmc','gac','sta','puc','rdc','wsr']
N_MECHANISMS: 16  N_GROUPS: 3
group sizes: {'core': 1, 'routing': 4, 'stability': 7}
```

Four of the 16 are `JAWPModule` methods, not modules: WIP, Spectral Gap,
Grassmann Optimization, Predictive Rank. Two of those four have **no caller in
`src/`** — `detect_workspace_dimension` and `grassmann_retract` /
`principal_angles` / `subspace_distance` appear only in tests. The trained path
retracts on `St(D,k)`, not `Gr(k,D)`, so "Grassmann Optimization" is not part of
the method.

## Mechanism → literature table

| # | Mechanism | Closest prior work | Verdict |
|---|---|---|---|
| 1 | JAWP | reduced-rank regression + Absil/Edelman/Lewen 2008 retraction | RECOMBINATION |
| 2 | WIP | principal angles / canonical correlations / CCA | RECOMBINATION, circular |
| 3 | Spectral gap | Marchenko–Pastur; Gavish–Donoho hard threshold | RECOMBINATION, dead in training |
| 4 | Grassmann optim. | Edelman/Arias/Smith 1998; Björck–Golub | RECOMBINATION, textbook, unused |
| 5 | Predictive rank | log-det barrier (Vogelstein); Vershynin eff. rank | RECOMBINATION |
| 6 | CGN | Gumbel-Softmax (Jang 2017); Concrete/L0 (Maddox 2017); MoE gating | RECOMBINATION |
| 7 | SWIP | ZCA/PCA whitening (Huang 2019); Barlow Twins; VICReg | RECOMBINATION |
| 8 | PCR | iterative refinement (I-JEPA, DDNM); DEQ; multigrid | RECOMBINATION |
| 9 | SPC | DFT-domain loss weighting; uncertainty weighting (Kendall 2018); GradNorm | RECOMBINATION |
| 10 | WSD | EMA target nets (BYOL, I-JEPA); online PCA / Oja | RECOMBINATION + contradiction |
| 11 | CMC | consistency reg. (Mean Teacher, FixMatch/UDA); BYOL two-view | RECOMBINATION |
| 12 | GAC | gradient-magnitude exploration bonus (RL/ES); dormant-unit reactivation | RECOMBINATION |
| 13 | STA | spectral stability penalties; continual-learning spectral anchoring | RECOMBINATION + design risk |
| 14 | PUC | VICReg variance term; Barlow Twins; log-det diversity | RECOMBINATION, **dead gradient path** |
| 15 | RDC | continual-learning drift penalties; representation stability | RECOMBINATION |
| 16 | WSR | SAM (Foret 2021); manifold-SAM | RECOMBINATION, sam path broken |

## Genuinely novel

**None.** Not one of the twelve modules is defensible as novel in isolation. That
is a statement about prior-art coverage, not about publishability.

Two survive as *combination* claims, both narrow, and I wrote them up as such:

- **JAWP** — restricting a JEPA prediction loss to a learned `k`-dim subspace on
  `St(D,k)` maintained by SVD retraction, with `k` annealed. Every component is
  published; what I could not find is the combination inside a latent predictor.
  Note the Courant–Fischer/Ky Fan result itself is textbook and cannot be
  claimed.
- **SWIP** — applying a covariance-shaping constraint only *outside* a learned
  protected subspace. VICReg and Barlow Twins shape the whole spectrum; I know of
  no prior work that carves out a protected subspace. Weakened by the fact that
  the branch that matters (with a JAWP `Q`) uses a trace-based Jensen
  approximation, not the per-eigenvalue matching the header describes.

## Claims I refused to make

1. No mechanism is novel. No exception.
2. Courant–Fischer optimality is textbook, not this repo's.
3. WIP is not an information-preservation guarantee — its default "exogenous
   feature" input is the target's own top-k PCA basis, so the score is subspace
   overlap against a variance-chosen reference.
4. "Automatic k*, no manual tuning" is unsupported: `k` comes from a cosine
   curriculum, `detect_workspace_dimension` is uncalled, and it hardcodes
   `c_est = 0.5`.
5. The method does not optimize on the Grassmannian — `train.py` retracts on
   `St(D,k)` and every `Gr(k,D)` method is uncalled.
6. SWIP is not scale-invariant (the module's own docstring retracts it).
7. "SWIP is the ONLY method that…" — unsupported universal claim; I have not
   exhaustively surveyed the literature.
8. `wsr_mode: sam` cannot be presented as a working SAM-on-Grassmann contribution.
9. PUC cannot be presented as an active regularizer — see below.
10. Gradient-mode WSR minimizes the gradient *norm*, which is a stationarity
    objective, not sharpness minimization.
11. `lambda_cgn_ortho: 0.01` is wired to a function returning zero.
12. No SigLIP-style sigmoid loss exists in the repo, so no claim in that direction
    is available either way.
13. `VISReg` is labelled "(2026)" with no citation — do not cite it.
14. No ablation *results* claimed; `config/ablations/` was read, not run.
15. No comparison numbers at all.

## Two findings I verified by running the code

These were not in my brief; I found them while checking what each module actually
computes, and both are load-bearing for how the paper should be written.

**PUC's loss cannot train.** With shipped defaults `use_differentiable_entropy=False`
(`puc.py:79`), and that flag is *not exposed* through `TextSpanJEPAConfig`, so
`jepa.py` never passes it. The eigenvalue path is
`F.softplus(self.running_eigenvalues - 5.0)`, reading an EMA buffer with no
autograd edge to `z_pred`. Measured:

```
use_differentiable_entropy=False  loss 0.1401  requires_grad False
use_differentiable_entropy=True   loss 0.1037  requires_grad True   z.grad norm 1.4e-05
```

So PUC contributes a constant to `total_loss`. `TextSpanJEPA.validate()` gates it
and every config key is non-zero, so the wiring *looks* live; the gradient is what
is missing. `docs/plans/2026-09-05-improve-3-mechanism-science-fixes.md:12` records
the un-exposed flag as known — I am not reporting it as new, but I am reporting
that it means the mechanism is inert as shipped, which the related-work section
must state rather than a reviewer discover.

**WSR `sam` mode is degenerate twice over.** The retraction in
`wsr.py:242-247` applies "canonical" QR sign normalization read from
`diag(Q_retracted[:k, :])`, which for a `(D,k)` matrix with `D>k` is not
triangular, so signs are arbitrary and the procedure flips columns instead of
nudging the subspace by `ρ`. The in-tree docstring measures ~4.0 Frobenius instead
of ~0.05 and `tests/test_training_state_guards.py` pins it `xfail`. Independently,
`L_perturbed` and `L_current` are both computed under `no_grad`
(`wsr.py:500-528`), so the returned scalar carries no gradient anyway. Measured:
`wsr sam-mode requires_grad: False, sharpness 0.0`. Shipped default is
`wsr_mode: gradient`.

This is the case the brief asked me to flag: **a retraction whose sign selection is
wrong is not a valid retraction.** For contrast, `JAWPModule.stiefel_retract`'s
polar/SVD step (`jawp.py:306-309`) is correct and correctly cited. The QR
*fallbacks* in `jawp.py:312`, `pcr.py:307`, `spc.py:284` are sign-ambiguous but
only reached in `except` branches and still land on `St(D,k)`, so low severity.

## Contradiction not recorded in the audit matrix

WSD pulls `Q` toward the **top-k eigenvectors of the target covariance**
(`wsd.py:196-215`) — highest-variance directions. JAWP's stated criterion selects
the **bottom-k eigenvectors of the residual** covariance, and its header
explicitly names high-variance directions as the failure mode to avoid
(`jawp.py:100-111`). The two mechanisms pull the same parameter toward opposite
optima and nothing reconciles them. The audit matrix records WSD's unenforced
assumptions and its circular Davis-Kahan reduction; this variance-vs-residual
conflict is separate. Marked UNCLEAR, needs the owner's judgement: either WSD
should track a residual subspace (overlapping STA's job) or a variance subspace
(fighting JAWP).

## Verification

```
$PY tools/rt.py tests/test_config_system.py --slow
571 passed, 21 skipped in 9.93s
```

(`--slow` needed; `rt.py` refuses the file without it.) `ruff check docs` clean;
`black` has nothing to say about a markdown file. No training was run.
