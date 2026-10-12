# THEORY_AND_PROOFS.md — agent report

Branch `agent/theory`, based on `main` @ `6832d35`. Worktree `C:\dev\wt-th`.
Owned file: `docs/THEORY_AND_PROOFS.md` (new). Nothing else in the repo edited.

## What was produced

`docs/THEORY_AND_PROOFS.md` — per-mechanism separation of (a) the proof's claim,
(b) whether the code satisfies the proof's ASSUMPTIONS, (c) the consequence for
anyone reading a number the mechanism prints. Plus a dry divergence table
(41 tagged items) and a "what is NOT wrong" section.

No attempt to make any mechanism look sound. Where the proof is right and the
code is right, that is said. Where the *proof document itself* is wrong, that is
recorded as a distinct type (`P`) from a hypothesis failing in the code (`H`) and
from a diagnostic whose meaning does not match its number (`S`).

## Mechanisms audited: 12 of 12

jawp, wsd, swip, pcr, cgn, spc, sta, gac, cmc, puc, rdc, wsr — all read as a
proof+code pair. `proofs/IMPLEMENTATION_STATUS.md` and `proofs/README.md` read in
full for the 12-vs-16 convention and for recorded headlines.

## Verified by execution (scripts in `.agent-notes/`, each asserts `src.__file__`)

1. **JAWP `predictive_relevance` is normalised by two different axes and is
   inverted.** `ws_pred_error` is a mean over `N·k` (`jawp.py:463`),
   `bg_pred_error` over `N·D` (`:462`), metric is `max(0, 1 − ws/bg)` (`:503`) ⇒
   `= max(0, 1 − (D/k)·w)`. Verified to <1e-4 at 8 values of `w` (D=64,k=6), and
   both endpoints constructed exactly (`U` verified orthogonal to that `Q`):

   | residual | `loss_workspace` | `predictive_relevance` |
   |---|---|---|
   | `w=0` — entirely **outside** `span(Q)` | `1.841e-11` (prediction **exact**) | **1.0000** |
   | `w=1` — entirely **inside** `span(Q)` | `1.037e+00` (prediction **worst**) | **0.0000** |

   It reads **maximal when workspace prediction is exact** (the residual is
   orthogonal to the workspace, so there is nothing to predict there) and
   **zero when workspace prediction is worst**. The `0.0` threshold is
   `w ≥ k/D` = chance level.
   **This is recorded nowhere in the repo**, and it undercuts load-bearing
   evidence: `proofs/wsd.md:30,263`, `proofs/jawp.md:264` and the code comment at
   `jawp.py:78` all read `jawk_predictive_relevance → 0` as "workspace collapsed".
   I did NOT disprove the WSD/JAWP antagonism — only that this symptom cannot
   support it.

2. **WSD `wsd_steady_state_error` is an exact tautology.**
   `ν = running_drift/T`, `λ = 1/T` ⇒ `ν/λ = running_drift`. Difference 0.0
   across 12 steps.

3. **WSD `wsd_lambda_estimate = 1/sync_interval`, not `lambda_wsd`** (`wsd.py:322`).
   They coincide at the default (0.01 vs 1/100) which hides it; the `wsd_on`
   ablation at `lambda_wsd: 0.1` is off 10×.

4. **WSD's `clamp(min=0)` reports a 3×-scaled `Q` as perfectly synchronised**
   (drift_sq 0.000000 vs 3.354395 orthonormal).

5. **WSD/JAWP reachability re-measured: 56 of 68 configs** under `config/`
   resolve `use_wsd: true` AND `use_jawp: true`. `proofs/wsd.md:226-237` and
   `proofs/jawp.md:261-262` still carry the stale "every config ships
   `use_wsd: false`" text, contradicted by their own addenda.

6. **SWIP property 2 is false for the executed code.** Background spectrum with
   8× anisotropy whose *mean* matches σ_target: proof form = 4.078781,
   executed trace form = 0.0. The term vanishes for any mean-matching spectrum.

7. **SPC "uniform weights reproduce standard MSE/B" is inverted by B².**
   Measured `SPC = B·MSE` for B ∈ {1,2,4} (ratios 1.0000 / 2.0000 / 4.0000).
   `proofs/README.md:137` records `test_spc_subsumes_uniform_mse` asserting
   `0.8 < SPC/(n_bands·MSE) < 1.2` — the **test agrees with the code**, the
   **document is the wrong half**.

8. **STA carries no gradient** (`requires_grad == False`, measured) — `eig_cur`
   is from `z.detach()` (`sta.py:286`).

9. **WSR carries no gradient in either mode** (measured). `wsr_spectral_sharpness`
   is character-identical to `wsr_sharpness` (`wsr.py:424` vs `:428`); Theorem 3's
   actual spectral term is never computed.

10. **WSR `mode='sam'` retraction, over 12 seeds:** subspace displacement
    `0.0707 = √2·ρ` on 12/12 (a correct retraction is nonexpansive ⇒ ≤ ρ), so QR
    is not a retraction. Frobenius 3.46–4.00 on 12/12.

11. **PUC entropy estimator is nearly blind.** `puc_entropy` reads 41–51 nats for
    every spectrum tested, including a true entropy of **−117.9** — a ~159-nat
    error. True entropies −36.4 and −117.9 give losses 0.1025 and 0.1028.

12. **RDC transient bound underflows.** `(1−0.01)^1000 = 4.3e-05`,
    `(1−0.01)^10000 = 2.2e-44`, `(1−0.1)^10000 = 0.0`. Reads as "drift provably
    vanishes"; it is float64 underflow.

13. **PUC's TASK-39 fix genuinely works** — confirmed, not assumed. `eigvalsh` on
    a non-detached batch covariance (`puc.py:261-263`) ⇒ `log_det_tracked` (`:293`)
    ⇒ `loss_tensor` (`:296`). **The fixed path does have the autograd edge.**

## Answering the card's specific questions

- **WSD/JAWP opposition** — confirmed real and live (live view, same backward
  pass, `jepa.py:1088-1093`); 56/68 configs reach it. Kept as an **open owner
  decision**, not a defect. Added the new caveat that the cited *symptom*
  (`predictive_relevance → 0`) is not diagnostic.
- **PUC autograd edge** — **yes, the fixed path has it.** Verified from the code,
  with the caveat that the switch gating the loss (`:304`) is itself detached via
  `.item()` at `:283`.
- **WSR retraction** — escalated, **not fixed**, kept as an open decision. I also
  *corrected the xfail's severity description*: the sign vector is essentially
  uniform on these 12 seeds (harmless for the subspace); what actually fails is
  the ρ-semantics (√2·ρ, not ρ).
- **SWIP / PCR / CGN "objects never implemented"** — CGN confirmed and sharpened:
  partition of unity holds for `gate_base` (`cgn.py:278-280`) then is destroyed by
  a data-dependent modulation (`:284`) and a clamp (`:287`); and the code makes
  the gates exact mirrors (`g_m = 1 − g_v`, `:264`), which *maximises* the
  redundancy the proof needs to be ~0. PCR's document is already correct — it
  retracts its v1 theorems in place; only the external O(D) precondition remains.

## Corrections I made to my own work

Three checks were wrong or mis-worded on first run and were fixed rather than
reported:

- The first SWIP construction put the spread outside the background block
  (min == max), giving a false pass. Rebuilt with the spread inside → D-SWIP-1.
- The JAWP endpoint construction used an **independent** random `U` as the
  complement of `Q`; two independent orthonormal draws are not mutually
  orthogonal, so `Uᵀ Q ≠ 0` and both endpoints came out `0.0000`. Rebuilt `U`
  from `qr(I − QQᵀ)` with an explicit orthogonality assertion. This also
  exposed that my original wording was wrong: at `w = 1` workspace prediction is
  the **worst**, not perfect. The corrected finding is *stronger* and simpler —
  the metric is simply inverted. Document and report both corrected.
- I initially characterised the WSR retraction defect as "seed-dependent".
  Twelve seeds say otherwise: the Frobenius and subspace failures are universal.
  Corrected in the document.

I also found mid-task that `import src` could resolve to a **different clone**
(`C:\Users\...\tmp\clone-check-1`) via a stale editable install / `__pycache__`.
Every verification script now asserts `src.__file__` before executing.

## Could not determine, and why

- **JAWP**: whether the iterate `Q` actually attains the `St(D,k)` minimiser.
  The corollary compares minima, not iterates; nothing measures the gap.
- **JAWP/WSD**: the measured crossover `λ_wsd ≈ 0.42` is scale- and
  schedule-specific (D=64, k=6, random tokens). Not re-measured; the card
  forbids training and a short run would not be portable anyway.
- **GAC**: whether `τ_grad = 1e-4` is calibrated against the real gradient scale.
  Depends on the other eleven `lambda_*`; a single training sample would not
  settle it.
- **RDC**: whether `Q`'s orthonormality is violated in a live run. Would need an
  instrumented forward pass.
- **STA**: whether `update_interval` makes reference and current spectra diverge
  in practice. Did not read the default.
- **PCR**: whether `pcr_ortho_score` is asserted anywhere in `tests/`.
- **SWIP**: whether both code paths agree on the background term's scale.

## Deliberately not done

- No edits to `src/`, `proofs/**`, or any existing file. Several findings have
  obvious one-line fixes; all withheld — which side is the intended science is
  the owner's call per `AGENTS.md`.
- No re-derivation of the Verified/Divergent/Gaps counts.
- No training. No full test suite (and a green one would not have caught
  D-JAWP-1, D-SWIP-1 or D-WSR-2 — three findings here are about numbers nothing
  asserts).

## Disagreements with recorded headlines

Stated explicitly, no replacement counts invented:

- **`proofs/wsd.md:30,263` / `proofs/jawp.md:264` / `jawp.py:78`** treat
  `jawk_predictive_relevance → 0` as evidence of workspace collapse. The metric
  cannot support that reading (finding 1).
- **`proofs/wsd.md:226-237` / `proofs/jawp.md:261-262`** are stale on config
  reachability, contradicted by their own addenda and by measurement (finding 5).
- **`proofs/spc.md:46-47`** is wrong where its own cited test is right (finding 7).
- **`tests/test_training_state_guards.py:510-517`** overstates the WSR retraction
  severity for the seeds measured (finding 10).
- **`proofs/IMPLEMENTATION_STATUS.md:107-116` pattern 2** ("EMA proxies") is
  accurate for WSR/RDC/WSD but does **not** describe PUC's default path, whose
  eigenvalues come from `eigvalsh` of this batch. Noted against the pattern; the
  PUC row itself makes no such claim.

## Scratch files

`.agent-notes/theory-contract.md`, `.agent-notes/verify_{jawp,wsd,rest,rest2,puc,spc}.py`
— untracked, not part of the deliverable, reproducible by the owner.