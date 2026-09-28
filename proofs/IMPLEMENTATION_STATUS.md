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

| Mechanism | Verified | Divergent | Gaps | Headline issue |
|---|---|---|---|---|
| WSR | 3 | 7 | 5 | detached loss cannot minimize the claimed objective; bound uses EMA proxies |
| WSD | 6 | 6 | 3 | orthonormality/lambda-coupling assumptions not enforced; Davis-Kahan bound circular |
| SWIP | 4 | 6 | 5 | theorem term 2 missing; scale-invariance false; dead conditional fixed in R11 |
| STA | 3 | 6 | 3 | W1(current,ref) was identically zero (sync refresh bug) — fixed in R11; DK reduction invalid |
| SPC | 3 | 5 | 5 | proved simplex/norm-squared object differs from implemented floored-softmax MSE |
| PCR | 2 | 7 | 3 | proof analyzes linear Stiefel cascade; code is gated MLP cascade |
| PUC | 2 | 8 | 5 | headline formula is dead code; executed loss matches neither stated form |
| RDC | 2 | 6 | 4 | transient bound internally inconsistent in proof, falsely justified in header |
| GAC | 3 | 6 | 4 | No-Dead-Zones theorem does not match detached-input batch-mean implementation |
| CGN | 4 | 5 | 5 | complementary-gates identity not what the module computes |
| CMC | 4 | 6 | 2 | stability theorem misstated (averaged vs pointwise); reuse_encoder path is a stub |
| JAWP | 1 | 6 | 4 | time-varying curriculum k vs fixed-k theorems; cited verification tests absent |
| WIP | **UNAUDITED** | **UNAUDITED** | **UNAUDITED** | **UNAUDITED** — no adversarial pass was run for this capability, so this row records no claim counts. WIP is the only proof document with no row in this matrix. [`wip.md`](wip.md) lacks the inline audit banner, though so do `cgn.md`/`pcr.md`/`spc.md`, which *are* audited above — the absence of the banner is not what marks a capability unaudited. `JAWPModule.workspace_information_preservation` (`src/models/jawp.py:574`) also has no `use_wip` key, so no config arm exercises it. Treat every claim in `wip.md` as unverified. |

## Cross-mechanism patterns

1. Loss formulas diverge from their stated form after refactors
   (PUC dead formula, SPC loss type, SWIP missing term).
2. Bounds are evaluated on EMA-smoothed proxies instead of theorem inputs
   (WSR, RDC, WSD).
3. Assumptions used by proofs (orthonormality, exact simplexes, fixed k)
   are not enforced by the corresponding modules.
4. Several theorems prove properties of objects that were never implemented
   (PCR cascade form, CGN complementary gates).

Recommended use: cite mechanisms by their IMPLEMENTED behavior; treat the
WCP unifying bound as a design sketch until per-theorem rewrites land.
