# Implementation Status of Proofs (audited 2026-08-24)

Automated adversarial comparison of every proof document against its
implementation (13 mechanisms). Treat proofs/ as DESIGN documents with
audited gaps, not as certified descriptions of the running code.

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
