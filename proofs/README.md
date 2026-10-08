# Mathematical Proofs & Formal Analysis

This directory contains formal mathematical proofs for the **12 GWP mechanism
modules**, plus one proof for the **WIP method of the JAWP module** — thirteen
proof documents in total. Organized by mechanism.

## Counting convention — read this before quoting a number

GWP is documented in two different units, and conflating them is the entire
source of the old "13 mechanisms" / "16 mechanisms" disagreement in this file.

- **12 modules.** `MechanismBundle.ALL_MECHANISMS`
  ([`src/models/mechanisms.py:100`](../src/models/mechanisms.py)) is the
  authority for this number. Twelve is also what the twelve `use_<mech>` keys
  in [`defaults.yaml`](../defaults.yaml), the twelve leave-one-out configs in
  [`config/ablations/`](../config/ablations/) and the twelve audited rows of
  [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) count. Everything
  that is trained, toggled or ablated is one of these twelve.
- **16 numbered capabilities.** The GWP header numbers sixteen items
  ([`src/models/mechanisms.py:37-56`](../src/models/mechanisms.py)) and
  `GWP.N_MECHANISMS = 16` ([`src/models/mechanisms.py:682`](../src/models/mechanisms.py))
  is pinned to that numbering. The four items beyond the twelve modules are
  **not** modules — they are methods of `JAWPModule`.

| # | GWP capability | Kind | Code |
|---|---|---|---|
| 1 | JAWP | module | [`src/models/jawp.py`](../src/models/jawp.py) — `JAWPModule` |
| 2 | WIP | JAWP method | `workspace_information_preservation` (`jawp.py:659`) |
| 3 | Spectral Gap | JAWP method | `detect_workspace_dimension` (`jawp.py:552`) |
| 4 | Grassmann Optimization | JAWP methods | `grassmann_retract` (`jawp.py:893`), `principal_angles` (`jawp.py:955`), `subspace_distance` (`jawp.py:1005`) |
| 5 | Predictive Rank | JAWP method | `predictive_rank_loss` (`jawp.py:1166`) |
| 6–9 | CGN, SWIP, PCR, SPC | modules | [`cgn.py`](../src/models/cgn.py), [`swip.py`](../src/models/swip.py), [`pcr.py`](../src/models/pcr.py), [`spc.py`](../src/models/spc.py) |
| 10–16 | WSD, CMC, GAC, STA, PUC, RDC, WSR | modules | [`wsd.py`](../src/models/wsd.py), [`cmc.py`](../src/models/cmc.py), [`gac.py`](../src/models/gac.py), [`sta.py`](../src/models/sta.py), [`puc.py`](../src/models/puc.py), [`rdc.py`](../src/models/rdc.py), [`wsr.py`](../src/models/wsr.py) |

> **All six `jawp.py` line numbers in that table were re-derived on 2026-10-08
> (TASK-41); every one of them had drifted by 84–85 lines.** A `file.py:NNN`
> citation is a number in prose with the same failure mode as any other: it
> stops being true the moment anything above it is inserted, and nothing
> notices. If you edit `src/models/jawp.py`, re-derive the row.

**Convention adopted: quote 12 when counting modules and 16 when counting
numbered capabilities, and never use either number unqualified.** The split is
deliberate. Collapsing to a bare "12" would falsify
`GWP.N_MECHANISMS == 16`, which
`tests/test_model.py::TestGWPFramk::test_gwp_import` asserts; keeping a bare
"16" would overstate the module count, because `ALL_MECHANISMS` is what
`active_mechanisms()`, `mechanism_groups()`, `dependency_dag()` and
`GWP.summary()` actually report.

Four consequences of that split, stated because they are easy to misread:

- **Predictive Rank is a trained loss term with no module entry.**
  `TextSpanJEPA` adds `lambda_predictive_rank * loss_pred_rank` to the total
  loss ([`src/models/jepa.py:858`](../src/models/jepa.py)), default `0.0`
  ([`defaults.yaml:198`](../defaults.yaml)), with an on-arm at
  `config/ablations/predictive_rank_on.yaml`. Because it has no
  `ALL_MECHANISMS` entry it is invisible to `active_mechanisms()`,
  `mechanism_groups()`, `dependency_dag()` and `GWP.summary()` — which report
  `Core: ['jawp']`. That is a known visibility gap, not a second module count.
- **WIP has no `use_wip` key anywhere in the repo.** It is reachable only
  through `MechanismBundle.compute_capacity_bound`
  ([`src/models/mechanisms.py:540`](../src/models/mechanisms.py)), a composite
  diagnostic the audit marked R15.
- **Spectral Gap and Grassmann Optimization are machinery, not losses.**
  `detect_workspace_dimension` selects the active rank; `grassmann_retract`
  holds the Stiefel/Grassmann constraint called by `retract()`;
  `principal_angles` and `subspace_distance` are diagnostics.
- **Only WIP has its own proof document among the four**, and it is the only
  one of the thirteen proof documents with no row in
  [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) — it is marked
  **UNAUDITED** there. Spectral Gap, Grassmann Optimization and Predictive
  Rank are argued inside [`jawp.md`](jawp.md).

## Unifying Principle

**[`unifying_principle.md`](unifying_principle.md)** — The Workspace-Conditioned Prediction (WCP) framework from which all 12 module mechanisms are derived as instances of a single optimization principle. This is the central theoretical contribution.

## Mechanism Proofs

Thirteen documents: one per module (12) plus `wip.md` for the JAWP method of
capability #2. See the counting convention above for why that is 12 + 1 and not
13 modules.

| Mechanism | File | Core Theorem |
|-----------|------|-------------|
| JAWP | [`jawp.md`](jawp.md) | Courant-Fischer optimality on St(D,k) |
| WIP | [`wip.md`](wip.md) | Workspace Information Preservation (contradiction + regularity) — **unaudited**, see the matrix |
| CGN | [`cgn.md`](cgn.md) | Information Routing + Partition of Unity |
| SWIP | [`swip.md`](swip.md) | Selective Spectral Shaping (log-eigenvalue matching) |
| PCR | [`pcr.md`](pcr.md) | Cascade Capacity theorem (orthogonal recovery) |
| SPC | [`spc.md`](spc.md) | Parseval's equality + simplex-constrained allocation |
| WSD | [`wsd.md`](wsd.md) | Drift Bound theorem (exponential convergence ODE) |
| CMC | [`cmc.md`](cmc.md) | Stability theorem (Cauchy-Schwarz bound) |
| GAC | [`gac.md`](gac.md) | No Dead Zones theorem (exploration guarantee) |
| STA | [`sta.md`](sta.md) | Davis-Kahan stability + Wasserstein-1 metric |
| PUC | [`puc.md`](puc.md) | Minimax optimality (Jaynes + Donsker-Varadhan) |
| RDC | [`rdc.md`](rdc.md) | Drift Compensation Bound (stationary + transient) |
| WSR | [`wsr.md`](wsr.md) | Workspace Generalization Bound + PAC-Bayes bound minimizer |

## Pre-Registered Hypotheses

**[`HYPOTHESES.md`](HYPOTHESES.md)** — **twelve** pre-experimental hypotheses
(H1–H12) registered before running any training experiments, following top-lab
standards (analogous to clinical trial pre-registration).

**None has been run, and three of them cannot be evaluated as written.**
`docs/results/README.md` establishes that no run any config in this repo can
currently produce emits a downstream metric: `src/eval/probes.py` has a working
`LinearProbe` but nothing in `src/train.py` calls it. H2, H10 and H12 state their
predictions in terms of "linear probe accuracy", so their **Predictions** lines
are unmeasurable today, not merely untested. The hypotheses are pre-registration
and are deliberately left unedited — a prediction edited after the fact is not a
pre-registration. Correct the *evaluation plan*, not the prediction.

## Proof Standards

Each proof document follows this structure:
1. **Statement** — Formal theorem statement with all assumptions
2. **Proof** — Complete step-by-step derivation
3. **Discussion** — Limitations, connections to other mechanisms, practical implications
4. **Novelty** — Explicit comparison with closest prior art

## Verification

**This is a partial list, not a blanket claim.** An earlier version of this file
opened with "All theorems are computationally verified in the test suite", which
contradicted the matrix's own JAWP row ("cited verification tests absent") and
`jawp.md`'s own banner. Of the thirteen proof documents: **six** have a
property test that pins what the document claims (JAWP, SPC, SWIP, CMC, STA, and
GAC partially), **six** have none (CGN, PCR, WSD, PUC, RDC, WSR), and `wip.md`
is **UNAUDITED** in the matrix and claims nothing that is tested.
Verified 2026-10-08 (TASK-41):

| claim | pinned by | status |
|---|---|---|
| JAWP `Q` orthonormality, `atol=1e-5` on `QᵀQ − I` | `tests/test_jawp.py::TestJAWPCore::test_stiefel_retract_keeps_orthonormal` | ✅ exists — but note the name: `jawp.md`'s own Verification section names `test_q_orthonormality`, `test_courant_fischer`, `test_wip_preservation`, `test_stiefel_retraction` and `test_predictive_rank`, and **none of those five exists**. The retraction property is tested under a different name. |
| SPC Parseval reconstruction | `tests/test_spc.py::TestSPCTheorem::test_spc_subsumes_uniform_mse` | ⚠️ exists, but the **tolerance was never `< 1e-4`** and is not now: the test asserts `0.8 < SPC/(n_bands·MSE) < 1.2` — a 20 % band. The old "relative error < 1e-4" in this file described a precision the suite never pinned. The `1e-4` figures that *are* pinned here are the DCT-basis orthonormality checks (`test_spc.py:131,184,206`), which are a different property. |
| CGN partition of unity | — | ❌ no test references partition/unity; "exact by construction" is a code-reading claim, not a test |
| SWIP loss non-negative | `tests/test_swip.py::TestSWIPCore::test_swip_loss_is_nonnegative` | ✅ exists |
| CMC loss non-negative + stability bound | `tests/test_cmc.py` (incl. `test_triangle_inequality_holds`) | ✅ exists |
| GAC exploration ratio bounded | `tests/test_gac.py::TestGACCore::test_starved_count` | ⚠️ partial — the *count* and *fraction* are pinned; the `0 ≤ ρ ≤ 1` bound is not asserted |
| STA `W₁` metric + Davis-Kahan | `tests/test_sta.py::TestSTAMathematical`, `::TestSTADavisKahan`, `::TestSTABasic`, `::TestSTAWasserstein`, `::TestSTAIntegration` | ✅ all five named in `sta.md` exist |
| PCR, WSD, PUC, RDC, WSR | — | ❌ no proof-level verification claimed |

So: **the claims in the middle column are pinned; the claims in the last row are
not.** Read the matrix for per-claim verdicts — it is the authority, not this
table.
