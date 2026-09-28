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
| 2 | WIP | JAWP method | `workspace_information_preservation` (`jawp.py:574`) |
| 3 | Spectral Gap | JAWP method | `detect_workspace_dimension` (`jawp.py:467`) |
| 4 | Grassmann Optimization | JAWP methods | `grassmann_retract` (`jawp.py:808`), `principal_angles` (`jawp.py:870`), `subspace_distance` (`jawp.py:921`) |
| 5 | Predictive Rank | JAWP method | `predictive_rank_loss` (`jawp.py:1082`) |
| 6–9 | CGN, SWIP, PCR, SPC | modules | [`cgn.py`](../src/models/cgn.py), [`swip.py`](../src/models/swip.py), [`pcr.py`](../src/models/pcr.py), [`spc.py`](../src/models/spc.py) |
| 10–16 | WSD, CMC, GAC, STA, PUC, RDC, WSR | modules | [`wsd.py`](../src/models/wsd.py), [`cmc.py`](../src/models/cmc.py), [`gac.py`](../src/models/gac.py), [`sta.py`](../src/models/sta.py), [`puc.py`](../src/models/puc.py), [`rdc.py`](../src/models/rdc.py), [`wsr.py`](../src/models/wsr.py) |

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
  loss ([`src/models/jepa.py:810`](../src/models/jepa.py)), default `0.0`
  ([`defaults.yaml:140`](../defaults.yaml)), with an on-arm at
  `config/ablations/predictive_rank_on.yaml`. Because it has no
  `ALL_MECHANISMS` entry it is invisible to `active_mechanisms()`,
  `mechanism_groups()`, `dependency_dag()` and `GWP.summary()` — which report
  `Core: ['jawp']`. That is a known visibility gap, not a second module count.
- **WIP has no `use_wip` key anywhere in the repo.** It is reachable only
  through `MechanismBundle.compute_capacity_bound`
  ([`src/models/mechanisms.py:544`](../src/models/mechanisms.py)), a composite
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

**[`HYPOTHESES.md`](HYPOTHESES.md)** — 10 pre-experimental hypotheses registered before running any training experiments, following top-lab standards (analogous to clinical trial pre-registration).

## Proof Standards

Each proof document follows this structure:
1. **Statement** — Formal theorem statement with all assumptions
2. **Proof** — Complete step-by-step derivation
3. **Discussion** — Limitations, connections to other mechanisms, practical implications
4. **Novelty** — Explicit comparison with closest prior art

## Verification

All theorems are **computationally verified** in the test suite:
- JAWP Q orthonormality error: < 1e-5
- SPC Parseval's reconstruction: relative error < 1e-4
- CGN partition of unity: exact (by construction)
- SWIP loss non-negative: ✅
- CMC loss non-negative + stability bound: ✅
- GAC No Dead Zones + exploration ratio bounded: ✅
- STA W1 metric (triangle inequality, symmetry, non-negativity): ✅
- STA Davis-Kahan bound: ✅
