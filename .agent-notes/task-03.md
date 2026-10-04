# TASK-03 — mechanism counts agree with the code

- **status:** done

- **files changed:**
  - `README.md` — replaced the bare "novel mechanisms (16)" with a full enumeration
  - `proofs/README.md` — added the counting-convention section + 16→12 mapping table; added the missing RDC and WSR rows; corrected the header and unifying-principle counts
  - `proofs/IMPLEMENTATION_STATUS.md` — corrected the header count; added the missing WIP row marked UNAUDITED
  - `tests/test_model.py` — renamed one test (and its docstring)
  - `docs/decisions.md` — new entry D-013
  - `.agent-notes/task-03.md` — this file

- **convention chosen:**

  > **12 when counting modules, 16 when counting numbered capabilities, never
  > either unqualified.**

  Stated in full (with justification and the 16→12 mapping table) in
  `proofs/README.md`; recorded as `D-013` in `docs/decisions.md`.

  Why: collapsing to a bare "12" falsifies `GWP.N_MECHANISMS = 16`, which
  `test_gwp_import` asserts; keeping a bare "16" overstates the module count,
  since only 12 things can be constructed, toggled, ablated, or counted by
  `active_mechanisms()`. The split states both numbers truthfully and names the
  four non-modules.

- **counts measured** (every number below was derived by running the code or
  re-deriving it from the files, not read off the card):

| What | Count | Evidence |
|---|---|---|
| `MechanismBundle.ALL_MECHANISMS` | **12** | `src/models/mechanisms.py:100-113`; `len()` = 12 |
| `GROUPS` flattened | **12** | `mechanisms.py:95-99`; zero orphans either direction vs `ALL_MECHANISMS` |
| `GWP.N_MECHANISMS` | **16** | `mechanisms.py:682` — a literal, pinned by `test_model.py:2316` |
| `GWP.N_GROUPS` | **3** | `mechanisms.py:683` — correct, verified, unchanged |
| GWP header numbered items | **16**, contiguous 1..16 | `mechanisms.py:37-56`, one line per number |
| `use_<mech>` keys in `defaults.yaml` | **12** | regex `^  use_[a-z]+:` → 12 |
| leave-one-out `no_<mech>.yaml` | **12** | `config/ablations/`: no_{cgn,cmc,gac,jawp,pcr,puc,rdc,spc,sta,swip,wsd,wsr}.yaml (excludes `no_decoder_loss.yaml`, `no_future_loss.yaml`, which are not mechanisms) |
| proof documents in `proofs/` | **13** | 12 modules + `wip.md`; `modules with NO proof doc: []`, `proof docs with NO module: ['wip']` |
| matrix rows (before) | 12 | header said 13 |
| matrix rows (after) | 13 | 12 numeric + 1 `UNAUDITED` (WIP) |
| `proofs/README.md` table rows (before) | 11 | RDC and WSR absent |
| `proofs/README.md` table rows (after) | 13 | all 13 proof docs listed |
| `__all__` in `mechanisms.py` | 15, all defined | `undefined = []` — verified clean, untouched |
| `test_mechanism_bundle_counts_16` asserts | `active == 12` | `tests/test_model.py:2299` |

**Where "16" comes from** (verified, each line number confirmed by re-reading
the `def` lines): the header numbers 16 items, of which items 1-5 are one
`JAWPModule` plus four of its methods.

| # | Capability | Kind | Location |
|---|---|---|---|
| 1 | JAWP | module | `src/models/jawp.py` |
| 2 | WIP | JAWP method | `workspace_information_preservation` — `jawp.py:574` |
| 3 | Spectral Gap | JAWP method | `detect_workspace_dimension` — `jawp.py:467` |
| 4 | Grassmann Optimization | JAWP methods | `grassmann_retract` `:808`, `principal_angles` `:870`, `subspace_distance` `:921` |
| 5 | Predictive Rank | JAWP method | `predictive_rank_loss` — `jawp.py:1082` |
| 6-9 | CGN, SWIP, PCR, SPC | 4 modules | `mechanisms.py:44-47` |
| 10-16 | WSD, CMC, GAC, STA, PUC, RDC, WSR | 7 modules | `mechanisms.py:50-56` |

  1 + 4 + 4 + 7 = 16 capabilities; 1 + 0 + 4 + 7 = 12 modules.

  Confirmed by inspection: `use_wip` appears **nowhere** in the repo.
  `lambda_predictive_rank` does, and Predictive Rank **is** a trained loss term
  (`jepa.py:810` adds `lambda_predictive_rank * loss_pred_rank`, default `0.0`
  at `defaults.yaml:140`, on-arm `config/ablations/predictive_rank_on.yaml`)
  while having no `ALL_MECHANISMS` entry — so it is invisible to
  `active_mechanisms()`, `mechanism_groups()`, `dependency_dag()` and
  `GWP.summary()`, the last of which prints `Core: ['jawp']`. Documented, not
  fixed: all three sites are under `src/**`.

- **VERIFIED CLEAN — untouched, as instructed:** `GROUPS` vs `ALL_MECHANISMS`
  have zero orphans in either direction; `__all__` (15/15 defined, `import *`
  safe); `retract()` covers all three manifold-bearing parameters.

## verify — full paste

### 1. The renamed test still runs, under its new name

```
PS> & $PY tools\rt.py tests/test_model.py --slow -k "counts_12_modules" -v
rt.py: ...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_model.py -k counts_12_modules -v
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collecting ... collected 151 items / 150 deselected / 1 selected

tests/test_model.py::TestDeepMergeAndDefaults::test_mechanism_bundle_counts_12_modules PASSED [100%]

====================== 1 passed, 150 deselected in 3.03s =======================
```

Collection check, so the rename cannot have silently de-selected the test:

```
PS> & $PY tools\rt.py tests/test_model.py --slow --collect-only -q -k "counts"
rt.py: ...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_model.py --collect-only -q -k counts
rt.py: threads=1  total_budget=90s  slow_ok=True
tests/test_model.py::TestDeepMergeAndDefaults::test_mechanism_bundle_counts_12_modules

1/151 tests collected (150 deselected) in 2.86s
```

### 2. `tests/test_model.py` — green

```
PS> & $PY tools\rt.py tests/test_model.py --slow
rt.py: ...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_model.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 151 items

tests\test_model.py .................................................... [ 34%]
........................................................................ [ 82%]
...........................                                              [100%]

============================== warnings summary ===============================
tests/test_model.py::TestData2VecBaseline::test_forward
tests/test_model.py::TestData2VecBaseline::test_loss_formula_data2vec
tests/test_model.py::TestV011TrainingReadiness::test_compute_loss_unified_interface
tests/test_model.py::TestV011TrainingReadiness::test_data2vec_training_loop
  C:\dev\wt-03\baselines\data2vec_baseline.py:121: UserWarning: data2vec target depth truncated: encoder exposes 2 layers, average_top_k_layers=8
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 151 passed, 4 warnings in 27.35s =======================
```

### 3. `tests/test_config_system.py` — the file that already pins `ALL_MECHANISMS == 12`

```
PS> & $PY tools\rt.py tests/test_config_system.py --slow
tests\test_config_system.py ............................................ [  7%]
........................................................................ [ 20%]
...........................................................s............ [ 32%]
........................................................................ [ 44%]
........................................................................ [ 57%]
........................................................................ [ 69%]
.....ss..........ssssssss..........................................ss... [ 82%]
.......ssssssss......................................................... [ 94%]
...................s..........                                           [100%]

======================= 556 passed, 22 skipped in 9.23s =======================
```

### 4. Lint and format

```
PS> & $PY -m ruff check . --output-format concise
All checks passed!
ruff exit=0

PS> & $PY -m black --check .
All done! ✨ 🍰 ✨
100 files would be left unchanged.
black exit=0
```

### 5. Re-reading the edited files against the measured counts

Every number written into the three docs was re-derived from the edited files
after editing. Output of the re-derivation script:

```
--- A. CLAIMS IN CODE (authoritative, re-derived now) ---
ALL_MECHANISMS entries = 12 (mechanisms.py:100)
N_MECHANISMS line = [682]
GROUPS flatten = 12
mechanisms.py:544 = wip_score, _ = self.jawp.workspace_information_preservation(z_pred, z_target)
jepa.py:810 = + self.config.lambda_predictive_rank * loss_pred_rank
defaults.yaml:140 = lambda_predictive_rank: 0.0
defaults.yaml use_<mech> keys = 12

GWP header numbered items = 16
  file:line -> number: [(37, 1), (38, 2), (39, 3), (40, 4), (41, 5), (44, 6), (45, 7), (46, 8), (47, 9), (50, 10), (51, 11), (52, 12), (53, 13), (54, 14), (55, 15), (56, 16)]
  range = lines 37 - 56
  contiguous 1..16 = True

--- B. CLAIMS I WROTE IN DOCS (re-derived from the edited files) ---
leave-one-out no_<mech>.yaml = 12 ['no_cgn.yaml', 'no_cmc.yaml', 'no_gac.yaml', 'no_jawp.yaml', 'no_pcr.yaml', 'no_puc.yaml', 'no_rdc.yaml', 'no_spc.yaml', 'no_sta.yaml', 'no_swip.yaml', 'no_wsd.yaml', 'no_wsr.yaml']
proof documents = 13
matrix rows = 14 | audited (numeric) = 12 | UNAUDITED = ['WIP']   # 14 = 13 data rows + 1 header row
proofs/README table rows = 13 ['JAWP', 'WIP', 'CGN', 'SWIP', 'PCR', 'SPC', 'WSD', 'CMC', 'GAC', 'STA', 'PUC', 'RDC', 'WSR']
proofs/README numbered-capability table rows = 7   # 7 rows covering 16 capabilities
```

Relative links added by this card all resolve:

```
README.md                          links=4
proofs/README.md                   links=38
proofs/IMPLEMENTATION_STATUS.md    links=3
BROKEN: none
```

## mutation-verdict — what now pins the truth

There are two independent pins, and one of them already existed and is better
than any prose count:

1. **`tests/test_config_system.py::TestAblationGridComplete::test_all_mechanisms_length_is_stable`**
   asserts `len(ALL_MECHANISMS) == 12`, and its own failure message names
   "the ablation grid, the mechanisms.py header, GWP.N_MECHANISMS and
   proofs/IMPLEMENTATION_STATUS.md together". **This is the pin that matters:**
   add or remove a module and this test fails and points at exactly the four
   sites that would go stale, including the file this card edited. It already
   existed; this card did not add it.

   ```
   PS> & $PY tools\rt.py tests/test_config_system.py --slow -k "length_is_stable" -v
   tests/test_config_system.py::TestAblationGridComplete::test_all_mechanisms_length_is_stable PASSED [100%]

   ====================== 1 passed, 577 deselected in 2.35s =======================
   ```

2. **`tests/test_model.py::TestGWPFramk::test_gwp_import`** asserts
   `GWP.N_MECHANISMS == 16`. Out of scope to change (it is a code assertion,
   not prose), so the docs now *explain* the 16 instead of fighting it. If
   someone renumbers the header, this test fails.

Claims that become false if the code changes again, and what catches each:

| If this changes | Which claim goes false | What catches it |
|---|---|---|
| a module added / removed | "12 modules" in all three docs; the 12-row matrix; the `proofs/README.md` table | `test_all_mechanisms_length_is_stable` fails, message names the four sites |
| a module renamed | the mapping table's `Kind`/`Code` columns and the README enumeration | `test_all_mechanisms_length_is_stable` fires only on a *length* change; a pure rename would **not** be caught — this is the known soft spot, see risks |
| `no_<mech>.yaml` added or removed | "the twelve leave-one-out configs" in `proofs/README.md` | `TestAblationGridComplete::test_every_mechanism_has_both_arms` (file existence, not the count) |
| one of the four JAWP methods promoted to a module | "16 = 12 modules + 4 JAWP methods"; the whole D-013 decomposition | no test; human review of D-013's "Reverses if" clause |
| Predictive Rank given an `ALL_MECHANISMS` entry + `use_*` flag | "Predictive Rank is invisible to `active_mechanisms()`" in both docs | `test_all_mechanisms_length_is_stable` fires (12 → 13) |
| `GWP.N_MECHANISMS` changed | nothing in the docs — the docs already say 16 is the *capability* count, which would then be wrong | `test_gwp_import` |

The one thing this card did **not** achieve: no test parses the markdown, so
the prose numbers themselves are not machine-pinned. The matrix row count and
the `proofs/README.md` table row count are human-checked only.

## не_сделано

- **The `src/` sites were not touched, as instructed.** `mechanisms.py:9,15,32`
  and the `GWP` docstring (`mechanisms.py:83,653,661`) still say "16
  mechanisms" unqualified, and `AGENTS.md:8-13,75` repeats it. All are now
  *explained* by D-013 and `proofs/README.md`, but a reader who opens only
  `mechanisms.py` still sees a bare "16 mechanisms". Fixing that needs the
  `src/**` owner. **This is the largest remaining inconsistency.**
- **The Predictive Rank visibility gap is documented, not closed.** Adding a
  13th `ALL_MECHANISMS` entry and a `use_predictive_rank` key would make
  `GWP.summary()` honest, but it would break
  `test_all_mechanisms_length_is_stable` (12 → 13), change the ablation grid,
  and put a new module in `GROUPS`. `src/**` and `config/**` were both
  forbidden. **A human decision, not a doc fix.**
- **`proofs/README.md:106` still says "All theorems are computationally
  verified in the test suite".** That overstates, and it contradicts the
  matrix (whose JAWP row says "cited verification tests absent"). It is not a
  count, so it was outside this card's mandate and was left alone rather than
  silently edited. **Flagged, not fixed — recommend a follow-up.**
- **The `no_<mech>.yaml` descriptions say "the other eleven enabled"** — true
  for 11 of the 12, but `no_jawp.yaml` enables ten (see D-007: WSD is
  constructed only under `use_wsd and use_jawp`). `config/**` was forbidden.
- **WIP's mathematics was not audited**, only labelled unaudited. Not asked
  for and would need a real adversarial pass.

## risks

- **Disclosed deviation from "change nothing else in that file".** I changed
  two things in `tests/test_model.py`, not one: the function name *and* the
  one-line docstring immediately under it. The docstring read
  `"""MechanismBundle must expose all 16 mechanisms."""`, so renaming the
  function to `..._counts_12_modules` without touching it would have left a
  test whose name and docstring contradict each other and its own assertion —
  reintroducing the exact defect this card exists to remove. No assertion,
  no test body and no other test was touched. Reversible in one line if the
  coordinator disagrees.
- **Docstring-only claims are human-checked.** The audit-matrix counts (the
  3/7/5, 6/6/3 … numbers) were carried over verbatim; I did not re-derive
  them, because re-auditing 12 mechanisms is not this card and the matrix is
  dated 2026-08-24. If any mechanism changed since that date, the row is stale
  in a way this card cannot detect.
- **A pure module *rename* is not machine-caught.** The stability test checks
  `len(ALL_MECHANISMS) == 12`, not its contents, so renaming `rdc` to
  `rdc2` keeps the count at 12 and would silently invalidate the module list
  in `README.md`, `proofs/README.md` and the matrix.
- **`proofs/README.md` is now long** and reads as a specification in places. A
  reviewer who wants the mechanism list may not read the convention, which is
  the one thing they need. Accepted trade-off: the alternative is a number
  that is wrong or unqualified.
- **Numbering in the mapping table is coupled to `mechanisms.py:37-56`.** If
  someone reorders that header, the table's `#2`…`#16` labels go stale with no
  test failing.
