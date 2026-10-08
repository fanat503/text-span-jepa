# text-span-jepa — agent rules

Project rules. These **override** the global `~/.config/opencode/AGENTS.md`.

## What this repo is

A self-supervised JEPA (Joint-Embedded Predictive Architecture) research codebase.
The model predicts **in latent space** — not token reconstruction.

GWP (Grassmann Workspace Prediction) is documented in two units, and conflating
them is the whole source of the old "13 / 16" disagreement. **Quote 12 when
counting modules and 16 when counting numbered capabilities, and never either
unqualified** — the convention, its justification and the 16→12 mapping table are
in [`proofs/README.md`](proofs/README.md), recorded as decision D-013 in
[`docs/decisions.md`](docs/decisions.md). In short: **12 modules** (the twelve
that are constructed, toggled and ablated — `MechanismBundle.ALL_MECHANISMS`)
and **16 numbered capabilities** (the twelve modules plus four *methods of
`JAWPModule`*: WIP, Spectral Gap, Grassmann Optimization, Predictive Rank).
`GWP.N_MECHANISMS = 16` counts the numbered capabilities, not the modules.

The 12 modules, in their three groups:

- **Core** — workspace construction: `jawp`
- **Routing** — information flow: `cgn`, `swip`, `pcr`, `spc`
- **Stability** — workspace integrity: `wsd`, `cmc`, `gac`, `sta`, `puc`, `rdc`, `wsr`

## Environment (matters — default `python` on this machine is broken)

The `python` on `PATH` resolves to a broken shim at `D:\Ilya\программы\python.exe`
(`ModuleNotFoundError: No module named 'encodings'`). **Always** use the
working interpreter:

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
```

Setup is already done (`pip install -e ".[dev,eval]"`). Verify with
`& $PY -c "import src, sklearn, scipy"`.

**Test count.** `1960` tests are *collected* on this tree
(`pytest tests --collect-only -q`, commit `9d5b0f9`). The most recent full CI
*execution* measured **1910 passed, 21 skipped, 1 xfailed, 0 failed in 65.66s**
on Linux / Python 3.11 at commit `9732de2` — 16 commits behind this tree, so it
is a floor, not the current number. Re-derive the count rather than trusting a
number written here; a stale baseline is worse than none, because a count
regression is invisible against a count nobody re-measured.

**Do not quote a local wall clock for the full suite.** It saturates every core
on this box, and the owner is using it. Take the number from CI (decisions
D-001/D-002 in [`docs/decisions.md`](docs/decisions.md)). CPU-only host:
`torch 2.13.0+cpu`, Python 3.10, no CUDA. Never write a test that silently
requires a GPU.

## Commands

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
& $PY -m pytest -q                        # full suite (saturates the box; prefer CI)
& $PY tools\rt.py tests\test_sta.py       # one file, via the runner
& $PY tools\rt.py tests\test_sta.py --slow   # include the slow-marked tests
& $PY -m pytest -q -k "mechanism_name"    # by keyword
& $PY -m pytest tests --collect-only -q   # count the suite without running it
& $PY -m ruff check .                     # lint
& $PY -m black --check .                  # format check
& $PY -m src.train --fname config/scaling/xsmall_30m.yaml   # smoke train
```

CPU-only training smoke test must stay under a couple of minutes. Do not
attempt `large_300m` or `kaggle` configs as a validation step.

## THE critical invariant: proofs are DESIGN DOCS, not specs

`proofs/IMPLEMENTATION_STATUS.md` is an audited matrix that records, per
mechanism, how many claims are **Verified**, **Divergent**, or **Gaps**. As of
the 2026-08-24 audit, every mechanism has divergences. The documented
cross-cutting failure patterns are:

1. Loss formulas diverge from their stated form after refactors (PUC dead
   formula, SPC loss type). SWIP's missing term 2 was in this list; it was
   implemented in R11 and the claim retracted, so it is gone from the list.
2. Bounds are computed on EMA-smoothed proxies instead of theorem inputs
   (WSR, RDC, WSD).
3. Proof assumptions (orthonormality, exact simplexes, fixed `k`) are not
   enforced by the modules.
4. Some theorems prove properties of objects never implemented (PCR cascade
   form, CGN complementary gates).

**Therefore: never "fix code to match the proof" without an explicit human
decision.** The proof may be aspirational; the code may be the deliberate
choice. Escalate instead.

When you touch a mechanism:
- Read `proofs/<mech>.md` and `src/models/<mech>.py` **together**.
- Check `proofs/IMPLEMENTATION_STATUS.md` for that mechanism's known state.
- If your change moves a mechanism between Verified / Divergent / Gaps, **update
  the matrix row** in the same commit. A stale matrix is worse than none.
- The matrix's per-row **headline issue** is prose, not a count, and it goes
  stale the moment code changes. If your change invalidates a headline, fix the
  headline in the same commit even when the Verified/Divergent/Gaps numbers are
  unchanged — and do not invent new counts. A count you did not re-derive is a
  guess wearing a measurement's clothes.

## Counts must agree

The 12-vs-16 split is **resolved by convention, not by a bug fix** — see
"What this repo is" above, [`proofs/README.md`](proofs/README.md) and
decision D-013. The number that used to be an open defect is now pinned by two
tests, which is stronger than any prose count:

- `tests/test_config_system.py::TestAblationGridComplete::test_all_mechanisms_length_is_stable`
  asserts `len(ALL_MECHANISMS) == 12`. Its failure message names the four sites
  to update: the ablation grid, the `mechanisms.py` header,
  `GWP.N_MECHANISMS`, and `proofs/IMPLEMENTATION_STATUS.md` — and its docstring
  points at this file's "Counts must agree" section as the convention.
- `tests/test_model.py::TestGWPFramk::test_gwp_import` asserts
  `GWP.N_MECHANISMS == 16` and `GWP.N_GROUPS == 3`.

So if either number changes, a test goes red and names the sites to update.
When you add/remove/rename a mechanism, reconcile **all** of: the `mechanisms.py`
header comment, `ALL_MECHANISMS`, `N_MECHANISMS`, `GROUPS`, the audit matrix, the
`use_<mech>` keys in `defaults.yaml`, the leave-one-out configs under
`config/ablations/`, and `proofs/README.md`'s 16→12 mapping table. Then re-run
the two tests above rather than re-reading the prose.

## Config system

`config/**/*.yaml` is **deep-merged over `defaults.yaml`**. Consequences:

- An ablation yaml should contain only the delta.
- `src/train.py` warns on keys absent from `defaults.yaml` (catches typos like
  `lamda_swip`). Never silence that warning.
- `_meta.*` subtrees are exempt from the check.
- CLI: `--fname`, `--output_dir`, `--no_defaults`.
- Checkpoint resume is `meta.load_checkpoint: true`, which loads
  `<logging.folder>/checkpoint-latest.pth.tar`.

## Code conventions

- Python 3.9-compatible syntax (no `X | Y` in runtime-evaluated positions, no
  `match`), even though the local interpreter is 3.10. `target-version = "py39"`.
- `from __future__ import annotations` at the top of modules using PEP 604 hints.
- Line length 100. `ruff` then `black`.
- Mechanism modules expose a `compute_loss` / `forward` plus an `info` dict of
  diagnostics. Preserve that shape — `MechanismBundle.forward` aggregates them.
- `retract()` must be called after `optimizer.step()` to hold the Stiefel
  manifold constraint. Do not remove it.
- Copyright header: `# Copyright 2026 Slyatski Ilya` + Apache-2.0 line.
  The repo has a single named author, so the holder is that person rather than
  a collective - a copyright that resolves to nobody will not survive review.

## Tests

- One test file per mechanism: `tests/test_<mech>.py`.
- `tests/test_sterility.py` guards the sterility property — if you make a
  mechanism stronger, this test is the one that must keep passing. Read it
  before changing any loss.
- `tests/test_mechanism_wiring.py` guards config → mechanism wiring.
- Tests must be deterministic. Use `src/utils/seed.py`; no unseeded randomness,
  no network, no GPU.
- Prefer asserting a *property* over a magic number. A test that pins a
  constant breaks on every refactor and catches nothing.

## Boundaries for parallel agents

One owner per file. Real collision hotspots in this repo — if two tasks need
any of these, they **cannot** run concurrently:

- `src/models/mechanisms.py` (the bundle wires all 12 modules)
- `src/train.py` (the loss aggregator)
- `defaults.yaml` (everything deep-merges onto it)
- `proofs/IMPLEMENTATION_STATUS.md` (one shared matrix)
- `src/models/jepa.py` (the model every test constructs)

Per-mechanism files (`src/models/<mech>.py`, `tests/test_<mech>.py`,
`proofs/<mech>.md`, `config/ablations/*_<mech>*.yaml`) are disjoint and safe to
work on in parallel. That is the intended partitioning axis.
