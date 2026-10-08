# text-span-jepa — agent rules

Project rules. These **override** the global `~/.config/opencode/AGENTS.md`.

## What this repo is

A self-supervised JEPA (Joint-Embedded Predictive Architecture) research codebase.
The model predicts **in latent space** — not token reconstruction. 16 claimed
mechanisms (GWP: Grassmann Workspace Prediction) are organized as:

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

Baseline at the time these rules were written: **686 tests pass, ~108s**, on
`torch 2.13.0+cpu`, Python 3.10, no CUDA. CPU-only: never write a test that
silently requires a GPU.

## Commands

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
& $PY -m pytest -q                        # full suite, ~110s
& $PY -m pytest tests/test_sta.py -q      # one file
& $PY -m pytest -q -k "mechanism_name"    # by keyword
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
   formula, SPC loss type, SWIP missing term).
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

## Counts must agree

`16 mechanisms` is claimed in `mechanisms.py` header, `GWP.N_MECHANISMS = 16`,
and `GWP.N_GROUPS = 3`, but `ALL_MECHANISMS` lists 12 module names and the audit
matrix covers 13 rows. When you add/remove/rename a mechanism, reconcile
**all** of: the header comment, `ALL_MECHANISMS`, `N_MECHANISMS`, the audit
matrix, and the ablation configs under `config/ablations/`. Mismatched counts
are a real, currently-open defect in this repo.

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

- `src/models/mechanisms.py` (the bundle wires all 16)
- `src/train.py` (the loss aggregator)
- `defaults.yaml` (everything deep-merges onto it)
- `proofs/IMPLEMENTATION_STATUS.md` (one shared matrix)
- `src/models/jepa.py` (the model every test constructs)

Per-mechanism files (`src/models/<mech>.py`, `tests/test_<mech>.py`,
`proofs/<mech>.md`, `config/ablations/*_<mech>*.yaml`) are disjoint and safe to
work on in parallel. That is the intended partitioning axis.
