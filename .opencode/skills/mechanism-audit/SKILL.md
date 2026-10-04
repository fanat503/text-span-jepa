---
name: mechanism-audit
description: Audit one GWP mechanism in text-span-jepa by comparing its proof document against implementation, tests, ablation configs, and the IMPLEMENTATION_STATUS matrix row. Use when reviewing, changing, or trusting any mechanism in src/models/.
---

# Mechanism audit

Compares what a mechanism **claims** against what it **does**. This repo's
whole value proposition is honesty about that gap, so run this before trusting
or changing any mechanism.

## Inputs

One mechanism name, e.g. `sta`. Paths are all under the repo root.

| Artifact | Path |
|---|---|
| Design doc | `proofs/<mech>.md` |
| Implementation | `src/models/<mech>.py` |
| Tests | `tests/test_<mech>.py` |
| Ablations | `config/ablations/*<mech>*.yaml` |
| Wiring | `src/models/mechanisms.py` |
| Matrix row | `proofs/IMPLEMENTATION_STATUS.md` |

## Procedure

### 1. Decompose the proof into falsifiable claims

Read `proofs/<mech>.md` and write down every theorem, bound, identity, and
invariant as a separate claim that *could be false*. Keep a running list; do
not batch it into prose.

Mark as **not falsifiable** anything motivational ("encourages the encoder to…")
and set it aside. The repo's own guidance is to cite mechanisms by *implemented*
behavior.

### 2. Locate the implementing code

For each claim, find the lines that would implement it. Quote `file:line`.
A claim with no implementing code is `ABSENT` — that is a legitimate finding,
not a search failure, but double-check you did not miss an indirect path.

### 3. Classify each claim

| Verdict | Meaning |
|---|---|
| `VERIFIED` | code provably does this — quote the lines |
| `DIVERGENT` | code does something real, but different — state both forms |
| `ABSENT` | nothing implements it |
| `UNTESTED` | implemented, no test pins it |
| `UNTESTABLE` | only by changing the design |

### 4. Check the theorem's preconditions

This is where most real defects hide. Verify each assumption against code:

- **orthonormality** — is `Q.T @ Q ≈ I` actually enforced, or just hoped for?
- **exact simplex / partition of unity** — softmax, or a floored softmax?
- **fixed `k`** — or a time-varying curriculum `k(step)`? A proof about
  fixed `k` does not transfer to a schedule.
- **EMA proxies** — is the bound evaluated on EMA-smoothed state instead of the
  quantity the theorem requires? (Known issue for WSR, RDC, WSD.)
- **detached tensors** — a `.detach()` where a gradient is required makes a
  claimed objective unminimizable, even though the code runs.

### 5. Check the test actually pins behavior

Read `tests/test_<mech>.py`. Does it assert a property, or only shapes and
finiteness? List the properties that are implemented but unpinned — those are
the regression risk.

### 6. Reconcile with the matrix

Compare your tallies to the row in `IMPLEMENTATION_STATUS.md` (audited
2026-08-24). If they disagree, state which you trust and why. The matrix is
dated; you are auditing now.

### 7. Recommend

Rank by correctness impact, not by ease. The best finding is the one with the
highest correctness-per-risk ratio — usually a one-line fix that removes a false
claim from a paper, or a test that pins an unpinned invariant.

## Environment

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
Push-Location C:\dev\text-span-jepa
& $PY -m pytest tests/test_<mech>.py -q
Pop-Location
```

`python` on `PATH` is a broken shim on this machine — always use `$PY`.
CPU-only environment; do not propose GPU-dependent checks.

## Rules

- Quote `file:line` for every verdict. A verdict without evidence is a guess.
- Do not propose changing code to match a proof. That is a human research
  decision — report the gap and stop.
- If you change behavior, update the matrix row in the same commit. A stale
  matrix is worse than no matrix.
- Do not fix code you were not asked to fix.
