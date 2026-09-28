---
name: verify-gate
description: The mandatory verification gate for text-span-jepa. Runs the full test suite, lint, format check, and a CPU smoke train, and refuses to pass on a count regression. Use before any commit, before claiming a task is done, and as the wave barrier in a parallel agent swarm.
---

# Verification gate

No agent in this setup claims completion without passing this. A "done" with
no command output is a claim, not a result.

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
Push-Location C:\dev\text-span-jepa
```

`python` on `PATH` is a broken shim on this machine (`No module named
'encodings'`). Always use `$PY`.

## Baseline to beat

**686 passed** in ~108s, on `torch 2.13.0+cpu`, Python 3.10, no CUDA.

The count must never drop. A net loss to make your own test pass is a
regression, not a trade-off.

## The four checks

### 1. Full suite

```powershell
& $PY -m pytest -q --no-header -p no:cacheprovider
```

Expect `686+ passed`. Watch the count, not just the exit code — a suite can
exit 0 with skips you did not intend. Record `passed / skipped / warnings`.

Two warnings are expected and benign:

- `data2vec target depth truncated` (encoder has 2 layers, `average_top_k_layers=8`)
- `'pin_memory' is set as true but no accelerator is found`

Any *new* warning class is a finding.

### 2. Lint

```powershell
& $PY -m ruff check .
```

Config in `pyproject.toml`: line length 100, `target-version = "py39"`, with a
specific ignore list. A new violation is a defect. Do not add to the ignore
list to make ruff pass — that is the same as weakening a test.

### 3. Format

```powershell
& $PY -m black --check .
```

If it fails, `& $PY -m black <files>` — then re-run the suite, because black
and ruff disagree occasionally.

### 4. CPU smoke train

Config and code changes need a runtime check, not just unit tests.

```powershell
& $PY -m src.train --fname config/scaling/xsmall_30m.yaml --output_dir .tmp_gate
```

Must finish on CPU in a couple of minutes. Watch for:

- the config typo warnings from `src/train.py` (keys absent from `defaults.yaml`)
- non-finite loss
- `retract()` errors after `optimizer.step()` — the Stiefel constraint must hold

Never validate with `large_300m` or the `kaggle` configs.

```powershell
Pop-Location
```

## Verdict

Report exactly one, with evidence:

- `PASS` — counts at or above baseline, lint and format clean, smoke train ran.
- `FAIL` — every defect listed with the exact command output and a minimal
  reproduction.

A vague PASS is worse than no gate. If a check could not be run, say
`NOT RUN` and why — never round it up to a pass.

## In a parallel swarm

This gate is the **wave barrier**. A coordinator may not start the next wave
until the previous wave is `PASS`. Carrying a failure forward compounds: the
next wave builds on a broken base and the defect becomes harder to localize.

If your change is intentionally red (a test documenting a known defect), the
gate is `FAIL` by design — report it that way, with the mechanism name and the
defect it pins, so the orchestrator can decide.
