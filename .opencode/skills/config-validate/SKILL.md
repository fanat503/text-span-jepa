---
name: config-validate
description: Validate the text-span-jepa YAML config system - deep-merge over defaults.yaml, ablation deltas, typo detection, mechanism toggle reachability, experiment-grid completeness, and checkpoint naming. Use before training, after editing any config, or when an ablation seems to do nothing.
---

# Config validation

`config/**/*.yaml` is **deep-merged over `defaults.yaml`**. That single fact
causes most config bugs: a key you think you set may be overridden, or a key
you set may be doing nothing because it does not exist.

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
Push-Location C:\dev\text-span-jepa
```

## 1. Every key must exist in `defaults.yaml`

`src/train.py` warns about keys absent from `defaults.yaml`, exempting
`_meta.*`. Run a training smoke test and read the warnings:

```powershell
& $PY -m src.train --fname config/ablations/sta_on.yaml --no_defaults
```

**Check the typo detector actually covers nested keys.** A nested typo that
slips through silently changes nothing while looking configured. That is the
`lamda_swip` class of bug. Test it:

```powershell
# temporarily add a nested typo, confirm it warns, then remove it
```

If nested typos are not detected, that is a bug in `src/train.py` — report it,
do not work around it by adding keys to `defaults.yaml`.

## 2. Ablations must be deltas only

An ablation file should contain only what differs from the defaults. A file
that restates defaults is a maintenance hazard: when the default changes, the
ablation silently keeps the old value and the experiment becomes a lie.

```powershell
Get-ChildItem config\ablations\*.yaml | ForEach-Object { $_.Name }
```

For each: list which keys it sets, and confirm each is genuinely a delta.

## 3. Experiment grid completeness

For every mechanism there should be a `no_<mech>.yaml` and an enable variant
(`<mech>_on.yaml`, or a group variant like `all_core.yaml`,
`sta_gac.yaml`, `wsr_sta.yaml`).

Build the table:

| Mechanism | off variant | on variant | wired in code |
|---|---|---|---|

A hole in this grid is a hole in the paper's ablation table. Cross-check
against `MechanismBundle.ALL_MECHANISMS` in `src/models/mechanisms.py` — the
code is the source of truth for which mechanisms exist.

## 4. Toggle reachability

Every `use_*` / `lambda_*` key read by `MechanismBundle.from_config` must be
settable from some config. Conversely, any `use_*` key present in
`defaults.yaml` but never read by `from_config` is dead config surface that
implies a capability that does not exist.

This is a two-way check. Both directions are defects.

## 5. Scaling ladder integrity

`config/scaling/` claims `xsmall_30m`, `small_100m`, `base_140m`,
`large_300m`. Verify:

- the four configs actually differ in depth / width / heads
- sequence length and batch size are **comparable** across them — a scaling
  study that varies batch size alongside model size cannot attribute anything
- the names match reality. Instantiate each model and count parameters. This
  claim is wrong often enough to always verify it.

```powershell
& $PY -c "import yaml,torch; from src.models.jepa import *; ..."
```

Note the parameter count *with mechanisms enabled* — the 16 mechanisms add
parameters and a quoted number may reflect only the bare encoder.

## 6. Checkpoint / resume paths

- `meta.load_checkpoint: true` loads `<logging.folder>/checkpoint-latest.pth.tar`
- `logging.keep_last_epoch_ckpts: <K>` prunes older `checkpoint-ep{N}` files

Verify the glob patterns in the trainer match those exact names. Confirm the
prune can never delete `checkpoint-latest`, and that resume restores optimizer,
scheduler, scaler, and global step — a resume missing the scheduler silently
changes the LR schedule.

## 7. CI-realistic smoke test

```powershell
& $PY -m src.train --fname config/scaling/xsmall_30m.yaml --output_dir .tmp_cfgcheck
```

Must complete quickly on CPU. If it does not, the config is wrong for the
documented CPU workflow. Never validate with `large_300m` or `kaggle`.

```powershell
Pop-Location
```

## Rules

- Read the consumer before judging a config. A key that looks unused may be
  read dynamically; a key that looks used may be ignored.
- Never "fix" a config by adding a key to `defaults.yaml` unless the mechanism
  genuinely needs it. That hides the typo instead of surfacing it.
- Report the grid as a table with holes marked. Do not paper over gaps.
- `defaults.yaml` is a collision hotspot — report required changes to it
  rather than editing it when other agents are active.
