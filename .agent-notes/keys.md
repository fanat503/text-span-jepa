# TASK: early-return key-set shrink (five mechanism modules)

Worktree `C:\dev\wt-keys`, branch `agent/keys`, base `main` @ `0947f29`.
Ownership: `src/models/{cgn,cmc,gac,jawp,pcr,puc,rdc,spc,sta,swip,wsd,wsr}.py`
and their tests. `mechanisms.py`, `jepa.py`, `train.py` untouched.

---

## 1. Which modules were defective — found by measurement, not by reading

`.agent-notes/probe_keys.py` drives **all twelve** modules' `forward` at step 0
(inside warmup) and at step 5000 (past every shipped `*_warmup_steps` in
`defaults.yaml`) and diffs the returned info key sets.

**Before the fix:**

```
jawp  ok      warmup= 9 late= 9
cgn   ok      warmup= 7 late= 7
swip  ok      warmup= 6 late= 6
pcr   ok      warmup=15 late=15
spc   ok      warmup=10 late=10
wsd   ok      warmup=12 late=12
cmc   ok      warmup=10 late=10
gac   SHRINKS warmup= 2 late= 9      missing 7
sta   SHRINKS warmup= 4 late=10      missing 6
puc   SHRINKS warmup= 3 late=11      missing 9, PLUS 1 key only on warmup
rdc   SHRINKS warmup= 2 late=11      missing 10, PLUS 1 key only on warmup
wsr   SHRINKS warmup= 3 late=10      missing 7

MODULES WHOSE EARLY RETURN DROPS KEYS: ['gac', 'sta', 'puc', 'rdc', 'wsr']
```

Five: **gac, sta, puc, rdc, wsr.** Exact sites:

| module | early-return | dropped |
|---|---|---|
| `gac.py:167` | `warmup_factor < 1e-6` → `{"gac_loss", "gac_warmup"}` | 7 keys |
| `sta.py:267` | same guard → 4 keys | 6 keys |
| `puc.py:244` | same guard → 3 keys | 9 keys **+ `puc_warmup` exists only here** |
| `rdc.py:149` | same guard → 2 keys | 10 keys **+ `rdc_warmup` exists only here** |
| `wsr.py:276` | same guard → 3 keys | 7 keys |

**After the fix: all twelve `ok`.**

### Two findings beyond the metrics audit

`docs/METRICS.md:257-271` names `rdc, puc, gac, cmc, sta`. Two corrections:

1. **`wsr` is missing from the audit** — and it is the second-worst offender,
   dropping 7 keys. The audit's `cmc` row (1 key) was already fixed in the tree
   by the single-return refactor documented at `cmc.py:511-514`; the audit
   predates it. So the live list is gac/sta/puc/rdc/**wsr**, and wsr replaced
   cmc.

2. **Two more key-set splits, same class, on different axes:**
   - **Reverse asymmetry.** `puc_warmup` and `rdc_warmup` existed *only* on the
     early return. That is the contract inverted: a key that vanishes after
     warmup is absent on every later step, i.e. for ~99% of a run. `gac`,
     `sta`, `wsr` already had `<mech>_warmup` on both paths.
   - **WSR's two modes.** `mode='gradient'` (the default) reported 10 keys;
     `mode='sam'` reported 12 — `wsr_loss_current` / `wsr_loss_perturbed` only
     in `sam`. A reader written against `sam` KeyErrors on the default mode.
     Fixed; the audit missed this entirely.

---

## 2. Which consumers would have crashed — the honest answer

**None in `src/`, today. The defect is latent, not currently fatal.** I checked
rather than assumed, and I am reporting what I measured:

| consumer | what it does | can it KeyError? |
|---|---|---|
| `mechanisms.py:440,445,464,470,478,484,490,502` | `info.update({f"{m}_{k}": v for k,v in m_info.items()})` — **iterates** | **No** |
| `jepa.py:902-924` | same `.items()` iteration | **No** |
| `train.py:1666-1667` | `for _k2,_v2 in gac_info.items(): loss_dict[f"gac_{_k2}"]=_v2` — iterates | **No** |
| `train.py:1624` | `cmc_info.get("cmc_skipped", False)` — the one host line naming a mechanism key; `.get`, not `[]` | **No** |
| `train.py:1482-1496, 1814` | `CSVLogger` header is a **fixed 12-column literal**; mechanism keys are never written to CSV | **No** |
| `tests/test_wsr.py:151`, `test_sta.py:60`, `test_rdc.py:114`, `test_puc.py:44`, `test_training_state_guards.py:377` | `info["<key>"]` — **indexes** | **Yes**, on any warmup step |

So the shipped trainer would not have died at step 3. What the defect
guarantees is that **the repo's own tests KeyError if they are ever pointed at
a warmup step**, and that the contract "a consumer may `info[k]` for any key
this module has ever emitted" is false — which is exactly what an external
metrics consumer (`docs/METRICS.md` is a 153-metric inventory built on that
assumption) relies on. `docs/METRICS.md:269` states the hazard; it was correct
about the hazard and wrong about the module list.

**Reported, not edited** (outside my ownership): if a future host wants a
fail-fast guard, `mechanisms.py:440-502` and `jepa.py:902-924` are where a
single shared `_emit(mech, info)` helper would live, so the prefixing and the
key-set check live in one place. Left alone deliberately — that is a design
change to a file whose owner is another agent.

---

## 3. The fix pattern

**An early return returns the SAME keys, with a sentinel for what cannot be
computed yet.** Two shapes, chosen per module:

- **GAC → one return.** The starvation count is a pure function of the step's
  arguments, so it is now computed on both paths and only the *loss* and the
  *training-state mutations* are gated by `in_warmup`. This follows the pattern
  `cmc.py:511-514` states outright: *"ONE return, so the key set cannot drift
  between branches."* Side effect worth having: `gac_n_starved` is now visible
  during warmup — which is the number the warmup exists to wait on.
- **STA / PUC / RDC / WSR → sentinel return**, full key set, zeros for what is
  not yet measurable, and the module's own `<mech>_warmup: True` flag saying why.
  Constants that *are* knowable during warmup are reported for real
  (`puc_target_entropy`, `puc_n_components`, `rdc_eta`, `rdc_k_workspace`,
  `wsr_rho`, `gac_running_starved_fraction`).

Choices worth naming:

- **`wsr_grad_source: "warmup"`** — a fourth value alongside
  `{lagged, live, unavailable}`, documented in `_resolve_euclidean_gradient`'s
  docstring. `"unavailable"` would be a lie (it means "we substituted the
  orthonormality proxy"), so `wsr_gradient_substituted` stays **False**: nothing
  was substituted, nothing was measured.
- **`wsr_loss_current`/`wsr_loss_perturbed` in `mode='gradient'` = `float("nan")`,
  not `0.0`.** Gradient mode never calls `loss_fn`, so a `0.0` would sit next to
  a real non-zero `wsr_sharpness` and invite the reader to derive
  `sharpness = perturbed − current = 0` — a wrong number computed from numbers
  that all *look* measured, which is the exact failure class
  `docs/THEORY_AND_PROOFS.md` §"pattern 5" calls the worst. NaN is still a real
  float, so `info[k]` holds and the gap shows in a plot. (The *warmup* path
  uses `0.0`, which is correct there: the loss genuinely is zero during warmup,
  and `<mech>_warmup: True` says so.)

---

## 4. Verification

### The suite (TDD — written first, watched fail, then fixed)

Red before the fix, `3 x 12` key-stability assertions + the WSR mode assertion:

```
FAILED test_warmup_path_returns_the_same_keys[gac|sta|puc|rdc|wsr]
FAILED test_indexing_every_key_survives_the_first_steps[gac|sta|puc|rdc|wsr]
FAILED test_warmup_values_are_real_sentinels_not_none[gac|sta|puc|rdc|wsr]
FAILED test_wsr_modes_report_the_same_keys
16 failed, 60 passed
...
E   KeyError: 'gac_max_grad_norm'
E   KeyError: 'sta_condition_number'
E   KeyError: 'puc_carries_grad'
E   KeyError: 'rdc_drift_ratio'
E   KeyError: 'wsr_directional_sharpness'
```

`test_indexing_every_key_survives_the_first_steps` is the crash itself, run as
a consumer would run it — learn the key set at step 5000, then `info[k]` for
each key at step 0.

Green after:

```
$ & $PY tools/rt.py tests/test_mechanism_wiring.py tests/test_sterility.py
rt.py: threads=1  total_budget=90s  slow_ok=False
collected 76 items
tests\test_mechanism_wiring.py ......................................... [ 53%]
...                                                                      [ 57%]
tests\test_sterility.py ................................                [100%]
======================= 76 passed, 4 warnings in 3.59s ========================
```

`test_sterility.py` green throughout: no mechanism got weaker. The 4 warnings
are pre-existing (WSR's orthonormality-proxy warning on the bundle path; PUC's
legacy-buffer warning) and appear on `main` too.

### The fix changes no value — `.agent-notes/diff_values.py`

Runs the same driver against pre-fix sources (`git show main:src/models/…`) and
the worktree in two separate processes and diffs every reported number plus
every buffer/parameter.

```
ok   gac.state (3 tensors identical)
ok   sta.state (7 tensors identical)
ok   puc.state (6 tensors identical)
ok   rdc.state (7 tensors identical)
ok   wsr_gradient.state (5 tensors identical)
ok   wsr_sam.state (5 tensors identical)
     puc step=12/20/33: added keys ['puc_warmup']
     rdc step=12/20/33: added keys ['rdc_warmup']
     wsr_gradient step=12/20/33: added keys ['wsr_loss_current','wsr_loss_perturbed']

POST-WARMUP VALUES AND STATE IDENTICAL: True
```

Every post-warmup loss, every pre-existing info value, and every training-state
tensor is bit-identical to `main`. The only difference is **added keys**.

### Mutation verdict — the check is not vacuous

Re-shrank `rdc.py`'s early return to its original two keys:

```
E   AssertionError: rdc drops ['rdc_drift_ratio', 'rdc_eta', 'rdc_k_workspace',
    'rdc_ortho_drift_norm', 'rdc_stationary_bound', 'rdc_theoretical_bound',
    'rdc_total_drift_norm', 'rdc_transient_bound', 'rdc_warmup_factor',
    'rdc_workspace_drift_norm'] on its step-0 (warmup) path.
E   KeyError: 'rdc_drift_ratio'
3 failed, 73 passed
```

Restored; `git diff main -- src/models/rdc.py` shows the file byte-for-byte
back to its fixed form. `ruff check` and `black --check` clean on all six files.

---

## 5. Риски / risks

1. **This is a behaviour change on the warmup path — a deliberate, visible one.**
   `gac_n_starved`, `gac_starved_fraction`, `gac_mean/min/max_grad_norm`,
   `puc_target_entropy`, `puc_n_components`, `rdc_eta`, `rdc_k_workspace` and
   `wsr_rho` are now reported as real values during warmup where they were
   previously absent. **Any downstream reader that used "key is absent" as a
   proxy for "module is warming up" must switch to the `<mech>_warmup` key.**
   I could not find such a reader in this repo, but I did not run the full
   suite (card forbids it) and cannot rule out one outside `src/`.

2. **`wsr_grad_source` gained the value `"warmup"`.** Any downstream `if
   wsr_grad_source == ...` chain that assumes a closed three-value enum needs
   the new case. In-repo, the only such assertions are
   `tests/test_training_state_guards.py:390,407,467,479,493`, all on
   post-warmup paths — verified unaffected. **That file is not mine; if the
   owner wants a tighter contract, narrowing the enum is a one-line change
   there.**

3. **NaN in `wsr_loss_current`/`wsr_loss_perturbed` in gradient mode** propagates
   through any mean/plot over them. That is the point — a NaN gap is honest,
   a `0.0` would not be — but a naive `np.mean` over a `wsr_*` column will now
   return NaN in gradient mode where it previously would not have had the keys
   at all. Downstream plot/table code should gate on `wsr_mode`.

4. **Two-return modules can drift again.** GAC is structurally immune (one
   return). STA/PUC/RDC/WSR have two dict literals that must agree; the three
   new parametrized assertions are what catch a drift, and they run over all
   twelve modules, but they only catch it if someone runs them.

5. **Test placement.** The regression test went into `tests/test_mechanism_wiring.py`,
   which is not strictly "a test file for one of the twelve". That file is the
   one the card designates for verification and the one the mutation verdict
   had to run through; splitting the assertions into `test_gac.py`,
   `test_sta.py`, … would put them in files the card does not let me run.
   If the owner prefers per-module placement, the class moves without edits.

6. **Not verified by execution:** nothing here ran a training step, so the claim
   is about *key sets and warmup-step values*, not about how a run behaves.
   The 76 passing tests and the two differential scripts are the whole of the
   evidence.

## 6. Files touched

```
src/models/gac.py   | 59 +++++----
src/models/puc.py   | 30 ++++-
src/models/rdc.py   | 25 +++-
src/models/sta.py   | 21 +++-
src/models/wsr.py   | 37 ++++-
tests/test_mechanism_wiring.py | 278 ++++++++++++++++++++++++++++++++++++++++-
6 files changed, 417 insertions(+), 33 deletions(-)
```

`.agent-notes/probe_keys.py` and `.agent-notes/diff_values.py` are the
reproduction scripts; both assert `src.__file__` starts at the worktree root
before doing anything (an editable install on this machine points at a
different clone — it did, and the first probe run failed on exactly that).
