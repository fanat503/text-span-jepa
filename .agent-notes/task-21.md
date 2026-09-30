# TASK-21 — cut the per-step device syncs in jawp / spc / pcr

**status: done.** Branch `agent/task-21`, worktree `C:\dev\wt-21`, base
`agent/wave-4` (8dbbb37). Commit `aa77896`. Not pushed.

---

## 0. Card-numbering discrepancy — read this first

The card I was given says "read card `### TASK-21` in TASKS.md", but that card
in `TASKS.md` is a **different task**: it is about halving peak activation
memory under gradient accumulation by dropping live autograd graphs from
`jepa._gac_z` / `jepa._cmc_pass["slots"]`, its `files_allowed` is
`src/models/jepa.py`, and it explicitly notes `jepa.py` collides with TASK-08.

The goal and file list in my spawn prompt ("cut the device syncs in the
per-step path, in `src/models/jawp.py`, `src/models/spc.py`, `src/models/pcr.py`")
are a **verbatim match for `### TASK-20`** in `TASKS.md` — same goal, same
three files, same forbidden list, and the same note about
`mechanisms.py:461,475,499`.

I did **not** do the TASKS.md TASK-21 (gac/cmc graph retention), because that
work is in `src/models/jepa.py`, which my prompt forbids and which another
worker is expected to own this tick. I did the card I was actually handed:
the sync work on the three files I own. **The real TASK-21 (jepa.py memory) is
still todo and still needs an owner who may touch `jepa.py`.** Flagging rather
than improvising, per the "out of bounds means stop" rule.

---

## 1. Syncs removed per step

Method: the card's own — patch `torch.Tensor.item` / `.tolist` / `.__bool__`
and record the calling frame's file, then run one step through
`jawp.compute_loss` + `spc.forward` + `pcr.forward` + backward + all three
retractions + `project_tangent_gradient`, on tiny tensors (D=64, N=8, B=8,
n_bands=8, n_levels=3).

| | host reads / step | over the 3-step loop | in my 3 files |
|---|---|---|---|
| **before** | **62** | 189 | 186 |
| **after**  | **8**  |  24 |  21 |

**Net: 54 host reads removed per step** (62 → 8), and — the more useful
property — **the count no longer scales with the loop bounds.** Before, it grew
linearly with `n_bands` (3 reads/band) and `n_levels` (4 reads/level); after,
it is constant. Measured directly: the SPC mutant count was
`{4 bands: 11, 8: 19, 16: 35}` before batching and `4 == 8 == 16` after; the PCR
mutant count was `{1 level: 5, 2: 9, 3: 13}` and is `1 == 2 == 3` after.

Per file, per step:

| file | before | after | what the syncs were |
|---|---|---|---|
| `spc.py` | 31 | 7 | 3 per band (`band_residuals`, `band_losses`, target var) + 5 scalars + 2 `.tolist()` |
| `pcr.py` | 21 | 1 | 4 per level (correction norm, gate, r_energy, total energy) + 3 overall + 3 `level_offsets[level].item()` |
| `jawp.py` | 13 | 1 | 9 `.item()` in `compute_loss`, 2 `bool(tensor)` guards, 2 `active_k.item()` in the retraction path |

The surviving 8 reads are **batched vector reads** — one `.tolist()` per module
(`jawp:476`, `pcr:419`, `spc:429,430,477,486,489`), plus one outside my files
(the probe's own `total.item()`, standing in for `train.py:1506`).

**The wall-clock claim is UNVERIFIED on this host.** There is no CUDA here, so
a synchronisation cannot be observed or timed. The sync count above is a real,
exactly-reproducible number; "this makes training faster" is an inference, not
a measurement. I did not time anything and do not claim a speedup.

---

## 2. Bit-identity — how the info dict types changed, and every consumer

### The rule I used
`torch.stack([...]).tolist()[i]` is bit-identical to that element's own
`.item()`. I verified this rather than assuming it: 3000 random 0-dim tensors
each in fp32 / fp64 / bf16 / fp16, **0 mismatches**, and a `torch.stack` of
mixed dtypes promotes to the *widest* input (a widening, so lossless). This
is why the batched reads are value-preserving and why I added **no dtype
casts** — an explicit `.to(torch.float32)` would have *narrowed* an fp64
module.

### Type changes in `info`

**None of the reported values changed type.** Every key that was a Python
`float`/`int`/`list[float]` still is. That was a deliberate choice: the card
warned that a float may become a 0-dim tensor and that consumers must be found
first. I found them, and then did not need to change any type — which removes
the whole class of consumer risk (JSON serialisation, `math.isfinite`,
`f"{v:.3f}" all keep working unchanged).

The two exceptions, both deliberate:

| value | before | after | why |
|---|---|---|---|
| `jawp._compute_pca_alignment` **return** | `float` | 0-dim tensor | private staticmethod, one caller. The `math.isfinite` + `min/max` policy moved to the host in `compute_loss`, on the same value, so `info["pca_alignment"]` is still a float. |
| `spc._update_running_statistics` **arg** | `list[float]` | `(n_bands,)` tensor | private, one caller. The buffer values are unchanged (`torch.equal`, see §3). |

### Consumers checked

| consumer | what it does | impact |
|---|---|---|
| `mechanisms.py:451,470,445` | `info.update({f"jawp_{k}": v …})` | none — pass-through |
| `jepa.py:865,871,887` | `loss_dict.update({f"jawk_{k}": v …})` | none — pass-through |
| `train.py:1540-1543` | `f'{loss_dict.get("jawk_workspace_utilization",0):.3f}'` | none — still a float. **This is where a sync belongs**: logging runs every `log_freq` steps, not every step, so deferring the read here is correct, not hidden. I did not move it. |
| `train.py:1556-1569` `CSVLogger.log` | `tv[0] % tv[1]` | none — the logged keys (`loss_span`, …) are untouched by this card |
| `interp/ablation.py:368` | `loss = loss - weight * info[key]` | none — `LOSS_TERM_ABLATIONS` covers only `loss_span/future/decoder/variance/covariance` |
| `tests/test_jawp.py:159,166-169,178` | `<`, chained compare, `math.isfinite(v)` over every info value | none — values are still floats |
| `tests/test_pcr.py:45,338` | `>= 0` | none |
| `tests/test_spc.py:476` | `"spc_band_weights" in info` | none — still `list[float]` |
| `visualization.py:984` `plot_spc_band_analysis` | takes arrays | none — fed by the smoke test, not by `info` |

**Where the sync legitimately belongs:** `train.py:1506` (`total_loss.item()`
for the meter) and the `:.3f` log formats. Both run at logging cadence, not per
step. I did not touch them — `train.py` is forbidden — and I am not claiming
they are optimal, only that they are not the per-step path.

### The one that would have moved a number

`spc_uniform_loss` was `sum(<python floats>) / n_bands`. My first version used
`band_residuals_t.sum() / n_bands`. **Measured divergence: up to 1.1e-6 (fp32)
and 2.7e-2 (bf16)** — Python's `sum` accumulates in float64 left-to-right,
torch's `.sum()` reduces in the tensor dtype in a different order. Reverted to
the Python `sum` over the already-materialised list, which costs **zero extra
syncs** (that list comes from the batched read anyway). Same reasoning kept
jawp's `1.0 - x`, `max(0.0, x)` and pcr's `+ 1e-10` and fraction division on
the host, and spc's `max(tv, eps)` as `clamp` (verified equivalent on an
exhaustive bf16 grid and an fp32 log grid: 0 mismatches, and both propagate
NaN the way `max(nan, eps)` does).

---

## 3. Verify (full paste)

```
$ & $PY tools\rt.py tests/test_sterility.py tests/test_mechanism_wiring.py tests/test_host_reads.py tests/test_spc.py tests/test_pcr.py tests/test_jawp.py
rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_sterility.py tests/test_mechanism_wiring.py tests/test_host_reads.py tests/test_spc.py tests/test_pcr.py tests/test_jawp.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 178 items

tests\test_sterility.py ................................                 [ 56%]
tests\test_mechanism_wiring.py ......                                    [ 66%]
tests\test_host_reads.py ...................                             [100%]
tests\test_spc.py ................................                       [100%]
tests\test_pcr.py ................................                       [100%]
tests\test_jawp.py .....................................................  [100%]

============================== warnings summary ===============================
tests/test_mechanism_wiring.py::TestGACHook::test_stashed_slots_receive_grads
tests/test_mechanism_wiring.py::TestLiveWorkspaceViews::test_regularizers_give_workspace_grads
tests/test_mechanism_wiring.py::TestLiveWorkspaceViews::test_retraction_projects_grad_inplace_and_keeps_orthogonal
  src\models\wsr.py:285: UserWarning: WSR mode='gradient': no dL/dQ is available (Q.is_leaf=False,
  Q.grad=n/a (non-leaf), set_lagged_gradient() never called). ... Call wsr.set_lagged_gradient(...)

====================== 178 passed, 3 warnings in 10.64s =======================
```

`tests/test_sterility.py` (38 → included above) **passes**; the sterility
property is untouched. The 3 warnings are pre-existing `wsr.py` warnings, not
mine. `black --check` and `ruff check` on all four files: clean.

Downstream consumers of these `info` dicts, run separately:
`test_wsd/test_wsr/test_swip/test_cgn` → **95 passed**;
`test_ablation_module/test_run_comparison/test_determinism/test_training_state_guards`
→ **111 passed, 1 xfailed** (the xfail is pre-existing).

### Independent bit-identity probe (scratch, gitignored)

Comparing the new code against a verbatim transcription of the pre-change logic:

```
JAWP: info dict, new vs original (exact float equality)
  jawp mismatches: 0
  zero-input branch (pred==target==0):
   workspace_cosine: 0.0 == 0.0 -> True
   predictive_relevance: 1.0 == 1.0 -> True
   pca_alignment: 0.0 == 0.0 -> True
SPC:  spc info mismatches: 0
SPC running-statistics buffers: vars equal=True predab equal=True   (3 configs)
PCR:  pcr mismatches: 0   (and the refined tensor is torch.equal in every case)
level_offsets host mirror == buffer, for every construction: match=True
```

---

## 4. Diff-stat

```
 src/models/jawp.py       | 196 ++++++++++++-----
 src/models/pcr.py        |  96 ++++++---
 src/models/spc.py        | 108 +++++++---
 tests/test_host_reads.py | 545 ++++++++++++++++++++++++++++++++++++++++++++++++
 4 files changed, 837 insertions(+), 108 deletions(-)
```

Commit `aa77896`, no `--no-verify`, no skip, no xfail, no deleted or weakened
test. Files touched: exactly the three I own, plus the new test file. The eight
`.agent-notes/_probe_*.py` scratch files are covered by `.gitignore` and are
**not** in the commit.

---

## 5. Mutation verdict

Four mutations, each reverted. Two of them found real holes in my own tests,
which is the point of doing this.

| # | mutation | result |
|---|---|---|
| 1 | SPC: restore the per-band `.item()` loop for `band_residuals`/`band_losses` | **RED** ×2 — `host reads scale with n_bands …: {4: 11, 8: 19, 16: 35}`; and `a host-read site runs 8x per step: [('spc.py:405', 8), ('spc.py:408', 8)]` |
| 2 | SPC: restore `.item()` in `_update_running_statistics` | **RED** ×2 — `{4: 15, 8: 27, 16: 51}`, `spc.py:360` runs 8× |
| 3 | JAWP: restore the `if pred_norm > 1e-10 and target_norm > 1e-10:` guard | **GREEN on the first attempt — my test was broken.** The counter tagged sites as `"jawp.py:416"` with no kind, so `"bool" in s` never matched. Fixed the counter to record the kind, then **RED**: `a tensor was converted to bool in the step path: ['jawp.py:416 [bool]', 'jawp.py:416 [bool]']` |
| 4 | JAWP: `active_k_value()` → `int(self.active_k.item())` | **GREEN — a real gap.** Correct value, so both mirror-agreement tests passed, but it reintroduces a sync on every retraction. Added `test_active_k_width_is_not_read_back_from_the_buffer`; re-ran against the still-mutant code → **RED**: `['jawp.py:240 [item]', 'jawp.py:240 [item]']`. Then reverted. |
| 5 | PCR: per-level reads present but discarded (values unchanged) | **RED** on the count test only, `{1: 5, 2: 9, 3: 13}`, with the value tests staying green — the intended separation: a mutation that is value-neutral must still be caught by the count test. |

**Load-bearing claims, each with the mutation that killed it:** the batching
(mutations 1, 2, 5), the absence of `bool(tensor)` in the step path
(mutation 3, after fixing the counter), and the `active_k` mirror *not* reading
the buffer (mutation 4, after adding the test it revealed was missing).

The 10 value-equality tests in `TestBatchedReadsReturnTheSameFloat` were never
the thing catching the regressions — they compare against the pre-change code
and would pass any value-preserving implementation, synchronising or not. That
is deliberate: they pin the "bit-identical" requirement, and the count tests
pin the perf requirement. Neither alone is sufficient, which is why both exist.

---

## 6. не_сделано / риски

**не_сделано**

- **The real `### TASK-21` card (jepa.py gac/cmc graph retention) is not done.**
  See §0. It needs `src/models/jepa.py`, which I was forbidden to touch.
- **The other syncs in the step path are not mine.** `mechanisms.py:460,474,498`
  and `jepa.py:1024,1054,1075,1082` each do `int(self.jawp.active_k.item())` —
  7 more per forward — and `train.py:1455` an 8th. `mechanisms.py` and
  `jepa.py` are forbidden, and `train.py` is forbidden. I exposed
  `JAWPModule.active_k_value()` precisely so those call sites are a one-line
  change for whoever owns them: **`int(model.jawp.active_k.item())` →
  `model.jawp.active_k_value()`**. Recommend the same for `mechanisms.py:461,475,499`
  (6 redundant `workspace_Q[:, :k_active]` slices) — the note in the real
  TASK-20 card, reported as asked, not fixed.
- **Wall-clock speedup is unmeasured** (no CUDA on this host). Sync count is the
  number; please re-measure on a GPU box before quoting a speedup.
- I did not run the full suite (`tools/rt.py` refuses it by design) and did not
  train. Per-file runs above cover every mechanism suite and every consumer I
  identified.

**риски**

- **`_active_k` mirror vs. `state_dict`.** A host mirror can desync from a
  buffer that travels in checkpoints. Mitigated by `_load_from_state_dict`
  re-reading it, and pinned by `test_active_k_mirror_resyncs_on_checkpoint_load`.
  **Residual risk:** a direct `jawp.active_k.fill_(3)` from outside the class
  would desync the mirror silently. Grep shows no such caller
  (`train.py:278`, `jepa.py`, `mechanisms.py`, `tests/` all only *read*
  `.item()`), and `_set_active_k` is the sanctioned writer, but nothing
  *enforces* that. If someone later writes the buffer directly, the mirror
  must be updated too — this is a real invariant, not an enforced one.
- **Same shape for `pcr._level_offsets_py`**, with one difference that makes it
  safer: it is a pure function of `level_dims`, a plain Python list fixed at
  construction and never mutated, whereas the `level_offsets` buffer *is* in
  the state_dict. If a checkpoint ever carried a `level_offsets` inconsistent
  with its own `level_dims`, the mirror would silently win. No such checkpoint
  can exist today (the mirror is the cumsum of `level_dims` by construction),
  and `test_level_offsets_mirror_matches_the_buffer` asserts it for five
  configurations. Flagging because the buffer is checkpoint-visible and the
  mirror is not.
- **`torch.stack` promotion is load-bearing for bit-identity.** If a future
  edit mixes a *narrower* dtype into one of the batched lists in a way that
  demotes a value (e.g. a bfloat16 scalar into an all-bf16 stack alongside an
  fp32 one is a widening — fine — but an explicit `.to(float32)` on an fp64
  value would not be), the reported numbers would shift. The value tests
  against the old code would catch it; the count tests would not.
- **`significant` is an int64 band count promoted into a float dtype.** Exact
  for any realistic `n_bands` (far below 2²⁴). At an absurd `n_bands` ≥ 2²⁴ the
  promoted float could round — noted rather than defended.
- The three `info` dicts are consumed by `mechanisms.py`/`jepa.py`/`train.py`,
  which I could not run as a unit beyond the suites above. If a future consumer
  needs a Python scalar and finds a tensor, that is a *new* consumer, not a
  break I introduced — no existing consumer's type changed.
