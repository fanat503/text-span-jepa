# TASK-07 — run_comparison.load_model reads the current checkpoint format

- **status:** done
- **files:** `src/interp/run_comparison.py` (only file touched, only file I own)

## sites fixed

**7 `load_state_dict` calls across 3 branches → 3**, plus the 3 duplicated
`safe_torch_load` calls hoisted to one.

Old line numbers (pre-fix, `campaign/integration` @ d92d8a7):

| line | branch | call |
|---|---|---|
| 37 | jepa | `model.encoder.load_state_dict(ckpt.get("encoder", {}))` |
| 38 | jepa | `model.target_encoder.load_state_dict(ckpt.get("target_encoder", {}))` |
| 39 | jepa | `model.predictor.load_state_dict(ckpt.get("predictor", {}))` |
| 40 | jepa | `model.decoder.load_state_dict(ckpt.get("decoder", {}))` |
| 52 | mlm | `model.load_state_dict(ckpt.get("model", {}))` — right key, **non-strict**, so an absent key silently left random weights |
| 64 | data2vec | `model.encoder.load_state_dict(ckpt.get("encoder", {}))` |
| 65 | data2vec | `model.target_encoder.load_state_dict(ckpt.get("target_encoder", {}))` |

The card counted four; there are seven. All seven now funnel through one call at
`src/interp/run_comparison.py:97`:

```python
model.load_state_dict(state, strict=True)
```

`state` comes from the new `_full_model_state()` (`run_comparison.py:29-51`),
which is the single place the format is validated. It reuses
`src.train.CheckpointLoadError` (lazily imported, so the happy path never pays
for importing `src.train`) so there is one error type across the whole
checkpoint read path.

Approaches considered: (1) minimal — swap each `.get(...)` for `ckpt["model"]`
inline in each branch; (2) chosen — hoist the read + validation out of the
branch and restore once. Same end state, but (2) makes "one format" structural
instead of repeated three times, which is what makes a future third format
visible in review. Cost of (2): the `model_type` guard moved above the file
read so an unknown type still raises `ValueError` before touching the path
(preserved semantics; nothing tested it, but the CLI passes `argparse`
`choices` anyway).

## regression_head — what I found before choosing

The card said the data2vec branch loaded a `regression_head`. **It did not.**
There was no `regression_head` access anywhere in `run_comparison.py`; the
data2vec branch loaded exactly two submodules, `encoder` and `target_encoder`,
and silently left the regression head at its initial values. That was a live
bug independent of the format change — any data2vec comparison in this pipeline
was scoring a random projection head.

Measured, not assumed (`baselines/data2vec_baseline.py:91`:
`self.regression_head = nn.Sequential(*projs)` — a registered submodule):

```
d2v   regression_head tensors inside 'model': ['regression_head.0.bias', 'regression_head.0.weight']
```

So `regression_head` is **inside** `model.state_dict()`, under the
`regression_head.*` prefix. It is **not** a separate top-level checkpoint key
any more — the writer never emits one. The single `load_state_dict(state,
strict=True)` restores it for free, and the round trip below confirms the
loaded head is the saved head, not a fresh one.

Same shape check for the other two models, prefixes of the full state_dict:

```
JEPA   total keys: 90   prefixes: encoder 28, target_encoder 28, predictor 25,
                        decoder 4, target_centering 1, sigreg 2, jawp 2
D2V    total keys: 58   prefixes: encoder 28, target_encoder 28, regression_head 2
MLM    total keys: 30   prefixes: encoder 28, mlm_head 1, decoder 1
```

The JEPA branch's four old reads covered 85 of 90 tensors and missed every
mechanism tensor. One call now covers all 90.

## verify

### 1. `tools/rt.py tests/test_interp.py` — green

```
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 94 items

tests\test_interp.py ................................................... [ 54%]
...........................................                              [100%]

============================= 94 passed in 8.79s =============================
```

### 2. real `save_checkpoint` → real `load_model`, exact parameter match

Throwaway script, kept **outside** the worktree
(`%LOCALAPPDATA%\Temp\opencode\task07_verify.py`, asserted
`src.__file__ == C:\dev\wt-07\src\__init__.py` so the editable install of the
main repo could not shadow the worktree). Toy 2-layer models, parameters
jittered by `randn * 0.05` first so a match cannot be luck, real
`save_checkpoint` → `safe_torch_load` → `load_model`, comparing every key of
`state_dict()` in float64:

```
src under test: C:\dev\wt-07\src\__init__.py
jepa  ckpt keys: ['ema_step', 'epoch', 'global_step', 'mask_step', 'mechanism_extras', 'model', 'model_name', 'opt', 'rng_state'] | tensors in 'model': 90
JEPA  max-abs-difference: 0.0
d2v   ckpt keys: ['ema_step', 'epoch', 'global_step', 'mask_step', 'model', 'model_name', 'num_updates', 'opt', 'rng_state']
d2v   regression_head tensors inside 'model': ['regression_head.0.bias', 'regression_head.0.weight']
D2V   max-abs-difference: 0.0
mlm   ckpt keys: ['ema_step', 'epoch', 'global_step', 'mask_step', 'model', 'model_name', 'opt', 'rng_state']
MLM   max-abs-difference: 0.0
```

**max-abs-difference 0.0** on all three branches.

### 3. corrupted / wrong-format checkpoint — clear error, not `KeyError`

The old writer's exact shape (top-level `encoder`/`predictor`/
`target_encoder`/`decoder`, no `"model"`), and a checkpoint missing `"model"`
entirely:

```
LEGACY CheckpointLoadError: Checkpoint ...\legacy.pt is not in the full-state_dict format written by src.train.save_checkpoint: expected a top-level 'model' key holding model.state_dict(), found ['decoder', 'encoder', 'epoch', 'global_step', 'model_name', 'predictor', 'target_encoder']... . Refusing to load -- a partial restore would silently leave unlisted tensors at their initial values.

NOMODEL CheckpointLoadError: Checkpoint ...\nomodel.pt is not in the full-state_dict format written by src.train.save_checkpoint: expected a top-level 'model' key holding model.state_dict(), found ['epoch', 'model_name']... . Refusing to load -- a partial restore would silently leave unlisted tensors at their initial values.
```

Non-dict payloads are named by type rather than crashing on `sorted()`.

The hard prohibition was respected: **nothing re-emits the legacy keys.** The
new reader refuses them, which is the only direction that cannot silently
reproduce the divergence.

### 4. lint / format — clean

```
> ruff check . --output-format concise
All checks passed!

> black --check .
All done! ✨ 🍰 ✨
100 files would be left unchanged.
```

## diff-stat

```
$ git diff --stat campaign/integration...HEAD
 src/interp/run_comparison.py | 57 +++++++++++++++++++++++++++++++++-----------
 1 file changed, 43 insertions(+), 14 deletions(-)
```

Commit `626e9bf` on `agent/task-07`. Not pushed.

## mutation-verdict

**No test fails. Nothing detects the revert.** Measured, not predicted: I
restored the original file with `git checkout --`, ran the card's verify
command, and it passed unchanged:

```
$ git checkout -- src/interp/run_comparison.py   # back to per-module .get()
$ python tools/rt.py tests/test_interp.py
collected 94 items
tests\test_interp.py ................................................... [ 54%]
...........................................                              [100%]
============================= 94 passed in 9.09s =============================
```

Then `git grep` across `tests/`: **zero** references to `run_comparison` from
any test file. No `pkgutil` / `walk_packages` smoke-import test exists either,
so the module is never even imported by the suite. `tests/test_interp.py`
imports 20 sibling interp modules by name and skips this one.

This is a pre-existing coverage hole, not something my change opened — the
function was equally untested before. But it means this fix is currently
unguarded: the next refactor can silently reintroduce a broken reader. I own no
test file on this card (`tests/**` is forbidden), so I did not add one.
**Recommend a test-author card** pinning: (a) a
`save_checkpoint`→`load_model` round trip with max-abs-diff 0, (b) the
legacy-shape rejection, (c) that the data2vec branch restores
`regression_head`. Property-based, no magic numbers, per `AGENTS.md`.

## не_сделано

- **No test added** — `tests/**` forbidden on this card. See mutation-verdict.
- **No architecture recovery.** `load_model` hardcodes `TextSpanJEPAConfig()`
  defaults (vocab 50304 / 768-dim / 12 layers) and the baselines hardcode
  50304/768/12. The checkpoint format stores **no config field** — I read the
  writer's key list: `model_name, model, opt, epoch, global_step, ema_step,
  mask_step, rng_state (+scaler, mechanism_extras, num_updates, schedulers,
  extra)`. So `load_model` can only restore a model trained at exactly those
  defaults. `strict=True` converts that from silent mis-restore to a loud
  shape-mismatch `RuntimeError` (I hit it during verification and captured the
  message), which is the correct failure mode — but it means the function still
  cannot load a real `xsmall_30m` checkpoint. Fixing it needs a config field in
  the writer = `src/train.py`, forbidden here. **Suggest a follow-up card.**
- **No `model_name` cross-check.** A data2vec checkpoint passed as `"jepa"`
  fails loudly on `strict=True`, but with a `Missing key(s)` message rather than
  one that says "wrong model type". One `ckpt["model_name"] != model_type`
  guard would say it plainly; skipped as unrequested scope. Cheap follow-up.
- **`mechanism_extras` not applied.** `_restore_mechanism_extras` in
  `src/train.py` restores non-tensor mechanism scalars; `run_comparison` only
  reads `model.encoder(...)` outputs, and every mechanism tensor is already in
  the state_dict. Nothing to restore for this consumer.
- **`num_updates` not restored** for data2vec (a plain int attribute, not a
  tensor, so not in `state_dict`; the writer stores it as a top-level key). It
  only feeds `get_annealed_decay()` during training, irrelevant for eval.
- Did not touch `src/train.py`, `src/utils/torchio.py`, `src/interp/compare.py`,
  any `tests/**`, or `config/**`. Ran no training. Only test invocation was
  `tools/rt.py` with a single file.

## риски

- **Untested, per above.** Highest residual risk.
- **`strict=True` is stricter than before in the mlm branch**, which used a
  non-strict load. If any archived mlm checkpoint was saved with a key
  mismatch, this now raises where it used to proceed. That is the intent of the
  card, but it is a behaviour change on a path with no test.
- **`load_model` remains unable to load non-default architectures** (see above).
  Loud, not silent, but `scripts/run_experiment.sh:55,64` calls this entry point
  — anyone running it against a `config/scaling/*` checkpoint now gets an
  exception instead of a silent garbage comparison. Worth pairing with a
  follow-up card before the script is used.
- Error message deliberately ASCII (`--`, not `—`): Windows consoles here are
  cp866 and render the em-dash as mojibake in a traceback.
